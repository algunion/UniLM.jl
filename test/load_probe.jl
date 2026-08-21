#!/usr/bin/env julia
#
# Opt-in concurrency/load falsification harness. NOT part of `Pkg.test` — it is a
# standalone script whose job is to try to BREAK the client under load and under
# adversarial error regimes, not to corroborate the unit suite.
#
#     julia --project=. test/load_probe.jl              # single-threaded is valid
#     julia --project=. -t auto,2 test/load_probe.jl    # preferred: real parallelism
#
# Zero-spend and network-free: every request goes to a mock HTTP server bound to
# 127.0.0.1 on an ephemeral port. No provider credentials are read or needed.
#
# Batches
#   control  100 mixed streaming/non-streaming calls fanned across 2 mock hosts,
#            repeated twice, with per-request markers SPLIT across SSE chunks
#            (only correct per-stream assembly reproduces a marker intact, so a
#            mis-routed delta shows up as a foreign/corrupt marker), plus
#            active-request high-water marks, FD and live-task deltas.
#   b429     retry-budget storm: every logical request is refused twice with
#            429 + `Retry-After: 1` before succeeding. Checks the budget is
#            exactly spent (no runaway attempts) and that Retry-After is honored.
#   bnoise   teardown noise after a COMPLETE stream: the server writes every SSE
#            chunk, the terminal event and `[DONE]`, then kills the socket without
#            terminating the HTTP message. Checks that a fully delivered stream
#            still reports success, delivers exactly one terminal callback, and —
#            the billing-critical part — is never re-POSTed.
#   bslow    slow consumer: a healthy stream whose user callback blocks longer
#            than `stream_idle_timeout`. Checks the idle kill surfaces as a TYPED
#            `UniLMTimeout(:stream_idle)` carried in the result, never as a
#            truncated success or a status-200 failure holding partial bytes.
#
# Exit codes
#   0   every batch passed (the state a fixed client must reach)
#   1   a control batch failed — load/isolation regression
#   2   harness fault (unexpected exception outside a batch)
#   3   control batches passed, one or more adversarial batches failed
#   99  hard-cap watchdog fired: a batch did not finish (reported, never silent)
#
# Output: a machine-readable `SUMMARY_START`/`SUMMARY_END` block per batch plus a
# final `batch=overall` block. `PROBE_N`, `PROBE_BATCHES` and `PROBE_CAP` override
# the control batch size, control batch count, and the watchdog cap.

using UniLM, HTTP, JSON, Sockets, Printf, Dates

const N        = parse(Int, get(ENV, "PROBE_N", "100"))
const NB       = parse(Int, get(ENV, "PROBE_BATCHES", "2"))
const HARD_CAP = parse(Float64, get(ENV, "PROBE_CAP", "420.0"))
const T_START  = time()

# ── watchdog ─────────────────────────────────────────────────────────────────
const FINISHED   = Threads.Atomic{Bool}(false)
const PROG_LOCK  = ReentrantLock()
const PROG_LABEL = Ref("init")
const PROG_DONE  = Ref(Bool[])

set_progress!(label::AbstractString, n::Int) = lock(PROG_LOCK) do
    PROG_LABEL[] = label; PROG_DONE[] = fill(false, n)
end
mark_done!(i::Int) = lock(PROG_LOCK) do
    i <= length(PROG_DONE[]) && (PROG_DONE[][i] = true)
end

# Prefer the interactive pool so the watchdog stays schedulable while the default
# pool is saturated; fall back when the process has no interactive threads.
spawn_watchdog(f::Function) =
    Threads.nthreads(:interactive) > 0 ? Threads.@spawn(:interactive, f()) : Threads.@spawn(f())

spawn_watchdog() do
    while !FINISHED[]
        sleep(1.0)
        if !FINISHED[] && (time() - T_START) > HARD_CAP
            println("\nWATCHDOG: hard cap $(HARD_CAP)s exceeded — declaring HANG")
            lock(PROG_LOCK) do
                inc = findall(!, PROG_DONE[])
                println("WATCHDOG: phase=$(PROG_LABEL[]) incomplete=$(length(inc))/$(length(PROG_DONE[]))")
                println("WATCHDOG: incomplete_indices=", inc)
            end
            try
                ccall(:jl_print_task_backtraces, Cvoid, ())
            catch e
                println("WATCHDOG: task backtraces unavailable ($e)")
            end
            flush(stdout); flush(stderr)
            exit(99)
        end
    end
end

# ── shared helpers ───────────────────────────────────────────────────────────
const MARKER_RE   = r"mk-\d{4}"
const CHAT_PATH   = "/v1/chat/completions"
const AGENTIC_PAT = "/v1/responses"

marker_of(i::Int)::String = @sprintf("mk-%04d", i)
"Pull the request's marker out of the serialized request body."
extract_marker(body::AbstractString)::String =
    (m = match(MARKER_RE, body); isnothing(m) ? "mk-XXXX" : String(m.match))

fd_count()   = try length(readdir("/dev/fd")) catch; -1 end
task_count() = try length(ccall(:jl_live_tasks, Array{Any,1}, ())) catch; -1 end

"Bind an ephemeral port, then re-bind it with `HTTP.listen!` (TOCTOU retry)."
function listen_retry(handler::Function)
    for attempt in 1:5
        tcp = Sockets.listen(Sockets.localhost, 0)
        port = Int(Sockets.getsockname(tcp)[2])
        close(tcp)
        try
            return (HTTP.listen!(handler, "127.0.0.1", port; verbose=false), port)
        catch
            attempt == 5 && rethrow()
        end
    end
end

shutdown(server) = try close(server) catch end

"One-line, machine-greppable rendering of any result value."
function shape(r)::String
    r isa LLMSuccess     && return "LLMSuccess(chars=$(sizeof(something(r.message.content, ""))))"
    r isa LLMFailure     && return "LLMFailure(status=$(r.status),body_bytes=$(sizeof(r.response)))"
    r isa ResponseSuccess && return "ResponseSuccess(chars=$(sizeof(output_text(r))))"
    r isa ResponseFailure && return "ResponseFailure(status=$(r.status),body_bytes=$(sizeof(r.response)))"
    if r isa LLMCallError || r isa ResponseCallError
        c = r.cause
        tag = c isa UniLMTimeout ? "UniLMTimeout(:$(c.phase))" : string(typeof(c))
        return "$(nameof(typeof(r)))(status=$(r.status),cause=$(tag))"
    end
    return string(typeof(r))
end

is_stream_idle_timeout(r)::Bool =
    (r isa LLMCallError || r isa ResponseCallError) &&
    r.cause isa UniLMTimeout && r.cause.phase === :stream_idle

# ─────────────────────────────────────────────────────────────────────────────
# CONTROL BATCH — 100 concurrent mixed calls across 2 hosts
# ─────────────────────────────────────────────────────────────────────────────

# Records the HIGH-WATER MARK of simultaneously in-flight requests: the
# measurement that distinguishes true client-side parallelism from serialization
# behind a connection pool.
mutable struct Gauge
    lock::ReentrantLock
    active::Int
    hwm::Int
    total::Int
    n_stream::Int
    n_plain::Int
    bad_target::Int
    Gauge() = new(ReentrantLock(), 0, 0, 0, 0, 0, 0)
end

function gauge_enter!(g::Gauge, streaming::Bool, target::AbstractString)
    lock(g.lock) do
        g.active += 1
        g.active > g.hwm && (g.hwm = g.active)
        g.total += 1
        streaming ? (g.n_stream += 1) : (g.n_plain += 1)
        target == CHAT_PATH || (g.bad_target += 1)
    end
end
gauge_exit!(g::Gauge) = lock(g.lock) do; g.active -= 1 end
gauge_reset!(g::Gauge) = lock(g.lock) do
    g.hwm = 0; g.total = 0; g.n_stream = 0; g.n_plain = 0; g.bad_target = 0
end

plain_body(marker::AbstractString, host::Int) = JSON.json(Dict(
    "id" => "chatcmpl-$marker", "object" => "chat.completion",
    "created" => 0, "model" => "mock-model-h$host",
    "choices" => [Dict("index" => 0, "finish_reason" => "stop",
        "message" => Dict("role" => "assistant",
                          "content" => "PLAIN h$host echo $marker END"))],
    "usage" => Dict("prompt_tokens" => 7, "completion_tokens" => 5, "total_tokens" => 12)))

sse_delta(content, fin) = "data: " * JSON.json(Dict("choices" => [Dict(
    "index" => 0, "delta" => Dict("content" => content), "finish_reason" => fin)])) * "\n\n"

# The marker is SPLIT across three deltas: correct assembly of THIS stream is the
# only way it reappears intact, so an interleaved delta corrupts it detectably.
function sse_chunks(marker::AbstractString, host::Int)
    head, mid, tail = marker[1:3], marker[4:5], marker[6:end]
    [sse_delta("STREAM h$host echo " * head, nothing), sse_delta(mid, nothing),
     sse_delta(tail * " END", "stop"),
     "data: " * JSON.json(Dict("choices" => [], "usage" => Dict(
        "prompt_tokens" => 7, "completion_tokens" => 5, "total_tokens" => 12))) * "\n\n",
     "data: [DONE]\n\n"]
end

function start_host(host_id::Int)
    g = Gauge()
    server, port = listen_retry() do http::HTTP.Stream
        body = String(read(http))
        target = http.message.target
        streaming = occursin("\"stream\":true", body)
        marker = extract_marker(body)
        gauge_enter!(g, streaming, target)
        try
            if streaming
                HTTP.setstatus(http, 200)
                HTTP.setheader(http, "Content-Type" => "text/event-stream")
                HTTP.startwrite(http)
                for c in sse_chunks(marker, host_id)
                    write(http, c); flush(http); sleep(0.015)
                end
            else
                payload = plain_body(marker, host_id)
                HTTP.setstatus(http, 200)
                HTTP.setheader(http, "Content-Type" => "application/json")
                HTTP.setheader(http, "Content-Length" => string(sizeof(payload)))
                HTTP.startwrite(http)
                write(http, payload)
            end
        finally
            gauge_exit!(g)
        end
    end
    (; server, gauge=g, base="http://127.0.0.1:$port", id=host_id)
end

struct Outcome
    idx::Int
    kind::Symbol          # :stream | :plain
    host::Int
    ok::Bool
    crosstalk::Bool
    err::String
end

# No retries in the control batch: every wire attempt is one task, so the
# server-observed total must equal N and no failure can be masked by a re-try.
const CTRL_CFG = RequestConfig(connect_timeout=5.0, request_timeout=45.0,
                               stream_idle_timeout=20.0, total_deadline=90.0,
                               max_attempts=1)

"Marker/host validation shared by every batch that echoes a marker back."
function check_text(i::Int, kind::Symbol, host::Int, marker::AbstractString,
                    text::AbstractString, host_tag::Union{Nothing,String})::Outcome
    isempty(text) && return Outcome(i, kind, host, false, false, "empty content")
    found = Set(m.match for m in eachmatch(MARKER_RE, text))
    foreign = setdiff(found, Set([marker]))
    isempty(foreign) ||
        return Outcome(i, kind, host, false, true,
                       "CROSSTALK own=$marker foreign=$(collect(foreign)) text=$(first(text, 120))")
    marker in found ||
        return Outcome(i, kind, host, false, false, "own marker $marker absent; text=$(first(text, 120))")
    if host_tag !== nothing && !occursin(host_tag, text)
        return Outcome(i, kind, host, false, true, "HOST MIX: expected $host_tag; text=$(first(text, 120))")
    end
    Outcome(i, kind, host, true, false, "")
end

function control_call(i::Int, hosts)::Outcome
    host = hosts[(i % 2) + 1]
    streaming = ((i - 1) ÷ 2) % 2 == 0        # T,T,F,F,… → equal split per host
    marker = marker_of(i)
    kind = streaming ? :stream : :plain
    try
        chat = Chat(service=GenericOpenAIEndpoint(host.base, "probe-key"),
                    model="mock-model", stream=streaming)
        push!(chat, Message(Val(:system), "you are a marker echo service"))
        push!(chat, Message(Val(:user), "please echo $marker verbatim"))
        if streaming
            buf = IOBuffer(); bl = ReentrantLock()
            res = fetch(chatrequest!(chat; config=CTRL_CFG,
                callback = (chunk, _) -> (chunk isa AbstractString &&
                                          lock(bl) do; print(buf, chunk) end; nothing)))
            text = String(take!(buf))
            res isa LLMSuccess ||
                return Outcome(i, kind, host.id, false, false, "non-success: $(shape(res))")
            return check_text(i, kind, host.id, marker, text, "h$(host.id)")
        else
            res = chatrequest!(chat; config=CTRL_CFG)
            res isa LLMSuccess ||
                return Outcome(i, kind, host.id, false, false, "non-success: $(shape(res))")
            return check_text(i, kind, host.id, marker, something(res.message.content, ""), "h$(host.id)")
        end
    catch e
        return Outcome(i, kind, host.id, false, false, "EXC " * first(sprint(showerror, e), 400))
    end
end

function run_control_batch(label::String, hosts)
    for h in hosts; gauge_reset!(h.gauge); end
    set_progress!(label, N)
    outcomes = Vector{Outcome}(undef, N)
    t0 = time()
    @sync for i in 1:N
        Threads.@spawn begin
            outcomes[i] = control_call(i, hosts)
            mark_done!(i)
        end
    end
    (; label, wall=time() - t0, outcomes)
end

function report_control_batch(b, hosts, fds_before, tasks_before)
    GC.gc(); GC.gc()
    fds_after, tasks_after = fd_count(), task_count()
    ok = count(o -> o.ok, b.outcomes)
    ct = count(o -> o.crosstalk, b.outcomes)
    fails = filter(o -> !o.ok, b.outcomes)
    L = b.label
    println("$(L)_wall_s=", round(b.wall, digits=3))
    println("$(L)_successes=", ok, "/", N)
    println("$(L)_failures=", length(fails))
    println("$(L)_crosstalk=", ct)
    println("$(L)_stream_ok=", count(o -> o.ok && o.kind === :stream, b.outcomes))
    println("$(L)_plain_ok=",  count(o -> o.ok && o.kind === :plain,  b.outcomes))
    for h in hosts
        lock(h.gauge.lock) do
            println("$(L)_host$(h.id)_requests=", h.gauge.total,
                    " stream=", h.gauge.n_stream, " plain=", h.gauge.n_plain,
                    " bad_target=", h.gauge.bad_target)
            println("$(L)_host$(h.id)_high_water_mark=", h.gauge.hwm)
        end
    end
    println("$(L)_fds_before=", fds_before, " fds_after_gc=", fds_after,
            " fd_delta=", fds_after - fds_before)
    println("$(L)_tasks_before=", tasks_before, " tasks_after_gc=", tasks_after,
            " task_delta=", tasks_after - tasks_before)
    for (n, o) in enumerate(first(fails, 5))
        println("$(L)_failure_$(n)= idx=$(o.idx) kind=$(o.kind) host=$(o.host) :: $(first(o.err, 400))")
    end
    (; ok, ct, nfail=length(fails), fds_after, tasks_after, wall=b.wall)
end

function batch_control()
    hosts = (start_host(1), start_host(2))
    println("host1=", hosts[1].base, " host2=", hosts[2].base)
    # Warm compilation on one call per shape so batch-1 wall time measures
    # concurrency, not first-call latency (which would poison the b1/b2 ratio).
    warm = [control_call(i, hosts) for i in 1:4]
    println("warmup_ok=", count(o -> o.ok, warm), "/4",
            all(o -> o.ok, warm) ? "" : " warm_err=" * first(warm[findfirst(o -> !o.ok, warm)].err, 300))
    for h in hosts; gauge_reset!(h.gauge); end
    GC.gc(); GC.gc()

    # The identical batch repeats NB times. A genuine FD/task leak grows without
    # bound; a connection pool reaching steady state plateaus. Only the shape of
    # the series across batches tells the two apart.
    fds0, tasks0 = fd_count(), task_count()
    fd_series, task_series, wall_series = [fds0], [tasks0], Float64[]
    reports = []
    for k in 1:NB
        b = run_control_batch("b$k", hosts)
        r = report_control_batch(b, hosts, fd_series[end], task_series[end])
        push!(reports, r); push!(fd_series, r.fds_after)
        push!(task_series, r.tasks_after); push!(wall_series, r.wall)
    end
    r1, r2 = reports[1], reports[min(2, NB)]
    total_ok = sum(r.ok for r in reports)
    total_ct = sum(r.ct for r in reports)
    total_fail = sum(r.nfail for r in reports)
    pass = total_ok == NB * N && total_ct == 0

    println("SUMMARY_START")
    println("batch=control")
    println("julia_version=", VERSION)
    println("nthreads=", Threads.nthreads(), " interactive=", Threads.nthreads(:interactive))
    println("http_version=", pkgversion(HTTP))
    println("unilm_version=", pkgversion(UniLM))
    println("tasks_per_batch=", N)
    println("batches=", NB)
    println("successes=", total_ok, "/", NB * N)
    println("failures=", total_fail)
    println("cross_talk=", total_ct)
    println("hangs=0")
    println("b1_wall_s=", round(r1.wall, digits=3))
    println("b2_wall_s=", round(r2.wall, digits=3))
    println("b2_over_b1_wall_ratio=", round(r1.wall > 0 ? r2.wall / r1.wall : Inf, digits=3))
    for h in hosts
        lock(h.gauge.lock) do
            println("host$(h.id)_last_batch_requests=", h.gauge.total)
            println("host$(h.id)_last_batch_high_water_mark=", h.gauge.hwm)
        end
    end
    println("fd_series=", join(fd_series, ","))
    println("fd_batch_deltas=", join(diff(fd_series), ","))
    println("fd_delta_total=", fd_series[end] - fds0)
    println("task_series=", join(task_series, ","))
    println("task_batch_deltas=", join(diff(task_series), ","))
    println("wall_series=", join(round.(wall_series, digits=3), ","))
    println("prediction=PASS")
    println("verdict=", pass ? "PASS" : "FAIL")
    println("SUMMARY_END")

    for h in hosts; shutdown(h.server); end
    pass
end

# ─────────────────────────────────────────────────────────────────────────────
# b429 — retry-budget storm: two 429s (Retry-After: 1) per request, then 200
# ─────────────────────────────────────────────────────────────────────────────
const N429       = 30
const REFUSALS   = 2                    # 429 responses before the 200
const B429_CFG   = RequestConfig(connect_timeout=5.0, request_timeout=30.0,
                                 stream_idle_timeout=30.0, total_deadline=120.0,
                                 max_attempts=REFUSALS + 1)

function start_429_host()
    lk = ReentrantLock()
    seen = Dict{String,Int}()
    total = Threads.Atomic{Int}(0)
    server, port = listen_retry() do http::HTTP.Stream
        body = String(read(http))
        marker = extract_marker(body)
        Threads.atomic_add!(total, 1)
        n = lock(lk) do; seen[marker] = get(seen, marker, 0) + 1 end
        if n <= REFUSALS
            payload = JSON.json(Dict("error" => Dict("message" => "rate limited $marker")))
            HTTP.setstatus(http, 429)
            HTTP.setheader(http, "Retry-After" => "1")
            HTTP.setheader(http, "Content-Type" => "application/json")
            HTTP.setheader(http, "Content-Length" => string(sizeof(payload)))
            HTTP.startwrite(http)
            write(http, payload)
        else
            payload = plain_body(marker, 9)
            HTTP.setstatus(http, 200)
            HTTP.setheader(http, "Content-Type" => "application/json")
            HTTP.setheader(http, "Content-Length" => string(sizeof(payload)))
            HTTP.startwrite(http)
            write(http, payload)
        end
    end
    (; server, base="http://127.0.0.1:$port", total, seen, lk)
end

function batch_429()
    set_progress!("b429", N429)
    host = start_429_host()
    ep = GenericOpenAIEndpoint(host.base, "probe-key")
    outcomes = Vector{Outcome}(undef, N429)
    t0 = time()
    @sync for i in 1:N429
        Threads.@spawn begin
            marker = marker_of(i)
            outcomes[i] = try
                chat = Chat(service=ep, model="mock-model")
                push!(chat, Message(Val(:system), "you are a marker echo service"))
                push!(chat, Message(Val(:user), "please echo $marker verbatim"))
                res = chatrequest!(chat; config=B429_CFG)
                res isa LLMSuccess ?
                    check_text(i, :plain, 9, marker, something(res.message.content, ""), nothing) :
                    Outcome(i, :plain, 9, false, false, "non-success: $(shape(res))")
            catch e
                Outcome(i, :plain, 9, false, false, "EXC " * first(sprint(showerror, e), 300))
            end
            mark_done!(i)
        end
    end
    wall = time() - t0
    ok = count(o -> o.ok, outcomes)
    ct = count(o -> o.crosstalk, outcomes)
    attempts = host.total[]
    expected = N429 * (REFUSALS + 1)
    per_request = lock(host.lk) do; sort(collect(values(host.seen))) end
    # Retry-After: 1 honored twice per request → no run can finish under ~2 s.
    # Loose LOWER bound only: upper bounds are not assertable on a shared runner.
    honored = wall >= 2.0
    pass = ok == N429 && ct == 0 && attempts == expected && honored

    println("SUMMARY_START")
    println("batch=b429")
    println("calls=", N429)
    println("max_attempts=", B429_CFG.max_attempts)
    println("refusals_per_request=", REFUSALS)
    println("successes=", ok, "/", N429)
    println("cross_talk=", ct)
    println("server_attempts_total=", attempts)
    println("server_attempts_expected=", expected)
    println("server_attempts_bounded=", attempts == expected)
    println("distinct_requests_seen=", length(per_request))
    println("attempts_per_request_min=", isempty(per_request) ? -1 : first(per_request))
    println("attempts_per_request_max=", isempty(per_request) ? -1 : last(per_request))
    println("wall_s=", round(wall, digits=3))
    println("retry_after_honored_lower_bound_2s=", honored)
    for (n, o) in enumerate(first(filter(o -> !o.ok, outcomes), 5))
        println("failure_$(n)= idx=$(o.idx) :: $(first(o.err, 300))")
    end
    println("prediction=PASS")
    println("verdict=", pass ? "PASS" : "FAIL")
    println("SUMMARY_END")

    shutdown(host.server)
    pass
end

# ─────────────────────────────────────────────────────────────────────────────
# bnoise — complete stream, then the socket dies without terminating the message
# ─────────────────────────────────────────────────────────────────────────────
# A raw TCP peer, not HTTP.listen!: the teardown under test is a protocol-level
# violation (no terminating chunk / short body), which a conforming server API
# will not emit.
const N_NOISE   = 20                      # 10 chat + 10 agentic
const NOISE_CFG = RequestConfig(connect_timeout=5.0, request_timeout=30.0,
                                stream_idle_timeout=15.0, total_deadline=90.0,
                                max_attempts=3)

const SOL_SOCKET = Cint(0xffff)           # macOS/BSD values; harmless elsewhere
const SO_LINGER  = Cint(0x0080)

# Ask the kernel for a RST instead of a FIN (struct linger{on=1, linger=0}).
# Returns the setsockopt status so the achieved teardown shape is reported, not
# assumed; a nonzero/failed result still leaves the abrupt close below.
function hard_reset!(conn)::Int
    try
        buf = Cint[1, 0]
        return Int(ccall(:setsockopt, Cint, (Cint, Cint, Cint, Ptr{Cvoid}, Cuint),
                         Base.cconvert(Cint, Base._fd(conn)), SOL_SOCKET, SO_LINGER,
                         buf, UInt32(sizeof(buf))))
    catch
        return -1
    end
end

noise_chat_sse(marker::AbstractString) = [
    sse_delta("NOISE echo " * marker[1:3], nothing), sse_delta(marker[4:5], nothing),
    sse_delta(marker[6:end] * " END", "stop"), "data: [DONE]\n\n"]

function noise_agentic_sse(marker::AbstractString)
    ev(name, d) = "event: $name\ndata: " * JSON.json(d) * "\n\n"
    full = "NOISE echo $marker END"
    [ev("response.output_text.delta", Dict("delta" => "NOISE echo " * marker[1:3])),
     ev("response.output_text.delta", Dict("delta" => marker[4:5])),
     ev("response.output_text.delta", Dict("delta" => marker[6:end] * " END")),
     ev("response.completed", Dict("response" => Dict(
        "id" => "resp_$marker", "status" => "completed", "model" => "mock-model",
        "output" => [Dict("type" => "message", "role" => "assistant",
                          "content" => [Dict("type" => "output_text", "text" => full)])],
        "usage" => Dict("input_tokens" => 7, "output_tokens" => 5, "total_tokens" => 12)))),
     "data: [DONE]\n\n"]
end

"Minimal HTTP/1.1 request read: header block, then the declared body."
function read_request(conn)
    clen, target = 0, ""
    while true
        line = readline(conn; keep=true)
        s = strip(line)
        startswith(s, "POST ") && (target = String(split(s)[2]))
        startswith(lowercase(s), "content-length:") && (clen = parse(Int, strip(split(s, ':')[2])))
        (isempty(line) || isempty(s)) && break
    end
    (target, clen > 0 ? String(read(conn, clen)) : "")
end

function serve_noise(conn, framing::Symbol, chat_posts, agentic_posts, rst_rc)
    try
        target, body = read_request(conn)
        agentic = occursin(AGENTIC_PAT, target)
        Threads.atomic_add!(agentic ? agentic_posts : chat_posts, 1)
        marker = extract_marker(body)
        chunks = agentic ? noise_agentic_sse(marker) : noise_chat_sse(marker)
        write(conn, "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n")
        if framing === :unterminated
            # Chunked framing whose terminating 0-length chunk is never written,
            # written slowly and held open so the peer consumes every chunk
            # before the kill, then reset at the socket level.
            write(conn, "Connection: close\r\nTransfer-Encoding: chunked\r\n\r\n"); flush(conn)
            for c in chunks
                write(conn, string(sizeof(c), base=16), "\r\n", c, "\r\n"); flush(conn)
                sleep(0.02)
            end
            sleep(0.15)
            Threads.atomic_max!(rst_rc, hard_reset!(conn))
        else
            # Declared length exceeds what is written, on a keep-alive connection
            # closed with a plain FIN the instant the payload is out: the tighter
            # race, where the peer's parser can fail before any delta is
            # delivered — the state in which a driver's retry gate is still open.
            payload = join(chunks)
            write(conn, "Content-Length: ", string(sizeof(payload) + 64), "\r\n\r\n")
            write(conn, payload); flush(conn)
        end
        close(conn)
    catch
    end
end

function start_noise_host(framing::Symbol)
    srv = Sockets.listen(Sockets.localhost, 0)
    port = Int(Sockets.getsockname(srv)[2])
    chat_posts, agentic_posts = Threads.Atomic{Int}(0), Threads.Atomic{Int}(0)
    rst_rc = Threads.Atomic{Int}(-1)
    Threads.@spawn begin
        while true
            conn = try Sockets.accept(srv) catch; break end
            Threads.@spawn serve_noise(conn, framing, chat_posts, agentic_posts, rst_rc)
        end
    end
    (; srv, base="http://127.0.0.1:$port", chat_posts, agentic_posts, rst_rc)
end

struct NoiseOutcome
    idx::Int
    path::Symbol          # :chat | :agentic
    success::Bool
    terminals::Int
    text_ok::Bool
    crosstalk::Bool
    shape::String
end

function noise_call(i::Int, base::AbstractString, agentic::Bool)::NoiseOutcome
    marker = marker_of(i)
    ep = GenericOpenAIEndpoint(base, "probe-key")
    terminals = Threads.Atomic{Int}(0)
    buf, bl = IOBuffer(), ReentrantLock()
    cb = (chunk, _) -> begin
        if chunk isa AbstractString
            lock(bl) do; print(buf, chunk) end
        else
            Threads.atomic_add!(terminals, 1)
        end
        nothing
    end
    res = try
        if agentic
            fetch(respond(Respond(service=ep, model="mock-model", stream=true,
                                  input="please echo $marker verbatim");
                          config=NOISE_CFG, callback=cb))
        else
            chat = Chat(service=ep, model="mock-model", stream=true)
            push!(chat, Message(Val(:system), "you are a marker echo service"))
            push!(chat, Message(Val(:user), "please echo $marker verbatim"))
            fetch(chatrequest!(chat; config=NOISE_CFG, callback=cb))
        end
    catch e
        return NoiseOutcome(i, agentic ? :agentic : :chat, false, terminals[], false, false,
                            "EXC " * first(sprint(showerror, e), 200))
    end
    # Assembled callback text is the isolation evidence: it exists whether or not
    # the driver turned the completed stream into a failure result.
    text = String(take!(buf))
    chk = check_text(i, :stream, 0, marker, text, nothing)
    NoiseOutcome(i, agentic ? :agentic : :chat,
                 res isa LLMSuccess || res isa ResponseSuccess,
                 terminals[], chk.ok, chk.crosstalk, shape(res))
end

"Run one teardown framing and return the raw measurements."
function noise_round(framing::Symbol, n::Int)
    host = start_noise_host(framing)
    outcomes = Vector{NoiseOutcome}(undef, n)
    @sync for i in 1:n
        Threads.@spawn begin
            outcomes[i] = noise_call(i, host.base, iseven(i))
            mark_done!(i)
        end
    end
    res = (; outcomes, chat_posts=host.chat_posts[], agentic_posts=host.agentic_posts[],
           rst_rc=host.rst_rc[])
    shutdown(host.srv)
    res
end

function batch_noise()
    set_progress!("bnoise", N_NOISE)
    r = noise_round(:unterminated, N_NOISE)
    chat = filter(o -> o.path === :chat, r.outcomes)
    agn  = filter(o -> o.path === :agentic, r.outcomes)
    # Supplementary, NON-verdict shape: the same complete payload under a short
    # declared body closed with a plain FIN. It answers the adjacent question the
    # verdict shape cannot — whether the driver re-POSTs a request whose teardown
    # error arrives before any callback has fired.
    set_progress!("bnoise_short_body", 4)
    r2 = noise_round(:short_body, 4)

    successes = count(o -> o.success, r.outcomes)
    terminals_ok = count(o -> o.terminals == 1, r.outcomes)
    text_ok = count(o -> o.text_ok, r.outcomes)
    ct = count(o -> o.crosstalk, r.outcomes)
    posts_ok = r.chat_posts == length(chat) && r.agentic_posts == length(agn)
    pass = successes == N_NOISE && terminals_ok == N_NOISE && ct == 0 && posts_ok

    println("SUMMARY_START")
    println("batch=bnoise")
    println("calls=", N_NOISE, " chat=", length(chat), " agentic=", length(agn))
    println("max_attempts=", NOISE_CFG.max_attempts)
    println("teardown_shape=chunked_body_without_terminating_chunk+SO_LINGER0_close")
    println("teardown_setsockopt_rc=", r.rst_rc)
    println("teardown_setsockopt_rc_legend=0:rst_requested;-1:plain_abrupt_close")
    println("server_stream_complete=chunks+terminal_event+DONE_all_written")
    println("successes=", successes, "/", N_NOISE)
    println("terminal_deliveries_exactly_one=", terminals_ok, "/", N_NOISE)
    println("assembled_text_complete=", text_ok, "/", N_NOISE)
    println("cross_talk=", ct)
    println("chat_posts=", r.chat_posts, " chat_calls=", length(chat),
            " duplicate_chat_posts=", r.chat_posts - length(chat))
    println("agentic_posts=", r.agentic_posts, " agentic_calls=", length(agn),
            " duplicate_agentic_posts=", r.agentic_posts - length(agn))
    println("no_duplicate_post=", posts_ok)
    println("chat_result_shapes=", join(sort(unique(o.shape for o in chat)), " | "))
    println("chat_terminals=", join((o.terminals for o in chat), ","))
    println("agentic_result_shapes=", join(sort(unique(o.shape for o in agn)), " | "))
    println("agentic_terminals=", join((o.terminals for o in agn), ","))
    println("short_body_shape=short_declared_content_length+plain_fin+keepalive")
    println("short_body_calls=", length(r2.outcomes))
    println("short_body_posts=", r2.chat_posts + r2.agentic_posts,
            " expected_if_no_reexecution=", length(r2.outcomes))
    println("short_body_successes=", count(o -> o.success, r2.outcomes))
    # Whether the deltas reached the user before the kill is what decides if the
    # driver's retry gate was still open, so it is reported, not assumed.
    println("short_body_text_complete=", count(o -> o.text_ok, r2.outcomes), "/", length(r2.outcomes))
    println("short_body_terminals=", join((o.terminals for o in r2.outcomes), ","))
    println("short_body_result_shapes=", join(sort(unique(o.shape for o in r2.outcomes)), " | "))
    println("prediction=FAIL")
    println("verdict=", pass ? "PASS" : "FAIL")
    println("SUMMARY_END")
    pass
end

# ─────────────────────────────────────────────────────────────────────────────
# bslow — a healthy stream whose consumer blocks past stream_idle_timeout
# ─────────────────────────────────────────────────────────────────────────────
const N_SLOW      = 6
const SLOW_IDLE   = 0.8
const SLOW_BLOCK  = 2.0
const SLOW_CHUNKS = 40
const SLOW_GAP    = 0.05
const SLOW_CFG    = RequestConfig(connect_timeout=5.0, request_timeout=30.0,
                                  stream_idle_timeout=SLOW_IDLE, total_deadline=120.0,
                                  max_attempts=1)

function start_slow_host()
    server, port = listen_retry() do http::HTTP.Stream
        body = String(read(http))
        marker = extract_marker(body)
        HTTP.setstatus(http, 200)
        HTTP.setheader(http, "Content-Type" => "text/event-stream")
        HTTP.startwrite(http)
        try
            for k in 1:SLOW_CHUNKS
                write(http, sse_delta("$marker#$k ", k == SLOW_CHUNKS ? "stop" : nothing))
                flush(http); sleep(SLOW_GAP)
            end
            write(http, "data: [DONE]\n\n")
        catch
            # The consumer's idle bound closes the socket mid-stream; the
            # resulting write failure is the expected server-side echo.
        end
    end
    (; server, base="http://127.0.0.1:$port")
end

function batch_slow()
    set_progress!("bslow", N_SLOW)
    host = start_slow_host()
    ep = GenericOpenAIEndpoint(host.base, "probe-key")
    shapes = String[]
    typed = 0
    # Serial: this batch is timing-sensitive and must not compete with itself.
    for i in 1:N_SLOW
        marker = marker_of(i)
        chat = Chat(service=ep, model="mock-model", stream=true)
        push!(chat, Message(Val(:system), "you are a marker echo service"))
        push!(chat, Message(Val(:user), "please echo $marker verbatim"))
        n = Threads.Atomic{Int}(0)
        res = try
            fetch(chatrequest!(chat; config=SLOW_CFG,
                callback = (chunk, _) -> (Threads.atomic_add!(n, 1) == 0 && sleep(SLOW_BLOCK); nothing)))
        catch e
            "EXC " * first(sprint(showerror, e), 200)
        end
        s = res isa AbstractString ? res : shape(res)
        push!(shapes, s)
        is_stream_idle_timeout(res) && (typed += 1)
        println("bslow_call_$(i)= callbacks=$(n[]) result=$(s)")
        mark_done!(i)
    end
    pass = typed == N_SLOW

    println("SUMMARY_START")
    println("batch=bslow")
    println("calls=", N_SLOW)
    println("stream_idle_timeout_s=", SLOW_IDLE)
    println("callback_block_s=", SLOW_BLOCK)
    println("server_chunk_gap_s=", SLOW_GAP, " chunks=", SLOW_CHUNKS)
    println("typed_stream_idle_timeouts=", typed, "/", N_SLOW)
    println("truncated_successes=", count(s -> startswith(s, "LLMSuccess"), shapes))
    println("status_200_failures=", count(s -> startswith(s, "LLMFailure(status=200"), shapes))
    println("result_shapes=", join(sort(unique(shapes)), " | "))
    println("prediction=FAIL")
    println("verdict=", pass ? "PASS" : "FAIL")
    println("SUMMARY_END")

    shutdown(host.server)
    pass
end

# ─────────────────────────────────────────────────────────────────────────────
function main()::Int
    println("probe start ", Dates.now())
    println("julia=", VERSION, " nthreads=", Threads.nthreads(),
            " interactive=", Threads.nthreads(:interactive),
            " HTTP=", pkgversion(HTTP), " UniLM=", pkgversion(UniLM))
    println("N=", N, " control_batches=", NB, " hard_cap_s=", HARD_CAP)

    control = batch_control()
    b429    = batch_429()
    bnoise  = batch_noise()
    bslow   = batch_slow()

    adversarial = ("b429" => b429, "bnoise" => bnoise, "bslow" => bslow)
    failed = [k for (k, v) in adversarial if !v]
    rc = !control ? 1 : isempty(failed) ? 0 : 3

    println("SUMMARY_START")
    println("batch=overall")
    println("control_verdict=", control ? "PASS" : "FAIL")
    for (k, v) in adversarial
        println("$(k)_verdict=", v ? "PASS" : "FAIL")
    end
    println("adversarial_failed=", isempty(failed) ? "none" : join(failed, ","))
    # b429 is the one adversarial batch predicted to pass; if it fails, that is a
    # new result, not a reproduction of a known defect — so it is called out.
    b429 || println("DISCOVERY=b429 was predicted to PASS and did not")
    println("total_wall_s=", round(time() - T_START, digits=3))
    println("exit_code=", rc)
    println("exit_semantics=0:all_pass 1:control_regression 2:harness_fault 3:adversarial_failed 99:watchdog")
    println("SUMMARY_END")
    rc
end

rc = try
    main()
catch e
    println("FATAL ", sprint(showerror, e, catch_backtrace()))
    2
end
FINISHED[] = true
flush(stdout); flush(stderr)
exit(rc)
