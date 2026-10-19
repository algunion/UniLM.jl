# test/ask_each.jl — ask_each, and the transport every System One exchange shares; zero-spend.
#
# Results in order, each what `ask` returns for its item alone; local refusals before any
# I/O; the bound on requests in flight; the backoff a batch shares on 429 and 529; the
# retry number TypeSafe's retries carry; cancellation; the exceptions that stop a batch,
# leaving no task or descriptor behind; a backend's own defaults; and the connection pool
# against a server that closes idle connections. Local mocks only; nothing here reaches a
# network.

using Test, HTTP, JSON, UniLM, Sockets

# An endpoint on TypeSafe's wire through the trait's fallback, as a gateway would be; one
# whose protocol speaks the same wire under another name (TypeSafe's transport is
# TypeSafe's alone); one whose protocol brings a tight default config; one whose protocol
# notes every request it encodes; and one that declares no System One at all.
struct _AEMock <: UniLM.ServiceEndpoint end
struct _AEOther <: UniLM.ServiceEndpoint end
struct _AETight <: UniLM.ServiceEndpoint end
struct _AECounted <: UniLM.ServiceEndpoint end
struct _AENone <: UniLM.ServiceEndpoint end
struct _AEOtherProtocol <: UniLM.SystemOneProtocol end
struct _AETightProtocol <: UniLM.SystemOneProtocol end
struct _AECountedProtocol <: UniLM.SystemOneProtocol end
const _AE_BASE = Ref("")
for T in (_AEMock, _AEOther, _AETight, _AECounted)
    @eval UniLM._resolve_base_url(::Type{$T}) = _AE_BASE[]
    @eval UniLM.auth_header(::Type{$T}) = ["Authorization" => "Bearer ae", "Content-Type" => "application/json"]
    @eval UniLM.provider_capabilities(::Type{$T}) = Set([:system_one])
end
UniLM.provider_capabilities(::Type{_AENone}) = Set([:chat])
UniLM.system_one_protocol(::Type{_AEOther}) = _AEOtherProtocol()
UniLM.system_one_protocol(::Type{_AETight}) = _AETightProtocol()
UniLM.system_one_protocol(::Type{_AECounted}) = _AECountedProtocol()
for P in (_AEOtherProtocol, _AETightProtocol, _AECountedProtocol)
    @eval UniLM.decode_system_one(::$P, s, r::SystemOneRequest, resp::HTTP.Response) =
        UniLM.decode_system_one(UniLM.TypeSafeProtocol(), s, r, resp)
end
for P in (_AEOtherProtocol, _AETightProtocol)
    @eval UniLM.encode_system_one(::$P, s, r::SystemOneRequest) = UniLM.encode_system_one(UniLM.TypeSafeProtocol(), s, r)
end
const _AE_TIGHT = RequestConfig(request_timeout=0.3, max_attempts=1, total_deadline=5.0)
UniLM._system_one_default_config(::_AETightProtocol, _) = _AE_TIGHT
# What happened, in order: each request encoded (`"encode <state>"`) and each one the
# server received (`"send <state>"`, noted by the scripts that want it).
const _AE_EVENTS = String[]
function UniLM.encode_system_one(::_AECountedProtocol, s, r::SystemOneRequest)
    @lock _AE_LOCK push!(_AE_EVENTS, "encode " * r.state)
    UniLM.encode_system_one(UniLM.TypeSafeProtocol(), s, r)
end

# ─── One mock for the whole file ─────────────────────────────────────────────
# A pooled connection stays reusable from one testset to the next, and the task and
# descriptor gauges compare like with like. Each testset installs its own script,
# `script(n, body)`, answering the n-th request to arrive.

const _AESeen = @NamedTuple{n::Int, at::Float64, state::Any, retry::Union{Nothing,String}}
const _AE_LOCK = ReentrantLock()
const _AE_SEEN = _AESeen[]
const _AE_SCRIPT = Ref{Function}((n, body) -> _ae_ok(body))
const _AE_INFLIGHT = Threads.Atomic{Int}(0)
const _AE_PEAK = Threads.Atomic{Int}(0)

_ae_retry(req)::Union{Nothing,String} = (v = HTTP.header(req, "X-TypeSafe-Retry-Count", ""); isempty(v) ? nothing : String(v))

function _ae_handler(req)
    body = JSON.parse(String(copy(req.body)))
    n = @lock _AE_LOCK begin
        push!(_AE_SEEN, (n=length(_AE_SEEN) + 1, at=time(), state=body["state"], retry=_ae_retry(req)))
        length(_AE_SEEN)
    end
    Threads.atomic_max!(_AE_PEAK, Threads.atomic_add!(_AE_INFLIGHT, 1) + 1)
    try
        # The server's tasks run in the world it started in; a script is defined after.
        Base.invokelatest(_AE_SCRIPT[], n, body)
    finally
        Threads.atomic_sub!(_AE_INFLIGHT, 1)
    end
end

_ae_json(status::Int, body::AbstractString; headers::Vector{Pair{String,String}}=Pair{String,String}[]) =
    HTTP.Response(status, ["Content-Type" => "application/json"; headers], Vector{UInt8}(body))

# A deterministic answer to every question about the body's state — the answer the same
# request gets on its own.
function _ae_answers(body::AbstractDict)::String
    s = Int(sum(codeunits(JSON.json(body["state"]))))
    out = Dict{String,Any}()
    for (name, q) in body["questions"]
        t = q["type"]
        out[name] = if t == "choice"
            opts = collect(keys(q["criteria"]))
            pick = opts[s % length(opts) + 1]
            Dict{String,Any}("type" => t, "choice" => pick, "confidence" => 1.0,
                             "probabilities" => Dict{String,Any}(o => (o == pick ? 1.0 : 0.0) for o in opts))
        elseif t == "score"
            k = length(q["criteria"])
            level = s % k
            Dict{String,Any}("type" => t, "score" => Float64(level), "confidence" => 1.0,
                             "legend" => Dict{String,Any}(string(i - 1) => q["criteria"][i] for i in 1:k),
                             "probabilities" => Dict{String,Any}(string(i - 1) => (i - 1 == level ? 1.0 : 0.0) for i in 1:k))
        else
            Dict{String,Any}("type" => "noul", "noul" => (s % 101) / 100)
        end
    end
    JSON.json(Dict{String,Any}("model" => body["model"], "answers" => out,
                               "usage" => Dict{String,Any}("input_tokens" => 10, "output_tokens" => 0)))
end

_ae_ok(body::AbstractDict) = _ae_json(200, _ae_answers(body); headers=["x-typesafe-request-id" => "req_ae"])

const _AE_SERVER = HTTP.serve!(_ae_handler, "127.0.0.1", 0; verbose=false)
_AE_BASE[] = "http://127.0.0.1:$(HTTP.port(_AE_SERVER))"

_ae_drained() = timedwait(() -> _AE_INFLIGHT[] == 0, 15.0)

"Run `f()` with `script` answering; return `(f(), the requests that arrived)`."
function _ae_run(f::Function, script::Function)
    _ae_drained()          # requests a previous testset abandoned may still be answering
    @lock _AE_LOCK empty!(_AE_SEEN)
    _AE_PEAK[] = 0
    _AE_SCRIPT[] = script
    out = f()
    out, @lock(_AE_LOCK, copy(_AE_SEEN))
end

_ae_caught(f::Function) = try f(); nothing catch e; e end
_ae_ok_script(n, body) = _ae_ok(body)
_ae_slow(seconds) = (n, body) -> (sleep(seconds); _ae_ok(body))

const _AE_Q = ("team" => choice(["billing", "technical", "shipping"]),
               "urgency" => score(["can wait", "this week", "today"]),
               "angry" => noul("Is the customer angry?"))

# The process default exactly as found — `nothing` while the user has set none — so a
# testset that relies on the backend defaults sees them, and leaves the process as it was.
function _ae_backend_defaults(f::Function)
    saved = @atomic UniLM._PROCESS_DEFAULT_CONFIG.cfg
    @atomic UniLM._PROCESS_DEFAULT_CONFIG.cfg = nothing
    try
        f()
    finally
        @atomic UniLM._PROCESS_DEFAULT_CONFIG.cfg = saved
    end
end

_ae_fd_count() = length(readdir("/dev/fd"))
_ae_task_count() = length(ccall(:jl_live_tasks, Array{Any,1}, ()))

# A raw HTTP/1.1 server that answers System One requests with keep-alive and closes a
# connection left idle for `idle` seconds, as servers behind proxies do: what a pooled
# connection meets when the next request comes later than the server would wait.
mutable struct _AEIdleServer
    const listener::Sockets.TCPServer
    const port::Int
    const accepted::Threads.Atomic{Int}
    const retries::Vector{Union{Nothing,String}}
    const lock::ReentrantLock
end

function _ae_idle_conn(srv::_AEIdleServer, sock::Sockets.TCPSocket, idle::Float64)
    try
        while true
            timer = Timer(_ -> close(sock), idle)
            line = try readline(sock) catch; "" end
            close(timer)
            isempty(line) && return
            len, retry = 0, nothing
            while true
                h = readline(sock)
                isempty(h) && break
                k, v = strip.(split(h, ':'; limit=2))
                lowercase(k) == "content-length" && (len = parse(Int, v))
                lowercase(k) == "x-typesafe-retry-count" && (retry = String(v))
            end
            body = JSON.parse(String(read(sock, len)))
            @lock srv.lock push!(srv.retries, retry)
            reply = _ae_answers(body)
            write(sock, "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: $(sizeof(reply))\r\n\r\n", reply)
        end
    catch
    finally
        close(sock)
    end
end

function _ae_idle_server(idle::Float64)::_AEIdleServer
    listener = Sockets.listen(Sockets.localhost, 0)
    srv = _AEIdleServer(listener, Int(Sockets.getsockname(listener)[2]), Threads.Atomic{Int}(0),
                        Union{Nothing,String}[], ReentrantLock())
    Threads.@spawn try
        while true
            sock = Sockets.accept(listener)
            Threads.atomic_add!(srv.accepted, 1)
            Threads.@spawn _ae_idle_conn(srv, sock, idle)
        end
    catch                    # the listener closed
    end
    srv
end

@testset "ask_each — one result per item, in order, each what ask returns for it alone" begin
    states = ["ticket $i: " * repeat("x", i) for i in 1:24]
    landed = Tuple{Int,Any}[]
    lk = ReentrantLock()
    each, seen = _ae_run(_ae_slow(0.02)) do
        ask_each(states, _AE_Q...; service=_AEMock, on_result=(i, r) -> @lock(lk, push!(landed, (i, r))))
    end
    @test each isa Vector{Union{SystemOneSuccess,SystemOneFailure,SystemOneCallError}}
    @test length(each) == 24 && all(r -> r isa SystemOneSuccess, each)
    @test sort!([s.state for s in seen]) == sort(states)               # one request per item, none repeated
    alone, _ = _ae_run(() -> [ask(s, _AE_Q...; service=_AEMock) for s in states], _ae_ok_script)
    raw(r) = Dict(k => v.raw for (k, v) in answers(r))
    @test all(i -> raw(each[i]) == raw(alone[i]), 1:24)
    @test length(unique(raw.(each))) > 1                               # the items' answers differ: order matters
    # on_result saw every result exactly once, under its own index.
    @test sort!(first.(landed)) == 1:24
    @test all(((i, r),) -> r === each[i], landed)

    # The same through requests built by hand, a model of their own included.
    reqs = [SystemOneRequest(s, collect(Pair{String,SystemOneQuestion}, _AE_Q); model="jev-1.13.0") for s in states[1:5]]
    byreq, _ = _ae_run(() -> ask_each(reqs; service=_AEMock), _ae_ok_script)
    @test all(i -> raw(byreq[i]) == raw(alone[i]), 1:5)
    @test all(r -> r.response.model == "jev-1.13.0", byreq)

    # No items: nothing sent, an empty vector of the same type.
    none, seen0 = _ae_run(() -> ask_each(String[], _AE_Q...; service=_AEMock), _ae_ok_script)
    @test none isa Vector{Union{SystemOneSuccess,SystemOneFailure,SystemOneCallError}} && isempty(none)
    @test isempty(seen0)
end

@testset "ask_each — the backend and model resolve as for ask, with_system_one included" begin
    rs, _ = _ae_run(_ae_ok_script) do
        with_system_one(_AEMock; model="jev-scoped") do
            ask_each(["a", "b"], "q" => noul("x"))
        end
    end
    @test all(r -> r isa SystemOneSuccess && r.response.model == "jev-scoped" && r.response.service === _AEMock, rs)
    named, _ = _ae_run(() -> ask_each(["a"], "q" => noul("x"); service=_AEMock, model="jev-named"), _ae_ok_script)
    @test only(named).response.model == "jev-named"
end

@testset "ask_each — an item that cannot be sent is refused by position, and nothing is sent" begin
    q = "pick" => choice(["a", "b"])
    big = "pick" => choice(["o$i" for i in 1:256])
    errs, seen = _ae_run(_ae_ok_script) do
        (_ae_caught(() -> ask_each(["fine", nothing, "fine"], q; service=_AEMock)),
         _ae_caught(() -> ask_each([SystemOneRequest("a", [q]), SystemOneRequest("b", [big]), SystemOneRequest("c", [q])];
                                   service=_AEMock)),
         _ae_caught(() -> ask_each(["a", "b"], q; service=_AEMock, temperature=0.2)),
         _ae_caught(() -> ask_each(["a", Dict("x" => NaN)], q; service=_AEMock)),
         _ae_caught(() -> ask_each(["a"], q; service=_AEMock, ntasks=0)),
         _ae_caught(() -> ask_each(["a"], q; service=_AENone)),
         _ae_caught(() -> ask_each(String[], "a" => noul("x"), "a" => noul("y"); service=_AEMock)),
         _ae_caught(() -> ask_each(["a"], q; service=_AEMock, model=" ")))
    end
    @test isempty(seen)                                     # not one request reached the server
    @test all(e -> e isa ArgumentError, errs)
    state, limit, option, nan, ntasks, capability, duplicate, model = errs
    @test occursin("item 2 of 3", state.msg) && occursin("state", state.msg)
    @test occursin("item 2 of 3", limit.msg) && occursin("255", limit.msg)
    @test occursin("item 1 of 2", option.msg) && occursin("temperature", option.msg)
    @test occursin("item 2 of 2", nan.msg) && occursin("NaN", nan.msg)
    @test occursin("ntasks", ntasks.msg)
    @test occursin("System One", capability.msg)
    @test occursin("duplicate", duplicate.msg)
    @test occursin("model", model.msg)
end

@testset "ask_each — at most ntasks requests in flight; otherwise the backend's default" begin
    q = "q" => noul("x")
    three, _ = _ae_run(() -> ask_each(["s$i" for i in 1:12], q; service=_AEMock, ntasks=3), _ae_slow(0.25))
    @test all(issuccess, three)
    @test _AE_PEAK[] == 3
    @test UniLM._system_one_concurrency(UniLM.TypeSafeProtocol(), TYPESAFEServiceEndpoint) == 16
    @test UniLM._system_one_concurrency(_AEOtherProtocol(), _AEOther) == 8   # a protocol that names none
    typesafe, _ = _ae_run(() -> ask_each(["s$i" for i in 1:40], q; service=_AEMock), _ae_slow(0.25))
    @test all(issuccess, typesafe)
    @test _AE_PEAK[] == 16
    other, _ = _ae_run(() -> ask_each(["s$i" for i in 1:24], q; service=_AEOther), _ae_slow(0.25))
    @test all(issuccess, other)
    @test _AE_PEAK[] == 8
    # More workers than items: one per item, never more.
    few, _ = _ae_run(() -> ask_each(["a", "b"], q; service=_AEMock, ntasks=50), _ae_slow(0.25))
    @test all(issuccess, few) && _AE_PEAK[] == 2
end

@testset "ask_each — ntasks is any Integer of at least 1, never a Bool" begin
    q = "q" => noul("x")
    widths, _ = _ae_run(_ae_ok_script) do
        [ask_each(["a", "b"], q; service=_AEMock, ntasks=w) for w in (Int128(2), UInt8(2), big(3), Int32(1))]
    end
    @test all(rs -> length(rs) == 2 && all(issuccess, rs), widths)
    errs, seen = _ae_run(_ae_ok_script) do
        [_ae_caught(() -> ask_each(["a"], q; service=_AEMock, ntasks=w)) for w in (0, -1, big(0), Int128(2)^70, true, false)]
    end
    @test isempty(seen)
    @test all(e -> e isa ArgumentError && contains(e.msg, "ntasks"), errs)
end

@testset "ask_each — a 429 holds every attempt of the batch until the wait it asked for" begin
    q = "q" => noul("x")
    # Whichever request arrives first is rate-limited and asked to wait 0.8 s; the others
    # answer after 0.2 s, so their workers are free to send long before the wait is over.
    limited = (n, body) -> n == 1 ?
        _ae_json(429, """{"detail":"slow down"}"""; headers=["retry-after-ms" => "800"]) : (sleep(0.2); _ae_ok(body))
    rs, seen = _ae_run(() -> ask_each(["s$i" for i in 1:12], q; service=_AEMock, ntasks=4), limited)
    @test all(issuccess, rs)                               # the limited one was retried and answered
    @test length(seen) == 13
    held = [s.at - seen[1].at for s in seen if s.n > 4]    # the first four were in flight together
    @test length(held) == 9 && minimum(held) >= 0.8 - 0.02
    @test count(s -> s.retry == "1", seen) == 1            # the retry, numbered

    # A 529 that names no wait holds the batch for the longest backoff the seam draws
    # for a first attempt: 1 s.
    overloaded = (n, body) -> n == 1 ? _ae_json(529, """{"detail":"overloaded"}""") : (sleep(0.2); _ae_ok(body))
    rs2, seen2 = _ae_run(() -> ask_each(["s$i" for i in 1:8], q; service=_AEMock, ntasks=4), overloaded)
    @test all(issuccess, rs2)
    held2 = [s.at - seen2[1].at for s in seen2 if s.n > 4]
    @test length(held2) == 5 && minimum(held2) >= 1.0 - 0.02

    # Calls that are not of one batch do not wait for each other.
    _, seen3 = _ae_run(limited) do
        tasks = [Threads.@spawn ask("s$i", q; service=_AEMock) for i in 1:4]
        sleep(0.3)
        late = ask("late", q; service=_AEMock)
        (fetch.(tasks), late)
    end
    @test only(s.at for s in seen3 if s.state == "late") - seen3[1].at < 0.8
end

@testset "ask_each — a held request waits at most its own total_deadline, and then nothing is sent for it" begin
    q = "q" => noul("x")
    # The first request to arrive is asked to wait 6 s; every request may take 1 s in all.
    cfg = RequestConfig(request_timeout=2.0, total_deadline=1.0, max_attempts=3)
    limited = (n, body) -> n == 1 ?
        _ae_json(429, """{"detail":"slow down"}"""; headers=["Retry-After" => "6"]) : _ae_ok(body)
    # A single ask is unchanged: a wait longer than its budget returns the 429 at once.
    (single, t_single), seen1 = _ae_run(limited) do
        t0 = time()
        r = ask("a", q; service=_AEMock, config=cfg)
        (r, time() - t0)
    end
    @test single isa SystemOneFailure && single.status == 429 && single.retry_after == 6.0
    @test t_single < 1.0 && length(seen1) == 1
    # In a batch, each item's clock starts when its worker takes it: the held ones spend
    # their own second each, one after the other, and end at their deadline unsent.
    (rs, wall), seen = _ae_run(limited) do
        t0 = time()
        r = ask_each(["a", "b", "c"], q; service=_AEMock, ntasks=1, config=cfg)
        (r, time() - t0)
    end
    @test rs[1] isa SystemOneFailure && rs[1].status == 429
    @test all(r -> r isa SystemOneCallError && r.cause isa UniLMTimeout && r.cause.phase === :deadline &&
                   r.cause.limit == 1.0, rs[2:3])
    @test length(seen) == 1                    # nothing was sent for the held items
    @test 2.0 - 0.05 <= wall < 4.0             # their two budgets, far short of the 6 s asked for
end

@testset "ask and ask_each — TypeSafe's retries are numbered in X-TypeSafe-Retry-Count, no other protocol's" begin
    q = "q" => noul("x")
    flaky = (n, body) -> n <= 2 ? _ae_json(503, "{}") : _ae_ok(body)
    r, seen = _ae_run(() -> ask("s", q; service=_AEMock), flaky)
    @test issuccess(r)
    @test [s.retry for s in seen] == [nothing, "1", "2"]        # never on the first attempt
    r2, seen2 = _ae_run(() -> ask("s", q; service=_AEOther), flaky)
    @test issuccess(r2)
    @test length(seen2) == 3 && all(s -> isnothing(s.retry), seen2)
    @test isnothing(UniLM._system_one_retry_header(_AEOtherProtocol()))
    # Not part of what a recording is keyed on: a retried request files its answer under the
    # key of its body, as one answered at once does.
    mktempdir() do dir
        rec, seen3 = _ae_run(flaky) do
            with_recorded_answers(dir; mode=:record) do
                ask("s", q; service=_AEMock, model="jev-1.13.0")
            end
        end
        @test issuccess(rec) && length(seen3) == 3
        mktempdir() do direct
            _ae_run(_ae_ok_script) do
                with_recorded_answers(direct; mode=:record) do
                    ask("s", q; service=_AEMock, model="jev-1.13.0")
                end
            end
            @test readdir(direct) == readdir(dir) && length(readdir(dir)) == 1
        end
    end
end

@testset "ask_each — a cancel ends the requests in flight and sends no more; every item reports it" begin
    q = "q" => noul("x")
    tok = CancelToken()
    reported = Threads.Atomic{Int}(0)
    _ae_drained()
    t = Threads.@spawn _ae_run(_ae_slow(2.0)) do
        ask_each(["s$i" for i in 1:10], q; service=_AEMock, ntasks=2, cancel=tok,
                 on_result=(i, r) -> Threads.atomic_add!(reported, 1))
    end
    @test timedwait(() -> _AE_INFLIGHT[] == 2, 10.0) === :ok      # two in flight
    cancelled_at = time()
    cancel!(tok)
    @test timedwait(() -> istaskdone(t), 5.0) === :ok
    @test time() - cancelled_at < 1.5                              # not the 2 s the server takes
    rs, seen = fetch(t)
    @test length(rs) == 10
    @test all(r -> r isa SystemOneCallError && r.cause isa UniLMCancelled && r.cause.source === :token, rs)
    @test length(seen) == 2                                        # the rest were never sent
    @test reported[] == 10
    @test isempty(tok.hooks)                                       # the batch let go of the token

    # The ambient token, already cancelled: nothing is sent at all.
    done = CancelToken()
    cancel!(done)
    rs2, seen2 = _ae_run(_ae_ok_script) do
        with_cancel(done) do
            ask_each(["a", "b", "c"], q; service=_AEMock)
        end
    end
    @test all(r -> r isa SystemOneCallError && r.cause isa UniLMCancelled, rs2)
    @test isempty(seen2)
end

@testset "ask_each — a lost recording or a failing on_result stops the batch and reaches the caller" begin
    q = "q" => noul("x")
    # Nothing recorded: the first miss stops the batch, as it stops ask.
    mktempdir() do dir
        e, seen = _ae_run(_ae_ok_script) do
            _ae_caught(() -> with_recorded_answers(dir) do
                ask_each(["a", "b", "c", "d"], q; service=_AEMock, ntasks=2)
            end)
        end
        @test e isa ReplayMissError
        @test isempty(seen)
    end
    # A paid answer whose recording cannot be written.
    dir = joinpath(mktempdir(), "answers")
    e, _ = _ae_run(_ae_ok_script) do
        with_recorded_answers(dir; mode=:record) do
            rm(dir; force=true, recursive=true)
            write(dir, "")                   # the checked directory is a file when the answer arrives
            _ae_caught(() -> ask_each(["a", "b"], q; service=_AEMock, ntasks=1))
        end
    end
    @test e isa RecordingWriteError

    # on_result throws on the first result: the requests in flight are cancelled, not
    # waited for, and the exception itself reaches the caller.
    quick_first = (n, body) -> (body["state"] == "quick" || sleep(2.0); _ae_ok(body))
    started = time()
    e2, seen2 = _ae_run(quick_first) do
        _ae_caught(() -> ask_each(["quick", "s2", "s3", "s4", "s5", "s6"], q; service=_AEMock, ntasks=3,
                                  on_result=(i, r) -> error("consumer failed at $i")))
    end
    @test e2 isa ErrorException && e2.msg == "consumer failed at 1"
    @test time() - started < 1.5
    @test length(seen2) <= 3                 # no item was claimed once the batch had stopped

    e3, _ = _ae_run(_ae_ok_script) do
        _ae_caught(() -> ask_each(["a", "b"], q; service=_AEMock, ntasks=1, on_result=(i, r) -> throw(InterruptException())))
    end
    @test e3 isa InterruptException
end

@testset "ask_each — on_result runs one call at a time, whatever the number of workers" begin
    q = "q" => noul("x")
    current, peak = Threads.Atomic{Int}(0), Threads.Atomic{Int}(0)
    landed, lk = Int[], ReentrantLock()
    rs, _ = _ae_run(_ae_ok_script) do
        ask_each(["s$i" for i in 1:12], q; service=_AEMock, ntasks=4, on_result=(i, r) -> begin
            Threads.atomic_max!(peak, Threads.atomic_add!(current, 1) + 1)
            @lock lk push!(landed, i)
            sleep(0.05)                          # long enough for the other workers' results to land
            Threads.atomic_sub!(current, 1)
        end)
    end
    @test all(issuccess, rs)
    @test peak[] == 1
    @test sort(landed) == 1:12
end

# A callable that is not a Function.
struct _AETally
    n::Threads.Atomic{Int}
end
(t::_AETally)(i, r) = Threads.atomic_add!(t.n, 1)

@testset "ask_each — on_result is any callable of (i, result), checked before anything is sent" begin
    q = "q" => noul("x")
    tally = _AETally(Threads.Atomic{Int}(0))
    rs, _ = _ae_run(() -> ask_each(["a", "b", "c"], q; service=_AEMock, on_result=tally), _ae_ok_script)
    @test all(issuccess, rs) && tally.n[] == 3
    errs, seen = _ae_run(_ae_ok_script) do
        [_ae_caught(() -> ask_each(["a"], q; service=_AEMock, on_result=r -> nothing)),
         _ae_caught(() -> ask_each(["a"], q; service=_AEMock, on_result=(i, r::SystemOneSuccess) -> nothing)),
         _ae_caught(() -> ask_each(["a"], q; service=_AEMock, on_result="not callable"))]
    end
    @test isempty(seen)                          # refused before the batch started
    @test all(e -> e isa ArgumentError && contains(e.msg, "on_result"), errs)
end

@testset "ask_each — every request is checked before the first is sent, and encoded again as it is sent" begin
    q = "q" => noul("x")
    noted = (n, body) -> (@lock(_AE_LOCK, push!(_AE_EVENTS, "send " * body["state"])); _ae_ok(body))
    @lock _AE_LOCK empty!(_AE_EVENTS)
    rs, _ = _ae_run(() -> ask_each(["a", "b", "c"], q; service=_AECounted, ntasks=1), noted)
    @test all(issuccess, rs)
    # Checked up front, then each worker encodes its item when it takes it, so no more than
    # `ntasks` bodies are held at once.
    @test @lock(_AE_LOCK, copy(_AE_EVENTS)) ==
          ["encode a", "encode b", "encode c", "encode a", "send a", "encode b", "send b", "encode c", "send c"]
end

@testset "ask_each — no task or descriptor outlives a batch, however it ends" begin
    q = "q" => noul("x")
    function batches()
        _ae_run(() -> ask_each(["s$i" for i in 1:20], q; service=_AEMock, ntasks=4), _ae_slow(0.02))
        tok = CancelToken()
        t = Threads.@spawn _ae_run(() -> ask_each(["s$i" for i in 1:8], q; service=_AEMock, ntasks=4, cancel=tok),
                                   _ae_slow(0.5))
        sleep(0.2); cancel!(tok); fetch(t)
        _ae_run(() -> _ae_caught(() -> ask_each(["s$i" for i in 1:8], q; service=_AEMock, ntasks=4,
                                                on_result=(i, r) -> error("stop"))), _ae_slow(0.05))
        _ae_drained()
    end
    batches()                                 # warm: compile every path, fill the pool
    GC.gc(); GC.gc(); sleep(1.0); GC.gc()
    base = (; tasks=_ae_task_count(), fds=_ae_fd_count())
    for _ in 1:3
        batches()
    end
    GC.gc(); GC.gc(); sleep(1.0); GC.gc()
    tasks, fds = _ae_task_count() - base.tasks, _ae_fd_count() - base.fds
    (tasks <= 8 && fds <= 8) || @warn "ask_each gauges above baseline" tasks fds
    @test tasks <= 8
    @test fds <= 8
end

@testset "a backend's own default config governs its System One calls, below every channel the user sets" begin
    q = "q" => noul("x")
    @test UniLM._system_one_default_config(UniLM.TypeSafeProtocol(), TYPESAFEServiceEndpoint) ==
          RequestConfig(request_timeout=60.0, total_deadline=150.0)
    @test isnothing(UniLM._system_one_default_config(_AEOtherProtocol(), _AEOther))
    limit(r) = r isa SystemOneCallError && r.cause isa UniLMTimeout ? r.cause.limit : nothing
    _ae_backend_defaults() do
        # The server takes 1 s; the backend allows 0.3 s per attempt, one attempt.
        rs, _ = _ae_run(() -> ask_each(["a", "b"], q; service=_AETight), _ae_slow(1.0))
        @test all(r -> limit(r) == 0.3, rs)
        @test limit(first(_ae_run(() -> ask("a", q; service=_AETight), _ae_slow(1.0)))) == 0.3
        # Any channel the user controls replaces it, as a whole.
        scoped = with_request_config(request_timeout=0.2, max_attempts=1) do
            first(_ae_run(() -> ask_each(["a"], q; service=_AETight), _ae_slow(1.0)))
        end
        @test limit(only(scoped)) == 0.2
        percall = first(_ae_run(() -> ask_each(["a"], q; service=_AETight,
                                               config=RequestConfig(request_timeout=0.25, max_attempts=1)), _ae_slow(1.0)))
        @test limit(only(percall)) == 0.25
        set_default_config!(RequestConfig(request_timeout=0.35, max_attempts=1))
        @test limit(first(_ae_run(() -> ask("a", q; service=_AETight), _ae_slow(1.0)))) == 0.35
    end
end

@testset "System One's pool — a connection the server closed while idle costs a retry, never a failure" begin
    q = "q" => noul("x")
    srv = _ae_idle_server(2.0)
    saved = _AE_BASE[]
    _AE_BASE[] = "http://127.0.0.1:$(srv.port)"
    try
        # Within the server's idle window the pooled connection is reused.
        quick = [ask("s$i", q; service=_AEMock) for i in 1:3]
        @test all(issuccess, quick)
        @test srv.accepted[] == 1
        # Three seconds apart, the server has closed it each time: the request written on
        # it fails, the seam retries it on a fresh connection, and the call succeeds.
        spaced = map(1:3) do i
            sleep(3.0)
            ask("t$i", q; service=_AEMock)
        end
        @test all(issuccess, spaced)
        @test srv.accepted[] == 4
        @test @lock(srv.lock, copy(srv.retries)) == [nothing, nothing, nothing, "1", "1", "1"]
    finally
        _AE_BASE[] = saved
        close(srv.listener)
    end
end

@testset "System One's pool — created by the first live request, never by loading or a replay" begin
    q = "q" => noul("x")
    @test UniLM._system_one_client() isa HTTP.Client
    @test UniLM._system_one_client() === UniLM._system_one_client()      # one per process
    dir = mktempdir()
    # `_AEMock` speaks TypeSafe's wire, so its recording is the one TypeSafe's request replays.
    _ae_run(_ae_ok_script) do
        with_recorded_answers(dir; mode=:record) do
            ask("lazy", q; service=_AEMock, model="jev-1.13.0")
        end
    end
    script = """
    using UniLM
    delete!(ENV, "TYPESAFE_API_KEY")
    before = UniLM._SYSTEM_ONE_CLIENT.value === nothing
    r = with_recorded_answers($(repr(dir))) do
        ask("lazy", "q" => noul("x"); model="jev-1.13.0")
    end
    print(before, " ", issuccess(r), " ", UniLM._SYSTEM_ONE_CLIENT.value === nothing)
    """
    proj = dirname(dirname(pathof(UniLM)))
    out = read(`$(Base.julia_cmd()) --startup-file=no --project=$proj -e $script`, String)
    @test out == "true true true"
end

close(_AE_SERVER)
