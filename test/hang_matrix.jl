# ============================================================================
# Hang matrix — bounded-timeout falsification suite. Every operation in UniLM
# (provider request, stream, MCP exchange) must complete or fail with a TYPED
# error within a configurable, bounded time. Each testset states the FIXED
# contract wrapped in @test_broken; a timeout fix flips @test_broken → @test.
#
# BOUNDED OBSERVATION: a test that would hang is not a test. Every pin either
# fails fast on a not-yet-implemented symbol/kwarg, or runs the operation in a
# task observed with a short `timedwait` bound. Fully offline (zero-spend):
# all servers are 127.0.0.1 and no provider key is read.
# ============================================================================

using Sockets

# Run `f()` on a task; observe it with a short wall-clock bound so a hang can
# never block the suite. Returns:
#   (:ok, value)       — finished within `bound`; `fetch` returned `value`
#   (:threw, exc)      — finished within `bound`; the task threw `exc`
#   (:timeout, nothing)— did not finish within `bound`
# The guarded `fetch` is what lets a task that threw a not-yet-defined symbol
# (UndefVarError/MethodError on the base) surface as (:threw, …) → the pin's
# ok-expression stays `false` → Broken, never Error.
#
# CALLER CONTRACT — reclaiming a `:timeout` task: on a `:timeout` outcome the
# spawned task is still alive (a Julia task cannot be killed from the outside).
# It is deliberately left for the caller's cleanup to reclaim: the caller MUST
# close the server, socket, or process its closure blocks on, inside a `finally`
# block. Closing that resource unblocks the abandoned task — its pending
# read/accept/wait throws — so the task finishes and is garbage-collected.
# Every pin below that opens such a resource therefore closes it in `finally`.
function _hm_bounded(f; bound::Float64 = 45.0)
    t = Threads.@spawn f()
    if timedwait(() -> istaskdone(t), bound) != :ok
        return (:timeout, nothing)
    end
    try
        (:ok, fetch(t))
    catch e
        (:threw, e)
    end
end

# ─── RequestConfig channels & validation (server-free unit contracts) ────────

@testset "config: constructor rejects NaN and non-positive; Inf disables" begin
    # FIXED contract: the inner constructor rejects NaN and ≤ 0 on every Float64
    # field with ArgumentError (NaN checked explicitly — `NaN < 0` is false, a
    # silently-reintroduced infinite hang), rejects max_attempts < 1, and ACCEPTS
    # Inf (Inf = "disabled", not an error).
    _rejects(f) = try
        f()
        false
    catch e
        e isa ArgumentError
    end
    ok = try
        _rejects(() -> UniLM.RequestConfig(connect_timeout = NaN)) &&
        _rejects(() -> UniLM.RequestConfig(request_timeout = 0.0)) &&
        _rejects(() -> UniLM.RequestConfig(total_deadline = -1.0)) &&
        _rejects(() -> UniLM.RequestConfig(mcp_request_timeout = NaN)) &&
        _rejects(() -> UniLM.RequestConfig(max_attempts = 0)) &&
        (UniLM.RequestConfig(request_timeout = Inf) isa UniLM.RequestConfig) &&
        (UniLM.RequestConfig(stream_idle_timeout = Inf) isa UniLM.RequestConfig)
    catch
        false
    end
    @test ok
end

@testset "config: channel precedence kwarg > scope > process default; streams capture at spawn" begin
    # FIXED contract: resolution precedence is per-call config > dynamic scope >
    # process default; and a config resolved INSIDE a scope is immune to a later
    # mutation of the process default (streams close over the resolved struct at
    # spawn time). `_resolve_config(nothing)` = current_config().
    ok = try
        # process default (lowest precedence)
        UniLM.set_default_config!(UniLM.RequestConfig(max_attempts = 7))
        default_seen = UniLM.current_config().max_attempts == 7
        # dynamic scope overrides the process default
        scoped_seen = UniLM.with_request_config(max_attempts = 3) do
            UniLM.current_config().max_attempts == 3
        end
        # per-call config overrides the scope
        kwarg_wins = UniLM.with_request_config(max_attempts = 3) do
            UniLM._resolve_config(UniLM.RequestConfig(max_attempts = 2)).max_attempts == 2
        end
        # capture-at-spawn: a value resolved in-scope is unaffected by a later
        # mutation of the process default
        captured = UniLM.with_request_config(max_attempts = 5) do
            snap = UniLM._resolve_config(nothing)
            UniLM.set_default_config!(UniLM.RequestConfig(max_attempts = 9))
            snap.max_attempts == 5
        end
        # The config machinery above satisfies precedence + snapshot on its own,
        # so this pin would pass the moment the core config machinery merges —
        # but its flip only lands with the requests migration that adds the
        # `config` kwarg to chatrequest! (the channel a running stream resolves
        # through). Gate on that kwarg so the pin cannot pass before its flip
        # owner ships. `hasmethod(f, Tuple{Chat}, (:config,))` is false while the
        # kwarg is absent and true once the migration adds it.
        hasmethod(chatrequest!, Tuple{Chat}, (:config,)) &&
            default_seen && scoped_seen && kwarg_wins && captured
    catch
        false
    finally
        # restore the process default so this unit does not leak into other tests
        try
            UniLM.set_default_config!(UniLM.RequestConfig())
        catch
        end
    end
    @test ok

    # Capture-at-spawn through a RUNNING STREAM — the clause only the streaming
    # driver can prove. A stream started under a scoped idle limit must honor
    # THAT limit even after the process default is mutated post-spawn, because
    # the driver resolves `cfg` BEFORE `Threads.@spawn` and the task closes over
    # the resolved struct. The server sends one delta then goes mute: under the
    # captured idle=1.0 s the guard fires ~1.25 s in (well inside the bound); a
    # driver that instead re-read the live default (30 s) would not finish in
    # time. The pre-rearchitecture driver has no idle guard at all → hangs → red.
    stream_captured = begin
        ev = "data: {\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"hi\"},\"finish_reason\":null}]}\n\n"
        server = nothing
        sport = 0
        for attempt in 1:5
            tcp = Sockets.listen(Sockets.localhost, 0)
            sport = Int(Sockets.getsockname(tcp)[2])
            close(tcp)
            try
                server = HTTP.listen!("127.0.0.1", sport; verbose = false) do http::HTTP.Stream
                    read(http)
                    HTTP.setstatus(http, 200)
                    HTTP.setheader(http, "Content-Type" => "text/event-stream")
                    HTTP.startwrite(http)
                    write(http, ev)
                    flush(http)
                    sleep(8)   # mute past the captured idle limit and the bound below
                end
                break
            catch
                attempt == 5 && rethrow()
            end
        end
        try
            # Live process default = a LONG idle limit; a stream resolving live
            # would inherit 30 s and miss the bound below.
            UniLM.set_default_config!(UniLM.RequestConfig(stream_idle_timeout = 30.0, max_attempts = 1))
            task = UniLM.with_request_config(stream_idle_timeout = 1.0, request_timeout = 5.0,
                                             total_deadline = 10.0, max_attempts = 1) do
                chat = Chat(service = GenericOpenAIEndpoint("http://127.0.0.1:$(sport)", ""),
                            model = "mock", stream = true)
                push!(chat, Message(Val(:system), "s"))
                push!(chat, Message(Val(:user), "u"))
                chatrequest!(chat)   # resolves scoped idle=1.0 at call entry, before the spawn
            end
            # Mutate the process default AGAIN post-spawn: the captured struct must win.
            UniLM.set_default_config!(UniLM.RequestConfig(stream_idle_timeout = 30.0, max_attempts = 1))
            done = timedwait(() -> istaskdone(task), 30.0)
            res = done === :ok ? fetch(task) : nothing
            done === :ok && res isa LLMCallError && res.cause isa UniLM.UniLMTimeout &&
                res.cause.phase === :stream_idle
        catch
            false
        finally
            isnothing(server) || HTTP.forceclose(server)
            try
                UniLM.set_default_config!(UniLM.RequestConfig())
            catch
            end
        end
    end
    @test stream_captured
end

# ─── HTTP hang/fault mock servers (all 127.0.0.1, race-free ephemeral bind) ───

# Raw-TCP mute server: accepts connections and never sends a valid HTTP
# response, so an HTTP client blocks on read until its own deadline fires.
# `drain=true` first reads and discards the request bytes (accept-then-zero-
# bytes, the HTTP.jl#1331 shape: the write phase completes, the read hangs).
# Bound directly to an ephemeral port — no probe/close/re-bind window, so the
# port-steal TOCTOU that HTTP.listen! callers must retry cannot occur here.
function _hm_mute_server(; drain::Bool = false)
    srv = Sockets.listen(Sockets.localhost, 0)
    port = Int(Sockets.getsockname(srv)[2])
    acceptor = Threads.@spawn begin
        try
            while true
                conn = Sockets.accept(srv)   # throws once `srv` is closed → loop ends
                Threads.@spawn begin
                    try
                        drain && while !eof(conn)
                            readavailable(conn)
                        end
                        # Never write a response; hold until the socket is closed.
                    catch
                    end
                end
            end
        catch
            # listen socket closed → stop accepting
        end
    end
    (srv, "http://127.0.0.1:$(port)", acceptor)
end

# HTTP server that returns a fixed retryable status (optionally with Retry-After)
# and counts requests. After `hang_after` requests it stops responding — an
# over-budget tripwire: a client that exceeds the attempt budget blocks here and
# is caught by the pin's `timedwait`, never producing a spurious pass.
function _hm_status_server(; status::Int, retry_after::Union{Int,Nothing} = nothing,
                           hang_after::Int = typemax(Int))
    server = nothing
    port = 0
    calls = Ref(0)
    for attempt in 1:5
        tcp = Sockets.listen(Sockets.localhost, 0)
        port = Int(Sockets.getsockname(tcp)[2])
        close(tcp)
        try
            server = HTTP.listen!("127.0.0.1", port; verbose = false) do http::HTTP.Stream
                read(http)                       # drain the request body
                n = (calls[] += 1)
                n > hang_after && wait(Condition())   # never responds past the budget
                HTTP.setstatus(http, status)
                retry_after === nothing ||
                    HTTP.setheader(http, "Retry-After" => string(retry_after))
                HTTP.setheader(http, "Content-Type" => "application/json")
                HTTP.startwrite(http)
                write(http, JSON.json(Dict{String,Any}("error" =>
                    Dict{String,Any}("message" => "mock $(status)"))))
            end
            break
        catch
            attempt == 5 && rethrow()
        end
    end
    (server, "http://127.0.0.1:$(port)", calls)
end

# Healthy non-streaming OpenAI-wire server: returns `body` verbatim to every
# POST and counts requests. Used to prove the connection pool survives repeated
# timeouts to a DIFFERENT (mute) peer.
function _hm_healthy_oai_server(body::String)
    server = nothing
    port = 0
    calls = Ref(0)
    for attempt in 1:5
        tcp = Sockets.listen(Sockets.localhost, 0)
        port = Int(Sockets.getsockname(tcp)[2])
        close(tcp)
        try
            server = HTTP.listen!("127.0.0.1", port; verbose = false) do http::HTTP.Stream
                read(http)
                calls[] += 1
                HTTP.setstatus(http, 200)
                HTTP.setheader(http, "Content-Type" => "application/json")
                HTTP.startwrite(http)
                write(http, body)
            end
            break
        catch
            attempt == 5 && rethrow()
        end
    end
    (server, "http://127.0.0.1:$(port)", calls)
end

# Mock service whose request encoding raises InterruptException, simulating a
# user Ctrl-C while the (synchronous) call is in flight — no network involved.
struct _HMInterruptService <: UniLM.ServiceEndpoint end
UniLM.encode_request(::_HMInterruptService, ::Chat) = throw(InterruptException())
UniLM.get_url(::_HMInterruptService, ::Chat) = "http://127.0.0.1:1/never"
UniLM.auth_header(::_HMInterruptService) = ["Content-Type" => "application/json"]

# ─── Non-stream / retry-budget / interrupt / pool contracts ──────────────────

@testset "chat non-stream: mute server yields typed timeout" begin
    # A peer that accepts then sends zero bytes (HTTP.jl#1331 shape) must not
    # hang the call forever. FIXED contract: with a short request_timeout the
    # watchdog closes the connection and the timeout lands INSIDE the value-based
    # result — LLMCallError with status === nothing and cause::UniLMTimeout.
    srv, url, _ = _hm_mute_server(drain = true)
    try
        chat = Chat(service = GenericOpenAIEndpoint(url, ""), model = "mock")
        push!(chat, Message(Val(:system), "s"))
        push!(chat, Message(Val(:user), "u"))
        outcome = _hm_bounded() do
            cfg = UniLM.RequestConfig(request_timeout = 1.0, total_deadline = 2.0, max_attempts = 1)
            chatrequest!(chat; config = cfg)
        end
        ok = try
            outcome[1] === :ok && let res = outcome[2]
                res isa LLMCallError && res.status === nothing && res.cause isa UniLM.UniLMTimeout
            end
        catch
            false
        end
        @test ok
    finally
        close(srv)
    end
end

@testset "retry budget: fail-N stops at max_attempts" begin
    # FIXED contract: the shared retry loop makes exactly max_attempts requests
    # to a persistently-retryable (503) peer, then returns the LAST real response
    # as LLMFailure(status=503) — no fabricated timeout, never past the budget.
    # The server hangs on request 4 (hang_after=3): an over-budget client would
    # block there and trip the timedwait bound.
    srv, url, calls = _hm_status_server(status = 503, hang_after = 3)
    try
        chat = Chat(service = GenericOpenAIEndpoint(url, ""), model = "mock")
        push!(chat, Message(Val(:system), "s"))
        push!(chat, Message(Val(:user), "u"))
        outcome = _hm_bounded(bound = 45.0) do
            cfg = UniLM.RequestConfig(request_timeout = 2.0, total_deadline = 30.0, max_attempts = 3)
            chatrequest!(chat; config = cfg)
        end
        ok = try
            outcome[1] === :ok && let res = outcome[2]
                res isa LLMFailure && res.status == 503 && calls[] == 3
            end
        catch
            false
        end
        @test ok
    finally
        close(srv)
    end
end

@testset "retry budget: Retry-After beyond remaining deadline fails immediately" begin
    # FIXED contract: when the server's Retry-After (60 s) exceeds the remaining
    # total_deadline (~5 s), the loop NEVER sleeps past the deadline — it returns
    # the real 429 immediately (LLMFailure(status=429)) after a single attempt.
    # `dt < 2.0` falsifies any implementation that honored the 60 s wait.
    srv, url, calls = _hm_status_server(status = 429, retry_after = 60, hang_after = 1)
    try
        chat = Chat(service = GenericOpenAIEndpoint(url, ""), model = "mock")
        push!(chat, Message(Val(:system), "s"))
        push!(chat, Message(Val(:user), "u"))
        outcome = _hm_bounded() do
            cfg = UniLM.RequestConfig(request_timeout = 5.0, total_deadline = 5.0, max_attempts = 5)
            t0 = time()
            res = chatrequest!(chat; config = cfg)
            (res, time() - t0)
        end
        ok = try
            outcome[1] === :ok && let (res, dt) = outcome[2]
                res isa LLMFailure && res.status == 429 && calls[] == 1 && dt < 2.0
            end
        catch
            false
        end
        @test ok
    finally
        close(srv)
    end
end

@testset "interrupts surface, never laundered into result values" begin
    # FIXED contract: an InterruptException raised while a call is in flight is
    # re-thrown FIRST (surfacing the user's cancel), never captured into an
    # LLMCallError result value. On the base the catch-to-value layer launders
    # it — so chatrequest! RETURNS (no throw) and the expression is false.
    chat = Chat(service = _HMInterruptService(), model = "mock")
    push!(chat, Message(Val(:system), "s"))
    push!(chat, Message(Val(:user), "u"))
    ok = try
        chatrequest!(chat)   # base: returns an LLMCallError (no throw)
        false
    catch e
        e isa InterruptException
    end
    @test ok
end

@testset "repeated HTTP timeouts leak no tasks and do not poison the pool" begin
    # FIXED contract: N=20 repeated timeouts to a mute peer each surface a TYPED
    # timeout AND complete within a per-cycle bound — a driver task that leaked
    # (never returned) shows up as :hung and fails the shape assertion — and they
    # leave the client healthy on BOTH connection-pool keys: a FRESH peer succeeds
    # (no process-global poison) AND the SAME host:port that just timed out, once
    # responsive, succeeds. That same-key clause is the load-bearing one: a
    # timed-out connection wrongly kept in the 2.x pool as reusable would be drawn
    # by the same-key request and fail, whereas a different peer draws a fresh pool
    # entry and can never witness the poison. A before/after live-guard COUNT would
    # need a tracking accessor deadline.jl does not expose; the per-cycle bound is
    # the task-completion evidence here.
    healthy_body = JSON.json(Dict{String,Any}(
        "id" => "c", "object" => "chat.completion",
        "choices" => [Dict{String,Any}("index" => 0, "finish_reason" => "stop",
            "message" => Dict{String,Any}("role" => "assistant", "content" => "pong"))],
        "usage" => Dict{String,Any}("prompt_tokens" => 1, "completion_tokens" => 1, "total_tokens" => 2)))
    hsrv, hurl, _ = _hm_healthy_oai_server(healthy_body)   # fresh-peer (cross-key) probe

    # One peer, two phases: it drains each request then holds past the client's
    # request_timeout (the deadline fires → typed timeout) while `responsive` is
    # false, and serves `healthy_body` once it flips true. Reusing this exact
    # host:port for the final health check drives the pool's reuse path for a key
    # that just timed out.
    responsive = Ref(false)
    tsrv = nothing
    tport = 0
    for attempt in 1:5
        tcp = Sockets.listen(Sockets.localhost, 0)
        tport = Int(Sockets.getsockname(tcp)[2])
        close(tcp)
        try
            tsrv = HTTP.listen!("127.0.0.1", tport; verbose = false) do http::HTTP.Stream
                read(http)
                if responsive[]
                    HTTP.setstatus(http, 200)
                    HTTP.setheader(http, "Content-Type" => "application/json")
                    HTTP.startwrite(http)
                    write(http, healthy_body)
                else
                    sleep(3.0)   # outlast the 0.5 s request_timeout below, then reap
                end
            end
            break
        catch
            attempt == 5 && rethrow()
        end
    end
    turl = "http://127.0.0.1:$(tport)"

    try
        outcome = _hm_bounded(bound = 120.0) do
            cfg = UniLM.RequestConfig(request_timeout = 0.5, total_deadline = 1.0, max_attempts = 1)
            mute_chat = Chat(service = GenericOpenAIEndpoint(turl, ""), model = "mock")
            push!(mute_chat, Message(Val(:system), "s"))
            push!(mute_chat, Message(Val(:user), "u"))
            # Each cycle runs on its own task, observed under a per-cycle bound: a
            # task that never completes surfaces as :hung (never a suite hang), so
            # the typed-outcome assertion doubles as a per-task no-leak check.
            results = map(1:20) do _
                t = Threads.@spawn chatrequest!(mute_chat; config = cfg)
                timedwait(() -> istaskdone(t), 25.0) === :ok || return :hung
                fetch(t)
            end
            # Fresh key: a brand-new peer still succeeds after the timeout storm.
            fresh_chat = Chat(service = GenericOpenAIEndpoint(hurl, ""), model = "mock")
            push!(fresh_chat, Message(Val(:system), "s"))
            push!(fresh_chat, Message(Val(:user), "u"))
            fresh = chatrequest!(fresh_chat;
                config = UniLM.RequestConfig(request_timeout = 5.0, total_deadline = 10.0, max_attempts = 1))
            # Same key (LAST): the port that just timed out 20× is now responsive;
            # a request to that SAME host:port must succeed — the pool is not poisoned.
            responsive[] = true
            same_chat = Chat(service = GenericOpenAIEndpoint(turl, ""), model = "mock")
            push!(same_chat, Message(Val(:system), "s"))
            push!(same_chat, Message(Val(:user), "u"))
            same = chatrequest!(same_chat;
                config = UniLM.RequestConfig(request_timeout = 5.0, total_deadline = 10.0, max_attempts = 1))
            (results, fresh, same)
        end
        # Labeled sub-assertions (semantics identical to the former single @test):
        # a future CI failure names the component that broke. When the whole cycle
        # times out the fallback drives every clause red, matching the old outcome.
        completed = outcome[1] === :ok
        @test completed   # whole cycle finished within the 120 s bound — no suite-level hang
        results, fresh, same = completed ? outcome[2] : (fill(:hung, 20), nothing, nothing)
        # task hygiene: every one of the 20 driver tasks completed (none leaked as :hung)
        @test count(==(:hung), results) == 0
        # per-cycle typed outcomes: each of the 20 timeouts surfaced as a typed UniLMTimeout value
        @test all(r -> r isa LLMCallError && r.cause isa UniLM.UniLMTimeout, results)
        # cross-peer health: a brand-new peer still succeeds after the timeout storm
        @test fresh isa LLMSuccess
        # same-peer recovery: the exact host:port that timed out 20× succeeds once responsive
        @test same isa LLMSuccess
    finally
        HTTP.forceclose(tsrv)
        close(hsrv)
    end
end

# ─── Streaming SSE mock servers (raw-byte dribble; idle is a BYTE-gap) ────────

# Write `s` to an HTTP.Stream in small byte-chunks separated by `gap` seconds,
# so the client's readavailable() sees genuine mid-stream byte gaps. The SSE
# payloads here are ASCII, so byte slicing never splits a character.
function _hm_dribble(http, s::String; chunk::Int = 12, gap::Float64 = 0.3)
    bytes = codeunits(s)
    i = 1
    while i <= length(bytes)
        j = min(i + chunk - 1, length(bytes))
        write(http, bytes[i:j])
        flush(http)
        sleep(gap)
        i = j + 1
    end
end

# Sends ONE valid content delta (no finish_reason), then goes silent forever.
# The idle guard must fire on the byte-gap and surface a :stream_idle timeout.
function _hm_first_event_then_hang_server()
    ev = "data: {\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"hi\"},\"finish_reason\":null}]}\n\n"
    server = nothing
    port = 0
    for attempt in 1:5
        tcp = Sockets.listen(Sockets.localhost, 0)
        port = Int(Sockets.getsockname(tcp)[2])
        close(tcp)
        try
            server = HTTP.listen!("127.0.0.1", port; verbose = false) do http::HTTP.Stream
                read(http)
                HTTP.setstatus(http, 200)
                HTTP.setheader(http, "Content-Type" => "text/event-stream")
                HTTP.startwrite(http)
                write(http, ev)
                flush(http)
                wait(Condition())   # silence past the idle limit → guard fires
            end
            break
        catch
            attempt == 5 && rethrow()
        end
    end
    (server, "http://127.0.0.1:$(port)")
end

# A healthy but SLOW stream: the payload is dribbled in byte-chunks with gaps
# BELOW the idle limit, and a would-be idle gap is bridged by SSE comment
# ("ping") lines. Because idle is a raw-BYTE gap, comment bytes reset the clock
# — the whole stream (total wall-clock > the idle limit) must survive.
function _hm_trickle_ping_server()
    head = "data: {\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"Hello\"},\"finish_reason\":null}]}\n\n"
    tail = "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\" world\"},\"finish_reason\":null}]}\n\n" *
           "data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n" *
           "data: [DONE]\n\n"
    server = nothing
    port = 0
    for attempt in 1:5
        tcp = Sockets.listen(Sockets.localhost, 0)
        port = Int(Sockets.getsockname(tcp)[2])
        close(tcp)
        try
            server = HTTP.listen!("127.0.0.1", port; verbose = false) do http::HTTP.Stream
                read(http)
                HTTP.setstatus(http, 200)
                HTTP.setheader(http, "Content-Type" => "text/event-stream")
                HTTP.startwrite(http)
                _hm_dribble(http, head; chunk = 12, gap = 0.3)
                for _ in 1:5                              # ~1.5 s bridged by pings (idle limit 1.0 s)
                    write(http, ": ping\n\n")
                    flush(http)
                    sleep(0.3)
                end
                _hm_dribble(http, tail; chunk = 12, gap = 0.3)
            end
            break
        catch
            attempt == 5 && rethrow()
        end
    end
    (server, "http://127.0.0.1:$(port)")
end

# Anthropic-wire mock: first connection dies with an in-band `overloaded_error`
# event (the documented 529-equivalent on an HTTP-200 stream); the second
# connection is a valid completion. Counts connections.
function _hm_inband_then_valid_server()
    calls = Ref(0)
    err_frame = "event: error\ndata: {\"type\":\"error\",\"error\":{\"type\":\"overloaded_error\",\"message\":\"Overloaded\"}}\n\n"
    ok_stream = "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":1,\"output_tokens\":1}}}\n\n" *
                "event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n" *
                "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"ok\"}}\n\n" *
                "event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\n" *
                "event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":2}}\n\n" *
                "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n"
    server = nothing
    port = 0
    for attempt in 1:5
        tcp = Sockets.listen(Sockets.localhost, 0)
        port = Int(Sockets.getsockname(tcp)[2])
        close(tcp)
        try
            server = HTTP.listen!("127.0.0.1", port; verbose = false) do http::HTTP.Stream
                read(http)
                n = (calls[] += 1)
                HTTP.setstatus(http, 200)
                HTTP.setheader(http, "Content-Type" => "text/event-stream")
                HTTP.startwrite(http)
                write(http, n == 1 ? err_frame : ok_stream)
                flush(http)
            end
            break
        catch
            attempt == 5 && rethrow()
        end
    end
    (server, "http://127.0.0.1:$(port)", calls)
end

# Routes chat streaming to a local mock server while delegating SSE semantics to
# the real Anthropic handler — the URL seam the production endpoint lacks.
struct _HMAnthropicWireMock <: UniLM.ServiceEndpoint
    base_url::String
end
UniLM.get_url(s::_HMAnthropicWireMock, ::Chat) = s.base_url
UniLM.auth_header(::_HMAnthropicWireMock) = ["Content-Type" => "application/json"]
UniLM.handle_sse_event!(::_HMAnthropicWireMock, event::AbstractString, payload::AbstractString,
                        state::UniLM.StreamState) =
    UniLM.handle_sse_event!(ANTHROPICServiceEndpoint, event, payload, state)

# ─── Streaming timeout contracts ─────────────────────────────────────────────

@testset "stream: mute pre-first-byte yields typed timeout" begin
    # FIXED contract: a mute peer that never sends response headers must not hang
    # the stream — the first-byte deadline (min(remaining total, request_timeout))
    # closes the connection and the stream task's result is LLMCallError with
    # status === nothing and cause::UniLMTimeout.
    srv, url, _ = _hm_mute_server()
    try
        chat = Chat(service = GenericOpenAIEndpoint(url, ""), model = "mock", stream = true)
        push!(chat, Message(Val(:system), "s"))
        push!(chat, Message(Val(:user), "u"))
        outcome = _hm_bounded() do
            cfg = UniLM.RequestConfig(request_timeout = 1.0, total_deadline = 2.0,
                max_attempts = 1, stream_idle_timeout = 5.0)
            fetch(chatrequest!(chat; config = cfg))
        end
        ok = try
            outcome[1] === :ok && let res = outcome[2]
                res isa LLMCallError && res.status === nothing && res.cause isa UniLM.UniLMTimeout
            end
        catch
            false
        end
        @test ok
    finally
        close(srv)
    end
end

@testset "stream: mid-stream byte-gap yields typed timeout" begin
    # FIXED contract: after the first byte, a raw-byte gap exceeding
    # stream_idle_timeout (with NO terminal state recorded) closes the stream and
    # surfaces UniLMTimeout(:stream_idle) as the LLMCallError cause.
    srv, url = _hm_first_event_then_hang_server()
    try
        chat = Chat(service = GenericOpenAIEndpoint(url, ""), model = "mock", stream = true)
        push!(chat, Message(Val(:system), "s"))
        push!(chat, Message(Val(:user), "u"))
        outcome = _hm_bounded() do
            cfg = UniLM.RequestConfig(stream_idle_timeout = 1.0, request_timeout = 5.0,
                total_deadline = 10.0, max_attempts = 1)
            fetch(chatrequest!(chat; config = cfg))
        end
        ok = try
            outcome[1] === :ok && let res = outcome[2]
                res isa LLMCallError && res.cause isa UniLM.UniLMTimeout &&
                    res.cause.phase === :stream_idle &&
                    res.cause.limit == 1.0 &&
                    # CONTRACT: elapsed is the BYTE GAP since the last raw chunk,
                    # not whole-call elapsed. Two mechanisms report the same gap:
                    # HTTP 2.x's native read_idle_timeout wins at ~1.0 s, our own
                    # guard (1.x, or if it wins the race) fires within
                    # [limit, limit+period]=[1.0,1.25]. Either way the gap sits
                    # near the 1.0 s limit — bounded well under the 10 s window a
                    # whole-call or total-deadline figure could not satisfy.
                    0.95 <= res.cause.elapsed <= 1.5
            end
        catch
            false
        end
        @test ok
    finally
        # Handler parks mid-stream forever, so a graceful close would quiesce
        # on the still-ACTIVE conn; force-close skips the quiesce loop.
        HTTP.forceclose(srv)
    end
end

@testset "stream: trickle and pings keep the stream alive" begin
    # FIXED contract: a healthy but slow stream (bytes every 0.3 s, a >1 s gap
    # bridged by SSE comment pings) must NOT be killed — idle resets on every raw
    # chunk, so the completed stream is LLMSuccess. A 1-byte trickle keeping a
    # stream alive is an accepted, documented consequence.
    srv, url = _hm_trickle_ping_server()
    try
        chat = Chat(service = GenericOpenAIEndpoint(url, ""), model = "mock", stream = true)
        push!(chat, Message(Val(:system), "s"))
        push!(chat, Message(Val(:user), "u"))
        outcome = _hm_bounded(bound = 45.0) do
            cfg = UniLM.RequestConfig(stream_idle_timeout = 1.0, request_timeout = 5.0,
                total_deadline = 10.0, max_attempts = 1)
            fetch(chatrequest!(chat; config = cfg))
        end
        ok = try
            outcome[1] === :ok && outcome[2] isa LLMSuccess &&
                # Content tightening: with the idle guard now ACTIVE, the sub-limit
                # trickle + ping bridge must survive AND assemble in full — every
                # delta forwarded, nothing truncated by a premature idle close.
                outcome[2].message.content == "Hello world"
        catch
            false
        end
        @test ok
    finally
        close(srv)
    end
end

@testset "stream: in-band overload before first callback retries like its status twin" begin
    # FIXED contract: an in-band `overloaded_error` (the 529-equivalent) arriving
    # BEFORE the first callback is as retryable as its HTTP-status twin — the
    # driver retries inside the stream task, the second attempt succeeds, and the
    # result is LLMSuccess after exactly two connections.
    srv, url, calls = _hm_inband_then_valid_server()
    try
        chat = Chat(service = _HMAnthropicWireMock(url), model = "mock", stream = true)
        push!(chat, Message(Val(:system), "s"))
        push!(chat, Message(Val(:user), "u"))
        outcome = _hm_bounded() do
            cfg = UniLM.RequestConfig(request_timeout = 5.0, total_deadline = 10.0,
                max_attempts = 2, stream_idle_timeout = 5.0)
            fetch(chatrequest!(chat; config = cfg))
        end
        ok = try
            outcome[1] === :ok && outcome[2] isa LLMSuccess && calls[] == 2 &&
                # Content tightening: the RETRIED second attempt is the one that
                # produced the message — its assembled text must be the valid
                # completion's ("ok"), never a remnant of the discarded overload.
                outcome[2].message.content == "ok"
        catch
            false
        end
        @test ok
    finally
        close(srv)
    end
end

@testset "stream: EOF-less terminal stream finalizes success at the idle gap" begin
    # A provider wire with no end-of-stream sentinel (the Gemini shape): the terminal
    # chunk records finish_reason, no sentinel follows, and the connection never closes.
    # The idle guard must finalize SUCCESS (the turn completed), not a timeout.
    server = nothing
    port = 0
    for attempt in 1:3
        tcp = Sockets.listen(Sockets.localhost, 0)
        port = Int(Sockets.getsockname(tcp)[2])
        close(tcp)
        try
            server = HTTP.listen!("127.0.0.1", port; verbose=false) do http::HTTP.Stream
                read(http)                            # drain the request body
                HTTP.setstatus(http, 200)
                HTTP.setheader(http, "Content-Type" => "text/event-stream")
                HTTP.startwrite(http)
                write(http, "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"done.\"},\"finish_reason\":\"stop\"}]}\n\n")
                flush(http)
                sleep(15)   # hold the socket open far past the idle limit: no EOF, no sentinel
            end
            break
        catch
            attempt == 3 && rethrow()
        end
    end
    try
        chat = Chat(service=GenericOpenAIEndpoint("http://127.0.0.1:$port", ""), model="mock", stream=true)
        push!(chat, Message(Val(:system), "s"))
        push!(chat, Message(Val(:user), "u"))
        cfg = RequestConfig(stream_idle_timeout=1.0, request_timeout=10.0,
                            total_deadline=10.0, max_attempts=1)
        task = chatrequest!(chat; config=cfg)
        @test timedwait(() -> istaskdone(task), 45.0) === :ok
        res = fetch(task)
        @test res isa LLMSuccess
        @test res.message.content == "done."
        @test res.message.finish_reason == "stop"
    finally
        isnothing(server) || close(server)
    end
end

@testset "stream: connect refusal surfaces a typed transport failure (task-boundary unwrap)" begin
    # A refused FIRST connection must not hang and must not leak the raw HTTP.jl
    # transport wrapper: the driver peels the `.error`-carrying ConnectError to
    # its root across the task boundary (`_unwrap_exception`) and returns a typed
    # LLMCallError. `max_attempts = 1` keeps it single-shot (no backoff wait);
    # port 1 never listens → connection refused, the dead-endpoint shape the mock
    # server suite also uses.
    chat = Chat(service = GenericOpenAIEndpoint("http://127.0.0.1:1", ""),
                model = "mock", stream = true)
    push!(chat, Message(Val(:system), "s"))
    push!(chat, Message(Val(:user), "u"))
    outcome = _hm_bounded() do
        cfg = UniLM.RequestConfig(connect_timeout = 2.0, request_timeout = 2.0,
            total_deadline = 5.0, max_attempts = 1, stream_idle_timeout = 5.0)
        fetch(chatrequest!(chat; config = cfg))
    end
    ok = try
        outcome[1] === :ok && outcome[2] isa LLMCallError
    catch
        false
    end
    @test ok
end

@testset "embeddings: mute server yields typed timeout" begin
    # Accept-then-silence listener: connections are accepted and never answered.
    srv = Sockets.listen(Sockets.localhost, 0)
    port = Int(Sockets.getsockname(srv)[2])
    accepter = Threads.@spawn begin
        try
            while isopen(srv)
                Sockets.accept(srv)   # hold the connection open, never respond
            end
        catch
        end
    end
    try
        emb = UniLM.Embeddings("x"; service=GenericOpenAIEndpoint("http://127.0.0.1:$port", ""),
                               model="mock")
        cfg = RequestConfig(connect_timeout=1.0, request_timeout=1.0,
                            total_deadline=2.0, max_attempts=1)
        t = Threads.@spawn embeddingrequest!(emb; config=cfg)
        @test timedwait(() -> istaskdone(t), 45.0) === :ok
        r = fetch(t)
        @test r isa EmbeddingCallError
        @test isnothing(r.status)
        @test r.cause isa UniLM.UniLMTimeout
        @test r.cause.phase in (:connect, :request, :deadline)
    finally
        close(srv)
        wait(accepter)
    end
end

@testset "embeddings: interrupt propagates, never laundered" begin
    srv = Sockets.listen(Sockets.localhost, 0)
    port = Int(Sockets.getsockname(srv)[2])
    accepter = Threads.@spawn begin
        try
            while isopen(srv)
                Sockets.accept(srv)
            end
        catch
        end
    end
    try
        emb = UniLM.Embeddings("x"; service=GenericOpenAIEndpoint("http://127.0.0.1:$port", ""),
                               model="mock")
        cfg = RequestConfig(connect_timeout=5.0, request_timeout=5.0,
                            total_deadline=5.0, max_attempts=1)
        t = Threads.@spawn embeddingrequest!(emb; config=cfg)
        sleep(0.3)                                   # let it block inside the exchange
        schedule(t, InterruptException(); error=true)
        @test timedwait(() -> istaskdone(t), 45.0) === :ok
        @test istaskfailed(t)                        # rethrown — not an EmbeddingCallError value
        @test t.exception isa InterruptException
    finally
        close(srv)
        wait(accepter)
    end
end

# ─── Subprocess MCP mock servers (stdio, newline-delimited JSON-RPC) ──────────

# Build a `julia --startup-file=no -e <script>` command running `body`, with a
# unique marker embedded in the argv so a hung child can be reaped by pkill even
# without a Process handle. The child uses the same project so JSON is available.
function _hm_stdio_cmd(body::String)
    marker = "HMSTDIO" * string(rand(UInt64); base = 16)
    proj = dirname(dirname(pathof(UniLM)))
    script = "# $(marker)\n" * body
    cmd = `$(Base.julia_cmd()) --startup-file=no --project=$(proj) -e $(script)`
    (cmd, marker)
end

# Wrapper server: spawns a long-lived, separately-marked grandchild (an npx-style
# wrapper's node child that holds our pipe), then answers only `initialize` and
# goes mute. A naive kill of the direct child orphans the grandchild — only a
# process-GROUP kill reaps it.
function _hm_wrapper_child_cmd()
    gc_marker = "HMGRANDCHILD" * string(rand(UInt64); base = 16)
    proj = dirname(dirname(pathof(UniLM)))
    body = """
    using JSON
    run(pipeline(`sh -c 'while :; do sleep 1; done # $(gc_marker)'`); wait = false)
    while !eof(stdin)
        line = readline(stdin)
        isempty(line) && continue
        msg = try
            JSON.parse(line)
        catch
            nothing
        end
        msg === nothing && continue
        if get(msg, "method", "") == "initialize"
            resp = Dict("jsonrpc" => "2.0", "id" => get(msg, "id", 0), "result" => Dict(
                "protocolVersion" => "2025-11-25", "capabilities" => Dict(),
                "serverInfo" => Dict("name" => "hm-wrapper", "version" => "0.0.0")))
            println(stdout, JSON.json(resp))
            flush(stdout)
        end
    end
    """
    marker = "HMWRAPPER" * string(rand(UInt64); base = 16)
    proj2 = dirname(dirname(pathof(UniLM)))
    script = "# $(marker)\n" * body
    cmd = `$(Base.julia_cmd()) --startup-file=no --project=$(proj2) -e $(script)`
    (cmd, marker, gc_marker)
end

_hm_pkill(marker::AbstractString) = (try
    run(pipeline(`pkill -f $(marker)`; stderr = devnull))
catch
end; nothing)

_hm_alive(marker::AbstractString)::Bool = try
    success(pipeline(`pgrep -f $(marker)`; stderr = devnull))
catch
    false
end

# A server that answers `initialize` (so mcp_connect SUCCEEDS) then swallows
# every later request — the request-phase watchdog must fire on the next call.
const _HM_HANDSHAKE_THEN_MUTE = raw"""
using JSON
while !eof(stdin)
    line = readline(stdin)
    isempty(line) && continue
    msg = try
        JSON.parse(line)
    catch
        nothing
    end
    msg === nothing && continue
    if get(msg, "method", "") == "initialize"
        resp = Dict("jsonrpc" => "2.0", "id" => get(msg, "id", 0), "result" => Dict(
            "protocolVersion" => "2025-11-25", "capabilities" => Dict(),
            "serverInfo" => Dict("name" => "hm-mute-after-init", "version" => "0.0.0")))
        println(stdout, JSON.json(resp))
        flush(stdout)
    end
    # notifications/initialized and every later request: swallowed, never answered.
end
"""

# ─── MCP stdio timeout contracts ─────────────────────────────────────────────

@testset "mcp: mute connect yields MCPTimeoutError naming the override" begin
    # FIXED contract: a server that never answers `initialize` must not hang the
    # connect — the handshake runs under mcp_connect_timeout and throws
    # MCPTimeoutError(:connect) whose message names the per-connect override.
    cmd, marker = _hm_stdio_cmd("while true; sleep(3600); end")
    try
        outcome = _hm_bounded() do
            cfg = UniLM.RequestConfig(mcp_connect_timeout = 0.5)
            try
                mcp_connect(cmd; config = cfg)
                (:connected, nothing)
            catch e
                (:threw, e)
            end
        end
        ok = try
            outcome[1] === :ok && let r = outcome[2]
                r[1] === :threw && r[2] isa UniLM.MCPTimeoutError && r[2].phase === :connect &&
                    occursin("mcp_connect_timeout", sprint(showerror, r[2]))
            end
        catch
            false
        end
        @test ok
    finally
        _hm_pkill(marker)
    end
end

@testset "mcp: stdio request timeout is session-fatal; respawn is opt-in" begin
    # FIXED contract: stdio framing has no id-demux, so a request timeout is
    # SESSION-FATAL — the session becomes :closed and MCPTimeoutError(:request)
    # is thrown. With auto_respawn=false (the default), the NEXT call on the
    # closed session raises an error whose message names auto_respawn.
    cmd, marker = _hm_stdio_cmd(_HM_HANDSHAKE_THEN_MUTE)
    try
        outcome = _hm_bounded() do
            # Generous session bound; the tight 0.5 s bound rides only the first probe (the
            # one meant to time out). The second probe errors at the reuse guard before it
            # touches the transport, so it shares no bound.
            cfg = UniLM.RequestConfig(mcp_request_timeout = 10.0)
            session = mcp_connect(cmd; config = cfg, auto_respawn = false)
            first_err = try
                call_tool(session, "probe", Dict{String,Any}(); timeout = 0.5)
                nothing
            catch e
                e
            end
            st = session.status
            second_err = try
                call_tool(session, "probe", Dict{String,Any}())
                nothing
            catch e
                e
            end
            (first_err, st, second_err)
        end
        ok = try
            outcome[1] === :ok && let (fe, st, se) = outcome[2]
                fe isa UniLM.MCPTimeoutError && fe.phase === :request && st === :closed &&
                    se isa Exception && occursin("auto_respawn", sprint(showerror, se))
            end
        catch
            false
        end
        @test ok
    finally
        _hm_pkill(marker)
    end
end

@testset "mcp: stdio teardown leaves no child processes (wrapper grandchild included)" begin
    # FIXED contract: on a session-fatal timeout the transport is spawned
    # detached (its own process group) and torn down by group-kill, so an
    # npx-style wrapper's grandchild does NOT survive holding our pipe.
    cmd, marker, gc_marker = _hm_wrapper_child_cmd()
    try
        outcome = _hm_bounded() do
            # Generous session bound; the tight 0.5 s bound rides only the probe meant to
            # time out (its session-fatal close drives the teardown under test).
            cfg = UniLM.RequestConfig(mcp_request_timeout = 10.0)
            session = mcp_connect(cmd; config = cfg, auto_respawn = false)
            err = try
                call_tool(session, "probe", Dict{String,Any}(); timeout = 0.5)
                nothing
            catch e
                e
            end
            (err, session.status)
        end
        sleep(0.5)                        # let the OS reap the killed group
        survivor = _hm_alive(gc_marker)
        ok = try
            outcome[1] === :ok && let (err, st) = outcome[2]
                err isa UniLM.MCPTimeoutError && st === :closed && !survivor
            end
        catch
            false
        end
        @test ok
    finally
        _hm_pkill(gc_marker)
        _hm_pkill(marker)
    end
end

@testset "mcp: command-not-found fails immediately, not at the timer" begin
    # FIXED contract: a nonexistent command surfaces the spawn error IMMEDIATELY
    # (not swallowed to ride the connect timer). `dt < 2.0` with a 5 s connect
    # timeout falsifies any implementation that let the failure ride the timer;
    # the error must not be an MCPTimeoutError.
    try
        outcome = _hm_bounded() do
            cfg = UniLM.RequestConfig(mcp_connect_timeout = 5.0)
            badcmd = `this-command-truly-does-not-exist-$(rand(UInt32))`
            t0 = time()
            err = try
                mcp_connect(badcmd; config = cfg)
                nothing
            catch e
                e
            end
            (err, time() - t0)
        end
        ok = try
            outcome[1] === :ok && let (err, dt) = outcome[2]
                # Until the MCP client migration adds the `config` kwarg to
                # mcp_connect, the call raises MethodError (a missing-kwarg
                # dispatch failure) — NOT the real spawn error. A bare
                # `err isa Exception` would let that MethodError satisfy the pin
                # the moment the core config layer lands. Exclude MethodError so
                # this pin cannot pass until the MCP client migration adds the
                # kwarg and the genuine command-not-found process error surfaces.
                err isa Exception && !(err isa MethodError) &&
                    !(err isa UniLM.MCPTimeoutError) && dt < 2.0
            end
        catch
            false
        end
        @test ok
    finally
    end
end

# ─── MCP HTTP transport timeout contract ─────────────────────────────────────

# HTTP MCP server that answers `initialize` (so mcp_connect succeeds) and the
# `notifications/initialized` 202, but never responds to `tools/call` — the
# request-phase watchdog must fire, WITHOUT tearing down the session (HTTP
# request/response correlation is per-POST, so an HTTP timeout is not fatal).
function _hm_mcp_http_mute_server()
    httpserver = nothing
    port = 0
    for attempt in 1:5
        tcp = Sockets.listen(Sockets.localhost, 0)
        port = Int(Sockets.getsockname(tcp)[2])
        close(tcp)
        try
            httpserver = HTTP.serve!("127.0.0.1", port; verbose = false) do req
                req.method == "DELETE" && return HTTP.Response(200, "")
                req.method == "POST" || return HTTP.Response(405, "Method Not Allowed")
                parsed = JSON.parse(String(req.body); dicttype = Dict{String,Any})
                method = get(parsed, "method", "")
                id = get(parsed, "id", nothing)
                if method == "initialize"
                    return HTTP.Response(200, ["Content-Type" => "application/json"],
                        JSON.json(Dict{String,Any}("jsonrpc" => "2.0", "id" => id,
                            "result" => Dict{String,Any}(
                                "protocolVersion" => UniLM._MCP_PROTOCOL_VERSION,
                                "capabilities" => Dict{String,Any}(),
                                "serverInfo" => Dict{String,Any}("name" => "hm-http-mute", "version" => "0.0.0")))))
                end
                isnothing(id) && return HTTP.Response(202, "")   # notifications/initialized
                wait(Condition())                                # tools/call: never respond
                HTTP.Response(200, "")                           # unreachable
            end
            break
        catch
            attempt == 5 && rethrow()
        end
    end
    (httpserver, "http://127.0.0.1:$(port)")
end

@testset "mcp: http request timeout yields MCPTimeoutError; session survives" begin
    # FIXED contract: an HTTP MCP request against a mute peer times out with
    # MCPTimeoutError(:request), but the session SURVIVES (:ready) — unlike
    # stdio, HTTP correlation is per-POST so the timeout is not session-fatal.
    httpserver, url = _hm_mcp_http_mute_server()
    try
        outcome = _hm_bounded() do
            # Generous session bound; the tight 0.5 s bound rides only the probe meant to
            # time out. The session then SURVIVES (:ready) — HTTP timeout is not fatal.
            cfg = UniLM.RequestConfig(mcp_request_timeout = 10.0)
            session = mcp_connect(url; config = cfg)
            err = try
                call_tool(session, "probe", Dict{String,Any}(); timeout = 0.5)
                nothing
            catch e
                e
            end
            (err, session.status)
        end
        ok = try
            outcome[1] === :ok && let (err, st) = outcome[2]
                err isa UniLM.MCPTimeoutError && err.phase === :request && st === :ready
            end
        catch
            false
        end
        @test ok
    finally
        # In the green flow the `tools/call` handler parks at `wait(Condition())`,
        # so a graceful close would spin in its quiesce loop waiting for that
        # still-ACTIVE connection to drain; force-close severs it and returns.
        HTTP.forceclose(httpserver)
    end
end
