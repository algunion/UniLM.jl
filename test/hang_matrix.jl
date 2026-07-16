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
function _hm_bounded(f; bound::Float64 = 10.0)
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
    @test_broken ok
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
        # so this pin would pass the moment the core config workstream merges —
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
    @test_broken ok
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
        @test_broken ok
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
        outcome = _hm_bounded(bound = 15.0) do
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
        @test_broken ok
    finally
        close(srv)
    end
end

@testset "retry budget: Retry-After beyond remaining deadline fails immediately" begin
    # FIXED contract: when the server's Retry-After (60 s) exceeds the remaining
    # total_deadline (~1 s), the loop NEVER sleeps past the deadline — it returns
    # the real 429 immediately (LLMFailure(status=429)) after a single attempt.
    # `dt < 2.0` falsifies any implementation that honored the 60 s wait.
    srv, url, calls = _hm_status_server(status = 429, retry_after = 60, hang_after = 1)
    try
        chat = Chat(service = GenericOpenAIEndpoint(url, ""), model = "mock")
        push!(chat, Message(Val(:system), "s"))
        push!(chat, Message(Val(:user), "u"))
        outcome = _hm_bounded() do
            cfg = UniLM.RequestConfig(request_timeout = 1.0, total_deadline = 1.0, max_attempts = 5)
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
        @test_broken ok
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
    @test_broken ok
end

@testset "repeated HTTP timeouts leak no tasks and do not poison the pool" begin
    # FIXED contract (observable half): repeated timeouts to a mute peer each
    # produce a typed timeout AND leave the client healthy — a subsequent request
    # to a WORKING server still succeeds (the 2.x connection pool is not poisoned
    # by a timed-out connection). The deeper task/timer-leak counting depends on
    # the timeout-guard internals and lands with the green expansion of this pin.
    srv, url, _ = _hm_mute_server(drain = true)
    healthy_body = JSON.json(Dict{String,Any}(
        "id" => "c", "object" => "chat.completion",
        "choices" => [Dict{String,Any}("index" => 0, "finish_reason" => "stop",
            "message" => Dict{String,Any}("role" => "assistant", "content" => "pong"))],
        "usage" => Dict{String,Any}("prompt_tokens" => 1, "completion_tokens" => 1, "total_tokens" => 2)))
    hsrv, hurl, _ = _hm_healthy_oai_server(healthy_body)
    try
        outcome = _hm_bounded(bound = 15.0) do
            cfg = UniLM.RequestConfig(request_timeout = 0.5, total_deadline = 1.0, max_attempts = 1)
            mute_chat = Chat(service = GenericOpenAIEndpoint(url, ""), model = "mock")
            push!(mute_chat, Message(Val(:system), "s"))
            push!(mute_chat, Message(Val(:user), "u"))
            all_typed = true
            for _ in 1:5
                r = chatrequest!(mute_chat; config = cfg)
                all_typed &= (r isa LLMCallError && r.cause isa UniLM.UniLMTimeout)
            end
            healthy_chat = Chat(service = GenericOpenAIEndpoint(hurl, ""), model = "mock")
            push!(healthy_chat, Message(Val(:system), "s"))
            push!(healthy_chat, Message(Val(:user), "u"))
            hr = chatrequest!(healthy_chat;
                config = UniLM.RequestConfig(request_timeout = 5.0, total_deadline = 10.0, max_attempts = 1))
            (all_typed, hr)
        end
        ok = try
            outcome[1] === :ok && let (all_typed, hr) = outcome[2]
                all_typed && hr isa LLMSuccess
            end
        catch
            false
        end
        @test_broken ok
    finally
        close(srv)
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
        @test_broken ok
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
                    res.cause.phase === :stream_idle
            end
        catch
            false
        end
        @test_broken ok
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
        outcome = _hm_bounded(bound = 15.0) do
            cfg = UniLM.RequestConfig(stream_idle_timeout = 1.0, request_timeout = 5.0,
                total_deadline = 10.0, max_attempts = 1)
            fetch(chatrequest!(chat; config = cfg))
        end
        ok = try
            outcome[1] === :ok && outcome[2] isa LLMSuccess
        catch
            false
        end
        @test_broken ok
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
            outcome[1] === :ok && outcome[2] isa LLMSuccess && calls[] == 2
        catch
            false
        end
        @test_broken ok
    finally
        close(srv)
    end
end
