# Unit tests for the bounded HTTP seam: translation, mapping, single attempt,
# retry loop, and streaming entry. All servers are localhost; zero-spend.

using Sockets

# Probe an ephemeral port, then serve on it. The close-then-rebind window can
# race under load, so retry with a fresh port a few times.
function _seam_server(handler)
    for _ in 1:5
        tcp = Sockets.listen(Sockets.localhost, 0)
        port = Int(Sockets.getsockname(tcp)[2])
        close(tcp)
        try
            server = HTTP.serve!("127.0.0.1", port; verbose=false) do req
                handler(req)
            end
            return server, "http://127.0.0.1:$port"
        catch e
            e isa Base.IOError || rethrow()
        end
    end
    error("could not bind an ephemeral port for the seam mock server")
end

# Raw TCP server that accepts connections and never writes a byte — the
# mute-peer shape. Tracks accepted connections; holds them open until stopped.
mutable struct MuteServer
    server::Sockets.TCPServer
    port::Int
    accepted::Threads.Atomic{Int}
    task::Task
end

function mute_server()
    server = Sockets.listen(Sockets.localhost, 0)
    port = Int(Sockets.getsockname(server)[2])
    accepted = Threads.Atomic{Int}(0)
    task = Threads.@spawn begin
        socks = Sockets.TCPSocket[]
        try
            while true
                sock = Sockets.accept(server)   # throws IOError once the server closes
                Threads.atomic_add!(accepted, 1)
                push!(socks, sock)              # hold open, never respond
            end
        catch
        finally
            foreach(close, socks)
        end
    end
    return MuteServer(server, port, accepted, task)
end

stop!(m::MuteServer) = (close(m.server); wait(m.task); nothing)

@testset "translation: Inf becomes native-off per phase" begin
    # Real seconds; 0 disables; Inf must never reach the library (it rejects
    # non-finite), and connect must be explicit (nothing => 30 s default).
    k = UniLM._native_timeout_kwargs(RequestConfig(connect_timeout=Inf, request_timeout=Inf), Inf)
    @test k.connect_timeout === 0.0
    @test k.request_timeout === 0.0
    kb = UniLM._native_timeout_kwargs(RequestConfig(connect_timeout=2.5), 0.4)
    @test kb.connect_timeout === 2.5
    @test kb.request_timeout === 0.4
    s = UniLM._native_stream_kwargs(RequestConfig(stream_idle_timeout=Inf))
    @test s.read_idle_timeout === 0.0
    sb = UniLM._native_stream_kwargs(RequestConfig(stream_idle_timeout=7.0))
    @test sb.connect_timeout === 10.0
    @test sb.read_idle_timeout === 7.0
    @test !haskey(sb, :request_timeout)   # no whole-exchange bound on a stream
end

@testset "streaming translation: a disabled idle bound still caps the header wait" begin
    # With the byte-gap bound off, the seam arms NO native non-connect timer, and
    # the request-phase watchdog's `close(io)` cannot reach the connection before
    # response headers exist — a mute peer would stall forever. Cap the header
    # wait natively at the same request bound instead. ONLY in that branch: with
    # a finite idle bound, read_idle_timeout already bounds the header wait
    # (HTTP.jl waits min(response_header_timeout, read_idle_timeout)), and a
    # second native non-connect timer would break the by-elimination phase
    # attribution `_classify_stream_timeout` relies on.
    off = UniLM._native_stream_kwargs(RequestConfig(stream_idle_timeout=Inf), 3.0)
    @test off.read_idle_timeout === 0.0
    @test off.response_header_timeout === 3.0
    on = UniLM._native_stream_kwargs(RequestConfig(stream_idle_timeout=7.0), 3.0)
    @test on.read_idle_timeout === 7.0
    @test !haskey(on, :response_header_timeout)
    # No finite request bound to cap with: nothing to arm.
    @test !haskey(UniLM._native_stream_kwargs(RequestConfig(stream_idle_timeout=Inf), Inf),
                  :response_header_timeout)
end

@testset "streams pin HTTP/1.1; non-streaming requests keep protocol negotiation" begin
    # HTTP/2 multiplexes every concurrent call to a host over one connection with
    # shared flow-control windows, so one stream whose consumer applies
    # backpressure would starve the others. The kwargs the seam hands HTTP.open
    # pin :h1; HTTP.request gets no protocol override (:auto).
    cfg = RequestConfig()
    open_kw = UniLM._open_kwargs(cfg, 5.0)
    @test open_kw.protocol === :h1
    @test open_kw.status_exception === false && open_kw.retry === false
    @test open_kw.read_idle_timeout === cfg.stream_idle_timeout   # native stream bounds ride along
    request_kw = UniLM._request_kwargs(cfg, 5.0)
    @test !haskey(request_kw, :protocol)
    @test request_kw.status_exception === false && request_kw.retry === false
    @test request_kw.request_timeout === 5.0
end

@testset "streaming with the idle bound disabled still fails typed at the request bound" begin
    # End-to-end guarantee behind the translation above: a peer that accepts the
    # connection and never sends response headers must fail as a typed
    # UniLMTimeout(:request) at the configured bound, with the byte-gap guard
    # switched off. The bounded observation below is the falsifier — an unbounded
    # wait fails the test instead of hanging the suite.
    m = mute_server()
    cfg = RequestConfig(connect_timeout=Inf, request_timeout=1.0, total_deadline=Inf,
                        stream_idle_timeout=Inf, max_attempts=1)
    try
        chat = Chat(model="mock", stream=true,
                    service=GenericOpenAIEndpoint("http://127.0.0.1:$(m.port)", ""),
                    messages=[Message(role=UniLM.RoleSystem, content="s"),
                              Message(role=UniLM.RoleUser, content="u")])
        task = chatrequest!(chat; config=cfg)
        # Bounded observation FIRST, and every assertion that reads the task
        # is gated on it: an unbounded wait must fail this test, never hang it.
        bounded = timedwait(() -> istaskdone(task), 15.0) === :ok
        @test bounded
        if bounded
            res = fetch(task)
            @test res isa LLMCallError
            @test res.cause isa UniLM.UniLMTimeout
            @test res.cause.phase === :request
            @test m.accepted[] == 1    # one wire attempt; the bound is not a retry storm
        end
    finally
        stop!(m)
    end
end

@testset "native timeout exceptions map to UniLMTimeout with phase attribution" begin
    cfg = RequestConfig(connect_timeout=1.0)
    t0 = time_ns()
    e_conn = HTTP.TimeoutError("connect", Int64(1_000_000_000), Int64(0))
    e_tls  = HTTP.TimeoutError("tls_handshake", Int64(1_000_000_000), Int64(0))
    e_req  = HTTP.TimeoutError("request", Int64(1_000_000_000), Int64(0))
    e_idle = HTTP.TimeoutError("read_idle", Int64(1_000_000_000), Int64(0))
    @test UniLM._map_native_timeout(e_conn, cfg, 5.0, t0).phase === :connect
    @test UniLM._map_native_timeout(e_tls, cfg, 5.0, t0).phase === :connect
    @test UniLM._map_native_timeout(e_req, cfg, 5.0, t0).phase === :request
    @test UniLM._map_native_timeout(e_idle, cfg, 5.0, t0).phase === :request
    # the limit follows the phase: connect-phase bound vs the attempt's bound
    @test UniLM._map_native_timeout(e_conn, cfg, 5.0, t0).limit == 1.0
    @test UniLM._map_native_timeout(e_req, cfg, 5.0, t0).limit == 5.0
    # nested inside a cause-carrying wrapper
    wrapped = HTTP.ConnectError("127.0.0.1:9", e_conn)
    @test UniLM._map_native_timeout(wrapped, cfg, 5.0, t0).phase === :connect
    # non-timeout transport errors are NOT mapped (they propagate unchanged)
    @test UniLM._map_native_timeout(Base.IOError("boom", 0), cfg, 5.0, t0) === nothing
    @test UniLM._map_native_timeout(ArgumentError("x"), cfg, 5.0, t0) === nothing
end

@testset "retry-loop predicate composes the one transport classifier" begin
    # per-attempt timeouts are retryable by PHASE (this is the composed clause
    # on top of _is_transport_error — see test/deadline.jl for the classifier)
    @test UniLM._retryable_exception(UniLM.UniLMTimeout(:connect, 1.0, 1.0))
    @test UniLM._retryable_exception(UniLM.UniLMTimeout(:request, 1.0, 1.0))
    @test !UniLM._retryable_exception(UniLM.UniLMTimeout(:deadline, 1.0, 1.0))
    @test !UniLM._retryable_exception(UniLM.UniLMTimeout(:stream_idle, 1.0, 1.0))
    @test !UniLM._retryable_exception(InterruptException())
    @test !UniLM._retryable_exception(UniLM._DeadlineBreach(:request, 1.0))
    @test UniLM._retryable_exception(Base.IOError("reset", 0))
    @test UniLM._retryable_exception(EOFError())
    @test UniLM._retryable_exception(Base.SystemError("read", 54))   # syscall-level peer reset
    @test !UniLM._retryable_exception(ArgumentError("nope"))
end

@testset "body passthrough: String, bytes, and multipart Form reach the wire unconverted" begin
    seen_body = Ref{Vector{UInt8}}(UInt8[])
    seen_ct = Ref("")
    server, base = _seam_server(req -> begin
        seen_body[] = Vector{UInt8}(req.body)
        seen_ct[] = HTTP.header(req, "Content-Type", "")
        HTTP.Response(200, [], Vector{UInt8}("{}"))
    end)
    try
        cfg = RequestConfig()
        # String body
        resp = UniLM._http("POST", base * "/", ["Content-Type" => "application/json"], "{\"s\":1}"; cfg)
        @test resp.status == 200
        @test String(copy(seen_body[])) == "{\"s\":1}"
        # Vector{UInt8} body
        resp = UniLM._http("POST", base * "/", ["Content-Type" => "application/json"], Vector{UInt8}("{\"b\":2}"); cfg)
        @test resp.status == 200
        @test String(copy(seen_body[])) == "{\"b\":2}"
        # HTTP.Form body: multipart passes through carrying its own content type
        resp = UniLM._http("POST", base * "/", [], HTTP.Form(Dict("field" => "form-value")); cfg)
        @test resp.status == 200
        @test occursin("multipart/form-data", seen_ct[])
        @test occursin("form-value", String(copy(seen_body[])))
    finally
        close(server)
    end
end

@testset "_http: one attempt, status passthrough, deadline short-circuit" begin
    hits = Threads.Atomic{Int}(0)
    server, base = _seam_server(req -> begin
        Threads.atomic_add!(hits, 1)
        status = req.target == "/missing" ? 404 : 200
        HTTP.Response(status, ["Content-Type" => "application/json"], Vector{UInt8}("{\"ok\":true}"))
    end)
    try
        resp = UniLM._http("POST", base * "/ok",
                           ["Content-Type" => "application/json"], Vector{UInt8}("{}");
                           cfg=RequestConfig())
        @test resp isa HTTP.Response
        @test resp.status == 200
        @test String(resp.body) == "{\"ok\":true}"
        # status_exception=false semantics: non-2xx RETURNS, never throws
        resp404 = UniLM._http("GET", base * "/missing"; cfg=RequestConfig())
        @test resp404.status == 404
        # an exhausted budget short-circuits BEFORE touching the network
        before = hits[]
        e = try
            UniLM._http("GET", base * "/ok"; cfg=RequestConfig(), remaining=0.0)
        catch ex
            ex
        end
        @test e isa UniLM.UniLMTimeout && e.phase === :deadline
        @test hits[] == before
    finally
        close(server)
    end
end

@testset "_http: mute server yields a typed per-attempt timeout at the bound" begin
    m = mute_server()
    try
        cfg = RequestConfig(connect_timeout=Inf, request_timeout=0.5, total_deadline=Inf)
        t = Threads.@spawn try
            UniLM._http("GET", "http://127.0.0.1:$(m.port)/"; cfg)
        catch e
            e
        end
        @test timedwait(() -> istaskdone(t), 15.0) === :ok   # bounded observation
        e = fetch(t)
        @test e isa UniLM.UniLMTimeout
        @test e.phase === :request
        @test e.limit > 0
        @test e.elapsed >= 0.0
        @test m.accepted[] >= 1   # the connection was accepted; the exchange stalled
    finally
        stop!(m)
    end
end

@testset "_http: non-timeout transport failures propagate unchanged" begin
    cfg = RequestConfig(connect_timeout=5.0, request_timeout=5.0)
    e = try
        UniLM._http("GET", "http://127.0.0.1:1/"; cfg)   # refused, not timed out
    catch ex
        ex
    end
    @test e isa Exception
    @test !(e isa UniLM.UniLMTimeout)
end

@testset "retry budget: fail-N stops at max_attempts" begin
    n = Threads.Atomic{Int}(0)
    server, base = _seam_server(req -> begin
        k = Threads.atomic_add!(n, 1)   # returns the OLD value
        HTTP.Response(k < 2 ? 500 : 200, ["Content-Type" => "application/json"], Vector{UInt8}("{}"))
    end)
    try
        # two 500s then 200: succeeds on the third attempt
        cfg = RequestConfig(max_attempts=3, total_deadline=Inf)
        resp = UniLM._http_with_retries(cfg, time_ns(), "GET", base * "/")
        @test resp.status == 200
        @test n[] == 3
        # a budget of 2 returns the SECOND response (a 500) after exactly 2 attempts
        n[] = 0
        cfg2 = RequestConfig(max_attempts=2, total_deadline=Inf)
        resp2 = @test_logs (:warn, r"max_attempts") match_mode=:any UniLM._http_with_retries(cfg2, time_ns(), "GET", base * "/")
        @test resp2.status == 500
        @test n[] == 2
        # max_attempts=1 disables retries entirely
        n[] = 0
        resp3 = UniLM._http_with_retries(RequestConfig(max_attempts=1, total_deadline=Inf), time_ns(), "GET", base * "/")
        @test resp3.status == 500
        @test n[] == 1
    finally
        close(server)
    end
end

@testset "retry budget: Retry-After beyond remaining deadline returns the last real response immediately" begin
    hits = Threads.Atomic{Int}(0)
    server, base = _seam_server(req -> begin
        Threads.atomic_add!(hits, 1)
        HTTP.Response(429, ["Retry-After" => "20", "Content-Type" => "application/json"], Vector{UInt8}("{}"))
    end)
    try
        cfg = RequestConfig(max_attempts=3, total_deadline=5.0)
        perform() = @test_logs (:warn, r"budget") match_mode=:any UniLM._http_with_retries(cfg, time_ns(), "GET", base * "/")
        # Compile the request and warning-capture path before timing retry
        # behavior. First-call JIT time is unrelated to the Retry-After wait.
        perform()
        hits[] = 0
        t_start = time_ns()
        resp = perform()
        elapsed = (time_ns() - t_start) / 1e9
        @test resp.status == 429                       # the last REAL response — no fabricated timeout
        @test hits[] == 1                              # no extra request after refusing the retry
        @test elapsed < 4.0                            # returned now; never slept toward the 20 s
    finally
        close(server)
    end
end

@testset "retry budget: Retry-After is a floor under the jitter, not a replacement for it" begin
    # When the header alone decides the wait, every client that got the same
    # Retry-After sleeps exactly that long and a fanned-out batch retries in
    # lockstep. The header is the earliest retry instant; jitter spreads above it.
    ra1 = HTTP.Response(429, ["Retry-After" => "1"])
    t0 = time_ns()
    draws = [UniLM._retry_pause(RequestConfig(total_deadline=Inf), t0, 1, ra1) for _ in 1:50]
    @test all(d -> first(d) === :sleep, draws)
    @test all(d -> last(d) >= 1.0, draws)
    @test length(unique(last.(draws))) >= 10
    # The spread never pushes a retry whose floor fits past the remaining budget.
    tight = [UniLM._retry_pause(RequestConfig(total_deadline=1.5), time_ns(), 1, ra1) for _ in 1:50]
    @test all(d -> first(d) === :sleep && 1.0 <= last(d) <= 1.5, tight)
    # The floor itself must fit: a Retry-After beyond the remaining budget is :budget.
    action, delay = UniLM._retry_pause(RequestConfig(total_deadline=5.0), time_ns(), 1,
                                       HTTP.Response(429, ["Retry-After" => "20"]))
    @test action === :budget && delay >= 20.0
end

@testset "retry budget: an exhausted total_deadline throws :deadline before any attempt" begin
    hits = Threads.Atomic{Int}(0)
    server, base = _seam_server(req -> begin
        Threads.atomic_add!(hits, 1)
        HTTP.Response(200, [], Vector{UInt8}("{}"))
    end)
    try
        cfg = RequestConfig(total_deadline=0.5, max_attempts=3)
        t0 = time_ns() - UInt64(1_000_000_000)   # entered one second ago
        e = try
            UniLM._http_with_retries(cfg, t0, "GET", base * "/")
        catch ex
            ex
        end
        @test e isa UniLM.UniLMTimeout
        @test e.phase === :deadline
        @test e.limit == 0.5
        @test e.elapsed >= 1.0
        @test hits[] == 0
    finally
        close(server)
    end
end

@testset "retry budget: per-attempt timeouts retry, then surface typed" begin
    m = mute_server()
    try
        cfg = RequestConfig(request_timeout=0.3, max_attempts=2, total_deadline=Inf, connect_timeout=Inf)
        t = Threads.@spawn try
            UniLM._http_with_retries(cfg, time_ns(), "GET", "http://127.0.0.1:$(m.port)/")
        catch e
            e
        end
        @test timedwait(() -> istaskdone(t), 20.0) === :ok
        e = fetch(t)
        @test e isa UniLM.UniLMTimeout
        @test e.phase === :request
        @test m.accepted[] == 2   # both attempts reached the wire
    finally
        stop!(m)
    end
end

@testset "_http_open: streaming seam round-trip passes io through untouched" begin
    server, base = _seam_server(req ->
        HTTP.Response(200, ["Content-Type" => "text/plain"], Vector{UInt8}("streamed-ok")))
    try
        cfg = RequestConfig()
        body_out = Ref("")
        resp = UniLM._http_open("POST", base * "/", ["Content-Type" => "application/json"];
                                cfg, t0=time_ns()) do io
            write(io, "{}")
            HTTP.closewrite(io)
            HTTP.startread(io)
            body_out[] = String(read(io))
        end
        @test resp.status == 200
        @test body_out[] == "streamed-ok"
    finally
        close(server)
    end
end
