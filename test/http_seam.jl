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

@testset "translation: Inf becomes native-off per phase per major" begin
    # 2.x branch: Real seconds; 0 disables; Inf must never reach the library
    # (it rejects non-finite), and connect must be explicit (nothing => 30 s default).
    k2 = UniLM._native_timeout_kwargs(RequestConfig(connect_timeout=Inf, request_timeout=Inf), Inf; major2=true)
    @test k2.connect_timeout === 0.0
    @test k2.request_timeout === 0.0
    k2b = UniLM._native_timeout_kwargs(RequestConfig(connect_timeout=2.5), 0.4; major2=true)
    @test k2b.connect_timeout === 2.5
    @test k2b.request_timeout === 0.4
    s2 = UniLM._native_stream_kwargs(RequestConfig(stream_idle_timeout=Inf); major2=true)
    @test s2.read_idle_timeout === 0.0
    s2b = UniLM._native_stream_kwargs(RequestConfig(stream_idle_timeout=7.0); major2=true)
    @test s2b.connect_timeout === 10.0
    @test s2b.read_idle_timeout === 7.0
    # 1.x branch: Int seconds; 0 disables; fractional bounds round UP (the
    # native path may be looser than the watchdog, never tighter).
    k1 = UniLM._native_timeout_kwargs(RequestConfig(connect_timeout=Inf, request_timeout=Inf), Inf; major2=false)
    @test k1.connect_timeout === 0
    @test k1.readtimeout === 0
    k1b = UniLM._native_timeout_kwargs(RequestConfig(connect_timeout=2.5), 0.4; major2=false)
    @test k1b.connect_timeout === 3
    @test k1b.readtimeout === 1
    s1 = UniLM._native_stream_kwargs(RequestConfig(); major2=false)
    @test !haskey(s1, :readtimeout)   # streams never get the 1.x whole-exchange bound
    @test s1.connect_timeout === 10
end

@testset "native timeout exceptions map to UniLMTimeout with phase attribution" begin
    cfg = RequestConfig(connect_timeout=1.0)
    t0 = time_ns()
    if UniLM._HTTP_MAJOR2
        e_conn = HTTP.TimeoutError("connect", Int64(1_000_000_000), Int64(0))
        e_tls  = HTTP.TimeoutError("tls_handshake", Int64(1_000_000_000), Int64(0))
        e_req  = HTTP.TimeoutError("request", Int64(1_000_000_000), Int64(0))
        e_idle = HTTP.TimeoutError("read_idle", Int64(1_000_000_000), Int64(0))
        @test UniLM._map_native_timeout(e_conn, cfg, 5.0, t0).phase === :connect
        @test UniLM._map_native_timeout(e_tls, cfg, 5.0, t0).phase === :connect
        @test UniLM._map_native_timeout(e_req, cfg, 5.0, t0).phase === :request
        @test UniLM._map_native_timeout(e_idle, cfg, 5.0, t0).phase === :request
        # nested inside a cause-carrying wrapper
        wrapped = HTTP.ConnectError("127.0.0.1:9", e_conn)
        @test UniLM._map_native_timeout(wrapped, cfg, 5.0, t0).phase === :connect
    else
        e_read = HTTP.TimeoutError(5)
        @test UniLM._map_native_timeout(e_read, cfg, 5.0, t0).phase === :request
        e_conn = HTTP.ConnectError("http://127.0.0.1:9",
                                   HTTP.Connections.ConnectTimeout("127.0.0.1", 9))
        @test UniLM._map_native_timeout(e_conn, cfg, 5.0, t0).phase === :connect
    end
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
