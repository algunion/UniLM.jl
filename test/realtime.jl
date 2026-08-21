@testset "Realtime API — config seam wiring (secret minting only)" begin
    @test _reached_seam(mint_realtime_secret(service=SeamProbe, config=_TINY_DEADLINE), UniLM.RealtimeCallError)
end

using Sockets

# Raw TCP listener that accepts connections and never writes a byte — the
# mute-peer shape: TCP connect succeeds, the WebSocket upgrade never answers.
mutable struct RTMuteServer
    server::Sockets.TCPServer
    port::Int
    task::Task
end

function _rt_mute_server()
    server = Sockets.listen(Sockets.localhost, 0)
    port = Int(Sockets.getsockname(server)[2])
    task = Threads.@spawn begin
        socks = Sockets.TCPSocket[]
        try
            while true
                push!(socks, Sockets.accept(server))   # hold open, never respond
            end
        catch
        finally
            foreach(close, socks)
        end
    end
    return RTMuteServer(server, port, task)
end

_rt_stop!(m::RTMuteServer) = (close(m.server); wait(m.task); nothing)

# Endpoints whose WS URL is chosen after the listener binds.
struct RTMuteEndpoint <: UniLM.ServiceEndpoint end
struct RTLiveEndpoint <: UniLM.ServiceEndpoint end
const _rt_mute_url = Ref("")
const _rt_live_url = Ref("")
UniLM._realtime_ws_url(::Type{RTMuteEndpoint}) = _rt_mute_url[]
UniLM._realtime_ws_url(::Type{RTLiveEndpoint}) = _rt_live_url[]
UniLM.auth_header(::Type{RTMuteEndpoint}) =
    ["Authorization" => "Bearer t", "Content-Type" => "application/json"]
UniLM.auth_header(::Type{RTLiveEndpoint}) = UniLM.auth_header(RTMuteEndpoint)
UniLM.provider_capabilities(::Type{RTMuteEndpoint}) = Set([:realtime])
UniLM.provider_capabilities(::Type{RTLiveEndpoint}) = Set([:realtime])

@testset "realtime_connect: a mute peer fails typed inside the connect budget" begin
    # The package's bound-everything guarantee excluded the Realtime socket: the
    # WebSocket open carried no timeout and no RequestConfig, so a peer that
    # accepts TCP and never completes the upgrade blocked the caller forever.
    m = _rt_mute_server()
    _rt_mute_url[] = "ws://127.0.0.1:$(m.port)"
    try
        # Ambient channel (scope): observed under a harness-level watchdog, because
        # an unbounded call would otherwise hang the suite rather than fail it.
        scoped = Threads.@spawn try
            with_request_config(connect_timeout=1.0, total_deadline=Inf) do
                realtime_connect(_ -> nothing; service=RTMuteEndpoint)
            end
        catch e
            e
        end
        @test timedwait(() -> istaskdone(scoped), 20.0) === :ok
        e1 = fetch(scoped)
        @test e1 isa UniLM.UniLMTimeout && e1.phase === :connect

        # Per-call channel (kwarg) wins the same way every other surface resolves it.
        kw = Threads.@spawn try
            realtime_connect(_ -> nothing; service=RTMuteEndpoint,
                             config=RequestConfig(connect_timeout=1.0, total_deadline=Inf))
        catch e
            e
        end
        @test timedwait(() -> istaskdone(kw), 20.0) === :ok
        e2 = fetch(kw)
        @test e2 isa UniLM.UniLMTimeout && e2.phase === :connect
    finally
        _rt_stop!(m)
    end
end

@testset "realtime_receive: a silent peer breaches the idle bound, typed" begin
    # A connected-but-silent server is the other unbounded wait: receive blocked
    # with no byte-gap bound at all.
    port = let s = Sockets.listen(Sockets.localhost, 0)
        p = Int(Sockets.getsockname(s)[2])
        close(s)
        p
    end
    _rt_live_url[] = "ws://127.0.0.1:$port"
    srv = HTTP.WebSockets.listen!("127.0.0.1", port) do ws
        # Complete the upgrade, then never send. Blocking on the client's own
        # frames (rather than sleeping) ends the handler when the client closes.
        try
            for _ in ws
            end
        catch
        end
    end
    try
        t = Threads.@spawn try
            realtime_connect(sess -> realtime_receive(sess); service=RTLiveEndpoint,
                             config=RequestConfig(connect_timeout=10.0, stream_idle_timeout=1.0,
                                                  total_deadline=Inf))
        catch e
            e
        end
        @test timedwait(() -> istaskdone(t), 25.0) === :ok
        e = fetch(t)
        @test e isa UniLM.UniLMTimeout && e.phase === :stream_idle
    finally
        close(srv)
    end
end

@testset "realtime session survives past the connect budget" begin
    # Falsifies the obvious way to get the bound wrong: arming a handshake timer
    # that outlives the upgrade would kill a healthy, deliberately quiet session.
    port = let s = Sockets.listen(Sockets.localhost, 0)
        p = Int(Sockets.getsockname(s)[2])
        close(s)
        p
    end
    _rt_live_url[] = "ws://127.0.0.1:$port"
    srv = HTTP.WebSockets.listen!("127.0.0.1", port) do ws
        for msg in ws
            HTTP.WebSockets.send(ws, msg)   # echo back
        end
    end
    try
        got = Ref{Any}(nothing)
        t = Threads.@spawn realtime_connect(model="gpt-realtime-2", service=RTLiveEndpoint,
                                            config=RequestConfig(connect_timeout=1.0,
                                                                 stream_idle_timeout=Inf,
                                                                 total_deadline=Inf)) do sess
            sleep(2.5)                       # idle well past the connect budget
            realtime_send(sess, session_update(Dict("voice" => "alloy")))
            got[] = realtime_receive(sess)
        end
        @test timedwait(() -> istaskdone(t), 30.0) === :ok
        fetch(t)
        @test got[]["type"] == "session.update"
        @test got[]["session"]["voice"] == "alloy"
    finally
        close(srv)
    end
end

@testset "RealtimeSecretSuccess show redacts the minted credential" begin
    # `.value` IS a live client secret and `.raw` carries it a second time, so the
    # default field dump hands it to anyone who echoes the result at the REPL or
    # logs it. Both must be unprintable while staying reachable programmatically.
    secret = "ek_live_SECRET0123456789abcdef"
    r = UniLM.RealtimeSecretSuccess(value=secret,
        raw=Dict{String,Any}("client_secret" => Dict{String,Any}("value" => secret),
                             "expires_at" => 1234))
    s = sprint(show, r)
    @test !occursin(secret, s)
    @test !occursin("SECRET0123456789", s)
    @test occursin("ek_l…[redacted]", s)
    @test occursin("raw=<2 keys>", s)   # raw is summarized, never dumped
    @test r.value == secret             # the field itself is untouched
    @test r.raw["expires_at"] == 1234
end

@testset "realtime WS URL — the model is a query value, not a URL shaper" begin
    # Asserted on the built string rather than through a listener: the two HTTP
    # majors name the server-side WebSocket's request field differently, so there
    # is no portable way to read the target back off a live upgrade.
    _rt_live_url[] = "ws://127.0.0.1:1/v1/realtime"
    @test UniLM._realtime_url(RTLiveEndpoint, "gpt-realtime-2") ==
          "ws://127.0.0.1:1/v1/realtime?model=gpt-realtime-2"   # golden: byte-identical
    @test UniLM._realtime_url(RTLiveEndpoint, "a b/../c?x=1#f") ==
          "ws://127.0.0.1:1/v1/realtime?model=a%20b%2F..%2Fc%3Fx%3D1%23f"
    @test UniLM._realtime_url(OPENAIServiceEndpoint, "gpt-realtime-2") ==
          UniLM.REALTIME_WS_URL * "?model=gpt-realtime-2"
end
