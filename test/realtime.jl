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

# An upgrade that completes only after the caller's connect budget ran out. The
# seam holds the open task between the 101 and handler admission until the test
# releases it — after the caller has already received its timeout.
struct RTLateEndpoint <: UniLM.ServiceEndpoint end
const _rt_late_url = Ref("")
const _rt_late_release = Ref(Base.Event())
UniLM._realtime_ws_url(::Type{RTLateEndpoint}) = _rt_late_url[]
UniLM.auth_header(::Type{RTLateEndpoint}) = UniLM.auth_header(RTMuteEndpoint)
UniLM.provider_capabilities(::Type{RTLateEndpoint}) = Set([:realtime])
UniLM._realtime_upgraded(::Type{RTLateEndpoint}) = wait(_rt_late_release[])

@testset "realtime_connect: an upgrade landing after the connect timeout never runs the handler" begin
    # The caller got UniLMTimeout(:connect); a handler starting afterwards would run
    # a session nobody is waiting for. The late socket must be closed unused.
    client_closed = Threads.Atomic{Bool}(false)
    srv = HTTP.WebSockets.listen!("127.0.0.1", 0) do ws
        try
            for _ in ws      # ends when the client closes
            end
        catch e
            e isa InterruptException && rethrow()
        end
        client_closed[] = true
    end
    _rt_late_url[] = "ws://" * HTTP.WebSockets.server_addr(srv)
    _rt_late_release[] = Base.Event()
    handler_ran = Threads.Atomic{Bool}(false)
    try
        t = Threads.@spawn try
            realtime_connect(_ -> (handler_ran[] = true); service=RTLateEndpoint,
                             config=RequestConfig(connect_timeout=1.0, total_deadline=Inf))
        catch e
            e
        end
        @test timedwait(() -> istaskdone(t), 25.0) === :ok
        e = fetch(t)
        @test e isa UniLM.UniLMTimeout && e.phase === :connect
        notify(_rt_late_release[])                 # the upgrade now reaches admission
        @test timedwait(() -> client_closed[], 25.0) === :ok
        @test !handler_ran[]                       # closed unused, never handed over
    finally
        notify(_rt_late_release[])
        close(srv)
    end
end

# A client-secret endpoint whose 200 body the test chooses.
struct RTSecretEndpoint <: UniLM.ServiceEndpoint end
const _rt_secret_base = Ref("")
UniLM._api_base_url(::Type{RTSecretEndpoint}) = _rt_secret_base[]
UniLM.auth_header(::Type{RTSecretEndpoint}) = UniLM.auth_header(RTMuteEndpoint)
UniLM.provider_capabilities(::Type{RTSecretEndpoint}) = Set([:realtime])

@testset "mint_realtime_secret: a 200 without a secret string is a call error" begin
    body = Ref("{}")
    srv = HTTP.serve!(_ -> HTTP.Response(200, ["Content-Type" => "application/json"], body[]),
                      "127.0.0.1", 0; verbose=false)
    _rt_secret_base[] = "http://127.0.0.1:$(HTTP.port(srv))"
    try
        for b in ("{}", """{"value":""}""", """{"value":null}""",
                  """{"client_secret":{"value":""}}""", """{"client_secret":{"value":42}}""")
            body[] = b
            @test mint_realtime_secret(service=RTSecretEndpoint) isa RealtimeCallError
        end
        body[] = """{"client_secret":{"value":"ek_nested"}}"""
        @test mint_realtime_secret(service=RTSecretEndpoint).value == "ek_nested"
        body[] = """{"value":"ek_top"}"""
        @test mint_realtime_secret(service=RTSecretEndpoint).value == "ek_top"
    finally
        close(srv)
    end
end

@testset "realtime_connect: an endpoint other than OpenAI is rejected before any I/O" begin
    # The Realtime socket lives on api.openai.com; opening it for another endpoint
    # would send that endpoint's credentials there.
    @test_throws ArgumentError UniLM._realtime_url(SeamProbe, "gpt-realtime-2")
    @test_throws ArgumentError realtime_connect(_ -> error("must never run"); service=SeamProbe)
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
    #
    # Budget: a loopback upgrade costs tens of milliseconds, but an instrumented
    # shared runner adds scheduler/delivery stalls of ~2 s before the handshake is
    # observed, so the connect bound is 3.0 s. The handler then
    # stays quiet for 8.0 s — a timer that outlived the upgrade fires by
    # 3.0 s + that same ~2 s stall = 5.0 s, well inside the quiet window, so the
    # gap between "still alive" and "would have been killed" stays wide.
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
                                            config=RequestConfig(connect_timeout=3.0,
                                                                 stream_idle_timeout=Inf,
                                                                 total_deadline=Inf)) do sess
            sleep(8.0)                       # idle well past the connect budget
            realtime_send(sess, session_update(Dict("voice" => "alloy")))
            got[] = realtime_receive(sess)
        end
        @test timedwait(() -> istaskdone(t), 45.0) === :ok
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
    # Asserted on the built string: no live upgrade is needed to pin the target.
    _rt_live_url[] = "ws://127.0.0.1:1/v1/realtime"
    @test UniLM._realtime_url(RTLiveEndpoint, "gpt-realtime-2") ==
          "ws://127.0.0.1:1/v1/realtime?model=gpt-realtime-2"   # golden: byte-identical
    @test UniLM._realtime_url(RTLiveEndpoint, "a b/../c?x=1#f") ==
          "ws://127.0.0.1:1/v1/realtime?model=a%20b%2F..%2Fc%3Fx%3D1%23f"
    @test UniLM._realtime_url(OPENAIServiceEndpoint, "gpt-realtime-2") ==
          UniLM.REALTIME_WS_URL * "?model=gpt-realtime-2"
end
