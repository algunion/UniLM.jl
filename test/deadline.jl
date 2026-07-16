# Unit tests for typed timeouts, deadline arithmetic, and watchdog guards.

using Sockets

@testset "timeout types carry sane phase/elapsed/limit and print them" begin
    e = UniLM.UniLMTimeout(:request, 1.234, 1.0)
    @test e isa Exception
    @test e.phase === :request && e.elapsed == 1.234 && e.limit == 1.0
    s = sprint(showerror, e)
    @test occursin("request", s) && occursin("1.0", s)
    m = UniLM.MCPTimeoutError(:connect, 2.0, 120.0,
        "initialize handshake exceeded mcp_connect_timeout=120.0; raise it per session via mcp_connect(cmd; config=RequestConfig(current_config(); mcp_connect_timeout=...))")
    @test m isa Exception
    sm = sprint(showerror, m)
    @test occursin("mcp_connect_timeout", sm)   # the message must surface the override
    @test occursin("connect", sm)
    # Default struct show is KEPT (showerror is additive only): downstream
    # interception matches on the type name inside string(e).
    @test occursin("UniLMTimeout", string(e))
    @test occursin("MCPTimeoutError", string(m))
end

@testset "monotonic deadline arithmetic" begin
    t0 = time_ns()
    @test UniLM._elapsed_s(t0) >= 0.0
    @test UniLM._elapsed_s(t0 - UInt64(2_000_000_000)) >= 2.0
    cfg = RequestConfig(total_deadline=10.0)
    @test 0.0 < UniLM._remaining_s(cfg, time_ns()) <= 10.0
    @test UniLM._remaining_s(cfg, time_ns() - UInt64(11_000_000_000)) == 0.0  # floors at zero
    @test UniLM._remaining_s(RequestConfig(total_deadline=Inf), time_ns()) == Inf
end

@testset "exception-chain walker finds nested causes; unwrap strips task layers" begin
    b = UniLM._DeadlineBreach(:request, 1.0)
    t = Threads.@spawn throw(b)
    try
        wait(t)
    catch
    end
    tfe = TaskFailedException(t)
    @test UniLM._find_exception(x -> x isa UniLM._DeadlineBreach, tfe) === b
    comp = CompositeException([ArgumentError("x"), tfe])
    @test UniLM._find_exception(x -> x isa UniLM._DeadlineBreach, comp) === b
    @test UniLM._find_exception(x -> x isa UniLM._DeadlineBreach, ArgumentError("x")) === nothing
    # cause-carrying wrappers (.error on the 1.x major, .cause on the 2.x) are traversed
    wrapped = HTTP.ConnectError("http://127.0.0.1:9", ErrorException("inner"))
    @test UniLM._find_exception(x -> x isa ErrorException, wrapped) isa ErrorException
    @test UniLM._unwrap_task_failure(tfe) === b
    @test UniLM._unwrap_task_failure(ArgumentError("y")) isa ArgumentError
end

@testset "transport-error classifier: IO shapes true, control-flow always false" begin
    major2 = pkgversion(HTTP) >= v"2"
    # connection-level shapes are transport errors
    @test UniLM._is_transport_error(Base.IOError("connection reset", 0))
    @test UniLM._is_transport_error(EOFError())
    @test UniLM._is_transport_error(HTTP.ConnectError("http://127.0.0.1:9", ErrorException("refused")))
    # unwrapped across task/composite layers
    t2 = Threads.@spawn throw(Base.IOError("reset mid-task", 0))
    try
        wait(t2)
    catch
    end
    @test UniLM._is_transport_error(TaskFailedException(t2))
    @test UniLM._is_transport_error(CompositeException([TaskFailedException(t2)]))
    # the ALWAYS-FALSE set — bare and nested
    @test !UniLM._is_transport_error(InterruptException())
    @test !UniLM._is_transport_error(UniLM._DeadlineBreach(:request, 1.0))
    @test !UniLM._is_transport_error(UniLM.UniLMTimeout(:connect, 1.0, 1.0))
    @test !UniLM._is_transport_error(UniLM.UniLMTimeout(:request, 1.0, 1.0))
    @test !UniLM._is_transport_error(UniLM.UniLMTimeout(:deadline, 1.0, 1.0))
    ti = Threads.@spawn throw(InterruptException())
    try
        wait(ti)
    catch
    end
    @test !UniLM._is_transport_error(TaskFailedException(ti))
    # an interrupt buried NEXT TO a transport error still wins: never retried
    @test !UniLM._is_transport_error(CompositeException([Base.IOError("x", 0), InterruptException()]))
    # status-carrying errors are responses, not transport failures
    status_err = major2 ?
        HTTP.StatusError(500, HTTP.Response(500, [], UInt8[])) :
        HTTP.StatusError(500, "GET", "/x", HTTP.Response(500, [], UInt8[]))
    @test !UniLM._is_transport_error(status_err)
    # native timeout errors ride the UniLMTimeout channel via the seam's mapping
    timeout_err = major2 ?
        HTTP.TimeoutError("request", Int64(1_000_000_000), Int64(0)) :
        HTTP.TimeoutError(1)
    @test !UniLM._is_transport_error(timeout_err)
    @test !UniLM._is_transport_error(ArgumentError("not transport"))
end
