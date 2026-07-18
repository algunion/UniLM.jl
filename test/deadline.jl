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

@testset "handle mode: close! unblocks a blocked read and yields a typed timeout" begin
    server = Sockets.listen(Sockets.localhost, 0)
    port = Int(Sockets.getsockname(server)[2])
    accepter = Threads.@spawn try
        Sockets.accept(server)   # accept, hold, never write
    catch
        nothing
    end
    sock = Sockets.connect(Sockets.localhost, port)
    try
        t = Threads.@spawn try
            UniLM._with_deadline(() -> read(sock, UInt8), () -> close(sock), 0.5, :request)
        catch e
            e
        end
        @test timedwait(() -> istaskdone(t), 10.0) === :ok   # bounded observation
        e = fetch(t)
        @test e isa UniLM.UniLMTimeout
        @test e.phase === :request
        @test e.limit == 0.5
        @test 0.5 <= e.elapsed < 10.0
    finally
        close(server)
        isopen(sock) && close(sock)
        wait(accepter)
    end
end

@testset "handle mode: fast completion never fires the guard" begin
    closed = Ref(0)
    @test UniLM._with_deadline(() -> :fast, () -> closed[] += 1, 5.0, :request) === :fast
    sleep(0.2)   # a leaked timer would fire close! later; the finally must have closed it
    @test closed[] == 0
end

@testset "handle mode: Inf runs f directly with no guard" begin
    @test UniLM._with_deadline(() -> :direct, () -> error("must never run"), Inf, :request) === :direct
end

@testset "handle mode: a real result arriving after the breach is still returned" begin
    closed = Ref(0)
    res = UniLM._with_deadline(() -> (sleep(0.4); :survived), () -> closed[] += 1, 0.1, :request)
    @test res === :survived   # the result is real; the close side effect stands
    @test closed[] == 1
end

@testset "handle mode: guard resolves exactly once under completion/breach races" begin
    # Completion time straddles the limit so :done and :fired race; the CAS
    # must admit exactly one winner and close! must run at most once.
    for _ in 1:40
        closed = Threads.Atomic{Int}(0)
        limit = 0.02
        outcome = try
            UniLM._with_deadline(() -> (sleep(limit * 2 * rand()); :real),
                                 () -> Threads.atomic_add!(closed, 1), limit, :request)
        catch e
            e
        end
        @test closed[] in (0, 1)
        @test outcome === :real || outcome isa UniLM.UniLMTimeout
        outcome isa UniLM.UniLMTimeout && @test closed[] == 1
    end
end

@testset "handle mode: post-breach errors surface as the timeout, real errors rethrow" begin
    # f fails AFTER the guard fired: the error is the echo of our own close
    e1 = try
        UniLM._with_deadline(() -> (sleep(0.4); error("io closed echo")),
                             () -> nothing, 0.1, :stream_idle)
    catch e
        e
    end
    @test e1 isa UniLM.UniLMTimeout && e1.phase === :stream_idle
    # f fails BEFORE any breach: the original error rethrows untouched
    e2 = try
        UniLM._with_deadline(() -> throw(ArgumentError("real bug")), () -> nothing, 5.0, :request)
    catch e
        e
    end
    @test e2 isa ArgumentError && e2.msg == "real bug"
end

@testset "reported handle mode: fast completion reports fired=false, no close!" begin
    closed = Ref(0)
    r, fired = UniLM._with_deadline_reported(() -> :fast, () -> closed[] += 1, 5.0, :request)
    @test r === :fast
    @test fired == false
    sleep(0.2)            # a leaked timer would fire close! later
    @test closed[] == 0
end

@testset "reported handle mode: a lost completion race reports fired=true from guard state" begin
    # f returns AFTER the timer fires: the result is real, the :armed→:done CAS lost
    # to :fired, so fired=true — read from GUARD STATE, independent of when close!
    # settles. Deterministic: 0.4 s completion strictly past the 0.1 s limit.
    closed = Ref(0)
    r, fired = UniLM._with_deadline_reported(() -> (sleep(0.4); :survived),
                                             () -> closed[] += 1, 0.1, :request)
    @test r === :survived
    @test fired == true
    @test closed[] == 1   # close! ran exactly once
end

@testset "reported handle mode: Inf runs f directly and reports fired=false" begin
    @test UniLM._with_deadline_reported(
        () -> :direct, () -> error("must never run"), Inf, :request) === (:direct, false)
end

@testset "reported handle mode: f-throw conversion matches _with_deadline" begin
    # post-breach echo → typed timeout (identical to _with_deadline)
    e1 = try
        UniLM._with_deadline_reported(() -> (sleep(0.4); error("echo")), () -> nothing, 0.1, :stream_idle)
    catch e; e end
    @test e1 isa UniLM.UniLMTimeout && e1.phase === :stream_idle
    # a clean failure rethrows untouched (no tuple binding)
    e2 = try
        UniLM._with_deadline_reported(() -> throw(ArgumentError("real bug")), () -> nothing, 5.0, :request)
    catch e; e end
    @test e2 isa ArgumentError && e2.msg == "real bug"
    # interrupt wins even after the guard fired
    @test_throws InterruptException UniLM._with_deadline_reported(
        () -> (sleep(0.3); throw(InterruptException())), () -> nothing, 0.05, :request)
end

@testset "task mode: breach delivery terminates a blocked task" begin
    t0 = time_ns()
    t = Threads.@spawn try
        UniLM._with_deadline_task(() -> (sleep(30); :never), 0.3, :request)
    catch e
        e
    end
    @test timedwait(() -> istaskdone(t), 10.0) === :ok
    e = fetch(t)
    @test e isa UniLM.UniLMTimeout
    @test e.phase === :request
    @test e.limit == 0.3
    @test 0.3 <= e.elapsed < 10.0
    @test (time_ns() - t0) / 1e9 < 10.0
end

@testset "task mode: Inf runs f directly; results and failures pass through" begin
    @test UniLM._with_deadline_task(() -> :direct, Inf, :request) === :direct
    @test UniLM._with_deadline_task(() -> 41 + 1, 5.0, :request) == 42
    e = try
        UniLM._with_deadline_task(() -> throw(ArgumentError("boom")), 5.0, :request)
    catch ex
        ex
    end
    @test e isa ArgumentError && e.msg == "boom"   # TaskFailedException unwrapped
end

@testset "interrupts rethrow first, never laundered into timeouts" begin
    @test_throws InterruptException UniLM._with_deadline(
        () -> throw(InterruptException()), () -> nothing, 5.0, :request)
    # even when the guard has ALREADY fired, interrupt still wins
    @test_throws InterruptException UniLM._with_deadline(
        () -> (sleep(0.3); throw(InterruptException())), () -> nothing, 0.05, :request)
    @test_throws InterruptException UniLM._with_deadline_task(
        () -> throw(InterruptException()), 5.0, :request)
end

@testset "idle guard fires within [limit, limit + period] absent touches" begin
    limit = 1.0
    period = min(limit / 4, 5.0)
    t0 = time_ns()
    fired_at = Ref(0.0)
    closed = Threads.Atomic{Int}(0)
    g = UniLM._idle_guard(limit) do
        Threads.atomic_add!(closed, 1)
        fired_at[] = (time_ns() - t0) / 1e9
    end
    @test g !== nothing
    @test !UniLM._idle_fired(g)
    @test timedwait(() -> UniLM._idle_fired(g), 10.0) === :ok
    @test closed[] == 1
    @test fired_at[] >= limit                    # never earlier than the limit
    @test fired_at[] <= limit + period + 1.0     # one check-period quantization + CI slop
    # the recorded breach gap is the quantity compared against the limit —
    # drivers surface it as UniLMTimeout(:stream_idle).elapsed
    @test limit <= UniLM._idle_gap_s(g) <= limit + period + 1.0
    sleep(3 * period)                            # resolved guards never re-fire
    @test closed[] == 1
    @test limit <= UniLM._idle_gap_s(g) <= limit + period + 1.0   # frozen at breach, not still growing
    UniLM._disarm!(g)                            # disarm after fire is a safe no-op
    @test UniLM._idle_fired(g)
end

@testset "idle gap before any breach is the live gap since the last touch" begin
    g = UniLM._idle_guard(() -> nothing, 30.0)
    UniLM._touch!(g)
    sleep(0.3)
    gap = UniLM._idle_gap_s(g)
    @test 0.2 <= gap < 5.0     # live gap tracks the wait since the touch
    UniLM._touch!(g)
    @test UniLM._idle_gap_s(g) < gap   # a touch resets the live gap
    UniLM._disarm!(g)
end

@testset "touches reset the idle clock" begin
    limit = 0.8
    closed = Threads.Atomic{Int}(0)
    g = UniLM._idle_guard(() -> Threads.atomic_add!(closed, 1), limit)
    for _ in 1:6
        sleep(0.3)          # every gap 0.3 < 0.8, but 1.8 s total > limit
        UniLM._touch!(g)
    end
    @test !UniLM._idle_fired(g)
    @test closed[] == 0
    @test timedwait(() -> UniLM._idle_fired(g), 10.0) === :ok   # stop touching → fires
    @test closed[] == 1
    UniLM._disarm!(g)
end

@testset "disarm is idempotent and prevents firing" begin
    closed = Threads.Atomic{Int}(0)
    g = UniLM._idle_guard(() -> Threads.atomic_add!(closed, 1), 0.4)
    @test UniLM._disarm!(g) === nothing
    @test UniLM._disarm!(g) === nothing   # idempotent
    sleep(1.0)
    @test !UniLM._idle_fired(g)
    @test closed[] == 0
end

@testset "Inf yields the nothing-guard; every operation no-ops" begin
    g = UniLM._idle_guard(() -> error("must never run"), Inf)
    @test g === nothing
    @test UniLM._touch!(g) === nothing
    @test UniLM._idle_fired(g) === false
    @test UniLM._idle_gap_s(g) === 0.0
    @test UniLM._disarm!(g) === nothing
end
