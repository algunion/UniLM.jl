# Unit tests for cooperative cancellation: the token, its scoping, the
# cancellable backoff sleep, and the HTTP seam's use of both. Local servers only.

using Sockets

# Local HTTP server bound directly to an OS-assigned port (port 0).
function _cx_serve(handler)
    server = HTTP.serve!(handler, "127.0.0.1", 0; verbose=false)
    return (; server, url="http://127.0.0.1:$(HTTP.port(server))/")
end

# Raw TCP peer that accepts connections and never writes a byte.
function _cx_mute()
    listener = Sockets.listen(Sockets.localhost, 0)
    port = Int(Sockets.getsockname(listener)[2])
    socks = Sockets.TCPSocket[]
    acceptor = Threads.@spawn try
        while true
            push!(socks, Sockets.accept(listener))   # hold open, never respond
        end
    catch e
        e isa Base.IOError || rethrow()              # the listener closed
    end
    stop() = (close(listener); wait(acceptor); foreach(close, socks))
    return (; url="http://127.0.0.1:$port/", stop)
end

@testset "UniLMCancelled carries source and elapsed and prints them" begin
    e = UniLMCancelled(:token, 0.25)
    @test e isa Exception && e.source === :token && e.elapsed == 0.25
    @test sprint(showerror, e) == "UniLMCancelled: cancelled by token after 0.25 s"
    @test UniLMCancelled(:callback, 1).elapsed === 1.0
    @test_throws ArgumentError UniLMCancelled(:bogus, 0.0)
end

@testset "cancel token: idempotent; every hook runs exactly once; late registration runs now" begin
    tok = CancelToken()
    @test !iscancelled(tok)
    @test !iscancelled(nothing)
    runs = Threads.Atomic{Int}(0)
    h1 = UniLM._on_cancel(() -> Threads.atomic_add!(runs, 1), tok)
    h2 = UniLM._on_cancel(() -> Threads.atomic_add!(runs, 1), tok)
    @test h1 !== nothing && h2 !== nothing && h1 !== h2
    @test length(tok.hooks) == 2
    @test cancel!(tok) === tok
    @test iscancelled(tok)
    @test runs[] == 2
    cancel!(tok)                                   # later calls are no-ops
    @test runs[] == 2
    @test isempty(tok.hooks)
    # Registering on a cancelled token runs the hook now, in the caller.
    ran_on = Ref{Any}(nothing)
    @test UniLM._on_cancel(() -> (ran_on[] = current_task()), tok) === nothing
    @test ran_on[] === current_task()
    @test isempty(tok.hooks)
    # A throwing hook never propagates and never stops the others.
    tok2 = CancelToken()
    later = Ref(false)
    UniLM._on_cancel(() -> error("hook bug"), tok2)
    UniLM._on_cancel(() -> (later[] = true), tok2)
    @test cancel!(tok2) === tok2
    @test later[]
end

@testset "cancel!: an interrupted hook does not skip the rest; the interrupt surfaces after them" begin
    # The hooks leave the token before they run, so one skipped here never runs: its
    # waiter would sleep out its own timer instead of waking on the cancel.
    tok = CancelToken()
    ran = Int[]
    UniLM._on_cancel(() -> push!(ran, 1), tok)
    UniLM._on_cancel(() -> throw(InterruptException()), tok)
    UniLM._on_cancel(() -> push!(ran, 3), tok)
    UniLM._on_cancel(() -> throw(InterruptException()), tok)
    UniLM._on_cancel(() -> push!(ran, 5), tok)
    @test_throws InterruptException cancel!(tok)
    @test ran == [1, 3, 5]
    @test iscancelled(tok) && isempty(tok.hooks)
    @test cancel!(tok) === tok                     # resolved: later calls are no-ops
end

@testset "cancel token: deregistration is by identity and idempotent" begin
    tok = CancelToken()
    ran = Ref(0)
    f = () -> (ran[] += 1)
    h1 = UniLM._on_cancel(f, tok)
    h2 = UniLM._on_cancel(f, tok)                  # the same function, registered twice
    UniLM._off_cancel(tok, h1)
    UniLM._off_cancel(tok, h1)                     # idempotent: h2 survives
    @test length(tok.hooks) == 1
    cancel!(tok)
    @test ran[] == 1
    @test UniLM._off_cancel(tok, h2) === nothing   # after cancel: nothing left, no-op
    @test UniLM._off_cancel(nothing, h2) === nothing
    @test UniLM._on_cancel(() -> error("must never run"), nothing) === nothing
end

@testset "cancel token: a registration racing cancel! runs exactly once" begin
    runs = map(1:500) do _
        tok = CancelToken()
        n = Threads.Atomic{Int}(0)
        go = Base.Event()
        t = Threads.@spawn (wait(go); cancel!(tok))
        notify(go)
        UniLM._on_cancel(() -> Threads.atomic_add!(n, 1), tok)
        wait(t)
        n[]
    end
    @test all(==(1), runs)
end

@testset "with_cancel: ambient token, innermost scope wins, propagates into spawned tasks" begin
    @test UniLM._current_cancel() === nothing
    outer, inner = CancelToken(), CancelToken()
    @test with_cancel(() -> 42, outer) == 42
    with_cancel(outer) do
        @test UniLM._current_cancel() === outer
        @test fetch(Threads.@spawn UniLM._current_cancel()) === outer
        with_cancel(inner) do
            @test UniLM._current_cancel() === inner
            @test fetch(Threads.@spawn UniLM._current_cancel()) === inner
        end
        @test UniLM._current_cancel() === outer
        @test UniLM._resolve_cancel(inner) === inner   # an explicit token wins
        @test UniLM._resolve_cancel(nothing) === outer
    end
    @test UniLM._current_cancel() === nothing
end

@testset "_cancel_sleep wakes at once on cancel and sleeps out otherwise" begin
    tok = CancelToken()
    t = Threads.@spawn (UniLM._cancel_sleep(tok, 30.0), time_ns())
    sleep(0.3)
    cancelled_at = time_ns()
    cancel!(tok)
    @test timedwait(() -> istaskdone(t), 25.0) === :ok
    woke, woke_at = fetch(t)
    @test woke === true
    @test (woke_at - cancelled_at) / 1e9 < 5.0                # not woken, it sleeps out 30 s
    @test isempty(tok.hooks)
    @test UniLM._cancel_sleep(tok, 30.0) === true             # already cancelled: no wait
    started = time_ns()
    @test UniLM._cancel_sleep(CancelToken(), 0.2) === false   # not cancelled: sleeps out
    @test (time_ns() - started) / 1e9 >= 0.2
    @test UniLM._cancel_sleep(nothing, 0.01) === false
end

@testset "cancellation is never transport-shaped or retryable, bare or nested" begin
    for e in (UniLMCancelled(:token, 0.1), HTTP.CanceledError("cancelled"),
              CompositeException([Base.IOError("reset", 0), HTTP.CanceledError("cancelled")]),
              HTTP.ConnectError("127.0.0.1:9", HTTP.CanceledError("cancelled")))
        @test !UniLM._is_transport_error(e)
        @test !UniLM._retryable_exception(e)
    end
end

@testset "task mode: a cancel resolves the guard and abandons the worker" begin
    tok = CancelToken()
    aborted = Threads.Atomic{Int}(0)
    finished = Threads.Atomic{Int}(0)
    t = Threads.@spawn try
        UniLM._with_deadline_task(30.0, :request; cancel=tok,
                                  on_cancel=() -> Threads.atomic_add!(aborted, 1)) do
            sleep(3.0)
            Threads.atomic_add!(finished, 1)
            :late
        end
    catch e
        e
    end
    sleep(0.2)
    cancel!(tok)
    @test timedwait(() -> istaskdone(t), 25.0) === :ok
    e = fetch(t)
    @test e isa UniLMCancelled && e.source === :token
    @test aborted[] == 1
    @test finished[] == 0                                   # returned before the worker
    @test timedwait(() -> finished[] == 1, 25.0) === :ok    # ...which was never killed
    @test isempty(tok.hooks)
end

@testset "seam: a pre-cancelled token throws before any network I/O" begin
    hits = Threads.Atomic{Int}(0)
    srv = _cx_serve(_ -> (Threads.atomic_add!(hits, 1); HTTP.Response(200, "ok")))
    try
        tok = cancel!(CancelToken())
        cfg = RequestConfig()
        outcomes = Any[
            (try UniLM._http("GET", srv.url; cfg, cancel=tok) catch e; e end),
            (try with_cancel(() -> UniLM._http("GET", srv.url; cfg), tok) catch e; e end),
            (try UniLM._http_with_retries(cfg, time_ns(), "GET", srv.url; cancel=tok) catch e; e end),
            (try UniLM._http_open(io -> nothing, "GET", srv.url, Pair{String,String}[];
                                  cfg, t0=time_ns(), cancel=tok) catch e; e end)]
        @test all(e -> e isa UniLMCancelled && e.source === :token, outcomes)
        @test hits[] == 0
        @test isempty(tok.hooks)
    finally
        close(srv.server)
    end
end

@testset "seam: a cancel mid-request aborts it promptly and is never retried" begin
    # Task mode (finite bound: the watchdog's waiter is released by the guard) and
    # direct mode (no bound: the request context's cancel aborts the exchange).
    for cfg in (RequestConfig(max_attempts=3, total_deadline=Inf),
                RequestConfig(max_attempts=3, total_deadline=Inf, request_timeout=Inf))
        hits = Threads.Atomic{Int}(0)
        srv = _cx_serve(_ -> (Threads.atomic_add!(hits, 1); sleep(20); HTTP.Response(200, "late")))
        try
            tok = CancelToken()
            t = Threads.@spawn (try
                UniLM._http_with_retries(cfg, time_ns(), "GET", srv.url; cancel=tok)
            catch e
                e
            end, time_ns())
            @test timedwait(() -> hits[] == 1, 25.0) === :ok   # the request is in: mid-exchange
            sleep(0.2)
            cancelled_at = time_ns()
            cancel!(tok)
            @test timedwait(() -> istaskdone(t), 25.0) === :ok
            e, done_at = fetch(t)
            @test e isa UniLMCancelled && e.source === :token
            @test (done_at - cancelled_at) / 1e9 < 5.0   # not aborted, it waits out the 20 s reply
            @test hits[] == 1                       # never retried
            @test isempty(tok.hooks)
        finally
            HTTP.forceclose(srv.server)
        end
    end
end

@testset "seam: a cancel during a Retry-After backoff returns at once" begin
    hits = Threads.Atomic{Int}(0)
    srv = _cx_serve(_ -> (Threads.atomic_add!(hits, 1);
                          HTTP.Response(429, ["Retry-After" => "30"], "slow down")))
    try
        tok = CancelToken()
        cfg = RequestConfig(max_attempts=3, total_deadline=Inf)
        t = Threads.@spawn (try
            UniLM._http_with_retries(cfg, time_ns(), "GET", srv.url; cancel=tok)
        catch e
            e
        end, time_ns())
        @test timedwait(() -> hits[] == 1, 25.0) === :ok   # the first 429 is in: backing off
        sleep(0.2)
        cancelled_at = time_ns()
        cancel!(tok)
        @test timedwait(() -> istaskdone(t), 25.0) === :ok
        e, done_at = fetch(t)
        @test e isa UniLMCancelled && e.source === :token
        @test (done_at - cancelled_at) / 1e9 < 5.0   # not woken, it sleeps out the 30 s backoff
        @test hits[] == 1
        @test isempty(tok.hooks)
    finally
        close(srv.server)
    end
end

@testset "_http_open: a cancel unblocks the header wait and a parked body read" begin
    # Whatever the handler's read raises, HTTP.open surfaces the cancelled
    # context as HTTP.CanceledError; _http_open hands it back unchanged. `read!`
    # raises `parked` just before the read the cancel must interrupt.
    function cancelled_read(url, read!)
        tok = CancelToken()
        parked = Threads.Atomic{Bool}(false)
        raised_at = Threads.Atomic{UInt64}(0)
        t = Threads.@spawn try
            UniLM._http_open("POST", url, Pair{String,String}[];
                             cfg=RequestConfig(), t0=time_ns(), cancel=tok) do io
                write(io, "{}")
                HTTP.closewrite(io)
                try
                    read!(io, parked)
                catch
                    raised_at[] = time_ns()
                    rethrow()
                end
            end
        catch e
            e
        end
        reached = timedwait(() -> parked[], 25.0) === :ok
        sleep(0.2)                                  # let the read block
        cancelled_at = time_ns()
        cancel!(tok)
        finished = timedwait(() -> istaskdone(t), 25.0) === :ok
        return (; reached, finished, err=finished ? fetch(t) : nothing,
                  after=(raised_at[] - cancelled_at) / 1e9, tok)
    end

    mute = _cx_mute()   # TCP accepted, response headers never sent
    try
        r = cancelled_read(mute.url, (io, parked) -> (parked[] = true; HTTP.startread(io)))
        @test r.reached && r.finished
        @test r.err isa HTTP.CanceledError
        @test 0.0 <= r.after < 1.0
        @test isempty(r.tok.hooks)
    finally
        mute.stop()
    end

    body_srv = HTTP.listen!("127.0.0.1", 0; verbose=false) do http::HTTP.Stream
        read(http)
        HTTP.setstatus(http, 200)
        HTTP.setheader(http, "Content-Type" => "text/event-stream")
        HTTP.startwrite(http)
        write(http, "data: one\n\n")
        flush(http)
        sleep(30)                                   # park the client's next read
    end
    try
        first_chunk = Ref("")
        r = cancelled_read("http://127.0.0.1:$(HTTP.port(body_srv))/", (io, parked) -> begin
            HTTP.startread(io)
            eof(io) || (first_chunk[] = String(readavailable(io)))
            parked[] = true
            while !eof(io)                          # blocks: the peer holds the body open
                readavailable(io)
            end
        end)
        @test r.reached && r.finished
        @test first_chunk[] == "data: one\n\n"      # the read was parked mid-body
        @test r.err isa HTTP.CanceledError
        @test 0.0 <= r.after < 1.0
        @test isempty(r.tok.hooks)
    finally
        HTTP.forceclose(body_srv)
    end
end

@testset "a token reused across many calls accumulates no hooks" begin
    srv = _cx_serve(_ -> HTTP.Response(200, "ok"))
    try
        tok = CancelToken()
        statuses = [UniLM._http("GET", srv.url; cfg=RequestConfig(), cancel=tok).status for _ in 1:20]
        push!(statuses, UniLM._http("GET", srv.url; cancel=tok,
                                    cfg=RequestConfig(request_timeout=Inf, total_deadline=Inf)).status)
        push!(statuses, UniLM._http_with_retries(RequestConfig(), time_ns(), "GET", srv.url;
                                                 cancel=tok).status)
        push!(statuses, UniLM._http_open(io -> (HTTP.startread(io); read(io)), "GET", srv.url,
                                         Pair{String,String}[]; cfg=RequestConfig(), t0=time_ns(),
                                         cancel=tok).status)
        @test all(==(200), statuses)
        @test isempty(tok.hooks)
        @test !iscancelled(tok)
    finally
        close(srv.server)
    end
end
