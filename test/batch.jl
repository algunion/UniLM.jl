@testset "Batch API — config seam wiring" begin
    @test _seam_timeout(create_batch("file_x", "/v1/chat/completions"; service=SeamProbe, config=_TINY_DEADLINE), UniLM.BatchCallError)
    @test _seam_timeout(retrieve_batch("batch_x"; service=SeamProbe, config=_TINY_DEADLINE), UniLM.BatchCallError)
    @test _seam_timeout(cancel_batch("batch_x"; service=SeamProbe, config=_TINY_DEADLINE), UniLM.BatchCallError)
    @test _seam_timeout(list_batches(service=SeamProbe, config=_TINY_DEADLINE), UniLM.BatchCallError)
    @test _seam_timeout(poll_batch("batch_x"; interval=0.01, timeout=0.05, service=SeamProbe, config=_TINY_DEADLINE), UniLM.BatchCallError)
end

_batch_json(status::String) = JSON.json(Dict("id" => "batch_1", "status" => status))

@testset "poll_batch: a wall-clock deadline that rides out transient failures" begin
    @test_throws ArgumentError poll_batch("batch_1"; interval=0, service=SeamProbe)
    @test_throws ArgumentError poll_batch("batch_1"; interval=-1, service=SeamProbe)
    @test_throws ArgumentError poll_batch("batch_1"; interval=NaN, service=SeamProbe)
    @test_throws ArgumentError poll_batch("batch_1"; timeout=0, service=SeamProbe)

    # A 503 on GET #2 is a hiccup, not the answer.
    script = n -> n == 2 ? _json(503, "{}") : _json(200, _batch_json(n == 1 ? "in_progress" : "completed"))
    r, seen = _with_scripted((n, _) -> script(n)) do
        poll_batch("batch_1"; interval=0.01, timeout=30.0, service=URLProbe)
    end
    @test r isa BatchSuccess && r.response.status == "completed"
    @test length(seen) == 3

    # So is one GET that outlives its request_timeout (its reply is held until the poll ends).
    release = Base.Event()
    held = n -> (n == 1 && wait(release); _json(200, _batch_json("completed")))
    r2, seen2 = _with_scripted((n, _) -> held(n)) do
        try
            poll_batch("batch_1"; interval=0.01, timeout=30.0, service=URLProbe,
                       config=UniLM.RequestConfig(request_timeout=1.0))
        finally
            notify(release)
        end
    end
    @test r2 isa BatchSuccess && r2.response.status == "completed"
    @test length(seen2) == 2

    # A non-transient failure ends the poll at once.
    r3, seen3 = _with_scripted((_, _) -> _json(404, "{}")) do
        poll_batch("batch_1"; interval=0.01, timeout=30.0, service=URLProbe)
    end
    @test r3 isa BatchFailure && r3.status == 404 && length(seen3) == 1
end

@testset "poll_batch: the timeout bounds wall-clock time, not an iteration count" begin
    # Each GET takes 0.3 s. Bounded by time, the call returns within timeout + one GET +
    # interval (plus a 2 s runner-stall budget); the old iteration bound,
    # ceil(timeout/interval) = 50 GETs, took ~15 s.
    slow = (_, _) -> (sleep(0.3); _json(200, _batch_json("in_progress")))
    r, _ = _with_scripted(slow) do
        poll_batch("batch_1"; interval=0.01, timeout=0.5, service=URLProbe)   # compile the path
        get_s = @elapsed retrieve_batch("batch_1"; service=URLProbe)          # one GET, as seen here
        elapsed = @elapsed (out = poll_batch("batch_1"; interval=0.01, timeout=0.5, service=URLProbe))
        @test elapsed <= 0.5 + get_s + 0.01 + 2.0
        out
    end
    @test r isa BatchCallError
    @test r.cause isa UniLMTimeout && r.cause.phase === :deadline && r.cause.limit == 0.5
    @test isnothing(r.status)                        # a timeout carries no HTTP status
    @test r.last_observed isa BatchObject && r.last_observed.status == "in_progress"
    @test occursin("in_progress", r.error)
end

@testset "poll_batch: a cancel ends the poll at once — mid-pause, and before any GET" begin
    # A non-terminal batch keeps the poll pausing; without a wakeup on the cancel it
    # would sleep out its 30 s interval.
    hits = Threads.Atomic{Int}(0)
    busy = (_, _) -> (Threads.atomic_add!(hits, 1); _json(200, _batch_json("in_progress")))
    for via in (:keyword, :scope)
        tok = CancelToken()
        before = hits[]
        (r, after_cancel), _ = _with_scripted(busy) do
            poll() = poll_batch("batch_1"; interval=30.0, timeout=120.0, service=URLProbe,
                                cancel=(via === :keyword ? tok : nothing))
            t = Threads.@spawn ((via === :scope ? with_cancel(poll, tok) : poll()), time_ns())
            @test timedwait(() -> hits[] > before, 25.0) === :ok   # first GET answered: pausing
            sleep(0.2)
            cancelled_at = time_ns()
            cancel!(tok)
            @test timedwait(() -> istaskdone(t), 25.0) === :ok
            res, done_at = fetch(t)
            res, (done_at - cancelled_at) / 1e9
        end
        @test after_cancel < 5.0                                   # the 30 s pause is not slept out
        @test r isa BatchCallError && r.cause isa UniLMCancelled && r.cause.source === :token
        @test isnothing(r.status)
        @test r.last_observed isa BatchObject && r.last_observed.status == "in_progress"
        @test occursin("cancelled", r.error) && occursin("in_progress", r.error)
    end
    # A token cancelled up front: the typed result, and no request sent.
    r, seen = _with_scripted(busy) do
        poll_batch("batch_1"; interval=30.0, timeout=120.0, service=URLProbe, cancel=cancel!(CancelToken()))
    end
    @test r isa BatchCallError && r.cause isa UniLMCancelled && isnothing(r.last_observed)
    @test isempty(seen)
end

@testset "Batch API — a failure keeps the request id the service sent" begin
    r = _answered(() -> retrieve_batch("batch_x"; service=URLProbe), 404; headers=["x-request-id" => "req_batch"])
    @test r isa UniLM.BatchFailure && r.request_id == "req_batch"
end

@testset "Batch API — a separator-bearing id stays one path segment" begin
    t = _recorded_targets() do
        retrieve_batch(_HOSTILE_ID; service=URLProbe)
        cancel_batch(_HOSTILE_ID; service=URLProbe)
        list_batches(; limit=2, after=_HOSTILE_ID, service=URLProbe)
    end
    @test t == ["/v1/batches/$_HOSTILE_ENC",
                "/v1/batches/$_HOSTILE_ENC/cancel",
                "/v1/batches?limit=2&after=$_HOSTILE_ENC"]
    g = _recorded_targets() do
        cancel_batch("batch_abc123"; service=URLProbe)
        list_batches(; after="batch_abc123", service=URLProbe)
    end
    @test g == ["/v1/batches/batch_abc123/cancel", "/v1/batches?after=batch_abc123"]
end
