@testset "Moderations API — config seam wiring" begin
    @test _seam_timeout(moderate("hello"; service=SeamProbe, config=_TINY_DEADLINE), UniLM.ModerationCallError)
end

@testset "moderate: a 200 without one verdict row per input is a call error, never clean" begin
    replied(body; input="hello") = _answered(() -> moderate(input; service=URLProbe), 200; body)
    missing_rows = replied("{}")
    @test missing_rows isa UniLM.ModerationCallError
    @test_throws ArgumentError is_flagged(missing_rows)   # no verdict, as for any failure
    @test replied("""{"results": {"flagged": false}}""") isa UniLM.ModerationCallError
    @test replied("""{"results": []}""") isa UniLM.ModerationCallError
    two = """{"results": [{"flagged": false}, {"flagged": false}]}"""
    @test replied(two) isa UniLM.ModerationCallError                     # two rows, one input
    @test replied(two; input=["a", "b"]) isa ModerationSuccess            # two inputs
    @test replied("""{"results": [{"flagged": false}]}"""; input=["a", "b"]) isa UniLM.ModerationCallError
    # A multi-modal parts array is one input and gets one row.
    parts = [Dict("type" => "text", "text" => "hi"),
             Dict("type" => "image_url", "image_url" => Dict("url" => "https://example.com/a.png"))]
    ok = replied("""{"results": [{"flagged": true}]}"""; input=parts)
    @test ok isa ModerationSuccess && is_flagged(ok)
end

@testset "Moderations API — a failure keeps the request id the service sent" begin
    r = _answered(() -> moderate("hello"; service=URLProbe), 400; headers=["x-request-id" => "req_mod"])
    @test r isa UniLM.ModerationFailure && r.request_id == "req_mod"
end

@testset "is_flagged never reports a failed call as clean" begin
    # The gate is written `is_flagged(moderate(text)) && reject()`. A `false` from a
    # call that never produced a verdict would pass unmoderated content through, so
    # the failure types throw instead of answering the question they cannot answer.
    fail = UniLM.ModerationFailure(response="{\"error\":\"rate limited\"}", status=429)
    callerr = UniLM.ModerationCallError(error="connect refused", status=nothing)
    @test_throws ArgumentError is_flagged(fail)
    @test_throws ArgumentError is_flagged(callerr)
    # The message names the failure so the caller can act on it.
    fail_err = try; is_flagged(fail); catch e; e; end
    call_err = try; is_flagged(callerr); catch e; e; end
    @test contains(sprint(showerror, fail_err), "429")
    @test contains(sprint(showerror, call_err), "connect refused")

    # Success results keep answering exactly as before.
    clean = UniLM.ModerationSuccess(response=UniLM.ModerationResponse(
        results=[UniLM.ModerationResult(flagged=false)], model="omni-moderation-latest"))
    flagged = UniLM.ModerationSuccess(response=UniLM.ModerationResponse(
        results=[UniLM.ModerationResult(flagged=false), UniLM.ModerationResult(flagged=true)],
        model="omni-moderation-latest"))
    @test is_flagged(clean) === false
    @test is_flagged(flagged) === true
    @test is_flagged(UniLM.ModerationResult(flagged=true)) === true
    @test is_flagged(flagged.response) === true
end

@testset "a result row without \"flagged\" fails the call, not the verdict" begin
    # A response missing the verdict field is malformed provider output. Defaulting
    # it to `false` would manufacture a clean verdict out of a broken payload.
    @test_throws ArgumentError UniLM._moderation_verdict(
        Dict{String,Any}("categories" => Dict{String,Any}("hate" => true)))
    @test UniLM._moderation_verdict(Dict{String,Any}("flagged" => true)) === true
    @test UniLM._moderation_verdict(Dict{String,Any}("flagged" => false)) === false
    # `moderate` wraps the throw as a typed call error, so no ModerationSuccess is
    # ever built from a row with no verdict.
    err = try; UniLM._moderation_verdict(Dict{String,Any}()); catch e; e; end
    @test err isa ArgumentError
    @test contains(sprint(showerror, err), "flagged")
end

@testset "moderate retries a transient status within its RequestConfig" begin
    # Moderation is free, and classifying the same input again changes nothing: a transient status or a
    # transport failure is sent again, as a Chat's request is.
    r, seen = _with_scripted((n, _) -> n == 1 ? _json(503, "{}") : _json(200, """{"results":[{"flagged":false}]}""")) do
        moderate("hello"; service=URLProbe, config=RequestConfig(max_attempts=2, total_deadline=60.0))
    end
    @test r isa ModerationSuccess && length(seen) == 2
    @test r isa ModerationSuccess && !is_flagged(r) && seen[1].body == seen[2].body
    # The attempts are the config's: one attempt returns the 503, and a 400 is never sent again.
    once, seen1 = _with_scripted((_, _) -> _json(503, "{}")) do
        moderate("hello"; service=URLProbe, config=RequestConfig(max_attempts=1))
    end
    @test once isa UniLM.ModerationFailure && once.status == 503 && length(seen1) == 1
    bad, seen2 = _with_scripted((_, _) -> _json(400, "{}")) do
        moderate("hello"; service=URLProbe, config=RequestConfig(max_attempts=2, total_deadline=60.0))
    end
    @test bad isa UniLM.ModerationFailure && bad.status == 400 && length(seen2) == 1
end
