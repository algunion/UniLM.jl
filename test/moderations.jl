@testset "Moderations API — config seam wiring" begin
    @test _reached_seam(moderate("hello"; service=SeamProbe, config=_TINY_DEADLINE), UniLM.ModerationCallError)
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
