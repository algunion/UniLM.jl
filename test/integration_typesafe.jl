# ─── TypeSafe System One Integration Tests ───────────────────────────────────
# Requires UNILM_LIVE=1 and TYPESAFE_API_KEY (both, or the suite skips).
# Spend is capped at exactly two calls: one evaluation and one models listing.

if !haskey(ENV, "TYPESAFE_API_KEY") || get(ENV, "UNILM_LIVE", "") != "1"
    @info "Skipping TypeSafe integration tests (set UNILM_LIVE=1 and TYPESAFE_API_KEY to run live)"
else

@testset "TypeSafe System One — live evaluation" begin
    request = SystemOneRequest(
        "Help! My payouts have been failing for 3 days and nobody has replied to my emails.",
        ["department" => choice("Which team should handle this ticket?", (
             billing   = "Payments, invoicing, payouts, refunds",
             technical = "Bugs, outages, integrations",
             sales     = "Pricing, upgrades, new accounts")),
         "urgency" => score("How urgent is this ticket?",
             ["Can wait", "Needs attention this week", "Needs attention today"]),
         "is_frustrated" => noul("Is the customer frustrated?";
             yes = "The customer expresses frustration or impatience",
             no  = "The customer is neutral or satisfied")];
        model = "jev-latest")

    result = ask(request; config=RequestConfig(max_attempts=1, total_deadline=60.0))
    @test result isa SystemOneSuccess
    if result isa SystemOneSuccess
        r = result.response
        # The alias answers with a versioned id, which is what to pin.
        @test !isempty(r.model)
        @test sort(collect(keys(result))) == ["department", "is_frustrated", "urgency"]

        dept = result["department"]
        @test dept isa ChoiceAnswer
        @test dept.choice in ("billing", "technical", "sales")
        @test length(dept.probabilities) == 3
        @test sum(values(dept.probabilities)) ≈ 1.0 atol = 0.1
        @test 0.0 <= dept.confidence <= 1.0

        urg = result["urgency"]
        @test urg isa ScoreAnswer
        @test haskey(urg.legend, 0)
        @test haskey(urg.probabilities, 0)
        @test 0.0 <= urg.score <= 2.0

        frus = result["is_frustrated"]
        @test frus isa NoulAnswer
        @test 0.0 <= frus.noul <= 1.0

        @test r.usage.prompt_tokens > 0
        @info "TypeSafe usage" model = r.model request_id = r.request_id input = r.usage.prompt_tokens output = r.usage.completion_tokens cost = estimated_cost(result)
    else
        @info "TypeSafe evaluation did not succeed" result
    end
end

@testset "TypeSafe models — live listing" begin
    result = list_models(; config=RequestConfig(max_attempts=1, total_deadline=60.0))
    @test result isa TypeSafeModelsSuccess
    if result isa TypeSafeModelsSuccess
        @test length(result.models) >= 1
        @test all(m -> !isempty(m.name), result.models)
        @info "TypeSafe models" names = [m.name for m in result.models]
    else
        @info "TypeSafe models listing did not succeed" result
    end
end

end
