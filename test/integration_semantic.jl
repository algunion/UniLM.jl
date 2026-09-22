# ─── Natural-language control-flow integration tests ─────────────────────────
# Requires UNILM_LIVE=1 and TYPESAFE_API_KEY (both, or the suite skips).
# Spend is capped at exactly two calls: one @branch and one nl_dispatch, each
# over a state whose intent is unambiguous, so a correct client is the only
# thing standing between the text and the asserted symbol.

# Defined in a module of its own: these methods must not merge with the unit
# suite's `route`, which `include` loads into the same test module and which
# offers a different set of meanings.
module SemanticLiveRoutes
using UniLM
route(::nl"the customer wants a refund", ticket) = :refund
route(::nl"the customer asks a pricing question", ticket) = :pricing
route(::nl"the customer reports a bug in the app", ticket) = :bug
end

const _SEMANTIC_LIVE_TICKET =
    "My package arrived crushed and the screen is cracked. I want my money back."

if !haskey(ENV, "TYPESAFE_API_KEY") || get(ENV, "UNILM_LIVE", "") != "1"
    @info "Skipping natural-language control-flow integration tests (set UNILM_LIVE=1 and TYPESAFE_API_KEY to run live)"
else

@testset "@branch — live routing" begin
    picked = @branch _SEMANTIC_LIVE_TICKET config=RequestConfig(max_attempts=1, total_deadline=60.0) begin
        "the customer wants a refund"           => :refund
        "the customer asks a pricing question"  => :pricing
        "the customer reports a bug in the app" => :bug
    end
    @test picked === :refund
end

@testset "nl_dispatch — live routing" begin
    @test nl_dispatch(SemanticLiveRoutes.route, _SEMANTIC_LIVE_TICKET;
                      config=RequestConfig(max_attempts=1, total_deadline=60.0)) === :refund
end

end
