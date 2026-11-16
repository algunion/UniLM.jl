# ─── Gemini Decisions API Integration Tests ──────────────────────────────────
# Requires UNILM_LIVE=1 and GEMINI_API_KEY (both, or the suite skips). Spend is capped at
# exactly sixteen calls: fourteen of 100 to 250 input tokens, one picture of about 400, and
# one refused by the API with HTTP 400; at $0.10 per 1M input tokens that is about $0.0003 a run
# (free on the free tier). Every state is unambiguous, so a correct client is the only thing
# standing between the input and the asserted answer.

# Defined in a module of its own, as the other decision suites' live routes are, so these
# methods never merge with a unit suite's.
module GeminiDecisionsLiveRoutes
using UniLM
route(::nl"the customer wants a refund", ticket) = :refund
route(::nl"the customer asks a pricing question", ticket) = :pricing
route(::nl"the customer reports a bug in the app", ticket) = :bug

const INTENT = (refund  = "the customer wants a refund",
                pricing = "the customer asks a pricing question",
                bug     = "the customer reports a bug in the app")
keyed(::Val{:refund}, ticket)  = :refund
keyed(::Val{:pricing}, ticket) = :pricing
keyed(::Val{:bug}, ticket)     = :bug
end

using Logging   # the missing-price warning a priced call must not log

const _GEMINI_DECISIONS_LIVE_TICKET =
    "My package arrived crushed and the screen is cracked. I want my money back."
# One attempt per call: a retry would make the call count, and the spend, open-ended.
const _GEMINI_DECISIONS_LIVE_CFG = RequestConfig(max_attempts=1, total_deadline=60.0)
const _GEMINI_DECISIONS_LIVE_PNG = joinpath(pkgdir(UniLM), "docs", "src", "assets", "red_circle.png")
const _GEMINI_DECISIONS_RATE = 0.10 / 1_000_000

"The live call `f()`, its usage logged; a call that did not succeed is logged too."
function _gemini_decisions_live(f::Function, what::String)
    r = f()
    if r isa SystemOneSuccess
        u = token_usage(r)
        @info "Gemini Decisions live: $what" model = r.response.model input = u.prompt_tokens cost = estimated_cost(r)
    else
        @info "Gemini Decisions call did not succeed: $what" r
    end
    r
end

if !haskey(ENV, "GEMINI_API_KEY") || get(ENV, "UNILM_LIVE", "") != "1"
    @info "Skipping Gemini Decisions integration tests (set UNILM_LIVE=1 and GEMINI_API_KEY to run live)"
else

HTTP.close_idle_connections!()

@testset "Gemini Decisions — a choice, with the published confidence" begin
    r = _gemini_decisions_live("choice") do
        ask("I was charged twice for my order.",
            "department" => choice("Which department should handle this request?", (
                billing   = "Payments, invoices, and refunds.",
                technical = "Problems using the product.",
                shipping  = "Delivery and tracking.",
                other     = nothing));
            service=GEMINIServiceEndpoint, config=_GEMINI_DECISIONS_LIVE_CFG)
    end
    @test r isa SystemOneSuccess
    if r isa SystemOneSuccess
        a = r["department"]
        @test a isa ChoiceAnswer && a.choice == "billing"
        @test abs(sum(values(a.probabilities)) - 1) <= 1e-9
        @test a.confidence ≈ (4 * a.probabilities["billing"] - 1) / 3            # TypeSafe's formula
        @test a.raw["confidence"] ≈ a.probabilities["billing"]                    # Google's: the top probability
        @test r.response.model == "gemini-decision-1" && r.response.request_id === nothing
        @test r.response.service === GEMINIServiceEndpoint
    end
end

@testset "Gemini Decisions — a score, its legend the levels asked" begin
    levels = ["Cosmetic: appearance only, no lost functionality.",
              "Workaround available: a task fails, but another way works.",
              "Fully blocked: a task fails with no workaround."]
    r = _gemini_decisions_live("score") do
        ask("The export button does nothing at all, in every browser, and there is no other way to export.",
            "severity" => score("How severe is this issue for the user?", levels);
            service=GEMINIServiceEndpoint, config=_GEMINI_DECISIONS_LIVE_CFG)
    end
    @test r isa SystemOneSuccess
    if r isa SystemOneSuccess
        s = r["severity"]
        @test s isa ScoreAnswer && s.legend == Dict(i - 1 => l for (i, l) in enumerate(levels))
        @test s.score > 1.5 && 0.0 <= s.confidence <= 1.0
        @test abs(s.score - sum(l * p for (l, p) in s.probabilities)) <= 1e-9
    end
end

@testset "Gemini Decisions — a noul with both criteria" begin
    r = _gemini_decisions_live("noul") do
        ask("Customer returned the sweater because they preferred green over blue.",
            "is_defect" => noul("Is the customer reporting a product defect?";
                                yes = "The product itself is broken or faulty.",
                                no  = "Preference, sizing, shipping, or pricing complaint.");
            service=GEMINIServiceEndpoint, config=_GEMINI_DECISIONS_LIVE_CFG)
    end
    @test r isa SystemOneSuccess
    if r isa SystemOneSuccess
        a = r["is_defect"]
        @test a isa NoulAnswer && 0.01 <= a.noul < 0.5
        @test a.raw["confidence"] ≈ max(a.noul, 1 - a.noul)
    end
end

@testset "Gemini Decisions — a noul with `yes` alone" begin
    r = _gemini_decisions_live("noul, yes alone") do
        ask(_GEMINI_DECISIONS_LIVE_TICKET, "damaged" => noul("Is the item damaged?"; yes = "It is broken.");
            service=GEMINIServiceEndpoint, config=_GEMINI_DECISIONS_LIVE_CFG)
    end
    @test r isa SystemOneSuccess
    r isa SystemOneSuccess && @test r["damaged"].noul > 0.5
end

@testset "Gemini Decisions — models/gemini-decision-1 is answered, named and priced as the bare id" begin
    r = _gemini_decisions_live("models/ id") do
        ask(_GEMINI_DECISIONS_LIVE_TICKET, "damaged" => noul("Does the customer report a damaged item?");
            service=GEMINIServiceEndpoint, model="models/gemini-decision-1", config=_GEMINI_DECISIONS_LIVE_CFG)
    end
    @test r isa SystemOneSuccess
    if r isa SystemOneSuccess
        @test r.response.model == "gemini-decision-1" && r["damaged"].noul > 0.5
        cost = @test_logs min_level=Logging.Warn estimated_cost(r)
        @test token_usage(r).prompt_tokens > 0 && cost ≈ token_usage(r).prompt_tokens * _GEMINI_DECISIONS_RATE
    end
end

@testset "Gemini Decisions — ask_each: one result per state, in input order" begin
    team = (billing = "Payments, invoices, and refunds.", technical = "Problems using the product.",
            shipping = "Delivery and tracking.", other = nothing)
    rs = ask_each(["I was charged twice for my order.", "The app crashes every time I open my order history."],
                  "department" => choice("Which department should handle this request?", team);
                  service=GEMINIServiceEndpoint, config=_GEMINI_DECISIONS_LIVE_CFG, ntasks=2)
    @test length(rs) == 2 && all(r -> r isa SystemOneSuccess, rs)
    if all(r -> r isa SystemOneSuccess, rs)
        @test [r["department"].choice for r in rs] == ["billing", "technical"]
        @info "Gemini Decisions live: ask_each" input = sum(r -> token_usage(r).prompt_tokens, rs) cost = sum(estimated_cost, rs)
    else
        @info "Gemini Decisions ask_each did not succeed" rs
    end
end

@testset "Gemini Decisions — choice, score and noul in one call, with its usage and cost" begin
    r = _gemini_decisions_live("mixed") do
        ask("Export fails in Safari but works in Chrome. I need this fixed before Friday's board meeting.",
            "severity" => score("How severe is this issue?", ["Cosmetic", "Workaround available", "Fully blocked"]),
            "department" => choice("Which department should handle this complaint?", (
                billing   = "Payments, invoices, and refunds.",
                technical = "Problems using the product.",
                shipping  = "Delivery and tracking.",
                other     = "Requests outside these categories.")),
            "deadline" => noul("Does the customer mention a deadline?");
            service=GEMINIServiceEndpoint, config=_GEMINI_DECISIONS_LIVE_CFG)
    end
    @test r isa SystemOneSuccess
    if r isa SystemOneSuccess
        @test r["department"].choice == "technical" && r["deadline"].noul > 0.5
        @test r["severity"].legend == Dict(0 => "Cosmetic", 1 => "Workaround available", 2 => "Fully blocked")
        u = token_usage(r)
        @test u.prompt_tokens > 0 && u.completion_tokens == 0 && u.total_tokens == u.prompt_tokens
        @test DEFAULT_PRICING["gemini-decision-1"] == UniLM._price(0.10, 0.10, 0.0)
        @test estimated_cost(r) ≈ u.prompt_tokens * _GEMINI_DECISIONS_RATE
        @info "Gemini Decisions live usage" raw_usage = get(r.response.raw, "usage", nothing)
    end
end

@testset "Gemini Decisions — a JSON state is sent as its JSON text" begin
    r = _gemini_decisions_live("JSON state") do
        ask((customer = "Ada", message = "Please refund order #55821. The jacket didn't fit and I sent it back on Monday."),
            "intent" => choice("What does the customer primarily want us to do?",
                               ["refund", "cancel_subscription", "technical_issue", "other"]),
            "order" => noul("Does `message` include an order number?");
            service=GEMINIServiceEndpoint, config=_GEMINI_DECISIONS_LIVE_CFG)
    end
    @test r isa SystemOneSuccess
    r isa SystemOneSuccess && @test r["intent"].choice == "refund" && r["order"].noul > 0.5
end

@testset "Gemini Decisions — text with an image at low resolution" begin
    png = ImageAttachment(_GEMINI_DECISIONS_LIVE_PNG; detail = "low")
    r = _gemini_decisions_live("image") do
        ask(Any["The customer attached this picture of the sticker they received:", png],
            "colour" => choice("What colour is the sticker?", ["red", "green", "blue"]),
            "round" => noul("Is the sticker round?");
            service=GEMINIServiceEndpoint, config=_GEMINI_DECISIONS_LIVE_CFG)
    end
    @test r isa SystemOneSuccess
    if r isa SystemOneSuccess
        @test r["colour"].choice == "red" && r["round"].noul > 0.5
        image = only(m["tokens"] for m in r.response.raw["usage"]["input_tokens_by_modality"] if m["modality"] == "image")
        @test image == 256 && token_usage(r).prompt_tokens > image          # "low": 256 tokens (2026-11-16)
    end
end

@testset "Gemini Decisions — a RawQuestion is sent as written, and one the API refuses comes back typed" begin
    r = _gemini_decisions_live("raw") do
        ask(_GEMINI_DECISIONS_LIVE_TICKET,
            "damaged" => RawQuestion(Dict("type" => "noul", "instructions" => "Does the customer report a damaged item?")),
            "team" => RawQuestion(JSON.Object{String,Any}("type" => "choice", "instructions" => "Which team should handle this?",
                                                          "criteria" => JSON.Object{String,Any}("returns" => "", "sales" => "")));
            service=GEMINIServiceEndpoint, config=_GEMINI_DECISIONS_LIVE_CFG)
    end
    @test r isa SystemOneSuccess
    if r isa SystemOneSuccess
        @test r["damaged"] isa NoulAnswer && r["damaged"].noul > 0.5
        @test r["team"] isa ChoiceAnswer && r["team"].choice == "returns"
    end
    refused = ask(_GEMINI_DECISIONS_LIVE_TICKET, "rank" => RawQuestion(Dict("type" => "rank", "instructions" => "Rank them.",
                                                                             "items" => ["a", "b"]));
                  service=GEMINIServiceEndpoint, config=_GEMINI_DECISIONS_LIVE_CFG)
    @test refused isa SystemOneFailure && refused.status == 400 && refused.error_type == "invalid_request"
    @test contains(refused.message, "'rank'")
end

@testset "Gemini Decisions — the natural-language verbs on GEMINIServiceEndpoint" begin
    seen = SystemOneSuccess[]
    @test nl_classify(_GEMINI_DECISIONS_LIVE_TICKET, GeminiDecisionsLiveRoutes.INTENT; service=GEMINIServiceEndpoint,
                      config=_GEMINI_DECISIONS_LIVE_CFG, on_response = r -> push!(seen, r)) === :refund
    @test only(seen).response.model == "gemini-decision-1"
    @test nl_dispatch(GeminiDecisionsLiveRoutes.route, _GEMINI_DECISIONS_LIVE_TICKET; service=GEMINIServiceEndpoint,
                      config=_GEMINI_DECISIONS_LIVE_CFG) === :refund
    keyed = with_system_one(GEMINIServiceEndpoint) do
        nl_dispatch(GeminiDecisionsLiveRoutes.keyed, _GEMINI_DECISIONS_LIVE_TICKET; texts=GeminiDecisionsLiveRoutes.INTENT,
                    config=_GEMINI_DECISIONS_LIVE_CFG)
    end
    @test keyed === :refund
    picked = @branch _GEMINI_DECISIONS_LIVE_TICKET service=GEMINIServiceEndpoint config=_GEMINI_DECISIONS_LIVE_CFG begin
        "the customer wants a refund"           => :refund
        "the customer asks a pricing question"  => :pricing
        "the customer reports a bug in the app" => :bug
    end
    @test picked === :refund
end

end
