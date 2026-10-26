# ─── OpenAI Decisions API Integration Tests ──────────────────────────────────
# Requires UNILM_LIVE=1 and OPENAI_API_KEY (both, or the suite skips). Spend is capped
# at exactly ten calls, nine of a few hundred input tokens and one of about two thousand,
# at $0.10 per 1M input tokens, the latter's cache write at the 12-hour write rate, $0.25
# per 1M (about $0.0004 for that call): a fraction of a cent. Every state is unambiguous, so a correct client is the
# only thing standing between the input and the asserted answer.

# Defined in a module of its own, as the System One suite's live routes are, so these
# methods never merge with a unit suite's.
module DecisionsLiveRoutes
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

const _DECISIONS_LIVE_TICKET =
    "My package arrived crushed and the screen is cracked. I want my money back."
# One attempt per call: a retry would make the call count, and the spend, open-ended.
const _DECISIONS_LIVE_CFG = RequestConfig(max_attempts=1, total_deadline=60.0)
const _DECISIONS_LIVE_PNG = joinpath(pkgdir(UniLM), "docs", "src", "assets", "red_circle.png")

if !haskey(ENV, "OPENAI_API_KEY") || get(ENV, "UNILM_LIVE", "") != "1"
    @info "Skipping OpenAI Decisions integration tests (set UNILM_LIVE=1 and OPENAI_API_KEY to run live)"
else

HTTP.close_idle_connections!()

@testset "OpenAI Decisions — predicate, choice and score in one live call" begin
    r = ask("Export fails in Safari but works in Chrome. I need this fixed before Friday's board meeting.",
            "severity" => score("How severe is this issue?", ["Cosmetic", "Workaround available", "Fully blocked"]),
            "department" => choice("Which department should handle this complaint?", (
                billing   = "Payments, invoices, and refunds.",
                technical = "Problems using the product.",
                shipping  = "Delivery and tracking.",
                other     = "Requests outside these categories.")),
            "deadline" => noul("Does the customer mention a deadline?");
            service=OPENAIServiceEndpoint, config=_DECISIONS_LIVE_CFG)
    @test r isa SystemOneSuccess
    if r isa SystemOneSuccess
        @test r.response.model == "gpt-6-luna" && r.response.service === OPENAIServiceEndpoint
        @test startswith(something(r.response.request_id, ""), "req_")
        sev = r["severity"]
        @test sev isa ScoreAnswer && sev.legend == Dict(0 => "Cosmetic", 1 => "Workaround available", 2 => "Fully blocked")
        @test 0.0 <= sev.score <= 2.0 && 0.0 <= sev.confidence <= 1.0
        @test r["department"] isa ChoiceAnswer && r["department"].choice == "technical"
        @test r["deadline"] isa NoulAnswer && r["deadline"].noul > 0.5
        u = token_usage(r)
        @test u.prompt_tokens > 0 && u.completion_tokens == 0
        @test estimated_cost(r) ≈ u.prompt_tokens * 0.10 / 1_000_000
        @info "Decisions live usage" model = r.response.model request_id = r.response.request_id input = u.prompt_tokens cached = u.cached_tokens cost = estimated_cost(r)
    else
        @info "Decisions call did not succeed" r
    end
end

@testset "OpenAI Decisions — a JSON state is sent as its JSON text" begin
    r = ask((customer = "Ada", message = "Please refund order #55821. The jacket didn't fit and I sent it back on Monday."),
            "intent" => choice("What does the customer primarily want us to do?",
                               ["refund", "cancel_subscription", "technical_issue", "other"]),
            "order" => noul("Does `message` include an order number?");
            service=OPENAIServiceEndpoint, config=_DECISIONS_LIVE_CFG)
    @test r isa SystemOneSuccess
    if r isa SystemOneSuccess
        @test r["intent"].choice == "refund" && r["order"].noul > 0.5
    else
        @info "Decisions call did not succeed" r
    end
end

@testset "OpenAI Decisions — an image state, and text with an image" begin
    png = ImageAttachment(_DECISIONS_LIVE_PNG)
    r = ask(png, "red" => noul("Is the shape in the image red?"),
            "shape" => choice("Which shape does the image show?", ["circle", "square", "triangle"]);
            service=OPENAIServiceEndpoint, config=_DECISIONS_LIVE_CFG)
    @test r isa SystemOneSuccess
    if r isa SystemOneSuccess
        @test r["red"].noul > 0.5 && r["shape"].choice == "circle"
    else
        @info "Decisions call did not succeed" r
    end
    mixed = ask(Any["The customer attached this picture of the sticker they received:", png],
                "colour" => choice("What colour is the sticker?", ["red", "green", "blue"]);
                service=OPENAIServiceEndpoint, config=_DECISIONS_LIVE_CFG)
    @test mixed isa SystemOneSuccess
    if mixed isa SystemOneSuccess
        @test mixed["colour"].choice == "red"
    else
        @info "Decisions call did not succeed" mixed
    end
end

@testset "OpenAI Decisions — prompt_cache_options are sent and accepted" begin
    r = ask("The package arrived with a cracked screen.", "damaged" => noul("Is the product damaged?");
            service=OPENAIServiceEndpoint, config=_DECISIONS_LIVE_CFG,
            prompt_cache_options=PromptCacheOptions(mode="implicit", ttl="12h"))
    @test r isa SystemOneSuccess
    @test r["damaged"] isa NoulAnswer && r["damaged"].noul > 0.5
    u = token_usage(r)
    # Below the 1,024-token caching minimum: nothing is read or written.
    @test u.prompt_tokens > 0 && u.cached_tokens == 0 && u.cache_write_tokens == 0
    @test estimated_cost(r) ≈ u.prompt_tokens * 0.10 / 1_000_000
    @info "Decisions live prompt_cache_options" request_id = r.response.request_id input = u.prompt_tokens cost = estimated_cost(r)
end

@testset "OpenAI Decisions — a 12-hour prompt-cache write is honoured and reported" begin
    # Above the 1,024-token caching minimum, and with a first line no earlier request shares,
    # the request writes its prompt to the cache at the 12-hour lifetime it asks for.
    ticket = "Ticket $(string(rand(UInt64); base=16)).\n" * repeat("The customer reports that the export button " *
        "of the reporting dashboard fails in Safari but works in Chrome, and asks for a fix before the review. ", 60)
    r = ask(ticket, "browser" => noul("Does the problem depend on the browser?");
            service=OPENAIServiceEndpoint, config=_DECISIONS_LIVE_CFG, prompt_cache_options=PromptCacheOptions(ttl="12h"))
    @test r isa SystemOneSuccess
    u = token_usage(r)
    @test u.prompt_tokens >= 1024
    @test u.cache_write_tokens > 0 && u.cache_write_extended_tokens > 0
    # Priced at gpt-6-luna's caching rates per 1M tokens: $0.10 uncached input, $0.01 a cache read,
    # $0.125 a 30-minute write and $0.25 a 12-hour one (the Decisions guide's "Pricing and availability";
    # the pricing page). On 2026-10-22 this call reported 1,849 input tokens, 1,696 of them 12-hour writes.
    fresh = u.prompt_tokens - u.cached_tokens - u.cache_write_tokens
    @test estimated_cost(r) ≈ (fresh * 0.10 + u.cached_tokens * 0.01 +
                               (u.cache_write_tokens - u.cache_write_extended_tokens) * 0.125 +
                               u.cache_write_extended_tokens * 0.25) / 1_000_000
    @info "Decisions live 12-hour cache write" request_id = r.response.request_id input = u.prompt_tokens cached = u.cached_tokens writes = u.cache_write_tokens writes_12h = u.cache_write_extended_tokens raw_usage = get(r.response.raw, "usage", nothing) cost = estimated_cost(r)
end

@testset "OpenAI Decisions — the natural-language verbs on OPENAIServiceEndpoint" begin
    seen = SystemOneSuccess[]
    @test nl_classify(_DECISIONS_LIVE_TICKET, DecisionsLiveRoutes.INTENT; service=OPENAIServiceEndpoint,
                      config=_DECISIONS_LIVE_CFG, on_response = r -> push!(seen, r)) === :refund
    @test only(seen).response.model == "gpt-6-luna"
    @test nl_dispatch(DecisionsLiveRoutes.route, _DECISIONS_LIVE_TICKET; service=OPENAIServiceEndpoint,
                      config=_DECISIONS_LIVE_CFG) === :refund
    keyed = with_system_one(OPENAIServiceEndpoint) do
        nl_dispatch(DecisionsLiveRoutes.keyed, _DECISIONS_LIVE_TICKET; texts=DecisionsLiveRoutes.INTENT,
                    config=_DECISIONS_LIVE_CFG)
    end
    @test keyed === :refund
    picked = @branch _DECISIONS_LIVE_TICKET service=OPENAIServiceEndpoint config=_DECISIONS_LIVE_CFG begin
        "the customer wants a refund"           => :refund
        "the customer asks a pricing question"  => :pricing
        "the customer reports a bug in the app" => :bug
    end
    @test picked === :refund
end

end
