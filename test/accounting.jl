import InteractiveUtils

@testset "TokenUsage" begin
    @testset "defaults" begin
        u = TokenUsage()
        @test u.prompt_tokens == 0
        @test u.completion_tokens == 0
        @test u.total_tokens == 0
    end

    @testset "custom values" begin
        u = TokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150)
        @test u.prompt_tokens == 100
        @test u.completion_tokens == 50
        @test u.total_tokens == 150
    end

    @testset "detail fields (cached/reasoning)" begin
        u = TokenUsage()
        @test u.cached_tokens == 0
        @test u.reasoning_tokens == 0
        # Chat-style usage dict → _parse_usage pulls *_tokens_details subtotals
        chat_data = Dict{String,Any}("usage" => Dict{String,Any}(
            "prompt_tokens" => 10, "completion_tokens" => 5, "total_tokens" => 15,
            "prompt_tokens_details" => Dict{String,Any}("cached_tokens" => 7),
            "completion_tokens_details" => Dict{String,Any}("reasoning_tokens" => 42)))
        cu = UniLM._parse_usage(chat_data)
        @test cu.cached_tokens == 7
        @test cu.reasoning_tokens == 42
        # Responses-style usage dict → token_usage(::ResponseSuccess) reads input/output details
        ro = UniLM.ResponseObject(id="r", status="completed", model="gpt-5.5", output=Any[],
            usage=Dict{String,Any}("input_tokens" => 3, "output_tokens" => 9, "total_tokens" => 12,
                "input_tokens_details" => Dict{String,Any}("cached_tokens" => 2),
                "output_tokens_details" => Dict{String,Any}("reasoning_tokens" => 4)),
            raw=Dict{String,Any}())
        ru = token_usage(ResponseSuccess(response=ro))
        @test ru.cached_tokens == 2
        @test ru.reasoning_tokens == 4
        @test ru.prompt_tokens == 3
        # JSON null subtotals must coalesce to 0, not crash (regression)
        nulled = Dict{String,Any}("usage" => Dict{String,Any}("prompt_tokens" => nothing,
            "completion_tokens" => 5, "total_tokens" => nothing,
            "prompt_tokens_details" => Dict{String,Any}("cached_tokens" => nothing)))
        nu = UniLM._parse_usage(nulled)
        @test nu.prompt_tokens == 0 && nu.total_tokens == 0 && nu.cached_tokens == 0
        @test nu.completion_tokens == 5
    end
end

@testset "token_usage" begin
    chat = Chat()

    @testset "LLMSuccess with usage" begin
        m = Message(role=UniLM.RoleAssistant, content="hi")
        u = TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        s = LLMSuccess(message=m, self=chat, usage=u)
        tu = token_usage(s)
        @test tu.prompt_tokens == 10
        @test tu.completion_tokens == 5
        @test tu.total_tokens == 15
    end

    @testset "LLMSuccess without usage" begin
        m = Message(role=UniLM.RoleAssistant, content="hi")
        s = LLMSuccess(message=m, self=chat)
        tu = token_usage(s)
        @test tu.prompt_tokens == 0
    end

    @testset "failure types return zero" begin
        @test token_usage(LLMFailure(response="err", status=500, self=chat)) == TokenUsage()
        @test token_usage(LLMCallError(error="err", self=chat)) == TokenUsage()
    end

    @testset "ResponseSuccess with usage" begin
        ro = UniLM.ResponseObject(
            id="resp_1", status="completed", model="gpt-5.2",
            output=Any[],
            usage=Dict{String,Any}("input_tokens" => 20, "output_tokens" => 10, "total_tokens" => 30),
            raw=Dict{String,Any}()
        )
        rs = UniLM.ResponseSuccess(response=ro)
        tu = token_usage(rs)
        @test tu.prompt_tokens == 20
        @test tu.completion_tokens == 10
        @test tu.total_tokens == 30
    end

    @testset "ResponseSuccess without usage" begin
        ro = UniLM.ResponseObject(
            id="resp_1", status="completed", model="gpt-5.2",
            output=Any[], usage=nothing, raw=Dict{String,Any}()
        )
        rs = UniLM.ResponseSuccess(response=ro)
        tu = token_usage(rs)
        @test tu.prompt_tokens == 0
    end

    @testset "image types return zero" begin
        ir = UniLM.ImageResponse(created=1, data=UniLM.ImageObject[], raw=Dict{String,Any}())
        @test token_usage(ImageSuccess(response=ir)) == TokenUsage()
        @test token_usage(ImageFailure(response="err", status=400)) == TokenUsage()
        @test token_usage(ImageCallError(error="err")) == TokenUsage()
    end
end

@testset "estimated_cost" begin
    chat = Chat(model="gpt-5.2")
    m = Message(role=UniLM.RoleAssistant, content="hi")
    u = TokenUsage(prompt_tokens=1_000_000, completion_tokens=1_000_000, total_tokens=2_000_000)
    s = LLMSuccess(message=m, self=chat, usage=u)

    cost = estimated_cost(s)
    # gpt-5.2 (live-verified 2026-06-21): input=1.75/1M, output=14.0/1M => 1.75 + 14.0 = 15.75
    @test cost ≈ 15.75

    @testset "explicit model override" begin
        cost2 = estimated_cost(s; model="gpt-5.4-mini")
        # gpt-5.4-mini: input=0.75/1M, output=4.5/1M => 0.75 + 4.5 = 5.25
        @test cost2 ≈ 5.25
    end

    @testset "unknown model returns 0, with a warning" begin
        chat2 = Chat(model="unknown-model")
        s2 = LLMSuccess(message=m, self=chat2, usage=u)
        @test (@test_logs (:warn, "no price row for this model; its estimated cost is 0.0") estimated_cost(s2)) == 0.0
    end

    @testset "failure returns 0" begin
        @test estimated_cost(LLMFailure(response="err", status=500, self=chat)) == 0.0
    end

    @testset "versioned Jev id without a row is priced at jev-latest" begin
        # Assumption: TypeSafe's Models page lists Jev 1.14 and Jev 1.13 at one price ($42 per Btok of
        # input; output free); a later version is assumed to keep it until the page says otherwise.
        # Responses report the versioned id that answered, e.g. a release newer than the table.
        ju = TokenUsage(prompt_tokens=1_000_000, completion_tokens=500, total_tokens=1_000_500)
        s1(model) = SystemOneSuccess(UniLM.SystemOneResponse(model, Dict{String,UniLM.SystemOneAnswer}(),
            ju, nothing, Dict{String,Any}()))
        @test !haskey(DEFAULT_PRICING, "jev-1.15.0")
        @test estimated_cost(s1("jev-1.15.0")) ≈ 0.042
        @test estimated_cost(s1("jev-1.15.0")) == estimated_cost(s1("jev-latest")) == estimated_cost(s1("jev-1.13.0"))
        @test estimated_cost(s1("jev-latest"); model="jev-2.0.10") ≈ 0.042
        @test estimated_cost(LLMSuccess(message=m, self=Chat(model="jev-2.0.10"), usage=ju)) ≈ 0.042
        unpriced = (:warn, "no price row for this model; its estimated cost is 0.0")
        for other in ("jev-9", "jev-1.14", "jev-1.14.0-rc1", "xjev-1.14.0")
            @test (@test_logs unpriced estimated_cost(s1(other))) == 0.0
        end
        # A caller-supplied table without a jev-latest row has no family rate to fall back to.
        @test (@test_logs unpriced estimated_cost(s1("jev-1.14.0"); pricing=Dict("gpt-5.4" => DEFAULT_PRICING["gpt-5.4"]))) == 0.0
        # A dated snapshot suffix is stripped before the lookup: the base version's row applies.
        @test estimated_cost(s1("jev-1.13.0-2026-09-22")) == estimated_cost(s1("jev-1.13.0")) > 0
    end
end

@testset "TypeSafe price rows: Jev 1.14 and Jev 1.13 at the Models page's price" begin
    # https://docs.typesafe.ai/models (2026-10-25): jev-1.14.0 and jev-1.13.0 at $42 per Btok of input, output
    # free; jev-latest and jev-preview point at jev-1.14.0.
    for model in ("jev-1.14.0", "jev-1.13.0", "jev-latest", "jev-preview")
        @test DEFAULT_PRICING[model] == UniLM._price(0.042, 0.042, 0.0)
    end
    # Both estimated_cost docstrings (one per method) state the price for the versions TypeSafe lists, not
    # for Jev 1.13 alone; read raw from the module's docs, which hold every method's docstring.
    docs = values(Base.Docs.meta(UniLM)[Base.Docs.Binding(UniLM, :estimated_cost)].docs)
    doc = replace(join((join(string.(d.text)) for d in docs), " "), r"\s+" => " ")
    @test !occursin("TypeSafe lists one price, for Jev 1.13", doc) && !occursin("lists for Jev 1.13)", doc)
    @test occursin("TypeSafe's Models page lists Jev 1.14 and Jev 1.13 at one price", doc)
    @test occursin("the price TypeSafe's Models page lists for Jev 1.14 and Jev 1.13", doc)
end

@testset "cumulative_cost" begin
    @testset "starts at zero" begin
        chat = Chat()
        @test cumulative_cost(chat) == 0.0
    end

    @testset "manual accumulation" begin
        chat = Chat()
        chat._cumulative_cost[] = 1.5
        @test cumulative_cost(chat) ≈ 1.5
    end
end

@testset "_accumulate_cost! across requests" begin
    chat = Chat(model="gpt-6-luna")
    m = Message(role=UniLM.RoleAssistant, content="hi")

    u1 = TokenUsage(prompt_tokens=1_000_000, completion_tokens=500_000, total_tokens=1_500_000)
    r1 = LLMSuccess(message=m, self=chat, usage=u1)
    UniLM._accumulate_cost!(chat, r1)
    cost1 = estimated_cost(r1)
    @test cost1 > 0.0
    @test cumulative_cost(chat) ≈ cost1

    u2 = TokenUsage(prompt_tokens=2_000_000, completion_tokens=1_000_000, total_tokens=3_000_000)
    r2 = LLMSuccess(message=m, self=chat, usage=u2)
    UniLM._accumulate_cost!(chat, r2)
    cost2 = estimated_cost(r2)
    @test cumulative_cost(chat) ≈ cost1 + cost2
end

@testset "fork cost independence" begin
    chat = Chat(model="gpt-6-luna")
    chat._cumulative_cost[] = 1.5
    forked = fork(chat)
    @test cumulative_cost(forked) ≈ 1.5

    forked._cumulative_cost[] += 0.5
    @test cumulative_cost(forked) ≈ 2.0
    @test cumulative_cost(chat) ≈ 1.5  # original unchanged
end

@testset "estimated_cost — ResponseSuccess" begin
    ro = UniLM.ResponseObject(
        id="resp_1", status="completed", model="gpt-5.4-mini",
        output=Any[],
        usage=Dict{String,Any}("input_tokens" => 1_000_000, "output_tokens" => 1_000_000, "total_tokens" => 2_000_000),
        raw=Dict{String,Any}()
    )
    rs = UniLM.ResponseSuccess(response=ro)
    cost = estimated_cost(rs)
    # gpt-5.4-mini: input=0.75/1M, output=4.5/1M => 0.75 + 4.5 = 5.25
    @test cost ≈ 5.25
end

@testset "DEFAULT_PRICING" begin
    @test haskey(DEFAULT_PRICING, "gpt-5.2")
    @test haskey(DEFAULT_PRICING, "gpt-6.1-sol")
    # Models OpenAI shut down on 2026-10-23 have no row.
    @test !haskey(DEFAULT_PRICING, "gpt-4.1-nano")
    @test !haskey(DEFAULT_PRICING, "o4-mini")

    # Verify structure + pin the live-verified gpt-5.2 rates (2026-06-21) so a revert is caught
    p = DEFAULT_PRICING["gpt-5.2"]
    @test p.input ≈ 1.75 / 1_000_000
    @test p.cached_input ≈ 0.175 / 1_000_000
    @test p.output ≈ 14.0 / 1_000_000

    # OpenAI pricing page, 2026-10-22 (USD per 1M tokens): 100K fresh input + 100K output, then
    # 100K input served entirely from the cache — short prompts, at most 272K input tokens.
    m = Message(role=UniLM.RoleAssistant, content="hi")
    fresh = TokenUsage(prompt_tokens=100_000, completion_tokens=100_000, total_tokens=200_000)
    cached = TokenUsage(prompt_tokens=100_000, cached_tokens=100_000, total_tokens=100_000)
    for (model, input_and_output, cached_input) in (("gpt-6-sol", 2.0 + 10.0, 0.20),
            ("gpt-6-luna", 0.10 + 0.50, 0.01), ("gpt-6.1-sol", 2.0 + 10.0, 0.10))
        @test estimated_cost(LLMSuccess(message=m, self=Chat(; model), usage=fresh)) ≈ input_and_output / 10
        @test estimated_cost(LLMSuccess(message=m, self=Chat(; model), usage=cached)) ≈ cached_input / 10
    end
end

@testset "estimated_cost — cached input + new rows" begin
    chat = Chat(model="gpt-5.4-mini")
    m = Message(role=UniLM.RoleAssistant, content="hi")
    rates = DEFAULT_PRICING["gpt-5.4-mini"]
    # cache-heavy: 900k of 1M input cached → billed at the cheaper cached rate
    u = TokenUsage(prompt_tokens=1_000_000, cached_tokens=900_000, completion_tokens=0)
    cost = estimated_cost(LLMSuccess(message=m, self=chat, usage=u))
    @test cost ≈ 100_000 * rates.input + 900_000 * rates.cached_input
    @test cost < 1_000_000 * rates.input          # strictly cheaper than full-rate (the regression this fixes)
    # cached_tokens clamped to prompt_tokens
    u2 = TokenUsage(prompt_tokens=10, cached_tokens=999, completion_tokens=0)
    @test estimated_cost(LLMSuccess(message=m, self=chat, usage=u2)) ≈ 10 * rates.cached_input
    # new rows exist with a discounted cached rate
    @test haskey(DEFAULT_PRICING, "gpt-5.5")
    @test haskey(DEFAULT_PRICING, "text-embedding-3-small")
    @test DEFAULT_PRICING["gpt-5.5"].cached_input < DEFAULT_PRICING["gpt-5.5"].input
end

@testset "estimated_cost — EmbeddingSuccess model resolution" begin
    # src/accounting.jl:69 — for an EmbeddingSuccess the priced model is read from
    # result.embeddings.model. text-embedding-3-small is priced at 0.02/1M input, 0.0 output.
    emb = Embeddings("hello")                 # defaults to text-embedding-3-small on OPENAI
    @test emb.model == "text-embedding-3-small"
    rates = DEFAULT_PRICING["text-embedding-3-small"]
    @test rates.input == 0.02 / 1_000_000
    @test rates.output == 0.0

    u = TokenUsage(prompt_tokens=500_000, completion_tokens=0, total_tokens=500_000)
    es = EmbeddingSuccess(embeddings=emb, usage=u, raw=Dict{String,Any}())
    # Hand-computed: 500_000 * 0.02/1M = 0.01 (no cached, no output).
    @test estimated_cost(es) ≈ 0.01
    @test estimated_cost(es) ≈ 500_000 * rates.input

    # The model genuinely comes from embeddings.model: an explicit override changes the cost,
    # and a non-zero completion count must NOT add anything (embedding output rate is 0.0).
    u2 = TokenUsage(prompt_tokens=500_000, completion_tokens=9_999, total_tokens=509_999)
    es2 = EmbeddingSuccess(embeddings=emb, usage=u2, raw=Dict{String,Any}())
    @test estimated_cost(es2) ≈ 0.01

    # A different priced embedding model resolves through line 69 to a different cost.
    emb_lg = Embeddings("hello"; model="text-embedding-3-large")
    es_lg = EmbeddingSuccess(embeddings=emb_lg, usage=u, raw=Dict{String,Any}())
    @test estimated_cost(es_lg) ≈ 500_000 * DEFAULT_PRICING["text-embedding-3-large"].input

    # An unpriced embedding model (resolved from embeddings.model, not in DEFAULT_PRICING) → 0, with a warning.
    emb_unk = Embeddings("hello"; model="text-embedding-unknown")
    es_unk = EmbeddingSuccess(embeddings=emb_unk, usage=u, raw=Dict{String,Any}())
    @test (@test_logs (:warn, "no price row for this model; its estimated cost is 0.0") estimated_cost(es_unk)) == 0.0
end

@testset "_price per-token conversion" begin
    # accounting.jl:5 — _price divides per-1M USD figures by 1_000_000. Distinct i/c/o values
    # prove no field is swapped and the /1e6 factor is applied to each.
    p = UniLM._price(2.0, 0.5, 8.0)
    @test p isa UniLM.PriceRow
    @test p.input == 2.0 / 1_000_000
    @test p.cached_input == 0.5 / 1_000_000
    @test p.output == 8.0 / 1_000_000
    # exact literal values (guards against an accidental /1e3 or missing division)
    @test p == (input = 2.0e-6, cached_input = 5.0e-7, output = 8.0e-6)
end

@testset "token_usage zero for Response/Embedding failures" begin
    # accounting.jl 42/43 (Response*) and 48/49 (Embedding*): every failure variant → zero usage.
    zero = TokenUsage()
    @test token_usage(ResponseFailure(response="boom", status=500)) == zero
    @test token_usage(ResponseCallError(error="net")) == zero
    @test token_usage(EmbeddingFailure(response="boom", status=500)) == zero
    @test token_usage(EmbeddingCallError(error="net")) == zero
    # sanity: a zero TokenUsage really is all-zero (so the equality above is meaningful)
    @test zero.prompt_tokens == 0 && zero.completion_tokens == 0 && zero.total_tokens == 0
end

@testset "results outside the token-billed APIs throw, they do not report zero" begin
    # `token_usage`/`estimated_cost` are documented on the LLMRequestResponse
    # supertype, but only the chat/Responses/embedding/image families carry usage.
    # Every other result type used to reach the caller as a raw MethodError from
    # inside the library; a fabricated 0.0 would be worse still — it is
    # indistinguishable from a genuinely free call and quietly under-counts spend.
    speech = SpeechSuccess(audio=UInt8[0x01], content_type="audio/mpeg")
    @test_throws ArgumentError token_usage(speech)
    @test_throws ArgumentError estimated_cost(speech)
    # An explicit model does not conjure usage out of a result that has none.
    @test_throws ArgumentError estimated_cost(speech; model="gpt-6-luna")
    err = try; estimated_cost(speech); catch e; e; end
    @test contains(sprint(showerror, err), "SpeechSuccess")   # names the offending type

    # The fallback covers the supertype, so no concrete result type falls through
    # to a MethodError any more.
    concrete = [T for T in InteractiveUtils.subtypes(UniLM.LLMRequestResponse) if isconcretetype(T)]
    @test length(concrete) > 50
    @test all(T -> hasmethod(token_usage, Tuple{T}), concrete)
    @test all(T -> hasmethod(estimated_cost, Tuple{T}), concrete)

    # Types that DO carry usage keep their exact behaviour: the fallback is only
    # reached when no more specific method exists.
    fallback = which(token_usage, Tuple{UniLM.LLMRequestResponse})
    chat = Chat(model="gpt-5.4-mini")
    covered = (LLMSuccess(message=Message(role=RoleAssistant, content="x"), self=chat,
                   usage=TokenUsage(prompt_tokens=1000, completion_tokens=500)),
               LLMFailure(response="e", status=500, self=chat),
               LLMCallError(error="e", self=chat),
               ResponseFailure(response="e", status=500),
               ResponseCallError(error="e"),
               ImageFailure(response="e", status=400),
               ImageCallError(error="e"),
               EmbeddingFailure(response="e", status=500),
               EmbeddingCallError(error="e"))
    for r in covered
        @test which(token_usage, Tuple{typeof(r)}) !== fallback
        @test token_usage(r) isa TokenUsage
    end
    @test token_usage(covered[1]) == TokenUsage(prompt_tokens=1000, completion_tokens=500)
    @test estimated_cost(covered[1]) ≈ (1000 * 0.75 + 500 * 4.5) / 1_000_000
    @test estimated_cost(covered[2]) == 0.0     # a failure is still a priced zero
end

@testset "Gemini pricing rows" begin
    # https://ai.google.dev/gemini-api/docs/pricing (2026-10-22), standard tier, USD per 1M tokens:
    # input, context-caching (cache read), output.
    for (model, i, c, o) in (("gemini-4-argon", 2.0, 0.10, 10.0), ("gemini-3.8-flash", 0.75, 0.075, 3.75),
                             ("gemini-3.6-flash", 0.75, 0.075, 3.75), ("gemini-3.5-flash-lite", 0.30, 0.03, 2.50),
                             ("gemini-3.1-pro-preview", 2.0, 0.20, 12.0),   # prompts up to 200K tokens
                             ("gemini-2.5-flash", 0.3, 0.03, 2.5), ("gemini-2.5-flash-lite", 0.1, 0.01, 0.4),
                             ("gemini-embedding-2", 0.20, 0.20, 0.0))
        @test DEFAULT_PRICING[model] == UniLM._price(i, c, o)
    end
    # Google serves gemini-3.7-flash as gemini-3.8-flash and gemini-3.5-flash as gemini-3.6-flash.
    @test DEFAULT_PRICING["gemini-3.7-flash"] == DEFAULT_PRICING["gemini-3.8-flash"]
    @test DEFAULT_PRICING["gemini-3.5-flash"] == DEFAULT_PRICING["gemini-3.6-flash"]
    # 1000 prompt tokens (200 of them cache reads) + 500 output on gemini-3.5-flash-lite.
    chat = Chat(service=GEMINIServiceEndpoint, model="gemini-3.5-flash-lite")
    usage = UniLM.TokenUsage(prompt_tokens=1000, cached_tokens=200, completion_tokens=500, total_tokens=1500)
    result = UniLM.LLMSuccess(message=Message(role=RoleAssistant, content="x"), self=chat, usage=usage)
    @test UniLM.estimated_cost(result) ≈ (800 * 0.30 + 200 * 0.03 + 500 * 2.50) / 1_000_000
end

@testset "Gemini billed cost — native and OpenAI-compatible endpoints" begin
    # Gemini caches prompts implicitly: a cache hit bills at the context-caching rate, every other
    # prompt token (a tool-use prompt included) at the input rate, and output, thinking included,
    # at the output rate.
    row = DEFAULT_PRICING["gemini-4-argon"]
    u = TokenUsage(prompt_tokens=10_000, cached_tokens=6_000, completion_tokens=2_000, total_tokens=12_000)
    want = (4_000 * 2.0 + 6_000 * 0.10 + 2_000 * 10.0) / 1_000_000
    default = which(UniLM._billed_cost, Tuple{Nothing,String,TokenUsage,UniLM.PriceRow})
    for service in (GEMINIServiceEndpoint, GEMINIOpenAIServiceEndpoint)
        @test which(UniLM._billed_cost, Tuple{Type{service},String,TokenUsage,UniLM.PriceRow}) !== default
        @test (@inferred UniLM._billed_cost(service, "gemini-4-argon", u, row)) ≈ want
        r = LLMSuccess(message=Message(role=RoleAssistant, content="x"), self=Chat(; service, model="gemini-4-argon"),
                       usage=u)
        @test estimated_cost(r) ≈ want
    end
end

@testset "Anthropic and DeepSeek price rows" begin
    # https://platform.claude.com/docs/en/about-claude/pricing.md (2026-10-22), USD per 1M tokens:
    # input, cache hit, output. https://api-docs.deepseek.com/quick_start/pricing: peak rates.
    for (model, i, c, o) in (("claude-fable-5-1", 10.0, 0.25, 50.0), ("claude-mythos-5-1", 10.0, 0.25, 50.0),
                             ("claude-fable-5", 10.0, 1.0, 50.0), ("claude-mythos-5", 10.0, 1.0, 50.0),
                             ("claude-opus-5-5", 4.0, 0.20, 20.0), ("claude-sonnet-5-5", 2.0, 0.10, 10.0),
                             ("claude-haiku-5-5", 0.10, 0.01, 0.50), ("claude-opus-5", 5.0, 0.50, 25.0),
                             ("claude-opus-4-8", 5.0, 0.50, 25.0), ("claude-opus-4-7", 5.0, 0.50, 25.0),
                             ("claude-opus-4-6", 5.0, 0.50, 25.0), ("claude-opus-4-5", 5.0, 0.50, 25.0),
                             ("claude-sonnet-5", 2.0, 0.20, 10.0), ("claude-sonnet-4-6", 3.0, 0.30, 15.0),
                             ("claude-haiku-4-5", 1.0, 0.10, 5.0),
                             ("deepseek-flash", 0.30, 0.006, 1.20), ("deepseek-v4-flash", 0.30, 0.006, 1.20),
                             ("deepseek-v4-flash-vision-exp", 0.30, 0.006, 1.20), ("deepseek-v4-pro", 1.32, 0.044, 3.96))
        @test DEFAULT_PRICING[model] == UniLM._price(i, c, o)
    end
    # Anthropic's dated snapshot ids (-YYYYMMDD) price at their alias row.
    m = Message(role=UniLM.RoleAssistant, content="x")
    u = TokenUsage(prompt_tokens=1_000_000, completion_tokens=1_000_000, total_tokens=2_000_000)
    cost(model) = estimated_cost(LLMSuccess(message=m, self=Chat(service=ANTHROPICServiceEndpoint, model=model), usage=u))
    @test cost("claude-haiku-4-5-20251001") == cost("claude-haiku-4-5") ≈ 6.0
    @test cost("claude-opus-4-5-20251101") ≈ 30.0
    @test (@test_logs (:warn, r"no price row") cost("claude-haiku-4-5-2025100")) == 0.0   # seven digits are not a date suffix
end

@testset "Anthropic billed cost: cache writes by retention, fast mode, US inference, Haiku 5.5 long prompts" begin
    # https://platform.claude.com/docs/en/about-claude/pricing.md: cache writes bill the input rate
    # × 1.25 (5-minute) or × 2 (1-hour); fast mode bills every rate × 2 (Opus 5.5: $8/$40 for
    # $4/$20); US-only inference every rate × 1.1; a Haiku 5.5 prompt over 100,000 tokens, cache
    # reads and writes counted, every rate × 5. The multipliers stack.
    m = Message(role=UniLM.RoleAssistant, content="x")
    cost(model, u) = estimated_cost(LLMSuccess(message=m, self=Chat(service=ANTHROPICServiceEndpoint, model=model), usage=u))
    usage(prompt; cached=0, writes=0, extended=0, out=0, tier=nothing, geo=nothing) =
        TokenUsage(prompt, out, prompt + out, cached, 0, writes, extended, tier, geo)
    # Opus 5.5 ($4 input, $0.20 cache hit, $20 output per 1M): 5,000 fresh, 2,000 reads, 2,000
    # 5-minute and 1,000 1-hour writes, 500 out.
    u = usage(10_000; cached=2_000, writes=3_000, extended=1_000, out=500)
    standard = (5_000 * 4.0 + 2_000 * 0.20 + 2_000 * 4.0 * 1.25 + 1_000 * 4.0 * 2.0 + 500 * 20.0) / 1e6
    @test isapprox(cost("claude-opus-5-5", u), standard; rtol=1e-12)
    for (tier, geo, factor) in (("standard", "global", 1.0), ("fast", nothing, 2.0), ("standard", "us", 1.1),
                                ("fast", "us", 2.2), (nothing, "us", 1.1))
        tagged = usage(10_000; cached=2_000, writes=3_000, extended=1_000, out=500, tier, geo)
        @test isapprox(cost("claude-opus-5-5", tagged), standard * factor; rtol=1e-12)
    end
    # A cache write the provider reports only in total is a 5-minute write.
    @test isapprox(cost("claude-sonnet-5-5", usage(1_000; writes=1_000)), 1_000 * 2.0 * 1.25 / 1e6; rtol=1e-12)
    # Haiku 5.5 ($0.10 / $0.01 / $0.50 per 1M): the prompt is every input token, reads and writes included.
    haiku(fresh, cached, w5, w1h, out) = (fresh * 0.10 + cached * 0.01 + w5 * 0.10 * 1.25 + w1h * 0.10 * 2.0 + out * 0.50) / 1e6
    at_threshold = usage(100_000; cached=30_000, writes=10_000, extended=4_000, out=1_000)
    @test isapprox(cost("claude-haiku-5-5", at_threshold), haiku(60_000, 30_000, 6_000, 4_000, 1_000); rtol=1e-12)
    over = usage(100_001; cached=30_000, writes=10_000, extended=4_000, out=1_000)
    @test isapprox(cost("claude-haiku-5-5", over), 5 * haiku(60_001, 30_000, 6_000, 4_000, 1_000); rtol=1e-12)
    # Cache reads alone can carry the prompt over the threshold.
    reads = usage(100_001; cached=90_001, out=10)
    @test isapprox(cost("claude-haiku-5-5", reads), 5 * haiku(10_000, 90_001, 0, 0, 10); rtol=1e-12)
    us_over = usage(100_001; cached=30_000, writes=10_000, extended=4_000, out=1_000, geo="us")
    @test isapprox(cost("claude-haiku-5-5", us_over), 5 * 1.1 * haiku(60_001, 30_000, 6_000, 4_000, 1_000); rtol=1e-12)
    # No other model has a prompt-length tier.
    long = usage(900_000; out=100)
    @test isapprox(cost("claude-opus-5-5", long), (900_000 * 4.0 + 100 * 20.0) / 1e6; rtol=1e-12)
    @test isapprox(cost("claude-haiku-4-5-20251001", long), (900_000 * 1.0 + 100 * 5.0) / 1e6; rtol=1e-12)
    # The served model decides the tier: a Haiku 5.5 reply to a request named otherwise.
    served = LLMSuccess(message=m, self=Chat(service=ANTHROPICServiceEndpoint, model="claude-opus-5-5"),
                        usage=over, model="claude-haiku-5-5")
    @test estimated_cost(served) == cost("claude-haiku-5-5", over)
    rates = DEFAULT_PRICING["claude-haiku-5-5"]
    @test (@inferred UniLM._billed_cost(ANTHROPICServiceEndpoint, "claude-haiku-5-5", over, rates)) ==
          cost("claude-haiku-5-5", over)
end

@testset "an unpriced model costs 0.0 and warns once per model id" begin
    m = Message(role=UniLM.RoleAssistant, content="x")
    r(model) = LLMSuccess(message=m, self=Chat(; model), usage=TokenUsage(prompt_tokens=10, completion_tokens=5))
    logs, total = Test.collect_test_logs() do
        estimated_cost(r("unpriced-a")) + estimated_cost(r("unpriced-a")) + estimated_cost(r("unpriced-b"))
    end
    @test total == 0.0
    warned = [l for l in logs if l.level == Base.CoreLogging.Warn]
    @test length(warned) == 2
    @test sort([l.kwargs[:model] for l in warned]) == ["unpriced-a", "unpriced-b"]
    logs, _ = Test.collect_test_logs(() -> estimated_cost(r("gpt-5.2")))
    @test isempty(logs)                                      # priced models stay silent
end

@testset "DEFAULT_PRICING is a lock-guarded table that still behaves like a Dict" begin
    @test DEFAULT_PRICING isa AbstractDict{String,UniLM.PriceRow}
    row = UniLM._price(1.0, 0.1, 2.0)
    try
        DEFAULT_PRICING["probe-model"] = row
        @test haskey(DEFAULT_PRICING, "probe-model") && DEFAULT_PRICING["probe-model"] == row
        @test get(DEFAULT_PRICING, "probe-model", nothing) == row
        @test get(DEFAULT_PRICING, "absent-model", nothing) === nothing
        @test "probe-model" in keys(DEFAULT_PRICING)
        @test length(DEFAULT_PRICING) == length(collect(DEFAULT_PRICING)) == length(keys(DEFAULT_PRICING))
        @test ("probe-model" => row) in collect(DEFAULT_PRICING)
        @test merge(DEFAULT_PRICING, Dict("x" => row))["x"] == row
        # Iteration walks a snapshot, so writing during iteration is safe.
        for (k, _) in DEFAULT_PRICING
            k == "probe-model" && (DEFAULT_PRICING["probe-model-2"] = row)
        end
        @test pop!(DEFAULT_PRICING, "probe-model-2") == row
        @test pop!(DEFAULT_PRICING, "probe-model-2", nothing) === nothing
    finally
        delete!(DEFAULT_PRICING, "probe-model")
        delete!(DEFAULT_PRICING, "probe-model-2")
    end
    @test !haskey(DEFAULT_PRICING, "probe-model")
    @test_throws KeyError DEFAULT_PRICING["absent-model"]

    # Readers on other tasks (every Chat success prices itself) never see a torn table
    # while rows are being added.
    m = Message(role=UniLM.RoleAssistant, content="x")
    r = LLMSuccess(message=m, self=Chat(model="gpt-5.2"),
                   usage=TokenUsage(prompt_tokens=1_000_000, completion_tokens=1_000_000, total_tokens=2_000_000))
    added = ["race-price-$i" for i in 1:20_000]
    bad, reads, done = Threads.Atomic{Int}(0), Threads.Atomic{Int}(0), Threads.Atomic{Bool}(false)
    try
        @sync begin
            for _ in 1:3
                Threads.@spawn while !done[]
                    ok = try
                        estimated_cost(r) ≈ 15.75
                    catch
                        false                            # a torn read can throw
                    end
                    Threads.atomic_add!(reads, 1)
                    ok || Threads.atomic_add!(bad, 1)
                    yield()                              # lets the writer run on any thread count
                end
            end
            Threads.@spawn begin
                while reads[] == 0; yield(); end         # write only while readers run
                foreach(k -> DEFAULT_PRICING[k] = row, added)   # forces repeated rehashes
                done[] = true
            end
        end
        @test bad[] == 0
        @test reads[] > 0
    finally
        foreach(k -> delete!(DEFAULT_PRICING, k), added)
    end
end

@testset "image and FIM results report the usage they carry" begin
    usage = Dict{String,Any}("input_tokens" => 50, "output_tokens" => 200, "total_tokens" => 250,
                             "input_tokens_details" => Dict{String,Any}("text_tokens" => 10, "image_tokens" => 40))
    ir = UniLM.ImageResponse(created=1, data=UniLM.ImageObject[], usage=usage, raw=Dict{String,Any}())
    @test token_usage(ImageSuccess(response=ir)) == TokenUsage(prompt_tokens=50, completion_tokens=200, total_tokens=250)
    fim = FIMSuccess(response=FIMResponse(choices=[FIMChoice(text="x")], model="deepseek-flash",
        usage=TokenUsage(prompt_tokens=1_000_000, completion_tokens=1_000_000, total_tokens=2_000_000)))
    @test token_usage(fim).prompt_tokens == 1_000_000
    @test estimated_cost(fim) ≈ 0.30 + 1.20
    @test token_usage(FIMSuccess(response=FIMResponse(choices=FIMChoice[]))) == TokenUsage()
    for failure in (FIMFailure(response="e", status=500), FIMCallError(error="e"))
        @test token_usage(failure) == TokenUsage()
        @test estimated_cost(failure) == 0.0
    end
end

@testset "an Ollama cloud model is priced by name, a local one is free" begin
    cloud = UniLM._ollama_cloud_model
    @test cloud("clef:cloud") && cloud("gpt-oss:120b-cloud") && cloud("registry.ollama.ai/library/clef:cloud")
    @test !cloud("gemma4:e4b") && !cloud("clef-cloud") && !cloud("cloudy:7b") && !cloud("kev")
    # The tag follows the last ':' only when no '/' comes after it: a registry's port is not a tag.
    @test !cloud("localhost:5000/library/clef-cloud") && cloud("localhost:5000/library/clef:cloud")
    @test cloud("clef:CLOUD") && cloud("gpt-oss:120B-CLOUD")                 # names resolve in any case
    @test !cloud("x:mycloud") && !cloud("x:cloudy") && !cloud("x:7b-cloudless")
    # Ollama trims a name before it parses it (internal/modelref): surrounding whitespace hides no cloud tag.
    @test cloud("clef:cloud ") && cloud(" gpt-oss:120b-cloud\n") && !cloud(" gemma4:e4b ")
    ep = OllamaEndpoint(base_url="http://box:11434")
    succ(model) = (c = Chat(service=ep, model=model); push!(c, Message(Val(:system), "s"));
                   push!(c, Message(Val(:user), "hi"));
                   LLMSuccess(message=Message(role=RoleAssistant, content="ok"), self=c,
                              usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15)))
    @test (@test_logs estimated_cost(succ("gemma4:e4b"))) == 0.0          # local: free, no warning
    @test (@test_logs (:warn, r"no price row") estimated_cost(succ("zz-unpriced:7b-cloud"))) == 0.0
    # With a row for each, the cloud model is charged and the local one is still free.
    row = UniLM._price(1.0, 0.1, 2.0)
    table = Dict("zz-priced:cloud" => row, "zz-priced:7b" => row)
    @test (@test_logs estimated_cost(succ("zz-priced:cloud"); pricing=table)) ≈ 10 * 1e-6 + 5 * 2e-6
    @test (@test_logs estimated_cost(succ("zz-priced:7b"); pricing=table)) == 0.0
    s1 = SystemOneSuccess(SystemOneResponse("zz-unpriced-s1:cloud", Dict{String,SystemOneAnswer}(),
                          TokenUsage(prompt_tokens=10, total_tokens=10), nothing, Dict{String,Any}(), ep))
    @test (@test_logs (:warn, r"no price row") estimated_cost(s1)) == 0.0
end

@testset "a call made directly to Ollama's cloud is priced by name, as a cloud model is" begin
    # https://docs.ollama.com/cloud: a direct API request to ollama.com names the model as the cloud lists it,
    # "such as `gemma4:31b`", with no cloud tag. Ollama bills such a call, so it is priced by name — no row, then
    # the missing-price warning — never as a free local call.
    direct = OllamaEndpoint(base_url="https://ollama.com")
    succ(model) = (c = Chat(service=direct, model=model); push!(c, Message(Val(:system), "s"));
                   push!(c, Message(Val(:user), "hi"));
                   LLMSuccess(message=Message(role=RoleAssistant, content="ok"), self=c,
                              usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15)))
    @test (@test_logs (:warn, r"no price row") estimated_cost(succ("zz-direct:31b"))) == 0.0
    row = UniLM._price(1.0, 0.1, 2.0)
    @test (@test_logs estimated_cost(succ("zz-direct-priced:31b"); pricing=Dict("zz-direct-priced:31b" => row))) ≈
          10 * 1e-6 + 5 * 2e-6
    s1 = SystemOneSuccess(SystemOneResponse("zz-direct-s1:31b", Dict{String,SystemOneAnswer}(),
                          TokenUsage(prompt_tokens=10, total_tokens=10), nothing, Dict{String,Any}(), direct))
    @test (@test_logs (:warn, r"no price row") estimated_cost(s1)) == 0.0
    # The same name through a server of the caller's own is a local model, and free.
    c = Chat(service=OllamaEndpoint(base_url="http://box:11434"), model="zz-direct:31b")
    push!(c, Message(Val(:system), "s")); push!(c, Message(Val(:user), "hi"))
    loc = LLMSuccess(message=Message(role=RoleAssistant, content="ok"), self=c,
                     usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15))
    @test (@test_logs estimated_cost(loc)) == 0.0
end

@testset "Responses usage reads cache-write details" begin
    # https://developers.openai.com/api/docs/guides/prompt-caching: input_tokens counts every
    # input token; cache_write_tokens_details splits the writes by lifetime (ttl_30m_tokens +
    # ttl_12h_tokens == cache_write_tokens). The billed tier is the response's service_tier.
    usage(details) = JSON.parse("""{"input_tokens":2224,"output_tokens":4,""" *
        """"input_tokens_details":$details,"output_tokens_details":{"reasoning_tokens":0}}"""; dicttype=Dict{String,Any})
    result(details, raw) = ResponseSuccess(response=UniLM.ResponseObject(id="resp_1", status="completed",
        model="gpt-6-luna", output=Any[], usage=usage(details), raw=raw))
    u = token_usage(result("""{"cached_tokens":0,"cache_write_tokens":2221,""" *
        """"cache_write_tokens_details":{"ttl_30m_tokens":0,"ttl_12h_tokens":2221}}""",
        Dict{String,Any}("service_tier" => "default")))
    @test u.cache_write_tokens == 2221
    @test u.cache_write_extended_tokens == 2221
    @test u.service_tier == "default"
    @test u == TokenUsage(prompt_tokens=2224, completion_tokens=4, cache_write_tokens=2221,
                          cache_write_extended_tokens=2221, service_tier="default")
    # 30-minute writes are not extended; a reply without details or a tier reports none.
    u30 = token_usage(result("""{"cached_tokens":3,"cache_write_tokens":300,""" *
        """"cache_write_tokens_details":{"ttl_30m_tokens":300,"ttl_12h_tokens":0}}""", Dict{String,Any}()))
    @test (u30.cached_tokens, u30.cache_write_tokens, u30.cache_write_extended_tokens, u30.service_tier) ==
          (3, 300, 0, nothing)
    bare = token_usage(result("""{"cached_tokens":0}""", Dict{String,Any}("service_tier" => nothing)))
    @test (bare.cache_write_tokens, bare.cache_write_extended_tokens, bare.service_tier) == (0, 0, nothing)
end

@testset "default billed cost" begin
    # An endpoint whose provider defines no pricing of its own is billed the standard rates:
    # fresh input and cache writes at the input rate, cache reads at the cached rate.
    pricing = Dict("m" => UniLM._price(1.0, 0.1, 4.0))
    u = TokenUsage(prompt_tokens=1000, cached_tokens=200, cache_write_tokens=300, completion_tokens=50,
                   total_tokens=1050)
    chat = Chat(service=GenericOpenAIEndpoint("http://localhost:9", ""), model="m")
    r = LLMSuccess(message=Message(role=RoleAssistant, content="x"), self=chat, usage=u)
    @test isapprox(estimated_cost(r; pricing), (500 * 1.0 + 200 * 0.1 + 300 * 1.0 + 50 * 4.0) / 1e6; rtol=1e-12)
    # The retention of a write, the tier and the region change nothing at the standard rates.
    tagged = TokenUsage(1000, 50, 1050, 200, 0, 300, 300, "flex", "us")
    @test isapprox(estimated_cost(LLMSuccess(message=r.message, self=chat, usage=tagged); pricing),
                   estimated_cost(r; pricing); rtol=1e-12)
    @test (@inferred UniLM._billed_cost(nothing, "m", u, pricing["m"])) ≈ estimated_cost(r; pricing)
    @test (@inferred UniLM._billed_cost(chat.service, "m", u, pricing["m"])) ≈ estimated_cost(r; pricing)
end

# A price row has no audio-token rates. Refuted if: a result billed at the standard rates bills its
# audio tokens at a row's text rates, or prices them without a warning.
@testset "default billed cost: audio tokens leave the call unpriced, never billed at text rates" begin
    pricing = Dict("m-hears" => UniLM._price(1.0, 0.1, 4.0), "m-speaks" => UniLM._price(1.0, 0.1, 4.0))
    generic = GenericOpenAIEndpoint("http://localhost:9", "")
    result(service, model, u) = LLMSuccess(message=Message(role=RoleAssistant, content="x"), self=Chat(; service, model),
                                           usage=u)
    heard = TokenUsage(prompt_tokens=1000, completion_tokens=50, total_tokens=1050, prompt_audio_tokens=600)
    spoken = TokenUsage(prompt_tokens=1000, completion_tokens=50, total_tokens=1050, completion_audio_tokens=40)
    says = "no audio-token rates for this model on this endpoint; its estimated cost is 0.0"
    warned(logs) = [(l.level, l.message, l.kwargs[:model], l.kwargs[:endpoint]) for l in logs]
    # Audio input or audio output: unpriced, with one warning per model and endpoint.
    logs, total = Test.collect_test_logs() do
        estimated_cost(result(generic, "m-hears", heard); pricing) +
            estimated_cost(result(generic, "m-hears", spoken); pricing) +
            estimated_cost(result(generic, "m-speaks", spoken); pricing)
    end
    @test total == 0.0
    @test warned(logs) == [(Base.CoreLogging.Warn, says, "m-hears", :GenericOpenAIEndpoint),
                           (Base.CoreLogging.Warn, says, "m-speaks", :GenericOpenAIEndpoint)]
    # gpt-audio-1.5's audio-token rates are OpenAI's: on Azure and DeepSeek its text row prices no audio,
    # and the warning of the one endpoint does not hide the other's.
    logs, total = Test.collect_test_logs() do
        estimated_cost(result(AZUREServiceEndpoint, "gpt-audio-1.5", spoken)) +
            estimated_cost(result(DeepSeekEndpoint("k"), "gpt-audio-1.5", heard))
    end
    @test total == 0.0
    @test warned(logs) == [(Base.CoreLogging.Warn, says, "gpt-audio-1.5", :AZUREServiceEndpoint),
                           (Base.CoreLogging.Warn, says, "gpt-audio-1.5", :DeepSeekEndpoint)]
    @test isapprox((@test_logs estimated_cost(result(OPENAIServiceEndpoint, "gpt-audio-1.5", spoken))),
                   (1000 * 2.5 + 10 * 10.0 + 40 * 64.0) / 1e6; rtol=1e-12)
    # Without audio tokens the same result bills the standard rates, silently.
    text = TokenUsage(prompt_tokens=1000, completion_tokens=50, total_tokens=1050)
    @test isapprox((@test_logs estimated_cost(result(generic, "m-hears", text); pricing)), (1000 * 1.0 + 50 * 4.0) / 1e6;
                   rtol=1e-12)
    @test (@test_logs (:warn, says) @inferred UniLM._billed_cost(nothing, "m-hears", heard, pricing["m-hears"])) === 0.0
end

@testset "served model prices the call" begin
    pricing = Dict("m-served" => UniLM._price(1.0, 0.1, 4.0), "m-asked" => UniLM._price(2.0, 0.2, 8.0))
    u = TokenUsage(prompt_tokens=1000, completion_tokens=50, total_tokens=1050)
    chat = Chat(service=GenericOpenAIEndpoint("http://localhost:9", ""), model="m-asked")
    m = Message(role=RoleAssistant, content="x")
    served = LLMSuccess(message=m, self=chat, usage=u, model="m-served")
    asked = LLMSuccess(message=m, self=chat, usage=u)
    @test isapprox(estimated_cost(served; pricing), (1000 * 1.0 + 50 * 4.0) / 1e6; rtol=1e-12)
    @test isapprox(estimated_cost(asked; pricing), (1000 * 2.0 + 50 * 8.0) / 1e6; rtol=1e-12)
    @test estimated_cost(served; pricing, model="m-asked") == estimated_cost(asked; pricing)   # explicit wins
    # A served model without a row is not priced as the one requested: it costs 0.0 and warns.
    unlisted = LLMSuccess(message=m, self=chat, usage=u, model="m-served-unlisted")
    @test (@test_logs (:warn, r"no price row") estimated_cost(unlisted; pricing)) == 0.0
    # A dated snapshot of the served model uses its base row.
    @test estimated_cost(LLMSuccess(message=m, self=chat, usage=u, model="m-served-2026-09-14"); pricing) ==
          estimated_cost(served; pricing)
end

@testset "an image result has no priced model; a failure costs 0.0" begin
    # The Images API bills text and image tokens at their own rates, which a price row does not
    # model: without `model=` the estimate throws instead of returning a zero.
    usage = Dict{String,Any}("input_tokens" => 50, "output_tokens" => 200, "total_tokens" => 250)
    img = ImageSuccess(response=UniLM.ImageResponse(created=1, data=UniLM.ImageObject[], usage=usage,
                                                    raw=Dict{String,Any}()))
    err = try estimated_cost(img); nothing catch e; e end
    @test err isa ArgumentError && occursin("ImageSuccess", err.msg) && occursin("model=", err.msg)
    row = Dict("my-image-rates" => UniLM._price(5.0, 1.25, 40.0))
    @test isapprox(estimated_cost(img; model="my-image-rates", pricing=row), (50 * 5.0 + 200 * 40.0) / 1e6;
                   rtol=1e-12)
    # A failure of any token-billed API is still a priced zero, a System One one included.
    for failure in (ImageFailure(response="e", status=400), ImageCallError(error="e"),
                    SystemOneFailure(response="{}", status=500), SystemOneCallError(error="e"))
        @test estimated_cost(failure) == 0.0
    end
end

@testset "OpenAI price rows" begin
    # https://developers.openai.com/api/docs/pricing (2026-10-22), USD per 1M tokens: input,
    # cached input, output. The pro models list no cached-input rate: a cache read bills as input;
    # so does gpt-audio-1.5, whose row is its text rates.
    for (model, i, c, o) in (("gpt-6-astra", 10.0, 1.0, 50.0), ("gpt-6.1-sol", 2.0, 0.10, 10.0),
            ("gpt-6-sol", 2.0, 0.20, 10.0), ("gpt-6-luna", 0.10, 0.01, 0.50),
            ("gpt-5.6-sol", 4.0, 0.40, 20.0), ("gpt-5.6-terra", 2.0, 0.20, 12.0), ("gpt-5.6-luna", 0.20, 0.02, 1.20),
            ("gpt-5.5", 5.0, 0.50, 30.0), ("gpt-5.5-pro", 30.0, 30.0, 180.0), ("gpt-5.4", 2.5, 0.25, 15.0),
            ("gpt-5.4-pro", 30.0, 30.0, 180.0), ("gpt-5.4-mini", 0.75, 0.075, 4.5), ("gpt-5.2", 1.75, 0.175, 14.0),
            ("chat-latest", 5.0, 0.50, 30.0), ("gpt-audio-1.5", 2.5, 2.5, 10.0), ("text-embedding-3-small", 0.02, 0.02, 0.0),
            ("text-embedding-3-large", 0.13, 0.13, 0.0), ("text-embedding-ada-002", 0.10, 0.10, 0.0))
        @test DEFAULT_PRICING[model] == UniLM._price(i, c, o)
    end
    # Shut down on 2026-10-23.
    @test !haskey(DEFAULT_PRICING, "o4-mini") && !haskey(DEFAULT_PRICING, "gpt-4.1-nano")
end

# An OpenAI Chat result of `model` with usage `u`, billed by the OpenAI endpoint's own pricing.
_oa_result(model, u) = LLMSuccess(message=Message(role=RoleAssistant, content="x"), self=Chat(; model), usage=u)

@testset "gpt-5-search-api, the Chat Completions search model, has its price row" begin
    # OpenAI's pricing page (2026-10-25), "Search": gpt-5-search-api $1.25 input, $0.125 cached input and
    # $10.00 output per 1M tokens. A Chat with web_search_options runs on it; without a row its cost read 0.0.
    @test DEFAULT_PRICING["gpt-5-search-api"] == UniLM._price(1.25, 0.125, 10.0)
    u = TokenUsage(prompt_tokens=1_000_000, cached_tokens=200_000, completion_tokens=100_000, total_tokens=1_100_000)
    @test isapprox((@test_logs estimated_cost(_oa_result("gpt-5-search-api", u))),
                   800_000 * 1.25e-6 + 200_000 * 0.125e-6 + 100_000 * 10.0e-6; rtol=1e-12)
end

@testset "gpt-5.2-pro has its price row" begin
    # OpenAI's pricing page (2026-10-26), Standard: "| gpt-5.2-pro | $21.00 | $168.00 | - |" per 1M tokens —
    # input, output, and no cached-input rate, so a cache read bills as input, as on the other pro models.
    # Without a row its cost read 0.0, with the missing-price warning.
    @test haskey(DEFAULT_PRICING, "gpt-5.2-pro")
    @test DEFAULT_PRICING["gpt-5.2-pro"] == UniLM._price(21.0, 21.0, 168.0)
    u = TokenUsage(prompt_tokens=1_000_000, cached_tokens=200_000, completion_tokens=100_000, total_tokens=1_100_000)
    for model in ("gpt-5.2-pro", "gpt-5.2-pro-2025-12-11")   # the snapshot OpenAI's models list names
        @test isapprox((@test_logs estimated_cost(_oa_result(model, u))), 1_000_000 * 21e-6 + 100_000 * 168e-6; rtol=1e-12)
    end
end
_oa_tiered(prompt, completion, tier) = TokenUsage(prompt, completion, prompt + completion, 0, 0, 0, 0, tier, nothing)

@testset "OpenAI cost: cache writes by lifetime" begin
    # The prompt-caching guide's worked example: 12,000 input tokens, of which 4,000 cache reads,
    # 3,000 30-minute and 4,000 12-hour writes, at $1 input and $0.10 cached per 1M tokens:
    # "(1000 + 4000 × 0.1 + 3000 × 1.25 + 4000 × 2.5) / 1000000 = $0.01515".
    usage = JSON.parse("""{"input_tokens":12000,"output_tokens":0,"input_tokens_details":{"cached_tokens":4000,
        "cache_write_tokens":7000,"cache_write_tokens_details":{"ttl_30m_tokens":3000,"ttl_12h_tokens":4000}}}""";
        dicttype=Dict{String,Any})
    response(model) = ResponseSuccess(response=UniLM.ResponseObject(id="r", status="completed", model=model,
        output=Any[], usage=usage, raw=Dict{String,Any}("service_tier" => "default")), service=OPENAIServiceEndpoint)
    row = UniLM._price(1.0, 0.1, 4.0)
    @test isapprox(estimated_cost(response("gpt-6-luna"); pricing=Dict("gpt-6-luna" => row)), 0.01515; rtol=1e-12)
    # Models before GPT-5.6 have "No additional cache-write charge": writes bill as input.
    @test isapprox(estimated_cost(response("gpt-5.5"); pricing=Dict("gpt-5.5" => row)),
                   (1000 + 4000 * 0.1 + 7000) / 1e6; rtol=1e-12)
    # Chat Completions writes at the 30-minute lifetime only.
    u = TokenUsage(prompt_tokens=10_000, cached_tokens=1_000, cache_write_tokens=4_000, completion_tokens=100,
                   total_tokens=10_100)
    @test isapprox(estimated_cost(_oa_result("gpt-6-luna", u)), (5_000 * 0.10 + 1_000 * 0.01 + 4_000 * 0.125 +
                   100 * 0.50) / 1e6; rtol=1e-12)
    @test (@inferred UniLM._billed_cost(OPENAIServiceEndpoint, "gpt-6-luna", u, DEFAULT_PRICING["gpt-6-luna"])) isa Float64
end

@testset "OpenAI cost: the billed tier's multiple of the standard rates" begin
    # Ultrafast 6x Standard on GPT-6 Astra and GPT-6.1 Sol; Flex half; Fast twice on GPT-6 and
    # GPT-5.6, 2.5x on gpt-5.5. GPT-5.6 and earlier report Fast mode as "priority".
    cost(model, tier) = estimated_cost(_oa_result(model, _oa_tiered(1_000, 100, tier)))
    @test cost("gpt-6.1-sol", nothing) ≈ (1_000 * 2.0 + 100 * 10.0) / 1e6
    @test cost("gpt-6.1-sol", "default") == cost("gpt-6.1-sol", nothing)
    @test cost("gpt-6.1-sol", "ultrafast") ≈ 6 * (1_000 * 2.0 + 100 * 10.0) / 1e6
    @test cost("gpt-6-astra", "ultrafast") ≈ 6 * (1_000 * 10.0 + 100 * 50.0) / 1e6
    @test cost("gpt-6-luna", "flex") ≈ 0.5 * (1_000 * 0.10 + 100 * 0.50) / 1e6
    for tier in ("priority", "fast")
        @test cost("gpt-6-sol", tier) ≈ 2 * (1_000 * 2.0 + 100 * 10.0) / 1e6
        @test cost("gpt-5.6-terra", tier) ≈ 2 * (1_000 * 2.0 + 100 * 12.0) / 1e6
        @test cost("gpt-5.5", tier) ≈ 2.5 * (1_000 * 5.0 + 100 * 30.0) / 1e6
        @test cost("gpt-5.2", tier) ≈ 2 * (1_000 * 1.75 + 100 * 14.0) / 1e6
    end
    # A tier scales every rate of the request, cache reads and writes included.
    u = TokenUsage(10_000, 100, 10_100, 1_000, 0, 4_000, 0, "flex", nothing)
    @test isapprox(estimated_cost(_oa_result("gpt-6-luna", u)),
                   0.5 * (5_000 * 0.10 + 1_000 * 0.01 + 4_000 * 0.125 + 100 * 0.50) / 1e6; rtol=1e-12)
    # A tier the page gives no number for — Scale Tier, or Fast on gpt-5.4-pro — is estimated at
    # the standard rates, with one warning per model and tier.
    logs, total = Test.collect_test_logs() do
        cost("gpt-6-luna", "scale") + cost("gpt-6-luna", "scale") + cost("gpt-5.4-pro", "priority")
    end
    @test total ≈ 2 * (1_000 * 0.10 + 100 * 0.50) / 1e6 + (1_000 * 30.0 + 100 * 180.0) / 1e6
    warned = [l for l in logs if l.level == Base.CoreLogging.Warn]
    @test [(l.kwargs[:model], l.kwargs[:tier]) for l in warned] == [("gpt-6-luna", "scale"), ("gpt-5.4-pro", "priority")]
    @test all(l -> occursin("service tier", l.message), warned)
end

@testset "OpenAI cost: prompts over 272K input tokens" begin
    # "Prompts with more than 272K input tokens are priced at 2x input and cache rates and 1.5x
    # output for the full request" — on the families the pricing page gives long-context rates.
    u = TokenUsage(prompt_tokens=300_000, cached_tokens=100_000, cache_write_tokens=50_000, completion_tokens=1_000,
                   total_tokens=301_000)
    long_sol = 2 * (150_000 * 2.0 + 100_000 * 0.20 + 50_000 * 1.25 * 2.0) / 1e6 + 1.5 * 1_000 * 10.0 / 1e6
    @test isapprox(estimated_cost(_oa_result("gpt-6-sol", u)), long_sol; rtol=1e-12)
    @test estimated_cost(_oa_result("gpt-6-sol-2026-09-14", u)) == estimated_cost(_oa_result("gpt-6-sol", u))
    edge(n) = estimated_cost(_oa_result("gpt-6-sol", TokenUsage(prompt_tokens=n, completion_tokens=1_000)))
    @test edge(272_000) ≈ (272_000 * 2.0 + 1_000 * 10.0) / 1e6           # short context: at most 272K
    @test edge(272_001) ≈ (2 * 272_001 * 2.0 + 1.5 * 1_000 * 10.0) / 1e6
    # gpt-5.2 lists one rate for every prompt length.
    @test estimated_cost(_oa_result("gpt-5.2", u)) ≈ (150_000 * 1.75 + 100_000 * 0.175 + 50_000 * 1.75 + 1_000 * 14.0) / 1e6
    # A tier's long-context rates are its multiple of the long-context standard rates.
    @test estimated_cost(_oa_result("gpt-6-astra", _oa_tiered(300_000, 1_000, "ultrafast"))) ≈
          6 * (2 * 300_000 * 10.0 + 1.5 * 1_000 * 50.0) / 1e6
    @test estimated_cost(_oa_result("gpt-5.5", _oa_tiered(300_000, 1_000, "flex"))) ≈
          0.5 * (2 * 300_000 * 5.0 + 1.5 * 1_000 * 30.0) / 1e6
    # gpt-5.5 lists Fast for short prompts only: a long one is estimated at the standard long-context rates.
    long_fast = @test_logs (:warn, r"long-context") estimated_cost(_oa_result("gpt-5.5", _oa_tiered(300_000, 1_000, "priority")))
    @test long_fast ≈ (2 * 300_000 * 5.0 + 1.5 * 1_000 * 30.0) / 1e6
end

@testset "OpenAI cost: a model without OpenAI price terms is billed at its row's standard rates" begin
    # No per-model term passes to an unlisted id: a caller's own row for a newer model prices it
    # without cache-write premiums or a long-context tier, and an unknown tier warns.
    pricing = Dict("gpt-7-x" => UniLM._price(1.0, 0.1, 4.0))
    u = TokenUsage(prompt_tokens=300_000, cache_write_tokens=100_000, completion_tokens=1_000, total_tokens=301_000)
    @test isapprox(estimated_cost(_oa_result("gpt-7-x", u); pricing), (300_000 * 1.0 + 1_000 * 4.0) / 1e6; rtol=1e-12)
    flex = TokenUsage(1_000, 100, 1_100, 0, 0, 0, 0, "flex", nothing)
    @test (@test_logs (:warn, r"service tier") estimated_cost(_oa_result("gpt-7-x", flex); pricing)) ≈
          (1_000 * 1.0 + 100 * 4.0) / 1e6
end

# A Chat usage of `prompt` input tokens, `heard` of them audio, and `completion` output tokens,
# `spoken` of them audio, billed at the default tier.
_oa_audio(prompt, heard, completion, spoken; cached=0) =
    TokenUsage(prompt_tokens=prompt, completion_tokens=completion, total_tokens=prompt + completion,
               cached_tokens=cached, service_tier="default", prompt_audio_tokens=heard,
               completion_audio_tokens=spoken)

@testset "OpenAI cost: Chat audio tokens bill at the model's audio rates, the rest at its text rates" begin
    # https://developers.openai.com/api/docs/pricing (2026-10-22), gpt-audio-1.5 per 1M tokens: text
    # $2.50 input and $10 output, audio $32 input and $64 output; no cached-input rate for either.
    u = _oa_audio(1_000, 600, 500, 420)
    # The audio subsets follow every other field, so the five positional counts alone build a usage.
    @test u == TokenUsage(1_000, 500, 1_500, 0, 0, 0, 0, "default", nothing, 600, 420)
    @test TokenUsage(10, 2, 12, 3, 1) == TokenUsage(10, 2, 12, 3, 1, 0, 0, nothing, nothing, 0, 0)
    want = (400 * 2.5 + 600 * 32.0 + 80 * 10.0 + 420 * 64.0) / 1e6
    @test isapprox((@test_logs estimated_cost(_oa_result("gpt-audio-1.5", u))), want; rtol=1e-12)   # priced: no warning
    # A cache read bills as input: the page gives this model no cached-input rate.
    @test isapprox(estimated_cost(_oa_result("gpt-audio-1.5", _oa_audio(1_000, 600, 500, 420; cached=300))), want;
                   rtol=1e-12)
    # Audio on one side only: a text reply to a spoken prompt, a spoken reply to a text prompt.
    @test isapprox(estimated_cost(_oa_result("gpt-audio-1.5", _oa_audio(1_000, 600, 500, 0))),
                   (400 * 2.5 + 600 * 32.0 + 500 * 10.0) / 1e6; rtol=1e-12)
    @test isapprox(estimated_cost(_oa_result("gpt-audio-1.5", _oa_audio(1_000, 0, 500, 420))),
                   (1_000 * 2.5 + 80 * 10.0 + 420 * 64.0) / 1e6; rtol=1e-12)
    # A dated snapshot, served or requested, bills its base model's text and audio rates.
    snapshot = "gpt-audio-1.5-2026-08-28"
    @test estimated_cost(_oa_result(snapshot, u)) == estimated_cost(_oa_result("gpt-audio-1.5", u))
    served = LLMSuccess(message=Message(role=RoleAssistant, content="x"), self=Chat(model="gpt-audio-1.5"),
                        usage=u, model=snapshot)
    @test estimated_cost(served) == estimated_cost(_oa_result("gpt-audio-1.5", u))
    @test (@inferred UniLM._billed_cost(OPENAIServiceEndpoint, "gpt-audio-1.5", u, DEFAULT_PRICING["gpt-audio-1.5"])) ==
          estimated_cost(_oa_result("gpt-audio-1.5", u))
end

@testset "OpenAI cost: audio tokens of a model without audio rates are unpriced, never billed as text" begin
    # As for a model without a price row: the estimate is 0.0, and a warning names the model once.
    u = _oa_audio(1_000, 600, 500, 420)
    custom = Dict("gpt-audio-2" => UniLM._price(2.5, 2.5, 10.0), "gpt-audio-1.5-mini" => UniLM._price(0.6, 0.6, 2.4))
    logs, total = Test.collect_test_logs() do
        estimated_cost(_oa_result("gpt-audio-1.5", u); model="gpt-6-luna") +       # a text model's row
        estimated_cost(_oa_result("gpt-audio-1.5", u); model="gpt-6-luna") +
        estimated_cost(_oa_result("gpt-audio-2", u); pricing=custom) +             # an unknown later id
        estimated_cost(_oa_result("gpt-audio-1.5-mini", _oa_audio(1_000, 600, 500, 0)); pricing=custom)  # a near miss
    end
    @test total == 0.0
    warned = [l for l in logs if l.level == Base.CoreLogging.Warn]
    @test [l.kwargs[:model] for l in warned] == ["gpt-6-luna", "gpt-audio-2", "gpt-audio-1.5-mini"]
    @test all(l -> occursin("no audio-token rates", l.message), warned)
    # Without audio tokens the same models bill their text rates, silently.
    text = TokenUsage(prompt_tokens=1_000, completion_tokens=500, total_tokens=1_500)
    @test isapprox((@test_logs estimated_cost(_oa_result("gpt-audio-2", text); pricing=custom)),
                   (1_000 * 2.5 + 500 * 10.0) / 1e6; rtol=1e-12)
    @test isapprox((@test_logs estimated_cost(_oa_result("gpt-6-luna", text))), (1_000 * 0.10 + 500 * 0.50) / 1e6;
                   rtol=1e-12)
end

@testset "OpenAI cost: a usage without audio tokens is billed by the text formula alone, to the last bit" begin
    # Each pinned value is the one the text formula gives in floating point: `isapprox` shows the
    # formula, `===` holds the bits, so the audio terms leave every text-only estimate unchanged.
    for (model, u, formula, pinned) in (
            ("gpt-6-luna", TokenUsage(10_000, 100, 10_100, 1_000, 0, 4_000, 1_500, "flex", nothing),
             0.5 * (5_000 * 0.10 + 2_500 * 0.125 + 1_500 * 0.25 + 1_000 * 0.01 + 100 * 0.50) / 1e6, 0.0006237500000000001),
            ("gpt-6-sol", TokenUsage(300_000, 1_000, 301_000, 100_000, 0, 50_000, 20_000, "default", nothing),
             (2 * (150_000 * 2.0 + 30_000 * 2.5 + 20_000 * 5.0 + 100_000 * 0.20) + 1.5 * 1_000 * 10.0) / 1e6, 1.005),
            ("gpt-6-astra", TokenUsage(5_000, 700, 5_700, 0, 0, 0, 0, "ultrafast", nothing),
             6 * (5_000 * 10.0 + 700 * 50.0) / 1e6, 0.51),
            ("gpt-5.5", TokenUsage(12_345, 678, 13_023, 2_345, 0, 0, 0, "priority", nothing),
             2.5 * (10_000 * 5.0 + 2_345 * 0.50 + 678 * 30.0) / 1e6, 0.17878125),
            ("chat-latest", TokenUsage(999, 111, 1_110, 99, 0, 0, 0, nothing, nothing),
             (900 * 5.0 + 99 * 0.50 + 111 * 30.0) / 1e6, 0.007879500000000001))
        cost = estimated_cost(_oa_result(model, u))
        @test isapprox(cost, formula; rtol=1e-12)
        @test cost === pinned
    end
end
