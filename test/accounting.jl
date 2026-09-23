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
        cost2 = estimated_cost(s; model="gpt-4.1-mini")
        # gpt-4.1-mini: input=0.4/1M, output=1.6/1M => 0.4 + 1.6 = 2.0
        @test cost2 ≈ 2.0
    end

    @testset "unknown model returns 0" begin
        chat2 = Chat(model="unknown-model")
        s2 = LLMSuccess(message=m, self=chat2, usage=u)
        @test estimated_cost(s2) == 0.0
    end

    @testset "failure returns 0" begin
        @test estimated_cost(LLMFailure(response="err", status=500, self=chat)) == 0.0
    end

    @testset "versioned Jev id without a row is priced at jev-latest" begin
        # Assumption: TypeSafe lists one price, for Jev 1.13 ($42 per Btok of input; output
        # free); a later version is assumed to keep it until the pricing page says otherwise.
        # Responses report the versioned id that answered, e.g. a release newer than the table.
        ju = TokenUsage(prompt_tokens=1_000_000, completion_tokens=500, total_tokens=1_000_500)
        s1(model) = SystemOneSuccess(UniLM.SystemOneResponse(model, Dict{String,UniLM.SystemOneAnswer}(),
            ju, nothing, Dict{String,Any}()))
        @test !haskey(DEFAULT_PRICING, "jev-1.14.0")
        @test estimated_cost(s1("jev-1.14.0")) ≈ 0.042
        @test estimated_cost(s1("jev-1.14.0")) == estimated_cost(s1("jev-latest")) == estimated_cost(s1("jev-1.13.0"))
        @test estimated_cost(s1("jev-latest"); model="jev-2.0.10") ≈ 0.042
        @test estimated_cost(LLMSuccess(message=m, self=Chat(model="jev-2.0.10"), usage=ju)) ≈ 0.042
        for other in ("jev-9", "jev-1.14", "jev-1.14.0-rc1", "xjev-1.14.0")
            @test estimated_cost(s1(other)) == 0.0
        end
        # A caller-supplied table without a jev-latest row has no family rate to fall back to.
        @test estimated_cost(s1("jev-1.14.0"); pricing=Dict("gpt-5.4" => DEFAULT_PRICING["gpt-5.4"])) == 0.0
        # A dated snapshot suffix is stripped before the lookup: the base version's row applies.
        @test estimated_cost(s1("jev-1.13.0-2026-09-22")) == estimated_cost(s1("jev-1.13.0")) > 0
    end
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
    chat = Chat(model="gpt-4.1-nano")
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
    chat = Chat(model="gpt-4.1-nano")
    chat._cumulative_cost[] = 1.5
    forked = fork(chat)
    @test cumulative_cost(forked) ≈ 1.5

    forked._cumulative_cost[] += 0.5
    @test cumulative_cost(forked) ≈ 2.0
    @test cumulative_cost(chat) ≈ 1.5  # original unchanged
end

@testset "estimated_cost — ResponseSuccess" begin
    ro = UniLM.ResponseObject(
        id="resp_1", status="completed", model="gpt-4.1-nano",
        output=Any[],
        usage=Dict{String,Any}("input_tokens" => 1_000_000, "output_tokens" => 1_000_000, "total_tokens" => 2_000_000),
        raw=Dict{String,Any}()
    )
    rs = UniLM.ResponseSuccess(response=ro)
    cost = estimated_cost(rs)
    # gpt-4.1-nano: input=0.1/1M, output=0.4/1M => 0.1 + 0.4 = 0.5
    @test cost ≈ 0.5
end

@testset "DEFAULT_PRICING" begin
    @test haskey(DEFAULT_PRICING, "gpt-5.2")
    @test haskey(DEFAULT_PRICING, "gpt-4.1")
    @test haskey(DEFAULT_PRICING, "gpt-4.1-mini")
    @test haskey(DEFAULT_PRICING, "gpt-4.1-nano")
    @test haskey(DEFAULT_PRICING, "o3")
    @test haskey(DEFAULT_PRICING, "o4-mini")

    # Verify structure + pin the live-verified gpt-5.2 rates (2026-06-21) so a revert is caught
    p = DEFAULT_PRICING["gpt-5.2"]
    @test p.input ≈ 1.75 / 1_000_000
    @test p.cached_input ≈ 0.175 / 1_000_000
    @test p.output ≈ 14.0 / 1_000_000

    # OpenAI pricing page, 2026-09-22 (USD per 1M tokens): 1M fresh input + 1M output,
    # then 1M input served entirely from the cache.
    m = Message(role=UniLM.RoleAssistant, content="hi")
    fresh = TokenUsage(prompt_tokens=1_000_000, completion_tokens=1_000_000, total_tokens=2_000_000)
    cached = TokenUsage(prompt_tokens=1_000_000, cached_tokens=1_000_000, total_tokens=1_000_000)
    for (model, input_and_output, cached_input) in (("gpt-6-sol", 2.0 + 10.0, 0.20),
            ("gpt-6-luna", 0.10 + 0.50, 0.01), ("gpt-5.4-nano", 0.20 + 1.25, 0.02))
        @test estimated_cost(LLMSuccess(message=m, self=Chat(; model), usage=fresh)) ≈ input_and_output
        @test estimated_cost(LLMSuccess(message=m, self=Chat(; model), usage=cached)) ≈ cached_input
    end
end

@testset "estimated_cost — cached input + new rows" begin
    chat = Chat(model="gpt-5.4")
    m = Message(role=UniLM.RoleAssistant, content="hi")
    rates = DEFAULT_PRICING["gpt-5.4"]
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

    # An unpriced embedding model (resolved from embeddings.model, not in DEFAULT_PRICING) → 0.
    emb_unk = Embeddings("hello"; model="text-embedding-unknown")
    es_unk = EmbeddingSuccess(embeddings=emb_unk, usage=u, raw=Dict{String,Any}())
    @test estimated_cost(es_unk) == 0.0
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
    @test_throws ArgumentError estimated_cost(speech; model="gpt-4.1")
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
    chat = Chat(model="gpt-4.1")
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
    @test estimated_cost(covered[1]) ≈ (1000 * 2.0 + 500 * 8.0) / 1_000_000
    @test estimated_cost(covered[2]) == 0.0     # a failure is still a priced zero
end

@testset "Gemini pricing rows" begin
    @test haskey(UniLM.DEFAULT_PRICING, "gemini-3.5-flash")
    @test haskey(UniLM.DEFAULT_PRICING, "gemini-3.1-flash-lite")
    # 1000 prompt + 500 output on gemini-3.1-flash-lite = 1000*0.25/1e6 + 500*1.5/1e6
    chat = Chat(service=GEMINIServiceEndpoint, model="gemini-3.1-flash-lite")
    usage = UniLM.TokenUsage(prompt_tokens=1000, completion_tokens=500, total_tokens=1500)
    result = UniLM.LLMSuccess(message=Message(role=RoleAssistant, content="x"), self=chat, usage=usage)
    @test UniLM.estimated_cost(result) ≈ (1000 * 0.25 + 500 * 1.5) / 1_000_000
end

@testset "Anthropic and DeepSeek price rows" begin
    # https://platform.claude.com/docs/en/about-claude/pricing.md (2026-09-24), USD per 1M tokens:
    # input, cache hit, output. https://api-docs.deepseek.com/quick_start/pricing: peak rates.
    for (model, i, c, o) in (("claude-fable-5-1", 10.0, 0.25, 50.0), ("claude-mythos-5-1", 10.0, 0.25, 50.0),
                             ("claude-fable-5", 10.0, 1.0, 50.0), ("claude-mythos-5", 10.0, 1.0, 50.0),
                             ("claude-opus-5-5", 4.0, 0.20, 20.0), ("claude-opus-5", 5.0, 0.50, 25.0),
                             ("claude-opus-4-8", 5.0, 0.50, 25.0), ("claude-opus-4-7", 5.0, 0.50, 25.0),
                             ("claude-opus-4-6", 5.0, 0.50, 25.0), ("claude-opus-4-5", 5.0, 0.50, 25.0),
                             ("claude-sonnet-5", 2.0, 0.20, 10.0), ("claude-sonnet-4-6", 3.0, 0.30, 15.0),
                             ("claude-sonnet-4-5", 3.0, 0.30, 15.0), ("claude-haiku-4-5", 1.0, 0.10, 5.0),
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
    @test cost("claude-sonnet-4-5-20250929") ≈ 18.0
    @test cost("claude-haiku-4-5-2025100") == 0.0          # seven digits are not a date suffix
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
