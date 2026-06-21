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
