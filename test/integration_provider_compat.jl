# Low-cost live coverage for current Gemini wire contracts.
if get(ENV, "UNILM_LIVE", "") == "1" && haskey(ENV, "OPENAI_API_KEY")
    @testset "OpenAI current reasoning, caching, and Chat tools" begin
        result = respond(Respond(input="Reply with exactly: hello", model="gpt-5.6-luna",
            reasoning=Reasoning(effort="none", mode="standard", context="current_turn"),
            prompt_cache_options=PromptCacheOptions(mode="explicit", ttl="30m"), max_output_tokens=128))
        @test result isa ResponseSuccess
        @test occursin("hello", output_text(result))
        chat = Chat(model="gpt-5.6-luna", reasoning_effort="none", max_completion_tokens=128,
            tools=[Tool(func=FunctionSignature(name="ping", parameters=Dict("type" => "object", "properties" => Dict())))],
            tool_choice="required", messages=[Message(role=UniLM.RoleUser, content="Call ping.")])
        result = chatrequest!(chat)
        @test result isa LLMSuccess
        @test only(result.message.tool_calls).func.name == "ping"
    end

    @testset "OpenAI GPT-6 Luna Chat tools, moderation, prewarm, and deferred tools" begin
        ping = Tool(func=FunctionSignature(name="ping", parameters=Dict("type" => "object", "properties" => Dict())))
        result = chatrequest!(Chat(model="gpt-6-luna", reasoning_effort="none", max_completion_tokens=64,
            tools=[ping], tool_choice="required", messages=[Message(role=UniLM.RoleUser, content="Call ping.")]))
        @test result isa LLMSuccess
        if result isa LLMSuccess
            calls = something(result.message.tool_calls, ToolCall[])
            @test length(calls) == 1 && calls[1].func.name == "ping"
        end

        # First run (2026-09-22): the moderation result came back in r.response.raw["moderation"],
        # an object with "input" and "output" entries of type "moderation_result".
        result = respond("Reply with exactly: hello"; model="gpt-5.4-mini", max_output_tokens=64,
            moderation=ModerationConfig(model="omni-moderation-latest"))
        @test result isa ResponseSuccess
        if result isa ResponseSuccess
            moderation = result.response.raw["moderation"]
            @test moderation["input"]["type"] == moderation["output"]["type"] == "moderation_result"
            @test moderation["input"]["model"] == "omni-moderation-latest"
        end

        # First run (2026-09-22): prewarm returned HTTP 200 decoded as ResponseSuccess with
        # status "completed", an empty output array, and 0 output tokens.
        result = respond("Reply with exactly: hello"; model="gpt-5.6-luna", max_output_tokens=64,
            prompt_cache_options=PromptCacheOptions(mode="explicit", ttl="30m", prewarm=true))
        @test result isa ResponseSuccess
        if result isa ResponseSuccess
            @test result.response.status == "completed"
            @test isempty(result.response.output)
        end

        # First run (2026-09-22): a deferred function tool without a tool_search tool is
        # rejected with HTTP 400 "Deferred tools require tools.tool_search."
        result = respond("Say hi"; model="gpt-5.4-mini", max_output_tokens=64,
            tools=[FunctionTool(name="noop", description="does nothing",
                parameters=Dict("type" => "object", "properties" => Dict()), defer_loading=true)])
        @test result isa ResponseFailure && result.status == 400 && occursin("tool_search", result.response)
    end
end

if get(ENV, "UNILM_LIVE", "") == "1" && haskey(ENV, "GEMINI_API_KEY")
    @testset "Gemini streamed JSON Schema tool round-trip" begin
        schema = Dict("type" => "object", "properties" => Dict("city" => Dict("type" => "string")),
                      "required" => ["city"], "additionalProperties" => false)
        chat = Chat(service=GEMINIServiceEndpoint, model="gemini-3.8-flash", reasoning_effort="low",
            max_tokens=1024, stream=true, tool_choice="required",
            tools=[Tool(func=FunctionSignature(name="weather", parameters=schema))],
            messages=[Message(role=UniLM.RoleUser, content="Call weather for Oslo.")])
        result = fetch(chatrequest!(chat))
        @test result isa LLMSuccess
        @test result.message.provider_content isa ProviderContent
        calls = result.message.tool_calls
        @test !isnothing(calls) && length(calls) == 1
        if !isnothing(calls) && length(calls) == 1
            @test calls[1].func.arguments["city"] == "Oslo"
            push!(chat, Message(role=UniLM.RoleTool, tool_call_id=calls[1].id, content="Sunny, 22C"))
            follow = Chat(service=chat.service, model=chat.model, messages=chat.messages,
                tools=chat.tools, reasoning_effort="low", max_tokens=1024)
            result = chatrequest!(follow)
            @test result isa LLMSuccess
            @test occursin("22", something(result.message.content, ""))
        end
    end

    @testset "Gemini OpenAI-compatible chat and embeddings" begin
        result = chatrequest!(Chat(service=GEMINIOpenAIServiceEndpoint, model="gemini-3.8-flash",
            reasoning_effort="low", max_completion_tokens=1024,
            messages=[Message(role=UniLM.RoleUser, content="Reply with exactly: hello")]))
        @test result isa LLMSuccess
        @test occursin("hello", lowercase(something(result.message.content, "")))
        result = embeddingrequest!(Embeddings("hello"; service=GEMINIOpenAIServiceEndpoint))
        @test result isa EmbeddingSuccess
    end

    capital_schema = Dict("type" => "object",
        "properties" => Dict("city" => Dict("type" => "string"), "country" => Dict("type" => "string")),
        "required" => ["city", "country"], "additionalProperties" => false)

    @testset "Gemini native structured output (response_format)" begin
        result = chatrequest!(Chat(service=GEMINIServiceEndpoint, model="gemini-3.8-flash",
            reasoning_effort="low", max_tokens=1024,
            response_format=UniLM.json_schema("capital", "A capital city and its country", capital_schema),
            messages=[Message(role=UniLM.RoleUser, content="Give the capital of Norway as JSON.")]))
        @test result isa LLMSuccess
        if result isa LLMSuccess
            parsed = JSON.parse(something(result.message.content, ""))
            @info "Gemini native structured output" json = JSON.json(parsed)
            @test haskey(parsed, "city") && haskey(parsed, "country")
        end
    end

    @testset "Gemini Interactions structured output (text → response_format)" begin
        result = respond("Give the capital of Norway as JSON."; service=GEMINIServiceEndpoint,
            model="gemini-3.8-flash", max_output_tokens=256,
            text=json_schema_format("capital", "A capital city and its country", capital_schema))
        @test result isa ResponseSuccess
        if result isa ResponseSuccess
            parsed = JSON.parse(output_text(result))
            @info "Gemini Interactions structured output" json = JSON.json(parsed)
            @test haskey(parsed, "city") && haskey(parsed, "country")
        end
    end

    @testset "Gemini native safety_identifier (request labels)" begin
        result = chatrequest!(Chat(service=GEMINIServiceEndpoint, model="gemini-3.8-flash",
            reasoning_effort="low", max_tokens=1024, safety_identifier="unilm-live-witness",
            messages=[Message(role=UniLM.RoleUser, content="Reply with exactly: hello")]))
        @test result isa LLMSuccess
    end

    @testset "Gemini embedding-2 through the OpenAI-compatible endpoint" begin
        result = embeddingrequest!(Embeddings("hello"; service=GEMINIOpenAIServiceEndpoint,
            model="gemini-embedding-2"))
        @info "gemini-embedding-2 via GEMINIOpenAIServiceEndpoint" result_type = typeof(result)
        @test result isa EmbeddingSuccess
    end
else
    @info "Skipping current Gemini wire integration tests (set UNILM_LIVE=1 and GEMINI_API_KEY)"
end
