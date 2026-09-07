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
else
    @info "Skipping current Gemini wire integration tests (set UNILM_LIVE=1 and GEMINI_API_KEY)"
end
