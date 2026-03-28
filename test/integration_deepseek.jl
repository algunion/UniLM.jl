# ─── DeepSeek Integration Tests ──────────────────────────────────────────────
# Requires DEEPSEEK_API_KEY environment variable

if !haskey(ENV, "DEEPSEEK_API_KEY")
    @info "Skipping DeepSeek integration tests (DEEPSEEK_API_KEY not set)"
else

@testset "DeepSeek Chat — basic" begin
    chat = Chat(service=DeepSeekEndpoint(), model="deepseek-chat")
    push!(chat, Message(Val(:system), "You are a helpful assistant."))
    push!(chat, Message(Val(:user), "Reply with exactly: hello"))
    result = chatrequest!(chat)
    @test result isa LLMSuccess
    @test !isempty(result.message.content)
end

@testset "DeepSeek FIM — basic" begin
    fim = FIMCompletion(
        service=DeepSeekEndpoint(),
        model="deepseek-chat",
        prompt="def fib(a):",
        suffix="    return fib(a-1) + fib(a-2)",
        max_tokens=128,
        stop=["\n\n"]
    )
    result = fim_complete(fim)
    @test result isa FIMSuccess
    @test !isempty(fim_text(result))
    @test result.response.usage.total_tokens > 0
    @info "FIM result: $(fim_text(result))"
end

@testset "DeepSeek FIM — convenience" begin
    result = fim_complete("def hello():",
        service=DeepSeekEndpoint(),
        max_tokens=64,
        stop=["\n\n"])
    @test result isa FIMSuccess
    @test !isempty(fim_text(result))
end

@testset "DeepSeek Prefix Completion" begin
    chat = Chat(service=DeepSeekEndpoint(), model="deepseek-chat")
    push!(chat, Message(Val(:system), "You are a helpful coding assistant."))
    push!(chat, Message(Val(:user), "Write a Python hello world"))
    push!(chat, Message(role=RoleAssistant, content="```python\n"))
    @test length(chat) == 3
    result = prefix_complete(chat)
    @test result isa LLMSuccess
    @test !isempty(result.message.content)
    # Verify chat history: prefix replaced with completed response, not rejected
    @test length(chat) == 3
    @test last(chat).role == UniLM.RoleAssistant
    @test last(chat).content == result.message.content
    @info "Prefix result: $(result.message.content)"
end

@testset "DeepSeek Chat — function calling" begin
    gptfsig = GPTFunctionSignature(
        name="get_current_weather",
        description="Get the current weather for a location",
        parameters=Dict(
            "type" => "object",
            "properties" => Dict(
                "location" => Dict("type" => "string", "description" => "City name"),
                "unit" => Dict("type" => "string", "enum" => ["celsius", "fahrenheit"])
            ),
            "required" => ["location"]
        )
    )

    chat = Chat(
        service=DeepSeekEndpoint(),
        model="deepseek-chat",
        tools=[GPTTool(func=gptfsig)],
        tool_choice="auto"
    )
    push!(chat, Message(Val(:system), "Use the provided tools when asked about weather."))
    push!(chat, Message(Val(:user), "What is the weather in Paris?"))

    result = chatrequest!(chat)
    @test result isa LLMSuccess
    m = result.message
    if m.finish_reason == UniLM.TOOL_CALLS
        @test !isnothing(m.tool_calls)
        @test length(m.tool_calls) >= 1
        @test m.tool_calls[1].func.name == "get_current_weather"
        @test haskey(m.tool_calls[1].func.arguments, "location")
    else
        @test m.finish_reason == UniLM.STOP
        @test !isnothing(m.content)
    end
end

@testset "DeepSeek Chat — JSON output" begin
    chat = Chat(
        service=DeepSeekEndpoint(),
        model="deepseek-chat",
        response_format=UniLM.json_object()
    )
    push!(chat, Message(Val(:system), "Always respond in JSON format."))
    push!(chat, Message(Val(:user), "Tell me a joke with keys 'setup' and 'punchline'."))

    result = chatrequest!(chat)
    @test result isa LLMSuccess
    parsed = JSON.parse(result.message.content)
    @test parsed isa AbstractDict
end

@testset "DeepSeek Chat — streaming" begin
    received_chunks = String[]

    chat = Chat(service=DeepSeekEndpoint(), model="deepseek-chat", stream=true)
    push!(chat, Message(Val(:system), "You are helpful."))
    push!(chat, Message(Val(:user), "Reply with exactly: hello"))

    callback = (chunk, close_ref) -> begin
        chunk isa String && push!(received_chunks, chunk)
    end
    task = chatrequest!(chat; callback=callback)
    result = fetch(task)

    @test result isa LLMSuccess
    @test !isempty(result.message.content)
    @test !isempty(received_chunks)
end

end  # if DEEPSEEK_API_KEY
