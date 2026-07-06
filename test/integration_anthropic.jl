# ─── Anthropic Integration Tests (live) ──────────────────────────────────────
# Requires ANTHROPIC_API_KEY. Uses claude-haiku-4-5 (cheapest) to minimize spend.

if !haskey(ENV, "ANTHROPIC_API_KEY")
    @info "Skipping Anthropic integration tests (ANTHROPIC_API_KEY not set)"
else

@testset "Anthropic Chat — basic" begin
    chat = Chat(service=ANTHROPICServiceEndpoint, model="claude-haiku-4-5", max_tokens=64)
    push!(chat, Message(Val(:system), "You are a helpful assistant."))
    push!(chat, Message(Val(:user), "Reply with exactly: hello"))
    result = chatrequest!(chat)
    @test result isa LLMSuccess
    @test !isempty(result.message.content)
    @test result.usage.completion_tokens > 0
    @test cumulative_cost(chat) > 0.0
end

@testset "Anthropic Chat — tool round-trip" begin
    sig = GPTFunctionSignature(name="get_current_weather",
        description="Get the current weather for a location",
        parameters=Dict("type" => "object",
            "properties" => Dict("location" => Dict("type" => "string", "description" => "City name")),
            "required" => ["location"]))
    chat = Chat(service=ANTHROPICServiceEndpoint, model="claude-haiku-4-5", max_tokens=256,
                tools=[GPTTool(func=sig)], tool_choice="auto")
    push!(chat, Message(Val(:system), "Use the weather tool when asked about weather."))
    push!(chat, Message(Val(:user), "What is the weather in Paris?"))
    result = chatrequest!(chat)
    @test result isa LLMSuccess
    m = result.message
    if m.finish_reason == UniLM.TOOL_CALLS
        @test m.tool_calls[1].func.name == "get_current_weather"
        @test haskey(m.tool_calls[1].func.arguments, "location")
        # feed the tool result back and continue the turn
        push!(chat, Message(role=RoleTool, tool_call_id=m.tool_calls[1].id, content="72F and sunny"))
        follow = chatrequest!(chat)
        @test follow isa LLMSuccess
        @test !isempty(something(follow.message.content, ""))
    else
        @test m.finish_reason == UniLM.STOP
    end
end

@testset "Anthropic Chat — streaming" begin
    chunks = String[]
    chat = Chat(service=ANTHROPICServiceEndpoint, model="claude-haiku-4-5", max_tokens=64, stream=true)
    push!(chat, Message(Val(:system), "You are helpful."))
    push!(chat, Message(Val(:user), "Reply with exactly: hello"))
    task = chatrequest!(chat; callback=(c, _) -> c isa String && push!(chunks, c))
    result = fetch(task)
    @test result isa LLMSuccess
    @test !isempty(result.message.content)
    @test !isempty(chunks)
end

end  # if ANTHROPIC_API_KEY
