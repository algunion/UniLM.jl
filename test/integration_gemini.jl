# ─── Gemini Integration Tests (live) ─────────────────────────────────────────
# Requires UNILM_LIVE=1 and GEMINI_API_KEY (billing-enabled). Uses gemini-3.8-flash — a
# thinking model: max_tokens budgets include thought tokens, so limits carry reasoning
# headroom (a tight budget yields an empty-text turn). Run once when green; do not rerun.

if !haskey(ENV, "GEMINI_API_KEY") || get(ENV, "UNILM_LIVE", "") != "1"
    @info "Skipping Gemini integration tests (set UNILM_LIVE=1 and GEMINI_API_KEY to run live)"
else

@testset "Gemini Chat — basic" begin
    chat = Chat(service=UniLM.GEMINIServiceEndpoint, model="gemini-3.8-flash", reasoning_effort="low", max_tokens=1024)
    push!(chat, Message(Val(:system), "You are a helpful assistant."))
    push!(chat, Message(Val(:user), "Reply with exactly: hello"))
    result = chatrequest!(chat)
    @test result isa LLMSuccess
    @test !isempty(result.message.content)
    @test result.usage.completion_tokens > 0
    @test cumulative_cost(chat) > 0.0
end

@testset "Gemini Chat — tool round-trip" begin
    sig = FunctionSignature(name="get_current_weather",
        description="Get the current weather for a location",
        parameters=Dict("type" => "object",
            "properties" => Dict("location" => Dict("type" => "string", "description" => "City name")),
            "required" => ["location"]))
    chat = Chat(service=UniLM.GEMINIServiceEndpoint, model="gemini-3.8-flash", reasoning_effort="low", max_tokens=2048,
                tools=[Tool(func=sig)], tool_choice="required")
    push!(chat, Message(Val(:system), "Use the weather tool when asked about weather."))
    push!(chat, Message(Val(:user), "What is the weather in Paris?"))
    result = chatrequest!(chat)
    @test result isa LLMSuccess
    m = result.message
    @test m.finish_reason == UniLM.TOOL_CALLS
    if m.finish_reason == UniLM.TOOL_CALLS
        @test m.tool_calls[1].func.name == "get_current_weather"
        @test haskey(m.tool_calls[1].func.arguments, "location")
        # feed the tool result back — exercises thoughtSignature echo on the next turn
        push!(chat, Message(role=UniLM.RoleTool, tool_call_id=m.tool_calls[1].id, content="72F and sunny"))
        follow_chat = Chat(service=GEMINIServiceEndpoint, model="gemini-3.8-flash",
            reasoning_effort="low", max_tokens=1024, messages=chat.messages, tools=chat.tools)
        follow = chatrequest!(follow_chat)
        @test follow isa LLMSuccess
        @test !isempty(something(follow.message.content, ""))
    end
end

@testset "Gemini Chat — streaming" begin
    payloads = Any[]
    chat = Chat(service=UniLM.GEMINIServiceEndpoint, model="gemini-3.8-flash", reasoning_effort="low",
                max_tokens=1024, stream=true)
    push!(chat, Message(Val(:system), "You are helpful."))
    push!(chat, Message(Val(:user), "Count from 1 to 10, one number per line."))
    task = chatrequest!(chat; callback=(c, _) -> push!(payloads, c))
    result = fetch(task)
    @test result isa LLMSuccess
    @test !isempty(result.message.content)   # handle_sse_event! accumulated real SSE
    @test !isempty(payloads)                 # callback fired (deltas and/or final message)
    @test result.message.provider_content isa ProviderContent
    push!(chat, Message(Val(:user), "Reply with exactly: done"))
    follow = fetch(chatrequest!(chat))
    @test follow isa LLMSuccess
    @test occursin("done", lowercase(something(follow.message.content, "")))
end

end  # if GEMINI_API_KEY
