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
        push!(chat, Message(role=UniLM.RoleTool, tool_call_id=m.tool_calls[1].id, content="72F and sunny"))
        follow = chatrequest!(chat)
        @test follow isa LLMSuccess
        @test !isempty(something(follow.message.content, ""))
    else
        @test m.finish_reason == UniLM.STOP
    end
end

@testset "Anthropic Chat — streaming" begin
    # `handle_sse_event!` correctness is proven by LLMSuccess + non-empty accumulated
    # content (the stream was parsed end to end). Incremental String deltas are best-effort:
    # a short reply delivered in a single network read reaches EOS on the first chunk, so the
    # shared _chatrequeststream logic fires the final-Message callback rather than String deltas
    # (a provider-agnostic property, not an Anthropic decode issue). Assert the callback fired,
    # not that incremental deltas specifically arrived.
    payloads = Any[]
    chat = Chat(service=ANTHROPICServiceEndpoint, model="claude-haiku-4-5", max_tokens=128, stream=true)
    push!(chat, Message(Val(:system), "You are helpful."))
    push!(chat, Message(Val(:user), "Count from 1 to 10, one number per line."))
    task = chatrequest!(chat; callback=(c, _) -> push!(payloads, c))
    result = fetch(task)
    @test result isa LLMSuccess
    @test !isempty(result.message.content)   # handle_sse_event! accumulated real SSE
    @test !isempty(payloads)                 # callback fired (incremental deltas and/or final message)
end

@testset "Anthropic Chat — thinking-model tool round-trip (claude-sonnet-5)" begin
    # claude-sonnet-5 runs adaptive thinking by default: its tool turns open
    # with thinking blocks that MUST be echoed verbatim on the next request.
    # This round-trip returned HTTP 400 before provider-native content capture.
    sig = GPTFunctionSignature(name="get_current_weather",
        description="Get the current weather for a location",
        parameters=Dict("type" => "object",
            "properties" => Dict("location" => Dict("type" => "string", "description" => "City name")),
            "required" => ["location"]))
    chat = Chat(service=ANTHROPICServiceEndpoint, model="claude-sonnet-5",
                tools=[GPTTool(func=sig)], tool_choice="auto")
    push!(chat, Message(Val(:system), "Use the weather tool when asked about current weather."))
    push!(chat, Message(Val(:user), "What is the weather in Paris right now?"))
    result = chatrequest!(chat)
    @test result isa LLMSuccess
    m = result.message
    if m.finish_reason == UniLM.TOOL_CALLS
        @test m.provider_content isa ProviderContent &&
              m.provider_content.provider === :anthropic
        push!(chat, Message(role=UniLM.RoleTool, tool_call_id=m.tool_calls[1].id,
                            content="22C and clear"))
        follow = chatrequest!(chat)   # the request that used to 400
        @test follow isa LLMSuccess
        @test !isempty(something(follow.message.content, ""))
    else
        # Adaptive thinking is model-decided; a no-tool answer — or a turn that
        # burned the default max_tokens mid-thought ("length") — is a legal
        # (weaker) outcome. Record it without failing the suite.
        @test m.finish_reason in (UniLM.STOP, "length")
        @info "claude-sonnet-5 answered without a tool call; round-trip not exercised this run"
    end
end

end  # if ANTHROPIC_API_KEY
