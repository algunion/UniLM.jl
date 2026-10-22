# ─── Anthropic Integration Tests (live) ──────────────────────────────────────
# Requires UNILM_LIVE=1 and ANTHROPIC_API_KEY. Minimal tokens on claude-haiku-5-5, the cheapest
# current Claude model ($0.10 input / $0.50 output per 1M tokens); one short call on
# claude-sonnet-5-5 for the rule that is its own, and a thinking tool round-trip on
# claude-sonnet-5, which thinks before the tool call it chooses.

if !haskey(ENV, "ANTHROPIC_API_KEY") || get(ENV, "UNILM_LIVE", "") != "1"
    @info "Skipping Anthropic integration tests (set UNILM_LIVE=1 and ANTHROPIC_API_KEY to run live)"
else

const _LIVE_CLAUDE = "claude-haiku-5-5"
const _LIVE_WEATHER = Tool(func=FunctionSignature(name="get_current_weather",
    description="Get the current weather for a location",
    parameters=Dict("type" => "object",
        "properties" => Dict("location" => Dict("type" => "string", "description" => "City name")),
        "required" => ["location"])))
function _live_claude(user; system="You are a helpful assistant.", kw...)
    chat = Chat(; service=ANTHROPICServiceEndpoint, model=_LIVE_CLAUDE, kw...)
    push!(chat, Message(Val(:system), system))
    push!(chat, Message(Val(:user), user))
    chat
end
_block_types(m::Message) = [b["type"] for b in m.provider_content.blocks]
# One log line per billed call: the served model, tokens in and out, the estimated cost.
_billed(r) = (r isa LLMSuccess && @info "billed" r.model r.usage.prompt_tokens r.usage.completion_tokens cost = estimated_cost(r); r)

@testset "Anthropic Chat — a basic call reports its served model and is priced at its row" begin
    chat = _live_claude("Reply with exactly: hello"; max_tokens=64, reasoning_effort="none")
    r = _billed(chatrequest!(chat))
    @test r isa LLMSuccess
    @test startswith(something(r.model, ""), _LIVE_CLAUDE)
    u = r.usage
    @test u.prompt_tokens > 0 && u.completion_tokens > 0
    @test (u.cached_tokens, u.cache_write_tokens, u.service_tier, u.inference_geo) == (0, 0, "standard", "global")
    # claude-haiku-5-5: $0.10 per 1M input tokens, $0.50 per 1M output tokens.
    @test isapprox(estimated_cost(r), (u.prompt_tokens * 0.10 + u.completion_tokens * 0.50) / 1e6; rtol=1e-12)
    @test cumulative_cost(chat) == estimated_cost(r)
end

@testset "Anthropic Chat — a forced tool call, then its result" begin
    chat = _live_claude("What is the weather in Paris?"; system="Use the weather tool when asked about weather.",
                        max_tokens=256, tools=[_LIVE_WEATHER], tool_choice=UniLM.GPTToolChoice(func="get_current_weather"))
    r = _billed(chatrequest!(chat))
    @test r isa LLMSuccess
    m = r.message
    @test m.finish_reason == UniLM.TOOL_CALLS
    @test [tc.func.name for tc in m.tool_calls] == ["get_current_weather"]
    @test haskey(m.tool_calls[1].func.arguments, "location")
    push!(chat, Message(role=UniLM.RoleTool, tool_call_id=m.tool_calls[1].id, content="22C and sunny"))
    @test _billed(chatrequest!(chat)) isa LLMSuccess
end

@testset "Anthropic Chat — a streamed call reports the usage of the same call unstreamed" begin
    call(stream) = (chat = _live_claude("Count from 1 to 5, separated by spaces."; max_tokens=64,
                                        reasoning_effort="none", stream);
                    _billed(stream ? fetch(chatrequest!(chat; callback=(_, _) -> nothing)) : chatrequest!(chat)))
    plain, streamed = call(false), call(true)
    @test plain isa LLMSuccess && streamed isa LLMSuccess
    @test streamed.usage.prompt_tokens == plain.usage.prompt_tokens
    @test (streamed.usage.service_tier, streamed.usage.inference_geo) == (plain.usage.service_tier, plain.usage.inference_geo)
    @test streamed.model == plain.model
    @test !isempty(streamed.message.content)
end

@testset "Anthropic Chat — tool turns go back verbatim after the tool result" begin
    # Claude Haiku 5.5 answers a forced tool_choice with the tool call and no thinking block
    # (https://platform.claude.com/docs/en/models/haiku-5-5/migration-guide.md); the captured
    # blocks are what the follow-up sends back.
    chat = _live_claude("What is the weather in Oslo?"; system="Use the weather tool when asked about weather.",
                        max_tokens=256, reasoning_effort="low", tools=[_LIVE_WEATHER],
                        tool_choice=UniLM.GPTToolChoice(func="get_current_weather"))
    r = _billed(chatrequest!(chat))
    @test r isa LLMSuccess
    m = r.message
    @test m.finish_reason == UniLM.TOOL_CALLS && _block_types(m) == ["tool_use"]
    push!(chat, Message(role=UniLM.RoleTool, tool_call_id=m.tool_calls[1].id, content="4C and cloudy"))
    body = JSON.parse(UniLM.encode_request(ANTHROPICServiceEndpoint, chat))
    @test body["messages"][2]["content"] == m.provider_content.blocks
    @test _billed(chatrequest!(chat)) isa LLMSuccess
end

@testset "Anthropic Chat — signed thinking blocks survive a tool round-trip (claude-sonnet-5)" begin
    # A thinking model that chooses the tool: Claude Haiku 5.5 called it without thinking in
    # three of three tries at effort "high" (2026-10-22); Claude Sonnet 5 thinks, then calls it.
    chat = Chat(service=ANTHROPICServiceEndpoint, model="claude-sonnet-5", max_tokens=1024, reasoning_effort="high",
                tools=[_LIVE_WEATHER], tool_choice="auto")
    push!(chat, Message(Val(:system), "Always call the weather tool for a weather question; think first."))
    push!(chat, Message(Val(:user), "Is it colder in Paris right now than the October average? Use the tool."))
    r = _billed(chatrequest!(chat))
    @test r isa LLMSuccess
    m = r.message
    @test m.finish_reason == UniLM.TOOL_CALLS && _block_types(m) == ["thinking", "tool_use"]
    @test !isempty(m.provider_content.blocks[1]["signature"])
    push!(chat, Message(role=UniLM.RoleTool, tool_call_id=m.tool_calls[1].id, content="9C, light rain"))
    body = JSON.parse(UniLM.encode_request(ANTHROPICServiceEndpoint, chat))
    @test body["messages"][2]["content"] == m.provider_content.blocks
    @test _billed(chatrequest!(chat)) isa LLMSuccess
end

@testset "Anthropic Chat — a turn cut at max_tokens with thinking only goes back as returned" begin
    chat = _live_claude("Is 1009 prime? Check every candidate divisor carefully before you answer.";
                        system="You are terse.", max_tokens=16, reasoning_effort="high")
    r = _billed(chatrequest!(chat))
    @test r isa LLMSuccess
    @test r.message.finish_reason == "length" && r.message.content == "" && _block_types(r.message) == ["thinking"]
    push!(chat, Message(Val(:user), "Answer with yes or no only."))
    follow = Chat(service=ANTHROPICServiceEndpoint, model=_LIVE_CLAUDE, max_tokens=256, reasoning_effort="low",
                  messages=chat.messages)
    body = JSON.parse(UniLM.encode_request(ANTHROPICServiceEndpoint, follow))
    @test [m["role"] for m in body["messages"]] == ["user", "assistant", "user"]
    @test _billed(chatrequest!(follow)) isa LLMSuccess
end

@testset "Anthropic Chat — a system message pushed after a reply, then a user turn" begin
    # Claude Haiku 5.5 rejects a thinking block whose prefix changed; the system message goes
    # after the next user turn, never into the top-level system prompt.
    chat = _live_claude("Is 391 prime? Think it through, then answer yes or no."; system="You are terse.",
                        max_tokens=1024, reasoning_effort="high")
    r1 = _billed(chatrequest!(chat))
    @test r1 isa LLMSuccess
    @test "thinking" in _block_types(r1.message)          # a block for the prefix check to check
    push!(chat, Message(Val(:system), "From now on, answer in French."))
    push!(chat, Message(Val(:user), "Is 397 prime?"))
    body = JSON.parse(UniLM.encode_request(ANTHROPICServiceEndpoint, chat))
    @test body["system"] == "You are terse." && body["messages"][end]["role"] == "system"
    @test _billed(chatrequest!(chat)) isa LLMSuccess
end

@testset "Anthropic Chat — Claude Sonnet 5.5 turns up-front thinking off with between_tools" begin
    chat = Chat(service=ANTHROPICServiceEndpoint, model="claude-sonnet-5-5", max_tokens=16, reasoning_effort="none")
    push!(chat, Message(Val(:system), "You are terse."))
    push!(chat, Message(Val(:user), "Reply with exactly: hello"))
    @test JSON.parse(UniLM.encode_request(ANTHROPICServiceEndpoint, chat))["thinking"] == Dict("type" => "between_tools")
    r = _billed(chatrequest!(chat))
    @test r isa LLMSuccess
    @test startswith(something(r.model, ""), "claude-sonnet-5-5")
end

@testset "Anthropic Chat — refused locally, before any request" begin
    for (model, kw) in ((_LIVE_CLAUDE, (temperature=0.2,)),
                        ("claude-sonnet-5-5", (tools=[_LIVE_WEATHER], tool_choice="required")))
        chat = Chat(; service=ANTHROPICServiceEndpoint, model, max_tokens=16, kw...)
        push!(chat, Message(Val(:system), "You are terse."))
        push!(chat, Message(Val(:user), "Reply with exactly: hello"))
        @test_throws ArgumentError chatrequest!(chat)
        @test length(chat) == 2 && cumulative_cost(chat) == 0.0
    end
end

end  # if ANTHROPIC_API_KEY
