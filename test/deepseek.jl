# DeepSeek thinking-mode round trip — deterministic, zero-spend unit tests.
# https://api-docs.deepseek.com/guides/thinking_mode: DeepSeek models think by default and return
# the chain of thought as `reasoning_content` beside `content` (`delta.reasoning_content` when
# streaming). A request that carries `tools` must pass back the reasoning_content of every earlier
# assistant turn, turns without tool calls included, or the API answers 400; without tools it is not
# needed and the API ignores it.
using UniLM
using UniLM: encode_request, decode_response, StreamState, _build_stream_message,
             RoleAssistant, RoleTool, TOOL_CALLS, GPTFunction
using Test, HTTP, JSON

const _DS = DeepSeekEndpoint("k")
const _DS_TOOLS = [Tool(func=FunctionSignature(name="get_weather",
    parameters=Dict("type" => "object", "properties" => Dict("city" => Dict("type" => "string")))))]
_ds_resp(d) = HTTP.Response(200, [], Vector{UInt8}(JSON.json(d)))
_ds_reasoning(m::Message) = m.provider_content isa ProviderContent && m.provider_content.provider === :deepseek ?
    m.provider_content.blocks : nothing

const _DS_TOOL_TURN = Dict("model" => "deepseek-flash", "choices" => [Dict("index" => 0, "finish_reason" => "tool_calls",
    "message" => Dict("role" => "assistant", "content" => nothing, "reasoning_content" => "Need the weather tool.",
        "tool_calls" => [Dict("id" => "call_1", "type" => "function",
            "function" => Dict("name" => "get_weather", "arguments" => "{\"city\":\"Oslo\"}"))]))],
    "usage" => Dict("prompt_tokens" => 10, "completion_tokens" => 20, "total_tokens" => 30))
const _DS_ANSWER_TURN = Dict("model" => "deepseek-flash", "choices" => [Dict("index" => 0, "finish_reason" => "stop",
    "message" => Dict("role" => "assistant", "content" => "It is 12C in Oslo.",
                      "reasoning_content" => "The tool said 12C."))])

@testset "decode — reasoning_content is captured as DeepSeek provider content" begin
    dec = decode_response(_DS, _ds_resp(_DS_TOOL_TURN))
    m = dec.message
    @test _ds_reasoning(m) == Any[Dict{String,Any}("reasoning_content" => "Need the weather tool.")]
    # The neutral fields are the OpenAI decoder's, unchanged.
    @test m.finish_reason == TOOL_CALLS && m.tool_calls[1].id == "call_1" &&
          m.tool_calls[1].func.arguments == Dict{String,Any}("city" => "Oslo") && isnothing(m.content)
    @test dec.usage.total_tokens == 30
    # No reasoning (thinking disabled, or an empty one) → no capture.
    for rc in (nothing, "")
        plain = Dict("choices" => [Dict("index" => 0, "finish_reason" => "stop",
            "message" => Dict("role" => "assistant", "content" => "hi", "reasoning_content" => rc))])
        @test isnothing(decode_response(_DS, _ds_resp(plain)).message.provider_content)
    end
    # A 200 without choices is still the OpenAI decoder's error.
    @test_throws ErrorException decode_response(_DS, _ds_resp(Dict("choices" => [])))
end

@testset "encode — reasoning_content is passed back on assistant turns when the request carries tools" begin
    tc = ToolCall(id="call_1", func=GPTFunction("get_weather", Dict{String,Any}("city" => "Oslo")))
    r(text) = ProviderContent(:deepseek, Any[Dict{String,Any}("reasoning_content" => text)])
    msgs = [Message(Val(:system), "s"), Message(Val(:user), "weather in Oslo?"),
            Message(role=RoleAssistant, tool_calls=[tc], finish_reason=TOOL_CALLS, provider_content=r("Need the weather tool.")),
            Message(role=RoleTool, tool_call_id="call_1", content="12C"),
            Message(role=RoleAssistant, content="It is 12C in Oslo.", provider_content=r("The tool said 12C.")),
            Message(Val(:user), "and tomorrow?")]
    chat = Chat(service=_DS, model="deepseek-flash", tools=_DS_TOOLS, messages=msgs)
    golden = JSON.parse(JSON.json(chat))                      # the OpenAI wire …
    golden["messages"][3]["reasoning_content"] = "Need the weather tool."   # … plus every turn's reasoning,
    golden["messages"][5]["reasoning_content"] = "The tool said 12C."       # a turn without tool calls too
    @test JSON.parse(encode_request(_DS, chat)) == golden
    # Without tools the reasoning is not needed and the API ignores it: the plain OpenAI body.
    chat2 = Chat(service=_DS, model="deepseek-flash", messages=msgs)
    @test encode_request(_DS, chat2) == JSON.json(chat2)
    # Another provider's captured blocks are never sent on the DeepSeek wire.
    other = [msgs[1:2]; Message(role=RoleAssistant, content="x", provider_content=ProviderContent(:anthropic,
             Any[Dict{String,Any}("type" => "text", "text" => "x")])); Message(Val(:user), "y")]
    chat3 = Chat(service=_DS, model="deepseek-flash", tools=_DS_TOOLS, messages=other)
    @test encode_request(_DS, chat3) == JSON.json(chat3)
end

@testset "tool loop — decoded turns round-trip their reasoning on every later request" begin
    chat = Chat(service=_DS, model="deepseek-flash", tools=_DS_TOOLS)
    push!(chat, Message(Val(:system), "Use the weather tool."))
    push!(chat, Message(Val(:user), "weather in Oslo?"))
    push!(chat, decode_response(_DS, _ds_resp(_DS_TOOL_TURN)).message)
    push!(chat, Message(role=RoleTool, tool_call_id="call_1", content="12C"))
    wire = JSON.parse(encode_request(_DS, chat))["messages"]                  # sub-request of turn 1
    @test [get(m, "reasoning_content", nothing) for m in wire] == [nothing, nothing, "Need the weather tool.", nothing]
    push!(chat, decode_response(_DS, _ds_resp(_DS_ANSWER_TURN)).message)
    push!(chat, Message(Val(:user), "and tomorrow?"))
    wire = JSON.parse(encode_request(_DS, chat))["messages"]                  # turn 2 still carries turn 1's
    @test [get(m, "reasoning_content", nothing) for m in wire] ==
          [nothing, nothing, "Need the weather tool.", nothing, "The tool said 12C.", nothing]
    @test wire[3]["tool_calls"][1]["id"] == "call_1" && wire[5]["content"] == "It is 12C in Oslo."
end

_ds_stream(chunks; done=true) =
    join(("data: " * JSON.json(c) for c in chunks), "\n") * (done ? "\ndata: [DONE]\n" : "\n")
const _DS_STREAM = [
    Dict("choices" => [Dict("index" => 0, "delta" => Dict("role" => "assistant", "reasoning_content" => "Need the "),
                            "finish_reason" => nothing)]),
    Dict("choices" => [Dict("index" => 0, "delta" => Dict("reasoning_content" => "weather tool."), "finish_reason" => nothing)]),
    Dict("choices" => [Dict("index" => 0, "delta" => Dict("tool_calls" => [Dict("index" => 0, "id" => "call_1", "type" => "function",
        "function" => Dict("name" => "get_weather", "arguments" => "{\"city\":"))]), "finish_reason" => nothing)]),
    Dict("choices" => [Dict("index" => 0, "delta" => Dict("tool_calls" => [Dict("index" => 0,
        "function" => Dict("arguments" => "\"Oslo\"}"))]), "finish_reason" => "tool_calls")]),
    Dict("choices" => Any[], "usage" => Dict("prompt_tokens" => 10, "completion_tokens" => 20, "total_tokens" => 30)),
]

@testset "stream — delta.reasoning_content is captured complete, like the non-streamed turn" begin
    st = StreamState()
    @test UniLM._sse_dispatch!(_DS, IOBuffer(), Ref(""), _ds_stream(_DS_STREAM), st) === :done
    @test String(take!(st.pending_delta)) == ""              # reasoning is not answer text
    streamed = _build_stream_message(st)
    decoded = decode_response(_DS, _ds_resp(_DS_TOOL_TURN)).message
    @test _ds_reasoning(streamed) == _ds_reasoning(decoded)
    @test streamed.tool_calls[1].func.arguments == decoded.tool_calls[1].func.arguments
    # The streamed turn encodes exactly like the non-streamed one.
    enc(m) = encode_request(_DS, Chat(service=_DS, model="deepseek-flash", tools=_DS_TOOLS,
        messages=[Message(Val(:user), "weather in Oslo?"), m, Message(role=RoleTool, tool_call_id="call_1", content="12C")]))
    @test enc(streamed) == enc(decoded)
    # A stream that ends at EOF after its finish_reason (no [DONE]) still yields the complete capture.
    st2 = StreamState()
    @test UniLM._sse_dispatch!(_DS, IOBuffer(), Ref(""), _ds_stream(_DS_STREAM; done=false), st2) === :continue
    @test _ds_reasoning(_build_stream_message(st2)) == _ds_reasoning(decoded)
    # A truncated stream (no finish_reason, no [DONE]) leaves the capture incomplete: no provider content.
    st3 = StreamState()
    UniLM._sse_dispatch!(_DS, IOBuffer(), Ref(""), _ds_stream(_DS_STREAM[1:2]; done=false), st3)
    @test isnothing(_build_stream_message(st3).provider_content)
end
