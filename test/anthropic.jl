# Native Anthropic translation — deterministic, zero-spend unit tests.
using UniLM
using UniLM: encode_request, decode_response, StreamState,
             _build_stream_message, ANTHROPICServiceEndpoint, GPTFunction,
             RoleSystem, RoleUser, RoleAssistant, RoleTool, TOOL_CALLS, STOP, CONTENT_FILTER
using Test, HTTP, JSON, Sockets

@testset "encode — system split + user turn" begin
    chat = Chat(service=ANTHROPICServiceEndpoint, model="claude-opus-4-8")
    push!(chat, Message(Val(:system), "You are helpful."))
    push!(chat, Message(Val(:user), "Hi"))
    body = JSON.parse(encode_request(ANTHROPICServiceEndpoint, chat))
    @test body["model"] == "claude-opus-4-8"
    @test body["system"] == "You are helpful."
    @test body["max_tokens"] == 16000                    # default supplied
    @test length(body["messages"]) == 1
    @test body["messages"][1]["role"] == "user"
    @test body["messages"][1]["content"] == "Hi"
    @test !haskey(body, "temperature")                   # unset → omitted
end

@testset "encode — explicit max_tokens & stop_sequences preserved" begin
    chat = Chat(service=ANTHROPICServiceEndpoint, model="claude-opus-4-8",
                max_tokens=1000, stop=["END"])
    push!(chat, Message(Val(:system), "s"))
    push!(chat, Message(Val(:user), "u"))
    body = JSON.parse(encode_request(ANTHROPICServiceEndpoint, chat))
    @test body["max_tokens"] == 1000
    @test body["stop_sequences"] == ["END"]
end

@testset "encode — tools become {name,description,input_schema}" begin
    sig = FunctionSignature(name="get_weather", description="Get weather",
        parameters=Dict("type" => "object",
                        "properties" => Dict("location" => Dict("type" => "string")),
                        "required" => ["location"]))
    chat = Chat(service=ANTHROPICServiceEndpoint, model="claude-opus-4-8",
                tools=[Tool(func=sig)], tool_choice="auto")
    push!(chat, Message(Val(:system), "s"))
    push!(chat, Message(Val(:user), "weather in Paris?"))
    body = JSON.parse(encode_request(ANTHROPICServiceEndpoint, chat))
    @test length(body["tools"]) == 1
    @test body["tools"][1]["name"] == "get_weather"
    @test body["tools"][1]["description"] == "Get weather"
    @test body["tools"][1]["input_schema"]["type"] == "object"
    @test !haskey(body["tools"][1], "parameters")        # renamed, not OpenAI's key
    # Chat's parallel_tool_calls defaults to false, which Anthropic expresses on tool_choice.
    @test body["tool_choice"] == Dict("type" => "auto", "disable_parallel_tool_use" => true)
end

@testset "encode — multi-turn tool_use → tool_result collapse" begin
    chat = Chat(service=ANTHROPICServiceEndpoint, model="claude-opus-4-8")
    push!(chat, Message(Val(:system), "s"))
    push!(chat, Message(Val(:user), "weather?"))
    tc = ToolCall(id="toolu_1", func=GPTFunction("get_weather", Dict("location" => "Paris")))
    push!(chat, Message(role=RoleAssistant, tool_calls=[tc], finish_reason=TOOL_CALLS))
    push!(chat, Message(role=RoleTool, tool_call_id="toolu_1", content="72F"))
    body = JSON.parse(encode_request(ANTHROPICServiceEndpoint, chat))
    msgs = body["messages"]
    @test [m["role"] for m in msgs] == ["user", "assistant", "user"]
    au = msgs[2]["content"]
    @test au[1]["type"] == "tool_use"
    @test au[1]["id"] == "toolu_1"
    @test au[1]["name"] == "get_weather"
    @test au[1]["input"] == Dict("location" => "Paris")
    tr = msgs[3]["content"]
    @test tr[1]["type"] == "tool_result"
    @test tr[1]["tool_use_id"] == "toolu_1"
    @test tr[1]["content"] == "72F"
end

@testset "encode — orphan tool_result fails loud" begin
    msgs = [Message(Val(:user), "hi"),
            Message(role=RoleTool, tool_call_id="ghost", content="x")]
    chat = Chat(service=ANTHROPICServiceEndpoint, model="claude-opus-4-8", messages=msgs)
    @test_throws ArgumentError encode_request(ANTHROPICServiceEndpoint, chat)
end

@testset "encode — OpenAI-only moderation fails loud" begin
    chat = Chat(service=ANTHROPICServiceEndpoint, model="claude-opus-4-8", messages=[Message(Val(:user), "q")],
                moderation=ModerationConfig(model="omni-moderation-latest", input_mode="block"))
    err = try encode_request(ANTHROPICServiceEndpoint, chat); nothing catch e; e end
    @test err isa ArgumentError && err.msg == "Anthropic Messages does not support moderation; it is an OpenAI-only option"
end

@testset "decode — plain text" begin
    body = JSON.json(Dict("type" => "message", "role" => "assistant",
        "stop_reason" => "end_turn",
        "content" => [Dict("type" => "text", "text" => "Hello there")],
        "usage" => Dict("input_tokens" => 10, "output_tokens" => 3)))
    r = decode_response(ANTHROPICServiceEndpoint, HTTP.Response(200, [], Vector{UInt8}(body)))
    @test r.message.role == RoleAssistant
    @test r.message.content == "Hello there"
    @test r.message.finish_reason == STOP
    @test r.usage.prompt_tokens == 10
    @test r.usage.completion_tokens == 3
    @test r.usage.total_tokens == 13
end

@testset "decode — text + tool_use" begin
    body = JSON.json(Dict("stop_reason" => "tool_use",
        "content" => [Dict("type" => "text", "text" => "Let me check."),
                      Dict("type" => "tool_use", "id" => "toolu_9",
                           "name" => "get_weather", "input" => Dict("location" => "Paris"))],
        "usage" => Dict("input_tokens" => 20, "output_tokens" => 15)))
    r = decode_response(ANTHROPICServiceEndpoint, HTTP.Response(200, [], Vector{UInt8}(body)))
    @test r.message.finish_reason == TOOL_CALLS
    @test r.message.content == "Let me check."
    @test length(r.message.tool_calls) == 1
    @test r.message.tool_calls[1].id == "toolu_9"
    @test r.message.tool_calls[1].func.name == "get_weather"
    @test r.message.tool_calls[1].func.arguments == Dict("location" => "Paris")
end

@testset "decode — tool_use only (no text) round-trips through flat Message" begin
    body = JSON.json(Dict("stop_reason" => "tool_use",
        "content" => [Dict("type" => "tool_use", "id" => "toolu_2",
                           "name" => "f", "input" => Dict())],
        "usage" => Dict("input_tokens" => 5, "output_tokens" => 8)))
    r = decode_response(ANTHROPICServiceEndpoint, HTTP.Response(200, [], Vector{UInt8}(body)))
    @test isnothing(r.message.content)
    @test r.message.tool_calls[1].id == "toolu_2"
end

@testset "decode — max_tokens → length" begin
    body = JSON.json(Dict("stop_reason" => "max_tokens",
        "content" => [Dict("type" => "text", "text" => "partial")],
        "usage" => Dict("input_tokens" => 5, "output_tokens" => 100)))
    r = decode_response(ANTHROPICServiceEndpoint, HTTP.Response(200, [], Vector{UInt8}(body)))
    @test r.message.finish_reason == "length"
    @test r.message.content == "partial"
end

@testset "decode — refusal" begin
    body = JSON.json(Dict("stop_reason" => "refusal", "content" => [],
        "usage" => Dict("input_tokens" => 4, "output_tokens" => 0)))
    r = decode_response(ANTHROPICServiceEndpoint, HTTP.Response(200, [], Vector{UInt8}(body)))
    @test r.message.finish_reason == CONTENT_FILTER
    @test !isnothing(r.message.refusal_message)
end

@testset "decode — cache_read counts toward prompt_tokens (for correct billing)" begin
    body = JSON.json(Dict("stop_reason" => "end_turn",
        "content" => [Dict("type" => "text", "text" => "hi")],
        "usage" => Dict("input_tokens" => 4, "output_tokens" => 2,
                        "cache_read_input_tokens" => 100)))
    r = decode_response(ANTHROPICServiceEndpoint, HTTP.Response(200, [], Vector{UInt8}(body)))
    @test r.usage.cached_tokens == 100
    @test r.usage.prompt_tokens == 104          # input + cache_read; estimated_cost bills fresh=input
end

@testset "decode — provider-native content captured verbatim" begin
    body = """
    {"id":"msg_1","type":"message","role":"assistant","model":"claude-sonnet-5",
     "content":[{"type":"thinking","thinking":"user wants weather","signature":"sig=="},
                {"type":"text","text":"Checking."},
                {"type":"tool_use","id":"toolu_1","name":"get_weather","input":{"city":"Oslo"}}],
     "stop_reason":"tool_use","usage":{"input_tokens":10,"output_tokens":5}}
    """
    dec = UniLM.decode_response(ANTHROPICServiceEndpoint, HTTP.Response(200, [], Vector{UInt8}(body)))
    pc = dec.message.provider_content
    @test pc isa ProviderContent && pc.provider === :anthropic
    @test length(pc.blocks) == 3
    @test pc.blocks[1]["type"] == "thinking" && pc.blocks[1]["signature"] == "sig=="
    @test pc.blocks[2]["text"] == "Checking."
    @test pc.blocks[3]["input"] == Dict{String,Any}("city" => "Oslo")
    # Neutral fields unchanged by the capture.
    @test dec.message.content == "Checking."
    @test length(dec.message.tool_calls) == 1

    # Plain-text responses capture too (uniform rule: non-empty content array).
    plain = """{"id":"m2","type":"message","role":"assistant","model":"claude-opus-4-8",
       "content":[{"type":"text","text":"Hi."}],"stop_reason":"end_turn",
       "usage":{"input_tokens":1,"output_tokens":1}}"""
    dec2 = UniLM.decode_response(ANTHROPICServiceEndpoint, HTTP.Response(200, [], Vector{UInt8}(plain)))
    @test dec2.message.provider_content isa ProviderContent
    @test dec2.message.provider_content.blocks[1]["text"] == "Hi."

    # Empty content array → NO ProviderContent (echoing [] back would 400).
    empty_body = """{"id":"m3","type":"message","role":"assistant","model":"claude-opus-4-8",
       "content":[],"stop_reason":"refusal","usage":{"input_tokens":1,"output_tokens":0}}"""
    dec3 = UniLM.decode_response(ANTHROPICServiceEndpoint, HTTP.Response(200, [], Vector{UInt8}(empty_body)))
    @test isnothing(dec3.message.provider_content)
end

@testset "encode — provider-native blocks echoed verbatim on provider match" begin
    blocks = Any[
        Dict{String,Any}("type" => "thinking", "thinking" => "w", "signature" => "sig=="),
        Dict{String,Any}("type" => "tool_use", "id" => "toolu_1", "name" => "get_weather",
                         "input" => Dict{String,Any}("city" => "Oslo")),
    ]
    tc = [ToolCall(id="toolu_1", func=GPTFunction("get_weather", Dict{String,Any}("city" => "Oslo")))]
    m = Message(role=UniLM.RoleAssistant, tool_calls=tc,
                provider_content=ProviderContent(:anthropic, blocks))
    # Verbatim echo: the SAME array object, blocks unmodified, thinking first.
    @test UniLM._anthropic_assistant_content(m) === blocks

    # Through the full message pipeline (tool_result correlation intact).
    msgs = [Message(role=UniLM.RoleUser, content="weather?"), m,
            Message(role=UniLM.RoleTool, content="12C", tool_call_id="toolu_1")]
    _, wire = UniLM._anthropic_messages(msgs)
    @test wire[2][:content] === blocks
    @test wire[3][:content][1][:tool_use_id] == "toolu_1"

    # Cross-provider tag → neutral reconstruction (thinking dropped, tool_use rebuilt).
    m_gem = Message(role=UniLM.RoleAssistant, tool_calls=tc,
                    provider_content=ProviderContent(:gemini, blocks))
    rec = UniLM._anthropic_assistant_content(m_gem)
    @test rec isa Vector{Dict{Symbol,Any}} && rec[1][:type] == "tool_use"

    # Empty blocks → reconstruction (never emit an empty content array).
    m_empty = Message(role=UniLM.RoleAssistant, content="hi",
                      provider_content=ProviderContent(:anthropic, Any[]))
    @test UniLM._anthropic_assistant_content(m_empty) == "hi"
end

@testset "stream — text deltas + usage" begin
    lines = [
        "event: message_start",
        "data: " * JSON.json(Dict("type" => "message_start",
            "message" => Dict("usage" => Dict("input_tokens" => 8, "output_tokens" => 1)))),
        "",
        "data: " * JSON.json(Dict("type" => "content_block_start", "index" => 0,
            "content_block" => Dict("type" => "text", "text" => ""))),
        "",
        "data: " * JSON.json(Dict("type" => "content_block_delta", "index" => 0,
            "delta" => Dict("type" => "text_delta", "text" => "Hello"))),
        "",
        "data: " * JSON.json(Dict("type" => "content_block_delta", "index" => 0,
            "delta" => Dict("type" => "text_delta", "text" => " world"))),
        "",
        "data: " * JSON.json(Dict("type" => "message_delta",
            "delta" => Dict("stop_reason" => "end_turn"), "usage" => Dict("output_tokens" => 5))),
        "",
        "data: " * JSON.json(Dict("type" => "message_stop")),
    ]
    state = StreamState()
    st = UniLM._sse_dispatch!(ANTHROPICServiceEndpoint, IOBuffer(), Ref(""), join(lines, "\n") * "\n", state)
    @test st === :done
    @test state.finish_reason == STOP
    @test state.usage.completion_tokens == 5
    @test state.usage.prompt_tokens == 8
    @test String(take!(state.pending_delta)) == "Hello world"
    msg = _build_stream_message(state)
    @test msg.content == "Hello world"
    @test msg.finish_reason == STOP
end

@testset "stream — tool_use with input_json_delta" begin
    lines = [
        "data: " * JSON.json(Dict("type" => "message_start",
            "message" => Dict("usage" => Dict("input_tokens" => 12, "output_tokens" => 1)))),
        "",
        "data: " * JSON.json(Dict("type" => "content_block_start", "index" => 0,
            "content_block" => Dict("type" => "tool_use", "id" => "toolu_7",
                                     "name" => "get_weather", "input" => Dict()))),
        "",
        "data: " * JSON.json(Dict("type" => "content_block_delta", "index" => 0,
            "delta" => Dict("type" => "input_json_delta", "partial_json" => "{\"loc"))),
        "",
        "data: " * JSON.json(Dict("type" => "content_block_delta", "index" => 0,
            "delta" => Dict("type" => "input_json_delta", "partial_json" => "ation\":\"Paris\"}"))),
        "",
        "data: " * JSON.json(Dict("type" => "message_delta",
            "delta" => Dict("stop_reason" => "tool_use"), "usage" => Dict("output_tokens" => 20))),
        "",
        "data: " * JSON.json(Dict("type" => "message_stop")),
    ]
    state = StreamState()
    st = UniLM._sse_dispatch!(ANTHROPICServiceEndpoint, IOBuffer(), Ref(""), join(lines, "\n") * "\n", state)
    @test st === :done
    @test state.finish_reason == TOOL_CALLS
    msg = _build_stream_message(state)
    @test msg.finish_reason == TOOL_CALLS
    @test length(msg.tool_calls) == 1
    @test msg.tool_calls[1].id == "toolu_7"
    @test msg.tool_calls[1].func.name == "get_weather"
    @test msg.tool_calls[1].func.arguments == Dict("location" => "Paris")
end

@testset "stream — provider-native blocks assembled verbatim (thinking + text + tool_use)" begin
    state = UniLM.StreamState()
    lines = [
        "event: message_start",
        "data: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":10,\"output_tokens\":1}}}",
        "event: content_block_start",
        "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"thinking\",\"thinking\":\"\",\"signature\":\"\"}}",
        "event: content_block_delta",
        "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"thinking_delta\",\"thinking\":\"user wants \"}}",
        "event: content_block_delta",
        "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"thinking_delta\",\"thinking\":\"weather\"}}",
        "event: content_block_delta",
        "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"signature_delta\",\"signature\":\"sig==\"}}",
        "event: content_block_stop",
        "data: {\"type\":\"content_block_stop\",\"index\":0}",
        "event: content_block_start",
        "data: {\"type\":\"content_block_start\",\"index\":1,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}",
        "event: content_block_delta",
        "data: {\"type\":\"content_block_delta\",\"index\":1,\"delta\":{\"type\":\"text_delta\",\"text\":\"Checking.\"}}",
        "event: content_block_stop",
        "data: {\"type\":\"content_block_stop\",\"index\":1}",
        "event: content_block_start",
        "data: {\"type\":\"content_block_start\",\"index\":2,\"content_block\":{\"type\":\"tool_use\",\"id\":\"toolu_1\",\"name\":\"get_weather\",\"input\":{}}}",
        "event: content_block_delta",
        "data: {\"type\":\"content_block_delta\",\"index\":2,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"city\\\":\"}}",
        "event: content_block_delta",
        "data: {\"type\":\"content_block_delta\",\"index\":2,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"\\\"Oslo\\\"}\"}}",
        "event: content_block_stop",
        "data: {\"type\":\"content_block_stop\",\"index\":2}",
        "event: message_delta",
        "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"tool_use\"},\"usage\":{\"output_tokens\":25}}",
        "event: message_stop",
        "data: {\"type\":\"message_stop\"}",
    ]
    st = UniLM._sse_dispatch!(ANTHROPICServiceEndpoint, IOBuffer(), Ref(""), join(lines, "\n") * "\n", state)
    @test st === :done
    @test state.raw_provider === :anthropic
    @test length(state.raw_blocks) == 3
    @test state.raw_blocks[1]["type"] == "thinking" &&
          state.raw_blocks[1]["thinking"] == "user wants weather" &&
          state.raw_blocks[1]["signature"] == "sig=="
    @test state.raw_blocks[2]["type"] == "text" && state.raw_blocks[2]["text"] == "Checking."
    @test state.raw_blocks[3]["type"] == "tool_use" &&
          state.raw_blocks[3]["input"] == Dict{String,Any}("city" => "Oslo")
    @test isempty(state.raw_pending) && isempty(state.raw_json)
    # Built message carries the blocks AND the neutral fields.
    msg = UniLM._build_stream_message(state)
    @test msg.provider_content isa ProviderContent
    @test msg.provider_content.blocks === state.raw_blocks
    @test msg.content == "Checking." && length(msg.tool_calls) == 1
    # Encoder echoes the streamed turn verbatim — the streamed round-trip.
    @test UniLM._anthropic_assistant_content(msg) === state.raw_blocks
end

@testset "stream — redacted_thinking snapshot and zero-arg tool_use finalize" begin
    state = UniLM.StreamState()
    lines = [
        "event: content_block_start",
        "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"redacted_thinking\",\"data\":\"opaque==\"}}",
        "event: content_block_stop",
        "data: {\"type\":\"content_block_stop\",\"index\":0}",
        "event: content_block_start",
        "data: {\"type\":\"content_block_start\",\"index\":1,\"content_block\":{\"type\":\"tool_use\",\"id\":\"t2\",\"name\":\"ping\",\"input\":{}}}",
        "event: content_block_stop",
        "data: {\"type\":\"content_block_stop\",\"index\":1}",
        "event: message_stop",
        "data: {\"type\":\"message_stop\"}",
    ]
    st = UniLM._sse_dispatch!(ANTHROPICServiceEndpoint, IOBuffer(), Ref(""), join(lines, "\n") * "\n", state)
    @test st === :done
    @test state.raw_blocks[1] == Dict{String,Any}("type" => "redacted_thinking", "data" => "opaque==")
    # No input_json_delta arrived: the start snapshot's empty input survives.
    @test state.raw_blocks[2]["input"] == Dict{String,Any}()

    # A block whose stop never arrives is NOT finalized (truncated stream).
    state2 = UniLM.StreamState()
    UniLM._sse_dispatch!(ANTHROPICServiceEndpoint, IOBuffer(), Ref(""),
        "event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n", state2)
    @test isempty(state2.raw_blocks) && haskey(state2.raw_pending, 0)
end

@testset "an empty Anthropic turn is reported empty, not filled in" begin
    # Truthful-empty contract (matching the OpenAI and Gemini decoders): a
    # well-formed turn that produced no text is a real turn. Thinking models reach
    # this routinely — the whole budget goes to thought blocks and the turn stops at
    # max_tokens with no text block. Substituting prose put content nobody generated
    # into the reply AND into the next request's history.
    body = JSON.json(Dict("stop_reason" => "max_tokens", "content" => Any[],
                          "usage" => Dict("input_tokens" => 5, "output_tokens" => 0)))
    m = UniLM.decode_response(ANTHROPICServiceEndpoint,
                              HTTP.Response(200, [], Vector{UInt8}(body))).message
    @test m.role == UniLM.RoleAssistant
    @test m.content == ""
    @test !occursin("No response from the model", something(m.content, ""))

    # Thinking-only: the thought blocks still ride along verbatim for round-trip.
    thinking = JSON.json(Dict("stop_reason" => "max_tokens",
        "content" => [Dict("type" => "thinking", "thinking" => "hmm", "signature" => "sig")],
        "usage" => Dict("input_tokens" => 5, "output_tokens" => 100)))
    mt = UniLM.decode_response(ANTHROPICServiceEndpoint,
                               HTTP.Response(200, [], Vector{UInt8}(thinking))).message
    @test mt.content == ""
    @test mt.finish_reason == "length"
    @test mt.provider_content.blocks[1]["type"] == "thinking"

    # Unchanged neighbours: text, tool-only and the refusal path.
    text_body = JSON.json(Dict("stop_reason" => "end_turn",
        "content" => [Dict("type" => "text", "text" => "hello")]))
    @test UniLM.decode_response(ANTHROPICServiceEndpoint,
            HTTP.Response(200, [], Vector{UInt8}(text_body))).message.content == "hello"
    tool_body = JSON.json(Dict("stop_reason" => "tool_use",
        "content" => [Dict("type" => "tool_use", "id" => "t1", "name" => "f",
                           "input" => Dict("a" => 1))]))
    mtool = UniLM.decode_response(ANTHROPICServiceEndpoint,
                HTTP.Response(200, [], Vector{UInt8}(tool_body))).message
    @test isnothing(mtool.content) && length(mtool.tool_calls) == 1
    refusal_body = JSON.json(Dict("stop_reason" => "refusal", "content" => Any[]))
    mref = UniLM.decode_response(ANTHROPICServiceEndpoint,
               HTTP.Response(200, [], Vector{UInt8}(refusal_body))).message
    @test mref.refusal_message == "Model refused to respond."
end

# Wire goldens transcribed from https://platform.claude.com/docs/en/build-with-claude/structured-outputs.md:
# JSON outputs are output_config.format = {type: "json_schema", schema}; strict tool use is a
# top-level `strict` on the tool definition; there is no schema-less JSON mode.
const _ANTHROPIC_SCHEMA = Dict("type" => "object", "properties" => Dict("city" => Dict("type" => "string")),
                               "required" => ["city"], "additionalProperties" => false)
_anthropic_body(; kw...) = JSON.parse(encode_request(ANTHROPICServiceEndpoint,
    Chat(; service=ANTHROPICServiceEndpoint, model="claude-opus-5-5", messages=[Message(Val(:user), "q")], kw...)))
_anthropic_err(; kw...) = try _anthropic_body(; kw...); nothing catch e; e end
_roles(body) = [m["role"] for m in body["messages"]]

@testset "encode — response_format json_schema → output_config.format" begin
    golden = Dict("format" => Dict("type" => "json_schema", "schema" => _ANTHROPIC_SCHEMA))
    for rf in (UniLM.json_schema("city", "A city", _ANTHROPIC_SCHEMA; strict=true),
               ResponseFormat(Dict("name" => "city", "schema" => _ANTHROPIC_SCHEMA, "strict" => true)),
               ResponseFormat(Dict(:name => "city", :schema => _ANTHROPIC_SCHEMA)))
        @test _anthropic_body(response_format=rf)["output_config"] == golden
    end
    @test !haskey(_anthropic_body(response_format=ResponseFormat(type="text")), "output_config")
    @test !haskey(_anthropic_body(), "output_config")
    err = _anthropic_err(response_format=UniLM.json_object())
    @test err isa ArgumentError && occursin("json_object", err.msg) && occursin("json_schema", err.msg)
    @test _anthropic_err(response_format=ResponseFormat(type="xml")) isa ArgumentError
    @test _anthropic_err(response_format=ResponseFormat(Dict("name" => "no-schema"))) isa ArgumentError
end

@testset "encode — FunctionSignature strict → the tool's top-level strict" begin
    strict = Tool(func=FunctionSignature(name="lookup", parameters=_ANTHROPIC_SCHEMA, strict=true))
    loose = Tool(func=FunctionSignature(name="loose", parameters=_ANTHROPIC_SCHEMA))
    tools = _anthropic_body(tools=[strict, loose], tool_choice="auto")["tools"]
    @test tools[1] == Dict("name" => "lookup", "input_schema" => _ANTHROPIC_SCHEMA, "strict" => true)
    @test !haskey(tools[2], "strict")
    @test _anthropic_body(tools=[Tool(func=FunctionSignature(name="off", strict=false))])["tools"][1]["strict"] === false
end

# Per-model thinking and effort contract, from
# https://platform.claude.com/docs/en/build-with-claude/effort.md (levels per model),
# https://platform.claude.com/docs/en/build-with-claude/thinking.md (which models think by
# default, which accept thinking.type "disabled", which are extended-thinking only) and the
# Sonnet 5.5 and Haiku 5.5 migration guides (between_tools; disabled at high effort or below).
@testset "encode — reasoning_effort per Claude model" begin
    enc(model, effort; kw...) = _anthropic_body(; model, reasoning_effort=effort, kw...)
    err(model, effort) = _anthropic_err(; model, reasoning_effort=effort)
    all5, no_xhigh = ("low", "medium", "high", "xhigh", "max"), ("low", "medium", "high", "max")
    for (model, efforts) in (("claude-fable-5-1", all5), ("claude-mythos-5-1", all5), ("claude-fable-5", all5),
                             ("claude-mythos-5", all5), ("claude-opus-5-5", all5), ("claude-sonnet-5-5", all5),
                             ("claude-haiku-5-5", all5), ("claude-opus-5", all5), ("claude-sonnet-5", all5),
                             ("claude-opus-4-8", all5), ("claude-opus-4-7", all5),
                             ("claude-opus-4-6", no_xhigh), ("claude-sonnet-4-6", no_xhigh))
        for effort in efforts
            body = enc(model, effort)
            # An explicit effort also turns adaptive thinking on where it is off by default (4.6–4.8),
            # and asks for the thinking summarized, which most current models omit unless asked.
            @test body["output_config"] == Dict("effort" => effort) &&
                  body["thinking"] == Dict("type" => "adaptive", "display" => "summarized")
        end
    end
    for model in ("claude-opus-4-6", "claude-sonnet-4-6")
        e = err(model, "xhigh")
        @test e isa ArgumentError && occursin(model, e.msg) && occursin("reasoning_effort", e.msg)
    end
    # Opus 4.5 is extended-thinking only but takes effort (low, medium, high) without adaptive thinking.
    for effort in ("low", "medium", "high")
        body = enc("claude-opus-4-5-20251101", effort)
        @test body["output_config"] == Dict("effort" => effort) && !haskey(body, "thinking")
    end
    @test err("claude-opus-4-5", "max") isa ArgumentError
    # "none": disable where thinking is on by default and can be turned off, turn off up-front
    # thinking where "disabled" is refused (Sonnet 5.5: between_tools, with no effort), send
    # nothing where it is off by default, refuse where thinking cannot be disabled.
    for model in ("claude-opus-5", "claude-sonnet-5", "claude-haiku-5-5")
        body = enc(model, "none")
        @test body["thinking"] == Dict("type" => "disabled") && !haskey(body, "output_config")
    end
    body = enc("claude-sonnet-5-5", "none")
    @test body["thinking"] == Dict("type" => "between_tools") && !haskey(body, "output_config")
    for model in ("claude-opus-4-8", "claude-opus-4-7", "claude-opus-4-6", "claude-sonnet-4-6", "claude-opus-4-5")
        body = enc(model, "none")
        @test !haskey(body, "thinking") && !haskey(body, "output_config")
    end
    for model in ("claude-fable-5-1", "claude-mythos-5-1", "claude-fable-5", "claude-mythos-5", "claude-opus-5-5")
        e = err(model, "none")
        @test e isa ArgumentError && occursin(model, e.msg) && occursin("reasoning_effort", e.msg)
    end
    # No Claude model has a minimal effort, and an unknown level is not an effort at all.
    for model in ("claude-opus-5-5", "claude-sonnet-5-5", "claude-sonnet-5", "claude-opus-4-8", "claude-opus-9")
        e = err(model, "minimal")
        @test e isa ArgumentError && occursin("minimal", e.msg) && occursin("low", e.msg)
        @test err(model, "ultra") isa ArgumentError
    end
    # Haiku 4.5 takes no effort at all: any reasoning_effort names the model, its dated id too.
    for model in ("claude-haiku-4-5", "claude-haiku-4-5-20251001"), effort in ("low", "none")
        e = err(model, effort)
        @test e isa ArgumentError && occursin(model, e.msg) && occursin("reasoning_effort", e.msg)
    end
    # An unknown (future) Claude id maps effort and adaptive thinking without validation; so do
    # retired models, whose rows are gone (the API answers for them).
    for model in ("claude-opus-9", "claude-opus-4-1", "claude-opus-4-20250514", "claude-3-7-sonnet-20250219")
        body = enc(model, "xhigh")
        @test body["output_config"] == Dict("effort" => "xhigh") &&
              body["thinking"] == Dict("type" => "adaptive", "display" => "summarized")
        @test enc(model, "none")["thinking"] == Dict("type" => "disabled")
    end
    # Unset effort sends neither field; effort and a structured format share output_config.
    @test !haskey(enc("claude-opus-5-5", nothing), "thinking") && !haskey(enc("claude-opus-5-5", nothing), "output_config")
    body = enc("claude-sonnet-5", "low"; response_format=UniLM.json_schema("c", "d", _ANTHROPIC_SCHEMA))
    @test body["output_config"] == Dict("effort" => "low",
        "format" => Dict("type" => "json_schema", "schema" => _ANTHROPIC_SCHEMA))
end

@testset "encode — Chat fields with no Anthropic mapping fail closed" begin
    for (field, value) in ((:seed, 7), (:logprobs, true), (:logprobs, false), (:top_logprobs, 2),
                           (:presence_penalty, 0.5), (:frequency_penalty, 0.5),
                           (:logit_bias, Dict("50256" => -100.0)), (:verbosity, "low"), (:store, false),
                           (:prompt_cache_key, "k"), (:stream_options, Dict("include_usage" => true)),
                           (:prediction, Dict("type" => "content", "content" => "x")), (:modalities, ["text"]),
                           (:audio, Dict("voice" => "alloy")), (:web_search_options, Dict{String,Any}()))
        e = _anthropic_err(; field => value)
        @test e isa ArgumentError && occursin(string(field), e.msg)
    end
    e = _anthropic_err(seed=1, store=true)
    @test e isa ArgumentError && occursin("seed", e.msg) && occursin("store", e.msg)
    e = _anthropic_err(n=2)
    @test e isa ArgumentError && occursin("n must be 1", e.msg)
    @test _anthropic_body(n=1)["model"] == "claude-opus-5-5"
end

@testset "encode — parallel_tool_calls, user ids, metadata, service_tier and tool_choice mappings" begin
    tools = [Tool(func=FunctionSignature(name="f", parameters=Dict("type" => "object", "properties" => Dict())))]
    b(; kw...) = _anthropic_body(; model="claude-sonnet-5", kw...)
    e(; kw...) = _anthropic_err(; model="claude-sonnet-5", kw...)
    # parallel_tool_calls=false (Chat's default when tools are set) rides on the tool_choice object.
    @test b(tools=tools)["tool_choice"] == Dict("type" => "auto", "disable_parallel_tool_use" => true)
    @test b(tools=tools, tool_choice="required")["tool_choice"] ==
          Dict("type" => "any", "disable_parallel_tool_use" => true)
    @test b(tools=tools, tool_choice=UniLM.GPTToolChoice(func="f"))["tool_choice"] ==
          Dict("type" => "tool", "name" => "f", "disable_parallel_tool_use" => true)
    @test b(tools=tools, tool_choice="none")["tool_choice"] == Dict("type" => "none")
    @test !haskey(b(tools=tools, parallel_tool_calls=true), "tool_choice")
    @test b(tools=tools, parallel_tool_calls=true, tool_choice="auto")["tool_choice"] == Dict("type" => "auto")
    @test !haskey(b(), "tool_choice")
    x = e(tools=tools, tool_choice="reqired")
    @test x isa ArgumentError && occursin("tool_choice", x.msg) && occursin("reqired", x.msg)
    # safety_identifier, user and metadata.user_id all land on metadata.user_id and must agree.
    @test b(safety_identifier="u1")["metadata"] == Dict("user_id" => "u1")
    @test b(user="u2")["metadata"] == Dict("user_id" => "u2")
    @test b(user="u3", safety_identifier="u3")["metadata"] == Dict("user_id" => "u3")
    @test b(metadata=Dict("user_id" => "u4"))["metadata"] == Dict("user_id" => "u4")
    @test b(metadata=Dict(:user_id => "u5"), safety_identifier="u5")["metadata"] == Dict("user_id" => "u5")
    @test !haskey(b(), "metadata")
    x = e(user="a", safety_identifier="b")
    @test x isa ArgumentError && occursin("user", x.msg) && occursin("safety_identifier", x.msg)
    @test e(metadata=Dict("user_id" => "a"), safety_identifier="b") isa ArgumentError
    x = e(metadata=Dict("user_id" => "a", "tier" => "gold"))
    @test x isa ArgumentError && occursin("metadata", x.msg) && occursin("tier", x.msg)
    # service_tier: Anthropic accepts auto and standard_only (Messages API reference).
    @test b(service_tier="auto")["service_tier"] == "auto"
    @test b(service_tier="standard_only")["service_tier"] == "standard_only"
    x = e(service_tier="flex")
    @test x isa ArgumentError && occursin("service_tier", x.msg)
end

# Model restrictions from https://platform.claude.com/docs/en/build-with-claude/thinking.md
# ("Limits and feature compatibility"), the Opus 5.5, Sonnet 5.5 and Haiku 5.5 migration
# guides, and https://platform.claude.com/docs/en/api/migration-2026-10-21.md (sampling
# parameters removed: HTTP 400 on every model, even at the default value).
@testset "encode — Claude model restrictions are validated locally" begin
    tools = [Tool(func=FunctionSignature(name="f", parameters=Dict("type" => "object", "properties" => Dict())))]
    b(model; kw...) = _anthropic_body(; model, kw...)
    e(model; kw...) = _anthropic_err(; model, kw...)
    # API version 2026-10-21 has no temperature or top_p, whatever the model or the value.
    for model in ("claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-5-5", "claude-fable-5-1", "claude-mythos-5-1",
                  "claude-fable-5", "claude-mythos-5", "claude-opus-5", "claude-sonnet-5", "claude-opus-4-8",
                  "claude-opus-4-7", "claude-opus-4-6", "claude-sonnet-4-6", "claude-opus-4-5", "claude-haiku-4-5",
                  "claude-haiku-4-5-20251001", "claude-x", "claude-sonnet-5-5-x"),
        (field, value) in ((:temperature, 1.0), (:temperature, 0.3), (:temperature, 0.0), (:top_p, 0.9), (:top_p, 1.0))
        x = e(model; field => value)
        @test x isa ArgumentError && occursin(string(field), x.msg) && occursin("2026-10-21", x.msg)
    end
    # Opus 5.5, Sonnet 5.5, Fable 5.1 and Mythos 5.1 reject forced tool use on every request.
    for model in ("claude-opus-5-5", "claude-sonnet-5-5", "claude-fable-5-1", "claude-mythos-5-1"),
        tc in ("required", UniLM.GPTToolChoice(func="f"))
        x = e(model; tools, tool_choice=tc)
        @test x isa ArgumentError && occursin(model, x.msg) && occursin("tool_choice", x.msg) &&
              occursin("auto", x.msg) && occursin("strict", x.msg)
    end
    for model in ("claude-opus-5-5", "claude-sonnet-5-5"), tc in ("auto", "none")
        @test b(model; tools, tool_choice=tc)["tool_choice"]["type"] == tc
    end
    # Haiku 5.5 takes a forced tool_choice, like Opus 5 and Fable 5.
    for model in ("claude-haiku-5-5", "claude-opus-5", "claude-sonnet-5")
        @test b(model; tools, tool_choice="required")["tool_choice"] == Dict("type" => "any", "disable_parallel_tool_use" => true)
        @test b(model; tools, tool_choice=UniLM.GPTToolChoice(func="f"))["tool_choice"] ==
              Dict("type" => "tool", "name" => "f", "disable_parallel_tool_use" => true)
    end
    @test b("claude-fable-5"; tools, tool_choice=UniLM.GPTToolChoice(func="f"))["tool_choice"]["type"] == "tool"
    # A trailing assistant turn (response prefill) is rejected from the 4.6 generation on.
    prefill = [Message(Val(:user), "q"), Message(role=RoleAssistant, content="Sure:")]
    for model in ("claude-opus-4-6", "claude-sonnet-4-6", "claude-opus-4-7", "claude-opus-4-8", "claude-sonnet-5",
                  "claude-opus-5", "claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-5-5", "claude-fable-5-1",
                  "claude-mythos-5-1", "claude-fable-5")
        x = e(model; messages=prefill)
        @test x isa ArgumentError && occursin(model, x.msg) && occursin("messages", x.msg)
    end
    for model in ("claude-haiku-4-5", "claude-opus-4-5", "claude-opus-9")
        @test b(model; messages=prefill)["messages"][end] == Dict("role" => "assistant", "content" => "Sure:")
    end
end

# A row applies to its exact id and to that id with a dated snapshot suffix (-YYYYMMDD) only:
# no generation inherits another's rules by a shared prefix, and an id no row names gets no
# per-model rule — the API answers for it — while the provider-wide rules still apply.
@testset "model rows: exact ids and dated snapshots, both directions" begin
    tools = [Tool(func=FunctionSignature(name="f", parameters=Dict("type" => "object", "properties" => Dict())))]
    b(model; kw...) = _anthropic_body(; model, kw...)
    e(model; kw...) = _anthropic_err(; model, kw...)
    forced = (; tools, tool_choice="required")
    s(x) = Message(Val(:system), x)
    later = [s("s0"), Message(Val(:user), "u1"), Message(role=RoleAssistant, content="a1"), Message(Val(:user), "u2"),
             s("s2")]
    # Sonnet 5.5 never takes Sonnet 5's rules (disabled thinking, forced tools, hoisted system)...
    @test b("claude-sonnet-5-5"; reasoning_effort="none")["thinking"] == Dict("type" => "between_tools")
    @test e("claude-sonnet-5-5"; forced...) isa ArgumentError
    @test b("claude-sonnet-5-5"; messages=later)["messages"][end] == Dict("role" => "system", "content" => "s2")
    # ...and Sonnet 5 never takes Sonnet 5.5's.
    @test b("claude-sonnet-5"; reasoning_effort="none")["thinking"] == Dict("type" => "disabled")
    @test b("claude-sonnet-5"; forced...)["tool_choice"]["type"] == "any"
    body = b("claude-sonnet-5"; messages=later)
    @test body["system"] == "s0\n\ns2" && _roles(body) == ["user", "assistant", "user"]
    # Dated snapshot ids take their base id's row.
    for effort in ("low", "none")
        x = e("claude-haiku-4-5-20251001"; reasoning_effort=effort)
        @test x isa ArgumentError && occursin("claude-haiku-4-5-20251001", x.msg)
    end
    @test b("claude-haiku-4-5-20251001"; messages=[Message(Val(:user), "q"), Message(role=RoleAssistant, content="Sure:")]
           )["messages"][end] == Dict("role" => "assistant", "content" => "Sure:")
    @test b("claude-sonnet-5-5-20261001"; reasoning_effort="none")["thinking"] == Dict("type" => "between_tools")
    @test e("claude-opus-5-5-20261001"; forced...) isa ArgumentError
    # Near misses are not dated snapshots: no row, so no per-model rule...
    for model in ("claude-sonnet-5-5-x", "claude-sonnet-5-5-2026100", "claude-haiku-5", "claude-opus-5-55")
        @test UniLM._claude_family(model) === nothing
        @test b(model; forced...)["tool_choice"]["type"] == "any"
        @test b(model; reasoning_effort="none")["thinking"] == Dict("type" => "disabled")
        # ...but the provider-wide rules hold.
        x = e(model; temperature=1.0)
        @test x isa ArgumentError && occursin("2026-10-21", x.msg)
    end
    # Retired models have no rows.
    for model in ("claude-opus-4-1", "claude-opus-4-1-20250805", "claude-opus-4-0", "claude-opus-4-20250514",
                  "claude-sonnet-4-0", "claude-sonnet-4-20250514", "claude-3", "claude-3-7-sonnet-20250219",
                  "claude-3-5-haiku-20241022", "claude-3-haiku-20240307")
        @test UniLM._claude_family(model) === nothing
    end
    @test (@inferred Nothing UniLM._claude_family("claude-haiku-5-5")) isa UniLM._ClaudeFamily
    @test UniLM._claude_family("claude-haiku-4-5-20251001") === UniLM._claude_family("claude-haiku-4-5")
end

@testset "stream — long blocks accumulate in linear space, byte-identical" begin
    # 8000 deltas of 40 bytes per block. Re-concatenating the growing String per delta
    # allocated ~3.7 GiB for this stream; buffered accumulation allocates ~37 MiB
    # (mostly per-event JSON parsing). The bound leaves a 5x margin over the linear cost.
    ev(d) = "data: " * JSON.json(d) * "\n"
    function stream(n, piece)
        io = IOBuffer()
        print(io, ev(Dict("type" => "content_block_start", "index" => 0,
                          "content_block" => Dict("type" => "text", "text" => ""))))
        for _ in 1:n
            print(io, ev(Dict("type" => "content_block_delta", "index" => 0,
                              "delta" => Dict("type" => "text_delta", "text" => piece))))
        end
        print(io, ev(Dict("type" => "content_block_stop", "index" => 0)))
        print(io, ev(Dict("type" => "content_block_start", "index" => 1, "content_block" =>
                          Dict("type" => "tool_use", "id" => "t1", "name" => "f", "input" => Dict()))))
        for pj in ("{\"s\":\"", ntuple(_ -> piece, n)..., "\"}")
            print(io, ev(Dict("type" => "content_block_delta", "index" => 1,
                              "delta" => Dict("type" => "input_json_delta", "partial_json" => pj))))
        end
        print(io, ev(Dict("type" => "content_block_stop", "index" => 1)))
        print(io, ev(Dict("type" => "message_delta", "delta" => Dict("stop_reason" => "tool_use"))))
        print(io, ev(Dict("type" => "message_stop")))
        String(take!(io))
    end
    dispatch(payload) = (st = StreamState();
        a = @allocated UniLM._sse_dispatch!(ANTHROPICServiceEndpoint, IOBuffer(), Ref(""), payload, st); (a, st))
    piece = "0123456789012345678901234567890123456789"
    dispatch(stream(10, piece))                        # compile outside the measurement
    n = 8000
    allocated, st = dispatch(stream(n, piece))
    @test allocated < 200 * 2^20
    @test st.raw_blocks[1] == Dict{String,Any}("type" => "text", "text" => piece^n)
    @test st.raw_blocks[2]["input"] == Dict{String,Any}("s" => piece^n)
    @test st.tool_calls[1]["function"]["arguments"] == "{\"s\":\"" * piece^n * "\"}"
    msg = _build_stream_message(st)
    @test msg.content == piece^n && msg.tool_calls[1].func.arguments == Dict{String,Any}("s" => piece^n)
end

@testset "stream — a tool call whose stop line was dropped still carries its arguments" begin
    lines = [
        "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"tool_use\",\"id\":\"a\",\"name\":\"f\",\"input\":{}}}",
        "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"x\\\":1}\"}}",
        # no content_block_stop for index 0
        "data: {\"type\":\"content_block_start\",\"index\":1,\"content_block\":{\"type\":\"tool_use\",\"id\":\"b\",\"name\":\"g\",\"input\":{}}}",
    ]
    st = StreamState()
    UniLM._sse_dispatch!(ANTHROPICServiceEndpoint, IOBuffer(), Ref(""), join(lines, "\n") * "\n", st)
    # index 0 is no longer the newest call, so the driver may fire it now
    @test st.tool_calls[0]["function"]["arguments"] == "{\"x\":1}"
    UniLM._sse_dispatch!(ANTHROPICServiceEndpoint, IOBuffer(), Ref(""),
        "data: {\"type\":\"content_block_delta\",\"index\":1,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"y\\\":2}\"}}\n" *
        "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"tool_use\"}}\n", st)
    @test st.tool_calls[1]["function"]["arguments"] == "{\"y\":2}"
    msg = _build_stream_message(st)
    @test [tc.func.arguments for tc in msg.tool_calls] == [Dict{String,Any}("x" => 1), Dict{String,Any}("y" => 2)]
    @test isnothing(msg.provider_content)              # blocks still pending: capture incomplete
end

@testset "stream — a tool_use cut off by max_tokens keeps the turn and drops the partial call" begin
    # max_tokens ends the turn inside the tool input: its block stops on a JSON fragment,
    # which is a truncation, not an undecodable line. The turn keeps its text, "length"
    # and usage; the partial call is dropped (no loop runs a "length" turn's calls), and
    # the unfinished block leaves the capture incomplete, so nothing partial is echoed.
    ev(d) = "data: " * JSON.json(d)
    lines = [
        ev(Dict("type" => "message_start", "message" => Dict("usage" => Dict("input_tokens" => 12, "output_tokens" => 1)))),
        ev(Dict("type" => "content_block_start", "index" => 0, "content_block" => Dict("type" => "text", "text" => ""))),
        ev(Dict("type" => "content_block_delta", "index" => 0, "delta" => Dict("type" => "text_delta", "text" => "Let me check."))),
        ev(Dict("type" => "content_block_stop", "index" => 0)),
        ev(Dict("type" => "content_block_start", "index" => 1, "content_block" => Dict("type" => "tool_use",
            "id" => "toolu_9", "name" => "get_weather", "input" => Dict()))),
        ev(Dict("type" => "content_block_delta", "index" => 1,
            "delta" => Dict("type" => "input_json_delta", "partial_json" => "{\"city\": \"Par"))),
        ev(Dict("type" => "content_block_stop", "index" => 1)),
        ev(Dict("type" => "message_delta", "delta" => Dict("stop_reason" => "max_tokens"),
                "usage" => Dict("output_tokens" => 50))),
        ev(Dict("type" => "message_stop")),
    ]
    st = StreamState()
    @test UniLM._sse_dispatch!(ANTHROPICServiceEndpoint, IOBuffer(), Ref(""), join(lines, "\n") * "\n", st) === :done
    @test st.sse_dropped == 0
    @test st.usage.completion_tokens == 50
    msg = _build_stream_message(st)
    @test msg.finish_reason == "length" && msg.content == "Let me check."
    @test isnothing(msg.tool_calls) && isnothing(msg.provider_content)
end

# Stop reasons and refusals: https://platform.claude.com/docs/en/build-with-claude/handling-stop-reasons.md,
# https://platform.claude.com/docs/en/build-with-claude/refusals-and-fallback.md (stop_details, discard
# partial output), and .../test-and-evaluate/strengthen-guardrails/handle-streaming-refusals.md
# (stop_details arrives on message_delta alongside stop_reason).
_anthropic_decode(d) = decode_response(ANTHROPICServiceEndpoint, HTTP.Response(200, [], Vector{UInt8}(JSON.json(d))))

@testset "decode — model_context_window_exceeded → length" begin
    m = _anthropic_decode(Dict("stop_reason" => "model_context_window_exceeded",
        "content" => [Dict("type" => "text", "text" => "cut")])).message
    @test m.finish_reason == "length" && m.content == "cut"
end

@testset "refusal — streamed and non-streamed turns agree; partial output is discarded" begin
    sd = Dict("type" => "refusal", "category" => "cyber",
              "explanation" => "This request was declined because it could enable cyber harm.")
    partial = [Dict("type" => "text", "text" => "Hello.."),
               Dict("type" => "tool_use", "id" => "t1", "name" => "f", "input" => Dict())]
    m = _anthropic_decode(Dict("content" => partial, "stop_reason" => "refusal", "stop_details" => sd,
                               "usage" => Dict("input_tokens" => 7, "output_tokens" => 3))).message
    @test isnothing(m.content) && isnothing(m.tool_calls) && isnothing(m.provider_content)
    @test m.refusal_message == sd["explanation"] && m.finish_reason == CONTENT_FILTER
    for details in (nothing, Dict("type" => "refusal", "category" => nothing, "explanation" => nothing))
        d = Dict{String,Any}("content" => Any[], "stop_reason" => "refusal")
        isnothing(details) || (d["stop_details"] = details)
        @test _anthropic_decode(d).message.refusal_message == "Model refused to respond."
    end
    lines = [
        "data: " * JSON.json(Dict("type" => "message_start", "message" => Dict("usage" => Dict("input_tokens" => 7, "output_tokens" => 1)))),
        "data: " * JSON.json(Dict("type" => "content_block_start", "index" => 0, "content_block" => Dict("type" => "text", "text" => ""))),
        "data: " * JSON.json(Dict("type" => "content_block_delta", "index" => 0, "delta" => Dict("type" => "text_delta", "text" => "Hello.."))),
        "data: " * JSON.json(Dict("type" => "content_block_stop", "index" => 0)),
        "data: " * JSON.json(Dict("type" => "content_block_start", "index" => 1, "content_block" => partial[2])),
        "data: " * JSON.json(Dict("type" => "content_block_stop", "index" => 1)),
        "data: " * JSON.json(Dict("type" => "message_delta", "delta" => Dict("stop_reason" => "refusal", "stop_details" => sd),
                                  "usage" => Dict("output_tokens" => 3))),
        "data: " * JSON.json(Dict("type" => "message_stop")),
    ]
    st = StreamState()
    @test UniLM._sse_dispatch!(ANTHROPICServiceEndpoint, IOBuffer(), Ref(""), join(lines, "\n") * "\n", st) === :done
    @test String(take!(st.pending_delta)) == ""        # unsent partial text is withheld
    ms = _build_stream_message(st)
    for f in (:role, :content, :refusal_message, :finish_reason, :tool_calls, :provider_content)
        @test getfield(ms, f) == getfield(m, f)
    end
end

# A local endpoint speaking the Anthropic wire, so the verb's handling of a malformed
# 200 is exercised end to end (the production endpoint's base URL is a constant).
struct _AnthropicLocalEndpoint <: UniLM.ServiceEndpoint
    url::String
end
UniLM.get_url(s::_AnthropicLocalEndpoint, ::Chat) = s.url
UniLM.auth_header(::_AnthropicLocalEndpoint) = ["Content-Type" => "application/json"]
UniLM.encode_request(::_AnthropicLocalEndpoint, chat::Chat) = encode_request(ANTHROPICServiceEndpoint, chat)
UniLM.decode_response(::_AnthropicLocalEndpoint, resp::HTTP.Response) = decode_response(ANTHROPICServiceEndpoint, resp)

@testset "decode — a 200 without a content array or stop_reason is an error, not an empty success" begin
    for body in (Dict("type" => "message", "stop_reason" => "end_turn"),
                 Dict("type" => "message", "content" => Any[]),
                 Dict("type" => "message", "content" => "text", "stop_reason" => "end_turn"),
                 Any[1, 2])
        @test_throws ErrorException _anthropic_decode(body)
    end
    srv, port = nothing, 0
    for attempt in 1:5                         # OS-assigned ephemeral port, bind retry
        tcp = Sockets.listen(Sockets.localhost, 0)
        port = Int(Sockets.getsockname(tcp)[2]); close(tcp)
        try
            srv = HTTP.serve!("127.0.0.1", port; verbose=false) do req
                HTTP.Response(200, ["Content-Type" => "application/json"], "{\"type\":\"message\"}")
            end
            break
        catch
            attempt == 5 && rethrow()
        end
    end
    try
        chat = Chat(service=_AnthropicLocalEndpoint("http://127.0.0.1:$port/v1/messages"), model="claude-opus-5-5",
                    messages=[Message(Val(:user), "q")])
        r = chatrequest!(chat; config=RequestConfig(max_attempts=1, total_deadline=10.0))
        @test r isa LLMCallError && r.cause isa ErrorException && occursin("stop_reason", r.error)
        @test length(chat) == 1                  # no empty turn committed to history
    finally
        close(srv)
    end
end

@testset "usage — cache writes count as input, thinking tokens are reasoning (both paths)" begin
    u = Dict("input_tokens" => 10, "cache_creation_input_tokens" => 20, "cache_read_input_tokens" => 30,
             "output_tokens" => 50, "output_tokens_details" => Dict("thinking_tokens" => 40))
    want = TokenUsage(prompt_tokens=60, completion_tokens=50, total_tokens=110, cached_tokens=30, reasoning_tokens=40,
                      cache_write_tokens=20)
    @test _anthropic_decode(Dict("stop_reason" => "end_turn", "usage" => u,
                                 "content" => [Dict("type" => "text", "text" => "x")])).usage == want
    start = "data: " * JSON.json(Dict("type" => "message_start", "message" => Dict("usage" =>
        Dict("input_tokens" => 10, "cache_creation_input_tokens" => 20, "cache_read_input_tokens" => 30, "output_tokens" => 1))))
    # The thinking breakdown arrives only on the final message_delta, whose counts are cumulative.
    delta = "data: " * JSON.json(Dict("type" => "message_delta", "delta" => Dict("stop_reason" => "end_turn"),
        "usage" => Dict("output_tokens" => 50, "output_tokens_details" => Dict("thinking_tokens" => 40))))
    st = StreamState()
    UniLM._sse_dispatch!(ANTHROPICServiceEndpoint, IOBuffer(), Ref(""), start * "\n" * delta * "\n", st)
    @test st.usage == want
    # A message_delta that repeats the cumulative input counts (server-tool loops) is authoritative.
    delta2 = "data: " * JSON.json(Dict("type" => "message_delta", "delta" => Dict("stop_reason" => "end_turn"),
        "usage" => Dict("input_tokens" => 15, "cache_creation_input_tokens" => 20, "cache_read_input_tokens" => 30,
                        "output_tokens" => 60)))
    st2 = StreamState()
    UniLM._sse_dispatch!(ANTHROPICServiceEndpoint, IOBuffer(), Ref(""), start * "\n" * delta2 * "\n", st2)
    @test st2.usage == TokenUsage(prompt_tokens=65, completion_tokens=60, total_tokens=125, cached_tokens=30,
                                  cache_write_tokens=20)
end

# https://platform.claude.com/docs/en/api/messages.md (Usage): cache_creation splits the writes by
# retention; service_tier, inference_geo and speed say how the call was billed. A stream carries
# the full usage on message_start; message_delta repeats only cumulative counts.
const _ANTHROPIC_USAGE = Dict("input_tokens" => 1000, "cache_creation_input_tokens" => 300,
    "cache_read_input_tokens" => 50,
    "cache_creation" => Dict("ephemeral_5m_input_tokens" => 200, "ephemeral_1h_input_tokens" => 100),
    "output_tokens" => 20, "output_tokens_details" => Dict("thinking_tokens" => 5),
    "service_tier" => "standard", "inference_geo" => "us", "speed" => "fast")
_anthropic_message(model, usage) = Dict("id" => "msg_u", "type" => "message", "role" => "assistant", "model" => model,
    "content" => [Dict("type" => "text", "text" => "hello")], "stop_reason" => "end_turn", "stop_sequence" => nothing,
    "usage" => usage)
function _anthropic_sse(model, usage; delta_usage=Dict("output_tokens" => usage["output_tokens"],
                                                       "output_tokens_details" => usage["output_tokens_details"]))
    ev(d) = "event: " * d["type"] * "\ndata: " * JSON.json(d) * "\n\n"
    start = merge(_anthropic_message(model, merge(usage, Dict("output_tokens" => 1,
        "output_tokens_details" => Dict("thinking_tokens" => 0)))), Dict("content" => Any[], "stop_reason" => nothing))
    ev(Dict("type" => "message_start", "message" => start)) *
    ev(Dict("type" => "content_block_start", "index" => 0, "content_block" => Dict("type" => "text", "text" => ""))) *
    ev(Dict("type" => "content_block_delta", "index" => 0, "delta" => Dict("type" => "text_delta", "text" => "hello"))) *
    ev(Dict("type" => "content_block_stop", "index" => 0)) *
    ev(Dict("type" => "message_delta", "delta" => Dict("stop_reason" => "end_turn", "stop_sequence" => nothing),
            "usage" => delta_usage)) *
    ev(Dict("type" => "message_stop"))
end

@testset "usage — cache writes by retention, the billed tier, the region and the served model (both paths)" begin
    want = TokenUsage(prompt_tokens=1350, completion_tokens=20, total_tokens=1370, cached_tokens=50, reasoning_tokens=5,
                      cache_write_tokens=300, cache_write_extended_tokens=100, service_tier="fast", inference_geo="us")
    dec = _anthropic_decode(_anthropic_message("claude-opus-5-5", _ANTHROPIC_USAGE))
    @test dec.usage == want
    @test dec.usage.prompt_tokens == 1000 + 50 + 300           # input + cache reads + cache writes
    @test dec.model == "claude-opus-5-5"
    st = StreamState()
    @test UniLM._sse_dispatch!(ANTHROPICServiceEndpoint, IOBuffer(), Ref(""),
                               _anthropic_sse("claude-opus-5-5", _ANTHROPIC_USAGE), st) === :done
    @test st.usage == want && st.model == "claude-opus-5-5"
    # A message_delta that repeats the cumulative input counts carries no breakdown, tier or region:
    # those stay as message_start reported them.
    st2 = StreamState()
    UniLM._sse_dispatch!(ANTHROPICServiceEndpoint, IOBuffer(), Ref(""), _anthropic_sse("claude-opus-5-5", _ANTHROPIC_USAGE;
        delta_usage=Dict("input_tokens" => 1000, "cache_creation_input_tokens" => 300, "cache_read_input_tokens" => 50,
                         "output_tokens" => 20, "output_tokens_details" => Dict("thinking_tokens" => 5))), st2)
    @test st2.usage == want
    # Standard speed bills at the reported tier; a reply without the fields reports none.
    standard = merge(_ANTHROPIC_USAGE, Dict("speed" => "standard", "inference_geo" => "global"))
    u = _anthropic_decode(_anthropic_message("claude-haiku-5-5", standard)).usage
    @test (u.service_tier, u.inference_geo) == ("standard", "global")
    bare = _anthropic_decode(_anthropic_message("claude-haiku-5-5", Dict("input_tokens" => 3, "output_tokens" => 1)))
    @test bare.usage == TokenUsage(prompt_tokens=3, completion_tokens=1, total_tokens=4)
    @test isnothing(_anthropic_decode(Dict("stop_reason" => "end_turn", "content" => Any[])).model)
end

# https://platform.claude.com/docs/en/about-claude/pricing.md (Data residency pricing): "For
# Claude 4.6 and later models, specifying US-only inference through the inference_geo parameter
# incurs a 1.1x multiplier on all token pricing categories ... Earlier models do not support the
# inference_geo parameter and always use standard pricing". Refuted if: a 4.5 model's usage that
# reports "us" bills more than its "global" twin, or a 4.6-and-later model's does not bill 1.1×.
@testset "US-only inference bills 1.1× on Claude 4.6 and later, the standard rates on the 4.5 models" begin
    m = Message(role=RoleAssistant, content="x")
    usage(geo) = TokenUsage(prompt_tokens=10_000, completion_tokens=500, total_tokens=10_500, cached_tokens=2_000,
                            cache_write_tokens=1_000, inference_geo=geo)
    cost(model, geo) =
        estimated_cost(LLMSuccess(message=m, self=Chat(service=ANTHROPICServiceEndpoint, model=model), usage=usage(geo)))
    for model in ("claude-haiku-5-5", "claude-opus-5-5", "claude-opus-4-6", "claude-sonnet-4-6")
        @test isapprox(cost(model, "us"), 1.1 * cost(model, "global"); rtol=1e-12)
    end
    # Haiku 4.5 usage reporting "us" bills 1.0×, as do the other 4.5 models, dated snapshots included.
    for model in ("claude-haiku-4-5", "claude-haiku-4-5-20251001", "claude-opus-4-5-20251101", "claude-sonnet-4-5")
        @test cost(model, "us") == cost(model, "global") > 0
    end
    # An id no row names keeps the multiplier: a newer model takes inference_geo.
    @test @inferred(UniLM._anthropic_rate_multiplier("claude-nova-6", usage("us"))) == 1.1
    @test UniLM._anthropic_rate_multiplier("claude-nova-6", usage("global")) == 1.0
end

# The real Anthropic headers, encoder, decoder, stream handler and pricing — of the bare type or
# of an endpoint instance — against a local server that records each request (the production
# endpoint's base URL is a constant).
struct _AnthropicCapture <: UniLM.ServiceEndpoint
    url::String
    real::Union{Type{ANTHROPICServiceEndpoint},ANTHROPICServiceEndpoint}
end
_AnthropicCapture(url::String) = _AnthropicCapture(url, ANTHROPICServiceEndpoint)
UniLM.get_url(s::_AnthropicCapture, ::Chat) = s.url
UniLM.auth_header(s::_AnthropicCapture) = UniLM.auth_header(s.real)
UniLM.encode_request(s::_AnthropicCapture, chat::Chat) = encode_request(s.real, chat)
UniLM.decode_response(s::_AnthropicCapture, resp::HTTP.Response) = decode_response(s.real, resp)
UniLM.handle_sse_event!(s::_AnthropicCapture, event::AbstractString, payload::AbstractString, state::StreamState) =
    UniLM.handle_sse_event!(s.real, event, payload, state)
UniLM._billed_cost(s::_AnthropicCapture, model::String, u::TokenUsage, rates::UniLM.PriceRow) =
    UniLM._billed_cost(s.real, model, u, rates)

# Answers a streamed request with `sse(n)` and any other with `json(n)` (n = the request's number);
# `seen` holds each request's lowercased headers and parsed body.
function _anthropic_capture_server(json, sse)
    seen = Tuple{Dict{String,String},Dict{String,Any}}[]
    lk = ReentrantLock()
    server, port = nothing, 0
    for attempt in 1:5                         # OS-assigned ephemeral port, bind retry
        tcp = Sockets.listen(Sockets.localhost, 0)
        port = Int(Sockets.getsockname(tcp)[2]); close(tcp)
        try
            server = HTTP.listen!("127.0.0.1", port; verbose=false) do http::HTTP.Stream
                body = JSON.parse(String(read(http)); dicttype=Dict{String,Any})
                headers = Dict(lowercase(String(k)) => String(v) for (k, v) in http.message.headers)
                n = @lock lk (push!(seen, (headers, body)); length(seen))
                streamed = get(body, "stream", false) === true
                HTTP.setstatus(http, 200)
                HTTP.setheader(http, "Content-Type" => streamed ? "text/event-stream" : "application/json")
                HTTP.startwrite(http)
                write(http, streamed ? sse(n) : json(n))
            end
            break
        catch
            attempt == 5 && rethrow()
        end
    end
    (; server, url="http://127.0.0.1:$port/v1/messages", seen)
end

@testset "every request sends anthropic-version 2026-10-21; streamed and non-streamed calls cost the same" begin
    usage = merge(_ANTHROPIC_USAGE, Dict("speed" => "standard", "output_tokens_details" => Dict("thinking_tokens" => 0)))
    srv = _anthropic_capture_server(_ -> JSON.json(_anthropic_message("claude-haiku-5-5", usage)),
                                    _ -> _anthropic_sse("claude-haiku-5-5", usage))
    cfg = RequestConfig(max_attempts=1, total_deadline=30.0)
    conversation() = [Message(Val(:system), "s"), Message(Val(:user), "q")]
    try
        withenv("ANTHROPIC_API_KEY" => "sk-ant-local-0000") do
            results = map((false, true)) do streamed
                chat = Chat(service=_AnthropicCapture(srv.url), model="claude-haiku-5-5", stream=streamed,
                            messages=conversation())
                r = streamed ? fetch(chatrequest!(chat; config=cfg)) : chatrequest!(chat; config=cfg)
                (r, cumulative_cost(chat))
            end
            (plain, plain_cost), (streamed, streamed_cost) = results
            @test plain isa LLMSuccess && streamed isa LLMSuccess
            @test plain.usage == streamed.usage == TokenUsage(prompt_tokens=1350, completion_tokens=20,
                total_tokens=1370, cached_tokens=50, cache_write_tokens=300, cache_write_extended_tokens=100,
                service_tier="standard", inference_geo="us")
            @test plain.model == streamed.model == "claude-haiku-5-5"
            # Haiku 5.5: $0.10 input, $0.01 cache hit, $0.50 output per 1M tokens; writes 1.25x (5 min)
            # and 2x (1 h) input; US-only inference 1.1x.
            want = (1000 * 0.10 + 50 * 0.01 + 200 * 0.10 * 1.25 + 100 * 0.10 * 2.0 + 20 * 0.50) * 1.1 / 1e6
            @test isapprox(plain_cost, want; rtol=1e-12) && plain_cost == streamed_cost
            @test length(srv.seen) == 2
            for (headers, body) in srv.seen
                @test headers["anthropic-version"] == "2026-10-21"
                @test headers["x-api-key"] == "sk-ant-local-0000"
            end
            @test [get(body, "stream", false) for (_, body) in srv.seen] == [false, true]
            # A request the encoder refuses never reaches the server.
            for kw in ((temperature=1.0,), (top_p=0.5,))
                chat = Chat(; service=_AnthropicCapture(srv.url), model="claude-haiku-5-5", messages=conversation(), kw...)
                @test_throws ArgumentError chatrequest!(chat; config=cfg)
            end
            # So does a system message after the reply with no user turn after it, even where
            # the model takes a prefill: it is not one.
            after_reply = [conversation(); Message(role=RoleAssistant, content="a"); Message(Val(:system), "s2")]
            chat = Chat(service=_AnthropicCapture(srv.url), model="claude-haiku-4-5", messages=after_reply)
            @test_throws ArgumentError chatrequest!(chat; config=cfg)
            @test length(srv.seen) == 2
        end
        @test Dict(withenv(() -> UniLM.auth_header(ANTHROPICServiceEndpoint), "ANTHROPIC_API_KEY" => "k")) ==
              Dict("x-api-key" => "k", "anthropic-version" => "2026-10-21", "Content-Type" => "application/json")
    finally
        close(srv.server)
    end
end

@testset "encode — assistant turns with no content and no tool calls are skipped" begin
    msgs = [Message(Val(:user), "a"), Message(role=RoleAssistant, content=""), Message(Val(:user), "b"),
            Message(role=RoleAssistant, refusal_message="Model refused to respond.", finish_reason=CONTENT_FILTER),
            Message(Val(:user), "c"), Message(role=RoleAssistant, content="")]
    body = _anthropic_body(model="claude-opus-5-5", messages=msgs)     # no trailing turn left to be a prefill
    @test body["messages"] == [Dict("role" => "user", "content" => c) for c in ("a", "b", "c")]
end

# https://platform.claude.com/docs/en/build-with-claude/mid-conversation-system-messages.md
const _MID_SYSTEM_MODELS = ("claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-5-5", "claude-opus-5",
                            "claude-opus-4-8", "claude-fable-5-1", "claude-mythos-5-1", "claude-fable-5",
                            "claude-mythos-5", "claude-opus-9")
const _NO_MID_SYSTEM_MODELS = ("claude-sonnet-5", "claude-opus-4-7", "claude-opus-4-6", "claude-haiku-4-5")
# The models that check each thinking block's prefix (the top-level system prompt, the tools and
# the earlier messages), https://platform.claude.com/docs/en/build-with-claude/preserved-thinking.md;
# they and an id no row names are placed append-only. The rest accept mid-conversation system
# messages without that check.
const _PREFIX_CHECKED_MODELS = ("claude-fable-5-1", "claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-5-5")
const _APPEND_ONLY_MODELS = (_PREFIX_CHECKED_MODELS..., "claude-opus-9")
const _UNCHECKED_MID_SYSTEM_MODELS = ("claude-mythos-5-1", "claude-fable-5", "claude-mythos-5", "claude-opus-5",
                                      "claude-opus-4-8")

@testset "encode — a later system message stays in place where the model accepts it" begin
    tc = ToolCall(id="toolu_1", func=GPTFunction("f", Dict{String,Any}()))
    msgs = [Message(Val(:system), "rules"), Message(Val(:system), "more rules"), Message(Val(:user), "q1"),
            Message(role=RoleAssistant, tool_calls=[tc]), Message(role=RoleTool, tool_call_id="toolu_1", content="ok"),
            Message(role=RoleSystem, content="new rule"), Message(role=RoleAssistant, content="a1"),
            Message(Val(:user), "q2")]
    for model in _MID_SYSTEM_MODELS
        body = _anthropic_body(; model, messages=msgs)
        @test body["system"] == "rules\n\nmore rules"
        @test _roles(body) == ["user", "assistant", "user", "system", "assistant", "user"]
        @test body["messages"][3]["content"][1]["type"] == "tool_result"         # results flushed before it
        @test body["messages"][4] == Dict("role" => "system", "content" => "new rule")
    end
    # Elsewhere it is hoisted into the top-level system prompt, which every model accepts.
    for model in _NO_MID_SYSTEM_MODELS
        body = _anthropic_body(; model, messages=msgs)
        @test body["system"] == "rules\n\nmore rules\n\nnew rule"
        @test _roles(body) == ["user", "assistant", "user", "assistant", "user"]
    end
    # Leading system messages alone are never mid-conversation.
    @test _anthropic_body(model="claude-sonnet-5", messages=msgs[1:3])["system"] == "rules\n\nmore rules"
end

@testset "encode — a mid-conversation system message the placement rule rejects is hoisted" begin
    # The rule: right after a user turn (tool results are one), and last or right before
    # an assistant turn; a run of system messages counts as one. Any other position is a
    # 400, so on the models without a prefix check it goes to the top-level system prompt, as
    # it does on the models without mid-conversation system messages.
    sys, u, a = Message(Val(:system), "s0"), Message(Val(:user), "u"), Message(role=RoleAssistant, content="a")
    s(x) = Message(Val(:system), x)
    tc(id) = ToolCall(id=id, func=GPTFunction("f", Dict{String,Any}()))
    tools = Message(role=RoleAssistant, tool_calls=[tc("t1"), tc("t2")])
    result(id) = Message(role=RoleTool, tool_call_id=id, content="ok")
    cases = [
        # (messages, roles on the wire, top-level system)
        ([sys, u, a, s("s1"), u], ["user", "assistant", "user"], "s0\n\ns1"),                   # after an assistant turn
        ([sys, u, s("s1"), u], ["user", "user"], "s0\n\ns1"),                                   # before a user turn
        ([sys, u, s("s1")], ["user", "system"], "s0"),                                          # last: kept
        ([sys, u, s("s1"), s("s2"), a, u], ["user", "system", "system", "assistant", "user"], "s0"),   # a run, kept
        ([sys, u, s("s1"), s("s2"), u], ["user", "user"], "s0\n\ns1\n\ns2"),                   # a run, hoisted
        ([sys, u, s("s1"), Message(role=RoleAssistant, content=""), a, u],                    # past a skipped turn
         ["user", "system", "assistant", "user"], "s0"),
        ([sys, u, tools, s("s1"), result("t1"), result("t2"), a, u],                          # before its results
         ["user", "assistant", "user", "assistant", "user"], "s0\n\ns1"),
        ([sys, u, tools, result("t1"), s("s1"), result("t2"), a, u],                          # between them
         ["user", "assistant", "user", "assistant", "user"], "s0\n\ns1"),
        ([sys, u, tools, result("t1"), result("t2"), s("s1"), a, u],                          # after them: kept
         ["user", "assistant", "user", "system", "assistant", "user"], "s0"),
    ]
    for model in _UNCHECKED_MID_SYSTEM_MODELS, (msgs, roles, system) in cases
        body = _anthropic_body(; model, messages=msgs)
        @test _roles(body) == roles
        @test body["system"] == system
    end
    # Hoisted between two results, the results stay one user turn of two tool_result blocks.
    body = _anthropic_body(; model="claude-opus-5", messages=cases[8][1])
    @test [b["tool_use_id"] for b in body["messages"][3]["content"]] == ["t1", "t2"]
    # The same messages on a model without mid-conversation system messages: all hoisted.
    body = _anthropic_body(; model="claude-haiku-4-5", messages=cases[1][1])
    @test _roles(body) == ["user", "assistant", "user"] && body["system"] == "s0\n\ns1"
end

# Wire forms for exact comparisons.
_wire(role, content) = Dict{String,Any}("role" => role, "content" => content)
_thinking_reply(text, sig) = Message(role=RoleAssistant, content=text, finish_reason=STOP,
    provider_content=ProviderContent(:anthropic, Any[Dict{String,Any}("type" => "thinking", "thinking" => "", "signature" => sig),
                                                     Dict{String,Any}("type" => "text", "text" => text)]))

@testset "encode — append-only placement never moves a later system message into the top-level prompt" begin
    # On the prefix-checked models a thinking block is valid only while the top-level system
    # prompt, the tools and the messages before it are unchanged; an id no row names is treated
    # the same. A system message after an assistant turn that cannot stay in place therefore
    # moves forward to right after the next user turn; one before the first assistant turn may
    # join the top-level prompt (no thinking block precedes it).
    s(x) = Message(Val(:system), x)
    u(x) = Message(Val(:user), x)
    a1 = _thinking_reply("a1", "sig1")
    blocks1 = a1.provider_content.blocks
    for model in _APPEND_ONLY_MODELS
        body = _anthropic_body(; model, messages=[s("s0"), u("u1"), a1, s("s2"), u("u2")])
        @test body["system"] == "s0"
        @test body["messages"] == [_wire("user", "u1"), _wire("assistant", blocks1), _wire("user", "u2"),
                                   _wire("system", "s2")]
        # Before the first assistant turn no thinking block exists: a misplaced one is hoisted.
        @test _anthropic_body(; model, messages=[s("s0"), u("u1"), s("s1"), u("u2")])["system"] == "s0\n\ns1"
        body = _anthropic_body(; model, messages=[s("s0"), u("u1"), s("s1")])
        @test body["system"] == "s0" && body["messages"] == [_wire("user", "u1"), _wire("system", "s1")]
        # Where it fits it stays in place.
        body = _anthropic_body(; model, messages=[s("s0"), u("u1"), a1, u("u2"), s("s2")])
        @test body["system"] == "s0" && _roles(body) == ["user", "assistant", "user", "system"]
    end
end

@testset "encode — under append-only placement every later request extends the earlier one" begin
    s(x) = Message(Val(:system), x)
    u(x) = Message(Val(:user), x)
    call(id) = ToolCall(id=id, func=GPTFunction("f", Dict{String,Any}()))
    tool_turn(ids...) = Message(role=RoleAssistant, tool_calls=[call(id) for id in ids], finish_reason=TOOL_CALLS,
        provider_content=ProviderContent(:anthropic, Any[
            Dict{String,Any}("type" => "thinking", "thinking" => "", "signature" => "sig-tools");
            [Dict{String,Any}("type" => "tool_use", "id" => id, "name" => "f", "input" => Dict{String,Any}()) for id in ids]]))
    result(id) = Message(role=RoleTool, tool_call_id=id, content="ok")
    a1 = _thinking_reply("a1", "sig1")
    shapes = [
        # (conversation as sent, the wire roles of its request)
        ([s("s0"), u("u1"), a1, s("s2"), u("u2")], ["user", "assistant", "user", "system"]),
        ([s("s0"), u("u1"), tool_turn("t1"), s("s2"), result("t1")], ["user", "assistant", "user", "system"]),
        ([s("s0"), u("u1"), tool_turn("t1", "t2"), result("t1"), s("s2"), result("t2")],
         ["user", "assistant", "user", "system"]),
        ([s("s0"), u("u1"), a1, u("u2"), s("s2")], ["user", "assistant", "user", "system"]),
        ([s("s0"), u("u1"), a1, u("u2"), s("s2"), u("u3")], ["user", "assistant", "user", "user", "system"]),
        ([s("s0"), u("u1"), a1, s("s2"), s("s3"), u("u2")], ["user", "assistant", "user", "system", "system"]),
        ([s("s0"), u("u1"), s("s1"), a1, s("s2"), u("u2")], ["user", "system", "assistant", "user", "system"]),
        ([s("s0"), u("u1"), s("s1"), u("u2"), a1, s("s2"), u("u3")], ["user", "user", "assistant", "user", "system"]),
    ]
    for model in _APPEND_ONLY_MODELS, (msgs, roles) in shapes
        sent = copy(msgs)
        earlier = _anthropic_body(; model, messages=sent)
        @test _roles(earlier) == roles
        @test !occursin("s2", something(get(earlier, "system", nothing), ""))
        # Two more rounds: the reply (thinking blocks included) and the next user turn appended.
        for round in 1:2
            push!(sent, _thinking_reply("r$round", "sig-r$round"), u("next $round"))
            later = _anthropic_body(; model, messages=sent)
            n = length(earlier["messages"])
            @test get(later, "system", nothing) == get(earlier, "system", nothing)
            @test later["messages"][1:n] == earlier["messages"]
            earlier = later
        end
    end
    # A later system message with no user turn after it has nowhere to go: the encoder refuses.
    for model in _APPEND_ONLY_MODELS,
        msgs in ([s("s0"), u("u1"), a1, s("s2")], [s("s0"), u("u1"), tool_turn("t1"), s("s2")],
                 [s("s0"), u("u1"), a1, s("s2"), _thinking_reply("a2", "sig2")])
        x = _anthropic_err(; model, messages=msgs)
        @test x isa ArgumentError && occursin(model, x.msg) && occursin("user", x.msg) && occursin("system", x.msg)
    end
    # The models without the check keep hoisting the same message into the top-level prompt.
    for model in _UNCHECKED_MID_SYSTEM_MODELS
        body = _anthropic_body(; model, messages=[s("s0"), u("u1"), a1, s("s2"), u("u2")])
        @test body["system"] == "s0\n\ns2" && _roles(body) == ["user", "assistant", "user"]
    end
end

@testset "push! takes a system message after a reply, and the encoder places it" begin
    chat = Chat(service=ANTHROPICServiceEndpoint, model="claude-haiku-5-5")
    push!(chat, Message(Val(:system), "s0"))
    push!(chat, Message(Val(:user), "u1"))
    push!(chat, _thinking_reply("a1", "sig1"))
    push!(chat, Message(Val(:system), "Answer in French from now on."))
    push!(chat, Message(Val(:user), "u2"))
    @test length(chat) == 5
    body = JSON.parse(encode_request(ANTHROPICServiceEndpoint, chat))
    @test body["system"] == "s0"
    @test body["messages"] == [_wire("user", "u1"), _wire("assistant", chat[3].provider_content.blocks),
                               _wire("user", "u2"), _wire("system", "Answer in French from now on.")]
end

@testset "encode — an assistant turn holding only thinking blocks is echoed verbatim" begin
    # A turn cut at max_tokens can carry thinking blocks and no text; to continue it, "pass the
    # assistant turn back unchanged and append a new user message"
    # (https://platform.claude.com/docs/en/build-with-claude/thinking.md).
    thinking = JSON.json(Dict("stop_reason" => "max_tokens", "model" => "claude-haiku-5-5",
        "content" => [Dict("type" => "thinking", "thinking" => "", "signature" => "sig==")],
        "usage" => Dict("input_tokens" => 5, "output_tokens" => 16)))
    cut = decode_response(ANTHROPICServiceEndpoint, HTTP.Response(200, [], Vector{UInt8}(thinking))).message
    @test cut.content == "" && isnothing(cut.tool_calls)
    msgs = [Message(Val(:user), "q1"), cut, Message(Val(:user), "q2")]
    for model in ("claude-haiku-5-5", "claude-opus-5-5", "claude-sonnet-5", "claude-opus-4-8")
        body = _anthropic_body(; model, messages=msgs)
        @test body["messages"] == [_wire("user", "q1"),
            _wire("assistant", [Dict("type" => "thinking", "thinking" => "", "signature" => "sig==")]),
            _wire("user", "q2")]
    end
    _, wire = UniLM._anthropic_messages(msgs)
    @test wire[2][:content] === cut.provider_content.blocks
    # A turn with nothing to send — no text, no tool call, no Anthropic blocks — is still left out.
    empty_gemini = Message(role=RoleAssistant, content="", provider_content=ProviderContent(:gemini, Any[Dict("text" => "x")]))
    body = _anthropic_body(; model="claude-haiku-5-5", messages=[Message(Val(:user), "q1"), empty_gemini, Message(Val(:user), "q2")])
    @test body["messages"] == [_wire("user", "q1"), _wire("user", "q2")]
end

@testset "encode — thinking blocks another Claude model produced go back as they were returned" begin
    # The API reads the blocks the current model can and drops the others without an error
    # (https://platform.claude.com/docs/en/build-with-claude/preserved-thinking.md, "Switching
    # models mid-conversation"): claude-haiku-5-5 answered a request whose last assistant turn
    # held claude-sonnet-5's thinking and tool_use blocks with HTTP 200 (2026-10-22). So a
    # conversation moved to another Claude model keeps every turn's blocks, unedited.
    blocks = Any[Dict{String,Any}("type" => "thinking", "thinking" => "", "signature" => "sonnet-5-signature"),
                 Dict{String,Any}("type" => "tool_use", "id" => "toolu_s5", "name" => "f",
                                  "input" => Dict{String,Any}("x" => 1))]
    turn = Message(role=RoleAssistant, finish_reason=TOOL_CALLS, provider_content=ProviderContent(:anthropic, blocks),
                   tool_calls=[ToolCall(id="toolu_s5", func=GPTFunction("f", Dict{String,Any}("x" => 1)))])
    msgs = [Message(Val(:system), "s"), Message(Val(:user), "q"), turn,
            Message(role=RoleTool, tool_call_id="toolu_s5", content="ok")]
    for model in ("claude-haiku-5-5", "claude-sonnet-5-5", "claude-opus-5-5", "claude-sonnet-5")
        body = _anthropic_body(; model, messages=msgs)
        @test body["messages"] == [_wire("user", "q"), _wire("assistant", blocks),
            _wire("user", [Dict("type" => "tool_result", "tool_use_id" => "toolu_s5", "content" => "ok")])]
    end
end

@testset "encode — a system message after the last reply, with no user turn after it, is refused" begin
    # The Messages API takes a mid-conversation system message only right after a user turn
    # (https://platform.claude.com/docs/en/build-with-claude/mid-conversation-system-messages.md),
    # and joined into the top-level prompt it would turn the request into a response prefill.
    s(x) = Message(Val(:system), x)
    u(x) = Message(Val(:user), x)
    a1 = Message(role=RoleAssistant, content="a1")
    # Prefill accepted (Haiku 4.5, Opus 4.5) or refused (Sonnet 5, Opus 5), append-only (Opus 5.5, an unknown id).
    for model in ("claude-haiku-4-5", "claude-opus-4-5", "claude-sonnet-5", "claude-opus-5", "claude-opus-5-5", "claude-opus-9"),
        msgs in ([s("s0"), u("u1"), a1, s("s2")], [s("s0"), u("u1"), a1, s("s2"), s("s3")])
        x = _anthropic_err(; model, messages=msgs)
        @test x isa ArgumentError && occursin(model, x.msg) && occursin("push a user message", x.msg)
    end
    # A prefill the caller writes stays one where the model takes it, and a reply with nothing to
    # send leaves the system message after the user turn.
    @test _anthropic_body(; model="claude-haiku-4-5", messages=[s("s0"), u("u1"), a1])["messages"][end] ==
          _wire("assistant", "a1")
    refusal = Message(role=RoleAssistant, refusal_message="Model refused to respond.", finish_reason=CONTENT_FILTER)
    body = _anthropic_body(; model="claude-haiku-4-5", messages=[s("s0"), u("u1"), refusal, s("s2")])
    @test body["system"] == "s0\n\ns2" && body["messages"] == [_wire("user", "u1")]
end

@testset "encode — an empty text block is never sent; a turn of nothing else is left out" begin
    # A streamed text block that got no delta stays {"type": "text", "text": ""}, and the Messages
    # API rejects empty text content.
    ev(d) = "data: " * JSON.json(d) * "\n"
    stream = ev(Dict("type" => "message_start", "message" => Dict("usage" => Dict("input_tokens" => 5, "output_tokens" => 1)))) *
             ev(Dict("type" => "content_block_start", "index" => 0, "content_block" => Dict("type" => "text", "text" => ""))) *
             ev(Dict("type" => "content_block_stop", "index" => 0)) *
             ev(Dict("type" => "message_delta", "delta" => Dict("stop_reason" => "end_turn"), "usage" => Dict("output_tokens" => 1))) *
             ev(Dict("type" => "message_stop"))
    st = StreamState()
    @test UniLM._sse_dispatch!(ANTHROPICServiceEndpoint, IOBuffer(), Ref(""), stream, st) === :done
    empty_turn = _build_stream_message(st)
    @test empty_turn.content == "" && empty_turn.provider_content.blocks == [Dict("type" => "text", "text" => "")]
    q1, q2 = Message(Val(:user), "q1"), Message(Val(:user), "q2")
    for model in ("claude-haiku-5-5", "claude-opus-5-5", "claude-haiku-4-5")
        @test _anthropic_body(; model, messages=[q1, empty_turn, q2])["messages"] == [_wire("user", "q1"), _wire("user", "q2")]
    end
    # Beside other blocks an empty text block is dropped, and the rest goes back as returned.
    thinking = Dict{String,Any}("type" => "thinking", "thinking" => "", "signature" => "sig==")
    tool_use = Dict{String,Any}("type" => "tool_use", "id" => "toolu_e", "name" => "f", "input" => Dict{String,Any}())
    empty = Dict{String,Any}("type" => "text", "text" => "")
    cut = Message(role=RoleAssistant, content="", finish_reason="length",
                  provider_content=ProviderContent(:anthropic, Any[thinking, empty]))
    @test _anthropic_body(; model="claude-haiku-5-5", messages=[q1, cut, q2])["messages"][2] == _wire("assistant", [thinking])
    calls = [ToolCall(id="toolu_e", func=GPTFunction("f", Dict{String,Any}()))]
    call = Message(role=RoleAssistant, finish_reason=TOOL_CALLS, tool_calls=calls,
                   provider_content=ProviderContent(:anthropic, Any[empty, tool_use]))
    body = _anthropic_body(; model="claude-haiku-5-5",
                           messages=[q1, call, Message(role=RoleTool, tool_call_id="toolu_e", content="ok")])
    @test body["messages"][2] == _wire("assistant", [tool_use])
    # With no empty text block the captured blocks go back as the same array.
    whole = Any[thinking, tool_use]
    @test UniLM._anthropic_assistant_content(Message(role=RoleAssistant, tool_calls=calls,
              provider_content=ProviderContent(:anthropic, whole))) === whole
end

@testset "a Claude rule row and a placement are valid by construction" begin
    @test_throws ArgumentError UniLM._ClaudeFamily(thinking=:sometimes)
    @test_throws ArgumentError UniLM._ClaudeFamily(thinking=:on, prefix_checked=true)     # needs system=true
    @test_throws ArgumentError UniLM._ClaudeFamily(thinking=:on, fast=:sometimes)
    @test UniLM._ClaudeFamily(thinking=:between, system=true, prefix_checked=true).prefix_checked
    @test all(row -> row.thinking in UniLM._CLAUDE_THINKING && (!row.prefix_checked || row.system) &&
                     row.fast in UniLM._CLAUDE_FAST, values(UniLM._CLAUDE_FAMILIES))
    @test_throws TypeError UniLM._anthropic_messages([Message(Val(:user), "q")]; placement=:hoist)
    @test UniLM._system_placement(nothing) == UniLM._APPEND_ONLY
end

# ─── Media, prompt caching, thinking display, fast mode and the configurable endpoint ───

const _ANT_FIXTURES = joinpath(@__DIR__, "fixtures")
const _ANT_PNG = read(joinpath(_ANT_FIXTURES, "ollama", "red_circle.png"))
const _ANT_PDF_PATH = joinpath(_ANT_FIXTURES, "documents", "secret_word.pdf")
const _ANT_PDF = read(_ANT_PDF_PATH)
const _ANT_WAV = Vector{UInt8}(b"RIFF\x24\x00\x00\x00WAVEfmt ")
_ant_source(mime, data) = Dict("type" => "base64", "media_type" => mime, "data" => UniLM.Base64.base64encode(data))
_ant_body(service; kw...) = JSON.parse(encode_request(service,
    Chat(; service, model="claude-opus-5-5", messages=[Message(Val(:user), "q")], kw...)))
_ant_err(service; kw...) = try _ant_body(service; kw...); nothing catch e; e end
_ant_user(attachments...; text="What is shown?") = [Message(role=RoleUser, content=text, attachments=Attachment[attachments...])]

# https://platform.claude.com/docs/en/build-with-claude/vision.md and .../pdf-support.md: base64 image
# (JPEG, PNG, GIF, WebP) and PDF document blocks; both guides place the media before the text.
@testset "encode — images and PDF documents become image and document blocks, before the text" begin
    jpeg, gif = UInt8[0xff, 0xd8, 0xff, 0xe0, 0x00, 0x10], Vector{UInt8}(b"GIF89a\x01\x00")
    webp = Vector{UInt8}(b"RIFF\x24\x00\x00\x00WEBPVP8 ")
    image(mime, data) = Dict("type" => "image", "source" => _ant_source(mime, data))
    document = Dict("type" => "document", "source" => _ant_source("application/pdf", _ANT_PDF))
    doc = DocumentAttachment(_ANT_PDF_PATH)
    msgs = _ant_user(ImageAttachment(_ANT_PNG), ImageAttachment(jpeg), ImageAttachment(gif), ImageAttachment(webp), doc)
    body = _anthropic_body(model="claude-haiku-5-5", messages=msgs)
    @test body["messages"] == [_wire("user", [image("image/png", _ANT_PNG), image("image/jpeg", jpeg),
        image("image/gif", gif), image("image/webp", webp), document, Dict("type" => "text", "text" => "What is shown?")])]
    # No text, or empty text: the media alone (the API refuses an empty text block).
    for text in (nothing, "")
        @test _anthropic_body(model="claude-haiku-5-5", messages=_ant_user(doc; text))["messages"] ==
              [_wire("user", [document])]
    end
    # Later requests send an earlier turn's media again; a turn without any keeps its string.
    convo = [_ant_user(ImageAttachment(_ANT_PNG); text="And this?"); Message(role=RoleAssistant, content="A red circle.");
             Message(Val(:user), "Thanks.")]
    @test _anthropic_body(model="claude-haiku-5-5", messages=convo)["messages"] ==
          [_wire("user", [image("image/png", _ANT_PNG), Dict("type" => "text", "text" => "And this?")]),
           _wire("assistant", "A red circle."), _wire("user", "Thanks.")]
    # The bare type and an endpoint instance send the same blocks, byte for byte.
    chat(service) = Chat(; service, model="claude-haiku-5-5", messages=msgs)
    @test encode_request(ANTHROPICServiceEndpoint(), chat(ANTHROPICServiceEndpoint())) ==
          encode_request(ANTHROPICServiceEndpoint, chat(ANTHROPICServiceEndpoint))
end

@testset "encode — a sound clip and an image's detail level are refused before any request" begin
    cases = ((AudioAttachment(_ANT_WAV), "audio"), (ImageAttachment(_ANT_PNG; detail="high"), "detail"),
             (ImageAttachment(_ANT_PNG; detail="original"), "detail"))
    for (att, word) in cases
        e = _anthropic_err(model="claude-haiku-5-5", messages=_ant_user(ImageAttachment(_ANT_PNG), att))
        @test e isa ArgumentError && occursin("Anthropic", e.msg) && occursin(word, e.msg)
    end
    srv = _anthropic_capture_server(_ -> JSON.json(_anthropic_message("claude-haiku-5-5",
                                                   Dict("input_tokens" => 3, "output_tokens" => 1))), _ -> "")
    try
        withenv("ANTHROPIC_API_KEY" => "sk-ant-local-0000") do
            for (att, _) in cases, real in (ANTHROPICServiceEndpoint, ANTHROPICServiceEndpoint())
                chat = Chat(service=_AnthropicCapture(srv.url, real), model="claude-haiku-5-5", messages=_ant_user(att))
                @test_throws ArgumentError chatrequest!(chat; config=RequestConfig(max_attempts=1, total_deadline=10.0))
            end
        end
        @test isempty(srv.seen)
    finally
        close(srv.server)
    end
end

# https://platform.claude.com/docs/en/build-with-claude/prompt-caching.md#automatic-caching: one
# top-level cache_control, {type: "ephemeral"} with an optional ttl "5m" or "1h".
@testset "encode — prompt_cache_options → a top-level cache_control (automatic caching)" begin
    cc(o) = get(_anthropic_body(model="claude-haiku-5-5", prompt_cache_options=o), "cache_control", nothing)
    @test cc(PromptCacheOptions(ttl="1h")) == Dict("type" => "ephemeral", "ttl" => "1h")
    @test cc(PromptCacheOptions(ttl="5m")) == Dict("type" => "ephemeral", "ttl" => "5m")
    @test cc(PromptCacheOptions()) == Dict("type" => "ephemeral")
    @test cc(PromptCacheOptions(mode="implicit")) == Dict("type" => "ephemeral")
    @test cc(PromptCacheOptions(mode="implicit", ttl="1h")) == Dict("type" => "ephemeral", "ttl" => "1h")
    # prewarm=false asks for what leaving prewarm out sends.
    @test cc(PromptCacheOptions(prewarm=false)) == Dict("type" => "ephemeral")
    @test cc(PromptCacheOptions(prewarm=false, ttl="1h")) == Dict("type" => "ephemeral", "ttl" => "1h")
    @test isnothing(cc(nothing))
    # What automatic caching cannot carry is refused, naming the options Anthropic takes.
    for (o, cause) in ((PromptCacheOptions(ttl="30m"), "\"30m\""), (PromptCacheOptions(ttl="12h"), "\"12h\""),
                       (PromptCacheOptions(mode="explicit"), "explicit"), (PromptCacheOptions(mode="explicit", ttl="5m"), "explicit"),
                       (PromptCacheOptions(prewarm=true), "prewarm=true"), (PromptCacheOptions(prewarm=true, ttl="5m"), "prewarm=true"),
                       (PromptCacheOptions(comparison_response_id="resp_1"), "comparison_response_id"))
        e = _anthropic_err(model="claude-haiku-5-5", prompt_cache_options=o)
        @test e isa ArgumentError && occursin("Anthropic", e.msg) && occursin("\"implicit\"", e.msg) &&
              occursin("\"5m\"", e.msg) && occursin("\"1h\"", e.msg) && occursin(cause, e.msg)
    end
    @test _ant_body(ANTHROPICServiceEndpoint(); model="claude-haiku-5-5", prompt_cache_options=PromptCacheOptions(ttl="1h"))["cache_control"] ==
          Dict("type" => "ephemeral", "ttl" => "1h")
end

# https://platform.claude.com/docs/en/build-with-claude/thinking.md#controlling-thinking-display:
# display "summarized" returns the thinking text, which most current models omit by default; it is
# invalid with thinking "disabled" and "between_tools".
@testset "encode — an explicit reasoning_effort asks for summarized thinking" begin
    for model in ("claude-opus-5-5", "claude-haiku-5-5", "claude-sonnet-5-5", "claude-fable-5-1", "claude-opus-4-8",
                  "claude-sonnet-4-6", "claude-opus-9"), service in (ANTHROPICServiceEndpoint, ANTHROPICServiceEndpoint())
        @test _ant_body(service; model, reasoning_effort="low")["thinking"] == Dict("type" => "adaptive", "display" => "summarized")
        @test !haskey(_ant_body(service; model), "thinking")
    end
    @test _anthropic_body(model="claude-haiku-5-5", reasoning_effort="none")["thinking"] == Dict("type" => "disabled")
    @test _anthropic_body(model="claude-sonnet-5-5", reasoning_effort="none")["thinking"] == Dict("type" => "between_tools")
    @test !haskey(_anthropic_body(model="claude-opus-4-5", reasoning_effort="low"), "thinking")
end

# https://platform.claude.com/docs/en/build-with-claude/fast-mode.md: speed "fast" on Claude Opus 5.5;
# in research preview on Opus 5 and Opus 4.8, only with the fast-mode-2026-02-01 beta header (HTTP 400
# without it); an error on other models, except Opus 4.6, which runs the request at standard speed.
@testset "encode — service_tier \"fast\" is fast mode, speed \"fast\", only where the model serves it" begin
    for model in ("claude-opus-5-5", "claude-opus-5-5-20261101", "claude-opus-9"), service in (ANTHROPICServiceEndpoint, ANTHROPICServiceEndpoint())
        body = _ant_body(service; model, service_tier="fast")
        @test body["speed"] == "fast" && !haskey(body, "service_tier")
    end
    for model in ("claude-haiku-5-5", "claude-sonnet-5-5", "claude-fable-5-1", "claude-mythos-5-1", "claude-sonnet-5",
                  "claude-opus-4-7", "claude-opus-4-6", "claude-sonnet-4-6", "claude-haiku-4-5", "claude-haiku-4-5-20251001")
        e = _anthropic_err(; model, service_tier="fast")
        @test e isa ArgumentError && occursin(model, e.msg) && occursin("fast", e.msg) && occursin("claude-opus-5-5", e.msg)
    end
    beta = ANTHROPICServiceEndpoint(betas=["fast-mode-2026-02-01"])
    for model in ("claude-opus-5", "claude-opus-4-8")
        e = _anthropic_err(; model, service_tier="fast")
        @test e isa ArgumentError && occursin(model, e.msg) && occursin("fast-mode-2026-02-01", e.msg)
        e = _ant_err(ANTHROPICServiceEndpoint(betas=["thinking-binding-controls-2026-08-01"]); model, service_tier="fast")
        @test e isa ArgumentError && occursin("fast-mode-2026-02-01", e.msg)
        body = _ant_body(beta; model, service_tier="fast")
        @test body["speed"] == "fast" && !haskey(body, "service_tier")
    end
    @test _ant_err(beta; model="claude-haiku-5-5", service_tier="fast") isa ArgumentError    # the header adds no model
    # The Messages API's own tiers go as service_tier, never with a speed.
    for tier in ("auto", "standard_only"), model in ("claude-opus-5-5", "claude-haiku-5-5")
        body = _anthropic_body(; model, service_tier=tier)
        @test body["service_tier"] == tier && !haskey(body, "speed")
    end
    @test !haskey(_anthropic_body(), "speed") && !haskey(_anthropic_body(), "service_tier")
    for tier in ("priority", "flex", "standard", "default")
        e = _anthropic_err(service_tier=tier)
        @test e isa ArgumentError && all(t -> occursin(repr(t), e.msg), ("auto", "standard_only", "fast"))
    end
end

# https://platform.claude.com/docs/en/api/messages.md (Headers; inference_geo) and
# https://platform.claude.com/docs/en/manage-claude/data-residency.md (inference_geo values; HTTP 400
# on the Claude 4.5 models and earlier).
@testset "ANTHROPICServiceEndpoint sends its betas, user profile and workspace as headers, its region in the body" begin
    withenv("ANTHROPIC_API_KEY" => "sk-ant-local-0000") do
        hdr(s) = Dict(UniLM.auth_header(s))
        base = Dict("x-api-key" => "sk-ant-local-0000", "anthropic-version" => "2026-10-21", "Content-Type" => "application/json")
        @test hdr(ANTHROPICServiceEndpoint) == hdr(ANTHROPICServiceEndpoint()) == base
        @test hdr(ANTHROPICServiceEndpoint(betas=["a", "b"])) == merge(base, Dict("anthropic-beta" => "a,b"))
        @test hdr(ANTHROPICServiceEndpoint(betas=["compact-2026-09-04"])) == merge(base, Dict("anthropic-beta" => "compact-2026-09-04"))
        @test hdr(ANTHROPICServiceEndpoint(user_profile_id="uprof_011CZkZCu8hGbp5mYRQgUmz9")) ==
              merge(base, Dict("anthropic-user-profile-id" => "uprof_011CZkZCu8hGbp5mYRQgUmz9"))
        @test hdr(ANTHROPICServiceEndpoint(workspace_id="wrkspc_011CZkZaBF1tNoB5wlCeusgy")) ==
              merge(base, Dict("anthropic-workspace-id" => "wrkspc_011CZkZaBF1tNoB5wlCeusgy"))
        @test hdr(ANTHROPICServiceEndpoint(inference_geo="us")) == base
        every = ANTHROPICServiceEndpoint(betas=["a", "b"], user_profile_id="uprof_1", workspace_id="wrkspc_1", inference_geo="us")
        @test hdr(every) == merge(base, Dict("anthropic-beta" => "a,b", "anthropic-user-profile-id" => "uprof_1",
                                             "anthropic-workspace-id" => "wrkspc_1"))
        @test (@inferred UniLM.auth_header(every)) isa Vector{Pair{String,String}}
    end
    for geo in ("us", "global"), model in ("claude-haiku-5-5", "claude-opus-4-6", "claude-sonnet-4-6", "claude-opus-9")
        @test _ant_body(ANTHROPICServiceEndpoint(inference_geo=geo); model)["inference_geo"] == geo
    end
    for service in (ANTHROPICServiceEndpoint, ANTHROPICServiceEndpoint(), ANTHROPICServiceEndpoint(betas=["a"], user_profile_id="uprof_1"))
        @test !haskey(_ant_body(service), "inference_geo")
    end
    for model in ("claude-opus-4-5", "claude-opus-4-5-20251101", "claude-haiku-4-5", "claude-haiku-4-5-20251001")
        e = _ant_err(ANTHROPICServiceEndpoint(inference_geo="us"); model)
        @test e isa ArgumentError && occursin(model, e.msg) && occursin("inference_geo", e.msg)
        @test !haskey(_ant_body(ANTHROPICServiceEndpoint(); model), "inference_geo")
    end
end

@testset "ANTHROPICServiceEndpoint refuses a header value it cannot send and copies its betas" begin
    caught(f) = try f(); nothing catch e; e end
    for kw in ((betas=[""],), (betas=["  "],), (betas=["a\nb"],), (betas=["a\r\nx-api-key: k"],), (betas=["a,b"],),
               (betas=["a", "b,"],), (betas=Any["a", 1],), (betas=Any[nothing],), (betas=["a\tb"],),
               (user_profile_id="",), (user_profile_id="uprof_1\n",), (user_profile_id="uprof\r1",),
               (workspace_id=" ",), (workspace_id="wrkspc_1\r\n",),
               (inference_geo="eu",), (inference_geo="US",), (inference_geo="",))
        e = caught(() -> ANTHROPICServiceEndpoint(; kw...))
        @test e isa ArgumentError && occursin(string(only(keys(kw)) === :betas ? "beta" : only(keys(kw))), e.msg)
    end
    e = caught(() -> ANTHROPICServiceEndpoint(inference_geo="eu"))
    @test occursin("\"global\"", e.msg) && occursin("\"us\"", e.msg)
    @test occursin("comma", caught(() -> ANTHROPICServiceEndpoint(betas=["a,b"])).msg)
    names = ["a"]
    ep = ANTHROPICServiceEndpoint(betas=names)
    push!(names, "b"); names[1] = "z"
    @test ep.betas == ["a"]
    sub = ANTHROPICServiceEndpoint(betas=[SubString("x-a", 1)], user_profile_id=SubString(" uprof_1", 2),
                                   workspace_id=SubString("wrkspc_1", 1), inference_geo=SubString("us", 1))
    @test sub.betas isa Vector{String} && sub.user_profile_id isa String && sub.workspace_id isa String &&
          sub.inference_geo isa String
    @test sub == ANTHROPICServiceEndpoint(betas=["x-a"], user_profile_id="uprof_1", workspace_id="wrkspc_1", inference_geo="us")
    d = ANTHROPICServiceEndpoint()
    @test d.betas == String[] && isnothing(d.user_profile_id) && isnothing(d.workspace_id) && isnothing(d.inference_geo)
end

@testset "a beta name changed in place is refused where the header is built, by the constructor's rule, before any request" begin
    caught(f) = try f(); nothing catch e; e end
    injected = "a\r\nx-injected: y"
    # (change made in place, the name the constructor refuses with the same message)
    changes = ((betas -> push!(betas, injected), injected), (betas -> (betas[1] = "a,b"), "a,b"),
               (betas -> (betas[1] = "  "), "  "))
    for (change!, name) in changes
        ep = ANTHROPICServiceEndpoint(betas=["thinking-binding-controls-2026-08-01"])
        change!(ep.betas)
        e = withenv(() -> caught(() -> UniLM.auth_header(ep)), "ANTHROPIC_API_KEY" => "sk-ant-local-0000")
        @test e isa ArgumentError && e.msg == caught(() -> ANTHROPICServiceEndpoint(betas=[name])).msg
    end
    usage = merge(_ANTHROPIC_USAGE, Dict("speed" => "standard"))
    srv = _anthropic_capture_server(_ -> JSON.json(_anthropic_message("claude-haiku-5-5", usage)),
                                    _ -> _anthropic_sse("claude-haiku-5-5", usage))
    rule = caught(() -> ANTHROPICServiceEndpoint(betas=[injected])).msg
    try
        withenv("ANTHROPIC_API_KEY" => "sk-ant-local-0000") do
            for stream in (false, true)
                ep = ANTHROPICServiceEndpoint(betas=["thinking-binding-controls-2026-08-01"])
                push!(ep.betas, injected)
                chat = Chat(service=_AnthropicCapture(srv.url, ep), model="claude-haiku-5-5", stream=stream,
                            messages=[Message(Val(:system), "s"), Message(Val(:user), "q")])
                cfg = RequestConfig(max_attempts=3, total_deadline=30.0)
                r = stream ? fetch(chatrequest!(chat; config=cfg)) : chatrequest!(chat; config=cfg)
                @test r isa LLMCallError && r.cause isa ArgumentError && r.cause.msg == rule && length(chat) == 2
            end
        end
        @test isempty(srv.seen)
    finally
        close(srv.server)
    end
end

@testset "ANTHROPICServiceEndpoint instances: equality, hash and show" begin
    kw = (betas=["x", "y"], user_profile_id="uprof_1", workspace_id="wrkspc_1", inference_geo="us")
    a, b = ANTHROPICServiceEndpoint(; kw...), ANTHROPICServiceEndpoint(; kw...)
    @test a == b && isequal(a, b) && hash(a) == hash(b)
    @test ANTHROPICServiceEndpoint() == ANTHROPICServiceEndpoint() && hash(ANTHROPICServiceEndpoint()) == hash(ANTHROPICServiceEndpoint())
    for other in (merge(kw, (betas=["y", "x"],)), merge(kw, (betas=["x"],)), merge(kw, (user_profile_id="uprof_2",)),
                  merge(kw, (workspace_id=nothing,)), merge(kw, (inference_geo="global",)))
        @test a != ANTHROPICServiceEndpoint(; other...)
    end
    @test length(Set([a, b, ANTHROPICServiceEndpoint(), ANTHROPICServiceEndpoint()])) == 2
    @test ANTHROPICServiceEndpoint() != ANTHROPICServiceEndpoint      # an instance, not the type
    @test sprint(show, ANTHROPICServiceEndpoint()) == "ANTHROPICServiceEndpoint()"
    @test sprint(show, a) ==
          "ANTHROPICServiceEndpoint(betas=[\"x\", \"y\"], user_profile_id=\"uprof_1\", workspace_id=\"wrkspc_1\", inference_geo=\"us\")"
    @test sprint(show, ANTHROPICServiceEndpoint(inference_geo="global")) == "ANTHROPICServiceEndpoint(inference_geo=\"global\")"
    @test eval(Meta.parse(sprint(show, a))) == a                       # the printed form builds the same endpoint
    # It holds no credential: the key is read from the environment when a request is sent.
    withenv("ANTHROPIC_API_KEY" => "sk-ant-secret-4242") do
        @test !occursin("sk-ant", sprint(show, a)) && !occursin("sk-ant", repr(Chat(service=a, model="claude-haiku-5-5")))
    end
end

@testset "the bare type and an instance answer every Anthropic method alike" begin
    msgs() = [Message(Val(:system), "s"), Message(Val(:user), "q")]
    plain = Chat(service=ANTHROPICServiceEndpoint, messages=msgs())
    usage = merge(_ANTHROPIC_USAGE, Dict("speed" => "fast"))                # fast mode, US inference
    reply = HTTP.Response(200, [], Vector{UInt8}(JSON.json(_anthropic_message("claude-opus-5-5", usage))))
    rates = DEFAULT_PRICING["claude-opus-5-5"]
    u = decode_response(ANTHROPICServiceEndpoint, reply).usage
    standard = (1000 * 4.0 + 50 * 0.20 + 200 * 4.0 * 1.25 + 100 * 4.0 * 2.0 + 20 * 20.0) / 1e6
    key(chat) = UniLM._answer_key("POST", UniLM._request_path(UniLM.get_url(chat)),
                                  UniLM._redacted(encode_request(chat.service, chat)))
    for service in (ANTHROPICServiceEndpoint, ANTHROPICServiceEndpoint())
        chat = Chat(; service, messages=msgs())
        @test chat.model == "claude-opus-5-5" == UniLM.default_model(service)       # Chat default and model resolution
        @test UniLM.default_max_tokens(service, "claude-haiku-5-5") == 16000
        @test UniLM.get_url(chat) == UniLM.get_url(service, chat) == "https://api.anthropic.com/v1/messages"
        @test provider_capabilities(service) == Set([:chat, :tools, :json_output, :streaming, :models])
        @test has_capability(service, :chat) && !has_capability(service, :embeddings)
        @test_throws ArgumentError UniLM.validate_capability(service, :embeddings, "Embeddings API")
        @test UniLM._service_name(service) === :ANTHROPICServiceEndpoint
        @test withenv(() -> UniLM.auth_header(service), "ANTHROPIC_API_KEY" => "k") ==
              ["x-api-key" => "k", "anthropic-version" => "2026-10-21", "Content-Type" => "application/json"]
        @test encode_request(service, chat) == encode_request(ANTHROPICServiceEndpoint, plain)
        dec = decode_response(service, reply)
        @test dec.usage == u && dec.model == "claude-opus-5-5" && dec.message.content == "hello"
        st = StreamState()
        @test UniLM._sse_dispatch!(service, IOBuffer(), Ref(""), _anthropic_sse("claude-opus-5-5", usage), st) === :done
        @test st.usage == u && _build_stream_message(st).content == "hello"
        # Fast mode bills every rate at 2x and US inference at 1.1x, on top of the cache writes.
        @test isapprox((@inferred UniLM._billed_cost(service, "claude-opus-5-5", u, rates)), standard * 2.0 * 1.1; rtol=1e-12)
        r = LLMSuccess(message=dec.message, self=chat, usage=u, model=dec.model)
        @test UniLM._result_service(r) === service && estimated_cost(r) == UniLM._billed_cost(ANTHROPICServiceEndpoint, "claude-opus-5-5", u, rates)
        f = fork(chat)
        @test f.service === chat.service && encode_request(f.service, f) == encode_request(service, chat)
        @test key(chat) == key(plain)                                               # one recording for both forms
    end
end

@testset "a recorded answer replays for the bare type and an instance; a region is another request" begin
    msgs() = [Message(Val(:system), "s"), Message(Val(:user), "q")]
    plain = Chat(service=ANTHROPICServiceEndpoint, model="claude-haiku-5-5", messages=msgs())
    body = UniLM._redacted(encode_request(ANTHROPICServiceEndpoint, plain))
    mktempdir() do dir
        write(joinpath(dir, UniLM._answer_key("POST", "/v1/messages", body) * ".json"), JSON.json(Dict(
            "request" => Dict("method" => "POST", "path" => "/v1/messages", "body" => JSON.parse(body)),
            "response" => Dict("status" => 200, "request_id" => "req_1", "request_id_header" => "request-id",
                               "body" => _anthropic_message("claude-haiku-5-5", Dict("input_tokens" => 3, "output_tokens" => 1))),
            "recorded_at" => "2026-10-22T00:00:00Z")))
        withenv("ANTHROPIC_API_KEY" => nothing) do
            for service in (ANTHROPICServiceEndpoint, ANTHROPICServiceEndpoint(), ANTHROPICServiceEndpoint(betas=["a"]))
                r = with_recorded_answers(() -> chatrequest!(Chat(; service, model="claude-haiku-5-5", messages=msgs())), dir)
                @test r isa LLMSuccess && r.message.content == "hello" && r.model == "claude-haiku-5-5"
            end
            geo = Chat(service=ANTHROPICServiceEndpoint(inference_geo="us"), model="claude-haiku-5-5", messages=msgs())
            @test_throws ReplayMissError with_recorded_answers(() -> chatrequest!(geo), dir)
        end
    end
end

@testset "an endpoint's headers and region reach the server, streamed or not; a request it refuses sends nothing" begin
    usage = merge(_ANTHROPIC_USAGE, Dict("speed" => "standard"))
    srv = _anthropic_capture_server(_ -> JSON.json(_anthropic_message("claude-haiku-5-5", usage)),
                                    _ -> _anthropic_sse("claude-haiku-5-5", usage))
    ep = ANTHROPICServiceEndpoint(betas=["thinking-binding-controls-2026-08-01", "inline-tools-2026-09-15"],
                                  user_profile_id="uprof_011CZkZCu8hGbp5mYRQgUmz9",
                                  workspace_id="wrkspc_011CZkZaBF1tNoB5wlCeusgy", inference_geo="us")
    cfg = RequestConfig(max_attempts=1, total_deadline=30.0)
    conversation() = [Message(Val(:system), "s"), Message(Val(:user), "q")]
    try
        withenv("ANTHROPIC_API_KEY" => "sk-ant-local-0000") do
            for stream in (false, true)
                chat = Chat(service=_AnthropicCapture(srv.url, ep), model="claude-haiku-5-5", stream=stream,
                            messages=conversation())
                r = stream ? fetch(chatrequest!(chat; config=cfg)) : chatrequest!(chat; config=cfg)
                @test r isa LLMSuccess && r.usage.inference_geo == "us"
            end
            @test length(srv.seen) == 2
            for (headers, body) in srv.seen
                @test headers["anthropic-beta"] == "thinking-binding-controls-2026-08-01,inline-tools-2026-09-15"
                @test headers["anthropic-user-profile-id"] == "uprof_011CZkZCu8hGbp5mYRQgUmz9"
                @test headers["anthropic-workspace-id"] == "wrkspc_011CZkZaBF1tNoB5wlCeusgy"
                @test headers["x-api-key"] == "sk-ant-local-0000" && headers["anthropic-version"] == "2026-10-21"
                @test body["inference_geo"] == "us"
            end
            # Refused before the round trip: a region on Haiku 4.5, fast mode on Haiku 5.5, a cache ttl
            # meant for OpenAI.
            for (model, kw) in (("claude-haiku-4-5", (;)), ("claude-haiku-5-5", (service_tier="fast",)),
                                ("claude-haiku-5-5", (prompt_cache_options=PromptCacheOptions(ttl="30m"),)))
                chat = Chat(; service=_AnthropicCapture(srv.url, ep), model, messages=conversation(), kw...)
                @test_throws ArgumentError chatrequest!(chat; config=cfg)
            end
            @test length(srv.seen) == 2
        end
    finally
        close(srv.server)
    end
end

@testset "encode — a system message after the last reply is refused, naming the fix, whatever turns are left out" begin
    s(x) = Message(Val(:system), x)
    u(x) = Message(Val(:user), x)
    a1 = Message(role=RoleAssistant, content="a1")
    refusal = Message(role=RoleAssistant, refusal_message="Model refused to respond.", finish_reason=CONTENT_FILTER)
    blank = Message(role=RoleAssistant, content="", finish_reason=STOP,
                    provider_content=ProviderContent(:anthropic, Any[Dict{String,Any}("type" => "text", "text" => "")]))
    call = Message(role=RoleAssistant, finish_reason=TOOL_CALLS,
                   tool_calls=[ToolCall(id="toolu_1", func=GPTFunction("f", Dict{String,Any}()))])
    result = Message(role=RoleTool, tool_call_id="toolu_1", content="ok")
    # Prefill taken (Haiku 4.5) or refused (Sonnet 5, Opus 5), append-only (Opus 5.5, an id no row names).
    for model in ("claude-haiku-4-5", "claude-sonnet-5", "claude-opus-5", "claude-opus-5-5", "claude-opus-9")
        # A turn the encoder leaves out after the system message neither makes the request a prefill
        # nor the refusal one about a prefill.
        for tail in (refusal, blank)
            x = _anthropic_err(; model, messages=[s("s0"), u("u1"), a1, s("s2"), tail])
            @test x isa ArgumentError && occursin(model, x.msg) && occursin("system message", x.msg) &&
                  occursin("push a user message", x.msg) && !occursin("rejects messages that end", x.msg)
        end
        # After a tool call the next user turn is its results, which is the fix.
        x = _anthropic_err(; model, messages=[s("s0"), u("u1"), call, s("s2")])
        @test x isa ArgumentError && occursin(model, x.msg) && occursin("results of its tool calls", x.msg) &&
              !occursin("push a user message", x.msg)
        body = _anthropic_body(; model, messages=[s("s0"), u("u1"), call, s("s2"), result])
        @test body["messages"][3]["content"] == [Dict("type" => "tool_result", "tool_use_id" => "toolu_1", "content" => "ok")]
        @test body["messages"][end]["role"] != "assistant"
    end
end
