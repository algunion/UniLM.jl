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

@testset "encode — OpenAI-only moderation and prompt_cache_options fail loud" begin
    for (field, kw) in ((:moderation, (moderation=ModerationConfig(model="omni-moderation-latest", input_mode="block"),)),
                        (:prompt_cache_options, (prompt_cache_options=PromptCacheOptions(mode="implicit", ttl="30m"),)))
        chat = Chat(service=ANTHROPICServiceEndpoint, model="claude-opus-4-8",
                    messages=[Message(Val(:user), "q")]; kw...)
        err = try encode_request(ANTHROPICServiceEndpoint, chat); nothing catch e; e end
        @test err isa ArgumentError &&
              err.msg == "Anthropic Messages does not support $field; it is an OpenAI-only option"
    end
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

# Per-family thinking and effort contract, from
# https://platform.claude.com/docs/en/build-with-claude/effort.md (levels per model) and
# https://platform.claude.com/docs/en/build-with-claude/thinking.md (which families think by
# default, which accept thinking.type "disabled", which are extended-thinking only).
@testset "encode — reasoning_effort per Claude family" begin
    enc(model, effort; kw...) = _anthropic_body(; model, reasoning_effort=effort, kw...)
    err(model, effort) = _anthropic_err(; model, reasoning_effort=effort)
    all5, no_xhigh = ("low", "medium", "high", "xhigh", "max"), ("low", "medium", "high", "max")
    for (model, efforts) in (("claude-fable-5-1", all5), ("claude-mythos-5-1", all5), ("claude-fable-5", all5),
                             ("claude-mythos-5", all5), ("claude-opus-5-5", all5), ("claude-opus-5", all5),
                             ("claude-sonnet-5", all5), ("claude-opus-4-8", all5), ("claude-opus-4-7", all5),
                             ("claude-opus-4-6", no_xhigh), ("claude-sonnet-4-6", no_xhigh),
                             ("claude-mythos-preview", no_xhigh))
        for effort in efforts
            body = enc(model, effort)
            # An explicit effort also turns adaptive thinking on where it is off by default (4.6–4.8).
            @test body["output_config"] == Dict("effort" => effort) &&
                  body["thinking"] == Dict("type" => "adaptive")
        end
    end
    for model in ("claude-opus-4-6", "claude-sonnet-4-6", "claude-mythos-preview")
        e = err(model, "xhigh")
        @test e isa ArgumentError && occursin(model, e.msg) && occursin("reasoning_effort", e.msg)
    end
    # Opus 4.5 is extended-thinking only but takes effort (low, medium, high) without adaptive thinking.
    for effort in ("low", "medium", "high")
        body = enc("claude-opus-4-5-20251101", effort)
        @test body["output_config"] == Dict("effort" => effort) && !haskey(body, "thinking")
    end
    @test err("claude-opus-4-5", "max") isa ArgumentError
    # "none": disable where thinking is on by default and can be turned off, send nothing where it
    # is off by default, refuse where thinking cannot be disabled.
    for model in ("claude-opus-5", "claude-sonnet-5", "claude-sonnet-5-5")
        body = enc(model, "none")
        @test body["thinking"] == Dict("type" => "disabled") && !haskey(body, "output_config")
    end
    for model in ("claude-opus-4-8", "claude-opus-4-7", "claude-opus-4-6", "claude-sonnet-4-6", "claude-opus-4-5")
        body = enc(model, "none")
        @test !haskey(body, "thinking") && !haskey(body, "output_config")
    end
    for model in ("claude-fable-5-1", "claude-mythos-5-1", "claude-fable-5", "claude-mythos-5",
                  "claude-opus-5-5", "claude-mythos-preview")
        e = err(model, "none")
        @test e isa ArgumentError && occursin(model, e.msg) && occursin("reasoning_effort", e.msg)
    end
    # No Claude model has a minimal effort, and an unknown level is not an effort at all.
    for model in ("claude-opus-5-5", "claude-sonnet-5", "claude-opus-4-8", "claude-opus-9")
        e = err(model, "minimal")
        @test e isa ArgumentError && occursin("minimal", e.msg) && occursin("low", e.msg)
        @test err(model, "ultra") isa ArgumentError
    end
    # Haiku 4.5 and older models take no effort at all: any reasoning_effort names the model.
    for model in ("claude-haiku-4-5", "claude-haiku-4-5-20251001", "claude-sonnet-4-5-20250929",
                  "claude-opus-4-1", "claude-opus-4-20250514", "claude-3-7-sonnet-20250219"), effort in ("low", "none")
        e = err(model, effort)
        @test e isa ArgumentError && occursin(model, e.msg) && occursin("reasoning_effort", e.msg)
    end
    # An unknown (future) Claude id maps effort and adaptive thinking without validation.
    body = enc("claude-opus-9", "xhigh")
    @test body["output_config"] == Dict("effort" => "xhigh") && body["thinking"] == Dict("type" => "adaptive")
    @test enc("claude-opus-9", "none")["thinking"] == Dict("type" => "disabled")
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
# ("Limits and feature compatibility") and the Opus 5.5 / Sonnet 5 migration guides.
@testset "encode — Claude model restrictions are validated locally" begin
    tools = [Tool(func=FunctionSignature(name="f", parameters=Dict("type" => "object", "properties" => Dict())))]
    b(model; kw...) = _anthropic_body(; model, kw...)
    e(model; kw...) = _anthropic_err(; model, kw...)
    # temperature is 0..1 on every Claude model (Chat itself allows up to 2).
    for model in ("claude-haiku-4-5", "claude-opus-4-6", "claude-opus-9")
        x = e(model; temperature=1.5)
        @test x isa ArgumentError && occursin("temperature", x.msg)
    end
    @test b("claude-haiku-4-5"; temperature=0.3)["temperature"] == 0.3
    # Models after Opus 4.6 reject non-default sampling on every request.
    for model in ("claude-opus-4-7", "claude-opus-4-8", "claude-sonnet-5", "claude-opus-5", "claude-opus-5-5",
                  "claude-fable-5-1", "claude-mythos-5-1", "claude-fable-5", "claude-mythos-5")
        x = e(model; temperature=0.5)
        @test x isa ArgumentError && occursin(model, x.msg) && occursin("temperature", x.msg)
        x = e(model; top_p=0.9)
        @test x isa ArgumentError && occursin(model, x.msg) && occursin("top_p", x.msg)
        @test b(model; temperature=1.0)["temperature"] == 1.0          # the default is accepted
    end
    @test b("claude-opus-4-6"; temperature=0.3)["temperature"] == 0.3
    @test b("claude-sonnet-4-6"; top_p=0.9)["top_p"] == 0.9
    # Opus 5.5, Fable 5.1 and Mythos 5.1 reject forced tool use on every request.
    for model in ("claude-opus-5-5", "claude-fable-5-1", "claude-mythos-5-1"),
        tc in ("required", UniLM.GPTToolChoice(func="f"))
        x = e(model; tools, tool_choice=tc)
        @test x isa ArgumentError && occursin(model, x.msg) && occursin("tool_choice", x.msg) &&
              occursin("auto", x.msg) && occursin("strict", x.msg)
    end
    @test b("claude-opus-5-5"; tools, tool_choice="auto")["tool_choice"]["type"] == "auto"
    @test b("claude-opus-5"; tools, tool_choice="required")["tool_choice"]["type"] == "any"
    @test b("claude-fable-5"; tools, tool_choice=UniLM.GPTToolChoice(func="f"))["tool_choice"]["type"] == "tool"
    # A trailing assistant turn (response prefill) is rejected from the 4.6 generation on.
    prefill = [Message(Val(:user), "q"), Message(role=RoleAssistant, content="Sure:")]
    for model in ("claude-opus-4-6", "claude-sonnet-4-6", "claude-opus-4-7", "claude-opus-4-8", "claude-sonnet-5",
                  "claude-opus-5", "claude-opus-5-5", "claude-fable-5-1", "claude-mythos-5-1", "claude-fable-5")
        x = e(model; messages=prefill)
        @test x isa ArgumentError && occursin(model, x.msg) && occursin("messages", x.msg)
    end
    for model in ("claude-haiku-4-5", "claude-opus-4-5", "claude-sonnet-4-5", "claude-opus-9")
        @test b(model; messages=prefill)["messages"][end] == Dict("role" => "assistant", "content" => "Sure:")
    end
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
    want = TokenUsage(prompt_tokens=60, completion_tokens=50, total_tokens=110, cached_tokens=30, reasoning_tokens=40)
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
    @test st2.usage == TokenUsage(prompt_tokens=65, completion_tokens=60, total_tokens=125, cached_tokens=30)
end

@testset "encode — assistant turns with no content and no tool calls are skipped" begin
    msgs = [Message(Val(:user), "a"), Message(role=RoleAssistant, content=""), Message(Val(:user), "b"),
            Message(role=RoleAssistant, refusal_message="Model refused to respond.", finish_reason=CONTENT_FILTER),
            Message(Val(:user), "c"), Message(role=RoleAssistant, content="")]
    body = _anthropic_body(model="claude-opus-5-5", messages=msgs)     # no trailing turn left to be a prefill
    @test body["messages"] == [Dict("role" => "user", "content" => c) for c in ("a", "b", "c")]
end

# https://platform.claude.com/docs/en/build-with-claude/mid-conversation-system-messages.md
@testset "encode — a later system message stays in place where the model accepts it" begin
    tc = ToolCall(id="toolu_1", func=GPTFunction("f", Dict{String,Any}()))
    msgs = [Message(Val(:system), "rules"), Message(Val(:system), "more rules"), Message(Val(:user), "q1"),
            Message(role=RoleAssistant, tool_calls=[tc]), Message(role=RoleTool, tool_call_id="toolu_1", content="ok"),
            Message(role=RoleSystem, content="new rule"), Message(role=RoleAssistant, content="a1"),
            Message(Val(:user), "q2")]
    for model in ("claude-opus-5-5", "claude-opus-5", "claude-opus-4-8", "claude-fable-5-1", "claude-mythos-5-1",
                  "claude-fable-5", "claude-mythos-5", "claude-opus-9")
        body = _anthropic_body(; model, messages=msgs)
        @test body["system"] == "rules\n\nmore rules"
        @test [m["role"] for m in body["messages"]] == ["user", "assistant", "user", "system", "assistant", "user"]
        @test body["messages"][3]["content"][1]["type"] == "tool_result"         # results flushed before it
        @test body["messages"][4] == Dict("role" => "system", "content" => "new rule")
    end
    for model in ("claude-sonnet-5", "claude-opus-4-7", "claude-opus-4-6", "claude-haiku-4-5")
        e = _anthropic_err(; model, messages=msgs)
        @test e isa ArgumentError && occursin(model, e.msg) && occursin("system", e.msg)
    end
    # Leading system messages alone are never mid-conversation.
    @test _anthropic_body(model="claude-sonnet-5", messages=msgs[1:3])["system"] == "rules\n\nmore rules"
end
