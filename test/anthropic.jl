# Native Anthropic translation — deterministic, zero-spend unit tests.
using UniLM
using UniLM: encode_request, decode_response, StreamState,
             _build_stream_message, ANTHROPICServiceEndpoint, GPTFunction,
             RoleSystem, RoleUser, RoleAssistant, RoleTool, TOOL_CALLS, STOP, CONTENT_FILTER
using Test, HTTP, JSON

@testset "encode — system split + user turn" begin
    chat = Chat(service=ANTHROPICServiceEndpoint, model="claude-opus-4-8")
    push!(chat, Message(Val(:system), "You are helpful."))
    push!(chat, Message(Val(:user), "Hi"))
    body = JSON.parse(encode_request(ANTHROPICServiceEndpoint, chat))
    @test body["model"] == "claude-opus-4-8"
    @test body["system"] == "You are helpful."
    @test body["max_tokens"] == 4096                     # default supplied
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
    sig = GPTFunctionSignature(name="get_weather", description="Get weather",
        parameters=Dict("type" => "object",
                        "properties" => Dict("location" => Dict("type" => "string")),
                        "required" => ["location"]))
    chat = Chat(service=ANTHROPICServiceEndpoint, model="claude-opus-4-8",
                tools=[GPTTool(func=sig)], tool_choice="auto")
    push!(chat, Message(Val(:system), "s"))
    push!(chat, Message(Val(:user), "weather in Paris?"))
    body = JSON.parse(encode_request(ANTHROPICServiceEndpoint, chat))
    @test length(body["tools"]) == 1
    @test body["tools"][1]["name"] == "get_weather"
    @test body["tools"][1]["description"] == "Get weather"
    @test body["tools"][1]["input_schema"]["type"] == "object"
    @test !haskey(body["tools"][1], "parameters")        # renamed, not OpenAI's key
    @test body["tool_choice"] == Dict("type" => "auto")
end

@testset "encode — multi-turn tool_use → tool_result collapse" begin
    chat = Chat(service=ANTHROPICServiceEndpoint, model="claude-opus-4-8")
    push!(chat, Message(Val(:system), "s"))
    push!(chat, Message(Val(:user), "weather?"))
    tc = GPTToolCall(id="toolu_1", func=GPTFunction("get_weather", Dict("location" => "Paris")))
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
