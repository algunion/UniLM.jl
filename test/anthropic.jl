# Native Anthropic translation — deterministic, zero-spend unit tests.
using UniLM
using UniLM: encode_request, decode_response, decode_stream_chunk, StreamState,
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
