# Native Gemini translation — deterministic, zero-spend unit tests.
using UniLM
using UniLM: encode_request, decode_response, StreamState,
             _build_stream_message, GEMINIServiceEndpoint, GPTFunction, GPTToolChoice,
             GEMINI_NATIVE_BASE, RoleSystem, RoleUser, RoleAssistant, RoleTool,
             TOOL_CALLS, STOP, CONTENT_FILTER
using Test, HTTP, JSON

@testset "routing — model in URL, stream branches on method" begin
    chat = Chat(service=GEMINIServiceEndpoint, model="gemini-3.5-flash")
    @test UniLM.get_url(chat) == "$(GEMINI_NATIVE_BASE)/models/gemini-3.5-flash:generateContent"
    schat = Chat(service=GEMINIServiceEndpoint, model="gemini-3.5-flash", stream=true)
    @test UniLM.get_url(schat) == "$(GEMINI_NATIVE_BASE)/models/gemini-3.5-flash:streamGenerateContent?alt=sse"
end

@testset "routing — a separator-bearing model cannot re-shape the URL" begin
    # The model is data in a path segment; the `:generateContent` verb colon and
    # the `?alt=sse` query are the template's own structure and stay literal.
    m = "a b/../c?x=1#f"
    enc = "a%20b%2F..%2Fc%3Fx%3D1%23f"
    @test UniLM.get_url(Chat(service=GEMINIServiceEndpoint, model=m)) ==
          "$(GEMINI_NATIVE_BASE)/models/$enc:generateContent"
    @test UniLM.get_url(Chat(service=GEMINIServiceEndpoint, model=m, stream=true)) ==
          "$(GEMINI_NATIVE_BASE)/models/$enc:streamGenerateContent?alt=sse"
end

@testset "auth — x-goog-api-key" begin
    withenv("GEMINI_API_KEY" => "test-key") do
        h = Dict(UniLM.auth_header(GEMINIServiceEndpoint))
        @test h["x-goog-api-key"] == "test-key"
        @test !haskey(h, "Authorization")            # NOT Bearer
    end
end

@testset "capabilities & default" begin
    # :json_output — response_format maps to generationConfig.responseFormat (golden below)
    @test UniLM.provider_capabilities(GEMINIServiceEndpoint) == Set([:chat, :tools, :streaming, :agentic, :json_output])
    @test has_capability(GEMINIServiceEndpoint, :json_output)
    @test UniLM.default_model(GEMINIServiceEndpoint) == "gemini-3.8-flash"
    @test_throws ArgumentError UniLM._api_base_url(GEMINIServiceEndpoint)
end

@testset "encode — system → systemInstruction, user turn" begin
    chat = Chat(service=GEMINIServiceEndpoint, model="gemini-3.5-flash")
    push!(chat, Message(Val(:system), "You are helpful."))
    push!(chat, Message(Val(:user), "Hi"))
    body = JSON.parse(encode_request(GEMINIServiceEndpoint, chat))
    @test body["systemInstruction"]["parts"][1]["text"] == "You are helpful."
    @test length(body["contents"]) == 1
    @test body["contents"][1]["role"] == "user"
    @test body["contents"][1]["parts"][1]["text"] == "Hi"
    @test !haskey(body, "generationConfig")                 # nothing set → omitted
    @test !haskey(body, "stream")                            # stream is URL-only
end

@testset "encode — generationConfig present only when set" begin
    chat = Chat(service=GEMINIServiceEndpoint, model="gemini-3.5-flash",
                max_tokens=256, temperature=0.5, stop=["END"])
    push!(chat, Message(Val(:system), "s"))
    push!(chat, Message(Val(:user), "u"))
    body = JSON.parse(encode_request(GEMINIServiceEndpoint, chat))
    @test body["generationConfig"]["maxOutputTokens"] == 256
    @test body["generationConfig"]["temperature"] == 0.5
    @test body["generationConfig"]["stopSequences"] == ["END"]
end

@testset "encode — tools → functionDeclarations + toolConfig" begin
    sig = FunctionSignature(name="get_weather", description="Get weather",
        parameters=Dict("type" => "object",
                        "properties" => Dict("location" => Dict("type" => "string")),
                        "required" => ["location"]))
    chat = Chat(service=GEMINIServiceEndpoint, model="gemini-3.5-flash",
                tools=[Tool(func=sig)], tool_choice="auto")
    push!(chat, Message(Val(:system), "s"))
    push!(chat, Message(Val(:user), "weather?"))
    body = JSON.parse(encode_request(GEMINIServiceEndpoint, chat))
    fd = body["tools"][1]["functionDeclarations"][1]
    @test fd["name"] == "get_weather"
    @test fd["description"] == "Get weather"
    @test fd["parametersJsonSchema"]["type"] == "object"
    @test body["toolConfig"]["functionCallingConfig"]["mode"] == "AUTO"
end

@testset "encode — multi-turn: functionCall echo (+thoughtSignature) & functionResponse correlation" begin
    chat = Chat(service=GEMINIServiceEndpoint, model="gemini-3.5-flash")
    # push! requires the first-ever message on a fresh Chat to be system (api.jl:663-671)
    push!(chat, Message(Val(:system), "s"))
    push!(chat, Message(Val(:user), "weather?"))
    tc = ToolCall(id="fc_1", func=GPTFunction("get_weather", Dict("location" => "Paris")),
                     thought_signature="SIG123")
    push!(chat, Message(role=RoleAssistant, tool_calls=[tc], finish_reason=TOOL_CALLS))
    push!(chat, Message(role=RoleTool, tool_call_id="fc_1", content="72F"))
    body = JSON.parse(encode_request(GEMINIServiceEndpoint, chat))
    c = body["contents"]
    @test [x["role"] for x in c] == ["user", "model", "user"]
    fc = c[2]["parts"][1]
    @test fc["functionCall"]["name"] == "get_weather"
    @test fc["functionCall"]["args"] == Dict("location" => "Paris")
    @test fc["thoughtSignature"] == "SIG123"                 # echoed
    fr = c[3]["parts"][1]["functionResponse"]
    @test fr["name"] == "get_weather"                        # correlated by id
    @test fr["id"] == "fc_1"
    @test fr["response"] == Dict("result" => "72F")          # string wrapped as object
end

@testset "encode — orphan functionResponse fails loud" begin
    msgs = [Message(Val(:user), "hi"),
            Message(role=RoleTool, tool_call_id="ghost", content="x")]
    chat = Chat(service=GEMINIServiceEndpoint, model="gemini-3.5-flash", messages=msgs)
    @test_throws ArgumentError encode_request(GEMINIServiceEndpoint, chat)
end

@testset "encode — functionResponse passes a JSON-object tool result through unwrapped" begin
    chat = Chat(service=GEMINIServiceEndpoint, model="gemini-3.5-flash")
    push!(chat, Message(Val(:system), "s"))
    push!(chat, Message(Val(:user), "weather?"))
    tc = ToolCall(id="fc_1", func=GPTFunction("get_weather", Dict("location" => "Paris")))
    push!(chat, Message(role=RoleAssistant, tool_calls=[tc], finish_reason=TOOL_CALLS))
    push!(chat, Message(role=RoleTool, tool_call_id="fc_1", content="{\"temp_f\":72}"))
    body = JSON.parse(encode_request(GEMINIServiceEndpoint, chat))
    fr = body["contents"][end]["parts"][1]["functionResponse"]
    @test fr["response"] == Dict("temp_f" => 72)   # object-valued JSON string parsed through, NOT wrapped
end

# JSON.parse(::IO) reads the stream before parsing, so an interrupt raised by that read
# stands in for a Ctrl-C landing while a tool result is being encoded.
struct _InterruptOnRead <: IO end
Base.read(::_InterruptOnRead) = throw(InterruptException())

@testset "encode — an interrupt while encoding a tool result propagates" begin
    # The parse-failure fallback wraps a non-JSON tool result; it must not swallow the
    # user's interrupt as if it were one.
    @test_throws InterruptException UniLM._gemini_tool_response(_InterruptOnRead())
    @test UniLM._gemini_tool_response("72F") == Dict("result" => "72F")
end

@testset "encode — consecutive tool results collapse into one user turn" begin
    chat = Chat(service=GEMINIServiceEndpoint, model="gemini-3.5-flash")
    push!(chat, Message(Val(:system), "s"))
    push!(chat, Message(Val(:user), "weather in Paris and London?"))
    tc1 = ToolCall(id="fc_1", func=GPTFunction("get_weather", Dict("location" => "Paris")))
    tc2 = ToolCall(id="fc_2", func=GPTFunction("get_weather", Dict("location" => "London")))
    push!(chat, Message(role=RoleAssistant, tool_calls=[tc1, tc2], finish_reason=TOOL_CALLS))
    push!(chat, Message(role=RoleTool, tool_call_id="fc_1", content="72F"))
    push!(chat, Message(role=RoleTool, tool_call_id="fc_2", content="60F"))
    body = JSON.parse(encode_request(GEMINIServiceEndpoint, chat))
    c = body["contents"]
    @test [x["role"] for x in c] == ["user", "model", "user"]   # two tool results → ONE user turn
    @test length(c[2]["parts"]) == 2                            # two functionCall parts
    resp = c[3]["parts"]
    @test length(resp) == 2                                     # two functionResponse parts
    @test resp[1]["functionResponse"]["id"] == "fc_1"
    @test resp[2]["functionResponse"]["id"] == "fc_2"
    @test resp[1]["functionResponse"]["name"] == "get_weather"
end

@testset "decode — plain text" begin
    body = JSON.json(Dict("candidates" => [Dict(
        "content" => Dict("role" => "model", "parts" => [Dict("text" => "Hello there")]),
        "finishReason" => "STOP")],
        "usageMetadata" => Dict("promptTokenCount" => 10, "candidatesTokenCount" => 3,
                                "totalTokenCount" => 13)))
    r = decode_response(GEMINIServiceEndpoint, HTTP.Response(200, [], Vector{UInt8}(body)))
    @test r.message.role == RoleAssistant
    @test r.message.content == "Hello there"
    @test r.message.finish_reason == STOP
    @test r.usage.prompt_tokens == 10
    @test r.usage.completion_tokens == 3
    @test r.usage.total_tokens == 13
end

@testset "decode — text + functionCall (thoughtSignature captured; presence → TOOL_CALLS)" begin
    body = JSON.json(Dict("candidates" => [Dict(
        "content" => Dict("role" => "model", "parts" => [
            Dict("text" => "Let me check."),
            Dict("functionCall" => Dict("id" => "fc_9", "name" => "get_weather",
                                        "args" => Dict("location" => "Paris")),
                 "thoughtSignature" => "SIGX")]),
        "finishReason" => "STOP")],                        # Gemini says STOP even for tool calls
        "usageMetadata" => Dict("promptTokenCount" => 20, "candidatesTokenCount" => 15,
                                "totalTokenCount" => 35)))
    r = decode_response(GEMINIServiceEndpoint, HTTP.Response(200, [], Vector{UInt8}(body)))
    @test r.message.finish_reason == TOOL_CALLS
    @test r.message.content == "Let me check."
    @test r.message.tool_calls[1].id == "fc_9"
    @test r.message.tool_calls[1].func.name == "get_weather"
    @test r.message.tool_calls[1].func.arguments == Dict("location" => "Paris")
    @test r.message.tool_calls[1].thought_signature == "SIGX"
end

@testset "decode — MAX_TOKENS → length" begin
    body = JSON.json(Dict("candidates" => [Dict(
        "content" => Dict("role" => "model", "parts" => [Dict("text" => "partial")]),
        "finishReason" => "MAX_TOKENS")],
        "usageMetadata" => Dict("promptTokenCount" => 5, "candidatesTokenCount" => 100,
                                "totalTokenCount" => 105)))
    r = decode_response(GEMINIServiceEndpoint, HTTP.Response(200, [], Vector{UInt8}(body)))
    @test r.message.finish_reason == "length"
    @test r.message.content == "partial"
end

@testset "decode — an empty-text turn is reported truthfully, never fabricated" begin
    # Observed with thinking models: the whole completion budget goes to thought
    # tokens and the turn comes back well-formed with no text part. Substituting
    # prose the provider never sent injects fabricated content into the
    # conversation and into the next request's history.
    budget_spent = """
    {"candidates":[{"finishReason":"MAX_TOKENS"}],
     "usageMetadata":{"promptTokenCount":9,"candidatesTokenCount":0,
                      "thoughtsTokenCount":128,"totalTokenCount":137}}
    """
    r = decode_response(GEMINIServiceEndpoint, HTTP.Response(200, [], Vector{UInt8}(budget_spent)))
    @test r.message.content == ""
    @test !occursin("No response from the model", something(r.message.content, ""))
    @test isnothing(r.message.tool_calls)
    @test r.message.finish_reason == "length"
    @test r.usage.reasoning_tokens == 128

    # Same rule for an explicit empty parts array under STOP.
    empty_stop = """
    {"candidates":[{"content":{"role":"model","parts":[]},"finishReason":"STOP"}],
     "usageMetadata":{"promptTokenCount":4,"candidatesTokenCount":0,"totalTokenCount":4}}
    """
    r2 = decode_response(GEMINIServiceEndpoint, HTTP.Response(200, [], Vector{UInt8}(empty_stop)))
    @test r2.message.content == ""
    @test r2.message.finish_reason == STOP

    # A tool-only turn keeps content nothing (unchanged) — no placeholder there either.
    tool_only = """
    {"candidates":[{"content":{"role":"model","parts":[
        {"functionCall":{"id":"fc1","name":"ping","args":{}}}]},"finishReason":"STOP"}],
     "usageMetadata":{"promptTokenCount":2,"candidatesTokenCount":1,"totalTokenCount":3}}
    """
    r3 = decode_response(GEMINIServiceEndpoint, HTTP.Response(200, [], Vector{UInt8}(tool_only)))
    @test isnothing(r3.message.content)
    @test r3.message.tool_calls[1].func.name == "ping"
end

@testset "decode — zero candidates fails loud instead of inventing a turn" begin
    # No candidate is not "the model said nothing": there is no assistant turn to
    # report. Throw so the verb returns its typed error result carrying the
    # provider's own diagnostics, rather than a Message nobody sent.
    blocked = """{"promptFeedback":{"blockReason":"SAFETY"},
                  "usageMetadata":{"promptTokenCount":3,"totalTokenCount":3}}"""
    err = try
        decode_response(GEMINIServiceEndpoint, HTTP.Response(200, [], Vector{UInt8}(blocked)))
        nothing
    catch e
        e
    end
    @test err isa ErrorException
    @test occursin("candidate", lowercase(err.msg))
    @test occursin("SAFETY", err.msg)          # the provider's diagnostics survive
end

@testset "decode — SAFETY → content_filter refusal" begin
    body = JSON.json(Dict("candidates" => [Dict(
        "content" => Dict("role" => "model", "parts" => []),
        "finishReason" => "SAFETY")],
        "usageMetadata" => Dict("promptTokenCount" => 4, "candidatesTokenCount" => 0,
                                "totalTokenCount" => 4)))
    r = decode_response(GEMINIServiceEndpoint, HTTP.Response(200, [], Vector{UInt8}(body)))
    @test r.message.finish_reason == CONTENT_FILTER
    @test !isnothing(r.message.refusal_message)
end

@testset "decode — an unlisted finishReason passes through lowercased, never as stop" begin
    body = JSON.json(Dict("candidates" => [Dict(
        "content" => Dict("role" => "model", "parts" => [Dict("text" => "partial")]),
        "finishReason" => "A_REASON_ADDED_LATER")],      # open enum: a value this code never saw
        "usageMetadata" => Dict("promptTokenCount" => 5, "candidatesTokenCount" => 2,
                                "totalTokenCount" => 7)))
    r = decode_response(GEMINIServiceEndpoint, HTTP.Response(200, [], Vector{UInt8}(body)))
    @test r.message.finish_reason == "a_reason_added_later"
    @test r.message.content == "partial"
end

@testset "decode — a candidate without a finishReason reports none (not stop, not tool_calls)" begin
    call = Dict("functionCall" => Dict("id" => "fc_1", "name" => "ping", "args" => Dict()))
    for parts in (Any[Dict("text" => "partial")], Any[call])
        body = JSON.json(Dict("candidates" => [Dict("content" => Dict("role" => "model", "parts" => parts))]))
        msg = decode_response(GEMINIServiceEndpoint, HTTP.Response(200, [], Vector{UInt8}(body))).message
        @test isnothing(msg.finish_reason)
    end
end

@testset "finishReason — one mapping on both paths; failures never read as stop" begin
    # Values from the FinishReason enum (ai.google.dev/api/generate-content). STOP is the
    # only normal completion, and only a STOP turn with function calls is a tool-call
    # turn: calls under any other reason keep that reason, so they are never dispatched.
    filters = ("SAFETY", "RECITATION", "BLOCKLIST", "PROHIBITED_CONTENT", "SPII", "IMAGE_SAFETY",
               "IMAGE_PROHIBITED_CONTENT", "IMAGE_RECITATION")
    cases = [("STOP", false, STOP), ("STOP", true, TOOL_CALLS),
             ("MAX_TOKENS", false, "length"), ("MAX_TOKENS", true, "length"),
             [(wire, calls, CONTENT_FILTER) for wire in filters for calls in (false, true)]...,
             # image outcomes the enum does not describe as a block
             ("IMAGE_OTHER", false, "image_other"), ("NO_IMAGE", false, "no_image"),
             ("MALFORMED_FUNCTION_CALL", false, "malformed_function_call"),
             ("UNEXPECTED_TOOL_CALL", false, "unexpected_tool_call"),
             ("TOO_MANY_TOOL_CALLS", true, "too_many_tool_calls"),
             ("MISSING_THOUGHT_SIGNATURE", true, "missing_thought_signature"),
             ("OTHER", false, "other"), ("FINISH_REASON_UNSPECIFIED", false, "finish_reason_unspecified")]
    call = Dict("functionCall" => Dict("id" => "fc_1", "name" => "ping", "args" => Dict()))
    chunk(candidate) = JSON.json(Dict("candidates" => [candidate]))
    @testset "$wire$(calls ? " + functionCall" : "")" for (wire, calls, want) in cases
        parts = calls ? Any[call] : Any[]
        body = chunk(Dict("content" => Dict("role" => "model", "parts" => parts), "finishReason" => wire))
        msg = decode_response(GEMINIServiceEndpoint, HTTP.Response(200, [], Vector{UInt8}(body))).message
        @test msg.finish_reason == want
        @test isnothing(msg.tool_calls) == !calls          # the calls are still reported
        # Streamed: the parts arrive first, the finishReason on a later chunk.
        state = StreamState()
        calls && UniLM.handle_sse_event!(GEMINIServiceEndpoint, "",
            chunk(Dict("content" => Dict("role" => "model", "parts" => parts))), state)
        UniLM.handle_sse_event!(GEMINIServiceEndpoint, "", chunk(Dict("finishReason" => wire)), state)
        @test state.finish_reason == want
    end
end

@testset "decode — usage: cached is a subset of prompt; thoughts bill as output" begin
    body = JSON.json(Dict("candidates" => [Dict(
        "content" => Dict("role" => "model", "parts" => [Dict("text" => "hi")]),
        "finishReason" => "STOP")],
        "usageMetadata" => Dict("promptTokenCount" => 104, "candidatesTokenCount" => 5,
                                "thoughtsTokenCount" => 20, "cachedContentTokenCount" => 100,
                                "totalTokenCount" => 129)))
    r = decode_response(GEMINIServiceEndpoint, HTTP.Response(200, [], Vector{UInt8}(body)))
    @test r.usage.prompt_tokens == 104                     # promptTokenCount already includes cached
    @test r.usage.cached_tokens == 100
    @test r.usage.completion_tokens == 25                  # candidates(5) + thoughts(20), billed as output
    @test r.usage.reasoning_tokens == 20
    @test r.usage.total_tokens == 129
end

@testset "stream — text deltas + final usage/finishReason (EOF-terminated, never :done)" begin
    lines = [
        "data: " * JSON.json(Dict("candidates" => [Dict("content" =>
            Dict("role" => "model", "parts" => [Dict("text" => "Hello")]))],
            "usageMetadata" => Dict("promptTokenCount" => 8))),
        "",
        "data: " * JSON.json(Dict("candidates" => [Dict("content" =>
            Dict("role" => "model", "parts" => [Dict("text" => " world")]))])),
        "",
        "data: " * JSON.json(Dict("candidates" => [Dict(
            "content" => Dict("role" => "model", "parts" => [Dict("text" => "")]),
            "finishReason" => "STOP")],
            "usageMetadata" => Dict("promptTokenCount" => 8, "candidatesTokenCount" => 5,
                                    "totalTokenCount" => 13))),
    ]
    state = StreamState()
    st = UniLM._sse_dispatch!(GEMINIServiceEndpoint, IOBuffer(), Ref(""), join(lines, "\n") * "\n", state)
    # Gemini has no sentinel — the handler NEVER returns :done; the
    # driver reads to EOF and finalizes on the recorded finishReason.
    @test st === :continue
    @test state.finish_reason == STOP
    @test state.usage.completion_tokens == 5
    @test state.usage.prompt_tokens == 8
    msg = _build_stream_message(state)
    @test msg.content == "Hello world"
    @test msg.finish_reason == STOP
end

@testset "stream — functionCall + thoughtSignature via _build_stream_message" begin
    lines = [
        "data: " * JSON.json(Dict("candidates" => [Dict("content" =>
            Dict("role" => "model", "parts" => [Dict(
                "functionCall" => Dict("id" => "fc_7", "name" => "get_weather",
                                       "args" => Dict("location" => "Paris")),
                "thoughtSignature" => "SIG7")]))])),
        "",
        "data: " * JSON.json(Dict("candidates" => [Dict(
            "content" => Dict("role" => "model", "parts" => [Dict("text" => "")]),
            "finishReason" => "STOP")],
            "usageMetadata" => Dict("promptTokenCount" => 12, "candidatesTokenCount" => 20,
                                    "totalTokenCount" => 32))),
    ]
    state = StreamState()
    st = UniLM._sse_dispatch!(GEMINIServiceEndpoint, IOBuffer(), Ref(""), join(lines, "\n") * "\n", state)
    @test st === :continue
    @test state.finish_reason == TOOL_CALLS                 # functionCall present overrides STOP
    msg = _build_stream_message(state)
    @test msg.finish_reason == TOOL_CALLS
    @test length(msg.tool_calls) == 1
    @test msg.tool_calls[1].id == "fc_7"
    @test msg.tool_calls[1].func.name == "get_weather"
    @test msg.tool_calls[1].func.arguments == Dict("location" => "Paris")
    @test msg.tool_calls[1].thought_signature == "SIG7"
end

@testset "stream — a refusal is recorded once, however many chunks follow it" begin
    blocked = """{"candidates":[{"content":{"role":"model","parts":[]},"finishReason":"SAFETY"}]}"""
    state = StreamState()
    # The filtered chunk, then a trailing candidate chunk carrying the usage totals.
    for payload in (blocked, """{"candidates":[{"content":{"role":"model","parts":[]}}],
                                 "usageMetadata":{"promptTokenCount":4,"totalTokenCount":4}}""")
        UniLM.handle_sse_event!(GEMINIServiceEndpoint, "", payload, state)
    end
    streamed = _build_stream_message(state)
    decoded = decode_response(GEMINIServiceEndpoint, HTTP.Response(200, [], Vector{UInt8}(blocked))).message
    @test streamed.refusal_message == decoded.refusal_message == "Model response blocked by safety filter."
    @test streamed.finish_reason == decoded.finish_reason == CONTENT_FILTER
end

@testset "decode+encode — parallel DIFFERENT-function tool calls correlate by id, not position" begin
    respbody = JSON.json(Dict("candidates" => [Dict(
        "content" => Dict("role" => "model", "parts" => [
            Dict("functionCall" => Dict("id" => "fc_a", "name" => "get_weather", "args" => Dict("location" => "Paris"))),
            Dict("functionCall" => Dict("id" => "fc_b", "name" => "get_time",    "args" => Dict("zone" => "CET")))]),
        "finishReason" => "STOP")],
        "usageMetadata" => Dict("promptTokenCount" => 10, "candidatesTokenCount" => 8, "totalTokenCount" => 18)))
    r = decode_response(GEMINIServiceEndpoint, HTTP.Response(200, [], Vector{UInt8}(respbody)))
    @test r.message.finish_reason == TOOL_CALLS
    @test length(r.message.tool_calls) == 2
    @test r.message.tool_calls[1].id == "fc_a" && r.message.tool_calls[1].func.name == "get_weather"
    @test r.message.tool_calls[2].id == "fc_b" && r.message.tool_calls[2].func.name == "get_time"
    # Round-trip: feed both results back in REVERSED order; correlation must be by id, not position.
    chat = Chat(service=GEMINIServiceEndpoint, model="gemini-3.5-flash")
    push!(chat, Message(Val(:system), "s"))
    push!(chat, Message(Val(:user), "weather and time?"))
    push!(chat, r.message)
    push!(chat, Message(role=RoleTool, tool_call_id="fc_b", content="12:00"))
    push!(chat, Message(role=RoleTool, tool_call_id="fc_a", content="72F"))
    body = JSON.parse(encode_request(GEMINIServiceEndpoint, chat))
    resp = body["contents"][end]["parts"]
    @test length(resp) == 2
    byid = Dict(p["functionResponse"]["id"] => p["functionResponse"]["name"] for p in resp)
    @test byid["fc_a"] == "get_weather"    # correct name despite reversed push order
    @test byid["fc_b"] == "get_time"
end

@testset "decode — provider-native parts captured (text-part thoughtSignature survives)" begin
    body = """
    {"candidates":[{"content":{"role":"model","parts":[
       {"text":"Weighing options.","thoughtSignature":"tsig=="},
       {"functionCall":{"id":"fc1","name":"get_weather","args":{"city":"Oslo"}}}]},
      "finishReason":"STOP"}],
     "usageMetadata":{"promptTokenCount":1,"candidatesTokenCount":2,"totalTokenCount":3}}
    """
    dec = UniLM.decode_response(GEMINIServiceEndpoint, HTTP.Response(200, [], Vector{UInt8}(body)))
    pc = dec.message.provider_content
    @test pc isa ProviderContent && pc.provider === :gemini
    @test length(pc.blocks) == 2
    @test pc.blocks[1]["thoughtSignature"] == "tsig=="   # dropped by the neutral IR, kept here
    @test dec.message.tool_calls[1].func.name == "get_weather"

    # No parts (e.g. safety block with empty candidate content) → no capture.
    blocked = """{"candidates":[{"finishReason":"SAFETY"}],
        "usageMetadata":{"promptTokenCount":1,"candidatesTokenCount":0,"totalTokenCount":1}}"""
    dec2 = UniLM.decode_response(GEMINIServiceEndpoint, HTTP.Response(200, [], Vector{UInt8}(blocked)))
    @test isnothing(dec2.message.provider_content)
end

@testset "encode — provider-native parts echoed verbatim, correlation preserved" begin
    parts = Any[
        Dict{String,Any}("text" => "Weighing options.", "thoughtSignature" => "tsig=="),
        Dict{String,Any}("functionCall" => Dict{String,Any}(
            "id" => "fc1", "name" => "get_weather", "args" => Dict{String,Any}("city" => "Oslo"))),
    ]
    tc = [ToolCall(id="fc1", func=GPTFunction("get_weather", Dict{String,Any}("city" => "Oslo")))]
    m = Message(role=UniLM.RoleAssistant, tool_calls=tc,
                provider_content=ProviderContent(:gemini, parts))
    msgs = [Message(role=UniLM.RoleUser, content="w?"), m,
            Message(role=UniLM.RoleTool, content="12C", tool_call_id="fc1")]
    _, contents = UniLM._gemini_contents(msgs)
    # Model turn is the captured parts array, identical object.
    @test contents[2][:parts] === parts
    # functionResponse correlation still resolved through the neutral tool_calls.
    fr = contents[3][:parts][1][:functionResponse]
    @test fr[:name] == "get_weather" && fr[:id] == "fc1"

    # Cross-provider tag → reconstruction (no thoughtSignature text part).
    m_anth = Message(role=UniLM.RoleAssistant, content="hi", tool_calls=tc,
                     provider_content=ProviderContent(:anthropic, parts))
    rec = UniLM._gemini_model_parts(m_anth, Dict{String,String}())
    @test rec isa Vector{Dict{Symbol,Any}} && rec[1] == Dict{Symbol,Any}(:text => "hi")
end

@testset "encode — empty gemini blocks reconstruct (never echo an empty parts array)" begin
    # Symmetry with the Anthropic empty-blocks guard: a captured-but-empty
    # ProviderContent(:gemini, Any[]) must fall through to reconstruction, not
    # echo an empty model turn (which would drop the assistant text on the wire).
    m = Message(role=UniLM.RoleAssistant, content="hi",
                provider_content=ProviderContent(:gemini, Any[]))
    rec = UniLM._gemini_model_parts(m, Dict{String,String}())
    @test rec isa Vector{Dict{Symbol,Any}} && rec == [Dict{Symbol,Any}(:text => "hi")]
end

@testset "encode — a model turn with nothing to send is left out of the next request" begin
    # A refusal and a turn whose whole budget went to thinking carry no text and no
    # function call. Sent as `{"role": "model", "parts": []}` they break the follow-up
    # request, so continuing the conversation drops them.
    decode(body) = decode_response(GEMINIServiceEndpoint, HTTP.Response(200, [], Vector{UInt8}(body))).message
    refused = decode("""{"candidates":[{"finishReason":"SAFETY"}]}""")
    spent = decode("""{"candidates":[{"finishReason":"MAX_TOKENS"}]}""")
    state = StreamState()
    UniLM.handle_sse_event!(GEMINIServiceEndpoint, "", """{"candidates":[{"finishReason":"SAFETY"}]}""", state)
    streamed = _build_stream_message(state)
    @test !isnothing(refused.refusal_message) && spent.content == "" && !isnothing(streamed.refusal_message)
    chat = Chat(service=GEMINIServiceEndpoint, messages=[
        Message(role=RoleUser, content="q1"), refused, Message(role=RoleUser, content="q2"), spent,
        Message(role=RoleUser, content="q3"), streamed, Message(role=RoleUser, content="q4")])
    contents = JSON.parse(encode_request(GEMINIServiceEndpoint, chat))["contents"]
    @test contents == [Dict("role" => "user", "parts" => [Dict("text" => q)]) for q in ("q1", "q2", "q3", "q4")]
end

@testset "decode — malformed non-vector parts → no capture, no throw" begin
    # Symmetry with the Anthropic malformed-content guard: if `content.parts` is a
    # stray object instead of an array, decode must not throw and must not capture
    # provider_content (echoing a non-array back would 400). The message still
    # decodes to a well-formed assistant turn.
    malformed = """{"candidates":[{"content":{"role":"model","parts":{"text":"x"}},
        "finishReason":"STOP"}],
        "usageMetadata":{"promptTokenCount":1,"candidatesTokenCount":1,"totalTokenCount":2}}"""
    dec = UniLM.decode_response(GEMINIServiceEndpoint, HTTP.Response(200, [], Vector{UInt8}(malformed)))
    @test isnothing(dec.message.provider_content)
    @test dec.message.role == UniLM.RoleAssistant
end

@testset "decode — id-less parallel calls get unique synthetic ids" begin
    body = """
    {"candidates":[{"content":{"role":"model","parts":[
       {"functionCall":{"name":"get_weather","args":{"city":"Oslo"}}},
       {"functionCall":{"name":"get_time","args":{"tz":"CET"}}}]},
      "finishReason":"STOP"}],
     "usageMetadata":{"promptTokenCount":1,"candidatesTokenCount":2,"totalTokenCount":3}}
    """
    dec = UniLM.decode_response(GEMINIServiceEndpoint, HTTP.Response(200, [], Vector{UInt8}(body)))
    tcs = dec.message.tool_calls
    @test length(tcs) == 2 && allunique([tc.id for tc in tcs])
    @test all(tc -> startswith(tc.id, "unilm_call_"), tcs)

    # Mixed: a real id is preserved; only the missing one is synthesized.
    mixed = """
    {"candidates":[{"content":{"role":"model","parts":[
       {"functionCall":{"id":"real_1","name":"a","args":{}}},
       {"functionCall":{"name":"b","args":{}}}]},
      "finishReason":"STOP"}],
     "usageMetadata":{"promptTokenCount":1,"candidatesTokenCount":1,"totalTokenCount":2}}
    """
    dec2 = UniLM.decode_response(GEMINIServiceEndpoint, HTTP.Response(200, [], Vector{UInt8}(mixed)))
    ids2 = [tc.id for tc in dec2.message.tool_calls]
    @test ids2[1] == "real_1" && startswith(ids2[2], "unilm_call_") && allunique(ids2)
end

@testset "encode — synthetic ids are omitted from the wire (both part kinds)" begin
    tcs = [ToolCall(id="unilm_call_1", func=UniLM.GPTFunction("get_weather", Dict{String,Any}("city" => "Oslo"))),
           ToolCall(id="real_2",       func=UniLM.GPTFunction("get_time",    Dict{String,Any}("tz" => "CET")))]
    m = Message(role=UniLM.RoleAssistant, tool_calls=tcs)
    msgs = [Message(role=UniLM.RoleUser, content="hi"), m,
            Message(role=UniLM.RoleTool, content="12C",   tool_call_id="unilm_call_1"),
            Message(role=UniLM.RoleTool, content="14:00", tool_call_id="real_2")]
    _, contents = UniLM._gemini_contents(msgs)
    parts = contents[2][:parts]
    @test !haskey(parts[1][:functionCall], :id)              # synthetic → omitted
    @test parts[2][:functionCall][:id] == "real_2"           # real → kept
    frs = [p[:functionResponse] for p in contents[3][:parts]]
    @test frs[1][:name] == "get_weather" && !haskey(frs[1], :id)
    @test frs[2][:name] == "get_time"    && frs[2][:id] == "real_2"
end

@testset "stream — id-less functionCall parts synthesize unique ids" begin
    state = UniLM.StreamState()
    payload = "{\"candidates\":[{\"content\":{\"role\":\"model\",\"parts\":[" *
              "{\"functionCall\":{\"name\":\"get_weather\",\"args\":{\"city\":\"Oslo\"}}}," *
              "{\"functionCall\":{\"name\":\"get_time\",\"args\":{\"tz\":\"CET\"}}}]}}]}"
    UniLM.handle_sse_event!(GEMINIServiceEndpoint, "", payload, state)
    ids = [state.tool_calls[i]["id"] for i in sort!(collect(keys(state.tool_calls)))]
    @test length(ids) == 2 && allunique(ids) && all(id -> startswith(id, "unilm_call_"), ids)
end

@testset "encode — response_format → generationConfig.responseFormat.text (wire golden)" begin
    schema = Dict("type" => "object",
                  "properties" => Dict("city" => Dict("type" => "string"), "country" => Dict("type" => "string")),
                  "required" => ["city", "country"], "additionalProperties" => false)
    chat = Chat(service=GEMINIServiceEndpoint, model="gemini-3.8-flash", reasoning_effort="low", max_tokens=1024,
                response_format=UniLM.json_schema("capital", "A capital city", schema; strict=true),
                messages=[Message(role=RoleUser, content="Give the capital of Norway as JSON.")])
    # TextResponseFormat.mimeType is an enum on this wire: "application/json" is a 400.
    @test JSON.parse(encode_request(GEMINIServiceEndpoint, chat)) == Dict(
        "contents" => [Dict("role" => "user", "parts" => [Dict("text" => "Give the capital of Norway as JSON.")])],
        "generationConfig" => Dict("maxOutputTokens" => 1024, "thinkingConfig" => Dict("thinkingLevel" => "LOW"),
            "responseFormat" => Dict("text" => Dict("mimeType" => "APPLICATION_JSON", "schema" => schema))))

    gen(rf) = get(JSON.parse(encode_request(GEMINIServiceEndpoint,
        Chat(service=GEMINIServiceEndpoint, response_format=rf))), "generationConfig", nothing)
    # an OpenAI-shaped Dict json_schema carries its schema the same way
    @test gen(ResponseFormat(Dict("name" => "capital", "schema" => schema, "strict" => true))) ==
          Dict("responseFormat" => Dict("text" => Dict("mimeType" => "APPLICATION_JSON", "schema" => schema)))
    # ... and so does a Symbol-keyed one
    @test gen(ResponseFormat(Dict(:name => "capital", :schema => schema))) ==
          Dict("responseFormat" => Dict("text" => Dict("mimeType" => "APPLICATION_JSON", "schema" => schema)))
    @test gen(UniLM.json_object()) == Dict("responseFormat" => Dict("text" => Dict("mimeType" => "APPLICATION_JSON")))
    @test isnothing(gen(ResponseFormat(type="text")))            # plain text: no format, no generationConfig
    # every other shape fails loud instead of vanishing from the request
    for rf in (ResponseFormat(type="xml"), ResponseFormat(type="json_schema"),
               ResponseFormat(Dict("name" => "no_schema")), ResponseFormat("json_object", Dict("schema" => schema)),
               ResponseFormat("text", Dict("schema" => schema)))
        @test_throws ArgumentError encode_request(GEMINIServiceEndpoint, Chat(service=GEMINIServiceEndpoint, response_format=rf))
    end
end

@testset "encode — safety_identifier → top-level labels (wire golden)" begin
    chat = Chat(service=GEMINIServiceEndpoint, model="gemini-3.8-flash", safety_identifier="user-42",
                messages=[Message(role=RoleUser, content="hi")])
    @test JSON.parse(encode_request(GEMINIServiceEndpoint, chat)) == Dict(
        "contents" => [Dict("role" => "user", "parts" => [Dict("text" => "hi")])],
        "labels" => Dict("safety_identifier" => "user-42"))
    @test !haskey(JSON.parse(encode_request(GEMINIServiceEndpoint, Chat(service=GEMINIServiceEndpoint))), "labels")
    # Label values: at most 63 characters, only lowercase letters, numeric characters,
    # underscores, and dashes; international characters are allowed.
    label(id) = JSON.parse(encode_request(GEMINIServiceEndpoint,
        Chat(service=GEMINIServiceEndpoint, safety_identifier=id)))["labels"]["safety_identifier"]
    for ok in ("user_session_123", repeat("a", 63), "ünïcode-7", "用户_7")
        @test label(ok) == ok
    end
    hexhash = repeat("0123456789abcdef", 4)          # 64 characters, like a SHA-256 hex digest
    for bad in ("User@Example.com", "UPPER", "with space", "trailing\n", "a.b", hexhash, repeat("é", 64))
        err = try label(bad); nothing catch e; e end
        @test err isa ArgumentError && occursin("at most 63 characters", err.msg)
    end
    err = try label(hexhash); nothing catch e; e end
    @test occursin("(got 64 characters)", err.msg) && occursin("one character too long for a Gemini label", err.msg)
end

@testset "encode — response_format and safety_identifier are mapped; other options still fail closed" begin
    @test :response_format ∉ UniLM._GEMINI_CHAT_UNMAPPED_FIELDS
    @test :safety_identifier ∉ UniLM._GEMINI_CHAT_UNMAPPED_FIELDS
    err = try encode_request(GEMINIServiceEndpoint, Chat(service=GEMINIServiceEndpoint, seed=7)); nothing catch e; e end
    @test err isa ArgumentError && occursin("seed", err.msg)
end

@testset "thinking levels — one row per model family of the provider table" begin
    # (family, accepted level, rejected level, error fragment), transcribed from the
    # "Controlling thinking" table (ai.google.dev/gemini-api/docs/thinking, 2026-09-22).
    # A row allowing all four levels can only reject a non-level.
    three, four = "supports low, medium, or high thinking effort", "must be minimal, low, medium, or high"
    rows = (("gemini-3.8-flash",            "medium",  "minimal", three),
            ("gemini-3.7-flash",            "high",    "minimal", three),
            ("gemini-3.6-flash",            "minimal", "xhigh",   four),
            ("gemini-3.5-flash-lite",       "minimal", "none",    four),
            ("gemini-3.1-pro-preview",      "medium",  "minimal", three),
            ("gemini-3.1-flash-lite-image", "minimal", "low",     "supports minimal or high thinking effort"),
            ("gemini-3-flash-preview",      "minimal", "xhigh",   four),
            ("gemini-3-pro-preview",        "high",    "medium",  "supports low or high thinking effort"),
            ("gemini-3.5-flash",            "minimal", "xhigh",   four),
            ("gemini-2.5-pro",              "low",     "minimal", three),
            ("gemini-2.5-flash",            "medium",  "minimal", three),
            ("gemini-2.5-flash-lite",       "high",    "minimal", three))
    @test Set(first.(rows)) == Set(first.(UniLM._GEMINI_THINKING_LEVELS))   # every table row is covered
    level(model, effort) = JSON.parse(UniLM.encode_agentic(GEMINIServiceEndpoint, Respond(
        service=GEMINIServiceEndpoint, model=model, input="x", reasoning=Reasoning(effort=effort))))["generation_config"]["thinking_level"]
    native(model, effort) = JSON.parse(encode_request(GEMINIServiceEndpoint, Chat(
        service=GEMINIServiceEndpoint, model=model, reasoning_effort=effort)))["generationConfig"]["thinkingConfig"]["thinkingLevel"]
    @testset "$family" for (family, ok, bad, fragment) in rows
        @test level(family, ok) == ok
        if startswith(family, "gemini-2.5-")   # generateContent on 2.5 takes a thinking budget, never a level
            @test_throws ArgumentError native(family, ok)
        else
            @test native(family, ok) == uppercase(ok)
        end
        err = try level(family, bad); nothing catch e; e end
        @test err isa ArgumentError && occursin(fragment, err.msg)
        @test_throws ArgumentError native(family, bad)
    end
end

@testset "thinking levels — the longest matching family wins; unlisted families keep the generic check" begin
    table = (("fam-a", ("low",)), ("fam-a-b", ("high",)))
    @test UniLM._gemini_thinking_levels("fam-a-b-2", table) == ("high",)          # both match
    @test UniLM._gemini_thinking_levels("fam-a-b-2", reverse(table)) == ("high",) # order-independent
    @test UniLM._gemini_thinking_levels("fam-a-2", table) == ("low",)
    @test isnothing(UniLM._gemini_thinking_levels("fam-c", table))
    @test UniLM._gemini_thinking_levels("gemini-3.5-flash-lite-preview") == ("minimal", "low", "medium", "high")
    @test isnothing(UniLM._gemini_thinking_levels("gemini-flash-latest"))
    @test UniLM._gemini_thinking_level("gemini-flash-latest", "minimal") == "minimal"
    @test_throws ArgumentError UniLM._gemini_thinking_level("gemini-flash-latest", "xhigh")
end

@testset "pricing — Gemini 3.6 Flash, 3.5 Flash-Lite, and Embedding 2 rows" begin
    # Google pricing page, 2026-09-22: standard paid tier, USD per 1M tokens.
    usage = TokenUsage(prompt_tokens=1000, completion_tokens=500, cached_tokens=200)
    for (model, input, cached, output) in (("gemini-3.6-flash", 0.75, 0.075, 3.75),
                                           ("gemini-3.5-flash-lite", 0.30, 0.03, 2.50))
        r = LLMSuccess(message=Message(role=RoleAssistant, content="x"),
                       self=Chat(service=GEMINIServiceEndpoint, model=model), usage=usage)
        @test estimated_cost(r) ≈ (800 * input + 200 * cached + 500 * output) / 1_000_000
    end
    emb = Embeddings("hello"; service=GEMINIOpenAIServiceEndpoint, model="gemini-embedding-2")
    es = EmbeddingSuccess(embeddings=emb, usage=TokenUsage(prompt_tokens=500_000, completion_tokens=7),
                          raw=Dict{String,Any}())
    @test estimated_cost(es) ≈ 500_000 * 0.20 / 1_000_000            # input only; output rate is zero
end
