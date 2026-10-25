# Native Gemini translation — deterministic, zero-spend unit tests.
using UniLM
using UniLM: encode_request, decode_response, StreamState,
             _build_stream_message, GEMINIServiceEndpoint, GPTFunction, GPTToolChoice,
             GEMINI_NATIVE_BASE, RoleSystem, RoleUser, RoleAssistant, RoleTool,
             TOOL_CALLS, STOP, CONTENT_FILTER
using Test, HTTP, JSON

@testset "routing — model in URL, stream branches on method" begin
    chat = Chat(service=GEMINIServiceEndpoint, model="gemini-3.8-flash")
    @test UniLM.get_url(chat) == "$(GEMINI_NATIVE_BASE)/models/gemini-3.8-flash:generateContent"
    schat = Chat(service=GEMINIServiceEndpoint, model="gemini-3.8-flash", stream=true)
    @test UniLM.get_url(schat) == "$(GEMINI_NATIVE_BASE)/models/gemini-3.8-flash:streamGenerateContent?alt=sse"
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
    @test UniLM.provider_capabilities(GEMINIServiceEndpoint) ==
          Set([:chat, :tools, :streaming, :agentic, :json_output, :images, :image_edits, :models])
    @test has_capability(GEMINIServiceEndpoint, :json_output)
    @test UniLM.default_model(GEMINIServiceEndpoint) == "gemini-3.8-flash"
    @test_throws ArgumentError UniLM._api_base_url(GEMINIServiceEndpoint)
end

@testset "encode — system → systemInstruction, user turn" begin
    chat = Chat(service=GEMINIServiceEndpoint, model="gemini-3.8-flash")
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
    # gemini-2.5-flash honours temperature, which later models ignore (sampling testset below).
    chat = Chat(service=GEMINIServiceEndpoint, model="gemini-2.5-flash",
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
    chat = Chat(service=GEMINIServiceEndpoint, model="gemini-3.8-flash",
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

@testset "encode — a strict tool asks for VALIDATED; a forced or disabled call keeps its mode" begin
    # Gemini has no per-tool strict flag. Its function-calling modes: VALIDATED "ensures function
    # schema adherence" while the model may answer in text; ANY "is constrained to always predict a
    # function call and ensures function schema adherence"
    # (https://ai.google.dev/gemini-api/docs/generate-content/function-calling).
    params = Dict("type" => "object", "properties" => Dict("city" => Dict("type" => "string")),
                  "required" => ["city"], "additionalProperties" => false)
    strict = Tool(func=FunctionSignature(name="get_weather", parameters=params, strict=true))
    loose = Tool(func=FunctionSignature(name="get_time", parameters=Dict("type" => "object")))
    function config(tools, tc=nothing)
        chat = Chat(service=GEMINIServiceEndpoint, model="gemini-3.8-flash", tools=tools, tool_choice=tc,
                    messages=[Message(Val(:system), "s"), Message(Val(:user), "u")])
        body = JSON.parse(encode_request(GEMINIServiceEndpoint, chat))
        haskey(body, "toolConfig") ? body["toolConfig"]["functionCallingConfig"] : nothing
    end
    @test config([strict]) == Dict("mode" => "VALIDATED")
    @test config([loose, strict], "auto") == Dict("mode" => "VALIDATED")     # one strict tool is enough
    @test config([strict], "required") == Dict("mode" => "ANY")
    @test config([strict], GPTToolChoice(func="get_weather")) ==
          Dict("mode" => "ANY", "allowedFunctionNames" => ["get_weather"])
    @test config([strict], "none") == Dict("mode" => "NONE")
    # Without a strict tool the request is as before.
    @test isnothing(config([loose])) && config([loose], "auto") == Dict("mode" => "AUTO")
    @test isnothing(config([Tool(func=FunctionSignature(name="f", strict=false))]))
end

@testset "encode — multi-turn: functionCall echo (+thoughtSignature) & functionResponse correlation" begin
    chat = Chat(service=GEMINIServiceEndpoint, model="gemini-3.8-flash")
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
    chat = Chat(service=GEMINIServiceEndpoint, model="gemini-3.8-flash", messages=msgs)
    @test_throws ArgumentError encode_request(GEMINIServiceEndpoint, chat)
end

@testset "encode — functionResponse passes a JSON-object tool result through unwrapped" begin
    chat = Chat(service=GEMINIServiceEndpoint, model="gemini-3.8-flash")
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
    chat = Chat(service=GEMINIServiceEndpoint, model="gemini-3.8-flash")
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
    chat = Chat(service=GEMINIServiceEndpoint, model="gemini-3.8-flash")
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

@testset "encode — response_format, safety_identifier, seed, store and service_tier are mapped; other options still fail closed" begin
    for f in (:response_format, :safety_identifier, :seed, :store, :service_tier)
        @test f ∉ UniLM._GEMINI_CHAT_UNMAPPED_FIELDS
    end
    err = try encode_request(GEMINIServiceEndpoint, Chat(service=GEMINIServiceEndpoint, presence_penalty=0.5)); nothing catch e; e end
    @test err isa ArgumentError && occursin("presence_penalty", err.msg)
end

@testset "thinking levels — one row per model of the provider table" begin
    # (model, accepted level, rejected level, error fragment), transcribed from the
    # "Controlling thinking" table (ai.google.dev/gemini-api/docs/thinking, 2026-10-22). The
    # -latest aliases take the levels of the model that served each on 2026-10-22, and
    # gemini-3.7-flash and gemini-3.5-flash those of the models Google routes them to.
    # A row allowing all four levels can only reject a non-level.
    three, four = "supports low, medium, or high thinking effort", "must be minimal, low, medium, or high"
    rows = (("gemini-4-argon",               "low",     "minimal", three),
            ("gemini-3.8-flash",             "medium",  "minimal", three),
            ("gemini-3.6-flash",             "minimal", "xhigh",   four),
            ("gemini-3.5-flash-lite",        "minimal", "none",    four),
            ("gemini-3.1-pro-preview",       "medium",  "minimal", three),
            ("gemini-3.1-flash-lite-image",  "minimal", "low",     "supports minimal or high thinking effort"),
            ("gemini-3-flash-preview",       "minimal", "xhigh",   four),
            ("gemini-2.5-pro",               "low",     "minimal", three),
            ("gemini-2.5-flash",             "medium",  "minimal", three),
            ("gemini-2.5-flash-lite",        "high",    "minimal", three),
            ("gemini-robotics-er-2-preview", "minimal", "xhigh",   four),
            ("gemini-pro-latest",            "high",    "minimal", three),   # served by gemini-3.1-pro-preview
            ("gemini-flash-latest",          "low",     "minimal", three),   # served by gemini-3.8-flash
            ("gemini-flash-lite-latest",     "minimal", "xhigh",   four),    # served by gemini-3.5-flash-lite
            ("gemini-3.7-flash",             "high",    "minimal", three),   # routed to gemini-3.8-flash
            ("gemini-3.5-flash",             "minimal", "xhigh",   four))    # routed to gemini-3.6-flash
    @test Set(first.(rows)) == Set(keys(UniLM._GEMINI_THINKING_LEVELS))   # every table row is covered
    @test !haskey(UniLM._GEMINI_THINKING_LEVELS, "gemini-3-pro-preview")  # shut down on 2026-03-09
    level(model, effort) = JSON.parse(UniLM.encode_agentic(GEMINIServiceEndpoint, Respond(
        service=GEMINIServiceEndpoint, model=model, input="x", reasoning=Reasoning(effort=effort))))["generation_config"]["thinking_level"]
    native(model, effort) = JSON.parse(encode_request(GEMINIServiceEndpoint, Chat(
        service=GEMINIServiceEndpoint, model=model, reasoning_effort=effort)))["generationConfig"]["thinkingConfig"]["thinkingLevel"]
    @testset "$model" for (model, ok, bad, fragment) in rows
        @test level(model, ok) == ok
        if startswith(model, "gemini-2.5-")   # generateContent on 2.5 takes a thinking budget, never a level
            @test_throws ArgumentError native(model, ok)
        else
            @test native(model, ok) == uppercase(ok)
        end
        err = try level(model, bad); nothing catch e; e end
        @test err isa ArgumentError && occursin(fragment, err.msg)
        @test_throws ArgumentError native(model, bad)
    end
    # Gemini 4 Argon and the aliases answer MINIMAL with HTTP 400 "Thinking level MINIMAL is not
    # supported for this model" (observed 2026-10-22): the request is refused before it is sent.
    for model in ("gemini-4-argon", "gemini-pro-latest", "gemini-flash-latest")
        for encode in (level, native)
            err = try encode(model, "minimal"); nothing catch e; e end
            @test err isa ArgumentError && occursin(model, err.msg) && occursin(three, err.msg)
        end
    end
end

@testset "thinking levels — a row matches its exact id; any other id gets the generic check" begin
    # Distinct models share prefixes (TTS and Live models beside the chat model) and Google's ids
    # carry no dated snapshots, so no row extends to a longer, suffixed or unknown id.
    for id in ("gemini-3.8-flash-tts", "gemini-3.8-flash-lite-tts", "gemini-3.8-live", "gemini-4-argon-2",
               "gemini-3.8-flash-2026-10-01", "gemini-3.5-flash-lite-preview", "gemini-5-x")
        @test isnothing(UniLM._gemini_thinking_levels(id))
        @test UniLM._gemini_thinking_level(id, "minimal") == "minimal"      # the server's answer is the check
        @test_throws ArgumentError UniLM._gemini_thinking_level(id, "xhigh") # no Gemini level at all
    end
    @test UniLM._gemini_thinking_levels("gemini-4-argon") == ["low", "medium", "high"]
    @test (@inferred UniLM._gemini_thinking_level("gemini-4-argon", "high")) == "high"
end

@testset "sampling — refused on every model but the earlier ones documented to honour it" begin
    # Gemini ignores temperature, top_p and top_k on Gemini 3.6 Flash, 3.5 Flash-Lite and every
    # later model (ai.google.dev/gemini-api/docs/whats-new-gemini-3.6); gemini-4-argon and
    # gemini-3.8-flash answered such requests with HTTP 200 on 2026-10-22. Refused, so a request
    # never goes out quietly ignoring them.
    native(model; kw...) = JSON.parse(encode_request(GEMINIServiceEndpoint,
        Chat(; service=GEMINIServiceEndpoint, model, kw...)))
    agentic(model; kw...) = JSON.parse(UniLM.encode_agentic(GEMINIServiceEndpoint,
        Respond(; service=GEMINIServiceEndpoint, model, input="x", kw...)))
    for model in ("gemini-4-argon", "gemini-3.8-flash", "gemini-3.6-flash", "gemini-3.5-flash-lite", "gemini-5-x",
                  "gemini-3.7-flash", "gemini-3.5-flash",       # routed to 3.8 and 3.6 Flash
                  "gemini-flash-latest", "gemini-pro-latest",   # an alias moves to newer models
                  "gemini-2.5-flash-exp"),                      # a near miss of an allow-listed id
        (field, kw) in (("temperature", (temperature=0.5,)), ("top_p", (top_p=0.9,)))
        for encode in (native, agentic)
            err = try encode(model; kw...); nothing catch e; e end
            @test err isa ArgumentError && occursin("Gemini ignores $field", err.msg) && occursin(model, err.msg)
        end
    end
    # The earlier text and image models, the 2.5 family among them, take them.
    @test Set(["gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-3.1-flash-lite-image",
               "gemini-3-pro-image", "gemini-3.1-flash-image", "gemini-2.5-flash-image"]) ⊆ Set(UniLM._GEMINI_PRE_3_6_MODELS)
    for model in UniLM._GEMINI_PRE_3_6_MODELS
        @test native(model; temperature=0.5)["generationConfig"] == Dict("temperature" => 0.5)
        @test native(model; top_p=0.9)["generationConfig"] == Dict("topP" => 0.9)
        @test agentic(model; temperature=0.5)["generation_config"] == Dict("temperature" => 0.5)
        @test agentic(model; top_p=0.9)["generation_config"] == Dict("top_p" => 0.9)
    end
    # Refused before any request: the encoder runs before any I/O, and the verb lets its sampling
    # refusal through.
    err = try chatrequest!(Chat(service=GEMINIServiceEndpoint, model="gemini-4-argon", temperature=0.5,
                                messages=[Message(role=RoleUser, content="hi")])); nothing catch e; e end
    @test err isa ArgumentError && occursin("Gemini ignores temperature", err.msg) && occursin("gemini-4-argon", err.msg)
end

@testset "a request ending with a model turn is refused where Gemini answers it with HTTP 400" begin
    # From Gemini 3.6 Flash and 3.5 Flash-Lite on, a request whose last non-empty turn is the
    # model's is a 400 "Requests ending with a model turn are not supported." (observed on
    # gemini-3.8-flash, 2026-10-22; ai.google.dev/gemini-api/docs/whats-new-gemini-3.6).
    user = Message(role=RoleUser, content="Translate 'Hello world' to Spanish.")
    prefill = Message(role=RoleAssistant, content="Translation:")
    contents(model, msgs) = JSON.parse(encode_request(GEMINIServiceEndpoint,
        Chat(; service=GEMINIServiceEndpoint, model, messages=msgs)))["contents"]
    for model in ("gemini-4-argon", "gemini-3.8-flash", "gemini-3.6-flash", "gemini-3.5-flash-lite",
                  "gemini-3.7-flash", "gemini-flash-latest", "gemini-5-x")
        err = try contents(model, [user, prefill]); nothing catch e; e end
        @test err isa ArgumentError && occursin(model, err.msg) && occursin("ends with a model turn", err.msg)
    end
    # The earlier models still continue a prefilled turn.
    for model in UniLM._GEMINI_PRE_3_6_MODELS
        @test last(contents(model, [user, prefill])) == Dict("role" => "model", "parts" => [Dict("text" => "Translation:")])
    end
    # A model turn with nothing to send is left out, so that request still ends with the user's turn,
    # and tool results answering the model's call are the user's turn too.
    refused = Message(role=RoleAssistant, refusal_message="blocked", finish_reason=CONTENT_FILTER)
    @test contents("gemini-3.8-flash", [user, refused]) == [Dict("role" => "user", "parts" => [Dict("text" => user.content)])]
    call = ToolCall(id="fc_1", func=GPTFunction("lookup", Dict{String,Any}("q" => "x")))
    turns = contents("gemini-3.8-flash", [user, Message(role=RoleAssistant, tool_calls=[call], finish_reason=TOOL_CALLS),
                                          Message(role=RoleTool, tool_call_id="fc_1", content="42")])
    @test [t["role"] for t in turns] == ["user", "model", "user"]
end

@testset "prompt_cache_options — Gemini caches prompts implicitly; its explicit caching is not wrapped" begin
    # ai.google.dev/gemini-api/docs/caching: implicit caching is on for every Gemini 2.5 and newer
    # model with nothing to configure; explicit caching (cached contents) is a generateContent API
    # of its own on Gemini 3 and 2.5 models, which UniLM does not wrap.
    for pco in (PromptCacheOptions(ttl="5m"), PromptCacheOptions(ttl="1h"), PromptCacheOptions(ttl="30m"),
                PromptCacheOptions(ttl="12h"), PromptCacheOptions(mode="implicit"))
        for f in (() -> encode_request(GEMINIServiceEndpoint, Chat(service=GEMINIServiceEndpoint, prompt_cache_options=pco)),
                  () -> UniLM.encode_agentic(GEMINIServiceEndpoint,
                                             Respond(service=GEMINIServiceEndpoint, input="x", prompt_cache_options=pco)))
            err = try f(); nothing catch e; e end
            @test err isa ArgumentError && occursin("prompt_cache_options", err.msg) && occursin("implicitly", err.msg) &&
                  occursin("does not wrap its explicit caching", err.msg)
        end
    end
end

@testset "decode and stream keep the served model and Gemini's own refusal text" begin
    decode(body) = decode_response(GEMINIServiceEndpoint, HTTP.Response(200, [], Vector{UInt8}(JSON.json(body))))
    # modelVersion names the model that answered; a routed id answers as its target (observed
    # 2026-10-22: a gemini-3.7-flash request reports gemini-3.8-flash).
    served = Dict("candidates" => [Dict("content" => Dict("role" => "model", "parts" => [Dict("text" => "OK")]),
                                        "finishReason" => "STOP")],
                  "usageMetadata" => Dict("promptTokenCount" => 8, "candidatesTokenCount" => 1, "totalTokenCount" => 9),
                  "modelVersion" => "gemini-3.8-flash", "responseId" => "r_1")
    @test decode(served).model == "gemini-3.8-flash"
    @test isnothing(decode(delete!(copy(served), "modelVersion")).model)
    state = StreamState()
    UniLM.handle_sse_event!(GEMINIServiceEndpoint, "", JSON.json(served), state)
    @test state.model == "gemini-3.8-flash"
    # A filtered turn's finishMessage, Google's account of the block, is its refusal, streamed or not.
    why = "The response was blocked because it may contain prohibited content."
    blocked = Dict("candidates" => [Dict("finishReason" => "PROHIBITED_CONTENT", "finishMessage" => why)],
                   "modelVersion" => "gemini-3.8-flash")
    m = decode(blocked).message
    @test m.refusal_message == why && m.finish_reason == CONTENT_FILTER
    state = StreamState()
    UniLM.handle_sse_event!(GEMINIServiceEndpoint, "", JSON.json(blocked), state)
    @test _build_stream_message(state).refusal_message == why
    # Without one, the refusal names the filter.
    @test decode(Dict("candidates" => [Dict("finishReason" => "SAFETY")])).message.refusal_message ==
          "Model response blocked by safety filter."
end

@testset "usage — tool-use prompt tokens are input, billed at the input rate" begin
    # totalTokenCount is prompt + tool-use prompt + thoughts + candidates (UsageMetadata,
    # ai.google.dev/api/generate-content): the tool-use prompt is input the model read.
    u = UniLM._gemini_usage(Dict("promptTokenCount" => 100, "toolUsePromptTokenCount" => 40,
        "cachedContentTokenCount" => 20, "candidatesTokenCount" => 10, "thoughtsTokenCount" => 5,
        "totalTokenCount" => 155))
    @test u == TokenUsage(prompt_tokens=140, completion_tokens=15, total_tokens=155, cached_tokens=20, reasoning_tokens=5)
    r = LLMSuccess(message=Message(role=RoleAssistant, content="x"),
                   self=Chat(service=GEMINIServiceEndpoint, model="gemini-3.8-flash"), usage=u)
    @test estimated_cost(r) ≈ (120 * 0.75 + 20 * 0.075 + 15 * 3.75) / 1_000_000
end

@testset "the served model prices the call; a routed id carries its target's rates" begin
    reply(model) = decode_response(GEMINIServiceEndpoint, HTTP.Response(200, [], Vector{UInt8}(JSON.json(Dict(
        "candidates" => [Dict("content" => Dict("role" => "model", "parts" => [Dict("text" => "OK")]),
                              "finishReason" => "STOP")],
        "usageMetadata" => Dict("promptTokenCount" => 1_000, "candidatesTokenCount" => 100, "totalTokenCount" => 1_100),
        "modelVersion" => model)))))
    result(asked, d) = LLMSuccess(message=d.message, self=Chat(service=GEMINIServiceEndpoint, model=asked),
                                  usage=d.usage, model=d.model)
    # An alias has no row of its own and is billed as the model that answered (observed
    # 2026-10-22: gemini-flash-lite-latest is served by gemini-3.5-flash-lite).
    lite = result("gemini-flash-lite-latest", reply("gemini-3.5-flash-lite"))
    @test lite.model == "gemini-3.5-flash-lite"
    @test estimated_cost(lite) ≈ (1_000 * 0.30 + 100 * 2.50) / 1_000_000
    @test (@test_logs (:warn, r"no price row") estimated_cost(lite; model="gemini-flash-lite-latest")) == 0.0
    # gemini-pro-latest was served by gemini-3.1-pro-preview on 2026-10-22: billed at that model's row.
    pro = result("gemini-pro-latest", reply("gemini-3.1-pro-preview"))
    @test estimated_cost(pro) ≈ (1_000 * 2.0 + 100 * 12.0) / 1_000_000
    # gemini-3.7-flash and gemini-3.5-flash answer as gemini-3.8-flash and gemini-3.6-flash, and are
    # billed at those rates whether or not the reply names the served model.
    for (asked, target) in (("gemini-3.7-flash", "gemini-3.8-flash"), ("gemini-3.5-flash", "gemini-3.6-flash"))
        d = reply(target)
        @test estimated_cost(result(asked, d)) == estimated_cost(result(target, d)) ≈ (1_000 * 0.75 + 100 * 3.75) / 1e6
        unreported = LLMSuccess(message=d.message, self=Chat(service=GEMINIServiceEndpoint, model=asked), usage=d.usage)
        @test estimated_cost(unreported) == estimated_cost(result(target, d))
    end
end

@testset "a reply streamed and not streamed reports the same usage and served model" begin
    # So cumulative_cost does not depend on `stream`.
    usage = Dict("promptTokenCount" => 120, "toolUsePromptTokenCount" => 30, "cachedContentTokenCount" => 64,
                 "candidatesTokenCount" => 12, "thoughtsTokenCount" => 40, "totalTokenCount" => 202)
    whole = Dict("candidates" => [Dict("content" => Dict("role" => "model", "parts" => [Dict("text" => "Hello world")]),
                                       "finishReason" => "STOP")],
                 "usageMetadata" => usage, "modelVersion" => "gemini-3.8-flash")
    d = decode_response(GEMINIServiceEndpoint, HTTP.Response(200, [], Vector{UInt8}(JSON.json(whole))))
    # The same reply streamed: the text in two chunks, the finish reason and the final usage on the
    # last, every chunk naming the model.
    chunks = (Dict("candidates" => [Dict("content" => Dict("role" => "model", "parts" => [Dict("text" => "Hello")]))],
                   "usageMetadata" => Dict("promptTokenCount" => 120, "totalTokenCount" => 120),
                   "modelVersion" => "gemini-3.8-flash"),
              Dict("candidates" => [Dict("content" => Dict("role" => "model", "parts" => [Dict("text" => " world")]),
                                         "finishReason" => "STOP")],
                   "usageMetadata" => usage, "modelVersion" => "gemini-3.8-flash"))
    state = StreamState()
    for c in chunks
        UniLM.handle_sse_event!(GEMINIServiceEndpoint, "", JSON.json(c), state)
    end
    streamed = _build_stream_message(state)
    @test streamed.content == d.message.content == "Hello world"
    @test state.usage == d.usage == TokenUsage(prompt_tokens=150, completion_tokens=52, total_tokens=202,
                                               cached_tokens=64, reasoning_tokens=40)
    @test state.model == d.model == "gemini-3.8-flash"
    chat = Chat(service=GEMINIServiceEndpoint, model="gemini-3.7-flash")
    cost(m, u, model) = estimated_cost(LLMSuccess(message=m, self=chat, usage=u, model=model))
    @test cost(streamed, state.usage, state.model) == cost(d.message, d.usage, d.model) > 0
end

@testset "pricing — Gemini rows (Google pricing page, 2026-10-22)" begin
    # Standard paid tier, USD per 1M tokens; output includes thinking.
    usage = TokenUsage(prompt_tokens=1000, completion_tokens=500, cached_tokens=200)
    for (model, input, cached, output) in (("gemini-4-argon", 2.0, 0.10, 10.0), ("gemini-3.8-flash", 0.75, 0.075, 3.75),
                                           ("gemini-3.6-flash", 0.75, 0.075, 3.75),
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

# ─── Media in a user message ─────────────────────────────────────────────────

const _GEMINI_FIXTURES = joinpath(@__DIR__, "fixtures")
_gemini_b64(bytes) = UniLM.Base64.base64encode(bytes)
_gemini_caught(f) = try f(); nothing catch e; e end

@testset "encode — attachments are inlineData parts, ahead of the text" begin
    img = ImageAttachment(joinpath(_GEMINI_FIXTURES, "ollama", "red_circle.png"))
    wav = AudioAttachment(joinpath(_GEMINI_FIXTURES, "ollama", "secret_word.wav"))
    pdf = DocumentAttachment(joinpath(_GEMINI_FIXTURES, "documents", "secret_word.pdf"))
    chat = Chat(service=GEMINIServiceEndpoint, model="gemini-3.8-flash",
                messages=[Message(Val(:system), "s"), Message(Val(:user), "What is in these?", img, wav, pdf)])
    # Google: "place the text prompt after the image part in the contents array".
    @test JSON.parse(encode_request(GEMINIServiceEndpoint, chat)) == Dict(
        "systemInstruction" => Dict("parts" => [Dict("text" => "s")]),
        "contents" => [Dict("role" => "user", "parts" => [
            Dict("inlineData" => Dict("mimeType" => "image/png", "data" => _gemini_b64(img.data))),
            Dict("inlineData" => Dict("mimeType" => "audio/wav", "data" => _gemini_b64(wav.data))),
            Dict("inlineData" => Dict("mimeType" => "application/pdf", "data" => _gemini_b64(pdf.data))),
            Dict("text" => "What is in these?")])])
    # An MP3 clip is audio/mp3; a message of media alone (no text, or empty text) sends no text part.
    mp3 = AudioAttachment(Vector{UInt8}(b"ID3\x04\x00\x00"))
    for m in (Message(role=RoleUser, attachments=Attachment[mp3]), Message(role=RoleUser, content="", attachments=Attachment[mp3]))
        body = JSON.parse(encode_request(GEMINIServiceEndpoint, Chat(service=GEMINIServiceEndpoint, messages=[m])))
        @test body["contents"] == [Dict("role" => "user", "parts" => [
            Dict("inlineData" => Dict("mimeType" => "audio/mp3", "data" => _gemini_b64(mp3.data)))])]
    end
    # The image-edit request builds its input images with the same helper.
    @test UniLM._gemini_inline_data(img) ==
          Dict(:inlineData => Dict(:mimeType => "image/png", :data => _gemini_b64(img.data)))
    @test (@inferred UniLM._gemini_inline_data(pdf)) isa Dict{Symbol,Any}
end

@testset "encode — an image's detail level is its per-part media resolution" begin
    png = read(joinpath(_GEMINI_FIXTURES, "ollama", "red_circle.png"))
    part(detail; model="gemini-3.8-flash") = JSON.parse(encode_request(GEMINIServiceEndpoint, Chat(
        service=GEMINIServiceEndpoint, model=model,
        messages=[Message(Val(:user), "x", ImageAttachment(png; detail))])))["contents"][1]["parts"][1]
    # LOW took a 1089-token image to 256 tokens on gemini-3.8-flash (observed 2026-10-22).
    @test part("low")["mediaResolution"] == Dict("level" => "MEDIA_RESOLUTION_LOW")
    @test part("high")["mediaResolution"] == Dict("level" => "MEDIA_RESOLUTION_HIGH")
    # "auto" is the model's default, which no level expresses better than leaving it out.
    @test !haskey(part("auto"), "mediaResolution") && !haskey(part(nothing), "mediaResolution")
    e = _gemini_caught(() -> part("original"))
    @test e isa ArgumentError && occursin("\"original\"", e.msg) && occursin("\"high\"", e.msg)
    # Google documents a per-part resolution for Gemini 4 and Gemini 3 models only: the earlier
    # ones refuse a level.
    for model in ("gemini-2.5-flash", "gemini-2.5-pro", "gemma-4-31b-it")
        e = _gemini_caught(() -> part("low"; model))
        @test e isa ArgumentError && occursin(model, e.msg) && occursin("detail", e.msg)
        @test !haskey(part("auto"; model), "mediaResolution")
    end
    # Any other id is left to the server.
    @test part("high"; model="gemini-4-argon")["mediaResolution"] == Dict("level" => "MEDIA_RESOLUTION_HIGH")
end

# ─── Service tier, seed, store ───────────────────────────────────────────────

@testset "encode — service_tier is the request's serviceTier" begin
    body(tier) = JSON.parse(encode_request(GEMINIServiceEndpoint, Chat(service=GEMINIServiceEndpoint,
        service_tier=tier, messages=[Message(Val(:user), "hi")])))
    for tier in ("standard", "flex", "priority")
        @test body(tier) == Dict("contents" => [Dict("role" => "user", "parts" => [Dict("text" => "hi")])],
                                 "serviceTier" => tier)
    end
    @test !haskey(body(nothing), "serviceTier")
    e = _gemini_caught(() -> body("deferred"))
    @test e isa ArgumentError && occursin("Interactions", e.msg) && occursin("respond", e.msg)
    for bad in ("auto", "default", "fast", "scale", "Standard", "unspecified")
        e = _gemini_caught(() -> body(bad))
        @test e isa ArgumentError && occursin("\"standard\", \"flex\" or \"priority\"", e.msg) && occursin(repr(bad), e.msg)
    end
end

@testset "encode — seed is generationConfig.seed; store is the request's store" begin
    chat = Chat(service=GEMINIServiceEndpoint, seed=7, store=false, messages=[Message(Val(:user), "hi")])
    @test JSON.parse(encode_request(GEMINIServiceEndpoint, chat)) == Dict(
        "contents" => [Dict("role" => "user", "parts" => [Dict("text" => "hi")])],
        "generationConfig" => Dict("seed" => 7), "store" => false)
    @test JSON.parse(encode_request(GEMINIServiceEndpoint, Chat(service=GEMINIServiceEndpoint, store=true,
        messages=[Message(Val(:user), "hi")])))["store"] == true
end

@testset "decode — the billed tier: the x-gemini-service-tier header, else usageMetadata.serviceTier" begin
    reply(usage; headers=Pair{String,String}[]) = decode_response(GEMINIServiceEndpoint, HTTP.Response(200, headers,
        Vector{UInt8}(JSON.json(Dict("candidates" => [Dict("content" => Dict("role" => "model",
            "parts" => [Dict("text" => "OK")]), "finishReason" => "STOP")], "usageMetadata" => usage)))))
    u = Dict{String,Any}("promptTokenCount" => 8, "candidatesTokenCount" => 1, "totalTokenCount" => 9)
    @test reply(merge(u, Dict("serviceTier" => "priority"))).usage ==
          TokenUsage(prompt_tokens=8, completion_tokens=1, total_tokens=9, service_tier="priority")
    # Gemini 4 Argon names its tier in the header only (observed 2026-10-22).
    @test reply(u; headers=["x-gemini-service-tier" => "flex"]).usage.service_tier == "flex"
    # Google names the header as the sign of a downgrade, and a downgraded Priority request bills at
    # the standard rate (ai.google.dev/gemini-api/docs/priority-inference): the header wins.
    @test reply(merge(u, Dict("serviceTier" => "priority")); headers=["x-gemini-service-tier" => "standard"]).usage.service_tier == "standard"
    @test isnothing(reply(u).usage.service_tier)
    # A stream reads the header before its first chunk, so the streamed usage carries the same tier.
    for (head, body, want) in ((["x-gemini-service-tier" => "flex"], u, "flex"),
                               (["x-gemini-service-tier" => "standard"], merge(u, Dict("serviceTier" => "priority")), "standard"),
                               (Pair{String,String}[], merge(u, Dict("serviceTier" => "priority")), "priority"),
                               (Pair{String,String}[], u, nothing))
        state = StreamState()
        UniLM._stream_head!(GEMINIServiceEndpoint, state, HTTP.Response(200, head))
        UniLM.handle_sse_event!(GEMINIServiceEndpoint, "", JSON.json(Dict("candidates" => [Dict("content" =>
            Dict("role" => "model", "parts" => [Dict("text" => "OK")]), "finishReason" => "STOP")],
            "usageMetadata" => body)), state)
        @test state.usage.service_tier == want
        @test state.usage == reply(body; headers=head).usage
    end
    # Every other endpoint reads nothing from a stream's head.
    state = StreamState()
    UniLM._stream_head!(OPENAIServiceEndpoint, state, HTTP.Response(200, ["x-gemini-service-tier" => "flex"]))
    @test isnothing(state.service_tier)
end

@testset "compat — service_tier takes the values Google's compatibility page names; any other throws before a request" begin
    # https://ai.google.dev/gemini-api/docs/openai#flex-and-priority-inference: service_tier="priority"
    # or "flex", and "When not explicitly assigned, service_tier defaults to standard".
    chat(tier) = Chat(service=GEMINIOpenAIServiceEndpoint, model="gemini-3.8-flash", service_tier=tier,
                      messages=[Message(Val(:user), "hi")])
    body(tier) = JSON.parse(encode_request(GEMINIOpenAIServiceEndpoint, chat(tier)))
    for tier in ("standard", "flex", "priority")
        @test body(tier)["service_tier"] == tier
    end
    @test !haskey(body(nothing), "service_tier")
    for bad in ("auto", "default", "scale", "fast", "ultrafast", "deferred", "Priority")
        e = _gemini_caught(() -> body(bad))
        @test e isa ArgumentError && occursin("\"standard\", \"flex\" or \"priority\"", e.msg) && occursin(repr(bad), e.msg)
        # The call throws as its body is built, before any request is sent.
        @test _gemini_caught(() -> chatrequest!(chat(bad))) isa ArgumentError
    end
end

@testset "compat — the billed tier: the x-gemini-service-tier header, else the reply's service_tier, streamed or not" begin
    usage = Dict("prompt_tokens" => 8, "completion_tokens" => 1, "total_tokens" => 9)
    with_tier(d, tier) = isnothing(tier) ? d : merge(d, Dict("service_tier" => tier))
    reply(tier; headers=Pair{String,String}[]) = HTTP.Response(200, headers, Vector{UInt8}(JSON.json(with_tier(Dict(
        "model" => "gemini-3.8-flash", "usage" => usage, "choices" => [Dict("index" => 0, "finish_reason" => "stop",
        "message" => Dict("role" => "assistant", "content" => "OK"))]), tier))))
    decoded(tier; headers=Pair{String,String}[]) = decode_response(GEMINIOpenAIServiceEndpoint, reply(tier; headers))
    flex, standard = ["x-gemini-service-tier" => "flex"], ["x-gemini-service-tier" => "standard"]
    # Neither the body nor the head of a flex or a priority reply named a tier on 2026-10-22.
    for (tier, head, want) in ((nothing, flex, "flex"), ("priority", standard, "standard"), ("priority", Pair{String,String}[], "priority"),
                               (nothing, Pair{String,String}[], nothing))
        d = decoded(tier; headers=head)
        @test d.usage == TokenUsage(prompt_tokens=8, completion_tokens=1, total_tokens=9, service_tier=want)
        @test d.message.content == "OK" && d.model == "gemini-3.8-flash"
        # Streamed: the head is read before the first chunk, and outranks a tier a chunk names.
        state = StreamState()
        UniLM._stream_head!(GEMINIOpenAIServiceEndpoint, state, HTTP.Response(200, head))
        for chunk in (Dict("model" => "gemini-3.8-flash", "choices" => [Dict("index" => 0, "delta" => Dict("content" => "OK"),
                                                                              "finish_reason" => "stop")]),
                      Dict("model" => "gemini-3.8-flash", "choices" => [], "usage" => usage))
            UniLM.handle_sse_event!(GEMINIOpenAIServiceEndpoint, "", JSON.json(with_tier(chunk, tier)), state)
        end
        @test state.usage == d.usage
    end
    # The reported tier prices the call: Flex at half the standard rates.
    r = LLMSuccess(message=Message(role=RoleAssistant, content="OK"), self=Chat(service=GEMINIOpenAIServiceEndpoint,
                   model="gemini-3.8-flash"), usage=decoded(nothing; headers=flex).usage)
    @test estimated_cost(r) ≈ (8 * 0.375 + 1 * 1.875) / 1_000_000
end

# The compatible wire to a local server, with that endpoint's own encoding, decoding, tiers and prices.
const _COMPAT_URL = Ref("")
struct _CompatHere <: UniLM.OpenAIWireEndpoint end
UniLM.get_url(::Type{_CompatHere}, ::Chat) = _COMPAT_URL[]
UniLM.auth_header(::Type{_CompatHere}) = ["Content-Type" => "application/json"]
UniLM.encode_request(::Type{_CompatHere}, c::Chat) = encode_request(GEMINIOpenAIServiceEndpoint, c)
UniLM.decode_response(::Type{_CompatHere}, r::HTTP.Response) = decode_response(GEMINIOpenAIServiceEndpoint, r)
UniLM.handle_sse_event!(::Type{_CompatHere}, e::AbstractString, p::AbstractString, s::StreamState) =
    UniLM.handle_sse_event!(GEMINIOpenAIServiceEndpoint, e, p, s)
UniLM._stream_head!(::Type{_CompatHere}, s::StreamState, h::HTTP.Response) = UniLM._stream_head!(GEMINIOpenAIServiceEndpoint, s, h)
UniLM._result_usage(::Type{_CompatHere}, c::Chat, u::TokenUsage) = UniLM._result_usage(GEMINIOpenAIServiceEndpoint, c, u)
UniLM._billed_cost(::Type{_CompatHere}, m::String, u::TokenUsage, r::UniLM.PriceRow) =
    UniLM._billed_cost(GEMINIOpenAIServiceEndpoint, m, u, r)

# Replies as Google's compatible endpoint sent them on 2026-10-22: no tier in the head or the body.
function _with_compat(f)
    usage = Dict("prompt_tokens" => 1_000, "completion_tokens" => 500, "total_tokens" => 1_500)
    server = HTTP.serve!("127.0.0.1", 0; listenany=true, verbose=false) do req
        if get(JSON.parse(String(copy(req.body))), "stream", false) === true
            chunks = [Dict("model" => "gemini-3.8-flash", "choices" => [Dict("index" => 0, "delta" => Dict("content" => "OK"),
                                                                           "finish_reason" => "stop")]),
                      Dict("model" => "gemini-3.8-flash", "choices" => [], "usage" => usage)]
            HTTP.Response(200, ["Content-Type" => "text/event-stream"],
                          Vector{UInt8}(join("data: " * JSON.json(c) * "\n\n" for c in chunks) * "data: [DONE]\n\n"))
        else
            HTTP.Response(200, ["Content-Type" => "application/json"], Vector{UInt8}(JSON.json(Dict("model" => "gemini-3.8-flash",
                "usage" => usage, "choices" => [Dict("index" => 0, "finish_reason" => "stop",
                                                     "message" => Dict("role" => "assistant", "content" => "OK"))]))))
        end
    end
    _COMPAT_URL[] = "http://127.0.0.1:$(HTTP.port(server))"
    try
        f()
    finally
        close(server)
    end
end

@testset "compat — a result records the tier its request asked for, which the reply does not name" begin
    # gemini-3.8-flash, 1,000 input and 500 output tokens: Flex $0.375 / $1.875, Priority $1.35 / $6.75
    # (an upper bound: Google bills a downgraded Priority request at Standard, which this endpoint does not
    # report), Standard $0.75 / $3.75 per 1M tokens.
    bill(input, output) = (1_000 * input + 500 * output) / 1_000_000
    chat(tier, stream) = Chat(service=_CompatHere, model="gemini-3.8-flash", service_tier=tier, messages=[Message(Val(:user), "hi")],
                              stream=(stream ? true : nothing), stream_options=(stream ? Dict("include_usage" => true) : nothing))
    cfg = RequestConfig(max_attempts=1, request_timeout=10.0, stream_idle_timeout=10.0, total_deadline=20.0)
    _with_compat() do
        for stream in (false, true), (tier, want) in (("flex", bill(0.375, 1.875)), ("priority", bill(1.35, 6.75)),
                                                      ("standard", bill(0.75, 3.75)), (nothing, bill(0.75, 3.75)))
            c = chat(tier, stream)
            out = chatrequest!(c; config=cfg)
            r = stream ? fetch(out) : out
            @test r isa LLMSuccess && r.usage.service_tier == tier && token_usage(r).prompt_tokens == 1_000
            @test estimated_cost(r) ≈ want && cumulative_cost(c) ≈ want
        end
        # A streamed reply a callback throws on after its finish is billed at the requested tier too.
        c = chat("priority", true)
        r = fetch(chatrequest!(c; config=cfg, callback=(x, _) -> x isa Message && error("callback failed")))
        @test r isa LLMCallError && cumulative_cost(c) ≈ bill(1.35, 6.75)
    end
    # A tier the reply names is kept; with no usage there is nothing to record.
    named = TokenUsage(prompt_tokens=1, service_tier="standard")
    compat = Chat(service=GEMINIOpenAIServiceEndpoint, service_tier="priority")
    @test UniLM._result_usage(GEMINIOpenAIServiceEndpoint, compat, named) === named
    @test isnothing(UniLM._result_usage(GEMINIOpenAIServiceEndpoint, compat, nothing))
    # Every other endpoint records the tier its reply reports, here none.
    u = TokenUsage(prompt_tokens=1)
    for service in (OPENAIServiceEndpoint, GEMINIServiceEndpoint, GenericOpenAIEndpoint("http://localhost:9", ""))
        @test UniLM._result_usage(service, Chat(; service, model="m", service_tier="flex"), u) === u
    end
end

@testset "cost — Flex bills half the standard rates and Priority 1.8 times (pricing page, 2026-10-22)" begin
    usage(tier) = TokenUsage(prompt_tokens=1000, completion_tokens=500, cached_tokens=200, service_tier=tier)
    cost(model, tier) = estimated_cost(LLMSuccess(message=Message(role=RoleAssistant, content="x"),
        self=Chat(service=GEMINIServiceEndpoint, model=model), usage=usage(tier)))
    bill(input, cached, output) = (800 * input + 200 * cached + 500 * output) / 1_000_000
    # Gemini 3.8 Flash: Flex $0.375 / $0.0375 / $1.875 and Priority $1.35 / $0.135 / $6.75 per 1M
    # tokens, for Standard's $0.75 / $0.075 / $3.75.
    @test cost("gemini-3.8-flash", "flex") ≈ bill(0.375, 0.0375, 1.875)
    @test cost("gemini-3.8-flash", "priority") ≈ bill(1.35, 0.135, 6.75)
    @test cost("gemini-3.8-flash", "standard") == cost("gemini-3.8-flash", "unspecified") ==
          cost("gemini-3.8-flash", nothing) ≈ bill(0.75, 0.075, 3.75)
    # Gemini 4 Argon: Flex $1.00 / $0.05 / $5.00, Priority $3.60 / $0.18 / $18.00.
    @test cost("gemini-4-argon", "flex") ≈ bill(1.0, 0.05, 5.0)
    @test cost("gemini-4-argon", "priority") ≈ bill(3.6, 0.18, 18.0)
    # Where a model's Flex table lists the context-caching price "Same as Standard", a cache hit on
    # Flex bills at the standard rate; on Priority it costs 1.8 times as the rest does.
    @test cost("gemini-3.1-pro-preview", "flex") ≈ bill(1.0, 0.20, 6.0)
    @test cost("gemini-3.1-pro-preview", "priority") ≈ bill(3.6, 0.36, 21.6)
    @test cost("gemini-2.5-flash", "flex") ≈ bill(0.15, 0.03, 1.25)
    @test cost("gemini-2.5-flash-lite", "priority") ≈ bill(0.18, 0.018, 0.72)
    # Gemini 3.5 Flash-Lite lists its cache hits at $0.02 on Flex and $0.05 on Priority, for the
    # standard $0.03 (and $0.15 / $1.25, $0.54 / $4.50 for input and output).
    @test cost("gemini-3.5-flash-lite", "flex") ≈ bill(0.15, 0.02, 1.25)
    @test cost("gemini-3.5-flash-lite", "priority") ≈ bill(0.54, 0.05, 4.50)
    @test cost("gemini-3.5-flash-lite", nothing) ≈ bill(0.30, 0.03, 2.50)
    # Every other Gemini row's cache hit costs the tier's multiple of its standard rate, as its
    # tables list: Gemini 3.1 Flash-Lite $0.0125 (Flex) and $0.045 (Priority) for $0.025.
    @test cost("gemini-3.1-flash-lite", "flex") ≈ bill(0.125, 0.0125, 0.75)
    @test cost("gemini-3.1-flash-lite", "priority") ≈ bill(0.45, 0.045, 2.70)
    @test cost("gemini-3.6-flash", "priority") ≈ bill(1.35, 0.135, 6.75)
    # gemini-3.1-pro-preview-customtools is priced on 3.1 Pro Preview's row of the page.
    @test cost("gemini-3.1-pro-preview-customtools", nothing) ≈ bill(2.0, 0.20, 12.0)
    @test cost("gemini-3.1-pro-preview-customtools", "flex") ≈ bill(1.0, 0.20, 6.0)
    # A tier the page has no price for is estimated at the standard rates, and says so once.
    @test (@test_logs (:warn, r"no price for this service tier") cost("gemini-3.6-flash", "deferred")) ≈
          bill(0.75, 0.075, 3.75)
    # The compatible endpoint bills the same.
    compat = LLMSuccess(message=Message(role=RoleAssistant, content="x"),
        self=Chat(service=GEMINIOpenAIServiceEndpoint, model="gemini-3.8-flash"), usage=usage("flex"))
    @test estimated_cost(compat) ≈ bill(0.375, 0.0375, 1.875)
end

@testset "decode — the audio of a prompt is its AUDIO token count, streamed or not" begin
    # gemini-3.1-flash-lite read a 1.5 s WAV clip as promptTokensDetails TEXT 16, AUDIO 38 (observed
    # 2026-10-22); the audio of a tool-use prompt is in toolUsePromptTokensDetails.
    meta = Dict{String,Any}("promptTokenCount" => 54, "candidatesTokenCount" => 1, "totalTokenCount" => 55,
        "promptTokensDetails" => [Dict("modality" => "TEXT", "tokenCount" => 16), Dict("modality" => "AUDIO", "tokenCount" => 38)])
    reply(m) = Dict("candidates" => [Dict("content" => Dict("role" => "model", "parts" => [Dict("text" => "banana")]),
                    "finishReason" => "STOP")], "usageMetadata" => m, "modelVersion" => "gemini-3.1-flash-lite")
    usage(m) = decode_response(GEMINIServiceEndpoint, HTTP.Response(200, [], Vector{UInt8}(JSON.json(reply(m))))).usage
    @test usage(meta) == TokenUsage(prompt_tokens=54, completion_tokens=1, total_tokens=55, prompt_audio_tokens=38)
    state = StreamState()
    UniLM.handle_sse_event!(GEMINIServiceEndpoint, "", JSON.json(reply(meta)), state)
    @test state.usage == usage(meta)
    tool = merge(meta, Dict("toolUsePromptTokenCount" => 20, "toolUsePromptTokensDetails" =>
        [Dict("modality" => "AUDIO", "tokenCount" => 5), Dict("modality" => "TEXT", "tokenCount" => 15)]))
    @test usage(tool).prompt_tokens == 74 && usage(tool).prompt_audio_tokens == 43
    # A prompt of text and images counts no audio; an entry without a count counts none.
    @test usage(merge(meta, Dict("promptTokensDetails" => [Dict("modality" => "IMAGE", "tokenCount" => 38),
                                                           Dict("modality" => "AUDIO")]))).prompt_audio_tokens == 0
    # A usage written back as usageMetadata (how the hops of a turn are joined) reads back whole.
    u = TokenUsage(prompt_tokens=1_010, completion_tokens=40, total_tokens=1_050, cached_tokens=6, reasoning_tokens=30,
                   service_tier="priority", prompt_audio_tokens=1_000)
    @test UniLM._gemini_usage(UniLM._gemini_usage_metadata(u)) == u
end

@testset "cost — audio bills at the input price the pricing page states for it; anything else is unpriced" begin
    # https://ai.google.dev/gemini-api/docs/pricing (2026-10-22), per 1M tokens: 3.1 Flash-Lite "$0.25
    # (text / image / video) $0.50 (audio)", 2.5 Flash $0.30 / $1.00, 2.5 Flash-Lite $0.10 / $0.30, 3 Flash
    # Preview $0.50 / $1.00; one price "(text / image / video / audio)": 3.5 Flash-Lite $0.30, Robotics-ER 2
    # Preview $1.00; one price that names no modality, every modality's: Gemini 4 Argon $2.00, 3.8 and 3.6
    # Flash $0.75, 3.1 Pro Preview $2.00, 2.5 Pro $1.25. A prompt of 1,000 audio and 10 text tokens, and 1
    # output token:
    heard(; tier=nothing, cached=0, spoken=0) = TokenUsage(prompt_tokens=1_010, completion_tokens=1, total_tokens=1_011,
        cached_tokens=cached, service_tier=tier, prompt_audio_tokens=1_000, completion_audio_tokens=spoken)
    cost(model, u; service=GEMINIServiceEndpoint, pricing=DEFAULT_PRICING) = estimated_cost(LLMSuccess(
        message=Message(role=RoleAssistant, content="x"), self=Chat(; service, model), usage=u); pricing)
    bill(audio, text, output) = (1_000 * audio + 10 * text + 1 * output) / 1_000_000
    @test cost("gemini-3.1-flash-lite", heard()) ≈ bill(0.50, 0.25, 1.50)          # $0.000504
    @test cost("gemini-2.5-flash", heard()) ≈ bill(1.00, 0.30, 2.50)
    @test cost("gemini-2.5-flash-lite", heard()) ≈ bill(0.30, 0.10, 0.40)
    @test cost("gemini-3.5-flash-lite", heard()) ≈ bill(0.30, 0.30, 2.50)
    rows = Dict("gemini-3-flash-preview" => UniLM._price(0.50, 0.05, 3.00),
                "gemini-robotics-er-2-preview" => UniLM._price(1.00, 0.10, 5.00))
    @test cost("gemini-3-flash-preview", heard(); pricing=rows) ≈ bill(1.00, 0.50, 3.00)
    @test cost("gemini-robotics-er-2-preview", heard(); pricing=rows) ≈ bill(1.00, 1.00, 5.00)
    # A one-price model's audio is its row's input rate, so a row that moves the price — a caller's own,
    # or Robotics-ER 2 Preview's doubling on 2027-01-01 — moves both; a model with its own audio price
    # keeps that one.
    moved = Dict("gemini-3.5-flash-lite" => UniLM._price(0.60, 0.06, 5.00),
                 "gemini-robotics-er-2-preview" => UniLM._price(2.00, 0.20, 10.00),
                 "gemini-3.1-flash-lite" => UniLM._price(0.40, 0.04, 2.00))
    @test cost("gemini-3.5-flash-lite", heard(); pricing=moved) ≈ bill(0.60, 0.60, 5.00)
    @test cost("gemini-robotics-er-2-preview", heard(); pricing=moved) ≈ bill(2.00, 2.00, 10.00)
    @test cost("gemini-3.1-flash-lite", heard(); pricing=moved) ≈ bill(0.50, 0.40, 2.00)
    # A billed tier scales the audio rate as it scales the text ones: 3.1 Flash-Lite lists audio at
    # $0.25 on Flex and $0.90 on Priority, 2.5 Flash at $1.80 on Priority.
    @test cost("gemini-3.1-flash-lite", heard(tier="flex")) ≈ bill(0.25, 0.125, 0.75)
    @test cost("gemini-3.1-flash-lite", heard(tier="priority")) ≈ bill(0.90, 0.45, 2.70)
    @test cost("gemini-2.5-flash", heard(tier="priority")) ≈ bill(1.80, 0.54, 4.50)
    # A usage of the OpenAI-compatible endpoint that counted audio would bill the same; its replies count
    # none (the next testset).
    @test cost("gemini-3.1-flash-lite", heard(); service=GEMINIOpenAIServiceEndpoint) ≈ bill(0.50, 0.25, 1.50)
    @test (@inferred UniLM._billed_cost(GEMINIServiceEndpoint, "gemini-3.1-flash-lite", heard(),
                                        DEFAULT_PRICING["gemini-3.1-flash-lite"])) ≈ bill(0.50, 0.25, 1.50)
    # An input price that names no modality is every modality's — the page spells modalities out wherever
    # their rates differ, and where audio is excluded — so audio bills at the row's input rate, at the
    # tier's and the long prompt's multiples as text does; the routed ids are priced as their targets.
    for (model, input, output) in (("gemini-4-argon", 2.00, 10.00), ("gemini-3.8-flash", 0.75, 3.75),
                                   ("gemini-3.7-flash", 0.75, 3.75), ("gemini-3.6-flash", 0.75, 3.75),
                                   ("gemini-3.5-flash", 0.75, 3.75), ("gemini-3.1-pro-preview", 2.00, 12.00),
                                   ("gemini-3.1-pro-preview-customtools", 2.00, 12.00), ("gemini-2.5-pro", 1.25, 10.00))
        @test (@test_logs cost(model, heard())) ≈ bill(input, input, output)
    end
    @test cost("gemini-4-argon", heard(tier="priority")) ≈ bill(3.60, 3.60, 18.00)
    long = TokenUsage(prompt_tokens=250_000, completion_tokens=1_000, total_tokens=251_000, prompt_audio_tokens=200_000)
    @test cost("gemini-3.1-pro-preview", long) ≈ (250_000 * 4.00 + 1_000 * 18.00) / 1_000_000   # "$4.00, prompts > 200k"
    @test cost("gemini-3.8-flash", heard(); service=GEMINIOpenAIServiceEndpoint) ≈ bill(0.75, 0.75, 3.75)
    # A one-price model's cache price names no modality either (3.6 Flash "$0.075", 2.5 Pro "$0.125, prompts
    # <= 200k tokens $0.25, prompts > 200k"), so a cache hit beside audio bills at the row's cached rate,
    # whatever the cached tokens hold, at a long prompt's multiples as text.
    @test (@test_logs cost("gemini-3.6-flash", heard(cached=500))) ≈ (510 * 0.75 + 500 * 0.075 + 1 * 3.75) / 1_000_000
    long_cached = TokenUsage(prompt_tokens=250_000, completion_tokens=1_000, total_tokens=251_000,
                             cached_tokens=50_000, prompt_audio_tokens=200_000)
    @test (@test_logs cost("gemini-2.5-pro", long_cached)) ≈ (200_000 * 2.50 + 50_000 * 0.25 + 1_000 * 15.00) / 1_000_000
    # Unpriced, with one warning per model: an id the table has no entry for, whatever row a caller adds — a
    # near-miss id, or an image model whose input price excludes audio ("$1.50 (text/image/video)") — any
    # audio output, and a cache hit beside audio on a model with an audio price of its own: the page lists
    # its cached audio apart (3.1 Flash-Lite "$0.025 (text / image / video) $0.05 (audio)"), and the usage
    # does not say how many cached tokens are audio.
    says = "no audio-token rates for this model on this endpoint; its estimated cost is 0.0"
    for model in ("gemini-3.1-flash-lite-preview", "gemini-nano-banana-2.1")
        @test (@test_logs (:warn, says) cost(model, heard(); pricing=Dict(model => UniLM._price(1.50, 0.15, 10.0)))) === 0.0
    end
    @test (@test_logs (:warn, says) cost("gemini-3.1-flash-lite", heard(cached=500))) === 0.0
    @test (@test_logs (:warn, says) cost("gemini-3.1-flash-lite", heard(spoken=1))) === 0.0
    @test (@test_logs (:warn, says) cost("gemini-3.8-flash", heard(spoken=1))) === 0.0
    # Without audio, a cache hit bills as before.
    @test (@test_logs cost("gemini-3.1-flash-lite", TokenUsage(prompt_tokens=1_010, completion_tokens=1, cached_tokens=500))) ≈
          (510 * 0.25 + 500 * 0.025 + 1 * 1.50) / 1_000_000
end

@testset "compat — a sound clip sent to a model with its own audio price warns that its estimate is a lower bound" begin
    # Google's OpenAI-compatible endpoint answered a WAV on gemini-3.1-flash-lite with 54 prompt tokens and
    # no token details (2026-10-23), where the native endpoint counts 38 of them as audio: the clip is
    # priced at the text rate. Refuted if: such a request is silent, warns more than once per model, or a
    # model whose one input price is its audio price warns.
    wav = AudioAttachment(joinpath(_GEMINI_FIXTURES, "ollama", "secret_word.wav"))
    asking(model, parts...) = Chat(service=GEMINIOpenAIServiceEndpoint, model=model,
                                   messages=[Message(Val(:user), "What is the secret word?", parts...)])
    says = "Gemini's OpenAI-compatible endpoint reports no audio token count, so this model's audio input is " *
           "estimated at its text rate, a lower bound of its audio rate; send audio through GEMINIServiceEndpoint " *
           "for an exact estimate"
    logs, _ = Test.collect_test_logs() do
        encode_request(GEMINIOpenAIServiceEndpoint, asking("gemini-3.1-flash-lite", wav))
        encode_request(GEMINIOpenAIServiceEndpoint, asking("gemini-3.1-flash-lite", wav))
        encode_request(GEMINIOpenAIServiceEndpoint, asking("gemini-2.5-flash", wav))
    end
    @test [(l.level, l.message, l.kwargs[:model]) for l in logs] ==
          [(Base.CoreLogging.Warn, says, "gemini-3.1-flash-lite"), (Base.CoreLogging.Warn, says, "gemini-2.5-flash")]
    # No warning where the text rate is the audio rate, without a clip, or on the native endpoint, which
    # counts the audio.
    @test_logs encode_request(GEMINIOpenAIServiceEndpoint, asking("gemini-3.5-flash-lite", wav))
    @test_logs encode_request(GEMINIOpenAIServiceEndpoint, asking("gemini-robotics-er-2-preview", wav))
    @test_logs encode_request(GEMINIOpenAIServiceEndpoint, asking("gemini-3.1-flash-lite"))
    @test_logs encode_request(GEMINIServiceEndpoint, Chat(service=GEMINIServiceEndpoint, model="gemini-3.1-flash-lite",
                                                          messages=[Message(Val(:user), "What is the secret word?", wav)]))
end

@testset "cost — 3.1 Pro Preview and 2.5 Pro bill a prompt over 200K tokens at their long-context rates" begin
    # The pricing page (2026-10-22), per 1M tokens: input "$2.00, prompts <= 200k tokens $4.00, prompts >
    # 200k tokens", output $12.00 / $18.00, cache hits $0.20 / $0.40; Priority $3.60 / $7.20, $21.60 /
    # $32.40, $0.36 / $0.72; Flex $1.00 / $2.00, $6.00 / $9.00, its cache "Same as Standard".
    use(prompt; cached=0, tier=nothing) = TokenUsage(prompt_tokens=prompt, completion_tokens=2_000, cached_tokens=cached,
                                                     service_tier=tier)
    cost(model, u; pricing=DEFAULT_PRICING) = estimated_cost(LLMSuccess(message=Message(role=RoleAssistant, content="x"),
                                               self=Chat(service=GEMINIServiceEndpoint, model=model), usage=u); pricing)
    bill(prompt, cached, input, cache, output) = ((prompt - cached) * input + cached * cache + 2_000 * output) / 1_000_000
    # 2.5 Pro (its row as the page lists it): $1.25 / $2.50, $10.00 / $15.00, $0.125 / $0.25; Priority $4.50 /
    # $27.00 / $0.45 above 200K; Flex $1.25 / $7.50 and its cache $0.25, as Standard's.
    pro25 = Dict("gemini-2.5-pro" => UniLM._price(1.25, 0.125, 10.0))
    @test cost("gemini-2.5-pro", use(300_000; cached=100_000); pricing=pro25) ≈ bill(300_000, 100_000, 2.50, 0.25, 15.0)
    @test cost("gemini-2.5-pro", use(300_000; cached=100_000, tier="priority"); pricing=pro25) ≈ bill(300_000, 100_000, 4.50, 0.45, 27.0)
    @test cost("gemini-2.5-pro", use(300_000; cached=100_000, tier="flex"); pricing=pro25) ≈ bill(300_000, 100_000, 1.25, 0.25, 7.50)
    @test cost("gemini-2.5-pro", use(200_000); pricing=pro25) ≈ bill(200_000, 0, 1.25, 0.125, 10.0)
    for model in ("gemini-3.1-pro-preview", "gemini-3.1-pro-preview-customtools")
        @test cost(model, use(300_000)) ≈ bill(300_000, 0, 4.0, 0.40, 18.0)          # $1.236
        @test cost(model, use(300_000; cached=100_000)) ≈ bill(300_000, 100_000, 4.0, 0.40, 18.0)
        @test cost(model, use(300_000; tier="priority")) ≈ bill(300_000, 0, 7.20, 0.72, 32.40)
        @test cost(model, use(300_000; cached=100_000, tier="flex")) ≈ bill(300_000, 100_000, 2.0, 0.40, 9.0)
        # A prompt of 200K tokens exactly is one of at most 200K.
        @test cost(model, use(200_000)) ≈ bill(200_000, 0, 2.0, 0.20, 12.0)
        @test cost(model, use(200_001)) ≈ bill(200_001, 0, 4.0, 0.40, 18.0)
    end
    # gemini-pro-latest is priced as the model that served it.
    d = decode_response(GEMINIServiceEndpoint, HTTP.Response(200, [], Vector{UInt8}(JSON.json(Dict(
        "candidates" => [Dict("content" => Dict("role" => "model", "parts" => [Dict("text" => "OK")]), "finishReason" => "STOP")],
        "usageMetadata" => Dict("promptTokenCount" => 300_000, "candidatesTokenCount" => 2_000, "totalTokenCount" => 302_000),
        "modelVersion" => "gemini-3.1-pro-preview")))))
    latest = LLMSuccess(message=d.message, self=Chat(service=GEMINIServiceEndpoint, model="gemini-pro-latest"),
                        usage=d.usage, model=d.model)
    @test estimated_cost(latest) ≈ bill(300_000, 0, 4.0, 0.40, 18.0)
    # A model whose table lists one price for every prompt length bills a long prompt at it.
    @test cost("gemini-3.8-flash", use(300_000)) ≈ bill(300_000, 0, 0.75, 0.075, 3.75)
end

# ─── Long generations sent in hops (Gemini long decode) ──────────────────────

@testset "continuation — the next request is the same body plus the latest token" begin
    body = encode_request(GEMINIServiceEndpoint, Chat(service=GEMINIServiceEndpoint, model="gemini-4-argon",
        max_tokens=262_144, reasoning_effort="high", messages=[Message(Val(:system), "s"), Message(Val(:user), "Prove it.")]))
    next = UniLM._continuation_body(GEMINIServiceEndpoint, body, "continuation", "TOK1")
    @test JSON.parse(next) == merge(JSON.parse(body), Dict("continuationToken" => "TOK1"))
    @test replace(next, ",\"continuationToken\":\"TOK1\"" => "") == body          # byte for byte
    # Every hop resends the turn's first body, so only the latest token travels.
    @test JSON.parse(UniLM._continuation_body(GEMINIServiceEndpoint, body, "continuation", "TOK2"))["continuationToken"] == "TOK2"
    # MAX_TOKENS also carries a token, but continuing it needs a larger max_output_tokens: the turn ends.
    @test isnothing(UniLM._continuation_body(GEMINIServiceEndpoint, body, "length", "TOK1"))
    @test isnothing(UniLM._continuation_body(GEMINIServiceEndpoint, body, "continuation", nothing))
    @test isnothing(UniLM._continuation_body(GEMINIServiceEndpoint, body, "stop", nothing))
    # Endpoints that do not continue never do.
    @test isnothing(UniLM._continuation_body(OPENAIServiceEndpoint, body, "continuation", "TOK1"))
    @test isnothing(UniLM._continuation_body(GEMINIOpenAIServiceEndpoint, body, "continuation", "TOK1"))
end

# A local server answering each request with the next scripted reply and keeping every
# request body, and the native Gemini wire pointed at it.
const _HOPS_URL = Ref("")
struct _GeminiHops <: UniLM.ServiceEndpoint end
UniLM.get_url(::Type{_GeminiHops}, c::Chat) = _HOPS_URL[] * (c.stream === true ? "/stream" : "/once")
UniLM.auth_header(::Type{_GeminiHops}) = ["Content-Type" => "application/json"]
UniLM.encode_request(::Type{_GeminiHops}, c::Chat) = encode_request(GEMINIServiceEndpoint, c)
UniLM.decode_response(::Type{_GeminiHops}, r::HTTP.Response) = decode_response(GEMINIServiceEndpoint, r)
UniLM.handle_sse_event!(::Type{_GeminiHops}, e::AbstractString, p::AbstractString, s::StreamState) =
    UniLM.handle_sse_event!(GEMINIServiceEndpoint, e, p, s)
# Runs when the driver asks whether a reply continues the turn: after the reply was received,
# before the next request is sent.
const _HOPS_BETWEEN = Ref{Function}(() -> nothing)
UniLM._continuation_body(::Type{_GeminiHops}, b::AbstractString, f, t) =
    (_HOPS_BETWEEN[](); UniLM._continuation_body(GEMINIServiceEndpoint, b, f, t))
UniLM._join_hops(::Type{_GeminiHops}, a::HTTP.Response, b::HTTP.Response) = UniLM._join_hops(GEMINIServiceEndpoint, a, b)
UniLM._request_output_cap(::Type{_GeminiHops}, stream::Bool) = UniLM._request_output_cap(GEMINIServiceEndpoint, stream)
UniLM._stream_head!(::Type{_GeminiHops}, s::StreamState, h::HTTP.Response) = UniLM._stream_head!(GEMINIServiceEndpoint, s, h)
UniLM._billed_cost(::Type{_GeminiHops}, m::String, u::TokenUsage, r::UniLM.PriceRow) =
    UniLM._billed_cost(GEMINIServiceEndpoint, m, u, r)

# Each reply: (status, GenerateContentResponse). A streamed request gets the response as one SSE
# chunk, or the chunks listed under "chunks"; each response is sent `delay` seconds after its request.
function _with_hops(f, replies; tier="priority", delay=0.0)
    seen = String[]
    lk = ReentrantLock()
    server = HTTP.serve!("127.0.0.1", 0; listenany=true, verbose=false) do req
        k = lock(() -> (push!(seen, String(copy(req.body))); length(seen)), lk)
        delay > 0 && sleep(delay)
        status, reply = replies[min(k, length(replies))]
        streamed = endswith(String(req.target), "/stream") && status == 200
        payload = !streamed ? JSON.json(reply) :
            join(("data: " * JSON.json(c) * "\r\n\r\n" for c in get(reply, "chunks", [reply])))
        HTTP.Response(status, ["Content-Type" => streamed ? "text/event-stream" : "application/json",
                               "x-gemini-service-tier" => tier, "x-request-id" => "req-$k"], Vector{UInt8}(payload))
    end
    _HOPS_URL[] = "http://127.0.0.1:$(HTTP.port(server))"
    try
        f(seen)
    finally
        close(server)
    end
end

_hop(parts, finish; token=nothing, usage) = Dict{String,Any}("candidates" => [merge(
    Dict{String,Any}("content" => Dict("role" => "model", "parts" => parts), "finishReason" => finish),
    isnothing(token) ? Dict{String,Any}() : Dict{String,Any}("continuationToken" => token))],
    "usageMetadata" => usage, "modelVersion" => "gemini-4-argon")

# A hop spent on thinking ends at Google's per-request cap with CONTINUATION and a token; the next
# finishes the answer. Argon names its tier in the header only, so the usage carries none.
const _HOP1_USAGE = Dict("promptTokenCount" => 142, "candidatesTokenCount" => 5, "thoughtsTokenCount" => 32_763,
                         "totalTokenCount" => 32_910)
const _HOP2_USAGE = Dict("promptTokenCount" => 32_910, "cachedContentTokenCount" => 31_000, "candidatesTokenCount" => 3,
                         "thoughtsTokenCount" => 40, "totalTokenCount" => 32_953)
_hops_two() = [
    (200, merge(_hop([Dict("text" => "Thinking about it.", "thought" => true),
                      Dict("text" => "The proof begins. ", "thoughtSignature" => "SIG1")], "CONTINUATION";
                     token="TOK1", usage=_HOP1_USAGE),
                Dict("chunks" => [
                    Dict("candidates" => [Dict("content" => Dict("role" => "model",
                         "parts" => [Dict("text" => "Thinking about it.", "thought" => true)]))],
                         "usageMetadata" => Dict("promptTokenCount" => 142, "totalTokenCount" => 142),
                         "modelVersion" => "gemini-4-argon"),
                    Dict("candidates" => [Dict("content" => Dict("role" => "model",
                         "parts" => [Dict("text" => "The proof begins. ", "thoughtSignature" => "SIG1")]),
                         "finishReason" => "CONTINUATION", "continuationToken" => "TOK1")],
                         "usageMetadata" => _HOP1_USAGE, "modelVersion" => "gemini-4-argon")]))),
    (200, _hop([Dict("text" => "It ends.")], "STOP"; usage=_HOP2_USAGE))]

const _HOPS_SUMMED = TokenUsage(prompt_tokens=33_052, completion_tokens=32_811, total_tokens=65_863,
                                cached_tokens=31_000, reasoning_tokens=32_803, service_tier="priority")
# Gemini 4 Argon at Priority: every rate 1.8 times $2.00 / $0.10 / $10.00 per 1M tokens.
const _HOPS_COST = 1.8 * (2_052 * 2.0 + 31_000 * 0.10 + 32_811 * 10.0) / 1_000_000
const _HOP1_COST = 1.8 * (142 * 2.0 + 32_768 * 10.0) / 1_000_000

_hops_chat(; kw...) = Chat(; service=_GeminiHops, model="gemini-4-argon", max_tokens=262_144,
                           reasoning_effort="high", messages=[Message(Val(:system), "s"), Message(Val(:user), "Prove it.")], kw...)
const _HOPS_CFG = RequestConfig(max_attempts=1, request_timeout=10.0, stream_idle_timeout=10.0, total_deadline=20.0)

@testset "continuation — the hops of one turn join into one assistant message, usage summed" begin
    _with_hops(_hops_two()) do seen
        chat = _hops_chat()
        r = chatrequest!(chat; config=_HOPS_CFG)
        @test r isa LLMSuccess
        @test length(seen) == 2
        @test JSON.parse(seen[2]) == merge(JSON.parse(seen[1]), Dict("continuationToken" => "TOK1"))
        @test r.message.content == "The proof begins. It ends."
        @test r.message.finish_reason == STOP
        @test r.message.provider_content.blocks == [Dict("text" => "Thinking about it.", "thought" => true),
            Dict("text" => "The proof begins. ", "thoughtSignature" => "SIG1"), Dict("text" => "It ends.")]
        # "Add each field across hops" (Google's accounting for a REST loop); the header's tier.
        @test r.usage == _HOPS_SUMMED
        @test r.model == "gemini-4-argon"
        @test [m.role for m in chat.messages] == [RoleSystem, RoleUser, RoleAssistant]   # one turn
        @test cumulative_cost(chat) ≈ _HOPS_COST
        @test estimated_cost(r) ≈ _HOPS_COST
    end
end

@testset "continuation — streamed hops stream into the one callback and end in one message" begin
    _with_hops(_hops_two()) do seen
        chat = _hops_chat(stream=true)
        got = Any[]
        r = fetch(chatrequest!(chat; config=_HOPS_CFG, callback=(c, _) -> push!(got, c)))
        @test r isa LLMSuccess
        @test length(seen) == 2
        @test JSON.parse(seen[2]) == merge(JSON.parse(seen[1]), Dict("continuationToken" => "TOK1"))
        @test filter(x -> x isa String, got) == ["The proof begins. ", "It ends."]
        @test count(x -> x isa Message, got) == 1 && last(got) === r.message
        @test r.usage == _HOPS_SUMMED && r.model == "gemini-4-argon"
        @test [m.role for m in chat.messages] == [RoleSystem, RoleUser, RoleAssistant]
        @test cumulative_cost(chat) ≈ _HOPS_COST
        # The same turn not streamed: the same message and usage, so the cost does not depend on `stream`.
        _with_hops(_hops_two()) do _
            plain = chatrequest!(_hops_chat(); config=_HOPS_CFG)
            @test (plain.message.content, plain.message.finish_reason, plain.message.provider_content.blocks) ==
                  (r.message.content, r.message.finish_reason, r.message.provider_content.blocks)
            @test plain.usage == r.usage
        end
    end
end

@testset "continuation — function calls of every hop, numbered across the turn, on both paths" begin
    call(q) = Dict("functionCall" => Dict("name" => "lookup", "args" => Dict("q" => q)))
    u = Dict("promptTokenCount" => 10, "candidatesTokenCount" => 2, "totalTokenCount" => 12)
    replies = [(200, _hop([call("a")], "CONTINUATION"; token="token-a", usage=u)), (200, _hop([call("b")], "STOP"; usage=u))]
    tool = Tool(func=FunctionSignature(name="lookup", parameters=Dict("type" => "object")))
    plain = _with_hops(seen -> chatrequest!(_hops_chat(tools=[tool]); config=_HOPS_CFG), replies)
    fired = ToolCall[]
    streamed = _with_hops(replies) do _
        fetch(chatrequest!(_hops_chat(tools=[tool], stream=true); config=_HOPS_CFG, on_tool_call=tc -> push!(fired, tc)))
    end
    for r in (plain, streamed)
        @test r.message.finish_reason == TOOL_CALLS
        @test [(tc.id, tc.func.arguments["q"]) for tc in r.message.tool_calls] == [("unilm_call_1", "a"), ("unilm_call_2", "b")]
        @test r.usage == TokenUsage(prompt_tokens=20, completion_tokens=4, total_tokens=24, service_tier="priority")
    end
    @test [tc.id for tc in fired] == ["unilm_call_1", "unilm_call_2"]
end

@testset "continuation — the requests the output budget needs, at least 8; then the turn so far, finish_reason \"continuation\"" begin
    # Google: "One hop returns up to 65,536 output tokens. Non-streaming REST requests return up to
    # 32,768." A turn takes cld(max_output_tokens, that cap) + 1 requests, and at least 8.
    u = Dict("promptTokenCount" => 10, "candidatesTokenCount" => 1, "totalTokenCount" => 11)
    replies = [(200, _hop([Dict("text" => "part $k. ")], "CONTINUATION"; token="token-$k", usage=u)) for k in 1:20]
    for (stream, max_tokens, n) in ((false, 262_144, 9), (true, 262_144, 8), (false, nothing, 8), (true, 1_048_576, 17))
        _with_hops(replies) do seen
            chat = _hops_chat(; stream, max_tokens)
            out = chatrequest!(chat; config=_HOPS_CFG)
            r = stream ? fetch(out) : out
            @test length(seen) == n
            @test [get(JSON.parse(b), "continuationToken", nothing) for b in seen] == [nothing; ["token-$k" for k in 1:n-1]]
            @test r.message.finish_reason == "continuation"
            @test r.message.content == join(["part $k. " for k in 1:n])
            @test r.usage == TokenUsage(prompt_tokens=10n, completion_tokens=n, total_tokens=11n, service_tier="priority")
            @test chat.messages[end] === r.message
        end
    end
    cap(; kw...) = UniLM._request_cap(_hops_chat(; kw...))
    @test cap(max_tokens=1_048_576) == 33 && cap(max_tokens=32_768 * 7) == 8 && cap(max_tokens=32_768 * 7 + 1) == 9
    @test cap(max_tokens=nothing, max_completion_tokens=1_048_576) == 33        # the field the encoder sends first
    @test UniLM._request_cap(Chat(service=GEMINIServiceEndpoint, max_tokens=1_048_576, stream=true)) == 17
    @test UniLM._request_cap(Chat(max_tokens=1_048_576)) == 8                    # an endpoint that does not continue
    @test (@inferred UniLM._request_cap(_hops_chat())) == 9
end

@testset "continuation — a CONTINUATION reply that produced no output token ends the turn" begin
    some = Dict("promptTokenCount" => 10, "candidatesTokenCount" => 1, "totalTokenCount" => 11)
    none = Dict("promptTokenCount" => 10, "totalTokenCount" => 10)
    # "A hop spent entirely on thinking returns empty text with finishReason: CONTINUATION. Treat this as
    # normal progress" (Google's long-decode guide): thinking tokens are output.
    thought = Dict("promptTokenCount" => 10, "thoughtsTokenCount" => 32_768, "totalTokenCount" => 32_778)
    ok = (200, _hop([Dict("text" => "Done.")], "STOP"; usage=some))
    for stream in (false, true), (replies, n, finish) in (
            ([(200, _hop(Any[], "CONTINUATION"; token="t1", usage=none)), ok], 1, "continuation"),
            ([(200, _hop([Dict("text" => "a")], "CONTINUATION"; token="t1", usage=some)),
              (200, _hop(Any[], "CONTINUATION"; token="t2", usage=none)), ok], 2, "continuation"),
            ([(200, _hop(Any[], "CONTINUATION"; token="t1", usage=thought)), ok], 2, STOP))
        _with_hops(replies) do seen
            chat = _hops_chat(; stream)
            out = chatrequest!(chat; config=_HOPS_CFG)
            r = stream ? fetch(out) : out
            @test length(seen) == n
            @test r isa LLMSuccess && r.message.finish_reason == finish && chat.messages[end] === r.message
        end
    end
end

@testset "continuation — the audio counts of every hop are summed, on both paths" begin
    audio(n) = Dict("promptTokenCount" => n + 10, "candidatesTokenCount" => 4, "totalTokenCount" => n + 14,
                    "promptTokensDetails" => [Dict("modality" => "AUDIO", "tokenCount" => n), Dict("modality" => "TEXT", "tokenCount" => 10)])
    lite(reply) = merge(reply, Dict("modelVersion" => "gemini-3.1-flash-lite"))
    replies = [(200, lite(_hop([Dict("text" => "a")], "CONTINUATION"; token="t1", usage=audio(1_000)))),
               (200, lite(_hop([Dict("text" => "b")], "STOP"; usage=audio(1_004))))]
    for stream in (false, true)
        _with_hops(replies) do seen
            chat = _hops_chat(; stream, model="gemini-3.1-flash-lite")
            out = chatrequest!(chat; config=_HOPS_CFG)
            r = stream ? fetch(out) : out
            @test length(seen) == 2
            @test r.usage == TokenUsage(prompt_tokens=2_024, completion_tokens=8, total_tokens=2_032,
                                        service_tier="priority", prompt_audio_tokens=2_004)
            # 3.1 Flash-Lite at Priority: audio $0.90, text $0.45, output $2.70 per 1M tokens.
            @test cumulative_cost(chat) ≈ (2_004 * 0.90 + 20 * 0.45 + 8 * 2.70) / 1_000_000
        end
    end
end

# A server streaming each GenerateContentResponse of `hops[k]` to request k (the last for any later
# one) in a write of its own, `gap` seconds apart, so a client reads them one at a time.
function _with_paced_hops(f, hops; gap=0.3)
    n = Threads.Atomic{Int}(0)
    server = HTTP.listen!("127.0.0.1", 0; verbose=false) do http::HTTP.Stream
        read(http)
        k = Threads.atomic_add!(n, 1) + 1
        HTTP.setstatus(http, 200)
        HTTP.setheader(http, "Content-Type" => "text/event-stream")
        HTTP.setheader(http, "x-gemini-service-tier" => "priority")
        HTTP.startwrite(http)
        for c in hops[min(k, length(hops))]
            write(http, "data: " * JSON.json(c) * "\r\n\r\n")
            flush(http)
            sleep(gap)
        end
    end
    _HOPS_URL[] = "http://127.0.0.1:$(HTTP.port(server))"
    try
        f()
    finally
        close(server)
    end
end
_with_paced_chunks(f, chunks; gap=0.3) = _with_paced_hops(f, [chunks]; gap)

@testset "continuation — user code that throws after a reply completed ends the call; that reply stays billed" begin
    # A callback or on_tool_call that throws on the chunk with the reply's finish reason and usage ends
    # the call with that exception and commits nothing, but the reply was answered: it is billed, on a
    # continuing hop as on a single-request stream.
    boom = ErrorException("user code failed")
    ends_call(r) = r isa LLMCallError && r.cause === boom
    _with_hops(_hops_two()) do seen
        chat = _hops_chat(stream=true)
        r = fetch(chatrequest!(chat; config=_HOPS_CFG, callback=(c, _) -> c == "The proof begins. " && throw(boom)))
        @test ends_call(r) && length(seen) == 1
        @test [m.role for m in chat.messages] == [RoleSystem, RoleUser]
        @test cumulative_cost(chat) ≈ _HOP1_COST
    end
    call(q) = Dict("functionCall" => Dict("name" => "lookup", "args" => Dict("q" => q)))
    u = Dict("promptTokenCount" => 10, "candidatesTokenCount" => 2, "totalTokenCount" => 12)
    one_cost = 1.8 * (10 * 2.0 + 2 * 10.0) / 1_000_000                          # Gemini 4 Argon at Priority
    tool = Tool(func=FunctionSignature(name="lookup", parameters=Dict("type" => "object")))
    _with_hops([(200, _hop([call("a")], "CONTINUATION"; token="token-a", usage=u)), (200, _hop([call("b")], "STOP"; usage=u))]) do seen
        chat = _hops_chat(tools=[tool], stream=true)
        r = fetch(chatrequest!(chat; config=_HOPS_CFG, on_tool_call=_ -> throw(boom)))
        @test ends_call(r) && length(seen) == 1
        @test cumulative_cost(chat) ≈ one_cost
    end
    # One request: a throw on the text of its last chunk, or on the final message — also when a stop
    # set on that text came first, so the final message is built on the stop's path.
    stop_then_throw = (c, close) -> c isa Message ? throw(boom) : (c == "Done." && (close[] = true))
    for cb in ((c, _) -> c == "Done." && throw(boom), (c, _) -> c isa Message && throw(boom), stop_then_throw)
        _with_hops([(200, _hop([Dict("text" => "Done.")], "STOP"; usage=u))]) do seen
            chat = _hops_chat(stream=true)
            r = fetch(chatrequest!(chat; config=_HOPS_CFG, callback=cb))
            @test ends_call(r) && length(seen) == 1
            @test [m.role for m in chat.messages] == [RoleSystem, RoleUser]
            @test cumulative_cost(chat) ≈ one_cost
        end
    end
    # A throw before the reply's finish reason and before any usage arrived: nothing is added.
    chunks = [Dict("candidates" => [Dict("content" => Dict("role" => "model", "parts" => [Dict("text" => "Partial ")]))],
                   "modelVersion" => "gemini-4-argon"),
              _hop([Dict("text" => "answer.")], "STOP"; usage=u)]
    _with_paced_chunks(chunks) do
        chat = _hops_chat(stream=true)
        r = fetch(chatrequest!(chat; config=_HOPS_CFG, callback=(c, _) -> c == "Partial " && throw(boom)))
        @test ends_call(r)
        @test cumulative_cost(chat) == 0.0
    end
end

@testset "a stream that ends uncommitted adds the usage Gemini reported, once, on one request and on a continued turn" begin
    # Gemini sends the usage so far on every chunk ("usage_metadata is cumulative within each stream",
    # https://ai.google.dev/gemini-api/docs/generate-content/long-decode), so a stream that ends on a
    # stop, a cancel or a throwing callback after a chunk was billed at least that usage: it is added,
    # once. Gemini 4 Argon at Priority: every rate 1.8 times $2.00 / $10.00 per 1M tokens.
    partial(text, prompt) = Dict("candidates" => [Dict("content" => Dict("role" => "model", "parts" => [Dict("text" => text)]))],
                                 "usageMetadata" => Dict("promptTokenCount" => prompt, "totalTokenCount" => prompt),
                                 "modelVersion" => "gemini-4-argon")
    u = Dict("promptTokenCount" => 10, "candidatesTokenCount" => 2, "totalTokenCount" => 12)
    rest = _hop([Dict("text" => "answer.")], "STOP"; usage=u)
    boom = ErrorException("user code failed")
    for how in (:close, :token, :throw)
        tok = CancelToken()
        act(close) = how === :close ? (close[] = true) : how === :token ? cancel!(tok) : throw(boom)
        stop_on(word) = (c, close) -> c == word && act(close)
        # One request, ended after its first chunk.
        _with_paced_chunks([partial("Partial ", 142), rest]) do
            chat = _hops_chat(stream=true)
            r = fetch(chatrequest!(chat; config=_HOPS_CFG, cancel=tok, callback=stop_on("Partial ")))
            @test r isa LLMCallError && [m.role for m in chat.messages] == [RoleSystem, RoleUser]
            @test cumulative_cost(chat) ≈ 1.8 * 142 * 2.0 / 1_000_000
        end
        # A continued turn, ended in its second request after that request's first chunk: hop 1 and
        # what hop 2 reported, each once.
        tok = CancelToken()
        hop1 = first(_hops_two())[2]["chunks"]
        _with_paced_hops([hop1, [partial("It ", 32_910), rest]]) do
            chat = _hops_chat(stream=true)
            r = fetch(chatrequest!(chat; config=_HOPS_CFG, cancel=tok, callback=stop_on("It ")))
            @test r isa LLMCallError && [m.role for m in chat.messages] == [RoleSystem, RoleUser]
            @test cumulative_cost(chat) ≈ _HOP1_COST + 1.8 * 32_910 * 2.0 / 1_000_000
        end
    end
    # Run to its end, the turn is billed its reply's own usage once, not the partial usage beside it.
    _with_paced_chunks([partial("Partial ", 142), rest]) do
        chat = _hops_chat(stream=true)
        r = fetch(chatrequest!(chat; config=_HOPS_CFG))
        @test r isa LLMSuccess && r.message.content == "Partial answer."
        @test cumulative_cost(chat) ≈ estimated_cost(r) ≈ 1.8 * (10 * 2.0 + 2 * 10.0) / 1_000_000
    end
end

@testset "continuation — a reply that does not continue ends the turn" begin
    u = Dict("promptTokenCount" => 10, "candidatesTokenCount" => 1, "totalTokenCount" => 11)
    for (finish, token, want) in (("MAX_TOKENS", "token-a", "length"), ("CONTINUATION", nothing, "continuation"), ("STOP", nothing, STOP))
        for stream in (false, true)
            _with_hops([(200, _hop([Dict("text" => "x")], finish; token, usage=u))]) do seen
                out = chatrequest!(_hops_chat(; stream); config=_HOPS_CFG)
                r = stream ? fetch(out) : out
                @test length(seen) == 1
                @test r.message.finish_reason == want
            end
        end
    end
end

@testset "continuation — a later hop that fails ends the call; the hops already answered stay billed" begin
    error_reply = Dict("error" => Dict("code" => 400, "message" => "Continuation token has already used output tokens",
                                       "status" => "INVALID_ARGUMENT"))
    replies = [first(_hops_two()), (400, error_reply)]
    for stream in (false, true)
        _with_hops(replies) do seen
            chat = _hops_chat(; stream)
            got = Any[]
            out = stream ? chatrequest!(chat; config=_HOPS_CFG, callback=(c, _) -> push!(got, c)) :
                           chatrequest!(chat; config=_HOPS_CFG)
            r = stream ? fetch(out) : out
            @test length(seen) == 2
            @test r isa LLMFailure && r.status == 400 && occursin("already used output tokens", r.response)
            @test [m.role for m in chat.messages] == [RoleSystem, RoleUser]      # nothing committed
            @test cumulative_cost(chat) ≈ _HOP1_COST                             # hop 1 was billed
            stream && @test filter(x -> x isa String, got) == ["The proof begins. "] && !any(x -> x isa Message, got)
        end
    end
    # A later hop whose 200 cannot be read is a call error naming that hop's request id; the hops
    # before it stay billed too.
    _with_hops([first(_hops_two()), (200, Dict("candidates" => "not a list"))]) do seen
        chat = _hops_chat()
        r = chatrequest!(chat; config=_HOPS_CFG)
        @test r isa LLMCallError && length(seen) == 2
        @test r.request_id == "req-2"
        @test cumulative_cost(chat) ≈ _HOP1_COST
    end
end

@testset "continuation — a stop on the reply that continues ends the call; that reply stays billed" begin
    # The caller stops on the text of hop 1's last chunk, the one that carries CONTINUATION and
    # hop 1's whole usage: by the close flag or by a cancel of its token. The call ends at the next
    # request, which is never sent; the stop is reported as hop 1's, timed from the call's start.
    for by_token in (false, true)
        _with_hops(_hops_two(); delay=0.4) do seen
            chat = _hops_chat(stream=true)
            tok = CancelToken()
            got = Any[]
            stop!(close) = by_token ? cancel!(tok) : (close[] = true)
            r = fetch(chatrequest!(chat; config=_HOPS_CFG, cancel=tok,
                                   callback=(c, close) -> (push!(got, c); c == "The proof begins. " && stop!(close))))
            @test r isa LLMCallError && r.cause isa UniLMCancelled
            @test r.cause.source === (by_token ? :token : :callback)
            @test r.cause.elapsed >= 0.4                                         # hop 1's response took 0.4 s
            @test r.request_id == "req-1"
            @test length(seen) == 1                                              # hop 2 never sent
            @test [m.role for m in chat.messages] == [RoleSystem, RoleUser]      # nothing committed
            @test !any(x -> x isa Message, got)                                  # no final message
            @test cumulative_cost(chat) ≈ _HOP1_COST                             # hop 1 was answered: billed
        end
    end
end

@testset "continuation — a cancel between hops ends the call before the next request; earlier hops stay billed" begin
    for stream in (false, true)
        _with_hops(_hops_two(); delay=0.4) do seen
            chat = _hops_chat(; stream)
            tok = CancelToken()
            _HOPS_BETWEEN[] = () -> cancel!(tok)          # once hop 1 is received, before hop 2 is sent
            try
                out = chatrequest!(chat; config=_HOPS_CFG, cancel=tok)
                r = stream ? fetch(out) : out
                @test r isa LLMCallError && r.cause isa UniLMCancelled && r.cause.source === :token
                @test r.cause.elapsed >= 0.4               # timed from the call's start
                @test length(seen) == 1
                # A stream's cancel lands while hop 1's response is still open; a plain call has read
                # that response whole, and no request of hop 2 was sent.
                @test r.request_id == (stream ? "req-1" : nothing)
                @test [m.role for m in chat.messages] == [RoleSystem, RoleUser]
                @test cumulative_cost(chat) ≈ _HOP1_COST
            finally
                _HOPS_BETWEEN[] = () -> nothing
            end
        end
    end
end

@testset "seed — an int32 on both Gemini wires, refused outside it before any request" begin
    native(seed) = JSON.parse(encode_request(GEMINIServiceEndpoint, Chat(service=GEMINIServiceEndpoint, seed=seed,
        messages=[Message(Val(:user), "hi")])))["generationConfig"]["seed"]
    interaction(seed) = JSON.parse(UniLM.encode_agentic(GEMINIServiceEndpoint,
        Respond(service=GEMINIServiceEndpoint, input="hi", seed=seed)))["generation_config"]["seed"]
    for wire in (native, interaction)
        @test wire(typemax(Int32)) == 2_147_483_647 && wire(typemin(Int32)) == -2_147_483_648 && wire(0) == 0
        for bad in (Int64(typemax(Int32)) + 1, Int64(typemin(Int32)) - 1, typemax(Int64))
            e = _gemini_caught(() -> wire(bad))
            @test e isa ArgumentError && occursin("-2147483648 to 2147483647", e.msg) && occursin(string(bad), e.msg)
        end
    end
end
