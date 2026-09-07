# Current provider contracts, with synthetic responses for deterministic coverage.
using Test, HTTP, JSON, UniLM

@testset "Gemini thinking and model parameter validation" begin
    chat = Chat(service=GEMINIServiceEndpoint, model="gemini-3.8-flash", reasoning_effort="low")
    body = JSON.parse(UniLM.encode_request(chat.service, chat))
    @test get(get(body, "generationConfig", Dict()), "thinkingConfig", nothing) ==
          Dict("thinkingLevel" => "LOW")
    r = Respond(service=GEMINIServiceEndpoint, model=chat.model, input="hi",
                reasoning=Reasoning(effort="low", summary="auto"))
    body = JSON.parse(UniLM.encode_agentic(r.service, r))
    @test body["generation_config"]["thinking_level"] == "low"
    @test body["generation_config"]["thinking_summaries"] == "auto"
    for effort in ("none", "minimal", "xhigh", "typo")
        @test_throws ArgumentError UniLM.encode_request(GEMINIServiceEndpoint,
            Chat(service=GEMINIServiceEndpoint, model=chat.model, reasoning_effort=effort))
    end
    for kw in ((temperature=0.5,), (top_p=0.9,), (n=2,))
        @test_throws ArgumentError UniLM.encode_request(GEMINIServiceEndpoint,
            Chat(service=GEMINIServiceEndpoint, model=chat.model; kw...))
    end
    for kw in ((temperature=0.5,), (top_p=0.9,))
        @test_throws ArgumentError UniLM.encode_agentic(GEMINIServiceEndpoint,
            Respond(service=GEMINIServiceEndpoint, model=chat.model, input="hi"; kw...))
    end
    @test_throws ArgumentError UniLM.encode_request(GEMINIServiceEndpoint,
        Chat(service=GEMINIServiceEndpoint, tool_choice="typo"))
    @test_throws ArgumentError UniLM.encode_request(GEMINIServiceEndpoint,
        Chat(service=GEMINIServiceEndpoint, response_format=UniLM.json_object()))
    legacy = Chat(service=GEMINIServiceEndpoint, model="gemini-3.5-flash", temperature=0.5)
    @test JSON.parse(UniLM.encode_request(legacy.service, legacy))["generationConfig"]["temperature"] == 0.5
end

@testset "Gemini function declarations accept JSON Schema" begin
    schema = Dict("type" => "object", "properties" => Dict("city" => Dict("type" => "string")),
                  "required" => ["city"], "additionalProperties" => false)
    chat = Chat(service=GEMINIServiceEndpoint,
                tools=[Tool(func=FunctionSignature(name="weather", parameters=schema))])
    body = JSON.parse(UniLM.encode_request(chat.service, chat))
    declaration = only(only(body["tools"])["functionDeclarations"])
    @test get(declaration, "parametersJsonSchema", nothing) == schema
    @test !haskey(declaration, "parameters")
end

@testset "Gemini streamed parts preserve signatures and exclude thought text" begin
    parts = Any[Dict("text" => "A thought summary", "thought" => true),
                Dict("text" => "hello"), Dict("text" => "", "thoughtSignature" => "signature==")]
    candidate = Dict("content" => Dict("role" => "model", "parts" => parts), "finishReason" => "STOP")
    payload = JSON.json(Dict("candidates" => [candidate]))
    decoded = UniLM.decode_response(GEMINIServiceEndpoint,
        HTTP.Response(200, [], Vector{UInt8}(payload)))
    @test decoded.message.content == "hello"
    state = UniLM.StreamState()
    for part in parts
        UniLM.handle_sse_event!(GEMINIServiceEndpoint, "", JSON.json(Dict("candidates" => [
            Dict("content" => Dict("role" => "model", "parts" => [part]))])), state)
    end
    @test String(take!(state.pending_delta)) == "hello"
    message = UniLM._build_stream_message(state)
    @test message.content == "hello"
    @test message.provider_content isa ProviderContent
    if message.provider_content isa ProviderContent
        @test message.provider_content.blocks == parts
        chat = Chat(service=GEMINIServiceEndpoint,
            messages=[Message(role=UniLM.RoleUser, content="hi"), message,
                      Message(role=UniLM.RoleUser, content="again")])
        body = JSON.parse(UniLM.encode_request(chat.service, chat))
        @test body["contents"][2]["parts"] == parts
    end
end

@testset "Gemini stream errors and prompt blocks terminate explicitly" begin
    for payload in (Dict("error" => Dict("code" => 503, "message" => "overloaded", "status" => "UNAVAILABLE")),
                    Dict("promptFeedback" => Dict("blockReason" => "SAFETY")))
        state = UniLM.StreamState()
        @test UniLM.handle_sse_event!(GEMINIServiceEndpoint, "", JSON.json(payload), state) == :error
        @test state.error isa AbstractDict
    end
    result = UniLM._stream_error_result(Chat(service=GEMINIServiceEndpoint),
        Dict{String,Any}("code" => 503, "message" => "overloaded"), nothing)
    @test result isa LLMFailure && result.status == 503
end

@testset "Invalid webhook tolerance cannot disable replay protection" begin
    for tolerance in (NaN, -Inf, -1.0)
        @test_throws ArgumentError verify_webhook("{}", Dict(), "secret"; tolerance_seconds=tolerance)
    end
end

@testset "Interactions stream preserves initial text and step order" begin
    state = UniLM.AgenticStreamState()
    event(name, data) = UniLM.decode_agentic_stream(GEMINIServiceEndpoint,
        "event: $name\ndata: $(JSON.json(data))\n\n", state)
    event("step.start", Dict("index" => 0, "step" => Dict("type" => "model_output",
        "content" => [Dict("type" => "text", "text" => "Hello")])) )
    event("step.delta", Dict("index" => 0, "delta" => Dict("type" => "text", "text" => " world")))
    event("step.start", Dict("index" => 1, "step" => Dict("type" => "function_call",
        "id" => "call_1", "name" => "weather", "arguments" => Dict())))
    terminal = event("interaction.completed", Dict("interaction" => Dict("id" => "i_1", "status" => "requires_action")))
    out = terminal.data["response"]["output"]
    @test get(first(out), "type", "") == "message"
    @test get(first(out), "content", []) == [Dict("type" => "output_text", "text" => "Hello world")]
    @test get(last(out), "type", "") == "function_call"
    @test String(take!(state.pending_delta)) == "Hello world"
    @test String(take!(state.textbuff)) == "Hello world"
    @test length(out) == 2
end

@testset "OpenAI current model validation and reasoning alias" begin
    for model in ("gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna")
        tools = [Tool(func=FunctionSignature(name="ping"))]
        @test_throws ArgumentError UniLM.encode_request(OPENAIServiceEndpoint, Chat(; model, tools))
        chat = Chat(; model, tools, reasoning_effort="none")
        @test JSON.parse(UniLM.encode_request(chat.service, chat))["reasoning_effort"] == "none"
    end
    for kw in ((temperature=0.5,), (top_p=0.5,), (reasoning_effort="none",),
               (tools=[Tool(func=FunctionSignature(name="ping"))],))
        chat = Chat(model="gpt-6-astra"; kw...)
        @test_throws ArgumentError UniLM.encode_request(chat.service, chat)
    end
    for kw in ((temperature=0.5,), (reasoning=Reasoning(effort="none"),))
        r = Respond(model="gpt-6-astra", input="hello"; kw...)
        @test_throws ArgumentError UniLM.encode_agentic(r.service, r)
    end
    r = Respond(model="gpt-6-astra", input="hello", reasoning=Reasoning(effort="low"),
                tools=[function_tool("ping", "Ping")])
    @test JSON.parse(UniLM.encode_agentic(r.service, r))["reasoning"]["effort"] == "low"
    @test JSON.lower(Reasoning(generate_summary="concise")) == Dict(:summary => "concise")
    @test_throws ArgumentError JSON.lower(Reasoning(summary="auto", generate_summary="concise"))
end

@testset "Chat token budgets reject invalid values" begin
    for kw in ((max_tokens=0,), (max_tokens=-1,), (max_completion_tokens=0,))
        @test_throws ArgumentError Chat(; kw...)
    end
end

@testset "OpenAI persisted reasoning and prompt cache controls" begin
    r = Respond(input="hello", model="gpt-5.6-luna",
        reasoning=Reasoning(effort="low", context="current_turn", mode="standard"),
        prompt_cache_options=PromptCacheOptions(mode="explicit", ttl="30m"))
    body = JSON.parse(UniLM.encode_agentic(r.service, r))
    @test body["reasoning"] == Dict("effort" => "low", "context" => "current_turn", "mode" => "standard")
    @test body["prompt_cache_options"] == Dict("mode" => "explicit", "ttl" => "30m")
    @test JSON.lower(Reasoning(mode="pro")) == Dict(:mode => "pro")
    @test_throws ArgumentError Reasoning(context="typo")
    @test_throws ArgumentError Reasoning(mode="typo")
    @test_throws ArgumentError PromptCacheOptions(mode="typo")
    @test_throws ArgumentError PromptCacheOptions(ttl="24h")
    @test_throws ArgumentError Respond(input="hi", prompt_cache_retention="24h",
        prompt_cache_options=PromptCacheOptions())
    @test_throws ArgumentError UniLM.encode_agentic(OPENAIServiceEndpoint,
        Respond(input="hi", model="gpt-5.6-luna", prompt_cache_retention="24h"))
    for reasoning in (Reasoning(context="auto"), Reasoning(mode="standard"))
        @test_throws ArgumentError UniLM.encode_agentic(GEMINIServiceEndpoint,
            Respond(service=GEMINIServiceEndpoint, input="hi"; reasoning))
    end
    copy = UniLM._next_respond(r; input="next", previous_response_id="r1")
    @test copy.prompt_cache_options == r.prompt_cache_options
    @test copy.reasoning == r.reasoning
end

@testset "Current transcription multipart arrays and legacy language" begin
    mktemp() do path, io
        write(io, "synthetic audio fixture")
        flush(io)
        request = TranscriptionRequest(file=path, languages=["en", "fr"], keywords=["Julia", "UniLM"])
        @test request.model == "gpt-transcribe"
        parts = UniLM._transcription_parts(request)
        @test [v for (k,v) in parts if k == "languages[]"] == ["en", "fr"]
        @test [v for (k,v) in parts if k == "keywords[]"] == ["Julia", "UniLM"]
        @test !any(p -> first(p) == "language", parts)
        for model in ("gpt-transcribe", "gpt-4o-transcribe")
            parts = UniLM._transcription_parts(TranscriptionRequest(file=path, model=model, language="en"))
            key = model == "gpt-transcribe" ? "languages[]" : "language"
            @test [v for (k,v) in parts if k == key] == ["en"]
        end
        @test_throws ArgumentError TranscriptionRequest(file=path, language="en", languages=["en"])
        @test_throws ArgumentError TranscriptionRequest(file=path, keywords=["two\nlines"])
    end
end

@testset "Current models and dated snapshots have price estimates" begin
    for model in ("gpt-5.6-luna", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-6-astra", "gemini-3.8-flash")
        @test haskey(DEFAULT_PRICING, model)
    end
    object(model) = ResponseSuccess(response=ResponseObject(id="r", status="completed", model=model,
        output=Any[], usage=Dict("input_tokens"=>100, "output_tokens"=>10, "total_tokens"=>110), raw=Dict()))
    @test estimated_cost(object("gpt-5.4-mini-2026-03-17")) == estimated_cost(object("gpt-5.4-mini")) > 0
    @test estimated_cost(object("gpt-5.4-mini-custom")) == 0
end
