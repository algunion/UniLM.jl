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
    # A response_format type with no generateContent counterpart still fails loud.
    @test_throws ArgumentError UniLM.encode_request(GEMINIServiceEndpoint,
        Chat(service=GEMINIServiceEndpoint, response_format=ResponseFormat(type="xml")))
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
    for kw in ((temperature=0.5,), (top_p=0.5,), (reasoning_effort="none",), (logprobs=true,), (top_logprobs=2,),
               (tools=[Tool(func=FunctionSignature(name="ping"))],))
        chat = Chat(model="gpt-6-astra"; kw...)
        @test_throws ArgumentError UniLM.encode_request(chat.service, chat)
    end
    # The sampling / log-probability refusal names the fields that were set.
    err = try UniLM.encode_request(OPENAIServiceEndpoint, Chat(model="gpt-6-astra", top_p=0.5, top_logprobs=2)); nothing catch e; e end
    @test err isa ArgumentError &&
          err.msg == "gpt-6-astra does not support sampling controls or log probabilities (top_p, top_logprobs)"
    err = try UniLM.encode_agentic(OPENAIServiceEndpoint, Respond(model="gpt-6-astra", input="hi", temperature=0.5)); nothing catch e; e end
    @test err isa ArgumentError && err.msg == "gpt-6-astra does not support sampling controls or log probabilities (temperature)"
    # logprobs=false requests no log probabilities: the same rule as GPT-6 Sol and Luna.
    astra = Chat(model="gpt-6-astra", logprobs=false)
    @test JSON.parse(UniLM.encode_request(astra.service, astra))["logprobs"] == false
    for kw in ((temperature=0.5,), (top_logprobs=2,), (reasoning=Reasoning(effort="none"),))
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

@testset "GPT-6 Sol and Luna: Chat tools and sampling need reasoning effort none" begin
    tools = [Tool(func=FunctionSignature(name="ping"))]
    for (model, family) in ("gpt-6-sol" => "GPT-6 Sol", "gpt-6-luna" => "GPT-6 Luna")
        err = try UniLM.encode_request(OPENAIServiceEndpoint, Chat(; model, tools)); nothing catch e; e end
        @test err isa ArgumentError &&
              startswith(err.msg, "$family Chat tools require reasoning_effort=\"none\"") && occursin("Respond", err.msg)
        @test_throws ArgumentError UniLM.encode_request(OPENAIServiceEndpoint, Chat(; model, tools, reasoning_effort="low"))
        chat = Chat(; model, tools, reasoning_effort="none")
        @test JSON.parse(UniLM.encode_request(chat.service, chat))["reasoning_effort"] == "none"

        # Omitted effort is the provider default ("medium"), so it is rejected like "low".
        for (effort, shown) in ((nothing, "\"medium (default)\""), ("low", "\"low\""))
            for kw in ((temperature=0.2,), (top_p=0.5,), (top_logprobs=2,), (logprobs=true,))
                err = try UniLM.encode_request(OPENAIServiceEndpoint,
                        Chat(; model, reasoning_effort=effort, kw...)); nothing catch e; e end
                @test err isa ArgumentError && err.msg == "$model with reasoning effort $shown does not " *
                    "support sampling controls or log probabilities ($(only(keys(kw)))); " *
                    "set reasoning_effort=\"none\" or remove them"
            end
            reasoning = isnothing(effort) ? nothing : Reasoning(; effort)
            for kw in ((temperature=0.2,), (top_p=0.5,), (top_logprobs=2,),
                       (include=["message.output_text.logprobs"],))
                err = try UniLM.encode_agentic(OPENAIServiceEndpoint,
                        Respond(; model, input="hi", reasoning, kw...)); nothing catch e; e end
                @test err isa ArgumentError && err.msg == "$model with reasoning effort $shown does not " *
                    "support sampling controls or log probabilities ($(only(keys(kw)))); " *
                    "set reasoning=Reasoning(effort=\"none\") or remove them"
            end
        end
        chat = Chat(; model, reasoning_effort="none", temperature=0.2, logprobs=true, top_logprobs=2)
        body = JSON.parse(UniLM.encode_request(chat.service, chat))
        @test (body["temperature"], body["logprobs"], body["top_logprobs"]) == (0.2, true, 2)
        r = Respond(; model, input="hi", reasoning=Reasoning(effort="none"), top_p=0.5, top_logprobs=2,
                    include=["message.output_text.logprobs"])
        body = JSON.parse(UniLM.encode_agentic(r.service, r))
        @test (body["top_p"], body["top_logprobs"], body["include"]) == (0.5, 2, ["message.output_text.logprobs"])
        # logprobs=false and unrelated include entries are not log-probability requests:
        # a live gpt-6-luna Chat call with logprobs=false at the default effort succeeded
        # on 2026-09-22.
        chat = Chat(; model, logprobs=false)
        @test JSON.parse(UniLM.encode_request(chat.service, chat))["logprobs"] == false
        r = Respond(; model, input="hi", include=["reasoning.encrypted_content"])
        @test JSON.parse(UniLM.encode_agentic(r.service, r))["include"] == ["reasoning.encrypted_content"]
        # GPT-5.6 and later configure cache lifetime through prompt_cache_options.ttl.
        err = try UniLM.encode_agentic(OPENAIServiceEndpoint,
                Respond(; model, input="hi", prompt_cache_retention="24h")); nothing catch e; e end
        @test err isa ArgumentError && occursin("prompt_cache_options", err.msg)
        # Neither model lists a "minimal" reasoning effort; the error points to "low".
        for request in (() -> UniLM.encode_request(OPENAIServiceEndpoint, Chat(; model, reasoning_effort="minimal")),
                        () -> UniLM.encode_agentic(OPENAIServiceEndpoint,
                                  Respond(; model, input="hi", reasoning=Reasoning(effort="minimal"))))
            err = try request(); nothing catch e; e end
            @test err isa ArgumentError && err.msg == "$model does not support minimal reasoning effort; use \"low\""
        end
    end
    older = Respond(model="gpt-5.5", input="hi", prompt_cache_retention="24h")
    @test JSON.parse(UniLM.encode_agentic(older.service, older))["prompt_cache_retention"] == "24h"
    # The restrictions belong to the native endpoint; a compatible server keeps its own contract.
    generic = Chat(service=GenericOpenAIEndpoint("http://localhost:9999", ""), model="gpt-6-sol",
                   temperature=0.2, tools=tools)
    @test JSON.parse(UniLM.encode_request(generic.service, generic))["temperature"] == 0.2
    generic = Chat(service=generic.service, model="gpt-6-sol", reasoning_effort="minimal")
    @test JSON.parse(UniLM.encode_request(generic.service, generic))["reasoning_effort"] == "minimal"
end

@testset "PromptCacheOptions prewarm and diagnostics; Chat prompt_cache_options wire key" begin
    @test isempty(JSON.lower(PromptCacheOptions()))
    @test JSON.parse(JSON.json(PromptCacheOptions(mode="explicit", ttl="30m"))) ==
          Dict("mode" => "explicit", "ttl" => "30m")
    @test JSON.parse(JSON.json(PromptCacheOptions(prewarm=true, comparison_response_id="resp_1"))) ==
          Dict("prewarm" => true, "comparison_response_id" => "resp_1")
    @test JSON.parse(JSON.json(PromptCacheOptions(prewarm=false))) == Dict("prewarm" => false)
    @test PromptCacheOptions("explicit", "30m") == PromptCacheOptions(mode="explicit", ttl="30m")
    @test_throws ArgumentError PromptCacheOptions(mode="typo", prewarm=true)
    @test_throws ArgumentError PromptCacheOptions(ttl="24h", comparison_response_id="resp_1")

    chat = Chat(model="gpt-5.6-luna", prompt_cache_options=PromptCacheOptions(mode="explicit", ttl="30m"))
    @test JSON.parse(JSON.json(chat))["prompt_cache_options"] == Dict("mode" => "explicit", "ttl" => "30m")
    @test JSON.parse(UniLM.encode_request(chat.service, chat))["prompt_cache_options"] ==
          Dict("mode" => "explicit", "ttl" => "30m")
    @test !haskey(JSON.parse(JSON.json(Chat(model="gpt-5.6-luna"))), "prompt_cache_options")
    # The Chat Completions object has only mode and ttl; a set Responses-only field fails loud.
    for pco in (PromptCacheOptions(prewarm=true), PromptCacheOptions(prewarm=false),
                PromptCacheOptions(mode="explicit", ttl="30m", comparison_response_id="resp_1"))
        bad = Chat(model="gpt-5.6-luna", prompt_cache_options=pco)
        err = try UniLM.encode_request(bad.service, bad); nothing catch e; e end
        @test err isa ArgumentError && err.msg == "Chat Completions prompt_cache_options accepts only " *
            "mode and ttl; prewarm and comparison_response_id are Responses-only"
    end
    r = Respond(input="hi", model="gpt-5.6-luna",
                prompt_cache_options=PromptCacheOptions(prewarm=true, comparison_response_id="resp_1"))
    @test JSON.parse(UniLM.encode_agentic(r.service, r))["prompt_cache_options"] ==
          Dict("prewarm" => true, "comparison_response_id" => "resp_1")
    # Native Gemini Chat has no equivalent, so the field fails loud there.
    @test_throws ArgumentError UniLM.encode_request(GEMINIServiceEndpoint,
        Chat(service=GEMINIServiceEndpoint, prompt_cache_options=PromptCacheOptions(mode="explicit")))
end

@testset "ModerationConfig validation and wire shape on Chat and Respond" begin
    # Both API references list moderation.model as required; the policy is optional.
    @test_throws UndefKeywordError ModerationConfig()
    @test_throws UndefKeywordError ModerationConfig(input_mode="block", output_mode="score")
    @test_throws MethodError ModerationConfig(model=nothing, input_mode="block")
    @test_throws ArgumentError ModerationConfig(model="omni-moderation-latest", input_mode="flag")
    @test_throws ArgumentError ModerationConfig(model="omni-moderation-latest", output_mode="typo")
    full = ModerationConfig(model="omni-moderation-latest", input_mode="block", output_mode="score")
    golden = Dict("model" => "omni-moderation-latest",
                  "policy" => Dict("input" => Dict("mode" => "block"), "output" => Dict("mode" => "score")))
    @test JSON.parse(JSON.json(full)) == golden
    @test JSON.parse(JSON.json(ModerationConfig(model="omni-moderation-latest"))) ==
          Dict("model" => "omni-moderation-latest")
    @test JSON.parse(JSON.json(ModerationConfig(model="omni-moderation-latest", output_mode="block"))) ==
          Dict("model" => "omni-moderation-latest", "policy" => Dict("output" => Dict("mode" => "block")))

    chat = Chat(model="gpt-5.4-mini", moderation=full)
    @test JSON.parse(UniLM.encode_request(chat.service, chat))["moderation"] == golden
    r = Respond(model="gpt-5.4-mini", input="hi", moderation=full)
    @test JSON.parse(UniLM.encode_agentic(r.service, r))["moderation"] == golden
    @test !haskey(JSON.parse(JSON.json(Chat(model="gpt-5.4-mini"))), "moderation")
    @test !haskey(JSON.parse(JSON.json(Respond(input="hi"))), "moderation")
    @test UniLM._next_respond(r; input="next").moderation == full    # tool_loop keeps it
    # The native Gemini wires have no equivalent, so the field fails loud there.
    @test_throws ArgumentError UniLM.encode_request(GEMINIServiceEndpoint,
        Chat(service=GEMINIServiceEndpoint, moderation=full))
    @test_throws ArgumentError UniLM.encode_agentic(GEMINIServiceEndpoint,
        Respond(service=GEMINIServiceEndpoint, input="hi", moderation=full))
end

@testset "ImageGeneration no longer carries the edit-only input_fidelity" begin
    @test !hasfield(ImageGeneration, :input_fidelity)
    @test hasfield(ImageEdit, :input_fidelity)
    @test_throws MethodError ImageGeneration(prompt="x", input_fidelity="high")
    @test_throws MethodError generate_image("x"; input_fidelity="high")    # fails before any request
    ig = ImageGeneration(prompt="x", model="gpt-image-2.5-flare", quality="xhigh", size="1536x864", moderation="low")
    @test JSON.parse(JSON.json(ig)) == Dict("model" => "gpt-image-2.5-flare", "prompt" => "x",
        "quality" => "xhigh", "size" => "1536x864", "moderation" => "low")
end

@testset "ImageGenerationTool and FunctionTool current fields" begin
    t = ImageGenerationTool(model="gpt-image-2.5-flare", action="edit", moderation="low", partial_images=2,
        input_fidelity="high", input_image_mask=Dict("file_id" => "file_mask"), quality="max")
    @test JSON.parse(JSON.json(t)) == Dict("type" => "image_generation", "model" => "gpt-image-2.5-flare",
        "action" => "edit", "moderation" => "low", "partial_images" => 2, "input_fidelity" => "high",
        "input_image_mask" => Dict("file_id" => "file_mask"), "quality" => "max")
    @test JSON.parse(JSON.json(ImageGenerationTool())) == Dict("type" => "image_generation")
    @test JSON.parse(JSON.json(ImageGenerationTool("opaque", "png", 80, "high", "1024x1024"))) ==
          Dict("type" => "image_generation", "background" => "opaque", "output_format" => "png",
               "output_compression" => 80, "quality" => "high", "size" => "1024x1024")
    @test ImageGenerationTool(action="auto", partial_images=0).action == "auto"
    for kw in ((action="typo",), (moderation="high",), (partial_images=4,), (partial_images=-1,),
               (input_fidelity="medium",))
        @test_throws ArgumentError ImageGenerationTool(; kw...)
    end

    schema = Dict("type" => "object", "properties" => Dict("ok" => Dict("type" => "boolean")))
    f = FunctionTool(name="lookup", parameters=Dict("type" => "object", "properties" => Dict()),
        async=true, allowed_callers=["direct", "programmatic"], defer_loading=true, output_schema=schema)
    @test JSON.parse(JSON.json(f)) == Dict("type" => "function", "name" => "lookup",
        "parameters" => Dict("type" => "object", "properties" => Dict()), "async" => true,
        "allowed_callers" => ["direct", "programmatic"], "defer_loading" => true, "output_schema" => schema)
    @test JSON.parse(JSON.json(FunctionTool(name="bare"))) == Dict("type" => "function", "name" => "bare")
    # allowed_callers is an enum array on the Responses wire: direct and/or programmatic.
    @test FunctionTool(name="f", allowed_callers=["programmatic"]).allowed_callers == ["programmatic"]
    err = try FunctionTool(name="f", allowed_callers=["direct", "indirect"]); nothing catch e; e end
    @test err isa ArgumentError &&
          err.msg == "FunctionTool allowed_callers must be \"direct\" or \"programmatic\" (got \"indirect\")"
    for callers in (["Direct"], [""], ["programmatic", "api", "tool"])
        @test_throws ArgumentError FunctionTool(name="f", allowed_callers=callers)
    end
    legacy = FunctionTool("get_weather", "Weather", Dict("type" => "object"), true)
    @test JSON.parse(JSON.json(legacy)) == Dict("type" => "function", "name" => "get_weather",
        "description" => "Weather", "parameters" => Dict("type" => "object"), "strict" => true)

    # Schema and mask dicts accept any key type, like `parameters`.
    @test JSON.parse(JSON.json(FunctionTool(name="f", output_schema=Dict(:type => "object")))) ==
          Dict("type" => "function", "name" => "f", "output_schema" => Dict("type" => "object"))
    @test JSON.parse(JSON.json(ImageGenerationTool(input_image_mask=Dict(:file_id => "file_1")))) ==
          Dict("type" => "image_generation", "input_image_mask" => Dict("file_id" => "file_1"))

    # function_tool carries the current fields from a dict (bare or wrapped) and as keywords.
    spec = JSON.parse("""{"name": "f", "async": true, "allowed_callers": ["direct"],
                          "defer_loading": true, "output_schema": {"type": "object"}}""")
    want = Dict("type" => "function", "name" => "f", "async" => true, "allowed_callers" => ["direct"],
                "defer_loading" => true, "output_schema" => Dict("type" => "object"))
    @test JSON.parse(JSON.json(function_tool(spec))) == want
    @test JSON.parse(JSON.json(function_tool(Dict("type" => "function", "function" => spec)))) == want
    @test JSON.parse(JSON.json(function_tool("f"; async=true, allowed_callers=["direct"], defer_loading=true,
                                             output_schema=Dict(:type => "object")))) == want
    @test_throws ArgumentError function_tool(Dict("name" => "f", "allowed_callers" => ["indirect"]))
end

@testset "Prompt-cache breakpoints and configuration updates" begin
    @test JSON.parse(JSON.json(input_text("Stable prefix"; cache_breakpoint=true))) == Dict("type" => "input_text",
        "text" => "Stable prefix", "prompt_cache_breakpoint" => Dict("mode" => "explicit"))
    @test JSON.parse(JSON.json(input_text("plain"))) == Dict("type" => "input_text", "text" => "plain")
    @test input_text("plain"; cache_breakpoint=false) == input_text("plain")
    @test JSON.parse(JSON.json(configuration_update(effort="high"))) ==
          Dict("type" => "configuration_update", "reasoning" => Dict("effort" => "high"))
    for effort in ("none", "minimal", "low", "medium", "high", "xhigh", "max")
        @test configuration_update(; effort)[:reasoning][:effort] == effort
    end
    @test_throws ArgumentError configuration_update(effort="ultra")
    @test_throws UndefKeywordError configuration_update()
    r = Respond(model="gpt-6-luna", prompt_cache_options=PromptCacheOptions(mode="explicit"),
        input=[configuration_update(effort="low"),
               InputMessage(role="user", content=[input_text("Stable prefix"; cache_breakpoint=true)])])
    body = JSON.parse(UniLM.encode_agentic(r.service, r))
    @test body["input"][1] == Dict("type" => "configuration_update", "reasoning" => Dict("effort" => "low"))
    @test body["input"][2]["content"][1]["prompt_cache_breakpoint"] == Dict("mode" => "explicit")
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
