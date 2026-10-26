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
    # gemini-2.5-flash predates Gemini 3.6 Flash, from which Gemini ignores sampling parameters.
    legacy = Chat(service=GEMINIServiceEndpoint, model="gemini-2.5-flash", temperature=0.5)
    @test JSON.parse(UniLM.encode_request(legacy.service, legacy))["generationConfig"]["temperature"] == 0.5
end

@testset "Gemini OpenAI-compatible endpoint refuses the sampling parameters Gemini ignores" begin
    compat(; kw...) = Chat(; service=GEMINIOpenAIServiceEndpoint, messages=[Message(role=UniLM.RoleUser, content="hi")], kw...)
    encode(chat) = UniLM.encode_request(GEMINIOpenAIServiceEndpoint, chat)
    for (field, kw) in (("temperature", (temperature=0.5,)), ("top_p", (top_p=0.9,)))
        # gemini-3.8-flash is the endpoint's default model.
        for chat in (compat(; kw...), compat(; model="gemini-4-argon", kw...), compat(; model="gemini-5-x", kw...))
            err = try encode(chat); nothing catch e; e end
            @test err isa ArgumentError && occursin("Gemini ignores $field", err.msg) && occursin(chat.model, err.msg)
        end
        # The models that honour them get the OpenAI-wire body unchanged.
        ok = compat(; model="gemini-2.5-flash", kw...)
        @test encode(ok) == JSON.json(ok)
        @test JSON.parse(encode(ok))[field] == only(values(kw))
    end
    # The server checks the rest: it answers HTTP 400 to prompt_cache_options and to a request ending
    # with an assistant turn (observed on gemini-3.8-flash, 2026-10-22), so both go as given.
    pco = compat(prompt_cache_options=PromptCacheOptions(ttl="30m"))
    @test JSON.parse(encode(pco))["prompt_cache_options"] == Dict("ttl" => "30m")
    prefill = Chat(service=GEMINIOpenAIServiceEndpoint, history=false, messages=[
        Message(role=UniLM.RoleUser, content="Translate 'Hello world' to Spanish."),
        Message(role=UniLM.RoleAssistant, content="Translation:")])
    @test last(JSON.parse(encode(prefill))["messages"]) == Dict("role" => "assistant", "content" => "Translation:")
    # Refused before any request: the encoder runs before any I/O, and the verb lets its refusal through.
    err = try chatrequest!(compat(temperature=0.5)); nothing catch e; e end
    @test err isa ArgumentError && occursin("Gemini ignores temperature", err.msg)
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
    @test takestring!(state.pending_delta) == "hello"
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
    @test takestring!(state.pending_delta) == "Hello world"
    @test takestring!(state.textbuff) == "Hello world"
    @test length(out) == 2
end

# OpenAI's per-model rules on the native endpoint come from the model pages and the GPT-6 guide
# (https://developers.openai.com/api/docs/guides/latest-model); the HTTP 400s quoted below are the
# API's answers to the same requests on 2026-10-22.
_oa_err(f) = try f(); nothing catch e; e end
_oa_chat(; kw...) = (c = Chat(; kw...); JSON.parse(UniLM.encode_request(c.service, c)))
_oa_respond(; kw...) = (r = Respond(; input="hi", kw...); JSON.parse(UniLM.encode_agentic(r.service, r)))
_oa_chat_err(; kw...) = _oa_err(() -> _oa_chat(; kw...))
_oa_respond_err(; kw...) = _oa_err(() -> _oa_respond(; kw...))
const _OA_PING = [Tool(func=FunctionSignature(name="ping"))]
const _OA_NO_ULTRAFAST = "has no ultrafast tier: OpenAI serves service_tier=\"ultrafast\" for gpt-6-astra and " *
    "gpt-6.1-sol (other models answer HTTP 400 \"Invalid service_tier argument\")"

@testset "OpenAI model rules: the exact id or a dated snapshot, never another generation" begin
    rules = UniLM._OPENAI_MODEL_RULES
    @test sort!(collect(keys(rules))) == ["gpt-5.6", "gpt-5.6-luna", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-6-astra",
                                          "gpt-6-luna", "gpt-6-sol", "gpt-6.1-sol"]
    row(model) = UniLM._openai_row(rules, model)
    @test row("gpt-6-sol-2026-09-14") === rules["gpt-6-sol"]
    @test row("gpt-6.1-sol") === rules["gpt-6.1-sol"]
    @test row("gpt-6.1-sol-2026-10-01") === rules["gpt-6.1-sol"]
    # The documented gpt-5.6 alias "routes requests to GPT-5.6 Sol" (its model page; the API
    # answered it as gpt-5.6-sol on 2026-10-22): an exact id with Sol's rules.
    @test row("gpt-5.6") === rules["gpt-5.6-sol"]
    # Near misses and future ids have no row: a family prefix, another date form, another generation.
    for model in ("gpt-7-x", "gpt-6", "gpt-6.1", "gpt-5", "gpt-6-sol-mini", "gpt-6-sol-20260914",
                  "gpt-6-sol-2026-09", "gpt-6.2-sol", "gpt-6-sol-2026-09-14-preview", "gpt-5.6-mini")
        @test isnothing(row(model))
    end
    @test (@inferred Nothing UniLM._openai_row(rules, "gpt-6-luna")) === rules["gpt-6-luna"]
    @test (@inferred Nothing UniLM._openai_row(rules, "gpt-7-x")) === nothing
    # gpt-6.1-sol keeps its own rules in both directions: no "none" and Ultrafast, where
    # gpt-6-sol takes "none" and has no Ultrafast.
    @test _oa_respond_err(model="gpt-6.1-sol", reasoning=Reasoning(effort="none")) isa ArgumentError
    @test _oa_respond(model="gpt-6-sol", reasoning=Reasoning(effort="none"))["reasoning"] == Dict("effort" => "none")
    @test _oa_respond(model="gpt-6.1-sol", service_tier="ultrafast")["service_tier"] == "ultrafast"
    @test _oa_respond_err(model="gpt-6-sol", service_tier="ultrafast").msg == "gpt-6-sol $_OA_NO_ULTRAFAST"
end

@testset "GPT-6 Astra and GPT-6.1 Sol: effort low and up, no sampling, tools through Responses, Ultrafast" begin
    for model in ("gpt-6-astra", "gpt-6.1-sol", "gpt-6.1-sol-2026-10-01")
        # No effort reaches "none", so no sampling control or log probability is ever taken
        # (gpt-6.1-sol: "'temperature' is not supported with this model").
        for effort in (nothing, "low", "xhigh")
            for (kw, name) in ((temperature=0.2,) => "temperature", (top_p=0.5,) => "top_p",
                               (logprobs=true,) => "logprobs", (top_logprobs=2,) => "top_logprobs")
                @test _oa_chat_err(; model, reasoning_effort=effort, kw...).msg ==
                      "$model does not support sampling controls or log probabilities ($name)"
            end
        end
        for effort in (nothing, "low", "max")
            reasoning = isnothing(effort) ? nothing : Reasoning(; effort)
            for (kw, name) in ((temperature=0.2,) => "temperature", (top_p=0.5,) => "top_p",
                               (top_logprobs=2,) => "top_logprobs", (include=["message.output_text.logprobs"],) => "include")
                @test _oa_respond_err(; model, reasoning, kw...).msg ==
                      "$model does not support sampling controls or log probabilities ($name)"
            end
        end
        # The refusal names every field that was set; logprobs=false asks for none.
        @test _oa_chat_err(; model, top_p=0.5, top_logprobs=2).msg ==
              "$model does not support sampling controls or log probabilities (top_p, top_logprobs)"
        @test _oa_chat(; model, logprobs=false)["logprobs"] == false
        @test _oa_respond(; model, include=["reasoning.encrypted_content"])["include"] == ["reasoning.encrypted_content"]
        # Chat Completions serves these models without tools (gpt-6.1-sol: "Function tools with
        # reasoning_effort are not supported"); Responses takes them.
        for effort in (nothing, "low")
            @test _oa_chat_err(; model, tools=_OA_PING, reasoning_effort=effort).msg ==
                  "$model takes function tools only through the Responses API: send them with Respond " *
                  "(Chat Completions serves $model without tools)"
        end
        @test only(_oa_respond(; model, reasoning=Reasoning(effort="low"),
                               tools=[function_tool("ping", "Ping")])["tools"])["name"] == "ping"
        @test _oa_respond(; model, service_tier="ultrafast")["service_tier"] == "ultrafast"
        @test occursin("prompt_cache_options", _oa_respond_err(; model, prompt_cache_retention="24h").msg)
    end
    @test JSON.lower(Reasoning(generate_summary="concise")) == Dict(:summary => "concise")
    @test_throws ArgumentError JSON.lower(Reasoning(summary="auto", generate_summary="concise"))
end

@testset "GPT-5.6: sampling, log probabilities and Chat tools only at effort none" begin
    said(err) = err isa ArgumentError ? err.msg : "no refusal: the request goes out"
    for model in ("gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna", "gpt-5.6", "gpt-5.6-luna-2026-07-01")
        for effort in (nothing, "low")
            @test _oa_chat_err(; model, tools=_OA_PING, reasoning_effort=effort).msg ==
                  "$model Chat tools require reasoning_effort=\"none\"; use Respond for reasoning with tools"
        end
        @test _oa_chat(; model, tools=_OA_PING, reasoning_effort="none")["reasoning_effort"] == "none"
        @test only(_oa_respond(; model, tools=[function_tool("ping", "Ping")])["tools"])["name"] == "ping"
        # Sampling controls and log probabilities only at "none", as on GPT-6 Sol and Luna: on 2026-10-25
        # gpt-5.6-luna answered temperature=0.5 at effort "low" with HTTP 400 on Chat Completions
        # ("Unsupported value: 'temperature' does not support 0.5 with this model. Only the default (1)
        # value is supported.") and on Responses ("Unsupported parameter: 'temperature' is not supported
        # with this model."), and took it at "none". An omitted effort is GPT-5.6's default, "medium".
        for (effort, shown) in ((nothing, "\"medium (default)\""), ("low", "\"low\""))
            for kw in ((temperature=0.5,), (top_p=0.5,), (logprobs=true,), (top_logprobs=2,))
                @test said(_oa_chat_err(; model, reasoning_effort=effort, kw...)) == "$model with reasoning effort " *
                    "$shown does not support sampling controls or log probabilities ($(only(keys(kw)))); " *
                    "set reasoning_effort=\"none\" or remove them"
            end
            reasoning = isnothing(effort) ? nothing : Reasoning(; effort)
            for kw in ((temperature=0.5,), (top_p=0.5,), (top_logprobs=2,), (include=["message.output_text.logprobs"],))
                @test said(_oa_respond_err(; model, reasoning, kw...)) == "$model with reasoning effort $shown " *
                    "does not support sampling controls or log probabilities ($(only(keys(kw)))); " *
                    "set reasoning=Reasoning(effort=\"none\") or remove them"
            end
        end
        @test _oa_chat(; model, reasoning_effort="none", temperature=0.5)["temperature"] == 0.5
        @test _oa_respond(; model, reasoning=Reasoning(effort="none"), top_p=0.5)["top_p"] == 0.5
        # gpt-5.6-sol was answered "Invalid service_tier argument" for "ultrafast".
        @test _oa_respond_err(; model, service_tier="ultrafast").msg == "$model $_OA_NO_ULTRAFAST"
        @test occursin("prompt_cache_options", _oa_respond_err(; model, prompt_cache_retention="24h").msg)
    end
end

@testset "OpenAI reasoning efforts per API: Chat Completions stops at xhigh, Responses takes max" begin
    # The API's answers on 2026-10-22, which win over the Chat Completions reference (it lists
    # "max"): Chat Completions refused "max" on every model below — "Supported values are:
    # 'none', 'low', 'medium', 'high', and 'xhigh'", from 'low' on Astra and 6.1 Sol — and
    # "minimal" on GPT-5.6; the Responses API took "max" (gpt-5.6-luna) and refused "minimal"
    # ("…'xhigh', and 'max'") and, on gpt-6.1-sol, "none".
    listed(efforts) = join(repr.(efforts), ", ")
    from_low, from_none = ["low", "medium", "high", "xhigh"], ["none", "low", "medium", "high", "xhigh"]
    for (models, chat) in ((("gpt-6-astra", "gpt-6.1-sol", "gpt-6.1-sol-2026-10-01"), from_low),
                           (("gpt-6-sol", "gpt-6-luna", "gpt-6-sol-2026-09-14", "gpt-5.6-sol", "gpt-5.6-terra",
                             "gpt-5.6-luna", "gpt-5.6", "gpt-5.6-terra-2026-07-01"), from_none))
        responses = [chat; "max"]
        for model in models, effort in ("none", "minimal", "low", "medium", "high", "xhigh", "max")
            if effort in chat
                @test _oa_chat(; model, reasoning_effort=effort)["reasoning_effort"] == effort
            else
                hint = effort == "max" ? "the Responses API takes it: send the request with Respond" : "use \"low\""
                @test _oa_chat_err(; model, reasoning_effort=effort).msg == "$model on OpenAI Chat Completions " *
                      "does not take reasoning effort $(repr(effort)) (it takes $(listed(chat))); $hint"
            end
            if effort in responses
                @test _oa_respond(; model, reasoning=Reasoning(; effort))["reasoning"] == Dict("effort" => effort)
            else
                @test _oa_respond_err(; model, reasoning=Reasoning(; effort)).msg == "$model on OpenAI Responses " *
                      "does not take reasoning effort $(repr(effort)) (it takes $(listed(responses))); use \"low\""
            end
        end
    end
    # Refused before any request is sent.
    @test_throws ArgumentError chatrequest!(Chat(model="gpt-6-luna", reasoning_effort="max",
                                                 messages=[Message(Val(:user), "hi")]))
    # A compatible server keeps its own contract: the effort goes out as set.
    generic = Chat(service=GenericOpenAIEndpoint("http://localhost:9999", ""), model="gpt-6-luna", reasoning_effort="max")
    @test JSON.parse(UniLM.encode_request(generic.service, generic))["reasoning_effort"] == "max"
end

@testset "an unknown OpenAI model gets no per-model rule; the provider-wide ones still apply" begin
    for model in ("gpt-7-x", "gpt-6-sol-mini")
        body = _oa_chat(; model, reasoning_effort="minimal", temperature=0.2, tools=_OA_PING)
        @test (body["reasoning_effort"], body["temperature"]) == ("minimal", 0.2)
        body = _oa_respond(; model, reasoning=Reasoning(effort="none"), top_logprobs=2, prompt_cache_retention="24h")
        @test (body["top_logprobs"], body["prompt_cache_retention"]) == (2, "24h")
        @test _oa_respond_err(; model, service_tier="ultrafast").msg == "$model $_OA_NO_ULTRAFAST"
        @test _oa_chat_err(; model, service_tier="turbo") isa ArgumentError
        @test _oa_chat_err(; model, user="u1") isa ArgumentError
    end
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
    @test_throws ArgumentError("prompt cache ttl must be one of \"5m\", \"1h\", \"30m\", \"12h\" (got \"24h\")") (
        PromptCacheOptions(ttl="24h"))
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

@testset "OpenAI cache TTLs per API" begin
    # https://developers.openai.com/api/docs/guides/prompt-caching: the Responses API takes
    # "30m" and "12h"; Chat Completions answered "12h" with HTTP 400 "Supported values are:
    # '30m'." (2026-10-22).
    chat(ttl) = Chat(model="gpt-6-luna", prompt_cache_options=PromptCacheOptions(; ttl))
    respond_(ttl) = Respond(model="gpt-6-luna", input="hi", prompt_cache_options=PromptCacheOptions(; ttl))
    failure(f) = try f(); nothing catch e; e end
    err = failure(() -> UniLM.encode_request(OPENAIServiceEndpoint, chat("12h")))
    @test err isa ArgumentError && occursin("\"30m\"", err.msg) && occursin("Chat Completions", err.msg)
    @test JSON.parse(UniLM.encode_request(OPENAIServiceEndpoint, chat("30m")))["prompt_cache_options"] ==
          Dict("ttl" => "30m")
    for ttl in ("30m", "12h")
        @test JSON.parse(UniLM.encode_agentic(OPENAIServiceEndpoint, respond_(ttl)))["prompt_cache_options"] ==
              Dict("ttl" => ttl)
    end
    # Another provider's retention is refused on both APIs, naming OpenAI's.
    for ttl in ("5m", "1h")
        err = failure(() -> UniLM.encode_request(OPENAIServiceEndpoint, chat(ttl)))
        @test err isa ArgumentError && occursin("OpenAI", err.msg) && occursin("\"30m\"", err.msg) &&
              occursin(repr(ttl), err.msg)
        err = failure(() -> UniLM.encode_agentic(OPENAIServiceEndpoint, respond_(ttl)))
        @test err isa ArgumentError && occursin("OpenAI", err.msg) &&
              occursin("\"30m\"", err.msg) && occursin("\"12h\"", err.msg) && occursin(repr(ttl), err.msg)
    end
    # Refused before any request is sent, by the ttl rule.
    @test_throws ArgumentError("OpenAI Chat Completions accepts prompt_cache_options ttl \"30m\" (got \"12h\"); " *
                               "the Responses API (Respond) also takes \"12h\"") chatrequest!(Chat(model="gpt-6-luna",
        messages=[Message(Val(:user), "hi")], prompt_cache_options=PromptCacheOptions(ttl="12h")))
    # A compatible server keeps its own contract: the object goes out unchanged.
    generic = Chat(service=GenericOpenAIEndpoint("http://localhost:9999", ""), model="gpt-6-luna",
                   prompt_cache_options=PromptCacheOptions(ttl="12h"))
    @test JSON.parse(UniLM.encode_request(generic.service, generic))["prompt_cache_options"] == Dict("ttl" => "12h")
end

@testset "GPT-6 Sol and Luna: sampling, log probabilities and Chat tools only at effort none" begin
    tools = [Tool(func=FunctionSignature(name="ping"))]
    for model in ("gpt-6-sol", "gpt-6-luna", "gpt-6-sol-2026-09-14")
        # Omitted effort is the provider default ("medium"), so it is refused like "low".
        for effort in (nothing, "low")
            @test _oa_chat_err(; model, tools, reasoning_effort=effort).msg ==
                  "$model Chat tools require reasoning_effort=\"none\"; use Respond for reasoning with tools"
        end
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
        # Neither model lists a "minimal" reasoning effort; OpenAI's guide moves such a request to "low".
        @test endswith(_oa_chat_err(; model, reasoning_effort="minimal").msg, "; use \"low\"")
        @test endswith(_oa_respond_err(; model, reasoning=Reasoning(effort="minimal")).msg, "; use \"low\"")
        # gpt-6-luna was answered "Invalid service_tier argument" for "ultrafast".
        @test _oa_respond_err(; model, service_tier="ultrafast").msg == "$model $_OA_NO_ULTRAFAST"
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

@testset "OpenAI service tiers: each API's values, Ultrafast on Responses for the models that serve it" begin
    chat_tiers = "\"auto\", \"default\", \"flex\", \"scale\", \"priority\", \"fast\""
    for tier in ("auto", "default", "flex", "scale", "priority", "fast")
        @test _oa_chat(model="gpt-6-luna", service_tier=tier)["service_tier"] == tier
        @test _oa_respond(model="gpt-6-luna", service_tier=tier)["service_tier"] == tier
    end
    # Chat Completions answered "ultrafast" on gpt-6.1-sol with "Invalid service_tier argument".
    @test _oa_chat_err(model="gpt-6.1-sol", service_tier="ultrafast").msg ==
          "OpenAI Chat Completions accepts service_tier $chat_tiers (got \"ultrafast\"); \"ultrafast\" is a " *
          "Responses tier: send it with Respond on gpt-6-astra or gpt-6.1-sol"
    @test _oa_chat_err(model="gpt-6-luna", service_tier="turbo").msg ==
          "OpenAI Chat Completions accepts service_tier $chat_tiers (got \"turbo\")"
    @test _oa_respond_err(model="gpt-6-luna", service_tier="turbo").msg ==
          "OpenAI Responses accepts service_tier $chat_tiers, \"ultrafast\" (got \"turbo\")"
    @test _oa_respond(model="gpt-6.1-sol", reasoning=Reasoning(effort="low"), service_tier="ultrafast")["service_tier"] ==
          "ultrafast"
    @test _oa_respond_err(model="gpt-6-luna", service_tier="ultrafast").msg == "gpt-6-luna $_OA_NO_ULTRAFAST"
    # Refused before any request is sent.
    @test_throws ArgumentError respond("hi"; model="gpt-6-luna", service_tier="ultrafast")
    # A compatible server keeps its own tier names.
    generic = GenericOpenAIEndpoint("http://localhost:9999", "")
    c = Chat(service=generic, model="gpt-6.1-sol", service_tier="ultrafast")
    @test JSON.parse(UniLM.encode_request(generic, c))["service_tier"] == "ultrafast"
    r = Respond(service=generic, model="gpt-6-luna", input="hi", service_tier="turbo")
    @test JSON.parse(UniLM.encode_agentic(generic, r))["service_tier"] == "turbo"
end

@testset "native OpenAI refuses user and sends max_tokens as max_completion_tokens" begin
    # Chat Completions reference: user "is being replaced by safety_identifier and prompt_cache_key";
    # max_tokens "is now deprecated in favor of max_completion_tokens".
    want = "OpenAI replaces the deprecated user field with safety_identifier (a stable id of the end " *
           "user, at most 128 characters) and prompt_cache_key (cache routing): set those instead of user"
    @test _oa_chat_err(model="gpt-6-luna", user="u1").msg == want
    @test _oa_respond_err(model="gpt-6-luna", user="u1").msg == want
    @test_throws ArgumentError chatrequest!(Chat(model="gpt-6-luna", user="u1", messages=[Message(Val(:user), "hi")]))
    body = _oa_chat(model="gpt-6-luna", safety_identifier="u1", prompt_cache_key="tenant-7")
    @test (body["safety_identifier"], body["prompt_cache_key"]) == ("u1", "tenant-7") && !haskey(body, "user")
    body = _oa_chat(model="gpt-6-luna", max_tokens=64)
    @test body["max_completion_tokens"] == 64 && !haskey(body, "max_tokens")
    @test _oa_chat(model="gpt-6-luna", max_completion_tokens=32)["max_completion_tokens"] == 32
    @test _oa_chat_err(model="gpt-6-luna", max_tokens=64, max_completion_tokens=32).msg ==
          "OpenAI sends max_tokens as max_completion_tokens, its current name; set one of them, not both"
    # A body without max_tokens is the Chat's own JSON, byte for byte (recordings key on the bytes).
    c = Chat(model="gpt-6-luna", reasoning_effort="none", temperature=0.2, stream=true)
    push!(c, Message(Val(:system), "s")); push!(c, Message(Val(:user), "u"))
    @test UniLM.encode_request(OPENAIServiceEndpoint, c) == JSON.json(c)
    # Every other OpenAI-wire endpoint sends both fields as set: its server's contract.
    for service in (GenericOpenAIEndpoint("http://localhost:9999", ""), DeepSeekEndpoint(api_key="k"),
                    AZUREServiceEndpoint, GEMINIOpenAIServiceEndpoint)
        b = JSON.parse(UniLM.encode_request(service, Chat(; service, model="m", max_tokens=64, user="u1")))
        @test (b["max_tokens"], b["user"]) == (64, "u1") && !haskey(b, "max_completion_tokens")
    end
    r = Respond(service=GenericOpenAIEndpoint("http://localhost:9999", ""), model="m", input="hi", user="u1")
    @test JSON.parse(UniLM.encode_agentic(r.service, r))["user"] == "u1"
end

@testset "DeepSeek and Mistral get a Chat's token cap as max_tokens, the name their references give; other servers get it as set" begin
    # DeepSeek's Chat Completions reference and Mistral's chat reference list `max_tokens` and no
    # `max_completion_tokens`.
    for service in (DeepSeekEndpoint(api_key="k"), MistralEndpoint(api_key="k"), GenericOpenAIEndpoint("https://api.mistral.ai/", "k"))
        b = JSON.parse(UniLM.encode_request(service, Chat(; service, model="m", max_completion_tokens=32)))
        @test b["max_tokens"] == 32 && !haskey(b, "max_completion_tokens")
        # Set beside max_tokens, max_completion_tokens is taken over it, as the native encoders take it.
        b = JSON.parse(UniLM.encode_request(service, Chat(; service, model="m", max_tokens=64, max_completion_tokens=32)))
        @test b["max_tokens"] == 32 && !haskey(b, "max_completion_tokens")
        # A Chat that sets no max_completion_tokens is its own JSON, byte for byte (recordings key on the bytes).
        c = Chat(; service, model="m", max_tokens=64, temperature=0.2, messages=[Message(Val(:user), "u")])
        @test UniLM.encode_request(service, c) == JSON.json(c)
    end
    # A prefix completion is built by the same encoder.
    chat = Chat(service=DeepSeekEndpoint(api_key="k"), max_completion_tokens=16,
                messages=[Message(Val(:user), "u"), Message(role=UniLM.RoleAssistant, content="```")])
    @test JSON.parse(UniLM._prefix_encode(chat.service, chat))["max_tokens"] == 16
    # OpenAI sends the cap as max_completion_tokens; Azure and Gemini's OpenAI-compatible endpoint send it as set.
    @test _oa_chat(model="gpt-6-luna", max_completion_tokens=32)["max_completion_tokens"] == 32
    for service in (AZUREServiceEndpoint, GEMINIOpenAIServiceEndpoint)
        b = JSON.parse(UniLM.encode_request(service, Chat(; service, model="m", max_completion_tokens=32)))
        @test b["max_completion_tokens"] == 32 && !haskey(b, "max_tokens")
    end
    # Every other generic server gets the names the Chat sets, byte for byte: Azure's v1 route, as the
    # Azure entry builds it, where "Reasoning models will only work with the `max_completion_tokens`
    # parameter" (Azure's reasoning guide); vLLM, which deprecates `max_tokens` "in favor of the
    # max_completion_tokens field"; a proxy to OpenAI's reasoning models.
    azure_v1 = GenericOpenAIEndpoint("https://your-resource.openai.azure.com/openai", "", "AZURE_OPENAI_API_KEY")
    for service in (azure_v1, GenericOpenAIEndpoint("http://localhost:8000", ""), GenericOpenAIEndpoint("https://proxy.example.com", "k"))
        c = Chat(; service, model="your-gpt52-deployment", reasoning_effort="low", max_completion_tokens=2000,
                 messages=[Message(Val(:user), "u")])
        @test UniLM.encode_request(service, c) == JSON.json(c)
        b = JSON.parse(UniLM.encode_request(service, c))
        @test b["max_completion_tokens"] == 2000 && !haskey(b, "max_tokens")
        both = Chat(; service, model="m", max_tokens=64, max_completion_tokens=32, messages=[Message(Val(:user), "u")])
        @test UniLM.encode_request(service, both) == JSON.json(both)
    end
end

@testset "Mistral gets a Chat's seed as random_seed; other servers get the request as set" begin
    # Mistral's chat reference names the seed `random_seed`, "The seed to use for random sampling", and
    # lists no `seed` (https://docs.mistral.ai/api/endpoint/chat).
    for mistral in (MistralEndpoint(api_key="k"), GenericOpenAIEndpoint("https://API.mistral.ai/", "k"))
        b = JSON.parse(UniLM.encode_request(mistral, Chat(service=mistral, model="mistral-large-latest", seed=7)))
        @test b["random_seed"] == 7 && !haskey(b, "seed")
        c = Chat(service=mistral, model="mistral-large-latest", temperature=0.2, messages=[Message(Val(:user), "u")])
        @test UniLM.encode_request(mistral, c) == JSON.json(c)               # no seed: the Chat's own JSON
    end
    for other in (GenericOpenAIEndpoint("http://localhost:8000", ""), GenericOpenAIEndpoint("https://api.mistral.ai.example.com", ""))
        c = Chat(service=other, model="m", seed=7, max_tokens=8, messages=[Message(Val(:user), "u")])
        @test UniLM.encode_request(other, c) == JSON.json(c)                 # byte for byte
    end
end

@testset "Mistral gets an embedding size as output_dimension; other servers as dimensions" begin
    # Mistral's embeddings reference names it `output_dimension`, "The dimension of the output embeddings
    # when feature available", and lists no `dimensions` (https://docs.mistral.ai/api/endpoint/embeddings).
    mistral = MistralEndpoint(api_key="k")
    b = JSON.parse(UniLM._encode_embeddings(mistral, Embeddings(["a", "b"]; service=mistral, model="mistral-embed-2312", dimensions=256)))
    @test b["output_dimension"] == 256 && !haskey(b, "dimensions") && b["input"] == ["a", "b"]
    e = Embeddings("a"; service=mistral, model="mistral-embed-2312")
    @test UniLM._encode_embeddings(mistral, e) == JSON.json(e)              # no size: as set
    other = GenericOpenAIEndpoint("http://localhost:8000", "")
    e = Embeddings("a"; service=other, model="m", dimensions=256)
    @test UniLM._encode_embeddings(other, e) == JSON.json(e)                # byte for byte
    @test JSON.parse(UniLM._encode_embeddings(other, e))["dimensions"] == 256
end

@testset "prompt_cache_retention: refused as deprecated on GPT-5.6 and later, combinable elsewhere" begin
    # The API still answered gpt-6-luna with prompt_cache_retention="24h" (HTTP 200, 2026-10-22): the refusal
    # follows OpenAI's deprecation ("Use prompt_cache_options.ttl instead"), not an API rejection.
    err = _oa_respond_err(model="gpt-6-luna", prompt_cache_retention="24h")
    @test err.msg == "gpt-6-luna takes its cache lifetime from prompt_cache_options, and OpenAI deprecates " *
          "prompt_cache_retention (\"Use prompt_cache_options.ttl instead\"): set " *
          "prompt_cache_options=PromptCacheOptions(ttl=\"30m\") or PromptCacheOptions(ttl=\"12h\")"
    @test !occursin(r"not supported|reject|400"i, err.msg)
    # The two fields "are independent and do not interact": one request may carry both.
    r = Respond(model="gpt-5.5", input="hi", prompt_cache_retention="24h",
                prompt_cache_options=PromptCacheOptions(ttl="30m"))
    body = JSON.parse(UniLM.encode_agentic(r.service, r))
    @test (body["prompt_cache_retention"], body["prompt_cache_options"]) == ("24h", Dict("ttl" => "30m"))
end

@testset "prompt_cache_retention on a Chat: sent on the OpenAI wire, refused where a request has no such field" begin
    msgs = [Message(Val(:user), "hi")]
    chat(service, model; kw...) = Chat(; service, model, messages=msgs, kw...)
    sent(service, model; kw...) = JSON.parse(UniLM.encode_request(service, chat(service, model; kw...)))
    refused(service, model; kw...) =
        try UniLM.encode_request(service, chat(service, model; kw...)); nothing catch e; e end
    # OpenAI's Chat Completions reference still lists it: "Deprecated. Use `prompt_cache_options.ttl` instead."
    @test sent(OPENAIServiceEndpoint, "gpt-5.5"; prompt_cache_retention="24h")["prompt_cache_retention"] == "24h"
    @test sent(OPENAIServiceEndpoint, "gpt-4.1"; prompt_cache_retention="in_memory")["prompt_cache_retention"] == "in_memory"
    @test !haskey(sent(OPENAIServiceEndpoint, "gpt-5.5"), "prompt_cache_retention")
    # The GPT-6 and GPT-5.6 rows take their cache lifetime from prompt_cache_options, as on a Respond; Chat
    # Completions takes ttl "30m" alone, so the fix names that one.
    for model in ("gpt-6-luna", "gpt-6-astra", "gpt-5.6-sol", "gpt-6-sol-2026-09-14")
        err = refused(OPENAIServiceEndpoint, model; prompt_cache_retention="24h")
        @test err isa ArgumentError && err.msg == "$model takes its cache lifetime from prompt_cache_options, and " *
              "OpenAI deprecates prompt_cache_retention (\"Use prompt_cache_options.ttl instead\"): set " *
              "prompt_cache_options=PromptCacheOptions(ttl=\"30m\")"
    end
    # The other OpenAI-wire encoders send it as given, as every other OpenAI field: the server's answer is the check.
    for service in (AZUREServiceEndpoint, GEMINIOpenAIServiceEndpoint, DeepSeekEndpoint(api_key="k"),
                    GenericOpenAIEndpoint("http://localhost:9999", ""), MistralEndpoint(api_key="k"))
        @test sent(service, "m"; prompt_cache_retention="24h")["prompt_cache_retention"] == "24h"
    end
    # Native Gemini, Anthropic and Ollama have no such field: refused before any request, never dropped.
    for (service, model) in ((GEMINIServiceEndpoint, "gemini-3.8-flash"), (ANTHROPICServiceEndpoint(), "claude-haiku-5-5"),
                             (OllamaEndpoint(), "gemma4:e4b"))
        err = refused(service, model; prompt_cache_retention="24h")
        @test err isa ArgumentError && occursin("prompt_cache_retention", err.msg)
    end
    # A Respond to Gemini Interactions refuses it too (its own field there, unmapped).
    @test :prompt_cache_retention in UniLM._INTERACTIONS_UNMAPPED_FIELDS
    # fork copies it, and the positional constructor of 37 arguments still builds a Chat.
    @test fork(chat(OPENAIServiceEndpoint, "gpt-5.5"; prompt_cache_retention="24h")).prompt_cache_retention == "24h"
    c = chat(OPENAIServiceEndpoint, "gpt-5.5")
    @test Chat((getfield(c, f) for f in fieldnames(Chat)[1:37])...).prompt_cache_retention === nothing
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
    # A breakpoint may keep its prefix longer than the request's ttl ("Set prompt_cache_breakpoint.ttl
    # to override it for an individual explicit breakpoint"): the Responses lifetimes, 30 minutes or 12 hours.
    for ttl in ("30m", "12h")
        @test JSON.parse(JSON.json(input_text("Shared prefix"; cache_breakpoint=true, cache_ttl=ttl))) ==
              Dict("type" => "input_text", "text" => "Shared prefix",
                   "prompt_cache_breakpoint" => Dict("mode" => "explicit", "ttl" => ttl))
    end
    err = try input_text("x"; cache_breakpoint=true, cache_ttl="1h"); nothing catch e; e end
    @test err isa ArgumentError &&
          err.msg == "input_text cache_ttl must be \"30m\" or \"12h\", the Responses API's cache lifetimes (got \"1h\")"
    # A lifetime without a breakpoint would have nothing to apply to.
    err = try input_text("x"; cache_ttl="12h"); nothing catch e; e end
    @test err isa ArgumentError && err.msg == "input_text cache_ttl applies to a cache breakpoint: set cache_breakpoint=true"
    r = Respond(model="gpt-6-luna", prompt_cache_options=PromptCacheOptions(mode="explicit", ttl="30m"),
        input=[InputMessage(role="user", content=[input_text("Shared prefix"; cache_breakpoint=true, cache_ttl="12h"),
                                                   input_text("Question")])])
    @test JSON.parse(UniLM.encode_agentic(r.service, r))["input"][1]["content"][1]["prompt_cache_breakpoint"] ==
          Dict("mode" => "explicit", "ttl" => "12h")
end

@testset "Current transcription multipart arrays and legacy language" begin
    mktemp() do path, io
        write(io, "synthetic audio fixture")
        flush(io)
        request = TranscriptionRequest(file=path, languages=["en", "fr"], keywords=["Julia", "UniLM"])
        @test isnothing(request.model) && UniLM._transcription_model(request) == "gpt-transcribe"
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
    for model in ("gpt-5.6-luna", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-6-astra", "gpt-6.1-sol", "gpt-6-sol",
                  "gpt-6-luna", "gemini-3.8-flash")
        @test haskey(DEFAULT_PRICING, model)
    end
    # Every model with an OpenAI rule row is priced, so its spend is never a silent 0.0. The
    # gpt-5.6 alias was answered as gpt-5.6-sol on both APIs (2026-10-22): the model served prices it.
    @test all(m -> haskey(DEFAULT_PRICING, m), setdiff(keys(UniLM._OPENAI_MODEL_RULES), ["gpt-5.6"]))
    alias = LLMSuccess(message=Message(role=RoleAssistant, content="x"), self=Chat(model="gpt-5.6"),
                       usage=TokenUsage(prompt_tokens=16, completion_tokens=5, total_tokens=21), model="gpt-5.6-sol")
    @test estimated_cost(alias) ≈ (16 * 4.0 + 5 * 20.0) / 1e6
    object(model) = ResponseSuccess(response=ResponseObject(id="r", status="completed", model=model,
        output=Any[], usage=Dict("input_tokens"=>100, "output_tokens"=>10, "total_tokens"=>110), raw=Dict()),
        service=OPENAIServiceEndpoint)
    @test estimated_cost(object("gpt-5.4-mini-2026-03-17")) == estimated_cost(object("gpt-5.4-mini")) > 0
    @test estimated_cost(object("gpt-6-sol-2026-09-14")) == estimated_cost(object("gpt-6-sol")) > 0
    @test (@test_logs (:warn, r"no price row") estimated_cost(object("gpt-5.4-mini-custom"))) == 0
end
