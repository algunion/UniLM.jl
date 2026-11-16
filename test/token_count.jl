# Token counting (`count_tokens`): the input tokens a request would bill, without running it — a Chat on
# Anthropic (POST /v1/messages/count_tokens) and native Gemini (models.countTokens), a Respond on OpenAI
# (POST /v1/responses/input_tokens). Each body is the one the generation verb's encoder builds, without
# the fields the counting route does not take; the network flow runs against a local server. Offline.

using UniLM: _AnthropicCount, _GeminiCount, _ResponsesCount

# `_CountAt(service)` (platform_seam_fixtures.jl) counts as `service` counts (its wire, its encoder, its
# headers), at a local server.

const _TC_CFG = RequestConfig(max_attempts=1, total_deadline=30.0)
_tc_chat(service; model="claude-haiku-5-5", kw...) =
    push!(push!(Chat(; service, model, kw...), Message(Val(:system), "Be brief.")), Message(Val(:user), "Hello"))
_tc_body(wire, service, request) = JSON.parse(UniLM._count_body(wire, service, request); dicttype=Dict{String,Any})
_tc_encoded(service, chat::Chat) = JSON.parse(UniLM.encode_request(service, chat); dicttype=Dict{String,Any})
_tc_encoded(service, r::Respond) = JSON.parse(UniLM.encode_agentic(service, r); dicttype=Dict{String,Any})
# `count_tokens(request)`, the local server answering each request with `respond(n, req)`: (result, requests).
_tc_count(request, respond::Function; config=_TC_CFG, cancel=nothing) =
    _with_scripted(() -> count_tokens(request; config, cancel), respond)
_tc_serving(status::Int, body::AbstractString; headers=Pair{String,String}[]) = (_, _) -> _json(status, body; headers)

# MessageCountTokensParams (Anthropic SDK v2.9.0): messages, model, cache_control, output_config, speed,
# system, thinking, tool_choice and tools; the count answered max_tokens, stop_sequences, stream, metadata,
# service_tier and inference_geo each with HTTP 400 "Extra inputs are not permitted" (2026-11-16).
# Refuted if: a field the count takes is dropped or changed from the Messages body, or one it refuses is sent.
@testset "Anthropic: the Messages body without the fields the count refuses" begin
    sig = FunctionSignature(name="get_weather", description="Weather",
                            parameters=Dict("type" => "object", "properties" => Dict("city" => Dict("type" => "string"))))
    service = ANTHROPICServiceEndpoint(inference_geo="us")
    chat = _tc_chat(service; model="claude-opus-5-5", max_tokens=64, stop=["END"], tools=[Tool(func=sig)],
                    tool_choice="auto", user="u-1", service_tier="fast", reasoning_effort="low",
                    prompt_cache_options=PromptCacheOptions(ttl="1h"), stream=true)
    full = _tc_encoded(service, chat)
    @test issubset(["max_tokens", "stop_sequences", "stream", "metadata", "inference_geo"], keys(full))
    counted = _tc_body(_AnthropicCount(), service, chat)
    @test counted == filter(((k, _),) -> !(k in ("max_tokens", "stop_sequences", "stream", "metadata", "inference_geo")), full)
    @test issubset(["model", "system", "messages", "tools", "tool_choice", "speed", "cache_control", "output_config"],
                   keys(counted))
    tiered = _tc_chat(ANTHROPICServiceEndpoint; service_tier="auto")
    @test haskey(_tc_encoded(ANTHROPICServiceEndpoint, tiered), "service_tier") &&
          !haskey(_tc_body(_AnthropicCount(), ANTHROPICServiceEndpoint, tiered), "service_tier")
    @test UniLM._count_url(_AnthropicCount(), ANTHROPICServiceEndpoint, chat) == "https://api.anthropic.com/v1/messages/count_tokens"
    # The encoder's refusals hold, before any request.
    e = try count_tokens(_tc_chat(ANTHROPICServiceEndpoint; temperature=0.5)); nothing catch x; x end
    @test e isa ArgumentError && occursin("temperature", e.msg)
end

# The SDK's GA count sends no anthropic-beta header; its beta count sends the caller's betas and
# "token-counting-2024-11-01" (src/anthropic/resources/beta/messages/messages.py, v2.9.0).
# Refuted if: a count without betas sends the beta header, or one with betas leaves out the counting beta,
# names it twice, or drops the endpoint's key, version, user profile or workspace.
@testset "Anthropic: an endpoint's betas go with the counting beta; none, no beta header" begin
    withenv("ANTHROPIC_API_KEY" => "sk-ant-local-0000") do
        h(service) = Dict(UniLM._count_headers(_AnthropicCount(), service))
        @test !haskey(h(ANTHROPICServiceEndpoint), "anthropic-beta")
        @test !haskey(h(ANTHROPICServiceEndpoint()), "anthropic-beta")
        @test h(ANTHROPICServiceEndpoint) == Dict(UniLM.auth_header(ANTHROPICServiceEndpoint))
        every = ANTHROPICServiceEndpoint(betas=["fast-mode-2026-02-01"], user_profile_id="uprof_1", workspace_id="wrkspc_1")
        @test h(every) == merge(Dict(UniLM.auth_header(every)),
                                Dict("anthropic-beta" => "fast-mode-2026-02-01,token-counting-2024-11-01"))
        @test h(every)["anthropic-user-profile-id"] == "uprof_1" && h(every)["anthropic-workspace-id"] == "wrkspc_1"
        @test h(ANTHROPICServiceEndpoint(betas=["token-counting-2024-11-01"]))["anthropic-beta"] == "token-counting-2024-11-01"
    end
end

# CountTokensRequest (Gemini discovery): a generateContentRequest whose `model` is required ("model is not
# specified" is HTTP 400); the count took systemInstruction, tools, toolConfig, generationConfig, labels,
# store and serviceTier on 2026-11-16.
# Refuted if: the count is not the generateContent body, or its model is missing or unprefixed.
@testset "Gemini: the generateContent body, with its model, as the generateContentRequest" begin
    sig = FunctionSignature(name="get_weather", description="Weather", parameters=Dict("type" => "object"))
    chat = _tc_chat(GEMINIServiceEndpoint; model="gemini-3.8-flash", tools=[Tool(func=sig)], tool_choice="auto",
                    reasoning_effort="low", max_tokens=16, safety_identifier="u1", store=false, service_tier="flex")
    @test _tc_body(_GeminiCount(), GEMINIServiceEndpoint, chat) ==
          Dict{String,Any}("generateContentRequest" =>
                           merge(_tc_encoded(GEMINIServiceEndpoint, chat), Dict("model" => "models/gemini-3.8-flash")))
    @test UniLM._count_url(_GeminiCount(), GEMINIServiceEndpoint, _tc_chat(GEMINIServiceEndpoint; model="a b/c")) ==
          "https://generativelanguage.googleapis.com/v1beta/models/a%20b%2Fc:countTokens"
    # The encoder's refusals hold, before any request.
    e = try count_tokens(_tc_chat(GEMINIServiceEndpoint; model="gemini-3.8-flash", user="u-1")); nothing catch x; x end
    @test e isa ArgumentError && occursin("user", e.msg)
end

# TokenCountsBody (OpenAI's OpenAPI spec) holds model, input, instructions, tools, tool_choice,
# parallel_tool_calls, previous_response_id, conversation, reasoning, text and truncation of a Respond's
# fields; the count answered store, include and max_output_tokens with HTTP 400 "Unknown parameter", and so
# context_management and image (2026-11-16).
# Refuted if: a counted field is dropped or changed from respond's body, an output or billing field is sent,
# or a field that changes the input the model reads (compaction, the image policy) is silently dropped.
@testset "OpenAI: a Respond's body as respond sends it, without the fields the count does not take" begin
    sig = FunctionSignature(name="lookup", description="Look up", parameters=Dict("type" => "object"))
    r = Respond(model="gpt-6-luna", input="Hello", instructions="Be brief.", tools=[Tool(func=sig)], tool_choice="auto",
                parallel_tool_calls=false, reasoning=Reasoning(effort="none"), text=TextConfig(verbosity="low"),
                truncation="auto", previous_response_id="resp_1", max_output_tokens=16, store=false,
                include=["reasoning.encrypted_content"], service_tier="flex", metadata=Dict("k" => "v"),
                safety_identifier="u1", prompt_cache_key="p", stream=true, background=false, max_tool_calls=2)
    full = _tc_encoded(OPENAIServiceEndpoint, r)
    counted = _tc_body(_ResponsesCount(), OPENAIServiceEndpoint, r)
    @test sort!(collect(keys(counted))) == ["input", "instructions", "model", "parallel_tool_calls", "previous_response_id",
                                            "reasoning", "text", "tool_choice", "tools", "truncation"]
    @test counted == filter(((k, _),) -> haskey(counted, k), full)
    @test counted["tools"] == [Dict{String,Any}("type" => "function", "name" => "lookup", "description" => "Look up",
                                                "parameters" => Dict{String,Any}("type" => "object"))]
    @test _tc_body(_ResponsesCount(), OPENAIServiceEndpoint, Respond(input="hi", conversation="conv_1"))["conversation"] == "conv_1"
    @test UniLM._count_url(_ResponsesCount(), OPENAIServiceEndpoint, r) == "https://api.openai.com/v1/responses/input_tokens"
    for (kw, says) in (((context_management=[Dict("type" => "compaction")],), "context_management"),
                       ((image_context_compression="auto",), "image_context_compression"))
        e = try count_tokens(Respond(; input="hi", kw...)); nothing catch x; x end
        @test e isa ArgumentError && occursin(says, e.msg) && occursin("count_tokens", e.msg)
    end
    # respond's encoder refusals hold, before any request.
    e = try count_tokens(Respond(input="hi", seed=1)); nothing catch x; x end
    @test e isa ArgumentError && occursin("seed", e.msg)
end

# Refuted if: a request a provider cannot count, or an endpoint without the capability, is sent at all.
@testset "a request its endpoint cannot count is refused before any request, naming what counts" begin
    for (request, says) in ((_tc_chat(OPENAIServiceEndpoint; model="gpt-6-luna"), "count a Respond"),
                            (Respond(service=GEMINIServiceEndpoint, input="hi"), "count a Chat"))
        e = try count_tokens(request); nothing catch x; x end
        @test e isa ArgumentError && occursin("count_tokens", e.msg) && occursin(says, e.msg)
    end
    for service in (AZUREServiceEndpoint, GEMINIOpenAIServiceEndpoint, DeepSeekEndpoint(api_key="k"),
                    GenericOpenAIEndpoint("http://127.0.0.1:1", ""), OllamaEndpoint())
        e = try count_tokens(_tc_chat(service; model="m")); nothing catch x; x end
        @test e isa ArgumentError && startswith(e.msg, "Token counting is not supported by")
    end
    e = try count_tokens(Respond(service=GenericOpenAIEndpoint("http://127.0.0.1:1", ""), model="m", input="hi")); nothing catch x; x end
    @test e isa ArgumentError && startswith(e.msg, "Token counting is not supported by GenericOpenAIEndpoint")
    for service in (OPENAIServiceEndpoint, ANTHROPICServiceEndpoint, ANTHROPICServiceEndpoint(), GEMINIServiceEndpoint)
        @test has_capability(service, :token_count)
    end
end

# Refuted if: a count's total is read from another key or type, the reply is lost from `raw`, or the request
# goes to another route, with another method, body or headers.
@testset "a count is the reply's total, with the reply as sent" begin
    withenv("ANTHROPIC_API_KEY" => "sk-ant-local-0000", "GEMINI_API_KEY" => "g-local", "OPENAI_API_KEY" => "sk-local") do
        gemini = """{"totalTokens":37,"promptTokensDetails":[{"modality":"TEXT","tokenCount":37}],"cachedContentTokenCount":5}"""
        for (service, request, reply, total) in (
                (ANTHROPICServiceEndpoint(betas=["fast-mode-2026-02-01"]), _tc_chat, """{"input_tokens":11}""", 11),
                (GEMINIServiceEndpoint, s -> _tc_chat(s; model="gemini-3.8-flash"), gemini, 37),
                (OPENAIServiceEndpoint, s -> Respond(service=s, model="gpt-6-luna", input="hi"),
                 """{"object":"response.input_tokens","input_tokens":10}""", 10))
            req = request(_CountAt(service))
            sent = Ref{Any}(nothing)
            r, seen = _tc_count(req, (_, http) -> (sent[] = Dict(lowercase(String(k)) => String(v) for (k, v) in http.headers);
                                                  _json(200, reply)))
            @test r isa TokenCountSuccess && issuccess(r) && r.input_tokens == total
            @test r.raw == JSON.parse(reply; dicttype=Dict{String,Any})
            @test [(s.method, s.target) for s in seen] == [("POST", "/count")]
            @test only(seen).body == UniLM._count_body(UniLM._token_counting(service, req), service, req)
            want = Dict(lowercase(k) => v for (k, v) in UniLM._count_headers(UniLM._token_counting(service, req), service))
            @test filter(((k, _),) -> haskey(want, k), sent[]) == want
        end
    end
end

# Refuted if: an error status reads as a count or loses its body, status or request id; a reply that is not
# the documented count reads as one; a missing key, a cancel or the deadline sends anything.
@testset "failures, malformed counts, a missing key, a cancel and the deadline" begin
    withenv("ANTHROPIC_API_KEY" => "sk-ant-local-0000", "GEMINI_API_KEY" => "g-local") do
        chat = _tc_chat(_CountAt(ANTHROPICServiceEndpoint))
        body = """{"type":"error","error":{"type":"invalid_request_error","message":"max_tokens: Extra inputs are not permitted"}}"""
        r, seen = _tc_count(chat, _tc_serving(400, body; headers=["request-id" => "req_011"]))
        @test r isa TokenCountFailure && !issuccess(r) && (r.status, r.response, r.request_id) == (400, body, "req_011")
        @test length(seen) == 1
        for reply in ("{}", """{"input_tokens":"11"}""", """{"input_tokens":-1}""", """{"input_tokens":1.5}""", "[11]", "nope")
            r, seen = _tc_count(chat, _tc_serving(200, reply))
            @test r isa TokenCountCallError && length(seen) == 1
        end
        r, _ = _tc_count(chat, _tc_serving(200, "{}"))
        @test occursin("input_tokens", r.error)
        r, _ = _tc_count(_tc_chat(_CountAt(GEMINIServiceEndpoint); model="gemini-3.8-flash"),
                         _tc_serving(200, """{"promptTokensDetails":[]}"""))
        @test r isa TokenCountCallError && occursin("totalTokens", r.error)
        tok = CancelToken(); cancel!(tok)
        r, seen = _tc_count(chat, _tc_serving(200, """{"input_tokens":1}"""); cancel=tok)
        @test r isa TokenCountCallError && r.cause isa UniLMCancelled && isempty(seen)
        r, seen = _tc_count(chat, _tc_serving(200, """{"input_tokens":1}"""); config=_TINY_DEADLINE)
        @test _seam_timeout(r, TokenCountCallError) && isempty(seen)
    end
    for (service, request, key) in ((ANTHROPICServiceEndpoint, _tc_chat(ANTHROPICServiceEndpoint), "ANTHROPIC_API_KEY"),
                                    (GEMINIServiceEndpoint, _tc_chat(GEMINIServiceEndpoint; model="gemini-3.8-flash"), "GEMINI_API_KEY"),
                                    (OPENAIServiceEndpoint, Respond(input="hi"), "OPENAI_API_KEY"))
        r = withenv(() -> count_tokens(request; config=_TC_CFG), key => nothing)
        @test r isa TokenCountCallError && r.cause isa KeyError && occursin(key, r.error)
    end
end
