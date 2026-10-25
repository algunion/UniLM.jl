# OllamaEndpoint: construction, the /api/chat wire (request, reply, NDJSON stream)
# and the driver paths through a local mock server. Offline.

using Sockets

# A local server speaking /api/chat: the n-th request (1-based) gets
# `respond(n, body)` → (status, lines::Vector{String}); each line is written and
# flushed on its own, `gap` seconds apart, so the client sees separate reads.
function _ollama_mock(respond; gap::Float64=0.05)
    bodies = String[]
    server, port = nothing, 0
    for attempt in 1:5
        tcp = Sockets.listen(Sockets.localhost, 0)
        port = Int(Sockets.getsockname(tcp)[2])
        close(tcp)
        try
            server = HTTP.listen!("127.0.0.1", port; verbose=false) do http::HTTP.Stream
                body = String(read(http))
                push!(bodies, body)
                status, lines = respond(length(bodies), body)
                HTTP.setstatus(http, status)
                HTTP.setheader(http, "Content-Type" => "application/x-ndjson")
                HTTP.startwrite(http)
                for l in lines
                    write(http, l)
                    flush(http)
                    gap > 0 && sleep(gap)
                end
            end
            break
        catch
            attempt == 5 && rethrow()
        end
    end
    server, "http://127.0.0.1:$port", bodies
end

_oline(d) = JSON.json(d) * "\n"
_final(; reason="stop", p=12, c=7, extra...) =
    _oline(Dict{String,Any}("model" => "gemma4:e2b", "message" => Dict("role" => "assistant", "content" => ""),
                            "done" => true, "done_reason" => reason, "prompt_eval_count" => p, "eval_count" => c, extra...))
_delta(; content="", thinking=nothing, tool_calls=nothing) = _oline(Dict{String,Any}("model" => "gemma4:e2b",
    "message" => filter(kv -> !isnothing(kv[2]), Dict{String,Any}("role" => "assistant", "content" => content,
                                                                  "thinking" => thinking, "tool_calls" => tool_calls)),
    "done" => false))
_reply(; content="", thinking=nothing, tool_calls=nothing, reason="stop", p=12, c=7) = JSON.json(Dict{String,Any}(
    "model" => "gemma4:e2b",
    "message" => filter(kv -> !isnothing(kv[2]), Dict{String,Any}("role" => "assistant", "content" => content,
                                                                  "thinking" => thinking, "tool_calls" => tool_calls)),
    "done" => true, "done_reason" => reason, "prompt_eval_count" => p, "eval_count" => c))

_ochat(ep; kws...) = begin
    c = Chat(; service=ep, model="gemma4:e2b", kws...)
    push!(c, Message(Val(:system), "Be brief."))
    push!(c, Message(Val(:user), "Hi"))
    c
end
_wire(c) = JSON.parse(UniLM.encode_request(c.service, c))

const _WEATHER = Tool(func=FunctionSignature(name="get_weather", description="Weather in a city",
    parameters=Dict("type" => "object", "properties" => Dict("city" => Dict("type" => "string")),
                    "required" => ["city"])))

@testset "endpoint construction" begin
    withenv("OLLAMA_HOST" => nothing) do
        e = OllamaEndpoint()
        @test e isa UniLM.OpenAIWireEndpoint
        @test e.base_url == "http://127.0.0.1:11434"
        @test isnothing(e.keep_alive)
        @test e.truncate === false && isnothing(e.shift)        # an input that does not fit is a failure
        @test e.options == OllamaOptions()
        @test sprint(show, e) == "OllamaEndpoint(\"http://127.0.0.1:11434\")"
    end
    withenv("OLLAMA_HOST" => "gpu-box:9999") do
        @test OllamaEndpoint().base_url == "http://gpu-box:9999"
    end
    # The official clients' host rules (ollama-python's _parse_host cases).
    for (host, url) in ("" => "http://127.0.0.1:11434", "1.2.3.4" => "http://1.2.3.4:11434",
                        ":56789" => "http://127.0.0.1:56789", "1.2.3.4:56789" => "http://1.2.3.4:56789",
                        "http://1.2.3.4" => "http://1.2.3.4:80", "https://1.2.3.4" => "https://1.2.3.4:443",
                        "https://1.2.3.4:56789" => "https://1.2.3.4:56789", "example.com" => "http://example.com:11434",
                        "http://example.com" => "http://example.com:80", "https://example.com/path" => "https://example.com:443/path",
                        "[0001:002:003:0004::1]" => "http://[0001:002:003:0004::1]:11434",
                        "[::1]:56789" => "http://[::1]:56789", "  0.0.0.0  " => "http://127.0.0.1:11434",
                        "0.0.0.0:8080" => "http://127.0.0.1:8080", "[::]" => "http://[::1]:11434")
        @test UniLM._ollama_host_url(host) == url
    end
    # Bare IPv6 addresses and Ollama's cloud host, as Ollama's own client reads them.
    @test UniLM._ollama_host_url("::1") == "http://[::1]:11434"
    @test UniLM._ollama_host_url("::") == "http://[::1]:11434"
    @test UniLM._ollama_host_url("fe80::1") == "http://[fe80::1]:11434"
    @test UniLM._ollama_host_url("ollama.com") == "https://ollama.com:443"
    @test UniLM._ollama_host_url("http://ollama.com") == "http://ollama.com:80"
    @test_throws ArgumentError UniLM._ollama_host_url("ftp://box")
    @test_throws ArgumentError UniLM._ollama_host_url("box:99999")
    @test_throws ArgumentError UniLM._ollama_host_url("box:port")
    e = OllamaEndpoint(base_url="http://box:1/", keep_alive=Inf, num_ctx=32_768, top_k=40, min_p=0.05)
    @test e.base_url == "http://box:1"
    @test sprint(show, e) == "OllamaEndpoint(\"http://box:1\"; keep_alive=Inf, num_ctx=32768, top_k=40, min_p=0.05)"
    @test OllamaEndpoint(keep_alive=600).keep_alive === 600.0
    @test sprint(show, OllamaEndpoint(base_url="http://b:1", truncate=true, shift=false)) ==
          "OllamaEndpoint(\"http://b:1\"; truncate=true, shift=false)"
    @test_throws ArgumentError OllamaEndpoint(base_url="box:11434")
    @test_throws ArgumentError OllamaEndpoint(keep_alive=-1)
    @test_throws ArgumentError OllamaEndpoint(keep_alive=NaN)
    # A finite keep_alive beyond 1e9 s would overflow the server's duration: Inf says "stay loaded".
    @test OllamaEndpoint(keep_alive=1e9).keep_alive == 1e9
    @test_throws ArgumentError OllamaEndpoint(keep_alive=1e9 + 1)
    @test_throws ArgumentError OllamaEndpoint(keep_alive=typemax(Int))
    @test OllamaOptions(top_k=0).top_k == 0                           # 0 turns top-k sampling off
    # draft_num_predict, a runner option of Ollama's: the draft tokens per step; 0 turns drafting off.
    @test OllamaOptions(draft_num_predict=0).draft_num_predict == 0
    @test _wire(_ochat(OllamaEndpoint(draft_num_predict=4)))["options"] == Dict("draft_num_predict" => 4)
    @test sprint(show, OllamaEndpoint(base_url="http://b:1", draft_num_predict=4)) ==
          "OllamaEndpoint(\"http://b:1\"; draft_num_predict=4)"
    # Ollama 0.40.11 reads no penalize_newline or numa ("invalid option provided"): refused, not sent.
    for k in (:penalize_newline, :numa)
        err = try OllamaEndpoint(; k => true); nothing catch x; x end
        @test err isa ArgumentError && occursin(String(k), err.msg)
    end
    err = try OllamaEndpoint(num_ctxx=10); nothing catch x; x end
    @test err isa ArgumentError && occursin("num_ctxx", err.msg) && occursin("num_ctx", err.msg)
    for (k, v) in (:num_ctx => 0, :num_batch => 0, :num_thread => 0, :num_gpu => -2, :main_gpu => -1,
                   :num_keep => -2, :top_k => -1, :min_p => 1.5, :repeat_last_n => -2,
                   :repeat_penalty => -0.1, :draft_num_predict => -1)
        @test_throws ArgumentError OllamaOptions(; k => v)
    end
    @test OllamaEndpoint() == OllamaEndpoint()                       # plain immutable value
    @test provider_capabilities(e) == Set([:chat, :embeddings, :fim, :tools, :streaming, :json_output, :responses, :models,
                                           :system_one, :transcription])
    @test !occursin("redacted", sprint(show, e))                     # no key to hide
    @test_throws ArgumentError Chat(service=e)                       # no default model
    @test occursin("list_models(service=OllamaEndpoint())", try Chat(service=e); "" catch x; x.msg end)
end

@testset "URL routing" begin
    e = OllamaEndpoint(base_url="http://box:11434")
    @test UniLM.get_url(e, Chat(service=e, model="m")) == "http://box:11434/api/chat"
    @test UniLM.get_url(e, Embeddings("x"; service=e, model="embeddinggemma", task=:none)) == "http://box:11434/api/embed"
    @test UniLM.get_url(e, FIMCompletion(service=e, model="qwen2.5-coder:1.5b-base", prompt="x")) ==
          "http://box:11434/v1/completions"
    @test UniLM._agentic_url(e) == "http://box:11434/v1/responses"
    @test UniLM.auth_header(e) == ["Content-Type" => "application/json"]
end

@testset "Ollama's cloud takes OLLAMA_API_KEY as a Bearer token; without it nothing is sent" begin
    # "Direct cloud inference at https://ollama.com/api and https://ollama.com/v1 requires an API key", sent in
    # the Authorization: Bearer header (https://docs.ollama.com/api/authentication). No request reaches the
    # cloud here: the header is what auth_header gives the transport, and every call below runs under a
    # deadline that ends it at the transport, before any connection.
    cloud = withenv(OllamaEndpoint, "OLLAMA_HOST" => "ollama.com")
    @test cloud.base_url == "https://ollama.com:443"
    json = "Content-Type" => "application/json"
    withenv("OLLAMA_API_KEY" => "test-key") do
        for ep in (cloud, OllamaEndpoint(base_url="https://ollama.com"), OllamaEndpoint(base_url="https://Ollama.COM/"))
            @test UniLM.auth_header(ep) == ["Authorization" => "Bearer test-key", json]
        end
        @test UniLM.auth_header_multipart(cloud) == ["Authorization" => "Bearer test-key"]   # transcription
        @test UniLM._stream_headers(cloud) == ["Authorization" => "Bearer test-key", json,  # a streamed chat
                                               "Accept-Encoding" => "identity"]
        # Any other server gets no key, the variable set or not: it is the cloud's credential.
        for url in ("http://127.0.0.1:11434", "https://gpu-box:11434", "https://ollama.com.example.net",
                    "https://api.ollama.com", "http://ollama.com")
            @test UniLM.auth_header(OllamaEndpoint(base_url=url)) == [json]
        end
    end
    # The key is read when a request is built: unset or blank, the request is refused, naming the variable.
    for unset in (nothing, "", "  ")
        withenv("OLLAMA_API_KEY" => unset) do
            for headers in (UniLM.auth_header, UniLM.auth_header_multipart, UniLM._stream_headers)
                e = try headers(cloud); nothing catch x; x end
                @test e isa ArgumentError && occursin("OLLAMA_API_KEY", e.msg)
            end
        end
    end
    # Through the verbs: without the key the call error's cause is that ArgumentError, before the transport is
    # reached; with it the request gets as far as the transport, where the deadline ends it unsent.
    tiny = RequestConfig(total_deadline=1e-300)
    calls = (() -> chatrequest!(Chat(service=cloud, model="gemma4:31b", messages=[Message(Val(:user), "hi")]); config=tiny),
             () -> respond(Respond(service=cloud, model="gemma4:31b", input="hi"); config=tiny),
             () -> embeddingrequest!(Embeddings("hi"; service=cloud, model="embeddinggemma", task=:none); config=tiny),
             () -> list_models(service=cloud, config=tiny))
    for call in calls
        r = withenv(call, "OLLAMA_API_KEY" => nothing)
        @test !issuccess(r) && r.cause isa ArgumentError && occursin("OLLAMA_API_KEY", r.cause.msg)
        r = withenv(call, "OLLAMA_API_KEY" => "test-key")
        @test !issuccess(r) && r.cause isa UniLMTimeout && r.cause.phase === :deadline
    end
    # A server of the caller's own sees no Authorization header, the variable set or not.
    auth = String[]
    srv = HTTP.serve!("127.0.0.1", 0; listenany=true, verbose=false) do req
        push!(auth, HTTP.header(req, "Authorization", "<none>"))
        HTTP.Response(200, ["Content-Type" => "application/json"], Vector{UInt8}(_reply(content="ok")))
    end
    try
        mine = OllamaEndpoint(base_url="http://127.0.0.1:$(HTTP.port(srv))")
        r = withenv(() -> chatrequest!(_ochat(mine)), "OLLAMA_API_KEY" => "test-key")
        @test r isa LLMSuccess && text(r) == "ok"
    finally
        close(srv)
    end
    @test auth == ["<none>"]
end

@testset "request: fields map or are refused" begin
    e = OllamaEndpoint(num_ctx=8192, keep_alive=Inf, top_k=64)
    w = _wire(_ochat(e; temperature=0.3, top_p=nothing, seed=7, max_completion_tokens=99, stop="END",
                     presence_penalty=0.1, frequency_penalty=0.2))
    @test w["model"] == "gemma4:e2b"
    @test w["stream"] === false                  # /api/chat streams unless told not to
    @test w["keep_alive"] == -1                  # Inf = stay loaded
    @test w["truncate"] === false                # an over-long prompt is an error, not a shortened prompt
    @test !haskey(w, "shift")
    @test _wire(_ochat(OllamaEndpoint(truncate=true, shift=false)))["truncate"] === true
    @test _wire(_ochat(OllamaEndpoint(shift=false)))["shift"] === false
    @test w["options"] == Dict("num_ctx" => 8192, "top_k" => 64, "temperature" => 0.3, "seed" => 7,
                               "num_predict" => 99, "stop" => ["END"], "presence_penalty" => 0.1,
                               "frequency_penalty" => 0.2)
    @test w["messages"] == [Dict("role" => "system", "content" => "Be brief."), Dict("role" => "user", "content" => "Hi")]
    @test !haskey(w, "think") && !haskey(w, "format") && !haskey(w, "tools")
    @test _wire(_ochat(OllamaEndpoint(); max_tokens=5))["options"] == Dict("num_predict" => 5)
    @test !haskey(_wire(_ochat(OllamaEndpoint())), "options")
    @test !haskey(_wire(_ochat(OllamaEndpoint())), "keep_alive")
    @test _wire(_ochat(OllamaEndpoint(keep_alive=0)))["keep_alive"] == 0
    @test _wire(_ochat(OllamaEndpoint(keep_alive=90.5)))["keep_alive"] == 90.5
    @test _wire(_ochat(OllamaEndpoint(); stream=true))["stream"] === true
    # Every Chat field without an /api/chat counterpart is refused before any I/O.
    for (f, v) in (:logit_bias => Dict("1" => 1), :user => "u", :stream_options => Dict("include_usage" => true),
                   :verbosity => "low", :store => true, :metadata => Dict("a" => "b"), :service_tier => "auto",
                   :logprobs => true, :top_logprobs => 2, :prediction => Dict("type" => "content"),
                   :modalities => ["text"], :audio => Dict("voice" => "x"), :web_search_options => Dict(),
                   :prompt_cache_key => "k", :safety_identifier => "s",
                   :prompt_cache_options => PromptCacheOptions(mode="implicit"),
                   :moderation => ModerationConfig(model="omni-moderation-latest"))
        err = try UniLM.encode_request(e, _ochat(e; f => v)); nothing catch x; x end
        @test err isa ArgumentError && occursin(String(f), err.msg)
    end
    @test Set(UniLM._OLLAMA_CHAT_UNMAPPED_FIELDS) == Set([:logit_bias, :user, :stream_options, :verbosity, :store,
        :metadata, :service_tier, :logprobs, :top_logprobs, :prediction, :modalities, :audio, :web_search_options,
        :prompt_cache_key, :safety_identifier, :prompt_cache_options, :moderation])
    @test_throws ArgumentError chatrequest!(_ochat(e; user="u"))     # thrown before any network I/O
end

@testset "request: thinking, format, tools" begin
    e = OllamaEndpoint()
    @test _wire(_ochat(e; reasoning_effort="none"))["think"] === false
    # Every other effort goes by name: Ollama resolves a level against the model, which applies one it
    # defines and uses its default for another; "max" is among the levels of Ollama's API reference.
    for lvl in ("minimal", "low", "medium", "high", "xhigh", "max")
        @test _wire(_ochat(e; reasoning_effort=lvl))["think"] == lvl
    end
    schema = Dict("type" => "object", "properties" => Dict("name" => Dict("type" => "string")), "required" => ["name"])
    w = _wire(_ochat(e; response_format=UniLM.json_schema("person", "p", schema)))
    @test w["format"] == schema
    # Ollama constrains a reply only after thinking ends: a format turns thinking off.
    @test w["think"] === false
    @test _wire(_ochat(e; response_format=ResponseFormat("json_schema", Dict("name" => "p", "schema" => schema))))["format"] == schema
    wo = _wire(_ochat(e; response_format=UniLM.json_object()))
    @test wo["format"] == Dict("type" => "object") && wo["think"] === false
    @test _wire(_ochat(e; response_format=UniLM.json_object(), reasoning_effort="none"))["think"] === false
    for lvl in ("minimal", "low", "medium", "high", "xhigh", "max")   # a thinking model may skip thinking: unconstrained
        err = try _wire(_ochat(e; response_format=UniLM.json_object(), reasoning_effort=lvl)); nothing catch x; x end
        @test err isa ArgumentError && occursin("response_format", err.msg)
    end
    wt = _wire(_ochat(e; response_format=ResponseFormat(type="text")))
    @test !haskey(wt, "format") && !haskey(wt, "think")
    @test_throws ArgumentError _wire(_ochat(e; response_format=ResponseFormat(type="xml")))
    w = _wire(_ochat(e; tools=[_WEATHER]))
    @test w["tools"] == [Dict("type" => "function", "function" => Dict("name" => "get_weather",
        "description" => "Weather in a city", "parameters" => _WEATHER.func.parameters))]
    @test !haskey(w, "parallel_tool_calls")      # no such switch on /api/chat; the default false is not sent
    @test haskey(_wire(_ochat(e; tools=[_WEATHER], tool_choice="auto")), "tools")
    for tc in ("none", "required", UniLM.GPTToolChoice(func="get_weather"))
        @test_throws ArgumentError _wire(_ochat(e; tools=[_WEATHER], tool_choice=tc))
    end
    strict = Tool(func=FunctionSignature(name="f", parameters=Dict("type" => "object"), strict=true))
    @test_throws ArgumentError _wire(_ochat(e; tools=[strict]))
end

@testset "request: messages" begin
    e = OllamaEndpoint()
    png = UInt8[0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 0x00]
    wav = Vector{UInt8}(b"RIFF\x24\x00\x00\x00WAVEfmt ")
    c = Chat(service=e, model="gemma4:e2b")
    push!(c, Message(Val(:system), "sys"))
    push!(c, Message(Val(:user), "Look and listen", ImageAttachment(png), AudioAttachment(wav)))
    call = ToolCall(id="call_1", func=UniLM.GPTFunction("get_weather", Dict{String,Any}("city" => "Paris")))
    push!(c, Message(role=RoleAssistant, content=nothing, tool_calls=[call],
                     provider_content=ProviderContent(:ollama, Any[Dict{String,Any}("thinking" => "Need the weather.")])))
    push!(c, Message(role=UniLM.RoleTool, content="18°C", tool_call_id="call_1"))
    push!(c, Message(role=RoleAssistant, content="It is 18°C.",
                     provider_content=ProviderContent(:anthropic, Any[Dict{String,Any}("type" => "thinking", "thinking" => "x")])))
    push!(c, Message(Val(:user), "Thanks"))
    m = _wire(c)["messages"]
    @test m[2] == Dict("role" => "user", "content" => "Look and listen",
                       "images" => [UniLM.Base64.base64encode(png), UniLM.Base64.base64encode(wav)])
    @test m[3] == Dict("role" => "assistant", "content" => "", "thinking" => "Need the weather.",
                       "tool_calls" => [Dict("id" => "call_1",
                                             "function" => Dict("name" => "get_weather", "arguments" => Dict("city" => "Paris")))])
    @test m[4] == Dict("role" => "tool", "content" => "18°C", "tool_name" => "get_weather", "tool_call_id" => "call_1")
    # A synthetic id (the reply named no call) never reaches the wire; the name matches the result.
    syn = Chat(service=e, model="m")
    push!(syn, Message(Val(:system), "s")); push!(syn, Message(Val(:user), "u"))
    push!(syn, Message(role=RoleAssistant, tool_calls=[ToolCall(id="unilm_call_1", func=UniLM.GPTFunction("f", Dict{String,Any}()))]))
    push!(syn, Message(role=UniLM.RoleTool, content="r", tool_call_id="unilm_call_1"))
    sm = _wire(syn)["messages"]
    @test sm[3]["tool_calls"] == [Dict("function" => Dict("name" => "f", "arguments" => Dict()))]
    @test sm[4] == Dict("role" => "tool", "content" => "r", "tool_name" => "f")
    @test m[5] == Dict("role" => "assistant", "content" => "It is 18°C.")    # another provider's thinking stays behind
    # A tool result must answer a call the conversation made; a name has no field here.
    orphan = Chat(service=e, model="m")
    push!(orphan, Message(Val(:system), "s")); push!(orphan, Message(Val(:user), "u"))
    push!(orphan, Message(role=RoleAssistant, content="a"))
    push!(orphan, Message(role=UniLM.RoleTool, content="r", tool_call_id="nope"))
    @test_throws ArgumentError _wire(orphan)
    named = Chat(service=e, model="m")
    push!(named, Message(Val(:system), "s")); push!(named, Message(role=RoleUser, content="u", name="ann"))
    @test_throws ArgumentError _wire(named)
    # A refusal from another provider travels as the turn's text.
    refused = Chat(service=e, model="m")
    push!(refused, Message(Val(:system), "s")); push!(refused, Message(Val(:user), "u"))
    push!(refused, Message(role=RoleAssistant, refusal_message="I can't help with that."))
    push!(refused, Message(Val(:user), "ok"))
    @test _wire(refused)["messages"][3] == Dict("role" => "assistant", "content" => "I can't help with that.")
end

@testset "reply decoding" begin
    e = OllamaEndpoint()
    dec(s) = UniLM.decode_response(e, HTTP.Response(200, [], Vector{UInt8}(s)))
    r = dec(_reply(content="No", thinking="391 = 17 × 23.", p=26, c=444))
    @test r.message.content == "No"
    @test r.message.finish_reason == "stop"
    @test reasoning_text(r.message) == "391 = 17 × 23."
    @test r.usage == TokenUsage(prompt_tokens=26, completion_tokens=444, total_tokens=470)
    r = dec(_reply(content="Hi"))
    @test isnothing(r.message.provider_content) && isnothing(reasoning_text(r.message))
    r = dec(_reply(content="", reason="length"))
    @test r.message.content == "" && r.message.finish_reason == "length"
    calls = [Dict("function" => Dict("index" => 0, "name" => "get_weather", "arguments" => Dict("city" => "Paris"))),
             Dict("id" => "call_x", "function" => Dict("index" => 1, "name" => "get_weather", "arguments" => Dict("city" => "Tokyo")))]
    r = dec(_reply(tool_calls=calls, thinking="two calls"))
    @test [tc.id for tc in r.message.tool_calls] == ["unilm_call_1", "call_x"]
    @test [tc.func.arguments["city"] for tc in r.message.tool_calls] == ["Paris", "Tokyo"]
    @test r.message.finish_reason == "tool_calls" && isnothing(r.message.content)
    # String arguments (some servers) parse; anything else is not a call.
    r = dec(_reply(tool_calls=[Dict("function" => Dict("name" => "f", "arguments" => "{\"a\":1}"))]))
    @test r.message.tool_calls[1].func.arguments == Dict("a" => 1)
    @test_throws Exception dec(_reply(tool_calls=[Dict("function" => Dict("name" => "f", "arguments" => [1]))]))
    @test_throws Exception dec("{\"done\":true}")                     # no message: not a reply
    @test_throws Exception dec("[1]")
    @test isnothing(dec(JSON.json(Dict("message" => Dict("role" => "assistant", "content" => "x")))).usage)
end

@testset "NDJSON stream: reads split anywhere assemble the same turn" begin
    e = OllamaEndpoint()
    lines = join([_delta(thinking="Let me "), _delta(thinking="think."), _delta(content="Hel"), _delta(content="lo ✓"),
                  _delta(tool_calls=[Dict("function" => Dict("index" => 0, "name" => "get_weather", "arguments" => Dict("city" => "Paris")))]),
                  _final(p=30, c=9)])
    bytes = codeunits(lines)
    for step in (1, 2, 3, 7, 64, length(bytes))
        st = UniLM.StreamState(); carry = IOBuffer(); ev = Ref(""); status = :continue
        for i in 1:step:length(bytes)
            status = UniLM._sse_dispatch!(e, carry, ev, String(bytes[i:min(i + step - 1, end)]), st)
            status === :continue || break
        end
        @test status === :done
        @test st.sse_dropped == 0
        @test st.usage == TokenUsage(prompt_tokens=30, completion_tokens=9, total_tokens=39)
        m = UniLM._build_stream_message(st)
        @test m.content == "Hello ✓"
        @test reasoning_text(m) == "Let me think."
        @test m.finish_reason == "tool_calls"
        @test only(m.tool_calls).func.arguments == Dict("city" => "Paris")
        @test only(m.tool_calls).id == "unilm_call_1"
    end
    # A stream cut before `done` carries no partial thinking.
    st = UniLM.StreamState()
    UniLM._sse_dispatch!(e, IOBuffer(), Ref(""), _delta(thinking="half"), st)
    @test isnothing(UniLM._build_stream_message(st).provider_content)
    # An undecodable line is dropped and counted, never re-queued.
    st = UniLM.StreamState()
    @test UniLM._sse_dispatch!(e, IOBuffer(), Ref(""), "{not json}\n" * _final(), st) === :done
    @test st.sse_dropped == 1
    # {"error": …} on the stream is terminal.
    st = UniLM.StreamState()
    @test UniLM._sse_dispatch!(e, IOBuffer(), Ref(""), _oline(Dict("error" => "model runner crashed")), st) === :error
    @test st.error == Dict("error" => "model runner crashed")
end

@testset "driver: non-streaming and streaming calls" begin
    server, base, bodies = _ollama_mock() do n, body
        req = JSON.parse(body)
        if req["stream"] === true
            200, [_delta(thinking="Think."), _delta(content="Hel"), _delta(content="lo"), _final(p=5, c=3)]
        else
            200, [_reply(content="Hello", thinking="Think.", p=5, c=3)]
        end
    end
    try
        e = OllamaEndpoint(base_url=base, num_ctx=4096)
        c = _ochat(e)
        r = chatrequest!(c)
        @test r isa LLMSuccess
        @test text(r) == "Hello" && reasoning_text(r) == "Think."
        @test last(c).content == "Hello"                                  # committed to history
        @test JSON.parse(bodies[1])["options"] == Dict("num_ctx" => 4096)
        @test estimated_cost(r) == 0.0 && cumulative_cost(c) == 0.0
        # The captured thinking goes back with the turn.
        push!(c, Message(Val(:user), "Again"))
        deltas = String[]
        t = chatrequest!(Chat(service=e, model="gemma4:e2b", stream=true, messages=copy(c.messages));
                         callback=(x, _) -> x isa String && push!(deltas, x))
        s = fetch(t)
        @test s isa LLMSuccess
        @test deltas == ["Hel", "lo"]
        @test text(s) == "Hello" && reasoning_text(s) == "Think."
        @test s.usage == TokenUsage(prompt_tokens=5, completion_tokens=3, total_tokens=8)
        @test JSON.parse(bodies[2])["messages"][3] == Dict("role" => "assistant", "content" => "Hello", "thinking" => "Think.")
    finally
        close(server)
    end
end

@testset "driver: failures keep their status" begin
    server, base, _ = _ollama_mock() do n, body
        n == 1 && return 404, [JSON.json(Dict("error" => "model 'gemma4:nope' not found"))]
        n == 2 && return 404, [JSON.json(Dict("error" => "model 'gemma4:nope' not found"))]
        n == 3 && return 400, [JSON.json(Dict("error" => "invalid format"))]
        200, [_delta(content="partial"), _oline(Dict("error" => "runner process has terminated"))]
    end
    try
        e = OllamaEndpoint(base_url=base)
        cfg = RequestConfig(max_attempts=1)
        r = chatrequest!(_ochat(e); config=cfg)
        @test r isa LLMFailure && r.status == 404 && occursin("not found", r.response)
        s = fetch(chatrequest!(_ochat(e; stream=true); config=cfg))
        @test s isa LLMFailure && s.status == 404 && occursin("not found", s.response)   # an error body, not an in-band error
        s = fetch(chatrequest!(_ochat(e; stream=true); config=cfg))
        @test s isa LLMFailure && s.status == 400
        s = fetch(chatrequest!(_ochat(e; stream=true); config=cfg))
        @test s isa LLMCallError && occursin("runner process has terminated", s.error)   # failure on a 200 stream
    finally
        close(server)
    end
end

@testset "unreachable server: the error says how to fix it" begin
    tcp = Sockets.listen(Sockets.localhost, 0)
    port = Int(Sockets.getsockname(tcp)[2]); close(tcp)                       # nothing listens here now
    e = OllamaEndpoint(base_url="http://127.0.0.1:$port")
    cfg = RequestConfig(max_attempts=1)
    for r in (chatrequest!(_ochat(e); config=cfg), fetch(chatrequest!(_ochat(e; stream=true); config=cfg)),
              embeddingrequest!(Embeddings("x"; service=e, model="embeddinggemma", task=:none); config=cfg))
        @test occursin("no Ollama server answered at http://127.0.0.1:$port", r.error)
        @test occursin("ollama serve", r.error)
    end
    g = chatrequest!(_ochat(GenericOpenAIEndpoint("http://127.0.0.1:$port", "")); config=cfg)
    @test !occursin("ollama serve", g.error)                                  # only Ollama knows the remedy
end

@testset "local models cost nothing, silently" begin
    e = OllamaEndpoint()
    c = _ochat(e)
    r = LLMSuccess(message=Message(role=RoleAssistant, content="x"), self=c,
                   usage=TokenUsage(prompt_tokens=10^6, completion_tokens=10^6, total_tokens=2 * 10^6))
    @test (@test_logs estimated_cost(r)) == 0.0
    # An explicit model still prices: what would this have cost elsewhere?
    @test estimated_cost(r; model="gpt-5.4-mini") > 0.0
    emb = EmbeddingSuccess(embeddings=Embeddings("x"; service=e, model="embeddinggemma", task=:none),
                           usage=TokenUsage(prompt_tokens=100, total_tokens=100), raw=Dict{String,Any}())
    @test (@test_logs estimated_cost(emb)) == 0.0
end

@testset "record and replay the native wire" begin
    server, base, bodies = _ollama_mock((n, b) -> (200, [_reply(content="Recorded", thinking="t")]))
    dir = mktempdir()
    try
        e = OllamaEndpoint(base_url=base)
        with_recorded_answers(dir; mode=:record) do
            @test text(chatrequest!(_ochat(e))) == "Recorded"
        end
        @test length(bodies) == 1
    finally
        close(server)
    end
    # Replayed with no server at all: the recording answers, thinking included.
    with_recorded_answers(dir; mode=:replay) do
        r = chatrequest!(_ochat(OllamaEndpoint(base_url="http://127.0.0.1:1")))
        @test text(r) == "Recorded" && reasoning_text(r) == "t"
    end
end

@testset "respond: fields Ollama's stateless Responses API ignores are refused" begin
    e = OllamaEndpoint()
    enc(; kw...) = JSON.parse(UniLM.encode_agentic(e, Respond(; service=e, model="gemma4:e2b", input="Hi", kw...)))
    w = enc(instructions="Be brief.", temperature=0.5, max_output_tokens=50, reasoning=Reasoning(effort="none"),
            text=TextConfig(format=TextFormatSpec(type="json_schema", name="x", schema=Dict("type" => "object"))),
            input=[InputMessage(role="user", content=[input_text("What is it?"),
                                                       input_image("data:image/png;base64,iVBORw0KGgo=")])])
    @test w["instructions"] == "Be brief." && w["reasoning"]["effort"] == "none"
    @test w["input"][1]["content"][2]["image_url"] == "data:image/png;base64,iVBORw0KGgo="
    for (f, v) in (:previous_response_id => "resp_1", :conversation => "conv_1", :store => true,
                   :metadata => Dict("a" => "b"), :truncation => "auto", :background => true,
                   :include => ["reasoning.encrypted_content"], :parallel_tool_calls => true, :user => "u",
                   :max_tool_calls => 2, :service_tier => "auto", :top_logprobs => 2, :safety_identifier => "s",
                   :prompt_cache_key => "k", :prompt_cache_options => PromptCacheOptions(mode="implicit"))
        err = try enc(; f => v); nothing catch x; x end
        @test err isa ArgumentError && occursin(String(f), err.msg)
    end
    # A field that asks the server to keep a response, or to read one it kept, is refused because
    # Ollama keeps none; any other because Ollama reads no such field.
    said(; kw...) = try enc(; kw...); "" catch x; x.msg end
    @test said(store=true) == "Ollama's Responses API ignores store: it keeps no state between calls"
    @test said(previous_response_id="r", background=true) == "Ollama's Responses API ignores previous_response_id, " *
        "background: it keeps no state between calls; send the whole conversation in `input`, or use a Chat"
    @test said(service_tier="auto") == "Ollama's Responses API reads no service_tier (its OpenAI-compatibility " *
        "page does not list it for /v1/responses), so it would be dropped: unset it"
    @test said(top_logprobs=2, include=["reasoning.encrypted_content"]) == "Ollama's Responses API reads none of " *
        "include, top_logprobs (its OpenAI-compatibility page does not list them for /v1/responses), so they would " *
        "be dropped: unset them"
    @test said(store=true, service_tier="auto") == said(store=true)      # the state field is named first
    @test_throws ArgumentError enc(tool_choice="required")
    @test haskey(enc(tool_choice="auto"), "model")
    # Every effort goes as reasoning.effort: /v1/responses reads model-defined names and maps "minimal" to
    # "low" and "xhigh" to "max" for a model without thinking metadata.
    for lvl in ("minimal", "low", "medium", "high", "xhigh", "max")
        @test enc(reasoning=Reasoning(effort=lvl))["reasoning"] == Dict("effort" => lvl)
    end
    @test_throws ArgumentError enc(reasoning=Reasoning(effort="low", summary="auto"))
    @test_throws ArgumentError enc(text=TextConfig(format=TextFormatSpec(type="json_object")))
    @test_throws ArgumentError enc(text=TextConfig(verbosity="low"))
    @test_throws ArgumentError enc(input=[InputMessage(role="user", content=[input_image(file_id="file_1")])])
    # A replayed message's phase has no field on Ollama's Responses API.
    err = try enc(input=[InputMessage(role="user", content="q"),
                         InputMessage(role="assistant", content="a", phase="commentary")]); nothing catch x; x end
    @test err isa ArgumentError && occursin("phase", err.msg) && occursin("Ollama", err.msg)
    @test haskey(enc(input=[InputMessage(role="user", content="q"), InputMessage(role="assistant", content="a")]), "model")
    # A seed is refused for what it is, not as state: /v1/responses reads none, Ollama's chat API does.
    err = try enc(seed=7); nothing catch x; x end
    @test err isa ArgumentError && err.msg == "Ollama's Responses API reads no seed — its OpenAI-compatibility " *
        "page lists none for /v1/responses — so the seed would be dropped: unset seed, or send the request as a " *
        "Chat, whose seed Ollama reads"
    # respond validates before any I/O: the refusal is a throw, not a result.
    @test_throws ArgumentError respond(Respond(service=e, model="m", input="x", store=true))
    @test_throws ArgumentError respond(Respond(service=e, model="m", input="x", seed=7))
    # A Respond tool loop cannot chain turns without server-side state.
    err = try tool_loop(Respond(service=e, model="m", input="x"), (n, a) -> "r"); nothing catch x; x end
    @test err isa ArgumentError && occursin("tool_loop!", err.msg)
end

@testset "embeddings: native /api/embed, over-long input is an error" begin
    e = OllamaEndpoint(keep_alive=60, num_ctx=2048)
    body = JSON.parse(UniLM._encode_embeddings(e, Embeddings(["a", "b"]; service=e, model="embeddinggemma", dimensions=256, task=:none)))
    @test body == Dict("model" => "embeddinggemma", "input" => ["a", "b"], "truncate" => false, "dimensions" => 256,
                       "keep_alive" => 60, "options" => Dict("num_ctx" => 2048))
    @test JSON.parse(UniLM._encode_embeddings(OllamaEndpoint(truncate=true), Embeddings("a"; service=e, model="m")))["truncate"] === true
    @test_throws ArgumentError embeddingrequest!(Embeddings("a"; service=e, model="m", user="u"))   # before any I/O
    server, base, bodies = _ollama_mock() do n, b
        req = JSON.parse(b)
        n == 1 && return 200, [JSON.json(Dict("model" => "embeddinggemma", "embeddings" => [[0.1, 0.2], [0.3, 0.4]],
                                               "prompt_eval_count" => 4))]
        n == 2 && return 200, [JSON.json(Dict("model" => "embeddinggemma", "embeddings" => [[1.0, 0.0, 0.0]],
                                               "prompt_eval_count" => 2))]
        400, [JSON.json(Dict("error" => "the input length exceeds the context length"))]
    end
    try
        ep = OllamaEndpoint(base_url=base)
        r = embeddingrequest!(Embeddings(["a", "b"]; service=ep, model="embeddinggemma", task=:none))
        @test r isa EmbeddingSuccess
        @test embedding_vectors(r) == [[0.1, 0.2], [0.3, 0.4]]
        @test r.usage == TokenUsage(prompt_tokens=4, total_tokens=4)
        @test (@test_logs estimated_cost(r)) == 0.0
        r1 = embeddingrequest!(Embeddings("a"; service=ep, model="embeddinggemma", task=:none))
        @test embedding_vectors(r1) == [1.0, 0.0, 0.0]                      # the model's own dimension
        f = embeddingrequest!(Embeddings("x"^10_000; service=ep, model="embeddinggemma", task=:none); config=RequestConfig(max_attempts=1))
        @test f isa EmbeddingFailure && f.status == 400 && occursin("exceeds the context length", f.response)
        @test all(b -> JSON.parse(b)["truncate"] === false, bodies)
    finally
        close(server)
    end
end

@testset "respond: a format turns thinking off; tools Ollama cannot honour are refused" begin
    e = OllamaEndpoint()
    schema = Dict("type" => "object", "properties" => Dict("name" => Dict("type" => "string")), "required" => ["name"])
    fmt = TextConfig(format=TextFormatSpec(type="json_schema", name="person", schema=schema))
    enc(; kw...) = JSON.parse(UniLM.encode_agentic(e, Respond(; service=e, model="gemma4:e4b", input="x", kw...)))
    # Refuted by: a format request that leaves Gemma 4 free to skip thinking and answer unconstrained.
    @test enc(text=fmt)["reasoning"] == Dict("effort" => "none")
    @test enc(text=fmt, reasoning=Reasoning(effort="none"))["reasoning"]["effort"] == "none"
    @test !haskey(enc(), "reasoning")                                   # no format: the model's default
    @test enc(reasoning=Reasoning(effort="high"))["reasoning"]["effort"] == "high"
    for lvl in ("minimal", "low", "medium", "high", "xhigh", "max")
        err = try enc(text=fmt, reasoning=Reasoning(effort=lvl)); nothing catch x; x end
        @test err isa ArgumentError && occursin("response_format", err.msg)
    end
    @test_throws ArgumentError enc(text=TextConfig(format=TextFormatSpec(type="json_schema", name="p")))  # no schema
    ft(; kw...) = FunctionTool(; name="f", parameters=Dict("type" => "object"), kw...)
    @test only(enc(tools=[ft()])["tools"])["type"] == "function"
    @test only(enc(tools=[Dict("type" => "function", "name" => "g", "parameters" => Dict())])["tools"])["name"] == "g"
    for bad in (ft(strict=true), ft(async=true), ft(allowed_callers=["direct"]), ft(defer_loading=true),
                ft(output_schema=Dict("type" => "object")), web_search(), Dict("type" => "file_search"))
        @test_throws ArgumentError enc(tools=[bad])
    end
end

@testset "a user message may carry media alone" begin
    img = ImageAttachment(UInt8[0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a])
    m = Message(role=RoleUser, attachments=[img])
    @test isnothing(m.content)
    @test [p[:type] for p in JSON.lower(m)[:content]] == ["image_url"]
    c = Chat(service=OllamaEndpoint(), model="m"); push!(c, Message(Val(:system), "s")); push!(c, m)
    @test _wire(c)["messages"][2] == Dict("role" => "user", "content" => "", "images" => [UniLM.Base64.base64encode(img.data)])
    @test_throws ArgumentError Message(role=RoleUser)                                  # nothing at all
    @test_throws ArgumentError Message(role=RoleUser, attachments=Attachment[])        # an empty list is nothing
end

@testset "runner: validated, shown, sent on the native routes" begin
    @test OllamaEndpoint().runner === nothing
    e = OllamaEndpoint(base_url="http://box:11434", runner=:llamacpp)
    @test e.runner === :llamacpp
    @test occursin("runner=:llamacpp", sprint(show, e))
    err = try OllamaEndpoint(runner=:cuda); nothing catch x; x end
    @test err isa ArgumentError && occursin(":mlx", err.msg) && occursin(":cuda", err.msg)
    @test _wire(_ochat(e))["runner"] == "llamacpp"
    @test !haskey(_wire(_ochat(OllamaEndpoint(base_url="http://box:11434"))), "runner")
    b = JSON.parse(UniLM._encode_embeddings(e, Embeddings("x"; service=e, model="nomic-embed-text")))
    @test b["runner"] == "llamacpp"
end

@testset "a local respond or FIM result is free, without a price warning" begin
    resp = JSON.json(Dict("id" => "resp_1", "object" => "response", "status" => "completed", "model" => "gemma4:e4b",
        "output" => [Dict("type" => "message", "role" => "assistant",
                          "content" => [Dict("type" => "output_text", "text" => "Bonjour")])],
        "usage" => Dict("input_tokens" => 5, "output_tokens" => 1, "total_tokens" => 6)))
    fim = JSON.json(Dict("id" => "cmpl-1", "object" => "text_completion", "model" => "qwen2.5-coder:1.5b-base",
        "choices" => [Dict("text" => "a + b", "index" => 0, "finish_reason" => "stop")],
        "usage" => Dict("prompt_tokens" => 10, "completion_tokens" => 3, "total_tokens" => 13)))
    server, url, bodies = _ollama_mock((n, body) -> (200, [occursin("\"prompt\"", body) ? fim : resp]))
    try
        ep = OllamaEndpoint(base_url=url)
        r = respond(Respond(service=ep, model="gemma4:e4b", input="Say hello in French.", reasoning=Reasoning(effort="none")))
        @test r isa ResponseSuccess && r.service === ep
        @test (@test_logs estimated_cost(r)) == 0.0
        f = fim_complete(FIMCompletion(service=ep, model="qwen2.5-coder:1.5b-base", prompt="def add(a, b):\n    return ",
                                       suffix="\n", max_tokens=8))
        @test f isa FIMSuccess && f.service === ep && (@test_logs estimated_cost(f)) == 0.0
        # The OpenAI-compatible routes take no runner: it rides only on the native ones.
        er = OllamaEndpoint(base_url=url, runner=:llamacpp)
        n = length(bodies)
        @test respond(Respond(service=er, model="gemma4:e4b", input="Say hello in French.",
                              reasoning=Reasoning(effort="none"))) isa ResponseSuccess
        @test fim_complete(FIMCompletion(service=er, model="qwen2.5-coder:1.5b-base", prompt="def add(a, b):\n    return ",
                                         suffix="\n", max_tokens=8)) isa FIMSuccess
        sent = [JSON.parse(b) for b in bodies[n+1:end]]
        @test length(sent) == 2 && haskey(sent[1], "input") && haskey(sent[2], "prompt")
        @test !any(b -> haskey(b, "runner"), sent)
    finally
        close(server)
    end
end

# Refuted if: a cloud model's respond or FIM result is priced 0.0 as a local call, or without the missing-price
# warning, because its reply drops the tag. Ollama's cloud proxy sends the cloud the name without its cloud tag
# and passes the reply through (server/cloud_proxy.go of Ollama 0.40.6), so the reply names `gpt-oss:120b`.
@testset "a cloud model keeps its requested name on respond and FIM results, so it is priced" begin
    obj = Dict("id" => "resp_1", "object" => "response", "status" => "completed", "model" => "gpt-oss:120b",
        "output" => [Dict("type" => "message", "role" => "assistant",
                          "content" => [Dict("type" => "output_text", "text" => "Bonjour")])],
        "usage" => Dict("input_tokens" => 5, "output_tokens" => 1, "total_tokens" => 6))
    fim = JSON.json(Dict("id" => "cmpl-1", "object" => "text_completion", "model" => "gpt-oss:120b",
        "choices" => [Dict("text" => "a + b", "index" => 0, "finish_reason" => "stop")],
        "usage" => Dict("prompt_tokens" => 10, "completion_tokens" => 3, "total_tokens" => 13)))
    sse(type, d) = "event: $type\ndata: $(JSON.json(merge(Dict{String,Any}("type" => type), d)))\n\n"
    server, url, _ = _ollama_mock((n, body) -> occursin("\"prompt\"", body) ? (200, [fim]) :
        occursin("\"stream\":true", body) ? (200, [sse("response.completed", Dict("response" => obj))]) :
        (200, [JSON.json(obj)]))
    try
        ep = OllamaEndpoint(base_url=url)
        table = Dict("gpt-oss:120b-cloud" => UniLM._price(1.0, 0.1, 2.0))
        r = respond(Respond(service=ep, model="gpt-oss:120b-cloud", input="Say hello in French."))
        @test r isa ResponseSuccess && r.response.model == "gpt-oss:120b-cloud" && r.response.raw["model"] == "gpt-oss:120b"
        @test (@test_logs (:warn, r"no price row") estimated_cost(r)) == 0.0     # unpriced: warned, never free as local
        @test (@test_logs estimated_cost(r; pricing=table)) ≈ 5 * 1e-6 + 1 * 2e-6
        s = fetch(respond(Respond(service=ep, model="gpt-oss:120b-cloud", input="Say hello in French.", stream=true)))
        @test s isa ResponseSuccess && s.response.model == "gpt-oss:120b-cloud" && s.response.raw["model"] == "gpt-oss:120b"
        @test (@test_logs estimated_cost(s; pricing=table)) ≈ 5 * 1e-6 + 1 * 2e-6
        f = fim_complete(FIMCompletion(service=ep, model="gpt-oss:120b-cloud", prompt="def add(a, b):\n    return ",
                                       suffix="\n", max_tokens=8))
        @test f isa FIMSuccess && f.response.model == "gpt-oss:120b-cloud" && f.response.raw["model"] == "gpt-oss:120b"
        @test (@test_logs (:warn, r"no price row") estimated_cost(f)) == 0.0
        @test (@test_logs estimated_cost(f; pricing=table)) ≈ 10 * 1e-6 + 3 * 2e-6
        # A local model answered under another name keeps the server's name, and stays free.
        l = respond(Respond(service=ep, model="gpt-oss", input="Say hello in French."))
        @test l isa ResponseSuccess && l.response.model == "gpt-oss:120b" && (@test_logs estimated_cost(l)) == 0.0
    finally
        close(server)
    end
end

@testset "a streamed respond and a fetched response record their endpoint" begin
    obj = Dict("id" => "resp_1", "object" => "response", "status" => "completed", "model" => "gemma4:e4b",
        "output" => [Dict("type" => "message", "role" => "assistant",
                          "content" => [Dict("type" => "output_text", "text" => "Bonjour")])],
        "usage" => Dict("input_tokens" => 5, "output_tokens" => 1, "total_tokens" => 6))
    sse(type, d) = "event: $type\ndata: $(JSON.json(merge(Dict{String,Any}("type" => type), d)))\n\n"
    # A GET carries no body: it fetches the stored response; a POST streams the events.
    server, url, _ = _ollama_mock((n, body) -> isempty(body) ? (200, [JSON.json(obj)]) :
        (200, [sse("response.output_text.delta", Dict("delta" => "Bonjour")),
               sse("response.completed", Dict("response" => obj))]))
    try
        ep = OllamaEndpoint(base_url=url)
        s = fetch(respond(Respond(service=ep, model="gemma4:e4b", input="Say hello in French.", stream=true,
                                  reasoning=Reasoning(effort="none"))))
        @test s isa ResponseSuccess && s.service === ep && output_text(s) == "Bonjour"
        @test (@test_logs estimated_cost(s)) == 0.0
        # Ollama stores no responses (get_response is refused there): a fetched response records the
        # endpoint of a server that stores them.
        gen = GenericOpenAIEndpoint(url, "")
        g = get_response("resp_1"; service=gen)
        @test g isa ResponseSuccess && g.service === gen && output_text(g) == "Bonjour"
    finally
        close(server)
    end
end

# Refuted if: a verb that works on a stored response reaches an Ollama server, which stores none (an
# unreachable one would answer with a call error, not an ArgumentError), or its refusal does not name
# the verb, the endpoint and what to do instead.
@testset "verbs on a stored response are refused before any request: Ollama stores none" begin
    off = OllamaEndpoint(base_url="http://127.0.0.1:1")
    refusal(call) = try call(); nothing catch e; e end
    for (verb, call, fix) in (("get_response", () -> get_response("resp_1"; service=off), "keep the ResponseSuccess"),
                              ("cancel_response", () -> cancel_response("resp_1"; service=off), "CancelToken"),
                              ("delete_response", () -> delete_response("resp_1"; service=off), "nothing stays"),
                              ("list_input_items", () -> list_input_items("resp_1"; service=off), "`input` you sent"))
        e = refusal(call)
        @test e isa ArgumentError && startswith(e.msg, "$verb works on a stored response, and OllamaEndpoint's") &&
              occursin(fix, e.msg)
    end
end

# Refuted if: compaction is refused on Ollama, or does not reach its /v1/responses/compact route with the
# model and the conversation. Ollama compacts the conversation sent with the request: nothing is stored.
@testset "compact_response reaches Ollama's compaction route" begin
    compacted = JSON.json(Dict("id" => "resp_compact_1", "object" => "response.compaction",
        "output" => [Dict("type" => "compaction", "encrypted_content" => "{\"type\":\"ollama_compaction\"}")],
        "usage" => Dict("input_tokens" => 285, "output_tokens" => 67, "total_tokens" => 352)))
    seen = Ref{Any}(nothing)
    handler = req -> (seen[] = (String(req.method), String(req.target),
                                applicable(copy, req.body) ? String(copy(req.body)) : "");
                      HTTP.Response(200, ["Content-Type" => "application/json"], Vector{UInt8}(compacted)))
    tcp = Sockets.listen(Sockets.localhost, 0); port = Int(Sockets.getsockname(tcp)[2]); close(tcp)
    server = HTTP.serve!(handler, "127.0.0.1", port; verbose=false)
    try
        r = compact_response(model="gemma4:e4b", input=[InputMessage(role="user", content="My name is Ada.")],
                             service=OllamaEndpoint(base_url="http://127.0.0.1:$port"))
        @test r isa AbstractDict && r["object"] == "response.compaction" && only(r["output"])["type"] == "compaction"
        method, target, body = seen[]
        @test (method, target) == ("POST", "/v1/responses/compact")
        @test JSON.parse(body)["model"] == "gemma4:e4b" && only(JSON.parse(body)["input"])["content"] == "My name is Ada."
        # Ollama has no default model: a call naming none is refused before any request, in Chat's words. Another
        # service is sent OpenAI's default model.
        seen[] = nothing
        e = try compact_response(input="hi", service=OllamaEndpoint(base_url="http://127.0.0.1:$port")); nothing catch x; x end
        chat = try Chat(service=OllamaEndpoint(base_url="http://127.0.0.1:$port")); nothing catch x; x end
        @test e isa ArgumentError && e.msg == chat.msg && occursin("list_models", e.msg) && isnothing(seen[])
        compact_response(input="hi", service=GenericOpenAIEndpoint("http://127.0.0.1:$port", ""))
        @test JSON.parse(seen[][3])["model"] == UniLM.default_model(OPENAIServiceEndpoint) == "gpt-6-sol"
    finally
        close(server)
    end
end

@testset "transcription on Ollama: its OpenAI-compatible route and its rules" begin
    wav = joinpath(@__DIR__, "fixtures", "ollama", "secret_word.wav")
    off = OllamaEndpoint(base_url="http://127.0.0.1:1")
    refusal(f) = try f(); nothing catch x; x end
    e = refusal(() -> transcribe(wav; service=off))
    @test e isa ArgumentError && occursin("no default transcription model", e.msg)
    # Each refusal comes from its own rule: one distinctive fragment per message.
    for (kw, fragment) in (((temperature=0.2,), "runs at temperature 0 and takes no `temperature`"),
                           ((languages=["en"],), "takes one `language`"),
                           ((keywords=["banana"],), "takes no `keywords`"),
                           ((response_format="text",), "read the text with transcript_text"),
                           ((response_format="srt",), "JSON only (got \"srt\")"))
        e = refusal(() -> transcribe(wav; service=off, model="gemma4:e4b", kw...))
        @test e isa ArgumentError && occursin(fragment, e.msg)
    end
    @test_throws ArgumentError speak(SpeechRequest(service=off, input="hi"))          # speech stays refused
    @test_throws ArgumentError translate(wav; service=off, model="gemma4:e4b")       # translation too
    seen = Ref{Any}(nothing)
    handler = req -> (seen[] = (String(req.target), applicable(copy, req.body) ? String(copy(req.body)) : "");
        HTTP.Response(200, ["Content-Type" => "application/json"], Vector{UInt8}("{\"text\":\"The secret word is banana.\"}")))
    tcp = Sockets.listen(Sockets.localhost, 0); port = Int(Sockets.getsockname(tcp)[2]); close(tcp)
    server = HTTP.serve!(handler, "127.0.0.1", port; verbose=false)
    try
        url = "http://127.0.0.1:$port"
        r = transcribe(wav; service=OllamaEndpoint(base_url=url), model="gemma4:e4b", language="en", prompt="a fruit")
        @test r isa TranscriptionSuccess && transcript_text(r) == "The secret word is banana."
        target, body = seen[]
        @test target == "/v1/audio/transcriptions"
        @test occursin("name=\"model\"", body) && occursin("gemma4:e4b", body) && occursin("name=\"language\"", body)
        @test occursin("name=\"prompt\"", body) && occursin("name=\"file\"", body) && !occursin("temperature", body)
    finally
        close(server)
    end
end

@testset "transcription on Ollama: its JSON reply under its chat stream's content type" begin
    # Ollama labels every transcription reply application/x-ndjson, the header of the chat stream it
    # runs (as _ollama_mock does); the reply is `{"text": …}` unless response_format is "text".
    wav = joinpath(@__DIR__, "fixtures", "ollama", "secret_word.wav")
    reply = "{\"text\":\"The secret word is banana\"}\n"
    server, url, bodies = _ollama_mock((n, body) -> (200, [n == 3 ? "" : reply]))
    try
        ep = OllamaEndpoint(base_url=url)
        for fmt in (nothing, "json")
            r = transcribe(wav; service=ep, model="gemma4:e4b", response_format=fmt)
            @test r isa TranscriptionSuccess && transcript_text(r) == "The secret word is banana"
            @test r.raw == Dict("text" => "The secret word is banana")
        end
        # A failure after Ollama has begun its reply leaves the 200 body empty (its TranscriptionWriter writes
        # only when the chat stream reports done): JSON reports it, so "text", which would read it as an empty
        # transcript, is refused before any request.
        e = try transcribe(wav; service=ep, model="gemma4:e4b", response_format="text"); nothing catch x; x end
        @test e isa ArgumentError && occursin("transcript_text", e.msg) && length(bodies) == 2
        r = transcribe(wav; service=ep, model="gemma4:e4b")
        @test r isa AudioCallError && length(bodies) == 3
    finally
        close(server)
    end
end
