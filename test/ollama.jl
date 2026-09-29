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
    err = try OllamaEndpoint(num_ctxx=10); nothing catch x; x end
    @test err isa ArgumentError && occursin("num_ctxx", err.msg) && occursin("num_ctx", err.msg)
    for (k, v) in (:num_ctx => 0, :num_batch => 0, :num_thread => 0, :num_gpu => -2, :main_gpu => -1,
                   :num_keep => -2, :top_k => 0, :min_p => 1.5, :repeat_last_n => -2,
                   :repeat_penalty => -0.1)
        @test_throws ArgumentError OllamaOptions(; k => v)
    end
    @test OllamaEndpoint() == OllamaEndpoint()                       # plain immutable value
    @test provider_capabilities(e) == Set([:chat, :embeddings, :fim, :tools, :streaming, :json_output, :responses, :models])
    @test !occursin("redacted", sprint(show, e))                     # no key to hide
    @test_throws ArgumentError Chat(service=e)                       # no default model
    @test occursin("ollama list", try Chat(service=e); "" catch x; x.msg end)
end

@testset "URL routing" begin
    e = OllamaEndpoint(base_url="http://box:11434")
    @test UniLM.get_url(e, Chat(service=e, model="m")) == "http://box:11434/api/chat"
    @test UniLM.get_url(e, Embeddings("x"; service=e, model="embeddinggemma")) == "http://box:11434/api/embed"
    @test UniLM.get_url(e, FIMCompletion(service=e, model="qwen2.5-coder:1.5b-base", prompt="x")) ==
          "http://box:11434/v1/completions"
    @test UniLM._agentic_url(e) == "http://box:11434/v1/responses"
    @test UniLM.auth_header(e) == ["Content-Type" => "application/json"]
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
    for lvl in ("low", "medium", "high")
        @test _wire(_ochat(e; reasoning_effort=lvl))["think"] == lvl
    end
    for bad in ("minimal", "xhigh", "max")
        err = try _wire(_ochat(e; reasoning_effort=bad)); nothing catch x; x end
        @test err isa ArgumentError && occursin(bad, err.msg)
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
    for lvl in ("low", "medium", "high")                    # a thinking model may skip thinking: unconstrained
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
              embeddingrequest!(Embeddings("x"; service=e, model="embeddinggemma"); config=cfg))
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
    emb = EmbeddingSuccess(embeddings=Embeddings("x"; service=e, model="embeddinggemma"),
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
    @test occursin("whole conversation", try enc(previous_response_id="r"); "" catch x; x.msg end)
    @test_throws ArgumentError enc(tool_choice="required")
    @test haskey(enc(tool_choice="auto"), "model")
    @test_throws ArgumentError enc(reasoning=Reasoning(effort="xhigh"))
    @test_throws ArgumentError enc(reasoning=Reasoning(effort="low", summary="auto"))
    @test_throws ArgumentError enc(text=TextConfig(format=TextFormatSpec(type="json_object")))
    @test_throws ArgumentError enc(text=TextConfig(verbosity="low"))
    @test_throws ArgumentError enc(input=[InputMessage(role="user", content=[input_image(file_id="file_1")])])
    # respond validates before any I/O: the refusal is a throw, not a result.
    @test_throws ArgumentError respond(Respond(service=e, model="m", input="x", store=true))
    # A Respond tool loop cannot chain turns without server-side state.
    err = try tool_loop(Respond(service=e, model="m", input="x"), (n, a) -> "r"); nothing catch x; x end
    @test err isa ArgumentError && occursin("tool_loop!", err.msg)
end

@testset "embeddings: native /api/embed, over-long input is an error" begin
    e = OllamaEndpoint(keep_alive=60, num_ctx=2048)
    body = JSON.parse(UniLM._encode_embeddings(e, Embeddings(["a", "b"]; service=e, model="embeddinggemma", dimensions=256)))
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
        r = embeddingrequest!(Embeddings(["a", "b"]; service=ep, model="embeddinggemma"))
        @test r isa EmbeddingSuccess
        @test embedding_vectors(r) == [[0.1, 0.2], [0.3, 0.4]]
        @test r.usage == TokenUsage(prompt_tokens=4, total_tokens=4)
        @test (@test_logs estimated_cost(r)) == 0.0
        r1 = embeddingrequest!(Embeddings("a"; service=ep, model="embeddinggemma"))
        @test embedding_vectors(r1) == [1.0, 0.0, 0.0]                      # the model's own dimension
        f = embeddingrequest!(Embeddings("x"^10_000; service=ep, model="embeddinggemma"); config=RequestConfig(max_attempts=1))
        @test f isa EmbeddingFailure && f.status == 400 && occursin("exceeds the context length", f.response)
        @test all(b -> JSON.parse(b)["truncate"] === false, bodies)
    finally
        close(server)
    end
end
