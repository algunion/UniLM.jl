# DeepSeek thinking-mode round trip — deterministic, zero-spend unit tests.
# https://api-docs.deepseek.com/guides/thinking_mode: DeepSeek models think by default and return
# the chain of thought as `reasoning_content` beside `content` (`delta.reasoning_content` when
# streaming). A request that carries `tools` must pass back the reasoning_content of every earlier
# assistant turn, turns without tool calls included, or the API answers 400; without tools it is not
# needed and the API ignores it.
using UniLM
using UniLM: encode_request, decode_response, StreamState, _build_stream_message,
             RoleAssistant, RoleTool, TOOL_CALLS, GPTFunction
using Test, HTTP, JSON

const _DS = DeepSeekEndpoint("k")
const _DS_TOOLS = [Tool(func=FunctionSignature(name="get_weather",
    parameters=Dict("type" => "object", "properties" => Dict("city" => Dict("type" => "string")))))]
_ds_resp(d) = HTTP.Response(200, [], Vector{UInt8}(JSON.json(d)))
_ds_reasoning(m::Message) = m.provider_content isa ProviderContent && m.provider_content.provider === :deepseek ?
    m.provider_content.blocks : nothing

const _DS_TOOL_TURN = Dict("model" => "deepseek-flash", "choices" => [Dict("index" => 0, "finish_reason" => "tool_calls",
    "message" => Dict("role" => "assistant", "content" => nothing, "reasoning_content" => "Need the weather tool.",
        "tool_calls" => [Dict("id" => "call_1", "type" => "function",
            "function" => Dict("name" => "get_weather", "arguments" => "{\"city\":\"Oslo\"}"))]))],
    "usage" => Dict("prompt_tokens" => 10, "completion_tokens" => 20, "total_tokens" => 30))
const _DS_ANSWER_TURN = Dict("model" => "deepseek-flash", "choices" => [Dict("index" => 0, "finish_reason" => "stop",
    "message" => Dict("role" => "assistant", "content" => "It is 12C in Oslo.",
                      "reasoning_content" => "The tool said 12C."))])

@testset "decode — reasoning_content is captured as DeepSeek provider content" begin
    dec = decode_response(_DS, _ds_resp(_DS_TOOL_TURN))
    m = dec.message
    @test _ds_reasoning(m) == Any[Dict{String,Any}("reasoning_content" => "Need the weather tool.")]
    # The neutral fields are the OpenAI decoder's, unchanged.
    @test m.finish_reason == TOOL_CALLS && m.tool_calls[1].id == "call_1" &&
          m.tool_calls[1].func.arguments == Dict{String,Any}("city" => "Oslo") && isnothing(m.content)
    @test dec.usage.total_tokens == 30
    # No reasoning (thinking disabled, or an empty one) → no capture.
    for rc in (nothing, "")
        plain = Dict("choices" => [Dict("index" => 0, "finish_reason" => "stop",
            "message" => Dict("role" => "assistant", "content" => "hi", "reasoning_content" => rc))])
        @test isnothing(decode_response(_DS, _ds_resp(plain)).message.provider_content)
    end
    # A 200 without choices is still the OpenAI decoder's error.
    @test_throws ErrorException decode_response(_DS, _ds_resp(Dict("choices" => [])))
end

@testset "encode — reasoning_content is passed back on assistant turns when the request carries tools" begin
    tc = ToolCall(id="call_1", func=GPTFunction("get_weather", Dict{String,Any}("city" => "Oslo")))
    r(text) = ProviderContent(:deepseek, Any[Dict{String,Any}("reasoning_content" => text)])
    msgs = [Message(Val(:system), "s"), Message(Val(:user), "weather in Oslo?"),
            Message(role=RoleAssistant, tool_calls=[tc], finish_reason=TOOL_CALLS, provider_content=r("Need the weather tool.")),
            Message(role=RoleTool, tool_call_id="call_1", content="12C"),
            Message(role=RoleAssistant, content="It is 12C in Oslo.", provider_content=r("The tool said 12C.")),
            Message(Val(:user), "and tomorrow?")]
    chat = Chat(service=_DS, model="deepseek-flash", tools=_DS_TOOLS, messages=msgs)
    golden = JSON.parse(JSON.json(chat))                      # the OpenAI wire …
    golden["messages"][3]["reasoning_content"] = "Need the weather tool."   # … plus every turn's reasoning,
    golden["messages"][5]["reasoning_content"] = "The tool said 12C."       # a turn without tool calls too
    @test JSON.parse(encode_request(_DS, chat)) == golden
    # Without tools the reasoning is not needed and the API ignores it: the plain OpenAI body.
    chat2 = Chat(service=_DS, model="deepseek-flash", messages=msgs)
    @test encode_request(_DS, chat2) == JSON.json(chat2)
    # Another provider's captured blocks are never sent on the DeepSeek wire.
    other = [msgs[1:2]; Message(role=RoleAssistant, content="x", provider_content=ProviderContent(:anthropic,
             Any[Dict{String,Any}("type" => "text", "text" => "x")])); Message(Val(:user), "y")]
    chat3 = Chat(service=_DS, model="deepseek-flash", tools=_DS_TOOLS, messages=other)
    @test encode_request(_DS, chat3) == JSON.json(chat3)
end

@testset "tool loop — decoded turns round-trip their reasoning on every later request" begin
    chat = Chat(service=_DS, model="deepseek-flash", tools=_DS_TOOLS)
    push!(chat, Message(Val(:system), "Use the weather tool."))
    push!(chat, Message(Val(:user), "weather in Oslo?"))
    push!(chat, decode_response(_DS, _ds_resp(_DS_TOOL_TURN)).message)
    push!(chat, Message(role=RoleTool, tool_call_id="call_1", content="12C"))
    wire = JSON.parse(encode_request(_DS, chat))["messages"]                  # sub-request of turn 1
    @test [get(m, "reasoning_content", nothing) for m in wire] == [nothing, nothing, "Need the weather tool.", nothing]
    push!(chat, decode_response(_DS, _ds_resp(_DS_ANSWER_TURN)).message)
    push!(chat, Message(Val(:user), "and tomorrow?"))
    wire = JSON.parse(encode_request(_DS, chat))["messages"]                  # turn 2 still carries turn 1's
    @test [get(m, "reasoning_content", nothing) for m in wire] ==
          [nothing, nothing, "Need the weather tool.", nothing, "The tool said 12C.", nothing]
    @test wire[3]["tool_calls"][1]["id"] == "call_1" && wire[5]["content"] == "It is 12C in Oslo."
end

_ds_stream(chunks; done=true) =
    join(("data: " * JSON.json(c) for c in chunks), "\n") * (done ? "\ndata: [DONE]\n" : "\n")
const _DS_STREAM = [
    Dict("choices" => [Dict("index" => 0, "delta" => Dict("role" => "assistant", "reasoning_content" => "Need the "),
                            "finish_reason" => nothing)]),
    Dict("choices" => [Dict("index" => 0, "delta" => Dict("reasoning_content" => "weather tool."), "finish_reason" => nothing)]),
    Dict("choices" => [Dict("index" => 0, "delta" => Dict("tool_calls" => [Dict("index" => 0, "id" => "call_1", "type" => "function",
        "function" => Dict("name" => "get_weather", "arguments" => "{\"city\":"))]), "finish_reason" => nothing)]),
    Dict("choices" => [Dict("index" => 0, "delta" => Dict("tool_calls" => [Dict("index" => 0,
        "function" => Dict("arguments" => "\"Oslo\"}"))]), "finish_reason" => "tool_calls")]),
    Dict("choices" => Any[], "usage" => Dict("prompt_tokens" => 10, "completion_tokens" => 20, "total_tokens" => 30)),
]

@testset "stream — delta.reasoning_content is captured complete, like the non-streamed turn" begin
    st = StreamState()
    @test UniLM._sse_dispatch!(_DS, IOBuffer(), Ref(""), _ds_stream(_DS_STREAM), st) === :done
    @test String(take!(st.pending_delta)) == ""              # reasoning is not answer text
    streamed = _build_stream_message(st)
    decoded = decode_response(_DS, _ds_resp(_DS_TOOL_TURN)).message
    @test _ds_reasoning(streamed) == _ds_reasoning(decoded)
    @test streamed.tool_calls[1].func.arguments == decoded.tool_calls[1].func.arguments
    # The streamed turn encodes exactly like the non-streamed one.
    enc(m) = encode_request(_DS, Chat(service=_DS, model="deepseek-flash", tools=_DS_TOOLS,
        messages=[Message(Val(:user), "weather in Oslo?"), m, Message(role=RoleTool, tool_call_id="call_1", content="12C")]))
    @test enc(streamed) == enc(decoded)
    # A stream that ends at EOF after its finish_reason (no [DONE]) still yields the complete capture.
    st2 = StreamState()
    @test UniLM._sse_dispatch!(_DS, IOBuffer(), Ref(""), _ds_stream(_DS_STREAM; done=false), st2) === :continue
    @test _ds_reasoning(_build_stream_message(st2)) == _ds_reasoning(decoded)
    # A truncated stream (no finish_reason, no [DONE]) leaves the capture incomplete: no provider content.
    st3 = StreamState()
    UniLM._sse_dispatch!(_DS, IOBuffer(), Ref(""), _ds_stream(_DS_STREAM[1:2]; done=false), st3)
    @test isnothing(_build_stream_message(st3).provider_content)
end

# https://api-docs.deepseek.com/guides/kv_cache: `usage` reports the prompt's context-cache hits as
# `prompt_cache_hit_tokens` and the rest as `prompt_cache_miss_tokens` (prompt_tokens = hit + miss).
const _DS_CACHE_USAGE = Dict("prompt_tokens" => 1000, "completion_tokens" => 50, "total_tokens" => 1050,
                             "prompt_cache_hit_tokens" => 768, "prompt_cache_miss_tokens" => 232)

@testset "usage — context-cache hits are cached input, streamed and not" begin
    turn(u) = Dict("model" => "deepseek-flash", "choices" => [Dict("index" => 0, "finish_reason" => "stop",
        "message" => Dict("role" => "assistant", "content" => "hi"))], "usage" => u)
    u = decode_response(_DS, _ds_resp(turn(_DS_CACHE_USAGE))).usage
    @test (u.prompt_tokens, u.cached_tokens, u.completion_tokens, u.total_tokens) == (1000, 768, 50, 1050)
    # Billed at the cache-hit rate: 232 misses, 768 hits, 50 output tokens.
    row = UniLM.DEFAULT_PRICING["deepseek-flash"]
    ok = LLMSuccess(message=Message(role=RoleAssistant, content="hi"),
                    self=Chat(service=_DS, model="deepseek-flash"), usage=u)
    @test estimated_cost(ok) ≈ 232 * row.input + 768 * row.cached_input + 50 * row.output
    # Beside the OpenAI-style detail the API reference also documents (the same count).
    both = merge(_DS_CACHE_USAGE, Dict("prompt_tokens_details" => Dict("cached_tokens" => 768)))
    @test decode_response(_DS, _ds_resp(turn(both))).usage.cached_tokens == 768
    # No cache fields: nothing is cached.
    plain = Dict("prompt_tokens" => 10, "completion_tokens" => 20, "total_tokens" => 30)
    @test decode_response(_DS, _ds_resp(turn(plain))).usage.cached_tokens == 0
    # Streamed, the usage arrives in the last chunk before [DONE] (stream_options.include_usage).
    st = StreamState()
    chunks = [Dict("choices" => [Dict("index" => 0, "delta" => Dict("role" => "assistant", "content" => "hi"),
                                      "finish_reason" => "stop")]),
              Dict("choices" => Any[], "usage" => _DS_CACHE_USAGE)]
    @test UniLM._sse_dispatch!(_DS, IOBuffer(), Ref(""), _ds_stream(chunks), st) === :done
    @test (st.usage.prompt_tokens, st.usage.cached_tokens) == (1000, 768)
end

@testset "served model and usage details survive the DeepSeek decode, streamed and not" begin
    turn = Dict("model" => "deepseek-flash", "choices" => [Dict("index" => 0, "finish_reason" => "stop",
        "message" => Dict("role" => "assistant", "content" => "hi"))], "usage" => _DS_CACHE_USAGE)
    dec = decode_response(_DS, _ds_resp(turn))
    @test dec.model == "deepseek-flash"
    @test dec.usage == TokenUsage(prompt_tokens=1000, completion_tokens=50, total_tokens=1050, cached_tokens=768)
    @test isnothing(decode_response(_DS, _ds_resp(delete!(copy(turn), "model"))).model)
    st = StreamState()
    chunks = [Dict("model" => "deepseek-flash", "choices" => [Dict("index" => 0,
                   "delta" => Dict("role" => "assistant", "content" => "hi"), "finish_reason" => "stop")]),
              Dict("model" => "deepseek-flash", "choices" => Any[], "usage" => _DS_CACHE_USAGE)]
    @test UniLM._sse_dispatch!(_DS, IOBuffer(), Ref(""), _ds_stream(chunks), st) === :done
    @test st.model == dec.model && st.usage == dec.usage
    # The cache-hit count replaces only cached_tokens: every other field of the usage is kept, the
    # audio counts included.
    wire = TokenUsage(1000, 50, 1050, 0, 7, 300, 100, "flex", "us", 12, 17)
    hit = @inferred UniLM._deepseek_usage(wire, _DS_CACHE_USAGE)
    @test hit == TokenUsage(1000, 50, 1050, 768, 7, 300, 100, "flex", "us", 12, 17)
    @test all(f -> f === :cached_tokens || getfield(hit, f) === getfield(wire, f), fieldnames(TokenUsage))
    @test UniLM._deepseek_usage(wire, Dict("prompt_tokens" => 1000)) === wire
    # So does a reply whose detail objects count audio beside the cache hits.
    voiced = merge(_DS_CACHE_USAGE, Dict("prompt_tokens_details" => Dict("audio_tokens" => 12),
                                         "completion_tokens_details" => Dict("audio_tokens" => 17)))
    u = decode_response(_DS, _ds_resp(merge(turn, Dict("usage" => voiced)))).usage
    @test u == TokenUsage(prompt_tokens=1000, completion_tokens=50, total_tokens=1050, cached_tokens=768,
                          prompt_audio_tokens=12, completion_audio_tokens=17)
end

@testset "DeepSeekEndpoint and MistralEndpoint read their key when a request is sent" begin
    withenv("DEEPSEEK_API_KEY" => nothing, "MISTRAL_API_KEY" => nothing) do
        @test DeepSeekEndpoint() isa DeepSeekEndpoint                 # no KeyError at construction
        @test MistralEndpoint() isa GenericOpenAIEndpoint
        r = chatrequest!(Chat(service=DeepSeekEndpoint(), messages=[Message(Val(:system), "s"), Message(Val(:user), "u")]);
                         config=RequestConfig(max_attempts=1))
        @test r isa LLMCallError && occursin("DEEPSEEK_API_KEY", r.error)
        @test_throws KeyError UniLM.auth_header(MistralEndpoint())    # a missing variable throws: no request goes out without it
        r = chatrequest!(Chat(service=MistralEndpoint(), model="mistral-large-latest",
                              messages=[Message(Val(:system), "s"), Message(Val(:user), "u")]); config=RequestConfig(max_attempts=1))
        @test r isa LLMCallError && occursin("MISTRAL_API_KEY", r.error)
    end
    withenv("DEEPSEEK_API_KEY" => "sk-test-1", "MISTRAL_API_KEY" => "sk-test-2") do
        @test ("Authorization" => "Bearer sk-test-1") in UniLM.auth_header(DeepSeekEndpoint())
        @test ("Authorization" => "Bearer sk-test-2") in UniLM.auth_header(MistralEndpoint())
    end
    shown = sprint(show, DeepSeekEndpoint(api_key="sk-secret-123456"))
    @test !occursin("secret-123456", shown) && occursin("[redacted]", shown)
    @test occursin("DEEPSEEK_API_KEY", sprint(show, DeepSeekEndpoint()))        # names the variable, never a key
    @test UniLM._MODEL_ID_PREFIXES.anthropic == ("claude-",)
end

@testset "a key variable is read on every request; a given key is sent as is" begin
    lazy = (DeepSeekEndpoint() => "DEEPSEEK_API_KEY", MistralEndpoint() => "MISTRAL_API_KEY")
    for (ep, var) in lazy, key in ("sk-first", "sk-second")        # one endpoint, its variable changed between requests
        withenv(var => key) do
            @test ("Authorization" => "Bearer $key") in UniLM.auth_header(ep)
        end
    end
    given = (DeepSeekEndpoint(api_key="sk-given") => "DEEPSEEK_API_KEY", MistralEndpoint(api_key="sk-given") => "MISTRAL_API_KEY",
             GenericOpenAIEndpoint("http://127.0.0.1:8000", "sk-given", "UNILM_TEST_KEY") => "UNILM_TEST_KEY")
    for (ep, var) in given, value in (nothing, "sk-env")            # unset: never read; set: never preferred
        withenv(var => value) do
            @test ("Authorization" => "Bearer sk-given") in UniLM.auth_header(ep)
        end
    end
    no_auth = ["Content-Type" => "application/json"]
    @test UniLM.auth_header(GenericOpenAIEndpoint("http://127.0.0.1:8000", "")) == no_auth
    withenv("MISTRAL_API_KEY" => "") do                             # an empty key, given or read, sends none
        @test UniLM.auth_header(MistralEndpoint()) == no_auth
    end
    sub = GenericOpenAIEndpoint(SubString("http://127.0.0.1:8000/v1", 1, 21), SubString("k-1", 1, 1))
    @test (sub.base_url, sub.api_key, sub.key_env) == ("http://127.0.0.1:8000", "k", "")
    withenv("DEEPSEEK_API_KEY" => "sk-env-secret-1", "MISTRAL_API_KEY" => "sk-env-secret-2") do
        for ep in (DeepSeekEndpoint(), DeepSeekEndpoint(api_key="sk-given"), MistralEndpoint())
            @test (@inferred UniLM.auth_header(ep)) isa Vector{Pair{String,String}}
        end
        for (ep, var) in lazy
            shown = sprint(show, ep)
            @test occursin(var, shown) && !occursin("secret", shown)  # the variable, never its value
            @test eval(Meta.parse(shown)) == ep                         # valid Julia that builds the same endpoint
        end
    end
end

@testset "a Chat with a strict tool goes to DeepSeek's beta base URL, where strict mode lives" begin
    # "To use `strict` mode, you need to: Use `base_url="https://api.deepseek.com/beta"` to enable Beta
    # features" (https://api-docs.deepseek.com/guides/tool_calls).
    tool(strict) = Tool(func=FunctionSignature(name="f", parameters=Dict("type" => "object"), strict=strict))
    chat(tools; kw...) = Chat(; service=_DS, tools, messages=[Message(Val(:user), "u")], kw...)
    @test UniLM.get_url(chat(nothing)) == "https://api.deepseek.com/v1/chat/completions"
    @test UniLM.get_url(chat([tool(nothing), tool(false)])) == "https://api.deepseek.com/v1/chat/completions"
    @test UniLM.get_url(chat([tool(true)])) == "https://api.deepseek.com/beta/v1/chat/completions"
    @test UniLM.get_url(chat([tool(false), tool(true)])) == "https://api.deepseek.com/beta/v1/chat/completions"
    # The request goes there, streamed or not: a replay scope that holds no recording names the path it missed.
    for stream in (nothing, true)
        miss = try
            with_recorded_answers(mktempdir(); mode=:replay) do
                r = chatrequest!(chat([tool(true)]; stream))
                r isa Task ? fetch(r) : r
            end
        catch e
            e
        end
        @test miss isa ReplayMissError && miss.path == "/beta/v1/chat/completions"
    end
end
