# Gemini context caching: the cache a Chat names (`cached_content`) on every encoder, and the cache verbs.
# Zero-spend: encoders run locally, and the verbs talk to a scripted local endpoint.
using UniLM
using UniLM: encode_request, GEMINIServiceEndpoint, GEMINIOpenAIServiceEndpoint, RoleSystem
using Test, HTTP, JSON, Dates

_cc_chat(; kw...) = Chat(; service=GEMINIServiceEndpoint, model="gemini-3.8-flash",
                         messages=[Message(Val(:user), "Which animal jumps?")], kw...)
_cc_err(f) = try f(); nothing catch e; e end

@testset "Chat cached_content — a cache's name, given so or by its id" begin
    @test isnothing(_cc_chat().cached_content)
    @test _cc_chat(cached_content="cachedContents/abc123").cached_content == "cachedContents/abc123"
    @test _cc_chat(cached_content="abc123").cached_content == "cachedContents/abc123"
    # The API's pattern is cachedContents/{id}: no empty id, no other segment, no dot segment.
    for bad in ("", "cachedContents/", "cachedContents/a/b", "files/abc", "a/b", "..", "cachedContents/..", ".")
        e = _cc_err(() -> _cc_chat(cached_content=bad))
        @test e isa ArgumentError && occursin("cached_content", e.msg) && occursin("cachedContents/", e.msg)
    end
    # A fork names the same cache.
    @test fork(_cc_chat(cached_content="abc123")).cached_content == "cachedContents/abc123"
end

@testset "Chat cached_content — native Gemini sends cachedContent, the count takes it, and the rules hold" begin
    body(c) = JSON.parse(encode_request(GEMINIServiceEndpoint, c))
    @test body(_cc_chat(cached_content="abc"))["cachedContent"] == "cachedContents/abc"
    @test !haskey(body(_cc_chat()), "cachedContent")
    # countTokens takes the generateContent request whole, cache included (observed: cachedContentTokenCount
    # counted on 2026-11-17).
    counted = JSON.parse(UniLM._count_body(UniLM._GeminiCount(), GEMINIServiceEndpoint, _cc_chat(cached_content="abc")))
    @test counted["generateContentRequest"]["cachedContent"] == "cachedContents/abc"
    # A cache carries the system instruction, tools and tool config; the API answers a request that names one
    # and sets any of them with HTTP 400.
    sig = FunctionSignature(name="noop", description="does nothing")
    for (kw, what) in (((messages=[Message(Val(:system), "Be brief."), Message(Val(:user), "Hi")],), "system message"),
                       ((tools=[Tool(func=sig)],), "tools"), ((tool_choice="none",), "tool_choice"))
        e = _cc_err(() -> encode_request(GEMINIServiceEndpoint, _cc_chat(; cached_content="abc", kw...)))
        @test e isa ArgumentError && occursin(what, e.msg) && occursin("create_cache", e.msg)
        @test haskey(body(_cc_chat(; kw...)), "contents")          # without a cache, each is sent
    end
    # Gemini 4 models take no explicit cache.
    for model in ("gemini-4-hydrogen", "gemini-4-argon", "gemini-4-carbon")
        e = _cc_err(() -> encode_request(GEMINIServiceEndpoint, _cc_chat(; model, cached_content="abc")))
        @test e isa ArgumentError && occursin("Gemini 4", e.msg) && occursin(model, e.msg)
    end
end

@testset "Chat cached_content — the compatible endpoint sends extra_body.google.cached_content" begin
    compat(; kw...) = Chat(; service=GEMINIOpenAIServiceEndpoint, model="gemini-3.8-flash",
                           messages=[Message(Val(:user), "Hi")], kw...)
    wire = JSON.parse(encode_request(GEMINIOpenAIServiceEndpoint, compat(cached_content="abc")))
    @test wire["extra_body"] == Dict("google" => Dict("cached_content" => "cachedContents/abc"))
    @test !haskey(JSON.parse(encode_request(GEMINIOpenAIServiceEndpoint, compat())), "extra_body")
    @test JSON.parse(encode_request(GEMINIOpenAIServiceEndpoint, compat(cached_content="abc", stream=true)))["extra_body"] ==
          wire["extra_body"]
    # The compatible endpoint answers a system message or tools beside a cache with the same HTTP 400.
    e = _cc_err(() -> encode_request(GEMINIOpenAIServiceEndpoint,
        compat(cached_content="abc", messages=[Message(Val(:system), "Be brief."), Message(Val(:user), "Hi")])))
    @test e isa ArgumentError && occursin("system message", e.msg)
    e = _cc_err(() -> encode_request(GEMINIOpenAIServiceEndpoint, compat(model="gemini-4-hydrogen", cached_content="abc")))
    @test e isa ArgumentError && occursin("Gemini 4", e.msg)
end

@testset "Chat cached_content — every other endpoint refuses it before any request" begin
    for (service, model) in ((OPENAIServiceEndpoint, "gpt-6-luna"), (AZUREServiceEndpoint, "gpt-5.2"),
                             (DeepSeekEndpoint(), "deepseek-flash"), (MistralEndpoint(api_key="k"), "mistral-small"),
                             (GenericOpenAIEndpoint("http://127.0.0.1:1/v1", ""), "local"))
        c = Chat(; service, model, messages=[Message(Val(:user), "Hi")], cached_content="abc")
        e = _cc_err(() -> encode_request(service, c))
        @test e isa ArgumentError && occursin("cached_content", e.msg) && occursin("Gemini", e.msg)
    end
    for (service, model, says) in ((ANTHROPICServiceEndpoint, "claude-haiku-5-5", "Anthropic Messages does not support"),
                                   (OllamaEndpoint(), "gemma4:e4b", "Ollama /api/chat does not support"))
        c = Chat(; service, model, messages=[Message(Val(:user), "Hi")], cached_content="abc")
        e = _cc_err(() -> encode_request(service, c))
        @test e isa ArgumentError && occursin(says, e.msg) && occursin("cached_content", e.msg)
    end
    # chatrequest! encodes before any I/O: the refusal reaches the caller, nothing is sent.
    @test_throws ArgumentError chatrequest!(Chat(model="gpt-6-luna", messages=[Message(Val(:user), "Hi")],
                                                 cached_content="abc"))
end

@testset "prompt_cache_options on Gemini — the refusal names the explicit cache a Chat can use" begin
    for f in (() -> encode_request(GEMINIServiceEndpoint, _cc_chat(prompt_cache_options=PromptCacheOptions(ttl="5m"))),
              () -> UniLM.encode_agentic(GEMINIServiceEndpoint, Respond(service=GEMINIServiceEndpoint, input="x",
                                                                        prompt_cache_options=PromptCacheOptions(ttl="5m"))))
        e = _cc_err(f)
        @test e isa ArgumentError && occursin("implicitly", e.msg) && occursin("create_cache", e.msg) &&
              occursin("cached_content", e.msg) && occursin("Interactions API caches implicitly only", e.msg)
    end
end
