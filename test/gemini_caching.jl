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

# ─── The cache verbs ─────────────────────────────────────────────────────────
# Served by `_with_scripted` at CacheProbe (platform_seam_fixtures.jl).

# A cache as the API answered its create, get and list on 2026-11-17.
const _CACHE = """{"name": "cachedContents/4ekeqvr7x0ipkt1midsuw60if7fi9vuh0j0vyjit", "model": "models/gemini-3.8-flash",
  "createTime": "2026-11-17T07:54:09.033140Z", "updateTime": "2026-11-17T07:54:10.315578Z",
  "expireTime": "2026-11-17T07:57:10.071548053Z", "displayName": "unilm-witness", "usageMetadata": {"totalTokenCount": 9557}}"""
const _CACHE_NAME = "cachedContents/4ekeqvr7x0ipkt1midsuw60if7fi9vuh0j0vyjit"

_probe_chat(; kw...) = Chat(; service=CacheProbe, model="gemini-3.8-flash", kw...)

# Refuted if: a field of the cache is misread, the body leaves out or renames what the Chat holds, a TTL or
# an expiry is sent in another form, or the reply's model keeps its models/ prefix.
@testset "create_cache — a Chat's system instruction, contents and tools, its TTL or expiry, decoded" begin
    sig = FunctionSignature(name="lookup", description="Looks a word up",
                            parameters=Dict("type" => "object", "properties" => Dict("w" => Dict("type" => "string"))))
    chat = _probe_chat(messages=[Message(Val(:system), "Answer in one word."), Message(Val(:user), "A long document.")],
                       tools=[Tool(func=sig)], tool_choice="auto")
    r, seen = _with_scripted(() -> create_cache(chat; ttl=120, display_name="unilm-witness"), (_, _) -> _json(200, _CACHE))
    @test [(q.method, q.target) for q in seen] == [("POST", "/v1beta/cachedContents")]
    body = JSON.parse(seen[1].body)
    @test body["model"] == "models/gemini-3.8-flash" && body["ttl"] == "120s" && body["displayName"] == "unilm-witness"
    @test body["systemInstruction"] == Dict("parts" => [Dict("text" => "Answer in one word.")])
    @test body["contents"] == [Dict("role" => "user", "parts" => [Dict("text" => "A long document.")])]
    @test body["tools"] == [Dict("functionDeclarations" => [Dict("name" => "lookup", "description" => "Looks a word up",
                                 "parametersJsonSchema" => sig.parameters)])]
    @test body["toolConfig"] == Dict("functionCallingConfig" => Dict("mode" => "AUTO"))
    @test !haskey(body, "expireTime")
    @test r isa CacheSuccess
    c = r.response
    @test (c.name, c.model, c.display_name, c.total_tokens) == (_CACHE_NAME, "gemini-3.8-flash", "unilm-witness", 9557)
    @test c.create_time == DateTime(2026, 11, 17, 7, 54, 9, 33)
    @test c.update_time == DateTime(2026, 11, 17, 7, 54, 10, 315)
    @test c.expire_time == DateTime(2026, 11, 17, 7, 57, 10, 71)
    @test c.raw["usageMetadata"]["totalTokenCount"] == 9557
    # An expiry instead of a TTL, as an RFC 3339 UTC time; a cache of contents alone sends no system instruction.
    _, seen = _with_scripted(() -> create_cache(_probe_chat(messages=[Message(Val(:user), "doc")]);
                                                expire_time=DateTime(2026, 11, 17, 8, 0, 0, 5)), (_, _) -> _json(200, _CACHE))
    body = JSON.parse(seen[1].body)
    @test body["expireTime"] == "2026-11-17T08:00:00.005Z" && !haskey(body, "ttl")
    @test !haskey(body, "systemInstruction") && !haskey(body, "tools") && !haskey(body, "displayName")
    # A Chat names the cache: its name, or the result's object.
    @test Chat(service=GEMINIServiceEndpoint, model="gemini-3.8-flash", cached_content=c).cached_content == _CACHE_NAME
end

# Refuted if: any of these reaches the network, or is refused without naming what to change.
@testset "create_cache — refused before any request" begin
    doc = [Message(Val(:user), "doc")]
    cases = [
        (() -> create_cache(Chat(service=OPENAIServiceEndpoint, model="gpt-6-luna", messages=doc)), "Context caching is not supported"),
        (() -> create_cache(_probe_chat(model="gemini-4-hydrogen", messages=doc)), "Gemini 4"),
        (() -> create_cache(_probe_chat(messages=doc); ttl=60, expire_time=DateTime(2027)), "ttl or expire_time"),
        (() -> create_cache(_probe_chat(messages=doc); ttl=0), "ttl"),
        (() -> create_cache(_probe_chat(messages=doc); display_name="x"^129), "128"),
        (() -> create_cache(_probe_chat(messages=doc, cached_content="abc")), "cached_content"),
        (() -> create_cache(_probe_chat(messages=doc, max_tokens=64)), "max_tokens"),
        (() -> create_cache(_probe_chat(messages=doc, reasoning_effort="low")), "reasoning_effort"),
        (() -> create_cache(_probe_chat(messages=doc, stream=true)), "stream"),
        (() -> create_cache(_probe_chat(messages=doc, response_format=ResponseFormat(type="json_object"))), "response_format"),
    ]
    for (call, says) in cases
        e, seen = _with_scripted(() -> (try call(); nothing catch x; x end), (_, _) -> _json(200, _CACHE))
        @test e isa ArgumentError && occursin(says, e.msg) && isempty(seen)
    end
end

# Refuted if: a verb reaches another path, sends another method or body, or reads a reply other than as documented.
@testset "retrieve, list, update and delete a cache" begin
    for name in ("abc", "cachedContents/abc")
        _, seen = _with_scripted(() -> retrieve_cache(name; service=CacheProbe), (_, _) -> _json(200, _CACHE))
        @test [(q.method, q.target) for q in seen] == [("GET", "/v1beta/cachedContents/abc")]
    end
    # A name's id is one escaped path segment.
    _, seen = _with_scripted(() -> retrieve_cache("a b?x#f"; service=CacheProbe), (_, _) -> _json(200, _CACHE))
    @test seen[1].target == "/v1beta/cachedContents/a%20b%3Fx%23f"
    for bad in ("", "a/b", "files/abc", "..")
        e = try retrieve_cache(bad; service=CacheProbe); nothing catch x; x end
        @test e isa ArgumentError && occursin("cachedContents/", e.msg)
    end
    page = """{"cachedContents": [$_CACHE], "nextPageToken": "tok2"}"""
    r, seen = _with_scripted(() -> list_caches(; limit=2, after="tok 1", service=CacheProbe), (_, _) -> _json(200, page))
    @test seen[1].target == "/v1beta/cachedContents?pageSize=2&pageToken=tok%201"
    @test r isa CacheListSuccess && length(r.response.data) == 1 && r.response.has_more &&
          r.response.data[1].name == _CACHE_NAME && r.response.raw["nextPageToken"] == "tok2"
    # An empty listing is `{}`: proto3 JSON leaves an empty list and the last page's token out.
    r, seen = _with_scripted(() -> list_caches(; service=CacheProbe), (_, _) -> _json(200, "{}"))
    @test seen[1].target == "/v1beta/cachedContents" && r isa CacheListSuccess && isempty(r.response.data) && !r.response.has_more
    @test_throws ArgumentError list_caches(; limit=0, service=CacheProbe)
    # Only the expiration updates, as a TTL or an expiry, one of them.
    r, seen = _with_scripted(() -> update_cache("abc"; ttl=300, service=CacheProbe), (_, _) -> _json(200, _CACHE))
    @test [(q.method, q.target) for q in seen] == [("PATCH", "/v1beta/cachedContents/abc?updateMask=ttl")]
    @test JSON.parse(seen[1].body) == Dict("ttl" => "300s") && r isa CacheSuccess
    _, seen = _with_scripted(() -> update_cache("abc"; expire_time=DateTime(2026, 11, 17, 8), service=CacheProbe),
                             (_, _) -> _json(200, _CACHE))
    @test seen[1].target == "/v1beta/cachedContents/abc?updateMask=expireTime" &&
          JSON.parse(seen[1].body) == Dict("expireTime" => "2026-11-17T08:00:00.000Z")
    for kw in ((;), (ttl=60, expire_time=DateTime(2027)), (ttl=-1,))
        e, seen = _with_scripted(() -> (try update_cache("abc"; service=CacheProbe, kw...); nothing catch x; x end),
                                 (_, _) -> _json(200, _CACHE))
        @test e isa ArgumentError && isempty(seen)
    end
    # The delete answers an empty message; the cache deleted is the one named.
    r, seen = _with_scripted(() -> delete_cache("abc"; service=CacheProbe), (_, _) -> _json(200, "{}"))
    @test [(q.method, q.target) for q in seen] == [("DELETE", "/v1beta/cachedContents/abc")]
    @test r isa CacheDeleteSuccess && r.name == "cachedContents/abc" && r.deleted
    # Gemini answers a cache that is gone with 403, "CachedContent not found (or permission denied)".
    gone = """{"error": {"code": 403, "message": "CachedContent not found (or permission denied)", "status": "PERMISSION_DENIED"}}"""
    for call in (() -> retrieve_cache("abc"; service=CacheProbe), () -> delete_cache("abc"; service=CacheProbe),
                 () -> update_cache("abc"; ttl=1, service=CacheProbe), () -> list_caches(; service=CacheProbe),
                 () -> create_cache(_probe_chat(messages=[Message(Val(:user), "doc")])))
        r = _answered(call, 403; body=gone)
        @test r isa CacheFailure && r.status == 403 && occursin("not found", r.response)
    end
end

# Refuted if: a 200 without a field the API always sends is read as a cache, or a field's type is coerced.
@testset "a cache reply without a documented field is a CacheCallError" begin
    full = JSON.parse(_CACHE)
    for key in ("name", "model", "createTime", "updateTime", "expireTime", "usageMetadata")
        r = _answered(() -> retrieve_cache("abc"; service=CacheProbe), 200; body=JSON.json(delete!(copy(full), key)))
        @test r isa CacheCallError && occursin(key, r.error)
    end
    for (key, v) in ("name" => "files/abc", "model" => "gemini-3.8-flash", "expireTime" => "tomorrow",
                     "usageMetadata" => Dict("totalTokenCount" => "9557"))
        r = _answered(() -> retrieve_cache("abc"; service=CacheProbe), 200; body=JSON.json(merge(full, Dict(key => v))))
        @test r isa CacheCallError
    end
    r = _answered(() -> list_caches(; service=CacheProbe), 200; body="""{"cachedContents": {}}""")
    @test r isa CacheCallError
    r = _answered(() -> retrieve_cache("abc"; service=CacheProbe), 200; body="[]")
    @test r isa CacheCallError
end

@testset "the cache verbs — capability, cancel and config" begin
    @test has_capability(GEMINIServiceEndpoint, :context_caching)
    @test !has_capability(GEMINIOpenAIServiceEndpoint, :context_caching) && !has_capability(OPENAIServiceEndpoint, :context_caching)
    e = try list_caches(; service=OPENAIServiceEndpoint); nothing catch x; x end
    @test e isa ArgumentError && occursin("Context caching is not supported by OPENAIServiceEndpoint", e.msg)
    # A token cancelled before the call sends nothing; a spent deadline ends the call before the network.
    r, seen = _with_scripted(() -> retrieve_cache("abc"; service=CacheProbe, cancel=cancel!(CancelToken())),
                             (_, _) -> _json(200, _CACHE))
    @test r isa CacheCallError && r.cause isa UniLMCancelled && isempty(seen)
    @test _seam_timeout(delete_cache("abc"; service=CacheProbe, config=_TINY_DEADLINE), CacheCallError)
end

# The reply of a generateContent request that named a cache, 2026-11-17: the cached tokens are part of the prompt,
# and bill at the model's cache-hit rate ("Cache token count: ... billed at a reduced rate"), the rest of the prompt at
# the input rate. Refuted if the cached tokens bill at the input rate or are counted twice.
@testset "a reply that read a cache bills its cached tokens at the cache-hit rate" begin
    reply = Dict("candidates" => [Dict("content" => Dict("role" => "model", "parts" => [Dict("text" => "Fox")]),
                                       "finishReason" => "STOP", "index" => 0)],
                 "usageMetadata" => Dict("promptTokenCount" => 9565, "candidatesTokenCount" => 1, "totalTokenCount" => 9610,
                                         "cachedContentTokenCount" => 9557, "thoughtsTokenCount" => 44,
                                         "promptTokensDetails" => [Dict("modality" => "TEXT", "tokenCount" => 9565)],
                                         "cacheTokensDetails" => [Dict("modality" => "TEXT", "tokenCount" => 9557)],
                                         "serviceTier" => "standard"),
                 "modelVersion" => "gemini-3.8-flash")
    d = UniLM.decode_response(GEMINIServiceEndpoint, HTTP.Response(200, [], Vector{UInt8}(JSON.json(reply))))
    u = d.usage
    @test (u.prompt_tokens, u.cached_tokens, u.completion_tokens) == (9565, 9557, 45)
    rates = UniLM._price_row(DEFAULT_PRICING, "gemini-3.8-flash")
    chat = Chat(service=GEMINIServiceEndpoint, model="gemini-3.8-flash", cached_content="abc")
    @test estimated_cost(LLMSuccess(message=d.message, self=chat, usage=u, model="gemini-3.8-flash")) ≈
          8 * rates.input + 9557 * rates.cached_input + 45 * rates.output
end
