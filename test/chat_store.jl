# Stored Chat Completions (src/chat_store.jl): list, get, update, delete and the input messages of the Chat
# Completions OpenAI stored. The replies are the shapes the API returned on 2026-11-17.

# The stored-completions wire at a dead port (at the scripted endpoint: StoreProbe, platform_seam_fixtures.jl).
struct StoreSeamProbe <: UniLM.ServiceEndpoint end
UniLM._resolve_base_url(::Type{StoreSeamProbe}) = "http://127.0.0.1:1"
UniLM.auth_header(::Type{StoreSeamProbe}) = ["Authorization" => "Bearer t", "Content-Type" => "application/json"]
UniLM.provider_capabilities(::Type{StoreSeamProbe}) = Set([:stored_completions])

_sc_item(; id="chatcmpl-1", metadata=Dict{String,Any}(), more...) = Dict{String,Any}(
    "object" => "chat.completion", "id" => id, "model" => "gpt-6-luna", "created" => 1794881215,
    "request_id" => "req_1", "tool_choice" => nothing, "seed" => 1546810885805346699, "temperature" => 1.0,
    "usage" => Dict("total_tokens" => 25, "completion_tokens" => 4, "prompt_tokens" => 21),
    "service_tier" => "default", "tools" => nothing, "metadata" => metadata, "response_format" => nothing,
    "choices" => [Dict("index" => 0, "message" => Dict("content" => "Red", "role" => "assistant", "tool_calls" => nothing,
                                                       "function_call" => nothing), "finish_reason" => "stop", "logprobs" => nothing)],
    (string(k) => v for (k, v) in more)...)
_sc_page(items; more=false) = Dict{String,Any}("object" => "list", "data" => items,
    "first_id" => isempty(items) ? nothing : first(items)["id"], "last_id" => isempty(items) ? nothing : last(items)["id"],
    "has_more" => more)
_sc_targets(seen) = [(s.method, s.target) for s in seen]
_sc_refused(f) = try f(); nothing catch e; e end

@testset "Stored Chat Completions: OpenAI serves them, and no other provider" begin
    @test has_capability(OPENAIServiceEndpoint, :stored_completions)
    for s in (GEMINIServiceEndpoint, ANTHROPICServiceEndpoint, GEMINIOpenAIServiceEndpoint, AZUREServiceEndpoint)
        @test !has_capability(s, :stored_completions)
        for call in (() -> list_chat_completions(; service=s), () -> retrieve_chat_completion("c"; service=s),
                     () -> update_chat_completion("c", Dict("k" => "v"); service=s),
                     () -> delete_chat_completion("c"; service=s), () -> list_chat_completion_messages("c"; service=s))
            @test _sc_refused(call) isa ArgumentError
        end
    end
end

# Refuted if: a filter is dropped or sent unencoded, an order the reference does not list is sent, or a page
# decodes without its completions' replies, usage and metadata.
@testset "list_chat_completions: the filters, and a page of completions" begin
    page = _sc_page([_sc_item(), _sc_item(id="chatcmpl-2", metadata=Dict("suite" => "unilm"))]; more=true)
    r, seen = _with_scripted((_, _) -> _json(200, JSON.json(page))) do
        list_chat_completions(; model="gpt-6-luna", metadata=Dict("suite" => "unilm", "a b" => "c&d"), after="chatcmpl-0",
                              limit=2, order="desc", service=StoreProbe)
    end
    @test _sc_targets(seen) == [("GET", "/v1/chat/completions?model=gpt-6-luna&after=chatcmpl-0&limit=2&order=desc" *
                                        "&metadata%5Ba%20b%5D=c%26d&metadata%5Bsuite%5D=unilm")]
    @test r isa StoredCompletionListSuccess
    l = r.response
    @test [c.id for c in l.data] == ["chatcmpl-1", "chatcmpl-2"] && l.has_more && (l.first_id, l.last_id) == ("chatcmpl-1", "chatcmpl-2")
    c = l.data[1]
    @test (c.model, c.created, c.metadata) == ("gpt-6-luna", 1794881215, Dict{String,String}())
    @test c.message.role == "assistant" && c.message.content == "Red" && c.message.finish_reason == "stop"
    @test (c.usage.prompt_tokens, c.usage.completion_tokens) == (21, 4) && c.raw["seed"] == 1546810885805346699
    @test l.data[2].metadata == Dict("suite" => "unilm") && l.raw == page
    empty, eseen = _with_scripted((_, _) -> _json(200, JSON.json(_sc_page([])))) do
        list_chat_completions(; service=StoreProbe)
    end
    @test empty isa StoredCompletionListSuccess && isempty(empty.response.data) && isnothing(empty.response.last_id)
    @test _sc_targets(eseen) == [("GET", "/v1/chat/completions")]
    # Refused before any request: an order the reference does not list, metadata past its bounds.
    for call in (() -> list_chat_completions(; order="newest", service=StoreSeamProbe),
                 () -> list_chat_completions(; metadata=Dict("k$i" => "v" for i in 1:17), service=StoreSeamProbe),
                 () -> list_chat_completions(; metadata=Dict("k" => 1), service=StoreSeamProbe),
                 () -> list_chat_completions(; metadata=Dict("k"^65 => "v"), service=StoreSeamProbe),
                 () -> list_chat_completions(; metadata=Dict("k" => "v"^513), service=StoreSeamProbe))
        @test _sc_refused(call) isa ArgumentError
    end
end

# A reply without what the reference requires is malformed — a call error — never a page or a completion with
# parts left out.
@testset "Stored Chat Completions: a malformed 200 is a call error" begin
    bad_pages = (Dict("object" => "list", "data" => [], "has_more" => false),                 # no first_id / last_id
                 Dict("object" => "list", "data" => [], "first_id" => nothing, "last_id" => nothing),   # no has_more
                 _sc_page([Dict("object" => "chat.completion", "id" => "c")]),             # a completion without its fields
                 _sc_page([_sc_item(choices=[])]),                                         # without a choice
                 _sc_page([_sc_item(object="chat.completion.chunk")]),
                 _sc_page([_sc_item(created="1794881215")]),
                 _sc_page([_sc_item(metadata=Dict("k" => 1))]),
                 Dict("object" => "list", "data" => [3], "first_id" => nothing, "last_id" => nothing, "has_more" => false))
    for body in bad_pages
        r = _answered(() -> list_chat_completions(; service=StoreProbe), 200; body=JSON.json(body))
        @test r isa StoredCompletionCallError && r.cause isa Exception
    end
    for body in (Dict("id" => "c"), _sc_item(model=nothing), _sc_item(choices=nothing))
        @test _answered(() -> retrieve_chat_completion("c"; service=StoreProbe), 200; body=JSON.json(body)) isa StoredCompletionCallError
    end
end

@testset "get, update and delete a stored completion; read its input messages" begin
    g, gseen = _with_scripted((_, _) -> _json(200, JSON.json(_sc_item()))) do
        retrieve_chat_completion(_HOSTILE_ID; service=StoreProbe)
    end
    @test g isa StoredCompletionSuccess && g.response.id == "chatcmpl-1" && g.response.message.content == "Red"
    @test _sc_targets(gseen) == [("GET", "/v1/chat/completions/$_HOSTILE_ENC")]
    nf = _answered(() -> retrieve_chat_completion("chatcmpl-x"; service=StoreProbe), 404;
                   body="{\"error\":{\"message\":\"Completion chatcmpl-x not found\",\"code\":\"not_found\"}}",
                   headers=["x-request-id" => "req_404"])
    @test nf isa StoredCompletionFailure && nf.status == 404 && nf.request_id == "req_404"
    # update: the metadata, the only field the API modifies.
    md = Dict("suite" => "unilm", "state" => "updated")
    u, useen = _with_scripted((_, _) -> _json(200, JSON.json(_sc_item(metadata=md)))) do
        update_chat_completion("chatcmpl-1", md; service=StoreProbe)
    end
    @test u isa StoredCompletionSuccess && u.response.metadata == md
    @test _sc_targets(useen) == [("POST", "/v1/chat/completions/chatcmpl-1")] && JSON.parse(only(useen).body) == Dict("metadata" => md)
    for bad in (Dict("k$i" => "v" for i in 1:17), Dict(:k => "v"), Dict("k" => "v"^513))
        @test _sc_refused(() -> update_chat_completion("c", bad; service=StoreSeamProbe)) isa ArgumentError
    end
    # delete: the reply names the deleted completion and confirms it.
    d, dseen = _with_scripted((_, _) -> _json(200, "{\"object\":\"chat.completion.deleted\",\"id\":\"chatcmpl-1\",\"deleted\":true}")) do
        delete_chat_completion("chatcmpl-1"; service=StoreProbe)
    end
    @test d isa StoredCompletionDeleteSuccess && d.id == "chatcmpl-1" && d.deleted
    @test _sc_targets(dseen) == [("DELETE", "/v1/chat/completions/chatcmpl-1")]
    for body in ("{\"object\":\"chat.completion.deleted\",\"id\":\"chatcmpl-1\",\"deleted\":false}",
                 "{\"object\":\"response.deleted\",\"id\":\"chatcmpl-1\",\"deleted\":true}",
                 "{\"object\":\"chat.completion.deleted\",\"id\":\"chatcmpl-1\"}")
        @test _answered(() -> delete_chat_completion("chatcmpl-1"; service=StoreProbe), 200; body) isa StoredCompletionCallError
    end
    # messages: the request's input, its parts as sent.
    msgs = Dict("object" => "list", "first_id" => "chatcmpl-1-0", "last_id" => "chatcmpl-1-1", "has_more" => false,
        "data" => [Dict("id" => "chatcmpl-1-0", "role" => "system", "content" => "Be brief.", "name" => nothing, "content_parts" => nothing),
                   Dict("id" => "chatcmpl-1-1", "role" => "user", "content" => "What colour is the circle? ", "name" => nothing,
                        "content_parts" => [Dict("type" => "text", "text" => "What colour is the circle?", "image_url" => nothing),
                                            Dict("type" => "image_url", "text" => nothing,
                                                 "image_url" => Dict("url" => "data:image/png;base64,iVBO", "detail" => "original"))])])
    m, mseen = _with_scripted((_, _) -> _json(200, JSON.json(msgs))) do
        list_chat_completion_messages("chatcmpl-1"; after="chatcmpl-1-0", limit=10, order="asc", service=StoreProbe)
    end
    @test _sc_targets(mseen) == [("GET", "/v1/chat/completions/chatcmpl-1/messages?after=chatcmpl-1-0&limit=10&order=asc")]
    @test m isa StoredCompletionMessageListSuccess && !m.response.has_more && m.response.last_id == "chatcmpl-1-1"
    s, q = m.response.data
    @test (s.id, s.role, s.content, s.content_parts, s.name) == ("chatcmpl-1-0", "system", "Be brief.", nothing, nothing)
    @test q.role == "user" && [p["type"] for p in q.content_parts] == ["text", "image_url"] && q.raw == msgs["data"][2]
    for item in (Dict("id" => "x", "role" => "user", "content_parts" => nothing),       # no content
                 Dict("id" => "x", "role" => "user", "content" => "q"),                 # no content_parts
                 Dict("role" => "user", "content" => "q", "content_parts" => nothing),  # no id
                 Dict("id" => "x", "role" => "user", "content" => "q", "content_parts" => [3]))
        body = JSON.json(Dict("object" => "list", "data" => [item], "first_id" => "x", "last_id" => "x", "has_more" => false))
        @test _answered(() -> list_chat_completion_messages("c"; service=StoreProbe), 200; body) isa StoredCompletionCallError
    end
    @test _sc_refused(() -> list_chat_completion_messages("c"; order="sideways", service=StoreSeamProbe)) isa ArgumentError
end

@testset "Stored Chat Completions: config and cancel reach every verb" begin
    for call in (cfg -> list_chat_completions(; service=StoreSeamProbe, config=cfg),
                 cfg -> retrieve_chat_completion("c"; service=StoreSeamProbe, config=cfg),
                 cfg -> update_chat_completion("c", Dict("k" => "v"); service=StoreSeamProbe, config=cfg),
                 cfg -> delete_chat_completion("c"; service=StoreSeamProbe, config=cfg),
                 cfg -> list_chat_completion_messages("c"; service=StoreSeamProbe, config=cfg))
        @test _seam_timeout(call(_TINY_DEADLINE), StoredCompletionCallError)
    end
    r, seen = _with_scripted((_, _) -> _json(200, JSON.json(_sc_item()))) do
        retrieve_chat_completion("c"; service=StoreProbe, cancel=cancel!(CancelToken()))
    end
    @test r isa StoredCompletionCallError && r.cause isa UniLMCancelled && isempty(seen)
end
