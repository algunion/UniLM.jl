# ============================================================================
# Stored Chat Completions — https://developers.openai.com/api/reference/resources/chat/subresources/completions
# A Chat Completion sent with `store=true` (Chat's `store`) is kept by OpenAI: listed, read with its input
# messages, its metadata updated, and deleted, under /v1/chat/completions.
# ============================================================================

# ─── Parsed objects ───────────────────────────────────────────────────────────

"""
    StoredCompletion

A Chat Completion OpenAI stored — one sent with `store=true` ([`Chat`](@ref)'s `store`) — as
[`get_chat_completion`](@ref), [`update_chat_completion`](@ref) and [`list_chat_completions`](@ref) return it:

- `id::String`, `model::String` (the model that served it) and `created::Int` (Unix seconds);
- `metadata::Dict{String,String}`: its metadata, empty when it has none;
- `message::Message`: the reply, its first choice's message, as [`chatrequest!`](@ref) decodes it;
- `usage::Union{TokenUsage,Nothing}`: the tokens it was billed for;
- `raw::Dict{String,Any}`: the object as sent, the request's parameters (`temperature`, `tools`, `seed`,
  `response_format`, …) among its fields.

Its input messages are read with [`list_chat_completion_messages`](@ref). A reply without the object's
`id`, `model`, `created` or a choice is no stored completion: the call returns its
[`StoredCompletionCallError`](@ref).
"""
@kwdef struct StoredCompletion
    id::String
    model::String
    created::Int
    metadata::Dict{String,String} = Dict{String,String}()
    message::Message
    usage::Union{TokenUsage,Nothing} = nothing
    raw::Dict{String,Any} = Dict{String,Any}()
end

"""
    StoredCompletionList

A page of [`StoredCompletion`](@ref)s from [`list_chat_completions`](@ref): `data`, `has_more` (a further page
follows: pass `last_id` as the next call's `after`), `first_id` and `last_id` (`nothing` on an empty page), and
`raw`, the page as sent.
"""
@kwdef struct StoredCompletionList
    data::Vector{StoredCompletion}
    has_more::Bool
    first_id::Union{String,Nothing} = nothing
    last_id::Union{String,Nothing} = nothing
    raw::Dict{String,Any} = Dict{String,Any}()
end

"""
    StoredCompletionMessage

One input message of a stored Chat Completion, from [`list_chat_completion_messages`](@ref): `id`, `role`
(`"system"`, `"developer"`, `"user"`, `"assistant"`, `"tool"` or `"function"`), `content` (its text, `nothing` for
none), `content_parts` (its parts as sent — `"text"`, `"image_url"`, `"input_audio"` or `"file"` — or `nothing`),
`name` (`nothing` for none) and `raw`, the message as sent.
"""
@kwdef struct StoredCompletionMessage
    id::String
    role::String
    content::Union{String,Nothing} = nothing
    content_parts::Union{Vector{Dict{String,Any}},Nothing} = nothing
    name::Union{String,Nothing} = nothing
    raw::Dict{String,Any} = Dict{String,Any}()
end

"""
    StoredCompletionMessageList

A page of [`StoredCompletionMessage`](@ref)s from [`list_chat_completion_messages`](@ref): `data`, `has_more`,
`first_id`, `last_id` and `raw`, as [`StoredCompletionList`](@ref).
"""
@kwdef struct StoredCompletionMessageList
    data::Vector{StoredCompletionMessage}
    has_more::Bool
    first_id::Union{String,Nothing} = nothing
    last_id::Union{String,Nothing} = nothing
    raw::Dict{String,Any} = Dict{String,Any}()
end

# ─── Result types ─────────────────────────────────────────────────────────────

"Successful [`get_chat_completion`](@ref) / [`update_chat_completion`](@ref) result wrapping a [`StoredCompletion`](@ref)."
@kwdef struct StoredCompletionSuccess <: LLMRequestResponse; response::StoredCompletion; end
"Successful [`list_chat_completions`](@ref) result wrapping a [`StoredCompletionList`](@ref)."
@kwdef struct StoredCompletionListSuccess <: LLMRequestResponse; response::StoredCompletionList; end
"Successful [`list_chat_completion_messages`](@ref) result wrapping a [`StoredCompletionMessageList`](@ref)."
@kwdef struct StoredCompletionMessageListSuccess <: LLMRequestResponse; response::StoredCompletionMessageList; end
"Successful [`delete_chat_completion`](@ref) result: the service confirmed (`deleted` is always `true`) the removal of `id`."
@kwdef struct StoredCompletionDeleteSuccess <: LLMRequestResponse; id::String; deleted::Bool; end
"Stored Chat Completions error result: HTTP `status`, the raw `response` body, and the `request_id` the service sent (`x-request-id`/`request-id` header), if any."
@kwdef struct StoredCompletionFailure <: LLMRequestResponse; response::String; status::Int; request_id::Union{String,Nothing} = nothing; end
"Stored Chat Completions call that produced no usable reply (transport failure, timeout, or a 200 that could not be decoded); `cause` is the underlying exception — a [`UniLMTimeout`](@ref) for a timeout."
@kwdef struct StoredCompletionCallError <: LLMRequestResponse; error::String; status::Union{Int,Nothing} = nothing; cause::Union{Nothing,Exception} = nothing; end

# ─── Replies ──────────────────────────────────────────────────────────────────
# A field the reference requires is never read with a default, and one of another JSON type is never coerced.

function _store_field(d::AbstractDict, key::String, ::Type{T}; nullable::Bool=false) where {T}
    haskey(d, key) || throw(ArgumentError("the stored completion reply has no $(repr(key))"))
    v = d[key]
    (nullable && isnothing(v)) || v isa T || throw(ArgumentError(
        "the stored completion reply's $(repr(key)) is a JSON $(_json_type(v)), not the documented type"))
    v
end
# A field the reference leaves optional: absent or null is none.
_store_optional(d::AbstractDict, key::String, ::Type{T}) where {T} =
    haskey(d, key) ? _store_field(d, key, T; nullable=true) : nothing
_store_object(x)::Dict{String,Any} =
    x isa Dict{String,Any} ? x : throw(ArgumentError("a stored completion reply entry is a JSON $(_json_type(x)), not an object"))

# Metadata: string pairs; null, or left out, is none.
function _store_metadata(m)::Dict{String,String}
    isnothing(m) && return Dict{String,String}()
    m isa AbstractDict && all(v -> v isa AbstractString, values(m)) || throw(ArgumentError(
        "the stored completion's metadata is not an object of strings (got $(repr(m)))"))
    Dict{String,String}(String(k) => String(v) for (k, v) in m)
end

# The object is the chat.completion the request returned; its reply decodes as chatrequest! decodes an OpenAI
# Chat Completions reply.
function _stored_completion(d::AbstractDict)::StoredCompletion
    kind = _store_field(d, "object", String)
    kind == "chat.completion" || throw(ArgumentError("the reply is a $(repr(kind)), not a stored chat.completion"))
    decoded = extract_message(HTTP.Response(200, JSON.json(d)))
    StoredCompletion(id=_store_field(d, "id", String), model=_store_field(d, "model", String),
        created=_store_field(d, "created", Int64), metadata=_store_metadata(get(d, "metadata", nothing)),
        message=decoded.message, usage=decoded.usage, raw=Dict{String,Any}(d))
end

function _stored_message(d::Dict{String,Any})::StoredCompletionMessage
    parts = _store_field(d, "content_parts", AbstractVector; nullable=true)
    StoredCompletionMessage(id=_store_field(d, "id", String), role=_store_field(d, "role", String),
        content=_store_field(d, "content", String; nullable=true),
        content_parts=isnothing(parts) ? nothing : Dict{String,Any}[_store_object(p) for p in parts],
        name=_store_optional(d, "name", String), raw=d)
end

# A page, as the reference requires it: `data`, `first_id`, `last_id` (null on an empty page) and `has_more`.
function _store_page(item::Function, d::Dict{String,Any})
    (data=[item(_store_object(x)) for x in _store_field(d, "data", AbstractVector)],
     has_more=_store_field(d, "has_more", Bool), first_id=_store_field(d, "first_id", String; nullable=true),
     last_id=_store_field(d, "last_id", String; nullable=true), raw=d)
end

# Metadata as the reference bounds it: "16 key-value pairs", "Keys are strings with a maximum length of 64
# characters. Values are strings with a maximum length of 512 characters."
function _store_metadata_arg(verb::String, metadata::AbstractDict)
    length(metadata) <= 16 || throw(ArgumentError("$verb: metadata holds at most 16 key-value pairs (got $(length(metadata)))"))
    for (k, v) in metadata
        k isa AbstractString && v isa AbstractString || throw(ArgumentError(
            "$verb: metadata keys and values are strings (got $(repr(k)) => $(repr(v)))"))
        length(k) <= 64 || throw(ArgumentError("$verb: a metadata key has at most 64 characters (got $(length(k)): $(repr(k)))"))
        length(v) <= 512 || throw(ArgumentError(
            "$verb: metadata value $(repr(k)) is a string of at most 512 characters (got $(length(v)))"))
    end
    metadata
end

const _STORE_CAP = "Stored Chat Completions"
_store_url(service, path::String="")::String = _api_base_url(service) * CHAT_COMPLETIONS_PATH * path
_store_call(decode::Function, method::String, url::String, service, config, cancel; body=nothing) =
    _vs_json_call(decode, StoredCompletionFailure, StoredCompletionCallError, method, url, service, config, cancel; body)

# ─── Verbs ────────────────────────────────────────────────────────────────────

const _STORE_CONFIG_DOC = """
Pass `config::Union{Nothing,RequestConfig}` to override the timeout budget for this call (a single bounded attempt; `max_attempts` does not apply).

Pass `cancel::Union{Nothing,CancelToken}` (default: the ambient [`with_cancel`](@ref) token) to make the call cancellable: once the token is cancelled it returns a `StoredCompletionCallError` whose `cause` is a [`UniLMCancelled`](@ref) — without sending anything when the token was already cancelled, at once when mid-request (never retried)."""

"""
    list_chat_completions(; model=nothing, metadata=nothing, after=nothing, limit=nothing, order=nothing,
                          service=OPENAIServiceEndpoint)

A page of the Chat Completions OpenAI stored (sent with `store=true`): `StoredCompletionListSuccess` (a
[`StoredCompletionList`](@ref)), `StoredCompletionFailure` or `StoredCompletionCallError`. `model` keeps those of
one model; `metadata`, a `Dict` of strings, those whose metadata holds each of its pairs (sent as
`metadata[key]=value`); `order` is `"asc"` (OpenAI's default) or `"desc"` by creation time; `after` starts the
page after that completion id (the previous page's `last_id`), and `limit` sizes it (OpenAI's default 20).
Another `order`, or metadata past the reference's bounds (16 pairs, 64-character keys, 512-character values), is
an `ArgumentError` before any request.

A completion is listed, and read, some time after its request returns: on 2026-11-17 one was still unknown
(HTTP 404) to [`get_chat_completion`](@ref) more than 15 minutes later.

$_STORE_CONFIG_DOC
"""
function list_chat_completions(; model::Union{String,Nothing}=nothing,
    metadata::Union{AbstractDict,Nothing}=nothing, after::Union{String,Nothing}=nothing,
    limit::Union{Int,Nothing}=nothing, order::Union{String,Nothing}=nothing,
    service::ServiceEndpointSpec=OPENAIServiceEndpoint, config::Union{Nothing,RequestConfig}=nothing,
    cancel::Union{Nothing,CancelToken}=nothing)
    validate_capability(service, :stored_completions, _STORE_CAP)
    query = _vs_list_query("list_chat_completions", "model" => model, "after" => after, "limit" => limit, "order" => order)
    if !isnothing(metadata)
        _store_metadata_arg("list_chat_completions", metadata)
        # The reference's deep-object filter, `metadata[k]=v`, its brackets encoded as every other query here.
        pairs = ["metadata%5B$(_uripart(k))%5D=$(_uripart(metadata[k]))" for k in sort!(collect(keys(metadata)))]
        isempty(pairs) || (query *= (isempty(query) ? "?" : "&") * join(pairs, "&"))
    end
    _store_call("GET", _store_url(service, query), service, config, cancel) do d
        StoredCompletionListSuccess(response=StoredCompletionList(; _store_page(_stored_completion, d)...))
    end
end

"""
    get_chat_completion(id; service=OPENAIServiceEndpoint)

The stored Chat Completion `id` (an [`LLMSuccess`](@ref)'s `raw` id, `chatcmpl-…`): `StoredCompletionSuccess` (a
[`StoredCompletion`](@ref)), `StoredCompletionFailure` (HTTP 404 for one not stored, deleted, or not yet
listed) or `StoredCompletionCallError`.

$_STORE_CONFIG_DOC
"""
function get_chat_completion(id::String; service::ServiceEndpointSpec=OPENAIServiceEndpoint,
    config::Union{Nothing,RequestConfig}=nothing, cancel::Union{Nothing,CancelToken}=nothing)
    validate_capability(service, :stored_completions, _STORE_CAP)
    _store_call(d -> StoredCompletionSuccess(response=_stored_completion(d)), "GET",
                _store_url(service, "/" * _uripart(id)), service, config, cancel)
end

"""
    update_chat_completion(id, metadata; service=OPENAIServiceEndpoint)

Set the metadata of the stored Chat Completion `id` — "the only supported modification" — to `metadata`, a
`Dict` of strings (at most 16 pairs, keys of at most 64 characters, values of at most 512, else `ArgumentError`
before any request): `StoredCompletionSuccess` with the updated completion, `StoredCompletionFailure` or
`StoredCompletionCallError`.

$_STORE_CONFIG_DOC
"""
function update_chat_completion(id::String, metadata::AbstractDict; service::ServiceEndpointSpec=OPENAIServiceEndpoint,
    config::Union{Nothing,RequestConfig}=nothing, cancel::Union{Nothing,CancelToken}=nothing)
    validate_capability(service, :stored_completions, _STORE_CAP)
    _store_metadata_arg("update_chat_completion", metadata)
    _store_call(d -> StoredCompletionSuccess(response=_stored_completion(d)), "POST",
                _store_url(service, "/" * _uripart(id)), service, config, cancel; body=Dict(:metadata => metadata))
end

"""
    delete_chat_completion(id; service=OPENAIServiceEndpoint)

Delete the stored Chat Completion `id`: `StoredCompletionDeleteSuccess` once the reply names the deleted
completion and confirms the delete, `StoredCompletionFailure` or `StoredCompletionCallError`.

$_STORE_CONFIG_DOC
"""
function delete_chat_completion(id::String; service::ServiceEndpointSpec=OPENAIServiceEndpoint,
    config::Union{Nothing,RequestConfig}=nothing, cancel::Union{Nothing,CancelToken}=nothing)
    validate_capability(service, :stored_completions, _STORE_CAP)
    _store_call(d -> _vs_deletion(StoredCompletionDeleteSuccess, d, "chat.completion.deleted"), "DELETE",
                _store_url(service, "/" * _uripart(id)), service, config, cancel)
end

"""
    list_chat_completion_messages(id; after=nothing, limit=nothing, order=nothing, service=OPENAIServiceEndpoint)

A page of the input messages of the stored Chat Completion `id` — the request's messages; its reply is the
[`StoredCompletion`](@ref)'s `message` —: `StoredCompletionMessageListSuccess` (a
[`StoredCompletionMessageList`](@ref)), `StoredCompletionFailure` or `StoredCompletionCallError`. `order` is
`"asc"` (OpenAI's default) or `"desc"`, else `ArgumentError` before any request; `after` and `limit` page as
[`list_chat_completions`](@ref) does.

$_STORE_CONFIG_DOC
"""
function list_chat_completion_messages(id::String; after::Union{String,Nothing}=nothing,
    limit::Union{Int,Nothing}=nothing, order::Union{String,Nothing}=nothing,
    service::ServiceEndpointSpec=OPENAIServiceEndpoint, config::Union{Nothing,RequestConfig}=nothing,
    cancel::Union{Nothing,CancelToken}=nothing)
    validate_capability(service, :stored_completions, _STORE_CAP)
    query = _vs_list_query("list_chat_completion_messages", "after" => after, "limit" => limit, "order" => order)
    _store_call("GET", _store_url(service, "/" * _uripart(id) * "/messages" * query), service, config, cancel) do d
        StoredCompletionMessageListSuccess(response=StoredCompletionMessageList(; _store_page(_stored_message, d)...))
    end
end
