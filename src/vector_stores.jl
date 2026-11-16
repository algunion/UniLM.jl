# ============================================================================
# OpenAI Vector Stores API — https://platform.openai.com/docs/api-reference/vector-stores
# Create/manage vector stores (+ their files and file batches) that power the
# Responses `file_search` tool, and search them directly. Upload files via the
# Files API first.
# ============================================================================

# ─── Parsed objects ───────────────────────────────────────────────────────────

"""
    VectorStoreObject

A vector store from the Vector Stores API: `id`, `name`, `status`, and
`file_counts`; `raw` holds the unparsed JSON response.
"""
@kwdef struct VectorStoreObject
    id::String
    name::Union{String,Nothing} = nothing
    status::Union{String,Nothing} = nothing
    file_counts::Dict{String,Any} = Dict{String,Any}()
    raw::Dict{String,Any} = Dict{String,Any}()
end

"""
    vector_store_id(v::VectorStoreObject) -> String

Return the `id` of a [`VectorStoreObject`](@ref), for use in file and batch calls.
"""
vector_store_id(v::VectorStoreObject) = v.id

"""
    VectorStoreFileObject

A file attached to a vector store: `id` and `status`; `raw` holds the
unparsed JSON response (its `attributes`, `usage_bytes`, `last_error`, …).
"""
@kwdef struct VectorStoreFileObject
    id::String
    status::Union{String,Nothing} = nothing
    raw::Dict{String,Any} = Dict{String,Any}()
end

"""
    VectorStoreFileBatch

A batch of files added to a vector store: `id`, `status`, and `file_counts`;
`raw` holds the unparsed JSON response.
"""
@kwdef struct VectorStoreFileBatch
    id::String
    status::Union{String,Nothing} = nothing
    file_counts::Dict{String,Any} = Dict{String,Any}()
    raw::Dict{String,Any} = Dict{String,Any}()
end

"""
    VectorStoreList

A page of [`VectorStoreObject`](@ref)s from [`list_vector_stores`](@ref);
`has_more` signals that further pages are available.
"""
@kwdef struct VectorStoreList
    data::Vector{VectorStoreObject}
    has_more::Bool = false
    raw::Dict{String,Any} = Dict{String,Any}()
end

"""
    VectorStoreFileList

A page of [`VectorStoreFileObject`](@ref)s from [`list_vector_store_files`](@ref) or
[`list_file_batch_files`](@ref); `has_more` signals that further pages are available.
"""
@kwdef struct VectorStoreFileList
    data::Vector{VectorStoreFileObject}
    has_more::Bool = false
    raw::Dict{String,Any} = Dict{String,Any}()
end

"""
    VectorStoreFileContent

The text OpenAI parsed from a vector store file, from [`vector_store_file_content`](@ref):
`data`, its parts as OpenAI sent them (each a `Dict` with `"type"` — `"text"`, the one type
OpenAI documents — and `"text"`), `has_more`, and `next_page`, the token of a further page
(`nothing` when there is none); `raw` holds the unparsed JSON response.
"""
@kwdef struct VectorStoreFileContent
    data::Vector{Dict{String,Any}}
    has_more::Bool = false
    next_page::Union{String,Nothing} = nothing
    raw::Dict{String,Any} = Dict{String,Any}()
end

"""
    VectorStoreSearchResult

One file a [`search_vector_store`](@ref) found: `file_id`, `filename`, `score` (the similarity,
0 to 1), the file's `attributes` (`nothing` when it has none) and `content`, the chunks that
matched as OpenAI sent them (each a `Dict` with `"type" => "text"` and `"text"`); `raw` holds the
unparsed JSON item.
"""
@kwdef struct VectorStoreSearchResult
    file_id::String
    filename::String
    score::Float64
    attributes::Union{Dict{String,Any},Nothing} = nothing
    content::Vector{Dict{String,Any}} = Dict{String,Any}[]
    raw::Dict{String,Any} = Dict{String,Any}()
end

"""
    VectorStoreSearchPage

A page of [`VectorStoreSearchResult`](@ref)s from [`search_vector_store`](@ref): `search_query`,
the queries searched (rewritten when `rewrite_query` asked for it), `data`, `has_more`, and
`next_page`, the token of a further page (`nothing` when there is none); `raw` holds the
unparsed JSON response.
"""
@kwdef struct VectorStoreSearchPage
    search_query::Vector{String}
    data::Vector{VectorStoreSearchResult}
    has_more::Bool = false
    next_page::Union{String,Nothing} = nothing
    raw::Dict{String,Any} = Dict{String,Any}()
end

"""
    VectorStoreFileEntry(; file_id, attributes=nothing, chunking_strategy=nothing)

One file of a [`create_file_batch`](@ref) that carries its own `attributes` and
`chunking_strategy`, sent as an entry of the batch's `files`. Its `attributes` are bounded as
[`add_vector_store_file`](@ref) bounds them, else `ArgumentError`.
"""
@kwdef struct VectorStoreFileEntry
    file_id::String
    attributes::Union{AbstractDict,Nothing} = nothing
    chunking_strategy::Union{AbstractDict,Nothing} = nothing
    VectorStoreFileEntry(file_id, attributes=nothing, chunking_strategy=nothing) =
        new(file_id, _vs_attributes("VectorStoreFileEntry", attributes), chunking_strategy)
end
JSON.lower(e::VectorStoreFileEntry) =
    _vs_body(:file_id => e.file_id, :attributes => e.attributes, :chunking_strategy => e.chunking_strategy)

# ─── Result types ─────────────────────────────────────────────────────────────

"Successful create/retrieve/update result wrapping a [`VectorStoreObject`](@ref)."
@kwdef struct VectorStoreSuccess <: LLMRequestResponse; response::VectorStoreObject; end
"Successful [`list_vector_stores`](@ref) result wrapping a [`VectorStoreList`](@ref)."
@kwdef struct VectorStoreListSuccess <: LLMRequestResponse; response::VectorStoreList; end
"Successful [`list_vector_store_files`](@ref) / [`list_file_batch_files`](@ref) result wrapping a [`VectorStoreFileList`](@ref)."
@kwdef struct VectorStoreFileListSuccess <: LLMRequestResponse; response::VectorStoreFileList; end
"Successful [`add_vector_store_file`](@ref), [`retrieve_vector_store_file`](@ref) or [`update_vector_store_file`](@ref) result wrapping a [`VectorStoreFileObject`](@ref)."
@kwdef struct VectorStoreFileSuccess <: LLMRequestResponse; response::VectorStoreFileObject; end
"Successful [`vector_store_file_content`](@ref) result wrapping a [`VectorStoreFileContent`](@ref)."
@kwdef struct VectorStoreContentSuccess <: LLMRequestResponse; response::VectorStoreFileContent; end
"Successful [`search_vector_store`](@ref) result wrapping a [`VectorStoreSearchPage`](@ref)."
@kwdef struct VectorStoreSearchSuccess <: LLMRequestResponse; response::VectorStoreSearchPage; end
"Successful file-batch result wrapping a [`VectorStoreFileBatch`](@ref)."
@kwdef struct VectorStoreBatchSuccess <: LLMRequestResponse; response::VectorStoreFileBatch; end
"Successful [`delete_vector_store`](@ref) or [`delete_vector_store_file`](@ref) result: the service confirmed (`deleted` is always `true`) the removal of `id`."
@kwdef struct VectorStoreDeleteSuccess <: LLMRequestResponse; id::String; deleted::Bool; end
"Vector Stores API error result: HTTP `status`, the raw `response` body, and the `request_id` the service sent (`x-request-id`/`request-id` header), if any."
@kwdef struct VectorStoreFailure <: LLMRequestResponse; response::String; status::Int; request_id::Union{String,Nothing} = nothing; end
"Vector Stores API call that produced no usable reply (transport failure, timeout, or a 200 that could not be decoded); `cause` is the underlying exception — a [`UniLMTimeout`](@ref) for a timeout. `last_observed` is set only when [`poll_file_batch`](@ref) runs out of time or is cancelled: the last [`VectorStoreFileBatch`](@ref) it saw, if any."
@kwdef struct VectorStoreCallError <: LLMRequestResponse; error::String; status::Union{Int,Nothing} = nothing; cause::Union{Nothing,Exception} = nothing; last_observed::Union{Nothing,VectorStoreFileBatch} = nothing; end

_transient(r::VectorStoreFailure) = _is_retryable(r.status, r.response)
_transient(r::VectorStoreCallError) = _per_attempt_timeout(r.cause)

# OpenAI's reference requires a store's, a batch's and a store file's `status`: a reply without
# it is malformed (a KeyError, the verb's *CallError), not one whose status is unknown — which
# would also keep `poll_file_batch` polling to its deadline.
_parse_vector_store(d::AbstractDict) = VectorStoreObject(id=d["id"], name=get(d, "name", nothing),
    status=d["status"], file_counts=Dict{String,Any}(get(d, "file_counts", Dict{String,Any}())), raw=Dict{String,Any}(d))
_parse_vs_batch(d::AbstractDict) = VectorStoreFileBatch(id=d["id"], status=d["status"],
    file_counts=Dict{String,Any}(get(d, "file_counts", Dict{String,Any}())), raw=Dict{String,Any}(d))
_parse_vs_file(d::AbstractDict) = VectorStoreFileObject(id=d["id"], status=d["status"], raw=Dict{String,Any}(d))
# Every field of a search result is required by OpenAI's reference (`attributes` may be null).
_parse_vs_hit(d::AbstractDict) = VectorStoreSearchResult(file_id=d["file_id"], filename=d["filename"],
    score=d["score"], attributes=d["attributes"], content=d["content"], raw=Dict{String,Any}(d))

# A list reply's page (containers.jl shares it): OpenAI's reference requires `data` and
# `has_more`, so a 200 without them is malformed (the verb's *CallError) rather than an empty
# last page.
_vs_page(d::AbstractDict) = (d["data"]::AbstractVector, d["has_more"]::Bool)

# A deletion record (`T` the family's delete result; containers.jl shares it): OpenAI's reference
# requires its `id` — never the one requested put in its place — and its `object`, which names
# what was deleted (`"vector_store.deleted"`, `"container.file.deleted"`, …); `_confirm_deleted`
# reads its `deleted`. The throws become the verb's *CallError.
function _vs_deletion(::Type{T}, d::AbstractDict, object::String) where {T<:LLMRequestResponse}
    d["object"] == object || throw(ArgumentError(
        "the delete reply is a $(repr(d["object"])) record, not the $(repr(object)) this delete answers with"))
    T(id=d["id"], deleted=_confirm_deleted(d))
end

# The query of a list call (containers.jl shares it): each parameter given, in the order named;
# an `order` or a file-status `filter` OpenAI's reference does not list is refused before
# anything is sent.
const _VS_FILE_STATUSES = ("in_progress", "completed", "failed", "cancelled")
function _vs_list_query(verb::String, params::Pair{String}...)::String
    parts = String[]
    for (k, v) in params
        isnothing(v) && continue
        allowed = k == "order" ? ("asc", "desc") : k == "filter" ? _VS_FILE_STATUSES : nothing
        isnothing(allowed) || v in allowed || throw(ArgumentError(
            "$verb: $k is $(join(repr.(allowed), ", ", " or ")) (got $(repr(v)))"))
        push!(parts, k * "=" * _uripart(v))
    end
    isempty(parts) ? "" : "?" * join(parts, "&")
end

# File attributes, bounded as OpenAI's reference bounds them: at most 16 pairs, each key at most
# 64 characters, each value a string of at most 512 characters, a number or a boolean.
function _vs_attributes(verb::String, attributes::Union{AbstractDict,Nothing})
    isnothing(attributes) && return nothing
    length(attributes) <= 16 || throw(ArgumentError(
        "$verb: attributes take at most 16 key-value pairs (got $(length(attributes)))"))
    for (k, v) in attributes
        key = string(k)
        length(key) <= 64 || throw(ArgumentError(
            "$verb: an attribute key has at most 64 characters (got $(length(key)): $(repr(key)))"))
        v isa Union{AbstractString,Real} || throw(ArgumentError(
            "$verb: attribute $(repr(key)) must be a string, a number or a boolean (got $(typeof(v)))"))
        v isa AbstractString && length(v) > 512 && throw(ArgumentError(
            "$verb: attribute $(repr(key)) is a string of at most 512 characters (got $(length(v)))"))
    end
    return attributes
end

# Shared HTTP wrapper: routes through the request seam (returns the raw HTTP.Response, or
# rethrows for the caller's catch). status_exception=false is imposed by _http.
function _vs_http(method::String, url::String, service, cfg::RequestConfig, remaining::Float64;
                  body::Union{String,Nothing}=nothing, cancel::Union{Nothing,CancelToken}=nothing)
    headers = auth_header(service)
    tok = _resolve_cancel(cancel)
    isnothing(body) ? _http(method, url, headers; cfg, remaining, cancel=tok) :
        _http(method, url, headers, body; cfg, remaining, cancel=tok)
end

# One JSON exchange of the Vector Stores or Containers API (`F` the family's failure, `E` its call
# error): a 200's parsed body goes to `decode`, another status is the failure, and anything
# thrown — a body that does not encode, a transport error, a timeout, a cancel, a reply that
# does not decode — the call error.
function _vs_json_call(decode::Function, ::Type{F}, ::Type{E}, method::String, url::String, service,
                       config::Union{Nothing,RequestConfig}, cancel::Union{Nothing,CancelToken};
                       body=nothing) where {F<:LLMRequestResponse,E<:LLMRequestResponse}
    cfg = _resolve_config(config); t0 = time_ns()
    try
        resp = _vs_http(method, url, service, cfg, _remaining_s(cfg, t0);
                        body=isnothing(body) ? nothing : JSON.json(body), cancel)
        resp.status == 200 ? decode(JSON.parse(resp.body; dicttype=Dict{String,Any})) : _failure(F, resp)
    catch e
        e isa InterruptException && rethrow()
        _callerr(E, e)
    end
end
_vs_call(decode::Function, method::String, path::String, service, config, cancel; body=nothing) =
    _vs_json_call(decode, VectorStoreFailure, VectorStoreCallError, method,
                  _api_base_url(service) * VECTOR_STORES_PATH * path, service, config, cancel; body)

_vs_file_success(d::AbstractDict) = VectorStoreFileSuccess(response=_parse_vs_file(d))
_vs_batch_success(d::AbstractDict) = VectorStoreBatchSuccess(response=_parse_vs_batch(d))
function _vs_file_list(d::AbstractDict)
    page, has_more = _vs_page(d)
    VectorStoreFileListSuccess(response=VectorStoreFileList(
        data=VectorStoreFileObject[_parse_vs_file(f) for f in page], has_more=has_more, raw=Dict{String,Any}(d)))
end

# A request body of the fields given, those left `nothing` out.
_vs_body(fields::Pair{Symbol}...) = Dict{Symbol,Any}(k => v for (k, v) in fields if !isnothing(v))

# ─── Requests ─────────────────────────────────────────────────────────────────

"""
    create_vector_store(; name=nothing, description=nothing, file_ids=nothing, expires_after=nothing,
                        chunking_strategy=nothing, metadata=nothing, service=OPENAIServiceEndpoint)

Create a vector store; `description` says what it is for. Returns `VectorStoreSuccess`, `VectorStoreFailure`, or `VectorStoreCallError`.

Pass `config::Union{Nothing,RequestConfig}` to override the timeout budget for this call (a single bounded attempt; `max_attempts` does not apply).

Pass `cancel::Union{Nothing,CancelToken}` (default: the ambient [`with_cancel`](@ref) token) to make the call cancellable: once the token is cancelled it returns a `VectorStoreCallError` whose `cause` is a [`UniLMCancelled`](@ref) — without sending anything when the token was already cancelled, at once when mid-request (never retried).
"""
function create_vector_store(; name::Union{String,Nothing}=nothing, description::Union{String,Nothing}=nothing,
    file_ids::Union{Vector{String},Nothing}=nothing,
    expires_after::Union{AbstractDict,Nothing}=nothing, chunking_strategy::Union{AbstractDict,Nothing}=nothing,
    metadata::Union{AbstractDict,Nothing}=nothing, service::ServiceEndpointSpec=OPENAIServiceEndpoint,
    config::Union{Nothing,RequestConfig}=nothing,
    cancel::Union{Nothing,CancelToken}=nothing)
    validate_capability(service, :vector_stores, "Vector Stores API")
    cfg = _resolve_config(config); t0 = time_ns()
    try
        d = Dict{Symbol,Any}()
        !isnothing(name) && (d[:name] = name)
        !isnothing(description) && (d[:description] = description)
        !isnothing(file_ids) && (d[:file_ids] = file_ids)
        !isnothing(expires_after) && (d[:expires_after] = expires_after)
        !isnothing(chunking_strategy) && (d[:chunking_strategy] = chunking_strategy)
        !isnothing(metadata) && (d[:metadata] = metadata)
        resp = _vs_http("POST", _api_base_url(service) * VECTOR_STORES_PATH, service, cfg, _remaining_s(cfg, t0); body=JSON.json(d), cancel)
        resp.status == 200 ? VectorStoreSuccess(response=_parse_vector_store(JSON.parse(resp.body; dicttype=Dict{String,Any}))) :
            _failure(VectorStoreFailure, resp)
    catch e
        e isa InterruptException && rethrow()
        _callerr(VectorStoreCallError, e)
    end
end

"""
    retrieve_vector_store(id; service=OPENAIServiceEndpoint)

Pass `config::Union{Nothing,RequestConfig}` to override the timeout budget for this call (a single bounded attempt; `max_attempts` does not apply).

Pass `cancel::Union{Nothing,CancelToken}` (default: the ambient [`with_cancel`](@ref) token) to make the call cancellable: once the token is cancelled it returns a `VectorStoreCallError` whose `cause` is a [`UniLMCancelled`](@ref) — without sending anything when the token was already cancelled, at once when mid-request (never retried).
"""
function retrieve_vector_store(id::String; service::ServiceEndpointSpec=OPENAIServiceEndpoint, config::Union{Nothing,RequestConfig}=nothing,
    cancel::Union{Nothing,CancelToken}=nothing)
    validate_capability(service, :vector_stores, "Vector Stores API")
    cfg = _resolve_config(config); t0 = time_ns()
    try
        resp = _vs_http("GET", _api_base_url(service) * VECTOR_STORES_PATH * "/" * _uripart(id), service, cfg, _remaining_s(cfg, t0); cancel)
        resp.status == 200 ? VectorStoreSuccess(response=_parse_vector_store(JSON.parse(resp.body; dicttype=Dict{String,Any}))) :
            _failure(VectorStoreFailure, resp)
    catch e
        e isa InterruptException && rethrow()
        _callerr(VectorStoreCallError, e)
    end
end

"""
    update_vector_store(id; name=nothing, expires_after=nothing, metadata=nothing, service=OPENAIServiceEndpoint)

Change a vector store's `name`, `expires_after` policy or `metadata`; only the fields given are
sent. Returns `VectorStoreSuccess` with the updated store, `VectorStoreFailure`, or
`VectorStoreCallError`.

Pass `config::Union{Nothing,RequestConfig}` to override the timeout budget for this call (a single bounded attempt; `max_attempts` does not apply).

Pass `cancel::Union{Nothing,CancelToken}` (default: the ambient [`with_cancel`](@ref) token) to make the call cancellable: once the token is cancelled it returns a `VectorStoreCallError` whose `cause` is a [`UniLMCancelled`](@ref) — without sending anything when the token was already cancelled, at once when mid-request (never retried).
"""
function update_vector_store(id::String; name::Union{String,Nothing}=nothing,
    expires_after::Union{AbstractDict,Nothing}=nothing, metadata::Union{AbstractDict,Nothing}=nothing,
    service::ServiceEndpointSpec=OPENAIServiceEndpoint, config::Union{Nothing,RequestConfig}=nothing,
    cancel::Union{Nothing,CancelToken}=nothing)
    validate_capability(service, :vector_stores, "Vector Stores API")
    _vs_call(d -> VectorStoreSuccess(response=_parse_vector_store(d)), "POST", "/" * _uripart(id), service, config, cancel;
             body=_vs_body(:name => name, :expires_after => expires_after, :metadata => metadata))
end

"""
    list_vector_stores(; limit=nothing, order=nothing, after=nothing, before=nothing, service=OPENAIServiceEndpoint)

List vector stores, a page at a time: `order` is `"asc"` or `"desc"` (OpenAI's default) by
creation time, and `after`/`before` the id that ends the page before or begins the page after.
Another `order` is an `ArgumentError` before anything is sent.

Pass `config::Union{Nothing,RequestConfig}` to override the timeout budget for this call (a single bounded attempt; `max_attempts` does not apply).

Pass `cancel::Union{Nothing,CancelToken}` (default: the ambient [`with_cancel`](@ref) token) to make the call cancellable: once the token is cancelled it returns a `VectorStoreCallError` whose `cause` is a [`UniLMCancelled`](@ref) — without sending anything when the token was already cancelled, at once when mid-request (never retried).
"""
function list_vector_stores(; limit::Union{Int,Nothing}=nothing, order::Union{String,Nothing}=nothing,
    after::Union{String,Nothing}=nothing, before::Union{String,Nothing}=nothing,
    service::ServiceEndpointSpec=OPENAIServiceEndpoint, config::Union{Nothing,RequestConfig}=nothing,
    cancel::Union{Nothing,CancelToken}=nothing)
    validate_capability(service, :vector_stores, "Vector Stores API")
    query = _vs_list_query("list_vector_stores", "limit" => limit, "order" => order, "after" => after, "before" => before)
    cfg = _resolve_config(config); t0 = time_ns()
    try
        url = _api_base_url(service) * VECTOR_STORES_PATH * query
        resp = _vs_http("GET", url, service, cfg, _remaining_s(cfg, t0); cancel)
        if resp.status == 200
            data = JSON.parse(resp.body; dicttype=Dict{String,Any})
            page, has_more = _vs_page(data)
            stores = VectorStoreObject[_parse_vector_store(s) for s in page]
            VectorStoreListSuccess(response=VectorStoreList(data=stores, has_more=has_more, raw=data))
        else
            _failure(VectorStoreFailure, resp)
        end
    catch e
        e isa InterruptException && rethrow()
        _callerr(VectorStoreCallError, e)
    end
end

"""
    delete_vector_store(id; service=OPENAIServiceEndpoint)

Pass `config::Union{Nothing,RequestConfig}` to override the timeout budget for this call (a single bounded attempt; `max_attempts` does not apply).

Pass `cancel::Union{Nothing,CancelToken}` (default: the ambient [`with_cancel`](@ref) token) to make the call cancellable: once the token is cancelled it returns a `VectorStoreCallError` whose `cause` is a [`UniLMCancelled`](@ref) — without sending anything when the token was already cancelled, at once when mid-request (never retried).
"""
function delete_vector_store(id::String; service::ServiceEndpointSpec=OPENAIServiceEndpoint, config::Union{Nothing,RequestConfig}=nothing,
    cancel::Union{Nothing,CancelToken}=nothing)
    validate_capability(service, :vector_stores, "Vector Stores API")
    cfg = _resolve_config(config); t0 = time_ns()
    try
        resp = _vs_http("DELETE", _api_base_url(service) * VECTOR_STORES_PATH * "/" * _uripart(id), service, cfg, _remaining_s(cfg, t0); cancel)
        if resp.status == 200
            _vs_deletion(VectorStoreDeleteSuccess, JSON.parse(resp.body; dicttype=Dict{String,Any}), "vector_store.deleted")
        else
            _failure(VectorStoreFailure, resp)
        end
    catch e
        e isa InterruptException && rethrow()
        _callerr(VectorStoreCallError, e)
    end
end

"""
    search_vector_store(vector_store_id, query; max_num_results=nothing, rewrite_query=nothing,
                        filters=nothing, ranking_options=nothing, service=OPENAIServiceEndpoint)

Search a vector store for the chunks that match `query`, a string or a vector of strings, as the
Responses `file_search` tool does: `max_num_results` (OpenAI's reference: 1 to 50, 10 by
default), `rewrite_query` to let OpenAI rewrite the query for vector search, `filters` on the
files' `attributes` (a comparison or compound filter `Dict`, as [`file_search`](@ref) takes) and
`ranking_options` (`Dict("ranker" => …, "score_threshold" => …)`), each sent as given when set.
Returns `VectorStoreSearchSuccess`, whose `response` is a [`VectorStoreSearchPage`](@ref),
`VectorStoreFailure`, or `VectorStoreCallError`.

Pass `config::Union{Nothing,RequestConfig}` to override the timeout budget for this call (a single bounded attempt; `max_attempts` does not apply).

Pass `cancel::Union{Nothing,CancelToken}` (default: the ambient [`with_cancel`](@ref) token) to make the call cancellable: once the token is cancelled it returns a `VectorStoreCallError` whose `cause` is a [`UniLMCancelled`](@ref) — without sending anything when the token was already cancelled, at once when mid-request (never retried).
"""
function search_vector_store(vs_id::String, query::Union{String,Vector{String}};
    max_num_results::Union{Int,Nothing}=nothing, rewrite_query::Union{Bool,Nothing}=nothing,
    filters::Union{AbstractDict,Nothing}=nothing, ranking_options::Union{AbstractDict,Nothing}=nothing,
    service::ServiceEndpointSpec=OPENAIServiceEndpoint, config::Union{Nothing,RequestConfig}=nothing,
    cancel::Union{Nothing,CancelToken}=nothing)
    validate_capability(service, :vector_stores, "Vector Stores API")
    body = _vs_body(:query => query, :max_num_results => max_num_results, :rewrite_query => rewrite_query,
                    :filters => filters, :ranking_options => ranking_options)
    _vs_call("POST", "/" * _uripart(vs_id) * "/search", service, config, cancel; body) do d
        # OpenAI's reference requires the page's search_query, data, has_more and next_page.
        page, has_more = _vs_page(d)
        VectorStoreSearchSuccess(response=VectorStoreSearchPage(search_query=d["search_query"],
            data=VectorStoreSearchResult[_parse_vs_hit(h) for h in page], has_more=has_more, next_page=d["next_page"],
            raw=Dict{String,Any}(d)))
    end
end

"""
    add_vector_store_file(vector_store_id, file_id; chunking_strategy=nothing, attributes=nothing,
                          service=OPENAIServiceEndpoint)

Attach an uploaded file to a vector store. `attributes` are values a search's `filters` compare:
at most 16 pairs, each key at most 64 characters, each value a string of at most 512 characters,
a number or a boolean, as OpenAI's reference bounds them; past those bounds, an `ArgumentError`
before anything is sent.

Pass `config::Union{Nothing,RequestConfig}` to override the timeout budget for this call (a single bounded attempt; `max_attempts` does not apply).

Pass `cancel::Union{Nothing,CancelToken}` (default: the ambient [`with_cancel`](@ref) token) to make the call cancellable: once the token is cancelled it returns a `VectorStoreCallError` whose `cause` is a [`UniLMCancelled`](@ref) — without sending anything when the token was already cancelled, at once when mid-request (never retried).
"""
function add_vector_store_file(vs_id::String, file_id::String; chunking_strategy::Union{AbstractDict,Nothing}=nothing,
    attributes::Union{AbstractDict,Nothing}=nothing,
    service::ServiceEndpointSpec=OPENAIServiceEndpoint, config::Union{Nothing,RequestConfig}=nothing,
    cancel::Union{Nothing,CancelToken}=nothing)
    validate_capability(service, :vector_stores, "Vector Stores API")
    _vs_attributes("add_vector_store_file", attributes)
    cfg = _resolve_config(config); t0 = time_ns()
    try
        d = Dict{Symbol,Any}(:file_id => file_id)
        !isnothing(chunking_strategy) && (d[:chunking_strategy] = chunking_strategy)
        !isnothing(attributes) && (d[:attributes] = attributes)
        resp = _vs_http("POST", _api_base_url(service) * VECTOR_STORES_PATH * "/" * _uripart(vs_id) * "/files", service, cfg, _remaining_s(cfg, t0); body=JSON.json(d), cancel)
        if resp.status == 200
            VectorStoreFileSuccess(response=_parse_vs_file(JSON.parse(resp.body; dicttype=Dict{String,Any})))
        else
            _failure(VectorStoreFailure, resp)
        end
    catch e
        e isa InterruptException && rethrow()
        _callerr(VectorStoreCallError, e)
    end
end

"""
    list_vector_store_files(vector_store_id; limit=nothing, order=nothing, after=nothing, before=nothing,
                            filter=nothing, service=OPENAIServiceEndpoint)

List the files of a vector store, a page at a time: `order` is `"asc"` or `"desc"` (OpenAI's
default) by creation time, `after`/`before` the id that ends the page before or begins the page
after, and `filter` keeps the files of one status, `"in_progress"`, `"completed"`, `"failed"` or
`"cancelled"`; another `order` or `filter` is an `ArgumentError` before anything is sent. Returns
`VectorStoreFileListSuccess`, `VectorStoreFailure`, or `VectorStoreCallError`.

Pass `config::Union{Nothing,RequestConfig}` to override the timeout budget for this call (a single bounded attempt; `max_attempts` does not apply).

Pass `cancel::Union{Nothing,CancelToken}` (default: the ambient [`with_cancel`](@ref) token) to make the call cancellable: once the token is cancelled it returns a `VectorStoreCallError` whose `cause` is a [`UniLMCancelled`](@ref) — without sending anything when the token was already cancelled, at once when mid-request (never retried).
"""
function list_vector_store_files(vs_id::String; limit::Union{Int,Nothing}=nothing, order::Union{String,Nothing}=nothing,
    after::Union{String,Nothing}=nothing, before::Union{String,Nothing}=nothing, filter::Union{String,Nothing}=nothing,
    service::ServiceEndpointSpec=OPENAIServiceEndpoint, config::Union{Nothing,RequestConfig}=nothing,
    cancel::Union{Nothing,CancelToken}=nothing)
    validate_capability(service, :vector_stores, "Vector Stores API")
    query = _vs_list_query("list_vector_store_files", "limit" => limit, "order" => order, "after" => after,
                           "before" => before, "filter" => filter)
    _vs_call(_vs_file_list, "GET", "/" * _uripart(vs_id) * "/files" * query, service, config, cancel)
end

"""
    retrieve_vector_store_file(vector_store_id, file_id; service=OPENAIServiceEndpoint)

Read a vector store file's record: its `status`, and in `raw` its `attributes`, `usage_bytes`
and `last_error`. Returns `VectorStoreFileSuccess`, `VectorStoreFailure`, or `VectorStoreCallError`.

Pass `config::Union{Nothing,RequestConfig}` to override the timeout budget for this call (a single bounded attempt; `max_attempts` does not apply).

Pass `cancel::Union{Nothing,CancelToken}` (default: the ambient [`with_cancel`](@ref) token) to make the call cancellable: once the token is cancelled it returns a `VectorStoreCallError` whose `cause` is a [`UniLMCancelled`](@ref) — without sending anything when the token was already cancelled, at once when mid-request (never retried).
"""
function retrieve_vector_store_file(vs_id::String, file_id::String; service::ServiceEndpointSpec=OPENAIServiceEndpoint,
    config::Union{Nothing,RequestConfig}=nothing, cancel::Union{Nothing,CancelToken}=nothing)
    validate_capability(service, :vector_stores, "Vector Stores API")
    _vs_call(_vs_file_success, "GET", "/" * _uripart(vs_id) * "/files/" * _uripart(file_id), service, config, cancel)
end

"""
    update_vector_store_file(vector_store_id, file_id, attributes; service=OPENAIServiceEndpoint)

Replace a vector store file's `attributes`, bounded as [`add_vector_store_file`](@ref) bounds
them; `nothing` clears them (sent as `"attributes": null`). Returns `VectorStoreFileSuccess`
with the updated file, `VectorStoreFailure`, or `VectorStoreCallError`.

Pass `config::Union{Nothing,RequestConfig}` to override the timeout budget for this call (a single bounded attempt; `max_attempts` does not apply).

Pass `cancel::Union{Nothing,CancelToken}` (default: the ambient [`with_cancel`](@ref) token) to make the call cancellable: once the token is cancelled it returns a `VectorStoreCallError` whose `cause` is a [`UniLMCancelled`](@ref) — without sending anything when the token was already cancelled, at once when mid-request (never retried).
"""
function update_vector_store_file(vs_id::String, file_id::String, attributes::Union{AbstractDict,Nothing};
    service::ServiceEndpointSpec=OPENAIServiceEndpoint, config::Union{Nothing,RequestConfig}=nothing,
    cancel::Union{Nothing,CancelToken}=nothing)
    validate_capability(service, :vector_stores, "Vector Stores API")
    _vs_attributes("update_vector_store_file", attributes)
    _vs_call(_vs_file_success, "POST", "/" * _uripart(vs_id) * "/files/" * _uripart(file_id), service, config, cancel;
             body=Dict(:attributes => attributes))
end

"""
    delete_vector_store_file(vector_store_id, file_id; service=OPENAIServiceEndpoint)

Detach a file from a vector store; the file itself stays in the Files API ([`delete_file`](@ref)
deletes it). Returns `VectorStoreDeleteSuccess` once the service confirms the removal,
`VectorStoreFailure`, or `VectorStoreCallError`.

Pass `config::Union{Nothing,RequestConfig}` to override the timeout budget for this call (a single bounded attempt; `max_attempts` does not apply).

Pass `cancel::Union{Nothing,CancelToken}` (default: the ambient [`with_cancel`](@ref) token) to make the call cancellable: once the token is cancelled it returns a `VectorStoreCallError` whose `cause` is a [`UniLMCancelled`](@ref) — without sending anything when the token was already cancelled, at once when mid-request (never retried).
"""
function delete_vector_store_file(vs_id::String, file_id::String; service::ServiceEndpointSpec=OPENAIServiceEndpoint,
    config::Union{Nothing,RequestConfig}=nothing, cancel::Union{Nothing,CancelToken}=nothing)
    validate_capability(service, :vector_stores, "Vector Stores API")
    _vs_call(d -> _vs_deletion(VectorStoreDeleteSuccess, d, "vector_store.file.deleted"), "DELETE",
             "/" * _uripart(vs_id) * "/files/" * _uripart(file_id), service, config, cancel)
end

"""
    vector_store_file_content(vector_store_id, file_id; service=OPENAIServiceEndpoint)

Read the text OpenAI parsed from a vector store file, as a `VectorStoreContentSuccess` whose
`response` is a [`VectorStoreFileContent`](@ref); else `VectorStoreFailure` or `VectorStoreCallError`.

Pass `config::Union{Nothing,RequestConfig}` to override the timeout budget for this call (a single bounded attempt; `max_attempts` does not apply).

Pass `cancel::Union{Nothing,CancelToken}` (default: the ambient [`with_cancel`](@ref) token) to make the call cancellable: once the token is cancelled it returns a `VectorStoreCallError` whose `cause` is a [`UniLMCancelled`](@ref) — without sending anything when the token was already cancelled, at once when mid-request (never retried).
"""
function vector_store_file_content(vs_id::String, file_id::String; service::ServiceEndpointSpec=OPENAIServiceEndpoint,
    config::Union{Nothing,RequestConfig}=nothing, cancel::Union{Nothing,CancelToken}=nothing)
    validate_capability(service, :vector_stores, "Vector Stores API")
    _vs_call("GET", "/" * _uripart(vs_id) * "/files/" * _uripart(file_id) * "/content", service, config, cancel) do d
        # OpenAI's reference requires the page's data, has_more and next_page.
        page, has_more = _vs_page(d)
        VectorStoreContentSuccess(response=VectorStoreFileContent(data=page, has_more=has_more, next_page=d["next_page"],
                                                                  raw=Dict{String,Any}(d)))
    end
end

"""
    create_file_batch(vector_store_id, file_ids; chunking_strategy=nothing, attributes=nothing,
                      service=OPENAIServiceEndpoint)
    create_file_batch(vector_store_id, files::Vector{VectorStoreFileEntry}; service=OPENAIServiceEndpoint)

Attach uploaded files to a vector store as one batch: `file_ids` with the `chunking_strategy` and
`attributes` (bounded as [`add_vector_store_file`](@ref) bounds them) of every file, or
[`VectorStoreFileEntry`](@ref)s, each with its own. OpenAI ignores a batch's `attributes` and
`chunking_strategy` beside entries, so that form does not take them. Returns
`VectorStoreBatchSuccess`, `VectorStoreFailure`, or `VectorStoreCallError`.

Pass `config::Union{Nothing,RequestConfig}` to override the timeout budget for this call (a single bounded attempt; `max_attempts` does not apply).

Pass `cancel::Union{Nothing,CancelToken}` (default: the ambient [`with_cancel`](@ref) token) to make the call cancellable: once the token is cancelled it returns a `VectorStoreCallError` whose `cause` is a [`UniLMCancelled`](@ref) — without sending anything when the token was already cancelled, at once when mid-request (never retried).
"""
function create_file_batch(vs_id::String, file_ids::Vector{String}; chunking_strategy::Union{AbstractDict,Nothing}=nothing,
    attributes::Union{AbstractDict,Nothing}=nothing,
    service::ServiceEndpointSpec=OPENAIServiceEndpoint, config::Union{Nothing,RequestConfig}=nothing,
    cancel::Union{Nothing,CancelToken}=nothing)
    validate_capability(service, :vector_stores, "Vector Stores API")
    _vs_attributes("create_file_batch", attributes)
    cfg = _resolve_config(config); t0 = time_ns()
    try
        d = Dict{Symbol,Any}(:file_ids => file_ids)
        !isnothing(chunking_strategy) && (d[:chunking_strategy] = chunking_strategy)
        !isnothing(attributes) && (d[:attributes] = attributes)
        resp = _vs_http("POST", _api_base_url(service) * VECTOR_STORES_PATH * "/" * _uripart(vs_id) * "/file_batches", service, cfg, _remaining_s(cfg, t0); body=JSON.json(d), cancel)
        resp.status == 200 ? VectorStoreBatchSuccess(response=_parse_vs_batch(JSON.parse(resp.body; dicttype=Dict{String,Any}))) :
            _failure(VectorStoreFailure, resp)
    catch e
        e isa InterruptException && rethrow()
        _callerr(VectorStoreCallError, e)
    end
end

function create_file_batch(vs_id::String, files::Vector{VectorStoreFileEntry};
    service::ServiceEndpointSpec=OPENAIServiceEndpoint, config::Union{Nothing,RequestConfig}=nothing,
    cancel::Union{Nothing,CancelToken}=nothing)
    validate_capability(service, :vector_stores, "Vector Stores API")
    _vs_call(_vs_batch_success, "POST", "/" * _uripart(vs_id) * "/file_batches", service, config, cancel;
             body=Dict(:files => files))
end

"""
    retrieve_file_batch(vector_store_id, batch_id; service=OPENAIServiceEndpoint)

Pass `config::Union{Nothing,RequestConfig}` to override the timeout budget for this call (a single bounded attempt; `max_attempts` does not apply).

Pass `cancel::Union{Nothing,CancelToken}` (default: the ambient [`with_cancel`](@ref) token) to make the call cancellable: once the token is cancelled it returns a `VectorStoreCallError` whose `cause` is a [`UniLMCancelled`](@ref) — without sending anything when the token was already cancelled, at once when mid-request (never retried).
"""
function retrieve_file_batch(vs_id::String, batch_id::String; service::ServiceEndpointSpec=OPENAIServiceEndpoint, config::Union{Nothing,RequestConfig}=nothing,
    cancel::Union{Nothing,CancelToken}=nothing)
    validate_capability(service, :vector_stores, "Vector Stores API")
    cfg = _resolve_config(config); t0 = time_ns()
    try
        resp = _vs_http("GET", _api_base_url(service) * VECTOR_STORES_PATH * "/" * _uripart(vs_id) * "/file_batches/" * _uripart(batch_id), service, cfg, _remaining_s(cfg, t0); cancel)
        resp.status == 200 ? VectorStoreBatchSuccess(response=_parse_vs_batch(JSON.parse(resp.body; dicttype=Dict{String,Any}))) :
            _failure(VectorStoreFailure, resp)
    catch e
        e isa InterruptException && rethrow()
        _callerr(VectorStoreCallError, e)
    end
end

"""
    cancel_file_batch(vector_store_id, batch_id; service=OPENAIServiceEndpoint)

Cancel a file batch: OpenAI stops processing its files as soon as it can. Returns
`VectorStoreBatchSuccess` with the batch as it then stands, `VectorStoreFailure`, or
`VectorStoreCallError`; [`poll_file_batch`](@ref) waits for its final status.

Pass `config::Union{Nothing,RequestConfig}` to override the timeout budget for this call (a single bounded attempt; `max_attempts` does not apply).

Pass `cancel::Union{Nothing,CancelToken}` (default: the ambient [`with_cancel`](@ref) token) to make the call cancellable: once the token is cancelled it returns a `VectorStoreCallError` whose `cause` is a [`UniLMCancelled`](@ref) — without sending anything when the token was already cancelled, at once when mid-request (never retried).
"""
function cancel_file_batch(vs_id::String, batch_id::String; service::ServiceEndpointSpec=OPENAIServiceEndpoint,
    config::Union{Nothing,RequestConfig}=nothing, cancel::Union{Nothing,CancelToken}=nothing)
    validate_capability(service, :vector_stores, "Vector Stores API")
    _vs_call(_vs_batch_success, "POST", "/" * _uripart(vs_id) * "/file_batches/" * _uripart(batch_id) * "/cancel",
             service, config, cancel)
end

"""
    list_file_batch_files(vector_store_id, batch_id; limit=nothing, order=nothing, after=nothing,
                          before=nothing, filter=nothing, service=OPENAIServiceEndpoint)

List the files of a file batch, a page at a time, with the paging and the status `filter` of
[`list_vector_store_files`](@ref). Returns `VectorStoreFileListSuccess`, `VectorStoreFailure`, or
`VectorStoreCallError`.

Pass `config::Union{Nothing,RequestConfig}` to override the timeout budget for this call (a single bounded attempt; `max_attempts` does not apply).

Pass `cancel::Union{Nothing,CancelToken}` (default: the ambient [`with_cancel`](@ref) token) to make the call cancellable: once the token is cancelled it returns a `VectorStoreCallError` whose `cause` is a [`UniLMCancelled`](@ref) — without sending anything when the token was already cancelled, at once when mid-request (never retried).
"""
function list_file_batch_files(vs_id::String, batch_id::String; limit::Union{Int,Nothing}=nothing,
    order::Union{String,Nothing}=nothing, after::Union{String,Nothing}=nothing, before::Union{String,Nothing}=nothing,
    filter::Union{String,Nothing}=nothing, service::ServiceEndpointSpec=OPENAIServiceEndpoint,
    config::Union{Nothing,RequestConfig}=nothing, cancel::Union{Nothing,CancelToken}=nothing)
    validate_capability(service, :vector_stores, "Vector Stores API")
    query = _vs_list_query("list_file_batch_files", "limit" => limit, "order" => order, "after" => after,
                           "before" => before, "filter" => filter)
    _vs_call(_vs_file_list, "GET", "/" * _uripart(vs_id) * "/file_batches/" * _uripart(batch_id) * "/files" * query,
             service, config, cancel)
end

"""
    poll_file_batch(vector_store_id, batch_id; interval=2.0, timeout=300.0, service=OPENAIServiceEndpoint)

Poll a file batch until it reaches a terminal status (`completed`/`failed`/`cancelled`) and
return that `VectorStoreBatchSuccess`. `timeout` bounds the wall-clock time of the whole
poll (`Inf` waits indefinitely); `interval` is the pause between GETs. Both must be
positive, else `ArgumentError`.

A transient failure — a status in the retryable set (408/429/500/502–504/520–524/529) or a
GET that timed out — is polled through; any other failure is returned as it came. When
the time runs out the result is a `VectorStoreCallError` with
`cause = UniLMTimeout(:deadline, …)` and `last_observed` set to the last
`VectorStoreFileBatch` seen (`nothing` if no GET succeeded).

Pass `cancel::Union{Nothing,CancelToken}` (default: the ambient [`with_cancel`](@ref)
token) to make the poll cancellable: a cancel ends it at once, mid-GET or mid-pause, with
a `VectorStoreCallError` whose `cause` is a [`UniLMCancelled`](@ref) and `last_observed`
as above; a token cancelled before the poll sends nothing.

Pass `config::Union{Nothing,RequestConfig}` to bound each GET (a single attempt; its
`total_deadline` is capped at the time left).
"""
function poll_file_batch(vs_id::String, batch_id::String; interval::Real=2.0, timeout::Real=300.0,
    service::ServiceEndpointSpec=OPENAIServiceEndpoint, config::Union{Nothing,RequestConfig}=nothing,
    cancel::Union{Nothing,CancelToken}=nothing)
    _poll(cfg -> retrieve_file_batch(vs_id, batch_id; service, config=cfg), VectorStoreBatchSuccess,
          r -> r.response.status in ("completed", "failed", "cancelled"),
          (seen, why) -> VectorStoreCallError(error=_poll_end_text("poll_file_batch", batch_id, why, seen),
                                              cause=why, last_observed=isnothing(seen) ? nothing : seen.response);
          interval, timeout, config, cancel)
end
