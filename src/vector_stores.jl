# ============================================================================
# OpenAI Vector Stores API — https://platform.openai.com/docs/api-reference/vector-stores
# Create/manage vector stores (+ their files and file batches) that power the
# Responses `file_search` tool. Upload files via the Files API first.
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
unparsed JSON response.
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

# ─── Result types ─────────────────────────────────────────────────────────────

"Successful create/retrieve result wrapping a [`VectorStoreObject`](@ref)."
@kwdef struct VectorStoreSuccess <: LLMRequestResponse; response::VectorStoreObject; end
"Successful [`list_vector_stores`](@ref) result wrapping a [`VectorStoreList`](@ref)."
@kwdef struct VectorStoreListSuccess <: LLMRequestResponse; response::VectorStoreList; end
"Successful [`add_vector_store_file`](@ref) result wrapping a [`VectorStoreFileObject`](@ref)."
@kwdef struct VectorStoreFileSuccess <: LLMRequestResponse; response::VectorStoreFileObject; end
"Successful file-batch result wrapping a [`VectorStoreFileBatch`](@ref)."
@kwdef struct VectorStoreBatchSuccess <: LLMRequestResponse; response::VectorStoreFileBatch; end
"Successful [`delete_vector_store`](@ref) result; `deleted` confirms removal of `id`."
@kwdef struct VectorStoreDeleteSuccess <: LLMRequestResponse; id::String; deleted::Bool; end
"Vector Stores API error result: HTTP `status` and the raw `response` body."
@kwdef struct VectorStoreFailure <: LLMRequestResponse; response::String; status::Int; end
"Local/transport error from a Vector Stores API call (the request never completed)."
@kwdef struct VectorStoreCallError <: LLMRequestResponse; error::String; status::Union{Int,Nothing} = nothing; end

_parse_vector_store(d::AbstractDict) = VectorStoreObject(id=d["id"], name=get(d, "name", nothing),
    status=get(d, "status", nothing), file_counts=Dict{String,Any}(get(d, "file_counts", Dict{String,Any}())), raw=Dict{String,Any}(d))
_parse_vs_batch(d::AbstractDict) = VectorStoreFileBatch(id=d["id"], status=get(d, "status", nothing),
    file_counts=Dict{String,Any}(get(d, "file_counts", Dict{String,Any}())), raw=Dict{String,Any}(d))

# Shared HTTP wrapper: returns the raw HTTP.Response (or rethrows for the caller's catch).
function _vs_http(method::String, url::String, service; body::Union{String,Nothing}=nothing)
    headers = auth_header(service)
    isnothing(body) ? HTTP.request(method, url, headers; status_exception=false) :
        HTTP.request(method, url, headers, body; status_exception=false)
end

# ─── Requests ─────────────────────────────────────────────────────────────────

"""
    create_vector_store(; name=nothing, file_ids=nothing, expires_after=nothing,
                        chunking_strategy=nothing, metadata=nothing, service=OPENAIServiceEndpoint)

Create a vector store. Returns `VectorStoreSuccess`, `VectorStoreFailure`, or `VectorStoreCallError`.
"""
function create_vector_store(; name::Union{String,Nothing}=nothing, file_ids::Union{Vector{String},Nothing}=nothing,
    expires_after::Union{AbstractDict,Nothing}=nothing, chunking_strategy::Union{AbstractDict,Nothing}=nothing,
    metadata::Union{AbstractDict,Nothing}=nothing, service::ServiceEndpointSpec=OPENAIServiceEndpoint)
    validate_capability(service, :vector_stores, "Vector Stores API")
    try
        d = Dict{Symbol,Any}()
        !isnothing(name) && (d[:name] = name)
        !isnothing(file_ids) && (d[:file_ids] = file_ids)
        !isnothing(expires_after) && (d[:expires_after] = expires_after)
        !isnothing(chunking_strategy) && (d[:chunking_strategy] = chunking_strategy)
        !isnothing(metadata) && (d[:metadata] = metadata)
        resp = _vs_http("POST", _api_base_url(service) * VECTOR_STORES_PATH, service; body=JSON.json(d))
        resp.status == 200 ? VectorStoreSuccess(response=_parse_vector_store(JSON.parse(resp.body; dicttype=Dict{String,Any}))) :
            VectorStoreFailure(response=String(resp.body), status=resp.status)
    catch e
        VectorStoreCallError(error=string(e), status=(hasproperty(e, :status) ? e.status : nothing))
    end
end

"""
    retrieve_vector_store(id; service=OPENAIServiceEndpoint)
"""
function retrieve_vector_store(id::String; service::ServiceEndpointSpec=OPENAIServiceEndpoint)
    validate_capability(service, :vector_stores, "Vector Stores API")
    try
        resp = _vs_http("GET", _api_base_url(service) * VECTOR_STORES_PATH * "/" * id, service)
        resp.status == 200 ? VectorStoreSuccess(response=_parse_vector_store(JSON.parse(resp.body; dicttype=Dict{String,Any}))) :
            VectorStoreFailure(response=String(resp.body), status=resp.status)
    catch e
        VectorStoreCallError(error=string(e), status=(hasproperty(e, :status) ? e.status : nothing))
    end
end

"""
    list_vector_stores(; limit=nothing, after=nothing, service=OPENAIServiceEndpoint)
"""
function list_vector_stores(; limit::Union{Int,Nothing}=nothing, after::Union{String,Nothing}=nothing,
    service::ServiceEndpointSpec=OPENAIServiceEndpoint)
    validate_capability(service, :vector_stores, "Vector Stores API")
    try
        url = _api_base_url(service) * VECTOR_STORES_PATH
        params = String[]
        !isnothing(limit) && push!(params, "limit=$limit")
        !isnothing(after) && push!(params, "after=$after")
        !isempty(params) && (url *= "?" * join(params, "&"))
        resp = _vs_http("GET", url, service)
        if resp.status == 200
            data = JSON.parse(resp.body; dicttype=Dict{String,Any})
            stores = VectorStoreObject[_parse_vector_store(s) for s in get(data, "data", [])]
            VectorStoreListSuccess(response=VectorStoreList(data=stores, has_more=get(data, "has_more", false), raw=data))
        else
            VectorStoreFailure(response=String(resp.body), status=resp.status)
        end
    catch e
        VectorStoreCallError(error=string(e), status=(hasproperty(e, :status) ? e.status : nothing))
    end
end

"""
    delete_vector_store(id; service=OPENAIServiceEndpoint)
"""
function delete_vector_store(id::String; service::ServiceEndpointSpec=OPENAIServiceEndpoint)
    validate_capability(service, :vector_stores, "Vector Stores API")
    try
        resp = _vs_http("DELETE", _api_base_url(service) * VECTOR_STORES_PATH * "/" * id, service)
        if resp.status == 200
            d = JSON.parse(resp.body; dicttype=Dict{String,Any})
            VectorStoreDeleteSuccess(id=get(d, "id", id), deleted=get(d, "deleted", false))
        else
            VectorStoreFailure(response=String(resp.body), status=resp.status)
        end
    catch e
        VectorStoreCallError(error=string(e), status=(hasproperty(e, :status) ? e.status : nothing))
    end
end

"""
    add_vector_store_file(vector_store_id, file_id; chunking_strategy=nothing, service=OPENAIServiceEndpoint)
"""
function add_vector_store_file(vs_id::String, file_id::String; chunking_strategy::Union{AbstractDict,Nothing}=nothing,
    service::ServiceEndpointSpec=OPENAIServiceEndpoint)
    validate_capability(service, :vector_stores, "Vector Stores API")
    try
        d = Dict{Symbol,Any}(:file_id => file_id)
        !isnothing(chunking_strategy) && (d[:chunking_strategy] = chunking_strategy)
        resp = _vs_http("POST", _api_base_url(service) * VECTOR_STORES_PATH * "/" * vs_id * "/files", service; body=JSON.json(d))
        if resp.status == 200
            f = JSON.parse(resp.body; dicttype=Dict{String,Any})
            VectorStoreFileSuccess(response=VectorStoreFileObject(id=f["id"], status=get(f, "status", nothing), raw=Dict{String,Any}(f)))
        else
            VectorStoreFailure(response=String(resp.body), status=resp.status)
        end
    catch e
        VectorStoreCallError(error=string(e), status=(hasproperty(e, :status) ? e.status : nothing))
    end
end

"""
    create_file_batch(vector_store_id, file_ids; chunking_strategy=nothing, service=OPENAIServiceEndpoint)
"""
function create_file_batch(vs_id::String, file_ids::Vector{String}; chunking_strategy::Union{AbstractDict,Nothing}=nothing,
    service::ServiceEndpointSpec=OPENAIServiceEndpoint)
    validate_capability(service, :vector_stores, "Vector Stores API")
    try
        d = Dict{Symbol,Any}(:file_ids => file_ids)
        !isnothing(chunking_strategy) && (d[:chunking_strategy] = chunking_strategy)
        resp = _vs_http("POST", _api_base_url(service) * VECTOR_STORES_PATH * "/" * vs_id * "/file_batches", service; body=JSON.json(d))
        resp.status == 200 ? VectorStoreBatchSuccess(response=_parse_vs_batch(JSON.parse(resp.body; dicttype=Dict{String,Any}))) :
            VectorStoreFailure(response=String(resp.body), status=resp.status)
    catch e
        VectorStoreCallError(error=string(e), status=(hasproperty(e, :status) ? e.status : nothing))
    end
end

"""
    retrieve_file_batch(vector_store_id, batch_id; service=OPENAIServiceEndpoint)
"""
function retrieve_file_batch(vs_id::String, batch_id::String; service::ServiceEndpointSpec=OPENAIServiceEndpoint)
    validate_capability(service, :vector_stores, "Vector Stores API")
    try
        resp = _vs_http("GET", _api_base_url(service) * VECTOR_STORES_PATH * "/" * vs_id * "/file_batches/" * batch_id, service)
        resp.status == 200 ? VectorStoreBatchSuccess(response=_parse_vs_batch(JSON.parse(resp.body; dicttype=Dict{String,Any}))) :
            VectorStoreFailure(response=String(resp.body), status=resp.status)
    catch e
        VectorStoreCallError(error=string(e), status=(hasproperty(e, :status) ? e.status : nothing))
    end
end

"""
    poll_file_batch(vector_store_id, batch_id; interval=2.0, timeout=300.0, service=OPENAIServiceEndpoint)

Poll a file batch until it reaches a terminal status (`completed`/`failed`/`cancelled`) or the
timeout elapses. Returns the terminal `VectorStoreBatchSuccess`, or a `VectorStoreCallError` on timeout.
"""
function poll_file_batch(vs_id::String, batch_id::String; interval::Real=2.0, timeout::Real=300.0,
    service::ServiceEndpointSpec=OPENAIServiceEndpoint)
    max_iters = max(1, ceil(Int, timeout / interval))
    for _ in 1:max_iters
        r = retrieve_file_batch(vs_id, batch_id; service=service)
        r isa VectorStoreBatchSuccess || return r
        r.response.status in ("completed", "failed", "cancelled") && return r
        sleep(interval)
    end
    VectorStoreCallError(error="poll_file_batch timed out after $(timeout)s", status=nothing)
end
