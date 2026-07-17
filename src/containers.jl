# ============================================================================
# OpenAI Containers API — /v1/containers. Sandboxed compute for the Code
# Interpreter tool; expires after idle. Files added via multipart.
# ============================================================================

"""
    ContainerObject

A code-interpreter container: `id`, `status`, and `name`; `raw` holds the
unparsed JSON response.
"""
@kwdef struct ContainerObject
    id::String
    status::Union{String,Nothing} = nothing
    name::Union{String,Nothing} = nothing
    raw::Dict{String,Any} = Dict{String,Any}()
end

"""
    ContainerList

A page of container records (raw JSON dicts) from [`list_containers`](@ref);
`has_more` signals that further pages are available.
"""
@kwdef struct ContainerList
    data::Vector{Dict{String,Any}}
    has_more::Bool = false
    raw::Dict{String,Any} = Dict{String,Any}()
end

"Successful create/retrieve/add-file result wrapping a [`ContainerObject`](@ref)."
@kwdef struct ContainerSuccess <: LLMRequestResponse; response::ContainerObject; end
"Successful [`list_containers`](@ref) result wrapping a [`ContainerList`](@ref)."
@kwdef struct ContainerListSuccess <: LLMRequestResponse; response::ContainerList; end
"Successful [`delete_container`](@ref) result; `deleted` confirms removal of `id`."
@kwdef struct ContainerDeleteSuccess <: LLMRequestResponse; id::String; deleted::Bool; end
"Containers API error result: HTTP `status` and the raw `response` body."
@kwdef struct ContainerFailure <: LLMRequestResponse; response::String; status::Int; end
"Local/transport error from a Containers API call (the request never completed)."
@kwdef struct ContainerCallError <: LLMRequestResponse; error::String; status::Union{Int,Nothing} = nothing; end

_parse_container(d::AbstractDict) = ContainerObject(id=d["id"], status=get(d, "status", nothing), name=get(d, "name", nothing), raw=Dict{String,Any}(d))
_cont_err(e) = ContainerCallError(error=string(e), status=(hasproperty(e, :status) ? e.status : nothing))
_cont_resp(resp) = resp.status == 200 ?
    ContainerSuccess(response=_parse_container(JSON.parse(resp.body; dicttype=Dict{String,Any}))) :
    ContainerFailure(response=String(resp.body), status=resp.status)

"""
    create_container(; name, file_ids=nothing, expires_after=nothing, service=OPENAIServiceEndpoint)

Pass `config::Union{Nothing,RequestConfig}` to override the timeout/retry budget for this call.
"""
function create_container(; name::String, file_ids::Union{Vector{String},Nothing}=nothing,
    expires_after::Union{AbstractDict,Nothing}=nothing, service::ServiceEndpointSpec=OPENAIServiceEndpoint,
    config::Union{Nothing,RequestConfig}=nothing)
    validate_capability(service, :containers, "Containers API")
    cfg = _resolve_config(config); t0 = time_ns()
    try
        d = Dict{Symbol,Any}(:name => name)
        !isnothing(file_ids) && (d[:file_ids] = file_ids)
        !isnothing(expires_after) && (d[:expires_after] = expires_after)
        _cont_resp(_http("POST", _api_base_url(service) * CONTAINERS_PATH, auth_header(service),
            JSON.json(d); cfg, remaining=_remaining_s(cfg, t0)))
    catch e
        e isa InterruptException && rethrow()
        _cont_err(e)
    end
end

"""
    retrieve_container(id; service=OPENAIServiceEndpoint)

Pass `config::Union{Nothing,RequestConfig}` to override the timeout/retry budget for this call.
"""
function retrieve_container(id::String; service::ServiceEndpointSpec=OPENAIServiceEndpoint, config::Union{Nothing,RequestConfig}=nothing)
    validate_capability(service, :containers, "Containers API")
    cfg = _resolve_config(config); t0 = time_ns()
    try
        _cont_resp(_http("GET", _api_base_url(service) * CONTAINERS_PATH * "/" * id, auth_header(service);
            cfg, remaining=_remaining_s(cfg, t0)))
    catch e
        e isa InterruptException && rethrow()
        _cont_err(e)
    end
end

"""
    list_containers(; limit=nothing, after=nothing, service=OPENAIServiceEndpoint)

Pass `config::Union{Nothing,RequestConfig}` to override the timeout/retry budget for this call.
"""
function list_containers(; limit::Union{Int,Nothing}=nothing, after::Union{String,Nothing}=nothing, service::ServiceEndpointSpec=OPENAIServiceEndpoint, config::Union{Nothing,RequestConfig}=nothing)
    validate_capability(service, :containers, "Containers API")
    cfg = _resolve_config(config); t0 = time_ns()
    try
        url = _api_base_url(service) * CONTAINERS_PATH
        params = String[]
        !isnothing(limit) && push!(params, "limit=$limit")
        !isnothing(after) && push!(params, "after=$after")
        !isempty(params) && (url *= "?" * join(params, "&"))
        resp = _http("GET", url, auth_header(service); cfg, remaining=_remaining_s(cfg, t0))
        resp.status == 200 || return ContainerFailure(response=String(resp.body), status=resp.status)
        data = JSON.parse(resp.body; dicttype=Dict{String,Any})
        ContainerListSuccess(response=ContainerList(data=Vector{Dict{String,Any}}(get(data, "data", [])), has_more=get(data, "has_more", false), raw=data))
    catch e
        e isa InterruptException && rethrow()
        _cont_err(e)
    end
end

"""
    delete_container(id; service=OPENAIServiceEndpoint)

Pass `config::Union{Nothing,RequestConfig}` to override the timeout/retry budget for this call.
"""
function delete_container(id::String; service::ServiceEndpointSpec=OPENAIServiceEndpoint, config::Union{Nothing,RequestConfig}=nothing)
    validate_capability(service, :containers, "Containers API")
    cfg = _resolve_config(config); t0 = time_ns()
    try
        resp = _http("DELETE", _api_base_url(service) * CONTAINERS_PATH * "/" * id, auth_header(service);
            cfg, remaining=_remaining_s(cfg, t0))
        resp.status == 200 || return ContainerFailure(response=String(resp.body), status=resp.status)
        d = JSON.parse(resp.body; dicttype=Dict{String,Any})
        ContainerDeleteSuccess(id=get(d, "id", id), deleted=get(d, "deleted", false))
    catch e
        e isa InterruptException && rethrow()
        _cont_err(e)
    end
end

"""
    add_container_file(container_id, path; service=OPENAIServiceEndpoint)  (multipart upload)

Pass `config::Union{Nothing,RequestConfig}` to override the timeout/retry budget for this call.
"""
function add_container_file(container_id::String, path::String; service::ServiceEndpointSpec=OPENAIServiceEndpoint, config::Union{Nothing,RequestConfig}=nothing)
    validate_capability(service, :containers, "Containers API")
    cfg = _resolve_config(config); t0 = time_ns()
    try
        isfile(path) || throw(ArgumentError("file not found: $path"))
        form = HTTP.Form(["file" => HTTP.Multipart(basename(path), IOBuffer(read(path)), _mime_for(path))])
        url = _api_base_url(service) * CONTAINERS_PATH * "/" * container_id * "/files"
        resp = _http("POST", url, auth_header_multipart(service), form; cfg, remaining=_remaining_s(cfg, t0))
        resp.status == 200 || return ContainerFailure(response=String(resp.body), status=resp.status)
        ContainerSuccess(response=_parse_container(JSON.parse(resp.body; dicttype=Dict{String,Any})))
    catch e
        e isa InterruptException && rethrow()
        _cont_err(e)
    end
end
