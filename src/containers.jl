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

"""    create_container(; name, file_ids=nothing, expires_after=nothing, service=OPENAIServiceEndpoint)"""
function create_container(; name::String, file_ids::Union{Vector{String},Nothing}=nothing,
    expires_after::Union{AbstractDict,Nothing}=nothing, service::ServiceEndpointSpec=OPENAIServiceEndpoint)
    validate_capability(service, :containers, "Containers API")
    try
        d = Dict{Symbol,Any}(:name => name)
        !isnothing(file_ids) && (d[:file_ids] = file_ids)
        !isnothing(expires_after) && (d[:expires_after] = expires_after)
        _cont_resp(HTTP.post(_api_base_url(service) * CONTAINERS_PATH, body=JSON.json(d), headers=auth_header(service); status_exception=false))
    catch e
        _cont_err(e)
    end
end

"""    retrieve_container(id; service=OPENAIServiceEndpoint)"""
function retrieve_container(id::String; service::ServiceEndpointSpec=OPENAIServiceEndpoint)
    validate_capability(service, :containers, "Containers API")
    try
        _cont_resp(HTTP.get(_api_base_url(service) * CONTAINERS_PATH * "/" * id, headers=auth_header(service); status_exception=false))
    catch e
        _cont_err(e)
    end
end

"""    list_containers(; limit=nothing, after=nothing, service=OPENAIServiceEndpoint)"""
function list_containers(; limit::Union{Int,Nothing}=nothing, after::Union{String,Nothing}=nothing, service::ServiceEndpointSpec=OPENAIServiceEndpoint)
    validate_capability(service, :containers, "Containers API")
    try
        url = _api_base_url(service) * CONTAINERS_PATH
        params = String[]
        !isnothing(limit) && push!(params, "limit=$limit")
        !isnothing(after) && push!(params, "after=$after")
        !isempty(params) && (url *= "?" * join(params, "&"))
        resp = HTTP.get(url, headers=auth_header(service); status_exception=false)
        resp.status == 200 || return ContainerFailure(response=String(resp.body), status=resp.status)
        data = JSON.parse(resp.body; dicttype=Dict{String,Any})
        ContainerListSuccess(response=ContainerList(data=Vector{Dict{String,Any}}(get(data, "data", [])), has_more=get(data, "has_more", false), raw=data))
    catch e
        _cont_err(e)
    end
end

"""    delete_container(id; service=OPENAIServiceEndpoint)"""
function delete_container(id::String; service::ServiceEndpointSpec=OPENAIServiceEndpoint)
    validate_capability(service, :containers, "Containers API")
    try
        resp = HTTP.request("DELETE", _api_base_url(service) * CONTAINERS_PATH * "/" * id, headers=auth_header(service); status_exception=false)
        resp.status == 200 || return ContainerFailure(response=String(resp.body), status=resp.status)
        d = JSON.parse(resp.body; dicttype=Dict{String,Any})
        ContainerDeleteSuccess(id=get(d, "id", id), deleted=get(d, "deleted", false))
    catch e
        _cont_err(e)
    end
end

"""    add_container_file(container_id, path; service=OPENAIServiceEndpoint)  (multipart upload)"""
function add_container_file(container_id::String, path::String; service::ServiceEndpointSpec=OPENAIServiceEndpoint)
    validate_capability(service, :containers, "Containers API")
    try
        isfile(path) || throw(ArgumentError("file not found: $path"))
        form = HTTP.Form(["file" => HTTP.Multipart(basename(path), IOBuffer(read(path)), _mime_for(path))])
        url = _api_base_url(service) * CONTAINERS_PATH * "/" * container_id * "/files"
        resp = HTTP.post(url, auth_header_multipart(service), form; status_exception=false)
        resp.status == 200 || return ContainerFailure(response=String(resp.body), status=resp.status)
        ContainerSuccess(response=_parse_container(JSON.parse(resp.body; dicttype=Dict{String,Any})))
    catch e
        _cont_err(e)
    end
end
