# ============================================================================
# OpenAI Uploads API — /v1/uploads. Resumable multi-part uploads for large files
# (beyond the single-request Files limit). create → add parts → complete (→ FileObject).
# ============================================================================

@kwdef struct UploadObject
    id::String
    status::Union{String,Nothing} = nothing
    filename::Union{String,Nothing} = nothing
    bytes::Union{Int,Nothing} = nothing
    file::Union{FileObject,Nothing} = nothing
    raw::Dict{String,Any} = Dict{String,Any}()
end

@kwdef struct UploadPartObject
    id::String
    raw::Dict{String,Any} = Dict{String,Any}()
end

@kwdef struct UploadSuccess <: LLMRequestResponse; response::UploadObject; end
@kwdef struct UploadPartSuccess <: LLMRequestResponse; response::UploadPartObject; end
@kwdef struct UploadFailure <: LLMRequestResponse; response::String; status::Int; end
@kwdef struct UploadCallError <: LLMRequestResponse; error::String; status::Union{Int,Nothing} = nothing; end

function _parse_upload(d::AbstractDict)
    f = get(d, "file", nothing)
    UploadObject(id=d["id"], status=get(d, "status", nothing), filename=get(d, "filename", nothing),
        bytes=get(d, "bytes", nothing), file=(f isa AbstractDict ? _parse_file_object(f) : nothing), raw=Dict{String,Any}(d))
end
_upl_err(e) = UploadCallError(error=string(e), status=(hasproperty(e, :status) ? e.status : nothing))
_upl_resp(resp) = resp.status == 200 ?
    UploadSuccess(response=_parse_upload(JSON.parse(resp.body; dicttype=Dict{String,Any}))) :
    UploadFailure(response=String(resp.body), status=resp.status)

"""    create_upload(; filename, purpose, bytes, mime_type, service=OPENAIServiceEndpoint)"""
function create_upload(; filename::String, purpose::String, bytes::Int, mime_type::String, service::ServiceEndpointSpec=OPENAIServiceEndpoint)
    validate_capability(service, :uploads, "Uploads API")
    try
        d = Dict{Symbol,Any}(:filename => filename, :purpose => purpose, :bytes => bytes, :mime_type => mime_type)
        _upl_resp(HTTP.post(_api_base_url(service) * UPLOADS_PATH, body=JSON.json(d), headers=auth_header(service); status_exception=false))
    catch e
        _upl_err(e)
    end
end

"""    add_upload_part(upload_id, data::Vector{UInt8}; service=OPENAIServiceEndpoint)  (multipart)"""
function add_upload_part(upload_id::String, data::Vector{UInt8}; service::ServiceEndpointSpec=OPENAIServiceEndpoint)
    validate_capability(service, :uploads, "Uploads API")
    try
        form = HTTP.Form(["data" => HTTP.Multipart("part", IOBuffer(data), "application/octet-stream")])
        url = _api_base_url(service) * UPLOADS_PATH * "/" * upload_id * "/parts"
        resp = HTTP.post(url, auth_header_multipart(service), form; status_exception=false)
        resp.status == 200 || return UploadFailure(response=String(resp.body), status=resp.status)
        d = JSON.parse(resp.body; dicttype=Dict{String,Any})
        UploadPartSuccess(response=UploadPartObject(id=d["id"], raw=Dict{String,Any}(d)))
    catch e
        _upl_err(e)
    end
end

"""    complete_upload(upload_id, part_ids; md5=nothing, service=OPENAIServiceEndpoint)"""
function complete_upload(upload_id::String, part_ids::Vector{String}; md5::Union{String,Nothing}=nothing, service::ServiceEndpointSpec=OPENAIServiceEndpoint)
    validate_capability(service, :uploads, "Uploads API")
    try
        d = Dict{Symbol,Any}(:part_ids => part_ids)
        !isnothing(md5) && (d[:md5] = md5)
        _upl_resp(HTTP.post(_api_base_url(service) * UPLOADS_PATH * "/" * upload_id * "/complete", body=JSON.json(d), headers=auth_header(service); status_exception=false))
    catch e
        _upl_err(e)
    end
end

"""    cancel_upload(upload_id; service=OPENAIServiceEndpoint)"""
function cancel_upload(upload_id::String; service::ServiceEndpointSpec=OPENAIServiceEndpoint)
    validate_capability(service, :uploads, "Uploads API")
    try
        _upl_resp(HTTP.post(_api_base_url(service) * UPLOADS_PATH * "/" * upload_id * "/cancel", headers=auth_header(service); status_exception=false))
    catch e
        _upl_err(e)
    end
end
