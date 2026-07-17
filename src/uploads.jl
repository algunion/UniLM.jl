# ============================================================================
# OpenAI Uploads API — /v1/uploads. Resumable multi-part uploads for large files
# (beyond the single-request Files limit). create → add parts → complete (→ FileObject).
# ============================================================================

"""
    UploadObject

A resumable upload session: `id`, `status`, `filename`, and `bytes`; once the
upload is completed, `file` holds the resulting [`FileObject`](@ref). `raw`
holds the unparsed JSON response.
"""
@kwdef struct UploadObject
    id::String
    status::Union{String,Nothing} = nothing
    filename::Union{String,Nothing} = nothing
    bytes::Union{Int,Nothing} = nothing
    file::Union{FileObject,Nothing} = nothing
    raw::Dict{String,Any} = Dict{String,Any}()
end

"""
    UploadPartObject

A single uploaded part; its `id` is passed to [`complete_upload`](@ref) to
assemble the final file. `raw` holds the unparsed JSON response.
"""
@kwdef struct UploadPartObject
    id::String
    raw::Dict{String,Any} = Dict{String,Any}()
end

"Successful create/complete/cancel result wrapping an [`UploadObject`](@ref)."
@kwdef struct UploadSuccess <: LLMRequestResponse; response::UploadObject; end
"Successful [`add_upload_part`](@ref) result wrapping an [`UploadPartObject`](@ref)."
@kwdef struct UploadPartSuccess <: LLMRequestResponse; response::UploadPartObject; end
"Uploads API error result: HTTP `status` and the raw `response` body."
@kwdef struct UploadFailure <: LLMRequestResponse; response::String; status::Int; end
"Local/transport error from an Uploads API call (the request never completed)."
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

"""
    create_upload(; filename, purpose, bytes, mime_type, service=OPENAIServiceEndpoint)

Pass `config::Union{Nothing,RequestConfig}` to override the timeout/retry budget for this call.
"""
function create_upload(; filename::String, purpose::String, bytes::Int, mime_type::String, service::ServiceEndpointSpec=OPENAIServiceEndpoint, config::Union{Nothing,RequestConfig}=nothing)
    validate_capability(service, :uploads, "Uploads API")
    cfg = _resolve_config(config); t0 = time_ns()
    try
        d = Dict{Symbol,Any}(:filename => filename, :purpose => purpose, :bytes => bytes, :mime_type => mime_type)
        _upl_resp(_http("POST", _api_base_url(service) * UPLOADS_PATH, auth_header(service),
            JSON.json(d); cfg, remaining=_remaining_s(cfg, t0)))
    catch e
        e isa InterruptException && rethrow()
        _upl_err(e)
    end
end

"""
    add_upload_part(upload_id, data::Vector{UInt8}; service=OPENAIServiceEndpoint)  (multipart)

Pass `config::Union{Nothing,RequestConfig}` to override the timeout/retry budget for this call.
"""
function add_upload_part(upload_id::String, data::Vector{UInt8}; service::ServiceEndpointSpec=OPENAIServiceEndpoint, config::Union{Nothing,RequestConfig}=nothing)
    validate_capability(service, :uploads, "Uploads API")
    cfg = _resolve_config(config); t0 = time_ns()
    try
        form = HTTP.Form(["data" => HTTP.Multipart("part", IOBuffer(data), "application/octet-stream")])
        url = _api_base_url(service) * UPLOADS_PATH * "/" * upload_id * "/parts"
        resp = _http("POST", url, auth_header_multipart(service), form; cfg, remaining=_remaining_s(cfg, t0))
        resp.status == 200 || return UploadFailure(response=String(resp.body), status=resp.status)
        d = JSON.parse(resp.body; dicttype=Dict{String,Any})
        UploadPartSuccess(response=UploadPartObject(id=d["id"], raw=Dict{String,Any}(d)))
    catch e
        e isa InterruptException && rethrow()
        _upl_err(e)
    end
end

"""
    complete_upload(upload_id, part_ids; md5=nothing, service=OPENAIServiceEndpoint)

Pass `config::Union{Nothing,RequestConfig}` to override the timeout/retry budget for this call.
"""
function complete_upload(upload_id::String, part_ids::Vector{String}; md5::Union{String,Nothing}=nothing, service::ServiceEndpointSpec=OPENAIServiceEndpoint, config::Union{Nothing,RequestConfig}=nothing)
    validate_capability(service, :uploads, "Uploads API")
    cfg = _resolve_config(config); t0 = time_ns()
    try
        d = Dict{Symbol,Any}(:part_ids => part_ids)
        !isnothing(md5) && (d[:md5] = md5)
        _upl_resp(_http("POST", _api_base_url(service) * UPLOADS_PATH * "/" * upload_id * "/complete",
            auth_header(service), JSON.json(d); cfg, remaining=_remaining_s(cfg, t0)))
    catch e
        e isa InterruptException && rethrow()
        _upl_err(e)
    end
end

"""
    cancel_upload(upload_id; service=OPENAIServiceEndpoint)

Pass `config::Union{Nothing,RequestConfig}` to override the timeout/retry budget for this call.
"""
function cancel_upload(upload_id::String; service::ServiceEndpointSpec=OPENAIServiceEndpoint, config::Union{Nothing,RequestConfig}=nothing)
    validate_capability(service, :uploads, "Uploads API")
    cfg = _resolve_config(config); t0 = time_ns()
    try
        _upl_resp(_http("POST", _api_base_url(service) * UPLOADS_PATH * "/" * upload_id * "/cancel",
            auth_header(service); cfg, remaining=_remaining_s(cfg, t0)))
    catch e
        e isa InterruptException && rethrow()
        _upl_err(e)
    end
end
