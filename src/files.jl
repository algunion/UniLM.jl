# ============================================================================
# OpenAI Files API — https://platform.openai.com/docs/api-reference/files
# Upload / list / retrieve / delete files and download their content. Files feed
# the Responses file_search & code_interpreter tools, the Batch API, and fine-tuning.
# ============================================================================

const _FILE_PURPOSES = ("assistants", "batch", "fine-tune", "vision", "user_data", "evals")

function _mime_for(path::AbstractString)
    ext = lowercase(splitext(path)[2])
    get(Dict(
        ".json" => "application/json", ".jsonl" => "application/jsonl",
        ".txt" => "text/plain", ".md" => "text/markdown", ".csv" => "text/csv",
        ".pdf" => "application/pdf", ".png" => "image/png", ".jpg" => "image/jpeg",
        ".jpeg" => "image/jpeg", ".webp" => "image/webp", ".gif" => "image/gif",
        ".wav" => "audio/wav", ".mp3" => "audio/mpeg", ".mpeg" => "audio/mpeg",
        ".mpga" => "audio/mpeg", ".m4a" => "audio/mp4", ".mp4" => "audio/mp4",
        ".oga" => "audio/ogg", ".ogg" => "audio/ogg", ".flac" => "audio/flac", ".webm" => "audio/webm",
    ), ext, "application/octet-stream")
end

# ─── Request type ─────────────────────────────────────────────────────────────

"""
    FileUpload(; file, purpose, service=OPENAIServiceEndpoint)

A file-upload request. `file` is a path on disk; `purpose` is one of
`"assistants"`, `"batch"`, `"fine-tune"`, `"vision"`, `"user_data"`, `"evals"`.
"""
@kwdef struct FileUpload
    service::ServiceEndpointSpec = OPENAIServiceEndpoint
    file::String
    purpose::String
    function FileUpload(service, file, purpose)
        isfile(file) || throw(ArgumentError("file not found: $file"))
        purpose in _FILE_PURPOSES || throw(ArgumentError("invalid purpose '$purpose'; expected one of $(_FILE_PURPOSES)"))
        new(service, file, purpose)
    end
end

# ─── Parsed objects ───────────────────────────────────────────────────────────

"""
    FileObject

A file stored by the Files API: `id`, `bytes`, `created_at`, `filename`,
`purpose`, and `status`; `raw` holds the unparsed JSON response.
"""
@kwdef struct FileObject
    id::String
    bytes::Int
    created_at::Int
    filename::String
    purpose::String
    status::Union{String,Nothing} = nothing
    raw::Dict{String,Any} = Dict{String,Any}()
end

"""
    FileList

A page of [`FileObject`](@ref)s from [`list_files`](@ref); `has_more` signals
that further pages are available.
"""
@kwdef struct FileList
    data::Vector{FileObject}
    has_more::Bool = false
    raw::Dict{String,Any} = Dict{String,Any}()
end

function _parse_file_object(d::AbstractDict)
    FileObject(
        id=d["id"], bytes=get(d, "bytes", 0), created_at=get(d, "created_at", 0),
        filename=get(d, "filename", ""), purpose=get(d, "purpose", ""),
        status=get(d, "status", nothing), raw=Dict{String,Any}(d))
end

# ─── Result types ─────────────────────────────────────────────────────────────

"Successful upload/retrieve result wrapping a [`FileObject`](@ref)."
@kwdef struct FileSuccess <: LLMRequestResponse; response::FileObject; end
"Successful [`list_files`](@ref) result wrapping a [`FileList`](@ref)."
@kwdef struct FileListSuccess <: LLMRequestResponse; response::FileList; end
"Successful [`file_content`](@ref) result; `content` holds the raw file bytes."
@kwdef struct FileContentSuccess <: LLMRequestResponse; content::Vector{UInt8}; end
"Successful [`delete_file`](@ref) result; `deleted` confirms removal of `id`."
@kwdef struct FileDeleteSuccess <: LLMRequestResponse; id::String; deleted::Bool; end
"Files API error result: HTTP `status` and the raw `response` body."
@kwdef struct FileFailure <: LLMRequestResponse; response::String; status::Int; end
"Local/transport error from a Files API call (the request never completed)."
@kwdef struct FileCallError <: LLMRequestResponse; error::String; status::Union{Int,Nothing} = nothing; end

_callerr(::Type{FileCallError}, e) = FileCallError(error=string(e), status=(hasproperty(e, :status) ? e.status : nothing))

# ─── Requests ─────────────────────────────────────────────────────────────────

"""
    upload_file(path, purpose; service=OPENAIServiceEndpoint) -> LLMRequestResponse
    upload_file(u::FileUpload) -> LLMRequestResponse

Upload a file (multipart/form-data). Returns `FileSuccess`, `FileFailure`, or `FileCallError`.
"""
function upload_file(u::FileUpload; retries::Int=0)
    validate_capability(u.service, :files, "Files API")
    try
        form = HTTP.Form([
            "purpose" => u.purpose,
            "file" => HTTP.Multipart(basename(u.file), IOBuffer(read(u.file)), _mime_for(u.file)),
        ])
        url = _api_base_url(u.service) * FILES_PATH
        resp = HTTP.post(url, auth_header_multipart(u.service), form; status_exception=false)
        if resp.status == 200
            return FileSuccess(response=_parse_file_object(JSON.parse(resp.body; dicttype=Dict{String,Any})))
        elseif _is_retryable(resp.status) && retries < _RETRY_MAX_ATTEMPTS
            sleep(_retry_delay(retries, resp))
            return upload_file(u; retries=retries + 1)
        else
            return FileFailure(response=String(resp.body), status=resp.status)
        end
    catch e
        return _callerr(FileCallError, e)
    end
end
upload_file(path::String, purpose::String; service::ServiceEndpointSpec=OPENAIServiceEndpoint) =
    upload_file(FileUpload(service=service, file=path, purpose=purpose))

"""
    list_files(; purpose=nothing, limit=nothing, after=nothing, service=OPENAIServiceEndpoint)

List uploaded files. Returns `FileListSuccess`, `FileFailure`, or `FileCallError`.
"""
function list_files(; purpose::Union{String,Nothing}=nothing, limit::Union{Int,Nothing}=nothing,
    after::Union{String,Nothing}=nothing, service::ServiceEndpointSpec=OPENAIServiceEndpoint)
    validate_capability(service, :files, "Files API")
    try
        url = _api_base_url(service) * FILES_PATH
        params = String[]
        !isnothing(purpose) && push!(params, "purpose=$purpose")
        !isnothing(limit) && push!(params, "limit=$limit")
        !isnothing(after) && push!(params, "after=$after")
        !isempty(params) && (url *= "?" * join(params, "&"))
        resp = HTTP.get(url, headers=auth_header(service); status_exception=false)
        if resp.status == 200
            data = JSON.parse(resp.body; dicttype=Dict{String,Any})
            files = FileObject[_parse_file_object(f) for f in get(data, "data", [])]
            return FileListSuccess(response=FileList(data=files, has_more=get(data, "has_more", false), raw=data))
        else
            return FileFailure(response=String(resp.body), status=resp.status)
        end
    catch e
        return _callerr(FileCallError, e)
    end
end

"""
    retrieve_file(file_id; service=OPENAIServiceEndpoint)

Retrieve a file's metadata. Returns `FileSuccess`, `FileFailure`, or `FileCallError`.
"""
function retrieve_file(file_id::String; service::ServiceEndpointSpec=OPENAIServiceEndpoint)
    validate_capability(service, :files, "Files API")
    try
        url = _api_base_url(service) * FILES_PATH * "/" * file_id
        resp = HTTP.get(url, headers=auth_header(service); status_exception=false)
        resp.status == 200 ?
            FileSuccess(response=_parse_file_object(JSON.parse(resp.body; dicttype=Dict{String,Any}))) :
            FileFailure(response=String(resp.body), status=resp.status)
    catch e
        _callerr(FileCallError, e)
    end
end

"""
    delete_file(file_id; service=OPENAIServiceEndpoint)

Delete a file. Returns `FileDeleteSuccess`, `FileFailure`, or `FileCallError`.
"""
function delete_file(file_id::String; service::ServiceEndpointSpec=OPENAIServiceEndpoint)
    validate_capability(service, :files, "Files API")
    try
        url = _api_base_url(service) * FILES_PATH * "/" * file_id
        resp = HTTP.request("DELETE", url, headers=auth_header(service); status_exception=false)
        if resp.status == 200
            d = JSON.parse(resp.body; dicttype=Dict{String,Any})
            return FileDeleteSuccess(id=get(d, "id", file_id), deleted=get(d, "deleted", false))
        else
            return FileFailure(response=String(resp.body), status=resp.status)
        end
    catch e
        _callerr(FileCallError, e)
    end
end

"""
    file_content(file_id; service=OPENAIServiceEndpoint)

Download a file's raw bytes. Returns `FileContentSuccess` (`.content::Vector{UInt8}`),
`FileFailure`, or `FileCallError`. Use [`save_file_content`](@ref) to write to disk.
"""
function file_content(file_id::String; service::ServiceEndpointSpec=OPENAIServiceEndpoint)
    validate_capability(service, :files, "Files API")
    try
        url = _api_base_url(service) * FILES_PATH * "/" * file_id * "/content"
        resp = HTTP.get(url, headers=auth_header(service); status_exception=false)
        resp.status == 200 ?
            FileContentSuccess(content=Vector{UInt8}(resp.body)) :
            FileFailure(response=String(resp.body), status=resp.status)
    catch e
        _callerr(FileCallError, e)
    end
end

"""
    save_file_content(r::FileContentSuccess, path) -> path

Write downloaded file bytes to `path`.
"""
function save_file_content(r::FileContentSuccess, path::String)
    open(io -> write(io, r.content), path, "w")
    path
end
