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

# ─── Shared by the platform API files ────────────────────────────────────────
# Every platform result family has the same three shapes: `*Success`, `*Failure` (a
# non-200 reply: body, status, request id) and `*CallError` (no usable reply: the
# rendered error, any carried status, and the exception itself). One constructor path
# per shape keeps those diagnostics identical across the families.

# The id the provider assigned to one exchange, the handle a support report needs.
# OpenAI sends `x-request-id`; other OpenAI-wire servers send `request-id`.
function _platform_request_id(resp::HTTP.Response)::Union{Nothing,String}
    for name in ("x-request-id", "request-id")
        v = HTTP.header(resp, name, "")
        isempty(v) || return String(v)
    end
    return nothing
end
_platform_request_id(::Nothing) = nothing

_failure(::Type{T}, resp::HTTP.Response) where {T<:LLMRequestResponse} =
    T(response=String(resp.body), status=resp.status, request_id=_platform_request_id(resp))

# `cause` is the ROOT exception — the one `error` renders — so a caller can dispatch
# on it (a `UniLMTimeout` carries phase, elapsed and limit); a transport wrapper
# would add nothing but the request it was carrying.
function _callerr(::Type{T}, e; kw...) where {T<:LLMRequestResponse}
    root = _unwrap_exception(e)
    T(; error=_error_text(e), status=(hasproperty(e, :status) ? e.status : nothing),
      cause=(root isa Exception ? root : nothing), kw...)
end

# Worth another poll rather than being the answer: a status in the seam's retryable
# set, or a GET that ran out its own time. Each polled result family defines its
# `_transient` methods beside its types.
_per_attempt_timeout(e)::Bool = e isa UniLMTimeout && e.phase in (:connect, :request)

"""
    _poll(fetch, S, terminal, ended; interval, timeout, config, cancel)

Call `fetch(cfg)` until a success (an `S`) satisfies `terminal`, a non-transient
failure comes back (returned as is), `timeout` seconds of wall-clock time pass, or
`cancel` (default: the ambient token) is cancelled — then
`ended(last_success_or_nothing, cause)` builds the result, `cause` being a
`UniLMTimeout(:deadline, …)` or a `UniLMCancelled`. Each GET's `total_deadline` and
each pause are capped at the time left, so the poll ends within `timeout` rather than
one GET or pause past it. The GETs run with the token ambient and a pause wakes at once
on a cancel, so a cancel ends the poll mid-GET or mid-pause alike.
"""
function _poll(fetch::Function, ::Type{S}, terminal::Function, ended::Function;
               interval::Real, timeout::Real, config::Union{Nothing,RequestConfig},
               cancel::Union{Nothing,CancelToken}) where {S<:LLMRequestResponse}
    (isfinite(interval) && interval > 0) ||
        throw(ArgumentError("interval must be a finite number of seconds > 0 (got $interval)"))
    limit = _validated_timeout(:timeout, timeout)
    cfg = _resolve_config(config); t0 = time_ns()
    tok = _resolve_cancel(cancel)
    seen = nothing
    while (left = limit - _elapsed_s(t0)) > 0
        c = RequestConfig(cfg; total_deadline=min(cfg.total_deadline, left))
        r = isnothing(tok) ? fetch(c) : with_cancel(() -> fetch(c), tok)
        if r isa S
            terminal(r) && return r
            seen = r
        elseif !_transient(r) && !iscancelled(tok)
            return r
        end
        _cancel_sleep(tok, min(interval, max(limit - _elapsed_s(t0), 0.0))) &&
            return ended(seen, UniLMCancelled(:token, _elapsed_s(t0)))
    end
    return ended(seen, UniLMTimeout(:deadline, _elapsed_s(t0), limit))
end

# A delete counts as done only when the reply says so: the documented body carries
# `"deleted": true`, and one without it (or with `false`) leaves the object in place as
# far as the caller can know. The throw becomes the verb's *CallError.
function _confirm_deleted(d::AbstractDict)::Bool
    haskey(d, "deleted") || throw(ArgumentError(
        "the delete reply has no \"deleted\" field, so the removal is unconfirmed"))
    d["deleted"] === true || throw(ArgumentError(
        "the service did not confirm the delete (\"deleted\": $(repr(d["deleted"])))"))
    return true
end

# Replace `path` with `bytes` atomically: write a temporary file beside the destination
# (same filesystem, so the final rename cannot degrade into a copy) and rename it over
# the destination. A write that fails part-way leaves the old contents, not a truncated
# file. As opening `path` for writing would, the save keeps an existing file's
# permission bits (a file made private stays private) and writes through a symlink, to
# a target a dangling link creates. A directory is refused, because
# `mv(...; force=true)` would delete it recursively.
function _atomic_write(path::String, bytes::AbstractVector{UInt8})::String
    dest = _write_target(path)
    isdir(dest) && throw(ArgumentError("cannot save over a directory: $path"))
    tmp = tempname(dirname(abspath(dest)); cleanup=false)
    try
        write(tmp, bytes)
        isfile(dest) && chmod(tmp, filemode(dest) & 0o7777)
        mv(tmp, dest; force=true)
    finally
        rm(tmp; force=true)   # already gone after a successful rename
    end
    return path
end

# The file a write to `path` lands in: `path`, or the end of its symlink chain, which
# need not exist yet. A loop, or a chain past the system's limit, makes `ispath` throw
# (ELOOP) before this recursion could.
function _write_target(path::String)::String
    ispath(path) && return realpath(path)
    islink(path) || return path
    target = readlink(path)
    _write_target(isabspath(target) ? target : joinpath(dirname(path), target))
end

# The error text of a poll that ran out of time or was cancelled.
_poll_end_text(verb::String, id::String, to::UniLMTimeout, seen)::String =
    "$verb timeout: $id reached no terminal status within $(to.limit) s" * _last_status(seen)
_poll_end_text(verb::String, id::String, c::UniLMCancelled, seen)::String =
    "$verb cancelled after $(round(c.elapsed; digits=3)) s: $id" * _last_status(seen)
_last_status(seen)::String =
    " (last observed status: $(isnothing(seen) ? "none" : something(seen.response.status, "none")))"

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
"Successful [`delete_file`](@ref) result: the service confirmed (`deleted` is always `true`) the removal of `id`."
@kwdef struct FileDeleteSuccess <: LLMRequestResponse; id::String; deleted::Bool; end
"Files API error result: HTTP `status`, the raw `response` body, and the `request_id` the service sent (`x-request-id`/`request-id` header), if any."
@kwdef struct FileFailure <: LLMRequestResponse; response::String; status::Int; request_id::Union{String,Nothing} = nothing; end
"Files API call that produced no usable reply (transport failure, timeout, or a 200 that could not be decoded); `cause` is the underlying exception — a [`UniLMTimeout`](@ref) for a timeout."
@kwdef struct FileCallError <: LLMRequestResponse; error::String; status::Union{Int,Nothing} = nothing; cause::Union{Nothing,Exception} = nothing; end

# ─── Requests ─────────────────────────────────────────────────────────────────

"""
    upload_file(path, purpose; service=OPENAIServiceEndpoint) -> LLMRequestResponse
    upload_file(u::FileUpload) -> LLMRequestResponse

Upload a file (multipart/form-data). Returns `FileSuccess`, `FileFailure`, or `FileCallError`.

Pass `config::Union{Nothing,RequestConfig}` to override the timeout budget for this call
(a single bounded attempt; `max_attempts` does not apply). Like every other create, an
upload is never retried: a POST that timed out, or drew a gateway 5xx after the backend
stored the file, may still have created it, and a second attempt would store it twice.
A `FileFailure`/`FileCallError` therefore does not prove that no file was created.
"""
function upload_file(u::FileUpload; config::Union{Nothing,RequestConfig}=nothing)
    validate_capability(u.service, :files, "Files API")
    cfg = _resolve_config(config)
    t0 = time_ns()
    try
        form = HTTP.Form(["purpose" => u.purpose,
            "file" => HTTP.Multipart(basename(u.file), IOBuffer(read(u.file)), _mime_for(u.file))])
        resp = _http("POST", _api_base_url(u.service) * FILES_PATH, auth_header_multipart(u.service), form;
                     cfg, remaining=_remaining_s(cfg, t0))
        return resp.status == 200 ?
               FileSuccess(response=_parse_file_object(JSON.parse(resp.body; dicttype=Dict{String,Any}))) :
               _failure(FileFailure, resp)
    catch e
        e isa InterruptException && rethrow()
        return _callerr(FileCallError, e)
    end
end
upload_file(path::String, purpose::String; service::ServiceEndpointSpec=OPENAIServiceEndpoint,
    config::Union{Nothing,RequestConfig}=nothing) =
    upload_file(FileUpload(service=service, file=path, purpose=purpose); config=config)

"""
    list_files(; purpose=nothing, limit=nothing, after=nothing, service=OPENAIServiceEndpoint)

List uploaded files. Returns `FileListSuccess`, `FileFailure`, or `FileCallError`.

Pass `config::Union{Nothing,RequestConfig}` to override the timeout budget for this call (a single bounded attempt; `max_attempts` does not apply).
"""
function list_files(; purpose::Union{String,Nothing}=nothing, limit::Union{Int,Nothing}=nothing,
    after::Union{String,Nothing}=nothing, service::ServiceEndpointSpec=OPENAIServiceEndpoint,
    config::Union{Nothing,RequestConfig}=nothing)
    validate_capability(service, :files, "Files API")
    cfg = _resolve_config(config)
    t0 = time_ns()
    try
        url = _api_base_url(service) * FILES_PATH
        params = String[]
        !isnothing(purpose) && push!(params, "purpose=$(_uripart(purpose))")
        !isnothing(limit) && push!(params, "limit=$(_uripart(limit))")
        !isnothing(after) && push!(params, "after=$(_uripart(after))")
        !isempty(params) && (url *= "?" * join(params, "&"))
        resp = _http("GET", url, auth_header(service); cfg, remaining=_remaining_s(cfg, t0))
        if resp.status == 200
            data = JSON.parse(resp.body; dicttype=Dict{String,Any})
            files = FileObject[_parse_file_object(f) for f in get(data, "data", [])]
            return FileListSuccess(response=FileList(data=files, has_more=get(data, "has_more", false), raw=data))
        else
            return _failure(FileFailure, resp)
        end
    catch e
        e isa InterruptException && rethrow()
        return _callerr(FileCallError, e)
    end
end

"""
    retrieve_file(file_id; service=OPENAIServiceEndpoint)

Retrieve a file's metadata. Returns `FileSuccess`, `FileFailure`, or `FileCallError`.

Pass `config::Union{Nothing,RequestConfig}` to override the timeout budget for this call (a single bounded attempt; `max_attempts` does not apply).
"""
function retrieve_file(file_id::String; service::ServiceEndpointSpec=OPENAIServiceEndpoint,
    config::Union{Nothing,RequestConfig}=nothing)
    validate_capability(service, :files, "Files API")
    cfg = _resolve_config(config)
    t0 = time_ns()
    try
        url = _api_base_url(service) * FILES_PATH * "/" * _uripart(file_id)
        resp = _http("GET", url, auth_header(service); cfg, remaining=_remaining_s(cfg, t0))
        resp.status == 200 ?
            FileSuccess(response=_parse_file_object(JSON.parse(resp.body; dicttype=Dict{String,Any}))) :
            _failure(FileFailure, resp)
    catch e
        e isa InterruptException && rethrow()
        _callerr(FileCallError, e)
    end
end

"""
    delete_file(file_id; service=OPENAIServiceEndpoint)

Delete a file. Returns `FileDeleteSuccess`, `FileFailure`, or `FileCallError`.

Pass `config::Union{Nothing,RequestConfig}` to override the timeout budget for this call (a single bounded attempt; `max_attempts` does not apply).
"""
function delete_file(file_id::String; service::ServiceEndpointSpec=OPENAIServiceEndpoint,
    config::Union{Nothing,RequestConfig}=nothing)
    validate_capability(service, :files, "Files API")
    cfg = _resolve_config(config)
    t0 = time_ns()
    try
        url = _api_base_url(service) * FILES_PATH * "/" * _uripart(file_id)
        resp = _http("DELETE", url, auth_header(service); cfg, remaining=_remaining_s(cfg, t0))
        if resp.status == 200
            d = JSON.parse(resp.body; dicttype=Dict{String,Any})
            return FileDeleteSuccess(id=get(d, "id", file_id), deleted=_confirm_deleted(d))
        else
            return _failure(FileFailure, resp)
        end
    catch e
        e isa InterruptException && rethrow()
        _callerr(FileCallError, e)
    end
end

"""
    file_content(file_id; service=OPENAIServiceEndpoint)

Download a file's raw bytes. Returns `FileContentSuccess` (`.content::Vector{UInt8}`),
`FileFailure`, or `FileCallError`. Use [`save_file_content`](@ref) to write to disk.

Pass `config::Union{Nothing,RequestConfig}` to override the timeout budget for this call (a single bounded attempt; `max_attempts` does not apply).
"""
function file_content(file_id::String; service::ServiceEndpointSpec=OPENAIServiceEndpoint,
    config::Union{Nothing,RequestConfig}=nothing)
    validate_capability(service, :files, "Files API")
    cfg = _resolve_config(config)
    t0 = time_ns()
    try
        url = _api_base_url(service) * FILES_PATH * "/" * _uripart(file_id) * "/content"
        resp = _http("GET", url, auth_header(service); cfg, remaining=_remaining_s(cfg, t0))
        resp.status == 200 ?
            FileContentSuccess(content=Vector{UInt8}(resp.body)) :
            _failure(FileFailure, resp)
    catch e
        e isa InterruptException && rethrow()
        _callerr(FileCallError, e)
    end
end

"""
    save_file_content(r::FileContentSuccess, path) -> path

Write downloaded file bytes to `path`, atomically: the bytes go to a temporary file in
the same directory, which is then renamed over `path`, so a failed write leaves any
existing file intact. An existing file keeps its permission bits. A symlink is written
through — a dangling one creates its target; a directory throws `ArgumentError`.
"""
save_file_content(r::FileContentSuccess, path::String) = _atomic_write(path, r.content)
