# ============================================================================
# Files API — upload, list, retrieve and delete files, and download their content, on
# each provider's own Files API:
#   OpenAI     /v1/files  https://developers.openai.com/api/reference/resources/files
#   Anthropic  /v1/files  https://platform.claude.com/docs/en/api/files/upload (list, retrieve, download, delete)
# OpenAI's files feed the Responses file_search & code_interpreter tools, the Batch API,
# and fine-tuning.
# ============================================================================

# The purposes an upload takes
# (https://developers.openai.com/api/reference/resources/files/methods/create).
const _FILE_PURPOSES = ("assistants", "batch", "fine-tune", "vision", "user_data")

# An `ArgumentError` naming the purposes the Files API takes, unless it takes `purpose`.
# "evals" fed OpenAI's Evals API, read-only from 2026-10-31 and shut down on 2026-11-30
# (https://developers.openai.com/api/docs/deprecations).
function _check_file_purpose(purpose)::Nothing
    accepted = join(repr.(_FILE_PURPOSES), ", ", " or ")
    purpose == "evals" && throw(ArgumentError(
        "purpose \"evals\" is for OpenAI's Evals API, which is read-only from 2026-10-31 and shuts down on " *
        "2026-11-30 (https://developers.openai.com/api/docs/deprecations); the Files API takes $accepted"))
    purpose in _FILE_PURPOSES || throw(ArgumentError("the Files API takes $accepted (got $(repr(purpose)))"))
    nothing
end

# ─── The provider wires ──────────────────────────────────────────────────────
# Where a provider keeps its files, how its replies read, and which options it takes. An
# endpoint not routed here speaks OpenAI's wire.
abstract type _FilesWire end
struct _OpenAIFiles <: _FilesWire end
struct _AnthropicFiles <: _FilesWire end

_files_wire(service) = _OpenAIFiles()
_files_wire(::_AnthropicEndpoint) = _AnthropicFiles()

_files_provider(::_OpenAIFiles) = "OpenAI"
_files_provider(::_AnthropicFiles) = "Anthropic"

# The Files API's address; a function, so tests can route an endpoint to a local server.
_files_url(service)::String = _api_base_url(service) * FILES_PATH
_files_url(::_AnthropicEndpoint)::String = ANTHROPIC_BASE_URL * FILES_PATH
_file_url(service, file_id::String)::String = _files_url(service) * "/" * _uripart(file_id)

# The headers of a Files request: the endpoint's own on OpenAI's wire. Anthropic's Files reference
# names `anthropic-version` and `anthropic-workspace-id`, and no `anthropic-beta` or user-profile
# header (https://platform.claude.com/docs/en/api/files/upload, Headers), and its SDK (v2.9.0) sends
# no beta for files: an Anthropic Files request sends the key, the API version and the endpoint's
# workspace, never its user profile or its betas.
_files_headers(service) = auth_header(service)
_files_headers(service::_AnthropicEndpoint)::Vector{Pair{String,String}} =
    _anthropic_base_headers(_anthropic_endpoint(service); profile=false)
# An upload's form sets its own Content-Type, with the boundary.
_files_upload_headers(service) = auth_header_multipart(service)
_files_upload_headers(service::_AnthropicEndpoint) = _files_headers(service)

# OpenAI's Files API requires a purpose; Anthropic's takes none.
function _check_upload_purpose(::_OpenAIFiles, purpose::Union{String,Nothing})::Nothing
    isnothing(purpose) && throw(ArgumentError(
        "upload_file: OpenAI's Files API requires a purpose: $(join(repr.(_FILE_PURPOSES), ", ", " or "))"))
    _check_file_purpose(purpose)
end
_check_upload_purpose(wire::_FilesWire, purpose::Union{String,Nothing})::Nothing =
    isnothing(purpose) ? nothing : throw(ArgumentError("upload_file: $(_files_provider(wire))'s Files API takes " *
        "no purpose (got $(repr(purpose))); call upload_file(path; service)"))

# Seconds from the upload until a file expires: OpenAI's `expires_after.seconds`, "between 3600 (1 hour)
# and 2592000 (30 days)" (FileExpirationAfter, on the Files and Uploads APIs alike); Anthropic's
# `expires_in_seconds`, "between 3600 (one hour) and 7776000 (ninety days)". Each API answered HTTP 400
# one second outside its range on 2026-11-16.
_file_expiry(::_OpenAIFiles) = 3600:2592000
_file_expiry(::_AnthropicFiles) = 3600:7776000

function _check_file_expiry(verb::String, wire::_FilesWire, expires_after::Union{Int,Nothing})::Nothing
    isnothing(expires_after) && return nothing
    r = _file_expiry(wire)
    expires_after in r || throw(ArgumentError("$verb: $(_files_provider(wire))'s Files API takes expires_after " *
        "from $(first(r)) to $(last(r)) seconds (got $expires_after)"))
    nothing
end

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
        r = _with_resolved_cancel(() -> fetch(c), cancel)
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
    FileUpload(; file, purpose=nothing, expires_after=nothing, service=OPENAIServiceEndpoint)

A file-upload request; `file` is a path on disk. On OpenAI `purpose` is required, one of
`"assistants"`, `"batch"`, `"fine-tune"`, `"vision"`, `"user_data"`; any other purpose throws
`ArgumentError`, `"evals"` naming the Evals API's shutdown on 2026-11-30. Anthropic's Files API
takes no purpose, and refuses one with `ArgumentError`.

`expires_after` is the number of seconds from the upload until the file expires: 3600 (one
hour) to 2592000 (30 days) on OpenAI, to 7776000 (90 days) on Anthropic; a value outside the
range throws `ArgumentError`. `nothing` keeps the provider's default: on OpenAI "files with
`purpose=batch` expire after 30 days and all other files are persisted until they are manually
deleted"; on Anthropic a file does not expire.
"""
@kwdef struct FileUpload
    service::ServiceEndpointSpec = OPENAIServiceEndpoint
    file::String
    purpose::Union{String,Nothing} = nothing
    expires_after::Union{Int,Nothing} = nothing
    function FileUpload(service, file, purpose, expires_after)
        isfile(file) || throw(ArgumentError("file not found: $file"))
        wire = _files_wire(service)
        _check_upload_purpose(wire, purpose)
        _check_file_expiry("upload_file", wire, expires_after)
        new(service, file, purpose, expires_after)
    end
end
FileUpload(service, file, purpose) = FileUpload(service, file, purpose, nothing)

# ─── Parsed objects ───────────────────────────────────────────────────────────

"""
    FileObject

A file stored by a provider's Files API; `raw` holds the object as the provider sent it.

- `id::String`: the file's id, which the other Files verbs take.
- `bytes::Union{Int,Nothing}`: the size — OpenAI's `bytes`, Anthropic's `size_bytes`. `nothing`
  when OpenAI gives the size as null, which its Files reference allows "In a completed file
  upload response, … when the file size is not yet available".
- `created_at::Int`: the creation time in Unix seconds — OpenAI's `created_at`; Anthropic's RFC 3339
  `created_at`, to the second.
- `filename::String`: the file's name.
- `purpose::Union{String,Nothing}`: OpenAI's purpose; `nothing` on Anthropic, which has none.
- `status::Union{String,Nothing}`: OpenAI's (deprecated) status; `nothing` on Anthropic.

Anthropic's `mime_type`, `downloadable` and `expires_at`, and OpenAI's `expires_at`, are in `raw`.
A reply without a field the provider's reference requires — OpenAI's `id`, `bytes`, `created_at`,
`filename` or `purpose`; Anthropic's `id`, `size_bytes`, `created_at` or `filename` — is no file:
the call returns its call error, a [`FileCallError`](@ref) (from `complete_upload`, an
`UploadCallError`).
"""
@kwdef struct FileObject
    id::String
    bytes::Union{Int,Nothing}
    created_at::Int
    filename::String
    purpose::Union{String,Nothing}
    status::Union{String,Nothing} = nothing
    raw::Dict{String,Any} = Dict{String,Any}()
end

"""
    FileList

A page of [`FileObject`](@ref)s from [`list_files`](@ref); `has_more` signals that further pages
are available, and `raw` holds the page as the provider sent it, with the cursor of the next page:
OpenAI's `last_id`, Anthropic's `next_page`.
"""
@kwdef struct FileList
    data::Vector{FileObject}
    has_more::Bool = false
    raw::Dict{String,Any} = Dict{String,Any}()
end

# A field of a Files reply: absent or null where the provider's reference allows it; any other JSON type
# than the documented one is a malformed reply, and the throw becomes the verb's call error.
function _file_field(d::AbstractDict, key::String, ::Type{T})::Union{T,Nothing} where {T}
    v = get(d, key, nothing)
    isnothing(v) || v isa T || throw(ArgumentError("Files reply field $(repr(key)) holds a $(typeof(v)), not the documented $T"))
    v
end

# A field the provider's reference requires: a 200 without it is no reply of that kind (OpenAI's list
# requires `data` and `has_more`), never an empty page or a value made up for it.
_file_required(d::AbstractDict, key::String, ::Type{T}) where {T} =
    @something(_file_field(d, key, T), throw(ArgumentError("the Files reply has no $(repr(key))")))

# An RFC 3339 time the reference requires, in Unix seconds.
function _file_time(d::AbstractDict, key::String)::Int
    s = _file_required(d, key, String)
    t = @something(_rfc3339_utc(s), throw(ArgumentError("Files reply field $(repr(key)) is not an RFC 3339 date-time: $(repr(s))")))
    floor(Int, Dates.datetime2unix(t))
end

_json_object(body)::Dict{String,Any} = JSON.parse(body; dicttype=Dict{String,Any})

function _parse_file_object(d::AbstractDict)
    FileObject(
        id=d["id"], bytes=d["bytes"], created_at=d["created_at"], filename=d["filename"], purpose=d["purpose"],
        status=get(d, "status", nothing), raw=Dict{String,Any}(d))
end

_parse_file(::_OpenAIFiles, d::AbstractDict)::FileObject = _parse_file_object(d)
# Anthropic's FileMetadata (https://platform.claude.com/docs/en/api/files/upload, Returns).
_parse_file(::_AnthropicFiles, d::AbstractDict)::FileObject = FileObject(
    id=_file_required(d, "id", String), bytes=_file_required(d, "size_bytes", Int),
    created_at=_file_time(d, "created_at"), filename=_file_required(d, "filename", String),
    purpose=nothing, status=nothing, raw=Dict{String,Any}(d))

# A page: OpenAI's `data` and `has_more`, both required. Anthropic's `data`, and `next_page`, its
# opaque cursor of the next page, null on the last.
_file_page(wire::_OpenAIFiles, d::Dict{String,Any})::FileList = FileList(
    data=FileObject[_parse_file(wire, f) for f in _file_required(d, "data", AbstractVector)],
    has_more=_file_required(d, "has_more", Bool), raw=d)
_file_page(wire::_AnthropicFiles, d::Dict{String,Any})::FileList = FileList(
    data=FileObject[_parse_file(wire, f) for f in _file_required(d, "data", AbstractVector)],
    has_more=!isnothing(_file_field(d, "next_page", String)), raw=d)

# A page's query. OpenAI's listFiles takes `purpose`, `limit`, `after` (a file id) and `order`;
# Anthropic's takes `limit` and `page`, the cursor `after` carries. An option the provider does not
# take is refused by name.
_query(kv::Pair...)::Vector{String} = String[k * "=" * _uripart(v) for (k, v) in kv if !isnothing(v)]

function _list_query(::_OpenAIFiles, purpose, limit, after, order)::Vector{String}
    # OpenAI's listFiles `order` is an enum; the API answered "'sideways' is not one of ['asc', 'desc']"
    # (HTTP 400) on 2026-11-16.
    isnothing(order) || order in ("asc", "desc") ||
        throw(ArgumentError("list_files: order is \"asc\" or \"desc\" (got $(repr(order)))"))
    _query("purpose" => purpose, "limit" => limit, "after" => after, "order" => order)
end
function _list_query(wire::_AnthropicFiles, purpose, limit, after, order)::Vector{String}
    _refuse_list_options(wire, purpose, order)
    _query("limit" => limit, "page" => after)
end

function _refuse_list_options(wire::_FilesWire, purpose, order)::Nothing
    set = [k for (k, v) in ("purpose" => purpose, "order" => order) if !isnothing(v)]
    isempty(set) || throw(ArgumentError("list_files: $(_files_provider(wire))'s Files API takes no " *
        "$(join(set, " or ")), which only OpenAI's takes; unset $(join(set, " and "))"))
    nothing
end

# A delete's reply. OpenAI's DeleteFileResponse requires `id`, `object` ("file") and `deleted`.
# Anthropic's DeletedFile has the `id`, and a `type` that is "file_deleted" when present: its 200 is
# the confirmation.
_file_deleted(::_OpenAIFiles, d::Dict{String,Any}, ::String)::FileDeleteSuccess = FileDeleteSuccess(
    id=_file_required(d, "id", String), deleted=_confirm_deleted(d), object=_file_required(d, "object", String))
function _file_deleted(::_AnthropicFiles, d::Dict{String,Any}, ::String)::FileDeleteSuccess
    type = _file_field(d, "type", String)
    isnothing(type) || type == "file_deleted" ||
        throw(ArgumentError("the delete reply's type is $(repr(type)), not \"file_deleted\""))
    FileDeleteSuccess(id=_file_required(d, "id", String), deleted=true, object=type)
end

# ─── Result types ─────────────────────────────────────────────────────────────

"Successful upload/retrieve result wrapping a [`FileObject`](@ref)."
@kwdef struct FileSuccess <: LLMRequestResponse; response::FileObject; end
"Successful [`list_files`](@ref) result wrapping a [`FileList`](@ref)."
@kwdef struct FileListSuccess <: LLMRequestResponse; response::FileList; end
"Successful [`file_content`](@ref) result; `content` holds the raw file bytes."
@kwdef struct FileContentSuccess <: LLMRequestResponse; content::Vector{UInt8}; end
"""
    FileDeleteSuccess

Successful [`delete_file`](@ref) result: the service confirmed (`deleted` is always `true`) the
removal of `id`, the id its reply names. `object` is the reply's type tag: OpenAI's `object`,
`"file"`; Anthropic's `type`, `"file_deleted"`, or `nothing` when its reply has none.
"""
@kwdef struct FileDeleteSuccess <: LLMRequestResponse; id::String; deleted::Bool; object::Union{String,Nothing} = nothing; end
FileDeleteSuccess(id, deleted) = FileDeleteSuccess(id, deleted, nothing)
"Files API error result: HTTP `status`, the raw `response` body, and the `request_id` the service sent (`x-request-id`/`request-id` header), if any."
@kwdef struct FileFailure <: LLMRequestResponse; response::String; status::Int; request_id::Union{String,Nothing} = nothing; end
"Files API call that produced no usable reply (transport failure, timeout, or a 200 that could not be decoded); `cause` is the underlying exception — a [`UniLMTimeout`](@ref) for a timeout."
@kwdef struct FileCallError <: LLMRequestResponse; error::String; status::Union{Int,Nothing} = nothing; cause::Union{Nothing,Exception} = nothing; end

_file_resp(wire::_FilesWire, resp::HTTP.Response) = resp.status == 200 ?
    FileSuccess(response=_parse_file(wire, _json_object(resp.body))) : _failure(FileFailure, resp)

# ─── Requests ─────────────────────────────────────────────────────────────────

# The form fields before the file: OpenAI's purpose and its `expires_after` object, which a form sends
# as bracketed fields (OpenAI's Files reference: `-F expires_after[anchor]="created_at"`); Anthropic's
# `expires_in_seconds`.
function _upload_fields(::_OpenAIFiles, u::FileUpload)::Vector{Pair{String,Any}}
    fields = Pair{String,Any}["purpose" => something(u.purpose)]
    isnothing(u.expires_after) ||
        push!(fields, "expires_after[anchor]" => "created_at", "expires_after[seconds]" => string(u.expires_after))
    fields
end
_upload_fields(::_AnthropicFiles, u::FileUpload)::Vector{Pair{String,Any}} =
    isnothing(u.expires_after) ? Pair{String,Any}[] : Pair{String,Any}["expires_in_seconds" => string(u.expires_after)]

function _upload(wire::_FilesWire, u::FileUpload, cfg::RequestConfig, t0::UInt64, tok)
    form = HTTP.Form(push!(_upload_fields(wire, u),
        "file" => HTTP.Multipart(basename(u.file), IOBuffer(read(u.file)), _mime_for(u.file))))
    _file_resp(wire, _http("POST", _files_url(u.service), _files_upload_headers(u.service), form;
                           cfg, remaining=_remaining_s(cfg, t0), cancel=tok))
end

"""
    upload_file(path, purpose; service=OPENAIServiceEndpoint, expires_after=nothing) -> LLMRequestResponse
    upload_file(path; service, expires_after=nothing) -> LLMRequestResponse
    upload_file(u::FileUpload) -> LLMRequestResponse

Upload a file. Returns `FileSuccess`, `FileFailure`, or `FileCallError`. OpenAI's Files API
requires a `purpose`; Anthropic's takes none, so `upload_file(path; service=ANTHROPICServiceEndpoint)`.
`expires_after` is the number of seconds until the file expires (see [`FileUpload`](@ref)).
The file goes as multipart/form-data, its MIME type read from its extension.

Pass `config::Union{Nothing,RequestConfig}` to override the timeout budget for this call
(a single bounded attempt; `max_attempts` does not apply). Like every other create, an
upload is never retried: a POST that timed out, or drew a gateway 5xx after the backend
stored the file, may still have created it, and a second attempt would store it twice.
A `FileFailure`/`FileCallError` therefore does not prove that no file was created.

Pass `cancel::Union{Nothing,CancelToken}` (default: the ambient [`with_cancel`](@ref) token) to make the call cancellable: once the token is cancelled it returns a `FileCallError` whose `cause` is a [`UniLMCancelled`](@ref) — without sending anything when the token was already cancelled, at once when mid-request (never retried).
"""
function upload_file(u::FileUpload; config::Union{Nothing,RequestConfig}=nothing,
                     cancel::Union{Nothing,CancelToken}=nothing)
    validate_capability(u.service, :files, "Files API")
    cfg = _resolve_config(config)
    t0 = time_ns()
    tok = _resolve_cancel(cancel)
    try
        # A cancelled token reads no file: the upload holds the whole file, read before `_http` checks the
        # token (and outside the request's time bound).
        iscancelled(tok) && throw(UniLMCancelled(:token, _elapsed_s(t0)))
        return _upload(_files_wire(u.service), u, cfg, t0, tok)
    catch e
        e isa InterruptException && rethrow()
        return _callerr(FileCallError, e)
    end
end
upload_file(path::String, purpose::String; service::ServiceEndpointSpec=OPENAIServiceEndpoint,
    expires_after::Union{Int,Nothing}=nothing, config::Union{Nothing,RequestConfig}=nothing,
    cancel::Union{Nothing,CancelToken}=nothing) =
    upload_file(FileUpload(; service, file=path, purpose, expires_after); config, cancel)
upload_file(path::String; service::ServiceEndpointSpec=OPENAIServiceEndpoint,
    expires_after::Union{Int,Nothing}=nothing, config::Union{Nothing,RequestConfig}=nothing,
    cancel::Union{Nothing,CancelToken}=nothing) =
    upload_file(FileUpload(; service, file=path, expires_after); config, cancel)

"""
    list_files(; purpose=nothing, limit=nothing, after=nothing, order=nothing, service=OPENAIServiceEndpoint)

List uploaded files, a page at a time. Returns `FileListSuccess`, `FileFailure`, or `FileCallError`.

- `limit`: the page's size.
- `after`: where the page starts — on OpenAI, the id of the last file of the page before; on
  Anthropic, the page before's `next_page` (in its `raw`).
- `purpose` (OpenAI only): only the files of that purpose.
- `order` (OpenAI only): by creation time, `"asc"` or `"desc"` (any other value throws
  `ArgumentError` before any request); `nothing` keeps the API's order.

`purpose` or `order` on Anthropic throws `ArgumentError` before any request.

Pass `config::Union{Nothing,RequestConfig}` to override the timeout budget for this call (a single bounded attempt; `max_attempts` does not apply).

Pass `cancel::Union{Nothing,CancelToken}` (default: the ambient [`with_cancel`](@ref) token) to make the call cancellable: once the token is cancelled it returns a `FileCallError` whose `cause` is a [`UniLMCancelled`](@ref) — without sending anything when the token was already cancelled, at once when mid-request (never retried).
"""
function list_files(; purpose::Union{String,Nothing}=nothing, limit::Union{Int,Nothing}=nothing,
    after::Union{String,Nothing}=nothing, order::Union{String,Nothing}=nothing,
    service::ServiceEndpointSpec=OPENAIServiceEndpoint,
    config::Union{Nothing,RequestConfig}=nothing, cancel::Union{Nothing,CancelToken}=nothing)
    validate_capability(service, :files, "Files API")
    wire = _files_wire(service)
    params = _list_query(wire, purpose, limit, after, order)
    cfg = _resolve_config(config)
    t0 = time_ns()
    try
        url = _files_url(service) * (isempty(params) ? "" : "?" * join(params, "&"))
        resp = _http("GET", url, _files_headers(service); cfg, remaining=_remaining_s(cfg, t0), cancel=_resolve_cancel(cancel))
        resp.status == 200 ? FileListSuccess(response=_file_page(wire, _json_object(resp.body))) : _failure(FileFailure, resp)
    catch e
        e isa InterruptException && rethrow()
        _callerr(FileCallError, e)
    end
end

"""
    retrieve_file(file_id; service=OPENAIServiceEndpoint)

Retrieve a file's metadata. Returns `FileSuccess`, `FileFailure`, or `FileCallError`.

Pass `config::Union{Nothing,RequestConfig}` to override the timeout budget for this call (a single bounded attempt; `max_attempts` does not apply).

Pass `cancel::Union{Nothing,CancelToken}` (default: the ambient [`with_cancel`](@ref) token) to make the call cancellable: once the token is cancelled it returns a `FileCallError` whose `cause` is a [`UniLMCancelled`](@ref) — without sending anything when the token was already cancelled, at once when mid-request (never retried).
"""
function retrieve_file(file_id::String; service::ServiceEndpointSpec=OPENAIServiceEndpoint,
    config::Union{Nothing,RequestConfig}=nothing, cancel::Union{Nothing,CancelToken}=nothing)
    validate_capability(service, :files, "Files API")
    cfg = _resolve_config(config)
    t0 = time_ns()
    try
        _file_resp(_files_wire(service), _http("GET", _file_url(service, file_id), _files_headers(service);
                                               cfg, remaining=_remaining_s(cfg, t0), cancel=_resolve_cancel(cancel)))
    catch e
        e isa InterruptException && rethrow()
        _callerr(FileCallError, e)
    end
end

"""
    delete_file(file_id; service=OPENAIServiceEndpoint)

Delete a file. Returns `FileDeleteSuccess`, `FileFailure`, or `FileCallError`.

Pass `config::Union{Nothing,RequestConfig}` to override the timeout budget for this call (a single bounded attempt; `max_attempts` does not apply).

Pass `cancel::Union{Nothing,CancelToken}` (default: the ambient [`with_cancel`](@ref) token) to make the call cancellable: once the token is cancelled it returns a `FileCallError` whose `cause` is a [`UniLMCancelled`](@ref) — without sending anything when the token was already cancelled, at once when mid-request (never retried).
"""
function delete_file(file_id::String; service::ServiceEndpointSpec=OPENAIServiceEndpoint,
    config::Union{Nothing,RequestConfig}=nothing, cancel::Union{Nothing,CancelToken}=nothing)
    validate_capability(service, :files, "Files API")
    cfg = _resolve_config(config)
    t0 = time_ns()
    try
        resp = _http("DELETE", _file_url(service, file_id), _files_headers(service);
                     cfg, remaining=_remaining_s(cfg, t0), cancel=_resolve_cancel(cancel))
        resp.status == 200 ? _file_deleted(_files_wire(service), _json_object(resp.body), file_id) :
                             _failure(FileFailure, resp)
    catch e
        e isa InterruptException && rethrow()
        _callerr(FileCallError, e)
    end
end

"""
    file_content(file_id; service=OPENAIServiceEndpoint)

Download a file's raw bytes. Returns `FileContentSuccess` (`.content::Vector{UInt8}`),
`FileFailure`, or `FileCallError`. Use [`save_file_content`](@ref) to write to disk.

Anthropic serves the bytes of a file a tool made (its metadata's `downloadable` is `true`); for
a file uploaded through its Files API the call returns Anthropic's HTTP 400 as a `FileFailure`.

Pass `config::Union{Nothing,RequestConfig}` to override the timeout budget for this call (a single bounded attempt; `max_attempts` does not apply).

Pass `cancel::Union{Nothing,CancelToken}` (default: the ambient [`with_cancel`](@ref) token) to make the call cancellable: once the token is cancelled it returns a `FileCallError` whose `cause` is a [`UniLMCancelled`](@ref) — without sending anything when the token was already cancelled, at once when mid-request (never retried).
"""
function file_content(file_id::String; service::ServiceEndpointSpec=OPENAIServiceEndpoint,
    config::Union{Nothing,RequestConfig}=nothing, cancel::Union{Nothing,CancelToken}=nothing)
    validate_capability(service, :files, "Files API")
    cfg = _resolve_config(config)
    t0 = time_ns()
    try
        resp = _http("GET", _file_url(service, file_id) * "/content", _files_headers(service);
                     cfg, remaining=_remaining_s(cfg, t0), cancel=_resolve_cancel(cancel))
        resp.status == 200 ? FileContentSuccess(content=Vector{UInt8}(resp.body)) : _failure(FileFailure, resp)
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
