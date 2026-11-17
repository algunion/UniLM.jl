# ============================================================================
# Gemini File Search — stores of indexed documents that the Interactions API's `file_search` tool
# (`gemini_file_search`, interactions.jl) searches (https://ai.google.dev/gemini-api/docs/file-search,
# https://ai.google.dev/api/file-search/file-search-stores). A document enters a store from a local file
# (`upload_to_file_search_store`) or from a Files API file (`import_to_file_search_store`) through a
# long-running operation (`poll_file_search_operation`). Each reply shape below is one the API returned on
# 2026-11-17; the reply helpers are gemini_caching.jl's.
# ============================================================================

const _STORE_SHAPE = ("fileSearchStores", nothing)
const _DOCUMENT_SHAPE = ("fileSearchStores", nothing, "documents", nothing)
# An import's operation, and an upload's.
const _OPERATION_SHAPES = (("fileSearchStores", nothing, "operations", nothing),
                           ("fileSearchStores", nothing, "upload", "operations", nothing))

# Where an upload sends its bytes (discovery's media.uploadToFileSearchStore); a function so tests can route an
# endpoint to a local server.
_gemini_upload_root(::Type{GEMINIServiceEndpoint})::String = "https://generativelanguage.googleapis.com/upload/v1beta"

# A document's or an operation's name, which the API gives whole: a call names it whole.
function _gemini_nested_name(name::AbstractString, shapes::Tuple, what::String)::String
    any(s -> _gemini_shaped(name, s), shapes) || throw(ArgumentError(
        "$what names $(join((_shape_text(s) for s in shapes), " or ")) (got $(repr(name)))"))
    String(name)
end
_store_name(store::AbstractString, verb::String)::String = _gemini_resource_name("fileSearchStores", store, verb)

# ─── Parsed objects ──────────────────────────────────────────────────────────

"""
    FileSearchStoreObject

A Gemini File Search store, as [`create_file_search_store`](@ref), [`retrieve_file_search_store`](@ref)
and [`list_file_search_stores`](@ref) return it; `raw` holds it as Gemini sent it.

- `name::String`: `"fileSearchStores/<id>"`, which [`gemini_file_search`](@ref) and the store verbs take
  (the verbs take the id alone too); Gemini derives the id from the display name.
- `display_name::Union{String,Nothing}`; `embedding_model::Union{String,Nothing}`: the model that embeds
  its documents, without its `models/` prefix (`"gemini-embedding-001"` unless the store was made with
  another).
- `active_documents`, `pending_documents`, `failed_documents` (`Int`): its documents ready for retrieval,
  being processed, and failed; `size_bytes::Int`: the raw bytes ingested. Each is `0` when Gemini leaves
  it out, as it leaves out a zero.
- `create_time`, `update_time` (`DateTime`, UTC).

A reply without a `name`, `createTime` or `updateTime`, or with a field of another JSON type, is no store:
the call returns its [`FileSearchCallError`](@ref).
"""
@kwdef struct FileSearchStoreObject
    name::String
    display_name::Union{String,Nothing} = nothing
    embedding_model::Union{String,Nothing} = nothing
    active_documents::Int = 0
    pending_documents::Int = 0
    failed_documents::Int = 0
    size_bytes::Int = 0
    create_time::DateTime
    update_time::DateTime
    raw::Dict{String,Any} = Dict{String,Any}()
end

"""
    FileSearchDocumentObject

A document in a File Search store, as [`retrieve_file_search_document`](@ref) and
[`list_file_search_documents`](@ref) return it; `raw` holds it as Gemini sent it.

- `name::String`: `"fileSearchStores/<store>/documents/<id>"`, which the document verbs take.
- `display_name::Union{String,Nothing}`; `mime_type::Union{String,Nothing}`; `size_bytes::Int`.
- `state::String`: `"STATE_PENDING"` while its chunks are processed, `"STATE_ACTIVE"` when they are
  ready, `"STATE_FAILED"`; `"STATE_UNSPECIFIED"` when Gemini leaves it out.
- `custom_metadata::Dict{String,Any}`: its custom metadata, each key's string, number or `Vector{String}`
  (Gemini's `stringValue`, `numericValue`, `stringListValue`).
- `create_time`, `update_time` (`DateTime`, UTC).
"""
@kwdef struct FileSearchDocumentObject
    name::String
    display_name::Union{String,Nothing} = nothing
    state::String = "STATE_UNSPECIFIED"
    mime_type::Union{String,Nothing} = nothing
    size_bytes::Int = 0
    custom_metadata::Dict{String,Any} = Dict{String,Any}()
    create_time::DateTime
    update_time::DateTime
    raw::Dict{String,Any} = Dict{String,Any}()
end

"""
    FileSearchOperation

The long-running operation that puts a document into a store, as [`upload_to_file_search_store`](@ref),
[`import_to_file_search_store`](@ref), [`retrieve_file_search_operation`](@ref) and
[`poll_file_search_operation`](@ref) return it; `raw` holds it as Gemini sent it.

- `name::String`: `"fileSearchStores/<store>/operations/<id>"` for an import,
  `"fileSearchStores/<store>/upload/operations/<id>"` for an upload.
- `done::Bool`: `false` while the document is being processed (Gemini leaves `done` out until then).
- `document::Union{String,Nothing}`: the name of the document it makes, once its reply names it — an
  upload's at once, an import's when done (an import's done reply gives the document's id alone, which is
  named in the operation's store).
- `error::Union{Dict{String,Any},Nothing}`: the `Status` of an operation that failed (`code`, `message`).
"""
@kwdef struct FileSearchOperation
    name::String
    done::Bool = false
    document::Union{String,Nothing} = nothing
    error::Union{Dict{String,Any},Nothing} = nothing
    raw::Dict{String,Any} = Dict{String,Any}()
end

"""
    FileSearchStoreList

A page of [`FileSearchStoreObject`](@ref)s from [`list_file_search_stores`](@ref); `has_more` signals a
further page, which starts at `raw["nextPageToken"]`.
"""
@kwdef struct FileSearchStoreList
    data::Vector{FileSearchStoreObject}
    has_more::Bool = false
    raw::Dict{String,Any} = Dict{String,Any}()
end

"""
    FileSearchDocumentList

A page of [`FileSearchDocumentObject`](@ref)s from [`list_file_search_documents`](@ref), oldest first;
`has_more` signals a further page, which starts at `raw["nextPageToken"]`.
"""
@kwdef struct FileSearchDocumentList
    data::Vector{FileSearchDocumentObject}
    has_more::Bool = false
    raw::Dict{String,Any} = Dict{String,Any}()
end

# ─── Result types ────────────────────────────────────────────────────────────

"Successful [`create_file_search_store`](@ref) or [`retrieve_file_search_store`](@ref) result wrapping a [`FileSearchStoreObject`](@ref)."
@kwdef struct FileSearchStoreSuccess <: LLMRequestResponse; response::FileSearchStoreObject; end
"Successful [`list_file_search_stores`](@ref) result wrapping a [`FileSearchStoreList`](@ref)."
@kwdef struct FileSearchStoreListSuccess <: LLMRequestResponse; response::FileSearchStoreList; end
"Successful [`retrieve_file_search_document`](@ref) result wrapping a [`FileSearchDocumentObject`](@ref)."
@kwdef struct FileSearchDocumentSuccess <: LLMRequestResponse; response::FileSearchDocumentObject; end
"Successful [`list_file_search_documents`](@ref) result wrapping a [`FileSearchDocumentList`](@ref)."
@kwdef struct FileSearchDocumentListSuccess <: LLMRequestResponse; response::FileSearchDocumentList; end
"Successful [`upload_to_file_search_store`](@ref), [`import_to_file_search_store`](@ref), [`retrieve_file_search_operation`](@ref) or [`poll_file_search_operation`](@ref) result wrapping a [`FileSearchOperation`](@ref)."
@kwdef struct FileSearchOperationSuccess <: LLMRequestResponse; response::FileSearchOperation; end
"Successful [`delete_file_search_store`](@ref) or [`delete_file_search_document`](@ref) result: Gemini deleted `name` (`deleted` is always `true`); its reply is empty, so `name` is the one the call named."
@kwdef struct FileSearchDeleteSuccess <: LLMRequestResponse; name::String; deleted::Bool; end
"File Search error result: HTTP `status`, the raw `response` body, and the `request_id` the service sent, if any."
@kwdef struct FileSearchFailure <: LLMRequestResponse; response::String; status::Int; request_id::Union{String,Nothing} = nothing; end
"File Search call that produced no usable reply (a missing key, transport failure, timeout, cancel, or a 200 that is not the documented object); `cause` is the underlying exception. `last_observed` is set only when [`poll_file_search_operation`](@ref) runs out of time or is cancelled: the last [`FileSearchOperation`](@ref) it saw, if any."
@kwdef struct FileSearchCallError <: LLMRequestResponse; error::String; status::Union{Int,Nothing} = nothing; cause::Union{Nothing,Exception} = nothing; last_observed::Union{Nothing,FileSearchOperation} = nothing; end

_transient(r::FileSearchFailure) = _is_retryable(r.status, r.response)
_transient(r::FileSearchCallError) = _per_attempt_timeout(r.cause)

# ─── Replies ─────────────────────────────────────────────────────────────────

function _parse_store(d::Dict{String,Any})::FileSearchStoreObject
    w = "the File Search store"
    model = _gemini_field(d, "embeddingModel", String, w)
    FileSearchStoreObject(name=_gemini_reply_name(d, _STORE_SHAPE, w), display_name=_gemini_field(d, "displayName", String, w),
        embedding_model=isnothing(model) ? nothing : String(chopprefix(model, "models/")),
        active_documents=_gemini_int64(d, "activeDocumentsCount", w), pending_documents=_gemini_int64(d, "pendingDocumentsCount", w),
        failed_documents=_gemini_int64(d, "failedDocumentsCount", w), size_bytes=_gemini_int64(d, "sizeBytes", w),
        create_time=_gemini_time(d, "createTime", w), update_time=_gemini_time(d, "updateTime", w), raw=d)
end

# A custom metadata entry: its key, and one of its values.
function _metadata_entry(x)::Pair{String,Any}
    w = "a custom metadata entry"
    m = _gemini_object(x, w)
    key = _gemini_required(m, "key", String, w)
    haskey(m, "stringValue") && return key => _gemini_required(m, "stringValue", String, w)
    if haskey(m, "numericValue")
        v = m["numericValue"]
        v isa Real && !(v isa Bool) || throw(ArgumentError("custom metadata $(repr(key)) has a numericValue that is no number"))
        return key => v
    end
    haskey(m, "stringListValue") || throw(ArgumentError("custom metadata $(repr(key)) has no value"))
    list = something(_gemini_field(_gemini_required(m, "stringListValue", AbstractDict, w), "values", AbstractVector, w), [])
    all(v -> v isa String, list) || throw(ArgumentError("custom metadata $(repr(key)) lists a value that is no string"))
    key => String[v for v in list]
end

function _parse_document(d::Dict{String,Any})::FileSearchDocumentObject
    w = "the File Search document"
    FileSearchDocumentObject(name=_gemini_reply_name(d, _DOCUMENT_SHAPE, w), display_name=_gemini_field(d, "displayName", String, w),
        state=something(_gemini_field(d, "state", String, w), "STATE_UNSPECIFIED"), mime_type=_gemini_field(d, "mimeType", String, w),
        size_bytes=_gemini_int64(d, "sizeBytes", w),
        custom_metadata=Dict{String,Any}(_metadata_entry(m) for m in something(_gemini_field(d, "customMetadata", AbstractVector, w), [])),
        create_time=_gemini_time(d, "createTime", w), update_time=_gemini_time(d, "updateTime", w), raw=d)
end

# The document an operation names: whole in an upload's reply, by its id alone in an import's done reply
# (observed 2026-11-17), which is named in the store the operation belongs to.
function _operation_document(op::String, doc::String)::String
    _gemini_shaped(doc, _DOCUMENT_SHAPE) && return doc
    _gemini_segment(doc) || throw(ArgumentError("the File Search operation names its document $(repr(doc)), no document name or id"))
    join(split(op, '/')[1:2], '/') * "/documents/" * doc
end

function _parse_operation(d::Dict{String,Any})::FileSearchOperation
    w = "the File Search operation"
    name = _gemini_required(d, "name", String, w)
    _gemini_nested_name(name, _OPERATION_SHAPES, "$w's name")
    response = _gemini_field(d, "response", AbstractDict, w)
    doc = isnothing(response) ? nothing : _gemini_field(response, "documentName", String, "$w's response")
    err = _gemini_field(d, "error", AbstractDict, w)
    FileSearchOperation(name=name, done=something(_gemini_field(d, "done", Bool, w), false),
                        document=isnothing(doc) ? nothing : _operation_document(name, doc),
                        error=isnothing(err) ? nothing : Dict{String,Any}(err), raw=d)
end

_store_success(x) = FileSearchStoreSuccess(response=_parse_store(_gemini_object(x, "the File Search store reply")))
_document_success(x) = FileSearchDocumentSuccess(response=_parse_document(_gemini_object(x, "the File Search document reply")))
_operation_success(x) = FileSearchOperationSuccess(response=_parse_operation(_gemini_object(x, "the File Search operation reply")))

# A page: its entries under `key` and `nextPageToken`, each left out when empty, so `{}` is an empty last page.
function _file_search_page(x, key::String)
    w = "the File Search listing"
    d = _gemini_object(x, w)
    token = _gemini_field(d, "nextPageToken", String, w)
    (entries=Dict{String,Any}[_gemini_object(e, "a listed entry") for e in something(_gemini_field(d, key, AbstractVector, w), [])],
     has_more=!(isnothing(token) || isempty(token)), raw=d)
end
function _store_page(x)::FileSearchStoreListSuccess
    p = _file_search_page(x, "fileSearchStores")
    FileSearchStoreListSuccess(response=FileSearchStoreList(data=FileSearchStoreObject[_parse_store(e) for e in p.entries],
                                                            has_more=p.has_more, raw=p.raw))
end
function _document_page(x)::FileSearchDocumentListSuccess
    p = _file_search_page(x, "documents")
    FileSearchDocumentListSuccess(response=FileSearchDocumentList(data=FileSearchDocumentObject[_parse_document(e) for e in p.entries],
                                                                  has_more=p.has_more, raw=p.raw))
end

_deleted(name::String) = x -> (_gemini_object(x, "the delete reply"); FileSearchDeleteSuccess(name=name, deleted=true))

_file_search_call(decode::Function, method::String, url::String, service, config, cancel; body=nothing) =
    _vs_json_call(decode, FileSearchFailure, FileSearchCallError, method, url, service, config, cancel; body)

# ─── Request settings ────────────────────────────────────────────────────────

# The listings take page sizes from 1 to 20: "page_size must be between 1 and 20" (HTTP 400 for 21, observed
# 2026-11-17; the references: "The maximum size limit is 20").
function _file_search_page_query(verb::String, limit, after)::String
    isnothing(limit) || 1 <= limit <= 20 || throw(ArgumentError(
        "$verb: limit is 1 to 20, the page sizes Gemini takes (got $limit)"))
    _gemini_query("pageSize" => limit, "pageToken" => after)
end

# Custom metadata: "A `Document` can have a maximum of 20 `CustomMetadata`", each key's value a string, a number
# or a list of strings (stringValue, numericValue, stringListValue).
function _custom_metadata(verb::String, metadata::AbstractDict)::Vector{Dict{String,Any}}
    length(metadata) <= 20 || throw(ArgumentError(
        "$verb: a document takes at most 20 custom_metadata entries (got $(length(metadata)))"))
    map(collect(metadata)) do (k, v)
        key = string(k)
        v isa AbstractString ? Dict{String,Any}("key" => key, "stringValue" => String(v)) :
        v isa Real && !(v isa Bool) ? Dict{String,Any}("key" => key, "numericValue" => v) :
        v isa AbstractVector{<:AbstractString} ? Dict{String,Any}("key" => key, "stringListValue" => Dict("values" => String.(v))) :
        throw(ArgumentError("$verb: custom_metadata $(repr(key)) is a string, a number or a vector of strings (got a $(typeof(v)))"))
    end
end

# White-space chunking: "Number of tokens per chunk must be between 0 and 512" (HTTP 400 for 5000, observed
# 2026-11-17; the reference sets "the upper limit to 2**9, which is 512 words"); a chunk of no tokens is none.
function _chunking(verb::String, max_tokens_per_chunk, max_overlap_tokens)::Union{Dict{String,Any},Nothing}
    isnothing(max_tokens_per_chunk) && isnothing(max_overlap_tokens) && return nothing
    ws = Dict{String,Any}()
    if !isnothing(max_tokens_per_chunk)
        1 <= max_tokens_per_chunk <= 512 || throw(ArgumentError(
            "$verb: max_tokens_per_chunk is 1 to 512 tokens (got $max_tokens_per_chunk)"))
        ws["maxTokensPerChunk"] = max_tokens_per_chunk
    end
    if !isnothing(max_overlap_tokens)
        max_overlap_tokens >= 0 || throw(ArgumentError(
            "$verb: max_overlap_tokens is a number of tokens >= 0 (got $max_overlap_tokens)"))
        ws["maxOverlapTokens"] = max_overlap_tokens
    end
    Dict{String,Any}("whiteSpaceConfig" => ws)
end

# The document settings an upload and an import share.
function _document_settings(verb::String, custom_metadata, max_tokens_per_chunk, max_overlap_tokens)::Dict{String,Any}
    d = Dict{String,Any}()
    isnothing(custom_metadata) || (d["customMetadata"] = _custom_metadata(verb, custom_metadata))
    (c = _chunking(verb, max_tokens_per_chunk, max_overlap_tokens)) === nothing || (d["chunkingConfig"] = c)
    d
end

# "Maximum file size / per document limit: 100 MB" (https://ai.google.dev/gemini-api/docs/file-search#rate-limits;
# discovery's media.uploadToFileSearchStore maxSize 104857600).
const _FILE_SEARCH_MAX_BYTES = 104_857_600

# ─── Stores ──────────────────────────────────────────────────────────────────

"""
    create_file_search_store(; display_name=nothing, embedding_model=nothing, service=GEMINIServiceEndpoint,
                             config=nothing, cancel=nothing)

Create an empty File Search store (fileSearchStores.create). `display_name` (at most 512 characters) is
the base of the store's name, which Gemini derives with a random suffix; `embedding_model` (e.g.
`"gemini-embedding-2"`, with or without its `models/` prefix) embeds its documents, Gemini's default
`gemini-embedding-001` when unset. Returns [`FileSearchStoreSuccess`](@ref), [`FileSearchFailure`](@ref)
or [`FileSearchCallError`](@ref).

A store keeps its documents "indefinitely until you manually delete" them; storing them is free, and
indexing a document bills its embeddings at the embedding model's rate ([File Search](@ref gemini_file_search)).
Throws `ArgumentError` before any request when `service` lacks `:file_search_stores` or the display name
is too long. `config` bounds the call (a single attempt); `cancel` (default: the ambient
[`with_cancel`](@ref) token) ends it with a [`UniLMCancelled`](@ref) in `cause`.
"""
function create_file_search_store(; display_name::Union{AbstractString,Nothing}=nothing,
                                  embedding_model::Union{AbstractString,Nothing}=nothing,
                                  service::ServiceEndpointSpec=GEMINIServiceEndpoint,
                                  config::Union{Nothing,RequestConfig}=nothing, cancel::Union{Nothing,CancelToken}=nothing)
    validate_capability(service, :file_search_stores, "File Search stores")
    # "The display name must be no more than 512 characters in length, including spaces."
    isnothing(display_name) || length(display_name) <= 512 || throw(ArgumentError(
        "create_file_search_store: display_name takes at most 512 characters (got $(length(display_name)))"))
    body = Dict{String,Any}()
    isnothing(display_name) || (body["displayName"] = display_name)
    isnothing(embedding_model) || (body["embeddingModel"] = "models/" * chopprefix(embedding_model, "models/"))
    _file_search_call(_store_success, "POST", _gemini_url(service, "fileSearchStores"), service, config, cancel; body)
end

"""
    retrieve_file_search_store(name; service=GEMINIServiceEndpoint, config=nothing, cancel=nothing)

A File Search store (fileSearchStores.get), by its name `"fileSearchStores/<id>"` or its id: its document
counts and size among it. Returns [`FileSearchStoreSuccess`](@ref), [`FileSearchFailure`](@ref) (a store
that does not exist is Gemini's 404) or [`FileSearchCallError`](@ref).
"""
function retrieve_file_search_store(name::AbstractString; service::ServiceEndpointSpec=GEMINIServiceEndpoint,
                                    config::Union{Nothing,RequestConfig}=nothing, cancel::Union{Nothing,CancelToken}=nothing)
    validate_capability(service, :file_search_stores, "File Search stores")
    url = _gemini_url(service, _store_name(name, "retrieve_file_search_store"))
    _file_search_call(_store_success, "GET", url, service, config, cancel)
end

"""
    list_file_search_stores(; limit=nothing, after=nothing, service=GEMINIServiceEndpoint, config=nothing, cancel=nothing)

A page of the project's File Search stores (fileSearchStores.list), oldest first: `limit` is the page
size, 1 to 20 (else `ArgumentError` before any request; Gemini's default is 10), `after` the previous
page's `raw["nextPageToken"]`. Returns [`FileSearchStoreListSuccess`](@ref), [`FileSearchFailure`](@ref)
or [`FileSearchCallError`](@ref).
"""
function list_file_search_stores(; limit::Union{Integer,Nothing}=nothing, after::Union{AbstractString,Nothing}=nothing,
                                 service::ServiceEndpointSpec=GEMINIServiceEndpoint,
                                 config::Union{Nothing,RequestConfig}=nothing, cancel::Union{Nothing,CancelToken}=nothing)
    validate_capability(service, :file_search_stores, "File Search stores")
    url = _gemini_url(service, "fileSearchStores") * _file_search_page_query("list_file_search_stores", limit, after)
    _file_search_call(_store_page, "GET", url, service, config, cancel)
end

"""
    delete_file_search_store(name; force=false, service=GEMINIServiceEndpoint, config=nothing, cancel=nothing)

Delete a File Search store (fileSearchStores.delete). A store that holds documents is deleted only with
`force=true`, which deletes them too: without it Gemini answers HTTP 400 "Cannot delete non-empty
FileSearchStore", a [`FileSearchFailure`](@ref). Returns [`FileSearchDeleteSuccess`](@ref),
`FileSearchFailure` or [`FileSearchCallError`](@ref).
"""
function delete_file_search_store(name::AbstractString; force::Bool=false, service::ServiceEndpointSpec=GEMINIServiceEndpoint,
                                  config::Union{Nothing,RequestConfig}=nothing, cancel::Union{Nothing,CancelToken}=nothing)
    validate_capability(service, :file_search_stores, "File Search stores")
    full = _store_name(name, "delete_file_search_store")
    _file_search_call(_deleted(full), "DELETE", _gemini_url(service, full) * (force ? "?force=true" : ""), service, config, cancel)
end

# ─── Documents into a store ──────────────────────────────────────────────────

"""
    upload_to_file_search_store(store, path; display_name=nothing, mime_type=nothing, custom_metadata=nothing,
                                max_tokens_per_chunk=nothing, max_overlap_tokens=nothing,
                                service=GEMINIServiceEndpoint, config=nothing, cancel=nothing)

Upload the local file `path` into the File Search store `store` (its name or id) as a document
(media.uploadToFileSearchStore), through Gemini's resumable upload: a start request with the document's
settings, then the bytes. Gemini chunks, embeds and indexes it in a long-running operation: the result's
[`FileSearchOperation`](@ref) names the document, and [`poll_file_search_operation`](@ref) waits for it to
be done. Returns [`FileSearchOperationSuccess`](@ref), [`FileSearchFailure`](@ref) or
[`FileSearchCallError`](@ref).

- `display_name`: the document's display name; `mime_type`: the file's MIME type, else the one its
  extension gives the upload (and Gemini infers the document's).
- `custom_metadata`: at most 20 key-value pairs, each value a string, a number or a vector of strings,
  which a search's `metadata_filter` reads ([`gemini_file_search`](@ref)).
- `max_tokens_per_chunk` (1 to 512) and `max_overlap_tokens` (≥ 0): white-space chunking, Gemini's
  defaults when unset.

Throws `ArgumentError` before any request for a missing file, a file over 100 MB (Gemini's per-document
limit), or a setting outside these bounds. Audio and video are not indexed ("Audio and video formats are
not currently supported"). A start that names no session address is a `FileSearchCallError`. `config`
bounds the call, both requests (a single attempt: an upload is never retried); `cancel` (default: the
ambient [`with_cancel`](@ref) token) ends it with a [`UniLMCancelled`](@ref) in `cause`, and a token
cancelled before the call reads no file.
"""
function upload_to_file_search_store(store::AbstractString, path::AbstractString;
        display_name::Union{AbstractString,Nothing}=nothing, mime_type::Union{AbstractString,Nothing}=nothing,
        custom_metadata::Union{AbstractDict,Nothing}=nothing, max_tokens_per_chunk::Union{Integer,Nothing}=nothing,
        max_overlap_tokens::Union{Integer,Nothing}=nothing, service::ServiceEndpointSpec=GEMINIServiceEndpoint,
        config::Union{Nothing,RequestConfig}=nothing, cancel::Union{Nothing,CancelToken}=nothing)
    verb = "upload_to_file_search_store"
    validate_capability(service, :file_search_stores, "File Search stores")
    name = _store_name(store, verb)
    isfile(path) || throw(ArgumentError("$verb: file not found: $path"))
    filesize(path) <= _FILE_SEARCH_MAX_BYTES || throw(ArgumentError(
        "$verb: a File Search document takes at most 100 MB (104857600 bytes; got $(filesize(path)))"))
    settings = _document_settings(verb, custom_metadata, max_tokens_per_chunk, max_overlap_tokens)
    isnothing(display_name) || (settings["displayName"] = display_name)
    isnothing(mime_type) || (settings["mimeType"] = mime_type)
    cfg = _resolve_config(config); t0 = time_ns(); tok = _resolve_cancel(cancel)
    try
        iscancelled(tok) && throw(UniLMCancelled(:token, _elapsed_s(t0)))
        bytes = read(path)
        start = _http("POST", _gemini_upload_root(service) * "/" * _gemini_path(name) * ":uploadToFileSearchStore",
            vcat(auth_header(service), ["X-Goog-Upload-Protocol" => "resumable", "X-Goog-Upload-Command" => "start",
                 "X-Goog-Upload-Header-Content-Length" => string(length(bytes)),
                 "X-Goog-Upload-Header-Content-Type" => String(something(mime_type, _mime_for(path)))]),
            JSON.json(settings); cfg, remaining=_remaining_s(cfg, t0), cancel=tok)
        start.status == 200 || return _failure(FileSearchFailure, start)
        session = HTTP.header(start, "X-Goog-Upload-URL", "")
        isempty(session) && throw(ArgumentError("Gemini's upload start answered no X-Goog-Upload-URL"))
        # As in the Files API's resumable upload, the key goes with the start only.
        done = _http("POST", session, ["X-Goog-Upload-Offset" => "0", "X-Goog-Upload-Command" => "upload, finalize"], bytes;
                     cfg, remaining=_remaining_s(cfg, t0), cancel=tok)
        done.status == 200 || return _failure(FileSearchFailure, done)
        _operation_success(JSON.parse(done.body; dicttype=Dict{String,Any}))
    catch e
        e isa InterruptException && rethrow()
        _callerr(FileSearchCallError, e)
    end
end

"""
    import_to_file_search_store(store, file; custom_metadata=nothing, max_tokens_per_chunk=nothing,
                                max_overlap_tokens=nothing, service=GEMINIServiceEndpoint, config=nothing, cancel=nothing)

Import a file of Gemini's Files API ([`upload_file`](@ref)`(path; service=GEMINIServiceEndpoint)`) into the
File Search store `store` as a document (fileSearchStores.importFile): `file` is its id or its name,
`"files/<id>"`. The settings are [`upload_to_file_search_store`](@ref)'s. The document outlives the file,
which Gemini deletes after 48 hours. Returns [`FileSearchOperationSuccess`](@ref) — the operation names
its document once done ([`poll_file_search_operation`](@ref)) —, [`FileSearchFailure`](@ref) or
[`FileSearchCallError`](@ref).
"""
function import_to_file_search_store(store::AbstractString, file::AbstractString;
        custom_metadata::Union{AbstractDict,Nothing}=nothing, max_tokens_per_chunk::Union{Integer,Nothing}=nothing,
        max_overlap_tokens::Union{Integer,Nothing}=nothing, service::ServiceEndpointSpec=GEMINIServiceEndpoint,
        config::Union{Nothing,RequestConfig}=nothing, cancel::Union{Nothing,CancelToken}=nothing)
    verb = "import_to_file_search_store"
    validate_capability(service, :file_search_stores, "File Search stores")
    name = _store_name(store, verb)
    body = _document_settings(verb, custom_metadata, max_tokens_per_chunk, max_overlap_tokens)
    body["fileName"] = _gemini_resource_name("files", file, "$verb: file")
    _file_search_call(_operation_success, "POST", _gemini_url(service, name, ":importFile"), service, config, cancel; body)
end

"""
    retrieve_file_search_operation(name; service=GEMINIServiceEndpoint, config=nothing, cancel=nothing)

The state of the operation that puts a document into a store (fileSearchStores.operations.get, and
fileSearchStores.upload.operations.get for an upload's), by its whole name. Returns
[`FileSearchOperationSuccess`](@ref), [`FileSearchFailure`](@ref) or [`FileSearchCallError`](@ref).
"""
function retrieve_file_search_operation(name::AbstractString; service::ServiceEndpointSpec=GEMINIServiceEndpoint,
                                        config::Union{Nothing,RequestConfig}=nothing, cancel::Union{Nothing,CancelToken}=nothing)
    validate_capability(service, :file_search_stores, "File Search stores")
    url = _gemini_url(service, _gemini_nested_name(name, _OPERATION_SHAPES, "retrieve_file_search_operation"))
    _file_search_call(_operation_success, "GET", url, service, config, cancel)
end

"""
    poll_file_search_operation(name; interval=2.0, timeout=300.0, service=GEMINIServiceEndpoint, config=nothing,
                               cancel=nothing)

Read the operation `name` until it is `done` and return that [`FileSearchOperationSuccess`](@ref): its
`document` is ready, or its `error` says why not. `timeout` bounds the wall-clock time of the whole poll
(`Inf` waits indefinitely), `interval` is the pause between reads; both must be positive, else
`ArgumentError`. A transient failure (a retryable status, a read that timed out) is polled through; any
other is returned as it came. When the time runs out, or `cancel` (default: the ambient
[`with_cancel`](@ref) token) is cancelled, the result is a [`FileSearchCallError`](@ref) whose `cause` is
the `UniLMTimeout` or [`UniLMCancelled`](@ref) and whose `last_observed` is the last operation seen.
`config` bounds each read.
"""
function poll_file_search_operation(name::AbstractString; interval::Real=2.0, timeout::Real=300.0,
                                    service::ServiceEndpointSpec=GEMINIServiceEndpoint,
                                    config::Union{Nothing,RequestConfig}=nothing, cancel::Union{Nothing,CancelToken}=nothing)
    validate_capability(service, :file_search_stores, "File Search stores")
    op = _gemini_nested_name(name, _OPERATION_SHAPES, "poll_file_search_operation")
    _poll(cfg -> retrieve_file_search_operation(op; service, config=cfg), FileSearchOperationSuccess, r -> r.response.done,
          (seen, why) -> FileSearchCallError(error=_operation_poll_end(op, why), cause=why,
                                             last_observed=isnothing(seen) ? nothing : seen.response);
          interval, timeout, config, cancel)
end
_operation_poll_end(op::String, why::UniLMTimeout)::String =
    "poll_file_search_operation timeout: $op was not done within $(why.limit) s"
_operation_poll_end(op::String, why::UniLMCancelled)::String =
    "poll_file_search_operation cancelled after $(round(why.elapsed; digits=3)) s: $op"

# ─── Documents ───────────────────────────────────────────────────────────────

"""
    list_file_search_documents(store; limit=nothing, after=nothing, service=GEMINIServiceEndpoint, config=nothing,
                               cancel=nothing)

A page of the documents in a File Search store (fileSearchStores.documents.list), oldest first: `store` is
its name or id, `limit` the page size, 1 to 20 (else `ArgumentError` before any request; Gemini's default
is 10), `after` the previous page's `raw["nextPageToken"]`. Returns
[`FileSearchDocumentListSuccess`](@ref), [`FileSearchFailure`](@ref) or [`FileSearchCallError`](@ref).
"""
function list_file_search_documents(store::AbstractString; limit::Union{Integer,Nothing}=nothing,
        after::Union{AbstractString,Nothing}=nothing, service::ServiceEndpointSpec=GEMINIServiceEndpoint,
        config::Union{Nothing,RequestConfig}=nothing, cancel::Union{Nothing,CancelToken}=nothing)
    verb = "list_file_search_documents"
    validate_capability(service, :file_search_stores, "File Search stores")
    url = _gemini_url(service, _store_name(store, verb), "/documents") * _file_search_page_query(verb, limit, after)
    _file_search_call(_document_page, "GET", url, service, config, cancel)
end

"""
    retrieve_file_search_document(name; service=GEMINIServiceEndpoint, config=nothing, cancel=nothing)

A document of a File Search store (fileSearchStores.documents.get), by its whole name,
`"fileSearchStores/<store>/documents/<id>"`. Returns [`FileSearchDocumentSuccess`](@ref),
[`FileSearchFailure`](@ref) (a document that does not exist is Gemini's 404) or
[`FileSearchCallError`](@ref).
"""
function retrieve_file_search_document(name::AbstractString; service::ServiceEndpointSpec=GEMINIServiceEndpoint,
                                       config::Union{Nothing,RequestConfig}=nothing, cancel::Union{Nothing,CancelToken}=nothing)
    validate_capability(service, :file_search_stores, "File Search stores")
    url = _gemini_url(service, _gemini_nested_name(name, (_DOCUMENT_SHAPE,), "retrieve_file_search_document"))
    _file_search_call(_document_success, "GET", url, service, config, cancel)
end

"""
    delete_file_search_document(name; force=true, service=GEMINIServiceEndpoint, config=nothing, cancel=nothing)

Delete a document from its File Search store (fileSearchStores.documents.delete), by its whole name. A
document's chunks are its indexed content, and Gemini deletes a document that has chunks only with
`force` — every indexed document has them: without it the API answered HTTP 400 "Cannot delete non-empty
Document" — so `force` is `true` unless set `false`. Returns [`FileSearchDeleteSuccess`](@ref),
[`FileSearchFailure`](@ref) or [`FileSearchCallError`](@ref).
"""
function delete_file_search_document(name::AbstractString; force::Bool=true, service::ServiceEndpointSpec=GEMINIServiceEndpoint,
                                     config::Union{Nothing,RequestConfig}=nothing, cancel::Union{Nothing,CancelToken}=nothing)
    validate_capability(service, :file_search_stores, "File Search stores")
    full = _gemini_nested_name(name, (_DOCUMENT_SHAPE,), "delete_file_search_document")
    _file_search_call(_deleted(full), "DELETE", _gemini_url(service, full) * (force ? "?force=true" : ""), service, config, cancel)
end
