# ============================================================================
# Batch APIs — asynchronous bulk requests, billed at a discount, on three wires:
#   OpenAI     /v1/batches, over a JSONL file of requests uploaded through the Files API (purpose "batch")
#              https://developers.openai.com/api/reference/resources/batches
#   Anthropic  Message Batches, /v1/messages/batches (src/anthropic_batch.jl)
#              https://platform.claude.com/docs/en/api/messages/batches/create
#   Gemini     batches and models.batchGenerateContent (src/gemini_batch.jl)
#              https://ai.google.dev/gemini-api/docs/batch-api
# Every verb runs its endpoint's wire (`_batch_wire`). A batch of Chats is encoded by the provider's Chat
# encoder, and its results decoded by the provider's Chat decoder into the results `chatrequest!` returns.
# ============================================================================

"""
    BatchObject

A batch job, as [`create_batch`](@ref), [`retrieve_batch`](@ref), [`cancel_batch`](@ref),
[`update_batch`](@ref) and [`list_batches`](@ref) return it: `id`, `status`, `request_counts` (the
provider's tallies of the batch's requests, as sent), and `raw`, the reply as sent. A reply without a
string id or status is no batch: the call returns its [`BatchCallError`](@ref).

- OpenAI: `status` is the batch's `status` (`validating`, `in_progress`, `finalizing`, `completed`,
  `failed`, `expired`, `cancelling`, `cancelled`); `endpoint`, `input_file_id`, `output_file_id` and
  `error_file_id` are set.
- Anthropic: `status` is the Message Batch's `processing_status` (`in_progress`, `canceling`, `ended`);
  `request_counts` its `request_counts` (`processing`, `succeeded`, `errored`, `canceled`, `expired`).
- Gemini: `id` is the batch's name without its `batches/` prefix; `status` its `state`
  (`BATCH_STATE_PENDING`, `BATCH_STATE_RUNNING`, `BATCH_STATE_SUCCEEDED`, `BATCH_STATE_FAILED`,
  `BATCH_STATE_CANCELLED`, `BATCH_STATE_EXPIRED`); `request_counts` its `batchStats` (counts as strings, as
  the API sends 64-bit integers); `input_file_id` and `output_file_id` the names of its input file and
  responses file, when it has them.
"""
@kwdef struct BatchObject
    id::String
    status::Union{String,Nothing} = nothing
    endpoint::Union{String,Nothing} = nothing
    input_file_id::Union{String,Nothing} = nothing
    output_file_id::Union{String,Nothing} = nothing
    error_file_id::Union{String,Nothing} = nothing
    request_counts::Dict{String,Any} = Dict{String,Any}()
    raw::Dict{String,Any} = Dict{String,Any}()
end

"""
    BatchList

A page of [`BatchObject`](@ref)s from [`list_batches`](@ref); `has_more` signals that further pages are
available, and `raw` holds the page as sent. The next page starts `after` the page's last batch id on
OpenAI and Anthropic, and at its `raw["nextPageToken"]` on Gemini.
"""
@kwdef struct BatchList
    data::Vector{BatchObject}
    has_more::Bool = false
    raw::Dict{String,Any} = Dict{String,Any}()
end

"Successful create/retrieve/cancel/update result wrapping a [`BatchObject`](@ref)."
@kwdef struct BatchSuccess <: LLMRequestResponse; response::BatchObject; end
"Successful [`list_batches`](@ref) result wrapping a [`BatchList`](@ref)."
@kwdef struct BatchListSuccess <: LLMRequestResponse; response::BatchList; end
"Successful [`delete_batch`](@ref) result: the provider deleted the batch `id` (`deleted` is always `true`)."
@kwdef struct BatchDeleteSuccess <: LLMRequestResponse; id::String; deleted::Bool; end
"""
    BatchResultsSuccess

Successful [`batch_results`](@ref) result: `results`, each request's custom id paired with its result —
an [`LLMSuccess`](@ref), [`LLMFailure`](@ref) or [`LLMCallError`](@ref) — in the order the provider
returns them.
"""
struct BatchResultsSuccess <: LLMRequestResponse
    results::Vector{Pair{String,Union{LLMSuccess,LLMFailure,LLMCallError}}}
end
"Batch API error result: HTTP `status`, the raw `response` body, and the `request_id` the service sent (`x-request-id`/`request-id` header), if any."
@kwdef struct BatchFailure <: LLMRequestResponse; response::String; status::Int; request_id::Union{String,Nothing} = nothing; end
"Batch API call that produced no usable reply (transport failure, timeout, or a 200 that could not be decoded); `cause` is the underlying exception — a [`UniLMTimeout`](@ref) for a timeout. `last_observed` is set only when [`poll_batch`](@ref) runs out of time or is cancelled: the last [`BatchObject`](@ref) it saw, if any."
@kwdef struct BatchCallError <: LLMRequestResponse; error::String; status::Union{Int,Nothing} = nothing; cause::Union{Nothing,Exception} = nothing; last_observed::Union{Nothing,BatchObject} = nothing; end

_transient(r::BatchFailure) = _is_retryable(r.status, r.response)
_transient(r::BatchCallError) = _per_attempt_timeout(r.cause)

const _BatchItem = Union{LLMSuccess,LLMFailure,LLMCallError}

# ─── Wires ───────────────────────────────────────────────────────────────────
# How a provider runs batches. An endpoint with no wire of its own speaks OpenAI's, /v1/batches under its
# base URL, as every OpenAI-wire platform endpoint does.
abstract type _BatchWire end
struct _OpenAIBatches <: _BatchWire end
_batch_wire(_) = _OpenAIBatches()

# The URL a provider's batch paths hang from; a function so tests can route an endpoint of their own to a
# local server.
_batch_root(service) = _api_base_url(service)

# The headers of a batch request; `create` marks the request that creates the batch.
_batch_headers(service, ::Bool) = auth_header(service)

_batches_url(::_OpenAIBatches, service) = _batch_root(service) * BATCHES_PATH
_batch_url(wire::_BatchWire, service, id::AbstractString, suffix::String="") =
    _batches_url(wire, service) * "/" * _uripart(id) * suffix
_cancel_suffix(::_BatchWire) = "/cancel"

function _batch_query(url::String, params::Pair{String}...)::String
    set = String[]
    for (k, v) in params
        isnothing(v) || push!(set, k * "=" * _uripart(v))
    end
    isempty(set) ? url : url * "?" * join(set, "&")
end
_batch_list_url(w::_OpenAIBatches, service, limit, after) = _batch_query(_batches_url(w, service), "limit" => limit, "after" => after)

# The statuses after which a batch changes no more.
_batch_ended(::_OpenAIBatches, status) = status in ("completed", "failed", "cancelled", "expired")

# An operation a wire does not serve, refused before any request.
_batch_unsupported(::_BatchWire, ::Symbol) = nothing
_batch_unsupported(::_OpenAIBatches, verb::Symbol) = throw(ArgumentError(verb === :delete ?
    "delete_batch: OpenAI's Batch API has no delete, and a batch stays listed; cancel_batch stops one, and " *
    "delete_file removes its input, output and error files" :
    "update_batch: OpenAI's Batch API has no update; cancel the batch and create another"))

# ─── Replies ─────────────────────────────────────────────────────────────────
# A reply is the documented object or a malformed one: a field the reference requires is never read with a
# default, and a field of another JSON type is never coerced.

function _batch_reply(resp::HTTP.Response)::Dict{String,Any}
    d = JSON.parse(resp.body; dicttype=Dict{String,Any})
    d isa Dict{String,Any} || throw(ArgumentError("the batch reply is not a JSON object"))
    d
end

function _batch_required(d::AbstractDict, key::String, ::Type{T}) where {T}
    v = get(d, key, nothing)
    v isa T || throw(ArgumentError("the batch reply has no $(repr(key)) of the documented type " *
                                   "(got $(isnothing(v) ? "none" : "a JSON " * _json_type(v)))"))
    v
end

function _batch_optional(d::AbstractDict, key::String, ::Type{T}) where {T}
    v = get(d, key, nothing)
    isnothing(v) || v isa T || throw(ArgumentError("the batch reply's $(repr(key)) is a JSON $(_json_type(v)), " *
                                                   "not the documented type"))
    v
end

_batch_entry(e)::Dict{String,Any} = e isa Dict{String,Any} ? e : throw(ArgumentError("a batch reply entry is not a JSON object"))

# The JSON objects of a JSONL body, one per non-blank line.
_jsonl_objects(body)::Vector{Dict{String,Any}} =
    [_batch_entry(JSON.parse(line; dicttype=Dict{String,Any})) for line in eachline(IOBuffer(body)) if !isempty(strip(line))]

# OpenAI's Batch object: `id` and `status` are required.
_parse_batch(::_OpenAIBatches, d::AbstractDict) = BatchObject(id=_batch_required(d, "id", String),
    status=_batch_required(d, "status", String), endpoint=_batch_optional(d, "endpoint", String),
    input_file_id=_batch_optional(d, "input_file_id", String), output_file_id=_batch_optional(d, "output_file_id", String),
    error_file_id=_batch_optional(d, "error_file_id", String),
    request_counts=Dict{String,Any}(something(_batch_optional(d, "request_counts", AbstractDict), Dict{String,Any}())),
    raw=Dict{String,Any}(d))

# A page, as OpenAI and Anthropic list it: the batches in `data`, and `has_more`, both required.
_batch_list(wire::_BatchWire, d::Dict{String,Any}) = BatchList(
    data=BatchObject[_parse_batch(wire, _batch_entry(b)) for b in _batch_required(d, "data", AbstractVector)],
    has_more=_batch_required(d, "has_more", Bool), raw=d)

# ─── The verbs' common path ──────────────────────────────────────────────────

# One batch call: its config and cancel token resolved, and any throw — a malformed reply, a transport failure,
# a timeout, a cancel — returned as the BatchCallError it is.
function _batch_call(f::Function, config::Union{Nothing,RequestConfig}, cancel::Union{Nothing,CancelToken})
    cfg = _resolve_config(config); t0 = time_ns(); tok = _resolve_cancel(cancel)
    try
        f(cfg, t0, tok)
    catch e
        e isa InterruptException && rethrow()
        _callerr(BatchCallError, e)
    end
end

_batch_send(method::String, url::String, headers, cfg::RequestConfig, t0::UInt64, tok, body=UInt8[]) =
    _http(method, url, headers, body; cfg, remaining=_remaining_s(cfg, t0), cancel=tok)

_batch_object(wire::_BatchWire, resp::HTTP.Response) =
    resp.status == 200 ? BatchSuccess(response=_parse_batch(wire, _batch_reply(resp))) : _failure(BatchFailure, resp)

_retrieve_batch(wire::_BatchWire, service, id, cfg, t0, tok) =
    _batch_object(wire, _batch_send("GET", _batch_url(wire, service, id), _batch_headers(service, false), cfg, t0, tok))

# OpenAI and Anthropic answer a cancel with the batch.
_cancel_batch(wire::_BatchWire, service, id, cfg, t0, tok) = _batch_object(wire,
    _batch_send("POST", _batch_url(wire, service, id, _cancel_suffix(wire)), _batch_headers(service, false), cfg, t0, tok))

# The config of a later request of the same call: what is left of `cfg`'s total deadline.
function _batch_rest(cfg::RequestConfig, t0::UInt64)::RequestConfig
    left = _remaining_s(cfg, t0)
    left > 0 || throw(UniLMTimeout(:deadline, _elapsed_s(t0), cfg.total_deadline))
    RequestConfig(cfg; total_deadline=left)
end

# A Files API failure met on the way, as the batch call's own.
_as_batch_result(r::FileFailure) = BatchFailure(response=r.response, status=r.status, request_id=r.request_id)
_as_batch_result(r::FileCallError) = BatchCallError(error=r.error, status=r.status, cause=r.cause)

# ─── A batch of Chats ────────────────────────────────────────────────────────

# The endpoint the batch runs on: every request's Chat names it, and so does `service` when given.
function _batch_service(requests::AbstractVector, service)
    isempty(requests) && throw(ArgumentError("create_batch: a batch needs at least one request"))
    svc = something(service, first(requests).second.service)
    for (id, chat) in requests
        chat.service == svc || throw(ArgumentError("create_batch: request $(repr(id)) is a Chat for " *
            "$(repr(chat.service)), and the batch runs on $(repr(svc)); a batch runs on one endpoint"))
    end
    svc
end

# Every custom id names one request: none is empty and none repeats.
function _check_custom_ids(ids::AbstractVector{<:AbstractString})
    i = findfirst(isempty, ids)
    isnothing(i) || throw(ArgumentError("create_batch: request $i has an empty custom id"))
    seen = Set{String}()
    for id in ids
        id in seen && throw(ArgumentError("create_batch: custom ids must be unique; $(repr(id)) names two requests"))
        push!(seen, id)
    end
end
# A provider's own rule for its custom ids.
_batch_custom_ids(::_BatchWire, ids) = nothing

# A batched request is not streamed: its result comes back with the batch's results.
_check_batch_chat(id, chat::Chat) = chat.stream === true && throw(ArgumentError(
    "create_batch: request $(repr(id)) is a stream (stream=true), and a batched request is not streamed: its " *
    "result comes back with the batch's results; leave stream unset"))

# metadata and output_expires_after are options of OpenAI's batch create.
_no_openai_options(service, metadata, output_expires_after) =
    isnothing(metadata) && isnothing(output_expires_after) || throw(ArgumentError(
        "create_batch: metadata and output_expires_after are options of OpenAI's Batch API; " *
        "$(repr(service)) takes neither"))

# output_expires_after: the seconds after the batch's output or error file is created, its only anchor, that the
# file expires — "between 3600 (1 hour) and 2592000 (30 days)" (BatchFileExpirationAfter in the Batch reference).
_batch_expiry(::Nothing) = nothing
function _batch_expiry(seconds::Integer)::Dict{Symbol,Any}
    3600 <= seconds <= 2_592_000 || throw(ArgumentError("create_batch: output_expires_after is the seconds after a " *
        "batch's output or error file is created that it expires, from 3600 (1 hour) to 2592000 (30 days) (got $seconds)"))
    Dict{Symbol,Any}(:anchor => "created_at", :seconds => seconds)
end

# The requests as OpenAI's input file: one line per request, its custom id and the Chat Completions body the
# Chat's encoder builds. "Each input file can only include requests to a single model"
# (https://developers.openai.com/api/docs/guides/batch).
function _batch_input(::_OpenAIBatches, service, requests, metadata, output_expires_after)
    models = unique(chat.model for (_, chat) in requests)
    length(models) == 1 || throw(ArgumentError("create_batch: an OpenAI batch's input file holds requests to a " *
        "single model; the requests name $(join(repr.(models), ", ")): create one batch per model"))
    io = IOBuffer()
    for (id, chat) in requests
        println(io, JSON.json(Dict("custom_id" => String(id), "method" => "POST", "url" => CHAT_COMPLETIONS_PATH,
                                   "body" => JSON.parse(encode_request(service, chat); dicttype=Dict{String,Any}))))
    end
    (; jsonl=takestring!(io), metadata, expires=_batch_expiry(output_expires_after))
end

# The input file goes up as a .jsonl file with purpose "batch", as the Batch API requires, then the batch is
# created from it. A failure after the upload leaves the file in place, under purpose "batch".
function _create_batch(::_OpenAIBatches, service, input, cfg, t0, tok)
    up = mktempdir() do dir
        path = joinpath(dir, "unilm-batch-input.jsonl")
        write(path, input.jsonl)
        upload_file(path, "batch"; service, config=cfg, cancel=tok)
    end
    up isa FileSuccess || return _as_batch_result(up)
    _create_from_file(service, up.response.id, CHAT_COMPLETIONS_PATH, "24h", input.metadata, input.expires, cfg, t0, tok)
end

function _create_from_file(service, input_file_id, endpoint, completion_window, metadata, expires, cfg, t0, tok)
    d = Dict{Symbol,Any}(:input_file_id => input_file_id, :endpoint => endpoint, :completion_window => completion_window)
    isnothing(metadata) || (d[:metadata] = metadata)
    isnothing(expires) || (d[:output_expires_after] = expires)
    wire = _OpenAIBatches()
    _batch_object(wire, _batch_send("POST", _batches_url(wire, service), _batch_headers(service, true), cfg, t0, tok, JSON.json(d)))
end

# ─── Results ─────────────────────────────────────────────────────────────────
# A batch keeps no Chat of its own: each result's `self` is a stand-in on the batch's endpoint, naming the model
# the provider reports for the request or the batch, else the endpoint's default model.
_batch_chat(service, model::Union{String,Nothing}) = Chat(service=service, model=something(model, ""))

# A reply as `chatrequest!` reads it, by the endpoint's decoder, with its usage at the tier "batch": the Batch
# API bills it at the provider's Batch rates, whatever tier the reply names (a batched Gemini reply says
# "standard"). A reply the decoder cannot read is that request's call error.
function _batch_success(service, body::AbstractDict, model::Union{String,Nothing})::_BatchItem
    d = try
        decode_response(service, HTTP.Response(200, JSON.json(body)))
    catch e
        e isa InterruptException && rethrow()
        return LLMCallError(error=_error_text(service, e), self=_batch_chat(service, model), cause=e isa Exception ? e : nothing)
    end
    chat = _batch_chat(service, something(_decoded_model(d), model, Some(nothing)))
    u = _result_usage(service, chat, d.usage)
    LLMSuccess(; message=d.message, self=chat, model=_billed_response(service, chat.model, _decoded_model(d)),
               usage=isnothing(u) ? nothing : TokenUsage(; _usage_fields(u)..., service_tier="batch"), _decoded_results(d)...)
end

_batch_error(service, text::String, model; request_id=nothing)::_BatchItem =
    LLMCallError(error=text, self=_batch_chat(service, model), request_id=request_id)

# A line of an OpenAI output or error file: its custom id, and `response` — the request's HTTP status, request id
# and body — or, for a request that got none, `error`.
function _openai_batch_item(service, model, line::Dict{String,Any})::Pair{String,_BatchItem}
    id = _batch_required(line, "custom_id", String)
    resp = _batch_optional(line, "response", AbstractDict)
    isnothing(resp) && return id => _batch_error(service, JSON.json(_batch_required(line, "error", AbstractDict)), model)
    status = _batch_required(resp, "status_code", Integer)
    body = _batch_required(resp, "body", AbstractDict)
    id => (status == 200 ? _batch_success(service, body, model) :
           LLMFailure(status=status, response=JSON.json(body), self=_batch_chat(service, model),
                      request_id=_batch_optional(resp, "request_id", String)))
end

# The results decode as Chat Completions replies, the requests create_batch(requests) sends. A batch of another
# endpoint's requests, created from a file of them, keeps its results in its output file as OpenAI writes them.
function _chat_completions_batch(b::BatchObject)
    b.endpoint == CHAT_COMPLETIONS_PATH && return nothing
    file = isnothing(b.output_file_id) ? "output_file_id) once it has one" : repr(b.output_file_id) * ")"
    throw(ArgumentError("batch_results: batch $(b.id) " * (isnothing(b.endpoint) ? "names no endpoint" : "runs $(b.endpoint)") *
        ", and batch_results decodes $CHAT_COMPLETIONS_PATH replies alone; read its results with file_content($file"))
end

# OpenAI's results: the output file's lines, then the error file's, once the batch has either.
function _batch_results(wire::_OpenAIBatches, service, id, cfg, t0, tok)
    got = _retrieve_batch(wire, service, id, cfg, t0, tok)
    got isa BatchSuccess || return got
    b = got.response
    _chat_completions_batch(b)
    files = String[f for f in (b.output_file_id, b.error_file_id) if !isnothing(f)]
    isempty(files) && throw(ArgumentError("batch $id has no output or error file yet (status $(b.status))"))
    model = _batch_optional(b.raw, "model", String)
    results = Pair{String,_BatchItem}[]
    for f in files
        c = file_content(f; service, config=_batch_rest(cfg, t0), cancel=tok)
        c isa FileContentSuccess || return _as_batch_result(c)
        append!(results, (_openai_batch_item(service, model, line) for line in _jsonl_objects(c.content)))
    end
    BatchResultsSuccess(results)
end

# ─── Verbs ───────────────────────────────────────────────────────────────────

const _BATCH_CONFIG_DOC = """
Pass `config::Union{Nothing,RequestConfig}` to override the timeout budget for this call (a single bounded attempt; `max_attempts` does not apply).

Pass `cancel::Union{Nothing,CancelToken}` (default: the ambient [`with_cancel`](@ref) token) to make the call cancellable: once the token is cancelled it returns a `BatchCallError` whose `cause` is a [`UniLMCancelled`](@ref) — without sending anything when the token was already cancelled, at once when mid-request (never retried)."""

"""
    create_batch(input_file_id, endpoint; completion_window="24h", metadata=nothing, output_expires_after=nothing,
                 service=OPENAIServiceEndpoint)

Create an OpenAI batch job from an uploaded file. `endpoint` is e.g. `"/v1/chat/completions"`,
`"/v1/responses"`, or `"/v1/embeddings"`. `input_file_id` comes from `upload_file(path, "batch")`.
`output_expires_after` is the number of seconds, from 3600 (1 hour) to 2592000 (30 days), after which the
batch's output and error files expire, counted from each file's creation; outside that range it throws
`ArgumentError` before any request. This is OpenAI's form: on an Anthropic or Gemini endpoint it throws
`ArgumentError` before any request; pass the requests themselves, `create_batch(requests; service)`.
[`batch_results`](@ref) decodes a `"/v1/chat/completions"` batch; another endpoint's results are its
output file, which [`file_content`](@ref)`(output_file_id)` reads.

$_BATCH_CONFIG_DOC
"""
function create_batch(input_file_id::String, endpoint::String; completion_window::String="24h",
    metadata::Union{AbstractDict,Nothing}=nothing, output_expires_after::Union{Integer,Nothing}=nothing,
    service::ServiceEndpointSpec=OPENAIServiceEndpoint, config::Union{Nothing,RequestConfig}=nothing,
    cancel::Union{Nothing,CancelToken}=nothing)
    validate_capability(service, :batch, "Batch API")
    _batch_wire(service) isa _OpenAIBatches || throw(ArgumentError("create_batch(input_file_id, endpoint) is " *
        "OpenAI's file-based batch; $(repr(service)) batches requests: create_batch([custom_id => chat, …]; service)"))
    expires = _batch_expiry(output_expires_after)
    _batch_call(config, cancel) do cfg, t0, tok
        _create_from_file(service, input_file_id, endpoint, completion_window, metadata, expires, cfg, t0, tok)
    end
end

"""
    create_batch(requests; service=nothing, metadata=nothing, output_expires_after=nothing)

Create a batch of chat requests: `requests` pairs each request's custom id with its [`Chat`](@ref), and
the batch runs on the endpoint the Chats name (`service`, when given, must be it). Each provider takes the
Chats as its own Chat requests, encoded as [`chatrequest!`](@ref) encodes them:

- OpenAI: a JSONL file of `/v1/chat/completions` requests, uploaded with purpose `"batch"` (one more
  request, which leaves the file in place should the create fail), then the batch;
  `metadata` and `output_expires_after` are its create's options. One model per batch: "each input file
  can only include requests to a single model".
- Anthropic: a Message Batch whose requests' `params` are the Messages bodies. A custom id is 1 to 64
  letters, digits, hyphens and underscores; fast mode (`service_tier="fast"`) is not batched. The
  endpoint's `betas` go in the `anthropic-beta` header (e.g. `"output-300k-2026-03-24"`, which raises
  `max_tokens` to 300,000 on a batch).
- Gemini: inline requests to `models/{model}:batchGenerateContent`, each keyed by its custom id in its
  `metadata`; one model per batch, and no `service_tier` (the API answers a batched request that sets one
  with HTTP 400).

Throws `ArgumentError` before any request for no request, an empty or repeated custom id, Chats for
different endpoints, a streamed Chat (`stream=true`), a Chat its provider's encoder refuses, and the
provider rules above. Returns `BatchSuccess` (its [`BatchObject`](@ref)), `BatchFailure` or
`BatchCallError`; [`batch_results`](@ref) reads the results once the batch ends.

$_BATCH_CONFIG_DOC The config bounds the upload and the create together on OpenAI.
"""
function create_batch(requests::AbstractVector{<:Pair{<:AbstractString,Chat}};
    service::Union{Nothing,ServiceEndpointSpec}=nothing, metadata::Union{AbstractDict,Nothing}=nothing,
    output_expires_after::Union{Integer,Nothing}=nothing, config::Union{Nothing,RequestConfig}=nothing,
    cancel::Union{Nothing,CancelToken}=nothing)
    svc = _batch_service(requests, service)
    validate_capability(svc, :batch, "Batch API")
    wire = _batch_wire(svc)
    ids = String[first(p) for p in requests]
    _check_custom_ids(ids)
    _batch_custom_ids(wire, ids)
    foreach(p -> _check_batch_chat(p...), requests)
    input = _batch_input(wire, svc, requests, metadata, output_expires_after)
    _batch_call((cfg, t0, tok) -> _create_batch(wire, svc, input, cfg, t0, tok), config, cancel)
end

"""
    retrieve_batch(id; service=OPENAIServiceEndpoint)

The batch `id` on `service`: `BatchSuccess` (its [`BatchObject`](@ref)), `BatchFailure` or `BatchCallError`.
On Gemini `id` may also be the batch's name, `batches/{id}`.

$_BATCH_CONFIG_DOC
"""
function retrieve_batch(id::String; service::ServiceEndpointSpec=OPENAIServiceEndpoint, config::Union{Nothing,RequestConfig}=nothing,
    cancel::Union{Nothing,CancelToken}=nothing)
    validate_capability(service, :batch, "Batch API")
    wire = _batch_wire(service)
    _batch_call((cfg, t0, tok) -> _retrieve_batch(wire, service, id, cfg, t0, tok), config, cancel)
end

"""
    cancel_batch(id; service=OPENAIServiceEndpoint)

Ask the provider to cancel the batch `id`; the result is the batch as the provider then reports it
(OpenAI `cancelling`, Anthropic `canceling`; Gemini answers the cancel with an empty body, so the result is
the batch as a GET right after it returns it). Requests already processed keep their results.

$_BATCH_CONFIG_DOC
"""
function cancel_batch(id::String; service::ServiceEndpointSpec=OPENAIServiceEndpoint, config::Union{Nothing,RequestConfig}=nothing,
    cancel::Union{Nothing,CancelToken}=nothing)
    validate_capability(service, :batch, "Batch API")
    wire = _batch_wire(service)
    _batch_call((cfg, t0, tok) -> _cancel_batch(wire, service, id, cfg, t0, tok), config, cancel)
end

"""
    list_batches(; limit=nothing, after=nothing, service=OPENAIServiceEndpoint)

A page of the batches on `service`: `BatchListSuccess` (a [`BatchList`](@ref)),
`BatchFailure` or `BatchCallError`. `limit` caps the page; `after` starts it after the batch id `after` on
OpenAI and Anthropic (`after_id`), and at the page token `after` on Gemini (`pageToken`: the previous page's
`raw["nextPageToken"]`). A 200 without the listing's required fields is a `BatchCallError`.

$_BATCH_CONFIG_DOC
"""
function list_batches(; limit::Union{Int,Nothing}=nothing, after::Union{String,Nothing}=nothing, service::ServiceEndpointSpec=OPENAIServiceEndpoint, config::Union{Nothing,RequestConfig}=nothing,
    cancel::Union{Nothing,CancelToken}=nothing)
    validate_capability(service, :batch, "Batch API")
    wire = _batch_wire(service)
    _batch_call(config, cancel) do cfg, t0, tok
        resp = _batch_send("GET", _batch_list_url(wire, service, limit, after), _batch_headers(service, false), cfg, t0, tok)
        resp.status == 200 ? BatchListSuccess(response=_batch_list(wire, _batch_reply(resp))) : _failure(BatchFailure, resp)
    end
end

"""
    delete_batch(id; service)

Delete the batch `id` on Anthropic or Gemini: `BatchDeleteSuccess`, `BatchFailure` or `BatchCallError`.
Anthropic deletes a batch only once it has ended ("If you'd like to delete an in-progress batch, you must
first cancel it"); Gemini answers a delete with an empty body, so the result names the batch the call
deleted. OpenAI's Batch API has no delete: there `delete_batch` throws `ArgumentError` before any request.

$_BATCH_CONFIG_DOC
"""
function delete_batch(id::String; service::ServiceEndpointSpec, config::Union{Nothing,RequestConfig}=nothing,
    cancel::Union{Nothing,CancelToken}=nothing)
    validate_capability(service, :batch, "Batch API")
    wire = _batch_wire(service)
    _batch_unsupported(wire, :delete)
    _batch_call(config, cancel) do cfg, t0, tok
        resp = _batch_send("DELETE", _batch_url(wire, service, id), _batch_headers(service, false), cfg, t0, tok)
        resp.status == 200 ? _batch_deleted(wire, id, _batch_reply(resp)) : _failure(BatchFailure, resp)
    end
end

"""
    update_batch(id; priority, service)

Set the `priority` of the Gemini batch `id` (batches with a higher priority run first; negative values are
allowed): `BatchSuccess` with the updated batch, `BatchFailure` or `BatchCallError`. Priority is the one field
the API updates. OpenAI's and Anthropic's Batch APIs update no batch: there `update_batch` throws
`ArgumentError` before any request.

$_BATCH_CONFIG_DOC
"""
function update_batch(id::String; priority::Int, service::ServiceEndpointSpec, config::Union{Nothing,RequestConfig}=nothing,
    cancel::Union{Nothing,CancelToken}=nothing)
    validate_capability(service, :batch, "Batch API")
    wire = _batch_wire(service)
    _batch_unsupported(wire, :update)
    _batch_call((cfg, t0, tok) -> _update_batch(wire, service, id, priority, cfg, t0, tok), config, cancel)
end

"""
    batch_results(id; service=OPENAIServiceEndpoint)

The results of the batch `id`, once it has ended: `BatchResultsSuccess`, whose `results` pair each
request's custom id with its result, in the order the provider returns them (neither OpenAI nor Anthropic
keeps the order of the requests; Gemini does), or the call's `BatchFailure` or `BatchCallError`.

Each result is what [`chatrequest!`](@ref) returns for its reply, decoded by the endpoint's decoder: an
[`LLMSuccess`](@ref) whose usage names the tier `"batch"`, so [`estimated_cost`](@ref) prices it at the
provider's Batch rates; an [`LLMFailure`](@ref) for an OpenAI request answered with an HTTP error status; an
[`LLMCallError`](@ref) for a request with no reply — its `error` holds the provider's error object as JSON
(Anthropic's `errored` result, with its `request_id`; Gemini's status; an OpenAI error-file entry), or why
there is none (an Anthropic request `canceled` or `expired` before it ran) — and for a reply the decoder
cannot read. A result's `self` is a stand-in [`Chat`](@ref) on `service`, without messages: its model is the
one the provider names for the reply or the batch, else the endpoint's default model.

The results are read from OpenAI's output and error files ([`file_content`](@ref): output lines, then error
lines), Anthropic's results stream, and Gemini's inlined responses or its responses file. On OpenAI the
batch is a `"/v1/chat/completions"` one, as [`create_batch`](@ref)`(requests)` makes it: a batch of another
endpoint is a `BatchCallError` whose `ArgumentError` names `file_content(output_file_id)`, raised after the
batch is read and before any file is downloaded. A batch with no
results yet is a `BatchFailure` on Anthropic (its 404) and a `BatchCallError` naming the batch's status on
OpenAI and Gemini; so is a Gemini batch that ended without results (its error is in the message), and a
Gemini response without the custom id [`create_batch`](@ref) sets in its `metadata` (`key`).

$_BATCH_CONFIG_DOC The config bounds every request of the call together.
"""
function batch_results(id::String; service::ServiceEndpointSpec=OPENAIServiceEndpoint,
    config::Union{Nothing,RequestConfig}=nothing, cancel::Union{Nothing,CancelToken}=nothing)
    validate_capability(service, :batch, "Batch API")
    wire = _batch_wire(service)
    _batch_call((cfg, t0, tok) -> _batch_results(wire, service, id, cfg, t0, tok), config, cancel)
end

"""
    poll_batch(id; interval=10.0, timeout=86400.0, service=OPENAIServiceEndpoint, cancel=nothing)

Poll a batch until it reaches a terminal status: on OpenAI `completed`, `failed`, `cancelled` or `expired`;
on Anthropic `ended`; on Gemini `BATCH_STATE_SUCCEEDED`, `BATCH_STATE_FAILED`, `BATCH_STATE_CANCELLED` or
`BATCH_STATE_EXPIRED`.
`timeout` bounds the wall-clock time of the whole poll (`Inf` waits indefinitely);
`interval` is the pause between GETs. Both must be positive, else `ArgumentError`.

A transient failure — a status in the retryable set (408/429/500/502–504/520–524/529) or a
GET that timed out — is polled through; any other failure is returned as it came. When
the time runs out the result is a `BatchCallError` with `cause = UniLMTimeout(:deadline, …)`
and `last_observed` set to the last `BatchObject` seen (`nothing` if no GET succeeded).

Pass `cancel::Union{Nothing,CancelToken}` (default: the ambient [`with_cancel`](@ref)
token) to make the poll cancellable: a cancel ends it at once, mid-GET or mid-pause, with
a `BatchCallError` whose `cause` is a [`UniLMCancelled`](@ref) and `last_observed` as
above; a token cancelled before the poll sends nothing.

Pass `config::Union{Nothing,RequestConfig}` to bound each GET (a single attempt; its
`total_deadline` is capped at the time left).
"""
function poll_batch(id::String; interval::Real=10.0, timeout::Real=86400.0,
                    service::ServiceEndpointSpec=OPENAIServiceEndpoint,
                    config::Union{Nothing,RequestConfig}=nothing,
                    cancel::Union{Nothing,CancelToken}=nothing)
    wire = _batch_wire(service)
    _poll(cfg -> retrieve_batch(id; service, config=cfg), BatchSuccess,
          r -> _batch_ended(wire, r.response.status),
          (seen, why) -> BatchCallError(error=_poll_end_text("poll_batch", id, why, seen), cause=why,
                                        last_observed=isnothing(seen) ? nothing : seen.response);
          interval, timeout, config, cancel)
end
