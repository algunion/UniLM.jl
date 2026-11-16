# ============================================================================
# Gemini batches — the batch verbs of src/batch.jl on the Gemini API's batches and
# models.batchGenerateContent (https://ai.google.dev/gemini-api/docs/batch-api). A request is its Chat's
# generateContent body, keyed by its custom id in the request's metadata; a response decodes as a
# generateContent reply. Each shape below is one the API returned on 2026-11-16.
# ============================================================================

struct _GeminiBatches <: _BatchWire end
_batch_wire(::Type{GEMINIServiceEndpoint}) = _GeminiBatches()
# The API's host: the batch resources are under /v1beta, a responses file's bytes under /download/v1beta.
_batch_root(::Type{GEMINIServiceEndpoint}) = chopsuffix(GEMINI_NATIVE_BASE, "/v1beta")
_batches_url(::_GeminiBatches, service) = _batch_root(service) * "/v1beta/batches"
# A batch's id is its name without `batches/`; either names it.
_batch_url(w::_GeminiBatches, service, id::AbstractString, suffix::String="") =
    _batches_url(w, service) * "/" * _uripart(chopprefix(id, "batches/")) * suffix
_cancel_suffix(::_GeminiBatches) = ":cancel"
_batch_list_url(w::_GeminiBatches, service, limit, after) =
    _batch_query(_batches_url(w, service), "pageSize" => limit, "pageToken" => after)
_batch_ended(::_GeminiBatches, state) =
    state in ("BATCH_STATE_SUCCEEDED", "BATCH_STATE_FAILED", "BATCH_STATE_CANCELLED", "BATCH_STATE_EXPIRED")

# One model per batch, the one its URL names. A batched request takes no serviceTier: the API answered one
# (flex, standard or priority) with HTTP 400 "Request contains an invalid argument".
function _batch_input(::_GeminiBatches, service, requests, metadata, output_expires_after)
    _no_openai_options(service, metadata, output_expires_after)
    models = unique(chat.model for (_, chat) in requests)
    length(models) == 1 || throw(ArgumentError("create_batch: a Gemini batch runs one model, the one its URL " *
        "names; the requests name $(join(repr.(models), ", "))"))
    reqs = map(requests) do (id, chat)
        isnothing(chat.service_tier) || throw(ArgumentError("create_batch: request $(repr(id)) sets service_tier, " *
            "which a batched Gemini request does not take (the API answers HTTP 400): the batch has its own rates"))
        Dict("request" => JSON.parse(encode_request(service, chat); dicttype=Dict{String,Any}),
             "metadata" => Dict("key" => String(id)))
    end
    (; model=only(models), body=JSON.json(Dict("batch" => Dict("inputConfig" => Dict("requests" => Dict("requests" => reqs))))))
end

_create_batch(wire::_GeminiBatches, service, input, cfg, t0, tok) = _batch_object(wire,
    _batch_send("POST", _batch_root(service) * "/v1beta/models/" * _uripart(input.model) * ":batchGenerateContent",
                _batch_headers(service, true), cfg, t0, tok, input.body))

# A batch reply: an Operation — the GenerateContentBatch in its `metadata`, and `done` with `response` or `error`
# once it ends — from create, get and list; the GenerateContentBatch itself from an update. Its `state` is the
# status, its `batchStats` the tallies; an input file's name is its `inputConfig.fileName`, a responses file's
# its `output.responsesFile`.
function _parse_batch(::_GeminiBatches, d::AbstractDict)
    batch = haskey(d, "metadata") ? _batch_required(d, "metadata", AbstractDict) : d
    name = _batch_required(batch, "name", String)
    id = chopprefix(name, "batches/")
    (id == name || isempty(id)) && throw(ArgumentError("the Gemini batch name $(repr(name)) is not batches/{id}"))
    input = something(_batch_optional(batch, "inputConfig", AbstractDict), Dict{String,Any}())
    output = something(_batch_optional(batch, "output", AbstractDict), Dict{String,Any}())
    BatchObject(id=id, status=_batch_required(batch, "state", String),
                input_file_id=_batch_optional(input, "fileName", String),
                output_file_id=_batch_optional(output, "responsesFile", String),
                request_counts=Dict{String,Any}(something(_batch_optional(batch, "batchStats", AbstractDict), Dict{String,Any}())),
                raw=Dict{String,Any}(d))
end

# A page: `operations`, which the API leaves out when there are none (proto3 JSON omits an empty list), and
# `nextPageToken`, left out on the last page.
function _batch_list(wire::_GeminiBatches, d::Dict{String,Any})
    ops = something(_batch_optional(d, "operations", AbstractVector), [])
    token = _batch_optional(d, "nextPageToken", String)
    BatchList(data=BatchObject[_parse_batch(wire, _batch_entry(o)) for o in ops],
              has_more=!isnothing(token) && !isempty(token), raw=d)
end

# The cancel answers with an empty message: the batch it reports is the one a GET then returns.
function _cancel_batch(wire::_GeminiBatches, service, id, cfg, t0, tok)
    resp = _batch_send("POST", _batch_url(wire, service, id, _cancel_suffix(wire)), _batch_headers(service, false), cfg, t0, tok)
    resp.status == 200 || return _failure(BatchFailure, resp)
    _batch_reply(resp)
    _retrieve_batch(wire, service, id, cfg, t0, tok)
end

# The delete answers with an empty message, which names no batch: the deleted batch is the one the call named.
function _batch_deleted(::_GeminiBatches, id::String, ::Dict{String,Any})
    BatchDeleteSuccess(id=chopprefix(id, "batches/"), deleted=true)
end

# batches.updateGenerateContentBatch takes `priority` alone in its update mask ("The accepted fields are: *,
# priority"), and answers with the GenerateContentBatch. The API's int64 is a JSON string.
_update_batch(wire::_GeminiBatches, service, id, priority::Int, cfg, t0, tok) = _batch_object(wire,
    _batch_send("PATCH", _batch_url(wire, service, id, ":updateGenerateContentBatch?updateMask=priority"),
                _batch_headers(service, false), cfg, t0, tok, JSON.json(Dict("priority" => string(priority)))))

# One response — an inlined one, keyed in its `metadata`, or a responses-file line, keyed at its top — as the
# result of the request its key names: the reply, or the request's error status. A response without a key
# pairs with no request.
function _gemini_batch_item(service, model, x::Dict{String,Any}, keyed::AbstractDict)::Pair{String,_BatchItem}
    key = _batch_required(keyed, "key", String)
    reply = _batch_optional(x, "response", AbstractDict)
    key => (isnothing(reply) ? _batch_error(service, JSON.json(_batch_required(x, "error", AbstractDict)), model) :
            _batch_success(service, reply, model))
end
_gemini_inlined_item(service, model, x::Dict{String,Any}) =
    _gemini_batch_item(service, model, x, something(_batch_optional(x, "metadata", AbstractDict), Dict{String,Any}()))

# The results of a done batch: the responses inlined in the Operation's `response`, in the requests' order, for a
# batch of inline requests; else the JSONL file its `responsesFile` names. A batch that ended without results
# carries the Operation's `error` instead.
function _batch_results(wire::_GeminiBatches, service, id, cfg, t0, tok)
    resp = _batch_send("GET", _batch_url(wire, service, id), _batch_headers(service, false), cfg, t0, tok)
    resp.status == 200 || return _failure(BatchFailure, resp)
    op = _batch_reply(resp)
    b = _parse_batch(wire, op)
    _batch_optional(op, "done", Bool) === true ||
        throw(ArgumentError("Gemini batch $(b.id) has no results: it is not done (state $(b.status))"))
    out = _batch_optional(op, "response", AbstractDict)
    isnothing(out) && throw(ArgumentError("Gemini batch $(b.id) ended without results (state $(b.status)): " *
                                          JSON.json(something(_batch_optional(op, "error", AbstractDict), Dict()))))
    model = _batch_optional(get(op, "metadata", op), "model", String)
    model = isnothing(model) ? nothing : String(chopprefix(model, "models/"))
    inlined = _batch_optional(out, "inlinedResponses", AbstractDict)
    if !isnothing(inlined)
        items = something(_batch_optional(inlined, "inlinedResponses", AbstractVector), [])
        return BatchResultsSuccess(Pair{String,_BatchItem}[_gemini_inlined_item(service, model, _batch_entry(x)) for x in items])
    end
    file = _batch_required(out, "responsesFile", String)
    name = chopprefix(file, "files/")
    (name == file || isempty(name)) && throw(ArgumentError("the Gemini responses file $(repr(file)) is not files/{id}"))
    got = _batch_send("GET", _batch_root(service) * "/download/v1beta/files/" * _uripart(name) * ":download?alt=media",
                      _batch_headers(service, false), cfg, t0, tok)
    got.status == 200 || return _failure(BatchFailure, got)
    BatchResultsSuccess(Pair{String,_BatchItem}[_gemini_batch_item(service, model, x, x) for x in _jsonl_objects(got.body)])
end
