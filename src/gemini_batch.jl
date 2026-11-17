# ============================================================================
# Gemini batches — the batch verbs of src/batch.jl on the Gemini API's batches,
# models.batchGenerateContent and models.asyncBatchEmbedContent
# (https://ai.google.dev/gemini-api/docs/batch-api). A request is its Chat's generateContent body (an
# Embeddings' embedContent body), keyed by its custom id in the request's metadata; a response decodes as a
# generateContent (embedContent) reply. Each shape below is one the API returned on 2026-11-16 (embeddings:
# 2026-11-17).
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
    (; model=only(models), method=":batchGenerateContent",
       body=JSON.json(Dict("batch" => Dict("inputConfig" => Dict("requests" => Dict("requests" => reqs))))))
end

# A batch of embeddings: one EmbedContentRequest per custom id, the body embeddingrequest! sends for one input,
# naming its model as a batchEmbedContents request does; one model per batch, the one its URL names.
function _batch_input(::_GeminiBatches, service, requests::AbstractVector{Pair{String,Embeddings}}, metadata,
                      output_expires_after)
    _no_openai_options(service, metadata, output_expires_after)
    models = unique(e.model for (_, e) in requests)
    length(models) == 1 || throw(ArgumentError("create_batch: a Gemini batch runs one model, the one its URL " *
        "names; the requests name $(join(repr.(models), ", "))"))
    reqs = map(requests) do (id, e)
        e.input isa AbstractVector && throw(ArgumentError("create_batch: request $(repr(id)) embeds " *
            "$(length(e.input)) inputs, and a request of a Gemini batch embeds one (an embedContent request): give " *
            "each input a custom id of its own"))
        _gemini_embed_options(e)
        Dict("request" => _gemini_embed_request(e, e.input; model=true), "metadata" => Dict("key" => id))
    end
    (; model=only(models), method=":asyncBatchEmbedContent",
       body=JSON.json(Dict("batch" => Dict("inputConfig" => Dict("requests" => Dict("requests" => reqs))))))
end

_create_batch(wire::_GeminiBatches, service, input, cfg, t0, tok) = _batch_object(wire,
    _batch_send("POST", string(_batch_root(service), "/v1beta/models/", _uripart(input.model), input.method::String),
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
# priority"), and answers with the GenerateContentBatch; a batch of embeddings takes the same update on
# updateEmbedContentBatch. The API answers the first on a batch of embeddings with HTTP 400 "Batch type does not
# match the request type." (2026-11-17), and that answer sends the update to the second. The API's int64 is a
# JSON string.
const _GEMINI_BATCH_TYPE_MISMATCH = "Batch type does not match the request type"
function _update_batch(wire::_GeminiBatches, service, id, priority::Int, cfg, t0, tok)
    patch(method) = _batch_send("PATCH", _batch_url(wire, service, id, method * "?updateMask=priority"),
        _batch_headers(service, false), cfg, t0, tok, JSON.json(Dict("priority" => string(priority))))
    resp = patch(":updateGenerateContentBatch")
    resp.status == 400 && occursin(_GEMINI_BATCH_TYPE_MISMATCH, String(copy(resp.body))) &&
        (resp = patch(":updateEmbedContentBatch"))
    _batch_object(wire, resp)
end

# A batch's name, its `displayName` ("The user-defined name of this batch"), set when it is created; the API
# updates none ("Invalid fields in update mask: display_name").
function _batch_display_name(::_GeminiBatches, _, input, name::String)
    isempty(strip(name)) && throw(ArgumentError("create_batch: display_name is empty: name the batch, or leave it unset"))
    body = JSON.parse(input.body; dicttype=Dict{String,Any})
    body["batch"]["displayName"] = name
    merge(input, (; body=JSON.json(body)))
end

# The kind of request a batch holds, by the type of its resource: an EmbedContentBatch's are Embeddings
# (`type.googleapis.com/google.ai.generativelanguage.v1main.EmbedContentBatch` in the Operation's metadata), any
# other's Chats.
_gemini_batch_kind(op::AbstractDict)::Type =
    endswith(something(_batch_optional(get(op, "metadata", op), "@type", String), ""), ".EmbedContentBatch") ? Embeddings : Chat

# One response — an inlined one, keyed in its `metadata`, or a responses-file line, keyed at its top — as the
# result of the request its key names: the reply, or the request's error status. A response without a key
# pairs with no request.
function _gemini_batch_item(kind::Type, service, model, x::Dict{String,Any},
                            keyed::AbstractDict)::Pair{String,Union{_BatchItem,_EmbeddingItem}}
    key = _batch_required(keyed, "key", String)
    reply = _batch_optional(x, "response", AbstractDict)
    key => (isnothing(reply) ? _batch_error(kind, service, JSON.json(_batch_required(x, "error", AbstractDict)), model) :
            _gemini_batch_reply(kind, service, reply, model))
end
_gemini_batch_reply(::Type{Chat}, service, reply, model) = _batch_success(service, reply, model)
_gemini_batch_reply(::Type{Embeddings}, service, reply, model) =
    _batch_embedding(r -> _decode_embeddings(_GeminiEmbeddings(), service, Dict{String,Any}(r)), service, reply, model)
_gemini_inlined_item(kind::Type, service, model, x::Dict{String,Any}) =
    _gemini_batch_item(kind, service, model, x, something(_batch_optional(x, "metadata", AbstractDict), Dict{String,Any}()))

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
    kind = _gemini_batch_kind(op)
    inlined = _batch_optional(out, "inlinedResponses", AbstractDict)
    if !isnothing(inlined)
        items = something(_batch_optional(inlined, "inlinedResponses", AbstractVector), [])
        return BatchResultsSuccess(Pair{String,Union{_BatchItem,_EmbeddingItem}}[
            _gemini_inlined_item(kind, service, model, _batch_entry(x)) for x in items])
    end
    file = _batch_required(out, "responsesFile", String)
    name = chopprefix(file, "files/")
    (name == file || isempty(name)) && throw(ArgumentError("the Gemini responses file $(repr(file)) is not files/{id}"))
    got = _batch_send("GET", _batch_root(service) * "/download/v1beta/files/" * _uripart(name) * ":download?alt=media",
                      _batch_headers(service, false), cfg, t0, tok)
    got.status == 200 || return _failure(BatchFailure, got)
    BatchResultsSuccess(Pair{String,Union{_BatchItem,_EmbeddingItem}}[
        _gemini_batch_item(kind, service, model, x, x) for x in _jsonl_objects(got.body)])
end
