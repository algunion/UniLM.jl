# ============================================================================
# Anthropic Message Batches — the batch verbs of src/batch.jl on /v1/messages/batches
# (https://platform.claude.com/docs/en/api/messages/batches/create; the guide:
# https://platform.claude.com/docs/en/build-with-claude/batch-processing). A request's `params` are its
# Chat's Messages body; a result's message decodes as a Messages reply.
# ============================================================================

struct _AnthropicBatches <: _BatchWire end
_batch_wire(::_AnthropicEndpoint) = _AnthropicBatches()
_batch_root(::_AnthropicEndpoint) = ANTHROPIC_BASE_URL
_batches_url(::_AnthropicBatches, service) = _batch_root(service) * "/v1/messages/batches"
_batch_list_url(w::_AnthropicBatches, service, limit, after) =
    _batch_query(_batches_url(w, service), "limit" => limit, "after_id" => after)
_batch_ended(::_AnthropicBatches, status) = status == "ended"

# "Can I update a batch after it's been submitted? No" (the batch processing guide's FAQ).
_batch_unsupported(::_AnthropicBatches, verb::Symbol) = verb === :update && throw(ArgumentError(
    "update_batch: a Message Batch cannot be modified once submitted; cancel it and create another"))

# The headers each operation's reference names: the key, the API version and the workspace on all of them, the
# user profile on the create alone ("Applies to every request in the batch"), and the endpoint's beta names,
# which make a call the beta one (beta.messages.batches; Anthropic's SDK sends no beta of its own there).
function _batch_headers(service::_AnthropicEndpoint, create::Bool)::Vector{Pair{String,String}}
    e = _anthropic_endpoint(service)
    headers = push!(_anthropic_base_headers(e; profile=create), "Content-Type" => "application/json")
    isempty(e.betas) || push!(headers, "anthropic-beta" => join(_anthropic_beta_names(e.betas), ","))
    headers
end

# A custom id "Must be 1 to 64 characters and contain only alphanumeric characters, hyphens, and underscores".
const _ANTHROPIC_CUSTOM_ID = r"^[a-zA-Z0-9_-]{1,64}$"
function _batch_custom_ids(::_AnthropicBatches, ids)
    for id in ids
        occursin(_ANTHROPIC_CUSTOM_ID, id) || throw(ArgumentError("create_batch: an Anthropic custom id is 1 to 64 " *
            "letters, digits, hyphens and underscores (got $(repr(id)))"))
    end
end

# Each request's params are its Chat's Messages body. Fast mode (`speed`) is one of the parameters a batched
# request does not take: it "tunes synchronous latency, which doesn't apply to asynchronous batch processing".
function _batch_input(::_AnthropicBatches, service, requests, metadata, output_expires_after)
    _no_openai_options(service, metadata, output_expires_after)
    reqs = map(requests) do (id, chat)
        chat.service_tier == "fast" && throw(ArgumentError("create_batch: request $(repr(id)) asks for fast mode " *
            "(service_tier=\"fast\"), which a Message Batch does not take: fast mode tunes synchronous latency"))
        Dict("custom_id" => String(id), "params" => JSON.parse(encode_request(service, chat); dicttype=Dict{String,Any}))
    end
    JSON.json(Dict("requests" => reqs))
end

_create_batch(wire::_AnthropicBatches, service, body::String, cfg, t0, tok) = _batch_object(wire,
    _batch_send("POST", _batches_url(wire, service), _batch_headers(service, true), cfg, t0, tok, body))

# A MessageBatch: its `processing_status` is the status, its `request_counts` the tallies; `results_url` and the
# timestamps stay in `raw`.
_parse_batch(::_AnthropicBatches, d::AbstractDict) = BatchObject(id=_batch_required(d, "id", String),
    status=_batch_required(d, "processing_status", String),
    request_counts=Dict{String,Any}(_batch_required(d, "request_counts", AbstractDict)), raw=Dict{String,Any}(d))

# A DeletedMessageBatch: the batch's id, and the type that says it is deleted.
function _batch_deleted(::_AnthropicBatches, ::String, d::Dict{String,Any})
    kind = _batch_required(d, "type", String)
    kind == "message_batch_deleted" || throw(ArgumentError("the delete reply is a $(repr(kind)), not a deleted Message Batch"))
    BatchDeleteSuccess(id=_batch_required(d, "id", String), deleted=true)
end

# A result line: the request's custom id and its result — `succeeded` with the message, `errored` with the
# standard error shape, or `canceled` / `expired` before the request ran (neither is billed).
function _anthropic_batch_item(service, line::Dict{String,Any})::Pair{String,_BatchItem}
    id = _batch_required(line, "custom_id", String)
    result = _batch_required(line, "result", AbstractDict)
    kind = _batch_required(result, "type", String)
    id => if kind == "succeeded"
        _batch_success(service, _batch_required(result, "message", AbstractDict), nothing)
    elseif kind == "errored"
        err = _batch_required(result, "error", AbstractDict)
        _batch_error(service, JSON.json(err), nothing; request_id=_batch_optional(err, "request_id", String))
    elseif kind == "canceled"
        _batch_error(service, "Anthropic batch request $(repr(id)) was canceled: the batch was canceled before " *
                              "the request was sent to the model (not billed)", nothing)
    elseif kind == "expired"
        _batch_error(service, "Anthropic batch request $(repr(id)) expired: the batch reached its 24-hour " *
                              "expiration before the request was sent to the model (not billed)", nothing)
    else
        _batch_error(service, "Anthropic batch request $(repr(id)) has a result of type $(repr(kind)), not one of " *
                              "succeeded, errored, canceled or expired", nothing)
    end
end

# The results stream: a JSONL body, one result per request, "not guaranteed to be in the same order as requests".
function _batch_results(wire::_AnthropicBatches, service, id, cfg, t0, tok)
    resp = _batch_send("GET", _batch_url(wire, service, id, "/results"), _batch_headers(service, false), cfg, t0, tok)
    resp.status == 200 || return _failure(BatchFailure, resp)
    BatchResultsSuccess(Pair{String,_BatchItem}[_anthropic_batch_item(service, line) for line in _jsonl_objects(resp.body)])
end
