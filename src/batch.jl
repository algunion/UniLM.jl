# ============================================================================
# OpenAI Batch API — async, ~50%-cheaper bulk processing. The input is a JSONL
# file uploaded via the Files API (purpose="batch"); output is fetched with
# `file_content(batch.output_file_id)`.
# ============================================================================

"""
    BatchObject

A batch job from the Batch API: `id`, `status`, `endpoint`, `input_file_id`,
`output_file_id`, `error_file_id`, and `request_counts`; `raw` holds the unparsed
JSON response.
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

A page of [`BatchObject`](@ref)s from [`list_batches`](@ref); `has_more` signals
that further pages are available.
"""
@kwdef struct BatchList
    data::Vector{BatchObject}
    has_more::Bool = false
    raw::Dict{String,Any} = Dict{String,Any}()
end

"Successful create/retrieve/cancel result wrapping a [`BatchObject`](@ref)."
@kwdef struct BatchSuccess <: LLMRequestResponse; response::BatchObject; end
"Successful [`list_batches`](@ref) result wrapping a [`BatchList`](@ref)."
@kwdef struct BatchListSuccess <: LLMRequestResponse; response::BatchList; end
"Batch API error result: HTTP `status`, the raw `response` body, and the `request_id` the service sent (`x-request-id`/`request-id` header), if any."
@kwdef struct BatchFailure <: LLMRequestResponse; response::String; status::Int; request_id::Union{String,Nothing} = nothing; end
"Batch API call that produced no usable reply (transport failure, timeout, or a 200 that could not be decoded); `cause` is the underlying exception — a [`UniLMTimeout`](@ref) for a timeout. `last_observed` is set only when [`poll_batch`](@ref) runs out of time or is cancelled: the last [`BatchObject`](@ref) it saw, if any."
@kwdef struct BatchCallError <: LLMRequestResponse; error::String; status::Union{Int,Nothing} = nothing; cause::Union{Nothing,Exception} = nothing; last_observed::Union{Nothing,BatchObject} = nothing; end

_transient(r::BatchFailure) = _is_retryable(r.status)
_transient(r::BatchCallError) = _per_attempt_timeout(r.cause)

_parse_batch(d::AbstractDict) = BatchObject(id=d["id"], status=get(d, "status", nothing),
    endpoint=get(d, "endpoint", nothing), input_file_id=get(d, "input_file_id", nothing),
    output_file_id=get(d, "output_file_id", nothing), error_file_id=get(d, "error_file_id", nothing),
    request_counts=Dict{String,Any}(get(d, "request_counts", Dict{String,Any}())), raw=Dict{String,Any}(d))

"""
    create_batch(input_file_id, endpoint; completion_window="24h", metadata=nothing, service=OPENAIServiceEndpoint)

Create a batch job. `endpoint` is e.g. `"/v1/chat/completions"`, `"/v1/responses"`, or
`"/v1/embeddings"`. `input_file_id` comes from `upload_file(path, "batch")`.

Pass `config::Union{Nothing,RequestConfig}` to override the timeout budget for this call (a single bounded attempt; `max_attempts` does not apply).
"""
function create_batch(input_file_id::String, endpoint::String; completion_window::String="24h",
    metadata::Union{AbstractDict,Nothing}=nothing, service::ServiceEndpointSpec=OPENAIServiceEndpoint,
    config::Union{Nothing,RequestConfig}=nothing)
    validate_capability(service, :batch, "Batch API")
    cfg = _resolve_config(config); t0 = time_ns()
    try
        d = Dict{Symbol,Any}(:input_file_id => input_file_id, :endpoint => endpoint, :completion_window => completion_window)
        !isnothing(metadata) && (d[:metadata] = metadata)
        resp = _http("POST", _api_base_url(service) * BATCHES_PATH, auth_header(service),
            JSON.json(d); cfg, remaining=_remaining_s(cfg, t0))
        resp.status == 200 ? BatchSuccess(response=_parse_batch(JSON.parse(resp.body; dicttype=Dict{String,Any}))) :
            _failure(BatchFailure, resp)
    catch e
        e isa InterruptException && rethrow()
        _callerr(BatchCallError, e)
    end
end

"""
    retrieve_batch(id; service=OPENAIServiceEndpoint)

Pass `config::Union{Nothing,RequestConfig}` to override the timeout budget for this call (a single bounded attempt; `max_attempts` does not apply).
"""
function retrieve_batch(id::String; service::ServiceEndpointSpec=OPENAIServiceEndpoint, config::Union{Nothing,RequestConfig}=nothing)
    validate_capability(service, :batch, "Batch API")
    cfg = _resolve_config(config); t0 = time_ns()
    try
        resp = _http("GET", _api_base_url(service) * BATCHES_PATH * "/" * _uripart(id), auth_header(service);
            cfg, remaining=_remaining_s(cfg, t0))
        resp.status == 200 ? BatchSuccess(response=_parse_batch(JSON.parse(resp.body; dicttype=Dict{String,Any}))) :
            _failure(BatchFailure, resp)
    catch e
        e isa InterruptException && rethrow()
        _callerr(BatchCallError, e)
    end
end

"""
    cancel_batch(id; service=OPENAIServiceEndpoint)

Pass `config::Union{Nothing,RequestConfig}` to override the timeout budget for this call (a single bounded attempt; `max_attempts` does not apply).
"""
function cancel_batch(id::String; service::ServiceEndpointSpec=OPENAIServiceEndpoint, config::Union{Nothing,RequestConfig}=nothing)
    validate_capability(service, :batch, "Batch API")
    cfg = _resolve_config(config); t0 = time_ns()
    try
        resp = _http("POST", _api_base_url(service) * BATCHES_PATH * "/" * _uripart(id) * "/cancel", auth_header(service);
            cfg, remaining=_remaining_s(cfg, t0))
        resp.status == 200 ? BatchSuccess(response=_parse_batch(JSON.parse(resp.body; dicttype=Dict{String,Any}))) :
            _failure(BatchFailure, resp)
    catch e
        e isa InterruptException && rethrow()
        _callerr(BatchCallError, e)
    end
end

"""
    list_batches(; limit=nothing, after=nothing, service=OPENAIServiceEndpoint)

Pass `config::Union{Nothing,RequestConfig}` to override the timeout budget for this call (a single bounded attempt; `max_attempts` does not apply).
"""
function list_batches(; limit::Union{Int,Nothing}=nothing, after::Union{String,Nothing}=nothing, service::ServiceEndpointSpec=OPENAIServiceEndpoint, config::Union{Nothing,RequestConfig}=nothing)
    validate_capability(service, :batch, "Batch API")
    cfg = _resolve_config(config); t0 = time_ns()
    try
        url = _api_base_url(service) * BATCHES_PATH
        params = String[]
        !isnothing(limit) && push!(params, "limit=$(_uripart(limit))")
        !isnothing(after) && push!(params, "after=$(_uripart(after))")
        !isempty(params) && (url *= "?" * join(params, "&"))
        resp = _http("GET", url, auth_header(service); cfg, remaining=_remaining_s(cfg, t0))
        if resp.status == 200
            data = JSON.parse(resp.body; dicttype=Dict{String,Any})
            BatchListSuccess(response=BatchList(data=BatchObject[_parse_batch(b) for b in get(data, "data", [])], has_more=get(data, "has_more", false), raw=data))
        else
            _failure(BatchFailure, resp)
        end
    catch e
        e isa InterruptException && rethrow()
        _callerr(BatchCallError, e)
    end
end

"""
    poll_batch(id; interval=10.0, timeout=86400.0, service=OPENAIServiceEndpoint, cancel=nothing)

Poll a batch until it reaches a terminal status (`completed`/`failed`/`cancelled`/`expired`).
`timeout` bounds the wall-clock time of the whole poll (`Inf` waits indefinitely);
`interval` is the pause between GETs. Both must be positive, else `ArgumentError`.

A transient failure — a status in the retryable set (408/429/500/502/503/504/529) or a
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
    _poll(cfg -> retrieve_batch(id; service, config=cfg), BatchSuccess,
          r -> r.response.status in ("completed", "failed", "cancelled", "expired"),
          (seen, why) -> BatchCallError(error=_poll_end_text("poll_batch", id, why, seen), cause=why,
                                        last_observed=isnothing(seen) ? nothing : seen.response);
          interval, timeout, config, cancel)
end
