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
"Batch API error result: HTTP `status` and the raw `response` body."
@kwdef struct BatchFailure <: LLMRequestResponse; response::String; status::Int; end
"Local/transport error from a Batch API call (the request never completed)."
@kwdef struct BatchCallError <: LLMRequestResponse; error::String; status::Union{Int,Nothing} = nothing; end

_parse_batch(d::AbstractDict) = BatchObject(id=d["id"], status=get(d, "status", nothing),
    endpoint=get(d, "endpoint", nothing), input_file_id=get(d, "input_file_id", nothing),
    output_file_id=get(d, "output_file_id", nothing), error_file_id=get(d, "error_file_id", nothing),
    request_counts=Dict{String,Any}(get(d, "request_counts", Dict{String,Any}())), raw=Dict{String,Any}(d))

_batch_err(e) = BatchCallError(error=string(e), status=(hasproperty(e, :status) ? e.status : nothing))

"""
    create_batch(input_file_id, endpoint; completion_window="24h", metadata=nothing, service=OPENAIServiceEndpoint)

Create a batch job. `endpoint` is e.g. `"/v1/chat/completions"`, `"/v1/responses"`, or
`"/v1/embeddings"`. `input_file_id` comes from `upload_file(path, "batch")`.

Pass `config::Union{Nothing,RequestConfig}` to override the timeout/retry budget for this call.
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
            BatchFailure(response=String(resp.body), status=resp.status)
    catch e
        e isa InterruptException && rethrow()
        _batch_err(e)
    end
end

"""
    retrieve_batch(id; service=OPENAIServiceEndpoint)

Pass `config::Union{Nothing,RequestConfig}` to override the timeout/retry budget for this call.
"""
function retrieve_batch(id::String; service::ServiceEndpointSpec=OPENAIServiceEndpoint, config::Union{Nothing,RequestConfig}=nothing)
    validate_capability(service, :batch, "Batch API")
    cfg = _resolve_config(config); t0 = time_ns()
    try
        resp = _http("GET", _api_base_url(service) * BATCHES_PATH * "/" * id, auth_header(service);
            cfg, remaining=_remaining_s(cfg, t0))
        resp.status == 200 ? BatchSuccess(response=_parse_batch(JSON.parse(resp.body; dicttype=Dict{String,Any}))) :
            BatchFailure(response=String(resp.body), status=resp.status)
    catch e
        e isa InterruptException && rethrow()
        _batch_err(e)
    end
end

"""
    cancel_batch(id; service=OPENAIServiceEndpoint)

Pass `config::Union{Nothing,RequestConfig}` to override the timeout/retry budget for this call.
"""
function cancel_batch(id::String; service::ServiceEndpointSpec=OPENAIServiceEndpoint, config::Union{Nothing,RequestConfig}=nothing)
    validate_capability(service, :batch, "Batch API")
    cfg = _resolve_config(config); t0 = time_ns()
    try
        resp = _http("POST", _api_base_url(service) * BATCHES_PATH * "/" * id * "/cancel", auth_header(service);
            cfg, remaining=_remaining_s(cfg, t0))
        resp.status == 200 ? BatchSuccess(response=_parse_batch(JSON.parse(resp.body; dicttype=Dict{String,Any}))) :
            BatchFailure(response=String(resp.body), status=resp.status)
    catch e
        e isa InterruptException && rethrow()
        _batch_err(e)
    end
end

"""
    list_batches(; limit=nothing, after=nothing, service=OPENAIServiceEndpoint)

Pass `config::Union{Nothing,RequestConfig}` to override the timeout/retry budget for this call.
"""
function list_batches(; limit::Union{Int,Nothing}=nothing, after::Union{String,Nothing}=nothing, service::ServiceEndpointSpec=OPENAIServiceEndpoint, config::Union{Nothing,RequestConfig}=nothing)
    validate_capability(service, :batch, "Batch API")
    cfg = _resolve_config(config); t0 = time_ns()
    try
        url = _api_base_url(service) * BATCHES_PATH
        params = String[]
        !isnothing(limit) && push!(params, "limit=$limit")
        !isnothing(after) && push!(params, "after=$after")
        !isempty(params) && (url *= "?" * join(params, "&"))
        resp = _http("GET", url, auth_header(service); cfg, remaining=_remaining_s(cfg, t0))
        if resp.status == 200
            data = JSON.parse(resp.body; dicttype=Dict{String,Any})
            BatchListSuccess(response=BatchList(data=BatchObject[_parse_batch(b) for b in get(data, "data", [])], has_more=get(data, "has_more", false), raw=data))
        else
            BatchFailure(response=String(resp.body), status=resp.status)
        end
    catch e
        e isa InterruptException && rethrow()
        _batch_err(e)
    end
end

"""
    poll_batch(id; interval=10.0, timeout=86400.0, service=OPENAIServiceEndpoint)

Poll a batch until terminal (`completed`/`failed`/`cancelled`/`expired`) or timeout.

Pass `config::Union{Nothing,RequestConfig}` to override the timeout/retry budget for this call.
"""
function poll_batch(id::String; interval::Real=10.0, timeout::Real=86400.0, service::ServiceEndpointSpec=OPENAIServiceEndpoint, config::Union{Nothing,RequestConfig}=nothing)
    max_iters = max(1, ceil(Int, timeout / interval))
    for _ in 1:max_iters
        r = retrieve_batch(id; service=service, config=config)
        r isa BatchSuccess || return r
        r.response.status in ("completed", "failed", "cancelled", "expired") && return r
        sleep(interval)
    end
    BatchCallError(error="poll_batch timed out after $(timeout)s", status=nothing)
end
