# ============================================================================
# OpenAI Fine-tuning API — /v1/fine_tuning/jobs. Training/validation data is
# uploaded via the Files API (purpose="fine-tune"). `method` selects supervised /
# dpo / reinforcement with hyperparameters.
# ============================================================================

"""
    FineTuningJob

A fine-tuning job: `id`, `status`, `model`, and `fine_tuned_model` (the resulting
model name once training completes); `raw` holds the unparsed JSON response.
"""
@kwdef struct FineTuningJob
    id::String
    status::Union{String,Nothing} = nothing
    model::Union{String,Nothing} = nothing
    fine_tuned_model::Union{String,Nothing} = nothing
    raw::Dict{String,Any} = Dict{String,Any}()
end

"""
    FineTuningList

A page of records from [`list_fine_tuning_jobs`](@ref), [`list_fine_tuning_events`](@ref),
or [`list_fine_tuning_checkpoints`](@ref); `data` holds the raw JSON entries and
`has_more` signals that further pages are available.
"""
@kwdef struct FineTuningList
    data::Vector{Dict{String,Any}}
    has_more::Bool = false
    raw::Dict{String,Any} = Dict{String,Any}()
end

"Successful create/retrieve/cancel result wrapping a [`FineTuningJob`](@ref)."
@kwdef struct FineTuningSuccess <: LLMRequestResponse; response::FineTuningJob; end
"Successful list result wrapping a [`FineTuningList`](@ref)."
@kwdef struct FineTuningListSuccess <: LLMRequestResponse; response::FineTuningList; end
"Fine-tuning API error result: HTTP `status` and the raw `response` body."
@kwdef struct FineTuningFailure <: LLMRequestResponse; response::String; status::Int; end
"Local/transport error from a Fine-tuning API call (the request never completed)."
@kwdef struct FineTuningCallError <: LLMRequestResponse; error::String; status::Union{Int,Nothing} = nothing; end

_parse_ft_job(d::AbstractDict) = FineTuningJob(id=d["id"], status=get(d, "status", nothing),
    model=get(d, "model", nothing), fine_tuned_model=get(d, "fine_tuned_model", nothing), raw=Dict{String,Any}(d))
_ft_err(e) = FineTuningCallError(error=string(e), status=(hasproperty(e, :status) ? e.status : nothing))
_ft_job_resp(resp) = resp.status == 200 ?
    FineTuningSuccess(response=_parse_ft_job(JSON.parse(resp.body; dicttype=Dict{String,Any}))) :
    FineTuningFailure(response=String(resp.body), status=resp.status)
function _ft_list_resp(resp)
    resp.status == 200 || return FineTuningFailure(response=String(resp.body), status=resp.status)
    data = JSON.parse(resp.body; dicttype=Dict{String,Any})
    FineTuningListSuccess(response=FineTuningList(data=Vector{Dict{String,Any}}(get(data, "data", [])),
        has_more=get(data, "has_more", false), raw=data))
end

"""
    create_fine_tuning_job(; model, training_file, validation_file=nothing, method=nothing,
                           suffix=nothing, metadata=nothing, service=OPENAIServiceEndpoint)

Create a fine-tuning job. `training_file` is a file id from `upload_file(path, "fine-tune")`.
`method` is e.g. `Dict("type"=>"supervised", "supervised"=>Dict("hyperparameters"=>...))`.

Pass `config::Union{Nothing,RequestConfig}` to override the timeout/retry budget for this call.
"""
function create_fine_tuning_job(; model::String, training_file::String,
    validation_file::Union{String,Nothing}=nothing, method::Union{AbstractDict,Nothing}=nothing,
    suffix::Union{String,Nothing}=nothing, metadata::Union{AbstractDict,Nothing}=nothing,
    service::ServiceEndpointSpec=OPENAIServiceEndpoint, config::Union{Nothing,RequestConfig}=nothing)
    validate_capability(service, :fine_tuning, "Fine-tuning API")
    cfg = _resolve_config(config); t0 = time_ns()
    try
        d = Dict{Symbol,Any}(:model => model, :training_file => training_file)
        !isnothing(validation_file) && (d[:validation_file] = validation_file)
        !isnothing(method) && (d[:method] = method)
        !isnothing(suffix) && (d[:suffix] = suffix)
        !isnothing(metadata) && (d[:metadata] = metadata)
        _ft_job_resp(_http("POST", _api_base_url(service) * FINE_TUNING_PATH, auth_header(service),
            JSON.json(d); cfg, remaining=_remaining_s(cfg, t0)))
    catch e
        e isa InterruptException && rethrow()
        _ft_err(e)
    end
end

"""
    retrieve_fine_tuning_job(id; service=OPENAIServiceEndpoint)

Pass `config::Union{Nothing,RequestConfig}` to override the timeout/retry budget for this call.
"""
function retrieve_fine_tuning_job(id::String; service::ServiceEndpointSpec=OPENAIServiceEndpoint, config::Union{Nothing,RequestConfig}=nothing)
    validate_capability(service, :fine_tuning, "Fine-tuning API")
    cfg = _resolve_config(config); t0 = time_ns()
    try
        _ft_job_resp(_http("GET", _api_base_url(service) * FINE_TUNING_PATH * "/" * id, auth_header(service);
            cfg, remaining=_remaining_s(cfg, t0)))
    catch e
        e isa InterruptException && rethrow()
        _ft_err(e)
    end
end

"""
    cancel_fine_tuning_job(id; service=OPENAIServiceEndpoint)

Pass `config::Union{Nothing,RequestConfig}` to override the timeout/retry budget for this call.
"""
function cancel_fine_tuning_job(id::String; service::ServiceEndpointSpec=OPENAIServiceEndpoint, config::Union{Nothing,RequestConfig}=nothing)
    validate_capability(service, :fine_tuning, "Fine-tuning API")
    cfg = _resolve_config(config); t0 = time_ns()
    try
        _ft_job_resp(_http("POST", _api_base_url(service) * FINE_TUNING_PATH * "/" * id * "/cancel", auth_header(service);
            cfg, remaining=_remaining_s(cfg, t0)))
    catch e
        e isa InterruptException && rethrow()
        _ft_err(e)
    end
end

"""
    list_fine_tuning_jobs(; limit=nothing, after=nothing, service=OPENAIServiceEndpoint)

Pass `config::Union{Nothing,RequestConfig}` to override the timeout/retry budget for this call.
"""
function list_fine_tuning_jobs(; limit::Union{Int,Nothing}=nothing, after::Union{String,Nothing}=nothing, service::ServiceEndpointSpec=OPENAIServiceEndpoint, config::Union{Nothing,RequestConfig}=nothing)
    validate_capability(service, :fine_tuning, "Fine-tuning API")
    cfg = _resolve_config(config); t0 = time_ns()
    try
        url = _api_base_url(service) * FINE_TUNING_PATH
        params = String[]
        !isnothing(limit) && push!(params, "limit=$limit")
        !isnothing(after) && push!(params, "after=$after")
        !isempty(params) && (url *= "?" * join(params, "&"))
        _ft_list_resp(_http("GET", url, auth_header(service); cfg, remaining=_remaining_s(cfg, t0)))
    catch e
        e isa InterruptException && rethrow()
        _ft_err(e)
    end
end

"""
    list_fine_tuning_events(id; service=OPENAIServiceEndpoint)

Pass `config::Union{Nothing,RequestConfig}` to override the timeout/retry budget for this call.
"""
function list_fine_tuning_events(id::String; service::ServiceEndpointSpec=OPENAIServiceEndpoint, config::Union{Nothing,RequestConfig}=nothing)
    validate_capability(service, :fine_tuning, "Fine-tuning API")
    cfg = _resolve_config(config); t0 = time_ns()
    try
        _ft_list_resp(_http("GET", _api_base_url(service) * FINE_TUNING_PATH * "/" * id * "/events", auth_header(service);
            cfg, remaining=_remaining_s(cfg, t0)))
    catch e
        e isa InterruptException && rethrow()
        _ft_err(e)
    end
end

"""
    list_fine_tuning_checkpoints(id; service=OPENAIServiceEndpoint)

Pass `config::Union{Nothing,RequestConfig}` to override the timeout/retry budget for this call.
"""
function list_fine_tuning_checkpoints(id::String; service::ServiceEndpointSpec=OPENAIServiceEndpoint, config::Union{Nothing,RequestConfig}=nothing)
    validate_capability(service, :fine_tuning, "Fine-tuning API")
    cfg = _resolve_config(config); t0 = time_ns()
    try
        _ft_list_resp(_http("GET", _api_base_url(service) * FINE_TUNING_PATH * "/" * id * "/checkpoints", auth_header(service);
            cfg, remaining=_remaining_s(cfg, t0)))
    catch e
        e isa InterruptException && rethrow()
        _ft_err(e)
    end
end
