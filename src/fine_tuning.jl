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
"Fine-tuning API error result: HTTP `status`, the raw `response` body, and the `request_id` the service sent (`x-request-id`/`request-id` header), if any."
@kwdef struct FineTuningFailure <: LLMRequestResponse; response::String; status::Int; request_id::Union{String,Nothing} = nothing; end
"Fine-tuning API call that produced no usable reply (transport failure, timeout, or a 200 that could not be decoded); `cause` is the underlying exception — a [`UniLMTimeout`](@ref) for a timeout."
@kwdef struct FineTuningCallError <: LLMRequestResponse; error::String; status::Union{Int,Nothing} = nothing; cause::Union{Nothing,Exception} = nothing; end

# A field of an OpenAI platform reply: absent or null where the reference allows it; any other JSON type
# than the documented one is a malformed reply, and the throw becomes the verb's call error.
function _reply_field(d::AbstractDict, key::String, ::Type{T}, api::String)::Union{T,Nothing} where {T}
    v = get(d, key, nothing)
    isnothing(v) || v isa T || throw(ArgumentError("$api reply field $(repr(key)) holds a $(typeof(v)), not the documented $T"))
    v
end

# A field the reference requires: a 200 without it is no reply of that kind, never an empty page or a
# value made up for it.
_reply_required(d::AbstractDict, key::String, ::Type{T}, api::String) where {T} =
    @something(_reply_field(d, key, T, api), throw(ArgumentError("the $api reply has no $(repr(key))")))

# OpenAI's FineTuningJob requires its `status`: a reply without one is no job of known state.
_parse_ft_job(d::AbstractDict) = FineTuningJob(id=_reply_required(d, "id", String, "Fine-tuning API"),
    status=_reply_required(d, "status", String, "Fine-tuning API"), model=get(d, "model", nothing), fine_tuned_model=get(d, "fine_tuned_model", nothing), raw=Dict{String,Any}(d))
_ft_job_resp(resp) = resp.status == 200 ?
    FineTuningSuccess(response=_parse_ft_job(JSON.parse(resp.body; dicttype=Dict{String,Any}))) :
    _failure(FineTuningFailure, resp)
# OpenAI's list replies require `data` and `has_more`.
function _ft_list_resp(resp)
    resp.status == 200 || return _failure(FineTuningFailure, resp)
    data = JSON.parse(resp.body; dicttype=Dict{String,Any})
    FineTuningListSuccess(response=FineTuningList(
        data=Vector{Dict{String,Any}}(_reply_required(data, "data", AbstractVector, "Fine-tuning API")),
        has_more=_reply_required(data, "has_more", Bool, "Fine-tuning API"), raw=data))
end
# The cursor query every fine-tuning list takes: `limit` page size, `after` the last id seen.
function _ft_page(url::String, limit::Union{Int,Nothing}, after::Union{String,Nothing})::String
    params = String[]
    isnothing(limit) || push!(params, "limit=$(_uripart(limit))")
    isnothing(after) || push!(params, "after=$(_uripart(after))")
    isempty(params) ? url : url * "?" * join(params, "&")
end

"""
    WandbIntegration(; project, name=nothing, entity=nothing, tags=nothing)

A Weights & Biases integration for [`create_fine_tuning_job`](@ref)'s `integrations`: OpenAI reports
the job's metrics to the W&B `project`, in a run named `name` (the job id when unset), under the
`entity` (the W&B API key's default when unset), with `tags` beside the ones OpenAI adds.
"""
@kwdef struct WandbIntegration
    project::String
    name::Union{String,Nothing} = nothing
    entity::Union{String,Nothing} = nothing
    tags::Union{Vector{String},Nothing} = nothing
end
JSON.lower(w::WandbIntegration) = Dict{Symbol,Any}(:type => "wandb", :wandb => Dict{Symbol,Any}(
    f => getfield(w, f) for f in fieldnames(WandbIntegration) if !isnothing(getfield(w, f))))

"""
    create_fine_tuning_job(; model, training_file, validation_file=nothing, method=nothing,
                           suffix=nothing, metadata=nothing, seed=nothing, integrations=nothing,
                           service=OPENAIServiceEndpoint)

Create a fine-tuning job. `training_file` is a file id from `upload_file(path, "fine-tune")`.
`method` is e.g. `Dict("type"=>"supervised", "supervised"=>Dict("hyperparameters"=>...))`.
`seed` (0 to 2147483647; an `ArgumentError` otherwise) makes a rerun reproducible, and
`integrations` is a vector of [`WandbIntegration`](@ref)s.

Pass `config::Union{Nothing,RequestConfig}` to override the timeout budget for this call (a single bounded attempt; `max_attempts` does not apply).

Pass `cancel::Union{Nothing,CancelToken}` (default: the ambient [`with_cancel`](@ref) token) to make the call cancellable: once the token is cancelled it returns a `FineTuningCallError` whose `cause` is a [`UniLMCancelled`](@ref) — without sending anything when the token was already cancelled, at once when mid-request (never retried).
"""
function create_fine_tuning_job(; model::String, training_file::String,
    validation_file::Union{String,Nothing}=nothing, method::Union{AbstractDict,Nothing}=nothing,
    suffix::Union{String,Nothing}=nothing, metadata::Union{AbstractDict,Nothing}=nothing,
    seed::Union{Int,Nothing}=nothing, integrations::Union{AbstractVector{WandbIntegration},Nothing}=nothing,
    service::ServiceEndpointSpec=OPENAIServiceEndpoint, config::Union{Nothing,RequestConfig}=nothing,
    cancel::Union{Nothing,CancelToken}=nothing)
    validate_capability(service, :fine_tuning, "Fine-tuning API")
    # The reference's range: a 32-bit signed seed that is not negative.
    isnothing(seed) || 0 <= seed <= typemax(Int32) || throw(ArgumentError(
        "create_fine_tuning_job takes a seed from 0 to 2147483647 (got $seed)"))
    cfg = _resolve_config(config); t0 = time_ns()
    try
        d = Dict{Symbol,Any}(:model => model, :training_file => training_file)
        !isnothing(validation_file) && (d[:validation_file] = validation_file)
        !isnothing(method) && (d[:method] = method)
        !isnothing(suffix) && (d[:suffix] = suffix)
        !isnothing(metadata) && (d[:metadata] = metadata)
        !isnothing(seed) && (d[:seed] = seed)
        !isnothing(integrations) && (d[:integrations] = integrations)
        _ft_job_resp(_http("POST", _api_base_url(service) * FINE_TUNING_PATH, auth_header(service),
            JSON.json(d); cfg, remaining=_remaining_s(cfg, t0), cancel=_resolve_cancel(cancel)))
    catch e
        e isa InterruptException && rethrow()
        _callerr(FineTuningCallError, e)
    end
end

"""
    retrieve_fine_tuning_job(id; service=OPENAIServiceEndpoint)

Pass `config::Union{Nothing,RequestConfig}` to override the timeout budget for this call (a single bounded attempt; `max_attempts` does not apply).

Pass `cancel::Union{Nothing,CancelToken}` (default: the ambient [`with_cancel`](@ref) token) to make the call cancellable: once the token is cancelled it returns a `FineTuningCallError` whose `cause` is a [`UniLMCancelled`](@ref) — without sending anything when the token was already cancelled, at once when mid-request (never retried).
"""
function retrieve_fine_tuning_job(id::String; service::ServiceEndpointSpec=OPENAIServiceEndpoint, config::Union{Nothing,RequestConfig}=nothing,
    cancel::Union{Nothing,CancelToken}=nothing)
    validate_capability(service, :fine_tuning, "Fine-tuning API")
    cfg = _resolve_config(config); t0 = time_ns()
    try
        _ft_job_resp(_http("GET", _api_base_url(service) * FINE_TUNING_PATH * "/" * _uripart(id), auth_header(service);
            cfg, remaining=_remaining_s(cfg, t0), cancel=_resolve_cancel(cancel)))
    catch e
        e isa InterruptException && rethrow()
        _callerr(FineTuningCallError, e)
    end
end

"""
    cancel_fine_tuning_job(id; service=OPENAIServiceEndpoint)

Pass `config::Union{Nothing,RequestConfig}` to override the timeout budget for this call (a single bounded attempt; `max_attempts` does not apply).

Pass `cancel::Union{Nothing,CancelToken}` (default: the ambient [`with_cancel`](@ref) token) to make the call cancellable: once the token is cancelled it returns a `FineTuningCallError` whose `cause` is a [`UniLMCancelled`](@ref) — without sending anything when the token was already cancelled, at once when mid-request (never retried).
"""
function cancel_fine_tuning_job(id::String; service::ServiceEndpointSpec=OPENAIServiceEndpoint, config::Union{Nothing,RequestConfig}=nothing,
    cancel::Union{Nothing,CancelToken}=nothing)
    validate_capability(service, :fine_tuning, "Fine-tuning API")
    cfg = _resolve_config(config); t0 = time_ns()
    try
        _ft_job_resp(_http("POST", _api_base_url(service) * FINE_TUNING_PATH * "/" * _uripart(id) * "/cancel", auth_header(service);
            cfg, remaining=_remaining_s(cfg, t0), cancel=_resolve_cancel(cancel)))
    catch e
        e isa InterruptException && rethrow()
        _callerr(FineTuningCallError, e)
    end
end

# A POST on a job's action route (`pause`, `resume`), answered with the job.
function _ft_job_action(id::String, action::String, service, config, cancel)
    validate_capability(service, :fine_tuning, "Fine-tuning API")
    cfg = _resolve_config(config); t0 = time_ns()
    try
        _ft_job_resp(_http("POST", _api_base_url(service) * FINE_TUNING_PATH * "/" * _uripart(id) * "/" * action,
            auth_header(service); cfg, remaining=_remaining_s(cfg, t0), cancel=_resolve_cancel(cancel)))
    catch e
        e isa InterruptException && rethrow()
        _callerr(FineTuningCallError, e)
    end
end

"""
    pause_fine_tuning_job(id; service=OPENAIServiceEndpoint)

Pause a running fine-tuning job; the result holds the job, whose `status` passes through `pausing`
to `paused`. A job that cannot pause in its state, or does not support pausing, is OpenAI's 400, a
`FineTuningFailure`.

Takes `config` and `cancel` as [`cancel_fine_tuning_job`](@ref) does: one bounded attempt, and a
cancelled token returns a `FineTuningCallError` whose `cause` is a [`UniLMCancelled`](@ref).
"""
pause_fine_tuning_job(id::String; service::ServiceEndpointSpec=OPENAIServiceEndpoint,
    config::Union{Nothing,RequestConfig}=nothing, cancel::Union{Nothing,CancelToken}=nothing) =
    _ft_job_action(id, "pause", service, config, cancel)

"""
    resume_fine_tuning_job(id; service=OPENAIServiceEndpoint)

Resume a paused fine-tuning job; the result holds the job. Otherwise as [`pause_fine_tuning_job`](@ref).
"""
resume_fine_tuning_job(id::String; service::ServiceEndpointSpec=OPENAIServiceEndpoint,
    config::Union{Nothing,RequestConfig}=nothing, cancel::Union{Nothing,CancelToken}=nothing) =
    _ft_job_action(id, "resume", service, config, cancel)

"""
    list_fine_tuning_jobs(; limit=nothing, after=nothing, service=OPENAIServiceEndpoint)

Pass `config::Union{Nothing,RequestConfig}` to override the timeout budget for this call (a single bounded attempt; `max_attempts` does not apply).

Pass `cancel::Union{Nothing,CancelToken}` (default: the ambient [`with_cancel`](@ref) token) to make the call cancellable: once the token is cancelled it returns a `FineTuningCallError` whose `cause` is a [`UniLMCancelled`](@ref) — without sending anything when the token was already cancelled, at once when mid-request (never retried).
"""
function list_fine_tuning_jobs(; limit::Union{Int,Nothing}=nothing, after::Union{String,Nothing}=nothing, service::ServiceEndpointSpec=OPENAIServiceEndpoint, config::Union{Nothing,RequestConfig}=nothing,
    cancel::Union{Nothing,CancelToken}=nothing)
    validate_capability(service, :fine_tuning, "Fine-tuning API")
    cfg = _resolve_config(config); t0 = time_ns()
    try
        _ft_list_resp(_http("GET", _ft_page(_api_base_url(service) * FINE_TUNING_PATH, limit, after),
            auth_header(service); cfg, remaining=_remaining_s(cfg, t0), cancel=_resolve_cancel(cancel)))
    catch e
        e isa InterruptException && rethrow()
        _callerr(FineTuningCallError, e)
    end
end

"""
    list_fine_tuning_events(id; limit=nothing, after=nothing, service=OPENAIServiceEndpoint)

List a job's events, a page at a time: `limit` sets the page size and `after` the last
event id of the previous page (see `has_more` on the result).

Pass `config::Union{Nothing,RequestConfig}` to override the timeout budget for this call (a single bounded attempt; `max_attempts` does not apply).

Pass `cancel::Union{Nothing,CancelToken}` (default: the ambient [`with_cancel`](@ref) token) to make the call cancellable: once the token is cancelled it returns a `FineTuningCallError` whose `cause` is a [`UniLMCancelled`](@ref) — without sending anything when the token was already cancelled, at once when mid-request (never retried).
"""
function list_fine_tuning_events(id::String; limit::Union{Int,Nothing}=nothing, after::Union{String,Nothing}=nothing,
    service::ServiceEndpointSpec=OPENAIServiceEndpoint, config::Union{Nothing,RequestConfig}=nothing,
    cancel::Union{Nothing,CancelToken}=nothing)
    validate_capability(service, :fine_tuning, "Fine-tuning API")
    cfg = _resolve_config(config); t0 = time_ns()
    try
        _ft_list_resp(_http("GET", _ft_page(_api_base_url(service) * FINE_TUNING_PATH * "/" * _uripart(id) * "/events", limit, after),
            auth_header(service); cfg, remaining=_remaining_s(cfg, t0), cancel=_resolve_cancel(cancel)))
    catch e
        e isa InterruptException && rethrow()
        _callerr(FineTuningCallError, e)
    end
end

"""
    list_fine_tuning_checkpoints(id; limit=nothing, after=nothing, service=OPENAIServiceEndpoint)

List a job's checkpoints, a page at a time: `limit` sets the page size and `after` the
last checkpoint id of the previous page (see `has_more` on the result).

Pass `config::Union{Nothing,RequestConfig}` to override the timeout budget for this call (a single bounded attempt; `max_attempts` does not apply).

Pass `cancel::Union{Nothing,CancelToken}` (default: the ambient [`with_cancel`](@ref) token) to make the call cancellable: once the token is cancelled it returns a `FineTuningCallError` whose `cause` is a [`UniLMCancelled`](@ref) — without sending anything when the token was already cancelled, at once when mid-request (never retried).
"""
function list_fine_tuning_checkpoints(id::String; limit::Union{Int,Nothing}=nothing, after::Union{String,Nothing}=nothing,
    service::ServiceEndpointSpec=OPENAIServiceEndpoint, config::Union{Nothing,RequestConfig}=nothing,
    cancel::Union{Nothing,CancelToken}=nothing)
    validate_capability(service, :fine_tuning, "Fine-tuning API")
    cfg = _resolve_config(config); t0 = time_ns()
    try
        _ft_list_resp(_http("GET", _ft_page(_api_base_url(service) * FINE_TUNING_PATH * "/" * _uripart(id) * "/checkpoints", limit, after),
            auth_header(service); cfg, remaining=_remaining_s(cfg, t0), cancel=_resolve_cancel(cancel)))
    catch e
        e isa InterruptException && rethrow()
        _callerr(FineTuningCallError, e)
    end
end
