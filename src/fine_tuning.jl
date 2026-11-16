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
[`list_fine_tuning_checkpoints`](@ref), [`list_fine_tuning_checkpoint_permissions`](@ref) or
[`create_fine_tuning_checkpoint_permission`](@ref); `data` holds the raw JSON entries and
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
"Successful [`delete_fine_tuning_checkpoint_permission`](@ref) result: the service confirmed (`deleted` is always `true`) the removal of the permission `id`."
@kwdef struct FineTuningDeleteSuccess <: LLMRequestResponse; id::String; deleted::Bool; end
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
# The cursor query every fine-tuning list (and `list_webhook_endpoints`) takes: `limit` page size,
# `after` the last id seen; then the list's own query parts, each already encoded.
function _ft_page(url::String, limit::Union{Int,Nothing}, after::Union{String,Nothing},
                  extra::Vector{String}=String[])::String
    params = String[]
    isnothing(limit) || push!(params, "limit=$(_uripart(limit))")
    isnothing(after) || push!(params, "after=$(_uripart(after))")
    append!(params, extra)
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
    list_fine_tuning_jobs(; limit=nothing, after=nothing, metadata=nothing, service=OPENAIServiceEndpoint)

`metadata`, a `Dict` of strings, keeps the jobs whose metadata holds each of its pairs (sent as
`metadata[key]=value`; an empty value matches a key set to `""`).

Pass `config::Union{Nothing,RequestConfig}` to override the timeout budget for this call (a single bounded attempt; `max_attempts` does not apply).

Pass `cancel::Union{Nothing,CancelToken}` (default: the ambient [`with_cancel`](@ref) token) to make the call cancellable: once the token is cancelled it returns a `FineTuningCallError` whose `cause` is a [`UniLMCancelled`](@ref) — without sending anything when the token was already cancelled, at once when mid-request (never retried).
"""
function list_fine_tuning_jobs(; limit::Union{Int,Nothing}=nothing, after::Union{String,Nothing}=nothing,
    metadata::Union{AbstractDict{<:AbstractString,<:AbstractString},Nothing}=nothing,
    service::ServiceEndpointSpec=OPENAIServiceEndpoint, config::Union{Nothing,RequestConfig}=nothing,
    cancel::Union{Nothing,CancelToken}=nothing)
    validate_capability(service, :fine_tuning, "Fine-tuning API")
    cfg = _resolve_config(config); t0 = time_ns()
    # The reference's deep-object filter, `metadata[k]=v`, its brackets encoded as every other query here.
    query = isnothing(metadata) ? String[] :
        ["metadata%5B$(_uripart(k))%5D=$(_uripart(metadata[k]))" for k in sort!(collect(keys(metadata)))]
    try
        _ft_list_resp(_http("GET", _ft_page(_api_base_url(service) * FINE_TUNING_PATH, limit, after, query),
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

# ─── Checkpoint permissions ───────────────────────────────────────────────────
# Sharing a fine-tuned model checkpoint with other projects of the organization. OpenAI's reference
# says each of these endpoints "requires an admin API key"; with a project key OpenAI refuses the
# call, which returns that refusal as a `FineTuningFailure`.

const _FT_CHECKPOINTS_PATH = "/v1/fine_tuning/checkpoints"
_ft_permissions_url(service, checkpoint::String) =
    _api_base_url(service) * _FT_CHECKPOINTS_PATH * "/" * _uripart(checkpoint) * "/permissions"

"""
    create_fine_tuning_checkpoint_permission(checkpoint, project_ids; service=OPENAIServiceEndpoint)

Share the fine-tuned model checkpoint `checkpoint` (e.g. `"ft:gpt-4.1-mini:org:custom:ckpt-step-10"`)
with the projects `project_ids`; the result is a `FineTuningListSuccess` of the permissions made.

The reference says this endpoint requires an admin API key: set `OPENAI_API_KEY` to one for the call.
With a project key OpenAI refuses it, a `FineTuningFailure`. Takes `config` and `cancel` as
[`cancel_fine_tuning_job`](@ref) does.
"""
function create_fine_tuning_checkpoint_permission(checkpoint::String, project_ids::Vector{String};
    service::ServiceEndpointSpec=OPENAIServiceEndpoint, config::Union{Nothing,RequestConfig}=nothing,
    cancel::Union{Nothing,CancelToken}=nothing)
    validate_capability(service, :fine_tuning, "Fine-tuning API")
    cfg = _resolve_config(config); t0 = time_ns()
    try
        _ft_list_resp(_http("POST", _ft_permissions_url(service, checkpoint), auth_header(service),
            JSON.json(Dict(:project_ids => project_ids)); cfg, remaining=_remaining_s(cfg, t0), cancel=_resolve_cancel(cancel)))
    catch e
        e isa InterruptException && rethrow()
        _callerr(FineTuningCallError, e)
    end
end

"""
    list_fine_tuning_checkpoint_permissions(checkpoint; project_id=nothing, limit=nothing, after=nothing,
                                            order=nothing, service=OPENAIServiceEndpoint)

List the permissions of the fine-tuned model checkpoint `checkpoint`, a page at a time (`limit`,
`after`), only those of `project_id` when given, in `order` `"ascending"` or `"descending"` (OpenAI's
default; another order is an `ArgumentError`). Needs an admin API key, as
[`create_fine_tuning_checkpoint_permission`](@ref) does.
"""
function list_fine_tuning_checkpoint_permissions(checkpoint::String; project_id::Union{String,Nothing}=nothing,
    limit::Union{Int,Nothing}=nothing, after::Union{String,Nothing}=nothing, order::Union{String,Nothing}=nothing,
    service::ServiceEndpointSpec=OPENAIServiceEndpoint, config::Union{Nothing,RequestConfig}=nothing,
    cancel::Union{Nothing,CancelToken}=nothing)
    validate_capability(service, :fine_tuning, "Fine-tuning API")
    isnothing(order) || order in ("ascending", "descending") || throw(ArgumentError(
        "list_fine_tuning_checkpoint_permissions takes order \"ascending\" or \"descending\" (got $(repr(order)))"))
    cfg = _resolve_config(config); t0 = time_ns()
    query = String[]
    isnothing(project_id) || push!(query, "project_id=$(_uripart(project_id))")
    isnothing(order) || push!(query, "order=$(_uripart(order))")
    try
        _ft_list_resp(_http("GET", _ft_page(_ft_permissions_url(service, checkpoint), limit, after, query),
            auth_header(service); cfg, remaining=_remaining_s(cfg, t0), cancel=_resolve_cancel(cancel)))
    catch e
        e isa InterruptException && rethrow()
        _callerr(FineTuningCallError, e)
    end
end

"""
    delete_fine_tuning_checkpoint_permission(checkpoint, permission_id; service=OPENAIServiceEndpoint)

Delete the permission `permission_id` of the fine-tuned model checkpoint `checkpoint`: a
[`FineTuningDeleteSuccess`](@ref) when OpenAI's record names the permission, is a `"checkpoint.permission"`
object and says `"deleted": true`, the `FineTuningCallError` otherwise. Needs an admin API key, as
[`create_fine_tuning_checkpoint_permission`](@ref) does.
"""
function delete_fine_tuning_checkpoint_permission(checkpoint::String, permission_id::String;
    service::ServiceEndpointSpec=OPENAIServiceEndpoint, config::Union{Nothing,RequestConfig}=nothing,
    cancel::Union{Nothing,CancelToken}=nothing)
    validate_capability(service, :fine_tuning, "Fine-tuning API")
    cfg = _resolve_config(config); t0 = time_ns()
    try
        resp = _http("DELETE", _ft_permissions_url(service, checkpoint) * "/" * _uripart(permission_id),
            auth_header(service); cfg, remaining=_remaining_s(cfg, t0), cancel=_resolve_cancel(cancel))
        resp.status == 200 || return _failure(FineTuningFailure, resp)
        d = JSON.parse(resp.body; dicttype=Dict{String,Any})
        _vs_deletion(FineTuningDeleteSuccess, d, "checkpoint.permission")
    catch e
        e isa InterruptException && rethrow()
        _callerr(FineTuningCallError, e)
    end
end
