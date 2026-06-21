# ============================================================================
# OpenAI Fine-tuning API — /v1/fine_tuning/jobs. Training/validation data is
# uploaded via the Files API (purpose="fine-tune"). `method` selects supervised /
# dpo / reinforcement with hyperparameters.
# ============================================================================

@kwdef struct FineTuningJob
    id::String
    status::Union{String,Nothing} = nothing
    model::Union{String,Nothing} = nothing
    fine_tuned_model::Union{String,Nothing} = nothing
    raw::Dict{String,Any} = Dict{String,Any}()
end

@kwdef struct FineTuningList
    data::Vector{Dict{String,Any}}
    has_more::Bool = false
    raw::Dict{String,Any} = Dict{String,Any}()
end

@kwdef struct FineTuningSuccess <: LLMRequestResponse; response::FineTuningJob; end
@kwdef struct FineTuningListSuccess <: LLMRequestResponse; response::FineTuningList; end
@kwdef struct FineTuningFailure <: LLMRequestResponse; response::String; status::Int; end
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
"""
function create_fine_tuning_job(; model::String, training_file::String,
    validation_file::Union{String,Nothing}=nothing, method::Union{AbstractDict,Nothing}=nothing,
    suffix::Union{String,Nothing}=nothing, metadata::Union{AbstractDict,Nothing}=nothing,
    service::ServiceEndpointSpec=OPENAIServiceEndpoint)
    validate_capability(service, :fine_tuning, "Fine-tuning API")
    try
        d = Dict{Symbol,Any}(:model => model, :training_file => training_file)
        !isnothing(validation_file) && (d[:validation_file] = validation_file)
        !isnothing(method) && (d[:method] = method)
        !isnothing(suffix) && (d[:suffix] = suffix)
        !isnothing(metadata) && (d[:metadata] = metadata)
        _ft_job_resp(HTTP.post(_api_base_url(service) * FINE_TUNING_PATH, body=JSON.json(d), headers=auth_header(service); status_exception=false))
    catch e
        _ft_err(e)
    end
end

"""    retrieve_fine_tuning_job(id; service=OPENAIServiceEndpoint)"""
function retrieve_fine_tuning_job(id::String; service::ServiceEndpointSpec=OPENAIServiceEndpoint)
    validate_capability(service, :fine_tuning, "Fine-tuning API")
    try
        _ft_job_resp(HTTP.get(_api_base_url(service) * FINE_TUNING_PATH * "/" * id, headers=auth_header(service); status_exception=false))
    catch e
        _ft_err(e)
    end
end

"""    cancel_fine_tuning_job(id; service=OPENAIServiceEndpoint)"""
function cancel_fine_tuning_job(id::String; service::ServiceEndpointSpec=OPENAIServiceEndpoint)
    validate_capability(service, :fine_tuning, "Fine-tuning API")
    try
        _ft_job_resp(HTTP.post(_api_base_url(service) * FINE_TUNING_PATH * "/" * id * "/cancel", headers=auth_header(service); status_exception=false))
    catch e
        _ft_err(e)
    end
end

"""    list_fine_tuning_jobs(; limit=nothing, after=nothing, service=OPENAIServiceEndpoint)"""
function list_fine_tuning_jobs(; limit::Union{Int,Nothing}=nothing, after::Union{String,Nothing}=nothing, service::ServiceEndpointSpec=OPENAIServiceEndpoint)
    validate_capability(service, :fine_tuning, "Fine-tuning API")
    try
        url = _api_base_url(service) * FINE_TUNING_PATH
        params = String[]
        !isnothing(limit) && push!(params, "limit=$limit")
        !isnothing(after) && push!(params, "after=$after")
        !isempty(params) && (url *= "?" * join(params, "&"))
        _ft_list_resp(HTTP.get(url, headers=auth_header(service); status_exception=false))
    catch e
        _ft_err(e)
    end
end

"""    list_fine_tuning_events(id; service=OPENAIServiceEndpoint)"""
function list_fine_tuning_events(id::String; service::ServiceEndpointSpec=OPENAIServiceEndpoint)
    validate_capability(service, :fine_tuning, "Fine-tuning API")
    try
        _ft_list_resp(HTTP.get(_api_base_url(service) * FINE_TUNING_PATH * "/" * id * "/events", headers=auth_header(service); status_exception=false))
    catch e
        _ft_err(e)
    end
end

"""    list_fine_tuning_checkpoints(id; service=OPENAIServiceEndpoint)"""
function list_fine_tuning_checkpoints(id::String; service::ServiceEndpointSpec=OPENAIServiceEndpoint)
    validate_capability(service, :fine_tuning, "Fine-tuning API")
    try
        _ft_list_resp(HTTP.get(_api_base_url(service) * FINE_TUNING_PATH * "/" * id * "/checkpoints", headers=auth_header(service); status_exception=false))
    catch e
        _ft_err(e)
    end
end
