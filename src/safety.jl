# ============================================================================
# OpenAI safety — read the safety case behind a warning or a deactivation
# (/v1/safety/cases, the `:safety` capability) and a project's safety alert
# (/v1/safety/alerts), the ids the `safety.*` webhooks carry.
# ============================================================================

const _SAFETY = "Safety API"

"""
    SafetyCase

A safety case, the record behind a `safety.warning_issued` or `safety.deactivation_issued`
webhook: its `id`, `created_at` (Unix seconds), the `entity_identifier` it concerns (a
`safety_identifier` your requests sent), its `reason` (`nothing` when OpenAI gives none) and the
`notice` it issued, `"warning"` or `"deactivation"`. `raw` holds the reply.
"""
@kwdef struct SafetyCase
    id::String
    created_at::Int
    entity_identifier::String
    reason::Union{String,Nothing} = nothing
    notice::String
    raw::Dict{String,Any} = Dict{String,Any}()
end

"""
    SafetyAlert

A project safety alert, the record behind a `safety.alert.created` webhook: its `id`,
`created_at` (Unix seconds), the `request_id`, `response_id` and `model` of the request it
concerns, `request_paused` (whether OpenAI registered the block; it does not confirm that the
response stopped), the `error_type` (`"potentially_unintended_data_transfer"`,
`"potentially_unintended_data_access"`, `"potentially_unintended_destructive_activity"` or `"other"`)
and its `reason`, `nothing` for a zero-data-retention request. `raw` holds the reply.
"""
@kwdef struct SafetyAlert
    id::String
    created_at::Int
    request_id::String
    response_id::String
    model::String
    request_paused::Bool
    error_type::String
    reason::Union{String,Nothing} = nothing
    raw::Dict{String,Any} = Dict{String,Any}()
end

"Successful [`retrieve_safety_case`](@ref) result wrapping a [`SafetyCase`](@ref)."
@kwdef struct SafetyCaseSuccess <: LLMRequestResponse; response::SafetyCase; end
"Successful [`retrieve_safety_alert`](@ref) result wrapping a [`SafetyAlert`](@ref)."
@kwdef struct SafetyAlertSuccess <: LLMRequestResponse; response::SafetyAlert; end
"Safety API error result: HTTP `status`, the raw `response` body, and the `request_id` the service sent (`x-request-id`/`request-id` header), if any."
@kwdef struct SafetyFailure <: LLMRequestResponse; response::String; status::Int; request_id::Union{String,Nothing} = nothing; end
"Safety API call that produced no usable reply (transport failure, timeout, or a 200 that could not be decoded); `cause` is the underlying exception — a [`UniLMTimeout`](@ref) for a timeout."
@kwdef struct SafetyCallError <: LLMRequestResponse; error::String; status::Union{Int,Nothing} = nothing; cause::Union{Nothing,Exception} = nothing; end

# The reference requires each field typed here but `reason`, which it allows to be null.
_parse_safety_case(d::AbstractDict) = SafetyCase(id=_reply_required(d, "id", String, _SAFETY),
    created_at=_reply_required(d, "created_at", Int, _SAFETY),
    entity_identifier=_reply_required(d, "entity_identifier", String, _SAFETY),
    reason=_reply_field(d, "reason", String, _SAFETY),
    notice=_reply_required(_reply_required(d, "notice", AbstractDict, _SAFETY), "type", String, _SAFETY),
    raw=Dict{String,Any}(d))
_parse_safety_alert(d::AbstractDict) = SafetyAlert(id=_reply_required(d, "id", String, _SAFETY),
    created_at=_reply_required(d, "created_at", Int, _SAFETY), request_id=_reply_required(d, "request_id", String, _SAFETY),
    response_id=_reply_required(d, "response_id", String, _SAFETY), model=_reply_required(d, "model", String, _SAFETY),
    request_paused=_reply_required(d, "request_paused", Bool, _SAFETY),
    error_type=_reply_required(d, "error_type", String, _SAFETY), reason=_reply_field(d, "reason", String, _SAFETY),
    raw=Dict{String,Any}(d))

function _safety_get(decode::Function, route::String, id::String, service, config, cancel)
    validate_capability(service, :safety, _SAFETY)
    cfg = _resolve_config(config); t0 = time_ns()
    try
        resp = _http("GET", _api_base_url(service) * route * "/" * _uripart(id), auth_header(service);
                     cfg, remaining=_remaining_s(cfg, t0), cancel=_resolve_cancel(cancel))
        resp.status == 200 ? decode(JSON.parse(resp.body; dicttype=Dict{String,Any})) : _failure(SafetyFailure, resp)
    catch e
        e isa InterruptException && rethrow()
        _callerr(SafetyCallError, e)
    end
end

"""
    retrieve_safety_case(id; service=OPENAIServiceEndpoint)

The safety case `id` (a `safety.warning_issued` or `safety.deactivation_issued` webhook's
`data["id"]`) as a [`SafetyCaseSuccess`](@ref); a case that does not exist, or that the key may not
read, is OpenAI's 404, a `SafetyFailure`.

Each safety verb makes one attempt bounded by its `config` and takes `cancel`: a
cancelled token returns a `SafetyCallError` whose `cause` is a [`UniLMCancelled`](@ref), without
sending anything when it was cancelled before the call.
"""
retrieve_safety_case(id::String; service::ServiceEndpointSpec=OPENAIServiceEndpoint,
    config::Union{Nothing,RequestConfig}=nothing, cancel::Union{Nothing,CancelToken}=nothing) =
    _safety_get(d -> SafetyCaseSuccess(response=_parse_safety_case(d)), "/v1/safety/cases", id, service, config, cancel)

"""
    retrieve_safety_alert(id; service=OPENAIServiceEndpoint)

The project safety alert `id` (a `safety.alert.created` webhook's `data["id"]`) as a
[`SafetyAlertSuccess`](@ref); an alert of another project, or none, is OpenAI's 404, a `SafetyFailure`.
"""
retrieve_safety_alert(id::String; service::ServiceEndpointSpec=OPENAIServiceEndpoint,
    config::Union{Nothing,RequestConfig}=nothing, cancel::Union{Nothing,CancelToken}=nothing) =
    _safety_get(d -> SafetyAlertSuccess(response=_parse_safety_alert(d)), "/v1/safety/alerts", id, service, config, cancel)
