# ============================================================================
# OpenAI safety and provenance — read the safety case behind a warning or a
# deactivation (/v1/safety/cases, the `:safety` capability) and a project's safety
# alert (/v1/safety/alerts), the ids the `safety.*` webhooks carry; and check a file
# for OpenAI's provenance signals (/v1/content_provenance_checks, `:content_provenance`).
# ============================================================================

const _SAFETY = "Safety API"
const _PROVENANCE = "Content provenance API"

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

"""
    ProvenanceResult

One provenance signal of a [`ProvenanceCheck`](@ref): its `type`, `"c2pa"` (C2PA Content
Credentials) or `"synthid"` (a SynthID watermark); its `outcome`, `"detected"` or `"not_detected"`;
for C2PA, the manifest's `validation_state` (`"trusted"`, `"valid"`, `"invalid"` or `"not_present"`)
and `issuer`; the OpenAI `model` and the `generated_at` time (RFC 3339) the signal records, when
it records them. A field the signal does not have is `nothing`.
"""
@kwdef struct ProvenanceResult
    type::String
    outcome::String
    validation_state::Union{String,Nothing} = nothing
    issuer::Union{String,Nothing} = nothing
    model::Union{String,Nothing} = nothing
    generated_at::Union{String,Nothing} = nothing
end

"""
    ProvenanceCheck

The result of [`check_content_provenance`](@ref): `created_at` (Unix seconds) and the
[`ProvenanceResult`](@ref)s that apply to the file. `raw` holds the reply.
"""
@kwdef struct ProvenanceCheck
    created_at::Int
    results::Vector{ProvenanceResult}
    raw::Dict{String,Any} = Dict{String,Any}()
end

"Successful [`retrieve_safety_case`](@ref) result wrapping a [`SafetyCase`](@ref)."
@kwdef struct SafetyCaseSuccess <: LLMRequestResponse; response::SafetyCase; end
"Successful [`retrieve_safety_alert`](@ref) result wrapping a [`SafetyAlert`](@ref)."
@kwdef struct SafetyAlertSuccess <: LLMRequestResponse; response::SafetyAlert; end
"Successful [`check_content_provenance`](@ref) result wrapping a [`ProvenanceCheck`](@ref)."
@kwdef struct ProvenanceCheckSuccess <: LLMRequestResponse; response::ProvenanceCheck; end
"Safety or content provenance API error result: HTTP `status`, the raw `response` body, and the `request_id` the service sent (`x-request-id`/`request-id` header), if any."
@kwdef struct SafetyFailure <: LLMRequestResponse; response::String; status::Int; request_id::Union{String,Nothing} = nothing; end
"Safety or content provenance API call that produced no usable reply (transport failure, timeout, or a 200 that could not be decoded); `cause` is the underlying exception — a [`UniLMTimeout`](@ref) for a timeout."
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
read, is OpenAI's 404, and a key without the `api.safety.read` scope gets its 403 (witnessed on
November 16, 2026): each a `SafetyFailure`.

Each safety and provenance verb makes one attempt bounded by its `config` and takes `cancel`: a
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

# A provenance signal: the reference requires its type and outcome, and allows the rest to be null.
_parse_provenance_result(d::AbstractDict) = ProvenanceResult(type=_reply_required(d, "type", String, _PROVENANCE),
    outcome=_reply_required(d, "outcome", String, _PROVENANCE),
    validation_state=_reply_field(d, "validation_state", String, _PROVENANCE), issuer=_reply_field(d, "issuer", String, _PROVENANCE),
    model=_reply_field(d, "model", String, _PROVENANCE), generated_at=_reply_field(d, "generated_at", String, _PROVENANCE))

"""
    check_content_provenance(path; mime_type=<from the extension>, service=OPENAIServiceEndpoint)

Check the image, audio or PDF file at `path` for OpenAI's provenance signals: a
[`ProvenanceCheckSuccess`](@ref) holding a [`ProvenanceResult`](@ref) per signal that applies to the
file. The [content provenance](https://developers.openai.com/api/docs/guides/content-provenance) guide
lists the formats: "Images: PNG, JPEG, WebP, and GIF. Audio: MP3, Opus, AAC, FLAC, WAV, and PCM.
Documents: PDF." (OpenAI's API specification names image and audio files only). The file goes as
the one multipart part `file`, whose own `Content-Type` is `mime_type`: by default the one its
extension names, as [`upload_file`](@ref) sends it; pass it for an extension
that names none, such as `.opus` or `.aac`. A path that is no file is an `ArgumentError`.

`"not_detected"` means the check found no supported signal: the file may still come from OpenAI,
its metadata stripped or its watermark degraded, or from another company's model, which the check
does not detect ([content provenance](https://developers.openai.com/api/docs/guides/content-provenance)).
"""
function check_content_provenance(path::String; mime_type::String=_mime_for(path),
    service::ServiceEndpointSpec=OPENAIServiceEndpoint, config::Union{Nothing,RequestConfig}=nothing,
    cancel::Union{Nothing,CancelToken}=nothing)
    validate_capability(service, :content_provenance, _PROVENANCE)
    isfile(path) || throw(ArgumentError("file not found: $path"))
    cfg = _resolve_config(config); t0 = time_ns()
    tok = _resolve_cancel(cancel)
    try
        # A cancelled token reads no file: the form holds the whole file, read before `_http` checks the token.
        iscancelled(tok) && throw(UniLMCancelled(:token, _elapsed_s(t0)))
        form = HTTP.Form(["file" => HTTP.Multipart(basename(path), IOBuffer(read(path)), mime_type)])
        resp = _http("POST", _api_base_url(service) * "/v1/content_provenance_checks", auth_header_multipart(service), form;
                     cfg, remaining=_remaining_s(cfg, t0), cancel=tok)
        resp.status == 200 || return _failure(SafetyFailure, resp)
        d = JSON.parse(resp.body; dicttype=Dict{String,Any})
        ProvenanceCheckSuccess(response=ProvenanceCheck(created_at=_reply_required(d, "created_at", Int, _PROVENANCE),
            results=ProvenanceResult[_parse_provenance_result(r) for r in _reply_required(d, "results", AbstractVector, _PROVENANCE)],
            raw=d))
    catch e
        e isa InterruptException && rethrow()
        _callerr(SafetyCallError, e)
    end
end
