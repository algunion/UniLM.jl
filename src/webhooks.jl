# ============================================================================
# OpenAI Webhooks — verify inbound webhook signatures (Standard Webhooks / HMAC-
# SHA256) and parse events, which needs no provider endpoint and no capability; and
# manage a project's webhook endpoints (/v1/webhook_endpoints, the `:webhooks`
# capability). HMAC is built on the `SHA` stdlib to avoid a TLS dep.
# ============================================================================

"""
The event types of OpenAI's webhooks
([reference](https://developers.openai.com/api/reference/resources/webhooks)): background
responses, batches and fine-tuning jobs that end, incoming SIP calls for the Realtime and
Live APIs, a Live session that connects (`live.transport.connected`, "Sent when a direct Live
WebRTC session is created or an Instant Connect session activates"), safety warnings and
deactivations for a safety identifier, safety alerts, and the agent sessions of OpenAI's Agents
API and the environments they run in (`agent.session.*`, `agent.environment.*`, which OpenAI's
API specification lists).
`live.call.incoming` is deprecated in favour of `live.transport.incoming` and retained for
existing subscriptions. The reference also lists `eval.run.succeeded`, `eval.run.failed` and
`eval.run.canceled`, which OpenAI sends until its Evals API shuts down on November 30, 2026
([deprecations](https://developers.openai.com/api/docs/deprecations)): they are left out, so a
filter on this list drops them.
"""
const WEBHOOK_EVENTS = (
    "response.completed", "response.cancelled", "response.failed", "response.incomplete",
    "batch.completed", "batch.cancelled", "batch.expired", "batch.failed",
    "fine_tuning.job.succeeded", "fine_tuning.job.failed", "fine_tuning.job.cancelled",
    "realtime.call.incoming", "live.call.incoming", "live.transport.incoming", "live.transport.connected",
    "safety.warning_issued", "safety.deactivation_issued", "safety.alert.created", "safety.org_alert.created",
    "agent.environment.ready", "agent.environment.failed", "agent.environment.suspended", "agent.environment.expired",
    "agent.session.created", "agent.session.action_required", "agent.session.in_progress", "agent.session.idle",
    "agent.session.failed",
)

# HMAC-SHA256 via the SHA stdlib: H((K⊕opad) ‖ H((K⊕ipad) ‖ msg)), K padded to 64 bytes.
function _hmac_sha256(key::Vector{UInt8}, msg::Vector{UInt8})
    blocksize = 64
    k = length(key) > blocksize ? sha256(key) : key
    k = vcat(k, zeros(UInt8, blocksize - length(k)))
    sha256(vcat(UInt8(0x5c) .⊻ k, sha256(vcat(UInt8(0x36) .⊻ k, msg))))
end

# Constant-time comparison.
function _consteq(a::AbstractString, b::AbstractString)
    length(a) == length(b) || return false
    r = UInt8(0)
    for (x, y) in zip(codeunits(a), codeunits(b))
        r |= x ⊻ y
    end
    r == 0
end

# A Standard Webhooks signing secret is base64 behind its `whsec_` prefix ("Secret serialization:
# base64 encoded, prefixed with `whsec_`",
# https://github.com/standard-webhooks/standard-webhooks/blob/main/spec/standard-webhooks.md), read
# strictly: "Implementations MUST reject the encoded data if it contains characters outside the base
# alphabet" (RFC 4648, section 3.3) — `base64decode` skips them, so such a secret would decode to
# another key. Whole groups of four, then a last group of two or three characters, padded or not
# (the Standard Webhooks reference library pads an unpadded secret before decoding it). Anchored
# with \A and \z: `$` would also match before a final line break.
const _BASE64_SECRET = r"\A(?:[A-Za-z0-9+/]{4})*(?:[A-Za-z0-9+/]{2}(?:==)?|[A-Za-z0-9+/]{3}=?)?\z"

_header_dict(headers::AbstractDict)::Dict{String,String} = Dict(lowercase(string(k)) => string(v) for (k, v) in headers)
_header_dict(headers)::Dict{String,String} = Dict(lowercase(string(first(p))) => string(last(p)) for p in headers)

"""
    verify_webhook(payload::AbstractString, headers, secret::AbstractString; tolerance_seconds=300) -> Bool

Verify an OpenAI webhook signature (Standard Webhooks). `payload` is the raw request body;
`headers` is a Dict or iterable of pairs containing `webhook-id`, `webhook-timestamp`, and
`webhook-signature`; `secret` is the endpoint signing secret, with or without the `whsec_`
prefix: without it, the secret is read as base64 too, as the Standard Webhooks libraries read
it (OpenAI's SDKs take an unprefixed secret's raw bytes as the key). Returns `true` iff a
fresh, validly-signed `v1` signature is present.

Replay protection: timestamps outside `±tolerance_seconds` of now are rejected, as are
timestamps that are not finite numbers — pass `tolerance_seconds=Inf` to skip the time
check (e.g. when replaying a stored fixture). Uses a constant-time digest compare.
Negative or NaN tolerances throw `ArgumentError`.

Whitespace around `secret` is ignored. Throws `ArgumentError` before any check when the rest
is empty (`""`, whitespace, or `whsec_` alone): anyone can sign with an empty key, so under
one a forged delivery would verify. Throws `ArgumentError` when the rest is not valid
base64 — a character outside `A-Z`, `a-z`, `0-9`, `+` and `/` (whitespace inside it
included), or `=` padding anywhere but at its end or more than its last group lacks — for a
delivery that reaches the signature check: that is a misconfiguration of this endpoint, and
reporting it as an unverified signature would silently drop every webhook. A `false` return
therefore always means the message failed verification.
"""
function verify_webhook(payload::AbstractString, headers, secret::AbstractString; tolerance_seconds::Real=300)
    # Whitespace around the secret — the line break a file or an env file ends it with — is
    # no part of it, and every decoder that accepts it yields the intended key (OpenAI's SDKs
    # decode with Python's non-validating `b64decode` and Node's `Buffer.from`, which skip it).
    trimmed = strip(secret)
    sec = startswith(trimmed, "whsec_") ? trimmed[7:end] : trimmed
    # An empty secret is no key: anyone can compute an HMAC under an empty key, so a forged delivery
    # would verify and every genuine one fail. The Standard Webhooks libraries refuse it when the
    # verifier is built ("webhook secret may not be empty"); so does this, whatever the delivery.
    isempty(sec) && throw(ArgumentError("webhook signing secret is empty: anyone can sign with an empty key; " *
        "pass the signing secret OpenAI gave the endpoint"))
    !isnan(tolerance_seconds) && tolerance_seconds >= 0 || throw(ArgumentError(
        "tolerance_seconds must be nonnegative; only Inf explicitly disables replay protection"))
    h = _header_dict(headers)
    wid = get(h, "webhook-id", "")
    wts = get(h, "webhook-timestamp", "")
    wsig = get(h, "webhook-signature", "")
    (isempty(wid) || isempty(wts) || isempty(wsig)) && return false
    if isfinite(tolerance_seconds)
        ts = tryparse(Float64, wts)
        # A non-finite timestamp is not a point in time: every comparison against
        # NaN is false, so `abs(time() - ts) > tolerance` would wave "NaN" through
        # the replay window. Reject it as the malformed message it is.
        (isnothing(ts) || !isfinite(ts) || abs(time() - ts) > tolerance_seconds) && return false
    end
    # A secret that is not base64 is a configuration error on this side, not a forged
    # message. Returning false here would silently reject every inbound webhook forever,
    # indistinguishable from an attacker's bad signature.
    occursin(_BASE64_SECRET, sec) || throw(ArgumentError("webhook signing secret is not valid base64: the secret " *
        "(after its whsec_ prefix, if any) holds only A-Z, a-z, 0-9, + and /, with = padding at its end only"))
    key = base64decode(sec)
    # `string(...)` is a single `::String` method, so the signed base is concretely a
    # `String` regardless of how the header-map value types infer at this call.
    signed = string(wid, ".", wts, ".", payload)
    expected = base64encode(_hmac_sha256(key, Vector{UInt8}(signed)))
    for part in split(wsig, ' ')
        seg = split(part, ',')
        length(seg) == 2 && seg[1] == "v1" && _consteq(String(seg[2]), expected) && return true
    end
    return false
end

"""
    WebhookEvent

A parsed webhook event from [`parse_webhook`](@ref): `id`, `type`, and
`created_at`; `data` holds the event payload and `raw` the unparsed JSON response.
"""
@kwdef struct WebhookEvent
    id::String
    type::String
    created_at::Union{Int,Nothing} = nothing
    data::Dict{String,Any} = Dict{String,Any}()
    raw::Dict{String,Any} = Dict{String,Any}()
end

"""
    parse_webhook(payload::AbstractString) -> WebhookEvent

Parse a webhook JSON payload into a typed [`WebhookEvent`](@ref). Verify the signature with
[`verify_webhook`](@ref) first.
"""
function parse_webhook(payload::AbstractString)
    d = JSON.parse(payload; dicttype=Dict{String,Any})
    WebhookEvent(id=get(d, "id", ""), type=get(d, "type", ""), created_at=get(d, "created_at", nothing),
        data=Dict{String,Any}(get(d, "data", Dict{String,Any}())), raw=d)
end

# ─── Webhook endpoints ────────────────────────────────────────────────────────
# A project's webhook endpoints (OpenAI's API specification, ListWebhookEndpoints and the
# operations beside it), managed with the project's API key.

const _WEBHOOK_ENDPOINTS_PATH = "/v1/webhook_endpoints"
const _WH = "Webhook endpoints API"

"""
    WebhookEndpoint

A project's webhook endpoint: its `id`, `name`, `url`, the `event_types` it receives, `created_at`
and `updated_at` (Unix seconds; `updated_at` moves with a change of its configuration or secret),
the masked `signing_secret_hint`, and the `signing_secret` itself, which OpenAI returns only from
[`create_webhook_endpoint`](@ref) and [`rotate_webhook_endpoint_secret`](@ref) (`nothing` otherwise):
keep it for [`verify_webhook`](@ref). `raw` holds the reply.
"""
@kwdef struct WebhookEndpoint
    id::String
    name::String
    url::String
    event_types::Vector{String}
    created_at::Int
    updated_at::Union{Int,Nothing} = nothing
    signing_secret_hint::Union{String,Nothing} = nothing
    signing_secret::Union{String,Nothing} = nothing
    raw::Dict{String,Any} = Dict{String,Any}()
end

"A page of [`WebhookEndpoint`](@ref)s from [`list_webhook_endpoints`](@ref); `has_more` signals further pages, `raw` holds the reply (`first_id`, `last_id`)."
@kwdef struct WebhookEndpointList
    data::Vector{WebhookEndpoint}
    has_more::Bool
    raw::Dict{String,Any} = Dict{String,Any}()
end

"The sample delivery of [`test_webhook_endpoint`](@ref): the `event_type` sent to the endpoint `webhook_endpoint_id`, and the HTTP `status_code` the endpoint answered with."
@kwdef struct WebhookEndpointTest
    webhook_endpoint_id::String
    event_type::String
    status_code::Int
    raw::Dict{String,Any} = Dict{String,Any}()
end

"Successful create, retrieve, update or secret rotation result wrapping a [`WebhookEndpoint`](@ref)."
@kwdef struct WebhookEndpointSuccess <: LLMRequestResponse; response::WebhookEndpoint; end
"Successful [`list_webhook_endpoints`](@ref) result wrapping a [`WebhookEndpointList`](@ref)."
@kwdef struct WebhookEndpointListSuccess <: LLMRequestResponse; response::WebhookEndpointList; end
"Successful [`delete_webhook_endpoint`](@ref) result: the service confirmed (`deleted` is always `true`) the removal of `id`."
@kwdef struct WebhookEndpointDeleteSuccess <: LLMRequestResponse; id::String; deleted::Bool; end
"Successful [`test_webhook_endpoint`](@ref) result wrapping a [`WebhookEndpointTest`](@ref)."
@kwdef struct WebhookTestSuccess <: LLMRequestResponse; response::WebhookEndpointTest; end
"Successful [`list_webhook_event_types`](@ref) result: the `event_types` the project may subscribe to."
@kwdef struct WebhookEventTypesSuccess <: LLMRequestResponse; event_types::Vector{String}; end
"Webhook endpoints API error result: HTTP `status`, the raw `response` body, and the `request_id` the service sent (`x-request-id`/`request-id` header), if any."
@kwdef struct WebhookEndpointFailure <: LLMRequestResponse; response::String; status::Int; request_id::Union{String,Nothing} = nothing; end
"Webhook endpoints API call that produced no usable reply (transport failure, timeout, or a 200 that could not be decoded); `cause` is the underlying exception — a [`UniLMTimeout`](@ref) for a timeout."
@kwdef struct WebhookEndpointCallError <: LLMRequestResponse; error::String; status::Union{Int,Nothing} = nothing; cause::Union{Nothing,Exception} = nothing; end

# The reference requires an endpoint's id, name, url, event types and creation time, and the
# signing secret of a create or a rotation, without which no delivery could be verified.
function _parse_webhook_endpoint(d::AbstractDict; secret::Bool=false)::WebhookEndpoint
    WebhookEndpoint(id=_reply_required(d, "id", String, _WH), name=_reply_required(d, "name", String, _WH),
        url=_reply_required(d, "url", String, _WH),
        event_types=Vector{String}(_reply_required(d, "event_types", AbstractVector, _WH)),
        created_at=_reply_required(d, "created_at", Int, _WH), updated_at=_reply_field(d, "updated_at", Int, _WH),
        signing_secret_hint=_reply_field(d, "signing_secret_hint", String, _WH),
        signing_secret=secret ? _reply_required(d, "signing_secret", String, _WH) : _reply_field(d, "signing_secret", String, _WH),
        raw=Dict{String,Any}(d))
end

# One exchange with the Webhook endpoints API: `route` under its base, the reply decoded by `decode`.
function _webhook_endpoints_call(decode::Function, method::String, route::String, service, config, cancel;
                                 body::Union{AbstractDict,Nothing}=nothing)
    validate_capability(service, :webhooks, _WH)
    cfg = _resolve_config(config); t0 = time_ns()
    try
        url = _api_base_url(service) * route
        kw = (; cfg, remaining=_remaining_s(cfg, t0), cancel=_resolve_cancel(cancel))
        resp = isnothing(body) ? _http(method, url, auth_header(service); kw...) :
                                 _http(method, url, auth_header(service), JSON.json(body); kw...)
        resp.status == 200 ? decode(JSON.parse(resp.body; dicttype=Dict{String,Any})) : _failure(WebhookEndpointFailure, resp)
    catch e
        e isa InterruptException && rethrow()
        _callerr(WebhookEndpointCallError, e)
    end
end
_webhook_endpoint_route(id::String) = _WEBHOOK_ENDPOINTS_PATH * "/" * _uripart(id)

"""
    create_webhook_endpoint(; name, url, event_types, service=OPENAIServiceEndpoint)

Make a webhook endpoint for the project: OpenAI delivers the `event_types` (see
[`list_webhook_event_types`](@ref)) to the HTTPS `url`. The result's endpoint holds its
`signing_secret`, which OpenAI returns only here and from [`rotate_webhook_endpoint_secret`](@ref);
a reply without one is the `WebhookEndpointCallError`.

Every Webhook endpoints verb needs the `:webhooks` capability, makes one attempt bounded by its
`config`, and takes `cancel`: a cancelled token returns a `WebhookEndpointCallError` whose `cause` is a
[`UniLMCancelled`](@ref), without sending anything when it was cancelled before the call.
"""
create_webhook_endpoint(; name::String, url::String, event_types::Vector{String},
    service::ServiceEndpointSpec=OPENAIServiceEndpoint, config::Union{Nothing,RequestConfig}=nothing,
    cancel::Union{Nothing,CancelToken}=nothing) =
    _webhook_endpoints_call("POST", _WEBHOOK_ENDPOINTS_PATH, service, config, cancel;
                            body=Dict(:name => name, :url => url, :event_types => event_types)) do d
        WebhookEndpointSuccess(response=_parse_webhook_endpoint(d; secret=true))
    end

"""
    list_webhook_endpoints(; limit=nothing, after=nothing, service=OPENAIServiceEndpoint)

The project's webhook endpoints, newest first, a page at a time: `limit` (1 to 100; OpenAI's
default 20) and `after`, the last id of the previous page. A reply without `data` or `has_more` is
the `WebhookEndpointCallError`.
"""
function list_webhook_endpoints(; limit::Union{Int,Nothing}=nothing, after::Union{String,Nothing}=nothing,
    service::ServiceEndpointSpec=OPENAIServiceEndpoint, config::Union{Nothing,RequestConfig}=nothing,
    cancel::Union{Nothing,CancelToken}=nothing)
    _webhook_endpoints_call("GET", _ft_page(_WEBHOOK_ENDPOINTS_PATH, limit, after), service, config, cancel) do d
        WebhookEndpointListSuccess(response=WebhookEndpointList(
            data=WebhookEndpoint[_parse_webhook_endpoint(e) for e in _reply_required(d, "data", AbstractVector, _WH)],
            has_more=_reply_required(d, "has_more", Bool, _WH), raw=d))
    end
end

"""
    retrieve_webhook_endpoint(id; service=OPENAIServiceEndpoint)

The webhook endpoint `id`, without its signing secret.
"""
retrieve_webhook_endpoint(id::String; service::ServiceEndpointSpec=OPENAIServiceEndpoint,
    config::Union{Nothing,RequestConfig}=nothing, cancel::Union{Nothing,CancelToken}=nothing) =
    _webhook_endpoints_call("GET", _webhook_endpoint_route(id), service, config, cancel) do d
        WebhookEndpointSuccess(response=_parse_webhook_endpoint(d))
    end

"""
    update_webhook_endpoint(id; name=nothing, url=nothing, event_types=nothing, service=OPENAIServiceEndpoint)

Change the webhook endpoint `id`: each of `name`, `url` and `event_types` (the complete new set)
that is given; the rest stay. The result holds the endpoint without its signing secret.
"""
function update_webhook_endpoint(id::String; name::Union{String,Nothing}=nothing, url::Union{String,Nothing}=nothing,
    event_types::Union{Vector{String},Nothing}=nothing, service::ServiceEndpointSpec=OPENAIServiceEndpoint,
    config::Union{Nothing,RequestConfig}=nothing, cancel::Union{Nothing,CancelToken}=nothing)
    body = Dict{Symbol,Any}(k => v for (k, v) in ((:name, name), (:url, url), (:event_types, event_types)) if !isnothing(v))
    _webhook_endpoints_call("POST", _webhook_endpoint_route(id), service, config, cancel; body) do d
        WebhookEndpointSuccess(response=_parse_webhook_endpoint(d))
    end
end

"""
    delete_webhook_endpoint(id; service=OPENAIServiceEndpoint)

Delete the webhook endpoint `id`: a [`WebhookEndpointDeleteSuccess`](@ref) when OpenAI's record
names the endpoint and says `"deleted": true`, the `WebhookEndpointCallError` otherwise.
"""
delete_webhook_endpoint(id::String; service::ServiceEndpointSpec=OPENAIServiceEndpoint,
    config::Union{Nothing,RequestConfig}=nothing, cancel::Union{Nothing,CancelToken}=nothing) =
    _webhook_endpoints_call("DELETE", _webhook_endpoint_route(id), service, config, cancel) do d
        WebhookEndpointDeleteSuccess(id=_reply_required(d, "id", String, _WH), deleted=_confirm_deleted(d))
    end

"""
    rotate_webhook_endpoint_secret(id; keep_old_secret_active_for_24_hours=nothing, service=OPENAIServiceEndpoint)

Give the webhook endpoint `id` a new signing secret, held by the result's endpoint. The previous
secret stops verifying at once, unless `keep_old_secret_active_for_24_hours=true` keeps it valid for
24 hours.
"""
function rotate_webhook_endpoint_secret(id::String; keep_old_secret_active_for_24_hours::Union{Bool,Nothing}=nothing,
    service::ServiceEndpointSpec=OPENAIServiceEndpoint, config::Union{Nothing,RequestConfig}=nothing,
    cancel::Union{Nothing,CancelToken}=nothing)
    body = isnothing(keep_old_secret_active_for_24_hours) ? Dict{Symbol,Any}() :
        Dict{Symbol,Any}(:keep_old_secret_active_for_24_hours => keep_old_secret_active_for_24_hours)
    _webhook_endpoints_call("POST", _webhook_endpoint_route(id) * "/rotate_secret", service, config, cancel; body) do d
        WebhookEndpointSuccess(response=_parse_webhook_endpoint(d; secret=true))
    end
end

"""
    test_webhook_endpoint(id, event_type; service=OPENAIServiceEndpoint)

Send a sample `event_type` delivery to the webhook endpoint `id`: a [`WebhookTestSuccess`](@ref)
holding the HTTP status code the endpoint answered with. OpenAI's 500 says the sample could not be
sent, a `WebhookEndpointFailure`.
"""
test_webhook_endpoint(id::String, event_type::String; service::ServiceEndpointSpec=OPENAIServiceEndpoint,
    config::Union{Nothing,RequestConfig}=nothing, cancel::Union{Nothing,CancelToken}=nothing) =
    _webhook_endpoints_call("POST", _webhook_endpoint_route(id) * "/test", service, config, cancel;
                            body=Dict(:event_type => event_type)) do d
        WebhookTestSuccess(response=WebhookEndpointTest(
            webhook_endpoint_id=_reply_required(d, "webhook_endpoint_id", String, _WH),
            event_type=_reply_required(d, "event_type", String, _WH),
            status_code=_reply_required(d, "status_code", Int, _WH), raw=d))
    end

"""
    list_webhook_event_types(; service=OPENAIServiceEndpoint)

The event types the project may subscribe a webhook endpoint to, as a
[`WebhookEventTypesSuccess`](@ref).
"""
list_webhook_event_types(; service::ServiceEndpointSpec=OPENAIServiceEndpoint,
    config::Union{Nothing,RequestConfig}=nothing, cancel::Union{Nothing,CancelToken}=nothing) =
    _webhook_endpoints_call("GET", "/v1/webhook_event_types", service, config, cancel) do d
        WebhookEventTypesSuccess(event_types=Vector{String}(_reply_required(d, "data", AbstractVector, _WH)))
    end
