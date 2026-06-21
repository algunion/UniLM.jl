# ============================================================================
# OpenAI Webhooks — verify inbound webhook signatures (Standard Webhooks / HMAC-
# SHA256) and parse events. This is an INBOUND utility (no provider endpoint), so
# there is no capability symbol. HMAC is built on the `SHA` stdlib to avoid a TLS dep.
# ============================================================================

"""Known OpenAI webhook event types."""
const WEBHOOK_EVENTS = (
    "response.completed", "response.cancelled", "response.failed", "response.incomplete",
    "batch.completed", "batch.cancelled", "batch.expired", "batch.failed",
    "fine_tuning.job.succeeded", "fine_tuning.job.failed", "fine_tuning.job.cancelled",
    "eval.run.succeeded", "eval.run.failed", "eval.run.canceled", "realtime.call.incoming",
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

_header_dict(headers::AbstractDict) = Dict(lowercase(string(k)) => string(v) for (k, v) in headers)
_header_dict(headers) = Dict(lowercase(string(first(p))) => string(last(p)) for p in headers)

"""
    verify_webhook(payload::AbstractString, headers, secret::AbstractString; tolerance_seconds=300) -> Bool

Verify an OpenAI webhook signature (Standard Webhooks). `payload` is the raw request body;
`headers` is a Dict or iterable of pairs containing `webhook-id`, `webhook-timestamp`, and
`webhook-signature`; `secret` is the endpoint signing secret (with or without the `whsec_`
prefix). Returns `true` iff a fresh, validly-signed `v1` signature is present.

Replay protection: timestamps outside `±tolerance_seconds` of now are rejected — pass
`tolerance_seconds=Inf` to skip the time check (e.g. when replaying a stored fixture). Uses
a constant-time digest compare.
"""
function verify_webhook(payload::AbstractString, headers, secret::AbstractString; tolerance_seconds::Real=300)
    h = _header_dict(headers)
    wid = get(h, "webhook-id", "")
    wts = get(h, "webhook-timestamp", "")
    wsig = get(h, "webhook-signature", "")
    (isempty(wid) || isempty(wts) || isempty(wsig)) && return false
    if isfinite(tolerance_seconds)
        ts = tryparse(Float64, wts)
        (isnothing(ts) || abs(time() - ts) > tolerance_seconds) && return false
    end
    sec = startswith(secret, "whsec_") ? secret[7:end] : secret
    key = try
        base64decode(sec)
    catch
        return false
    end
    signed = wid * "." * wts * "." * String(payload)
    expected = base64encode(_hmac_sha256(key, Vector{UInt8}(signed)))
    for part in split(wsig, ' ')
        seg = split(part, ',')
        length(seg) == 2 && seg[1] == "v1" && _consteq(String(seg[2]), expected) && return true
    end
    return false
end

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
