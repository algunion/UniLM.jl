# ============================================================================
# Anthropic (Claude) native Messages API
# Plugs into the wire-translation seam (encode_request / decode_response /
# decode_stream_chunk from requests.jl) so all chat orchestration is shared.
# Wire shape verified against the claude-api reference on 2026-07-06.
# ============================================================================

# ─── Routing & auth ──────────────────────────────────────────────────────────

get_url(::Type{ANTHROPICServiceEndpoint}, ::Chat) = ANTHROPIC_BASE_URL * ANTHROPIC_MESSAGES_PATH

function auth_header(::Type{ANTHROPICServiceEndpoint})
    [
        "x-api-key" => ENV[ANTHROPIC_API_KEY],
        "anthropic-version" => ANTHROPIC_VERSION,
        "Content-Type" => "application/json",
    ]
end

# ─── Capabilities & defaults ─────────────────────────────────────────────────

provider_capabilities(::Type{ANTHROPICServiceEndpoint}) =
    Set([:chat, :tools, :json_output, :streaming])

default_model(::Type{ANTHROPICServiceEndpoint}) = "claude-opus-4-8"

"""
    default_max_tokens(service, model::AbstractString) -> Int

`max_tokens` supplied when the caller leaves it unset. Anthropic *requires* the
field; OpenAI does not. Returns a moderate, overridable default (see
`_ANTHROPIC_DEFAULT_MAX_TOKENS`), not the model ceiling.
"""
default_max_tokens(::Type{ANTHROPICServiceEndpoint}, ::AbstractString) = _ANTHROPIC_DEFAULT_MAX_TOKENS
