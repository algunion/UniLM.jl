# ============================================================================
# Google Gemini native generateContent API
# Plugs into the wire-translation seam (encode_request / decode_response /
# decode_stream_chunk from requests.jl) so all chat orchestration is shared.
# Wire shape verified against ai.google.dev live docs on 2026-07-07.
# ============================================================================

# ─── Routing & auth ──────────────────────────────────────────────────────────
# Model is in the URL (like Azure); streaming is the URL METHOD, not a body flag.

function get_url(::Type{GEMINIServiceEndpoint}, chat::Chat)
    if chat.stream === true
        "$(GEMINI_NATIVE_BASE)/models/$(chat.model):streamGenerateContent?alt=sse"
    else
        "$(GEMINI_NATIVE_BASE)/models/$(chat.model):generateContent"
    end
end

_api_base_url(::Type{GEMINIServiceEndpoint}) =
    throw(ArgumentError("Responses API is only supported with OPENAIServiceEndpoint"))

auth_header(::Type{GEMINIServiceEndpoint}) = [
    "x-goog-api-key" => ENV[GEMINI_API_KEY],
    "Content-Type"   => "application/json",
]

# ─── Capabilities & defaults ─────────────────────────────────────────────────

provider_capabilities(::Type{GEMINIServiceEndpoint}) = Set([:chat, :tools, :streaming])

default_model(::Type{GEMINIServiceEndpoint}) = "gemini-3.5-flash"
