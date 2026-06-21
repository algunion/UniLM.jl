# ============================================================================
# OpenAI Realtime API — low-latency speech/text over a WebSocket. This client
# covers the WebSocket event transport + ephemeral client-secret minting. WebRTC
# media capture and SIP telephony are OUT OF SCOPE (need a native media stack
# Julia lacks); audio is exchanged as base64 PCM inside events.
# ============================================================================

@kwdef struct RealtimeSecretSuccess <: LLMRequestResponse
    value::String
    raw::Dict{String,Any} = Dict{String,Any}()
end
@kwdef struct RealtimeFailure <: LLMRequestResponse; response::String; status::Int; end
@kwdef struct RealtimeCallError <: LLMRequestResponse; error::String; status::Union{Int,Nothing} = nothing; end

"""
    mint_realtime_secret(; session=nothing, service=OPENAIServiceEndpoint)

Create an ephemeral client secret for client-side Realtime connections
(`POST /v1/realtime/client_secrets`). `session` is an optional session-config dict.
Returns `RealtimeSecretSuccess` (`.value`), `RealtimeFailure`, or `RealtimeCallError`.
"""
function mint_realtime_secret(; session::Union{AbstractDict,Nothing}=nothing, service::ServiceEndpointSpec=OPENAIServiceEndpoint)
    validate_capability(service, :realtime, "Realtime API")
    try
        d = Dict{Symbol,Any}()
        !isnothing(session) && (d[:session] = session)
        resp = HTTP.post(_api_base_url(service) * REALTIME_CLIENT_SECRETS_PATH, body=JSON.json(d), headers=auth_header(service); status_exception=false)
        resp.status == 200 || return RealtimeFailure(response=String(resp.body), status=resp.status)
        data = JSON.parse(resp.body; dicttype=Dict{String,Any})
        cs = get(data, "client_secret", nothing)
        val = get(data, "value", cs isa AbstractDict ? get(cs, "value", "") : "")
        RealtimeSecretSuccess(value=val, raw=data)
    catch e
        RealtimeCallError(error=string(e), status=(hasproperty(e, :status) ? e.status : nothing))
    end
end

# ─── Client → server event builders (return plain dicts) ─────────────────────

"""    realtime_event(type; kwargs...) — a generic client event dict."""
realtime_event(type::String; kwargs...) = Dict{Symbol,Any}(:type => type, kwargs...)

"""    session_update(session) — a `session.update` event."""
session_update(session::AbstractDict) = Dict{Symbol,Any}(:type => "session.update", :session => session)

"""    input_audio_append(audio_b64) — append base64 PCM to the input audio buffer."""
input_audio_append(audio_b64::String) = Dict{Symbol,Any}(:type => "input_audio_buffer.append", :audio => audio_b64)

"""    response_create(; response=nothing) — request a model response."""
function response_create(; response::Union{AbstractDict,Nothing}=nothing)
    d = Dict{Symbol,Any}(:type => "response.create")
    !isnothing(response) && (d[:response] = response)
    return d
end

# ─── WebSocket transport ──────────────────────────────────────────────────────

"""A live Realtime WebSocket session. Created by [`realtime_connect`](@ref)."""
mutable struct RealtimeSession
    ws::Any
    model::String
end

"""
    realtime_connect(handler; model="gpt-realtime-2", service=OPENAIServiceEndpoint)

Open a Realtime WebSocket and run `handler(session::RealtimeSession)`. Inside the handler use
[`realtime_send`](@ref) to send event dicts and [`realtime_receive`](@ref) to read server
events. The socket closes when `handler` returns.
"""
# WS base URL is a function so tests can point realtime_connect at a local echo server.
_realtime_ws_url(service) = REALTIME_WS_URL

function realtime_connect(handler; model::String="gpt-realtime-2", service::ServiceEndpointSpec=OPENAIServiceEndpoint)
    validate_capability(service, :realtime, "Realtime API")
    url = _realtime_ws_url(service) * "?model=" * model
    # auth_header_multipart drops the JSON Content-Type, which is meaningless on a WS upgrade.
    HTTP.WebSockets.open(url; headers=auth_header_multipart(service)) do ws
        handler(RealtimeSession(ws, model))
    end
end

"""    realtime_send(session, event::AbstractDict)"""
realtime_send(s::RealtimeSession, event::AbstractDict) = HTTP.WebSockets.send(s.ws, JSON.json(event))

"""    realtime_receive(session) -> Dict   (blocks for the next server event)"""
realtime_receive(s::RealtimeSession) = JSON.parse(String(HTTP.WebSockets.receive(s.ws)); dicttype=Dict{String,Any})
