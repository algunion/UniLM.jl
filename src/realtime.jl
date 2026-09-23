# ============================================================================
# OpenAI Realtime API — low-latency speech/text over a WebSocket. This client
# covers the WebSocket event transport + ephemeral client-secret minting. WebRTC
# media capture and SIP telephony are OUT OF SCOPE (need a native media stack
# Julia lacks); audio is exchanged as base64 PCM inside events.
# ============================================================================

"Successful [`mint_realtime_secret`](@ref) result; `value` is the ephemeral client secret and `raw` the unparsed JSON response."
@kwdef struct RealtimeSecretSuccess <: LLMRequestResponse
    value::String
    raw::Dict{String,Any} = Dict{String,Any}()
end
"Realtime API error result: HTTP `status` and the raw `response` body."
@kwdef struct RealtimeFailure <: LLMRequestResponse; response::String; status::Int; end
"Local/transport error from a Realtime API call (the request never completed)."
@kwdef struct RealtimeCallError <: LLMRequestResponse; error::String; status::Union{Int,Nothing} = nothing; end

"""
    mint_realtime_secret(; session=nothing, service=OPENAIServiceEndpoint)

Create an ephemeral client secret for client-side Realtime connections
(`POST /v1/realtime/client_secrets`). `session` is an optional session-config dict.
Returns `RealtimeSecretSuccess` (`.value`), `RealtimeFailure`, or `RealtimeCallError`.

Pass `config::Union{Nothing,RequestConfig}` to override the timeout budget for this call (a single bounded attempt; `max_attempts` does not apply).
"""
function mint_realtime_secret(; session::Union{AbstractDict,Nothing}=nothing, service::ServiceEndpointSpec=OPENAIServiceEndpoint, config::Union{Nothing,RequestConfig}=nothing)
    validate_capability(service, :realtime, "Realtime API")
    cfg = _resolve_config(config); t0 = time_ns()
    try
        d = Dict{Symbol,Any}()
        !isnothing(session) && (d[:session] = session)
        resp = _http("POST", _api_base_url(service) * REALTIME_CLIENT_SECRETS_PATH, auth_header(service),
            JSON.json(d); cfg, remaining=_remaining_s(cfg, t0))
        resp.status == 200 || return RealtimeFailure(response=String(resp.body), status=resp.status)
        data = JSON.parse(resp.body; dicttype=Dict{String,Any})
        cs = get(data, "client_secret", nothing)
        val = get(data, "value", cs isa AbstractDict ? get(cs, "value", nothing) : nothing)
        # A 200 that carries no usable secret is not a success: handing back an
        # empty value would fail later, at connect time, far from the cause.
        val isa AbstractString && !isempty(val) ||
            return RealtimeCallError(error="client-secret response carried no secret value")
        RealtimeSecretSuccess(value=val, raw=data)
    catch e
        e isa InterruptException && rethrow()
        RealtimeCallError(error=_error_text(e), status=(hasproperty(e, :status) ? e.status : nothing))
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

"""A live Realtime WebSocket session. Created by [`realtime_connect`](@ref), which resolves a [`RequestConfig`](@ref) and captures it on `config` — [`realtime_receive`](@ref) reads its `stream_idle_timeout` from there. The two-argument constructor inherits the ambient config."""
mutable struct RealtimeSession
    ws::Any; model::String
    config::RequestConfig   # connect-time budget; bounds realtime_receive
end

# The Realtime WebSocket is served by api.openai.com, so only OPENAIServiceEndpoint
# has a socket URL: opening it for any other endpoint would send that endpoint's
# credentials to OpenAI (mint_realtime_secret, by contrast, honours the endpoint).
# A function so tests can route their own endpoint types to a local server.
_realtime_ws_url(::Type{OPENAIServiceEndpoint}) = REALTIME_WS_URL
_realtime_ws_url(service) = throw(ArgumentError(
    "realtime_connect supports only OPENAIServiceEndpoint (got " *
    "$(service isa Type ? service : typeof(service))): the Realtime WebSocket is " *
    "served by api.openai.com, and another endpoint's credentials must not be sent there"))

# Test seam: runs as the upgrade completes, before the handler is admitted.
_realtime_upgraded(service) = nothing

# Open-phase resolution, exactly once: :pending → :open (the handler runs) |
# :abandoned (the caller already got UniLMTimeout(:connect); a late upgrade is
# closed unused).
mutable struct _RealtimeGate
    @atomic state::Symbol
end

# Source-compatible with the pre-config shape: without an explicit budget a
# session inherits the ambient one.
RealtimeSession(ws, model::String) = RealtimeSession(ws, model, current_config())

# Native handshake bounds. `connect_timeout` bounds TCP/TLS, and the same value
# caps the wait for the upgrade response headers — a handshake-phase bound that
# cannot outlive the upgrade, so an abandoned handshake releases its socket
# instead of holding it forever. No native read-idle bound is armed:
# `realtime_receive` owns idle enforcement, so a breach surfaces as UniLMTimeout.
_realtime_native_kwargs(cfg::RequestConfig) =
    (connect_timeout = _native_seconds_real(cfg.connect_timeout),
     response_header_timeout = _native_seconds_real(cfg.connect_timeout))

# Split out of `realtime_connect` so the target the WebSocket opens is assertable
# without a live upgrade.
_realtime_url(service, model::String)::String =
    _realtime_ws_url(service) * "?model=" * _uripart(model)

"""
    realtime_connect(handler; model="gpt-realtime-2", service=OPENAIServiceEndpoint, config=nothing)

Open a Realtime WebSocket and run `handler(session::RealtimeSession)`. Inside the handler use
[`realtime_send`](@ref) to send event dicts and [`realtime_receive`](@ref) to read server
events. The socket closes when `handler` returns.

Pass `config::Union{Nothing,RequestConfig}` to override the timeout budget for this
session. ONLY the open phase is bounded, by `connect_timeout`: a peer that accepts
the TCP connection but never completes the upgrade throws
`UniLMTimeout(:connect, …)` instead of blocking forever. Once `handler` is running
the session's lifetime is the caller's to decide, so no bound applies to it. The
resolved config travels on the session, so [`realtime_receive`](@ref) inherits
`stream_idle_timeout`. `handler` runs on an internal task (that is what makes the
open phase separable); dynamically scoped values propagate into it. An upgrade
that completes only after the caller received the timeout is closed without
calling `handler`.

Throws `ArgumentError` before any I/O for a `service` other than
`OPENAIServiceEndpoint`: the Realtime WebSocket is served by api.openai.com, and
another endpoint's credentials must not be sent there.
"""
function realtime_connect(handler; model::String="gpt-realtime-2",
                          service::ServiceEndpointSpec=OPENAIServiceEndpoint,
                          config::Union{Nothing,RequestConfig}=nothing)
    validate_capability(service, :realtime, "Realtime API")
    cfg = _resolve_config(config)
    url = _realtime_url(service, model)
    t0 = time_ns()
    gate = _RealtimeGate(:pending)
    ready = Base.Event()   # the handler was admitted, the open task ended, or the bound fired
    session = Threads.@spawn try
        # auth_header_multipart drops the JSON Content-Type, which is meaningless on a WS upgrade.
        HTTP.WebSockets.open(url; headers=auth_header_multipart(service),
                             _realtime_native_kwargs(cfg)...) do ws
            _realtime_upgraded(service)
            (@atomicreplace gate.state :pending => :open).success || return close(ws)
            notify(ready)
            handler(RealtimeSession(ws, model, cfg))
        end
    finally
        notify(ready)
    end
    # Wait for the handshake only. On breach the worker is abandoned rather than
    # killed — the same policy as the HTTP seam's task-mode watchdog, and safe for
    # the same reason: the native bounds above end it on its own.
    if cfg.connect_timeout < Inf
        timer = Timer(cfg.connect_timeout; spawn=true) do _
            (@atomicreplace gate.state :pending => :abandoned).success && notify(ready)
        end
        try
            wait(ready)
        finally
            errormonitor(Threads.@spawn close(timer))   # off-path: see _with_deadline_task
        end
        (@atomic gate.state) === :abandoned &&
            throw(UniLMTimeout(:connect, _elapsed_s(t0), cfg.connect_timeout))
    end
    try
        return fetch(session)
    catch e
        e isa InterruptException && rethrow()
        err = _unwrap_task_failure(e)
        # A typed timeout raised inside the handler is the contract: surface it
        # over any wrapper layer that carries it (chain walk).
        typed = _find_exception(x -> x isa UniLMTimeout, err)
        typed !== nothing && throw(typed)
        # The native bounds above race the watchdog at the same limit, and only
        # handshake timers are armed, so a native timeout raised before the handler
        # was entered IS the open phase breaching: report it as the same typed error.
        (@atomic gate.state) !== :open && _find_exception(x -> x isa HTTP.TimeoutError, err) !== nothing &&
            throw(UniLMTimeout(:connect, _elapsed_s(t0), cfg.connect_timeout))
        throw(err)
    end
end

"""    realtime_send(session, event::AbstractDict)"""
realtime_send(s::RealtimeSession, event::AbstractDict) = HTTP.WebSockets.send(s.ws, JSON.json(event))

"""
    realtime_receive(session) -> Dict

Block for the next server event, bounded by the session's `stream_idle_timeout`.
A parked read cannot be polled out of, so a breach closes the socket — the
universal unblocker — and throws `UniLMTimeout(:stream_idle, …)`.
"""
function realtime_receive(s::RealtimeSession)
    limit = s.config.stream_idle_timeout
    t0 = time_ns()
    raw, fired = _with_deadline_reported(() -> HTTP.WebSockets.receive(s.ws),
                                         () -> close(s.ws), limit, :stream_idle)
    # A fired guard means our own close is what ended the read: anything it handed
    # back is the echo of that close on a dead socket, never a server event.
    fired && throw(UniLMTimeout(:stream_idle, _elapsed_s(t0), limit))
    return JSON.parse(String(raw); dicttype=Dict{String,Any})
end

# A minted client secret is a live credential: the default field dump prints it
# twice (`value` and again inside `raw`), so a REPL echo or a log line hands it
# out. Redact the value and report `raw` by size only — the parsed payload stays
# reachable programmatically, it just stops printing itself.
function Base.show(io::IO, r::RealtimeSecretSuccess)
    print(io, "RealtimeSecretSuccess(value=")
    show(io, _redact_api_key(r.value))
    print(io, ", raw=<", length(r.raw), " keys>)")
end
