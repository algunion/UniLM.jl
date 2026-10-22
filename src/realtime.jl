# ============================================================================
# OpenAI Realtime API — low-latency speech/text over a WebSocket. This client
# covers the WebSocket event transport + ephemeral client-secret minting, and
# drives whole sessions for `speak` (a voice session) and `translate` (a
# translation session). WebRTC media capture and SIP telephony are OUT OF SCOPE
# (need a native media stack Julia lacks); audio is exchanged as base64 PCM
# inside events.
# ============================================================================

"Successful [`mint_realtime_secret`](@ref) result; `value` is the ephemeral client secret and `raw` the unparsed JSON response."
@kwdef struct RealtimeSecretSuccess <: LLMRequestResponse
    value::String
    raw::Dict{String,Any} = Dict{String,Any}()
end
"Realtime API error result: HTTP `status` and the raw `response` body."
@kwdef struct RealtimeFailure <: LLMRequestResponse; response::String; status::Int; end
"Realtime API call that produced no usable reply (transport failure, timeout, cancel, or a 200 without a secret); `cause` is the underlying exception — a [`UniLMTimeout`](@ref) for a timeout, a [`UniLMCancelled`](@ref) for a cancel — and `request_id` is the id the service sent with a reply that could not be used."
@kwdef struct RealtimeCallError <: LLMRequestResponse; error::String; status::Union{Int,Nothing} = nothing; request_id::Union{String,Nothing} = nothing; cause::Union{Nothing,Exception} = nothing; end

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
    local resp
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
            return RealtimeCallError(error="client-secret response carried no secret value",
                                     request_id=_platform_request_id(resp))
        RealtimeSecretSuccess(value=val, raw=data)
    catch e
        e isa InterruptException && rethrow()
        _callerr(RealtimeCallError, e; request_id=_platform_request_id(@isdefined(resp) ? resp : nothing))
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
    "the Realtime WebSocket (realtime_connect, speak, translate) supports only OPENAIServiceEndpoint " *
    "(got $(service isa Type ? service : typeof(service))): it is served by api.openai.com, and " *
    "another endpoint's credentials must not be sent there"))

# Test seam: runs as the upgrade completes, before the handler is admitted.
_realtime_upgraded(service) = nothing

# Open-phase resolution, exactly once: :pending → :open (the handler runs) | :failed
# (the open task ended first: its error is the outcome) | :abandoned (the bound fired
# first: the caller gets UniLMTimeout(:connect), and a late upgrade is closed unused).
# The open task records its end BEFORE it wakes the caller, so the first event decides:
# a bound that fires while the caller is still being scheduled cannot turn an earlier
# failure into a timeout.
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

# The translation endpoint sits under the Realtime WebSocket
# (wss://api.openai.com/v1/realtime/translations), the model again a query value:
# https://developers.openai.com/api/docs/guides/realtime-translation
_realtime_translation_url(service, model::String)::String =
    _realtime_ws_url(service) * "/translations?model=" * _uripart(model)

"""
    realtime_connect(handler; model="gpt-realtime-2.1", service=OPENAIServiceEndpoint, config=nothing)

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
function realtime_connect(handler; model::String="gpt-realtime-2.1",
                          service::ServiceEndpointSpec=OPENAIServiceEndpoint,
                          config::Union{Nothing,RequestConfig}=nothing)
    validate_capability(service, :realtime, "Realtime API")
    cfg = _resolve_config(config)
    _realtime_open(handler, _realtime_url(service, model), model, service, cfg)
end

# The open phase every Realtime session shares: `handler(session)` runs once the upgrade of `url`
# completes inside `cfg.connect_timeout`; returns what the handler returns.
function _realtime_open(handler, url::String, model::String, service, cfg::RequestConfig)
    t0 = time_ns()
    gate = _RealtimeGate(:pending)
    ready = Base.Event()   # the handler was admitted, the open task ended, or the bound fired
    session = Threads.@spawn :default try
        # auth_header_multipart drops the JSON Content-Type, which is meaningless on a WS upgrade.
        HTTP.WebSockets.open(url; headers=auth_header_multipart(service),
                             _realtime_native_kwargs(cfg)...) do ws
            _realtime_upgraded(service)
            (@atomicreplace gate.state :pending => :open).success || return close(ws)
            notify(ready)
            handler(RealtimeSession(ws, model, cfg))
        end
    finally
        @atomicreplace gate.state :pending => :failed
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
            errormonitor(Threads.@spawn :default close(timer))   # off-path: see _with_deadline_task
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

# ─── Sessions UniLM drives to their end: speak and translate ─────────────────
# Unlike a `realtime_connect` session, these are one bounded call: the caller's wait ends at
# total_deadline, or at once on a cancel, whatever the socket does; every send and read of the
# session waits at most stream_idle_timeout; and every outcome is a typed result.

# Ends a session's connection. HTTP.jl's close(ws) first takes the send lock, which a send parked on
# a full socket buffer holds, so only closing the connection itself unblocks that send (HTTP.jl
# 2.11.2 has no public call for it).
_realtime_drop!(ws) = ws.close_transport!()

# What the caller of a driven session shares with the task that runs it: the session's socket once
# it is open, and whether the caller stopped waiting.
mutable struct _RealtimeCall
    @atomic ws::Any
    @atomic abandoned::Bool
end

function _realtime_abandon!(call::_RealtimeCall)::Nothing
    @atomic call.abandoned = true
    ws = @atomic call.ws
    ws === nothing || _realtime_drop!(ws)
    nothing
end

# HTTP.jl reports an upgrade the server answered with another status only as this close reason, and
# does not keep the response: the provider's refusal (a bad key is a 401), with that status.
function _realtime_refused_upgrade(e)::Union{AudioFailure,Nothing}
    (e isa HTTP.WebSockets.WebSocketError && e.message.code == 1002) || return nothing
    m = match(r"^websocket handshake failed: status (\d{3})$", e.message.reason)
    m === nothing && return nothing
    AudioFailure(response=e.message.reason, status=parse(Int, something(m[1])))
end

# Runs `driver(session)` on a session it opens, on a worker task (the task-mode guard of
# deadline.jl), so the caller's wait ends at total_deadline, or at once on a cancel of the ambient
# token; then the session's connection is dropped so the abandoned task ends too. A token already
# cancelled sends nothing. The open phase is bounded by connect_timeout and by what is left of
# total_deadline. Returns what the driver returns.
function _realtime_call(driver, url::String, model::String, service, cfg::RequestConfig, t0::UInt64)
    tok = _current_cancel()
    iscancelled(tok) && throw(UniLMCancelled(:token, _elapsed_s(t0)))
    left = _remaining_s(cfg, t0)
    ispositive(left) || throw(UniLMTimeout(:deadline, _elapsed_s(t0), cfg.total_deadline))
    capped = left < cfg.connect_timeout
    open_cfg = capped ? RequestConfig(cfg; connect_timeout=left) : cfg
    call = _RealtimeCall(nothing, false)
    admitted = session -> begin
        @atomic call.ws = session.ws
        # A caller that gave up before the socket was recorded could not drop it: it closes unused.
        (@atomic call.abandoned) ? nothing : driver(session)
    end
    try
        # Finite even without a deadline, so the wait always runs on a worker and a cancel ends it.
        _with_deadline_task(min(left, _MAX_FINITE_TIMEOUT), :deadline; cancel=tok,
                            on_cancel=() -> _realtime_abandon!(call)) do
            _realtime_open(admitted, url, model, service, open_cfg)
        end
    catch e
        _realtime_abandon!(call)
        e isa InterruptException && rethrow()
        refused = _realtime_refused_upgrade(e)
        refused === nothing || return refused
        (e isa UniLMTimeout && (e.phase === :deadline || (capped && e.phase === :connect))) &&
            throw(UniLMTimeout(:deadline, _elapsed_s(t0), cfg.total_deadline))
        rethrow()
    end
end

# The next server event of a driven session, waiting at most stream_idle_timeout: a breach drops the
# connection, which unblocks the read, and throws the typed timeout. `nothing` when the peer closed.
function _realtime_next(s::RealtimeSession)::Union{Dict{String,Any},Nothing}
    limit = s.config.stream_idle_timeout
    r0 = time_ns()
    raw, fired = try
        _with_deadline_reported(() -> HTTP.WebSockets.receive(s.ws), () -> _realtime_drop!(s.ws), limit, :stream_idle)
    catch e
        e isa HTTP.WebSockets.WebSocketError && return nothing
        rethrow()
    end
    # Our own drop ended the read: what it handed back is not a server event.
    fired && throw(UniLMTimeout(:stream_idle, _elapsed_s(r0), limit))
    JSON.parse(String(raw); dicttype=Dict{String,Any})
end

# Sends one client event of a driven session, waiting at most stream_idle_timeout: a peer that stops
# reading cannot park the session on a full socket buffer. `false` when the peer closed the session.
function _realtime_post(s::RealtimeSession, event::AbstractDict)::Bool
    limit = s.config.stream_idle_timeout
    r0 = time_ns()
    fired = try
        last(_with_deadline_reported(() -> realtime_send(s, event), () -> _realtime_drop!(s.ws), limit, :stream_idle))
    catch e
        e isa HTTP.WebSockets.WebSocketError && return false
        rethrow()
    end
    fired && throw(UniLMTimeout(:stream_idle, _elapsed_s(r0), limit))
    true
end

# Reads events until one of `type`. An `error` event on the way is the provider's refusal; a close
# is a session that ended early.
function _realtime_await(s::RealtimeSession, type::String)::Union{Dict{String,Any},AudioFailure,AudioCallError}
    while true
        ev = _realtime_next(s)
        ev === nothing && return _realtime_closed(type)
        kind = get(ev, "type", nothing)
        kind == "error" && return _realtime_refusal(ev)
        kind == type && return ev
    end
end

# A failure the provider reported as an event: there is no HTTP status, and the event is the response.
_realtime_refusal(ev::AbstractDict)::AudioFailure = AudioFailure(response=JSON.json(ev), status=nothing)
_realtime_closed(awaited::String)::AudioCallError = AudioCallError(error="the Realtime session closed before $awaited")

# A string field of a server event; anything else is a malformed event.
function _event_string(ev::AbstractDict, key::String)::String
    v = get(ev, key, nothing)
    v isa AbstractString || error("Realtime $(get(ev, "type", "")) event without a string `$key`")
    String(v)
end

# Realtime models are conversational, so a speech session's instructions tell the model to read the
# text rather than answer it; the caller's `instructions` follow as delivery directions.
const _REALTIME_READ_ALOUD = "Read the user's message aloud, word for word, exactly as written. " *
                             "Do not answer it or comment on it, and do not add, drop or change words."

# session.update for speech: audio output as 24 kHz PCM16, in the request's voice and speed.
function _speech_session(req::SpeechRequest)::Dict{Symbol,Any}
    output = Dict{Symbol,Any}(:format => Dict(:type => "audio/pcm", :rate => _SPEECH_RATE), :voice => req.voice)
    isnothing(req.speed) || (output[:speed] = req.speed)
    instructions = isnothing(req.instructions) ? _REALTIME_READ_ALOUD :
                   string(_REALTIME_READ_ALOUD, " Delivery: ", req.instructions)
    session_update(Dict{Symbol,Any}(:type => "realtime", :output_modalities => ["audio"],
                                    :instructions => instructions, :audio => Dict(:output => output)))
end

_speech_item(req::SpeechRequest)::Dict{Symbol,Any} = realtime_event("conversation.item.create"; item=Dict(
    :type => "message", :role => "user", :content => [Dict(:type => "input_text", :text => req.input)]))

# One speech response (https://developers.openai.com/api/docs/guides/realtime-conversations): update
# the session and wait until it took the update, add the text, ask for a response, and collect its
# audio deltas until response.done.
function _realtime_speak(s::RealtimeSession, req::SpeechRequest)::LLMRequestResponse
    _realtime_post(s, _speech_session(req)) || return _realtime_closed("session.updated")
    (ev = _realtime_await(s, "session.updated")) isa AbstractDict || return ev
    (_realtime_post(s, _speech_item(req)) && _realtime_post(s, response_create())) ||
        return _realtime_closed("response.done")
    pcm = IOBuffer()
    while true
        ev = _realtime_next(s)
        ev === nothing && return _realtime_closed("response.done")
        kind = get(ev, "type", nothing)
        if kind == "error"
            return _realtime_refusal(ev)
        elseif kind == "response.output_audio.delta"
            write(pcm, base64decode(_event_string(ev, "delta")))
        elseif kind == "response.done"
            return _speech_done(req, ev, take!(pcm))
        end
    end
end

# Only a completed response is the whole speech: a cancelled, failed or incomplete one carries its
# reason in `status_details` and is the provider's failure.
function _speech_done(req::SpeechRequest, ev::AbstractDict, pcm::Vector{UInt8})::LLMRequestResponse
    response = get(ev, "response", nothing)
    status = response isa AbstractDict ? get(response, "status", nothing) : nothing
    status == "completed" || return _realtime_refusal(ev)
    isempty(pcm) && return AudioCallError(error="the Realtime response completed without audio")
    _speech_success(req, pcm)
end

# A translation session consumes 200 ms engine frames; 200 ms of 24 kHz PCM16 mono is 9,600 bytes.
const _TRANSLATION_FRAME_BYTES = 9_600

# One translation (https://developers.openai.com/api/docs/guides/realtime-translation): set the
# target language and wait until the session took it, stream the audio, close the session, and join
# the translated transcript until session.closed. The translated audio deltas are not kept.
function _realtime_translate(s::RealtimeSession, pcm::Vector{UInt8}, language::String)::LLMRequestResponse
    update = session_update(Dict{Symbol,Any}(:audio => Dict(:output => Dict(:language => language))))
    _realtime_post(s, update) || return _realtime_closed("session.updated")
    (ev = _realtime_await(s, "session.updated")) isa AbstractDict || return ev
    for frame in Iterators.partition(pcm, _TRANSLATION_FRAME_BYTES)
        _realtime_post(s, realtime_event("session.input_audio_buffer.append"; audio=base64encode(frame))) ||
            return _realtime_closed("session.closed")
    end
    _realtime_post(s, realtime_event("session.close")) || return _realtime_closed("session.closed")
    text = IOBuffer()
    while true
        ev = _realtime_next(s)
        ev === nothing && return _realtime_closed("session.closed")
        kind = get(ev, "type", nothing)
        if kind == "error"
            return _realtime_refusal(ev)
        elseif kind == "session.output_transcript.delta"
            print(text, _event_string(ev, "delta"))
        elseif kind == "session.closed"
            return TranscriptionSuccess(text=takestring!(text))
        end
    end
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
