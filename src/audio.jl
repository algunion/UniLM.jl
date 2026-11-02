# ============================================================================
# Audio — speech (OpenAI Realtime sessions, Gemini TTS, Ollama), transcription
# (OpenAI Audio API, Ollama) and translation (OpenAI Realtime translation
# sessions, Ollama). The Realtime session drivers live in realtime.jl.
# ============================================================================

"Audio error result: the HTTP `status` (`nothing` when a Realtime session reported the failure as an event, which carries no HTTP status), the raw `response` (the reply body, the Realtime event's JSON, or for a refused WebSocket upgrade HTTP.jl's report of its status) and the `request_id` the service sent (`x-request-id`/`request-id` header), if any."
@kwdef struct AudioFailure <: LLMRequestResponse; response::String; status::Union{Int,Nothing}; request_id::Union{String,Nothing} = nothing; end
"Audio call that produced no usable reply (transport failure, timeout, cancel, a session that closed before its result, or a reply that could not be decoded); `cause` is the underlying exception — a [`UniLMTimeout`](@ref) for a timeout, a [`UniLMCancelled`](@ref) for a cancel, and, when the server closed a Realtime session, the `HTTP.WebSockets.WebSocketError` that carries its status and reason, which `error` names too."
@kwdef struct AudioCallError <: LLMRequestResponse; error::String; status::Union{Int,Nothing} = nothing; cause::Union{Nothing,Exception} = nothing; end

# ─── Speech ──────────────────────────────────────────────────────────────────

# Speech from every service is 16-bit little-endian mono PCM at 24 kHz: the one rate of the
# Realtime API's PCM output, the rate UniLM asks Gemini TTS for, and the one rate Ollama speaks at.
const _SPEECH_RATE = 24_000
const _SPEECH_FORMATS = ("wav", "pcm")

# How a service speaks: an OpenAI Realtime session (every service but Gemini's and Ollama's;
# `_realtime_ws_url` refuses all but OPENAIServiceEndpoint), Gemini TTS through generateContent, or
# Ollama's speech route.
abstract type _SpeechWire end
struct _RealtimeSpeechWire <: _SpeechWire end
struct _GeminiSpeechWire <: _SpeechWire end
struct _OllamaSpeechWire <: _SpeechWire end
_speech_wire(_) = _RealtimeSpeechWire()
_speech_wire(::Type{GEMINIServiceEndpoint}) = _GeminiSpeechWire()
_speech_wire(::OllamaEndpoint) = _OllamaSpeechWire()
# Google's OpenAI-compatible API documents no speech route (https://ai.google.dev/gemini-api/docs/openai):
# Gemini speaks through its native API only.
_speech_wire(::Type{GEMINIOpenAIServiceEndpoint}) = throw(ArgumentError(
    "GEMINIOpenAIServiceEndpoint has no speech route (Google's OpenAI-compatible API documents none): " *
    "Gemini TTS is on the native Gemini API, so pass service=GEMINIServiceEndpoint " *
    "(https://ai.google.dev/gemini-api/docs/speech-generation)"))

# What a speech wire accepts, from its provider's docs: the default model and voice, the voices,
# and the speed range (`nothing`: no speed control).
struct _SpeechRules
    provider::String
    model::String
    voice::String
    voices::Vector{String}
    speed::Union{Nothing,Tuple{Float64,Float64}}
    docs::String
end

# The Realtime session's built-in voices and output speed (session.update `audio.output`), and the
# voice a session starts with (`marin`, in session.created):
# https://developers.openai.com/api/reference/resources/realtime/client-events
const _REALTIME_SPEECH = _SpeechRules("OpenAI Realtime", "gpt-realtime-2.1-mini", "marin",
    ["alloy", "ash", "ballad", "coral", "echo", "sage", "shimmer", "verse", "marin", "cedar"], (0.25, 1.5),
    "https://developers.openai.com/api/reference/resources/realtime/client-events")
# Gemini 3.8 TTS's 30 prebuilt studio voices; it takes other voices too (`_speech_voice`). It has no
# speed parameter: the pace is part of a text's speech style.
# https://ai.google.dev/gemini-api/docs/speech-generation
const _GEMINI_SPEECH = _SpeechRules("Gemini TTS", "gemini-3.8-flash-tts", "Kore",
    ["Zephyr", "Puck", "Charon", "Kore", "Fenrir", "Leda", "Orus", "Aoede", "Callirrhoe", "Autonoe",
     "Enceladus", "Iapetus", "Umbriel", "Algieba", "Despina", "Erinome", "Algenib", "Rasalgethi",
     "Laomedeia", "Achernar", "Alnilam", "Schedar", "Gacrux", "Pulcherrima", "Achird", "Zubenelgenubi",
     "Vindemiatrix", "Sadachbia", "Sadaltager", "Sulafat"], nothing,
    "https://ai.google.dev/gemini-api/docs/speech-generation")
# Ollama's speech route (Ollama 0.40.9 or later): no default model, each model's own voices (none named: the
# model's own voice, `voice` not sent) and speed 1 only (https://docs.ollama.com/api/openai-compatibility).
const _OLLAMA_SPEECH = _SpeechRules("Ollama", "", "", String[], (1.0, 1.0),
    "https://docs.ollama.com/api/openai-compatibility")
_speech_rules(::_RealtimeSpeechWire) = _REALTIME_SPEECH
_speech_rules(::_GeminiSpeechWire) = _GEMINI_SPEECH
_speech_rules(::_OllamaSpeechWire) = _OLLAMA_SPEECH

# The voice a request names, as its wire takes it: a Realtime session one of its built-in voices.
function _speech_voice(w::_RealtimeSpeechWire, v::AbstractString)::String
    r = _speech_rules(w)
    v in r.voices || throw(ArgumentError(
        "$(r.provider) voices are $(join(r.voices, ", ")) (got $(repr(v))): $(r.docs)"))
    String(v)
end
# Gemini 3.8 TTS takes "a prebuilt voice name, an Extended Voice Library ID, a custom Voice design ID
# (`voice_...`), or a Voice replication ID (`voice_...`, or optional stateless `voicekey_...`)"
# (https://ai.google.dev/gemini-api/docs/generate-content/speech-generation). The Extended Voice
# Library's hundreds of names are listed only by `GET /v1beta/voices`, so any voice is sent as given,
# and the API answers one it does not have; a blank one names no voice.
function _speech_voice(::_GeminiSpeechWire, v::AbstractString)::String
    isempty(strip(v)) && throw(ArgumentError(
        "Gemini TTS takes a voice by name or id: a prebuilt voice such as \"Kore\" (the Extended Voice " *
        "Library's among them), a designed or replicated voice's voice_… id, or a replicated voice's voicekey_… key " *
        "(got $(repr(v))): https://ai.google.dev/gemini-api/docs/speech-generation#voices"))
    String(v)
end
# Ollama's voices are each model's own, as `model_info` lists them (`voices`): a voice is sent as given, and Ollama
# answers one the model does not have with HTTP 400 naming the voices it has; none (`""`) leaves the model's own
# voice (https://docs.ollama.com/capabilities/speech).
function _speech_voice(::_OllamaSpeechWire, v::AbstractString)::String
    isempty(v) || !isempty(strip(v)) || throw(ArgumentError(
        "Ollama takes a voice by name, one of those model_info lists for the model (got $(repr(v))): leave voice " *
        "unset for the model's own voice"))
    String(v)
end

# The model a request names, else its wire's default; Ollama has none.
_speech_model(w::_SpeechWire, model::Union{AbstractString,Nothing})::String = String(something(model, _speech_rules(w).model))
function _speech_model(::_OllamaSpeechWire, model::Union{AbstractString,Nothing})::String
    (isnothing(model) || isempty(strip(model))) && throw(ArgumentError(
        "Ollama has no default speech model: name an installed one, e.g. model = \"phonon\" (`ollama pull phonon`)"))
    String(model)
end

# The refusal of a format no speech wire produces; Ollama's in its own words (server/speech.go), except an empty
# format, which Ollama reads as wav: that one is refused by UniLM's own rule.
_speech_format_rule(fmt::AbstractString)::String =
    "response_format must be \"wav\" (a WAV file) or \"pcm\" (raw 16-bit mono PCM at 24 kHz); got $(repr(fmt))"
_speech_format_refusal(::_SpeechWire, fmt::AbstractString)::String = _speech_format_rule(fmt)
_speech_format_refusal(::_OllamaSpeechWire, fmt::AbstractString)::String = isempty(fmt) ? _speech_format_rule(fmt) :
    "Ollama answers HTTP 400 \"response_format $(repr(fmt)) is not supported, use wav or pcm\": set response_format " *
    "to \"wav\" (a WAV file) or \"pcm\" (raw 16-bit mono PCM at 24 kHz)"

# What a speech wire cannot carry beyond its voices, formats and speeds, refused before any request.
_check_speech(::_SpeechWire, ::AbstractString, ::Union{AbstractString,Nothing}) = nothing
# Ollama reads "up to 4,096 characters" and answers non-empty `instructions` with HTTP 400 "instructions are not
# supported" (https://docs.ollama.com/api/openai-compatibility, server/speech.go); empty ones are none, and UniLM
# never sends them.
const _OLLAMA_SPEECH_CHARS = 4096
function _check_speech(::_OllamaSpeechWire, input::AbstractString, instructions::Union{AbstractString,Nothing})
    n = length(input)
    n <= _OLLAMA_SPEECH_CHARS || throw(ArgumentError(
        "Ollama reads at most 4,096 characters per speech request (got $n): split the text, e.g. at paragraph or " *
        "sentence breaks, and speak each part"))
    isnothing(instructions) || isempty(instructions) || throw(ArgumentError(
        "Ollama answers `instructions` with HTTP 400 \"instructions are not supported\": remove instructions"))
    nothing
end

function _speech_speed(r::_SpeechRules, speed::Real)::Float64
    s = Float64(speed)
    range = r.speed
    range === nothing && throw(ArgumentError(
        "$(r.provider) has no speed control: describe the pace in `instructions` instead, e.g. " *
        "instructions = \"speaking slowly\" ($(r.docs))"))
    first(range) == last(range) && s != first(range) && throw(ArgumentError(
        "$(r.provider) speaks at speed $(first(range)) only (got $s): leave speed unset ($(r.docs))"))
    first(range) <= s <= last(range) || throw(ArgumentError(
        "$(r.provider) speed must be between $(first(range)) and $(last(range)) (got $s)"))
    s
end

"""
    SpeechRequest(; input, service=OPENAIServiceEndpoint, model=nothing, voice=nothing,
                  response_format="wav", speed=nothing, instructions=nothing)

A text-to-speech request; pass it to [`speak`](@ref). `input` is the text to read aloud. `model`
and `voice` left `nothing` are the service's defaults:

| `service` | speaks through | default `model` | default `voice` | `voice` | `speed` |
|:--|:--|:--|:--|:--|:--|
| `OPENAIServiceEndpoint` | a Realtime API session | `gpt-realtime-2.1-mini` | `marin` | `alloy`, `ash`, `ballad`, `coral`, `echo`, `sage`, `shimmer`, `verse`, `marin`, `cedar` | `0.25` to `1.5` |
| `GEMINIServiceEndpoint` | Gemini TTS (`generateContent`) | `gemini-3.8-flash-tts` | `Kore` | a prebuilt voice (the 30 studio voices or one of the Extended Voice Library), a designed or replicated voice's `voice_…` id, or a replicated voice's `voicekey_…` key | none |
| [`OllamaEndpoint`](@ref) | Ollama's speech route (Ollama 0.40.9 or later) | none: name an installed speech model, such as `phonon` | the model's own (`voice` not sent) | one of the model's voices, which [`model_info`](@ref) lists (`voices`), sent as given | `1` only |

Every service produces 16-bit mono PCM at 24 kHz: `response_format` `"wav"` returns it as a WAV file,
`"pcm"` as the raw samples. `instructions` direct the delivery (tone, pace, accent): on OpenAI they
follow the session's instruction to read `input` word for word, on Gemini they are the text's speech
style (`speechMetadata.style`). Gemini TTS has no speed parameter: describe the pace in
`instructions` (`"speaking slowly"`). Ollama takes neither: it answers non-empty `instructions` and
any speed but 1 with HTTP 400.

Throws `ArgumentError`, before any request, for an empty `input`, a voice the Realtime session does
not have or a blank Gemini or Ollama voice, a `response_format` other than `"wav"` or `"pcm"`, a
`speed` out of range, and any `speed` on Gemini; on Ollama, for a request without a `model`, an
`input` of more than 4,096 characters and non-empty `instructions`; and for `GEMINIOpenAIServiceEndpoint`,
which has no speech route, naming `GEMINIServiceEndpoint`. The Extended Voice Library's names are
listed only by Gemini's `GET /v1beta/voices`, so a Gemini voice is sent as given, and Gemini answers a
name it does not have; an Ollama voice is sent as given too, and Ollama answers one the model does not
have with HTTP 400 naming those it has.
Voices: [OpenAI Realtime](https://developers.openai.com/api/reference/resources/realtime/client-events),
[Gemini TTS](https://ai.google.dev/gemini-api/docs/speech-generation#voices),
[Ollama](https://docs.ollama.com/capabilities/speech).
"""
struct SpeechRequest
    service::ServiceEndpointSpec
    model::String
    input::String
    voice::String
    response_format::String
    speed::Union{Float64,Nothing}
    instructions::Union{String,Nothing}
    function SpeechRequest(service::ServiceEndpointSpec, model::Union{AbstractString,Nothing},
                           input::AbstractString, voice::Union{AbstractString,Nothing},
                           response_format::AbstractString, speed::Union{Real,Nothing},
                           instructions::Union{AbstractString,Nothing})
        wire = _speech_wire(service)
        rules = _speech_rules(wire)
        isempty(strip(input)) && throw(ArgumentError("speak needs text to read: input is empty"))
        v = _speech_voice(wire, something(voice, rules.voice))
        response_format in _SPEECH_FORMATS || throw(ArgumentError(_speech_format_refusal(wire, response_format)))
        _check_speech(wire, input, instructions)
        new(service, _speech_model(wire, model), input, v, response_format,
            isnothing(speed) ? nothing : _speech_speed(rules, speed), instructions)
    end
end
SpeechRequest(; input::AbstractString, service::ServiceEndpointSpec=OPENAIServiceEndpoint,
              model::Union{AbstractString,Nothing}=nothing, voice::Union{AbstractString,Nothing}=nothing,
              response_format::AbstractString="wav", speed::Union{Real,Nothing}=nothing,
              instructions::Union{AbstractString,Nothing}=nothing) =
    SpeechRequest(service, model, input, voice, response_format, speed, instructions)

"Successful [`speak`](@ref) result: `audio` holds a WAV file (`content_type` `\"audio/wav\"`) or the raw 16-bit little-endian mono samples at 24 kHz (`\"audio/pcm\"`). Save it with [`save_audio`](@ref)."
@kwdef struct SpeechSuccess <: LLMRequestResponse
    audio::Vector{UInt8}
    content_type::String = ""
end

"""
    speak(s::SpeechRequest; config=nothing) -> SpeechSuccess | AudioFailure | AudioCallError
    speak(input; service=OPENAIServiceEndpoint, model=nothing, voice=nothing, response_format="wav",
          speed=nothing, instructions=nothing, config=nothing)

Synthesize speech from text; the keywords build a [`SpeechRequest`](@ref).

On `OPENAIServiceEndpoint` the speech comes from a Realtime API session
(`wss://api.openai.com/v1/realtime`, `gpt-realtime-2.1-mini` unless `model` names another), the
replacement OpenAI names for its retiring text-to-speech models
([text to speech](https://developers.openai.com/api/docs/guides/text-to-speech)): UniLM sets the
voice, 24 kHz PCM output and instructions to read the text word for word, adds the text as a user
message, asks for a response and collects its audio. Realtime models are conversational, and OpenAI
notes that instructions are not guaranteed to be followed. On `GEMINIServiceEndpoint` it comes from
Gemini TTS (`generateContent` on `gemini-3.8-flash-tts` unless `model` names another,
[speech generation](https://ai.google.dev/gemini-api/docs/speech-generation)), which reads the text
verbatim. On [`OllamaEndpoint`](@ref) it comes from Ollama's OpenAI-compatible speech route
(Ollama 0.40.9 or later, [speech](https://docs.ollama.com/capabilities/speech)), one request
carrying the model, the text, the format, and the voice and speed when set; there is no default
model, so name an installed speech model, such as `phonon`:

```julia
r = speak("Hello from UniLM."; service=OllamaEndpoint(), model="phonon", voice="Zoey")
r isa SpeechSuccess && save_audio(r, "hello.wav")
```

Returns [`SpeechSuccess`](@ref): a WAV file (`response_format = "wav"`) or the raw samples (`"pcm"`).
Ollama's bytes come back as it streamed them: its WAV header, written before the length is known,
holds the maximum size in its size fields.
A failure the provider reports is an [`AudioFailure`](@ref): an HTTP error status, a WebSocket upgrade
the server refuses included (a bad key is a 401; `response` is then HTTP.jl's report of the status,
since it does not keep the reply); a Gemini reply whose candidate did not finish (a `finishReason`
other than `STOP`, such as `MAX_TOKENS` or `SAFETY`) or that has no candidate (a blocked prompt, its
reason in `promptFeedback`), with `status` 200 and the reply as `response`; or on a Realtime session
the `error` event, read even when the server closes the session while UniLM is still sending, or a
`response.done` that did not complete (`status` is then `nothing` and `response` the event's JSON). A
session that closes before its response is done (`error` names the close's status and reason, and
`cause` is the close), a timeout, a cancel and a reply without usable audio are an
[`AudioCallError`](@ref). Audio received before a failure is never returned.

Pass `config::Union{Nothing,RequestConfig}` to bound the call. On OpenAI the session is a single
attempt, which `max_attempts` does not change: a retry after the session's `response.create` may
bill a second response, and a session's audio cannot be resumed. There `total_deadline` bounds the
whole call whatever the socket does, `connect_timeout` the WebSocket open, and `stream_idle_timeout`
every send and every wait for the next event. On Gemini and Ollama, transient statuses
(408/429/500/502–504/520–524/529) and transport failures are retried with backoff under the
config's `max_attempts` and `total_deadline` (`retry-after-ms`, else `Retry-After`, honored), as
[`chatrequest!`](@ref) retries them, each attempt on Gemini a billed `generateContent` request; a 429
that no retry can serve is returned at once. All three observe the ambient [`CancelToken`](@ref)
([`with_cancel`](@ref)): a token already cancelled sends nothing, and a cancel ends the call at once,
with an `AudioCallError` whose `cause` is a [`UniLMCancelled`](@ref). Speech is not priced:
[`token_usage`](@ref) and [`estimated_cost`](@ref) throw for a `SpeechSuccess`.
"""
function speak(s::SpeechRequest; config::Union{Nothing,RequestConfig}=nothing,
               cancel::Union{Nothing,CancelToken}=nothing)
    wire = _speech_wire(s.service)
    target = _speech_target(wire, s)    # the route is checked before any I/O
    cfg = _resolve_config(config); t0 = time_ns()
    try
        _with_resolved_cancel(() -> _speak(wire, s, target, cfg, t0), cancel)
    catch e
        e isa InterruptException && rethrow()
        _callerr(AudioCallError, e)
    end
end
speak(input::AbstractString; service::ServiceEndpointSpec=OPENAIServiceEndpoint,
      model::Union{AbstractString,Nothing}=nothing, voice::Union{AbstractString,Nothing}=nothing,
      response_format::AbstractString="wav", speed::Union{Real,Nothing}=nothing,
      instructions::Union{AbstractString,Nothing}=nothing, config::Union{Nothing,RequestConfig}=nothing,
      cancel::Union{Nothing,CancelToken}=nothing) =
    speak(SpeechRequest(service, model, input, voice, response_format, speed, instructions); config, cancel)

# Where a request goes; throws ArgumentError, before any I/O, for a service the wire cannot reach.
function _speech_target(::_RealtimeSpeechWire, s::SpeechRequest)::String
    validate_capability(s.service, :speech, "speak (OpenAI Realtime API)")
    _realtime_url(s.service, s.model)
end
_speech_target(::_GeminiSpeechWire, s::SpeechRequest)::String = _gemini_speech_url(s.service, s.model)
function _speech_target(::_OllamaSpeechWire, s::SpeechRequest)::String
    validate_capability(s.service, :speech, "speak (Ollama speech route)")
    _api_base_url(s.service) * _OLLAMA_SPEECH_PATH
end

function _speak(::_RealtimeSpeechWire, s::SpeechRequest, url::String, cfg::RequestConfig, t0::UInt64)
    r = _realtime_call(session -> _realtime_speak(session, s), url, s.model, s.service, cfg, t0)
    r isa LLMRequestResponse ? r : _realtime_closed("response.done")
end

function _speak(::_GeminiSpeechWire, s::SpeechRequest, url::String, cfg::RequestConfig, t0::UInt64)
    resp = _http_with_retries(cfg, t0, "POST", url, auth_header(s.service), _gemini_speech_body(s))
    resp.status == 200 || return _failure(AudioFailure, resp)
    cand = _gemini_finished_candidate(JSON.parse(resp.body; dicttype=Dict{String,Any}))
    cand === nothing && return _failure(AudioFailure, resp)
    _speech_success(s, _gemini_speech_pcm(cand))
end

function _speak(::_OllamaSpeechWire, s::SpeechRequest, url::String, cfg::RequestConfig, t0::UInt64)
    resp = _http_with_retries(cfg, t0, "POST", url, auth_header(s.service), _ollama_speech_body(s))
    resp.status == 200 || return _failure(AudioFailure, resp)
    _ollama_speech(s, resp)
end

# Speech in the requested format: a canonical WAV file around the samples, or the samples.
_speech_success(s::SpeechRequest, pcm::Vector{UInt8})::SpeechSuccess =
    s.response_format == "wav" ? SpeechSuccess(audio=_wav_bytes(pcm), content_type="audio/wav") :
                                 SpeechSuccess(audio=_pcm16_checked(pcm), content_type="audio/pcm")

"""
    save_audio(r::SpeechSuccess, path) -> path

Write the synthesized audio bytes to `path`, atomically: the bytes go to a temporary
file in the same directory, which is then renamed over `path`, so a failed write leaves
any existing file intact. An existing file keeps its permission bits. A symlink is written
through — a dangling one creates its target; a directory throws `ArgumentError`.
"""
save_audio(r::SpeechSuccess, path::String) = _atomic_write(path, r.audio)

# ─── Ollama's speech route ───────────────────────────────────────────────────
# https://docs.ollama.com/api/openai-compatibility: the model, the text and the format, and the voice and speed
# when the request sets them.

const _OLLAMA_SPEECH_PATH = "/v1/audio/speech"   # Ollama's OpenAI-compatible speech route

function _ollama_speech_body(s::SpeechRequest)::String
    body = Dict{String,Any}("model" => s.model, "input" => s.input, "response_format" => s.response_format)
    isempty(s.voice) || (body["voice"] = s.voice)
    isnothing(s.speed) || (body["speed"] = s.speed)
    JSON.json(body)
end

# Ollama streams the audio as `audio/<format>`, 16-bit mono at 24 kHz: a WAV file whose header, written before the
# length is known, holds the maximum size, or the bare samples (https://docs.ollama.com/capabilities/speech). The
# bytes are returned as they came; a reply under another label, a WAV reply without its RIFF/WAVE header, or
# samples cut mid-frame, is a call error.
function _ollama_speech(s::SpeechRequest, resp::HTTP.Response)::SpeechSuccess
    kind = "audio/" * s.response_format
    label = HTTP.header(resp, "Content-Type", "")
    lowercase(strip(first(split(label, ';')))) == kind ||
        error("Ollama answered speech with Content-Type $(repr(label)), not $kind")
    audio = Vector{UInt8}(resp.body)
    if s.response_format == "wav"
        length(audio) >= 12 && view(audio, 1:4) == b"RIFF" && view(audio, 9:12) == b"WAVE" ||
            error("Ollama's speech reply is not a WAV file (no RIFF/WAVE header)")
    else
        _pcm16_checked(audio)
    end
    SpeechSuccess(audio=audio, content_type=kind)
end

# ─── Gemini TTS (generateContent) ────────────────────────────────────────────
# https://ai.google.dev/gemini-api/docs/speech-generation. The base is a function so tests can
# route their own endpoint types to a local server.

_gemini_speech_base(::Type{GEMINIServiceEndpoint}) = GEMINI_NATIVE_BASE
_gemini_speech_url(service, model::String)::String =
    _gemini_speech_base(service) * "/models/" * _uripart(model) * ":generateContent"

function _gemini_speech_body(s::SpeechRequest)::String
    part = Dict{Symbol,Any}(:text => s.input)
    # Gemini 3.8 TTS reads `text` verbatim; a delivery direction is the part's speech style.
    isnothing(s.instructions) || (part[:speechMetadata] = Dict(:style => s.instructions))
    # VoiceConfig holds `prebuiltVoiceConfig`, "The configuration for the prebuilt voice to use", or
    # `voice`, which takes a prebuilt voice's name, a stored voice's id (`voice_…`) and a replication key
    # (`voicekey_…`) (https://ai.google.dev/api/generate-content#VoiceConfig): a studio voice keeps the
    # prebuilt form, and every other voice goes as `voice`.
    voice = s.voice in _GEMINI_SPEECH.voices ? Dict(:prebuiltVoiceConfig => Dict(:voiceName => s.voice)) :
                                                Dict(:voice => s.voice)
    JSON.json(Dict(
        :contents => [Dict(:role => "user", :parts => [part])],
        :generationConfig => Dict(
            :responseModalities => ["AUDIO"],
            :speechConfig => Dict(:voiceConfig => voice),
            # Raw 16-bit PCM at 24 kHz: Gemini 3.8 TTS answers WAV unless asked, and UniLM writes
            # its own header.
            :responseFormat => Dict(:audio => Dict(:mimeType => "AUDIO_L16", :sampleRate => _SPEECH_RATE)))))
end

# The reply's candidate if it finished (finishReason STOP), else `nothing`: a candidate cut at
# MAX_TOKENS or stopped by a filter (SAFETY, OTHER, …), or a reply without one (a blocked prompt),
# holds no whole speech (https://ai.google.dev/api/generate-content#FinishReason).
function _gemini_finished_candidate(data)::Union{Dict{String,Any},Nothing}
    cands = data isa AbstractDict ? get(data, "candidates", nothing) : nothing
    (cands isa AbstractVector && !isempty(cands)) || return nothing
    cand = first(cands)
    (cand isa Dict{String,Any} && get(cand, "finishReason", nothing) == "STOP") ? cand : nothing
end

# The 16-bit PCM of a finished Gemini TTS candidate: its inline audio parts, in order. A candidate
# without audio (a text answer) throws.
function _gemini_speech_pcm(cand::Dict{String,Any})::Vector{UInt8}
    content = get(cand, "content", nothing)
    parts = content isa AbstractDict ? get(content, "parts", nothing) : nothing
    pcm = UInt8[]
    for part in (parts isa AbstractVector ? parts : Any[])
        blob = part isa AbstractDict ? get(part, "inlineData", nothing) : nothing
        blob isa AbstractDict || continue
        append!(pcm, _gemini_audio_pcm(string(get(blob, "mimeType", "")), base64decode(string(get(blob, "data", "")))))
    end
    isempty(pcm) && error("Gemini TTS reply carried no audio")
    pcm
end

# One inline audio part as 24 kHz mono 16-bit PCM: `audio/L16` (headerless little-endian PCM,
# 24 kHz mono unless its parameters say otherwise, per the speech-generation guide) or a WAV file.
function _gemini_audio_pcm(mime::String, bytes::Vector{UInt8})::Vector{UInt8}
    fields = split(mime, ';')
    kind = lowercase(strip(first(fields)))
    params = Dict(lowercase(strip(k)) => strip(v)
                  for (k, v) in (split(f, '='; limit=2) for f in fields[2:end] if occursin('=', f)))
    if kind == "audio/l16"
        rate, channels, pcm = parse(Int, get(params, "rate", "24000")), parse(Int, get(params, "channels", "1")), bytes
    elseif kind in ("audio/wav", "audio/wave", "audio/x-wav")
        w = _read_wav(bytes)
        rate, channels, pcm = w.rate, w.channels, _pcm16le(w.samples)
    else
        error("Gemini TTS returned $(repr(mime)) audio, not 16-bit PCM (audio/L16) or WAV")
    end
    (rate, channels) == (_SPEECH_RATE, 1) ||
        error("Gemini TTS returned $channels-channel audio at $rate Hz; UniLM asks for 24000 Hz mono")
    pcm
end

# ─── 16-bit PCM and WAV ──────────────────────────────────────────────────────

# 16-bit samples come in byte pairs: an odd count is a truncated stream.
function _pcm16_checked(pcm::Vector{UInt8})::Vector{UInt8}
    iseven(length(pcm)) || throw(ArgumentError("16-bit PCM audio has an even number of bytes (got $(length(pcm)))"))
    pcm
end

# 16-bit little-endian mono samples as a WAV file with the canonical 44-byte header: RIFF/WAVE, a
# 16-byte `fmt ` chunk (PCM, 1 channel, `rate` Hz, 16-bit) and one `data` chunk.
function _wav_bytes(pcm::Vector{UInt8}; rate::Int=_SPEECH_RATE)::Vector{UInt8}
    n = length(_pcm16_checked(pcm))
    n <= typemax(UInt32) - 36 || throw(ArgumentError("$n bytes of audio do not fit in a WAV file"))
    io = IOBuffer(; sizehint=44 + n)
    write(io, b"RIFF", htol(UInt32(36 + n)), b"WAVE",
          b"fmt ", htol(UInt32(16)), htol(UInt16(1)), htol(UInt16(1)), htol(UInt32(rate)),
          htol(UInt32(2rate)), htol(UInt16(2)), htol(UInt16(16)),
          b"data", htol(UInt32(n)), pcm)
    take!(io)
end

_le16(b::AbstractVector{UInt8}, i::Int)::UInt16 = UInt16(b[i]) | UInt16(b[i+1]) << 8
_le32(b::AbstractVector{UInt8}, i::Int)::UInt32 =
    UInt32(b[i]) | UInt32(b[i+1]) << 8 | UInt32(b[i+2]) << 16 | UInt32(b[i+3]) << 24

# WAV format codes other than PCM (1), named in refusals.
const _WAV_FORMATS = Dict{UInt16,String}(0x0002 => "ADPCM", 0x0003 => "32-bit float", 0x0006 => "A-law",
                                         0x0007 => "μ-law", 0x0011 => "IMA ADPCM", 0x0055 => "MP3")

# The interleaved samples of a 16-bit PCM WAV file. Walks the RIFF chunks, so chunks before `data`
# (LIST, FLLR, …) are skipped; reads PCM inside WAVE_FORMAT_EXTENSIBLE too; takes the bytes there are
# when a streamed file overstates its data size; drops an incomplete trailing frame. Anything that is
# not 16-bit PCM throws ArgumentError naming what it is.
function _read_wav(b::AbstractVector{UInt8})::@NamedTuple{samples::Vector{Int16}, rate::Int, channels::Int}
    n = length(b)
    (n >= 12 && view(b, 1:4) == b"RIFF" && view(b, 9:12) == b"WAVE") ||
        throw(ArgumentError("not a WAV file (no RIFF/WAVE header)"))
    fmt = nothing
    pos = 13
    while pos + 7 <= n
        id, size, body = view(b, pos:pos+3), Int(_le32(b, pos + 4)), pos + 8
        if id == b"fmt "
            (size >= 16 && body + 15 <= n) || throw(ArgumentError("WAV fmt chunk is truncated"))
            code = _le16(b, body)
            # WAVE_FORMAT_EXTENSIBLE names the encoding in the first two bytes of its SubFormat GUID.
            code == 0xfffe && size >= 40 && body + 25 <= n && (code = _le16(b, body + 24))
            fmt = (code, Int(_le16(b, body + 2)), Int(_le32(b, body + 4)), Int(_le16(b, body + 14)))
        elseif id == b"data"
            fmt === nothing && throw(ArgumentError("WAV data chunk comes before its fmt chunk"))
            code, channels, rate, bits = fmt
            code == 0x0001 || throw(ArgumentError(
                "WAV audio is $(get(_WAV_FORMATS, code, "format $(repr(code))")), not PCM"))
            bits == 16 || throw(ArgumentError("WAV audio is $bits-bit PCM, not 16-bit"))
            (channels >= 1 && rate >= 1) ||
                throw(ArgumentError("WAV fmt chunk declares $channels channels at $rate Hz"))
            frames = div(min(size, n - body + 1), 2channels)
            samples = Int16[reinterpret(Int16, _le16(b, body + 2k)) for k in 0:frames*channels-1]
            return (samples=samples, rate=rate, channels=channels)
        end
        pos = body + size + isodd(size)   # chunks are padded to an even size
    end
    throw(ArgumentError("WAV file has no data chunk"))
end

# One channel: the mean of each frame's samples.
_downmix(samples::AbstractVector{Int16}, channels::Int)::Vector{Float64} =
    Float64[sum(Float64, view(samples, (f-1)*channels+1:f*channels)) / channels
            for f in 1:div(length(samples), channels)]

# `x`, sampled at `from` Hz, at `to` Hz by linear interpolation between neighbouring samples:
# speech-grade, with no anti-aliasing filter, so content above the lower rate's Nyquist frequency
# folds back when downsampling. The duration is kept: cld(length(x) * to, from) samples, the last
# ones holding the final input sample.
function _resample_linear(x::AbstractVector{Float64}, from::Int, to::Int)::Vector{Float64}
    n = length(x)
    (from == to || n == 0) && return collect(x)
    y = Vector{Float64}(undef, cld(n * to, from))
    for j in eachindex(y)
        i, r = divrem((j - 1) * from, to)   # the output sample's place on the input grid, in 1/to steps
        a, b = x[min(i, n - 1) + 1], x[min(i + 1, n - 1) + 1]
        y[j] = a + (r / to) * (b - a)
    end
    y
end

# Samples as 16-bit little-endian PCM, rounded and clipped to the 16-bit range.
function _pcm16le(x::AbstractVector{<:Real})::Vector{UInt8}
    out = Vector{UInt8}(undef, 2length(x))
    for (k, v) in enumerate(x)
        s = reinterpret(UInt16, round(Int16, clamp(v, -32768, 32767)))
        out[2k-1], out[2k] = s % UInt8, (s >> 8) % UInt8
    end
    out
end

# ─── Transcription (multipart upload → text or JSON) ─────────────────────────

const _TRANSCRIPTION_FORMATS = ("json", "text", "srt", "verbose_json", "vtt", "diarized_json")

"""
    TranscriptionRequest(; file, model=nothing, service=OPENAIServiceEndpoint)

An audio transcription request. `file` is a path on disk. `model` left
`nothing` is the service's default: `gpt-transcribe` for OpenAI; Ollama has none, so a
request to it names an installed model that hears audio (e.g. `model="gemma4:e4b"`) or
throws `ArgumentError` before any request, and so does a [`GenericOpenAIEndpoint`](@ref),
whose request names the model its server serves. Optional
`languages`, `keywords`, `prompt`, `response_format`, and `temperature` refine decoding.
For `gpt-transcribe`, a legacy singular `language` is translated to `languages`.
Other models retain their singular `language` field. Never set both forms.
`response_format` is one of `json`, `text`, `srt`, `verbose_json`, `vtt`,
`diarized_json`, and `temperature` lies in `0 … 1` (on Ollama, any finite number); anything else is
an `ArgumentError`.
On OpenAI, `srt`, `vtt` and `verbose_json` (subtitles and timestamps) are documented only
for `whisper-1`, and `diarized_json` (speaker labels) only for `gpt-4o-transcribe-diarize`:
deprecated models that shut down on February 26, 2027
([speech to text](https://developers.openai.com/api/docs/guides/speech-to-text)).
Ollama's `/v1/audio/transcriptions` (Ollama 0.40.12 or later) reads one `language`, the
`prompt`, `temperature` (any finite number; its default 0) and a `response_format` of `json`, `text` or
`verbose_json`: a request to it that sets `languages`, `keywords` or another format throws
`ArgumentError` before any request, in Ollama's words.
Pass to [`transcribe`](@ref).
"""
@kwdef struct TranscriptionRequest
    service::ServiceEndpointSpec = OPENAIServiceEndpoint
    file::String
    model::Union{String,Nothing} = nothing
    language::Union{String,Nothing} = nothing
    prompt::Union{String,Nothing} = nothing
    response_format::Union{String,Nothing} = nothing
    temperature::Union{Float64,Nothing} = nothing
    languages::Union{Vector{String},Nothing} = nothing
    keywords::Union{Vector{String},Nothing} = nothing
    function TranscriptionRequest(service, file, model, language, prompt, response_format, temperature,
                                  languages=nothing, keywords=nothing)
        isfile(file) || throw(ArgumentError("file not found: $file"))
        isnothing(language) || isnothing(languages) || throw(ArgumentError("Set language or languages, not both"))
        if !isnothing(keywords) && any(k -> occursin(r"[<>\r\n]", k), keywords)
            throw(ArgumentError("Transcription keywords cannot contain <, >, or line breaks"))
        end
        isnothing(response_format) || response_format in _TRANSCRIPTION_FORMATS || throw(ArgumentError(
            "response_format must be one of $(_TRANSCRIPTION_FORMATS); got $(repr(response_format))"))
        refusal = isnothing(temperature) ? nothing : _transcription_temperature_refusal(service, temperature)
        isnothing(refusal) || throw(ArgumentError(refusal))
        new(service, file, model, language, prompt, response_format, temperature, languages, keywords)
    end
end

# The temperatures a transcription route takes: OpenAI's reference gives 0 to 1; Ollama's route parses any number
# and refuses only NaN and the infinities as an "invalid temperature" (server/transcription.go).
_transcription_temperature_refusal(_, t::Real)::Union{Nothing,String} =
    0 <= t <= 1 ? nothing : "temperature must be between 0 and 1; got $t"
_transcription_temperature_refusal(::OllamaEndpoint, t::Real)::Union{Nothing,String} =
    isfinite(t) ? nothing : "Ollama refuses a temperature that is not a finite number (\"invalid temperature\"); got $t"

# The model a transcription names, else the service's default: `gpt-transcribe` for OpenAI and an
# endpoint of the caller's own; none for Ollama or a generic endpoint, whose models are the caller's
# to name, as for their chat (vLLM's Transcriptions API lists `model` as "required").
_default_transcription_model(_) = "gpt-transcribe"
_default_transcription_model(::OllamaEndpoint) = throw(ArgumentError(
    "Ollama has no default transcription model: name an installed model that hears audio, e.g. " *
    "model = \"gemma4:e4b\""))
_default_transcription_model(::GenericOpenAIEndpoint) = throw(ArgumentError(
    "GenericOpenAIEndpoint has no default transcription model: name the model the server serves, e.g. " *
    "model = \"openai/whisper-large-v3-turbo\""))
# `@something` evaluates the default only when no model is named.
_transcription_model(t::TranscriptionRequest)::String = @something(t.model, _default_transcription_model(t.service))

# What a service's transcription route cannot carry, refused before any request.
_check_transcription(_, ::TranscriptionRequest) = nothing
# Ollama's route (0.40.12 or later) takes one `language` and no `keywords` — its OpenAI-compatibility page lists
# file, model, language, prompt, response_format, temperature and stream — and answers a format other than json,
# text and verbose_json with HTTP 400 (server/transcription.go); each is refused in Ollama's words.
const _OLLAMA_TRANSCRIPTION_FORMATS = ("json", "text", "verbose_json")
function _check_transcription(::OllamaEndpoint, t::TranscriptionRequest)
    isnothing(t.languages) || throw(ArgumentError(
        "Ollama's transcription takes one `language`, an \"ISO 639-1 code such as `fr`, or an English language " *
        "name; omit it to let the model detect the language\" (its OpenAI-compatibility page): pass language=… " *
        "in place of languages"))
    isnothing(t.keywords) || throw(ArgumentError(
        "Ollama's OpenAI-compatibility page lists no `keywords` for transcription (it lists file, model, language, " *
        "prompt, response_format, temperature and stream): remove them"))
    fmt = t.response_format
    isnothing(fmt) || fmt in _OLLAMA_TRANSCRIPTION_FORMATS || throw(ArgumentError(
        "Ollama answers HTTP 400 \"response_format $(repr(fmt)) is not supported, use json, text or verbose_json\": " *
        "leave response_format unset (json), or ask for \"text\" or \"verbose_json\""))
    nothing
end

# Whether a 200 transcription reply is the JSON object `{"text": …}` rather than the bare text.
_transcription_json(_, ::TranscriptionRequest, resp::HTTP.Response)::Bool =
    occursin("application/json", HTTP.header(resp, "Content-Type", ""))
_transcription_json(::OllamaEndpoint, ::TranscriptionRequest, resp::HTTP.Response)::Bool = _ollama_audio_json(resp)

# Ollama 0.40.12 and later label a JSON reply (json, verbose_json) application/json and a text one text/plain
# (server/transcription.go). Any other label is refused rather than read as a transcript: before 0.40.12 Ollama
# labelled its JSON application/x-ndjson, the header of the chat stream it transcribed with.
function _ollama_audio_json(resp::HTTP.Response)::Bool
    label = HTTP.header(resp, "Content-Type", "")
    kind = lowercase(strip(first(split(label, ';'))))
    kind == "application/json" && return true
    kind == "text/plain" && return false
    error("Ollama answered with Content-Type $(repr(label)), neither application/json nor text/plain: UniLM reads " *
          "the audio replies of Ollama 0.40.12 or later")
end

"Successful [`transcribe`](@ref)/[`translate`](@ref) result; `text` holds the transcript or translation (via [`transcript_text`](@ref)), `raw` the parsed JSON a transcription or Ollama translation reply carried (`nothing` for a text reply and for an OpenAI Realtime translation)."
@kwdef struct TranscriptionSuccess <: LLMRequestResponse
    text::String
    raw::Union{Dict{String,Any},Nothing} = nothing
end

"""
    transcript_text(r::TranscriptionSuccess) -> String

The transcript text from a [`transcribe`](@ref) or [`translate`](@ref) result.
"""
transcript_text(r::TranscriptionSuccess) = r.text

function _transcription_parts(t::TranscriptionRequest)::Vector{Pair{String,Any}}
    model = _transcription_model(t)
    parts = Pair{String,Any}[
        "file" => HTTP.Multipart(basename(t.file), IOBuffer(read(t.file)), _mime_for(t.file)),
        "model" => model]
    languages = t.languages
    if !isnothing(t.language)
        if _model_family(model, "gpt-transcribe")
            languages = [t.language]
        else
            push!(parts, "language" => t.language)
        end
    end
    isnothing(languages) || append!(parts, ("languages[]" => lang for lang in languages))
    isnothing(t.keywords) || append!(parts, ("keywords[]" => word for word in t.keywords))
    !isnothing(t.prompt) && push!(parts, "prompt" => t.prompt)
    !isnothing(t.response_format) && push!(parts, "response_format" => t.response_format)
    !isnothing(t.temperature) && push!(parts, "temperature" => string(t.temperature))
    parts
end

"""
    transcribe(t::TranscriptionRequest) / transcribe(path; model=nothing, kwargs...)

Transcribe audio to text in the source language. Returns `TranscriptionSuccess`
(`.text`, via [`transcript_text`](@ref)), `AudioFailure`, or `AudioCallError`. `model`
left `nothing` is the service's default (`gpt-transcribe` on OpenAI).

On Ollama (0.40.12 or later), an installed model that hears audio transcribes through
Ollama's OpenAI-compatible `/v1/audio/transcriptions` route, which takes `language`, `prompt`,
`temperature` and a `response_format` of `json`, `text` or `verbose_json`; the reply is read
by its Content-Type, JSON or, for `"text"`, the bare text. There is no default model, and
what that route cannot carry — `languages`, `keywords`, another format — throws
`ArgumentError` before any request:

```julia
r = transcribe("memo.wav"; service=OllamaEndpoint(), model="gemma4:e4b", language="en",
               temperature=0.0, response_format="text")
r isa TranscriptionSuccess && println(transcript_text(r))
```

A [`GenericOpenAIEndpoint`](@ref) sends the request to its server's
`/v1/audio/transcriptions` with every field it sets, as given — vLLM serves OpenAI's
Transcriptions API there — and has no default model either.

Transient statuses (408/429/500/502–504/520–524/529) and transport failures are retried with
backoff and jitter under the resolved [`RequestConfig`](@ref) (`max_attempts`,
`total_deadline`; `retry-after-ms`, else `Retry-After`, honored), as [`chatrequest!`](@ref)
retries them, and each attempt sends the whole file again; a 429 that no retry can serve is
returned at once. Pass `config::Union{Nothing,RequestConfig}` to override the timeout and
retry budget for this call.
"""
function transcribe(t::TranscriptionRequest; config::Union{Nothing,RequestConfig}=nothing,
                    cancel::Union{Nothing,CancelToken}=nothing)
    # An endpoint that declares `:audio` transcribes too (`has_capability`), as does one that declares
    # `:transcription` alone.
    validate_capability(t.service, :transcription, "Audio API")
    # Local validation throws before the request (inside the `try` it would become an AudioCallError).
    _check_transcription(t.service, t)
    _transcription_model(t)
    cfg = _resolve_config(config); t0 = time_ns()
    try
        # Retried as the endpoint's other inference verbs are. The attempt that sends a multipart
        # Form reads it to its end, so every attempt builds its own from the file (`_BodyFactory`).
        resp = _http_with_retries(cfg, t0, "POST", _api_base_url(t.service) * AUDIO_TRANSCRIPTIONS_PATH,
            auth_header_multipart(t.service), _BodyFactory(() -> HTTP.Form(_transcription_parts(t)));
            cancel=_resolve_cancel(cancel))
        resp.status == 200 ? _transcription_success(_transcription_json(t.service, t, resp), resp) :
                             _failure(AudioFailure, resp)
    catch e
        e isa InterruptException && rethrow()
        _callerr(AudioCallError, e)
    end
end
transcribe(path::String; model::Union{String,Nothing}=nothing, service::ServiceEndpointSpec=OPENAIServiceEndpoint, config::Union{Nothing,RequestConfig}=nothing,
           cancel::Union{Nothing,CancelToken}=nothing, kwargs...) =
    transcribe(TranscriptionRequest(; service=service, file=path, model=model, kwargs...); config=config, cancel=cancel)

# A 200 transcription or translation reply: the JSON object `{"text": …}`, or the bare text.
function _transcription_success(json::Bool, resp::HTTP.Response)::TranscriptionSuccess
    json || return TranscriptionSuccess(text=String(resp.body))
    d = JSON.parse(resp.body; dicttype=Dict{String,Any})
    TranscriptionSuccess(text=get(d, "text", ""), raw=d)
end

# ─── Translation (OpenAI Realtime translation sessions, Ollama) ──────────────

"""
    translate(path; language="en", model=nothing, service=OPENAIServiceEndpoint,
              config=nothing) -> TranscriptionSuccess | AudioFailure | AudioCallError

Translate the speech in the audio file at `path` into `language` and return the translation as
text: a [`TranscriptionSuccess`](@ref), read with [`transcript_text`](@ref). `model` left `nothing`
is the service's default: `gpt-realtime-translate` on OpenAI; Ollama has none.

On OpenAI the audio goes to an OpenAI Realtime translation session
(`wss://api.openai.com/v1/realtime/translations`,
[Realtime translation](https://developers.openai.com/api/docs/guides/realtime-translation)): UniLM
sets the target language (the session's `audio.output.language`, such as `"fr"` or `"es"`; English
unless named), streams the audio in 200 ms frames, closes the session and joins the translated
transcript. The session also returns translated speech, which `translate` does not keep. Speech
already in the target language can come back as empty text: an English clip translated into English
did. A short clip's translation also varies from call to call: of ten translations into English of a
two-second French sentence made with [`speak`](@ref), one came back empty and six named words the
sentence does not have (October 2026).

There `path` must be a 16-bit PCM WAV file, at any sample rate and channel count: a translation
session takes 24 kHz mono PCM16 only, so UniLM mixes the channels down and resamples by linear
interpolation (speech-grade: there is no anti-aliasing filter). Any other file throws `ArgumentError`
before any I/O, naming the conversion; so do an empty file, a blank `language` and a service that
has neither OpenAI's Realtime API nor Ollama's translation route.

On [`OllamaEndpoint`](@ref) (Ollama 0.40.12 or later) the file goes as it is to Ollama's
OpenAI-compatible translation route, one multipart request carrying the `model` and the target
`language` as `to_language`, the "ISO 639-1 code of the output language"
([OpenAI compatibility](https://docs.ollama.com/api/openai-compatibility)); the translation is the
reply's `text`, and `raw` holds the reply. Ollama decodes a WAV file itself and passes any other
container to the model whole. There is no default model: name an installed model that hears audio,
such as `gemma4:e4b`; a call without one, a blank `language`, and a missing or empty file throw
`ArgumentError` before any request:

```julia
r = translate("memo.wav"; language="fr", service=OllamaEndpoint(), model="gemma4:e4b")
r isa TranscriptionSuccess && println(transcript_text(r))
```

A status Ollama answers is an [`AudioFailure`](@ref) (a language it does not translate into is a
400); transient statuses and transport failures are retried as [`transcribe`](@ref) retries them,
the whole file sent each time.

On OpenAI, a failure the provider reports is an [`AudioFailure`](@ref): a Realtime `error` event,
for example for a language it does not translate into (`status = nothing`, the event's JSON as
`response`), read even when the server closes the session during the upload, or a WebSocket upgrade
the server refuses (a bad key is a 401). A session that closes before `session.closed` (`error`
names the close's status and reason), a timeout and a cancel are an [`AudioCallError`](@ref).
`config::Union{Nothing,RequestConfig}` bounds the call as for [`speak`](@ref): `total_deadline` the
whole call whatever the socket does, `connect_timeout` the open, `stream_idle_timeout` every send and
every wait for the next event. On OpenAI it is a single attempt, which `max_attempts` does not
change: a retry after the session's audio was sent may bill a second translation, and a session's
audio cannot be resumed. The ambient [`CancelToken`](@ref)
([`with_cancel`](@ref)) is observed: a token already cancelled sends nothing, and a cancel ends the
call at once, with an `AudioCallError` whose `cause` is a [`UniLMCancelled`](@ref).
"""
function translate(path::AbstractString; language::AbstractString="en",
                   model::Union{AbstractString,Nothing}=nothing,
                   service::ServiceEndpointSpec=OPENAIServiceEndpoint,
                   config::Union{Nothing,RequestConfig}=nothing,
                   cancel::Union{Nothing,CancelToken}=nothing)
    _with_resolved_cancel(() -> _translate(service, path, language, model, config), cancel)
end

# OpenAI: a Realtime translation session, on `gpt-realtime-translate` unless the call names another model.
function _translate(service, path::AbstractString, language::AbstractString, model::Union{AbstractString,Nothing},
                    config::Union{Nothing,RequestConfig})
    validate_capability(service, :realtime, "translate (OpenAI Realtime API)")
    m = String(something(model, "gpt-realtime-translate"))
    url = _realtime_translation_url(service, m)
    lang = _translation_language(language)
    pcm = _translation_pcm(path)
    cfg = _resolve_config(config); t0 = time_ns()
    try
        r = _realtime_call(session -> _realtime_translate(session, pcm, lang), url, m, service, cfg, t0)
        r isa LLMRequestResponse ? r : _realtime_closed("session.closed")
    catch e
        e isa InterruptException && rethrow()
        _callerr(AudioCallError, e)
    end
end

const _OLLAMA_TRANSLATIONS_PATH = "/v1/audio/translations"   # Ollama's OpenAI-compatible translation route

# Ollama (0.40.12 or later): the file as it is, the model and the target language as `to_language`, in one multipart
# request, answered as a transcription is (https://docs.ollama.com/api/openai-compatibility). Retried as `transcribe`
# is, each attempt building its form from the file.
function _translate(service::OllamaEndpoint, path::AbstractString, language::AbstractString,
                    model::Union{AbstractString,Nothing}, config::Union{Nothing,RequestConfig})
    validate_capability(service, :audio, "translate (Ollama translation route)")
    (isnothing(model) || isempty(strip(model))) && throw(ArgumentError(
        "Ollama has no default translation model: name an installed model that hears audio, e.g. " *
        "model = \"gemma4:e4b\""))
    lang = _translation_language(language)
    isfile(path) || throw(ArgumentError("file not found: $path"))
    filesize(path) > 0 || throw(ArgumentError(
        "$(basename(path)) is empty, and Ollama answers an empty upload with HTTP 400 \"audio file is empty\": " *
        "translate a recording"))
    file, m = String(path), String(model)
    form() = HTTP.Form(Pair{String,Any}["file" => HTTP.Multipart(basename(file), IOBuffer(read(file)), _mime_for(file)),
                                        "model" => m, "to_language" => lang])
    cfg = _resolve_config(config); t0 = time_ns()
    try
        resp = _http_with_retries(cfg, t0, "POST", _api_base_url(service) * _OLLAMA_TRANSLATIONS_PATH,
                                  auth_header_multipart(service), _BodyFactory(form))
        resp.status == 200 ? _transcription_success(_ollama_audio_json(resp), resp) : _failure(AudioFailure, resp)
    catch e
        e isa InterruptException && rethrow()
        _callerr(AudioCallError, e)
    end
end

function _translation_language(language::AbstractString)::String
    l = strip(language)
    isempty(l) && throw(ArgumentError("translate needs a target language, e.g. language = \"fr\""))
    String(l)
end

# The audio a translation session takes, from a 16-bit PCM WAV file: mixed down to mono and resampled
# to 24 kHz (https://developers.openai.com/api/reference/resources/realtime/translation-client-events).
function _translation_pcm(path::AbstractString)::Vector{UInt8}
    isfile(path) || throw(ArgumentError("file not found: $path"))
    w = try
        _read_wav(read(path))
    catch e
        e isa ArgumentError || rethrow()
        throw(ArgumentError("cannot translate $(basename(path)): $(e.msg). translate reads 16-bit PCM WAV files " *
                            "(any rate, any channels): convert it to 16-bit PCM WAV first, e.g. " *
                            "`ffmpeg -i in.m4a -c:a pcm_s16le out.wav`"))
    end
    isempty(w.samples) && throw(ArgumentError("$(basename(path)) holds no audio samples"))
    _pcm16le(_resample_linear(_downmix(w.samples, w.channels), w.rate, _SPEECH_RATE))
end
