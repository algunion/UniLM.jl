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
    SpeechTurn(speaker, text; style=nothing)

One turn of a dialogue that Gemini TTS reads in several voices: the `speaker`, a name the
request's `speakers` gives a voice, the `text` the speaker reads, and the turn's delivery `style`
(its `speechMetadata.style`, such as `"whispered urgently"`). A blank speaker or text throws
`ArgumentError`. Give the turns to [`speak`](@ref) or [`SpeechRequest`](@ref) as `input`.
"""
struct SpeechTurn
    speaker::String
    text::String
    style::Union{String,Nothing}
    function SpeechTurn(speaker::AbstractString, text::AbstractString; style::Union{AbstractString,Nothing}=nothing)
        isempty(strip(speaker)) && throw(ArgumentError("a speech turn names its speaker (got $(repr(speaker)))"))
        isempty(strip(text)) && throw(ArgumentError("the speech turn of $(repr(speaker)) has no text"))
        new(speaker, text, style)
    end
end

# A dialogue's voices, by speaker; refused, with the speech wires that read one voice per request, before any
# request. Gemini TTS: "Every turn in a multi-speaker request must explicitly include `speaker` inside
# `speech_metadata` matching one of the configured speakers" (speech-generation guide). Its limits section says
# multi-speaker generation "supports up to 2 speakers using prebuilt voices", yet three speakers and a designed
# voice beside a prebuilt one were each spoken (HTTP 200, 2026-11-17): neither is refused here.
function _dialogue_voices(::_SpeechWire, rules::_SpeechRules, _, _, _, _)
    throw(ArgumentError("$(rules.provider) reads one voice per request: a dialogue in several voices (SpeechTurns " *
        "with speakers) is Gemini TTS's: service=GEMINIServiceEndpoint ($(rules.docs))"))
end
function _dialogue_voices(wire::_GeminiSpeechWire, ::_SpeechRules, turns::Vector{SpeechTurn}, speakers, voice,
                          instructions)::Vector{Pair{String,String}}
    isnothing(voice) || throw(ArgumentError(
        "a dialogue's voices are its speakers': speakers = [\"Joe\" => \"Puck\", \"Jane\" => \"Kore\"]; unset voice"))
    isnothing(instructions) || throw(ArgumentError(
        "a dialogue's delivery is each turn's style (SpeechTurn(speaker, text; style=…)); unset instructions"))
    (isnothing(speakers) || isempty(speakers)) && throw(ArgumentError(
        "a dialogue names each speaker's voice: speakers = [\"Joe\" => \"Puck\", \"Jane\" => \"Kore\"]"))
    cast = Pair{String,String}[String(first(p)) => _speech_voice(wire, last(p)) for p in speakers]
    names = first.(cast)
    any(n -> isempty(strip(n)), names) && throw(ArgumentError("a speaker's name is blank"))
    allunique(names) || throw(ArgumentError("each speaker is named once (got $(join(repr.(names), ", ")))"))
    for t in turns
        t.speaker in names || throw(ArgumentError(
            "the turn of $(repr(t.speaker)) names no speaker in speakers ($(join(repr.(names), ", "))): every turn's " *
            "speaker must be one of them"))
    end
    cast
end

# The language a speech wire reads in: Gemini TTS's `speechConfig.languageCode`, a BCP-47 tag (it answered
# "xx-ZZ" with HTTP 400 "Invalid language_code", 2026-11-17); the other wires have no such field.
_speech_language(::_GeminiSpeechWire, ::_SpeechRules, l::AbstractString)::String =
    isempty(strip(l)) ? throw(ArgumentError("the speech language is blank: a BCP-47 tag such as \"fr-FR\"")) : String(l)
_speech_language(::_SpeechWire, rules::_SpeechRules, ::AbstractString) = throw(ArgumentError(
    "$(rules.provider) takes no speech language: it reads the text in the language it is written in; " *
    "Gemini TTS takes one (service=GEMINIServiceEndpoint)"))
_speech_language(::_SpeechWire, ::_SpeechRules, ::Nothing) = nothing

"""
    SpeechRequest(; input, service=OPENAIServiceEndpoint, model=nothing, voice=nothing,
                  response_format="wav", speed=nothing, instructions=nothing, speakers=nothing, language=nothing)

A text-to-speech request; pass it to [`speak`](@ref). `input` is the text to read aloud, or on
Gemini a dialogue: a vector of [`SpeechTurn`](@ref)s, read in the voices `speakers` gives each speaker
(below). `model` and `voice` left `nothing` are the service's defaults:

| `service` | speaks through | default `model` | default `voice` | `voice` | `speed` |
|:--|:--|:--|:--|:--|:--|
| `OPENAIServiceEndpoint` | a Realtime API session | `gpt-realtime-2.1-mini` | `marin` | `alloy`, `ash`, `ballad`, `coral`, `echo`, `sage`, `shimmer`, `verse`, `marin`, `cedar` | `0.25` to `1.5` |
| `GEMINIServiceEndpoint` | Gemini TTS | `gemini-3.8-flash-tts` | `Kore` | a prebuilt voice (the 30 studio voices or one of the Extended Voice Library), a designed or replicated voice's `voice_…` id, or a replicated voice's `voicekey_…` key | none |
| [`OllamaEndpoint`](@ref) | Ollama's speech route (Ollama 0.40.9 or later) | none: name an installed speech model, such as `phonon` | the model's own (`voice` not sent) | one of the model's voices, which [`model_info`](@ref) lists (`voices`), sent as given | `1` only |

Every service produces 16-bit mono PCM at 24 kHz: `response_format` `"wav"` returns it as a WAV file,
`"pcm"` as the raw samples. `instructions` direct the delivery (tone, pace, accent): on OpenAI they
follow the session's instruction to read `input` word for word, on Gemini they are the text's speech
style (`speechMetadata.style`). Gemini TTS has no speed parameter: describe the pace in
`instructions` (`"speaking slowly"`). Ollama takes neither: it answers non-empty `instructions` and
any speed but 1 with HTTP 400.

On Gemini, `language` is the BCP-47 tag the text is read in (`speechConfig.languageCode`, such as
`"fr-FR"`; Gemini answers an unknown one with HTTP 400), and a dialogue reads several voices in one
request ([speech generation](https://ai.google.dev/gemini-api/docs/generate-content/speech-generation)):

```julia
speak([SpeechTurn("Joe", "How's it going today, Jane?"), SpeechTurn("Jane", "Not too bad!"; style="cheerful")];
      service=GEMINIServiceEndpoint, speakers=["Joe" => "Puck", "Jane" => "Kore"])
```

Each turn is a part of its own with its speaker (`speechMetadata.speaker`) and style, and `speakers`
becomes `multiSpeakerVoiceConfig`, each voice a prebuilt one or any other, as `voice` is; the
request's `input` field holds the turns' texts, joined by line breaks, and its `voice` is `""`. Every
turn's speaker must be one of `speakers`; a dialogue takes no `voice` (the speakers have theirs) and
no `instructions` (each turn has its style). Google's guide says multi-speaker speech "supports up to
2 speakers using prebuilt voices"; three speakers, and a designed voice beside a prebuilt one, were
each spoken on 17 November 2026, so neither is refused. Other services read one voice and take no
language: a dialogue, `speakers` or `language` there throws `ArgumentError`.

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
    turns::Vector{SpeechTurn}
    speakers::Vector{Pair{String,String}}
    language::Union{String,Nothing}
    function SpeechRequest(service::ServiceEndpointSpec, model::Union{AbstractString,Nothing},
                           input::Union{AbstractString,AbstractVector{SpeechTurn}}, voice::Union{AbstractString,Nothing},
                           response_format::AbstractString, speed::Union{Real,Nothing},
                           instructions::Union{AbstractString,Nothing}, speakers=nothing,
                           language::Union{AbstractString,Nothing}=nothing)
        wire = _speech_wire(service)
        rules = _speech_rules(wire)
        turns = input isa AbstractString ? SpeechTurn[] : collect(SpeechTurn, input)
        text = input isa AbstractString ? String(input) : join((t.text for t in turns), "\n")
        isempty(strip(text)) && throw(ArgumentError("speak needs text to read: input is empty"))
        if input isa AbstractString
            isnothing(speakers) || throw(ArgumentError(
                "speakers give a dialogue's voices: pass input as SpeechTurns (SpeechTurn(speaker, text)), or unset speakers"))
            v, cast = _speech_voice(wire, something(voice, rules.voice)), Pair{String,String}[]
        else
            v, cast = "", _dialogue_voices(wire, rules, turns, speakers, voice, instructions)
        end
        response_format in _SPEECH_FORMATS || throw(ArgumentError(_speech_format_refusal(wire, response_format)))
        _check_speech(wire, text, instructions)
        new(service, _speech_model(wire, model), text, v, response_format,
            isnothing(speed) ? nothing : _speech_speed(rules, speed), instructions, turns, cast,
            _speech_language(wire, rules, language))
    end
end
SpeechRequest(; input::Union{AbstractString,AbstractVector{SpeechTurn}}, service::ServiceEndpointSpec=OPENAIServiceEndpoint,
              model::Union{AbstractString,Nothing}=nothing, voice::Union{AbstractString,Nothing}=nothing,
              response_format::AbstractString="wav", speed::Union{Real,Nothing}=nothing,
              instructions::Union{AbstractString,Nothing}=nothing, speakers=nothing,
              language::Union{AbstractString,Nothing}=nothing) =
    SpeechRequest(service, model, input, voice, response_format, speed, instructions, speakers, language)

"Successful [`speak`](@ref) result: `audio` holds a WAV file (`content_type` `\"audio/wav\"`) or the raw 16-bit little-endian mono samples at 24 kHz (`\"audio/pcm\"`). Save it with [`save_audio`](@ref)."
@kwdef struct SpeechSuccess <: LLMRequestResponse
    audio::Vector{UInt8}
    content_type::String = ""
end

"""
    speak(s::SpeechRequest; config=nothing, cancel=nothing) -> SpeechSuccess | AudioFailure | AudioCallError
    speak(input; service=OPENAIServiceEndpoint, model=nothing, voice=nothing, response_format="wav",
          speed=nothing, instructions=nothing, speakers=nothing, language=nothing, config=nothing, cancel=nothing)

Synthesize speech from text, or on Gemini from a dialogue of [`SpeechTurn`](@ref)s read in the
voices `speakers` names; the keywords build a [`SpeechRequest`](@ref).

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
that no retry can serve is returned at once. Speech is not priced: [`token_usage`](@ref) and
[`estimated_cost`](@ref) throw for a `SpeechSuccess`.

Pass `cancel::Union{Nothing,CancelToken}` (default: the ambient [`with_cancel`](@ref) token) to make
the call cancellable: once the token is cancelled it returns an `AudioCallError` whose `cause` is a
[`UniLMCancelled`](@ref) — without sending anything when the token was already cancelled, at once
when mid-request (never retried).
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
speak(input::Union{AbstractString,AbstractVector{SpeechTurn}}; service::ServiceEndpointSpec=OPENAIServiceEndpoint,
      model::Union{AbstractString,Nothing}=nothing, voice::Union{AbstractString,Nothing}=nothing,
      response_format::AbstractString="wav", speed::Union{Real,Nothing}=nothing,
      instructions::Union{AbstractString,Nothing}=nothing, speakers=nothing,
      language::Union{AbstractString,Nothing}=nothing, config::Union{Nothing,RequestConfig}=nothing,
      cancel::Union{Nothing,CancelToken}=nothing) =
    speak(SpeechRequest(service, model, input, voice, response_format, speed, instructions, speakers, language);
          config, cancel)

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

# Gemini's audio models (TTS, and Transcribe for `transcribe`) answer on generateContent.
_gemini_speech_base(::Type{GEMINIServiceEndpoint}) = GEMINI_NATIVE_BASE
_gemini_speech_url(service, model::String)::String =
    _gemini_speech_base(service) * "/models/" * _uripart(model) * ":generateContent"

# Gemini 3.8 TTS reads `text` verbatim; a delivery direction is the part's speech style, and in a dialogue the
# part's speaker is the turn's.
function _gemini_speech_part(text::String, speaker::Union{String,Nothing}, style::Union{String,Nothing})
    part = Dict{Symbol,Any}(:text => text)
    meta = Dict{Symbol,Any}()
    isnothing(speaker) || (meta[:speaker] = speaker)
    isnothing(style) || (meta[:style] = style)
    isempty(meta) || (part[:speechMetadata] = meta)
    part
end

# VoiceConfig holds `prebuiltVoiceConfig`, "The configuration for the prebuilt voice to use", or `voice`, which
# takes a prebuilt voice's name, a stored voice's id (`voice_…`) and a replication key (`voicekey_…`)
# (https://ai.google.dev/api/generate-content#VoiceConfig): a studio voice keeps the prebuilt form, and every other
# voice goes as `voice`.
_gemini_voice_config(voice::String) = voice in _GEMINI_SPEECH.voices ?
    Dict(:prebuiltVoiceConfig => Dict(:voiceName => voice)) : Dict(:voice => voice)

function _gemini_speech_body(s::SpeechRequest)::String
    dialogue = !isempty(s.turns)
    parts = dialogue ? [_gemini_speech_part(t.text, t.speaker, t.style) for t in s.turns] :
                       [_gemini_speech_part(s.input, nothing, s.instructions)]
    # A dialogue's voices are its MultiSpeakerVoiceConfig, "mutually exclusive with the voice_config field".
    speech = dialogue ?
        Dict{Symbol,Any}(:multiSpeakerVoiceConfig => Dict(:speakerVoiceConfigs =>
            [Dict(:speaker => name, :voiceConfig => _gemini_voice_config(v)) for (name, v) in s.speakers])) :
        Dict{Symbol,Any}(:voiceConfig => _gemini_voice_config(s.voice))
    isnothing(s.language) || (speech[:languageCode] = s.language)
    JSON.json(Dict(
        :contents => [Dict(:role => "user", :parts => parts)],
        :generationConfig => Dict(
            :responseModalities => ["AUDIO"],
            :speechConfig => speech,
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

# ─── Transcription (OpenAI-wire multipart upload; Gemini generateContent) ────

const _TRANSCRIPTION_FORMATS = ("json", "text", "srt", "verbose_json", "vtt", "diarized_json")
# OpenAI's CreateTranscriptionRequest enums: the `include` items and the timestamp granularities. An `include`
# item other than "logprobs" is answered with HTTP 400 "Input should be 'logprobs'" (2026-11-17).
const _TRANSCRIPTION_INCLUDES = ("logprobs",)
const _TIMESTAMP_GRANULARITIES = ("word", "segment")
# Gemini 3.5 Transcribe's two modes (https://ai.google.dev/gemini-api/docs/transcribe#transcription-modes).
const _TRANSCRIPTION_MODES = ("verbatim", "smart")
# "Up to 4 speakers are supported" (CreateTranscriptionRequest.known_speaker_names).
const _KNOWN_SPEAKERS_MAX = 4

"""
    TranscriptionVAD(; threshold=nothing, prefix_padding_ms=nothing, silence_duration_ms=nothing)

Server-side voice activity detection that cuts a transcription's audio into chunks: OpenAI's
`chunking_strategy` `{"type": "server_vad", …}`, given to [`TranscriptionRequest`](@ref) as
`chunking_strategy` (the string `"auto"` lets the server choose the chunking). A field left
`nothing` is the API's default: `threshold` (sensitivity, `0` to `1`; `0.5`), `prefix_padding_ms`
(audio kept before detected speech; `300`) and `silence_duration_ms` (the silence that ends speech;
`200`). A threshold outside `0 … 1` or a negative duration throws `ArgumentError`.
"""
struct TranscriptionVAD
    threshold::Union{Float64,Nothing}
    prefix_padding_ms::Union{Int,Nothing}
    silence_duration_ms::Union{Int,Nothing}
    function TranscriptionVAD(threshold::Union{Real,Nothing}, prefix_padding_ms::Union{Integer,Nothing},
                              silence_duration_ms::Union{Integer,Nothing})
        # OpenAI answers a threshold above 1 with HTTP 400 "Expected a value <= 1.0" (2026-11-17).
        isnothing(threshold) || 0 <= threshold <= 1 || throw(ArgumentError(
            "TranscriptionVAD threshold is a sensitivity from 0.0 to 1.0 (got $threshold)"))
        for (name, ms) in (("prefix_padding_ms", prefix_padding_ms), ("silence_duration_ms", silence_duration_ms))
            isnothing(ms) || ms >= 0 || throw(ArgumentError("TranscriptionVAD $name is a duration in milliseconds, ≥ 0 (got $ms)"))
        end
        new(threshold, prefix_padding_ms, silence_duration_ms)
    end
end
TranscriptionVAD(; threshold::Union{Real,Nothing}=nothing, prefix_padding_ms::Union{Integer,Nothing}=nothing,
                 silence_duration_ms::Union{Integer,Nothing}=nothing) =
    TranscriptionVAD(threshold, prefix_padding_ms, silence_duration_ms)

"""
    TranscriptionRequest(; file, model=nothing, service=OPENAIServiceEndpoint, …)

An audio transcription request. `file` is a path on disk. `model` left
`nothing` is the service's default: `gpt-transcribe` for OpenAI, `gemini-3.5-transcribe` for
`GEMINIServiceEndpoint`; Ollama has none, so a
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

OpenAI's further options ([create transcription](https://developers.openai.com/api/reference/resources/audio/subresources/transcriptions/methods/create)):
- `include`: `["logprobs"]` adds the tokens' log probabilities to the reply (`raw["logprobs"]`).
- `timestamp_granularities`: `"word"` and/or `"segment"`, with `response_format = "verbose_json"`
  only (OpenAI's rule: with `json` it answers without them); the reply's segments and words are
  the result's [`TranscriptSegment`](@ref)s and [`TranscriptWord`](@ref)s.
- `chunking_strategy`: `"auto"`, or a [`TranscriptionVAD`](@ref) with server VAD settings.
- `known_speaker_names` and `known_speaker_references`: up to 4 speakers for a diarization model,
  one reference each — an audio sample of 2 to 10 seconds, as a data URL (`data:audio/wav;base64,…`)
  or a path to an audio file, which UniLM sends as one.
- `stream = true`: [`transcribe`](@ref) streams the transcript (see there); `whisper-1` does not
  stream ("Streaming is not supported for the whisper-1 model and will be ignored"), so a stream on
  it throws `ArgumentError`.

On `GEMINIServiceEndpoint` (`gemini-3.5-transcribe`, [audio transcription](https://ai.google.dev/gemini-api/docs/transcribe)):
`languages` (or one `language`) are BCP-47 codes such as `"en-US"`; `keywords` are its custom
vocabulary (up to 1,000 terms); `diarization = true` labels the speakers; `timestamp_granularities
= ["word"]` adds word timestamps; `mode` is `"verbatim"` (the default) or `"smart"`, which removes
disfluencies and formats the text, and `source_text`, in smart mode, is a text the audio edits.
Gemini rejects custom vocabulary beside diarization or word timestamps, and smart mode beside
either, so those combinations throw `ArgumentError`, as do the options Gemini's transcription
does not take (`prompt`, `temperature`, `response_format`, `include`, `chunking_strategy`, known
speakers, `stream`, segment timestamps). `diarization`, `mode` and `source_text` are Gemini's:
any other service throws `ArgumentError` for them (on OpenAI, speaker labels come from
`gpt-4o-transcribe-diarize` with `response_format = "diarized_json"`).

Ollama's `/v1/audio/transcriptions` (Ollama 0.40.12 or later) reads one `language`, the
`prompt`, `temperature` (any finite number; its default 0), `stream` and a `response_format` of
`json`, `text` or `verbose_json`: a request to it that sets `languages`, `keywords`, another
format or any of OpenAI's further options but `stream` throws `ArgumentError` before any request.
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
    include::Union{Vector{String},Nothing} = nothing
    timestamp_granularities::Union{Vector{String},Nothing} = nothing
    chunking_strategy::Union{String,TranscriptionVAD,Nothing} = nothing
    known_speaker_names::Union{Vector{String},Nothing} = nothing
    known_speaker_references::Union{Vector{String},Nothing} = nothing
    stream::Bool = false
    diarization::Bool = false
    mode::Union{String,Nothing} = nothing
    source_text::Union{String,Nothing} = nothing
    function TranscriptionRequest(service, file, model, language, prompt, response_format, temperature,
                                  languages=nothing, keywords=nothing, include=nothing, timestamp_granularities=nothing,
                                  chunking_strategy=nothing, known_speaker_names=nothing, known_speaker_references=nothing,
                                  stream=false, diarization=false, mode=nothing, source_text=nothing)
        isfile(file) || throw(ArgumentError("file not found: $file"))
        isnothing(language) || isnothing(languages) || throw(ArgumentError("Set language or languages, not both"))
        if !isnothing(keywords) && any(k -> occursin(r"[<>\r\n]", k), keywords)
            throw(ArgumentError("Transcription keywords cannot contain <, >, or line breaks"))
        end
        isnothing(response_format) || response_format in _TRANSCRIPTION_FORMATS || throw(ArgumentError(
            "response_format must be one of $(_TRANSCRIPTION_FORMATS); got $(repr(response_format))"))
        refusal = isnothing(temperature) ? nothing : _transcription_temperature_refusal(service, temperature)
        isnothing(refusal) || throw(ArgumentError(refusal))
        _check_listed("include", include, _TRANSCRIPTION_INCLUDES)
        _check_listed("timestamp_granularities", timestamp_granularities, _TIMESTAMP_GRANULARITIES)
        chunking_strategy isa AbstractString && chunking_strategy != "auto" && throw(ArgumentError(
            "chunking_strategy is \"auto\" or a TranscriptionVAD (server VAD settings); got $(repr(chunking_strategy))"))
        _check_known_speakers(known_speaker_names, known_speaker_references)
        isnothing(mode) || mode in _TRANSCRIPTION_MODES || throw(ArgumentError(
            "mode is \"verbatim\" or \"smart\" (Gemini's transcription modes); got $(repr(mode))"))
        # "When provided in SMART mode, the model evaluates the audio input against this text"
        # (AudioTranscriptionConfig.sourceText, Gemini API reference).
        isnothing(source_text) || mode == "smart" || throw(ArgumentError(
            "source_text is the text a smart transcription edits: set mode = \"smart\" with it"))
        new(service, file, model, language, prompt, response_format, temperature, languages, keywords, include,
            timestamp_granularities, chunking_strategy, known_speaker_names, known_speaker_references, stream,
            diarization, mode, source_text)
    end
end

# A list option whose items come from a fixed set; an empty list asks for nothing and is refused too.
function _check_listed(name::String, items, allowed::Tuple)::Nothing
    isnothing(items) && return nothing
    (!isempty(items) && all(in(allowed), items)) || throw(ArgumentError(
        "$name takes one or more of $(join(repr.(allowed), ", ")); got $(repr(items))"))
    nothing
end

# Known speakers come in pairs, a name and its reference, at most four: "Optional list of speaker names that
# correspond to the audio samples provided in `known_speaker_references[]` … Up to 4 speakers are supported".
# A reference is a data URL or a path to an audio file, sent as one (`_speaker_reference`).
function _check_known_speakers(names, refs)::Nothing
    isnothing(names) && isnothing(refs) && return nothing
    (isnothing(names) || isnothing(refs) || length(names) != length(refs)) && throw(ArgumentError(
        "known_speaker_names and known_speaker_references go together, one reference per name; got " *
        "$(something(names, String[]) |> length) names and $(something(refs, String[]) |> length) references"))
    1 <= length(names) <= _KNOWN_SPEAKERS_MAX || throw(ArgumentError(
        "known speakers: 1 to $_KNOWN_SPEAKERS_MAX (got $(length(names)))"))
    any(n -> isempty(strip(n)), names) && throw(ArgumentError("a known speaker's name is blank"))
    for r in refs
        startswith(r, "data:") || isfile(r) || throw(ArgumentError(
            "a known speaker reference is a data URL (data:audio/wav;base64,…) or a path to an audio file; " *
            "$(repr(first(r, 40))) is neither"))
    end
    nothing
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

# The options a request sets, of those named: a value given, or a flag set.
_options_set(t::TranscriptionRequest, names)::Vector{String} =
    String[String(n) for n in names if (v = getfield(t, n); !(isnothing(v) || v === false))]

# What a service's transcription route cannot carry, refused before any request. Every OpenAI-wire route
# takes the multipart fields as given; diarization, mode and source_text are Gemini's
# (AudioTranscriptionConfig), which no OpenAI-wire route has.
const _GEMINI_TRANSCRIPTION_ONLY = (:diarization, :mode, :source_text)
function _check_transcription(service, t::TranscriptionRequest)
    set = _options_set(t, _GEMINI_TRANSCRIPTION_ONLY)
    isempty(set) || throw(ArgumentError(
        "$(join(set, ", ")): options of Gemini's transcription (service=GEMINIServiceEndpoint), which " *
        "$(_service_name(service)) does not take; on OpenAI, speaker labels come from the model " *
        "gpt-4o-transcribe-diarize with response_format=\"diarized_json\""))
    nothing
end
# OpenAI's own rules (https://developers.openai.com/api/reference/resources/audio/subresources/transcriptions/methods/create):
# "`response_format` must be set `verbose_json` to use timestamp granularities" — with json, gpt-transcribe answered
# 200 without timestamps (2026-11-17) — and "Streaming is not supported for the whisper-1 model and will be
# ignored": whisper-1 answered stream=true with its JSON (2026-11-17). Neither is left to the server to ignore.
function _check_transcription(service::Type{OPENAIServiceEndpoint}, t::TranscriptionRequest)
    invoke(_check_transcription, Tuple{Any,TranscriptionRequest}, service, t)
    isnothing(t.timestamp_granularities) || t.response_format == "verbose_json" || throw(ArgumentError(
        "OpenAI returns timestamp_granularities only with response_format=\"verbose_json\" (got " *
        "$(repr(t.response_format))): set it, on a model that answers it (whisper-1)"))
    t.stream && _model_family(_transcription_model(t), "whisper-1") && throw(ArgumentError(
        "whisper-1 does not stream (OpenAI: \"Streaming is not supported for the whisper-1 model and will be " *
        "ignored\"): unset stream, or stream gpt-transcribe"))
    nothing
end
# Ollama's route (0.40.12 or later) takes one `language` and no `keywords` — its OpenAI-compatibility page lists
# file, model, language, prompt, response_format, temperature and stream — and answers a format other than json,
# text and verbose_json with HTTP 400 (server/transcription.go); each is refused in Ollama's words.
const _OLLAMA_TRANSCRIPTION_FORMATS = ("json", "text", "verbose_json")
const _OLLAMA_TRANSCRIPTION_UNLISTED = (:include, :timestamp_granularities, :chunking_strategy, :known_speaker_names,
                                        :known_speaker_references, _GEMINI_TRANSCRIPTION_ONLY...)
function _check_transcription(::OllamaEndpoint, t::TranscriptionRequest)
    isnothing(t.languages) || throw(ArgumentError(
        "Ollama's transcription takes one `language`, an \"ISO 639-1 code such as `fr`, or an English language " *
        "name; omit it to let the model detect the language\" (its OpenAI-compatibility page): pass language=… " *
        "in place of languages"))
    isnothing(t.keywords) || throw(ArgumentError(
        "Ollama's OpenAI-compatibility page lists no `keywords` for transcription (it lists file, model, language, " *
        "prompt, response_format, temperature and stream): remove them"))
    set = _options_set(t, _OLLAMA_TRANSCRIPTION_UNLISTED)
    isempty(set) || throw(ArgumentError(
        "Ollama's OpenAI-compatibility page lists no $(join(set, ", ")) for transcription (it lists file, model, " *
        "language, prompt, response_format, temperature and stream): remove them"))
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

"""
    TranscriptSegment(; text, start_time=nothing, end_time=nothing, speaker=nothing, id=nothing)

A stretch of a transcript: its `text`, its start and end in seconds from the start of the audio
(`nothing` where the reply gives none), the `speaker` it is attributed to, and the provider's `id`
for it. From OpenAI, a `verbose_json` reply's segments (`id` an integer index; no speaker), a
`diarized_json` reply's and a streamed diarization's `transcript.text.segment` events (`id` such as
`"seg_0"`, `speaker` such as `"A"`, or a name from `known_speaker_names`); from Gemini, each part of
its transcription, one per speaker turn with diarization (`speaker` such as `"spk:0"`), whose start
and end are those of its first and last word when word timestamps were asked for.
"""
@kwdef struct TranscriptSegment
    text::String
    start_time::Union{Float64,Nothing} = nothing
    end_time::Union{Float64,Nothing} = nothing
    speaker::Union{String,Nothing} = nothing
    id::Union{String,Int,Nothing} = nothing
end

"""
    TranscriptWord(; text, start_time=nothing, end_time=nothing, speaker=nothing)

One recognized word and its start and end in seconds from the start of the audio: OpenAI's
`verbose_json` words (`timestamp_granularities` including `"word"`), Gemini's word timestamps
(`timestamp_granularities = ["word"]`), which carry the `speaker` of their turn with diarization.
"""
@kwdef struct TranscriptWord
    text::String
    start_time::Union{Float64,Nothing} = nothing
    end_time::Union{Float64,Nothing} = nothing
    speaker::Union{String,Nothing} = nothing
end

"""
    TranscriptionSuccess

Successful [`transcribe`](@ref)/[`translate`](@ref) result. `text` holds the transcript or
translation (via [`transcript_text`](@ref)); `raw` the parsed JSON a transcription or Ollama
translation reply carried — for a streamed transcription its `transcript.text.done` event, for
Gemini its `generateContent` reply — and `nothing` for a text reply and for an OpenAI Realtime
translation. `segments` and `words` are the reply's [`TranscriptSegment`](@ref)s and
[`TranscriptWord`](@ref)s, empty when it has none. The reply's usage (`raw["usage"]`, Gemini's
`raw["usageMetadata"]`) is not priced: [`token_usage`](@ref) and [`estimated_cost`](@ref) throw for a
`TranscriptionSuccess`, streamed or not.
"""
@kwdef struct TranscriptionSuccess <: LLMRequestResponse
    text::String
    raw::Union{Dict{String,Any},Nothing} = nothing
    segments::Vector{TranscriptSegment} = TranscriptSegment[]
    words::Vector{TranscriptWord} = TranscriptWord[]
end
TranscriptionSuccess(text, raw) = TranscriptionSuccess(text, raw, TranscriptSegment[], TranscriptWord[])

# A reply field, absent or null where the provider allows it; any other JSON type than the documented one is a
# malformed reply, and the throw becomes the verb's AudioCallError.
function _audio_field(d::AbstractDict, key::String, ::Type{T}, what::String)::Union{T,Nothing} where {T}
    v = get(d, key, nothing)
    isnothing(v) || v isa T || throw(ArgumentError("the $what's $(repr(key)) holds a $(typeof(v)), not the documented $T"))
    v
end
_audio_required(d::AbstractDict, key::String, ::Type{T}, what::String) where {T} =
    @something(_audio_field(d, key, T, what), throw(ArgumentError("the $what has no $(repr(key))")))
_audio_seconds(d::AbstractDict, key::String, what::String)::Union{Float64,Nothing} =
    (v = _audio_field(d, key, Real, what); isnothing(v) ? nothing : Float64(v))

# OpenAI's segments: a verbose_json TranscriptionSegment (`id` an integer) or a diarized one (`id` a string, and a
# `speaker`), each requiring its text, start and end.
function _transcript_segment(s)::TranscriptSegment
    s isa AbstractDict || throw(ArgumentError("a transcription segment is a JSON object, not a $(typeof(s))"))
    TranscriptSegment(text=_audio_required(s, "text", String, "segment"),
                      start_time=Float64(_audio_required(s, "start", Real, "segment")),
                      end_time=Float64(_audio_required(s, "end", Real, "segment")),
                      speaker=_audio_field(s, "speaker", String, "segment"),
                      id=_audio_required(s, "id", Union{String,Int}, "segment"))
end
# A TranscriptionWord requires the word; its start and end come with it.
function _transcript_word(w)::TranscriptWord
    w isa AbstractDict || throw(ArgumentError("a transcription word is a JSON object, not a $(typeof(w))"))
    TranscriptWord(text=_audio_required(w, "word", String, "word"), start_time=_audio_seconds(w, "start", "word"),
                   end_time=_audio_seconds(w, "end", "word"))
end
_reply_list(d::AbstractDict, key::String) = something(_audio_field(d, key, AbstractVector, "transcription reply"), Any[])
_transcript_segments(d::AbstractDict)::Vector{TranscriptSegment} = TranscriptSegment[_transcript_segment(s) for s in _reply_list(d, "segments")]
_transcript_words(d::AbstractDict)::Vector{TranscriptWord} = TranscriptWord[_transcript_word(w) for w in _reply_list(d, "words")]

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
    isnothing(t.include) || append!(parts, ("include[]" => x for x in t.include))
    isnothing(t.timestamp_granularities) || append!(parts, ("timestamp_granularities[]" => g for g in t.timestamp_granularities))
    isnothing(t.chunking_strategy) || push!(parts, "chunking_strategy" => _chunking_field(t.chunking_strategy))
    isnothing(t.known_speaker_names) || append!(parts, ("known_speaker_names[]" => n for n in t.known_speaker_names))
    isnothing(t.known_speaker_references) ||
        append!(parts, ("known_speaker_references[]" => _speaker_reference(r) for r in t.known_speaker_references))
    t.stream && push!(parts, "stream" => "true")
    parts
end

# OpenAI reads `chunking_strategy` as one form field holding "auto" or the VadConfig object's JSON: that JSON with
# a bogus type was answered HTTP 400 "Value must be 'server_vad'", while the bracketed fields a form gives other
# objects (`chunking_strategy[type]=bogus`) were ignored with a 200 (2026-11-17).
_chunking_field(auto::String)::String = auto
function _chunking_field(v::TranscriptionVAD)::String
    d = Dict{String,Any}("type" => "server_vad")
    isnothing(v.threshold) || (d["threshold"] = v.threshold)
    isnothing(v.prefix_padding_ms) || (d["prefix_padding_ms"] = v.prefix_padding_ms)
    isnothing(v.silence_duration_ms) || (d["silence_duration_ms"] = v.silence_duration_ms)
    JSON.json(d)
end

# A known speaker's reference as OpenAI takes it, a data URL: one given as such, or an audio file's bytes as one.
_speaker_reference(r::String)::String =
    startswith(r, "data:") ? r : string("data:", _mime_for(r), ";base64,", base64encode(read(r)))

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

Pass `cancel::Union{Nothing,CancelToken}` (default: the ambient [`with_cancel`](@ref) token) to make
the call cancellable: once the token is cancelled it returns an `AudioCallError` whose `cause` is a
[`UniLMCancelled`](@ref) — without sending anything when the token was already cancelled, at once
when mid-request (never retried).

**Streaming** (`stream = true`, on OpenAI and the OpenAI-wire routes): the call returns a `Task`
whose `fetch` yields the result, and the transcript arrives as OpenAI's server-sent events. Pass a
`callback`:

```julia
callback(chunk::Union{String,TranscriptSegment}, close::Ref{Bool})
```

Each `transcript.text.delta` is a `String` (the text added), and each `transcript.text.segment`
(a streamed diarization, `gpt-4o-transcribe-diarize` with `response_format = "diarized_json"`) a
[`TranscriptSegment`](@ref). The result is the same [`TranscriptionSuccess`](@ref) as unstreamed:
`text` is the `transcript.text.done` event's, `raw` that event (its `usage`, `logprobs` and
`languages`), `segments` the streamed segments. Setting `close[] = true` stops the stream at once
with an `AudioCallError` whose `cause` is a `UniLMCancelled`; an exception the callback throws ends
the call with that exception as `cause`. A status before the stream is an [`AudioFailure`](@ref),
retried as unstreamed while no callback has run; an error the stream itself carries
(`{"error": …}`) is an `AudioFailure` with `status = nothing`; a stream that ends before
`transcript.text.done`, a gap longer than `stream_idle_timeout`, a timeout and a cancel are an
`AudioCallError`. A `callback` without `stream = true` throws `ArgumentError`.
"""
function transcribe(t::TranscriptionRequest; config::Union{Nothing,RequestConfig}=nothing,
                    cancel::Union{Nothing,CancelToken}=nothing, callback=nothing)
    # An endpoint that declares `:audio` transcribes too (`has_capability`), as does one that declares
    # `:transcription` alone.
    validate_capability(t.service, :transcription, "Audio API")
    # Local validation throws before the request (inside the `try` it would become an AudioCallError).
    _check_transcription(t.service, t)
    _transcription_model(t)
    t.stream || isnothing(callback) || throw(ArgumentError(
        "transcribe: a callback runs on a stream only; set stream = true, or drop the callback"))
    cfg = _resolve_config(config); t0 = time_ns()
    t.stream && return _transcribe_stream(t, callback, cfg, t0, _resolve_cancel(cancel))
    try
        _transcribe(_transcription_wire(t.service), t, cfg, t0, _resolve_cancel(cancel))
    catch e
        e isa InterruptException && rethrow()
        _callerr(AudioCallError, e)
    end
end

# How a service transcribes: OpenAI's multipart route (OpenAI, Ollama, a generic server, an endpoint of the caller's
# own), or Gemini's generateContent.
abstract type _TranscriptionWire end
struct _OpenAITranscriptionWire <: _TranscriptionWire end
struct _GeminiTranscriptionWire <: _TranscriptionWire end
_transcription_wire(_) = _OpenAITranscriptionWire()
_transcription_wire(::Type{GEMINIServiceEndpoint}) = _GeminiTranscriptionWire()

# Retried as the endpoint's other inference verbs are. The attempt that sends a multipart Form reads it to its
# end, so every attempt builds its own from the file (`_BodyFactory`).
function _transcribe(::_OpenAITranscriptionWire, t::TranscriptionRequest, cfg::RequestConfig, t0::UInt64, tok)
    resp = _http_with_retries(cfg, t0, "POST", _api_base_url(t.service) * AUDIO_TRANSCRIPTIONS_PATH,
        auth_header_multipart(t.service), _BodyFactory(() -> HTTP.Form(_transcription_parts(t))); cancel=tok)
    resp.status == 200 ? _transcription_success(_transcription_json(t.service, t, resp), resp) :
                         _failure(AudioFailure, resp)
end
transcribe(path::String; model::Union{String,Nothing}=nothing, service::ServiceEndpointSpec=OPENAIServiceEndpoint, config::Union{Nothing,RequestConfig}=nothing,
           cancel::Union{Nothing,CancelToken}=nothing, callback=nothing, kwargs...) =
    transcribe(TranscriptionRequest(; service=service, file=path, model=model, kwargs...); config, cancel, callback)

# A 200 transcription or translation reply: the JSON object `{"text": …}`, with its segments and words when it
# has them (verbose_json, diarized_json), or the bare text.
function _transcription_success(json::Bool, resp::HTTP.Response)::TranscriptionSuccess
    json || return TranscriptionSuccess(text=String(resp.body))
    d = JSON.parse(resp.body; dicttype=Dict{String,Any})
    TranscriptionSuccess(text=get(d, "text", ""), raw=d, segments=_transcript_segments(d), words=_transcript_words(d))
end

# ─── Streamed transcription (OpenAI's server-sent transcript events) ─────────
# CreateTranscriptionResponseStreamEvent: `transcript.text.delta` (text, with its logprobs when asked),
# `transcript.text.segment` (a diarized segment) and `transcript.text.done` (the whole text, its usage, logprobs and
# languages); OpenAI ends the stream with `data: [DONE]` (2026-11-17), and Ollama reports a failure mid-stream as
# `{"error": …}` (server/transcription.go).

mutable struct _TranscriptStream
    const carry::IOBuffer
    const event::Base.RefValue{String}
    const queue::Vector{Union{String,TranscriptSegment}}   # chunks not yet handed to the callback, in order
    const segments::Vector{TranscriptSegment}
    done::Union{Nothing,Dict{String,Any}}                   # the transcript.text.done event
    error::Union{Nothing,Dict{String,Any}}                  # an error the stream carried
    dropped::Int                                            # payloads that could not be read
end
_TranscriptStream() = _TranscriptStream(IOBuffer(), Ref(""), Union{String,TranscriptSegment}[], TranscriptSegment[],
                                        nothing, nothing, 0)

# One payload. A delta without its text, a segment without a required field and a done event without the text
# throw: the reader drops that payload, and a stream whose done event was dropped ends without its transcript.
function _transcript_event!(s::_TranscriptStream, payload::AbstractString)::Nothing
    payload == "[DONE]" && return nothing
    d = JSON.parse(payload; dicttype=Dict{String,Any})
    d isa Dict{String,Any} || throw(ArgumentError("a transcript event is a JSON object, not a $(typeof(d))"))
    kind = get(d, "type", nothing)
    if haskey(d, "error") || kind == "error"
        s.error = d
    elseif kind == "transcript.text.delta"
        push!(s.queue, _audio_required(d, "delta", String, "transcript.text.delta event"))
    elseif kind == "transcript.text.segment"
        seg = _transcript_segment(d)
        push!(s.segments, seg); push!(s.queue, seg)
    elseif kind == "transcript.text.done"
        _audio_required(d, "text", String, "transcript.text.done event")
        s.done = d
    end
    nothing
end

# One read's events, each handed on as it is read: `flush()` passes what it queued to the callback and tells whether
# the call was stopped, and a stop leaves the events after it unread. True once the stream's outcome — its
# transcript or its error — has been read.
function _transcript_events!(s::_TranscriptStream, chunk::String, flush::Function)::Bool
    for (_, payload) in _sse_events!(s.carry, s.event, chunk)
        try
            _transcript_event!(s, payload)
        catch e
            e isa InterruptException && rethrow()
            Threads.atomic_add!(_SSE_DROPPED_LINES, 1)
            s.dropped += 1
            @debug "transcription SSE: dropped undecodable data payload" payload = String(payload) exception = e
        end
        flush() && return false
        (s.done !== nothing || s.error !== nothing) && return true
    end
    false
end

function _flush_transcript!(cb, s::_TranscriptStream, close_ref)::Nothing
    isnothing(cb) || foreach(x -> cb(x, close_ref), s.queue)
    empty!(s.queue)
    nothing
end

_streamed_transcription(s::_TranscriptStream)::TranscriptionSuccess =
    (done = s.done::Dict{String,Any}; TranscriptionSuccess(text=done["text"], raw=done, segments=copy(s.segments)))

_cancelled_transcription(ctl::_StreamCtl, t0::UInt64)::AudioCallError =
    (c = _stop_cause(ctl, t0); AudioCallError(error=sprint(showerror, c), cause=c))

# The stream driver, the agentic driver's rules on a multipart request (`_respond_stream`): a task whose value is
# the result; retries only before user code runs; a missing first byte within min(remaining, request_timeout) is a
# :request timeout and a byte gap beyond stream_idle_timeout a :stream_idle one (time in the callback is no gap);
# once the outcome is read, teardown noise does not discard it; a stop (the caller's token or the close flag) ends
# the call with UniLMCancelled, and a callback's exception with that exception. Not recorded: a multipart body's
# boundary is random, so no recording could be keyed on it.
function _transcribe_stream(t::TranscriptionRequest, callback, cfg::RequestConfig, t0::UInt64,
                            token::Union{Nothing,CancelToken})
    Threads.@spawn begin
        ctl = _StreamCtl(token, t0)
        link = _on_cancel(() -> cancel!(ctl.stop), token)
        try
            _transcribe_drive(t, callback, cfg, t0, ctl)
        finally
            _off_cancel(token, link)
        end
    end
end

function _transcribe_drive(t::TranscriptionRequest, callback, cfg::RequestConfig, t0::UInt64, ctl::_StreamCtl)
    url = _api_base_url(t.service) * AUDIO_TRANSCRIPTIONS_PATH
    attempt = 1
    while true
        state = _TranscriptStream()
        raw = IOBuffer()   # the reply's bytes: a failure's body
        io_ref = Ref{Any}(nothing)
        guard = Ref{Union{Nothing,_IdleGuard}}(nothing)
        bound = Ref{Union{Nothing,UniLMTimeout}}(nothing)
        cb = isnothing(callback) ? nothing : (x, c) -> _user_call(callback, ctl, guard[], x, c)
        ctx = HTTP.RequestContext()
        body = Ref(UInt8[])
        function consume(io)
            io_ref[] = io
            _with_recorded_deadline(() -> (write(io, body[]); HTTP.closewrite(io); HTTP.startread(io)),
                                    () -> _abort_request_phase(io, ctx), _first_byte_bound(io, cfg, t0), :request, bound)
            guard[] = _idle_guard(() -> close(io), _idle_bound(io, cfg))
            done = false
            while !done && !iscancelled(ctl.stop)
                # At EOF a newline completes a last line the peer did not end.
                at_eof = eof(io)
                chunk = at_eof ? "\n" : String(readavailable(io))
                at_eof || (_touch!(guard[]); print(raw, chunk))
                done = _transcript_events!(state, chunk,
                                           () -> (_flush_transcript!(cb, state, ctl.close); iscancelled(ctl.stop)))
                at_eof && break
            end
            if iscancelled(ctl.stop)
                done || throw(_stop_cause(ctl, t0))
            else
                if !done
                    breach = _exit_breach(guard[], bound, cfg)
                    breach === nothing || throw(breach)
                end
                HTTP.closeread(io)
            end
        end
        try
            # Each attempt reads the file into a form of its own; its boundary names the body's type, as
            # HTTP.request sets it for an HTTP.Form, and the body goes whole, its length known.
            form = HTTP.Form(_transcription_parts(t))
            body[] = read(form)
            headers = [auth_header_multipart(t.service); "Content-Type" => HTTP.content_type(form);
                       "Content-Length" => string(length(body[])); "Accept-Encoding" => "identity"]
            resp = _http_open(consume, "POST", url, headers; cfg, t0, cancel=ctl.stop, context=ctx, decompress=false)
            if resp.status == 200
                isnothing(state.done) || return _streamed_transcription(state)
                isnothing(state.error) ||
                    return AudioFailure(response=JSON.json(state.error), status=nothing, request_id=_get_request_id(resp))
                return AudioCallError(error="the transcription stream ended before its transcript.text.done event " *
                    "(Content-Type $(repr(HTTP.header(resp, "Content-Type", ""))), $(state.dropped) payloads unread)")
            elseif _is_retryable(resp.status, raw) && !ctl.fired && attempt < cfg.max_attempts
                action, delay = _retry_pause(cfg, t0, attempt, resp)
                action === :budget &&
                    return AudioFailure(response=takestring!(raw), status=resp.status, request_id=_get_request_id(resp))
                _cancel_sleep(ctl.stop, delay) && return _cancelled_transcription(ctl, t0)
                attempt += 1
                continue
            end
            return AudioFailure(response=takestring!(raw), status=resp.status, request_id=_get_request_id(resp))
        catch e
            _find_exception(x -> x isa InterruptException, e) !== nothing && rethrow()
            failure = ctl.failure
            if !isnothing(failure)
                thrown = failure.thrown
                _escapes_tool(thrown) && throw(thrown)
                return AudioCallError(error=_error_text(thrown), cause=thrown isa Exception ? thrown : nothing)
            end
            breach = _classify_stream_timeout(e, guard[], cfg, t0)
            stopped = iscancelled(ctl.stop)
            if (stopped || _stream_teardown_noise(e, breach)) && !isnothing(state.done)
                return _streamed_transcription(state)
            end
            stopped && return _cancelled_transcription(ctl, t0)
            breach === nothing || return AudioCallError(error=sprint(showerror, breach), cause=breach)
            u = _unwrap_exception(e)
            bt = bound[]
            mapped = bt !== nothing ? bt : u isa UniLMTimeout ? u :
                     _map_native_timeout(e, cfg, min(_remaining_s(cfg, t0), cfg.request_timeout), t0)
            retry = mapped isa UniLMTimeout ? mapped.phase in (:connect, :request) : _is_transport_error(e)
            if retry && !ctl.fired && attempt < cfg.max_attempts
                action, delay = _retry_pause(cfg, t0, attempt, nothing)
                if action !== :budget
                    _cancel_sleep(ctl.stop, delay) && return _cancelled_transcription(ctl, t0)
                    attempt += 1; continue
                end
            end
            mapped isa UniLMTimeout && return AudioCallError(error=sprint(showerror, mapped), cause=mapped)
            return _callerr(AudioCallError, e)
        finally
            _warn_sse_drops(state.dropped, _transcription_model(t), "transcription")
        end
    end
end

# ─── Gemini transcription (generateContent on gemini-3.5-transcribe) ─────────
# https://ai.google.dev/gemini-api/docs/transcribe. The audio goes inline in one generateContent request whose
# GenerationConfig carries the AudioTranscriptionConfig; each part of the reply carries an AudioTranscription (its
# text, the speaker's label with diarization, the words with word timestamps), one part per speaker turn. The
# Interactions API takes the same options as `transcription_config` and gave the same transcript (2026-11-17), but
# stores the interaction unless told not to and answers words without turns: transcribe uses generateContent.

_default_transcription_model(::Type{GEMINIServiceEndpoint}) = "gemini-3.5-transcribe"

const _GEMINI_TRANSCRIPTION_UNTAKEN = (:prompt, :temperature, :response_format, :include, :chunking_strategy,
                                       :known_speaker_names, :known_speaker_references, :stream)
# "Supply up to 1,000 terms in the `custom_vocabulary` array" (the transcription guide).
const _GEMINI_VOCABULARY_MAX = 1000
function _check_transcription(::Type{GEMINIServiceEndpoint}, t::TranscriptionRequest)
    set = _options_set(t, _GEMINI_TRANSCRIPTION_UNTAKEN)
    isempty(set) || throw(ArgumentError(
        "Gemini's transcription (AudioTranscriptionConfig) takes no $(join(set, ", ")): remove them"))
    words = !isnothing(t.timestamp_granularities)
    words && t.timestamp_granularities != ["word"] && throw(ArgumentError(
        "Gemini timestamps words only: timestamp_granularities = [\"word\"] (got $(repr(t.timestamp_granularities)))"))
    vocabulary = t.keywords
    isnothing(vocabulary) || length(vocabulary) <= _GEMINI_VOCABULARY_MAX || throw(ArgumentError(
        "Gemini takes up to $_GEMINI_VOCABULARY_MAX custom vocabulary terms (got $(length(vocabulary)) keywords)"))
    # "You cannot combine `custom_vocabulary` with speaker diarization or word-level timestamps; the API rejects
    # requests that specify `custom_vocabulary` alongside either feature": answered HTTP 400 "custom_vocabulary is
    # incompatible with diarization." (2026-11-17).
    isnothing(vocabulary) || !(t.diarization || words) || throw(ArgumentError(
        "Gemini rejects custom vocabulary (keywords) beside diarization or word timestamps: drop the keywords, or both"))
    # "Smart transcription (`"smart"`) cannot be combined with `timestamp_granularities` or `diarization_mode`".
    t.mode == "smart" && (t.diarization || words) && throw(ArgumentError(
        "Gemini's smart mode takes neither diarization nor word timestamps: use mode = \"verbatim\" for them"))
    nothing
end

# GenerationConfig.audioTranscriptionConfig (https://ai.google.dev/api/generate-content#AudioTranscriptionConfig):
# languageCodes (BCP-47), customVocabulary, diarization, wordTimestamp, mode (VERBATIM or SMART) and sourceText.
function _gemini_transcription_body(t::TranscriptionRequest)::String
    config = Dict{String,Any}()
    languages = isnothing(t.language) ? t.languages : [t.language]
    isnothing(languages) || (config["languageCodes"] = languages)
    isnothing(t.keywords) || (config["customVocabulary"] = t.keywords)
    t.diarization && (config["diarization"] = true)
    isnothing(t.timestamp_granularities) || (config["wordTimestamp"] = true)
    isnothing(t.mode) || (config["mode"] = uppercase(t.mode))
    isnothing(t.source_text) || (config["sourceText"] = t.source_text)
    audio = Dict{String,Any}("inlineData" => Dict{String,Any}("mimeType" => _mime_for(t.file),
                                                            "data" => base64encode(read(t.file))))
    body = Dict{String,Any}("contents" => [Dict{String,Any}("role" => "user", "parts" => [audio])])
    isempty(config) || (body["generationConfig"] = Dict{String,Any}("audioTranscriptionConfig" => config))
    JSON.json(body)
end

# Retried as Gemini's speech is (a String body goes again as it is).
function _transcribe(::_GeminiTranscriptionWire, t::TranscriptionRequest, cfg::RequestConfig, t0::UInt64, tok)
    url = _gemini_speech_url(t.service, _transcription_model(t))
    resp = _http_with_retries(cfg, t0, "POST", url, auth_header(t.service), _gemini_transcription_body(t); cancel=tok)
    resp.status == 200 || return _failure(AudioFailure, resp)
    _gemini_transcription(resp)
end

# A google.protobuf.Duration in its JSON form, seconds with an "s" suffix ("0.400s", "3s").
function _google_seconds(d::AbstractDict, key::String)::Union{Float64,Nothing}
    v = _audio_field(d, key, String, "word")
    isnothing(v) && return nothing
    @something(endswith(v, "s") ? tryparse(Float64, chop(v)) : nothing,
               throw(ArgumentError("the word's $(repr(key)) is no duration in seconds: $(repr(v))")))
end

# A finished candidate's parts, each an AudioTranscription (a part that holds a thought is skipped). A turn's start
# and end are its first and last word's; the text is the turns' joined with spaces, as Interactions' output_text
# joins them. A candidate that did not finish, or a blocked prompt, is the provider's failure.
function _gemini_transcription(resp::HTTP.Response)::LLMRequestResponse
    d = JSON.parse(resp.body; dicttype=Dict{String,Any})
    cand = _gemini_finished_candidate(d)
    cand === nothing && return _failure(AudioFailure, resp)
    content = _audio_required(cand, "content", AbstractDict, "candidate")
    segments, words = TranscriptSegment[], TranscriptWord[]
    for p in _audio_required(content, "parts", AbstractVector, "candidate's content")
        p isa AbstractDict || throw(ArgumentError("a reply part is a JSON object, not a $(typeof(p))"))
        get(p, "thought", false) === true && continue
        a = _audio_required(p, "audioTranscription", AbstractDict, "reply part")
        speaker = _audio_field(a, "speakerLabel", String, "audioTranscription")
        turn = TranscriptWord[TranscriptWord(text=_audio_required(w, "word", String, "word"),
                                             start_time=_google_seconds(w, "startOffset"),
                                             end_time=_google_seconds(w, "endOffset"), speaker=speaker)
                              for w in _reply_list(a, "words")]
        append!(words, turn)
        push!(segments, TranscriptSegment(text=_audio_required(a, "text", String, "audioTranscription"), speaker=speaker,
            start_time=isempty(turn) ? nothing : first(turn).start_time, end_time=isempty(turn) ? nothing : last(turn).end_time))
    end
    isempty(segments) && throw(ArgumentError("Gemini's transcription reply carried no audioTranscription part"))
    TranscriptionSuccess(text=join((s.text for s in segments), " "), raw=d, segments=segments, words=words)
end

# ─── Translation (OpenAI Realtime translation sessions, Ollama) ──────────────

"""
    translate(path; language="en", source_language=nothing, model=nothing, service=OPENAIServiceEndpoint,
              config=nothing, cancel=nothing) -> TranscriptionSuccess | AudioFailure | AudioCallError

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
has neither OpenAI's Realtime API nor Ollama's translation route, and so does a `source_language`:
the session sets only the target language and has no field for the spoken one
([translation client events](https://developers.openai.com/api/reference/resources/realtime/translation-client-events)).

On [`OllamaEndpoint`](@ref) (Ollama 0.40.12 or later) the file goes as it is to Ollama's
OpenAI-compatible translation route, one multipart request carrying the `model`, the target
`language` as `to_language`, the "ISO 639-1 code of the output language", and `source_language`,
when given, as Ollama's own `language`, "the spoken language, if known"
([OpenAI compatibility](https://docs.ollama.com/api/openai-compatibility)); the translation is the
reply's `text`, and `raw` holds the reply. Ollama decodes a WAV file itself and passes any other
container to the model whole. There is no default model: name an installed model that hears audio,
such as `gemma4:e4b`; a call without one, a blank `language` or `source_language`, and a missing or
empty file throw `ArgumentError` before any request:

```julia
r = translate("memo.wav"; language="fr", service=OllamaEndpoint(), model="gemma4:e4b")
r isa TranscriptionSuccess && println(transcript_text(r))
```

A status Ollama answers is an [`AudioFailure`](@ref) (a language it does not translate into or
from is a 400); transient statuses and transport failures are retried as [`transcribe`](@ref) retries them,
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
audio cannot be resumed.

Pass `cancel::Union{Nothing,CancelToken}` (default: the ambient [`with_cancel`](@ref) token) to make
the call cancellable: once the token is cancelled it returns an `AudioCallError` whose `cause` is a
[`UniLMCancelled`](@ref) — without sending anything when the token was already cancelled, at once
when mid-request (never retried).
"""
function translate(path::AbstractString; language::AbstractString="en",
                   source_language::Union{Nothing,AbstractString}=nothing,
                   model::Union{AbstractString,Nothing}=nothing,
                   service::ServiceEndpointSpec=OPENAIServiceEndpoint,
                   config::Union{Nothing,RequestConfig}=nothing,
                   cancel::Union{Nothing,CancelToken}=nothing)
    _with_resolved_cancel(() -> _translate(service, path, language, source_language, model, config), cancel)
end

# The model of an OpenAI Realtime translation session when the call names none.
const _TRANSLATE_DEFAULT_MODEL = "gpt-realtime-translate"

# OpenAI: a Realtime translation session, on `_TRANSLATE_DEFAULT_MODEL` unless the call names another model.
function _translate(service, path::AbstractString, language::AbstractString,
                    source_language::Union{Nothing,AbstractString}, model::Union{AbstractString,Nothing},
                    config::Union{Nothing,RequestConfig})
    validate_capability(service, :realtime, "translate (OpenAI Realtime API)")
    # The session's `session.update` sets the target (`audio.output.language`) and no spoken language
    # (https://developers.openai.com/api/reference/resources/realtime/translation-client-events).
    isnothing(source_language) || throw(ArgumentError(
        "translate on an OpenAI Realtime translation session takes no source_language: the session sets only the " *
        "target language (audio.output.language) and has no field for the spoken one " *
        "(https://developers.openai.com/api/reference/resources/realtime/translation-client-events); " *
        "omit source_language"))
    m = String(something(model, _TRANSLATE_DEFAULT_MODEL))
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

# Ollama (0.40.12 or later): the file as it is, the model, the target language as `to_language` and the spoken one,
# when given, as `language`, in one multipart request, answered as a transcription is
# (https://docs.ollama.com/api/openai-compatibility). Retried as `transcribe` is, each attempt building its form from
# the file.
function _translate(service::OllamaEndpoint, path::AbstractString, language::AbstractString,
                    source_language::Union{Nothing,AbstractString}, model::Union{AbstractString,Nothing},
                    config::Union{Nothing,RequestConfig})
    validate_capability(service, :audio, "translate (Ollama translation route)")
    (isnothing(model) || isempty(strip(model))) && throw(ArgumentError(
        "Ollama has no default translation model: name an installed model that hears audio, e.g. " *
        "model = \"gemma4:e4b\""))
    fields = Pair{String,Any}["model" => String(model), "to_language" => _translation_language(language)]
    if !isnothing(source_language)
        spoken = strip(source_language)
        isempty(spoken) && throw(ArgumentError(
            "translate's source_language is blank: name the spoken language, e.g. source_language = \"fr\", or omit it"))
        push!(fields, "language" => String(spoken))
    end
    isfile(path) || throw(ArgumentError("file not found: $path"))
    filesize(path) > 0 || throw(ArgumentError(
        "$(basename(path)) is empty, and Ollama answers an empty upload with HTTP 400 \"audio file is empty\": " *
        "translate a recording"))
    file = String(path)
    form() = HTTP.Form(Pair{String,Any}["file" => HTTP.Multipart(basename(file), IOBuffer(read(file)), _mime_for(file)),
                                        fields...])
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
