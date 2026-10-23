# ============================================================================
# Audio — speech (OpenAI Realtime sessions, Gemini TTS), transcription (OpenAI
# Audio API, Ollama) and translation (OpenAI Realtime translation sessions).
# The Realtime session drivers live in realtime.jl.
# ============================================================================

"Audio error result: the HTTP `status` (`nothing` when a Realtime session reported the failure as an event, which carries no HTTP status), the raw `response` (the reply body, the Realtime event's JSON, or for a refused WebSocket upgrade HTTP.jl's report of its status) and the `request_id` the service sent (`x-request-id`/`request-id` header), if any."
@kwdef struct AudioFailure <: LLMRequestResponse; response::String; status::Union{Int,Nothing}; request_id::Union{String,Nothing} = nothing; end
"Audio call that produced no usable reply (transport failure, timeout, cancel, a session that closed before its result, or a reply that could not be decoded); `cause` is the underlying exception — a [`UniLMTimeout`](@ref) for a timeout, a [`UniLMCancelled`](@ref) for a cancel, and, when the server closed a Realtime session, the `HTTP.WebSockets.WebSocketError` that carries its status and reason, which `error` names too."
@kwdef struct AudioCallError <: LLMRequestResponse; error::String; status::Union{Int,Nothing} = nothing; cause::Union{Nothing,Exception} = nothing; end

# ─── Speech ──────────────────────────────────────────────────────────────────

# Speech from either service is 16-bit little-endian mono PCM at 24 kHz: the one rate of the
# Realtime API's PCM output, and the rate UniLM asks Gemini TTS for.
const _SPEECH_RATE = 24_000
const _SPEECH_FORMATS = ("wav", "pcm")

# How a service speaks: an OpenAI Realtime session (every service but Gemini's; `_realtime_ws_url`
# refuses all but OPENAIServiceEndpoint) or Gemini TTS through generateContent.
abstract type _SpeechWire end
struct _RealtimeSpeechWire <: _SpeechWire end
struct _GeminiSpeechWire <: _SpeechWire end
_speech_wire(_) = _RealtimeSpeechWire()
_speech_wire(::Type{GEMINIServiceEndpoint}) = _GeminiSpeechWire()
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
# Gemini 3.8 TTS's 30 prebuilt voices. It has no speed parameter: the pace is part of a text's
# speech style. https://ai.google.dev/gemini-api/docs/speech-generation
const _GEMINI_SPEECH = _SpeechRules("Gemini TTS", "gemini-3.8-flash-tts", "Kore",
    ["Zephyr", "Puck", "Charon", "Kore", "Fenrir", "Leda", "Orus", "Aoede", "Callirrhoe", "Autonoe",
     "Enceladus", "Iapetus", "Umbriel", "Algieba", "Despina", "Erinome", "Algenib", "Rasalgethi",
     "Laomedeia", "Achernar", "Alnilam", "Schedar", "Gacrux", "Pulcherrima", "Achird", "Zubenelgenubi",
     "Vindemiatrix", "Sadachbia", "Sadaltager", "Sulafat"], nothing,
    "https://ai.google.dev/gemini-api/docs/speech-generation")
_speech_rules(::_RealtimeSpeechWire) = _REALTIME_SPEECH
_speech_rules(::_GeminiSpeechWire) = _GEMINI_SPEECH

function _speech_speed(r::_SpeechRules, speed::Real)::Float64
    s = Float64(speed)
    range = r.speed
    range === nothing && throw(ArgumentError(
        "$(r.provider) has no speed control: describe the pace in `instructions` instead, e.g. " *
        "instructions = \"speaking slowly\" ($(r.docs))"))
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
| `GEMINIServiceEndpoint` | Gemini TTS (`generateContent`) | `gemini-3.8-flash-tts` | `Kore` | the 30 prebuilt voices | none |

Both services produce 16-bit mono PCM at 24 kHz: `response_format` `"wav"` returns it as a WAV file,
`"pcm"` as the raw samples. `instructions` direct the delivery (tone, pace, accent): on OpenAI they
follow the session's instruction to read `input` word for word, on Gemini they are the text's speech
style (`speechMetadata.style`). Gemini TTS has no speed parameter: describe the pace in
`instructions` (`"speaking slowly"`).

Throws `ArgumentError`, before any request, for an empty `input`, a voice the service does not have,
a `response_format` other than `"wav"` or `"pcm"`, a `speed` out of range, and any `speed` on Gemini;
and for `GEMINIOpenAIServiceEndpoint`, which has no speech route, naming `GEMINIServiceEndpoint`.
Voices: [OpenAI Realtime](https://developers.openai.com/api/reference/resources/realtime/client-events),
[Gemini TTS](https://ai.google.dev/gemini-api/docs/speech-generation#voices).
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
        rules = _speech_rules(_speech_wire(service))
        isempty(strip(input)) && throw(ArgumentError("speak needs text to read: input is empty"))
        v = something(voice, rules.voice)
        v in rules.voices || throw(ArgumentError(
            "$(rules.provider) voices are $(join(rules.voices, ", ")) (got $(repr(v))): $(rules.docs)"))
        response_format in _SPEECH_FORMATS || throw(ArgumentError(
            "response_format must be \"wav\" (a WAV file) or \"pcm\" (raw 16-bit mono PCM at 24 kHz); " *
            "got $(repr(response_format))"))
        new(service, something(model, rules.model), input, v, response_format,
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
verbatim.

Returns [`SpeechSuccess`](@ref): a WAV file (`response_format = "wav"`) or the raw samples (`"pcm"`).
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

Pass `config::Union{Nothing,RequestConfig}` to bound the call (a single attempt; `max_attempts` does
not apply). On OpenAI, `total_deadline` bounds the whole call whatever the socket does,
`connect_timeout` the WebSocket open, and `stream_idle_timeout` every send and every wait for the
next event; on Gemini it bounds the one HTTP request. Both observe the ambient [`CancelToken`](@ref)
([`with_cancel`](@ref)): a token already cancelled sends nothing, and a cancel ends the call at once,
with an `AudioCallError` whose `cause` is a [`UniLMCancelled`](@ref). Speech is not priced:
[`token_usage`](@ref) and [`estimated_cost`](@ref) throw for a `SpeechSuccess`.
"""
function speak(s::SpeechRequest; config::Union{Nothing,RequestConfig}=nothing)
    wire = _speech_wire(s.service)
    target = _speech_target(wire, s)    # the route is checked before any I/O
    cfg = _resolve_config(config); t0 = time_ns()
    try
        _speak(wire, s, target, cfg, t0)
    catch e
        e isa InterruptException && rethrow()
        _callerr(AudioCallError, e)
    end
end
speak(input::AbstractString; service::ServiceEndpointSpec=OPENAIServiceEndpoint,
      model::Union{AbstractString,Nothing}=nothing, voice::Union{AbstractString,Nothing}=nothing,
      response_format::AbstractString="wav", speed::Union{Real,Nothing}=nothing,
      instructions::Union{AbstractString,Nothing}=nothing, config::Union{Nothing,RequestConfig}=nothing) =
    speak(SpeechRequest(service, model, input, voice, response_format, speed, instructions); config)

# Where a request goes; throws ArgumentError, before any I/O, for a service the wire cannot reach.
function _speech_target(::_RealtimeSpeechWire, s::SpeechRequest)::String
    validate_capability(s.service, :realtime, "speak (OpenAI Realtime API)")
    _realtime_url(s.service, s.model)
end
_speech_target(::_GeminiSpeechWire, s::SpeechRequest)::String = _gemini_speech_url(s.service, s.model)

function _speak(::_RealtimeSpeechWire, s::SpeechRequest, url::String, cfg::RequestConfig, t0::UInt64)
    r = _realtime_call(session -> _realtime_speak(session, s), url, s.model, s.service, cfg, t0)
    r isa LLMRequestResponse ? r : _realtime_closed("response.done")
end

function _speak(::_GeminiSpeechWire, s::SpeechRequest, url::String, cfg::RequestConfig, t0::UInt64)
    resp = _http("POST", url, auth_header(s.service), _gemini_speech_body(s); cfg, remaining=_remaining_s(cfg, t0))
    resp.status == 200 || return _failure(AudioFailure, resp)
    cand = _gemini_finished_candidate(JSON.parse(resp.body; dicttype=Dict{String,Any}))
    cand === nothing && return _failure(AudioFailure, resp)
    _speech_success(s, _gemini_speech_pcm(cand))
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
    JSON.json(Dict(
        :contents => [Dict(:role => "user", :parts => [part])],
        :generationConfig => Dict(
            :responseModalities => ["AUDIO"],
            :speechConfig => Dict(:voiceConfig => Dict(:prebuiltVoiceConfig => Dict(:voiceName => s.voice))),
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
throws `ArgumentError` before any request. Optional
`languages`, `keywords`, `prompt`, `response_format`, and `temperature` refine decoding.
For `gpt-transcribe`, a legacy singular `language` is translated to `languages`.
Other models retain their singular `language` field. Never set both forms.
`response_format` is one of `json`, `text`, `srt`, `verbose_json`, `vtt`,
`diarized_json`, and `temperature` lies in `0 … 1`; anything else is an `ArgumentError`.
On OpenAI, `srt`, `vtt` and `verbose_json` (subtitles and timestamps) are documented only
for `whisper-1`, and `diarized_json` (speaker labels) only for `gpt-4o-transcribe-diarize`:
deprecated models that shut down on February 26, 2027
([speech to text](https://developers.openai.com/api/docs/guides/speech-to-text)).
Ollama's `/v1/audio/transcriptions` reads one `language` and the `prompt`, transcribes at
temperature 0, and is asked for JSON only, the one form in which it reports a failure that
arrives after its reply has begun (as `text` that failure reads as an empty transcript): a
request to it that sets `temperature`, `languages`, `keywords`, or a `response_format`
other than `json` throws `ArgumentError` before any request; read the text with
[`transcript_text`](@ref).
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
        isnothing(temperature) || 0 <= temperature <= 1 ||
            throw(ArgumentError("temperature must be between 0 and 1; got $temperature"))
        new(service, file, model, language, prompt, response_format, temperature, languages, keywords)
    end
end

# The model a transcription names, else the service's default: `gpt-transcribe` for OpenAI and the
# OpenAI-compatible services; none for Ollama, whose models are the caller's to name.
_default_transcription_model(_) = "gpt-transcribe"
_default_transcription_model(::OllamaEndpoint) = throw(ArgumentError(
    "Ollama has no default transcription model: name an installed model that hears audio, e.g. " *
    "model = \"gemma4:e4b\""))
# `@something` evaluates the default only when no model is named.
_transcription_model(t::TranscriptionRequest)::String = @something(t.model, _default_transcription_model(t.service))

# What a service's transcription route cannot carry, refused before any request.
_check_transcription(_, ::TranscriptionRequest) = nothing
function _check_transcription(::OllamaEndpoint, t::TranscriptionRequest)
    for (f, rule) in ((:temperature, "runs at temperature 0 and takes no `temperature`: remove it"),
                      (:languages, "takes one `language`: remove `languages`"),
                      (:keywords, "takes no `keywords`: remove them"))
        isnothing(getfield(t, f)) || throw(ArgumentError("Ollama's transcription $rule"))
    end
    # Ollama writes the transcript only when the model's chat stream ends: a failure after the reply has
    # begun leaves a 200 with an empty body (Ollama 0.40.6, middleware/openai.go TranscriptionWriter), which
    # its JSON form reports and "text" would read as an empty transcript. Any other format is answered as
    # JSON too, so JSON is the one form asked for.
    fmt = t.response_format
    fmt == "text" && throw(ArgumentError(
        "Ollama's transcription reports a failure that arrives after its reply has begun only as JSON, and " *
        "\"text\" would read it as an empty transcript: leave response_format unset (JSON) and read the " *
        "text with transcript_text"))
    isnothing(fmt) || fmt == "json" || throw(ArgumentError(
        "Ollama transcribes to JSON only (got $(repr(fmt))): leave response_format unset and read the text " *
        "with transcript_text"))
    nothing
end

# Whether a 200 transcription reply is the JSON object `{"text": …}` rather than the bare text.
_transcription_json(_, ::TranscriptionRequest, resp::HTTP.Response)::Bool =
    occursin("application/json", HTTP.header(resp, "Content-Type", ""))
# Ollama is asked for JSON only, and answers it under the `application/x-ndjson` header of the chat
# stream it transcribes with: that header is sent before the JSON is written (Ollama 0.40.6).
_transcription_json(::OllamaEndpoint, ::TranscriptionRequest, ::HTTP.Response)::Bool = true

"Successful [`transcribe`](@ref)/[`translate`](@ref) result; `text` holds the transcript or translation (via [`transcript_text`](@ref)), `raw` the parsed JSON a transcription reply carried (`nothing` for a translation)."
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

On Ollama, an installed model that hears audio transcribes through Ollama's
OpenAI-compatible `/v1/audio/transcriptions` route. There is no default model, and what
that route cannot carry — `temperature`, `languages`, `keywords`, a `response_format`
other than `json` (`text` included: a failure after the reply has begun would read as an
empty transcript) — throws `ArgumentError` before any request:

```julia
r = transcribe("memo.wav"; service=OllamaEndpoint(), model="gemma4:e4b", language="en")
r isa TranscriptionSuccess && println(transcript_text(r))
```

Pass `config::Union{Nothing,RequestConfig}` to override the timeout budget for this call (a single bounded attempt; `max_attempts` does not apply).
"""
function transcribe(t::TranscriptionRequest; config::Union{Nothing,RequestConfig}=nothing)
    # An endpoint that declares `:audio` transcribes too (`has_capability`), as does one that declares
    # only `:transcription` (Ollama).
    validate_capability(t.service, :transcription, "Audio API")
    # Local validation throws before the request (inside the `try` it would become an AudioCallError).
    _check_transcription(t.service, t)
    _transcription_model(t)
    cfg = _resolve_config(config); t0 = time_ns()
    try
        parts = _transcription_parts(t)
        resp = _http("POST", _api_base_url(t.service) * AUDIO_TRANSCRIPTIONS_PATH, auth_header_multipart(t.service),
            HTTP.Form(parts); cfg, remaining=_remaining_s(cfg, t0))
        if resp.status == 200
            if _transcription_json(t.service, t, resp)
                d = JSON.parse(resp.body; dicttype=Dict{String,Any})
                TranscriptionSuccess(text=get(d, "text", ""), raw=d)
            else
                TranscriptionSuccess(text=String(resp.body))
            end
        else
            _failure(AudioFailure, resp)
        end
    catch e
        e isa InterruptException && rethrow()
        _callerr(AudioCallError, e)
    end
end
transcribe(path::String; model::Union{String,Nothing}=nothing, service::ServiceEndpointSpec=OPENAIServiceEndpoint, config::Union{Nothing,RequestConfig}=nothing, kwargs...) =
    transcribe(TranscriptionRequest(; service=service, file=path, model=model, kwargs...); config=config)

# ─── Translation (OpenAI Realtime translation sessions) ──────────────────────

"""
    translate(path; language="en", model="gpt-realtime-translate", service=OPENAIServiceEndpoint,
              config=nothing) -> TranscriptionSuccess | AudioFailure | AudioCallError

Translate the speech in the WAV file at `path` into `language` and return the translation as text:
a [`TranscriptionSuccess`](@ref), read with [`transcript_text`](@ref).

The audio goes to an OpenAI Realtime translation session
(`wss://api.openai.com/v1/realtime/translations`,
[Realtime translation](https://developers.openai.com/api/docs/guides/realtime-translation)): UniLM
sets the target language (the session's `audio.output.language`, such as `"fr"` or `"es"`; English
unless named), streams the audio in 200 ms frames, closes the session and joins the translated
transcript. The session also returns translated speech, which `translate` does not keep. Speech
already in the target language can come back as empty text: an English clip translated into English
did.

`path` must be a 16-bit PCM WAV file, at any sample rate and channel count: a translation session
takes 24 kHz mono PCM16 only, so UniLM mixes the channels down and resamples by linear interpolation
(speech-grade: there is no anti-aliasing filter). Any other file throws `ArgumentError` before any
I/O, naming the conversion; so do an empty file, a blank `language` and a service without OpenAI's
Realtime API.

A failure the provider reports is an [`AudioFailure`](@ref): a Realtime `error` event, for example
for a language it does not translate into (`status = nothing`, the event's JSON as `response`), read
even when the server closes the session during the upload, or a WebSocket upgrade the server refuses
(a bad key is a 401). A session that closes before `session.closed` (`error` names the close's status
and reason), a timeout and a cancel are an [`AudioCallError`](@ref).
`config::Union{Nothing,RequestConfig}` bounds the call as for [`speak`](@ref): `total_deadline` the
whole call whatever the socket does, `connect_timeout` the open, `stream_idle_timeout` every send and
every wait for the next event; a single attempt. The ambient [`CancelToken`](@ref)
([`with_cancel`](@ref)) is observed: a token already cancelled sends nothing, and a cancel ends the
call at once, with an `AudioCallError` whose `cause` is a [`UniLMCancelled`](@ref).
"""
function translate(path::AbstractString; language::AbstractString="en",
                   model::AbstractString="gpt-realtime-translate",
                   service::ServiceEndpointSpec=OPENAIServiceEndpoint,
                   config::Union{Nothing,RequestConfig}=nothing)
    validate_capability(service, :realtime, "translate (OpenAI Realtime API)")
    m = String(model)
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
