# ============================================================================
# OpenAI Audio API — text-to-speech (binary out) + transcription / translation.
# ============================================================================

"Audio API error result: HTTP `status`, the raw `response` body, and the `request_id` the service sent (`x-request-id`/`request-id` header), if any."
@kwdef struct AudioFailure <: LLMRequestResponse; response::String; status::Int; request_id::Union{String,Nothing} = nothing; end
"Audio API call that produced no usable reply (transport failure, timeout, or a 200 that could not be decoded); `cause` is the underlying exception — a [`UniLMTimeout`](@ref) for a timeout."
@kwdef struct AudioCallError <: LLMRequestResponse; error::String; status::Union{Int,Nothing} = nothing; cause::Union{Nothing,Exception} = nothing; end

# ─── Text-to-speech (JSON request, binary response) ──────────────────────────

"""
    SpeechRequest(; input, voice="alloy", model="gpt-4o-mini-tts", service=OPENAIServiceEndpoint)

A text-to-speech request. `input` is the text to synthesize; `voice` selects the
speaker; optional `response_format` (`mp3`|`opus`|`aac`|`flac`|`wav`|`pcm`),
`speed` (`0.25` to `4.0`; outside that range an `ArgumentError`), and `instructions`
tune the output. Pass to [`speak`](@ref).
"""
@kwdef struct SpeechRequest
    service::ServiceEndpointSpec = OPENAIServiceEndpoint
    model::String = "gpt-4o-mini-tts"
    input::String
    voice::String = "alloy"
    response_format::Union{String,Nothing} = nothing   # mp3|opus|aac|flac|wav|pcm
    speed::Union{Float64,Nothing} = nothing
    instructions::Union{String,Nothing} = nothing
    function SpeechRequest(service, model, input, voice, response_format, speed, instructions)
        isnothing(speed) || 0.25 <= speed <= 4.0 ||
            throw(ArgumentError("speed must be between 0.25 and 4.0; got $speed"))
        new(service, model, input, voice, response_format, speed, instructions)
    end
end
function JSON.lower(s::SpeechRequest)
    d = Dict{Symbol,Any}(:model => s.model, :input => s.input, :voice => s.voice)
    !isnothing(s.response_format) && (d[:response_format] = s.response_format)
    !isnothing(s.speed) && (d[:speed] = s.speed)
    !isnothing(s.instructions) && (d[:instructions] = s.instructions)
    return d
end

"Successful [`speak`](@ref) result; `audio` holds the raw audio bytes, `content_type` the MIME type. Save with [`save_audio`](@ref)."
@kwdef struct SpeechSuccess <: LLMRequestResponse
    audio::Vector{UInt8}
    content_type::String = ""
end

"""
    speak(s::SpeechRequest) -> LLMRequestResponse
    speak(input; voice="alloy", model="gpt-4o-mini-tts", service=OPENAIServiceEndpoint, kwargs...)

Synthesize speech. On success returns `SpeechSuccess` with raw audio bytes; otherwise
`AudioFailure`/`AudioCallError`. Use [`save_audio`](@ref) to write the bytes to disk.

Pass `config::Union{Nothing,RequestConfig}` to override the timeout budget for this call (a single bounded attempt; `max_attempts` does not apply).
"""
function speak(s::SpeechRequest; config::Union{Nothing,RequestConfig}=nothing)
    validate_capability(s.service, :audio, "Audio API")
    cfg = _resolve_config(config); t0 = time_ns()
    try
        resp = _http("POST", _api_base_url(s.service) * AUDIO_SPEECH_PATH, auth_header(s.service),
            JSON.json(s); cfg, remaining=_remaining_s(cfg, t0))
        resp.status == 200 ?
            SpeechSuccess(audio=Vector{UInt8}(resp.body), content_type=HTTP.header(resp, "Content-Type", "")) :
            _failure(AudioFailure, resp)
    catch e
        e isa InterruptException && rethrow()
        _callerr(AudioCallError, e)
    end
end
speak(input::String; voice::String="alloy", model::String="gpt-4o-mini-tts",
    service::ServiceEndpointSpec=OPENAIServiceEndpoint, config::Union{Nothing,RequestConfig}=nothing, kwargs...) =
    speak(SpeechRequest(; service=service, model=model, input=input, voice=voice, kwargs...); config=config)

"""
    save_audio(r::SpeechSuccess, path) -> path

Write the synthesized audio bytes to `path`, atomically: the bytes go to a temporary
file in the same directory, which is then renamed over `path`, so a failed write leaves
any existing file intact. An existing file keeps its permission bits. A symlink is written
through — a dangling one creates its target; a directory throws `ArgumentError`.
"""
save_audio(r::SpeechSuccess, path::String) = _atomic_write(path, r.audio)

# ─── Transcription / translation (multipart upload → text or JSON) ───────────

const _TRANSCRIPTION_FORMATS = ("json", "text", "srt", "verbose_json", "vtt", "diarized_json")

"""
    TranscriptionRequest(; file, model=nothing, service=OPENAIServiceEndpoint)

An audio transcription/translation request. `file` is a path on disk. `model` left
`nothing` is the service's default: `gpt-transcribe` for OpenAI; Ollama has none, so a
request to it names an installed model that hears audio (e.g. `model="gemma4:e4b"`) or
throws `ArgumentError` before any request. Optional
`languages`, `keywords`, `prompt`, `response_format`, and `temperature` refine decoding.
For `gpt-transcribe`, a legacy singular `language` is translated to `languages`.
Other models retain their singular `language` field. Never set both forms.
`response_format` is one of `json`, `text`, `srt`, `verbose_json`, `vtt`,
`diarized_json`, and `temperature` lies in `0 … 1`; anything else is an `ArgumentError`.
Ollama's `/v1/audio/transcriptions` reads one `language` and the `prompt`, transcribes at
temperature 0, and is asked for JSON only, the one form in which it reports a failure that
arrives after its reply has begun (as `text` that failure reads as an empty transcript): a
request to it that sets `temperature`, `languages`, `keywords`, or a `response_format`
other than `json` throws `ArgumentError` before any request; read the text with
[`transcript_text`](@ref).
Pass to [`transcribe`](@ref) or [`translate`](@ref).
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

"Successful [`transcribe`](@ref)/[`translate`](@ref) result; `text` holds the transcript (via [`transcript_text`](@ref)), `raw` the parsed JSON when the API returns it."
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

function _transcribe(t::TranscriptionRequest, path::String, cap::Symbol;
                     config::Union{Nothing,RequestConfig}=nothing)
    # An endpoint that declares `:audio` transcribes too (`has_capability`), while one that declares
    # only `:transcription` (Ollama) neither speaks nor translates.
    validate_capability(t.service, cap, "Audio API")
    # Local validation throws before the request (inside the `try` it would become an AudioCallError).
    _check_transcription(t.service, t)
    _transcription_model(t)
    cfg = _resolve_config(config); t0 = time_ns()
    try
        parts = _transcription_parts(t)
        resp = _http("POST", _api_base_url(t.service) * path, auth_header_multipart(t.service),
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
transcribe(t::TranscriptionRequest; config::Union{Nothing,RequestConfig}=nothing) =
    _transcribe(t, AUDIO_TRANSCRIPTIONS_PATH, :transcription; config=config)
transcribe(path::String; model::Union{String,Nothing}=nothing, service::ServiceEndpointSpec=OPENAIServiceEndpoint, config::Union{Nothing,RequestConfig}=nothing, kwargs...) =
    transcribe(TranscriptionRequest(; service=service, file=path, model=model, kwargs...); config=config)

"""
    translate(t::TranscriptionRequest) / translate(path; model="whisper-1", kwargs...)

Translate audio into English text. The translations endpoint takes no `languages` or
`keywords` hints and has no `diarized_json` output, so a request carrying any of them
throws `ArgumentError` before any network I/O.

Pass `config::Union{Nothing,RequestConfig}` to override the timeout budget for this call (a single bounded attempt; `max_attempts` does not apply).
"""
function translate(t::TranscriptionRequest; config::Union{Nothing,RequestConfig}=nothing)
    isnothing(t.languages) && isnothing(t.keywords) ||
        throw(ArgumentError("translations take no `languages` or `keywords`; the output is always English"))
    t.response_format == "diarized_json" &&
        throw(ArgumentError("translations do not offer response_format \"diarized_json\""))
    _transcribe(t, AUDIO_TRANSLATIONS_PATH, :audio; config=config)
end
translate(path::String; model::String="whisper-1", service::ServiceEndpointSpec=OPENAIServiceEndpoint, config::Union{Nothing,RequestConfig}=nothing, kwargs...) =
    translate(TranscriptionRequest(; service=service, file=path, model=model, kwargs...); config=config)
