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
    TranscriptionRequest(; file, model="gpt-transcribe", service=OPENAIServiceEndpoint)

An audio transcription/translation request. `file` is a path on disk; optional
`languages`, `keywords`, `prompt`, `response_format`, and `temperature` refine decoding.
For `gpt-transcribe`, a legacy singular `language` is translated to `languages`.
Other models retain their singular `language` field. Never set both forms.
`response_format` is one of `json`, `text`, `srt`, `verbose_json`, `vtt`,
`diarized_json`, and `temperature` lies in `0 … 1`; anything else is an `ArgumentError`.
Pass to [`transcribe`](@ref) or [`translate`](@ref).
"""
@kwdef struct TranscriptionRequest
    service::ServiceEndpointSpec = OPENAIServiceEndpoint
    file::String
    model::String = "gpt-transcribe"
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
    parts = Pair{String,Any}[
        "file" => HTTP.Multipart(basename(t.file), IOBuffer(read(t.file)), _mime_for(t.file)),
        "model" => t.model]
    languages = t.languages
    if !isnothing(t.language)
        if _model_family(t.model, "gpt-transcribe")
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

function _transcribe(t::TranscriptionRequest, path::String; config::Union{Nothing,RequestConfig}=nothing)
    validate_capability(t.service, :audio, "Audio API")
    cfg = _resolve_config(config); t0 = time_ns()
    try
        parts = _transcription_parts(t)
        resp = _http("POST", _api_base_url(t.service) * path, auth_header_multipart(t.service),
            HTTP.Form(parts); cfg, remaining=_remaining_s(cfg, t0))
        if resp.status == 200
            if occursin("application/json", HTTP.header(resp, "Content-Type", ""))
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
    transcribe(t::TranscriptionRequest) / transcribe(path; model="gpt-transcribe", kwargs...)

Transcribe audio to text in the source language. Returns `TranscriptionSuccess`
(`.text`, via [`transcript_text`](@ref)), `AudioFailure`, or `AudioCallError`.

Pass `config::Union{Nothing,RequestConfig}` to override the timeout budget for this call (a single bounded attempt; `max_attempts` does not apply).
"""
transcribe(t::TranscriptionRequest; config::Union{Nothing,RequestConfig}=nothing) = _transcribe(t, AUDIO_TRANSCRIPTIONS_PATH; config=config)
transcribe(path::String; model::String="gpt-transcribe", service::ServiceEndpointSpec=OPENAIServiceEndpoint, config::Union{Nothing,RequestConfig}=nothing, kwargs...) =
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
    _transcribe(t, AUDIO_TRANSLATIONS_PATH; config=config)
end
translate(path::String; model::String="whisper-1", service::ServiceEndpointSpec=OPENAIServiceEndpoint, config::Union{Nothing,RequestConfig}=nothing, kwargs...) =
    translate(TranscriptionRequest(; service=service, file=path, model=model, kwargs...); config=config)
