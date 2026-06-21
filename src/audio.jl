# ============================================================================
# OpenAI Audio API — text-to-speech (binary out) + transcription / translation.
# ============================================================================

@kwdef struct AudioFailure <: LLMRequestResponse; response::String; status::Int; end
@kwdef struct AudioCallError <: LLMRequestResponse; error::String; status::Union{Int,Nothing} = nothing; end
_audio_err(e) = AudioCallError(error=string(e), status=(hasproperty(e, :status) ? e.status : nothing))

# ─── Text-to-speech (JSON request, binary response) ──────────────────────────

@kwdef struct SpeechRequest
    service::ServiceEndpointSpec = OPENAIServiceEndpoint
    model::String = "gpt-4o-mini-tts"
    input::String
    voice::String = "alloy"
    response_format::Union{String,Nothing} = nothing   # mp3|opus|aac|flac|wav|pcm
    speed::Union{Float64,Nothing} = nothing
    instructions::Union{String,Nothing} = nothing
end
function JSON.lower(s::SpeechRequest)
    d = Dict{Symbol,Any}(:model => s.model, :input => s.input, :voice => s.voice)
    !isnothing(s.response_format) && (d[:response_format] = s.response_format)
    !isnothing(s.speed) && (d[:speed] = s.speed)
    !isnothing(s.instructions) && (d[:instructions] = s.instructions)
    return d
end

@kwdef struct SpeechSuccess <: LLMRequestResponse
    audio::Vector{UInt8}
    content_type::String = ""
end

"""
    speak(s::SpeechRequest) -> LLMRequestResponse
    speak(input; voice="alloy", model="gpt-4o-mini-tts", service=OPENAIServiceEndpoint, kwargs...)

Synthesize speech. On success returns `SpeechSuccess` with raw audio bytes; otherwise
`AudioFailure`/`AudioCallError`. Use [`save_audio`](@ref) to write the bytes to disk.
"""
function speak(s::SpeechRequest)
    validate_capability(s.service, :audio, "Audio API")
    try
        resp = HTTP.post(_api_base_url(s.service) * AUDIO_SPEECH_PATH, body=JSON.json(s),
            headers=auth_header(s.service); status_exception=false)
        resp.status == 200 ?
            SpeechSuccess(audio=Vector{UInt8}(resp.body), content_type=HTTP.header(resp, "Content-Type", "")) :
            AudioFailure(response=String(resp.body), status=resp.status)
    catch e
        _audio_err(e)
    end
end
speak(input::String; voice::String="alloy", model::String="gpt-4o-mini-tts",
    service::ServiceEndpointSpec=OPENAIServiceEndpoint, kwargs...) =
    speak(SpeechRequest(; service=service, model=model, input=input, voice=voice, kwargs...))

"""
    save_audio(r::SpeechSuccess, path) -> path
"""
function save_audio(r::SpeechSuccess, path::String)
    open(io -> write(io, r.audio), path, "w")
    path
end

# ─── Transcription / translation (multipart upload → text or JSON) ───────────

@kwdef struct TranscriptionRequest
    service::ServiceEndpointSpec = OPENAIServiceEndpoint
    file::String
    model::String = "gpt-4o-transcribe"
    language::Union{String,Nothing} = nothing
    prompt::Union{String,Nothing} = nothing
    response_format::Union{String,Nothing} = nothing
    temperature::Union{Float64,Nothing} = nothing
    function TranscriptionRequest(service, file, model, language, prompt, response_format, temperature)
        isfile(file) || throw(ArgumentError("file not found: $file"))
        new(service, file, model, language, prompt, response_format, temperature)
    end
end

@kwdef struct TranscriptionSuccess <: LLMRequestResponse
    text::String
    raw::Union{Dict{String,Any},Nothing} = nothing
end
transcript_text(r::TranscriptionSuccess) = r.text

function _transcribe(t::TranscriptionRequest, path::String)
    validate_capability(t.service, :audio, "Audio API")
    try
        parts = Pair{String,Any}[
            "file" => HTTP.Multipart(basename(t.file), IOBuffer(read(t.file)), _mime_for(t.file)),
            "model" => t.model]
        !isnothing(t.language) && push!(parts, "language" => t.language)
        !isnothing(t.prompt) && push!(parts, "prompt" => t.prompt)
        !isnothing(t.response_format) && push!(parts, "response_format" => t.response_format)
        !isnothing(t.temperature) && push!(parts, "temperature" => string(t.temperature))
        resp = HTTP.post(_api_base_url(t.service) * path, auth_header_multipart(t.service), HTTP.Form(parts); status_exception=false)
        if resp.status == 200
            if occursin("application/json", HTTP.header(resp, "Content-Type", ""))
                d = JSON.parse(resp.body; dicttype=Dict{String,Any})
                TranscriptionSuccess(text=get(d, "text", ""), raw=d)
            else
                TranscriptionSuccess(text=String(resp.body))
            end
        else
            AudioFailure(response=String(resp.body), status=resp.status)
        end
    catch e
        _audio_err(e)
    end
end

"""
    transcribe(t::TranscriptionRequest) / transcribe(path; model="gpt-4o-transcribe", kwargs...)

Transcribe audio to text in the source language. Returns `TranscriptionSuccess`
(`.text`, via [`transcript_text`](@ref)), `AudioFailure`, or `AudioCallError`.
"""
transcribe(t::TranscriptionRequest) = _transcribe(t, AUDIO_TRANSCRIPTIONS_PATH)
transcribe(path::String; model::String="gpt-4o-transcribe", service::ServiceEndpointSpec=OPENAIServiceEndpoint, kwargs...) =
    transcribe(TranscriptionRequest(; service=service, file=path, model=model, kwargs...))

"""
    translate(t::TranscriptionRequest) / translate(path; model="whisper-1", kwargs...)

Translate audio into English text.
"""
translate(t::TranscriptionRequest) = _transcribe(t, AUDIO_TRANSLATIONS_PATH)
translate(path::String; model::String="whisper-1", service::ServiceEndpointSpec=OPENAIServiceEndpoint, kwargs...) =
    translate(TranscriptionRequest(; service=service, file=path, model=model, kwargs...))
