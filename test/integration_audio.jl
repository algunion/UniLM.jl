# ─── Speech, translation and transcription integration tests (live) ──────────
# Requires UNILM_LIVE=1 and OPENAI_API_KEY (OpenAI) / GEMINI_API_KEY (Gemini). Each run
# costs about a cent: one short gpt-realtime-2.1-mini speech response, one
# gemini-3.8-flash-tts reply, and 1.5 s of audio translated and transcribed.

const _IA_WAV = joinpath(@__DIR__, "fixtures", "ollama", "secret_word.wav")   # "The secret word is banana."

# The canonical 44-byte header speak writes, read straight from the bytes; returns the seconds of audio.
function _ia_wav_seconds(b::Vector{UInt8})::Float64
    u16(i) = Int(b[i]) | Int(b[i+1]) << 8
    u32(i) = Int(b[i]) | Int(b[i+1]) << 8 | Int(b[i+2]) << 16 | Int(b[i+3]) << 24
    @test (String(b[1:4]), String(b[9:12]), String(b[13:16]), String(b[37:40])) == ("RIFF", "WAVE", "fmt ", "data")
    @test (u16(21), u16(23), u32(25), u16(35)) == (1, 1, 24000, 16)   # PCM, mono, 24 kHz, 16-bit
    @test u32(41) == length(b) - 44
    (length(b) - 44) / (24000 * 2)
end

if get(ENV, "UNILM_LIVE", "") == "1" && haskey(ENV, "OPENAI_API_KEY")
    @testset "OpenAI speak: gpt-realtime-2.1-mini over the Realtime API" begin
        r = speak("Hello.")
        @test r isa SpeechSuccess && r.content_type == "audio/wav"
        seconds = _ia_wav_seconds(r.audio)
        @info "OpenAI speak" seconds
        @test 0.3 <= seconds <= 5.0
    end

    @testset "OpenAI translate: gpt-realtime-translate into French" begin
        r = translate(_IA_WAV; language="fr")
        @test r isa TranscriptionSuccess
        text = transcript_text(r)
        @info "OpenAI translate" text
        @test occursin("banane", lowercase(text))   # French for the clip's secret word
    end

    @testset "OpenAI transcribe: gpt-transcribe" begin
        r = transcribe(_IA_WAV)
        @test r isa TranscriptionSuccess
        @test occursin("banana", lowercase(transcript_text(r)))
    end
else
    @info "Skipping OpenAI speech, translation and transcription integration tests (set UNILM_LIVE=1 and OPENAI_API_KEY)"
end

if get(ENV, "UNILM_LIVE", "") == "1" && haskey(ENV, "GEMINI_API_KEY")
    @testset "Gemini speak: gemini-3.8-flash-tts" begin
        r = speak("Hello."; service=GEMINIServiceEndpoint)
        @test r isa SpeechSuccess && r.content_type == "audio/wav"
        seconds = _ia_wav_seconds(r.audio)
        @info "Gemini speak" seconds
        @test seconds >= 0.3
    end
else
    @info "Skipping Gemini speech integration tests (set UNILM_LIVE=1 and GEMINI_API_KEY)"
end
