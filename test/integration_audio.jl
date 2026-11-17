# ─── Speech, translation, transcription and voices integration tests (live) ──
# Requires UNILM_LIVE=1 and OPENAI_API_KEY (OpenAI) / GEMINI_API_KEY (Gemini). The first
# sections cost about a cent at most: two short gpt-realtime-2.1-mini speech responses, one
# gemini-3.8-flash-tts reply, two to six translations of about 1.5 to 2 s of audio and one
# transcription; the last ones, below, about two cents more.

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
        # The secret word comes back translated ("banane") or kept as a code word ("banana"): both seen.
        @test occursin(r"banan[ae]", lowercase(text))
    end

    @testset "OpenAI translate: French speech into the default language, English" begin
        # The English clip translated into English came back empty (2026-10-22), so the default target
        # is witnessed on French speech, which speak makes. The translation of this 2 s sentence varies
        # from call to call (October 2026: of ten, one empty and six with words the sentence does not
        # have, eight naming "banana"), so a translation without the word is retried, up to five of the
        # same clip; a failed call is not retried. The word may come back translated ("banana") or kept
        # ("banane"), as the French-target test above accepts.
        fr = speak("Le mot secret est banane.")
        @test fr isa SpeechSuccess
        path = tempname() * ".wav"
        write(path, fr.audio)
        texts = String[]
        try
            for _ in 1:5
                r = translate(path)
                @test r isa TranscriptionSuccess
                r isa TranscriptionSuccess || break
                push!(texts, transcript_text(r))
                occursin(r"banan[ae]", lowercase(last(texts))) && break
            end
        finally
            rm(path; force=true)
        end
        @info "OpenAI translate, default language" texts
        @test any(t -> occursin(r"banan[ae]", lowercase(t)), texts)   # "banana", or the French word kept
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

# ─── Transcription options, streaming, dialogues and custom voices (live) ────
# OpenAI: gpt-transcribe with logprobs and server VAD, then streamed; gpt-4o-transcribe-diarize streamed; a prompt
# voice made, sent to speak (whose session refuses a prompt voice before any audio) and deleted, and the consent listing (HTTP 404 to an organization without the eligibility). Gemini:
# gemini-3.5-transcribe with diarization and word timestamps; a two-speaker dialogue; a designed voice made, read,
# looked up and deleted, and the prebuilt catalog's first page, one of its voices read. About two cents a run,
# whatever OpenAI charges for a prompt voice aside (its price is not on the pricing page).

if get(ENV, "UNILM_LIVE", "") == "1" && haskey(ENV, "OPENAI_API_KEY")
    @testset "OpenAI transcribe: logprobs, server VAD, and the transcript streamed" begin
        r = transcribe(_IA_WAV; include=["logprobs"], chunking_strategy=TranscriptionVAD(threshold=0.5))
        @test r isa TranscriptionSuccess && occursin("banana", lowercase(transcript_text(r)))
        @test r isa TranscriptionSuccess && !isempty(r.raw["logprobs"])
        heard = String[]
        s = fetch(transcribe(_IA_WAV; stream=true, callback=(c, _) -> c isa String && push!(heard, c)))
        @test s isa TranscriptionSuccess && occursin("banana", lowercase(transcript_text(s)))
        @test s isa TranscriptionSuccess && join(heard) == transcript_text(s) && haskey(s.raw, "usage")
        @test_throws ArgumentError token_usage(s)
        @info "OpenAI streamed transcription" deltas = length(heard) usage = (s isa TranscriptionSuccess ? s.raw["usage"] : s)
    end

    @testset "OpenAI transcribe: a diarized stream's segments" begin
        segs = TranscriptSegment[]
        d = fetch(transcribe(_IA_WAV; model="gpt-4o-transcribe-diarize", response_format="diarized_json",
                             chunking_strategy="auto", stream=true, callback=(c, _) -> c isa TranscriptSegment && push!(segs, c)))
        @test d isa TranscriptionSuccess && !isempty(d.segments) && d.segments == segs
        @test d isa TranscriptionSuccess && all(x -> !isnothing(x.speaker) && x.end_time >= x.start_time, d.segments)
        @info "OpenAI diarized stream" segments = (d isa TranscriptionSuccess ? d.segments : d)
    end

    @testset "OpenAI voices: a prompt voice is made and deleted; consents are for eligible organizations" begin
        v = create_voice(; name="unilm-live", prompt="A calm, clear adult narrator.")
        @test v isa VoiceSuccess
        try
            @test v isa VoiceSuccess && startswith(v.response.id, "voice_") && v.response.type == "prompt"
            # speak sends the voice to its Realtime session as {"id": …}, which reads a voice made from a recording
            # and refuses a prompt voice by name (2026-11-17); a plain string was refused as no built-in voice.
            if v isa VoiceSuccess
                s = speak("Hello."; voice=v.response.id)
                @test s isa SpeechSuccess ||
                      (s isa AudioFailure && occursin("Prompt-based voices are not supported by the Realtime API", s.response))
                @info "OpenAI speak with a custom voice" result = (s isa SpeechSuccess ? length(s.audio) : s)
            end
        finally
            if v isa VoiceSuccess
                d = delete_voice(v.response.id)
                @test d isa VoiceDeleteSuccess && d.id == v.response.id && d.object == "audio.voice.deleted"
                again = delete_voice(v.response.id)
                @test again isa VoiceFailure && again.status == 404
            end
        end
        # Without the eligibility the consent routes answer 404 (2026-11-17); with it, a page.
        c = list_voice_consents(; limit=1)
        @test c isa VoiceConsentListSuccess || (c isa VoiceFailure && c.status == 404)
        @info "OpenAI voice consents" result = c
    end
else
    @info "Skipping OpenAI transcription options and voices integration tests (set UNILM_LIVE=1 and OPENAI_API_KEY)"
end

if get(ENV, "UNILM_LIVE", "") == "1" && haskey(ENV, "GEMINI_API_KEY")
    @testset "Gemini transcribe: gemini-3.5-transcribe, its speaker turns and word timestamps" begin
        r = transcribe(_IA_WAV; service=GEMINIServiceEndpoint, languages=["en-US"], diarization=true,
                       timestamp_granularities=["word"])
        @test r isa TranscriptionSuccess && occursin("banana", lowercase(transcript_text(r)))
        @test r isa TranscriptionSuccess && !isempty(r.segments) && all(s -> !isnothing(s.speaker), r.segments)
        @test r isa TranscriptionSuccess && !isempty(r.words) && all(w -> w.end_time >= w.start_time, r.words)
        @info "Gemini transcription" segments = (r isa TranscriptionSuccess ? r.segments : r)
    end

    @testset "Gemini speak: a two-speaker dialogue in a language" begin
        r = speak([SpeechTurn("Joe", "Hello Jane."), SpeechTurn("Jane", "Hi Joe."; style="cheerful")];
                  service=GEMINIServiceEndpoint, speakers=["Joe" => "Puck", "Jane" => "Kore"], language="en-US")
        @test r isa SpeechSuccess && r.content_type == "audio/wav"
        if r isa SpeechSuccess
            @test _ia_wav_seconds(r.audio) >= 0.5
        end
    end

    @testset "Gemini voices: design a voice, read with it, look it up, list the catalog, delete it" begin
        v = create_voice(; service=GEMINIServiceEndpoint, name="unilm-live", prompt="A calm, clear adult narrator.",
                         language="en-US")
        @test v isa VoiceSuccess
        try
            if v isa VoiceSuccess
                id = v.response.id
                @test startswith(id, "voice_") && v.response.type == "prompted" && haskey(v.response.raw, "sample_audio")
                s = speak("Hello."; service=GEMINIServiceEndpoint, voice=id)
                @test s isa SpeechSuccess
                g = get_voice(id; service=GEMINIServiceEndpoint)
                @test g isa VoiceSuccess && g.response.id == id
            end
            page = list_voices(; service=GEMINIServiceEndpoint, type="prebuilt", limit=3)
            @test page isa VoiceListSuccess && length(page.response.data) == 3 && page.response.has_more
            @test page isa VoiceListSuccess && all(x -> x.type == "prebuilt", page.response.data)
            # A listed prebuilt voice is read by its id (speak sent "achernar" and "am-et-assistant-1", 2026-11-17).
            p = page isa VoiceListSuccess ? speak("Hi."; service=GEMINIServiceEndpoint, voice=first(page.response.data).id) : page
            @test p isa SpeechSuccess
        finally
            if v isa VoiceSuccess
                d = delete_voice(v.response.id; service=GEMINIServiceEndpoint)
                @test d isa VoiceDeleteSuccess && d.id == v.response.id
                gone = get_voice(v.response.id; service=GEMINIServiceEndpoint)
                @test gone isa VoiceFailure && gone.status == 404
            end
        end
    end
else
    @info "Skipping Gemini transcription, dialogue and voices integration tests (set UNILM_LIVE=1 and GEMINI_API_KEY)"
end
