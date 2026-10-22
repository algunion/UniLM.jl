using Base64, Sockets

# 16 kHz mono 16-bit PCM, "The secret word is banana.", with a filler chunk between fmt and data.
const _SP_WAV = joinpath(@__DIR__, "fixtures", "ollama", "secret_word.wav")

# The directive every OpenAI speech session carries: Realtime models are conversational.
const _SP_READ_ALOUD = "Read the user's message aloud, word for word, exactly as written. " *
                       "Do not answer it or comment on it, and do not add, drop or change words."

# ─── WAV bytes built and read field by field, independently of UniLM's codec ──

_sp_le(x) = collect(reinterpret(UInt8, [htol(x)]))
_sp_u16(b, i) = Int(b[i]) | Int(b[i+1]) << 8
_sp_u32(b, i) = Int(b[i]) | Int(b[i+1]) << 8 | Int(b[i+2]) << 16 | Int(b[i+3]) << 24

function _sp_wav(pcm::Vector{UInt8}; rate=24000, channels=1, bits=16, format=0x0001,
                 fmt_extra=UInt8[], before_data=UInt8[])
    block = channels * bits ÷ 8
    fmt = vcat(_sp_le(UInt16(format)), _sp_le(UInt16(channels)), _sp_le(UInt32(rate)),
               _sp_le(UInt32(rate * block)), _sp_le(UInt16(block)), _sp_le(UInt16(bits)), fmt_extra)
    body = vcat(Vector{UInt8}("WAVE"), Vector{UInt8}("fmt "), _sp_le(UInt32(length(fmt))), fmt, before_data,
                Vector{UInt8}("data"), _sp_le(UInt32(length(pcm))), pcm)
    vcat(Vector{UInt8}("RIFF"), _sp_le(UInt32(length(body))), body)
end

_sp_header(b) = (riff=String(b[1:4]), riff_len=_sp_u32(b, 5), wave=String(b[9:12]), fmt=String(b[13:16]),
                 fmt_len=_sp_u32(b, 17), format=_sp_u16(b, 21), channels=_sp_u16(b, 23), rate=_sp_u32(b, 25),
                 byte_rate=_sp_u32(b, 29), block=_sp_u16(b, 33), bits=_sp_u16(b, 35), data=String(b[37:40]),
                 data_len=_sp_u32(b, 41))

_sp_pcm(samples::Vector{Int16}) = collect(reinterpret(UInt8, htol.(samples)))

# ─── A local Realtime WebSocket (speech and translation sessions) ────────────

struct SpeechWSProbe <: UniLM.ServiceEndpoint end
const _sp_ws_base = Ref("")
UniLM._realtime_ws_url(::Type{SpeechWSProbe}) = _sp_ws_base[]
UniLM.auth_header(::Type{SpeechWSProbe}) = ["Authorization" => "Bearer test-key", "Content-Type" => "application/json"]
UniLM.provider_capabilities(::Type{SpeechWSProbe}) = Set([:realtime])

_sp_event(type; kw...) = Dict{String,Any}("type" => type, "event_id" => "event_$(type)", (String(k) => v for (k, v) in kw)...)

# Sends each item: an event dict, or `:close` (a normal close) or a CloseFrameBody (a close with that
# status and reason) to close the socket, which ends the script.
function _sp_emit(ws, items)::Bool
    for it in items
        it === :close && (close(ws); return false)
        it isa HTTP.WebSockets.CloseFrameBody && (close(ws, it); return false)
        HTTP.WebSockets.send(ws, JSON.json(it))
    end
    true
end

"""
Run `f()` against a local Realtime WebSocket that sends `opening` on connect and answers each
client event `ev` with `reply(ev)` (events to send, `:close` closing the socket). Returns
`(f(), seen)`: the handshake targets and Authorization headers, every client event parsed, and how
many connections have ended; `_sp_seen[]` is `seen` while `f` runs.
"""
const _sp_seen = Ref{Any}(nothing)
function _sp_with_realtime(f::Function, reply::Function; opening=Any[_sp_event("session.created")])
    seen = (targets=String[], auth=String[], events=Dict{String,Any}[], ended=Threads.Atomic{Int}(0))
    guard = ReentrantLock()
    srv = HTTP.WebSockets.listen!("127.0.0.1", 0) do ws
        req = ws.handshake_request
        @lock guard begin
            push!(seen.targets, String(req.target))
            push!(seen.auth, HTTP.header(req, "Authorization", ""))
        end
        try
            _sp_emit(ws, opening) || return
            for msg in ws
                ev = JSON.parse(String(msg); dicttype=Dict{String,Any})
                @lock guard push!(seen.events, ev)
                _sp_emit(ws, reply(ev)) || return
            end
        catch e
            e isa HTTP.WebSockets.WebSocketError || e isa Base.IOError || rethrow()
        finally
            Threads.atomic_add!(seen.ended, 1)
        end
    end
    _sp_ws_base[] = "ws://" * HTTP.WebSockets.server_addr(srv)
    _sp_seen[] = seen
    try
        return f(), seen
    finally
        close(srv)
    end
end

# Run `f()` against a peer that completes the WebSocket upgrade, sends session.created and
# session.updated, and never reads again: the client's sends back up once the socket buffers fill.
function _sp_with_stall_peer(f::Function)
    server = Sockets.listen(Sockets.localhost, 0)
    socks = Sockets.TCPSocket[]
    accepting = Threads.@spawn try
        while true
            sock = Sockets.accept(server)
            push!(socks, sock)
            key = match(r"(?i)sec-websocket-key:\s*(\S+)", readuntil(sock, "\r\n\r\n"))[1]
            accept = base64encode(UniLM.SHA.sha1(string(key, "258EAFA5-E914-47DA-95CA-C5AB0DC85B11")))
            write(sock, "HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n",
                        "Sec-WebSocket-Accept: ", accept, "\r\n\r\n")
            for p in ("{\"type\":\"session.created\"}", "{\"type\":\"session.updated\"}")
                write(sock, UInt8[0x81, ncodeunits(p)], p)   # an unmasked text frame
            end
        end
    catch e
        e isa Base.IOError || rethrow()
    end
    _sp_ws_base[] = "ws://127.0.0.1:$(Int(Sockets.getsockname(server)[2]))"
    try
        return f()
    finally
        close(server)
        foreach(close, socks)
        wait(accepting)
    end
end

_sp_delta(chunk) = _sp_event("response.output_audio.delta"; response_id="resp_1", item_id="item_1",
                             output_index=0, content_index=0, delta=base64encode(chunk))
_sp_done(status; details=nothing) = _sp_event("response.done"; response=Dict{String,Any}(
    "object" => "realtime.response", "id" => "resp_1", "status" => status, "status_details" => details,
    "output" => Any[], "usage" => Dict("input_tokens" => 60, "output_tokens" => 25)))

# A speech session: the update is acknowledged, the item added, and response.create answered with
# `on_response` (by default: one delta per chunk, then a completed response.done).
function _sp_speech_reply(chunks; on_response=Any[(_sp_delta(c) for c in chunks)..., _sp_done("completed")],
                          on_update=nothing)
    ev -> begin
        t = ev["type"]
        t == "session.update" ? something(on_update, Any[_sp_event("session.updated"; session=ev["session"])]) :
        t == "conversation.item.create" ? Any[_sp_event("conversation.item.added"; item=ev["item"])] :
        t == "response.create" ? on_response : Any[]
    end
end

_sp_session_update(; voice="marin", instructions=_SP_READ_ALOUD, speed=nothing) =
    Dict{String,Any}("type" => "session.update", "session" => Dict{String,Any}(
        "type" => "realtime", "output_modalities" => ["audio"], "instructions" => instructions,
        "audio" => Dict{String,Any}("output" => Dict{String,Any}(
            "format" => Dict{String,Any}("type" => "audio/pcm", "rate" => 24000), "voice" => voice,
            (isnothing(speed) ? () : ("speed" => speed,))...))))
_sp_item(text) = Dict{String,Any}("type" => "conversation.item.create", "item" => Dict{String,Any}(
    "type" => "message", "role" => "user", "content" => Any[Dict{String,Any}("type" => "input_text", "text" => text)]))

# A translation session: the update is acknowledged; the close is answered with `on_close`.
function _sp_translation_reply(deltas; on_close=Any[(_sp_event("session.output_transcript.delta"; delta=d) for d in deltas)...,
                                                    _sp_event("session.output_audio.delta"; delta=base64encode(UInt8[0, 0]),
                                                              sample_rate=24000, channels=1, format="pcm16"),
                                                    _sp_event("session.closed")], on_update=nothing)
    ev -> begin
        t = ev["type"]
        t == "session.update" ? something(on_update, Any[_sp_event("session.updated"; session=Dict{String,Any}(
            "type" => "translation", "model" => "gpt-realtime-translate", "audio" => ev["session"]["audio"]))]) :
        t == "session.close" ? on_close : Any[]
    end
end

# A Gemini TTS endpoint served by `_with_scripted`.
struct GeminiSpeechProbe <: UniLM.ServiceEndpoint end
UniLM._speech_wire(::Type{GeminiSpeechProbe}) = UniLM._GeminiSpeechWire()
UniLM._gemini_speech_base(::Type{GeminiSpeechProbe}) = _url_probe_base[]
UniLM.auth_header(::Type{GeminiSpeechProbe}) = ["x-goog-api-key" => "test-key", "Content-Type" => "application/json"]

_sp_inline(mime, bytes) = Dict("inlineData" => Dict("mimeType" => mime, "data" => base64encode(bytes)))
_sp_gemini_reply(parts...; finish="STOP") = JSON.json(Dict(
    "candidates" => [Dict("content" => Dict("role" => "model", "parts" => collect(Any, parts)),
                          "finishReason" => finish, "index" => 0)],
    "usageMetadata" => Dict("promptTokenCount" => 4, "candidatesTokenCount" => 25, "totalTokenCount" => 29),
    "modelVersion" => "gemini-3.8-flash-tts"))
_sp_gemini_body(; text="Hi.", voice="Kore", style=nothing) = Dict{String,Any}(
    "contents" => Any[Dict{String,Any}("role" => "user", "parts" => Any[Dict{String,Any}("text" => text,
        (isnothing(style) ? () : ("speechMetadata" => Dict{String,Any}("style" => style),))...)])],
    "generationConfig" => Dict{String,Any}(
        "responseModalities" => Any["AUDIO"],
        "speechConfig" => Dict{String,Any}("voiceConfig" => Dict{String,Any}(
            "prebuiltVoiceConfig" => Dict{String,Any}("voiceName" => voice))),
        "responseFormat" => Dict{String,Any}("audio" => Dict{String,Any}("mimeType" => "AUDIO_L16", "sampleRate" => 24000))))

# ─── Config wiring and request validation ────────────────────────────────────

@testset "Audio API — config seam wiring" begin
    apath = tempname() * ".wav"
    write(apath, UInt8[0x52, 0x49, 0x46, 0x46])   # "RIFF"; content irrelevant (never sent)
    try
        t = UniLM.TranscriptionRequest(service=SeamProbe, file=apath, model="stt")
        @test _seam_timeout(transcribe(t; config=_TINY_DEADLINE), UniLM.AudioCallError)
        @test _seam_timeout(transcribe(apath; service=SeamProbe, model="stt", config=_TINY_DEADLINE), UniLM.AudioCallError)
    finally
        rm(apath; force=true)
    end
    # A spent budget ends speech and translation before any connection or request.
    (r1, r2, r3), seen = _sp_with_realtime(_ -> Any[]) do
        (speak("hi"; service=SpeechWSProbe, config=_TINY_DEADLINE),
         speak(SpeechRequest(input="hi", service=SpeechWSProbe); config=_TINY_DEADLINE),
         translate(_SP_WAV; service=SpeechWSProbe, config=_TINY_DEADLINE))
    end
    @test all(r -> _seam_timeout(r, UniLM.AudioCallError) && r.cause.phase === :deadline, (r1, r2, r3))
    @test isempty(seen.targets)
    r, requests = _with_scripted((_, _) -> _json(200, "{}")) do
        speak("hi"; service=GeminiSpeechProbe, config=_TINY_DEADLINE)
    end
    @test _seam_timeout(r, UniLM.AudioCallError) && isempty(requests)
end

@testset "Audio API — request fields are checked against the documented values" begin
    apath = tempname() * ".wav"
    write(apath, UInt8[0x52, 0x49, 0x46, 0x46])
    try
        for fmt in ("json", "text", "srt", "verbose_json", "vtt", "diarized_json")
            @test UniLM.TranscriptionRequest(file=apath, response_format=fmt).response_format == fmt
        end
        @test_throws ArgumentError UniLM.TranscriptionRequest(file=apath, response_format="mp3")
        @test_throws ArgumentError UniLM.TranscriptionRequest(file=apath, temperature=1.5)
        @test_throws ArgumentError UniLM.TranscriptionRequest(file=apath, temperature=-0.1)
        @test UniLM.TranscriptionRequest(file=apath, temperature=0.0).temperature == 0.0
    finally
        rm(apath; force=true)
    end
end

@testset "the transcription docstring names the formats only deprecated models answer" begin
    doc = string(@doc TranscriptionRequest)
    @test occursin("`srt`, `vtt` and `verbose_json`", doc) && occursin("`whisper-1`", doc)
    @test occursin("`diarized_json`", doc) && occursin("`gpt-4o-transcribe-diarize`", doc)
    @test occursin("February 26, 2027", doc)
end

@testset "SpeechRequest: each service's model and voice by default" begin
    r = SpeechRequest(input="Hi.")
    @test (r.service, r.model, r.voice, r.response_format, r.speed, r.instructions) ==
          (OPENAIServiceEndpoint, "gpt-realtime-2.1-mini", "marin", "wav", nothing, nothing)
    g = SpeechRequest(input="Hi.", service=GEMINIServiceEndpoint)
    @test (g.model, g.voice, g.response_format) == ("gemini-3.8-flash-tts", "Kore", "wav")
    # Explicit values are kept as given, in the concrete types the fields hold.
    e = SpeechRequest(input=SubString("Hi. Bye.", 1, 3), model="gpt-realtime-2.1", voice="cedar",
                      response_format="pcm", speed=1, instructions="Calm.")
    @test (e.input, e.model, e.voice, e.response_format, e.speed, e.instructions) ==
          ("Hi.", "gpt-realtime-2.1", "cedar", "pcm", 1.0, "Calm.")
    @test e.input isa String && e.speed isa Float64
end

@testset "SpeechRequest: voices, formats and speed follow each service's docs" begin
    refusal(f) = try f(); nothing catch x; x end
    for v in ("alloy", "ash", "ballad", "coral", "echo", "sage", "shimmer", "verse", "marin", "cedar")
        @test SpeechRequest(input="Hi.", voice=v).voice == v
    end
    gemini_voices = ("Zephyr", "Puck", "Charon", "Kore", "Fenrir", "Leda", "Orus", "Aoede", "Callirrhoe",
        "Autonoe", "Enceladus", "Iapetus", "Umbriel", "Algieba", "Despina", "Erinome", "Algenib", "Rasalgethi",
        "Laomedeia", "Achernar", "Alnilam", "Schedar", "Gacrux", "Pulcherrima", "Achird", "Zubenelgenubi",
        "Vindemiatrix", "Sadachbia", "Sadaltager", "Sulafat")
    @test length(gemini_voices) == 30
    for v in gemini_voices
        @test SpeechRequest(input="Hi.", service=GEMINIServiceEndpoint, voice=v).voice == v
    end
    # Speech API voices the Realtime API does not have, another service's voices, near-miss spellings.
    for v in ("nova", "onyx", "fable", "Kore", "Marin", "marin ", "")
        e = refusal(() -> SpeechRequest(input="Hi.", voice=v))
        @test e isa ArgumentError && occursin("OpenAI Realtime voices are alloy, ash", e.msg) && occursin(repr(v), e.msg)
    end
    for v in ("kore", "marin", "alloy", "Kore2")
        e = refusal(() -> SpeechRequest(input="Hi.", service=GEMINIServiceEndpoint, voice=v))
        @test e isa ArgumentError && occursin("Gemini TTS voices are Zephyr, Puck", e.msg) && occursin(repr(v), e.msg)
    end
    @test SpeechRequest(input="Hi.", response_format="pcm").response_format == "pcm"
    for fmt in ("mp3", "opus", "aac", "flac", "WAV", "", "audio/wav")
        e = refusal(() -> SpeechRequest(input="Hi.", response_format=fmt))
        @test e isa ArgumentError && occursin("\"wav\"", e.msg) && occursin("\"pcm\"", e.msg)
        @test refusal(() -> SpeechRequest(input="Hi.", service=GEMINIServiceEndpoint, response_format=fmt)) isa ArgumentError
    end
    # OpenAI Realtime output speed: 0.25 to 1.5 (session.update reference).
    for s in (0.25, 1.0, 1.5)
        @test SpeechRequest(input="Hi.", speed=s).speed == s
    end
    for s in (0.24, 1.51, 4.0, 0.0, -1.0, NaN, Inf)
        e = refusal(() -> SpeechRequest(input="Hi.", speed=s))
        @test e isa ArgumentError && occursin("between 0.25 and 1.5", e.msg)
    end
    # Gemini TTS has no speed parameter: the pace is a speech style.
    for s in (1.0, 0.5)
        e = refusal(() -> SpeechRequest(input="Hi.", service=GEMINIServiceEndpoint, speed=s))
        @test e isa ArgumentError && occursin("Gemini TTS has no speed control", e.msg) && occursin("instructions", e.msg)
    end
    for text in ("", "   ", "\n")
        @test refusal(() -> SpeechRequest(input=text)) isa ArgumentError
        @test refusal(() -> speak(text; service=GEMINIServiceEndpoint)) isa ArgumentError
    end
end

@testset "save_audio replaces the destination atomically" begin
    mktempdir() do dir
        path = joinpath(dir, "speech.wav")
        write(path, "previous audio")
        before = stat(path).inode
        @test save_audio(SpeechSuccess(audio=Vector{UInt8}("new audio"), content_type="audio/wav"), path) == path
        @test read(path, String) == "new audio"
        @test stat(path).inode != before   # renamed into place, never truncated and rewritten
        @test readdir(dir) == ["speech.wav"]
    end
end

@testset "Audio API — a failure keeps the request id the service sent" begin
    r = _answered(() -> speak("hi"; service=GeminiSpeechProbe), 401; headers=["x-request-id" => "req_audio"])
    @test r isa UniLM.AudioFailure && r.request_id == "req_audio" && r.status == 401
end

@testset "the transcription model is resolved per service" begin
    wav = joinpath(@__DIR__, "fixtures", "ollama", "secret_word.wav")
    @test TranscriptionRequest(file=wav).model === nothing
    @test UniLM._transcription_model(TranscriptionRequest(file=wav)) == "gpt-transcribe"     # OpenAI: unchanged
    @test UniLM._transcription_model(TranscriptionRequest(file=wav, model="whisper-1")) == "whisper-1"
end

# Custom endpoints served by _with_scripted: one written against the audio API (`:audio` alone, as a
# backend written before `:transcription` existed declares it) and one that declares neither.
struct AudioOnlyProbe <: UniLM.ServiceEndpoint end
UniLM._resolve_base_url(::Type{AudioOnlyProbe}) = _url_probe_base[]
UniLM.auth_header(::Type{AudioOnlyProbe}) = ["Content-Type" => "application/json"]
UniLM.provider_capabilities(::Type{AudioOnlyProbe}) = Set([:audio])
struct NoAudioProbe <: UniLM.ServiceEndpoint end
UniLM._resolve_base_url(::Type{NoAudioProbe}) = _url_probe_base[]
UniLM.auth_header(::Type{NoAudioProbe}) = ["Content-Type" => "application/json"]
UniLM.provider_capabilities(::Type{NoAudioProbe}) = Set([:chat])

@testset "an endpoint that declares the audio API keeps transcription" begin
    wav = joinpath(@__DIR__, "fixtures", "ollama", "secret_word.wav")
    r, seen = _with_scripted((_, _) -> _json(200, "{\"text\":\"The secret word is banana.\"}")) do
        transcribe(wav; service=AudioOnlyProbe, model="stt")
    end
    @test r isa TranscriptionSuccess && transcript_text(r) == "The secret word is banana."
    @test only(seen).method == "POST" && only(seen).target == "/v1/audio/transcriptions"
    @test has_capability(AudioOnlyProbe, :transcription)                 # the predicate says what the gate does
    # Speech and translation run on the Realtime API, which this endpoint does not declare.
    refusal(f) = try f(); nothing catch x; x end
    (e1, e2), seen2 = _with_scripted((_, _) -> _json(200, "{}")) do
        (refusal(() -> speak("hi"; service=AudioOnlyProbe)), refusal(() -> translate(wav; service=AudioOnlyProbe)))
    end
    @test e1 isa ArgumentError && occursin("speak (OpenAI Realtime API) is not supported by AudioOnlyProbe", e1.msg)
    @test e2 isa ArgumentError && occursin("translate (OpenAI Realtime API) is not supported by AudioOnlyProbe", e2.msg)
    @test isempty(seen2)
end

@testset "an endpoint that declares neither audio capability is refused before any request" begin
    wav = joinpath(@__DIR__, "fixtures", "ollama", "secret_word.wav")
    e, seen = _with_scripted((_, _) -> _json(200, "{\"text\":\"x\"}")) do
        try transcribe(wav; service=NoAudioProbe, model="stt"); nothing catch x; x end
    end
    @test e isa ArgumentError && occursin("Audio API is not supported by NoAudioProbe", e.msg)
    @test isempty(seen)
    @test !has_capability(NoAudioProbe, :transcription)
end

@testset "speak and translate never call the retiring Audio API speech and translation routes" begin
    src = normpath(joinpath(@__DIR__, "..", "src"))
    hits = [(basename(f), i) for f in readdir(src; join=true) if endswith(f, ".jl")
                             for (i, line) in enumerate(eachline(f)) if occursin(r"audio/(speech|translations)\b", line)]
    @test isempty(hits)
    @test !isdefined(UniLM, :AUDIO_SPEECH_PATH) && !isdefined(UniLM, :AUDIO_TRANSLATIONS_PATH)
end

# ─── 16-bit PCM and WAV ──────────────────────────────────────────────────────

@testset "the WAV writer: a canonical 44-byte header over the samples" begin
    pcm = UInt8[0x01, 0x00, 0xff, 0x7f]   # samples 1 and 32767
    @test (@inferred UniLM._wav_bytes(pcm)) == UInt8[
        0x52, 0x49, 0x46, 0x46, 0x28, 0x00, 0x00, 0x00, 0x57, 0x41, 0x56, 0x45,   # RIFF, 36 + 4, WAVE
        0x66, 0x6d, 0x74, 0x20, 0x10, 0x00, 0x00, 0x00, 0x01, 0x00, 0x01, 0x00,   # fmt , 16, PCM, mono
        0xc0, 0x5d, 0x00, 0x00, 0x80, 0xbb, 0x00, 0x00, 0x02, 0x00, 0x10, 0x00,   # 24000 Hz, 48000 B/s, 2, 16-bit
        0x64, 0x61, 0x74, 0x61, 0x04, 0x00, 0x00, 0x00, 0x01, 0x00, 0xff, 0x7f]   # data, 4 bytes, samples
    @test UniLM._wav_bytes(UInt8[]) == _sp_wav(UInt8[])
    @test length(UniLM._wav_bytes(zeros(UInt8, 48_000))) == 44 + 48_000
    @test_throws ArgumentError UniLM._wav_bytes(UInt8[0x01, 0x00, 0x02])   # half a sample
end

@testset "the WAV reader: 16-bit PCM at any rate and channel count, nothing else" begin
    w = @inferred UniLM._read_wav(read(_SP_WAV))
    @test (w.rate, w.channels, length(w.samples)) == (16000, 1, 23785)   # the filler chunk is skipped
    samples = Int16[1, -1, 32767, -32768, 0, 12]
    @test UniLM._read_wav(_sp_wav(_sp_pcm(samples); rate=44100, channels=2)) ==
          (samples=samples, rate=44100, channels=2)
    # WAVE_FORMAT_EXTENSIBLE carrying PCM (how files with more than two channels are written).
    guid_pcm = UInt8[0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x10, 0x00, 0x80, 0x00, 0x00, 0xaa, 0x00, 0x38, 0x9b, 0x71]
    ext = vcat(_sp_le(UInt16(22)), _sp_le(UInt16(16)), _sp_le(UInt32(0x3f)), guid_pcm)
    @test UniLM._read_wav(_sp_wav(_sp_pcm(samples); channels=6, format=0xfffe, fmt_extra=ext)).channels == 6
    # A streamed WAV overstates its data size; an incomplete trailing frame holds no sample.
    streamed = _sp_wav(_sp_pcm(samples))
    streamed[41:44] .= 0xff
    @test UniLM._read_wav(streamed).samples == samples
    @test UniLM._read_wav(vcat(streamed, UInt8[0x07])).samples == samples
    refusal(b) = try UniLM._read_wav(b); nothing catch x; x end
    @test occursin("RIFF/WAVE", refusal(Vector{UInt8}("ID3\x04 an mp3 file")).msg)
    @test occursin("32-bit float", refusal(_sp_wav(zeros(UInt8, 8); format=0x0003, bits=32)).msg)
    @test occursin("A-law", refusal(_sp_wav(zeros(UInt8, 8); format=0x0006, bits=8)).msg)
    @test occursin("8-bit", refusal(_sp_wav(zeros(UInt8, 8); bits=8)).msg)
    @test occursin("24-bit", refusal(_sp_wav(zeros(UInt8, 9); bits=24)).msg)
    @test occursin("no data chunk", refusal(_sp_wav(UInt8[])[1:36]).msg)
end

@testset "the resampler: linear interpolation that keeps a sine's frequency" begin
    sine(rate, n) = [sin(2π * 440 * k / rate) for k in 0:n-1]
    y = @inferred UniLM._resample_linear(sine(16000, 16000), 16000, 24000)
    @test length(y) == 24000                                   # 16 → 24 kHz: length × 3/2
    @test maximum(abs.(y[1:end-1] .- sine(24000, 24000)[1:end-1])) < 0.005   # the same 440 Hz sine
    @test count(k -> y[k] < 0 <= y[k+1], 1:length(y)-1) == 439               # a rising crossing every 1/440 s after 0
    @test UniLM._resample_linear([1.0, 2.0, 3.0], 24000, 24000) == [1.0, 2.0, 3.0]
    @test UniLM._resample_linear([0.0, 4.0], 16000, 32000) == [0.0, 2.0, 4.0, 4.0]   # halfway, then held
    @test length(UniLM._resample_linear(sine(48000, 4801), 48000, 24000)) == 2401
    @test UniLM._resample_linear(Float64[], 16000, 24000) == Float64[]
    # Stereo is averaged into one channel.
    @test (@inferred UniLM._downmix(Int16[1000, 3000, -2000, 2000, 7, 7], 2)) == [2000.0, 0.0, 7.0]
    @test UniLM._downmix(Int16[5, -5], 1) == [5.0, -5.0]
    @test (@inferred UniLM._pcm16le([1.0, -1.0, 40000.0, -40000.0, 0.4])) ==
          UInt8[0x01, 0x00, 0xff, 0xff, 0xff, 0x7f, 0x00, 0x80, 0x00, 0x00]
end

@testset "translate: its input becomes 24 kHz mono 16-bit PCM before any connection" begin
    pcm = UniLM._translation_pcm(_SP_WAV)
    @test length(pcm) == 2 * cld(23785 * 3, 2)                 # 16 kHz → 24 kHz
    mktempdir() do dir
        stereo = joinpath(dir, "stereo.wav")
        write(stereo, _sp_wav(_sp_pcm(Int16[100, 300, -100, -300]); channels=2, rate=24000))
        @test UniLM._translation_pcm(stereo) == _sp_pcm(Int16[200, -200])
        mp3 = joinpath(dir, "clip.mp3")
        write(mp3, "ID3\x04 not a wav")
        float = joinpath(dir, "float.wav")
        write(float, _sp_wav(zeros(UInt8, 8); format=0x0003, bits=32))
        empty = joinpath(dir, "empty.wav")
        write(empty, _sp_wav(UInt8[]))
        refusal(f) = try f(); nothing catch x; x end
        (e1, e2, e3, e4, e5), seen = _sp_with_realtime(_ -> Any[]) do
            (refusal(() -> translate(mp3; service=SpeechWSProbe)),
             refusal(() -> translate(float; service=SpeechWSProbe)),
             refusal(() -> translate(empty; service=SpeechWSProbe)),
             refusal(() -> translate(joinpath(dir, "missing.wav"); service=SpeechWSProbe)),
             refusal(() -> translate(_SP_WAV; service=SpeechWSProbe, language=" ")))
        end
        @test e1 isa ArgumentError && occursin("convert it to 16-bit PCM WAV", e1.msg)
        @test e2 isa ArgumentError && occursin("32-bit float", e2.msg) && occursin("convert it to 16-bit PCM WAV", e2.msg)
        @test e3 isa ArgumentError && occursin("no audio", e3.msg)
        @test e4 isa ArgumentError && occursin("file not found", e4.msg)
        @test e5 isa ArgumentError && occursin("target language", e5.msg)
        @test isempty(seen.targets)
    end
end

# ─── Speech on OpenAI Realtime ───────────────────────────────────────────────

@testset "speak on OpenAI: a Realtime session, its 24 kHz PCM16 deltas wrapped as WAV" begin
    c1, c2 = UInt8[0x01, 0x00, 0x02, 0x00], UInt8[0xff, 0x7f, 0x00, 0x80, 0x10, 0x00]
    r, seen = _sp_with_realtime(_sp_speech_reply([c1, c2])) do
        speak("Hi."; service=SpeechWSProbe)
    end
    @test r isa SpeechSuccess && r.content_type == "audio/wav"
    h = _sp_header(r.audio)
    @test (h.riff, h.wave, h.fmt, h.data) == ("RIFF", "WAVE", "fmt ", "data")
    @test (h.fmt_len, h.format, h.channels, h.rate, h.byte_rate, h.block, h.bits) == (16, 1, 1, 24000, 48000, 2, 16)
    @test h.data_len == length(c1) + length(c2) && h.riff_len == 36 + h.data_len
    @test r.audio == _sp_wav(vcat(c1, c2))
    @test only(seen.targets) == "/?model=gpt-realtime-2.1-mini" && only(seen.auth) == "Bearer test-key"
    @test seen.events == [_sp_session_update(), _sp_item("Hi."), Dict{String,Any}("type" => "response.create")]
end

@testset "speak on OpenAI: raw PCM, the chosen voice, speed and delivery" begin
    c1, c2 = UInt8[0x01, 0x00], UInt8[0x02, 0x00, 0x03, 0x00]
    r, seen = _sp_with_realtime(_sp_speech_reply([c1, c2])) do
        speak("Good morning."; service=SpeechWSProbe, model="gpt-realtime-2.1", voice="cedar",
              response_format="pcm", speed=1.25, instructions="Cheerful and warm.")
    end
    @test r isa SpeechSuccess && r.content_type == "audio/pcm" && r.audio == vcat(c1, c2)
    @test only(seen.targets) == "/?model=gpt-realtime-2.1"
    @test seen.events == [_sp_session_update(voice="cedar", speed=1.25,
                                             instructions=_SP_READ_ALOUD * " Delivery: Cheerful and warm."),
                          _sp_item("Good morning."), Dict{String,Any}("type" => "response.create")]
end

@testset "speak on OpenAI: an error event is the provider's refusal, never partial audio" begin
    err = _sp_event("error"; error=Dict("type" => "invalid_request_error", "code" => "invalid_value",
                                        "message" => "Voice 'marin' is not available for this session.",
                                        "param" => "session.audio.output.voice", "event_id" => nothing))
    # Refused at the update: nothing more is sent.
    r, seen = _sp_with_realtime(_sp_speech_reply([UInt8[0x01, 0x00]]; on_update=Any[err])) do
        speak("Hi."; service=SpeechWSProbe)
    end
    @test r isa UniLM.AudioFailure && isnothing(r.status) && isnothing(r.request_id)
    @test occursin("Voice 'marin' is not available for this session.", r.response)
    @test JSON.parse(r.response)["error"]["code"] == "invalid_value"
    @test [ev["type"] for ev in seen.events] == ["session.update"]
    # An error after audio has started: the audio so far is not a result.
    r2, _ = _sp_with_realtime(_sp_speech_reply(UInt8[]; on_response=Any[_sp_delta(UInt8[0x01, 0x00]), err])) do
        speak("Hi."; service=SpeechWSProbe)
    end
    @test r2 isa UniLM.AudioFailure && occursin("is not available", r2.response)
end

@testset "speak on OpenAI: a response that does not complete is a failure" begin
    for (status, details) in (("failed", Dict("type" => "failed", "error" => Dict("message" => "Internal error"))),
                              ("incomplete", Dict("type" => "incomplete", "reason" => "content_filter")),
                              ("cancelled", Dict("type" => "cancelled", "reason" => "turn_detected")))
        r, _ = _sp_with_realtime(_sp_speech_reply(UInt8[]; on_response=Any[_sp_delta(UInt8[0x01, 0x00]),
                                                                             _sp_done(status; details)])) do
            speak("Hi."; service=SpeechWSProbe)
        end
        @test r isa UniLM.AudioFailure && JSON.parse(r.response)["response"]["status"] == status
    end
    # Completed, but without a byte of audio.
    r, _ = _sp_with_realtime(_sp_speech_reply(UInt8[]; on_response=Any[_sp_done("completed")])) do
        speak("Hi."; service=SpeechWSProbe)
    end
    @test r isa UniLM.AudioCallError && occursin("without audio", r.error)
    # Audio that is not whole 16-bit samples.
    r, _ = _sp_with_realtime(_sp_speech_reply([UInt8[0x01, 0x00, 0x02]])) do
        speak("Hi."; service=SpeechWSProbe)
    end
    @test r isa UniLM.AudioCallError
end

@testset "speak on OpenAI: a session that closes before response.done is a call error" begin
    r, _ = _sp_with_realtime(_sp_speech_reply(UInt8[]; on_response=Any[_sp_delta(UInt8[0x01, 0x00]),
                                                                         _sp_delta(UInt8[0x02, 0x00]), :close])) do
        speak("Hi."; service=SpeechWSProbe)
    end
    @test r isa UniLM.AudioCallError && occursin("closed before response.done", r.error)
    r2, _ = _sp_with_realtime(_ -> Any[:close]) do       # closed at the first client event
        speak("Hi."; service=SpeechWSProbe)
    end
    @test r2 isa UniLM.AudioCallError && occursin("closed before", r2.error)
    # The server's close status and reason are kept, in the message and as the cause.
    overloaded = HTTP.WebSockets.CloseFrameBody(1011, "overloaded")
    r3, _ = _sp_with_realtime(_sp_speech_reply(UInt8[]; on_response=Any[_sp_delta(UInt8[0x01, 0x00]), overloaded])) do
        speak("Hi."; service=SpeechWSProbe)
    end
    @test r3 isa UniLM.AudioCallError &&
          r3.error == "the Realtime session closed before response.done (websocket closed with status 1011: overloaded)"
    @test r3.cause isa HTTP.WebSockets.WebSocketError && r3.cause.message == overloaded
end

@testset "speak and translate on OpenAI: a server that closes the session while UniLM sends is still heard" begin
    # The server acknowledges the update and closes at once, so UniLM's next send fails: the events
    # the server sent before its close are read before the close is reported.
    err = _sp_event("error"; error=Dict("type" => "invalid_request_error", "code" => "test_code",
                                        "message" => "The test peer ends this session.", "param" => nothing))
    closing = HTTP.WebSockets.CloseFrameBody(1011, "test peer closing")
    mktempdir() do dir
        long = joinpath(dir, "long.wav")
        write(long, _sp_wav(zeros(UInt8, 2 * 24_000 * 60)))   # 60 s of audio: still sending when the close lands
        for (call, awaited) in ((() -> speak("Hi."; service=SpeechWSProbe), "response.done"),
                                (() -> translate(long; service=SpeechWSProbe), "session.closed"))
            # An error event before the close is the provider's refusal.
            r, _ = _sp_with_realtime(call, ev -> ev["type"] == "session.update" ?
                                         Any[_sp_event("session.updated"), err, closing] : Any[])
            @test r isa UniLM.AudioFailure && isnothing(r.status) &&
                  JSON.parse(r.response; dicttype=Dict{String,Any}) == err
            # Without one, the close is the call's error, with its status and reason.
            r2, _ = _sp_with_realtime(call, ev -> ev["type"] == "session.update" ?
                                          Any[_sp_event("session.updated"), closing] : Any[])
            @test r2 isa UniLM.AudioCallError && r2.error ==
                  "the Realtime session closed before $awaited (websocket closed with status 1011: test peer closing)"
            @test r2.cause isa HTTP.WebSockets.WebSocketError && r2.cause.message == closing
        end
    end
end

@testset "speak on OpenAI: a silent peer ends the call at its idle bound or its deadline" begin
    cfg = RequestConfig(connect_timeout=10.0, stream_idle_timeout=1.0, total_deadline=Inf)
    t = time()
    r, _ = _sp_with_realtime(_sp_speech_reply(UInt8[]; on_response=Any[_sp_delta(UInt8[0x01, 0x00])])) do
        speak("Hi."; service=SpeechWSProbe, config=cfg)
    end
    @test r isa UniLM.AudioCallError && r.cause isa UniLMTimeout && r.cause.phase === :stream_idle
    @test time() - t < 15.0
    deadline = RequestConfig(connect_timeout=10.0, stream_idle_timeout=Inf, total_deadline=1.5)
    t = time()
    r2, _ = _sp_with_realtime(_ -> Any[]; opening=Any[]) do      # never answers anything
        speak("Hi."; service=SpeechWSProbe, config=deadline)
    end
    @test r2 isa UniLM.AudioCallError && r2.cause isa UniLMTimeout && r2.cause.phase === :deadline
    @test r2.cause.limit == 1.5 && time() - t < 15.0
end

@testset "speak: what the request cannot carry is refused before any connection" begin
    refusal(f) = try f(); nothing catch x; x end
    (e1, e2, e3, e4), seen = _sp_with_realtime(_sp_speech_reply([UInt8[0x01, 0x00]])) do
        (refusal(() -> speak("Hi."; service=SpeechWSProbe, voice="nova")),
         refusal(() -> speak("Hi."; service=SpeechWSProbe, response_format="mp3")),
         refusal(() -> speak("Hi."; service=SpeechWSProbe, speed=2.0)),
         refusal(() -> speak("Hi."; service=SpeechWSProbe, voice="Kore")))
    end
    @test all(e -> e isa ArgumentError, (e1, e2, e3, e4)) && isempty(seen.targets)
    # Google's OpenAI-compatible endpoint has no speech route: its refusals name the endpoint that
    # speaks Gemini, for a Gemini voice too.
    for f in (() -> speak("Hi."; service=GEMINIOpenAIServiceEndpoint),
              () -> SpeechRequest(input="Hi.", service=GEMINIOpenAIServiceEndpoint, voice="Kore"))
        e5 = refusal(f)
        @test e5 isa ArgumentError && e5.msg == "GEMINIOpenAIServiceEndpoint has no speech route (Google's " *
              "OpenAI-compatible API documents none): Gemini TTS is on the native Gemini API, so pass " *
              "service=GEMINIServiceEndpoint (https://ai.google.dev/gemini-api/docs/speech-generation)"
    end
    # Endpoints without a Realtime WebSocket: never sent another endpoint's credentials.
    e6 = refusal(() -> speak("Hi."; service=SeamProbe))
    @test e6 isa ArgumentError && occursin("supports only OPENAIServiceEndpoint", e6.msg)
end

@testset "speak and translate on OpenAI: an upgrade refused with 401 is an AudioFailure with that status" begin
    srv = HTTP.serve!(_ -> HTTP.Response(401, ["Content-Type" => "application/json"],
                                         "{\"error\":{\"message\":\"Incorrect API key provided.\"}}"),
                      "127.0.0.1", 0; verbose=false)
    _sp_ws_base[] = "ws://127.0.0.1:$(HTTP.port(srv))"
    try
        for r in (speak("Hi."; service=SpeechWSProbe), translate(_SP_WAV; service=SpeechWSProbe))
            @test r isa UniLM.AudioFailure && r.status == 401 && occursin("401", r.response)
        end
    finally
        close(srv)
    end
end

@testset "speak and translate on OpenAI: a token already cancelled sends nothing" begin
    tok = cancel!(CancelToken())
    (r1, r2), seen = _sp_with_realtime(_sp_speech_reply([UInt8[0x01, 0x00]])) do
        with_cancel(tok) do
            (speak("Hi."; service=SpeechWSProbe), translate(_SP_WAV; service=SpeechWSProbe))
        end
    end
    @test all(r -> r isa UniLM.AudioCallError && r.cause isa UniLMCancelled && r.cause.source === :token, (r1, r2))
    @test isempty(seen.targets)
    r3, requests = _with_scripted((_, _) -> _json(200, "{}")) do
        with_cancel(() -> speak("Hi."; service=GeminiSpeechProbe), tok)
    end
    @test r3 isa UniLM.AudioCallError && r3.cause isa UniLMCancelled && isempty(requests)
end

@testset "speak and translate on OpenAI: a cancel ends a silent session at once and frees its socket" begin
    unbounded = RequestConfig(connect_timeout=10.0, stream_idle_timeout=Inf, total_deadline=Inf)
    for (call, last_event) in ((() -> speak("Hi."; service=SpeechWSProbe, config=unbounded), "response.create"),
                               (() -> translate(_SP_WAV; service=SpeechWSProbe, config=unbounded), "session.close"))
        waiting = Threads.Atomic{Bool}(false)
        # The update is acknowledged; after the call's last client event the peer goes silent.
        reply = ev -> (ev["type"] == last_event && (waiting[] = true);
                       ev["type"] == "session.update" ? Any[_sp_event("session.updated"; session=ev["session"])] : Any[])
        tok = CancelToken()
        (r, elapsed, freed), _ = _sp_with_realtime(reply) do
            t = Threads.@spawn with_cancel(call, tok)
            @test timedwait(() -> waiting[], 20.0) === :ok
            t0 = time()
            cancel!(tok)
            done = timedwait(() -> istaskdone(t), 10.0) === :ok
            elapsed = time() - t0
            freed = timedwait(() -> _sp_seen[].ended[] == 1, 10.0)   # the peer saw the connection end
            (done ? fetch(t) : nothing, elapsed, freed)
        end
        @test r isa UniLM.AudioCallError && r.cause isa UniLMCancelled && r.cause.source === :token
        @test elapsed < 2.0 && freed === :ok
    end
end

# ─── Speech on Gemini TTS ────────────────────────────────────────────────────

@testset "speak on Gemini: generateContent with the voice, its 24 kHz L16 reply wrapped as WAV" begin
    @test UniLM._gemini_speech_url(GEMINIServiceEndpoint, "gemini-3.8-flash-tts") ==
          "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.8-flash-tts:generateContent"
    pcm = _sp_pcm(Int16[1, -2, 300, -32768])
    reply = _sp_gemini_reply(_sp_inline("audio/L16;codec=pcm;rate=24000", pcm))
    r, seen = _with_scripted((_, _) -> _json(200, reply)) do
        speak("Hi."; service=GeminiSpeechProbe)
    end
    @test r isa SpeechSuccess && r.content_type == "audio/wav" && r.audio == _sp_wav(pcm)
    h = _sp_header(r.audio)
    @test (h.format, h.channels, h.rate, h.bits, h.data_len) == (1, 1, 24000, 16, length(pcm))
    req = only(seen)
    @test req.method == "POST" && req.target == "/models/gemini-3.8-flash-tts:generateContent"
    body = JSON.parse(req.body; dicttype=Dict{String,Any})
    @test body == _sp_gemini_body()
    @test body["generationConfig"]["responseModalities"] == ["AUDIO"]
    @test body["generationConfig"]["speechConfig"]["voiceConfig"]["prebuiltVoiceConfig"]["voiceName"] == "Kore"
end

@testset "speak on Gemini: the delivery is the part's speech style; raw PCM on request" begin
    pcm = _sp_pcm(Int16[7, 8])
    r, seen = _with_scripted((_, _) -> _json(200, _sp_gemini_reply(_sp_inline("audio/l16", pcm)))) do
        speak("Hello there."; service=GeminiSpeechProbe, voice="Puck", model="gemini-3.8-flash-lite-tts",
              instructions="whispered urgently", response_format="pcm")
    end
    @test r isa SpeechSuccess && r.content_type == "audio/pcm" && r.audio == pcm
    @test only(seen).target == "/models/gemini-3.8-flash-lite-tts:generateContent"
    @test JSON.parse(only(seen).body; dicttype=Dict{String,Any}) ==
          _sp_gemini_body(text="Hello there.", voice="Puck", style="whispered urgently")
end

@testset "speak on Gemini: a WAV reply, several parts, and replies that hold no usable audio" begin
    pcm1, pcm2 = _sp_pcm(Int16[1, 2]), _sp_pcm(Int16[3])
    run(reply; status=200) = first(_with_scripted((_, _) -> _json(status, reply)) do
        speak("Hi."; service=GeminiSpeechProbe)
    end)
    r = run(_sp_gemini_reply(_sp_inline("audio/wav", _sp_wav(pcm1; before_data=vcat(Vector{UInt8}("LIST"),
                                        _sp_le(UInt32(4)), Vector{UInt8}("INFO")))), _sp_inline("audio/L16;rate=24000", pcm2)))
    @test r isa SpeechSuccess && r.audio == _sp_wav(vcat(pcm1, pcm2))
    for (reply, fragment) in (
            (_sp_gemini_reply(Dict("text" => "I can't read that.")), "no audio"),
            (_sp_gemini_reply(_sp_inline("audio/L16;codec=pcm;rate=16000", pcm1)), "16000 Hz"),
            (_sp_gemini_reply(_sp_inline("audio/wav", _sp_wav(pcm1; rate=48000))), "48000 Hz"),
            (_sp_gemini_reply(_sp_inline("audio/mpeg", UInt8[0xff, 0xfb])), "audio/mpeg"),
            (_sp_gemini_reply(_sp_inline("audio/L16;rate=24000", UInt8[0x01])), "16-bit"),
            ("not json", ""))
        r = run(reply)
        @test r isa UniLM.AudioCallError && occursin(fragment, r.error)
    end
    f = run("{\"error\":{\"code\":400,\"message\":\"Voice not found\",\"status\":\"INVALID_ARGUMENT\"}}"; status=400)
    @test f isa UniLM.AudioFailure && f.status == 400 && occursin("Voice not found", f.response)
end

@testset "speak on Gemini: audio from a candidate that did not finish is a failure, never a success" begin
    pcm = _sp_pcm(Int16[1, 2, 3, 4])
    run(reply) = first(_with_scripted((_, _) -> _json(200, reply)) do
        speak("Hi."; service=GeminiSpeechProbe)
    end)
    # Cut at the output cap, stopped by a filter, or stopped for another reason mid-way: each still
    # carries the audio generated so far.
    for finish in ("MAX_TOKENS", "SAFETY", "OTHER")
        reply = _sp_gemini_reply(_sp_inline("audio/L16;codec=pcm;rate=24000", pcm); finish)
        r = run(reply)
        @test r isa UniLM.AudioFailure && r.status == 200 && r.response == reply
        @test JSON.parse(r.response)["candidates"][1]["finishReason"] == finish
    end
    noreason = JSON.json(Dict("candidates" => [Dict("content" => Dict("role" => "model",
        "parts" => [_sp_inline("audio/L16;codec=pcm;rate=24000", pcm)]))]))
    @test run(noreason) isa UniLM.AudioFailure
    # A blocked prompt (no candidate) and a filtered candidate without audio are failures too.
    blocked = JSON.json(Dict("promptFeedback" => Dict("blockReason" => "SAFETY")))
    @test (b = run(blocked)) isa UniLM.AudioFailure && b.response == blocked
    @test run(_sp_gemini_reply(; finish="SAFETY")) isa UniLM.AudioFailure
    @test run(_sp_gemini_reply(_sp_inline("audio/L16;codec=pcm;rate=24000", pcm))) isa SpeechSuccess   # STOP
end

# ─── Translation on OpenAI Realtime ──────────────────────────────────────────

@testset "translate: a Realtime translation session in 200 ms frames, its transcript as the text" begin
    pcm = UniLM._translation_pcm(_SP_WAV)
    r, seen = _sp_with_realtime(_sp_translation_reply(["Le mot ", "secret est ", "banane."])) do
        translate(_SP_WAV; service=SpeechWSProbe, language="fr")
    end
    @test r isa TranscriptionSuccess && transcript_text(r) == "Le mot secret est banane."
    @test only(seen.targets) == "/translations?model=gpt-realtime-translate" && only(seen.auth) == "Bearer test-key"
    types = [ev["type"] for ev in seen.events]
    @test first(seen.events) == Dict{String,Any}("type" => "session.update", "session" => Dict{String,Any}(
        "audio" => Dict{String,Any}("output" => Dict{String,Any}("language" => "fr"))))
    @test last(types) == "session.close" && all(==("session.input_audio_buffer.append"), types[2:end-1])
    chunks = [base64decode(ev["audio"]) for ev in seen.events[2:end-1]]
    @test length(chunks) == cld(length(pcm), 9600)
    @test all(==(9600) ∘ length, chunks[1:end-1]) && length(last(chunks)) <= 9600
    @test reduce(vcat, chunks) == pcm
    # English is the default target, and the model is a query value.
    r2, seen2 = _sp_with_realtime(_sp_translation_reply(["The secret word is banana."])) do
        translate(_SP_WAV; service=SpeechWSProbe, model="gpt-realtime-translate-preview")
    end
    @test transcript_text(r2) == "The secret word is banana."
    @test only(seen2.targets) == "/translations?model=gpt-realtime-translate-preview"
    @test first(seen2.events)["session"]["audio"]["output"]["language"] == "en"
end

@testset "translate: an error event, an early close and silence are typed failures" begin
    err = _sp_event("error"; error=Dict("type" => "invalid_request_error", "code" => "unsupported_language",
                                        "message" => "Unsupported output language: xx.", "param" => nothing))
    r, seen = _sp_with_realtime(_sp_translation_reply(String[]; on_update=Any[err])) do
        translate(_SP_WAV; service=SpeechWSProbe, language="xx")
    end
    @test r isa UniLM.AudioFailure && isnothing(r.status) && occursin("Unsupported output language: xx.", r.response)
    @test [ev["type"] for ev in seen.events] == ["session.update"]    # no audio sent after the refusal
    r2, _ = _sp_with_realtime(_sp_translation_reply(String[]; on_close=Any[_sp_event("session.output_transcript.delta";
                                                                                        delta="Le mot"), :close])) do
        translate(_SP_WAV; service=SpeechWSProbe, language="fr")
    end
    @test r2 isa UniLM.AudioCallError && occursin("closed before session.closed", r2.error)
    r3, _ = _sp_with_realtime(_sp_translation_reply(String[]; on_close=Any[])) do
        translate(_SP_WAV; service=SpeechWSProbe, language="fr",
                  config=RequestConfig(stream_idle_timeout=1.0, total_deadline=Inf))
    end
    @test r3 isa UniLM.AudioCallError && r3.cause isa UniLMTimeout && r3.cause.phase === :stream_idle
    refusal(f) = try f(); nothing catch x; x end
    e = refusal(() -> translate(_SP_WAV; service=GEMINIServiceEndpoint))
    @test e isa ArgumentError && occursin("translate (OpenAI Realtime API) is not supported by GEMINIServiceEndpoint", e.msg)
end

@testset "translate: a peer that stops reading ends the call at its deadline, or at the idle bound" begin
    # 300 s of audio is about 19 MB of append events: more than the socket buffers hold, so the
    # sends park once the peer stops reading.
    mktempdir() do dir
        long = joinpath(dir, "long.wav")
        write(long, _sp_wav(zeros(UInt8, 2 * 24_000 * 300)))
        # Observed under a watchdog: a call that never returns fails the test instead of hanging the suite.
        bounded(cfg, limit) = (task = Threads.@spawn translate(long; service=SpeechWSProbe, config=cfg);
                               timedwait(() -> istaskdone(task), limit + 4.0) === :ok ? fetch(task) : nothing)
        _sp_with_stall_peer() do
            r = bounded(RequestConfig(connect_timeout=10.0, stream_idle_timeout=Inf, total_deadline=3.0), 3.0)
            @test r isa UniLM.AudioCallError && r.cause isa UniLMTimeout && r.cause.phase === :deadline &&
                  r.cause.limit == 3.0
            r2 = bounded(RequestConfig(connect_timeout=10.0, stream_idle_timeout=2.0, total_deadline=Inf), 2.0)
            @test r2 isa UniLM.AudioCallError && r2.cause isa UniLMTimeout && r2.cause.phase === :stream_idle &&
                  r2.cause.limit == 2.0
        end
    end
end

@testset "realtime_receive: a send parked on a peer that stopped reading cannot hold its idle bound open" begin
    # A send parked on full socket buffers holds HTTP.jl's send lock, which closing the WebSocket
    # waits for: the bound holds only if the read is ended without that lock.
    cfg = RequestConfig(connect_timeout=10.0, stream_idle_timeout=1.0, total_deadline=Inf)
    outcome, took = Ref{Any}(nothing), Ref(NaN)
    _sp_with_stall_peer() do
        call = Threads.@spawn realtime_connect(; service=SpeechWSProbe, config=cfg) do s
            realtime_receive(s); realtime_receive(s)          # session.created, session.updated
            big = input_audio_append(repeat("A", 1_000_000))
            sender = Threads.@spawn try
                while true
                    realtime_send(s, big)                     # parks once the socket buffers fill
                end
            catch e
                e
            end
            sleep(2.0)                                        # 1 MB per event: parked long before this
            t = time()
            outcome[] = try realtime_receive(s) catch e; e end
            took[] = time() - t
            fetch(sender)
        end
        @test timedwait(() -> !isnothing(outcome[]), 15.0) === :ok
        @test outcome[] isa UniLMTimeout && outcome[].phase === :stream_idle && outcome[].limit == 1.0
        @test took[] < 4.0
        # Ending the connection also unblocked the parked send, so the handler returned.
        @test timedwait(() -> istaskdone(call), 15.0) === :ok
    end
end
