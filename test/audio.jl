@testset "Audio API — config seam wiring" begin
    apath = tempname() * ".wav"
    write(apath, UInt8[0x52, 0x49, 0x46, 0x46])   # "RIFF"; content irrelevant (never sent)
    try
        s = UniLM.SpeechRequest(service=SeamProbe, input="hi", model="tts")
        @test _seam_timeout(speak(s; config=_TINY_DEADLINE), UniLM.AudioCallError)
        @test _seam_timeout(speak("hi"; service=SeamProbe, model="tts", config=_TINY_DEADLINE), UniLM.AudioCallError)
        t = UniLM.TranscriptionRequest(service=SeamProbe, file=apath, model="stt")
        @test _seam_timeout(transcribe(t; config=_TINY_DEADLINE), UniLM.AudioCallError)
        @test _seam_timeout(translate(t; config=_TINY_DEADLINE), UniLM.AudioCallError)
        @test _seam_timeout(transcribe(apath; service=SeamProbe, model="stt", config=_TINY_DEADLINE), UniLM.AudioCallError)
    finally
        rm(apath; force=true)
    end
end

@testset "Audio API — a failure keeps the request id the service sent" begin
    r = _answered(() -> speak("hi"; service=URLProbe, model="tts"), 401; headers=["x-request-id" => "req_audio"])
    @test r isa UniLM.AudioFailure && r.request_id == "req_audio"
end
