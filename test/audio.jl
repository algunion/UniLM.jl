@testset "Audio API — config seam wiring" begin
    apath = tempname() * ".wav"
    write(apath, UInt8[0x52, 0x49, 0x46, 0x46])   # "RIFF"; content irrelevant (never sent)
    try
        s = UniLM.SpeechRequest(service=SeamProbe, input="hi", model="tts")
        @test _reached_seam(speak(s; config=_TINY_DEADLINE), UniLM.AudioCallError)
        @test _reached_seam(speak("hi"; service=SeamProbe, model="tts", config=_TINY_DEADLINE), UniLM.AudioCallError)
        t = UniLM.TranscriptionRequest(service=SeamProbe, file=apath, model="stt")
        @test _reached_seam(transcribe(t; config=_TINY_DEADLINE), UniLM.AudioCallError)
        @test _reached_seam(translate(t; config=_TINY_DEADLINE), UniLM.AudioCallError)
        @test _reached_seam(transcribe(apath; service=SeamProbe, model="stt", config=_TINY_DEADLINE), UniLM.AudioCallError)
    finally
        rm(apath; force=true)
    end
end
