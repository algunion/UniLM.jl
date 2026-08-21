@testset "Videos API — config seam wiring" begin
    @test _reached_seam(create_video(prompt="a cat", model="sora-2", service=SeamProbe, config=_TINY_DEADLINE), UniLM.VideoCallError)
    @test _reached_seam(retrieve_video("video_x"; service=SeamProbe, config=_TINY_DEADLINE), UniLM.VideoCallError)
    @test _reached_seam(list_videos(service=SeamProbe, config=_TINY_DEADLINE), UniLM.VideoCallError)
    @test _reached_seam(video_content("video_x"; service=SeamProbe, config=_TINY_DEADLINE), UniLM.VideoCallError)
end

@testset "Videos API — a separator-bearing id stays one path segment" begin
    t = _recorded_targets() do
        retrieve_video(_HOSTILE_ID; service=URLProbe)
        video_content(_HOSTILE_ID; service=URLProbe)
        list_videos(; limit=2, after=_HOSTILE_ID, service=URLProbe)
    end
    @test t == ["/v1/videos/$_HOSTILE_ENC",
                "/v1/videos/$_HOSTILE_ENC/content",
                "/v1/videos?limit=2&after=$_HOSTILE_ENC"]
    # Real ids are unreserved, so encoding must leave the wire byte-identical.
    g = _recorded_targets() do
        retrieve_video("video_abc123"; service=URLProbe)
        list_videos(; limit=2, after="video_abc123", service=URLProbe)
    end
    @test g == ["/v1/videos/video_abc123", "/v1/videos?limit=2&after=video_abc123"]
end
