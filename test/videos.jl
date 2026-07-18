@testset "Videos API — config seam wiring" begin
    @test _reached_seam(create_video(prompt="a cat", model="sora-2", service=SeamProbe, config=_TINY_DEADLINE), UniLM.VideoCallError)
    @test _reached_seam(retrieve_video("video_x"; service=SeamProbe, config=_TINY_DEADLINE), UniLM.VideoCallError)
    @test _reached_seam(list_videos(service=SeamProbe, config=_TINY_DEADLINE), UniLM.VideoCallError)
    @test _reached_seam(video_content("video_x"; service=SeamProbe, config=_TINY_DEADLINE), UniLM.VideoCallError)
end
