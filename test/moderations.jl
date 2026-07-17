@testset "Moderations API — config seam wiring" begin
    @test _reached_seam(moderate("hello"; service=SeamProbe, config=_TINY_DEADLINE), UniLM.ModerationCallError)
end
