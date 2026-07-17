@testset "Realtime API — config seam wiring (secret minting only)" begin
    @test _reached_seam(mint_realtime_secret(service=SeamProbe, config=_TINY_DEADLINE), UniLM.RealtimeCallError)
end
