@testset "Batch API — config seam wiring" begin
    @test _reached_seam(create_batch("file_x", "/v1/chat/completions"; service=SeamProbe, config=_TINY_DEADLINE), UniLM.BatchCallError)
    @test _reached_seam(retrieve_batch("batch_x"; service=SeamProbe, config=_TINY_DEADLINE), UniLM.BatchCallError)
    @test _reached_seam(cancel_batch("batch_x"; service=SeamProbe, config=_TINY_DEADLINE), UniLM.BatchCallError)
    @test _reached_seam(list_batches(service=SeamProbe, config=_TINY_DEADLINE), UniLM.BatchCallError)
    @test _reached_seam(poll_batch("batch_x"; interval=0.01, timeout=0.05, service=SeamProbe, config=_TINY_DEADLINE), UniLM.BatchCallError)
end
