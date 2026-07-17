@testset "Vector Stores API — config seam wiring" begin
    @test _reached_seam(create_vector_store(service=SeamProbe, config=_TINY_DEADLINE), UniLM.VectorStoreCallError)
    @test _reached_seam(retrieve_vector_store("vs_x"; service=SeamProbe, config=_TINY_DEADLINE), UniLM.VectorStoreCallError)
    @test _reached_seam(list_vector_stores(service=SeamProbe, config=_TINY_DEADLINE), UniLM.VectorStoreCallError)
    @test _reached_seam(delete_vector_store("vs_x"; service=SeamProbe, config=_TINY_DEADLINE), UniLM.VectorStoreCallError)
    @test _reached_seam(add_vector_store_file("vs_x", "file_x"; service=SeamProbe, config=_TINY_DEADLINE), UniLM.VectorStoreCallError)
    @test _reached_seam(create_file_batch("vs_x", ["file_x"]; service=SeamProbe, config=_TINY_DEADLINE), UniLM.VectorStoreCallError)
    @test _reached_seam(retrieve_file_batch("vs_x", "batch_x"; service=SeamProbe, config=_TINY_DEADLINE), UniLM.VectorStoreCallError)
    # poll_file_batch forwards config to retrieve_file_batch (one iteration reaches the seam)
    @test _reached_seam(poll_file_batch("vs_x", "batch_x"; interval=0.01, timeout=0.05, service=SeamProbe, config=_TINY_DEADLINE), UniLM.VectorStoreCallError)
end
