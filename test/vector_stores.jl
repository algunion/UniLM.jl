@testset "Vector Stores API — config seam wiring" begin
    @test _seam_timeout(create_vector_store(service=SeamProbe, config=_TINY_DEADLINE), UniLM.VectorStoreCallError)
    @test _seam_timeout(retrieve_vector_store("vs_x"; service=SeamProbe, config=_TINY_DEADLINE), UniLM.VectorStoreCallError)
    @test _seam_timeout(list_vector_stores(service=SeamProbe, config=_TINY_DEADLINE), UniLM.VectorStoreCallError)
    @test _seam_timeout(delete_vector_store("vs_x"; service=SeamProbe, config=_TINY_DEADLINE), UniLM.VectorStoreCallError)
    @test _seam_timeout(add_vector_store_file("vs_x", "file_x"; service=SeamProbe, config=_TINY_DEADLINE), UniLM.VectorStoreCallError)
    @test _seam_timeout(create_file_batch("vs_x", ["file_x"]; service=SeamProbe, config=_TINY_DEADLINE), UniLM.VectorStoreCallError)
    @test _seam_timeout(retrieve_file_batch("vs_x", "batch_x"; service=SeamProbe, config=_TINY_DEADLINE), UniLM.VectorStoreCallError)
    # poll_file_batch forwards config to retrieve_file_batch (one iteration reaches the seam)
    @test _seam_timeout(poll_file_batch("vs_x", "batch_x"; interval=0.01, timeout=0.05, service=SeamProbe, config=_TINY_DEADLINE), UniLM.VectorStoreCallError)
end

@testset "Vector Stores API — a failure keeps the request id the service sent" begin
    r = _answered(() -> retrieve_vector_store("vs_x"; service=URLProbe), 404;
                  headers=["x-request-id" => "req_vs"])
    @test r isa UniLM.VectorStoreFailure && r.request_id == "req_vs"
end

@testset "Vector Stores API — separator-bearing ids stay single path segments" begin
    t = _recorded_targets() do
        retrieve_vector_store(_HOSTILE_ID; service=URLProbe)
        delete_vector_store(_HOSTILE_ID; service=URLProbe)
        add_vector_store_file(_HOSTILE_ID, "file-1"; service=URLProbe)
        create_file_batch(_HOSTILE_ID, ["file-1"]; service=URLProbe)
        retrieve_file_batch(_HOSTILE_ID, _HOSTILE_ID; service=URLProbe)
        list_vector_stores(; limit=2, after=_HOSTILE_ID, service=URLProbe)
    end
    @test t == ["/v1/vector_stores/$_HOSTILE_ENC",
                "/v1/vector_stores/$_HOSTILE_ENC",
                "/v1/vector_stores/$_HOSTILE_ENC/files",
                "/v1/vector_stores/$_HOSTILE_ENC/file_batches",
                "/v1/vector_stores/$_HOSTILE_ENC/file_batches/$_HOSTILE_ENC",
                "/v1/vector_stores?limit=2&after=$_HOSTILE_ENC"]
    g = _recorded_targets() do
        retrieve_file_batch("vs_abc123", "vsfb_abc123"; service=URLProbe)
        list_vector_stores(; after="vs_abc123", service=URLProbe)
    end
    @test g == ["/v1/vector_stores/vs_abc123/file_batches/vsfb_abc123",
                "/v1/vector_stores?after=vs_abc123"]
end
