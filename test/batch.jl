@testset "Batch API — config seam wiring" begin
    @test _seam_timeout(create_batch("file_x", "/v1/chat/completions"; service=SeamProbe, config=_TINY_DEADLINE), UniLM.BatchCallError)
    @test _seam_timeout(retrieve_batch("batch_x"; service=SeamProbe, config=_TINY_DEADLINE), UniLM.BatchCallError)
    @test _seam_timeout(cancel_batch("batch_x"; service=SeamProbe, config=_TINY_DEADLINE), UniLM.BatchCallError)
    @test _seam_timeout(list_batches(service=SeamProbe, config=_TINY_DEADLINE), UniLM.BatchCallError)
    @test _seam_timeout(poll_batch("batch_x"; interval=0.01, timeout=0.05, service=SeamProbe, config=_TINY_DEADLINE), UniLM.BatchCallError)
end

@testset "Batch API — a failure keeps the request id the service sent" begin
    r = _answered(() -> retrieve_batch("batch_x"; service=URLProbe), 404; headers=["x-request-id" => "req_batch"])
    @test r isa UniLM.BatchFailure && r.request_id == "req_batch"
end

@testset "Batch API — a separator-bearing id stays one path segment" begin
    t = _recorded_targets() do
        retrieve_batch(_HOSTILE_ID; service=URLProbe)
        cancel_batch(_HOSTILE_ID; service=URLProbe)
        list_batches(; limit=2, after=_HOSTILE_ID, service=URLProbe)
    end
    @test t == ["/v1/batches/$_HOSTILE_ENC",
                "/v1/batches/$_HOSTILE_ENC/cancel",
                "/v1/batches?limit=2&after=$_HOSTILE_ENC"]
    g = _recorded_targets() do
        cancel_batch("batch_abc123"; service=URLProbe)
        list_batches(; after="batch_abc123", service=URLProbe)
    end
    @test g == ["/v1/batches/batch_abc123/cancel", "/v1/batches?after=batch_abc123"]
end
