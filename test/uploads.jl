@testset "Uploads API — config seam wiring" begin
    @test _seam_timeout(create_upload(filename="a.bin", purpose="assistants", bytes=4,
        mime_type="application/octet-stream", service=SeamProbe, config=_TINY_DEADLINE), UniLM.UploadCallError)
    @test _seam_timeout(add_upload_part("upload_x", UInt8[1,2,3]; service=SeamProbe, config=_TINY_DEADLINE), UniLM.UploadCallError)
    @test _seam_timeout(complete_upload("upload_x", ["part_1"]; service=SeamProbe, config=_TINY_DEADLINE), UniLM.UploadCallError)
    @test _seam_timeout(cancel_upload("upload_x"; service=SeamProbe, config=_TINY_DEADLINE), UniLM.UploadCallError)
end

@testset "create_upload accepts only the Files purposes" begin
    @test_throws ArgumentError create_upload(filename="a.bin", purpose="bogus", bytes=4,
        mime_type="application/octet-stream", service=SeamProbe)
    for p in ("assistants", "batch", "fine-tune", "vision", "user_data", "evals")
        @test _seam_timeout(create_upload(filename="a.bin", purpose=p, bytes=4,
            mime_type="application/octet-stream", service=SeamProbe, config=_TINY_DEADLINE), UniLM.UploadCallError)
    end
end

@testset "Uploads API — a failure keeps the request id the service sent" begin
    r = _answered(() -> cancel_upload("upload_x"; service=URLProbe), 400; headers=["x-request-id" => "req_upl"])
    @test r isa UniLM.UploadFailure && r.request_id == "req_upl"
end

@testset "Uploads API — a separator-bearing upload id stays one path segment" begin
    t = _recorded_targets() do
        add_upload_part(_HOSTILE_ID, UInt8[1, 2, 3]; service=URLProbe)
        complete_upload(_HOSTILE_ID, ["part_1"]; service=URLProbe)
        cancel_upload(_HOSTILE_ID; service=URLProbe)
    end
    @test t == ["/v1/uploads/$_HOSTILE_ENC/parts",
                "/v1/uploads/$_HOSTILE_ENC/complete",
                "/v1/uploads/$_HOSTILE_ENC/cancel"]
    g = _recorded_targets() do
        cancel_upload("upload_abc123"; service=URLProbe)
    end
    @test g == ["/v1/uploads/upload_abc123/cancel"]
end
