@testset "Uploads API — config seam wiring" begin
    @test _reached_seam(create_upload(filename="a.bin", purpose="assistants", bytes=4,
        mime_type="application/octet-stream", service=SeamProbe, config=_TINY_DEADLINE), UniLM.UploadCallError)
    @test _reached_seam(add_upload_part("upload_x", UInt8[1,2,3]; service=SeamProbe, config=_TINY_DEADLINE), UniLM.UploadCallError)
    @test _reached_seam(complete_upload("upload_x", ["part_1"]; service=SeamProbe, config=_TINY_DEADLINE), UniLM.UploadCallError)
    @test _reached_seam(cancel_upload("upload_x"; service=SeamProbe, config=_TINY_DEADLINE), UniLM.UploadCallError)
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
