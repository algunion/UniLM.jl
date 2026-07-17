@testset "Uploads API — config seam wiring" begin
    @test _reached_seam(create_upload(filename="a.bin", purpose="assistants", bytes=4,
        mime_type="application/octet-stream", service=SeamProbe, config=_TINY_DEADLINE), UniLM.UploadCallError)
    @test _reached_seam(add_upload_part("upload_x", UInt8[1,2,3]; service=SeamProbe, config=_TINY_DEADLINE), UniLM.UploadCallError)
    @test _reached_seam(complete_upload("upload_x", ["part_1"]; service=SeamProbe, config=_TINY_DEADLINE), UniLM.UploadCallError)
    @test _reached_seam(cancel_upload("upload_x"; service=SeamProbe, config=_TINY_DEADLINE), UniLM.UploadCallError)
end
