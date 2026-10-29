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
    @test_throws ArgumentError create_upload(filename="a.bin", purpose="evals", bytes=4,
        mime_type="application/octet-stream", service=SeamProbe)
    for p in ("assistants", "batch", "fine-tune", "vision", "user_data")
        @test _seam_timeout(create_upload(filename="a.bin", purpose=p, bytes=4,
            mime_type="application/octet-stream", service=SeamProbe, config=_TINY_DEADLINE), UniLM.UploadCallError)
    end
end

@testset "a completed upload whose file size is not yet known decodes, its bytes nothing" begin
    # Files reference, `bytes`: "In a completed file upload response, this can be null when the file
    # size is not yet available."
    file = """{"id": "file-1", "object": "file", "bytes": null, "created_at": 1, "filename": "a.jsonl",
               "purpose": "batch", "status": "processed"}"""
    upload = """{"id": "upload_1", "object": "upload", "status": "completed", "filename": "a.jsonl", "bytes": 3,
                 "purpose": "batch", "file": $file}"""
    r = _answered(() -> complete_upload("upload_1", ["part_1"]; service=URLProbe), 200; body=upload)
    @test r isa UploadSuccess && r.response.status == "completed"
    @test r.response.file.id == "file-1" && r.response.file.bytes === nothing
    f = _answered(() -> retrieve_file("file-1"; service=URLProbe), 200; body=file)
    @test f isa FileSuccess && f.response.bytes === nothing
    path = tempname() * ".jsonl"
    write(path, "{}
")
    try
        u = _answered(() -> upload_file(path, "batch"; service=URLProbe), 200; body=file)
        @test u isa FileSuccess && u.response.bytes === nothing
    finally
        rm(path; force=true)
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
