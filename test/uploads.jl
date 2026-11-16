@testset "Uploads API — config seam wiring" begin
    @test _seam_timeout(create_upload(filename="a.bin", purpose="assistants", bytes=4,
        mime_type="application/octet-stream", service=SeamProbe, config=_TINY_DEADLINE), UniLM.UploadCallError)
    @test _seam_timeout(add_upload_part("upload_x", UInt8[1,2,3]; service=SeamProbe, config=_TINY_DEADLINE), UniLM.UploadCallError)
    @test _seam_timeout(complete_upload("upload_x", ["part_1"]; service=SeamProbe, config=_TINY_DEADLINE), UniLM.UploadCallError)
    @test _seam_timeout(cancel_upload("upload_x"; service=SeamProbe, config=_TINY_DEADLINE), UniLM.UploadCallError)
end

# OpenAI's CreateUploadRequest lists four purposes; the Uploads API accepted `user_data` as well on
# 2026-11-16 (an upload created with it, then cancelled), so create_upload takes the Files purposes.
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

@testset "an upload object without a field the Uploads reference requires is an UploadCallError" begin
    # OpenAI's Upload object has an `id`, `bytes`, `created_at`, `expires_at`, `filename`, `purpose` and
    # `status`, none of them optional, and a `file` that is optional or null: a reply missing one of the
    # fields UniLM reads is no Upload object, never an UploadSuccess whose status, name or size is nothing.
    full = Dict{String,Any}("id" => "upload_1", "object" => "upload", "bytes" => 3, "created_at" => 1,
                            "expires_at" => 2, "filename" => "a.jsonl", "purpose" => "batch", "status" => "pending")
    ok = _answered(() -> complete_upload("upload_1", ["part_1"]; service=URLProbe), 200; body=JSON.json(full))
    @test ok isa UploadSuccess && (ok.response.status, ok.response.filename, ok.response.bytes) == ("pending", "a.jsonl", 3)
    @test ok.response.raw["created_at"] == 1 && ok.response.raw["expires_at"] == 2 && ok.response.raw["purpose"] == "batch"
    for key in ("status", "filename", "bytes")
        r = _answered(() -> complete_upload("upload_1", ["part_1"]; service=URLProbe), 200;
                      body=JSON.json(delete!(copy(full), key)))
        @test r isa UniLM.UploadCallError && occursin(key, r.error)
    end
    # `file` is null until the upload completes.
    pending = _answered(() -> complete_upload("upload_1", ["part_1"]; service=URLProbe), 200;
                        body=JSON.json(merge(full, Dict("file" => nothing))))
    @test pending isa UploadSuccess && pending.response.file === nothing
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

# OpenAI's CreateUploadRequest takes `expires_after`, a FileExpirationAfter: `anchor` "created_at" and
# `seconds`, 3600 to 2592000.
# Refuted if: expires_after is not sent, is sent in another shape or when unset, or a value outside
# the range reaches the network.
@testset "create_upload sends expires_after as OpenAI's FileExpirationAfter, 3600 to 2592000 seconds" begin
    up = JSON.json(Dict("id" => "upload_1", "object" => "upload", "status" => "pending", "filename" => "a.jsonl",
                        "bytes" => 3, "purpose" => "batch", "created_at" => 1, "expires_at" => 3601))
    call(service; kw...) = create_upload(; filename="a.jsonl", purpose="batch", bytes=3, mime_type="application/jsonl",
                                         service, kw...)
    r, seen = _with_scripted(() -> call(URLProbe; expires_after=7200), (_, _) -> _json(200, up))
    @test r isa UploadSuccess
    @test JSON.parse(only(seen).body)["expires_after"] == Dict("anchor" => "created_at", "seconds" => 7200)
    _, seen = _with_scripted(() -> call(URLProbe), (_, _) -> _json(200, up))
    @test !haskey(JSON.parse(only(seen).body), "expires_after")
    for s in (3599, 2592001)
        e = try call(SeamProbe; expires_after=s); nothing catch x; x end
        @test e isa ArgumentError &&
              e.msg == "create_upload: OpenAI's Files API takes expires_after from 3600 to 2592000 seconds (got $s)"
    end
end
