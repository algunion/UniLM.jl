@testset "Files API (unit)" begin
    @testset "FileUpload validation" begin
        @test_throws ArgumentError UniLM.FileUpload(file="/no/such/file.txt", purpose="user_data")
        path = tempname() * ".txt"
        write(path, "hello")
        @test_throws ArgumentError UniLM.FileUpload(file=path, purpose="bogus")
        u = UniLM.FileUpload(file=path, purpose="user_data")
        @test u.purpose == "user_data"
        @test u.file == path
        rm(path)
    end

    @testset "parse FileObject" begin
        f = UniLM._parse_file_object(Dict{String,Any}(
            "id" => "file-1", "bytes" => 10, "created_at" => 1,
            "filename" => "a.txt", "purpose" => "user_data", "status" => "processed"))
        @test f.id == "file-1"
        @test f.bytes == 10
        @test f.filename == "a.txt"
        @test f.status == "processed"
    end

    @testset "mime types (incl. audio)" begin
        @test UniLM._mime_for("a.wav") == "audio/wav"
        @test UniLM._mime_for("a.mp3") == "audio/mpeg"
        @test UniLM._mime_for("a.flac") == "audio/flac"
        @test UniLM._mime_for("a.m4a") == "audio/mp4"
        @test UniLM._mime_for("a.json") == "application/json"
        @test UniLM._mime_for("a.unknown") == "application/octet-stream"
    end

    @testset "auth_header_multipart strips content-type" begin
        g = UniLM.GenericOpenAIEndpoint("http://x", "k")
        h = UniLM.auth_header_multipart(g)
        @test !any(p -> lowercase(String(first(p))) == "content-type", h)
        @test any(p -> first(p) == "Authorization", h)
    end

    @testset "capability gating (non-OpenAI rejects :files)" begin
        @test_throws ArgumentError list_files(service=UniLM.GEMINIOpenAIServiceEndpoint)
        @test UniLM.has_capability(UniLM.OPENAIServiceEndpoint, :files)
    end

    @testset "config seam wiring" begin
        fpath = tempname() * ".txt"
        write(fpath, "probe")
        try
            u = UniLM.FileUpload(service=SeamProbe, file=fpath, purpose="user_data")
            @test _seam_timeout(upload_file(u; config=_TINY_DEADLINE), FileCallError)
            @test _seam_timeout(upload_file(fpath, "user_data"; service=SeamProbe, config=_TINY_DEADLINE), FileCallError)
            @test _seam_timeout(list_files(service=SeamProbe, config=_TINY_DEADLINE), FileCallError)
            @test _seam_timeout(retrieve_file("file-x"; service=SeamProbe, config=_TINY_DEADLINE), FileCallError)
            @test _seam_timeout(delete_file("file-x"; service=SeamProbe, config=_TINY_DEADLINE), FileCallError)
            @test _seam_timeout(file_content("file-x"; service=SeamProbe, config=_TINY_DEADLINE), FileCallError)
            @test_throws MethodError upload_file(u; retries=1)
        finally
            rm(fpath; force=true)
        end
    end
end

@testset "Files API — a failure keeps the request id the service sent" begin
    r = _answered(404; headers=["x-request-id" => "req_files"]) do
        retrieve_file("file-x"; service=URLProbe)
    end
    @test r isa FileFailure && r.status == 404
    @test r.request_id == "req_files"
    # OpenAI-wire servers other than OpenAI name the header `request-id`.
    @test _answered(() -> list_files(; service=URLProbe), 500;
                    headers=["request-id" => "req_plain"]).request_id == "req_plain"
    @test isnothing(_answered(() -> delete_file("f"; service=URLProbe), 404).request_id)
end

@testset "delete_file reports success only when the service confirms the delete" begin
    del(body) = _answered(() -> delete_file("file-1"; service=URLProbe), 200; body)
    @test del("""{"id": "file-1", "object": "file", "deleted": true}""") ==
          FileDeleteSuccess(id="file-1", deleted=true)
    unconfirmed = del("""{"id": "file-1", "object": "file"}""")
    @test unconfirmed isa FileCallError && occursin("deleted", unconfirmed.error)
    @test del("""{"id": "file-1", "object": "file", "deleted": false}""") isa FileCallError
end

@testset "upload_file: a create is sent once, never retried" begin
    # A create is not idempotent. A POST the client stopped waiting for, or a gateway
    # 5xx sent after the backend stored the file, may already have created the file,
    # so a second attempt can store it twice — while max_attempts here would allow three.
    payload = "single-create-payload-" * repeat("x", 256)
    fpath = tempname() * ".txt"
    write(fpath, payload)
    ok = JSON.json(Dict("id" => "file-1", "bytes" => sizeof(payload), "created_at" => 1,
                        "filename" => basename(fpath), "purpose" => "user_data"))
    cfg = UniLM.RequestConfig(request_timeout=1.0, total_deadline=30.0, max_attempts=3)
    try
        # The one attempt carries the whole file and the purpose field.
        done, sent = _with_scripted((_, _) -> _json(200, ok)) do
            upload_file(fpath, "user_data"; service=URLProbe, config=cfg)
        end
        @test done isa FileSuccess && done.response.id == "file-1"
        @test occursin(payload, only(sent).body) && occursin("user_data", only(sent).body)

        busy, seen503 = _with_scripted((_, _) -> _json(503, "{}")) do
            upload_file(fpath, "user_data"; service=URLProbe, config=cfg)
        end
        @test busy isa FileFailure && busy.status == 503   # returned as it came
        @test length(seen503) == 1

        # The first reply is held until the call has returned, so the attempt can only
        # end in its request_timeout; any second POST would be a retry.
        release = Base.Event()
        slow, seen = _with_scripted((n, _) -> (n == 1 && wait(release); _json(200, ok))) do
            try
                upload_file(fpath, "user_data"; service=URLProbe, config=cfg)
            finally
                notify(release)
            end
        end
        @test count(q -> q.method == "POST", seen) == 1
        @test slow isa FileCallError && slow.cause isa UniLMTimeout
    finally
        rm(fpath; force=true)
    end
end

@testset "Files API — a separator-bearing id and filter stay single values" begin
    t = _recorded_targets() do
        retrieve_file(_HOSTILE_ID; service=URLProbe)
        delete_file(_HOSTILE_ID; service=URLProbe)
        file_content(_HOSTILE_ID; service=URLProbe)
        list_files(; purpose=_HOSTILE_ID, limit=2, after=_HOSTILE_ID, service=URLProbe)
    end
    @test t == ["/v1/files/$_HOSTILE_ENC",
                "/v1/files/$_HOSTILE_ENC",
                "/v1/files/$_HOSTILE_ENC/content",
                "/v1/files?purpose=$_HOSTILE_ENC&limit=2&after=$_HOSTILE_ENC"]
    g = _recorded_targets() do
        retrieve_file("file-abc123"; service=URLProbe)
        list_files(; purpose="user_data", after="file-abc123", service=URLProbe)
    end
    @test g == ["/v1/files/file-abc123", "/v1/files?purpose=user_data&after=file-abc123"]
end
