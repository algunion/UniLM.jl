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
            @test _reached_seam(upload_file(u; config=_TINY_DEADLINE), FileCallError)
            @test _reached_seam(upload_file(fpath, "user_data"; service=SeamProbe, config=_TINY_DEADLINE), FileCallError)
            @test _reached_seam(list_files(service=SeamProbe, config=_TINY_DEADLINE), FileCallError)
            @test _reached_seam(retrieve_file("file-x"; service=SeamProbe, config=_TINY_DEADLINE), FileCallError)
            @test _reached_seam(delete_file("file-x"; service=SeamProbe, config=_TINY_DEADLINE), FileCallError)
            @test _reached_seam(file_content("file-x"; service=SeamProbe, config=_TINY_DEADLINE), FileCallError)
            @test_throws MethodError upload_file(u; retries=1)
        finally
            rm(fpath; force=true)
        end
    end
end

using Sockets

# Local upload target whose base URL is chosen after the listener binds.
struct FilesRetryProbe <: UniLM.ServiceEndpoint end
const _files_probe_base = Ref("")
UniLM._resolve_base_url(::Type{FilesRetryProbe}) = _files_probe_base[]
UniLM.auth_header(::Type{FilesRetryProbe}) =
    ["Authorization" => "Bearer t", "Content-Type" => "application/json"]
UniLM.provider_capabilities(::Type{FilesRetryProbe}) = Set([:files])

# Probe an ephemeral port, then serve on it; the close-then-rebind window can race.
function _files_retry_server(handler)
    for _ in 1:5
        tcp = Sockets.listen(Sockets.localhost, 0)
        port = Int(Sockets.getsockname(tcp)[2])
        close(tcp)
        try
            return HTTP.serve!(handler, "127.0.0.1", port; verbose=false), "http://127.0.0.1:$port"
        catch e
            e isa Base.IOError || rethrow()
        end
    end
    error("could not bind an ephemeral port for the upload retry fixture")
end

@testset "upload_file: a retried attempt sends a complete multipart body" begin
    # The seam disables HTTP.jl's own retry layer, so its mark/reset body rewind
    # never runs. One Form handed to the retry loop is left consumed by attempt 1;
    # attempt 2 would then put a zero-length body on the wire and a transient 503
    # would turn into a hard 400. Every attempt must build a fresh Form.
    payload = "multipart-retry-payload-" * repeat("x", 512)
    fpath = tempname() * ".txt"
    write(fpath, payload)
    seen = Vector{Int}()          # body length per attempt, in order
    complete = Ref(false)
    server, base = _files_retry_server(req -> begin
        body = String(copy(req.body))
        push!(seen, sizeof(body))
        if length(seen) == 1
            return HTTP.Response(503, ["Retry-After" => "0"], Vector{UInt8}("{}"))
        end
        complete[] = occursin(payload, body) && occursin("user_data", body)
        return HTTP.Response(200, ["Content-Type" => "application/json"],
                             Vector{UInt8}(JSON.json(Dict(
                                 "id" => "file-1", "bytes" => sizeof(payload), "created_at" => 1,
                                 "filename" => basename(fpath), "purpose" => "user_data",
                                 "status" => "processed"))))
    end)
    _files_probe_base[] = base
    try
        cfg = UniLM.RequestConfig(max_attempts=2, total_deadline=Inf)
        r = upload_file(fpath, "user_data"; service=FilesRetryProbe, config=cfg)
        @test length(seen) == 2                 # the 503 was actually retried
        # Attempt totals are not comparable: each rebuild draws a fresh random
        # boundary, whose hex width varies on HTTP.jl 1.x. A truncated or empty
        # replay still cannot reach the size of the payload it must carry.
        @test seen[2] >= sizeof(payload)
        @test complete[]                        # ...and carries the whole file + fields
        @test r isa FileSuccess
        @test r.response.id == "file-1"
    finally
        close(server)
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
