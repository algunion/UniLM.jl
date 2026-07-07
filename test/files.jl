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
end
