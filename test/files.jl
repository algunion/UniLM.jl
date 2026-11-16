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

    @testset "the Files purposes; evals is refused, naming the Evals shutdown" begin
        path = tempname() * ".jsonl"
        write(path, "{}")
        try
            @test UniLM._FILE_PURPOSES == ("assistants", "batch", "fine-tune", "vision", "user_data")
            for p in UniLM._FILE_PURPOSES
                @test UniLM.FileUpload(file=path, purpose=p).purpose == p
            end
            e = try UniLM.FileUpload(file=path, purpose="evals"); nothing catch err; err end
            @test e isa ArgumentError && e.msg ==
                "purpose \"evals\" is for OpenAI's Evals API, which is read-only from 2026-10-31 and shuts down on " *
                "2026-11-30 (https://developers.openai.com/api/docs/deprecations); the Files API takes " *
                "\"assistants\", \"batch\", \"fine-tune\", \"vision\" or \"user_data\""
            # Refused before any request: the seam probe would answer one with a timeout.
            @test_throws ArgumentError upload_file(path, "evals"; service=SeamProbe, config=_TINY_DEADLINE)
            # Any other purpose names the same list.
            e = try UniLM.FileUpload(file=path, purpose="bogus"); nothing catch err; err end
            @test e isa ArgumentError && e.msg ==
                "the Files API takes \"assistants\", \"batch\", \"fine-tune\", \"vision\" or \"user_data\" (got \"bogus\")"
        finally
            rm(path; force=true)
        end
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

@testset "save_file_content replaces the destination atomically" begin
    mktempdir() do dir
        path = joinpath(dir, "out.bin")
        write(path, "previous contents")
        before = stat(path).inode
        @test save_file_content(FileContentSuccess(content=Vector{UInt8}("new bytes")), path) == path
        @test read(path, String) == "new bytes"
        @test stat(path).inode != before   # renamed into place, never truncated and rewritten
        @test readdir(dir) == ["out.bin"]   # no temporary file left behind
        # A symlink is written through, as opening it for writing would be.
        link = joinpath(dir, "link.bin")
        symlink(path, link)
        save_file_content(FileContentSuccess(content=Vector{UInt8}("via link")), link)
        @test islink(link) && read(path, String) == "via link"
        # A directory is refused, never replaced.
        sub = mkdir(joinpath(dir, "sub"))
        write(joinpath(sub, "keep.txt"), "x")
        @test_throws ArgumentError save_file_content(FileContentSuccess(content=UInt8[1]), sub)
        @test isfile(joinpath(sub, "keep.txt"))
    end
end

@testset "save_file_content keeps the destination's mode and writes through a dangling symlink" begin
    save(bytes, path) = save_file_content(FileContentSuccess(content=Vector{UInt8}(bytes)), path)
    mktempdir() do dir
        # The renamed-in file takes the destination's permission bits, not the umask
        # default: a file the caller made private stays private.
        path = joinpath(dir, "secret.jsonl")
        write(path, "old")
        for mode in (0o600, 0o640)
            chmod(path, mode)
            save("new", path)
            @test read(path, String) == "new"
            @test filemode(path) & 0o7777 == mode
        end
        # A link whose target does not exist yet is written through, as opening it for
        # writing would be: the target is created and the link survives.
        target, link = joinpath(dir, "target.bin"), joinpath(dir, "link.bin")
        symlink(target, link)
        save("x", link)
        @test islink(link) && readlink(link) == target
        @test read(target, String) == "x"
        # ...a relative target resolves against the link's directory, through a chain.
        symlink("hop.bin", joinpath(dir, "rel.bin"))
        symlink("final.bin", joinpath(dir, "hop.bin"))
        save("y", joinpath(dir, "rel.bin"))
        @test islink(joinpath(dir, "rel.bin")) && islink(joinpath(dir, "hop.bin"))
        @test read(joinpath(dir, "final.bin"), String) == "y"
        # A symlink loop has no target to write: refused (ELOOP), nothing replaced.
        symlink("loop_b", joinpath(dir, "loop_a"))
        symlink("loop_a", joinpath(dir, "loop_b"))
        @test_throws Base.IOError save("z", joinpath(dir, "loop_a"))
        @test islink(joinpath(dir, "loop_a")) && islink(joinpath(dir, "loop_b"))
        @test sort(readdir(dir)) == sort(["secret.jsonl", "target.bin", "link.bin", "rel.bin",
                                          "hop.bin", "final.bin", "loop_a", "loop_b"])
    end
end

@testset "delete_file reports success only when the service confirms the delete" begin
    del(body) = _answered(() -> delete_file("file-1"; service=URLProbe), 200; body)
    @test del("""{"id": "file-1", "object": "file", "deleted": true}""") ==
          FileDeleteSuccess(id="file-1", deleted=true, object="file")
    unconfirmed = del("""{"id": "file-1", "object": "file"}""")
    @test unconfirmed isa FileCallError && occursin("deleted", unconfirmed.error)
    @test del("""{"id": "file-1", "object": "file", "deleted": false}""") isa FileCallError
end

@testset "a file object without a field the Files reference requires is a FileCallError" begin
    # OpenAI's File object has an `id`, `bytes` ("this can be null when the file size is not yet
    # available"), `created_at`, `filename` and `purpose`: a reply missing one is no File object, never a
    # FileSuccess whose size, time or name was made up for it.
    full = Dict{String,Any}("id" => "file-1", "object" => "file", "bytes" => 3, "created_at" => 1,
                            "filename" => "a.jsonl", "purpose" => "batch")
    page(f) = JSON.json(Dict("object" => "list", "data" => [f], "has_more" => false))
    @test _answered(() -> retrieve_file("file-1"; service=URLProbe), 200; body=JSON.json(full)) isa FileSuccess
    @test _answered(() -> list_files(; service=URLProbe), 200; body=page(full)) isa FileListSuccess
    for key in ("bytes", "created_at", "filename", "purpose")
        partial = delete!(copy(full), key)
        r = _answered(() -> retrieve_file("file-1"; service=URLProbe), 200; body=JSON.json(partial))
        @test r isa FileCallError && occursin(key, r.error)
        @test _answered(() -> list_files(; service=URLProbe), 200; body=page(partial)) isa FileCallError
        done = JSON.json(Dict("id" => "upload_1", "object" => "upload", "status" => "completed", "bytes" => 3,
                              "created_at" => 1, "expires_at" => 2, "filename" => "a.jsonl", "purpose" => "batch",
                              "file" => partial))
        @test _answered(() -> complete_upload("upload_1", ["part_1"]; service=URLProbe), 200; body=done) isa
              UniLM.UploadCallError
    end
end

# OpenAI's File object requires `filename` and `purpose` as strings; of its fields only `bytes` may be null.
# Refuted if: a reply whose filename or purpose is null decodes as a file, from a retrieve, a listing or an upload.
@testset "an OpenAI file object whose filename or purpose is null is a FileCallError" begin
    full = Dict{String,Any}("id" => "file-1", "object" => "file", "bytes" => 3, "created_at" => 1,
                            "filename" => "a.jsonl", "purpose" => "batch")
    for key in ("filename", "purpose")
        null = merge(full, Dict{String,Any}(key => nothing))
        r = _answered(() -> retrieve_file("file-1"; service=URLProbe), 200; body=JSON.json(null))
        @test r isa FileCallError && occursin(key, r.error)
        page = JSON.json(Dict("object" => "list", "data" => [null], "has_more" => false))
        @test _answered(() -> list_files(; service=URLProbe), 200; body=page) isa FileCallError
        done = JSON.json(Dict("id" => "upload_1", "object" => "upload", "status" => "completed", "bytes" => 3,
                              "created_at" => 1, "expires_at" => 2, "filename" => "a.jsonl", "purpose" => "batch",
                              "file" => null))
        @test _answered(() -> complete_upload("upload_1", ["part_1"]; service=URLProbe), 200; body=done) isa
              UniLM.UploadCallError
    end
end

@testset "upload_file: a cancelled token reads no file" begin
    # The file exists, but its mode forbids reading it: a call that read it would end in that
    # SystemError. (Root reads it anyway; the result is the same.)
    fpath = tempname() * ".jsonl"
    write(fpath, "{}\n")
    chmod(fpath, 0o000)
    try
        r, seen = _with_scripted(() -> upload_file(fpath, "batch"; service=URLProbe, cancel=cancel!(CancelToken())),
                                 (_, _) -> _json(200, "{}"))
        @test (typeof(r), typeof(r.cause), length(seen)) == (FileCallError, UniLMCancelled, 0)
    finally
        chmod(fpath, 0o600)
        rm(fpath; force=true)
    end
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

# OpenAI's ListFilesResponse requires `data` and `has_more`.
# Refuted if: a 200 without either decodes as a page — `{}` as an empty last page.
@testset "list_files: a 200 without data or has_more is a FileCallError, never an empty page" begin
    page(d) = _answered(() -> list_files(; service=URLProbe), 200; body=JSON.json(d))
    r = page(Dict())
    @test r isa FileCallError && occursin("\"data\"", r.error)
    r = page(Dict("object" => "list", "data" => []))
    @test r isa FileCallError && occursin("\"has_more\"", r.error)
    @test page(Dict("object" => "list", "data" => Dict(), "has_more" => false)) isa FileCallError
    # The reply OpenAI sent for `order=asc&limit=1` on 2026-11-16: an empty page with more after it.
    ok = page(Dict("object" => "list", "data" => [], "has_more" => true, "first_id" => nothing, "last_id" => nothing))
    @test ok isa FileListSuccess && isempty(ok.response.data) && ok.response.has_more
end

# OpenAI's DeleteFileResponse requires `id`.
# Refuted if: a reply without it reports the requested id as the one deleted.
@testset "delete_file: a reply without the deleted file's id is a FileCallError, never the requested id" begin
    r = _answered(() -> delete_file("file-1"; service=URLProbe), 200; body="""{"object": "file", "deleted": true}""")
    @test r isa FileCallError && occursin("\"id\"", r.error)
    r = _answered(() -> delete_file("file-1"; service=URLProbe), 200; body="""{"id": 7, "object": "file", "deleted": true}""")
    @test r isa FileCallError && occursin("\"id\"", r.error)
end

# OpenAI's DeleteFileResponse: `id`, `object` (always "file") and `deleted`, all required.
# Refuted if: the reply's `object` is dropped, or a reply without it decodes; or the two-field
# positional form no longer builds a result.
@testset "delete_file decodes the reply's object, which OpenAI's reference requires" begin
    del(body) = _answered(() -> delete_file("file-1"; service=URLProbe), 200; body)
    # The reply OpenAI sent on 2026-11-16.
    r = del("""{"object": "file", "deleted": true, "id": "file-72AHrfpfi5ccZAiuHbFdhf"}""")
    @test r isa FileDeleteSuccess && (r.id, r.deleted, r.object) == ("file-72AHrfpfi5ccZAiuHbFdhf", true, "file")
    r = del("""{"id": "file-1", "deleted": true}""")
    @test r isa FileCallError && occursin("\"object\"", r.error)
    @test FileDeleteSuccess("file-1", true) == FileDeleteSuccess(id="file-1", deleted=true, object=nothing)
end

# OpenAI's DeleteFileResponse: `object` is always "file".
# Refuted if: a delete reply that names another object (a vector store file's delete, a batch) reports a deleted file.
@testset "delete_file on OpenAI: a reply whose object is not \"file\" is a FileCallError" begin
    for object in ("vector_store.file.deleted", "batch", "")
        r = _answered(() -> delete_file("file-1"; service=URLProbe), 200;
                      body=JSON.json(Dict("id" => "file-1", "object" => object, "deleted" => true)))
        @test r isa FileCallError
        r isa FileCallError && @test occursin(repr(object), r.error)
    end
end

# OpenAI's FileExpirationAfter: `anchor` "created_at" and `seconds`, 3600 to 2592000; a form sends the
# object as bracketed fields (the Files reference's curl: `-F expires_after[anchor]="created_at"`).
# Refuted if: expires_after is not sent, is sent in another shape or when unset, or a value outside
# the range reaches the network.
@testset "upload_file sends expires_after as OpenAI's bracketed form fields, 3600 to 2592000 seconds" begin
    path = tempname() * ".jsonl"
    write(path, "{}\n")
    ok = JSON.json(Dict("id" => "file-1", "object" => "file", "bytes" => 3, "created_at" => 1, "expires_at" => 3601,
                        "filename" => basename(path), "purpose" => "batch", "status" => "processed"))
    try
        for call in (() -> upload_file(path, "batch"; service=URLProbe, expires_after=3600),
                     () -> upload_file(FileUpload(service=URLProbe, file=path, purpose="batch", expires_after=3600)))
            r, seen = _with_scripted(call, (_, _) -> _json(200, ok))
            body = only(seen).body
            @test r isa FileSuccess && r.response.raw["expires_at"] == 3601
            @test occursin("name=\"expires_after[anchor]\"\r\n\r\ncreated_at\r\n", body)
            @test occursin("name=\"expires_after[seconds]\"\r\n\r\n3600\r\n", body)
        end
        _, seen = _with_scripted(() -> upload_file(path, "batch"; service=URLProbe), (_, _) -> _json(200, ok))
        @test !occursin("expires_after", only(seen).body)
        for s in (3599, 2592001)
            e = try upload_file(path, "batch"; service=SeamProbe, expires_after=s); nothing catch x; x end
            @test e isa ArgumentError &&
                  e.msg == "upload_file: OpenAI's Files API takes expires_after from 3600 to 2592000 seconds (got $s)"
        end
        @test FileUpload(file=path, purpose="batch", expires_after=2592000).expires_after == 2592000
        # The three-field positional form still builds a request.
        @test FileUpload(SeamProbe, path, "batch").expires_after === nothing
    finally
        rm(path; force=true)
    end
end

# OpenAI's listFiles `order`: "asc" or "desc", by creation time.
# Refuted if: order is not sent, or is sent when unset, or another value reaches the network.
@testset "list_files sends order, \"asc\" or \"desc\"" begin
    t = _recorded_targets() do
        list_files(; order="asc", service=URLProbe)
        list_files(; purpose="batch", limit=2, after="file-1", order="desc", service=URLProbe)
        list_files(; service=URLProbe)
    end
    @test t == ["/v1/files?order=asc", "/v1/files?purpose=batch&limit=2&after=file-1&order=desc", "/v1/files"]
    e = try list_files(; order="sideways", service=SeamProbe); nothing catch x; x end
    @test e isa ArgumentError && e.msg == "list_files: order is \"asc\" or \"desc\" (got \"sideways\")"
end

# ── Anthropic's Files API ────────────────────────────────────────────────────
# Its replies here are those Anthropic sent on 2026-11-16 for a 5-byte text file.
const _ANTHROPIC_FILE = """{"type":"file","id":"file_01YLNcm6NoeziqXvw8FX4zaY","size_bytes":5,""" *
    """"created_at":"2026-11-16T20:10:09.427661Z","expires_at":"2026-11-16T21:10:09.427661Z","filename":"w.txt",""" *
    """"mime_type":"text/plain","downloadable":false}"""

# Refuted if: an Anthropic endpoint, the type or an instance, refuses the Files verbs, or sends them
# anywhere but api.anthropic.com/v1/files.
@testset "Anthropic: the Files verbs are served, at api.anthropic.com/v1/files" begin
    path = tempname() * ".txt"
    write(path, "hello")
    try
        withenv("ANTHROPIC_API_KEY" => "sk-ant-local-0000") do
            for s in (ANTHROPICServiceEndpoint, ANTHROPICServiceEndpoint())
                @test has_capability(s, :files)
                @test _seam_timeout(upload_file(path; service=s, config=_TINY_DEADLINE), FileCallError)
                @test _seam_timeout(list_files(; service=s, config=_TINY_DEADLINE), FileCallError)
                @test _seam_timeout(retrieve_file("file_1"; service=s, config=_TINY_DEADLINE), FileCallError)
                @test _seam_timeout(delete_file("file_1"; service=s, config=_TINY_DEADLINE), FileCallError)
                @test _seam_timeout(file_content("file_1"; service=s, config=_TINY_DEADLINE), FileCallError)
            end
        end
        @test UniLM._files_url(ANTHROPICServiceEndpoint) == UniLM._files_url(ANTHROPICServiceEndpoint()) ==
              "https://api.anthropic.com/v1/files"
    finally
        rm(path; force=true)
    end
end

# ── Gemini's Files API ───────────────────────────────────────────────────────
# Its replies here are those Gemini sent on 2026-11-16 for a 5-byte text file (`sha256Hash` shortened).
const _GEMINI_FILE = """{"name":"files/rvbxh1rsu85u","displayName":"w.txt","mimeType":"text/plain","sizeBytes":"5",""" *
    """"createTime":"2026-11-16T20:14:01.385879Z","updateTime":"2026-11-16T20:14:01.385879Z",""" *
    """"expirationTime":"2026-11-18T20:14:00.555605689Z","sha256Hash":"MmNm",""" *
    """"uri":"https://generativelanguage.googleapis.com/v1beta/files/rvbxh1rsu85u","state":"ACTIVE","source":"UPLOADED"}"""

# Refuted if: the native Gemini endpoint refuses the Files verbs or registration, sends them anywhere but
# generativelanguage.googleapis.com, or asks Gemini for an uploaded file's bytes, which it does not serve.
@testset "Gemini: the Files verbs and register_files are served, at generativelanguage.googleapis.com" begin
    path = tempname() * ".txt"
    write(path, "hello")
    g = GEMINIServiceEndpoint
    try
        withenv("GEMINI_API_KEY" => "local-0000") do
            @test has_capability(g, :files) && has_capability(g, :file_registration)
            @test _seam_timeout(upload_file(path; service=g, config=_TINY_DEADLINE), FileCallError)
            @test _seam_timeout(list_files(; service=g, config=_TINY_DEADLINE), FileCallError)
            @test _seam_timeout(retrieve_file("files/f1"; service=g, config=_TINY_DEADLINE), FileCallError)
            @test _seam_timeout(delete_file("files/f1"; service=g, config=_TINY_DEADLINE), FileCallError)
            @test _seam_timeout(poll_file("files/f1"; service=g, config=_TINY_DEADLINE), FileCallError)
            @test _seam_timeout(register_files(["gs://bucket/w.txt"]; config=_TINY_DEADLINE), FileCallError)
        end
        @test UniLM._files_url(g) == "https://generativelanguage.googleapis.com/v1beta/files"
        @test UniLM._files_upload_url(g) == "https://generativelanguage.googleapis.com/upload/v1beta/files"
        e = try file_content("files/f1"; service=g); nothing catch x; x end
        @test e isa ArgumentError && e.msg == "file_content: Gemini serves no download of an uploaded file " *
              "(\"you can't download user-uploaded files\", https://ai.google.dev/gemini-api/docs/files)"
        for s in (OPENAIServiceEndpoint, ANTHROPICServiceEndpoint, GEMINIOpenAIServiceEndpoint)
            e = try register_files(["gs://bucket/w.txt"]; service=s); nothing catch x; x end
            @test e isa ArgumentError && startswith(e.msg, "File registration is not supported by $(nameof(s))")
        end
        @test !has_capability(GEMINIOpenAIServiceEndpoint, :files)
    finally
        rm(path; force=true)
    end
end

# `FilesProbe{W}` (platform_seam_fixtures.jl) speaks Files wire `W`, served by `_with_scripted`.
const _AF = FilesProbe{UniLM._AnthropicFiles}

# Anthropic's upload: a form with the file and `expires_in_seconds` (3600 to 7776000), no purpose; its
# FileMetadata: `size_bytes`, an RFC 3339 `created_at`, no purpose or status.
# Refuted if: a purpose is sent or accepted, the expiry goes under another name or outside its range, or
# a field is read from the wrong key, to the wrong type or a shifted time.
@testset "Anthropic upload_file: the file and expires_in_seconds, and the FileMetadata it answers" begin
    path = tempname() * ".txt"
    write(path, "hello")
    try
        r, seen = _with_scripted(() -> upload_file(path; service=_AF, expires_after=3600), (_, _) -> _json(200, _ANTHROPIC_FILE))
        req = only(seen)
        @test (req.method, req.target) == ("POST", "/v1/files")
        @test occursin("name=\"expires_in_seconds\"\r\n\r\n3600\r\n", req.body)
        @test occursin("name=\"file\"; filename=\"$(basename(path))\"\r\nContent-Type: text/plain\r\n\r\nhello\r\n", req.body)
        @test !occursin("purpose", req.body) && !occursin("expires_after", req.body)
        f = r.response
        @test r isa FileSuccess
        @test (f.id, f.bytes, f.filename, f.purpose, f.status) == ("file_01YLNcm6NoeziqXvw8FX4zaY", 5, "w.txt", nothing, nothing)
        @test f.created_at == 1794859809   # 2026-11-16T20:10:09Z
        @test (f.raw["mime_type"], f.raw["downloadable"], f.raw["expires_at"]) ==
              ("text/plain", false, "2026-11-16T21:10:09.427661Z")
        _, seen = _with_scripted(() -> upload_file(FileUpload(service=_AF, file=path)), (_, _) -> _json(200, _ANTHROPIC_FILE))
        @test !occursin("expires_in_seconds", only(seen).body)
        # Refused before any request.
        e = try upload_file(path, "user_data"; service=_AF); nothing catch x; x end
        @test e isa ArgumentError && e.msg ==
              "upload_file: Anthropic's Files API takes no purpose (got \"user_data\"); call upload_file(path; service)"
        for s in (3599, 7776001)
            e = try upload_file(path; service=_AF, expires_after=s); nothing catch x; x end
            @test e isa ArgumentError &&
                  e.msg == "upload_file: Anthropic's Files API takes expires_after from 3600 to 7776000 seconds (got $s)"
        end
        @test FileUpload(service=_AF, file=path, expires_after=7776000).expires_after == 7776000
        # OpenAI's wire still requires a purpose.
        e = try upload_file(path; service=SeamProbe); nothing catch x; x end
        @test e isa ArgumentError && e.msg ==
              "upload_file: OpenAI's Files API requires a purpose: \"assistants\", \"batch\", \"fine-tune\", \"vision\" or \"user_data\""
    finally
        rm(path; force=true)
    end
end

# A FileMetadata without a field Anthropic's reference requires, or with one of another type or an
# `created_at` that is no RFC 3339 time, is no file.
# Refuted if: such a reply decodes as a FileSuccess.
@testset "Anthropic: a FileMetadata without id, size_bytes, created_at or filename is a FileCallError" begin
    full = JSON.parse(_ANTHROPIC_FILE)
    get1(body) = _answered(() -> retrieve_file("file_1"; service=_AF), 200; body)
    @test get1(JSON.json(full)) isa FileSuccess
    for key in ("id", "size_bytes", "created_at", "filename")
        r = get1(JSON.json(delete!(copy(full), key)))
        @test r isa FileCallError && occursin(key, r.error)
    end
    @test get1(JSON.json(merge(full, Dict("size_bytes" => "5")))) isa FileCallError
    r = get1(JSON.json(merge(full, Dict("created_at" => "yesterday"))))
    @test r isa FileCallError && occursin("RFC 3339", r.error)
end

# Anthropic's list: `limit` and the `page` cursor; `data` and `next_page`, null on the last page.
# Refuted if: `after` goes as anything but `page`, OpenAI's purpose or order are sent, a page without
# `data` decodes, or has_more does not follow `next_page`.
@testset "Anthropic list_files: limit and page, data and next_page" begin
    page(next) = """{"data":[$_ANTHROPIC_FILE],"next_page":$(next)}"""
    r, seen = _with_scripted(() -> list_files(; limit=2, after=_HOSTILE_ID, service=_AF), (_, _) -> _json(200, page("\"page_2\"")))
    @test only(seen).target == "/v1/files?limit=2&page=$_HOSTILE_ENC"
    @test r isa FileListSuccess && r.response.has_more && r.response.raw["next_page"] == "page_2"
    @test only(r.response.data).id == "file_01YLNcm6NoeziqXvw8FX4zaY"
    last = _answered(() -> list_files(; service=_AF), 200; body=page("null"))
    @test last isa FileListSuccess && !last.response.has_more
    @test _answered(() -> list_files(; service=_AF), 200; body="""{"next_page":null}""") isa FileCallError
    @test _answered(() -> list_files(; service=_AF), 200; body="""{"data":[{"id":"f"}],"next_page":null}""") isa FileCallError
    for (kw, name) in ((; purpose="batch") => "purpose", (; order="asc") => "order")
        e = try list_files(; service=_AF, kw...); nothing catch x; x end
        @test e isa ArgumentError && e.msg ==
              "list_files: Anthropic's Files API takes no $name, which only OpenAI's takes; unset $name"
    end
    e = try list_files(; service=_AF, purpose="batch", order="asc"); nothing catch x; x end
    @test e isa ArgumentError && occursin("takes no purpose or order", e.msg)
end

# Anthropic's retrieve_metadata, download and delete: GET /v1/files/{id}, GET …/content, DELETE; the
# DeletedFile names the id, and a `type` "file_deleted" when present.
# Refuted if: an id leaves its path segment, a download of a file Anthropic does not serve is anything
# but its HTTP 400, or a delete reply without the id, or with another type, reports a delete.
@testset "Anthropic retrieve_file, file_content and delete_file" begin
    t = _recorded_targets() do
        retrieve_file(_HOSTILE_ID; service=_AF)
        file_content(_HOSTILE_ID; service=_AF)
        delete_file(_HOSTILE_ID; service=_AF)
    end
    @test t == ["/v1/files/$_HOSTILE_ENC", "/v1/files/$_HOSTILE_ENC/content", "/v1/files/$_HOSTILE_ENC"]
    got = first(_with_scripted(() -> file_content("file_1"; service=_AF),
                               (_, _) -> HTTP.Response(200, ["Content-Type" => "application/octet-stream"], UInt8[0x00, 0xff])))
    @test got isa FileContentSuccess && got.content == UInt8[0x00, 0xff]
    # The reply Anthropic sent on 2026-11-16 for the bytes of an uploaded file.
    refused = """{"type":"error","error":{"type":"invalid_request_error","message":"File `file_1` is not downloadable. """ *
        """Only files generated by a tool (for example, the code execution tool) can be downloaded.",""" *
        """"details":{"error_code":"file_not_downloadable"}},"request_id":"req_011Ch7DNiTgCSxzi4zaGWbJk"}"""
    r = _answered(() -> file_content("file_1"; service=_AF), 400; body=refused, headers=["request-id" => "req_011Ch7DNiTgCSxzi4zaGWbJk"])
    @test r isa FileFailure && r.status == 400 && r.request_id == "req_011Ch7DNiTgCSxzi4zaGWbJk"
    @test occursin("file_not_downloadable", r.response)
    del(body) = _answered(() -> delete_file("file_1"; service=_AF), 200; body)
    @test del("""{"id":"file_01YLNcm6NoeziqXvw8FX4zaY","type":"file_deleted"}""") ==
          FileDeleteSuccess(id="file_01YLNcm6NoeziqXvw8FX4zaY", deleted=true, object="file_deleted")
    @test del("""{"id":"file_1"}""") == FileDeleteSuccess(id="file_1", deleted=true, object=nothing)
    r = del("""{"type":"file_deleted"}""")
    @test r isa FileCallError && occursin("\"id\"", r.error)
    @test del("""{"id":"file_1","type":"file"}""") isa FileCallError
end

# An Anthropic endpoint whose Files requests `_with_scripted` serves, with the headers that endpoint sends.
struct _AnthropicFilesAt <: UniLM.ServiceEndpoint
    endpoint::UniLM._AnthropicEndpoint
end
UniLM._files_wire(::_AnthropicFilesAt) = UniLM._AnthropicFiles()
UniLM._files_url(::_AnthropicFilesAt) = _url_probe_base[] * "/v1/files"
UniLM._files_headers(s::_AnthropicFilesAt) = UniLM._files_headers(s.endpoint)
UniLM._files_upload_headers(s::_AnthropicFilesAt) = UniLM._files_upload_headers(s.endpoint)
UniLM.provider_capabilities(::_AnthropicFilesAt) = Set([:files])

# https://platform.claude.com/docs/en/api/files/upload (Headers) names `anthropic-version` and
# `anthropic-workspace-id`, and no `anthropic-beta` or user-profile header; the SDK (v2.9.0) sends no
# beta for files.
# Refuted if: an Anthropic Files request sends the endpoint's betas or its user profile, or leaves out
# its key, its API version or its workspace — from the type or from an instance.
@testset "an Anthropic Files request sends the key, the version and the workspace, never the user profile or the betas" begin
    path = tempname() * ".txt"
    write(path, "hello")
    try
        withenv("ANTHROPIC_API_KEY" => "sk-ant-local-0000") do
            base = ["Authorization" => "Bearer sk-ant-local-0000", "anthropic-version" => "2026-10-21"]
            every = ANTHROPICServiceEndpoint(betas=["fast-mode-2026-02-01"], user_profile_id="uprof_011CZkZCu8hGbp5mYRQgUmz9",
                                             workspace_id="wrkspc_011CZkZaBF1tNoB5wlCeusgy")
            for (service, want) in ((ANTHROPICServiceEndpoint, base), (ANTHROPICServiceEndpoint(), base),
                                    (every, [base; "anthropic-workspace-id" => "wrkspc_011CZkZaBF1tNoB5wlCeusgy"]))
                @test (@inferred UniLM._files_headers(service)) == want
                @test UniLM._files_upload_headers(service) == want
                # On the wire, beside the headers HTTP.jl adds: exactly these authorization and anthropic-
                # headers, and a Content-Type only on the upload, the form's own.
                sent = Dict{String,String}[]
                _with_scripted((_, req) -> (push!(sent, Dict(lowercase(String(k)) => String(v) for (k, v) in req.headers));
                                            _json(200, _ANTHROPIC_FILE))) do
                    upload_file(path; service=_AnthropicFilesAt(service))
                    retrieve_file("file_1"; service=_AnthropicFilesAt(service))
                end
                for (i, h) in enumerate(sent)
                    @test Dict(k => v for (k, v) in h if k == "authorization" || startswith(k, "anthropic-")) ==
                          Dict(lowercase(k) => v for (k, v) in want)
                    @test i == 1 ? startswith(h["content-type"], "multipart/form-data; boundary=") : !haskey(h, "content-type")
                end
            end
        end
    finally
        rm(path; force=true)
    end
end

# `GeminiFilesProbe` (platform_seam_fixtures.jl): a Gemini endpoint whose Files requests `_with_scripted` serves.
const _GF = GeminiFilesProbe

# Gemini's resumable upload (https://ai.google.dev/gemini-api/docs/files): a `start` POST with the key,
# the size, the MIME type and the display name; the bytes, `upload, finalize`, at the address the start
# answers in X-Goog-Upload-URL, without the key; `{"file": File}` back.
# Refuted if: a header or the display name is missing or wrong, the key goes with the bytes, the bytes
# go anywhere but the session's address, a refused start still sends them, or the File is misread.
@testset "Gemini upload_file: a resumable start, then the bytes at the session's address" begin
    path = tempname() * ".txt"
    write(path, "hello")
    session() = _url_probe_base[] * "/upload/v1beta/files?upload_id=u1&upload_protocol=resumable"
    started() = HTTP.Response(200, ["X-Goog-Upload-URL" => session(), "X-Goog-Upload-Status" => "active"], UInt8[])
    try
        hdrs = Dict{String,String}[]
        r, seen = _with_scripted((n, req) -> (push!(hdrs, Dict(lowercase(String(k)) => String(v) for (k, v) in req.headers));
                                             n == 1 ? started() : _json(200, """{"file":$_GEMINI_FILE}"""))) do
            upload_file(path; service=_GF)
        end
        @test [(q.method, q.target) for q in seen] ==
              [("POST", "/upload/v1beta/files"), ("POST", "/upload/v1beta/files?upload_id=u1&upload_protocol=resumable")]
        @test JSON.parse(seen[1].body) == Dict("file" => Dict("displayName" => basename(path)))
        start, bytes = hdrs
        @test [start[k] for k in ("x-goog-api-key", "content-type", "x-goog-upload-protocol", "x-goog-upload-command",
                                  "x-goog-upload-header-content-length", "x-goog-upload-header-content-type")] ==
              ["t", "application/json", "resumable", "start", "5", "text/plain"]
        @test (bytes["x-goog-upload-offset"], bytes["x-goog-upload-command"]) == ("0", "upload, finalize")
        @test !haskey(bytes, "x-goog-api-key")
        @test seen[2].body == "hello"
        @test r isa FileSuccess
        f = r.response
        @test (f.id, f.bytes, f.filename, f.purpose, f.status) == ("rvbxh1rsu85u", 5, "w.txt", nothing, "ACTIVE")
        @test f.created_at == 1794860041   # 2026-11-16T20:14:01Z
        @test (f.raw["uri"], f.raw["mimeType"], f.raw["expirationTime"]) ==
              ("https://generativelanguage.googleapis.com/v1beta/files/rvbxh1rsu85u", "text/plain",
               "2026-11-18T20:14:00.555605689Z")
        # A refused start is the upload's failure, and sends no bytes.
        r, seen = _with_scripted(() -> upload_file(path; service=_GF), (_, _) -> _json(429, """{"error":{"code":429}}"""))
        @test r isa FileFailure && r.status == 429 && length(seen) == 1
        # A start that names no session is a call error, and sends no bytes.
        r, seen = _with_scripted(() -> upload_file(path; service=_GF), (_, _) -> _json(200, "{}"))
        @test r isa FileCallError && occursin("X-Goog-Upload-URL", r.error) && length(seen) == 1
        # A refused finalize is the upload's failure; one that answers no file, a call error.
        r, _ = _with_scripted(() -> upload_file(path; service=_GF), (n, _) -> n == 1 ? started() : _json(400, "{}"))
        @test r isa FileFailure && r.status == 400
        r, _ = _with_scripted(() -> upload_file(path; service=_GF), (n, _) -> n == 1 ? started() : _json(200, _GEMINI_FILE))
        @test r isa FileCallError && occursin("\"file\"", r.error)
        # Refused before any request.
        e = try upload_file(path, "user_data"; service=_GF); nothing catch x; x end
        @test e isa ArgumentError &&
              e.msg == "upload_file: Gemini's Files API takes no purpose (got \"user_data\"); call upload_file(path; service)"
        e = try upload_file(path; service=_GF, expires_after=3600); nothing catch x; x end
        @test e isa ArgumentError && e.msg == "upload_file: Gemini stores an uploaded file for 48 hours and takes no " *
              "expiry (https://ai.google.dev/gemini-api/docs/files); unset expires_after"
    finally
        rm(path; force=true)
    end
end

# Gemini's File: `name` "files/<id>", `sizeBytes` an int64 string, `createTime` RFC 3339, `state`;
# `displayName`, `sizeBytes` and `state` may be absent.
# Refuted if: a File without its name or creation time, or with a name that is not "files/<id>" or a
# size that is no integer, decodes; or an absent optional field is made up.
@testset "Gemini: a File without name or createTime is a FileCallError; absent optional fields are nothing" begin
    full = JSON.parse(_GEMINI_FILE)
    get1(d) = _answered(() -> retrieve_file("files/rvbxh1rsu85u"; service=_GF), 200; body=JSON.json(d))
    for key in ("name", "createTime")
        r = get1(delete!(copy(full), key))
        @test r isa FileCallError && occursin(key, r.error)
    end
    for bad in (Dict("name" => "rvbxh1rsu85u"), Dict("name" => "files/"), Dict("sizeBytes" => "five"), Dict("sizeBytes" => 5))
        @test get1(merge(full, bad)) isa FileCallError
    end
    r = get1(delete!(delete!(delete!(copy(full), "displayName"), "sizeBytes"), "state"))
    @test r isa FileSuccess && (r.response.filename, r.response.bytes, r.response.status) == (nothing, nothing, nothing)
end

# Gemini's files.list: `pageSize` and `pageToken`; `files` and `nextPageToken`, each left out when empty.
# Refuted if: `limit` or `after` go under OpenAI's names, OpenAI's purpose or order are sent, `{}` is not
# an empty last page, or has_more does not follow `nextPageToken`.
@testset "Gemini list_files: pageSize and pageToken, files and nextPageToken" begin
    r, seen = _with_scripted(() -> list_files(; limit=1, after=_HOSTILE_ID, service=_GF),
                             (_, _) -> _json(200, """{"files":[$_GEMINI_FILE],"nextPageToken":"tok_2"}"""))
    @test only(seen).target == "/v1beta/files?pageSize=1&pageToken=$_HOSTILE_ENC"
    @test r isa FileListSuccess && r.response.has_more && only(r.response.data).id == "rvbxh1rsu85u"
    empty = _answered(() -> list_files(; service=_GF), 200; body="{}")
    @test empty isa FileListSuccess && isempty(empty.response.data) && !empty.response.has_more
    @test _answered(() -> list_files(; service=_GF), 200; body="""{"files":{}}""") isa FileCallError
    e = try list_files(; service=_GF, order="desc"); nothing catch x; x end
    @test e isa ArgumentError && e.msg == "list_files: Gemini's Files API takes no order, which only OpenAI's takes; unset order"
end

# Gemini's files.get and files.delete at v1beta/files/<id>; the delete answers `{}` (2026-11-16).
# Refuted if: either form of the id ("files/<id>" or "<id>") leaves one path segment, or the delete
# reports another id than the one requested.
@testset "Gemini retrieve_file and delete_file take the id or the name" begin
    t = _recorded_targets() do
        retrieve_file("files/rvbxh1rsu85u"; service=_GF)
        retrieve_file("rvbxh1rsu85u"; service=_GF)
        delete_file("files/rvbxh1rsu85u"; service=_GF)
        retrieve_file(_HOSTILE_ID; service=_GF)
    end
    @test t == ["/v1beta/files/rvbxh1rsu85u", "/v1beta/files/rvbxh1rsu85u", "/v1beta/files/rvbxh1rsu85u",
                "/v1beta/files/$_HOSTILE_ENC"]
    for id in ("files/rvbxh1rsu85u", "rvbxh1rsu85u")
        @test _answered(() -> delete_file(id; service=_GF), 200; body="{}") ==
              FileDeleteSuccess(id="rvbxh1rsu85u", deleted=true, object=nothing)
    end
    # Gemini answered a GET after the delete with HTTP 403 (2026-11-16).
    @test _answered(() -> retrieve_file("rvbxh1rsu85u"; service=_GF), 403).status == 403
end

# poll_file: GET until the file is no longer processing — Gemini's state past PROCESSING, OpenAI's status
# past "uploaded"; an Anthropic file at once.
# Refuted if: the poll ends on a processing file or polls past a terminal one, a non-transient failure is
# polled through, the time running out loses the last file seen, or a cancelled token sends a request.
@testset "poll_file waits out Gemini's PROCESSING and OpenAI's uploaded" begin
    processing = replace(_GEMINI_FILE, "\"ACTIVE\"" => "\"PROCESSING\"")
    r, seen = _with_scripted(() -> poll_file("files/rvbxh1rsu85u"; service=_GF, interval=0.01, timeout=30.0),
                             (n, _) -> _json(200, n < 3 ? processing : _GEMINI_FILE))
    @test r isa FileSuccess && r.response.status == "ACTIVE" && length(seen) == 3
    @test all(q -> q.target == "/v1beta/files/rvbxh1rsu85u", seen)
    failed = replace(_GEMINI_FILE, "\"ACTIVE\"" => "\"FAILED\"")
    r, seen = _with_scripted(() -> poll_file("rvbxh1rsu85u"; service=_GF, interval=0.01, timeout=30.0), (_, _) -> _json(200, failed))
    @test r isa FileSuccess && r.response.status == "FAILED" && length(seen) == 1
    r, _ = _with_scripted(() -> poll_file("rvbxh1rsu85u"; service=_GF, interval=0.05, timeout=0.5), (_, _) -> _json(200, processing))
    @test r isa FileCallError && r.cause isa UniLMTimeout && r.last_observed.status == "PROCESSING"
    @test startswith(r.error, "poll_file timeout: rvbxh1rsu85u") && endswith(r.error, "(last observed status: PROCESSING)")
    file(status) = JSON.json(Dict("id" => "file-1", "object" => "file", "bytes" => 3, "created_at" => 1,
                                  "filename" => "a.jsonl", "purpose" => "fine-tune", "status" => status))
    r, seen = _with_scripted(() -> poll_file("file-1"; service=URLProbe, interval=0.01, timeout=30.0),
                             (n, _) -> _json(200, file(n == 1 ? "uploaded" : "processed")))
    @test r isa FileSuccess && r.response.status == "processed" && length(seen) == 2
    r, seen = _with_scripted(() -> poll_file("file_1"; service=_AF, interval=0.01, timeout=30.0), (_, _) -> _json(200, _ANTHROPIC_FILE))
    @test r isa FileSuccess && length(seen) == 1
    # A transient status is polled through; another failure is the result.
    r, seen = _with_scripted(() -> poll_file("f"; service=_GF, interval=0.01, timeout=30.0),
                             (n, _) -> n == 1 ? _json(503, "{}") : _json(200, _GEMINI_FILE))
    @test r isa FileSuccess && length(seen) == 2
    @test _answered(() -> poll_file("f"; service=_GF, interval=0.01, timeout=30.0), 403) isa FileFailure
    r, seen = _with_scripted(() -> poll_file("f"; service=_GF, cancel=cancel!(CancelToken())), (_, _) -> _json(200, processing))
    @test r isa FileCallError && r.cause isa UniLMCancelled && isempty(seen)
    @test_throws ArgumentError poll_file("f"; service=_GF, interval=0)
    # The three-field positional form still builds a call error.
    @test FileCallError("e", nothing, nothing).last_observed === nothing
end

# Gemini's files.register: POST v1beta/files:register `{"uris": [...]}`, answered `{"files": [File…]}`.
# Refuted if: the URIs are sent in another shape or place, an empty list is sent, a 200 without `files`
# decodes, or Gemini's refusal is anything but a FileFailure.
@testset "register_files: Gemini's files.register, one File per gs:// URI" begin
    r, seen = _with_scripted(() -> register_files(["gs://bucket/w.txt"]; service=_GF),
                             (_, _) -> _json(200, """{"files":[$_GEMINI_FILE]}"""))
    @test (only(seen).method, only(seen).target) == ("POST", "/v1beta/files:register")
    @test JSON.parse(only(seen).body) == Dict("uris" => ["gs://bucket/w.txt"])
    @test r isa FileListSuccess && only(r.response.data).id == "rvbxh1rsu85u" && !r.response.has_more
    # Gemini's reply on 2026-11-16 for an object the key cannot reach.
    r = _answered(() -> register_files(["gs://bucket/w.txt"]; service=_GF), 404;
                  body="""{"error":{"code":404,"message":"Requested entity was not found.","status":"NOT_FOUND"}}""")
    @test r isa FileFailure && r.status == 404
    @test _answered(() -> register_files(["gs://bucket/w.txt"]; service=_GF), 200; body="{}") isa FileCallError
    e = try register_files(String[]; service=_GF); nothing catch x; x end
    @test e isa ArgumentError && e.msg == "register_files: uris is empty; Gemini registers one or more gs:// objects"
end

@testset "every method of the Files verbs takes `cancel` as its last keyword" begin
    for f in (upload_file, list_files, retrieve_file, delete_file, file_content, poll_file, register_files), m in methods(f)
        @test (f, last(filter(k -> !endswith(String(k), "..."), Base.kwarg_decl(m)))) == (f, :cancel)
    end
end

# Refuted if: an endpoint on Gemini's Files wire without an upload address of its own fails other than by
# naming that, or sends anything.
struct _GeminiWireNoUpload <: UniLM.ServiceEndpoint end
UniLM._files_wire(::_GeminiWireNoUpload) = UniLM._GeminiFiles()
UniLM.auth_header(::_GeminiWireNoUpload) = ["x-goog-api-key" => "t", "Content-Type" => "application/json"]
UniLM.provider_capabilities(::_GeminiWireNoUpload) = Set([:files])
@testset "an endpoint on Gemini's Files wire without an upload address: a FileCallError naming it" begin
    path = tempname() * ".txt"
    write(path, "hello")
    try
        r = upload_file(path; service=_GeminiWireNoUpload())
        @test r isa FileCallError && r.cause isa ArgumentError &&
              r.cause.msg == "upload_file: _GeminiWireNoUpload speaks Gemini's Files wire and names no upload address"
    finally
        rm(path; force=true)
    end
end
