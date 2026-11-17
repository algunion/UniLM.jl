@testset "Containers API — config seam wiring" begin
    cpath = tempname() * ".txt"
    write(cpath, "probe")
    try
        @test _seam_timeout(create_container(name="c", service=SeamProbe, config=_TINY_DEADLINE), UniLM.ContainerCallError)
        @test _seam_timeout(retrieve_container("cntr_x"; service=SeamProbe, config=_TINY_DEADLINE), UniLM.ContainerCallError)
        @test _seam_timeout(list_containers(service=SeamProbe, config=_TINY_DEADLINE), UniLM.ContainerCallError)
        @test _seam_timeout(delete_container("cntr_x"; service=SeamProbe, config=_TINY_DEADLINE), UniLM.ContainerCallError)
        @test _seam_timeout(add_container_file("cntr_x", cpath; service=SeamProbe, config=_TINY_DEADLINE), UniLM.ContainerCallError)
    finally
        rm(cpath; force=true)
    end
end

@testset "add_container_file refuses a path that is no file before anything is sent, as upload_file does" begin
    err, seen = _with_scripted((_, _) -> _json(200, "{}")) do
        try add_container_file("cntr_1", "/no/such/file.csv"; service=URLProbe); nothing catch e; e end
    end
    @test err isa ArgumentError && err.msg == "file not found: /no/such/file.csv" && isempty(seen)
    @test_throws ArgumentError upload_file("/no/such/file.csv", "user_data"; service=URLProbe)
end

@testset "add_container_file: a cancelled token reads no file" begin
    # As for upload_file: a call that read this unreadable file would end in that SystemError.
    cpath = tempname() * ".txt"
    write(cpath, "x")
    chmod(cpath, 0o000)
    try
        r, seen = _with_scripted(() -> add_container_file("cntr_1", cpath; service=URLProbe,
                                                          cancel=cancel!(CancelToken())),
                                 (_, _) -> _json(200, "{}"))
        @test (typeof(r), typeof(r.cause), length(seen)) == (ContainerCallError, UniLMCancelled, 0)
    finally
        chmod(cpath, 0o600)
        rm(cpath; force=true)
    end
end

@testset "add_container_file decodes the container-file object it gets back" begin
    path, io = mktemp()
    write(io, "x")
    close(io)
    try
        body = """{"id": "cfile_1", "object": "container.file", "created_at": 1747848842, "bytes": 880,
                   "container_id": "cntr_1", "path": "/mnt/data/a.txt", "source": "user"}"""
        r = _answered(() -> add_container_file("cntr_1", path; service=URLProbe), 200; body)
        @test r isa ContainerSuccess
        f = r.response
        @test f isa UniLM.ContainerFileObject
        @test (f.id, f.container_id, f.path, f.bytes, f.source, f.created_at) ==
              ("cfile_1", "cntr_1", "/mnt/data/a.txt", 880, "user", 1747848842)
    finally
        rm(path; force=true)
    end
end

@testset "Containers API — a failure keeps the request id the service sent" begin
    r = _answered(() -> retrieve_container("cntr_x"; service=URLProbe), 404; headers=["x-request-id" => "req_cntr"])
    @test r isa UniLM.ContainerFailure && r.request_id == "req_cntr"
end

@testset "Containers API — a separator-bearing id stays one path segment" begin
    cpath = tempname() * ".txt"
    write(cpath, "x")
    try
        t = _recorded_targets() do
            retrieve_container(_HOSTILE_ID; service=URLProbe)
            delete_container(_HOSTILE_ID; service=URLProbe)
            add_container_file(_HOSTILE_ID, cpath; service=URLProbe)
            list_containers(; limit=2, after=_HOSTILE_ID, service=URLProbe)
        end
        @test t == ["/v1/containers/$_HOSTILE_ENC",
                    "/v1/containers/$_HOSTILE_ENC",
                    "/v1/containers/$_HOSTILE_ENC/files",
                    "/v1/containers?limit=2&after=$_HOSTILE_ENC"]
        g = _recorded_targets() do
            add_container_file("cntr_abc123", cpath; service=URLProbe)
            list_containers(; after="cntr_abc123", service=URLProbe)
        end
        @test g == ["/v1/containers/cntr_abc123/files", "/v1/containers?after=cntr_abc123"]
    finally
        rm(cpath; force=true)
    end
end

@testset "delete_container reports success only when the service confirms the delete" begin
    del(body) = _answered(() -> delete_container("cntr_1"; service=URLProbe), 200; body)
    @test del("""{"id": "cntr_1", "object": "container.deleted", "deleted": true}""") ==
          UniLM.ContainerDeleteSuccess(id="cntr_1", deleted=true)
    @test del("""{"id": "cntr_1", "object": "container.deleted"}""") isa UniLM.ContainerCallError
    @test del("""{"id": "cntr_1", "object": "container.deleted", "deleted": false}""") isa UniLM.ContainerCallError
end

@testset "containers: a reply without what OpenAI's reference requires is a call error, never filled in" begin
    # A list without data or has_more is not an empty last page.
    lst(body) = _answered(() -> list_containers(; service=URLProbe), 200; body)
    @test lst("{}") isa ContainerCallError
    @test lst("""{"object": "list", "data": []}""") isa ContainerCallError
    @test lst("""{"object": "list", "has_more": false}""") isa ContainerCallError
    ok = lst("""{"object": "list", "data": [{"id": "cntr_1", "status": "running", "name": "a"}], "has_more": false}""")
    @test ok isa ContainerListSuccess && [c["id"] for c in ok.response.data] == ["cntr_1"] && !ok.response.has_more
    # A container without its status.
    @test _answered(() -> retrieve_container("cntr_1"; service=URLProbe), 200;
                    body="""{"id": "cntr_1", "name": "a"}""") isa ContainerCallError
    @test _answered(() -> create_container(; name="a", service=URLProbe), 200;
                    body="""{"id": "cntr_1", "name": "a"}""") isa ContainerCallError
    # A deletion record without its id, or naming another kind of object.
    del(body) = _answered(() -> delete_container("cntr_1"; service=URLProbe), 200; body)
    @test del("""{"object": "container.deleted", "deleted": true}""") isa ContainerCallError
    @test del("""{"id": "cntr_1", "deleted": true}""") isa ContainerCallError
    @test del("""{"id": "cntr_1", "object": "container.file.deleted", "deleted": true}""") isa ContainerCallError
    @test del("""{"id": "cntr_2", "object": "container.deleted", "deleted": true}""").id == "cntr_2"
end

@testset "create_container: memory_limit, network_policy and skills" begin
    ok = """{"id": "cntr_1", "object": "container", "status": "running", "name": "a"}"""
    sent(f) = JSON.parse(only(last(_with_scripted(f, (_, _) -> _json(200, ok)))).body)
    policy = ContainerNetworkAllowlist(["pypi.org", "api.example.com"];
                                       domain_secrets=[ContainerDomainSecret("api.example.com", "API_TOKEN", "s3cr3t")])
    skills = [SkillReference("skill_abc"), SkillReference("skill_def"; version=2), SkillReference("skill_ghi"; version="latest")]
    @test sent(() -> create_container(; name="a", memory_limit="4g", network_policy=policy, skills, service=URLProbe)) ==
          Dict("name" => "a", "memory_limit" => "4g",
               "network_policy" => Dict("type" => "allowlist", "allowed_domains" => ["pypi.org", "api.example.com"],
                                        "domain_secrets" => [Dict("domain" => "api.example.com", "name" => "API_TOKEN",
                                                                  "value" => "s3cr3t")]),
               "skills" => [Dict("type" => "skill_reference", "skill_id" => "skill_abc"),
                            Dict("type" => "skill_reference", "skill_id" => "skill_def", "version" => "2"),
                            Dict("type" => "skill_reference", "skill_id" => "skill_ghi", "version" => "latest")])
    @test sent(() -> create_container(; name="a", network_policy=ContainerNetworkDisabled(), service=URLProbe)) ==
          Dict("name" => "a", "network_policy" => Dict("type" => "disabled"))
    @test sent(() -> create_container(; name="a", network_policy=ContainerNetworkAllowlist(["pypi.org"]),
                                      service=URLProbe))["network_policy"] ==
          Dict("type" => "allowlist", "allowed_domains" => ["pypi.org"])
    # Values outside OpenAI's reference are refused before anything is sent.
    err, seen = _with_scripted((_, _) -> _json(200, ok)) do
        try create_container(; name="a", memory_limit="2g", service=URLProbe); nothing catch e; e end
    end
    @test err isa ArgumentError && occursin("create_container", err.msg) && occursin("\"4g\"", err.msg) && isempty(seen)
    @test_throws ArgumentError ContainerNetworkAllowlist(String[])
    @test_throws ArgumentError ContainerDomainSecret("api.example.com", "", "v")
    @test_throws ArgumentError ContainerDomainSecret("", "API_TOKEN", "v")
    @test_throws ArgumentError ContainerDomainSecret("api.example.com", "API_TOKEN", "")
    @test_throws ArgumentError SkillReference("")
    @test_throws ArgumentError SkillReference("s"^65)
    @test SkillReference("s"^64).skill_id == "s"^64
    for v in (0, -1, "0", "beta", "", true, "1\n")
        @test_throws ArgumentError SkillReference("skill_abc"; version=v)
    end
    # A secret's value is never printed.
    @test !occursin("s3cr3t", repr(policy)) && !occursin("s3cr3t", repr(MIME"text/plain"(), policy))
    @test occursin("API_TOKEN", repr(policy))
end

@testset "list_containers takes order and a name filter" begin
    t = _recorded_targets() do
        list_containers(; limit=3, order="asc", after="cntr_a", name="price check", service=URLProbe)
        list_containers(; name="sandbox", service=URLProbe)
    end
    @test t == ["/v1/containers?limit=3&order=asc&after=cntr_a&name=price%20check", "/v1/containers?name=sandbox"]
    err, seen = _with_scripted((_, _) -> _json(200, "{}")) do
        try list_containers(; order="up", service=URLProbe); nothing catch e; e end
    end
    @test err isa ArgumentError && occursin("list_containers", err.msg) && isempty(seen)
end

# A container file as OpenAI's reference shows one.
const _CONTAINER_FILE_JSON = """{"id": "cfile_1", "object": "container.file", "created_at": 1747848842, "bytes": 880,
    "container_id": "cntr_1", "path": "/mnt/data/a.txt", "source": "user"}"""

@testset "container files: list, retrieve, delete, and download" begin
    page = """{"object": "list", "data": [$_CONTAINER_FILE_JSON], "first_id": "cfile_1", "last_id": "cfile_1", "has_more": false}"""
    l, seen = _with_scripted((_, _) -> _json(200, page)) do
        list_container_files("cntr_1"; limit=10, order="asc", after="cfile_0", service=URLProbe)
    end
    @test l isa ContainerFileListSuccess && l.response isa ContainerFileList && !l.response.has_more
    f = only(l.response.data)
    @test f isa UniLM.ContainerFileObject && (f.id, f.container_id, f.path, f.bytes) == ("cfile_1", "cntr_1", "/mnt/data/a.txt", 880)
    @test (only(seen).method, only(seen).target) == ("GET", "/v1/containers/cntr_1/files?limit=10&order=asc&after=cfile_0")
    @test _answered(() -> list_container_files("cntr_1"; service=URLProbe), 200; body="{}") isa ContainerCallError
    @test_throws ArgumentError list_container_files("cntr_1"; order="up", service=URLProbe)

    g = _answered(() -> retrieve_container_file("cntr_1", "cfile_1"; service=URLProbe), 200; body=_CONTAINER_FILE_JSON)
    @test g isa ContainerSuccess && g.response isa UniLM.ContainerFileObject && g.response.id == "cfile_1"

    del(body) = _answered(() -> delete_container_file("cntr_1", "cfile_1"; service=URLProbe), 200; body)
    @test del("""{"id": "cfile_1", "object": "container.file.deleted", "deleted": true}""") ==
          ContainerDeleteSuccess(id="cfile_1", deleted=true)
    @test del("""{"id": "cfile_1", "object": "container.deleted", "deleted": true}""") isa ContainerCallError
    @test del("""{"object": "container.file.deleted", "deleted": true}""") isa ContainerCallError
    @test del("""{"id": "cfile_1", "object": "container.file.deleted", "deleted": false}""") isa ContainerCallError

    bytes = UInt8[0x89, 0x50, 0x4e, 0x47, 0x00, 0xff]
    c, seen = _with_scripted((_, _) -> HTTP.Response(200, ["Content-Type" => "application/octet-stream"], bytes)) do
        container_file_content("cntr_1", "cfile_1"; service=URLProbe)
    end
    @test c isa FileContentSuccess && c.content == bytes
    @test (only(seen).method, only(seen).target) == ("GET", "/v1/containers/cntr_1/files/cfile_1/content")
    # Saved as a Files download is.
    path = tempname()
    try
        @test save_file_content(c, path) == path && read(path) == bytes
    finally
        rm(path; force=true)
    end
    @test _answered(() -> container_file_content("cntr_1", "cfile_x"; service=URLProbe), 404) isa ContainerFailure
end

@testset "container files: methods, ids as single path segments, the config seam, the capability" begin
    E = _HOSTILE_ENC
    sent = [(r.method, r.target) for r in last(_with_scripted((_, _) -> _json(200, "{}")) do
        list_container_files(_HOSTILE_ID; service=URLProbe)
        retrieve_container_file(_HOSTILE_ID, _HOSTILE_ID; service=URLProbe)
        delete_container_file(_HOSTILE_ID, _HOSTILE_ID; service=URLProbe)
        container_file_content(_HOSTILE_ID, _HOSTILE_ID; service=URLProbe)
    end)]
    @test sent == [("GET", "/v1/containers/$E/files"), ("GET", "/v1/containers/$E/files/$E"),
                   ("DELETE", "/v1/containers/$E/files/$E"), ("GET", "/v1/containers/$E/files/$E/content")]
    for call in (cfg -> list_container_files("cntr_x"; service=SeamProbe, config=cfg),
                 cfg -> retrieve_container_file("cntr_x", "f"; service=SeamProbe, config=cfg),
                 cfg -> delete_container_file("cntr_x", "f"; service=SeamProbe, config=cfg),
                 cfg -> container_file_content("cntr_x", "f"; service=SeamProbe, config=cfg))
        @test _seam_timeout(call(_TINY_DEADLINE), ContainerCallError)
    end
    ds = DeepSeekEndpoint("k")
    @test_throws ArgumentError list_container_files("c"; service=ds)
    @test_throws ArgumentError retrieve_container_file("c", "f"; service=ds)
    @test_throws ArgumentError delete_container_file("c", "f"; service=ds)
    @test_throws ArgumentError container_file_content("c", "f"; service=ds)
end

@testset "add_container_file(id; file_id): an uploaded file copied in, sent as JSON" begin
    cfile = """{"id": "cfile_1", "object": "container.file", "created_at": 1, "bytes": 3, "container_id": "cntr_1",
                "path": "/mnt/data/a.txt", "source": "user"}"""
    r, seen = _with_scripted((_, _) -> _json(200, cfile)) do
        add_container_file("cntr_1"; file_id="file-abc", service=URLProbe)
    end
    @test r isa ContainerSuccess && r.response isa UniLM.ContainerFileObject && r.response.path == "/mnt/data/a.txt"
    req = only(seen)
    @test (req.method, req.target, JSON.parse(req.body)) == ("POST", "/v1/containers/cntr_1/files", Dict("file_id" => "file-abc"))
    @test _seam_timeout(add_container_file("cntr_x"; file_id="f", service=SeamProbe, config=_TINY_DEADLINE), ContainerCallError)
    @test _recorded_targets(() -> add_container_file(_HOSTILE_ID; file_id="f", service=URLProbe)) ==
          ["/v1/containers/$_HOSTILE_ENC/files"]
    @test_throws UndefKeywordError add_container_file("cntr_1"; service=URLProbe)
    @test_throws ArgumentError add_container_file("cntr_1"; file_id="f", service=DeepSeekEndpoint("k"))
end

@testset "the container verbs take cancel as their last keyword, and a cancelled token sends nothing" begin
    for f in (create_container, list_containers, add_container_file, list_container_files, retrieve_container_file,
              delete_container_file, container_file_content),
        m in methods(f)
        @test (f, last(filter(k -> !endswith(String(k), "..."), Base.kwarg_decl(m)))) == (f, :cancel)
    end
    tok = cancel!(CancelToken())
    for call in (() -> add_container_file("cntr_1"; file_id="f", service=URLProbe, cancel=tok),
                 () -> list_container_files("cntr_1"; service=URLProbe, cancel=tok),
                 () -> retrieve_container_file("cntr_1", "f"; service=URLProbe, cancel=tok),
                 () -> delete_container_file("cntr_1", "f"; service=URLProbe, cancel=tok),
                 () -> container_file_content("cntr_1", "f"; service=URLProbe, cancel=tok))
        r, seen = _with_scripted(call, (_, _) -> _json(200, "{}"))
        @test r isa ContainerCallError && r.cause isa UniLMCancelled && isempty(seen)
    end
end

# OpenAI's CreateContainerBody takes `skills` by reference or inline: an InlineSkillParam is a name, a description and
# its zip as base64 (`source`: type base64, media_type application/zip); across a list's inline skills "decoded ZIP data
# must total at most 50 MiB and Base64 text must total at most 70,254,592 characters" (OpenAI's API specification).
# Refuted if: an inline skill goes in another shape or with other bytes, a reference beside it changes, what is no zip
# is taken, or more than 50 MiB of zip reaches the endpoint.
@testset "create_container: inline skills beside references, their zip base64-encoded; their limits" begin
    ok = """{"id": "cntr_1", "object": "container", "status": "running", "name": "a"}"""
    zip = UInt8[0x50, 0x4b, 0x03, 0x04, 0x14, 0x00, 0xff, 0x0a]
    path = joinpath(mktempdir(), "csv-sum.zip")
    write(path, zip)
    inline = InlineSkill(path; name="csv-sum", description="Sums a CSV column")
    @test inline isa ContainerSkill && SkillReference("skill_abc") isa ContainerSkill
    @test (inline.name, inline.description, inline.zip) == ("csv-sum", "Sums a CSV column", zip)
    r, seen = _with_scripted(() -> create_container(; name="a", skills=[SkillReference("skill_abc"; version=2), inline],
                                                     service=URLProbe), (_, _) -> _json(200, ok))
    @test r isa ContainerSuccess
    @test JSON.parse(only(seen).body)["skills"] ==
          [Dict("type" => "skill_reference", "skill_id" => "skill_abc", "version" => "2"),
           Dict("type" => "inline", "name" => "csv-sum", "description" => "Sums a CSV column",
                "source" => Dict("type" => "base64", "media_type" => "application/zip", "data" => UniLM.Base64.base64encode(zip)))]
    @test JSON.lower(InlineSkill(zip; name="n", description="d"))[:source][:data] == "UEsDBBQA/wo="
    # Printed without its bytes.
    @test repr(inline) == "InlineSkill(\"csv-sum\", 8 bytes)"
    txt = joinpath(mktempdir(), "skill.txt")
    write(txt, zip)
    for (make, msg) in ((() -> InlineSkill(UInt8[0x50, 0x4b, 0x05, 0x06]; name="n", description="d"), "the bytes are no zip archive"),
                        (() -> InlineSkill(UInt8[]; name="n", description="d"), "the bytes are no zip archive"),
                        (() -> InlineSkill(txt; name="n", description="d"), "$txt is no .zip file"),
                        (() -> InlineSkill(dirname(path); name="n", description="d"), "is no .zip file"))
        e = try make(); nothing catch x; x end
        @test e isa ArgumentError && occursin(msg, e.msg)
    end
    # 50 MiB of zip across the inline skills passes the check (a 70 MB body is more than the scripted endpoint
    # reads), a byte more is refused before anything is sent.
    half = vcat(zip, zeros(UInt8, 25 * 2^20 - length(zip)))
    big = [InlineSkill(half; name="a", description="d"), InlineSkill(half; name="b", description="d")]
    @test isnothing(UniLM._check_inline_skills([SkillReference("skill_abc"); big]))
    push!(big, InlineSkill(zip[1:4]; name="c", description="d"))
    e, seen = _with_scripted((_, _) -> _json(200, ok)) do
        try create_container(; name="a", skills=[SkillReference("skill_abc"); big], service=URLProbe); nothing catch x; x end
    end
    @test isempty(seen) && e isa ArgumentError && e.msg ==
          "create_container: OpenAI takes at most 50 MiB (52428800 bytes) of zip data across a container's inline skills " *
          "(got 52428804 bytes in 3); upload a large skill with create_skill and pass a SkillReference"
end
