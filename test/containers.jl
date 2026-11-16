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
    for v in (0, -1, "0", "beta", "", true)
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
