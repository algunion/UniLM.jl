@testset "Vector Stores API — config seam wiring" begin
    @test _seam_timeout(create_vector_store(service=SeamProbe, config=_TINY_DEADLINE), UniLM.VectorStoreCallError)
    @test _seam_timeout(retrieve_vector_store("vs_x"; service=SeamProbe, config=_TINY_DEADLINE), UniLM.VectorStoreCallError)
    @test _seam_timeout(list_vector_stores(service=SeamProbe, config=_TINY_DEADLINE), UniLM.VectorStoreCallError)
    @test _seam_timeout(delete_vector_store("vs_x"; service=SeamProbe, config=_TINY_DEADLINE), UniLM.VectorStoreCallError)
    @test _seam_timeout(add_vector_store_file("vs_x", "file_x"; service=SeamProbe, config=_TINY_DEADLINE), UniLM.VectorStoreCallError)
    @test _seam_timeout(create_file_batch("vs_x", ["file_x"]; service=SeamProbe, config=_TINY_DEADLINE), UniLM.VectorStoreCallError)
    @test _seam_timeout(retrieve_file_batch("vs_x", "batch_x"; service=SeamProbe, config=_TINY_DEADLINE), UniLM.VectorStoreCallError)
    # poll_file_batch forwards config to retrieve_file_batch (one iteration reaches the seam)
    @test _seam_timeout(poll_file_batch("vs_x", "batch_x"; interval=0.01, timeout=0.05, service=SeamProbe, config=_TINY_DEADLINE), UniLM.VectorStoreCallError)
end

@testset "poll_file_batch: a wall-clock deadline that rides out transient failures" begin
    fb(status) = JSON.json(Dict("id" => "vsfb_1", "status" => status, "file_counts" => Dict("total" => 1)))
    @test_throws ArgumentError poll_file_batch("vs_1", "vsfb_1"; interval=0, service=SeamProbe)
    @test_throws ArgumentError poll_file_batch("vs_1", "vsfb_1"; timeout=-1, service=SeamProbe)

    script = n -> n == 2 ? _json(503, "{}") : _json(200, fb(n == 1 ? "in_progress" : "completed"))
    r, seen = _with_scripted((n, _) -> script(n)) do
        poll_file_batch("vs_1", "vsfb_1"; interval=0.01, timeout=30.0, service=URLProbe)
    end
    @test r isa UniLM.VectorStoreBatchSuccess && r.response.status == "completed"
    @test length(seen) == 3

    # Never terminal: the timeout reports the last batch it saw.
    t, _ = _with_scripted((_, _) -> _json(200, fb("in_progress"))) do
        poll_file_batch("vs_1", "vsfb_1"; interval=0.1, timeout=1.0, service=URLProbe)
    end
    @test t isa UniLM.VectorStoreCallError
    @test t.cause isa UniLMTimeout && t.cause.phase === :deadline
    @test t.last_observed isa UniLM.VectorStoreFileBatch && t.last_observed.status == "in_progress"
end

@testset "poll_file_batch: a cancel ends the poll at once" begin
    fb = JSON.json(Dict("id" => "vsfb_1", "status" => "in_progress", "file_counts" => Dict("total" => 1)))
    hits = Threads.Atomic{Int}(0)
    tok = CancelToken()
    (r, after_cancel), _ = _with_scripted((_, _) -> (Threads.atomic_add!(hits, 1); _json(200, fb))) do
        t = Threads.@spawn (poll_file_batch("vs_1", "vsfb_1"; interval=30.0, timeout=120.0,
                                            service=URLProbe, cancel=tok), time_ns())
        @test timedwait(() -> hits[] >= 1, 25.0) === :ok           # first GET answered: pausing
        sleep(0.2)
        cancelled_at = time_ns()
        cancel!(tok)
        @test timedwait(() -> istaskdone(t), 25.0) === :ok
        res, done_at = fetch(t)
        res, (done_at - cancelled_at) / 1e9
    end
    @test after_cancel < 5.0                                       # the 30 s pause is not slept out
    @test r isa UniLM.VectorStoreCallError && r.cause isa UniLMCancelled
    @test r.last_observed isa UniLM.VectorStoreFileBatch && r.last_observed.status == "in_progress"
end

@testset "Vector Stores API — a failure keeps the request id the service sent" begin
    r = _answered(() -> retrieve_vector_store("vs_x"; service=URLProbe), 404;
                  headers=["x-request-id" => "req_vs"])
    @test r isa UniLM.VectorStoreFailure && r.request_id == "req_vs"
end

@testset "Vector Stores API — separator-bearing ids stay single path segments" begin
    t = _recorded_targets() do
        retrieve_vector_store(_HOSTILE_ID; service=URLProbe)
        delete_vector_store(_HOSTILE_ID; service=URLProbe)
        add_vector_store_file(_HOSTILE_ID, "file-1"; service=URLProbe)
        create_file_batch(_HOSTILE_ID, ["file-1"]; service=URLProbe)
        retrieve_file_batch(_HOSTILE_ID, _HOSTILE_ID; service=URLProbe)
        list_vector_stores(; limit=2, after=_HOSTILE_ID, service=URLProbe)
    end
    @test t == ["/v1/vector_stores/$_HOSTILE_ENC",
                "/v1/vector_stores/$_HOSTILE_ENC",
                "/v1/vector_stores/$_HOSTILE_ENC/files",
                "/v1/vector_stores/$_HOSTILE_ENC/file_batches",
                "/v1/vector_stores/$_HOSTILE_ENC/file_batches/$_HOSTILE_ENC",
                "/v1/vector_stores?limit=2&after=$_HOSTILE_ENC"]
    g = _recorded_targets() do
        retrieve_file_batch("vs_abc123", "vsfb_abc123"; service=URLProbe)
        list_vector_stores(; after="vs_abc123", service=URLProbe)
    end
    @test g == ["/v1/vector_stores/vs_abc123/file_batches/vsfb_abc123",
                "/v1/vector_stores?after=vs_abc123"]
end

@testset "list_vector_stores: a 200 without data or has_more is a call error, not an empty last page" begin
    list(body) = _answered(() -> list_vector_stores(; service=URLProbe), 200; body)
    @test list("{}") isa VectorStoreCallError
    @test list("""{"object": "list", "data": []}""") isa VectorStoreCallError
    @test list("""{"object": "list", "has_more": false}""") isa VectorStoreCallError
    ok = list("""{"object": "list", "data": [{"id": "vs_1", "name": "a", "status": "completed"}], "has_more": true}""")
    @test ok isa VectorStoreListSuccess && [v.id for v in ok.response.data] == ["vs_1"] && ok.response.has_more
end

@testset "vector-store decoders: a reply without the status OpenAI requires is a call error" begin
    @test _answered(() -> retrieve_vector_store("vs_1"; service=URLProbe), 200;
                    body="""{"id": "vs_1", "name": "a"}""") isa VectorStoreCallError
    @test _answered(() -> retrieve_file_batch("vs_1", "vsfb_1"; service=URLProbe), 200;
                    body="""{"id": "vsfb_1"}""") isa VectorStoreCallError
    @test _answered(() -> add_vector_store_file("vs_1", "file-1"; service=URLProbe), 200;
                    body="""{"id": "file-1"}""") isa VectorStoreCallError
    # A batch without a status is malformed, not still running: the poll ends at its first reply.
    r, seen = _with_scripted((_, _) -> _json(200, """{"id": "vsfb_1"}""")) do
        poll_file_batch("vs_1", "vsfb_1"; interval=0.05, timeout=5.0, service=URLProbe)
    end
    @test r isa VectorStoreCallError && !(r.cause isa UniLMTimeout) && length(seen) == 1
end

@testset "delete_vector_store reports success only when the service confirms the delete" begin
    del(body) = _answered(() -> delete_vector_store("vs_1"; service=URLProbe), 200; body)
    @test del("""{"id": "vs_1", "object": "vector_store.deleted", "deleted": true}""") ==
          UniLM.VectorStoreDeleteSuccess(id="vs_1", deleted=true)
    @test del("""{"id": "vs_1", "object": "vector_store.deleted"}""") isa UniLM.VectorStoreCallError
    @test del("""{"id": "vs_1", "object": "vector_store.deleted", "deleted": false}""") isa UniLM.VectorStoreCallError
end

@testset "delete_vector_store: the record's id and object are required, never filled in" begin
    del(body) = _answered(() -> delete_vector_store("vs_1"; service=URLProbe), 200; body)
    # No id: the requested one is not put in its place.
    @test del("""{"object": "vector_store.deleted", "deleted": true}""") isa VectorStoreCallError
    # The record names what it deleted: no name, or another kind's, does not confirm this delete.
    @test del("""{"id": "vs_1", "deleted": true}""") isa VectorStoreCallError
    @test del("""{"id": "vs_1", "object": "vector_store.file.deleted", "deleted": true}""") isa VectorStoreCallError
    @test del("""{"id": "vs_2", "object": "vector_store.deleted", "deleted": true}""").id == "vs_2"
end

@testset "create_vector_store sends a description; list_vector_stores takes order and before" begin
    r, seen = _with_scripted((_, _) -> _json(200, """{"id": "vs_1", "name": "n", "status": "completed"}""")) do
        create_vector_store(; name="n", description="Return policies", service=URLProbe)
    end
    @test r isa VectorStoreSuccess
    @test JSON.parse(only(seen).body) == Dict("name" => "n", "description" => "Return policies")
    t = _recorded_targets() do
        list_vector_stores(; limit=2, order="asc", after="vs_a", before="vs_b", service=URLProbe)
        list_vector_stores(; order="desc", service=URLProbe)
    end
    @test t == ["/v1/vector_stores?limit=2&order=asc&after=vs_a&before=vs_b", "/v1/vector_stores?order=desc"]
    # An order OpenAI's reference does not list is refused before anything is sent.
    err, seen = _with_scripted((_, _) -> _json(200, "{}")) do
        try list_vector_stores(; order="ascending", service=URLProbe); nothing catch e; e end
    end
    @test err isa ArgumentError && occursin("list_vector_stores", err.msg) && occursin("\"asc\"", err.msg) && isempty(seen)
end

@testset "update_vector_store: a POST to the store with the fields given" begin
    body = """{"id": "vs_1", "name": "renamed", "status": "completed", "metadata": {"team": "docs"}}"""
    r, seen = _with_scripted((_, _) -> _json(200, body)) do
        update_vector_store("vs_1"; name="renamed", expires_after=Dict("anchor" => "last_active_at", "days" => 7),
                            metadata=Dict("team" => "docs"), service=URLProbe)
    end
    @test r isa VectorStoreSuccess && r.response.name == "renamed"
    req = only(seen)
    @test (req.method, req.target) == ("POST", "/v1/vector_stores/vs_1")
    @test JSON.parse(req.body) == Dict("name" => "renamed", "metadata" => Dict("team" => "docs"),
                                       "expires_after" => Dict("anchor" => "last_active_at", "days" => 7))
    # Only the fields given are sent.
    _, seen = _with_scripted((_, _) -> _json(200, body)) do
        update_vector_store("vs_1"; metadata=Dict("team" => "docs"), service=URLProbe)
    end
    @test JSON.parse(only(seen).body) == Dict("metadata" => Dict("team" => "docs"))
    @test _answered(() -> update_vector_store("vs_1"; name="n", service=URLProbe), 200;
                    body="""{"id": "vs_1"}""") isa VectorStoreCallError
    @test _seam_timeout(update_vector_store("vs_x"; name="n", service=SeamProbe, config=_TINY_DEADLINE), VectorStoreCallError)
    @test _recorded_targets(() -> update_vector_store(_HOSTILE_ID; name="n", service=URLProbe)) ==
          ["/v1/vector_stores/$_HOSTILE_ENC"]
    @test_throws ArgumentError update_vector_store("vs_1"; name="n", service=DeepSeekEndpoint("k"))
end

# A vector store file as OpenAI's reference shows one.
const _VS_FILE_JSON = """{"id": "file-1", "object": "vector_store.file", "usage_bytes": 120, "created_at": 1,
    "vector_store_id": "vs_1", "status": "completed", "last_error": null, "attributes": {"team": "docs"}}"""

# Each request a call sends, as (method, target), against an endpoint that answers `{}`.
_sent_requests(f::Function) = [(r.method, r.target) for r in last(_with_scripted(f, (_, _) -> _json(200, "{}")))]

@testset "vector store files: list, retrieve, delete, and read the parsed content" begin
    page = """{"object": "list", "data": [$_VS_FILE_JSON], "first_id": "file-1", "last_id": "file-1", "has_more": false}"""
    l, seen = _with_scripted((_, _) -> _json(200, page)) do
        list_vector_store_files("vs_1"; limit=5, order="asc", after="file-0", before="file-9", filter="completed",
                                service=URLProbe)
    end
    @test l isa VectorStoreFileListSuccess && l.response isa VectorStoreFileList
    @test [(f.id, f.status) for f in l.response.data] == [("file-1", "completed")] && !l.response.has_more
    @test l.response.data[1].raw["attributes"] == Dict("team" => "docs")
    @test only(seen).target == "/v1/vector_stores/vs_1/files?limit=5&order=asc&after=file-0&before=file-9&filter=completed"
    @test _answered(() -> list_vector_store_files("vs_1"; service=URLProbe), 200; body="{}") isa VectorStoreCallError
    @test_throws ArgumentError list_vector_store_files("vs_1"; filter="done", service=URLProbe)
    @test_throws ArgumentError list_vector_store_files("vs_1"; order="up", service=URLProbe)

    g = _answered(() -> retrieve_vector_store_file("vs_1", "file-1"; service=URLProbe), 200; body=_VS_FILE_JSON)
    @test g isa VectorStoreFileSuccess && (g.response.id, g.response.status) == ("file-1", "completed")
    @test _answered(() -> retrieve_vector_store_file("vs_1", "file-1"; service=URLProbe), 200;
                    body="""{"id": "file-1"}""") isa VectorStoreCallError

    del(body) = _answered(() -> delete_vector_store_file("vs_1", "file-1"; service=URLProbe), 200; body)
    @test del("""{"id": "file-1", "object": "vector_store.file.deleted", "deleted": true}""") ==
          VectorStoreDeleteSuccess(id="file-1", deleted=true)
    @test del("""{"id": "file-1", "object": "vector_store.deleted", "deleted": true}""") isa VectorStoreCallError
    @test del("""{"object": "vector_store.file.deleted", "deleted": true}""") isa VectorStoreCallError
    @test del("""{"id": "file-1", "object": "vector_store.file.deleted", "deleted": false}""") isa VectorStoreCallError

    content = """{"object": "vector_store.file_content.page", "data": [{"type": "text", "text": "Returns within 30 days."}],
                  "has_more": false, "next_page": null}"""
    c = _answered(() -> vector_store_file_content("vs_1", "file-1"; service=URLProbe), 200; body=content)
    @test c isa VectorStoreContentSuccess && c.response isa VectorStoreFileContent
    @test (c.response.data, c.response.has_more, c.response.next_page) ==
          ([Dict{String,Any}("type" => "text", "text" => "Returns within 30 days.")], false, nothing)
    @test _answered(() -> vector_store_file_content("vs_1", "file-1"; service=URLProbe), 200;
                    body="""{"data": [], "has_more": false}""") isa VectorStoreCallError
end

@testset "vector store files: methods, ids as single path segments, the config seam, the capability" begin
    E = _HOSTILE_ENC
    sent = _sent_requests() do
        list_vector_store_files(_HOSTILE_ID; service=URLProbe)
        retrieve_vector_store_file(_HOSTILE_ID, _HOSTILE_ID; service=URLProbe)
        delete_vector_store_file(_HOSTILE_ID, _HOSTILE_ID; service=URLProbe)
        vector_store_file_content(_HOSTILE_ID, _HOSTILE_ID; service=URLProbe)
    end
    @test sent == [("GET", "/v1/vector_stores/$E/files"), ("GET", "/v1/vector_stores/$E/files/$E"),
                   ("DELETE", "/v1/vector_stores/$E/files/$E"), ("GET", "/v1/vector_stores/$E/files/$E/content")]
    for call in (cfg -> list_vector_store_files("vs_x"; service=SeamProbe, config=cfg),
                 cfg -> retrieve_vector_store_file("vs_x", "f"; service=SeamProbe, config=cfg),
                 cfg -> delete_vector_store_file("vs_x", "f"; service=SeamProbe, config=cfg),
                 cfg -> vector_store_file_content("vs_x", "f"; service=SeamProbe, config=cfg))
        @test _seam_timeout(call(_TINY_DEADLINE), VectorStoreCallError)
    end
    ds = DeepSeekEndpoint("k")
    @test_throws ArgumentError list_vector_store_files("vs_1"; service=ds)
    @test_throws ArgumentError retrieve_vector_store_file("vs_1", "f"; service=ds)
    @test_throws ArgumentError delete_vector_store_file("vs_1", "f"; service=ds)
    @test_throws ArgumentError vector_store_file_content("vs_1", "f"; service=ds)
end

@testset "vector store file attributes: sent on add and on batches, updated, cleared, bounded as OpenAI documents" begin
    attrs = Dict("team" => "docs", "year" => 2026, "public" => true, "score" => 0.5)
    ok = """{"id": "x", "status": "in_progress"}"""
    sent(f) = JSON.parse(only(last(_with_scripted(f, (_, _) -> _json(200, ok)))).body)
    @test sent(() -> add_vector_store_file("vs_1", "file-1"; attributes=attrs, service=URLProbe)) ==
          Dict("file_id" => "file-1", "attributes" => attrs)
    @test sent(() -> create_file_batch("vs_1", ["file-1", "file-2"]; attributes=attrs, service=URLProbe)) ==
          Dict("file_ids" => ["file-1", "file-2"], "attributes" => attrs)
    # Entries of their own go as `files`, each with its attributes and chunking.
    static = Dict("type" => "static", "static" => Dict("max_chunk_size_tokens" => 400, "chunk_overlap_tokens" => 100))
    entries = [VectorStoreFileEntry(file_id="file-1", attributes=Dict("team" => "a")),
               VectorStoreFileEntry(file_id="file-2", chunking_strategy=static), VectorStoreFileEntry(file_id="file-3")]
    @test sent(() -> create_file_batch("vs_1", entries; service=URLProbe)) ==
          Dict("files" => [Dict("file_id" => "file-1", "attributes" => Dict("team" => "a")),
                           Dict("file_id" => "file-2", "chunking_strategy" => static), Dict("file_id" => "file-3")])
    # OpenAI ignores a batch's own attributes and chunking beside entries, so they cannot be given.
    @test_throws MethodError create_file_batch("vs_1", entries; attributes=attrs, service=URLProbe)
    @test_throws MethodError create_file_batch("vs_1", entries; chunking_strategy=static, service=URLProbe)

    u, seen = _with_scripted((_, _) -> _json(200, """{"id": "file-1", "status": "completed", "attributes": {"team": "ops"}}""")) do
        update_vector_store_file("vs_1", "file-1", Dict("team" => "ops"); service=URLProbe)
    end
    @test u isa VectorStoreFileSuccess && u.response.raw["attributes"] == Dict("team" => "ops")
    @test (only(seen).method, only(seen).target, JSON.parse(only(seen).body)) ==
          ("POST", "/v1/vector_stores/vs_1/files/file-1", Dict("attributes" => Dict("team" => "ops")))
    # `nothing` clears them: the field OpenAI requires goes as null.
    @test only(last(_with_scripted(() -> update_vector_store_file("vs_1", "file-1", nothing; service=URLProbe),
                                   (_, _) -> _json(200, ok)))).body == """{"attributes":null}"""
    @test _seam_timeout(update_vector_store_file("vs_x", "f", nothing; service=SeamProbe, config=_TINY_DEADLINE),
                        VectorStoreCallError)
    @test _recorded_targets(() -> update_vector_store_file(_HOSTILE_ID, _HOSTILE_ID, nothing; service=URLProbe)) ==
          ["/v1/vector_stores/$_HOSTILE_ENC/files/$_HOSTILE_ENC"]
    @test_throws ArgumentError update_vector_store_file("vs_1", "f", nothing; service=DeepSeekEndpoint("k"))

    # 16 pairs, a 64-character key and a 512-character value are within OpenAI's limits.
    edge = merge(Dict("k$i" => i for i in 1:14), Dict("k"^64 => 1, "v" => "v"^512))
    @test sent(() -> add_vector_store_file("vs_1", "file-1"; attributes=edge, service=URLProbe))["attributes"] == edge
    # Past them, the call is refused before anything is sent.
    function refused(attrs)
        err, seen = _with_scripted((_, _) -> _json(200, ok)) do
            try add_vector_store_file("vs_1", "file-1"; attributes=attrs, service=URLProbe); nothing catch e; e end
        end
        err isa ArgumentError && isempty(seen) ? err.msg : ""
    end
    @test occursin("16", refused(Dict("k$i" => i for i in 1:17)))
    @test occursin("64", refused(Dict("k"^65 => 1)))
    @test occursin("512", refused(Dict("k" => "v"^513)))
    @test occursin("a string, a number or a boolean", refused(Dict("k" => [1, 2])))
    @test occursin("add_vector_store_file", refused(Dict("k" => nothing)))
    @test_throws ArgumentError VectorStoreFileEntry(file_id="f", attributes=Dict("k" => [1]))
    @test_throws ArgumentError update_vector_store_file("vs_1", "f", Dict("k" => "v"^513); service=URLProbe)
    @test_throws ArgumentError create_file_batch("vs_1", ["f"]; attributes=Dict("k"^65 => 1), service=URLProbe)
end

@testset "cancel_file_batch and list_file_batch_files" begin
    batch = """{"id": "vsfb_1", "object": "vector_store.file_batch", "status": "cancelling", "file_counts": {"total": 2}}"""
    c, seen = _with_scripted((_, _) -> _json(200, batch)) do
        cancel_file_batch("vs_1", "vsfb_1"; service=URLProbe)
    end
    @test c isa VectorStoreBatchSuccess && c.response.status == "cancelling"
    @test (only(seen).method, only(seen).target) == ("POST", "/v1/vector_stores/vs_1/file_batches/vsfb_1/cancel")
    @test _answered(() -> cancel_file_batch("vs_1", "vsfb_1"; service=URLProbe), 200;
                    body="""{"id": "vsfb_1"}""") isa VectorStoreCallError
    page = """{"object": "list", "data": [{"id": "file-1", "status": "cancelled"}], "has_more": true}"""
    l, seen = _with_scripted((_, _) -> _json(200, page)) do
        list_file_batch_files("vs_1", "vsfb_1"; limit=1, order="desc", after="file-0", before="file-2", filter="cancelled",
                              service=URLProbe)
    end
    @test l isa VectorStoreFileListSuccess && [(f.id, f.status) for f in l.response.data] == [("file-1", "cancelled")]
    @test l.response.has_more
    @test (only(seen).method, only(seen).target) ==
          ("GET", "/v1/vector_stores/vs_1/file_batches/vsfb_1/files?limit=1&order=desc&after=file-0&before=file-2&filter=cancelled")
    @test _answered(() -> list_file_batch_files("vs_1", "vsfb_1"; service=URLProbe), 200;
                    body="""{"data": []}""") isa VectorStoreCallError
    @test_throws ArgumentError list_file_batch_files("vs_1", "vsfb_1"; order="newest", service=URLProbe)
    @test_throws ArgumentError list_file_batch_files("vs_1", "vsfb_1"; filter="expired", service=URLProbe)
    t = _recorded_targets() do
        cancel_file_batch(_HOSTILE_ID, _HOSTILE_ID; service=URLProbe)
        list_file_batch_files(_HOSTILE_ID, _HOSTILE_ID; service=URLProbe)
    end
    @test t == ["/v1/vector_stores/$_HOSTILE_ENC/file_batches/$_HOSTILE_ENC/cancel",
                "/v1/vector_stores/$_HOSTILE_ENC/file_batches/$_HOSTILE_ENC/files"]
    @test _seam_timeout(cancel_file_batch("vs_x", "b"; service=SeamProbe, config=_TINY_DEADLINE), VectorStoreCallError)
    @test _seam_timeout(list_file_batch_files("vs_x", "b"; service=SeamProbe, config=_TINY_DEADLINE), VectorStoreCallError)
    @test_throws ArgumentError cancel_file_batch("vs_1", "b"; service=DeepSeekEndpoint("k"))
    @test_throws ArgumentError list_file_batch_files("vs_1", "b"; service=DeepSeekEndpoint("k"))
end
