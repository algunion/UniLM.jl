# Gemini File Search: stores, their documents and the operations that fill them, and the Interactions
# file_search tool. Zero-spend: the verbs talk to a scripted local endpoint.
using UniLM
using UniLM: GEMINIServiceEndpoint
using Test, HTTP, JSON, Dates

# A Gemini endpoint whose File Search requests `_with_scripted` serves (platform_seam_fixtures.jl).
const _FS = FileSearchProbe

# Replies as the API sent them on 2026-11-17.
const _STORE = """{"name": "fileSearchStores/unilmwitness-5r9bfby3t61y", "displayName": "unilm-witness",
  "createTime": "2026-11-17T07:55:37.939169Z", "updateTime": "2026-11-17T07:55:37.939169Z",
  "embeddingModel": "models/gemini-embedding-001"}"""
const _STORE_FILLED = """{"name": "fileSearchStores/unilmwitness-5r9bfby3t61y", "displayName": "unilm-witness",
  "createTime": "2026-11-17T07:55:37.939169Z", "updateTime": "2026-11-17T07:55:37.939169Z",
  "activeDocumentsCount": "2", "sizeBytes": "151", "embeddingModel": "models/gemini-embedding-001"}"""
const _DOC = """{"name": "fileSearchStores/unilmwitness-5r9bfby3t61y/documents/unilmnote-q9cxqhi9tga3",
  "displayName": "unilm-note", "customMetadata": [{"key": "topic", "stringValue": "birds"}],
  "updateTime": "2026-11-17T07:55:42.884117Z", "createTime": "2026-11-17T07:55:41.196631Z", "state": "STATE_ACTIVE",
  "sizeBytes": "76", "mimeType": "text/plain"}"""
const _DOC_RIVER = """{"name": "fileSearchStores/unilmwitness2-tvwoyhlnmjpq/documents/unilmriver-5dwx4aoj6mmn",
  "displayName": "unilm-river", "customMetadata": [{"key": "tags", "stringListValue": {"values": ["a", "b"]}},
  {"key": "n", "numericValue": 2.5}], "updateTime": "2026-11-17T08:00:15.829757Z",
  "createTime": "2026-11-17T08:00:14.137119Z", "state": "STATE_ACTIVE", "sizeBytes": "79", "mimeType": "text/plain"}"""
const _UPLOAD_OP = """{"name": "fileSearchStores/unilmwitness-5r9bfby3t61y/upload/operations/unilmnote-q9cxqhi9tga3",
  "response": {"@type": "type.googleapis.com/google.ai.generativelanguage.v1main.UploadToFileSearchStoreResponse",
  "parent": "fileSearchStores/unilmwitness-5r9bfby3t61y",
  "documentName": "fileSearchStores/unilmwitness-5r9bfby3t61y/documents/unilmnote-q9cxqhi9tga3",
  "mimeType": "text/plain", "sizeBytes": "76"}}"""
const _IMPORT_OP = """{"name": "fileSearchStores/unilmwitness-5r9bfby3t61y/operations/u5qxauk3eomh-vufwqsj9n41q",
  "response": {"@type": "type.googleapis.com/google.ai.generativelanguage.v1main.ImportFileResponse"}}"""
const _IMPORT_OP_PENDING = """{"name": "fileSearchStores/unilmwitness-5r9bfby3t61y/operations/u5qxauk3eomh-vufwqsj9n41q"}"""
const _IMPORT_OP_DONE = """{"name": "fileSearchStores/unilmwitness-5r9bfby3t61y/operations/u5qxauk3eomh-vufwqsj9n41q",
  "done": true, "response": {"@type": "type.googleapis.com/google.ai.generativelanguage.v1main.ImportFileResponse",
  "parent": "unilmwitness-5r9bfby3t61y", "documentName": "u5qxauk3eomh-vufwqsj9n41q"}}"""
const _STORE_NAME = "fileSearchStores/unilmwitness-5r9bfby3t61y"

_fs_err(f) = try f(); nothing catch e; e end
_hdrs(req) = Dict(lowercase(String(k)) => String(v) for (k, v) in req.headers)

# Refuted if: a store's field is misread, a count Gemini leaves out at zero is not zero, or a request reaches
# another path or sends another body.
@testset "File Search stores — create, retrieve, list, delete" begin
    r, seen = _with_scripted(() -> create_file_search_store(; display_name="unilm-witness", embedding_model="gemini-embedding-2",
                                                            service=_FS), (_, _) -> _json(200, _STORE))
    @test [(q.method, q.target) for q in seen] == [("POST", "/v1beta/fileSearchStores")]
    @test JSON.parse(seen[1].body) == Dict("displayName" => "unilm-witness", "embeddingModel" => "models/gemini-embedding-2")
    @test r isa FileSearchStoreSuccess
    s = r.response
    @test (s.name, s.display_name, s.embedding_model) == (_STORE_NAME, "unilm-witness", "gemini-embedding-001")
    @test (s.active_documents, s.pending_documents, s.failed_documents, s.size_bytes) == (0, 0, 0, 0)
    @test s.create_time == s.update_time == DateTime(2026, 11, 17, 7, 55, 37, 939)
    _, seen = _with_scripted(() -> create_file_search_store(; service=_FS), (_, _) -> _json(200, _STORE))
    @test JSON.parse(seen[1].body) == Dict()
    for name in ("unilmwitness-5r9bfby3t61y", _STORE_NAME)
        r, seen = _with_scripted(() -> retrieve_file_search_store(name; service=_FS), (_, _) -> _json(200, _STORE_FILLED))
        @test [(q.method, q.target) for q in seen] == [("GET", "/v1beta/" * _STORE_NAME)]
        @test r isa FileSearchStoreSuccess && (r.response.active_documents, r.response.size_bytes) == (2, 151)
    end
    page = """{"fileSearchStores": [$_STORE], "nextPageToken": "n"}"""
    r, seen = _with_scripted(() -> list_file_search_stores(; limit=20, after="t", service=_FS), (_, _) -> _json(200, page))
    @test seen[1].target == "/v1beta/fileSearchStores?pageSize=20&pageToken=t"
    @test r isa FileSearchStoreListSuccess && only(r.response.data).name == _STORE_NAME && r.response.has_more
    r = _answered(() -> list_file_search_stores(; service=_FS), 200; body="{}")
    @test r isa FileSearchStoreListSuccess && isempty(r.response.data) && !r.response.has_more
    # A store with documents is deleted only with force: the API answered "Cannot delete non-empty
    # FileSearchStore" (HTTP 400) without it.
    r, seen = _with_scripted(() -> delete_file_search_store("unilmwitness-5r9bfby3t61y"; service=_FS), (_, _) -> _json(200, "{}"))
    @test [(q.method, q.target) for q in seen] == [("DELETE", "/v1beta/" * _STORE_NAME)]
    @test r isa FileSearchDeleteSuccess && r.name == _STORE_NAME && r.deleted
    _, seen = _with_scripted(() -> delete_file_search_store(_STORE_NAME; force=true, service=_FS), (_, _) -> _json(200, "{}"))
    @test seen[1].target == "/v1beta/" * _STORE_NAME * "?force=true"
    nonempty = """{"error": {"code": 400, "message": "Cannot delete non-empty FileSearchStore", "status": "FAILED_PRECONDITION"}}"""
    r = _answered(() -> delete_file_search_store(_STORE_NAME; service=_FS), 400; body=nonempty)
    @test r isa FileSearchFailure && r.status == 400 && occursin("non-empty", r.response)
end

# Refuted if: any of these reaches the network.
@testset "File Search stores — refused before any request" begin
    cases = [(() -> create_file_search_store(; service=OPENAIServiceEndpoint), "File Search stores is not supported"),
             (() -> create_file_search_store(; display_name="x"^513, service=_FS), "512"),
             (() -> list_file_search_stores(; limit=21, service=_FS), "1 to 20"),
             (() -> list_file_search_stores(; limit=0, service=_FS), "1 to 20"),
             (() -> retrieve_file_search_store("a/b"; service=_FS), "fileSearchStores/"),
             (() -> retrieve_file_search_store(""; service=_FS), "fileSearchStores/"),
             (() -> delete_file_search_store("cachedContents/x"; service=_FS), "fileSearchStores/"),
             (() -> list_file_search_documents(_STORE_NAME; limit=21, service=_FS), "1 to 20"),
             (() -> retrieve_file_search_document(_STORE_NAME; service=_FS), "documents/"),
             (() -> retrieve_file_search_document("fileSearchStores/s/documents/a/b"; service=_FS), "documents/"),
             (() -> delete_file_search_document("documents/d"; service=_FS), "documents/"),
             (() -> retrieve_file_search_operation("operations/o"; service=_FS), "operations/"),
             (() -> retrieve_file_search_operation("fileSearchStores/s/documents/o"; service=_FS), "operations/"),
             (() -> import_to_file_search_store(_STORE_NAME, "a/b"; service=_FS), "files/"),
             (() -> poll_file_search_operation("operations/o"; service=_FS), "operations/")]
    for (call, says) in cases
        e, seen = _with_scripted(() -> _fs_err(call), (_, _) -> _json(200, _STORE))
        @test e isa ArgumentError && occursin(says, e.msg) && isempty(seen)
    end
end

# Gemini's resumable upload into a store: a `start` POST at the store's upload address with the key, the size, the
# MIME type and the document's settings, the bytes at the session's address without the key, and the operation back.
# Refuted if: a header, a setting or the session is wrong, the key goes with the bytes, or the operation is misread.
@testset "upload_to_file_search_store — a resumable upload, its settings, the operation back" begin
    path = tempname() * ".txt"
    write(path, "UniLM witness note.")
    session() = _url_probe_base[] * "/upload/v1beta/session?upload_id=u1"
    started() = HTTP.Response(200, ["X-Goog-Upload-URL" => session(), "X-Goog-Upload-Status" => "active"], UInt8[])
    try
        hdrs = Dict{String,String}[]
        r, seen = _with_scripted((n, req) -> (push!(hdrs, _hdrs(req)); n == 1 ? started() : _json(200, _UPLOAD_OP))) do
            upload_to_file_search_store("unilmwitness-5r9bfby3t61y", path; display_name="unilm-note", mime_type="text/plain",
                                        custom_metadata=Dict("topic" => "birds"), max_tokens_per_chunk=100,
                                        max_overlap_tokens=10, service=_FS)
        end
        @test [(q.method, q.target) for q in seen] ==
              [("POST", "/upload/v1beta/" * _STORE_NAME * ":uploadToFileSearchStore"), ("POST", "/upload/v1beta/session?upload_id=u1")]
        @test JSON.parse(seen[1].body) == Dict("displayName" => "unilm-note", "mimeType" => "text/plain",
            "customMetadata" => [Dict("key" => "topic", "stringValue" => "birds")],
            "chunkingConfig" => Dict("whiteSpaceConfig" => Dict("maxTokensPerChunk" => 100, "maxOverlapTokens" => 10)))
        start, bytes = hdrs
        @test [start[k] for k in ("x-goog-api-key", "x-goog-upload-protocol", "x-goog-upload-command",
                                  "x-goog-upload-header-content-length", "x-goog-upload-header-content-type")] ==
              ["t", "resumable", "start", "19", "text/plain"]
        @test (bytes["x-goog-upload-offset"], bytes["x-goog-upload-command"]) == ("0", "upload, finalize")
        @test !haskey(bytes, "x-goog-api-key") && seen[2].body == "UniLM witness note."
        @test r isa FileSearchOperationSuccess
        op = r.response
        @test op.name == _STORE_NAME * "/upload/operations/unilmnote-q9cxqhi9tga3" && !op.done && isnothing(op.error)
        @test op.document == _STORE_NAME * "/documents/unilmnote-q9cxqhi9tga3"
        # Without settings the start sends `{}`, and the MIME type its extension gives.
        hdrs = Dict{String,String}[]
        _, seen = _with_scripted((n, req) -> (push!(hdrs, _hdrs(req)); n == 1 ? started() : _json(200, _UPLOAD_OP))) do
            upload_to_file_search_store(_STORE_NAME, path; service=_FS)
        end
        @test JSON.parse(seen[1].body) == Dict() && hdrs[1]["x-goog-upload-header-content-type"] == "text/plain"
        # A refused start is the call's failure, and sends no bytes; one that names no session, a call error.
        r, seen = _with_scripted(() -> upload_to_file_search_store(_STORE_NAME, path; service=_FS), (_, _) -> _json(404, "{}"))
        @test r isa FileSearchFailure && r.status == 404 && length(seen) == 1
        r, seen = _with_scripted(() -> upload_to_file_search_store(_STORE_NAME, path; service=_FS), (_, _) -> _json(200, "{}"))
        @test r isa FileSearchCallError && occursin("X-Goog-Upload-URL", r.error) && length(seen) == 1
        # Refused before any request: a missing file, document settings outside what the API takes.
        cases = [(() -> upload_to_file_search_store(_STORE_NAME, path * ".missing"; service=_FS), "not found"),
                 (() -> upload_to_file_search_store(_STORE_NAME, path; max_tokens_per_chunk=513, service=_FS), "512"),
                 (() -> upload_to_file_search_store(_STORE_NAME, path; max_tokens_per_chunk=0, service=_FS), "512"),
                 (() -> upload_to_file_search_store(_STORE_NAME, path; max_overlap_tokens=-1, service=_FS), "max_overlap_tokens"),
                 (() -> upload_to_file_search_store(_STORE_NAME, path; custom_metadata=Dict("flag" => true), service=_FS), "flag"),
                 (() -> upload_to_file_search_store(_STORE_NAME, path; custom_metadata=Dict("ns" => [1, 2]), service=_FS), "ns"),
                 (() -> upload_to_file_search_store(_STORE_NAME, path;
                        custom_metadata=Dict("k$i" => i for i in 1:21), service=_FS), "20"),
                 (() -> upload_to_file_search_store("a/b", path; service=_FS), "fileSearchStores/")]
        for (call, says) in cases
            e, seen = _with_scripted(() -> _fs_err(call), (_, _) -> _json(200, _UPLOAD_OP))
            @test e isa ArgumentError && occursin(says, e.msg) && isempty(seen)
        end
    finally
        rm(path; force=true)
    end
    # A document of more than 100 MB, the API's limit, is refused unread (a sparse file: no disk is written).
    big = tempname() * ".txt"
    try
        open(io -> truncate(io, 104_857_601), big, "w")
        e, seen = _with_scripted(() -> _fs_err(() -> upload_to_file_search_store(_STORE_NAME, big; service=_FS)),
                                 (_, _) -> _json(200, _UPLOAD_OP))
        @test e isa ArgumentError && occursin("100 MB", e.msg) && isempty(seen)
    finally
        rm(big; force=true)
    end
end

# Refuted if: the import names another file, the metadata takes another shape, or a done operation's document is
# not named as the document verbs name it.
@testset "import_to_file_search_store, and the operation polled to its end" begin
    for file in ("abc", "files/abc")
        r, seen = _with_scripted(() -> import_to_file_search_store("unilmwitness-5r9bfby3t61y", file;
                                     custom_metadata=Dict("year" => 1934, "tags" => ["a", "b"], "n" => 2.5), service=_FS),
                                 (_, _) -> _json(200, _IMPORT_OP))
        @test [(q.method, q.target) for q in seen] == [("POST", "/v1beta/" * _STORE_NAME * ":importFile")]
        body = JSON.parse(seen[1].body)
        @test body["fileName"] == "files/abc"
        @test Set(body["customMetadata"]) == Set([Dict("key" => "year", "numericValue" => 1934),
            Dict("key" => "tags", "stringListValue" => Dict("values" => ["a", "b"])), Dict("key" => "n", "numericValue" => 2.5)])
        @test r isa FileSearchOperationSuccess && !r.response.done && isnothing(r.response.document)
    end
    opname = _STORE_NAME * "/operations/u5qxauk3eomh-vufwqsj9n41q"
    r, seen = _with_scripted(() -> retrieve_file_search_operation(opname; service=_FS), (_, _) -> _json(200, _IMPORT_OP_DONE))
    @test [(q.method, q.target) for q in seen] == [("GET", "/v1beta/" * opname)]
    # The import's done reply names the document and its store by id alone (observed 2026-11-17): the document is
    # named in the store its operation belongs to.
    @test r isa FileSearchOperationSuccess && r.response.done &&
          r.response.document == _STORE_NAME * "/documents/u5qxauk3eomh-vufwqsj9n41q"
    # The poll reads the operation until it is done: a pending one first, then the done one.
    r, seen = _with_scripted(() -> poll_file_search_operation(opname; interval=0.01, timeout=30.0, service=_FS),
                             (n, _) -> _json(200, n == 1 ? _IMPORT_OP_PENDING : _IMPORT_OP_DONE))
    @test length(seen) == 2 && r isa FileSearchOperationSuccess && r.response.done
    # An operation that ended in an error is done, its Status in `error`.
    failed = """{"name": "$opname", "done": true, "error": {"code": 3, "message": "Unsupported file type"}}"""
    r = _answered(() -> poll_file_search_operation(opname; interval=0.01, service=_FS), 200; body=failed)
    @test r isa FileSearchOperationSuccess && r.response.done && r.response.error["message"] == "Unsupported file type" &&
          isnothing(r.response.document)
    # Out of time: the call error carries the last operation seen.
    r = _answered(() -> poll_file_search_operation(opname; interval=0.05, timeout=0.3, service=_FS), 200; body=_IMPORT_OP_PENDING)
    @test r isa FileSearchCallError && r.cause isa UniLMTimeout && r.last_observed isa FileSearchOperation &&
          !r.last_observed.done
    # A reply without the operation's name is malformed.
    r = _answered(() -> retrieve_file_search_operation(opname; service=_FS), 200; body="""{"done": true}""")
    @test r isa FileSearchCallError && occursin("name", r.error)
end

# Refuted if: a document's metadata, state, size or times are misread, or a delete is sent without the force a
# document with chunks needs.
@testset "File Search documents — list, retrieve, delete" begin
    page = """{"documents": [$_DOC, $_DOC_RIVER], "nextPageToken": "n"}"""
    r, seen = _with_scripted(() -> list_file_search_documents("unilmwitness-5r9bfby3t61y"; limit=5, after="t", service=_FS),
                             (_, _) -> _json(200, page))
    @test seen[1].target == "/v1beta/" * _STORE_NAME * "/documents?pageSize=5&pageToken=t"
    @test r isa FileSearchDocumentListSuccess && length(r.response.data) == 2 && r.response.has_more
    d, river = r.response.data
    @test (d.name, d.display_name, d.state, d.mime_type, d.size_bytes) ==
          (_STORE_NAME * "/documents/unilmnote-q9cxqhi9tga3", "unilm-note", "STATE_ACTIVE", "text/plain", 76)
    @test d.custom_metadata == Dict("topic" => "birds")
    @test river.custom_metadata == Dict("tags" => ["a", "b"], "n" => 2.5)
    @test d.create_time == DateTime(2026, 11, 17, 7, 55, 41, 196) && d.update_time == DateTime(2026, 11, 17, 7, 55, 42, 884)
    r = _answered(() -> list_file_search_documents(_STORE_NAME; service=_FS), 200; body="{}")
    @test r isa FileSearchDocumentListSuccess && isempty(r.response.data)
    name = _STORE_NAME * "/documents/unilmnote-q9cxqhi9tga3"
    r, seen = _with_scripted(() -> retrieve_file_search_document(name; service=_FS), (_, _) -> _json(200, _DOC))
    @test [(q.method, q.target) for q in seen] == [("GET", "/v1beta/" * name)] && r isa FileSearchDocumentSuccess
    # A document with chunks is deleted only with force (HTTP 400 "Cannot delete non-empty Document" without it).
    r, seen = _with_scripted(() -> delete_file_search_document(name; service=_FS), (_, _) -> _json(200, "{}"))
    @test [(q.method, q.target) for q in seen] == [("DELETE", "/v1beta/" * name * "?force=true")]
    @test r isa FileSearchDeleteSuccess && r.name == name && r.deleted
    _, seen = _with_scripted(() -> delete_file_search_document(name; force=false, service=_FS), (_, _) -> _json(200, "{}"))
    @test seen[1].target == "/v1beta/" * name
    r = _answered(() -> retrieve_file_search_document(name; service=_FS), 404;
                  body="""{"error": {"code": 404, "message": "Documents does not exist", "status": "NOT_FOUND"}}""")
    @test r isa FileSearchFailure && r.status == 404
end

# Refuted if: a 200 without a field the API always sends is read as a store or a document.
@testset "a File Search reply without a documented field is a FileSearchCallError" begin
    store = JSON.parse(_STORE)
    for key in ("name", "createTime", "updateTime")
        r = _answered(() -> retrieve_file_search_store(_STORE_NAME; service=_FS), 200; body=JSON.json(delete!(copy(store), key)))
        @test r isa FileSearchCallError && occursin(key, r.error)
    end
    r = _answered(() -> retrieve_file_search_store(_STORE_NAME; service=_FS), 200;
                  body=JSON.json(merge(store, Dict("activeDocumentsCount" => 2))))
    @test r isa FileSearchCallError                                          # an int64 is a JSON string
    doc = JSON.parse(_DOC)
    for key in ("name", "createTime")
        r = _answered(() -> retrieve_file_search_document(_STORE_NAME * "/documents/d"; service=_FS), 200;
                      body=JSON.json(delete!(copy(doc), key)))
        @test r isa FileSearchCallError && occursin(key, r.error)
    end
    r = _answered(() -> retrieve_file_search_document(_STORE_NAME * "/documents/d"; service=_FS), 200;
                  body=JSON.json(merge(doc, Dict("customMetadata" => [Dict("key" => "k")]))))
    @test r isa FileSearchCallError                                          # a metadata entry without a value
    r = _answered(() -> list_file_search_documents(_STORE_NAME; service=_FS), 200; body="""{"documents": {}}""")
    @test r isa FileSearchCallError
end

@testset "the File Search verbs — capability, cancel and config" begin
    @test has_capability(GEMINIServiceEndpoint, :file_search_stores)
    @test !has_capability(GEMINIOpenAIServiceEndpoint, :file_search_stores)
    r, seen = _with_scripted(() -> list_file_search_stores(; service=_FS, cancel=cancel!(CancelToken())),
                             (_, _) -> _json(200, "{}"))
    @test r isa FileSearchCallError && r.cause isa UniLMCancelled && isempty(seen)
    @test _seam_timeout(retrieve_file_search_store(_STORE_NAME; service=_FS, config=_TINY_DEADLINE), FileSearchCallError)
end

# The Interactions reference's FileSearch tool: `type` "file_search", `file_search_store_names`, `top_k`,
# `metadata_filter`. Refuted if: the tool takes another shape, a store is named in another form, a request sends it
# changed, or an empty store list or a top_k below 1 is sent.
@testset "gemini_file_search — the Interactions file_search tool" begin
    @test gemini_file_search("unilmwitness-5r9bfby3t61y") ==
          Dict("type" => "file_search", "file_search_store_names" => [_STORE_NAME])
    t = gemini_file_search([_STORE_NAME, "other"]; top_k=2, metadata_filter="topic = \"birds\"")
    @test t == Dict("type" => "file_search", "file_search_store_names" => [_STORE_NAME, "fileSearchStores/other"],
                    "top_k" => 2, "metadata_filter" => "topic = \"birds\"")
    body = JSON.parse(UniLM.encode_agentic(GEMINIServiceEndpoint,
        Respond(service=GEMINIServiceEndpoint, model="gemini-3.8-flash", input="Which bird?", tools=[t])))
    @test body["tools"] == [t]
    for (call, says) in ((() -> gemini_file_search(String[]), "store"),
                         (() -> gemini_file_search("a/b"), "fileSearchStores/"),
                         (() -> gemini_file_search("s"; top_k=0), "top_k"))
        e = _fs_err(call)
        @test e isa ArgumentError && occursin(says, e.msg)
    end
end
