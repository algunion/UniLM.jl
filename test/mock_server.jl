using Sockets

# ─── Mock HTTP Server Setup ───────────────────────────────────────────────────
# This mock server exists ONLY for scenarios that cannot be tested against the
# real/live server: error codes, retry exhaustion, and connection failures.

response_status = Ref{Int}(200)
response_body = Ref{String}("{}")
response_headers = Ref{Vector{Pair{String,String}}}(Pair{String,String}[])
request_body = Ref{String}("")   # captures the last request body for wire-level assertions
# Optional FIFO of (status, body) responses. When non-empty, the handler pops one entry
# per request, enabling multi-request flows (e.g. retry-then-succeed) without 30 real
# backoff sleeps. Empty by default → handler falls back to the single canned response,
# so existing tests are unaffected.
response_queue = Ref{Vector{Tuple{Int,String}}}(Tuple{Int,String}[])

tcp_server = Sockets.listen(Sockets.localhost, 0)
mock_port = Int(Sockets.getsockname(tcp_server)[2])
close(tcp_server)

mock_server = HTTP.serve!("127.0.0.1", mock_port; verbose=false) do req
    request_body[] = String(req.body)
    # Queued multi-response flow takes precedence (drains one entry per request).
    if !isempty(response_queue[])
        status, body = popfirst!(response_queue[])
        return HTTP.Response(status, ["Content-Type" => "application/json"], Vector{UInt8}(body))
    end
    # Default to application/json, but let a test override the response Content-Type.
    has_ct = any(p -> lowercase(p.first) == "content-type", response_headers[])
    headers = has_ct ? response_headers[] : vcat(["Content-Type" => "application/json"], response_headers[])
    return HTTP.Response(response_status[], headers, Vector{UInt8}(response_body[]))
end

mock_base_url = "http://127.0.0.1:$mock_port"

# ─── Test Service Endpoint ────────────────────────────────────────────────────

struct MockServiceEndpoint <: UniLM.ServiceEndpoint end
UniLM._api_base_url(::Type{MockServiceEndpoint}) = mock_base_url
UniLM.get_url(::Type{MockServiceEndpoint}, ::Chat) = mock_base_url * UniLM.CHAT_COMPLETIONS_PATH
UniLM.get_url(::Type{MockServiceEndpoint}, ::Embeddings) = mock_base_url * UniLM.EMBEDDINGS_PATH
UniLM.get_url(::Type{MockServiceEndpoint}, ::FIMCompletion) = mock_base_url * UniLM.COMPLETIONS_PATH
UniLM.auth_header(::Type{MockServiceEndpoint}) = ["Content-Type" => "application/json"]
UniLM.provider_capabilities(::Type{MockServiceEndpoint}) = Set([:chat, :responses, :embeddings, :images, :tools, :fim, :prefix_completion, :files, :vector_stores, :conversations, :moderation, :audio, :batch, :image_edits, :fine_tuning, :containers, :uploads, :video, :realtime])
UniLM.default_model(::Type{MockServiceEndpoint}) = "mock-model"
UniLM.default_embedding_model(::Type{MockServiceEndpoint}) = "mock-embedding"
UniLM.default_image_model(::Type{MockServiceEndpoint}) = "mock-image"
UniLM.default_fim_model(::Type{MockServiceEndpoint}) = "mock-fim"

# A capable-but-unreachable endpoint: has every capability, but points at a dead port so
# requests raise a connection error → exercises each module's `catch → *CallError` branch.
struct DeadEndpoint <: UniLM.ServiceEndpoint end
UniLM._api_base_url(::Type{DeadEndpoint}) = "http://127.0.0.1:1"
UniLM.auth_header(::Type{DeadEndpoint}) = ["Content-Type" => "application/json"]
UniLM.provider_capabilities(::Type{DeadEndpoint}) = Set([:files, :vector_stores, :conversations,
    :moderation, :audio, :batch, :image_edits, :fine_tuning, :containers, :uploads, :video, :realtime])

# Local-echo endpoint for exercising the Realtime WebSocket transport.
struct WSMockEndpoint <: UniLM.ServiceEndpoint end
const _ws_mock_port = let s = Sockets.listen(Sockets.localhost, 0); p = Int(Sockets.getsockname(s)[2]); close(s); p end
UniLM.auth_header(::Type{WSMockEndpoint}) = ["Authorization" => "Bearer t", "Content-Type" => "application/json"]
UniLM.provider_capabilities(::Type{WSMockEndpoint}) = Set([:realtime])
UniLM._realtime_ws_url(::Type{WSMockEndpoint}) = "ws://127.0.0.1:$_ws_mock_port"

# Helper: set error response
function set_error!(status, msg="error"; headers::Vector{Pair{String,String}}=Pair{String,String}[])
    response_status[] = status
    response_body[] = JSON.json(Dict("error" => Dict("message" => msg, "type" => "server_error")))
    response_headers[] = headers
end

try

    # ═══════════════════════════════════════════════════════════════════════
    # Chat Completions: error / retry paths
    # ═══════════════════════════════════════════════════════════════════════

    @testset "chatrequest! with 500 (retry exhausted)" begin
        set_error!(500, "Internal Server Error")

        chat = Chat(service=MockServiceEndpoint, model="gpt-4o")
        push!(chat, Message(role=UniLM.RoleSystem, content="sys"))
        push!(chat, Message(role=UniLM.RoleUser, content="usr"))

        result = chatrequest!(chat; retries=30)
        @test result isa LLMFailure
        @test result.status == 500
    end

    @testset "chatrequest! with 503 (retry exhausted)" begin
        set_error!(503, "Service Unavailable")

        chat = Chat(service=MockServiceEndpoint, model="gpt-4o")
        push!(chat, Message(role=UniLM.RoleSystem, content="sys"))
        push!(chat, Message(role=UniLM.RoleUser, content="usr"))

        result = chatrequest!(chat; retries=30)
        @test result isa LLMFailure
        @test result.status == 503
    end

    @testset "chatrequest! with 400 (non-retryable error)" begin
        set_error!(400, "Bad Request")

        chat = Chat(service=MockServiceEndpoint, model="gpt-4o")
        push!(chat, Message(role=UniLM.RoleSystem, content="sys"))
        push!(chat, Message(role=UniLM.RoleUser, content="usr"))

        result = chatrequest!(chat)
        @test result isa LLMFailure
        @test result.status == 400
    end

    # ═══════════════════════════════════════════════════════════════════════
    # Chat Completions: streaming connection error
    # ═══════════════════════════════════════════════════════════════════════

    @testset "_chatrequeststream connection error" begin
        struct DeadServiceEndpoint <: UniLM.ServiceEndpoint end
        UniLM.get_url(::Type{DeadServiceEndpoint}, ::Chat) = "http://127.0.0.1:1/v1/chat/completions"
        UniLM.auth_header(::Type{DeadServiceEndpoint}) = ["Content-Type" => "application/json"]

        chat = Chat(service=DeadServiceEndpoint, model="gpt-4o", stream=true)
        push!(chat, Message(role=UniLM.RoleSystem, content="sys"))
        push!(chat, Message(role=UniLM.RoleUser, content="usr"))

        body = JSON.json(chat)
        task = UniLM._chatrequeststream(chat, body, nothing)
        result = fetch(task)

        @test result isa LLMCallError
        @test !isempty(result.error)
    end

    # ═══════════════════════════════════════════════════════════════════════
    # Streaming error STATUS (regression): a non-200 on the streamed path must
    # surface as LLMFailure/ResponseFailure with the real status + error body.
    # Under HTTP 2.x the pull-model leaves resp.body empty, so the body must be
    # captured in-block — this guards against silently degrading to *CallError.
    # ═══════════════════════════════════════════════════════════════════════

    @testset "_chatrequeststream non-200 → LLMFailure with status + body" begin
        set_error!(400, "Bad Request")

        chat = Chat(service=MockServiceEndpoint, model="gpt-4o", stream=true)
        push!(chat, Message(role=UniLM.RoleSystem, content="sys"))
        push!(chat, Message(role=UniLM.RoleUser, content="usr"))

        body = JSON.json(chat)
        task = UniLM._chatrequeststream(chat, body, nothing)
        result = fetch(task)

        @test result isa LLMFailure
        @test result.status == 400
        @test !isempty(result.response)
        @test occursin("Bad Request", result.response)
    end

    @testset "_respond_stream non-200 → ResponseFailure with status + body" begin
        set_error!(400, "Bad Request")

        r = Respond(input="test", service=MockServiceEndpoint, stream=true)
        result = respond(r)
        @test result isa Task
        inner = fetch(result)

        @test inner isa ResponseFailure
        @test inner.status == 400
        @test !isempty(inner.response)
        @test occursin("Bad Request", inner.response)
    end

    @testset "_respond_stream 200 + streamed response.failed → structured ResponseFailure" begin
        # HTTP 200, but the SSE stream ends in response.failed — the structured error must be
        # surfaced, not silently degraded to a 200 with a raw SSE blob (the pre-fix behavior).
        response_status[] = 200
        response_headers[] = Pair{String,String}[]
        response_body[] = "event: response.created\ndata: {\"type\":\"response.created\"}\n\n" *
            "event: response.failed\ndata: {\"type\":\"response.failed\",\"response\":{\"id\":\"resp_1\",\"status\":\"failed\",\"error\":{\"code\":\"server_error\",\"message\":\"boom\"}}}\n\n"

        r = Respond(input="test", service=MockServiceEndpoint, stream=true)
        inner = fetch(respond(r))
        @test inner isa ResponseFailure
        @test occursin("server_error", inner.response)
        @test occursin("boom", inner.response)
    end

    @testset "_respond_stream 200 + completed without response key → degrades, no crash" begin
        response_status[] = 200
        response_headers[] = Pair{String,String}[]
        response_body[] = "event: response.completed\ndata: {\"type\":\"response.completed\"}\n\n"
        inner = fetch(respond(Respond(input="x", service=MockServiceEndpoint, stream=true)))
        @test inner isa ResponseFailure   # degraded safely (regression: was a KeyError → ResponseCallError)
        set_error!(200, "")
    end

    # ═══════════════════════════════════════════════════════════════════════
    # Responses API: error / retry paths
    # ═══════════════════════════════════════════════════════════════════════

    @testset "respond with 500 (retry exhausted)" begin
        set_error!(500, "Internal Server Error")

        r = Respond(input="test", service=MockServiceEndpoint)
        result = respond(r; retries=30)

        @test result isa ResponseFailure
        @test result.status == 500
    end

    @testset "respond with 503 (retry exhausted)" begin
        set_error!(503, "Service Unavailable")

        r = Respond(input="test", service=MockServiceEndpoint)
        result = respond(r; retries=30)

        @test result isa ResponseFailure
        @test result.status == 503
    end

    @testset "respond with 400 (non-retryable error)" begin
        set_error!(400, "Bad Request")

        r = Respond(input="test", service=MockServiceEndpoint)
        result = respond(r)

        @test result isa ResponseFailure
        @test result.status == 400
    end

    @testset "get_response with 404 (mock failure)" begin
        set_error!(404, "Not Found")

        result = get_response("resp_nonexistent"; service=MockServiceEndpoint)

        @test result isa ResponseFailure
        @test result.status == 404
    end

    @testset "delete_response with 404 (mock failure)" begin
        set_error!(404, "Not Found")

        result = delete_response("resp_nonexistent"; service=MockServiceEndpoint)

        @test result isa ResponseFailure
        @test result.status == 404
    end

    @testset "list_input_items with 404 (mock failure)" begin
        set_error!(404, "Not Found")

        result = list_input_items("resp_nonexistent"; service=MockServiceEndpoint)

        @test result isa ResponseFailure
        @test result.status == 404
    end

    @testset "cancel_response with 400 (mock failure)" begin
        set_error!(400, "Response not in progress")

        result = cancel_response("resp_not_in_progress"; service=MockServiceEndpoint)

        @test result isa ResponseFailure
        @test result.status == 400
    end

    @testset "compact_response with 400 (mock failure)" begin
        set_error!(400, "Invalid input")

        result = compact_response(model="gpt-5.2", input="test"; service=MockServiceEndpoint)

        @test result isa ResponseFailure
        @test result.status == 400
    end

    @testset "count_input_tokens with 400 (mock failure)" begin
        set_error!(400, "Invalid model")

        result = count_input_tokens(model="invalid", input="test"; service=MockServiceEndpoint)

        @test result isa ResponseFailure
        @test result.status == 400
    end

    # ═══════════════════════════════════════════════════════════════════════
    # Image Generation: error / retry paths
    # ═══════════════════════════════════════════════════════════════════════

    @testset "generate_image with 500 (retry exhausted)" begin
        set_error!(500, "Internal Server Error")

        ig = ImageGeneration(prompt="test", service=MockServiceEndpoint)
        result = generate_image(ig; retries=30)

        @test result isa ImageFailure
        @test result.status == 500
    end

    @testset "generate_image with 503 (retry exhausted)" begin
        set_error!(503, "Service Unavailable")

        ig = ImageGeneration(prompt="test", service=MockServiceEndpoint)
        result = generate_image(ig; retries=30)

        @test result isa ImageFailure
        @test result.status == 503
    end

    @testset "generate_image with 400 (non-retryable error)" begin
        set_error!(400, "Bad Request")

        ig = ImageGeneration(prompt="test", service=MockServiceEndpoint)
        result = generate_image(ig)

        @test result isa ImageFailure
        @test result.status == 400
    end

    # ═══════════════════════════════════════════════════════════════════════
    # 429 Rate Limit handling
    # ═══════════════════════════════════════════════════════════════════════

    @testset "chatrequest! with 429 (retry exhausted)" begin
        set_error!(429, "Rate limited"; headers=["Retry-After" => "2"])

        chat = Chat(service=MockServiceEndpoint, model="gpt-4o")
        push!(chat, Message(role=UniLM.RoleSystem, content="sys"))
        push!(chat, Message(role=UniLM.RoleUser, content="usr"))

        result = chatrequest!(chat; retries=30)
        @test result isa LLMFailure
        @test result.status == 429
    end

    @testset "respond with 429 (retry exhausted)" begin
        set_error!(429, "Rate limited"; headers=["Retry-After" => "2"])

        r = Respond(input="test", service=MockServiceEndpoint)
        result = respond(r; retries=30)

        @test result isa ResponseFailure
        @test result.status == 429
    end

    @testset "generate_image with 429 (retry exhausted)" begin
        set_error!(429, "Rate limited"; headers=["Retry-After" => "2"])

        ig = ImageGeneration(prompt="test", service=MockServiceEndpoint)
        result = generate_image(ig; retries=30)

        @test result isa ImageFailure
        @test result.status == 429
    end

    @testset "embeddingrequest! with 429 (retry exhausted)" begin
        set_error!(429, "Rate limited"; headers=["Retry-After" => "2"])

        emb = UniLM.Embeddings("test"; service=MockServiceEndpoint)
        result = embeddingrequest!(emb; retries=30)
        @test result isa EmbeddingFailure
        @test result.status == 429

        # Reset mock state so subsequent tests don't inherit 429 + Retry-After
        set_error!(200, "")
    end

    # ═══════════════════════════════════════════════════════════════════════
    # Embeddings: error paths
    # ═══════════════════════════════════════════════════════════════════════

    @testset "embeddingrequest! with non-200 error status" begin
        set_error!(401, "Unauthorized")
        emb = UniLM.Embeddings("test embedding"; service=MockServiceEndpoint)
        result = embeddingrequest!(emb)
        @test result isa EmbeddingFailure
        @test result.status == 401
    end

    @testset "embeddingrequest! with 500 (retry exhausted)" begin
        set_error!(500, "Internal Server Error")
        emb = UniLM.Embeddings("test"; service=MockServiceEndpoint)
        result = embeddingrequest!(emb; retries=30)
        @test result isa EmbeddingFailure
        @test result.status == 500
    end

    @testset "embeddingrequest! with 503 (retry exhausted)" begin
        set_error!(503, "Service Unavailable")
        emb = UniLM.Embeddings("test"; service=MockServiceEndpoint)
        result = embeddingrequest!(emb; retries=30)
        @test result isa EmbeddingFailure
        @test result.status == 503
    end

    @testset "embeddingrequest! catch block (connection error)" begin
        dead = UniLM.GenericOpenAIEndpoint("http://127.0.0.1:1", "")
        emb = UniLM.Embeddings("test"; service=dead, model="test-embed")
        result = embeddingrequest!(emb)
        @test result isa EmbeddingCallError
    end

    @testset "Files API (mock)" begin
        response_status[] = 200
        response_headers[] = Pair{String,String}[]
        response_body[] = JSON.json(Dict("id" => "file-1", "bytes" => 5, "created_at" => 1,
            "filename" => "a.txt", "purpose" => "user_data", "status" => "processed"))
        path = tempname() * ".txt"
        write(path, "hello")
        r = upload_file(path, "user_data"; service=MockServiceEndpoint)
        @test r isa FileSuccess
        @test r.response.id == "file-1"
        @test r.response.filename == "a.txt"
        rm(path)

        response_body[] = JSON.json(Dict("data" => [Dict("id" => "file-1", "bytes" => 5,
            "created_at" => 1, "filename" => "a.txt", "purpose" => "user_data")], "has_more" => false))
        rl = list_files(service=MockServiceEndpoint)
        @test rl isa FileListSuccess
        @test length(rl.response.data) == 1
        @test rl.response.data[1].id == "file-1"

        response_body[] = JSON.json(Dict("id" => "file-1", "object" => "file", "deleted" => true))
        rd = delete_file("file-1"; service=MockServiceEndpoint)
        @test rd isa FileDeleteSuccess
        @test rd.deleted

        response_body[] = "rawbytes"
        rc = file_content("file-1"; service=MockServiceEndpoint)
        @test rc isa FileContentSuccess
        @test String(rc.content) == "rawbytes"

        set_error!(404, "not found")
        @test retrieve_file("file-x"; service=MockServiceEndpoint) isa FileFailure
        set_error!(200, "")
    end

    @testset "Vector Stores API (mock)" begin
        response_status[] = 200
        response_headers[] = Pair{String,String}[]
        response_body[] = JSON.json(Dict("id" => "vs-1", "name" => "docs", "status" => "completed",
            "file_counts" => Dict("total" => 0)))
        r = create_vector_store(name="docs", service=MockServiceEndpoint)
        @test r isa VectorStoreSuccess
        @test vector_store_id(r.response) == "vs-1"

        response_body[] = JSON.json(Dict("data" => [Dict("id" => "vs-1", "name" => "docs")], "has_more" => false))
        rl = list_vector_stores(service=MockServiceEndpoint)
        @test rl isa VectorStoreListSuccess && length(rl.response.data) == 1

        response_body[] = JSON.json(Dict("id" => "vsf-1", "status" => "completed"))
        rf = add_vector_store_file("vs-1", "file-1"; service=MockServiceEndpoint)
        @test rf isa VectorStoreFileSuccess && rf.response.id == "vsf-1"

        response_body[] = JSON.json(Dict("id" => "batch-1", "status" => "completed", "file_counts" => Dict("completed" => 1)))
        @test create_file_batch("vs-1", ["file-1"]; service=MockServiceEndpoint) isa VectorStoreBatchSuccess
        rp = poll_file_batch("vs-1", "batch-1"; interval=0.01, timeout=1.0, service=MockServiceEndpoint)
        @test rp isa VectorStoreBatchSuccess && rp.response.status == "completed"

        response_body[] = JSON.json(Dict("id" => "vs-1", "deleted" => true))
        rd = delete_vector_store("vs-1"; service=MockServiceEndpoint)
        @test rd isa VectorStoreDeleteSuccess && rd.deleted

        @test_throws ArgumentError create_vector_store(service=UniLM.GEMINIServiceEndpoint)

        set_error!(404, "nope")
        @test retrieve_vector_store("vs-x"; service=MockServiceEndpoint) isa VectorStoreFailure
        set_error!(200, "")
    end

    @testset "Conversations API (mock)" begin
        response_status[] = 200
        response_headers[] = Pair{String,String}[]
        response_body[] = JSON.json(Dict("id" => "conv-1", "created_at" => 1, "metadata" => Dict("k" => "v")))
        r = create_conversation(metadata=Dict("k" => "v"), service=MockServiceEndpoint)
        @test r isa ConversationSuccess
        @test conversation_id(r.response) == "conv-1"

        response_body[] = JSON.json(Dict("data" => [Dict("id" => "item-1", "type" => "message")],
            "has_more" => false, "first_id" => "item-1", "last_id" => "item-1"))
        ra = add_conversation_items("conv-1", [Dict("type" => "message", "role" => "user", "content" => "hi")]; service=MockServiceEndpoint)
        @test ra isa ConversationItemListSuccess && length(ra.response.data) == 1
        @test list_conversation_items("conv-1"; service=MockServiceEndpoint) isa ConversationItemListSuccess

        response_body[] = JSON.json(Dict("id" => "conv-1", "deleted" => true))
        rd = delete_conversation("conv-1"; service=MockServiceEndpoint)
        @test rd isa ConversationDeleteSuccess && rd.deleted

        set_error!(404, "nope")
        @test retrieve_conversation("conv-x"; service=MockServiceEndpoint) isa ConversationFailure
        set_error!(200, "")
    end

    @testset "Moderations API (mock)" begin
        response_status[] = 200
        response_headers[] = Pair{String,String}[]
        response_body[] = JSON.json(Dict("model" => "omni-moderation-latest", "results" => [
            Dict("flagged" => true, "categories" => Dict("violence" => true),
                 "category_scores" => Dict("violence" => 0.9))]))
        r = moderate("something"; service=MockServiceEndpoint)
        @test r isa ModerationSuccess
        @test is_flagged(r)
        @test r.response.results[1].category_scores["violence"] == 0.9
        set_error!(400, "bad")
        @test moderate("x"; service=MockServiceEndpoint) isa ModerationFailure
        set_error!(200, "")
    end

    @testset "Audio API (mock)" begin
        # TTS: binary response with a real audio Content-Type (mock now honors it)
        response_status[] = 200
        response_headers[] = ["Content-Type" => "audio/mpeg"]
        response_body[] = "AUDIOBYTES"
        r = speak("hello"; voice="verse", service=MockServiceEndpoint)
        @test r isa SpeechSuccess
        @test r.content_type == "audio/mpeg"
        let sent = JSON.parse(request_body[])          # speak sends JSON
            @test sent["input"] == "hello" && sent["voice"] == "verse"
        end
        p = tempname() * ".mp3"
        save_audio(r, p)                          # save BEFORE String(): String(::Vector{UInt8}) empties the buffer
        @test read(p, String) == "AUDIOBYTES"
        rm(p)

        # Transcription: JSON-response branch + multipart request shape
        response_headers[] = ["Content-Type" => "application/json"]
        response_body[] = JSON.json(Dict("text" => "hello world"))
        af = tempname() * ".wav"
        write(af, "fakeaudio")
        rt = transcribe(af; service=MockServiceEndpoint)
        @test rt isa TranscriptionSuccess && transcript_text(rt) == "hello world"
        @test occursin("name=\"model\"", request_body[]) && occursin("name=\"file\"", request_body[])  # multipart wire shape

        # Transcription: text/plain-response branch (non-JSON path) via translate()
        response_headers[] = ["Content-Type" => "text/plain"]
        response_body[] = "plain transcript"
        @test transcript_text(translate(af; service=MockServiceEndpoint)) == "plain transcript"
        rm(af)

        response_headers[] = Pair{String,String}[]
        set_error!(401, "no")
        @test speak("x"; service=MockServiceEndpoint) isa AudioFailure
        set_error!(200, "")
    end

    @testset "Batch API (mock)" begin
        response_status[] = 200
        response_headers[] = Pair{String,String}[]
        response_body[] = JSON.json(Dict("id" => "batch-1", "status" => "completed",
            "endpoint" => "/v1/chat/completions", "output_file_id" => "file-out",
            "request_counts" => Dict("completed" => 2, "total" => 2)))
        r = create_batch("file-in", "/v1/chat/completions"; service=MockServiceEndpoint)
        @test r isa BatchSuccess
        @test r.response.output_file_id == "file-out"
        rp = poll_batch("batch-1"; interval=0.01, timeout=1.0, service=MockServiceEndpoint)
        @test rp isa BatchSuccess && rp.response.status == "completed"
        response_body[] = JSON.json(Dict("data" => [Dict("id" => "batch-1", "status" => "completed")], "has_more" => false))
        @test list_batches(service=MockServiceEndpoint) isa BatchListSuccess
        set_error!(404, "no")
        @test retrieve_batch("batch-x"; service=MockServiceEndpoint) isa BatchFailure
        set_error!(200, "")
    end

    @testset "Image edits (mock)" begin
        response_status[] = 200
        response_headers[] = Pair{String,String}[]
        response_body[] = JSON.json(Dict("created" => 1, "data" => [Dict("b64_json" => "EDITED")]))
        img = tempname() * ".png"
        write(img, "fakepng")
        r = edit_image(img, "make it blue"; service=MockServiceEndpoint)
        @test r isa ImageSuccess
        @test image_data(r) == ["EDITED"]
        rm(img)
        # missing file is caught inside the request → ImageCallError
        @test edit_image("/no/such.png", "x"; service=MockServiceEndpoint) isa ImageCallError
        set_error!(400, "bad")
        img2 = tempname() * ".png"
        write(img2, "fakepng")
        @test edit_image(img2, "x"; service=MockServiceEndpoint) isa ImageFailure
        rm(img2)
        set_error!(200, "")
    end

    @testset "Fine-tuning API (mock)" begin
        response_status[] = 200; response_headers[] = Pair{String,String}[]
        response_body[] = JSON.json(Dict("id" => "ftjob-1", "status" => "running", "model" => "gpt-4.1-mini"))
        r = create_fine_tuning_job(model="gpt-4.1-mini", training_file="file-1", service=MockServiceEndpoint)
        @test r isa FineTuningSuccess && r.response.id == "ftjob-1"
        @test retrieve_fine_tuning_job("ftjob-1"; service=MockServiceEndpoint) isa FineTuningSuccess
        response_body[] = JSON.json(Dict("data" => [Dict("id" => "evt-1")], "has_more" => false))
        @test list_fine_tuning_events("ftjob-1"; service=MockServiceEndpoint) isa FineTuningListSuccess
        set_error!(404, "no")
        @test retrieve_fine_tuning_job("x"; service=MockServiceEndpoint) isa FineTuningFailure
        set_error!(200, "")
    end

    @testset "Containers API (mock)" begin
        response_status[] = 200; response_headers[] = Pair{String,String}[]
        response_body[] = JSON.json(Dict("id" => "cntr-1", "status" => "running", "name" => "box"))
        @test create_container(name="box", service=MockServiceEndpoint) isa ContainerSuccess
        response_body[] = JSON.json(Dict("id" => "cfile-1", "status" => "ok"))
        p = tempname() * ".txt"; write(p, "x")
        @test add_container_file("cntr-1", p; service=MockServiceEndpoint) isa ContainerSuccess
        rm(p)
        response_body[] = JSON.json(Dict("id" => "cntr-1", "deleted" => true))
        @test delete_container("cntr-1"; service=MockServiceEndpoint) isa ContainerDeleteSuccess
        set_error!(404, "no")
        @test retrieve_container("x"; service=MockServiceEndpoint) isa ContainerFailure
        set_error!(200, "")
    end

    @testset "Uploads API (mock)" begin
        response_status[] = 200; response_headers[] = Pair{String,String}[]
        response_body[] = JSON.json(Dict("id" => "upload-1", "status" => "pending", "filename" => "big.bin", "bytes" => 10))
        @test create_upload(filename="big.bin", purpose="batch", bytes=10, mime_type="application/octet-stream", service=MockServiceEndpoint) isa UploadSuccess
        response_body[] = JSON.json(Dict("id" => "part-1"))
        @test add_upload_part("upload-1", Vector{UInt8}("chunk"); service=MockServiceEndpoint) isa UploadPartSuccess
        response_body[] = JSON.json(Dict("id" => "upload-1", "status" => "completed",
            "file" => Dict("id" => "file-9", "bytes" => 10, "created_at" => 1, "filename" => "big.bin", "purpose" => "batch")))
        rc = complete_upload("upload-1", ["part-1"]; service=MockServiceEndpoint)
        @test rc isa UploadSuccess && !isnothing(rc.response.file) && rc.response.file.id == "file-9"
        set_error!(400, "no")
        @test cancel_upload("upload-1"; service=MockServiceEndpoint) isa UploadFailure
        set_error!(200, "")
    end

    @testset "Videos API (mock)" begin
        response_status[] = 200; response_headers[] = Pair{String,String}[]
        response_body[] = JSON.json(Dict("id" => "video-1", "status" => "queued", "model" => "sora-2"))
        @test create_video(prompt="a cat", service=MockServiceEndpoint) isa VideoSuccess
        response_body[] = "VIDEOBYTES"
        rc = video_content("video-1"; service=MockServiceEndpoint)
        @test rc isa VideoContentSuccess && String(copy(rc.content)) == "VIDEOBYTES"
        set_error!(404, "no")
        @test retrieve_video("x"; service=MockServiceEndpoint) isa VideoFailure
        set_error!(200, "")
    end

    @testset "Realtime API (mock secret + event builders)" begin
        response_status[] = 200; response_headers[] = Pair{String,String}[]
        response_body[] = JSON.json(Dict("value" => "ek_123", "expires_at" => 1))
        r = mint_realtime_secret(service=MockServiceEndpoint)
        @test r isa RealtimeSecretSuccess && r.value == "ek_123"
        response_body[] = JSON.json(Dict("client_secret" => Dict("value" => "ek_456")))
        @test mint_realtime_secret(service=MockServiceEndpoint).value == "ek_456"
        @test session_update(Dict("voice" => "alloy"))[:type] == "session.update"
        @test input_audio_append("BASE64")[:audio] == "BASE64"
        @test response_create()[:type] == "response.create"
        @test realtime_event("custom"; foo="bar")[:foo] == "bar"
        set_error!(401, "no")
        @test mint_realtime_secret(service=MockServiceEndpoint) isa RealtimeFailure
        set_error!(200, "")
    end

    @testset "Realtime WebSocket transport (local echo)" begin
        srv = HTTP.WebSockets.listen!("127.0.0.1", _ws_mock_port) do ws
            for msg in ws
                HTTP.WebSockets.send(ws, msg)   # echo back
            end
        end
        try
            got = Ref{Any}(nothing)
            realtime_connect(model="gpt-realtime-2", service=WSMockEndpoint) do sess
                realtime_send(sess, session_update(Dict("voice" => "alloy")))
                got[] = realtime_receive(sess)
            end
            @test got[]["type"] == "session.update"
            @test got[]["session"]["voice"] == "alloy"
        finally
            close(srv)
        end
    end

    @testset "review-remediation coverage (wire format, untested fns, poll timeout)" begin
        response_status[] = 200
        response_headers[] = Pair{String,String}[]

        # JSON request-body wire-format assertions (mock captures request_body[])
        response_body[] = JSON.json(Dict("id" => "batch-9", "status" => "validating", "endpoint" => "/v1/responses"))
        create_batch("file-in", "/v1/responses"; completion_window="24h", service=MockServiceEndpoint)
        let b = JSON.parse(request_body[])
            @test b["input_file_id"] == "file-in" && b["endpoint"] == "/v1/responses" && b["completion_window"] == "24h"
        end
        response_body[] = JSON.json(Dict("id" => "ftjob-9", "status" => "queued"))
        create_fine_tuning_job(model="gpt-4.1-mini", training_file="file-t"; service=MockServiceEndpoint)
        let b = JSON.parse(request_body[])
            @test b["model"] == "gpt-4.1-mini" && b["training_file"] == "file-t"
        end
        response_body[] = JSON.json(Dict("id" => "upload-9", "status" => "pending"))
        create_upload(filename="big.bin", purpose="batch", bytes=123, mime_type="application/octet-stream", service=MockServiceEndpoint)
        let b = JSON.parse(request_body[])
            @test b["filename"] == "big.bin" && b["bytes"] == 123 && b["mime_type"] == "application/octet-stream"
        end
        # multipart wire shape for upload_file
        response_body[] = JSON.json(Dict("id" => "file-9", "bytes" => 1, "created_at" => 1, "filename" => "u.txt", "purpose" => "user_data"))
        pth = tempname() * ".txt"; write(pth, "u")
        upload_file(pth, "user_data"; service=MockServiceEndpoint)
        @test occursin("name=\"purpose\"", request_body[]) && occursin("name=\"file\"", request_body[]) && occursin("filename=", request_body[])
        rm(pth)

        # previously-untested exported functions (smoke)
        response_body[] = JSON.json(Dict("data" => [Dict("id" => "x")], "has_more" => false))
        @test list_fine_tuning_jobs(service=MockServiceEndpoint) isa FineTuningListSuccess
        @test list_fine_tuning_checkpoints("ftjob-1"; service=MockServiceEndpoint) isa FineTuningListSuccess
        @test list_containers(service=MockServiceEndpoint) isa ContainerListSuccess
        @test list_videos(service=MockServiceEndpoint) isa VideoListSuccess
        response_body[] = JSON.json(Dict("id" => "batch-1", "status" => "cancelling"))
        @test cancel_batch("batch-1"; service=MockServiceEndpoint) isa BatchSuccess
        response_body[] = JSON.json(Dict("id" => "ftjob-1", "status" => "cancelled"))
        @test cancel_fine_tuning_job("ftjob-1"; service=MockServiceEndpoint) isa FineTuningSuccess
        response_body[] = JSON.json(Dict("id" => "batch-1", "status" => "completed", "file_counts" => Dict("completed" => 1)))
        @test retrieve_file_batch("vs-1", "batch-1"; service=MockServiceEndpoint) isa VectorStoreBatchSuccess
        response_body[] = JSON.json(Dict("id" => "conv-1", "metadata" => Dict("k" => "v2")))
        @test update_conversation("conv-1", Dict("k" => "v2"); service=MockServiceEndpoint) isa ConversationSuccess
        response_body[] = JSON.json(Dict("id" => "item-1", "deleted" => true))
        @test delete_conversation_item("conv-1", "item-1"; service=MockServiceEndpoint) isa ConversationDeleteSuccess
        # save_file_content round-trip
        response_body[] = "filebytes"
        fc = file_content("file-1"; service=MockServiceEndpoint)
        fp = tempname(); save_file_content(fc, fp); @test read(fp, String) == "filebytes"; rm(fp)

        # poll loop continuation + timeout (status never terminal → CallError)
        response_body[] = JSON.json(Dict("id" => "batch-1", "status" => "in_progress"))
        @test poll_batch("batch-1"; interval=0.01, timeout=0.03, service=MockServiceEndpoint) isa BatchCallError
        @test poll_file_batch("vs-1", "batch-1"; interval=0.01, timeout=0.03, service=MockServiceEndpoint) isa VectorStoreCallError

        # moderations is_flagged is failure-safe
        @test is_flagged(ModerationFailure(response="x", status=500)) == false
        @test is_flagged(ModerationCallError(error="x")) == false

        set_error!(200, "")
    end

    @testset "every endpoint *CallError branch (dead endpoint)" begin
        tf = tempname() * ".txt"; write(tf, "x")
        png = tempname() * ".png"; write(png, "p")
        # Files
        @test upload_file(tf, "user_data"; service=DeadEndpoint) isa FileCallError
        @test list_files(service=DeadEndpoint) isa FileCallError
        @test retrieve_file("f"; service=DeadEndpoint) isa FileCallError
        @test delete_file("f"; service=DeadEndpoint) isa FileCallError
        @test file_content("f"; service=DeadEndpoint) isa FileCallError
        # Vector stores
        @test create_vector_store(name="v", service=DeadEndpoint) isa VectorStoreCallError
        @test retrieve_vector_store("v"; service=DeadEndpoint) isa VectorStoreCallError
        @test list_vector_stores(service=DeadEndpoint) isa VectorStoreCallError
        @test delete_vector_store("v"; service=DeadEndpoint) isa VectorStoreCallError
        @test add_vector_store_file("v", "f"; service=DeadEndpoint) isa VectorStoreCallError
        @test create_file_batch("v", ["f"]; service=DeadEndpoint) isa VectorStoreCallError
        @test retrieve_file_batch("v", "b"; service=DeadEndpoint) isa VectorStoreCallError
        # Conversations
        @test create_conversation(service=DeadEndpoint) isa ConversationCallError
        @test retrieve_conversation("c"; service=DeadEndpoint) isa ConversationCallError
        @test update_conversation("c", Dict("k" => "v"); service=DeadEndpoint) isa ConversationCallError
        @test delete_conversation("c"; service=DeadEndpoint) isa ConversationCallError
        @test add_conversation_items("c", [Dict("type" => "message")]; service=DeadEndpoint) isa ConversationCallError
        @test list_conversation_items("c"; service=DeadEndpoint) isa ConversationCallError
        @test delete_conversation_item("c", "i"; service=DeadEndpoint) isa ConversationCallError
        # Moderations / Audio / Image edits
        @test moderate("x"; service=DeadEndpoint) isa ModerationCallError
        @test speak("hi"; service=DeadEndpoint) isa AudioCallError
        @test transcribe(tf; service=DeadEndpoint) isa AudioCallError
        @test translate(tf; service=DeadEndpoint) isa AudioCallError
        @test edit_image(png, "p"; model="gpt-image-2", service=DeadEndpoint) isa ImageCallError
        # Batch
        @test create_batch("f", "/v1/responses"; service=DeadEndpoint) isa BatchCallError
        @test retrieve_batch("b"; service=DeadEndpoint) isa BatchCallError
        @test cancel_batch("b"; service=DeadEndpoint) isa BatchCallError
        @test list_batches(service=DeadEndpoint) isa BatchCallError
        # Fine-tuning
        @test create_fine_tuning_job(model="m", training_file="f", service=DeadEndpoint) isa FineTuningCallError
        @test retrieve_fine_tuning_job("j"; service=DeadEndpoint) isa FineTuningCallError
        @test cancel_fine_tuning_job("j"; service=DeadEndpoint) isa FineTuningCallError
        @test list_fine_tuning_jobs(service=DeadEndpoint) isa FineTuningCallError
        @test list_fine_tuning_events("j"; service=DeadEndpoint) isa FineTuningCallError
        @test list_fine_tuning_checkpoints("j"; service=DeadEndpoint) isa FineTuningCallError
        # Containers
        @test create_container(name="c", service=DeadEndpoint) isa ContainerCallError
        @test retrieve_container("c"; service=DeadEndpoint) isa ContainerCallError
        @test list_containers(service=DeadEndpoint) isa ContainerCallError
        @test delete_container("c"; service=DeadEndpoint) isa ContainerCallError
        @test add_container_file("c", tf; service=DeadEndpoint) isa ContainerCallError
        # Uploads
        @test create_upload(filename="f", purpose="batch", bytes=1, mime_type="text/plain", service=DeadEndpoint) isa UploadCallError
        @test add_upload_part("u", Vector{UInt8}("x"); service=DeadEndpoint) isa UploadCallError
        @test complete_upload("u", ["p"]; service=DeadEndpoint) isa UploadCallError
        @test cancel_upload("u"; service=DeadEndpoint) isa UploadCallError
        # Videos / Realtime
        @test create_video(prompt="p", service=DeadEndpoint) isa VideoCallError
        @test retrieve_video("v"; service=DeadEndpoint) isa VideoCallError
        @test list_videos(service=DeadEndpoint) isa VideoCallError
        @test video_content("v"; service=DeadEndpoint) isa VideoCallError
        @test mint_realtime_secret(service=DeadEndpoint) isa RealtimeCallError
        rm(tf); rm(png)
    end

    @testset "every endpoint *Failure branch (non-200)" begin
        set_error!(400, "bad")   # 400 is non-retryable → immediate *Failure
        tf = tempname() * ".txt"; write(tf, "x")
        png = tempname() * ".png"; write(png, "p")
        @test upload_file(tf, "user_data"; service=MockServiceEndpoint) isa FileFailure
        @test list_files(service=MockServiceEndpoint) isa FileFailure
        @test delete_file("f"; service=MockServiceEndpoint) isa FileFailure
        @test file_content("f"; service=MockServiceEndpoint) isa FileFailure
        @test create_vector_store(name="v", service=MockServiceEndpoint) isa VectorStoreFailure
        @test list_vector_stores(service=MockServiceEndpoint) isa VectorStoreFailure
        @test delete_vector_store("v"; service=MockServiceEndpoint) isa VectorStoreFailure
        @test add_vector_store_file("v", "f"; service=MockServiceEndpoint) isa VectorStoreFailure
        @test create_file_batch("v", ["f"]; service=MockServiceEndpoint) isa VectorStoreFailure
        @test retrieve_file_batch("v", "b"; service=MockServiceEndpoint) isa VectorStoreFailure
        @test create_conversation(service=MockServiceEndpoint) isa ConversationFailure
        @test update_conversation("c", Dict("k" => "v"); service=MockServiceEndpoint) isa ConversationFailure
        @test delete_conversation("c"; service=MockServiceEndpoint) isa ConversationFailure
        @test add_conversation_items("c", [Dict("type" => "message")]; service=MockServiceEndpoint) isa ConversationFailure
        @test list_conversation_items("c"; service=MockServiceEndpoint) isa ConversationFailure
        @test delete_conversation_item("c", "i"; service=MockServiceEndpoint) isa ConversationFailure
        @test moderate("x"; service=MockServiceEndpoint) isa ModerationFailure
        @test speak("hi"; service=MockServiceEndpoint) isa AudioFailure
        @test transcribe(tf; service=MockServiceEndpoint) isa AudioFailure
        @test translate(tf; service=MockServiceEndpoint) isa AudioFailure
        @test edit_image(png, "p"; model="gpt-image-2", service=MockServiceEndpoint) isa ImageFailure
        @test create_batch("f", "/v1/responses"; service=MockServiceEndpoint) isa BatchFailure
        @test cancel_batch("b"; service=MockServiceEndpoint) isa BatchFailure
        @test list_batches(service=MockServiceEndpoint) isa BatchFailure
        @test create_fine_tuning_job(model="m", training_file="f", service=MockServiceEndpoint) isa FineTuningFailure
        @test cancel_fine_tuning_job("j"; service=MockServiceEndpoint) isa FineTuningFailure
        @test list_fine_tuning_jobs(service=MockServiceEndpoint) isa FineTuningFailure
        @test list_fine_tuning_events("j"; service=MockServiceEndpoint) isa FineTuningFailure
        @test create_container(name="c", service=MockServiceEndpoint) isa ContainerFailure
        @test list_containers(service=MockServiceEndpoint) isa ContainerFailure
        @test delete_container("c"; service=MockServiceEndpoint) isa ContainerFailure
        @test add_container_file("c", tf; service=MockServiceEndpoint) isa ContainerFailure
        @test create_upload(filename="f", purpose="batch", bytes=1, mime_type="text/plain", service=MockServiceEndpoint) isa UploadFailure
        @test add_upload_part("u", Vector{UInt8}("x"); service=MockServiceEndpoint) isa UploadFailure
        @test complete_upload("u", ["p"]; service=MockServiceEndpoint) isa UploadFailure
        @test cancel_upload("u"; service=MockServiceEndpoint) isa UploadFailure
        @test create_video(prompt="p", service=MockServiceEndpoint) isa VideoFailure
        @test list_videos(service=MockServiceEndpoint) isa VideoFailure
        @test video_content("v"; service=MockServiceEndpoint) isa VideoFailure
        @test mint_realtime_secret(service=MockServiceEndpoint) isa RealtimeFailure
        rm(tf); rm(png)
        set_error!(200, "")
    end

    # ═══════════════════════════════════════════════════════════════════════
    # Accessor helpers on failure types
    # ═══════════════════════════════════════════════════════════════════════

    @testset "image_data on failure types" begin
        @test image_data(ImageFailure(response="err", status=400)) == String[]
        @test image_data(ImageCallError(error="err")) == String[]
    end

    @testset "output_text on error types" begin
        @test output_text(ResponseFailure(response="body", status=400)) == "Error (HTTP 400): body"
        @test output_text(ResponseCallError(error="timeout")) == "Error: timeout"
    end

    @testset "function_calls on error types" begin
        @test function_calls(ResponseFailure(response="body", status=400)) == Dict{String,Any}[]
        @test function_calls(ResponseCallError(error="timeout")) == Dict{String,Any}[]
    end

    # ═══════════════════════════════════════════════════════════════════════
    # Catch-block tests: connection errors (dead server)
    # Cannot be tested against a live server — requires unreachable endpoint.
    # ═══════════════════════════════════════════════════════════════════════

    @testset "chatrequest! catch block (connection error)" begin
        struct ChatDeadEndpoint <: UniLM.ServiceEndpoint end
        UniLM.get_url(::Type{ChatDeadEndpoint}, ::Chat) = "http://127.0.0.1:1/v1/chat/completions"
        UniLM.auth_header(::Type{ChatDeadEndpoint}) = ["Content-Type" => "application/json"]

        chat = Chat(service=ChatDeadEndpoint, model="gpt-4o")
        push!(chat, Message(role=UniLM.RoleSystem, content="sys"))
        push!(chat, Message(role=UniLM.RoleUser, content="usr"))

        result = chatrequest!(chat)
        @test result isa LLMCallError
        @test !isempty(result.error)
    end

    @testset "respond catch block (connection error)" begin
        struct RespondDeadEndpoint <: UniLM.ServiceEndpoint end
        UniLM._api_base_url(::Type{RespondDeadEndpoint}) = "http://127.0.0.1:1"
        UniLM.auth_header(::Type{RespondDeadEndpoint}) = ["Content-Type" => "application/json"]
        UniLM.default_model(::Type{RespondDeadEndpoint}) = "dead-model"

        r = Respond(input="test", service=RespondDeadEndpoint)
        result = respond(r)

        @test result isa ResponseCallError
        @test !isempty(result.error)
    end

    @testset "get_response catch block (connection error)" begin
        result = get_response("resp_x"; service=RespondDeadEndpoint)
        @test result isa ResponseCallError
        @test !isempty(result.error)
    end

    @testset "delete_response catch block (connection error)" begin
        result = delete_response("resp_x"; service=RespondDeadEndpoint)
        @test result isa ResponseCallError
        @test !isempty(result.error)
    end

    @testset "list_input_items catch block (connection error)" begin
        result = list_input_items("resp_x"; service=RespondDeadEndpoint)
        @test result isa ResponseCallError
        @test !isempty(result.error)
    end

    @testset "cancel_response catch block (connection error)" begin
        result = cancel_response("resp_x"; service=RespondDeadEndpoint)
        @test result isa ResponseCallError
        @test !isempty(result.error)
    end

    @testset "compact_response catch block (connection error)" begin
        result = compact_response(model="gpt-5.2", input="test"; service=RespondDeadEndpoint)
        @test result isa ResponseCallError
        @test !isempty(result.error)
    end

    @testset "count_input_tokens catch block (connection error)" begin
        result = count_input_tokens(model="gpt-5.2", input="test"; service=RespondDeadEndpoint)
        @test result isa ResponseCallError
        @test !isempty(result.error)
    end

    @testset "generate_image catch block (connection error)" begin
        struct ImageDeadEndpoint <: UniLM.ServiceEndpoint end
        UniLM._api_base_url(::Type{ImageDeadEndpoint}) = "http://127.0.0.1:1"
        UniLM.auth_header(::Type{ImageDeadEndpoint}) = ["Content-Type" => "application/json"]
        UniLM.default_image_model(::Type{ImageDeadEndpoint}) = "dead-image"

        ig = ImageGeneration(prompt="test", service=ImageDeadEndpoint)
        result = generate_image(ig)

        @test result isa ImageCallError
        @test !isempty(result.error)
    end

    @testset "_respond_stream catch block (connection error)" begin
        r = Respond(input="test", service=RespondDeadEndpoint, stream=true)
        result = respond(r)
        @test result isa Task
        inner = fetch(result)
        @test inner isa ResponseCallError
        @test !isempty(inner.error)
    end

    # ═══════════════════════════════════════════════════════════════════════
    # Tool Loop: Chat Completions
    # ═══════════════════════════════════════════════════════════════════════

    @testset "tool_loop! two-turn cycle (Chat)" begin
        response_status[] = 200

        # First response: tool call
        response_body[] = JSON.json(Dict(
            "choices" => [Dict(
                "finish_reason" => "tool_calls",
                "message" => Dict(
                    "role" => "assistant",
                    "tool_calls" => [Dict(
                        "id" => "call_123",
                        "type" => "function",
                        "function" => Dict(
                            "name" => "add",
                            "arguments" => """{"a": 3, "b": 5}"""
                        )
                    )]
                )
            )],
            "usage" => Dict("prompt_tokens" => 10, "completion_tokens" => 5, "total_tokens" => 15)
        ))

        tool = GPTTool(func=GPTFunctionSignature(
            name="add", description="Add two numbers",
            parameters=Dict("type"=>"object",
                "properties"=>Dict("a"=>Dict("type"=>"number"),"b"=>Dict("type"=>"number")),
                "required"=>["a","b"])))

        chat = Chat(service=MockServiceEndpoint, model="gpt-4o", tools=[tool])
        push!(chat, Message(role=UniLM.RoleSystem, content="You are a calculator"))
        push!(chat, Message(role=UniLM.RoleUser, content="What is 3+5?"))

        dispatcher = (name, args) -> begin
            # Swap to text response for next API call
            response_body[] = JSON.json(Dict(
                "choices" => [Dict(
                    "finish_reason" => "stop",
                    "message" => Dict(
                        "role" => "assistant",
                        "content" => "The result is 8."
                    )
                )],
                "usage" => Dict("prompt_tokens" => 20, "completion_tokens" => 10, "total_tokens" => 30)
            ))
            name == "add" ? string(args["a"] + args["b"]) : error("Unknown: $name")
        end

        result = tool_loop!(chat, dispatcher; max_turns=5)

        @test result.completed
        @test result.turns_used == 2
        @test length(result.tool_calls) == 1
        @test result.tool_calls[1].tool_name == "add"
        @test result.tool_calls[1].success
        @test result.tool_calls[1].result.result == "8"
        @test result.response isa LLMSuccess
        @test result.response.message.content == "The result is 8."
    end

    @testset "tool_loop! max turns exhausted (Chat)" begin
        response_status[] = 200

        # Always return tool calls
        response_body[] = JSON.json(Dict(
            "choices" => [Dict(
                "finish_reason" => "tool_calls",
                "message" => Dict(
                    "role" => "assistant",
                    "tool_calls" => [Dict(
                        "id" => "call_loop",
                        "type" => "function",
                        "function" => Dict(
                            "name" => "noop",
                            "arguments" => "{}"
                        )
                    )]
                )
            )],
            "usage" => Dict("prompt_tokens" => 5, "completion_tokens" => 3, "total_tokens" => 8)
        ))

        tool = GPTTool(func=GPTFunctionSignature(name="noop"))
        chat = Chat(service=MockServiceEndpoint, model="gpt-4o", tools=[tool])
        push!(chat, Message(role=UniLM.RoleSystem, content="sys"))
        push!(chat, Message(role=UniLM.RoleUser, content="go"))

        result = tool_loop!(chat, (name, args) -> "ok"; max_turns=3)

        @test !result.completed
        @test result.turns_used == 3
        @test length(result.tool_calls) == 3
        @test contains(result.llm_error, "max turns")
    end

    @testset "tool_loop! with parallel tool calls (Chat)" begin
        response_status[] = 200

        # Response with two parallel tool calls
        response_body[] = JSON.json(Dict(
            "choices" => [Dict(
                "finish_reason" => "tool_calls",
                "message" => Dict(
                    "role" => "assistant",
                    "tool_calls" => [
                        Dict("id" => "call_a", "type" => "function",
                            "function" => Dict("name" => "add", "arguments" => """{"a":1,"b":2}""")),
                        Dict("id" => "call_b", "type" => "function",
                            "function" => Dict("name" => "add", "arguments" => """{"a":3,"b":4}"""))
                    ]
                )
            )],
            "usage" => Dict("prompt_tokens" => 10, "completion_tokens" => 5, "total_tokens" => 15)
        ))

        tool = GPTTool(func=GPTFunctionSignature(name="add"))
        chat = Chat(service=MockServiceEndpoint, model="gpt-4o", tools=[tool])
        push!(chat, Message(role=UniLM.RoleSystem, content="sys"))
        push!(chat, Message(role=UniLM.RoleUser, content="calc"))

        dispatcher = (name, args) -> begin
            response_body[] = JSON.json(Dict(
                "choices" => [Dict(
                    "finish_reason" => "stop",
                    "message" => Dict("role" => "assistant", "content" => "Done."))],
                "usage" => Dict("prompt_tokens" => 15, "completion_tokens" => 5, "total_tokens" => 20)
            ))
            string(args["a"] + args["b"])
        end

        result = tool_loop!(chat, dispatcher; max_turns=5)

        @test result.completed
        @test result.turns_used == 2
        @test length(result.tool_calls) == 2
        @test result.tool_calls[1].result.result == "3"
        @test result.tool_calls[2].result.result == "7"
    end

    # ═══════════════════════════════════════════════════════════════════════
    # Tool Loop: Responses API
    # ═══════════════════════════════════════════════════════════════════════

    @testset "tool_loop two-turn cycle (Respond)" begin
        response_status[] = 200

        # First response: function call
        response_body[] = JSON.json(Dict(
            "id" => "resp_001",
            "status" => "completed",
            "model" => "gpt-4o",
            "output" => [Dict(
                "type" => "function_call",
                "id" => "fc_001",
                "call_id" => "call_abc",
                "name" => "add",
                "arguments" => """{"a": 3, "b": 5}""",
                "status" => "completed"
            )]
        ))

        ft = FunctionTool(name="add", description="Add two numbers",
            parameters=Dict("type"=>"object",
                "properties"=>Dict("a"=>Dict("type"=>"number"),"b"=>Dict("type"=>"number"))))

        r = Respond(input="What is 3+5?", tools=[ft], service=MockServiceEndpoint)

        dispatcher = (name, args) -> begin
            # Swap to text response for next call
            response_body[] = JSON.json(Dict(
                "id" => "resp_002",
                "status" => "completed",
                "model" => "gpt-4o",
                "output" => [Dict(
                    "type" => "message",
                    "role" => "assistant",
                    "status" => "completed",
                    "content" => [Dict("type" => "output_text", "text" => "The result is 8.")]
                )]
            ))
            name == "add" ? string(args["a"] + args["b"]) : error("Unknown: $name")
        end

        result = tool_loop(r, dispatcher; max_turns=5)

        @test result.completed
        @test result.turns_used == 2
        @test length(result.tool_calls) == 1
        @test result.tool_calls[1].tool_name == "add"
        @test result.tool_calls[1].success
        @test result.response isa ResponseSuccess
        @test output_text(result.response) == "The result is 8."
    end

    @testset "tool_loop max turns exhausted (Respond)" begin
        response_status[] = 200

        # Always return function call
        response_body[] = JSON.json(Dict(
            "id" => "resp_loop",
            "status" => "completed",
            "model" => "gpt-4o",
            "output" => [Dict(
                "type" => "function_call",
                "id" => "fc_loop",
                "call_id" => "call_loop",
                "name" => "noop",
                "arguments" => "{}",
                "status" => "completed"
            )]
        ))

        ft = FunctionTool(name="noop")
        r = Respond(input="go", tools=[ft], service=MockServiceEndpoint)

        result = tool_loop(r, (name, args) -> "ok"; max_turns=3)

        @test !result.completed
        @test result.turns_used == 3
        @test length(result.tool_calls) == 3
        @test contains(result.llm_error, "max turns")
    end

    # ═══════════════════════════════════════════════════════════════════════
    # Chat Completions: SUCCESSFUL streaming + non-stream retry recursion
    # Covers requests.jl: 264–307 (stream accumulation, callbacks, on_tool_call,
    # eos→_build_stream_message→LLMSuccess→_accumulate_cost!), 343–346 (non-stream
    # retry-then-recover recursion), 355–356 (chatrequest! stream dispatch).
    # The streamed resp.body is empty under HTTP 2.x → assert on the parsed
    # LLMSuccess message, never the raw body (mirrors the existing stream tests).
    # SSE chunk wire-format is reused verbatim from test/requests.jl.
    # ═══════════════════════════════════════════════════════════════════════

    @testset "chatrequest! stream=true success → LLMSuccess via dispatch (355–356, 302–307)" begin
        # Content deltas "Hello" + " world", a usage chunk, then finish_reason stop + [DONE].
        response_status[] = 200
        response_headers[] = Pair{String,String}[]
        response_body[] =
            """data: {"id":"chatcmpl-1","object":"chat.completion.chunk","created":1,"model":"gpt-4o","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}\n\n""" *
            """data: {"id":"chatcmpl-1","object":"chat.completion.chunk","created":1,"model":"gpt-4o","choices":[{"index":0,"delta":{"content":" world"},"finish_reason":null}]}\n\n""" *
            """data: {"id":"chatcmpl-1","choices":[],"usage":{"prompt_tokens":7,"completion_tokens":2,"total_tokens":9}}\n\n""" *
            """data: {"id":"chatcmpl-1","object":"chat.completion.chunk","created":1,"model":"gpt-4o","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n""" *
            """data: [DONE]"""

        chat = Chat(service=MockServiceEndpoint, model="gpt-4o", stream=true)
        push!(chat, Message(role=UniLM.RoleSystem, content="sys"))
        push!(chat, Message(role=UniLM.RoleUser, content="usr"))

        # Go through chatrequest! (the public dispatch) — NOT _chatrequeststream directly.
        task = chatrequest!(chat)
        @test task isa Task
        result = fetch(task)

        @test result isa LLMSuccess
        @test result.message.role == UniLM.RoleAssistant
        @test result.message.content == "Hello world"      # exact accumulated content
        @test result.message.finish_reason == UniLM.STOP
        @test result.usage isa UniLM.TokenUsage              # populated from the stream usage chunk
        @test result.usage.prompt_tokens == 7
        @test result.usage.completion_tokens == 2
        @test result.usage.total_tokens == 9
        # update! appended the assistant message onto the chat (history defaults true)
        @test last(chat).content == "Hello world"
        set_error!(200, "")
    end

    @testset "chatrequest! stream callback receives final Message on eos (286)" begin
        # Under HTTP 2.x the mock delivers the whole SSE body in a single readavailable,
        # so _parse_chunk reports eos on the first loop iteration: the eos branch (282–286)
        # runs and invokes callback(m[], close_ref) with the assembled Message. The
        # incremental text-delta branch (287–297) requires multiple network reads, which a
        # single canned HTTP.Response cannot produce — so it is not asserted here (asserting
        # incremental deltas against this transport would test a fiction, not the source).
        response_status[] = 200
        response_headers[] = Pair{String,String}[]
        response_body[] =
            """data: {"id":"c","choices":[{"index":0,"delta":{"content":"Ahoy"},"finish_reason":null}]}\n\n""" *
            """data: {"id":"c","choices":[{"index":0,"delta":{"content":" there"},"finish_reason":null}]}\n\n""" *
            """data: {"id":"c","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n""" *
            """data: [DONE]"""

        chat = Chat(service=MockServiceEndpoint, model="gpt-4o", stream=true)
        push!(chat, Message(role=UniLM.RoleSystem, content="sys"))
        push!(chat, Message(role=UniLM.RoleUser, content="usr"))

        received = Any[]
        cb = (chunk, close_ref) -> push!(received, chunk)

        result = fetch(chatrequest!(chat; callback=cb))

        @test result isa LLMSuccess
        @test result.message.content == "Ahoy there"
        # The callback fired exactly once — with the final assembled Message (eos branch).
        @test length(received) == 1
        @test received[1] isa Message
        @test received[1].content == "Ahoy there"
        @test received[1].finish_reason == UniLM.STOP
        set_error!(200, "")
    end

    @testset "chatrequest! stream on_tool_call fires with parsed GPTToolCall (263–280, 220–229)" begin
        # A single streamed tool call: id, name, then argument fragments, finishing tool_calls.
        response_status[] = 200
        response_headers[] = Pair{String,String}[]
        response_body[] =
            """data: {"id":"c","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_xyz","type":"function","function":{"name":"get_weather","arguments":""}}]},"finish_reason":null}]}\n\n""" *
            """data: {"id":"c","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\\\"location\\\":"}}]},"finish_reason":null}]}\n\n""" *
            """data: {"id":"c","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\\\"NYC\\\"}"}}]},"finish_reason":null}]}\n\n""" *
            """data: {"id":"c","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}\n\n""" *
            """data: [DONE]"""

        tool = GPTTool(func=GPTFunctionSignature(name="get_weather", description="Weather",
            parameters=Dict("type"=>"object", "properties"=>Dict("location"=>Dict("type"=>"string")))))
        chat = Chat(service=MockServiceEndpoint, model="gpt-4o", stream=true, tools=[tool])
        push!(chat, Message(role=UniLM.RoleSystem, content="sys"))
        push!(chat, Message(role=UniLM.RoleUser, content="weather in NYC?"))

        received_calls = GPTToolCall[]
        on_tc = tc -> push!(received_calls, tc)

        result = fetch(chatrequest!(chat; on_tool_call=on_tc))

        @test result isa LLMSuccess
        @test result.message.finish_reason == UniLM.TOOL_CALLS
        # The final message carries the assembled tool call (built by _build_stream_message).
        @test length(result.message.tool_calls) == 1
        @test result.message.tool_calls[1].id == "call_xyz"
        @test result.message.tool_calls[1].func.name == "get_weather"
        @test result.message.tool_calls[1].func.arguments["location"] == "NYC"
        # on_tool_call fired exactly once, with the fully-parsed GPTToolCall.
        @test length(received_calls) == 1
        @test received_calls[1].id == "call_xyz"
        @test received_calls[1].func.name == "get_weather"
        @test received_calls[1].func.arguments["location"] == "NYC"
        set_error!(200, "")
    end

    @testset "chatrequest! non-stream retry-then-recover recursion (343–346)" begin
        # First request: retryable 503 (drains queue entry 1) → recurse → 200 success (entry 2).
        # retries defaults to 0 < _RETRY_MAX_ATTEMPTS, so 343–346 (delay/sleep/recurse) runs once.
        success_body = JSON.json(Dict(
            "choices" => [Dict(
                "finish_reason" => "stop",
                "message" => Dict("role" => "assistant", "content" => "recovered after retry"))],
            "usage" => Dict("prompt_tokens" => 4, "completion_tokens" => 3, "total_tokens" => 7)))
        response_queue[] = [(503, ""), (200, success_body)]

        chat = Chat(service=MockServiceEndpoint, model="gpt-4o")
        push!(chat, Message(role=UniLM.RoleSystem, content="sys"))
        push!(chat, Message(role=UniLM.RoleUser, content="usr"))

        result = chatrequest!(chat)   # default retries=0 → genuinely retries

        @test result isa LLMSuccess
        @test result.message.content == "recovered after retry"
        @test result.usage.total_tokens == 7
        @test isempty(response_queue[])   # both queued responses consumed → it really retried once
        set_error!(200, "")
    end

    # ═══════════════════════════════════════════════════════════════════════
    # Embeddings: SUCCESS + retry-then-recover recursion
    # Covers requests.jl: 428–430 (200 → update! → EmbeddingSuccess) and
    # 433–436 (retryable → recurse → recover). _store_embedding! resizes the
    # preallocated buffer to the returned length, so short exact vectors are safe.
    # ═══════════════════════════════════════════════════════════════════════

    @testset "embeddingrequest! success → EmbeddingSuccess with exact vectors + usage (428–430)" begin
        response_status[] = 200
        response_headers[] = Pair{String,String}[]
        response_body[] = JSON.json(Dict(
            "object" => "list",
            "model" => "mock-embedding",
            "data" => [Dict("object" => "embedding", "index" => 0,
                "embedding" => [0.5, -0.25, 0.125])],
            "usage" => Dict("prompt_tokens" => 3, "total_tokens" => 3)))

        emb = UniLM.Embeddings("hello"; service=MockServiceEndpoint)
        result = embeddingrequest!(emb)

        @test result isa EmbeddingSuccess
        @test embedding_vectors(result) == [0.5, -0.25, 0.125]   # exact returned vector
        @test emb.embeddings == [0.5, -0.25, 0.125]              # filled in place on the request
        @test result.usage isa UniLM.TokenUsage
        @test result.usage.prompt_tokens == 3
        @test result.usage.total_tokens == 3
        @test result.raw["model"] == "mock-embedding"           # raw JSON retained
        set_error!(200, "")
    end

    @testset "embeddingrequest! retry-then-recover recursion (433–436)" begin
        emb_body = JSON.json(Dict(
            "object" => "list", "model" => "mock-embedding",
            "data" => [Dict("object" => "embedding", "index" => 0, "embedding" => [1.0, 2.0])],
            "usage" => Dict("prompt_tokens" => 2, "total_tokens" => 2)))
        response_queue[] = [(503, ""), (200, emb_body)]

        emb = UniLM.Embeddings("retry me"; service=MockServiceEndpoint)
        result = embeddingrequest!(emb)   # default retries=0 → genuinely retries once

        @test result isa EmbeddingSuccess
        @test embedding_vectors(result) == [1.0, 2.0]
        @test result.usage.total_tokens == 2
        @test isempty(response_queue[])   # both queued responses consumed → it retried then recovered
        set_error!(200, "")
    end

finally
    close(mock_server)
end
