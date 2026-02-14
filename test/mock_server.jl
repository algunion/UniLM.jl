using Sockets

# ─── Mock HTTP Server Setup ───────────────────────────────────────────────────
# Start a local HTTP server to test error handling, retry logic,
# and success paths without hitting real APIs.

response_status = Ref{Int}(200)
response_body = Ref{String}("{}")

tcp_server = Sockets.listen(Sockets.localhost, 0)
mock_port = Int(Sockets.getsockname(tcp_server)[2])
close(tcp_server)

# Use the discovered free port
mock_server = HTTP.serve!("127.0.0.1", mock_port; verbose=false) do req
    return HTTP.Response(response_status[], ["Content-Type" => "application/json"], Vector{UInt8}(response_body[]))
end

mock_base_url = "http://127.0.0.1:$mock_port"

# ─── Test Service Endpoint ────────────────────────────────────────────────────

struct MockServiceEndpoint <: UniLM.ServiceEndpoint end
UniLM._api_base_url(::Type{MockServiceEndpoint}) = mock_base_url
UniLM.get_url(::Type{MockServiceEndpoint}, ::Chat) = mock_base_url * UniLM.CHAT_COMPLETIONS_PATH
UniLM.auth_header(::Type{MockServiceEndpoint}) = ["Content-Type" => "application/json"]

# Helper: standard success response for Chat Completions
function set_chat_success!(content="Mock response")
    response_status[] = 200
    response_body[] = JSON.json(Dict(
        "choices" => [Dict(
            "finish_reason" => "stop",
            "message" => Dict("role" => "assistant", "content" => content)
        )]
    ))
end

# Helper: standard success response for Responses API
function set_respond_success!(text="Mock response", id="resp_mock")
    response_status[] = 200
    response_body[] = JSON.json(Dict(
        "id" => id,
        "object" => "response",
        "status" => "completed",
        "model" => "gpt-5.2",
        "output" => [Dict(
            "type" => "message",
            "content" => [Dict("type" => "output_text", "text" => text)]
        )],
        "usage" => Dict("input_tokens" => 10, "output_tokens" => 5, "total_tokens" => 15)
    ))
end

# Helper: standard success response for Image Generation
function set_image_success!(b64="dGVzdA==")
    response_status[] = 200
    response_body[] = JSON.json(Dict(
        "created" => 1234567890,
        "data" => [Dict("b64_json" => b64)]
    ))
end

# Helper: set error response
function set_error!(status, msg="error")
    response_status[] = status
    response_body[] = JSON.json(Dict("error" => Dict("message" => msg, "type" => "server_error")))
end

try

    # ─── Chat Completions: retry on 500/503 ────────────────────────────────

    @testset "chatrequest! with 500 (retry exhausted)" begin
        set_error!(500, "Internal Server Error")

        chat = Chat(service=MockServiceEndpoint, model="gpt-4o")
        push!(chat, Message(role=UniLM.RoleSystem, content="sys"))
        push!(chat, Message(role=UniLM.RoleUser, content="usr"))

        # retries=30 → 30 < 30 is false → returns LLMFailure immediately (no sleep)
        result = chatrequest!(chat; retries=30)
        @test result isa LLMFailure
        @test result.status == 500
    end

    @testset "chatrequest! with 503 (retry once then exhaust)" begin
        set_error!(503, "Service Unavailable")

        chat = Chat(service=MockServiceEndpoint, model="gpt-4o")
        push!(chat, Message(role=UniLM.RoleSystem, content="sys"))
        push!(chat, Message(role=UniLM.RoleUser, content="usr"))

        # retries=29 → 29 < 30 triggers one retry (with sleep(1))
        # then retries=30 → exhausted → LLMFailure
        result = chatrequest!(chat; retries=29)
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

    @testset "chatrequest! with 200 (mock success)" begin
        set_chat_success!("Hello from mock!")

        chat = Chat(service=MockServiceEndpoint, model="gpt-4o")
        push!(chat, Message(role=UniLM.RoleSystem, content="sys"))
        push!(chat, Message(role=UniLM.RoleUser, content="usr"))

        result = chatrequest!(chat)
        @test result isa LLMSuccess
        @test result.message.content == "Hello from mock!"
        @test result.message.finish_reason == UniLM.STOP
    end

    # ─── Chat Completions: streaming error path ───────────────────────────

    @testset "_chatrequeststream connection error" begin
        # Point to a non-listening address to cause a connection error
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

    # ─── Responses API: retry on 500/503 ──────────────────────────────────

    @testset "respond with 500 (retry exhausted)" begin
        set_error!(500, "Internal Server Error")

        r = Respond(input="test", service=MockServiceEndpoint)
        result = respond(r; retries=30)

        @test result isa ResponseFailure
        @test result.status == 500
    end

    @testset "respond with 503 (retry once then exhaust)" begin
        set_error!(503, "Service Unavailable")

        r = Respond(input="test", service=MockServiceEndpoint)
        result = respond(r; retries=29)

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

    @testset "respond with 200 (mock success)" begin
        set_respond_success!("Hello from mock!")

        r = Respond(input="test", service=MockServiceEndpoint)
        result = respond(r)

        @test result isa ResponseSuccess
        @test output_text(result) == "Hello from mock!"
    end

    # ─── Responses API: get_response success path ─────────────────────────

    @testset "get_response with 200 (mock success)" begin
        set_respond_success!("Retrieved response", "resp_123")

        result = get_response("resp_123"; service=MockServiceEndpoint)

        @test result isa ResponseSuccess
        @test result.response.id == "resp_123"
        @test result.response.status == "completed"
    end

    @testset "get_response with 404 (mock failure)" begin
        set_error!(404, "Not Found")

        result = get_response("resp_nonexistent"; service=MockServiceEndpoint)

        @test result isa ResponseFailure
        @test result.status == 404
    end

    # ─── Responses API: delete_response success path ──────────────────────

    @testset "delete_response with 200 (mock success)" begin
        response_status[] = 200
        response_body[] = JSON.json(Dict("id" => "resp_123", "object" => "response", "deleted" => true))

        result = delete_response("resp_123"; service=MockServiceEndpoint)

        @test result isa Dict
        @test result["deleted"] == true
        @test result["id"] == "resp_123"
    end

    @testset "delete_response with 404 (mock failure)" begin
        set_error!(404, "Not Found")

        result = delete_response("resp_nonexistent"; service=MockServiceEndpoint)

        @test result isa ResponseFailure
        @test result.status == 404
    end

    # ─── Responses API: list_input_items success path ─────────────────────

    @testset "list_input_items with 200 (mock success)" begin
        response_status[] = 200
        response_body[] = JSON.json(Dict(
            "data" => [Dict("type" => "message", "role" => "user", "content" => "test")],
            "first_id" => "item_001",
            "last_id" => "item_001",
            "has_more" => false
        ))

        result = list_input_items("resp_123"; service=MockServiceEndpoint)

        @test result isa Dict
        @test haskey(result, "data")
        @test result["has_more"] == false
        @test length(result["data"]) == 1
    end

    @testset "list_input_items with 404 (mock failure)" begin
        set_error!(404, "Not Found")

        result = list_input_items("resp_nonexistent"; service=MockServiceEndpoint)

        @test result isa ResponseFailure
        @test result.status == 404
    end

    # ─── Image Generation: retry on 500/503 ───────────────────────────────

    @testset "generate_image with 500 (retry exhausted)" begin
        set_error!(500, "Internal Server Error")

        ig = ImageGeneration(prompt="test", service=MockServiceEndpoint)
        result = generate_image(ig; retries=30)

        @test result isa ImageFailure
        @test result.status == 500
    end

    @testset "generate_image with 503 (retry once then exhaust)" begin
        set_error!(503, "Service Unavailable")

        ig = ImageGeneration(prompt="test", service=MockServiceEndpoint)
        result = generate_image(ig; retries=29)

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

    @testset "generate_image with 200 (mock success)" begin
        set_image_success!("aGVsbG8=")

        ig = ImageGeneration(prompt="test", service=MockServiceEndpoint)
        result = generate_image(ig)

        @test result isa ImageSuccess
        @test length(image_data(result)) == 1
        @test image_data(result)[1] == "aGVsbG8="
    end

    # ─── Embeddings: error branch with status_exception=false ─────────────

    @testset "embeddingrequest! with non-200 error status" begin
        withenv("OPENAI_API_KEY" => "sk-invalid-test-key-mock") do
            emb = UniLM.Embeddings("test embedding")
            result = embeddingrequest!(emb)
            # With status_exception=false, 401 → else branch → returns nothing
            @test isnothing(result)
        end
    end

    # ─── Embeddings: retry on 500 via mock server ─────────────────────────
    # Temporarily redirect Embeddings URL to mock server

    @testset "embeddingrequest! with 500 (retry exhausted)" begin
        # Override get_url for Embeddings to point to mock server
        UniLM.get_url(::UniLM.Embeddings) = mock_base_url * UniLM.EMBEDDINGS_PATH

        set_error!(500, "Internal Server Error")

        withenv("OPENAI_API_KEY" => "test-key") do
            emb = UniLM.Embeddings("test")
            # retries=30 → 30 < 30 is false → max retries exceeded → nothing
            result = embeddingrequest!(emb; retries=30)
            @test isnothing(result)
        end
    end

    @testset "embeddingrequest! with 503 (retry once then exhaust)" begin
        set_error!(503, "Service Unavailable")

        withenv("OPENAI_API_KEY" => "test-key") do
            emb = UniLM.Embeddings("test")
            result = embeddingrequest!(emb; retries=29)
            @test isnothing(result)
        end
    end

    @testset "embeddingrequest! catch block (connection error)" begin
        UniLM.get_url(::UniLM.Embeddings) = "http://127.0.0.1:1" * UniLM.EMBEDDINGS_PATH

        withenv("OPENAI_API_KEY" => "test-key") do
            emb = UniLM.Embeddings("test")
            result = embeddingrequest!(emb)
            @test isnothing(result)
        end

        # Restore original get_url
        UniLM.get_url(::UniLM.Embeddings) = UniLM.OPENAI_BASE_URL * UniLM.EMBEDDINGS_PATH
    end

    # ─── image_data for failure types ─────────────────────────────────────

    @testset "image_data on failure types" begin
        @test image_data(ImageFailure(response="err", status=400)) == String[]
        @test image_data(ImageCallError(error="err")) == String[]
    end

    # ─── output_text / function_calls for error types ─────────────────────

    @testset "output_text on error types" begin
        @test output_text(ResponseFailure(response="body", status=400)) == "Error (HTTP 400): body"
        @test output_text(ResponseCallError(error="timeout")) == "Error: timeout"
    end

    @testset "function_calls on error types" begin
        @test function_calls(ResponseFailure(response="body", status=400)) == Dict{String,Any}[]
        @test function_calls(ResponseCallError(error="timeout")) == Dict{String,Any}[]
    end

    # ─── Exception-level (catch block) tests ──────────────────────────────
    # Point to a dead server to trigger connection errors that hit catch blocks

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

    @testset "generate_image catch block (connection error)" begin
        struct ImageDeadEndpoint <: UniLM.ServiceEndpoint end
        UniLM._api_base_url(::Type{ImageDeadEndpoint}) = "http://127.0.0.1:1"
        UniLM.auth_header(::Type{ImageDeadEndpoint}) = ["Content-Type" => "application/json"]

        ig = ImageGeneration(prompt="test", service=ImageDeadEndpoint)
        result = generate_image(ig)

        @test result isa ImageCallError
        @test !isempty(result.error)
    end

    @testset "_respond_stream catch block (connection error)" begin
        r = Respond(input="test", service=RespondDeadEndpoint, stream=true)
        result = respond(r)
        # Returns a Task; fetch it to get the error
        @test result isa Task
        inner = fetch(result)
        @test inner isa ResponseCallError
        @test !isempty(inner.error)
    end

finally
    close(mock_server)
end
