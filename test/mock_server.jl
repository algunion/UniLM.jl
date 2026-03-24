using Sockets

# ─── Mock HTTP Server Setup ───────────────────────────────────────────────────
# This mock server exists ONLY for scenarios that cannot be tested against the
# real/live server: error codes, retry exhaustion, and connection failures.

response_status = Ref{Int}(200)
response_body = Ref{String}("{}")
response_headers = Ref{Vector{Pair{String,String}}}(Pair{String,String}[])

tcp_server = Sockets.listen(Sockets.localhost, 0)
mock_port = Int(Sockets.getsockname(tcp_server)[2])
close(tcp_server)

mock_server = HTTP.serve!("127.0.0.1", mock_port; verbose=false) do req
    headers = vcat(["Content-Type" => "application/json"], response_headers[])
    return HTTP.Response(response_status[], headers, Vector{UInt8}(response_body[]))
end

mock_base_url = "http://127.0.0.1:$mock_port"

# ─── Test Service Endpoint ────────────────────────────────────────────────────

struct MockServiceEndpoint <: UniLM.ServiceEndpoint end
UniLM._api_base_url(::Type{MockServiceEndpoint}) = mock_base_url
UniLM.get_url(::Type{MockServiceEndpoint}, ::Chat) = mock_base_url * UniLM.CHAT_COMPLETIONS_PATH
UniLM.get_url(::Type{MockServiceEndpoint}, ::Embeddings) = mock_base_url * UniLM.EMBEDDINGS_PATH
UniLM.auth_header(::Type{MockServiceEndpoint}) = ["Content-Type" => "application/json"]

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
        @test_throws ErrorException embeddingrequest!(emb; retries=30)

        # Reset mock state so subsequent tests don't inherit 429 + Retry-After
        set_error!(200, "")
    end

    # ═══════════════════════════════════════════════════════════════════════
    # Embeddings: error paths
    # ═══════════════════════════════════════════════════════════════════════

    @testset "embeddingrequest! with non-200 error status" begin
        set_error!(401, "Unauthorized")
        emb = UniLM.Embeddings("test embedding"; service=MockServiceEndpoint)
        @test_throws ErrorException embeddingrequest!(emb)
    end

    @testset "embeddingrequest! with 500 (retry exhausted)" begin
        set_error!(500, "Internal Server Error")
        emb = UniLM.Embeddings("test"; service=MockServiceEndpoint)
        @test_throws ErrorException embeddingrequest!(emb; retries=30)
    end

    @testset "embeddingrequest! with 503 (retry exhausted)" begin
        set_error!(503, "Service Unavailable")
        emb = UniLM.Embeddings("test"; service=MockServiceEndpoint)
        @test_throws ErrorException embeddingrequest!(emb; retries=30)
    end

    @testset "embeddingrequest! catch block (connection error)" begin
        dead = UniLM.GenericOpenAIEndpoint("http://127.0.0.1:1", "")
        emb = UniLM.Embeddings("test"; service=dead)
        @test_throws ErrorException embeddingrequest!(emb)
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

finally
    close(mock_server)
end
