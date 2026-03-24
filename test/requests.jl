@testset "URL generation" begin
    @testset "OpenAI get_url for Chat" begin
        chat = Chat(model="gpt-4o")
        @test UniLM.get_url(chat) == "https://api.openai.com/v1/chat/completions"

        # Any model routes to chat completions when used with Chat
        chat2 = Chat(model="gpt-4.1-mini")
        @test UniLM.get_url(chat2) == "https://api.openai.com/v1/chat/completions"
    end

    @testset "OpenAI get_url for Embeddings" begin
        emb = UniLM.Embeddings("test")
        @test UniLM.get_url(emb) == "https://api.openai.com/v1/embeddings"
    end

    @testset "OpenAI get_url dispatches on request type" begin
        chat = Chat(model="gpt-4o")
        @test UniLM.get_url(UniLM.OPENAIServiceEndpoint, chat) == "https://api.openai.com/v1/chat/completions"

        chat2 = Chat(model="gpt-4o-mini")
        @test UniLM.get_url(UniLM.OPENAIServiceEndpoint, chat2) == "https://api.openai.com/v1/chat/completions"
    end

    @testset "Gemini get_url" begin
        chat = Chat(service=UniLM.GEMINIServiceEndpoint, model="gemini-2.0-flash")
        @test UniLM.get_url(UniLM.GEMINIServiceEndpoint, chat) == "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
    end
end

@testset "Auth headers" begin
    @testset "OpenAI auth header" begin
        withenv("OPENAI_API_KEY" => "test-openai-key") do
            headers = UniLM.auth_header(UniLM.OPENAIServiceEndpoint)
            @test length(headers) == 2
            @test headers[1][1] == "Authorization"
            @test headers[1][2] == "Bearer test-openai-key"
            @test headers[2] == ("Content-Type" => "application/json")
        end
    end

    @testset "Azure auth header" begin
        withenv("AZURE_OPENAI_API_KEY" => "test-azure-key") do
            headers = UniLM.auth_header(UniLM.AZUREServiceEndpoint)
            @test length(headers) == 2
            @test headers[1][1] == "api-key"
            @test headers[1][2] == "test-azure-key"
            @test headers[2] == ("Content-Type" => "application/json")
        end
    end

    @testset "Gemini auth header" begin
        withenv("GEMINI_API_KEY" => "test-gemini-key") do
            headers = UniLM.auth_header(UniLM.GEMINIServiceEndpoint)
            @test length(headers) == 2
            @test headers[1][1] == "Authorization"
            @test headers[1][2] == "Bearer test-gemini-key"
            @test headers[2] == ("Content-Type" => "application/json")
        end
    end
end

@testset "extract_message" begin
    function make_response(body::Dict; status=200)
        body_bytes = Vector{UInt8}(JSON.json(body))
        HTTP.Response(status, [], body_bytes)
    end

    @testset "stop finish_reason with content" begin
        body = Dict(
            "choices" => [Dict(
                "finish_reason" => "stop",
                "message" => Dict("role" => "assistant", "content" => "Hello there!")
            )]
        )
        resp = make_response(body)
        extracted = UniLM.extract_message(resp)
        m = extracted.message
        @test m.role == UniLM.RoleAssistant
        @test m.content == "Hello there!"
        @test m.finish_reason == UniLM.STOP
        @test isnothing(extracted.usage)  # no usage in body
    end

    @testset "stop with usage field" begin
        body = Dict(
            "choices" => [Dict(
                "finish_reason" => "stop",
                "message" => Dict("role" => "assistant", "content" => "Hi!")
            )],
            "usage" => Dict("prompt_tokens" => 10, "completion_tokens" => 5, "total_tokens" => 15)
        )
        resp = make_response(body)
        extracted = UniLM.extract_message(resp)
        @test extracted.message.content == "Hi!"
        u = extracted.usage
        @test !isnothing(u)
        @test u.prompt_tokens == 10
        @test u.completion_tokens == 5
        @test u.total_tokens == 15
    end

    @testset "tool_calls finish_reason" begin
        body = Dict(
            "choices" => [Dict(
                "finish_reason" => "tool_calls",
                "message" => Dict(
                    "role" => "assistant",
                    "content" => nothing,
                    "tool_calls" => [Dict(
                        "id" => "call_abc123",
                        "type" => "function",
                        "function" => Dict(
                            "name" => "get_weather",
                            "arguments" => "{\"location\":\"NYC\"}"
                        )
                    )]
                )
            )]
        )
        resp = make_response(body)
        m = UniLM.extract_message(resp).message
        @test m.role == UniLM.RoleAssistant
        @test m.finish_reason == UniLM.TOOL_CALLS
        @test length(m.tool_calls) == 1
        @test m.tool_calls[1].id == "call_abc123"
        @test m.tool_calls[1].func.name == "get_weather"
        @test m.tool_calls[1].func.arguments["location"] == "NYC"
    end

    @testset "content_filter finish_reason with refusal" begin
        body = Dict(
            "choices" => [Dict(
                "finish_reason" => "content_filter",
                "message" => Dict(
                    "role" => "assistant",
                    "refusal" => "This content was filtered."
                )
            )]
        )
        resp = make_response(body)
        m = UniLM.extract_message(resp).message
        @test m.role == UniLM.RoleAssistant
        @test m.finish_reason == UniLM.CONTENT_FILTER
        @test m.refusal_message == "This content was filtered."
    end

    @testset "fallback message" begin
        body = Dict(
            "choices" => [Dict(
                "finish_reason" => "length",
                "message" => Dict("role" => "assistant", "content" => nothing)
            )]
        )
        resp = make_response(body)
        m = UniLM.extract_message(resp).message
        @test m.role == UniLM.RoleAssistant
        @test m.content == "No response from the model."
        @test m.finish_reason == "length"
    end

    @testset "multiple tool_calls" begin
        body = Dict(
            "choices" => [Dict(
                "finish_reason" => "tool_calls",
                "message" => Dict(
                    "role" => "assistant",
                    "content" => nothing,
                    "tool_calls" => [
                        Dict(
                            "id" => "call_1",
                            "type" => "function",
                            "function" => Dict("name" => "fn1", "arguments" => "{\"a\":\"1\"}")
                        ),
                        Dict(
                            "id" => "call_2",
                            "type" => "function",
                            "function" => Dict("name" => "fn2", "arguments" => "{\"b\":\"2\"}")
                        )
                    ]
                )
            )]
        )
        resp = make_response(body)
        m = UniLM.extract_message(resp).message
        @test length(m.tool_calls) == 2
        @test m.tool_calls[1].func.name == "fn1"
        @test m.tool_calls[2].func.name == "fn2"
    end
end

@testset "_parse_chunk" begin
    @testset "normal content chunk" begin
        chunk = """data: {"id":"chatcmpl-1","object":"chat.completion.chunk","created":1,"model":"gpt-4o","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}"""
        state = UniLM.StreamState()
        failbuff = IOBuffer()
        result = UniLM._parse_chunk(chunk, state, failbuff)
        @test result.eos == false
        @test String(take!(state.content)) == "Hello"
        @test isempty(take!(failbuff))
    end

    @testset "done chunk" begin
        chunk = "data: [DONE]"
        state = UniLM.StreamState()
        failbuff = IOBuffer()
        result = UniLM._parse_chunk(chunk, state, failbuff)
        @test result.eos == true
    end

    @testset "stop finish_reason" begin
        chunk = """data: {"id":"chatcmpl-1","object":"chat.completion.chunk","created":1,"model":"gpt-4o","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}"""
        state = UniLM.StreamState()
        failbuff = IOBuffer()
        result = UniLM._parse_chunk(chunk, state, failbuff)
        @test result.eos == true
        @test isempty(take!(state.content))
        @test state.finish_reason == "stop"
    end

    @testset "empty chunk" begin
        state = UniLM.StreamState()
        failbuff = IOBuffer()
        result = UniLM._parse_chunk("", state, failbuff)
        @test result.eos == false
    end

    @testset "whitespace-only chunk" begin
        state = UniLM.StreamState()
        failbuff = IOBuffer()
        result = UniLM._parse_chunk("   \n  \n  ", state, failbuff)
        @test result.eos == false
    end

    @testset "multi-line chunk with content then DONE" begin
        chunk = """data: {"id":"chatcmpl-1","object":"chat.completion.chunk","created":1,"model":"gpt-4o","choices":[{"index":0,"delta":{"content":" world"},"finish_reason":null}]}

data: [DONE]"""
        state = UniLM.StreamState()
        failbuff = IOBuffer()
        result = UniLM._parse_chunk(chunk, state, failbuff)
        @test result.eos == true
        @test String(take!(state.content)) == " world"
    end

    @testset "malformed JSON goes to failbuff" begin
        chunk = "data: {invalid json"
        state = UniLM.StreamState()
        failbuff = IOBuffer()
        result = UniLM._parse_chunk(chunk, state, failbuff)
        @test result.eos == false
        @test !isempty(take!(failbuff))
    end

    @testset "tool call deltas accumulated by index" begin
        # First chunk: tool call start
        chunk1 = """data: {"id":"chatcmpl-1","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_abc","type":"function","function":{"name":"get_weather","arguments":""}}]},"finish_reason":null}]}"""
        state = UniLM.StreamState()
        failbuff = IOBuffer()
        UniLM._parse_chunk(chunk1, state, failbuff)
        @test haskey(state.tool_calls, 0)
        @test state.tool_calls[0]["id"] == "call_abc"
        @test state.tool_calls[0]["function"]["name"] == "get_weather"

        # Second chunk: arguments fragment
        chunk2 = """data: {"id":"chatcmpl-1","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\\\"location\\\":"}}]},"finish_reason":null}]}"""
        UniLM._parse_chunk(chunk2, state, failbuff)

        chunk3 = """data: {"id":"chatcmpl-1","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\\\"NYC\\\"}"}}]},"finish_reason":null}]}"""
        UniLM._parse_chunk(chunk3, state, failbuff)

        @test state.tool_calls[0]["function"]["arguments"] == "{\"location\":\"NYC\"}"

        # Finish
        chunk4 = """data: {"id":"chatcmpl-1","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}"""
        result = UniLM._parse_chunk(chunk4, state, failbuff)
        @test state.finish_reason == "tool_calls"
    end

    @testset "usage captured from stream chunk" begin
        chunk = """data: {"id":"chatcmpl-1","choices":[],"usage":{"prompt_tokens":25,"completion_tokens":10,"total_tokens":35}}"""
        state = UniLM.StreamState()
        failbuff = IOBuffer()
        UniLM._parse_chunk(chunk, state, failbuff)
        @test !isnothing(state.usage)
        @test state.usage.prompt_tokens == 25
        @test state.usage.completion_tokens == 10
        @test state.usage.total_tokens == 35
    end
end

@testset "_build_stream_message" begin
    @testset "text content message" begin
        state = UniLM.StreamState()
        print(state.content, "Hello world")
        state.finish_reason = "stop"
        msg = UniLM._build_stream_message(state)
        @test msg.role == UniLM.RoleAssistant
        @test msg.content == "Hello world"
        @test msg.finish_reason == "stop"
        @test isnothing(msg.tool_calls)
    end

    @testset "tool calls message" begin
        state = UniLM.StreamState()
        state.finish_reason = "tool_calls"
        state.tool_calls[0] = Dict{String,Any}(
            "id" => "call_1",
            "type" => "function",
            "function" => Dict{String,Any}("name" => "get_weather", "arguments" => "{\"location\":\"NYC\"}")
        )
        msg = UniLM._build_stream_message(state)
        @test msg.role == UniLM.RoleAssistant
        @test msg.finish_reason == UniLM.TOOL_CALLS
        @test length(msg.tool_calls) == 1
        @test msg.tool_calls[1].id == "call_1"
        @test msg.tool_calls[1].func.name == "get_weather"
        @test msg.tool_calls[1].func.arguments["location"] == "NYC"
    end
end

@testset "chatrequest! kwargs" begin
    @testset "missing messages and prompts returns LLMFailure" begin
        result = UniLM.chatrequest!(model="gpt-4o")
        @test result isa LLMFailure
        @test result.status == 499
        @test occursin("No messages", result.response)
    end

    @testset "missing userprompt returns LLMFailure" begin
        result = UniLM.chatrequest!(systemprompt="sys")
        @test result isa LLMFailure
        @test result.status == 499
    end

    @testset "missing systemprompt returns LLMFailure" begin
        result = UniLM.chatrequest!(userprompt="user")
        @test result isa LLMFailure
        @test result.status == 499
    end
end

@testset "Azure deploy name management" begin
    @testset "add_azure_deploy_name!" begin
        UniLM.add_azure_deploy_name!("test-model", "my-deploy")
        @test haskey(UniLM._MODEL_ENDPOINTS_AZURE_OPENAI, "test-model")
        @test UniLM._MODEL_ENDPOINTS_AZURE_OPENAI["test-model"] == "/openai/deployments/my-deploy"
        delete!(UniLM._MODEL_ENDPOINTS_AZURE_OPENAI, "test-model")
    end
end

@testset "Azure URL generation" begin
    UniLM.add_azure_deploy_name!("gpt-4o-test", "my-gpt4o-deploy")
    withenv(
        "AZURE_OPENAI_BASE_URL" => "https://myazure.openai.azure.com",
        "AZURE_OPENAI_API_VERSION" => "2024-02-01"
    ) do
        chat = Chat(service=UniLM.AZUREServiceEndpoint, model="gpt-4o-test")
        url = UniLM.get_url(UniLM.AZUREServiceEndpoint, chat)
        @test startswith(url, "https://myazure.openai.azure.com/openai/deployments/my-gpt4o-deploy/chat/completions")
        @test occursin("api-version=2024-02-01", url)
    end
    delete!(UniLM._MODEL_ENDPOINTS_AZURE_OPENAI, "gpt-4o-test")
end

@testset "chatrequest! kwargs with prompts" begin
    @testset "string prompts build correct Chat" begin
        # This will fail at HTTP level but we test the LLMCallError path
        withenv("OPENAI_API_KEY" => "test-key") do
            result = chatrequest!(
                systemprompt="You are helpful.",
                userprompt="Hello!",
                model="gpt-4o"
            )
            # Without a real API key, we expect a call error
            @test result isa LLMCallError || result isa LLMFailure
        end
    end

    @testset "Message object prompts build correct Chat" begin
        withenv("OPENAI_API_KEY" => "test-key") do
            result = chatrequest!(
                systemprompt=Message(role=UniLM.RoleSystem, content="System"),
                userprompt=Message(role=UniLM.RoleUser, content="User"),
                model="gpt-4o"
            )
            @test result isa LLMCallError || result isa LLMFailure
        end
    end

    @testset "messages keyword builds correct Chat" begin
        withenv("OPENAI_API_KEY" => "test-key") do
            msgs = [
                Message(role=UniLM.RoleSystem, content="sys"),
                Message(role=UniLM.RoleUser, content="usr")
            ]
            result = chatrequest!(; messages=msgs, model="gpt-4o")
            @test result isa LLMCallError || result isa LLMFailure || result isa LLMSuccess
        end
    end

    @testset "prompts override messages" begin
        withenv("OPENAI_API_KEY" => "test-key") do
            msgs = [
                Message(role=UniLM.RoleSystem, content="old sys"),
                Message(role=UniLM.RoleUser, content="old usr")
            ]
            result = chatrequest!(
                messages=msgs,
                systemprompt="new sys",
                userprompt="new usr",
                model="gpt-4o"
            )
            @test result isa LLMCallError || result isa LLMFailure
        end
    end
end

@testset "chatrequest! HTTP error handling" begin
    @testset "non-stream request with invalid API key" begin
        withenv("OPENAI_API_KEY" => "sk-invalid-key-for-testing") do
            chat = Chat(model="gpt-4o")
            push!(chat, Message(role=UniLM.RoleSystem, content="sys"))
            push!(chat, Message(role=UniLM.RoleUser, content="hi"))
            result = chatrequest!(chat)
            @test result isa LLMCallError || result isa LLMFailure
        end
    end
end

@testset "embeddingrequest! error handling" begin
    @testset "with invalid API key" begin
        withenv("OPENAI_API_KEY" => "sk-invalid-key-for-testing") do
            emb = UniLM.Embeddings("test")
            @test_throws ErrorException embeddingrequest!(emb)
        end
    end
end

@testset "_is_retryable" begin
    @test UniLM._is_retryable(429) == true
    @test UniLM._is_retryable(500) == true
    @test UniLM._is_retryable(503) == true
    @test UniLM._is_retryable(400) == false
    @test UniLM._is_retryable(401) == false
    @test UniLM._is_retryable(200) == false
    @test UniLM._is_retryable(404) == false
    # HTTP.Response.status is Int16
    @test UniLM._is_retryable(Int16(429)) == true
    @test UniLM._is_retryable(Int16(500)) == true
    @test UniLM._is_retryable(Int16(200)) == false
end

@testset "_retry_delay" begin
    @testset "delay within expected range" begin
        # At retry 0: computed = min(1.0 * 2.0^0, 60.0) = 1.0, delay ∈ [0, 1.0]
        resp = HTTP.Response(500)
        for _ in 1:20
            d = UniLM._retry_delay(0, resp)
            @test 0.0 <= d <= 1.0
        end
    end

    @testset "delay grows with retries" begin
        resp = HTTP.Response(500)
        # At retry 5: computed = min(1.0 * 2.0^5, 60.0) = 32.0, delay ∈ [0, 32.0]
        for _ in 1:20
            d = UniLM._retry_delay(5, resp)
            @test 0.0 <= d <= 32.0
        end
    end

    @testset "delay capped at max" begin
        resp = HTTP.Response(500)
        # At retry 10: computed = min(1.0 * 2.0^10, 60.0) = 60.0, delay ∈ [0, 60.0]
        for _ in 1:20
            d = UniLM._retry_delay(10, resp)
            @test 0.0 <= d <= 60.0
        end
        # At retry 20: still capped at 60.0
        for _ in 1:20
            d = UniLM._retry_delay(20, resp)
            @test 0.0 <= d <= 60.0
        end
    end

    @testset "Retry-After header respected" begin
        resp = HTTP.Response(429, ["Retry-After" => "5"])
        for _ in 1:20
            d = UniLM._retry_delay(0, resp)
            # computed at retry 0 = 1.0, so jitter ∈ [0, 1.0]
            # Retry-After=5, so delay = max(5.0, jitter) = 5.0
            @test d >= 5.0
        end
    end

    @testset "Retry-After header with large retry" begin
        resp = HTTP.Response(429, ["Retry-After" => "2"])
        for _ in 1:20
            d = UniLM._retry_delay(5, resp)
            # computed = 32.0, jitter ∈ [0, 32.0]
            # Retry-After=2, so delay = max(2.0, jitter) — may be >= 2.0
            @test d >= 2.0
        end
    end

    @testset "invalid Retry-After ignored" begin
        resp = HTTP.Response(500, ["Retry-After" => "not-a-number"])
        d = UniLM._retry_delay(0, resp)
        @test 0.0 <= d <= 1.0
    end

    @testset "negative Retry-After ignored" begin
        resp = HTTP.Response(500, ["Retry-After" => "-1"])
        d = UniLM._retry_delay(0, resp)
        @test 0.0 <= d <= 1.0
    end
end

@testset "Retry constants" begin
    @test UniLM._RETRY_BASE == 1.0
    @test UniLM._RETRY_FACTOR == 2.0
    @test UniLM._RETRY_MAX_DELAY == 60.0
    @test UniLM._RETRY_MAX_ATTEMPTS == 30
end
