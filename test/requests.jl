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
        # Ensure env var is set
        if haskey(ENV, "OPENAI_API_KEY")
            headers = UniLM.auth_header(UniLM.OPENAIServiceEndpoint)
            @test length(headers) == 2
            @test headers[1][1] == "Authorization"
            @test startswith(headers[1][2], "Bearer ")
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
        m = UniLM.extract_message(resp)
        @test m.role == UniLM.RoleAssistant
        @test m.content == "Hello there!"
        @test m.finish_reason == UniLM.STOP
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
        m = UniLM.extract_message(resp)
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
        m = UniLM.extract_message(resp)
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
        m = UniLM.extract_message(resp)
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
        m = UniLM.extract_message(resp)
        @test length(m.tool_calls) == 2
        @test m.tool_calls[1].func.name == "fn1"
        @test m.tool_calls[2].func.name == "fn2"
    end
end

@testset "_parse_chunk" begin
    @testset "normal content chunk" begin
        chunk = """data: {"id":"chatcmpl-1","object":"chat.completion.chunk","created":1,"model":"gpt-4o","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}"""
        iobuff = IOBuffer()
        failbuff = IOBuffer()
        result = UniLM._parse_chunk(chunk, iobuff, failbuff)
        @test result.eos == false
        @test String(take!(iobuff)) == "Hello"
        @test isempty(take!(failbuff))
    end

    @testset "done chunk" begin
        chunk = "data: [DONE]"
        iobuff = IOBuffer()
        failbuff = IOBuffer()
        result = UniLM._parse_chunk(chunk, iobuff, failbuff)
        @test result.eos == true
    end

    @testset "stop finish_reason" begin
        chunk = """data: {"id":"chatcmpl-1","object":"chat.completion.chunk","created":1,"model":"gpt-4o","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}"""
        iobuff = IOBuffer()
        failbuff = IOBuffer()
        result = UniLM._parse_chunk(chunk, iobuff, failbuff)
        @test result.eos == true
        @test isempty(take!(iobuff))  # no content from stop signal
    end

    @testset "empty chunk" begin
        iobuff = IOBuffer()
        failbuff = IOBuffer()
        result = UniLM._parse_chunk("", iobuff, failbuff)
        @test result.eos == false
    end

    @testset "whitespace-only chunk" begin
        iobuff = IOBuffer()
        failbuff = IOBuffer()
        result = UniLM._parse_chunk("   \n  \n  ", iobuff, failbuff)
        @test result.eos == false
    end

    @testset "multi-line chunk with content then DONE" begin
        chunk = """data: {"id":"chatcmpl-1","object":"chat.completion.chunk","created":1,"model":"gpt-4o","choices":[{"index":0,"delta":{"content":" world"},"finish_reason":null}]}

data: [DONE]"""
        iobuff = IOBuffer()
        failbuff = IOBuffer()
        result = UniLM._parse_chunk(chunk, iobuff, failbuff)
        @test result.eos == true
        @test String(take!(iobuff)) == " world"
    end

    @testset "malformed JSON goes to failbuff" begin
        chunk = "data: {invalid json"
        iobuff = IOBuffer()
        failbuff = IOBuffer()
        result = UniLM._parse_chunk(chunk, iobuff, failbuff)
        @test result.eos == false
        @test !isempty(take!(failbuff))
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
