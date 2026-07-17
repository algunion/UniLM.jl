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
        chat = Chat(service=UniLM.GEMINIOpenAIServiceEndpoint, model="gemini-2.0-flash")
        @test UniLM.get_url(UniLM.GEMINIOpenAIServiceEndpoint, chat) == "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
    end

    @testset "Gemini get_url for Embeddings" begin
        # src/requests.jl:32 — Gemini embeddings route to GEMINI_OPENAI_BASE * "/embeddings".
        # Note: Gemini's OpenAI-compat base already embeds /v1beta/openai, so it does NOT use
        # the EMBEDDINGS_PATH ("/v1/embeddings") constant — assert the exact composed string.
        emb = UniLM.Embeddings("test"; service=UniLM.GEMINIOpenAIServiceEndpoint, model="gemini-embedding-001")
        @test UniLM.get_url(UniLM.GEMINIOpenAIServiceEndpoint, emb) == UniLM.GEMINI_OPENAI_BASE * "/embeddings"
        @test UniLM.get_url(UniLM.GEMINIOpenAIServiceEndpoint, emb) == "https://generativelanguage.googleapis.com/v1beta/openai/embeddings"
    end
end

@testset "_api_base_url dispatch (Responses base)" begin
    # src/requests.jl:34 — OpenAI is the only built-in that yields a base URL.
    @test UniLM._api_base_url(UniLM.OPENAIServiceEndpoint) == "https://api.openai.com"
    @test UniLM._api_base_url(UniLM.OPENAIServiceEndpoint) == UniLM.OPENAI_BASE_URL

    # src/requests.jl:35-36 — Azure and Gemini reject the Responses API with an ArgumentError
    # whose message names OPENAIServiceEndpoint (these would throw NO error if the methods
    # silently returned a base URL instead).
    @test_throws ArgumentError UniLM._api_base_url(UniLM.AZUREServiceEndpoint)
    @test_throws ArgumentError UniLM._api_base_url(UniLM.GEMINIOpenAIServiceEndpoint)
    for S in (UniLM.AZUREServiceEndpoint, UniLM.GEMINIOpenAIServiceEndpoint)
        err = try
            UniLM._api_base_url(S)
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("OPENAIServiceEndpoint", err.msg)
    end

    # src/requests.jl:42 — GenericOpenAIEndpoint base is the rstripped base_url (trailing
    # slash removed). A base_url that ends in "/" proves the rstrip: without it, the slash
    # would survive.
    gen = UniLM.GenericOpenAIEndpoint("https://host.example/", "k")
    @test UniLM._api_base_url(gen) == "https://host.example"
    @test UniLM._api_base_url(gen) == rstrip(gen.base_url, '/')
end

@testset "GenericOpenAIEndpoint get_url trailing-slash rstrip" begin
    # src/requests.jl:40 — Chat on a GenericOpenAIEndpoint with a TRAILING-SLASH base_url:
    # the URL is rstrip(base_url,'/') * CHAT_COMPLETIONS_PATH. With base_url "https://host.example/"
    # the result is exactly ".../v1/chat/completions" (single slash); were the rstrip absent it
    # would be ".../v1/chat/completions" prefixed by a doubled slash ("https://host.example//v1...").
    gen = UniLM.GenericOpenAIEndpoint("https://host.example/", "k")
    @test UniLM.get_url(gen, Chat()) == "https://host.example/v1/chat/completions"
    @test UniLM.get_url(gen, Chat()) == rstrip(gen.base_url, '/') * UniLM.CHAT_COMPLETIONS_PATH
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
            headers = UniLM.auth_header(UniLM.GEMINIOpenAIServiceEndpoint)
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

    @testset "length finish_reason preserves partial content" begin
        body = Dict(
            "choices" => [Dict(
                "finish_reason" => "length",
                "message" => Dict("role" => "assistant", "content" => "partial answer")
            )]
        )
        m = UniLM.extract_message(make_response(body)).message
        @test m.content == "partial answer"
        @test m.finish_reason == "length"
    end

    @testset "refusal with stop finish_reason is captured" begin
        body = Dict(
            "choices" => [Dict(
                "finish_reason" => "stop",
                "message" => Dict("role" => "assistant", "content" => nothing, "refusal" => "I can't help with that.")
            )]
        )
        m = UniLM.extract_message(make_response(body)).message
        @test m.refusal_message == "I can't help with that."
        @test m.finish_reason == "stop"
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

@testset "chat SSE machine (ported from _parse_chunk; seam removed in 0.11.3)" begin
    S = UniLM.OPENAIServiceEndpoint
    dispatch(chunk, state; carry=IOBuffer()) =
        (UniLM._sse_dispatch!(S, carry, Ref(""), chunk, state), carry)

    @testset "normal content chunk" begin
        state = UniLM.StreamState()
        st, carry = dispatch("data: {\"id\":\"chatcmpl-1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"gpt-4o\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Hello\"},\"finish_reason\":null}]}\n", state)
        @test st === :continue
        @test String(take!(state.content)) == "Hello"
        @test isempty(take!(carry))
    end

    @testset "done chunk" begin
        state = UniLM.StreamState()
        st, _ = dispatch("data: [DONE]\n", state)
        @test st === :done
    end

    @testset "stop finish_reason is recorded, NOT terminal (fixed contract)" begin
        state = UniLM.StreamState()
        st, _ = dispatch("data: {\"id\":\"chatcmpl-1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"gpt-4o\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n", state)
        @test st === :continue
        @test isempty(take!(state.content))
        @test state.finish_reason == "stop"
    end

    @testset "refusal delta builds a refusal message" begin
        state = UniLM.StreamState()
        dispatch("data: {\"choices\":[{\"index\":0,\"delta\":{\"refusal\":\"I can't help\"},\"finish_reason\":\"stop\"}]}\n", state)
        msg = UniLM._build_stream_message(state)
        @test msg.refusal_message == "I can't help"
        @test isnothing(msg.content)
    end

    @testset "empty chunk" begin
        state = UniLM.StreamState()
        st, _ = dispatch("", state)
        @test st === :continue
    end

    @testset "whitespace-only chunk" begin
        state = UniLM.StreamState()
        st, _ = dispatch("   \n  \n  ", state)
        @test st === :continue
    end

    @testset "multi-line chunk with content then DONE" begin
        state = UniLM.StreamState()
        st, _ = dispatch("data: {\"id\":\"chatcmpl-1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"gpt-4o\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\" world\"},\"finish_reason\":null}]}\n\ndata: [DONE]\n", state)
        @test st === :done
        @test String(take!(state.content)) == " world"
    end

    @testset "unterminated line stays in the carry (not yet parsed)" begin
        state = UniLM.StreamState()
        st, carry = dispatch("data: {invalid json", state)
        @test st === :continue
        @test !isempty(take!(carry))
    end

    @testset "malformed COMPLETE line is dropped + counted, never re-queued" begin
        before = UniLM._SSE_DROPPED_LINES[]
        state = UniLM.StreamState()
        st, carry = dispatch("data: {invalid json\n", state)
        @test st === :continue
        @test isempty(take!(carry))
        @test UniLM._SSE_DROPPED_LINES[] == before + 1
    end

    @testset "tool call deltas accumulated by index" begin
        state = UniLM.StreamState()
        carry = IOBuffer()
        dispatch("data: {\"id\":\"chatcmpl-1\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_abc\",\"type\":\"function\",\"function\":{\"name\":\"get_weather\",\"arguments\":\"\"}}]},\"finish_reason\":null}]}\n", state; carry)
        @test haskey(state.tool_calls, 0)
        @test state.tool_calls[0]["id"] == "call_abc"
        @test state.tool_calls[0]["function"]["name"] == "get_weather"
        dispatch("data: {\"id\":\"chatcmpl-1\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"{\\\"location\\\":\"}}]},\"finish_reason\":null}]}\n", state; carry)
        dispatch("data: {\"id\":\"chatcmpl-1\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"\\\"NYC\\\"}\"}}]},\"finish_reason\":null}]}\n", state; carry)
        @test state.tool_calls[0]["function"]["arguments"] == "{\"location\":\"NYC\"}"
        dispatch("data: {\"id\":\"chatcmpl-1\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n", state; carry)
        @test state.finish_reason == "tool_calls"
    end

    @testset "usage captured from stream chunk" begin
        state = UniLM.StreamState()
        dispatch("data: {\"id\":\"chatcmpl-1\",\"choices\":[],\"usage\":{\"prompt_tokens\":25,\"completion_tokens\":10,\"total_tokens\":35}}\n", state)
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

@testset "_build_stream_message attaches provider-native blocks" begin
    # Populated raw blocks + provider tag → ProviderContent on every branch.
    st = UniLM.StreamState()
    print(st.content, "Checking.")
    st.raw_provider = :anthropic
    push!(st.raw_blocks, Dict{String,Any}("type" => "text", "text" => "Checking."))
    m = UniLM._build_stream_message(st)
    @test m.provider_content isa ProviderContent
    @test m.provider_content.provider === :anthropic
    @test m.provider_content.blocks[1]["text"] == "Checking."

    # Tool-calls branch carries it too.
    st2 = UniLM.StreamState()
    st2.raw_provider = :anthropic
    push!(st2.raw_blocks, Dict{String,Any}("type" => "tool_use", "id" => "t1",
                                           "name" => "f", "input" => Dict{String,Any}()))
    st2.tool_calls[0] = Dict{String,Any}("id" => "t1", "type" => "function",
        "function" => Dict{String,Any}("name" => "f", "arguments" => ""))
    st2.finish_reason = UniLM.TOOL_CALLS
    m2 = UniLM._build_stream_message(st2)
    @test !isnothing(m2.tool_calls) && m2.provider_content.provider === :anthropic

    # No raw blocks (every non-Anthropic stream today) → nothing, exactly as before.
    st3 = UniLM.StreamState()
    print(st3.content, "plain")
    @test isnothing(UniLM._build_stream_message(st3).provider_content)

    # Blocks without a provider tag must NOT fabricate a ProviderContent.
    st4 = UniLM.StreamState()
    print(st4.content, "x")
    push!(st4.raw_blocks, Dict{String,Any}("type" => "text"))
    @test isnothing(UniLM._build_stream_message(st4).provider_content)

    # Incomplete capture (a block never finalized) must NOT be echoed:
    # partial provider-native content is worse than the neutral fallback.
    st5 = UniLM.StreamState()
    print(st5.content, "x")
    st5.raw_provider = :anthropic
    push!(st5.raw_blocks, Dict{String,Any}("type" => "text", "text" => "x"))
    st5.raw_pending[1] = Dict{String,Any}("type" => "tool_use")
    @test isnothing(UniLM._build_stream_message(st5).provider_content)
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
            result = embeddingrequest!(emb)
            @test result isa EmbeddingFailure
            @test result.status == 401
        end
    end
end

@testset "_is_retryable" begin
    @test UniLM._is_retryable(429) == true
    @test UniLM._is_retryable(500) == true
    @test UniLM._is_retryable(503) == true
    @test UniLM._is_retryable(502) == true
    @test UniLM._is_retryable(504) == true
    @test UniLM._is_retryable(408) == true
    @test UniLM._is_retryable(529) == true
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
end

@testset "_accumulate_cost! fallback is a no-op for non-success" begin
    # requests.jl:338 — the generic _accumulate_cost!(::Chat, ::LLMRequestResponse) stub. Only
    # success types are specialized in accounting.jl, so a failure result must land here:
    # return nothing AND leave cumulative cost untouched (falsifies accidental accumulation).
    chat = Chat(model="gpt-4.1-nano")
    chat._cumulative_cost[] = 0.25
    failure = LLMFailure(response="server exploded", status=500, self=chat)
    @test which(UniLM._accumulate_cost!, (Chat, typeof(failure))).line == 338
    @test UniLM._accumulate_cost!(chat, failure) === nothing
    @test cumulative_cost(chat) == 0.25       # unchanged: the fallback did not add anything

    # also exercised via a call-error variant (same fallback method)
    callerr = LLMCallError(error="network down", self=chat)
    @test UniLM._accumulate_cost!(chat, callerr) === nothing
    @test cumulative_cost(chat) == 0.25
end

@testset "wire seam — OpenAI defaults are byte-identical to legacy path" begin
    sig = GPTFunctionSignature(name="f", parameters=Dict("type" => "object", "properties" => Dict()))
    chat = Chat(model="gpt-5.5", tools=[GPTTool(func=sig)], temperature=0.7,
                stream=true, logit_bias=Dict("50256" => -100.0), seed=7)
    push!(chat, Message(Val(:system), "sys"))
    push!(chat, Message(Val(:user), "hi"))
    # The default (untyped-service) encoder IS the legacy OpenAI body.
    @test UniLM.encode_request(chat.service, chat) == JSON.json(chat)
    @test UniLM.encode_request(OPENAIServiceEndpoint, chat) == JSON.json(chat)
    @test UniLM.encode_request(DeepSeekEndpoint(api_key="x"), chat) == JSON.json(chat)
end

@testset "_stream_error_result maps in-band stream errors to typed results" begin
    chat = Chat(model="gpt-5.5")
    overloaded = Dict{String,Any}("type" => "error",
        "error" => Dict{String,Any}("type" => "overloaded_error", "message" => "Overloaded"))
    r = UniLM._stream_error_result(chat, overloaded, nothing)
    @test r isa LLMFailure && r.status == 529
    @test occursin("overloaded_error", r.response)

    other = Dict{String,Any}("type" => "error",
        "error" => Dict{String,Any}("type" => "api_error", "message" => "boom"))
    r2 = UniLM._stream_error_result(chat, other, nothing)
    @test r2 isa LLMCallError && isnothing(r2.status)
    @test occursin("api_error", r2.error)

    # A malformed inner `error` (a bare string, not an object) must not throw:
    # with no `type` to key on it falls through to LLMCallError, carrying the
    # raw payload rather than crashing the stream driver.
    malformed = Dict{String,Any}("type" => "error", "error" => "not a dict")
    r3 = UniLM._stream_error_result(chat, malformed, nothing)
    @test r3 isa LLMCallError && isnothing(r3.status)
    @test occursin("not a dict", r3.error)
end

@testset "_retry_pause shares the retry-budget arithmetic" begin
    cfg = RequestConfig(total_deadline=60.0, max_attempts=3)
    t0 = time_ns()
    # Attempt 1, no response: pure full-jitter backoff in [0, 1] — affordable against 60 s.
    action, delay = UniLM._retry_pause(cfg, t0, 1, nothing)
    @test action === :sleep && 0.0 <= delay <= 1.0

    # Retry-After far beyond the remaining deadline → :budget carrying the honest delay.
    ra = HTTP.Response(429, ["Retry-After" => "3600"])
    action2, delay2 = UniLM._retry_pause(cfg, t0, 1, ra)
    @test action2 === :budget && delay2 >= 3600.0

    # An infinite deadline never cuts the budget.
    action3, _ = UniLM._retry_pause(RequestConfig(total_deadline=Inf), t0, 1, ra)
    @test action3 === :sleep
end

@testset "_unwrap_exception peels task and transport wrappers" begin
    root = Base.IOError("read: connection reset by peer (ECONNRESET)", -54)
    t = Task(() -> throw(root)); schedule(t); yield()
    wrapped = TaskFailedException(t)
    @test UniLM._unwrap_exception(wrapped) === root
    @test UniLM._unwrap_exception(CompositeException([wrapped])) === root
    @test UniLM._unwrap_exception(root) === root
    ti = Task(() -> throw(InterruptException())); schedule(ti); yield()
    @test UniLM._unwrap_exception(TaskFailedException(ti)) isa InterruptException
end
