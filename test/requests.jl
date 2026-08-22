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

    # Fail-loud total coverage: a non-OpenAI-wire endpoint (a native provider) has no
    # platform/Responses base URL, so it raises ArgumentError rather than a bare MethodError.
    # This keeps the abstract _api_base_url(service::ServiceEndpointSpec) dispatch total.
    @test_throws ArgumentError UniLM._api_base_url(UniLM.ANTHROPICServiceEndpoint)
    @test_throws ArgumentError UniLM._api_base_url(UniLM.GEMINIServiceEndpoint)

    # GenericOpenAIEndpoint base is the rstripped base_url (trailing slash removed). A
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

    @testset "an empty turn is reported empty, not filled in" begin
        # Truthful-empty contract: a well-formed choice whose content is null is a
        # real turn the model produced (a reasoning model can burn the whole
        # completion budget on thought tokens and stop at "length"). The decoder
        # reports it as the empty turn it is. Substituting prose used to put text
        # nobody generated into the reply AND into the next request's history.
        body = Dict(
            "choices" => [Dict(
                "finish_reason" => "length",
                "message" => Dict("role" => "assistant", "content" => nothing)
            )]
        )
        resp = make_response(body)
        m = UniLM.extract_message(resp).message
        @test m.role == UniLM.RoleAssistant
        @test m.content == ""
        @test !occursin("No response from the model", something(m.content, ""))
        @test m.finish_reason == "length"
    end

    @testset "a tool-only turn keeps its calls whatever the finish_reason" begin
        # Some providers close a tool-only turn with "stop" rather than "tool_calls".
        # That used to land in the fallback branch, which fabricated text AND dropped
        # the calls entirely.
        body = Dict(
            "choices" => [Dict(
                "finish_reason" => "stop",
                "message" => Dict("role" => "assistant", "content" => nothing,
                    "tool_calls" => [Dict("id" => "call_1",
                        "function" => Dict("name" => "get_weather",
                                           "arguments" => "{\"city\":\"Cluj\"}"))]))]
        )
        m = UniLM.extract_message(make_response(body)).message
        @test isnothing(m.content)
        @test length(m.tool_calls) == 1
        @test m.tool_calls[1].id == "call_1"
        @test m.tool_calls[1].func.name == "get_weather"
        @test m.tool_calls[1].func.arguments["city"] == "Cluj"
        @test m.finish_reason == "stop"
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

    @testset "registry survives concurrent readers and writers" begin
        # The registry is read while building every Azure request URL. `Dict` is not
        # concurrency-safe: a write that rehashes reallocates the arrays a reader is
        # walking, so an unsynchronized lookup can miss a present key or throw.
        added = ["race-model-$i" for i in 1:64]
        try
            UniLM.add_azure_deploy_name!("snapshot-model", "snap-deploy")
            @test UniLM._azure_deployment_path("snapshot-model") == "/openai/deployments/snap-deploy"
            @test_throws KeyError UniLM._azure_deployment_path("never-registered-model")

            bad = Threads.Atomic{Int}(0)
            @sync begin
                Threads.@spawn for k in added
                    UniLM.add_azure_deploy_name!(k, "deploy-" * k)   # forces repeated rehashes
                end
                for _ in 1:3
                    Threads.@spawn for _ in 1:5000
                        ok = try
                            UniLM._azure_deployment_path("snapshot-model") == "/openai/deployments/snap-deploy"
                        catch
                            false                                     # a torn read throws
                        end
                        ok || Threads.atomic_add!(bad, 1)
                    end
                end
            end
            @test bad[] == 0
            @test all(UniLM._azure_deployment_path(k) == "/openai/deployments/deploy-" * k for k in added)
        finally
            delete!(UniLM._MODEL_ENDPOINTS_AZURE_OPENAI, "snapshot-model")
            foreach(k -> delete!(UniLM._MODEL_ENDPOINTS_AZURE_OPENAI, k), added)
        end
    end
end

@testset "Azure deployment registry rejects a malformed entry" begin
    # `add_azure_deploy_name!` is the registry's only writer and always stores the
    # assembled `/openai/deployments/<name>` path, so an entry without that prefix
    # can only come from a write straight into the dict. Recovering the name by
    # chopping a prefix that is not there would encode the WHOLE entry and
    # re-prefix it, inventing a deployment path nobody registered. The reader
    # names the bad entry instead of silently repairing it.
    try
        UniLM._MODEL_ENDPOINTS_AZURE_OPENAI["malformed-entry-model"] = "my-deploy_01"
        @test_throws ArgumentError UniLM._azure_deployment_path("malformed-entry-model")
    finally
        delete!(UniLM._MODEL_ENDPOINTS_AZURE_OPENAI, "malformed-entry-model")
    end
end

@testset "Azure deployment path encodes the deployment name" begin
    # A deployment name is one path segment of caller data: a `/`, `?` or `#`
    # inside it must not add segments or open a query string on the wire. The
    # registry keeps the name as registered; encoding happens where the path is
    # built, so what callers read back from the registry is unchanged.
    try
        UniLM.add_azure_deploy_name!("hostile-deploy-model", "d n/../x?y=1#f")
        @test UniLM._azure_deployment_path("hostile-deploy-model") ==
              "/openai/deployments/d%20n%2F..%2Fx%3Fy%3D1%23f"

        # A real-shaped name draws from the unreserved set: byte-identical no-op.
        UniLM.add_azure_deploy_name!("plain-deploy-model", "my-deploy_01")
        @test UniLM._azure_deployment_path("plain-deploy-model") ==
              "/openai/deployments/my-deploy_01"

        # End to end: the encoded segment reaches the URL, while the operator's
        # own base URL and api-version stay literal.
        withenv(
            "AZURE_OPENAI_BASE_URL" => "https://myazure.openai.azure.com",
            "AZURE_OPENAI_API_VERSION" => "2024-02-01",
        ) do
            url = UniLM.get_url(UniLM.AZUREServiceEndpoint,
                                Chat(service=UniLM.AZUREServiceEndpoint, model="hostile-deploy-model"))
            @test url == "https://myazure.openai.azure.com/openai/deployments/" *
                         "d%20n%2F..%2Fx%3Fy%3D1%23f/chat/completions?api-version=2024-02-01"
        end
    finally
        delete!(UniLM._MODEL_ENDPOINTS_AZURE_OPENAI, "hostile-deploy-model")
        delete!(UniLM._MODEL_ENDPOINTS_AZURE_OPENAI, "plain-deploy-model")
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

# IMF-fixdate for a UTC instant, derived independently of the parser under test
# (civil-from-days; the parser computes the inverse). Weekday from the epoch:
# 1970-01-01 was a Thursday.
const _IMF_WD = ("Thu", "Fri", "Sat", "Sun", "Mon", "Tue", "Wed")
const _IMF_MO = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")
function _imf_fixdate(t::Real)::String
    days, secs = fldmod(floor(Int, t), 86400)
    z = days + 719468
    era = fld(z, 146097)
    doe = z - era * 146097
    yoe = div(doe - div(doe, 1460) + div(doe, 36524) - div(doe, 146096), 365)
    doy = doe - (365 * yoe + div(yoe, 4) - div(yoe, 100))
    mp = div(5 * doy + 2, 153)
    d = doy - div(153 * mp + 2, 5) + 1
    m = mp < 10 ? mp + 3 : mp - 9
    y = yoe + era * 400 + (m <= 2)
    h, r = fldmod(secs, 3600)
    mi, s = fldmod(r, 60)
    return string(_IMF_WD[mod(days, 7)+1], ", ", lpad(d, 2, '0'), " ", _IMF_MO[m], " ", y, " ",
                  lpad(h, 2, '0'), ":", lpad(mi, 2, '0'), ":", lpad(s, 2, '0'), " GMT")
end

@testset "Retry-After accepts both RFC 7231 forms" begin
    # RFC 7231 allows delta-seconds OR an IMF-fixdate (always GMT). Ignoring the
    # date form under-waits during a 429 storm: at retry 0 the jitter is capped at
    # _RETRY_BASE (1 s), so any delay above that can only come from the header.
    # The fixdate floors to whole seconds and the runner can stall between
    # constructing it and evaluating the delay, so the gap must dwarf both: a 6 s
    # header with the discriminating bound at the 1 s jitter cap tolerates the
    # floor plus ~4 s of stall without ever passing on jitter alone.
    ahead = HTTP.Response(429, ["Retry-After" => _imf_fixdate(time() + 6)])
    @test UniLM._retry_delay(0, ahead) > 1.0

    # Ground truth for the date math: RFC 7231's own example instant.
    @test _imf_fixdate(784111777) == "Sun, 06 Nov 1994 08:49:37 GMT"
    @test UniLM._utc_epoch_seconds(1994, 11, 6, 8, 49, 37) == 784111777.0

    # Parser: future date → the remaining gap; past date → 0; garbage → nothing
    # (the caller then keeps its default backoff — a server must never make us throw).
    @test 2.0 <= UniLM._retry_after_seconds(ahead) <= 6.0
    @test UniLM._retry_after_seconds(HTTP.Response(429, ["Retry-After" => "Sun, 06 Nov 1994 08:49:37 GMT"])) == 0.0
    @test UniLM._retry_after_seconds(HTTP.Response(429, ["Retry-After" => "next tuesday"])) === nothing
    @test UniLM._retry_after_seconds(HTTP.Response(429, ["Retry-After" => "Sun, 06 Nov 1994 08:49:37 PST"])) === nothing
    @test UniLM._retry_after_seconds(HTTP.Response(429)) === nothing
    @test UniLM._retry_after_seconds(HTTP.Response(429, ["Retry-After" => "7"])) == 7.0

    # A past date and garbage both fall back to jitter alone.
    past = HTTP.Response(429, ["Retry-After" => "Sun, 06 Nov 1994 08:49:37 GMT"])
    @test 0.0 <= UniLM._retry_delay(0, past) <= 1.0

    # The date form goes through the SAME budget arithmetic as the seconds form:
    # a wait beyond the remaining deadline is reported as :budget, never slept.
    action, delay = UniLM._retry_pause(RequestConfig(total_deadline=1.0), time_ns(), 1, ahead)
    @test action === :budget && delay > 2.0

    # Reading a capture asserts that the group actually matched. Every group of the
    # fixdate pattern is mandatory, so this branch is unreachable through the header
    # parser — a pattern with an optional group is the only way to reach it, and it
    # must name the group loudly rather than hand `nothing` on to the date math.
    @test UniLM._fixdate_int(match(r"^(\d+)$", "42"), 1) == 42
    @test_throws ArgumentError UniLM._fixdate_int(match(r"^x(\d+)?$", "x"), 1)
end

@testset "Retry constants" begin
    @test UniLM._RETRY_BASE == 1.0
    @test UniLM._RETRY_FACTOR == 2.0
    @test UniLM._RETRY_MAX_DELAY == 60.0
end

@testset "_accumulate_cost! fallback is a no-op for non-success" begin
    # requests.jl:599 — the generic _accumulate_cost!(::Chat, ::LLMRequestResponse) stub. Only
    # success types are specialized in accounting.jl, so a failure result must land here:
    # return nothing AND leave cumulative cost untouched (falsifies accidental accumulation).
    # The line is a locator, not the contract: re-point it (here and in the note above)
    # whenever code is added above the stub.
    chat = Chat(model="gpt-4.1-nano")
    chat._cumulative_cost[] = 0.25
    failure = LLMFailure(response="server exploded", status=500, self=chat)
    @test which(UniLM._accumulate_cost!, (Chat, typeof(failure))).line == 599
    @test UniLM._accumulate_cost!(chat, failure) === nothing
    @test cumulative_cost(chat) == 0.25       # unchanged: the fallback did not add anything

    # also exercised via a call-error variant (same fallback method)
    callerr = LLMCallError(error="network down", self=chat)
    @test UniLM._accumulate_cost!(chat, callerr) === nothing
    @test cumulative_cost(chat) == 0.25
end

@testset "wire seam — OpenAI defaults are byte-identical to legacy path" begin
    sig = FunctionSignature(name="f", parameters=Dict("type" => "object", "properties" => Dict()))
    chat = Chat(model="gpt-5.5", tools=[Tool(func=sig)], temperature=0.7,
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

@testset "_classify_stream_timeout decides :stream_idle from the armed-timer set" begin
    # Constructed exceptions + directly-constructed guard states: no servers, no
    # timers, no timing — the classification is a pure function of (exception,
    # guard state, config), so the phase can be pinned deterministically.
    cfg = RequestConfig(stream_idle_timeout = 1.0)
    t0 = time_ns()

    # A FIRED guard is a breach on both majors — the caught error is the echo of
    # our own close — and elapsed reports the frozen byte GAP, not call time.
    fired = UniLM._IdleGuard(:fired, time_ns(), 1.23, 1.0, nothing)
    to = UniLM._classify_stream_timeout(Base.IOError("read: connection reset", 0), fired, cfg, t0)
    @test to isa UniLM.UniLMTimeout
    @test to.phase === :stream_idle
    @test to.elapsed == 1.23
    @test to.limit == 1.0
    @test !UniLM._retryable_exception(to)   # :stream_idle is never blanket-retried

    if UniLM._HTTP_MAJOR2
        # 2.x: with the read-idle fast path armed (finite idle bound), a native
        # non-connect TimeoutError IS the read-idle timer by elimination — the
        # streaming seam arms no other native non-connect timer. That holds even
        # BEFORE the idle guard exists (guard === nothing): HTTP 2.x also bounds
        # the response-header wait by read_idle_timeout, so the breach can land
        # pre-first-byte and the phase must not depend on guard arming order.
        e_req = HTTP.TimeoutError("request", Int64(1_000_000_000), Int64(0))
        pre = UniLM._classify_stream_timeout(e_req, nothing, cfg, t0)
        @test pre isa UniLM.UniLMTimeout
        @test pre.phase === :stream_idle
        @test pre.limit == 1.0
        @test !UniLM._retryable_exception(pre)
        # Same decision with an armed-but-unfired guard, and chain-walked out of
        # a task wrapper (how a spawned attempt actually delivers it).
        tt = Task(() -> throw(e_req)); schedule(tt); yield()
        armed = UniLM._IdleGuard(:armed, time_ns(), 0.0, 1.0, nothing)
        mid = UniLM._classify_stream_timeout(TaskFailedException(tt), armed, cfg, t0)
        @test mid isa UniLM.UniLMTimeout && mid.phase === :stream_idle
        # Connect/TLS-labeled native timeouts are NOT idle breaches: they fall
        # through to the :connect mapping and remain retryable by phase.
        e_conn = HTTP.TimeoutError("connect", Int64(1_000_000_000), Int64(0))
        e_tls = HTTP.TimeoutError("tls_handshake", Int64(1_000_000_000), Int64(0))
        @test UniLM._classify_stream_timeout(e_conn, nothing, cfg, t0) === nothing
        @test UniLM._classify_stream_timeout(e_tls, nothing, cfg, t0) === nothing
        # Idle disabled (Inf): the seam arms no read-idle timer, so a non-connect
        # native timeout cannot be attributed to it.
        @test UniLM._classify_stream_timeout(e_req, nothing,
            RequestConfig(stream_idle_timeout = Inf), t0) === nothing
    else
        # 1.x: streams arm NO native read timer, so a native TimeoutError alone is
        # never an idle breach — only the guard's own close (the fired branch) is.
        @test UniLM._classify_stream_timeout(HTTP.TimeoutError(5), nothing, cfg, t0) === nothing
        @test UniLM._classify_stream_timeout(HTTP.TimeoutError(5),
            UniLM._IdleGuard(:armed, time_ns(), 0.0, 1.0, nothing), cfg, t0) === nothing
    end

    # Non-timeout exceptions without a fired guard are not breaches (fallthrough
    # to transport classification), whether or not the guard is armed.
    @test UniLM._classify_stream_timeout(Base.IOError("boom", 0), nothing, cfg, t0) === nothing
    @test UniLM._classify_stream_timeout(Base.IOError("boom", 0),
        UniLM._IdleGuard(:armed, time_ns(), 0.0, 1.0, nothing), cfg, t0) === nothing
end

@testset "_with_recorded_deadline: the typed bound survives library teardown noise" begin
    # Constructed exceptions, limit=Inf (no timer): the recording contract is a
    # pure function of what escapes the deadline block, so it pins without
    # timing. The driver catches consume the slot with the precedence rule
    # "recorded typed cause wins over teardown noise" — on the 1.x major,
    # HTTP.jl's cleanup of a bound-closed socket can raise EPIPE/reset while
    # the typed UniLMTimeout is unwinding, and the replacement is what escapes
    # HTTP.open; the slot is what restores the typed cause.
    slotT() = Ref{Union{Nothing,UniLM.UniLMTimeout}}(nothing)

    # A UniLMTimeout escaping the block is recorded AND rethrown unchanged
    # (identity: the very exception, not a copy).
    slot = slotT()
    to = UniLM.UniLMTimeout(:request, 1.0, 1.0)
    caught = try
        UniLM._with_recorded_deadline(() -> throw(to), () -> nothing, Inf, :request, slot)
        nothing
    catch e
        e
    end
    @test caught === to
    @test slot[] === to
    # Restoration shape the driver applies when teardown noise displaced the
    # typed exception in flight: the recorded bound, not the noise, is surfaced
    # — and the noise itself is no idle breach (it cannot re-enter typed
    # classification through the byte-gap limb either).
    noise = Base.IOError("write: broken pipe (EPIPE)", -32)
    @test (slot[] !== nothing ? slot[] : noise) === to
    @test UniLM._classify_stream_timeout(noise, nothing,
        RequestConfig(stream_idle_timeout = 5.0), time_ns()) === nothing

    # A transport error with NO bound fired is NOT recorded: it must keep
    # surfacing as a transport failure (the connect-refusal contract).
    slot = slotT()
    caught = try
        UniLM._with_recorded_deadline(() -> throw(noise), () -> nothing, Inf, :request, slot)
        nothing
    catch e
        e
    end
    @test caught === noise
    @test slot[] === nothing

    # InterruptException rethrows unrecorded — user intent is never converted.
    slot = slotT()
    caught = try
        UniLM._with_recorded_deadline(() -> throw(InterruptException()), () -> nothing, Inf, :request, slot)
        nothing
    catch e
        e
    end
    @test caught isa InterruptException
    @test slot[] === nothing

    # Clean completion: value passes through, nothing recorded.
    slot = slotT()
    @test UniLM._with_recorded_deadline(() -> 42, () -> nothing, Inf, :request, slot) == 42
    @test slot[] === nothing
end

@testset "_with_recorded_deadline: a lost completion race surfaces the typed bound" begin
    # The timer can win the resolution CAS and close the guarded socket at the
    # very moment the guarded call returns: nothing throws, yet the resource is
    # gone. Left unreported, the next read raises a bare IOError on the closed
    # socket, which classifies as a RETRYABLE transport failure — an extra billed
    # wire attempt, with the request phase lost. Deterministic by construction:
    # completion at 0.4 s is strictly past the 0.1 s bound (the shape the
    # reported-mode race test in test/deadline.jl uses).
    slot = Ref{Union{Nothing,UniLM.UniLMTimeout}}(nothing)
    closed = Ref(0)
    caught = try
        UniLM._with_recorded_deadline(() -> (sleep(0.4); :survived),
                                      () -> closed[] += 1, 0.1, :request, slot)
    catch e
        e
    end
    @test caught isa UniLM.UniLMTimeout
    @test caught.phase === :request && caught.limit == 0.1
    @test slot[] === caught                   # recorded: the driver restores it over teardown noise
    @test closed[] == 1                       # the guard really did close the resource
    @test UniLM._retryable_exception(caught)  # a :request bound keeps its budgeted retry
    # A clean win is untouched: the value passes through and nothing is recorded.
    slot2 = Ref{Union{Nothing,UniLM.UniLMTimeout}}(nothing)
    @test UniLM._with_recorded_deadline(() -> :fast, () -> closed[] += 1, 5.0, :request, slot2) === :fast
    @test slot2[] === nothing
    @test closed[] == 1
end

@testset "stream finalize is at-most-once and records the terminal message" begin
    # Message assembly `take!`s the accumulation buffers, so a SECOND finalize
    # would deliver another terminal callback carrying an EMPTY message — and
    # commit that empty turn. The recorded-message slot is the structural guard:
    # a re-entry (a late byte-gap fire unwinding through the breach path after
    # the stream already ended) returns the recorded turn and fires nothing.
    state = UniLM.StreamState()
    print(state.content, "hello world")
    state.finish_reason = "stop"
    m = Ref{Union{Message,Nothing}}(nothing)
    seen = Message[]
    cb = (x, _) -> x isa Message && push!(seen, x)
    fin1 = UniLM._finalize_stream_message!(state, cb, nothing, Ref(false), m)
    @test fin1.msg.content == "hello world"
    @test m[] === fin1.msg                    # recorded before the terminal callback returns
    fin2 = UniLM._finalize_stream_message!(state, cb, nothing, Ref(false), m)
    @test fin2.msg === fin1.msg               # the recorded turn, never a rebuild from emptied buffers
    @test length(seen) == 1                   # exactly one terminal callback
end

@testset "teardown noise is distinguished from a real failure" begin
    # The predicate both stream drivers consult once a terminal result is
    # recorded. Connection-level errors, typed bounds, and an already classified
    # byte-gap breach are the teardown of an exchange that already finished: they
    # must neither re-POST a billed generation nor discard it. Anything else — a
    # throwing user callback, a bug in decoding — still surfaces.
    cfg = RequestConfig(stream_idle_timeout=1.0)
    t0 = time_ns()
    idle_breach = UniLM.UniLMTimeout(:stream_idle, 1.0, 1.0)
    @test UniLM._stream_teardown_noise(Base.IOError("read: connection reset", 0), nothing)
    @test UniLM._stream_teardown_noise(EOFError(), nothing)
    @test UniLM._stream_teardown_noise(Base.SystemError("read", 54), nothing)
    @test UniLM._stream_teardown_noise(UniLM.UniLMTimeout(:request, 1.0, 1.0), nothing)
    @test UniLM._stream_teardown_noise(ArgumentError("anything"), idle_breach)
    @test !UniLM._stream_teardown_noise(ArgumentError("callback bug"), nothing)

    # The breach argument is why this takes the CLASSIFIED result and not the
    # guard handle. Whichever timer wins the byte-gap race must read as teardown:
    # our own guard's close echoes as an IOError (transport-shaped anyway), but
    # the 2.x native read-idle timer surfaces as an HTTP.TimeoutError, which is
    # deliberately NEITHER transport-shaped NOR a UniLMTimeout — only the
    # classifier recognises it, and missing it discards a completed turn.
    fired = UniLM._IdleGuard(:fired, time_ns(), 1.0, 1.0, nothing)
    own_close = Base.IOError("read: connection reset", 0)
    @test UniLM._stream_teardown_noise(own_close,
        UniLM._classify_stream_timeout(own_close, fired, cfg, t0))
    if UniLM._HTTP_MAJOR2
        native = HTTP.TimeoutError("request", Int64(1_000_000_000), Int64(0))
        @test !UniLM._is_transport_error(native)          # not transport-shaped by design
        @test UniLM._stream_teardown_noise(native,
            UniLM._classify_stream_timeout(native, nothing, cfg, t0))
    end
end

@testset "a guard that fired turns a clean loop exit into the typed breach" begin
    # Closing the socket is how a guard unblocks a blocked read; when the close
    # lands while the driver is inside a user callback, the truncated read comes
    # back as a CLEAN EOF and the loop exits with NO exception to classify. The
    # exit path must therefore consult the guards itself, or a killed stream is
    # reported as a 200 with partial bytes.
    cfg = RequestConfig(stream_idle_timeout=2.0)
    empty_bound = Ref{Union{Nothing,UniLM.UniLMTimeout}}(nothing)
    fired = UniLM._IdleGuard(:fired, time_ns(), 1.75, 2.0, nothing)
    idle = UniLM._exit_breach(fired, empty_bound, cfg)
    @test idle isa UniLM.UniLMTimeout
    @test idle.phase === :stream_idle
    @test idle.elapsed == 1.75      # the frozen byte gap, not whole-call time
    @test idle.limit == 2.0
    # A recorded request bound with no idle fire surfaces as that bound.
    req = UniLM.UniLMTimeout(:request, 3.0, 3.0)
    @test UniLM._exit_breach(nothing, Ref{Union{Nothing,UniLM.UniLMTimeout}}(req), cfg) === req
    # Nothing fired: a clean EOF is a clean EOF.
    @test UniLM._exit_breach(nothing, empty_bound, cfg) === nothing
    @test UniLM._exit_breach(UniLM._IdleGuard(:armed, time_ns(), 0.0, 2.0, nothing),
                             empty_bound, cfg) === nothing
end

using Sockets

# Stand-in for the HTTP.jl 1.x transport error whose rendering IS a full request
# dump. Declared at top level because a struct cannot be defined inside a testset.
struct _DumpingError <: Exception; dump::String; end
Base.showerror(io::IO, e::_DumpingError) = print(io, e.dump)

@testset "_error_text never lets a credential reach a result value" begin
    # HTTP.jl 1.x renders a mid-exchange transport failure as a FULL request dump —
    # every header and the body — and its masking covers only Authorization,
    # Proxy-Authorization and Cookie. Providers that authenticate with their own
    # header (Anthropic x-api-key, Gemini native x-goog-api-key, Azure api-key)
    # therefore had the key in cleartext inside `.error`. The redaction layer is
    # major-agnostic, so a stand-in exception whose showerror IS such a dump is the
    # testable contract on either major.
    anth  = "sk-ant-api03-SECRETVALUE0123456789"
    goog  = "AIzaSyGOOGLENATIVESECRET123"
    az    = "azure-key-0011223344556677"
    bear  = "Bearer sk-proxy-SECRET-99887766"
    dump = """
    HTTP.Exceptions.RequestError:
    HTTP.Request:
    POST /v1/messages HTTP/1.1
    Host: api.anthropic.com
    x-api-key: $anth
    x-goog-api-key: $goog
    api-key: $az
    Authorization: $bear
    Content-Type: application/json

    {"model":"claude-opus-4","messages":[{"role":"user","content":"hi"}]}
    Underlying error:
    IOError: read: connection reset by peer (ECONNRESET)
    """
    out = UniLM._error_text(_DumpingError(dump))
    for secret in (anth, goog, az, bear)
        @test !occursin(secret, out)
    end
    # A short, non-reversible marker survives — enough to tell WHICH key was in play.
    @test occursin("sk-a…[redacted]", out)
    @test occursin("AIza…[redacted]", out)
    @test occursin("azur…[redacted]", out)
    @test occursin("Bear…[redacted]", out)
    # Non-auth headers and the underlying cause stay intact: this is redaction, not truncation.
    @test occursin("Content-Type: application/json", out)
    @test occursin("ECONNRESET", out)

    # Julia's pair rendering of a header vector is masked the same way.
    pairs = "[\"x-api-key\" => \"$anth\", \"content-type\" => \"application/json\"]"
    @test !occursin(anth, UniLM._mask_auth_headers(pairs))
    @test occursin("\"content-type\" => \"application/json\"", UniLM._mask_auth_headers(pairs))

    # Root cause preferred over the wrapper: showerror text, wrappers peeled.
    @test UniLM._error_text(ErrorException("boom")) == "boom"
    @test occursin("UniLMTimeout", UniLM._error_text(UniLM.UniLMTimeout(:request, 1.0, 2.0)))
    t = Task(() -> error("inner boom")); schedule(t)
    @test_throws TaskFailedException wait(t)
    @test UniLM._error_text(try wait(t) catch e; e end) == "inner boom"
end

@testset "the auth mask matches header names, not words that merely end in one" begin
    # The pattern had no leading word boundary, so any longer word ending in a header
    # name took its value with it: `reauthorization: <text>` was redacted. Fail-safe,
    # but it deletes diagnostic text that never carried a credential.
    secret = "secret-value-0123456789"
    for word in ("reauthorization", "xauthorization", "deauthorization")
        @test UniLM._mask_auth_headers("$word: $secret") == "$word: $secret"
    end

    # Every real header name still loses its value, in header position, on both forms.
    for name in ("authorization", "Authorization", "x-api-key", "x-goog-api-key",
                 "api-key", "proxy-authorization")
        wire = UniLM._mask_auth_headers("$name: $secret")
        pair = UniLM._mask_auth_headers("\"$name\" => \"$secret\"")
        @test !occursin(secret, wire)
        @test !occursin(secret, pair)
        @test startswith(wire, "$name: secr…[redacted]")
    end

    # The alternation order is load-bearing: `x-api-key` must claim its own tail, and
    # `api-key` must still match where a non-word character precedes it.
    @test UniLM._mask_auth_headers("x-api-key: $secret") == "x-api-key: secr…[redacted]"
    @test UniLM._mask_auth_headers("api-key: $secret") == "api-key: secr…[redacted]"
end

@testset "a transport failure mid-exchange cannot leak the configured key" begin
    # End-to-end absence contract on the live seam: a peer that accepts and
    # immediately closes drives the non-stream driver into its catch, and whatever
    # the HTTP major hands over there must not carry the endpoint's key into the
    # typed result — nor into the result's printed form.
    key = "sk-live-MUSTNOTLEAK-0123456789"
    listener = Sockets.listen(Sockets.localhost, 0)
    port = Int(Sockets.getsockname(listener)[2])
    @async begin
        try
            while true
                close(Sockets.accept(listener))   # abrupt close, mid-exchange
            end
        catch
        end
    end
    try
        ep = GenericOpenAIEndpoint("http://127.0.0.1:$port", key)
        chat = Chat(service=ep, model="m",
                    messages=[Message(role=UniLM.RoleUser, content="hello")])
        r = chatrequest!(chat; config=RequestConfig(max_attempts=1, connect_timeout=3.0,
                                                    request_timeout=5.0, total_deadline=10.0))
        @test r isa LLMCallError
        @test !occursin(key, r.error)
        @test !occursin(key, sprint(show, r))
    finally
        close(listener)
    end
end
