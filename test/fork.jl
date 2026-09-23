@testset "fork" begin
    @testset "basic fork creates independent copy" begin
        chat = Chat(model="gpt-4.1", temperature=0.7)
        push!(chat, Message(Val(:system), "You are helpful."))
        push!(chat, Message(Val(:user), "Hello!"))

        forked = fork(chat)

        # Same config
        @test forked.model == chat.model
        @test forked.temperature == chat.temperature
        @test forked.service == chat.service
        @test forked.history == chat.history

        # Same messages content
        @test length(forked) == length(chat)
        @test forked[1].content == chat[1].content
        @test forked[2].content == chat[2].content

        # Independent messages (deep copy)
        push!(forked, Message(role=UniLM.RoleAssistant, content="Hi there!"))
        @test length(forked) == 3
        @test length(chat) == 2  # original unchanged
    end

    @testset "fork copies tools (no shared vector or signature)" begin
        chat = Chat(tools=[Tool(func=FunctionSignature(name="fn"))])
        forked = fork(chat)

        @test forked.tools !== chat.tools
        @test forked.tools[1].func !== chat.tools[1].func   # FunctionSignature is mutable
        @test forked.tools[1].func.name == "fn"
    end

    @testset "mutating a fork never reaches its parent; the service is shared" begin
        chat = Chat(service=GenericOpenAIEndpoint("http://127.0.0.1:1", ""), model="m",
                    tools=[Tool(func=FunctionSignature(name="fn", parameters=Dict("type" => "object")))],
                    stop=["END"], logit_bias=Dict("100" => 1.0), metadata=Dict("k" => "v"),
                    stream_options=Dict("include_usage" => true), modalities=["text"])
        f = fork(chat)
        push!(f.tools, Tool(func=FunctionSignature(name="extra")))
        f.tools[1].func.description = "changed"
        f.tools[1].func.parameters["type"] = "array"
        f.metadata["k"] = "changed"
        push!(f.stop, "HALT")
        f.logit_bias["100"] = -5.0
        f.stream_options["include_usage"] = false
        push!(f.modalities, "audio")

        @test length(chat.tools) == 1
        @test chat.tools[1].func.description === nothing
        @test chat.tools[1].func.parameters == Dict("type" => "object")
        @test chat.metadata == Dict("k" => "v")
        @test chat.stop == ["END"]
        @test chat.logit_bias == Dict("100" => 1.0)
        @test chat.stream_options == Dict("include_usage" => true)
        @test chat.modalities == ["text"]
        @test f.service === chat.service
    end

    @testset "fork has independent cumulative cost" begin
        chat = Chat()
        chat._cumulative_cost[] = 0.05

        forked = fork(chat)
        @test cumulative_cost(forked) == 0.05  # copied value

        forked._cumulative_cost[] += 0.10
        @test cumulative_cost(forked) ≈ 0.15
        @test cumulative_cost(chat) ≈ 0.05  # original unchanged
    end

    @testset "fork(chat, n) creates n independent branches" begin
        chat = Chat()
        push!(chat, Message(Val(:system), "sys"))

        forks = fork(chat, 3)
        @test length(forks) == 3
        @test all(f -> length(f) == 1, forks)

        # Each fork is independent
        push!(forks[1], Message(Val(:user), "branch 1"))
        @test length(forks[1]) == 2
        @test length(forks[2]) == 1
        @test length(forks[3]) == 1
        @test length(chat) == 1
    end

    @testset "fork preserves all config fields" begin
        sig = FunctionSignature(name="fn")
        chat = Chat(
            model="gpt-4.1-mini",
            tools=[Tool(func=sig)],
            tool_choice="auto",
            parallel_tool_calls=true,
            temperature=0.5,
            n=2,
            stream=true,
            stop=["END"],
            max_tokens=100,
            presence_penalty=0.5,
            response_format=ResponseFormat(),
            frequency_penalty=0.3,
            logit_bias=Dict("100" => 1.0),
            user="user_1",
            seed=42
        )
        forked = fork(chat)

        @test forked.model == "gpt-4.1-mini"
        @test forked.tool_choice == "auto"
        @test forked.parallel_tool_calls == true
        @test forked.temperature == 0.5
        @test forked.n == 2
        @test forked.stream == true
        @test forked.stop == ["END"]
        @test forked.max_tokens == 100
        @test forked.presence_penalty == 0.5
        @test forked.frequency_penalty == 0.3
        @test forked.user == "user_1"
        @test forked.seed == 42
    end

    @testset "fork copies every config field (generic)" begin
        chat = Chat(model="gpt-5.5", top_p=0.9,
                    reasoning_effort="high", max_completion_tokens=222,
                    stream_options=Dict("include_usage" => true), verbosity="low",
                    store=true, metadata=Dict("k" => "v"), service_tier="auto",
                    logprobs=true, top_logprobs=3,
                    prediction=Dict("type" => "content"), modalities=["text"],
                    audio=Dict("voice" => "alloy"),
                    web_search_options=Dict("search_context_size" => "low"),
                    prompt_cache_key="pck", safety_identifier="sid")
        f = fork(chat)
        for name in fieldnames(Chat)
            name in (:messages, :_cumulative_cost) && continue
            @test isequal(getfield(f, name), getfield(chat, name))
        end
        # messages independent, cost Ref fresh but equal in value
        @test f.messages !== chat.messages
        @test f._cumulative_cost !== chat._cumulative_cost
        @test f._cumulative_cost[] == chat._cumulative_cost[]
    end

    @testset "fork copies parallel_tool_calls verbatim (no rewrite)" begin
        # With tools present and parallel_tool_calls UNSET, the constructor keeps
        # `nothing` (the provider default applies); forking must not rewrite it to
        # an explicit `false`, which WOULD be emitted on the wire and silently
        # turn parallel calling off for the fork.
        chat2 = Chat(model="gpt-5.5",
                     tools=[Tool(func=FunctionSignature(name="t"))],
                     parallel_tool_calls=nothing)
        @test fork(chat2).parallel_tool_calls === nothing
    end

    @testset "fork(chat, n) forks inherit the generic copy" begin
        chat = Chat(model="gpt-5.5", prompt_cache_key="pck2")
        fs = fork(chat, 3)
        @test length(fs) == 3
        @test all(f -> f.prompt_cache_key == "pck2", fs)
        @test allunique([objectid(f.messages) for f in fs])
    end
end
