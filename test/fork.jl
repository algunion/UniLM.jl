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

    @testset "fork shares tools (shallow copy)" begin
        sig = GPTFunctionSignature(name="fn")
        tools = [GPTTool(func=sig)]
        chat = Chat(tools=tools)
        forked = fork(chat)

        @test forked.tools === chat.tools  # same reference
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
        sig = GPTFunctionSignature(name="fn")
        chat = Chat(
            model="gpt-4.1-mini",
            tools=[GPTTool(func=sig)],
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
end
