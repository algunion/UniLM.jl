@testset "openai-api.jl" begin


    @testset "chat/conversation operation/manipulation" begin
        chat = UniLM.Chat()
        push!(chat, UniLM.Message(role=UniLM.RoleSystem, content="Act as a helpful AI agent."))
        push!(chat, UniLM.Message(role=UniLM.RoleUser, content="Please tell me a one-liner joke."))

        @test UniLM.issendvalid(chat) == true

        @test_throws UniLM.InvalidConversationError push!(chat, UniLM.Message(role=UniLM.RoleUser, content="Please tell me a one-liner joke."))
        @test_throws ArgumentError UniLM.Chat(temperature=0.2, top_p=0.5)
    end

    @testset "regular conversation" begin
        chat = UniLM.Chat()
        push!(chat, UniLM.Message(role=UniLM.RoleSystem, content="Act as a helpful AI agent."))
        push!(chat, UniLM.Message(role=UniLM.RoleUser, content="Please tell me a one-liner joke."))

        m, _ = UniLM.chatrequest!(chat)

        @test m isa UniLM.Message
        @test m.role == UniLM.RoleAssistant
    end


    @testset "streaming" begin
        callback = (msg, close) -> begin
            "from callback - echo: $msg"
        end

        chat_with_stream = UniLM.Chat(stream=true, temperature=0.2)
        push!(chat_with_stream, UniLM.Message(role=UniLM.RoleSystem, content="Act as a helpful AI agent."))
        push!(chat_with_stream, UniLM.Message(role=UniLM.RoleUser, content="Please tell me a one-liner joke."))
        t = UniLM.chatrequest!(chat_with_stream, callback=callback)
        wait(t)
        @test t.state == :done
        m, _ = t.result

        @test m isa UniLM.Message
        @test m.role == UniLM.RoleAssistant
    end

    @testset "function call" begin
        gptfsig = UniLM.GPTFunctionSignature(
            name="get_current_weather",
            description="Getting the current weather",
            parameters=UniLM.JsonObject(
                properties=Dict(
                    "location" => UniLM.JsonString(description="The city and state, e.g. San Francisco, CA"),
                    "unit" => UniLM.JsonString(enum=["celsius", "fahrenheit"])
                ),
                required=["location"]
            )
        )

        funchat = UniLM.Chat(tools=[UniLM.GPTTool(func=gptfsig)], tool_choice=UniLM.GPTToolChoice(func=:get_current_weather))
        push!(funchat, UniLM.Message(role=UniLM.RoleSystem, content="Act as a helpful AI agent."))
        push!(funchat, UniLM.Message(role=UniLM.RoleUser, content="What is the weather in New York?."))

        (m, _) = UniLM.chatrequest!(funchat)

        #@test UniLM.makecall(m) isa Expr
        #@test isnothing(m.content)

        #fcall_result = UniLM.evalcall!(funchat)

        #@show m

    end

    @testset "embedding" begin
        emb = UniLM.Embeddings("Embed this!")

        (result, emb) = UniLM.embeddingrequest!(emb)

        @show typeof(result)


        @show keys(result)

    end



end