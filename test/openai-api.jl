@testset "openai-api.jl" begin


    @testset "chat/conversation operation/manipulation" begin
        chat = UniLM.Chat()
        push!(chat, UniLM.Message(role=UniLM.GPTSystem, content="Act as a helpful AI agent."))
        push!(chat, UniLM.Message(role=UniLM.GPTUser, content="Please tell me a one-liner joke."))

        @test UniLM.issendvalid(chat) == true

        try
            push!(chat, UniLM.Message(role=UniLM.GPTUser, content="Please tell me a one-liner joke."))
        catch e
            @error "test expected error:" e
            @test e isa UniLM.InvalidConversationError
        end

        try
            UniLM.Chat(temperature=0.2, top_p=0.5)
        catch e
            @error "test expected error:" e
            @test e isa ArgumentError
        end

    end

    @testset "regular conversation" begin
        chat = UniLM.Chat()
        push!(chat, UniLM.Message(role=UniLM.GPTSystem, content="Act as a helpful AI agent."))
        push!(chat, UniLM.Message(role=UniLM.GPTUser, content="Please tell me a one-liner joke."))

        m, _ = UniLM.chatrequest!(chat)

        @test m isa UniLM.Message
        @test m.role == UniLM.GPTAssistant
    end


    @testset "streaming" begin
        callback = (msg, close) -> begin
            "from callback - echo: $msg"
        end

        chat_with_stream = UniLM.Chat(stream=true, temperature=0.2)
        push!(chat_with_stream, UniLM.Message(role=UniLM.GPTSystem, content="Act as a helpful AI agent."))
        push!(chat_with_stream, UniLM.Message(role=UniLM.GPTUser, content="Please tell me a one-liner joke."))
        t = UniLM.chatrequest!(chat_with_stream, callback=callback)
        wait(t)
        @test t.state == :done
        m, _ = t.result

        @test m isa UniLM.Message
        @test m.role == UniLM.GPTAssistant
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

        funchat = UniLM.Chat(functions=[gptfsig], function_call=("name" => "get_current_weather"))
        push!(funchat, UniLM.Message(role=UniLM.GPTSystem, content="Act as a helpful AI agent."))
        push!(funchat, UniLM.Message(role=UniLM.GPTUser, content="What is the weather in boston?."))

        (m, _) = UniLM.chatrequest!(funchat)

        #@test UniLM.makecall(m) isa Expr
        @test isnothing(m.content)

        fcall_result = UniLM.evalcall!(funchat)
        @info "fcall_result: " fcall_result

        @show m

    end

    @testset "embedding" begin
        emb = UniLM.Embedding(input="Embed this!")

        (result, emb) = UniLM.embeddingrequest!(emb)

        @show typeof(result)


        @show keys(result)
    end




    # #funchat.messages[3].function_call["arguments"] = funchat.messages[3].function_call["arguments"]
    # (m2, _) = UniLM.chatrequest!(funchat)
    # #@info "answer: " m2

    # emb = UniLM.Embedding(input="Embed this!")

    # result = embeddingrequest!(emb)
    #@info "result: " result
    #@test result isa Vector



end