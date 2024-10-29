@testset "api.jl" begin


    @testset "chat/conversation operation/manipulation" begin
        chat = LLM.Chat()
        push!(chat, LLM.Message(role=LLM.RoleSystem, content="Act as a helpful AI agent."))
        push!(chat, LLM.Message(role=LLM.RoleUser, content="Please tell me a one-liner joke."))

        @test LLM.issendvalid(chat) == true

        inilength = length(chat)
        push!(chat, LLM.Message(role=LLM.RoleUser, content="Please tell me a one-liner joke.")) # duplicate user message (avoided)

        @test length(chat) == inilength
        @test_throws ArgumentError LLM.Chat(temperature=0.2, top_p=0.5)
    end

    @testset "regular conversation" begin
        chat = LLM.Chat()
        push!(chat, LLM.Message(role=LLM.RoleSystem, content="Act as a helpful AI agent."))
        push!(chat, LLM.Message(role=LLM.RoleUser, content="Please tell me a one-liner joke."))

        cr = LLM.chatrequest!(chat)

        if cr isa LLM.LLMSuccess
            m = getfield(cr, :message)
            @test m.role == LLM.RoleAssistant
        else
            @test cr <: LLMRequestResponse
        end

    end

    @testset "JSON_OBJECT" begin
        chat = LLM.Chat(response_format=LLM.json_object())
        push!(chat, LLM.Message(role=LLM.RoleSystem, content="Act as a helpful AI agent answering only in JSON."))
        push!(chat, LLM.Message(role=LLM.RoleUser, content="Please tell me a one-liner joke."))

        cr = LLM.chatrequest!(chat)

        m = getfield(cr, :message)

        @test m isa LLM.Message
        @test m.role == LLM.RoleAssistant

        @info "JSON OBJECT result: $(m.content)"
    end

    @testset "JSON SCHEMA" begin
        schema = Dict(
            "properties" => Dict(
                "location" => LLM.JsonString(description="The city and state, e.g. San Francisco, CA"),
                "unit" => LLM.JsonString(enum=["celsius", "fahrenheit"])
            ),
            additionalProperties=false,
            required=["location", "unit"]
        )

        chat = LLM.Chat(response_format=LLM.json_schema(LLM.JsonSchemaAPI("get_current_weather", "Getting the current weather", schema)))
        push!(chat, LLM.Message(role=LLM.RoleSystem, content="Act as a helpful AI agent - only answering in JSON."))
        push!(chat, LLM.Message(role=LLM.RoleUser, content="What is the weather in New York?."))

        cr = LLM.chatrequest!(chat)

        m = getfield(cr, :message)

        @test m isa LLM.Message
        @test m.role == LLM.RoleAssistant

        @info "JSON SCHEMA Result: $(m.content)"

    end


    @testset "streaming" begin
        callback = (msg, close) -> begin
            "from callback - echo: $msg"
        end

        chat_with_stream = LLM.Chat(stream=true)
        push!(chat_with_stream, LLM.Message(role=LLM.RoleSystem, content="Act as a helpful AI agent."))
        push!(chat_with_stream, LLM.Message(role=LLM.RoleUser, content="Please tell me a one-liner joke."))
        t = LLM.chatrequest!(chat_with_stream, callback=callback)
        wait(t)
        @test t.state == :done
        m, _ = t.result

        #@test m isa LLM.Message
        #@test m.role == LLM.RoleAssistant

        @test true
    end

    @testset "function call" begin
        gptfsig = LLM.GPTFunctionSignature(
            name="get_current_weather",
            description="Getting the current weather",
            parameters=Dict(
                properties => Dict(
                    "location" => LLM.JsonString(description="The city and state, e.g. San Francisco, CA"),
                    "unit" => LLM.JsonString(enum=["celsius", "fahrenheit"])
                ),
                required=["location"]
            )
        )

        funchat = LLM.Chat(tools=[LLM.GPTTool(func=gptfsig)], tool_choice=LLM.GPTToolChoice(func=:get_current_weather))
        push!(funchat, LLM.Message(role=LLM.RoleSystem, content="Act as a helpful AI agent."))
        push!(funchat, LLM.Message(role=LLM.RoleUser, content="What is the weather in New York?."))

        #(m, _) = LLM.chatrequest!(funchat)

        #@test LLM.makecall(m) isa Dict
        #@test isnothing(m.content)

        #fcall_result = LLM.evalcall!(funchat)




        # fchat2 = LLM.Chat()
        # for m in funchat.messages
        #     @show "message is: $m"
        #     push!(fchat2, m)
        # end
        # (m, _) = LLM.chatrequest!(fchat2)

        # @show m

        #@show m

    end

    @testset "embedding" begin
        emb = LLM.Embeddings("Embed this!")

        (result, emb) = LLM.embeddingrequest!(emb)

        @show typeof(result)


        @show keys(result)

    end

end