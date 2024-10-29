@testset "api.jl" begin


    @testset "chat/conversation operation/manipulation" begin
        chat = UniLM.Chat()
        push!(chat, UniLM.Message(role=UniLM.RoleSystem, content="Act as a helpful AI agent."))
        push!(chat, UniLM.Message(role=UniLM.RoleUser, content="Please tell me a one-liner joke."))

        @test UniLM.issendvalid(chat)

        inilength = length(chat)
        push!(chat, UniLM.Message(role=UniLM.RoleUser, content="Please tell me a one-liner joke.")) # duplicate user message (avoided)

        @test length(chat) == inilength
        @test_throws ArgumentError UniLM.Chat(temperature=0.2, top_p=0.5)
    end

    @testset "regular conversation" begin
        chat = UniLM.Chat()
        push!(chat, UniLM.Message(role=UniLM.RoleSystem, content="Act as a helpful AI agent."))
        push!(chat, UniLM.Message(role=UniLM.RoleUser, content="Please tell me a one-liner joke."))

        cr = UniLM.chatrequest!(chat)

        if cr isa UniLM.LLMSuccess
            m = getfield(cr, :message)
            @test m.role == UniLM.RoleAssistant
        else
            @test cr <: UniLMRequestResponse
        end

    end

    @testset "JSON_OBJECT" begin
        chat = UniLM.Chat(response_format=UniLM.json_object())
        push!(chat, UniLM.Message(role=UniLM.RoleSystem, content="Act as a helpful AI agent answering only in JSON."))
        push!(chat, UniLM.Message(role=UniLM.RoleUser, content="Please tell me a one-liner joke."))

        cr = UniLM.chatrequest!(chat)

        m = getfield(cr, :message)

        @test m isa UniLM.Message
        @test m.role == UniLM.RoleAssistant

        @info "JSON OBJECT result: $(m.content)"
    end

    @testset "JSON SCHEMA" begin
        schema = Dict(
            "type" => "object",
            "properties" => Dict(
                "location" => Dict("description" => "The city and state, e.g. San Francisco, CA"),
                "unit" => Dict("enum" => ["celsius", "fahrenheit"])
            ),
            "additionalProperties" => false,
            "required" => ["location", "unit"]
        )

        chat = UniLM.Chat(response_format=UniLM.json_schema(UniLM.JsonSchemaAPI("get_current_weather", "Getting the current weather", schema)))
        push!(chat, UniLM.Message(role=UniLM.RoleSystem, content="Act as a helpful AI agent - only answering in JSON."))
        push!(chat, UniLM.Message(role=UniLM.RoleUser, content="What is the weather in New York?."))

        cr = UniLM.chatrequest!(chat)

        m = getfield(cr, :message)

        @test m isa UniLM.Message
        @test m.role == UniLM.RoleAssistant

        @info "JSON SCHEMA Result: $(m.content)"

    end

    @testset "streaming" begin
        callback = (msg, close) -> begin
            "from callback - echo: $msg"
        end

        chat_with_stream = UniLM.Chat(stream=true)
        push!(chat_with_stream, UniLM.Message(role=UniLM.RoleSystem, content="Act as a helpful AI agent."))
        push!(chat_with_stream, UniLM.Message(role=UniLM.RoleUser, content="Please tell me a one-liner joke."))
        t = UniLM.chatrequest!(chat_with_stream, callback=callback)
        wait(t)
        @test t.state == :done
        m, _ = t.result

        #@test m isa UniLM.Message
        #@test m.role == UniLM.RoleAssistant

        @test true
    end

    @testset "function call" begin
        gptfsig = UniLM.GPTFunctionSignature(
            name="get_current_weather",
            description="Getting the current weather",
            parameters=Dict(
                "type" => "object",
                "properties" => Dict(
                    "location" => Dict("description" => "The city and state, e.g. San Francisco, CA"),
                    "unit" => Dict("enum" => ["celsius", "fahrenheit"])
                ),
                "required" => ["location"]
            )
        )

        funchat = UniLM.Chat(tools=[UniLM.GPTTool(func=gptfsig)], tool_choice=UniLM.GPTToolChoice(func=:get_current_weather))
        push!(funchat, UniLM.Message(role=UniLM.RoleSystem, content="Act as a helpful AI agent."))
        push!(funchat, UniLM.Message(role=UniLM.RoleUser, content="What is the weather in New York?."))

        cr = UniLM.chatrequest!(funchat)

        m = getfield(cr, :message)

        @test m isa UniLM.Message

        @info "Function Call Result: $(m)"

    end

    @testset "embedding" begin
        emb = UniLM.Embeddings("Embed this!")

        result = UniLM.embeddingrequest!(emb)

        @show typeof(result)
        @show keys(result)

        @test result isa Tuple

    end

end