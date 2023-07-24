@testset "openai-api.jl" begin
    chat = UniLM.Chat()
    push!(chat, UniLM.Message(role=UniLM.GPTSystem, content="Act as a helpful AI agent."))
    @test length(chat) == 1

    push!(chat, UniLM.Message(role=UniLM.GPTUser, content="Please tell me a one-liner joke."))
    @test length(chat) == 2

    try
        push!(chat, UniLM.Message(role=UniLM.GPTUser, content="Please tell me a one-liner joke."))
    catch e
        @test e isa UniLM.InvalidConversationError
    end

    @test UniLM.issendvalid(chat) == true

    try
        UniLM.Chat(temperature=0.2, top_p=0.5)
    catch e
        @error e
        @test e isa ArgumentError
    end

    m, _ = UniLM.chatrequest!(chat)
    #@info m
    @test m isa UniLM.Message
    @test m.role == UniLM.GPTAssistant



    # test streaming
    callback = (msg, close) -> begin
        "from callback - echo: $msg"
    end

    chat_with_stream = UniLM.Chat(stream=true, temperature=0.2)
    push!(chat_with_stream, UniLM.Message(role=UniLM.GPTSystem, content="Act as a helpful AI agent."))
    @test length(chat_with_stream) == 1

    push!(chat_with_stream, UniLM.Message(role=UniLM.GPTUser, content="Please tell me a one-liner joke."))
    @test length(chat_with_stream) == 2

    #@info "Starting chat with stream"
    t = UniLM.chatrequest!(chat_with_stream, callback=callback)
    wait(t)
    @test t.state == :done
    m, _ = t.result
    #@info m
    @test m isa UniLM.Message
    @test m.role == UniLM.GPTAssistant


    function get_current_weather(; location, unit="fahrenheit")
        weather_info = Dict(
            "location" => location,
            "temperature" => "72",
            "unit" => unit,
            "forecast" => ["sunny", "windy"]
        )
        return JSON3.write(weather_info)
    end

    get_current_weather_schema = Dict(
        "name" => "get_current_weather",
        "description" => "Get the current weather in a given location",
        "parameters" => Dict(
            "type" => "object",
            "properties" => Dict(
                "location" => Dict(
                    "type" => "string",
                    "description" => "The city and state, e.g. San Francisco, CA"
                ),
                "unit" => Dict(
                    "type" => "string",
                    "enum" => ["celsius", "fahrenheit"]
                )
            ),
            "required" => ["location"]
        )
    )

    gptfsig = UniLM.GPTFunctionSignature(name=get_current_weather_schema["name"], description=get_current_weather_schema["description"], parameters=get_current_weather_schema["parameters"])

    funchat = UniLM.Chat(functions=[gptfsig], function_call="auto")

    push!(funchat, UniLM.Message(role=UniLM.GPTSystem, content="Act as a helpful AI agent."))
    push!(funchat, UniLM.Message(role=UniLM.GPTUser, content="What's the weather like in Boston? Give answer in celsius"))

    (m, _) = UniLM.chatrequest!(funchat)

    #@info "fun answer: " m
    @test UniLM.makecall(m) isa Expr
    @test isnothing(m.content)


    r = UniLM.evalcall!(funchat)
    #@info "result evalcall!: " r

    #funchat.messages[3].function_call["arguments"] = funchat.messages[3].function_call["arguments"]
    (m2, _) = UniLM.chatrequest!(funchat)
    #@info "answer: " m2

    emb = UniLM.Embedding(input="Embed this!")

    result = embeddingrequest!(emb)
    #@info "result: " result
    #@test result isa Vector



end