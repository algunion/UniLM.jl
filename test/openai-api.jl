@testset "openai-api.jl" begin
    conv = UniLM.Conversation()
    push!(conv, UniLM.Message(role=UniLM.GPTSystem, content="Act as a helpful AI agent."))
    @test length(conv) == 1

    push!(conv, UniLM.Message(role=UniLM.GPTUser, content="Please tell me a one-liner joke."))
    @test length(conv) == 2

    try
        push!(conv, UniLM.Message(role=UniLM.GPTUser, content="Please tell me a one-liner joke."))
    catch e
        @test e isa UniLM.InvalidConversationError
    end

    @test UniLM.is_send_valid(conv) == true

    try 
        UniLM.ChatParams(temperature=0.2, top_p=0.5)
    catch e
        @error e
        @test e isa ArgumentError
    end
    
    params = UniLM.ChatParams()

    @test params.messages |> isempty

    params_with_stream = UniLM.ChatParams(stream=true, temperature=0.2)

    UniLM.chat_request(conv, params=params)

    callback = (msg, close) -> begin         
        @info "from callback - echo: $msg"         
    end
    
    # when stream=true, a task is returned
    t = UniLM.chat_request(conv, params=params_with_stream, callback=callback)
    wait(t)
    @test t.state == :done
    @info "t.result = $(t.result)"

    function get_current_weather(;location, unit="fahrenheit")
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

    funchatparams = UniLM.ChatParams(functions=[gptfsig], function_call="auto")

    conv2 = UniLM.Conversation()
    push!(conv2, UniLM.Message(role=UniLM.GPTSystem, content="Act as a helpful AI agent."))
    push!(conv2, UniLM.Message(role=UniLM.GPTUser, content="What's the weather like in Boston?"))

    fanswer = UniLM.chat_request(conv2, params=funchatparams)

    @info "fun answer: " fanswer
    @test !isnothing(fanswer.function_call)
    @test isnothing(fanswer.content)

    @info UniLM.call_function(fanswer)


    


end