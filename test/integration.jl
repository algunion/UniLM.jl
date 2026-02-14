@testset "regular conversation" begin
    chat = Chat()
    push!(chat, Message(role=UniLM.RoleSystem, content="Act as a helpful AI agent."))
    push!(chat, Message(role=UniLM.RoleUser, content="Please tell me a one-liner joke."))

    cr = chatrequest!(chat)

    @test cr isa LLMSuccess
    m = cr.message
    @test m.role == UniLM.RoleAssistant
    @test !isnothing(m.content)
    @test m.finish_reason == UniLM.STOP
    # History should be updated
    @test length(chat) == 3
    @test last(chat) == m
end

@testset "regular conversation / kwargs with messages" begin
    messages = Message[]
    push!(messages, Message(role=UniLM.RoleSystem, content="Act as a helpful AI agent."))
    push!(messages, Message(role=UniLM.RoleUser, content="Say 'hello' and nothing else."))

    cr = chatrequest!(; messages=messages)

    @test cr isa LLMSuccess
    m = cr.message
    @test m.role == UniLM.RoleAssistant
    @test !isnothing(m.content)
end

@testset "regular conversation / kwargs / individual prompts" begin
    cr = chatrequest!(
        systemprompt="Act as a helpful AI agent.",
        userprompt="Say 'hello' and nothing else."
    )

    @test cr isa LLMSuccess
    m = cr.message
    @test m.role == UniLM.RoleAssistant
end

@testset "kwargs with Message objects as prompts" begin
    cr = chatrequest!(
        systemprompt=Message(role=UniLM.RoleSystem, content="You are helpful."),
        userprompt=Message(role=UniLM.RoleUser, content="Say 'yes'.")
    )

    @test cr isa LLMSuccess
end

@testset "JSON_OBJECT" begin
    chat = Chat(response_format=UniLM.json_object())
    push!(chat, Message(role=UniLM.RoleSystem, content="Act as a helpful AI agent. Always respond in JSON."))
    push!(chat, Message(role=UniLM.RoleUser, content="Tell me a joke in JSON format with keys 'setup' and 'punchline'."))

    cr = chatrequest!(chat)
    @test cr isa LLMSuccess

    m = cr.message
    @test m isa Message
    @test m.role == UniLM.RoleAssistant
    # Verify it's valid JSON
    parsed = JSON.parse(m.content)
    @test parsed isa AbstractDict
end

@testset "JSON SCHEMA" begin
    schema = Dict(
        "type" => "object",
        "properties" => Dict(
            "location" => Dict("type" => "string", "description" => "The city and state, e.g. San Francisco, CA"),
            "unit" => Dict("type" => "string", "enum" => ["celsius", "fahrenheit"])
        ),
        "additionalProperties" => false,
        "required" => ["location", "unit"]
    )

    chat = Chat(response_format=UniLM.json_schema("get_current_weather", "Getting the current weather", schema))
    push!(chat, Message(role=UniLM.RoleSystem, content="Act as a helpful AI agent. Only answer in JSON."))
    push!(chat, Message(role=UniLM.RoleUser, content="What is the weather in New York?"))

    cr = chatrequest!(chat)
    @test cr isa LLMSuccess

    m = cr.message
    @test m isa Message
    @test m.role == UniLM.RoleAssistant
    parsed = JSON.parse(m.content)
    @test haskey(parsed, "location")
    @test haskey(parsed, "unit")
end

@testset "streaming" begin
    received_chunks = String[]
    final_msg = Ref{Union{Message,Nothing}}(nothing)

    callback = (msg, close) -> begin
        if msg isa Message
            final_msg[] = msg
        elseif msg isa String
            push!(received_chunks, msg)
        end
    end

    chat_with_stream = Chat(stream=true)
    push!(chat_with_stream, Message(role=UniLM.RoleSystem, content="Act as a helpful AI agent."))
    push!(chat_with_stream, Message(role=UniLM.RoleUser, content="Say 'hello world' and nothing else."))
    t = chatrequest!(chat_with_stream; callback=callback)
    result = fetch(t)

    @test !isnothing(result)
    m, _ = result
    @test m isa Message
    @test m.role == UniLM.RoleAssistant
    @test !isnothing(m.content)
    @test m.finish_reason == UniLM.STOP
    # Verify streaming updated chat history
    @test length(chat_with_stream) == 3
end

@testset "function call" begin
    gptfsig = GPTFunctionSignature(
        name="get_current_weather",
        description="Getting the current weather",
        parameters=Dict(
            "type" => "object",
            "properties" => Dict(
                "location" => Dict("type" => "string", "description" => "The city and state, e.g. San Francisco, CA"),
                "unit" => Dict("type" => "string", "enum" => ["celsius", "fahrenheit"])
            ),
            "required" => ["location"]
        )
    )

    funchat = Chat(
        tools=[GPTTool(func=gptfsig)],
        tool_choice=UniLM.GPTToolChoice(func=:get_current_weather)
    )
    push!(funchat, Message(role=UniLM.RoleSystem, content="Act as a helpful AI agent. Always use the provided tools."))
    push!(funchat, Message(role=UniLM.RoleUser, content="What is the weather in New York?"))

    cr = chatrequest!(funchat)
    @test cr isa LLMSuccess

    m = cr.message
    @test m isa Message
    # Model may return tool_calls or stop depending on behavior
    if m.finish_reason == UniLM.TOOL_CALLS
        @test !isnothing(m.tool_calls)
        @test length(m.tool_calls) >= 1
        @test m.tool_calls[1].func.name == "get_current_weather"
        @test haskey(m.tool_calls[1].func.arguments, "location")
    else
        @test m.finish_reason == UniLM.STOP
        @test !isnothing(m.content)
    end
end

@testset "embedding" begin
    emb = UniLM.Embeddings("Embed this!")
    @test all(x -> x == 0.0, emb.embeddings)

    result = embeddingrequest!(emb)

    @test result isa Tuple
    @test !all(x -> x == 0.0, emb.embeddings)  # embeddings updated in-place
    @test length(emb.embeddings) == 1536
end

@testset "conversation with temperature" begin
    chat = Chat(temperature=0.0)
    push!(chat, Message(role=UniLM.RoleSystem, content="You are a calculator."))
    push!(chat, Message(role=UniLM.RoleUser, content="What is 2+2? Answer with just the number."))

    cr = chatrequest!(chat)
    @test cr isa LLMSuccess
    @test occursin("4", cr.message.content)
end

@testset "chat with seed for reproducibility" begin
    chat = Chat(temperature=0.0, seed=42)
    push!(chat, Message(role=UniLM.RoleSystem, content="You are a calculator."))
    push!(chat, Message(role=UniLM.RoleUser, content="What is 1+1? Answer with just the number."))

    cr = chatrequest!(chat)
    @test cr isa LLMSuccess
    @test occursin("2", cr.message.content)
end
