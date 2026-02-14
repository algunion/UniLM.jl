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

# ── Responses API Tests ──────────────────────────────────────────────────────

@testset "Responses API — basic text" begin
    r = respond("Say 'hello world' and nothing else.")
    @test r isa ResponseSuccess
    @test r.response.status == "completed"
    @test !isempty(output_text(r))
    @test !isnothing(r.response.id)
    @test !isnothing(r.response.model)
    @test !isnothing(r.response.usage)
end

@testset "Responses API — with instructions" begin
    r = respond(
        "Translate to French: Hello",
        instructions="You are a translator. Respond only with the translation."
    )
    @test r isa ResponseSuccess
    text = output_text(r)
    @test !isempty(text)
end

@testset "Responses API — multi-turn" begin
    r1 = respond("Say the number 42 and nothing else.")
    @test r1 isa ResponseSuccess
    @test occursin("42", output_text(r1))

    r2 = respond("Now add 8 to that number and say only the result.", previous_response_id=r1.response.id)
    @test r2 isa ResponseSuccess
    @test occursin("50", output_text(r2))
end

@testset "Responses API — structured output (json_schema_format)" begin
    fmt = json_schema_format(
        "answer", "A structured answer",
        Dict(
            "type" => "object",
            "properties" => Dict(
                "result" => Dict("type" => "integer"),
                "explanation" => Dict("type" => "string")
            ),
            "required" => ["result", "explanation"],
            "additionalProperties" => false
        ),
        strict=true
    )
    r = respond("What is 6 * 7? Give the result and a brief explanation.", text=fmt)
    @test r isa ResponseSuccess

    parsed = JSON.parse(output_text(r))
    @test parsed isa AbstractDict
    @test haskey(parsed, "result")
    @test haskey(parsed, "explanation")
    @test parsed["result"] == 42
end

@testset "Responses API — function calling" begin
    tool = function_tool(
        "get_temperature",
        "Get the temperature for a city",
        parameters=Dict(
            "type" => "object",
            "properties" => Dict(
                "city" => Dict("type" => "string", "description" => "City name"),
                "unit" => Dict("type" => "string", "enum" => ["celsius", "fahrenheit"])
            ),
            "required" => ["city", "unit"],
            "additionalProperties" => false
        ),
        strict=true
    )
    r = respond("What is the temperature in London? Use celsius.", tools=[tool])
    @test r isa ResponseSuccess

    calls = function_calls(r)
    @test length(calls) >= 1
    @test calls[1]["name"] == "get_temperature"

    args = JSON.parse(calls[1]["arguments"])
    @test haskey(args, "city")
    @test args["unit"] == "celsius"
end

@testset "Responses API — web search" begin
    r = respond(
        "What is the latest stable release of the Julia programming language?",
        tools=[web_search()]
    )
    @test r isa ResponseSuccess
    text = output_text(r)
    @test !isempty(text)
    @test occursin(r"[Jj]ulia", text)
end

@testset "Responses API — streaming" begin
    chunks = String[]
    final_response = Ref{Any}(nothing)

    task = respond("Say 'streaming works' and nothing else.") do chunk, close
        if chunk isa String
            push!(chunks, chunk)
        elseif chunk isa ResponseObject
            final_response[] = chunk
        end
    end

    r = fetch(task)
    @test r isa ResponseSuccess
    @test !isempty(output_text(r))
    @test !isempty(chunks)
end

# ── Image Generation Tests ───────────────────────────────────────────────────

@testset "Image Generation — basic" begin
    r = generate_image(
        "A simple blue square on white background",
        size="1024x1024",
        quality="low"
    )
    @test r isa ImageSuccess
    @test !isnothing(r.response)
    @test length(r.response.data) >= 1
    @test !isnothing(r.response.data[1].b64_json)
end

@testset "Image Generation — image_data accessor" begin
    r = generate_image(
        "A small red circle",
        size="1024x1024",
        quality="low"
    )
    @test r isa ImageSuccess

    imgs = image_data(r)
    @test imgs isa Vector{String}
    @test length(imgs) >= 1
    @test length(imgs[1]) > 100  # non-trivial base64 data
end

@testset "Image Generation — save_image" begin
    r = generate_image(
        "A green triangle",
        size="1024x1024",
        quality="low"
    )
    @test r isa ImageSuccess

    tmpfile = tempname() * ".png"
    try
        result = save_image(image_data(r)[1], tmpfile)
        @test result == tmpfile
        @test isfile(tmpfile)
        @test filesize(tmpfile) > 0
    finally
        isfile(tmpfile) && rm(tmpfile)
    end
end
