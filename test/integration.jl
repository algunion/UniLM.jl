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

    @test result isa LLMSuccess
    m = result.message
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

@testset "batch embedding" begin
    inputs = ["Julia is fast", "Python is popular", "Rust is safe"]
    emb = UniLM.Embeddings(inputs)

    @test emb.embeddings isa Vector{Vector{Float64}}
    @test length(emb.embeddings) == 3
    @test all(v -> all(x -> x == 0.0, v), emb.embeddings)

    result = embeddingrequest!(emb)

    @test result isa Tuple
    @test length(emb.embeddings) == 3
    for i in 1:3
        @test !all(x -> x == 0.0, emb.embeddings[i])  # each embedding updated
        @test length(emb.embeddings[i]) == 1536
    end
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

# ── Responses API: get_response / delete_response / list_input_items ─────────

@testset "Responses API — get_response" begin
    # Create a response first, then retrieve it
    r = respond("Say 'stored' and nothing else.", store=true)
    @test r isa ResponseSuccess
    rid = r.response.id

    retrieved = get_response(rid)
    @test retrieved isa ResponseSuccess
    @test retrieved.response.id == rid
    @test retrieved.response.status == "completed"
    @test !isempty(output_text(retrieved))
end

@testset "Responses API — list_input_items" begin
    r = respond("Say 'items test' and nothing else.", store=true)
    @test r isa ResponseSuccess

    items = list_input_items(r.response.id)
    @test items isa Dict
    @test haskey(items, "data")
    @test length(items["data"]) >= 1
end

@testset "Responses API — delete_response" begin
    r = respond("Say 'delete me' and nothing else.", store=true)
    @test r isa ResponseSuccess
    rid = r.response.id

    result = delete_response(rid)
    @test result isa Dict
    @test result["deleted"] == true
    @test result["id"] == rid
end

# ── Responses API: count_input_tokens ─────────────────────────────────────────

@testset "Responses API — count_input_tokens" begin
    result = count_input_tokens(input="Tell me a joke about programming")
    @test result isa Dict
    @test haskey(result, "input_tokens")
    @test result["input_tokens"] > 0
end

@testset "Responses API — count_input_tokens with instructions and tools" begin
    tool = function_tool(
        "lookup",
        "Look something up",
        parameters=Dict(
            "type" => "object",
            "properties" => Dict("query" => Dict("type" => "string")),
            "required" => ["query"],
            "additionalProperties" => false
        ),
        strict=true
    )
    result = count_input_tokens(
        input="Search for Julia programming language",
        instructions="You are helpful.",
        tools=[tool]
    )
    @test result isa Dict
    @test result["input_tokens"] > 0
end

# ── Responses API: compact_response ──────────────────────────────────────────

@testset "Responses API — compact_response" begin
    input_items = [
        Dict("role" => "user", "content" => "Hello, I'm starting a long conversation."),
        Dict("type" => "message", "role" => "assistant", "status" => "completed",
            "content" => [Dict("type" => "output_text",
                "text" => "Hello! I'm here to help. What would you like to discuss?")])
    ]
    result = compact_response(input=input_items)
    @test result isa Dict
    @test haskey(result, "output")
    @test haskey(result, "usage")
end

# ── Responses API: new fields ────────────────────────────────────────────────

@testset "Responses API — with service_tier" begin
    r = respond("Say 'tier test' and nothing else.", service_tier="auto")
    @test r isa ResponseSuccess
    @test !isempty(output_text(r))
end

@testset "Responses API — with structured input messages" begin
    r = respond([
        InputMessage(role="developer", content="You are helpful."),
        InputMessage(role="user", content="Say 'structured' and nothing else.")
    ])
    @test r isa ResponseSuccess
    @test occursin("structured", lowercase(output_text(r)))
end

@testset "Responses API — with multimodal input" begin
    r = respond([InputMessage(
        role="user",
        content=[input_text("What does 2+2 equal? Reply with just the number.")]
    )])
    @test r isa ResponseSuccess
    @test occursin("4", output_text(r))
end

@testset "Responses API — with reasoning (o-series)" begin
    r = respond(
        "What is 15 * 23? Reply with just the number.",
        model="o4-mini",
        reasoning=Reasoning(effort="low")
    )
    @test r isa ResponseSuccess
    @test occursin("345", output_text(r))
end

@testset "Responses API — with max_output_tokens" begin
    r = respond("Say 'token limit' and nothing else.", max_output_tokens=Int64(100))
    @test r isa ResponseSuccess
    @test !isempty(output_text(r))
end

@testset "Responses API — with temperature" begin
    r = respond("Say 'temp test' and nothing else.", temperature=0.0)
    @test r isa ResponseSuccess
    @test !isempty(output_text(r))
end

@testset "Responses API — with store=false" begin
    r = respond("Say 'no store' and nothing else.", store=false)
    @test r isa ResponseSuccess
    @test !isempty(output_text(r))
end

@testset "Responses API — with metadata" begin
    r = respond("Say 'meta' and nothing else.", metadata=Dict("test_id" => "integration_123"))
    @test r isa ResponseSuccess
    @test !isempty(output_text(r))
end
