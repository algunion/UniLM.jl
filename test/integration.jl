if !haskey(ENV, "OPENAI_API_KEY") || get(ENV, "UNILM_LIVE", "") != "1"
    @info "Skipping OpenAI integration tests (set UNILM_LIVE=1 and OPENAI_API_KEY to run live)"
else

# Live tests run after many minutes of offline testsets. A connection pooled
# during an earlier testset (the offline error-path tests do contact the real
# API host) can be silently dropped by NATs/load balancers during that idle
# window; reusing it stalls the first request here until the kernel read
# timeout (~15 min on Linux). Start from fresh connections.
HTTP.close_idle_connections!()

@testset "regular conversation" begin
    chat = Chat(model="gpt-5.6-luna")
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

    cr = chatrequest!(; messages=messages, model="gpt-5.6-luna")

    @test cr isa LLMSuccess
    m = cr.message
    @test m.role == UniLM.RoleAssistant
    @test !isnothing(m.content)
end

@testset "regular conversation / kwargs / individual prompts" begin
    cr = chatrequest!(
        systemprompt="Act as a helpful AI agent.",
        userprompt="Say 'hello' and nothing else.",
        model="gpt-5.6-luna"
    )

    @test cr isa LLMSuccess
    m = cr.message
    @test m.role == UniLM.RoleAssistant
end

@testset "kwargs with Message objects as prompts" begin
    cr = chatrequest!(
        systemprompt=Message(role=UniLM.RoleSystem, content="You are helpful."),
        userprompt=Message(role=UniLM.RoleUser, content="Say 'yes'."),
        model="gpt-5.6-luna"
    )

    @test cr isa LLMSuccess
end

@testset "JSON_OBJECT" begin
    chat = Chat(model="gpt-5.6-luna", response_format=UniLM.json_object())
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

    chat = Chat(model="gpt-5.6-luna", response_format=UniLM.json_schema("get_current_weather", "Getting the current weather", schema))
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

    chat_with_stream = Chat(model="gpt-5.6-luna", stream=true)
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
    gptfsig = FunctionSignature(
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
        model="gpt-5.6-luna",
        reasoning_effort="none",
        tools=[Tool(func=gptfsig)],
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

@testset "strict function calling — transmission witness" begin
    # The same strict-invalid schema (typed additionalProperties map) must be accepted
    # non-strict and rejected 400 with strict=true. The two requests differ only in the
    # strict flag, so the 400 is proof the flag reaches tools[].function on the wire.
    map_params = Dict(
        "type" => "object",
        "properties" => Dict(
            "kwargs" => Dict("type" => "object", "additionalProperties" => Dict("type" => "string"))
        ),
        "required" => ["kwargs"],
        "additionalProperties" => false
    )
    mktool(strict) = Tool(func=FunctionSignature(
        name="apply_kwargs", description="Apply keyword arguments",
        parameters=map_params, strict=strict))
    function strict_call(strict)
        chat = Chat(model="gpt-5.6-luna", reasoning_effort="none", tools=[mktool(strict)], max_completion_tokens=16)
        push!(chat, Message(Val(:system), "You are terse."))
        push!(chat, Message(Val(:user), "hi"))
        chatrequest!(chat)
    end

    control = strict_call(nothing)
    @test control isa LLMSuccess   # non-strict default unchanged: map schema accepted

    witness = strict_call(true)
    @test witness isa LLMFailure   # strict=true engages the strict schema validator
    if witness isa LLMFailure
        @test witness.status == 400
        # Pin the error CODE — it discriminates schema-rejection from unrelated
        # 400s (auth, bad model, rate limit). Deliberately no pin on err["param"]
        # or message wording: those are OpenAI-internal formats.
        err = JSON.parse(witness.response)["error"]
        @test err["code"] == "invalid_function_parameters"
    end
end

@testset "embedding" begin
    emb = UniLM.Embeddings("Embed this!")
    @test all(x -> x == 0.0, emb.embeddings)

    result = embeddingrequest!(emb)

    @test result isa EmbeddingSuccess
    @test !all(x -> x == 0.0, emb.embeddings)  # embeddings updated in-place
    @test length(emb.embeddings) == 1536
    @test embedding_vectors(result) === emb.embeddings
end

@testset "batch embedding" begin
    inputs = ["Julia is fast", "Python is popular", "Rust is safe"]
    emb = UniLM.Embeddings(inputs)

    @test emb.embeddings isa Vector{Vector{Float64}}
    @test length(emb.embeddings) == 3
    @test all(v -> all(x -> x == 0.0, v), emb.embeddings)

    result = embeddingrequest!(emb)

    @test result isa EmbeddingSuccess
    @test length(emb.embeddings) == 3
    for i in 1:3
        @test !all(x -> x == 0.0, emb.embeddings[i])  # each embedding updated
        @test length(emb.embeddings[i]) == 1536
    end
end

@testset "conversation with temperature" begin
    chat = Chat(model="gpt-5.6-luna", reasoning_effort="none", temperature=0.0)
    push!(chat, Message(role=UniLM.RoleSystem, content="You are a calculator."))
    push!(chat, Message(role=UniLM.RoleUser, content="What is 2+2? Answer with just the number."))

    cr = chatrequest!(chat)
    @test cr isa LLMSuccess
    @test occursin("4", cr.message.content)
end

@testset "chat with seed for reproducibility" begin
    chat = Chat(model="gpt-5.6-luna", reasoning_effort="none", temperature=0.0, seed=42)
    push!(chat, Message(role=UniLM.RoleSystem, content="You are a calculator."))
    push!(chat, Message(role=UniLM.RoleUser, content="What is 1+1? Answer with just the number."))

    cr = chatrequest!(chat)
    @test cr isa LLMSuccess
    @test occursin("2", cr.message.content)
end

# ── Responses API Tests ──────────────────────────────────────────────────────

@testset "Responses API — basic text" begin
    r = respond("Say 'hello world' and nothing else.", model="gpt-5.6-luna")
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
        model="gpt-5.6-luna",
        instructions="You are a translator. Respond only with the translation."
    )
    @test r isa ResponseSuccess
    text = output_text(r)
    @test !isempty(text)
end

@testset "Responses API — multi-turn" begin
    r1 = respond("Say the number 42 and nothing else.", model="gpt-5.6-luna")
    @test r1 isa ResponseSuccess
    @test occursin("42", output_text(r1))

    r2 = respond("Now add 8 to that number and say only the result.", model="gpt-5.6-luna", previous_response_id=r1.response.id)
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
    r = respond("What is 6 * 7? Give the result and a brief explanation.", model="gpt-5.6-luna", text=fmt)
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
    r = respond("What is the temperature in London? Use celsius.", model="gpt-5.6-luna", tools=[tool])
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
        model="gpt-5.6-luna",
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

    task = respond("Say 'streaming works' and nothing else.", model="gpt-5.6-luna") do chunk, close
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
    r = respond("Say 'stored' and nothing else.", model="gpt-5.6-luna", store=true)
    @test r isa ResponseSuccess
    rid = r.response.id

    retrieved = get_response(rid)
    @test retrieved isa ResponseSuccess
    @test retrieved.response.id == rid
    @test retrieved.response.status == "completed"
    @test !isempty(output_text(retrieved))

    # The token count takes the stored response as the turns before its input (a count bills no tokens).
    alone = count_input_tokens(model="gpt-5.6-luna", input="And again?")
    after = count_input_tokens(model="gpt-5.6-luna", input="And again?", previous_response_id=rid)
    @test after isa Dict && alone isa Dict && after["input_tokens"] > alone["input_tokens"]
end

@testset "Responses API — list_input_items" begin
    r = respond("Say 'items test' and nothing else.", model="gpt-5.6-luna", store=true)
    @test r isa ResponseSuccess

    items = list_input_items(r.response.id)
    @test items isa Dict
    @test haskey(items, "data")
    @test length(items["data"]) >= 1
end

@testset "Responses API — delete_response" begin
    r = respond("Say 'delete me' and nothing else.", model="gpt-5.6-luna", store=true)
    @test r isa ResponseSuccess
    rid = r.response.id

    result = delete_response(rid)
    @test result isa Dict
    @test result["deleted"] == true
    @test result["id"] == rid
end

# ── Responses API: count_input_tokens ─────────────────────────────────────────

@testset "Responses API — count_input_tokens" begin
    result = count_input_tokens(model="gpt-5.6-luna", input="Tell me a joke about programming")
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
        model="gpt-5.6-luna",
        input="Search for Julia programming language",
        instructions="You are helpful.",
        tools=[tool]
    )
    @test result isa Dict
    @test result["input_tokens"] > 0
end

# The token count body's fields, all at once, and a Chat Tool sent as the Responses function tool (the
# Chat shape is HTTP 400 "Missing required parameter: 'tools[0].name'"). A count reports no usage, and
# OpenAI publishes no price for it; the lookups and the made-up delete bill nothing.
@testset "Responses API — count_input_tokens sends every field of the token count body" begin
    sig = FunctionSignature(name="lookup", description="Look something up",
                            parameters=Dict("type" => "object", "properties" => Dict("q" => Dict("type" => "string"))))
    plain = count_input_tokens(model="gpt-6-luna", input="Tell me a joke")
    every = count_input_tokens(model="gpt-6-luna", input="Tell me a joke", instructions="Be brief.",
                               tools=[Tool(func=sig)], tool_choice="auto", parallel_tool_calls=false,
                               reasoning=Reasoning(effort="none"), text=TextConfig(verbosity="low"), truncation="auto",
                               personality="friendly")
    @test plain isa Dict && every isa Dict
    @test every["input_tokens"] > plain["input_tokens"] > 0
    # A conversation's items count as the turns before the input (creating one bills nothing).
    conv = create_conversation(; items=[Dict("type" => "message", "role" => "user",
                                             "content" => "My name is Ada, and I write Julia every day.")])
    @test conv isa ConversationSuccess
    if conv isa ConversationSuccess
        try
            within = count_input_tokens(model="gpt-6-luna", input="Tell me a joke", conversation=conv.response.id)
            @test within isa Dict && within["input_tokens"] > plain["input_tokens"]
        finally
            @test delete_conversation(conv.response.id) isa ConversationDeleteSuccess
        end
    end
end

@testset "count_tokens — a Respond's input tokens, as count_input_tokens counts them" begin
    sig = FunctionSignature(name="lookup", description="Look something up", parameters=Dict("type" => "object"))
    r = Respond(model="gpt-6-luna", input="Tell me a joke", instructions="Be brief.", tools=[Tool(func=sig)],
                reasoning=Reasoning(effort="none"), max_output_tokens=16, store=false)
    c = count_tokens(r)
    @test c isa TokenCountSuccess
    @test c.input_tokens == count_input_tokens(model="gpt-6-luna", input="Tell me a joke", instructions="Be brief.",
                                               tools=[Tool(func=sig)], reasoning=Reasoning(effort="none"))["input_tokens"]
    @test c.raw["object"] == "response.input_tokens"
end

@testset "retrieve_model — one OpenAI model; an unknown one is a 404" begin
    r = retrieve_model("gpt-6-luna")
    @test r isa ModelSuccess && r.model.id == "gpt-6-luna" && r.model.raw["object"] == "model"
    @test !isnothing(r.model.created)
    r = retrieve_model("gpt-made-up-0")
    @test r isa ModelListFailure && r.status == 404
    # A fine-tuned id's colons go percent-encoded; OpenAI decodes them, so its 404 names the id as given.
    r = retrieve_model("ft:gpt-4o-mini:unilm:made-up:abc123")
    @test r isa ModelListFailure && r.status == 404 && occursin("'ft:gpt-4o-mini:unilm:made-up:abc123'", r.response)
end

# Never a real model: the delete of a made-up fine-tuned model is OpenAI's free witness, a 404.
@testset "delete a fine-tuned model — a made-up id is a 404" begin
    r = delete_model("ft:gpt-4o-mini:unilm:made-up:abc123"; service=OPENAIServiceEndpoint)
    @test r isa ModelListFailure && r.status == 404 && occursin("model_not_found", r.response)
end

# ── Responses API: compact_response ──────────────────────────────────────────

@testset "Responses API — compact_response" begin
    input_items = [
        Dict("role" => "user", "content" => "Hello, I'm starting a long conversation."),
        Dict("type" => "message", "role" => "assistant", "status" => "completed",
            "content" => [Dict("type" => "output_text",
                "text" => "Hello! I'm here to help. What would you like to discuss?")])
    ]
    result = compact_response(model="gpt-5.6-luna", input=input_items)
    @test result isa Dict
    @test haskey(result, "output")
    @test haskey(result, "usage")
end

# ── Responses API: new fields ────────────────────────────────────────────────

@testset "Responses API — with service_tier" begin
    r = respond("Say 'tier test' and nothing else.", model="gpt-5.6-luna", service_tier="auto")
    @test r isa ResponseSuccess
    @test !isempty(output_text(r))
end

@testset "Responses API — with structured input messages" begin
    r = respond([
        InputMessage(role="developer", content="You are helpful."),
        InputMessage(role="user", content="Say 'structured' and nothing else.")
    ], model="gpt-5.6-luna")
    @test r isa ResponseSuccess
    @test occursin("structured", lowercase(output_text(r)))
end

@testset "Responses API — with multimodal input" begin
    r = respond([InputMessage(
        role="user",
        content=[input_text("What does 2+2 equal? Reply with just the number.")]
    )], model="gpt-5.6-luna")
    @test r isa ResponseSuccess
    @test occursin("4", output_text(r))
end

@testset "Responses API — with reasoning (o-series)" begin
    r = respond(
        "What is 15 * 23? Reply with just the number.",
        model="gpt-5.6-luna",
        reasoning=Reasoning(effort="low")
    )
    @test r isa ResponseSuccess
    @test occursin("345", output_text(r))
end

@testset "Responses API — with max_output_tokens" begin
    r = respond("Say 'token limit' and nothing else.", model="gpt-5.6-luna", reasoning=Reasoning(effort="none"), max_output_tokens=Int64(100))
    @test r isa ResponseSuccess
    @test !isempty(output_text(r))
end

@testset "Responses API — with temperature" begin
    r = respond("Say 'temp test' and nothing else.", model="gpt-5.6-luna", reasoning=Reasoning(effort="none"), temperature=0.0)
    @test r isa ResponseSuccess
    @test !isempty(output_text(r))
end

@testset "Responses API — with store=false" begin
    r = respond("Say 'no store' and nothing else.", model="gpt-5.6-luna", store=false)
    @test r isa ResponseSuccess
    @test !isempty(output_text(r))
end

@testset "Responses API — with metadata" begin
    r = respond("Say 'meta' and nothing else.", model="gpt-5.6-luna", metadata=Dict("test_id" => "integration_123"))
    @test r isa ResponseSuccess
    @test !isempty(output_text(r))
end

# ── Cost Tracking ────────────────────────────────────────────────────────────

@testset "cost tracking — token_usage from real API call" begin
    chat = Chat(model="gpt-6-luna", reasoning_effort="none", temperature=0.0)
    push!(chat, Message(Val(:system), "You are a calculator."))
    push!(chat, Message(Val(:user), "What is 1+1? Answer with just the number."))

    result = chatrequest!(chat)
    @test result isa LLMSuccess

    u = token_usage(result)
    @test u.prompt_tokens > 0
    @test u.completion_tokens > 0
    @test u.total_tokens == u.prompt_tokens + u.completion_tokens
end

@testset "cost tracking — estimated_cost with real usage" begin
    chat = Chat(model="gpt-6-luna", reasoning_effort="none", temperature=0.0)
    push!(chat, Message(Val(:system), "You are helpful."))
    push!(chat, Message(Val(:user), "Say 'hi'."))

    result = chatrequest!(chat)
    @test result isa LLMSuccess

    cost = estimated_cost(result)
    @test cost > 0.0
end

@testset "cost tracking — cumulative_cost accumulates" begin
    chat = Chat(model="gpt-6-luna", reasoning_effort="none", temperature=0.0)
    push!(chat, Message(Val(:system), "You are a calculator."))
    push!(chat, Message(Val(:user), "What is 1+1? Answer with just the number."))

    r1 = chatrequest!(chat)
    @test r1 isa LLMSuccess
    cost_after_1 = cumulative_cost(chat)
    @test cost_after_1 > 0.0

    push!(chat, Message(Val(:user), "What is 2+2? Answer with just the number."))
    r2 = chatrequest!(chat)
    @test r2 isa LLMSuccess
    @test cumulative_cost(chat) > cost_after_1
end

# ── Fork Integration ─────────────────────────────────────────────────────────

@testset "fork — continue conversation from fork" begin
    chat = Chat(model="gpt-5.6-luna", reasoning_effort="none", temperature=0.0)
    push!(chat, Message(Val(:system), "You are a calculator. Answer with just the number."))
    push!(chat, Message(Val(:user), "What is 42?"))

    r1 = chatrequest!(chat)
    @test r1 isa LLMSuccess
    @test occursin("42", r1.message.content)
    @test length(chat) == 3

    forked = fork(chat)
    push!(forked, Message(Val(:user), "Add 8 to that. Answer with just the number."))
    r2 = chatrequest!(forked)
    @test r2 isa LLMSuccess
    @test occursin("50", r2.message.content)

    @test length(forked) == 5  # system + user1 + assistant1 + user2 + assistant2
    @test length(chat) == 3    # original unchanged
end

# ── Multi-turn Chat Completions ──────────────────────────────────────────────

@testset "multi-turn Chat Completions" begin
    chat = Chat(model="gpt-5.6-luna", reasoning_effort="none", temperature=0.0)
    push!(chat, Message(Val(:system), "You are a calculator. Answer with just the number."))

    push!(chat, Message(Val(:user), "What is 2+2?"))
    r1 = chatrequest!(chat)
    @test r1 isa LLMSuccess
    @test occursin("4", r1.message.content)

    push!(chat, Message(Val(:user), "Multiply that by 3."))
    r2 = chatrequest!(chat)
    @test r2 isa LLMSuccess
    @test occursin("12", r2.message.content)

    @test length(chat) == 5  # system + 2 user + 2 assistant
end

# ── GenericOpenAIEndpoint ────────────────────────────────────────────────────

@testset "GenericOpenAIEndpoint — against OpenAI" begin
    endpoint = GenericOpenAIEndpoint("https://api.openai.com", ENV["OPENAI_API_KEY"])
    chat = Chat(service=endpoint, model="gpt-5.6-luna")
    push!(chat, Message(Val(:system), "You are helpful."))
    push!(chat, Message(Val(:user), "Say 'hello' and nothing else."))

    result = chatrequest!(chat)
    @test result isa LLMSuccess
    @test !isempty(result.message.content)
end

# ── Responses API: what a reply says about its inputs, and the GPT-Image-2.6 tool controls ──────────────

@testset "Responses API — input_transformations: a reasoning item replayed on another model is withheld" begin
    # On 2026-11-16 gpt-5.6-luna withheld gpt-6-luna's reasoning item: [{"action":"removed","id":"rs_…",
    # "reason":"model_mismatch","type":"reasoning"}]. A first turn that makes no reasoning item has none to withhold.
    q1 = InputMessage(role="user", content="How many primes are there below 200? Answer with the number only.")
    ask(model, input) = respond(Respond(; model, input, reasoning=Reasoning(effort="medium"), store=false,
                                        include=["reasoning.encrypted_content"], max_output_tokens=3000))
    r1 = ask("gpt-6-luna", [q1])
    @test r1 isa ResponseSuccess && input_transformations(r1) == InputTransformation[]
    @test haskey(r1.response.raw, "input_transformations")          # the reply carries the list, empty
    ids = [item["id"] for item in reasoning_items(r1)]
    r2 = ask("gpt-5.6-luna", Any[q1, r1.response.output...,
                                 InputMessage(role="user", content="And below 300? Answer with the number only.")])
    @test r2 isa ResponseSuccess
    withheld = input_transformations(r2)
    @info "input_transformations live" ids withheld cost = estimated_cost(r1) + estimated_cost(r2)
    @test [t.id for t in withheld] == ids
    @test all(t -> (t.type, t.action, t.reason) == ("reasoning", "removed", :model_mismatch), withheld)
end

@testset "Responses API — image_context_compression is sent and the reply's resolved policy is read" begin
    r = respond(Respond(model="gpt-6-luna", input="Reply with the single word: ok", reasoning=Reasoning(effort="none"),
                        max_output_tokens=16, store=false, image_context_compression="disabled"))
    @test r isa ResponseSuccess && r.response.image_context_compression == "disabled"
    c = compact_response(model="gpt-6-luna", image_context_compression="disabled",
                         input=[Dict("role" => "user", "content" => "Hello."),
                                Dict("type" => "message", "role" => "assistant", "status" => "completed",
                                     "content" => [Dict("type" => "output_text", "text" => "Hi.")])])
    @test c isa AbstractDict && c["image"] == Dict("context_compression" => "disabled")
end

@testset "Responses API — a streamed reply's input_transformations: a reasoning item replayed on another model" begin
    # As the unstreamed testset above, the second turn streamed: the list arrives in response.completed.
    q1 = InputMessage(role="user", content="How many primes are there below 200? Answer with the number only.")
    ask(model, input; kw...) = respond(Respond(; model, input, reasoning=Reasoning(effort="medium"), store=false,
                                               include=["reasoning.encrypted_content"], max_output_tokens=3000, kw...))
    r1 = ask("gpt-6-luna", [q1])
    @test r1 isa ResponseSuccess
    ids = [item["id"] for item in reasoning_items(r1)]
    r2 = fetch(ask("gpt-5.6-luna", Any[q1, r1.response.output...,
                                        InputMessage(role="user", content="And below 300? Answer with the number only.")];
                   stream=true))
    @test r2 isa ResponseSuccess
    withheld = input_transformations(r2)
    @info "input_transformations live (streamed)" ids withheld cost = estimated_cost(r1) + estimated_cost(r2)
    @test [t.id for t in withheld] == ids
    @test all(t -> (t.type, t.action, t.reason) == ("reasoning", "removed", :model_mismatch), withheld)
end

@testset "Responses API — image_context_compression enabled, streamed, and auto, which resolves to a policy" begin
    ask(v; kw...) = respond(Respond(; model="gpt-6-luna", input="Reply with the single word: ok",
                                    reasoning=Reasoning(effort="none"), max_output_tokens=16, store=false,
                                    image_context_compression=v, kw...))
    s = fetch(ask("enabled"; stream=true))
    @test s isa ResponseSuccess && s.response.image_context_compression == "enabled"
    a = ask("auto")
    @test a isa ResponseSuccess && a.response.image_context_compression in ("enabled", "disabled")
    @info "image_context_compression live" streamed = s.response.image_context_compression auto = a.response.image_context_compression
end

@testset "Files API — expires_after, order, poll_file, the delete's object; an Uploads file with user_data and an expiry" begin
    # Free: file operations bill nothing; two 22-byte files, deleted at the end, and storage for seconds.
    path = joinpath(mktempdir(), "unilm-live.txt")
    write(path, "UniLM live Files test\n")
    up = upload_file(path, "user_data"; expires_after=3600)
    @test up isa FileSuccess
    if up isa FileSuccess
        f = up.response
        try
            @test (f.filename, f.bytes, f.purpose) == ("unilm-live.txt", 22, "user_data")
            @test f.raw["expires_at"] - f.created_at == 3600
            listed = list_files(; purpose="user_data", order="desc", limit=20)
            @test listed isa FileListSuccess && any(x -> x.id == f.id, listed.response.data)
            @test poll_file(f.id; interval=1.0, timeout=60.0) isa FileSuccess
        finally
            del = delete_file(f.id)
            @test del isa FileDeleteSuccess && (del.id, del.object) == (f.id, "file")
        end
    end
    # The Uploads API takes `user_data`, which its reference does not list, and passes the expiry on.
    u = create_upload(; filename="unilm-live.txt", purpose="user_data", bytes=22, mime_type="text/plain",
                      expires_after=3600)
    @test u isa UploadSuccess
    if u isa UploadSuccess
        part = add_upload_part(u.response.id, read(path))
        @test part isa UploadPartSuccess
        done = part isa UploadPartSuccess ? complete_upload(u.response.id, [part.response.id]) : cancel_upload(u.response.id)
        @test done isa UploadSuccess && done.response.status == "completed"
        if done isa UploadSuccess && !isnothing(done.response.file)
            file = done.response.file
            @test file.purpose == "user_data" && file.raw["expires_at"] - file.created_at == 3600
            @test delete_file(file.id) isa FileDeleteSuccess
        end
    end
end

@testset "Vector Stores API — a store's update, its files and their attributes, a batch's cancel and files, search" begin
    # Two small files, one store that expires after a day of disuse; everything made is deleted below.
    dir = mktempdir()
    texts = ("returns.txt" => "Jackets can be returned within 30 days of delivery for a full refund to the original card.\n",
             "shipping.txt" => "Orders ship within 2 business days. Express shipping arrives the next business day.\n")
    up = [upload_file((p = joinpath(dir, name); write(p, text); p), "user_data") for (name, text) in texts]
    @test all(u -> u isa FileSuccess, up)
    fa, fb = up[1].response.id, up[2].response.id
    vs = create_vector_store(; name="unilm-live-vector-stores", description="UniLM live test store",
                             expires_after=Dict("anchor" => "last_active_at", "days" => 1))
    @test vs isa VectorStoreSuccess
    vid = vector_store_id(vs.response)
    @info "create_vector_store live" vs.response.raw
    try
        added = add_vector_store_file(vid, fa; attributes=Dict("team" => "returns", "year" => 2026))
        @test added isa VectorStoreFileSuccess
        # The file is indexed when its status leaves "in_progress".
        f = added
        for _ in 1:60
            f.response.status == "in_progress" || break
            sleep(1)
            f = retrieve_vector_store_file(vid, fa)
            f isa VectorStoreFileSuccess || break
        end
        @test f isa VectorStoreFileSuccess && f.response.status == "completed"
        @test f.response.raw["attributes"] == Dict("team" => "returns", "year" => 2026)

        # A batch of one entry with attributes of its own, cancelled at once, then its file list.
        b = create_file_batch(vid, [VectorStoreFileEntry(file_id=fb, attributes=Dict("team" => "shipping"))])
        @test b isa VectorStoreBatchSuccess
        c = cancel_file_batch(vid, b.response.id)
        @info "cancel_file_batch live" c
        done = poll_file_batch(vid, b.response.id; interval=1.0, timeout=60.0)
        @test done isa VectorStoreBatchSuccess && done.response.status in ("completed", "cancelled")
        bf = list_file_batch_files(vid, b.response.id; order="asc")
        @info "list_file_batch_files live" done.response.status bf
        @test bf isa VectorStoreFileListSuccess && [x.id for x in bf.response.data] == [fb]

        listed = list_vector_store_files(vid; filter="completed", order="asc", limit=10)
        @test listed isa VectorStoreFileListSuccess && fa in [x.id for x in listed.response.data]

        u = update_vector_store_file(vid, fa, Dict("team" => "returns", "reviewed" => true))
        @test u isa VectorStoreFileSuccess && u.response.raw["attributes"] == Dict("team" => "returns", "reviewed" => true)

        content = vector_store_file_content(vid, fa)
        @test content isa VectorStoreContentSuccess
        @test occursin("30 days", join(part["text"] for part in content.response.data))

        hits = search_vector_store(vid, "How long do I have to return a jacket?"; max_num_results=2,
                                   filters=Dict("type" => "eq", "key" => "team", "value" => "returns"))
        @test hits isa VectorStoreSearchSuccess && !isempty(hits.response.data)
        h = first(hits.response.data)
        @test (h.file_id, h.filename, h.attributes["team"]) == (fa, "returns.txt", "returns")
        @test 0 <= h.score <= 1 && occursin("30 days", join(part["text"] for part in h.content))
        @info "search_vector_store live" hits.response.search_query h.score

        renamed = update_vector_store(vid; name="unilm-live-vector-stores-renamed", metadata=Dict("suite" => "live"))
        @test renamed isa VectorStoreSuccess && renamed.response.name == "unilm-live-vector-stores-renamed"
        @test renamed.response.raw["metadata"] == Dict("suite" => "live")

        page = list_vector_stores(; limit=5, order="desc")
        @test page isa VectorStoreListSuccess
        older = list_vector_stores(; limit=5, order="desc", after=vid)
        @test older isa VectorStoreListSuccess && vid ∉ [x.id for x in older.response.data]
        newer = list_vector_stores(; limit=5, order="desc", before=vid)
        @test newer isa VectorStoreListSuccess && vid ∉ [x.id for x in newer.response.data]

        gone = delete_vector_store_file(vid, fa)
        @test gone isa VectorStoreDeleteSuccess && gone.id == fa
    finally
        @test delete_vector_store(vid) isa VectorStoreDeleteSuccess
        foreach(id -> delete_file(id), (fa, fb))
    end
end

@testset "Containers API — memory, network policy, a name filter, files by id and by upload, their content" begin
    # No outbound access: an organization whose policy disables container network access gets a 400 for an allowlist.
    dir = mktempdir()
    a, b = joinpath(dir, "prices.csv"), joinpath(dir, "notes.txt")
    write(a, "item,price\nlantern,24.50\n")
    write(b, "two files\n")
    name = "unilm-live-container-$(rand(UInt32))"
    c = create_container(; name, memory_limit="4g", network_policy=ContainerNetworkDisabled())
    @test c isa ContainerSuccess
    cid = c.response.id
    up = upload_file(a, "user_data")
    try
        @test c.response.raw["memory_limit"] == "4g"
        @info "create_container live" c.response.raw
        # The listing can trail a create by a few seconds (on 2026-11-17 a release-gate run did not list it at once):
        # ask again for up to 20 s; the filter must then return exactly the new container.
        l = list_containers(; name, order="asc", limit=5)
        for _ in 1:10
            l isa ContainerListSuccess && !isempty(l.response.data) && break
            sleep(2)
            l = list_containers(; name, order="asc", limit=5)
        end
        @info "list_containers(name) live" ids = (l isa ContainerListSuccess ? [x["id"] for x in l.response.data] : l)
        @test l isa ContainerListSuccess && [x["id"] for x in l.response.data] == [cid]
        @test up isa FileSuccess
        byid = add_container_file(cid; file_id=up.response.id)
        @test byid isa ContainerSuccess && byid.response isa UniLM.ContainerFileObject
        byup = add_container_file(cid, b)
        @test byup isa ContainerSuccess
        files = list_container_files(cid; order="asc", limit=10)
        @test files isa ContainerFileListSuccess
        @test issubset([byid.response.id, byup.response.id], [x.id for x in files.response.data])
        g = retrieve_container_file(cid, byid.response.id)
        @test g isa ContainerSuccess && g.response.path == byid.response.path
        bytes = container_file_content(cid, byid.response.id)
        @test bytes isa FileContentSuccess && bytes.content == read(a)
        gone = delete_container_file(cid, byid.response.id)
        @test gone isa ContainerDeleteSuccess && gone.id == byid.response.id
    finally
        @test delete_container(cid) isa ContainerDeleteSuccess
        up isa FileSuccess && delete_file(up.response.id)
    end
end

@testset "Containers API — a skill reference reaches OpenAI" begin
    # No skill of that id exists, so OpenAI's answer names the id it read from the reference.
    r = create_container(; name="unilm-live-skill", skills=[SkillReference("skill_unilm_live_missing"; version="latest")])
    r isa ContainerSuccess && delete_container(r.response.id)
    @info "create_container with a skill reference live" r
    @test r isa ContainerFailure && occursin("skill_unilm_live_missing", r.response)
end

@testset "Containers API — an allowlist with a domain secret reaches OpenAI, whose organization policy refuses it" begin
    # OpenAI checks the policy's shape before the organization's rule: a secret without its value, or a string as
    # `allowed_domains`, draws a 400 naming that field, and a well-formed allowlist one naming `network_policy.type`
    # and the organization policy (witnessed 2026-11-16). Nothing is made.
    p = ContainerNetworkAllowlist(["example.com"];
                                  domain_secrets=[ContainerDomainSecret("example.com", "UNILM_LIVE_TOKEN", "not-a-secret")])
    r = create_container(; name="unilm-live-allowlist", network_policy=p)
    r isa ContainerSuccess && delete_container(r.response.id)
    @info "create_container with an allowlist live" r
    @test r isa ContainerFailure && r.status == 400
    err = r isa ContainerFailure ? JSON.parse(r.response)["error"] : Dict{String,Any}()
    @test get(err, "param", nothing) == "network_policy.type" && occursin("organization policy", get(err, "message", ""))
end

@testset "Responses API — code_interpreter(memory_limit) sets the memory of the container it makes" begin
    r = respond(Respond(model="gpt-6-luna", input="Use the python tool to print 6*7.", reasoning=Reasoning(effort="none"),
                        tools=[code_interpreter(memory_limit="4g")], tool_choice="required", max_output_tokens=300))
    @test r isa ResponseSuccess
    calls = code_interpreter_outputs(r)
    @test !isempty(calls)
    cid = first(calls)["container_id"]
    try
        c = retrieve_container(cid)
        @test c isa ContainerSuccess && c.response.raw["memory_limit"] == "4g"
        @info "code_interpreter(memory_limit) live" cost = estimated_cost(r) memory = c.response.raw["memory_limit"]
    finally
        @test delete_container(cid) isa ContainerDeleteSuccess
    end
end

@testset "Batch API — a batch of two Chats: create, retrieve, list and cancel" begin
    # Two 16-token gpt-6-luna requests at effort "none", cancelled before they run: OpenAI bills the tokens of the
    # requests a batch completed, so the run costs at most those two requests at the Batch rates. The gate does
    # not wait for this batch's results (OpenAI's 24-hour window): the next testset reads those of a batch an
    # earlier run left, and test/batch.jl reads every result shape from fixtures.
    luna() = Chat(model="gpt-6-luna", reasoning_effort="none", max_tokens=16, messages=[Message(Val(:user), "Reply with exactly: ok")])
    created = create_batch(["one" => luna(), "two" => luna()]; metadata=Dict("suite" => "unilm-live"),
                           output_expires_after=3600)
    @test created isa BatchSuccess
    if created isa BatchSuccess
        id = created.response.id
        @test created.response.endpoint == "/v1/chat/completions" && created.response.raw["metadata"] == Dict("suite" => "unilm-live")
        seen = retrieve_batch(id)
        @test seen isa BatchSuccess && seen.response.id == id
        page = list_batches(; limit=10)
        @test page isa BatchListSuccess && any(b -> b.id == id, page.response.data)
        stopped = cancel_batch(id)
        @test stopped isa BatchSuccess && stopped.response.status in ("cancelling", "cancelled")
        @info "OpenAI batch" id status = stopped.response.status input = created.response.input_file_id
        @test delete_file(created.response.input_file_id) isa FileDeleteSuccess
    end
end

# OpenAI's queue held a batch for over an hour on 2026-11-16, so the gate waits for none: each run leaves one tagged
# 1-request batch behind, never cancelled, and reads the newest completed one an earlier run left. A run that finds
# none fails: the gate's claim to read OpenAI batch results rests on it. A run bills its one request (USD ~2e-6).
@testset "Batch API — the results of a batch an earlier run left; one left for the next run" begin
    tag = "unilm_gate" => "results"
    luna = Chat(model="gpt-6-luna", reasoning_effort="none", max_tokens=16,
                messages=[Message(Val(:user), "Reply with exactly: ok")])
    left = create_batch(["gate" => luna]; metadata=Dict(tag), output_expires_after=2_592_000)
    @test left isa BatchSuccess && left.response.raw["metadata"] == Dict(tag)
    mine = left isa BatchSuccess ? left.response.id : nothing
    page = list_batches(; limit=100)
    @test page isa BatchListSuccess
    tagged(b) = b.id != mine && b.status == "completed" &&
                get(something(get(b.raw, "metadata", nothing), Dict()), first(tag), nothing) == last(tag)
    earlier = page isa BatchListSuccess ?
        sort(filter(tagged, page.response.data); by=b -> b.raw["created_at"], rev=true) : BatchObject[]
    @test !isempty(earlier)
    if !isempty(earlier)
        b = first(earlier)
        r = batch_results(b.id)
        @test r isa BatchResultsSuccess && first.(r.results) == ["gate"]
        x = r isa BatchResultsSuccess ? only(r.results).second : nothing
        @test x isa LLMSuccess && x.usage.service_tier == "batch" && estimated_cost(x) > 0
        @info "OpenAI gate batch" read = b.id left = mine cost = x isa LLMSuccess ? estimated_cost(x) : 0.0
    end
end

# The gate only reads stored Chat Completions. On 2026-11-17 OpenAI did not apply writes to them: completions sent with
# store=true were not readable 90 minutes later, a metadata update was answered HTTP 500, and a delete was answered
# `"deleted": true` while the completion stayed readable. So this testset stores, updates and deletes nothing, and takes
# no 5xx for a pass: it reads the oldest completion the listing shows and its input messages, and an empty listing fails
# it. Update and delete are covered keyless (test/chat_store.jl). USD 0 a run.
@testset "Stored Chat Completions — the oldest listed: get and its messages, read-only" begin
    page = list_chat_completions(; order="asc", limit=1)
    @test page isa StoredCompletionListSuccess
    listed = page isa StoredCompletionListSuccess ? page.response.data : StoredCompletion[]
    isempty(listed) && error("no stored Chat Completion to read: the listing answered a $(nameof(typeof(page))) with none")
    c = only(listed)
    got = get_chat_completion(c.id)
    @test got isa StoredCompletionSuccess && got.response.id == c.id && got.response.message.role == "assistant"
    msgs = list_chat_completion_messages(c.id; order="asc", limit=10)
    @test msgs isa StoredCompletionMessageListSuccess && !isempty(msgs.response.data)
    @info "Stored completion read" c.id c.model c.created messages = (
        msgs isa StoredCompletionMessageListSuccess ? length(msgs.response.data) : typeof(msgs))
end

# OpenAI's queue held a batch for over an hour on 2026-11-16, so the gate waits for none: each run leaves one tagged
# 2-request batch of embeddings behind and reads the newest completed one an earlier run left (a curl witness of
# 2026-11-17 left the first). A run that finds none fails. A run bills 3 tokens of text-embedding-3-small at the
# batch rate (USD ~3e-8).
@testset "Batch API — embeddings: the results of a batch an earlier run left; one left for the next run" begin
    tag = "unilm_gate" => "embeddings"
    small(x) = Embeddings(x; model="text-embedding-3-small", dimensions=8)
    requests = ["one" => small("alpha"), "two" => small(["beta", "gamma"])]
    left = create_batch(requests; metadata=Dict(tag), output_expires_after=2_592_000)
    @test left isa BatchSuccess && left.response.endpoint == "/v1/embeddings" && left.response.raw["metadata"] == Dict(tag)
    mine = left isa BatchSuccess ? left.response.id : nothing
    page = list_batches(; limit=100)
    @test page isa BatchListSuccess
    tagged(b) = b.id != mine && b.status == "completed" && b.endpoint == "/v1/embeddings" &&
                get(something(get(b.raw, "metadata", nothing), Dict()), first(tag), nothing) == last(tag)
    earlier = page isa BatchListSuccess ?
        sort(filter(tagged, page.response.data); by=b -> b.raw["created_at"], rev=true) : BatchObject[]
    @test !isempty(earlier)
    if !isempty(earlier)
        b = first(earlier)
        r = batch_results(b.id; requests)
        @test r isa BatchResultsSuccess && sort(first.(r.results)) == ["one", "two"]
        got = r isa BatchResultsSuccess ? Dict(r.results) : Dict()
        @test get(got, "one", nothing) isa EmbeddingSuccess && length(got["one"].embeddings.embeddings) == 8
        @test get(got, "two", nothing) isa EmbeddingSuccess && length.(got["two"].embeddings.embeddings) == [8, 8]
        @test all(x -> x isa EmbeddingSuccess && x.usage.service_tier == "batch" && estimated_cost(x) > 0, values(got))
        @info "OpenAI embedding gate batch" read = b.id left = mine cost = sum(
            (estimated_cost(x) for x in values(got) if x isa EmbeddingSuccess); init=0.0)
    end
end

end  # if OPENAI_API_KEY
