# ─── DeepSeek Integration Tests ──────────────────────────────────────────────
# Requires UNILM_LIVE=1 and DEEPSEEK_API_KEY (both, or the suite skips).
# Seven billed calls on deepseek-flash; what they cost, estimated at DeepSeek's peak rates (it bills
# half of them off-peak), is logged at the end.

if !haskey(ENV, "DEEPSEEK_API_KEY") || get(ENV, "UNILM_LIVE", "") != "1"
    @info "Skipping DeepSeek integration tests (set UNILM_LIVE=1 and DEEPSEEK_API_KEY to run live)"
else

const _DS_LIVE_SPENT = Ref(0.0)
_ds_billed(r) = (_DS_LIVE_SPENT[] += estimated_cost(r); r)     # a failure is priced at 0.0

@testset "DeepSeek Chat — basic" begin
    chat = Chat(service=DeepSeekEndpoint(), model="deepseek-flash")
    push!(chat, Message(Val(:system), "You are a helpful assistant."))
    push!(chat, Message(Val(:user), "Reply with exactly: hello"))
    result = _ds_billed(chatrequest!(chat))
    @test result isa LLMSuccess
    @test !isempty(result.message.content)
end

@testset "DeepSeek FIM — basic" begin
    fim = FIMCompletion(
        service=DeepSeekEndpoint(),
        model="deepseek-flash",
        prompt="def fib(a):",
        suffix="    return fib(a-1) + fib(a-2)",
        max_tokens=128,
        stop=["\n\n"]
    )
    result = _ds_billed(fim_complete(fim))
    @test result isa FIMSuccess
    @test !isempty(fim_text(result))
    @test result.response.usage.total_tokens > 0
    @info "FIM result: $(fim_text(result))"
end

@testset "DeepSeek FIM — convenience" begin
    result = _ds_billed(fim_complete("def hello():",
        service=DeepSeekEndpoint(),
        max_tokens=64,
        stop=["\n\n"]))
    @test result isa FIMSuccess
    @test !isempty(fim_text(result))
end

@testset "DeepSeek Prefix Completion" begin
    chat = Chat(service=DeepSeekEndpoint(), model="deepseek-flash")
    push!(chat, Message(Val(:system), "You are a helpful coding assistant."))
    push!(chat, Message(Val(:user), "Write a Python hello world"))
    push!(chat, Message(role=RoleAssistant, content="```python\n"))
    @test length(chat) == 3
    result = _ds_billed(prefix_complete(chat))
    @test result isa LLMSuccess
    @test !isempty(result.message.content)
    # The result is the continuation; the history keeps the whole turn, prefix first.
    @test length(chat) == 3
    @test last(chat).role == UniLM.RoleAssistant
    @test last(chat).content == "```python\n" * result.message.content
    @info "Prefix result: $(result.message.content)"
end

@testset "DeepSeek Chat — function calling" begin
    gptfsig = FunctionSignature(
        name="get_current_weather",
        description="Get the current weather for a location",
        parameters=Dict(
            "type" => "object",
            "properties" => Dict(
                "location" => Dict("type" => "string", "description" => "City name"),
                "unit" => Dict("type" => "string", "enum" => ["celsius", "fahrenheit"])
            ),
            "required" => ["location"]
        )
    )

    chat = Chat(
        service=DeepSeekEndpoint(),
        model="deepseek-flash",
        tools=[Tool(func=gptfsig)],
        tool_choice="auto"
    )
    push!(chat, Message(Val(:system), "Use the provided tools when asked about weather."))
    push!(chat, Message(Val(:user), "What is the weather in Paris?"))

    result = _ds_billed(chatrequest!(chat))
    @test result isa LLMSuccess
    m = result.message
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

@testset "DeepSeek Chat — JSON output" begin
    chat = Chat(
        service=DeepSeekEndpoint(),
        model="deepseek-flash",
        response_format=UniLM.json_object()
    )
    push!(chat, Message(Val(:system), "Always respond in JSON format."))
    push!(chat, Message(Val(:user), "Tell me a joke with keys 'setup' and 'punchline'."))

    result = _ds_billed(chatrequest!(chat))
    @test result isa LLMSuccess
    parsed = JSON.parse(result.message.content)
    @test parsed isa AbstractDict
end

@testset "DeepSeek Chat — streaming" begin
    received_chunks = String[]

    chat = Chat(service=DeepSeekEndpoint(), model="deepseek-flash", stream=true)
    push!(chat, Message(Val(:system), "You are helpful."))
    push!(chat, Message(Val(:user), "Reply with exactly: hello"))

    callback = (chunk, close_ref) -> begin
        chunk isa String && push!(received_chunks, chunk)
    end
    task = chatrequest!(chat; callback=callback)
    result = _ds_billed(fetch(task))

    @test result isa LLMSuccess
    @test !isempty(result.message.content)
    @test !isempty(received_chunks)
end

@info "DeepSeek integration spend, estimated at peak rates" cost_usd = _DS_LIVE_SPENT[]

end  # if DEEPSEEK_API_KEY
