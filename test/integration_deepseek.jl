# ─── DeepSeek Integration Tests ──────────────────────────────────────────────
# Requires DEEPSEEK_API_KEY environment variable

if !haskey(ENV, "DEEPSEEK_API_KEY")
    @info "Skipping DeepSeek integration tests (DEEPSEEK_API_KEY not set)"
else

@testset "DeepSeek Chat — basic" begin
    chat = Chat(service=DeepSeekEndpoint(), model="deepseek-chat")
    push!(chat, Message(Val(:system), "You are a helpful assistant."))
    push!(chat, Message(Val(:user), "Reply with exactly: hello"))
    result = chatrequest!(chat)
    @test result isa LLMSuccess
    @test !isempty(result.message.content)
end

@testset "DeepSeek FIM — basic" begin
    fim = FIMCompletion(
        service=DeepSeekEndpoint(),
        model="deepseek-chat",
        prompt="def fib(a):",
        suffix="    return fib(a-1) + fib(a-2)",
        max_tokens=128,
        stop=["\n\n"]
    )
    result = fim_complete(fim)
    @test result isa FIMSuccess
    @test !isempty(fim_text(result))
    @test result.response.usage.total_tokens > 0
    @info "FIM result: $(fim_text(result))"
end

@testset "DeepSeek FIM — convenience" begin
    result = fim_complete("def hello():",
        service=DeepSeekEndpoint(),
        max_tokens=64,
        stop=["\n\n"])
    @test result isa FIMSuccess
    @test !isempty(fim_text(result))
end

@testset "DeepSeek Prefix Completion" begin
    chat = Chat(service=DeepSeekEndpoint(), model="deepseek-chat")
    push!(chat, Message(Val(:system), "You are a helpful coding assistant."))
    push!(chat, Message(Val(:user), "Write a Python hello world"))
    push!(chat, Message(role=RoleAssistant, content="```python\n"))
    result = prefix_complete(chat)
    @test result isa LLMSuccess
    @test !isempty(result.message.content)
    @info "Prefix result: $(result.message.content)"
end

end  # if DEEPSEEK_API_KEY
