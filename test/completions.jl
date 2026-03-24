@testset "FIMCompletion construction" begin
    fim = FIMCompletion(service=DeepSeekEndpoint("key"), prompt="def fib(a):")
    @test fim.model == "deepseek-chat"
    @test fim.prompt == "def fib(a):"
    @test isnothing(fim.suffix)
    @test fim.max_tokens == 128
end

@testset "FIMCompletion JSON serialization" begin
    fim = FIMCompletion(service=DeepSeekEndpoint("k"), prompt="hello", suffix="world",
        max_tokens=64, stop=["\n"])
    d = JSON.lower(fim)
    @test d[:prompt] == "hello"
    @test d[:suffix] == "world"
    @test d[:max_tokens] == 64
    @test d[:stop] == ["\n"]
    @test d[:model] == "deepseek-chat"
    # service not serialized
    @test !haskey(d, :service)
    # nil fields excluded
    @test !haskey(d, :echo)
    @test !haskey(d, :logprobs)
    @test !haskey(d, :temperature)
end

@testset "FIMCompletion URL routing" begin
    ds = DeepSeekEndpoint("key")
    fim = FIMCompletion(service=ds, prompt="test")
    @test UniLM.get_url(ds, fim) == "https://api.deepseek.com/beta/v1/completions"

    ollama = OllamaEndpoint()
    fim2 = FIMCompletion(service=ollama, prompt="test")
    @test UniLM.get_url(ollama, fim2) == "http://localhost:11434/v1/completions"

    gen = GenericOpenAIEndpoint("http://localhost:8000/", "")
    fim3 = FIMCompletion(service=gen, prompt="test")
    @test UniLM.get_url(gen, fim3) == "http://localhost:8000/v1/completions"
end

@testset "FIM response parsing" begin
    body = Dict{String,Any}(
        "choices" => [Dict{String,Any}("text" => "  if a <= 1:\n    return a\n", "index" => 0, "finish_reason" => "stop")],
        "model" => "deepseek-chat",
        "usage" => Dict{String,Any}("prompt_tokens" => 10, "completion_tokens" => 15, "total_tokens" => 25)
    )
    resp = HTTP.Response(200, [], Vector{UInt8}(JSON.json(body)))
    parsed = UniLM._parse_fim_response(resp)
    @test length(parsed.choices) == 1
    @test parsed.choices[1].text == "  if a <= 1:\n    return a\n"
    @test parsed.choices[1].finish_reason == "stop"
    @test parsed.usage.total_tokens == 25
    @test parsed.model == "deepseek-chat"
end

@testset "fim_text accessor" begin
    choice = FIMChoice(text="hello", finish_reason="stop")
    resp = FIMResponse(choices=[choice], usage=nothing, model="m", raw=Dict{String,Any}())
    @test fim_text(FIMSuccess(response=resp)) == "hello"
    @test fim_text(FIMFailure(response="err", status=400)) == ""
    @test fim_text(FIMCallError(error="err")) == ""

    # Empty choices
    empty_resp = FIMResponse(choices=FIMChoice[], model="m")
    @test fim_text(FIMSuccess(response=empty_resp)) == ""
end

@testset "fim_complete capability validation" begin
    # OpenAI does not support FIM
    fim = FIMCompletion(service=OPENAIServiceEndpoint, prompt="test")
    @test_throws ArgumentError fim_complete(fim)
end

@testset "prefix_complete validation" begin
    # Empty chat
    chat = Chat(service=DeepSeekEndpoint("k"))
    @test_throws ArgumentError prefix_complete(chat)

    # Last message not assistant
    chat2 = Chat(service=DeepSeekEndpoint("k"))
    push!(chat2, Message(Val(:user), "hello"))
    @test_throws ArgumentError prefix_complete(chat2)
end

@testset "DeepSeekEndpoint dispatch" begin
    ds = DeepSeekEndpoint("test-key")
    @test ds isa ServiceEndpoint
    @test ds isa ServiceEndpointSpec
    @test UniLM.get_url(ds, Chat()) == "https://api.deepseek.com/v1/chat/completions"
    @test UniLM.get_url(ds, Embeddings("test")) == "https://api.deepseek.com/v1/embeddings"
    @test UniLM._api_base_url(ds) == "https://api.deepseek.com"

    hdrs = UniLM.auth_header(ds)
    @test any(p -> p.first == "Authorization" && p.second == "Bearer test-key", hdrs)
end
