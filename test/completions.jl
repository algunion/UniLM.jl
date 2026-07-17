@testset "FIMCompletion construction" begin
    fim = FIMCompletion(service=DeepSeekEndpoint("key"), prompt="def fib(a):")
    @test fim.model == ""  # resolved to "deepseek-chat" during serialization
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

@testset "_prefix_complete_url dispatch" begin
    # src/completions.jl:194 — DeepSeek prefix completion uses the BETA base URL (the chat
    # path is appended to DEEPSEEK_BETA_BASE_URL, not the plain DEEPSEEK_BASE_URL). Assert the
    # exact composed string AND the literal so a wrong base constant would be caught.
    ds = DeepSeekEndpoint("key")
    @test UniLM._prefix_complete_url(ds) == UniLM.DEEPSEEK_BETA_BASE_URL * UniLM.CHAT_COMPLETIONS_PATH
    @test UniLM._prefix_complete_url(ds) == "https://api.deepseek.com/beta/v1/chat/completions"

    # src/completions.jl:195 — GenericOpenAIEndpoint prefix URL is rstrip(base_url,'/') *
    # CHAT_COMPLETIONS_PATH. A trailing-slash base_url proves the rstrip (no doubled slash).
    gen = GenericOpenAIEndpoint("https://host.example/", "k")
    @test UniLM._prefix_complete_url(gen) == rstrip(gen.base_url, '/') * UniLM.CHAT_COMPLETIONS_PATH
    @test UniLM._prefix_complete_url(gen) == "https://host.example/v1/chat/completions"
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
    push!(chat2, Message(Val(:system), "sys"))
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

@testset "FIMCallError carries an optional cause" begin
    to = UniLM.UniLMTimeout(:request, 0.5, 0.5)
    @test FIMCallError(error="timeout", status=nothing, cause=to).cause === to
    @test FIMCallError(error="x").cause === nothing
end

# ─── config-seam migration: bounded timeouts + interrupt discipline ──────────

using Sockets

# Localhost mute server (drains request, then holds without responding).
function _compl_mute_server(; hold::Float64=5.0)
    server = nothing; port = 0
    for attempt in 1:5
        tcp = Sockets.listen(Sockets.localhost, 0)
        port = Int(Sockets.getsockname(tcp)[2]); close(tcp)
        try
            server = HTTP.listen!("127.0.0.1", port; verbose=false) do http::HTTP.Stream
                read(http); sleep(hold)
            end
            break
        catch; attempt == 5 && rethrow(); end
    end
    server, "http://127.0.0.1:$port"
end

# Localhost canned-status server (fixed status + headers + body).
function _compl_canned_server(status::Int, body::String, hdrs::Vector{Pair{String,String}})
    server = nothing; port = 0
    for attempt in 1:5
        tcp = Sockets.listen(Sockets.localhost, 0)
        port = Int(Sockets.getsockname(tcp)[2]); close(tcp)
        try
            server = HTTP.listen!("127.0.0.1", port; verbose=false) do http::HTTP.Stream
                read(http)
                HTTP.setstatus(http, status)
                for (k, v) in hdrs; HTTP.setheader(http, k => v); end
                HTTP.startwrite(http); write(http, body)
            end
            break
        catch; attempt == 5 && rethrow(); end
    end
    server, "http://127.0.0.1:$port"
end

# Mock endpoint with :fim + :prefix_completion, base URL swappable per test.
const _COMPL_URL = Ref("http://127.0.0.1:0")
struct _ComplTimeoutMock <: UniLM.ServiceEndpoint end
UniLM.provider_capabilities(::Type{_ComplTimeoutMock}) = Set([:fim, :prefix_completion])
UniLM.auth_header(::Type{_ComplTimeoutMock}) = ["Content-Type" => "application/json"]
UniLM.default_fim_model(::Type{_ComplTimeoutMock}) = "mock-fim"
UniLM.get_url(::Type{_ComplTimeoutMock}, ::FIMCompletion) = _COMPL_URL[]
UniLM._prefix_complete_url(::Type{_ComplTimeoutMock}) = _COMPL_URL[]

@testset "fim_complete: mute server yields a bounded typed timeout" begin
    server, url = _compl_mute_server()
    _COMPL_URL[] = url
    cfg = RequestConfig(request_timeout=0.5, total_deadline=1.0, max_attempts=1)
    try
        fim = FIMCompletion(service=_ComplTimeoutMock, model="mock-fim", prompt="def f():")
        t = Threads.@spawn fim_complete(fim; config=cfg)
        @test timedwait(() -> istaskdone(t), 10.0) == :ok
        result = fetch(t)
        @test result isa FIMCallError
        @test result.status === nothing
        @test result.cause isa UniLM.UniLMTimeout
    finally
        close(server)
    end
end

@testset "fim_complete: Retry-After beyond the deadline fails immediately" begin
    server, url = _compl_canned_server(503, "{}", ["Retry-After" => "3600"])
    _COMPL_URL[] = url
    cfg = RequestConfig(request_timeout=0.5, total_deadline=1.0, max_attempts=3)
    try
        fim = FIMCompletion(service=_ComplTimeoutMock, model="mock-fim", prompt="x")
        t = Threads.@spawn fim_complete(fim; config=cfg)
        @test timedwait(() -> istaskdone(t), 10.0) == :ok
        result = fetch(t)
        @test result isa FIMFailure
        @test result.status == 503
    finally
        close(server)
    end
end

@testset "fim_complete: InterruptException is rethrown" begin
    struct _FimInterruptMock <: UniLM.ServiceEndpoint end
    UniLM.provider_capabilities(::Type{_FimInterruptMock}) = Set([:fim])
    UniLM.default_fim_model(::Type{_FimInterruptMock}) = "m"
    UniLM.get_url(::Type{_FimInterruptMock}, ::FIMCompletion) = "http://127.0.0.1:1/v1/completions"
    UniLM.auth_header(::Type{_FimInterruptMock}) = throw(InterruptException())
    @test_throws InterruptException fim_complete(FIMCompletion(service=_FimInterruptMock, model="m", prompt="x"))
end

@testset "prefix_complete: mute server yields a bounded typed timeout" begin
    server, url = _compl_mute_server()
    _COMPL_URL[] = url
    cfg = RequestConfig(request_timeout=0.5, total_deadline=1.0, max_attempts=1)
    try
        # `push!` refuses a lone assistant message on an empty chat (the
        # empty-conversation guard), so build the assistant-prefix chat through the
        # constructor — the same idiom the prefix mock-server testsets use.
        chat = Chat(service=_ComplTimeoutMock, model="mock-fim",
                    messages=[Message(role=UniLM.RoleAssistant, content="```python\n")])
        t = Threads.@spawn prefix_complete(chat; config=cfg)
        @test timedwait(() -> istaskdone(t), 10.0) == :ok
        result = fetch(t)
        @test result isa LLMCallError
        @test result.cause isa UniLM.UniLMTimeout   # the timeout is threaded into LLMCallError.cause
    finally
        close(server)
    end
end
