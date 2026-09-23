import Sockets

@testset "provider_capabilities" begin
    @test :chat in UniLM.provider_capabilities(OPENAIServiceEndpoint)
    @test :responses in UniLM.provider_capabilities(OPENAIServiceEndpoint)
    @test :embeddings in UniLM.provider_capabilities(OPENAIServiceEndpoint)
    @test :images in UniLM.provider_capabilities(OPENAIServiceEndpoint)
    @test :tools in UniLM.provider_capabilities(OPENAIServiceEndpoint)
    @test !(:fim in UniLM.provider_capabilities(OPENAIServiceEndpoint))

    @test :chat in UniLM.provider_capabilities(AZUREServiceEndpoint)
    @test :tools in UniLM.provider_capabilities(AZUREServiceEndpoint)
    @test !(:embeddings in UniLM.provider_capabilities(AZUREServiceEndpoint))

    @test :chat in UniLM.provider_capabilities(GEMINIOpenAIServiceEndpoint)
    @test :embeddings in UniLM.provider_capabilities(GEMINIOpenAIServiceEndpoint)

    ds = DeepSeekEndpoint("k")
    @test :chat in provider_capabilities(ds)
    @test :fim in provider_capabilities(ds)
    @test :prefix_completion in provider_capabilities(ds)
    @test :tools in provider_capabilities(ds)
    @test !(:images in provider_capabilities(ds))
    @test !(:responses in provider_capabilities(ds))

    gen = GenericOpenAIEndpoint("http://x", "")
    @test :chat in provider_capabilities(gen)
    @test :fim in provider_capabilities(gen)
    @test :embeddings in provider_capabilities(gen)
end

@testset "has_capability" begin
    @test has_capability(OPENAIServiceEndpoint, :chat)
    @test !has_capability(OPENAIServiceEndpoint, :fim)
    @test has_capability(DeepSeekEndpoint("k"), :fim)
    @test has_capability(DeepSeekEndpoint("k"), :prefix_completion)
end

@testset "validate_capability" begin
    @test_throws ArgumentError UniLM.validate_capability(OPENAIServiceEndpoint, :fim, "FIM")
    @test_throws ArgumentError UniLM.validate_capability(AZUREServiceEndpoint, :responses, "Responses API")

    # Should not throw
    UniLM.validate_capability(DeepSeekEndpoint("k"), :fim, "FIM")
    UniLM.validate_capability(OPENAIServiceEndpoint, :chat, "Chat")
    UniLM.validate_capability(GenericOpenAIEndpoint("http://x", ""), :fim, "FIM")
end

@testset "OllamaEndpoint" begin
    ollama = OllamaEndpoint()
    @test ollama isa GenericOpenAIEndpoint
    @test ollama.base_url == "http://localhost:11434"
    @test ollama.api_key == ""
    caps = provider_capabilities(ollama)
    @test :chat in caps
    @test :embeddings in caps
    @test :fim in caps
    @test :tools in caps
    @test :responses in caps
    @test UniLM.default_model(ollama) === nothing
end

@testset "MistralEndpoint" begin
    mistral = MistralEndpoint(api_key="test-key")
    @test mistral isa GenericOpenAIEndpoint
    @test mistral.base_url == "https://api.mistral.ai"
    @test mistral.api_key == "test-key"
    caps = provider_capabilities(mistral)
    @test :chat in caps
    @test :embeddings in caps
    @test :fim in caps
    @test :tools in caps
    @test UniLM.default_model(mistral) === nothing
end

@testset "default-model fallbacks return nothing" begin
    gen = GenericOpenAIEndpoint("http://x", "")
    ds = DeepSeekEndpoint("k")

    # src/capabilities.jl:65 — the GenericOpenAIEndpoint-specific embedding method (more
    # specific than the `_` fallback) returns nothing. Asserting `=== nothing` (not just
    # falsy) pins the exact return.
    @test UniLM.default_embedding_model(gen) === nothing

    # src/capabilities.jl:66 — the `_` embedding fallback. AZUREServiceEndpoint is a TYPE with no
    # specific default_embedding_model method (only OPENAI/GEMINI types have one), so it lands on
    # the catch-all → nothing. Unknown services must have NO default embedding model.
    @test UniLM.default_embedding_model(UniLM.AZUREServiceEndpoint) === nothing

    # src/capabilities.jl:70 — default_image_model has only an OPENAI method (line 69) and the
    # catch-all `_` (line 70); any instance other than the OPENAI type hits line 70 → nothing.
    @test UniLM.default_image_model(ds) === nothing
    @test UniLM.default_image_model(gen) === nothing

    # src/capabilities.jl:74 — GenericOpenAIEndpoint-specific FIM method (more specific than
    # both ::DeepSeekEndpoint and `_`) returns nothing.
    @test UniLM.default_fim_model(gen) === nothing

    # src/capabilities.jl:75 — the `_` FIM fallback. OPENAIServiceEndpoint is a TYPE (not a
    # DeepSeekEndpoint/GenericOpenAIEndpoint instance), so it lands on the catch-all → nothing.
    @test UniLM.default_fim_model(OPENAIServiceEndpoint) === nothing
end

@testset "0.10 endpoint capabilities (consolidation)" begin
    new_caps = (:files, :vector_stores, :conversations, :moderation, :audio, :batch,
        :image_edits, :fine_tuning, :containers, :uploads, :realtime)
    # OpenAI has them all
    for c in new_caps
        @test has_capability(OPENAIServiceEndpoint, c)
    end
    # Every non-OpenAI provider rejects them (has_capability false + validate throws)
    for svc in (GEMINIOpenAIServiceEndpoint, AZUREServiceEndpoint, DeepSeekEndpoint("k"), GenericOpenAIEndpoint("http://x", ""))
        for c in new_caps
            @test !has_capability(svc, c)
            @test_throws ArgumentError UniLM.validate_capability(svc, c, "X")
        end
    end
    # Representative request fns reject a non-OpenAI provider BEFORE any network call
    @test_throws ArgumentError list_files(service=GEMINIOpenAIServiceEndpoint)
    @test_throws ArgumentError create_vector_store(service=DeepSeekEndpoint("k"))
    @test_throws ArgumentError create_conversation(service=AZUREServiceEndpoint)
    @test_throws ArgumentError moderate("x"; service=AZUREServiceEndpoint)
    @test_throws ArgumentError create_batch("f", "/v1/responses"; service=GEMINIOpenAIServiceEndpoint)
    @test_throws ArgumentError create_fine_tuning_job(model="m", training_file="f", service=GEMINIOpenAIServiceEndpoint)
    @test_throws ArgumentError create_container(name="c", service=DeepSeekEndpoint("k"))
    @test_throws ArgumentError mint_realtime_secret(service=GenericOpenAIEndpoint("http://x", ""))
end

@testset "default-model resolution returns exact provider strings" begin
    ds = DeepSeekEndpoint("k")

    # default_model — Type dispatch for OPENAI/AZURE/GEMINI (capabilities.jl 55-57),
    # instance dispatch for DeepSeek (line 58). Each distinct return value uniquely
    # identifies (and covers) its specific method, and falsifies a wrong model string.
    @test UniLM.default_model(OPENAIServiceEndpoint) == "gpt-5.6-sol"
    @test UniLM.default_model(AZUREServiceEndpoint) == "gpt-5.2"
    @test UniLM.default_model(GEMINIOpenAIServiceEndpoint) == "gemini-3.8-flash"
    @test UniLM.default_model(ds) == "deepseek-flash"

    # default_embedding_model — Type dispatch for OPENAI/GEMINI (62/63), instance for DeepSeek (64→nothing)
    @test UniLM.default_embedding_model(OPENAIServiceEndpoint) == "text-embedding-3-small"
    @test UniLM.default_embedding_model(GEMINIOpenAIServiceEndpoint) == "gemini-embedding-001"
    @test UniLM.default_embedding_model(ds) === nothing

    # default_image_model — OPENAI Type method (line 69)
    @test UniLM.default_image_model(OPENAIServiceEndpoint) == "gpt-image-2"

    # default_fim_model — DeepSeek instance method (line 73)
    @test UniLM.default_fim_model(ds) == "deepseek-flash"
end

@testset "Anthropic — capabilities & defaults" begin
    @test has_capability(ANTHROPICServiceEndpoint, :chat)
    @test has_capability(ANTHROPICServiceEndpoint, :tools)
    @test has_capability(ANTHROPICServiceEndpoint, :streaming)
    @test !has_capability(ANTHROPICServiceEndpoint, :embeddings)
    @test UniLM.default_model(ANTHROPICServiceEndpoint) == "claude-opus-5-5"
    @test UniLM.default_max_tokens(ANTHROPICServiceEndpoint, "claude-opus-5-5") == 16000
    # A Chat with no model resolves to the Anthropic default.
    chat = Chat(service=ANTHROPICServiceEndpoint)
    @test chat.model == "claude-opus-5-5"
    @test UniLM.get_url(chat) == "https://api.anthropic.com/v1/messages"
    @test haskey(UniLM.DEFAULT_PRICING, "claude-opus-4-8")
    @test UniLM.DEFAULT_PRICING["claude-haiku-4-5"].output ≈ 5.0 / 1_000_000
    # Every default model is priced, so default-model spend is never a silent 0.0.
    ds = DeepSeekEndpoint("k")
    for model in (UniLM.default_model(ANTHROPICServiceEndpoint), UniLM.default_model(ds), UniLM.default_fim_model(ds))
        @test haskey(UniLM.DEFAULT_PRICING, model)
    end
end

# A user-defined endpoint — the documented way to reach an OpenAI-compatible
# backend this package does not ship. It declares NO capabilities, which is
# exactly why the request verbs must not refuse it.
struct _UndeclaredEndpoint <: UniLM.OpenAIWireEndpoint
    base_url::String
end
UniLM._api_base_url(e::_UndeclaredEndpoint) = e.base_url
UniLM.auth_header(::_UndeclaredEndpoint) = ["Content-Type" => "application/json"]
UniLM.get_url(e::_UndeclaredEndpoint, ::Chat) = e.base_url * "/v1/chat/completions"
UniLM.get_url(chat::Chat, e::_UndeclaredEndpoint) = UniLM.get_url(e, chat)

@testset "declared-capability predicate separates 'lacks it' from 'never said'" begin
    # provider_capabilities has no fallback method, so the two cases are distinct:
    # a declared endpoint answers, an undeclared one has no applicable method.
    @test UniLM._capability_declared(OPENAIServiceEndpoint)
    @test UniLM._capability_declared(AZUREServiceEndpoint)
    @test UniLM._capability_declared(GenericOpenAIEndpoint("http://x", ""))
    @test !UniLM._capability_declared(_UndeclaredEndpoint("http://x"))

    # Declared and lacking → throws. Declared and having → returns. Undeclared → returns.
    @test_throws ArgumentError UniLM._validate_declared_capability(AZUREServiceEndpoint, :images, "Images")
    @test UniLM._validate_declared_capability(OPENAIServiceEndpoint, :images, "Images") === nothing
    @test UniLM._validate_declared_capability(_UndeclaredEndpoint("http://x"), :images, "Images") === nothing

    # The agentic surface is declared under two names across the shipped wires.
    @test UniLM._validate_agentic_capability(OPENAIServiceEndpoint) === nothing      # :responses
    @test UniLM._validate_agentic_capability(GEMINIServiceEndpoint) === nothing      # :agentic
    @test UniLM._validate_agentic_capability(GenericOpenAIEndpoint("http://x", "")) === nothing
    @test UniLM._validate_agentic_capability(_UndeclaredEndpoint("http://x")) === nothing
    @test_throws ArgumentError UniLM._validate_agentic_capability(AZUREServiceEndpoint)
    @test_throws ArgumentError UniLM._validate_agentic_capability(ANTHROPICServiceEndpoint)
end

# Declares capabilities but no :chat — every shipped endpoint declares :chat, so the
# chat verb's gate needs an endpoint that says "I do embeddings, not chat".
struct _NoChatEndpoint <: UniLM.OpenAIWireEndpoint end
UniLM.provider_capabilities(::Type{_NoChatEndpoint}) = Set([:embeddings])

@testset "the four primary verbs validate before any network I/O" begin
    # The docs promise capability validation before dispatch; the verbs did none, so
    # e.g. generate_image against an endpoint that declares no :images opened a live
    # POST. The check now precedes every I/O statement — including URL construction —
    # so a rejection cannot have touched the network: there is no local fixture here
    # to hit, and a call that reached the wire would fail differently.
    @test_throws ArgumentError generate_image(ImageGeneration(prompt="p", service=AZUREServiceEndpoint))
    @test_throws ArgumentError generate_image("p"; service=AZUREServiceEndpoint)
    @test_throws ArgumentError embeddingrequest!(UniLM.Embeddings("x"; service=AZUREServiceEndpoint,
                                                                  model="text-embedding-3-small"))
    @test_throws ArgumentError respond(Respond(service=AZUREServiceEndpoint, input="x"))
    @test_throws ArgumentError chatrequest!(Chat(service=_NoChatEndpoint, model="m",
                                                 messages=[Message(Val(:user), "hi")]))
    # The message names the feature and what the provider does support.
    err = try generate_image(ImageGeneration(prompt="p", service=AZUREServiceEndpoint)); nothing catch e; e end
    @test occursin("Image Generation API is not supported", err.msg)
    @test occursin("chat", err.msg)
end

@testset "an undeclared endpoint still dispatches" begin
    # The non-breaking rule: a custom backend declares nothing, so the package has no
    # basis to refuse it. It must reach the wire exactly as before.
    hits = Ref(0)
    # OS-assigned ephemeral port with bind retry (the fixture idiom used across this
    # suite): a hand-picked port can collide with a live local service, whose reply
    # would poison the assertion.
    srv, port = nothing, 0
    for attempt in 1:5
        tcp = Sockets.listen(Sockets.localhost, 0)
        port = Int(Sockets.getsockname(tcp)[2])
        close(tcp)
        try
            srv = HTTP.serve!("127.0.0.1", port; verbose=false) do req
                hits[] += 1
                HTTP.Response(200, ["Content-Type" => "application/json"], JSON.json(Dict(
                    "choices" => [Dict("finish_reason" => "stop",
                                       "message" => Dict("role" => "assistant", "content" => "pong"))])))
            end
            break
        catch
            attempt == 5 && rethrow()
        end
    end
    try
        ep = _UndeclaredEndpoint("http://127.0.0.1:$port")
        chat = Chat(service=ep, model="m", messages=[Message(Val(:user), "ping")])
        r = chatrequest!(chat; config=RequestConfig(max_attempts=1, total_deadline=10.0))
        @test r isa LLMSuccess
        @test text(r) == "pong"
        @test hits[] == 1
    finally
        close(srv)
    end
end

@testset "capability flags: no Videos API; OpenAI-wire endpoints stream and emit JSON" begin
    # The OpenAI Videos API shut down on 2026-09-24.
    @test !has_capability(OPENAIServiceEndpoint, :video)
    for svc in (OPENAIServiceEndpoint, AZUREServiceEndpoint, GEMINIOpenAIServiceEndpoint,
                DeepSeekEndpoint("k"), GenericOpenAIEndpoint("http://x", ""))
        @test has_capability(svc, :streaming) && has_capability(svc, :json_output)
    end
end

# A marker-type endpoint that declares nothing and has no default model.
struct _NoDefaultModelEndpoint <: UniLM.OpenAIWireEndpoint end

@testset "capability and model errors name the endpoint, not DataType" begin
    for (svc, name) in ((OPENAIServiceEndpoint, "OPENAIServiceEndpoint"), (DeepSeekEndpoint("k"), "DeepSeekEndpoint"))
        err = try UniLM.validate_capability(svc, :system_one, "System One"); nothing catch e; e end
        @test err isa ArgumentError && occursin("System One is not supported by $name.", err.msg)
        @test !occursin("DataType", err.msg)
    end
    err = try UniLM._validate_agentic_capability(ANTHROPICServiceEndpoint); nothing catch e; e end
    @test err isa ArgumentError && occursin("ANTHROPICServiceEndpoint", err.msg) && !occursin("DataType", err.msg)
    # No default_model method: the documented ArgumentError, not a MethodError.
    err = try Chat(service=_NoDefaultModelEndpoint); nothing catch e; e end
    @test err isa ArgumentError && err.msg == "model must be specified when using _NoDefaultModelEndpoint"
    @test UniLM.default_model(_NoDefaultModelEndpoint) === nothing
    @test Chat(service=_NoDefaultModelEndpoint, model="m").model == "m"
    err = try UniLM.get_url(ANTHROPICServiceEndpoint, FIMCompletion(service=ANTHROPICServiceEndpoint, model="m", prompt="x")); nothing catch e; e end
    @test err isa ArgumentError && !occursin("DataType", err.msg)
    err = try JSON.lower(FIMCompletion(service=OPENAIServiceEndpoint, prompt="x")); nothing catch e; e end
    @test err isa ArgumentError && occursin("OPENAIServiceEndpoint", err.msg) && !occursin("DataType", err.msg)
end

@testset "Azure deployment: registry first, then AZURE_OPENAI_DEPLOY_NAME_<MODEL>" begin
    # <MODEL> is the model id upper-cased with every non-alphanumeric mapped to `_`.
    @test UniLM._azure_deploy_env_var("gpt-5.2") == "AZURE_OPENAI_DEPLOY_NAME_GPT_5_2"
    @test UniLM._azure_deploy_env_var("gpt-4o-mini") == "AZURE_OPENAI_DEPLOY_NAME_GPT_4O_MINI"
    withenv("AZURE_OPENAI_DEPLOY_NAME_GPT_4O_MINI" => "mini deploy", "AZURE_OPENAI_DEPLOY_NAME_GPT_5_2" => "d52") do
        @test UniLM._azure_deployment_path("gpt-4o-mini") == "/openai/deployments/mini%20deploy"
        @test UniLM._azure_deployment_path("gpt-5.2") == "/openai/deployments/d52"
        try
            add_azure_deploy_name!("gpt-4o-mini", "registered")
            @test UniLM._azure_deployment_path("gpt-4o-mini") == "/openai/deployments/registered"   # registry wins
        finally
            delete!(UniLM._MODEL_ENDPOINTS_AZURE_OPENAI, "gpt-4o-mini")
        end
    end
    for value in (nothing, "")                                  # unset, or set to nothing usable
        withenv("AZURE_OPENAI_DEPLOY_NAME_NEVER_CONFIGURED" => value) do
            err = try UniLM._azure_deployment_path("never-configured"); nothing catch e; e end
            @test err isa ArgumentError && occursin("AZURE_OPENAI_DEPLOY_NAME_NEVER_CONFIGURED", err.msg) &&
                  occursin("add_azure_deploy_name!", err.msg)
        end
    end
end
