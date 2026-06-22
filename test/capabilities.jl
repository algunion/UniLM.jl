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

    @test :chat in UniLM.provider_capabilities(GEMINIServiceEndpoint)
    @test :embeddings in UniLM.provider_capabilities(GEMINIServiceEndpoint)

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
    @test which(UniLM.default_embedding_model, (typeof(gen),)).line == 65
    @test UniLM.default_embedding_model(gen) === nothing

    # src/capabilities.jl:66 — the `_` embedding fallback. AZUREServiceEndpoint is a TYPE with no
    # specific default_embedding_model method (only OPENAI/GEMINI types have one), so it lands on
    # the catch-all → nothing. Unknown services must have NO default embedding model.
    @test which(UniLM.default_embedding_model, (Type{UniLM.AZUREServiceEndpoint},)).line == 66
    @test UniLM.default_embedding_model(UniLM.AZUREServiceEndpoint) === nothing

    # src/capabilities.jl:70 — default_image_model has only an OPENAI method (line 69) and the
    # catch-all `_` (line 70); any instance other than the OPENAI type hits line 70 → nothing.
    @test which(UniLM.default_image_model, (typeof(ds),)).line == 70
    @test UniLM.default_image_model(ds) === nothing
    @test UniLM.default_image_model(gen) === nothing

    # src/capabilities.jl:74 — GenericOpenAIEndpoint-specific FIM method (more specific than
    # both ::DeepSeekEndpoint and `_`) returns nothing.
    @test which(UniLM.default_fim_model, (typeof(gen),)).line == 74
    @test UniLM.default_fim_model(gen) === nothing

    # src/capabilities.jl:75 — the `_` FIM fallback. OPENAIServiceEndpoint is a TYPE (not a
    # DeepSeekEndpoint/GenericOpenAIEndpoint instance), so it lands on the catch-all → nothing.
    @test which(UniLM.default_fim_model, (Type{OPENAIServiceEndpoint},)).line == 75
    @test UniLM.default_fim_model(OPENAIServiceEndpoint) === nothing
end

@testset "0.10 endpoint capabilities (consolidation)" begin
    new_caps = (:files, :vector_stores, :conversations, :moderation, :audio, :batch,
        :image_edits, :fine_tuning, :containers, :uploads, :video, :realtime)
    # OpenAI has them all
    for c in new_caps
        @test has_capability(OPENAIServiceEndpoint, c)
    end
    # Every non-OpenAI provider rejects them (has_capability false + validate throws)
    for svc in (GEMINIServiceEndpoint, AZUREServiceEndpoint, DeepSeekEndpoint("k"), GenericOpenAIEndpoint("http://x", ""))
        for c in new_caps
            @test !has_capability(svc, c)
            @test_throws ArgumentError UniLM.validate_capability(svc, c, "X")
        end
    end
    # Representative request fns reject a non-OpenAI provider BEFORE any network call
    @test_throws ArgumentError list_files(service=GEMINIServiceEndpoint)
    @test_throws ArgumentError create_vector_store(service=DeepSeekEndpoint("k"))
    @test_throws ArgumentError create_conversation(service=AZUREServiceEndpoint)
    @test_throws ArgumentError moderate("x"; service=AZUREServiceEndpoint)
    @test_throws ArgumentError create_batch("f", "/v1/responses"; service=GEMINIServiceEndpoint)
    @test_throws ArgumentError create_fine_tuning_job(model="m", training_file="f", service=GEMINIServiceEndpoint)
    @test_throws ArgumentError create_container(name="c", service=DeepSeekEndpoint("k"))
    @test_throws ArgumentError create_video(prompt="p", service=AZUREServiceEndpoint)
    @test_throws ArgumentError mint_realtime_secret(service=GenericOpenAIEndpoint("http://x", ""))
end

@testset "default-model resolution returns exact provider strings" begin
    ds = DeepSeekEndpoint("k")

    # default_model — Type dispatch for OPENAI/AZURE/GEMINI (capabilities.jl 55-57),
    # instance dispatch for DeepSeek (line 58). which().line pins the precise method;
    # the value assertion falsifies a wrong model string.
    @test which(UniLM.default_model, (Type{OPENAIServiceEndpoint},)).line == 55
    @test UniLM.default_model(OPENAIServiceEndpoint) == "gpt-5.5"
    @test which(UniLM.default_model, (Type{AZUREServiceEndpoint},)).line == 56
    @test UniLM.default_model(AZUREServiceEndpoint) == "gpt-5.2"
    @test which(UniLM.default_model, (Type{GEMINIServiceEndpoint},)).line == 57
    @test UniLM.default_model(GEMINIServiceEndpoint) == "gemini-2.5-flash"
    @test which(UniLM.default_model, (typeof(ds),)).line == 58
    @test UniLM.default_model(ds) == "deepseek-chat"

    # default_embedding_model — Type dispatch for OPENAI/GEMINI (62/63), instance for DeepSeek (64→nothing)
    @test which(UniLM.default_embedding_model, (Type{OPENAIServiceEndpoint},)).line == 62
    @test UniLM.default_embedding_model(OPENAIServiceEndpoint) == "text-embedding-3-small"
    @test which(UniLM.default_embedding_model, (Type{GEMINIServiceEndpoint},)).line == 63
    @test UniLM.default_embedding_model(GEMINIServiceEndpoint) == "gemini-embedding-001"
    @test which(UniLM.default_embedding_model, (typeof(ds),)).line == 64
    @test UniLM.default_embedding_model(ds) === nothing

    # default_image_model — OPENAI Type method (line 69)
    @test which(UniLM.default_image_model, (Type{OPENAIServiceEndpoint},)).line == 69
    @test UniLM.default_image_model(OPENAIServiceEndpoint) == "gpt-image-2"

    # default_fim_model — DeepSeek instance method (line 73)
    @test which(UniLM.default_fim_model, (typeof(ds),)).line == 73
    @test UniLM.default_fim_model(ds) == "deepseek-chat"
end
