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
