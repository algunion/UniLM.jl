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
