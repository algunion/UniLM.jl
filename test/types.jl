@testset "types.jl" begin
    x = UniLM.model.chat
    @test x.endpoint == :chat
    y = UniLM.model.chat.gpt4
    @test y.name == :gpt4
    @test y.endpoint == :chat
end