@testset "openai-api.jl" begin
    conv = UniLM.Conversation()
    push!(conv, UniLM.Message(role=UniLM.GPTSystem, content="Act as a helpful AI agent."))
    @test length(conv) == 1

    push!(conv, UniLM.Message(role=UniLM.GPTUser, content="What is the purpose of life?"))
    @test length(conv) == 2

    try
        push!(conv, UniLM.Message(role=UniLM.GPTUser, content="What is the purpose of life?"))
    catch e
        @test e isa UniLM.InvalidConversationError
    end

    @test UniLM.is_send_valid(conv) == true


end