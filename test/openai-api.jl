@testset "openai-api.jl" begin
    conv = UniLM.Conversation()
    push!(conv, UniLM.Message(role=UniLM.GPTSystem, content="Act as a helpful AI agent."))
    @test length(conv) == 1

    push!(conv, UniLM.Message(role=UniLM.GPTUser, content="Please tell me a one-liner joke."))
    @test length(conv) == 2

    try
        push!(conv, UniLM.Message(role=UniLM.GPTUser, content="Please tell me a one-liner joke."))
    catch e
        @test e isa UniLM.InvalidConversationError
    end

    @test UniLM.is_send_valid(conv) == true

    params = UniLM.ChatParams()

    @test params.messages |> isempty

    params_with_stream = UniLM.ChatParams(stream=true)
    UniLM.chat_request(conv, params=params)
    UniLM.chat_request(conv, params=params_with_stream)
    @info "Sleeping for 20 seconds to allow the streaming work"
    sleep(20)    


end