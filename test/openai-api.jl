@testset "openai-api.jl" begin
    conv = UniLM.Conversation()
    push!(conv, UniLM.Message(role=UniLM.GPTSystem, content="Act as a helpful AI agent."))
    @test length(conv) == 1

    push!(conv, UniLM.Message(role=UniLM.GPTUser, content="Please offer 5 reasons why someone should program in Julia Programming Language."))
    @test length(conv) == 2

    try
        push!(conv, UniLM.Message(role=UniLM.GPTUser, content="Please offer 5 reasons why someone should program in Julia Programming Language."))
    catch e
        @test e isa UniLM.InvalidConversationError
    end

    @test UniLM.is_send_valid(conv) == true

    params = UniLM.ChatParams()

    @test params.messages |> isempty

    params_with_stream = UniLM.ChatParams(stream=true)
    UniLM.chat_request(conv, params=params)
    callback = (msg, close) -> begin         
        @info "from callback - echo: $msg" 
    end
    
    m = UniLM.chat_request(conv, params=params_with_stream, callback=callback) |> wait
    @info "awaited msg: $m"

 


end