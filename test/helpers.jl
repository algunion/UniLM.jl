@testset "helpers.jl" begin
    # test for extractanswer function
    @test UniLM.extractanswer("```hello```") == "hello"
    @test UniLM.extractanswer("```hello```", include=true) == "```hello```"
    @test UniLM.extractanswer("```ggg-hello.aa```", first="```ggg-", last=".aa```") == "hello"
    @test UniLM.extractanswer("hey") == "hey"
    @test UniLM.extractanswer("<hello>", first="<", last=">") == "hello"    
end