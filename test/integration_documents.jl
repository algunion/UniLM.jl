# Live witnesses that a PDF document reaches the model: the one-page fixture whose only text is
# "The secret word is PELICAN." (Helvetica), asked for the word. A model that answers it has read
# the document; without the document there is nothing to answer from.
const _DOC_PDF = joinpath(@__DIR__, "fixtures", "documents", "secret_word.pdf")
const _DOC_QUESTION = "What is the secret word? Answer with the word only."

if get(ENV, "UNILM_LIVE", "") == "1" && haskey(ENV, "OPENAI_API_KEY")
    @testset "OpenAI Chat reads a PDF document (file content part)" begin
        chat = Chat(model="gpt-6-luna", reasoning_effort="none", max_completion_tokens=16,
            messages=[Message(Val(:user), _DOC_QUESTION, DocumentAttachment(_DOC_PDF))])
        result = chatrequest!(chat)
        @info "OpenAI Chat document witness" result_type = typeof(result) usage = token_usage(result) cost = estimated_cost(result)
        @test result isa LLMSuccess
        @test occursin("PELICAN", uppercase(something(text(result), "")))
    end

    @testset "OpenAI Responses reads a PDF document (input_file)" begin
        result = respond(Respond(model="gpt-6-luna", reasoning=Reasoning(effort="none"), max_output_tokens=16,
            input=[InputMessage(role="user",
                content=[input_text(_DOC_QUESTION), input_file(DocumentAttachment(_DOC_PDF))])]))
        @info "OpenAI Responses document witness" result_type = typeof(result) usage = token_usage(result) cost = estimated_cost(result)
        @test result isa ResponseSuccess
        @test occursin("PELICAN", uppercase(output_text(result)))
    end
else
    @info "Skipping the OpenAI document witnesses (set UNILM_LIVE=1 and OPENAI_API_KEY)"
end
