# ─── Gemini Interactions integration tests (live) ────────────────────────────
# Requires GEMINI_API_KEY (billing-enabled). Uses gemini-3.1-flash-lite (cheapest).
# Run once when green; do not rerun. Exercises the agentic verb end-to-end:
# encode → HTTP → decode → neutral ResponseObject accessors.

if !haskey(ENV, "GEMINI_API_KEY")
    @info "Skipping Gemini Interactions integration tests (GEMINI_API_KEY not set)"
else

@testset "Interactions — text" begin
    r = Respond(service=UniLM.GEMINIServiceEndpoint, model="gemini-3.1-flash-lite",
                input="Reply with exactly one word: hello")
    result = respond(r)
    @test result isa ResponseSuccess
    @test !isempty(output_text(result))
    @test result.response.status == "completed"
end

@testset "Interactions — tool round-trip" begin
    r = Respond(service=UniLM.GEMINIServiceEndpoint, model="gemini-3.1-flash-lite",
                input="What is the weather in Tokyo? Call the get_weather function.",
                tools=[function_tool("get_weather", "Get current weather for a city",
                       parameters=Dict("type" => "object",
                                       "properties" => Dict("city" => Dict("type" => "string")),
                                       "required" => ["city"]))])
    result = respond(r)
    @test result isa ResponseSuccess
    calls = function_calls(result)
    @test !isempty(calls)
    @test calls[1]["name"] == "get_weather"
    @test result.response.status == "requires_action"
    # continue: submit the tool result via previous_interaction_id (captured shape:
    # function_result requires call_id + name + result)
    call = calls[1]
    r2 = Respond(service=UniLM.GEMINIServiceEndpoint, model="gemini-3.1-flash-lite",
                 previous_response_id=result.response.id,
                 input=[Dict("type" => "function_result", "call_id" => call["call_id"],
                             "name" => call["name"], "result" => Dict("temperature" => "22C"))])
    follow = respond(r2)
    @test follow isa ResponseSuccess
    @test !isempty(output_text(follow))
end

@testset "Interactions — streaming" begin
    payloads = Any[]
    r = Respond(service=UniLM.GEMINIServiceEndpoint, model="gemini-3.1-flash-lite",
                input="Count from one to five.", stream=true)
    task = respond(r; callback=(c, _) -> push!(payloads, c))
    result = fetch(task)
    @test result isa ResponseSuccess
    # interaction.completed omits steps → final output_text is rebuilt from streamed deltas
    @test !isempty(output_text(result))
    @test !isempty(payloads)
end

end  # if GEMINI_API_KEY
