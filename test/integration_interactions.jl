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

@testset "Interactions — tool loop (live, neutral tool_result round-trip)" begin
    r = Respond(service=UniLM.GEMINIServiceEndpoint, model="gemini-3.1-flash-lite",
                input="What is the weather in Tokyo? Use the get_weather tool, then tell me.",
                tools=[function_tool("get_weather", "Get current weather for a city",
                    parameters=Dict("type" => "object",
                        "properties" => Dict("city" => Dict("type" => "string")), "required" => ["city"]))])
    # the whole point of Plan 3a: tool_loop submits the neutral tool_result item, which the
    # Gemini encoder translates to function_result — a round-trip green mocks could not prove.
    result = tool_loop(r, (name, args) -> "The weather in $(get(args, "city", "?")) is sunny, 22C."; max_turns=4)
    @test result.completed
    @test !isempty(result.tool_calls)
    @test result.tool_calls[1].tool_name == "get_weather"
    @test result.response isa ResponseSuccess
    @test !isempty(output_text(result.response))
end

@testset "Interactions — tool_choice required forces a call (live)" begin
    forced = respond(Respond(service=UniLM.GEMINIServiceEndpoint, model="gemini-3.1-flash-lite",
        input="Just say hello, nothing else.", tool_choice="required",
        tools=[function_tool("get_weather", "Get weather",
            parameters=Dict("type" => "object", "properties" => Dict("city" => Dict("type" => "string"))))]))
    @test forced isa ResponseSuccess
    @test !isempty(function_calls(forced))
end

@testset "Interactions — estimated_cost > 0 (live usage)" begin
    res = respond(Respond(service=UniLM.GEMINIServiceEndpoint, model="gemini-3.1-flash-lite",
                          input="Name three primary colors."))
    @test res isa ResponseSuccess
    @test token_usage(res).prompt_tokens > 0
    @test estimated_cost(res) > 0
end

@testset "Interactions — background create → poll → cancel (live)" begin
    started = respond(Respond(service=UniLM.GEMINIServiceEndpoint, model="gemini-3.1-flash-lite",
                              input="Write one sentence about the ocean.", background=true))
    @test started isa ResponseSuccess
    id = started.response.id
    got = get_response(id; service=UniLM.GEMINIServiceEndpoint)   # bounded single poll
    @test got isa ResponseSuccess
    @test got.response.id == id
    @test got.response.status in ("in_progress", "completed")
    cancelled = cancel_response(id; service=UniLM.GEMINIServiceEndpoint)
    @test cancelled isa ResponseSuccess || cancelled isa ResponseFailure   # 200 decode, or benign if already completed
end

@testset "Interactions — google_search grounded round-trip (live)" begin
    res = respond(Respond(service=UniLM.GEMINIServiceEndpoint, model="gemini-3.1-flash-lite",
                          input="Who won the 2022 FIFA World Cup? Search if needed.",
                          tools=[gemini_google_search()]))
    @test res isa ResponseSuccess
    @test !isempty(output_text(res))
    @test any(o -> get(o, "type", "") == "google_search_call", res.response.output)
end

end  # if GEMINI_API_KEY
