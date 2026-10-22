# ─── Gemini Interactions integration tests (live) ────────────────────────────
# Requires UNILM_LIVE=1 and GEMINI_API_KEY (billing-enabled). Uses gemini-3.8-flash (a
# thinking model; low effort and a bounded output budget keep these probes small).
# Run once when green; do not rerun. Exercises the agentic verb end-to-end:
# encode → HTTP → decode → neutral ResponseObject accessors.

if !haskey(ENV, "GEMINI_API_KEY") || get(ENV, "UNILM_LIVE", "") != "1"
    @info "Skipping Gemini Interactions integration tests (set UNILM_LIVE=1 and GEMINI_API_KEY to run live)"
else

@testset "Interactions — text" begin
    r = Respond(service=UniLM.GEMINIServiceEndpoint, model="gemini-3.8-flash", reasoning=Reasoning(effort="low"), max_output_tokens=2048,
                input="Reply with exactly one word: hello")
    result = respond(r)
    @test result isa ResponseSuccess
    @test !isempty(output_text(result))
    @test result.response.status == "completed"
end

@testset "Interactions — tool round-trip" begin
    r = Respond(service=UniLM.GEMINIServiceEndpoint, model="gemini-3.8-flash", reasoning=Reasoning(effort="low"), max_output_tokens=2048,
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
    # continue: submit the tool result via previous_interaction_id (observed shape:
    # function_result requires call_id + name + result)
    call = calls[1]
    r2 = Respond(service=UniLM.GEMINIServiceEndpoint, model="gemini-3.8-flash", reasoning=Reasoning(effort="low"), max_output_tokens=2048,
                 previous_response_id=result.response.id,
                 input=[Dict("type" => "function_result", "call_id" => call["call_id"],
                             "name" => call["name"], "result" => Dict("temperature" => "22C"))])
    follow = respond(r2)
    @test follow isa ResponseSuccess
    @test !isempty(output_text(follow))
end

@testset "Interactions — streaming" begin
    payloads = Any[]
    r = Respond(service=UniLM.GEMINIServiceEndpoint, model="gemini-3.8-flash", reasoning=Reasoning(effort="low"), max_output_tokens=2048,
                input="Count from one to five.", stream=true)
    task = respond(r; callback=(c, _) -> push!(payloads, c))
    result = fetch(task)
    @test result isa ResponseSuccess
    # interaction.completed omits steps → final output_text is rebuilt from streamed deltas
    @test !isempty(output_text(result))
    @test !isempty(payloads)
end

@testset "Interactions — tool loop (live, neutral tool_result round-trip)" begin
    r = Respond(service=UniLM.GEMINIServiceEndpoint, model="gemini-3.8-flash", reasoning=Reasoning(effort="low"), max_output_tokens=2048,
                input="What is the weather in Tokyo? Use the get_weather tool, then tell me.",
                tools=[function_tool("get_weather", "Get current weather for a city",
                    parameters=Dict("type" => "object",
                        "properties" => Dict("city" => Dict("type" => "string")), "required" => ["city"]))])
    # the whole point here: tool_loop submits the neutral tool_result item, which the
    # Gemini encoder translates to function_result — a round-trip green mocks could not prove.
    result = tool_loop(r, (name, args) -> "The weather in $(get(args, "city", "?")) is sunny, 22C."; max_turns=4)
    @test result.completed
    @test !isempty(result.tool_calls)
    @test result.tool_calls[1].tool_name == "get_weather"
    @test result.response isa ResponseSuccess
    @test !isempty(output_text(result.response))
end

@testset "Interactions — tool_choice required forces a call (live)" begin
    forced = respond(Respond(service=UniLM.GEMINIServiceEndpoint, model="gemini-3.8-flash", reasoning=Reasoning(effort="low"), max_output_tokens=2048,
        input="Just say hello, nothing else.", tool_choice="required",
        tools=[function_tool("get_weather", "Get weather",
            parameters=Dict("type" => "object", "properties" => Dict("city" => Dict("type" => "string"))))]))
    @test forced isa ResponseSuccess
    @test !isempty(function_calls(forced))
end

@testset "Interactions — estimated_cost > 0 (live usage)" begin
    res = respond(Respond(service=UniLM.GEMINIServiceEndpoint, model="gemini-3.8-flash", reasoning=Reasoning(effort="low"), max_output_tokens=2048,
                          input="Name three primary colors."))
    @test res isa ResponseSuccess
    @test token_usage(res).prompt_tokens > 0
    @test estimated_cost(res) > 0
end

@testset "Interactions — background create → poll → cancel (live)" begin
    started = respond(Respond(service=UniLM.GEMINIServiceEndpoint, model="gemini-3.8-flash", reasoning=Reasoning(effort="low"), max_output_tokens=2048,
                              input="Write one sentence about the ocean.", background=true))
    @test started isa ResponseSuccess
    id = started.response.id
    # A freshly created background id can briefly return 403 permission_denied
    # before the resource becomes visible (observed live 2026-08-20: immediate
    # GET 403, same id retrievable seconds later). Bounded poll retries ONLY
    # that propagation shape; any other failure still fails immediately.
    got = get_response(id; service=UniLM.GEMINIServiceEndpoint)
    deadline = time() + 10
    while got isa ResponseFailure && got.status == 403 && time() < deadline
        sleep(1)
        got = get_response(id; service=UniLM.GEMINIServiceEndpoint)
    end
    @test got isa ResponseSuccess
    @test got.response.id == id
    @test got.response.status in ("in_progress", "completed")
    cancelled = cancel_response(id; service=UniLM.GEMINIServiceEndpoint)
    @test cancelled isa ResponseSuccess || cancelled isa ResponseFailure   # 200 decode, or benign if already completed
end

@testset "Interactions — google_search grounded round-trip (live)" begin
    # A thinking model answers well-known facts from memory and skips a merely
    # permitted search — compel the tool: explicit instruction + a question
    # about post-training state.
    res = respond(Respond(service=UniLM.GEMINIServiceEndpoint, model="gemini-3.8-flash", reasoning=Reasoning(effort="low"), max_output_tokens=2048,
                          input="Use Google Search to find the current stable version of the Julia programming language, then answer with what the search returned.",
                          tools=[gemini_google_search()]))
    @test res isa ResponseSuccess
    @test !isempty(output_text(res))
    @test any(o -> get(o, "type", "") == "google_search_call", res.response.output)
    @test any(o -> get(o, "type", "") == "google_search_result", res.response.output)
end

@testset "Interactions — streaming surfaces a function call (live)" begin
    # Streamed tool use: the terminal event is interaction.completed with
    # status requires_action and NO steps — the function call must be
    # assembled from the incremental step events.
    r = Respond(service=UniLM.GEMINIServiceEndpoint, model="gemini-3.8-flash", reasoning=Reasoning(effort="low"), max_output_tokens=2048,
                input="What is the weather in Oslo right now? Use the tool.",
                stream=true, tool_choice="required",
                tools=[function_tool("get_weather", "Get current weather for a city",
                    parameters=Dict("type" => "object",
                        "properties" => Dict("city" => Dict("type" => "string")),
                        "required" => ["city"]))])
    res = fetch(respond(r))
    @test res isa ResponseSuccess
    ro = res.response
    @test ro.status == "requires_action"
    calls = function_calls(ro)
    @test length(calls) >= 1
    @test calls[1]["name"] == "get_weather"
    @test !isempty(calls[1]["call_id"])
    @test haskey(JSON.parse(calls[1]["arguments"]; dicttype=Dict{String,Any}), "city")
end

# A gemini-3.8-flash interaction at its lowest thinking level.
_ix_live(input; kw...) = respond(Respond(; service=GEMINIServiceEndpoint, model="gemini-3.8-flash",
                                         reasoning=Reasoning(effort="low"), max_output_tokens=256, input, kw...))

@testset "Interactions — messages are steps: a user's image is read, an earlier turn remembered" begin
    # Refuted by: an HTTP 400 for the input (the API takes steps, not turns) or an answer without
    # the circle's colour or the name the earlier turn gave.
    circle = joinpath(@__DIR__, "fixtures", "ollama", "red_circle.png")
    img = _ix_live([InputMessage(role="user", content=[input_text("What colour is the circle? Answer with one word."),
                                                       ImageAttachment(circle)])])
    @test img isa ResponseSuccess && occursin(r"\bred\b"i, output_text(img))
    turns = _ix_live([InputMessage(role="user", content="My name is Ada."),
                      InputMessage(role="assistant", content="Hello, Ada."),
                      InputMessage(role="user", content="What is my name? Answer with the name only.")])
    @test turns isa ResponseSuccess && occursin("ada", lowercase(output_text(turns)))
end

@testset "Interactions — service_tier priority, seed and the safety label are accepted; the tier prices the call" begin
    r = _ix_live("Reply with the single word OK."; max_output_tokens=64, service_tier="priority", seed=7,
                 safety_identifier="unilm-test")
    @test r isa ResponseSuccess
    u = token_usage(r)
    @info "Interactions priority" usage = u cost = estimated_cost(r)
    @test u.service_tier == "priority"
    @test estimated_cost(r) ≈ 1.8 * ((u.prompt_tokens - u.cached_tokens) * 0.75 + u.cached_tokens * 0.075 +
                                     u.completion_tokens * 3.75) / 1e6
end

end  # if GEMINI_API_KEY
