# Native Gemini Interactions API (agentic verb) — deterministic, zero-spend.
# Golden/canned bodies verified against the live Gemini Interactions API on 2026-07-07 (gemini-3.1-flash-lite).

@testset "Interactions routing + capability" begin
    r = Respond(service=GEMINIServiceEndpoint, model="gemini-3.1-flash-lite", input="hi")
    @test UniLM.get_url(r) == "https://generativelanguage.googleapis.com/v1beta/interactions"
    @test UniLM.get_url(GEMINIServiceEndpoint, r) == "https://generativelanguage.googleapis.com/v1beta/interactions"
    @test has_capability(GEMINIServiceEndpoint, :agentic)
    @test has_capability(OPENAIServiceEndpoint, :agentic)
end

@testset "Interactions encode (Respond → snake_case body)" begin
    r = Respond(service=GEMINIServiceEndpoint, model="gemini-2.5-flash",
                input="Say hi", instructions="Be terse",
                tools=[function_tool("get_weather", "Get weather",
                       parameters=Dict("type" => "object",
                                       "properties" => Dict("city" => Dict("type" => "string"))))],
                temperature=0.2, previous_response_id="v1_prev", store=true, stream=true)
    b = JSON.parse(UniLM.encode_agentic(GEMINIServiceEndpoint, r); dicttype=Dict{String,Any})
    @test b["model"] == "gemini-2.5-flash"
    @test b["input"] == "Say hi"
    @test b["system_instruction"] == "Be terse"
    @test b["tools"][1]["type"] == "function"
    @test b["tools"][1]["name"] == "get_weather"
    @test b["tools"][1]["parameters"]["properties"]["city"]["type"] == "string"
    @test b["generation_config"]["temperature"] == 0.2
    @test b["previous_interaction_id"] == "v1_prev"     # neutral previous_response_id → Gemini name
    @test b["store"] == true
    @test b["stream"] == true
    @test !haskey(b, "tool_choice")
end

@testset "Interactions decode (steps[] → neutral output[])" begin
    make(b) = HTTP.Response(200, [], Vector{UInt8}(JSON.json(b)))

    # text (observed shape): thought step + model_output content
    txt = Dict("id" => "v1_t", "object" => "interaction", "model" => "gemini-3.1-flash-lite",
        "status" => "completed",
        "usage" => Dict("total_tokens" => 10, "total_input_tokens" => 8,
                        "total_output_tokens" => 2, "total_thought_tokens" => 0),
        "steps" => [Dict("type" => "thought", "signature" => "sig"),
                    Dict("type" => "model_output",
                         "content" => [Dict("type" => "text", "text" => "Hello.")])])
    ro = UniLM.decode_agentic(GEMINIServiceEndpoint, make(txt))
    @test ro isa ResponseObject
    @test ro.id == "v1_t"
    @test ro.status == "completed"
    @test output_text(ro) == "Hello."
    @test ro.usage["output_tokens"] == 2       # normalized to OpenAI shape at decode
    @test ro.usage["input_tokens"] == 8

    # function call (observed shape): arguments is an OBJECT → normalized to a JSON STRING
    fc = Dict("id" => "v1_f", "object" => "interaction", "model" => "gemini-3.1-flash-lite",
        "status" => "requires_action",
        "steps" => [Dict("id" => "6eG7YnHo", "type" => "function_call", "name" => "get_weather",
                         "arguments" => Dict("city" => "Tokyo"), "signature" => "sig")])
    ro2 = UniLM.decode_agentic(GEMINIServiceEndpoint, make(fc))
    @test ro2.status == "requires_action"
    calls = function_calls(ro2)
    @test length(calls) == 1
    @test calls[1]["name"] == "get_weather"
    @test calls[1]["call_id"] == "6eG7YnHo"
    @test JSON.parse(calls[1]["arguments"])["city"] == "Tokyo"

    # thought steps surface RAW (signature preserved for future replay) —
    # not collapsed into an empty reasoning stub.
    ro3 = UniLM.decode_agentic(GEMINIServiceEndpoint, make(txt))
    thoughts = [o for o in ro3.output if o isa AbstractDict && get(o, "type", "") == "thought"]
    @test length(thoughts) == 1 && thoughts[1]["signature"] == "sig"
    @test !any(o -> o isa AbstractDict && get(o, "type", "") == "reasoning", ro3.output)
    @test output_text(ro3) == "Hello."   # text extraction unaffected
end

@testset "Interactions stream decode (interaction.* SSE)" begin
    state = UniLM.AgenticStreamState()
    UniLM.decode_agentic_stream(GEMINIServiceEndpoint,
        "event: step.delta\ndata: {\"index\":1,\"delta\":{\"text\":\"One, \",\"type\":\"text\"}}\n\n", state)
    st1 = UniLM.decode_agentic_stream(GEMINIServiceEndpoint,
        "event: step.delta\ndata: {\"index\":1,\"delta\":{\"text\":\"two.\",\"type\":\"text\"}}\n\n", state)
    @test st1.terminal == :none

    # real interaction.completed carries NO steps → final output rebuilt from the deltas
    completed = Dict("interaction" => Dict("id" => "v1_s", "status" => "completed",
        "model" => "gemini-3.1-flash-lite", "usage" => Dict("total_tokens" => 17)),
        "event_type" => "interaction.completed")
    st = UniLM.decode_agentic_stream(GEMINIServiceEndpoint,
        "event: interaction.completed\ndata: $(JSON.json(completed))\n\n", state)
    @test st.terminal == :completed
    @test st.data["response"]["id"] == "v1_s"
    @test st.data["response"]["output"][1]["type"] == "message"
    @test st.data["response"]["output"][1]["content"][1]["text"] == "One, two."

    # [DONE] sentinel → terminal :done
    st2 = UniLM.decode_agentic_stream(GEMINIServiceEndpoint,
        "event: done\ndata: [DONE]\n\n", UniLM.AgenticStreamState())
    @test st2.terminal == :done
end

@testset "Interactions encode — tool passthrough, fail-loud, optional gen fields" begin
    # Dict tool passes through _interactions_tool unchanged (pre-shaped escape hatch)
    b = JSON.parse(UniLM.encode_agentic(GEMINIServiceEndpoint,
        Respond(service=GEMINIServiceEndpoint, input="x",
                tools=[Dict("type" => "function", "name" => "raw_fn")])); dicttype=Dict{String,Any})
    @test b["tools"][1]["name"] == "raw_fn"

    # a non-FunctionTool/Dict tool fails LOUD, not silently
    @test_throws ArgumentError UniLM.encode_agentic(GEMINIServiceEndpoint,
        Respond(service=GEMINIServiceEndpoint, input="x", tools=[42]))

    # optional generation_config fields (top_p / max_output_tokens) + background pass through
    b2 = JSON.parse(UniLM.encode_agentic(GEMINIServiceEndpoint,
        Respond(service=GEMINIServiceEndpoint, model="gemini-2.5-flash", input="x",
                top_p=0.9, max_output_tokens=128, background=true)); dicttype=Dict{String,Any})
    @test b2["generation_config"]["top_p"] == 0.9
    @test b2["generation_config"]["max_output_tokens"] == 128
    @test b2["background"] == true
end

@testset "Interactions stream decode — carry-over + malformed handling" begin
    # (a) a chunk with NO newline is buffered whole (carry-over), nothing emitted yet
    state = UniLM.AgenticStreamState()
    st = UniLM.decode_agentic_stream(GEMINIServiceEndpoint, "event: step.delta", state)
    @test st.done == false && st.terminal == :none
    # the buffered partial line reassembles with the next chunk
    UniLM.decode_agentic_stream(GEMINIServiceEndpoint,
        "\ndata: {\"index\":0,\"delta\":{\"text\":\"hi\",\"type\":\"text\"}}\n\n", state)
    @test takestring!(state.textbuff) == "hi"

    # (b) a trailing fragment after the last newline is stashed for the next chunk
    state2 = UniLM.AgenticStreamState()
    UniLM.decode_agentic_stream(GEMINIServiceEndpoint,
        "event: step.delta\ndata: {\"index\":0,\"delta\":{\"text\":\"a\",\"type\":\"text\"}}\n\n" *
        "event: step.delta\ndata: {\"index\":0,\"delta\":{\"text\":\"b\"", state2)
    @test takestring!(state2.textbuff) == "a"                       # first delta consumed
    UniLM.decode_agentic_stream(GEMINIServiceEndpoint, ",\"type\":\"text\"}}\n\n", state2)
    @test takestring!(state2.textbuff) == "b"                       # stashed fragment completed

    # (c) a malformed COMPLETE data line is dropped + counted (never re-queued
    # into the carry) — the shared machine's contract; no crash, stream continues.
    before = UniLM._SSE_DROPPED_LINES[]
    state3 = UniLM.AgenticStreamState()
    st3 = UniLM.decode_agentic_stream(GEMINIServiceEndpoint,
        "event: step.delta\ndata: {not valid json\n\n", state3)
    @test st3.terminal == :none
    @test isempty(take!(state3.carry))
    @test UniLM._SSE_DROPPED_LINES[] == before + 1
    @test state3.sse_dropped == 1        # ... and attributed to THIS stream, not just the process
end

@testset "Interactions encode — tool-result translation" begin
    b = JSON.parse(UniLM.encode_agentic(GEMINIServiceEndpoint,
        Respond(service=GEMINIServiceEndpoint, previous_response_id="v1_prev",
                input=[tool_result("c1", "get_weather", "sunny")])); dicttype=Dict{String,Any})
    @test b["previous_interaction_id"] == "v1_prev"
    @test b["input"][1] == Dict("type" => "function_result", "call_id" => "c1",
                                "name" => "get_weather", "result" => Dict("result" => "sunny"))
    # JSON-object output → object result (not double-wrapped)
    b2 = JSON.parse(UniLM.encode_agentic(GEMINIServiceEndpoint,
        Respond(service=GEMINIServiceEndpoint, input=[tool_result("c2", "f", "{\"temp\":\"22C\"}")])); dicttype=Dict{String,Any})
    @test b2["input"][1]["result"] == Dict("temp" => "22C")
    # a function_call_output missing `name` → fail loud (Gemini requires it)
    @test_throws ArgumentError UniLM.encode_agentic(GEMINIServiceEndpoint,
        Respond(service=GEMINIServiceEndpoint,
                input=[Dict("type" => "function_call_output", "call_id" => "c", "output" => "x")]))
    # a non-tool-result item in a Vector passes through unchanged (not re-translated)
    b4 = JSON.parse(UniLM.encode_agentic(GEMINIServiceEndpoint,
        Respond(service=GEMINIServiceEndpoint,
                input=[Dict("type" => "function_result", "call_id" => "c", "name" => "f", "result" => Dict("x" => 1))])); dicttype=Dict{String,Any})
    @test b4["input"][1]["type"] == "function_result"
    @test b4["input"][1]["result"] == Dict("x" => 1)
    # String input still passes through
    b3 = JSON.parse(UniLM.encode_agentic(GEMINIServiceEndpoint,
        Respond(service=GEMINIServiceEndpoint, input="hi")); dicttype=Dict{String,Any})
    @test b3["input"] == "hi"
end

@testset "Interactions encode — CallableTool tool unwraps to function shape" begin
    ct = UniLM.CallableTool(function_tool("f", "d"), (n, a) -> "x")
    @test UniLM._interactions_tool(ct) == Dict{Symbol,Any}(:type => "function", :name => "f", :description => "d")
end

@testset "Interactions encode — tool_choice mapping" begin
    gc(tc) = JSON.parse(UniLM.encode_agentic(GEMINIServiceEndpoint,
        Respond(service=GEMINIServiceEndpoint, input="x", tool_choice=tc));
        dicttype=Dict{String,Any})["generation_config"]["tool_choice"]["allowed_tools"]
    @test gc("auto")["mode"] == "auto"
    @test gc("none")["mode"] == "none"
    @test gc("required")["mode"] == "any"
    f = gc(UniLM.tool_choice_function("get_weather"))
    @test f["mode"] == "any" && f["tools"] == ["get_weather"]
    # hosted-tool selector is not applicable to Gemini function tools → fail loud
    @test_throws ArgumentError UniLM.encode_agentic(GEMINIServiceEndpoint,
        Respond(service=GEMINIServiceEndpoint, input="x", tool_choice=UniLM.tool_choice_hosted("web_search")))
    # unknown tool_choice string → fail loud
    @test_throws ArgumentError UniLM._interactions_tool_choice("bogus")
end

@testset "Interactions decode — usage normalized for cost accounting" begin
    # Tool-use prompt tokens are input the model read, billed at the input rate as on
    # generateContent; thoughts bill at the output rate.
    raw = Dict("total_input_tokens" => 12, "total_output_tokens" => 69, "total_thought_tokens" => 5,
               "total_tool_use_tokens" => 3, "total_cached_tokens" => 4, "total_tokens" => 89)
    u = UniLM._interaction_usage(raw)
    @test u["input_tokens"] == 15                               # 12 + 3 tool-use prompt (billable input)
    @test u["output_tokens"] == 74                              # 69 + 5 thought (billable output)
    @test u["input_tokens_details"]["cached_tokens"] == 4
    @test u["output_tokens_details"]["reasoning_tokens"] == 5
    @test isnothing(UniLM._interaction_usage(nothing))
    # As a url_context interaction on gemini-3.8-flash reported it on 2026-10-22: the tool-use prompt
    # lies outside total_input_tokens, and total_tokens counts it apart.
    seen = UniLM._interaction_usage(Dict("total_input_tokens" => 30, "total_tool_use_tokens" => 61,
        "total_output_tokens" => 64, "total_thought_tokens" => 0, "total_cached_tokens" => 0, "total_tokens" => 155))
    @test (seen["input_tokens"], seen["output_tokens"]) == (91, 64)
    @test seen["input_tokens"] + seen["output_tokens"] == seen["total_tokens"]
    # end-to-end: a decoded interaction yields correct token_usage + non-zero cost (cached=0 → clean rate identity)
    raw0 = Dict("total_input_tokens" => 12, "total_output_tokens" => 69, "total_thought_tokens" => 5,
                "total_tool_use_tokens" => 3, "total_cached_tokens" => 0, "total_tokens" => 89)
    ro = UniLM._interaction_response_object(Dict("id" => "v1_u", "status" => "completed",
        "model" => "gemini-3.5-flash-lite",
        "steps" => [Dict("type" => "model_output", "content" => [Dict("type" => "text", "text" => "hi")])],
        "usage" => raw0))
    res = ResponseSuccess(response=ro, service=GEMINIServiceEndpoint)
    @test token_usage(res).prompt_tokens == 15
    @test token_usage(res).completion_tokens == 74
    @test token_usage(res).reasoning_tokens == 5
    @test estimated_cost(res) ≈ (15 * 0.30 + 74 * 2.50) / 1_000_000   # gemini-3.5-flash-lite rates
    @test !haskey(u["input_tokens_details"], "audio_tokens")           # no audio: the shape is unchanged
end

@testset "Interactions decode — the audio of the input is its audio tokens by modality, priced as on generateContent" begin
    # gemini-3.1-flash-lite read a 1.5 s WAV clip as input_tokens_by_modality text 16, audio 38 (observed
    # 2026-10-22); the audio of a tool-use prompt is in tool_use_tokens_by_modality.
    usage = Dict("total_tokens" => 55, "total_input_tokens" => 54, "total_output_tokens" => 1, "total_thought_tokens" => 0,
                 "total_cached_tokens" => 0, "total_tool_use_tokens" => 0,
                 "input_tokens_by_modality" => [Dict("modality" => "text", "tokens" => 16), Dict("modality" => "audio", "tokens" => 38)])
    success(u; model="gemini-3.1-flash-lite") = ResponseSuccess(service=GEMINIServiceEndpoint,
        response=UniLM._interaction_response_object(Dict("id" => "v1_a", "status" => "completed", "model" => model,
            "steps" => [Dict("type" => "model_output", "content" => [Dict("type" => "text", "text" => "banana")])], "usage" => u)))
    @test token_usage(success(usage)) == TokenUsage(prompt_tokens=54, completion_tokens=1, total_tokens=55, prompt_audio_tokens=38)
    # 3.1 Flash-Lite: audio $0.50, text $0.25, output $1.50 per 1M tokens (ai.google.dev/gemini-api/docs/pricing).
    @test estimated_cost(success(usage)) ≈ (38 * 0.50 + 16 * 0.25 + 1 * 1.50) / 1_000_000
    tool = merge(usage, Dict("total_tool_use_tokens" => 20, "tool_use_tokens_by_modality" =>
        [Dict("modality" => "audio", "tokens" => 5), Dict("modality" => "text", "tokens" => 15)]))
    @test (token_usage(success(tool)).prompt_tokens, token_usage(success(tool)).prompt_audio_tokens) == (74, 43)
    # 3.6 Flash's one input price names no modality, so it is the audio's too: $0.75 in, $3.75 out per 1M.
    @test (@test_logs estimated_cost(success(usage; model="gemini-3.6-flash"))) ≈ (54 * 0.75 + 1 * 3.75) / 1_000_000
    # A cache hit beside the audio: 3.6 Flash's cache price names no modality either, so the hit bills at
    # its cached rate ($0.075 per 1M); 3.1 Flash-Lite lists cached audio apart and the usage does not say
    # how many cached tokens are audio, so that call is unpriced.
    cached = merge(usage, Dict("total_cached_tokens" => 10))
    @test (@test_logs estimated_cost(success(cached; model="gemini-3.6-flash"))) ≈ (44 * 0.75 + 10 * 0.075 + 1 * 3.75) / 1_000_000
    @test (@test_logs (:warn, r"no audio-token rates") estimated_cost(success(cached))) === 0.0
end

@testset "Agentic lifecycle URL is service-dispatched (_agentic_url)" begin
    @test UniLM._agentic_url(GEMINIServiceEndpoint) == "https://generativelanguage.googleapis.com/v1beta/interactions"
    @test UniLM._agentic_url(OPENAIServiceEndpoint) == UniLM._api_base_url(OPENAIServiceEndpoint) * UniLM.RESPONSES_PATH
    # the create URL still resolves through the same source of truth (no divergence)
    r = Respond(service=GEMINIServiceEndpoint, input="x")
    @test UniLM.get_url(GEMINIServiceEndpoint, r) == UniLM._agentic_url(GEMINIServiceEndpoint)
end

@testset "Gemini hosted-tool constructors + encode passthrough" begin
    @test gemini_google_search()  == Dict("type" => "google_search")
    @test gemini_code_execution() == Dict("type" => "code_execution")
    @test gemini_url_context()    == Dict("type" => "url_context")
    b = JSON.parse(UniLM.encode_agentic(GEMINIServiceEndpoint,
        Respond(service=GEMINIServiceEndpoint, input="x", tools=[gemini_google_search()])); dicttype=Dict{String,Any})
    @test b["tools"][1] == Dict("type" => "google_search")
end

@testset "Interactions decode — hosted-tool steps surfaced, not dropped" begin
    data = Dict("id" => "v1_h", "status" => "completed", "model" => "gemini-3.1-flash-lite", "steps" => [
        Dict("type" => "google_search_call", "id" => "s1", "arguments" => Dict("queries" => ["x"])),
        Dict("type" => "google_search_result", "id" => "s1"),
        Dict("type" => "model_output", "content" => [Dict("type" => "text", "text" => "Answer.")])])
    ro = UniLM._interaction_response_object(data)
    types = [get(o, "type", "") for o in ro.output]
    @test "google_search_call" in types                 # surfaced
    @test "google_search_result" in types
    res = ResponseSuccess(response=ro)
    @test output_text(res) == "Answer."                 # message still decoded
    @test isempty(function_calls(res))                  # hosted step ≠ function call
end

@testset "Interactions stream — function-call assembly (requires_action)" begin
    st = UniLM.AgenticStreamState()
    UniLM.decode_agentic_stream(GEMINIServiceEndpoint,
        "event: step.start\ndata: {\"event_type\":\"step.start\",\"index\":0,\"step\":{\"type\":\"function_call\",\"id\":\"fc_9\",\"name\":\"lookup\",\"arguments\":{}}}\n\n", st)
    UniLM.decode_agentic_stream(GEMINIServiceEndpoint,
        "event: step.delta\ndata: {\"event_type\":\"step.delta\",\"index\":0,\"delta\":{\"type\":\"arguments_delta\",\"arguments\":\"{\\\"q\\\":\"}}\n\n", st)
    UniLM.decode_agentic_stream(GEMINIServiceEndpoint,
        "event: step.delta\ndata: {\"event_type\":\"step.delta\",\"index\":0,\"delta\":{\"type\":\"arguments_delta\",\"arguments\":\"\\\"julia\\\"}\"}}\n\n", st)
    UniLM.decode_agentic_stream(GEMINIServiceEndpoint,
        "event: step.stop\ndata: {\"event_type\":\"step.stop\",\"index\":0}\n\n", st)
    r = UniLM.decode_agentic_stream(GEMINIServiceEndpoint,
        "event: interaction.completed\ndata: {\"event_type\":\"interaction.completed\",\"interaction\":{\"id\":\"i_9\",\"status\":\"requires_action\",\"model\":\"m\",\"usage\":{\"total_input_tokens\":1,\"total_output_tokens\":1,\"total_tokens\":2}}}\n\n", st)
    @test r.done == true && r.terminal == :completed
    rd = r.data["response"]
    @test rd["status"] == "requires_action"
    calls = [o for o in rd["output"] if get(o, "type", "") == "function_call"]
    @test length(calls) == 1
    @test calls[1]["call_id"] == "fc_9" && calls[1]["name"] == "lookup"
    @test JSON.parse(calls[1]["arguments"]; dicttype=Dict{String,Any}) == Dict{String,Any}("q" => "julia")
    # usage normalized to OpenAI keys as everywhere else
    @test rd["usage"]["input_tokens"] == 1
end

@testset "Interactions stream — zero-argument function call assembles empty args" begin
    # A no-parameter tool: step.start carries an empty `arguments` object and NO
    # arguments_delta ever arrives. The assembled function_call must still surface
    # with arguments == "{}" (parsing to an empty Dict) — the start snapshot's
    # empty object is the fallback — not be dropped or throw on an absent delta.
    st = UniLM.AgenticStreamState()
    UniLM.decode_agentic_stream(GEMINIServiceEndpoint,
        "event: step.start\ndata: {\"event_type\":\"step.start\",\"index\":0,\"step\":{\"type\":\"function_call\",\"id\":\"fc_0\",\"name\":\"ping\",\"arguments\":{}}}\n\n", st)
    UniLM.decode_agentic_stream(GEMINIServiceEndpoint,
        "event: step.stop\ndata: {\"event_type\":\"step.stop\",\"index\":0}\n\n", st)
    r = UniLM.decode_agentic_stream(GEMINIServiceEndpoint,
        "event: interaction.completed\ndata: {\"event_type\":\"interaction.completed\",\"interaction\":{\"id\":\"i_0\",\"status\":\"requires_action\",\"model\":\"m\"}}\n\n", st)
    @test r.done == true && r.terminal == :completed
    calls = [o for o in r.data["response"]["output"] if get(o, "type", "") == "function_call"]
    @test length(calls) == 1
    @test calls[1]["call_id"] == "fc_0" && calls[1]["name"] == "ping"
    @test calls[1]["arguments"] == "{}"
    @test JSON.parse(calls[1]["arguments"]; dicttype=Dict{String,Any}) == Dict{String,Any}()
end

@testset "Interactions stream — text + thought signature assembly" begin
    st = UniLM.AgenticStreamState()
    UniLM.decode_agentic_stream(GEMINIServiceEndpoint,
        "event: step.start\ndata: {\"event_type\":\"step.start\",\"index\":0,\"step\":{\"type\":\"thought\"}}\n\n", st)
    UniLM.decode_agentic_stream(GEMINIServiceEndpoint,
        "event: step.delta\ndata: {\"event_type\":\"step.delta\",\"index\":0,\"delta\":{\"type\":\"thought_signature\",\"signature\":\"enc==\"}}\n\n", st)
    UniLM.decode_agentic_stream(GEMINIServiceEndpoint,
        "event: step.delta\ndata: {\"event_type\":\"step.delta\",\"index\":1,\"delta\":{\"type\":\"text\",\"text\":\"Hi \"}}\n\n", st)
    UniLM.decode_agentic_stream(GEMINIServiceEndpoint,
        "event: step.delta\ndata: {\"event_type\":\"step.delta\",\"index\":1,\"delta\":{\"type\":\"text\",\"text\":\"there\"}}\n\n", st)
    r = UniLM.decode_agentic_stream(GEMINIServiceEndpoint,
        "event: interaction.completed\ndata: {\"event_type\":\"interaction.completed\",\"interaction\":{\"id\":\"i_t\",\"status\":\"completed\",\"model\":\"m\"}}\n\n", st)
    rd = r.data["response"]
    thoughts = [o for o in rd["output"] if get(o, "type", "") == "thought"]
    @test length(thoughts) == 1 && thoughts[1]["signature"] == "enc=="
    @test output_text(ResponseObject(id=rd["id"], status=rd["status"], model=rd["model"],
                                     output=rd["output"], usage=rd["usage"],
                                     error=nothing, metadata=nothing, raw=rd)) == "Hi there"
end

@testset "Interactions — the same interaction decoded both ways yields the same output" begin
    thought = Dict("type" => "thought", "signature" => "sig==",
                   "summary" => [Dict("type" => "text", "text" => "Plan the answer.")])
    answer = Dict("type" => "model_output", "content" => [Dict("type" => "text", "text" => "Part one."),
                                                          Dict("type" => "text", "text" => "Part two.")])
    call = Dict("type" => "function_call", "id" => "fc_1", "name" => "lookup", "arguments" => Dict("q" => "julia"))
    interaction = Dict("id" => "v1_p", "status" => "requires_action", "model" => "m")
    ro = UniLM.decode_agentic(GEMINIServiceEndpoint, HTTP.Response(200, [], Vector{UInt8}(
        JSON.json(merge(interaction, Dict("steps" => [thought, answer, call]))))))

    # Streamed: the thought's summary and signature arrive as deltas, a text delta
    # completes the answer's second part, the call's arguments arrive as partial JSON,
    # and interaction.completed carries no steps.
    ev(name, data) = "event: $name\ndata: $(JSON.json(data))\n\n"
    delta(index, d) = ev("step.delta", Dict("index" => index, "delta" => d))
    sse = ev("step.start", Dict("index" => 0, "step" => Dict("type" => "thought"))) *
          delta(0, Dict("type" => "thought_summary", "content" => Dict("type" => "text", "text" => "Plan the answer."))) *
          delta(0, Dict("type" => "thought_signature", "signature" => "sig==")) *
          ev("step.stop", Dict("index" => 0)) *
          ev("step.start", Dict("index" => 1, "step" => Dict("type" => "model_output", "content" => [
              Dict("type" => "text", "text" => "Part one."), Dict("type" => "text", "text" => "Part ")]))) *
          delta(1, Dict("type" => "text", "text" => "two.")) *
          ev("step.stop", Dict("index" => 1)) *
          ev("step.start", Dict("index" => 2, "step" => Dict("type" => "function_call", "id" => "fc_1",
                                                             "name" => "lookup", "arguments" => Dict()))) *
          delta(2, Dict("type" => "arguments_delta", "arguments" => "{\"q\":")) *
          delta(2, Dict("type" => "arguments_delta", "arguments" => "\"julia\"}")) *
          ev("step.stop", Dict("index" => 2)) *
          ev("interaction.completed", Dict("interaction" => interaction))
    rd = UniLM.decode_agentic_stream(GEMINIServiceEndpoint, sse, UniLM.AgenticStreamState()).data["response"]
    streamed = ResponseObject(id=rd["id"], status=rd["status"], model=rd["model"], output=rd["output"], raw=rd)
    @test streamed.output == ro.output
    @test output_text(streamed) == output_text(ro) == "Part one.Part two."   # one answer, joined as written
    @test function_calls(streamed) == function_calls(ro)
end

@testset "Interactions stream — deltas accumulate without re-copying the step" begin
    # 800 deltas of 1 kB: re-concatenating a String per delta copies ~320 MB, while
    # appending to a buffer read once stays near the ~6 MB that decoding the events
    # allocates on its own.
    piece = repeat("x", 1_000)
    function stream(kind, step, field)
        st = UniLM.AgenticStreamState()
        UniLM.decode_agentic_stream(GEMINIServiceEndpoint,
            "event: step.start\ndata: $(JSON.json(Dict("index" => 0, "step" => step)))\n\n", st)
        chunk = "event: step.delta\ndata: $(JSON.json(Dict("index" => 0, "delta" => Dict("type" => kind, field => piece))))\n\n"
        for _ in 1:800
            UniLM.decode_agentic_stream(GEMINIServiceEndpoint, chunk, st)
        end
        st
    end
    for (kind, step, field) in (("arguments_delta", Dict("type" => "function_call", "id" => "c", "name" => "f"), "arguments"),
                                ("thought_signature", Dict("type" => "thought"), "signature"))
        stream(kind, step, field)                                  # compile first
        @test (@allocated stream(kind, step, field)) < 30_000_000
        out = only(UniLM._assembled_interaction_output(stream(kind, step, field)))
        @test out[field] == repeat(piece, 800)                     # every byte assembled
    end
end

using Sockets

# Localhost SSE endpoint riding the Interactions wire seam, so the real streaming
# driver runs offline. The handler writes the whole event stream in ONE flush, so
# the driver sees the text deltas and the terminal event in a single read — the
# shape a short generation actually produces.
const _IX_MOCK_URL = Ref("http://127.0.0.1:0")
struct _IxStreamMock <: UniLM.ServiceEndpoint end
UniLM._agentic_url(::Type{_IxStreamMock}) = _IX_MOCK_URL[]
UniLM.encode_agentic(::Type{_IxStreamMock}, r::Respond) = UniLM.encode_agentic(GEMINIServiceEndpoint, r)
UniLM.decode_agentic_stream(::Type{_IxStreamMock}, chunk::String, st::UniLM.AgenticStreamState) =
    UniLM.decode_agentic_stream(GEMINIServiceEndpoint, chunk, st)
UniLM.auth_header(::Type{_IxStreamMock}) = ["Content-Type" => "application/json"]
UniLM.default_model(::Type{_IxStreamMock}) = "mock-model"

function _ix_sse_server(body::String)
    server = nothing; port = 0
    for attempt in 1:5
        tcp = Sockets.listen(Sockets.localhost, 0)
        port = Int(Sockets.getsockname(tcp)[2]); close(tcp)
        try
            server = HTTP.listen!("127.0.0.1", port; verbose=false) do http::HTTP.Stream
                read(http)
                HTTP.setstatus(http, 200)
                HTTP.setheader(http, "Content-Type" => "text/event-stream")
                HTTP.startwrite(http)
                write(http, body)
            end
            break
        catch; attempt == 5 && rethrow(); end
    end
    server, "http://127.0.0.1:$port"
end

@testset "Interactions stream — a failed interaction is a typed failure, not a success" begin
    interaction = Dict("id" => "v1_x", "object" => "interaction", "status" => "failed",
        "model" => "gemini-3.1-flash-lite",
        "error" => Dict("code" => "safety_block", "message" => "blocked"),
        "metadata" => Dict("trace" => "t7"),
        "usage" => Dict("total_input_tokens" => 3, "total_output_tokens" => 0, "total_tokens" => 3))
    ro = UniLM.decode_agentic(GEMINIServiceEndpoint,
        HTTP.Response(200, [], Vector{UInt8}(JSON.json(interaction))))
    @test ro.status == "failed" && ro.error["code"] == "safety_block" && ro.metadata["trace"] == "t7"

    st = UniLM.decode_agentic_stream(GEMINIServiceEndpoint,
        "event: interaction.completed\ndata: " *
        JSON.json(Dict("event_type" => "interaction.completed", "interaction" => interaction)) * "\n\n",
        UniLM.AgenticStreamState())
    # the driver's typed-failure limb, exactly as OpenAI's response.failed takes it
    @test st.terminal == :failed
    rd = st.data["response"]
    @test rd["status"] == "failed"
    @test rd["error"] == ro.error                          # error surface kept
    @test rd["metadata"] == ro.metadata                    # metadata surface kept
    @test issubset(keys(ro.raw), keys(rd))                 # raw capture not reduced vs non-stream
    res = UniLM._agentic_terminal_result(st.data, 200, nothing)
    @test res isa ResponseFailure && res.status == 200
    @test JSON.parse(res.response; dicttype=Dict{String,Any})["error"]["code"] == "safety_block"
end

@testset "Interactions decode — null step content is empty, not a crash" begin
    # `content: null` is spec-legal: a model_output step that produced no parts. It adds no
    # text to the answer the next step continues.
    body = Dict("id" => "v1_n", "status" => "completed", "model" => "gemini-3.1-flash-lite",
        "steps" => [Dict("type" => "model_output", "content" => nothing),
                    Dict("type" => "model_output",
                         "content" => [Dict("type" => "text", "text" => "ok")])])
    ro = UniLM.decode_agentic(GEMINIServiceEndpoint,
        HTTP.Response(200, [], Vector{UInt8}(JSON.json(body))))
    @test output_text(ro) == "ok"
    @test ro.output == [Dict("type" => "message", "role" => "assistant",
                             "content" => [Dict("type" => "output_text", "text" => "ok")])]
    # Alone, such a step is an answer with no text.
    alone = UniLM._interaction_response_object(Dict("id" => "v1_n", "status" => "completed",
        "steps" => [Dict("type" => "model_output", "content" => nothing)]))
    @test alone.output[1]["content"] == Any[] && output_text(alone) == ""

    # streamed: a throw here is swallowed by the SSE drop policy (counted as a
    # dropped line), turning a clean interaction into a failure or an idle stall.
    before = UniLM._SSE_DROPPED_LINES[]
    st = UniLM.decode_agentic_stream(GEMINIServiceEndpoint,
        "event: interaction.completed\ndata: " *
        JSON.json(Dict("event_type" => "interaction.completed", "interaction" => body)) * "\n\n",
        UniLM.AgenticStreamState())
    @test UniLM._SSE_DROPPED_LINES[] == before
    @test st.terminal == :completed
    @test st.data["response"]["output"] == ro.output
end

@testset "Interactions decode — a body without an id or a status is not an interaction" begin
    # Both are required on every Interaction; decoding `{}` as one reports an empty success.
    make(b) = HTTP.Response(200, [], Vector{UInt8}(JSON.json(b)))
    for body in (Dict(), Dict("status" => "completed"), Dict("id" => "v1_x"),
                 Dict("id" => "", "status" => "completed"), Dict("id" => "v1_x", "status" => nothing),
                 Dict("id" => 7, "status" => "completed"))
        @test_throws ErrorException UniLM.decode_agentic(GEMINIServiceEndpoint, make(body))
    end
    err = try UniLM.decode_agentic(GEMINIServiceEndpoint, make(Dict("status" => "completed"))); nothing catch e; e end
    @test err isa ErrorException && occursin("\"id\"", err.msg)
end

@testset "Interactions stream — a terminal without an id or a status is a call error, as unstreamed" begin
    # The streamed terminal took what it got: `{"interaction": {}}` was a success whose id is "".
    for interaction in (Dict(), Dict("status" => "completed"), Dict("id" => "v1_x"),
                        Dict("id" => "", "status" => "completed"), Dict("id" => "v1_x", "status" => nothing))
        state = UniLM.AgenticStreamState()
        st = UniLM.decode_agentic_stream(GEMINIServiceEndpoint, "event: interaction.completed\ndata: " *
            JSON.json(Dict("event_type" => "interaction.completed", "interaction" => interaction)) * "\n\n", state)
        @test st.done && st.terminal === :error
        @test occursin("carries no", st.data["message"])
        @test state.sse_dropped == 0                              # an outcome, not a dropped line
        res = UniLM._agentic_terminal_result(st.data, 200, nothing)
        @test res isa ResponseCallError && occursin("carries no", res.error)
    end
    # ...and through the stream driver.
    server, url = _ix_sse_server("event: interaction.completed\ndata: " *
        JSON.json(Dict("event_type" => "interaction.completed", "interaction" => Dict())) * "\n\n" *
        "event: done\ndata: [DONE]\n\n")
    _IX_MOCK_URL[] = url
    try
        t = respond(Respond(service=_IxStreamMock, input="hi", stream=true);
                    config=RequestConfig(request_timeout=5.0, total_deadline=20.0,
                                         stream_idle_timeout=5.0, max_attempts=1))
        @test timedwait(() -> istaskdone(t), 25.0) == :ok
        res = fetch(t)
        @test res isa ResponseCallError && occursin("carries no \"id\"", res.error)
    finally
        close(server)
    end
end

@testset "Interactions stream decode — no method boxes a variable" begin
    # A local that a closure captures and that is assigned more than once becomes a
    # Core.Box: every access is a dynamic lookup, paid on each SSE event of a stream.
    file = joinpath(pkgdir(UniLM), "src", "interactions.jl")
    fns = [getfield(UniLM, n) for n in names(UniLM; all=true) if isdefined(UniLM, n) && getfield(UniLM, n) isa Function]
    ms = unique([m for f in fns for m in methods(f) if String(m.file) == file])
    @test !isempty(ms)
    @test [m for m in ms if occursin("Core.Box", string(Base.uncompressed_ir(m)))] == Method[]
end

@testset "Interactions stream — a one-read stream still delivers its text deltas" begin
    sse = "event: step.delta\ndata: {\"index\":0,\"delta\":{\"type\":\"text\",\"text\":\"Hello \"}}\n\n" *
          "event: step.delta\ndata: {\"index\":0,\"delta\":{\"type\":\"text\",\"text\":\"world\"}}\n\n" *
          "event: interaction.completed\ndata: {\"event_type\":\"interaction.completed\",\"interaction\":{\"id\":\"i_1r\",\"status\":\"completed\",\"model\":\"m\",\"usage\":{\"total_tokens\":3}}}\n\n" *
          "event: done\ndata: [DONE]\n\n"
    # (a) decode layer: the terminal rebuild must leave the driver-owned text buffer
    # intact — the driver emits deltas by diffing it against what it already sent.
    st = UniLM.AgenticStreamState()
    r = UniLM.decode_agentic_stream(GEMINIServiceEndpoint, sse, st)
    @test r.terminal == :completed
    @test r.data["response"]["output"][1]["content"][1]["text"] == "Hello world"
    @test takestring!(st.textbuff) == "Hello world"

    # (b) end-to-end through the real driver: every delta reaches the callback.
    server, url = _ix_sse_server(sse)
    _IX_MOCK_URL[] = url
    deltas = String[]
    try
        t = respond(Respond(service=_IxStreamMock, input="hi", stream=true);
                    config=RequestConfig(request_timeout=5.0, total_deadline=20.0,
                                         stream_idle_timeout=5.0, max_attempts=1),
                    callback=(c, _close) -> c isa String && push!(deltas, c))
        @test timedwait(() -> istaskdone(t), 25.0) == :ok
        res = fetch(t)
        @test res isa ResponseSuccess
        @test join(deltas) == "Hello world"      # nothing swallowed by the terminal rebuild
        @test output_text(res) == "Hello world"  # ... and the final output is still complete
    finally
        close(server)
    end
end

@testset "Interactions stream — a non-failed terminal (incomplete) stays a success" begin
    # Only "failed" flips the typed-failure limb, on the streamed path exactly as on
    # the non-streamed one. An interaction that stopped early still delivers a real
    # response object with usable partial output; the truncation is reported in
    # `status`/`incomplete_details`, not by discarding the result.
    interaction = Dict("id" => "v1_inc", "object" => "interaction", "status" => "incomplete",
        "model" => "gemini-3.1-flash-lite",
        "incomplete_details" => Dict("reason" => "max_output_tokens"),
        "usage" => Dict("total_input_tokens" => 2, "total_output_tokens" => 1, "total_tokens" => 3))
    completed = "event: interaction.completed\ndata: " *
        JSON.json(Dict("event_type" => "interaction.completed", "interaction" => interaction)) * "\n\n"

    # (a) decode layer: the terminal maps to :completed, never :failed.
    st = UniLM.decode_agentic_stream(GEMINIServiceEndpoint, completed, UniLM.AgenticStreamState())
    @test st.terminal == :completed
    @test st.data["response"]["status"] == "incomplete"
    @test st.data["response"]["incomplete_details"]["reason"] == "max_output_tokens"

    # (b) end-to-end through the real driver: a typed SUCCESS carrying the details.
    sse = "event: step.delta\ndata: {\"index\":0,\"delta\":{\"type\":\"text\",\"text\":\"partial\"}}\n\n" *
          completed * "event: done\ndata: [DONE]\n\n"
    server, url = _ix_sse_server(sse)
    _IX_MOCK_URL[] = url
    try
        t = respond(Respond(service=_IxStreamMock, input="hi", stream=true);
                    config=RequestConfig(request_timeout=5.0, total_deadline=20.0,
                                         stream_idle_timeout=5.0, max_attempts=1))
        @test timedwait(() -> istaskdone(t), 25.0) == :ok
        res = fetch(t)
        @test res isa ResponseSuccess
        @test res.response.status == "incomplete"
        @test incomplete_details(res)["reason"] == "max_output_tokens"
        @test output_text(res) == "partial"           # the partial answer survived
        @test token_usage(res).total_tokens == 3
    finally
        close(server)
    end
end

@testset "Interactions stream — a dropped payload is counted on the result" begin
    # An undecodable data line is dropped so it cannot poison the carry; the count
    # is what tells the caller the turn was assembled from an incomplete wire.
    sse = "event: step.delta\ndata: {\"index\":0,\"delta\":{\"type\":\"text\",\"text\":\"ok\"}}\n\n" *
          "event: step.delta\ndata: {not valid json\n\n" *
          "event: interaction.completed\ndata: " *
          JSON.json(Dict("event_type" => "interaction.completed",
                         "interaction" => Dict("id" => "v1_d", "status" => "completed", "model" => "m",
                                               "usage" => Dict("total_tokens" => 2)))) * "\n\n" *
          "event: done\ndata: [DONE]\n\n"
    server, url = _ix_sse_server(sse)
    _IX_MOCK_URL[] = url
    try
        t = respond(Respond(service=_IxStreamMock, input="hi", stream=true);
                    config=RequestConfig(request_timeout=5.0, total_deadline=20.0,
                                         stream_idle_timeout=5.0, max_attempts=1))
        @test timedwait(() -> istaskdone(t), 25.0) == :ok
        res = fetch(t)
        @test res isa ResponseSuccess
        @test res.sse_dropped == 1
        @test output_text(res) == "ok"
    finally
        close(server)
    end
end

@testset "Interactions encode — an unsupported Respond field fails loud" begin
    # Every field the Interactions body maps except `text` (its response_format has
    # its own golden), set at once: the encoding is byte-for-byte the one this wire
    # produced before the guard existed.
    r = Respond(service=GEMINIServiceEndpoint, model="gemini-2.5-flash", input="Say hi",
                instructions="Be terse", tools=[function_tool("get_weather", "Get weather")],
                tool_choice="auto", temperature=0.2, max_output_tokens=64,
                previous_response_id="v1_prev", store=true, background=false, stream=true)
    @test UniLM.encode_agentic(GEMINIServiceEndpoint, r) ==
        """{"background":false,"generation_config":{"max_output_tokens":64,"temperature":0.2,"tool_choice":{"allowed_tools":{"mode":"auto"}}},"input":"Say hi","model":"gemini-2.5-flash","previous_interaction_id":"v1_prev","store":true,"stream":true,"system_instruction":"Be terse","tools":[{"description":"Get weather","name":"get_weather","type":"function"}]}"""

    # A field this wire has no mapping for must not vanish from the request.
    cases = (:metadata => Dict("k" => "v"), :truncation => "auto",
             :parallel_tool_calls => true, :user => "u1", :include => ["a"],
             :max_tool_calls => 2, :top_logprobs => 3,
             :prompt_cache_key => "k", :prompt_cache_retention => "24h",
             :conversation => "conv_1", :context_management => [Dict("type" => "x")],
             :stream_options => Dict("include_usage" => true),
             :prompt_cache_options => PromptCacheOptions(mode="explicit"),
             :moderation => ModerationConfig(model="omni-moderation-latest"))
    @test Set(first.(cases)) == Set(UniLM._INTERACTIONS_UNMAPPED_FIELDS)   # every unmapped field covered
    for (field, value) in cases
        rr = Respond(; service=GEMINIServiceEndpoint, input="x", (field => value,)...)
        err = try UniLM.encode_agentic(GEMINIServiceEndpoint, rr); nothing catch e; e end
        @test err isa ArgumentError
        @test err isa ArgumentError && occursin(String(field), err.msg)
    end
end

@testset "Interactions encode — OpenAI-only FunctionTool options fail loud" begin
    params = Dict("type" => "object", "properties" => Dict())
    encode(tool) = UniLM.encode_agentic(GEMINIServiceEndpoint,
        Respond(service=GEMINIServiceEndpoint, input="x", tools=[tool]))
    for (field, value) in (:async => true, :allowed_callers => ["direct"], :defer_loading => true,
                           :output_schema => Dict{String,Any}("type" => "object"))
        err = try encode(FunctionTool(; name="noop", parameters=params, (field => value,)...)); nothing catch e; e end
        @test err isa ArgumentError && err.msg == "Gemini Interactions function tools do not support $field"
    end
    # false is still a value the request would lose; every set field is named.
    err = try encode(FunctionTool(name="noop", async=false, defer_loading=false)); nothing catch e; e end
    @test err isa ArgumentError && err.msg == "Gemini Interactions function tools do not support async, defer_loading"
    # strict is not sent.
    @test JSON.parse(encode(FunctionTool(name="noop", description="d", parameters=params, strict=true)))["tools"] ==
          [Dict("type" => "function", "name" => "noop", "description" => "d", "parameters" => params)]
end

@testset "Interactions encode — text format → top-level response_format (wire golden)" begin
    schema = Dict("type" => "object",
                  "properties" => Dict("city" => Dict("type" => "string"), "country" => Dict("type" => "string")),
                  "required" => ["city", "country"], "additionalProperties" => false)
    r = Respond(service=GEMINIServiceEndpoint, model="gemini-3.8-flash", input="Give the capital of Norway as JSON.",
                text=json_schema_format("capital", "A capital city", schema; strict=true), max_output_tokens=256)
    @test JSON.parse(UniLM.encode_agentic(GEMINIServiceEndpoint, r)) == Dict(
        "model" => "gemini-3.8-flash", "input" => "Give the capital of Norway as JSON.",
        "generation_config" => Dict("max_output_tokens" => 256),
        "response_format" => Dict("type" => "text", "mime_type" => "application/json", "schema" => schema))

    fmt(text) = get(JSON.parse(UniLM.encode_agentic(GEMINIServiceEndpoint,
        Respond(service=GEMINIServiceEndpoint, input="x", text=text))), "response_format", nothing)
    @test fmt(json_schema_format(Dict("name" => "capital", "schema" => schema))) ==
          Dict("type" => "text", "mime_type" => "application/json", "schema" => schema)
    @test fmt(json_object_format()) == Dict("type" => "text", "mime_type" => "application/json")
    @test isnothing(fmt(text_format()))                          # plain text: nothing to send
    @test :text ∉ UniLM._INTERACTIONS_UNMAPPED_FIELDS
    # shapes with no Interactions counterpart fail loud instead of vanishing
    for text in (text_format(type="xml"), text_format(type="json_schema"),
                 text_format(type="json_object", schema=schema), text_format(verbosity="low"))
        @test_throws ArgumentError UniLM.encode_agentic(GEMINIServiceEndpoint,
            Respond(service=GEMINIServiceEndpoint, input="x", text=text))
    end
end

@testset "Interactions stream — byte re-split invariance of assembly" begin
    golden = "event: step.start\ndata: {\"event_type\":\"step.start\",\"index\":0,\"step\":{\"type\":\"function_call\",\"id\":\"fc_s\",\"name\":\"f\",\"arguments\":{}}}\n\n" *
             "event: step.delta\ndata: {\"event_type\":\"step.delta\",\"index\":0,\"delta\":{\"type\":\"arguments_delta\",\"arguments\":\"{\\\"a\\\":1}\"}}\n\n" *
             "event: step.stop\ndata: {\"event_type\":\"step.stop\",\"index\":0}\n\n" *
             "event: interaction.completed\ndata: {\"event_type\":\"interaction.completed\",\"interaction\":{\"id\":\"i_s\",\"status\":\"requires_action\",\"model\":\"m\"}}\n\n"
    bytes = Vector{UInt8}(golden)
    ok = true
    for k in 1:length(bytes)-1
        st = UniLM.AgenticStreamState()
        r1 = UniLM.decode_agentic_stream(GEMINIServiceEndpoint, String(bytes[1:k]), st)
        r = r1.done ? r1 : UniLM.decode_agentic_stream(GEMINIServiceEndpoint, String(bytes[k+1:end]), st)
        calls = r.done && r.data isa AbstractDict ?
            [o for o in get(r.data["response"], "output", Any[]) if get(o, "type", "") == "function_call"] : Any[]
        ok &= length(calls) == 1 && calls[1]["call_id"] == "fc_s" &&
              JSON.parse(calls[1]["arguments"]; dicttype=Dict{String,Any}) == Dict{String,Any}("a" => 1)
    end
    @test ok
end

# Non-streaming Interactions mock: the SAME decode_agentic seam the native endpoint
# uses, pointed at a local fixture. Kept off the real endpoint so no other test's URL
# dispatch is disturbed.
const _IX_NONSTREAM_URL = Ref("http://127.0.0.1:0")
struct _IxNonStreamMock <: UniLM.ServiceEndpoint end
UniLM._agentic_url(::Type{_IxNonStreamMock}) = _IX_NONSTREAM_URL[]
UniLM.encode_agentic(::Type{_IxNonStreamMock}, r::Respond) = UniLM.encode_agentic(GEMINIServiceEndpoint, r)
UniLM.decode_agentic(::Type{_IxNonStreamMock}, resp::HTTP.Response) =
    UniLM.decode_agentic(GEMINIServiceEndpoint, resp)
UniLM.auth_header(::Type{_IxNonStreamMock}) = ["Content-Type" => "application/json"]
UniLM.default_model(::Type{_IxNonStreamMock}) = "mock-model"

@testset "a non-streamed failed interaction is a failure, not a success" begin
    # Same contract as the OpenAI wire: `respond` decodes both wires through the one
    # decode_agentic seam, so a failed interaction must produce the ResponseFailure its
    # streamed twin produces rather than a ResponseSuccess carrying status "failed".
    bodies = Dict(
        "failed" => JSON.json(Dict(
            "id" => "int_f1", "status" => "failed", "model" => "gemini-3.1-flash-lite",
            "steps" => [], "error" => Dict("code" => "safety_block", "message" => "blocked"),
            "metadata" => Dict("trace" => "t-9"))),
        "completed" => JSON.json(Dict(
            "id" => "int_ok", "status" => "completed", "model" => "gemini-3.1-flash-lite",
            "steps" => [])),
        "requires_action" => JSON.json(Dict(
            "id" => "int_ra", "status" => "requires_action", "model" => "gemini-3.1-flash-lite",
            "steps" => [])),
        "not_an_interaction" => "{}")
    which = Ref("failed")
    tcp = Sockets.listen(Sockets.localhost, 0)
    port = Int(Sockets.getsockname(tcp)[2]); close(tcp)
    srv = HTTP.serve!("127.0.0.1", port; verbose=false) do req
        HTTP.Response(200, ["Content-Type" => "application/json"], bodies[which[]])
    end
    _IX_NONSTREAM_URL[] = "http://127.0.0.1:$port/v1beta/interactions"
    try
        r = respond(Respond(service=_IxNonStreamMock, input="hi"))
        @test r isa ResponseFailure
        @test !issuccess(r)
        raw = JSON.parse(r.response; dicttype=Dict{String,Any})
        @test raw["status"] == "failed"
        @test raw["error"]["code"] == "safety_block"   # error surface preserved
        @test raw["metadata"]["trace"] == "t-9"        # metadata surface preserved

        # Only "failed" flips: a completed interaction and a requires_action turn
        # (the tool-action flow) stay successes.
        for st in ("completed", "requires_action")
            which[] = st
            r2 = respond(Respond(service=_IxNonStreamMock, input="hi"))
            @test r2 isa ResponseSuccess
            @test r2.response.status == st
        end

        # A 200 body with no id and no status is a call error, never a success.
        which[] = "not_an_interaction"
        r3 = respond(Respond(service=_IxNonStreamMock, input="hi"))
        @test r3 isa ResponseCallError
        @test r3 isa ResponseCallError && occursin("\"id\"", r3.error)
    finally
        close(srv)
    end
end

@testset "Interactions decode — consecutive model_output steps are one answer; a tool or input step ends it" begin
    # A model answering across several steps — the hops of a long decode, thoughts between them —
    # adds a model_output step each, read as one text joined with no separator: Google's SDK
    # (python-genai 180c3453) continues "across `thought` steps and empty `model_output` steps …
    # while still stopping at `user_input` and tool call/result boundaries".
    thought(sig) = Dict("type" => "thought", "signature" => sig)
    say(texts...) = Dict("type" => "model_output", "content" => [Dict("type" => "text", "text" => t) for t in texts])
    decode(steps) = UniLM._interaction_response_object(Dict("id" => "v1_m", "status" => "completed",
        "model" => "gemini-3.8-flash", "steps" => steps))
    ro = decode([thought("a"), say("A"), thought("b"), say("B")])
    @test output_text(ro) == "AB"
    @test [o["type"] for o in ro.output] == ["thought", "message", "thought"]
    @test ro.output[2]["content"] == [Dict("type" => "output_text", "text" => "AB")]
    # The text parts of a step join the same way, and a step with none continues the answer.
    @test output_text(decode([say("First part. ", "Second part. "), Dict("type" => "model_output"),
                              say("Third part.")])) == "First part. Second part. Third part."
    # A call and its result, or a hosted tool's step, end the answer: the text after one is a message of
    # its own (the SDK's tool_boundary case), and output_text joins the messages with a newline.
    call = Dict("type" => "function_call", "id" => "c1", "name" => "lookup", "arguments" => Dict("q" => "x"))
    result = Dict("type" => "function_result", "call_id" => "c1", "name" => "lookup", "result" => "y")
    ro = decode([say("Before. "), call, result, thought("a"), say("Final 1. "), thought("b"), say("Final 2.")])
    @test [o["type"] for o in ro.output] == ["message", "function_call", "function_result", "thought", "message", "thought"]
    @test [only(o["content"])["text"] for o in ro.output if o["type"] == "message"] == ["Before. ", "Final 1. Final 2."]
    @test output_text(ro) == "Before. \nFinal 1. Final 2."
    search = Dict("type" => "google_search_call", "id" => "s1")
    @test output_text(decode([say("A"), search, say("B")])) == "A\nB"
    @test count(o -> o["type"] == "message", decode([say("A"), search, say("B")]).output) == 2
    # An input step the interaction lists is not the model's answer: a user_input step starts a new one.
    asked = Dict("type" => "user_input", "content" => [Dict("type" => "text", "text" => "Q")])
    @test [o["content"][1]["text"] for o in decode([say("Earlier."), asked, say("Now.")]).output if o["type"] == "message"] ==
          ["Earlier.", "Now."]
    # Streamed, the same steps assemble into the same output.
    ev(name, data) = "event: $name\ndata: $(JSON.json(data))\n\n"
    signature(index, sig) = ev("step.delta", Dict("index" => index,
                                                  "delta" => Dict("type" => "thought_signature", "signature" => sig)))
    text(index, t) = ev("step.delta", Dict("index" => index, "delta" => Dict("type" => "text", "text" => t)))
    sse = ev("step.start", Dict("index" => 0, "step" => Dict("type" => "thought"))) * signature(0, "a") * text(1, "A") *
          ev("step.start", Dict("index" => 2, "step" => Dict("type" => "thought"))) * signature(2, "b") * text(3, "B") *
          ev("interaction.completed", Dict("interaction" => Dict("id" => "v1_m", "status" => "completed",
                                                                 "model" => "gemini-3.8-flash")))
    rd = UniLM.decode_agentic_stream(GEMINIServiceEndpoint, sse, UniLM.AgenticStreamState()).data["response"]
    @test rd["output"] == decode([thought("a"), say("A"), thought("b"), say("B")]).output
end

@testset "Interactions decode — the faults an interaction records reach ResponseObject.error" begin
    # An Interaction reports its faults in `errors`, an array of {code, message}
    # (ai.google.dev/api/interactions-api): the error surface holds the first, raw keeps them all.
    faults = [Dict("code" => "https://example.com/errors/hop", "message" => "hop 2 failed"),
              Dict("code" => "https://example.com/errors/retry", "message" => "retried")]
    data = Dict("id" => "v1_e", "status" => "completed", "model" => "gemini-4-argon", "errors" => faults,
                "steps" => [Dict("type" => "model_output", "content" => [Dict("type" => "text", "text" => "partial")])])
    ro = UniLM.decode_agentic(GEMINIServiceEndpoint, HTTP.Response(200, [], Vector{UInt8}(JSON.json(data))))
    @test ro.error == faults[1]
    @test ro.raw["errors"] == faults
    terminal = Dict("event_type" => "interaction.completed", "interaction" => delete!(copy(data), "steps"))
    sse = "event: step.delta\ndata: {\"index\":0,\"delta\":{\"type\":\"text\",\"text\":\"partial\"}}\n\n" *
          "event: interaction.completed\ndata: $(JSON.json(terminal))\n\n" * "event: done\ndata: [DONE]\n\n"
    st = UniLM.decode_agentic_stream(GEMINIServiceEndpoint, sse, UniLM.AgenticStreamState())
    @test st.terminal == :completed && st.data["response"]["error"] == faults[1]
    # Through the stream driver, the result carries the same error surface as unstreamed.
    server, url = _ix_sse_server(sse)
    _IX_MOCK_URL[] = url
    try
        t = respond(Respond(service=_IxStreamMock, input="hi", stream=true);
                    config=RequestConfig(request_timeout=5.0, total_deadline=20.0, stream_idle_timeout=5.0, max_attempts=1))
        @test timedwait(() -> istaskdone(t), 25.0) == :ok
        res = fetch(t)
        @test res isa ResponseSuccess && res.response.error == faults[1] && output_text(res) == "partial"
    finally
        close(server)
    end
    # An `error` object is kept as given; an interaction without faults has none.
    @test UniLM._interaction_response_object(Dict("id" => "v1_x", "status" => "failed", "error" => faults[2])).error == faults[2]
    @test isnothing(UniLM._interaction_response_object(Dict("id" => "v1_y", "status" => "completed", "errors" => [])).error)
    @test !haskey(UniLM._interaction_response_dict(Dict("id" => "v1_y", "status" => "completed")), "error")
end

@testset "Interactions encode — Gemini 4 Argon takes more than 65,536 output tokens only in the background" begin
    # ai.google.dev/gemini-api/docs/long-decode: with background=false, a max_output_tokens above
    # 65,536 is a 400 "max_output_tokens greater than 65536 is not yet supported with
    # background=false. Please use background=true." (observed 2026-10-22).
    argon(; kw...) = UniLM.encode_agentic(GEMINIServiceEndpoint,
        Respond(; service=GEMINIServiceEndpoint, model="gemini-4-argon", input="x", kw...))
    for kw in ((max_output_tokens=70_000,), (max_output_tokens=65_537, background=false))
        err = try argon(; kw...); nothing catch e; e end
        @test err isa ArgumentError && occursin("gemini-4-argon", err.msg) && occursin("background=true", err.msg)
    end
    b = JSON.parse(argon(max_output_tokens=70_000, background=true))
    @test b["generation_config"] == Dict("max_output_tokens" => 70_000) && b["background"] == true
    @test JSON.parse(argon(max_output_tokens=65_536))["generation_config"] == Dict("max_output_tokens" => 65_536)
    # The limit is that model's: another model's cap goes as given, and the server's answer is the check.
    @test JSON.parse(UniLM.encode_agentic(GEMINIServiceEndpoint, Respond(service=GEMINIServiceEndpoint,
        model="gemini-3.8-flash", input="x", max_output_tokens=70_000)))["generation_config"] ==
          Dict("max_output_tokens" => 70_000)
end

@testset "Interactions — the model that answered prices the interaction" begin
    # An interaction names the model that served it (observed 2026-10-22: a gemini-3.7-flash request
    # answers as gemini-3.8-flash), so an alias is billed at the rates of the model behind it.
    usage = Dict("total_input_tokens" => 1_000, "total_output_tokens" => 100, "total_thought_tokens" => 0,
                 "total_tokens" => 1_100)
    ro = UniLM._interaction_response_object(Dict("id" => "v1_r", "status" => "completed",
        "model" => "gemini-3.5-flash-lite", "steps" => [], "usage" => usage))   # asked: gemini-flash-lite-latest
    @test estimated_cost(ResponseSuccess(response=ro, service=GEMINIServiceEndpoint)) ≈ (1_000 * 0.30 + 100 * 2.50) / 1e6
end

@testset "Interactions — one interaction streamed and not reports the same usage and model" begin
    # So the cost of a respond call does not depend on `stream`.
    interaction = Dict("id" => "v1_p", "status" => "completed", "model" => "gemini-3.8-flash", "service_tier" => "standard",
        "usage" => Dict("total_input_tokens" => 40, "total_tool_use_tokens" => 6, "total_cached_tokens" => 8,
                        "total_output_tokens" => 9, "total_thought_tokens" => 20, "total_tokens" => 75),
        "steps" => [Dict("type" => "thought", "signature" => "s"),
                    Dict("type" => "model_output", "content" => [Dict("type" => "text", "text" => "OK")])])
    terminal = Dict("event_type" => "interaction.completed", "interaction" => delete!(copy(interaction), "steps"))
    sse = "event: step.start\ndata: {\"index\":0,\"step\":{\"type\":\"thought\",\"signature\":\"s\"}}\n\n" *
          "event: step.delta\ndata: {\"index\":1,\"delta\":{\"type\":\"text\",\"text\":\"OK\"}}\n\n" *
          "event: interaction.completed\ndata: $(JSON.json(terminal))\n\n" * "event: done\ndata: [DONE]\n\n"
    cfg = RequestConfig(request_timeout=5.0, total_deadline=20.0, stream_idle_timeout=5.0, max_attempts=1)
    server, url = _ix_sse_server(sse)
    _IX_MOCK_URL[] = url
    streamed = try
        t = respond(Respond(service=_IxStreamMock, input="hi", stream=true); config=cfg)
        @test timedwait(() -> istaskdone(t), 25.0) == :ok
        fetch(t)
    finally
        close(server)
    end
    tcp = Sockets.listen(Sockets.localhost, 0)
    port = Int(Sockets.getsockname(tcp)[2]); close(tcp)
    srv = HTTP.serve!("127.0.0.1", port; verbose=false) do req
        HTTP.Response(200, ["Content-Type" => "application/json"], JSON.json(interaction))
    end
    _IX_NONSTREAM_URL[] = "http://127.0.0.1:$port/v1beta/interactions"
    plain = try respond(Respond(service=_IxNonStreamMock, input="hi"); config=cfg) finally close(srv) end
    @test streamed isa ResponseSuccess && plain isa ResponseSuccess
    @test token_usage(streamed) == token_usage(plain) ==
          TokenUsage(prompt_tokens=46, completion_tokens=29, total_tokens=75, cached_tokens=8, reasoning_tokens=20,
                     service_tier="standard")
    @test streamed.response.model == plain.response.model == "gemini-3.8-flash"
    @test streamed.response.output == plain.response.output
    @test estimated_cost(streamed) == estimated_cost(plain) > 0
end

# ─── Input: messages as steps, media as content parts ─────────────────────────

const _IX_FIXTURES = joinpath(@__DIR__, "fixtures")
_ix_b64(bytes) = UniLM.Base64.base64encode(bytes)
_ix_caught(f) = try f(); nothing catch e; e end
_ix_input(input; model="gemini-3.8-flash", kw...) = JSON.parse(UniLM.encode_agentic(GEMINIServiceEndpoint,
    Respond(; service=GEMINIServiceEndpoint, model, input, kw...)); dicttype=Dict{String,Any})["input"]
_ix_text(t) = Dict("type" => "text", "text" => t)

@testset "Interactions encode — messages are steps (the API refuses a list of turns)" begin
    # A list of {role, content} turns is HTTP 400 "use step_list input format instead of turn_list"
    # (observed 2026-10-22): a user message is a user_input step, an assistant message a model_output step.
    @test _ix_input([InputMessage(role="user", content="Hi"), InputMessage(role="assistant", content="Hello."),
                     InputMessage(role="user", content="Again")]) ==
        [Dict("type" => "user_input", "content" => [_ix_text("Hi")]),
         Dict("type" => "model_output", "content" => [_ix_text("Hello.")]),
         Dict("type" => "user_input", "content" => [_ix_text("Again")])]
    # A message given as a Dict, typed "message" or not, is the same step.
    for item in (Dict("role" => "user", "content" => "Hi"), Dict("type" => "message", "role" => "user", "content" => "Hi"),
                 Dict(:role => "user", :content => "Hi"))
        @test _ix_input([item]) == [Dict("type" => "user_input", "content" => [_ix_text("Hi")])]
    end
    # The neutral Responses text parts of a replayed turn become Interactions text parts.
    @test _ix_input([InputMessage(role="user", content=[input_text("Hi")]),
                     Dict("role" => "assistant", "content" => [Dict("type" => "output_text", "text" => "Hello.")])]) ==
        [Dict("type" => "user_input", "content" => [_ix_text("Hi")]),
         Dict("type" => "model_output", "content" => [_ix_text("Hello.")])]
    # A String input, a tool result and a step already in the Interactions shape pass as given.
    @test _ix_input("Hi") == "Hi"
    step = Dict("type" => "user_input", "content" => [_ix_text("Hi")])
    @test _ix_input([step, tool_result("c1", "f", "ok")]) == [step, Dict("type" => "function_result",
        "call_id" => "c1", "name" => "f", "result" => Dict("result" => "ok"))]
    # System and developer guidance has no step: it is the request's instructions.
    for role in ("system", "developer")
        e = _ix_caught(() -> _ix_input([InputMessage(; role, content="Be brief.")]))
        @test e isa ArgumentError && occursin(repr(role), e.msg) && occursin("instructions", e.msg)
    end
end

@testset "Interactions encode — a message phase is refused before any request" begin
    msg = "Gemini Interactions has no message phase: unset InputMessage phase, or send the request to an OpenAI Responses service"
    hits = Threads.Atomic{Int}(0)
    srv = HTTP.serve!("127.0.0.1", 0; listenany=true, verbose=false) do req
        Threads.atomic_add!(hits, 1); HTTP.Response(500, [], Vector{UInt8}("{}"))
    end
    _IX_NONSTREAM_URL[] = "http://127.0.0.1:$(HTTP.port(srv))/v1beta/interactions"
    try
        for item in (InputMessage(role="assistant", content="Checking.", phase="commentary"),
                     Dict("role" => "assistant", "content" => "Done.", "phase" => "final_answer"))
            input = [InputMessage(role="user", content="Hi"), item]
            e = _ix_caught(() -> _ix_input(input))
            @test e isa ArgumentError && e.msg == msg
            @test_throws ArgumentError respond(Respond(service=_IxNonStreamMock, input=input))
        end
    finally
        close(srv)
    end
    @test hits[] == 0
end

@testset "Interactions encode — images, audio and PDF documents are content parts" begin
    png = read(joinpath(_IX_FIXTURES, "ollama", "red_circle.png"))
    wav = read(joinpath(_IX_FIXTURES, "ollama", "secret_word.wav"))
    pdf = DocumentAttachment(joinpath(_IX_FIXTURES, "documents", "secret_word.pdf"))
    @test _ix_input([InputMessage(role="user", content=[input_text("What is in these?"),
                                                        ImageAttachment(png), AudioAttachment(wav), pdf])])[1]["content"] ==
        [_ix_text("What is in these?"),
         Dict("type" => "image", "mime_type" => "image/png", "data" => _ix_b64(png)),
         Dict("type" => "audio", "mime_type" => "audio/wav", "data" => _ix_b64(wav)),
         Dict("type" => "document", "mime_type" => "application/pdf", "data" => _ix_b64(pdf.data))]
    # The neutral Responses parts: input_file(document), a PDF data URL, and a data-URL input_image.
    png_url = "data:image/png;base64," * _ix_b64(png)
    @test _ix_input([InputMessage(role="user", content=[input_file(pdf), input_image(png_url; detail="low")])])[1]["content"] ==
        [Dict("type" => "document", "mime_type" => "application/pdf", "data" => _ix_b64(pdf.data)),
         Dict("type" => "image", "mime_type" => "image/png", "data" => _ix_b64(png), "resolution" => "low")]
    # An image's detail level is its resolution: low and high; auto is the model's default; original has no level.
    res(detail; model="gemini-3.8-flash") = get(_ix_input([InputMessage(role="user",
        content=[ImageAttachment(png; detail)])]; model)[1]["content"][1], "resolution", nothing)
    @test res("low") == "low" && res("high") == "high" && isnothing(res("auto")) && isnothing(res(nothing))
    e = _ix_caught(() -> res("original"))
    @test e isa ArgumentError && occursin("\"original\"", e.msg)
    e = _ix_caught(() -> res("low"; model="gemini-2.5-flash"))
    @test e isa ArgumentError && occursin("gemini-2.5-flash", e.msg)
    # Media by URL or by an OpenAI file id is refused: UniLM sends media as bytes.
    for part in (input_image("https://example.com/cat.png"), input_image(; file_id="file-1"),
                 input_file(url="https://example.com/report.pdf"), input_file(id="file-1"))
        e = _ix_caught(() -> _ix_input([InputMessage(role="user", content=[input_text("Read."), part])]))
        @test e isa ArgumentError && occursin("Gemini Interactions", e.msg)
    end
    # A cache breakpoint has nowhere to go: Gemini caches prompts implicitly.
    e = _ix_caught(() -> _ix_input([InputMessage(role="user", content=[input_text("x"; cache_breakpoint=true)])]))
    @test e isa ArgumentError && occursin("cache_breakpoint", e.msg)
    # A value that is no content part at all.
    e = _ix_caught(() -> _ix_input([InputMessage(role="user", content=[42])]))
    @test e isa ArgumentError && occursin("Int64", e.msg)
end

@testset "Interactions encode — service_tier, seed and the safety label" begin
    body(; kw...) = JSON.parse(UniLM.encode_agentic(GEMINIServiceEndpoint,
        Respond(; service=GEMINIServiceEndpoint, model="gemini-3.8-flash", input="hi", kw...)); dicttype=Dict{String,Any})
    # flex, standard, priority and — on this API only — deferred (https://ai.google.dev/api/interactions-api).
    for tier in ("standard", "flex", "priority", "deferred")
        @test body(service_tier=tier) == Dict("model" => "gemini-3.8-flash", "input" => "hi", "service_tier" => tier)
    end
    for bad in ("auto", "default", "fast", "scale", "ultrafast")
        e = _ix_caught(() -> body(service_tier=bad))
        @test e isa ArgumentError && occursin("\"deferred\"", e.msg) && occursin(repr(bad), e.msg)
    end
    @test body(seed=7) == Dict("model" => "gemini-3.8-flash", "input" => "hi", "generation_config" => Dict("seed" => 7))
    # safety_identifier is the request label Google documents for an aggregator's end-user id.
    @test body(safety_identifier="user-42")["labels"] == Dict("safety_identifier" => "user-42")
    e = _ix_caught(() -> body(safety_identifier=repeat("0123456789abcdef", 4)))
    @test e isa ArgumentError && occursin("Gemini Interactions", e.msg) && occursin("one character too long", e.msg)
    for f in (:service_tier, :safety_identifier, :seed)
        @test f ∉ UniLM._INTERACTIONS_UNMAPPED_FIELDS
    end
end

@testset "Interactions — the billed tier is the interaction's service_tier, priced at its rates" begin
    usage = Dict("total_input_tokens" => 1_000, "total_output_tokens" => 100, "total_thought_tokens" => 0,
                 "total_tokens" => 1_100)
    success(tier) = ResponseSuccess(response=UniLM._interaction_response_object(Dict("id" => "v1_t",
        "status" => "completed", "model" => "gemini-3.8-flash", "steps" => [], "usage" => usage,
        "service_tier" => tier)), service=GEMINIServiceEndpoint)
    @test token_usage(success("priority")).service_tier == "priority"
    # Gemini 3.8 Flash: Priority $1.35 / $6.75 and Flex $0.375 / $1.875 per 1M tokens.
    @test estimated_cost(success("priority")) ≈ (1_000 * 1.35 + 100 * 6.75) / 1e6
    @test estimated_cost(success("flex")) ≈ (1_000 * 0.375 + 100 * 1.875) / 1e6
    @test estimated_cost(success("standard")) ≈ (1_000 * 0.75 + 100 * 3.75) / 1e6
end

@testset "seed — Gemini Interactions sends it; every OpenAI-wire Responses encoder refuses it" begin
    # OpenAI's Responses API takes no seed (Chat Completions does): on that wire a set seed would vanish.
    for (svc, model) in ((OPENAIServiceEndpoint, "gpt-6-luna"), (UniLM.AZUREServiceEndpoint, "m"),
                         (GEMINIOpenAIServiceEndpoint, "gemini-3.8-flash"), (GenericOpenAIEndpoint("http://127.0.0.1:1", ""), "m"))
        e = _ix_caught(() -> UniLM.encode_agentic(svc, Respond(; service=svc, model, input="x", seed=7)))
        @test e isa ArgumentError && occursin("seed", e.msg) && occursin(string(UniLM._service_name(svc)), e.msg)
        @test UniLM.encode_agentic(svc, Respond(; service=svc, model, input="x")) isa String
    end
    # Ollama's Responses API reads a fixed set of fields; it refuses every other one, seed among them.
    e = _ix_caught(() -> UniLM.encode_agentic(OllamaEndpoint(), Respond(service=OllamaEndpoint(), model="m", input="x", seed=7)))
    @test e isa ArgumentError && occursin("seed", e.msg) && occursin("Ollama", e.msg)
    # Gemini Interactions sends it as generation_config.seed.
    @test JSON.parse(UniLM.encode_agentic(GEMINIServiceEndpoint, Respond(service=GEMINIServiceEndpoint, input="x",
        seed=7)))["generation_config"] == Dict("seed" => 7)
    # The OpenAI Responses JSON never carries it, and the refusal comes before any request.
    @test !haskey(JSON.lower(Respond(input="x", seed=7)), :seed)
    hits = Threads.Atomic{Int}(0)
    srv = HTTP.serve!("127.0.0.1", 0; listenany=true, verbose=false) do req
        Threads.atomic_add!(hits, 1); HTTP.Response(500, [], Vector{UInt8}("{}"))
    end
    try
        url = "http://127.0.0.1:$(HTTP.port(srv))"
        @test_throws ArgumentError respond(Respond(service=GenericOpenAIEndpoint(url, ""), model="m", input="x", seed=7))
    finally
        close(srv)
    end
    @test hits[] == 0
end
