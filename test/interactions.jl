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
    r = Respond(service=GEMINIServiceEndpoint, model="gemini-3.1-flash-lite",
                input="Say hi", instructions="Be terse",
                tools=[function_tool("get_weather", "Get weather",
                       parameters=Dict("type" => "object",
                                       "properties" => Dict("city" => Dict("type" => "string"))))],
                temperature=0.2, previous_response_id="v1_prev", store=true, stream=true)
    b = JSON.parse(UniLM.encode_agentic(GEMINIServiceEndpoint, r); dicttype=Dict{String,Any})
    @test b["model"] == "gemini-3.1-flash-lite"
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
        Respond(service=GEMINIServiceEndpoint, input="x",
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
    @test String(take!(state.textbuff)) == "hi"

    # (b) a trailing fragment after the last newline is stashed for the next chunk
    state2 = UniLM.AgenticStreamState()
    UniLM.decode_agentic_stream(GEMINIServiceEndpoint,
        "event: step.delta\ndata: {\"index\":0,\"delta\":{\"text\":\"a\",\"type\":\"text\"}}\n\n" *
        "event: step.delta\ndata: {\"index\":0,\"delta\":{\"text\":\"b\"", state2)
    @test String(take!(state2.textbuff)) == "a"                       # first delta consumed
    UniLM.decode_agentic_stream(GEMINIServiceEndpoint, ",\"type\":\"text\"}}\n\n", state2)
    @test String(take!(state2.textbuff)) == "b"                       # stashed fragment completed

    # (c) a malformed COMPLETE data line is dropped + counted (never re-queued
    # into the carry) — the shared machine's contract; no crash, stream continues.
    before = UniLM._SSE_DROPPED_LINES[]
    state3 = UniLM.AgenticStreamState()
    st3 = UniLM.decode_agentic_stream(GEMINIServiceEndpoint,
        "event: step.delta\ndata: {not valid json\n\n", state3)
    @test st3.terminal == :none
    @test isempty(take!(state3.carry))
    @test UniLM._SSE_DROPPED_LINES[] == before + 1
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
    raw = Dict("total_input_tokens" => 12, "total_output_tokens" => 69, "total_thought_tokens" => 5,
               "total_tool_use_tokens" => 3, "total_cached_tokens" => 4, "total_tokens" => 93)
    u = UniLM._interaction_usage(raw)
    @test u["input_tokens"] == 12
    @test u["output_tokens"] == 77                              # 69 + 5 thought + 3 tool_use (billable output)
    @test u["input_tokens_details"]["cached_tokens"] == 4
    @test u["output_tokens_details"]["reasoning_tokens"] == 5
    @test isnothing(UniLM._interaction_usage(nothing))
    # end-to-end: a decoded interaction yields correct token_usage + non-zero cost (cached=0 → clean rate identity)
    raw0 = Dict("total_input_tokens" => 12, "total_output_tokens" => 69, "total_thought_tokens" => 5,
                "total_tool_use_tokens" => 3, "total_cached_tokens" => 0, "total_tokens" => 89)
    ro = UniLM._interaction_response_object(Dict("id" => "v1_u", "status" => "completed",
        "model" => "gemini-3.1-flash-lite",
        "steps" => [Dict("type" => "model_output", "content" => [Dict("type" => "text", "text" => "hi")])],
        "usage" => raw0))
    res = ResponseSuccess(response=ro)
    @test token_usage(res).prompt_tokens == 12
    @test token_usage(res).completion_tokens == 77
    @test token_usage(res).reasoning_tokens == 5
    @test estimated_cost(res) ≈ 12 * 0.25/1_000_000 + 77 * 1.5/1_000_000   # gemini-3.1-flash-lite rates
    @test estimated_cost(res) > 0
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
