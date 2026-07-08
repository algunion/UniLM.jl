# Native Gemini Interactions (Layer-B agentic verb) — deterministic, zero-spend.
# Golden/canned bodies from the LIVE wire capture 2026-07-07 (gemini-3.1-flash-lite).

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
    # tool_choice must fail LOUD (Plan 3), never silently drop the caller's intent
    @test_throws ArgumentError UniLM.encode_agentic(GEMINIServiceEndpoint,
        Respond(service=GEMINIServiceEndpoint, input="x", tool_choice="required"))
end

@testset "Interactions decode (steps[] → neutral output[])" begin
    make(b) = HTTP.Response(200, [], Vector{UInt8}(JSON.json(b)))

    # text (captured shape): thought step + model_output content
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
    @test ro.usage["total_output_tokens"] == 2

    # function call (captured): arguments is an OBJECT → normalized to a JSON STRING
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
end

@testset "Interactions stream decode (interaction.* SSE)" begin
    tb = IOBuffer(); fb = IOBuffer(); ev = Ref("")
    UniLM.decode_agentic_stream(GEMINIServiceEndpoint,
        "event: step.delta\ndata: {\"index\":1,\"delta\":{\"text\":\"One, \",\"type\":\"text\"}}\n\n", tb, fb, ev)
    st1 = UniLM.decode_agentic_stream(GEMINIServiceEndpoint,
        "event: step.delta\ndata: {\"index\":1,\"delta\":{\"text\":\"two.\",\"type\":\"text\"}}\n\n", tb, fb, ev)
    @test st1.terminal == :none

    # real interaction.completed carries NO steps → final output rebuilt from the deltas
    completed = Dict("interaction" => Dict("id" => "v1_s", "status" => "completed",
        "model" => "gemini-3.1-flash-lite", "usage" => Dict("total_tokens" => 17)),
        "event_type" => "interaction.completed")
    st = UniLM.decode_agentic_stream(GEMINIServiceEndpoint,
        "event: interaction.completed\ndata: $(JSON.json(completed))\n\n", tb, fb, ev)
    @test st.terminal == :completed
    @test st.data["response"]["id"] == "v1_s"
    @test st.data["response"]["output"][1]["type"] == "message"
    @test st.data["response"]["output"][1]["content"][1]["text"] == "One, two."

    # [DONE] sentinel → terminal :done
    st2 = UniLM.decode_agentic_stream(GEMINIServiceEndpoint,
        "event: done\ndata: [DONE]\n\n", IOBuffer(), IOBuffer(), Ref(""))
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
    tb = IOBuffer(); fb = IOBuffer(); ev = Ref("")
    st = UniLM.decode_agentic_stream(GEMINIServiceEndpoint, "event: step.delta", tb, fb, ev)
    @test st.done == false && st.terminal == :none
    # the buffered partial line reassembles with the next chunk
    UniLM.decode_agentic_stream(GEMINIServiceEndpoint,
        "\ndata: {\"index\":0,\"delta\":{\"text\":\"hi\",\"type\":\"text\"}}\n\n", tb, fb, ev)
    @test String(take!(tb)) == "hi"

    # (b) a trailing fragment after the last newline is stashed for the next chunk
    tb2 = IOBuffer(); fb2 = IOBuffer(); ev2 = Ref("")
    UniLM.decode_agentic_stream(GEMINIServiceEndpoint,
        "event: step.delta\ndata: {\"index\":0,\"delta\":{\"text\":\"a\",\"type\":\"text\"}}\n\n" *
        "event: step.delta\ndata: {\"index\":0,\"delta\":{\"text\":\"b\"", tb2, fb2, ev2)
    @test String(take!(tb2)) == "a"                       # first delta consumed
    UniLM.decode_agentic_stream(GEMINIServiceEndpoint, ",\"type\":\"text\"}}\n\n", tb2, fb2, ev2)
    @test String(take!(tb2)) == "b"                       # stashed fragment completed

    # (c) a malformed data line is routed to failbuff via catch — no crash
    tb3 = IOBuffer(); fb3 = IOBuffer(); ev3 = Ref("")
    st3 = UniLM.decode_agentic_stream(GEMINIServiceEndpoint,
        "event: step.delta\ndata: {not valid json\n\n", tb3, fb3, ev3)
    @test st3.terminal == :none
    @test !isempty(take!(fb3))
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
    # String input still passes through
    b3 = JSON.parse(UniLM.encode_agentic(GEMINIServiceEndpoint,
        Respond(service=GEMINIServiceEndpoint, input="hi")); dicttype=Dict{String,Any})
    @test b3["input"] == "hi"
end

@testset "Interactions encode — CallableTool tool unwraps to function shape" begin
    ct = UniLM.CallableTool(function_tool("f", "d"), (n, a) -> "x")
    @test UniLM._interactions_tool(ct) == Dict{Symbol,Any}(:type => "function", :name => "f", :description => "d")
end
