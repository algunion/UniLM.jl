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
    # `content: null` is spec-legal: a model_output step that produced no parts.
    body = Dict("id" => "v1_n", "status" => "completed", "model" => "gemini-3.1-flash-lite",
        "steps" => [Dict("type" => "model_output", "content" => nothing),
                    Dict("type" => "model_output",
                         "content" => [Dict("type" => "text", "text" => "ok")])])
    ro = UniLM.decode_agentic(GEMINIServiceEndpoint,
        HTTP.Response(200, [], Vector{UInt8}(JSON.json(body))))
    @test output_text(ro) == "ok"
    @test ro.output[1]["content"] == Any[]

    # streamed: a throw here is swallowed by the SSE drop policy (counted as a
    # dropped line), turning a clean interaction into a failure or an idle stall.
    before = UniLM._SSE_DROPPED_LINES[]
    st = UniLM.decode_agentic_stream(GEMINIServiceEndpoint,
        "event: interaction.completed\ndata: " *
        JSON.json(Dict("event_type" => "interaction.completed", "interaction" => body)) * "\n\n",
        UniLM.AgenticStreamState())
    @test UniLM._SSE_DROPPED_LINES[] == before
    @test st.terminal == :completed
    @test st.data["response"]["output"][1]["content"] == Any[]
    @test st.data["response"]["output"][2]["content"][1]["text"] == "ok"
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
    @test String(take!(st.textbuff)) == "Hello world"

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
    # Every field the Interactions body maps, set at once: the encoding is
    # byte-for-byte the one this wire produced before the guard existed.
    r = Respond(service=GEMINIServiceEndpoint, model="gemini-3.1-flash-lite", input="Say hi",
                instructions="Be terse", tools=[function_tool("get_weather", "Get weather")],
                tool_choice="auto", temperature=0.2, max_output_tokens=64,
                previous_response_id="v1_prev", store=true, background=false, stream=true)
    @test UniLM.encode_agentic(GEMINIServiceEndpoint, r) ==
        """{"background":false,"generation_config":{"max_output_tokens":64,"temperature":0.2,"tool_choice":{"allowed_tools":{"mode":"auto"}}},"input":"Say hi","model":"gemini-3.1-flash-lite","previous_interaction_id":"v1_prev","store":true,"stream":true,"system_instruction":"Be terse","tools":[{"description":"Get weather","name":"get_weather","type":"function"}]}"""

    # A field this wire has no mapping for must not vanish from the request.
    cases = (:text => UniLM.TextConfig(), :reasoning => UniLM.Reasoning(effort="low"),
             :metadata => Dict("k" => "v"), :truncation => "auto",
             :parallel_tool_calls => true, :user => "u1", :include => ["a"],
             :max_tool_calls => 2, :service_tier => "flex", :top_logprobs => 3,
             :prompt => Dict("id" => "p"), :prompt_cache_key => "k",
             :prompt_cache_retention => "24h", :safety_identifier => "s",
             :conversation => "conv_1", :context_management => [Dict("type" => "x")],
             :stream_options => Dict("include_usage" => true))
    @test Set(first.(cases)) == Set(UniLM._INTERACTIONS_UNMAPPED_FIELDS)   # every unmapped field covered
    for (field, value) in cases
        rr = Respond(; service=GEMINIServiceEndpoint, input="x", (field => value,)...)
        err = try UniLM.encode_agentic(GEMINIServiceEndpoint, rr); nothing catch e; e end
        @test err isa ArgumentError
        @test err isa ArgumentError && occursin(String(field), err.msg)
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
            "steps" => [])))
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
    finally
        close(srv)
    end
end
