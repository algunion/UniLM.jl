# Gemini Interactions — Implementation Plan (Plan 2: native provider on the agentic seam)

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:executing-plans / subagent-driven-development. Steps use `- [ ]`.

**Goal:** Add native Gemini Interactions as a second dispatch target on the agentic seam built in Plan 1, so `respond(input; service=GEMINIServiceEndpoint)` works for text, function tools, and streaming — proving the neutral IR against a *second live surface*.

**Architecture:** New self-contained `src/interactions.jl` overriding the three agentic-seam generics + `get_url` for `GEMINIServiceEndpoint`. The decoder normalizes Interactions `steps[]` → OpenAI-style `output[]` so the existing `ResponseObject` accessors (`output_text`, `function_calls`) are reused unchanged. Included at `src/UniLM.jl:57` (after `gemini.jl`, so the seam generics + `ResponseObject` from `responses.jl:52` are in scope).

**Tech Stack:** Julia, HTTP.jl, JSON layer, Test.

**Wire source of truth:** LIVE capture 2026-07-07 (`scratchpad/interactions-wire-capture.md`), NOT the research summary. Every golden/canned test below is built from those captured bytes. Divergences the capture caught: `usage.total_input_tokens`/`total_output_tokens` (not `input_tokens`/`promptTokenCount`); `function_call.arguments` is a JSON **object**; `function_result` requires `call_id`+`name`+`result`; SSE ends with `interaction.completed` **then** `event: done`/`[DONE]`.

## Global Constraints
- Neutral-IR discipline: normalize into the OpenAI-shaped `output[]`; provider specifics stay inside the translator. No raw escape-hatch dicts leaking to the accessor layer.
- Zero-spend: `env -u OPENAI_API_KEY -u DEEPSEEK_API_KEY -u ANTHROPIC_API_KEY -u GEMINI_API_KEY julia --project=. -e 'using Pkg; Pkg.test()'`. One key-gated live witness runs deliberately via `source ~/.zshrc` (billing-enabled; cheapest model; once).
- Both HTTP majors green.
- Git: conventional commits, no self-attribution.

## Scope
**IN:** `get_url` + `INTERACTIONS_PATH`; `:agentic` capability declaration; `encode_agentic` (model, string `input`, `system_instruction`, function `tools`, `generation_config` = temperature/top_p/max_output_tokens, `previous_interaction_id`, `store`/`background`/`stream`); `decode_agentic` (`steps[]`→`output[]`, status, raw usage); `decode_agentic_stream` (`interaction.*` SSE → text accumulation + `interaction.completed` assembly); unit tests from captures + one live witness.
**OUT (Plan 3):** cross-provider neutral tool-result *input* item (multi-turn tool submission — captured shape `{type:"function_result",call_id,name,result}`); `tool_choice` for Gemini (throws clearly for now); `background`/`get_response`/`cancel_response` generalization; hosted tools (google_search/code_execution/…); usage→TokenUsage normalization + pricing wiring.

## File Structure
- Create `src/interactions.jl` — all `GEMINIServiceEndpoint` agentic overrides + `_interaction_*` helpers.
- Modify `src/UniLM.jl:56-57` — `include("interactions.jl")` after `gemini.jl`.
- Modify `src/constants.jl` — `const INTERACTIONS_PATH`.
- Modify `src/capabilities.jl` / `src/gemini.jl:29` — add `:agentic`.
- Create `test/interactions.jl`; register in `test/runtests.jl`.

---

### Task 1: Routing + capability (`get_url`, constant, `:agentic`)

**Files:** Create `src/interactions.jl`; Modify `src/constants.jl`, `src/UniLM.jl:57`, `src/gemini.jl:29`, `src/capabilities.jl:22`; Test `test/interactions.jl` + `test/runtests.jl`.

**Interfaces produced:** `get_url(::Type{GEMINIServiceEndpoint}, ::Respond)::String`.

- [ ] **Step 1 — failing test** (`test/interactions.jl`):
```julia
@testset "Interactions routing + capability" begin
    r = Respond(service=GEMINIServiceEndpoint, model="gemini-3.1-flash-lite", input="hi")
    @test UniLM.get_url(r) == "https://generativelanguage.googleapis.com/v1beta/interactions"
    @test has_capability(GEMINIServiceEndpoint, :agentic)
    @test has_capability(OPENAIServiceEndpoint, :agentic)
end
```
- [ ] **Step 2 — run, expect FAIL** (`get_url(::GEMINI,::Respond)` currently throws via `_api_base_url`; `:agentic` undeclared).
- [ ] **Step 3 — implement:**
  - `src/constants.jl`: `const INTERACTIONS_PATH::String = "/interactions"`.
  - `src/interactions.jl` (new): `get_url(::Type{GEMINIServiceEndpoint}, ::Respond) = GEMINI_NATIVE_BASE * INTERACTIONS_PATH` (streaming is a body flag, so no URL branch).
  - `src/gemini.jl:29`: add `:agentic` → `Set([:chat, :tools, :streaming, :agentic])`.
  - `src/capabilities.jl:22`: add `:agentic` to the OpenAI set.
  - `src/UniLM.jl`: `include("interactions.jl")` immediately after `include("gemini.jl")`.
  - `test/runtests.jl`: add `@testset "interactions.jl" begin include("interactions.jl") end`.
- [ ] **Step 4 — run, expect PASS.**
- [ ] **Step 5 — commit** `feat(interactions): route GEMINIServiceEndpoint agentic verb + :agentic capability`.

### Task 2: `encode_agentic` (Respond → Interactions body)

**Files:** Modify `src/interactions.jl`; Test `test/interactions.jl`.

- [ ] **Step 1 — failing test** (golden bodies, parsed back to compare — key order-independent):
```julia
@testset "Interactions encode" begin
    r = Respond(service=GEMINIServiceEndpoint, model="gemini-3.1-flash-lite",
                input="Say hi", instructions="Be terse",
                tools=[function_tool("get_weather", "Get weather",
                       parameters=Dict("type"=>"object","properties"=>Dict("city"=>Dict("type"=>"string"))))],
                temperature=0.2, previous_response_id="v1_prev", store=true, stream=true)
    b = JSON.parse(UniLM.encode_agentic(GEMINIServiceEndpoint, r); dicttype=Dict{String,Any})
    @test b["model"] == "gemini-3.1-flash-lite"
    @test b["input"] == "Say hi"
    @test b["system_instruction"] == "Be terse"
    @test b["tools"][1]["type"] == "function"
    @test b["tools"][1]["name"] == "get_weather"
    @test b["generation_config"]["temperature"] == 0.2
    @test b["previous_interaction_id"] == "v1_prev"   # neutral previous_response_id → Gemini name
    @test b["store"] == true
    @test b["stream"] == true
    # tool_choice deferred: must fail loud, not silently drop
    r2 = Respond(service=GEMINIServiceEndpoint, input="x", tool_choice="required")
    @test_throws ArgumentError UniLM.encode_agentic(GEMINIServiceEndpoint, r2)
end
```
- [ ] **Step 2 — run, expect FAIL** (`encode_agentic(::GEMINI, ::Respond)` falls to the OpenAI default `JSON.json(r)`).
- [ ] **Step 3 — implement** in `src/interactions.jl`:
```julia
function encode_agentic(::Type{GEMINIServiceEndpoint}, r::Respond)::String
    isnothing(r.tool_choice) || throw(ArgumentError(
        "tool_choice is not yet supported for Gemini Interactions (Plan 3); omit it or force via the prompt"))
    body = Dict{Symbol,Any}(:model => r.model, :input => r.input)
    isnothing(r.instructions) || (body[:system_instruction] = r.instructions)
    isnothing(r.tools) || (body[:tools] = [_interactions_tool(t) for t in r.tools])
    gen = Dict{Symbol,Any}()
    isnothing(r.temperature)         || (gen[:temperature] = r.temperature)
    isnothing(r.top_p)               || (gen[:top_p] = r.top_p)
    isnothing(r.max_output_tokens)   || (gen[:max_output_tokens] = r.max_output_tokens)
    isempty(gen) || (body[:generation_config] = gen)
    isnothing(r.previous_response_id) || (body[:previous_interaction_id] = r.previous_response_id)
    isnothing(r.store)      || (body[:store] = r.store)
    isnothing(r.background) || (body[:background] = r.background)
    isnothing(r.stream)     || (body[:stream] = r.stream)
    JSON.json(body)
end

# Gemini Interactions tools = flat OpenAI-Responses shape {type:function,name,description?,parameters?}
function _interactions_tool(t)
    t isa FunctionTool && return let d = Dict{Symbol,Any}(:type => "function", :name => t.name)
        isnothing(t.description) || (d[:description] = t.description)
        isnothing(t.parameters)  || (d[:parameters] = t.parameters)
        d
    end
    t isa AbstractDict && return t                      # pre-shaped passthrough
    throw(ArgumentError("Gemini Interactions supports only FunctionTool/Dict tools (got $(typeof(t)))"))
end
```
- [ ] **Step 4 — run, expect PASS.**
- [ ] **Step 5 — commit** `feat(interactions): encode Respond → Interactions request body`.

### Task 3: `decode_agentic` (`steps[]` → neutral `output[]`)

**Files:** Modify `src/interactions.jl`; Test `test/interactions.jl`.

- [ ] **Step 1 — failing test** (canned bodies = captured bytes):
```julia
@testset "Interactions decode" begin
    make(b) = HTTP.Response(200, [], Vector{UInt8}(JSON.json(b)))
    # text (captured)
    txt = Dict("id"=>"v1_t","object"=>"interaction","model"=>"gemini-3.1-flash-lite","status"=>"completed",
        "usage"=>Dict("total_tokens"=>10,"total_input_tokens"=>8,"total_output_tokens"=>2,"total_thought_tokens"=>0),
        "steps"=>[Dict("type"=>"thought","signature"=>"sig"),
                  Dict("type"=>"model_output","content"=>[Dict("type"=>"text","text"=>"Hello.")])])
    ro = UniLM.decode_agentic(GEMINIServiceEndpoint, make(txt))
    @test ro.id == "v1_t"
    @test ro.status == "completed"
    @test output_text(ro) == "Hello."
    @test ro.usage["total_output_tokens"] == 2
    # function call (captured): arguments is an OBJECT → normalized to a JSON STRING for the accessor
    fc = Dict("id"=>"v1_f","object"=>"interaction","model"=>"gemini-3.1-flash-lite","status"=>"requires_action",
        "steps"=>[Dict("id"=>"6eG7YnHo","type"=>"function_call","name"=>"get_weather",
                       "arguments"=>Dict("city"=>"Tokyo"),"signature"=>"sig")])
    ro2 = UniLM.decode_agentic(GEMINIServiceEndpoint, make(fc))
    @test ro2.status == "requires_action"
    calls = function_calls(ro2)
    @test length(calls) == 1
    @test calls[1]["name"] == "get_weather"
    @test calls[1]["call_id"] == "6eG7YnHo"
    @test JSON.parse(calls[1]["arguments"])["city"] == "Tokyo"
end
```
- [ ] **Step 2 — run, expect FAIL.**
- [ ] **Step 3 — implement** in `src/interactions.jl`:
```julia
# steps[] → OpenAI-Responses-shaped output[] (so ResponseObject accessors are reused verbatim)
function _interaction_output(steps)::Vector{Any}
    out = Any[]
    for s in (steps isa AbstractVector ? steps : ())
        s isa AbstractDict || continue
        t = get(s, "type", "")
        if t == "model_output"
            parts = Any[]
            for c in get(s, "content", ())
                c isa AbstractDict && get(c, "type", "") == "text" &&
                    push!(parts, Dict{String,Any}("type"=>"output_text", "text"=>get(c, "text", "")))
            end
            push!(out, Dict{String,Any}("type"=>"message", "role"=>"assistant", "content"=>parts))
        elseif t == "function_call"
            args = get(s, "arguments", Dict{String,Any}())
            push!(out, Dict{String,Any}("type"=>"function_call",
                "call_id"=>get(s, "id", ""), "name"=>get(s, "name", ""),
                "arguments"=> args isa AbstractString ? args : JSON.json(args)))
        elseif t == "thought"
            push!(out, Dict{String,Any}("type"=>"reasoning", "summary"=>Any[]))
        end
    end
    out
end

_interaction_response_object(data::AbstractDict)::ResponseObject = ResponseObject(
    id=get(data,"id",""), status=get(data,"status",""), model=get(data,"model",""),
    output=_interaction_output(get(data,"steps",Any[])),
    usage=get(data,"usage",nothing), error=get(data,"error",nothing),
    metadata=get(data,"metadata",nothing), raw=Dict{String,Any}(data))

decode_agentic(::Type{GEMINIServiceEndpoint}, resp::HTTP.Response)::ResponseObject =
    _interaction_response_object(JSON.parse(resp.body; dicttype=Dict{String,Any}))
```
- [ ] **Step 4 — run, expect PASS.**
- [ ] **Step 5 — commit** `feat(interactions): decode steps[] → neutral ResponseObject output[]`.

### Task 4: `decode_agentic_stream` (`interaction.*` SSE)

**Files:** Modify `src/interactions.jl`; Test `test/interactions.jl`.

Contract: mirror `_parse_response_stream_chunk`'s return `(; done, event, data, terminal)` + partial-line carry-over into `failbuff`, so `_respond_stream` (Plan 1) assembles unchanged. On `interaction.completed`, wrap the normalized response as `data=Dict("response"=>{id,status,model,output,usage})` — the shape `_respond_stream` already assembles.

- [ ] **Step 1 — failing test** (canned SSE = captured events):
```julia
@testset "Interactions stream decode" begin
    tb=IOBuffer(); fb=IOBuffer(); ev=Ref("")
    UniLM.decode_agentic_stream(GEMINIServiceEndpoint,
        "event: step.delta\ndata: {\"index\":1,\"delta\":{\"text\":\"One, \",\"type\":\"text\"}}\n\n", tb, fb, ev)
    UniLM.decode_agentic_stream(GEMINIServiceEndpoint,
        "event: step.delta\ndata: {\"index\":1,\"delta\":{\"text\":\"two.\",\"type\":\"text\"}}\n\n", tb, fb, ev)
    @test String(take!(tb)) == "One, two."
    tb2=IOBuffer(); fb2=IOBuffer(); ev2=Ref("")
    completed = Dict("interaction"=>Dict("id"=>"v1_s","status"=>"completed","model"=>"gemini-3.1-flash-lite",
        "usage"=>Dict("total_tokens"=>17),"steps"=>[Dict("type"=>"model_output","content"=>[Dict("type"=>"text","text"=>"One, two, three.")])]),
        "event_type"=>"interaction.completed")
    st = UniLM.decode_agentic_stream(GEMINIServiceEndpoint,
        "event: interaction.completed\ndata: $(JSON.json(completed))\n\n", tb2, fb2, ev2)
    @test st.terminal == :completed
    @test st.data["response"]["id"] == "v1_s"
    @test st.data["response"]["output"][1]["type"] == "message"
end
```
- [ ] **Step 2 — run, expect FAIL.**
- [ ] **Step 3 — implement** in `src/interactions.jl` (mirror `_parse_response_stream_chunk` carry-over; parse `event:`/`data:`; on `step.delta` with `delta.text` → `print(textbuff, …)`; on `interaction.completed` → normalize + return terminal `:completed` with `data["response"]`; on `event: done`/`[DONE]` → `done=true` no data). *(Full body written during execution against `_parse_response_stream_chunk` as the template; reuse `_interaction_output` for the completed step normalization.)*
- [ ] **Step 4 — run, expect PASS.**
- [ ] **Step 5 — commit** `feat(interactions): stream decode interaction.* SSE`.

## Exit gate
- Full zero-spend suite + Aqua green, **both HTTP majors**.
- **One live witness** (`test/integration_interactions.jl`, key-gated), run by `source ~/.zshrc` + `env -u` the other three, cheapest model, once: a text call (`output_text` non-empty), a tool call (`function_calls` non-empty, `requires_action`), and a stream (accumulated text) — the end-to-end proof mocks cannot give.
- Falsifier: if any accessor can't be filled from `steps[]` without a raw escape-hatch dict, the neutral-IR claim is refuted for that path → add a neutral field, don't hack.

## Ledger — confirm during/after
- `usage`: does `total_output_tokens` include `total_thought_tokens`? (all captures had thought=0). Needed before wiring cost.
- Streamed `function_call` delta shape (only text stream captured) — decoder must tolerate; capture if a tool stream is exercised.
- `tool_choice` real shape (deferred; throws for now).
