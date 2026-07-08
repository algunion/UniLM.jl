# Cross-provider tool-loop completeness — Implementation Plan (Plan 3a)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use `- [ ]`.

**Goal:** Make multi-turn tool calling work on both providers of the agentic verb — a neutral `tool_result` item the Gemini encoder translates, plus Gemini `tool_choice` — fixing the silently-broken Gemini tool loop.

**Architecture:** OpenAI-shape-as-neutral (verified: OpenAI ignores an extra `name` on `function_call_output`). The Gemini encoder gains an input-item translation (`function_call_output` → `function_result{call_id, name, result}`) and a `tool_choice` mapping into `generation_config`; OpenAI's encoder is untouched. All Gemini wire confirmed live 2026-07-08.

**Tech Stack:** Julia (latest), HTTP.jl (`1.9, 2`), the repo JSON layer, `Test` + `Aqua`.

## Global Constraints

- **Neutral-IR discipline:** the neutral tool-result is OpenAI's `function_call_output` + optional `name` (OpenAI ignores it, verified); the Gemini encoder translates.
- **OpenAI encoder unchanged** — only `encode_agentic(::GEMINIServiceEndpoint)` and `tool_loop`/`_interactions_tool` change.
- **Confirmed Gemini wire:** tool result = `{type:"function_result", call_id, name, result:<object>}` (`name` **required**); `tool_choice` = `generation_config.tool_choice.allowed_tools.{mode, tools}` (`mode` `auto`/`any`/`none`; `tools` = function-name strings); a continuation **tolerates** re-sent `tools`.
- **Zero-spend testing** (⚠️ OpenAI+Anthropic keys LIVE in sandbox — bare `Pkg.test()` bills): `env -u OPENAI_API_KEY -u DEEPSEEK_API_KEY -u ANTHROPIC_API_KEY -u GEMINI_API_KEY julia --project=. -e 'using Pkg; Pkg.test()'`. The live witness runs deliberately via `source ~/.zshrc` (Gemini only; cheapest model; once).
- **Both HTTP majors** green; hold the project's ~99.5% coverage bar (cover the new branches + fail-loud paths).
- **Git:** conventional commits; no `Co-Authored-By: Claude` / "Generated with Claude Code" trailers.

## File Structure

- **Modify `src/responses.jl`** — add the `tool_result(call_id, name, output)` constructor (near the other item constructors, ~`:428`).
- **Modify `src/UniLM.jl:180`** — export `tool_result`.
- **Modify `src/interactions.jl`** — `encode_agentic` gains input translation (`:25`) + `tool_choice` mapping (remove the throw at `:23-24`, add to `generation_config`); add `_interactions_input`/`_interactions_input_item`/`_interactions_tool_choice` helpers.
- **Modify `src/tool_loop.jl`** — the Responses loop adds `"name"` to result items (`:237`); add `_interactions_tool(::CallableTool)` unwrap next to the existing `JSON.lower(::CallableTool)` (`:30`).
- **Modify `test/responses.jl`, `test/interactions.jl`, `test/mock_server.jl`, `test/integration_interactions.jl`.**

---

### Task 1: `tool_result` helper + export

**Files:** Modify `src/responses.jl` (~`:428`), `src/UniLM.jl:180`; Test `test/responses.jl`.

**Interfaces produced:** `tool_result(call_id::AbstractString, name::AbstractString, output::AbstractString)::Dict{String,Any}`.

- [ ] **Step 1: failing test** — append to `test/responses.jl`:
```julia
@testset "tool_result helper" begin
    tr = tool_result("call_1", "get_weather", "sunny")
    @test tr["type"] == "function_call_output"
    @test tr["call_id"] == "call_1"
    @test tr["name"] == "get_weather"
    @test tr["output"] == "sunny"
end
```
- [ ] **Step 2: run → FAIL** (`UndefVarError: tool_result`):
`env -u OPENAI_API_KEY -u DEEPSEEK_API_KEY -u ANTHROPIC_API_KEY -u GEMINI_API_KEY julia --project=. -e 'using Test, HTTP, JSON, UniLM; @testset "r" begin include("test/responses.jl") end' 2>&1 | tail -5`
- [ ] **Step 3: implement** — in `src/responses.jl`, after the `function_tool(d::AbstractDict)` method:
```julia
"""
    tool_result(call_id, name, output) -> Dict

Neutral multi-turn tool-result input item for the agentic verb. Feed a function's
output back via `respond(previous_response_id=id, input=[tool_result(...)])` or through
`tool_loop`. Wire-neutral: OpenAI serializes it as `function_call_output` (ignoring
`name`); the Gemini encoder translates it to `function_result` (which requires `name`).
`output` is the function's return value as a string.
"""
tool_result(call_id::AbstractString, name::AbstractString, output::AbstractString) =
    Dict{String,Any}("type" => "function_call_output", "call_id" => call_id,
                     "name" => name, "output" => output)
```
Then in `src/UniLM.jl`, add `tool_result,` on the line after `    function_tool,` (`:180`).
- [ ] **Step 4: run → PASS** (same command).
- [ ] **Step 5: commit**
```bash
git add src/responses.jl src/UniLM.jl test/responses.jl
git commit -m "feat(agentic): add neutral tool_result() multi-turn item helper"
```

---

### Task 2: Gemini input-item translation (`function_call_output` → `function_result`)

**Files:** Modify `src/interactions.jl` (`:25` + new helpers after `_interactions_tool`, ~`:52`); Test `test/interactions.jl`.

**Interfaces:**
- Consumes: `tool_result` (Task 1), `_gemini_tool_response` (`src/gemini.jl:131`).
- Produces: `_interactions_input(input)` used by `encode_agentic(::GEMINIServiceEndpoint)`.

- [ ] **Step 1: failing test** — append to `test/interactions.jl`:
```julia
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
```
- [ ] **Step 2: run → FAIL** (Gemini encode currently passes `input` through, so `input[1]` is the raw `function_call_output`, and the missing-name case doesn't throw):
`env -u OPENAI_API_KEY -u DEEPSEEK_API_KEY -u ANTHROPIC_API_KEY -u GEMINI_API_KEY julia --project=. -e 'using Test, HTTP, JSON, UniLM; @testset "i" begin include("test/interactions.jl") end' 2>&1 | tail -8`
- [ ] **Step 3: implement** — in `src/interactions.jl`, change line 25 from `body = Dict{Symbol,Any}(:model => r.model, :input => r.input)` to:
```julia
    body = Dict{Symbol,Any}(:model => r.model, :input => _interactions_input(r.input))
```
and add, after the `_interactions_tool` function (~`:52`):
```julia
# Neutral input items → Interactions input. A `function_call_output` tool-result item
# (OpenAI-shaped neutral, from tool_result/tool_loop) → Gemini `function_result{call_id,
# name, result}`; a String input and any other item pass through unchanged.
_interactions_input(input::AbstractString) = input
_interactions_input(input::AbstractVector) = Any[_interactions_input_item(x) for x in input]
_interactions_input(input) = input

function _interactions_input_item(x)
    (x isa AbstractDict && get(x, "type", "") == "function_call_output") || return x
    haskey(x, "name") || throw(ArgumentError(
        "Gemini function_result requires a name; build the item with tool_result(call_id, name, output)"))
    Dict{String,Any}(
        "type" => "function_result",
        "call_id" => get(x, "call_id", ""),
        "name" => x["name"],
        "result" => _gemini_tool_response(get(x, "output", "")))
end
```
- [ ] **Step 4: run → PASS** (same command).
- [ ] **Step 5: commit**
```bash
git add src/interactions.jl test/interactions.jl
git commit -m "feat(interactions): translate neutral function_call_output → Gemini function_result"
```

---

### Task 3: `tool_loop` carries `name` + `CallableTool` tool encoding

**Files:** Modify `src/tool_loop.jl` (`:237` + a method near `:30`); Test `test/interactions.jl`, `test/mock_server.jl`.

**Interfaces:**
- Consumes: `_interactions_tool` (`src/interactions.jl`), `function_calls` (`src/responses.jl:834`).
- Produces: Responses-loop result items now carry `"name"`; `_interactions_tool(::CallableTool)` unwraps.

- [ ] **Step 1: failing tests.** (a) In `test/mock_server.jl`, inside the `@testset "tool_loop two-turn cycle (Respond)"` block, after the `output_text(result.response) == "The result is 8."` assertion (~`:1149`), add:
```julia
        @test get(JSON.parse(request_body[]; dicttype=Dict{String,Any})["input"][1], "name", nothing) == "add"
```
(b) Append to `test/interactions.jl`:
```julia
@testset "Interactions encode — CallableTool tool unwraps to function shape" begin
    ct = UniLM.CallableTool(function_tool("f", "d"), (n, a) -> "x")
    @test UniLM._interactions_tool(ct) == Dict{Symbol,Any}(:type => "function", :name => "f", :description => "d")
end
```
- [ ] **Step 2: run → FAIL** — full zero-spend suite (mock test) + interactions (CallableTool):
`env -u OPENAI_API_KEY -u DEEPSEEK_API_KEY -u ANTHROPIC_API_KEY -u GEMINI_API_KEY julia --project=. -e 'using Pkg; Pkg.test()' 2>&1 | tail -12`
Expected: the mock assertion fails (loop omits `name`) and the CallableTool test errors (`_interactions_tool(::CallableTool)` → the "supports only FunctionTool/Dict" throw).
- [ ] **Step 3: implement** — in `src/tool_loop.jl`, change the result-item build (`:236-240`) to include `name`:
```julia
            push!(output_items, Dict{String,Any}(
                "type" => "function_call_output",
                "call_id" => call["call_id"],
                "name" => call["name"],
                "output" => content
            ))
```
and add, immediately after `JSON.lower(ct::CallableTool) = JSON.lower(ct.tool)` (`:30`):
```julia
# CallableTool unwraps to its inner tool for the Gemini Interactions encoder (mirrors the
# JSON.lower unwrap the OpenAI wire uses). Defined here — _interactions_tool lives in
# interactions.jl (loaded before tool_loop.jl), CallableTool is defined just above.
_interactions_tool(ct::CallableTool) = _interactions_tool(ct.tool)
```
- [ ] **Step 4: run → PASS** (same full-suite command).
- [ ] **Step 5: commit**
```bash
git add src/tool_loop.jl test/mock_server.jl test/interactions.jl
git commit -m "feat(agentic): tool_loop result items carry name; CallableTool encodes on Gemini"
```

---

### Task 4: Gemini `tool_choice` mapping

**Files:** Modify `src/interactions.jl` (remove throw `:23-24`, add to `gen`, add helper); Test `test/interactions.jl`.

**Interfaces:**
- Consumes: `tool_choice_function` (`src/responses.jl:463`), `tool_choice_hosted` (`:471`).
- Produces: `_interactions_tool_choice(tc)` folded into `generation_config.tool_choice`.

- [ ] **Step 1a: delete the now-obsolete throw assertion** — in `test/interactions.jl`, inside the `@testset "Interactions encode (Respond → snake_case body)"` block, remove these three lines (`:31-33`) — `tool_choice` is no longer a fail-loud (keep the `@test !haskey(b, "tool_choice")` line above them, which still holds for a `tool_choice`-less request):
```julia
    # tool_choice must fail LOUD (Plan 3), never silently drop the caller's intent
    @test_throws ArgumentError UniLM.encode_agentic(GEMINIServiceEndpoint,
        Respond(service=GEMINIServiceEndpoint, input="x", tool_choice="required"))
```
- [ ] **Step 1b: failing test** — append to `test/interactions.jl`:
```julia
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
end
```
- [ ] **Step 2: run → FAIL** (Gemini encode currently throws on any `tool_choice`):
`env -u OPENAI_API_KEY -u DEEPSEEK_API_KEY -u ANTHROPIC_API_KEY -u GEMINI_API_KEY julia --project=. -e 'using Test, HTTP, JSON, UniLM; @testset "i" begin include("test/interactions.jl") end' 2>&1 | tail -8`
- [ ] **Step 3: implement** — in `src/interactions.jl`:
  1. **Delete** the throw at the top of `encode_agentic` (`:23-24`):
     ```julia
         isnothing(r.tool_choice) || throw(ArgumentError(
             "tool_choice is not yet supported for Gemini Interactions (Plan 3); omit it or steer via the prompt"))
     ```
  2. After the `max_output_tokens` line in the `gen` block (`:31`), add:
     ```julia
         isnothing(r.tool_choice) || (gen[:tool_choice] = _interactions_tool_choice(r.tool_choice))
     ```
  3. Add the helper (near `_interactions_tool`):
```julia
# Neutral tool_choice → generation_config.tool_choice.allowed_tools.{mode, tools}
# (confirmed live: mode auto/any/none; tools = function-name strings).
_interactions_tool_choice(tc::AbstractString) =
    tc == "auto"     ? Dict{Symbol,Any}(:allowed_tools => Dict{Symbol,Any}(:mode => "auto")) :
    tc == "none"     ? Dict{Symbol,Any}(:allowed_tools => Dict{Symbol,Any}(:mode => "none")) :
    tc == "required" ? Dict{Symbol,Any}(:allowed_tools => Dict{Symbol,Any}(:mode => "any")) :
    throw(ArgumentError("Unknown tool_choice string $(repr(tc)) for Gemini Interactions"))

function _interactions_tool_choice(tc::AbstractDict)
    _g(k) = get(tc, k, get(tc, String(k), nothing))   # tolerate Symbol- or String-keyed dicts
    _g(:type) == "function" || throw(ArgumentError(
        "Gemini Interactions tool_choice supports \"auto\"/\"none\"/\"required\" or a specific " *
        "function (tool_choice_function); hosted-tool selectors are not applicable. Got $(repr(tc))"))
    Dict{Symbol,Any}(:allowed_tools => Dict{Symbol,Any}(:mode => "any", :tools => [_g(:name)]))
end
```
- [ ] **Step 4: run → PASS** (same command).
- [ ] **Step 5: commit**
```bash
git add src/interactions.jl test/interactions.jl
git commit -m "feat(interactions): map tool_choice into Gemini generation_config.tool_choice"
```

---

### Task 5: live witness — full Gemini tool loop + forced tool_choice

**Files:** Modify `test/integration_interactions.jl`.

- [ ] **Step 1: add the witness** — append inside the `if haskey(ENV, "GEMINI_API_KEY")` block of `test/integration_interactions.jl`:
```julia
@testset "Interactions — tool loop (live, neutral tool_result round-trip)" begin
    r = Respond(service=UniLM.GEMINIServiceEndpoint, model="gemini-3.1-flash-lite",
                input="What is the weather in Tokyo? Use the get_weather tool, then tell me.",
                tools=[function_tool("get_weather", "Get current weather for a city",
                    parameters=Dict("type" => "object",
                        "properties" => Dict("city" => Dict("type" => "string")), "required" => ["city"]))])
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
```
- [ ] **Step 2: run the witness deliberately** (Gemini only; billing-enabled; once):
```bash
source ~/.zshrc >/dev/null 2>&1
env -u OPENAI_API_KEY -u DEEPSEEK_API_KEY -u ANTHROPIC_API_KEY julia --project=. -e 'using Test, HTTP, JSON, UniLM; @testset "witness" begin include("test/integration_interactions.jl") end' 2>&1 | tail -8
```
Expected: PASS — the loop completes (turn 2 submits `function_result` via the neutral item + `previous_interaction_id` + re-sent tools), and `tool_choice="required"` forces a `function_call`.
- [ ] **Step 3: commit**
```bash
git add test/integration_interactions.jl
git commit -m "test(interactions): live witness — Gemini tool loop + forced tool_choice"
```

---

## Exit gate

- Full zero-spend suite + Aqua green on the local HTTP major, then **both majors**:
  `julia --project=. -e 'using Pkg; Pkg.add(PackageSpec(name="HTTP", version="1"))'` → zero-spend suite; repeat `version="2"`.
- **Live witness on both HTTP majors** (streaming/transport is major-sensitive; the loop streams nothing but the tool round-trip crosses the wire): run Task-5 Step-2 under HTTP 1 and HTTP 2.
- **Coverage:** confirm the new `interactions.jl`/`tool_loop.jl` branches (translation, fail-loud, tool_choice map, CallableTool unwrap) are covered — hold the project's ~99.5% bar (add unit cases if `codecov/patch` dips).
- **Falsifier:** any existing OpenAI `tool_loop`/responses/mock test or Aqua failing means the OpenAI path regressed — the change must be Gemini-only (it is: OpenAI encoder untouched; `tool_loop`'s extra `name` is OpenAI-tolerated, verified).

## Self-Review

**1. Spec coverage:** neutral tool-result item → Tasks 1-3 (`tool_result`, Gemini translation, loop `name` + CallableTool); `tool_choice` → Task 4; live witness → Task 5; continuation-tolerates-tools → confirmed at Phase 0, so no drop-tools task (correct). All spec IN-scope items map to a task.

**2. Placeholder scan:** none — every step has exact code + commands. The Gemini wire is captured, not deferred.

**3. Type consistency:** `tool_result` returns `Dict{String,Any}` with keys `type`/`call_id`/`name`/`output`, which `_interactions_input_item` reads (`get(x,"call_id",…)`, `x["name"]`, `get(x,"output",…)`) and `tool_loop` produces (same keys). `_interactions_tool_choice` builds Symbol-keyed dicts serialized under `generation_config[:tool_choice]`; the test reads the JSON-parsed String keys (`["allowed_tools"]["mode"]`). `_interactions_tool(::CallableTool)` delegates to the existing `_interactions_tool(::FunctionTool)`. Consistent across tasks.
