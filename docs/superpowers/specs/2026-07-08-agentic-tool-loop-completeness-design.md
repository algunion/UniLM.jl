# Design: Cross-provider tool-loop completeness (Plan 3a)

- **Date:** 2026-07-08
- **Status:** Approved (brainstorming) → ready for implementation plan
- **Keystone:** make multi-turn tool calling work on **both** providers of the agentic verb — a **neutral tool-result item** + Gemini `tool_choice` — closing the last neutrality gap in the merged `respond()` surface.
- **Approach:** OpenAI-shape-as-neutral (verified), the same pattern as `GPTToolCall.thought_signature`; the Gemini encoder translates the input items; OpenAI's encoder is untouched.

## Problem (one sentence)

`tool_loop(r::Respond)` (`src/tool_loop.jl:236-240`) hardcodes OpenAI's `function_call_output` tool-result shape, and `encode_agentic(::GEMINIServiceEndpoint)` passes `r.input` through verbatim (`src/interactions.jl:25`), so a **Gemini multi-turn tool loop is silently broken** — it POSTs `function_call_output` items to the Interactions API, which rejects them (`"function_call_output is not supported for type"`; Gemini needs `function_result{call_id, name, result}` with **`name` required**).

## Context — what exists, what's the gap

The merged agentic verb (PR #12 + #13) unifies the *request* and *response*: `respond()` dispatches `encode_agentic`/`decode_agentic`/`decode_agentic_stream` on `service`, and the Gemini decoder normalizes `steps[]` → OpenAI-shaped `output[]` so accessors are reused. **But feeding tool results *back* is not neutral:**

- `tool_loop(r::Respond)` (`src/tool_loop.jl:204-251`) is the agentic tool loop. Each turn it calls `respond`, reads `function_calls(result)`, dispatches, and builds the next `input` as OpenAI `function_call_output` items (`:237-239`), chaining via `previous_response_id` (`:244`).
- `function_calls` returns dicts with `"call_id"`, `"name"`, `"arguments"` (`src/responses.jl:822`) — for **both** providers (the Gemini decoder emits `{type:"function_call", call_id, name, arguments}`, `src/interactions.jl`).
- Gemini's `function_result` **requires `name`** (captured live 2026-07-07); OpenAI's `function_call_output` does **not** carry it. That asymmetry is the entire design problem.

## Decisions locked in brainstorming

1. **Neutral tool-result item = OpenAI `function_call_output` + an optional `name`.** VERIFIED 2026-07-08 via a live OpenAI probe: `POST /v1/responses` with `{type:"function_call_output", call_id, output, name}` errored only on the (fake) `call_id` — *"No tool call found for call_id call_probe123"* — **not** on the extra `name` field. OpenAI tolerates/ignores `name`. So the neutral item is the OpenAI shape + a field OpenAI ignores (exactly the `thought_signature` pattern). No distinct neutral type, no change to OpenAI's encoder.
2. **The Gemini encoder translates input items; OpenAI's is unchanged.** Only `encode_agentic(::GEMINIServiceEndpoint)` gains input-item translation (it already diverges from OpenAI wire); OpenAI stays pure pass-through.
3. **Public helper: `tool_result(call_id, name, output)`** (neutral verb; produces the `function_call_output`-typed item — consumers never see the type).
4. **Scope = tool-result item + `tool_choice` only** (Plan 3a). Background/`get_response`/`cancel_response`, usage→cost/pricing, and hosted tools are **Plan 3b** (independent, deferred).

## Design

### A. Neutral multi-turn tool-result item

- **`tool_loop(r::Respond)`** (`src/tool_loop.jl:237`): add `"name" => call["name"]` to the result dict. `call["name"]` is present for both providers (from `function_calls`). The loop is now provider-agnostic — turn 1 sends the string prompt; turn 2+ sends the (translated) results + continuation id.
- **`tool_result(call_id, name, output)`** (new, `src/responses.jl` near the other item constructors): returns `Dict{String,Any}("type"=>"function_call_output", "call_id"=>call_id, "name"=>name, "output"=>output)`. Exported.
- **OpenAI encoder:** unchanged. `JSON.lower(::Respond)` serializes `input` verbatim; OpenAI ignores the extra `name`.
- **Gemini encoder** (`encode_agentic(::GEMINIServiceEndpoint)`): replace `:input => r.input` with `:input => _interactions_input(r.input)`:
  - `_interactions_input(s::AbstractString) = s` (pass through).
  - `_interactions_input(v::AbstractVector)` — map each item: an `AbstractDict` with `type == "function_call_output"` → `{type:"function_result", call_id, name, result}`, where `result = _gemini_tool_response(item["output"])` (reuses `src/gemini.jl`'s wrap-string-as-`{"result":…}`-else-JSON-object helper). Every other item passes through unchanged.
  - **Fail loud** if a `function_call_output` item lacks `name` (Gemini requires it): `ArgumentError("Gemini function_result requires a name; build it with tool_result(call_id, name, output)")`.

### B. `tool_choice` for Gemini

Remove the fail-loud throw in `encode_agentic(::GEMINIServiceEndpoint)`. Map the neutral `tool_choice` → the Interactions shape (**exact wire confirmed at the Phase-0 gate**; research: `tool_choice.allowed_tools.mode`, lowercase):

| neutral (`Respond.tool_choice`) | Gemini Interactions (to confirm) |
|---|---|
| `"auto"` | `{allowed_tools: {mode: "auto"}}` |
| `"none"` | `{allowed_tools: {mode: "none"}}` |
| `"required"` | `{allowed_tools: {mode: "any"}}` |
| `tool_choice_function(name)` → `{type:"function", name}` | `{allowed_tools: {mode: "any", tools: [{name}]}}` |
| `tool_choice_hosted/_mcp/_custom` (OpenAI hosted-tool selectors) | **fail loud** — not applicable to Gemini function tools (hosted tools are Plan 3b) |

OpenAI's `tool_choice` is unchanged (its own wire).

## Phase 0 — live-wire gates (before coding)

Falsification-first, billing-enabled `GEMINI_API_KEY` (`source ~/.zshrc`; small spend; capture once):

1. **`tool_choice` wire.** `curl` an Interactions request with a `tool_choice` and confirm the exact shape/casing (`allowed_tools.mode`? enum values `auto`/`none`/`any`?). The mapping table above is provisional until this runs.
2. **Continuation-with-tools tolerance (a real risk).** `_next_respond` (`src/tool_loop.jl:188`) copies **all** fields, so turn 2+ re-sends `tools` alongside `previous_interaction_id` — but the Plan-2 capture's continuation *omitted* tools. `curl` a continuation that includes both `tools` and `previous_interaction_id` + a `function_result`. If Gemini rejects it, the Gemini encoder must **drop `tools` (and `tool_choice`) when `previous_interaction_id` is set** (server already holds the tool declarations). Pin this before building the loop.

## Phase 1 — neutral tool-result item (correctness fix)

`tool_loop.jl` name-field addition · `tool_result` helper + export · `_interactions_input` translation in `interactions.jl` · (conditional on Phase-0.2) drop `tools`/`tool_choice` on Gemini continuations.

**Falsifier (RED-first):** golden test — `encode_agentic(GEMINI, Respond(previous_response_id="p", input=[tool_result("c","get_weather","sunny")]))` → body `input[1] == {type:"function_result", call_id:"c", name:"get_weather", result:{"result":"sunny"}}` and `previous_interaction_id == "p"`. Mutation: drop the `name` add in `tool_loop` → a test asserting the item carries `name` fails. OpenAI unaffected (existing `tool_loop`/responses suite green).

## Phase 2 — `tool_choice` for Gemini (additive)

Remove the throw; add the mapping (per Phase-0.1) + fail-loud for hosted-tool selectors.

**Falsifier:** golden bodies for `"auto"`/`"none"`/`"required"` + `tool_choice_function`; `@test_throws` for `tool_choice_hosted` with Gemini; the removed-throw path replaced by real encoding.

## Testing — zero-spend first, then one key-gated live witness

Zero-spend rule (⚠️ OpenAI+Anthropic keys LIVE in sandbox): `env -u OPENAI_API_KEY -u DEEPSEEK_API_KEY -u ANTHROPIC_API_KEY -u GEMINI_API_KEY`. Both HTTP majors.

- **Deterministic, `test/interactions.jl`:** the tool-result translation golden (+ String passthrough, + non-tool-result passthrough, + missing-`name` fail-loud); `tool_choice` mapping goldens + hosted-selector throw. **`test/tool_loop.jl`:** the `function_call_output` item now carries `name` (via a mock/canned round-trip).
- **Live witness (key-gated `GEMINI_API_KEY`, cheapest model), `test/integration_interactions.jl`:** a **full multi-turn tool loop via `tool_loop`** — model calls the tool, `tool_loop` submits the neutral result, the follow-up completes — on both HTTP majors. Green mocks alone don't suffice (the whole bug was that mocks matched a wrong wire shape).
- **Coverage:** hold the project's ~99.5% bar (cover the new translation branches + fail-loud paths).

## Scope

**IN:** neutral `tool_result` item + `tool_loop` name-field fix + Gemini input-item translation · Gemini `tool_choice` mapping · (conditional) drop tools on continuation · unit tests + live tool-loop witness.

**OUT (Plan 3b):** `background`/`get_response`/`cancel_response` generalization · usage→`TokenUsage` + pricing · Gemini hosted tools (google_search/code_execution/…) · neutral translation of non-tool-result `input` items (e.g. `InputMessage`/`text`) for Gemini — the stateful loop doesn't resend those, so out of scope here.

## Assumption ledger — verify against the LIVE Interactions API

- exact `tool_choice` shape + casing + enum values (`allowed_tools.mode`: `auto`/`none`/`any`? or `required`?).
- whether a continuation may carry `tools`/`tool_choice` alongside `previous_interaction_id`, or they must be dropped.
- `function_result.result` must be a JSON **object** (captured: yes — `_gemini_tool_response` already wraps a bare string as `{"result": …}`).
- whether `result` accepts arbitrary nesting or only flat objects (decoder wrap is defensive regardless).

## Falsification — what would make us retract this design

- OpenAI starts rejecting the extra `name` on `function_call_output` → the `thought_signature`-style neutral item is invalid; fall back to a distinct neutral type + OpenAI-side translation. (Probe on 2026-07-08 says it's tolerated.)
- A live `tool_loop` round-trip on Gemini cannot complete through the neutral `tool_result` item → the translation is wrong regardless of green mocks.
- Gemini rejects a continuation that carries `tools` and the encoder's tool-drop breaks OpenAI (whose continuations *do* resend tools) → the drop must be Gemini-only (it is — the change lives in the Gemini encoder).
