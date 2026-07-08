# Design: Complete the Gemini agentic verb (Plan 3b)

- **Date:** 2026-07-08
- **Status:** Approved (brainstorming) → ready for implementation plan
- **Keystone:** finish the remaining Gemini agentic surface so the unified `respond()` verb is release-ready — **cost accounting**, the **background/get/cancel lifecycle**, and **native hosted tools** — extending the same neutral-IR seam the Gemini decoder already uses.
- **Approach:** the Gemini decoder already normalizes `steps[]`→OpenAI-shaped `output[]`; extend that seam so usage, lifecycle responses, and hosted-tool results all emerge OpenAI-shaped. Downstream code (accessors, `token_usage`, `estimated_cost`) is unchanged. OpenAI paths untouched. Three independent, risk-ordered phases.

## Problem (one sentence per phase)

- **A (usage→cost):** `token_usage(::ResponseSuccess)` (`src/accounting.jl:43-48`) reads OpenAI keys `input_tokens`/`output_tokens`, but a Gemini interaction's `usage` uses `total_input_tokens`/`total_output_tokens`/`total_thought_tokens`/`total_cached_tokens`, so **every Gemini interaction reports zero usage → ~zero `estimated_cost`**.
- **B (lifecycle):** `get_response`/`cancel_response`/`delete_response`/`list_input_items` (`src/responses.jl:1273-1367`) hardcode `RESPONSES_PATH` + `parse_response`, so a Gemini interaction (incl. one created with `background=true`, which already serializes) **cannot be retrieved or cancelled** — the call hits the wrong URL and OpenAI decoder.
- **C (hosted tools):** `_interactions_tool` (`src/interactions.jl:42`) handles only function/dict/CallableTool and `_interaction_output` skips hosted-tool steps (`src/interactions.jl:113`), so Gemini's **google_search / code_execution / url_context are unusable** through the verb.

## Phase 0 — live-wire captures (RAN 2026-07-08, gemini-3.1-flash-lite; results folded in)

**A · usage dict (RESOLVED).** A completed interaction returns:
```json
{"total_input_tokens":12,"total_output_tokens":69,"total_thought_tokens":0,
 "total_cached_tokens":0,"total_tokens":81,"total_tool_use_tokens":0,
 "input_tokens_by_modality":[{"modality":"text","tokens":12}]}
```
The identity `total_tokens = input + output + thought` (81 = 12+69+0) confirms **`total_output_tokens` excludes thought**. Billing math (Gemini bills thought + tool-use at the output rate):
- `input_tokens` ← `total_input_tokens` · `cached_tokens` ← `total_cached_tokens` (cached ⊆ input; billed at the discounted rate)
- `output_tokens` ← `total_output_tokens + total_thought_tokens + total_tool_use_tokens`

**B · lifecycle (RESOLVED).** `GET /v1beta/interactions/{id}` → **200**, full interaction object (keys `created/id/model/object/service_tier/status/steps/updated/usage` — same shape as create, so `decode_agentic` decodes it unchanged). `background:true` create → **200 `in_progress` + id immediately** (async create→poll works). Cancel: Google-style `/{id}:cancel` → **404**, OpenAI-style **`POST /{id}/cancel` → 200** — the `/{id}` and `/{id}/cancel` subpaths match OpenAI's exactly; only the base path (`INTERACTIONS_PATH` vs `RESPONSES_PATH`) and decoder differ.

**C · hosted-tool wire (RESOLVED).** Declaration is the flat `{"type":"<name>"}` (same convention as function tools); the wrapped `{"google_search":{}}` form → **400 "The 'type' parameter is required"**. Confirmed 200 for all three; each emits a **`<name>_call` + `<name>_result` step pair** before `thought`/`model_output`:
- `{"type":"google_search"}` → `google_search_call {arguments:{queries:[…]}, id, search_type, signature}` + `google_search_result`
- `{"type":"code_execution"}` → `code_execution_call {arguments:{code, language}}` + `code_execution_result`
- `{"type":"url_context"}` → `url_context_call {arguments:{urls:[…]}, id, signature}` + `url_context_result`

## Phase A — usage→cost (smallest, near-deterministic)

**Normalize usage at decode.** In `src/interactions.jl`, where the interaction dict is built for the neutral `ResponseObject` (`_interaction_response_dict` / decode path), rewrite the raw Gemini `usage` into an OpenAI-shaped dict so the existing `token_usage(::ResponseSuccess)` path works unchanged:
```
input_tokens          = total_input_tokens
output_tokens         = total_output_tokens + total_thought_tokens + total_tool_use_tokens
input_tokens_details  = { cached_tokens = total_cached_tokens }
total_tokens          = total_tokens        # preserved
```
Keep the raw Gemini keys too (additive; nothing else reads them, but they aid debugging). `DEFAULT_PRICING` already carries the Gemini rows; add any model a witness uses that is missing, verified against the live pricing page. `estimated_cost`'s existing cached logic (`cached = min(cached, prompt)`) then applies correctly.

**Falsifier:** decode golden — a canned interaction dict → normalized `usage` has `output_tokens == total_output_tokens + total_thought_tokens + total_tool_use_tokens`. Live — `estimated_cost(gemini_result) > 0` and equals hand-computed `input×in_rate + output×out_rate`.

## Phase B — lifecycle for Gemini (get / cancel; background poll)

**Dispatch the base path; reuse the decoder.** Add `_agentic_base_path(service)` → `RESPONSES_PATH` (default/OpenAI) / `INTERACTIONS_PATH` (Gemini). In `get_response`/`cancel_response` (and, uniformly, `delete_response`/`list_input_items`), build the URL from `_agentic_base_path(service)` instead of the literal `RESPONSES_PATH`, and replace `parse_response(resp)` with `decode_agentic(service, resp)` (default = `parse_response`, so OpenAI is behavior-preserving; Gemini gets the interaction decoder). Subpaths (`/{id}`, `/{id}/cancel`) are identical across providers (captured).

`get_response` + `cancel_response` are the confirmed core (async flow: `respond(background=true)` → poll `get_response` → optional `cancel_response`). `delete_response`/`list_input_items` route to the correct base path for free; Gemini support is unverified, so an unsupported op returns `ResponseFailure(status=404)` — honest, not a silent wrong-endpoint. (No new poll helper — callers loop `get_response` until `status=="completed"`; a helper is Plan-3c YAGNI.)

**Falsifier:** live — `respond(background=true)` on Gemini returns `in_progress` + id; poll `get_response` to `completed` and read non-empty `output_text`; `cancel_response` on an in-flight interaction returns a decoded object (200). Zero-spend — `get_response`/`cancel_response` against the mock at `INTERACTIONS_PATH` decode via the Gemini seam.

## Phase C — hosted tools (Gemini native set)

**Encode:** three provider-native constructors (naming decision locked with the user) returning the flat declaration dict:
```
gemini_google_search()  = Dict("type" => "google_search")
gemini_code_execution() = Dict("type" => "code_execution")
gemini_url_context()    = Dict("type" => "url_context")
```
`_interactions_tool` already passes an `AbstractDict` through unchanged, so no encoder change is needed — the constructors just produce the captured shape. Exported.

**Decode:** `_interaction_output` currently skips hosted-tool steps; instead **surface** each `<name>_call` (and `<name>_result`) step as a neutral output item preserving its native type and `arguments`/`id`, so nothing is dropped and `output_text` (from `model_output`) is unaffected. Hosted-tool calls are *not* function calls, so `function_calls()` must continue to ignore them (they carry no `function_call` type).

**Falsifier:** encode goldens — each constructor emits `{"type":"<name>"}`. Live — a `gemini_google_search()`-grounded answer completes with non-empty `output_text` and the decoded `output[]` contains a `google_search_call` item. Zero-spend — a canned interaction with hosted-tool steps decodes into output items (not dropped) while `function_calls` stays empty.

## Testing — zero-spend first, then key-gated live witnesses

Zero-spend rule (⚠️ OpenAI+Anthropic keys LIVE in sandbox): `env -u OPENAI_API_KEY -u DEEPSEEK_API_KEY -u ANTHROPIC_API_KEY -u GEMINI_API_KEY`. Both HTTP majors; hold the ~99.5% coverage bar.

- **Deterministic** (`test/interactions.jl`, `test/accounting.jl` or `test/responses.jl`, `test/mock_server.jl`): usage-normalization golden + `estimated_cost>0`; hosted-tool encode goldens + hosted-step decode (surfaced, `function_calls` empty); `get_response`/`cancel_response` against a mock at `INTERACTIONS_PATH`.
- **Live witnesses** (`test/integration_interactions.jl`, GEMINI_API_KEY, cheapest model, both majors): (A) `estimated_cost>0` on a real interaction; (B) background create → poll `get_response` → cancel; (C) one `gemini_google_search` grounded round-trip.

## Scope

**IN:** usage-at-decode normalization + pricing check · `get_response`/`cancel_response` (+ uniform base-path for delete/list) via `_agentic_base_path` + `decode_agentic` · `gemini_google_search`/`gemini_code_execution`/`gemini_url_context` constructors + hosted-step decode · unit + live witnesses.

**OUT (post-release follow-ons):** a `poll_until_complete` helper · Gemini hosted tools beyond the core three (file_search/mcp/maps/…) · `file_search`/`mcp` neutral selectors · the other native surfaces (Live API WebSocket, native embeddings, Batch, context caching, multimodal, Anthropic Batches/Files/citations) — each its own spec. Anthropic agentic (Managed Agents) stays future Layer-C.

## Falsification — what would retract this design

- Gemini `total_output_tokens` actually *includes* thought (the `total = in+out+thought` identity says it does not) → the additive fold would double-count; revert to `output = total_output_tokens`.
- `GET /interactions/{id}` returns a shape `decode_agentic` can't parse → lifecycle needs a distinct decoder, not the create decoder (capture says identical).
- A hosted-tool `*_call` step surfaced into `output[]` is mistaken for a function call by `function_calls()` → the decode must gate on the `function_call` type only (guarded by the "function_calls empty" test).

## Assumption ledger (verify against LIVE Interactions API)

- ~~usage key names + thought relationship~~ **RESOLVED** (Phase 0-A).
- ~~GET/cancel/background wire~~ **RESOLVED** (Phase 0-B); `delete`/`list_input_items` Gemini support **unverified** (honest 404→ResponseFailure).
- ~~hosted-tool declaration + result-step shapes~~ **RESOLVED** (Phase 0-C).
- `total_input_tokens` includes `total_cached_tokens` (cached ⊆ input) — assumed (cached=0 in capture); `min(cached,prompt)` caps any error.
- **What `estimated_cost` cannot see:** it is token-based only. Hosted-tool *per-call* fees (e.g. google_search per-1k-queries) are **not** modeled; `total_tool_use_tokens` folded into output tokens is a token approximation, not the billed search fee. Document this on the hosted-tool constructors.
