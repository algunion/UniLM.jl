# Design: One neutral agentic verb — OpenAI Responses ⇄ Gemini Interactions

- **Date:** 2026-07-07
- **Status:** Approved (brainstorming) → ready for implementation plan
- **Keystone:** Layer-B — a single opinionated agentic verb (`respond()`) that multiple-dispatches to OpenAI *Responses* and Gemini *Interactions* over one neutral IR.
- **Approach:** promote the **existing** `Respond`/`ResponseObject` (already `service`-aware) to the neutral agentic IR, add a **dispatched agentic seam** mirroring the proven chat seam, and route **both** providers through it in one effort. Neutrality is tested against two live witnesses, not asserted.

## Problem (one sentence)

UniLM's agentic verb (`respond()`) is hardwired to OpenAI — it builds `RESPONSES_PATH`, serializes via a single `JSON.lower(::Respond)`, and parses an OpenAI-shaped `output[]` (`src/responses.jl:1122,1132,960`) — so it cannot reach Gemini's stateful **Interactions** surface, and the `service` field on `Respond` (`src/responses.jl:657`) today steers only URL/auth/model, never wire translation.

## Context — chat rides a dispatched seam; the agentic verb does not

The chat verb unified across OpenAI/Anthropic/Gemini by translating a **neutral IR** (`Chat`/`Message`) through three generics dispatched on `service`, untyped-default = OpenAI wire, providers override; `chatrequest!`/`_chatrequeststream` keep all HTTP/retry/cost/stream orchestration provider-agnostic (`src/requests.jl:252-356`):

| generic (chat) | default (untyped `service`) | returns |
|---|---|---|
| `encode_request(service, chat::Chat)` | `JSON.json(chat)` (`src/requests.jl:257`) | `String` |
| `decode_response(service, resp)` | `extract_message(resp)` (`src/requests.jl:265`) | `(; message::Message, usage)` |
| `decode_stream_chunk(service, chunk, state, failbuff)` | `_parse_chunk` (`src/requests.jl:272`) | `(; eos::Bool)` |

The agentic verb never adopted this. `respond(r::Respond)` (`src/responses.jl:1122`) posts to `_api_base_url(r.service) * RESPONSES_PATH`, and `_api_base_url` **throws** for every non-OpenAI service (`src/requests.jl:35-36`, `src/gemini.jl:19-20`). So the whole change is: *give the agentic verb the same seam chat already has, then add Gemini as a second dispatch target.*

## Decisions locked in brainstorming

1. **Neutral IR (portable), not provider surfaces.** One agentic struct + verb; encode/decode dispatched per provider; consumer swaps `service=`. (Rejected: parallel provider structs; hybrid neutral-core-plus-escape-hatches.)
2. **Both providers through the IR now.** OpenAI Responses is migrated onto the seam in the *same* effort as Gemini Interactions, so "neutral" is corroborated against two live surfaces — not a single-witness generalization (CLAUDE.md: *"Never generalize from a single passing case"*). Public `respond(input; kwargs...)` (`src/responses.jl:1184`) is preserved.
3. **Anthropic deferred.** Two witnesses validate neutrality; Anthropic has no stateful-agentic endpoint to fold in (as of research; recheck when it does). Forcing its stateless Messages API into an agentic-stateful verb would duplicate the chat verb.
4. **Naming: keep `respond()` / `Respond`** as the neutral verb/struct; `service=` selects the surface. Matches the "same verb, multiple dispatch" vision and the preserve-`respond()` requirement. (Rejected: new `interact()`/`agent()` primary + `respond()` alias.)
5. **Endpoint hosting: same `GEMINIServiceEndpoint`, dispatched by operation** — exactly as `OPENAIServiceEndpoint` hosts both `Chat` and `Respond`. `get_url` already dispatches on `(service, requesttype)` (`src/requests.jl:24-42`); Interactions is one more `(GEMINIServiceEndpoint, Respond)` method. No new endpoint type.

## The central move — OpenAI-shape-as-neutral, reused on both ends

The elegant reuse that makes this tractable (same trick chat plays with `Message`):

- **Neutral request IR = the existing `Respond`** (`src/responses.jl:656`). It already carries `service`. OpenAI's encoder stays `JSON.lower(::Respond)` **verbatim** (`src/responses.jl:710`) → byte-identical OpenAI wire, guarded by the existing `responses.jl` tests. Gemini's encoder is a new override producing Interactions snake_case wire.
- **Neutral response IR = the existing `ResponseObject` + every accessor** (`output_text`, `function_calls`, `reasoning_summaries`, `refusals`, …; `src/responses.jl:740,798,834,863…`). The Gemini decoder **normalizes Interactions `steps[]` into OpenAI-style `output[]` items**, so every accessor works for both providers with zero accessor changes.

**Pre-registered refutation (write the falsification first).** *The IR is not neutral if* encoding Interactions from `Respond` requires a raw escape-hatch dict instead of a clean semantic field, **or** decoding Interactions cannot populate the accessors without lossy hacks. When that happens the fix is a new **neutral, typed, defaulted** field (the way `GPTToolCall.thought_signature` was added for Gemini chat — `src/api.jl:85-94`), **never** a provider-specific dict. Both witnesses must pass the *same* accessor tests.

### The agentic seam (distinct generic names — a collision forces it)

Chat's `decode_response(service, resp::HTTP.Response)` and an agentic decode would have **identical argument types** (both `service, ::HTTP.Response`) but different return types (`Message` vs `ResponseObject`) — Julia cannot dispatch them apart. So the agentic verb gets its **own** trio, OpenAI-wire defaults extracted (moved, not rewritten) from today's `respond()`:

| generic (agentic) | OpenAI default (from current code) | Gemini override |
|---|---|---|
| `encode_agentic(service, r::Respond)` | `JSON.json(r)` = `JSON.lower(::Respond)` (`src/responses.jl:710`) | Interactions body |
| `decode_agentic(service, resp)` | `parse_response(resp)` (`src/responses.jl:960`) | `steps[]`→`output[]` |
| `decode_agentic_stream(service, chunk, state, failbuff)` | `_parse_response_stream_chunk` (`src/responses.jl:977`) | `interaction.*` SSE |

`respond`, `_respond_stream`, `get_response`, `cancel_response` call **only** these generics + `get_url(r.service, r)`; retry/HTTP/cost/streaming orchestration stays provider-agnostic — the same discipline `chatrequest!` follows.

## State model unification (open question resolved)

Neutralize server-state to three concepts the response threads back via its `.id`:

| neutral (field on `Respond`) | OpenAI Responses | Gemini Interactions |
|---|---|---|
| continuation handle | `previous_response_id` (`src/responses.jl:673`) | `previous_interaction_id` |
| `store` | `store` (`:671`) | `store` (default true) |
| `background` | `background` (`:675`) | `background` |

- The continuation handle is the **existing `previous_response_id` field** — the Gemini encoder maps it to `previous_interaction_id`. Minimal churn (no new field, no back-compat shim), at the cost of an OpenAI-flavored name serving a neutral role. *(Open, low-stakes: add a neutral `previous` alias later if the name grates; deferred, not blocking.)*
- OpenAI's richer `conversation=`/Conversations objects (`src/responses.jl:684`, `src/conversations.jl`) stay an **OpenAI-only optional field** — the Gemini encoder ignores it. Not forced into the neutral core.
- **Neutral status** on the decoded response: superset `{in_progress, completed, requires_action, incomplete, failed}` (raw kept in `ResponseObject.raw`). Gemini's `requires_action` ≈ OpenAI "a `function_call` item is present in `output`". The tool round-trip is shape-identical for both: run the tool, then `respond(previous=id, input=[tool results])`.
- **Background/async:** lift `get_response` (`src/responses.jl:1225`) and `cancel_response` (`:1319`) off their OpenAI-locked `_api_base_url` onto `get_url(service, …)` dispatch, so Gemini `background` polling/cancel flows through the same functions. The OpenAI-only Responses sub-endpoints (`compact_response`, `count_input_tokens`, `list_input_items`, `delete_response`) stay OpenAI-only, capability-gated — out of scope for Gemini.

## Response normalization (Gemini `steps[]` → neutral `output[]`)

The Gemini `decode_agentic` translates each Interactions step into an OpenAI-shaped output item so the reused accessors fire:

| Interactions step | neutral `output[]` item | accessor it feeds |
|---|---|---|
| `model_output` text | `{type:"message", content:[{type:"output_text", text}]}` | `output_text` (`:798`) |
| `function_call {call_id, arguments}` | `{type:"function_call", call_id, name, arguments:<JSON string>}` | `function_calls` (`:834`) |
| `thoughts` | `{type:"reasoning", summary:[{type:"summary_text", text}]}` | `reasoning_summaries` (`:863`) |
| `function_result` | `{type:"function_call_output", call_id, output}` (raw-preserved) | — |

- `arguments` is emitted as a **JSON string** because `function_calls` consumers `JSON.parse(call["arguments"])` (`src/responses.jl` docstring `:822-831`).
- `id`/`status`/`model`/`usage` map onto `ResponseObject`'s fields (`src/responses.jl:740-749`); the whole Interactions body is retained in `.raw` (nothing is discarded).
- **thoughtSignature is (assumed) NOT needed here.** The stateless-`generateContent` echo requirement (`src/api.jl:88-93`) exists because the client re-sends history; server-stateful Interactions holds thoughts itself. Flagged in the ledger to verify live — if wrong, it becomes a neutral field on the normalized function-call item, not a hack.

## Streaming (the known landmine)

`decode_agentic_stream` default = today's `response.*` parser (`src/responses.jl:977`); the Gemini override parses the Interactions SSE — **named events** `interaction.created` / `step.delta` / `interaction.completed`, snake_case `delta.text` — accumulating text + `output[]` items + `status` + `id` into a neutral agentic stream-state that assembles a `ResponseObject`, exactly as `_respond_stream` does today (`src/responses.jl:1029-1096`).

**Gate:** streaming diverges across HTTP majors — this exact class of bug hit Gemini *chat* and CI caught it (identity-encoding / `decompress=false`, `src/requests.jl:285-287`). Interactions streaming **must** be verified live on HTTP **1.x and 2.x** before "done".

## Phase 0 — reconfirm the wire live (gate, no code)

Falsification-first: `curl` the real Interactions endpoint for (text), (tool round-trip), (stream) and **capture** the actual request/response/SSE bytes. Every golden body and canned response in Phase 2 is built from *this capture*, not from the 2026-07-07 research summary (which is unverified against a live call). Blocks the Phase 2 encoder. `GEMINI_API_KEY` is billing-enabled → one small spend, captured once.

## Phase 1 — dispatch the agentic verb (behavior-preserving for OpenAI)

Introduce `get_url(service, ::Respond)` (`get_url(r::Respond) = get_url(r.service, r)`, mirroring `src/requests.jl:24`) with `get_url(::Type{OPENAIServiceEndpoint}, ::Respond) = OPENAI_BASE_URL * RESPONSES_PATH`, and the agentic seam trio with **OpenAI defaults = current logic moved verbatim**. Rewire `respond` / `_respond_stream` / `get_response` / `cancel_response` to call the seam + `get_url`. Add `validate_capability(service, :agentic, "respond()")` at the verb head. **No Gemini.**

**Falsifier (RED-first):** the existing `test/responses.jl` + `test/mock_server.jl` Responses coverage stays green — byte-identical OpenAI wire. Any diff means the refactor was not behavior-preserving. This phase carries the refactor risk alone, isolated and independently revertable.

## Phase 2 — native Gemini Interactions (additive)

Self-contained `src/interactions.jl` (mirroring `src/gemini.jl`'s structure), `include`d after `responses.jl`/`capabilities.jl`.

- **Constants:** `const INTERACTIONS_PATH = "/interactions"` on `GEMINI_NATIVE_BASE` (`src/constants.jl:123`) → `$GEMINI_NATIVE_BASE/interactions`.
- **Routing:** `get_url(::Type{GEMINIServiceEndpoint}, r::Respond)` (+ stream branch — verify whether Interactions streaming is a URL suffix or the body `stream` flag; research says body flag). Replaces the `_api_base_url(::GEMINIServiceEndpoint)` throw (`src/gemini.jl:19`). Auth reuses `auth_header(::Type{GEMINIServiceEndpoint})` (`x-goog-api-key`, `src/gemini.jl:22`).
- **`encode_agentic(::Type{GEMINIServiceEndpoint}, r)`** — `Respond` → Interactions snake_case body: `instructions`→`system_instruction`, `input`→`input` (translate OpenAI input items → Interactions items), `tools` (`FunctionTool`)→Interactions `function_call`-type tools, `tool_choice`→`tool_choice.allowed_tools.mode` (**lowercase**), `temperature`/`top_p`/`max_output_tokens`→`generation_config`, `previous_response_id`→`previous_interaction_id`, `store`/`background`/`stream` pass through. OpenAI-only fields (`reasoning`, `text`, `conversation`, hosted-tool `ResponseTool` subtypes) dropped by omission / deferred.
- **`decode_agentic(::Type{GEMINIServiceEndpoint}, resp)`** — the `steps[]`→`output[]` normalization above; neutral `status`; Interactions `usage`→`TokenUsage`.
- **`decode_agentic_stream(::Type{GEMINIServiceEndpoint}, …)`** — the `interaction.*` SSE parser above.
- **Capabilities:** add `:agentic` to `OPENAIServiceEndpoint` (`src/capabilities.jl:22`) and `GEMINIServiceEndpoint` (`src/gemini.jl:29`). Keep OpenAI's `:responses` for its OpenAI-only sub-endpoints.
- **Pricing:** Gemini agentic pricing keys on model-ID in `DEFAULT_PRICING` (`src/accounting.jl`) — provider-agnostic, so `_accumulate_cost!` works unchanged.

**Falsifier (RED-first):** golden Interactions bodies, canned `steps[]` decode → accessor assertions, canned `interaction.*` SSE, a **novel `status`** (open-enum, no crash), and the key-gated live witness (text + tool round-trip + stream) on both HTTP majors.

## Data flow (orchestration unchanged)

`Respond(service=GEMINIServiceEndpoint, input, tools, previous_response_id)` → `respond` → `validate_capability(:agentic)` → `encode_agentic(service, r)` → `HTTP.post(get_url(service, r), body, auth_header(service))` → shared retry on 408/429/5xx (`src/requests.jl:8`) → `decode_agentic(service, resp)` → neutral `ResponseObject` → `_accumulate_cost!` → `ResponseSuccess`. `output_text`/`function_calls` operate on the neutral `output[]`; the Interactions `steps[]` shape lives entirely inside the translator.

## Error handling

- **Neutral `status` is an OPEN enum** — unknown values degrade to raw pass-through, never an exception (same discipline as Gemini chat's open `finishReason`, `src/gemini.jl:146-152`).
- **Fail loud** on encode ambiguity: a tool-result input item whose call-id matches no prior `function_call` → `ArgumentError`, never a silently malformed `function_result` (mirrors `_gemini_contents`, `src/gemini.jl:80-81`).
- Non-2xx → `ResponseFailure` (shared path); Interactions error bodies captured raw.
- Capability gate: agentic `respond()` against a service lacking `:agentic` → clear `ArgumentError` (`src/capabilities.jl:41-45`).

## Testing — zero-spend first, then one key-gated live witness

Zero-spend rule (⚠️ OpenAI+Anthropic keys are LIVE in the sandbox; a bare `Pkg.test()` bills): `env -u OPENAI_API_KEY -u DEEPSEEK_API_KEY -u ANTHROPIC_API_KEY -u GEMINI_API_KEY`. Both HTTP majors in CI.

- **Phase 1 (deterministic):** existing Responses suite + Aqua green after the dispatch refactor.
- **Phase 2 (deterministic, golden/canned, RED-first), `test/interactions.jl`:** encode goldens (system_instruction, tools + lowercase tool_choice, `previous_interaction_id` mapping, `store`/`background`); decode canned `steps[]` → assert `output_text`/`function_calls`/`reasoning_summaries` + neutral `status` + `TokenUsage`; open-enum status (no crash); canned `interaction.*` SSE (text deltas, final `interaction.completed`, a streamed `function_call`); a **mutation** test (drop `previous_interaction_id` mapping → a test that asserts it is present catches the omission).
- **Live witness (key-gated `GEMINI_API_KEY`, `test/integration_interactions.jl`), cheapest durable model:** one text call + one stateful two-turn tool round-trip (`previous` chaining) + one stream, on both HTTP majors. `GEMINI_API_KEY` is **billing-enabled** and lives only in `~/.zshrc` (absent in sandbox) → run once when green, never rerun. Green mocks alone do not suffice — theory-laden mocks can encode the same wire mistake twice.

## Scope

**IN:** Phase 1 agentic-verb dispatch refactor (OpenAI behavior-preserving) · Phase 2 Gemini Interactions (function tools, `previous` state, `store`/`background`, streaming, usage/cost) · `steps[]`→`output[]` normalization reusing `ResponseObject` accessors · `:agentic` capability · `get_response`/`cancel_response` generalization · pricing rows · golden+canned mocks + key-gated witness.

**OUT (each its own later spec):** Anthropic agentic surface · Gemini *hosted* tools (google_search / code_execution) beyond function tools · OpenAI-only Responses sub-endpoints for Gemini (`compact`/`count_input_tokens`/`list_input_items`) · neutral `previous` alias · Gemini `thinkingConfig`/structured-output in the agentic verb · the many other deferred native surfaces (working-memory §"Other deferred").

## Assumption ledger — verify against the LIVE Interactions API, not the research summary

- exact `POST` path (`/v1beta/interactions`?) and whether streaming is a **body `stream` flag** or a **URL variant** (chat's `generateContent` used a URL method — do not assume Interactions matches).
- exact Interactions **`input` item shapes** for multi-turn + tool results (`function_result` fields, `call_id` correlation).
- **`usage` field names/inclusion semantics** (input/output/thoughts; is thoughts included in output? — decides add-vs-record, wrong choice double-counts cost; same trap as Gemini chat, `src/gemini.jl:154-168`).
- whether server-stateful Interactions still requires a **`thoughtSignature`-equivalent** echo (assumed NO; if YES it becomes a neutral field on the normalized call item).
- exact SSE **event names** and terminal event (`interaction.completed` vs an `[DONE]`-style sentinel).
- `store` default (research: true; 55d paid / 1d free) and whether `previous_interaction_id` requires `store=true` (as OpenAI `previous_response_id` effectively does).
- cheapest durable witness model-ID still GA; Interactions pricing rows for `DEFAULT_PRICING`.

## Falsification — what would make us retract this design

- Phase 1: the existing Responses suite or Aqua fails after the dispatch refactor → it was not behavior-preserving.
- A common Interactions text+tool multi-turn cannot round-trip through `Respond` + `ResponseObject` accessors without a raw escape-hatch dict → the neutral-IR claim is wrong; the IR must gain a **neutral** field (or, at the limit, the "neutral IR" decision itself is refuted and we fall back to provider surfaces + a shared verb).
- The Gemini decoder cannot fill `output_text`/`function_calls` from `steps[]` without lossy hacks → the OpenAI-shape-as-neutral reuse does not hold for the agentic surface.
- The open-enum `status` decode crashes on an unknown value → the defensive decode is not defensive.
- The live witness cannot complete a text call, a stateful tool round-trip, or a stream on either HTTP major → the translator is wrong regardless of green mocks.
