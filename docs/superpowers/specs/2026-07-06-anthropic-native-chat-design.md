# Design: Native Anthropic chat via a dispatched translation seam

- **Date:** 2026-07-06
- **Status:** Approved (brainstorming) → ready for implementation plan
- **Keystone:** Anthropic native chat (messages + tools + streaming + usage/cost)
- **Approach:** A — dispatched translation seam + neutral IR (chosen over `Chat{P}` parametric types and over per-provider request structs)

## Problem (one sentence)

UniLM's request-encoding and response-decoding are single hardcoded OpenAI-wire methods, so a provider with a different wire format (Anthropic, native Gemini) cannot be added without either forking the request path into parallel tracks or branching OpenAI-vs-not inside the orchestration.

## Why this is the real problem (current architecture)

The codebase **already** dispatches four things per `ServiceEndpoint` subtype:

- `get_url(service, request)` — URL routing (`src/requests.jl:24-54`)
- `auth_header(service)` — auth (`src/requests.jl:44-81`)
- `provider_capabilities(service)` — feature gating (`src/capabilities.jl:22-26`)
- `default_model` / `default_embedding_model` / … (`src/capabilities.jl:55-75`)

But three things are **not** dispatched — they are single OpenAI-only methods:

- `JSON.lower(chat::Chat)` (`src/api.jl:530`) — emits OpenAI's `{model, messages, tools, …}` body
- `extract_message(resp)` (`src/requests.jl:122`) — parses OpenAI's `choices[].message`
- `_parse_chunk(...)` (`src/requests.jl:163`) — parses OpenAI's SSE `choices[].delta`

DeepSeek, the Gemini OpenAI-compat shim (`GEMINIServiceEndpoint`, `src/constants.jl:114`), and `GenericOpenAIEndpoint` all work **precisely because they speak OpenAI's wire format**. Anthropic's Messages API and native Gemini do not (content blocks; `system` as a top-level field; `user`/`model` roles; `max_tokens` required; model-in-URL). The missing dispatch axis is the **wire-translation seam**.

## The seam — three generic functions dispatched on `service`

| generic fn | replaces (today) | returns |
|---|---|---|
| `encode_request(service, chat)` | `JSON.json(chat)` via `JSON.lower(::Chat)` | request body (`String`) |
| `decode_response(service, resp)` | `extract_message(resp)` | `(; message::Message, usage::Union{TokenUsage,Nothing})` |
| `decode_stream_chunk(service, chunk, state)` | `_parse_chunk` | mutates `StreamState` |

`chatrequest!` and `_chatrequeststream` (`src/requests.jl:241,329`) call these generics instead of the OpenAI functions. **Everything else in the orchestration stays shared and untouched:** retry/backoff (`_is_retryable`, `_retry_delay`), HTTP, `_accumulate_cost!`, `tool_loop!`, `fork`, streaming callbacks, the user close-`Ref`.

This is additive: OpenAI's current code *becomes* its own methods by moving, not rewriting.

## Phase 1 — pure refactor, zero behavior change (ships first, independently revertable)

Introduce the three generics; OpenAI-family endpoints delegate to the existing code:

```julia
encode_request(::Union{Type{OPENAIServiceEndpoint},Type{AZUREServiceEndpoint},Type{GEMINIServiceEndpoint},
               GenericOpenAIEndpoint,DeepSeekEndpoint}, chat::Chat) = JSON.json(chat)
decode_response(<same set>, resp) = extract_message(resp)
decode_stream_chunk(<same set>, chunk, state) = _parse_chunk(chunk, state, ...)
```

(Exact grouping — a shared abstract marker `OpenAIWire` vs. a `Union` — is an implementation detail for the plan; the requirement is that DeepSeek / Azure / Gemini-shim / Generic reuse one implementation, not copies.)

**Falsifier (RED-first):**
1. A test asserting `encode_request(OPENAIServiceEndpoint, chat)` is **byte-identical** to legacy `JSON.json(chat)` across representative chats (with tools, streaming params, `response_format`, `logit_bias`, etc.). Written to pass — corroborates the move is lossless.
2. The full existing zero-spend suite (2815 tests as of 0.10.3) stays green.

If either fails, the refactor changed behavior and must be corrected before Phase 2.

## Phase 2 — native Anthropic (additive)

New endpoint + constants:

- `struct ANTHROPICServiceEndpoint <: ServiceEndpoint end`
- `const ANTHROPIC_API_KEY = "ANTHROPIC_API_KEY"`, `const ANTHROPIC_BASE_URL = "https://api.anthropic.com"`, `const ANTHROPIC_MESSAGES_PATH = "/v1/messages"`
- `get_url(::Type{ANTHROPICServiceEndpoint}, ::Chat)` → base + messages path
- `auth_header(::Type{ANTHROPICServiceEndpoint})` → `x-api-key: ENV[ANTHROPIC_API_KEY]`, `anthropic-version: <verified>`, `content-type: application/json`

**`encode_request(::Type{ANTHROPICServiceEndpoint}, chat)`**

- First `role == system` message → top-level `system` string (Anthropic has no system *message*).
- Roles map assistant→assistant, user→user.
- Assistant message with `tool_calls` → content array `[{type:"text",text}?, {type:"tool_use", id, name, input}]` (input is the parsed args dict).
- Consecutive `role == "tool"` messages **collapse** into one `user` message with `[{type:"tool_result", tool_use_id, content}]` blocks (Anthropic represents tool results as a user turn).
- `max_tokens` is **required** by Anthropic → filled from `default_max_tokens(service, model)` when the user left it `nothing`.
- `stop` → `stop_sequences`; `temperature`/`top_p`/`stream`/`metadata` pass through; `tools` → `[{name, description, input_schema}]`; `tool_choice` mapped to Anthropic's `{type:"auto"|"any"|"tool", name?}`.
- OpenAI-only params with no Anthropic equivalent (`logit_bias`, `n`, `presence_penalty`, `frequency_penalty`, `seed`, `response_format`) are dropped by omission (not error), consistent with capability gating.

**`decode_response(::Type{ANTHROPICServiceEndpoint}, resp)`**

- Parse `content[]`: `text` blocks concatenated → `Message.content`; `tool_use` blocks → `Message.tool_calls::Vector{GPTToolCall}` (id, name, input).
- `stop_reason` → `finish_reason`: `end_turn`/`stop_sequence` → `"stop"`, `tool_use` → `"tool_calls"`, `max_tokens` → `"length"`.
- `usage.input_tokens` → `prompt_tokens`, `usage.output_tokens` → `completion_tokens`, `usage.cache_read_input_tokens` → `cached_tokens`. Total computed if absent.

**`decode_stream_chunk(::Type{ANTHROPICServiceEndpoint}, chunk, state)`**

- Anthropic SSE events → `StreamState`: `message_start` (seed usage), `content_block_start` (block type + index), `content_block_delta` (`text_delta` → append content; `input_json_delta` → append tool args for that block index), `content_block_stop`, `message_delta` (`stop_reason` + output usage), `message_stop` (EOS).
- `StreamState` may gain 1–2 provider-neutral fields (e.g. current block index → tool-call slot) so `_build_stream_message` (already neutral) reuses unchanged.

**Capabilities / defaults / pricing**

- `provider_capabilities(::Type{ANTHROPICServiceEndpoint}) = Set([:chat, :tools, :json_output, :streaming])`
- `default_model(::Type{ANTHROPICServiceEndpoint}) = <verified current Claude default>`
- `default_max_tokens(service, model)` — new small dispatch supplying a **sane, documented, overridable** default when the user leaves `max_tokens` unset (Anthropic requires the field). Bias toward a moderate value the caller can raise, **not** the model's ceiling — implementation to pick the number and document it; unused headroom is not billed, but an implicit ceiling-sized cap invites runaway output.
- Anthropic pricing rows added to `DEFAULT_PRICING` so `estimated_cost` / `_accumulate_cost!` work with no change to accounting logic.

## Why the neutral IR needs NO type change for the keystone

Today's `Message` (`src/api.jl:227`) already holds `content` **and** `tool_calls` at once — its validation requires only *one* of `content`/`tool_calls`/`refusal_message` to be non-`nothing`. So Anthropic's "text + tool_use" response decodes cleanly into the flat `Message`, and Anthropic's block structure is **assembled in `encode_request` and disassembled in `decode_response`** — it lives inside the translator, never in the user-facing type. This is the smallest reversible change.

**Deferred enrichments (each named with the concrete failure it prevents — own later specs):**

- Multimodal content blocks — *prevents:* cannot send images/PDFs to Claude.
- Thinking-block preservation — *prevents:* losing or mis-ordering extended-thinking content across multi-turn tool use.
- `cache_control` on blocks — *prevents:* paying full input price on repeated large context.

## Data flow (orchestration unchanged)

`Chat(service=ANTHROPICServiceEndpoint, model, messages, tools)` → `chatrequest!` → `encode_request` → `HTTP.post(get_url, body, auth_header)` → shared retry on 408/429/5xx/529 → `decode_response` → neutral `Message` appended via shared `update!` → `_accumulate_cost!` → `LLMSuccess`. `tool_loop!` and `fork` operate on the neutral `Message` and need no Anthropic-specific code.

## Error handling

- Non-2xx → `LLMFailure` (shared path). Anthropic error bodies (`{type:"error", error:{type, message}}`) are captured raw in `LLMFailure.response`, same as today's OpenAI errors.
- Capability gate: `validate_capability` stays at request entry. Attempting embeddings/responses/images against `ANTHROPICServiceEndpoint` → clear `ArgumentError` listing supported caps.
- **Fail loud** on encode ambiguities: a `role == "tool"` message with no matching preceding `tool_use` id → `ArgumentError`, never a silently malformed body.

## Testing — zero-spend first, then one key-gated live witness

Matches repo convention (`test/mock_server.jl`, `test/integration_deepseek.jl`) and the zero-spend rule (`env -u OPENAI_API_KEY -u DEEPSEEK_API_KEY -u ANTHROPIC_API_KEY`).

**Phase 1 (deterministic):** byte-equality test (above) + existing suite green.

**Phase 2 (deterministic, mock/golden, RED-first):**
- *encode:* golden Anthropic bodies for (system-split), (tools), (multi-turn `tool_use`→`tool_result` collapse), (`max_tokens` defaulting), (`stop`→`stop_sequences`).
- *decode:* canned Anthropic responses (text; text+tool_use; `max_tokens` stop; error body) → assert neutral `Message`, `finish_reason` mapping, `TokenUsage`.
- *stream:* a canned Anthropic SSE event sequence → assert accumulated `Message` + usage; extend `mock_server.jl` with an Anthropic route.

**Live witness (key-gated `ANTHROPIC_API_KEY`, new `test/integration_anthropic.jl`):** one text call + one tool-call round-trip + one stream. Runs in CI across both HTTP majors; not rerun locally once green. This is the end-to-end observation required before claiming completion — green units alone do not suffice.

## Scope

**IN:** Phase 1 seam refactor + Phase 2 Anthropic chat / tools / streaming / usage / cost, capabilities/defaults, mocks + witness, Anthropic pricing rows.

**OUT (each its own later spec):** native Gemini (+ its stateful/Interactions surface), embeddings, images, Anthropic Batches / Files / prompt-caching / citations / PDF, multimodal & thinking-block IR enrichment, the `GPT*`→neutral-name rename, docs overhaul.

## Assumption ledger — verify in implementation, not from memory

Source of truth: the `claude-api` skill (canonical, current) + live Anthropic docs. To confirm before/while coding:

- current `anthropic-version` header value
- default Claude model ID(s) + per-model `max_tokens` ceilings
- exact SSE event/field names (`content_block_delta`, `input_json_delta`, `message_delta` usage shape)
- `tool_choice` object shapes (`auto` / `any` / `tool`)
- error-body shape
- Anthropic pricing numbers for `DEFAULT_PRICING`

The design assumes only the **stable** Messages API core (POST `/v1/messages`; `x-api-key` + `anthropic-version`; `system` top-level; content blocks; `tool_use`/`tool_result`; `usage.input_tokens`/`output_tokens`; `content_block` SSE deltas) — unchanged since 2024.

## Falsification — what would make us retract this design

- Phase 1 byte-equality or the existing suite fails → the seam is not behavior-preserving.
- A common Anthropic chat+tool multi-turn cannot round-trip through the flat `Message` without loss → the "no IR change" claim is wrong and the IR-enrichment must move from *deferred* into *keystone*.
- The live witness cannot complete a text call, a tool round-trip, or a stream → the translator is wrong regardless of green mocks (theory-laden mocks can encode the same mistake twice).
