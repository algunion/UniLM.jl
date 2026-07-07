# Design: Native Gemini chat via the dispatched translation seam

- **Date:** 2026-07-07
- **Status:** Approved (brainstorming) → ready for implementation plan
- **Keystone:** Gemini native `generateContent` chat (messages + tools + streaming + usage/cost)
- **Approach:** ride the **existing** dispatched translation seam (built by the Anthropic keystone, `5036dcc`); native Gemini = one more `service` dispatch target + exactly one minimal neutral-IR field. No parallel request tracks.

## Problem (one sentence)

UniLM speaks Gemini today only through an OpenAI-compatibility shim (`GEMINIServiceEndpoint`), so it cannot reach native-only Gemini behaviour (the current `generateContent` wire, Gemini-3 function-calling with `thoughtSignature`, native `usageMetadata`) and cannot host Gemini's stateful Interactions surface later.

## Context — the seam already exists; what is new for Gemini

The Anthropic keystone already introduced and proved the wire-translation seam — three generics dispatched on `service`, called by the shared orchestration:

| generic fn | default (untyped `service`, OpenAI-wire) | Gemini overrides? |
|---|---|---|
| `encode_request(service, chat)` | `JSON.json(chat)` (`src/requests.jl:256`) | **yes** |
| `decode_response(service, resp)` | `extract_message(resp)` (`src/requests.jl:264`) | **yes** |
| `decode_stream_chunk(service, chunk, state, failbuff)` | `_parse_chunk(...)` (`src/requests.jl:271`) | **yes** |

`chatrequest!` / `_chatrequeststream` (`src/requests.jl:362,274`) call these generics; retry/backoff, HTTP, `_accumulate_cost!`, `tool_loop!`, `fork`, streaming callbacks stay shared and untouched. Native Gemini is therefore a self-contained `src/gemini.jl` mirroring `src/anthropic.jl`, `include`d after `requests.jl`/`capabilities.jl` (so the `state::StreamState` annotation resolves at definition time — `src/requests.jl:248-249`).

**Three wrinkles Anthropic did not have:**

1. **Naming collision (breaking).** The natural name `GEMINIServiceEndpoint` is already the shipped OpenAI-compat shim (`src/api.jl:339`, exported `src/UniLM.jl:81`, in 0.10.3). Decision (brainstorming): the **native** endpoint takes `GEMINIServiceEndpoint`; the shim is renamed `GEMINIOpenAIServiceEndpoint`. First-class native API keeps the canonical name; the compat adapter carries the qualifier.
2. **Model-in-URL.** Native Gemini routes the model in the *path* (`…/models/{model}:generateContent`), not the body. In-repo precedent already exists: Azure interpolates `chat.model` in `get_url` (`src/requests.jl:28`). `get_url` receives the whole `chat`, so no new machinery.
3. **`thoughtSignature`.** Gemini-3 models think by default; in **stateless** `generateContent` the assistant's `functionCall` parts carry a `thoughtSignature` that must be echoed verbatim next turn or multi-turn tool-calling errors. This forces exactly one additive neutral-IR field (below).

## Decisions locked in brainstorming

- **Surface:** stateless `generateContent` (current, non-deprecated; Google labels it "legacy" but fully supported). Gemini's stateful **Interactions** API (GA 2026-06-30) is the *Layer-B* follow-on — the unified agentic verb over OpenAI Responses ⇄ Gemini Interactions ⇄ … — its own spec, hosted on this **same** `GEMINIServiceEndpoint` type, dispatched by operation (as `OPENAIServiceEndpoint` hosts both chat and `respond()`).
- **Naming:** native = `GEMINIServiceEndpoint`; shim → `GEMINIOpenAIServiceEndpoint` (breaking; `## Breaking changes` note at release).
- **Tools:** included, with the one `thought_signature` field → full multi-turn `tool_loop!`.
- **Default model:** `gemini-3.5-flash` (GA, durable). Live witness uses `gemini-3.1-flash-lite` (cheapest durable model). The `GEMINI_API_KEY` is **billing-enabled** → the witness spends real (small) money; run once when green, never rerun.

## Phase 1 — mechanical shim rename (behavior-preserving, ships first, independently revertable)

Rename every current `GEMINIServiceEndpoint` reference → `GEMINIOpenAIServiceEndpoint`, changing **no behaviour**:

- `src/api.jl:339` (struct def) + docstrings (`:327`, `:338`)
- `src/UniLM.jl:81` (export: replace the name)
- `src/requests.jl:29,32` (`get_url` for `Chat`/`Embeddings`), `:36` (`_api_base_url` throw), `:76-81` (`auth_header` Bearer)
- `src/capabilities.jl:24` (`provider_capabilities`), `:57` (`default_model`), `:63` (`default_embedding_model`)
- any `test/*` references to the shim

The shim has **no** seam overrides — it falls through the untyped OpenAI-wire defaults (`src/requests.jl:256/264/271`); rename does not touch the seam. **Bundled fix:** the shim's `default_model` is `gemini-2.5-flash`, which Google retires **2026-10-16**; bump to `gemini-3.5-flash` in this commit (the line is already being edited).

**Falsifier (RED-first):** the full existing zero-spend suite (2886 tests as of `5036dcc`) + Aqua stay green after the rename+bump. Any failure means the rename was not behaviour-preserving. This phase carries the breaking-change surface alone, isolated from the feature.

## Phase 2 — native Gemini `generateContent` (additive, `src/gemini.jl`)

**Endpoint + constants.** `struct GEMINIServiceEndpoint <: ServiceEndpoint end` (re-added, now native); `const GEMINI_NATIVE_BASE = "https://generativelanguage.googleapis.com/v1beta"`; reuse `GEMINI_API_KEY` (`src/constants.jl:8`).

**Routing / auth.**
- `get_url(::Type{GEMINIServiceEndpoint}, chat::Chat)` branches on `chat.stream` (it receives the whole `chat`): `false` → `"$GEMINI_NATIVE_BASE/models/$(chat.model):generateContent"`, `true` → `"…/models/$(chat.model):streamGenerateContent?alt=sse"`. Native Gemini expresses streaming in the URL **method**, not a body flag — so `encode_request` never emits `stream` in the body (contrast OpenAI/Anthropic).
- `auth_header(::Type{GEMINIServiceEndpoint})` → `x-goog-api-key: ENV[GEMINI_API_KEY]` + `content-type: application/json` (header form, **not** Bearer, **not** `?key=`).
- `_api_base_url(::Type{GEMINIServiceEndpoint})` throws (no Responses API on native Gemini; Interactions is the deferred surface).

**`encode_request(::Type{GEMINIServiceEndpoint}, chat)`** — neutral `Chat` → Gemini body:
- `role == "system"` message → top-level `systemInstruction: {parts:[{text}]}` (Gemini has no system role).
- `"user"` → `contents` entry role `"user"`, `parts:[{text}]`. `"assistant"` → role `"model"`, `parts:` text part (if any) + one `{functionCall:{id,name,args}}` per `tool_calls` entry, **plus that entry's `thought_signature`** on the part when present.
- `"tool"` result → role `"user"`, `parts:[{functionResponse:{id,name,response}}]`. `name` is recovered by correlating `tool_call_id` back to the emitting assistant `functionCall` (encoder-side, mirroring `_anthropic_messages`); `response` is the tool content parsed as a JSON object, else wrapped `{"result": <string>}` (Gemini requires an object).
- `tools` → `[{functionDeclarations:[{name,description,parameters}]}]`; `tool_choice` → `toolConfig.functionCallingConfig.mode` (`AUTO`/`ANY`/`NONE`, uppercase) + `allowedFunctionNames`.
- `generationConfig`: `temperature`/`topP`/`stopSequences` pass through **when set**. `maxOutputTokens` passes through only when the user set `chat.max_tokens`; **omitted otherwise** (unlike Anthropic, Gemini does not require it — and a low cap truncates Gemini-3 thinking before any answer). **No `default_max_tokens` override for Gemini.**
- OpenAI-only params with no Gemini equivalent are dropped by omission (consistent with capability gating). `thinkingConfig` / `responseSchema` are **not** emitted in the keystone (deferred).

**`decode_response(::Type{GEMINIServiceEndpoint}, resp)`** — `candidates[0]`:
- `content.parts`: `text` parts concatenated → `Message.content`; `functionCall` parts → `Message.tool_calls::Vector{GPTToolCall}` (`id`, `func`, and the part's `thoughtSignature` → `thought_signature`).
- `finishReason` → `finish_reason` as an **open enum** (below). Presence of `functionCall` parts forces `"tool_calls"`.
- `usageMetadata` → `TokenUsage`: `promptTokenCount`→`prompt_tokens`, `candidatesTokenCount`→`completion_tokens`, `totalTokenCount`→`total_tokens`, `cachedContentTokenCount`→`cached_tokens`. **Inclusion semantics must be verified** (ledger): if `promptTokenCount` already includes cached/thoughts, do **not** re-add (contrast Anthropic, which adds `cache_read` because its `input_tokens` excludes it).

**`decode_stream_chunk(::Type{GEMINIServiceEndpoint}, chunk, state, failbuff)`** — each SSE `data:` line is a full **partial** `GenerateContentResponse`:
- append `candidates[0].content.parts[].text` → `state.content`; accumulate `functionCall` parts into `state.tool_calls` (by index), stashing `thoughtSignature` under a `"thought_signature"` key.
- `usageMetadata` present → overwrite `state.usage` (final chunk authoritative); `finishReason` present → `state.finish_reason` + signal **EOS** (there is **no `[DONE]` sentinel**; the stream simply ends).
- `_build_stream_message` (`src/requests.jl:219`) stays shared, gaining **one optional line**: set `GPTToolCall.thought_signature` from the accumulated `"thought_signature"` key when present (no-op for providers that never set it). `StreamState` needs **no new fields**.

**Capabilities / defaults / pricing.**
- `provider_capabilities(::Type{GEMINIServiceEndpoint}) = Set([:chat, :tools, :streaming])` (declare only what the keystone implements; `:json_output` added when `responseSchema` lands).
- `default_model(::Type{GEMINIServiceEndpoint}) = "gemini-3.5-flash"`.
- Native Gemini pricing rows added to `DEFAULT_PRICING` (`src/accounting.jl`) so `estimated_cost`/`_accumulate_cost!` work unchanged (keys on model-ID, provider-agnostic).

## The one neutral-IR change — `GPTToolCall.thought_signature`

Add one optional, typed, defaulted field to `GPTToolCall` (`src/api.jl:80`):

```julia
thought_signature::Union{Nothing,String} = nothing  # Gemini-3 opaque function-calling
# signature; MUST be echoed verbatim next turn or stateless multi-turn tool calls 400.
# Set only by the Gemini decoder; ignored by every other provider.
```

- **Concrete failure it prevents:** Gemini-3 stateless multi-turn `tool_loop!` errors (HTTP 400) when the signature is stripped.
- **Per-tool-call granularity** also handles Gemini-3 **parallel** function calls (each carries its own signature) with no extra structure.
- **No OpenAI-wire leak:** `JSON.lower(::GPTToolCall)` (`src/api.jl:86`) emits only `{id,type,function}` and is left unchanged, so OpenAI/Azure/DeepSeek/shim serialization is byte-identical.
- **Positional-constructor audit (0.10.3 lesson):** `@kwdef` defaults do **not** extend positional constructors; `GPTToolCall` has no explicit positional ctor today. The plan must (a) confirm no positional `GPTToolCall(...)` construction exists, or (b) add an explicit old-arity positional ctor — mirroring `GPTFunctionSignature` (`src/api.jl:41`).

## Data flow (orchestration unchanged)

`Chat(service=GEMINIServiceEndpoint, model="gemini-3.5-flash", messages, tools)` → `chatrequest!` → `encode_request` → `HTTP.post(get_url, body, auth_header)` → shared retry on 408/429/5xx → `decode_response` → neutral `Message` → `_accumulate_cost!` → `LLMSuccess`. `tool_loop!` and `fork` operate on the neutral `Message`; the Gemini `functionCall`/`functionResponse` shape lives entirely inside the translator.

## Error handling

- **`finishReason` is an OPEN enum** — Google adds values unannounced (live crash reports on unknown ordinals). Map known values (`STOP`/absent → `"stop"`; `MAX_TOKENS` → `"length"`; `SAFETY`/`RECITATION`/`BLOCKLIST`/`PROHIBITED_CONTENT`/`SPII`/`IMAGE_SAFETY` → `"content_filter"`); **any unknown value → `"stop"`, never an exception**. Tool-call presence overrides to `"tool_calls"`.
- **Fail loud** on encode ambiguity: a `"tool"` message whose `tool_call_id` matches no preceding `functionCall` → `ArgumentError`, never a silently malformed `functionResponse`.
- Non-2xx → `LLMFailure` (shared path); Gemini error bodies (`{error:{code,message,status}}`) captured raw in `LLMFailure.response`.
- Capability gate (`validate_capability`) unchanged: embeddings/responses/images against native `GEMINIServiceEndpoint` → clear `ArgumentError`.

## Testing — zero-spend first, then one key-gated live witness

Zero-spend rule: `env -u OPENAI_API_KEY -u DEEPSEEK_API_KEY -u ANTHROPIC_API_KEY -u GEMINI_API_KEY`. Both HTTP majors in CI.

**Phase 1 (deterministic):** existing suite + Aqua green after the rename+bump.

**Phase 2 (deterministic, golden/canned, RED-first), `test/gemini.jl`:**
- *encode:* golden Gemini bodies for (system → `systemInstruction`), (tools + `toolConfig`), (multi-turn `functionCall`→`functionResponse` with `name` correlation), (`thought_signature` round-trip), (`maxOutputTokens` omitted when unset / present when set).
- *decode:* canned responses (text; text+functionCall; usage; error body) → assert neutral `Message`, finish-reason mapping, `TokenUsage`.
- *open-enum:* a canned response with a **novel `finishReason`** → asserts no crash (pre-registered refutation).
- *stream:* canned `alt=sse` sequence (text deltas; final chunk with `finishReason`+`usageMetadata`; a streamed `functionCall`) → assert accumulated `Message`, `usage`, and captured `thought_signature`.
- *mutation:* drop `thought_signature` in the encoder → a test asserts it is present, so the omission is caught.

**Live witness (key-gated `GEMINI_API_KEY`, `test/integration_gemini.jl`), `gemini-3.1-flash-lite` (cheapest durable model):** one text call + one tool round-trip + one stream, asserting the `…:generateContent` + `x-goog-api-key` contract. **Key visibility (billing-critical):** `GEMINI_API_KEY` lives in the user's `~/.zshrc` (interactive shell) but is **not** exported into the sandboxed non-interactive shell — whereas `OPENAI_API_KEY`/`ANTHROPIC_API_KEY` **are** live in the sandbox. So a bare `Pkg.test()` from the sandbox fires **billed OpenAI/Anthropic** witnesses while skipping Gemini; the `env -u …` unset-all is mandatory every run. The Gemini witness runs deliberately by sourcing `~/.zshrc` — but `GEMINI_API_KEY` is **billing-enabled**, so it spends real (small) money: run once when green, never rerun (zero-spend rule). It is the end-to-end observation required before claiming completion; green mocks alone do not suffice (theory-laden mocks can encode the same mistake twice).

## Scope

**IN:** Phase 1 shim rename (+ deprecated-default bump) · Phase 2 native `generateContent` chat / tools / streaming / usage / cost · `GPTToolCall.thought_signature` · capabilities/default/pricing · golden+canned mocks + key-gated witness.

**OUT (each its own later spec):** the **Interactions** API + the unified Layer-B agentic verb · `thinkingConfig`/`thinkingLevel` control · `responseSchema`/structured output (`:json_output`) · native embeddings (shim covers OpenAI-compat today) · Live API (WebSocket) · context caching · Batch · multimodal `inlineData` parts · the `GPT*`→neutral-name rename.

## Assumption ledger — verify against LIVE Google docs, not memory

There is no `claude-api`-equivalent skill for Gemini; verify while coding (`ai.google.dev`):

- exact `default_model` / cheapest-witness model IDs still GA (`gemini-3.5-flash`, `gemini-3.1-flash-lite`) — the model line moves fast.
- **pricing numbers** for `DEFAULT_PRICING` — research flagged a surprising `3.1-flash-lite` > `2.5-flash-lite` inversion; re-confirm cents before committing.
- **`usageMetadata` inclusion semantics** — does `promptTokenCount` already include `cachedContentTokenCount`? does `candidatesTokenCount` include `thoughtsTokenCount`? (decides add-vs-record; wrong choice double-counts cost).
- exact placement of `thoughtSignature` on the part and whether **text-only** (non-tool) assistant turns also require echo (keystone assumes function-call parts only).
- whether streamed `functionCall` parts ever arrive **fragmented** (decoder must tolerate; docs unconfirmed).
- newer `finishReason` spellings (`IMAGE_SAFETY`/`UNEXPECTED_TOOL_CALL`/`TOO_MANY_TOOL_CALLS`) — handled by the open-enum default regardless.

## Falsification — what would make us retract this design

- Phase 1: the existing suite or Aqua fails after the rename → the rename was not behaviour-preserving.
- A common Gemini-3 chat+tool multi-turn cannot round-trip through `Message` + `thought_signature` without loss → the "one field" claim is wrong and the IR enrichment must grow (e.g. text-part signatures) inside the keystone.
- The open-enum decode test crashes on an unknown `finishReason` → the defensive decode is not actually defensive.
- The live witness cannot complete a text call, a tool round-trip, or a stream → the translator is wrong regardless of green mocks.
