# Design — Documentation sync (v0.10.3 → main) + scoped drift gate

**Date:** 2026-07-08 · **Status:** proposed (awaiting user review) · **Scope choice:** "Sync + drift gate"

## Problem (one sentence)
The Documenter site, `README.md`, and `CHANGELOG.md` do not describe the surface added on `main` since v0.10.3 (native Anthropic/Gemini chat, the breaking Gemini endpoint rename, the cross-provider agentic verb, hosted tools, `tool_result`/`tool_choice_*`, `thought_signature`), and nothing in the build **prevents** public symbols from silently falling out of sync — which is how the gap arose (`warnonly=[:missing_docs]`).

## Definition of done / pre-registered falsification
This work is **NOT** done if any of these hold:
1. A public (exported) symbol added since v0.10.3 is absent from every `@docs` block in `docs/src/**`.
2. `docs/make.jl` builds green when a *new* undocumented export is introduced (the gate must **bite** — proven by a negative test).
3. A Gemini or Anthropic single-shot example is shown as a non-executed ` ```julia ` block where the existing OpenAI house style would run it live.
4. `CHANGELOG.md [Unreleased]` still omits the agentic verb.
5. Any example uses a model name that is not the actual code default (a refuted example).
6. The migration note for the breaking `GEMINIServiceEndpoint` rename is missing.

## Scope
**In:** surface new since v0.10.3 + the drift gate + the `Documentation.yml` key wiring.
**Out (deferred, tracked in the allowlist as explicit debt):** the ~13 pre-existing undocumented feature areas (Files, Vector Stores, Conversations, Audio, Batch, Moderations, Realtime, Fine-tuning, Webhooks, Containers, Uploads, Videos) and `@docs` blocks for cost tracking / `fork`. These stay in the allowlist; they are not documented this session but are made *explicit and falsifiable* rather than silent.

## Ground truth (verified against source, 2026-07-08)
- **Breaking Gemini rename — CONFIRMED.** `GEMINIServiceEndpoint` (`src/api.jl:353`) is now **native** `generateContent` (`src/gemini.jl:11-17`, `x-goog-api-key`, model-in-URL, SSE `:streamGenerateContent?alt=sse`). The OpenAI-compat shim is renamed `GEMINIOpenAIServiceEndpoint` (`src/api.jl:350`, `get_url → GEMINI_CHAT_URL` at `src/requests.jl:29`). A previously-exported name changed meaning → breaking (git `55864d1`, `c821533`).
- **Native Anthropic — CONFIRMED.** `ANTHROPICServiceEndpoint` (`src/api.jl:357`) → `/v1/messages` (`src/anthropic.jl:10`), `x-api-key` + `anthropic-version` headers, native encode/decode.
- **Agentic verb — CONFIRMED.** `Respond` (`src/responses.jl:669`), `respond()` (`:1171`); seam `_agentic_url`/`encode_agentic`/`decode_agentic`/`decode_agentic_stream` (`src/responses.jl:1139-1146`); Gemini overrides (`src/interactions.jl:18,22,168,178`). Lifecycle `get_response`/`delete_response`/`list_input_items`/`cancel_response` (`src/responses.jl:1274,1300,1328,1368`) route through the shared seam — Gemini works via `_agentic_url` without dedicated overrides.
- **Hosted tools — CONFIRMED.** `gemini_google_search`/`gemini_code_execution`/`gemini_url_context` (`src/interactions.jl:56-58`, exported `src/UniLM.jl:195-197`).
- **`thought_signature` — CONFIRMED.** `GPTToolCall.thought_signature::Union{Nothing,String}` (`src/api.jl:93`), set by Gemini decoders, echoed on the wire, omitted from OpenAI-wire lowering.
- **Cross-provider cost — CONFIRMED.** `estimated_cost`/`token_usage` (`src/accounting.jl:66,42`); Gemini Interactions usage normalized to OpenAI-Responses keys (`src/interactions.jl:137-146`); Gemini priced in `DEFAULT_PRICING`.
- **Docs CI keys — CORRECTED.** All provider keys already exist as **GitHub repo secrets** (`gh secret list`: OPENAI/ANTHROPIC/GEMINI/DEEPSEEK). `CI.yml:17-22` forwards all four; `Documentation.yml:33-36` forwards **only** `OPENAI_API_KEY` — the sole reason live cross-provider examples don't run in the docs build.

## Design decisions
**D1 — Scoped drift gate (the falsification mechanism).** A small function in `docs/make.jl`, run at build time (local == CI):
- `exported = filter(≠:UniLM, names(UniLM))`.
- `documented =` symbols parsed from every ` ```@docs ` fence under `docs/src/**` (normalize: strip `UniLM.` qualifier and leading `@`).
- Assert `setdiff(exported, documented) ⊆ KNOWN_UNDOCUMENTED`; also `KNOWN_UNDOCUMENTED ⊆ exported` (no stale entries) and `isempty(intersect(documented, KNOWN_UNDOCUMENTED))` (documenting a symbol forces removing it from the list). Fail with a clear `error()` listing offenders.
- `KNOWN_UNDOCUMENTED` lives in a checked-in file (e.g. `docs/undocumented_allowlist.jl`) — the visible "known-debt ledger" that can only shrink.
- **Seed empirically, not by guess:** run the check once, capture its initial report, seed the allowlist from that output.
- **Why not Documenter `checkdocs=:exports` strict:** all-or-nothing (forces Full-audit now) and blind to exports lacking a docstring. The custom gate is scoped + catches undocumented-and-undocstring'd exports.
- **`warnonly` unchanged:** the custom gate is the enforcement; Documenter's `[:missing_docs, :cross_references]` stay as warnings (pre-existing docstring'd-but-unincluded symbols would otherwise fail the build — Full-audit scope).

**D2 — Examples (corrected).** Match the existing house style for *all* providers: single-shot calls are **live-guarded `@example`** blocks (real round-trip, build-verified), using the established guard idiom so a failed/keyless call degrades to deterministic fallback text instead of breaking the build:
```
```@example tag
result = respond("…"; service = ANTHROPICServiceEndpoint)   # or GEMINIServiceEndpoint
println(result isa ResponseSuccess ? output_text(result) : "Request failed — " * output_text(result))
```
```
Multi-turn / lifecycle / infra flows (`tool_loop!`, background `get_response`/`cancel_response`, MCP servers) stay non-executed ` ```julia ` — matching the docs' current judgment, not a key constraint.

**D3 — Page structure (user-selected).** New dedicated `docs/src/guide/agentic.md` for the cross-provider verb; `responses_api.md` keeps OpenAI-specifics and links to it.

**D0 — CI wiring (prerequisite).** Add `ANTHROPIC_API_KEY`/`GEMINI_API_KEY`/`DEEPSEEK_API_KEY` to `Documentation.yml` `env:`, mirroring `CI.yml:17-22`. Secrets already exist.

## Content plan (per file)
- `.github/workflows/Documentation.yml` — forward the three missing provider keys.
- `docs/make.jl` + `docs/undocumented_allowlist.jl` — the gate + seeded allowlist.
- `docs/src/guide/multi_backend.md` — breaking-rename migration callout; rewrite stale Gemini section (native `generateContent`, `x-goog-api-key`, model-in-URL, `thoughtSignature`, shim = `GEMINIOpenAIServiceEndpoint`); replace "not production-ready" Anthropic shim note with a real native `ANTHROPICServiceEndpoint` section.
- `docs/src/guide/agentic.md` (NEW) — unified `respond` across OpenAI Responses ⇄ Gemini Interactions: `service=` swap, `tool_loop`/`tool_loop!`, lifecycle across providers, hosted tools, `tool_result`, `tool_choice_*`, cross-provider `estimated_cost`. Add to `pages=` tree.
- `docs/src/guide/tool_calling.md` — `tool_result`, `tool_choice_*` builders, Gemini hosted tools, `thought_signature`.
- `docs/src/api/endpoints.md` / `docs/src/api/responses.md` — `@docs` for `GEMINIOpenAIServiceEndpoint`, `ANTHROPICServiceEndpoint`, `tool_result`, `tool_choice_*`, `gemini_google_search`/`_code_execution`/`_url_context` (removes them from the allowlist).
- `docs/src/llm.md` — new surface + fix stale "v0.9.1" self-label.
- `README.md`, `docs/src/getting_started.md`, `docs/src/index.md` — native Anthropic/Gemini + agentic mentions; **align all example model names to the actual code defaults** (verify against source before writing).
- `CHANGELOG.md` — expand `[Unreleased]` to cover the whole agentic verb (currently only native chat).

## Sequencing (piecemeal — each commit builds green + gate stays green)
1. **CI wiring + gate + seeded allowlist** — lands the falsification mechanism first; all new surface still in the allowlist so build is green.
2. Gemini breaking rename + migration (multi_backend, endpoints `@docs`) → remove Gemini-native symbols from allowlist.
3. Native Anthropic (multi_backend, endpoints `@docs`) → remove from allowlist.
4. Agentic verb (`agentic.md`, responses/`@docs`, hosted tools, tool_result, tool_choice_*, lifecycle, cost) → remove from allowlist.
5. `tool_calling.md` + API `@docs` (thought_signature, hosted tools).
6. `llm.md` + README + getting_started/index model-name alignment.
7. CHANGELOG `[Unreleased]` expansion.

## Verification plan (theory-laden observation — state what each check can and cannot see)
- **Build passes:** `julia --project=docs docs/make.jl` locally (`source ~/.zshrc` first for the Gemini key) and in CI. *Cannot see:* semantic correctness of prose, only that examples execute and links resolve.
- **Gate bites (negative test):** temporarily drop one real undocumented symbol from `KNOWN_UNDOCUMENTED` (no `src/` edit needed) → confirm `make.jl` **errors** naming that symbol; revert. Without this, a green build is not evidence the gate works.
- **Examples are real:** live-guarded blocks execute against the real APIs in CI (keys now wired); a diff of the rendered output confirms non-fallback text for at least one Gemini and one Anthropic example.
- **Spend/flakiness (known limitation):** every docs build now bills 4 providers; a provider outage publishes `"Request failed — …"` text. Accepted per house style; recorded here.

## Risks / what would refute this design
- The `@docs`-fence parser mis-normalizes a symbol form (e.g. `@macro`, qualified names) → gate false-positives/negatives. Mitigation: unit-cover the parser on the real `docs/src` corpus.
- Live examples flake in CI and publish fallback text. Mitigation: keep only single-shot calls live; fence anything multi-step.
- Model-default drift assumed rather than verified. Mitigation: read the actual defaults from source before writing any example.
