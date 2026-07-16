# Changelog

## Unreleased

### Changed
- **Breaking:** `call_tool(session, name, args)` now returns an `MCPToolResult`
  instead of a bare `String`, and no longer throws when the tool reports an
  execution error (`isError: true`). The struct carries `content::String` (the
  rendered text, unchanged from before), `structured::Union{Nothing,Dict{String,Any}}`
  (the server's `structuredContent` verbatim), `is_error::Bool`, and
  `parts::Vector{Any}` (the raw content array) — so a tool-execution error is now
  distinguishable from a JSON-RPC protocol error (still thrown as `MCPError`). The
  `mcp_tools` / `mcp_tools_respond` tool-loop bridges are unchanged for callers:
  they still hand the model a string (the faithful `content`), now surfacing
  `structuredContent` when the content is otherwise empty and raising the tool's
  own error content on `isError`.

### Fixed
- Azure OpenAI deployment names configured through `AZURE_OPENAI_DEPLOY_NAME_*`
  environment variables are now read when the request URL is built, instead of
  being captured once when the package loads. A deployment name exported after
  `using UniLM` is honored; `add_azure_deploy_name!` registrations still take
  precedence.
- The tool-calling loop no longer swallows a user interrupt: an
  `InterruptException` (e.g. Ctrl-C) raised inside a tool function now
  propagates and aborts the loop instead of being recorded as a tool-call
  failure and retried. All other exceptions still become tool-error outcomes.
- MCP client now validates and honors the negotiated protocol version and
  recovers expired HTTP sessions (Streamable HTTP, MCP spec 2025-11-25). After
  `initialize`, a server `protocolVersion` outside the client's supported set
  (`2025-11-25`, `2025-06-18`, `2025-03-26`) closes the transport and raises an
  error naming both the requested and the returned version, instead of being
  accepted unchecked; a supported older version is stored and used. Every HTTP
  request after `initialize` now sends the negotiated `Mcp-Protocol-Version`
  header (previously a fixed constant), while the `initialize` request itself
  advertises the client's latest supported revision. A `404` on a request that
  carries a live session id triggers a single re-initialize (fresh session id)
  followed by one replay of the request before giving up, rather than failing
  outright. `401`/`403` responses raise an error that names the status and
  directs credentials to the `headers` kwarg of `mcp_connect`.

## 0.12.0

### Added
- `ProviderContent` and `Message.provider_content`: provider-native assistant
  content (Anthropic thinking/redacted_thinking blocks, Gemini parts with
  text-part `thoughtSignature`s) is captured verbatim at decode time — for
  both non-streaming and streaming Anthropic responses, and non-streaming
  Gemini responses — and echoed verbatim when the same provider encodes the
  conversation again. Never serialized on the OpenAI wire.

### Changed
- **Breaking:** `serve(server; transport=:http)` now blocks until the server
  is closed (matching stdio serving and `HTTP.serve`). Pass `block=false` for
  the previous behavior: it returns the running server handle, which you
  `close` yourself.
- The agentic streaming decode seam now threads one `AgenticStreamState`
  (text buffer, line carry, sticky event name, per-step assembly registry)
  instead of three loose buffer arguments. The seam is unexported; provider
  packages overriding `decode_agentic_stream` must adopt the new signature.
- MCP HTTP server transport now validates the `Origin` header (a Streamable
  HTTP requirement; DNS-rebinding defense): requests without an `Origin` and
  requests from localhost origins are accepted, anything else gets 403 unless
  listed in the new `allowed_origins` kwarg of `serve`.

### Fixed
- Anthropic tool calling on thinking models (e.g. `claude-sonnet-5`): assistant
  turns rebuilt from text+tool_calls dropped the thinking blocks the API
  requires back verbatim, so multi-turn tool use failed with HTTP 400.
- Gemini Interactions `thought` steps are now surfaced verbatim in
  `ResponseObject.output` instead of being collapsed into an empty
  `reasoning` stub (their `signature` was previously lost; `reasoning_items`
  no longer returns stub entries for them; filter `output` for type ==
  `"thought"`).
- Gemini Interactions streaming with tools: function-call steps
  (`step.start` + `arguments_delta` + `step.stop`) are now assembled and
  surfaced in the terminal response's `output`, so streamed
  `respond(...; tools=…)` returns a usable `requires_action` result instead
  of failing on a 200 stream. Streamed thought steps keep their signature.
- Gemini chat: parallel tool calls without wire `id`s now receive unique
  synthetic positional ids (reserved prefix `unilm_call_`), fixing tool-result
  correlation that previously collapsed to the last call; synthetic ids are
  omitted on re-encode.
- `fork(chat)` now copies every `Chat` field by construction (previously 15
  config fields — `reasoning_effort`, `max_completion_tokens`,
  `stream_options`, `verbosity`, `store`, `metadata`, `service_tier`,
  `logprobs`, `top_logprobs`, `prediction`, `modalities`, `audio`,
  `web_search_options`, `prompt_cache_key`, `safety_identifier` — were
  silently dropped, so forked chats behaved differently). Forks no longer
  normalize `parallel_tool_calls`; the copy is verbatim.
- MCP server: a syntactically valid JSON frame that is not a JSON object
  (array, string, number — e.g. a legacy JSON-RPC batch) now gets a `-32600`
  Invalid Request response; the stdio serve loop continues with the next frame
  (previously a `MethodError` killed it) and the HTTP transport returns 400
  instead of 500.
- MCP client: a request now reads frames until the response with its own id
  arrives — interleaved server notifications are skipped instead of being
  returned as the (empty) result and desyncing every subsequent call, and
  server-initiated `ping` requests are answered inline. After
  `notifications/tools/list_changed` the session marks its cached tool list
  stale (`session.tools_stale`; refresh with `list_tools!`). The whole
  exchange, including request-id allocation, runs under a session lock
  (previously only writes were locked and the id counter was racy). Custom
  `MCPTransport` subtypes must now also implement `_transport_read!`.

## 0.11.3

### Fixed
- Streaming (all providers, one shared SSE machine in `src/sse.jl`): `on_tool_call` now fires exactly once per completed streamed tool call (was: never); `stream_options.include_usage` no longer loses usage or turns a successful stream into `LLMFailure` (chat EOS is `data: [DONE]` only — `finish_reason` never ends the stream, and empty-`choices` chunks are tolerated, unbreaking Azure preambles and `: keep-alive` proxies); Anthropic mid-stream `error` events now produce `LLMFailure`(529 for `overloaded_error`)/`LLMCallError` instead of a truncated `LLMSuccess`; streamed messages keep assistant text alongside tool calls, and zero-argument streamed tool calls parse as `Dict{String,Any}()` instead of throwing. Failed SSE lines are logged and dropped, never re-queued; partial lines carry over verbatim (no whitespace loss at chunk boundaries).
- Gemini chat streaming reads to EOF (no sentinel exists), so trailing `usageMetadata` chunks are consumed instead of being cut off at `finishReason`.
- Streaming callbacks now receive each text delta as parsed (verbatim forwarding replaces O(n²) buffer re-diffing that could split multibyte characters).

### Changed
- Behavior change: an Anthropic stream truncated after `message_delta` but before `message_stop` now completes as `LLMSuccess` (the driver's EOF + recorded-finish_reason rule) where it previously produced `LLMFailure`.
- SSE parsing tolerates `data:` lines with or without the single optional space
  after the colon (`data:{…}` and `data: {…}` are equivalent), per the SSE spec.

### Removed
- Internal (unexported, documented) streaming seam `decode_stream_chunk` and `_parse_chunk`, replaced by `handle_sse_event!(service, event, payload, state) -> :continue | :done | :error` in `src/sse.jl` (no known external overriders).

## 0.11.2

### Added
- Exposed `x-request-id` header on all non-success/failure result structs (`LLMFailure`, `LLMCallError`, `FIMFailure`, `FIMCallError`, `ResponseFailure`, `ResponseCallError`).
- Implemented streaming residual flushing in the `:completed` terminal branch of `_respond_stream` to ensure text deltas received in the final stream chunk are not dropped.

## 0.11.1

Documentation-only release: an exhaustive review pass over the docs. No API or behavior changes.

### Changed
- Onboarding: the registered install (`Pkg.add("UniLM")`) now leads the README, docs home, and Getting Started (the GitHub-URL install is kept for tracking unreleased changes); the Julia 1.12+ prerequisite is stated up front.
- Positioning: reframed around first-class **native** backends (OpenAI, Anthropic, Gemini) plus the OpenAI-compatible tier, rather than "via the OpenAI-compatible API standard"; the Chat-Completions-vs-Responses comparison is now scoped to OpenAI.
- LLM reference (`llm.md`): the Complete Exports List is regenerated from `names(UniLM)` (previously missing ~40 symbols and the platform-API families); provider-compatibility table, capability sets, `tool_choice`/`DEFAULT_PRICING` types, and documented default models corrected.
- New guides: **Cost Tracking** (incl. the silent-`$0` behavior for unpriced models) and **Retrieval & File Search** (end-to-end Files → Vector Store → `file_search`).

### Fixed
- Corrected stale documentation: image default `gpt-image-2`, chat default `gpt-5.5`, the Gemini embeddings example (`GEMINIOpenAIServiceEndpoint`), retry-status lists, and the module / `provider_capabilities` docstrings. Added a root `llms.txt` for LLM/agent consumers.

## 0.11.0

### Breaking changes

- **`GEMINIServiceEndpoint` now targets Google's native `generateContent` API** (auth header `x-goog-api-key`; model in the URL), not the OpenAI-compatibility endpoint. The OpenAI-compatible Gemini shim is renamed **`GEMINIOpenAIServiceEndpoint`**. Migrate any code that used `GEMINIServiceEndpoint` for the OpenAI-compatible path — including `Embeddings(...; service=GEMINIServiceEndpoint)`, which the native endpoint does not support — to `GEMINIOpenAIServiceEndpoint`.

### Added

- Native Google Gemini chat (`GEMINIServiceEndpoint`): `generateContent` messages, tools (with Gemini-3 `thoughtSignature` echo for multi-turn tool calls), SSE streaming, and usage/cost accounting. Default model `gemini-3.5-flash`.
- Native Anthropic chat (`ANTHROPICServiceEndpoint`): Messages API messages, tools, streaming, usage/cost.
- Optional `GPTToolCall.thought_signature` field (set by the Gemini decoder; ignored by other providers).
- Unified agentic verb across providers: `respond`/`Respond` now also targets Google's **Gemini Interactions** API via `service=GEMINIServiceEndpoint` (in addition to OpenAI Responses), sharing inputs, `tool_loop`/`tool_loop!`, lifecycle (`get_response`/`cancel_response`), and usage/cost accounting.
- Gemini hosted-tool constructors `gemini_google_search`, `gemini_code_execution`, `gemini_url_context` for use in `respond(...; tools=[...])`.
- Cross-provider `estimated_cost`/`token_usage` for Gemini Interactions results (usage normalized to the shared shape; `gemini-3.5-flash` priced in `DEFAULT_PRICING`).

## 0.10.3

Chat-path `strict` structured outputs are now expressible. Additive and non-breaking:
when `strict` is not set, request bodies are identical to 0.10.2 (the field is omitted,
which is the API default — non-strict).

### Added
- `GPTFunctionSignature` gains `strict::Union{Bool,Nothing} = nothing` (strict function
  calling on the Chat Completions tool path). `true`/`false` are emitted inside the
  `"function"` object per the Chat Completions wire shape; `nothing` omits the field.
  `GPTTool(::AbstractDict)` reads `"strict"` back symmetrically (bare and wrapped
  formats), so dict-rendered tool definitions carrying `"strict": true` are now
  transmitted instead of silently dropped.
- `JsonSchemaAPI` gains the documented `strict` field for chat `response_format`
  structured outputs, and `json_schema(name, description, schema; strict=...)` threads
  it. `JsonSchemaAPI` now declares `JSON.omit_null` (all its previous fields were
  required, so existing serialized output is unchanged).

### Behavior note (deliberate bug fix)
- Tool definitions ingested as dicts (`GPTTool(::AbstractDict)` / `to_tool(::AbstractDict)`)
  that already carry a `"strict"` key now transmit it — previously the key was silently
  dropped. If a stored `"strict": true` definition has a strict-invalid schema, the API
  will now reject it with a 400; that rejection reflects what the definition always
  declared. Non-Bool `"strict"` values raise a descriptive `ArgumentError` instead of a
  raw `MethodError`.
- The pre-0.10.3 3-argument positional constructors `GPTFunctionSignature(name,
  description, parameters)` and `JsonSchemaAPI(name, description, schema)` are preserved
  via explicit methods (`@kwdef` field defaults do not extend positional constructors).

UniLM does not validate schemas against strict-mode rules (transport, not policy); the
API rejects strict-invalid schemas with a 400. Live transmission is witnessed by a
key-gated integration test: the same tool is accepted without `strict` and rejected
(400, `invalid_function_parameters`) with `strict=true` on a strict-invalid schema.

## 0.10.2

Documentation and CI maintenance only — **no functional changes to the library** (`src/`
is unchanged from 0.10.1). Released so the versioned/`stable` documentation reflects 0.10.x.

### Documentation
- MCP guide macro examples are now **executed at build time** (realistic, self-verifying),
  which prevents silent doc rot; added usage-contract notes for `@mcp_tool`/`@mcp_resource`.

### CI / tooling
- Updated GitHub Actions to current majors (`actions/checkout` v7, `julia-actions/setup-julia`
  v3, `julia-actions/cache` v3, `codecov/codecov-action` v7) and fixed CI-vs-Documentation drift.
- `Documentation` workflow gains `workflow_dispatch` for manual versioned-docs rebuilds.
- `CompatHelper` workflow now installs Julia (a `setup-julia` step was missing).

## 0.10.1

Patch release: correctness fixes for the MCP server macros, plus a large test-coverage
hardening pass. No breaking changes.

### Fixed
- `@mcp_resource` template form now binds the matched path parameters into the handler's
  declared arguments. The documented example
  `@mcp_resource server "file://{path}" function(path::String) read(path, String) end`
  previously raised `UndefVarError` at read time because the params were never unpacked;
  each declared argument is now bound from the matched URI `{param}` of the same name.
- `@mcp_prompt` now supports the documented anonymous `function(arg) … end` form. The first
  declared argument was previously dropped (omitted from the prompt's argument schema and
  left unbound in the handler); a shared, signature-shape-robust argument extractor now
  handles both the named and anonymous forms.
- `@mcp_tool` registers under the function name, so it now raises a clear error when given
  an anonymous function instead of silently registering a tool named after the first
  argument node. Use the documented named form `@mcp_tool server function name(args…) … end`.

### Internal
- Test suite substantially expanded — the MCP client/server operation layers, streaming,
  retry recursion, and error paths are now exercised end-to-end via deterministic in-process
  mocks. Project coverage ~99.6%. This work is tests-only; no behavior change.

## 0.10.0

"OpenAI first-class" release: correctness fixes, a fully-modeled Responses API, cache-aware
cost accounting, and broad new endpoint coverage.

> Model ids and `DEFAULT_PRICING` figures were verified against the live OpenAI `/v1/models` and
> official model pricing pages on 2026-06-21 (18/18 model ids present; the `gpt-5.2` rate was
> corrected to 1.75/0.175/14.0 per 1M tokens in that pass). Prices drift — re-verify over time.

### Breaking changes
- **`embeddingrequest!`** now returns `EmbeddingSuccess` / `EmbeddingFailure` / `EmbeddingCallError`
  (previously a `(dict, emb)` tuple on success and **threw** on failure). `emb.embeddings` is still
  filled in place; use `embedding_vectors(result)` to read the vectors.
- **`extract_message`** preserves partial assistant `content` for any `finish_reason` (e.g.
  `"length"`) instead of replacing it with `"No response from the model."` (that fallback is kept
  only for genuinely-empty responses).
- **`WebSearchTool`** defaults to the GA `type = "web_search"` (was `"web_search_preview"`). Pass
  `WebSearchTool(type="web_search_preview")` to restore the previous wire output.
- **`Respond.tool_choice`** widened to `Union{String,AbstractDict,Nothing}` (source-compatible;
  existing `String`/`nothing` callers are unaffected).
- **Default models** bumped: OpenAI chat/responses → `gpt-5.5` (was `gpt-5.2`), image → `gpt-image-2`
  (was `gpt-image-1.5`). These cost more per token — pin an explicit `model=` to control spend.

### Fixed
- Chat now sends **`max_completion_tokens`**; `max_tokens` is deprecated and rejected by reasoning models.
- Refusals are captured in both non-streaming and streaming paths, regardless of `finish_reason`.
- Responses streaming recognizes terminal `response.failed` / `response.incomplete` / `error` events and
  surfaces structured failures (previously lost as `ResponseFailure(status=200, raw)`); unknown/future
  events degrade gracefully.
- Embeddings `dimensions` / `encoding_format` supported; `update!` is resize-tolerant for non-1536-dim
  models (e.g. 3072-dim `text-embedding-3-large`).
- `_is_retryable` adds 408 / 502 / 504 / 529.

### Added — Responses API
- `tool_choice` builders: `tool_choice_function` / `_hosted` / `_mcp` / `_custom` / `_allowed`.
- `text.verbosity`; `WebSearchTool` `filters`; `MCPTool` `connector_id` / `authorization` /
  `server_description` / `tunnel_id` + `mcp_approval_response`.
- Typed output accessors: `reasoning_summaries`, `reasoning_items`, `refusals`, `url_citations`,
  `web_search_results`, `file_search_results`, `image_generation_results`, `code_interpreter_outputs`,
  `mcp_call_outputs`, `mcp_approval_requests`, `response_status`, `incomplete_details`, `usage_details`.
- New tool types: `LocalShellTool`, `ShellTool`, `ApplyPatchTool`, `CustomTool` (incl. grammar format),
  and GA `ComputerTool`. Input parts: `input_image(file_id=…)`, `input_file(file_data=…, filename=…)`.

### Added — Chat Completions
- `reasoning_effort`, `stream_options`, `verbosity`, `store`, `metadata`, `service_tier`, `logprobs` /
  `top_logprobs`, `prediction`, `modalities`, `audio`, `web_search_options`, `prompt_cache_key`,
  `safety_identifier`.

### Added — Accounting
- `TokenUsage` gains `cached_tokens` / `reasoning_tokens`. `estimated_cost` bills cached input at the
  discounted rate (no longer overcharges cache-heavy workloads); pricing table gains a `cached_input`
  column and refreshed/extended rows (incl. embeddings).

### Added — new endpoints
Files, Vector Stores (+ `poll_file_batch`), Conversations, Audio (TTS + transcription/translation),
Batch (+ `poll_batch`), Moderations, Image edits, Fine-tuning, Webhooks (HMAC-SHA256 verification),
Containers, Uploads (resumable), Videos, and Realtime (WebSocket transport + ephemeral client-secret
minting; WebRTC/SIP out of scope). Each is gated by a provider capability; non-OpenAI providers reject them.

### Dependencies
- Added `SHA` (stdlib) for webhook signature verification (no TLS dependency).
