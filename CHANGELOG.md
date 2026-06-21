# Changelog

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
