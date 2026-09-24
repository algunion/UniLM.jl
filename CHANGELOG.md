# Changelog

## Unreleased

## 0.20.0

### Breaking
- **Requirements:** Julia 1.13 or later, HTTP.jl 2.7.1 or later within the 2.x major,
  and JSON.jl 1.8.1 or later. HTTP.jl 1.x is no longer supported, and with it go the
  1.x-only caveats (the `connect_timeout = Inf` restriction and the process-global
  connection-pool cap). HTTP.jl 2.7.1 fixes an HTTP/2 flow-control leak in which every
  response body closed unread shrank the connection's receive window until later
  requests on that connection hung.
- **The Videos API is removed.** OpenAI shut the Videos API and the `sora-2` models
  down on September 24, 2026. `VideoObject`, `VideoList`, `VideoSuccess`,
  `VideoListSuccess`, `VideoContentSuccess`, `VideoFailure`, `VideoCallError`,
  `create_video`, `retrieve_video`, `list_videos`, `video_content` and the `:video`
  capability are gone.
- **Changed defaults.** Native Anthropic defaults to `claude-opus-5-5` (was
  `claude-opus-4-8`) with a default `max_tokens` of 16000 (was 4096): `max_tokens` caps
  thinking plus text, and current Claude models think by default, so 4096 ended turns
  at `"length"` before an answer. DeepSeek defaults to `deepseek-flash` for chat and
  FIM (was `deepseek-chat`, which DeepSeek discontinued on July 24, 2026).
- **Local validation throws before any network I/O**, streaming or not.
  `chatrequest!` and `respond` now let an encoder's `ArgumentError` propagate — an
  option the provider or model cannot express (the GPT-6 and GPT-5.6 rules, the
  Anthropic and Gemini field mappings, the Interactions field set) — where 0.19.0
  returned it inside an `LLMCallError` / `ResponseCallError`. With `history=true`,
  `chatrequest!` throws `InvalidConversationError` for a conversation that ends with an
  assistant message, instead of billing a call whose reply could not be appended.
  `chatrequest!` throws `ArgumentError` for a `callback` or `on_tool_call`, and `respond`
  for a `callback`, passed without `stream=true`: they run only on a stream, and were
  silently ignored. `tool_loop!` passes them through, so it raises the same error.
  `chatrequest!(; kwargs...)` takes either `messages` or both `systemprompt` and
  `userprompt` and throws `ArgumentError` otherwise (it returned a fabricated
  `LLMFailure(status=499)`); it copies `messages` instead of emptying the caller's
  vector.
- **Stricter construction.** `Chat(n=…)` must be 1: a result carries one choice, and
  extra choices were silently dropped (or, streamed, merged into one garbled text).
  `Chat` and `Reasoning` reject a reasoning effort outside `none`, `minimal`, `low`,
  `medium`, `high`, `xhigh`, `max`; `Chat` rejects `top_logprobs` outside [0, 20] and
  `logit_bias` values outside [-100, 100] (the field now takes any
  `AbstractDict{String,<:Real}`); `Message` rejects roles other than system, user,
  assistant and tool; `Embeddings` rejects an `encoding_format` other than `"float"`;
  `Respond` rejects `conversation` together with `previous_response_id`, and
  `background=true` with `store=false`; `FIMCompletion(stream=true)` throws, since
  `fim_complete` has no streaming path. Audio and upload requests are checked against
  the documented values: `TranscriptionRequest.response_format` (json, text, srt,
  verbose_json, vtt, diarized_json) and `temperature` (0–1), `SpeechRequest.speed`
  (0.25–4), `translate` (no `languages`, `keywords` or `diarized_json`) and
  `create_upload`'s `purpose`. `RequestConfig` and the MCP per-call `timeout` reject a
  finite value above 1e9 s (use `Inf` to disable a bound).
- **Stopping a stream is a typed cancellation.** Setting the callback's
  `close[] = true` — now honoured from any task, and at once — ends the call with
  `LLMCallError` / `ResponseCallError` (`status = nothing`, `cause =
  UniLMCancelled(:callback, …)`), where it returned `LLMFailure` / `ResponseFailure`
  with `status == 200` and the raw partial wire. A stop that arrives after the
  provider's completion marker keeps the turn: it is committed, the final-message
  callback runs, and usage may be missing (on the OpenAI wire the marker is the chunk
  carrying `finish_reason`, which precedes the usage chunk and `[DONE]`). An exception
  thrown by a streaming `callback` or `on_tool_call` now ends the call with that
  exception in `cause` — never retried, nothing appended; `on_tool_call` exceptions used
  to be logged and ignored.
- **Result accessors throw on failures.** `output_text` (which returned the error text
  as if it were model output), `embedding_vectors` (a `MethodError`), `image_data` (an
  empty list) and `fim_text` (`""`) throw `LLMResultError` on a non-success result, as
  `text` already did. `LLMResultError.result` now holds any `LLMRequestResponse`.
- **`fork` deep-copies every field except `service`.** It copied only `messages`, so a
  fork's `push!(fork.tools, t)`, a `metadata` edit or a `stop` change also changed its
  parent and siblings.
- **Tool loops.**
  - `tool_loop!` runs tool calls only on a turn whose `finish_reason` is
    `"tool_calls"`. A turn with calls that finished otherwise (`"length"`,
    `"content_filter"`, a provider-specific value) may hold partial calls: none runs, the
    loop returns `completed=false` with an `llm_error` naming the reason, and the
    unanswered assistant turn is removed from the chat so it stays sendable. A call cut
    off before its arguments formed a JSON object is dropped from such a turn by the
    decoder; a turn left without calls ends the loop as a text turn with that reason.
  - `tool_loop!` completes on a text turn only when it finished with `"stop"` or no
    reason, or with `"tool_calls"` but no calls (which used to resend the conversation);
    a content-filtered or otherwise cut-off text turn returns `completed=false` with an
    `llm_error` naming the reason, where it counted as completed.
  - `tool_loop!` on a `Chat` with `history=false` throws `ArgumentError`, and both
    loops throw `ArgumentError` for `max_turns < 1`.
  - On `max_turns` exhaustion, `response` is the last response the model sent, with
    `completed=false` and `llm_error = "max turns (N) exhausted"`; it was a fabricated
    call error.
  - A dispatcher result that is not a `String` is sent to the model JSON-encoded, not as
    its Julia `string` (`Dict("temp" => 21.5)` arrives as `{"temp":21.5}`, `nothing` as
    `null`).
  - The Responses loop chains a `Respond` that sets `conversation` through the
    conversation alone (the API rejects it together with `previous_response_id`);
    answers call arguments that are not a JSON object with an
    `"Error: invalid arguments: …"` output instead of throwing; and stops with
    `completed=false`, running none of the turn's calls, when a turn requests a
    client-side action it cannot execute (`custom_tool_call`, `apply_patch_call`,
    `local_shell_call`, `computer_call`, a `shell_call` the platform did not answer with
    a `shell_call_output` in the same output, `mcp_approval_request`) — such a turn used
    to end the loop as completed. The `llm_error` of a turn that did not complete names
    the `incomplete_details` reason with its status
    (`"Response did not complete (status=incomplete, reason=max_output_tokens)"`).
- **Platform verbs.**
  - `upload_file` makes a single attempt; `max_attempts` no longer applies. A POST that
    timed out, or drew a gateway 5xx after the backend stored the file, was retried and
    stored the file twice. A `FileFailure` / `FileCallError` therefore does not prove
    that no file was created.
  - `delete_file`, `delete_vector_store`, `delete_container` and
    `delete_conversation` report success only when the reply carries
    `"deleted": true`; anything else is the family's `*CallError`.
    `delete_conversation_item` returns a `ConversationSuccess` holding the updated
    conversation the service sends back.
  - `poll_batch` and `poll_file_batch` bound the poll by wall-clock time (`timeout`),
    capping each GET and pause at the time left; they poll through retryable statuses
    and per-attempt timeouts, require `interval` and `timeout` > 0 (`ArgumentError`),
    and on timeout return the family's call error with `cause =
    UniLMTimeout(:deadline, …)` and the last object seen in the new `last_observed`
    field.
  - `moderate` returns a `ModerationCallError` (on which `is_flagged` throws) for a
    `200` that does not carry exactly one result row per input: such a reply decoded as
    a success that was not flagged, so unmoderated content passed a moderation gate.
  - `add_container_file` returns a `ContainerSuccess` whose `response` is a new
    `ContainerFileObject`; `ContainerSuccess.response` is
    `Union{ContainerObject,ContainerFileObject}`.
  - `ChoiceQuestion`'s constructor validates descriptions the way `choice` does, and a
    System One answer without a string `type` fails to decode.
- **Result types gained fields**, so their positional constructors changed (keyword
  construction is unchanged): the `*Failure` types of the Files, Vector Stores,
  Conversations, Moderations, Audio, Batch, Fine-tuning, Containers and Uploads APIs,
  `ImageFailure`, `ImageCallError`, `SystemOneCallError` and `RealtimeCallError` carry
  `request_id`; the `*CallError` types of those APIs, `ImageCallError`,
  `SystemOneCallError` and `RealtimeCallError` carry `cause`; `BatchCallError` and `VectorStoreCallError` carry
  `last_observed`; `ImageObject` carries `url`. `MCPServer` dropped its internal
  `_initialized` field and gained a registry `lock`.
- **`DEFAULT_PRICING` is a lock-guarded `AbstractDict{String,PriceRow}`**, not a
  `Dict`: every Chat success reads it, streaming ones on their own task, while callers
  may add rows, and a `Dict` is not safe to read during a write. It keeps
  `DEFAULT_PRICING[model] = row`, `get`, `haskey`, `delete!`, `pop!`, `empty!`, `keys`,
  `length` and snapshot iteration; `copy`, `merge` and `filter` return a plain `Dict`,
  and `DEFAULT_PRICING isa Dict` is false. `estimated_cost(…; pricing)` accepts any
  `AbstractDict{String,PriceRow}`.
- **Native Anthropic fails closed.** A `Chat` field the Messages API cannot express —
  `seed`, `logprobs`, `top_logprobs`, `presence_penalty`, `frequency_penalty`,
  `logit_bias`, `verbosity`, `store`, `prompt_cache_key`, `stream_options`,
  `prediction`, `modalities`, `audio`, `web_search_options` — throws `ArgumentError`
  naming the fields instead of being dropped, `n` must be 1, a `json_object`
  `response_format` throws (there is no schema-less JSON mode), an unknown
  `tool_choice` string throws (it became `auto`), and `metadata` may carry only
  `user_id`. Requests Claude answers with HTTP 400 are refused locally: `temperature`
  outside [0, 1]; any `top_p`, or a temperature other than 1.0, on Opus 4.7 and later;
  a forced `tool_choice` on Opus 5.5, Fable 5.1 and Mythos 5.1; a
  trailing assistant turn (prefill) from the 4.6 generation on; and a
  `reasoning_effort` the model does not take. Because `parallel_tool_calls` defaults to
  `false` whenever `tools` is set, every Anthropic tool request now sends
  `tool_choice.disable_parallel_tool_use = true` (at most one tool call per turn); set
  `parallel_tool_calls = true` for the previous wire.
- **MCP.**
  - A call on a closed or not-connected session throws the new
    `MCPSessionClosedError` (cause `:disconnected`, `:timeout` or `:crash`) instead of
    an `ErrorException`, on both transports; an HTTP session no longer keeps working
    after `mcp_disconnect!`.
  - The inferred-schema `register_tool!(server, name, description, handler)` binds the
    `arguments` object to the handler's positional parameters by name, as `@mcp_tool`
    does, and rejects a handler that takes one `Dict` or varargs with `ArgumentError`
    (register those with an explicit schema). Both bindings validate instead of
    coercing: a missing required argument or a value of the wrong JSON type is answered
    with an `isError: true` tool result naming the argument.
  - An exception from a resource or prompt handler is answered with a generic JSON-RPC
    `-32603` `"Internal error"` and logged on the server, instead of relaying its text
    (file paths, argument values) to the peer.
  - `mcp_tools` / `mcp_tools_respond` advertise provider-safe tool names: any character
    outside `[A-Za-z0-9_-]` becomes `_` and names are cut to 128 characters (the call
    still goes out under the MCP name); two tools that map to one name raise
    `ArgumentError`.
- **Realtime.** `realtime_connect` throws `ArgumentError` before any I/O for a service
  other than `OPENAIServiceEndpoint` — it always dialled OpenAI, so another endpoint's
  credentials would have been sent to `api.openai.com` — and `mint_realtime_secret`
  returns `RealtimeCallError` for a `200` without a non-empty secret.

### Added
- **Cooperative cancellation:** `CancelToken`, `cancel!`, `iscancelled`, `with_cancel`
  and the `UniLMCancelled` exception. A `cancel` keyword (default: the ambient
  `with_cancel` token) on `chatrequest!`, `respond`, `embeddingrequest!`,
  `generate_image`, `edit_image`, `fim_complete`, `prefix_complete`, `ask`,
  `list_models`, `poll_batch`, `poll_file_batch`, `tool_loop!` / `tool_loop`,
  `nl_dispatch` and `@branch`; every other HTTP verb observes the ambient token, which
  propagates into spawned tasks. A cancel — before connecting, during the header wait,
  mid-stream, during a retry backoff or a poll pause — ends the call with its call-error
  result (`status = nothing`, `cause = UniLMCancelled(:token, …)`), never retried and
  never committed — unless the provider's completion marker already arrived on a
  stream: then the turn is committed, the final-message callback runs, and usage may be
  missing. A pre-cancelled token sends nothing. Tokens are level-triggered and
  `cancel!` runs every registered hook, even when one of them raises an interrupt. A
  TCP connect or TLS handshake in progress is not interrupted (it completes or reaches
  `connect_timeout`), and the Realtime WebSocket and MCP stdio exchanges do not observe
  the token; a cancelled MCP call over HTTP throws `UniLMCancelled` and sends the
  server a best-effort `notifications/cancelled`.
- `tool_concurrency` on `tool_loop!` and `tool_loop`: up to that many of one turn's tool
  calls run at once on spawned tasks, with their results sent back in call order.
- **MCP:** the `:queue` phase of `MCPTimeoutError` (a call's `timeout` also bounds its
  wait for the session); a `stderr` keyword on `mcp_connect(::Cmd)` and
  `StdioTransport` (an `IO` or a file path, appended to); an exit hook that tears down
  stdio servers still running when Julia exits, and a watcher that kills a server's
  whole process group when its leader exits (so killing an `npx` wrapper also stops the
  server it launched); a best-effort `notifications/cancelled` for an HTTP request that
  timed out; `client_version` defaults to the package version; custom `MCPTransport`
  subtypes can connect (they bound their own IO).
- **Anthropic:** structured outputs (a JSON Schema `response_format` becomes
  `output_config.format`), `FunctionSignature.strict` as the tool's `strict`,
  `reasoning_effort` mapped to `output_config.effort` plus adaptive thinking per model
  family (`"none"` disables thinking where it can be disabled), `service_tier`
  (`"auto"`, `"standard_only"`), and `safety_identifier` / `user` sent as
  `metadata.user_id`.
- **DeepSeek:** `reasoning_content` is kept on the assistant `Message` as
  `ProviderContent(:deepseek, …)` and sent back on requests that carry tools, as
  thinking-mode tool use requires; context-cache hits (`prompt_cache_hit_tokens`) fill
  `TokenUsage.cached_tokens` and are priced at the cached rate.
- Price rows for the current Claude models (Fable 5.1, Mythos 5.1, Fable 5, Mythos 5,
  Opus 5.5, Opus 5, Sonnet 5 at \$2/\$10, and the 4.x family) and for `deepseek-flash`,
  `deepseek-v4-pro` and the legacy Flash names (DeepSeek peak rates; off-peak bills
  half). Anthropic's dated `-YYYYMMDD` ids resolve to their alias row, and an unpriced
  model id logs one warning. `token_usage` reads the usage an `ImageSuccess` carries,
  and FIM results report usage and cost.
- OpenAI and DeepSeek streams request `stream_options.include_usage` when it is unset,
  so a streamed turn carries usage and accrues cost like a non-streamed one.
- `ImageObject.url`: `image_data` returns each image's base64 data or, when it was
  delivered by URL, its URL. `limit` and `after` on `list_fine_tuning_events` and
  `list_fine_tuning_checkpoints`.
- The provider extension API — `get_url`, `auth_header`, `default_model`,
  `encode_request`, `decode_response`, `handle_sse_event!`, `StreamState`,
  `encode_agentic`, `decode_agentic`, `decode_agentic_stream`, `AgenticStreamState` —
  is declared `public` (not exported) and documented in the new Extension API reference.
- Documentation: a "Concurrency, Tasks and Cancellation" guide (what a concurrent call
  may share, fan-out patterns, streaming into a `Channel`, cancellation, the thread
  layout, HTTP, tool loops and MCP under load).

### Changed
- `Retry-After` is a floor under the jittered backoff instead of a replacement for it:
  clients that received the same header slept exactly the header's wait and retried in
  lockstep. The spread above the floor is at most the exponential backoff and at most
  half of the budget left after the floor, so the next attempt keeps at least half of
  what remains; a header whose own wait does not fit `total_deadline` still returns the
  last real response at once.
- Streams use HTTP/1.1, one connection per stream; non-streaming requests keep protocol
  negotiation (HTTP/2 over TLS).
- `stream_idle_timeout` measures wire idleness only: time spent inside a streaming
  callback or `on_tool_call` is not counted, and the clock restarts when it returns.
- An in-band stream error on the OpenAI wire that carries a numeric `code` from 400 to
  599 is an `LLMFailure` with that `status` (other in-band errors stay an
  `LLMCallError`), so a retryable one (`408`, `429`, `500`, `502`, `503`, `504`, `529`)
  is retried with a new request while no callback has run yet, like its HTTP-status twin.
- `update!` on a `Chat` with `history=false` logs at debug level instead of warning on
  every successful call: leaving the chat unchanged is what `history=false` means.
- Chat decoding keeps the provider's finish reason on a tool-call turn (`"tool_calls"`
  stands in only for `"stop"` or none), and a stream without a finish reason reports
  `nothing`. Partial calls are dropped: on a turn that keeps another reason, a call
  whose arguments were cut off before they formed a JSON object is removed and the turn
  kept with its text, reason and usage; under `"tool_calls"` such arguments are an
  `LLMCallError`, since those calls would run. Native Gemini maps `STOP` to `"stop"`,
  `MAX_TOKENS` to `"length"` and the
  content filters, the image ones included (`IMAGE_SAFETY`, `IMAGE_PROHIBITED_CONTENT`,
  `IMAGE_RECITATION`), to `"content_filter"`; any other
  `finishReason` is reported as its lowercased wire value (`"malformed_function_call"`)
  instead of `"stop"`, and a candidate without one as `nothing`. `reasoning_summaries` also reads
  Gemini Interactions `thought` steps.
- A native Anthropic system message after the start of a conversation stays in place as
  a mid-conversation system message on models that accept one, where its placement is
  valid; anywhere else it is hoisted into the top-level `system` prompt as before.
- Requests send a refusal as `{"content": null, "refusal": …}` and never send the
  response-only `finish_reason`. `Respond(tools=…)` converts a Chat `Tool` to the
  equivalent `FunctionTool`, and `Chat(tools=Tool[])` stores `nothing`.
- MCP: stdio lines that are not JSON are skipped with a warning; `MCPError.data` holds
  any JSON value; the disconnect `DELETE` carries `MCP-Protocol-Version`; the server negotiates
  the protocol revision the client requests when it supports it, answers `400` over HTTP
  to a request whose `MCP-Protocol-Version` names an unsupported revision, runs
  `serve(:http)` handlers concurrently on the default thread pool (handlers must be
  thread-safe; protocol requests stay inline), runs stdio handlers one at a time with
  the process `stdout` pointed at `stderr`, and accepts a JSON-RPC response sent to it
  without answering (HTTP `202`).
- Capability errors name the endpoint (they printed `DataType` for marker types); a
  custom endpoint without a `default_model` method gets the documented `ArgumentError`
  instead of a `MethodError`; the OpenAI-wire endpoints (OpenAI, Azure, the Gemini
  compat shim, DeepSeek, generic) declare `:streaming` and `:json_output`.
- `nl_dispatch` and `meanings` offer options in definition order (REPL input sorted
  `REPL[10]` before `REPL[2]`), and a method defined after the call began is callable.
- CI runs a single latest-stable Julia leg; CompatHelper has the write permissions it
  needs to open its pull requests.

### Fixed
- Every non-streaming call paid a ~100 ms latency floor: the request watchdog polled
  for completion every 0.1 s. Completion now wakes the caller directly, and a timeout
  lands within a few milliseconds of its bound.
- A busy main task stalled calls running on other threads: closing a watchdog timer
  waits for the event loop, which shares thread 1 with the main task. Timers are now
  closed off the caller's path.
- Over HTTP/2, concurrent streams to one host shared a connection and its flow-control
  windows, so a stream whose consumer applied backpressure starved its siblings, and a
  stream closed early could leave the connection stalled. Streams now each own an
  HTTP/1.1 connection.
- A slow stream consumer lost a healthy, fully delivered stream to `:stream_idle`,
  because the time spent in its callback counted as a byte gap.
- A cancelled HTTP request (`HTTP.CanceledError`) was classified as a retryable
  transport failure.
- Streaming parsed each SSE line and accumulated each tool call's arguments in
  quadratic time: a 4 MiB line read in 16 KiB chunks allocated about 1 GB. Native
  Anthropic stream blocks (16,000 deltas allocated 3.9 GB) and Gemini Interactions
  steps accumulated the same way. All three are linear now.
- An `InterruptException` raised while parsing a streamed payload, inside
  `on_tool_call`, while rendering an error, while parsing a TypeSafe error body, while
  encoding a Gemini tool result, or inside an MCP server handler was swallowed or
  converted; it propagates.
- Chat decoding dropped text sent alongside tool calls, failed on array-shaped message
  content and on empty tool-call arguments (`""` is `{}`), relabelled a tool-call turn
  cut at `"length"` or filtered as `"tool_calls"` (so a tool loop could run partial
  calls), failed a turn cut off inside a call's arguments with an `LLMCallError` (a
  JSON parse error, its billed usage never accrued), invented `"stop"` for a stream
  that sent no finish reason, and did not treat an in-band OpenAI-wire `{"error": …}`
  payload as terminal. `request_id` also falls back to the `request-id` header, which
  Anthropic sends.
- Before its first byte, a stream was bounded by `stream_idle_timeout`, not by
  `min(request_timeout, remaining total_deadline)`: closing the stream does nothing in
  HTTP.jl 2.x while the request is uploading or the response headers are awaited, so a
  peer that never answered held each attempt until the read-idle timer fired (120 s by
  default, however small `request_timeout` was). The first-byte deadline now also
  cancels the attempt's request context and ends it typed `UniLMTimeout(:request, …)`.
- A Chat stream waited for the end of its HTTP body after `[DONE]`: a server holding
  the body open delayed the final-message callback and the result as long as it held
  it, and a body that never ended cost `stream_idle_timeout`. The turn is final at the
  sentinel; the body's end is awaited for at most 0.5 s (it returns the connection to
  the pool), after which the connection is closed instead of reused.
- An `InterruptException` from a `tool_loop!` dispatch left the assistant turn — and
  any tool results already appended — in the chat, so the next request carried tool
  calls without results. The interrupted turn is now removed first, with one tool at a
  time or `tool_concurrency > 1`.
- `ResponseCallError.cause` is set for every exception from `respond` and the lifecycle
  operations, not only for timeouts.
- A Chat `Tool` wrapped in a `CallableTool` — the form a Chat tool loop takes — went
  out from `Respond` in the Chat shape (the function nested under `function`), which
  the Responses API rejects. It now converts to the equivalent `FunctionTool` like a
  bare `Tool`, keeping its callable, alone or in a mixed tool list.
- FIM: a `200` whose body was not JSON escaped `fim_complete` as an `ArgumentError`, a
  body without choices decoded as an empty completion, call errors lost their `cause`,
  Mistral FIM was sent to `/v1/completions` (now `/v1/fim/completions`, with its
  `message.content` choices read), and `prefix_complete` replaced the prefix in history
  with the bare continuation.
- Native Anthropic advertised `:json_output` yet dropped `response_format`,
  `reasoning_effort` and `strict`; it now maps them. `model_context_window_exceeded`
  maps to `"length"`; a refusal carries no partial content, tool calls or captured
  blocks, and its `refusal_message` is the explanation Anthropic sends; a `200` without
  a content array or a `stop_reason` is an `LLMCallError`, not an empty success;
  `prompt_tokens` counts cache writes and `reasoning_tokens` comes from
  `thinking_tokens`; an assistant turn with neither text nor tool calls, which the API
  rejects, is left out of the next request.
- DeepSeek multi-turn tool loops failed with HTTP 400, because the earlier turns'
  `reasoning_content` was not sent back; the old default `deepseek-chat` had no price
  row, so its cost read `\$0`.
- Native Gemini: function calls under `MAX_TOKENS` or a safety filter read as
  `"tool_calls"`; a model turn with neither text nor function calls (a refusal, a turn
  spent on thinking) broke the follow-up request; a streamed refusal repeated its text
  once per trailing chunk. Gemini Interactions: a body without an `id` or `status` —
  non-streamed or as a streamed terminal — decoded as an empty success; streamed thought
  steps lacked their `summary`; streamed text parts were merged into one; `null`
  function arguments read as `"null"` (now `"{}"`).
- MCP client: a looping caller could starve a queued one (the session lock was not
  fair), a woken waiter that was interrupted stranded the callers behind it, a waiter's
  wait was unbounded, and an exception while a caller waited (a huge `timeout`, an
  interrupt) could wedge the session so that even `mcp_disconnect!` hung; the lock is
  now a FIFO hand-off. A stdio reply arriving after the request watchdog fired was
  returned as a success, and the timeout surfaced up to ~7 s after its bound; blank
  stdout lines read as EOF (a spurious `MCPCrashError`); a busy stdio server outlived
  its client process; Streamable HTTP treated SSE priming events as frames and ignored
  frames after the response; a concurrent `list_changed` could be lost to a refresh;
  and `mcp_disconnect!`'s `DELETE` (like a cancellation notice) was skipped inside a
  cancelled `with_cancel` scope.
- MCP server: registering a tool, resource or prompt while `serve` dispatched could
  corrupt the registry and crash the process; `serve(:http)` ran handlers on HTTP.jl's
  interactive pool, where a busy handler stalled the event loop; schema inference threw
  on `where` signatures and some unions; the by-name binding failed on `Symbol`,
  `Vector` and `Dict` parameters it advertised; a handler that printed corrupted the
  stdio protocol stream. The client reported version `0.8.0` in `clientInfo`.
- Realtime: an upgrade that completed just after `realtime_connect` had thrown
  `UniLMTimeout(:connect)` still ran the handler, and an open that failed just before
  the bound was reported as that timeout instead of its own error.
- `save_file_content` and `save_audio` truncated the destination before writing, so a
  failed write left a corrupt file. They now write a temporary file and rename it into
  place, keeping the destination's permission bits and writing through a symlink
  (dangling included); a directory destination throws `ArgumentError`.
- Platform: `image_data` returned `[]` for URL-delivered images; the poll helpers
  counted iterations instead of time, ended on the first transient failure and threw
  `InexactError` for `interval = 0`; `add_container_file` decoded the container-file
  object as a container; the fine-tuning event and checkpoint lists could not page.
- Azure read the deployment variable of `gpt-5.2` only; `AZURE_OPENAI_DEPLOY_NAME_<MODEL>`
  is read for every model, and a model without a deployment fails naming the variable
  instead of with a `KeyError`.
- The `Embeddings` missing-model error named `DataType` instead of the service.
- Documentation matches the code: examples that threw or no longer ran (the
  `chatrequest!(chat) do … end` form, which does not exist; examples that printed
  `output_text` of a failed result; the inferred-schema `register_tool!` example) are
  corrected, as are the claims on timeouts, retries, cancellation, cost accounting, MCP
  and the provider mappings.

## 0.19.0

### Breaking
- `ImageGeneration` no longer has an `input_fidelity` field: it is an image-edit
  parameter that `/images/generations` does not accept, so `generate_image` never
  sent anything the provider could act on. Set it on `ImageEdit` instead. Passing
  `input_fidelity=` to `ImageGeneration` or `generate_image` now throws
  `MethodError`, and the positional `ImageGeneration` constructor takes 11
  arguments instead of 12.

### Added
- GPT-6 Sol and Luna restrictions on the native OpenAI endpoint fail before a
  request is sent: Chat tool calling requires `reasoning_effort="none"` (as for
  GPT-5.6); unless reasoning effort is `"none"` (an omitted effort counts as the
  provider default, `"medium"`), `temperature`, `top_p`, `top_logprobs`, Chat
  `logprobs=true` and a Responses `include` of `"message.output_text.logprobs"`
  are refused; `prompt_cache_retention` is refused for every `gpt-6*` model.
  `encode_request` / `encode_agentic` throw `ArgumentError`; `chatrequest!` /
  `respond` return it as `LLMCallError` / `ResponseCallError`.
- `ModerationConfig` and a `moderation` field on `Chat` and `Respond` for OpenAI
  moderated completions: a required moderation `model` (for example
  `"omni-moderation-latest"`) plus an optional `"score"` / `"block"` policy mode
  for the input and output sides. A `respond` result carries the outcome in
  `r.response.raw["moderation"]`.
- `PromptCacheOptions` (now defined alongside `Chat`) gains two Responses-only
  options: `prewarm` prepares the prompt cache without generating output, and
  `comparison_response_id` requests prompt-cache diagnostics, returned in
  `r.response.raw["prompt_cache_diagnostics"]`. `Chat.prompt_cache_options`
  carries `mode` and `ttl` to Chat Completions for gpt-5.6 and later (older
  models answer 400: `gpt-5.4-mini` did on 2026-09-22); setting a Responses-only
  option on a native OpenAI `Chat` throws `ArgumentError`, and `mode="explicit"`
  on a `Chat` means no prompt caching because Chat messages cannot carry cache
  breakpoints.
- `input_text(text; cache_breakpoint=true)` marks an explicit prompt-cache
  breakpoint, and `configuration_update(; effort)` builds the GPT-6
  `configuration_update` input item that changes reasoning effort mid-conversation
  without rewriting the cached prefix.
- `FunctionTool` gains `async`, `allowed_callers` (validated: `"direct"` /
  `"programmatic"`), `defer_loading` and `output_schema`, and `function_tool`
  (dict and keyword forms) carries them. `ImageGenerationTool` gains `model`,
  `action`, `moderation`, `partial_images`, `input_fidelity` and
  `input_image_mask`; `action`, `moderation`, `partial_images` and
  `input_fidelity` are validated at construction. `output_schema` and
  `input_image_mask` accept any `AbstractDict`. Existing positional constructors
  keep working.
- Native Gemini structured output: `Chat(service=GEMINIServiceEndpoint,
  response_format=…)` sends JSON-object and JSON-Schema formats as
  `generationConfig.responseFormat`, and `respond(…; service=GEMINIServiceEndpoint,
  text=json_schema_format(…))` (or `json_object_format()`) sends the Interactions
  `response_format`. `GEMINIServiceEndpoint` now declares `:json_output`. Other
  shapes, including a set text `verbosity`, raise `ArgumentError`; the schema
  `name`, `description` and `strict` have no Gemini counterpart and are not sent.
- Native Gemini `Chat` sends `safety_identifier` as the request label
  `safety_identifier`, checked locally against Google's label rule (at most 63
  characters of lowercase letters, digits, `_` and `-`), so a 64-character hex
  digest throws `ArgumentError` before any request.
- Price rows for `gpt-6-sol`, `gpt-6-luna`, `gpt-5.4-nano`, `gemini-3.6-flash`,
  `gemini-3.5-flash-lite` and `gemini-embedding-2` (text input).

### Changed
- Gemini thinking levels are validated per model family against Google's
  thinking table, matching the longest family name: for example
  `gemini-3-pro-preview` accepts only `low`/`high` and
  `gemini-3.1-flash-lite-image` only `minimal`/`high`. Models outside the table
  keep the previous check.
- Gemini Interactions rejects a `FunctionTool` whose `async`, `allowed_callers`,
  `defer_loading` or `output_schema` is set, and native Anthropic rejects a `Chat`
  whose `moderation` or `prompt_cache_options` is set, instead of dropping the
  field silently.
- GPT-6 Astra accepts `logprobs=false` (only `logprobs=true` and `top_logprobs`
  are refused), matching the observed provider behaviour.
- The System One guide, API page and cost-tracking guide now execute their
  single-shot `ask` and `list_models` examples during the documentation build,
  like the OpenAI examples; a live documentation build renders real answers.
- Documentation: the `Reasoning` effort set is `none`, `minimal`, `low`,
  `medium`, `high`, `xhigh`, `max`; Responses `service_tier` lists `scale` and
  `ultrafast`; reusable prompts (`v1/prompts`) shut down on November 30, 2026;
  the Videos API and `sora-2` models shut down on September 24, 2026 (the
  wrappers stay and return provider errors from that date); image `quality`
  accepts `xhigh` and `max` on the 2.5 models and `gpt-image-2` and later accept
  arbitrary `WIDTHxHEIGHT` sizes; the model table adds GPT-6 Sol/Luna and
  `gpt-5.4-mini` and drops `o3`; the Agentic Workflows examples use
  `gemini-3.8-flash` (`gemini-3.1-flash-lite` shuts down May 7, 2027);
  `gemini-embedding-001` shuts down May 14, 2028 and `gemini-embedding-2` works
  through `GEMINIOpenAIServiceEndpoint`; the README notes that the Agents API and
  GPT-Live sessions are not wrapped.

### Fixed
- `estimated_cost` prices a versioned Jev id that has no row of its own (for
  example `jev-1.14.0`, including on `SystemOneSuccess` results) at the
  `jev-latest` rate instead of `0.0`; TypeSafe lists one price (Jev 1.13), and a
  later version is assumed to keep it until the pricing page says otherwise.
- Five documentation examples rendered misleading output (an empty block for the
  FIM failure accessors, a leaked Azure deployment entry across pages, a validity
  comment that contradicted its output, a struct dump after `push!`, and a
  compaction example on the flagship model); each now renders what its prose says.

## 0.18.0

### Added
- TypeSafe System One (Jev) client on `POST /v1/systemone`: `ask` evaluates every
  question against one ingestion of the state in a single request, with the three
  question primitives `choice` / `score` / `noul` and the matching typed answers
  `ChoiceAnswer` / `ScoreAnswer` / `NoulAnswer` (an unrecognised answer type is
  preserved as `UnknownAnswer` rather than dropped). Results are
  `SystemOneSuccess` / `SystemOneFailure` / `SystemOneCallError`; the accessors
  `answers`, `answer`, `getindex`, `haskey` and `keys` throw `SystemOneError` on a
  failed call instead of reading as an empty set of answers. `list_models` lists
  the aliases the account may name, and `token_usage` / `estimated_cost` price a
  call against the versioned model that answered — only input tokens are billed.
  The endpoint type is `TYPESAFEServiceEndpoint` (`TYPESAFE_API_KEY`,
  `TYPESAFE_BASE_URL`, `TYPESAFE_DEFAULT_MODEL`), which declares only
  `:system_one` and `:models`, so the chat, agentic and embedding verbs reject it
  up front.
- `SystemOneFailure.retry_after`: the wait a rate-limited or overloaded response
  asked for, in seconds, read off the final response from `retry-after-ms` (which
  wins, and keeps sub-second precision) or `Retry-After` (delta-seconds or an
  HTTP-date), and `nothing` when neither arrived or neither parsed. The retry seam
  already waits out `Retry-After` between attempts; the field is what a caller
  needs to schedule its own next attempt once `max_attempts` is spent, and
  `showerror` on a `SystemOneError` reports it.
- Natural-language control flow on top of that client. `@branch` is a `switch`
  whose cases are written in plain language: the option names become the criteria
  of one Choice question, only the selected body is evaluated, and a `_` line
  catches an answer below `min_confidence`. `nl"..."` lifts a description into the
  type domain as a `Meaning`, so a meaning is writable in an ordinary method
  signature; `nl_dispatch` resolves every `Meaning` slot of a generic function in a
  single request and calls the method Julia's own dispatch selects, leaving the
  remaining arguments to dispatch on their types. `meanings` previews the options
  exactly as they will be sent, and `LowConfidenceError` is raised when an answer
  falls below the threshold and no fallback was given.
- A "System One (TypeSafe Jev)" documentation section: a guide to typed judgments
  with Jev, a guide to multiple dispatch on natural language, and an API reference
  page.

### Fixed
- Auth-header masking indexed the result of its own re-match directly. `match`
  is typed `Union{Nothing,RegexMatch}` and a capture group `Union{Nothing,SubString}`,
  which Julia 1.13 surfaces as a type instability in the one renderer every
  `*CallError` passes through. Both are now guarded: an unmatched hit is handed
  back unchanged instead of being indexed.

## 0.17.0

### Breaking
- Default models are now `gpt-5.6-sol` for OpenAI chat and Responses,
  `gemini-3.8-flash` for Gemini, and `gpt-transcribe` for transcription.
  Set `model` explicitly to retain a previous model where the provider still
  offers it.
- Native Gemini 3.8 rejects unsupported sampling controls locally. Native Gemini Chat
  now rejects unmapped options instead of silently ignoring them. Remove
  `temperature` and `top_p` for Gemini 3.8, and use `reasoning_effort` to control
  thinking. Remove other unsupported options reported by validation.
- GPT-6 Astra requests validate unsupported reasoning and sampling options;
  tool callers receive guidance to use the Responses API.
- GPT-5.6 Chat tools require explicit `reasoning_effort="none"`; reasoning with
  tools uses `Respond` and `respond` instead.
- For GPT-5.6 and later, replace legacy `prompt_cache_retention` with typed
  `PromptCacheOptions`, passed through `Respond(prompt_cache_options=...)`.

### Changed
- Documentation builds and releases run examples offline by default. Live
  examples require the manual Documentation workflow's `live_examples` input.

### Added
- Provider availability notices for OpenAI video retirement and fine-tuning restrictions.
- OpenAI persisted reasoning (`Reasoning.context`), execution mode
  (`Reasoning.mode`), and validated `PromptCacheOptions` for GPT-5.6 and later.
- Gemini thinking control through `Chat.reasoning_effort` and `Respond.reasoning`,
  including Interactions thought summaries with `Reasoning(summary="auto")`.
- Typed transcription `languages` and `keywords`, with native multipart encoding
  and compatibility for a singular language hint.
- Current GPT-6, GPT-5.6, and Gemini 3.8 price estimates and inexpensive live
  coverage using GPT-5.6 Luna and Gemini 3.8 Flash.

### Fixed
- Tool loops preserve truncated or incomplete responses as unfinished results
  and do not execute partial function calls.
- Webhook verification rejects NaN and negative replay tolerances; only an
  explicit positive infinity disables the replay window.
- Gemini function schemas use `parametersJsonSchema`, allowing standard JSON
  Schema constraints that the restricted `parameters` field rejects.
- Streamed Gemini chat preserves all provider parts and thought signatures;
  thought summaries are excluded from answer text and in-band errors terminate.
- Interactions streaming preserves initial text in `step.start` and the order
  of text and tool steps, using per-step buffers for text accumulation.
- Deprecated `Reasoning.generate_summary` serializes as `summary`; conflicting
  aliases raise `ArgumentError`. Chat token limits reject nonpositive values.
- Cost estimates recognize dated model snapshots instead of silently reporting
  zero when only the corresponding base model is priced.
- Azure deployment registry: a registered value that did not carry the
  `/openai/deployments/` prefix every registration writes was silently encoded
  whole and re-prefixed, producing a request path nobody registered. The reader
  now throws an `ArgumentError` naming the malformed entry.

## 0.16.0

### Fixed
- Gemini Interactions streaming: an interaction whose terminal status is
  `"failed"` reported terminal completion, so the driver built a
  `ResponseSuccess` and dropped the wire's error and metadata. It now takes the
  typed-failure limb, and the response is overlaid on a copy of the raw
  interaction, so a streamed result keeps the same status/error/metadata/raw
  surface the non-streamed decode preserves.
- Gemini Interactions: a step with `content: null` — spec-legal for a
  `model_output` that produced no parts — threw a `MethodError` while decoding.
  It was user-visible on the non-streamed path and silently swallowed as a
  dropped SSE line on the streamed one. Null or absent content now decodes as no
  parts.
- Gemini Interactions streaming: a stream whose text deltas arrived in the same
  read as `interaction.completed` delivered **zero** deltas to the callback,
  because the terminal rebuild consumed the driver's text buffer. The rebuild
  now reads the buffer back without consuming it.
- MCP over HTTP: the transport's SSE reader required `data: ` with the space
  (rejecting spec-legal `data:foo`) and emitted one frame per data line,
  corrupting any event whose payload spans several `data:` lines. The body now
  goes through the shared SSE machine, cutting events at blank lines and joining
  each event's data lines with a newline.
- MCP over HTTP: notifications omitted the dual `Accept` header the request path
  sends and ignored the response status, so a server's rejection of
  `notifications/initialized` vanished and left a session the client believed was
  initialized. Both paths now share one header set and one status policy; a
  non-2xx notification fails loud, naming the status and body, which means a
  rejected `notifications/initialized` now fails `mcp_connect`.
- MCP over HTTP: connect-phase exchanges (handshake discovery, status
  `:initializing`) were bounded by `mcp_request_timeout` instead of
  `mcp_connect_timeout`, aborting handshakes the connect budget still covered. A
  breach there is now classified `:connect` and names that override.
- MCP over HTTP: `_mcp_notify!` reached the transport without the session lock;
  it now takes it re-entrantly, so notifications are serialized with exchanges.
- Multipart uploads: the request seam owns retries and passes `retry=false`,
  which also disables HTTP.jl's body rewind, so a single form handed to the retry
  loop was left consumed by the first attempt and the second put a zero-length
  multipart body on the wire — a transient 429/503 became a hard protocol
  failure. Non-replayable bodies now travel as a factory and are rebuilt per
  attempt, re-reading from source rather than buffering a copy. `edit_image` also
  validates its file paths before the loop, so a missing file is reported once, up
  front.
- `Retry-After` in its HTTP-date form is now honored. RFC 7231 allows
  delta-seconds **or** an HTTP-date and servers behind CDNs send both; only the
  seconds form was parsed, so the date form was silently ignored and the client
  under-waited a 429 storm by whatever the header actually asked for. The parsed
  date goes through the same clamping and deadline arithmetic as the seconds
  form: a past date yields zero, and a malformed value still yields the default
  backoff — a server sending garbage must never make a client throw.
- Decoders no longer fabricate text for an empty turn. The OpenAI, Anthropic and
  Gemini decoders each substituted `"No response from the model."` whenever a turn
  produced no text — content the provider never sent, which then landed in the
  returned `Message` and in the next request's history. Reasoning models hit this
  routinely, spending the whole completion budget on thought tokens. An empty turn
  is now reported as the empty turn it is. A response with no candidates or
  choices at all remains a different thing — there is no assistant turn to report
  — and still fails loud into the verb's typed error result.
- OpenAI-wire tool calls are now decoded whenever they are present, whatever the
  finish reason. A tool-only reply finishing with anything other than
  `"tool_calls"` — which some providers do — previously fell into the empty-turn
  fallback, which fabricated prose *and* dropped the calls.
- MCP server: survives spec-legal parameter shapes (`params: null` and positional
  `params` no longer crash a request). Errors below the tool handler now answer
  JSON-RPC `-32603` with a generic message, with the exception and backtrace
  logged locally rather than shipped to the peer, since an exception string can
  carry file paths and argument values a remote client has no business reading.
  Frames and request bodies are capped at 16 MiB on both transports and an
  oversized one is answered rather than parsed — parsing an attacker-sized payload
  allocates a multiple of it, which is an out-of-memory kill rather than a
  protocol error. Tool-handler exceptions are unaffected: they still reach the
  client as tool results (`isError`), which is what lets a model correct itself.
- Embeddings: a response that misses or duplicates a row index is now an
  `EmbeddingCallError` instead of a zero vector. The buffers are pre-zeroed, so an
  uncovered slot stayed a valid-looking all-zero embedding that still compares,
  normalizes and indexes — a silent corruption. A response must now cover every
  input exactly once.
- `save_image` decodes the payload **before** opening the destination. `open(…,
  "w")` truncates, so decoding inside the block let a malformed payload destroy
  whatever already lived at the path and leave a 0-byte stub. A failed save now
  costs nothing.
- The Azure model-to-deployment registry is now lock-guarded. It is read while
  building every Azure request URL and written by `add_azure_deploy_name!`,
  potentially from another task; a `Dict` write that rehashes reallocates the
  arrays a reader is walking, which yields a wrong hit or a bounds error rather
  than merely a stale answer.
- The keyword form of `set_default_config!` is now atomic. It is a
  read-modify-write on the process default, so two concurrent calls read the same
  snapshot and the second write installed a state predating the first, silently
  dropping its field. The merge now runs in a compare-and-swap loop.
- API keys and request bodies are redacted before they reach a result value. A
  mid-exchange transport failure on HTTP.jl 1.x arrives as an error whose
  rendering is a full request dump — every header and the whole body — and the
  library masks only `Authorization`, `Proxy-Authorization` and `Cookie` there, so
  a provider authenticating with its own header (Anthropic `x-api-key`, Gemini
  `x-goog-api-key`, Azure `api-key`) had its key in cleartext inside the
  user-visible `.error` of the returned result, and from there in logs and bug
  reports. Every error-construction site now renders through one helper that
  prefers the root cause's own text and then masks auth-shaped header values as
  defense in depth. The same redaction covers debug/warn logs, the displayed form
  of MCP transports and sessions, and a minted Realtime client secret. Call-error
  results additionally define a `show` that names `cause` by TYPE instead of
  letting Julia's default recurse into it and reprint the dump; `.cause` itself is
  unchanged and still reachable.
- The auth-header mask is bound to header position. The pattern had no leading
  word boundary, so any longer word merely *ending* in a header name was matched
  and everything after it erased — a message containing `reauthorization: <text>`
  lost the text. Fail-safe, but destructive of diagnostics that never carried a
  credential. The real names (`authorization`, `proxy-authorization`, `api-key`,
  `x-api-key`, `x-goog-api-key`) are still masked in both the wire form and the
  Julia pair form.
- The MCP HTTP server answers a bodyless `POST` with the same `-32700` JSON-RPC
  parse error a malformed body gets, instead of crashing the handler into a 500.
  On HTTP.jl 2.x a payload-less request carries a typed empty body that supports
  no `length`; body size is now read through a shape-dispatched helper on both
  majors.
- Caller-supplied values interpolated into request URLs are percent-encoded. Ids,
  model names, pagination cursors and list filters went into path segments and
  query values raw across the platform APIs (files, batches, uploads, containers,
  conversations, vector stores, fine-tuning, videos), the Responses lifecycle
  operations (response ids, pagination cursors and filters), the Azure deployment
  path, the Gemini model-in-URL and the Realtime model query. A value carrying `/`, `?`, `#`, `&`, `=` or a space
  silently re-shaped the request target — extra path segments, a spurious query
  string or fragment, a smuggled parameter — or, on HTTP.jl 2.x, was rejected
  client-side as a synthetic 400 that never reached the server. Only the
  interpolated value is encoded; the templates' own separators and Gemini's
  `:generateContent` verb colon stay literal. Ordinary identifiers draw from the
  unreserved set, so the common case is byte-identical on the wire.
- `edit_image` follows the declared-endpoint capability rule the four primary
  verbs use. It validated unconditionally, so a user-defined endpoint — one that
  declares no capabilities, because the package does not ship it — raised a raw
  `MethodError` instead of dispatching. An endpoint that declares nothing now
  passes through to the wire; one that declares its capabilities and omits image
  edits still gets the typed `ArgumentError`.
- Agentic streaming no longer costs O(n²) in delivered bytes. Emitting a delta
  meant `take!`-ing the whole accumulated text plus a mirror of what had already
  been sent, slicing the tail and re-printing both — four copies of the full text
  per read. Deltas are now drained from a pending buffer the decoders append to,
  matching the chat driver. Measured on the emission loop at 200/400/800/1600
  reads: 0.98/3.4/12.7/49.4 MB allocated before, 33/63/129/261 KB after, with
  byte-identical output. Callers see the identical delta sequence.

### Added
- `sse_dropped::Int` on `LLMSuccess`, `LLMFailure`, `ResponseSuccess` and
  `ResponseFailure`. A `data:` payload the parser cannot read is dropped rather
  than re-queued — a re-queued partial line is the stream-poisoning mechanism the
  drop policy exists to prevent — but the only trace was a process-global counter
  no caller could attribute to its own request, so a turn assembled from a
  truncated wire was indistinguishable from a clean one. Every drop is now counted
  on the stream that saw it and the count rides the result. It defaults to `0`, so
  a non-streamed call reports `0` rather than nothing. Both drivers state a
  non-zero count once per attempt at finalize, naming the model and the surface —
  a warning, since an undecodable provider payload is anomalous by construction,
  not the per-line debug message. A retried attempt starts from fresh state and
  reports only its own. On a truncated stream (HTTP 200, no terminal event) the
  `LLMFailure` now carries the count that explains why no message could be built.

### Changed
These entries change behavior or public surface that callers may depend on. Most
turn a silently-succeeding path into a typed exception; the rest change a result
type or drop a field. They are collected here pending a version decision.

- `Respond` fields the Gemini Interactions wire does not map now throw
  `ArgumentError` at encode time instead of being silently dropped. Structured
  output via `text`, `reasoning`, `metadata` and others simply vanished between
  the caller's request and the wire. The unmapped set is derived from
  `fieldnames(Respond)`, so a newly added field is unsupported until deliberately
  mapped, and the error names the offending fields.
- `is_flagged` throws `ArgumentError` on a failed moderation call. A failed call
  has no verdict, and returning `false` made "not flagged" indistinguishable from
  "never checked".
- `verify_webhook` throws `ArgumentError` for a signing secret that is not valid
  base64, and rejects non-finite timestamps, instead of failing verification as
  though the signature were wrong. A malformed secret is a configuration error,
  not a failed verification.
- `token_usage` and `estimated_cost` throw `ArgumentError` for result types
  outside the token-billed APIs (audio, batch, files, moderations, vector stores,
  video, …), where they previously raised `MethodError`. Those calls report no
  token usage at all, and a `0.0` would be indistinguishable from a genuinely free
  call.
- `mcp_disconnect!` now takes the session lock, so a disconnect racing a call in
  flight waits for that exchange to finish instead of tearing the transport down
  under its reader — the same concurrency-1 semantics every other call obeys,
  bounded transitively by the exchange's own `mcp_request_timeout`.
- **`HTTPTransport` no longer has a `lock` field.** Nothing ever took it.
  Exchanges, notifications and the disconnect are all ordered by the session lock,
  which is the only ordering an HTTP transport needs, so the field was a public
  promise the type never kept. Code reading `transport.lock` must drop it.
- `realtime_connect` and `realtime_receive` are bounded by `RequestConfig`, and
  `RealtimeSession` gained a `config` field carrying the budget resolved at
  connect time. The WebSocket was the one surface outside the bound-everything
  guarantee: a peer that accepted TCP and never completed the upgrade, or a
  connected peer that went silent, blocked the caller forever. `connect_timeout`
  now bounds **only** the open phase and `stream_idle_timeout` bounds
  `realtime_receive`; a live session's lifetime stays deliberately unbounded,
  since it is the caller's to decide. The two-argument `RealtimeSession`
  constructor is retained and inherits the ambient config.
- A non-streamed generation whose terminal status is `"failed"` is now a
  `ResponseFailure` on both agentic wires, carrying the wire body verbatim so
  error and metadata are preserved. The streamed limb already did this, so
  `issuccess` depended on how the call was made rather than on what happened.
  Only `"failed"` flips: `cancelled`, `expired`, `in_progress`, `queued` and
  `requires_action` are legitimate terminals of the background and tool-action
  flows and remain successes.
- A **streamed** generation whose terminal event is `response.incomplete` is now a
  `ResponseSuccess`, not a `ResponseFailure`. The streamed driver routed that
  terminal to the typed-failure limb while the non-streamed path decoded the very
  same object as a success, so whether a truncated generation had failed depended
  on `stream` rather than on what happened. `response.incomplete` carries a real
  terminal response object with usable partial output, and the truncation is
  reported in `status` and `incomplete_details`; the streamed limb now finalizes
  it exactly like `response.completed` — same result type, same status and
  details, usage recorded, raw capture complete, deltas already delivered. Only
  `"failed"` is a failure, on both paths. The typed-failure limb still catches an
  `incomplete` terminal carrying no response object at all, which has no result to
  hand back. Code that tested `result isa ResponseFailure` to detect a truncated
  stream must read `response_status`/`incomplete_details` instead.
- The four primary verbs — `chatrequest!`, `embeddingrequest!`, `respond` and
  `generate_image` — now validate provider capabilities before dispatch, giving a
  typed `ArgumentError` instead of a provider 404. Validation applies only to
  endpoints that **declare** their capabilities (`respond` accepts either
  `:responses` or `:agentic`, since the two agentic wires name the same surface
  differently). A custom endpoint that declares nothing passes through
  unvalidated: refusing to dispatch a backend the package knows nothing about
  would be a false negative.
- `call_tool` and `get_prompt` accept any `AbstractDict` for `arguments`, so the
  natural `Dict("k" => "v")` form works without conversion.

## 0.15.1

### Fixed
- Streaming: the byte-gap idle watchdog read the clock before loading the
  last-byte stamp, so a chunk arriving between the two reads made the unsigned
  gap wrap to an astronomically large value and killed a healthy,
  actively-streaming connection with a spurious `:stream_idle` timeout. The gap
  is now computed stamp-first and saturating. The probability per check was
  tiny, but it scaled with concurrent streams, so sustained high-concurrency
  streaming workloads would eventually hit it.
- Streaming: a completed turn now survives connection-teardown noise on both
  streaming surfaces. Previously, once the terminal event had been delivered, a
  transport error raised while the socket wound down (a reset from the peer, a
  framing fault on the read that would have carried the end-of-stream sentinel)
  could either discard the finished generation (surfacing a call error after
  the user's callback already received the result) or — when no callback was
  registered — re-send the entire request, billing a second generation for one
  call. The recorded outcome now stands: finalization and the terminal callback
  are structurally at-most-once, teardown noise after completion neither
  discards nor re-POSTs, and the chat surface also recognizes the provider's
  recorded finish reason as completion when the connection dies on the very
  read that would have carried the closing sentinel. Failures that are not
  teardown-shaped (a throwing user callback, a decoding defect) surface
  exactly as before.
- Streaming: a watchdog kill that lands while the driver is inside a user
  callback truncates the socket read into a clean end-of-stream, so the read
  loop ended without an exception and the kill was reported as a status-200
  failure carrying partial bytes (or, with a recorded finish reason, as a
  silently truncated success). Both drivers now consult the guards on the
  no-exception exit path and surface the same typed `:stream_idle`/`:request`
  timeout the throwing path produces.
- Streaming: when the request-phase bound fires at the same instant the guarded
  call completes, the watchdog closes the socket without anything unwinding;
  the next read then raised a bare `IOError` that classified as a retryable
  transport failure — one extra billed wire attempt and a lost phase. The
  fired bound is now detected from guard state and surfaces as the typed
  `:request` timeout.
- Responses/Interactions streaming: a terminal event arriving as the very last
  bytes of the stream without a trailing newline was never dispatched (the
  chat surface already handled this), so a completed response surfaced as a
  status-200 failure. The agentic read loop now feeds the terminating newline
  at end-of-stream.
- Streaming on HTTP 2.x with `stream_idle_timeout=Inf`: nothing bounded the
  response-header wait (the request-phase watchdog's close is inert before
  response headers exist on that major), so a mute peer stalled the stream
  task indefinitely. The header wait is now natively capped at the effective
  request bound whenever the idle bound is disabled, and breaches surface as
  typed `:request` timeouts.
- `SystemError` now classifies as a connection-level transport error: a peer
  reset can surface as a raw `SystemError("read", ECONNRESET)` rather than an
  `IOError`, and previously was neither retried nor mapped to a typed cause.
- MCP: `mcp_request_timeout` now bounds a call's own exchange, measured from
  the moment it acquires the session lock — not from the moment it asked. A
  caller queued behind a concurrent exchange on the same session previously
  burned its bound while waiting and, on breach, group-killed the server in
  the middle of the holder's healthy exchange, which then surfaced to the
  holder as a fabricated server-crash error.
- MCP: the liveness check, auto-respawn, and the exchange now run under one
  session-lock acquisition. Two concurrent calls on a session closed by a
  timeout or crash previously could both observe it closed and both respawn,
  spawning two server processes — one of them orphaned beyond the kill
  ladder — and interleaving their handshakes on shared session state.
- MCP: once a server process is spawned, any connect or respawn failure now
  tears it down before the error propagates. Previously only timeout- and
  crash-shaped failures did (a stdout banner that breaks JSON-RPC framing, or
  an `initialize` error reply, leaked a live child process holding the
  session's pipes), and a failed respawn attempt relabelled every recorded
  close cause as a crash; the recorded cause is now preserved unless the
  attempt itself diagnosed a new one.
- Streaming: when the request-phase bound (first-byte deadline) closes a mute
  connection, the typed `UniLMTimeout` is now recorded before it unwinds
  through HTTP.jl and is restored as the surfaced cause. Previously — observed
  on the HTTP 1.x major under CI load — the library's teardown of the closed
  socket could raise its own transport error (for example a broken-pipe
  `IOError` from writing the terminating chunk), which replaced the in-flight
  typed exception, so the result carried a raw transport error as its cause
  instead of the promised `UniLMTimeout`. Transport errors that arrive with no
  bound fired (for example a refused connection) classify exactly as before.
  Chat streaming and the Responses streaming surface share the fix.

### Added
- `test/load_probe.jl`: standalone, opt-in concurrency/load harness (not part
  of `Pkg.test`) driving hundreds of concurrent mixed streaming and
  non-streaming calls against local mock providers, with cross-talk markers,
  descriptor/task-leak plateaus, retry-storm and teardown-noise batches.

## 0.15.0

### Breaking
- `push!(chat, msg)` and `pop!(chat)` now throw `InvalidConversationError` on an
  invalid mutation (a conversation not started by a system message, a system
  message after the start, consecutive same-role non-tool messages, or popping an
  empty conversation) instead of emitting a warning and returning the chat
  unchanged. The error message names the violated rule and the offending role and
  never includes conversation content or endpoint configuration. Valid mutations
  are unchanged.

### Breaking (provider extension)
- The OpenAI-wire request/response encoding and SSE handling are now an explicit
  opt-in via the new exported `OpenAIWireEndpoint <: ServiceEndpoint` supertype. A
  `ServiceEndpoint` subtype that does not implement the wire seam
  (`encode_request` / `decode_response` / `handle_sse_event!`, the agentic
  encode/decode/stream defaults, and `_agentic_url` routing) now fails with a
  `MethodError` at call time instead of
  silently emitting OpenAI-shaped requests at a foreign API. Migration: subtype
  `OpenAIWireEndpoint` instead of `ServiceEndpoint` for OpenAI-compatible
  backends; backends with a genuinely different wire keep subtyping
  `ServiceEndpoint` and implement the seam (see the new Custom Backends guide).
  All built-in endpoints are unaffected.

### Added
- `OpenAIWireEndpoint` abstract supertype (exported) and a "Custom Backends"
  guide documenting the provider-extension contract.
- A "Versioning & Stability" policy page (what 0.x means here, breaking-change
  batching, alias guarantees, 1.0 intent) plus a "When to choose UniLM"
  positioning section in the README.
- Result-consumption helpers: `issuccess` / `isfailure` for any request result,
  `text(::LLMSuccess)` returning the reply content (`nothing` for tool-calls-only
  turns), and `LLMResultError` — thrown by `text` on a `LLMFailure`/`LLMCallError`;
  its `showerror` reveals only the status and a trimmed response excerpt.
- Cost accounting: pricing entry for `gemini-3.7-flash` (introductory rate through
  2026-12-31; the output rate includes thinking tokens). The live Gemini suites now
  pin this model.
- "Release gate" workflow: on `release/**` branches and manual dispatch, runs a JET
  whole-package type-stability analysis and then the full test suite. JET is
  expensive, so it gates releases instead of running in routine push/PR CI; the
  workflow sets no provider keys and makes no billed calls. The analysis is its own
  step in its own process, ahead of the suite and without coverage instrumentation,
  so it reports on the package exactly as `using UniLM` leaves it — not on a session
  the test files have extended with mock endpoints and test-only method overloads,
  which widen dispatch and change what inference concludes about package code. The
  step fails on any package-code finding. The same analysis stays available inside
  the test suite for local use behind `UNILM_RUN_JET=true`, where it likewise runs
  ahead of the behavioural tests.

### Changed
- Type-strengthening pass driven by the new JET whole-package release gate: over-wide
  internal seam signatures were narrowed and URL-construction inference made concrete,
  bringing JET to zero findings. One user-visible effect: calling a platform API with a
  native (non-OpenAI-wire) endpoint now raises a descriptive `ArgumentError` instead of
  a bare `MethodError`.
- Streaming keep-alive test margins widened for loaded CI runners (test-only).
- Tool-surface types renamed to provider-neutral canonical names: `GPTTool` →
  `Tool`, `GPTToolCall` → `ToolCall`, `GPTFunctionSignature` →
  `FunctionSignature`, `GPTFunctionCallResult` → `FunctionCallResult`.
  Non-breaking: the former `GPT*` names remain exported as `const` aliases —
  construction, dispatch, `isa`, field access, and the parametric
  `GPTFunctionCallResult{T}` all keep working unchanged. The `GPT*` aliases are
  scheduled for removal at 1.0 per the new stability policy; docs teach the
  neutral names as canonical.
- `GenericOpenAIEndpoint` and `DeepSeekEndpoint` now redact `api_key` when shown: a
  short prefix plus a `…[redacted]` marker (never the full key or its length),
  inherited when an endpoint prints nested inside a `Chat` or a result value.
- Aqua's method-ambiguity check is enabled in the test suite (`ambiguities=true`)
  after measuring zero ambiguities in the package.

### Fixed
- Streaming: a byte-gap idle breach is now always classified as
  `UniLMTimeout(:stream_idle)` — non-retryable — on the HTTP.jl 2.x major.
  Previously the classification depended on when the breach fired: HTTP 2.x also
  applies its native `read_idle_timeout` (armed at `stream_idle_timeout`) to the
  response-header wait, so a breach landing before the first byte was reported as
  a retryable `:request`-phase timeout with `request_timeout` as its limit, and a
  mute stream could be silently retried. Whether real runs hit that window is
  timing-sensitive (newer HTTP 2.6.x releases changed server task scheduling,
  flipping it under load); the classifier now decides from which native timers
  the streaming seam arms, so the phase is deterministic. Chat streaming and the
  Responses streaming surface share the fix; HTTP 1.x behavior (idle-guard
  enforcement only) is unchanged.

### CI and docs
- The test environment admits JET 0.12 alongside 0.11 (`JET = "0.11, 0.12"`), keeping
  the whole-package release gate on the current JET release; 0.12 is the first series
  to support Julia 1.13. The gate still measures zero findings.
- The CI matrix, documentation build, and release gate track the latest stable Julia
  release instead of a pinned 1.12; the supported floor stays `julia = "1.12"`.
  CompatHelper now also sweeps the `test/` and `docs/` environments, so their bounds
  cannot go stale unnoticed.
- Pull-request documentation builds run without provider API keys — examples render
  offline — eliminating live-API spend on PRs; push, tag, and manual builds still
  render live outputs. The routine CI test workflow no longer receives provider
  keys at all.
- Live integration suites are explicitly opt-in: set `UNILM_LIVE=1` in addition to
  the provider key (the live MCP suite keeps its own `UNILM_LIVE_MCP=1` gate).
- Executed chat and agentic documentation examples that make live calls are pinned
  to each provider's cheapest model (`gpt-5.4-mini`, `gemini-3.1-flash-lite`,
  `claude-haiku-4-5`); image-generation examples are static code blocks that reuse
  the committed sample image, so docs builds never bill for image generation.
  Token-counting, response-compaction, embeddings, and FIM examples keep their API
  defaults.

## 0.14.0

### Breaking
- Abrupt stdio MCP server death (killed, crashed, or broken pipe) now surfaces the
  new typed `MCPCrashError` and closes the session, instead of leaking a raw
  `Base.IOError`/`ErrorException` while the session stayed `:ready`. Code that
  caught raw transport errors around MCP calls should catch `MCPCrashError`.
- `mcp_connect` itself now throws `MCPCrashError` (instead of a raw transport
  error) when the server dies during the connect handshake.

### Added
- `MCPCrashError` (exported): message with recovery guidance plus best-effort
  `exitcode`/`termsignal` diagnostics.
- `auto_respawn=true` now also respawns after a server crash (previously hangs
  only); the crashing call always throws, the next call heals the session.

## 0.13.0

### Added
- `RequestConfig` and four resolution channels bound and tune every request.
  Fields (all `Float64` seconds unless noted; `Inf` disables a bound; the
  constructor rejects `NaN` and non-positive values): `connect_timeout` (10.0),
  `request_timeout` (600.0, whole non-streaming exchange), `stream_idle_timeout`
  (120.0, byte-gap between raw stream chunks), `total_deadline` (900.0, across
  all retry attempts; for streams, until the first byte), `max_attempts` (`Int`,
  3), `mcp_connect_timeout` (120.0), `mcp_request_timeout` (120.0). Resolve a
  config, in precedence order, via the per-call `config=` keyword on every
  request verb, a `with_request_config(f; kwargs...)` dynamic scope (propagates
  into `Threads.@spawn`), a process default set by `set_default_config!(cfg)` /
  `set_default_config!(; kwargs...)`, or the built-in defaults. `current_config()`
  returns the config in force. `RequestConfig(base; kwargs...)` copies with
  overrides.
- Typed timeout errors: `UniLMTimeout` (`phase` ∈ `:connect`/`:request`/
  `:stream_idle`/`:deadline`, with `elapsed`/`limit` seconds) and
  `MCPTimeoutError` (`phase` ∈ `:connect`/`:request`, whose message names the
  applicable per-connect/per-call override). Value-returning surfaces carry the
  timeout on a new `cause::Union{Nothing,Exception}` field of their call-error
  result (`LLMCallError`, `EmbeddingCallError`, `ResponseCallError`, …) with
  `status = nothing`; no fabricated HTTP status is invented for a timeout.
- `mcp_connect(...; auto_respawn=false)` and a call-time `timeout` keyword on
  `call_tool` / `list_tools!`. The request-phase timeout resolves at call time
  (explicit `timeout` > ambient `with_request_config` scope > the config captured
  at `mcp_connect`), so tools bridged into a tool loop — which pass no keywords —
  still honor an ambient scope.
- `tool_loop(input::String; tools, …)`: a no-dispatcher convenience method that
  wraps the prompt and `tools` in a `Respond` and runs the Responses-API tool
  loop, so `tool_loop("…"; tools=mcp_tools_respond(session))` works as a
  documented one-liner. `max_turns` drives the loop and `config` bounds each
  underlying request; every other keyword is forwarded to the `Respond`
  constructor (an unknown one raises).

### Changed
- **Breaking:** the `retries` keyword is removed from every request verb
  (`chatrequest!`, `embeddingrequest!`, `respond`, `fim_complete`,
  `prefix_complete`, `generate_image`, `edit_image`, `upload_file`, and the
  `tool_loop!` / `tool_loop` verbs). Retry behavior is now governed by
  `RequestConfig.max_attempts` (default `3`). Migrate by intent: code that passed
  `retries=30` — the real disable switch, since `retries=N` meant "N attempts
  already spent" and `30` hit the ceiling — now passes
  `config=RequestConfig(max_attempts=1)`; code that passed `retries=0` or nothing
  (the old default seed, which allowed up to thirty attempts) needs no change and
  takes the new default (`max_attempts=3`). There is no compatibility alias: the
  old `retries` counted toward a 30-attempt ceiling (`retries=30` disabled
  retries, `retries=0` allowed thirty) — the inverse of an attempt count — so any
  silent shim would invert real call sites. A removed keyword raises `MethodError`.
- **Breaking:** requests now carry bounded default timeouts, so an operation that
  previously blocked forever on a silent or stalled peer instead fails with a
  typed error once its bound elapses. Defaults (all overridable per call, per
  scope, or process-wide via `RequestConfig`; set any field to `Inf` to disable
  it): connect `10s`, whole non-streaming request `600s`, stream byte-gap idle
  `120s`, total across retries `900s`, `max_attempts` `3`, MCP connect `120s`,
  MCP request `120s`. Non-streaming timeouts land inside the existing error
  result (`status = nothing`, timeout on `cause`); streaming timeouts surface when
  the returned `Task` is `fetch`ed.
- **Breaking:** an MCP **stdio** request that exceeds `mcp_request_timeout` now
  closes the session (`status = :closed`) and throws `MCPTimeoutError`. Stdio
  framing has no response-id demultiplexing, so a late reply could be misdelivered
  to the next caller as a fabricated result — tearing the session down is the only
  safe response. A subsequent call raises unless the session was opened with
  `mcp_connect(...; auto_respawn=true)`, which respawns the server (fresh
  handshake, logged with `@warn`) and retries the call once; in-memory server
  state is lost on respawn, so respawn is opt-in. MCP **HTTP** request timeouts
  are not session-fatal (request/response correlation is per-POST).
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
- `Chat(; tools=…)` now accepts a `Vector{<:CallableTool}` and stores the
  unwrapped inner `GPTTool`s, so MCP tools bridge in directly —
  `Chat(tools=mcp_tools(session))`, with no manual `map(t -> t.tool, tools)`.
  The `tools` field type is unchanged (`Union{Vector{GPTTool},Nothing}`);
  conversion happens only at construction.

### Fixed
- A user interrupt (`InterruptException`, e.g. Ctrl-C) raised during a chat
  request or inside a streaming task now propagates instead of being laundered
  into a call-error result and retried. Every request catch layer rethrows an
  interrupt first; a streamed interrupt surfaces as a `TaskFailedException` when
  the returned `Task` is `fetch`ed. (Same class as the tool-loop interrupt fix
  below.)
- Azure OpenAI deployment names configured through `AZURE_OPENAI_DEPLOY_NAME_*`
  environment variables are now read when the request URL is built, instead of
  being captured once when the package loads. A deployment name exported after
  `using UniLM` is honored; `add_azure_deploy_name!` registrations still take
  precedence.
- The tool-calling loop no longer swallows a user interrupt: an
  `InterruptException` (e.g. Ctrl-C) raised inside a tool function now
  propagates and aborts the loop instead of being recorded as a tool-call
  failure and retried. All other exceptions still become tool-error outcomes.
- Tool-error messages now reach the model faithfully instead of wrapped in
  exception-constructor noise. A tool that raises `error("kaboom")` is reported
  to the model as `Error: kaboom` (previously `Error: ErrorException("kaboom")`);
  other exceptions render through `showerror` (e.g. `KeyError: key "x" not found`)
  rather than their `string(e)` constructor form. The single loop-level `Error: `
  prefix is unchanged, so a tool whose own message already begins with `Error:` is
  still not rewritten.
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
