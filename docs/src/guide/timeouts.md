# [Timeouts & Retries](@id timeouts_guide)

```@setup timeouts
using UniLM
```

Every UniLM operation — a provider request, a stream, an MCP exchange — waits on a
peer only under a bounded, configurable limit, and reports a breach as a **typed**
error. This guide covers the bounds, how to change them, the retries they bound, and
the typed failures you get when one fires.

```@example timeouts
service, model = OPENAIServiceEndpoint, "gpt-5.4-mini"
nothing # hide
```

The examples use only the names bound above. `service` and `model` are the provider and model they run
on: bind your own there. The table [at the end](@ref timeouts_providers) links each provider's rules.

## [The one config struct](@id timeouts_config)

All bounds live on a single [`RequestConfig`](@ref), resolved per call on every
request verb (and captured at connect time by an MCP session):

| Field | Default | Bounds |
|---|---|---|
| `connect_timeout` | `10.0` s | Establishing the connection, per attempt. |
| `request_timeout` | `600.0` s | The whole non-streaming exchange, per attempt; a stream's exchange up to its first byte. |
| `stream_idle_timeout` | `120.0` s | Byte-gap between raw stream chunks (see [Streaming](@ref timeout_streams)). |
| `total_deadline` | `900.0` s | Across **all** attempts including backoff; for streams, until the first byte. |
| `max_attempts` | `3` | Total attempts (1 = no retry). |
| `mcp_connect_timeout` | `120.0` s | MCP spawn → `initialize` handshake complete. |
| `mcp_request_timeout` | `120.0` s | One MCP request/response exchange. |

All timeout fields are seconds (`Float64`). Set any of them to `Inf` to disable
that bound. The constructor **rejects** `NaN`, non-positive values and finite
values above `1e9` seconds (use `Inf` to disable a bound) with an `ArgumentError`,
and requires `max_attempts ≥ 1` — a silently-`NaN` bound would compare false
against every check and reintroduce an unbounded wait, and a finite value too large
for a timer would fail deep inside a call instead of at construction.

```@example timeouts
cfg = RequestConfig(request_timeout = 120.0, max_attempts = 5)

# Copy-with-overrides — change one field, keep the rest:
patient = RequestConfig(cfg; total_deadline = 3600.0)

# Disable the stream idle bound entirely (never idle-kill a stream):
no_idle = RequestConfig(stream_idle_timeout = Inf)

(cfg.request_timeout, patient.total_deadline, patient.max_attempts, no_idle.stream_idle_timeout)
```

### [Why these defaults](@id timeouts_defaults)

The defaults mirror those of the official provider SDKs the Evidence note below names, with headroom for
HTTP.jl's heavier connect path:

- **`request_timeout = 600s`** matches their 10-minute request timeout.
- **`max_attempts = 3`** matches their two automatic retries (two retries
  after the first attempt = three attempts).
- **`connect_timeout = 10s`** is 2× the SDKs' 5s connect, headroom for HTTP.jl's
  heavier connect; a false connect-timeout costs one retry and has no side
  effects.
- **`stream_idle_timeout = 120s`** is stricter than the reference SDK stack's
  600s streaming read timeout. It is provisional, backed by the per-provider gap
  measurement under [Measured stream gaps](@ref timeouts_gaps) below, and the per-call override is
  the escape hatch for unusually silent
  reasoning streams.
- **`total_deadline = 900s`** and the **`120s` MCP** bounds cap cold starts (a
  cold `npx` MCP server on a slow link can legitimately approach the connect
  bound — the timeout message names the override to raise).

!!! details "Evidence: the SDK defaults"
    Read on 25 October 2026 (UTC) from the default branches of the SDKs' repositories: the Python SDKs of
    OpenAI and Anthropic set
    `DEFAULT_TIMEOUT = httpx2.Timeout(timeout=600, connect=5.0)` (Anthropic's as `10 * 60`) and
    `DEFAULT_MAX_RETRIES = 2` in their `_constants.py`; OpenAI's Node SDK
    says "Requests time out after 10 minutes by default" and "Certain errors will be automatically retried
    2 times by default" (its README).

### [Measured stream gaps](@id timeouts_gaps)

The `120s` idle default is provisional and backed by a measurement
(`scripts/measure_stream_gaps.jl`) of the maximum inter-byte gap per provider —
the exact quantity the idle guard watches, which keep-alive bytes (SSE comments,
a provider's ping events) reset. Both measured maxima sit well under the `60s`
raise-the-default threshold. A healthy gap above `60s` in a future measurement
raises the default.

!!! details "Evidence: the stream gaps measured on 2026-07-18"
    Measured with `scripts/measure_stream_gaps.jl` on 2026-07-18, on models that are no longer the
    defaults:

    | Provider | Model | Max inter-byte gap | Headroom vs 120s |
    |---|---|---|---|
    | OpenAI (reasoning) | `gpt-5.5` | 8.5s (504 chunks, 16.1s stream) | 14× |
    | Anthropic (extended thinking) | — | not measured this release | — |
    | Gemini | `gemini-3.5-flash`, routed to `gemini-3.6-flash` since 2026-10-08 ([Models](@ref gemini_models)) | 15.9s (55 chunks, 19.8s stream) | 7.5× |

    The Gemini stream held a healthy ~16-second silent gap mid-thinking — the class
    of pause the byte-gap default must tolerate.

## [The four channels](@id timeouts_channels)

A config is resolved, in precedence order:

1. the per-call `config=` keyword on any request verb,
2. a `with_request_config` dynamic scope,
3. the process default (`set_default_config!`),
4. the built-in defaults above.

`current_config()` returns whatever is in force right now.

### [1. Per call](@id timeouts_per_call)

```julia
chatrequest!(chat; config = RequestConfig(request_timeout = 60.0))
respond("summarize this"; service, model, config = RequestConfig(max_attempts = 1))
```

### [2. A dynamic scope](@id timeouts_scope)

`with_request_config` merges the given keywords over the current config for the
duration of the block, and propagates into tasks spawned inside it (including the
task a streaming call runs on):

```julia
with_request_config(request_timeout = 30.0, max_attempts = 1) do
    chatrequest!(chat)          # both calls see the 30s / no-retry config
    embeddingrequest!(emb)
end
```

### [3. The process default (notebooks)](@id timeouts_process_default)

A notebook cell cannot hold a dynamic scope open across cells, so set a process
default once:

```julia
set_default_config!(request_timeout = 45.0, stream_idle_timeout = 300.0)
# ... every later call in the session inherits it, unless overridden ...
current_config().request_timeout    # 45.0
```

`set_default_config!(cfg)` replaces the default with `cfg`; the keyword form
merges over the current default.

### [4. Defaults](@id timeouts_builtin_defaults)

Do nothing and you get the table above — except where a provider brings defaults of
its own for some of its calls, which its entry states, such as a decision provider's
bounds on `ask` or longer bounds for a model that thinks for minutes. They hold
until one of the three channels above is set: a config you set in any channel
replaces them as a whole ([Request Configuration & Timeouts](@ref timeouts_api)).

## [Stream semantics](@id timeout_streams)

Streams are governed differently from single-shot requests:

- **Until the first byte**, a stream attempt is bound by `min(request_timeout,
  remaining total_deadline)`: a peer that has not sent the response headers by then
  — still reading the request, or never answering — ends the attempt with
  `UniLMTimeout(:request, …)`, which is retried like any per-attempt timeout.
  `stream_idle_timeout` bounds that wait too, because HTTP.jl caps the
  response-header wait with its read-idle timer, so the effective pre-first-byte
  bound is `min(request_timeout, remaining total_deadline, stream_idle_timeout)`;
  when the idle bound is the smallest, the breach reports phase `:stream_idle` even
  though no byte arrived. A TCP connect or TLS handshake already in progress is not
  cut short: it finishes, or reaches `connect_timeout`, first. A stream that never
  starts fails typed like any other request.
- **After the first byte**, only the **idle** bound runs: `stream_idle_timeout`
  is the maximum gap between raw byte chunks off the socket — NOT between parsed
  events. Keep-alive bytes — SSE comment lines, a provider's ping events — are real
  bytes, so they reset the idle clock. The bound measures **wire** idleness only: time the stream
  task spends inside your `callback` or `on_tool_call` is not counted, and the clock
  restarts when the callback returns, so a slow consumer never idle-kills a healthy
  stream. A stream that keeps delivering bytes within the bound runs as long as it
  needs; one whose provider goes silent for longer than the bound (a long silent
  reasoning phase, say) is ended — raise the bound for those.
- **A 1-byte-per-interval trickle keeps a stream alive.** The guard watches raw
  byte arrival, so a server dribbling a byte inside every idle window is, by
  construction, not idle. This is a deliberate, documented limit: distinguishing
  "healthy but slow" from "hung but trickling" is not possible from byte timing
  alone, and the safe choice is to not kill a stream that is still delivering
  bytes.
- **A completed turn is never discarded or re-sent.** Once a stream has recorded
  its terminal state — the assembled message, or the provider's own completion
  marker — anything that goes wrong while the connection is torn down is
  *teardown noise*: an idle breach, the native read-idle timer, a transport reset,
  a truncated read. The generation is already billed and its deltas already
  delivered, so the result is finalized as a **success** rather than failed or
  retried; re-POSTing it would bill a second generation for one caller request. The
  same holds for a stop or cancel that lands then: the turn is committed, the
  final-message callback runs, and usage may be missing. Trailing bytes past the
  breach (a late usage frame) may be lost — accepted. This holds on every
  provider and on both the Chat and agentic stream drivers, a stream that has no
  end-of-stream sentinel and simply ends when its body does among them.
  Failures that are *not* teardown-shaped — a throwing callback, a decoding bug
  — still surface as failures.

```julia
# Raise the idle bound for a reasoning-heavy stream expected to go quiet
# (`chat` was built with stream=true):
task = chatrequest!(chat; config = RequestConfig(stream_idle_timeout = 300.0),
                    callback = (chunk, close) -> chunk isa String && print(chunk))
result = fetch(task)
```

## [Typed failures](@id timeouts_failures)

A timeout is never a silent stall and never a fabricated HTTP status.

- Value-returning surfaces (`chatrequest!`, `embeddingrequest!`, `respond`, the
  image, FIM, System One and platform verbs) return their usual call-error result
  with `status = nothing` and the [`UniLMTimeout`](@ref) on the `cause` field —
  every `*CallError` type carries `cause`:

```julia
result = chatrequest!(chat; config = RequestConfig(request_timeout = 5.0))
if result isa LLMCallError && result.cause isa UniLMTimeout
    t = result.cause
    @warn "timed out" phase=t.phase elapsed=t.elapsed limit=t.limit
end
```

`UniLMTimeout.phase` is one of `:connect`, `:request`, `:stream_idle`, or
`:deadline`.

- **Streaming** returns a `Task` that resolves to the same typed results: a
  timeout anywhere in the stream is an `LLMCallError` / `ResponseCallError` with
  `cause::UniLMTimeout`, never a `TaskFailedException` (the task throws only for a
  user `InterruptException`). Always `fetch` a streaming task and handle failure:

```julia
task = chatrequest!(chat; callback = (chunk, close) -> chunk isa String && print(chunk))
result = fetch(task)     # LLMSuccess, or a call-error carrying the UniLMTimeout
```

- **Latency of a breach:** a non-streaming attempt's bound fires within a few
  milliseconds of its limit, and a successful call pays no detection delay — its
  completion wakes the caller directly. The stream idle check runs every
  `min(stream_idle_timeout / 4, 5)` s, so an idle breach surfaces within
  `[limit, limit + period]`.

- **MCP** surfaces are throw-based: a timeout throws [`MCPTimeoutError`](@ref),
  whose message names the override to raise.

## [Retries](@id timeouts_retries)

Automatic retries apply to the inference verbs — `chatrequest!`,
`embeddingrequest!`, `respond`, `fim_complete`, `prefix_complete`,
`generate_image`, `edit_image`, `ask`, `list_models`, and the `tool_loop` family
through them (streams retry only before the first callback fires). Every other
verb makes a single bounded attempt and `max_attempts` has no effect on it: the
platform and lifecycle verbs (batch, container, conversation, file, fine-tuning,
moderation, upload, vector-store, audio, realtime, and the Responses lifecycle
operations), including `upload_file` — a create is never retried, because a POST
that timed out or drew a gateway 5xx may already have stored the file. The polling
helpers [`poll_batch`](@ref) and [`poll_file_batch`](@ref) instead poll through
retryable statuses and per-attempt timeouts until their own wall-clock `timeout`.

`max_attempts` (default 3) caps the total attempts. All attempts of a request share
its single `total_deadline`, timeouts and retries alike:

- A retryable outcome (HTTP `408`/`429`/`500`/`502`–`504`/`520`–`524`/`529`, a
  per-attempt connect/request timeout, or a transport-level IO error) is retried
  with full-jitter exponential backoff.
- **`Retry-After` is a floor, not a replacement.** The pause is the header's wait
  plus a random spread above it — at most the exponential backoff, and at most half
  the budget left after the floor — so clients that received the same header do not
  retry in lockstep, and the next attempt keeps at least half of what remains.
- **A retry that could not finish inside the remaining deadline is not
  attempted.** If the pause would exceed the remaining `total_deadline` — with a
  header, only when the header's own wait does — the call fails *immediately* and
  returns the last real response: a budget-exhausted `429` stays a `429`, never a
  fabricated timeout.
- An `InterruptException` and a cancellation are never retried.

```julia
# Three attempts, but give up entirely after 30s of wall-clock:
respond("…"; service, model, config = RequestConfig(max_attempts = 3, total_deadline = 30.0))
```

## [MCP timeouts](@id timeouts_mcp)

```julia
# Per-connection bounds captured at connect time:
session = mcp_connect(`npx -y @modelcontextprotocol/server-filesystem /tmp`;
    config = RequestConfig(mcp_connect_timeout = 60.0, mcp_request_timeout = 30.0))

# The request-phase timeout resolves at CALL time:
#   explicit timeout  >  ambient with_request_config scope  >  captured config
call_tool(session, "read_file", Dict("path" => "/tmp/x"); timeout = 10.0)
```

- **Connect** wraps spawn → `initialize` → `notifications/initialized` →
  tool/resource/prompt discovery under `mcp_connect_timeout`. A command-not-found
  fails immediately (it does not ride the timer). The timeout message names the
  per-connect override.
- **The per-call bound covers the wait for the session too.** Calls on one session
  run one at a time, in arrival order. A call that cannot acquire the session within
  its bound throws `MCPTimeoutError` with phase `:queue` without touching it; once it
  holds the session, its exchange gets the full bound (phase `:request`), so a
  waiter's clock never cuts the exchange in progress short.
- **Stdio requests are session-fatal on timeout.** A read blocked on an
  unresponsive server can be released only by killing the server, so on timeout the
  watchdog closes the pipe, `MCPTimeoutError` is thrown at about the bound, and the
  transport (including any wrapper's child process group) is torn down. A reply
  that races the teardown is discarded: the call is a timeout.
- **Auto-respawn is opt-in and covers hangs and crashes.** A timeout closes a stdio
  session with cause `:timeout`; a server that exits or whose pipe breaks throws
  `MCPCrashError` on the call in flight and closes it with cause `:crash`. Either
  way the next call raises [`MCPSessionClosedError`](@ref) — unless you opened the
  session with `auto_respawn = true`, which respawns the same command (fresh
  handshake, logged loudly) and runs the call. **In-memory server state is lost on
  respawn**, which is why it is off by default; silent respawn would fabricate
  session continuity.
- **Ambient scope reaches bridged tools.** Tools bridged into a tool loop pass no
  keyword arguments, so wrap the loop in `with_request_config` to bound their MCP
  calls:

```julia
with_request_config(mcp_request_timeout = 15.0) do
    tool_loop!(chat; tools = mcp_tools(session))
end
```

- **MCP over HTTP** is not session-fatal on a request timeout (request/response
  correlation is per-POST); the session survives, and the timed-out request is
  cancelled on the server with a best-effort `notifications/cancelled`.

## [Concurrency](@id timeout_concurrency)

Bounds are per operation. What may be shared between concurrent operations — one
`Chat` or `Embeddings` per in-flight call, `fork` for fan-out, one `MCPSession` per
parallel worker — and how cancellation, streaming backpressure and the thread
layout interact with these bounds is covered in
[Concurrency, Tasks and Cancellation](@ref concurrency_guide). Two facts that
matter for the bounds themselves:

- A call queued behind another on an `MCPSession` waits within its own per-call
  bound (`MCPTimeoutError(:queue)` when it runs out), and `mcp_disconnect!` waits for
  the exchange in progress.
- Each stream holds its own HTTP/1.1 connection, and HTTP.jl sets no per-host
  connection cap by default, so a wide streaming fan-out does not queue for
  connections below the reach of `RequestConfig`.

## [Migrating from `retries`](@id timeouts_migrating)

The `retries` keyword is **removed** (no compatibility alias). It was a recursion
seed that counted toward a 30-attempt ceiling, so its direction is the inverse of
an attempt count. Migrate by **intent**:

| Old intent | Old code | New code |
|---|---|---|
| Disable retries | `chatrequest!(chat; retries = 30)` *(`retries=N` = "N attempts already spent"; `30` hit the ceiling)* | `chatrequest!(chat; config = RequestConfig(max_attempts = 1))` |
| Old default (retries) | `chatrequest!(chat; retries = 0)` or `chatrequest!(chat)` | `chatrequest!(chat)` *(now 3 attempts, was up to 30)* |

```julia
# Before — disable retries (the old disable switch was retries=30, not retries=0)
result = respond("…"; retries = 30)

# After
result = respond("…"; config = RequestConfig(max_attempts = 1))
```

A removed keyword raises `MethodError` — there is no silent behavior change.

## [See also](@id timeouts_see_also)

- [Concurrency, Tasks and Cancellation](@ref concurrency_guide) — sharing, fan-out,
  cancellation and the thread layout.
- [Streaming](@ref streaming_guide) — the callback / do-block streaming API.
- [MCP (Model Context Protocol)](@ref mcp_guide) — the MCP client and server.
- [Timeouts & Request Configuration](@ref timeouts_api) — the type/function reference.

## [On your provider](@id timeouts_providers)

```@eval
Main.ProviderFacts.on_your_provider(Main.ProviderFacts.llm, :failures)
```
