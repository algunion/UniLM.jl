# [Timeouts & Retries](@id timeouts_guide)

Every UniLM operation — a provider request, a stream, or an MCP exchange —
completes or fails with a **typed** error within a bounded, configurable time.
There is no code path that blocks forever on a silent or stalled peer. This guide
covers the bounds, how to change them, and the typed failures you get when one
fires.

## The one config struct

All bounds live on a single [`RequestConfig`](@ref):

| Field | Default | Bounds |
|---|---|---|
| `connect_timeout` | `10.0` s | Establishing the connection, per attempt. |
| `request_timeout` | `600.0` s | The whole non-streaming exchange, per attempt. |
| `stream_idle_timeout` | `120.0` s | Byte-gap between raw stream chunks (see [Streaming](@ref timeout_streams)). |
| `total_deadline` | `900.0` s | Across **all** attempts including backoff; for streams, until the first byte. |
| `max_attempts` | `3` | Total attempts (1 = no retry). |
| `mcp_connect_timeout` | `120.0` s | MCP spawn → `initialize` handshake complete. |
| `mcp_request_timeout` | `120.0` s | One MCP request/response exchange. |

All timeout fields are seconds (`Float64`). Set any of them to `Inf` to disable
that bound. The constructor **rejects** `NaN` and non-positive values with an
`ArgumentError`, and requires `max_attempts ≥ 1` — a silently-`NaN` bound would
compare false against every check and reintroduce an unbounded wait.

```julia
using UniLM

cfg = RequestConfig(request_timeout = 120.0, max_attempts = 5)

# Copy-with-overrides — change one field, keep the rest:
patient = RequestConfig(cfg; total_deadline = 3600.0)

# Disable the stream idle bound entirely (never idle-kill a stream):
no_idle = RequestConfig(stream_idle_timeout = Inf)
```

### Why these defaults

The defaults mirror the official provider SDKs, with headroom for HTTP.jl's
heavier connect path:

- **`request_timeout = 600s`** matches the OpenAI and Anthropic Python SDKs
  (`Timeout(600.0, connect=5.0)`) and the OpenAI Node SDK (10-minute default).
- **`max_attempts = 3`** matches those SDKs' two automatic retries (two retries
  after the first attempt = three attempts).
- **`connect_timeout = 10s`** is 2× the SDKs' 5s connect, headroom for HTTP.jl's
  heavier connect; a false connect-timeout costs one retry and has no side
  effects.
- **`stream_idle_timeout = 120s`** is stricter than the reference SDK stack's
  600s streaming read timeout. It is provisional: a per-provider live gap
  measurement runs before release (recorded under **Measured stream gaps**
  below), and the per-call override is the escape hatch for unusually silent
  reasoning streams.
- **`total_deadline = 900s`** and the **`120s` MCP** bounds cap cold starts (a
  cold `npx` MCP server on a slow link can legitimately approach the connect
  bound — the timeout message names the override to raise).

### Measured stream gaps

The `120s` idle default is provisional and backed by a pre-release measurement
(`scripts/measure_stream_gaps.jl`). The maximum inter-byte gap observed per
provider — the exact quantity the idle guard watches, which provider keep-alives
(SSE comments, Anthropic `ping`s) reset — is recorded here before each release:

| Provider | Model | Max inter-byte gap | Headroom vs 120s |
|---|---|---|---|
| OpenAI (reasoning) | `gpt-5.5` | 8.5s (504 chunks, 16.1s stream) | 14× |
| Anthropic (extended thinking) | — | not measured this release | — |
| Gemini | `gemini-3.5-flash` | 15.9s (55 chunks, 19.8s stream) | 7.5× |

Measured 2026-07-18. The Gemini stream held a healthy ~16-second silent gap
mid-thinking — the class of pause the byte-gap default must tolerate; both
measured maxima sit well under the `60s` raise-the-default threshold. A healthy
gap above `60s` in a future measurement raises the default before tagging.

## The four channels

A config is resolved, in precedence order:

1. the per-call `config=` keyword on any request verb,
2. a `with_request_config` dynamic scope,
3. the process default (`set_default_config!`),
4. the built-in defaults above.

`current_config()` returns whatever is in force right now.

### 1. Per call

```julia
chatrequest!(chat; config = RequestConfig(request_timeout = 60.0))
respond("summarize this"; config = RequestConfig(max_attempts = 1))
```

### 2. A dynamic scope

`with_request_config` merges the given keywords over the current config for the
duration of the block, and propagates into tasks spawned inside it (including
streaming's internal `Threads.@spawn`):

```julia
with_request_config(request_timeout = 30.0, max_attempts = 1) do
    chatrequest!(chat)          # both calls see the 30s / no-retry config
    embeddingrequest!(emb)
end
```

### 3. The process default (notebooks)

A notebook cell cannot hold a dynamic scope open across cells, so set a process
default once:

```julia
set_default_config!(request_timeout = 45.0, stream_idle_timeout = 300.0)
# ... every later call in the session inherits it, unless overridden ...
current_config().request_timeout    # 45.0
```

`set_default_config!(cfg)` replaces the default with `cfg`; the keyword form
merges over the current default.

### 4. Defaults

Do nothing and you get the table above.

## [Stream semantics](@id timeout_streams)

Streams are governed differently from single-shot requests:

- **Until the first byte**, a stream is bound by `min(total_deadline,
  request_timeout)`. A stream that never starts fails typed like any other
  request.
- **After the first byte**, only the **idle** bound runs: `stream_idle_timeout`
  is the maximum gap between raw byte chunks off the socket — NOT between parsed
  events. SSE comment lines and Anthropic `ping` events are real bytes, so they
  reset the idle clock. A long, healthy stream never idle-fails.
- **A 1-byte-per-interval trickle keeps a stream alive.** The guard watches raw
  byte arrival, so a server dribbling a byte inside every idle window is, by
  construction, not idle. This is a deliberate, documented limit: distinguishing
  "healthy but slow" from "hung but trickling" is not possible from byte timing
  alone, and the safe choice is to not kill a stream that is still delivering
  bytes.
- **EOF-less streams (Gemini)** have no terminal sentinel; the stream ends at EOF.
  If the idle guard fires *after* a terminal state was already recorded, the
  result is finalized as a **success** (trailing usage bytes past the gap may be
  lost — accepted), not a timeout.

```julia
# Raise the idle bound for a reasoning-heavy stream expected to go quiet:
chatrequest!(chat; config = RequestConfig(stream_idle_timeout = 300.0)) do chunk, close
    chunk isa String && print(chunk)
end
```

## Typed failures

A timeout is never a silent stall and never a fabricated HTTP status.

- Value-returning surfaces (`chatrequest!`, `embeddingrequest!`, `respond`, …)
  return their usual call-error result with `status = nothing` and the
  [`UniLMTimeout`](@ref) on the `cause` field:

```julia
result = chatrequest!(chat; config = RequestConfig(request_timeout = 5.0))
if result isa LLMCallError && result.cause isa UniLMTimeout
    t = result.cause
    @warn "timed out" phase=t.phase elapsed=t.elapsed limit=t.limit
end
```

`UniLMTimeout.phase` is one of `:connect`, `:request`, `:stream_idle`, or
`:deadline`.

- **Streaming** returns a `Task`; a timeout surfaces as a `TaskFailedException`
  when you `fetch` it (or on the call-error result the task resolves to,
  depending on where the stream failed). Always `fetch` a streaming task and
  handle failure:

```julia
task = chatrequest!(chat) do chunk, close
    chunk isa String && print(chunk)
end
result = fetch(task)     # LLMSuccess, or a call-error carrying the UniLMTimeout
```

- **MCP** surfaces are throw-based: a timeout throws [`MCPTimeoutError`](@ref),
  whose message names the override to raise.

## Retries

Automatic retries apply to the inference verbs — `chatrequest!`, `embeddingrequest!`,
`respond`, `fim_complete`, `prefix_complete`, `generate_image`, `edit_image`,
`upload_file`, and the `tool_loop` family (streams retry only before the first
callback fires). Platform and lifecycle verbs (batch, container, conversation,
file, fine-tuning, moderation, upload, vector-store, video, audio, and the
Responses lifecycle operations) make a single bounded attempt; `max_attempts` has
no effect there.

`max_attempts` (default 3) caps the total attempts. All attempts share the single
`total_deadline`:

- A retryable outcome (HTTP `408`/`429`/`500`/`502`/`503`/`504`/`529`, a
  per-attempt connect/request timeout, or a transport-level IO error) is retried
  with full-jitter exponential backoff, honoring a `Retry-After` header.
- **A retry that could not finish inside the remaining deadline is not
  attempted.** If the backoff delay alone would exceed the remaining
  `total_deadline`, the call fails *immediately* and returns the last real
  response — a budget-exhausted `429` stays a `429`, never a fabricated timeout.
- An `InterruptException` is never retried.

```julia
# Three attempts, but give up entirely after 30s of wall-clock:
respond("…"; config = RequestConfig(max_attempts = 3, total_deadline = 30.0))
```

## MCP timeouts

```julia
# Per-connection bounds captured at connect time:
session = mcp_connect(`npx -y @modelcontextprotocol/server-filesystem /tmp`;
    config = RequestConfig(mcp_connect_timeout = 60.0, mcp_request_timeout = 30.0))

# The request-phase timeout resolves at CALL time:
#   explicit timeout  >  ambient with_request_config scope  >  captured config
call_tool(session, "read_file", Dict("path" => "/tmp/x"); timeout = 10.0)
```

- **Connect** wraps spawn → `initialize` → `notifications/initialized` under
  `mcp_connect_timeout`. A command-not-found fails immediately (it does not ride
  the timer). The timeout message names the per-connect override.
- **Stdio requests are session-fatal on timeout.** Stdio framing has no
  response-id demultiplexing, so a late reply could be misdelivered to the next
  caller. On timeout the session is closed and `MCPTimeoutError` is thrown; the
  transport (including any wrapper's child process group) is torn down.
- **Auto-respawn is opt-in.** After a timeout-closed stdio session, the next call
  raises — unless you opened the session with `auto_respawn = true`, which
  respawns the same command (fresh handshake, logged loudly) and retries once.
  **In-memory server state is lost on respawn**, which is why it is off by
  default; silent respawn would fabricate session continuity.
- **Ambient scope reaches bridged tools.** Tools bridged into a tool loop pass no
  keyword arguments, so wrap the loop in `with_request_config` to bound their MCP
  calls:

```julia
with_request_config(mcp_request_timeout = 15.0) do
    tool_loop!(chat; tools = mcp_tools(session))
end
```

- **MCP over HTTP** is not session-fatal on a request timeout (request/response
  correlation is per-POST); the session survives.

## Migrating from `retries`

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

## See also

- [Streaming](@ref streaming_guide) — the callback / do-block streaming API.
- [MCP (Model Context Protocol)](@ref mcp_guide) — the MCP client and server.
- [Timeouts & Request Configuration](@ref timeouts_api) — the type/function reference.
