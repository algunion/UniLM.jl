# [Timeouts & Retries](@id timeouts_guide)

Every UniLM operation — a provider request, a stream, an MCP exchange, or a
Realtime WebSocket — waits on a peer only under a bounded, configurable limit, and
reports a breach as a **typed** error. This guide covers the bounds, how to change
them, and the typed failures you get when one fires.

One deliberate exception: an open Realtime session's *lifetime* is the caller's to
decide, so nothing bounds it (see [Realtime sessions](@ref timeout_realtime)).

## The one config struct

All bounds live on a single [`RequestConfig`](@ref), resolved per call on every
request verb (and captured at connect time by the MCP and Realtime sessions):

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

!!! warning "`connect_timeout = Inf` is unsupported on the HTTP 1.x major"
    Every task-mode watchdog abandons its worker on breach rather than killing it,
    which is safe only because the same attempt also carries a native bound that
    ends the worker on its own. On the 1.x major, `connect_timeout` is the only
    native bound covering connection acquisition — the native read bound starts
    counting after the request is written — so disabling it leaves an abandoned
    worker with nothing to terminate it. The wait you asked to be unbounded stays
    unbounded, and the worker may never self-terminate. Disable it only on the 2.x
    major, where the per-attempt request bound also covers acquisition.

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
  request_timeout)` — and on the HTTP 2.x major additionally by
  `stream_idle_timeout`, because that major bounds the response-header wait by
  the read-idle timer. The effective pre-first-byte bound there is
  `min(total_deadline, request_timeout, stream_idle_timeout)`, and a breach
  attributable to the idle timer reports phase `:stream_idle` even though no byte
  ever arrived. A stream that never starts fails typed like any other request.
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
- **A completed turn is never discarded or re-sent.** Once a stream has recorded
  its terminal state — the assembled message, or (for providers that send no
  end-of-stream sentinel) the provider's own completion marker — anything that
  goes wrong while the connection is torn down is *teardown noise*: an idle
  breach, the native read-idle timer, a transport reset, a truncated read. The
  generation is already billed and its deltas already delivered, so the result is
  finalized as a **success** rather than failed or retried; re-POSTing it would
  bill a second generation for one caller request. Trailing bytes past the
  breach (a late usage frame) may be lost — accepted. This holds on every
  provider and on both the Chat and agentic stream drivers. Gemini is the
  clearest case, since its stream has no sentinel at all and simply ends at EOF.
  Failures that are *not* teardown-shaped — a throwing callback, a decoding bug
  — still surface as failures.

```julia
# Raise the idle bound for a reasoning-heavy stream expected to go quiet:
chatrequest!(chat; config = RequestConfig(stream_idle_timeout = 300.0)) do chunk, close
    chunk isa String && print(chunk)
end
```

## [Realtime sessions](@id timeout_realtime)

The Realtime WebSocket surface takes the same `config=` keyword and resolves it
the same four ways. Two phases are bounded, and one deliberately is not:

```julia
realtime_connect(model = "gpt-realtime-2",
                 config = RequestConfig(connect_timeout = 5.0,
                                        stream_idle_timeout = 30.0)) do session
    realtime_send(session, response_create())
    event = realtime_receive(session)     # bounded by stream_idle_timeout
end
```

- **Opening** is bounded by `connect_timeout`. A peer that accepts the TCP
  connection but never finishes the upgrade throws
  `UniLMTimeout(:connect, …)` instead of blocking.
- **`realtime_receive`** is bounded by `stream_idle_timeout`, taken from the
  config the session captured at connect time. A breach throws
  `UniLMTimeout(:stream_idle, …)`. A parked socket read cannot be polled out of,
  so the guard closes the socket to unblock it — and HTTP.jl's WebSocket close
  gives the peer up to ~5 s to acknowledge, so the breach surfaces within
  `[limit, limit + ~5 s]` rather than exactly at the limit. Set the bound with
  that grace in mind; `Inf` disables it and parks indefinitely.
- **The session's lifetime is not bounded.** Once your handler is running, how
  long it stays connected is your decision, not a timeout's — a Realtime session
  is *meant* to sit idle waiting for the user to speak. Bound the conversation
  yourself if you need to.

The unlike-HTTP shapes here are that Realtime **throws** rather than returning a
typed result value, and that `realtime_connect` makes a single attempt —
`max_attempts` does not apply to it.

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
file, fine-tuning, moderation, upload, vector-store, video, audio, realtime, and
the Responses lifecycle operations) make a single bounded attempt; `max_attempts`
has no effect there.

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
  default; silent respawn would fabricate session continuity. It covers
  timeout-closures only: a server that crashes outright surfaces a transport
  error on the in-flight call and is never respawned — reconnect explicitly.
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

## [Concurrency](@id timeout_concurrency)

Bounds are per operation; the rules below say how many operations may share one
object, and what waiting behind another operation costs you.

### One `Chat` per in-flight call

A [`Chat`](@ref) is mutable, unsynchronized state. `push!`ing the response and
accumulating the running cost are plain, unlocked mutations, so two concurrent
calls sharing one `Chat` can interleave into a corrupted conversation or a lost
cost update. This is by design — locking every conversation would tax the
overwhelmingly common single-threaded use to buy safety for a pattern with a
better answer.

That answer is [`fork`](@ref), the sanctioned fan-out. It deep-copies `messages`
and gives the copy its own cost accumulator, so the forks share nothing mutable:

```julia
# Fan out four independent continuations of the same conversation:
tasks = map(fork(chat, 4)) do f
    push!(f, Message(Val(:user), "Continue differently."))
    Threads.@spawn chatrequest!(f)
end
results = fetch.(tasks)
```

The same rule reads forward: never hand one `Chat` to two `tool_loop!` calls, and
do not `push!` to a `Chat` while a streaming task on it is still running.

The stateless verbs have no such constraint — [`respond`](@ref),
[`embeddingrequest!`](@ref) and [`generate_image`](@ref) take a fresh request
struct per call and are safe to run concurrently as-is.

### An MCP session is concurrency-1

Every call on an [`MCPSession`](@ref) — liveness check, id allocation, and the
request/response exchange — runs under one session lock, so a concurrent caller
waits for the exchange in progress. Two consequences worth planning around:

- The queued caller's `mcp_request_timeout` is measured **from the moment it
  takes the lock**, not from when it asked. Waiting behind another call never
  counts against your own bound, and never tears down a healthy exchange. The
  wait itself stays bounded transitively, since the call ahead runs under that
  same per-exchange bound.
- `mcp_disconnect!` takes the lock too, so a disconnect racing a call in flight
  **waits** for that exchange to finish rather than tearing the transport down
  under its reader.

For genuine MCP parallelism, open one session per concurrent worker.

### Pick the HTTP major for your fan-out

UniLM supports both HTTP.jl majors, and they pool connections very differently:

- **HTTP 1.x** shares one **process-global** connection pool across *all* hosts,
  capped at `max(16, 4 × Threads.nthreads())`. Past that cap, acquiring a
  connection queues — so a wide fan-out silently serializes, and the queueing
  happens below the layer any `RequestConfig` bound can see except
  `connect_timeout` (which is why `connect_timeout = Inf` is unsupported there).
- **HTTP 2.x** pools per host with no per-host cap by default, and its
  per-attempt request bound also covers connection acquisition.

**For high fan-out, prefer HTTP 2.x.** On 1.x, either keep concurrency under the
cap or raise it with `HTTP.set_default_connection_limit!(n)` before the first
request.

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
- [Realtime API](@ref realtime_api) — the WebSocket type/function reference.
- [Timeouts & Request Configuration](@ref timeouts_api) — the type/function reference.
