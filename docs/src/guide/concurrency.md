# [Concurrency, Tasks and Cancellation](@id concurrency_guide)

UniLM calls are ordinary blocking Julia calls: run many at once by putting them on
tasks. This guide covers which objects a concurrent call may share, the fan-out
patterns, streaming into a `Channel`, cooperative cancellation, the thread layout
the package assumes, and what the HTTP, tool-loop and MCP layers do under load.
The time bounds themselves are in [Timeouts & Retries](@ref timeouts_guide).

```@setup concurrency
using UniLM
```

## What a concurrent call may share

| Object | Mutated by its verb? | Rule |
|---|---|---|
| [`Chat`](@ref) | yes — the reply is appended to `messages`, the cost accrues | one `Chat` per in-flight call; fan out with [`fork`](@ref) |
| [`Embeddings`](@ref) | yes — the vectors are written into `emb.embeddings`, and the result aliases them | one `Embeddings` per in-flight call |
| [`Respond`](@ref), [`ImageGeneration`](@ref), [`ImageEdit`](@ref), [`FIMCompletion`](@ref), [`SystemOneRequest`](@ref) | no | share freely |
| [`MCPSession`](@ref) | its exchanges are serialized | one session per parallel worker (see [MCP](@ref concurrency_mcp)) |
| [`RequestConfig`](@ref), [`CancelToken`](@ref) | immutable / atomic | share freely |

A `Chat` is a conversation, not a request: `chatrequest!` pushes the reply onto
`chat.messages` and adds the call's estimated cost to its cumulative-cost `Ref`,
both without a lock. Two calls on one `Chat` interleave into a corrupted
conversation or a lost cost update, and a `push!` while its streaming task runs does
the same. [`fork`](@ref) is the fan-out: `fork(chat)` deep-copies every field except
`service` (the endpoint is shared), so forks share nothing mutable:

```@example concurrency
base = Chat(model="gpt-5.4-mini", stop=["END"])
push!(base, Message(Val(:system), "You are terse."))
forks = fork(base, 3)
push!(forks[1], Message(Val(:user), "first branch"))
push!(forks[1].stop, "HALT")
(length(base), length(forks[1]), length(forks[2]), base.stop)
```

Each fork gets its own cost accumulator, but it starts at the parent's running total,
not at zero: summing `cumulative_cost` over a parent and its forks counts the parent's
spend once more per fork. To account per fork, subtract the parent's total at fork
time from `cumulative_cost(fork)`, or sum `estimated_cost` over the fork's own results.

`embeddingrequest!` fills its `Embeddings` in place and returns a result whose
`embeddings` field is that same object, so a second call on it overwrites what the
first returned. Build one `Embeddings` per concurrent call, and `copy` the vectors
before reusing a request.

## Fan-out patterns

The request verbs return failures as values (`*Failure`, `*CallError`). What they
*throw* is a local validation error raised before any I/O (`ArgumentError`,
`InvalidConversationError`) or an `InterruptException`, so `fetch` on a task running
one returns its typed result and raises `TaskFailedException` only for those. The
`Task` a streaming call returns throws only for an interrupt: its validation ran
before the task was spawned. The MCP client, the Realtime WebSocket, `nl_dispatch`
and `@branch` are throw-based instead; their exceptions reach `fetch` the same way.

`Threads.@spawn` + `fetch` — one task per call, on the default thread pool:

```julia
prompts = ["Summarize A", "Summarize B", "Summarize C"]
tasks = map(prompts) do p
    Threads.@spawn respond(p; model="gpt-5.4-mini")
end
results = fetch.(tasks)                 # typed results, in submission order
ok = filter(r -> r isa ResponseSuccess, results)
```

`asyncmap(...; ntasks)` — a concurrency cap for a large batch. Its tasks run on
the calling thread, which suits I/O-bound calls; keep heavy post-processing out of
them (see [Thread layout](@ref concurrency_threads)):

```julia
results = asyncmap(prompts; ntasks=8) do p
    respond(p; model="gpt-5.4-mini")
end
```

`@sync` — wait for a group of tasks that write into preallocated slots:

```julia
replies = Vector{LLMRequestResponse}(undef, 4)
@sync for (i, f) in enumerate(fork(chat, 4))
    push!(f, Message(Val(:user), "Continue with ending $i."))
    Threads.@spawn replies[i] = chatrequest!(f)
end
```

A `with_request_config` or `with_cancel` scope opened around any of these reaches
the spawned tasks: both are scoped values, which Julia propagates into
`Threads.@spawn` and `@async` tasks.

## [Streaming into a Channel](@id concurrency_streaming)

A streaming callback runs on the stream's own task. Hand each delta to a consumer
through a `Channel` so the callback stays short:

```julia
chat = Chat(model="gpt-5.4-mini", stream=true)
push!(chat, Message(Val(:system), "You are a storyteller."))
push!(chat, Message(Val(:user), "Tell a long story."))

deltas = Channel{String}(64)
consumer = Threads.@spawn for piece in deltas
    print(piece)                        # slow work here is fine
end
bind(deltas, consumer)                  # a consumer that dies closes the channel

task = chatrequest!(chat; callback = (chunk, close) -> chunk isa String && put!(deltas, chunk))
result = fetch(task)                    # LLMSuccess, or a typed failure
close(deltas)
wait(consumer)
```

- **Backpressure is TCP's.** When the channel is full, `put!` blocks the callback,
  the stream task stops reading, and TCP flow control slows the sender. Nothing is
  buffered beyond the channel and the socket.
- **A slow consumer does not kill a healthy stream.** `stream_idle_timeout` bounds
  the gap between bytes arriving off the socket; time the stream task spends inside
  your callback (or `on_tool_call`) is not counted. The flip side: no UniLM bound
  covers a callback that never returns. `bind(deltas, consumer)` closes the channel
  when the consumer fails, so the blocked `put!` throws and the stream ends with an
  `LLMCallError` whose `cause` is that exception.
- **Stopping early:** cancel the call's token from any task (next section), or set
  `close[] = true` in the callback. Either stops the stream at once — or, while the
  stream task is inside your callback, as soon as that callback returns.

## [Cancellation](@id concurrency_cancellation)

A [`CancelToken`](@ref) is a cooperative, level-triggered flag: [`cancel!`](@ref) it
from any task, and every operation that observes it stops.

```@example concurrency
tok = CancelToken()
cancel!(tok)                            # stays cancelled: level-triggered
chat = Chat(service=GenericOpenAIEndpoint("http://127.0.0.1:9", ""), model="any")
push!(chat, Message(Val(:system), "You are terse."))
push!(chat, Message(Val(:user), "Hello"))
r = chatrequest!(chat; cancel=tok)      # sends nothing: the token is already cancelled
(typeof(r), r.status, r.cause.source)
```

- **How an operation gets its token.** Pass `cancel=tok`, or run the code inside
  `with_cancel(tok) do … end`, which makes `tok` the ambient token for everything in
  the block, tasks spawned inside it included; an explicit `cancel` wins. The
  `cancel` keyword is on [`chatrequest!`](@ref), [`respond`](@ref),
  [`embeddingrequest!`](@ref), [`tool_loop!`](@ref) / [`tool_loop`](@ref),
  [`generate_image`](@ref), [`edit_image`](@ref), [`fim_complete`](@ref),
  [`prefix_complete`](@ref), [`ask`](@ref), [`list_models`](@ref),
  [`nl_dispatch`](@ref), [`@branch`](@ref), [`poll_batch`](@ref) and
  [`poll_file_batch`](@ref). Every other HTTP verb — the platform and lifecycle
  calls — observes the ambient token.
- **What a cancelled call returns.** The verb's usual call-error result with
  `status = nothing` and `cause::UniLMCancelled` ([`UniLMCancelled`](@ref)): never
  retried, nothing committed to a `Chat`, no terminal callback. `cause.source` is
  `:token` for a `cancel!` and `:callback` for a stream stopped with
  `close[] = true`; `cause.elapsed` is the time since the call started. The cancel
  takes effect wherever it lands — before connecting, during the response-header
  wait, mid-stream, or during a retry backoff — unless the provider's completion
  marker already arrived: then the turn is committed, the final-message callback
  runs, and usage may be missing. On a Chat stream that marker is the chunk carrying
  the finish reason, which on the OpenAI wire precedes the usage chunk and `[DONE]`;
  on a Responses stream it is the terminal event, whose response (usage included)
  is the result. `nl_dispatch` and `@branch`
  throw `SystemOneError` wrapping the `SystemOneCallError`, as for any failed call. A
  cancelled tool loop returns `completed=false` with the cancelled turn's call error
  as `response`.
- **Level-triggered means one token per unit of work.** A cancelled token stays
  cancelled, and every later operation that sees it stops before any network I/O.
  Create a fresh token for each request, batch or user action you may want to
  abandon.
- **Limits.** A TCP connect or TLS handshake already in progress cannot be
  interrupted (HTTP.jl 2.7.1): a cancel during one takes effect when the connect
  completes or reaches `connect_timeout`. A callback that is running finishes
  first. The Realtime WebSocket does not observe the token, and neither does an MCP
  stdio exchange (bounded by `mcp_request_timeout`); an MCP call over HTTP does
  (see [MCP](@ref concurrency_mcp)).

```julia
tok = CancelToken()
batch = Threads.@spawn with_cancel(tok) do
    asyncmap(p -> respond(p; model="gpt-5.4-mini"), prompts; ntasks=8)
end
# … the user pressed "stop":
cancel!(tok)
results = fetch(batch)   # finished calls keep their results; the rest carry UniLMCancelled
```

The Julia 1.14 development branch adds task cancellation built on scoped,
level-triggered cancellation tokens (`Base.CancellationTokenSource`, carried in the
scoped value `Base.CANCEL_TOKEN`, with a `cancel` keyword on blocking operations).
UniLM's token, ambient scope and `cancel` keyword have the same shape so a later
release can bridge to it. No bridge exists yet, and UniLM is not tested on that
branch.

## [Thread layout](@id concurrency_threads)

Since Julia 1.12 a plain `julia` starts one default thread and one interactive
thread. The main task (and so the REPL) runs on thread 1 in the interactive pool,
and so do `@async` and `asyncmap` tasks started from it; `Threads.@spawn` lands on
the default pool. libuv's event loop, which fires every `Timer` in the process,
shares thread 1 with the main task.

- **Keep CPU-heavy work off thread 1.** A task on thread 1 that computes without
  yielding holds up the event loop, and with it every timer: on Julia 1.13 a 0.1 s
  `sleep` or `Timer` on a default-pool thread fired only when a 2 s non-yielding loop
  on thread 1 ended. UniLM's timeout watchdogs, stream idle checks and retry backoffs
  are timers, so they fire late too. Move heavy work to `Threads.@spawn` tasks and
  feed them through a `Channel`.
- **Keep streaming callbacks short.** A callback runs on its stream's task, on the
  default pool — a single thread unless you start Julia with more — so a
  CPU-heavy callback holds up its own stream and every other spawned task waiting
  for that thread. Hand the work to a consumer (see
  [Streaming into a Channel](@ref concurrency_streaming)).
- **Start with more default threads for parallel work:** `julia -t auto` or
  `julia -t 8,1` (8 default, 1 interactive).
- **Everything works on a single thread** (`julia -t 1,0`): calls, streams,
  timeouts and cancellation are cooperative, so concurrency comes from tasks
  yielding on I/O. Parallel CPU work does not.

## HTTP under fan-out

- **Each stream has its own HTTP/1.1 connection.** Streams pin HTTP/1.1, so a
  consumer applying backpressure to one stream cannot starve its siblings through a
  shared HTTP/2 flow-control window. HTTP.jl 2.7.1's default pool sets no per-host
  connection cap, so N concurrent streams hold N connections.
- **Non-streaming calls may share one HTTP/2 connection.** They keep protocol
  negotiation, and concurrent requests to one `https` host can multiplex over one
  connection.
- **Rate limits do not retry in lockstep.** A `Retry-After` header is a floor under
  the jittered backoff, not a replacement for it: each call waits at least what the
  header asked for, plus a random share of at most half the budget left after it,
  so a rate-limited batch spreads its retries instead of waking at one instant. A
  wait that does not fit the remaining `total_deadline` returns the last real
  response (the 429) at once.

## Tool loops

A tool loop runs one turn's tool calls one at a time in the calling task by
default. `tool_concurrency = n` runs up to `n` of a turn's calls at once on
spawned tasks: the dispatcher (or each `CallableTool` callable) must then be
thread-safe. Results are sent back in call order once every call of the turn has
finished, so the next request does not depend on completion order. An
`InterruptException` from any dispatch propagates once the interrupted turn is
removed from the chat, and a cancelled token stops the hand-out of further calls.

```julia
result = tool_loop!(chat; tools=my_callable_tools, tool_concurrency=4)
```

## [MCP](@id concurrency_mcp)

- **A session serializes its exchanges, first come first served.** Calls on one
  [`MCPSession`](@ref) run one at a time, in arrival order. A call's `timeout`
  (default `mcp_request_timeout`) bounds its wait for the session as well: a call
  that cannot acquire it in time throws [`MCPTimeoutError`](@ref) with phase
  `:queue` and never touches the session. Once a call holds the session its
  exchange gets its full bound, so waiting callers never cut it short.
- **For parallelism, open one session per worker.**
- **Cancellation depends on the transport.** Over HTTP a cancel ends the call at
  once: it throws `UniLMCancelled`, sends the server a best-effort
  `notifications/cancelled`, and the session stays open. A stdio exchange cannot be
  interrupted; it runs to completion under its `mcp_request_timeout`.
- **Server handlers:** [`serve`](@ref) over HTTP runs `tools/call`,
  `resources/read` and `prompts/get` handlers concurrently, one task per request on
  the default pool, so handlers must be thread-safe (protocol requests such as
  `initialize` and `ping` are answered inline and stay prompt while handlers are
  busy). Over stdio, handlers run one at a time. Registering tools, resources or
  prompts while the server is serving is synchronized.

## Known limits

- A process that blocks thread 1 without yielding delays every timer in the
  process, UniLM's timeouts and backoffs included, until it yields.
- Streaming callbacks run on the stream's task: a callback that blocks holds that
  stream (and nothing else) until it returns, and no bound covers it.
- A cancel cannot interrupt a TCP connect or TLS handshake in progress (it waits
  for the connect or for `connect_timeout`), a running callback, a Realtime
  WebSocket, or an MCP stdio exchange.

## See also

- [Timeouts & Retries](@ref timeouts_guide) — every bound, and the typed failures.
- [Streaming](@ref streaming_guide) — callbacks, stopping a stream, streamed tool calls.
- [Tool Calling](@ref tools_guide) — the tool-loop contract.
- [MCP](@ref mcp_guide) — sessions, transports and serving.
