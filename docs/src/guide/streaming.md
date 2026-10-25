# [Streaming](@id streaming_guide)

```@setup streaming
using UniLM
```

A streamed call returns a `Task` at once and hands the reply to your callback as it is generated, so
you can show partial results as they arrive; `fetch` gives the same typed result as an unstreamed
call. A `Chat` streams with `stream=true` and a `callback`, and `respond` with a `do` block.

```@example streaming
service, model = OPENAIServiceEndpoint, "gpt-5.4-mini"
nothing # hide
```

The examples use only the names bound above. `service` and `model` are the provider and model they run
on: bind your own there. The table [at the end](@ref streaming_providers) links each provider's rules.

## [Streaming a Chat](@id streaming_chat)

Set `stream=true` and provide a callback:

```@example streaming
chat = Chat(service=service, model=model, stream=true)
push!(chat, Message(Val(:system), "You are a poet."))
push!(chat, Message(Val(:user), "Write a very short 2-line poem about coding."))
task = chatrequest!(chat, callback=function(chunk, close)
    if chunk isa String
        print(chunk)
    elseif chunk isa Message
        println("\n--- done ---")
    end
end)
result = fetch(task)
if result isa LLMSuccess
    println(result.message.content)
else
    println("Request failed — see result for details")
end
nothing # hide
```

The callback fires in a fixed sequence: with a `String` argument as text
arrives — each argument is newly-generated text forwarded **verbatim** (several wire deltas
may be coalesced into one callback), so multibyte characters are never split across chunk
boundaries — and then exactly once at end-of-stream with the fully assembled
[`Message`](@ref), whose `content` equals the concatenation of every forwarded `String`.

A `Chat` streams this way on every provider. When the turn ends, how its usage arrives and when a
tool call reaches `on_tool_call` are the provider's, as its entry's streaming section says.

### [Stopping a Stream Early](@id streaming_stop)

The callback receives a `Ref{Bool}` that you can set to `true` to stop streaming:

```julia
task = chatrequest!(chat, callback=function(chunk, close)
    if chunk isa String
        print(chunk)
        if contains(chunk, "bad word")
            close[] = true  # stop the stream
        end
    end
end)
result = fetch(task)   # LLMCallError whose cause is UniLMCancelled(:callback, …)
```

A stop takes effect at once: the connection is aborted, and the call ends with
`LLMCallError(status = nothing, cause = UniLMCancelled(:callback, elapsed))` (a
`ResponseCallError` on a `Respond`). It is never retried, no further callback
runs, and the partial reply is not appended to the chat — unless the provider's
completion marker already arrived: then the turn is committed, the final-message
callback runs, and usage may be missing. On a `Chat` stream the marker is the chunk
that carries the finish reason, so a stop after it — or from the final-message callback
itself — returns the completed turn as the success it is. On a `Respond` stream the marker is
the terminal event: the response it carries stands, and its callback has already run.

The flag can also be set from another task that holds it. To stop a stream from
outside the callback, cancel a [`CancelToken`](@ref) instead: pass `cancel=tok` (or
run the call inside `with_cancel(tok)`) and call `cancel!(tok)` from any task. The
result has the same shape, with `cause.source == :token`:

```julia
tok = CancelToken()
task = chatrequest!(chat; cancel=tok, callback=(chunk, close) -> chunk isa String && print(chunk))
# … later, from any task:
cancel!(tok)
result = fetch(task)   # LLMCallError whose cause is UniLMCancelled(:token, …)
```

A stop requested while the stream task is inside your callback takes effect when
the callback returns. See [Cancellation](@ref concurrency_cancellation) for the full
contract.

### [Exceptions from callbacks](@id streaming_exceptions)

An exception thrown by `callback` or `on_tool_call` ends the call: `fetch` returns an
`LLMCallError` (or `ResponseCallError`) whose `cause` is that exception. The call is
never retried, nothing is appended to the chat, and no callback runs after it. On a `Chat`,
the usage the provider had reported by then is added to `cumulative_cost`, as for a stop
([Cumulative cost](@ref cost_guide)): a reply whose final usage had arrived counts whole, and when a
provider reports its usage is on its entry. A user `InterruptException` is the exception to that rule,
as are the `ReplayMissError` and `RecordingWriteError` of a recorded call made in a callback (see
[Recorded answers](@ref jev_testing_recorded)): each propagates, bare or inside the
exception a fetched task threw, so `fetch` throws a `TaskFailedException` wrapping it.

### [Streamed tool calls](@id streaming_tool_calls)

When the model streams tool calls, pass `on_tool_call` to be notified as each call
completes. It fires **at most once per tool call**, in call order, receiving a fully
assembled [`ToolCall`](@ref) whose arguments are already parsed (a zero-argument call
arrives as an empty `Dict`); a call whose arguments do not parse as JSON is skipped
with a warning rather than delivered. The text `callback` and `on_tool_call` are
independent, so a single request can stream assistant text and surface tool calls as
they finish:

```julia
# weather_tool defined as in the Tool Calling guide
chat = Chat(service=service, model=model, tools=[weather_tool], stream=true)
push!(chat, Message(Val(:system), "Use the tools you are given."))
push!(chat, Message(Val(:user), "What's the weather in Paris and Tokyo?"))

task = chatrequest!(chat;
    callback = (chunk, close) -> chunk isa String && print(chunk),
    on_tool_call = tc -> println("\ntool call: ", tc.func.name, " ", tc.func.arguments),
)
result = fetch(task)
```

`on_tool_call` is a keyword of `chatrequest!`, on every provider's `Chat` stream; `respond` has none. It
is a notification hook — the final assembled `Message` still carries every
tool call (alongside any assistant text), so code that does not set `on_tool_call` loses
nothing and can read `result.message.tool_calls` after `fetch`. The one call it omits is
a partial one: on a turn that did not finish with `"tool_calls"` (cut at `"length"`, say),
a call whose arguments were cut off before they formed a JSON object is dropped, while the
turn keeps its text, finish reason and usage.

`callback` and `on_tool_call` require `stream=true`: passed to a non-streaming call, they
raise `ArgumentError` before any request is sent, rather than being ignored.

## [Streaming a Respond](@id streaming_respond)

`respond` streams with Julia's `do`-block syntax:

```@example streaming
task = respond("Write a haiku about Julia programming."; service, model) do chunk, close
    if chunk isa String
        print(chunk)
    elseif chunk isa UniLM.ResponseObject
        println("\nDone! Status: ", chunk.status)
    end
end
result = fetch(task)
if result isa ResponseSuccess
    println(output_text(result))
else
    println("Request failed — ", result)
end
```

The `do`-block form automatically sets `stream=true`.

### [With explicit configuration](@id streaming_respond_config)

```@example streaming
r = Respond(
    service=service,
    model=model,
    input="Explain quantum computing step by step",
    stream=true,
    max_output_tokens=2000,
)
println("Stream enabled: ", r.stream)
println("Request preview:")
println(UniLM.encode_agentic(service, r))
```

## [Dropped SSE payloads](@id streaming_dropped)

An SSE `data:` payload the parser cannot read is dropped rather than allowed to
abort the turn, and each drop is counted for that stream. The count rides the
result as `sse_dropped` — on `LLMSuccess`, `LLMFailure`, `ResponseSuccess` and
`ResponseFailure` alike — and a single warning naming the count, model and
surface fires once the stream finalizes, not per line. `sse_dropped == 0` means a
clean stream, and is what a non-streamed call always reports. A non-zero count
means the turn was assembled from an incomplete wire: a provider-side or
transport anomaly, never routine operation. On a truncated stream it is often the
reason no message could be built at all, so it is worth reading on a failure
result as well as on a success.

## [Notes](@id streaming_notes)

- A streaming call returns a `Task` (spawned with `Threads.@spawn`, on the default
  thread pool); `fetch` it for the final typed result. It works on any thread
  layout, a single thread included — but your callbacks run on that task, so keep
  them short and hand heavy work to another task (see
  [Concurrency, Tasks and Cancellation](@ref concurrency_guide)).
- The returned `Task` throws only for a user `InterruptException`, and for a recorded call's
  errors ([Exceptions from callbacks](@ref streaming_exceptions)); every failure — a timeout, a stop,
  a callback exception — is a typed result from `fetch`.
- The `close` `Ref{Bool}` can be set to `true` from the callback (or from any task
  holding it) to terminate the stream early; see
  [Stopping a Stream Early](@ref streaming_stop).
- On completion, a `Chat`'s callback receives a `Message`; a `Respond`'s callback receives a `ResponseObject`.
- A `Chat` stream whose provider sends an end-of-stream marker is final at it: the final `Message`
  callback runs at once, and the result follows when the HTTP body ends — at most 0.5 s later. A
  server that keeps the body open longer has that connection closed instead of reused.
- **Each stream uses its own HTTP/1.1 connection**, so a consumer that applies
  backpressure to one stream cannot stall its siblings; non-streaming requests may
  share an HTTP/2 connection.
- **Streamed usage**: a streamed turn carries token usage (on `.usage`) and adds its cost to
  `cumulative_cost` like an unstreamed one where its provider reports usage in the stream — on its
  own, or because the endpoint asks for it with `stream_options = {"include_usage": true}` when the
  `Chat` sets no `stream_options`; each entry's streaming section says which.
  `DeepSeekEndpoint` streams ask for it; `AZUREServiceEndpoint` streams do not, and their content-filter preambles are tolerated.
  Elsewhere, set `stream_options=Dict("include_usage" => true)` where the server supports it, or the
  streamed turn reports no usage and accrues \$0. Empty-`choices` chunks, `:` keep-alive comment lines
  and a provider's preamble chunks are all tolerated without affecting the stream.
- A provider error mid-stream on an otherwise-`200` response surfaces as an `LLMFailure`/`LLMCallError`, never a truncated `LLMSuccess` — the `else` branch in the examples above catches it.
- **One `Chat` per in-flight call.** A `Chat` is unsynchronized mutable state, so do not share one across concurrent streams, and do not `push!` to it while its stream task is still running. Use [`fork`](@ref) to fan out — see [Concurrency, Tasks and Cancellation](@ref concurrency_guide).

## [See also](@id streaming_see_also)

- [Timeouts & Retries](@ref timeouts_guide) — the stream idle bound (wire idleness
  only: time in your callbacks is not counted) and what a completed-then-torn-down
  turn resolves to.
- [Concurrency, Tasks and Cancellation](@ref concurrency_guide) — streaming into a
  `Channel`, backpressure, and cancelling a stream from another task.
- [Decisions with LLMs](@ref jev_llm_guide) — check the finished text before it is sent on

## [On your provider](@id streaming_providers)

```@eval
Main.ProviderFacts.on_your_provider(Main.ProviderFacts.llm, :streaming)
```
