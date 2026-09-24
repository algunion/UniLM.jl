# [Streaming](@id streaming_guide)

Both APIs support **real-time streaming** of generated tokens, so you can display partial
results as they arrive.

```@setup streaming
using UniLM
using JSON
```

## Chat Completions Streaming

Set `stream=true` and provide a callback:

```@example streaming
chat = Chat(model="gpt-5.4-mini", stream=true)
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

The Chat Completions callback fires in a fixed sequence: with a `String` argument as text
arrives — each argument is newly-generated text forwarded **verbatim** (several wire deltas
may be coalesced into one callback), so multibyte characters are never split across chunk
boundaries — and then exactly once at end-of-stream with the fully assembled
[`Message`](@ref), whose `content` equals the concatenation of every forwarded `String`.

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
`ResponseCallError` on the Responses path). It is never retried, no further callback
runs, and the partial reply is not appended to the chat — unless the provider's
completion marker already arrived: then the turn is committed, the final-message
callback runs, and usage may be missing. On a Chat stream the marker is the chunk
that carries the finish reason, which on the OpenAI wire comes before the usage chunk
and `[DONE]`, so a stop in that window — or from the final-message callback itself —
returns the completed turn as the success it is. On the Responses path the marker is
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

### Exceptions from callbacks

An exception thrown by `callback` or `on_tool_call` ends the call: `fetch` returns an
`LLMCallError` (or `ResponseCallError`) whose `cause` is that exception. The call is
never retried, nothing is appended to the chat, and no callback runs after it. A user
`InterruptException` is the exception to that rule: it propagates, so `fetch` throws
a `TaskFailedException` wrapping it.

### Streamed Tool Calls

When the model streams tool calls, pass `on_tool_call` to be notified as each call
completes. It fires **at most once per tool call**, in call order, receiving a fully
assembled [`ToolCall`](@ref) whose arguments are already parsed (a zero-argument call
arrives as an empty `Dict`); a call whose arguments do not parse as JSON is skipped
with a warning rather than delivered. The text `callback` and `on_tool_call` are
independent, so a single request can stream assistant text and surface tool calls as
they finish:

```julia
# weather_tool defined as in the Tool Calling guide
chat = Chat(model="gpt-5.2", tools=[weather_tool], stream=true)
push!(chat, Message(Val(:system), "Use the tools you are given."))
push!(chat, Message(Val(:user), "What's the weather in Paris and Tokyo?"))

task = chatrequest!(chat;
    callback = (chunk, close) -> chunk isa String && print(chunk),
    on_tool_call = tc -> println("\ntool call: ", tc.func.name, " ", tc.func.arguments),
)
result = fetch(task)
```

`on_tool_call` is supported on the `chatrequest!` streaming path for every chat provider
(OpenAI-wire, native Anthropic, native Gemini); the Responses-API `respond` path does not
surface it. It is a notification hook — the final assembled `Message` still carries every
tool call (alongside any assistant text), so code that does not set `on_tool_call` loses
nothing and can read `result.message.tool_calls` after `fetch`. The one call it omits is
a partial one: on a turn that did not finish with `"tool_calls"` (cut at `"length"`, say),
a call whose arguments were cut off before they formed a JSON object is dropped, while the
turn keeps its text, finish reason and usage.

`callback` and `on_tool_call` require `stream=true`: passed to a non-streaming call, they
raise `ArgumentError` before any request is sent, rather than being ignored.

## Responses API Streaming

The Responses API provides an even cleaner streaming interface using Julia's `do`-block syntax:

```@example streaming
task = respond("Write a haiku about Julia programming.", model="gpt-5.4-mini") do chunk, close
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

### With Explicit Configuration

```@example streaming
r = Respond(
    input="Explain quantum computing step by step",
    model="gpt-5.2",
    stream=true,
    max_output_tokens=2000,
)
println("Stream enabled: ", r.stream)
println("Request preview:")
println(JSON.json(r))
```

## Streaming Across Providers

Streaming is not OpenAI-only. The **native Anthropic** (`ANTHROPICServiceEndpoint`) and
**native Gemini** (`GEMINIServiceEndpoint`) backends stream with the *same* callback /
`do`-block API shown above — only the `service` (and model) change:

```julia
# Native Anthropic streaming (Chat Completions)
chat = Chat(service=ANTHROPICServiceEndpoint, stream=true)
push!(chat, Message(Val(:system), "You are a poet."))
push!(chat, Message(Val(:user), "Two lines about the sea."))
task = chatrequest!(chat, callback=(chunk, close) -> chunk isa String && print(chunk))
fetch(task)

# Native Gemini streaming (Chat Completions)
chat = Chat(service=GEMINIServiceEndpoint, stream=true)
push!(chat, Message(Val(:system), "You are a poet."))
push!(chat, Message(Val(:user), "Two lines about the mountains."))
task = chatrequest!(chat, callback=(chunk, close) -> chunk isa String && print(chunk))
fetch(task)
```

Streaming preserves provider-native fidelity too: for Anthropic streams, the
content blocks (including thinking blocks and their signatures) are
re-assembled verbatim from the SSE deltas and attached to the final
`Message.provider_content`, so a streamed tool-calling turn round-trips
exactly like a non-streamed one.

Providers on the OpenAI-compatible Chat Completions standard (DeepSeek, Ollama, vLLM, LM
Studio, …) stream through the same `stream=true` + callback path.

## Dropped SSE Payloads

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

## Notes

- A streaming call returns a `Task` (spawned with `Threads.@spawn`, on the default
  thread pool); `fetch` it for the final typed result. It works on any thread
  layout, a single thread included — but your callbacks run on that task, so keep
  them short and hand heavy work to another task (see
  [Concurrency, Tasks and Cancellation](@ref concurrency_guide)).
- The returned `Task` throws only for a user `InterruptException`; every failure —
  a timeout, a stop, a callback exception — is a typed result from `fetch`.
- The `close` `Ref{Bool}` can be set to `true` from the callback (or from any task
  holding it) to terminate the stream early; see
  [Stopping a Stream Early](@ref streaming_stop).
- On completion, the Chat Completions callback receives a `Message`; the Responses API callback receives a `ResponseObject`.
- A Chat stream is final at its end-of-stream sentinel (`[DONE]`, Anthropic's
  `message_stop`): the final `Message` callback runs at once, and the result follows
  when the HTTP body ends — at most 0.5 s later. A server that keeps the body open
  longer has that connection closed instead of reused.
- **Each stream uses its own HTTP/1.1 connection**, so a consumer that applies
  backpressure to one stream cannot stall its siblings; non-streaming requests may
  share an HTTP/2 connection.
- **Streamed usage**: `OPENAIServiceEndpoint` and `DeepSeekEndpoint` streams request
  `stream_options = {"include_usage": true}` automatically when `stream_options` is
  unset, so a streamed turn carries token usage (on `.usage`) and accrues cost like a
  non-streamed one. Native Anthropic and Gemini streams report usage on their own.
  Other OpenAI-compatible servers (Azure, the Gemini compat shim, generic endpoints)
  are not sent the field, since some reject it: set
  `stream_options=Dict("include_usage" => true)` where the server supports it, or the
  streamed turn reports no usage and accrues \$0. Empty-`choices` chunks, `:`
  keep-alive comment lines, and provider preambles (e.g. Azure content-filter
  results) are all tolerated without affecting the stream.
- A provider error mid-stream on an otherwise-`200` response (e.g. an Anthropic `overloaded_error`) surfaces as an `LLMFailure`/`LLMCallError`, never a truncated `LLMSuccess` — the `else` branch in the examples above catches it.
- **One `Chat` per in-flight call.** A `Chat` is unsynchronized mutable state, so do not share one across concurrent streams, and do not `push!` to it while its stream task is still running. Use [`fork`](@ref) to fan out — see [Concurrency, Tasks and Cancellation](@ref concurrency_guide).

## See Also

- [Timeouts & Retries](@ref timeouts_guide) — the stream idle bound (wire idleness
  only: time in your callbacks is not counted) and what a completed-then-torn-down
  turn resolves to.
- [Concurrency, Tasks and Cancellation](@ref concurrency_guide) — streaming into a
  `Channel`, backpressure, and cancelling a stream from another task.
