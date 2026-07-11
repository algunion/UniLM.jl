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
chat = Chat(model="gpt-4o-mini", stream=true)
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

### Stopping a Stream Early

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
```

### Streamed Tool Calls

When the model streams tool calls, pass `on_tool_call` to be notified as each call
completes. It fires **exactly once per tool call**, in call order, receiving a fully
assembled [`GPTToolCall`](@ref) whose arguments are already parsed (a zero-argument call
arrives as an empty `Dict`). The text `callback` and `on_tool_call` are independent, so a
single request can stream assistant text and surface tool calls as they finish:

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
nothing and can read `result.message.tool_calls` after `fetch`.

## Responses API Streaming

The Responses API provides an even cleaner streaming interface using Julia's `do`-block syntax:

```@example streaming
task = respond("Write a haiku about Julia programming.") do chunk, close
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
    println("Request failed — ", output_text(result))
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

Providers on the OpenAI-compatible Chat Completions standard (DeepSeek, Ollama, vLLM, LM
Studio, …) stream through the same `stream=true` + callback path.

## Notes

- Streaming runs on a **separate Julia thread** via `Threads.@spawn`. Make sure Julia is started with multiple threads (`julia -t auto`).
- The returned `Task` can be `fetch`ed to get the final result.
- The `close` `Ref{Bool}` can be set to `true` from the callback to terminate the stream early.
- On completion, the Chat Completions callback receives a `Message`; the Responses API callback receives a `ResponseObject`.
- **Streamed usage**: set `stream_options=Dict("include_usage" => true)` to capture token usage — it lands on the result's `.usage` once the stream completes. Empty-`choices` chunks, `:` keep-alive comment lines, and provider preambles (e.g. Azure content-filter results) are all tolerated without affecting the stream.
- A provider error mid-stream on an otherwise-`200` response (e.g. an Anthropic `overloaded_error`) surfaces as an `LLMFailure`/`LLMCallError`, never a truncated `LLMSuccess` — the `else` branch in the examples above catches it.
