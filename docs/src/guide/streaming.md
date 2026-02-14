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

## Notes

- Streaming runs on a **separate Julia thread** via `Threads.@spawn`. Make sure Julia is started with multiple threads (`julia -t auto`).
- The returned `Task` can be `fetch`ed to get the final result.
- The `close` `Ref{Bool}` can be set to `true` from the callback to terminate the stream early.
- On completion, the Chat Completions callback receives a `Message`; the Responses API callback receives a `ResponseObject`.
