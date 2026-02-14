# [Streaming](@id streaming_guide)

Both APIs support **real-time streaming** of generated tokens, so you can display partial
results as they arrive.

## Chat Completions Streaming

Set `stream=true` and provide a callback:

```julia
using UniLM

chat = Chat(model="gpt-5.2", stream=true)
push!(chat, Message(Val(:system), "You are a storyteller."))
push!(chat, Message(Val(:user), "Tell me a short story about a robot learning Julia."))

task = chatrequest!(chat, callback=function(chunk, close)
    if chunk isa String
        print(chunk)  # partial text delta
    elseif chunk isa Message
        println("\n--- Done! ---")
    end
end)

# The task runs on a separate thread
result = fetch(task)
# Output (streamed token by token):
# Once upon a time, in a server room humming with electricity,
# there lived a small robot named JL-42...
# --- Done! ---
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

```julia
using UniLM

task = respond("Tell me a story about a robot learning Julia") do chunk, close
    if chunk isa String
        print(chunk)           # partial text delta
    elseif chunk isa ResponseObject
        println("\nDone! Status: ", chunk.status)
    end
end

result = fetch(task)
# Output (streamed):
# In the year 2026, a small autonomous robot named Dispatch
# discovered a programming language that changed everything...
# Done! Status: completed
```

The `do`-block form automatically sets `stream=true`.

### With Explicit Configuration

```@example streaming
using UniLM
using JSON

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
