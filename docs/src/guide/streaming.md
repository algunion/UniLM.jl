# [Streaming](@id streaming_guide)

Both APIs support **real-time streaming** of generated tokens, so you can display partial
results as they arrive.

## Chat Completions Streaming

Set `stream=true` and provide a callback:

```julia
julia> chat = Chat(model="gpt-4o-mini", stream=true)

julia> push!(chat, Message(Val(:system), "You are a poet."))

julia> push!(chat, Message(Val(:user), "Write a very short 2-line poem about coding."))

julia> task = chatrequest!(chat, callback=function(chunk, close)
           if chunk isa String
               print(chunk)  # tokens stream in real-time
           elseif chunk isa Message
               println("\n--- done ---")
           end
       end)
In lines of logic, dreams arise,  
Crafting worlds behind the screen's wise guise
--- done ---

julia> msg, _ = fetch(task)

julia> msg.content
"In lines of logic, dreams arise,  \nCrafting worlds behind the screen's wise guise"
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
julia> task = respond("Write a haiku about Julia programming.") do chunk, close
           if chunk isa String
               print(chunk)  # tokens stream in real-time
           elseif chunk isa ResponseObject
               println("\nDone! Status: ", chunk.status)
           end
       end
Multiple dispatch sings,  
Types align in swift fusion—  
Loops bloom into speed.
Done! Status: completed

julia> result = fetch(task)

julia> output_text(result)
"Multiple dispatch sings,  \nTypes align in swift fusion—  \nLoops bloom into speed."
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
