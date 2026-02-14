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

julia> result = fetch(task)

julia> result.message.content
"In lines of logic, dreams arise,  \nCrafting worlds behind the screen's wise guise"
```
