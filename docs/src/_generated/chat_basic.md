```julia
julia> chat = Chat(model="gpt-4o-mini")

julia> push!(chat, Message(Val(:system), "You are a concise Julia programming tutor."))

julia> push!(chat, Message(Val(:user), "What is multiple dispatch? Answer in 2-3 sentences."))

julia> result = chatrequest!(chat)

julia> result.message.content
"Multiple dispatch is a feature in programming languages, including Julia, that allows the selection of a method to execute based on the types of all its arguments, rather than just the first one. This enables more flexible and expressive code, as it can define different behaviors for a function depending on the combination of argument types. It supports polymorphism, making it easier to write generic code that works with multiple types."

julia> result.message.finish_reason
"stop"

julia> length(chat)  # system + user + assistant
3
```
