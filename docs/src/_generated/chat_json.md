```julia
julia> chat = Chat(model="gpt-4o-mini", response_format=ResponseFormat())

julia> push!(chat, Message(Val(:system), "You always respond in valid JSON."))

julia> push!(chat, Message(Val(:user), "List 3 programming languages with their year of creation."))

julia> result = chatrequest!(chat)

julia> JSON.parse(result.message.content)
{
  "languages": [
    {
      "name": "Python",
      "year": 1991
    },
    {
      "name": "Java",
      "year": 1995
    },
    {
      "name": "JavaScript",
      "year": 1995
    }
  ]
}
```
