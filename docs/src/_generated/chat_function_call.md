```julia
julia> sig = GPTFunctionSignature(
           name="get_weather",
           description="Get the current weather for a location",
           parameters=Dict(
               "type" => "object",
               "properties" => Dict(
                   "location" => Dict("type" => "string", "description" => "City name"),
                   "unit" => Dict("type" => "string", "enum" => ["celsius", "fahrenheit"])
               ),
               "required" => ["location"]
           )
       )

julia> chat = Chat(model="gpt-4o-mini", tools=[GPTTool(func=sig)])

julia> push!(chat, Message(Val(:system), "Always use the provided tools to answer."))

julia> push!(chat, Message(Val(:user), "What is the weather in Paris?"))

julia> result = chatrequest!(chat)

julia> result.message.finish_reason
"tool_calls"

julia> tc = result.message.tool_calls[1]

julia> tc.func.name
"get_weather"

julia> tc.func.arguments
{
  "location": "Paris"
}
```
