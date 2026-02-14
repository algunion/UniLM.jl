```julia
julia> schema = UniLM.json_schema("weather", "Weather data", Dict(
           "type" => "object",
           "properties" => Dict(
               "location" => Dict("type" => "string"),
               "temperature" => Dict("type" => "number"),
               "unit" => Dict("type" => "string"),
               "conditions" => Dict("type" => "string")
           ),
           "required" => ["location", "temperature", "unit", "conditions"],
           "additionalProperties" => false
       ))

julia> chat = Chat(model="gpt-4o-mini", response_format=schema)

julia> push!(chat, Message(Val(:system), "Respond with weather data in structured JSON."))

julia> push!(chat, Message(Val(:user), "What's the weather like in Tokyo?"))

julia> result = chatrequest!(chat)

julia> JSON.parse(result.message.content)
{
  "location": "Tokyo",
  "temperature": 22,
  "unit": "celsius",
  "conditions": "Partly Cloudy"
}
```
