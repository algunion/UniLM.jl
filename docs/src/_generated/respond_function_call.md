```julia
julia> weather_tool = function_tool(
           "get_weather", "Get the current weather for a given location",
           parameters=Dict(
               "type" => "object",
               "properties" => Dict(
                   "location" => Dict("type" => "string", "description" => "City name"),
                   "unit" => Dict("type" => "string", "enum" => ["celsius", "fahrenheit"])
               ),
               "required" => ["location", "unit"],
               "additionalProperties" => false
           ),
           strict=true
       )

julia> result = respond("What's the weather in Tokyo? Use celsius.", tools=[weather_tool])

julia> calls = function_calls(result)

julia> calls[1]["name"]
"get_weather"

julia> JSON.parse(calls[1]["arguments"])
{
  "location": "Tokyo",
  "unit": "celsius"
}
```
