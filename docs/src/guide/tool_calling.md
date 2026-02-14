# [Tool Calling](@id tools_guide)

Both the Chat Completions and Responses APIs support **function/tool calling** — the model
can decide to invoke functions you define, and you return the results.

## Chat Completions Tool Calling

### Defining Tools

Wrap your function schema in a [`GPTTool`](@ref):

```@example tools
using UniLM
using JSON

weather_tool = GPTTool(
    func=GPTFunctionSignature(
        name="get_weather",
        description="Get current weather for a location",
        parameters=Dict(
            "type" => "object",
            "properties" => Dict(
                "location" => Dict("type" => "string", "description" => "City name"),
                "unit" => Dict("type" => "string", "enum" => ["celsius", "fahrenheit"])
            ),
            "required" => ["location"]
        )
    )
)
println("Tool type: ", weather_tool.type)
println("Function name: ", weather_tool.func.name)
println("Tool JSON:")
println(JSON.json(JSON.lower(weather_tool)))
```

### Making Tool-Enabled Requests

```@example tools
chat = Chat(model="gpt-5.2", tools=[weather_tool])
push!(chat, Message(Val(:system), "You are a helpful assistant with access to weather data."))
push!(chat, Message(Val(:user), "What's the weather in Paris?"))
println("Chat has ", length(chat.tools), " tool(s) registered")
println("Request body:")
println(JSON.json(chat))
```

### Handling Tool Calls

When the model wants to call a function, the result message will have `finish_reason == "tool_calls"`:

```@example tools
chat = Chat(
    model="gpt-5.2",
    tools=[weather_tool],
    tool_choice=UniLM.GPTToolChoice(func=:get_weather)
)
push!(chat, Message(Val(:system), "Use the provided tools to answer."))
push!(chat, Message(Val(:user), "What's the weather in Paris?"))
result = chatrequest!(chat)
println("Finish reason: ", result.message.finish_reason)
tc = result.message.tool_calls[1]
println("Function: ", tc.func.name)
println("Arguments: ", JSON.json(tc.func.arguments, 2))
```

### Controlling Tool Choice

```julia
# Let the model decide
chat = Chat(tools=[weather_tool], tool_choice="auto")

# Force the model to use a tool
chat = Chat(tools=[weather_tool], tool_choice="required")

# Prevent tool use
chat = Chat(tools=[weather_tool], tool_choice="none")
```

## Responses API Tool Calling

The Responses API makes tool calling more ergonomic with dedicated types.

### Function Tools

```@example tools
tool = function_tool(
    "calculate",
    "Evaluate a math expression",
    parameters=Dict(
        "type" => "object",
        "properties" => Dict(
            "expression" => Dict("type" => "string")
        ),
        "required" => ["expression"]
    ),
    strict=true
)
println("Tool: ", tool.name, " (strict=", tool.strict, ")")
println("JSON: ", JSON.json(JSON.lower(tool)))
```

```@example tools
weather_fn = function_tool(
    "get_weather",
    "Get current weather for a location",
    parameters=Dict(
        "type" => "object",
        "properties" => Dict(
            "location" => Dict("type" => "string", "description" => "City name"),
            "unit" => Dict("type" => "string", "enum" => ["celsius", "fahrenheit"])
        ),
        "required" => ["location"]
    )
)
result = respond("What's the weather in Tokyo? Use celsius.", tools=[weather_fn])
calls = function_calls(result)
println("Function: ", calls[1]["name"])
println("Arguments: ", JSON.json(JSON.parse(calls[1]["arguments"]), 2))
```

### Web Search

The model can search the web — no function implementation needed:

```@example tools
ws = web_search(context_size="high")
println("Web search tool type: ", typeof(ws))
println("Context size: ", ws.search_context_size)
```

```@example tools
result = respond(
    "What is the latest stable release of the Julia programming language?",
    tools=[web_search()]
)
println(output_text(result))
```

### File Search

Search over your uploaded vector stores:

```julia
result = respond(
    "Find the error handling policy",
    tools=[file_search(["vs_store_id_123"], max_results=5)]
)
```

### Combining Tools

Mix different tool types freely:

```@example tools
tools = [
    web_search(),
    function_tool("save_summary", "Save a summary to the database",
        parameters=Dict(
            "type" => "object",
            "properties" => Dict(
                "title" => Dict("type" => "string"),
                "content" => Dict("type" => "string")
            )
        )
    )
]
println("Number of tools: ", length(tools))
for t in tools
    println("  - ", typeof(t))
end
```

## See Also

- [`GPTTool`](@ref), [`GPTFunctionSignature`](@ref) — Chat Completions tool types
- [`FunctionTool`](@ref), [`WebSearchTool`](@ref), [`FileSearchTool`](@ref) — Responses API tool types
- [`function_tool`](@ref), [`web_search`](@ref), [`file_search`](@ref) — convenience constructors
