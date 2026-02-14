# [Tool Calling](@id tools_guide)

Both the Chat Completions and Responses APIs support **function/tool calling** — the model
can decide to invoke functions you define, and you return the results.

## Chat Completions Tool Calling

### Defining Tools

Wrap your function schema in a [`GPTTool`](@ref):

```julia
using UniLM

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
```

### Making Tool-Enabled Requests

```julia
chat = Chat(model="gpt-4o", tools=[weather_tool])
push!(chat, Message(Val(:system), "You are a helpful assistant with access to weather data."))
push!(chat, Message(Val(:user), "What's the weather in Paris?"))

result = chatrequest!(chat)
```

### Handling Tool Calls

When the model wants to call a function, the result message will have `finish_reason == "tool_calls"`:

```julia
if result isa LLMSuccess && result.message.finish_reason == "tool_calls"
    for tc in result.message.tool_calls
        func_name = tc.func.name       # "get_weather"
        func_args = tc.func.arguments  # Dict("location" => "Paris")
        
        # Execute your function
        weather_result = my_get_weather(func_args["location"])
        
        # Send the result back
        push!(chat.messages, Message(
            role="tool",
            content=string(weather_result),
            tool_call_id=tc.id
        ))
    end
    
    # Get the final response
    final = chatrequest!(chat)
    println(final.message.content)
end
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

```julia
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

result = respond("What is 2^10?", tools=[tool])

# Extract calls
for call in function_calls(result)
    println(call["name"])       # "calculate"
    println(call["arguments"])  # JSON string
end
```

### Web Search

The model can search the web — no function implementation needed:

```julia
result = respond(
    "What are the latest developments in quantum computing?",
    tools=[web_search(context_size="high")]
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

```julia
result = respond(
    "Search the web for Julia benchmarks, then summarize key findings",
    tools=[
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
)
```

## See Also

- [`GPTTool`](@ref), [`GPTFunctionSignature`](@ref) — Chat Completions tool types
- [`FunctionTool`](@ref), [`WebSearchTool`](@ref), [`FileSearchTool`](@ref) — Responses API tool types
- [`function_tool`](@ref), [`web_search`](@ref), [`file_search`](@ref) — convenience constructors
