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
if result isa LLMSuccess
    println("Finish reason: ", result.message.finish_reason)
    tc = result.message.tool_calls[1]
    println("Function: ", tc.func.name)
    println("Arguments: ", JSON.json(tc.func.arguments, 2))
else
    println("Request failed — see result for details")
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
if !isempty(calls)
    println("Function: ", calls[1]["name"])
    println("Arguments: ", JSON.json(JSON.parse(calls[1]["arguments"]), 2))
else
    println("No function calls — ", output_text(result))
end
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
if result isa ResponseSuccess
    println(output_text(result))
else
    println("Request failed — ", output_text(result))
end
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

## Using DescribedTypes.jl for Tool Definitions

[DescribedTypes.jl](https://github.com/algunion/DescribedTypes.jl) can generate tool/function schemas directly from Julia structs, replacing hand-written `Dict` parameter schemas. Use the `OPENAI_TOOLS` adapter to get the exact format expected by both ChatCompletions and Responses API tools.

**Why use DescribedTypes.jl for tools?**

- **Annotate types you don't own.** The `annotate` method can be defined for _any_ type — including those from third-party packages or Base. You can expose existing structs as tool parameters without modifying their source code.
- **Higher-quality tool calls.** Bare parameter schemas carry only type information. DescribedTypes adds field-level descriptions and constrained enum values that guide the model, producing more accurate argument values compared to raw schemas.
- **Single source of truth.** The Julia struct _is_ the parameter schema. Changes to your types automatically propagate — no more keeping `Dict` literals in sync by hand.

!!! note
    Install the package first: `Pkg.add(url="https://github.com/algunion/DescribedTypes.jl")`.

### Defining Tool Parameters as Structs

```@example described_tools
using UniLM, DescribedTypes, JSON

struct GetWeather
    location::String
    unit::String
end

DescribedTypes.annotate(::Type{GetWeather}) = Annotation(
    name="get_weather",
    description="Get current weather for a location.",
    parameters=Dict(
        :location => Annotation(name="location", description="City name"),
        :unit     => Annotation(name="unit", description="Unit of measurement",
                                enum=["celsius", "fahrenheit"]),
    ),
)

s = schema(GetWeather, llm_adapter=OPENAI_TOOLS)
println(JSON.json(s, 2))
```

### Chat Completions with DescribedTypes

Pass the schema dict directly to `GPTTool` — multiple dispatch handles the unpacking:

```@example described_tools
tool = GPTTool(s)
println(JSON.json(JSON.lower(tool), 2))
```

```@example described_tools
chat = Chat(model="gpt-5.2", tools=[GPTTool(s)])
push!(chat, Message(Val(:system), "Use the provided tools to answer."))
push!(chat, Message(Val(:user), "What's the weather in Paris?"))
result = chatrequest!(chat)
if result isa LLMSuccess && result.message.finish_reason == "tool_calls"
    tc = result.message.tool_calls[1]
    println("Function: ", tc.func.name)
    println("Arguments: ", JSON.json(tc.func.arguments, 2))
else
    println("Result: ", output_text(result))
end
```

### Responses API with DescribedTypes

Same dict, different dispatch — `function_tool` does the same:

```@example described_tools
ftool = function_tool(s)
println(JSON.json(JSON.lower(ftool), 2))
```

```@example described_tools
result = respond("What's the weather in Tokyo? Use celsius.", tools=[function_tool(s)])
calls = function_calls(result)
if !isempty(calls)
    println("Function: ", calls[1]["name"])
    println("Arguments: ", JSON.json(JSON.parse(calls[1]["arguments"]), 2))
else
    println("No function calls — ", output_text(result))
end
```

### Multiple Tools from Structs

Define several tool parameter structs and combine them:

```@example described_tools
struct SearchDatabase
    query::String
    max_results::Union{Nothing, Int}
end

DescribedTypes.annotate(::Type{SearchDatabase}) = Annotation(
    name="search_database",
    description="Search the internal database.",
    parameters=Dict(
        :query       => Annotation(name="query", description="Search query text"),
        :max_results => Annotation(name="max_results", description="Maximum results to return"),
    ),
)

tools = function_tool.([  # broadcast over a vector of schema dicts
    schema(GetWeather, llm_adapter=OPENAI_TOOLS),
    schema(SearchDatabase, llm_adapter=OPENAI_TOOLS),
])
for t in tools
    println("  - ", t.name, " (strict=", t.strict, ")")
end
```

## See Also

- [`GPTTool`](@ref), [`GPTFunctionSignature`](@ref) — Chat Completions tool types
- [`FunctionTool`](@ref), [`WebSearchTool`](@ref), [`FileSearchTool`](@ref) — Responses API tool types
- [`function_tool`](@ref), [`web_search`](@ref), [`file_search`](@ref) — convenience constructors
- [DescribedTypes.jl](https://github.com/algunion/DescribedTypes.jl) — Schema generation from Julia types
