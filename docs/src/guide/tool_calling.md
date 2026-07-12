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

### Strict Function Calling

Pass `strict=true` to make the API guarantee that tool-call arguments conform to your
schema (no extra keys, all required fields present). A strict schema must set
`additionalProperties => false` on every object and mark every property as `required`
— the API rejects strict-invalid schemas with a 400. Omitting `strict` (the default)
sends no flag at all: the request body is identical to previous UniLM versions.

```@example tools
strict_tool = GPTTool(
    func=GPTFunctionSignature(
        name="get_weather",
        description="Get current weather for a location",
        parameters=Dict(
            "type" => "object",
            "properties" => Dict(
                "location" => Dict("type" => "string", "description" => "City name")
            ),
            "required" => ["location"],
            "additionalProperties" => false
        ),
        strict=true
    )
)
println(JSON.json(strict_tool))
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

!!! tip "Streaming tool calls"
    When streaming (`stream=true`), pass `on_tool_call` to [`chatrequest!`](@ref) to be
    notified as each tool call completes — exactly once per call — instead of waiting for
    the final message. See the [Streaming guide](@ref streaming_guide).

!!! note "Gemini calls without wire ids"
    Gemini's chat path (`generateContent`) may omit `FunctionCall.id`. UniLM assigns
    such calls a synthetic positional id (`unilm_call_1`, `unilm_call_2`, …) so parallel
    tool results correlate correctly; synthetic ids never appear on the Gemini wire
    (the re-encoded request omits the id and correlates positionally, per the
    API contract). The `unilm_call_` prefix is reserved. This applies to the chat
    surface only — the Interactions API always returns server-generated call ids.

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

### Tool Choice, Tool Results & Hosted Tools

Constrain which tool the model may call with the `tool_choice=` builders
([`tool_choice_function`](@ref), [`tool_choice_hosted`](@ref),
[`tool_choice_allowed`](@ref), [`tool_choice_mcp`](@ref),
[`tool_choice_custom`](@ref)):

```@example tools
r = Respond(input="What's the weather?",
            tools=[function_tool("get_weather", "Get weather",
                       parameters=Dict("type" => "object",
                                       "properties" => Dict("location" => Dict("type" => "string"))))],
            tool_choice=tool_choice_function("get_weather"))
println(r.tool_choice)
```

Return a tool's output on the next turn with [`tool_result`](@ref):

```julia
respond(; previous_response_id=r1.response.id,
        input=[tool_result("call_abc", "get_weather", "72F and sunny")])
```

Gemini Interactions adds server-side hosted tools — see the
[Agentic Workflows guide](@ref agentic_guide) for [`gemini_google_search`](@ref)
and friends. When Gemini returns tool calls, the provider's opaque reasoning
token is preserved on [`GPTToolCall`](@ref)`.thought_signature` and echoed
automatically on the next turn.

## Automated Tool Loop

Instead of manually handling tool calls, use [`tool_loop!`](@ref) (Chat Completions) or
[`tool_loop`](@ref) (Responses API) for automatic dispatch:

### Chat Completions

```@example tools
ct = CallableTool(weather_tool, (name, args) -> "22C, sunny in $(args["location"])")
println("Callable tool wrapping: ", ct.tool.func.name)
```

```julia
chat = Chat(model="gpt-5.2", tools=[ct.tool])
push!(chat, Message(Val(:system), "You are a helpful assistant."))
push!(chat, Message(Val(:user), "What's the weather in Paris?"))
result = tool_loop!(chat; tools=[ct])
# result.completed == true when the model gives a text response
```

### Responses API

```julia
ct = CallableTool(
    function_tool("get_weather", "Get weather", parameters=Dict(...)),
    (name, args) -> "22C, sunny")
result = tool_loop("What's the weather?"; tools=[ct])
```

## MCP Tool Integration

MCP servers expose tools that integrate directly with the tool loop via
[`mcp_tools`](@ref) and [`mcp_tools_respond`](@ref).
See the [MCP Guide](@ref mcp_guide) for full details.

```julia
# Chat Completions + MCP
session = mcp_connect(`npx server`)
tools = mcp_tools(session)
chat = Chat(model="gpt-5.2", tools=map(t -> t.tool, tools))
push!(chat, Message(Val(:system), "You are a helpful assistant."))
push!(chat, Message(Val(:user), "Do something"))
result = tool_loop!(chat; tools)

# Responses API + MCP
tools = mcp_tools_respond(session)
result = tool_loop("Do something"; tools=tools)
```

## Inspecting the Result

`tool_loop` / `tool_loop!` return a [`ToolLoopResult`](@ref): the final `response`, the list
of `tool_calls` that ran (each a [`ToolCallOutcome`](@ref)), `turns_used`, whether it
`completed`, and any `llm_error`.

```julia
result = tool_loop("What's the weather in Paris and Tokyo?"; tools=[ct])

if result.completed
    println(output_text(result.response))
else
    # completed=false means it hit max_turns or an llm_error before a final text answer
    println("Stopped after $(result.turns_used) turns: ", result.llm_error)
end

for oc in result.tool_calls          # one ToolCallOutcome per executed tool call
    status = oc.success ? "ok" : "error: $(oc.error)"
    println(oc.tool_name, oc.arguments, " -> ", status)
end
```

## See Also

- [`GPTTool`](@ref), [`GPTFunctionSignature`](@ref) — Chat Completions tool types
- [`FunctionTool`](@ref), [`WebSearchTool`](@ref), [`FileSearchTool`](@ref) — Responses API tool types
- [`function_tool`](@ref), [`web_search`](@ref), [`file_search`](@ref) — convenience constructors
- [`CallableTool`](@ref), [`ToolCallOutcome`](@ref), [`ToolLoopResult`](@ref) — tool loop types
- [`tool_loop!`](@ref), [`tool_loop`](@ref) — automated tool dispatch
- [MCP Guide](@ref mcp_guide) — MCP server integration
