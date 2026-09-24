# [Tool Calling](@id tools_guide)

Both the Chat Completions and Responses APIs support **function/tool calling** — the model
can decide to invoke functions you define, and you return the results.

## Chat Completions Tool Calling

GPT-5.6, GPT-6 Sol, and GPT-6 Luna models require `reasoning_effort="none"` when
using Chat tools. Use [`Respond`](@ref) to combine reasoning with tool calls; GPT-6
Astra tools also require the Responses API.

### Defining Tools

Wrap your function schema in a [`Tool`](@ref):

```@example tools
using UniLM
using JSON

weather_tool = Tool(
    func=FunctionSignature(
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
strict_tool = Tool(
    func=FunctionSignature(
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
    model="gpt-5.4-mini",
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

The finish reason is the provider's own: a turn that carries tool calls reads
`"tool_calls"` when the provider finished it with `"stop"` or reported none, and keeps
any other reason (`"length"`, `"content_filter"`, a provider-specific value) — such a
turn may hold partial calls, so do not run them. Partial calls are dropped: a call
cut off before its arguments formed a JSON object is removed from such a turn, which
keeps its text, finish reason and usage. On a `"tool_calls"` turn, whose calls would
run, arguments that do not parse are an `LLMCallError` instead.

!!! tip "Streaming tool calls"
    When streaming (`stream=true`), pass `on_tool_call` to [`chatrequest!`](@ref) to be
    notified as each tool call completes — at most once per call — instead of waiting for
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
chat = Chat(tools=[weather_tool], tool_choice="auto", reasoning_effort="none")

# Force the model to use a tool
chat = Chat(tools=[weather_tool], tool_choice="required", reasoning_effort="none")

# Prevent tool use
chat = Chat(tools=[weather_tool], tool_choice="none", reasoning_effort="none")
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
result = respond("What's the weather in Tokyo? Use celsius.", tools=[weather_fn], model="gpt-5.4-mini")
calls = function_calls(result)   # empty for a failed call
if !isempty(calls)
    println("Function: ", calls[1]["name"])
    println("Arguments: ", JSON.json(JSON.parse(calls[1]["arguments"]), 2))
else
    println("No function calls — ", result isa ResponseSuccess ? output_text(result) : result)
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
    tools=[web_search()],
    model="gpt-5.4-mini"
)
if result isa ResponseSuccess
    println(output_text(result))
else
    println("Request failed — ", result)
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
respond(Respond(; previous_response_id=r1.response.id,
                input=[tool_result("call_abc", "get_weather", "72F and sunny")]))
```

Gemini Interactions adds server-side hosted tools — see the
[Agentic Workflows guide](@ref agentic_guide) for [`gemini_google_search`](@ref)
and friends. When Gemini returns tool calls, the provider's opaque reasoning
token is preserved on [`ToolCall`](@ref)`.thought_signature` and echoed
automatically on the next turn.

## Automated Tool Loop

Instead of manually handling tool calls, use [`tool_loop!`](@ref) (Chat Completions) or
[`tool_loop`](@ref) (Responses API) for automatic dispatch. Both send a request, run the
calls the model asked for through your dispatcher (or the `CallableTool` callables), send
the results back, and repeat until the model answers in text, a request fails, or
`max_turns` (default 10) round-trips are used.

- **What reaches the model.** A dispatcher's `String` result is sent as is; any other value
  is sent JSON-encoded (a `Dict` becomes a JSON object, `nothing` becomes `null`). A
  dispatcher that throws sends `"Error: <message>"` as that call's output and records a
  failed [`ToolCallOutcome`](@ref), so the model can react; an `InterruptException`
  propagates instead, after `tool_loop!` removes the interrupted turn from the chat.
- **Concurrency and cancellation.** `tool_concurrency = n` runs up to `n` of a turn's calls
  at once on spawned tasks (the dispatcher must then be thread-safe; results go back in
  call order). `cancel = tok` — or an ambient `with_cancel` scope — makes the loop
  cancellable: every turn, its request and its dispatches observe the token, and a
  cancelled loop returns `completed=false` with a call error whose `cause` is
  [`UniLMCancelled`](@ref). See [Concurrency, Tasks and Cancellation](@ref concurrency_guide).
- **`max_turns`** below 1 throws `ArgumentError`; when the turns run out, `response` is the
  last response the model sent (a tool-call turn whose calls ran), with `completed=false`
  and `llm_error = "max turns (N) exhausted"`.

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

`tool_loop!` needs `chat.history == true` — each follow-up request carries the assistant
turn its tool results answer — and throws `ArgumentError` before any request otherwise.
It runs a turn's calls only when the turn finished with `"tool_calls"`. A turn that carries
calls but finished for another reason (`"length"`, `"content_filter"`, …) may hold partial
calls: none runs, the loop stops with `completed=false` and an `llm_error` naming the
reason, and that unanswered assistant turn is removed from `chat`, so the conversation
stays sendable. A turn cancelled between dispatches, or interrupted by an
`InterruptException` from a dispatch, is removed the same way. A text turn completes the
loop only when it finished with `"stop"` or reported no reason (a `"tool_calls"` finish
with no calls counts too); `"length"` ends it with `completed=false` and
`llm_error = "Model output was truncated by the token limit"`, and any other reason
(`"content_filter"`, a provider-specific value) with an `llm_error` naming the reason.
`callback` and `on_tool_call` are passed to [`chatrequest!`](@ref) unchanged, so they need
a `Chat` with `stream=true`.

### Responses API

```julia
ct = CallableTool(
    function_tool("get_weather", "Get weather", parameters=Dict(...)),
    (name, args) -> "22C, sunny")
result = tool_loop("What's the weather?"; tools=[ct])
```

The Responses loop chains turns through `previous_response_id` — or, when the `Respond`
sets `conversation`, through the conversation alone (the API rejects the two together).
It runs function calls only on a `completed` or `requires_action` turn; any other status
(for example `incomplete`, whose calls may be partial) stops it with `completed=false` and
an `llm_error` naming the status and the `incomplete_details` reason. A call whose
`arguments` are not a JSON object is answered with an `"Error: invalid arguments: …"`
output, like a dispatcher error, and the loop goes on. A turn that requests a client-side
action the loop cannot execute — `custom_tool_call`, `apply_patch_call`,
`local_shell_call`, `computer_call`, a `shell_call` the platform did not run itself (a
hosted shell answers its own call with a `shell_call_output` in the same output) or an
`mcp_approval_request` — stops it with `completed=false`, naming the pending type, and
none of that turn's calls run.

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

`tool_loop` / `tool_loop!` return a [`ToolLoopResult`](@ref): the last `response` the loop
received, the list of `tool_calls` that ran (each a [`ToolCallOutcome`](@ref)),
`turns_used`, whether it `completed`, and — when it did not — the `llm_error` saying why.

```julia
result = tool_loop("What's the weather in Paris and Tokyo?"; tools=[ct])

if result.completed
    println(output_text(result.response))
else
    # completed=false: a failed or cancelled request, max_turns, truncated or filtered
    # output, or a pending action the loop cannot run — llm_error says which
    println("Stopped after $(result.turns_used) turns: ", result.llm_error)
end

for oc in result.tool_calls          # one ToolCallOutcome per executed tool call
    status = oc.success ? "ok" : "error: $(oc.error)"
    println(oc.tool_name, oc.arguments, " -> ", status)
end
```

## See Also

- [`Tool`](@ref), [`FunctionSignature`](@ref) — Chat Completions tool types
- [`FunctionTool`](@ref), [`WebSearchTool`](@ref), [`FileSearchTool`](@ref) — Responses API tool types
- [`function_tool`](@ref), [`web_search`](@ref), [`file_search`](@ref) — convenience constructors
- [`CallableTool`](@ref), [`ToolCallOutcome`](@ref), [`ToolLoopResult`](@ref) — tool loop types
- [`tool_loop!`](@ref), [`tool_loop`](@ref) — automated tool dispatch
- [MCP Guide](@ref mcp_guide) — MCP server integration
