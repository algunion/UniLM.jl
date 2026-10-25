# [Tool Calling](@id tools_guide)

```@setup tool_calling
using UniLM
using JSON
```

A model can ask to call functions you define: it answers with tool calls — a function's name and its
arguments — instead of text, your code runs them, and their results go back on the next turn until the
model answers. A `Chat` takes [`Tool`](@ref)s and a `Respond` [`FunctionTool`](@ref)s;
[`tool_loop!`](@ref) and [`tool_loop`](@ref) run the round trips for you.

```@example tool_calling
service, model = OPENAIServiceEndpoint, "gpt-5.4-mini"
nothing # hide
```

The examples use only the names bound above. `service` and `model` are the provider and model they run
on: bind your own there. The table [at the end](@ref tool_calling_providers) links each provider's
rules.

## [Tools on a Chat](@id tool_calling_chat)

### [Defining tools](@id tool_calling_define)

Wrap your function schema in a [`Tool`](@ref):

```@example tool_calling
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

### [Strict function calling](@id tool_calling_strict)

Pass `strict=true` to ask the provider to hold each call's arguments to your schema. What a provider
does with the flag — and what a strict schema needs there — is on its entry. Omitting `strict` (the
default) sends no flag at all: the request body is identical to previous UniLM versions.
On `OPENAIServiceEndpoint`, a strict schema that misses those needs is refused by the API: "the request
will be rejected with details about the missing constraints"
([function calling](https://developers.openai.com/api/docs/guides/function-calling#strict-mode)).

```@example tool_calling
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

### [Making tool-enabled requests](@id tool_calling_request)

A `Chat` with `tools` sends them with every request. It has `parallel_tool_calls = false` unless it
sets another value, and what a provider does with that is on its entry.
[`UniLM.encode_request`](@ref) shows the body the provider is sent:

```@example tool_calling
chat = Chat(service=service, model=model, tools=[weather_tool])
push!(chat, Message(Val(:system), "You are a helpful assistant with access to weather data."))
push!(chat, Message(Val(:user), "What's the weather in Paris?"))
println("Chat has ", length(chat.tools), " tool(s) registered")
println("Request body:")
println(UniLM.encode_request(service, chat))
```

### [Handling tool calls](@id tool_calling_calls)

When the model wants to call a function, the result message will have `finish_reason == "tool_calls"`:

```@example tool_calling
chat = Chat(
    service=service,
    model=model,
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

### [Controlling tool choice](@id tool_calling_choice)

```julia
# Let the model decide
chat = Chat(service=service, model=model, tools=[weather_tool], tool_choice="auto")

# Force the model to use a tool
chat = Chat(service=service, model=model, tools=[weather_tool], tool_choice="required")

# Prevent tool use
chat = Chat(service=service, model=model, tools=[weather_tool], tool_choice="none")

# Force one function
chat = Chat(service=service, model=model, tools=[weather_tool],
            tool_choice=UniLM.GPTToolChoice(func=:get_weather))
```

Which choices a provider takes, and how it sends them, is on its entry.

## [Tools on a Respond](@id tool_calling_respond)

A `Respond` takes [`FunctionTool`](@ref)s, made by [`function_tool`](@ref):

```@example tool_calling
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

[`function_calls`](@ref) reads a response's calls, each a `Dict` with its `call_id`, `name` and
`arguments` (a JSON string):

```@example tool_calling
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
result = respond("What's the weather in Tokyo? Use celsius.", tools=[weather_fn], service=service, model=model)
calls = function_calls(result)   # empty for a failed call
if !isempty(calls)
    println("Function: ", calls[1]["name"])
    println("Arguments: ", JSON.json(JSON.parse(calls[1]["arguments"]), 2))
else
    println("No function calls — ", result isa ResponseSuccess ? output_text(result) : result)
end
```

### [Tool choice and tool results](@id tool_calling_respond_choice)

`tool_choice` is `"auto"`, `"none"` or `"required"`, or the object a builder makes —
[`tool_choice_function`](@ref), [`tool_choice_hosted`](@ref), [`tool_choice_allowed`](@ref),
[`tool_choice_mcp`](@ref), [`tool_choice_custom`](@ref):

```@example tool_calling
r = Respond(service=service, model=model, input="What's the weather?",
            tools=[function_tool("get_weather", "Get weather",
                       parameters=Dict("type" => "object",
                                       "properties" => Dict("location" => Dict("type" => "string"))))],
            tool_choice=tool_choice_function("get_weather"))
println(r.tool_choice)
```

Return a tool's output on the next turn with [`tool_result`](@ref):

```julia
respond(Respond(; service, model, previous_response_id=r1.response.id,
                input=[tool_result("call_abc", "get_weather", "72F and sunny")]))
```

A provider may also run tools of its own, such as web search or code execution, which a `Respond`
takes in `tools`, beside function tools. Which choices and tools a provider takes is on its entry.

## [Automated tool loop](@id tool_calling_loop)

Instead of manually handling tool calls, use [`tool_loop!`](@ref) (on a `Chat`) or
[`tool_loop`](@ref) (on a `Respond`) for automatic dispatch. Both send a request, run the
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

### [On a Chat](@id tool_calling_loop_chat)

```@example tool_calling
ct = CallableTool(weather_tool, (name, args) -> "22C, sunny in $(args["location"])")
println("Callable tool wrapping: ", ct.tool.func.name)
```

```julia
chat = Chat(service=service, model=model, tools=[ct.tool])
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

### [On a Respond](@id tool_calling_loop_respond)

```julia
ct = CallableTool(
    function_tool("get_weather", "Get weather", parameters=Dict(...)),
    (name, args) -> "22C, sunny")
result = tool_loop("What's the weather?"; service, model, tools=[ct])
```

The `Respond` loop chains its turns with `previous_response_id`, or through the `conversation`
the `Respond` sets. It runs function calls only on a `completed` or `requires_action` turn; any
other status (for example `incomplete`, whose calls may be partial) stops it with `completed=false`
and an `llm_error` naming the status and the `incomplete_details` reason. A call whose
`arguments` are not a JSON object is answered with an `"Error: invalid arguments: …"`
output, like a dispatcher error, and the loop goes on. A turn that asks for a call the
application must carry out itself — a kind of call the loop cannot run — stops it with
`completed=false`, naming the pending call's type, and none of that turn's calls run.

## [MCP tool integration](@id tool_calling_mcp)

MCP servers expose tools that integrate directly with the tool loop via
[`mcp_tools`](@ref) and [`mcp_tools_respond`](@ref).
See the [MCP Guide](@ref mcp_guide) for full details.

```julia
# A Chat + MCP
session = mcp_connect(`npx server`)
tools = mcp_tools(session)
chat = Chat(service=service, model=model, tools=map(t -> t.tool, tools))
push!(chat, Message(Val(:system), "You are a helpful assistant."))
push!(chat, Message(Val(:user), "Do something"))
result = tool_loop!(chat; tools)

# A Respond + MCP
tools = mcp_tools_respond(session)
result = tool_loop("Do something"; service, model, tools=tools)
```

## [Inspecting the result](@id tool_calling_result)

`tool_loop` / `tool_loop!` return a [`ToolLoopResult`](@ref): the last `response` the loop
received, the list of `tool_calls` that ran (each a [`ToolCallOutcome`](@ref)),
`turns_used`, whether it `completed`, and — when it did not — the `llm_error` saying why.

```julia
result = tool_loop("What's the weather in Paris and Tokyo?"; service, model, tools=[ct])

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

## [See also](@id tool_calling_see_also)

- [`Tool`](@ref), [`FunctionSignature`](@ref) — the tools of a `Chat`
- [`FunctionTool`](@ref), [`function_tool`](@ref) — the function tools of a `Respond`
- [`CallableTool`](@ref), [`ToolCallOutcome`](@ref), [`ToolLoopResult`](@ref) — tool loop types
- [`tool_loop!`](@ref), [`tool_loop`](@ref) — automated tool dispatch
- [MCP Guide](@ref mcp_guide) — MCP server integration
- [Decisions with LLMs](@ref jev_llm_guide) — approve a tool call before it runs, or offer the model only the tools a request needs

## [On your provider](@id tool_calling_providers)

```@eval
Main.ProviderFacts.on_your_provider(Main.ProviderFacts.llm, :tools)
```
