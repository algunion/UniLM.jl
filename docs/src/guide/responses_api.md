# [Responses API](@id responses_guide)

The Responses API is OpenAI's newer, more flexible alternative to Chat Completions.
Key advantages include built-in tools (web search, file search), stateless multi-turn
via `previous_response_id`, and reasoning support for O-series models.

```@setup responses
using UniLM
using JSON
```

## Basic Usage

The simplest call — just a string:

```@example responses
result = respond("Explain Julia's multiple dispatch in 2-3 sentences.")
if result isa ResponseSuccess
    println(output_text(result))
else
    println("Request failed — ", output_text(result))
end
```

```@example responses
if result isa ResponseSuccess
    println("ID:     ", result.response.id)
    println("Status: ", result.response.status)
    println("Model:  ", result.response.model)
else
    println("No response metadata available")
end
```

## The `Respond` Type

For full control, construct a [`Respond`](@ref) object:

```@example responses
r = Respond(
    model="gpt-5.2",
    input="Explain monads simply",
    instructions="You are a functional programming expert. Be concise.",
    temperature=0.5,
    max_output_tokens=500,
)
println("Model: ", r.model)
println("Request body:")
println(JSON.json(r))
```

## Instructions (System Prompt)

Unlike Chat Completions where you push a system `Message`, the Responses API uses the
`instructions` parameter:

```@example responses
result = respond(
    "Translate to French: The quick brown fox jumps over the lazy dog.",
    instructions="You are a professional translator. Respond only with the translation."
)
if result isa ResponseSuccess
    println(output_text(result))
else
    println("Request failed — ", output_text(result))
end
```

## Structured Input

For multimodal inputs, use [`InputMessage`](@ref) with content helpers:

```@example responses
# Text-only structured input
msgs = [
    InputMessage(role="system", content="You analyze images."),
    InputMessage(role="user", content=[
        input_text("What do you see in this image?"),
        input_image("https://example.com/photo.jpg"),
    ]),
]
r = Respond(input=msgs, model="gpt-5.2")
println("Input is structured: ", r.input isa Vector)
println("Number of input messages: ", length(r.input))
```

### Input Helpers

| Function              | Purpose                       |
| :-------------------- | :---------------------------- |
| [`input_text`](@ref)  | Text content part             |
| [`input_image`](@ref) | Image URL content part        |
| [`input_file`](@ref)  | File (URL or ID) content part |

## Multi-Turn Conversations

Chain requests using `previous_response_id` — no need to re-send the full history:

```@example responses
r1 = respond("Tell me a one-liner programming joke.", instructions="Be concise.")
if r1 isa ResponseSuccess
    println(output_text(r1))
else
    println("Request failed — ", output_text(r1))
end
```

```@example responses
if r1 isa ResponseSuccess
    r2 = respond("Explain why that's funny, in one sentence.", previous_response_id=r1.response.id)
    if r2 isa ResponseSuccess
        println(output_text(r2))
    else
        println("Request failed — ", output_text(r2))
    end
else
    println("Skipped — first request failed")
end
```

## Built-in Tools

### Web Search

```@example responses
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

```julia
result = respond(
    "Find information about error handling",
    tools=[file_search(["vs_abc123"])]
)
```

### Function Tools

```@example responses
weather_tool = function_tool(
    "get_weather",
    "Get current weather for a location",
    parameters=Dict(
        "type" => "object",
        "properties" => Dict(
            "location" => Dict("type" => "string", "description" => "City name")
        ),
        "required" => ["location"]
    )
)
println("Tool name: ", weather_tool.name)
println("Tool JSON: ", JSON.json(JSON.lower(weather_tool)))
```

```@example responses
result = respond("What's the weather in Tokyo? Use celsius.", tools=[weather_tool])
calls = function_calls(result)
if !isempty(calls)
    println("Function: ", calls[1]["name"])
    println("Arguments: ", JSON.json(JSON.parse(calls[1]["arguments"]), 2))
else
    println("No function calls — ", output_text(result))
end
```

## Reasoning (O-Series Models)

For models like `o3` that support extended reasoning:

```@example responses
r = Respond(
    input="Prove that √2 is irrational",
    model="o3",
    reasoning=Reasoning(effort="high", summary="detailed")
)
println("Model: ", r.model)
println("Reasoning effort: ", r.reasoning.effort)
println(JSON.json(r))
```

## Structured Output

Force JSON-conformant output:

```@example responses
# JSON Schema format
fmt = json_schema_format(
    "colors",
    "A list of colors",
    Dict(
        "type" => "object",
        "properties" => Dict(
            "colors" => Dict(
                "type" => "array",
                "items" => Dict("type" => "string")
            )
        ),
        "required" => ["colors"],
        "additionalProperties" => false
    ),
    strict=true
)
println("Format type: ", fmt.format.type)
println("Schema name: ", fmt.format.name)
```

```@example responses
result = respond("List 5 popular colors", text=fmt)
if result isa ResponseSuccess
    println(JSON.json(JSON.parse(output_text(result)), 2))
else
    println("Request failed — ", output_text(result))
end
```

## Response Accessors

```julia
result = respond("Hello!")

if result isa ResponseSuccess
    r = result.response

    output_text(result)      # full text output
    function_calls(result)   # Vector of function call Dicts (empty if none)

    r.id                     # "resp_00e791c8..."
    r.status                 # "completed"
    r.model                  # "gpt-5.2-2025-12-11"
    r.output                 # full output array
    r.usage                  # Dict with token counts
end
```

## Managing Stored Responses

When you pass `store=true`, the response is saved on OpeAnI's servers and can be
retrieved, inspected, or deleted later:

```@example responses
r = respond("Say 'stored response test' and nothing else.", store=true)
if r isa ResponseSuccess
    rid = r.response.id
    println("Stored response ID: ", rid)

    # Retrieve
    retrieved = get_response(rid)
    if retrieved isa ResponseSuccess
        println("Retrieved text: ", output_text(retrieved))
    end

    # List input items
    items = list_input_items(rid)
    if items isa Dict
        println("Input items: ", length(items["data"]))
    end

    # Delete
    del = delete_response(rid)
    if del isa Dict
        println("Deleted: ", del["deleted"])
    end
else
    println("Request failed — ", output_text(r))
end
```

## Metadata

Attach arbitrary key-value metadata to any request for tracking, filtering, or debugging:

```@example responses
result = respond(
    "Say 'metadata test' and nothing else.",
    metadata=Dict("env" => "docs", "request_id" => "demo_123")
)
if result isa ResponseSuccess
    println(output_text(result))
else
    println("Request failed — ", output_text(result))
end
```

## Service Tier

Control the processing tier for your request (`"auto"`, `"default"`, `"flex"`, `"priority"`):

```@example responses
result = respond("Say 'tier test' and nothing else.", service_tier="auto")
if result isa ResponseSuccess
    println(output_text(result))
else
    println("Request failed — ", output_text(result))
end
```

## Counting Input Tokens

Estimate token usage **before** making a full request — useful for cost estimation or
verifying that input fits within the context window:

```@example responses
result = count_input_tokens(input="Tell me a joke about programming")
if result isa Dict
    println("Input tokens: ", result["input_tokens"])
else
    println("Request failed — see result for details")
end
```

With tools and instructions:

```@example responses
tool = function_tool("search", "Search for information",
    parameters=Dict("type" => "object", "properties" => Dict(
        "query" => Dict("type" => "string")
    ), "required" => ["query"], "additionalProperties" => false),
    strict=true
)
result = count_input_tokens(
    input="Search for Julia language news",
    instructions="You are a helpful assistant.",
    tools=[tool]
)
if result isa Dict
    println("Tokens with tools: ", result["input_tokens"])
else
    println("Request failed — see result for details")
end
```

## Compacting Conversations

For long conversations, [`compact_response`](@ref) compresses the history into opaque,
encrypted items that reduce token usage while preserving context:

```@example responses
items = [
    Dict("role" => "user", "content" => "Hello, I want to learn about Julia."),
    Dict("type" => "message", "role" => "assistant", "status" => "completed",
         "content" => [Dict("type" => "output_text",
            "text" => "Julia is a high-performance programming language for technical computing.")])
]
result = compact_response(input=items)
if result isa Dict
    println("Compact succeeded")
    println("Output items: ", length(result["output"]))
    println("Usage: ", result["usage"])
else
    println("Request failed — see result for details")
end
```

## Cancelling Responses

Cancel an in-progress (background) response:

```julia
# Start a background response
result = respond("Write a very long essay about Julia", background=true)

# Cancel it
if result isa ResponseSuccess
    cancel_result = cancel_response(result.response.id)
    if cancel_result isa ResponseSuccess
        println("Cancelled: ", cancel_result.response.status)
    end
end
```

## Parameters Reference

| Parameter                | Type           | Default      | Description                                               |
| :----------------------- | :------------- | :----------- | :-------------------------------------------------------- |
| `model`                  | String         | `"gpt-5.2"`  | Model to use                                              |
| `input`                  | Any            | *(required)* | String or `Vector{InputMessage}`                          |
| `instructions`           | String         | —            | System-level instructions                                 |
| `tools`                  | Vector         | —            | Available tools (function, web search, file search)       |
| `tool_choice`            | String         | —            | `"auto"`, `"none"`, `"required"`                          |
| `parallel_tool_calls`    | Bool           | —            | Allow parallel tool calls                                 |
| `temperature`            | Float64        | —            | 0.0–2.0 (mutually exclusive with `top_p`)                 |
| `top_p`                  | Float64        | —            | 0.0–1.0 (mutually exclusive with `temperature`)           |
| `max_output_tokens`      | Int64          | —            | Maximum tokens in the response                            |
| `stream`                 | Bool           | —            | Enable streaming                                          |
| `text`                   | TextConfig     | —            | Output format (text, json_object, json_schema)            |
| `reasoning`              | Reasoning      | —            | Reasoning config for O-series models                      |
| `truncation`             | String         | —            | `"auto"` or `"disabled"`                                  |
| `store`                  | Bool           | —            | Store response for later retrieval                        |
| `metadata`               | Dict           | —            | Arbitrary key-value metadata                              |
| `previous_response_id`   | String         | —            | Chain to a previous response for multi-turn               |
| `user`                   | String         | —            | End-user identifier                                       |
| `background`             | Bool           | —            | Run in background (cancellable)                           |
| `include`                | Vector{String} | —            | Extra data to include (e.g. `"file_search_call.results"`) |
| `max_tool_calls`         | Int64          | —            | Max number of tool calls per turn                         |
| `service_tier`           | String         | —            | `"auto"`, `"default"`, `"flex"`, `"priority"`             |
| `top_logprobs`           | Int64          | —            | 0–20, top log probabilities                               |
| `prompt`                 | Dict           | —            | Prompt template reference                                 |
| `prompt_cache_key`       | String         | —            | Cache key for prompt caching                              |
| `prompt_cache_retention` | String         | —            | `"in-memory"` or `"24h"`                                  |
| `conversation`           | Any            | —            | Conversation context (String or Dict)                     |
| `context_management`     | Vector         | —            | Context management strategies                             |
| `stream_options`         | Dict           | —            | Streaming options (e.g. `include_usage`)                  |

## See Also

- [`Respond`](@ref) — full type reference
- [`ResponseObject`](@ref) — response structure
- [Tool Calling](@ref tools_guide) — detailed tool calling guide
- [Streaming](@ref streaming_guide) — streaming with `do`-blocks
