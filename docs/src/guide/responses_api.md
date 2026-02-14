# [Responses API](@id responses_guide)

The Responses API is OpenAI's newer, more flexible alternative to Chat Completions.
Key advantages include built-in tools (web search, file search), stateless multi-turn
via `previous_response_id`, and reasoning support for O-series models.

## Basic Usage

The simplest call — just a string:

```julia
using UniLM

result = respond("Tell me a joke about Julia programming")

if result isa ResponseSuccess
    println(output_text(result))
end
```

## The `Respond` Type

For full control, construct a [`Respond`](@ref) object:

```julia
r = Respond(
    model="gpt-4.1",
    input="Explain monads simply",
    instructions="You are a functional programming expert. Be concise.",
    temperature=0.5,
    max_output_tokens=500,
)

result = respond(r)
```

## Instructions (System Prompt)

Unlike Chat Completions where you push a system `Message`, the Responses API uses the
`instructions` parameter:

```julia
result = respond(
    "Translate to French: The quick brown fox",
    instructions="You are a professional translator. Respond only with the translation.",
)
```

## Structured Input

For multimodal inputs, use [`InputMessage`](@ref) with content helpers:

```julia
# Text-only structured input
msgs = [
    InputMessage(role="system", content="You analyze images."),
    InputMessage(role="user", content=[
        input_text("What do you see in this image?"),
        input_image("https://example.com/photo.jpg"),
    ]),
]

result = respond(Respond(input=msgs, model="gpt-4.1"))
```

### Input Helpers

| Function              | Purpose                       |
| :-------------------- | :---------------------------- |
| [`input_text`](@ref)  | Text content part             |
| [`input_image`](@ref) | Image URL content part        |
| [`input_file`](@ref)  | File (URL or ID) content part |

## Multi-Turn Conversations

Chain requests using `previous_response_id` — no need to re-send the full history:

```julia
r1 = respond("Tell me a joke")
joke_id = r1.response.id

r2 = respond("Explain why that's funny", previous_response_id=joke_id)
println(output_text(r2))
```

## Built-in Tools

### Web Search

```julia
result = respond(
    "What are the latest Julia language releases?",
    tools=[web_search()]
)
```

### File Search

```julia
result = respond(
    "Find information about error handling",
    tools=[file_search(["vs_abc123"])]
)
```

### Function Tools

```julia
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

result = respond("What's the weather in Tokyo?", tools=[weather_tool])

# Extract function calls
for call in function_calls(result)
    println("Call: ", call["name"], " with: ", call["arguments"])
end
```

## Reasoning (O-Series Models)

For models like `o3-mini` that support extended reasoning:

```julia
result = respond(
    "Prove that √2 is irrational",
    model="o3-mini",
    reasoning=Reasoning(effort="high", summary="detailed")
)
```

## Structured Output

Force JSON-conformant output:

```julia
# Free-form JSON
result = respond("List 3 colors as JSON", text=json_object_format())

# JSON Schema
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
        "required" => ["colors"]
    ),
    strict=true
)
result = respond("List 3 colors", text=fmt)
```

## Response Accessors

```julia
result = respond("Hello!")

if result isa ResponseSuccess
    r = result.response

    # Convenience accessors
    output_text(result)      # concatenated text output
    function_calls(result)   # vector of function call dicts

    # Raw fields
    r.id                     # response ID
    r.status                 # "completed", "failed", etc.
    r.model                  # model used
    r.output                 # full output array
    r.usage                  # token usage dict
    r.raw                    # complete raw JSON
end
```

## Managing Stored Responses

```julia
# Retrieve a previously stored response
result = get_response("resp_abc123")

# Delete a stored response
delete_response("resp_abc123")

# List input items for a response
items = list_input_items("resp_abc123", limit=50)
```

## See Also

- [`Respond`](@ref) — full type reference
- [`ResponseObject`](@ref) — response structure
- [Tool Calling](@ref tools_guide) — detailed tool calling guide
- [Streaming](@ref streaming_guide) — streaming with `do`-blocks
