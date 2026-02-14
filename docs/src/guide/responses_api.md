# [Responses API](@id responses_guide)

The Responses API is OpenAI's newer, more flexible alternative to Chat Completions.
Key advantages include built-in tools (web search, file search), stateless multi-turn
via `previous_response_id`, and reasoning support for O-series models.

## Basic Usage

The simplest call — just a string:

```julia
julia> result = respond("Explain Julia's multiple dispatch in 2-3 sentences.")

julia> output_text(result)
"Julia's multiple dispatch means a function can have many method definitions, and Julia chooses which one to run based on the types of *all* arguments in a call (not just the first). This makes it easy to write generic code while still getting specialized, high-performance behavior for specific type combinations."

julia> result.response.id
"resp_00e791c82448c27d006990c7a81de88194975ba388932de6b8"

julia> result.response.status
"completed"

julia> result.response.model
"gpt-5.2-2025-12-11"
```

## The `Respond` Type

For full control, construct a [`Respond`](@ref) object:

```@example responses
using UniLM
using JSON

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

```julia
julia> result = respond(
           "Translate to French: The quick brown fox jumps over the lazy dog.",
           instructions="You are a professional translator. Respond only with the translation."
       )

julia> output_text(result)
"Le rapide renard brun saute par-dessus le chien paresseux."
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

```julia
julia> r1 = respond("Tell me a one-liner programming joke.", instructions="Be concise.")

julia> output_text(r1)
"There are only 10 kinds of people in the world: those who understand binary and those who don't."

julia> r2 = respond("Explain why that's funny, in one sentence.", previous_response_id=r1.response.id)

julia> output_text(r2)
"It's funny because \"10\" looks like ten in decimal but equals two in binary, so it sets up a nerdy misdirection that only people who know binary immediately get."
```

## Built-in Tools

### Web Search

```julia
julia> result = respond(
           "What is the latest stable release of the Julia programming language?",
           tools=[web_search()]
       )

julia> output_text(result)
"The latest **stable** release of the Julia programming language is **Julia v1.12.5**."
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

```julia
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
        "required" => ["colors"]
    ),
    strict=true
)
println("Format type: ", fmt.format.type)
println("Schema name: ", fmt.format.name)
```

```julia
julia> result = respond(
           "List Julia, Python, and Rust with their release year and primary paradigm.",
           text=fmt
       )

julia> JSON.parse(output_text(result))
{
  "languages": [
    {
      "name": "Julia",
      "year": 2012,
      "paradigm": "Multi-paradigm (scientific/numerical, functional, concurrent)"
    },
    {
      "name": "Python",
      "year": 1991,
      "paradigm": "Multi-paradigm (object-oriented, imperative, functional)"
    },
    {
      "name": "Rust",
      "year": 2010,
      "paradigm": "Multi-paradigm (systems programming, functional, imperative)"
    }
  ]
}
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

```julia
# Retrieve a previously stored response
result = get_response("resp_00e791c82448c27d...")

# Delete a stored response
delete_response("resp_00e791c82448c27d...")

# List input items for a response
items = list_input_items("resp_00e791c82448c27d...", limit=50)
```

## See Also

- [`Respond`](@ref) — full type reference
- [`ResponseObject`](@ref) — response structure
- [Tool Calling](@ref tools_guide) — detailed tool calling guide
- [Streaming](@ref streaming_guide) — streaming with `do`-blocks
