# Responses API

Types and functions for the **Responses API** — the newer, more flexible alternative
to Chat Completions.

## Request Type

```@docs
Respond
```

### Construction

```@example responses_api
using UniLM
using JSON

# Simple text request
r = Respond(input="Tell me a joke")
println("Model: ", r.model)
println("Input: ", r.input)

# With instructions and tools
r2 = Respond(
    input="What's the weather in Paris?",
    instructions="You are a helpful weather assistant",
    tools=[web_search()]
)
println("Has instructions: ", !isnothing(r2.instructions))
println("Tools: ", length(r2.tools))
```

## Response Object

```@docs
ResponseObject
```

## Result Types

```@docs
ResponseSuccess
ResponseFailure
ResponseCallError
```

## Accessor Functions

```@docs
output_text
function_calls
```

## Request Functions

```@docs
respond
get_response
delete_response
list_input_items
cancel_response
compact_response
count_input_tokens
```

## Input Helpers

```@docs
InputMessage
input_text
input_image
input_file
```

### Multimodal Input

```@example responses_api
# Text-only input
msg = InputMessage(role="user", content="What is Julia?")
println("Role: ", msg.role)

# Multimodal input
parts = [
    input_text("Describe this image:"),
    input_image("https://example.com/photo.jpg", detail="high")
]
println("Parts: ", length(parts))
println("Part types: ", [p[:type] for p in parts])
```

## Tool Types

```@docs
ResponseTool
FunctionTool
WebSearchTool
FileSearchTool
```

### Tool Constructors

```@docs
function_tool
web_search
file_search
```

```@example responses_api
# Function tool
ft = function_tool("calculate", "Evaluate a math expression",
    parameters=Dict("type" => "object", "properties" => Dict(
        "expr" => Dict("type" => "string")
    ))
)
println("Function tool: ", ft.name)

# Web search
ws = web_search(context_size="high")
println("Web search context: ", ws.search_context_size)
```

## Text Format

```@docs
TextConfig
TextFormatSpec
text_format
json_schema_format
json_object_format
```

```@example responses_api
tf = text_format()
println("Default format: ", tf.format.type)

jf = json_object_format()
println("JSON format: ", jf.format.type)
```

## Reasoning

```@docs
Reasoning
```

```@example responses_api
reasoning = UniLM.Reasoning(effort="high")
println("Effort: ", reasoning.effort)
```
