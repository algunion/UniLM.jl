# [Structured Output](@id structured_guide)

Force the model to produce valid JSON conforming to a schema. Both APIs support this.

## Chat Completions

Use [`ResponseFormat`](@ref) to control the output format:

### Free-Form JSON

```julia
using UniLM

chat = Chat(
    model="gpt-4o",
    response_format=ResponseFormat()  # type="json_object"
)
push!(chat, Message(Val(:system), "You output JSON. Always respond with valid JSON."))
push!(chat, Message(Val(:user), "List 3 programming languages with their year of creation."))

result = chatrequest!(chat)
data = JSON.parse(result.message.content)
```

### JSON Schema (Strict)

```julia
using UniLM

schema = ResponseFormat(JsonSchemaAPI(
    name="languages",
    description="A list of programming languages",
    schema=Dict(
        "type" => "object",
        "properties" => Dict(
            "languages" => Dict(
                "type" => "array",
                "items" => Dict(
                    "type" => "object",
                    "properties" => Dict(
                        "name" => Dict("type" => "string"),
                        "year" => Dict("type" => "integer")
                    ),
                    "required" => ["name", "year"]
                )
            )
        ),
        "required" => ["languages"]
    )
))

chat = Chat(model="gpt-4o", response_format=schema)
push!(chat, Message(Val(:system), "Return structured data about programming languages."))
push!(chat, Message(Val(:user), "List Julia, Python, and Rust"))

result = chatrequest!(chat)
# result.message.content is guaranteed to match the schema
```

## Responses API

The Responses API uses [`TextConfig`](@ref) with convenience constructors:

### JSON Object

```julia
result = respond(
    "List 3 colors as a JSON object",
    text=json_object_format()
)
```

### JSON Schema

```julia
fmt = json_schema_format(
    "colors",
    "A structured list of colors",
    Dict(
        "type" => "object",
        "properties" => Dict(
            "colors" => Dict(
                "type" => "array",
                "items" => Dict(
                    "type" => "object",
                    "properties" => Dict(
                        "name" => Dict("type" => "string"),
                        "hex" => Dict("type" => "string")
                    ),
                    "required" => ["name", "hex"]
                )
            )
        ),
        "required" => ["colors"]
    ),
    strict=true
)

result = respond("List red, green, and blue with hex codes", text=fmt)
colors = JSON.parse(output_text(result))
```

### Plain Text Format

```julia
result = respond("Hello", text=text_format())  # default: plain text
```

## Convenience Constructors

| Constructor                              | Format                  |
| :--------------------------------------- | :---------------------- |
| `json_object_format()`                   | Unstructured JSON       |
| `json_schema_format(name, desc, schema)` | Schema-constrained JSON |
| `text_format()`                          | Plain text (default)    |

## See Also

- [`ResponseFormat`](@ref) — Chat Completions format type
- [`TextConfig`](@ref), [`TextFormatSpec`](@ref) — Responses API format types
