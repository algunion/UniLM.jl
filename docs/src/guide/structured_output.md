# [Structured Output](@id structured_guide)

Force the model to produce valid JSON conforming to a schema. Both APIs support this.

## Chat Completions

Use [`ResponseFormat`](@ref) to control the output format:

### Free-Form JSON

```@example structured
using UniLM
using JSON

chat = Chat(
    model="gpt-5.2",
    response_format=ResponseFormat()  # type="json_object"
)
push!(chat, Message(Val(:system), "You output JSON. Always respond with valid JSON."))
push!(chat, Message(Val(:user), "List 3 programming languages with their year of creation."))
println("Response format type: ", chat.response_format.type)
println("Request body:")
println(JSON.json(chat))
```

```julia
julia> chat = Chat(model="gpt-4o-mini", response_format=ResponseFormat())

julia> push!(chat, Message(Val(:system), "You always respond in valid JSON."))

julia> push!(chat, Message(Val(:user), "List 3 programming languages with their year of creation."))

julia> result = chatrequest!(chat)

julia> JSON.parse(result.message.content)
{
  "languages": [
    {
      "name": "Python",
      "year": 1991
    },
    {
      "name": "Java",
      "year": 1995
    },
    {
      "name": "JavaScript",
      "year": 1995
    }
  ]
}
```

### JSON Schema (Strict)

```@example structured
schema = ResponseFormat(UniLM.JsonSchemaAPI(
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

chat = Chat(model="gpt-5.2", response_format=schema)
push!(chat, Message(Val(:system), "Return structured data about programming languages."))
push!(chat, Message(Val(:user), "List Julia, Python, and Rust"))
println("Schema name: ", schema.json_schema.name)
println("Response format type: ", schema.type)
```

```julia
julia> result = chatrequest!(chat)

julia> data = JSON.parse(result.message.content)
# result.message.content is guaranteed to match the schema
```

## Responses API

The Responses API uses [`TextConfig`](@ref) with convenience constructors:

### JSON Object

```julia
julia> result = respond("List 3 colors as a JSON object", text=json_object_format())

julia> output_text(result)
# Returns valid JSON conforming to "json_object" format
```

### JSON Schema

```@example structured
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
println("Format type: ", fmt.format.type)
println("Schema strict: ", fmt.format.strict)
```

```julia
julia> result = respond("List red, green, and blue with their hex codes", text=fmt)

julia> JSON.parse(output_text(result))
# Returns JSON matching the defined schema
```

### Plain Text Format

```@example structured
tc = text_format()
println("Default format type: ", tc.format.type)
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
