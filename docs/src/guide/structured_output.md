# [Structured Output](@id structured_guide)

Force the model to produce valid JSON conforming to a schema. Both APIs support this.

!!! note "Provider support varies"
    Schema-constrained output is fully supported on **OpenAI and OpenAI-compatible** providers.
    Support differs on the native backends — see the provider notes in the
    [Multi-Backend guide](@ref backend_guide) before relying on strict schemas elsewhere.

## Chat Completions

Use [`ResponseFormat`](@ref) to control the output format:

### Free-Form JSON

```@example structured
using UniLM
using JSON

chat = Chat(
    model="gpt-5.4-mini",
    response_format=ResponseFormat()  # type="json_object"
)
push!(chat, Message(Val(:system), "You output JSON. Always respond with valid JSON."))
push!(chat, Message(Val(:user), "List 3 programming languages with their year of creation."))
println("Response format type: ", chat.response_format.type)
println("Request body:")
println(JSON.json(chat))
```

```@example structured
result = chatrequest!(chat)
if result isa LLMSuccess
    println(JSON.json(JSON.parse(result.message.content), 2))
else
    println("Request failed — see result for details")
end
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
                    "required" => ["name", "year"],
                    "additionalProperties" => false
                )
            )
        ),
        "required" => ["languages"],
        "additionalProperties" => false
    ),
    strict=true
))

chat = Chat(model="gpt-5.4-mini", response_format=schema)
push!(chat, Message(Val(:system), "Return structured data about programming languages."))
push!(chat, Message(Val(:user), "List Julia, Python, and Rust"))
println("Schema name: ", schema.json_schema.name)
println("Response format type: ", schema.type)
println("Strict: ", schema.json_schema.strict)
```

```@example structured
result = chatrequest!(chat)
if result isa LLMSuccess
    println(JSON.json(JSON.parse(result.message.content), 2))
else
    println("Request failed — see result for details")
end
```

## Responses API

The Responses API uses [`TextConfig`](@ref) with convenience constructors:

### JSON Object

```@example structured
result = respond("List 3 colors as a JSON object", text=json_object_format(), model="gpt-5.4-mini")
if result isa ResponseSuccess
    println(output_text(result))
else
    println("Request failed — ", result)
end
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
                    "required" => ["name", "hex"],
                    "additionalProperties" => false
                )
            )
        ),
        "required" => ["colors"],
        "additionalProperties" => false
    ),
    strict=true
)
println("Format type: ", fmt.format.type)
println("Schema strict: ", fmt.format.strict)
```

```@example structured
result = respond("List red, green, and blue with their hex codes", text=fmt, model="gpt-5.4-mini")
if result isa ResponseSuccess
    println(JSON.json(JSON.parse(output_text(result)), 2))
else
    println("Request failed — ", result)
end
```

### Plain Text Format

```@example structured
tc = text_format()
println("Default format type: ", tc.format.type)
```

## Gemini

Native Gemini takes the same options: a `Chat`'s `response_format` becomes
`generationConfig.responseFormat` on `generateContent`, and the `text` format of a
[`respond`](@ref) call becomes the Interactions `response_format`. Only the schema
is sent — Gemini has no counterpart for the schema `name`, `description`, or `strict`.

```@example structured
capital = ResponseFormat(UniLM.JsonSchemaAPI(
    name="capital",
    description="A capital city and its country",
    schema=Dict(
        "type" => "object",
        "properties" => Dict(
            "city" => Dict("type" => "string"),
            "country" => Dict("type" => "string")
        ),
        "required" => ["city", "country"],
        "additionalProperties" => false
    )
))

chat = Chat(service=GEMINIServiceEndpoint, model="gemini-3.8-flash",
            reasoning_effort="low", max_tokens=1024, response_format=capital)
push!(chat, Message(Val(:system), "You output JSON."))
push!(chat, Message(Val(:user), "Give the capital of Norway as JSON."))
result = chatrequest!(chat)
if result isa LLMSuccess
    println(sort(collect(keys(JSON.parse(result.message.content)))))
else
    println("Request failed — ", result)
end
```

The Interactions form passes the same schema through `text`:

```julia
fmt = json_schema_format("capital", "A capital city and its country", capital.json_schema.schema)
result = respond("Give the capital of Norway as JSON."; service=GEMINIServiceEndpoint,
                 model="gemini-3.8-flash", text=fmt, max_output_tokens=256)
```

## Anthropic

Native Anthropic maps a JSON Schema `response_format` onto Claude's structured outputs:
it becomes `output_config.format = {type: "json_schema", schema}`, and only the schema
is sent (the `name`, `description` and `strict` have no counterpart — the output is
always constrained to the schema). Claude has no schema-less JSON mode, so
`ResponseFormat()` (`json_object`) throws `ArgumentError` before any request.
`FunctionSignature(strict=true)` becomes the tool's own `strict` flag.

```julia
chat = Chat(service=ANTHROPICServiceEndpoint, response_format=capital)   # default: claude-opus-5-5
push!(chat, Message(Val(:system), "You output JSON."))
push!(chat, Message(Val(:user), "Give the capital of Norway as JSON."))
result = chatrequest!(chat)
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
