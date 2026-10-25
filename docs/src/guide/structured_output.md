# [Structured Output](@id structured_guide)

```@setup structured_output
using UniLM
using JSON
```

A format makes the model answer in JSON: any JSON object, or one that follows a JSON Schema you give. A
`Chat` takes it as `response_format`, a `Respond` as `text`. How a provider constrains the reply, which
parts of a format it sends, and which formats it refuses are on its entry.

```@example structured_output
service, model = OPENAIServiceEndpoint, "gpt-5.4-mini"
nothing # hide
```

The examples use only the names bound above. `service` and `model` are the provider and model they run
on: bind your own there. The table [at the end](@ref structured_output_providers) links each provider's
rules.

## [On a Chat](@id structured_output_chat)

Use [`ResponseFormat`](@ref) to control the output format:

### [Free-form JSON](@id structured_output_json_object)

```@example structured_output
chat = Chat(
    service=service,
    model=model,
    response_format=ResponseFormat()  # type="json_object"
)
push!(chat, Message(Val(:system), "You output JSON. Always respond with valid JSON."))
push!(chat, Message(Val(:user), "List 3 programming languages with their year of creation."))
println("Response format type: ", chat.response_format.type)
println("Request body:")
println(UniLM.encode_request(service, chat))
```

```@example structured_output
result = chatrequest!(chat)
if result isa LLMSuccess
    println(JSON.json(JSON.parse(result.message.content), 2))
else
    println("Request failed — see result for details")
end
```

### [JSON Schema (strict)](@id structured_output_schema)

```@example structured_output
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

chat = Chat(service=service, model=model, response_format=schema)
push!(chat, Message(Val(:system), "Return structured data about programming languages."))
push!(chat, Message(Val(:user), "List Julia, Python, and Rust"))
println("Schema name: ", schema.json_schema.name)
println("Response format type: ", schema.type)
println("Strict: ", schema.json_schema.strict)
```

```@example structured_output
result = chatrequest!(chat)
if result isa LLMSuccess
    println(JSON.json(JSON.parse(result.message.content), 2))
else
    println("Request failed — see result for details")
end
```

`strict=true` asks for a reply that follows the schema exactly; whether a provider sends the format's
`name`, `description` and `strict`, or the schema alone, is on its entry.

## [On a Respond](@id structured_output_respond)

A `Respond` takes a [`TextConfig`](@ref) in `text`, made by the convenience constructors:

### [JSON object](@id structured_output_respond_object)

```@example structured_output
result = respond("List 3 colors as a JSON object", text=json_object_format(), service=service, model=model)
if result isa ResponseSuccess
    println(output_text(result))
else
    println("Request failed — ", result)
end
```

### [JSON Schema](@id structured_output_respond_schema)

```@example structured_output
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

```@example structured_output
result = respond("List red, green, and blue with their hex codes", text=fmt, service=service, model=model)
if result isa ResponseSuccess
    println(JSON.json(JSON.parse(output_text(result)), 2))
else
    println("Request failed — ", result)
end
```

### [Plain text format](@id structured_output_text)

```@example structured_output
tc = text_format()
println("Default format type: ", tc.format.type)
```

## [Convenience constructors](@id structured_output_constructors)

| Constructor                              | Format                  |
| :--------------------------------------- | :---------------------- |
| `json_object_format()`                   | Unstructured JSON       |
| `json_schema_format(name, desc, schema)` | Schema-constrained JSON |
| `text_format()`                          | Plain text (default)    |

## [See also](@id structured_output_see_also)

- [`ResponseFormat`](@ref) — the format of a `Chat`
- [`TextConfig`](@ref), [`TextFormatSpec`](@ref) — the formats of a `Respond`
- [Decisions with LLMs](@ref jev_llm_guide) — a decision model beside the LLM, for answers that are categories, yes or no, or scores; a whole record of such fields in one request: [Fill a struct in one request](@ref jev_fill_struct)

## [On your provider](@id structured_output_providers)

```@eval
Main.ProviderFacts.on_your_provider(Main.ProviderFacts.llm, :structured_output)
```
