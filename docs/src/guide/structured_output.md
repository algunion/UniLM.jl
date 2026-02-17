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
result = respond("List 3 colors as a JSON object", text=json_object_format())
if result isa ResponseSuccess
    println(output_text(result))
else
    println("Request failed — ", output_text(result))
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
result = respond("List red, green, and blue with their hex codes", text=fmt)
if result isa ResponseSuccess
    println(JSON.json(JSON.parse(output_text(result)), 2))
else
    println("Request failed — ", output_text(result))
end
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

## Using DescribedTypes.jl

[DescribedTypes.jl](https://github.com/algunion/DescribedTypes.jl) lets you derive JSON Schemas directly from Julia structs instead of hand-writing `Dict` literals. Define your types, annotate them, and let the library produce the exact format OpenAI expects.

**Why use DescribedTypes.jl?**

- **Annotate types you don't own.** The `annotate` method can be defined for _any_ type — including those from third-party packages or Base. This means you can attach rich schema metadata (names, descriptions, enum constraints) to existing structs without modifying their source code.
- **Higher-quality structured output.** Bare JSON Schemas carry only type information. DescribedTypes adds field-level descriptions and constrained enum values that guide the model, producing more accurate and consistent responses compared to raw schemas.
- **Single source of truth.** The Julia struct _is_ the schema. Changes to your types automatically propagate to the generated schema — no more keeping `Dict` literals in sync by hand.

!!! note
    Install the package first: `Pkg.add(url="https://github.com/algunion/DescribedTypes.jl")`.

### Defining Annotated Types

```@example described_structured
using UniLM, DescribedTypes, JSON

struct Language
    name::String
    year::Int
    paradigm::String
end

struct LanguageList
    languages::Vector{Language}
end

DescribedTypes.annotate(::Type{Language}) = Annotation(
    name="Language",
    description="A programming language.",
    parameters=Dict(
        :name     => Annotation(name="name", description="Name of the language"),
        :year     => Annotation(name="year", description="Year of first release"),
        :paradigm => Annotation(name="paradigm", description="Primary paradigm",
                                enum=["functional", "imperative", "object-oriented", "multi-paradigm"]),
    ),
)

DescribedTypes.annotate(::Type{LanguageList}) = Annotation(
    name="language_list",
    description="A list of programming languages with metadata.",
    parameters=Dict(
        :languages => Annotation(name="languages", description="The language entries"),
    ),
)
nothing  # hide
```

### Chat Completions with DescribedTypes

Use the `OPENAI` adapter to generate a schema and pass it directly to `ResponseFormat` — multiple dispatch takes care of the rest:

```@example described_structured
s = schema(LanguageList, llm_adapter=OPENAI)
println(JSON.json(s, 2))
```

```@example described_structured
chat = Chat(model="gpt-5.2", response_format=ResponseFormat(s))
push!(chat, Message(Val(:system), "Return structured data about programming languages."))
push!(chat, Message(Val(:user), "List Julia, Python, and Rust"))
println("Request body:")
println(JSON.json(chat))
```

```@example described_structured
result = chatrequest!(chat)
if result isa LLMSuccess
    println(JSON.json(JSON.parse(result.message.content), 2))
else
    println("Request failed — see result for details")
end
```

### Responses API with DescribedTypes

The same schema dict dispatches to `json_schema_format` just as cleanly:

```@example described_structured
fmt = json_schema_format(s)
println("Format type: ", fmt.format.type)
println("Schema strict: ", fmt.format.strict)
```

```@example described_structured
result = respond("List Julia, Python, and Rust with their year and paradigm", text=json_schema_format(s))
if result isa ResponseSuccess
    println(JSON.json(JSON.parse(output_text(result)), 2))
else
    println("Request failed — ", output_text(result))
end
```

### Optional Fields

Mark a field as `Union{Nothing, T}` and DescribedTypes will handle it correctly — the field stays required in the schema but its type becomes a union with `null`:

```@example described_structured
struct MovieReview
    title::String
    rating::Int
    comment::Union{Nothing, String}
end

DescribedTypes.annotate(::Type{MovieReview}) = Annotation(
    name="movie_review",
    description="A movie review with optional comment.",
    parameters=Dict(
        :title   => Annotation(name="title",   description="Movie title"),
        :rating  => Annotation(name="rating",  description="Rating from 1 to 10"),
        :comment => Annotation(name="comment",  description="Optional review text"),
    ),
)

println(JSON.json(schema(MovieReview, llm_adapter=OPENAI), 2))
```

### Type Mapping Reference

| Julia type                  | JSON Schema type  |
| :-------------------------- | :---------------- |
| `String` / `AbstractString` | `string`          |
| `Bool`                      | `boolean`         |
| `<:Integer`                 | `integer`         |
| `<:Real`                    | `number`          |
| `Nothing` / `Missing`       | `null`            |
| `<:AbstractArray`           | `array`           |
| `<:Enum`                    | `string` + `enum` |
| Any other `struct`          | `object`          |

## See Also

- [`ResponseFormat`](@ref) — Chat Completions format type
- [`TextConfig`](@ref), [`TextFormatSpec`](@ref) — Responses API format types
- [DescribedTypes.jl](https://github.com/algunion/DescribedTypes.jl) — Schema generation from Julia types
