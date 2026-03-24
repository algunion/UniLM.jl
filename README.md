# UniLM.jl

[![CI](https://github.com/algunion/UniLM.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/algunion/UniLM.jl/actions/workflows/CI.yml)
[![codecov](https://codecov.io/gh/algunion/UniLM.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/algunion/UniLM.jl)
[![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)
[![](https://img.shields.io/badge/%F0%9F%9B%A9%EF%B8%8F_tested_with-JET.jl-233f9a)](https://github.com/aviatesk/JET.jl)
[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://algunion.github.io/UniLM.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://algunion.github.io/UniLM.jl/dev/)

A **Julian**, type-safe interface to **LLM providers** via the OpenAI-compatible API standard — covering the **Chat Completions API**, the **Responses API**, **Image Generation**, **Embeddings**, and **MCP**. Works with OpenAI, Azure, Gemini, Mistral, DeepSeek, Ollama, vLLM, LM Studio, and any OpenAI-compatible provider.

## Features

- **Chat Completions** — stateful conversations with automatic history management
- **Responses API** — OpenAI's newer API with built-in tools, multi-turn chaining, and reasoning
- **Image Generation** — create images from text prompts with `gpt-image-1.5`
- **Tool/Function Calling** — first-class support for function tools in both APIs, with automated `tool_loop`
- **MCP (Model Context Protocol)** — connect to MCP servers or build your own, with seamless tool loop integration
- **Embeddings** — text embedding generation with `text-embedding-3-small`
- **Streaming** — real-time token streaming with `do`-block syntax
- **Structured Output** — JSON Schema–constrained generation
- **Multi-Backend** — OpenAI, Azure OpenAI, and Google Gemini
- **Type Safety** — invalid states are unrepresentable; tested with [JET.jl](https://github.com/aviatesk/JET.jl) and [Aqua.jl](https://github.com/JuliaTesting/Aqua.jl)

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/algunion/UniLM.jl")
```

Or in the Pkg REPL:

```
pkg> add https://github.com/algunion/UniLM.jl
```

## Quick Start

Set your API key:

```bash
export OPENAI_API_KEY="sk-..."
```

### Responses API (recommended for new code)

```julia
julia> using UniLM

julia> result = respond("Explain Julia's multiple dispatch in 2-3 sentences.")

julia> output_text(result)
"Julia's multiple dispatch means a function can have many method definitions, and Julia chooses which one to run based on the types of *all* arguments in a call (not just the first). This makes it easy to write generic code while still getting specialized, high-performance behavior for specific type combinations."

julia> result.response.model
"gpt-5.2-2025-12-11"
```

### Chat Completions

```julia
julia> chat = Chat(model="gpt-4o-mini")

julia> push!(chat, Message(Val(:system), "You are a concise Julia programming tutor."))

julia> push!(chat, Message(Val(:user), "What is multiple dispatch? Answer in 2-3 sentences."))

julia> result = chatrequest!(chat)

julia> result.message.content
"Multiple dispatch is a feature in programming languages, including Julia, that allows the selection of a method to execute based on the types of all its arguments, rather than just the first one. This enables more flexible and expressive code, as it can define different behaviors for a function depending on the combination of argument types. It supports polymorphism, making it easier to write generic code that works with multiple types."

julia> length(chat)  # system + user + assistant
3
```

### One-Shot Convenience

```julia
julia> result = chatrequest!(
           systemprompt="You are a calculator. Respond only with the number.",
           userprompt="What is 42 * 17?",
           model="gpt-4o-mini",
           temperature=0.0
       )

julia> result.message.content
"714"
```

### Image Generation

```julia
julia> result = generate_image(
           "A watercolor painting of a friendly robot reading a Julia programming book",
           size="1024x1024", quality="medium"
       )

julia> result isa ImageSuccess
true

julia> save_image(image_data(result)[1], "robot_julia.png")
"robot_julia.png"
```

### Embeddings

```julia
julia> emb = Embeddings("Julia is a high-performance programming language for technical computing.")

julia> embeddingrequest!(emb)

julia> emb.embeddings[1:5]
5-element Vector{Float64}:
  -0.039474
  -0.009283
   0.001706
  -0.028087
   0.063363
```

### Streaming

```julia
julia> task = respond("Write a haiku about Julia programming.") do chunk, close
           if chunk isa String
               print(chunk)
           elseif chunk isa ResponseObject
               println("\nDone! Status: ", chunk.status)
           end
       end
Multiple dispatch sings,
Types align in swift fusion—
Loops bloom into speed.
Done! Status: completed
```

### Structured Output

```julia
julia> fmt = json_schema_format(
           "languages", "A list of programming languages",
           Dict(
               "type" => "object",
               "properties" => Dict(
                   "languages" => Dict(
                       "type" => "array",
                       "items" => Dict(
                           "type" => "object",
                           "properties" => Dict(
                               "name" => Dict("type" => "string"),
                               "year" => Dict("type" => "integer"),
                               "paradigm" => Dict("type" => "string")
                           ),
                           "required" => ["name", "year", "paradigm"],
                           "additionalProperties" => false
                       )
                   )
               ),
               "required" => ["languages"],
               "additionalProperties" => false
           ),
           strict=true
       )

julia> result = respond("List Julia, Python, and Rust with their release year and primary paradigm.", text=fmt)

julia> JSON.parse(output_text(result))
{
  "languages": [
    {"name": "Julia", "year": 2012, "paradigm": "Multi-paradigm (scientific/numerical, functional, concurrent)"},
    {"name": "Python", "year": 1991, "paradigm": "Multi-paradigm (object-oriented, imperative, functional)"},
    {"name": "Rust", "year": 2010, "paradigm": "Multi-paradigm (systems programming, functional, imperative)"}
  ]
}
```

### Tool / Function Calling

**Responses API:**

```julia
julia> weather_tool = function_tool(
           "get_weather", "Get the current weather for a given location",
           parameters=Dict(
               "type" => "object",
               "properties" => Dict(
                   "location" => Dict("type" => "string", "description" => "City name"),
                   "unit" => Dict("type" => "string", "enum" => ["celsius", "fahrenheit"])
               ),
               "required" => ["location", "unit"],
               "additionalProperties" => false
           ),
           strict=true
       )

julia> result = respond("What's the weather in Tokyo? Use celsius.", tools=[weather_tool])

julia> calls = function_calls(result)

julia> calls[1]["name"]
"get_weather"

julia> JSON.parse(calls[1]["arguments"])
{"location": "Tokyo", "unit": "celsius"}
```

**Web Search:**

```julia
julia> result = respond(
           "What is the latest stable release of the Julia programming language?",
           tools=[web_search()]
       )

julia> output_text(result)
"The latest **stable** release of the Julia programming language is **Julia v1.12.5**."
```

### Multi-Turn Conversations

**Responses API** (via `previous_response_id`):

```julia
julia> r1 = respond("Tell me a one-liner programming joke.", instructions="Be concise.")

julia> output_text(r1)
"There are only 10 kinds of people in the world: those who understand binary and those who don't."

julia> r2 = respond("Explain why that's funny, in one sentence.", previous_response_id=r1.response.id)

julia> output_text(r2)
"It's funny because \"10\" looks like ten in decimal but equals two in binary, so it sets up a nerdy misdirection that only people who know binary immediately get."
```

## Multi-Backend Support

UniLM.jl supports multiple backends. Switch via the `service` parameter:

| Backend       | Type                    | Env Variables                                                               |
| :------------ | :---------------------- | :-------------------------------------------------------------------------- |
| OpenAI        | `OPENAIServiceEndpoint` | `OPENAI_API_KEY`                                                            |
| Azure OpenAI  | `AZUREServiceEndpoint`  | `AZURE_OPENAI_BASE_URL`, `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_API_VERSION` |
| Google Gemini | `GEMINIServiceEndpoint` | `GEMINI_API_KEY`                                                            |

```julia
# Azure
chat = Chat(service=AZUREServiceEndpoint, model="gpt-5.2")

# Gemini
chat = Chat(service=GEMINIServiceEndpoint, model="gemini-2.5-flash")
```

## Two APIs, One Package

| Feature                |       Chat Completions       |            Responses API            |
| :--------------------- | :--------------------------: | :---------------------------------: |
| Stateful conversations |       `Chat` + `push!`       |       `previous_response_id`        |
| System prompt          | `Message(Val(:system), ...)` |        `instructions` kwarg         |
| Tool calling           |  `GPTTool` / `GPTToolCall`   |  `FunctionTool` / `function_tool`   |
| Web search             |              —               |           `WebSearchTool`           |
| File search            |              —               |          `FileSearchTool`           |
| Streaming              |   `stream=true` + callback   |          `do`-block syntax          |
| Structured output      |       `ResponseFormat`       | `TextConfig` / `json_schema_format` |
| Reasoning (O-series)   |              —               |             `Reasoning`             |
| Automated tool loop    |       `tool_loop!`           |          `tool_loop`                |
| MCP integration        |    `mcp_tools` bridge        |   `MCPTool` / `mcp_tool`            |

## Documentation

Full documentation with guides and API reference: **[https://algunion.github.io/UniLM.jl/dev/](https://algunion.github.io/UniLM.jl/dev/)**

- [Getting Started](https://algunion.github.io/UniLM.jl/dev/getting_started/) — setup and first requests
- [Chat Completions Guide](https://algunion.github.io/UniLM.jl/dev/guide/chat_completions/) — `Chat` and `chatrequest!`
- [Responses API Guide](https://algunion.github.io/UniLM.jl/dev/guide/responses_api/) — the newer Responses API
- [Image Generation Guide](https://algunion.github.io/UniLM.jl/dev/guide/image_generation/) — create images from text
- [Tool Calling Guide](https://algunion.github.io/UniLM.jl/dev/guide/tool_calling/) — function calling
- [Streaming Guide](https://algunion.github.io/UniLM.jl/dev/guide/streaming/) — real-time streaming
- [Structured Output Guide](https://algunion.github.io/UniLM.jl/dev/guide/structured_output/) — JSON Schema output
- [Multi-Backend Guide](https://algunion.github.io/UniLM.jl/dev/guide/multi_backend/) — Azure, Gemini
- [MCP Guide](https://algunion.github.io/UniLM.jl/dev/guide/mcp/) — MCP client/server
