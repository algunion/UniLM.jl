# UniLM.jl

*A unified Julia interface for large language models.*

[![Coverage Status](https://coveralls.io/repos/github/algunion/UniLM.jl/badge.svg)](https://coveralls.io/github/algunion/UniLM.jl)
[![](https://img.shields.io/badge/%F0%9F%9B%A9%EF%B8%8F_tested_with-JET.jl-233f9a)](https://github.com/aviatesk/JET.jl)

## What is UniLM.jl?

UniLM.jl provides a **Julian**, type-safe interface to OpenAI's language models — covering both the classic **Chat Completions API**, the newer **Responses API**, and the **Image Generation API**. It aims to become a unified solution for accessing multiple LLM providers from Julia.

### Key Features

- 🗣️ **Chat Completions** — stateful conversations with automatic history management
- 🔮 **Responses API** — OpenAI's newer API with built-in tools, multi-turn chaining, and reasoning
- 🖼️ **Image Generation** — create images from text prompts with `gpt-image-1.5`
- 🔧 **Tool/Function Calling** — first-class support for function tools in both APIs
- 📊 **Embeddings** — text embedding generation
- 🌊 **Streaming** — real-time token streaming with `do`-block syntax
- 📐 **Structured Output** — JSON Schema–constrained generation
- ☁️ **Multi-Backend** — OpenAI, Azure OpenAI, and Google Gemini
- ✅ **Type Safety** — invalid states are unrepresentable; tested with [JET.jl](https://github.com/aviatesk/JET.jl) and [Aqua.jl](https://github.com/JuliaTesting/Aqua.jl)

### Two APIs, One Package

| Feature                |       Chat Completions       |            Responses API            |
| :--------------------- | :--------------------------: | :---------------------------------: |
| Stateful conversations |       `Chat` + `push!`       |       `previous_response_id`        |
| System prompt          | `Message(Val(:system), ...)` |        `instructions` kwarg         |
| Tool calling           |  `GPTTool` / `GPTToolCall`   |  `FunctionTool` / `function_tool`   |
| Web search             |              ✗               |           `WebSearchTool`           |
| File search            |              ✗               |          `FileSearchTool`           |
| Streaming              |   `stream=true` + callback   |          `do`-block syntax          |
| Structured output      |       `ResponseFormat`       | `TextConfig` / `json_schema_format` |
| Reasoning (O-series)   |              ✗               |             `Reasoning`             |

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/algunion/UniLM.jl")
```

Or in the Pkg REPL:

```
pkg> add https://github.com/algunion/UniLM.jl
```

## Quick Example

Building requests — these construct objects locally without calling the API:

```@example quickstart
using UniLM
using JSON

# Chat Completions request
chat = Chat(model="gpt-5.2")
push!(chat, Message(Val(:system), "You are a Julia expert."))
push!(chat, Message(Val(:user), "Explain multiple dispatch in one sentence."))
println("Chat has ", length(chat), " messages, model: ", chat.model)
println("Request body preview:")
println(JSON.json(chat))
```

```@example quickstart
# Responses API request
r = Respond(input="What makes Julia special?")
println("Respond model: ", r.model)
println(JSON.json(r))
```

```@example quickstart
# Image Generation request
ig = ImageGeneration(prompt="A watercolor Julia logo", quality="high")
println("Image model: ", ig.model)
println(JSON.json(ig))
```

With a valid API key, actual API calls return structured results:

**Responses API** (recommended for new code):
```julia
julia> result = respond("Explain Julia's multiple dispatch in 2-3 sentences.")

julia> output_text(result)
"Julia's multiple dispatch means a function can have many method definitions, and Julia chooses which one to run based on the types of *all* arguments in a call (not just the first). This makes it easy to write generic code while still getting specialized, high-performance behavior for specific type combinations."
```

**Chat Completions:**
```julia
julia> chat = Chat(model="gpt-4o-mini")
julia> push!(chat, Message(Val(:system), "You are a concise Julia programming tutor."))
julia> push!(chat, Message(Val(:user), "What is multiple dispatch? Answer in 2-3 sentences."))

julia> result = chatrequest!(chat)

julia> result.message.content
"Multiple dispatch is a feature in programming languages, including Julia, that allows the selection of a method to execute based on the types of all its arguments, rather than just the first one. This enables more flexible and expressive code, as it can define different behaviors for a function depending on the combination of argument types. It supports polymorphism, making it easier to write generic code that works with multiple types."
```

**Image Generation:**
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

![Generated robot reading Julia](assets/generated_robot.png)

## Next Steps

- [Getting Started](@ref) — setup and first requests
- [Chat Completions Guide](@ref chat_guide) — deep dive into `Chat` and `chatrequest!`
- [Responses API Guide](@ref responses_guide) — the newer Responses API
- [Image Generation Guide](@ref images_guide) — create images from text prompts
- [API Reference](@ref chat_api) — full type and function reference
