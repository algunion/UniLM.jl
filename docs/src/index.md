# UniLM.jl

*A unified Julia interface for large language models.*

[![Coverage Status](https://coveralls.io/repos/github/algunion/UniLM.jl/badge.svg)](https://coveralls.io/github/algunion/UniLM.jl)
[![](https://img.shields.io/badge/%F0%9F%9B%A9%EF%B8%8F_tested_with-JET.jl-233f9a)](https://github.com/aviatesk/JET.jl)

## What is UniLM.jl?

UniLM.jl provides a **Julian**, type-safe interface to OpenAI's language models — covering both the classic **Chat Completions API** and the newer **Responses API**. It aims to become a unified solution for accessing multiple LLM providers from Julia.

### Key Features

- 🗣️ **Chat Completions** — stateful conversations with automatic history management
- 🔮 **Responses API** — OpenAI's newer API with built-in tools, multi-turn chaining, and reasoning
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

```julia
using UniLM

# Set your API key
ENV["OPENAI_API_KEY"] = "sk-..."

# --- Responses API (recommended for new code) ---
result = respond("What makes Julia special?")
println(output_text(result))

# --- Chat Completions ---
chat = Chat(model="gpt-4o")
push!(chat, Message(Val(:system), "You are a Julia expert."))
push!(chat, Message(Val(:user), "Explain multiple dispatch in one sentence."))
result = chatrequest!(chat)
if result isa LLMSuccess
    println(result.message.content)
end
```

## Next Steps

- [Getting Started](@ref) — setup and first requests
- [Chat Completions Guide](@ref chat_guide) — deep dive into `Chat` and `chatrequest!`
- [Responses API Guide](@ref responses_guide) — the newer Responses API
- [API Reference](@ref chat_api) — full type and function reference
