# UniLM.jl

*A unified Julia interface for large language models.*

[![CI](https://github.com/algunion/UniLM.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/algunion/UniLM.jl/actions/workflows/CI.yml)
[![codecov](https://codecov.io/gh/algunion/UniLM.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/algunion/UniLM.jl)
[![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)
[![](https://img.shields.io/badge/%F0%9F%9B%A9%EF%B8%8F_tested_with-JET.jl-233f9a)](https://github.com/aviatesk/JET.jl)
[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://algunion.github.io/UniLM.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://algunion.github.io/UniLM.jl/dev/)

## What is UniLM.jl?

UniLM.jl provides a **Julian**, type-safe interface to **LLM providers** with **first-class native backends** — OpenAI (Chat Completions + Responses), Anthropic (Messages), and Google Gemini (generateContent + agentic Interactions) — plus any **OpenAI-compatible** provider (Azure, DeepSeek, Mistral, Ollama, vLLM, LM Studio). It covers Chat Completions & Responses, a cross-provider agentic `respond` verb, Image Generation/Edits, Embeddings, Files/Vector Stores, Conversations, Audio, Batch, Moderations, Fine-tuning, Webhooks, Realtime, and MCP (client & server) — with built-in token/cost accounting.

### Key Features

- 🗣️ **Chat Completions** — stateful conversations with automatic history management
- 🔮 **Responses API & Agentic Verb** — OpenAI's Responses API plus a cross-provider `respond` verb that also drives Google's Gemini Interactions
- 🖼️ **Image Generation & Edits** — create and edit images with `gpt-image-2`
- 🔧 **Tool/Function Calling** — first-class function tools in both APIs, with an automated `tool_loop`
- 🔌 **MCP (Model Context Protocol)** — connect to MCP servers or build your own, with seamless tool-loop integration
- 📊 **Embeddings** — text embedding generation
- 💰 **Cost & Token Accounting** — per-call `estimated_cost`, per-`Chat` `cumulative_cost`, and a built-in multi-provider pricing table
- 🌊 **Streaming** — real-time token streaming with `do`-block syntax
- 📐 **Structured Output** — JSON Schema–constrained generation
- ☁️ **Multi-Backend** — native OpenAI/Anthropic/Gemini plus Azure, DeepSeek, Ollama, Mistral, vLLM, LM Studio, and any OpenAI-compatible provider
- ✅ **Type Safety & Capability Introspection** — invalid states are unrepresentable and unsupported requests fail fast via `provider_capabilities`; tested with [JET.jl](https://github.com/aviatesk/JET.jl) and [Aqua.jl](https://github.com/JuliaTesting/Aqua.jl)

### Two APIs, One Package

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

## Installation

UniLM requires **Julia 1.12+** and is registered in Julia's General registry:

```julia
using Pkg
Pkg.add("UniLM")
```

Or in the Pkg REPL:

```
pkg> add UniLM
```

For the latest unreleased changes, install from GitHub instead:

```julia
Pkg.add(url="https://github.com/algunion/UniLM.jl")
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
```@example quickstart
result = respond("Explain Julia's multiple dispatch in 2-3 sentences.")
if result isa ResponseSuccess
    println(output_text(result))
else
    println("Request failed — ", output_text(result))
end
```

**Chat Completions:**
```@example quickstart
chat = Chat(model="gpt-4o-mini")
push!(chat, Message(Val(:system), "You are a concise Julia programming tutor."))
push!(chat, Message(Val(:user), "What is multiple dispatch? Answer in 2-3 sentences."))
result = chatrequest!(chat)
if result isa LLMSuccess
    println(result.message.content)
else
    println("Request failed — see result for details")
end
```

**Image Generation:**
```@example quickstart
result = generate_image(
    "A watercolor painting of a friendly robot reading a Julia programming book",
    size="1024x1024", quality="medium"
)
println("Success: ", result isa ImageSuccess)
if result isa ImageSuccess
    save_image(image_data(result)[1], joinpath(@__DIR__, "assets", "generated_robot.png"))
    println("Saved to assets/generated_robot.png")
else
    println("Image generation failed — see result for details")
end
```

![Generated robot reading Julia](assets/generated_robot.png)

## Next Steps

- [Getting Started](@ref) — setup and first requests
- [Chat Completions Guide](@ref chat_guide) — deep dive into `Chat` and `chatrequest!`
- [Responses API Guide](@ref responses_guide) — the newer Responses API
- [Image Generation Guide](@ref images_guide) — create images from text prompts
- [MCP Guide](@ref mcp_guide) — connect to MCP servers or build your own
- [API Reference](@ref chat_api) — full type and function reference

### Platform APIs

Beyond chat and generation, UniLM wraps the full OpenAI platform surface (OpenAI-only) — each has an API-reference page:

- **Storage & retrieval** — [Files](api/files.md), [Vector Stores](api/vector_stores.md), [Conversations](api/conversations.md), [Uploads](api/uploads.md), [Containers](api/containers.md)
- **Generation & media** — [Audio](api/audio.md), [Videos](api/videos.md), [Images](api/images.md)
- **Jobs & ops** — [Batch](api/batch.md), [Fine-tuning](api/fine_tuning.md), [Moderations](api/moderations.md), [Webhooks](api/webhooks.md), [Realtime](api/realtime.md)
- **Cross-cutting** — [Cost Tracking](@ref cost_guide), [Provider Capabilities](api/capabilities.md), [Retrieval & File Search](@ref retrieval_guide)
