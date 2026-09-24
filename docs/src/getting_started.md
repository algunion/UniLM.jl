# Getting Started

## Prerequisites

- **Julia 1.13+** (as specified in `Project.toml`)
- An **API key** for your chosen provider (OpenAI, DeepSeek, Gemini, Mistral, etc.) — or none at all for local providers like Ollama

## Installation

UniLM is registered in Julia's General registry:

```julia
using Pkg
Pkg.add("UniLM")
```

Or from the Pkg REPL:

```
pkg> add UniLM
```

To track the latest unreleased changes, install from GitHub instead:

```julia
Pkg.add(url="https://github.com/algunion/UniLM.jl")
```

## Configuration

UniLM.jl reads API credentials from environment variables. Set them before making
any requests:

### OpenAI (default)

```julia
ENV["OPENAI_API_KEY"] = "sk-..."
```

Or via your shell:

```bash
export OPENAI_API_KEY="sk-..."
```

### Azure OpenAI

```bash
export AZURE_OPENAI_BASE_URL="https://your-resource.openai.azure.com"
export AZURE_OPENAI_API_KEY="your-key"
export AZURE_OPENAI_API_VERSION="2024-10-21"
export AZURE_OPENAI_DEPLOY_NAME_GPT_5_2="your-gpt52-deployment"
```

Each model needs its deployment: `AZURE_OPENAI_DEPLOY_NAME_<MODEL>`, the model id
upper-cased with every other character mapped to `_` (`gpt-5.2` →
`AZURE_OPENAI_DEPLOY_NAME_GPT_5_2`), read when a request is built — or register one with
[`add_azure_deploy_name!`](@ref).

### Google Gemini

Native `generateContent` API (default model `gemini-3.8-flash`):

```bash
export GEMINI_API_KEY="your-gemini-key"
```

### Anthropic (Claude)

Native Messages API (default model `claude-opus-5-5`, default `max_tokens` 16000):

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

### DeepSeek

Default model `deepseek-flash` (chat and FIM):

```bash
export DEEPSEEK_API_KEY="sk-..."
```

### TypeSafe System One (Jev)

Typed judgments and calibrated probabilities instead of generated text — routing,
screening, ranking and verification that your code can act on directly:

```bash
export TYPESAFE_API_KEY="..."
```

Optional: `TYPESAFE_BASE_URL` overrides the API root (default
`https://api.typesafe.ai`) and `TYPESAFE_DEFAULT_MODEL` the model used when a call
names none (default `jev-latest`). See [Typed Judgments with Jev](@ref system_one_guide).

### Ollama (local — no key needed)

Just have the Ollama server running on `localhost:11434`. No API key required.

```@setup gs
using UniLM
using JSON
```

!!! tip "Try it free, locally"
    Hosted API calls cost money and need a funded key. To experiment with **zero cost and no
    signup**, run a local model with [Ollama](https://ollama.com) — set
    `service=OllamaEndpoint()` (no key required), as configured above.

## Your First Request

### Using the Responses API

The simplest way to get started — one function call:

```@example gs
result = respond("Explain Julia's type system in 3 bullet points", model="gpt-5.4-mini")
if result isa ResponseSuccess
    println(output_text(result))
else
    println("Request failed — ", result)
end
```

### Using Chat Completions

For stateful, multi-turn conversations:

```@example gs
chat = Chat(model="gpt-5.4-mini")
push!(chat, Message(Val(:system), "You are a concise Julia programming tutor."))
push!(chat, Message(Val(:user), "What is multiple dispatch? Answer in 2-3 sentences."))
result = chatrequest!(chat)
if result isa LLMSuccess
    println(result.message.content)
    println("\nFinish reason: ", result.message.finish_reason)
    println("Conversation length: ", length(chat))
else
    println("Request failed — see result for details")
end
```

### Generating Images

```julia
result = generate_image(
    "A watercolor painting of a friendly robot reading a Julia programming book",
    size="1024x1024", quality="medium"
)
if result isa ImageSuccess
    println("Success: true")
    println("Images: ", length(image_data(result)))
else
    println("Success: false")
    println("Images: 0")
end
```

### Using Keyword Arguments

For one-shot requests without managing `Chat` objects:

```@example gs
result = chatrequest!(
    systemprompt="You are a calculator. Respond only with the number.",
    userprompt="What is 42 * 17?",
    model="gpt-5.4-mini",
    temperature=0.0
)
if result isa LLMSuccess
    println(result.message.content)
else
    println("Request failed — see result for details")
end
```

## Handling Results

All API calls return subtypes of [`LLMRequestResponse`](@ref). Use Julia's pattern matching:

```@example results
using UniLM
using InteractiveUtils

# Construct a chat to show the result type hierarchy
chat = Chat(model="gpt-5.4-mini")
push!(chat, Message(Val(:system), "You are helpful."))
push!(chat, Message(Val(:user), "Hello!"))

# Show the type hierarchy:
println("LLMRequestResponse subtypes:")
for T in subtypes(UniLM.LLMRequestResponse)
    println("  ", T)
end
```

```@example results
result = chatrequest!(chat)

if result isa LLMSuccess
    println("Assistant: ", result.message.content)
    println("Finish reason: ", result.message.finish_reason)
elseif result isa LLMFailure
    @warn "API returned HTTP $(result.status): $(result.response)"
elseif result isa LLMCallError
    @error "Call failed: $(result.error)"
end
```

For the Responses API:

```@example results
result = respond("Hello!", model="gpt-5.4-mini")

if result isa ResponseSuccess
    println(output_text(result))
    println("Status: ", result.response.status)
    println("Model: ", result.response.model)
elseif result isa ResponseFailure
    @warn "HTTP $(result.status)"
elseif result isa ResponseCallError
    @error result.error
end
```

## What's Next?

| Want to...                     | Read...                                          |
| :----------------------------- | :----------------------------------------------- |
| Build multi-turn conversations | [Chat Completions Guide](@ref chat_guide)        |
| Use the newer Responses API    | [Responses API Guide](@ref responses_guide)      |
| Generate images from prompts   | [Image Generation Guide](@ref images_guide)      |
| Call functions from the model  | [Tool Calling Guide](@ref tools_guide)           |
| Stream tokens in real-time     | [Streaming Guide](@ref streaming_guide)          |
| Get structured JSON output     | [Structured Output Guide](@ref structured_guide) |
| Use any provider               | [Multi-Backend Guide](@ref backend_guide)        |
| Track token usage & cost       | [Cost Tracking Guide](@ref cost_guide)           |
| Ground answers in your files   | [Retrieval & File Search](@ref retrieval_guide)  |
| Bound timeouts and retries     | [Timeouts & Retries](@ref timeouts_guide)        |
| Fan out, stream to tasks, cancel | [Concurrency, Tasks and Cancellation](@ref concurrency_guide) |
