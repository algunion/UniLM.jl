# Getting Started

## Prerequisites

- **Julia 1.12+** (as specified in `Project.toml`)
- An **OpenAI API key** (or Azure/Gemini credentials for those backends)

## Installation

```julia
using Pkg
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
export AZURE_OPENAI_API_VERSION="2024-02-01"
export AZURE_OPENAI_DEPLOY_NAME_GPT_5_2="your-gpt52-deployment"
```

### Google Gemini

```bash
export GEMINI_API_KEY="your-gemini-key"
```

```@setup gs
using UniLM
using JSON
```

## Your First Request

### Using the Responses API

The simplest way to get started — one function call:

```@example gs
result = respond("Explain Julia's type system in 3 bullet points")
println(output_text(result))
```

### Using Chat Completions

For stateful, multi-turn conversations:

```@example gs
chat = Chat(model="gpt-4o-mini")
push!(chat, Message(Val(:system), "You are a concise Julia programming tutor."))
push!(chat, Message(Val(:user), "What is multiple dispatch? Answer in 2-3 sentences."))
result = chatrequest!(chat)
println(result.message.content)
println("\nFinish reason: ", result.message.finish_reason)
println("Conversation length: ", length(chat))
```

### Generating Images

```@example gs
result = generate_image(
    "A watercolor painting of a friendly robot reading a Julia programming book",
    size="1024x1024", quality="medium"
)
println("Success: ", result isa ImageSuccess)
println("Images: ", length(image_data(result)))
```

### Using Keyword Arguments

For one-shot requests without managing `Chat` objects:

```@example gs
result = chatrequest!(
    systemprompt="You are a calculator. Respond only with the number.",
    userprompt="What is 42 * 17?",
    model="gpt-4o-mini",
    temperature=0.0
)
println(result.message.content)
```

## Handling Results

All API calls return subtypes of [`LLMRequestResponse`](@ref). Use Julia's pattern matching:

```@example results
using UniLM
using InteractiveUtils

# Construct a chat to show the result type hierarchy
chat = Chat()
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
result = respond("Hello!")

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
| Use Azure or Gemini            | [Multi-Backend Guide](@ref backend_guide)        |
