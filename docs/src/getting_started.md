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

## Your First Request

### Using the Responses API

The simplest way to get started — one function call:

```julia
using UniLM

result = respond("Explain Julia's type system in 3 bullet points")

if result isa ResponseSuccess
    println(output_text(result))
    # • Julia uses a dynamic type system with optional type annotations...
    # • The type hierarchy is rooted at `Any`, with abstract types forming...
    # • Parametric types allow generic programming, e.g., `Array{Float64,2}`...
end
```

### Using Chat Completions

For stateful, multi-turn conversations:

```julia
using UniLM

# Create a chat session
chat = Chat(model="gpt-5.2")

# Build the conversation
push!(chat, Message(Val(:system), "You are a concise Julia tutor."))
push!(chat, Message(Val(:user), "What is multiple dispatch?"))

# Send the request
result = chatrequest!(chat)

if result isa LLMSuccess
    println(result.message.content)
    # => "Multiple dispatch is Julia's core paradigm where the method
    #     called is determined by the types of *all* arguments..."

    # Continue the conversation — history is managed automatically
    push!(chat, Message(Val(:user), "Give me a code example."))
    result2 = chatrequest!(chat)
    println(result2.message.content)
    # => "```julia\nf(x::Int) = x + 1\nf(x::Float64) = x + 0.5\n..."
end
```

### Generating Images

```julia
using UniLM

result = generate_image(
    "A cute robot writing Julia code",
    size="1024x1024",
    quality="high"
)

if result isa ImageSuccess
    save_image(image_data(result)[1], "robot_coder.png")
    println("Image saved!")
end
```

### Using Keyword Arguments

For one-shot requests without managing `Chat` objects:

```julia
result = chatrequest!(
    systemprompt="You are a helpful assistant.",
    userprompt="What is 2+2?",
    model="gpt-5.2-mini",
    temperature=0.0
)
```

## Handling Results

All API calls return subtypes of [`LLMRequestResponse`](@ref). Use Julia's pattern matching:

```@example results
using UniLM

# Construct a chat to show the result type hierarchy
chat = Chat()
push!(chat, Message(Val(:system), "You are helpful."))
push!(chat, Message(Val(:user), "Hello!"))

# Without a real API key, we show the type hierarchy:
println("LLMRequestResponse subtypes:")
for T in subtypes(UniLM.LLMRequestResponse)
    println("  ", T)
end
```

```julia
result = chatrequest!(chat)

if result isa LLMSuccess
    println("Assistant: ", result.message.content)
elseif result isa LLMFailure
    @warn "API returned HTTP $(result.status): $(result.response)"
elseif result isa LLMCallError
    @error "Call failed: $(result.error)"
end
```

For the Responses API:

```julia
result = respond("Hello!")

if result isa ResponseSuccess
    println(output_text(result))
    # => "Hello! How can I help you today?"
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
