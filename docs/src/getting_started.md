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
julia> using UniLM

julia> result = respond("Explain Julia's type system in 3 bullet points")

julia> output_text(result)
"Julia's multiple dispatch means a function can have many method definitions, and Julia chooses which one to run based on the types of *all* arguments in a call (not just the first). This makes it easy to write generic code while still getting specialized, high-performance behavior for specific type combinations."
```

### Using Chat Completions

For stateful, multi-turn conversations:

```julia
julia> using UniLM

julia> chat = Chat(model="gpt-4o-mini")

julia> push!(chat, Message(Val(:system), "You are a concise Julia programming tutor."))

julia> push!(chat, Message(Val(:user), "What is multiple dispatch? Answer in 2-3 sentences."))

julia> result = chatrequest!(chat)

julia> result.message.content
"Multiple dispatch is a feature in programming languages, including Julia, that allows the selection of a method to execute based on the types of all its arguments, rather than just the first one. This enables more flexible and expressive code, as it can define different behaviors for a function depending on the combination of argument types. It supports polymorphism, making it easier to write generic code that works with multiple types."

julia> result.message.finish_reason
"stop"

julia> length(chat)  # system + user + assistant
3
```

### Generating Images

```julia
julia> result = generate_image(
           "A watercolor painting of a friendly robot reading a Julia programming book",
           size="1024x1024", quality="medium"
       )

julia> result isa ImageSuccess
true

julia> length(image_data(result))
1

julia> save_image(image_data(result)[1], "robot_julia.png")
"robot_julia.png"
```

### Using Keyword Arguments

For one-shot requests without managing `Chat` objects:

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

```julia
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

```julia
result = respond("Hello!")

if result isa ResponseSuccess
    println(output_text(result))
    println("Status: ", result.response.status)  # "completed"
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
