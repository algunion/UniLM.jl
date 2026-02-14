# Result Types

Abstract and concrete types for handling API call outcomes. All result types
are subtypes of [`LLMRequestResponse`](@ref).

## Abstract Type

```@docs
LLMRequestResponse
```

## Chat Completions Results

```@docs
LLMSuccess
LLMFailure
LLMCallError
```

### Pattern Matching

```julia
result = chatrequest!(chat)

if result isa LLMSuccess
    println("Assistant: ", result.message.content)
elseif result isa LLMFailure
    println("HTTP error $(result.status): ", result.response)
elseif result isa LLMCallError
    println("Call error: ", result.error)
end
```

## Responses API Results

```@docs
ResponseSuccess
ResponseFailure
ResponseCallError
```

### Pattern Matching

```julia
result = respond("Tell me a joke")

if result isa ResponseSuccess
    println(output_text(result))
    # => "Why did the Julia programmer bring a ladder?..."
elseif result isa ResponseFailure
    println("HTTP error $(result.status)")
elseif result isa ResponseCallError
    println("Error: ", result.error)
end
```

## Image Generation Results

```@docs
ImageSuccess
ImageFailure
ImageCallError
```

### Pattern Matching

```julia
result = generate_image("A robot writing Julia code")

if result isa ImageSuccess
    imgs = image_data(result)
    println("Generated $(length(imgs)) image(s)")
    save_image(imgs[1], "robot.png")
elseif result isa ImageFailure
    println("HTTP error $(result.status): ", result.response)
elseif result isa ImageCallError
    println("Error: ", result.error)
end
```

## Type Hierarchy

All result types share the abstract parent [`LLMRequestResponse`](@ref):

```
LLMRequestResponse
├── LLMSuccess          (Chat Completions)
├── LLMFailure          (Chat Completions)
├── LLMCallError        (Chat Completions)
├── ResponseSuccess     (Responses API)
├── ResponseFailure     (Responses API)
├── ResponseCallError   (Responses API)
├── ImageSuccess        (Image Generation)
├── ImageFailure        (Image Generation)
└── ImageCallError      (Image Generation)
```
