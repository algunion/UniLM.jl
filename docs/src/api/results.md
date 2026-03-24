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
    println(result.message.content)
    println(result.message.finish_reason)  # "stop"
elseif result isa LLMFailure
    @warn "HTTP $(result.status): $(result.response)"
elseif result isa LLMCallError
    @error "Call error: $(result.error)"
end
```

## Responses API Results

See also the [Responses API reference](responses.md).

- [`ResponseSuccess`](@ref)
- [`ResponseFailure`](@ref)
- [`ResponseCallError`](@ref)

### Pattern Matching

```julia
result = respond("Tell me a joke")

if result isa ResponseSuccess
    println(output_text(result))
    println(result.response.status)  # "completed"
elseif result isa ResponseFailure
    @warn "HTTP $(result.status)"
elseif result isa ResponseCallError
    @error result.error
end
```

## Image Generation Results

See also the [Images API reference](images.md).

- [`ImageSuccess`](@ref)
- [`ImageFailure`](@ref)
- [`ImageCallError`](@ref)

### Pattern Matching

```julia
result = generate_image("A robot writing Julia code")

if result isa ImageSuccess
    save_image(image_data(result)[1], "robot.png")
elseif result isa ImageFailure
    @warn "HTTP $(result.status): $(result.response)"
elseif result isa ImageCallError
    @error result.error
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
├── ImageCallError      (Image Generation)
├── FIMSuccess          (FIM Completion)
├── FIMFailure          (FIM Completion)
└── FIMCallError        (FIM Completion)
```
