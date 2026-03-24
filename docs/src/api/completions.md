# [FIM Types](@id fim_api)

Types and functions for **FIM (Fill-in-the-Middle) Completion** and
**Chat Prefix Completion**.

## Request Type

```@docs
FIMCompletion
```

## Response Types

```@docs
FIMChoice
FIMResponse
```

## Result Types

```@docs
FIMSuccess
FIMFailure
FIMCallError
```

## Request Functions

```@docs
fim_complete
fim_text
prefix_complete
```

## Example

```@example fim_api
using UniLM
using JSON

fim = FIMCompletion(
    service=DeepSeekEndpoint("demo"),
    prompt="def add(a, b):",
    suffix="    return result",
    max_tokens=64
)
println("JSON body:")
println(JSON.json(fim, 2))
```
