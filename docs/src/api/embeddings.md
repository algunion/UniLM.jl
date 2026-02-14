# [Embeddings API](@id embeddings_api)

Types and functions for the **Embeddings API**.

## Embeddings Object

```@docs
Embeddings
```

### Construction

```@example embeddings_api
using UniLM

# Single input
emb = Embeddings("Julia is a great language")
println("Model: ", emb.model)
println("Embedding dims: ", length(emb.embeddings))

# Batch input
batch = Embeddings(["Hello", "World", "Julia"])
println("Batch size: ", length(batch.input))
println("Each embedding dims: ", length(batch.embeddings[1]))
```

## Request Function

```@docs
embeddingrequest!
```

## Model Constants

```@example embeddings_api
println("Default embedding model: ", UniLM.GPTTextEmbedding3Small)
```

## Usage Example

```julia
julia> emb = Embeddings("Julia is a high-performance programming language for technical computing.")

julia> embeddingrequest!(emb)

julia> emb.embeddings[1:5]  # first 5 dimensions
5-element Vector{Float64}:
  -0.039474
  -0.009283
  0.001706
  -0.028087
  0.063363

julia> sqrt(sum(x^2 for x in emb.embeddings))  # L2 norm ≈ 1.0
1.0
```
