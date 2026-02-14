# Embeddings API

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
emb = Embeddings("The Julia programming language")
embeddingrequest!(emb)
println("First 5 dimensions: ", emb.embeddings[1:5])
# => [0.0023, -0.0091, 0.0148, -0.0032, 0.0076]
println("Embedding norm: ", sqrt(sum(x^2 for x in emb.embeddings)))
# => ≈ 1.0
```
