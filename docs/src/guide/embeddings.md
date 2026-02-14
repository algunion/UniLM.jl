# Embeddings

UniLM.jl supports text embeddings via the OpenAI Embeddings API using the
`text-embedding-3-small` model (1536 dimensions) by default.

## Basic Usage

```@example emb
using UniLM

# Single text
emb = Embeddings("Julia is a high-performance programming language")
println("Model: ", emb.model)
println("Embedding dimensions: ", length(emb.embeddings))
println("Pre-allocated (all zeros): ", all(x -> x == 0.0, emb.embeddings))
```

After calling the API, the embeddings are filled in-place:

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

## Batch Embeddings

Embed multiple texts in a single API call:

```@example emb
texts = [
    "Julia is fast",
    "Python is popular",
    "Rust is safe"
]

emb = Embeddings(texts)
println("Model: ", emb.model)
println("Number of texts: ", length(emb.input))
println("Embeddings per text: ", length(emb.embeddings[1]), " dimensions")
```

```julia
embeddingrequest!(emb)

# Each embedding is a 1536-dimensional Float64 vector
length(emb.embeddings[1])  # 1536
length(emb.embeddings[2])  # 1536
```

## Computing Similarity

A common use case is computing cosine similarity between embeddings:

```julia
using LinearAlgebra

emb = Embeddings(["Julia", "Python", "Rust", "Fortran"])
embeddingrequest!(emb)

# Cosine similarity between embeddings
sim = dot(emb.embeddings[1], emb.embeddings[4]) /
      (norm(emb.embeddings[1]) * norm(emb.embeddings[4]))
# Similarity score between 0 and 1
```

## Available Models

| Model                    | Dimensions | Notes                           |
| :----------------------- | :--------- | :------------------------------ |
| `text-embedding-3-small` | 1536       | Default, good balance           |
| `text-embedding-3-large` | 3072       | Higher quality, more dimensions |

!!! note
    The default `Embeddings` constructor pre-allocates for 1536 dimensions
    (`text-embedding-3-small`). To use `text-embedding-3-large`, you would need
    to adjust the embedding vector size accordingly.

## In-Place Design

The `Embeddings` struct pre-allocates the embedding vectors at construction time.
`embeddingrequest!` fills them **in-place** — no allocation on the hot path. This is
idiomatic Julia for performance-sensitive workloads.

```@example emb
emb = Embeddings("test")
println("Pre-allocated length: ", length(emb.embeddings))
println("All zeros before API call: ", all(x -> x == 0.0, emb.embeddings))
```

## API Reference

See the [Embeddings API](@ref embeddings_api) page for full type documentation.
