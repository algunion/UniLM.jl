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
embeddingrequest!(emb)
emb.embeddings  # => Float64[0.0123, -0.0456, ...] (1536 dims)
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

# Each embedding is accessible by index
emb.embeddings[1]  # => Float64[...] for "Julia is fast"
emb.embeddings[2]  # => Float64[...] for "Python is popular"
```

## Computing Similarity

A common use case is computing cosine similarity between embeddings:

```julia
using LinearAlgebra

emb = Embeddings(["Julia", "Python", "Rust", "Fortran"])
embeddingrequest!(emb)

# Cosine similarity between "Julia" and "Fortran"
sim = dot(emb.embeddings[1], emb.embeddings[4]) /
      (norm(emb.embeddings[1]) * norm(emb.embeddings[4]))
# => 0.82 (high similarity — both scientific computing languages)
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
