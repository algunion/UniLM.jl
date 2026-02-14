# Embeddings

UniLM.jl supports text embeddings via the OpenAI Embeddings API using the
`text-embedding-ada-002` model (1536 dimensions).

## Basic Usage

```julia
using UniLM

# Single text
emb = Embeddings("Julia is a high-performance programming language")
embeddingrequest!(emb)

# The embedding vector is stored in-place
emb.embeddings  # => Vector{Float64} of length 1536
```

## Batch Embeddings

Embed multiple texts in a single API call:

```julia
texts = [
    "Julia is fast",
    "Python is popular",
    "Rust is safe"
]

emb = Embeddings(texts)
embeddingrequest!(emb)

# Each embedding is accessible by index
emb.embeddings[1]  # => Vector{Float64} for "Julia is fast"
emb.embeddings[2]  # => Vector{Float64} for "Python is popular"
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
```

## In-Place Design

The `Embeddings` struct pre-allocates the embedding vectors at construction time.
`embeddingrequest!` fills them **in-place** — no allocation on the hot path. This is
idiomatic Julia for performance-sensitive workloads.

```julia
emb = Embeddings("test")       # pre-allocates zeros(1536)
embeddingrequest!(emb)          # fills in-place
@assert length(emb.embeddings) == 1536
```

## API Reference

See the [Embeddings API](@ref embeddings_api) page for full type documentation.
