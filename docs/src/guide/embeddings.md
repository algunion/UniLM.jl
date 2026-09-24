# Embeddings

UniLM.jl supports text embeddings across multiple providers. The default is
OpenAI's `text-embedding-3-small` (1536 dimensions), but you can target Ollama,
Gemini, Mistral, or any OpenAI-compatible server via the `service` parameter.

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

```@example emb
emb = Embeddings("Julia is a high-performance programming language for technical computing.")
embeddingrequest!(emb)
println("First 5 dimensions:")
for v in emb.embeddings[1:5]
    println("  ", round(v, digits=6))
end
println("L2 norm: ", round(sqrt(sum(x^2 for x in emb.embeddings)), digits=4))
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

```@example emb
embeddingrequest!(emb)
println("Embedding dimensions per text: ", length(emb.embeddings[1]))
```

## Computing Similarity

A common use case is computing cosine similarity between embeddings:

```@example emb
using LinearAlgebra

emb = Embeddings(["Julia", "Python", "Rust", "Fortran"])
embeddingrequest!(emb)

sim = dot(emb.embeddings[1], emb.embeddings[4]) /
      (norm(emb.embeddings[1]) * norm(emb.embeddings[4]))
println("Cosine similarity (Julia vs Fortran): ", round(sim, digits=4))
```

## Available Models

| Model                    | Dimensions | Notes                           |
| :----------------------- | :--------- | :------------------------------ |
| `text-embedding-3-small` | 1536       | Default, good balance           |
| `text-embedding-3-large` | 3072       | Higher quality, more dimensions |

!!! note
    The buffers start at `something(dimensions, 1536)` zeros per input and are resized to
    whatever length the model returns, so `text-embedding-3-large` (3072 dimensions)
    needs no adjustment.

## Using Other Providers

Pass `service` and `model` to embed with a different backend:

```julia
# Ollama (local)
emb = Embeddings("test"; service=OllamaEndpoint(), model="nomic-embed-text")
embeddingrequest!(emb)

# Gemini (OpenAI-compatible endpoint; the native generateContent API has no embeddings)
emb = Embeddings("test"; service=GEMINIOpenAIServiceEndpoint, model="gemini-embedding-001")
embeddingrequest!(emb)
```

!!! note
    Different providers return different embedding dimensions. The buffers are
    pre-allocated for 1536 (OpenAI's `text-embedding-3-small`) and resized to the length
    the provider returns.

!!! note "Gemini embedding models"
    `gemini-embedding-001`, the default for `GEMINIOpenAIServiceEndpoint`, shuts down on
    May 14, 2028. Its successor `gemini-embedding-2` is generally available and works
    through the same OpenAI-compatible endpoint:
    `Embeddings("test"; service=GEMINIOpenAIServiceEndpoint, model="gemini-embedding-2")`.
    The OpenAI-compatible Gemini embeddings endpoint returns no `usage`, so
    [`estimated_cost`](@ref) on its results is `0.0`; the `gemini-embedding-2` price row
    applies only to results that carry usage.

## In-Place Design

The `Embeddings` struct pre-allocates the embedding vectors at construction time.
`embeddingrequest!` fills them **in-place** — no allocation on the hot path when the
model returns the pre-allocated length. This is idiomatic Julia for performance-sensitive
workloads. The result aliases the request (`result.embeddings === emb`), so a second call
on the same `Embeddings` overwrites the first one's vectors: use one `Embeddings` per
concurrent call, and `copy` vectors you keep before reusing a request.

```@example emb
emb = Embeddings("test")
println("Pre-allocated length: ", length(emb.embeddings))
println("All zeros before API call: ", all(x -> x == 0.0, emb.embeddings))
```

## Retry Behaviour

`embeddingrequest!` returns an `EmbeddingSuccess`/`EmbeddingFailure`/`EmbeddingCallError` and fills `emb.embeddings` in place (use `embedding_vectors(result)` for the vectors). It automatically retries transient HTTP statuses (408, 429, 500, 502, 503, 504, 529) with exponential backoff and jitter — a `Retry-After` header is a floor under the jittered wait — bounded by the resolved [`RequestConfig`](@ref) (`max_attempts`, default 3; `total_deadline`, default 900 s). Pass `config=RequestConfig(max_attempts=1)` to disable retries; timeouts surface as `EmbeddingCallError` with the `UniLMTimeout` in `.cause`, and a cancelled call (`cancel=`, or an ambient `with_cancel` token) with a `UniLMCancelled` there. `embedding_vectors` throws `LLMResultError` on a failure result. `encoding_format` accepts only `"float"` (the vectors are stored as `Float64`); anything else throws `ArgumentError` at construction.

## API Reference

See the [Embeddings API](@ref embeddings_api) page for full type documentation.
