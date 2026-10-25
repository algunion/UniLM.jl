# [Embeddings](@id embeddings_guide)

```@setup embeddings
using UniLM
```

An embedding is a vector of numbers that stands for an input: inputs with close meanings get vectors that
lie close together, so texts can be searched, grouped and compared by their vectors. An
[`Embeddings`](@ref) request holds one input or several, and [`embeddingrequest!`](@ref) sends it and fills
its vectors in place. Which providers embed, and what they embed besides text, is on each provider's entry.

```@example embeddings
service, model = OPENAIServiceEndpoint, "text-embedding-3-small"
nothing # hide
```

The examples use only the names bound above. `service` and `model` are the provider and model they run
on: bind your own there. The table [at the end](@ref embeddings_providers) links each provider's rules.

## [Embedding a text](@id embeddings_basic)

A request holds a vector for each input before it is sent, all zeros:

```@example embeddings
emb = Embeddings("Julia is a high-performance programming language"; service, model)
println("Model: ", emb.model)
println("Embedding dimensions: ", length(emb.embeddings))
println("Pre-allocated (all zeros): ", all(x -> x == 0.0, emb.embeddings))
```

`embeddingrequest!` sends it, and the reply fills the vectors in place:

```@example embeddings
emb = Embeddings("Julia is a high-performance programming language for technical computing."; service, model)
embeddingrequest!(emb)
println("First 5 dimensions:")
for v in emb.embeddings[1:5]
    println("  ", round(v, digits=6))
end
println("L2 norm: ", round(sqrt(sum(x^2 for x in emb.embeddings)), digits=4))
```

An `Embeddings` without `model` takes the provider's default embedding model, and its constructor throws
`ArgumentError` for a provider that has none. An endpoint whose declared capabilities lack `:embeddings`
refuses the request with an `ArgumentError` before anything is sent
([Capabilities by provider](@ref capabilities_guide)).

## [Several texts at once](@id embeddings_batch)

A vector of texts is one request, with a vector for each text:

```@example embeddings
texts = [
    "Julia is fast",
    "Python is popular",
    "Rust is safe"
]

emb = Embeddings(texts; service, model)
println("Model: ", emb.model)
println("Number of texts: ", length(emb.input))
println("Embeddings per text: ", length(emb.embeddings[1]), " dimensions")
```

```@example embeddings
embeddingrequest!(emb)
println("Embedding dimensions per text: ", length(emb.embeddings[1]))
```

## [Comparing texts](@id embeddings_similarity)

The cosine of two vectors measures how close their texts are:

```@example embeddings
using LinearAlgebra

emb = Embeddings(["Julia", "Python", "Rust", "Fortran"]; service, model)
embeddingrequest!(emb)

sim = dot(emb.embeddings[1], emb.embeddings[4]) /
      (norm(emb.embeddings[1]) * norm(emb.embeddings[4]))
println("Cosine similarity (Julia vs Fortran): ", round(sim, digits=4))
```

## [Task prompts and titles](@id embeddings_tasks)

Some embedding models read a short prompt in front of a text that says what the text is for: a query to
search with, or a document to be found. For the models UniLM knows to read one, `task` writes it:
`:search_query`, `:question_answering`, `:fact_checking` or `:code_retrieval` for the query side;
`:document` for the corpus, with an [`EmbeddingInput`](@ref)'s `title` when a text has one; and
`:classification`, `:clustering` or `:sentence_similarity` for a text compared with others of its kind.
`task = :none` sends the text as it is. Which models read a prompt, and the prompt each task writes, is on
the entry of the provider that serves them.

A model whose prompts UniLM does not know takes `task = :none` or no task, and no `title`: anything else
is refused before the request, so write such a model's own prefix into the text instead:

```@example embeddings
refusal = try Embeddings("How long do refunds take?"; service, model, task=:search_query); nothing catch err; err end
refusal isa ArgumentError || error("expected the task to be refused")
print(refusal.msg)
```

## [Images and sound](@id embeddings_media)

An input can also be an [`ImageAttachment`](@ref), an [`AudioAttachment`](@ref), or an
[`EmbeddingInput`](@ref) that holds a text with an image or a sound clip. A provider that embeds text alone
refuses such an input before the request; one that embeds media says so on its entry.

## [Vector sizes](@id embeddings_dimensions)

`dimensions` asks for vectors of that length, where the model takes it; a value that is not positive
throws `ArgumentError`. The buffers start at `something(dimensions, 1536)` zeros per input and take the
length the reply has, so a model with longer or shorter vectors needs no adjustment.

## [In-place design](@id embeddings_in_place)

The `Embeddings` struct pre-allocates the embedding vectors at construction time.
`embeddingrequest!` fills them **in-place** — no allocation on the hot path when the
model returns the pre-allocated length. This is idiomatic Julia for performance-sensitive
workloads. The result aliases the request (`result.embeddings === emb`), so a second call
on the same `Embeddings` overwrites the first one's vectors: use one `Embeddings` per
concurrent call, and `copy` vectors you keep before reusing a request.

```@example embeddings
emb = Embeddings("test"; service, model)
println("Pre-allocated length: ", length(emb.embeddings))
println("All zeros before API call: ", all(x -> x == 0.0, emb.embeddings))
```

## [Results and retries](@id embeddings_retries)

`embeddingrequest!` returns an `EmbeddingSuccess`/`EmbeddingFailure`/`EmbeddingCallError` and fills `emb.embeddings` in place (use `embedding_vectors(result)` for the vectors). It automatically retries transient HTTP statuses (408, 429, 500, 502–504, 520–524, 529) with exponential backoff and jitter — a `Retry-After` header is a floor under the jittered wait — bounded by the resolved [`RequestConfig`](@ref) (`max_attempts`, default 3; `total_deadline`, default 900 s). Pass `config=RequestConfig(max_attempts=1)` to disable retries; timeouts surface as `EmbeddingCallError` with the `UniLMTimeout` in `.cause`, and a cancelled call (`cancel=`, or an ambient `with_cancel` token) with a `UniLMCancelled` there. `embedding_vectors` throws `LLMResultError` on a failure result. `encoding_format` accepts only `"float"` (the vectors are stored as `Float64`); anything else throws `ArgumentError` at construction.

## [See also](@id embeddings_see_also)

- [Embeddings API](@ref embeddings_api) — every type and function
- [Cost Tracking](@ref cost_guide) — what an embeddings call costs

## [On your provider](@id embeddings_providers)

```@eval
Main.ProviderFacts.on_your_provider(Main.ProviderFacts.llm, :embeddings)
```
