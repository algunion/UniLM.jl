# Embeddings

UniLM.jl embeds text with OpenAI, Gemini, local EmbeddingGemma through Ollama, or any
OpenAI-compatible server, chosen by the `service` parameter — and, through Ollama,
images and sound too. The default is OpenAI's `text-embedding-3-small` (1536
dimensions).

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

## Task prompts

Some embedding models read a short prompt that says what a text is for: a query to
search with, or a document to be found. EmbeddingGemma does — v1 and v2, on any
service — and neither Ollama nor an OpenAI-compatible server adds it, so `task`
writes it:

```julia
Embeddings(documents; service=OllamaEndpoint(), model="embeddinggemma-2", task=:document)
Embeddings("How long do refunds take?"; service=OllamaEndpoint(), model="embeddinggemma-2",
           task=:search_query)
```

The query side takes `:search_query`, `:question_answering`, `:fact_checking` or
`:code_retrieval`; the corpus `:document`, with an [`EmbeddingInput`](@ref)'s `title`
when a text has one; and a text compared with others of its kind `:classification`,
`:clustering` or `:sentence_similarity`. An EmbeddingGemma request without a task is
an `ArgumentError` before anything is sent; `task = :none` sends the text as it is,
for a server that writes the prompts itself. Any other model takes `task = :none`
or no task: UniLM knows EmbeddingGemma's prompts only, so write another model's own
prefix into the text. [Local Models with Ollama](@ref ollama_embeddings) lists the
prompt each task writes, and measures what the prompts are worth.

## Local retrieval

Find the passages that answer a question, on your own machine: embed the documents
once with `task = :document`, each question with `task = :search_query`, and rank by
the dot product — EmbeddingGemma's vectors have unit length, so it is the cosine:

```@example local
using UniLM, LinearAlgebra

ollama = OllamaEndpoint()
gemma = "embeddinggemma-2:740m-q8_0"

faq = ["Refunds"  => "Refunds reach the original payment method within 5 working days of approval.",
       "Shipping" => "Orders over 50 EUR ship free within the EU; other orders pay a flat 4.90 EUR.",
       "Returns"  => "Unused items can be returned within 30 days in their original packaging.",
       "Accounts" => "Reset a forgotten password from the sign-in page; the link expires after an hour.",
       "Warranty" => "Electronics carry a two-year warranty against manufacturing defects.",
       "Payments" => "We accept cards, bank transfer and PayPal; invoices arrive by email."]

D = embedding_vectors(embeddingrequest!(Embeddings(
    [EmbeddingInput(text; title) for (title, text) in faq];
    service=ollama, model=gemma, task=:document)))

function search(question; k=2)
    q = embedding_vectors(embeddingrequest!(Embeddings(question;
            service=ollama, model=gemma, task=:search_query)))
    scores = [dot(q, d) for d in D]
    [first(faq[i]) => round(scores[i]; digits=3) for i in partialsortperm(scores, 1:k; rev=true)]
end

search("When will I get my money back?")
```

```@example local
search("Can I send back shoes I never wore?")
```

The first question finds the refund policy, the second the returns policy. The
documents are embedded once; each search embeds only its question. This example ran
on Ollama 0.40.6 with `embeddinggemma-2:740m-q8_0` on 21 October 2026, and the manual
replays its recorded answers.

## Images and audio

Through Ollama, an [`ImageAttachment`](@ref) or [`AudioAttachment`](@ref) is an item
of its own, and `EmbeddingInput(text, image)` embeds a text with its picture as one
item; texts, images and sounds land in one space, so a word can find a picture:

```julia
pictures = Embeddings([ImageAttachment("label.png"), ImageAttachment("box.png")];
                      service=OllamaEndpoint(), model="embeddinggemma-2", task=:document)
```

Only a model that sees or hears embeds media — among EmbeddingGemma 2's builds the
440m ones take images, the 570m ones sound and the 740m ones both, as their tags say
(the 740m ones were measured). Every other service embeds text, and refuses media
before the request. [Local Models with Ollama](@ref ollama_embeddings) shows a
picture and a recording found by words.

## Available Models

| Model                    | Dimensions | Notes                           |
| :----------------------- | :--------- | :------------------------------ |
| `text-embedding-3-small` | 1536       | Default, good balance           |
| `text-embedding-3-large` | 3072       | Higher quality, more dimensions |
| `embeddinggemma-2` (Ollama) | 768, 512, 256 or 128 | Local; 8,192 tokens; text, and images or sound by build; needs a `task` |
| `embeddinggemma` (Ollama) | 768, 512, 256 or 128 | Local; 2,048 tokens; text; needs a `task` |

!!! note
    The buffers start at `something(dimensions, 1536)` zeros per input and are resized to
    whatever length the model returns, so `text-embedding-3-large` (3072 dimensions)
    needs no adjustment.

## Using Other Providers

Pass `service` and `model` to embed with a different backend:

```julia
# Ollama (local): native /api/embed; an input longer than the model's window fails
emb = Embeddings("test"; service=OllamaEndpoint(), model="embeddinggemma-2", task=:search_query)
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

`embeddingrequest!` returns an `EmbeddingSuccess`/`EmbeddingFailure`/`EmbeddingCallError` and fills `emb.embeddings` in place (use `embedding_vectors(result)` for the vectors). It automatically retries transient HTTP statuses (408, 429, 500, 502–504, 520–524, 529) with exponential backoff and jitter — a `Retry-After` header is a floor under the jittered wait — bounded by the resolved [`RequestConfig`](@ref) (`max_attempts`, default 3; `total_deadline`, default 900 s). Pass `config=RequestConfig(max_attempts=1)` to disable retries; timeouts surface as `EmbeddingCallError` with the `UniLMTimeout` in `.cause`, and a cancelled call (`cancel=`, or an ambient `with_cancel` token) with a `UniLMCancelled` there. `embedding_vectors` throws `LLMResultError` on a failure result. `encoding_format` accepts only `"float"` (the vectors are stored as `Float64`); anything else throws `ArgumentError` at construction.

## API Reference

See the [Embeddings API](@ref embeddings_api) page for full type documentation.
