# [Embeddings](@id ollama_embeddings_page)

```@setup ollama
using UniLM
```

An [`Embeddings`](@ref) request on `OllamaEndpoint` goes to Ollama's `/api/embed`, which "returns
L2‑normalized (unit‑length) vectors" ([embeddings](https://docs.ollama.com/capabilities/embeddings)).
It holds one item or several — a text, an [`EmbeddingInput`](@ref), an image or a sound clip — and
names an installed embedding model, since Ollama has no default; `embeddingrequest!` fills its vectors
in place. The usage counts the input tokens, Ollama's `prompt_eval_count`, and a local model's
embeddings cost `0.0`, as its chats do ([First call](@ref ollama_first)). The request carries the
endpoint's `keep_alive`, options and `runner` ([Options](@ref ollama_options)).

## [Embeddings with EmbeddingGemma 2](@id ollama_embeddings)

EmbeddingGemma reads a short prompt that says what a text is for, and Ollama adds none: the
Modelfile template of `embeddinggemma-2:740m-q8_0` is the text alone, `{{ .Prompt }}`, as
`ollama show --modelfile` printed it on 25 October 2026. `task` writes the prompt. Documents take
`task = :document`, with a `title` each when they have one; a search takes `task = :search_query`:

```@example ollama
ollama = OllamaEndpoint()
gemma = "embeddinggemma-2:740m-q8_0"

docs = [EmbeddingInput("Refunds reach the original payment method within 5 working days.";
                       title="Refund policy"),
        EmbeddingInput("Orders over 50 EUR ship free within the EU."; title="Shipping"),
        EmbeddingInput("Reset a forgotten password from the sign-in page."; title="Account help")]
D = embedding_vectors(embeddingrequest!(
    Embeddings(docs; service=ollama, model=gemma, task=:document)))

q = embedding_vectors(embeddingrequest!(
    Embeddings("When do I get my money back?"; service=ollama, model=gemma, task=:search_query)))

# The vectors have unit length, so the dot product is the cosine.
[(d.title, round(sum(q .* v); digits=3)) for (d, v) in zip(docs, D)]
```

An EmbeddingGemma request without a `task` is an `ArgumentError` before anything is sent;
`task = :none` sends the text as it is. A `title` goes only with `:document`, and an empty or
whitespace-only title is none. The text the model reads, for each task — the prompts of
EmbeddingGemma 2's [model card](https://ai.google.dev/gemma/docs/embeddinggemma/model_card_2), written
in front of the text:

| `task` | Use it for | The model reads |
| :--- | :--- | :--- |
| `:search_query` | a search | `task: search result \| query: …` |
| `:question_answering` | a question a document answers | `task: question answering \| query: …` |
| `:fact_checking` | a claim to check | `task: fact checking \| query: …` |
| `:code_retrieval` | a request for code | `task: code retrieval \| query: …` |
| `:document` | a document to be found | `title: <its title, else none> \| text: …` |
| `:classification` | a text to classify | `task: classification \| query: …` |
| `:clustering` | a text to group | `task: clustering \| query: …` |
| `:sentence_similarity` | a text to compare | `task: sentence similarity \| query: …` |
| `:none` | your own prompt, or none | `…` |

The same prompts hold for EmbeddingGemma v1 (`embeddinggemma`, its
[model card](https://ai.google.dev/gemma/docs/embeddinggemma/model_card)) and on any other service that
serves EmbeddingGemma; a server that writes the prompts itself takes `task = :none`. A model whose prompts
UniLM does not know takes `task = :none` or no task, and a title is refused for it
([Limits and refusals](@ref ollama_limits)).

**Smaller vectors.** EmbeddingGemma returns 768 numbers, and EmbeddingGemma 2's
[card](https://ai.google.dev/gemma/docs/embeddinggemma/model_card_2) says "The supported dimensions are
768, 512, 256, and 128", as v1's lists "smaller options available (512, 256, or 128)": `dimensions`
asks for one of these, and any other is refused before the request. The shorter vector still has unit length:

```@example ollama
v = embedding_vectors(embeddingrequest!(Embeddings("Orders over 50 EUR ship free.";
        service=ollama, model=gemma, task=:document, dimensions=256)))
length(v), round(sqrt(sum(abs2, v)); digits=3)
```

| Model | Dimensions | Window | Takes |
| :--- | :--- | :--- | :--- |
| `embeddinggemma-2:740m-q8_0`, `embeddinggemma-2:740m-mxfp8` | 768, 512, 256 or 128 | 8,192 tokens ([Long inputs](@ref ollama_embedding_window)) | text, images and sound; a `task` |
| `embeddinggemma-2:270m-q8_0-text` | 768, 512, 256 or 128 | 8,192 tokens | text; a `task` |
| `embeddinggemma` (v1) | 768, 512, 256 or 128 | 2,048 tokens | text; a `task` |

A reply row that is empty, all zeros or holds a value that is no finite number is no embedding: the
call is an `EmbeddingCallError` naming the row, not a vector that would still compare and index. The
concepts every provider shares are on [Embeddings](@ref embeddings_guide).

!!! details "Evidence"
    Measured on 21 October 2026 with Ollama 0.40.6 on an Apple M4 Max, through
    UniLM. **Do the task prompts help?** On SciFact (BEIR: 300 claims searched
    against 5,183 abstracts), claims embedded with `:search_query` and abstracts
    with `:document` and their titles, against the raw texts (`task = :none`),
    768 dimensions; paired bootstrap over the claims, 10,000 resamples, 95%
    intervals. The question was written down before the run: the prompts help
    where the interval lies above 0.

    | Model | nDCG@10 with prompts | without | Difference |
    | :--- | :--- | :--- | :--- |
    | `embeddinggemma-2:740m-mxfp8` | 0.8515 | 0.8245 | +0.027 [+0.009, +0.046] |
    | `embeddinggemma-2:270m-q8_0-text` | 0.8536 | 0.8207 | +0.033 [+0.014, +0.053] |
    | `embeddinggemma` (v1) | 0.7921 | 0.7727 | +0.019 [+0.003, +0.037] |

    One public dataset, one run per condition. On a 24-document, 12-query set of our
    own, the raw texts came out ahead by about one query on every model — for v1
    refuting the comparison written down before that run — a set too small to tell
    the two apart. For text, `740m-q8_0` and `270m-q8_0-text` returned identical
    vectors: the 0.3 GB build embeds text as the 1.3 GB one does.

## [Images and sound](@id ollama_embedding_media)

An [`ImageAttachment`](@ref) or [`AudioAttachment`](@ref) is an item of its own, and
`EmbeddingInput(text, image)` embeds a text with its picture as one item, `EmbeddingInput(text, clip)`
a text with its sound: Ollama's `/api/embed` takes an item "with any of `text` (string), `image`
(base64-encoded bytes), `audio` (base64-encoded bytes)" on models that embed media
([API reference](https://github.com/ollama/ollama/blob/v0.40.11/docs/api.md#generate-embeddings)). An item
holds an image or a sound clip, not both: Ollama passed an item's media to the model in a random
order — ten requests of one item holding a text, a picture and a clip gave
`embeddinggemma-2:740m-q8_0` two different vectors (8 and 2 of the 10, largest difference 0.0183;
Ollama 0.40.6, 21 October 2026) — so such an item is refused before any request: send the image and
the audio as separate items. Texts, images and sounds land in one space, so a word can find a
picture:

```@example ollama
P, C = embedding_vectors(embeddingrequest!(Embeddings(
    [ImageAttachment("../../assets/red_circle.png"), AudioAttachment("../../assets/secret_word.wav")];
    service=ollama, model=gemma, task=:document)))

words = ["a red circle", "a blue square", "banana", "apple"]
W = embedding_vectors(embeddingrequest!(
    Embeddings(words; service=ollama, model=gemma, task=:search_query)))

for (name, item) in ("the picture" => P, "the recording" => C)
    println(rpad(name, 14), join(("$w $(round(sum(item .* v); digits=3))" for (w, v) in zip(words, W)), ", "))
end
```

The picture lies nearest to "a red circle" and the recording — *"The secret word is banana"* — nearest
to "banana". Media take no prompt; the text of a mixed item does, and both of its parts shape its
vector (Evidence below). Ollama's [EmbeddingGemma 2](https://ollama.com/library/embeddinggemma-2) page
describes "a 270M parameter text backbone" with "selectively loadable vision (170M) and audio (300M)
encoders", and its tags carry those sizes: the 270m builds embed text, the 440m ones images too, the
570m ones sound too, and the 740m ones all three. Images and sound were measured on `740m-q8_0` and
`740m-mxfp8`; the 440m and 570m builds are untested. Only Ollama embeds media: every other service
embeds text, and refuses media before the request.

!!! details "Evidence"
    `embeddinggemma-2:740m-q8_0`, Ollama 0.40.6, Apple M4 Max, 21 October 2026. **Images
    and sound**: the red-circle picture scored 0.754 against "a red circle" and 0.633
    against "a blue square"; the recording 0.792 against "banana" and 0.626 against
    "apple". **A text with its media as one item**: "a red circle" with the picture gave a
    vector of its own, cosine 0.870 with the text alone and 0.903 with the picture alone
    (0.752 between the two); "banana" with the recording 0.847 and 0.957 (0.794) — on
    `740m-mxfp8` 0.857 and 0.907, 0.845 and 0.958 — so neither part was dropped.

## [Long inputs](@id ollama_embedding_window)

EmbeddingGemma 2 has an "8K token context window" ([its page](https://ollama.com/library/embeddinggemma-2)),
and UniLM asks Ollama for that window with each of its requests unless the endpoint sets `num_ctx` or
`num_batch` itself ([Options](@ref ollama_options)). An input longer than the window is an
[`EmbeddingFailure`](@ref) with HTTP 400, not a vector of part of it: split long documents into
chunks.

| Model | Window | Measured on Ollama 0.40.6 |
| :--- | :--- | :--- |
| EmbeddingGemma 2: `740m-q8_0`, `740m-mxfp8`, `270m-q8_0-text` | 8,192 tokens | 5,991 tokens embedded whole; 8,331 refused (HTTP 400) |
| EmbeddingGemma (v1) | 2,048 tokens | 3,203 tokens refused (HTTP 400) |

The 440m and 570m builds were not measured. `truncate=true` on the endpoint lets Ollama cut the end of
an input that does not fit instead: the API reference's `truncate` "truncates the end of each input to
fit within context length", and UniLM sends `false` unless the endpoint says otherwise.

Ollama reports a 262,144-token context for the GGUF builds of EmbeddingGemma 2, those its `:llamacpp`
runner serves, while the longest input they take follows `num_batch` (Evidence below). So an
embeddings request — and a [`load_model`](@ref) — for an EmbeddingGemma 2 model carries `num_ctx` and
`num_batch` of 8192 unless the endpoint sets either, and the window is part of a recording's key. An
endpoint that sets `num_ctx` sends it instead of the 8,192-token window, and the GGUF builds then keep
Ollama's default batch, which refused the long inputs measured: embed through an endpoint that sets
neither `num_ctx` nor `num_batch`.

!!! details "Evidence: the window"
    Ollama 0.40.6, Apple M4 Max, 21 October 2026. The MLX build, `740m-mxfp8`,
    loads at 8,192 tokens; at Ollama's default batch the GGUF builds measured
    refused long inputs with HTTP 413 — `740m-q8_0` anything over about 2,037
    tokens — and so they did through an endpoint that set `num_ctx` alone.

## [Local retrieval](@id ollama_retrieval)

Find the passages that answer a question, on your own machine: embed the documents once with
`task = :document`, each question with `task = :search_query`, and rank by the dot product —
EmbeddingGemma's vectors have unit length, so it is the cosine:

```@example ollama
using LinearAlgebra

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

```@example ollama
search("Can I send back shoes I never wore?")
```

The first question finds the refund policy, the second the returns policy. The documents are embedded
once; each search embeds only its question.

Recorded on 21 October 2026 on `embeddinggemma-2:740m-q8_0` with UniLM's development branch after
0.24.0 and Ollama 0.40.6, on an Apple M4 Max with 64 GB.
