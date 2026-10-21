# [Local Models with Ollama](@id ollama_guide)

[Ollama](https://ollama.com) runs open models on your own machine: no API key, no
bill, and nothing leaves the computer. [`OllamaEndpoint`](@ref) speaks Ollama's
native API, so what a local model offers is reachable from Julia — Gemma 4's
thinking, tool calls, structured output, images, audio and transcription;
EmbeddingGemma 2's embeddings of text, images and sound; decisions from local
System One models; a context window of your choosing — and the ways a local
server fails come back as typed results that say what to do.

**Setup:** install Ollama and start it (the desktop app, or `ollama serve`), then
pull a model:

```bash
ollama pull gemma4:e4b                  # text, images, audio, tools (9.5 GB)
ollama pull embeddinggemma-2:740m-q8_0  # embeddings of text, images and audio (1.3 GB)
ollama pull tev1:4b                     # decisions: System One questions (4.4 GB)
```

The examples name the tags they ran on: `embeddinggemma-2` alone pulls its
`latest` tag, a different 740m build, which was not measured here.

`OllamaEndpoint()` finds the server at `OLLAMA_HOST` when that is set (the same
rules as Ollama's own clients, so `gpu-box` means `http://gpu-box:11434`), else at
`http://127.0.0.1:11434`.

## Which path?

| I want to… | Use | You get |
| :--- | :--- | :--- |
| [chat with a local model](@ref ollama_first) | `Chat(service=OllamaEndpoint(), model="gemma4:e4b")` | the reply, its token counts, a cost of 0.0 |
| [see how the model reasoned](@ref ollama_thinking) | `reasoning_effort="low"`, [`reasoning_text`](@ref) | the thinking text, when the model thinks |
| [let the model call Julia functions](@ref ollama_tools) | `tools`, [`tool_loop!`](@ref) | the calls run, then a final answer |
| [get JSON that matches a schema](@ref ollama_json) | `response_format=UniLM.json_schema(…)` | JSON that follows the schema |
| [ask about an image or a recording](@ref ollama_media) | [`ImageAttachment`](@ref), [`AudioAttachment`](@ref) | an answer about what the model saw or heard |
| [transcribe a recording](@ref ollama_transcription) | [`transcribe`](@ref)`(path; service=OllamaEndpoint(), model="gemma4:e4b")` | the words it heard |
| [decide about a text](@ref backends_ollama), or a picture with a [vision model](@ref backends_ollama_images) | [`ask`](@ref)`(…; service=OllamaEndpoint(), model="tev1:4b")` | typed answers with probabilities, as on Jev |
| [embed texts, images and sound for search](@ref ollama_embeddings) | `Embeddings(…; service=OllamaEndpoint(), model="embeddinggemma-2:740m-q8_0", task=:document)` | one vector per item, written for its task |
| [give the model more context, keep it loaded, or pin its runner](@ref ollama_options) | `OllamaEndpoint(num_ctx=…, keep_alive=…, runner=…)` | the model run the way you asked (a runner it has no variant for: the server's 400) |
| [install, inspect, unload or delete models](@ref ollama_models) | [`list_models`](@ref), [`pull_model`](@ref), [`model_info`](@ref), [`delete_model`](@ref), … | typed model cards |
| [know what went wrong](@ref ollama_failures) | the result's status and message | the fix for each failure |

Every executed example on this page ran on Ollama 0.40.6 on an Apple M4 Max with
64 GB, on 21 October 2026, against `gemma4:e4b` (the embeddings ones against
`embeddinggemma-2:740m-q8_0`); the manual replays those recorded answers ([Test
and Develop](@ref jev_testing_recorded)), so its build needs no Ollama server.

## [First chat](@id ollama_first)

```@example ollama
using UniLM

chat = Chat(service=OllamaEndpoint(), model="gemma4:e4b",
            reasoning_effort="none")
push!(chat, Message(Val(:system), "You are a concise assistant."))
push!(chat, Message(Val(:user),
                    "In one sentence: why run a language model locally?"))

result = chatrequest!(chat)
text(result)
```

```@example ollama
result.usage, estimated_cost(result)
```

A local model has no price, so its cost is 0.0 — without the missing-price warning
other unlisted models get — and so is the cost of a `respond` or FIM result from it.
A cloud model reached through the same server, one whose tag is `cloud` or ends in
`-cloud` (`gpt-oss:120b-cloud`), runs on Ollama's machines and bills your Ollama
account: UniLM prices each of its results — chat, embeddings, System One, `respond`
and FIM — by the requested name, like any other model, never as a free local call; a
name missing from the price table gets the missing-price warning. Ollama sends the
cloud the name without its tag and passes the reply through, so a reply may name
`gpt-oss:120b`: the result keeps the requested name, and the server's stays in its
`raw` (read from Ollama 0.40.6's source; no cloud call was made). A cloud model is
recognised by its tag alone, so one copied under a name without it (`ollama cp`) is
priced as local, at 0.0, with no warning. The first request to a model also loads it
into memory: about 1.7 s for Gemma 4 E4B on the machine above, after which a short
answer took about 0.1 s. [`load_model`](@ref) loads it ahead of time.

## [Thinking](@id ollama_thinking)

Gemma 4 can think before it answers. `reasoning_effort="none"` turns thinking off,
`"low"`, `"medium"` or `"high"` turn it on (Gemma 4 has a single thinking mode, so
the three behave alike), and leaving it unset keeps the model's default: on. The
thinking comes back beside the answer, and [`reasoning_text`](@ref) reads it:

```@example ollama
chat = Chat(service=OllamaEndpoint(), model="gemma4:e4b",
            reasoning_effort="low", temperature=0.0)
push!(chat, Message(Val(:system), "You are concise."))
push!(chat, Message(Val(:user),
    "A bat and a ball cost 1.10 in total. The bat costs 1.00 more " *
    "than the ball. How much does the ball cost? " *
    "Answer with the number only."))
result = chatrequest!(chat)
text(result)
```

```@example ollama
print(first(reasoning_text(result), 300), "…")
```

The thinking stays on its turn in `chat`, and goes back to Ollama with the next
request. Ollama's Gemma 4 template shows the model the thinking of the turns since
the last user message (the current tool loop) and leaves older thinking out, as
Gemma 4's own guidance asks; in a tool round trip measured with `gemma4:e2b` on
Ollama 0.34.4, the turn succeeded 10 of 10 times with the thinking sent back and 8
of 10 without it.

!!! note "A thinking model decides when to think"
    With thinking on, Gemma 4 E4B still answers easy prompts without thinking:
    [`reasoning_text`](@ref) is then `nothing`. On the bat-and-ball prompt above
    (Ollama 0.34.4) it thought in 2 to 5 of 6 runs at the default temperature, and
    in every run at `temperature=0.0`. Thinking spends the reply's token budget
    first: a small `max_tokens` can end a reply at finish reason `"length"` with no
    answer text.

## [Tools](@id ollama_tools)

Tools work as with every backend (see [Tool Calling](@ref tools_guide)):

```@example ollama
city = Dict("type" => "object", "required" => ["city"],
            "properties" => Dict("city" => Dict("type" => "string")))
weather = UniLM.CallableTool(
    Tool(func=FunctionSignature(name="get_weather",
                                description="Current weather in a city",
                                parameters=city)),
    (name, args) -> "18°C and sunny in $(args["city"])")

chat = Chat(service=OllamaEndpoint(), model="gemma4:e4b", tools=[weather],
            temperature=0.0)
push!(chat, Message(Val(:system), "You are concise."))
push!(chat, Message(Val(:user), "What's the weather in Lyon? " *
    "Use the tool, then answer in one sentence."))

loop = tool_loop!(chat; tools=[weather])
calls = [(o.tool_name, o.arguments) for o in loop.tool_calls]
loop.completed, calls, text(loop.response)
```

Ollama has no `tool_choice`: the model decides whether to call a tool, so a
`tool_choice` other than `"auto"` is refused, and it may call several tools in one
turn whatever `parallel_tool_calls` says. A tool marked `strict=true` is refused
too, since Ollama does not enforce tool schemas.

## [Structured output](@id ollama_json)

A `response_format` constrains the reply to a JSON schema (or, with
`UniLM.json_object()`, to any JSON object):

```@example ollama
schema = Dict("type" => "object", "additionalProperties" => false,
              "required" => ["name", "year", "city"],
              "properties" => Dict("name" => Dict("type" => "string"),
                                   "year" => Dict("type" => "integer"),
                                   "city" => Dict("type" => "string")))

person = UniLM.json_schema("person", "A person", schema)
chat = Chat(service=OllamaEndpoint(), model="gemma4:e4b",
            response_format=person)
push!(chat, Message(Val(:system), "You extract facts."))
push!(chat, Message(Val(:user),
    "Extract the person: Ada Lovelace, born 1815 in London."))

using JSON
JSON.parse(text(chatrequest!(chat)))
```

With a `response_format`, the request turns thinking off. Ollama applies the format
only after the model's thinking ends, so a thinking model that answers without
thinking would not be constrained at all; for the same reason a `response_format`
together with a thinking effort (`"low"`, `"medium"`, `"high"`) is refused before
the request is sent.

!!! details "Evidence"
    The extraction request above, sent to `/api/chat` 10 times per setting on
    Ollama 0.34.4: `gemma4:e4b` returned valid JSON in 0 of 10 runs with thinking at
    its default (it answered plain text), 1 of 10 with thinking on, and 10 of 10 with
    it off; `gemma4:e2b` in 10, 8 and 10 of 10. Every failure was a run in which the
    model did not think.

## [Images and audio](@id ollama_media)

A user message carries images and sound clips as attachments, read from a file
(or from bytes) and recognised by their content: PNG, JPEG, GIF or WebP images; WAV
or MP3 audio.

![A red disc on a white background](../assets/red_circle.png)

```@example ollama
chat = Chat(service=OllamaEndpoint(), model="gemma4:e4b", temperature=0.0)
push!(chat, Message(Val(:system), "You are concise."))
push!(chat, Message(Val(:user), "What colour is the shape? One word.",
                    ImageAttachment("../assets/red_circle.png")))
text(chatrequest!(chat))
```

The [recording](https://github.com/algunion/UniLM.jl/raw/main/docs/src/assets/secret_word.wav)
says *"The secret word is banana."*:

```@example ollama
chat = Chat(service=OllamaEndpoint(), model="gemma4:e4b", temperature=0.0)
push!(chat, Message(Val(:system), "You are concise."))
push!(chat, Message(Val(:user),
    "What is the secret word in this recording? One word.",
    AudioAttachment("../assets/secret_word.wav")))
text(chatrequest!(chat))
```

Gemma 4 E2B, E4B and 12B hear audio (up to 30 s); the larger Gemma 4 models see
images but do not hear. A model without the capability answers HTTP 400 — after it
has been loaded. [`model_info`](@ref) lists what a model can do. Images are sent as
bytes: Ollama does not fetch image URLs.

## [Transcription](@id ollama_transcription)

A model that hears audio also transcribes it:

```julia
r = transcribe("secret_word.wav"; service=OllamaEndpoint(), model="gemma4:e4b")
transcript_text(r)    # "The secret word is banana" (Ollama 0.40.6, the recording above)
```

The call goes to Ollama's OpenAI-compatible `/v1/audio/transcriptions`, and there is
no default model: name one that hears. Ollama transcribes at temperature 0 and reads
one `language` and a `prompt`; the transcript is asked for as JSON, the only form in
which Ollama reports a failure that arrives after its reply has begun, as an
[`AudioCallError`](@ref). `response_format = "text"` is refused before the request,
since such a failure would read as an empty transcript (by Ollama 0.40.6's source,
not reproduced here): leave `response_format` unset and read the text with
[`transcript_text`](@ref). `temperature`, `languages`, `keywords` and any other format
are refused before the request too. Transcription calls are not recorded by
[`with_recorded_answers`](@ref), so this example does not run when the manual is
built.

## [Embeddings with EmbeddingGemma 2](@id ollama_embeddings)

EmbeddingGemma reads a short prompt that says what a text is for, and Ollama does
not add it: `task` writes it. Documents take `task = :document`, with a `title`
each when they have one; a search takes `task = :search_query`:

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

An EmbeddingGemma request without a `task` is an `ArgumentError` before anything is
sent; `task = :none` sends the text as it is. A `title` goes only with `:document`,
and an empty or whitespace-only title is none. The text the model reads, for each
task — the model card's prompts, written in front of the text:

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

The same prompts hold for EmbeddingGemma v1 (`embeddinggemma`) and on any other
service that serves EmbeddingGemma; a server that writes the prompts itself takes
`task = :none`. A model whose prompts UniLM does not know takes `task = :none` or
no task, and a title is refused for it.

**Smaller vectors.** EmbeddingGemma returns 768 numbers; `dimensions` asks for one
of its trained shorter (Matryoshka) sizes, 512, 256 or 128 (any other is refused),
still of unit length:

```@example ollama
v = embedding_vectors(embeddingrequest!(Embeddings("Orders over 50 EUR ship free.";
        service=ollama, model=gemma, task=:document, dimensions=256)))
length(v), round(sqrt(sum(abs2, v)); digits=3)
```

**Images and sound.** An [`ImageAttachment`](@ref) or [`AudioAttachment`](@ref) is
an item of its own, and `EmbeddingInput(text, image)` embeds a text with its
picture as one item, `EmbeddingInput(text, clip)` a text with its sound. An item
holds an image or a sound clip, not both: Ollama passes an item's media to the
model in a random order — ten requests of one item holding a text, a picture and a
clip gave `embeddinggemma-2:740m-q8_0` two different vectors (8 and 2 of the 10,
largest difference 0.0183; Ollama 0.40.6, the machine above, 21 October 2026) — so
such an item is refused before any request: send the image and the audio as
separate items. Texts, images and sounds land in one space, so a word can find a
picture:

```@example ollama
P, C = embedding_vectors(embeddingrequest!(Embeddings(
    [ImageAttachment("../assets/red_circle.png"), AudioAttachment("../assets/secret_word.wav")];
    service=ollama, model=gemma, task=:document)))

words = ["a red circle", "a blue square", "banana", "apple"]
W = embedding_vectors(embeddingrequest!(
    Embeddings(words; service=ollama, model=gemma, task=:search_query)))

for (name, item) in ("the picture" => P, "the recording" => C)
    println(rpad(name, 14), join(("$w $(round(sum(item .* v); digits=3))" for (w, v) in zip(words, W)), ", "))
end
```

The picture lies nearest to "a red circle" and the recording — *"The secret word is
banana"* — nearest to "banana". Media take no prompt; the text of a mixed item
does, and both of its parts shape its vector (Evidence below). The tags say what
each EmbeddingGemma 2 build takes: `270m-…-text` text, `440m-…-text-vision` images
too and `570m-…-text-audio` sound too — these two untested here — and the 740m
builds all three: images and sound were measured on `740m-q8_0` and `740m-mxfp8`.
Images and sound go only to Ollama: every other service embeds text, and refuses
media before the request.

**Long inputs.** EmbeddingGemma 2 was trained on 8,192 tokens, and UniLM asks Ollama
for that window with each of its requests unless the endpoint sets `num_ctx` or
`num_batch` itself ([Runtime options](@ref ollama_options)). An input longer than
the window is an [`EmbeddingFailure`](@ref) with HTTP 400, not a vector of part of
it: split long documents into chunks.

| Model | Window | Measured on Ollama 0.40.6 |
| :--- | :--- | :--- |
| EmbeddingGemma 2: `740m-q8_0`, `740m-mxfp8`, `270m-q8_0-text` | 8,192 tokens | 5,991 tokens embedded whole; 8,331 refused (HTTP 400) |
| EmbeddingGemma (v1) | 2,048 tokens | 3,203 tokens refused (HTTP 400) |

The 440m and 570m builds were not measured.

!!! details "How the window is sent"
    Ollama reports a 262,144-token context for the llama.cpp builds of EmbeddingGemma
    2 (the MLX build, `740m-mxfp8`, loads at 8,192), and at its default batch the
    llama.cpp builds measured refused long inputs with HTTP 413 — `740m-q8_0`
    anything over about 2,037 tokens: their limit follows `num_batch`.
    So an embeddings request — and a [`load_model`](@ref) — for an EmbeddingGemma 2
    model carries `num_ctx` and `num_batch` of 8192 unless the endpoint sets either;
    the window is part of a recording's key. `truncate=true` lets Ollama cut the end
    of an input that does not fit instead.

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
    the two apart. **Images and sound**, `740m-q8_0`: the red-circle picture scored
    0.754 against "a red circle" and 0.633 against "a blue square"; the recording
    0.792 against "banana" and 0.626 against "apple". **A text with its media as one
    item**: "a red circle" with the picture gave a vector of its own, cosine 0.870
    with the text alone and 0.903 with the picture alone (0.752 between the two);
    "banana" with the recording 0.847 and 0.957 (0.794) — on `740m-mxfp8` 0.857 and
    0.907, 0.845 and 0.958 — so neither part was dropped. For text, `740m-q8_0` and
    `270m-q8_0-text` returned identical vectors: the 0.3 GB build embeds text as the
    1.3 GB one does.

## [Runtime options](@id ollama_options)

What would be a command-line flag or a Modelfile line in Ollama is an argument of
[`OllamaEndpoint`](@ref), sent with every chat and embeddings request (System One
requests take none: [Models](@ref ollama_models)):

```julia
ollama = OllamaEndpoint(num_ctx=32_768,  # context window, in tokens
                        keep_alive=3600, # seconds loaded after a call
                        top_k=64, min_p=0.05)
chat = Chat(service=ollama, model="gemma4:e4b")
```

This endpoint sets `num_ctx`, so its embeddings requests carry that instead of the
8,192-token window UniLM asks for EmbeddingGemma 2
([Long inputs](@ref ollama_embeddings)), and its llama.cpp builds keep Ollama's
default batch: measured with `num_ctx` alone, they still refused long inputs with
HTTP 413 (`740m-q8_0`: over about 2,037 tokens at the defaults; Ollama 0.40.6, the
machine above, 21 October 2026). Embed through an endpoint that sets neither
`num_ctx` nor `num_batch`.

- **Context window.** Ollama picks a default from the GPU memory it finds — 4096
  tokens under 23 GiB, 32768 from 23 GiB, 262144 from 47 GiB — capped at what the
  model was trained for (128K for Gemma 4 E2B and E4B). `num_ctx` sets it when the
  model loads (a model with vision gets at least 2,048 tokens); what a later request
  with another `num_ctx` does depends on the runner ([below](@ref ollama_runners)).
- **Input that does not fit.** By default (`truncate=false`), a prompt longer than
  the window is an [`LLMFailure`](@ref) with HTTP 400 — `"request (… tokens) exceeds
  the available context size (… tokens), try increasing it"` — on the llama.cpp
  runner, instead of an answer to a prompt Ollama shortened. `truncate=true` restores
  Ollama's own behaviour, which can cut inside a single over-long message, keeping its
  first tokens and its tail: the start of a long document is lost without a word, not
  only the oldest messages. `shift=false` also stops a reply that fills the window
  (finish reason `"length"`) instead of letting it continue on shortened context.
- **Sampling.** [`OllamaOptions`](@ref) lists the knobs `Chat` has no field for
  (`top_k`, `min_p`, `repeat_penalty`, …); `temperature`, `top_p`, `seed`, `stop`
  and `max_tokens` stay on the `Chat`. Unset options keep the model's own defaults
  — Gemma 4 ships temperature 1.0, top-k 64, top-p 0.95.

### [Runners](@id ollama_runners)

Ollama runs a model on one of its runners: MLX on Apple Silicon, for models stored
as safetensors, or llama.cpp or ggml, for GGUF models. [`running_models`](@ref)
reports the runner serving each loaded model, and [`list_models`](@ref) and
[`model_info`](@ref) the one the model's listing names, which can differ: a GGUF
model listed as `ggml` is served by llama.cpp. Unset, the server chooses the runner.
`OllamaEndpoint(runner=:mlx)` — or `:llamacpp`, `:ggml` — pins one, and Ollama
matches it against the runner the model's manifest names, the one `list_models`
reports as `OllamaModel.runner`:

- A model published for one runner — every model measured — answers as if unpinned
  when that runner is pinned. Any other runner is the server's error, never a
  fallback: HTTP 400 `no compatible manifest found for runners: <runner>` (404 from
  [`model_info`](@ref), and so from [`load_model`](@ref)), even while the model is
  loaded.
- A model published for several runners is served — and pulled by
  [`pull_model`](@ref) — as its variant for the one pinned (by Ollama's source;
  untested).
- Pin the runner the listing names, not the one `running_models` reports: a GGUF
  model listed as `:ggml` (`qwen2.5-coder:1.5b-base`, `embeddinggemma:latest`)
  answers pinned to `:ggml`, is served by llama.cpp, and is the 400 pinned to
  `:llamacpp`.
- The pin goes with every request of the endpoint — chat, embeddings, System One,
  [`model_info`](@ref), [`load_model`](@ref), [`unload_model`](@ref) and
  [`pull_model`](@ref); [`delete_model`](@ref), the listings and the
  OpenAI-compatible routes ([`respond`](@ref), [`compact_response`](@ref), FIM,
  transcription) take none — so a pinned endpoint fails for each model without a
  variant for its runner. Pin it on an endpoint you use for one model, or leave it
  unset.

The runner decides what the window does:

| | MLX | llama.cpp |
| :--- | :--- | :--- |
| The model loads with | `num_ctx` | `num_ctx` |
| A later request for a larger `num_ctx` | keeps the loaded window | loads the model again with it |
| A chat prompt longer than the window | is answered | is an HTTP 400, unless `truncate=true` |
| With `truncate=true` | not measured | a 5,407-token message cut to 2,050 tokens: the first 4 and the tail kept |

To give a loaded MLX model another window, [`unload_model`](@ref) it first. A model
with vision loads with at least 2,048 tokens whatever `num_ctx` asks: by Ollama
0.40.6's source its scheduler raises a smaller one ("multimodal models require at
least 2048 context"), and `embeddinggemma-2:740m-q8_0` asked for 1,024 loaded with
2,048 on llama.cpp. The other [`OllamaOptions`](@ref) are passed through unchanged;
their effect on MLX is not verified.

!!! details "Evidence"
    Measured with Ollama 0.40.6 on an Apple M4 Max: `gemma4:e4b` on MLX, loaded
    with `num_ctx=4096`, kept 4,096 after a request for 8,192 and answered a
    10,842-token prompt, reading a code word from its first sentence;
    `qwen2.5-coder:1.5b-base` on llama.cpp (listed as `ggml`) loaded again for 8,192
    and refused a 5,407-token prompt on 4,096 with HTTP 400. No model ran on the
    ggml runner. Runner pins, 21 October 2026: `:llamacpp` on `kev:0.8b-q8_0`
    answered and [`running_models`](@ref) reported `:llamacpp`; `:mlx` and `:ggml` on
    it, and `:llamacpp` on `tev1:0.8b`, were HTTP 400 `no compatible manifest found
    for runners: <runner>`, also while the model was loaded, and so was every runner
    but the listed one on each of the nine decision models of
    [Which model](@ref backends_ollama_models). The 2,048-token floor of a model with
    vision was measured on 22 October 2026 (and on Ollama 0.40.7 on Linux).

## [Models](@id ollama_models)

The model-management verbs return typed results like every other call:

```julia
ollama = OllamaEndpoint()

models = list_models(service=ollama).response  # Vector{OllamaModel}: size, runner, format, capabilities
info = model_info("gemma4:e4b"; service=ollama).response
info.capabilities     # [:completion, :vision, :audio, :tools, :thinking]
info.context_length   # 131072
info.thinking_levels  # [false, true]

percent(p) = isnothing(p.total) ? "" :
             string(round(Int, 100p.completed / p.total), "%")
pull_model("all-minilm:22m"; service=ollama,
           progress = p -> print("\r", p.status, " ", percent(p)))

load_model("gemma4:e4b"; service=ollama)  # with the endpoint's settings
running_models(service=ollama).response    # context, memory, runner, expiry
unload_model("gemma4:e4b"; service=ollama)
delete_model("all-minilm:22m"; service=ollama)  # off the disk; pull_model brings it back
```

[`load_model`](@ref) loads a model that completes text with an empty chat request,
and any other with an empty embeddings request, so the first real request finds it
in memory. It sends what the model's own requests send: a chat model's load the
endpoint's options and `shift`, an embedding model's those options (EmbeddingGemma 2
with its 8,192-token window), and a decision model's no options at all, as System
One requests carry none. So an endpoint's `num_ctx`, or any other option, does not
apply to a decision model, neither when it is loaded nor when it is asked: the
server's default context applies (73,728 tokens for `kev:0.8b-q8_0` on the machine
above). Through `OllamaEndpoint(num_ctx=4096)`, the first ask after
`load_model("kev:0.8b-q8_0")` took 0.036–0.049 s in five cycles, and an ask
without it, which loads the model, 0.81–1.36 s. Witnessed on Ollama 0.40.6 on the
machine above (21 October 2026) for a chat model (`gemma4:e4b`), embedding models
(`embeddinggemma` and EmbeddingGemma 2) and one decision model (`kev:0.8b-q8_0`);
other decision models and other kinds of model are untested.

!!! details "Choosing a small Gemma 4"
    Measured on Ollama 0.34.4 (Apple M4 Max, 64 GB; 10 runs per text cell, 6 for the
    prime question, 5 per media cell). Both E2B (7.2 GB) and E4B (9.6 GB) called tools correctly in 10 of 10
    runs and named the colour in 5 of 5; E4B named the spoken word in 5 of 5, E2B in 4
    of 5. With thinking off, E2B misjudged "Is 391 prime?" in 6 of 6 runs and E4B
    answered correctly in 6 of 6. E2B decodes faster (86 against 54 tokens per
    second) but thought in 68 of 80 runs where E4B thought in 15, so E4B answered
    sooner on short prompts. This page and UniLM's own tests use E4B.

## Streaming, `respond` and FIM

Streaming works as on every backend (see [Streaming](@ref streaming_guide)):

```julia
chat = Chat(service=OllamaEndpoint(), model="gemma4:e4b", stream=true)
push!(chat, Message(Val(:system), "You are concise."))
push!(chat, Message(Val(:user), "Write a haiku about Julia."))
result = fetch(chatrequest!(chat;
    callback=(chunk, _) -> chunk isa String && print(chunk)))
```

[`respond`](@ref) reaches Ollama's OpenAI-compatible Responses API, which keeps no
state between calls: `previous_response_id`, `store`, `conversation` and the other
fields it would ignore are refused, and a [`Respond`](@ref) tool loop is refused in
favour of `tool_loop!` on a `Chat`. So are [`get_response`](@ref),
[`cancel_response`](@ref), [`delete_response`](@ref) and [`list_input_items`](@ref),
before any request: Ollama stores no response. The structured-output rule holds
there too: a `json_schema` text format turns thinking off, and only function tools
are accepted. [`compact_response`](@ref) reaches Ollama's `/v1/responses/compact`
and needs a `model`: Ollama has no default model, so a call that names none is an
`ArgumentError` before any request, as a `Chat`'s is. [`fim_complete`](@ref) works
with models that fill in the middle, such as `qwen2.5-coder` (Gemma 4 does not).

## [When something fails](@id ollama_failures)

| What you see | Why | What to do |
| :--- | :--- | :--- |
| `LLMCallError` "no Ollama server answered at http://127.0.0.1:11434" | the server is not running, or runs elsewhere | start it (`ollama serve` or the app), or set `OLLAMA_HOST` / `base_url` |
| `LLMFailure` 404 "model 'gemma4:e2b' not found" | the model is not installed | [`pull_model`](@ref) or `ollama pull` |
| `LLMFailure` 400 "exceeds the available context size" | the prompt does not fit the window (llama.cpp) | raise `num_ctx`, or shorten the conversation |
| `LLMFailure` 400 "does not support tools" / "does not support thinking" | the model lacks the capability | check [`model_info`](@ref); pick a model that has it |
| `LLMFailure` 400 after a pause, for an image or a sound | the model sees or hears nothing | use a model with `:vision` / `:audio` |
| `ArgumentError` "expects a task prompt" | an EmbeddingGemma request without `task` | pass the [task](@ref ollama_embeddings) the text is for |
| `EmbeddingFailure` 400 "exceeds the context length" | the input is longer than the model's window | split it into chunks |
| `SystemOneFailure` 400 from a decision model | the model takes less than the route: a question without `instructions`, fewer or more options than it takes, a picture it cannot see | give every question `instructions`; see [what each model takes](@ref backends_ollama_limits) |
| a 400 (404 from [`model_info`](@ref) and [`load_model`](@ref)) "no compatible manifest found for runners: …" | the endpoint pins a runner the model has no variant for | pin the runner [`list_models`](@ref) names for the model, or leave `runner` unset ([Runners](@ref ollama_runners)) |
| `ArgumentError` before any request | an option Ollama would ignore | the message names it |
| an empty answer with finish reason `"length"` | thinking used the whole `max_tokens` budget | raise `max_tokens`, or `reasoning_effort="none"` |
| a slow first call | the model is loading | [`load_model`](@ref) first, or a longer `keep_alive` |
| a streamed call that stays silent, then fails with a timeout | Ollama sends nothing while it loads the model, while the call waits its turn, or while the model writes a tool call | raise `stream_idle_timeout` |

Ollama runs one request at a time per model unless `OLLAMA_NUM_PARALLEL` says
otherwise: concurrent calls queue on the server. A streamed call receives nothing
until its turn comes — and nothing while the model writes a tool call — so with many
long streams in flight, or a slow machine, raise `stream_idle_timeout` ([Timeouts &
Retries](@ref timeouts_guide)); it bounds the wait for the first byte too. An unreachable
server is retried like any transport failure; `RequestConfig(max_attempts=1)`
reports it at once.
