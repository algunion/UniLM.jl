# [Local Models with Ollama](@id ollama_guide)

[Ollama](https://ollama.com) runs open models on your own machine: no API key, no
bill, and nothing leaves the computer. [`OllamaEndpoint`](@ref) speaks Ollama's
native API, so what a local model offers is reachable from Julia — Gemma 4's
thinking, tool calls, structured output, images and audio, embeddings, a context
window of your choosing — and the ways a local server fails come back as typed
results that say what to do.

**Setup:** install Ollama and start it (the desktop app, or `ollama serve`), then
pull a model:

```bash
ollama pull gemma4:e4b      # Gemma 4 E4B: text, images, audio, tools (9.6 GB)
ollama pull embeddinggemma  # embeddings (0.6 GB)
```

`OllamaEndpoint()` finds the server at `OLLAMA_HOST` when that is set (the same
rules as Ollama's own clients, so `gpu-box` means `http://gpu-box:11434`), else at
`http://127.0.0.1:11434`.

## Which path?

| I want to… | Use | You get | Section |
| :--- | :--- | :--- | :--- |
| chat with a local model | `Chat(service=OllamaEndpoint(), model="gemma4:e4b")` | the reply, its token counts, a cost of 0.0 | [First chat](@ref ollama_first) |
| see how the model reasoned | `reasoning_effort="low"` and [`reasoning_text`](@ref) | the thinking text, when the model thinks | [Thinking](@ref ollama_thinking) |
| let the model call Julia functions | `tools` and [`tool_loop!`](@ref) | the calls run and a final answer | [Tools](@ref ollama_tools) |
| get JSON that matches a schema | `response_format=UniLM.json_schema(…)` | JSON that follows the schema | [Structured output](@ref ollama_json) |
| ask about an image or a recording | [`ImageAttachment`](@ref), [`AudioAttachment`](@ref) | an answer about what the model saw or heard | [Images and audio](@ref ollama_media) |
| embed texts for search | `Embeddings(…; service=OllamaEndpoint(), model="embeddinggemma")` | one vector per text | [Embeddings](@ref ollama_embeddings) |
| give the model more context, or keep it loaded | `OllamaEndpoint(num_ctx=…, keep_alive=…)` | the model run the way you asked | [Runtime options](@ref ollama_options) |
| install, inspect or unload models from Julia | [`list_models`](@ref), [`pull_model`](@ref), [`model_info`](@ref), … | typed model cards | [Models](@ref ollama_models) |
| know what went wrong | the result's status and text | a remedy in the message | [When something fails](@ref ollama_failures) |

Every example on this page ran against `gemma4:e4b` on Ollama 0.34.4, on an Apple
M4 Max with 64 GB; the manual replays those recorded answers ([Test and
Develop](@ref jev_testing_recorded)), so its build needs no Ollama server.

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
other unlisted models get. The first request to a model also loads it into memory:
about 2 s for Gemma 4 E4B on the machine above, then about 0.05 s to the first token.
[`load_model`](@ref) loads it ahead of time.

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
    "A bat and a ball cost 1.10 in total. The bat costs 1.00 more than " *
    "the ball. How much does the ball cost? Answer with the number only."))
result = chatrequest!(chat)
text(result)
```

```@example ollama
print(first(reasoning_text(result), 300), "…")
```

The thinking stays on its turn in `chat`, and goes back to Ollama with the next
request: during a tool loop Gemma 4 reads its earlier thinking, and it drops the
thinking of finished turns itself.

!!! note "A thinking model decides when to think"
    With thinking on, Gemma 4 E4B still answers easy prompts without thinking:
    [`reasoning_text`](@ref) is then `nothing`. On the bat-and-ball prompt above it
    thought in 2 to 5 of 6 runs at the default temperature, and in every run at
    `temperature=0.0`. Thinking spends the reply's token budget first: a small
    `max_tokens` can end a reply at finish reason `"length"` with no answer text.

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

chat = Chat(service=OllamaEndpoint(), model="gemma4:e4b",
            response_format=UniLM.json_schema("person", "A person", schema))
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
    The extraction request above, 10 runs each on Ollama 0.34.4: with Ollama's
    default (thinking on), `gemma4:e4b` answered plain text in 10 of 10 runs and
    `gemma4:e2b` in 3 of 10; with thinking off, both returned valid JSON in 10 of 10.
    With a thinking effort set, `gemma4:e4b` returned valid JSON in 5 of 10.

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

## [Embeddings](@id ollama_embeddings)

```@example ollama
texts = ["The cat purrs on the sofa.", "A kitten is sleeping.",
         "Stock markets fell sharply."]
emb = Embeddings(texts; service=OllamaEndpoint(), model="embeddinggemma")
v = embedding_vectors(embeddingrequest!(emb))
cosine(a, b) = sum(a .* b) / sqrt(sum(abs2, a) * sum(abs2, b))
near, far = cosine(v[1], v[2]), cosine(v[1], v[3])
length(v[1]), round(near; digits=2), round(far; digits=2)
```

An input longer than the model's context window (2048 tokens for EmbeddingGemma) is
an [`EmbeddingFailure`](@ref) with HTTP 400, not a vector of its first 2048 tokens:
split long documents into chunks.

## [Runtime options](@id ollama_options)

What would be a command-line flag or a Modelfile line in Ollama is an argument of
[`OllamaEndpoint`](@ref), sent with every chat and embeddings request:

```julia
ollama = OllamaEndpoint(num_ctx=32_768,  # context window, in tokens
                        keep_alive=3600, # seconds loaded after a call
                        top_k=64, min_p=0.05)
chat = Chat(service=ollama, model="gemma4:e4b")
```

- **Context window.** Ollama picks a default from the GPU memory it finds — 4096
  tokens under 23 GiB, 32768 from 23 GiB, 262144 from 47 GiB — capped at what the
  model was trained for (128K for Gemma 4 E2B and E4B). `num_ctx` sets it; Ollama
  reloads the model when it changes, so keep one value per model.
- **Input that does not fit.** By default a prompt longer than the window is an
  [`LLMFailure`](@ref) with HTTP 400 — `"request (… tokens) exceeds the available
  context size (… tokens), try increasing it"` — instead of an answer to a prompt
  Ollama shortened by dropping the oldest messages. `truncate=true` restores
  Ollama's own behaviour. `shift=false` also stops a reply that fills the window
  (finish reason `"length"`) instead of letting it continue on shortened context.
- **Sampling.** [`OllamaOptions`](@ref) lists the knobs `Chat` has no field for
  (`top_k`, `min_p`, `repeat_penalty`, …); `temperature`, `top_p`, `seed`, `stop`
  and `max_tokens` stay on the `Chat`. Unset options keep the model's own defaults
  — Gemma 4 ships temperature 1.0, top-k 64, top-p 0.95.

## [Models](@id ollama_models)

The model-management verbs return typed results like every other call:

```julia
ollama = OllamaEndpoint()

models = list_models(service=ollama).response  # Vector{OllamaModel}
info = model_info("gemma4:e4b"; service=ollama).response
info.capabilities     # [:completion, :vision, :audio, :tools, :thinking]
info.context_length   # 131072
info.thinking_levels  # [false, true]

percent(p) = isnothing(p.total) ? "" :
             string(round(Int, 100p.completed / p.total), "%")
pull_model("gemma4:e2b"; service=ollama,
           progress = p -> print("\r", p.status, " ", percent(p)))

load_model("gemma4:e4b"; service=ollama)  # with the endpoint's settings
running_models(service=ollama).response    # context, memory, expiry
unload_model("gemma4:e4b"; service=ollama)
```

!!! details "Choosing a small Gemma 4"
    Measured on Ollama 0.34.4 (Apple M4 Max, 64 GB; 10 runs per text cell, 5 per
    media cell). Both E2B (7.2 GB) and E4B (9.6 GB) called tools correctly in 10 of 10
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
favour of `tool_loop!` on a `Chat`. [`fim_complete`](@ref) works with models that
fill in the middle, such as `qwen2.5-coder` (Gemma 4 does not).

## [When something fails](@id ollama_failures)

| What you see | Why | What to do |
| :--- | :--- | :--- |
| `LLMCallError` "no Ollama server answered at http://127.0.0.1:11434" | the server is not running, or runs elsewhere | start it (`ollama serve` or the app), or set `OLLAMA_HOST` / `base_url` |
| `LLMFailure` 404 "model 'gemma4:e2b' not found" | the model is not installed | [`pull_model`](@ref) or `ollama pull` |
| `LLMFailure` 400 "exceeds the available context size" | the prompt does not fit the window | raise `num_ctx`, or shorten the conversation |
| `LLMFailure` 400 "does not support tools" / "does not support thinking" | the model lacks the capability | check [`model_info`](@ref); pick a model that has it |
| `LLMFailure` 400 after a pause, for an image or a sound | the model sees or hears nothing | use a model with `:vision` / `:audio` |
| `ArgumentError` before any request | an option Ollama would ignore | the message names it |
| an empty answer with finish reason `"length"` | thinking used the whole `max_tokens` budget | raise `max_tokens`, or `reasoning_effort="none"` |
| a slow first call | the model is loading | [`load_model`](@ref) first, or a longer `keep_alive` |

Ollama runs one request at a time per model unless `OLLAMA_NUM_PARALLEL` says
otherwise: concurrent calls queue on the server. A streamed call waits for its turn
before its first byte, so with many long streams in flight raise
`stream_idle_timeout` ([Timeouts & Retries](@ref timeouts_guide)). An unreachable
server is retried like any transport failure; `RequestConfig(max_attempts=1)`
reports it at once.
