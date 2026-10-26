# [Runtime & model management](@id ollama_runtime_page)

```@setup ollama
using UniLM
```

## [Options](@id ollama_options)

What would be a command-line flag or a Modelfile line in Ollama is an argument of
[`OllamaEndpoint`](@ref): the endpoint's requests carry it, and the constructor checks it.

| Argument | Sent as | Checked when the endpoint is built |
| :--- | :--- | :--- |
| `base_url` | the server every request goes to; unset, the server `OLLAMA_HOST` names, else `http://127.0.0.1:11434` ([Setup](@ref ollama_setup)) | starts with `http://` or `https://` |
| `keep_alive` | `keep_alive`, the seconds a model stays loaded after a request (`Inf` as `-1`, stay loaded), on chat, embeddings and decision requests and on [`load_model`](@ref); unset, the server's default | a number of seconds from 0 to 1e9, or `Inf` |
| `truncate` | `truncate` on every chat and embeddings request, `false` unless set | a `Bool` |
| `shift` | `shift` on chat requests and a chat model's load: `false` ends a reply that fills the context window at finish reason `"length"`; unset, the server's default | a `Bool`, or `nothing` |
| `runner` | `runner` on chat, embeddings and decision requests, [`model_info`](@ref), [`load_model`](@ref), [`unload_model`](@ref) and [`pull_model`](@ref) ([Runners](@ref ollama_runners)) | `:mlx`, `:llamacpp`, `:ggml` or `nothing` |
| any field of [`OllamaOptions`](@ref): `num_ctx`, `num_batch`, `num_gpu`, `main_gpu`, `num_thread`, `use_mmap`, `num_keep`, `top_k`, `min_p`, `repeat_last_n`, `repeat_penalty`, `draft_num_predict`, `typical_p` | the request's `options`, with the `Chat`'s sampling fields, on chat and embeddings requests and on a chat or embedding model's load | each in its range: `OllamaOptions` lists them |

Any other keyword is refused when the endpoint is built, its `ArgumentError` naming the keywords the
endpoint takes, and so is a value that breaks a rule above; a value of the wrong type is Julia's own
error instead, such as a `TypeError` for `truncate=1` or `runner="mlx"`; a request made through the
endpoint meets no further check of these arguments. [`UniLM.encode_request`](@ref) returns the body `chatrequest!`
sends, with no request:

```@example ollama
ollama = OllamaEndpoint(num_ctx=32_768,  # context window, in tokens
                        keep_alive=3600, # seconds loaded after a call
                        top_k=64, min_p=0.05)
chat = Chat(service=ollama, model="gemma4:e4b", temperature=0.2, max_tokens=200)
push!(chat, Message(Val(:system), "You are concise."))
push!(chat, Message(Val(:user), "Name a prime number."))
print(UniLM.encode_request(chat.service, chat))
```

The `Chat`'s `temperature`, `top_p`, `seed`, `stop`, `presence_penalty`, `frequency_penalty` and
`max_completion_tokens`, else `max_tokens` (as `num_predict`), join the endpoint's options, which hold
none of those names, so no option is set in two places. Unset options keep the model's own defaults:
Gemma 4's page gives `temperature=1.0`, `top_p=0.95` and `top_k=64`
([Gemma 4](https://ollama.com/library/gemma4)). Of the options Ollama's
[API reference](https://github.com/ollama/ollama/blob/v0.40.11/docs/api.md#generate-request-with-options)
lists, `OllamaOptions` has no `penalize_newline` or `numa`, and the endpoint refuses them as unknown
keywords: Ollama 0.40.11 does not read them, and its log says "invalid option provided" and the request
goes on without them (observed on 25 October 2026). `draft_num_predict` caps the speculative draft tokens
per step of a model that has a draft model: "Set to 0 to disable speculative drafting"
([Modelfile](https://docs.ollama.com/modelfile)). `typical_p` is one the reference marks: "`typical_p` is
deprecated and may be removed in a future release"; Ollama 0.40.11 still reads it per request, and its
log says "deprecated option provided" (observed on 25 October 2026). `respond`, `compact_response`, FIM and transcription use Ollama's
OpenAI-compatible routes, which carry none of `keep_alive`, the options or `runner`. Decision requests
(`ask`) carry the endpoint's `keep_alive` and `runner` and none of its options
([Models](@ref ollama_models)).

- **Context window.** Ollama "defaults to the following context lengths based on VRAM": 4k under
  24 GiB, 32k from 24 to 48 GiB, 256k from 48 GiB ([context length](https://docs.ollama.com/context-length)),
  and on the machine the examples ran on, `gemma4:e4b` loaded with its whole 131,072-token context
  (Ollama 0.40.11; [Models](@ref ollama_models)). `num_ctx` sets the window when the model loads; a model with
  vision gets at least 2,048 tokens, and what a later request with another `num_ctx` does depends on the
  runner ([Runners](@ref ollama_runners)).
- **Input that does not fit.** With `truncate=false`, the default, a prompt longer than the window is an
  [`LLMFailure`](@ref) with HTTP 400 — `"request (… tokens) exceeds the available context size (…
  tokens), try increasing it"` — on the `:llamacpp` runner (Ollama 0.40.6), instead of an answer to a
  prompt Ollama shortened. `truncate=true` restores Ollama's own behaviour, which can cut inside a single over-long
  message, keeping its first tokens and its tail: a 5,407-token message was cut to 2,050 tokens, the
  first 4 kept (Ollama 0.40.6), so the start of a long document is lost without a word, not only the
  oldest messages. An embeddings input is cut at its end instead
  ([Long inputs](@ref ollama_embedding_window)).
- **A reply that fills the window.** `shift=false` stops it at finish reason `"length"` instead of
  letting it continue on shortened context.
- **Keep-alive.** "By default models are kept in memory for 5 minutes before being unloaded", and
  `OLLAMA_KEEP_ALIVE` on the server changes that default, which a request's `keep_alive` overrides
  ([FAQ](https://docs.ollama.com/faq)). `keep_alive=0` unloads the model after each request.

## [Runners](@id ollama_runners)

Ollama runs a model on one of its runners, which UniLM names `:mlx` (Apple Silicon, for models stored
as safetensors), `:llamacpp` and `:ggml` (for GGUF models). [`running_models`](@ref) reports the runner
serving each loaded model, and [`list_models`](@ref) the one the model's listing names, the runner to
pin, which can differ: a GGUF model listed as `:ggml` is served by `:llamacpp`. [`model_info`](@ref)
reports `:llamacpp` or `:mlx`, and `nothing` for such a legacy GGUF model (Ollama 0.40.6). Unset, the
server chooses the runner. `OllamaEndpoint(runner=:mlx)` — or `:llamacpp`, `:ggml` — pins one, and
Ollama matches it against the runner the model's manifest names, the one `list_models` reports as
`OllamaModel.runner`:

- A model published for one runner — every model measured — answers as if unpinned when that runner
  is pinned. Any other runner is the server's error, never a fallback: HTTP 400 `no compatible manifest
  found for runners: <runner>` (404 from [`model_info`](@ref), and so from [`load_model`](@ref)), even
  while the model is loaded.
- A model published for several runners is served — and pulled by [`pull_model`](@ref) — as its
  variant for the one pinned (by Ollama's source; untested).
- Pin the runner the listing names, not the one `running_models` reports: a GGUF model listed as
  `:ggml` (`qwen2.5-coder:1.5b-base`, `embeddinggemma:latest`) answers pinned to `:ggml`, is served by
  `:llamacpp`, and is the 400 pinned to `:llamacpp`.
- The pin goes with every request of the endpoint that the options table lists; [`delete_model`](@ref),
  the listings and the OpenAI-compatible routes (`respond`, `compact_response`, FIM, transcription)
  take none. So a pinned endpoint fails for each model without a variant for its runner: pin it on an
  endpoint you use for one model, or leave it unset.

The runner decides what the window does:

| | `:mlx` | `:llamacpp` |
| :--- | :--- | :--- |
| The model loads with | `num_ctx` | `num_ctx` |
| A later request for a larger `num_ctx` | keeps the loaded window | loads the model again with it |
| A chat prompt longer than the window | is answered | is an HTTP 400, unless `truncate=true` |
| With `truncate=true` | not measured | a 5,407-token message cut to 2,050 tokens: the first 4 and the tail kept |

To give a loaded `:mlx` model another window, [`unload_model`](@ref) it first. A model with vision loads
with at least 2,048 tokens whatever `num_ctx` asks: by Ollama 0.40.6's source its scheduler raises a
smaller one ("multimodal models require at least 2048 context"), and `embeddinggemma-2:740m-q8_0` asked
for 1,024 loaded with 2,048 on `:llamacpp`. The other [`OllamaOptions`](@ref) are passed through
unchanged; their effect on `:mlx` is not verified.

!!! details "Evidence"
    Measured with Ollama 0.40.6 on an Apple M4 Max: `gemma4:e4b` on MLX, loaded
    with `num_ctx=4096`, kept 4,096 after a request for 8,192 and answered a
    10,842-token prompt, reading a code word from its first sentence;
    `qwen2.5-coder:1.5b-base` on llama.cpp (listed as `ggml`) loaded again for 8,192
    and refused a 5,407-token prompt on 4,096 with HTTP 400. No model ran on the
    ggml runner. Runner pins, 21 October 2026: `:llamacpp` on `kev:0.8b-q8_0`
    answered and `running_models` reported `:llamacpp`; `:mlx` and `:ggml` on
    it, and `:llamacpp` on `tev1:0.8b`, were HTTP 400 `no compatible manifest found
    for runners: <runner>`, also while the model was loaded, and so was every runner
    but the listed one on each of the nine decision models measured. The 2,048-token
    floor of a model with vision was measured on 22 October 2026 (and on Ollama
    0.40.7 on Linux).

## [Models](@id ollama_models)

The model-management verbs return typed results like every other call: an [`OllamaSuccess`](@ref)
holding what the verb reports, an [`OllamaFailure`](@ref) with the HTTP status and Ollama's error body
(404 for a model that is not installed), or an [`OllamaCallError`](@ref) for a call that brought no
usable reply — no server, a timeout, a cancel, a malformed reply. They are not recorded, so their
output here is quoted from a live run:

```julia
ollama = OllamaEndpoint()

r = list_models(service=ollama)      # /api/tags: the installed models
r isa OllamaSuccess || error("the listing failed: $r")
[(m.name, m.runner, m.format) for m in r.response if startswith(m.name, "gemma4") || startswith(m.name, "qwen")]
# 2-element Vector{Tuple{String, Symbol, Symbol}}:
#  ("qwen2.5-coder:1.5b-base", :ggml, :gguf)
#  ("gemma4:e4b", :mlx, :safetensors)

info = model_info("gemma4:e4b"; service=ollama).response    # /api/show
info.capabilities, info.context_length, info.thinking_levels, info.thinking_default
# ([:completion, :vision, :audio, :tools, :thinking], 131072, Union{Bool, String}[false, true], true)

load_model("qwen2.5-coder:1.5b-base"; service=ollama)        # loaded now, with the endpoint's settings
running_models(service=ollama).response                       # /api/ps: memory, context, runner, expiry
# 2-element Vector{OllamaRunningModel}:
#  OllamaRunningModel("qwen2.5-coder:1.5b-base", 2.1 GB, 2.1 GB in VRAM, context 32768)
#  OllamaRunningModel("gemma4:e4b", 9.4 GB, 9.4 GB in VRAM, context 131072)  # loaded by earlier requests
unload_model("qwen2.5-coder:1.5b-base"; service=ollama)
# Run on 25 October 2026 with UniLM's development branch after 0.26.0 and Ollama 0.40.11.
```

[`pull_model`](@ref) downloads a model and [`delete_model`](@ref) removes one from the disk; this page
ran neither:

```julia
percent(p) = isnothing(p.total) ? "" : string(round(Int, 100p.completed / p.total), "%")
pull_model("all-minilm:22m"; service=ollama,
           progress = p -> print("\r", p.status, " ", percent(p)))
delete_model("all-minilm:22m"; service=ollama)  # off the disk; pull_model brings it back
```

Each takes `config` and `cancel`, and an empty name is an `ArgumentError` before any request.
`list_models` needs its `service`: without one it lists the models of the Decisions chapter's default
provider, while the other verbs default to `OllamaEndpoint()`. A listing's [`OllamaModel`](@ref)s hold
each model's `size` on disk, `family`, `parameter_size`, `quantization`, `context_length`, the `runner`
and `format` the listing names and its `capabilities`; [`OllamaModelInfo`](@ref) adds the thinking
levels and the Modelfile `parameters`, and an [`OllamaRunningModel`](@ref) holds the memory a loaded
model takes, the window it loaded with, its runner and when it unloads (`expires_at`).

[`load_model`](@ref) loads a model that completes text with an empty chat request, and any other with
an empty embeddings request, so the first real request finds it in memory; Ollama's API loads a model
"If the messages array is empty" ([API reference](https://github.com/ollama/ollama/blob/v0.40.11/docs/api.md#load-a-model-1)).
It sends what the model's own requests send: a chat model's load the endpoint's options and `shift`,
an embedding model's those options (EmbeddingGemma 2 with its 8,192-token window), and a decision
model's no options at all, as its decision requests carry none. So an endpoint's `num_ctx`, or any
other option, does not apply to a decision model, neither when it is loaded nor when it is asked: the
server's default context applies (73,728 tokens for `kev:0.8b-q8_0` on Ollama 0.40.6, an Apple M4 Max).
Through `OllamaEndpoint(num_ctx=4096)`, the first ask after `load_model("kev:0.8b-q8_0")` took
0.036–0.049 s in five cycles, and an ask without it, which loads the model, 0.81–1.36 s. Witnessed on
Ollama 0.40.6 (21 October 2026) for a chat model (`gemma4:e4b`), embedding models (`embeddinggemma` and
EmbeddingGemma 2) and one decision model (`kev:0.8b-q8_0`); other decision models and other kinds of
model are untested. One `total_deadline` bounds both of a load's requests, the show and the load, and
an endpoint with `keep_alive=0` is refused, since Ollama reads such a load as an unload.
[`unload_model`](@ref) sends an empty chat request with `keep_alive` 0, "a model will be unloaded from
memory" ([API reference](https://github.com/ollama/ollama/blob/v0.40.11/docs/api.md#unload-a-model-1)), which
works for every kind of model.

[`pull_model`](@ref) makes one attempt, never retried: its response headers must arrive within the
`RequestConfig`'s `request_timeout` and remaining `total_deadline`, and after that only
`stream_idle_timeout` bounds the download, the longest gap between bytes. A pull that fails after the
200 has been sent, such as an unknown tag, fails in-band: the call is an `OllamaCallError` holding
Ollama's message (`pull model manifest: file does not exist`).

!!! details "Evidence: choosing a small Gemma 4"
    Measured on Ollama 0.34.4 (Apple M4 Max, 64 GB; 10 runs per text cell, 6 for the
    prime question, 5 per media cell). Both E2B (7.2 GB) and E4B (9.6 GB) called tools correctly in 10 of 10
    runs and named the colour in 5 of 5; E4B named the spoken word in 5 of 5, E2B in 4
    of 5. With thinking off, E2B misjudged "Is 391 prime?" in 6 of 6 runs and E4B
    answered correctly in 6 of 6. E2B decodes faster (86 against 54 tokens per
    second) but thought in 68 of 80 runs where E4B thought in 15, so E4B answered
    sooner on short prompts. The pages of this entry and UniLM's own tests use E4B.

## [When something fails](@id ollama_failures)

`chatrequest!`, `respond`, `embeddingrequest!`, `fim_complete` and `transcribe` return what the server or
the network did as a typed result, and do not throw it ([Results](@ref results_guide)): a failure holds
Ollama's HTTP `status` and, as `response`, the body that came with it, `{"error": …}` with the message;
a call error holds the text of an exception as `error` and the exception as `cause`. UniLM sends a
request again after a transient status — 408, 429, 500, 502 to 504, 520 to 524 and 529 — or a failed
connection, as often as the call's [`RequestConfig`](@ref) allows, and then returns the last answer
([Timeouts & Retries](@ref timeouts_guide)); a streamed request is sent again only while none of its
callbacks has run, and a pull makes one attempt. `RequestConfig(max_attempts=1)`
reports an unreachable server at once. Ollama brings no `RequestConfig` of its own, for decision requests
either: every call takes the field defaults unless the call, a scope or `set_default_config!` sets a
configuration, so a model that takes minutes to load may need a longer bound.

| What you see | Why | What to do |
| :--- | :--- | :--- |
| `ArgumentError` from the call, nothing sent | the request breaks a rule UniLM checks before sending ([Limits and refusals](@ref ollama_limits)) | what its row says |
| `ArgumentError` when the endpoint is built, or a `TypeError` for a value of the wrong type | an argument breaks its rule, or is no argument of the endpoint ([Options](@ref ollama_options)) | what the message names |
| `InvalidConversationError` from `chatrequest!`, nothing sent | with `history=true`, the conversation ends with an assistant message, so the reply could not be appended | a user message after it, or `history=false` |
| `LLMCallError` "no Ollama server answered at http://127.0.0.1:11434" | the server is not running, or runs elsewhere: the connection was refused | start it (`ollama serve` or the app), or point `OLLAMA_HOST` or `base_url` at it |
| `LLMFailure` 404 "model 'gemma4:e2b' not found" | the model is not installed | [`pull_model`](@ref), or `ollama pull` |
| `LLMFailure` 400 "exceeds the available context size" | the prompt does not fit the window, on the `:llamacpp` runner ([Options](@ref ollama_options)) | raise `num_ctx`, or shorten the conversation |
| `LLMFailure` 400 "does not support tools" or "does not support thinking" | the model lacks the capability | check [`model_info`](@ref); pick a model that has it |
| `LLMFailure` 400 after a pause, for an image or a sound: "Multimodal data provided, but model does not support multimodal requests." | the model sees or hears nothing; it was loaded first | a model with `:vision` or `:audio` among its capabilities |
| a 400 (404 from [`model_info`](@ref) and [`load_model`](@ref)) "no compatible manifest found for runners: …" | the endpoint pins a runner the model has no variant for | pin the runner [`list_models`](@ref) names for the model, or leave `runner` unset ([Runners](@ref ollama_runners)) |
| `EmbeddingFailure` 400 "exceeds the context length" | the input is longer than the model's window | split it into chunks ([Long inputs](@ref ollama_embedding_window)) |
| `EmbeddingFailure` 413 | an EmbeddingGemma 2 GGUF build, through an endpoint that sets `num_ctx` but not `num_batch` ([Long inputs](@ref ollama_embedding_window)) | an endpoint that sets neither |
| `EmbeddingCallError` "an empty or all-zero vector" | a reply row that is no embedding ([Embeddings](@ref ollama_embeddings)) | the model, or another one |
| `LLMCallError` with `status` `nothing` from a stream, the line in `error` | an `{"error": …}` line in the HTTP 200 stream ([Streaming](@ref ollama_streaming)); not sent again, and nothing is appended | the turn again |
| `LLMFailure` with `status` 200 from a stream | the stream ended before its last line, the one whose `done` is `true`; `response` holds what arrived, and nothing is appended | the turn again |
| `LLMFailure` 503 | "the server is overloaded": more requests wait than `OLLAMA_MAX_QUEUE` lets queue ([FAQ](https://docs.ollama.com/faq)); sent again as a transient status | fewer requests at once, or a larger queue on the server |
| a call error with `status` `nothing` and a `UniLMTimeout` as `cause` | a bound of the call's `RequestConfig` ran out: a model loading, a request waiting its turn, or a long reply ([Timeouts & Retries](@ref timeouts_guide)) | [`load_model`](@ref) first, a longer bound, or a stream |
| a call error whose `cause` is a `UniLMCancelled` | a stop: the stream's `close[] = true` or a cancelled `CancelToken` ([Streaming](@ref ollama_streaming)) | nothing, if the stop was yours |
| a call error whose `cause` is the exception your `callback` or `on_tool_call` threw | the exception ended the stream; it is not sent again, and nothing is appended; a thrown value that is no `Exception` leaves `cause` `nothing`, its text in `error`; an `InterruptException`, or a recorded call's `ReplayMissError` or `RecordingWriteError`, is thrown at `fetch` instead | the callback |
| an empty answer with finish reason `"length"` | thinking used the whole `max_tokens` budget ([Reasoning](@ref ollama_thinking)) | a larger `max_tokens`, or `reasoning_effort="none"` |
| `OllamaFailure` 404 from a model verb | the model is not installed | [`pull_model`](@ref), or a name `list_models` lists |
| `OllamaCallError` "pull model manifest: file does not exist" | [`pull_model`](@ref) asked for a tag the registry does not hold | a tag the model's page lists |
| `LLMResultError` from `text`, `output_text`, `embedding_vectors` or `fim_text` | the result is a failure, which holds no reply | check `issuccess(result)` first |

Ollama runs one request at a time per model unless `OLLAMA_NUM_PARALLEL` says otherwise — "The maximum
number of parallel requests each model will process at the same time, default 1" — and queues the rest
([FAQ](https://docs.ollama.com/faq)). A streamed call receives nothing until its turn comes, nor while
Ollama loads its model, so with many long streams in flight, or a slow machine, raise
`stream_idle_timeout` ([Timeouts & Retries](@ref timeouts_guide)); it bounds the wait for the first
byte too.
