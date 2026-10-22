# UniLM.jl — LLM Reference

> **Single-file reference for LLM code-generation systems.** This is a **Julia** package.
> Julia ≥ 1.13 · Deps: `HTTP.jl` (≥ 2.11.2), `JSON.jl` (≥ 1.12), `PrecompileTools.jl`, `Base64`, `Dates`, `SHA`
> Repo: <https://github.com/algunion/UniLM.jl>

## Installation

```julia
using Pkg
Pkg.add("UniLM")
using UniLM
```

## Environment Variables

| Variable                           | Required by                       | Description                                     |
| ---------------------------------- | --------------------------------- | ----------------------------------------------- |
| `OPENAI_API_KEY`                   | `OPENAIServiceEndpoint` (default) | OpenAI API key                                  |
| `AZURE_OPENAI_BASE_URL`            | `AZUREServiceEndpoint`            | Azure deployment base URL                       |
| `AZURE_OPENAI_API_KEY`             | `AZUREServiceEndpoint`            | Azure API key                                   |
| `AZURE_OPENAI_API_VERSION`         | `AZUREServiceEndpoint`            | Azure API version string                        |
| `AZURE_OPENAI_DEPLOY_NAME_<MODEL>` | `AZUREServiceEndpoint`            | Deployment for a model, read at call time: model id upper-cased, other characters → `_` (`AZURE_OPENAI_DEPLOY_NAME_GPT_5_2`) |
| `GEMINI_API_KEY`                   | `GEMINIServiceEndpoint`           | Google Gemini API key                           |
| `ANTHROPIC_API_KEY`                | `ANTHROPICServiceEndpoint`        | Anthropic (Claude) API key                      |
| `DEEPSEEK_API_KEY`                 | `DeepSeekEndpoint`                | DeepSeek API key                                |
| `MISTRAL_API_KEY`                  | `MistralEndpoint`                 | Mistral AI API key                              |
| `TYPESAFE_API_KEY`                 | `TYPESAFEServiceEndpoint`         | TypeSafe System One (Jev) API key               |
| `TYPESAFE_BASE_URL`                | `TYPESAFEServiceEndpoint`         | Optional API root override (default `https://api.typesafe.ai`) |
| `TYPESAFE_DEFAULT_MODEL`           | `TYPESAFEServiceEndpoint`         | Model used when a call names none (default `jev-latest`) |
| `OLLAMA_HOST`                      | `OllamaEndpoint`                  | Server read by `OllamaEndpoint()` when no `base_url` is given, by the rules of Ollama's own clients (`"gpu-box"` → `http://gpu-box:11434`; default `http://127.0.0.1:11434`) |

## The APIs

UniLM.jl wraps these surfaces:

1. **Chat Completions** (`Chat` + `chatrequest!`) — stateful, message-based conversations with tool calling, streaming, structured output. Supports OpenAI, Azure, Gemini (native), Anthropic (native), DeepSeek, Ollama, Mistral, and any OpenAI-compatible provider.
2. **Responses API** (`Respond` + `respond`) — newer, more flexible API with built-in tools (web search, file search), multi-turn chaining via `previous_response_id`, reasoning support for O-series models, structured output. OpenAI Responses by default; the unified `respond` verb also targets Google's Gemini Interactions via `service=GEMINIServiceEndpoint` (see the Agentic Workflows guide).
3. **Image Generation** (`ImageGeneration` + `generate_image`, `ImageEdit` + `edit_image`) — text-to-image and edits with OpenAI's `gpt-image-2.5-sunburst`, or Gemini's Nano Banana (`gemini-nano-banana-2.1`) with `service=GEMINIServiceEndpoint`.
4. **Embeddings** (`Embeddings` + `embeddingrequest!`) — vector embeddings. Multi-provider via `service` parameter; the task prompts of EmbeddingGemma and `gemini-embedding-2` via `task`; images and audio (`EmbeddingInput`) through Ollama.
5. **FIM Completion** (`FIMCompletion` + `fim_complete`) — code infilling. DeepSeek (beta), Mistral, Ollama.
6. **System One** (`ask`, `ask_each`, plus `nl_classify`, `nl_dispatch` and `@branch`) — enumerated questions about a piece of `state`, answered with a typed value and a calibrated probability distribution instead of generated text. One surface over three backends: TypeSafe's Jev (`TYPESAFEServiceEndpoint`, the default), OpenAI's Decisions API (`OPENAIServiceEndpoint`, public beta) and local decision models on Ollama (`OllamaEndpoint()`); any endpoint declaring `:system_one` is reached through a provider protocol (`UniLM.system_one_protocol`); `with_system_one` sets the backend for a scope. Not a chat surface.

**Which API to use:**
- **Chat Completions** — best for multi-turn conversations; broadest provider support. Use for chat, tool calling, or streaming across any supported backend.
- **Responses API** — simpler for single-shot or chained requests; built-in web search, file search, MCP, computer use tools. OpenAI Responses plus Google's Gemini Interactions via the unified `respond` verb (see the Agentic Workflows guide).
- **FIM Completion** — code infilling between prefix and suffix. DeepSeek (beta), Mistral and Ollama (vLLM's completions endpoint does not accept `suffix`).
- **System One** — classify, rank, screen or route unstructured input when the outcomes are a fixed set you can enumerate and you want a probability, not prose. `ask` returns typed results and `ask_each` one per item, in order; `nl_dispatch` turns the answer into ordinary Julia multiple dispatch and `@branch` into an inline switch, both throwing rather than guessing a branch. A `min_confidence` tuned on one backend does not carry over to another.

---

## Service Endpoints

```julia
abstract type ServiceEndpoint end
abstract type OpenAIWireEndpoint <: ServiceEndpoint end   # OpenAI-wire backends inherit encode/decode/SSE
struct OPENAIServiceEndpoint <: OpenAIWireEndpoint end   # default — uses OPENAI_API_KEY
struct AZUREServiceEndpoint  <: OpenAIWireEndpoint end   # uses AZURE_OPENAI_* env vars
struct GEMINIServiceEndpoint <: ServiceEndpoint end       # native generateContent — GEMINI_API_KEY
struct GEMINIOpenAIServiceEndpoint <: OpenAIWireEndpoint end # Gemini via OpenAI-compat shim — GEMINI_API_KEY
struct ANTHROPICServiceEndpoint <: ServiceEndpoint       # native Messages API — ANTHROPIC_API_KEY
    betas::Vector{String}                                 # comma-joined in the anthropic-beta header
    user_profile_id::Union{String,Nothing}                # anthropic-user-profile-id header
    workspace_id::Union{String,Nothing}                   # anthropic-workspace-id header
    inference_geo::Union{String,Nothing}                  # "global" | "us": the request body's inference_geo
end
struct TYPESAFEServiceEndpoint <: ServiceEndpoint end     # TypeSafe System One (Jev) — TYPESAFE_API_KEY
struct GenericOpenAIEndpoint <: OpenAIWireEndpoint    # any OpenAI-compatible provider
    base_url::String
    api_key::String
end
# DeepSeekEndpoint <: OpenAIWireEndpoint  (constructor below)
struct OllamaEndpoint <: OpenAIWireEndpoint   # local Ollama: native /api/chat, /api/embed, /v1/systemone;
                                              #   /v1/responses, /v1/responses/compact, /v1/completions,
                                              #   /v1/audio/transcriptions
    base_url::String                          # server root; default OLLAMA_HOST, else "http://127.0.0.1:11434"
    keep_alive::Union{Float64,Nothing}        # seconds the model stays loaded (Inf: until the server stops)
    truncate::Bool                            # false (default): input that does not fit the context window is a 400
    shift::Union{Bool,Nothing}                # context shift while generating; nothing = server default
    runner::Union{Symbol,Nothing}             # :mlx | :llamacpp | :ggml, as a model's listing names it; nothing = the server's choice
    options::OllamaOptions                    # num_ctx, top_k, min_p, … (fields Chat does not have)
end

# Convenience constructors
OllamaEndpoint(; base_url=<OLLAMA_HOST or "http://127.0.0.1:11434">, keep_alive=nothing,
               truncate=false, shift=nothing, runner=nothing, options...)   # options: OllamaOptions keywords, e.g. num_ctx=32_768
MistralEndpoint(; api_key=ENV["MISTRAL_API_KEY"])     # Mistral AI
DeepSeekEndpoint(; api_key=ENV["DEEPSEEK_API_KEY"])   # DeepSeek
ANTHROPICServiceEndpoint(; betas=String[], user_profile_id=nothing, workspace_id=nothing, inference_geo=nothing)
    # the bare type (service=ANTHROPICServiceEndpoint) is the endpoint with no options; a blank value, a
    # control character, a comma in a beta name or another inference_geo → ArgumentError; == and hash by
    # value; list_models sends the profile and workspace headers but no betas (Anthropic marks anthropic-beta
    # deprecated on GET /v1/models)

# Type alias for service fields — accepts both marker types and instances:
const ServiceEndpointSpec = Union{Type{<:ServiceEndpoint}, ServiceEndpoint}
# Built-in types: Chat(service=OPENAIServiceEndpoint)      — passed as the type
# Instance types: Chat(service=DeepSeekEndpoint())          — passed as a constructed value
```

### Provider Compatibility

UniLM talks to **native** provider APIs where it implements them, and rides the **OpenAI-compatible** standard elsewhere:

| Access path | Providers |
|---|---|
| **Native backends** (own wire format) | OpenAI (Chat + Responses), Anthropic (Messages), Gemini (generateContent + agentic Interactions), Ollama (`/api/chat` + `/api/embed`) |
| Chat Completions (OpenAI-compat) | OpenAI, Azure, DeepSeek, Mistral, vLLM, LM Studio, Gemini (compat shim) |
| Embeddings (OpenAI-compat) | OpenAI, Gemini (compat shim), Mistral, vLLM |
| Responses API | OpenAI, Ollama (stateless), vLLM, Amazon Bedrock (emerging Open Responses) |
| Image Generation | OpenAI (Images API), Gemini (native `generateContent`, Nano Banana) |

Anthropic and native Gemini use their **own** APIs here (`ANTHROPICServiceEndpoint` / `GEMINIServiceEndpoint`) — not an OpenAI-compat shim — and that is the recommended path. The OpenAI-compat Gemini shim (`GEMINIOpenAIServiceEndpoint`) exists for embeddings and drop-in compatibility. Google calls `generateContent` legacy (still fully supported) and launches new models and features on its Interactions API, which `respond(…; service=GEMINIServiceEndpoint)` speaks.

Register additional Azure deployments at runtime:

```julia
add_azure_deploy_name!(model::String, deploy_name::String)
# e.g. add_azure_deploy_name!("gpt-5.2", "my-deployment")
```

Pass the backend via the `service` keyword on `Chat`, `Respond`, or `ImageGeneration`:

```julia
Chat(service=AZUREServiceEndpoint, model="gpt-5.2")
Respond(service=OPENAIServiceEndpoint, input="Hello")
```

### Ollama (local)

`OllamaEndpoint` needs no key and has no default model (`Chat(service=OllamaEndpoint())` throws `ArgumentError`; name an installed model, e.g. `"gemma4:e4b"`). Contract of its native chat wire (`/api/chat`):

- **Mapped**: `model`, `messages` (attachments → `images`, images and sound clips only — a `DocumentAttachment` or an image's `detail` level throws `ArgumentError` before I/O; captured `:ollama` thinking → `thinking`; tool calls with their ids; tool results → `tool_name` + `tool_call_id`), `tools`, `stream` (always sent), `temperature`/`top_p`/`seed`/`stop`/`presence_penalty`/`frequency_penalty` and `max_completion_tokens` (else `max_tokens`) → `options` (as `num_predict`), `response_format` → `format` (a schema; `json_object` → `{"type":"object"}`), `reasoning_effort` → `think`, plus the endpoint's `OllamaOptions` → `options`, `keep_alive`, `truncate` (always sent, default `false`), `shift` and `runner` (`"mlx"`, `"llamacpp"`, `"ggml"`; also sent on `/api/embed`, `/v1/systemone`, `model_info`, `load_model`, `unload_model` and `pull_model`; never on `delete_model`, the listings or the OpenAI-compatible routes). Ollama matches `runner` against the runner the model's manifest names, `OllamaModel.runner` (which can differ from the serving one `running_models` reports: a GGUF model listed `:ggml` is served by llama.cpp, and pinned to `:llamacpp` is the 400): a model published for one runner — every model measured, Ollama 0.40.6 — answers as if unpinned for that runner, and any other is HTTP 400 `no compatible manifest found for runners: <runner>` (404 from `/api/show`, and so from `model_info` and `load_model`), never a fallback, even while the model is loaded; a model published for several is served by its variant for the pinned one (by Ollama's source; untested). The pin goes with every request of the endpoint.
- **Context window per runner** (Ollama 0.40.6, measured; ggml untested): llama.cpp loads the model with `num_ctx`, loads it again for another `num_ctx`, and answers a prompt that does not fit with 400 unless `truncate=true` — which can cut inside one over-long message, keeping its first tokens and its tail (5,407 → 2,050 tokens, first 4 kept). MLX (Apple Silicon) loads with `num_ctx`, keeps a loaded model's window when a later request asks for more (`unload_model` first to change it), and does not bound a chat prompt by it (a 10,842-token prompt answered on a 4,096 load). A model with vision loads with at least 2,048 tokens whatever `num_ctx` asks (Ollama's scheduler, by its source; measured: `embeddinggemma-2:740m-q8_0` asked for 1,024 loaded with 2,048 on Ollama 0.40.6 and 0.40.7). Other `OllamaOptions` are passed through; their effect on MLX is unverified.
- **Thinking**: `"none"` → `think: false`; `"low"`/`"medium"`/`"high"` pass through; other efforts throw. Unset → the model's default, except that a `response_format` sends `think: false` (Ollama constrains output only after thinking ends); a `response_format` with a thinking effort throws.
- **Refused** (`ArgumentError` before I/O): `logit_bias`, `user`, `stream_options`, `verbosity`, `store`, `metadata`, `service_tier`, `logprobs`, `top_logprobs`, `prediction`, `modalities`, `audio`, `web_search_options`, `prompt_cache_key`, `safety_identifier`, `prompt_cache_options`, `moderation`; `tool_choice` other than `"auto"`; tools with `strict=true`; a message `name`; a tool result that answers no earlier call. `parallel_tool_calls` is not sent (the model may return several calls in a turn).
- **Replies**: `message.thinking` → `ProviderContent(:ollama, [Dict("thinking" => …)])`; a tool call without an id gets `"unilm_call_<k>"` (never sent back); usage = `prompt_eval_count` / `eval_count` (thinking included), cached = `prompt_eval_cached_count`; `done_reason` → `finish_reason` (`"tool_calls"` when calls are present). Streams are newline-delimited JSON; an `{"error": …}` line on a 200 stream is an `LLMCallError`, an error body with another status an `LLMFailure` with that status.
- **Embeddings** go to `/api/embed` (`truncate`, `keep_alive`, `options` and `runner` from the endpoint; `user` refused): a text item is sent as its (prompted) string, an item with media as `{"text"?, "image"?}` or `{"text"?, "audio"?}` with raw base64 media (an item holding both an image and a clip throws `ArgumentError` before I/O: Ollama passes an item's media in a random order, and ten requests of one such item gave `embeddinggemma-2:740m-q8_0` two different vectors on 0.40.6); an EmbeddingGemma 2 request (model name matching `embeddinggemma-?2`, not followed by a digit) adds `options.num_ctx = options.num_batch = 8192` unless the endpoint sets either (otherwise the llama.cpp builds measured refuse long inputs with 413 — `740m-q8_0` over ~2,037 tokens; with the window, measured on `740m-q8_0`, `740m-mxfp8` and `270m-q8_0-text`: 5,991 tokens embedded, 8,331 refused with 400; the 440m and 570m builds untested). **`transcribe`** goes to `/v1/audio/transcriptions` (multipart `model`, `file`, `language`, `prompt`, `response_format` `"json"` or unset; no default model — `ArgumentError`; `"text"` is refused before I/O, since Ollama reports a failure after its reply has begun only in JSON and `"text"` would read it as an empty transcript — leave it unset and read the text with `transcript_text`; `temperature`, `languages`, `keywords` and other formats refused before I/O too; the reply is read as JSON, though Ollama labels it `application/x-ndjson`). **`respond`** goes to the stateless `/v1/responses`: only `model`, `input`, `instructions`, `tools` (function tools without `strict=true`, `async`, `allowed_callers`, `defer_loading` or `output_schema`), `tool_choice="auto"`, `temperature`, `top_p`, `max_output_tokens`, `stream`, `text` (a `json_schema` format with its schema, no verbosity) and `reasoning` (effort only) are accepted; a `json_schema` format sends `reasoning.effort="none"` when unset and throws with a thinking effort; an `input_file` part in any form (`input_file(document)` or `input_file(; url, id, file_data)`: Ollama documents no file inputs there), an `input_image` given by file id, an `InputMessage` `phase` and a `seed` (`/v1/responses` reads none; a `Chat`'s seed reaches the model) throw `ArgumentError` before I/O; `tool_loop(::Respond)` throws; `get_response`, `cancel_response`, `delete_response` and `list_input_items` throw `ArgumentError` before I/O (nothing is stored); `compact_response` goes to `/v1/responses/compact` and needs `model=` (none → the `Chat`'s `ArgumentError` before I/O). **FIM** uses `/v1/completions`. A `respond` (streamed or not) or FIM result of a cloud model whose reply drops the tag keeps the requested name in `response.model`, the server's in `response.raw["model"]`.
- **System One** (`ask` and the NL verbs) goes to `/v1/systemone` (`UniLM.OllamaSystemOneProtocol`; Ollama 0.35 or later, 0.35.1 or later for images, 0.40.0 or later for the MLX decision models tev1 and clef-flash; measured on 0.40.6; recordings under `/ollama/v1/systemone`). No default model: `model=` (e.g. `"tev1:4b"`) or `with_system_one(OllamaEndpoint(); model=…) do … end`, else `ArgumentError`. Refused before I/O — what the route refuses in its shared checks or documents as its maximum (Ollama 0.40.6): more than 64 questions; a Choice of more than 255 options (one option is sent); Score levels outside 2–10; a blank text `state` without images; Box questions (by Ollama's source, a Kev model on Ollama's own runner, such as `kev:0.8b-mxfp8`, skips the shared checks and takes 1–255 levels and a blank state; untested); what Ollama's API reference does not list, though some models read it: `instructions` (optional) that are not a string, object or array, an option described by anything but a string or `nothing`, Score levels or Noul criteria that are not strings (measured: `kev:0.8b-q8_0` and `d1:3b` answered an object description, `tev1:0.8b` refused it); UniLM's own refusal (Ollama would read them as none), blank `instructions` (its message: "Ollama reads blank `instructions` as none; those of "<name>" are blank: say what it asks (most local decision models require instructions)"); any option but `keep_alive` (seconds, `Inf` = stay loaded; unset → the endpoint's `keep_alive`; written last; left out of the recording key and file, `record_body`). Images: the state is an `ImageAttachment` or a vector/tuple of images then strings; the images go in top-level `images` (raw base64, request order, shared by all questions, shown before the text), the strings become `state` (none → `""`, one → that string, several → an array; a blank string allowed with images); refused before I/O: an image after a string, inside a JSON object/array, beside non-strings, or in a question. The body is `state`, `model`, `questions`, then `runner` (when set; keyed), `images` (when any; keyed as `data:<mime>;sha256,<hex>`), `keep_alive` (when set; not keyed). A model that takes less answers its own 400 → `SystemOneFailure` with Ollama's message (measured, Ollama 0.40.6: tev1 ≤ 24 options, 27 → `criteria must contain 2–26 candidates`; nimble/clef-flash ≤ 26; kev, strands-decider, d1, d1-omni answered 27; one option: `kev:0.8b-q8_0`, `kev:9b-q4_K_M`, `d1-omni:600m` and `d1:3b` answered `{"a": 1.0}`, `tev1:0.8b`, `tev1:4b`, `nimble:9b` and `clef-flash:9b` 400 `question "pick": criteria must contain 2–26 candidates`, `strands-decider:2b` 400 `question "pick": criteria must contain 2–255 candidates`; the route accepts a question without `instructions`, but `tev1:4b`, `nimble:9b` and `strands-decider:2b` answered 400 `instructions must be a nonempty string, object, or array` and `kev:0.8b-q8_0` 400 `questions.<name>: "instructions" must be provided`, while `clef-flash:9b` answered — pass `instructions` with local models; a model without vision → its own 400, in its own words: `tev1:0.8b` `image inputs are not supported by this decision model`, `kev:0.8b-q8_0` `registry.ollama.ai/library/kev:0.8b-q8_0 does not support vision`; vision measured on `clef-flash:9b`, `d1:3b`, `d1-omni:600m`, other models whose `/api/show` capabilities include vision untested). Ollama caps a request's text plus schema (images excluded) at 64 KiB for the models measured — not checked locally; a 76,284-byte state → a 413 `SystemOneFailure` (`text and schema must not exceed 64 KiB`) on `kev:0.8b-q8_0` (the installed GGUF Kev), `d1-omni:600m` and `tev1:0.8b`; by Ollama's source a Kev model on Ollama's own runner gets 1 MiB (untested). A `RawQuestion` is sent verbatim. Choice/Score `confidence` is recomputed from the probabilities with TypeSafe's published formulas (`(n·p_max − 1)/(n − 1)` clamped to `0 … 1`; `max(0, 1 − Σ pᵢ·|i − m| / MAD_unif)`, `m` the lowest most likely level; `1` for a single outcome); the server's value (measured: the five MLX models `1 − H(p)/ln n`; Kev and the llama.cpp path TypeSafe's) stays in `raw`. Probability-sum tolerance `n·0.5e-4 + 1e-6`, and the chosen option may sit `1e-4 + 1e-6` below the most probable. `request_id` is `nothing`; usage cached = `prompt_eval_cached_count` (0 on the llama.cpp path, which omits it); a cloud model whose reply drops its tag keeps the requested name; an `{"error": …}` body is a `SystemOneFailure` with that message, and a missing model's 404 adds `ollama pull <model>`. No default `RequestConfig` of its own (the field defaults apply: loading a model can take minutes); `ask_each` keeps 1 request in flight unless `ntasks` says otherwise.
- `estimated_cost` of an Ollama `LLMSuccess` / `EmbeddingSuccess` / `SystemOneSuccess` / `ResponseSuccess` / `FIMSuccess` is `0.0` without a warning (unless `model=` is passed) for a local model; a cloud model — its tag (after the last `:` with no `/` after it, surrounding whitespace trimmed, compared case-insensitively) is `cloud` or ends in `-cloud`, e.g. `gpt-oss:120b-cloud` — is priced by the requested name like any other (no row → `0.0` with the one-time warning), never as a free local call: a `SystemOneSuccess`, `ResponseSuccess` or `FIMSuccess` whose reply drops the tag keeps the requested name, the server's in its `raw` (the untagged reply is inferred from Ollama's source, which sends the cloud the name without its tag and passes the reply through; not observed). Detection goes by the name alone: a cloud model copied under a name without the tag (`ollama cp`) is priced as local, `0.0` with no warning. A connection failure's error text ends with how to start the server.
- `OllamaEndpoint` keywords: `keep_alive` in [0, 1e9] seconds or `Inf`; `runner` ∈ (`:mlx`, `:llamacpp`, `:ggml`) or `nothing` (anything else, a `String` included, throws); `OLLAMA_HOST` read like Ollama's clients (bare or bracketed IPv6; `0.0.0.0`/`::` → loopback; `ollama.com` → `https://ollama.com:443`).

#### Ollama model management

```julia
struct OllamaModel                       # list_models(service=OllamaEndpoint()) → OllamaSuccess{Vector{OllamaModel}} (/api/tags)
    name::String
    size::Int                            # bytes on disk
    digest::String
    modified_at::String                  # RFC 3339 text
    family::Union{String,Nothing}        # details; "" reads as nothing
    parameter_size::Union{String,Nothing}
    quantization::Union{String,Nothing}
    context_length::Union{Int,Nothing}
    runner::Union{Symbol,Nothing}        # the runner the listing names (:mlx, :llamacpp, :ggml), the one OllamaEndpoint(runner=…)
                                         #   must name to reach it; may differ from the serving one
    format::Union{Symbol,Nothing}        # :gguf | :safetensors (lower case; "" → nothing)
    capabilities::Vector{Symbol}         # e.g. :completion, :tools, :insert, :vision, :embedding, :thinking, :audio, :decision
    raw::Dict{String,Any}
end

struct OllamaModelInfo                   # model_info → OllamaSuccess{OllamaModelInfo} (/api/show)
    name::String
    capabilities::Vector{Symbol}
    context_length::Union{Int,Nothing}
    thinking_levels::Vector{Union{Bool,String}}
    thinking_default::Union{Bool,String,Nothing}
    parameters::Dict{String,Any}         # Modelfile defaults, numbers parsed
    family::Union{String,Nothing}
    parameter_size::Union{String,Nothing}
    quantization::Union{String,Nothing}
    runner::Union{Symbol,Nothing}        # the runner /api/show's details name; nothing when none
    format::Union{Symbol,Nothing}        # :gguf | :safetensors
    raw::Dict{String,Any}
end

struct OllamaRunningModel                # running_models → OllamaSuccess{Vector{OllamaRunningModel}} (/api/ps)
    name::String
    size::Int                            # bytes of memory
    size_vram::Int
    context_length::Union{Int,Nothing}   # the window it was loaded with
    expires_at::String
    runner::Union{Symbol,Nothing}        # the runner serving it now
    raw::Dict{String,Any}
end

struct OllamaPullProgress
    status::String
    digest::Union{String,Nothing}
    completed::Union{Int,Nothing}
    total::Union{Int,Nothing}
end

model_info(name; service=OllamaEndpoint(), config=nothing, cancel=nothing)    # /api/show with the endpoint's runner
running_models(; service=OllamaEndpoint(), config=nothing, cancel=nothing)
pull_model(name; service=OllamaEndpoint(), progress=nothing, config=nothing, cancel=nothing)   # progress(::OllamaPullProgress);
    # the endpoint's runner goes with it (by Ollama's source, untested: a model published for several runners
    # pulls that runner's variant; one published for another runner fails in-band → OllamaCallError)
load_model(name; service=OllamaEndpoint(), config=nothing, cancel=nothing)     # /api/show, then an empty /api/chat
    # (a :completion model: the endpoint's options and shift) or an empty /api/embed (an :embedding model: the
    # options its embeddings requests carry, EmbeddingGemma 2's 8192 window included; any other, e.g. a decision
    # model: no options, as System One requests carry none); keep_alive and runner go with it; keep_alive=0 →
    # ArgumentError; a runner the model has no variant for → OllamaFailure 404 from the show; one total_deadline
    # bounds both requests
unload_model(name; service=OllamaEndpoint(), config=nothing, cancel=nothing)   # empty /api/chat, keep_alive 0, runner
delete_model(name; service=OllamaEndpoint(), config=nothing, cancel=nothing)   # DELETE /api/delete
    # each → OllamaSuccess{T} | OllamaFailure (404: not installed) | OllamaCallError; an empty name → ArgumentError
```

#### Provider model listings

`list_models(; service)` on `OPENAIServiceEndpoint`, `ANTHROPICServiceEndpoint` (the bare type or an
instance) or `GEMINIServiceEndpoint` lists every model the provider serves the key (`GET /v1/models`
on OpenAI and Anthropic, `GET /v1beta/models` on Gemini), every page of a paged listing (Anthropic's
`after_id`, Gemini's `pageToken`; a repeated cursor is a `ModelListCallError`); all three declare
`:models`. It rides the shared retry seam (`max_attempts`; `total_deadline` bounds the whole listing)
and is not recorded by `with_recorded_answers`.

```julia
struct ProviderModel
    id::String                                  # the request's model: OpenAI/Anthropic id, Gemini name without "models/"
    display_name::Union{String,Nothing}         # Anthropic display_name, Gemini displayName; nothing on OpenAI
    created::Union{DateTime,Nothing}            # UTC: OpenAI created, Anthropic created_at (the Unix epoch → nothing); nothing on Gemini
    input_token_limit::Union{Int,Nothing}       # Anthropic max_input_tokens, Gemini inputTokenLimit; nothing on OpenAI
    output_token_limit::Union{Int,Nothing}      # Anthropic max_tokens, Gemini outputTokenLimit; nothing on OpenAI
    capabilities::Dict{String,Any}              # Anthropic's capabilities object (thinking types, effort levels, …),
                                                #   Gemini's supportedGenerationMethods and thinking; empty on OpenAI
    deprecated::Union{Date,Nothing}             # Anthropic deprecated_at (UTC date); nothing otherwise
    retires::Union{Date,Nothing}                # OpenAI shutdown_date, Anthropic retires_at (UTC date); nothing on Gemini
    raw::Dict{String,Any}                       # the entry as sent
end

struct ModelListSuccess <: LLMRequestResponse
    models::Vector{ProviderModel}               # every page's models, in the order listed
    raw::Vector{Dict{String,Any}}               # the parsed body of each page, in the order fetched
end
@kwdef struct ModelListFailure <: LLMRequestResponse   # an error status on any page
    response::String
    status::Int
    request_id::Union{String,Nothing} = nothing
end
@kwdef struct ModelListCallError <: LLMRequestResponse # no listing: a missing key (named), no connection, a timeout,
    error::String                                      #   a cancel, or a reply that is not the documented listing
    status::Union{Int,Nothing} = nothing
    cause::Union{Nothing,Exception} = nothing
end
```

The Anthropic listing sends the endpoint's user-profile and workspace headers but no `anthropic-beta`
(Anthropic's Models API reference marks that header deprecated on `GET /v1/models`). OpenAI keeps listing
a model after its `shutdown_date`, so a listed model is not always one that still answers: compare
`retires` with today's date. Anthropic lists active and deprecated models, not retired ones, and may
revise a `retires_at` it lists. A listing is the provider's view for the key that asks.

---

## Chat Completions API

### Chat

```julia
@kwdef struct Chat
    service::ServiceEndpointSpec = OPENAIServiceEndpoint
    model::String = ""
    messages::Vector{Message} = Message[]
    history::Bool = true
    tools::Union{Vector{Tool},Nothing} = nothing
    tool_choice::Union{String,GPTToolChoice,Nothing} = nothing
    parallel_tool_calls::Union{Bool,Nothing} = false
    temperature::Union{Float64,Nothing} = nothing       # 0.0–2.0, mutually exclusive with top_p
    top_p::Union{Float64,Nothing} = nothing              # 0.0–1.0, mutually exclusive with temperature
    n::Union{Int64,Nothing} = nothing                    # must be 1: a result carries one choice
    stream::Union{Bool,Nothing} = nothing
    stop::Union{Vector{String},String,Nothing} = nothing # max 4 sequences
    max_tokens::Union{Int64,Nothing} = nothing
    max_completion_tokens::Union{Int64,Nothing} = nothing # preferred over max_tokens; native OpenAI sends max_tokens under this name
    presence_penalty::Union{Float64,Nothing} = nothing   # -2.0 to 2.0
    response_format::Union{ResponseFormat,Nothing} = nothing
    frequency_penalty::Union{Float64,Nothing} = nothing  # -2.0 to 2.0
    logit_bias::Union{AbstractDict{String,<:Real},Nothing} = nothing  # values in [-100, 100]
    user::Union{String,Nothing} = nothing                # refused on native OpenAI (safety_identifier, prompt_cache_key)
    seed::Union{Int64,Nothing} = nothing
    reasoning_effort::Union{String,Nothing} = nothing    # none|minimal|low|medium|high|xhigh|max (per-model subset; native OpenAI GPT-6/GPT-5.6 Chat stops at xhigh)
    stream_options::Union{AbstractDict,Nothing} = nothing # unset: OpenAI/DeepSeek streams send {"include_usage": true}
    verbosity::Union{String,Nothing} = nothing           # low|medium|high
    store::Union{Bool,Nothing} = nothing
    metadata::Union{AbstractDict,Nothing} = nothing
    service_tier::Union{String,Nothing} = nothing        # OpenAI auto|default|flex|scale|priority|fast; Gemini standard|flex|priority; Anthropic auto|standard_only|fast
    logprobs::Union{Bool,Nothing} = nothing
    top_logprobs::Union{Int64,Nothing} = nothing
    prediction::Union{AbstractDict,Nothing} = nothing
    modalities::Union{Vector{String},Nothing} = nothing
    audio::Union{AbstractDict,Nothing} = nothing
    web_search_options::Union{AbstractDict,Nothing} = nothing
    prompt_cache_key::Union{String,Nothing} = nothing
    safety_identifier::Union{String,Nothing} = nothing   # replaces deprecated `user`
    _cumulative_cost::Ref{Float64} = Ref(0.0)            # internal
    prompt_cache_options::Union{PromptCacheOptions,Nothing} = nothing  # OpenAI gpt-5.6 and later; Anthropic (top-level cache_control)
    moderation::Union{ModerationConfig,Nothing} = nothing
end
```

`_cumulative_cost` is internal bookkeeping for [`cumulative_cost`](@ref) — read it
through that accessor, never directly.

- **Model defaults**: the declared default is the sentinel `""`; the constructor resolves it, so `Chat().model` reads back `"gpt-6-sol"`. Per provider: `"gpt-6-sol"` for OpenAI, `"gpt-5.2"` for Azure, `"gemini-3.8-flash"` for Gemini (native and OpenAI-compat), `"claude-opus-5-5"` for native Anthropic (whose `max_tokens` defaults to 16000), `"deepseek-flash"` for DeepSeek. For `GenericOpenAIEndpoint` / `OllamaEndpoint` / `MistralEndpoint` and user-defined endpoints without a `UniLM.default_model` method there is no default — an unset model throws `ArgumentError` at construction.
- `history=true`: responses are automatically appended to `messages`.
- `temperature` and `top_p` are mutually exclusive (constructor throws `ArgumentError`).
- `parallel_tool_calls` is auto-set to `nothing` when `tools` is `nothing`.
- **Parameter validation**: the constructor validates at construction time — `temperature` ∈ [0.0, 2.0], `top_p` ∈ [0.0, 1.0], `n` must be `1` (a result carries a single choice), `max_tokens` / `max_completion_tokens` ≥ 1, `presence_penalty` ∈ [-2.0, 2.0], `frequency_penalty` ∈ [-2.0, 2.0], `top_logprobs` ∈ [0, 20], every `logit_bias` value ∈ [-100, 100], `reasoning_effort` ∈ {`"none"`, `"minimal"`, `"low"`, `"medium"`, `"high"`, `"xhigh"`, `"max"`}. Violations throw `ArgumentError`. An empty `tools` vector is stored as `nothing`; a `Vector{<:CallableTool}` is accepted and stored as its `Tool`s.
- **Provider/model limits** (GPT-5.6/GPT-6 tool and sampling rules, the Anthropic and native-Gemini field mappings and the Gemini sampling rule, on both Gemini endpoints) are checked by the provider's encoder: `chatrequest!` throws their `ArgumentError` before any network I/O.
- **OpenAI per-model rules** (native endpoint; `gpt-6-astra`, `gpt-6.1-sol`, `gpt-6-sol`, `gpt-6-luna`, `gpt-5.6`, `gpt-5.6-sol`/`-terra`/`-luna`, by exact id or a `-YYYY-MM-DD` snapshot; other models are checked by the server): Responses efforts `"low"`..`"max"` on Astra and 6.1 Sol, `"none"`..`"max"` on the rest; Chat Completions the same without `"max"` (HTTP 400 on 2026-10-22, though its reference lists it; the refusal points `"max"` to `Respond`); `"minimal"` refused on all (use `"low"`); sampling controls and log probabilities only at `"none"` on `gpt-6-sol`/`gpt-6-luna`, never on Astra or 6.1 Sol; Chat tools only at `"none"` on `gpt-6-sol`/`gpt-6-luna` and GPT-5.6, never on Astra or 6.1 Sol; the `gpt-5.6` alias takes GPT-5.6 Sol's rules. On every model the native endpoint sends `max_tokens` as `max_completion_tokens` (setting both throws), refuses `user` (use `safety_identifier` and `prompt_cache_key`), and takes `service_tier` `"auto"`, `"default"`, `"flex"`, `"scale"`, `"priority"` or `"fast"` (Responses adds `"ultrafast"` on Astra and 6.1 Sol); anything else throws.
- `prompt_cache_options` (each endpoint's rule: `PromptCacheOptions`, below): on OpenAI it is supported for gpt-5.6 and later (gpt-5.4-mini answered HTTP 400 "prompt_cache_options is not supported on this model" on 2026-09-22); Chat Completions takes only its `mode` and `ttl`, and native OpenAI only `ttl="30m"` — `chatrequest!` on a native OpenAI `Chat` with `prewarm` or `comparison_response_id` set throws `ArgumentError` before any request. `Message` content is a plain string, so Chat messages cannot carry cache breakpoints, and the Chat Completions reference says an explicit-mode request without breakpoints "does not use prompt caching": `mode="explicit"` on a `Chat` turns prompt caching off. `moderation` takes a [`ModerationConfig`](@ref).

### Message

```julia
@kwdef struct Message
    role::String                                          # "system", "user", "assistant" or "tool" (else ArgumentError)
    content::Union{String,Nothing} = nothing
    name::Union{String,Nothing} = nothing
    finish_reason::Union{String,Nothing} = nothing        # response-only: "stop", "tool_calls", "length", "content_filter", …; nothing if none reported
    refusal_message::Union{String,Nothing} = nothing
    tool_calls::Union{Nothing,Vector{ToolCall}} = nothing
    tool_call_id::Union{String,Nothing} = nothing         # required when role == "tool"
    provider_content::Union{Nothing,ProviderContent} = nothing
    attachments::Union{Nothing,Vector{Attachment}} = nothing   # user messages only: ImageAttachment / AudioAttachment / DocumentAttachment
    audio_output::Union{Nothing,AudioOutput} = nothing          # assistant messages only: an OpenAI audio reply
end
```

**Validation**: `role` must be one of the four roles; at least one of `content`, `tool_calls`, `refusal_message`, `attachments` or `audio_output` must be non-`nothing`; `tool_call_id` is required when `role == "tool"`; `attachments` only on user messages (an empty vector is stored as `nothing`); `audio_output` only on assistant messages, and with `content` unset or empty the content is its transcript.

**Audio replies**: `AudioOutput(; id, data, expires_at, transcript)` — fields `id::String`, `data::Vector{UInt8}` (decoded from base64), `format::Union{String,Nothing}` (`"wav"`, `"mp3"`, `"flac"`, `"opus"`, `"aac"` by file signature; `nothing` for raw `pcm16`), `expires_at::Union{Int,Nothing}` (Unix seconds; `nothing` when a stream closed after the turn's finish, before the update that alone carries it — the turn is kept; after it OpenAI answers the id with HTTP 400 `audio_not_found`, as observed for an id it never issued), `transcript::String`; an empty `id` throws `ArgumentError`. A Chat with `modalities=["text", "audio"]` and `audio=Dict("voice" => …, "format" => …)` (e.g. `gpt-audio-1.5`) gets it on the reply's `audio_output`, streamed (assembled from `delta.audio`; the transcript is forwarded to the callback as text) or not, with `content` = the transcript (also when the turn's `content` is unset or empty). Sent back on `OPENAIServiceEndpoint` as `{"role":"assistant","content":content,"audio":{"id":id}}`; every other encoder sends the content alone. On `OPENAIServiceEndpoint`, `estimated_cost` bills its audio tokens at the model's audio-token rates (`gpt-audio-1.5`; see `TokenUsage`).

**Attachments**: `ImageAttachment(path_or_bytes; detail=nothing)` (PNG, JPEG, GIF or WebP; `detail` ∈ `"auto"`, `"low"`, `"high"`, `"original"`), `AudioAttachment(path_or_bytes)` (WAV or MP3) and `DocumentAttachment(path)` / `DocumentAttachment(bytes; filename="document.pdf")` (PDF: `%PDF-` only, else `ArgumentError("document data is not PDF (checked by file signature; N bytes)")`; a blank `filename` throws), all `<: Attachment`, recognised by their leading bytes (a URL in place of a path throws). OpenAI-wire endpoints send them as `image_url` (data URL; `detail` only when set) / `input_audio` / `file` (`{"filename", "file_data": "data:application/pdf;base64,…"}`) content parts after the text part; `OllamaEndpoint` sends images and clips in the message's `images` and refuses a document and a detail level; native Anthropic sends an image and a document as base64 `image` / `document` blocks before the text and throws for a sound clip or an image `detail`; native Gemini sends every kind as an `inlineData` part ahead of the text, an image's `detail` as its per-part media resolution (`"low"`/`"high"`; `"auto"` sends none; `"original"`, or any level on the Gemini 2.5 and Gemma 4 models, throws), and on `respond` (Interactions) as `image`/`audio`/`document` parts of an `InputMessage`'s content; a System One request and an embeddings input refuse documents and detail levels. Each refusal is an `ArgumentError` before any request.

**Reasoning**: `reasoning_text(m::Message)` / `reasoning_text(r::LLMSuccess)` → the thinking text a provider returned (`:ollama` `thinking`, `:deepseek` `reasoning_content`, `:anthropic` thinking blocks joined by a blank line, `:gemini` thought parts), else `nothing`; throws `LLMResultError` on a failure result. A native Anthropic request whose `reasoning_effort` is a level (not `"none"`) turns adaptive thinking on with `thinking.display = "summarized"` (except on Opus 4.5, which has no adaptive thinking), so `reasoning_text` holds the summarized thinking.

On the wire a message never sends `finish_reason` (response-only) or `provider_content`, and a refusal travels as `refusal` (with `content: null`). A tool-call turn reads `finish_reason == "tool_calls"` when the provider finished it with `"stop"` or reported none, and keeps any other reason (`"length"`, `"content_filter"`, …). Partial calls are dropped: on a turn that kept another reason, a call whose arguments are not a JSON object (cut off by the token limit) is removed and the turn — text, reason, usage — kept; under `"tool_calls"` such arguments are an `LLMCallError`.

`provider_content` carries provider-native blocks captured for verbatim round-trip — Anthropic thinking blocks (`:anthropic`), Gemini parts with thought signatures (`:gemini`), DeepSeek `reasoning_content` (`:deepseek`, echoed on requests that carry tools), Ollama `thinking` (`:ollama`, echoed as the assistant message's `thinking`); it never serializes on the OpenAI wire.

**Convenience constructors**:

```julia
Message(Val(:system), "You are a helpful assistant")
Message(Val(:user), "Hello!")
Message(Val(:user), "What colour is the shape?", ImageAttachment("shape.png"))   # attachments...
```

**Role constants**: `RoleSystem = "system"`, `RoleUser = "user"`, `RoleAssistant = "assistant"`.

### chatrequest!

```julia
# Mutating form — sends chat.messages, appends response when history=true
chatrequest!(chat::Chat; config=nothing, callback=nothing, on_tool_call=nothing, cancel=nothing)
    -> LLMSuccess | LLMFailure | LLMCallError | Task

# Keyword-argument convenience form — builds a Chat internally
chatrequest!(; messages, config=nothing, cancel=nothing, chat_kwargs...) -> same
chatrequest!(; systemprompt, userprompt, config=nothing, cancel=nothing, chat_kwargs...) -> same
```

- Non-streaming: returns `LLMSuccess`, `LLMFailure`, or `LLMCallError`.
- Streaming (`stream=true`): returns a `Task` whose `fetch` yields the same typed results. Pass a `callback(chunk::Union{String,Message}, close::Ref{Bool})` — text deltas arrive as `String`s (verbatim, in order), then the assembled `Message` at end-of-stream.
- Streaming tool calls: pass `on_tool_call(tc::ToolCall)` to be notified at most once per completed streamed tool call, as calls finish (a call whose arguments do not parse is skipped with a warning; see the [Streaming guide](@ref streaming_guide)).
- Local validation throws before any network I/O, streaming or not: `ArgumentError` when `callback` or `on_tool_call` is passed without `stream=true` ("callback and on_tool_call require stream=true"), the service declares capabilities without `:chat`, or the provider's encoder rejects the request; `InvalidConversationError` when `history=true` and the conversation ends with an assistant message (the reply could not be appended).
- The keyword form takes EITHER `messages` (copied, never mutated) OR both `systemprompt` and `userprompt` (a `String` or a `Message` each); anything else throws `ArgumentError`. Every other keyword is a `Chat` field; `history` controls whether the reply is appended, not what is sent.
- Retries transient statuses (408/429/500/502–504/520–524/529) with exponential backoff and jitter under the resolved [`RequestConfig`](@ref) — `max_attempts` (default 3) and `total_deadline` bound the attempts. `Retry-After` is a floor under the jitter (at most half the budget left after the floor is spread above it), so a rate-limited fan-out does not retry in lockstep; a retry whose wait would exceed the remaining `total_deadline` is not attempted — the call returns the last real response rather than sleeping past the deadline. A 429 whose `error.code` is `credit_balance_exhausted`, `organization_spend_limit_exceeded`, `project_spend_limit_exceeded` or `organization_usage_limit_exceeded` is returned at once (no retry restores spent credit or a reached limit). Timeouts surface as `LLMCallError` with `status=nothing` and the `UniLMTimeout` in `.cause`.
- `cancel::Union{Nothing,CancelToken}` (default: the ambient [`with_cancel`](@ref) token): a cancel before connecting, during the header wait, mid-stream or during a backoff ends the call with `LLMCallError(status=nothing, cause=UniLMCancelled(:token, …))` — never retried, nothing appended, no terminal callback — unless the provider's completion marker already arrived: then the turn is committed, the final-message callback runs, and usage may be missing (on the OpenAI wire the marker is the chunk carrying `finish_reason`, which precedes the usage chunk and `[DONE]`). A pre-cancelled token sends nothing. A TCP connect / TLS handshake in progress finishes (or reaches `connect_timeout`) first.
- Streaming stop: `close[] = true` (in the callback or from any task) ends the stream at once with `LLMCallError(status=nothing, cause=UniLMCancelled(:callback, …))`, unless the provider's completion marker already arrived — then the turn is committed, the final-message callback runs, and usage may be missing. An exception thrown by `callback`/`on_tool_call` ends the call with that exception in `.cause` (never retried, nothing appended). Time inside callbacks is not counted by `stream_idle_timeout`.
- A stream is final at its sentinel (`[DONE]`, Anthropic `message_stop`): the final `Message` callback runs at once, and the result follows when the HTTP body ends, at most 0.5 s later (a body held open longer has its connection closed, not reused).
- Streaming retry boundary: transient failures (including the in-band `overloaded_error`, the documented 529 equivalent) are retried inside the task only until the first `callback`/`on_tool_call` invocation; afterwards failures surface typed. A user `InterruptException` propagates — `fetch` throws a `TaskFailedException` instead of returning a result value.
- Streams use HTTP/1.1 (one connection per stream); OpenAI and DeepSeek streams request `include_usage` when `stream_options` is unset, so streamed turns are costed.
- Long generations (Gemini 4 Argon's long decode on `GEMINIServiceEndpoint`): a reply ending with `CONTINUATION` and a `continuationToken` is continued inside the call — the same request again with the latest token, up to 8 requests, each with the whole `RequestConfig` budget — and the result is one assistant turn, the replies' parts joined and their usage summed field by field, priced at the tier the last reply reported. Still unfinished after the 8th request, the turn is returned with `finish_reason == "continuation"`. A request that fails ends the call with nothing committed, as does a stop or cancel before the last reply's finish reason (a `CONTINUATION` is not the turn's completion marker); the replies already answered — one whose `CONTINUATION` chunk a stop landed on included — are added to `cumulative_cost`. A streamed turn streams every reply into the same callback (one final `Message`). Proven against a local mock; a live CONTINUATION reply was not observed.

### Conversation Management

```julia
push!(chat, message)       # append a Message (the FIRST message must be a system Message)
pop!(chat)                 # remove last message
update!(chat, message)     # append if history=true
issendvalid(chat) -> Bool  # check conversation rules (see below)
length(chat)               # number of messages
isempty(chat)              # true if no messages
chat[i]                    # index into messages
chat[i] = msg              # replace a message — NOT validated (nor are direct edits to chat.messages)
```

`issendvalid` is `true` only when ALL of these hold — note it is stricter than
`push!`, with no tool exemption:

1. at least 2 messages;
2. the first message is a system message;
3. the **last** message is a user message;
4. no two adjacent messages share a role — including `tool`, which `push!` does
   permit consecutively.

**Important:** A `Chat` must begin with a system message. `push!` throws
[`InvalidConversationError`](@ref) on a non-system message pushed onto an empty `Chat`
and on consecutive same-role messages (except `tool`), so
`chat = Chat(); push!(chat, Message(Val(:user), "…"))` raises rather than silently
leaving the chat empty. `pop!` on an empty `Chat` likewise throws
`InvalidConversationError` rather than returning `nothing`. Use `respond("…")`
for a single turn without a system prompt.

A later system message changes the instructions from that point on. It goes in place on
the OpenAI wire and `OllamaEndpoint`; native Gemini joins it into `systemInstruction`;
native Anthropic keeps it in place where the model and position allow, else joins the
top-level `system` — except on Claude Fable 5.1, Opus 5.5, Sonnet 5.5, Haiku 5.5 and any
Claude id without a rule row, where after an assistant turn one that cannot stay in place
goes right after the next user turn (the prompt stays append-only). On native Anthropic a
system message after an assistant turn needs a user message after it (`ArgumentError`
until one follows).

### Tool Calling Types

```julia
# Define a function the model can call
@kwdef mutable struct FunctionSignature
    name::String
    description::Union{String,Nothing} = nothing
    parameters::Union{AbstractDict,Nothing} = nothing   # JSON Schema dict
    strict::Union{Bool,Nothing} = nothing               # strict function calling; nothing = omit (API default)
end

# Wrap it for the tools parameter
@kwdef struct Tool
    type::String = "function"
    func::FunctionSignature
end
Tool(d::AbstractDict)   # construct from dict with keys "name", "description", "parameters", "strict"

# Returned by model when it wants to call a function
@kwdef struct ToolCall
    id::String
    type::String = "function"
    func::GPTFunction       # has .name::String and .arguments::AbstractDict
    # Gemini-3's opaque tool-call signature. Set only by the Gemini decoder
    # (nothing on every other provider) and echoed verbatim on the next turn;
    # deliberately excluded from JSON.lower, so OpenAI-wire bytes are unchanged.
    thought_signature::Union{Nothing,String} = nothing
end

# Your result after executing the function
struct FunctionCallResult{T}
    name::Union{String,Symbol}
    origincall::GPTFunction
    result::T
end
```

The pre-rename names `GPTTool`, `GPTToolCall`, `GPTFunctionSignature`, and
`GPTFunctionCallResult` remain exported as aliases of these types, so existing
code keeps working unchanged.

### ResponseFormat (Structured Output)

```julia
@kwdef struct ResponseFormat
    type::String = "json_object"                               # "json_object" or "json_schema"
    json_schema::Union{JsonSchemaAPI,AbstractDict,Nothing} = nothing
end
ResponseFormat(json_schema)  # shorthand, sets type="json_schema"

@kwdef struct JsonSchemaAPI   # not exported — construct as UniLM.JsonSchemaAPI(...)
    name::String
    description::String
    schema::AbstractDict
    strict::Union{Bool,Nothing} = nothing   # strict schema adherence; nothing = omit (API default)
end
UniLM.json_schema(name, description, schema; strict=nothing)  # → ResponseFormat(JsonSchemaAPI(...)); not exported
```

### Chat Completions Example

```julia
using UniLM

# Build conversation
chat = Chat(model="gpt-5.2")
push!(chat, Message(Val(:system), "You are a helpful assistant."))
push!(chat, Message(Val(:user), "What is the capital of France?"))

result = chatrequest!(chat)
if result isa LLMSuccess
    println(result.message.content)     # "Paris..."
    # chat.messages already has the response appended (history=true)
end

# One-shot via keywords
result = chatrequest!(
    systemprompt="You are a translator.",
    userprompt="Translate 'hello' to French.",
    model="gpt-5.2"
)
```

### Tool Calling Example (Chat)

```julia
weather_tool = Tool(func=FunctionSignature(
    name="get_weather",
    description="Get current weather",
    parameters=Dict(
        "type" => "object",
        "properties" => Dict("location" => Dict("type" => "string")),
        "required" => ["location"]
    )
))

chat = Chat(model="gpt-5.2", tools=[weather_tool])
push!(chat, Message(Val(:system), "You help with weather."))
push!(chat, Message(Val(:user), "Weather in Paris?"))

result = chatrequest!(chat)
if result isa LLMSuccess && result.message.finish_reason == "tool_calls"
    for tc in result.message.tool_calls
        # tc.func.name == "get_weather", tc.func.arguments == Dict("location" => "Paris")
        answer = "22°C, sunny"  # your function result
        push!(chat, Message(role="tool", content=answer, tool_call_id=tc.id))
    end
    result2 = chatrequest!(chat)
    println(result2.message.content)
end
```

### Streaming Example (Chat)

```julia
chat = Chat(model="gpt-5.2", stream=true)
push!(chat, Message(Val(:system), "You are helpful."))
push!(chat, Message(Val(:user), "Tell me a story."))

task = chatrequest!(chat; callback = function (chunk, close_ref)
    if chunk isa String
        print(chunk)            # partial text delta
    elseif chunk isa Message
        println("\n[Done]")     # final assembled message
        # close_ref[] = true    # to stop early → LLMCallError, cause = UniLMCancelled(:callback, …)
    end
end)

result = fetch(task)  # LLMSuccess when complete, else a typed failure
```

---

## Responses API

### Respond

```julia
@kwdef struct Respond
    service::ServiceEndpointSpec = OPENAIServiceEndpoint
    model::String = ""                                # declared as "" and resolved by the constructor
    input::Union{String, Vector}                             # String, Vector{InputMessage}, or Vector{Dict}
    instructions::Union{String,Nothing} = nothing
    tools::Union{Vector,Nothing} = nothing                  # untyped: ResponseTool, CallableTool, Dict; a Chat Tool becomes a FunctionTool
    tool_choice::Union{String,AbstractDict,Nothing} = nothing  # "auto"/"none"/"required", or a tool_choice_* Dict (see below)
    parallel_tool_calls::Union{Bool,Nothing} = nothing
    temperature::Union{Float64,Nothing} = nothing           # 0.0–2.0, mutually exclusive with top_p
    top_p::Union{Float64,Nothing} = nothing                 # 0.0–1.0
    max_output_tokens::Union{Int64,Nothing} = nothing
    stream::Union{Bool,Nothing} = nothing
    text::Union{TextConfig,Nothing} = nothing               # output format
    reasoning::Union{Reasoning,Nothing} = nothing           # O-series models
    truncation::Union{String,Nothing} = nothing             # "auto" or "disabled"
    store::Union{Bool,Nothing} = nothing                    # store for later retrieval
    metadata::Union{AbstractDict,Nothing} = nothing
    previous_response_id::Union{String,Nothing} = nothing   # multi-turn chaining
    user::Union{String,Nothing} = nothing                   # refused on native OpenAI (safety_identifier, prompt_cache_key)
    background::Union{Bool,Nothing} = nothing
    include::Union{Vector{String},Nothing} = nothing
    max_tool_calls::Union{Int64,Nothing} = nothing
    service_tier::Union{String,Nothing} = nothing           # OpenAI "auto","default","flex","scale","priority","fast"; "ultrafast" on gpt-6-astra/gpt-6.1-sol
    top_logprobs::Union{Int64,Nothing} = nothing            # 0–20
    prompt_cache_key::Union{String,Nothing} = nothing
    prompt_cache_retention::Union{String,Nothing} = nothing  # "in_memory","24h" ("24h" only on gpt-5.5/gpt-5.5-pro, server-checked; refused on the gpt-6*/gpt-5.6* rule models)
    safety_identifier::Union{String,Nothing} = nothing
    conversation::Union{Any,Nothing} = nothing              # String or Dict (not with previous_response_id); the declared type collapses to Any
    context_management::Union{Vector,Nothing} = nothing
    stream_options::Union{AbstractDict,Nothing} = nothing
    prompt_cache_options::Union{PromptCacheOptions,Nothing} = nothing
    moderation::Union{ModerationConfig,Nothing} = nothing
    seed::Union{Int64,Nothing} = nothing                    # Gemini Interactions only: generation_config.seed
end
```

`seed` is sent only by Gemini Interactions, as an int32 (outside that range it throws);
OpenAI's Responses API has none, so `OPENAIServiceEndpoint`, every other OpenAI-wire endpoint
(Azure, `GEMINIOpenAIServiceEndpoint`, generic) and `OllamaEndpoint` throw `ArgumentError` for
it before any request. `service_tier` on `GEMINIServiceEndpoint` is `"standard"`, `"flex"`,
`"priority"` or `"deferred"`. The positional constructor takes all 32 fields in declaration order;
build a `Respond` with keywords, which name each field.

!!! warning "Gemini Interactions rejects unmappable fields"
    With `service=GEMINIServiceEndpoint`, a set `Respond` field the Interactions
    wire has no equivalent for makes `respond` throw `ArgumentError` before any
    request rather than being silently dropped, so a request never goes out quietly ignoring what you
    asked for. That wire maps `model`, `input` (each `InputMessage`, or Dict with a
    `role`, becomes a `user_input` or `model_output` step, its content Interactions
    `text`, `image`, `audio` and `document` parts; a system or developer message, a
    `phase`, a web-URL or file-id media part and a cache breakpoint are rejected),
    `instructions`, `tools`, `tool_choice`, `temperature` and `top_p` (only for the
    models released before Gemini 3.6 Flash that honour them — `gemini-2.5-*`,
    `gemini-3-flash-preview`, `gemini-3.1-pro-preview`, `gemini-3.1-flash-lite`, Gemma 4
    and the earlier image models; Gemini ignores them on later models, so they throw, as on
    native Chat and `GEMINIOpenAIServiceEndpoint`), `max_output_tokens` (above 65,536 on
    `gemini-4-argon` only with `background=true`), `stream`, `text` (its format becomes
    the top-level `response_format`; `verbosity` is rejected), `store`,
    `previous_response_id`, `background`, `reasoning` (effort and automatic summaries
    only), `service_tier` (`"standard"`, `"flex"`, `"priority"` or `"deferred"`),
    `safety_identifier` (the request label `safety_identifier`) and `seed`
    (`generation_config.seed`). Every other field — `moderation`,
    `prompt_cache_options` (Gemini caches prompts implicitly; UniLM does not wrap its
    explicit caching), and the rest of the OpenAI-only options — is rejected; leave it
    unset, or send the request to an OpenAI Responses service. A `FunctionTool` with
    `async`, `allowed_callers`, `defer_loading`, or `output_schema` set is rejected the
    same way.

### Input Helpers

```julia
# Structured input messages
InputMessage(role="user", content="Hello")
InputMessage(role="user", content=[input_text("Describe:"), input_image("https://...")])
InputMessage(role="assistant", content="I'll check the logs first.", phase="commentary")
    # phase: "commentary" | "final_answer", assistant messages only (else ArgumentError); OpenAI asks
    # callers to resend it on replayed assistant messages; Gemini Interactions and Ollama refuse it

# Content part constructors
input_text(text::String; cache_breakpoint=false, cache_ttl=nothing)   # → Dict(:type=>"input_text", :text=>...); cache_breakpoint=true adds prompt_cache_breakpoint={"mode":"explicit"};
                                                                      #   cache_ttl "30m"|"12h" gives that breakpoint its own ttl (needs cache_breakpoint=true)
input_image(url=nothing; detail=nothing, file_id=nothing)             # → Dict(:type=>"input_image", ...); pass url OR file_id, detail: "auto","low","high","original"
input_file(; url=nothing, id=nothing, file_data=nothing, filename=nothing)  # → Dict(:type=>"input_file", ...); pass one of url / id / file_data (base64)
input_file(document::DocumentAttachment)                              # file_data = the PDF's data URL, filename = document.filename

# Input item: change reasoning effort mid-conversation (GPT-6) without rewriting the cached prefix
configuration_update(; effort)   # → Dict(:type=>"configuration_update", :reasoning=>Dict(:effort=>...)); effort: "none","minimal","low","medium","high","xhigh","max"
```

`input_image` and `input_file` throw `ArgumentError` when none of their
source arguments is given.

### Tool Types

```julia
abstract type ResponseTool end

@kwdef struct FunctionTool <: ResponseTool
    name::String
    description::Union{String,Nothing} = nothing
    parameters::Union{AbstractDict,Nothing} = nothing
    strict::Union{Bool,Nothing} = nothing
    async::Union{Bool,Nothing} = nothing                    # GPT-6 async tool calling
    allowed_callers::Union{Vector{String},Nothing} = nothing  # "direct" and/or "programmatic"; anything else throws
    defer_loading::Union{Bool,Nothing} = nothing            # loaded via tool search
    output_schema::Union{AbstractDict,Nothing} = nothing    # JSON Schema of the string output
end

@kwdef struct WebSearchTool <: ResponseTool
    type::String = "web_search"                             # GA; legacy "web_search_preview"
    search_context_size::String = "medium"                  # "low","medium","high"
    user_location::Union{AbstractDict,Nothing} = nothing
    filters::Union{AbstractDict,Nothing} = nothing          # "allowed_domains"/"blocked_domains" (GA only)
end

@kwdef struct FileSearchTool <: ResponseTool
    vector_store_ids::Vector{String}
    max_num_results::Union{Int,Nothing} = nothing
    ranking_options::Union{AbstractDict,Nothing} = nothing
    filters::Union{AbstractDict,Nothing} = nothing
end
```

```julia
# Reach a remote MCP server (server_url), an OpenAI connector (connector_id, e.g.
# "connector_googledrive"), or a Secure MCP Tunnel (tunnel_id) — hence no single
# required target field beyond server_label.
@kwdef struct MCPTool <: ResponseTool
    server_label::String
    server_url::Union{String, Nothing} = nothing
    connector_id::Union{String, Nothing} = nothing
    authorization::Union{String, Nothing} = nothing         # OAuth access token
    server_description::Union{String, Nothing} = nothing
    require_approval::Union{String, AbstractDict, Nothing} = "never"
    allowed_tools::Union{Vector{String}, AbstractDict, Nothing} = nothing
    headers::Union{AbstractDict, Nothing} = nothing
    tunnel_id::Union{String, Nothing} = nothing
end

# action, moderation, partial_images, and input_fidelity are validated at construction
# (ArgumentError); quality, background, and output_format pass through unchecked
# (the accepted quality values depend on the model).
@kwdef struct ImageGenerationTool <: ResponseTool
    background::Union{String, Nothing} = nothing
    output_format::Union{String, Nothing} = nothing
    output_compression::Union{Int, Nothing} = nothing
    quality::Union{String, Nothing} = nothing               # "low","medium","high","auto"; 2.5 models add "xhigh","max"
    size::Union{String, Nothing} = nothing
    model::Union{String, Nothing} = nothing                 # e.g. "gpt-image-2.5-flare"
    action::Union{String, Nothing} = nothing                # "generate","edit","auto"
    moderation::Union{String, Nothing} = nothing            # "auto","low"
    partial_images::Union{Int, Nothing} = nothing           # 0–3
    input_fidelity::Union{String, Nothing} = nothing        # "high","low"
    input_image_mask::Union{AbstractDict, Nothing} = nothing  # "file_id" or "image_url"
end

@kwdef struct CodeInterpreterTool <: ResponseTool
    container::Union{AbstractDict, Nothing} = nothing
    file_ids::Union{Vector{String}, Nothing} = nothing
end
```

**Convenience constructors**:

```julia
function_tool(name, description=nothing; parameters=nothing, strict=nothing, async=nothing,
              allowed_callers=nothing, defer_loading=nothing, output_schema=nothing)
function_tool(d::AbstractDict)           # from a dict with "name" and optionally "description", "parameters", "strict",
                                         # "async", "allowed_callers", "defer_loading", "output_schema"
web_search(; context_size="medium", location=nothing, type="web_search", filters=nothing)
file_search(store_ids; max_results=nothing, ranking=nothing, filters=nothing)
mcp_tool(label, url=nothing; require_approval="never", allowed_tools=nothing, headers=nothing,
         connector_id=nothing, authorization=nothing, server_description=nothing, tunnel_id=nothing)
image_generation_tool(; kwargs...)
code_interpreter(; container=nothing, file_ids=nothing)
```

### Text Format / Structured Output

```julia
@kwdef struct TextFormatSpec
    type::String = "text"                                   # "text","json_object","json_schema"
    name::Union{String,Nothing} = nothing
    description::Union{String,Nothing} = nothing
    schema::Union{AbstractDict,Nothing} = nothing
    strict::Union{Bool,Nothing} = nothing
end

@kwdef struct TextConfig
    format::TextFormatSpec = TextFormatSpec()
    verbosity::Union{String,Nothing} = nothing              # "low","medium","high" (gpt-5.x)
end
```

**Convenience constructors**:

```julia
text_format(; verbosity=nothing, kwargs...)                  # kwargs go to TextFormatSpec; verbosity to TextConfig
json_schema_format(name, description, schema; strict=nothing) # JSON Schema output
json_schema_format(d::AbstractDict)                          # from dict with keys "name", "description", "schema"
json_object_format()                                         # unstructured JSON
```

### Reasoning controls

```julia
@kwdef struct Reasoning
    effort::Union{String,Nothing} = nothing                 # "none","minimal","low","medium","high","xhigh","max" (per-model subset)
    generate_summary::Union{String,Nothing} = nothing       # deprecated alias, serialized as summary
    summary::Union{String,Nothing} = nothing                # "auto","concise","detailed"
    context::Union{String,Nothing} = nothing                # "auto","current_turn","all_turns"
    mode::Union{String,Nothing} = nothing                   # "standard","pro"
end
```

```julia
Respond(input="Hard math problem", model="gpt-5.4-mini", reasoning=Reasoning(effort="high"))
```

OpenAI GPT-5.6 and later (on `Respond` and `Chat`) and native Anthropic (on `Chat`) take typed prompt-cache options:

```julia
@kwdef struct PromptCacheOptions
    mode::Union{String,Nothing} = nothing                    # "implicit" or "explicit"
    ttl::Union{String,Nothing} = nothing                     # "5m" or "1h" (Anthropic); "30m" or "12h" (OpenAI)
    prewarm::Union{Bool,Nothing} = nothing                   # Responses only: prepare the cache, no output
    comparison_response_id::Union{String,Nothing} = nothing  # Responses only: request cache diagnostics
end
Respond(input="Hello", model="gpt-5.6-luna",
        reasoning=Reasoning(effort="low", context="current_turn", mode="standard"),
        prompt_cache_options=PromptCacheOptions(mode="explicit", ttl="30m"))
```

Another `ttl` throws `ArgumentError` at construction, and each endpoint takes its own subset,
refusing the rest with an `ArgumentError` naming it, before any request: native OpenAI takes
`"30m"` on Chat Completions (which answers any other ttl with HTTP 400) and `"30m"` or `"12h"`
on Responses and on its Decisions API; on `ANTHROPICServiceEndpoint` the options become a
top-level `cache_control` (automatic caching, the breakpoint on the last cacheable block) with
`ttl` `"5m"` or `"1h"` and `mode` `"implicit"` or unset — `"explicit"`, `prewarm=true`,
`comparison_response_id` and the OpenAI ttls throw; native Gemini (Chat and Interactions) and
`OllamaEndpoint` refuse `prompt_cache_options` (Gemini caches prompts implicitly; UniLM does not
wrap its explicit caching); Azure, `GEMINIOpenAIServiceEndpoint`, DeepSeek and generic
endpoints send the object unchanged, the server's answer being the check.

On OpenAI, `prompt_cache_retention` throws `ArgumentError` for `gpt-6-astra`, `gpt-6.1-sol`,
`gpt-6-sol`, `gpt-6-luna`, `gpt-5.6` and `gpt-5.6-sol`/`-terra`/`-luna` (and their dated
snapshots), whose cache lifetime `prompt_cache_options` sets; on other models it is sent and
may be combined with `prompt_cache_options`; on `gpt-5.5` and `gpt-5.5-pro` it takes only
`"24h"` (OpenAI's reference; the server checks it). Explicit mode requires cache breakpoints in input
content to create cache writes — mark them with `input_text(text; cache_breakpoint=true)`, a
Responses input part; `Chat` messages cannot carry one, so explicit mode on a `Chat` means no
prompt caching. Chat Completions takes only `mode` and `ttl` (a native OpenAI `Chat` with
`prewarm` or `comparison_response_id` set throws `ArgumentError`). Diagnostics requested with
`comparison_response_id` come back in `r.response.raw["prompt_cache_diagnostics"]`.
Unset fields are omitted.
[OpenAI prompt caching](https://developers.openai.com/api/docs/guides/prompt-caching),
[Anthropic prompt caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching).

### Moderation

```julia
@kwdef struct ModerationConfig
    model::String                                 # required, e.g. "omni-moderation-latest"
    input_mode::Union{String,Nothing} = nothing   # "score" or "block"
    output_mode::Union{String,Nothing} = nothing  # "score" or "block"
end
Respond(input="Hello", moderation=ModerationConfig(model="omni-moderation-latest"))
Chat(moderation=ModerationConfig(model="omni-moderation-latest", input_mode="block"))
```

The `moderation` field on `Respond` and `Chat` runs a moderation model over the
request input and the generated output. `model` is required, as in both the
Responses and the Chat Completions API references: `ModerationConfig()` without it
throws `UndefKeywordError`, and a mode other than `"score"`/`"block"` throws
`ArgumentError`. It serializes as
`{"model": …, "policy": {"input": {"mode": …}, "output": {"mode": …}}}` with unset
policy parts omitted. A `respond` result carries the outcome in
`r.response.raw["moderation"]` (`"input"` and `"output"` moderation results);
`LLMSuccess` keeps no raw body, so Chat results do not expose it.

### respond

```julia
# Struct form
respond(r::Respond; config=nothing, callback=nothing, cancel=nothing)
    -> ResponseSuccess | ResponseFailure | ResponseCallError | Task

# Convenience — builds Respond internally
respond(input; kwargs...) -> same

# do-block streaming — auto-sets stream=true
respond(callback::Function, input; kwargs...) -> Task
```

- Streaming callback signature: `callback(chunk::Union{String, ResponseObject}, close::Ref{Bool})`; `close[] = true` (or a `cancel!` on the call's token) stops the stream with `ResponseCallError(status=nothing, cause=UniLMCancelled(...))`, unless the provider's completion marker — the terminal event — already arrived: then the outcome it recorded stands (a response with its usage, whose final callback has run; or the reported failure). A callback exception ends the call with it in `.cause`.
- Local validation throws `ArgumentError` before any network I/O: a `callback` passed without `stream=true` ("callback requires stream=true"), a service that declares capabilities with neither `:responses` nor `:agentic`, or an encoder rejection (a field the wire or model cannot express).
- Retries retryable statuses (408/429/500/502–504/520–524/529) up to `config.max_attempts` (default 3) with full-jitter backoff bounded by `config.total_deadline`; `Retry-After` is a floor under the jitter, and a retry whose wait would exceed the remaining deadline is not attempted — it returns the last real response instead of sleeping past it; a 429 whose `error.code` reports spent credit or a reached spend or usage limit (the four codes under `chatrequest!`) is returned at once, streamed or not. Every attempt is time-bounded — a silent peer fails with a typed timeout inside `ResponseCallError` (`status = nothing`, `cause::UniLMTimeout`), never a hang. Every exception that ends a call (including a cancellation, `cause::UniLMCancelled`) is in `.cause`.
- **Parameter validation**: `temperature` ∈ [0.0, 2.0], `top_p` ∈ [0.0, 1.0], `max_output_tokens` ≥ 1, `top_logprobs` ∈ [0, 20]; `conversation` cannot be combined with `previous_response_id`, and `background=true` requires `store` not `false`; `Reasoning(effort=…)` must be one of `none`, `minimal`, `low`, `medium`, `high`, `xhigh`, `max`. Violations throw `ArgumentError`.

### Response Accessors

```julia
output_text(result::ResponseSuccess)::String                # the answer: message text without phase "commentary" ("\n" between parts);
                                                            #   a Gemini answer's consecutive model_output steps are one part, joined with no separator
output_text(result::ResponseFailure)                        # throws LLMResultError
output_text(result::ResponseCallError)                      # throws LLMResultError

function_calls(result::ResponseSuccess)::Vector{Dict{String,Any}}   # empty on a failure result
# Each dict has: "id", "call_id", "name", "arguments" (JSON string), "status"
```

### Response Management Functions

```julia
get_response(id::String; service=OPENAIServiceEndpoint, config=nothing)    -> ResponseSuccess | ResponseFailure | ResponseCallError
delete_response(id::String; service=OPENAIServiceEndpoint, config=nothing) -> Dict | ResponseFailure | ResponseCallError
list_input_items(id::String; limit=20, order="desc", after=nothing, service=OPENAIServiceEndpoint, config=nothing) -> Dict | ...
cancel_response(id::String; service=OPENAIServiceEndpoint, config=nothing) -> ResponseSuccess | ...
compact_response(; model=nothing, input, service=OPENAIServiceEndpoint, config=nothing) -> Dict | ...
    # model nothing → "gpt-6-sol" (OpenAI's default), except on an OllamaEndpoint (no default model: ArgumentError before I/O)
count_input_tokens(; model="gpt-6-sol", input, instructions=nothing, tools=nothing, service=OPENAIServiceEndpoint, config=nothing) -> Dict | ...
```

Each makes a single bounded attempt (no retries) and observes the ambient
[`with_cancel`](@ref) token; every call error carries its exception in `.cause`.

### ResponseObject

```julia
@kwdef struct ResponseObject
    id::String
    status::String
    model::String              # through an OllamaEndpoint, a cloud model whose reply drops its tag keeps the requested name (the reply's in raw)
    output::Vector{Any}
    usage::Union{Dict{String,Any},Nothing} = nothing
    error::Union{Dict{String,Any}, String, Nothing} = nothing   # provider error object, or a plain message (Gemini Interactions: the first of its errors, all in raw)
    metadata::Union{Dict{String,Any},Nothing} = nothing
    raw::Dict{String,Any}
end

@kwdef struct ResponseSuccess <: LLMRequestResponse
    response::ResponseObject
    sse_dropped::Int = 0                 # undecodable SSE data: payloads dropped while assembling a streamed turn
    service::Union{Nothing,ServiceEndpointSpec} = nothing   # the endpoint that answered (prices the call); nothing when built by hand
end
```

A generation whose terminal status is `"failed"` is reported as a
`ResponseFailure`, not a success — on both agentic wires, streamed and
non-streamed alike. The failure carries the wire body verbatim, so the response's
own `error` and `metadata` are preserved on `.response`. Only `"failed"` flips:
`incomplete`, `cancelled`, `expired`, `in_progress`, `queued` and
`requires_action` are legitimate terminals of the capped, background and
tool-action flows and stay `ResponseSuccess`, inspectable via
[`response_status`](@ref) and [`incomplete_details`](@ref). An `incomplete`
generation — one cut short by a token or output cap — therefore arrives as a
`ResponseSuccess` carrying its partial output, `.response.status ==
"incomplete"`, and the wire's `incomplete_details` verbatim. A terminal
`response.incomplete` event decodes exactly as the non-streamed body does, so
`issuccess` reflects what happened rather than how the call was made.

### Responses API Examples

```julia
using UniLM, JSON

# Basic
result = respond("Tell me a joke")
if result isa ResponseSuccess
    println(output_text(result))
end

# With instructions
result = respond("Hello", instructions="You are a pirate. Respond in pirate speak.")

# Multi-turn via chaining
r1 = respond("Tell me a joke")
r2 = respond("Tell me another", previous_response_id=r1.response.id)

# Structured output
schema = Dict(
    "type" => "object",
    "properties" => Dict(
        "name" => Dict("type" => "string"),
        "age" => Dict("type" => "integer")
    ),
    "required" => ["name", "age"],
    "additionalProperties" => false
)
result = respond("Extract: John is 30 years old",
    text=json_schema_format("person", "A person", schema, strict=true))
parsed = JSON.parse(output_text(result))

# Web search
result = respond("Latest Julia language news", tools=[web_search()])

# Function calling
tool = function_tool("get_weather", "Get weather",
    parameters=Dict(
        "type" => "object",
        "properties" => Dict("location" => Dict("type" => "string")),
        "required" => ["location"]
    ))
result = respond("Weather in NYC?", tools=ResponseTool[tool])
for call in function_calls(result)
    println(call["name"], ": ", call["arguments"])
end

# Streaming (do-block)
respond("Tell me a story") do chunk, close_ref
    if chunk isa String
        print(chunk)
    elseif chunk isa ResponseObject
        println("\nDone: ", chunk.status)
    end
end

# Reasoning
result = respond("Prove that √2 is irrational", model="gpt-5.4-mini",
    reasoning=Reasoning(effort="high", summary="concise"))

# Multimodal input
result = respond([
    InputMessage(role="user", content=[
        input_text("What's in this image?"),
        input_image("https://example.com/photo.jpg")
    ])
])

# Count tokens without generating
tokens = count_input_tokens(model="gpt-5.2", input="Hello world")
println(tokens["input_tokens"])
```

---

## Image Generation API

### ImageGeneration

```julia
@kwdef struct ImageGeneration
    service::ServiceEndpointSpec = OPENAIServiceEndpoint      # OPENAIServiceEndpoint, or GEMINIServiceEndpoint (Nano Banana)
    model::String = ""                                      # sentinel — see below
    prompt::String
    n::Union{Int,Nothing} = nothing                         # 1–10 on OpenAI; Gemini takes only 1
    size::Union{String,Nothing} = nothing                   # OpenAI: "1024x1024","1536x1024","1024x1536","auto"; gpt-image-2+: any WIDTHxHEIGHT, multiples of 16, up to 3840x2160
    quality::Union{String,Nothing} = nothing                # OpenAI: "low","medium","high","auto"; gpt-image-2.5-flare/-sunburst add "xhigh","max"
    background::Union{String,Nothing} = nothing             # OpenAI: "transparent","opaque","auto"
    output_format::Union{String,Nothing} = nothing          # OpenAI: "png","webp","jpeg"
    output_compression::Union{Int,Nothing} = nothing        # OpenAI: 0–100 (webp/jpeg only)
    safety_identifier::Union{String,Nothing} = nothing      # both: OpenAI 1–128 chars; Gemini the request label (≤ 63 of a-z 0-9 _ -)
    moderation::Union{String,Nothing} = nothing             # OpenAI: "auto","low"
    aspect_ratio::Union{String,Nothing} = nothing           # Gemini: "1:1","1:4","4:1","1:8","8:1","2:3","3:2","3:4","4:3","4:5","5:4","9:16","16:9","21:9"
    resolution::Union{String,Nothing} = nothing             # Gemini: "512","1K","2K","4K"; gemini-nano-banana-2.1 renders 1K, 2K or 4K
end

@kwdef struct ImageEdit
    service::ServiceEndpointSpec = OPENAIServiceEndpoint
    image::Union{String,Vector{String}}                     # input file path(s); an empty vector throws in edit_image
    prompt::String
    mask::Union{String,Nothing} = nothing                   # OpenAI: transparent area = what to edit
    model::String = ""                                      # sentinel, as on ImageGeneration
    n::Union{Int,Nothing} = nothing
    size::Union{String,Nothing} = nothing                   # OpenAI
    quality::Union{String,Nothing} = nothing                # OpenAI
    input_fidelity::Union{String,Nothing} = nothing         # OpenAI: "high","low"
    background::Union{String,Nothing} = nothing             # OpenAI
    output_format::Union{String,Nothing} = nothing          # OpenAI
    safety_identifier::Union{String,Nothing} = nothing      # both, as on ImageGeneration (an OpenAI form field)
    aspect_ratio::Union{String,Nothing} = nothing           # Gemini; unset, the edit keeps the input's ratio
    resolution::Union{String,Nothing} = nothing             # Gemini
end
```

`input_fidelity` is an edit-only parameter: it lives on `ImageEdit`, and
`ImageGeneration(...; input_fidelity=...)` is a `MethodError`. A field the request's
provider does not take (an OpenAI-only field on Gemini; `aspect_ratio` or `resolution` on
OpenAI), `n` other than 1 on Gemini, a Gemini `aspect_ratio` or `resolution` outside its
list, and a `safety_identifier` outside its provider's rule (empty or over 128 characters on
OpenAI) throw `ArgumentError` naming the provider and the field — when the request is
serialized, and in `generate_image` / `edit_image` before any request; other OpenAI values
are sent as given, for OpenAI to judge.

Unlike `Chat` and `Respond`, `ImageGeneration` does **not** resolve its model at
construction: the field stays `""` and resolves at serialization time, to
`"gpt-image-2.5-sunburst"` for OpenAI or `"gemini-nano-banana-2.1"` for Gemini. So
`ImageGeneration(prompt="…").model` reads back as the empty string — pass `model=`
explicitly if you need to read it, or inspect `JSON.json(ig)` to see what will go on the
wire (Gemini names the model in the URL instead). A service with no default image model
throws `ArgumentError` at serialization.

### generate_image

```julia
generate_image(ig::ImageGeneration; config=nothing, cancel=nothing) -> ImageSuccess | ImageFailure | ImageCallError
generate_image(prompt::String; config=nothing, cancel=nothing, kwargs...) -> same   # convenience
edit_image(e::ImageEdit; config=nothing, cancel=nothing)                 -> same
edit_image(image, prompt; mask=nothing, service=OPENAIServiceEndpoint, config=nothing, cancel=nothing, kwargs...)
```

Retries transient statuses (408/429/500/502–504/520–524/529) inside the request budget (`RequestConfig.max_attempts`, default 3; `Retry-After` is a floor under the jitter) — a retry whose wait would exceed the remaining `total_deadline` is not attempted, so the call returns the last real response rather than sleeping past it. Bound per call with `config=RequestConfig(...)`; a cancelled call returns `ImageCallError` with `cause::UniLMCancelled`.

**Gemini (Nano Banana)**: with `service=GEMINIServiceEndpoint` both verbs call `generateContent` (`GEMINI_API_KEY`) with `responseModalities: ["IMAGE"]` and an `imageConfig` from `aspect_ratio` / `resolution`, a `safety_identifier` as the request label `safety_identifier`; an edit sends its input image(s) inline ahead of the prompt (no mask). The model's image parts are the result's images, in order — JPEG as observed (each part's `mimeType` is in `raw`); text it sends with an image is the `revised_prompt`, interim thinking images are not results; a reply with no image is an `ImageCallError` naming Gemini's finish reason (`IMAGE_SAFETY`, `NO_IMAGE`, …); `created` is the reply's arrival time (Gemini sends none); `usage` is the reply's `usageMetadata`, read by `token_usage` (thinking counted as output). `estimated_cost` of an image result throws `ArgumentError` unless `model=` names a row in `pricing`.

### Response Types

```julia
@kwdef struct ImageObject
    b64_json::Union{String,Nothing} = nothing
    revised_prompt::Union{String,Nothing} = nothing
    url::Union{String,Nothing} = nothing          # set when the image was delivered by URL
end

@kwdef struct ImageResponse
    created::Int64
    data::Vector{ImageObject}
    usage::Union{Dict{String,Any},Nothing} = nothing
    raw::Dict{String,Any}
end
```

### Accessors

```julia
image_data(result::ImageSuccess)::Vector{String}       # per image: its base64 data, else its URL
image_data(result::ImageFailure)                        # throws LLMResultError
image_data(result::ImageCallError)                      # throws LLMResultError
save_image(img_b64::String, filepath::String)           # decode + write to disk, returns filepath
```

### Image Generation Example

```julia
using UniLM

result = generate_image("A watercolor painting of a Julia butterfly",
    size="1024x1024", quality="high")

if result isa ImageSuccess
    imgs = image_data(result)
    save_image(imgs[1], "butterfly.png")
    println("Saved! Revised prompt: ", result.response.data[1].revised_prompt)
end

# Multiple images with transparent background
result = generate_image("Minimalist logo",
    n=3, background="transparent", output_format="png")
```

---

## Embeddings API

### Embeddings

```julia
struct Embeddings
    service::ServiceEndpointSpec     # default: OPENAIServiceEndpoint
    model::String                    # default resolved per provider at construction
    input::Union{String,Vector{String},EmbeddingInput,Vector{EmbeddingInput}}
    embeddings::Union{Vector{Float64},Vector{Vector{Float64}}}   # pre-allocated, filled in place
    user::Union{String,Nothing}
    dimensions::Union{Int,Nothing}          # requested output dimensionality; must be positive
    encoding_format::Union{String,Nothing}  # "float" or nothing; anything else throws ArgumentError
    task::Union{Symbol,Nothing}             # what the texts are for: a task prompt (below); nothing = none
end

Embeddings(input::Union{AbstractString,EmbeddingInput,Attachment}; service=OPENAIServiceEndpoint, model="",
           dimensions=nothing, encoding_format=nothing, user=nothing, task=nothing)
Embeddings(input::AbstractVector; service=OPENAIServiceEndpoint, model="",
           dimensions=nothing, encoding_format=nothing, user=nothing, task=nothing)
    # a vector of strings is copied into a Vector{String}; any other item (an EmbeddingInput,
    # ImageAttachment, AudioAttachment) makes every item an EmbeddingInput ("" then refused);
    # a DocumentAttachment, or an image with a detail level, is an ArgumentError

struct EmbeddingInput                       # one item, embedded as one vector: a text, an image, a sound clip, or a
    text::Union{String,Nothing}             #   text with an image or a clip (one of them per item on Ollama); "" beside
                                            #   media = no text
    title::Union{String,Nothing}            # the document's title: task = :document of EmbeddingGemma and gemini-embedding-2 only; "" or
                                            #   whitespace = no title (sent as "title: none | text: …")
    image::Union{ImageAttachment,Nothing}
    audio::Union{AudioAttachment,Nothing}
end
EmbeddingInput(text; title=nothing)
EmbeddingInput(image::ImageAttachment); EmbeddingInput(audio::AudioAttachment)
EmbeddingInput(text, media::Attachment...)  # at most one image and one clip; an item holding both is refused
                                            #   for an OllamaEndpoint before I/O (Ollama orders an item's media at random)
EmbeddingInput(; text=nothing, title=nothing, image=nothing, audio=nothing)
    # ArgumentError: nothing to embed, a title without a text, two images or clips, an image with a detail
    # level, another attachment kind (a DocumentAttachment: no embedding endpoint UniLM sends to takes one);
    # == and hash compare the fields (media by content)
```

Model defaults: `"text-embedding-3-small"` for OpenAI, `"gemini-embedding-2"` for Gemini (OpenAI-compat shim). For generic/DeepSeek endpoints, model must be specified explicitly or the constructor throws `ArgumentError` naming the service. An empty vector input also throws. A reply row without an index beside numbered ones is row 0 (Gemini's compatible endpoint leaves the index out of its first row), a reply numbering none is read in order, and a second row 0 is an `EmbeddingCallError`.

**Task prompts** (checked at construction, `ArgumentError`, no I/O): `task` ∈ `:search_query`, `:question_answering`, `:fact_checking`, `:code_retrieval`, `:document`, `:classification`, `:clustering`, `:sentence_similarity`, `:none`. An EmbeddingGemma model (name contains `embeddinggemma`, case-insensitive; v1 and v2, any service) requires a task and gets the model card's prompt in front of each text: `"task: search result | query: "`, `"task: question answering | query: "`, `"task: fact checking | query: "`, `"task: code retrieval | query: "`, `"task: classification | query: "`, `"task: clustering | query: "`, `"task: sentence similarity | query: "`, and for `:document` `"title: " * something(title, "none") * " | text: "`; `:none` sends the text as is. A title is refused with any other task; EmbeddingGemma's `dimensions` ∈ (768, 512, 256, 128). `gemini-embedding-2` (exact id) gets the same prompts but also takes no task (its text then sent as is; Google's embeddings guide recommends the prompts for text), with no `dimensions` check. Any other model takes `task = :none` or `nothing` (another task, or a title, is refused). Media items need an `OllamaEndpoint` (`/api/embed`): every other service's wire is text-only and throws before I/O; media take no prompt, the text of a mixed item does (Ollama 0.40.6, Apple M4 Max, 2026-10-21, `embeddinggemma-2:740m-q8_0` and `:740m-mxfp8`: a text with its picture or clip as one item gave a vector distinct from each part's, cosine 0.845–0.958 with each); the 440m (text and images) and 570m (text and audio) builds are untested.

The `embeddings` buffer is pre-allocated to `something(dimensions, 1536)` zeros per input, filled in place, and resized to the length the model returns (a 3072-dimension model needs no adjustment). Because a pre-zeroed slot would be indistinguishable from a real vector, a response that does not cover every input exactly once — a missing or duplicated row index — is reported as an `EmbeddingCallError` rather than left as zeros, and so is a row whose vector is empty, all zeros or holds a value that is not a finite number (a JSON `true` included), a row without a vector, and a row index that is not an integer; every row is checked before any is stored, for every provider. The result aliases the request (`result.embeddings === emb`), so use one `Embeddings` per concurrent call.

### embeddingrequest!

```julia
embeddingrequest!(emb::Embeddings; config=nothing, cancel=nothing) -> EmbeddingSuccess | EmbeddingFailure | EmbeddingCallError
```

Returns an `EmbeddingSuccess`/`EmbeddingFailure`/`EmbeddingCallError` (a `<: LLMRequestResponse`). Fills `emb.embeddings` in-place; `embedding_vectors(result)` returns the vectors (and throws `LLMResultError` on a failure). Retries transient statuses (408/429/500/502–504/520–524/529) with backoff and jitter under the resolved [`RequestConfig`](@ref) (`max_attempts`, `total_deadline`; `Retry-After` is a floor under the jitter), but a retry whose wait would exceed the remaining deadline is not attempted — it returns the last real response rather than sleeping past it. Timeouts surface as `EmbeddingCallError` with `status=nothing` and the `UniLMTimeout` in `.cause`; a cancelled call carries `UniLMCancelled` there. A service that declares capabilities without `:embeddings` throws `ArgumentError` before any request.

### Embeddings Example

```julia
using UniLM, LinearAlgebra

emb = Embeddings("What is Julia?")
embeddingrequest!(emb)
println(emb.embeddings[1:5])  # first 5 dimensions

# Batch + cosine similarity
emb = Embeddings(["cat", "dog", "airplane"])
embeddingrequest!(emb)
similarity = dot(emb.embeddings[1], emb.embeddings[2]) /
    (norm(emb.embeddings[1]) * norm(emb.embeddings[2]))
```

---

## Cost Tracking

### TokenUsage

```julia
@kwdef struct TokenUsage
    prompt_tokens::Int = 0                          # every input token, cache reads and writes included
    completion_tokens::Int = 0
    total_tokens::Int = 0
    cached_tokens::Int = 0                          # subset of prompt_tokens read from the prompt cache
    reasoning_tokens::Int = 0                       # subset of completion_tokens spent on hidden reasoning
    cache_write_tokens::Int = 0                     # subset of prompt_tokens written to the cache
    cache_write_extended_tokens::Int = 0            # subset of cache_write_tokens at the extended retention (OpenAI "12h", Anthropic "1h")
    service_tier::Union{String,Nothing} = nothing   # the tier the provider reports billing the call at
    inference_geo::Union{String,Nothing} = nothing  # the inference region it reports
    prompt_audio_tokens::Int = 0                    # subset of prompt_tokens that is audio input
    completion_audio_tokens::Int = 0                # subset of completion_tokens that is audio output
end
TokenUsage(prompt_tokens, completion_tokens, total_tokens, cached_tokens, reasoning_tokens)
    # the 5-count positional form still builds; positionally the fields go in the order above
```

The counts come from the providers' usage objects and default to `0` when the provider
omits them. `cached_tokens` is load-bearing for cost: it is billed at the discounted
cached-input rate. OpenAI (Chat streamed and not, Responses, Decisions) and Anthropic report
cache writes, and the extended-retention share of them (OpenAI's 12-hour, Anthropic's 1-hour); `service_tier` comes from OpenAI's
`service_tier`, Anthropic's (`"fast"` for a request served in fast mode) and Gemini's
(`usageMetadata.serviceTier`, else the `x-gemini-service-tier` header, or an interaction's
`service_tier`); `inference_geo` from Anthropic. The audio counts come from OpenAI Chat
Completions' `prompt_tokens_details.audio_tokens` / `completion_tokens_details.audio_tokens`,
streamed and not: `estimated_cost` bills them at the model's audio-token rates on
`OPENAIServiceEndpoint` (`gpt-audio-1.5`: \$32 / \$64 per 1M); audio tokens of any other
model there, or of any model on an endpoint billed at the standard rates (Azure, DeepSeek,
generic, custom), give `0.0` with one warning per model.

### Functions

```julia
token_usage(result::LLMRequestResponse)::TokenUsage   # never Nothing — zero usage for a failure
estimated_cost(result; model=nothing, pricing=DEFAULT_PRICING)   # per-call cost estimate (Float64)
cumulative_cost(chat::Chat)::Float64                             # running total for a Chat instance

DEFAULT_PRICING   # lock-guarded AbstractDict{String, PriceRow} (not a Dict)
                  # where PriceRow = @NamedTuple{input::Float64, cached_input::Float64, output::Float64}
```

- **`token_usage` returns a `TokenUsage`, never `nothing`**, for every result of a
  token-billed API — chat, Responses, embeddings, image generation, FIM and System
  One. Failures of those APIs report all-zero usage.
- **Both functions throw `ArgumentError` for result types outside the token-billed
  APIs** — audio, batch, container, conversation, file, fine-tuning, moderation,
  upload, vector-store, realtime. Those calls carry no token usage at all,
  and a `0.0` would be indistinguishable from a genuinely free call.
- The model priced is `model=` when given; else the one the provider reports having
  served (`LLMSuccess.model`; `response.model` for Responses, FIM and System One); else the
  one requested (`self.model` for Chat, `embeddings.model` for Embeddings) — so a routed id
  or an alias bills at the rates of the model that served it. An image result reports its
  usage, but `estimated_cost` throws `ArgumentError` for it unless `model=` names a row in
  `pricing`: UniLM does not model image-token rates, and a zero would read as a free call.
- **`DEFAULT_PRICING` is safe to share across tasks**: add or replace a row with
  `DEFAULT_PRICING[model] = row` while requests run; `get`, `haskey`, `delete!`,
  `pop!`, `empty!`, `keys` and `length` work as on a `Dict`, iteration walks a
  snapshot, and `copy` / `merge` / `filter` return a plain `Dict`. `pricing=` accepts
  any `AbstractDict{String, PriceRow}`.
- **`DEFAULT_PRICING` values are USD *per token*, not per 1M tokens** — provider
  list prices divided by `1_000_000` (e.g. `"gpt-6-sol"` is
  `(input = 2.0e-6, cached_input = 2.0e-7, output = 1.0e-5)`). A `pricing=`
  dict you supply must use the same per-token convention, or your estimate is
  off by a factor of a million.
- Lookup: the exact id, then the id without a dated snapshot suffix (OpenAI
  `-YYYY-MM-DD`, Anthropic `-YYYYMMDD`), and a versioned `jev-X.Y.Z` id without its
  own row is priced at the `jev-latest` row. Other unpriced models return `0.0` and
  log one warning per model id — pass `pricing=` (or add a row) to price custom
  models. The formula bills
  `min(cached_tokens, prompt_tokens)` at `cached_input`, the prompt tokens other than
  cache reads (cache writes included) at `input`, and `completion_tokens` at `output`
  (reasoning tokens are already counted within completion tokens); surcharges apply where
  UniLM models a provider's (below). A `SystemOneSuccess` is priced by the
  backend that answered (`response.service`): one from OpenAI's Decisions API
  bills every input token, cached and cache-written ones included, at the row's
  `input` rate (twice that above 272K input tokens), whatever tier a reply names, and no
  output; one from a local model on an
  `OllamaEndpoint` costs `0.0` unless `model=` is passed, as do its chat,
  embeddings, Responses and FIM results (a cloud model reached through Ollama, a
  tag of `cloud` or ending in `-cloud`, is priced by the requested name).

The OpenAI, Anthropic and Gemini rows were verified October 22, 2026, the TypeSafe rows
September 22, 2026 and the DeepSeek rows September 24, 2026 (its peak rates; off-peak hours
bill half). The rows are standard short-context text rates; surcharges apply only where
UniLM models the provider's, and multimodal rates (except OpenAI's audio-token rates) and
hosted-tool fees are not included. OpenAI estimates bill cache writes at 1.25× input
(`"30m"`) or 2.5× (`"12h"`, `cache_write_extended_tokens`) on GPT-6 and GPT-5.6, a prompt
over 272K input tokens at 2× the input and cache rates and 1.5× the output rate, and the
billed tier's multiple of every rate (Flex 0.5×; Fast 2× on GPT-6, GPT-5.6, `gpt-5.4`,
`gpt-5.4-mini` and `gpt-5.2`, 2.5× on `gpt-5.5`, 1.75× on GPT-4.1 and `o3`; Ultrafast 6× on
Astra and 6.1 Sol; a reported `"priority"` is Fast); a tier the pricing page gives no price
for is estimated at the standard rates with one warning per model and tier. Anthropic
`prompt_tokens` include cache reads and writes (`cached_tokens` the reads,
`cache_write_tokens` the writes and `cache_write_extended_tokens` the 1-hour ones), and its
results bill cache writes at 1.25× (5-minute) or 2× (1-hour) input, every rate 2× in fast
mode (`service_tier == "fast"`), 1.1× for `inference_geo == "us"` and 5× for a Claude Haiku
5.5 prompt over 100,000 tokens, the multipliers stacking. DeepSeek cache hits
(`prompt_cache_hit_tokens`) are billed at the cached rate. A Gemini result is priced as the
model that served it (`result.model` from `modelVersion`, `response.model`), so routed ids
and `-latest` aliases bill at their targets' rates (`gemini-3.1-pro-preview`, which serves
`gemini-pro-latest`, at its rates for prompts up to 200K tokens), and a tool-use prompt
counts as prompt tokens. A Gemini result is billed at the tier it reports
(`usage.service_tier`: `usageMetadata.serviceTier`, else the `x-gemini-service-tier` header,
or an interaction's `service_tier`) — input and output at half the standard rates on Flex
and 1.8 times on Priority, a cache hit at the same multiple except where a model's table
lists another cache-hit price (Flex's standard rate on 3.1 Pro Preview, 3 Flash Preview and
the 2.5 models; \$0.02 on Flex and \$0.05 on Priority for 3.5 Flash-Lite's \$0.03),
`"deferred"` at the standard rates with a warning; a long generation Gemini sends in several
requests is one turn whose usage is the requests' summed, priced at the last reply's tier. A
streamed Chat turn accrues cost when its stream reports usage: OpenAI and DeepSeek streams
request it automatically, native Anthropic and Gemini report it, and other OpenAI-compatible
servers need `stream_options=Dict("include_usage" => true)`.
Gemini 4 Argon's introductory rates hold through November 30, 2026 and Gemini 3.8 and 3.6
Flash's through December 31, 2026; refresh pricing before estimating later calls.

### Cost Tracking Example

```julia
chat = Chat(model="gpt-5.2")
push!(chat, Message(Val(:system), "You are helpful."))
push!(chat, Message(Val(:user), "Hello!"))

result = chatrequest!(chat)
if result isa LLMSuccess
    usage = token_usage(result)
    cost = estimated_cost(result)
    println("Tokens: $(usage.total_tokens), Cost: \$$(round(cost; digits=6))")
    println("Cumulative: \$$(round(cumulative_cost(chat); digits=6))")
end
```

---

## Conversation Forking

```julia
fork(chat::Chat)::Chat          # deep-copy every field but `service` (shared); cumulative cost copied by value (fresh Ref)
fork(chat::Chat, n::Int)::Vector{Chat}  # create n independent forks
```

### Fork Example

```julia
chat = Chat(model="gpt-5.2")
push!(chat, Message(Val(:system), "You are a creative writer."))
push!(chat, Message(Val(:user), "Start a story about a robot."))
chatrequest!(chat)

# Fork into 3 independent continuations
forks = fork(chat, 3)
for (i, f) in enumerate(forks)
    push!(f, Message(Val(:user), "Continue the story with ending $i."))
    chatrequest!(f)
end
```

---

## Tool Loop

Automated tool dispatch for both APIs. Wraps a tool schema with a callable function.

### CallableTool

```julia
struct CallableTool{T}
    tool::T              # Tool or FunctionTool
    callable::Function   # (name::String, args::Dict{String,Any}) -> String (any other value is sent JSON-encoded)
end
```

### to_tool

```julia
to_tool(x)  # identity for Tool, FunctionTool, CallableTool; converts AbstractDict to Tool
```

### ToolCallOutcome / ToolLoopResult

```julia
# Per-call record
struct ToolCallOutcome
    tool_name::String
    arguments::Dict{String,Any}
    result::Union{FunctionCallResult,Nothing}
    success::Bool
    error::Union{String,Nothing}
end

# Loop result
struct ToolLoopResult
    response::LLMRequestResponse        # the last response received (on max_turns: the last tool-call turn)
    tool_calls::Vector{ToolCallOutcome}
    turns_used::Int
    completed::Bool                     # true only for a final text answer that finished normally
    llm_error::Union{String,Nothing}    # why it stopped, when not completed
end
```

### tool_loop! (Chat Completions)

```julia
tool_loop!(chat, dispatcher; max_turns=10, config=nothing, callback=nothing, on_tool_call=nothing,
           cancel=nothing, tool_concurrency=1) -> ToolLoopResult
tool_loop!(chat; tools::Vector{<:CallableTool}, kwargs...) -> ToolLoopResult   # kwargs as above
```

`callback` and `on_tool_call` are the streaming hooks from `chatrequest!`, applied
to every turn of the loop and passed through unchanged (so they need `stream=true`, else
`ArgumentError` before any request). Rules:

- `chat.history` must be `true` (else `ArgumentError` before any request); `max_turns < 1`
  and `tool_concurrency < 1` throw `ArgumentError`.
- Calls run only on a turn whose `finish_reason == "tool_calls"`. A turn with calls that
  finished otherwise (`"length"`, `"content_filter"`, …) runs none: `completed=false`,
  `llm_error` names the reason, and the unanswered assistant turn is removed from `chat`.
- A text turn completes the loop only when it finished with `"stop"` or no reason (or
  `"tool_calls"` with no calls); `"length"` gives `completed=false` with
  `llm_error = "Model output was truncated by the token limit"`, any other reason
  `completed=false` with an `llm_error` naming it.
- A dispatcher's `String` result is sent as is, any other value JSON-encoded; a throwing
  dispatcher sends `"Error: <message>"` and records a failed `ToolCallOutcome`; an
  `InterruptException`, a `ReplayMissError` or a `RecordingWriteError` propagates after
  the interrupted turn is removed from `chat`.
- On `max_turns` exhaustion `response` is the last real response, `completed=false`,
  `llm_error = "max turns (N) exhausted"`.
- `cancel` (default: the ambient token) scopes every turn and dispatch; a cancelled loop
  returns `completed=false` with an `LLMCallError` whose `cause` is `UniLMCancelled`
  (a turn cancelled between dispatches is removed from `chat`).
- `tool_concurrency = n > 1` runs up to `n` of a turn's calls at once on spawned tasks
  (thread-safe dispatcher required); results are sent in call order.

### tool_loop (Responses API)

```julia
tool_loop(r::Respond, dispatcher; max_turns=10, config=nothing, cancel=nothing, tool_concurrency=1) -> ToolLoopResult
tool_loop(r::Respond; max_turns=10, config=nothing, cancel=nothing, tool_concurrency=1) -> ToolLoopResult   # callables from r.tools
tool_loop(input, dispatcher; max_turns=10, config=nothing, cancel=nothing, tool_concurrency=1, respond_kwargs...)
tool_loop(input::String; tools, max_turns=10, config=nothing, cancel=nothing, tool_concurrency=1, respond_kwargs...)
```

Turns chain through `previous_response_id`, or through `conversation` alone when the
`Respond` sets it. Function calls run only on a `completed` or `requires_action` turn;
another status stops the loop (`completed=false`, `llm_error` naming the status and the
`incomplete_details` reason). Call `arguments` that are not a JSON object are answered
with `"Error: invalid arguments: …"` and the loop continues. A turn requesting a
client-side action the loop cannot run (`custom_tool_call`, `apply_patch_call`,
`computer_call`, a `shell_call` without a `shell_call_output` in the
same output, `mcp_approval_request`) stops it with `completed=false`, naming the type,
and runs none of the turn's calls. Result encoding,
`max_turns`, `cancel` and `tool_concurrency` behave as in `tool_loop!` (a cancelled loop's
`response` is a `ResponseCallError`).

---

## MCP Client

Native MCP client (JSON-RPC 2.0 over stdio or Streamable HTTP). Client and server
negotiate revisions 2025-11-25 (preferred), 2025-06-18 and 2025-03-26; the stateless
2026-07-28 revision is not supported yet, so a 2026-07-28-only peer cannot connect.

### Types

```julia
MCPSession            # live connection — manages transport + cached tools/resources/prompts
MCPToolInfo           # tool definition from tools/list
MCPToolResult         # typed tools/call result: content, structured, is_error, parts
MCPResourceInfo       # resource definition from resources/list
MCPPromptInfo         # prompt definition from prompts/list
MCPServerCapabilities # capabilities from initialize
MCPTransport          # abstract (subtypes: StdioTransport, HTTPTransport)
MCPError <: Exception # JSON-RPC error (code, message, data — any JSON value)
MCPCrashError <: Exception # stdio server process exited / pipe broke (exitcode, termsignal, cause)
MCPSessionClosedError <: Exception # call on a closed / not-connected session (cause :disconnected|:timeout|:crash, msg)
```

### Lifecycle

```julia
mcp_connect(command::Cmd; stderr=nothing, client_name="UniLM.jl", client_version=<package version>,
            protocol_version="2025-11-25", config=nothing, auto_respawn=false) -> MCPSession  # stdio subprocess
mcp_connect(url::String; headers=Pair{String,String}[], kwargs...) -> MCPSession            # Streamable HTTP
mcp_connect(transport::MCPTransport; kwargs...) -> MCPSession      # a custom transport must bound its own IO
mcp_connect(f::Function, args...; kwargs...)                       # do-block, auto-disconnect
mcp_disconnect!(session)
```

`stderr` is where a stdio server's stderr goes — an `IO` (e.g. `devnull`) or a file path
(appended to); `nothing` inherits this process's. A stdio server runs in its own process
group: disconnecting or a fatal timeout tears it down (stdin EOF, SIGTERM, then SIGKILL
of the group), the group is killed when its leader exits, and servers still running at
exit are torn down by an exit hook.

### Discovery

```julia
list_tools!(session; timeout=nothing) -> Vector{MCPToolInfo}
list_resources!(session) -> Vector{MCPResourceInfo}
list_prompts!(session) -> Vector{MCPPromptInfo}
```

When the client reads a `notifications/tools/list_changed`, `session.tools_stale` is set to `true` to flag the cached tool list as out of date; call `list_tools!(session)` to refresh it (which clears the flag). Server frames are read only during an exchange: a stdio notification sent between calls is seen on the next call, and over HTTP (no standing listener) only notifications inside a response body are seen.

### Operations

```julia
call_tool(session, name, arguments; timeout=nothing) -> MCPToolResult
read_resource(session, uri) -> String
get_prompt(session, name, arguments) -> Vector{Dict}
ping(session)
```

`call_tool` and `get_prompt` accept any `AbstractDict` for `arguments`, so the
natural `Dict("path" => "/tmp/x")` form works without conversion.

Timeouts surface as the exported `MCPTimeoutError` (`phase` `:connect`, `:queue` or
`:request`). A stdio request timeout closes the session — killing the server is the
only way to release a read blocked on it; the error surfaces at about the bound and a
reply that raced the teardown is discarded. An HTTP request timeout does not close the
session, and the timed-out request gets a best-effort `notifications/cancelled`. A
stdio server that exits or whose pipe breaks surfaces the exported `MCPCrashError`
(the session closes with cause `:crash`). With `auto_respawn=true` the next call on a
session closed by a timeout or a crash respawns the server; otherwise, and after
`mcp_disconnect!`, every call throws `MCPSessionClosedError`.

A `CancelToken` (ambient `with_cancel`) reaches MCP calls over HTTP: a cancel throws
`UniLMCancelled` at once, sends a best-effort `notifications/cancelled`, and leaves the
session open. A stdio exchange does not observe the token (bounded by
`mcp_request_timeout`). Teardown requests (the disconnect `DELETE`, a cancellation
notice) are sent even inside a cancelled scope.

### Session Concurrency

An `MCPSession` runs one call at a time: each call's liveness check, id allocation
and request/response exchange holds the session, and waiting callers are served in
arrival order (FIFO).

- A call's per-call bound (`timeout`, default `mcp_request_timeout`) also bounds its
  **wait** for the session: a call that cannot acquire it in time throws
  `MCPTimeoutError(:queue, …)` without touching it. Once held, the exchange gets the
  full bound, measured from acquisition, so a waiter never cuts it short.
- `mcp_disconnect!` waits its turn too, so a disconnect racing a call in flight
  **waits** for that exchange to finish instead of tearing the transport down
  under its reader.
- Interleaved server → client frames are handled in place: notifications are
  skipped, server `ping` requests answered, other server requests answered `-32601`,
  and a stdio line that is not JSON skipped with a warning.

For genuine parallelism, open one session per concurrent worker.

### Tool Result

`call_tool` returns a typed result — a tool-execution error (`isError: true`) is
carried on `is_error`, **not** thrown; only JSON-RPC protocol errors throw (`MCPError`).

```julia
struct MCPToolResult
    content::String                              # text parts joined by "\n"; non-text parts JSON-encoded
    structured::Union{Nothing,Dict{String,Any}}  # server's structuredContent verbatim, or nothing
    is_error::Bool                               # true when the tool reported an execution error (isError)
    parts::Vector{Any}                           # raw content array verbatim
end
```

The `mcp_tools` / `mcp_tools_respond` bridges surface `content` to the model on
success (falling back to `JSON.json(structured)` when `content` is empty), and
raise `content` as an error when `is_error` is set.

### Tool Bridge

```julia
mcp_tools(session) -> Vector{CallableTool{Tool}}         # for tool_loop!
mcp_tools_respond(session) -> Vector{CallableTool{FunctionTool}}  # for tool_loop
```

Bridged tool names are provider-safe: any character outside `[A-Za-z0-9_-]` becomes `_`
and names are cut to 128 characters (the callable still calls the MCP name); two tools
mapping to one name raise `ArgumentError`.

### Client Example

```julia
session = mcp_connect(`npx -y @modelcontextprotocol/server-filesystem /tmp`)
tools = mcp_tools(session)
chat = Chat(model="gpt-5.2", tools=map(t -> t.tool, tools))
push!(chat, Message(Val(:system), "You are a helpful assistant with filesystem access."))
push!(chat, Message(Val(:user), "List files"))
result = tool_loop!(chat; tools)
mcp_disconnect!(session)
```

---

## MCP Server

Build MCP servers that expose tools, resources, and prompts.

### Types

```julia
MCPServer(name, version; description=nothing)
MCPServerPrimitive    # abstract (MCPServerTool, MCPServerResource, MCPServerResourceTemplate, MCPServerPrompt)
```

### Registration

```julia
register_tool!(server, name, description, schema, handler)
register_tool!(server, name, description, handler)           # auto-schema from signature
register_tool!(server, ct::CallableTool{Tool})            # bridge from Chat API
register_tool!(server, ct::CallableTool{FunctionTool})       # bridge from Responses API
register_resource!(server, uri, name, handler; mime_type="text/plain", description=nothing)
register_resource_template!(server, uri_template, name, handler; ...)
register_prompt!(server, name, handler; description=nothing, arguments=Dict{String,Any}[])
```

The inferred-schema `register_tool!(server, name, description, handler)` binds each
`tools/call` `arguments` object to the handler's positional parameters **by name**, as
`@mcp_tool` does: a parameter whose type admits `nothing` is optional, every other one
required, and a value must have its parameter's JSON type (a `Symbol` binds from a
string; typed `Vector`/`Dict` parameters convert element-wise). A violation is answered
with an `isError: true` tool result naming the argument. A handler taking one `Dict` or
varargs throws `ArgumentError` — pass an explicit schema for those. Registration is
synchronized: it may run while `serve` is dispatching. `@mcp_tool` registers the tool
without a description.

### Macros

```julia
@mcp_tool server function name(args...) body end
@mcp_resource server uri function(args...) body end
@mcp_prompt server name function(args...) body end
```

### Serving

```julia
serve(server; transport=:stdio)                          # default — stdio
serve(server; transport=:http, host="127.0.0.1", port=8080)  # HTTP — blocks until closed
```

The HTTP transport blocks until the server is closed (like `HTTP.serve`); pass
`block=false` to get the running server handle back and `close` it yourself. It
also validates the `Origin` header (DNS-rebinding defense): requests with no
`Origin` header and localhost origins pass, any other origin gets 403 unless
listed in `allowed_origins`. A request other than `initialize` whose
`MCP-Protocol-Version` header names an unsupported revision gets 400; `initialize`
answers the requested revision when supported, otherwise 2025-11-25.

**Concurrency:** over HTTP, `tools/call`, `resources/read` and `prompts/get` handlers
run concurrently, one task per request on the default thread pool — handlers must be
thread-safe; protocol requests (`initialize`, lists, `ping`) are answered inline. Over
stdio handlers run one at a time, and the process `stdout` points at `stderr` while
serving on it, so a printing handler cannot corrupt the frame stream.

```julia
handle = serve(server; transport=:http, port=8080, block=false,
               allowed_origins=["https://app.example.com"])
close(handle)
```

### Error and Limit Contracts

- **A throwing tool handler is a tool result, not a protocol error.** The client
  receives `isError: true` with the text `Error: <showerror text>`, which is exactly
  what lets a model see and correct its own mistake. Write handlers accordingly:
  raise with a message you are willing to show the model *and* the client. A
  `ReplayMissError` or `RecordingWriteError` from a handler's recorded request is
  no mistake a model can correct: it is answered like a dispatch-layer error below.
- **Resource, prompt and dispatch-layer errors are generic.** An exception from a
  resource or prompt handler, or any unhandled error below the handler, answers
  JSON-RPC `-32603` with the message `"Internal error"` — the exception and
  backtrace go to the server's own logs, not to the peer, because an exception
  string can carry file paths and argument values a remote client has no business
  reading. A single bad frame never takes the transport down. An
  `InterruptException` is never converted: it propagates.
- **Frames and request bodies are capped at 16 MiB.** stdio answers an oversized
  frame with `-32600`, HTTP an oversized body with `413 Payload Too Large`, rather
  than parsing it, since parsing an attacker-sized payload allocates a multiple of
  it — an out-of-memory kill rather than a protocol error.
- The server tolerates spec-legal parameter shapes: `params: null` and positional
  `params` do not crash it, and a JSON-RPC response sent to it is accepted without
  an answer (HTTP `202`). Non-object `arguments`, a missing tool `name` and an unknown
  tool answer `-32602`.

### Server Example

```julia
server = MCPServer("calc", "1.0.0")
@mcp_tool server function add(a::Float64, b::Float64)::String
    string(a + b)
end
serve(server)
```

---

## FIM Completion

Fill-in-the-Middle: generate text between a `prompt` (prefix) and `suffix`.
Supported by DeepSeek (beta), Mistral (routed to `/v1/fim/completions`), and Ollama;
vLLM's completions endpoint rejects `suffix`.

```julia
@kwdef struct FIMCompletion
    service::ServiceEndpointSpec
    model::String = ""    # sentinel: resolves at serialization to "deepseek-flash" for DeepSeek
    prompt::String
    suffix::Union{String,Nothing} = nothing
    max_tokens::Union{Int,Nothing} = 128
    temperature::Union{Float64,Nothing} = nothing
    top_p::Union{Float64,Nothing} = nothing
    stream::Union{Bool,Nothing} = nothing          # stream=true throws ArgumentError (no streaming path)
    stop::Union{Vector{String},String,Nothing} = nothing
    echo::Union{Bool,Nothing} = nothing
    logprobs::Union{Int,Nothing} = nothing
    frequency_penalty::Union{Float64,Nothing} = nothing
    presence_penalty::Union{Float64,Nothing} = nothing
end

@kwdef struct FIMChoice
    text::String
    index::Int = 0
    finish_reason::Union{String,Nothing} = nothing
end

@kwdef struct FIMResponse
    choices::Vector{FIMChoice}
    usage::Union{TokenUsage,Nothing} = nothing
    model::String = ""         # through an OllamaEndpoint, a cloud model whose reply drops its tag keeps the requested name (the reply's in raw)
    raw::Dict{String,Any} = Dict{String,Any}()
end

@kwdef struct FIMSuccess <: LLMRequestResponse
    response::FIMResponse
    service::Union{Nothing,ServiceEndpointSpec} = nothing   # the endpoint that answered (prices the call); nothing when built by hand
end

@kwdef struct FIMFailure <: LLMRequestResponse
    response::String
    status::Int
    request_id::Union{String,Nothing} = nothing
end

@kwdef struct FIMCallError <: LLMRequestResponse
    error::String
    status::Union{Int,Nothing} = nothing
    request_id::Union{String,Nothing} = nothing
    cause::Union{Nothing,Exception} = nothing
end

fim_complete(fim::FIMCompletion; config=nothing, cancel=nothing) -> LLMRequestResponse
fim_complete(prompt; suffix=nothing, config=nothing, cancel=nothing, kwargs...) -> LLMRequestResponse  # convenience
fim_text(result) -> String  # generated text; throws LLMResultError on a FIMFailure/FIMCallError
```

Unlike `Chat`, `FIMCompletion` does not resolve its model at construction — the
field stays `""` until serialization, where it becomes `"deepseek-flash"` for
`DeepSeekEndpoint`. Endpoints without a default FIM model must set `model=`
explicitly, or `fim_complete` throws `ArgumentError`. Local validation (capability —
strict, so an endpoint that declares nothing throws `MethodError` — model resolution,
routing) throws before any request; every later failure, including a 200 body that is
not a completions response, is a `FIMCallError` whose `cause` holds the exception.
Retries follow the shared policy; `FIMSuccess` results report usage and cost.

### FIM Example

```julia
result = fim_complete("def fib(a):",
    service=DeepSeekEndpoint(), suffix="    return fib(a-1) + fib(a-2)",
    max_tokens=128, stop=["\n\n"])
println(fim_text(result))
```

---

## Chat Prefix Completion

Continue from a partial assistant message. The model generates text continuing
from the assistant's prefix. DeepSeek beta feature.

```julia
prefix_complete(chat::Chat; config=nothing, cancel=nothing) -> LLMRequestResponse
# Last message must be role=assistant with the prefix text
```

The result's `message` is the continuation the API returns; with `chat.history` the
conversation keeps the whole assistant turn, prefix followed by continuation. A
cancelled call returns `LLMCallError` (`cause::UniLMCancelled`) and leaves `chat`
untouched.

### Prefix Example

```julia
chat = Chat(service=DeepSeekEndpoint())   # deepseek-flash
push!(chat, Message(Val(:system), "You are a coding assistant."))
push!(chat, Message(Val(:user), "Write quicksort in Python"))
push!(chat, Message(role=RoleAssistant, content="```python\n"))
result = prefix_complete(chat)
```

---

## System One API (Jev, OpenAI Decisions, Ollama)

A System One model answers enumerated questions about a piece of `state` with a
probability distribution over the outcomes you named — no generated text, no
parsing. The generic surface below works with any backend that declares
`:system_one`. Three ship, each behind a wrapper (its protocol) that checks
before anything is sent what its wire cannot carry, with an `ArgumentError`
naming the backend, the rule and the fix:

| Backend | `service` | Model when a call names none |
| --- | --- | --- |
| TypeSafe's Jev (default) | `TYPESAFEServiceEndpoint` (`TYPESAFE_API_KEY`) | `TYPESAFE_DEFAULT_MODEL`, else `"jev-latest"` |
| OpenAI's Decisions API (public beta since 2026-10-06) | `OPENAIServiceEndpoint` (`OPENAI_API_KEY`) | `"gpt-6-luna"` (the only model) |
| Ollama decision models (Ollama ≥ 0.35; images ≥ 0.35.1; the MLX tev1 and clef-flash ≥ 0.40.0; measured on 0.40.6) | `OllamaEndpoint()` (no key) | none: `model = "tev1:4b"` (`"clef-flash:9b"` for images) or a `with_system_one` model, else `ArgumentError` |

Each wrapper's rules, refused before any I/O (on every backend also a `DocumentAttachment` and an image's `detail` level, which no System One wire carries):

- **TypeSafe:** ≤ 255 Choice options, ≤ 10 Score levels; images PNG/JPEG,
  still, ≤ 16 MiB, 280–1568 px a side, a JPEG ending with its end-of-image
  marker (FF D9); ≤ 64 levels of nesting in state/instructions/criteria; body
  ≤ 32 MiB; a Box needs an image in the request (and Jev 1.14+:
  `"jev-1.14.0"`/`"jev-preview"`); one option, `long_context::Bool`. Pin a
  model, e.g. `"jev-1.13.0"`, to keep tuned thresholds meaningful.
- **OpenAI:** 1–200 questions; string `instructions` on every question;
  Choice 2–255 options, descriptions strings or `nothing`; Score 2–10 levels,
  each a string; Noul `yes`/`no` criteria and Box refused; images only in the
  state (an `ImageAttachment`, or a `Vector` of strings and images; ≤ 128); each
  text — the input, or each text part — ≤ 10,485,760 characters; options
  `prompt_cache_key::String` (≤ 64 chars), `safety_identifier::String` (≤ 128),
  `prompt_cache_options::PromptCacheOptions` (`mode`; `ttl` `"30m"` or `"12h"`;
  `prewarm` / `comparison_response_id` refused).
- **Ollama:** see the Ollama bullet under Service Endpoints: 1–64 questions,
  `instructions` optional, at most 255 options (one is sent), 2–10 levels (a
  model may take fewer and says so in its 400), images in the state only — images
  then texts — for vision models, Box refused, one option `keep_alive`.

Calibration differs per backend: on 129 labelled tickets with 7 options
(2026-10-20) Jev (`jev-1.13.0`) scored accuracy 0.969 / ECE 0.016 and OpenAI
(`gpt-6-luna`) 0.946 / 0.080, under-confident (accuracy ≥ mean top probability
in every bin above 0.5) — tune `min_confidence` per backend.

- **TypeSafe: images, Box, long context.** An `ImageAttachment` anywhere in the
  state, instructions or criteria (directly, or inside a `Dict`, vector, tuple or
  `NamedTuple`; inside a struct of your own it is an `ArgumentError`) is sent as an
  asset: a `multipart/form-data` request whose first part, `request`, is the JSON
  body, then one part per distinct image (`asset-0`, `asset-1`, …); every place
  the image stood holds `"<image:" * first 32 hex digits of its SHA-256 * ">"`, and
  the body lists them under `assets`. `TypeSafe-Danger-Zone-Images: true` is sent
  whenever a request has an image or a Box question (TypeSafe marks both
  experimental). Each use of an image is billed. A Score `legend` entry that is an
  image comes back as the request's `ImageAttachment`. A recording is keyed on the
  JSON part plus one `asset-k <sha256>` line per image (`key_extra`), never on or
  with the bytes. `box(instructions)` needs Jev 1.14+ (`"jev-1.14.0"`,
  `"jev-preview"`; a 1.13 model's 400 `"Invalid request."` gets a hint naming
  them) and an image in the request; an inside-out box is a `SystemOneCallError`.
  `long_context = true` sends `TypeSafe-Danger-Zone-Long-Context: true` (32k
  tokens per question and 64k per request → 262k, same price, slower); a 400
  `max_tokens_exceeded` failure's message names it. A one-option Choice was
  answered live with probability 1 and confidence 1.0.
- **OpenAI's Decisions wire.** `POST https://api.openai.com/v1/decisions` with
  `{model, input, questions}` plus the given options (`prompt_cache_options` as
  `{"mode", "ttl"}`), which are sent and left out of `record_body` (never keyed or
  recorded). `input` is the state string,
  a JSON value's JSON text, or — for an `ImageAttachment` or a `Vector` of strings
  and images — one user message of `input_text` and `input_image` (data URL)
  parts; a recording keys and stores each data URL as `data:<mime>;sha256,<hex>`.
  `questions` is an array in request order, each entry carrying its name:
  `noul` → `predicate`, `choice` → `choice` (`choices: [{value, description?}]`),
  `score` → `score` (`levels: [{label}]`), a `RawQuestion` verbatim with `name`
  added. Answers are matched by name and must come in the order asked; a
  `predicate` reads as a `NoulAnswer`, a `refusal` as a `RefusalAnswer`, and a
  score's `legend` maps each level to its label. Usage counts input, cached,
  cache-write (`cache_write_tokens`; 12-hour ones in `cache_write_extended_tokens`),
  output and reasoning tokens, and a reported `service_tier` (a count that is not a non-negative whole number, or an
  `input_tokens_details` / `output_tokens_details` that is not an object, reads as
  0 with a warning); the request id is `x-request-id`; an error body
  `{"error": {message, type, param, code}}` gives `error_type` = `code`, else
  `type`. `estimated_cost` bills every input token, cached and cache-written ones
  included, at the row's input rate, whatever tier a reply names (`gpt-6-luna`: 0.10 USD per million), twice that above 272K
  input tokens, and no output.

```julia
# What a piece of guidance may be: string | object | array | image (an image only where the backend sends images)
const SystemOneEntry = Union{AbstractString, AbstractDict, AbstractVector, NamedTuple, ImageAttachment}
# What `state` may be. No `nothing`: a request about nothing asks nothing.
const SystemOneState = Union{AbstractString, AbstractDict, AbstractVector, Tuple, NamedTuple, ImageAttachment}

abstract type SystemOneQuestion end

struct ChoiceQuestion <: SystemOneQuestion          # ≥ 1 option, names not blank, insertion order kept
    instructions::Union{Nothing,SystemOneEntry}
    criteria::JSON.Object{String,Any}               # option name => description (or nothing)
end

struct ScoreQuestion <: SystemOneQuestion           # ≥ 1 level, criteria[1] is level 0
    instructions::Union{Nothing,SystemOneEntry}
    criteria::Vector{Any}
end

struct NoulCriteria                                 # wire keys "true" / "false"
    yes::Union{Nothing,SystemOneEntry}
    no::Union{Nothing,SystemOneEntry}
end

struct NoulQuestion <: SystemOneQuestion            # needs instructions or one criterion
    instructions::Union{Nothing,SystemOneEntry}
    criteria::Union{Nothing,NoulCriteria}
end

struct BoxQuestion <: SystemOneQuestion             # where in an image; instructions required
    instructions::SystemOneEntry                    # (a string must not be blank)
end

struct RawQuestion <: SystemOneQuestion             # a kind this version has no type for
    fields::JSON.Object{String,Any}                 # sent verbatim; needs a non-blank String "type"
end

struct SystemOneRequest
    state::SystemOneState
    questions::Vector{Pair{String,SystemOneQuestion}}   # ordered, unique names, not blank
    model::Union{Nothing,String}                    # nothing: resolved when sent (scope's, else backend default)
    options::NamedTuple                             # provider options, checked by the backend's wrapper
end

abstract type SystemOneAnswer end

struct NoulAnswer <: SystemOneAnswer                # no confidence: the value is the belief
    noul::Float64
    raw::Dict{String,Any}
end

struct ChoiceAnswer <: SystemOneAnswer
    choice::String
    confidence::Float64
    probabilities::Dict{String,Float64}             # keyed by option name, unordered
    raw::Dict{String,Any}
end

struct ScoreAnswer <: SystemOneAnswer
    score::Float64                                  # expectation; may fall between levels
    confidence::Float64
    legend::Dict{Int,Any}                           # 0-based level number => description (an image level: its ImageAttachment)
    probabilities::Dict{Int,Float64}                # same keys as legend
    raw::Dict{String,Any}
end

struct BoxCoordinates                               # fractions of the image's height / width
    top::Float64
    bottom::Float64
    left::Float64
    right::Float64
end

struct BoxAnswer <: SystemOneAnswer
    box::BoxCoordinates
    confidence::Union{Nothing,Float64}              # nothing when the backend reports none
    raw::Dict{String,Any}
end

struct RefusalAnswer <: SystemOneAnswer             # a backend declined one question
    raw::Dict{String,Any}                           #   (accepted only from one that documents refusals)
end

struct UnknownAnswer <: SystemOneAnswer             # an unknown type: kept for a RawQuestion,
    type::String                                    #   a call error for a question of a known kind
    raw::Dict{String,Any}
end

struct SystemOneResponse
    model::String                                   # the id that answered (TypeSafe: versioned)
    answers::Dict{String,SystemOneAnswer}           # keyed by question name, one per question asked
    usage::TokenUsage                               # prompt = input, completion = output
    request_id::Union{Nothing,String}               # the backend's request id (TypeSafe: x-typesafe-request-id)
    raw::Dict{String,Any}
    service::Union{Nothing,ServiceEndpointSpec}     # the backend that answered; nothing when built by hand
end
# SystemOneResponse(model, answers, usage, request_id, raw) fills service = nothing

struct SystemOneSuccess <: LLMRequestResponse
    response::SystemOneResponse
end

@kwdef struct SystemOneFailure <: LLMRequestResponse   # a status other than 200 (every backend's success)
    response::String
    status::Int
    request_id::Union{Nothing,String} = nothing
    error_type::Union{Nothing,String} = nothing     # TypeSafe: "authentication_error", "api_usage_error"
    message::String = ""
    retry_after::Union{Nothing,Float64} = nothing   # seconds; retry-after-ms, else Retry-After
end

@kwdef struct SystemOneCallError <: LLMRequestResponse # no usable response, or answers that do not fit the request
    error::String
    status::Union{Int,Nothing} = nothing
    request_id::Union{Nothing,String} = nothing
    cause::Union{Nothing,Exception} = nothing       # e.g. a UniLMTimeout or UniLMCancelled
end

struct SystemOneError <: Exception                  # thrown by accessors on a non-success
    result::Union{SystemOneFailure,SystemOneCallError}
end

struct TypeSafeModelCard
    name::String
    description::String
    release_date::String                            # opaque string (live returns RFC 3339)
    raw::Dict{String,Any}
end

struct TypeSafeModelsSuccess <: LLMRequestResponse
    models::Vector{TypeSafeModelCard}
    raw::Dict{String,Any}
end
```

### Constructors, verbs and accessors

```julia
choice(instructions, criteria) -> ChoiceQuestion    # criteria: NamedTuple | Dict (sorted) |
choice(criteria) -> ChoiceQuestion                  #   Vector{Pair} | Vector{String}
score(instructions, levels::AbstractVector) -> ScoreQuestion
score(levels::AbstractVector) -> ScoreQuestion
noul(instructions; yes=nothing, no=nothing) -> NoulQuestion
noul(; yes=nothing, no=nothing) -> NoulQuestion
box(instructions) -> BoxQuestion
RawQuestion(fields::AbstractDict)                   # copied into a JSON.Object (a plain Dict sorted)

SystemOneRequest(state, questions; model=nothing, options=(;))

ask(request::SystemOneRequest; service=nothing, config=nothing, cancel=nothing)
ask(state, questions...; model=nothing, service=nothing, config=nothing, cancel=nothing, options...)
    # -> SystemOneSuccess | SystemOneFailure | SystemOneCallError
    # backend: service, else the with_system_one scope's, else TYPESAFEServiceEndpoint
    # model: model (or the request's), else (no service) the scope's, else the backend's default, read at send time
    # any other keyword is a provider option, checked by the backend's wrapper: TypeSafe long_context::Bool;
    #   OpenAI prompt_cache_key, safety_identifier, prompt_cache_options; Ollama keep_alive (seconds, Inf = stay loaded);
    #   any other: ArgumentError listing the accepted ones; none given: (no service) the scope's options

ask_each(states::AbstractVector, questions...; model=nothing, service=nothing, config=nothing,
         cancel=nothing, ntasks=nothing, on_result=nothing, options...)
ask_each(requests::AbstractVector{<:SystemOneRequest}; service=nothing, config=nothing,
         cancel=nothing, ntasks=nothing, on_result=nothing)
    # -> Vector{Union{SystemOneSuccess,SystemOneFailure,SystemOneCallError}}, result i for item i (input order),
    #    each what ask returns for that item alone (one request per item; nothing is packed)
    # every request is built and checked by its wrapper before any is sent: "item i of n" ArgumentError, 0 sent;
    #   the worker that sends an item encodes it again (at most ntasks request bodies alive)
    # ntasks: requests in flight, any Integer ≥ 1 (a Bool is an ArgumentError); default 16 on TypeSafe's wire
    #   and OpenAI's, 1 on Ollama, 8 otherwise
    # a 429/529 to any request holds every not-yet-started attempt (retry-after-ms, else Retry-After, else backoff);
    #   an item's total_deadline runs from when a worker takes it, so a hold waits at most what remains of it:
    #   an item whose budget runs out while held is SystemOneCallError(cause = UniLMTimeout(:deadline, …)), unsent
    # on_result(i, result): any callable with a method for (Int, each result type), checked (hasmethod) before
    #   anything is sent; in the worker task, as each result lands (completion order), one call at a time
    # cancel (default: ambient token): pending items become SystemOneCallError(cause = UniLMCancelled);
    #   ReplayMissError, RecordingWriteError, InterruptException or an on_result exception stops the batch and
    #   the first is rethrown once every worker has stopped

with_system_one(f, service; model=nothing, options...)   # -> f(); the backend of every call inside f
    # that names no service, spawned tasks included, and for those calls the model and the provider options
    # (ask's keywords) when the call names none; service must declare :system_one, model must not be blank;
    # options are checked by the backend's wrapper when each request is sent

list_models(; service=nothing, config=nothing, cancel=nothing)
    # -> TypeSafeModelsSuccess | SystemOneFailure | SystemOneCallError (TypeSafe);
    #    ModelListSuccess | ModelListFailure | ModelListCallError (OPENAIServiceEndpoint, ANTHROPICServiceEndpoint,
    #    GEMINIServiceEndpoint; every page; a missing key is a call error naming it — see Provider model listings);
    #    OllamaSuccess{Vector{OllamaModel}} | OllamaFailure | OllamaCallError (OllamaEndpoint);
    #    ArgumentError for any other backend with no listing

answers(result) -> Dict{String,SystemOneAnswer}     # throws SystemOneError on a non-success
answer(result, name)                                # name::Union{AbstractString,Symbol}
result[name]; haskey(result, name); keys(result)    # same, on Success or Response
```

`questions` accepts `name => question` pairs (vector, tuple, `NamedTuple`, or
`AbstractDict` — a plain `Dict` in sorted name order), or bare questions
auto-named `"q1"`, `"q2"`, … in order. Mixing the two, repeating a name, a blank
name, or passing none is an `ArgumentError`; so is a `state` that is not a
string, object, array or image, a `NaN`/`Inf` anywhere in the content, and
anything the backend's wrapper refuses — all before any request. Every answer
is checked against the request before a success is built (each question answered
once and nothing else; the question's kind, except for a `RawQuestion`; a Choice
of an offered option that is the most probable, with probabilities for exactly
the offered options; Score levels exactly `0 … n−1` in canonical decimal, a score
between them that is `Σ i·pᵢ` within `(n − 1)·tolerance + step` (the sum
tolerance and the rounding step below), and a `legend` that repeats each level
asked as a string unchanged — OpenAI's echoed labels likewise; object, array and
image levels are not compared; every probability, confidence, noul and box edge
in `0 … 1`, a box not inside out (`top ≤ bottom`, `left ≤ right`); probabilities
summing to 1 within the backend's rounding — `0.005n` on TypeSafe and OpenAI,
`n·0.5e-4` on Ollama, plus a floating-point margin — and the chosen option at most
one rounding step below the most probable (0.01, 0.01, 1e-4); a `model` string
that is not blank); a reply that fails is a `SystemOneCallError`. A usage value
that is not a non-negative whole number reads as 0 with a warning (on OpenAI, so
does a usage details value that is not an object). `ask`, `ask_each` and
`list_models` ride the shared retry seam, so `RequestConfig.max_attempts` applies to 408, 429, 500,
502–504, 520–524 and 529, and the final `SystemOneFailure` carries `retry_after`
— the wait the service asked for in seconds, or `nothing` when it sent no hint.
With no `RequestConfig` set by the user (no `config`, no `with_request_config`
scope, no `set_default_config!`), calls on TypeSafe's wire and OpenAI's use
`request_timeout = 60`, `total_deadline = 150`; Ollama's use the field defaults.
TypeSafe retries carry `X-TypeSafe-Retry-Count` (`"1"` on the second attempt).
Every System One exchange uses one connection pool per process (256 idle
connections per host, kept 30 s), created by the first live request.

### Provider protocol (public, not exported)

```julia
abstract type SystemOneProtocol end
struct TypeSafeProtocol <: SystemOneProtocol end    # TypeSafe's wire; the trait's untyped fallback
struct OpenAIDecisionsProtocol <: SystemOneProtocol end   # OpenAI's /v1/decisions; system_one_protocol(::Type{OPENAIServiceEndpoint})
struct OllamaSystemOneProtocol <: SystemOneProtocol end   # Ollama's /v1/systemone; system_one_protocol(::OllamaEndpoint)

UniLM.system_one_protocol(service) -> SystemOneProtocol    # Holy trait: Type{T} for singletons, instances
                                                           #   otherwise; fallback TypeSafeProtocol()
UniLM.encode_system_one(p, service, request::SystemOneRequest) -> SystemOneWire
    # the backend's checks (limits, kinds, content, options): ArgumentError before any I/O
UniLM.decode_system_one(p, service, request, response::HTTP.Response)
    # -> SystemOneSuccess (answers checked) | SystemOneFailure; a malformed 200 throws (→ SystemOneCallError)

struct SystemOneWire
    method::String
    url::String
    path::String                                    # the keyed path: never derived from url
    body::String                                    # the JSON body exactly as sent
    parts::Vector{Pair{String,ImageAttachment}}     # non-empty: multipart/form-data, "request" part first
    headers::Vector{Pair{String,String}}            # non-credential extra headers, after auth_header
    key_extra::String                               # appended to the key input (after "\n") when non-empty
    record_body::String                             # the body as keyed and recorded: body, or body without
end                                                 #   fields that change how a request is served, never its answer
SystemOneWire(method, url, path, body, parts, headers, key_extra[, record_body])   # record_body defaults to body
SystemOneWire(method, url, path, body; parts=[], headers=[], key_extra="", record_body=body)
```

The exchange: resolve the backend, model and options, check `:system_one`, encode
(outside the error handling), then — through any `with_recorded_answers` scope,
keyed by `"/<namespace>" * path` (TypeSafe's namespace is empty, OpenAI's
`"openai-decisions"`, Ollama's `"ollama"`, a protocol of your own its type's
name) and `record_body` plus `key_extra` (TypeSafe: `record_body == body`; Ollama
leaves out `keep_alive`, OpenAI `prompt_cache_key`, `safety_identifier` and `prompt_cache_options`) — send
`body` with `auth_header(service)` read only
for a live call, through the System One connection pool, and decode. An endpoint
that declares `:system_one` and no protocol speaks TypeSafe's wire: a gateway
needs `provider_capabilities` (with `:system_one`), a base URL (the internal hook
`UniLM._api_base_url`) and `auth_header`, or no code at all when
`TYPESAFE_BASE_URL` points `TYPESAFEServiceEndpoint` at it.

### Example

```julia
r = ask("Help! My payouts have been failing for 3 days.",
        "department" => choice("Which team should handle this ticket?",
            (billing="Payments, invoicing, payouts, refunds",
             technical="Bugs, outages, integrations",
             sales="Pricing, upgrades, new accounts")),
        "urgency" => score("How urgent is this ticket?",
            ["Can wait", "Needs attention this week", "Needs attention today"]),
        "is_frustrated" => noul("Is the customer frustrated?"; yes="Frustrated", no="Neutral"))
issuccess(r) && println(r["department"].choice, " ", r["urgency"].score, " ", r["is_frustrated"].noul)
```

### Recorded answers

Identical requests are not answered identically, so tests and docs builds
replay recorded answers. Inside the scope, every `ask` (hence `ask_each`,
`nl_dispatch`, `nl_classify` and `@branch`) and `list_models`, and every non-streaming
`chatrequest!`, `respond` and `embeddingrequest!` (hence `tool_loop!` and
`tool_loop`), exchanges through `dir`; everything else, including streaming,
reaches the network as usual (images, audio, files, MCP, the Responses lifecycle
operations), in every mode. Scopes cover spawned tasks and nest (an inner scope's
service is the enclosing scope).

```julia
with_recorded_answers(f, dir::AbstractString; mode::Symbol=:replay)   # -> f()
    # :replay         answer from dir: no recorded verb reaches the network or needs a key
    #                 (an unrecorded verb behaves as outside the scope); no recording or an
    #                 unreadable one throws ReplayMissError
    # :record         call the service; write each HTTP 200 to dir, replacing any earlier file
    # :record_missing replay what dir holds; call and record the rest; an unreadable
    #                 file throws ReplayMissError (never overwritten)
    # both recording modes create dir and prove it writable before f runs (else ArgumentError);
    #   a recording that cannot be written after a paid answer throws RecordingWriteError out
    #   of the verb, and out of a tool loop whose dispatcher made the request
    # file: <dir>/<key>.json, key = lowercase hex sha256("<METHOD> <path>\n" * body),
    #   <path> = the request URL's path alone (no host, no query), after "/<namespace>" for a
    #   System One backend other than TypeSafe ("/<namespace>/v1/systemone");
    #   a System One request keyed on more (TypeSafe's long_context as a sorted name=<JSON> line,
    #   image-part digests) appends "\n" * that text to the body it is keyed on;
    #   body = the exact body (for System One, the wire's record_body: Ollama's without
    #   keep_alive, OpenAI's without prompt_cache_key, safety_identifier and prompt_cache_options), except that in its
    #   top-level "tools" array an MCP tool's ("type": "mcp") "authorization" and "headers", a
    #   Gemini "mcp_server"'s "headers" and a Gemini "retrieval" tool's exa_ai_search_config /
    #   parallel_ai_search_config "api_key" read "<redacted>", and every string
    #   "data:<mime>;base64,<payload>" (an OpenAI chat image_url, a Decisions or Responses
    #   input_image) reads "data:<mime>;sha256,<hex sha256 of the payload text>" (key, file and
    #   ReplayMissError alike; the request sent is unchanged); so are the images of an Ollama System One
    #   request ("images") and the image/audio of an Ollama /api/embed item (`record_body`, the body keyed);
    #   media sent another way (Ollama chat "images", "input_audio") is written as sent
    # {"request": {method, path, body, key_extra (only when not empty)},
    #  "response": {status, request_id, request_id_header, body}, "recorded_at"}
    #   request_id_header: "x-typesafe-request-id" | "x-request-id" | "request-id", the one header
    #   kept; absent when the reply carried no id, and in recordings made before it existed,
    #   which replay their id as "x-typesafe-request-id"

struct ReplayMissError <: Exception       # thrown out of the recorded verbs, never a *CallError
    dir::String
    key::String
    method::String                        # "POST" | "GET"
    path::String                          # as keyed: "/v1/systemone" | "/<namespace>/v1/systemone" | "/v1/models" | …
    body::String                          # as keyed: credentials "<redacted>", data-URL images as digests ("" for GET),
                                          #   then "\n" * key_extra if any
    reason::String                        # "no recording" | "unreadable recording: <what is wrong>"
end

struct RecordingWriteError <: Exception   # a paid answer whose recording could not be written;
    file::String                          #   thrown out of the recorded verbs and the tool loops,
    cause::Exception                      #   never a *CallError; an MCP handler's: -32603, logged
end
```

### Natural-language control flow

A Choice answer is already a branch decision, so two constructs compile ordinary
control flow into ONE System One request. `@branch` is a natural-language switch
that evaluates only the selected body; `nl_dispatch` sends one Choice per
`Meaning` slot and then hands the resolved meanings to Julia's own dispatch, so
the remaining arguments still dispatch on their types.

```julia
struct Meaning{S} end                     # S::Symbol IS the description
Meaning(text::AbstractString)             # the instance; nl"..." is the TYPE
# the type prints as Julia prints types, Meaning{Symbol("…")} (in MethodErrors and methods(f) too;
# a one-word sentence prints as Meaning{:x});
# an instance prints as nl"…"(); identical sentences are one type (===)

struct LowConfidenceError <: Exception    # raised when confidence < min_confidence
    question::String
    answer::ChoiceAnswer                  # the full distribution, for diagnostics
    min_confidence::Float64
end

struct DecisionDeclinedError <: Exception # raised when `decide` returned nothing, or the backend
    question::String                      #   declined the question (a RefusalAnswer); "branch" for @branch
    answer::Union{ChoiceAnswer,RefusalAnswer}
end

@branch state [key = value ...] begin
    "option name"                  => expression   # the name is what the model reads
    ("option name", "description") => expression   # the only way to add a description
    _                              => expression   # requires min_confidence > 0 or a decide policy
end
# an option name is a String, or a Symbol sent as its String; any other value is an
#   ArgumentError naming its line, before the request
# keys: model, min_confidence, decide, instructions, service, config, cancel, on_response, options
#   min_confidence: a finite number in 0 … 1; anything else, NaN included, is an ArgumentError
#     before the request (nl_dispatch and nl_classify alike)
#   decide: ChoiceAnswer -> any option name, or nothing to decline; not with min_confidence
#   on_response: SystemOneSuccess -> ignored; called once before the policy, never on a failed call
#   service/model/options: resolved as for ask (the with_system_one scope, else TYPESAFEServiceEndpoint);
#     options = (long_context = true,): a NamedTuple of the provider options ask takes as keywords
# a RefusalAnswer (the backend declined the question) takes `_`, else DecisionDeclinedError; decide not called
# -> the selected body's value; LowConfidenceError, DecisionDeclinedError or SystemOneError otherwise

nl_dispatch(f, args...; model=nothing, service=nothing, config=nothing,
            cancel=nothing, min_confidence=0.0, decide=nothing, fallback=nothing,
            instructions=nothing, state=nothing, texts=nothing, on_response=nothing,
            options=(;))
    # service/model/options: resolved as for ask (the with_system_one scope, else TYPESAFEServiceEndpoint);
    #   options: a NamedTuple of provider options, e.g. (long_context = true,), checked by the backend's wrapper
    # -> f(resolved meanings — or keys — spliced into their positions, args...)
    # state: nothing builds it from args, every value as JSON.jl writes it — every field of a struct
    #   included, and written to disk by a :record scope; a function, type, module, Ref or Ptr, Task,
    #   Channel, lock, condition, event, IO, Expr or a value JSON.jl cannot write (an AudioAttachment)
    #   anywhere in it, or a container nested deeper than 64 levels, is an ArgumentError before the
    #   request naming the argument and the path; an ImageAttachment is data (TypeSafe: an asset;
    #   OpenAI's and Ollama's wrappers refuse it there); an explicit state is sent as given
    # decide: nothing (argmax gated by min_confidence) | a callable | a Vector with one per slot;
    #   ChoiceAnswer -> any offered meaning, or nothing to decline; not with min_confidence;
    #   a slot the backend refused (RefusalAnswer) is declined without calling its policy
    # fallback: nothing | callable as fallback(args...) on a decline; one that cannot take
    #   arguments of the types of args is an ArgumentError before the request
    # texts: nothing (meanings are nl"..." in signatures) | a table | a Tuple of tables, one per keyed slot
    #   table: NamedTuple of sentences | Vector of key => sentence pairs (a Dict is refused: no order)
    #   key: Symbol (passed as Val(key)) | singleton instance (Val(:k), Refund()) | type | enum value;
    #   never nothing or missing; two keys whose dispatched values are isequal are one key
    #   only the sentences are sent; the chosen one maps back to its key; decide may also
    #   return an offered key as the table writes it (:refund, not Val(:refund))
    # on_response: nothing | SystemOneSuccess -> ignored; called once before the policy
    #   (audits: request_id, model, raw), never on a failed call; its exception propagates
nl_classify(state, texts; min_confidence=0.0, decide=nothing, fallback=nothing,
            instructions=nothing, model=nothing, service=nothing,
            config=nothing, cancel=nothing, on_response=nothing, options=(;))
    # -> the chosen key as the table writes it; ONE table; one Choice "classify" over every
    #   sentence in table order; a decline (a refusal included) calls fallback(state), else
    #   LowConfidenceError / DecisionDeclinedError; a fallback that cannot take state is an
    #   ArgumentError before the request
meanings(f; texts=nothing) -> Dict{Int,Vector{String}}   # slot position => options over every method
meanings(f, argtypes::Type{<:Tuple}; texts=nothing) -> Dict{Int,Vector{String}}
    # the options a call with ordinary arguments of these types sends, in send order
meaning_gaps(f, argtypes::Type{<:Tuple}; texts=nothing) -> Vector{Vector{String}}
    # offered combinations with no method (missing or ambiguous), first slot slowest
```

Methods without a concrete `Meaning` argument (including wildcard `::Meaning`
ones) are ordinary methods and are not part of the natural-language interface.
Every natural-language method of `f` must agree on arity and on which positions
are slots, so a partial wildcard (`f(::nl"a", ::Meaning, x)`) is an
`ArgumentError`. A call offers only the meanings of methods that accept the
types of its ordinary arguments. Types no such method accepts, and a combination
of offered meanings with no method or an ambiguous one, are each an
`ArgumentError` raised before any request. A combination with no method is
closed by defining it, or by one method wild in every slot
(`f(::Meaning, ::Meaning, x)`), which is not an option and covers every
combination that has no method; an ambiguous one only by a method for the
intersection of the methods it collides on, which the error names. Without
`state`, the state is a `JSON.Object{String,Any}` keyed by the ordinary argument
names of the first such method, in argument order; the questions are named from the
same method. `confidence` is `(n·p_max − 1)/(n − 1)` clamped
to `0 … 1` over the `n` offered options, so a `min_confidence` is a different bar
whenever the option list changes.

With `texts`, the methods of `f` dispatch on keys, and the request is the one the same
sentences written as `nl"..."` would make. The call's arity is `length(args)` plus one per
table, and only methods of that arity take part. A table fills the one position whose
declared type its keys have (`::Val{:refund}`, `::Refund`, `::Type{<:Billing}`, `::Val`;
`::Any` declares none). Refused before any request: a `Dict`, an empty table, more entries
than the backend lets one question offer (255 on TypeSafe), a blank sentence or one that is
not a string, a sentence given twice, two keys whose
dispatched values are `isequal` (`:a` and `Val(:a)`, `Vector` and `Array{S,1} where S`),
`nothing` or `missing` as a key, a Tuple of pairs, a key that is not a Symbol, singleton
instance, type or enum value, a table with no position or several, two tables in one position,
a method of that arity pinning a `Meaning`, and a key no method takes at its position whatever
the other arguments are (a catch-all `f(k, t)` takes every key). Options follow the argument
types in **table order**, where meanings are offered in definition order; gaps are refused
alike. The questions and the state are named from the first method, in definition order, that
pins a table — never from a catch-all defined before it.

```julia
ticket = "My package arrived crushed and the screen is cracked. I want my money back."

action = @branch ticket min_confidence=0.6 begin
    "the customer wants a refund"           => :refund
    "the customer reports a bug in the app" => :bug
    _                                       => :escalate
end                                         # => :refund

route(::nl"the customer wants a refund", t)           = (:refund, t)
route(::nl"the customer reports a bug in the app", t) = (:bug, t)
nl_dispatch(route, ticket)                            # => (:refund, ticket)
route(nl"the customer wants a refund"(), ticket)      # direct call, no request

# The same request with the sentences in a table and the keys in the signatures.
const INTENT = (refund = "the customer wants a refund", bug = "the customer reports a bug in the app")
triage(::Val{:refund}, t) = (:refund, t)
triage(::Val{:bug}, t)    = (:bug, t)
nl_dispatch(triage, ticket; texts = INTENT)           # => (:refund, ticket), via triage(Val(:refund), ticket)
nl_classify(ticket, INTENT)                           # => :refund

# A policy may take any offered meaning, not only the argmax, or decline (-> fallback).
refund_first(a) = a.probabilities["the customer wants a refund"] >= 0.3 ?
                  "the customer wants a refund" : (a.confidence >= 0.7 ? a.choice : nothing)
nl_dispatch(route, ticket; decide = refund_first, fallback = t -> :escalate)
```

---

## Audio: speech, transcription, translation

```julia
struct SpeechRequest                                # built by keyword: SpeechRequest(; input, service, …)
    service::ServiceEndpointSpec                    # OPENAIServiceEndpoint (Realtime session) or GEMINIServiceEndpoint (Gemini TTS)
    model::String                                   # nothing → "gpt-realtime-2.1-mini" (OpenAI) / "gemini-3.8-flash-tts" (Gemini)
    input::String                                   # the text to read; empty → ArgumentError
    voice::String                                   # nothing → "marin" (OpenAI) / "Kore" (Gemini)
    response_format::String                         # "wav" (default) or "pcm": 16-bit mono PCM at 24 kHz
    speed::Union{Float64,Nothing}                   # OpenAI 0.25 … 1.5; Gemini has none (describe the pace in instructions)
    instructions::Union{String,Nothing}             # the delivery (tone, pace, accent)
end
@kwdef struct SpeechSuccess <: LLMRequestResponse
    audio::Vector{UInt8}                            # a WAV file, or the raw samples
    content_type::String = ""                       # "audio/wav" | "audio/pcm"
end

speak(s::SpeechRequest; config=nothing) -> SpeechSuccess | AudioFailure | AudioCallError
speak(input; service=OPENAIServiceEndpoint, model=nothing, voice=nothing, response_format="wav",
      speed=nothing, instructions=nothing, config=nothing)
save_audio(r::SpeechSuccess, path)
translate(path; language="en", model="gpt-realtime-translate", service=OPENAIServiceEndpoint, config=nothing)
    -> TranscriptionSuccess | AudioFailure | AudioCallError

@kwdef struct TranscriptionRequest
    service::ServiceEndpointSpec = OPENAIServiceEndpoint
    file::String                                    # a path; a missing file throws ArgumentError
    model::Union{String,Nothing} = nothing          # nothing: the service's default, resolved when sent
    language::Union{String,Nothing} = nothing
    prompt::Union{String,Nothing} = nothing
    response_format::Union{String,Nothing} = nothing   # json, text, srt, verbose_json, vtt, diarized_json (Ollama: json)
    temperature::Union{Float64,Nothing} = nothing      # 0 … 1
    languages::Union{Vector{String},Nothing} = nothing # not with language
    keywords::Union{Vector{String},Nothing} = nothing  # no <, > or line breaks
end

transcribe(t::TranscriptionRequest; config=nothing) -> TranscriptionSuccess | AudioFailure | AudioCallError
transcribe(path::String; model=nothing, service=OPENAIServiceEndpoint, config=nothing, kwargs...)
transcript_text(r::TranscriptionSuccess) -> String
```

**Speech.** On `OPENAIServiceEndpoint`, `speak` runs on a Realtime API session
(`wss://api.openai.com/v1/realtime`, `gpt-realtime-2.1-mini` unless `model` names another — OpenAI's
replacement for its text-to-speech models, which shut down on 2027-01-06): UniLM sets the voice
(`alloy`, `ash`, `ballad`, `coral`, `echo`, `sage`, `shimmer`, `verse`, `marin`, `cedar`), 24 kHz PCM
output and instructions to read the text word for word, and collects the response's audio (Realtime
models are conversational; OpenAI notes instructions are not guaranteed to be followed). On
`GEMINIServiceEndpoint` it is Gemini TTS (`generateContent` on `gemini-3.8-flash-tts`, one of its 30
prebuilt voices, `instructions` as the speech style), which reads the text verbatim. An empty
`input`, a voice the service lacks, a `response_format` other than `"wav"`/`"pcm"` (mp3, opus, aac and
flac are refused), a `speed` out of range and any `speed` on Gemini throw `ArgumentError` before any
request, as does `GEMINIOpenAIServiceEndpoint` (no speech route there: the error points to
`GEMINIServiceEndpoint`). **`translate`** streams a 16-bit PCM WAV file (any rate and channel count: mixed down to mono
and resampled to 24 kHz) through an OpenAI Realtime translation session
(`wss://api.openai.com/v1/realtime/translations`; OpenAI shuts `/v1/audio/translations` down on
2027-02-26) into `language` (English unless named) and returns the translated text as a
`TranscriptionSuccess` (speech already in the target language can come back empty); any other file,
an empty one, a blank `language` and a service without the Realtime API throw `ArgumentError` before
I/O. A failure the provider reports is an `AudioFailure` — an HTTP error status, a refused WebSocket
upgrade (a bad key is a 401), a Gemini candidate that did not finish (`finishReason` other than
`STOP`) or a reply without one (a blocked prompt), both with `status` 200, or a Realtime `error`
event — read even when the server closes the session while UniLM sends — or unfinished
`response.done` (`status = nothing`, the event's JSON as `response`); a session that closes before
its result (the close's status and reason in `error`, the `HTTP.WebSockets.WebSocketError` in
`cause`), a timeout, a cancel and a reply without usable audio are an `AudioCallError`, and audio
received before a failure is never returned. Both
make a single bounded attempt (`total_deadline` the whole call, `connect_timeout` the WebSocket open,
`stream_idle_timeout` every send and every wait for the next event) and observe the ambient
`CancelToken` (`with_cancel`): a token already cancelled sends nothing, and a cancel ends the call at
once with `cause = UniLMCancelled`. Speech is not priced: `token_usage` and `estimated_cost` throw for
a `SpeechSuccess`.

**Transcription.** The default model is `"gpt-transcribe"` on OpenAI (and OpenAI-compatible services); an
`OllamaEndpoint` has none, so a request to it names an installed model that hears audio
(`model = "gemma4:e4b"`) or throws `ArgumentError` before any request. On Ollama the request
goes to `/v1/audio/transcriptions` and asks for JSON only; `temperature`, `languages`,
`keywords` and a `response_format` other than `"json"` throw before I/O — `"text"` with the
fix, since Ollama reports a failure that arrives after its reply has begun only as JSON and
`"text"` would read it as an empty transcript: leave `response_format` unset and read the text
with `transcript_text` — and the reply is read as JSON (Ollama labels it
`application/x-ndjson`). On OpenAI, `srt`, `vtt` and `verbose_json` (subtitles and
timestamps) are documented only for `whisper-1` and `diarized_json` (speaker labels) only for
`gpt-4o-transcribe-diarize`, models that shut down on 2027-02-26. `transcribe` needs
`:transcription`, which `has_capability` reports for any endpoint declaring `:audio`; `speak` on
OpenAI and `translate` need `:realtime`; `speak` on `GEMINIServiceEndpoint` is Gemini TTS.

---

## Provider Capabilities

Each endpoint declares supported features. Request functions validate before
dispatch, so an unsupported feature is an `ArgumentError` naming the endpoint and what
it does support, not an HTTP 404.

```julia
provider_capabilities(service) -> Set{Symbol}
has_capability(service, cap::Symbol) -> Bool
```

### Capabilities by Provider

| Provider | Capabilities |
|---|---|
| OpenAI | `:chat`, `:responses`, `:agentic`, `:embeddings`, `:images`, `:image_edits`, `:tools`, `:streaming`, `:json_output`, `:files`, `:vector_stores`, `:conversations`, `:moderation`, `:audio`, `:transcription`, `:batch`, `:fine_tuning`, `:containers`, `:uploads`, `:realtime`, `:system_one` (its Decisions API), `:models` |
| Azure | `:chat`, `:tools`, `:streaming`, `:json_output` |
| Gemini (native) | `:chat`, `:tools`, `:json_output`, `:streaming`, `:agentic`, `:images`, `:image_edits`, `:models` |
| Gemini (OpenAI-compat) | `:chat`, `:embeddings`, `:tools`, `:streaming`, `:json_output` |
| Anthropic (native) | `:chat`, `:tools`, `:json_output`, `:streaming`, `:models` (the bare type and an instance alike) |
| TypeSafe (System One) | `:system_one`, `:models` |
| DeepSeek | `:chat`, `:tools`, `:streaming`, `:fim`, `:prefix_completion`, `:json_output` |
| Generic | `:chat`, `:embeddings`, `:fim`, `:tools`, `:streaming`, `:json_output`, `:responses` |
| Ollama | `:chat`, `:embeddings`, `:fim`, `:tools`, `:streaming`, `:json_output`, `:responses`, `:models`, `:system_one`, `:transcription` |

Validation comes in two strengths, and the difference matters if you write your
own endpoint:

- **Platform and lifecycle verbs** (files, vector stores, conversations,
  moderations, audio, batch, fine-tuning, containers, uploads, realtime, FIM,
  prefix completion, System One) validate **strictly**: an endpoint with no
  declaration at all is not dispatched — the call throws `MethodError` (no
  `provider_capabilities` method), or for the System One verbs (`ask`,
  `ask_each`, `with_system_one`, the natural-language verbs) an `ArgumentError`
  saying to declare `:system_one` — because those surfaces are provider-specific
  and an undeclared backend would just 404.
- **The four primary verbs** — `chatrequest!` (`:chat`), `embeddingrequest!`
  (`:embeddings`), `respond` (`:responses` **or** `:agentic`), and
  `generate_image` (`:images`) — plus `edit_image` (`:image_edits`) validate only
  endpoints that **declare** their capabilities. A custom endpoint that defines no `provider_capabilities` method
  passes through unvalidated: the package has no basis for claiming what someone
  else's OpenAI-compatible server cannot do, and refusing to dispatch it would be
  a false negative. Declaring capabilities is therefore opt-in strictness — once
  you declare, you are held to the declaration.

Either way you never need to check capabilities manually before a request.

---

## Request Configuration & Timeouts

Every network operation is bounded. One struct carries all knobs — all time
fields are seconds, `Inf` disables that bound:

```julia
Base.@kwdef struct RequestConfig
    connect_timeout::Float64     = 10.0
    request_timeout::Float64     = 600.0
    stream_idle_timeout::Float64 = 120.0
    total_deadline::Float64      = 900.0
    max_attempts::Int            = 3
    mcp_connect_timeout::Float64 = 120.0
    mcp_request_timeout::Float64 = 120.0
end
```

- `connect_timeout` — per-attempt connection establishment. `request_timeout` — per-attempt whole exchange (non-stream); a stream's exchange up to its first byte. `stream_idle_timeout` — byte-gap between raw stream chunks. `total_deadline` — across ALL attempts including backoff (streams: until first byte). `max_attempts` — wire attempts (`1` disables retries). `mcp_connect_timeout` / `mcp_request_timeout` — MCP handshake / per-exchange bounds.
- **Validation**: every `Float64` field rejects `NaN`, values `≤ 0` and finite values above `1e9` s with `ArgumentError` (`Inf` = disabled); `max_attempts ≥ 1`.
- **Copy-with-overrides**: `RequestConfig(base::RequestConfig; kwargs...)`.
- **Five channels, struct-wise precedence** (a channel supplies a complete struct):
  1. per-call `config::Union{Nothing,RequestConfig}` keyword on request verbs;
  2. dynamic scope: `with_request_config(f; kwargs...)` — merges kwargs over `current_config()` at entry; propagates into `Threads.@spawn`;
  3. process default, once set: `set_default_config!(cfg)` or `set_default_config!(; kwargs...)` (merges over the current default, or over the built-in defaults while none is set) — the channel for REPL/notebook sessions;
  4. a backend's own default — System One: `ask`, `ask_each` and the NL verbs on TypeSafe's wire and on OpenAI's Decisions API, and `list_models` on TypeSafe, use `request_timeout = 60`, `total_deadline = 150`; Ollama brings none — or a model's own default: `chatrequest!` and `respond` to a Gemini 4 model (`gemini-4-…` on `GEMINIServiceEndpoint` or `GEMINIOpenAIServiceEndpoint`) use `request_timeout = stream_idle_timeout = total_deadline = 1200` (Google ends a non-streamed Argon call after 10 minutes, a stream after 20, and advises a 20-minute read timeout). Either applies only while channels 1–3 supply nothing: a process default you set, even one equal to the built-in defaults, replaces it;
  5. the built-in defaults above.
  `current_config()` returns the ambient struct (active scope, else the process default you set, else the built-in defaults) — channel 4 is never in it.

```julia
with_request_config(request_timeout=30.0, max_attempts=1) do
    chatrequest!(chat)          # bounded, no retries, inside this scope only
end
set_default_config!(total_deadline=120.0)   # process-wide default
```

### Which verbs retry

`max_attempts` applies to the inference verbs only: `chatrequest!`,
`embeddingrequest!`, `respond`, `fim_complete`, `prefix_complete`,
`generate_image`, `edit_image`, `ask`, `ask_each` (per request), `list_models`, and the `tool_loop` family
(streams retry only before the first callback fires). Platform and lifecycle verbs
— batch, container, conversation, file (`upload_file` included: a create is never
retried), fine-tuning, moderation, upload, vector-store, audio, realtime, and the
Responses lifecycle operations — make a **single bounded attempt**, and
`max_attempts` has no effect on them. A 429 whose `error.code` reports spent credit or a
reached spend or usage limit (the four codes under `chatrequest!`) is never retried. `poll_batch` /
`poll_file_batch` poll through retryable statuses (such a 429 ends the poll with its failure) and
per-attempt timeouts until their own wall-clock `timeout`
(`interval` and `timeout` must be > 0); a poll that times out returns the family's
call error with `cause = UniLMTimeout(:deadline, …)` and the last object seen in
`last_observed`. Both take `cancel`.

### Cancellation

```julia
tok = CancelToken()                 # level-triggered: once cancelled, stays cancelled
cancel!(tok)                        # from any task; idempotent; runs every registered hook
iscancelled(tok)::Bool              # iscancelled(nothing) == false
with_cancel(f, tok)                 # tok is the ambient token inside f (and tasks spawned there)
```

HTTP verbs take `cancel::Union{Nothing,CancelToken}=nothing` (`chatrequest!`,
`respond`, `embeddingrequest!`, `generate_image`, `edit_image`, `fim_complete`,
`prefix_complete`, `ask`, `ask_each`, `list_models`, `poll_batch`, `poll_file_batch`, the tool
loops, `nl_dispatch`, `nl_classify`, `@branch`); `nothing` resolves the ambient token, and every other
HTTP verb observes the ambient token. A cancelled call returns its call-error result
with `status = nothing` and `cause = UniLMCancelled(source, elapsed)` (`source` is
`:token`, or `:callback` for a stream stopped with `close[] = true`), is never retried,
commits nothing — unless the provider's completion marker already arrived on a stream:
then the turn is committed, the final-message callback runs, and usage may be missing
(a Chat stream's marker is the chunk carrying the finish reason, before the usage chunk
and `[DONE]` on the OpenAI wire; a Responses stream's is its terminal event, whose
response carries its usage). A pre-cancelled token sends nothing. Limits: a TCP connect / TLS
handshake in progress is not interrupted (it completes or hits `connect_timeout`), a
running callback finishes first, and a `realtime_connect` session and MCP stdio exchanges do
not observe the token (`speak` and `translate`, which run Realtime sessions of their own, do). See
the Concurrency, Tasks and Cancellation guide.

### Sharp edges

- **Streams, pre-first-byte,** are bounded by `min(request_timeout, remaining
  total_deadline)` per attempt (a peer that never answers, or never reads the request,
  ends the attempt with `UniLMTimeout(:request, …)`) and additionally by
  `stream_idle_timeout` (HTTP.jl caps the response-header wait with the read-idle
  timer), so the effective bound is `min(request_timeout, remaining total_deadline,
  stream_idle_timeout)`; when the idle bound is the smallest, a breach reports phase
  `:stream_idle` even though no byte arrived. A TCP connect / TLS handshake in progress
  completes, or reaches `connect_timeout`, first.
- **The idle bound measures wire idleness only.** Time a stream spends inside the
  user's `callback` / `on_tool_call` is not counted, so a slow consumer never
  idle-kills a healthy stream — and no bound covers a callback that never returns.
- **Timeouts land on time.** A non-streaming attempt's bound fires within a few ms of
  its limit, and a successful call has no detection latency.
- **A completed turn is never discarded or re-sent.** Once a stream records its
  terminal state, teardown noise — idle breach, transport reset, truncated read —
  finalizes it as a success rather than failing or retrying it, on every provider
  and both stream drivers. Re-POSTing would bill a second generation.
- **Realtime**: `realtime_connect` (OpenAI only: any other `service` throws
  `ArgumentError` before any I/O; it opens `gpt-realtime-2.1` unless `model` names another) bounds the open phase with `connect_timeout`, surfaces
  an open that failed for another reason as that error, and closes an upgrade that
  completes after the timeout without running the handler; `realtime_receive` bounds
  the read with `stream_idle_timeout` and surfaces a breach just after the limit: it ends
  the connection without the send lock, so a `realtime_send` parked on a peer that stopped
  reading cannot hold it open (that send throws too).
  An open session's lifetime is deliberately unbounded. `mint_realtime_secret` returns
  `RealtimeCallError` for a 200 without a non-empty secret.

### Concurrency

- **One `Chat` per in-flight call.** `messages` and the cumulative-cost `Ref` are
  unsynchronized by design; use `fork(chat)` / `fork(chat, n)` to fan out. An
  `Embeddings` is filled in place (and aliased by its result): one per concurrent
  call. `Respond`, `ImageGeneration`, `FIMCompletion` and `SystemOneRequest` are not
  mutated by their verbs and may be shared.
- **An `MCPSession` runs one call at a time**, FIFO; a call's `timeout` bounds its
  wait for the session too (`MCPTimeoutError(:queue)`), and `mcp_disconnect!` waits
  for the in-flight exchange. One session per concurrent worker. `serve(:http)` runs
  handlers concurrently (thread-safe handlers required); stdio runs them one at a time.
- **Streams use one HTTP/1.1 connection each**; non-streaming calls may multiplex over
  HTTP/2. HTTP.jl sets no per-host connection cap by default. `Retry-After` is a floor
  under jitter, so a rate-limited batch does not retry in lockstep.
- **Thread 1** (the main task, and the libuv event loop) must not be blocked: a task
  that computes there without yielding delays every timer. Keep callbacks short.
  `tool_concurrency = n` runs a turn's tool calls on spawned tasks.

---

## Result Type Hierarchy

All 71 API call result types inherit from `LLMRequestResponse`, in 18 per-API
families:

```
LLMRequestResponse   (abstract parent of every result type below)
│
├─ Chat Completions   LLMSuccess · LLMFailure · LLMCallError
├─ Responses API      ResponseSuccess · ResponseFailure · ResponseCallError
├─ Embeddings         EmbeddingSuccess · EmbeddingFailure · EmbeddingCallError
├─ Image Generation   ImageSuccess · ImageFailure · ImageCallError
├─ FIM Completion     FIMSuccess · FIMFailure · FIMCallError
├─ Files              FileSuccess · FileListSuccess · FileContentSuccess · FileDeleteSuccess · FileFailure · FileCallError
├─ Vector Stores      VectorStoreSuccess · VectorStoreListSuccess · VectorStoreFileSuccess · VectorStoreBatchSuccess · VectorStoreDeleteSuccess · VectorStoreFailure · VectorStoreCallError
├─ Conversations      ConversationSuccess · ConversationItemSuccess · ConversationItemListSuccess · ConversationDeleteSuccess · ConversationFailure · ConversationCallError
├─ Moderations        ModerationSuccess · ModerationFailure · ModerationCallError
├─ Audio              SpeechSuccess · TranscriptionSuccess · AudioFailure · AudioCallError
├─ Batch              BatchSuccess · BatchListSuccess · BatchFailure · BatchCallError
├─ Fine-tuning        FineTuningSuccess · FineTuningListSuccess · FineTuningFailure · FineTuningCallError
├─ Containers         ContainerSuccess · ContainerListSuccess · ContainerDeleteSuccess · ContainerFailure · ContainerCallError
├─ Uploads            UploadSuccess · UploadPartSuccess · UploadFailure · UploadCallError
├─ Realtime           RealtimeSecretSuccess · RealtimeFailure · RealtimeCallError
├─ Model listings     ModelListSuccess · ModelListFailure · ModelListCallError
├─ Ollama models      OllamaSuccess · OllamaFailure · OllamaCallError
└─ System One         SystemOneSuccess · TypeSafeModelsSuccess · SystemOneFailure · SystemOneCallError
```

Every `*Success` wraps a parsed response object, every `*Failure` carries the HTTP
status and body (every family but Embeddings and Realtime also the `request_id`; an
`AudioFailure` a Realtime session reported has `status = nothing` and the event's JSON as
`response`), and
every `*CallError` carries the rendered `error` and, in
`cause`, the exception behind it when there is one (a `UniLMTimeout` for a timeout, a
`UniLMCancelled` for a cancellation). Fields of the five primary families:

```julia
LLMSuccess        (.message::Message, .self::Chat, .usage::Union{TokenUsage,Nothing}, .sse_dropped::Int, .model::Union{String,Nothing})
                  # model: the id the provider reports having served (OpenAI-wire, DeepSeek, Anthropic, Gemini modelVersion);
                  #   nothing where none is reported, and from an OllamaEndpoint; estimated_cost prices it before self.model
LLMFailure        (.response::String, .status::Int, .self::Chat, .request_id::Union{String,Nothing}, .sse_dropped::Int)
LLMCallError      (.error::String, .status::Union{Int,Nothing}, .self::Chat, .request_id::Union{String,Nothing}, .cause::Union{Nothing,Exception})
ResponseSuccess   (.response::ResponseObject, .sse_dropped::Int, .service::Union{Nothing,ServiceEndpointSpec})
ResponseFailure   (.response::String, .status::Int, .request_id::Union{String,Nothing}, .sse_dropped::Int)
ResponseCallError (.error::String, .status::Union{Int,Nothing}, .request_id::Union{String,Nothing}, .cause::Union{Nothing,Exception})
EmbeddingSuccess  (.embeddings::Embeddings, .usage::Union{TokenUsage,Nothing}, .raw::Dict{String,Any})
EmbeddingFailure  (.response::String, .status::Int)
EmbeddingCallError(.error::String, .status::Union{Int,Nothing}, .cause::Union{Nothing,Exception})
ImageSuccess      (.response::ImageResponse)
ImageFailure      (.response::String, .status::Int, .request_id::Union{String,Nothing})
ImageCallError    (.error::String, .status::Union{Int,Nothing}, .request_id::Union{String,Nothing}, .cause::Union{Nothing,Exception})
FIMSuccess        (.response::FIMResponse, .service::Union{Nothing,ServiceEndpointSpec})
FIMFailure        (.response::String, .status::Int, .request_id::Union{String,Nothing})
FIMCallError      (.error::String, .status::Union{Int,Nothing}, .request_id::Union{String,Nothing}, .cause::Union{Nothing,Exception})
```

`request_id` carries the provider's request id for support escalation — the
`x-request-id` header, else `request-id` (Anthropic). It exists on the Chat /
Responses / FIM / image failure and call-error types above, on the System One results
and on every platform `*Failure` except `RealtimeFailure` (a `RealtimeCallError`
carries one); the embedding limbs do not carry it.

`sse_dropped` (default `0`) counts the undecodable SSE `data:` payloads dropped
while assembling **that** streamed turn. It is `0` for a non-streamed call and
for a clean stream; anything higher means the result was built from an incomplete
wire — see [Streaming](@ref streaming_guide).

**Credentials never reach a result value.** API keys and request bodies are
redacted from the `error` strings these types carry, and a `*CallError`'s
`cause` is named by type rather than dumped when the result is displayed.

**Standard pattern-matching idiom**:

```julia
result = chatrequest!(chat)
if result isa LLMSuccess
    println(result.message.content)
elseif result isa LLMFailure
    @error "HTTP $(result.status): $(result.response)"
elseif result isa LLMCallError
    @error "Exception: $(result.error)"
end

result = respond("Hello")
if result isa ResponseSuccess
    println(output_text(result))
elseif result isa ResponseFailure
    @error "HTTP $(result.status)"
elseif result isa ResponseCallError
    @error result.error
end

result = generate_image("A cat")
if result isa ImageSuccess
    save_image(image_data(result)[1], "cat.png")
elseif result isa ImageFailure
    @error "HTTP $(result.status)"
elseif result isa ImageCallError
    @error result.error
end
```

---

## API Constants

```julia
const OPENAI_BASE_URL = "https://api.openai.com"
const CHAT_COMPLETIONS_PATH = "/v1/chat/completions"
const EMBEDDINGS_PATH = "/v1/embeddings"
const RESPONSES_PATH = "/v1/responses"
const IMAGES_GENERATIONS_PATH = "/v1/images/generations"
const COMPLETIONS_PATH = "/v1/completions"                     # FIM endpoint
const DEEPSEEK_BASE_URL = "https://api.deepseek.com"
const DEEPSEEK_BETA_BASE_URL = "https://api.deepseek.com/beta"
const GEMINI_CHAT_URL = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
```

## Exceptions

```julia
struct InvalidConversationError <: Exception
    reason::String
end
```

Thrown by `push!` (a non-system message onto an empty `Chat`, or a same-role repeat other than `tool`), by `pop!` on an empty `Chat`, and by `chatrequest!` before any request when `history=true` and the conversation ends with an assistant message. `issendvalid` never throws — it returns a `Bool`.

```julia
struct LLMResultError <: Exception
    result::LLMRequestResponse
end
```

Thrown by the result accessors `text`, `output_text`, `embedding_vectors`, `image_data` and `fim_text` when the result is not a success. `showerror` prints only the status and a trimmed (≤200-char) excerpt of the response body or error message — never the conversation, endpoint, or API key.

```julia
struct UniLMTimeout <: Exception
    phase::Symbol        # :connect | :request | :stream_idle | :deadline
    elapsed::Float64     # seconds, monotonic
    limit::Float64
end

struct MCPTimeoutError <: Exception
    phase::Symbol        # :connect | :queue (never acquired its session) | :request
    elapsed::Float64
    limit::Float64
    msg::String          # names the applicable timeout override
end

struct UniLMCancelled <: Exception
    source::Symbol       # :token (cancel! on a CancelToken) | :callback (stream close[] = true)
    elapsed::Float64     # seconds since the operation started
end

struct MCPSessionClosedError <: Exception
    cause::Symbol        # :disconnected | :timeout | :crash
    msg::String          # recovery guidance
end

struct MCPCrashError <: Exception
    msg::String                          # recovery guidance (embeds exit info)
    exitcode::Union{Int,Nothing}         # exit code if the server exited and was reaped in time
    termsignal::Union{Int,Nothing}       # signal number if the server was signal-killed
    cause::Union{Exception,Nothing}      # underlying transport exception; nothing on clean read-EOF
end
```

Raised when a `RequestConfig` bound is exceeded. Value-returning surfaces
(chat, embeddings, responses, images, FIM, System One, platform verbs) deliver
`UniLMTimeout` — and a cancellation's `UniLMCancelled` — inside their call-error
results; the MCP surface throws `MCPTimeoutError` (and `UniLMCancelled` for a cancelled
HTTP call). A stdio MCP server that exits or whose pipe breaks throws `MCPCrashError`
instead (the session closes with cause `:crash`), and a call on a closed session throws
`MCPSessionClosedError`.

---

## Public Extension API (not exported)

Declared public, not exported (19 names) — qualify them (`UniLM.get_url`). These
are the functions a new backend adds methods to and the stream-state types its
handlers mutate (see the Custom Backends guide): `get_url`, `auth_header`,
`default_model`, `encode_request`, `decode_response`, `handle_sse_event!`,
`StreamState`, `encode_agentic`, `decode_agentic`, `decode_agentic_stream`,
`AgenticStreamState`; and the System One protocol seam: `SystemOneProtocol`,
`TypeSafeProtocol`, `OpenAIDecisionsProtocol`, `OllamaSystemOneProtocol`,
`system_one_protocol`, `encode_system_one`, `decode_system_one`, `SystemOneWire`.

---

## Complete Exports List

Every exported symbol (401), grouped by area:

**Chat Completions**: `Chat`, `Message`, `ProviderContent`, `Attachment`, `ImageAttachment`, `AudioAttachment`, `DocumentAttachment`, `AudioOutput`, `RoleSystem`, `RoleUser`, `RoleAssistant`, `Tool`, `ToolCall`, `FunctionSignature`, `FunctionCallResult`, `ResponseFormat`, `InvalidConversationError`, `issendvalid`, `chatrequest!`, `update!`, `fork`
- *Legacy aliases* (pre-rename names, exported and non-breaking, retained until 1.0): `GPTTool` → `Tool`, `GPTToolCall` → `ToolCall`, `GPTFunctionSignature` → `FunctionSignature`, `GPTFunctionCallResult` → `FunctionCallResult`

**Responses API & Agentic**: `Respond`, `InputMessage`, `ResponseObject`, `ResponseSuccess`, `ResponseFailure`, `ResponseCallError`, `Reasoning`, `PromptCacheOptions`, `ModerationConfig`, `TextConfig`, `TextFormatSpec`, `respond`, `get_response`, `delete_response`, `cancel_response`, `list_input_items`, `compact_response`, `count_input_tokens`, `text_format`, `json_schema_format`, `json_object_format`
- *Input builders*: `input_text`, `input_image`, `input_file`, `configuration_update`
- *Tool types*: `ResponseTool`, `FunctionTool`, `WebSearchTool`, `FileSearchTool`, `MCPTool`, `ComputerTool`, `ImageGenerationTool`, `CodeInterpreterTool`, `ShellTool`, `ApplyPatchTool`, `CustomTool`
- *Tool constructors*: `function_tool`, `web_search`, `file_search`, `mcp_tool`, `computer_tool`, `image_generation_tool`, `code_interpreter`, `shell`, `apply_patch_tool`, `custom_tool`, `tool_result`, `mcp_approval_response`
- *Hosted Gemini tools*: `gemini_google_search`, `gemini_code_execution`, `gemini_url_context`
- *tool_choice builders*: `tool_choice_function`, `tool_choice_hosted`, `tool_choice_mcp`, `tool_choice_custom`, `tool_choice_allowed`
- *Result accessors*: `output_text`, `function_calls`, `refusals`, `reasoning_summaries`, `reasoning_items`, `url_citations`, `web_search_results`, `file_search_results`, `image_generation_results`, `code_interpreter_outputs`, `mcp_call_outputs`, `mcp_approval_requests`, `response_status`, `incomplete_details`, `usage_details`

**Image Generation & Edits**: `ImageGeneration`, `ImageEdit`, `ImageObject`, `ImageResponse`, `ImageSuccess`, `ImageFailure`, `ImageCallError`, `generate_image`, `edit_image`, `image_data`, `save_image`

**Embeddings**: `Embeddings`, `EmbeddingInput`, `embeddingrequest!`, `embedding_vectors`, `EmbeddingSuccess`, `EmbeddingFailure`, `EmbeddingCallError`

**Cost Tracking**: `TokenUsage`, `token_usage`, `estimated_cost`, `cumulative_cost`, `DEFAULT_PRICING`

**Service Endpoints**: `ServiceEndpoint`, `OpenAIWireEndpoint`, `ServiceEndpointSpec`, `OPENAIServiceEndpoint`, `AZUREServiceEndpoint`, `GEMINIServiceEndpoint`, `GEMINIOpenAIServiceEndpoint`, `ANTHROPICServiceEndpoint`, `GenericOpenAIEndpoint`, `OllamaEndpoint`, `OllamaOptions`, `MistralEndpoint`, `DeepSeekEndpoint`, `add_azure_deploy_name!`

**Provider model listings**: `ProviderModel`, `ModelListSuccess`, `ModelListFailure`, `ModelListCallError` (and `list_models` with `service=OPENAIServiceEndpoint`, `ANTHROPICServiceEndpoint` or `GEMINIServiceEndpoint`)

**Ollama model management**: `OllamaModel`, `OllamaModelInfo`, `OllamaRunningModel`, `OllamaPullProgress`, `OllamaSuccess`, `OllamaFailure`, `OllamaCallError`, `model_info`, `pull_model`, `running_models`, `load_model`, `unload_model`, `delete_model` (and `list_models` with `service=OllamaEndpoint()`)

**Provider Capabilities**: `provider_capabilities`, `has_capability`

**Tool Loop**: `CallableTool`, `ToolCallOutcome`, `ToolLoopResult`, `tool_loop!`, `tool_loop`, `to_tool`

**Forking**: `fork`

**MCP Client**: `MCPSession`, `MCPToolInfo`, `MCPToolResult`, `MCPResourceInfo`, `MCPPromptInfo`, `MCPServerCapabilities`, `MCPTransport`, `StdioTransport`, `HTTPTransport`, `MCPError`, `MCPCrashError`, `MCPSessionClosedError`, `mcp_connect`, `mcp_disconnect!`, `mcp_tools`, `mcp_tools_respond`, `list_tools!`, `list_resources!`, `list_prompts!`, `call_tool`, `read_resource`, `get_prompt`, `ping`

**MCP Server**: `MCPServer`, `MCPServerTool`, `MCPServerResource`, `MCPServerResourceTemplate`, `MCPServerPrompt`, `MCPServerPrimitive`, `register_tool!`, `register_resource!`, `register_resource_template!`, `register_prompt!`, `serve`, `@mcp_tool`, `@mcp_resource`, `@mcp_prompt`

**FIM / Completions**: `FIMCompletion`, `FIMChoice`, `FIMResponse`, `FIMSuccess`, `FIMFailure`, `FIMCallError`, `fim_complete`, `fim_text`, `prefix_complete`

**Files**: `FileUpload`, `FileObject`, `FileList`, `FileSuccess`, `FileListSuccess`, `FileContentSuccess`, `FileDeleteSuccess`, `FileFailure`, `FileCallError`, `upload_file`, `list_files`, `retrieve_file`, `delete_file`, `file_content`, `save_file_content`

**Vector Stores**: `VectorStoreObject`, `VectorStoreFileObject`, `VectorStoreFileBatch`, `VectorStoreList`, `VectorStoreSuccess`, `VectorStoreListSuccess`, `VectorStoreFileSuccess`, `VectorStoreBatchSuccess`, `VectorStoreDeleteSuccess`, `VectorStoreFailure`, `VectorStoreCallError`, `create_vector_store`, `retrieve_vector_store`, `list_vector_stores`, `delete_vector_store`, `add_vector_store_file`, `create_file_batch`, `retrieve_file_batch`, `poll_file_batch`, `vector_store_id`

**Conversations**: `ConversationObject`, `ConversationItem`, `ConversationItemList`, `ConversationSuccess`, `ConversationItemSuccess`, `ConversationItemListSuccess`, `ConversationDeleteSuccess`, `ConversationFailure`, `ConversationCallError`, `create_conversation`, `retrieve_conversation`, `update_conversation`, `delete_conversation`, `add_conversation_items`, `list_conversation_items`, `delete_conversation_item`, `conversation_id`

**Moderations**: `ModerationResponse`, `ModerationResult`, `ModerationSuccess`, `ModerationFailure`, `ModerationCallError`, `moderate`, `is_flagged`

**System One (TypeSafe's Jev and other backends)**: `TYPESAFEServiceEndpoint`, `SystemOneQuestion`, `ChoiceQuestion`, `ScoreQuestion`, `NoulQuestion`, `NoulCriteria`, `BoxQuestion`, `RawQuestion`, `choice`, `score`, `noul`, `box`, `SystemOneRequest`, `ask`, `ask_each`, `with_system_one`, `SystemOneAnswer`, `ChoiceAnswer`, `ScoreAnswer`, `NoulAnswer`, `BoxCoordinates`, `BoxAnswer`, `RefusalAnswer`, `UnknownAnswer`, `SystemOneResponse`, `SystemOneSuccess`, `SystemOneFailure`, `SystemOneCallError`, `SystemOneError`, `answers`, `answer`, `TypeSafeModelCard`, `TypeSafeModelsSuccess`, `list_models`
- *Natural-language control flow*: `Meaning`, `@nl_str`, `@branch`, `nl_dispatch`, `nl_classify`, `meanings`, `meaning_gaps`, `LowConfidenceError`, `DecisionDeclinedError`
- *Recorded answers*: `with_recorded_answers`, `ReplayMissError`, `RecordingWriteError`

**Audio**: `SpeechRequest`, `TranscriptionRequest`, `SpeechSuccess`, `TranscriptionSuccess`, `AudioFailure`, `AudioCallError`, `speak`, `save_audio`, `transcribe`, `translate`, `transcript_text`

**Batch**: `BatchObject`, `BatchList`, `BatchSuccess`, `BatchListSuccess`, `BatchFailure`, `BatchCallError`, `create_batch`, `retrieve_batch`, `cancel_batch`, `list_batches`, `poll_batch`

**Fine-tuning**: `FineTuningJob`, `FineTuningList`, `FineTuningSuccess`, `FineTuningListSuccess`, `FineTuningFailure`, `FineTuningCallError`, `create_fine_tuning_job`, `retrieve_fine_tuning_job`, `cancel_fine_tuning_job`, `list_fine_tuning_jobs`, `list_fine_tuning_events`, `list_fine_tuning_checkpoints`

**Containers**: `ContainerObject`, `ContainerList`, `ContainerSuccess`, `ContainerListSuccess`, `ContainerDeleteSuccess`, `ContainerFailure`, `ContainerCallError`, `create_container`, `retrieve_container`, `list_containers`, `delete_container`, `add_container_file`

**Uploads**: `UploadObject`, `UploadPartObject`, `UploadSuccess`, `UploadPartSuccess`, `UploadFailure`, `UploadCallError`, `create_upload`, `add_upload_part`, `complete_upload`, `cancel_upload`

**Webhooks**: `WebhookEvent`, `WEBHOOK_EVENTS`, `verify_webhook`, `parse_webhook`

**Realtime**: `RealtimeSession`, `RealtimeSecretSuccess`, `RealtimeFailure`, `RealtimeCallError`, `mint_realtime_secret`, `realtime_connect`, `realtime_send`, `realtime_receive`, `realtime_event`, `session_update`, `input_audio_append`, `response_create`

**Result Types (base)**: `LLMRequestResponse`, `LLMSuccess`, `LLMFailure`, `LLMCallError`, `LLMResultError`, `issuccess`, `isfailure`, `text`, `reasoning_text`

**Request Config, Timeouts & Cancellation**: `RequestConfig`, `current_config`, `with_request_config`, `set_default_config!`, `UniLMTimeout`, `MCPTimeoutError`, `UniLMCancelled`, `CancelToken`, `cancel!`, `iscancelled`, `with_cancel`
