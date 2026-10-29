# [Mistral & OpenAI-compatible services](@id openai_compatible_start)

```@setup openai_compatible
using UniLM
```

Mistral makes the Mistral models — Mistral Large, Codestral, Mistral Embed, the Voxtral audio models
and others — and serves them through its own API. `MistralEndpoint()` is a `GenericOpenAIEndpoint` on
Mistral's host, `https://api.mistral.ai`: UniLM speaks Mistral's chat, embeddings, FIM and
transcription routes as OpenAI-compatible ones, and a hosted service of another vendor with an
OpenAI-compatible API is reached the same way, with `GenericOpenAIEndpoint(base_url, key)`. The rules
on this page are that endpoint's; the facts marked as Mistral's are Mistral's alone. UniLM holds no
rule of Mistral's or of any other service: it sends every field a request sets as it is, under its own
name but for those [First call](@ref openai_compatible_first_call) lists, and sends the requests of the
capabilities the table below lists without knowing whether the service serves them. A
verb that needs a capability the table lists as not available refuses the endpoint, and so does `ask`;
`respond`, which takes `responses` or `agentic`, and `transcribe`, which takes `transcription` or
`audio`, are sent. Of Mistral's API, UniLM does not use the Agents and Conversations APIs, OCR or the
classifiers either. Servers of your own have their entry: [vLLM, LM Studio & llama.cpp](@ref local_servers_start).
No Mistral key was at hand for this page: its examples are not run, and their code stands without
output.

```@eval
Main.ProviderFacts.at_a_glance(:openai_compatible)
```

| I want to… | Use | You get |
| :--- | :--- | :--- |
| [set my key, or point UniLM at another service](@ref openai_compatible_setup) | `MistralEndpoint()`, `GenericOpenAIEndpoint(base_url, key)` | requests to the service's `/v1/…` routes |
| [make a first call](@ref openai_compatible_first_call) | `chatrequest!(chat)`, with a `model` | an `LLMSuccess`: the reply and its usage |
| [stream a reply](@ref openai_compatible_streaming) | `stream=true` and a `callback` | the text as it arrives, then the whole message |
| [let the model call my functions](@ref openai_compatible_tools) | `tools` and `tool_loop!` | the calls run and answered until the model replies |
| [get JSON that fits a schema](@ref openai_compatible_structured_output) | a JSON Schema `response_format` | a reply constrained to the schema, where the service enforces it |
| [embed texts](@ref openai_compatible_embeddings) | `Embeddings` and `embeddingrequest!` | one vector per text |
| [transcribe a recording](@ref openai_compatible_audio) | `transcribe` | the transcript |
| [fill in the middle of code](@ref openai_compatible_completions) | `fim_complete` | the completion between a prefix and a suffix |
| [price the calls](@ref openai_compatible_cost) | `estimated_cost` with a price row of your own | USD estimates, or `0.0` without a row |
| [choose a model](@ref openai_compatible_models) | the model's name | the model, with no default |
| [handle a failure](@ref openai_compatible_failures) | the result's type | the status or the exception, and what to do |

## [Setup](@id openai_compatible_setup)

Set your Mistral API key in the environment Julia starts from:

```bash
export MISTRAL_API_KEY="..."
```

`MistralEndpoint()` is `GenericOpenAIEndpoint("https://api.mistral.ai", "", "MISTRAL_API_KEY")`, and
prints as that call. It reads the variable each time a request is built, so the endpoint can be made
before the key is set, and a call without it returns a call-error result whose `cause` is the
`KeyError` naming it; `MistralEndpoint(api_key="…")` sends the key it is given instead. The key goes
as `Authorization: Bearer`, the header Mistral documents
([chat completions](https://docs.mistral.ai/api/endpoint/chat)). `MistralEndpoint` is a constructor
with no type of its own: an error names `GenericOpenAIEndpoint`, and code cannot dispatch on Mistral.

Another service's endpoint takes the root of its API and a key, given or read from a variable:

```julia
mistral = MistralEndpoint()
service = GenericOpenAIEndpoint("https://api.example.com", "", "EXAMPLE_API_KEY")
```

Under the root, without `/v1`, UniLM sends a `Chat` to `/v1/chat/completions`, an `Embeddings` request
to `/v1/embeddings`, `transcribe` to `/v1/audio/transcriptions` and `respond` to `/v1/responses`, and
drops a trailing `/`; `fim_complete` goes to Mistral's `/v1/fim/completions` on Mistral's host, and to
`/v1/completions` on any other. A service whose routes are not those cannot be reached through this
endpoint; a backend of your own can reach it ([Write your own backend](@ref custom_backends_guide)). A key
given as `""`, or a variable holding `""`, sends no `Authorization` header. The endpoint has no default
model: every request names the model the service serves under that name
([Models](@ref openai_compatible_models)), or is refused ([Limits and refusals](@ref openai_compatible_limits)).

## [First call](@id openai_compatible_first_call)

A `Chat` starts with its system message; `chatrequest!` sends it and returns a typed result, and `text`
reads the reply — or throws, on a failure:

```julia
chat = Chat(service=MistralEndpoint(), model="mistral-large-latest")
push!(chat, Message(Val(:system), "You are a concise assistant."))
push!(chat, Message(Val(:user), "In one sentence: what is a prime number?"))
result = chatrequest!(chat)
text(result)
result.model, result.message.finish_reason, result.usage
```

The reply is read as a Chat Completions reply: `result.model` is the model the service reports, and
`finish_reason` and `usage` are read as the OpenAI-compatible API defines them. No Mistral model has a
price row, so the first `chatrequest!` logs a warning and the call costs `0.0`
([Cost and usage](@ref openai_compatible_cost)).

A content that is a list of chunks, the shape Mistral's reasoning models answer in with
`reasoning_effort="high"` — a `thinking` chunk, then the answer's `text` chunks
([reasoning](https://docs.mistral.ai/studio/conversations/reasoning)) — is read as its `text` chunks
joined, and its `thinking` chunks are kept apart as the turn's reasoning, which
[`reasoning_text`](@ref) reads, streamed or not ([Streaming](@ref openai_compatible_streaming)). A
reply's `reasoning_content` or `reasoning` field, in which other servers return the reasoning, is kept
the same way ([vLLM, LM Studio & llama.cpp](@ref local_servers_start)). No reasoning goes back with
the next request.

Fields go out under the names of the `Chat`, but on Mistral's host a `max_completion_tokens` goes out
as `max_tokens`, taken over a `max_tokens` also set, the name Mistral's chat reference gives the cap
([chat completions](https://docs.mistral.ai/api/endpoint/chat)), and a `seed` as `random_seed`, "The
seed to use for random sampling", the name it gives the seed; any other service gets both as the `Chat`
names them. Mistral's `safe_prompt`, `min_tokens`, `prompt_mode` and `guardrails` are not `Chat` fields
([chat completions](https://docs.mistral.ai/api/endpoint/chat)). What Mistral does with a field its
reference does not list is untested.

## [Conversations](@id openai_compatible_conversations)

A `Chat` keeps the conversation: each reply is appended to it, and the next `chatrequest!` sends the
whole conversation again; a system message pushed after the conversation started is sent in place.

```julia
push!(chat, Message(Val(:user), "And a composite number?"))
text(chatrequest!(chat))
```

A reply's reasoning never goes back with a later turn: an unstreamed reply keeps none
([First call](@ref openai_compatible_first_call)), and the reasoning a streamed one keeps is not sent. What `push!` refuses is the same on every provider ([Chat Completions](@ref chat_guide)).

## [Streaming](@id openai_compatible_streaming)

With `stream=true`, `chatrequest!` returns a `Task` at once, and `fetch` gives the same typed result as
an unstreamed call; the `callback` receives the text as it arrives, then the assembled `Message`:

```julia
chat = Chat(service=MistralEndpoint(), model="mistral-large-latest", stream=true)
push!(chat, Message(Val(:system), "You are a concise assistant."))
push!(chat, Message(Val(:user), "List the planets of the solar system in order, one per line."))
task = chatrequest!(chat; callback=(chunk, close) -> chunk isa String && print(chunk))
text(fetch(task))
```

A `Chat` on this endpoint sends no `stream_options` unless it sets them, a field Mistral's chat
reference does not list; whether Mistral's stream reports its usage unasked is untested, and a stream
without usage adds `0.0` to `cumulative_cost`. The stream is the Chat Completions one: the turn is final
at `data: [DONE]`, or, without one, when the HTTP body ends after the finish reason; `on_tool_call(tc)`
runs once per call whose arguments decode, as the next call begins or at the end of the stream; and
setting `close[] = true` stops it ([Stopping a Stream Early](@ref streaming_stop)).

A delta whose `content` is a list of chunks is read chunk by chunk. Mistral streams a reasoning reply's
thinking in such lists, then "a single list containing both a closing `ThinkChunk` and the first
`TextChunk`", then the answer as strings
([reasoning](https://docs.mistral.ai/studio/conversations/reasoning)): the `text` chunks reach the
callback and the final `Message` in order, and the `thinking` chunks are kept apart as the turn's
reasoning, which [`reasoning_text`](@ref) reads once the finish reason or `data: [DONE]` has arrived; a
stream cut before then keeps none. No thinking reaches the callback. The concepts every provider shares
are on [Streaming](@ref streaming_guide).

## [Tools](@id openai_compatible_tools)

Each `Tool` in `tools` goes out as a function tool, and `tool_choice`, `parallel_tool_calls` and a
tool's `strict` go as they are set; a `Chat` with tools sends `parallel_tool_calls: false` unless it
sets another value, where Mistral's own default is `true`. Mistral's `tool_choice` takes `"auto"`,
`"none"`, `"any"`, `"required"` or a function: "`any` or `required` means the model must call one or
more tools" ([chat completions](https://docs.mistral.ai/api/endpoint/chat)). `tool_loop!` runs the
calls and sends their results back until the model answers in text:

```julia
weather = UniLM.CallableTool(
    Tool(func=FunctionSignature(name="get_weather", description="Current weather in a city",
        parameters=Dict("type" => "object", "required" => ["city"],
                        "properties" => Dict("city" => Dict("type" => "string"))))),
    (name, args) -> "18°C and sunny in $(args["city"])")
chat = Chat(service=MistralEndpoint(), model="mistral-large-latest", tools=[weather])
push!(chat, Message(Val(:system), "You are concise."))
push!(chat, Message(Val(:user), "What is the weather in Lyon? Use the tool, then answer in one sentence."))
loop = tool_loop!(chat; tools=[weather])
text(loop.response)
```

The loop every provider shares is on [Tool Calling](@ref tools_guide).

## [Structured output](@id openai_compatible_structured_output)

A JSON Schema `response_format` goes out as `{"type": "json_schema", "json_schema": {"name": …,
"schema": …}}`, and `UniLM.json_object()` as `{"type": "json_object"}`; the service constrains the
reply. On Mistral, `json_schema` "guarantees the message the model generates is in JSON and follows the
schema you provide", and with `json_object` "you MUST also instruct the model to produce JSON yourself
with a system or a user message" ([chat completions](https://docs.mistral.ai/api/endpoint/chat)):

```julia
using JSON
schema = Dict("type" => "object", "additionalProperties" => false, "required" => ["name", "year"],
              "properties" => Dict("name" => Dict("type" => "string"), "year" => Dict("type" => "integer")))
chat = Chat(service=MistralEndpoint(), model="mistral-large-latest",
            response_format=UniLM.json_schema("person", "A person", schema))
push!(chat, Message(Val(:system), "You extract facts."))
push!(chat, Message(Val(:user), "Extract the person: Ada Lovelace, born 1815."))
JSON.parse(text(chatrequest!(chat)))
```

The formats every provider shares are on [Structured Output](@ref structured_guide).

## [Attachments](@id openai_compatible_media)

A user message's attachments go out as content parts: an image as an `image_url` part, with its
`detail` when one is set, a sound clip as an `input_audio` part, and a document as a `file` part
holding its name and a data URL. The service decides what it reads, and UniLM checks none of it before
sending. The attachments every provider shares are on [Attachments](@ref attachments_guide).

## [Embeddings](@id openai_compatible_embeddings)

An `Embeddings` request goes to `/v1/embeddings` with the texts, the model and the fields it sets:

```julia
emb = Embeddings(["first text", "second text"]; service=MistralEndpoint(), model="mistral-embed")
r = embeddingrequest!(emb)
length(embedding_vectors(r)), length(first(embedding_vectors(r)))
```

Mistral's embeddings route takes `input`, `model`, `output_dimension`, `output_dtype`,
`encoding_format` and `metadata` ([embeddings](https://docs.mistral.ai/api/endpoint/embeddings)). On
Mistral's host an `Embeddings` request's `dimensions` goes out as `output_dimension`, "The dimension of
the output embeddings when feature available", and on any other service as `dimensions`; its effect on
Mistral is untested. Each item is text: an image or a sound clip is refused before the request
([Limits and refusals](@ref openai_compatible_limits)). The concepts every provider shares are on
[Embeddings](@ref embeddings_guide).

## [Audio](@id openai_compatible_audio)

`transcribe` sends the recording and the fields a request sets to `/v1/audio/transcriptions`, as a
multipart form, and reads a JSON reply's `text`:

```julia
heard = transcribe("memo.wav"; service=MistralEndpoint(), model="voxtral-mini-latest", language="en")
transcript_text(heard)
```

Mistral's route takes the `file`, a `model` it marks required, a `language` ("Providing the language
can boost accuracy"), `temperature`, `timestamp_granularities`, `context_bias` and `diarize`
([transcriptions](https://docs.mistral.ai/api/endpoint/audio/transcriptions)); a request's `prompt`
and `response_format`, which that reference does not list, go out as set, untested. The endpoint has no
default transcription model, so a call names one ([Limits and refusals](@ref openai_compatible_limits)).
`transcribe` is sent again after a transient status, as a `Chat` is, whole each time
([When something fails](@ref openai_compatible_failures)).

## [FIM completion](@id openai_compatible_completions)

On Mistral's host, `fim_complete` goes to `/v1/fim/completions`, Mistral's fill-in-the-middle route,
and reads each choice's `message.content`, the chat shape Mistral answers in; on another host it goes
to `/v1/completions` and reads each choice's `text`. A request names a model with FIM, Codestral on
Mistral:

```julia
fim = fim_complete("def fib(n):"; suffix="    return fib(n - 1) + fib(n - 2)",
                   service=MistralEndpoint(), model="codestral-latest", max_tokens=64)
fim_text(fim)
```

Mistral's FIM route takes `model`, `prompt`, `suffix`, `max_tokens`, `min_tokens`, `stop`,
`temperature`, `top_p`, `random_seed`, `stream`, `prompt_cache_key` and `metadata`
([FIM](https://docs.mistral.ai/api/endpoint/fim)). A `FIMCompletion` sends `max_tokens` 128 unless it
sets another value, has no streaming path ([Limits and refusals](@ref openai_compatible_limits)), and
sends its `echo`, `logprobs`, `frequency_penalty` and `presence_penalty` only when it sets them, fields
that reference does not list. The concepts every provider shares are on
[FIM & Prefix Completion](@ref completions_guide).

## [Limits and refusals](@id openai_compatible_limits)

UniLM checks no limit of a service: a request longer than the model's context, or one the service
cannot serve, is the service's error ([When something fails](@ref openai_compatible_failures)). Every
field goes out as it is set; the checks every request gets still hold, such as a `temperature` from 0 to
2, though Mistral's error glossary gives "`temperature` outside 0 to 1 range" as a cause of HTTP 400
([error glossary](https://docs.mistral.ai/resources/error-glossary)). `respond` sends a `Respond` to
`/v1/responses`, its `seed` among its fields, while Mistral's API reference lists no Responses route. It
refuses before anything is sent what the table below lists; the verbs of the capabilities the table at
the top of this page lists as not available refuse the endpoint too.

| Refused | Why: the message's words | What to do |
| :--- | :--- | :--- |
| a `Chat`, `Respond`, `Embeddings` or FIM request without a `model` | `model must be specified`: the endpoint has no default model | the model's name ([Models](@ref openai_compatible_models)) |
| a `transcribe` call without a `model` | `has no default transcription model` | the model's name, such as `voxtral-mini-latest` |
| an `Embeddings` item holding an image or a sound clip | `takes text only` | the item's text alone |
| an `Embeddings` request with a `title`, for a model other than EmbeddingGemma and gemini-embedding-2 | `has no task prompts, so a title has nowhere to go` | no title |
| an `Embeddings` request with a `task` other than `:none`, for a model other than EmbeddingGemma and gemini-embedding-2 | `UniLM knows the task prompts of EmbeddingGemma models and gemini-embedding-2 only` | `task = :none`, with the model's own prefix in the text |
| a `FIMCompletion` with `stream=true` | `does not support stream=true`: `fim_complete` has no streaming path | leave `stream` unset |

## [Cost and usage](@id openai_compatible_cost)

A result is priced by the model id it reports, whatever the service. No Mistral model has a row in
[`DEFAULT_PRICING`](@ref): its [`estimated_cost`](@ref) is `0.0`, with one warning per model id, its
embeddings' too, and [`cumulative_cost`](@ref) adds `0.0`. A model another service reports is priced at
its id's row where `DEFAULT_PRICING` has one, as a priced provider's model served through another
service is, and at `0.0` otherwise. Give the model's rates, USD per token, as `pricing`, or add the row
to `DEFAULT_PRICING`; Mistral lists Mistral Large 3, `mistral-large-2512`, at "Input \$0.5 /M Tokens"
and "Output \$1.5 /M Tokens" ([API pricing](https://docs.mistral.ai/inference/pricing)), and "Cached
tokens are billed at 10% of the standard input token price"
([chat completions](https://docs.mistral.ai/api/endpoint/chat)):

```julia
large = (input=0.5e-6, cached_input=0.05e-6, output=1.5e-6)
estimated_cost(result; pricing=Dict(result.model => large))
DEFAULT_PRICING[result.model] = large                  # cumulative_cost prices with it from now on
```

The row prices the model the reply reports. A row holds no audio rates, so a result whose usage counts
audio tokens is estimated at `0.0`, with a warning, even then. A transcription, which Mistral prices per
minute of audio, carries no token usage: `estimated_cost` throws `ArgumentError` for it. How a call is
priced on every provider is on [Cost Tracking](@ref cost_guide).

## [When something fails](@id openai_compatible_failures)

`chatrequest!`, `embeddingrequest!`, `fim_complete` and `transcribe` return what the service or the
network did as a typed result, and do not throw it ([Results](@ref results_guide)). A failure holds the
service's HTTP `status` and, as `response`, the body that came with it; a call error holds the text of
an exception as `error` and the exception as `cause`. UniLM sends a request again after a transient
status — 408, 429, 500, 502 to 504, 520 to 524 and 529 — or a failed connection, as often as the call's
[`RequestConfig`](@ref) allows, and then returns the last answer ([Timeouts & Retries](@ref timeouts_guide));
a streamed request is sent again only while none of its callbacks has run. The statuses below are Mistral's ([error glossary](https://docs.mistral.ai/resources/error-glossary)):

| What you see | Why | What to do |
| :--- | :--- | :--- |
| `ArgumentError` from the call, nothing sent | the request breaks a rule UniLM checks before sending ([Limits and refusals](@ref openai_compatible_limits)) | what its row says |
| a call error — an `LLMCallError`, `EmbeddingCallError`, `FIMCallError` or `AudioCallError` — whose `cause` is a `KeyError` | `MISTRAL_API_KEY`, or the variable the endpoint reads, is not set | set it ([Setup](@ref openai_compatible_setup)) |
| a failure with `status` 401 | "The API key is missing, invalid, or expired" | the key |
| a failure with `status` 402 | "A billing condition on your Organization blocks the request" | what the message's link says |
| a failure with `status` 403 or 404 | a model your subscription does not include, or "Invalid model ID or endpoint path"; not sent again | the model's name, or the route |
| a failure with `status` 400 or 422 | "The request body is malformed or missing required fields", or it "contains invalid parameter values"; not sent again | what the message says |
| a failure with `status` 429 | "You have exceeded the rate limit for your subscription tier"; sent again until the attempts or the deadline run out | fewer requests, or a higher tier |
| a failure with `status` 500 or another transient status | a fault on the service, sent again until the attempts or the deadline run out | the request again, later |
| an `LLMFailure` with `status` 200 from a stream | the stream ended before a finish reason; `response` holds what arrived, and nothing is appended | the turn again |
| a call error with `status` `nothing` and a `UniLMTimeout` as `cause` | a bound of the call's `RequestConfig` ran out | a longer bound, or a stream for a long reply ([Timeouts & Retries](@ref timeouts_guide)) |
| a call error whose `cause` is a `UniLMCancelled` | a stop: the stream's `close[] = true` or a cancelled `CancelToken` | nothing, if the stop was yours |
| a call error whose `cause` is the exception your `callback` or `on_tool_call` threw | the exception ended the stream; it is not sent again, and nothing is appended; a thrown value that is no `Exception` leaves `cause` `nothing`, its text in `error`; an `InterruptException`, or a recorded call's `ReplayMissError` or `RecordingWriteError`, is thrown at `fetch` instead | the callback |
| `LLMResultError` from `text`, `embedding_vectors` or `fim_text`, or a `MethodError` from `transcript_text` | the result is a failure, which holds no reply | check `issuccess(result)` first |

## [Models](@id openai_compatible_models)

The endpoint has no default model: a request names one by the name the service serves it under. On
Mistral, many API names are dated, such as `mistral-large-2512` (Mistral Large 3), `codestral-2508` or
`mistral-embed-2312`, while some of the newest are not, such as `mistral-medium-3-5` and
`mistral-large-4`, and "A deprecated model still answers requests until its retirement date. A retired
model no longer does: use its replacement" ([models](https://docs.mistral.ai/models)); the examples of
Mistral's API reference name aliases, such as `mistral-large-latest`, `codestral-latest`,
`mistral-embed` and `voxtral-mini-latest`. Mistral lists its models at `GET /v1/models`
([models API](https://docs.mistral.ai/api/endpoint/models)).

Checked against Mistral's API documentation of 25 October 2026 with UniLM's development branch after
0.26.0; nothing on this page was run.

See also: [`GenericOpenAIEndpoint`](@ref), [`MistralEndpoint`](@ref), [`Chat`](@ref),
[`chatrequest!`](@ref), [`fim_complete`](@ref) and [`LLMSuccess`](@ref) in the API reference.
