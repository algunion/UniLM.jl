# [vLLM, LM Studio & llama.cpp](@id local_servers_start)

```@setup local_servers
using UniLM
```

vLLM, LM Studio and llama.cpp's `llama-server` serve open models from your own machine, or a server of
yours, through an OpenAI-compatible API, and `GenericOpenAIEndpoint` speaks it: under the server's
base URL, a `Chat` goes to `/v1/chat/completions`, an `Embeddings` request to `/v1/embeddings`,
`respond` to `/v1/responses` and `fim_complete` to `/v1/completions`. UniLM is an OpenAI-compatible
client there and no more: it holds no rule of any server or model, sends every field a request sets as
it is, and refuses only what its OpenAI-compatible requests have no field for. No verb of a capability
the table below lists as not available sends a request through this endpoint, `transcribe` among
them, though vLLM serves transcription at `/v1/audio/transcriptions`; UniLM does not use the servers'
own APIs either, such as `llama-server`'s `/completion` and `/infill` or LM Studio's REST API. The
examples on this page ran on `llama-server`.

```@eval
Main.ProviderFacts.at_a_glance(:local_servers)
```

| I want to… | Use | You get |
| :--- | :--- | :--- |
| [point UniLM at my server](@ref local_servers_setup) | `GenericOpenAIEndpoint(base_url, "")` | requests to the server's `/v1/…` routes, with no key |
| [make a first call](@ref local_servers_first_call) | `chatrequest!(chat)` | an `LLMSuccess`: the reply, its usage and the model the server names |
| [stream a reply](@ref local_servers_streaming) | `stream=true`, a `callback` and `stream_options` | the text as it arrives, then the whole message and its usage |
| [let the model call my functions](@ref local_servers_tools) | `tools` and `tool_loop!` | the calls run and answered until the model replies |
| [get JSON that fits a schema](@ref local_servers_structured_output) | a JSON Schema `response_format` | a reply constrained to the schema, where the server enforces it |
| [embed texts, use respond, or fill in the middle](@ref local_servers_other_apis) | `Embeddings`, `respond`, `fim_complete` | what the server serves at those routes |
| [handle a failure](@ref local_servers_failures) | the result's type | the status or the exception, and what to do |

## [Setup](@id local_servers_setup)

Start the server with the model it serves:

```bash
vllm serve Qwen/Qwen3-8B                       # vLLM, at http://localhost:8000
lms server start                               # LM Studio, at http://localhost:1234
llama-server -m model.gguf --alias d1-3b       # llama.cpp, at http://127.0.0.1:8080
```

vLLM's examples reach its server at `http://localhost:8000/v1`
([OpenAI-compatible server](https://docs.vllm.ai/en/latest/serving/online_serving/openai_compatible_server/)),
LM Studio's assume "the server port is `1234`"
([OpenAI compatibility](https://lmstudio.ai/docs/developer/openai-compat)), and `llama-server` "by default
listens on `127.0.0.1:8080`" ([server](https://github.com/ggml-org/llama.cpp/blob/b9730/tools/server/README.md)).
A `GenericOpenAIEndpoint` takes the server's base URL and a key:

```julia
vllm = GenericOpenAIEndpoint("http://localhost:8000", "")
lmstudio = GenericOpenAIEndpoint("http://localhost:1234", "")
llama = GenericOpenAIEndpoint("http://127.0.0.1:8080", "")
```

The base URL is the server's root, without `/v1`: UniLM appends `/v1/chat/completions` and the other
paths, and drops a trailing `/`. An empty key sends no `Authorization` header, which suits the servers
as they start: "By default, LM Studio does not require authentication for API requests"
([authentication](https://lmstudio.ai/docs/developer/core/authentication)), and vLLM and `llama-server`
check a key only when started with one, `--api-key` (vLLM also `VLLM_API_KEY`). A key given, `GenericOpenAIEndpoint(url, "sk-…")`, goes
as `Authorization: Bearer`; with `GenericOpenAIEndpoint(url, "", "MY_SERVER_KEY")` UniLM reads the key
from that environment variable each time a request is built, and a request without it fails, naming the
variable.

The endpoint has no default model: a `Chat`, `Respond`, `Embeddings` or FIM request names the model the
server serves, or throws `ArgumentError` before anything is sent. vLLM serves a model under the name
`vllm serve` was given, LM Studio under the identifier of a loaded model, and `llama-server` under its
`--alias`, else the path given with `-m`: "By default, model `id` field is the path to model file"
([server](https://github.com/ggml-org/llama.cpp/blob/b9730/tools/server/README.md)). `list_models`
refuses this endpoint, so read the server's own `/v1/models`.

## [First call](@id local_servers_first_call)

A `Chat` starts with its system message; `chatrequest!` sends it and returns a typed result, and `text`
reads the reply — or throws, on a failure:

```@example local_servers
llama = GenericOpenAIEndpoint("http://127.0.0.1:8080", "")
chat = Chat(service=llama, model="d1-3b")
push!(chat, Message(Val(:system), "You are a concise assistant."))
push!(chat, Message(Val(:user), "In one sentence: what is a prime number?"))
result = chatrequest!(chat)
text(result)
```

```@example local_servers
result.model, result.message.finish_reason
```

```@example local_servers
u = result.usage
(; u.prompt_tokens, u.completion_tokens, cost = estimated_cost(result))
```

`result.model` is the model the server names in its reply, and the reply's `finish_reason` and
`usage` are read as the OpenAI-compatible API defines them. A model has no price row unless you add
one, so [`estimated_cost`](@ref) gives `0.0` and logs one warning per model id — the first
`chatrequest!` logs it, as it adds each call's cost to `cumulative_cost`. Pass `estimated_cost` a
`pricing` table holding the model's row, or add the row to [`DEFAULT_PRICING`](@ref), to price your own
server's time ([Cost Tracking](@ref cost_guide)); a result whose usage counts audio tokens is estimated at
`0.0` with a warning even then, since a row holds no audio rates.

A reasoning model's reasoning is not read: the servers return it beside the answer — `llama-server`
"via the `reasoning_content` field"
([server](https://github.com/ggml-org/llama.cpp/blob/b9730/tools/server/README.md)), vLLM in a
`reasoning` field, which "used to be called `reasoning_content`"
([reasoning outputs](https://docs.vllm.ai/en/latest/features/reasoning_outputs/)) — and UniLM keeps
neither: [`reasoning_text`](@ref) gives `nothing`, and the reasoning does not go back with the next
turn. The result types are on [Results](@ref results_guide).

## [Streaming](@id local_servers_streaming)

With `stream=true`, `chatrequest!` returns a `Task` at once, and `fetch` gives the same typed result as
an unstreamed call; the `callback` receives the text as it arrives, then the assembled `Message`. A
`Chat` on this endpoint sends no `stream_options` unless it sets them, so a streamed turn reports no
usage, and adds `0.0` to `cumulative_cost`, unless the `Chat` sets
`stream_options=Dict("include_usage" => true)`, which `llama-server` honoured below:

```@example local_servers
chat = Chat(service=llama, model="d1-3b", stream=true, stream_options=Dict("include_usage" => true))
push!(chat, Message(Val(:system), "You are a concise assistant."))
push!(chat, Message(Val(:user), "List the planets of the solar system in order, one per line."))
deltas = String[]
task = chatrequest!(chat; callback=(chunk, close) -> chunk isa String && push!(deltas, chunk))
result = fetch(task)
text(result)
```

```@example local_servers
length(deltas), result.usage.completion_tokens
```

The stream is the OpenAI-compatible one: the turn is final at `data: [DONE]`, or, without one, when the
HTTP body ends after the finish reason; `on_tool_call(tc)` runs once per call whose arguments decode, as
the next call begins or at the end of the stream; and setting `close[] = true` stops it
([Stopping a Stream Early](@ref streaming_stop)). The concepts every provider
shares are on [Streaming](@ref streaming_guide).

## [Tools](@id local_servers_tools)

Each `Tool` in `tools` goes out as a function tool, and `tool_choice`, `parallel_tool_calls` and a
tool's `strict` go as they are set; a `Chat` with tools sends `parallel_tool_calls: false` unless it
sets another value. What the server does with them is its own: vLLM lets the model choose a call only
when started with `--enable-auto-tool-choice` and a `--tool-call-parser`
([tool calling](https://docs.vllm.ai/en/latest/features/tool_calling/)), `llama-server` takes tools
through its Jinja chat templates, which its `--jinja` flag turns on, "default: enabled"
([server](https://github.com/ggml-org/llama.cpp/blob/b9730/tools/server/README.md)), and on it
"Multiple/parallel tool calling is supported on some models but disabled by default"
([function calling](https://github.com/ggml-org/llama.cpp/blob/b9730/docs/function-calling.md)),
and LM Studio warns that "Smaller models and models that were not trained for tool use may output
improperly formatted tool calls" ([tool use](https://lmstudio.ai/docs/developer/openai-compat/tools)).
`tool_loop!` runs the calls and sends their results back until the model answers in text:

```@example local_servers
weather = UniLM.CallableTool(
    Tool(func=FunctionSignature(name="get_weather", description="Current weather in a city",
        parameters=Dict("type" => "object", "required" => ["city"],
                        "properties" => Dict("city" => Dict("type" => "string"))))),
    (name, args) -> "18°C and sunny in $(args["city"])")
chat = Chat(service=llama, model="d1-3b", tools=[weather])
push!(chat, Message(Val(:system), "You are concise."))
push!(chat, Message(Val(:user), "What is the weather in Lyon? Use the tool, then answer in one sentence."))
loop = tool_loop!(chat; tools=[weather])
calls = [(o.tool_name, o.arguments) for o in loop.tool_calls]
loop.completed, calls, text(loop.response)
```

The loop every provider shares is on [Tool Calling](@ref tools_guide).

## [Structured output](@id local_servers_structured_output)

A JSON Schema `response_format` goes out as the OpenAI-compatible API defines it,
`{"type": "json_schema", "json_schema": {"name": …, "schema": …}}`, and `UniLM.json_object()` as
`{"type": "json_object"}`; the server constrains the reply. vLLM takes a `json_schema`
`response_format` ([structured outputs](https://docs.vllm.ai/en/latest/features/structured_outputs/)),
LM Studio a JSON schema on `/v1/chat/completions`
([structured output](https://lmstudio.ai/docs/developer/openai-compat/structured-output)), and
`llama-server` "schema-constrained JSON"
([server](https://github.com/ggml-org/llama.cpp/blob/b9730/tools/server/README.md)):

```@example local_servers
using JSON
schema = Dict("type" => "object", "additionalProperties" => false,
              "required" => ["name", "year", "city"],
              "properties" => Dict("name" => Dict("type" => "string"),
                                   "year" => Dict("type" => "integer"),
                                   "city" => Dict("type" => "string")))
chat = Chat(service=llama, model="d1-3b", response_format=UniLM.json_schema("person", "A person", schema))
push!(chat, Message(Val(:system), "You extract facts."))
push!(chat, Message(Val(:user), "Extract the person: Ada Lovelace, born 1815 in London."))
JSON.parse(text(chatrequest!(chat)))
```

The formats every provider shares are on [Structured Output](@ref structured_guide).

## [Embeddings, respond and FIM](@id local_servers_other_apis)

These calls reach the server's other OpenAI-compatible routes; this page did not run them, so their
code stands without output:

```julia
emb = Embeddings(["first text", "second text"]; service=vllm, model="BAAI/bge-m3")
embeddingrequest!(emb)                                                    # /v1/embeddings
result = respond("Name a prime number."; service=llama, model="d1-3b")   # /v1/responses
fim = fim_complete("def area(r):\n    "; suffix="\n    return a\n",       # /v1/completions
                   service=lmstudio, model="qwen2.5-coder-1.5b")
```

- **Embeddings** need an embedding model: vLLM's embeddings route is "Only applicable to embedding
  models" ([OpenAI-compatible server](https://docs.vllm.ai/en/latest/serving/online_serving/openai_compatible_server/)),
  and `llama-server`'s "requires that the model uses a pooling different than type `none`". An item
  holding an image or a sound clip is refused before the request: these routes take text. EmbeddingGemma's
  task prompts hold on any server ([task prompts](@ref ollama_embeddings__next)).
- **respond** sends a `Respond`'s fields as they are, except a `seed`, which UniLM refuses
  ([Limits and refusals](@ref local_servers_limits)). vLLM serves stored responses
  (`/v1/responses/{response_id}`), LM Studio takes a `previous_response_id`, and `llama-server`'s route
  "works by converting Responses request into Chat Completions request".
- **FIM** sends the `suffix` to `/v1/completions`, where vLLM's Completions API notes "`suffix` parameter
  is not supported", so a vLLM server cannot fill in the middle.

## [Limits and refusals](@id local_servers_limits)

UniLM checks no limit of a server: a request longer than the model's context window, or one the model
cannot serve, is the server's error ([When something fails](@ref local_servers_failures)). It refuses
before anything is sent only what the table below lists, and the checks every `Chat` gets, such as a
`temperature` from 0 to 2 or an `n` of 1; the verbs of the capabilities the table at the top of this page
lists as not available refuse the endpoint too.

| Refused | Why: the message's words | What to do |
| :--- | :--- | :--- |
| a `Chat`, `Respond`, `Embeddings` or FIM request without a `model` | `model must be specified`: the endpoint has no default model | the name the server serves the model under |
| a `Respond` with a `seed` | `takes no seed`: the Responses request UniLM writes has no `seed` field | a `Chat`, which sends its `seed` |
| an `Embeddings` item holding an image or a sound clip | `takes text only` | the item's text alone |
| an EmbeddingGemma request without a `task` | `expects a task prompt` | a [task](@ref ollama_embeddings__next), or `task = :none` where the server writes the prompts |
| a `FIMCompletion` with `stream=true` | `does not support stream=true`: `fim_complete` has no streaming path | leave `stream` unset |

## [When something fails](@id local_servers_failures)

`chatrequest!`, `respond`, `embeddingrequest!` and `fim_complete` return what the server or the network
did as a typed result, and do not throw it ([Results](@ref results_guide)). A failure holds the
server's HTTP `status` and, as `response`, the body that came with it; a call error holds the text of
an exception as `error` and the exception as `cause`. UniLM sends a request again after a transient
status — 408, 429, 500, 502 to 504, 520 to 524 and 529 — or a failed connection, as often as the call's
[`RequestConfig`](@ref) allows, and then returns the last answer ([Timeouts & Retries](@ref timeouts_guide));
a streamed request is sent again only while none of its callbacks has run.

| What you see | Why | What to do |
| :--- | :--- | :--- |
| `ArgumentError` from the call, nothing sent | the request breaks a rule UniLM checks before sending ([Limits and refusals](@ref local_servers_limits)) | what its row says |
| `InvalidConversationError` from `chatrequest!`, nothing sent | with `history=true`, the conversation ends with an assistant message, so the reply could not be appended | a user message after it, or `history=false` |
| `LLMCallError` whose `cause` is an `HTTP.ConnectError` | no server answered at the base URL: the connection was refused, on each attempt the call's `RequestConfig` allows | start the server, or correct the base URL |
| `LLMCallError` whose `cause` is a `KeyError` | the endpoint reads its key from an environment variable that is not set | set it |
| a failure with `status` 401 | the server requires a key ([Setup](@ref local_servers_setup)) | the key |
| a failure with `status` 400, 404 or another status not listed above | the server refused the request — a model it does not serve, a prompt longer than its context window, a field it rejects — and UniLM does not send it again: `response` holds the server's error | what the error says |
| a failure with `status` 500, 503 or another transient status | a fault or an overload on the server, sent again until the attempts or the deadline run out | the request again, later |
| an `LLMFailure` with `status` 200 from a stream | the stream ended before a finish reason; `response` holds what arrived, and nothing is appended | the turn again |
| an `LLMFailure` with the code of an error object, or an `LLMCallError` with `status` `nothing`, from a stream | an `{"error": …}` object in the HTTP 200 stream: a failure whose `status` is its numeric `code`, from 400 to 599, a call error otherwise | the turn again |
| a call error with `status` `nothing` and a `UniLMTimeout` as `cause` | a bound of the call's `RequestConfig` ran out, as when the server loads a model or queues the request ([Timeouts & Retries](@ref timeouts_guide)) | a longer bound, or a stream for a long reply |
| a call error whose `cause` is a `UniLMCancelled` | a stop: the stream's `close[] = true` or a cancelled `CancelToken` ([Streaming](@ref local_servers_streaming)) | nothing, if the stop was yours |
| a call error whose `cause` is the exception your `callback` or `on_tool_call` threw | the exception ended the stream; it is not sent again, and nothing is appended; a thrown value that is no `Exception` leaves `cause` `nothing`, its text in `error`; an `InterruptException`, or a recorded call's `ReplayMissError` or `RecordingWriteError`, is thrown at `fetch` instead | the callback |
| `LLMResultError` from `text`, `output_text`, `embedding_vectors` or `fim_text` | the result is a failure, which holds no reply | check `issuccess(result)` first |

Recorded on 25 October 2026 on `d1-3b`, a 2.9 GB GGUF model, with UniLM's development branch after 0.26.0
and llama.cpp's `llama-server` build 9730 (`e475fa2b5`), on an Apple M4 Max with 64 GB.

See also: [`GenericOpenAIEndpoint`](@ref), [`Chat`](@ref), [`chatrequest!`](@ref) and [`LLMSuccess`](@ref)
in the API reference.
