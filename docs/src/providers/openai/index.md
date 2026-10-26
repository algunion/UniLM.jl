# [OpenAI](@id openai_start)

```@setup openai
using UniLM
```

OpenAI makes the GPT models. `OPENAIServiceEndpoint` speaks two of OpenAI's APIs for them: Chat
Completions, where every request is a `Chat` sent by `chatrequest!` or `tool_loop!`, and Responses,
where every request is a `Respond` sent by `respond` or `tool_loop`. The same endpoint reaches
OpenAI's embeddings, image, audio and platform APIs, and OpenAI's Decisions API, which `ask` speaks,
has an entry of its own: [OpenAI Decisions API](@ref openai_decisions_start). No verb of a capability
the table below lists as not available sends a request through this endpoint. Of the two APIs,
UniLM sends no Responses `prompt` template and no `access_programs`, does not use the Responses
API's WebSocket mode, and does not read back the completions a Chat Completions request stores
with `store=true`.

```@eval
Main.ProviderFacts.at_a_glance(:openai)
```

| I want to… | Use | You get |
| :--- | :--- | :--- |
| [set my key](@ref openai_setup) | `OPENAI_API_KEY` in the environment | every request through `OPENAIServiceEndpoint` sent with it |
| [choose between Chat Completions and Responses](@ref openai_apis) | a `Chat` or a `Respond` | the API whose features the request needs |
| [make a first call](@ref openai_first_call) | `chatrequest!(chat)` or `respond(input)` | an `LLMSuccess` or a `ResponseSuccess`: the reply and its usage, priced by `estimated_cost` |
| [hold a conversation](@ref openai_conversations) | `push!` and `chatrequest!` again, or `previous_response_id` | each turn answered with the turns before it |
| [stream a reply](@ref openai_streaming) | `stream=true` and a `callback`, or `respond` with a `do` block | the text as it arrives, then the whole reply and its usage |
| [let the model call my functions](@ref openai_tools) | `function_tool` and `tool_loop` | the calls run and answered until the model replies |
| [search the web or reach an MCP server](@ref openai_hosted_tools) | `web_search()`, `mcp_tool(…)` and the other hosted tools | tools OpenAI runs, their results in the response |
| [get JSON that fits a schema](@ref openai_structured_output) | `response_format`, or `text=json_schema_format(…)` | a reply constrained to the schema |
| [control reasoning](@ref openai_reasoning) | `reasoning_effort`, or `reasoning=Reasoning(effort=…)` | that effort, checked against the model's row in the rule table |
| [send images, audio and PDFs](@ref openai_media) | `ImageAttachment`, `AudioAttachment`, `DocumentAttachment` | the files as content parts of the message |
| [get a spoken reply](@ref chat_audio_output) | `modalities=["text", "audio"]` and an `audio` voice and format | the audio, its transcript, and the audio sent back by id on the next turn |
| [embed text](@ref openai_embeddings) | `Embeddings` and `embeddingrequest!` | the vectors, filled in place |
| [cache a long prompt](@ref openai_caching) | `prompt_cache_options` | cache reads and writes, counted in the usage |
| [read back, delete or cancel a stored response](@ref openai_stored_responses) | `get_response`, `delete_response`, `cancel_response` | the stored response, or its removal |
| [know what a call cost](@ref openai_cost) | `estimated_cost`, `cumulative_cost` | USD estimates from the reported usage |
| [generate or edit images](@ref openai_images) | `generate_image`, `edit_image` | images from OpenAI's Images API |
| [transcribe or synthesize speech](@ref openai_audio) | `transcribe`, `speak` | text from audio, audio from text |
| [hold a Realtime session](@ref timeout_realtime) | `realtime_connect` | a WebSocket session of OpenAI's Realtime API |
| [answer from my own files](@ref retrieval_guide) | `file_search` over a vector store | answers grounded in your files, and the chunks they came from |
| [upload files, run batches, moderate content](@ref openai_platform_page) | `upload_file`, `create_batch`, `moderate` and the other platform verbs | OpenAI's platform APIs |
| [list the models my key can use](@ref openai_models) | `list_models(service=OPENAIServiceEndpoint)` | each model's id, creation time and shutdown date |
| [handle a failure](@ref openai_failures) | the result's type | the status or the exception, never an empty success |

## [Setup](@id openai_setup)

Set your API key in the environment Julia starts from:

```bash
export OPENAI_API_KEY="sk-..."
```

UniLM reads the variable each time a request is sent: a `chatrequest!` or `respond` without it
returns a call-error result, an `LLMCallError` or a `ResponseCallError`, whose `cause` is the
`KeyError` naming it. `OPENAIServiceEndpoint` is the default `service` of `Chat`, `Respond` and
`Embeddings`; the examples on these pages pass it as `service=OPENAIServiceEndpoint`, the bare type, or
leave it to the default.
Requests go to `https://api.openai.com` — a `Chat` to `/v1/chat/completions`, a `Respond` to
`/v1/responses` — with the key in an `Authorization: Bearer` header.

A `Chat` or `Respond` without `model` takes the default model named at the top of this page. A
request that sets no reasoning effort runs at the model's default effort, and UniLM checks the
rules of the model's row in the [rule table](@ref openai_reasoning) at that effort, the row's
Default effort cell. Where the row takes sampling controls or Chat Completions tools only at
another effort, as the default model's row does, a request that sets them must name that effort:
the check runs when the request is encoded, before anything is sent, and throws `ArgumentError`
([Limits and refusals](@ref openai_limits)), while the `Chat` constructor accepts the request:

```@example openai
chat = Chat(service=OPENAIServiceEndpoint, temperature=0.2)
push!(chat, Message(Val(:system), "You are a concise assistant."))
push!(chat, Message(Val(:user), "Name a prime number."))
refusal = try UniLM.encode_request(chat.service, chat); nothing catch err; err end
refusal isa ArgumentError || error("expected the request to be refused")
print(refusal.msg)
```

With the effort the message names, the same request is sent as it stands:

```@example openai
chat = Chat(service=OPENAIServiceEndpoint, temperature=0.2, reasoning_effort="none")
push!(chat, Message(Val(:system), "You are a concise assistant."))
push!(chat, Message(Val(:user), "Name a prime number."))
print(UniLM.encode_request(chat.service, chat))
```

[`UniLM.encode_request`](@ref) returns the body `chatrequest!` sends, or throws the refusal, with no
key and no request. Another model is a `model` away; which models UniLM holds rules and prices for
is on [Models](@ref openai_models).

## [Chat Completions or Responses](@id openai_apis)

OpenAI serves the default model on both APIs, Chat Completions and Responses
([GPT-6 Sol](https://developers.openai.com/api/docs/models/gpt-6-sol), Endpoints), and a program may use
both through the same endpoint. They differ in where the conversation lives and in what a request may
carry:

| Feature | Chat Completions | Responses |
| :--- | :--- | :--- |
| A request | a `Chat`, sent by `chatrequest!` | a `Respond`, sent by `respond` |
| The conversation | kept by the `Chat` and sent whole each turn | stored by OpenAI and chained with `previous_response_id` or `conversation` |
| System prompt | a system `Message` | `instructions` |
| Function tools | `Tool` and `tool_loop!`, at the efforts the rule table allows | `function_tool` and `tool_loop`, at every effort |
| Hosted tools | web search on OpenAI's search models only (`web_search_options`) | `web_search()`, `file_search(…)`, `mcp_tool(…)`, `code_interpreter()` and more |
| Structured output | `response_format` | `text=json_schema_format(…)` |
| Reasoning | `reasoning_effort` | `Reasoning(effort, summary, context, mode)` |
| Streaming | `stream=true` and a `callback` | `respond` with a `do` block |
| Audio | audio input, and spoken replies | — |
| Cost | `estimated_cost`, and `cumulative_cost` of the `Chat` | `estimated_cost` of each result |

For tool calling, OpenAI's guide to its GPT-6 models points to the Responses API: "Use Responses
for reasoning with tools"
([GPT-6 guide](https://developers.openai.com/api/docs/guides/latest-model/gpt-6-astra#update-api-and-model-parameters)).
On Chat Completions, a model of the [rule table](@ref openai_reasoning) takes function tools only at
the efforts its Chat Completions tools column lists, none where the cell is "—", and the reasoning
efforts the two APIs take differ per model: UniLM refuses a request that breaks either rule, and the
refusal's message names the fix ([Limits and refusals](@ref openai_limits)). The hosted tools are tools of the Responses API: Chat Completions
"supports only specialized search models for web search"
([web search](https://developers.openai.com/api/docs/guides/tools-web-search#chat-completions-api)).
Audio input and spoken replies are Chat Completions features, which OpenAI's audio guide directs
to "Chat Completions with an audio-capable model"
([Audio in Chat Completions](https://developers.openai.com/api/docs/guides/audio-chat-completions)).

## [First call](@id openai_first_call)

On Chat Completions, a `Chat` starts with its system message; `chatrequest!` sends it and returns a
typed result, and `text` reads the reply — or throws, on a failure:

```@example openai
chat = Chat(service=OPENAIServiceEndpoint)
push!(chat, Message(Val(:system), "You are a concise assistant."))
push!(chat, Message(Val(:user), "In one sentence: what is a prime number?"))
result = chatrequest!(chat)
text(result)
```

```@example openai
result.model, result.message.finish_reason
```

`result.model` is the model that served the reply, as OpenAI reports it, and the model
[`estimated_cost`](@ref) prices the call at. `finish_reason` is OpenAI's: `"stop"`, `"length"` "if the
maximum number of tokens specified in the request was reached", `"tool_calls"`, or `"content_filter"`
([Create chat completion](https://developers.openai.com/api/reference/resources/chat)); a turn that
carries tool calls reads `"tool_calls"` when OpenAI finished it with `"stop"`.

```@example openai
u = result.usage
(; u.prompt_tokens, u.completion_tokens, u.reasoning_tokens, u.cached_tokens, u.service_tier,
   cost = estimated_cost(result))
```

`prompt_tokens` counts all the input, the cache reads (`cached_tokens`) and writes
(`cache_write_tokens`) among it, and `completion_tokens` all the output, the reasoning
(`reasoning_tokens`) included; `service_tier` is the tier OpenAI reports processing the call at.
How these become a price is on [Cost and usage](@ref openai_cost), and the result types on
[Results](@ref results_guide).

On Responses, `respond` takes the input and the request's fields as keywords, and `output_text`
reads the answer:

```@example openai
result = respond("In one sentence: what is a prime number?"; service=OPENAIServiceEndpoint,
                 instructions="You are a concise assistant.")
output_text(result)
```

```@example openai
u = token_usage(result)
(; result.response.model, result.response.status, u.prompt_tokens, u.completion_tokens,
   u.reasoning_tokens, cost = round(estimated_cost(result); digits=6))
```

A Responses result holds the response as OpenAI sent it: `result.response` is a
[`ResponseObject`](@ref) with its `id`, `status`, `model`, `output` items, `usage` and the whole body
as `raw`, and [`token_usage`](@ref) reads its usage into the same [`TokenUsage`](@ref) a `Chat` result
carries. What else a `Respond` takes is on
[Responses & hosted tools](@ref responses_guide).

## [Conversations](@id openai_conversations)

A `Chat` keeps the conversation: each reply is appended to it, and the next `chatrequest!` sends the
whole conversation again.

```@example openai
chat = Chat(service=OPENAIServiceEndpoint)
push!(chat, Message(Val(:system), "You are a concise assistant: answer in one short sentence."))
push!(chat, Message(Val(:user), "Which planet of the solar system is the largest?"))
text(chatrequest!(chat))
```

```@example openai
push!(chat, Message(Val(:user), "And the smallest?"))
text(chatrequest!(chat))
```

```@example openai
[m.role for m in chat.messages], cumulative_cost(chat)
```

The messages go out in the order they were pushed, system messages where they stand, and a reply
goes back as its text and tool calls; a spoken reply also goes back by its audio id
([Audio replies](@ref chat_audio_output)). A `Message` needs a text, tool calls, a refusal,
attachments or an audio reply, and its constructor throws `ArgumentError` for one that has none;
what `push!` refuses is the same on every provider ([Conversations with Chat](@ref chat_guide)).

A response is stored by OpenAI unless the request sets `store=false`: OpenAI keeps it "for at least
30 days" ([Create a model response](https://developers.openai.com/api/reference/resources/responses/methods/create)).
The next request names it with `previous_response_id` and sends only the new input:

```@example openai
r1 = respond("Which planet of the solar system is the largest?"; service=OPENAIServiceEndpoint,
             instructions="You are a concise assistant: answer in one short sentence.")
output_text(r1)
```

```@example openai
r2 = respond("And the smallest?"; service=OPENAIServiceEndpoint, previous_response_id=r1.response.id,
             instructions="You are a concise assistant: answer in one short sentence.")
output_text(r2)
```

Each request names its `instructions` again, since "the instructions from a previous response will not
be carried over to the next response"
([Create a model response](https://developers.openai.com/api/reference/resources/responses/methods/create),
`instructions`), and "all previous input tokens for responses in the chain are billed as input tokens"
([conversation state](https://developers.openai.com/api/docs/guides/conversation-state#passing-context-from-the-previous-response)).
A Conversations API object holds a conversation instead: `respond(input; conversation=id)` adds the
response's input and output to it, and a `Respond` refuses `conversation` together with
`previous_response_id` ([Limits and refusals](@ref openai_limits)); the conversation itself is made
by [`create_conversation`](@ref) ([Platform APIs](@ref openai_conversations_api)). A request that replays
the conversation's items itself, rather than chaining, sends them as its `input`
([Input](@ref openai_responses_input)).

## [Streaming](@id openai_streaming)

With `stream=true`, `chatrequest!` returns a `Task` at once, and `fetch` gives the same typed result as
an unstreamed call. The `callback` receives the text as it arrives, a `String` at a time, then the
assembled `Message`:

```@example openai
chat = Chat(service=OPENAIServiceEndpoint, stream=true)
push!(chat, Message(Val(:system), "You are a concise assistant."))
push!(chat, Message(Val(:user), "List the planets of the solar system in order, one per line."))
deltas = String[]
task = chatrequest!(chat; callback=(chunk, close) -> chunk isa String && push!(deltas, chunk))
result = fetch(task)
text(result)
```

```@example openai
length(deltas), result.usage.completion_tokens, estimated_cost(result)
```

A `Chat` whose `stream_options` is unset asks OpenAI for the stream's usage, as
`stream_options = {"include_usage": true}`, so a streamed turn has `usage` and adds its cost to
`cumulative_cost` as an unstreamed one does. OpenAI then streams "an additional chunk" before
`data: [DONE]`, whose `usage` covers the whole request and whose `choices` is empty, and "If the
stream is interrupted or cancelled, you may not receive the final usage chunk"
([Create chat completion](https://developers.openai.com/api/reference/resources/chat), `stream_options`
and the chunk's `usage`):

- **End.** The turn is final at `[DONE]`, or, without one, when the HTTP body ends after the finish
  reason: the `Message` reaches the callback then.
- **Stops.** `close[] = true` in the callback, or a cancelled `CancelToken`, ends the call with an
  `LLMCallError` whose `cause` is a `UniLMCancelled` — unless the finish reason had arrived: then the
  turn is committed, and its usage may be missing ([Stopping a Stream Early](@ref streaming_stop)).
  A stream stopped before its usage chunk adds nothing to `cumulative_cost`.
- **Tool calls.** `on_tool_call(tc)` runs once per call whose arguments decode, when its arguments
  are whole: as the next call begins, or at the end of the stream
  ([Tools on Chat Completions](@ref openai_chat_tools)).
- **Spoken replies.** The transcript of an audio reply reaches the callback as text, and the audio is
  assembled from its chunks ([Audio replies](@ref chat_audio_output)).

`respond` streams with a `do` block, whose function receives the text, then the
[`ResponseObject`](@ref) of a response that completed or ended incomplete; the response's usage comes
with it:

```@example openai
deltas = String[]
task = respond("Name the four largest moons of Jupiter, one per line."; service=OPENAIServiceEndpoint) do chunk, close
    chunk isa String && push!(deltas, chunk)
end
result = fetch(task)
output_text(result)
```

```@example openai
length(deltas), token_usage(result).completion_tokens, estimated_cost(result)
```

A Responses stream ends at its terminal event: `response.completed` and `response.incomplete` give a
`ResponseSuccess`, `response.failed` a `ResponseFailure`. A stop before that event ends the call with
a `ResponseCallError` whose `cause` is the `UniLMCancelled`. The concepts every provider shares are
on [Streaming](@ref streaming_guide).

## [Embeddings](@id openai_embeddings)

An [`Embeddings`](@ref) request on `OPENAIServiceEndpoint` goes to `/v1/embeddings`, on OpenAI's
`text-embedding-3-small` unless it names a `model`. It holds one text or several, and
`embeddingrequest!` fills its pre-allocated vectors in place:

```@example openai
emb = Embeddings(["The Pacific is the largest ocean.", "Mercury is the smallest planet."];
                 service=OPENAIServiceEndpoint, dimensions=256)
result = embeddingrequest!(emb)
length.(embedding_vectors(result)), token_usage(result).prompt_tokens, estimated_cost(result)
```

By default "the length of the embedding vector is `1536` for `text-embedding-3-small` or `3072` for
`text-embedding-3-large`"
([embeddings guide](https://developers.openai.com/api/docs/guides/embeddings#how-to-get-embeddings)).
`dimensions` is sent as it is: the `text-embedding-3` models return a shorter vector when a request
asks for one
([embeddings guide](https://developers.openai.com/api/docs/guides/embeddings#reducing-embedding-dimensions)),
and UniLM checks only that it is positive. The buffers start at `dimensions`, else 1536, numbers per
text, and take the length the reply has. An embedding bills its input tokens alone, at the rate of the
model's row in the [price table](@ref openai_cost). An image or a sound clip, a `title`, and a `task`
other than `:none` are refused before the request: the OpenAI Embeddings API takes text, and UniLM
writes a task prompt only for the models it knows to read one ([Limits and refusals](@ref openai_limits)).
The concepts every provider shares are on [Embeddings](@ref embeddings_guide).

## [When something fails](@id openai_failures)

`chatrequest!`, `respond` and `embeddingrequest!` return what the service or the network did as a
typed result, and do not throw it ([Results](@ref results_guide)). A failure is an `LLMFailure`,
`ResponseFailure` or `EmbeddingFailure`, holding OpenAI's HTTP `status` and, as `response`, the body
that came with it; or an `LLMCallError`, `ResponseCallError` or `EmbeddingCallError`, holding the text
of the exception as `error` and the exception as `cause`. A failure of a `Chat` or a `Respond` keeps
the reply's `x-request-id` header, when one came, as `request_id`.

UniLM sends a request again after a transient status — 408, 429, 500, 502 to 504, 520 to 524 and
529 — as often as the call's [`RequestConfig`](@ref) allows, waiting at least as long as the reply asks,
by its `retry-after-ms` header, else `Retry-After`, and then returns the last answer
([Timeouts & Retries](@ref timeouts_guide)).
A 429 whose `error.code` is `credit_balance_exhausted`, `organization_spend_limit_exceeded`,
`project_spend_limit_exceeded` or `organization_usage_limit_exceeded`, or whose `error.type` or
`error.code` is `insufficient_quota`, is returned at once: "Retrying billing, spend, or quota errors
won't restore API access" ([error codes](https://developers.openai.com/api/docs/guides/error-codes)).
A streamed request is sent again only while none of its callbacks has run. The lifecycle calls of
stored responses — `get_response`, `delete_response`, `list_input_items`, `cancel_response`,
`compact_response` and `count_input_tokens` — make one attempt, which `max_attempts` does not change.

| What you see | Why | What to do |
| :--- | :--- | :--- |
| `ArgumentError` from the call, nothing sent | the request breaks a rule UniLM checks before sending ([Limits and refusals](@ref openai_limits); for images, audio and Realtime, [Images, audio & realtime](@ref openai_media_limits); for the platform verbs, [Platform APIs](@ref openai_platform_limits)) | what its row says |
| `InvalidConversationError` from `chatrequest!`, nothing sent | with `history=true`, the conversation ends with an assistant message, so the reply could not be appended | a user message after it, or `history=false` |
| a call error whose `cause` is a `KeyError` | `OPENAI_API_KEY` is not set ([Setup](@ref openai_setup)) | set it |
| a failure with `status` 400, 401, 403, 404 or another status not listed above | OpenAI refused the request, and UniLM does not send it again: `response` holds OpenAI's error, whose `message` names the reason | what the message says |
| a failure with `status` 429 | a rate limit, sent again; or, with one of the codes above, spent credit or a reached spend or usage limit, returned at once ([rate limits](https://developers.openai.com/api/docs/guides/rate-limits)) | fewer requests or tokens per minute, or more `max_attempts`; for a limit, the credits or limits the error names |
| a failure with `status` 500, 503 or another transient status | a fault or an overload on OpenAI's side, sent again until the attempts or the deadline run out | the request again, later |
| an `LLMFailure` with `status` 200 from a stream | the stream ended before a finish reason; `response` holds what arrived, and nothing is appended | the turn again |
| an `LLMFailure` with the code of an error object, or an `LLMCallError` with `status` `nothing`, from a stream | an `{"error": …}` object in the HTTP 200 stream: a failure whose `status` is its numeric `code`, from 400 to 599, a call error otherwise | the turn again |
| a `ResponseFailure` with `status` 200 | the response ended `failed`, streamed or not: `response` holds it, its `error` saying why | what the error says |
| a `ResponseSuccess` whose `response_status` is `"incomplete"` | the response ended before it was complete, as when the output reaches `max_output_tokens` or the context window; [`incomplete_details`](@ref) gives the reason, and the output so far is kept | a higher `max_output_tokens`, or a shorter input |
| a call error with `status` `nothing` and a `UniLMTimeout` as `cause` | a bound of the call's `RequestConfig` ran out ([Timeouts & Retries](@ref timeouts_guide)) | a longer bound, or a stream for a long reply |
| a call error whose `cause` is a `UniLMCancelled` | a stop: the stream's `close[] = true` or a cancelled `CancelToken` ([Streaming](@ref openai_streaming)) | nothing, if the stop was yours |
| a call error whose `cause` is the exception your `callback` or `on_tool_call` threw | the exception ended the stream; it is not sent again, and nothing is appended; an `InterruptException`, or a recorded call's `ReplayMissError` or `RecordingWriteError`, is thrown at `fetch` instead | the callback |
| `LLMResultError` from `text`, `output_text` or `embedding_vectors` | the result is a failure, which holds no reply | check `issuccess(result)` first |

`ResponseFailure` is reserved for a response that `failed`; every other status — `incomplete`,
`cancelled`, `queued`, `in_progress` — is a `ResponseSuccess`, read with
[`response_status`](@ref). A response cut at `max_output_tokens` before any visible output can "incur
costs for input and reasoning tokens without receiving a visible response"
([reasoning](https://developers.openai.com/api/docs/guides/reasoning#allocating-space-for-reasoning)).

Recorded on 25 October 2026 on `gpt-6-sol` and `text-embedding-3-small` with UniLM's development
branch after 0.26.0.

See also: [`OPENAIServiceEndpoint`](@ref), [`Chat`](@ref), [`chatrequest!`](@ref), [`Respond`](@ref),
[`respond`](@ref) and [`LLMSuccess`](@ref) in the API reference; the
[OpenAI Decisions API](@ref openai_decisions_start) in the Decisions chapter.
