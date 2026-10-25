# [Ollama](@id ollama_guide__next)

```@setup ollama
using UniLM
```

[Ollama](https://ollama.com) runs open models on your own machine, and `OllamaEndpoint` speaks its
native API: a `Chat` goes to `/api/chat` and an `Embeddings` request to `/api/embed`, the routes that
take the context window, how long a model stays loaded, thinking and truncation. `respond`,
`compact_response`, `fim_complete` and `transcribe` reach Ollama's OpenAI-compatible routes instead,
which carry none of the endpoint's options. No key is sent to a server of your own, and a local model
costs nothing. No verb of a capability the table below lists as not available sends a request through
this endpoint: `speak` refuses it, though Ollama serves speech at `/v1/audio/speech`. Of Ollama's own
API, UniLM does not wrap `/api/generate`, `/api/version`, creating, copying or pushing a model, or web
search and web fetch. The decision models `ask` reaches through the same endpoint have an entry of their
own: [Local decision models](@ref ollama_decisions_start).

```@eval
Main.ProviderFacts.at_a_glance(:ollama)
```

| I want to… | Use | You get |
| :--- | :--- | :--- |
| [run a local model](@ref ollama_setup) | `ollama pull`, then `service=OllamaEndpoint(), model=…` | requests to the server at `OLLAMA_HOST`, else on this machine |
| [make a first call](@ref ollama_first__next) | `chatrequest!(chat)` | an `LLMSuccess`: the reply and its token counts, at a cost of `0.0` |
| [hold a conversation](@ref ollama_conversations) | `push!` a message, then `chatrequest!` again | each reply appended to the `Chat` and sent back with the next turn |
| [stream a reply](@ref ollama_streaming) | `stream=true` and a `callback` | the text as it arrives, then the whole message and its usage |
| [let the model call my functions](@ref ollama_tools__next) | `tools` and `tool_loop!` | the calls run and answered until the model replies |
| [get JSON that fits a schema](@ref ollama_json__next) | a JSON Schema `response_format` | a reply constrained to the schema |
| [see how the model reasoned](@ref ollama_thinking__next) | `reasoning_effort`, `reasoning_text` | the thinking, when the model thinks |
| [ask about an image or a recording](@ref ollama_media__next) | `ImageAttachment`, `AudioAttachment` | an answer about what the model saw or heard |
| [embed texts, images and sound](@ref ollama_embeddings__next) | `Embeddings` with a `task`, and `embeddingrequest!` | one unit-length vector per item, written for its task |
| [transcribe a recording](@ref ollama_transcription__next) | `transcribe(path; service=OllamaEndpoint(), model=…)` | the words the model heard |
| [use respond, or fill in the middle](@ref ollama_respond) | `respond`, `fim_complete` | answers from Ollama's OpenAI-compatible routes |
| [set the context window, keep a model loaded, or pin its runner](@ref ollama_options__next) | `OllamaEndpoint(num_ctx=…, keep_alive=…, runner=…)` | every request of the endpoint carrying them |
| [install, inspect, load or delete models](@ref ollama_models__next) | `list_models`, `model_info`, `pull_model`, `load_model`, … | typed model cards and results |
| [decide about a text or a picture](@ref ollama_decisions_start) | `ask(…; service=OllamaEndpoint(), model=…)` | typed answers with probabilities |
| [handle a failure](@ref ollama_failures__next) | the result's type | the status or the exception, and what to do |

## [Setup](@id ollama_setup)

Install Ollama and start it, the desktop app or `ollama serve`, then pull the models the examples
run on:

```bash
ollama pull gemma4:e4b                  # chat: text, images, audio, tools, thinking (6.6–9.5 GB by runner; 9.5 GB on Apple Silicon)
ollama pull embeddinggemma-2:740m-q8_0  # embeddings of text, images and audio (1.3 GB)
ollama pull qwen2.5-coder:1.5b-base     # fill in the middle (986 MB)
```

A model's name is `model:tag`, and a name without a tag means its `latest` tag
([model names](https://github.com/ollama/ollama/blob/v0.40.11/docs/api.md#model-names)): `embeddinggemma-2`
alone pulls a different 740m build, which was not measured here, so the examples name the tags they
ran on. To run a local model with Ollama, `ollama pull gemma4:e4b`, then pass
`service=OllamaEndpoint(), model="gemma4:e4b"`; Ollama has no default model. A `Chat`, `Respond`,
`Embeddings`, FIM or transcription request that names none throws `ArgumentError` before anything is
sent, and the `Chat`'s message says what to name:

```@example ollama
refusal = try Chat(service=OllamaEndpoint()); nothing catch err; err end
refusal isa ArgumentError || error("expected the Chat to be refused")
print(refusal.msg)
```

`OllamaEndpoint()` reads `OLLAMA_HOST` (optional; default `http://127.0.0.1:11434`) each time it is
built. A host without a scheme is reached over `http` on port 11434 (`gpu-box` becomes
`http://gpu-box:11434`), an `http://` or `https://` address without a port on port 80 or 443, the
bind-all addresses `0.0.0.0` and `::` on the loopback address, and `ollama.com`, given without a scheme,
over `https` on port 443; an IPv6 address may be bare or bracketed, and a path is kept. A value UniLM
cannot read this way, another scheme or a port outside 1–65535, throws `ArgumentError` when the
endpoint is built. The endpoint's other arguments are on [Options](@ref ollama_options__next).

Ollama's local API "does not require authentication", and UniLM sends a server of your own no key. Direct
cloud inference at `https://ollama.com/api` and `https://ollama.com/v1` "requires an API key", used "in
the `Authorization: Bearer` header" ([authentication](https://docs.ollama.com/api/authentication)): an
endpoint whose server is `https://ollama.com`, as `OLLAMA_HOST=ollama.com` makes it, sends the
`OLLAMA_API_KEY` environment variable that way, read each time a request is built, and to no other
host. With the variable unset or blank nothing is sent: the call returns a call error whose `cause` is
an `ArgumentError` naming `OLLAMA_API_KEY`. No request was sent to the cloud for this page (its
requests bill). A cloud model runs through the local server once you sign in with
`ollama signin`, under a cloud tag such as `gemma4:cloud`; [First call](@ref ollama_first__next) says how
UniLM prices it.

## [First call](@id ollama_first__next)

A `Chat` starts with its system message; `chatrequest!` sends it and returns a typed result, and `text`
reads the reply — or throws, on a failure:

```@example ollama
chat = Chat(service=OllamaEndpoint(), model="gemma4:e4b", reasoning_effort="none")
push!(chat, Message(Val(:system), "You are a concise assistant."))
push!(chat, Message(Val(:user), "In one sentence: why run a language model locally?"))
result = chatrequest!(chat)
text(result)
```

```@example ollama
result.message.finish_reason, result.model
```

```@example ollama
u = result.usage
(; u.prompt_tokens, u.completion_tokens, u.cached_tokens, cost = estimated_cost(result))
```

`reasoning_effort="none"` turns Gemma 4's thinking off for a short answer
([Reasoning](@ref ollama_thinking__next)). The reply's `done_reason` becomes `finish_reason`: `"stop"`,
or `"length"` for a reply that reached its `max_tokens` (sent as `num_predict`); a turn that carries
tool calls reads `"tool_calls"` unless it was cut at `"length"`. `result.model` is `nothing`: UniLM
keeps no model name from an Ollama reply, and [`estimated_cost`](@ref) prices the model the request
named. Of the usage, `prompt_tokens` is Ollama's `prompt_eval_count`, "How many input tokens were in
the prompt", `cached_tokens` its `prompt_eval_cached_count`, "How many prompt tokens were read from the
cache", and `completion_tokens` its `eval_count`, the output tokens
([usage](https://docs.ollama.com/api/usage)), thinking included ([Reasoning](@ref ollama_thinking__next)
shows a count). The result types are on [Results](@ref results_guide).

A local model has no price, so `estimated_cost` gives `0.0` for a chat, embeddings, `respond` or FIM
result, without the missing-price warning other unlisted models get; pass `model=` to price one as if
it had run elsewhere. A cloud model reached through the same server — one whose tag is `cloud` or
ends in `-cloud`, such as `gemma4:cloud` or `gpt-oss:120b-cloud` — runs in
[Ollama's cloud](https://docs.ollama.com/cloud), and UniLM prices each of its results by the requested
name, like any other model, never as a free local call; a name missing from the price table gets the
missing-price warning. A cloud reply may name the model without its tag, `gpt-oss:120b` (by Ollama
0.40.6's source, which sends the cloud the name without it; untested, as no cloud call was made): a
System One, `respond` or FIM result keeps the requested name, and the server's stays in its `raw`,
while a chat or embeddings result is priced by the model its request names. UniLM recognises a cloud
model by its tag alone, so one copied under a name without it (`ollama cp`) is priced as local, at
`0.0`, with no warning.

The first request to a model also loads it into memory: about 1.7 s for Gemma 4 E4B on an Apple M4
Max with 64 GB (Ollama 0.40.6), after which a short answer took about 0.1 s. [`load_model`](@ref) loads
a model ahead of time ([Models](@ref ollama_models__next)).

## [Conversations](@id ollama_conversations)

A `Chat` keeps the conversation: each reply is appended to it, and the next `chatrequest!` sends the
whole conversation again.

```@example ollama
chat = Chat(service=OllamaEndpoint(), model="gemma4:e4b", reasoning_effort="none")
push!(chat, Message(Val(:system), "You are a concise assistant: answer in one short sentence."))
push!(chat, Message(Val(:user), "Which planet of the solar system is the largest?"))
text(chatrequest!(chat))
```

```@example ollama
push!(chat, Message(Val(:user), "And the smallest?"))
text(chatrequest!(chat))
```

```@example ollama
[m.role for m in chat.messages], cumulative_cost(chat)
```

The messages go out in the order they were pushed, each with its role, so a system message pushed
later stays where it stands. An assistant turn goes back with its text, its thinking as `thinking`
([Reasoning](@ref ollama_thinking__next)) and its tool calls with their ids; a tool result goes back
with `tool_name`, the name of the function it answers, as Ollama's
[tool calling](https://docs.ollama.com/capabilities/tool-calling) page sends results, and with its
`tool_call_id` ([Tools](@ref ollama_tools__next)). A message with a `name`, and a tool result that
answers no tool call of an earlier assistant turn, are refused before anything is sent
([Limits and refusals](@ref ollama_limits)); what `push!` refuses is the same on every provider
([Conversations with Chat](@ref chat_guide)).

## [Streaming](@id ollama_streaming)

With `stream=true`, `chatrequest!` returns a `Task` at once, and `fetch` gives the same typed result as
an unstreamed call. The `callback` receives the text as it arrives, a `String` at a time, then the
assembled `Message`:

```@example ollama
chat = Chat(service=OllamaEndpoint(), model="gemma4:e4b", reasoning_effort="none", stream=true)
push!(chat, Message(Val(:system), "You are a concise assistant."))
push!(chat, Message(Val(:user), "List the planets of the solar system in order, one per line."))
deltas = String[]
task = chatrequest!(chat; callback=(chunk, close) -> chunk isa String && push!(deltas, chunk))
result = fetch(task)
text(result)
```

```@example ollama
length(deltas), result.usage.completion_tokens, estimated_cost(result)
```

Ollama streams "newline-delimited JSON", one object per line
([streaming](https://docs.ollama.com/api/streaming)), and sends the usage "as part of the final chunk,
where `done` is `true`" ([usage](https://docs.ollama.com/api/usage)), so a streamed turn has `usage`
with no `stream_options`, a field this endpoint refuses ([Limits and refusals](@ref ollama_limits)).
Setting `close[] = true` stops the stream ([Stopping a Stream Early](@ref streaming_stop)).

- **End.** The turn is final at the line whose `done` is `true`: the `Message` reaches the callback
  then.
- **Thinking.** No thinking reaches the callback, only text: the thinking is assembled into the
  `provider_content` of the `Message` the callback receives last, where
  [`reasoning_text`](@ref) reads it, and goes back with the next request as an unstreamed turn's does.
- **Tool calls.** A line carries whole tool calls, so `on_tool_call(tc)` runs as soon as the line
  that holds a call arrives.
- **Errors.** "If an error occurs mid-stream, the error will be returned as an object" with an `error`
  property, and the status stays 200 ([errors](https://docs.ollama.com/api/errors)): UniLM ends the call
  with an `LLMCallError` holding it ([When something fails](@ref ollama_failures__next)).
- **Silence.** A stream receives nothing while Ollama loads the model or while the request waits its
  turn behind others, which Ollama queues ([FAQ](https://docs.ollama.com/faq)), and
  `stream_idle_timeout` bounds that wait too ([When something fails](@ref ollama_failures__next)).

The concepts every provider shares are on [Streaming](@ref streaming_guide).

## [Tools](@id ollama_tools__next)

Each `Tool` in `tools` goes out as a function tool with its `name`, `description` and `parameters`.
`tool_loop!` runs the calls and sends their results back until the model answers in text:

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

```@example ollama
call = chat.messages[3]   # the turn that called the tool
call.tool_calls[1].id, reasoning_text(call) !== nothing
```

Ollama has no `tool_choice`: the model decides whether to call a tool, so a `tool_choice` other than
`"auto"` is refused, and it may call several tools in one turn — Ollama's
[tool calling](https://docs.ollama.com/capabilities/tool-calling) page shows parallel calls —
whatever `parallel_tool_calls` says: UniLM does not send it. Ollama's tool definition has `name`,
`description` and `parameters` and no `strict` flag ([API reference](https://docs.ollama.com/api/chat)), so a
tool marked `strict=true` is refused too ([Limits and refusals](@ref ollama_limits)).

Ollama's replies carry an id for each call, `call_mb4i6gbs` above, though its
[API reference](https://docs.ollama.com/api/chat) lists none; UniLM sends the id back with the call and
with its result, beside `tool_name`. A call that arrives without an id gets `unilm_call_<k>`, which is
never sent: its result is matched by the function's name alone. The turn that called the tool also thought first, and that thinking went back
with it ([Reasoning](@ref ollama_thinking__next)). A streamed turn's tool calls are on
[Streaming](@ref ollama_streaming), and the loop every provider shares is on
[Tool Calling](@ref tools_guide).

## [Structured output](@id ollama_json__next)

A `response_format` becomes Ollama's `format`: a JSON Schema format sends its schema, and
`UniLM.json_object()` sends `{"type": "object"}`, the schema of any JSON object. Only the schema goes
out; the format's `name`, `description` and `strict` have no counterpart in `format`
([structured outputs](https://docs.ollama.com/capabilities/structured-outputs): "Provide a JSON schema to
the `format` field").

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

With a `response_format`, the request turns thinking off (`think: false`), and a `response_format`
together with a thinking effort (any effort but `"none"`) is refused before the request is sent
([Limits and refusals](@ref ollama_limits)). UniLM's reason, which its message gives: Ollama applies a
format only once the model's thinking ends, so a reply that skips thinking is not constrained at all —
in the measurements below, every reply that was not valid JSON came from a run with thinking on in
which the model did not think. A format limits which
tokens the model may emit, not how many, so a reply can run on without closing the object. Give a
structured-output request a `max_tokens` (sent as `num_predict`): a reply that reaches it stops with
`finish_reason` `"length"` and is not complete JSON, instead of generating until the request times
out. Ollama's page says its cloud "currently does not support structured outputs", and UniLM sends a
cloud model's format all the same. The formats every provider shares are on
[Structured Output](@ref structured_guide).

!!! details "Evidence"
    The extraction request above, sent to `/api/chat` 10 times per setting on
    Ollama 0.34.4: `gemma4:e4b` returned valid JSON in 0 of 10 runs with thinking at
    its default (it answered plain text), 1 of 10 with thinking on, and 10 of 10 with
    it off; `gemma4:e2b` in 10, 8 and 10 of 10. Every failure was a run in which the
    model did not think. Without `max_tokens`, the same request on Ollama 0.40.6
    twice generated until the 600 s default request timeout (more than 60,000 tokens
    in one run); the retry answered in under a second.

## [Reasoning](@id ollama_thinking__next)

`reasoning_effort` becomes Ollama's `think`: `"none"` sends `false`, and every other effort goes as it
is, `"max"` among them, a level of `think` in Ollama's
[API reference](https://github.com/ollama/ollama/blob/v0.40.11/docs/api.md#generate-a-chat-completion); a
request without one sends none, so the model's default applies, and Gemma 4 thinks by default. Ollama's
thinking levels are the model's own: `/api/show` lists the values a model takes and its default, and a
model whose values are `true` and `false` thinks or does not; "If the model resolves named levels from
`/api/show` metadata, Ollama applies supported names exactly. Unsupported names use the model default."
([thinking](https://docs.ollama.com/capabilities/thinking)). [`model_info`](@ref) reads them as
`thinking_levels` and `thinking_default`: `[false, true]` and `true` for Gemma 4 E4B, which names no
levels. No page says what such a model does with a named level; on Ollama 0.40.11, `think` `"low"`,
`"medium"` and `"high"` gave `gemma4:e4b` the same thinking and the same answer to the bat-and-ball
prompt below at temperature 0 (observed on 25 October 2026), and a `think: "max"` request to
`gemma4:e4b` was answered with HTTP 200, an answer without thinking, in the one request sent that day.
To a model without thinking metadata, `/api/chat` answers a name other than `"low"`, `"medium"`, `"high"`
and `"max"`, such as `"minimal"` or `"xhigh"`, with HTTP 400 `invalid think value` (Ollama 0.40.11:
`qwen2.5-coder:1.5b-base`, whose `/api/show` reports no thinking, on 25 October 2026), while `respond`'s
`/v1/responses` reads `"minimal"` as `"low"` and `"xhigh"` as `"max"` for such a model
([OpenAI compatibility](https://docs.ollama.com/api/openai-compatibility)).
The thinking comes back beside the answer, and [`reasoning_text`](@ref) reads it:

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

```@example ollama
result.usage.completion_tokens
```

`reasoning_text` gives the turn's `thinking`, or `nothing` for a turn without one, and the output count
holds the thinking as well as the few characters of the answer. The thinking stays
on its turn in the `Chat` and goes back to Ollama with the next request, which renders the
conversation for the model itself (Gemma 4's page: "Ollama already handles the complexities of the
chat template for you"). Gemma 4's guidance asks that "Thoughts from previous model turns must *not be
added* before the next user turn begins"
([Gemma 4](https://ollama.com/library/gemma4)); in a tool round trip measured with `gemma4:e2b` on
Ollama 0.34.4, the turn succeeded 10 of 10 times with the thinking sent back and 8 of 10 without it.
Thinking spends the reply's token budget first: a small `max_tokens` can end a reply at finish reason
`"length"` with no answer text.

!!! note "A thinking model decides when to think"
    With thinking on, Gemma 4 E4B still answers easy prompts without thinking:
    [`reasoning_text`](@ref) is then `nothing`. On the bat-and-ball prompt above
    (Ollama 0.34.4) it thought in 2 to 5 of 6 runs at the default temperature, and
    in every run at `temperature=0.0`.

## [Attachments](@id ollama_media__next)

A user message carries images and sound clips as attachments, read from a file or from bytes and
recognised by their content: an [`ImageAttachment`](@ref) (PNG, JPEG, GIF or WebP) or an
[`AudioAttachment`](@ref) (WAV or MP3). Each goes out as a base64 entry of the message's `images`, the
form Ollama's API takes: "the REST API expects base64-encoded image data"
([vision](https://docs.ollama.com/capabilities/vision)), so a model never fetches an image itself.

![A red disc on a white background](../../assets/red_circle.png)

```@example ollama
chat = Chat(service=OllamaEndpoint(), model="gemma4:e4b", temperature=0.0)
push!(chat, Message(Val(:system), "You are concise."))
push!(chat, Message(Val(:user), "What colour is the shape? One word.",
                    ImageAttachment("../../assets/red_circle.png")))
text(chatrequest!(chat))
```

The [recording](../../assets/secret_word.wav) says *"The secret word is banana."*:

```@example ollama
chat = Chat(service=OllamaEndpoint(), model="gemma4:e4b", temperature=0.0)
push!(chat, Message(Val(:system), "You are concise."))
push!(chat, Message(Val(:user),
    "What is the secret word in this recording? One word.",
    AudioAttachment("../../assets/secret_word.wav")))
text(chatrequest!(chat))
```

Ollama's [Gemma 4](https://ollama.com/library/gemma4) page lists text, image and audio for E2B and
E4B, and text and image for the 26B and 31B models. [`model_info`](@ref) lists what an installed model
takes, `:vision` and `:audio` among its `capabilities` ([Models](@ref ollama_models__next)); UniLM
checks neither, and a model without the capability answers with an HTTP 400 once it has been loaded:
on Ollama 0.40.11, `qwen2.5-coder:1.5b-base` loaded and then answered an image with "Multimodal data
provided, but model does not support multimodal requests."
([When something fails](@ref ollama_failures__next)). A [`DocumentAttachment`](@ref) and an image's
`detail` level are refused before the request: the `images` entry has room for neither
([Limits and refusals](@ref ollama_limits)). The attachment types are on
[Attachments](@ref attachments_guide).

## [Audio](@id ollama_transcription__next)

Of UniLM's audio verbs, only `transcribe` reaches Ollama: a model that hears audio transcribes it,
while `speak` and `translate` refuse the endpoint before sending anything. A transcription is not
recorded by [`with_recorded_answers`](@ref), so its output here is quoted from a live run:

```julia
r = transcribe("secret_word.wav"; service=OllamaEndpoint(), model="gemma4:e4b")
r isa TranscriptionSuccess || error("the transcription failed: $r")
transcript_text(r)
# Transcribed on 25 October 2026 with UniLM's development branch after 0.26.0 and Ollama 0.40.11,
# the recording above:
# "The secret word is banana"
```

The request goes to `/v1/audio/transcriptions`, a route Ollama's
[OpenAI compatibility](https://docs.ollama.com/api/openai-compatibility) page does not list; Ollama
0.40.11 answered it, as above. There is no default model: name one that hears. Ollama transcribes at
temperature 0 and reads one `language` and a `prompt` (by Ollama 0.40.6's source). UniLM asks for the
transcript as JSON, the only form in which Ollama reports a failure that arrives after its reply has
begun, as an [`AudioCallError`](@ref) (by the same source; untested), and reads the reply as JSON,
though Ollama labels it `application/x-ndjson`. `response_format = "text"` is refused before the
request, since such a failure would read as an empty transcript: leave `response_format` unset and
read the text with [`transcript_text`](@ref). `temperature`, `languages`, `keywords` and any other
format are refused too ([Limits and refusals](@ref ollama_limits)). A transcription makes one
attempt, which `max_attempts` does not change.

## [respond and FIM](@id ollama_respond)

[`respond`](@ref) reaches Ollama's `/v1/responses`, which serves "Only the non-stateful flavor" of the
Responses API: "there is no `previous_response_id` or `conversation` support"
([OpenAI compatibility](https://docs.ollama.com/api/openai-compatibility)). Each call is answered and
forgotten:

```@example ollama
result = respond("In one sentence: what is a prime number?"; service=OllamaEndpoint(),
                 model="gemma4:e4b", instructions="You are a concise assistant.",
                 reasoning=Reasoning(effort="none"))
output_text(result)
```

```@example ollama
u = token_usage(result)
(; result.response.status, u.prompt_tokens, u.completion_tokens, cost = estimated_cost(result))
```

Of a [`Respond`](@ref), UniLM sends `model`, `input`, `instructions`, function `tools`, `tool_choice`
`"auto"`, `temperature`, `top_p`, `max_output_tokens`, `stream`, a `text` format and the `effort` of
`reasoning`, and refuses every other field before the request rather than have Ollama drop it: the
fields that need a stored response (`store`, `previous_response_id`, `background`, `conversation`),
the other fields its compatibility page does not list as supported, and `seed`, which a `Chat` sends to Ollama
instead ([Limits and refusals](@ref ollama_limits)). Ollama's page lists no `text` field either; UniLM sends a
`json_schema` text format with its schema, which Ollama 0.40.11 followed in the one such request sent
on 25 October 2026, and, as for a `Chat`, such a format turns thinking off (`reasoning.effort`
`"none"`) and refuses a thinking effort. Cap a request with `max_output_tokens`:
Ollama reports a reply cut at that limit as `"completed"`, with no `incomplete_details` (Ollama
0.40.8), so a reply whose output tokens ([`token_usage`](@ref)) reach the limit is the cut one. A
`Respond` tool loop, [`get_response`](@ref), [`cancel_response`](@ref), [`delete_response`](@ref) and
[`list_input_items`](@ref) are refused before any request, since Ollama stores no response: run a tool
loop on a `Chat` with `tool_loop!`. [`compact_response`](@ref) reaches `/v1/responses/compact`, a route
the compatibility page does not list, which Ollama 0.40.11 answered with a `compaction` item (observed
on 25 October 2026), and, like a `Chat`, needs a `model`: a call that names none throws before any
request.

[`fim_complete`](@ref) reaches `/v1/completions` with a prompt and the `suffix` the reply fills up to,
on a model that fills in the middle, one whose capabilities include `:insert`, such as
`qwen2.5-coder:1.5b-base` (Gemma 4 E4B's do not):

```@example ollama
result = fim_complete("def area(r):\n    "; suffix="\n    return a\n", service=OllamaEndpoint(),
                      model="qwen2.5-coder:1.5b-base", max_tokens=32, temperature=0.0)
fim_text(result)
```

The completions route takes `suffix` among its fields
([OpenAI compatibility](https://docs.ollama.com/api/openai-compatibility)), and a FIM request has no
default model. These routes carry none of the endpoint's `keep_alive`, options or `runner`
([Options](@ref ollama_options__next)). A local model's `respond` and FIM results cost `0.0`, as its chats
do ([First call](@ref ollama_first__next)); the verbs every provider shares are on
[Agentic respond](@ref agentic_guide) and [FIM & Prefix Completion](@ref completions_guide).

## [Limits and refusals](@id ollama_limits)

A request meets the model's context window, which the endpoint's `num_ctx` sets when the model loads: by
default a chat prompt that does not fit is an HTTP 400 on the `:llamacpp` runner, not a silently
shortened prompt, while a model on the `:mlx` runner answers it (Ollama 0.40.6;
[Runners](@ref ollama_runners__next)). UniLM checks no size itself: not the window, not the length of
a sound clip, not an image's size. An embeddings input longer than the model's window is the server's
HTTP 400 too ([Long inputs](@ref ollama_embedding_window)).

A `Chat` field UniLM maps to no part of the `/api/chat` request is refused, not dropped, since dropping it
would send a request other than the one asked for, and so is each request below that Ollama would answer with an
error or run other than asked. The verbs throw these `ArgumentError`s before anything is sent.
Building the endpoint refuses an argument that breaks its rule ([Options](@ref ollama_options__next)).

| Refused | Why: the message's words | What to do |
| :--- | :--- | :--- |
| a `Chat`, `Respond`, `Embeddings`, `compact_response` or FIM request without a `model` | `model must be specified`: Ollama has no default model | an installed model's name, such as `"gemma4:e4b"` |
| on a `Chat`: `logit_bias`, `user`, `stream_options`, `verbosity`, `store`, `metadata`, `service_tier`, `logprobs`, `top_logprobs`, `prediction`, `modalities`, `audio`, `web_search_options`, `prompt_cache_key`, `safety_identifier`, `prompt_cache_options` or `moderation` | `Ollama /api/chat does not support field(s)`: UniLM maps none of them to Ollama's request (`/api/chat` takes `logprobs` and `top_logprobs`, but a result has no field for them) | remove it |
| a `tool_choice` other than `"auto"`, on a `Chat` or a `Respond` | `Ollama has no tool_choice`: the model decides whether to call a tool | leave it unset, or send no tools to rule calls out |
| a tool with `strict=true`, on a `Chat` or a `Respond` | `Ollama does not enforce strict tool schemas`: its tool definition has no `strict` flag | leave `strict` unset |
| a `response_format`, or a `json_schema` text format, with a thinking effort | `Ollama applies a response_format only after the model's thinking ends`: a reply that skips thinking would be unconstrained | `reasoning_effort="none"` or unset, or no format |
| a `response_format` of another type, or a JSON Schema format without a schema object | `Ollama supports response_format text, json_object or json_schema`, `needs a schema object` | a JSON Schema format holding its schema, or `UniLM.json_object()` |
| a [`DocumentAttachment`](@ref) | `takes images and sound clips, not documents`: the `images` entry holds images and clips | the document's text in the message |
| an `ImageAttachment` with a `detail` level | `takes an image's bytes alone and no detail level` | the attachment without `detail` |
| a message with a `name` | `Ollama messages have no name field` | the message without `name` |
| a `tool` message whose `tool_call_id` matches no tool call of an earlier assistant turn | `answers no earlier tool call`: Ollama identifies a result by the called function's name | the `id` of the tool call it answers |
| on a `Respond`: `store`, `previous_response_id`, `background` or `conversation` | `Ollama's Responses API ignores`: it keeps no state between calls | the whole conversation in `input`, or a `Chat` |
| on a `Respond`: another field its compatibility page does not list as supported, such as `include`, `truncation` or `service_tier` | `Ollama's Responses API reads no`, `reads none of`: Ollama would drop it | remove it |
| a `Respond` with a `seed` | `Ollama's Responses API reads no seed` | a `Chat`, whose `seed` Ollama reads |
| a `Reasoning` with a summary, a context or a mode | `Ollama reads only reasoning.effort` | the `effort` alone |
| a `text` format with a `verbosity`, or of a type other than `text` and `json_schema` | `Ollama ignores text.verbosity`, `constrains output only to a json_schema text.format`: Ollama's compatibility page lists no `text` field (by Ollama 0.40.6's source; untested) | a `json_schema` text format without `verbosity` |
| a `json_schema` text format without its schema | `needs its schema`: without one Ollama constrains nothing (by Ollama 0.40.6's source; untested) | the schema in the format |
| an `input_image` given by file id | `Ollama reads input images from data URLs only` | the image's data URL |
| an `input_file` part, in any form | `documents no input_file parts`: the file might never reach the model | the document's text in an `input_text` part |
| an `InputMessage` with a `phase` | `Ollama's Responses API has no message phase` | the message without `phase` |
| a `Respond` tool other than a function tool | `Ollama's Responses API runs function tools here` | function tools |
| a function tool with `async`, `allowed_callers`, `defer_loading` or `output_schema` | `Ollama's Responses tools have no` | the tool without them |
| `tool_loop` on a `Respond` | `keeps no state between calls, so a Respond tool loop cannot chain its turns` | `tool_loop!` on a `Chat` |
| [`get_response`](@ref), [`cancel_response`](@ref), [`delete_response`](@ref) or [`list_input_items`](@ref) | `works on a stored response`: Ollama stores none | keep the `ResponseSuccess` that `respond` returns |
| an `Embeddings` request with a `user` | `Ollama embeddings take no user field` | remove it |
| an embeddings item holding both an image and a sound clip | `in a random order`: such an item embeds differently from one request to the next | the image and the clip as two items ([Images and sound](@ref ollama_embedding_media)) |
| an EmbeddingGemma request without a `task`, a `dimensions` it was not trained for, or a `title` with a task other than `:document` | `expects a task prompt`, `returns 768, 512, 256, 128 dimensions`, `reads with task = :document only` | what the [task prompts](@ref ollama_embeddings__next) say |
| an `Embeddings` request with a `title`, or a `task` other than `:none`, for a model whose prompts UniLM does not know | `has no task prompts, so a title has nowhere to go`, `UniLM knows the task prompts of EmbeddingGemma models and gemini-embedding-2 only` | no title, and `task = :none` with the model's own prefix in the text |
| a transcription without a `model` | `Ollama has no default transcription model` | a model that hears audio, such as `"gemma4:e4b"` |
| a transcription with `temperature`, `languages` or `keywords` | `runs at temperature 0 and takes no`, `takes one`, `takes no` | remove it; one `language` is taken |
| a transcription `response_format` other than `"json"` | `reports a failure that arrives after its reply has begun only as JSON`, `transcribes to JSON only` | leave it unset, and read the text with `transcript_text` |
| [`load_model`](@ref) through an endpoint with `keep_alive=0` | `would unload the model`: Ollama reads such a request as an unload | `keep_alive` unset, or a number of seconds |

The examples of First call, Tools, Structured output, Reasoning and Attachments were recorded on 21 October
2026 on `gemma4:e4b` with UniLM's development branch after 0.24.0 and Ollama 0.40.6; those of Conversations,
Streaming, and respond and FIM on 25 October 2026 on `gemma4:e4b` and `qwen2.5-coder:1.5b-base` with UniLM's
development branch after 0.26.0 and Ollama 0.40.11; all on an Apple M4 Max with 64 GB.

See also: [`OllamaEndpoint`](@ref), [`OllamaOptions`](@ref), [`Chat`](@ref), [`chatrequest!`](@ref) and
[`LLMSuccess`](@ref) in the API reference, and the [Ollama API reference](@ref ollama_api) page; the
[Local decision models](@ref ollama_decisions_start) entry in the Decisions chapter.
