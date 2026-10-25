# [The generateContent API](@id gemini_generate_content_page)

```@setup gemini
using UniLM
```

`GEMINIServiceEndpoint` turns each request field of a `Chat` into its generateContent counterpart. A
field with none is refused before the request is sent, except two that are not sent:
`parallel_tool_calls`, which a `Chat` with tools sets on its own ([Tools](@ref gemini_tools)), and `n`,
which a `Chat` takes only as 1, the one candidate every request gets.

| `Chat` field | generateContent | Rules |
| :--- | :--- | :--- |
| system messages | `systemInstruction` | [Conversations](@ref gemini_conversations) |
| user messages, their attachments | `user` turns: `inlineData` parts before the text | [Attachments](@ref gemini_media) |
| assistant messages, their `tool_calls` | `model` turns: the parts the reply arrived in, else its text and `functionCall` parts | [Conversations](@ref gemini_conversations), [Tools](@ref gemini_tools) |
| `tool` messages | `functionResponse` parts, consecutive ones in one `user` turn | [Tools](@ref gemini_tools) |
| `tools`, `tool_choice` | `tools[].functionDeclarations`, `toolConfig.functionCallingConfig` | [Tools](@ref gemini_tools) |
| `response_format` | `generationConfig.responseFormat` | [Structured output](@ref gemini_structured_output) |
| `reasoning_effort` | `generationConfig.thinkingConfig.thinkingLevel` | [Reasoning](@ref gemini_reasoning) |
| `model` | the URL's `models/{model}` | the [thinking table](@ref gemini_reasoning) |
| `max_completion_tokens`, else `max_tokens` | `generationConfig.maxOutputTokens` | [Setup](@ref gemini_setup) |
| `temperature`, `top_p`, `stop`, `seed` | `generationConfig.temperature`, `topP`, `stopSequences`, `seed` | [Request options](@ref gemini_request_options) |
| `safety_identifier`, `store`, `service_tier` | `labels.safety_identifier`, `store`, `serviceTier` | [Request options](@ref gemini_request_options) |
| `stream` | the URL's `:streamGenerateContent?alt=sse` | [Streaming](@ref gemini_streaming) |
| `prompt_cache_options`, `moderation` and the fields with no counterpart | refused | [Limits and refusals](@ref gemini_limits) |

## [Reasoning](@id gemini_reasoning)

`reasoning_effort` sets the request's thinking level, `thinkingConfig.thinkingLevel`, and a
[`Reasoning`](@ref)'s `effort` sets a `Respond`'s, `thinking_level` on the Interactions API
([Thinking on the Interactions API](@ref gemini_interactions_thinking)). A `reasoning_effort` of `"minimal"`,
`"low"`, `"medium"` or `"high"` is checked against the table below; `"none"`, `"xhigh"` and
`"max"` throw `ArgumentError` on every model ([Limits and refusals](@ref gemini_limits)).

```@eval
Main.ProviderFacts.gemini_thinking_table()
```

A model id takes the row of the same id, never the row of an id it merely starts with, since
distinct models share prefixes (`gemini-3.8-flash` and `gemini-3.8-flash-tts`). For an id no row
names, a valid level goes to the server, which answers for it. The columns:

- **Levels with respond (Interactions)**: the levels a `Reasoning` effort may take on a `Respond`.
- **Levels with Chat (generateContent)**: the levels `reasoning_effort` may take on a `Chat`; on the
  `gemini-2.5-` rows, none, with the reason UniLM gives: generateContent sets a Gemini 2.5 model's
  thinking with a token budget, which UniLM does not send. That refusal holds for every
  `gemini-2.5-` id, listed or not.

The `-latest` aliases and the ids Google routes to their successors ([Models](@ref gemini_models))
take the levels of the models that served them when this release was made. A request without
`reasoning_effort` sends no level, and the model thinks at its own default: "If you don't specify a
thinking level, Gemini will use the model's default thinking level (e.g., `"medium"` for Gemini 4
Argon, Gemini 3.8 Flash, and Gemini 3.6 Flash)", and "You cannot disable thinking for Gemini 4
Argon. Gemini 3 Flash and Flash-Lite also do not support full thinking-off"
([thinking](https://ai.google.dev/gemini-api/docs/generate-content/thinking)).

```@example gemini
chat = Chat(service=GEMINIServiceEndpoint, reasoning_effort="low")
push!(chat, Message(Val(:system), "You are a concise assistant."))
push!(chat, Message(Val(:user), "What is the smallest positive integer that leaves a remainder of 1 " *
    "when divided by 2, 3, 4, 5 or 6, and is divisible by 7? Answer with the number only."))
result = chatrequest!(chat)
text(result)
```

```@example gemini
u = result.usage
(; u.completion_tokens, u.reasoning_tokens, summary = reasoning_text(result))
```

The model thought, and `reasoning_tokens` counts it, but the reply holds no text of the thinking.
Google returns thought summaries to a request that enables them — "You can enable thought summaries
by setting `includeThoughts` to `true` in your request configuration"
([thought summaries](https://ai.google.dev/gemini-api/docs/generate-content/thinking#summaries)) —
and a `Chat` does not, so [`reasoning_text`](@ref) gives `nothing` for a generateContent reply. A
`Respond` asks for them with `Reasoning(summary="auto")`, and [`reasoning_summaries`](@ref) reads
them ([Thinking on the Interactions API](@ref gemini_interactions_thinking)).

The thinking is billed as output — Google lists each model's "Output price (including thinking
tokens)" ([pricing](https://ai.google.dev/gemini-api/docs/pricing)) — and `maxOutputTokens` bounds
it with the answer: "If the model hits this limit while reasoning, it stops generating with
`finish_reason: MAX_TOKENS` and returns truncated or empty output (while still billing for any
thinking tokens generated)"
([thinking](https://ai.google.dev/gemini-api/docs/generate-content/thinking)). The turn then reads
`finish_reason == "length"`; lower the level rather than the limit. A reply's thought signatures
travel on its parts, which UniLM sends back as they came ([Conversations](@ref gemini_conversations)).

## [Tools](@id gemini_tools)

Each `Tool` in `tools` goes out as a function declaration: its `name`, its `description` when it
has one, and its `parameters` as `parametersJsonSchema`, the JSON Schema whole. Google's reference
gives a declaration's name "Must be a-z, A-Z, 0-9, or contain underscores, colons, dots, and dashes,
with a maximum length of 128", and the name a call and its response carry "Must be a-z, A-Z, 0-9, or
contain underscores and dashes, with a maximum length of 128"
([generateContent reference](https://ai.google.dev/api/generate-content)); UniLM sends a name as
given, so a name of letters, digits, `_` and `-` holds for both. `tool_choice` becomes the request's
function-calling mode:

| `tool_choice` | `toolConfig.functionCallingConfig` | With a tool marked `strict=true` |
| :--- | :--- | :--- |
| unset | none sent: the API's default mode | `{"mode": "VALIDATED"}` |
| `"auto"` | `{"mode": "AUTO"}` | `{"mode": "VALIDATED"}` |
| `"required"` | `{"mode": "ANY"}` | the same |
| `UniLM.GPTToolChoice(func=:get_weather)` | `{"mode": "ANY", "allowedFunctionNames": ["get_weather"]}` | the same |
| `"none"` | `{"mode": "NONE"}` | the same |

A declaration has no `strict` flag: a tool marked `strict=true` sets the request's mode, for every
function the request declares. Google's modes: `VALIDATED` — "The model is constrained to predict
either function calls or natural language, and ensures function schema adherence"; `AUTO` — "The
model decides whether to generate a natural language response or suggest a function call"; `ANY` —
"The model is constrained to always predict a function call and ensures function schema
adherence"; `NONE` — "The model is *prohibited* from making function calls"
([function calling](https://ai.google.dev/gemini-api/docs/generate-content/function-calling#function_calling_modes)).
Another `tool_choice` string is refused ([Limits and refusals](@ref gemini_limits)).

```@example gemini
strict = Tool(func=FunctionSignature(name="get_weather", description="The current weather in a city",
    parameters=Dict("type" => "object", "properties" => Dict("city" => Dict("type" => "string")),
                    "required" => ["city"], "additionalProperties" => false), strict=true))
chat = Chat(service=GEMINIServiceEndpoint, tools=[strict])
push!(chat, Message(Val(:system), "You are a concise assistant."))
push!(chat, Message(Val(:user), "Is it raining in Oslo?"))
print(UniLM.encode_request(chat.service, chat))
```

A `Chat` with `tools` has `parallel_tool_calls = false` unless it sets another value, and
generateContent has no field for it, so UniLM sends nothing: the model decides whether to call
several functions in one turn, which it may ("The model also supports calling multiple functions in
a single turn",
[parallel function calling](https://ai.google.dev/gemini-api/docs/generate-content/function-calling#parallel_function_calling)).

A call's `id` is the one Google sends — "Gemini 3 now always returns a unique `id` with every
`functionCall`"
([function calling](https://ai.google.dev/gemini-api/docs/generate-content/function-calling)) — and
a call that comes without one gets `unilm_call_1`, `unilm_call_2`, … by its place in the turn, so
its result can name it; such an id is never sent, and the call and its result go back without one.
The `unilm_call_` prefix is reserved. The `tool` messages that answer a turn become one user turn of
`functionResponse` parts, each with the call's name: a result whose text is a JSON object is sent as
that object, and any other text as `{"result": text}`, since Google takes "The function response in
JSON object format"
([generateContent reference](https://ai.google.dev/api/generate-content)). A `tool`
message whose `tool_call_id` matches no call of an earlier turn is refused
([Limits and refusals](@ref gemini_limits)).

`tool_loop!` runs the calls and sends their results back until the model answers in text:

```@example gemini
weather = CallableTool(
    Tool(func=FunctionSignature(name="get_weather", description="The current weather in a city",
        parameters=Dict("type" => "object", "properties" => Dict("city" => Dict("type" => "string")),
                        "required" => ["city"]))),
    (name, args) -> "18 °C and sunny in $(args["city"])")
chat = Chat(service=GEMINIServiceEndpoint, tools=[weather])
push!(chat, Message(Val(:system), "You are a concise assistant."))
push!(chat, Message(Val(:user),
    "What is the weather in the capital of the country that hosted the 2016 Summer Olympics?"))
loop = tool_loop!(chat; tools=[weather])
text(loop.response)
```

```@example gemini
call = chat.messages[3]   # the turn that called the tool
call.tool_calls[1].func.name, call.tool_calls[1].func.arguments, [collect(keys(p)) for p in call.provider_content.blocks]
```

The turn that called the tool holds its `functionCall` part with a `thoughtSignature`, and
`tool_loop!` sent that turn back as it came, its result after it, in the request that got the
answer: "passing back thought signatures is mandatory for function calling"
([function calling](https://ai.google.dev/gemini-api/docs/generate-content/function-calling)).
A call's signature is also on its `ToolCall`, as `thought_signature`, which goes back with the call
when the turn holds none of Gemini's parts. A streamed turn's calls are on
[Streaming](@ref gemini_streaming), the tools of the Interactions API on
[The Interactions API](@ref gemini_interactions_tools), and the loop every provider shares on
[Tool Calling](@ref tools_guide).

## [Structured output](@id gemini_structured_output)

A JSON Schema `response_format` becomes
`generationConfig.responseFormat = {"text": {"mimeType": "APPLICATION_JSON", "schema": …}}`, and
`ResponseFormat()`, a `json_object` format, the same without a schema: JSON of any shape. Only the
schema is sent; the format's `name`, `description` and `strict` have no counterpart.

```@example gemini
using JSON
capital = UniLM.json_schema("capital", "A capital city and its country",
    Dict("type" => "object",
         "properties" => Dict("city" => Dict("type" => "string"), "country" => Dict("type" => "string")),
         "required" => ["city", "country"], "additionalProperties" => false))
chat = Chat(service=GEMINIServiceEndpoint, response_format=capital)
push!(chat, Message(Val(:system), "You are a concise assistant."))
push!(chat, Message(Val(:user), "Give the capital of Norway."))
JSON.parse(text(chatrequest!(chat)))
```

The reply's text is the JSON document. Google's structured output "supports a subset of the JSON
Schema specification", and "The model ignores unsupported properties"
([structured output](https://ai.google.dev/gemini-api/docs/generate-content/structured-output)):
among the keywords Google lists, an array takes `minItems` and `maxItems`, a number `minimum` and
`maximum`, a string `enum` and `format`, and no string length — so check such constraints on the
parsed answer. The same format on a `Respond` is on
[The Interactions API](@ref gemini_interactions_structured_output), and the formats every provider
shares are on [Structured Output](@ref structured_guide).

## [Attachments](@id gemini_media)

A user message's attachments go out as `inlineData` parts — the bytes in base64, under the MIME type
their leading bytes give — before its text, as Google advises for one image: "place the text prompt
*after* the image part in the `contents` array"
([image understanding](https://ai.google.dev/gemini-api/docs/generate-content/image-understanding)).
An [`ImageAttachment`](@ref) goes as `image/png`, `image/jpeg`, `image/gif` or `image/webp`, an
[`AudioAttachment`](@ref) as `audio/wav` or `audio/mp3`, and a [`DocumentAttachment`](@ref) as
`application/pdf`, without its `filename`, which inline data has no field for. Google lists PNG,
JPEG, WEBP, HEIC and HEIF as the image types it supports, and not GIF
([image understanding](https://ai.google.dev/gemini-api/docs/generate-content/image-understanding)):
a GIF is sent, and the API answers for it. A message with empty text sends the parts alone.

![A red disc on a white background](../../assets/red_circle.png)

```@example gemini
chat = Chat(service=GEMINIServiceEndpoint)
push!(chat, Message(Val(:system), "You are a concise assistant."))
push!(chat, Message(Val(:user), "What colour is the shape? One word.",
                    ImageAttachment("../../assets/red_circle.png")))
text(chatrequest!(chat))
```

A [sound clip](../../assets/secret_word.wav) of one spoken word goes the same way, and the usage counts
its tokens as audio:

```@example gemini
chat = Chat(service=GEMINIServiceEndpoint)
push!(chat, Message(Val(:system), "You are a concise assistant."))
push!(chat, Message(Val(:user), "Which word is spoken? Answer with the word only.",
                    AudioAttachment("../../assets/secret_word.wav")))
result = chatrequest!(chat)
text(result), result.usage.prompt_audio_tokens
```

An image's `detail` is its media resolution, the part's `mediaResolution`: `"low"` and `"high"`
are `MEDIA_RESOLUTION_LOW` and `MEDIA_RESOLUTION_HIGH`, and `"auto"` sends none, leaving the model's
default. Google documents a resolution per part for "Gemini 3 only"
([media resolution](https://ai.google.dev/gemini-api/docs/generate-content/media-resolution)); UniLM
refuses a level on the Gemini 2.5 and Gemma 4 models, and `"original"` on every model, and sends a
level for any other id ([Limits and refusals](@ref gemini_limits)). UniLM checks no size: Google
bounds inline data by the request, "Inline image data limits your total request size (text prompts,
system instructions, and inline bytes) to 20MB"
([image understanding](https://ai.google.dev/gemini-api/docs/generate-content/image-understanding#inline-image)).
Audio input is priced at the rate Google lists for it ([Cost and usage](@ref gemini_cost)), and the
attachment types are on [Attachments](@ref attachments_guide).

## [Streaming](@id gemini_streaming)

With `stream=true`, `chatrequest!` returns a `Task` at once, and `fetch` gives the same typed result
as an unstreamed call. The `callback` receives the text as it arrives, a `String` at a time, then the
assembled `Message`:

```@example gemini
chat = Chat(service=GEMINIServiceEndpoint, stream=true)
push!(chat, Message(Val(:system), "You are a concise assistant."))
push!(chat, Message(Val(:user), "List the planets of the solar system in order, one per line."))
deltas = String[]
task = chatrequest!(chat; callback=(chunk, close) -> chunk isa String && push!(deltas, chunk))
result = fetch(task)
text(result)
```

```@example gemini
deltas
```

```@example gemini
u = result.usage
(; u.prompt_tokens, u.completion_tokens, cost = estimated_cost(result))
```

A `Chat` streams through this `callback` keyword; `respond` streams the Interactions API
([Streaming on the Interactions API](@ref gemini_interactions_streaming)). Setting `close[] = true`
stops the stream ([Stopping a Stream Early](@ref streaming_stop)).

- **End.** A generateContent stream has no end-of-stream marker: the turn is final when the HTTP
  body ends after a chunk that carried the finish reason, and the `Message` reaches the callback
  then. A body that ends with no finish reason is an `LLMFailure` with status 200
  ([When something fails](@ref gemini_failures)).
- **Usage.** UniLM keeps the usage of the last chunk that carries one — Google: "`usage_metadata`
  is cumulative within each stream"
  ([long decode](https://ai.google.dev/gemini-api/docs/generate-content/long-decode)) — so a
  streamed turn has `usage` and adds its cost to `cumulative_cost` as an unstreamed one does. A
  stream that ends without committing its turn — stopped, cancelled, timed out, cut off or ended by a
  callback's exception — adds the usage its latest chunk had reported, as a lower bound of what was
  billed.
- **Parts.** Every part of every chunk is kept as it came, one that holds only a signature included,
  in the `provider_content` of the final `Message`: the text parts are not merged, as Google asks
  ("Don't combine two `Parts` that both contain signatures"), so the streamed turn goes back on the
  next request as an unstreamed one does.
- **Thinking.** No thinking reaches the callback, only the answer's text: a `Chat` asks for no thought
  summaries ([Reasoning](@ref gemini_reasoning)).
- **Tool calls.** UniLM takes each `functionCall` part as a whole call: `on_tool_call(tc)` runs once
  per call as soon as its chunk arrives, before the turn ends, and the final `Message` holds every
  call.
- **Blocks and errors.** A chunk whose `promptFeedback` names a `blockReason` ends the call with an
  `LLMCallError`, and an `error` object in the stream with an `LLMFailure` whose `status` is the
  error's HTTP `code` ([When something fails](@ref gemini_failures)). Text already passed to the
  callback cannot be recalled.

The concepts every provider shares are on [Streaming](@ref streaming_guide).

## [Finish reasons and safety](@id gemini_finish_reasons)

The candidate's `finishReason` becomes the message's `finish_reason`, streamed or not: `STOP` gives
`"stop"`, or `"tool_calls"` when the turn holds function calls; `MAX_TOKENS` gives `"length"`; the
filters `SAFETY`, `RECITATION`, `BLOCKLIST`, `PROHIBITED_CONTENT`, `SPII`, `IMAGE_SAFETY`,
`IMAGE_PROHIBITED_CONTENT` and `IMAGE_RECITATION` give `"content_filter"`; and every other value
passes through in lower case — `"malformed_function_call"`, `"unexpected_tool_call"`,
`"continuation"` ([Long generations](@ref gemini_long_decode)), `"language"`, `"escalation"`, … —
never as `"stop"`. A candidate without a `finishReason` reads `nothing`, and function calls under a
reason other than `STOP` keep that reason, so `tool_loop!` does not run them. Google's reference
lists every value ([FinishReason](https://ai.google.dev/api/generate-content#FinishReason)).

A turn the filters stopped with no text has no content: its `refusal_message` is the candidate's
`finishMessage`, Google's account of the stop, else "Model response blocked by safety filter.", and
such a turn, when it holds no part, is left out of the next request. A blocked prompt gets no
candidate at all — "If set, the prompt was blocked and no candidates are returned"
([generateContent reference](https://ai.google.dev/api/generate-content)) — which
UniLM returns as an `LLMCallError` naming the `promptFeedback`
([When something fails](@ref gemini_failures)). A `Chat` sends no `safetySettings`, so Google's
default settings apply; UniLM has no field for them. The end user a request is made for goes as a
request label ([Request options](@ref gemini_request_options)).

## [Request options](@id gemini_request_options)

`temperature` and `top_p` reach the models released before Gemini 3.6 Flash, from which on, Google
says, "`temperature`, `top_p`, and `top_k` are deprecated. The API ignores these parameters and
returns an error in future model generations"
([what's new in Gemini 3.6](https://ai.google.dev/gemini-api/docs/whats-new-gemini-3.6)). UniLM takes
them only for these exact ids, on a `Chat`, a `Respond` and `GEMINIOpenAIServiceEndpoint` alike, and
refuses them for any other ([Limits and refusals](@ref gemini_limits)):

- `gemini-3.1-pro-preview` and `gemini-3.1-pro-preview-customtools`;
- `gemini-3.1-flash-lite` and `gemini-3-flash-preview`;
- `gemini-2.5-pro`, `gemini-2.5-flash` and `gemini-2.5-flash-lite`;
- `gemma-4-31b-it` and `gemma-4-26b-a4b-it`;
- the image models `gemini-3.1-flash-lite-image`, `gemini-3-pro-image`, `gemini-3.1-flash-image` and
  `gemini-2.5-flash-image`.

The same ids are the ones a `Chat` may end with an assistant turn on ([Conversations](@ref gemini_conversations)).

- `stop` becomes `stopSequences`, a single string a list of one; Google takes "up to 5"
  ([GenerationConfig](https://ai.google.dev/api/generate-content#v1beta.GenerationConfig)).
- `seed` becomes `generationConfig.seed`, an int32; a seed outside its range is refused.
- `store` is the request's `store`, which "Configures the logging behavior for a given request. If
  set, it takes precedence over the project-level logging config"
  ([generateContent reference](https://ai.google.dev/api/generate-content)).
- `safety_identifier` becomes the request label `safety_identifier`, the key Google names for "Safety
  identifiers from aggregators", whose values "can be no longer than 63 characters (Unicode
  codepoints) and can only contain lowercase letters, numeric characters, underscores, and dashes"
  ([generateContent reference](https://ai.google.dev/api/generate-content)): a 64-character
  SHA-256 hex digest is one too long, so send 32 hex characters of one. A value outside the rule is
  refused before the request.
- `n` is not sent: a request gets one candidate.

`service_tier` takes `"standard"`, `"flex"` and `"priority"`, sent as `serviceTier`, and on a
`Respond` also `"deferred"` ([The Interactions API](@ref gemini_interactions_options)). Flex costs
half the standard rates and its latency is "Minutes (1--15 min target)"; Priority costs "75% to
100% over Standard rates", and a request over its limits is downgraded: "Downgraded requests are
billed at the standard rate"
([optimization](https://ai.google.dev/gemini-api/docs/optimization),
[priority inference](https://ai.google.dev/gemini-api/docs/priority-inference)). The tier that
billed a call is `usage.service_tier`: the reply's `x-gemini-service-tier` header, which Google
names as the way "to detect if requests are being frequently downgraded to `standard`", else its
`usageMetadata.serviceTier`; [Cost and usage](@ref gemini_cost) prices it. Google recommends
"increasing client-side timeouts to 10 minutes or more" for Flex, whose requests "may sit in a
queue" ([Flex inference](https://ai.google.dev/gemini-api/docs/generate-content/flex-inference)), and
a stream's wait for its headers counts against `stream_idle_timeout`, so give a Flex call a longer
bound, such as `config=RequestConfig(request_timeout=900, stream_idle_timeout=900)`
([Timeouts & Retries](@ref timeouts_guide)).

[`UniLM.encode_request`](@ref) returns the body `chatrequest!` sends, or throws the refusal the table
in [Limits and refusals](@ref gemini_limits) lists, with no key and no request:

```@example gemini
chat = Chat(service=GEMINIServiceEndpoint, service_tier="flex", stop="END", seed=7,
            safety_identifier="user_42", store=false)
push!(chat, Message(Val(:system), "You are a concise assistant."))
push!(chat, Message(Val(:user), "Count to five."))
print(UniLM.encode_request(chat.service, chat))
```

`JSON.json(chat)` is not this body: it writes the `Chat` in the Chat Completions shape of the
OpenAI-compatible wire.

## [Long generations](@id gemini_long_decode)

Gemini 4 Argon writes past one request's output cap — "One hop returns up to 65,536 output tokens.
Non-streaming REST requests return up to 32,768" — and a reply that reaches the cap with tokens left
"ends with `finish_reason: CONTINUATION` and a `continuation_token`"
([long decode](https://ai.google.dev/gemini-api/docs/generate-content/long-decode)). `chatrequest!`
continues such a turn itself, as Google's loop does: the same request again with the latest token,
the parts of every reply joined into one assistant turn, and its usage the replies' summed. It sends
at most `max(8, cld(budget, cap) + 1)` requests, the budget being `max_completion_tokens`, else
`max_tokens` (8 when neither is set), and the cap 32,768 unstreamed or 65,536 streamed. A turn still
unfinished after the last of them, or after a `CONTINUATION` reply that produced no output token,
returns with `finish_reason == "continuation"`. Each request has the whole `RequestConfig` budget,
and a streamed turn streams every request into the same callback. A request that fails ends the call
with nothing committed, as does a stop or cancel before the last reply's finish reason; the replies
already received are added to `cumulative_cost`. Continuation is exercised against a local mock: a
live `CONTINUATION` reply has not been observed for this release. `GEMINIOpenAIServiceEndpoint` does
not continue.

Gemini 4 Argon also thinks for minutes: on a non-streaming call "The server closes the connection after
10 minutes", on a stream "The server closes the stream after 20 minutes", and Google advises "Set your
HTTP read timeout to 20 minutes"
([long decode](https://ai.google.dev/gemini-api/docs/generate-content/long-decode)). A
`chatrequest!` or `respond` call to a model whose id starts `gemini-4-`, on either endpoint, takes
`request_timeout`, `stream_idle_timeout` and `total_deadline` of 1200 seconds while no
`RequestConfig` of yours is set in any channel; one you set replaces them as a whole
([Timeouts & Retries](@ref timeouts_guide)). On the Interactions API, a long answer needs a background
request ([Background requests](@ref gemini_background)).

## [Prompt caching](@id gemini_caching)

Gemini caches repeated prompt prefixes on its own: "Implicit caching is enabled by default for all
Gemini 2.5 and newer models", and "We automatically pass on cost savings if your request hits
caches", for a prompt of at least the model's minimum — 4,096 tokens on Gemini 4 Argon, 3.8 Flash,
3.6 Flash and 3.1 Pro Preview, 2,048 on Gemini 2.5 Flash and 2.5 Pro
([context caching](https://ai.google.dev/gemini-api/docs/caching)); Google gives "no cost saving
guarantee"
([context caching on generateContent](https://ai.google.dev/gemini-api/docs/generate-content/caching)).
A cache hit is counted in `cached_tokens`, part of `prompt_tokens` ([First call](@ref gemini_first_call)),
and UniLM bills it at the cached-input rate and the rest of the prompt at the input rate, with no
premium for writing the cache ([Cost and usage](@ref gemini_cost)). A `Chat` sends its whole
conversation each turn, so the next turn starts with the previous one's parts while the system
messages and the earlier turns stay as they were; on Gemini 4, "Gemini caches the full previous
request only if the new request adds at least 2,050 tokens"
([context caching](https://ai.google.dev/gemini-api/docs/caching)). Explicit caching — cached
contents, created ahead and named in a request — is an API of its own that UniLM does not wrap, so
`prompt_cache_options` has nowhere to go and is refused, on a `Chat` and a `Respond`
([Limits and refusals](@ref gemini_limits)).

## [Cost and usage](@id gemini_cost)

[`estimated_cost`](@ref) prices a result with the row of the table below for the model that served
it ([First call](@ref gemini_first_call)), and [`cumulative_cost`](@ref) adds up the calls of a
`Chat`. Of the usage, it bills the cache hits at the cached-input rate, the rest of the input — a
tool's input included — at the input rate, and all the output, thinking included, at the output rate
([pricing](https://ai.google.dev/gemini-api/docs/pricing)). Every rate is then multiplied by each
factor that applies:

- the billed tier, `usage.service_tier`: 0.5 for `"flex"` and 1.8 for `"priority"` on input and
  output, and on a cache hit the same, except where a model's table lists another cache-hit price —
  the standard one on Flex for `gemini-3.1-pro-preview`, `gemini-3.1-pro-preview-customtools`,
  `gemini-3-flash-preview` and the `gemini-2.5-` models, and \$0.02 (Flex) and \$0.05 (Priority) per 1M
  for `gemini-3.5-flash-lite`, whose standard one is \$0.03;
- a prompt over 200,000 tokens on `gemini-3.1-pro-preview`, `gemini-3.1-pro-preview-customtools` or
  `gemini-2.5-pro`: 2 on the input and cache-hit rates and 1.5 on the output rate, the whole request.

Audio input bills at the input price the page lists for audio, at the same factors: the model's own
audio price where its table lists one apart — per 1M tokens, \$0.50 on `gemini-3.1-flash-lite`, \$1.00
on `gemini-3-flash-preview` and `gemini-2.5-flash`, \$0.30 on `gemini-2.5-flash-lite` — and
otherwise the row's input rate, since an input price that names no modality, or names audio among
others, is every modality's. Three cases are estimated at `0.0`, with one warning per model and
endpoint: audio on a model the page prices without audio (the image models) or does not list, audio
in the output, and audio in a call that also hit the cache on one of those four models, whose
cached-audio price the usage cannot tell apart.

A reply that names a tier other than standard, flex and priority — `"deferred"`, say — is estimated
at the standard rates, with one warning per model and tier. A model with no row, any other id
included, is estimated at `0.0`, with one warning per model id: pass [`estimated_cost`](@ref) a
`pricing` table that holds its row, or add the row to [`DEFAULT_PRICING`](@ref). A failure costs
`0.0`, since it carries no usage. How a stream reports its usage is on [Streaming](@ref gemini_streaming),
and a continued turn's on [Long generations](@ref gemini_long_decode). [`cumulative_cost`](@ref) is a
lower bound of the bill: it adds nothing for an attempt UniLM retried, a stream Google ended with an
error, or an unstreamed reply UniLM could not decode ([Cost Tracking](@ref cost_guide)), and it does
not price the fees of Google's hosted tools ([Hosted tools](@ref gemini_hosted_tools)).

The rows are Google's list prices for the standard tier as checked on 22 October 2026. The prices of
Gemini 4 Argon hold "through November 30, 2026", and those of Gemini 3.8 and 3.6 Flash "through
December 31, 2026", after which Google lists higher ones
([pricing](https://ai.google.dev/gemini-api/docs/pricing)): check Google's page before relying on an
estimate.

```@eval
Main.ProviderFacts.price_table(:gemini)
```

Recorded on 25 October 2026 on `gemini-3.8-flash` with UniLM's development branch after 0.26.0 and the Gemini
API at `v1beta`.
