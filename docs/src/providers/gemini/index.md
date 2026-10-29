# [Google Gemini](@id gemini_start)

```@setup gemini
using UniLM
```

Google makes the Gemini models and serves them through two native APIs, both of which
`GEMINIServiceEndpoint` speaks: generateContent, which a `Chat` reaches through `chatrequest!` and
`tool_loop!`, and the Interactions API, which `respond` and the `tool_loop` that takes an input or a
`Respond` reach. Image generation and speech go through generateContent too.
`GEMINIOpenAIServiceEndpoint` speaks Google's OpenAI-compatible endpoint, which UniLM uses for
embeddings. Of Google's other APIs — the Files, Batch, Live and token-counting APIs, explicit context
caching, managed agents such as Deep Research, music and video generation — none is available. The
verbs of the capabilities the table below lists as not available refuse both endpoints, but one kind:
the lifecycle verbs of stored responses read, cancel and delete stored interactions
([Background requests](@ref gemini_background)). `speak` reads aloud through Gemini TTS, and
`transcribe` refuses both endpoints ([Audio](@ref gemini_audio)).

```@eval
Main.ProviderFacts.at_a_glance(:gemini)
```

| I want to… | Use | You get |
| :--- | :--- | :--- |
| [choose between the APIs](@ref gemini_apis) | `chatrequest!`, `respond`, or `GEMINIOpenAIServiceEndpoint` for embeddings | the API that serves the job |
| [make a first call](@ref gemini_first_call) | `chatrequest!(chat)` | an `LLMSuccess`: the reply, its usage and the model that served it |
| [hold a conversation](@ref gemini_conversations) | `push!` a message, then `chatrequest!` again | each reply appended to the `Chat` and sent back as it arrived |
| [stream a reply](@ref gemini_streaming) | `stream=true` and a `callback` | the text as it arrives, then the whole message |
| [let Gemini call my functions](@ref gemini_tools) | `tools` and `tool_loop!` | the calls run and answered until the model replies |
| [get JSON that fits a schema](@ref gemini_structured_output) | a JSON Schema `response_format` | a reply constrained to the schema |
| [control thinking](@ref gemini_reasoning) | `reasoning_effort` at a level the thinking table lists | that thinking level, counted in `reasoning_tokens` |
| [send images, audio and PDFs](@ref gemini_media) | `ImageAttachment`, `AudioAttachment`, `DocumentAttachment` | the files sent inline, before the message's text |
| [use the Interactions API](@ref gemini_interactions_page) | `respond(input; service=GEMINIServiceEndpoint)` | a `ResponseSuccess`, its steps read by `output_text` and `function_calls` |
| [search the web, run code or read a page](@ref gemini_hosted_tools) | `gemini_google_search()`, `gemini_code_execution()`, `gemini_url_context()` | tools Google runs, on the Interactions API |
| [generate or edit an image](@ref gemini_images) | `generate_image`, `edit_image` with `service=GEMINIServiceEndpoint` | an `ImageSuccess` holding the image |
| [read text aloud](@ref gemini_audio) | `speak(text; service=GEMINIServiceEndpoint)` | 24 kHz speech as WAV or PCM |
| [embed text](@ref gemini_embeddings) | `embeddingrequest!` with `service=GEMINIOpenAIServiceEndpoint` | one vector per input |
| [pay less, or get answers sooner](@ref gemini_request_options) | `service_tier="flex"` or `"priority"` | the tier the reply reports, priced at its rates |
| [know what a call cost](@ref gemini_cost) | `estimated_cost`, `cumulative_cost` | USD estimates from the reported usage |
| [list the models my key can use](@ref gemini_models) | `list_models(service=GEMINIServiceEndpoint)` | each model's token limits and generation methods |
| [handle a failure](@ref gemini_failures) | the result's type | the status or the exception, never an empty success |

## [Setup](@id gemini_setup)

Set your API key in the environment Julia starts from:

```bash
export GEMINI_API_KEY="your-gemini-key"
```

UniLM reads the variable each time a request is sent, and reads no other: not `GOOGLE_API_KEY`,
which Google's own SDKs also read. A call without it returns its call-error result —
`LLMCallError` from `chatrequest!`, `ResponseCallError` from `respond`, and so on — whose `cause`
is the `KeyError` naming the variable. Both endpoints read the same key. A `Chat` or a `Respond`
reaches Gemini with `service=GEMINIServiceEndpoint`, the bare type (`GEMINIServiceEndpoint()` is the
type too), and with no `model` it takes the default model named at the top of this page.
generateContent requests go to
`https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent`, a streamed one
to `:streamGenerateContent?alt=sse`, and Interactions requests to
`https://generativelanguage.googleapis.com/v1beta/interactions`, each with the key in
`x-goog-api-key`.

A `Chat` sends its `max_completion_tokens`, else its `max_tokens`, as `maxOutputTokens`, and a
`Chat` that sets neither sends none: the model's own output limit applies. The limit counts the
thinking with the answer, so a low one can end a turn with `finish_reason == "length"` before any
text ([Reasoning](@ref gemini_reasoning)).

## [Which API for what](@id gemini_apis)

| Job | API | Call | Default model |
| :--- | :--- | :--- | :--- |
| chat, tools, streaming, JSON, thinking, media input | generateContent | `chatrequest!`, `tool_loop!` on a `Chat` | the default at the top of this page |
| server-side state, hosted tools, background requests | Interactions | `respond`, `tool_loop` | the default at the top of this page |
| image generation and editing | generateContent | `generate_image`, `edit_image` | `gemini-nano-banana-2.1` |
| speech | generateContent | `speak` | `gemini-3.8-flash-tts` |
| embeddings | OpenAI-compatible | `embeddingrequest!` with `GEMINIOpenAIServiceEndpoint` | `gemini-embedding-2` |
| the model listing | models | `list_models` | — |

Google calls the Interactions API "the best way to build with Gemini models and agents" and
recommends it "for all new projects", and generateContent legacy, though it "remains fully
supported"; "all new models, multimodal capabilities, tools, and agentic features will launch on the
Interactions API"
([Interactions overview](https://ai.google.dev/gemini-api/docs/interactions-overview)). In UniLM,
`Chat` and its verbs speak generateContent, and `respond` speaks Interactions:
[The Interactions API](@ref gemini_interactions_page) has what differs there. A `Chat` sent to
`GEMINIOpenAIServiceEndpoint` reaches the same models on the OpenAI-compatible wire, with fewer of
the rules on these pages ([The OpenAI-compatible endpoint](@ref gemini_compatible)); to code that
does not already use a client of that wire, Google recommends that you "call the Gemini API
directly" ([OpenAI compatibility](https://ai.google.dev/gemini-api/docs/openai)).

## [First call](@id gemini_first_call)

A `Chat` starts with its system message; `chatrequest!` sends it and returns a typed result, and
`text` reads the reply — or throws, on a failure:

```@example gemini
chat = Chat(service=GEMINIServiceEndpoint)
push!(chat, Message(Val(:system), "You are a concise assistant."))
push!(chat, Message(Val(:user), "In one sentence: what is a prime number?"))
result = chatrequest!(chat)
text(result)
```

```@example gemini
result.model, result.message.finish_reason
```

`result.model` is the model that served the reply, from the reply's `modelVersion`: Google routes
a deprecated id to its successor ([Models](@ref gemini_models)), which then names itself here — as
both routed ids did on 22 October 2026 — and is the model [`estimated_cost`](@ref) prices. The reply's `finishReason` becomes `finish_reason`
([Finish reasons and safety](@ref gemini_finish_reasons)).

```@example gemini
u = result.usage
(; u.prompt_tokens, u.completion_tokens, u.reasoning_tokens, u.service_tier, cost = estimated_cost(result))
```

`prompt_tokens` counts all the input — the cache hits (`cached_tokens`) and the input a tool
read (`toolUsePromptTokenCount`) included — and `completion_tokens` all the output, thinking
included, of which `reasoning_tokens` is the thinking. `service_tier` is the tier that billed the
call, and `prompt_audio_tokens` the input that was audio. How these become a price is in
[Cost and usage](@ref gemini_cost); the result types are on [Results](@ref results_guide).

## [Conversations](@id gemini_conversations)

A `Chat` keeps the conversation: each reply is appended to it, and the next `chatrequest!` sends the
whole conversation again.

```@example gemini
chat = Chat(service=GEMINIServiceEndpoint)
push!(chat, Message(Val(:system), "You are a concise assistant: answer in one short sentence."))
push!(chat, Message(Val(:user), "Which planet of the solar system is the largest?"))
text(chatrequest!(chat))
```

```@example gemini
push!(chat, Message(Val(:user), "And the smallest?"))
text(chatrequest!(chat))
```

```@example gemini
[m.role for m in chat.messages], cumulative_cost(chat)
```

A reply keeps the parts it arrived in, in its message's `provider_content`, and goes back as those
parts, their thought signatures intact:

```@example gemini
[collect(keys(part)) for part in chat.messages[3].provider_content.blocks]
```

Google asks for the signatures back — "passing back thought signatures is mandatory for function
calling" — and for each part as it came: "Don't merge a `Part` containing a signature with one
that does not"
([function calling](https://ai.google.dev/gemini-api/docs/generate-content/function-calling)).
A reply that holds none of Gemini's parts — written by hand, or answered by another provider —
goes as its text and one `functionCall` part per tool call. A reply that holds no part, no text
and no function call — a refusal, say, or a turn whose output went to thinking — is left out. What
`push!` refuses is the same on every provider ([Conversations with Chat](@ref chat_guide)).

Every system message becomes part of the request's `systemInstruction`, in order, joined with a
blank line: one pushed later changes the instructions for the whole conversation, not from where
it stands. The results of a reply's tool calls go back as one user turn of `functionResponse`
parts ([Tools](@ref gemini_tools)).

From Gemini 3.6 Flash and 3.5 Flash-Lite on, "If the last non-empty turn in the request is a `model`
turn, the API returns a `400` error"
([what's new in Gemini 3.6](https://ai.google.dev/gemini-api/docs/whats-new-gemini-3.6)): a
request that ends with an assistant message, a response prefill (a `Chat` with `history=false`),
is refused on those models before it is sent ([Limits and refusals](@ref gemini_limits)). With
`history=true`, `chatrequest!` throws `InvalidConversationError` for such a conversation on every
model, since the reply could not be appended.

## [Models](@id gemini_models)

The listing is the endpoint's `models` capability: [`list_models`](@ref) with
`service=GEMINIServiceEndpoint` asks Google's Models API (`GET /v1beta/models`) which models your
key can use, fetches every page of the answer, and returns a `ModelListSuccess` whose `models` are
[`ProviderModel`](@ref)s. Each holds the model's `id` (its `name` without `models/`) and
`display_name`, its `input_token_limit` and `output_token_limit`, the `capabilities` Google
reports — `supportedGenerationMethods` and whether the model thinks (`thinking`) — and the entry as
sent in `raw`; the listing carries no dates, so `created`, `deprecated` and `retires` are `nothing`
([Model Listings](@ref models_api)). An error status on any page is a `ModelListFailure`; a call
that brings no listing — no key, no connection, a timeout, a reply that is not the listing — is a
`ModelListCallError`. `list_models` throws `ArgumentError` for `GEMINIOpenAIServiceEndpoint`:
UniLM lists no models through it, though Google's compatible endpoint has a listing of its own,
`GET /v1beta/openai/models` ([OpenAI compatibility](https://ai.google.dev/gemini-api/docs/openai)).

A listing bills no tokens and is not recorded, so its output here is quoted from a live run:

```julia
r = list_models(service=GEMINIServiceEndpoint)
r isa ModelListSuccess || error("the listing failed: $r")
chat_models = [m for m in r.models if "generateContent" in get(m.capabilities, "supportedGenerationMethods", [])]
length(r.models), length(chat_models)
# Listed on 25 October 2026 with UniLM's development branch after 0.26.0 and Google's Models API
# (GET /v1beta/models):
# (60, 46)

[(m.id, m.input_token_limit, m.output_token_limit) for m in chat_models if startswith(m.id, "gemini-3.8")]
# 3-element Vector{Tuple{String, Int64, Int64}}:
#  ("gemini-3.8-flash", 1048576, 65536)
#  ("gemini-3.8-flash-tts", 8192, 16384)
#  ("gemini-3.8-flash-lite-tts", 8192, 16384)

filter(id -> !haskey(DEFAULT_PRICING, id), [m.id for m in chat_models if startswith(m.id, "gemini-")])
# 20-element Vector{String}:
#  "gemini-2.5-flash-preview-tts"
#  "gemini-2.5-pro-preview-tts"
#  "gemini-flash-latest"
#  "gemini-flash-lite-latest"
#  "gemini-pro-latest"
#  "gemini-2.5-flash-image"
#  "gemini-3.1-flash-lite-preview"
#  "gemini-3-pro-image-preview"
#  "gemini-3-pro-image"
#  "gemini-3.1-flash-image-preview"
#  "gemini-3.1-flash-image"
#  "gemini-3.1-flash-lite-image"
#  "gemini-nano-banana-2.1"
#  "gemini-omni-flash-preview"
#  "gemini-omni-1.1-flash"
#  "gemini-3.5-transcribe"
#  "gemini-3.1-flash-tts-preview"
#  "gemini-3.8-flash-tts"
#  "gemini-3.8-flash-lite-tts"
#  "gemini-2.5-computer-use-preview-10-2025"
```

Google routes two deprecated ids to their successors: `gemini-3.7-flash` "automatically routes API
requests to `gemini-3.8-flash`", and `gemini-3.5-flash` to `gemini-3.6-flash`
([deprecations](https://ai.google.dev/gemini-api/docs/deprecations)), both since the changelog entry of
October 8, 2026 ([changelog](https://ai.google.dev/gemini-api/docs/changelog)). Such a request meets the
rules of the id it names and is priced as the model that served it ([First call](@ref gemini_first_call)).
A `-latest` alias "will get hot-swapped with every new release of a specific model variation"
([models](https://ai.google.dev/gemini-api/docs/models)); the thinking table holds the aliases to the
levels of the models that served them when this release was made ([Reasoning](@ref gemini_reasoning)).
Before you use a newly listed model, check that it has a price row, as the last line above does: a
`Chat` or `Respond` result of a model without one is estimated at `0.0`
([Cost and usage](@ref gemini_cost)). Of the ids that line lists, the aliases bill at the rows of the
models that serve them, since a reply names its serving model (on 22 October 2026, `gemini-flash-latest`
answered as `gemini-3.8-flash` and `gemini-pro-latest` as `gemini-3.1-pro-preview`); the image and
speech models serve verbs that `estimated_cost` does not price
([Images, speech, embeddings & the compatible endpoint](@ref gemini_media_page)); the others would be
estimated at `0.0`.

## [Limits and refusals](@id gemini_limits)

A model's input and output token limits are in its [`list_models`](@ref) entry
([Models](@ref gemini_models)); UniLM checks a request against neither, nor the size of its media
([Attachments](@ref gemini_media)), except one rule of the Interactions API below. Rate limits are on
[When something fails](@ref gemini_failures).

A field a Gemini API has no counterpart for, or one whose counterpart UniLM does not map, is refused,
not dropped, since dropping it would send a request other than the one asked for. The one exception is a `Chat`'s `parallel_tool_calls`, which a
`Chat` with tools sets on its own and which is not sent, whatever its value ([Tools](@ref gemini_tools)).
Each request in the table below that the API would answer with an error is refused too.
`chatrequest!`, `tool_loop!`, `respond`, `tool_loop`, `generate_image`, `edit_image`, `speak` and
`embeddingrequest!` throw these `ArgumentError`s before anything is sent. A row that names the
thinking table holds for the ids of its rows ([Reasoning](@ref gemini_reasoning)); an id no row
names meets only the other rules.

| Refused | Why: the message's words | What to do |
| :--- | :--- | :--- |
| `temperature` or `top_p`, on a `Chat`, a `Respond` or `GEMINIOpenAIServiceEndpoint`, for a model other than the earlier ones listed in [Request options](@ref gemini_request_options) | `so UniLM sends sampling parameters only to`: from Gemini 3.6 Flash and 3.5 Flash-Lite on, the API ignores them | remove it |
| a `Chat` that ends with an assistant turn, on a model other than those earlier ones | `answers HTTP 400 to a request that ends with a model turn` | end with a user message; shape the answer with a system message or `response_format` |
| a `Respond` input that ends with an assistant message, on a model other than those earlier ones | `answers HTTP 400 to an input that ends with an assistant message` | end with a user message; shape the answer with `instructions` or `text` |
| `moderation`, or a `Chat` field native Gemini does not map: `presence_penalty`, `frequency_penalty`, `logit_bias`, `user`, `stream_options`, `verbosity`, `metadata`, `logprobs`, `top_logprobs`, `prediction`, `modalities`, `audio`, `web_search_options`, `prompt_cache_key`, `prompt_cache_retention` | `Native Gemini Chat does not support field(s)`: UniLM sends none of them to generateContent, though some have a counterpart there: `GenerationConfig`'s `presencePenalty`, `frequencyPenalty`, `responseLogprobs`, `logprobs`, `responseModalities` and `speechConfig`, the request's `labels` for `metadata` and `user`, and the `googleSearch` tool for `web_search_options` | remove it, or send the `Chat` through `GEMINIOpenAIServiceEndpoint` |
| `prompt_cache_options`, on a `Chat` or a `Respond` | `Gemini takes no prompt_cache_options`: Gemini caches repeated prefixes on its own, and UniLM does not wrap its explicit caching | remove it ([Prompt caching](@ref gemini_caching)) |
| `reasoning_effort` `"none"`, `"xhigh"` or `"max"`, on a `Chat` or in a `Reasoning` | `Gemini thinking effort must be minimal, low, medium, or high` | one of those four |
| a level the model's row of the thinking table does not list | `supports`, then the levels it takes, then `thinking effort` | one of them |
| any `reasoning_effort` on a `Chat` for a `gemini-2.5-` model | `Gemini 2.5 generateContent uses thinkingBudget` | leave it unset, or send a `Respond` with `Reasoning` |
| a `service_tier` other than `"standard"`, `"flex"` and `"priority"`, on a `Chat` or a `Respond`; `"deferred"` among them, which the API answers with HTTP 400 | `Gemini generateContent takes service_tier`, `Gemini's OpenAI-compatible endpoint takes service_tier`, `Gemini Interactions takes service_tier` | one of them |
| a `seed` outside the int32 range, on a `Chat` or a `Respond` | `Gemini takes a seed from`: `seed` is an int32 | a seed from -2,147,483,648 to 2,147,483,647 |
| a `safety_identifier` longer than 63 characters, or holding a character other than lowercase letters, letters of scripts without case (such as Chinese), numeric characters, `_` and `-` | `sends safety_identifier as a request label`: it goes as a request label, which Google's label rules bound | a shorter id, such as 32 hex characters of a hash |
| a `tool_choice` string other than `"auto"`, `"none"` and `"required"` | `Unknown Gemini tool_choice`, `Unknown tool_choice string` | one of them, or a named function |
| a `tool` message whose `tool_call_id` matches no tool call of an earlier assistant turn | `functionResponse references unknown tool_call id` | the `id` of the call it answers |
| a `response_format` whose type is not `text`, `json_object` or `json_schema`, or a JSON Schema format without a schema object | `supports response_format text, json_object, or json_schema`, `needs a schema object` | a format of those types, with its schema |
| an image with `detail="original"`: an `ImageAttachment`, or an `input_image` on a `Respond` | `Gemini has no media resolution for an image at its own dimensions` | `"low"`, `"high"` or `"auto"` |
| an `input_image` on a `Respond` whose `detail` is not `"auto"`, `"low"`, `"high"` or `"original"` | `an image's detail level on Gemini is` | `"low"`, `"high"` or `"auto"` |
| an image `detail` level on `gemini-2.5-pro`, `gemini-2.5-flash`, `gemini-2.5-flash-lite`, `gemini-2.5-flash-image`, `gemma-4-31b-it` or `gemma-4-26b-a4b-it` | `takes no per-image media resolution` | the image without `detail` |
| a `Respond` field with no Interactions counterpart: `parallel_tool_calls`, `truncation`, `metadata`, `user`, `include`, `max_tool_calls`, `top_logprobs`, `prompt_cache_key`, `prompt_cache_retention`, `conversation`, `context_management`, `stream_options`, `moderation` | `Gemini Interactions does not support the Respond field(s)` | remove it |
| `max_output_tokens` above 65,536 on a `gemini-4-argon` `Respond` without `background=true` | `takes max_output_tokens above 65536 only in the background` | `background=true` ([Background requests](@ref gemini_background)), or a lower `max_output_tokens` |
| `background=true` with `store=false` | `background=true requires a stored response`: the API answers it with HTTP 400 | leave `store` unset |
| `list_input_items` with `service=GEMINIServiceEndpoint` | `list_input_items is not supported by GEMINIServiceEndpoint`: the Interactions API lists no input items | keep the `input` you sent ([Background requests](@ref gemini_background)) |
| a `Reasoning` with `context` or `mode`, or with a `summary` other than `"auto"` | `does not support reasoning context or mode`, `supports only Reasoning(summary="auto")` | `Reasoning(effort=…, summary="auto")` |
| a `text` format with `verbosity`, or whose type is not `text`, `json_object` or `json_schema` with a schema | `does not support text verbosity`, `supports text formats text, json_object, or json_schema` | a JSON Schema `text` format |
| a tool that is neither a `FunctionTool` nor a `Dict`, such as `web_search()` or `file_search(…)` | `supports only FunctionTool/Dict tools` | a function tool, a hosted Gemini tool ([Hosted tools](@ref gemini_hosted_tools)) or a `Dict` in the API's shape |
| a `FunctionTool` with `async`, `allowed_callers`, `defer_loading` or `output_schema` set | `function tools do not support` | the tool without them |
| a `tool_choice` selector for another tool than a function | `hosted-tool selectors are not applicable` | `"auto"`, `"none"`, `"required"` or `tool_choice_function(name)` |
| an `InputMessage` with a `phase`, or with the role `system` or `developer` | `has no message phase`, `takes user and assistant messages as input` | send the guidance as `instructions` |
| a media part that is a web URL or a file id, or a text part with a cache breakpoint | `takes an image as bytes`, `takes a file as bytes`, `with no breakpoint to mark` | an attachment, or a base64 `data:` URL; `input_text(text)` |
| a `function_call_output` item without a `name` | `Gemini function_result requires a name` | `tool_result(call_id, name, output)` |
| an image request's `size`, `quality`, `background`, `output_format`, `output_compression` or `moderation`, or an edit's `mask` or `input_fidelity` | `Gemini image generation does not take`, `Gemini image editing does not take` | `aspect_ratio` and `resolution` ([Image generation](@ref gemini_images)) |
| an image request's `n` other than 1 | `Gemini returns one image per request` | one request per image |
| an edit's image path that is no file | `image not found:` | the path of the file |
| an `aspect_ratio` or a `resolution` outside Gemini's lists, or a resolution the model does not render | `Gemini aspect_ratio must be one of`, `must be one of` | a value the [Image generation](@ref gemini_images) section lists |
| `speak` with a blank `voice`, any `speed`, a `response_format` other than `"wav"` and `"pcm"`, or an empty input | `Gemini TTS takes a voice by name or id`, `has no speed control`, `response_format must be "wav"`, `speak needs text to read`: Gemini TTS has these formats, and no speed | a voice's name or id, the pace in `instructions`, `"wav"` or `"pcm"`, some text ([Audio](@ref gemini_audio)) |
| an `EmbeddingInput` with an image or a sound clip, on `GEMINIOpenAIServiceEndpoint` | `takes text only`: the compatible wire carries text | the text alone ([Embeddings](@ref gemini_embeddings)) |
| `speak` through `GEMINIOpenAIServiceEndpoint` | `GEMINIOpenAIServiceEndpoint has no speech route` | `service=GEMINIServiceEndpoint` |

## [When something fails](@id gemini_failures)

`chatrequest!` and `respond` return what the service or the network did as a typed result, and do
not throw it ([Results](@ref results_guide)). An `LLMFailure` or a `ResponseFailure` holds an HTTP
`status` and, as `response`, the body that came with it; an `LLMCallError` or a `ResponseCallError`
holds, as `error`, the text of an exception or of an error the reply reported, and the exception as
`cause` when there is one. UniLM sends a request again after a transient status — 408, 429, 500,
502 to 504, 520 to 524 and 529 — as often as the call's [`RequestConfig`](@ref) allows, and then
returns the last answer ([Timeouts & Retries](@ref timeouts_guide)); a streamed request is sent again only
while none of its callbacks has run. Google sends rate limits as 429: on generateContent a
`RESOURCE_EXHAUSTED` for "one of the API's rate limits (RPM, TPM, RPD, spend, etc.)", which UniLM
sends again, and on the Interactions API `rate_limit_exceeded` and `too_many_requests`, which it
sends again, or `quota_exceeded`, "your daily quota", which it returns at once
([generateContent errors](https://ai.google.dev/gemini-api/docs/generate-content/api-errors),
[Interactions errors](https://ai.google.dev/gemini-api/docs/api-errors)).

| What you see | Why | What to do |
| :--- | :--- | :--- |
| `ArgumentError` from a verb, nothing sent | the request breaks a rule UniLM checks before sending ([Limits and refusals](@ref gemini_limits)) | what its row says |
| `InvalidConversationError` from `chatrequest!`, nothing sent | with `history=true`, the conversation ends with an assistant message, so the reply could not be appended | a user message after it |
| a call-error result whose `cause` is a `KeyError` | `GEMINI_API_KEY` is not set ([Setup](@ref gemini_setup)) | set it |
| `LLMFailure` or `ResponseFailure` with `status` 400, 401, 402, 403 or 404 | Google refused the request, and UniLM does not send it again: `response` holds Google's error, `INVALID_ARGUMENT`, `PERMISSION_DENIED`, `invalid_request`, … ([generateContent errors](https://ai.google.dev/gemini-api/docs/generate-content/api-errors), [Interactions errors](https://ai.google.dev/gemini-api/docs/api-errors)); a 402 is a Prepay credit balance that is used up | what the error says; for a 402, add credit |
| `LLMFailure` or `ResponseFailure` with `status` 429 | a rate limit, which UniLM sent again until the call's `RequestConfig` ran out, or the Interactions API's daily quota (`quota_exceeded`), returned at once | fewer requests, more `max_attempts` or a longer `total_deadline`; for the quota, wait until it resets |
| `LLMFailure` or `ResponseFailure` with `status` 500, 503 or 504 | Google failed or was overloaded, on every attempt | the request again, later |
| `LLMCallError` whose `error` starts "Gemini response contained no candidates" | Google blocked the prompt: the reply has no candidate, and `promptFeedback` says why | another prompt |
| `LLMCallError` from a stream, its `error` naming "Gemini prompt blocked" | the same block, in a stream | another prompt |
| `LLMFailure` from a stream, its `status` the HTTP `code` of an error the stream carried | Google sent an error in the stream; it is sent again only while no callback has run | the request again |
| `LLMFailure` with `status` 200 from a stream | the stream ended before a finish reason; `response` holds what arrived, and nothing is appended | the turn again |
| a call-error result with a `UniLMTimeout` as `cause` | a bound of the call's `RequestConfig` ran out ([Timeouts & Retries](@ref timeouts_guide)); a `gemini-4-` model has longer bounds by default, and each request of a continued turn the whole budget ([Long generations](@ref gemini_long_decode)) | a longer bound, or a stream for a long reply |
| a call-error result whose `cause` is the connection's exception | the network failed, on each attempt the call's `RequestConfig` allows, or on a stream once a callback had run | the connection, then the request again |
| a call-error result whose `cause` is a `UniLMCancelled` | a stop before the turn's finish reason arrived: the stream's `close[] = true` (`:callback`) or a cancelled `CancelToken` (`:token`); nothing is appended | nothing, if the stop was yours |
| a call-error result whose `cause` is the exception your `callback` or `on_tool_call` threw | the exception ended the stream; it is not sent again, and nothing is appended; an `InterruptException`, or a recorded call's `ReplayMissError` or `RecordingWriteError`, is thrown at `fetch` instead | the callback |
| `ResponseCallError` whose `error` names a missing `"id"` or `"status"` | a 200 that is not an interaction ([The Interactions API](@ref gemini_interactions_page)) | the request again |
| `ResponseFailure` with `status` 200 from `respond` | the interaction's `status` is `"failed"`; `response` holds it | the request again |
| `ResponseCallError` from a streamed `respond`, its `error` the stream's error event | Google sent an `error` event in the stream, as its error reference describes ([Interactions errors](https://ai.google.dev/gemini-api/docs/api-errors)) | the request again |
| `finish_reason == "length"` | the reply reached `maxOutputTokens` or the model's output limit, thinking included | a higher `max_tokens`, or a lower thinking level |
| `finish_reason == "content_filter"` | a filter or a policy stopped the reply — a safety filter, an unsupported language, an escalation rule, or an account limited for Prohibited Use Policy violations; `refusal_message` holds Google's `finishMessage`, else the stop's name | another request; for `PUP_LIMITED_DISABLED`, the account's standing, since another request cannot help ([Finish reasons and safety](@ref gemini_finish_reasons)) |
| `LLMResultError` from `text` or `reasoning_text` | the result is a failure, which holds no reply | check `issuccess(result)` first |

Recorded on 25 October 2026 on `gemini-3.8-flash` with UniLM's development branch after 0.26.0 and the Gemini
API at `v1beta`.

See also: [`GEMINIServiceEndpoint`](@ref), [`GEMINIOpenAIServiceEndpoint`](@ref), [`Chat`](@ref),
[`chatrequest!`](@ref), [`Respond`](@ref) and [`respond`](@ref) in the API reference.
