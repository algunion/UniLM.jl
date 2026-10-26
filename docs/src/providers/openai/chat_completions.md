# [Chat Completions](@id openai_chat_completions_page)

```@setup openai
using UniLM
```

On `OPENAIServiceEndpoint` a `Chat` goes to OpenAI's Chat Completions API as its own JSON,
`JSON.json(chat)`, which writes each field the `Chat` sets under the API's name: a `stream=true` request
without `stream_options` also asks for the stream's usage, a spoken reply goes back by its audio id, and
only `max_tokens` is renamed, to `max_completion_tokens`. Before anything is sent, the request is checked
against OpenAI's rules and the model's row in the [rule table](@ref openai_reasoning), and a request that
breaks one is refused ([Limits and refusals](@ref openai_limits)).

| `Chat` field | Chat Completions | Rules |
| :--- | :--- | :--- |
| messages, their attachments | `messages`; attachments as content parts | [Attachments](@ref openai_media) |
| `tools`, `tool_choice`, `parallel_tool_calls` | the same fields; `parallel_tool_calls: false` with `tools` unless set | [Tools on Chat Completions](@ref openai_chat_tools) |
| `response_format` | `response_format` | [Structured output](@ref openai_structured_output) |
| `reasoning_effort` | `reasoning_effort` | [Reasoning](@ref openai_reasoning) |
| `temperature`, `top_p`, `logprobs`, `top_logprobs` | the same fields, at the efforts the model's row allows | [Request options](@ref openai_chat_options) |
| `max_completion_tokens`, else `max_tokens` | `max_completion_tokens` | [Request options](@ref openai_chat_options) |
| `modalities`, `audio` | the same fields | [Audio replies](@ref chat_audio_output) |
| `stream`, `stream_options` | the same fields; `{"include_usage": true}` when unset | [Streaming](@ref openai_streaming) |
| `prompt_cache_options`, `prompt_cache_key`, `prompt_cache_retention` | the same fields | [Prompt caching](@ref openai_caching) |
| `service_tier`, `web_search_options`, `moderation`, `safety_identifier`, `metadata`, `store`, `seed`, `stop`, `verbosity`, `prediction`, `presence_penalty`, `frequency_penalty`, `logit_bias`, `n` | the same fields | [Request options](@ref openai_chat_options) |
| `user` | refused | [Limits and refusals](@ref openai_limits) |

## [Tools on Chat Completions](@id openai_chat_tools)

Each `Tool` in `tools` goes out as a `function` tool with its `name` and, where its `FunctionSignature`
sets them, its `description`, `parameters` and `strict` flag. A function's name "Must be a-z, A-Z, 0-9, or
contain underscores and dashes, with a maximum length of 64"
([Create chat completion](https://developers.openai.com/api/reference/resources/chat)); UniLM sends a name
as given, and [`mcp_tools`](@ref) and `mcp_tools_respond`, for this endpoint or with no `service`, advertise
an MCP tool under a name of at most 64 characters, a longer one cut and marked with a hash of its MCP name
([MCP](@ref mcp_guide)). `tool_choice` is sent as
given: `"auto"`, `"none"`, `"required"`, or `UniLM.GPTToolChoice(func=:get_weather)` for
`{"type": "function", "function": {"name": "get_weather"}}`. Strict mode makes the calls "reliably adhere
to the function schema", and OpenAI recommends "always enabling" it; its schema needs
`additionalProperties: false` on every object and every field `required`, and a request whose strict
schema misses them "will be rejected with details about the missing constraints"
([function calling](https://developers.openai.com/api/docs/guides/function-calling#strict-mode)).
A `Tool` whose `FunctionSignature` sets no `strict` goes without one, and "Chat Completions requests
remain non-strict by default"; the same tool given to a `Respond` meets the Responses default instead
([Tools](@ref openai_tools)).

A `Chat` with `tools` sends `parallel_tool_calls: false` unless it sets another value: the model may
otherwise "call multiple functions in a single turn", and `false` "ensures exactly zero or one tool is
called" ([parallel function calling](https://developers.openai.com/api/docs/guides/function-calling#parallel-function-calling)).
Set `parallel_tool_calls=true` to let it call several at once. A model of the
[rule table](@ref openai_reasoning) takes tools on Chat Completions only at the efforts its Chat
Completions tools column lists, so a `Chat` on the default model names that effort; with tools, OpenAI's
tool path is the Responses API ([Tools](@ref openai_tools)).

`tool_loop!` runs the calls and sends their results back until the model answers in text. Here the
model needs one call per city, and makes them one turn at a time:

```@example openai
weather = CallableTool(
    Tool(func=FunctionSignature(name="get_weather", description="The current weather in a city",
        parameters=Dict("type" => "object", "properties" => Dict("city" => Dict("type" => "string")),
                        "required" => ["city"], "additionalProperties" => false), strict=true)),
    (name, args) -> "18 °C and sunny in $(args["city"])")
chat = Chat(service=OPENAIServiceEndpoint, tools=[weather], reasoning_effort="none")
push!(chat, Message(Val(:system), "You are a concise assistant."))
push!(chat, Message(Val(:user), "What is the weather in Oslo and in Lima?"))
occursin("\"parallel_tool_calls\":false", UniLM.encode_request(chat.service, chat)) || error("not sent")
loop = tool_loop!(chat; tools=[weather])
text(loop.response)
```

```@example openai
[(m.role, isnothing(m.tool_calls) ? 0 : length(m.tool_calls)) for m in chat.messages], loop.turns_used
```

An assistant turn that called tools goes back with its `tool_calls`, and each `tool` message answers
one by its `tool_call_id`. A turn whose calls OpenAI finished with another reason than `"tool_calls"`
— `"length"`, say — runs none of them, and `tool_loop!` stops with `completed=false`. A streamed turn's
calls reach `on_tool_call` ([Streaming](@ref openai_streaming)), and the loop every provider shares is on
[Tool Calling](@ref tools_guide).

## [Structured output](@id openai_structured_output)

`response_format` is sent as given. A JSON Schema format —
`UniLM.json_schema(name, description, schema; strict)`, or `ResponseFormat` around a
`UniLM.JsonSchemaAPI` — asks for a reply that follows the schema: with `strict=true`, "the model will always follow the exact schema", "Only a subset of JSON
Schema" being supported then ([Create chat completion](https://developers.openai.com/api/reference/resources/chat),
`response_format`). A strict schema follows the rules of strict tools above:
"`additionalProperties: false` must always be set in objects", and "All fields must be `required`"
([structured outputs](https://developers.openai.com/api/docs/guides/structured-outputs#supported-schemas)).
`ResponseFormat()` asks for any JSON object, "An older method of generating JSON responses", and the
model "will not generate JSON without a system or user message instructing it to do so".

```@example openai
using JSON
capital = UniLM.json_schema("capital", "A capital city and its country",
    Dict("type" => "object",
         "properties" => Dict("city" => Dict("type" => "string"), "country" => Dict("type" => "string")),
         "required" => ["city", "country"], "additionalProperties" => false); strict=true)
chat = Chat(service=OPENAIServiceEndpoint, response_format=capital)
push!(chat, Message(Val(:system), "You are a concise assistant."))
push!(chat, Message(Val(:user), "Give the capital of Norway."))
JSON.parse(text(chatrequest!(chat)))
```

The reply's text is the JSON document. A `Respond` asks for the same with `text`
([Structured output on Responses](@ref openai_responses_structured_output)), and the formats every
provider shares are on [Structured Output](@ref structured_guide).

## [Attachments](@id openai_media)

A user message's attachments go out as content parts after its text, in their order: an
[`ImageAttachment`](@ref) (PNG, JPEG, GIF or WebP) as an `image_url` part holding a base64 data URL, with
its `detail` when it sets one; an [`AudioAttachment`](@ref) (WAV or MP3) as an `input_audio` part; and a
[`DocumentAttachment`](@ref) (PDF) as a `file` part, with its file name and a data URL. A message with
empty text sends the parts alone. `detail` is `"low"`, `"high"`, `"original"` or `"auto"`, the default:
"Supported values depend on the model"
([images and vision](https://developers.openai.com/api/docs/guides/images-vision#choose-an-image-detail-level)).
Chat Completions "accepts only PDF files as `file` content parts", which take no `detail`
([file inputs](https://developers.openai.com/api/docs/guides/file-inputs)).

![A red disc on a white background](../../assets/red_circle.png)

```@example openai
chat = Chat(service=OPENAIServiceEndpoint)
push!(chat, Message(Val(:system), "You are a concise assistant."))
push!(chat, Message(Val(:user), "What colour is the shape? One word.",
                    ImageAttachment("../../assets/red_circle.png")))
text(chatrequest!(chat))
```

The [one-page PDF](../../assets/bridge_report.pdf) holds one sentence:

```@example openai
chat = Chat(service=OPENAIServiceEndpoint)
push!(chat, Message(Val(:system), "You are a concise assistant."))
push!(chat, Message(Val(:user), "What does the report conclude? Answer in one sentence.",
                    DocumentAttachment("../../assets/bridge_report.pdf")))
text(chatrequest!(chat))
```

UniLM sends every kind of attachment to every model and checks no size or page count: the API answers
for a model that does not read the kind, and for media past its limits. Which input a model reads is on
its OpenAI model page; an audio clip needs an audio model such as `gpt-audio-1.5`
([Audio in Chat Completions](https://developers.openai.com/api/docs/guides/audio-chat-completions)). A
`Respond` takes images and PDFs as parts of its input, and no audio
([Input](@ref openai_responses_input)); the attachment types are on [Attachments](@ref attachments_guide).

## [Audio replies](@id chat_audio_output)

A model that speaks answers in audio when the `Chat` asks for it with `modalities=["text", "audio"]` and
an `audio` voice and format; OpenAI's guide uses `gpt-audio-1.5`
([Audio in Chat Completions](https://developers.openai.com/api/docs/guides/audio-chat-completions)).
The reply's `audio_output` is an [`AudioOutput`](@ref): the audio's bytes in `data`, their `format` as
their file signature shows it, OpenAI's `id` for the audio, `expires_at` and the `transcript`, which is
also the reply's text.

```@example openai
chat = Chat(service=OPENAIServiceEndpoint, model="gpt-audio-1.5", modalities=["text", "audio"],
            audio=Dict("voice" => "alloy", "format" => "mp3"))
push!(chat, Message(Val(:system), "Answer in one short sentence."))
push!(chat, Message(Val(:user), "Say hello to a new colleague."))
spoken = chatrequest!(chat).message.audio_output
(; spoken.format, bytes = length(spoken.data), spoken.transcript)
```

The next request sends the audio turn back as its text and `"audio": {"id": …}`, "Data about a previous
audio response from the model", until a minute before the turn's `expires_at`, the Unix time "for when
this audio response will no longer be accessible on the server for use in multi-turn conversations"
([Create chat completion](https://developers.openai.com/api/reference/resources/chat), the assistant
message's `audio` and the reply's `audio`). From then on it sends the turn as its text alone, so that a
request built just before the expiry does not arrive after it; a turn whose `expires_at` is `nothing`
goes by its id. The request's bytes therefore depend on the clock, and so does a recording of it
([Testing and recordings](@ref testing_guide)): recorded while the audio was live, it sends the id and
replays only until a minute before the expiry. This page replays recorded replies, and the request below
was recorded after the first one's `expires_at`, so it sends that reply as its transcript and replays at
any later time:

```@example openai
push!(chat, Message(Val(:user), "Now say it in French."))
result = chatrequest!(chat)
u = result.usage
(; result.message.audio_output.transcript, u.prompt_audio_tokens, u.completion_audio_tokens,
   cost = estimated_cost(result))
```

The usage counts the audio apart, `prompt_audio_tokens` of the input and `completion_audio_tokens` of the
output: here the input holds no audio, its first reply having gone back as text, while a reply sent back
by its id counts as input audio. [`estimated_cost`](@ref) bills them at the model's audio rates
([Cost and usage](@ref openai_cost)). A streamed reply assembles the same `audio_output` from its chunks,
the transcript reaching the callback as text and the expiry coming in the last update; a stream cut
between the finish and that update keeps the turn with `expires_at == nothing`. Speech and transcription
of their own are on [Audio](@ref openai_audio).

## [Request options](@id openai_chat_options)

- **`max_completion_tokens`, `max_tokens`.** OpenAI deprecates `max_tokens` "in favor of
  `max_completion_tokens`", which bounds "visible output tokens and reasoning tokens"
  ([Create chat completion](https://developers.openai.com/api/reference/resources/chat)): a `Chat` that
  sets `max_tokens` sends it as `max_completion_tokens`, and one that sets both is refused
  ([Limits and refusals](@ref openai_limits)).
- **`service_tier`.** `"auto"`, `"default"`, `"flex"`, `"scale"`, `"priority"` or `"fast"`; Fast mode is
  `"fast"` or its former name `"priority"`, and OpenAI reports `"priority"` for it "For GPT-5.6 and
  earlier models" ([Fast mode](https://developers.openai.com/api/docs/guides/fast-mode)). Ultrafast is a
  Responses tier. The tier OpenAI reports is the result's `usage.service_tier`, which
  [Cost and usage](@ref openai_cost) prices.
- **`logprobs`, `top_logprobs`.** Sent unless the model's row refuses them at the request's effort
  ([Limits and refusals](@ref openai_limits)), `logprobs=true` asking for "the log probabilities of each
  output token"; a `Chat` result keeps no log probabilities, since an `LLMSuccess` holds the reply's
  message and usage. A `Respond` with
  `include=["message.output_text.logprobs"]` keeps them in its output items,
  `result.response.output`.
- **`web_search_options`.** Sent as given; Chat Completions "supports only specialized search models
  for web search", which "always search before responding"
  ([web search](https://developers.openai.com/api/docs/guides/tools-web-search#chat-completions-api)).
  `gpt-5-search-api`, the search model OpenAI's guide names for Chat Completions, is priced at its row in
  the price table, and a search model without a row is estimated at `0.0`, with a warning; neither
  estimate includes a fee OpenAI lists for web search apart from the token rates. The Responses API's
  `web_search()` tool works on the other models ([Hosted tools](@ref openai_hosted_tools)).
- **`moderation`.** A [`ModerationConfig`](@ref) runs a moderation model over the request's input and the
  generated output. OpenAI returns the results in the completion's `moderation`, which an `LLMSuccess`
  does not keep: read them from a `Respond`'s result ([Request options](@ref openai_responses_options)).
- **`safety_identifier`.** "A stable identifier used to help detect users of your application", at most
  128 characters; `user`, which OpenAI replaces with it and `prompt_cache_key`, is refused
  ([Limits and refusals](@ref openai_limits)).
- **`store`, `metadata`.** `store=true` keeps the completion "for use in our model distillation or evals
  products"; UniLM reads no stored completion back. `metadata` holds up to 16 key-value pairs.
- **`verbosity`, `prediction`, `seed`, `stop`, `presence_penalty`, `frequency_penalty`, `logit_bias`.**
  Sent as given, within the ranges the `Chat` constructor checks; `n` is 1 or unset, since a result holds
  one reply.

[`UniLM.encode_request`](@ref) returns the body `chatrequest!` sends, with no key and no request:

```@example openai
chat = Chat(service=OPENAIServiceEndpoint, max_tokens=200, service_tier="flex", stop="END",
            safety_identifier="user-42", verbosity="low")
push!(chat, Message(Val(:system), "You are a concise assistant."))
push!(chat, Message(Val(:user), "Count to five."))
print(UniLM.encode_request(chat.service, chat))
```

The concepts every provider shares are on [Conversations with Chat](@ref chat_guide).

Recorded on 25 October 2026 on `gpt-6-sol` and `gpt-audio-1.5` with UniLM's development branch after
0.26.0.
