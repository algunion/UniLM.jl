# [Thinking, tools & output](@id anthropic_requests_page)

```@setup anthropic
using UniLM
```

`ANTHROPICServiceEndpoint` turns each request field of a `Chat` into its Messages API counterpart. A
field with none is refused before the request is sent, except `n`, which a `Chat` takes only as 1: the
one reply every request gets.

| `Chat` field | Messages API | Rules |
| :--- | :--- | :--- |
| system messages | the top-level `system` prompt, or a `system` turn | [Conversations](@ref anthropic_conversations) |
| user messages, their attachments | `user` turns; `image` and `document` blocks before the text | [Attachments](@ref anthropic_media) |
| assistant messages, their `tool_calls` | `assistant` turns: the blocks the reply arrived in, else its text and `tool_use` blocks | [Conversations](@ref anthropic_conversations), [Tools](@ref anthropic_tools) |
| `tool` messages | `tool_result` blocks, consecutive ones in one `user` turn | [Tools](@ref anthropic_tools) |
| `tools` | `tools`, each with an `input_schema` | [Tools](@ref anthropic_tools) |
| `tool_choice`, `parallel_tool_calls` | `tool_choice`, with `disable_parallel_tool_use` | [Tools](@ref anthropic_tools) |
| `response_format` | `output_config.format` | [Structured output](@ref anthropic_structured_output) |
| `reasoning_effort` | `output_config.effort` and `thinking` | [Reasoning](@ref anthropic_reasoning) |
| `model` | `model` | the [family table](@ref anthropic_reasoning) |
| `max_completion_tokens`, else `max_tokens` | `max_tokens` | [Setup](@ref anthropic_setup) |
| `safety_identifier`, `user`, `metadata` | `metadata.user_id` | [Request options](@ref anthropic_request_options) |
| `service_tier` | `service_tier`, or `speed` for fast mode | [Request options](@ref anthropic_request_options) |
| `stop`, `stream` | `stop_sequences`, `stream` | [Request options](@ref anthropic_request_options) |
| `n` | not sent: one reply per request | [Request options](@ref anthropic_request_options) |
| `prompt_cache_options` | `cache_control` | [Prompt caching](@ref anthropic_caching) |
| `temperature`, `top_p`, `moderation`, `seed` and the other fields with no counterpart | refused | [Limits and refusals](@ref anthropic_limits) |

## [Tools](@id anthropic_tools)

Each `Tool` in `tools` goes out with its `name`, its `description` when it has one, its `parameters` as
`input_schema` — `{"type": "object", "properties": {}}` for a function without parameters — and its
`strict` flag when its `FunctionSignature` sets one: `strict=true` makes the API validate each call's
name and input against the tool's schema. The Messages API takes tool names of 1 to 128 letters, digits,
`_` and `-`. UniLM sends a name as given, and [`mcp_tools`](@ref) advertises each tool of an MCP server
under such a name: its own, or that name with each character outside that set replaced by `_`, cut to
128 characters ([MCP](@ref mcp_guide)).

`tool_choice` becomes the API's own: `"auto"` sends `{"type": "auto"}`, `"none"` sends
`{"type": "none"}`, `"required"` sends `{"type": "any"}`, and `UniLM.GPTToolChoice(func=:get_weather)`
sends `{"type": "tool", "name": "get_weather"}`. A forced choice, `"required"` or a named tool, is
refused on the models whose Forced tool choice cell in the [family table](@ref anthropic_reasoning) is
"—", and any other string on every model ([Limits and refusals](@ref anthropic_limits)).

A `Chat` with `tools` has `parallel_tool_calls = false` unless it sets another value, and sends it as
`disable_parallel_tool_use: true` on its `tool_choice`, which is `{"type": "auto"}` when it sets none:
the model then makes at most one tool call per turn. Set `parallel_tool_calls = true` to let it make
several in one turn; a `tool_choice` of `"none"` carries no such flag.

An assistant turn that called tools goes back in the blocks it arrived in, as
[Conversations](@ref anthropic_conversations) describes; a turn that holds none of Anthropic's blocks,
written by hand or answered by another provider, goes as its text and one `tool_use` block per call.
The `tool` messages that answer it become `tool_result` blocks, consecutive ones in one user turn. A
`tool` message whose `tool_call_id` matches no tool call of an earlier assistant turn is refused
([Limits and refusals](@ref anthropic_limits)).

`tool_loop!` runs the calls and sends their results back until the model answers in text. Here the
model thinks before it calls the tool:

```@example anthropic
weather = CallableTool(
    Tool(func=FunctionSignature(name="get_weather", description="The current weather in a city",
        parameters=Dict("type" => "object", "properties" => Dict("city" => Dict("type" => "string")),
                        "required" => ["city"]))),
    (name, args) -> "18 °C and sunny in $(args["city"])")
chat = Chat(service=ANTHROPICServiceEndpoint, tools=[weather], reasoning_effort="high")
push!(chat, Message(Val(:system), "You are a concise assistant."))
push!(chat, Message(Val(:user),
    "What is the weather in the capital of the country that hosted the 2016 Summer Olympics?"))
loop = tool_loop!(chat; tools=[weather])
text(loop.response)
```

```@example anthropic
call = chat.messages[3]   # the turn that called the tool
[b["type"] => first(get(b, "signature", ""), 12) for b in call.provider_content.blocks]
```

The turn that called the tool holds a signed `thinking` block before its `tool_use` block, and
`tool_loop!` sent that turn back unchanged, the tool result after it, in the request that got the
answer: Anthropic requires a tool result to come with the thinking blocks of the turn that called the
tool, complete and unmodified
([thinking in tool workflows](https://platform.claude.com/docs/en/build-with-claude/thinking-tool-workflows)).
A streamed turn's tool calls are on [Streaming](@ref anthropic_streaming), and the loop every provider
shares is on [Tool Calling](@ref tools_guide).

## [Structured output](@id anthropic_structured_output)

A JSON Schema `response_format` becomes `output_config.format = {"type": "json_schema", "schema": …}`:
Claude's [structured outputs](https://platform.claude.com/docs/en/build-with-claude/structured-outputs),
which constrain the reply to the schema. Only the schema is sent; the format's `name`, `description`
and `strict` have no counterpart.

```@example anthropic
using JSON
capital = UniLM.json_schema("capital", "A capital city and its country",
    Dict("type" => "object",
         "properties" => Dict("city" => Dict("type" => "string"), "country" => Dict("type" => "string")),
         "required" => ["city", "country"], "additionalProperties" => false))
chat = Chat(service=ANTHROPICServiceEndpoint, response_format=capital)
push!(chat, Message(Val(:system), "You are a concise assistant."))
push!(chat, Message(Val(:user), "Give the capital of Norway."))
JSON.parse(text(chatrequest!(chat)))
```

The reply's text is the JSON document. `ResponseFormat()`, a `json_object` format, is refused
([Limits and refusals](@ref anthropic_limits)). Strict tool calls are a flag of each tool
([Tools](@ref anthropic_tools)), and the formats every provider shares are on
[Structured Output](@ref structured_guide).

## [Reasoning](@id anthropic_reasoning)

`reasoning_effort` sets the effort of a request, `output_config.effort`. With adaptive thinking the
model decides on each request whether to think and how much, and the effort steers that decision
([thinking](https://platform.claude.com/docs/en/build-with-claude/thinking)). What else
`reasoning_effort` sends depends on the model's Thinking cell in the family table below:

| Thinking cell | `"low"` … `"max"` sends | `"none"` sends |
| :--- | :--- | :--- |
| adaptive; cannot be disabled | the effort and `thinking = {"type": "adaptive", "display": "summarized"}` | refused |
| adaptive by default; `disabled` turns it off | the effort and adaptive thinking, summarized | `thinking = {"type": "disabled"}` |
| adaptive by default, cannot be disabled; `between_tools` turns up-front thinking off | the effort and adaptive thinking, summarized | `thinking = {"type": "between_tools"}` |
| adaptive on request | the effort and adaptive thinking, summarized, which turns thinking on | nothing: thinking stays off |
| extended thinking (`budget_tokens`) only | the effort alone: UniLM sends no thinking budget | nothing |
| an id no row names | the effort and adaptive thinking, summarized, unchecked | `thinking = {"type": "disabled"}` |

UniLM refuses every `reasoning_effort` on a model whose Effort levels cell is "—", `"none"`
included, and `"minimal"` on every model ([Limits and refusals](@ref anthropic_limits)). A request
without `reasoning_effort` sends neither field: a model whose thinking is on by default keeps it, at the
API's default effort, and on most current models the API leaves the text out of the thinking blocks
unless asked for a summary, so [`reasoning_text`](@ref) gives `nothing`.

```@eval
Main.ProviderFacts.claude_family_table()
```

A model id takes the row of the same id, or of that id without a `-YYYYMMDD` snapshot suffix, never
the row of an id it merely starts with; an id no row names, such as a model newer than this release,
is refused only by the rules that hold for every model. The columns:

- **Thinking**: the model's thinking mode, which decides what `reasoning_effort` sends (the table above).
- **Effort levels**: the `reasoning_effort` levels from `"low"` to `"max"` the model takes; "—": it takes
  no `reasoning_effort` at all. What `"none"` does follows the Thinking cell (the table above).
- **Forced tool choice**: whether `tool_choice` may force a call ([Tools](@ref anthropic_tools)).
- **Prefill**: whether a request may end with an assistant turn
  ([Request options](@ref anthropic_request_options)).
- **Mid-conversation system messages** and **Thinking prefix check**: where a system message pushed
  after the conversation starts goes ([Conversations](@ref anthropic_conversations)).
- **Fast mode**: whether `service_tier="fast"` is served, and the beta header it needs where it is in
  research preview ([Request options](@ref anthropic_request_options)).
- **`inference_geo`**: whether the model takes the endpoint's `inference_geo` ([Options](@ref anthropic_options)).

`reasoning_text` reads the summary of a turn's thinking:

```@example anthropic
chat = Chat(service=ANTHROPICServiceEndpoint, reasoning_effort="low")
push!(chat, Message(Val(:system), "You are a concise assistant."))
push!(chat, Message(Val(:user), "What is the smallest positive integer that leaves a remainder of 1 " *
    "when divided by 2, 3, 4, 5 or 6, and is divisible by 7? Answer with the number only."))
result = chatrequest!(chat)
text(result)
```

```@example anthropic
reasoning_text(result)
```

```@example anthropic
u = result.usage
(; u.completion_tokens, u.reasoning_tokens)
```

`reasoning_text` joins the text of a turn's `thinking` blocks with a blank line, and gives `nothing`
for a turn with none; a `redacted_thinking` block carries no text and adds none. Thinking is billed in
full as output, whatever summary comes back
([thinking cost](https://platform.claude.com/docs/en/build-with-claude/thinking-steering-and-cost#pricing)),
and [`estimated_cost`](@ref) prices it at the output rate ([Cost and usage](@ref anthropic_cost)); how
the usage counts it is on [First call](@ref anthropic_first_call), and how `max_tokens` bounds it on
[Setup](@ref anthropic_setup).

The blocks of a reply that is not a refusal — `thinking` blocks with their signatures,
`redacted_thinking` blocks, text and tool calls — are kept in its message's `provider_content` and go
back on the next request ([Conversations](@ref anthropic_conversations)). A streamed reply keeps them
only when every block arrived whole: with one cut short — its stop event lost, or a tool call's input
that is no JSON object, as when `max_tokens` cuts it — the message holds no `provider_content`, and the
turn goes back as the text and tool calls it holds, or is left out when it holds neither. Kept blocks go
back whatever Claude model the next request names, and a turn cut at `max_tokens` that holds only
thinking blocks goes back too, as Anthropic's guide to continuing such a turn asks. Another provider's
encoder ignores them and sends the turn's text and tool calls alone.

## [Attachments](@id anthropic_media)

A user message's attachments go out as base64 blocks before its text, in their order — the order
Anthropic's [vision](https://platform.claude.com/docs/en/build-with-claude/vision) and
[PDF](https://platform.claude.com/docs/en/build-with-claude/pdf-support) guides recommend: an
[`ImageAttachment`](@ref) (PNG, JPEG, GIF or WebP) as an `image` block, and a
[`DocumentAttachment`](@ref) (PDF) as a `document` block, without its `filename`. A message with
empty text sends the blocks alone.

![A red disc on a white background](../../assets/red_circle.png)

```@example anthropic
chat = Chat(service=ANTHROPICServiceEndpoint)
push!(chat, Message(Val(:system), "You are a concise assistant."))
push!(chat, Message(Val(:user), "What colour is the shape? One word.",
                    ImageAttachment("../../assets/red_circle.png")))
text(chatrequest!(chat))
```

The [one-page PDF](https://github.com/algunion/UniLM.jl/raw/main/docs/src/assets/bridge_report.pdf)
holds one sentence:

```@example anthropic
chat = Chat(service=ANTHROPICServiceEndpoint)
push!(chat, Message(Val(:system), "You are a concise assistant."))
push!(chat, Message(Val(:user), "What does the report conclude? Answer in one sentence.",
                    DocumentAttachment("../../assets/bridge_report.pdf")))
text(chatrequest!(chat))
```

An `AudioAttachment` and an image's `detail` level are refused before the request
([Limits and refusals](@ref anthropic_limits)). UniLM checks no size or page count of its own: the API
answers for media past its limits. Which models read images and PDFs is in each model's
`capabilities`, listed by [`list_models`](@ref) ([Models](@ref anthropic_models)); the attachment types
are on [Attachments](@ref attachments_guide).

## [Request options](@id anthropic_request_options)

`safety_identifier`, `user` and `metadata["user_id"]` each name the end user, and the Messages API has
one field for that, `metadata.user_id`: set one of them, or set them to the same id. `metadata` holds
no other key.

`service_tier` takes the Messages API's `"auto"` and `"standard_only"`, sent as they are, and
`"fast"`: [fast mode](https://platform.claude.com/docs/en/build-with-claude/fast-mode), sent as
`speed: "fast"` without a `service_tier`. Fast mode is served on the models whose Fast mode cell in the
[family table](@ref anthropic_reasoning) is "yes", and on those whose cell names a beta header only
through an endpoint whose `betas` hold it; UniLM refuses it on the other rows, and sends it unchecked
for an id no row names. How a fast call is reported and priced is on
[First call](@ref anthropic_first_call) and [Cost and usage](@ref anthropic_cost). Where inference runs
is the endpoint's `inference_geo` ([Options](@ref anthropic_options)).

`stop` becomes `stop_sequences`, a single string a list of one, and `stream=true` becomes
`stream: true` ([Streaming](@ref anthropic_streaming)). `n` is not sent: a Messages API request has one
reply, and a `Chat` takes no `n` other than 1.

A response prefill, a request that ends with an assistant turn for the model to continue, is a `Chat`
with `history=false` whose last message is an assistant message: with `history=true`, `chatrequest!`
throws `InvalidConversationError` for such a conversation, since the reply could not be appended.
UniLM sends a prefill on the models whose Prefill cell is "yes" and for an id no row names, and refuses
it on the others ([Limits and refusals](@ref anthropic_limits)).

[`UniLM.encode_request`](@ref) returns the body `chatrequest!` sends, or throws the refusal the table
in [Limits and refusals](@ref anthropic_limits) lists, with no key and no request:

```@example anthropic
chat = Chat(service=ANTHROPICServiceEndpoint, service_tier="fast", stop="END", safety_identifier="user-42")
push!(chat, Message(Val(:system), "You are a concise assistant."))
push!(chat, Message(Val(:user), "Count to five."))
print(UniLM.encode_request(chat.service, chat))
```

`JSON.json(chat)` is not this body: it writes the `Chat` in the Chat Completions shape of the
OpenAI-compatible wire.

## [Limits and refusals](@id anthropic_limits)

A request's output stops at its `max_tokens` ([Setup](@ref anthropic_setup)), and a model's largest
`max_tokens` and its input limit are in its [`list_models`](@ref) entry ([Models](@ref anthropic_models)).
UniLM checks a request against neither, and leaves the size of media to the API
([Attachments](@ref anthropic_media)). Rate limits and overload are on
[When something fails](@ref anthropic_failures).

A field the Messages API has no counterpart for is refused, not dropped, since dropping it would send a
request other than the one asked for; so is each request in the table below that the API would answer
with an error, or would run other than asked. `chatrequest!` and `tool_loop!` throw these
`ArgumentError`s before anything is sent. A rule that names a column of the
[family table](@ref anthropic_reasoning) holds for the ids of its rows, and an id no row names meets
only the other rules. Building an endpoint refuses an option value that breaks its rule
([Options](@ref anthropic_options)).

| Refused | Why: the message's words | What to do |
| :--- | :--- | :--- |
| `temperature` or `top_p`, at any value | "answers HTTP 400 for it on every model, even at its default value": API version `2026-10-21` removed the sampling parameters | remove it |
| `moderation`, or a field with no counterpart: `presence_penalty`, `frequency_penalty`, `logit_bias`, `seed`, `stream_options`, `verbosity`, `store`, `logprobs`, `top_logprobs`, `prediction`, `modalities`, `audio`, `web_search_options`, `prompt_cache_key` | "does not support": the Messages API has no such field | remove it |
| a `service_tier` other than `"auto"`, `"standard_only"` and `"fast"` | "service_tier must be" one of the three | one of them |
| `service_tier="fast"` on a model whose Fast mode cell is "—", or through an endpoint whose `betas` lack the beta header the cell names | "has no fast mode", "serves fast mode": the model has no fast mode, or serves it in research preview only with the beta header | a model with fast mode, or an endpoint whose `betas` hold the header |
| `metadata` with a key other than `user_id`, or a `user_id` that is not a string | "accepts only user_id", "must be a string": the Messages API's `metadata` holds the end user's id alone | the id as a string in `metadata["user_id"]`, or in `safety_identifier` |
| `safety_identifier`, `user` and `metadata["user_id"]` set to different ids | "set one, or set them equal": each is sent as `metadata.user_id` | one of them, or the same id in each |
| `reasoning_effort="minimal"` | "no minimal reasoning_effort": no Claude model has the level | `"low"` |
| `reasoning_effort` on a model whose Effort levels cell is "—" | "does not support reasoning_effort": the model has no effort parameter | leave it unset |
| a level the model's Effort levels cell does not list | "supports reasoning_effort", then the levels it takes | one of them |
| `reasoning_effort="none"` where the Thinking cell is "adaptive; cannot be disabled" | "cannot disable thinking": the model's thinking cannot be turned off | `"low"` |
| `tool_choice="required"` or a named tool on a model whose Forced tool choice cell is "—" | "rejects forced tool_choice": the API answers HTTP 400 for a forced choice there | `tool_choice="auto"` with strict tools, or a JSON Schema `response_format` |
| a `tool_choice` string other than `"auto"`, `"none"` and `"required"` | "Unknown Anthropic tool_choice" | one of them, or `UniLM.GPTToolChoice(func=:name)` |
| a `tool` message whose `tool_call_id` matches no tool call of an earlier assistant turn | "references unknown tool_use id": a `tool_result` answers a `tool_use` the conversation holds | the `id` of the tool call it answers |
| `ResponseFormat()`, a `json_object` format | "no schema-less JSON mode": the Messages API has no JSON mode without a schema | a JSON Schema `response_format` |
| a `response_format` whose type is not `json_schema`, other than a `text` one without a schema, or a JSON Schema format without a schema object | "supports response_format json_schema", "needs a schema object" | a JSON Schema format holding its schema |
| an `AudioAttachment` | "not audio": the Messages API takes images and PDF documents | the clip's transcript in the message |
| an `ImageAttachment` with a `detail` level | "has no detail level": the image block has no such field | the attachment without `detail` |
| the endpoint's `inference_geo` on a model whose `inference_geo` cell is "—" | "takes no inference_geo": the API answers HTTP 400 for it | an endpoint without `inference_geo` ([Options](@ref anthropic_options)) |
| a request that ends with an assistant turn, on a model whose Prefill cell is "—" | "rejects messages that end with an assistant turn": the model takes no response prefill | end with a user turn, or ask for JSON with a JSON Schema `response_format` |
| a system message pushed after a reply, with no user turn after it | "takes a system message after an assistant turn only once a user turn follows it": the request would end with the assistant turn, a response prefill | a user message after it, or the reply's tool results ([Conversations](@ref anthropic_conversations)) |
| `prompt_cache_options` with `mode="explicit"`, `prewarm=true`, a `comparison_response_id`, or a `ttl` other than `"5m"` and `"1h"` | "Anthropic prompt caching takes prompt_cache_options": a `Chat` marks no cache breakpoints, takes no `max_tokens` of 0 to pre-warm a cache with, and maps no cache diagnostics | `mode="implicit"` or unset, and `ttl` `"5m"`, `"1h"` or unset ([Prompt caching](@ref anthropic_caching)) |

Recorded on 24 October 2026 on `claude-opus-5-5` with UniLM's development branch after 0.26.0 and the
Messages API at `anthropic-version` 2026-10-21.
