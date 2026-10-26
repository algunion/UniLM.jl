# [DeepSeek](@id deepseek_start)

```@setup deepseek
using UniLM
```

DeepSeek makes the DeepSeek models and serves them through an OpenAI-compatible API.
`DeepSeekEndpoint()` speaks its Chat Completions API and two of its beta features, FIM completion and
chat prefix completion: a `Chat` reaches DeepSeek through `chatrequest!`, `tool_loop!` and
`prefix_complete`, and a `FIMCompletion` through `fim_complete`. The agentic verbs — `respond` and the
`tool_loop` that takes an input or a `Respond` — refuse this endpoint before sending anything, and no
verb of a capability the table below lists as not available sends a request through it. Of DeepSeek's
API, UniLM does not use the Anthropic-format API or the Search and Fetch APIs either.

```@eval
Main.ProviderFacts.at_a_glance(:deepseek)
```

| I want to… | Use | You get |
| :--- | :--- | :--- |
| [make a first call](@ref deepseek_first_call) | `chatrequest!(chat)` | an `LLMSuccess`: the reply, its thinking and its usage, priced by `estimated_cost` |
| [hold a conversation](@ref deepseek_conversations) | `push!` a message, then `chatrequest!` again | each reply appended to the `Chat` and sent back with the next turn |
| [stream a reply](@ref deepseek_streaming) | `stream=true` and a `callback` | the text as it arrives, then the whole message and its usage |
| [let the model call my functions](@ref deepseek_tools) | `tools` and `tool_loop!` | the calls run and answered until the model replies, its thinking sent back as DeepSeek requires |
| [get JSON](@ref deepseek_structured_output) | `response_format=UniLM.json_object()` | a reply that is one JSON object |
| [control thinking](@ref deepseek_reasoning) | `reasoning_effort` | thinking off, or at an effort, and its text in `reasoning_text` |
| [fill in the middle or continue a prefix](@ref deepseek_completions) | `fim_complete`, `prefix_complete` | the text between a prefix and a suffix; the rest of an assistant turn |
| [reuse a cached prompt](@ref deepseek_caching) | nothing: DeepSeek caches on its own | cache hits in `usage.cached_tokens`, billed at the cached rate |
| [know what a call cost](@ref deepseek_cost) | `estimated_cost`, `cumulative_cost` | USD estimates at DeepSeek's peak rates |
| [handle a failure](@ref deepseek_failures) | the result's type | the status or the exception, and what to do |

## [Setup](@id deepseek_setup)

Set your API key in the environment Julia starts from:

```bash
export DEEPSEEK_API_KEY="sk-..."
```

`DeepSeekEndpoint()` reads the variable each time a request is built, so the endpoint can be made
before the key is set; a call without it returns a call-error result whose `cause` is the `KeyError`
naming it. `DeepSeekEndpoint(api_key="sk-…")` sends the key it is given instead. Neither prints its
key: the first shows as `DeepSeekEndpoint(#= DEEPSEEK_API_KEY =#)`, the second with the key cut to
its first four characters and `…[redacted]`. A `Chat` goes to
`https://api.deepseek.com/v1/chat/completions` with the key as `Authorization: Bearer`, and with no
`model` it takes the default model named at the top of this page, as FIM does.

## [First call](@id deepseek_first_call)

A `Chat` starts with its system message; `chatrequest!` sends it and returns a typed result, and
`text` reads the reply — or throws, on a failure:

```@example deepseek
chat = Chat(service=DeepSeekEndpoint())
push!(chat, Message(Val(:system), "You are a concise assistant."))
push!(chat, Message(Val(:user), "In one sentence: what is a prime number?"))
result = chatrequest!(chat)
text(result)
```

```@example deepseek
result.model, result.message.finish_reason
```

```@example deepseek
u = result.usage
(; u.prompt_tokens, u.completion_tokens, u.reasoning_tokens, u.cached_tokens, cost = estimated_cost(result))
```

`result.model` is the model DeepSeek reports serving, and the model [`estimated_cost`](@ref) prices
the call at. The reply's `finish_reason` is DeepSeek's — `stop`, `length`, `content_filter`,
`tool_calls`, `insufficient_system_resource` or `aborted`
([Chat Completions](https://api-docs.deepseek.com/api/create-chat-completion)) — with `tool_calls`
standing in for `stop` on a turn that calls tools. The model thought before it answered: DeepSeek's
models think by default ([Reasoning](@ref deepseek_reasoning)). `prompt_tokens` counts the input, of
which `cached_tokens` came from DeepSeek's cache ([Prompt caching](@ref deepseek_caching)), and
`completion_tokens` the output, of which `reasoning_tokens` is the thinking, which DeepSeek reports in
the completion's breakdown. The result types are on [Results](@ref results_guide).

## [Conversations](@id deepseek_conversations)

A `Chat` keeps the conversation: each reply is appended to it, and the next `chatrequest!` sends the
whole conversation again.

```@example deepseek
push!(chat, Message(Val(:user), "And a composite number?"))
again = chatrequest!(chat)
text(again)
```

```@example deepseek
talk = chat
[m.role for m in talk.messages], cumulative_cost(talk)
```

A reply's thinking stays on its message, where [`reasoning_text`](@ref) reads it, and goes back to
DeepSeek only on a request that carries `tools` ([Tools](@ref deepseek_tools)): without tools,
DeepSeek says the thinking "does not need to be passed back; even if passed to the API, it will be
ignored" ([Thinking Mode](https://api-docs.deepseek.com/guides/thinking_mode)). A system message
pushed after the conversation started is sent in place, which DeepSeek takes: its Chat Completions API
"does support inserting `system` messages mid-conversation"
([Tool Calls](https://api-docs.deepseek.com/guides/tool_calls)). What `push!` refuses is the same on
every provider ([Chat Completions](@ref chat_guide)).

## [Streaming](@id deepseek_streaming)

With `stream=true`, `chatrequest!` returns a `Task` at once, and `fetch` gives the same typed result
as an unstreamed call; the `callback` receives the text as it arrives, then the assembled `Message`:

```@example deepseek
chat = Chat(service=DeepSeekEndpoint(), stream=true)
push!(chat, Message(Val(:system), "You are a concise assistant."))
push!(chat, Message(Val(:user), "List the planets of the solar system in order, one per line."))
deltas = String[]
task = chatrequest!(chat; callback=(chunk, close) -> chunk isa String && push!(deltas, chunk))
streamed = fetch(task)
text(streamed)
```

```@example deepseek
length(deltas), streamed.usage.completion_tokens, reasoning_text(streamed) !== nothing
```

- **Usage.** A `Chat` that sets no `stream_options` sends `"stream_options": {"include_usage": true}`
  with its stream, so a streamed turn has `usage`, cache hits included, and adds its cost to
  `cumulative_cost` as an unstreamed one does; a `Chat` that sets `stream_options` sends its own.
  DeepSeek reports the usage on "the last chunk before the `data: [DONE]` message"
  ([Chat Completions](https://api-docs.deepseek.com/api/create-chat-completion)).
- **Thinking.** No thinking reaches the callback, only the answer's text. The thinking arrives in
  `delta.reasoning_content` and is assembled into the final message once its finish reason arrives
  or the stream ends; a stream cut short before then keeps none of it.
- **End.** The turn is final at `data: [DONE]`, or, without one, when the HTTP body ends after the
  finish reason; `on_tool_call(tc)` runs once per call whose arguments decode, as the next call
  begins or at the end of the stream; setting `close[] = true` stops the stream
  ([Stopping a Stream Early](@ref streaming_stop)).
- **Waiting.** While a request waits for inference, DeepSeek sends `: keep-alive` comments on a
  stream, which UniLM skips, and empty lines before an unstreamed reply; it closes the connection of a
  request that "has not started inference after 10 minutes"
  ([Rate Limit & Isolation](https://api-docs.deepseek.com/quick_start/rate_limit)).

The concepts every provider shares are on [Streaming](@ref streaming_guide).

## [Tools](@id deepseek_tools)

Each `Tool` in `tools` goes out as a function tool, and `tool_choice`, `parallel_tool_calls` and a
tool's `strict` go as they are set; a `Chat` with tools sends `parallel_tool_calls: false` unless it
sets another value. A function's name "Must be a-z, A-Z, 0-9, or contain underscores and dashes, with a
maximum length of 128" ([Chat Completions](https://api-docs.deepseek.com/api/create-chat-completion));
UniLM sends a name as given, and [`mcp_tools`](@ref)`(session; service)`, with `service` this endpoint,
advertises an MCP server's tools under such names, one over 128 characters cut and marked with a hash of
its MCP name ([MCP](@ref mcp_guide)). `tool_loop!` runs the calls and sends their results back until the
model answers in text:

```@example deepseek
weather = UniLM.CallableTool(
    Tool(func=FunctionSignature(name="get_weather", description="Current weather in a city",
        parameters=Dict("type" => "object", "required" => ["city"],
                        "properties" => Dict("city" => Dict("type" => "string"))))),
    (name, args) -> "18°C and sunny in $(args["city"])")
chat = Chat(service=DeepSeekEndpoint(), tools=[weather])
push!(chat, Message(Val(:system), "You are concise."))
push!(chat, Message(Val(:user), "What is the weather in Lyon? Use the tool, then answer in one sentence."))
loop = tool_loop!(chat; tools=[weather])
calls = [(o.tool_name, o.arguments) for o in loop.tool_calls]
loop.completed, calls, text(loop.response)
```

A request that carries tools must send back the thinking of every earlier assistant turn: "the
`reasoning_content` must be fully passed back to the API in all subsequent requests — even for turns
where the model did not perform a tool call. If your code does not correctly pass back
`reasoning_content`, the API will return a 400 error"
([Thinking Mode](https://api-docs.deepseek.com/guides/thinking_mode)). UniLM does: on such a request,
each assistant turn that holds DeepSeek's thinking carries it as `reasoning_content`, the replies
`chatrequest!`, `tool_loop!` and `prefix_complete` appended among them. The loop's replies above hold
none: the model answered those turns without thinking. The conversation of
[Conversations](@ref deepseek_conversations) holds thinking on both replies, and sent with tools, both
carry it, the third and the fifth message:

```@example deepseek
using JSON
with_tools = Chat(service=DeepSeekEndpoint(), tools=[weather], messages=copy(talk.messages))
sent = JSON.parse(UniLM.encode_request(with_tools.service, with_tools))
[(i, m["role"]) for (i, m) in enumerate(sent["messages"]) if haskey(m, "reasoning_content")]
```

An assistant message you push yourself holds no thinking, so it goes back without any, and DeepSeek's
Chat Completions API "does not support inserting tool calls mid-conversation"
([Tool Calls](https://api-docs.deepseek.com/guides/tool_calls)). DeepSeek
refuses two `tool_choice` values while the model thinks: "`required` and named tool choices are not
supported in thinking mode; the API returns a `400` error. Disable thinking mode first to use them"
([Chat Completions](https://api-docs.deepseek.com/api/create-chat-completion)), which
`reasoning_effort="none"` does ([Reasoning](@ref deepseek_reasoning)). A tool's `strict` is a beta
feature that needs `base_url="https://api.deepseek.com/beta"`
([Tool Calls](https://api-docs.deepseek.com/guides/tool_calls)), while a `Chat` goes to the non-beta
URL, so `strict` through this endpoint is untested. The loop every provider shares is on
[Tool Calling](@ref tools_guide).

## [Structured output](@id deepseek_structured_output)

`UniLM.json_object()` sends `{"type": "json_object"}`, DeepSeek's JSON Output, which "guarantees the
message the model generates is valid JSON"; DeepSeek adds that "you must also instruct the model to
produce JSON yourself via a system or user message"
([Chat Completions](https://api-docs.deepseek.com/api/create-chat-completion)):

```@example deepseek
chat = Chat(service=DeepSeekEndpoint(), response_format=UniLM.json_object())
push!(chat, Message(Val(:system), "Answer in JSON with the keys name and year."))
push!(chat, Message(Val(:user), "Who wrote the first published computer program, and in which year?"))
JSON.parse(text(chatrequest!(chat)))
```

DeepSeek's `response_format` takes two types, `text` and `json_object`. A JSON Schema format,
`UniLM.json_schema(…)`, goes out as `json_schema` all the same: UniLM does not refuse it, and
DeepSeek's answer is the check. The formats every provider shares are on
[Structured Output](@ref structured_guide).

## [Reasoning](@id deepseek_reasoning)

DeepSeek's models think before they answer, by default at the effort `high`
([Thinking Mode](https://api-docs.deepseek.com/guides/thinking_mode)), and return the thinking beside
the answer; UniLM keeps it on the message, where [`reasoning_text`](@ref) reads it:

```@example deepseek
thought = reasoning_text(result)
length(thought), result.usage.reasoning_tokens
```

A `Chat`'s `reasoning_effort` goes out as it is set. DeepSeek takes `"none"`, which turns thinking
off, and `"low"`, `"high"` and `"max"`, and maps three more: `"minimal"` to `low`, `"medium"` and
`"xhigh"` to `high` ([Chat Completions](https://api-docs.deepseek.com/api/create-chat-completion)).
With thinking off, the reply carries none and reports no reasoning tokens:

```@example deepseek
quick = Chat(service=DeepSeekEndpoint(), reasoning_effort="none")
push!(quick, Message(Val(:system), "You are a concise assistant."))
push!(quick, Message(Val(:user), "In one sentence: what is a prime number?"))
plain = chatrequest!(quick)
reasoning_text(plain), plain.usage.reasoning_tokens
```

DeepSeek's `thinking` field, its other switch, is not a `Chat` field: `reasoning_effort="none"` is the
one way to turn thinking off. While the model thinks, `temperature` has no effect ("setting these
parameters will not trigger an error but will also have no effect"), and the effect of `top_p`
depends on the model ([Thinking Mode](https://api-docs.deepseek.com/guides/thinking_mode)). The
thinking is output: it counts in `completion_tokens` and bills at the output rate, and a request that
sets no `max_tokens` may generate "8K in non-thinking mode, 64K in thinking mode (128K with
`reasoning_effort` set to `max`)"
([Chat Completions](https://api-docs.deepseek.com/api/create-chat-completion)). Set `max_tokens`,
DeepSeek's name for the cap: a `Chat`'s `max_completion_tokens` goes out under that name, which
DeepSeek's reference does not list.

## [Attachments](@id deepseek_media)

A user message's attachments go out as content parts: an image as an `image_url` part, with its
`detail` when one is set, a sound clip as an `input_audio` part, and a document as a `file` part
holding its name and a data URL. DeepSeek's reference takes `text`, `image_url` and `file` parts —
JPEG, PNG, GIF and WebP images, and a `file_data` that is "A base64-encoded data URL of the image" —
and lists no audio part, and its pricing page lists vision for `deepseek-flash` only
([Models & Pricing](https://api-docs.deepseek.com/quick_start/pricing/)). UniLM checks none of this
before sending: a part DeepSeek does not take is DeepSeek's error. The attachments every provider
shares are on [Attachments](@ref attachments_guide).

## [FIM and prefix completion](@id deepseek_completions)

Both are DeepSeek beta features, which UniLM sends to the beta base URL,
`https://api.deepseek.com/beta`. `fim_complete` fills in the middle between a prompt and an optional
suffix, on the default model unless the request names another:

```@example deepseek
fim = fim_complete("def fib(a):"; service=DeepSeekEndpoint(), suffix="    return fib(a-1) + fib(a-2)",
                   max_tokens=128)
fim_text(fim)
```

```@example deepseek
fim.response.model, fim.response.usage.completion_tokens, estimated_cost(fim)
```

DeepSeek states that "The max tokens of FIM completion is 4K"
([FIM Completion](https://api-docs.deepseek.com/guides/fim_completion)), and its pricing page lists FIM
in "Non-thinking mode only". A `FIMCompletion` sends `max_tokens` 128 unless it sets another value,
and has no streaming path ([Limits and refusals](@ref deepseek_limits)).

`prefix_complete` continues an assistant turn from a prefix you give: the conversation's last message
is the assistant's partial reply, which goes out with `"prefix": true`
([Chat Prefix Completion](https://api-docs.deepseek.com/guides/chat_prefix_completion)):

```@example deepseek
code = Chat(service=DeepSeekEndpoint(), stop=["```"])
push!(code, Message(Val(:system), "You are a coding assistant."))
push!(code, Message(Val(:user), "Write a quicksort function in Python."))
push!(code, Message(role=RoleAssistant, content="```python\n"))
continued = prefix_complete(code)
text(continued)
```

```@example deepseek
startswith(code.messages[end].content, "```python\n")
```

The result's message is the continuation DeepSeek returns; with `history` (the default), the
conversation keeps the whole assistant turn, the prefix followed by the continuation, and a failure or
a cancellation leaves the `Chat` untouched. The request goes through this endpoint's own encoder and
decoder, so the reply's thinking, when it has any, is kept, and goes back on a later request with
tools, as a `chatrequest!` reply's does. DeepSeek also takes a `reasoning_content` of your own on the prefix
message, the start of its thinking; UniLM sends none. A `Chat` with `stream=true` is sent as it is,
asking DeepSeek for a stream, which `prefix_complete` does not read: the call returns an
`LLMCallError`, so leave `stream` unset. The concepts both verbs share are on
[FIM & Prefix Completion](@ref completions_guide).

## [Limits and refusals](@id deepseek_limits)

DeepSeek's models take a context of "1M" tokens and generate at most "384K"
([Models & Pricing](https://api-docs.deepseek.com/quick_start/pricing/)), a request's `max_tokens`
"between 1 and 384K (393216)"
([Chat Completions](https://api-docs.deepseek.com/api/create-chat-completion)). UniLM checks none of
these limits, and no rule of DeepSeek's: every field a `Chat` sets goes out as it is — `user`,
`max_tokens`, a JSON Schema format, the prompt-cache fields — and DeepSeek's answer is the check
([When something fails](@ref deepseek_failures)). It refuses before anything is sent what the table
below lists, and the checks every `Chat` gets, such as a `temperature` from 0 to 2 or an `n` of 1;
the verbs of the capabilities the table at the top of this page lists as not available refuse the
endpoint too.

| Refused | Why: the message's words | What to do |
| :--- | :--- | :--- |
| a `reasoning_effort` outside the seven UniLM takes, such as DeepSeek's `"ultra"` | `reasoning_effort must be one of`: UniLM takes `"none"`, `"minimal"`, `"low"`, `"medium"`, `"high"`, `"xhigh"` and `"max"`, and DeepSeek maps `"ultra"` to `max` | `"max"` |
| `prefix_complete` on a `Chat` without messages | `Chat must not be empty for prefix completion`: there is no turn to continue | the conversation, then the assistant's prefix |
| `prefix_complete` on a `Chat` whose last message is not the assistant's | `Last message must be role=assistant for prefix completion`: DeepSeek continues the last message, which must be the assistant's | push the prefix as an assistant message |
| a `FIMCompletion` with `stream=true` | `does not support stream=true`: `fim_complete` has no streaming path | leave `stream` unset |

## [Prompt caching](@id deepseek_caching)

DeepSeek caches prompts without being asked: its context caching "is enabled by default for all
users", and a later request whose prompt fully matches a prefix DeepSeek has stored reads that part
from the cache, on a "best-effort" basis, with no guarantee of a hit
([Context Caching](https://api-docs.deepseek.com/guides/kv_cache)). DeepSeek stores a prefix at the
end of each request's input and of its output, the prefix it detects that several requests share, and
prefixes of long ones at fixed intervals, so a request that
continues an earlier one can find part of it stored; the second request of the
[tool loop](@ref deepseek_tools) read part of its prompt from the cache:

```@example deepseek
u = loop.response.usage
u.cached_tokens, u.prompt_tokens, estimated_cost(loop.response)
```

DeepSeek reports the cached part as `prompt_cache_hit_tokens`, which UniLM reads into
`cached_tokens` for `chatrequest!`, streamed or not, and for `prefix_complete`; `fim_complete` reads
`prompt_tokens_details.cached_tokens`, the same count ("Same as `prompt_cache_hit_tokens`",
[FIM Completion API](https://api-docs.deepseek.com/api/create-completion)). A cache hit bills at the
cached rate of the price table ([Cost and usage](@ref deepseek_cost)). `prompt_cache_options`,
`prompt_cache_key` and `prompt_cache_retention` go out as they are set, though DeepSeek's reference
lists none of them.

## [Cost and usage](@id deepseek_cost)

[`estimated_cost`](@ref) prices a result at its model's row of the table below: input that missed the
cache at the input rate, cache hits at the cached rate, and `completion_tokens`, thinking included,
at the output rate. [`cumulative_cost`](@ref) adds up the turns `chatrequest!` and `tool_loop!` sent
on a `Chat`; a `prefix_complete` turn is appended to the `Chat` and not added. The rows are DeepSeek's
peak rates, as checked on 25 October 2026. "Off-peak rates are half of the peak rates", and peak hours
are "01:00 - 04:00 and 06:00 - 10:00 UTC, Monday through Friday, excluding Chinese public holidays"
([Models & Pricing](https://api-docs.deepseek.com/quick_start/pricing/)), so an estimate of a call made
off-peak is twice what DeepSeek bills. The legacy names `deepseek-v4-flash` and
`deepseek-v4-flash-vision-exp` are "billed at the Flash price".

```@eval
Main.ProviderFacts.price_table(:deepseek)
```

How a call is priced on every provider is on [Cost Tracking](@ref cost_guide).

## [When something fails](@id deepseek_failures)

`chatrequest!`, `prefix_complete` and `fim_complete` return what DeepSeek or the network did as a
typed result, and do not throw it ([Results](@ref results_guide)). A failure holds DeepSeek's HTTP
`status` and, as `response`, the body that came with it; a call error holds the text of an exception
as `error` and the exception as `cause`. UniLM sends a request again after a transient status — 408,
429, 500, 502 to 504, 520 to 524 and 529 — or a failed connection, as often as the call's
[`RequestConfig`](@ref) allows, and then returns the last answer ([Timeouts & Retries](@ref timeouts_guide));
a streamed request is sent again only while none of its callbacks has run. The statuses below are
DeepSeek's ([Error Codes](https://api-docs.deepseek.com/quick_start/error_codes)):

| What you see | Why | What to do |
| :--- | :--- | :--- |
| `ArgumentError` from the call, nothing sent | the request breaks a rule UniLM checks before sending ([Limits and refusals](@ref deepseek_limits)) | what its row says |
| a call error — an `LLMCallError`, or a `FIMCallError` from `fim_complete` — whose `cause` is a `KeyError` | `DEEPSEEK_API_KEY` is not set, and the endpoint was given no key | set it ([Setup](@ref deepseek_setup)) |
| a failure with `status` 401 | "Authentication fails due to the wrong API key" | the key |
| a failure with `status` 402 | "You have run out of balance" | top up the account |
| a failure with `status` 400 or 422 | an "Invalid request body format" or "invalid parameters": a field or a value DeepSeek does not take, such as a forced tool choice while the model thinks ([Tools](@ref deepseek_tools)); not sent again | what DeepSeek's message says |
| a failure with `status` 429 | "You are sending requests too quickly, or have exceeded a concurrency or QPS limit"; sent again until the attempts or the deadline run out | fewer requests at once ([Models](@ref deepseek_models)) |
| a failure with `status` 500 or 503 | a fault, or "The server is overloaded due to high traffic"; sent again until the attempts or the deadline run out | the request again, later |
| a reply whose `finish_reason` is `insufficient_system_resource` | the request "is interrupted due to insufficient resource of the inference system" ([Chat Completions](https://api-docs.deepseek.com/api/create-chat-completion)) | the turn again |
| an `LLMFailure` with `status` 200 from a stream | the stream ended before a finish reason; `response` holds what arrived, and nothing is appended | the turn again |
| a call error with `status` `nothing` and a `UniLMTimeout` as `cause` | a bound of the call's `RequestConfig` ran out | a longer bound, or a stream for a long reply ([Timeouts & Retries](@ref timeouts_guide)) |
| a call error whose `cause` is a `UniLMCancelled` | a stop: the stream's `close[] = true` or a cancelled `CancelToken` | nothing, if the stop was yours |
| a call error whose `cause` is the exception your `callback` or `on_tool_call` threw | the exception ended the stream; it is not sent again, and nothing is appended; a thrown value that is no `Exception` leaves `cause` `nothing`, its text in `error`; an `InterruptException`, or a recorded call's `ReplayMissError` or `RecordingWriteError`, is thrown at `fetch` instead | the callback |
| `LLMResultError` from `text` or `fim_text` | the result is a failure, which holds no reply | check `issuccess(result)` first |

## [Models](@id deepseek_models)

DeepSeek serves two models, `deepseek-flash` and `deepseek-v4-pro`. `deepseek-flash` is "now served
by DeepSeek-V4.1-Flash-1017", and the legacy names `deepseek-v4-flash` and
`deepseek-v4-flash-vision-exp` "are still accepted", served by the same model
([Your First API Call](https://api-docs.deepseek.com/)). Both take a 1M-token context, tools, JSON
Output, chat prefix completion and FIM; only `deepseek-flash` takes images
([Models & Pricing](https://api-docs.deepseek.com/quick_start/pricing/)). An account may have at most
2500 requests to `deepseek-flash` and 500 to `deepseek-v4-pro` in progress at once, and more get HTTP
429 ([Rate Limit & Isolation](https://api-docs.deepseek.com/quick_start/rate_limit)). DeepSeek lists
its models at `GET https://api.deepseek.com/models`
([Lists Models](https://api-docs.deepseek.com/api/list-models)).

```julia
chat = Chat(service=DeepSeekEndpoint(), model="deepseek-v4-pro")
```

Recorded on 25 October 2026 on `deepseek-flash`, served by DeepSeek-V4.1-Flash-1017, with UniLM's
development branch after 0.26.0.

See also: [`DeepSeekEndpoint`](@ref), [`Chat`](@ref), [`chatrequest!`](@ref), [`fim_complete`](@ref),
[`prefix_complete`](@ref) and [`LLMSuccess`](@ref) in the API reference.
