# [Anthropic](@id anthropic_start)

```@setup anthropic
using UniLM
```

Anthropic makes the Claude models; `ANTHROPICServiceEndpoint` speaks its native Messages API, and
every Messages API request is a `Chat`, sent by `chatrequest!` or `tool_loop!`. The agentic verbs —
`respond` and the `tool_loop` that takes an input or a `Respond` — refuse this endpoint before
sending anything, and no verb of a capability the table below lists as not available sends a request
through it. Of Anthropic's own API, UniLM wraps Messages and the model listing: Message Batches,
Files, token counting, server tools (the tools Anthropic runs, such as web search), citations and the
request fields of beta features are not available.

```@eval
Main.ProviderFacts.at_a_glance(:anthropic)
```

| I want to… | Use | You get |
| :--- | :--- | :--- |
| [make a first call](@ref anthropic_first_call) | `chatrequest!(chat)` | an `LLMSuccess`: the reply and its usage, priced by `estimated_cost` |
| [set a workspace, a region or a beta](@ref anthropic_options) | `ANTHROPICServiceEndpoint(; workspace_id, inference_geo, betas, user_profile_id)` | every Messages API request through the endpoint carrying them |
| [hold a conversation](@ref anthropic_conversations) | `push!` a message, then `chatrequest!` again | each reply appended to the `Chat` and sent back with the next turn |
| [stream a reply](@ref anthropic_streaming) | `stream=true` and a `callback` | the text as it arrives, then the whole message |
| [let Claude call my functions](@ref anthropic_tools) | `tools` and `tool_loop!` | the tool calls run and answered until the model replies |
| [get JSON that fits a schema](@ref anthropic_structured_output) | a JSON Schema `response_format` | a reply constrained to the schema |
| [control thinking](@ref anthropic_reasoning) | `reasoning_effort` at a level your model's row in the family table lists; a row without levels refuses one | that effort, and where the row's thinking is adaptive, a summary of the thinking in `reasoning_text` |
| [send images and PDFs](@ref anthropic_media) | `ImageAttachment`, `DocumentAttachment` | the files placed before the message's text |
| [cache a long prompt](@ref anthropic_caching) | `prompt_cache_options` | cache reads and writes, counted in `usage` |
| [know what a call cost](@ref anthropic_cost) | `estimated_cost`, `cumulative_cost` | USD estimates from the reported usage |
| [list the models my key can use](@ref anthropic_models) | `list_models(service=ANTHROPICServiceEndpoint)` | each model's token limits, capabilities and lifecycle dates |
| [handle a failure](@ref anthropic_failures) | the result's type | the status or the exception, never an empty success |

## [Setup](@id anthropic_setup)

Set your API key in the environment Julia starts from:

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

UniLM reads the variable each time a request is sent; a `chatrequest!` without it returns an
`LLMCallError` whose `cause` is the `KeyError` naming it. A `Chat` reaches Anthropic with
`service=ANTHROPICServiceEndpoint`, the bare type, or with an instance that carries the options
below; with no `model` it takes the default model named at the top of this page. Requests go to
`https://api.anthropic.com/v1/messages` with the key in `x-api-key` and
`anthropic-version: 2026-10-21`.

The Messages API requires `max_tokens`: a `Chat` that sets neither `max_tokens` nor
`max_completion_tokens` sends 16000, a moderate default rather than the model's ceiling. It caps the
thinking and the text of a reply together, and headroom a reply leaves unused is not billed: raise it
for long replies or a high `reasoning_effort`, since a turn whose thinking reaches the cap ends with
`finish_reason == "length"` before the answer.

## [Options](@id anthropic_options)

The bare type is the endpoint with no options. An instance,
`ANTHROPICServiceEndpoint(; betas, user_profile_id, workspace_id, inference_geo)`, adds them to every
Messages API request made through it, and goes wherever a `service` goes:

| Option | Sent as | Checked |
| :--- | :--- | :--- |
| `betas` | the names, comma-joined, in the `anthropic-beta` header | each name non-blank, without a control character or a comma, when the endpoint is built and again when a request is sent |
| `user_profile_id` | the `anthropic-user-profile-id` header: the `uprof_` id of the end customer the requests are made for | non-blank, without a control character (CR, LF, …) |
| `workspace_id` | the `anthropic-workspace-id` header: the workspace a request runs in, for a key that can act in more than one | non-blank, without a control character |
| `inference_geo` | the body's `inference_geo`, where inference runs: `"global"` or `"us"`; unset, the workspace default | one of the two; a model whose `inference_geo` cell in the [family table](@ref anthropic_reasoning) is "—" refuses it before the request is sent |

```julia
claude = ANTHROPICServiceEndpoint(workspace_id="wrkspc_01ExampleWorkspace", inference_geo="us")
chat = Chat(service=claude)
```

Building the endpoint throws `ArgumentError` for a value that breaks one of these rules; two checks
come later, when a request is made: the model's `inference_geo` rule, which makes `chatrequest!`
throw `ArgumentError` before anything is sent, and the re-check of `betas` below. UniLM sends a
beta's name only: it models no beta's request fields or response blocks, and reads one name itself,
the beta `service_tier="fast"` needs on the models whose Fast mode cell in the family table names a
beta header. [`list_models`](@ref) sends only the workspace of these options: of the headers they
set, Anthropic's Models API reference names only `anthropic-workspace-id` for `GET /v1/models`.
Instances with the same options are equal (`==`, `hash`). The fields are read-only:
to change an option, build a new endpoint. A name changed in place in the `betas` vector is checked
again when the header is built, and one that breaks the rule fails the call before anything is sent,
its call-error result holding the `ArgumentError` as `cause`. US-only inference is priced at a
premium ([Cost and usage](@ref anthropic_cost)).

## [First call](@id anthropic_first_call)

A `Chat` starts with its system message; `chatrequest!` sends it and returns a typed result, and
`text` reads the reply — or throws, on a failure:

```@example anthropic
chat = Chat(service=ANTHROPICServiceEndpoint)
push!(chat, Message(Val(:system), "You are a concise assistant."))
push!(chat, Message(Val(:user), "In one sentence: what is a prime number?"))
result = chatrequest!(chat)
text(result)
```

```@example anthropic
result.model, result.message.finish_reason
```

`result.model` is the model that served the reply, as Anthropic reports it, and the model
[`estimated_cost`](@ref) prices the call at. The reply's `stop_reason` becomes `finish_reason`:
`end_turn` and `stop_sequence` give `"stop"`, `tool_use` gives `"tool_calls"`, `max_tokens` and
`model_context_window_exceeded` give `"length"`, and `refusal` gives `"content_filter"`, a turn with
no content and the explanation in `refusal_message`. An unstreamed reply with status 200 that is not
a message — no `content` array or no `stop_reason` — is an `LLMCallError`, never an empty success.
A failure result, `LLMFailure` or `LLMCallError`, keeps the `request-id` header of Anthropic's
reply, when one came, as `request_id` ([Rate limits, overload and failures](@ref anthropic_failures)).

```@example anthropic
u = result.usage
(; u.prompt_tokens, u.completion_tokens, u.reasoning_tokens, u.service_tier, u.inference_geo,
   cost = estimated_cost(result))
```

`prompt_tokens` counts all the input — the uncached tokens, the cache reads (`cached_tokens`) and the
cache writes (`cache_write_tokens`) — and `completion_tokens` all the output, thinking included, of
which `reasoning_tokens` is the thinking. `service_tier` is the tier Anthropic reports billing the
call at (`"fast"` for a call served in fast mode) and `inference_geo` the region inference ran in.
How these become a price is in [Cost and usage](@ref anthropic_cost); the result types are on
[Results](@ref results_guide).

## [Conversations](@id anthropic_conversations)

A `Chat` keeps the conversation: each reply is appended to it, and the next `chatrequest!` sends the
whole conversation again.

```@example anthropic
chat = Chat(service=ANTHROPICServiceEndpoint)
push!(chat, Message(Val(:system), "You are a concise assistant: answer in one short sentence."))
push!(chat, Message(Val(:user), "Which planet of the solar system is the largest?"))
text(chatrequest!(chat))
```

```@example anthropic
push!(chat, Message(Val(:user), "And the smallest?"))
text(chatrequest!(chat))
```

```@example anthropic
[m.role for m in chat.messages], cumulative_cost(chat)
```

A reply goes back as the content blocks it arrived in — thinking blocks with their signatures
intact — without its empty text blocks, and a turn with nothing to send, such as a refusal, is left
out. What `push!` refuses is the same on every provider ([Conversations with Chat](@ref chat_guide)).

The system messages a conversation starts with become the top-level `system` prompt, joined with a
blank line. Where a system message pushed later goes depends on two columns of the
[family table](@ref anthropic_reasoning):

- On a model whose Mid-conversation system messages cell is "yes", it stays in place, as a `system`
  turn, where the Messages API takes one: right after a user turn (tool results count as one), and
  last or right before an assistant turn. Anywhere else it joins the top-level `system` prompt,
  except on an append-only model once an assistant turn is sent (below).
- On a model whose cell is "—", it joins the top-level `system` prompt.
- Append-only are the models whose Thinking prefix check cell is "yes", and any id that matches no
  row (an id with a `-YYYYMMDD` snapshot suffix takes the row of the id without it), since Anthropic
  asks every integration to be append-only
  ([preserved thinking](https://platform.claude.com/docs/en/build-with-claude/preserved-thinking)).
  The models with the check accept an earlier turn's thinking blocks only while everything before
  them is unchanged — the top-level `system` prompt, the tools and the earlier messages. So once an
  assistant turn is sent, UniLM never edits the top-level `system` prompt of an append-only model: a
  later system message that cannot stay in place is held back, then sent right after a later user
  turn, and the turns before each sent reply stay as they were in the request that got it. Push your
  own messages and send the replies back as they came: an edit to an earlier turn, or to the tools,
  invalidates the thinking blocks after it, and the API answers HTTP 400.

On every model, a system message pushed after a reply needs a user turn after it — a user message,
or the results of the reply's tool calls. Until one follows, `chatrequest!` throws `ArgumentError`
before sending: as it stands, the request would end with the assistant turn, a response prefill.

## [Streaming](@id anthropic_streaming)

With `stream=true`, `chatrequest!` returns a `Task` at once, and `fetch` gives the same typed result
as an unstreamed call. The `callback` receives the text as it arrives, a `String` at a time, then the
assembled `Message`:

```@example anthropic
chat = Chat(service=ANTHROPICServiceEndpoint, stream=true)
push!(chat, Message(Val(:system), "You are a concise assistant."))
push!(chat, Message(Val(:user), "List the planets of the solar system in order, one per line."))
deltas = String[]
task = chatrequest!(chat; callback=(chunk, close) -> chunk isa String && push!(deltas, chunk))
result = fetch(task)
text(result)
```

```@example anthropic
deltas
```

```@example anthropic
u = result.usage
(; u.prompt_tokens, u.completion_tokens, cost = estimated_cost(result))
```

A `Chat` streams through this `callback` keyword only: `chatrequest!` has no `do`-block form, which
is `respond`'s, and `respond` refuses this endpoint. Setting `close[] = true` stops the stream
([Stopping a Stream Early](@ref streaming_stop)).

- **End.** The turn is final at Anthropic's `message_stop` event: the `Message` reaches the callback
  then, and `fetch` returns when the HTTP body ends, at most 0.5 s later.
- **Usage.** The stream reports it itself — the input in `message_start`, the output in
  `message_delta` — so a streamed turn has `usage` and adds its cost to `cumulative_cost` as an
  unstreamed one does, with no `stream_options` (a field this endpoint refuses). A stream that ends
  without committing its turn — stopped, cancelled, timed out, cut off or ended by a callback's
  exception — adds the usage Anthropic had reported by then, the input arriving with the first
  event, as a lower bound of what was billed.
- **Thinking.** No thinking delta reaches the callback, only text: the thinking arrives in the
  `provider_content` of the `Message` the callback receives last, the result's message, where
  [`reasoning_text`](@ref) reads it.
- **Content blocks.** Thinking blocks with their signatures, text and tool calls are re-assembled
  from the deltas into the message's `provider_content` once every block has arrived whole, so such a
  streamed turn goes back on the next request as an unstreamed one does
  ([what goes back otherwise](@ref anthropic_reasoning)).
- **Tool calls.** `on_tool_call(tc)` runs once per call as soon as its block closes
  (`content_block_stop`), before the turn ends; the final `Message` holds the calls too.
- **Refusals.** A refusal ends the turn with no content, as an unstreamed refusal does; the text and
  the tool calls already passed to the callbacks cannot be recalled.
- **Pings.** Anthropic's `ping` events are bytes on the wire, so they reset the idle clock:
  `stream_idle_timeout` bounds the gap between bytes. The longest silent gap of a Claude stream that
  thinks has not been measured for this release ([Timeouts & Retries](@ref timeout_streams)).

The concepts every provider shares are on [Streaming](@ref streaming_guide).

Recorded on 24 October 2026 on `claude-opus-5-5` with UniLM's development branch after 0.26.0 and
the Messages API at `anthropic-version` 2026-10-21.

See also: [`ANTHROPICServiceEndpoint`](@ref), [`Chat`](@ref), [`chatrequest!`](@ref) and
[`LLMSuccess`](@ref) in the API reference.
