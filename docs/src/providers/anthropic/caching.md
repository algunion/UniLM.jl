# [Caching, cost & limits](@id anthropic_caching_page)

```@setup anthropic
using UniLM
```

## [Prompt caching](@id anthropic_caching)

A `Chat` with `prompt_cache_options` turns on Anthropic's automatic caching: UniLM sends a top-level
`cache_control`, `{"type": "ephemeral"}` with the options' `ttl` when one is set, and the API puts the
cache breakpoint on the request's last cacheable block
([prompt caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching)).
`PromptCacheOptions()` asks for the cache's default lifetime, 5 minutes, and `PromptCacheOptions(ttl="1h")`
for an hour. The options a `Chat` cannot send to Anthropic are refused before the request
([Limits and refusals](@ref anthropic_limits)).

The breakpoint is where the cache is written: one entry, holding everything up to and including that
block. A later request reads the entry when it starts with the same blocks up to that breakpoint, and
writes what follows them; the API looks back at most 20 blocks for such an entry. The next turn of a
conversation starts with them, since a `Chat` sends the whole conversation each time, unless a system
message pushed after a reply joins the top-level `system` prompt, which can happen only on the models
whose Thinking prefix check cell in the [family table](@ref anthropic_reasoning) is "—"
([Conversations](@ref anthropic_conversations)): that request then starts with another prompt, so it
reads nothing and writes it all again. Two requests that share only a long system prompt, each with its own
question, write two entries and read none: a `Chat` marks no breakpoint of its own. A prefix shorter than
the model's minimum cacheable prompt, which Anthropic sets per model between 512 and 4,096 tokens, is not
cached, and the API returns no error: the usage then shows neither a write nor a read.

The usage counts the reads and the writes ([First call](@ref anthropic_first_call)), and
`cache_write_extended_tokens` the writes kept for an hour; what each costs is in
[Cost and usage](@ref anthropic_cost). On most models, reads do not count toward Anthropic's rate limit of
input tokens per minute ([rate limits](https://platform.claude.com/docs/en/api/rate-limits)). Here a
generated parts list is the system prompt of a conversation, and each turn asks about one part:

```@example anthropic
parts = join(("Part $(1000 + i): $(4 + i % 9) mm bolt, $(7i % 50) in stock." for i in 1:80), "\n")
chat = Chat(service=ANTHROPICServiceEndpoint, prompt_cache_options=PromptCacheOptions())
push!(chat, Message(Val(:system), "Answer questions about this parts list in one short sentence.\n\n" * parts))
push!(chat, Message(Val(:user), "How many of part 1042 are in stock?"))
result = chatrequest!(chat)
text(result)
```

```@example anthropic
u = result.usage
(; u.prompt_tokens, u.cache_write_tokens, u.cached_tokens, cost = round(estimated_cost(result); digits=6))
```

```@example anthropic
push!(chat, Message(Val(:user), "And part 1077?"))
result = chatrequest!(chat)
text(result)
```

```@example anthropic
u = result.usage
(; u.prompt_tokens, u.cache_write_tokens, u.cached_tokens, cost = round(estimated_cost(result); digits=6))
```

The first request wrote the system prompt and its question to the cache. The second read them back and
wrote only what the conversation had added, the reply and the next question.

## [Cost and usage](@id anthropic_cost)

[`estimated_cost`](@ref) prices a result with the row of the table below for its model
([First call](@ref anthropic_first_call) says which model that is), an id with a `-YYYYMMDD` snapshot
suffix with the row of the id without it, and [`cumulative_cost`](@ref) adds up the calls of a `Chat`. Of
the usage, it bills the uncached input at the input rate, the cache reads at the cached-input rate, the
cache writes at 1.25 times the input rate, or 2 times for the writes kept for an hour
(`cache_write_extended_tokens`), and all the output, thinking included, at the output rate
([pricing](https://platform.claude.com/docs/en/about-claude/pricing)). Every rate is then multiplied by
each factor that applies, the factors stacking:

- 2 for a call served in fast mode, whose usage reports `service_tier == "fast"`
  ([Request options](@ref anthropic_request_options));
- 1.1 for US-only inference, a usage that reports `inference_geo == "us"`, except on the models whose
  `inference_geo` cell in the [family table](@ref anthropic_reasoning) is "—", which bill the standard
  rates;
- a model's long-prompt multiplier, for a prompt over its threshold, all of its input counted, cache
  reads and writes included.

Two kinds of model are priced otherwise: a model with a long-prompt multiplier bills more over its
threshold, and a model of the family table without a price row is estimated at `0.0`:

```@eval
Main.ProviderFacts.claude_price_rules()
```

A call Anthropic reports at Priority Tier (`service_tier == "priority"`) is estimated at the list rates,
as a standard one is: Priority Tier draws down an organization's capacity commitment
([service tiers](https://platform.claude.com/docs/en/api/service-tiers)), which no rate here models. A
model with no row, one of the family table as above or any other id, is estimated at `0.0`, with one
warning per model id: pass [`estimated_cost`](@ref) a `pricing` table that holds its row, or add the row
to [`DEFAULT_PRICING`](@ref). A failure costs `0.0`, since it carries no usage.

How a stream reports its usage, and what a stream that ends uncommitted adds, is on
[Streaming](@ref anthropic_streaming). [`cumulative_cost`](@ref) is a lower bound of the bill: it adds
nothing for an attempt UniLM retried, a stream Anthropic ended with an error event, or an unstreamed
reply UniLM could not decode ([Cost Tracking](@ref cost_guide)).

The rows are Anthropic's list prices as checked on 22 October 2026; prices change, so check Anthropic's
page before relying on an estimate.

```@eval
Main.ProviderFacts.price_table(:anthropic)
```

## [Rate limits, overload and failures](@id anthropic_failures)

`chatrequest!` returns what the service or the network did as a typed result, and does not throw it
([Results](@ref results_guide)). An `LLMFailure` holds an HTTP `status` and, as `response`, what came
with it: the body of a status other than 200; the event of an in-band `overloaded_error`, with status
529; or what a stream that ended before its turn had sent, with status 200. An `LLMCallError` holds, as
`error`, the text of an exception or of another in-band error event, and the exception as `cause` when
there is one. UniLM sends a request again after a transient status, 429 and 529 among them and
Anthropic's 500 and 504, as often as the call's [`RequestConfig`](@ref) allows, and then returns the last
answer ([Timeouts & Retries](@ref timeouts_guide)); a streamed request is sent again only while none of
its callbacks has run.

| What you see | Why | What to do |
| :--- | :--- | :--- |
| `ArgumentError` from `chatrequest!` or `tool_loop!`, nothing sent | the request breaks a rule UniLM checks before sending ([Limits and refusals](@ref anthropic_limits)) | what its row says |
| `InvalidConversationError` from `chatrequest!`, nothing sent | with `history=true`, the conversation ends with an assistant message, so the reply could not be appended | a user message after it, or `history=false` to send it as a prefill ([Request options](@ref anthropic_request_options)) |
| `LLMCallError` whose `cause` is a `KeyError` | `ANTHROPIC_API_KEY` is not set ([Setup](@ref anthropic_setup)) | set it |
| `LLMFailure` with `status` 400, 401, 402, 403, 404, 409 or 413 | Anthropic refused the request, and UniLM does not send it again: `response` holds Anthropic's error, whose `type` names the kind (`invalid_request_error`, `authentication_error`, `request_too_large`, …) and whose `message` the reason ([errors](https://platform.claude.com/docs/en/api/errors)) | what the message says; a 400 that names a spend limit you set lasts until the time its message names, or until you raise or remove the limit |
| `LLMFailure` with `status` 429 | a rate limit (`rate_limit_error`); UniLM sends every Anthropic 429 again, the one of the monthly spend cap too, whose error names `enforced_spend_limit_reached` and which goes on failing until access resumes ([rate limits](https://platform.claude.com/docs/en/api/rate-limits)) | fewer requests or tokens per minute, or more `max_attempts` or a longer `total_deadline`; at the spend cap, a higher usage tier |
| `LLMFailure` with `status` 529 | Anthropic is overloaded (`overloaded_error`) by traffic across all its users; sent again as a 429 is | the request again, later |
| `LLMFailure` with `status` 529 from a stream | an `overloaded_error` event in an HTTP 200 stream, which UniLM retries as a 529 until the request's first `callback` or `on_tool_call` runs, and returns from then on: the text already passed on stays passed, and nothing is appended to the `Chat` | the turn again |
| `LLMCallError` with `status` `nothing` from a stream, the event in `error` | an error event of another type, such as `api_error`; not sent again | the turn again |
| `LLMFailure` with `status` 200 from a stream | the stream ended before the turn did, with no stop reason and no `message_stop`; `response` holds what arrived, and nothing is appended | the turn again |
| `LLMCallError` with `status` `nothing` and a `UniLMTimeout` as `cause` | a bound of the call's `RequestConfig` ran out ([Timeouts & Retries](@ref timeouts_guide)) | a longer bound, or a stream for a long reply |
| `LLMCallError` whose `cause` is the connection's exception, such as an `HTTP.ConnectError` for a refused connection | the network failed, on each attempt the call's `RequestConfig` allows, or on a stream once a callback had run | the connection, then the request again |
| `LLMCallError` whose `cause` is a `UniLMCancelled` | a stop before the turn's stop reason arrived: the stream's `close[] = true` (`:callback`) or a cancelled `CancelToken` (`:token`); nothing is appended ([Streaming](@ref anthropic_streaming)) | nothing, if the stop was yours |
| `LLMCallError` whose `cause` is what your `callback` or `on_tool_call` threw | the exception ended the stream; it is not sent again, and nothing is appended | the callback |
| `LLMCallError`: "Anthropic response is not a message …" | a 200 that is not a message ([First call](@ref anthropic_first_call)) | report it to Anthropic with its `request_id` |
| `finish_reason == "length"` | the reply reached `max_tokens` or the context window ([First call](@ref anthropic_first_call)) | a higher `max_tokens` ([Setup](@ref anthropic_setup)), or a shorter conversation |
| `finish_reason == "content_filter"` | Claude refused: the turn has no content, and `refusal_message` the explanation | another request; the refused turn is left out of the next one ([Conversations](@ref anthropic_conversations)) |
| `LLMResultError` from `text` or `reasoning_text` | the result is a failure, which holds no reply | check `issuccess(result)` first |

## [Models](@id anthropic_models)

The listing is the endpoint's `models` capability: [`list_models`](@ref) with
`service=ANTHROPICServiceEndpoint` asks Anthropic's Models API (`GET /v1/models`) which models your key can
use, fetches every page of the answer, and returns a `ModelListSuccess` whose `models` are
[`ProviderModel`](@ref)s, the most recently released first. Each holds the model's `id` and
`display_name`, its release as `created` (`nothing` where Anthropic gives the Unix epoch, its value for an
unknown date), its `input_token_limit` and its `output_token_limit`, the largest `max_tokens` a request
may set, the `capabilities` object Anthropic sends, the dates it was deprecated (`deprecated`) and
retires (`retires`), and the entry as sent in `raw` ([Model Listings](@ref models_api)). Anthropic lists
the active and the deprecated models, not the retired ones: `raw["lifecycle"]` says which, and a
`retires` date may still move until the model retires
([List Models](https://platform.claude.com/docs/en/api/models/list)). An error status on any page is a
`ModelListFailure`, with its `status`, `response` and `request_id`; a call that brings no listing — no
key, which it names, no connection, a timeout, a reply that is not the listing — is a
`ModelListCallError`.

A listing bills no tokens and is not recorded, so its output here is quoted from a live run:

```julia
r = list_models(service=ANTHROPICServiceEndpoint)
r isa ModelListSuccess || error("the listing failed: $r")
[(m.id, m.input_token_limit, m.output_token_limit, m.retires) for m in r.models]
# Listed on 24 October 2026 with UniLM's development branch after 0.26.0 and the Models API at
# anthropic-version 2026-10-21:
# 13-element Vector{Tuple{String, Int64, Int64, Nothing}}:
#  ("claude-haiku-5-5", 1000000, 128000, nothing)
#  ("claude-sonnet-5-5", 1000000, 128000, nothing)
#  ("claude-opus-5-5", 1000000, 128000, nothing)
#  ("claude-fable-5-1", 1000000, 128000, nothing)
#  ("claude-opus-5", 1000000, 128000, nothing)
#  ("claude-sonnet-5", 1000000, 128000, nothing)
#  ("claude-fable-5", 1000000, 128000, nothing)
#  ("claude-opus-4-8", 1000000, 128000, nothing)
#  ("claude-opus-4-7", 1000000, 128000, nothing)
#  ("claude-sonnet-4-6", 1000000, 128000, nothing)
#  ("claude-opus-4-6", 1000000, 128000, nothing)
#  ("claude-opus-4-5-20251101", 200000, 64000, nothing)
#  ("claude-haiku-4-5-20251001", 200000, 64000, nothing)

caps = r.models[findfirst(m -> m.id == "claude-opus-5-5", r.models)].capabilities
supported(d) = sort([k for (k, v) in d if v isa AbstractDict && get(v, "supported", false) === true])
supported(caps["thinking"]["types"]), supported(caps["effort"])
# (["adaptive"], ["high", "low", "max", "medium", "xhigh"])

priced(id) = haskey(DEFAULT_PRICING, id) || haskey(DEFAULT_PRICING, replace(id, r"-\d{8}$" => ""))
filter(!priced, [m.id for m in r.models])
# String[]
```

The capabilities say what Anthropic takes on each model: its thinking types and effort levels, which the
[family table](@ref anthropic_reasoning) holds UniLM's requests to, image and PDF input, structured
outputs, server tools and more. They show what a model newer than this release takes, too. Before you use
a newly listed model, check that it has a price row, as the last line above does with the lookup of
[`estimated_cost`](@ref): a model without one is estimated at `0.0` ([Cost and usage](@ref anthropic_cost)).
UniLM's release gate runs the same listing, and fails when a listed Claude model of 4.5 or later has no
price row, unless Anthropic publishes no price for it, or no row in the family table, or when a row's
thinking types or effort levels differ from the listing's.

## [The OpenAI-compatible endpoint](@id anthropic_compatible)

Anthropic also serves the OpenAI-compatible wire, at `https://api.anthropic.com/v1/chat/completions`,
"primarily intended to test and compare model capabilities", and not "a long-term or production-ready
solution for most use cases"
([compatibility layer](https://platform.claude.com/docs/en/cli-sdks-libraries/libraries/openai-sdk)).
`ANTHROPICServiceEndpoint` does not use it. To evaluate it, give a [`GenericOpenAIEndpoint`](@ref) the
bare host, since it appends `/v1/chat/completions`, and name a model:

```julia
compat = GenericOpenAIEndpoint("https://api.anthropic.com", "", "ANTHROPIC_API_KEY")
chat = Chat(service=compat, model="claude-opus-5-5")
```

There UniLM is an OpenAI-compatible client and no more:

- **No default model, no Anthropic rules.** A `Chat` without `model` throws `ArgumentError`. Only the
  checks every `Chat` gets run, its constructor's among them (a `temperature` from 0 to 2, `n` of 1, …):
  every field the `Chat` sets is sent, `temperature` and `reasoning_effort` included, and no row of
  [Limits and refusals](@ref anthropic_limits) applies. Anthropic ignores most fields it does not
  support, `response_format`, `reasoning_effort` and a tool's `strict` among them, and joins every system
  message into the top-level `system` prompt.
- **No refusal of other verbs.** The endpoint declares the Responses, embeddings and FIM capabilities, so
  `respond`, `embeddingrequest!` and `fim_complete` send their requests rather than refuse them.
- **No Anthropic headers.** The key goes as `Authorization: Bearer`, with no `anthropic-version` and none
  of the [Options](@ref anthropic_options): no `anthropic-workspace-id`, which Anthropic asks for with a
  key that is not scoped to a workspace.
- **No caching, no thinking.** Anthropic's layer supports no prompt caching and returns no thinking, and
  UniLM keeps no `provider_content` from it: `reasoning_text` gives `nothing`, and no thinking block goes
  back.
- **Standard rates.** [`estimated_cost`](@ref) prices the model's row at its standard rates, none of the
  factors of [Cost and usage](@ref anthropic_cost) applied, and a stream reports its usage only when the
  `Chat` sets `stream_options=Dict("include_usage" => true)`.

Recorded on 24 October 2026 on `claude-opus-5-5` with UniLM's development branch after 0.26.0 and the
Messages API at `anthropic-version` 2026-10-21.
