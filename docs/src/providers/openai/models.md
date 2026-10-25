# [Models & request rules](@id openai_models_page)

```@setup openai
using UniLM
```

## [Models](@id openai_models)

A request names its model with `model`, and UniLM sends any id as given; without one, a `Chat` or a
`Respond` takes the default model named at the top of the [start page](@ref openai_start). The models
UniLM holds request rules for are the rows of the [rule table](@ref openai_reasoning), and the models
it prices are the rows of the [price table](@ref openai_cost). A model id takes the row of the same
id, or of that id without a `-YYYY-MM-DD` snapshot suffix, never the row of an id it merely starts
with: a request on a model no row names gets none of the per-model checks, and OpenAI's answer is the
check; a result from a model with no price row is estimated at `0.0`, with a warning.

[`list_models`](@ref) with `service=OPENAIServiceEndpoint` asks OpenAI's Models API (`GET /v1/models`)
which models your key can use, and returns a `ModelListSuccess` whose `models` are
[`ProviderModel`](@ref)s. OpenAI lists each model's `id`, its creation time, as `created`, its owner and
its `shutdown_date`, "the date when the model will shut down, or null if not announced"
([List models](https://developers.openai.com/api/reference/resources/models/methods/list)); UniLM reads
the shutdown date as `retires`, keeps the entry as sent in `raw`, and has no token limits or
capabilities to fill in for OpenAI ([Model Listings](@ref models_api)). An error status is a
`ModelListFailure`, and a call that brings no listing — no key, which it names, no connection, a
timeout, a reply that is not the listing — a `ModelListCallError`. A listing bills no tokens and is not
recorded, so its output here is quoted from a live run:

```julia
r = list_models(service=OPENAIServiceEndpoint)
r isa ModelListSuccess || error("the listing failed: $r")
ids = sort!([m.id for m in r.models if startswith(m.id, "gpt-6") || startswith(m.id, "gpt-5.6")])
priced(id) = haskey(DEFAULT_PRICING, id) || haskey(DEFAULT_PRICING, replace(id, r"-\d{4}-\d{2}-\d{2}$" => ""))
ids, filter(!priced, ids)
# Listed on 25 October 2026 with UniLM's development branch after 0.26.0 (133 models in all):
# (["gpt-5.6-luna", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-6-astra", "gpt-6-luna", "gpt-6-sol", "gpt-6.1-sol"], String[])

m = r.models[findfirst(m -> m.id == "gpt-6-sol", r.models)]
m.id, m.created, m.retires, m.raw["owned_by"]
# ("gpt-6-sol", DateTime("2026-09-14T17:12:17"), nothing, "system")
```

The listing also names models whose shutdown date has passed. Before you use a newly listed model,
check that it has a price row, as the lookup above does, and whether it needs request rules UniLM does
not hold yet: OpenAI's model pages give each model's reasoning efforts and the APIs it serves.

OpenAI describes the models of the rule table, and `gpt-5.4-mini`, on which the manual's concept pages
run their examples, in its own words:

| Model | OpenAI's description |
| :--- | :--- |
| `gpt-6-sol` | "built for complex coding and agentic workflows" ([GPT-6 Sol](https://developers.openai.com/api/docs/models/gpt-6-sol)) |
| `gpt-6.1-sol` | "Near-Astra performance for complex work at a lower cost" ([GPT-6 guide](https://developers.openai.com/api/docs/guides/latest-model/gpt-6-astra)) |
| `gpt-6-astra` | "Highest intelligence", "For the most demanding reasoning, coding, and professional work" ([GPT-6 guide](https://developers.openai.com/api/docs/guides/latest-model/gpt-6-astra)) |
| `gpt-6-luna` | "Fastest and most cost-effective", "Strong performance for focused, high-volume tasks" ([GPT-6 guide](https://developers.openai.com/api/docs/guides/latest-model/gpt-6-astra)) |
| `gpt-5.6` | "The `gpt-5.6` alias routes requests to `gpt-5.6-sol`" ([GPT-5.6 guide](https://developers.openai.com/api/docs/guides/latest-model/gpt-5.6)) |
| `gpt-5.6-sol` | "the model for flagship capability" ([GPT-5.6 guide](https://developers.openai.com/api/docs/guides/latest-model/gpt-5.6)), whose "promotional pricing is available at least through November 21, 2026" ([pricing](https://developers.openai.com/api/docs/pricing)) |
| `gpt-5.6-terra` | "for a balance of intelligence and cost" ([GPT-5.6 guide](https://developers.openai.com/api/docs/guides/latest-model/gpt-5.6)) |
| `gpt-5.6-luna` | "for efficient, high-volume workloads" ([GPT-5.6 guide](https://developers.openai.com/api/docs/guides/latest-model/gpt-5.6)) |
| `gpt-5.4-mini` | "Reasoning.effort supports: none (default), low, medium, high and xhigh" ([GPT-5.4 Mini](https://developers.openai.com/api/docs/models/gpt-5.4-mini)) |

## [Reasoning](@id openai_reasoning)

`reasoning_effort` on a `Chat` and `reasoning=Reasoning(effort=…)` on a `Respond` set how much a
reasoning model reasons before it answers. UniLM takes the efforts `"none"`, `"minimal"`, `"low"`,
`"medium"`, `"high"`, `"xhigh"` and `"max"`, and checks the request against the model's row of the
table below when it is encoded, before anything is sent ([Limits and refusals](@ref openai_limits)):

```@eval
Main.ProviderFacts.openai_model_rules()
```

The columns:

- **Efforts on Responses** and **Efforts on Chat Completions**: the efforts each API takes for the
  model. Another is refused, and the message names the next effort up that the row lists, or `Respond`
  where only the Responses API takes it. OpenAI's reference for Chat Completions lists `"max"` among
  the efforts, but the API answered `"max"` with HTTP 400 when these rows were written, on
  22 October 2026, and again on the default model on 25 October 2026, so their Chat Completions cells
  end at `"xhigh"`.
- **Sampling and log probabilities at**: the efforts at which UniLM sends `temperature`, `top_p` and
  log probabilities, refusing them at the others; "—": at none; "any": UniLM checks no effort for the
  model, OpenAI's answer being the check. OpenAI's GPT-6 guide: "When reasoning effort is not `none`,
  remove `temperature`, `top_p`, and `top_logprobs`. For Chat Completions, also remove `logprobs`. For
  Responses, remove `message.output_text.logprobs` from `include`"
  ([GPT-6 guide](https://developers.openai.com/api/docs/guides/latest-model/gpt-6-astra#update-api-and-model-parameters)).
  OpenAI's pages state no such rule for the GPT-5.6 models, whose cells are "any"; on 25 October 2026 the
  API refused `temperature` on one of them at an effort other than `"none"`, and took it at `"none"`
  (the evidence below).
- **Chat Completions tools at**: the efforts at which a `Chat` may carry `tools`; "—": the model takes
  tools only on the Responses API ([Tools on Chat Completions](@ref openai_chat_tools)).
- **Ultrafast tier**: whether a `Respond` may ask for `service_tier="ultrafast"`
  ([Request options](@ref openai_responses_options)).
- **`prompt_cache_retention`**: whether a `Chat` or a `Respond` may set it ([Prompt caching](@ref openai_caching)).
- **Default effort**: the effort UniLM checks a request that sets none at ([Setup](@ref openai_setup)),
  meant as the one OpenAI runs it at. OpenAI's reasoning guide: "If you omit `reasoning.effort`, GPT-5.6
  defaults to `medium`", and "GPT-6.1 Sol, GPT-6 Sol, and GPT-6 Luna also default to `medium` reasoning
  effort" ([reasoning](https://developers.openai.com/api/docs/guides/reasoning#reasoning-mode)); its pages
  name no default for GPT-6 Astra, whose response to a request that set none reported `reasoning.effort`
  `"medium"` on 25 October 2026 (the evidence below).

!!! details "Evidence"
    Live requests of 25 October 2026, sent as raw HTTP. `gpt-6-astra` on the Responses API with no `reasoning` and `max_output_tokens=16`:
    HTTP 200, the response's `reasoning` was `{"context": "all_turns", "effort": "medium", "mode":
    "standard", "summary": null}`. `gpt-5.6-luna` with `temperature=0.5`: at `reasoning_effort="low"`,
    HTTP 400 on Chat Completions, "Unsupported value: 'temperature' does not support 0.5 with this model.
    Only the default (1) value is supported."; with `reasoning={"effort": "low"}`, HTTP 400 on the
    Responses API, "Unsupported parameter: 'temperature' is not supported with this model."; at
    `reasoning_effort="none"`, HTTP 200 on Chat Completions.

`Reasoning` configures the reasoning of whatever model a `Respond` names: UniLM checks its effort per
model only for the rows above, and sends it unchecked for a model no row names, OpenAI's answer being
the check. On the Responses API a summary of the reasoning comes back when the request asks
for one, as the `summary` array of the response's `reasoning` items
([reasoning summaries](https://developers.openai.com/api/docs/guides/reasoning#reasoning-summaries)), and
[`reasoning_summaries`](@ref) reads it:

```@example openai
result = respond("How many integers from 1 to 1000 are divisible by 3 or by 5 but not by 7? " *
                 "Answer with the number only."; service=OPENAIServiceEndpoint,
                 reasoning=Reasoning(effort="high", summary="auto"))
output_text(result)
```

```@example openai
foreach(println, reasoning_summaries(result))
```

```@example openai
u = token_usage(result)
(; u.completion_tokens, u.reasoning_tokens)
```

Reasoning tokens "are billed as output tokens"
([reasoning](https://developers.openai.com/api/docs/guides/reasoning#how-reasoning-works)):
`completion_tokens` counts them, `reasoning_tokens` is their part, and
[`estimated_cost`](@ref) prices them at the output rate. Chat Completions returns no reasoning text, so
[`reasoning_text`](@ref) of a `Chat` result is `nothing`. A request's output cap —
`max_completion_tokens` on a `Chat`, `max_output_tokens` on a `Respond` — bounds the reasoning and the
visible output together: a reply that reaches it ends with `finish_reason == "length"`, or a response
with status `"incomplete"`, possibly before any visible text.

`Reasoning` also takes OpenAI's persisted reasoning, `context` (`"auto"`, `"current_turn"` or
`"all_turns"`), and its execution mode, `mode` (`"standard"` or `"pro"`), and a `Respond` input may hold
a [`configuration_update`](@ref) item, which changes the effort of later responses without rewriting
the request-level effort, so the cached prompt prefix stays reusable; OpenAI supports it on "the GPT-6
model family in standard, single-agent mode"
([change reasoning mid-conversation](https://developers.openai.com/api/docs/guides/reasoning#change-reasoning-mid-conversation)).

## [Prompt caching](@id openai_caching)

OpenAI caches prompt prefixes by itself. On GPT-5.6 and later, `prompt_cache_options` chooses how: a
[`PromptCacheOptions`](@ref) with a `mode`, `"implicit"` (the API's default: OpenAI places a breakpoint
at the end of the latest eligible message) or `"explicit"` (only the breakpoints you mark), and a
`ttl`, the minimum lifetime of what a request writes. A prefix "must meet the model's minimum cacheable
token length", 1,024 tokens on GPT-5.6 and later
([prompt caching](https://developers.openai.com/api/docs/guides/prompt-caching#how-caching-works)). The
two APIs take different options:

- **Chat Completions** takes `mode` and `ttl="30m"`; another `ttl`, `prewarm` and
  `comparison_response_id` are refused ([Limits and refusals](@ref openai_limits)). A `Chat` marks no
  breakpoint, so with `mode="explicit"` its request "does not use prompt caching"
  ([Create chat completion](https://developers.openai.com/api/reference/resources/chat), `prompt_cache_options`).
- **Responses** takes `ttl="30m"` or `"12h"`; `input_text(text; cache_breakpoint=true)` marks a
  breakpoint at the end of that block, with its own `cache_ttl` if given; `prewarm=true` "Prepares the
  prompt cache without generating output"; and `comparison_response_id` asks for diagnostics against an
  earlier response, which come back in `result.response.raw["prompt_cache_diagnostics"]`
  ([Create a model response](https://developers.openai.com/api/reference/resources/responses/methods/create),
  `prompt_cache_options`).

`prompt_cache_key`, on both APIs, is the key OpenAI uses "to cache responses for similar requests to
optimize your cache hit rates". The deprecated `prompt_cache_retention` of a `Chat` or a `Respond` is
refused on the models whose `prompt_cache_retention` cell in the [rule table](@ref openai_reasoning) is
"—", whose cache lifetime `prompt_cache_options` sets, and sent as given on the others. Here a generated parts list is the system prompt of a conversation, and each
turn asks about one part:

```@example openai
parts = join(("Part $(2000 + i): $(3 + i % 7) mm washer, $(11i % 60) in stock." for i in 1:120), "\n")
chat = Chat(service=OPENAIServiceEndpoint, prompt_cache_options=PromptCacheOptions(),
            prompt_cache_key="parts-list")
push!(chat, Message(Val(:system), "Answer questions about this parts list in one short sentence.\n\n" * parts))
push!(chat, Message(Val(:user), "How many of part 2042 are in stock?"))
result = chatrequest!(chat)
text(result)
```

```@example openai
u = result.usage
(; u.prompt_tokens, u.cache_write_tokens, u.cached_tokens, cost = round(estimated_cost(result); digits=6))
```

```@example openai
push!(chat, Message(Val(:user), "And part 2077?"))
result = chatrequest!(chat)
u = result.usage
(; text = text(result), u.prompt_tokens, u.cache_write_tokens, u.cached_tokens,
   cost = round(estimated_cost(result); digits=6))
```

The first request wrote the system prompt and its question to the cache. The second read them back and
wrote only what the conversation had added, the reply and the next question. The usage counts the cache
reads as `cached_tokens` and the writes as `cache_write_tokens`, of which
`cache_write_extended_tokens` are the writes kept for 12 hours, which only the Responses API reports;
what each costs is on [Cost and usage](@ref openai_cost). A `Respond` that marks its own breakpoints
sends them in its input; [`UniLM.encode_agentic`](@ref) shows the body `respond` sends, with no key and
no request:

```@example openai
r = Respond(service=OPENAIServiceEndpoint,
    input=[InputMessage(role="developer",
                        content=[input_text("You answer questions about our parts catalogue."; cache_breakpoint=true,
                                            cache_ttl="12h")]),
           InputMessage(role="user", content="How many washers are in stock?")],
    prompt_cache_options=PromptCacheOptions(mode="explicit", ttl="30m"))
print(UniLM.encode_agentic(r.service, r))
```

## [Cost and usage](@id openai_cost)

[`estimated_cost`](@ref) prices a result with the row of the price table below for the model that
served it — `result.model` of a `Chat` result, `result.response.model` of a `Respond` result — and an id
with a `-YYYY-MM-DD` snapshot suffix with the row of the id without it; [`cumulative_cost`](@ref) adds
up the calls of a `Chat`. Of the usage, it bills the uncached input at the input rate, the cache reads
at the cached-input rate, the cache writes at the multiples of the Cache writes column of the table just
below, and all the output, the reasoning included, at the output rate. Every rate is then multiplied by
the factors that apply:

- the billed tier's multiple, for the tier the result reports as its `service_tier`: `"flex"`, `"fast"`
  or `"priority"` (Fast mode's former name, which OpenAI reports for Fast on GPT-5.6 and earlier), and
  `"ultrafast"`;
- for a prompt over the short-context limit, the long-context multiples, on the whole request.

```@eval
Main.ProviderFacts.openai_price_terms()
```

A "—" in the table is a price OpenAI's pricing page does not give: a write then bills at the input
rate, any prompt at the row's rates, and a tier at the standard rates, with one warning per model and
tier. So does a tier the table has no column for, such as Scale, and every tier of a model without a
row in this table. OpenAI bills a request that Fast mode downgraded at standard rates and reports it as
`service_tier: "default"` ([Fast mode](https://developers.openai.com/api/docs/guides/fast-mode)), and the
estimate follows the reported tier.

Audio tokens — `prompt_audio_tokens` and `completion_audio_tokens` — bill at the audio rates below,
scaled by the factors above as the text rates are, and the rest of the call at the text rates of the
model's row; a call whose usage counts audio on a model without audio rates is estimated at `0.0`, with a
warning ([Audio replies](@ref chat_audio_output)):

```@eval
Main.ProviderFacts.openai_audio_rates()
```

[`cumulative_cost`](@ref) is a lower bound of the bill: it adds nothing for an attempt UniLM retried, a
failure, or a stream stopped before its usage arrived ([Streaming](@ref openai_streaming)), and no
estimate includes the fees of hosted tools — web search, file search and containers bill per call or
per session — or the 10% uplift OpenAI charges on its regional-processing endpoints for the models
released since 5 March 2026 ([pricing](https://developers.openai.com/api/docs/pricing)). A model with no
row below is estimated at `0.0`, with one warning per model id: pass `estimated_cost` a `pricing` table
that holds its row, or add the row to [`DEFAULT_PRICING`](@ref) ([Cost Tracking](@ref cost_guide)).

The rows are OpenAI's list prices as checked on 25 October 2026; prices change, so check OpenAI's
[pricing page](https://developers.openai.com/api/docs/pricing) before relying on an estimate.

```@eval
Main.ProviderFacts.price_table(:openai)
```

## [Limits and refusals](@id openai_limits)

A request's output stops at its cap — `max_completion_tokens` on a `Chat`, `max_output_tokens` on a
`Respond` — which counts the reasoning too ([Reasoning](@ref openai_reasoning)), and a model's context
window and largest output are on its OpenAI model page. UniLM checks a request against neither, and
leaves the size of images and files to the API. Rate limits are on
[When something fails](@ref openai_failures).

`chatrequest!`, `respond`, the tool loops and `embeddingrequest!` throw these `ArgumentError`s before
anything is sent, and so do the constructors of the request's parts where the row names one. A rule
that names a column of the [rule table](@ref openai_reasoning) holds for the ids of its rows, and an id
no row names meets only the other rules.

| Refused | Why: the message's words | What to do |
| :--- | :--- | :--- |
| an effort the model's row does not list for the API: `reasoning_effort` on a `Chat`, `Reasoning(effort=…)` on a `Respond` | `on OpenAI Chat Completions does not take reasoning effort`, `on OpenAI Responses does not take reasoning effort`: the model does not take it there; the message names the next effort up the row lists, or `Respond` where only the Responses API takes it | an effort the row lists |
| `temperature`, `top_p`, `logprobs=true` or `top_logprobs` on a `Chat`; `temperature`, `top_p`, `top_logprobs` or `include=["message.output_text.logprobs"]` on a `Respond`; at an effort the row's Sampling cell does not list, the Default effort when none is set | `does not support sampling controls or log probabilities`: OpenAI asks to remove them when the effort is not one the model takes them at | the effort the message names, or none of them |
| `tools` on a `Chat` at an effort the row's Chat Completions tools cell does not list | `takes function tools only through the Responses API`, `Chat tools require reasoning_effort`: the model calls tools on Chat Completions only at that effort, or not at all | the effort the message names, or a `Respond` ([Tools](@ref openai_tools)) |
| a `service_tier` the API does not take | `OpenAI Chat Completions accepts service_tier`, `OpenAI Responses accepts service_tier`: Chat Completions takes `"auto"`, `"default"`, `"flex"`, `"scale"`, `"priority"` and `"fast"`; Responses also `"ultrafast"` | one of them |
| `service_tier="ultrafast"` on a `Respond` whose model's Ultrafast tier cell is "—", or names no row | `has no ultrafast tier`: OpenAI serves the tier only on the models whose cell is "yes" | `"fast"`, or one of those models |
| `user` on a `Chat` or a `Respond` | `OpenAI replaces the deprecated user field`: the field "is being replaced by `safety_identifier` and `prompt_cache_key`" | `safety_identifier` for the end user, `prompt_cache_key` for cache routing |
| `max_tokens` and `max_completion_tokens` both set on a `Chat` | `OpenAI sends max_tokens as max_completion_tokens`: `max_tokens` "is now deprecated in favor of `max_completion_tokens`", which UniLM sends it as | one of them |
| `prompt_cache_options` with `prewarm` or `comparison_response_id` on a `Chat` | `Chat Completions prompt_cache_options accepts only mode and ttl`: the Chat Completions object has no other field | a `Respond` ([Prompt caching](@ref openai_caching)) |
| a `prompt_cache_options` `ttl` of `"5m"` or `"1h"`, or `"12h"` on a `Chat` | `OpenAI Chat Completions accepts prompt_cache_options ttl`, `OpenAI Responses accepts prompt_cache_options ttl`: the API answers HTTP 400 for it | `"30m"`, or `"12h"` on a `Respond` |
| `prompt_cache_retention` on a `Chat` or a `Respond` whose model's `prompt_cache_retention` cell is "—" | `takes its cache lifetime from prompt_cache_options`: OpenAI deprecates the field, "Use `prompt_cache_options.ttl` instead" | `prompt_cache_options=PromptCacheOptions(ttl="30m")`, or `"12h"` on a `Respond` |
| `seed` on a `Respond` | `Responses API takes no seed`: the Responses API has no seed, while Chat Completions has one | a `Chat`, or no seed |
| `conversation` and `previous_response_id` together on a `Respond` | `conversation and previous_response_id cannot be combined`: `previous_response_id` "Cannot be used in conjunction with `conversation`" | one of them |
| `background=true` with `store=false` on a `Respond` | `background=true requires a stored response`: a rule of UniLM's, stricter than OpenAI's [background guide](https://developers.openai.com/api/docs/guides/background), which lets a background request use `store=false` | omit `store`, or `store=true` |
| an `ImageAttachment`, `AudioAttachment` or `DocumentAttachment` in a `Respond`'s input | `cannot be written as JSON`: the Responses API takes an image as an `input_image` part and a document as an `input_file` part, and UniLM sends it no audio | `input_image` with a data URL, `input_file(document)`; audio on a `Chat` ([Input](@ref openai_responses_input)) |
| an `InputMessage` `phase` other than `"commentary"` and `"final_answer"`, or a phase on a message that is not the assistant's | `InputMessage phase must be`, `InputMessage phase labels an assistant message`: a phase labels "an `assistant` message" and is "Not used for user messages" | the phase the model gave the message, on that assistant message |
| `input_text` with a `cache_ttl` other than `"30m"` and `"12h"`, or without `cache_breakpoint=true` | `input_text cache_ttl must be`, `input_text cache_ttl applies to a cache breakpoint`: a lifetime belongs to a breakpoint | `cache_breakpoint=true` and one of the two |
| `input_image` with neither `url` nor `file_id`; `input_file` with none of `url`, `id` and `file_data` | `must be provided`: a part needs its content | one of them |
| a `FunctionTool` `allowed_callers` other than `"direct"` and `"programmatic"` | `FunctionTool allowed_callers must be`: UniLM sends only those two invocation contexts | one of them |
| a `Reasoning` `context` or `mode`, or a `configuration_update` effort, outside OpenAI's values | `reasoning context must be`, `reasoning mode must be`, `configuration_update effort must be one of`: UniLM sends only the values the message lists | one of those values |
| an `image_generation_tool` `action`, `moderation`, `partial_images` or `input_fidelity` outside its values | `image_generation action must be`, `image_generation moderation must be`, `image_generation partial_images must be`, `image_generation input_fidelity must be`: UniLM sends only the values the message lists | one of those values |
| a `ModerationConfig` `input_mode` or `output_mode` other than `"score"` and `"block"` | `moderation input_mode must be`, `moderation output_mode must be`: a moderation policy's mode is one of the two | `"score"` or `"block"` |
| an image or a sound clip in an `Embeddings` input | `the OpenAI Embeddings API takes text only`: the Embeddings API embeds text | text ([Embeddings](@ref openai_embeddings)) |
| an embedding `task` other than `:none`, or a `title`, on an OpenAI embedding model, by the `Embeddings` constructor | `gets none: pass task = :none`, `has no task prompts, so a title has nowhere to go`: UniLM writes a task prompt, and a title in it, only for the embedding models it knows to read one, and OpenAI's are not among them | no `task` and no `title` |

The image, audio and Realtime verbs refuse what [Images, audio & realtime](@ref openai_media_limits)
lists, and the platform verbs what [Platform APIs](@ref openai_platform_limits) lists.

The request's parts refuse a value outside their own sets first, on every endpoint: `PromptCacheOptions`
any `ttl` other than `"5m"`, `"1h"`, `"30m"` and `"12h"` (`prompt cache ttl must be one of`), and
`Reasoning` and `Chat` an effort outside the seven above (`reasoning effort must be one of`,
`reasoning_effort must be one of`).

Recorded on 25 October 2026 on `gpt-6-sol` with UniLM's development branch after 0.26.0.
