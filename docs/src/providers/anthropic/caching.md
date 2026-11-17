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

Anthropic's cache "references the entire prompt: `tools`, `system`, and `messages` (in that order)",
and the breakpoint is where it is written: one entry, holding everything up to and including that
block. Anthropic isolates caches by workspace, and a cache is per model ("The cache is per-model",
[cache diagnostics](https://platform.claude.com/docs/en/build-with-claude/cache-diagnostics); "Prompt
caches are per-model", [fallback credit](https://platform.claude.com/docs/en/build-with-claude/fallback-credit)):
a later request to the same model in the same workspace ([Options](@ref anthropic_options)) that starts
with the same blocks up to that breakpoint reads the entry while it lives, unless a change has invalidated
it, and writes what follows them; the API checks
at most 20 positions for such an entry, counting the request's breakpoint as the first and a run of
consecutive `tool_use` blocks, or of consecutive `tool_result` blocks, as one position. Changed tools
invalidate the entire cache, and a changed `system` prompt that prompt and the messages; a changed
`tool_choice`, an image added or removed anywhere in the prompt and changed thinking parameters
invalidate at least the messages, and Anthropic's table lists the other changes and what each
invalidates ([what invalidates the cache](https://platform.claude.com/docs/en/build-with-claude/prompt-caching#what-invalidates-the-cache)).
A `Chat` builds each request from its fields as they are then, the whole conversation and tools included,
so the next turn starts with the previous one's blocks as long as the tools, the `system` prompt and the
earlier turns go out as they did before. A system message pushed mid-conversation can change that: it
may join the top-level `system` prompt, and the request then reads none of what the earlier requests
wrote and writes it all again, or move within the messages ([Conversations](@ref anthropic_conversations)
says how UniLM places one). Two requests that share only a long system prompt, each with its own
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
  reads and writes included;
- 0.5 for a request run in a Message Batch, whose usage reports `service_tier == "batch"`: "a 50 percent
  discount on both input and output tokens" ([Message Batches](@ref anthropic_batches)).

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
[Streaming](@ref anthropic_streaming). [`cumulative_cost`](@ref) is a lower bound of the bill, refusals
Anthropic does not bill aside: it adds nothing for an attempt UniLM retried, a stream Anthropic ended
with an error event, or an unstreamed reply UniLM could not decode ([Cost Tracking](@ref cost_guide)).
It prices every refusal at the usage Anthropic reports, while Anthropic bills a refusal that arrives
before any output only when its `stop_details.category` is one its page lists as billed: `"bio"`,
`"frontier_llm"` and `"reasoning_extraction"` on 25 October 2026
([how refusals are billed](https://platform.claude.com/docs/en/build-with-claude/refusals-and-fallback#how-refusals-are-billed)).

The rows are Anthropic's list prices as checked on 22 October 2026 (`claude-fable-5-2`'s and
`claude-mythos-5-2`'s on 16 November); prices change, so check Anthropic's page before relying on an
estimate.

```@eval
Main.ProviderFacts.price_table(:anthropic)
```

## [Message Batches](@id anthropic_batches)

The Message Batches API runs Messages requests asynchronously, "most batches finishing in less than 1
hour while reducing costs by 50 percent"; a batch whose processing does not end within 24 hours
expires, and its results stay available for 29 days
([batch processing](https://platform.claude.com/docs/en/build-with-claude/batch-processing)).
[`create_batch`](@ref) takes the requests as `custom_id => Chat` pairs, each `Chat` on the batch's
endpoint — `ANTHROPICServiceEndpoint`, or one instance of it, whose options every call on the batch
carries — and each request's `params` are its Messages body, encoded as [`chatrequest!`](@ref) encodes
it, so the request rules of these pages hold for it. A custom id is 1 to 64 letters, digits, hyphens and
underscores, unique in its batch; a streamed `Chat` and fast mode (`service_tier="fast"`, which goes as
`speed`, a parameter a batched request does not take) are refused with an `ArgumentError` before
anything is sent, as is a `Chat` for another endpoint, and `display_name`, which names a Gemini batch. Anthropic
serves no embeddings, so a batch of `Embeddings` is refused there too.

```julia
ask() = Chat(service=ANTHROPICServiceEndpoint, model="claude-haiku-5-5", max_tokens=16,
             messages=[Message(Val(:user), "Reply with exactly: ok")])
batch = create_batch(["one" => ask(), "two" => ask()])
batch isa BatchSuccess || error("the batch was not created: $batch")
batch.response.status, batch.response.request_counts["processing"]
# ("in_progress", 2)

done = poll_batch(batch.response.id; service=ANTHROPICServiceEndpoint, interval=30)
results = batch_results(batch.response.id; service=ANTHROPICServiceEndpoint)
results isa BatchResultsSuccess || error("the results were not read: $results")
[(id, r.message.content, r.usage.service_tier, estimated_cost(r)) for (id, r) in results.results]
# Run on 16 November 2026 with UniLM's development branch after 0.33.0; the batch ended in under
# 3 minutes:
# [("one", "ok", "batch", 1.9e-6), ("two", "ok", "batch", 1.9e-6)]
```

A batch's `status` is its `processing_status`: `in_progress`, `canceling` after
[`cancel_batch`](@ref), and `ended` once every request has a result; [`poll_batch`](@ref) reads it until
`ended`, and `request_counts` holds the tallies Anthropic reports. [`batch_results`](@ref) then reads the
results, which come "not guaranteed to be in the same order as requests", each paired with its custom
id: a succeeded request is the [`LLMSuccess`](@ref) `chatrequest!` returns for its message, its usage
naming the tier `"batch"`, so [`estimated_cost`](@ref) bills it at half the rates
([Cost and usage](@ref anthropic_cost)); an errored one is an [`LLMCallError`](@ref) whose `error` is
Anthropic's error object as JSON — `invalid_request_error` for a request to fix, another type for one
that "can be retried directly" — and whose `request_id` is the one it names; a request `canceled` or `expired` before it
ran, which Anthropic does not bill, is an `LLMCallError` saying so. Before the batch ends, the results
are Anthropic's HTTP 404, a `BatchFailure`. A result's `self` is a `Chat` naming the model that served
it, without messages: a batch keeps no request of its own. Pass the pairs that created the batch,
`batch_results(id; service, requests)`, and each result's `self` is its request's own `Chat`, a copy with
the reply appended for a success, so the conversation goes on with `chatrequest!`.

[`list_batches`](@ref) lists the workspace's batches, most recent first (`limit`, and `after`, a batch
id, sent as `after_id`), [`retrieve_batch`](@ref) reads one, and [`delete_batch`](@ref) deletes an
ended one ("If you'd like to delete an in-progress batch, you must first cancel it"); a batch cannot be
changed once submitted, so [`update_batch`](@ref) throws `ArgumentError` here. Every call sends the
endpoint's workspace, and its `betas` in the `anthropic-beta` header — the beta form of each call:
`"output-300k-2026-03-24"` raises `max_tokens` to 300,000 in a batch on the models Anthropic lists for it
—, and the create alone sends its user profile, which "Applies to every request in the batch".

## [Rate limits, overload and failures](@id anthropic_failures)

`chatrequest!` returns what the service or the network did as a typed result, and does not throw it
([Results](@ref results_guide)). An `LLMFailure` holds an HTTP `status` and, as `response`, what came
with it: the body of a status other than 200; the event of an in-band `overloaded_error`, with status
529; or what a stream that ended before its turn had sent, with status 200. An `LLMCallError` holds, as
`error`, the text of an exception or of another in-band error event, and the exception as `cause` when
there is one. UniLM sends a request again after a transient status, 429 and 529 among them and
Anthropic's 500 and 504, as often as the call's [`RequestConfig`](@ref) allows, and then returns the last
answer ([Timeouts & Retries](@ref timeouts_guide)); a streamed request is sent again only while none of
its callbacks has run. A 429 of the monthly spend cap is not transient: UniLM returns it at once.

| What you see | Why | What to do |
| :--- | :--- | :--- |
| `ArgumentError` from `chatrequest!` or `tool_loop!`, nothing sent | the request breaks a rule UniLM checks before sending ([Limits and refusals](@ref anthropic_limits)) | what its row says |
| `InvalidConversationError` from `chatrequest!`, nothing sent | with `history=true`, the conversation ends with an assistant message, so the reply could not be appended | a user message after it, or `history=false` to send it as a prefill ([Request options](@ref anthropic_request_options)) |
| `LLMCallError` whose `cause` is a `KeyError` | `ANTHROPIC_API_KEY` is not set ([Setup](@ref anthropic_setup)) | set it |
| `LLMFailure` with `status` 400, 401, 402, 403, 404, 409 or 413 | Anthropic refused the request, and UniLM does not send it again: `response` holds Anthropic's error, whose `type` names the kind (`invalid_request_error`, `authentication_error`, `request_too_large`, …) and whose `message` the reason ([errors](https://platform.claude.com/docs/en/api/errors)) | what the message says; a 400 that names a spend limit you set lasts until the time its message names, or until you raise or remove the limit |
| `LLMFailure` with `status` 429 | a rate limit (`rate_limit_error`), which UniLM sends again; or the monthly spend cap, whose error's `details.error_code` is `enforced_spend_limit_reached` and which goes on failing until access resumes, so UniLM returns it at once ([rate limits](https://platform.claude.com/docs/en/api/rate-limits)) | fewer requests or tokens per minute, or more `max_attempts` or a longer `total_deadline`; at the spend cap, a higher usage tier |
| `LLMFailure` with `status` 529 | Anthropic is overloaded (`overloaded_error`) by traffic across all its users; sent again as a 429 is | the request again, later |
| `LLMFailure` with `status` 529 from a stream | an `overloaded_error` event in an HTTP 200 stream, which UniLM retries as a 529 until the request's first `callback` or `on_tool_call` runs, and returns from then on: the text already passed on stays passed, and nothing is appended to the `Chat` | the turn again |
| `LLMCallError` with `status` `nothing` from a stream, the event in `error` | an error event of another type, such as `api_error`; not sent again | the turn again |
| `LLMFailure` with `status` 200 from a stream | the stream ended before the turn did, with no stop reason and no `message_stop`; `response` holds what arrived, and nothing is appended | the turn again |
| `LLMCallError` with `status` `nothing` and a `UniLMTimeout` as `cause` | a bound of the call's `RequestConfig` ran out ([Timeouts & Retries](@ref timeouts_guide)) | a longer bound, or a stream for a long reply |
| `LLMCallError` whose `cause` is the connection's exception, such as an `HTTP.ConnectError` for a refused connection | the network failed, on each attempt the call's `RequestConfig` allows, or on a stream once a callback had run | the connection, then the request again |
| `LLMCallError` whose `cause` is a `UniLMCancelled` | a stop before the turn's stop reason arrived: the stream's `close[] = true` (`:callback`) or a cancelled `CancelToken` (`:token`); nothing is appended ([Streaming](@ref anthropic_streaming)) | nothing, if the stop was yours |
| `LLMCallError` whose `cause` is the exception your `callback` or `on_tool_call` threw | the exception ended the stream; it is not sent again, and nothing is appended; a thrown value that is no `Exception` leaves `cause` `nothing`, its text in `error`; an `InterruptException`, or a recorded call's `ReplayMissError` or `RecordingWriteError`, is thrown at `fetch` instead | the callback |
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

[`retrieve_model`](@ref) with `service=ANTHROPICServiceEndpoint` looks up one model by id
(`GET /v1/models/{model_id}`), with the same headers as the listing, and returns a `ModelSuccess` whose
`model` is the `ProviderModel` the listing would hold for it. An alias comes back as the dated id it
names, and a retired model, or one the key cannot use, is a `ModelListFailure` with status 404
([One model by id](@ref models_lookup)):

```julia
r = retrieve_model("claude-opus-4-5"; service=ANTHROPICServiceEndpoint)
r isa ModelSuccess && (r.model.id, r.model.output_token_limit)
# Looked up on 16 November 2026: ("claude-opus-4-5-20251101", 64000)
```

The listing asks for 1000 models a page, the most the API serves, and sends no `lifecycle` filter and no
`anthropic-beta` header: without a filter Anthropic lists the active and the deprecated models, and the
listing came back the same with each of the endpoint betas tried ([Model Listings](@ref models_api)).

## [Token counts](@id anthropic_token_counts)

[`count_tokens`](@ref) asks Anthropic how many input tokens a `Chat` would bill (`POST
/v1/messages/count_tokens`), without creating a message. It sends the body `chatrequest!` sends without
`max_tokens`, `stop_sequences`, `stream`, `metadata`, `service_tier` and `inference_geo`, which the count
answers with HTTP 400 "Extra inputs are not permitted" and which change no input token; system prompt,
tools, tool choice, thinking, effort, output format, fast mode and `cache_control` are counted as sent.
An endpoint with `betas` counts on the beta route, as Anthropic's SDK does: its betas and
`token-counting-2024-11-01` in the `anthropic-beta` header; one without sends none. Anthropic calls the
count "an estimate" that "might include tokens added automatically by Anthropic for system
optimizations", which are not billed, and token counting "is free to use", within its own requests-per-
minute limits
([Token counting](https://platform.claude.com/docs/en/build-with-claude/token-counting)):

```julia
chat = Chat(service=ANTHROPICServiceEndpoint, model="claude-haiku-5-5", messages=[Message(Val(:user), "Hello")])
r = count_tokens(chat)
r isa TokenCountSuccess && r.input_tokens
# Counted on 16 November 2026: 11
```

The result is a `TokenCountSuccess`, a `TokenCountFailure` for an error status, or a `TokenCountCallError`
([Token Counting](@ref token_counting_api)).

## [Files](@id anthropic_files)

The endpoint's `files` capability: the [Files API](@ref files_api) verbs reach Anthropic's Files API
(`/v1/files`), with the key, `anthropic-version` and the endpoint's `workspace_id`, and neither its `betas`
nor its `user_profile_id`, which the Files reference does not name. An upload takes no purpose, so
`upload_file(path; service=ANTHROPICServiceEndpoint)`, and an `expires_after` from 3600 to 7776000 seconds
(Anthropic's `expires_in_seconds`; outside it, `ArgumentError` before the request); without one the file
does not expire. The [`FileObject`](@ref) holds the `id`, `size_bytes` as `bytes`, `created_at` in Unix
seconds and `filename`; `mime_type`, `downloadable` and `expires_at` are in `raw`. [`list_files`](@ref)
takes `limit`, and `after`, the `next_page` cursor in the page before's `raw`; OpenAI's `purpose` and
`order` are refused. [`delete_file`](@ref) reports the id Anthropic names. [`file_content`](@ref) returns the
bytes of a file a tool made, whose `downloadable` is `true`; for an uploaded file Anthropic answers HTTP 400,
returned as a `FileFailure`. "Files API operations are free", and "File content used in Messages requests
is priced as input tokens" ([Files API](https://platform.claude.com/docs/en/build-with-claude/files)); a
message cannot refer to an uploaded file through UniLM yet.

```julia
up = upload_file("notes.txt"; service=ANTHROPICServiceEndpoint, expires_after=3600)
up.response.id, up.response.bytes, up.response.raw["downloadable"]
# ("file_01YLNcm6NoeziqXvw8FX4zaY", 5, false)   — a 5-byte file, 16 November 2026
file_content(up.response.id; service=ANTHROPICServiceEndpoint).status
# 400: "File `file_01YLNcm6NoeziqXvw8FX4zaY` is not downloadable. Only files generated by a tool …"
delete_file(up.response.id; service=ANTHROPICServiceEndpoint)
# FileDeleteSuccess("file_01YLNcm6NoeziqXvw8FX4zaY", true, "file_deleted")
```

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
