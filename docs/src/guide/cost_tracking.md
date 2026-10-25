# [Cost Tracking](@id cost_guide)

Every result of a token-billed API carries the provider's token counts, and the package can
turn those counts into a USD estimate. This works uniformly across Chat Completions, the
Responses API, the agentic verb, Embeddings, FIM completions and System One — Gemini,
Anthropic and DeepSeek usage is normalized to the shared shape at decode time.

Two accessors do the work: [`token_usage`](@ref) (raw counts) and [`estimated_cost`](@ref)
(dollars). [`Chat`](@ref) additionally keeps a running total via [`cumulative_cost`](@ref).

!!! warning "The zero-dollar footgun"
    [`estimated_cost`](@ref) returns `0.0` — with no error, only a warning logged once per
    model id — for any model that is not a key in [`DEFAULT_PRICING`](@ref) and has no
    fallback row (a dated snapshot uses its base model's row; a versioned `jev-X.Y.Z` id
    uses the `jev-latest` row). A current OpenAI model such as `gpt-5.2-pro` is **not**
    priced, so cost tracking on it reports `\$0.00` until you supply your own pricing. See
    [Unpriced models cost \$0](@ref unpriced-zero) below.

## Reading usage and cost

[`token_usage`](@ref) returns a [`TokenUsage`](@ref) for every result of a token-billed API —
chat, Responses, embeddings, image generation, FIM and System One — zero-filled for their
failures and call errors; [`estimated_cost`](@ref) returns a `Float64` in USD. Both throw
`ArgumentError` for results of APIs that report no token usage (files, batches, audio,
moderations, vector stores, …), where a zero would be indistinguishable from a free call.
A real result comes back from `chatrequest!` / `respond` / `embeddingrequest!`, but the
accessors are pure, so we build one here to keep the example deterministic and offline:

```@example cost
using UniLM

# Normally returned by chatrequest!; constructed here so the example needs no API call.
result = LLMSuccess(
    message = Message(Val(:user), "(assistant reply placeholder)"),
    self    = Chat(model="gpt-5.2"),
    usage   = TokenUsage(prompt_tokens=1200, completion_tokens=350, cached_tokens=800),
)

u = token_usage(result)
println("prompt=$(u.prompt_tokens)  cached=$(u.cached_tokens)  completion=$(u.completion_tokens)")
println("cost USD = ", round(estimated_cost(result), digits=6))
```

By default the model is the one the provider reports having served the call —
`result.model` for Chat ([`LLMSuccess`](@ref); it differs from the model requested when the
provider routes the request to another model, or names the dated snapshot behind an alias,
which is priced at the alias's row), `result.response.model` for Responses, FIM
and System One — else the one requested: `result.self.model` for Chat,
`result.embeddings.model` for Embeddings. Override it explicitly when needed:

```julia
estimated_cost(result; model="gpt-5.4")   # price as if it were gpt-5.4
```

## [How the estimate is computed](@id cost_estimate)

`estimated_cost` splits the prompt into fresh, cache-read and cache-written tokens and bills
each at its own rate:

- fresh input tokens × `input`
- cache reads (`cached_tokens`) × `cached_input`, the discounted prompt-cache rate
- cache writes (`cache_write_tokens`) × `input`
- completion tokens × `output`

`prompt_tokens` counts every input token: `cached_tokens` and `cache_write_tokens` are
subsets of it. These are the standard rates. Where UniLM models a provider's surcharges — a
premium on cache writes (`cache_write_extended_tokens` counts the writes at the longer
retention), a long-context tier, a multiplier for the `service_tier` or `inference_geo` the
provider reports in [`TokenUsage`](@ref) — they apply to that provider's results; every
other result is billed at the standard rates. `GEMINIOpenAIServiceEndpoint`'s replies name no
billed tier, so a Chat result there records the tier the request asked for: `"flex"` is
priced at the Flex rates and `"priority"` at the Priority rates — an upper bound, since
Google bills a downgraded Priority request at Standard and this endpoint does not report the
downgrade.

OpenAI bills every audio token of a Chat call — an [`AudioAttachment`](@ref) clip, an audio
reply ([Audio replies](@ref chat_audio_output)), a reply sent back by its id — at the model's
audio-token rates: `prompt_audio_tokens` and `completion_audio_tokens`, the audio parts of
`prompt_tokens` and `completion_tokens`, bill at those rates, and the rest of the call at the
row's text rates. UniLM has the audio-token rates of `gpt-audio-1.5` on
`OPENAIServiceEndpoint`, and a price row has none, so audio tokens are not priced for any
other model on `OPENAIServiceEndpoint`, nor for any model on an endpoint billed at the standard
rates (Azure, DeepSeek, generic endpoints, your own): such a result is estimated at `0.0`,
with one warning per model and endpoint, as for a model without a row, rather than its audio
billed at text rates.

Gemini bills the audio input of a call — an [`AudioAttachment`](@ref) on native Chat or
[`respond`](@ref) — at the input price its
[pricing page](https://ai.google.dev/gemini-api/docs/pricing) states for audio, scaled by the
billed tier and a long prompt's multiples as the text rates are. The page spells input modalities
out wherever their rates differ, so a model's own audio price applies where it lists one apart —
per 1M tokens, \$0.50 on `gemini-3.1-flash-lite`, \$1.00 on `gemini-3-flash-preview` and
`gemini-2.5-flash`, \$0.30 on `gemini-2.5-flash-lite` — and otherwise the row's input rate, since
an input price that names no modality, or names audio among the others, is every modality's (Gemini
4 Argon, 3.8 and 3.6 Flash, 3.5 Flash-Lite, 3.1 Pro Preview, 2.5 Pro, Robotics-ER 2 Preview). Their
cache prices name no modality either, so a cache hit beside audio bills at the row's cached rate.
Audio on a model the page prices without audio (the image models) or does not list is estimated at
`0.0`, with one warning per model and endpoint, and so is a call that counts audio and cache hits
on one of the four models with an audio price of their own: its usage does not say how many cached
tokens are audio, whose cache price the page lists apart.
`GEMINIOpenAIServiceEndpoint` reports no audio count — a clip's tokens are in `prompt_tokens`
alone — so its audio input is priced at the text rate: a lower bound on the four models with an
audio price of their own, for which such a request warns once per model. Send audio through
`GEMINIServiceEndpoint` for an exact estimate.

Reasoning tokens are already counted inside `completion_tokens`, so they carry no separate
charge. Failures and call errors report zero usage and `0.0` cost. Image results report the
usage the Images API returned, but that API bills image tokens at rates a price row does not
model: `estimated_cost` on an image result throws `ArgumentError` unless you pass `model=`,
and a `model=` without a row is estimated at `0.0`, with the no-row warning, so pass a
`pricing=` row for it too. A streamed Chat turn is costed like a non-streamed one when the stream
reports usage: `OPENAIServiceEndpoint`, `DeepSeekEndpoint` and `GEMINIOpenAIServiceEndpoint`
streams request it automatically, native Anthropic and Gemini streams report it on their own, and
other OpenAI-compatible servers need `stream_options=Dict("include_usage" => true)` (without it a
streamed turn reports no usage and accrues `\$0`). An embeddings reply that reports no usage —
`GEMINIOpenAIServiceEndpoint`'s never does — is estimated at `0.0`, with one warning per model.

## The pricing table

[`DEFAULT_PRICING`](@ref) is a lock-guarded `AbstractDict{String, PriceRow}` (not a `Dict`),
where a `PriceRow` is a `NamedTuple` of **per-token** USD rates: `(input, cached_input,
output)`. (Provider list prices are usually quoted per 1M tokens; the stored rows are those
figures divided by 1,000,000.) Every Chat success prices itself against it — streaming ones
on their own task — so it is safe to read and write from any task: add or replace a row with
`DEFAULT_PRICING[model] = row`, and use `get`, `haskey`, `delete!`, `pop!`, `keys` and
`length` as on a `Dict`. Iteration walks a snapshot, and `copy`, `merge` and `filter` return
a plain `Dict`.

A model id is looked up exactly first, then without a dated snapshot suffix
(`gpt-5.4-mini-2026-03-17` → `gpt-5.4-mini`, `claude-opus-4-5-20251101` →
`claude-opus-4-5`), and a versioned Jev id with no row of its own (`jev-X.Y.Z`) falls back
to the `jev-latest` row. Only those date suffixes are stripped, so custom names stay
unpriced rather than matching a row by accident.

```@example cost
priced = sort(collect(keys(DEFAULT_PRICING)))
println(length(priced), " models priced")
println("gpt-5.2 row: ", DEFAULT_PRICING["gpt-5.2"])
```

## [Unpriced models cost \$0](@id unpriced-zero)

If the model key is absent from the pricing table, `estimated_cost` returns `0.0` — it does
**not** raise; it logs one warning per model id. (Two fallbacks apply first: a dated snapshot
uses its base model's row, and a versioned `jev-X.Y.Z` id uses the `jev-latest` row.) This
bites `gpt-5.2-pro`, a current OpenAI model the table does not list:

```@example cost
println("gpt-5.2-pro priced? ", haskey(DEFAULT_PRICING, "gpt-5.2-pro"))

unpriced = LLMSuccess(
    message = Message(Val(:user), "…"),
    self    = Chat(model="gpt-5.2-pro"),
    usage   = TokenUsage(prompt_tokens=100_000, completion_tokens=50_000),
)
println("estimated_cost with default pricing = \$", estimated_cost(unpriced))
```

150,000 tokens, and the estimate is still `\$0.0`. Always confirm your model is a key in the
pricing table before trusting a cost number.

## Supplying custom pricing

Pass a `pricing=` dict (any `AbstractDict{String, PriceRow}`) to fix an unpriced (or
mispriced) model. Build a `PriceRow` NamedTuple of per-token rates and `merge` it into the
defaults — `merge` returns a new `Dict` and leaves [`DEFAULT_PRICING`](@ref) untouched:

```@example cost
# Fill in the provider's *current* list price (per 1M tokens); divide by 1_000_000.
# gpt-5.2-pro lists no cached-input rate, so a cache read is billed as input.
pricing = merge(DEFAULT_PRICING, Dict(
    "gpt-5.2-pro" => (input        = 21.00 / 1_000_000,
                      cached_input  = 21.00 / 1_000_000,
                      output        = 168.00 / 1_000_000),
))

println("now priced? ", haskey(pricing, "gpt-5.2-pro"))
println("cost USD = ", round(estimated_cost(unpriced; pricing), digits=6))
```

A row cannot price audio: a `PriceRow` has no audio-token rates. Whatever `pricing=` holds, a
result whose usage counts audio tokens stays at `0.0` unless UniLM has that model's audio-token
rates (`gpt-audio-1.5` on `OPENAIServiceEndpoint`, and the Gemini models above, whose text
tokens your row then prices; on a Gemini model whose one input price covers audio, your row's
input rate prices its audio too).

## Cumulative cost is Chat-only

[`cumulative_cost`](@ref) returns the running USD total that `chatrequest!` accrues on a
[`Chat`](@ref). Accumulation happens **only** inside `chatrequest!` (both streaming and
non-streaming paths, and so every turn of a `tool_loop!`) — nothing else adds to the total:

```julia
chat = Chat(model="gpt-5.2")
push!(chat, Message(Val(:system), "You are helpful."))
push!(chat, Message(Val(:user), "Hello!"))
chatrequest!(chat)                     # cost accrues automatically

push!(chat, Message(Val(:user), "And again, in French."))
chatrequest!(chat)                     # accrues again

cumulative_cost(chat)                  # e.g. 0.00231  (USD, summed over both calls)
```

A call that ends without committing a turn adds what its provider reported billing it, as a
lower bound. A stream stopped, cancelled, timed out or cut off, or ended by an exception from its
`callback` or `on_tool_call`, adds the usage the provider had reported by then: Anthropic reports
a request's input when its stream starts, and Gemini the usage so far on every chunk, while
OpenAI reports it after the finish reason, so a stopped OpenAI stream adds nothing. A continued
Gemini turn that fails or is stopped adds the replies it had received. A request stopped before
any usage arrived adds nothing, though the provider bills it; so do an attempt UniLM retried,
a stream the provider ended with an error event, and an unstreamed reply UniLM could not decode.

!!! warning "Auto-accumulation always uses the default pricing table"
    `chatrequest!` prices each call with [`DEFAULT_PRICING`](@ref); there is no `pricing=`
    for the running total. So a `Chat` on an **unpriced** model (e.g. `gpt-5.2-pro`) reports
    `cumulative_cost == 0.0` even after successful calls — unless you add its row to the
    table (`DEFAULT_PRICING["gpt-5.2-pro"] = row`, safe while requests run), after which
    later calls accrue. Alternatively, ignore the running total and sum
    [`estimated_cost`](@ref)`(result; pricing=…)` yourself.

## Totaling Responses, agentic, and Embeddings costs

The Responses API ([`respond`](@ref) / [`ResponseSuccess`](@ref)), the agentic tool loops, and
[`Embeddings`](@ref embeddings_api) are **not** auto-accumulated. Sum [`estimated_cost`](@ref) over the
results yourself:

```julia
# Responses API — one estimate per call, add them up.
r1 = respond("Summarize the Julia language in one line.")
r2 = respond("Now in French.")
responses_total = estimated_cost(r1) + estimated_cost(r2)   # USD

# Agentic loops issue several underlying calls; each success is a result you can price.
# Collect them (e.g. via a callback) and sum estimated_cost over the trace.
# See the Agentic Workflows guide.

# Embeddings — price the batch result directly.
emb = Embeddings(["alpha", "beta", "gamma"])
res = embeddingrequest!(emb)
embeddings_cost = estimated_cost(res)                       # USD for this batch
```

Note that embedding models missing from the default table (a Mistral model, or one on your
own server, for example) hit the same `0.0` behavior — supply `pricing=` for them too. Chat, embeddings,
`respond` and FIM results from a local model on an [`OllamaEndpoint`](@ref) cost
`0.0` without a warning; pass `model=` to price one as if it had run elsewhere. A
cloud model reached through Ollama — a tag of `cloud`, or one ending in `-cloud`,
such as `gpt-oss:120b-cloud` — runs on Ollama's servers and is priced by the
requested name, like any other model, never as a free local call: its `respond` and
FIM results too, whose reply may drop the tag ([Local Models with
Ollama](@ref ollama_first) says how a cloud model is recognised, and where that
fails).

## System One (TypeSafe Jev) is input-only

A [`SystemOneSuccess`](@ref) reports usage through the same [`TokenUsage`](@ref) shape:
`prompt_tokens` is the service's `input_tokens`, `completion_tokens` its `output_tokens`,
and `total_tokens` their sum. Only **input** tokens are billed — output tokens are
currently free, which the shipped `jev-*` rows encode as an output rate of `0.0`:

```@example cost
println("jev-1.13.0 row: ", DEFAULT_PRICING["jev-1.13.0"])
```

[`estimated_cost`](@ref)`(::SystemOneSuccess)` prices the **versioned** model id in
`result.response.model` — the version that actually answered — not the alias the request
named. A request sent as `jev-latest` therefore looks up the version behind the alias
at that moment (for example `jev-1.13.0`), so log
`result.response.model` alongside the cost if you need to explain a bill later. A
versioned `jev-X.Y.Z` id without its own row is priced at the `jev-latest` row; as
everywhere else, any other unpriced name returns `0.0` (with its one-time warning):

```@example cost
ticket = "Help! My payouts have been failing for 3 days."
levels = ["Can wait", "Needs attention this week", "Needs attention today"]

r = ask(ticket, "urgency" => score("How urgent is this ticket?", levels))
if r isa SystemOneSuccess
    println(answer(r, "urgency"))
    u = token_usage(r)
    println(u.prompt_tokens, " billable input tokens on ", r.response.model)
    println("USD ", estimated_cost(r))
else
    println("Request failed — ", r)
end
```

Batching matters more here than the per-token rate: every question in one
[`ask`](@ref) shares a single ingestion of the state, so N questions in one call bill
far fewer input tokens than N calls. See [Ask many questions at once](@ref jev_many_questions).

## Prices drift

[`DEFAULT_PRICING`](@ref) is a hardcoded snapshot (OpenAI, Anthropic and Gemini rows verified
2026-10-22, OpenAI's `gpt-5-search-api` 2026-10-25; TypeSafe rows 2026-10-25; DeepSeek rows
2026-09-24). The DeepSeek rows are its peak
rates — off-peak hours bill half — and DeepSeek context-cache hits
(`prompt_cache_hit_tokens`) are billed at the cached rate. Provider list prices change;
re-verify against the provider's current pricing before relying on any number, and pass your
own `pricing=` dict when you need authoritative figures.

## API Reference

See [Cost Tracking & Token Usage](@ref accounting_api) for full type and function docs.

## See Also

- [`token_usage`](@ref) — raw [`TokenUsage`](@ref) counts from any token-billed result
- [`estimated_cost`](@ref) — per-result USD estimate, with optional `pricing=`
- [`cumulative_cost`](@ref) — running USD total on a [`Chat`](@ref)
- [`DEFAULT_PRICING`](@ref) — the built-in per-token pricing table
