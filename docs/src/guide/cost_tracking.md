# [Cost Tracking](@id cost_guide)

```@setup cost_tracking
using UniLM
```

Every result of a token-billed call carries the token counts its provider reported, and UniLM turns those
counts into a USD estimate. Each provider's usage is read into one shape, [`TokenUsage`](@ref), when its
reply is decoded, so the same code totals a bill on every provider; how a provider bills, and the prices
UniLM holds for it, are on its entry.

Two accessors do the work: [`token_usage`](@ref) (raw counts) and [`estimated_cost`](@ref)
(dollars). [`Chat`](@ref) additionally keeps a running total via [`cumulative_cost`](@ref).

```@example cost_tracking
service, model = OPENAIServiceEndpoint, "gpt-5.4-mini"
own_model = "my-model"
nothing # hide
```

The examples use only the names bound above. `service` and `model` are the provider and model they run
on, and `own_model` names a model the price table does not hold, such as one of your own server: bind
your own there. On an endpoint that treats its models as local, the estimates below are `0.0`
([Local models](@ref cost_tracking_local)). The table [at the end](@ref cost_tracking_providers) links each
provider's rules.

!!! warning "The zero-dollar footgun"
    [`estimated_cost`](@ref) returns `0.0` — with no error, only a warning logged once per
    model id — for any model that is not a key in [`DEFAULT_PRICING`](@ref) and has no
    fallback row, such as a model of your own server or one newer than the table. Cost
    tracking on it reports `\$0.00` until you supply its price. See
    [Unpriced models cost \$0](@ref unpriced-zero) below. A model its endpoint treats as local is
    the exception: it costs `0.0` with no warning, whatever `pricing=` or the table holds
    ([Local models](@ref cost_tracking_local)).

## [Reading usage and cost](@id cost_tracking_reading)

[`token_usage`](@ref) returns a [`TokenUsage`](@ref) for every result of a token-billed call —
`chatrequest!`, `respond`, `embeddingrequest!`, the image verbs, `fim_complete`, `prefix_complete` and
`ask` — zero-filled for their failures and call errors; [`estimated_cost`](@ref) returns a `Float64` in
USD. Both throw `ArgumentError` for results of APIs that report no token usage (files, batches, audio,
moderations, vector stores, …), where a zero would be indistinguishable from a free call.
A real result comes back from `chatrequest!` / `respond` / `embeddingrequest!`, but the
accessors are pure, so we build one here to keep the example deterministic and offline:

```@example cost_tracking
# Normally returned by chatrequest!; constructed here so the example needs no API call.
result = LLMSuccess(
    message = Message(Val(:user), "(assistant reply placeholder)"),
    self    = Chat(service=service, model=model),
    usage   = TokenUsage(prompt_tokens=1200, completion_tokens=350, cached_tokens=800),
)

u = token_usage(result)
println("prompt=$(u.prompt_tokens)  cached=$(u.cached_tokens)  completion=$(u.completion_tokens)")
println("cost USD = ", round(estimated_cost(result), digits=6))
```

By default the model priced is the one the result reports having served the call — `result.model` of a
`Chat` result ([`LLMSuccess`](@ref)), where UniLM keeps the name the reply gives, and
`result.response.model` of a `respond`, FIM or `ask` result — else the one requested: `result.self.model`
for a `Chat`, `result.embeddings.model` for `Embeddings`. The model served can differ from the one
requested: a provider that answers an alias with the dated snapshot behind it names the snapshot, which is
priced at the alias's row. Each entry's first call says which model its results report. Override it
explicitly when needed:

```julia
estimated_cost(result; model=model)   # priced as `model`, whatever served it
```

The decision providers' results — what `ask` returns — are priced as each one's entry says: the second
table [at the end](@ref cost_tracking_providers) links their cost sections.

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
provider reports in [`TokenUsage`](@ref) — they apply to that provider's results, as its entry's cost
section says; every other result is billed at the standard rates.

A price row holds no audio rates. A result whose usage counts audio tokens — `prompt_audio_tokens` and
`completion_audio_tokens`, the audio parts of `prompt_tokens` and `completion_tokens` — is priced only where
UniLM prices that model's audio, as its entry's cost section says; any other such result is estimated at
`0.0`, with one warning per model and endpoint, as for a model without a row, rather than its audio
billed at text rates.

Reasoning tokens are already counted inside `completion_tokens`, so they carry no separate
charge. Failures and call errors report zero usage and `0.0` cost. Image results report the
usage their reply gives, but image tokens bill at rates a price row does not model:
`estimated_cost` on an image result throws `ArgumentError` unless you pass `model=`, and a `model=`
without a row is estimated at `0.0`, with the no-row warning, so pass a `pricing=` row for it too.

A streamed `Chat` turn is priced like an unstreamed one where its provider reports the usage in the
stream — on its own, or because the endpoint asks for it with `stream_options = {"include_usage": true}`
when the `Chat` sets no `stream_options`; each entry says which, in its streaming section or the section of
the endpoint it concerns. Elsewhere, set
`stream_options=Dict("include_usage" => true)` where the server supports it, or the streamed turn reports
no usage and accrues `\$0`. An embeddings reply that reports no usage is estimated at `0.0`, with one
warning per model.

## [The pricing table](@id cost_tracking_table)

[`DEFAULT_PRICING`](@ref) is a lock-guarded `AbstractDict{String, PriceRow}` (not a `Dict`),
where a `PriceRow` is a `NamedTuple` of **per-token** USD rates: `(input, cached_input,
output)`. (Provider list prices are usually quoted per 1M tokens; the stored rows are those
figures divided by 1,000,000.) Every Chat success prices itself against it — streaming ones
on their own task — so it is safe to read and write from any task: add or replace a row with
`DEFAULT_PRICING[model] = row`, and use `get`, `haskey`, `delete!`, `pop!`, `keys` and
`length` as on a `Dict`. Iteration walks a snapshot, and `copy`, `merge` and `filter` return
a plain `Dict`.

A model id is looked up exactly first, then without a dated snapshot suffix — `-YYYY-MM-DD` or
`-YYYYMMDD` — so a snapshot is priced at its model's row. Only those date suffixes are stripped, so
custom names stay unpriced rather than matching a row by accident. A provider's own fallback, where it
has one, is on its entry.

```@example cost_tracking
priced = sort(collect(keys(DEFAULT_PRICING)))
println(length(priced), " models priced")
println(model, " row: ", DEFAULT_PRICING[model])
```

## [Unpriced models cost \$0](@id unpriced-zero)

If the model key is absent from the pricing table, `estimated_cost` returns `0.0` — it does
**not** raise; it logs one warning per model id. (The dated-snapshot fallback above applies
first.) This bites a model of your own server that its endpoint does not treat as local, or one newer
than the table:

```@example cost_tracking
println(own_model, " priced? ", haskey(DEFAULT_PRICING, own_model))

unpriced = LLMSuccess(
    message = Message(Val(:user), "…"),
    self    = Chat(service=service, model=own_model),
    usage   = TokenUsage(prompt_tokens=100_000, completion_tokens=50_000),
)
using Logging # hide
no_source(level, _...) = (Logging.default_logcolor(level), (level == Logging.Warn ? "Warning" : string(level)) * ":", "") # hide
with_logger(ConsoleLogger(stderr; meta_formatter=no_source)) do # hide
println("estimated_cost with default pricing = \$", estimated_cost(unpriced))
end # hide
```

150,000 tokens, and the estimate is still `\$0.0`. Always confirm your model is a key in the
pricing table before trusting a cost number.

## [Local models](@id cost_tracking_local)

A model its endpoint treats as local costs `0.0`, with no warning, as that endpoint's entry says
([a local model's cost](@ref ollama_cost)): `estimated_cost` returns before it reads a price row, so
neither `pricing=` nor [`DEFAULT_PRICING`](@ref) changes the estimate, and `cumulative_cost` adds `0.0` for
each turn. Only `model=` prices such a result, at the row of the model it names.

## [Supplying custom pricing](@id cost_tracking_custom)

Pass a `pricing=` dict (any `AbstractDict{String, PriceRow}`) to fix an unpriced (or
mispriced) model. Build a `PriceRow` NamedTuple of per-token rates and `merge` it into the
defaults — `merge` returns a new `Dict` and leaves [`DEFAULT_PRICING`](@ref) untouched:

```@example cost_tracking
# Fill in the provider's *current* list price (per 1M tokens); divide by 1_000_000.
pricing = merge(DEFAULT_PRICING, Dict(
    own_model => (input        = 2.00 / 1_000_000,
                  cached_input = 0.50 / 1_000_000,
                  output       = 8.00 / 1_000_000),
))

println("now priced? ", haskey(pricing, own_model))
println("cost USD = ", round(estimated_cost(unpriced; pricing), digits=6))
```

A row cannot price audio: a `PriceRow` has no audio-token rates. Whatever `pricing=` holds, a
result whose usage counts audio tokens stays at `0.0` unless UniLM prices that model's audio, as its
entry's cost section says.

## [Cumulative cost is Chat-only](@id cost_tracking_cumulative)

[`cumulative_cost`](@ref) returns the running USD total that `chatrequest!` accrues on a
[`Chat`](@ref). Accumulation happens **only** inside `chatrequest!` (both streaming and
non-streaming paths, and so every turn of a `tool_loop!`) — nothing else adds to the total:

```julia
chat = Chat(service=service, model=model)
push!(chat, Message(Val(:system), "You are helpful."))
push!(chat, Message(Val(:user), "Hello!"))
chatrequest!(chat)                     # cost accrues automatically

push!(chat, Message(Val(:user), "And again, in French."))
chatrequest!(chat)                     # accrues again

cumulative_cost(chat)                  # e.g. 0.00231  (USD, summed over both calls)
```

A call that ends without committing a turn adds what its provider reported billing it, as a lower bound. A
stream stopped, cancelled, timed out or cut off, or ended by an exception from its `callback` or
`on_tool_call`, adds the usage the provider had reported by then; where an entry knows when its provider
reports usage, its streaming section says so. A request stopped before any usage arrived adds nothing, though
the provider bills it; so do an attempt UniLM retried, a stream the provider ended with an error event, and an
unstreamed reply UniLM could not decode.

!!! warning "Auto-accumulation always uses the default pricing table"
    `chatrequest!` prices each call with [`DEFAULT_PRICING`](@ref); there is no `pricing=`
    for the running total. So a `Chat` on an **unpriced** model reports
    `cumulative_cost == 0.0` even after successful calls — unless you add its row to the
    table (`DEFAULT_PRICING[own_model] = row`, safe while requests run), after which
    later calls accrue — but for a model its endpoint treats as local, which accrues `0.0`
    ([Local models](@ref cost_tracking_local)). Alternatively, ignore the running total and sum
    [`estimated_cost`](@ref)`(result; pricing=…)` yourself.

## [Totaling respond, tool-loop and embeddings costs](@id cost_tracking_totals)

[`respond`](@ref) ([`ResponseSuccess`](@ref)), the `respond` tool loops, and
[`Embeddings`](@ref embeddings_api) are **not** auto-accumulated. Sum [`estimated_cost`](@ref) over the
results yourself:

```julia
# respond — one estimate per call, add them up.
r1 = respond("Summarize the Julia language in one line."; service, model)
r2 = respond("Now in French."; service, model)
responses_total = estimated_cost(r1) + estimated_cost(r2)   # USD

# Embeddings — price the batch result directly (the provider's default embedding model).
emb = Embeddings(["alpha", "beta", "gamma"]; service)
res = embeddingrequest!(emb)
embeddings_cost = estimated_cost(res)                       # USD for this batch
```

A `respond` tool loop sends a request per turn and returns its last response alone, as `loop.response`
([Agentic respond](@ref agentic_guide)): `estimated_cost(loop.response)` prices that last request, not the
turns before it. A `tool_loop!` on a `Chat` adds every turn to the `Chat`'s `cumulative_cost`. Embedding
models missing from the default table (one on your own server that its endpoint does not treat as local,
for example) hit the same `0.0` behavior — supply `pricing=` for them too.

## [Prices drift](@id cost_tracking_drift)

[`DEFAULT_PRICING`](@ref) is a hardcoded snapshot of the providers' list prices, and each entry with a price
table says in its cost section when its rows were checked. Provider list prices change; re-verify against the
provider's current pricing before relying on any number, and pass your own `pricing=` dict when you need
authoritative figures.

## [See also](@id cost_tracking_see_also)

- [Cost Tracking & Token Usage](@ref accounting_api) — full type and function docs
- [`token_usage`](@ref) — raw [`TokenUsage`](@ref) counts from any token-billed result
- [`estimated_cost`](@ref) — per-result USD estimate, with optional `pricing=`
- [`cumulative_cost`](@ref) — running USD total on a [`Chat`](@ref)
- [`DEFAULT_PRICING`](@ref) — the built-in per-token pricing table

## [On your provider](@id cost_tracking_providers)

```@eval
Main.ProviderFacts.on_your_provider(Main.ProviderFacts.llm, :cost)
```

The decision providers' cost sections, for what `ask` returns:

```@eval
Main.ProviderFacts.on_your_provider(Main.ProviderFacts.decisions, :cost)
```
