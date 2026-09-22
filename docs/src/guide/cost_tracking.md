# [Cost Tracking](@id cost_guide)

Every UniLM.jl result carries the provider's token counts, and the package can turn those
counts into a USD estimate. This works uniformly across Chat Completions, the Responses API,
the agentic verb, and Embeddings — Gemini and Anthropic usage is normalized to the shared
shape at decode time.

Two accessors do the work: [`token_usage`](@ref) (raw counts) and [`estimated_cost`](@ref)
(dollars). [`Chat`](@ref) additionally keeps a running total via [`cumulative_cost`](@ref).

!!! warning "The silent \$0 footgun"
    [`estimated_cost`](@ref) returns `0.0` — with no error and no warning — for any model
    that is not a key in [`DEFAULT_PRICING`](@ref). The tutorial model `gpt-4o-mini` is **not**
    priced, so cost tracking on it reports `\$0.00` until you supply your own pricing. See
    [Unpriced models return \$0 silently](@ref unpriced-zero) below.

## Reading usage and cost

[`token_usage`](@ref) returns a [`TokenUsage`](@ref) for any result (zero-filled for failures);
[`estimated_cost`](@ref) returns a `Float64` in USD. A real result comes back from
`chatrequest!` / `respond` / `embeddingrequest!`, but the accessors are pure, so we build one
here to keep the example deterministic and offline:

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

By default the model is inferred from the result (`result.self.model` for Chat,
`result.response.model` for Responses, `result.embeddings.model` for Embeddings). Override it
explicitly when needed:

```julia
estimated_cost(result; model="gpt-5.4")   # price as if it were gpt-5.4
```

## How the estimate is computed

`estimated_cost` splits the prompt into fresh vs. cached tokens and bills each stream at its
own rate:

- fresh input tokens × `input`
- cached input tokens × `cached_input` (the discounted prompt-cache rate; `cached_tokens` is a subset of `prompt_tokens`)
- completion tokens × `output`

Reasoning tokens are already counted inside `completion_tokens`, so they carry no separate
charge. Failures, call errors, and image results all report zero usage and `0.0` cost.

## The pricing table

[`DEFAULT_PRICING`](@ref) is a `Dict{String, PriceRow}`, where a `PriceRow` is a `NamedTuple`
of **per-token** USD rates: `(input, cached_input, output)`. (Provider list prices are usually
quoted per 1M tokens; the stored rows are those figures divided by 1,000,000.)

```@example cost
priced = sort(collect(keys(DEFAULT_PRICING)))
println(length(priced), " models priced")
println("gpt-5.2 row: ", DEFAULT_PRICING["gpt-5.2"])
```

## [Unpriced models return \$0 silently](@id unpriced-zero)

If the model key is absent from the pricing table, `estimated_cost` returns `0.0` — it does
**not** raise. This bites the tutorial model `gpt-4o-mini`:

```@example cost
println("gpt-4o-mini priced? ", haskey(DEFAULT_PRICING, "gpt-4o-mini"))

unpriced = LLMSuccess(
    message = Message(Val(:user), "…"),
    self    = Chat(model="gpt-4o-mini"),
    usage   = TokenUsage(prompt_tokens=100_000, completion_tokens=50_000),
)
println("estimated_cost with default pricing = \$", estimated_cost(unpriced))
```

150,000 tokens, and the estimate is still `\$0.0`. Always confirm your model is a key in the
pricing table before trusting a cost number.

## Supplying custom pricing

Pass a `pricing=` dict to fix an unpriced (or mispriced) model. Build a `PriceRow` NamedTuple
of per-token rates and `merge` it into the defaults — `merge` returns a new dict and leaves
[`DEFAULT_PRICING`](@ref) untouched:

```@example cost
# Fill in the provider's *current* list price (per 1M tokens); divide by 1_000_000.
pricing = merge(DEFAULT_PRICING, Dict(
    "gpt-4o-mini" => (input        = 0.15 / 1_000_000,
                      cached_input  = 0.075 / 1_000_000,
                      output        = 0.60 / 1_000_000),
))

println("now priced? ", haskey(pricing, "gpt-4o-mini"))
println("cost USD = ", round(estimated_cost(unpriced; pricing), digits=6))
```

## Cumulative cost is Chat-only

[`cumulative_cost`](@ref) returns the running USD total that `chatrequest!` accrues on a
[`Chat`](@ref). Accumulation happens **only** inside `chatrequest!` (both streaming and
non-streaming paths) — nothing else adds to the total:

```julia
chat = Chat(model="gpt-5.2")
push!(chat, Message(Val(:system), "You are helpful."))
push!(chat, Message(Val(:user), "Hello!"))
chatrequest!(chat)                     # cost accrues automatically

push!(chat, Message(Val(:user), "And again, in French."))
chatrequest!(chat)                     # accrues again

cumulative_cost(chat)                  # e.g. 0.00231  (USD, summed over both calls)
```

!!! warning "Auto-accumulation always uses `DEFAULT_PRICING`"
    There is no way to inject a custom `pricing=` into the running total — `chatrequest!`
    prices each call with the defaults. So a `Chat` on an **unpriced** model (e.g.
    `gpt-4o-mini`) reports `cumulative_cost == 0.0` even after successful calls. On unpriced
    models, ignore the running total and sum [`estimated_cost`](@ref)`(result; pricing=…)`
    yourself.

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

Note that non-OpenAI embedding models (e.g. Ollama's `nomic-embed-text`,
`gemini-embedding-001`) are not in the default table and hit the same silent-`\$0` behavior —
supply `pricing=` for them too.

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
named. A request sent as `jev-latest` therefore looks up `jev-1.13.0`, so log
`result.response.model` alongside the cost if you need to explain a bill later. As
everywhere else, an unpriced name returns `0.0` silently:

```julia
r = ask(ticket, "urgency" => score("How urgent is this ticket?", levels))
if r isa SystemOneSuccess
    u = token_usage(r)
    println(u.prompt_tokens, " billable input tokens on ", r.response.model)
    println("USD ", estimated_cost(r))
end
```

Batching matters more here than the per-token rate: every question in one
[`ask`](@ref) shares a single ingestion of the state, so N questions in one call bill
far fewer input tokens than N calls. See [Typed Judgments with Jev](@ref system_one_guide).

## Prices drift

[`DEFAULT_PRICING`](@ref) is a hardcoded snapshot (OpenAI verified 2026-06-21, Anthropic
2026-07-06, Gemini 2026-07-07). Provider list prices change; re-verify against the provider's
current pricing before relying on any number, and pass your own `pricing=` dict when you need
authoritative figures.

## API Reference

See [Cost Tracking & Token Usage](@ref accounting_api) for full type and function docs.

## See Also

- [`token_usage`](@ref) — raw [`TokenUsage`](@ref) counts from any result
- [`estimated_cost`](@ref) — per-result USD estimate, with optional `pricing=`
- [`cumulative_cost`](@ref) — running USD total on a [`Chat`](@ref)
- [`DEFAULT_PRICING`](@ref) — the built-in per-token pricing table
