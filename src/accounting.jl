"""
    _PricingTable <: AbstractDict{String,PriceRow}

The table behind [`DEFAULT_PRICING`](@ref): a `Dict` guarded by a lock. Every Chat
success prices itself — streaming ones on their own task — while callers may add or
replace rows, and a `Dict` is not safe to read while another task writes it (an insert
that rehashes moves the entries a reader is walking). Supports `t[k] = row`, `t[k]`, `get`, `haskey`, `delete!`, `pop!`, `empty!`,
`length` and `keys`; iteration walks a snapshot taken under the lock.
"""
struct _PricingTable <: AbstractDict{String,PriceRow}
    rows::Base.Lockable{Dict{String,PriceRow},ReentrantLock}
end
_PricingTable(rows::Dict{String,PriceRow}) = _PricingTable(Base.Lockable(rows))

Base.getindex(t::_PricingTable, k) = @lock t.rows t.rows[][k]
Base.get(t::_PricingTable, k, default) = @lock t.rows get(t.rows[], k, default)
Base.haskey(t::_PricingTable, k) = @lock t.rows haskey(t.rows[], k)
Base.setindex!(t::_PricingTable, v, k) = (@lock t.rows t.rows[][k] = v; t)
Base.delete!(t::_PricingTable, k) = (@lock t.rows delete!(t.rows[], k); t)
Base.pop!(t::_PricingTable, k) = @lock t.rows pop!(t.rows[], k)
Base.pop!(t::_PricingTable, k, default) = @lock t.rows pop!(t.rows[], k, default)
Base.empty!(t::_PricingTable) = (@lock t.rows empty!(t.rows[]); t)
Base.length(t::_PricingTable) = @lock t.rows length(t.rows[])
Base.iterate(t::_PricingTable, (snapshot, i)=(@lock(t.rows, collect(t.rows[])), 1)) =
    i > length(snapshot) ? nothing : (snapshot[i], (snapshot, i + 1))
# The snapshot may hold more or fewer rows than an earlier `length` call reported, so
# `collect` must not preallocate from it.
Base.IteratorSize(::Type{_PricingTable}) = Base.SizeUnknown()

"""Default per-token pricing; OpenAI, Anthropic and Gemini rows verified on 2026-10-22 (OpenAI's
`gpt-5-search-api` on 2026-10-25), TypeSafe rows on 2026-09-22, DeepSeek rows on 2026-09-24 (prices
drift — re-verify before relying on them).
Cached input is billed at the discounted `cached_input` rate; reasoning tokens are already
counted within output tokens. The rows are standard short-context text rates. Where UniLM
models a provider's surcharges — cache-write premiums, long-context tiers, service-tier,
speed or region multipliers — [`estimated_cost`](@ref) applies them to these rates for that
provider's results; every other result is billed at the standard rates. A billed service tier
the provider's pricing page lists no price for — such as OpenAI's Scale tier, Fast on
`gpt-5.4-pro` or on a `gpt-5.5` prompt over 272K tokens, or the Gemini Interactions
`"deferred"` tier — is estimated at the standard rates, with one warning per model and tier.

A row has no audio-token rates. [`estimated_cost`](@ref) bills the audio tokens
(`prompt_audio_tokens`, `completion_audio_tokens`) of an `OPENAIServiceEndpoint` result at the
model's audio-token rates, which UniLM has for `gpt-audio-1.5`, and the audio input of a native
Gemini or Interactions result at the input price Google's pricing page states for audio, scaled
by the billed tier and a long prompt's multiples as text is: the model's own audio price where
the page lists one apart (`gemini-3.1-flash-lite`, `gemini-3-flash-preview`, `gemini-2.5-flash`,
`gemini-2.5-flash-lite`), else the row's input rate, since the page spells modalities out
wherever their rates differ and an input price that names none is every modality's; a cache hit
bills at the row's cached rate, since a one-price model's cache price names no modality either.
Any other audio — on a model the page prices without audio (the image models) or does not list,
on another endpoint, Gemini audio output, or audio in a call that also hit the cache on a model
with its own audio price (a usage does not say how many cached tokens are audio, and the page
lists that model's cached audio apart) — leaves the estimate at `0.0`, with one warning per
model and endpoint.
`GEMINIOpenAIServiceEndpoint` reports no audio count, so its audio input is priced at the text
rate, a lower bound where the model's audio price is its own; such a request warns once per model.
Off-peak discounts, other multimodal rates and hosted-tool fees are not included.

A lock-guarded `AbstractDict{String,PriceRow}`: add or replace rows with
`DEFAULT_PRICING[model] = row` from any task while requests are running."""
const DEFAULT_PRICING = _PricingTable(Dict{String, PriceRow}(
    # OpenAI (https://developers.openai.com/api/docs/pricing, 2026-10-22): the standard rates
    # for prompts of at most 272K input tokens. Cache writes, longer prompts and the billed
    # service tier are priced from these rows by the OpenAI endpoint's `_billed_cost`, which
    # bills an audio model's audio tokens at its audio-token rates.
    "gpt-6-astra"   => _price(10.0, 1.0,  50.0),
    "gpt-6.1-sol"   => _price(2.0,  0.10, 10.0),
    "gpt-6-sol"     => _price(2.0,  0.20, 10.0),
    "gpt-6-luna"    => _price(0.10, 0.01, 0.50),
    "gpt-5.6-sol"   => _price(4.0,  0.40, 20.0),   # promotional, at least through 2026-11-21
    "gpt-5.6-terra" => _price(2.0,  0.20, 12.0),
    "gpt-5.6-luna"  => _price(0.20, 0.02, 1.20),
    "gpt-5.5"       => _price(5.0,  0.50, 30.0),
    "gpt-5.5-pro"   => _price(30.0, 30.0, 180.0),  # no cached-input rate: a cache read bills as input
    "gpt-5.4"       => _price(2.5,  0.25, 15.0),
    "gpt-5.4-pro"   => _price(30.0, 30.0, 180.0),  # no cached-input rate
    "gpt-5.4-mini"  => _price(0.75, 0.075, 4.5),
    "gpt-5.4-nano"  => _price(0.20, 0.02, 1.25),   # shuts down 2027-04-01
    "gpt-5.2"       => _price(1.75, 0.175, 14.0),
    "chat-latest"   => _price(5.0,  0.50, 30.0),
    "gpt-5-search-api" => _price(1.25, 0.125, 10.0),  # the Chat Completions search model; page of 2026-10-25
    "gpt-audio-1.5" => _price(2.5,  2.5,  10.0),   # text tokens; no cached-input rate: a cache read bills as input
    "gpt-4.1"       => _price(2.0,  0.50, 8.0),    # shuts down 2027-01-14
    "gpt-4.1-mini"  => _price(0.4,  0.10, 1.6),    # shuts down 2027-01-14
    "o3"            => _price(2.0,  0.50, 8.0),    # its snapshot o3-2025-04-16 shuts down 2026-12-11
    # Anthropic Claude (https://platform.claude.com/docs/en/about-claude/pricing.md,
    # 2026-10-22). Cache reads are 0.1x input, except 0.025x on Fable 5.1 and Mythos 5.1
    # and 0.05x on Opus 5.5 and Sonnet 5.5; Haiku 5.5's row is its rate for prompts up to
    # 100,000 tokens. Cache writes, fast mode, US-only inference and Haiku 5.5's longer
    # prompts are billed by the Anthropic `_billed_cost`. Dated snapshot ids (-YYYYMMDD)
    # resolve to these rows.
    "claude-fable-5-1"  => _price(10.0, 0.25, 50.0),
    "claude-mythos-5-1" => _price(10.0, 0.25, 50.0),
    "claude-fable-5"    => _price(10.0, 1.0,  50.0),
    "claude-mythos-5"   => _price(10.0, 1.0,  50.0),
    "claude-opus-5-5"   => _price(4.0,  0.20, 20.0),
    "claude-sonnet-5-5" => _price(2.0,  0.10, 10.0),
    "claude-haiku-5-5"  => _price(0.10, 0.01, 0.50),
    "claude-opus-5"     => _price(5.0,  0.50, 25.0),
    "claude-opus-4-8"   => _price(5.0,  0.50, 25.0),
    "claude-opus-4-7"   => _price(5.0,  0.50, 25.0),
    "claude-opus-4-6"   => _price(5.0,  0.50, 25.0),
    "claude-opus-4-5"   => _price(5.0,  0.50, 25.0),
    "claude-sonnet-5"   => _price(2.0,  0.20, 10.0),
    "claude-sonnet-4-6" => _price(3.0,  0.30, 15.0),
    "claude-sonnet-4-5" => _price(3.0,  0.30, 15.0),   # deprecated, retires 2026-11-30
    "claude-haiku-4-5"  => _price(1.0,  0.10, 5.0),
    # DeepSeek (https://api-docs.deepseek.com/quick_start/pricing, 2026-09-24): peak rates
    # (01:00-04:00 and 06:00-10:00 UTC, Monday-Friday); off-peak hours bill half. The
    # legacy names deepseek-v4-flash(-vision-exp) are served and billed as deepseek-flash.
    "deepseek-flash"               => _price(0.30, 0.006, 1.20),
    "deepseek-v4-flash"            => _price(0.30, 0.006, 1.20),
    "deepseek-v4-flash-vision-exp" => _price(0.30, 0.006, 1.20),
    "deepseek-v4-pro"              => _price(1.32, 0.044, 3.96),
    # Google Gemini, native and OpenAI-compatible (https://ai.google.dev/gemini-api/docs/pricing,
    # 2026-10-22): standard tier, output includes thinking; the Gemini `_billed_cost` prices Flex
    # and Priority from these rows, with the page's own cache-hit prices. Introductory rates double on
    # 2026-12-01 for Gemini 4 Argon and on 2027-01-01 for Gemini 3.8 and 3.6 Flash. Google routes
    # the deprecated gemini-3.7-flash to gemini-3.8-flash and gemini-3.5-flash to gemini-3.6-flash,
    # so those ids carry the served model's rates.
    "gemini-4-argon"        => _price(2.0,  0.10,  10.0),
    "gemini-3.8-flash"      => _price(0.75, 0.075, 3.75),
    "gemini-3.7-flash"      => _price(0.75, 0.075, 3.75),
    "gemini-3.6-flash"      => _price(0.75, 0.075, 3.75),
    "gemini-3.5-flash"      => _price(0.75, 0.075, 3.75),
    "gemini-3.5-flash-lite" => _price(0.30, 0.03,  2.50),
    "gemini-3.1-pro-preview" => _price(2.0, 0.20,  12.0),  # prompts up to 200K tokens; longer ones bill more
    "gemini-3.1-pro-preview-customtools" => _price(2.0, 0.20, 12.0),  # priced with 3.1 Pro Preview; up to 200K too
    "gemini-3.1-flash-lite" => _price(0.25, 0.025, 1.5),   # shuts down 2027-05-07
    "gemini-3-flash-preview" => _price(0.50, 0.05, 3.0),
    "gemini-2.5-pro"        => _price(1.25, 0.125, 10.0),  # prompts up to 200K tokens; longer ones bill more
    "gemini-2.5-flash"      => _price(0.3,  0.03,  2.5),
    "gemini-2.5-flash-lite" => _price(0.1,  0.01,  0.4),
    "gemini-robotics-er-2-preview" => _price(1.0, 0.10, 5.0),  # doubles on 2027-01-01
    "gemini-embedding-2"    => _price(0.20, 0.20,  0.0),   # text input only
    # TypeSafe System One / Jev (docs.typesafe.ai models page, 2026-09-22):
    # $42 per Btok of input, and output tokens are currently free. The aliases
    # resolve to the versioned id, which is what the response reports back.
    "jev-1.13.0"    => _price(0.042, 0.042, 0.0),
    "jev-latest"    => _price(0.042, 0.042, 0.0),
    "jev-preview"   => _price(0.042, 0.042, 0.0),
    # OpenAI embeddings (billed on input tokens only)
    "text-embedding-3-small" => _price(0.02, 0.02, 0.0),
    "text-embedding-3-large" => _price(0.13, 0.13, 0.0),
    "text-embedding-ada-002" => _price(0.10, 0.10, 0.0),
))

"""
    token_usage(result::LLMRequestResponse) -> TokenUsage

Extract token usage from a token-billed API result (chat, Responses, embeddings, image
generation and FIM completions); failures of those APIs report zero usage.

Results from APIs that do not report token usage at all — audio, files, batches,
moderations, vector stores, … — **throw** an `ArgumentError`. A zero would be
indistinguishable from a genuinely free call and would quietly under-count spend.
"""
token_usage(r::LLMRequestResponse)::TokenUsage = throw(ArgumentError(
    "$(typeof(r)) carries no token usage or pricing information"))
token_usage(r::LLMSuccess)::TokenUsage = something(r.usage, TokenUsage())
# The Responses and Images APIs name usage input/output tokens; a response reports the
# tier it was billed at beside its usage, as `service_tier`.
_input_output_usage(::Nothing, _=nothing)::TokenUsage = TokenUsage()
_input_output_usage(u::AbstractDict, tier::Union{String,Nothing}=nothing)::TokenUsage =
    _token_usage_from(u; prompt_key="input_tokens", completion_key="output_tokens",
                      prompt_details="input_tokens_details", completion_details="output_tokens_details",
                      service_tier=tier)
token_usage(r::ResponseSuccess)::TokenUsage =
    _input_output_usage(r.response.usage, _wire_string(r.response.raw, "service_tier"))
token_usage(::LLMFailure)::TokenUsage = TokenUsage()
token_usage(::LLMCallError)::TokenUsage = TokenUsage()
token_usage(::ResponseFailure)::TokenUsage = TokenUsage()
token_usage(::ResponseCallError)::TokenUsage = TokenUsage()
token_usage(r::ImageSuccess)::TokenUsage = _input_output_usage(r.response.usage)
token_usage(::ImageFailure)::TokenUsage = TokenUsage()
token_usage(::ImageCallError)::TokenUsage = TokenUsage()
token_usage(r::EmbeddingSuccess)::TokenUsage = something(r.usage, TokenUsage())
token_usage(::EmbeddingFailure)::TokenUsage = TokenUsage()
token_usage(::EmbeddingCallError)::TokenUsage = TokenUsage()
token_usage(r::FIMSuccess)::TokenUsage = something(r.response.usage, TokenUsage())
token_usage(::FIMFailure)::TokenUsage = TokenUsage()
token_usage(::FIMCallError)::TokenUsage = TokenUsage()

"""
    estimated_cost(result::LLMRequestResponse; model=nothing, pricing=DEFAULT_PRICING) -> Float64

Estimate the cost in USD of a single API call from the token usage its result carries.

The model priced is `model` when given; else the one the provider reports having served —
`result.model` of a Chat result ([`LLMSuccess`](@ref)), `result.response.model` of a
Responses or FIM result; else the one requested — `result.self.model` for Chat,
`result.embeddings.model` for Embeddings.

Returns `0.0` for failures, which carry no billable usage, for an unpriced model, for audio
tokens UniLM has no rate for (see [`DEFAULT_PRICING`](@ref) for the audio it prices) and for an
embeddings reply that reported no usage (Gemini's OpenAI-compatible endpoint reports none);
the last three also log a warning, once per model id (for audio, per model and endpoint). A
billed service tier the provider's pricing page lists no price for is estimated at the standard
rates, with one warning per model and tier. Throws
`ArgumentError` for result types
outside the token-billed APIs — see [`token_usage`](@ref) — and, unless `model` is given,
for an [`ImageSuccess`](@ref): the Images API bills image tokens at rates a price row does
not model, and a price this function cannot compute is not a zero it can report.
`pricing` is any `AbstractDict{String,PriceRow}`.

Model lookup: the exact id first, then the id without a dated snapshot suffix
(`gpt-5.4-mini-2026-03-17` → `gpt-5.4-mini`, `claude-opus-4-5-20251101` →
`claude-opus-4-5`). A versioned Jev id with no row of its
own (`jev-X.Y.Z`, e.g. a release newer than this table) is priced at the `jev-latest`
row. That is an assumption: TypeSafe lists one price, for Jev 1.13; a later version is
assumed to keep it until the pricing page says otherwise. The same lookup prices
[`SystemOneSuccess`](@ref) results.

The row's rates bill fresh input tokens and cache writes at `input`, cache reads at
`cached_input` and completion tokens at `output`, except where UniLM models a provider's
surcharges, which apply to that provider's results (see [`DEFAULT_PRICING`](@ref)).
"""
function estimated_cost(result::LLMRequestResponse;
    model::Union{String,Nothing}=nothing,
    pricing::AbstractDict{String, PriceRow}=DEFAULT_PRICING)::Float64

    u = token_usage(result)
    mdl = isnothing(model) ? _priced_model(result) : model
    isnothing(mdl) && return 0.0                   # a failure: nothing was billed
    service = _result_service(result)
    isnothing(model) && _is_local(service, mdl) && return 0.0
    _unreported_usage(result, mdl)
    rates = _price_row(pricing, mdl)
    isnothing(rates) && return 0.0
    _billed_cost(service, mdl, u, rates)
end

# A reply that reported no usage leaves nothing to price, and a silent 0.0 would read as a free
# call, so an embeddings reply without usage says so once per model (Gemini's OpenAI-compatible
# endpoint returns embeddings without `usage`).
_unreported_usage(::LLMRequestResponse, ::String)::Nothing = nothing
function _unreported_usage(r::EmbeddingSuccess, model::String)::Nothing
    isnothing(r.usage) && @warn "the embeddings reply reported no token usage; its estimated cost is 0.0" model _id = Symbol(
        "unilm_unreported_usage_", model) maxlog = 1
    nothing
end

# The model a result is priced at when the caller names none: the one the provider reports
# having served, else the one requested. A failure has none: it carries no billed usage. A
# success with no method here throws — UniLM does not model its token rates (image
# generation), and a zero would read as a free call.
_priced_model(r::LLMSuccess)::String = something(r.model, r.self.model)
_priced_model(r::ResponseSuccess)::String = r.response.model
_priced_model(r::FIMSuccess)::String = r.response.model
_priced_model(r::EmbeddingSuccess)::String = r.embeddings.model
_priced_model(r::LLMRequestResponse)::Nothing = issuccess(r) ? throw(ArgumentError(
    "$(nameof(typeof(r))) results have no model their usage is priced at: UniLM does not model " *
    "their token rates; pass model= with a pricing= row of your own to price the usage they report")) : nothing

# The standard rates, billed for every provider without a `_billed_cost` method of its own. A
# price row has no audio-token rates, so audio tokens leave the call unpriced.
function _billed_cost(service, model::String, u::TokenUsage, rates::PriceRow)::Float64
    _counts_audio(u) && return _unpriced_audio(service, model)
    cached = min(u.cached_tokens, u.prompt_tokens)        # cache reads, at the discounted rate
    fresh = u.prompt_tokens - cached - u.cache_write_tokens
    fresh * rates.input + cached * rates.cached_input + u.cache_write_tokens * rates.input +
        u.completion_tokens * rates.output
end

_counts_audio(u::TokenUsage)::Bool = u.prompt_audio_tokens > 0 || u.completion_audio_tokens > 0

# Audio tokens that no audio-token rate covers leave the call unpriced, as a model without a price
# row is: `0.0`, and a warning that names the model and the endpoint, once for each pair, so the
# warning of one endpoint does not hide another's.
function _unpriced_audio(service, model::String)::Float64
    endpoint = _service_name(service)
    @warn "no audio-token rates for this model on this endpoint; its estimated cost is 0.0" model endpoint _id = Symbol(
        "unilm_unpriced_audio_", endpoint, "_", model) maxlog = 1
    0.0
end

# Exact id, then the dated-snapshot alias (OpenAI answers an alias with a
# -YYYY-MM-DD snapshot, Anthropic's pre-4.6 ids carry -YYYYMMDD; only those date
# suffixes are stripped, so arbitrary custom names stay unpriced), then the Jev family
# rate for a versioned Jev id. A miss logs once per model id: the 0.0 it becomes is
# indistinguishable from a free call.
function _price_row(pricing::AbstractDict{String,PriceRow}, model::String)::Union{PriceRow,Nothing}
    rates = @something(get(pricing, model, nothing),
                       get(pricing, replace(model, r"-(\d{4}-\d{2}-\d{2}|\d{8})$" => ""), nothing),
                       occursin(r"^jev-\d+\.\d+\.\d+$", model) ? get(pricing, "jev-latest", nothing) : nothing,
                       Some(nothing))
    isnothing(rates) && @warn "no price row for this model; its estimated cost is 0.0" model _id = Symbol(
        "unilm_unpriced_", model) maxlog = 1
    rates
end

# The endpoint a result came from, where the result records it; a model local to it prices
# at 0.0 (`_is_local`, defined with the endpoint).
_result_service(r::LLMSuccess) = r.self.service
_result_service(r::EmbeddingSuccess) = r.embeddings.service
_result_service(r::ResponseSuccess) = r.service
_result_service(r::FIMSuccess) = r.service
_result_service(_) = nothing
_is_local(_, ::AbstractString) = false

# Override the stub from requests.jl to accumulate cost automatically
_accumulate_cost!(chat::Chat, result::LLMSuccess) = (chat._cumulative_cost[] += estimated_cost(result); nothing)

"""
    cumulative_cost(chat::Chat) -> Float64

Return the estimated cost, in USD, that [`chatrequest!`](@ref) calls on this `Chat` have
accrued, priced with [`DEFAULT_PRICING`](@ref) as [`estimated_cost`](@ref) prices a result.
A committed turn adds its result's cost. A call that ends without committing one adds what
its provider reported billing it, a lower bound: the replies of a continued Gemini turn
received before the failure or stop, and the usage a stream had reported when it was stopped,
cancelled, timed out, cut off or ended by an exception from its `callback` or `on_tool_call`
— Anthropic reports a request's input when its stream starts and Gemini the usage so far on
every chunk, while OpenAI reports it after the finish reason. A request stopped before any
usage arrived adds nothing, though the provider bills it, as do an attempt that was retried,
a stream the provider ended with an error event, and an unstreamed reply UniLM could not
decode.
"""
cumulative_cost(chat::Chat)::Float64 = chat._cumulative_cost[]
