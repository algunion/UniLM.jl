"""A per-token pricing row: `input` / `cached_input` / `output` USD per token."""
const PriceRow = @NamedTuple{input::Float64, cached_input::Float64, output::Float64}

"""Build a [`PriceRow`](@ref) from per-1M-token USD figures (input, cached-input, output)."""
_price(i, c, o) = (input = i / 1_000_000, cached_input = c / 1_000_000, output = o / 1_000_000)

"""Default per-token pricing, verified against the live OpenAI pricing page on 2026-06-21
(prices drift — re-verify before relying on them). Cached input is billed at the discounted
`cached_input` rate; reasoning tokens are already counted within output tokens."""
const DEFAULT_PRICING = Dict{String, PriceRow}(
    # GPT-5.x  (live-verified 2026-06-21)
    "gpt-5.5"       => _price(5.0,  0.50,  30.0),
    "gpt-5.4"       => _price(2.5,  0.25,  15.0),
    "gpt-5.4-mini"  => _price(0.75, 0.075, 4.5),
    "gpt-5.2"       => _price(1.75, 0.175, 14.0),
    # GPT-4.1 family
    "gpt-4.1"       => _price(2.0,  0.50,  8.0),
    "gpt-4.1-mini"  => _price(0.4,  0.10,  1.6),
    "gpt-4.1-nano"  => _price(0.1,  0.025, 0.4),
    # O-series
    "o3"            => _price(2.0,  0.50,  8.0),
    "o4-mini"       => _price(1.1,  0.275, 4.4),
    # Anthropic Claude (claude-api reference, 2026-07-06; cache-read input ≈ 0.1× input)
    "claude-opus-4-8"  => _price(5.0, 0.50, 25.0),
    "claude-sonnet-5"  => _price(3.0, 0.30, 15.0),
    "claude-haiku-4-5" => _price(1.0, 0.10, 5.0),
    # Google Gemini (native + OpenAI-compat shim; live-verified 2026-07-07)
    "gemini-3.5-flash"      => _price(1.5,  0.15,  9.0),
    "gemini-3.1-flash-lite" => _price(0.25, 0.025, 1.5),
    "gemini-2.5-flash"      => _price(0.3,  0.03,  2.5),
    "gemini-2.5-flash-lite" => _price(0.1,  0.01,  0.4),
    # Embeddings (billed on input tokens only)
    "text-embedding-3-small" => _price(0.02, 0.02, 0.0),
    "text-embedding-3-large" => _price(0.13, 0.13, 0.0),
)

"""
    token_usage(result::LLMRequestResponse) -> TokenUsage

Extract token usage from any API result. Returns zero-usage for failures.
"""
token_usage(r::LLMSuccess)::TokenUsage = something(r.usage, TokenUsage())
token_usage(r::ResponseSuccess)::TokenUsage = begin
    u = r.response.usage
    isnothing(u) && return TokenUsage()
    _token_usage_from(u; prompt_key="input_tokens", completion_key="output_tokens",
        prompt_details="input_tokens_details", completion_details="output_tokens_details")
end
token_usage(::LLMFailure)::TokenUsage = TokenUsage()
token_usage(::LLMCallError)::TokenUsage = TokenUsage()
token_usage(::ResponseFailure)::TokenUsage = TokenUsage()
token_usage(::ResponseCallError)::TokenUsage = TokenUsage()
token_usage(::ImageSuccess)::TokenUsage = TokenUsage()
token_usage(::ImageFailure)::TokenUsage = TokenUsage()
token_usage(::ImageCallError)::TokenUsage = TokenUsage()
token_usage(r::EmbeddingSuccess)::TokenUsage = something(r.usage, TokenUsage())
token_usage(::EmbeddingFailure)::TokenUsage = TokenUsage()
token_usage(::EmbeddingCallError)::TokenUsage = TokenUsage()

"""
    estimated_cost(result::LLMRequestResponse; model=nothing, pricing=DEFAULT_PRICING) -> Float64

Estimate the cost in USD for a single API call result.
If `model` is not provided, it is inferred from the result when possible.
"""
function estimated_cost(result::LLMRequestResponse;
    model::Union{String,Nothing}=nothing,
    pricing::Dict{String, PriceRow}=DEFAULT_PRICING)::Float64

    u = token_usage(result)
    mdl = if !isnothing(model)
        model
    elseif result isa LLMSuccess
        result.self.model
    elseif result isa ResponseSuccess
        result.response.model
    elseif result isa EmbeddingSuccess
        result.embeddings.model
    else
        return 0.0
    end
    rates = get(pricing, mdl, nothing)
    isnothing(rates) && return 0.0
    cached = min(u.cached_tokens, u.prompt_tokens)        # cached input billed at the discounted rate
    fresh = u.prompt_tokens - cached
    fresh * rates.input + cached * rates.cached_input + u.completion_tokens * rates.output
end

# Override the stub from requests.jl to accumulate cost automatically
_accumulate_cost!(chat::Chat, result::LLMSuccess) = (chat._cumulative_cost[] += estimated_cost(result); nothing)

"""
    cumulative_cost(chat::Chat) -> Float64

Return the cumulative estimated cost accumulated by `chatrequest!` calls on this `Chat`.
"""
cumulative_cost(chat::Chat)::Float64 = chat._cumulative_cost[]
