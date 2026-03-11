"""Per-token pricing: `Dict{String, @NamedTuple{input::Float64, output::Float64}}`.
Values are cost per token (i.e. per-million / 1_000_000)."""
const DEFAULT_PRICING = Dict{String, @NamedTuple{input::Float64, output::Float64}}(
    # GPT-5.2
    "gpt-5.2"       => (input=2.0/1_000_000,  output=8.0/1_000_000),
    # GPT-4.1 family
    "gpt-4.1"       => (input=2.0/1_000_000,  output=8.0/1_000_000),
    "gpt-4.1-mini"  => (input=0.4/1_000_000,  output=1.6/1_000_000),
    "gpt-4.1-nano"  => (input=0.1/1_000_000,  output=0.4/1_000_000),
    # O-series
    "o3"            => (input=2.0/1_000_000,  output=8.0/1_000_000),
    "o4-mini"       => (input=1.1/1_000_000,  output=4.4/1_000_000),
)

"""
    token_usage(result::LLMRequestResponse) -> TokenUsage

Extract token usage from any API result. Returns zero-usage for failures.
"""
token_usage(r::LLMSuccess)::TokenUsage = something(r.usage, TokenUsage())
token_usage(r::ResponseSuccess)::TokenUsage = begin
    u = r.response.usage
    isnothing(u) && return TokenUsage()
    TokenUsage(
        prompt_tokens=get(u, "input_tokens", get(u, "prompt_tokens", 0)),
        completion_tokens=get(u, "output_tokens", get(u, "completion_tokens", 0)),
        total_tokens=get(u, "total_tokens", 0)
    )
end
token_usage(::LLMFailure)::TokenUsage = TokenUsage()
token_usage(::LLMCallError)::TokenUsage = TokenUsage()
token_usage(::ResponseFailure)::TokenUsage = TokenUsage()
token_usage(::ResponseCallError)::TokenUsage = TokenUsage()
token_usage(::ImageSuccess)::TokenUsage = TokenUsage()
token_usage(::ImageFailure)::TokenUsage = TokenUsage()
token_usage(::ImageCallError)::TokenUsage = TokenUsage()

"""
    estimated_cost(result::LLMRequestResponse; model=nothing, pricing=DEFAULT_PRICING) -> Float64

Estimate the cost in USD for a single API call result.
If `model` is not provided, it is inferred from the result when possible.
"""
function estimated_cost(result::LLMRequestResponse;
    model::Union{String,Nothing}=nothing,
    pricing::Dict{String, @NamedTuple{input::Float64, output::Float64}}=DEFAULT_PRICING)::Float64

    u = token_usage(result)
    mdl = if !isnothing(model)
        model
    elseif result isa LLMSuccess
        result.self.model
    elseif result isa ResponseSuccess
        result.response.model
    else
        return 0.0
    end
    rates = get(pricing, mdl, nothing)
    isnothing(rates) && return 0.0
    u.prompt_tokens * rates.input + u.completion_tokens * rates.output
end

# Override the stub from requests.jl to accumulate cost automatically
_accumulate_cost!(chat::Chat, result::LLMSuccess) = (chat._cumulative_cost[] += estimated_cost(result); nothing)

"""
    cumulative_cost(chat::Chat) -> Float64

Return the cumulative estimated cost accumulated by `chatrequest!` calls on this `Chat`.
"""
cumulative_cost(chat::Chat)::Float64 = chat._cumulative_cost[]
