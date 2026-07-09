# [Cost Tracking & Token Usage](@id accounting_api)

Token accounting works across every provider — Gemini and Anthropic usage is
normalized to the shared shape at decode time, so these accessors and the pricing
table apply uniformly to Chat, Responses, and the agentic verb.

For a worked walkthrough — reading usage, the silent-`\$0` footgun on unpriced models,
custom pricing, and per-`Chat` cumulative totals — see the [Cost Tracking guide](@ref cost_guide).

## Token Usage

```@docs
TokenUsage
token_usage
```

## Cost Estimation

```@docs
estimated_cost
cumulative_cost
DEFAULT_PRICING
```
