# [Moderations API](@id moderations_api)

Classify text and images for policy violations, returning per-category flags and
confidence scores. Moderation is free. OpenAI only.

## Parsed Objects

```@docs
ModerationResult
ModerationResponse
```

## Result Types

```@docs
ModerationSuccess
ModerationFailure
ModerationCallError
```

## Functions

```@docs
moderate
is_flagged
```

## Usage

```julia
# Classify text for policy violations (free)
result = moderate("...text to check...")
result isa ModerationSuccess && println("Flagged? ", is_flagged(result))

# Inspect per-category flags and scores
if result isa ModerationSuccess
    for m in result.response.results
        m.flagged && println(m.categories)
    end
end
```
