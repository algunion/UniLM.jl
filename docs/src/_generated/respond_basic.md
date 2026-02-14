```julia
julia> result = respond("Explain Julia's multiple dispatch in 2-3 sentences.")

julia> output_text(result)
"Julia’s multiple dispatch means a function can have many method definitions, and Julia chooses which one to run based on the types of *all* arguments in a call (not just the first). This makes it easy to write generic code while still getting specialized, high-performance behavior for specific type combinations."

julia> result.response.id
"resp_00e791c82448c27d006990c7a81de88194975ba388932de6b8"

julia> result.response.status
"completed"

julia> result.response.model
"gpt-5.2-2025-12-11"

julia> result.response.usage
{
  "input_tokens": 18,
  "input_tokens_details": {
    "cached_tokens": 0
  },
  "output_tokens_details": {
    "reasoning_tokens": 0
  },
  "total_tokens": 81,
  "output_tokens": 63
}
```
