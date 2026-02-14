```julia
julia> r1 = respond("Tell me a one-liner programming joke.", instructions="Be concise.")

julia> output_text(r1)
"There are only 10 kinds of people in the world: those who understand binary and those who don’t."

julia> r2 = respond("Explain why that's funny, in one sentence.", previous_response_id=r1.response.id)

julia> output_text(r2)
"It’s funny because “10” looks like ten in decimal but equals two in binary, so it sets up a nerdy misdirection that only people who know binary immediately get."
```
