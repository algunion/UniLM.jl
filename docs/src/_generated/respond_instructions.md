```julia
julia> result = respond(
           "Translate to French: The quick brown fox jumps over the lazy dog.",
           instructions="You are a professional translator. Respond only with the translation."
       )

julia> output_text(result)
"Le rapide renard brun saute par-dessus le chien paresseux."
```
