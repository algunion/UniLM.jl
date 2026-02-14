```julia
julia> result = respond(
           "What is the latest stable release of the Julia programming language?",
           tools=[web_search()]
       )

julia> output_text(result)
"The latest **stable** release of the Julia programming language is **Julia v1.12.5**. ([julialang.org](https://julialang.org/))"
```
