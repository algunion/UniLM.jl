```julia
julia> fmt = json_schema_format(
           "languages", "A list of programming languages",
           Dict(
               "type" => "object",
               "properties" => Dict(
                   "languages" => Dict(
                       "type" => "array",
                       "items" => Dict(
                           "type" => "object",
                           "properties" => Dict(
                               "name" => Dict("type" => "string"),
                               "year" => Dict("type" => "integer"),
                               "paradigm" => Dict("type" => "string")
                           ),
                           "required" => ["name", "year", "paradigm"],
                           "additionalProperties" => false
                       )
                   )
               ),
               "required" => ["languages"],
               "additionalProperties" => false
           ),
           strict=true
       )

julia> result = respond(
           "List Julia, Python, and Rust with their release year and primary paradigm.",
           text=fmt
       )

julia> JSON.parse(output_text(result))
{
  "languages": [
    {
      "name": "Julia",
      "year": 2012,
      "paradigm": "Multi-paradigm (scientific/numerical, functional, concurrent)"
    },
    {
      "name": "Python",
      "year": 1991,
      "paradigm": "Multi-paradigm (object-oriented, imperative, functional)"
    },
    {
      "name": "Rust",
      "year": 2010,
      "paradigm": "Multi-paradigm (systems programming, functional, imperative)"
    }
  ]
}
```
