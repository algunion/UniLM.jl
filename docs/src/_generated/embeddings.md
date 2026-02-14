```julia
julia> emb = Embeddings("Julia is a high-performance programming language for technical computing.")

julia> embeddingrequest!(emb)

julia> emb.embeddings[1:5]  # first 5 dimensions
5-element Vector{Float64}:
  -0.039474
  -0.009283
  0.001706
  -0.028087
  0.063363

julia> sqrt(sum(x^2 for x in emb.embeddings))  # L2 norm ≈ 1.0
1.0
```
