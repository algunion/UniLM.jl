```julia
julia> task = respond("Write a haiku about Julia programming.") do chunk, close
           if chunk isa String
               print(chunk)  # tokens stream in real-time
           elseif chunk isa ResponseObject
               println("\nDone! Status: ", chunk.status)
           end
       end
Multiple dispatch sings,  
Types align in swift fusion—  
Loops bloom into speed.
Done! Status: completed

julia> result = fetch(task)

julia> output_text(result)
"Multiple dispatch sings,  \nTypes align in swift fusion—  \nLoops bloom into speed."
```
