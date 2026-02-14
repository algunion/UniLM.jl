```julia
julia> result = chatrequest!(
           systemprompt="You are a calculator. Respond only with the number.",
           userprompt="What is 42 * 17?",
           model="gpt-4o-mini",
           temperature=0.0
       )

julia> result.message.content
"714"
```
