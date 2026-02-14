```julia
julia> result = generate_image(
           "A watercolor painting of a friendly robot reading a Julia programming book",
           size="1024x1024",
           quality="medium"
       )

julia> result isa ImageSuccess
true

julia> length(image_data(result))
1

julia> length(image_data(result)[1])  # base64 string length
2654660

julia> result.response.data[1].revised_prompt
nothing

julia> result.response.usage
{
  "input_tokens": 25,
  "input_tokens_details": {
    "image_tokens": 0,
    "text_tokens": 25
  },
  "output_tokens_details": {
    "image_tokens": 1056,
    "text_tokens": 446
  },
  "total_tokens": 1527,
  "output_tokens": 1502
}

julia> save_image(image_data(result)[1], "robot_julia.png")
"robot_julia.png"
```

**Generated image:**

![A watercolor painting of a friendly robot reading a Julia programming book](assets/generated_robot.png)
