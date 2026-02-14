# [Image Generation](@id images_guide)

UniLM.jl supports image generation via the OpenAI Images API using models like
`gpt-image-1.5`.

## Basic Usage

```julia
using UniLM

result = generate_image("A watercolor painting of a mountain sunset")

if result isa ImageSuccess
    # Save the first image to disk
    save_image(image_data(result)[1], "sunset.png")
    println("Image saved!")
end
```

## The `ImageGeneration` Type

For full control, construct an [`ImageGeneration`](@ref) object:

```@example images
using UniLM
using JSON

ig = ImageGeneration(
    prompt="A minimalist logo for a Julia programming package",
    model="gpt-image-1.5",
    size="1024x1024",
    quality="high",
    background="transparent",
    output_format="png"
)
println("Model: ", ig.model)
println("Size: ", ig.size)
println("Quality: ", ig.quality)
println("\nRequest JSON:")
println(JSON.json(ig))
```

## Configuration Options

| Parameter            | Values                                                | Default           |
| :------------------- | :---------------------------------------------------- | :---------------- |
| `model`              | `"gpt-image-1.5"`                                     | `"gpt-image-1.5"` |
| `size`               | `"1024x1024"`, `"1536x1024"`, `"1024x1536"`, `"auto"` | API default       |
| `quality`            | `"low"`, `"medium"`, `"high"`, `"auto"`               | API default       |
| `background`         | `"transparent"`, `"opaque"`, `"auto"`                 | API default       |
| `output_format`      | `"png"`, `"webp"`, `"jpeg"`                           | API default       |
| `output_compression` | `0`–`100` (for webp/jpeg)                             | API default       |
| `n`                  | `1`–`10`                                              | `1`               |

## Multiple Images

Generate multiple images in a single request:

```julia
result = generate_image(
    "A cute robot learning to program",
    n=3,
    size="1024x1024"
)

if result isa ImageSuccess
    imgs = image_data(result)
    println("Generated $(length(imgs)) images")
    for (i, img) in enumerate(imgs)
        save_image(img, "robot_$i.png")
    end
end
```

## Transparent Backgrounds

Perfect for logos and icons:

```julia
result = generate_image(
    "A simple geometric icon of a butterfly",
    background="transparent",
    output_format="png",
    quality="high"
)

if result isa ImageSuccess
    save_image(image_data(result)[1], "butterfly_icon.png")
    # => PNG with transparent background
end
```

## Result Structure

```@example images
# Show the type hierarchy for image results
println("ImageSuccess <: ", supertype(ImageSuccess))
println("ImageFailure <: ", supertype(ImageFailure))
println("ImageCallError <: ", supertype(ImageCallError))
```

```julia
result = generate_image("A sunset")

if result isa ImageSuccess
    r = result.response

    r.created       # => 1713833628 (Unix timestamp)
    r.data          # => Vector{ImageObject}
    r.data[1].b64_json        # => "iVBORw0KGgo..." (base64 image data)
    r.data[1].revised_prompt  # => "A vibrant sunset over..." (may be nothing)
    r.usage         # => Dict("total_tokens" => 100, ...)

    # Convenience accessors
    image_data(result)   # => ["iVBORw0KGgo..."] (Vector{String})
end
```

## Saving Images

The `save_image` helper decodes base64 and writes to disk:

```@example images
# Demonstrate save_image with a tiny test payload
tmpfile = tempname() * ".txt"
UniLM.save_image("aGVsbG8=", tmpfile)  # "hello" in base64
println("File saved to: ", basename(tmpfile))
println("Contents: ", read(tmpfile, String))
rm(tmpfile)
```

## Error Handling

```julia
result = generate_image("test prompt")

if result isa ImageSuccess
    println("Generated $(length(result.response.data)) image(s)")
elseif result isa ImageFailure
    @warn "HTTP $(result.status): $(result.response[1:min(200,end)])"
elseif result isa ImageCallError
    @error "Call failed: $(result.error)"
end
```

## See Also

- [`ImageGeneration`](@ref) — request configuration type
- [`ImageResponse`](@ref) — response type
- [`generate_image`](@ref) — request function
- [`image_data`](@ref), [`save_image`](@ref) — accessor/utility functions
