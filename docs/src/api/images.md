# Image Generation API

Types and functions for the **Image Generation API** (`/v1/images/generations`).

## Request Type

```@docs
ImageGeneration
```

### Construction

```@example images_api
using UniLM
using JSON

# Minimal request
ig = ImageGeneration(prompt="A watercolor painting of a sunset")
println("Model: ", ig.model)
println("Prompt: ", ig.prompt)

# Full options
ig2 = ImageGeneration(
    prompt="A minimalist logo for a Julia package",
    size="1024x1024",
    quality="high",
    background="transparent",
    output_format="png",
    n=2
)
println("\nFull options:")
println("  Size: ", ig2.size)
println("  Quality: ", ig2.quality)
println("  Background: ", ig2.background)
println("  Format: ", ig2.output_format)
println("  Count: ", ig2.n)
```

### JSON Serialization

```@example images_api
ig = ImageGeneration(prompt="A cute robot", quality="high", size="1024x1024")
println(JSON.json(ig))
```

## Response Types

```@docs
ImageObject
ImageResponse
```

## Result Types

```@docs
ImageSuccess
ImageFailure
ImageCallError
```

## Request Function

```@docs
generate_image
```

### Usage Examples

```julia
julia> result = generate_image(
           "A watercolor painting of a friendly robot reading a Julia programming book",
           size="1024x1024", quality="medium"
       )

julia> result isa ImageSuccess
true

julia> length(image_data(result))
1

julia> save_image(image_data(result)[1], "robot_julia.png")
"robot_julia.png"
```

## Accessor Functions

```@docs
image_data
save_image
```

### Saving Images

```julia
result = generate_image("A sunset over mountains", n=3)
if result isa ImageSuccess
    for (i, img) in enumerate(image_data(result))
        save_image(img, "sunset_$i.png")
    end
end
```

## Parameters Reference

| Parameter            | Type   | Default           | Description                        |
| :------------------- | :----- | :---------------- | :--------------------------------- |
| `model`              | String | `"gpt-image-1.5"` | Image generation model             |
| `prompt`             | String | *(required)*      | Text description of the image      |
| `n`                  | Int    | `1`               | Number of images (1–10)            |
| `size`               | String | `"auto"`          | `"1024x1024"`, `"1536x1024"`, etc. |
| `quality`            | String | `"auto"`          | `"low"`, `"medium"`, `"high"`      |
| `background`         | String | `"auto"`          | `"transparent"`, `"opaque"`        |
| `output_format`      | String | `"png"`           | `"png"`, `"webp"`, `"jpeg"`        |
| `output_compression` | Int    | —                 | 0–100, for webp/jpeg               |
| `user`               | String | —                 | End-user identifier                |
