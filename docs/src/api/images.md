# [Image Generation API](@id images_api)

Types and functions for generating and editing images: OpenAI's Images API
(`/v1/images/generations`, `/v1/images/edits`) and Gemini's native image generation
(Nano Banana) through `generateContent`, chosen by the request's `service`.

## Request Type

```@docs
ImageGeneration
ImageEdit
```

### Construction

```@example images_api
using UniLM
using JSON

# Minimal request. `model` is a sentinel: it stays "" on the struct and resolves
# only when the request is serialized, so read the wire body to see what will be
# sent rather than the field.
ig = ImageGeneration(prompt="A watercolor painting of a sunset")
println("Model field: ", repr(ig.model))
println("Model on the wire: ", JSON.parse(JSON.json(ig))["model"])
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
ig = ImageGeneration(prompt="A cute robot", quality="high", size="1024x1024",
                     safety_identifier="user-7f3a")
println(JSON.json(ig))
```

A Gemini request serializes to its `generateContent` body; the model,
`gemini-nano-banana-2.1` unless set, goes in the URL:

```@example images_api
gig = ImageGeneration(service=GEMINIServiceEndpoint, prompt="A lighthouse at dusk",
                      aspect_ratio="16:9", resolution="2K")
println(JSON.json(gig))
```

A field the other provider owns is refused before anything is sent:

```@example images_api
try
    JSON.json(ImageGeneration(service=GEMINIServiceEndpoint, prompt="A lighthouse", quality="high"))
catch e
    println(e.msg)
end
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
edit_image
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

| Parameter            | Provider | Type   | Default           | Description                        |
| :------------------- | :------- | :----- | :---------------- | :--------------------------------- |
| `model`              | both     | String | `""`              | Sentinel; resolves on serialization to `"gpt-image-2.5-sunburst"` (OpenAI) or `"gemini-nano-banana-2.1"` (Gemini) |
| `prompt`             | both     | String | *(required)*      | Text description of the image      |
| `n`                  | both     | Int    | `1`               | Number of images (1–10); Gemini takes only 1 |
| `safety_identifier`  | both     | String | —                 | Stable end-user id for abuse monitoring: 1–128 characters on OpenAI; on Gemini the request label, at most 63 lowercase letters, numerals, `_` or `-`, international ones included |
| `size`               | OpenAI   | String | `"auto"`          | `"1024x1024"`, `"1536x1024"`, etc.; on `gpt-image-2` and later any `WIDTHxHEIGHT` with multiples of 16, up to `3840x2160` |
| `quality`            | OpenAI   | String | `"auto"`          | `"low"`, `"medium"`, `"high"`; `"xhigh"` and `"max"` on `gpt-image-2.5-flare` / `gpt-image-2.5-sunburst` |
| `background`         | OpenAI   | String | `"auto"`          | `"transparent"`, `"opaque"`        |
| `output_format`      | OpenAI   | String | `"png"`           | `"png"`, `"webp"`, `"jpeg"`        |
| `output_compression` | OpenAI   | Int    | —                 | 0–100, for webp/jpeg               |
| `moderation`         | OpenAI   | String | `"auto"`          | `"auto"`, `"low"`                  |
| `aspect_ratio`       | Gemini   | String | `"1:1"`           | `"1:1"`, `"1:4"`, `"4:1"`, `"1:8"`, `"8:1"`, `"2:3"`, `"3:2"`, `"3:4"`, `"4:3"`, `"4:5"`, `"5:4"`, `"9:16"`, `"16:9"`, `"21:9"`; an edit keeps its input's ratio by default |
| `resolution`         | Gemini   | String | `"1K"`            | `"512"`, `"1K"`, `"2K"`, `"4K"`; `gemini-nano-banana-2.1` renders 1K, 2K or 4K |

Defaults shown as quoted strings other than `model` are the **API's** defaults for
an omitted field; the struct stores `nothing` and omits them from the request body.
A field set for the provider that does not take it throws `ArgumentError` naming the
provider and the field. `input_fidelity` and `mask` are edit parameters (OpenAI): set
them on [`ImageEdit`](@ref) — `ImageGeneration` does not accept them.
