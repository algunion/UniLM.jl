# [Image Generation](@id images_guide)

```@setup image_generation
using UniLM
using JSON
```

[`generate_image`](@ref) makes images from a text prompt, and [`edit_image`](@ref) changes images you give
it, on a provider that serves them. The request's `service` picks the provider; the verbs, the request
types and the result types are the same on each, and the options each provider takes are on its entry.

```@example image_generation
service, model = OPENAIServiceEndpoint, "gpt-image-2.5-sunburst"
nothing # hide
```

The examples use only the names bound above. `service` and `model` are the provider and model they run
on: bind your own there. The table [at the end](@ref image_generation_providers) links each provider's rules.

## [Generating an image](@id image_generation_generate)

An image request is not recorded, so its code stands here without output:

```julia
result = generate_image(
    "A watercolor painting of a friendly robot reading a Julia programming book";
    service, model
)
if result isa ImageSuccess
    imgs = image_data(result)
    println("Images: ", length(imgs))
    save_image(imgs[1], "robot.png")
else
    println("Image generation failed — see result for details")
end
```

`generate_image(prompt; kwargs...)` builds an [`ImageGeneration`](@ref) from its keywords and sends it;
`config` and `cancel` go to the request.

## [The request](@id image_generation_request)

For full control, build the [`ImageGeneration`](@ref) yourself:

```@example image_generation
ig = ImageGeneration(service=service, prompt="A minimalist logo for a Julia programming package")
println("Model field: ", repr(ig.model))
println("\nRequest JSON:")
println(JSON.json(ig))
```

These fields hold on every provider; the others, which size and shape the image, are each provider's,
on its entry:

| Field | What it holds |
| :--- | :--- |
| `service` | the provider |
| `model` | the image model; `""`, the default, is the provider's default image model |
| `prompt` | a description of the image |
| `n` | how many images one request makes, as many as the provider makes in one |
| `safety_identifier` | a stable identifier for your end user, for the provider's abuse monitoring |

The `model` field is a sentinel, not a resolved default: unlike [`Chat`](@ref), `ImageGeneration` leaves
it `""` until the request is serialized, so `ig.model` reads back as the empty string, and `JSON.json(ig)`
shows the body that goes on the wire. A service with no default image model throws `ArgumentError` at
serialization. Set `model=` to read the model back from the request.

`safety_identifier` names the end user a request is made for, so the provider can tell your users apart;
send a hash of a username or email address, never the address itself. Each provider's rule for its form
is on its entry.

A field the request's provider does not take throws `ArgumentError` naming the provider and the field —
when the request is serialized, and in `generate_image` before any request — and so does a value UniLM
checks against the provider's rules; each entry lists what its provider refuses.

## [Results](@id image_generation_results)

```@example image_generation
# Show the type hierarchy for image results
println("ImageSuccess <: ", supertype(ImageSuccess))
println("ImageFailure <: ", supertype(ImageFailure))
println("ImageCallError <: ", supertype(ImageCallError))
```

```julia
result = generate_image("A sunset over mountains"; service, model)

if result isa ImageSuccess
    r = result.response

    r.created                 # Unix timestamp
    r.data                    # Vector{ImageObject}
    r.data[1].b64_json        # base64-encoded image data (nothing when delivered by URL)
    r.data[1].url             # the image URL, when the API delivered it that way
    r.data[1].revised_prompt  # revised prompt (may be nothing)
    r.usage                   # token usage Dict (token_usage(result) reads it)

    # Convenience accessors
    image_data(result)        # Vector{String}: each image's base64 data, else its URL
    save_image(image_data(result)[1], "sunset.png")
end
```

A request that makes several images (`n`) gives them in order: save each one.

```julia
for (i, img) in enumerate(image_data(result))
    save_image(img, "image_$i.png")
end
```

### [Saving images](@id image_generation_save)

The `save_image` helper decodes base64 and writes to disk:

```@example image_generation
# Demonstrate save_image with a tiny test payload
tmpfile = tempname() * ".txt"
UniLM.save_image("aGVsbG8=", tmpfile)  # "hello" in base64
println("File saved to: ", basename(tmpfile))
println("Contents: ", read(tmpfile, String))
rm(tmpfile)
```

### [Error handling](@id image_generation_errors)

`image_data` throws [`LLMResultError`](@ref) on an `ImageFailure` or `ImageCallError` — a
failed call has no images — so branch on the result type first:

```julia
result = generate_image("A sunset over mountains"; service, model)

if result isa ImageSuccess
    save_image(image_data(result)[1], "sunset.png")
elseif result isa ImageFailure
    @warn "HTTP $(result.status): $(result.response)"
elseif result isa ImageCallError
    @error "Call failed: $(result.error)"
end
```

## [Editing images](@id image_generation_edit)

[`edit_image`](@ref) changes existing images as a text prompt asks. The convenience form takes the source
image (a file path, or a vector of paths) and the prompt:

```julia
result = edit_image("room.png", "Add a large window with a sea view"; service, model)

if result isa ImageSuccess
    save_image(image_data(result)[1], "room_edited.png")
else
    println("Edit failed — see result for details")
end
```

For full control, build an [`ImageEdit`](@ref) and call `edit_image(e)`. `model` left `""` is the
provider's default image model, resolved when the edit is sent. An empty vector of images, or an
image or mask path that is no file, throws `ArgumentError` before any request. Image editing needs the `:image_edits` capability: an endpoint that does not declare it,
one that declares no capabilities included, throws `ArgumentError` before any request. The fields of an edit that one provider
alone takes, such as a mask, are on its entry.

## [Retry behaviour](@id image_generation_retries)

`generate_image` and `edit_image` retry transient HTTP statuses (408, 429, 500, 502–504, 520–524, 529) within the request budget configured by `RequestConfig` (`max_attempts`, default 3), with exponential backoff and jitter; the wait a reply asks for (its `retry-after-ms` header, else `Retry-After`) is a floor under the jittered wait. Override per call with `config=RequestConfig(...)`. Both take `cancel=` (default: the ambient `with_cancel` token); a cancelled call returns an `ImageCallError` whose `cause` is a [`UniLMCancelled`](@ref). Every `ImageFailure` and `ImageCallError` carries the provider's `request_id` when one was sent, and a call error its exception in `cause`.

## [See also](@id image_generation_see_also)

- [`ImageGeneration`](@ref) — request configuration type
- [`ImageResponse`](@ref) — response type
- [`generate_image`](@ref) — request function
- [`image_data`](@ref), [`save_image`](@ref) — accessor/utility functions

## [On your provider](@id image_generation_providers)

```@eval
Main.ProviderFacts.on_your_provider(Main.ProviderFacts.llm, :images)
```
