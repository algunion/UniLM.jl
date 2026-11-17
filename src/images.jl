# ============================================================================
# Image generation and editing
# OpenAI Images API: https://developers.openai.com/api/reference/resources/images
# Gemini image generation (Nano Banana) through generateContent:
# https://ai.google.dev/gemini-api/docs/image-generation
#
# One request type per verb; the service picks the wire. Every endpoint is sent
# the OpenAI Images API request except GEMINIServiceEndpoint, which is sent
# generateContent with image output. Each wire refuses, before any I/O, the
# fields only the other provider has.
# ============================================================================

# ─── Request Type ─────────────────────────────────────────────────────────────

"""
    ImageGeneration(; prompt, service=OPENAIServiceEndpoint, model="", kwargs...)

An image-generation request for [`generate_image`](@ref). It goes to OpenAI's Images
API (`/v1/images/generations`), or, with `service=GEMINIServiceEndpoint`, to Gemini's
`generateContent` with image output (Nano Banana). Each field below says which
provider takes it.

# Fields — both providers
- `service`: `OPENAIServiceEndpoint` (default) or `GEMINIServiceEndpoint`; any other
  endpoint is sent the OpenAI Images API request.
- `model::String`: The default is the sentinel `""`, which resolves at serialization
  time to the service's default image model (`"gpt-image-2.6-sunburst"` for OpenAI,
  `"gemini-nano-banana-2.1"` for Gemini) — so the field READS BACK as `""` until you set
  it explicitly, and a service with no default image model throws `ArgumentError` when
  the request is serialized. Inspect `JSON.json(ig)` to see the body that will go on the
  wire; Gemini names the model in the URL instead.
- `prompt::String`: A text description of the desired image
- `n::Union{Int,Nothing}`: Number of images to generate: 1–10 on OpenAI; Gemini returns
  one image per request, so it takes only `1`
- `safety_identifier::Union{String,Nothing}`: A stable identifier for your end user (hash
  a username or email address), for the provider's abuse monitoring. OpenAI takes 1 to
  128 characters
  ([safety identifiers](https://developers.openai.com/api/docs/guides/safety-best-practices#implement-safety-identifiers));
  Gemini sends it as the request label `safety_identifier`, which allows at most 63
  lowercase letters, numerals, underscores and dashes, international ones included.

# Fields — OpenAI only
- `size::Union{String,Nothing}`: Size (`"1024x1024"`, `"1536x1024"`, `"1024x1536"`, `"auto"`).
  On `gpt-image-2` and later, any `WIDTHxHEIGHT` with both edges multiples of 16, up to
  `3840x2160` (aspect ratio between 1:3 and 3:1)
- `quality::Union{String,Nothing}`: Quality level (`"low"`, `"medium"`, `"high"`, `"auto"`); the
  Sunburst and Flare models of GPT-Image-2.6 and GPT-Image-2.5 also accept `"xhigh"` and `"max"`
- `background::Union{String,Nothing}`: Background (`"transparent"`, `"opaque"`, `"auto"`)
- `output_format::Union{String,Nothing}`: File format (`"png"`, `"webp"`, `"jpeg"`, `"gif"`, `"svg"`);
  GIF output "Requires access to the GIF preview for the selected model" and takes no
  `output_compression`. Gemini takes `"jpeg"` (below)
- `output_compression::Union{Int,Nothing}`: Compression (0–100, for `"webp"` and `"jpeg"`)
- `moderation::Union{String,Nothing}`: Content-moderation strictness (`"auto"`, `"low"`)
- `complexity::Union{String,Nothing}`: `"low"`, `"medium"` or `"high"`, "Computation spent planning
  the image, independently of rendering quality" (the GPT-Image-2.6 models, which "use `high`" when it
  is omitted); another value throws `ArgumentError` when the request is built
- `frame_count::Union{Int,Nothing}`: the frames of a GIF, 2 to 16 (9 when omitted), with
  `output_format="gif"` only
- `frame_duration_ms::Union{Int,Nothing}`: each GIF frame's delay in milliseconds, at least 20 (200
  when omitted), with `output_format="gif"` only; OpenAI "caps the encoded delay at 655,350 ms per
  frame", and UniLM sets no largest
- `svg::Union{SvgOptions,Nothing}`: the model that authors an SVG ([`SvgOptions`](@ref)), with
  `output_format="svg"` only. "SVG generation requires n=1 and does not support streaming,
  partial images, prompt_rewriter, or output_compression"; the SVG comes back as the image's base64
  data, its author's tokens as the reply's `svg_usage`. An [`ImageEdit`](@ref) takes it too
- `stream::Bool`: stream the image (default `false`): the partial images reach the `callback` of
  [`generate_image`](@ref) as [`PartialImage`](@ref)s while it is made, and the call returns once it
  arrives. OpenAI streams one image per request ("Streaming is only supported with n=1.", its HTTP 400
  for `n=2` on 2026-11-17), and neither GIF nor SVG output
- `partial_images::Union{Int,Nothing}`: how many partial images a stream sends, 0 to 3 ("Value must
  be between 0 and 3"), with `stream=true` only; "the final image may be sent before the full number
  of partial images are generated if the full image is generated more quickly"

# Fields — Gemini only
[Gemini's image controls](https://ai.google.dev/gemini-api/docs/image-generation#aspect-ratios-and-image-size):
- `aspect_ratio::Union{String,Nothing}`: `"1:1"`, `"1:4"`, `"4:1"`, `"1:8"`, `"8:1"`,
  `"2:3"`, `"3:2"`, `"3:4"`, `"4:3"`, `"4:5"`, `"5:4"`, `"9:16"`, `"16:9"` or `"21:9"`;
  unset, the image is square
- `resolution::Union{String,Nothing}`: `"512"`, `"1K"`, `"2K"` or `"4K"`; unset, `"1K"`.
  `gemini-nano-banana-2.1` renders at `"1K"`, `"2K"` or `"4K"`.
- `output_format`: on Gemini `"jpeg"` only, sent as `generationConfig.responseFormat.image.mimeType`
  `IMAGE_JPEG`, the one format Gemini's reference lists (it answers JPEG unasked)
- `delivery::Union{String,Nothing}`: `"inline"` or `"uri"`, Gemini's "delivery mode for the image
  output" (`responseFormat.image.delivery`): the image in the reply, or a URI, which becomes the
  image's `url`. On 2026-11-17 `gemini-nano-banana-2.1` answered `"uri"` with the image inline

A field the request's provider does not take (an OpenAI-only field on Gemini;
`aspect_ratio`, `resolution` or `delivery` on OpenAI), `n` other than 1 on Gemini, a Gemini
`aspect_ratio`, `resolution`, `output_format` or `delivery` outside the values above, and a `safety_identifier` outside
its provider's rule throw `ArgumentError` naming the provider and the field — when the
request is serialized, and in [`generate_image`](@ref) before any request — as do a frame
field without `output_format="gif"`, `svg` without `output_format="svg"`, SVG output with `n`
other than 1, with an `output_compression` or streamed, GIF output with an `output_compression` or
streamed, and a stream with `n` other than 1; `partial_images` outside 0 to 3, or without
`stream=true`, throws when the request is built. Other OpenAI values are sent as given, for OpenAI to
judge. `input_fidelity` is an image-edit
parameter: set it on [`ImageEdit`](@ref), not here.

# Examples
```julia
# Simple prompt
ImageGeneration(prompt="A watercolor painting of a mountain sunset")

# With all options
ImageGeneration(
    prompt="A minimalist logo for a Julia package",
    size="1024x1024",
    quality="high",
    background="transparent",
    output_format="png"
)

# Gemini, wide and at 2K
ImageGeneration(service=GEMINIServiceEndpoint, prompt="A lighthouse at dusk",
                aspect_ratio="16:9", resolution="2K")
```
"""
@kwdef struct ImageGeneration
    service::ServiceEndpointSpec = OPENAIServiceEndpoint
    model::String = ""
    prompt::String
    n::Union{Int,Nothing} = nothing
    size::Union{String,Nothing} = nothing
    quality::Union{String,Nothing} = nothing
    background::Union{String,Nothing} = nothing
    output_format::Union{String,Nothing} = nothing
    output_compression::Union{Int,Nothing} = nothing
    safety_identifier::Union{String,Nothing} = nothing
    moderation::Union{String,Nothing} = nothing
    aspect_ratio::Union{String,Nothing} = nothing
    resolution::Union{String,Nothing} = nothing
    complexity::Union{String,Nothing} = nothing
    frame_count::Union{Int,Nothing} = nothing
    frame_duration_ms::Union{Int,Nothing} = nothing
    svg::Union{SvgOptions,Nothing} = nothing
    stream::Bool = false
    partial_images::Union{Int,Nothing} = nothing
    delivery::Union{String,Nothing} = nothing
    function ImageGeneration(service, model, prompt, n, size, quality, background, output_format, output_compression,
                             safety_identifier, moderation, aspect_ratio, resolution, complexity, frame_count,
                             frame_duration_ms, svg, stream, partial_images, delivery)
        _check_image_controls("OpenAI image", complexity, frame_count, frame_duration_ms)
        _check_partial_images(stream, partial_images)
        new(service, model, prompt, n, size, quality, background, output_format, output_compression,
            safety_identifier, moderation, aspect_ratio, resolution, complexity, frame_count, frame_duration_ms, svg,
            stream, partial_images, delivery)
    end
end

# "The number of partial images to generate. This parameter is used for streaming responses that return
# partial images. Value must be between 0 and 3" (the Images reference; the API answers 9 with HTTP 400).
function _check_partial_images(stream::Bool, partial_images::Union{Integer,Nothing})::Nothing
    isnothing(partial_images) && return nothing
    0 <= partial_images <= 3 || throw(ArgumentError(
        "OpenAI image partial_images must be 0 to 3, the partial images a stream sends (got $partial_images)"))
    stream || throw(ArgumentError(
        "OpenAI image partial_images come only on a stream: set stream=true, or unset partial_images"))
    nothing
end

# The GPT-Image-2.6 controls of the Images API and the image generation tool
# (https://developers.openai.com/api/reference/resources/images/methods/generate), checked where the
# request is built: `who` names the request in the message.
const _IMAGE_COMPLEXITIES = ("low", "medium", "high")
function _check_image_controls(who::String, complexity::Union{AbstractString,Nothing},
                               frame_count::Union{Integer,Nothing}=nothing, frame_duration_ms::Union{Integer,Nothing}=nothing)::Nothing
    isnothing(complexity) || complexity in _IMAGE_COMPLEXITIES || throw(ArgumentError(
        "$who complexity must be \"low\", \"medium\" or \"high\" (got $(repr(complexity)))"))
    isnothing(frame_count) || 2 <= frame_count <= 16 || throw(ArgumentError(
        "$who frame_count must be 2 to 16, the frames of a GIF (got $frame_count)"))
    isnothing(frame_duration_ms) || frame_duration_ms >= 20 || throw(ArgumentError(
        "$who frame_duration_ms must be at least 20, a GIF frame's milliseconds (got $frame_duration_ms)"))
    nothing
end

# A GIF's frame fields are "Only supported with output_format=gif", and `svg` "Requires output_format=svg".
function _check_image_format(who::String, output_format::Union{AbstractString,Nothing}, frame_count::Union{Integer,Nothing},
                             frame_duration_ms::Union{Integer,Nothing}, svg::Union{SvgOptions,Nothing}=nothing)::Nothing
    frames = String[f for (f, v) in (("frame_count", frame_count), ("frame_duration_ms", frame_duration_ms)) if !isnothing(v)]
    isempty(frames) || output_format == "gif" || throw(ArgumentError(
        "$who $(join(frames, " and ")): only with output_format=\"gif\", the frames of a GIF (got output_format=" *
        "$(repr(output_format))); set output_format=\"gif\", or unset $(join(frames, " and "))"))
    isnothing(svg) || output_format == "svg" || throw(ArgumentError(
        "$who svg: only with output_format=\"svg\", the settings of an SVG's author (got output_format=" *
        "$(repr(output_format))); set output_format=\"svg\", or unset svg"))
    nothing
end

# What `generate_image` sends, checked against its provider's rules first: the body
# `JSON.json(ig)` shows is one that may go on the wire.
function JSON.lower(ig::ImageGeneration)
    _check_image_request(ig.service, ig)
    _image_payload(ig.service, ig)
end

# The OpenAI Images API body.
function _image_payload(service, ig::ImageGeneration)::Dict{Symbol,Any}
    model = ig.model
    if isempty(model)
        dm = default_image_model(service)
        isnothing(dm) && throw(ArgumentError("model must be specified for image generation with $(typeof(service))"))
        model = dm
    end
    d = Dict{Symbol,Any}(:model => model, :prompt => ig.prompt)
    for f in (:n, :size, :quality, :background, :output_format, :output_compression, :safety_identifier, :moderation,
              :complexity, :frame_count, :frame_duration_ms, :svg, :partial_images)
        v = getfield(ig, f)
        !isnothing(v) && (d[f] = v)
    end
    ig.stream && (d[:stream] = true)
    return d
end


# ─── Response Types ───────────────────────────────────────────────────────────

"""
    ImageObject

A single generated image from the API response.

# Fields
- `b64_json::Union{String,Nothing}`: Base64-encoded image data; Gemini sends JPEG (the
  `mimeType` of each image part in the response's `raw` says so)
- `revised_prompt::Union{String,Nothing}`: The prompt as revised by the model; on Gemini,
  the text the model sent with the image (`nothing` when it sent none)
- `url::Union{String,Nothing}`: Where to download the image, when the service
  delivered it by URL instead of inline
"""
@kwdef struct ImageObject
    b64_json::Union{String,Nothing} = nothing
    revised_prompt::Union{String,Nothing} = nothing
    url::Union{String,Nothing} = nothing
end

"""
    ImageResponse

Parsed response from the Image Generation API.

# Accessors
- `image_data(r)` — the image payloads: base64 data, or the URL of a URL-delivered image
- `r.created`, `r.data`, `r.usage` — basic fields
- `r.raw` — the complete raw JSON dict

# Fields
- `created::Int64`: Timestamp when the response was created; a Gemini reply carries none,
  so on Gemini it is the time the reply arrived (Unix seconds)
- `data::Vector{ImageObject}`: Generated images
- `usage::Union{Dict{String,Any},Nothing}`: Token usage information under the Images API's
  keys (`input_tokens`, `output_tokens`, `total_tokens`, and their details); Gemini's
  `usageMetadata` is mapped to them as on Gemini Chat, its thinking counted as output and
  as `reasoning_tokens`, its tokens by modality as the details' `text_tokens` and
  `image_tokens`. [`image_usage`](@ref) reads it typed; a streamed call's is its completed event's
- `svg_usage::Union{Dict{String,Any},Nothing}`: an OpenAI SVG's "Token usage for SVG authoring,
  separate from image rendering usage" (`input_tokens`, `output_tokens`, `reasoning_tokens`,
  `total_tokens` and the author `model`), as the reply sends it; `nothing` for any other image.
  OpenAI bills these tokens at "the authoring model's rate", which [`estimated_cost`](@ref) leaves
  out of an image's estimate, with a warning
- `raw::Dict{String,Any}`: Complete raw JSON response; for a streamed call, the stream's completed
  event, which holds the image's `size`, `quality`, `background` and `output_format` as an unstreamed
  reply does
"""
@kwdef struct ImageResponse
    created::Int64
    data::Vector{ImageObject}
    usage::Union{Dict{String,Any},Nothing} = nothing
    svg_usage::Union{Dict{String,Any},Nothing} = nothing
    raw::Dict{String,Any}
end

"""
    PartialImage

One partial image of a streamed OpenAI image generation or edit, as the `callback` of
[`generate_image`](@ref) and [`edit_image`](@ref) receives it: OpenAI's
`image_generation.partial_image` or `image_edit.partial_image` event, decoded.

# Fields
- `index::Int`: its `partial_image_index`, "0-based index for the partial image"
- `bytes::Vector{UInt8}`: the partial image, decoded from the event's base64 `b64_json`
- `size::String`, `quality::String`, `background::String`, `output_format::String`: the
  settings of the image being made, as the event reports them
- `created_at::Int64`: when the event was created, Unix seconds
- `raw::Dict{String,Any}`: the event as OpenAI sent it
"""
@kwdef struct PartialImage
    index::Int
    bytes::Vector{UInt8}
    size::String
    quality::String
    background::String
    output_format::String
    created_at::Int64
    raw::Dict{String,Any}
end


# ─── Result Types ─────────────────────────────────────────────────────────────

"""
    ImageSuccess <: LLMRequestResponse

Successful image generation. Access the parsed response via `.response`.

# Examples
```julia
result = generate_image("A cute robot")
if result isa ImageSuccess
    imgs = image_data(result)   # Vector{String} of base64 images
    save_image(imgs[1], "robot.png")
end
```
"""
@kwdef struct ImageSuccess <: LLMRequestResponse
    response::ImageResponse
end

"""
    ImageFailure <: LLMRequestResponse

HTTP-level failure from the Image Generation API: the raw `response` body, the
`status` code, and the `request_id` the service sent (`x-request-id`/`request-id`
header), if any.
"""
@kwdef struct ImageFailure <: LLMRequestResponse
    response::String
    status::Int
    request_id::Union{String,Nothing} = nothing
end

"""
    ImageCallError <: LLMRequestResponse

Exception-level error during an Image Generation API call (network, timeout, or a
200 that could not be parsed — on Gemini, one that holds no image). `cause` is the
underlying exception — a [`UniLMTimeout`](@ref) for a timeout — and `request_id` is set
when a reply arrived but could not be decoded.
"""
@kwdef struct ImageCallError <: LLMRequestResponse
    error::String
    status::Union{Int,Nothing} = nothing
    request_id::Union{String,Nothing} = nothing
    cause::Union{Nothing,Exception} = nothing
end


# ─── Accessor Functions ──────────────────────────────────────────────────────

"""
    image_data(r::ImageResponse)::Vector{String}
    image_data(r::ImageSuccess)::Vector{String}

The image payloads of a response, one per generated image: the base64 data, or —
for an image the service delivered by URL — its `url` (download it; only base64
data can go to [`save_image`](@ref)). An entry carrying neither is skipped.

On an [`ImageFailure`](@ref) or [`ImageCallError`](@ref) this throws an
[`LLMResultError`](@ref), as [`text`](@ref) does: a call that did not succeed has no
images, and an empty list would read as "zero images generated".

# Examples
```julia
result = generate_image("A sunset over mountains")
imgs = image_data(result)       # Vector{String}
length(imgs)                     # number of images generated
```
"""
image_data(r::ImageResponse)::Vector{String} =
    String[something(img.b64_json, img.url) for img in r.data if !isnothing(img.b64_json) || !isnothing(img.url)]

image_data(r::ImageSuccess) = image_data(r.response)
image_data(r::Union{ImageFailure,ImageCallError}) = throw(LLMResultError(r))

"""
    ImageUsage

The token usage of an image result, typed, as [`image_usage`](@ref) reads it on OpenAI and on Gemini.

# Fields
- `input_tokens::Int`, `output_tokens::Int`, `total_tokens::Int`: the counts [`token_usage`](@ref)
  reads; Gemini's thinking is output, as on Gemini Chat
- `input_text_tokens`, `input_image_tokens::Union{Int,Nothing}`: the input's text and image tokens —
  OpenAI's `input_tokens_details`, Gemini's `promptTokensDetails` by modality
- `output_image_tokens`, `output_text_tokens::Union{Int,Nothing}`: the output's image tokens and the
  rest — OpenAI's `output_tokens_details`; on Gemini the `IMAGE` count of `candidatesTokensDetails`,
  and the other output tokens, text and thinking, which Google prices apart from image output

A split the reply does not give is `nothing`, not a zero that would read as none.
"""
@kwdef struct ImageUsage
    input_tokens::Int
    output_tokens::Int
    total_tokens::Int
    input_text_tokens::Union{Int,Nothing} = nothing
    input_image_tokens::Union{Int,Nothing} = nothing
    output_image_tokens::Union{Int,Nothing} = nothing
    output_text_tokens::Union{Int,Nothing} = nothing
end

"""
    image_usage(r::ImageResponse) -> Union{ImageUsage,Nothing}
    image_usage(r::ImageSuccess) -> Union{ImageUsage,Nothing}

The usage of an image result as an [`ImageUsage`](@ref) — on OpenAI, streamed or not, and on Gemini —
read from `r.usage`. `nothing` when the reply reported no usage, and for an [`ImageFailure`](@ref)
or [`ImageCallError`](@ref).

```julia
result = edit_image("room.png", "Add a window"; quality="low")
u = image_usage(result)
u.input_image_tokens, u.input_text_tokens, u.output_tokens
```
"""
function image_usage(r::ImageResponse)::Union{ImageUsage,Nothing}
    u = r.usage
    isnothing(u) && return nothing
    t = _input_output_usage(u)
    detail(group, key) = (g = get(u, group, nothing); v = g isa AbstractDict ? get(g, key, nothing) : nothing;
                          v isa Integer ? Int(v) : nothing)
    ImageUsage(input_tokens=t.prompt_tokens, output_tokens=t.completion_tokens, total_tokens=t.total_tokens,
               input_text_tokens=detail("input_tokens_details", "text_tokens"),
               input_image_tokens=detail("input_tokens_details", "image_tokens"),
               output_image_tokens=detail("output_tokens_details", "image_tokens"),
               output_text_tokens=detail("output_tokens_details", "text_tokens"))
end
image_usage(r::ImageSuccess) = image_usage(r.response)
image_usage(::Union{ImageFailure,ImageCallError})::Nothing = nothing

# What `showerror(::LLMResultError)` reports for an image result.
_llm_result_status(r::Union{ImageFailure,ImageCallError}) = r.status
_llm_result_body(r::ImageFailure) = r.response
_llm_result_body(r::ImageCallError) = r.error

"""
    save_image(img_b64::String, filepath::String)

Decode a base64-encoded image and save it to a file.

Decoding happens **before** `filepath` is opened, so a malformed payload raises
without touching whatever already lives there — a failed save costs nothing.

# Examples
```julia
result = generate_image("A watercolor landscape")
if result isa ImageSuccess
    save_image(image_data(result)[1], "landscape.png")
end
```
"""
function save_image(img_b64::String, filepath::String)
    # Decode BEFORE opening: `open(…, "w")` truncates, so decoding inside the block
    # let a malformed payload destroy whatever already lived at `filepath` and leave
    # a 0-byte stub. A decode failure must cost nothing.
    bytes = base64decode(img_b64)
    write(filepath, bytes)
    return filepath
end


# ─── Parsing ─────────────────────────────────────────────────────────────────

function parse_image_response(resp::HTTP.Response)::ImageResponse
    data = JSON.parse(resp.body; dicttype=Dict{String,Any})
    images = [
        ImageObject(
            b64_json=get(img, "b64_json", nothing),
            revised_prompt=get(img, "revised_prompt", nothing),
            url=get(img, "url", nothing)
        )
        for img in get(data, "data", Any[])
    ]
    ImageResponse(
        created=data["created"],
        data=images,
        usage=get(data, "usage", nothing),
        svg_usage=get(data, "svg_usage", nothing),
        raw=data
    )
end


# ─── Request Functions ───────────────────────────────────────────────────────

"""
    generate_image(ig::ImageGeneration; config=nothing, callback=nothing, cancel=nothing)

Send an image-generation request: to OpenAI's Images API, or, for
`service=GEMINIServiceEndpoint`, to Gemini's `models/{model}:generateContent`, asking
for image output only and sizing the image by `imageConfig` when `aspect_ratio` or
`resolution` is set ([Gemini image generation](https://ai.google.dev/gemini-api/docs/image-generation)).

Returns [`ImageSuccess`](@ref), [`ImageFailure`](@ref), or [`ImageCallError`](@ref).
A Gemini reply's image parts become its [`ImageObject`](@ref)s, in order, each carrying
the text the model sent, if any, as `revised_prompt`; the interim images of the model's
thinking are not among them. A Gemini 200 that holds no image is an `ImageCallError`
naming Gemini's finish reason (such as `IMAGE_SAFETY` or `NO_IMAGE`).

With `ig.stream` (OpenAI), the reply is OpenAI's image stream: `callback(p::PartialImage)` runs
once per partial image, in order, on the calling task, and the call returns once the image
arrives — the same `ImageSuccess`, its `created`, `usage` and `raw` read from the stream's
completed event, priced as an unstreamed call's. A streamed call is sent again after a retryable
status or a failure before the reply's head, never after it. What `callback` throws ends the call
with an `ImageCallError` whose `cause` is that exception; its time does not count toward
`stream_idle_timeout`. A stream that ends before its completed event, an `error` event, and an event
without a member OpenAI's reference requires are an `ImageCallError`.

Throws `ArgumentError` before any network I/O when `ig.service` does not declare `:images`
(an endpoint type that declares no capabilities at all included), when `callback` is given without
`ig.stream`, and when the request sets
a field its provider does not take, `n` other than 1 on Gemini, a Gemini `aspect_ratio`
or `resolution` outside its list, or a `safety_identifier` outside its provider's rule
(see [`ImageGeneration`](@ref)). Other OpenAI values are sent as given.

# Examples
```julia
ig = ImageGeneration(prompt="A cute robot learning Julia", quality="high")
result = generate_image(ig)
if result isa ImageSuccess
    println("Generated \$(length(result.response.data)) image(s)")
    save_image(image_data(result)[1], "robot.png")
end
```

Pass `config::Union{Nothing,RequestConfig}` to override the timeout/retry budget for this call.

Pass `cancel::Union{Nothing,CancelToken}` (default: the ambient [`with_cancel`](@ref)
token) to make the call cancellable: once the token is cancelled it returns an
`ImageCallError` whose `cause` is a [`UniLMCancelled`](@ref) — without sending anything
when the token was already cancelled, at once when mid-request (never retried).
"""
function generate_image(ig::ImageGeneration; config::Union{Nothing,RequestConfig}=nothing, callback=nothing,
                        cancel::Union{Nothing,CancelToken}=nothing)
    _check_callback(ig, callback)
    validate_capability(ig.service, :images, "Image Generation API")
    _check_image_request(ig.service, ig)
    ig.stream ? _image_stream_call(ig.service, ig, callback, config, cancel) : _image_call(ig.service, ig, config, cancel)
end

"""
    generate_image(prompt::String; kwargs...)

Convenience method: create an [`ImageGeneration`](@ref) from a prompt + keyword arguments
and send it. `config`, `cancel` and `callback` go to the request, as for `generate_image(ig)`.

# Examples
```julia
# Simple generation
result = generate_image("A watercolor painting of a Julia butterfly")

# With options
result = generate_image(
    "A minimalist logo",
    size="1024x1024",
    quality="high",
    background="transparent"
)

# Gemini
result = generate_image("A watercolor painting of a Julia butterfly";
                        service=GEMINIServiceEndpoint, aspect_ratio="3:2")

# Save to file
if result isa ImageSuccess
    save_image(image_data(result)[1], "logo.png")
end
```
"""
function generate_image(prompt::String; config::Union{Nothing,RequestConfig}=nothing, callback=nothing,
                        cancel::Union{Nothing,CancelToken}=nothing, kwargs...)
    generate_image(ImageGeneration(; prompt=prompt, kwargs...); config, callback, cancel)
end

# The callback receives the partial images of a stream, and runs on nothing else: given without one, it
# would never run.
_check_callback(r, callback)::Nothing = r.stream || isnothing(callback) ? nothing :
    throw(ArgumentError("callback requires stream=true: it receives the partial images of a stream"))


# ─── Image Edits ──────────────────────────────────────────────────────────────

"""
    ImageRef(; file_id=nothing, image_url=nothing)

An input image of an OpenAI image edit given by reference, not read from a local file: the id of
an image uploaded with [`upload_file`](@ref) (purpose `"vision"`), or `image_url`, "A fully
qualified URL or base64-encoded data URL" (the edit reference), such as
`"data:image/png;base64,…"`. Exactly one of the two: the API answers a reference with both
"Exactly one of 'image_url' or 'file_id' must be provided." (HTTP 400, 2026-11-17), and this
constructor throws `ArgumentError` for that, for neither, for an empty `file_id`, and for an
`image_url` with no URL scheme or longer than the 20971520 characters the API takes.

An [`ImageEdit`](@ref) whose `image` is an `ImageRef`, or a vector of them, goes to OpenAI as the
JSON edit, with its `mask` and `markup` by reference too.

```julia
up = upload_file("room.png", "vision")
edit_image(ImageRef(file_id=up.response.id), "Add a large window")
```
"""
struct ImageRef
    file_id::Union{String,Nothing}
    image_url::Union{String,Nothing}
    function ImageRef(; file_id::Union{AbstractString,Nothing}=nothing, image_url::Union{AbstractString,Nothing}=nothing)
        isnothing(file_id) == isnothing(image_url) && throw(ArgumentError(
            "an ImageRef names exactly one of file_id and image_url (got $(isnothing(file_id) ? "neither" : "both")): " *
            "\"Exactly one of 'image_url' or 'file_id' must be provided.\""))
        if !isnothing(file_id)
            isempty(file_id) && throw(ArgumentError("an ImageRef's file_id is empty: pass the id of an uploaded file"))
            return new(String(file_id), nothing)
        end
        url = String(image_url)
        occursin(r"^[A-Za-z][A-Za-z0-9+.-]*:", url) || throw(ArgumentError(
            "an ImageRef's image_url is \"A fully qualified URL or base64-encoded data URL\", and $(repr(first(url, 80))) " *
            "names no scheme: pass a local file's path as the edit's image instead"))
        length(url) <= 20_971_520 || throw(ArgumentError(
            "an ImageRef's image_url takes at most 20971520 characters (got $(length(url))): upload the image " *
            "(upload_file(path, \"vision\")) and pass ImageRef(file_id=…)"))
        new(nothing, url)
    end
end

JSON.lower(r::ImageRef) = isnothing(r.file_id) ? Dict(:image_url => r.image_url) : Dict(:file_id => r.file_id)

"""
    ImageEdit(; image, prompt, service=OPENAIServiceEndpoint, model="", kwargs...)

An image-edit request for [`edit_image`](@ref): the image(s) at `image` changed as `prompt` asks.
`image` is a path, a vector of paths, or, on OpenAI, an [`ImageRef`](@ref) or a vector of them:
an uploaded file's id or an image URL. OpenAI is sent the files as multipart/form-data at
`/v1/images/edits`, and the references as the JSON edit at the same path, "Use `images` (array of
`ImageRefParam`) instead of multipart `image` uploads" (its API specification); a request is one form
or the other, so it gives its images, `mask` and `markup` all as files or all by reference.
`GEMINIServiceEndpoint` is sent `generateContent` with each image as an inline part ahead of the prompt,
its format read from the file's bytes (PNG, JPEG, GIF or WebP). `model` left `""` is the service's
default image model (`"gpt-image-2.6-sunburst"` on OpenAI, `"gemini-nano-banana-2.1"` on Gemini),
resolved when the edit is sent.

# Fields — both providers
`image` and `model` (above); `service`, `prompt`, `n` and `safety_identifier`, as on
[`ImageGeneration`](@ref): Gemini takes `n=1` only, and local files only.

# Fields — OpenAI only
- `mask::Union{String,ImageRef,Nothing}`: a mask whose transparent area marks what to edit, by path or,
  in an edit by reference, as an `ImageRef`; not with `output_format="gif"`
- `markup::Union{String,Vector{UInt8},ImageRef,Nothing}`: a PNG, by path or as its bytes, sent as the form
  part `markup`, or in an edit by reference an `ImageRef`: "A transparent PNG overlaid on the first image to guide the edit. Must match that
  image's dimensions after orientation and contain transparent pixels" (the edit reference). It
  "Cannot be combined with mask": set beside one, [`edit_image`](@ref) throws `ArgumentError`, as it
  does for bytes that are no PNG. OpenAI states it "Supports GPT Image 2.5 and 2.6" (its API
  specification); the model is OpenAI's to check. On 2026-11-16 the API took a multipart markup edit on
  `gpt-image-2.6-flare` (HTTP 200)
- `size`, `quality`, `background`, `output_format`, `output_compression`, `moderation`, `stream`,
  `partial_images`: as on [`ImageGeneration`](@ref), each a field of the edit; a streamed edit's
  partial images reach the `callback` of [`edit_image`](@ref)
- `input_fidelity::Union{String,Nothing}`: how closely the edit keeps the input's
  features (`"high"`, `"low"`)
- `complexity`, `frame_count`, `frame_duration_ms`: as on [`ImageGeneration`](@ref)
- `svg::Union{SvgOptions,Nothing}`: as on [`ImageGeneration`](@ref), with `output_format="svg"` only, sent as
  the form fields `svg[model]` and `svg[reasoning][effort]` ("Use multipart fields such as `svg[model]`", the
  edit reference). OpenAI's guide says "The image edit endpoint does not support SVG output", but on
  2026-11-16 the API answered such an edit with HTTP 200, an SVG and its `svg_usage`; an edit by reference
  sends `svg` as its JSON object. SVG output takes `n=1` and no `output_compression`; a `mask` beside `svg`
  is sent as given, since no answer of OpenAI's shows a rule for the pair

# Fields — Gemini only
`aspect_ratio`, `resolution`, `output_format` (`"jpeg"`) and `delivery`, as on
[`ImageGeneration`](@ref); unset, the edited image follows the input's aspect ratio.

A vector that mixes paths and `ImageRef`s throws `ArgumentError` when the edit is built. `image` names
at least one image: [`edit_image`](@ref) throws `ArgumentError` for an empty
vector before any request, as it does for a `markup` beside a `mask`, a `mask` or `markup` of the other
kind than the images (a file in an edit by reference, a reference in an edit of files) and, naming the provider and the
field, for a field the request's provider does not take, an `ImageRef` on Gemini, `n` other than 1 on Gemini, a Gemini `aspect_ratio` or
`resolution` outside its list, a `safety_identifier` outside its provider's rule, a frame field
without `output_format="gif"`, a `mask`, an `output_compression` or a stream with it, `svg` without
`output_format="svg"`, SVG output with `n` other than 1, with an `output_compression` or streamed, and a
stream with `n` other than 1; `partial_images` outside 0 to 3, or without `stream=true`, throws when the
edit is built. Other OpenAI values are sent as given, for OpenAI to judge.
"""
@kwdef struct ImageEdit
    service::ServiceEndpointSpec = OPENAIServiceEndpoint
    image::Union{String,ImageRef,Vector{String},Vector{ImageRef}}
    prompt::String
    mask::Union{String,ImageRef,Nothing} = nothing
    markup::Union{String,Vector{UInt8},ImageRef,Nothing} = nothing
    model::String = ""
    n::Union{Int,Nothing} = nothing
    size::Union{String,Nothing} = nothing
    quality::Union{String,Nothing} = nothing
    input_fidelity::Union{String,Nothing} = nothing
    background::Union{String,Nothing} = nothing
    output_format::Union{String,Nothing} = nothing
    safety_identifier::Union{String,Nothing} = nothing
    aspect_ratio::Union{String,Nothing} = nothing
    resolution::Union{String,Nothing} = nothing
    complexity::Union{String,Nothing} = nothing
    frame_count::Union{Int,Nothing} = nothing
    frame_duration_ms::Union{Int,Nothing} = nothing
    svg::Union{SvgOptions,Nothing} = nothing
    output_compression::Union{Int,Nothing} = nothing
    moderation::Union{String,Nothing} = nothing
    stream::Bool = false
    partial_images::Union{Int,Nothing} = nothing
    delivery::Union{String,Nothing} = nothing
    function ImageEdit(service, image, prompt, mask, markup, model, n, size, quality, input_fidelity, background,
                       output_format, safety_identifier, aspect_ratio, resolution, complexity, frame_count,
                       frame_duration_ms, svg, output_compression, moderation, stream, partial_images, delivery)
        _check_image_controls("OpenAI image", complexity, frame_count, frame_duration_ms)
        _check_partial_images(stream, partial_images)
        new(service, _edit_inputs(image), prompt, mask, markup, model, n, size, quality, input_fidelity, background,
            output_format, safety_identifier, aspect_ratio, resolution, complexity, frame_count, frame_duration_ms,
            svg, output_compression, moderation, stream, partial_images, delivery)
    end
end

# An edit's input images: local files (paths) or references, never both, since each of OpenAI's two edit
# forms carries one kind — multipart/form-data uploads files, the JSON edit references images.
_edit_inputs(image::AbstractString) = String(image)
_edit_inputs(image::ImageRef) = image
_edit_inputs(images::Union{Vector{String},Vector{ImageRef}}) = images
function _edit_inputs(images::AbstractVector)
    all(x -> x isa AbstractString, images) && return String[images...]
    all(x -> x isa ImageRef, images) && return ImageRef[images...]
    throw(ArgumentError("an image edit's inputs are all local files (paths) or all references (ImageRef), not both: " *
        "OpenAI uploads files in a multipart edit and references images in a JSON edit; upload the files " *
        "(upload_file(path, \"vision\")) and pass an ImageRef(file_id=…) for each, or pass every image as a path"))
end
_edit_inputs(image) = image   # anything else fails the field's conversion, naming its type

# Whether an edit goes as OpenAI's JSON edit: its images are references.
_by_reference(e::ImageEdit)::Bool = e.image isa Union{ImageRef,Vector{ImageRef}}

"""
    edit_image(e::ImageEdit; config=nothing, callback=nothing, cancel=nothing) -> LLMRequestResponse
    edit_image(image, prompt; mask=nothing, service=OPENAIServiceEndpoint, config=nothing, callback=nothing,
               cancel=nothing, kwargs...)

Edit/extend image(s) under a text prompt — on OpenAI's Images API, or on Gemini's
`generateContent` for `service=GEMINIServiceEndpoint`. Returns the same
`ImageSuccess`/`ImageFailure`/`ImageCallError` shapes as [`generate_image`](@ref), and a
Gemini reply decodes the same way. A streamed edit (`e.stream`, OpenAI) hands its partial images
to `callback` and returns as a streamed generation does.

Throws `ArgumentError` before any network I/O when `e.service` does not declare
`:image_edits` (an endpoint type that declares no capabilities at all included), when `callback`
is given without `e.stream`, when `e.image`
is an empty vector, when an input image, the mask or a markup path
is no file (as [`upload_file`](@ref) and [`transcribe`](@ref) refuse one), when an OpenAI edit sets both
`mask` and `markup`, gives markup bytes that are no PNG, or gives a mask or markup of the other kind than
its images (files and [`ImageRef`](@ref)s do not mix), and when the request sets a
field its provider does not take, an `ImageRef` on Gemini, `n` other than 1 on Gemini, a Gemini `aspect_ratio` or
`resolution` outside its list, or a `safety_identifier` outside its provider's rule (see
[`ImageEdit`](@ref)). Other OpenAI values are sent as given.

Pass `config::Union{Nothing,RequestConfig}` to override the timeout/retry budget for this
call, and `cancel::Union{Nothing,CancelToken}` to make it cancellable exactly as for
[`generate_image`](@ref): a cancelled call returns an `ImageCallError` whose `cause` is a
[`UniLMCancelled`](@ref).
"""
function edit_image(e::ImageEdit; config::Union{Nothing,RequestConfig}=nothing, callback=nothing,
                    cancel::Union{Nothing,CancelToken}=nothing)
    _check_callback(e, callback)
    validate_capability(e.service, :image_edits, "Image Edits API")
    # An edit needs an input image: sent without one, Gemini would read the prompt alone as a
    # text-to-image request.
    e.image isa Vector && isempty(e.image) && throw(ArgumentError(
        "an image edit needs at least one input image, and `image` is an empty vector: pass a path, an ImageRef, " *
        "or a vector of either"))
    _check_image_request(e.service, e)
    _check_edit_files(e)
    e.stream ? _image_stream_call(e.service, e, callback, config, cancel) : _image_call(e.service, e, config, cancel)
end
edit_image(image, prompt::String; mask::Union{String,ImageRef,Nothing}=nothing,
    service::ServiceEndpointSpec=OPENAIServiceEndpoint, config::Union{Nothing,RequestConfig}=nothing,
    callback=nothing, cancel::Union{Nothing,CancelToken}=nothing, kwargs...) =
    edit_image(ImageEdit(; service=service, image=image, prompt=prompt, mask=mask, kwargs...); config, callback, cancel)


# ─── The exchange, and the wire each service speaks ──────────────────────────
# Three seams dispatch on the service: `_check_image_request` (the provider's rules,
# before any I/O), `_image_request` (url, headers and body of the one request) and
# `_image_reply` (a 200 reply → ImageResponse). The OpenAI Images API is the default
# for every endpoint; GEMINIServiceEndpoint has methods of its own.

const _ImageRequest = Union{ImageGeneration,ImageEdit}

# Build, send through the retry seam, decode a 200. Whatever fails from here on is the
# call's result (an ImageCallError), not a throw.
function _image_call(service, r::_ImageRequest, config::Union{Nothing,RequestConfig},
                     cancel::Union{Nothing,CancelToken})
    cfg = _resolve_config(config)
    tok = _resolve_cancel(cancel)
    t0 = time_ns()
    local resp
    try
        # A cancelled token reads no input image: a Gemini edit reads each one into its JSON body as the
        # request is built, before the retry loop checks the token.
        iscancelled(tok) && throw(UniLMCancelled(:token, _elapsed_s(t0)))
        url, headers, body = _image_request(service, r)
        resp = _http_with_retries(cfg, t0, "POST", url, headers, body; cancel=tok)
        return resp.status == 200 ?
               ImageSuccess(response=_image_reply(service, resp)) :
               _failure(ImageFailure, resp)
    catch err
        err isa InterruptException && rethrow()
        return _callerr(ImageCallError, err; request_id=_platform_request_id(@isdefined(resp) ? resp : nothing))
    end
end

# ── Streamed images (OpenAI) ──
# OpenAI's image stream: per partial image an `image_generation.partial_image` (an edit's
# `image_edit.partial_image`) event, then the image in a `….completed` event, its usage beside it (the
# Images reference's ImageGenStreamEvent and ImageEditStreamEvent).

_stream_event_kind(::ImageGeneration) = "image_generation"
_stream_event_kind(::ImageEdit) = "image_edit"

# A member a streamed event requires, of type `T`: an event without it is malformed, and ends the call.
function _event_field(d::Dict{String,Any}, key::String, ::Type{T}) where {T}
    v = get(d, key, nothing)
    v isa T || error("OpenAI's $(get(d, "type", "image stream")) event has no $key",
                     isnothing(v) ? "" : " of the documented type (got $(repr(v)))")
    v
end

# One event of an image stream. A partial image goes to `partial`; the completed event is the reply — its
# image, its creation time and its usage, `raw` the event as sent. An error event ends the call, as does a
# payload that is no JSON object or an event without a member the reference requires; an event of another
# kind is not read.
function _image_stream_event(r::_ImageRequest, partial, payload::AbstractString)::Union{Nothing,ImageResponse}
    JSON.isvalidjson(payload) || error("OpenAI's image stream sent a payload that is no JSON: ", repr(first(payload, 200)))
    d = JSON.parse(payload; dicttype=Dict{String,Any})
    d isa Dict{String,Any} ||
        error("OpenAI's image stream sent a payload that is no JSON object: ", repr(first(payload, 200)))
    kind, prefix = get(d, "type", nothing), _stream_event_kind(r)
    if kind == "$prefix.partial_image"
        partial(PartialImage(index=_event_field(d, "partial_image_index", Integer),
                             bytes=base64decode(_event_field(d, "b64_json", String)),
                             size=_event_field(d, "size", String), quality=_event_field(d, "quality", String),
                             background=_event_field(d, "background", String),
                             output_format=_event_field(d, "output_format", String),
                             created_at=_event_field(d, "created_at", Integer), raw=d))
    elseif kind == "$prefix.completed"
        return ImageResponse(created=_event_field(d, "created_at", Integer),
                             data=[ImageObject(b64_json=_event_field(d, "b64_json", String))],
                             usage=Dict{String,Any}(_event_field(d, "usage", AbstractDict)), raw=d)
    elseif kind == "error" || haskey(d, "error")
        error("OpenAI's image stream reported an error: ", payload)
    end
    nothing
end

# A streamed request's URL, headers and body. A multipart edit's form is read into its bytes, under the
# content type that names its boundary; the stream is read as the server wrote it.
function _image_stream_request(service, r::_ImageRequest)
    url, headers, body = _image_request(service, r)
    headers = [headers; "Accept-Encoding" => "identity"]
    body isa _BodyFactory || return url, headers, body
    form = body.build()
    url, [headers; "Content-Type" => HTTP.content_type(form)], read(form)
end

# A streamed image call: attempts as `cfg` allows, with an unstreamed call's backoff (`_retry_pause`). An
# attempt is made again only after a retryable status, or a retryable failure before the reply's head: from the
# head on, a partial image may have reached the callback, and another attempt would bill another image.
function _image_stream_call(service, r::_ImageRequest, callback, config::Union{Nothing,RequestConfig},
                            cancel::Union{Nothing,CancelToken})
    cfg = _resolve_config(config)
    tok = _resolve_cancel(cancel)
    t0 = time_ns()
    attempt = 0
    while true
        attempt += 1
        # A cancelled token reads no input image: the attempt builds the request.
        iscancelled(tok) && return _callerr(ImageCallError, UniLMCancelled(:token, _elapsed_s(t0)))
        _remaining_s(cfg, t0) > 0 ||
            return _callerr(ImageCallError, UniLMTimeout(:deadline, _elapsed_s(t0), cfg.total_deadline))
        out = _image_stream_attempt(service, r, callback, cfg, t0, tok)
        (out.retry && attempt < cfg.max_attempts) || return out.result
        action, delay = _retry_pause(cfg, t0, attempt, out.resp)
        action === :sleep || return out.result
        @debug "image stream attempt failed; retrying" attempt delay
        _cancel_sleep(tok, delay) && return _callerr(ImageCallError, UniLMCancelled(:token, _elapsed_s(t0)))
    end
end

# One attempt: the request written and the head read within the request bound, then the events, each partial
# image handed to `callback` between reads (its time is no byte gap; what it throws ends the call). Returns the
# result, whether another attempt may be made, and the reply whose wait a retry honours.
function _image_stream_attempt(service, r::_ImageRequest, callback, cfg::RequestConfig, t0::UInt64,
                               tok::Union{Nothing,CancelToken})
    idle = Ref{Union{Nothing,_IdleGuard}}(nothing)        # armed when the head arrives
    bound = Ref{Union{Nothing,UniLMTimeout}}(nothing)     # the request-phase bound, recorded as it fires
    thrown = Ref{Any}(nothing)                            # what the callback threw
    head = Ref{Union{Nothing,HTTP.Response}}(nothing)
    reply = Ref{Union{Nothing,ImageResponse}}(nothing)
    failed = IOBuffer()                                   # the body of a non-200 reply
    ctx = HTTP.RequestContext()
    function partial(p::PartialImage)::Nothing
        isnothing(callback) && return nothing
        _enter_user!(idle[])
        try
            callback(p)
        catch e
            e isa InterruptException || (thrown[] = e)
            rethrow()
        finally
            _exit_user!(idle[])
        end
        nothing
    end
    function events!(carry::IOBuffer, name::Ref{String}, chunk::String)::Nothing
        for (_, payload) in _sse_events!(carry, name, chunk)
            reply[] = _image_stream_event(r, partial, payload)
            isnothing(reply[]) || return nothing
        end
        nothing
    end
    try
        url, headers, body = _image_stream_request(service, r)
        resp = _http_open("POST", url, headers; cfg, t0, cancel=tok, context=ctx, decompress=false) do io
            h = _with_recorded_deadline(() -> begin
                    write(io, body)
                    HTTP.closewrite(io)
                    HTTP.startread(io)
                end, () -> _abort_request_phase(io, ctx), min(_remaining_s(cfg, t0), cfg.request_timeout), :request,
                bound)
            head[] = h
            # From the head on, only the byte-gap bound runs: a long healthy stream is no failure.
            idle[] = _idle_guard(() -> close(io), cfg.stream_idle_timeout)
            carry, name = IOBuffer(), Ref("")
            while isnothing(reply[]) && !iscancelled(tok) && !eof(io)
                chunk = String(readavailable(io))
                _touch!(idle[])
                h.status == 200 ? events!(carry, name, chunk) : write(failed, chunk)
            end
            iscancelled(tok) && throw(UniLMCancelled(:token, _elapsed_s(t0)))
            h.status == 200 || return nothing
            # A last line the server never newline-terminated is still a line.
            if isnothing(reply[])
                tail = takestring!(carry)
                isempty(tail) || events!(carry, name, tail * "\n")
            end
            if isnothing(reply[])
                # A guard's close landing between reads ends the loop at a clean EOF.
                breach = _exit_breach(idle[], bound, cfg)
                breach === nothing || throw(breach)
                error("OpenAI's image stream ended before its completed event")
            end
            _await_body_end(io)
            HTTP.closeread(io)
            nothing
        end
        resp.status == 200 && return (result=ImageSuccess(response=something(reply[])), retry=false, resp=resp)
        failure = ImageFailure(response=takestring!(failed), status=resp.status, request_id=_platform_request_id(resp))
        (result=failure, retry=_is_retryable(failure.status, failure.response), resp=resp)
    catch e
        _find_exception(x -> x isa InterruptException, e) !== nothing && rethrow()
        rid = _platform_request_id(head[])
        u = thrown[]
        isnothing(u) || return (result=ImageCallError(error=_error_text(u), request_id=rid,
                                                      cause=(u isa Exception ? u : nothing)), retry=false, resp=nothing)
        # Cancelled: whatever HTTP.jl surfaced is the abort's echo. Otherwise a typed bound (the byte gap, then the
        # request phase) wins over its teardown noise.
        cause = iscancelled(tok) ? UniLMCancelled(:token, _elapsed_s(t0)) :
            something(_classify_stream_timeout(e, idle[], cfg, t0), bound[],
                      _map_native_timeout(e, cfg, min(_remaining_s(cfg, t0), cfg.request_timeout), t0), Some(e))
        retry = isnothing(head[]) && !(cause isa UniLMCancelled) && _retryable_exception(cause)
        (result=_callerr(ImageCallError, cause; request_id=rid), retry=retry, resp=nothing)
    finally
        _disarm!(idle[])
    end
end

# The names of the fields of `r`, among `fields`, that are set: not `nothing`, and `stream` only when true.
_set_image_fields(r::_ImageRequest, fields)::Vector{String} =
    String[String(f) for f in fields if (v = getfield(r, f); !(isnothing(v) || v === false))]

# An edit's input paths, each a file: a missing one is the caller's error, which `edit_image`
# reports before any request (`_check_edit_files`), as `upload_file` and `transcribe` do. An edit
# by reference has none: its request is built from the references (`_edit_json`).
_edit_images(e::ImageEdit)::Vector{String} = _edit_files(e.image)
_edit_files(image::String)::Vector{String} = _edit_files([image])
function _edit_files(images::Vector{String})::Vector{String}
    absent = findfirst(!isfile, images)
    isnothing(absent) || throw(ArgumentError("image not found: $(images[absent])"))
    images
end
_edit_files(::Union{ImageRef,Vector{ImageRef}}) =
    throw(ArgumentError("an image edit by reference (ImageRef) reads no local file"))

function _check_edit_files(e::ImageEdit)::Nothing
    _by_reference(e) && return nothing
    _edit_images(e)
    mask = e.mask
    mask isa String && !isfile(mask) && throw(ArgumentError("mask not found: $mask"))
    markup = e.markup
    markup isa String && !isfile(markup) && throw(ArgumentError("markup not found: $markup"))
    nothing
end

# ── OpenAI Images API (the default) ──

# The Images API has no aspect-ratio or resolution field: `size` names both; and no delivery mode: its image
# models return base64 data. A safety identifier is 1 to 128 characters: OpenAI's guide says to keep it to 128 or
# fewer, and the API reads an empty one as none
# (https://developers.openai.com/api/docs/guides/safety-best-practices#implement-safety-identifiers).
function _check_image_request(service, r::_ImageRequest)
    gemini = _set_image_fields(r, (:aspect_ratio, :resolution, :delivery))
    isempty(gemini) || throw(ArgumentError(
        "the OpenAI Images API has no $(join(gemini, " or ")), which only Gemini takes" *
        (any(in(("aspect_ratio", "resolution")), gemini) ? "; set size=\"WIDTHxHEIGHT\" instead" : "")))
    id = r.safety_identifier
    isnothing(id) || 1 <= length(id) <= 128 || throw(ArgumentError(
        "the OpenAI Images API takes a safety_identifier of 1 to 128 characters (got $(length(id))); it reads " *
        "an empty one as none, and a longer identifier can be hashed (a SHA-256 hex digest is 64 characters)"))
    _check_image_format("OpenAI image", r.output_format, r.frame_count, r.frame_duration_ms, _svg(r))
    r.stream && !isnothing(r.n) && r.n != 1 && throw(ArgumentError(
        "OpenAI streams one image per request (got n=$(r.n)): \"Streaming is only supported with n=1.\", its HTTP 400 " *
        "for n=2; unset n, or send one request per image"))
    _check_svg_output(r)
    _check_gif_output(r)
    _check_markup(r)
    _check_edit_form(r)
end
_svg(r::_ImageRequest)::Union{SvgOptions,Nothing} = r.svg

# OpenAI's two edit forms: multipart/form-data uploads files, the JSON edit takes references
# (`ImageRefParam`) for its images, mask and markup alike. A request is one or the other.
_check_edit_form(::ImageGeneration)::Nothing = nothing
function _check_edit_form(e::ImageEdit)::Nothing
    byref = _by_reference(e)
    odd = String[String(f) for f in (:mask, :markup) if (v = getfield(e, f); !isnothing(v) && (v isa ImageRef) != byref)]
    isempty(odd) && return nothing
    throw(ArgumentError(byref ?
        "an OpenAI image edit by reference (ImageRef images, sent as the JSON edit) takes its $(join(odd, " and ")) " *
        "by reference too: pass an ImageRef, a file id or an image URL such as a data URL" :
        "an OpenAI image edit of local files (sent as multipart/form-data) takes its $(join(odd, " and ")) as a file: " *
        "pass a path, or give every image as an ImageRef"))
end

# "SVG generation requires n=1 and does not support streaming, partial images, prompt_rewriter, or
# output_compression" (the Images reference).
function _check_svg_output(r::_ImageRequest)::Nothing
    r.output_format == "svg" || return nothing
    isnothing(r.n) || r.n == 1 || throw(ArgumentError(
        "OpenAI SVG output takes n=1 (got n=$(r.n)): \"SVG generation requires n=1\"; send one request per SVG"))
    isnothing(r.output_compression) || throw(ArgumentError(
        "OpenAI SVG output takes no output_compression: SVG generation \"does not support ... output_compression\"; " *
        "unset output_compression"))
    r.stream && throw(ArgumentError(
        "OpenAI SVG output does not stream: SVG generation \"does not support streaming, partial images\"; " *
        "unset stream and partial_images"))
    nothing
end

# "GIF output does not support streaming, partial images, masks, or custom output compression" (the Images
# reference): the API answers an output_compression on a GIF generation and a mask on a GIF edit with HTTP 400
# (2026-11-16).
_gif_refused_fields(::ImageGeneration) = (:output_compression, :stream, :partial_images)
_gif_refused_fields(::ImageEdit) = (:mask, :output_compression, :stream, :partial_images)
function _check_gif_output(r::_ImageRequest)::Nothing
    r.output_format == "gif" || return nothing
    set = _set_image_fields(r, _gif_refused_fields(r))
    isempty(set) || throw(ArgumentError(
        "OpenAI GIF output takes no $(join(set, " or ")): \"GIF output does not support streaming, partial images, " *
        "masks, or custom output compression\"; unset $(join(set, " and ")), or choose another output_format"))
    nothing
end

# OpenAI's edit markup "Cannot be combined with mask", and is "A transparent PNG": bytes that are no PNG
# would go out under a content type they do not have.
_check_markup(::ImageGeneration)::Nothing = nothing
function _check_markup(e::ImageEdit)::Nothing
    markup = e.markup
    isnothing(markup) && return nothing
    isnothing(e.mask) || throw(ArgumentError("an OpenAI image edit takes a mask or a markup, not both: the " *
        "markup \"Cannot be combined with mask\"; unset one of them"))
    markup isa Vector{UInt8} && _sniff_image_mime(markup) != "image/png" && throw(ArgumentError(
        "an image edit's markup is \"A transparent PNG\", and these bytes are no PNG: pass a PNG's bytes or its path"))
    nothing
end

# The markup's form part: a file under its own name and type, as the mask is sent, or PNG bytes.
_markup_part(path::String) = HTTP.Multipart(basename(path), IOBuffer(read(path)), _mime_for(path))
_markup_part(png::Vector{UInt8}) = HTTP.Multipart("markup.png", IOBuffer(png), "image/png")

_image_request(service, ig::ImageGeneration) =
    (_api_base_url(service) * IMAGES_GENERATIONS_PATH, auth_header(service), JSON.json(_image_payload(service, ig)))

function _image_request(service, e::ImageEdit)
    model = isempty(e.model) ? something(default_image_model(service), "") : e.model
    isempty(model) && throw(ArgumentError("model must be specified for image edits with $(typeof(service))"))
    _by_reference(e) &&
        return (_api_base_url(service) * IMAGES_EDITS_PATH, auth_header(service), JSON.json(_edit_json(e, model)))
    images = _edit_images(e)
    # A Form is consumed by the attempt that sends it, so the retry loop gets a
    # factory: every attempt re-reads the images into its own multipart body.
    body = _BodyFactory() do
        parts = Pair{String,Any}["prompt" => e.prompt, "model" => model]
        for img in images
            push!(parts, "image[]" => HTTP.Multipart(basename(img), IOBuffer(read(img)), _mime_for(img)))
        end
        # Bind the optional mask to a local before testing it: the guard then
        # narrows the path that reads the file to a plain path string, instead
        # of re-reading a nullable field on every use. A multipart edit's mask and
        # markup are files (`_check_edit_form`).
        mask = e.mask
        mask isa String &&
            push!(parts, "mask" => HTTP.Multipart(basename(mask), IOBuffer(read(mask)), _mime_for(mask)))
        markup = e.markup
        markup isa Union{String,Vector{UInt8}} && push!(parts, "markup" => _markup_part(markup))
        for (k, f) in (("n", :n), ("size", :size), ("quality", :quality), ("input_fidelity", :input_fidelity),
                       ("background", :background), ("output_format", :output_format),
                       ("output_compression", :output_compression), ("moderation", :moderation),
                       ("safety_identifier", :safety_identifier), ("complexity", :complexity),
                       ("frame_count", :frame_count), ("frame_duration_ms", :frame_duration_ms),
                       ("partial_images", :partial_images))
            v = getfield(e, f)
            !isnothing(v) && push!(parts, k => string(v))
        end
        e.stream && push!(parts, "stream" => "true")
        svg = e.svg
        isnothing(svg) || append!(parts, _svg_form_fields(svg))
        HTTP.Form(parts)
    end
    (_api_base_url(service) * IMAGES_EDITS_PATH, auth_header_multipart(service), body)
end

# An edit's `svg` as multipart fields: "Use multipart fields such as `svg[model]`" (the edit reference).
_svg_form_fields(s::SvgOptions)::Vector{Pair{String,Any}} =
    Pair{String,Any}[k => v for (k, v) in (("svg[model]", s.model), ("svg[reasoning][effort]", s.effort)) if !isnothing(v)]

# The JSON edit (`EditImageBodyJsonParam`) of an edit by reference: "Use `images` (array of `ImageRefParam`)
# instead of multipart `image` uploads", its mask and markup references, and the edit's other fields as on a
# multipart edit, `svg` as its JSON object.
function _edit_json(e::ImageEdit, model::String)::Dict{Symbol,Any}
    image = e.image
    d = Dict{Symbol,Any}(:model => model, :prompt => e.prompt, :images => image isa ImageRef ? [image] : image)
    for f in (:mask, :markup, :n, :size, :quality, :input_fidelity, :background, :output_format, :output_compression,
              :moderation, :safety_identifier, :complexity, :frame_count, :frame_duration_ms, :svg, :partial_images)
        v = getfield(e, f)
        isnothing(v) || (d[f] = v)
    end
    e.stream && (d[:stream] = true)
    d
end

_image_reply(service, resp::HTTP.Response) = parse_image_response(resp)

# ── Gemini generateContent with image output ──

# Gemini's ImageConfig values (https://ai.google.dev/api/generate-content#ImageConfig), and
# the models whose docs list fewer resolutions: Nano Banana 2.1 renders at 1K, 2K and 4K
# (https://ai.google.dev/gemini-api/docs/models/gemini-nano-banana-2.1). An unlisted model
# is checked against Gemini's lists alone; its own limits are the server's answer.
const _GEMINI_ASPECT_RATIOS = ("1:1", "1:4", "4:1", "1:8", "8:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4",
                               "9:16", "16:9", "21:9")
const _GEMINI_RESOLUTIONS = ("512", "1K", "2K", "4K")
# ImageResponseFormat.delivery (https://ai.google.dev/api/generate-content): INLINE or URI.
const _GEMINI_DELIVERIES = ("inline", "uri")
const _GEMINI_MODEL_RESOLUTIONS = (("gemini-nano-banana-2.1", ("1K", "2K", "4K")),)

# A model's own resolution list, where its docs give one.
function _gemini_model_resolutions(model::String)
    row = findfirst(row -> first(row) == model, _GEMINI_MODEL_RESOLUTIONS)
    isnothing(row) ? nothing : last(_GEMINI_MODEL_RESOLUTIONS[row])
end

# The OpenAI Images API fields generateContent has no counterpart for, and how a refusal names
# the request. Gemini streams no partial images: on 2026-11-17 its streamGenerateContent answered an
# image request with one event that held the whole image.
_openai_only_image_fields(::ImageGeneration) =
    (:size, :quality, :background, :output_compression, :moderation, :complexity, :frame_count,
     :frame_duration_ms, :svg, :stream, :partial_images)
_openai_only_image_fields(::ImageEdit) =
    (:mask, :markup, :size, :quality, :input_fidelity, :background, :output_compression, :moderation,
     :complexity, :frame_count, :frame_duration_ms, :svg, :stream, :partial_images)
_image_verb(::ImageGeneration) = "image generation"
_image_verb(::ImageEdit) = "image editing"

_gemini_image_model(r::_ImageRequest)::String =
    isempty(r.model) ? default_image_model(GEMINIServiceEndpoint)::String : r.model

function _check_image_request(::Type{GEMINIServiceEndpoint}, r::_ImageRequest)
    openai = _set_image_fields(r, _openai_only_image_fields(r))
    isempty(openai) || throw(ArgumentError(
        "Gemini $(_image_verb(r)) does not take $(join(openai, ", ")), which only the OpenAI Images API takes" *
        ("size" in openai ? "; size a Gemini image with aspect_ratio and resolution" : "") *
        ("stream" in openai ? "; Gemini sends an image whole, in one event of its stream, so it has no partial " *
                              "images to stream: unset stream" : "")))
    r isa ImageEdit && _by_reference(r) && throw(ArgumentError(
        "Gemini image editing reads its input images from local files, sent inline: an ImageRef (an OpenAI file id " *
        "or image URL) has no form there; pass the image's path"))
    isnothing(r.n) || r.n == 1 || throw(ArgumentError(
        "Gemini returns one image per request; n must be 1 (got $(r.n))"))
    ratio = r.aspect_ratio
    isnothing(ratio) || ratio in _GEMINI_ASPECT_RATIOS || throw(ArgumentError(
        "Gemini aspect_ratio must be one of $(join(_GEMINI_ASPECT_RATIOS, ", ")) (got $(repr(ratio)))"))
    res = r.resolution
    if !isnothing(res)
        model = _gemini_image_model(r)
        own = _gemini_model_resolutions(model)
        sizes = something(own, _GEMINI_RESOLUTIONS)
        res in sizes || throw(ArgumentError("Gemini resolution$(isnothing(own) ? "" : " for $model") " *
            "must be one of $(join(sizes, ", ")) (got $(repr(res)))"))
    end
    id = r.safety_identifier
    isnothing(id) || _gemini_safety_label(id)
    fmt = r.output_format
    isnothing(fmt) || fmt == "jpeg" || throw(ArgumentError(
        "Gemini image output_format is \"jpeg\", the one format its ImageResponseFormat lists (IMAGE_JPEG) " *
        "(got $(repr(fmt))): Gemini answers JPEG; set \"jpeg\", or unset output_format"))
    d = r.delivery
    isnothing(d) || d in _GEMINI_DELIVERIES || throw(ArgumentError(
        "Gemini image delivery is \"inline\" or \"uri\", the image in the reply or a URI to fetch it from (got $(repr(d)))"))
    nothing
end

_gemini_image_url(r::_ImageRequest) =
    "$(GEMINI_NATIVE_BASE)/models/$(_uripart(_gemini_image_model(r))):generateContent"

# Image output only (`responseModalities`), sized by `imageConfig` when the request sets a
# control, its format and delivery in `responseFormat.image` (ImageResponseFormat, whose aspect
# ratio and size are the controls imageConfig already carries); the end user's id goes in the
# request label Google documents for it.
function _gemini_image_body(parts::Vector{Dict{Symbol,Any}}, r::_ImageRequest)::Dict{Symbol,Any}
    config = Dict{Symbol,Any}()
    isnothing(r.aspect_ratio) || (config[:aspectRatio] = r.aspect_ratio)
    isnothing(r.resolution) || (config[:imageSize] = r.resolution)
    gen = Dict{Symbol,Any}(:responseModalities => ["IMAGE"])
    isempty(config) || (gen[:imageConfig] = config)
    format = Dict{Symbol,Any}()
    isnothing(r.output_format) || (format[:mimeType] = "IMAGE_JPEG")
    d = r.delivery
    isnothing(d) || (format[:delivery] = uppercase(d))
    isempty(format) || (gen[:responseFormat] = Dict(:image => format))
    body = Dict{Symbol,Any}(:contents => [Dict{Symbol,Any}(:role => "user", :parts => parts)],
                            :generationConfig => gen)
    isnothing(r.safety_identifier) || (body[:labels] = Dict(:safety_identifier => r.safety_identifier))
    body
end

_image_payload(::Type{GEMINIServiceEndpoint}, ig::ImageGeneration) =
    _gemini_image_body([Dict{Symbol,Any}(:text => ig.prompt)], ig)

_image_request(::Type{GEMINIServiceEndpoint}, ig::ImageGeneration) =
    (_gemini_image_url(ig), auth_header(GEMINIServiceEndpoint), JSON.json(_image_payload(GEMINIServiceEndpoint, ig)))

# Each input image is an inline part (`_gemini_inline_data`, the part a chat attachment is), its
# format read from its bytes, ahead of the prompt.
function _image_request(::Type{GEMINIServiceEndpoint}, e::ImageEdit)
    parts = Dict{Symbol,Any}[_gemini_inline_data(ImageAttachment(read(file))) for file in _edit_images(e)]
    push!(parts, Dict{Symbol,Any}(:text => e.prompt))
    (_gemini_image_url(e), auth_header(GEMINIServiceEndpoint), JSON.json(_gemini_image_body(parts, e)))
end

# A generateContent reply → ImageResponse. Every image part of the first candidate, in
# order — inline data, or a URI (fileData) — is an ImageObject carrying the candidate's text, if any, as its revised prompt;
# thought parts (a thinking image model's interim images and reasoning) are skipped. A
# reply without an image fails loud with Gemini's reason, which the verb returns as an
# ImageCallError.
function _image_reply(::Type{GEMINIServiceEndpoint}, resp::HTTP.Response)::ImageResponse
    data = JSON.parse(resp.body; dicttype=Dict{String,Any})
    cands = get(data, "candidates", Any[])
    (cands isa AbstractVector && !isempty(cands)) || error("Gemini returned no image and no candidate ",
        "(promptFeedback: ", JSON.json(get(data, "promptFeedback", Dict{String,Any}())), ")")
    cand = cands[1]
    content = get(cand, "content", nothing)
    parts = content isa AbstractDict ? get(content, "parts", Any[]) : Any[]
    images = ImageObject[]          # each without its revised prompt yet: the text comes from every part
    text = IOBuffer()
    for p in parts
        (p isa AbstractDict && get(p, "thought", false) !== true) || continue
        inline, file = get(p, "inlineData", nothing), get(p, "fileData", nothing)
        if inline isa AbstractDict && startswith(string(get(inline, "mimeType", "")), "image/")
            b64 = get(inline, "data", nothing)
            b64 isa AbstractString && push!(images, ImageObject(b64_json=b64))
        elseif file isa AbstractDict && startswith(string(get(file, "mimeType", "image/")), "image/")
            # An image delivered by URI (`delivery="uri"`): a FileData part, whose mimeType is optional.
            uri = get(file, "fileUri", nothing)
            uri isa AbstractString && push!(images, ImageObject(url=uri))
        elseif (t = get(p, "text", nothing)) isa AbstractString
            print(text, t)
        end
    end
    said = takestring!(text)
    if isempty(images)
        why = String["finishReason $(something(get(cand, "finishReason", nothing), "none"))"]
        (m = get(cand, "finishMessage", nothing)) isa AbstractString && push!(why, "finishMessage $(repr(m))")
        isempty(said) || push!(why, "text $(repr(said))")
        error("Gemini returned no image (", join(why, "; "), ")")
    end
    revised = isempty(said) ? nothing : said
    ImageResponse(created=trunc(Int64, time()),
                  data=[ImageObject(b64_json=img.b64_json, url=img.url, revised_prompt=revised) for img in images],
                  usage=_gemini_image_usage(get(data, "usageMetadata", nothing)), raw=data)
end

# Gemini's usageMetadata under the Images API usage keys `token_usage` reads, counted as
# Gemini Chat counts it (`_gemini_usage`).
function _gemini_image_usage(u)::Union{Dict{String,Any},Nothing}
    t = _gemini_usage(u)
    (isnothing(t) || !(u isa AbstractDict)) && return nothing
    input = Dict{String,Any}("cached_tokens" => t.cached_tokens)
    output = Dict{String,Any}("reasoning_tokens" => t.reasoning_tokens)
    # The splits OpenAI's details give, where Gemini lists its tokens by modality: the input's text and image,
    # and the output's image; the rest of the output is its text and thinking.
    prompt = get(u, "promptTokensDetails", nothing)
    if prompt isa AbstractVector
        input["text_tokens"] = _modality_tokens(prompt, "TEXT", "tokenCount")
        input["image_tokens"] = _modality_tokens(prompt, "IMAGE", "tokenCount")
    end
    candidates = get(u, "candidatesTokensDetails", nothing)
    if candidates isa AbstractVector
        image = _modality_tokens(candidates, "IMAGE", "tokenCount")
        output["image_tokens"] = image
        output["text_tokens"] = t.completion_tokens - image
    end
    Dict{String,Any}("input_tokens" => t.prompt_tokens, "output_tokens" => t.completion_tokens,
                     "total_tokens" => t.total_tokens, "input_tokens_details" => input, "output_tokens_details" => output)
end
