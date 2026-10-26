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
  time to the service's default image model (`"gpt-image-2.5-sunburst"` for OpenAI,
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
- `quality::Union{String,Nothing}`: Quality level (`"low"`, `"medium"`, `"high"`, `"auto"`);
  `gpt-image-2.5-flare` and `gpt-image-2.5-sunburst` also accept `"xhigh"` and `"max"`
- `background::Union{String,Nothing}`: Background (`"transparent"`, `"opaque"`, `"auto"`)
- `output_format::Union{String,Nothing}`: File format (`"png"`, `"webp"`, `"jpeg"`)
- `output_compression::Union{Int,Nothing}`: Compression (0–100, for `"webp"` and `"jpeg"`)
- `moderation::Union{String,Nothing}`: Content-moderation strictness (`"auto"`, `"low"`)

# Fields — Gemini only
[Gemini's image controls](https://ai.google.dev/gemini-api/docs/image-generation#aspect-ratios-and-image-size):
- `aspect_ratio::Union{String,Nothing}`: `"1:1"`, `"1:4"`, `"4:1"`, `"1:8"`, `"8:1"`,
  `"2:3"`, `"3:2"`, `"3:4"`, `"4:3"`, `"4:5"`, `"5:4"`, `"9:16"`, `"16:9"` or `"21:9"`;
  unset, the image is square
- `resolution::Union{String,Nothing}`: `"512"`, `"1K"`, `"2K"` or `"4K"`; unset, `"1K"`.
  `gemini-nano-banana-2.1` renders at `"1K"`, `"2K"` or `"4K"`.

A field the request's provider does not take (an OpenAI-only field on Gemini;
`aspect_ratio` or `resolution` on OpenAI), `n` other than 1 on Gemini, a Gemini
`aspect_ratio` or `resolution` outside the lists above, and a `safety_identifier` outside
its provider's rule throw `ArgumentError` naming the provider and the field — when the
request is serialized, and in [`generate_image`](@ref) before any request. Other OpenAI
values are sent as given, for OpenAI to judge. `input_fidelity` is an image-edit
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
    for f in (:n, :size, :quality, :background, :output_format, :output_compression, :safety_identifier, :moderation)
        v = getfield(ig, f)
        !isnothing(v) && (d[f] = v)
    end
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
  as `reasoning_tokens`
- `raw::Dict{String,Any}`: Complete raw JSON response
"""
@kwdef struct ImageResponse
    created::Int64
    data::Vector{ImageObject}
    usage::Union{Dict{String,Any},Nothing} = nothing
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
        raw=data
    )
end


# ─── Request Functions ───────────────────────────────────────────────────────

"""
    generate_image(ig::ImageGeneration; config=nothing, cancel=nothing)

Send an image-generation request: to OpenAI's Images API, or, for
`service=GEMINIServiceEndpoint`, to Gemini's `models/{model}:generateContent`, asking
for image output only and sizing the image by `imageConfig` when `aspect_ratio` or
`resolution` is set ([Gemini image generation](https://ai.google.dev/gemini-api/docs/image-generation)).

Returns [`ImageSuccess`](@ref), [`ImageFailure`](@ref), or [`ImageCallError`](@ref).
A Gemini reply's image parts become its [`ImageObject`](@ref)s, in order, each carrying
the text the model sent, if any, as `revised_prompt`; the interim images of the model's
thinking are not among them. A Gemini 200 that holds no image is an `ImageCallError`
naming Gemini's finish reason (such as `IMAGE_SAFETY` or `NO_IMAGE`).

Throws `ArgumentError` before any network I/O when `ig.service` does not declare `:images`
(an endpoint type that declares no capabilities at all included), and when the request sets
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
function generate_image(ig::ImageGeneration; config::Union{Nothing,RequestConfig}=nothing,
                        cancel::Union{Nothing,CancelToken}=nothing)
    validate_capability(ig.service, :images, "Image Generation API")
    _check_image_request(ig.service, ig)
    _image_call(ig.service, ig, config, cancel)
end

"""
    generate_image(prompt::String; kwargs...)

Convenience method: create an [`ImageGeneration`](@ref) from a prompt + keyword arguments
and send it. `config` and `cancel` go to the request, as for `generate_image(ig)`.

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
function generate_image(prompt::String; config::Union{Nothing,RequestConfig}=nothing,
                        cancel::Union{Nothing,CancelToken}=nothing, kwargs...)
    generate_image(ImageGeneration(; prompt=prompt, kwargs...); config, cancel)
end


# ─── Image Edits ──────────────────────────────────────────────────────────────

"""
    ImageEdit(; image, prompt, service=OPENAIServiceEndpoint, model="", kwargs...)

An image-edit request for [`edit_image`](@ref): the image(s) at `image` (a path, or a
vector of paths) changed as `prompt` asks. OpenAI is sent multipart/form-data at
`/v1/images/edits`; `GEMINIServiceEndpoint` is sent `generateContent` with each image
as an inline part ahead of the prompt, its format read from the file's bytes (PNG,
JPEG, GIF or WebP). `model` left `""` is the service's default image model
(`"gpt-image-2.5-sunburst"` on OpenAI, `"gemini-nano-banana-2.1"` on Gemini), resolved
when the edit is sent.

# Fields — both providers
`image` and `model` (above); `service`, `prompt`, `n` and `safety_identifier`, as on
[`ImageGeneration`](@ref): Gemini takes `n=1` only.

# Fields — OpenAI only
- `mask::Union{String,Nothing}`: path of a mask whose transparent area marks what to edit
- `size`, `quality`, `background`, `output_format`: as on [`ImageGeneration`](@ref)
- `input_fidelity::Union{String,Nothing}`: how closely the edit keeps the input's
  features (`"high"`, `"low"`)

# Fields — Gemini only
`aspect_ratio` and `resolution`, as on [`ImageGeneration`](@ref); unset, the edited
image follows the input's aspect ratio.

`image` names at least one file: [`edit_image`](@ref) throws `ArgumentError` for an empty
vector before any request, as it does, naming the provider and the field, for a field the
request's provider does not take, `n` other than 1 on Gemini, a Gemini `aspect_ratio` or
`resolution` outside its list, and a `safety_identifier` outside its provider's rule. Other
OpenAI values are sent as given, for OpenAI to judge.
"""
@kwdef struct ImageEdit
    service::ServiceEndpointSpec = OPENAIServiceEndpoint
    image::Union{String,Vector{String}}
    prompt::String
    mask::Union{String,Nothing} = nothing
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
end

"""
    edit_image(e::ImageEdit; config=nothing, cancel=nothing) -> LLMRequestResponse
    edit_image(image, prompt; mask=nothing, service=OPENAIServiceEndpoint, config=nothing, cancel=nothing, kwargs...)

Edit/extend image(s) under a text prompt — on OpenAI's Images API, or on Gemini's
`generateContent` for `service=GEMINIServiceEndpoint`. Returns the same
`ImageSuccess`/`ImageFailure`/`ImageCallError` shapes as [`generate_image`](@ref), and a
Gemini reply decodes the same way.

Throws `ArgumentError` before any network I/O when `e.service` does not declare
`:image_edits` (an endpoint type that declares no capabilities at all included), when `e.image`
is an empty vector, when an input image or the mask
is no file (as [`upload_file`](@ref) and [`transcribe`](@ref) refuse one), and when the request sets a
field its provider does not take, `n` other than 1 on Gemini, a Gemini `aspect_ratio` or
`resolution` outside its list, or a `safety_identifier` outside its provider's rule (see
[`ImageEdit`](@ref)). Other OpenAI values are sent as given.

Pass `config::Union{Nothing,RequestConfig}` to override the timeout/retry budget for this
call, and `cancel::Union{Nothing,CancelToken}` to make it cancellable exactly as for
[`generate_image`](@ref): a cancelled call returns an `ImageCallError` whose `cause` is a
[`UniLMCancelled`](@ref).
"""
function edit_image(e::ImageEdit; config::Union{Nothing,RequestConfig}=nothing,
                    cancel::Union{Nothing,CancelToken}=nothing)
    validate_capability(e.service, :image_edits, "Image Edits API")
    # An edit needs an input image: sent without one, Gemini would read the prompt alone as a
    # text-to-image request.
    e.image isa Vector && isempty(e.image) && throw(ArgumentError(
        "an image edit needs at least one input image, and `image` is an empty vector: pass a path or a vector of paths"))
    _check_image_request(e.service, e)
    _check_edit_files(e)
    _image_call(e.service, e, config, cancel)
end
edit_image(image, prompt::String; mask::Union{String,Nothing}=nothing,
    service::ServiceEndpointSpec=OPENAIServiceEndpoint, config::Union{Nothing,RequestConfig}=nothing,
    cancel::Union{Nothing,CancelToken}=nothing, kwargs...) =
    edit_image(ImageEdit(; service=service, image=image, prompt=prompt, mask=mask, kwargs...); config, cancel)


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

# The names of the fields of `r`, among `fields`, that are set.
_set_image_fields(r::_ImageRequest, fields)::Vector{String} =
    String[String(f) for f in fields if !isnothing(getfield(r, f))]

# An edit's input paths, each a file: a missing one is the caller's error, which `edit_image`
# reports before any request (`_check_edit_files`), as `upload_file` and `transcribe` do.
function _edit_images(e::ImageEdit)::Vector{String}
    images = e.image isa String ? [e.image] : e.image
    absent = findfirst(!isfile, images)
    isnothing(absent) || throw(ArgumentError("image not found: $(images[absent])"))
    images
end

function _check_edit_files(e::ImageEdit)::Nothing
    _edit_images(e)
    isnothing(e.mask) || isfile(e.mask) || throw(ArgumentError("mask not found: $(e.mask)"))
    nothing
end

# ── OpenAI Images API (the default) ──

# The Images API has no aspect-ratio or resolution field: `size` names both. A safety
# identifier is 1 to 128 characters: OpenAI's guide says to keep it to 128 or fewer, and
# the API reads an empty one as none
# (https://developers.openai.com/api/docs/guides/safety-best-practices#implement-safety-identifiers).
function _check_image_request(service, r::_ImageRequest)
    gemini = _set_image_fields(r, (:aspect_ratio, :resolution))
    isempty(gemini) || throw(ArgumentError(
        "the OpenAI Images API has no $(join(gemini, " or ")), which only Gemini takes; " *
        "set size=\"WIDTHxHEIGHT\" instead"))
    id = r.safety_identifier
    isnothing(id) || 1 <= length(id) <= 128 || throw(ArgumentError(
        "the OpenAI Images API takes a safety_identifier of 1 to 128 characters (got $(length(id))); it reads " *
        "an empty one as none, and a longer identifier can be hashed (a SHA-256 hex digest is 64 characters)"))
    nothing
end

_image_request(service, ig::ImageGeneration) =
    (_api_base_url(service) * IMAGES_GENERATIONS_PATH, auth_header(service), JSON.json(_image_payload(service, ig)))

function _image_request(service, e::ImageEdit)
    model = isempty(e.model) ? something(default_image_model(service), "") : e.model
    isempty(model) && throw(ArgumentError("model must be specified for image edits with $(typeof(service))"))
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
        # of re-reading a nullable field on every use.
        mask = e.mask
        isnothing(mask) ||
            push!(parts, "mask" => HTTP.Multipart(basename(mask), IOBuffer(read(mask)), _mime_for(mask)))
        for (k, f) in (("n", :n), ("size", :size), ("quality", :quality), ("input_fidelity", :input_fidelity),
                       ("background", :background), ("output_format", :output_format),
                       ("safety_identifier", :safety_identifier))
            v = getfield(e, f)
            !isnothing(v) && push!(parts, k => string(v))
        end
        HTTP.Form(parts)
    end
    (_api_base_url(service) * IMAGES_EDITS_PATH, auth_header_multipart(service), body)
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
const _GEMINI_MODEL_RESOLUTIONS = (("gemini-nano-banana-2.1", ("1K", "2K", "4K")),)

# A model's own resolution list, where its docs give one.
function _gemini_model_resolutions(model::String)
    row = findfirst(row -> first(row) == model, _GEMINI_MODEL_RESOLUTIONS)
    isnothing(row) ? nothing : last(_GEMINI_MODEL_RESOLUTIONS[row])
end

# The OpenAI Images API fields generateContent has no counterpart for, and how a refusal names
# the request.
_openai_only_image_fields(::ImageGeneration) =
    (:size, :quality, :background, :output_format, :output_compression, :moderation)
_openai_only_image_fields(::ImageEdit) = (:mask, :size, :quality, :input_fidelity, :background, :output_format)
_image_verb(::ImageGeneration) = "image generation"
_image_verb(::ImageEdit) = "image editing"

_gemini_image_model(r::_ImageRequest)::String =
    isempty(r.model) ? default_image_model(GEMINIServiceEndpoint)::String : r.model

function _check_image_request(::Type{GEMINIServiceEndpoint}, r::_ImageRequest)
    openai = _set_image_fields(r, _openai_only_image_fields(r))
    isempty(openai) || throw(ArgumentError(
        "Gemini $(_image_verb(r)) does not take $(join(openai, ", ")), which only the OpenAI Images API takes" *
        ("size" in openai ? "; size a Gemini image with aspect_ratio and resolution" : "")))
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
    nothing
end

_gemini_image_url(r::_ImageRequest) =
    "$(GEMINI_NATIVE_BASE)/models/$(_uripart(_gemini_image_model(r))):generateContent"

# Image output only (`responseModalities`), sized by `imageConfig` when the request sets a
# control; the end user's id goes in the request label Google documents for it.
function _gemini_image_body(parts::Vector{Dict{Symbol,Any}}, r::_ImageRequest)::Dict{Symbol,Any}
    config = Dict{Symbol,Any}()
    isnothing(r.aspect_ratio) || (config[:aspectRatio] = r.aspect_ratio)
    isnothing(r.resolution) || (config[:imageSize] = r.resolution)
    gen = Dict{Symbol,Any}(:responseModalities => ["IMAGE"])
    isempty(config) || (gen[:imageConfig] = config)
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
# order, is an ImageObject carrying the candidate's text, if any, as its revised prompt;
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
    images = String[]
    text = IOBuffer()
    for p in parts
        (p isa AbstractDict && get(p, "thought", false) !== true) || continue
        inline = get(p, "inlineData", nothing)
        if inline isa AbstractDict && startswith(string(get(inline, "mimeType", "")), "image/")
            b64 = get(inline, "data", nothing)
            b64 isa AbstractString && push!(images, b64)
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
                  data=[ImageObject(b64_json=b64, revised_prompt=revised) for b64 in images],
                  usage=_gemini_image_usage(get(data, "usageMetadata", nothing)), raw=data)
end

# Gemini's usageMetadata under the Images API usage keys `token_usage` reads, counted as
# Gemini Chat counts it (`_gemini_usage`).
function _gemini_image_usage(u)::Union{Dict{String,Any},Nothing}
    t = _gemini_usage(u)
    isnothing(t) && return nothing
    Dict{String,Any}("input_tokens" => t.prompt_tokens, "output_tokens" => t.completion_tokens,
                     "total_tokens" => t.total_tokens,
                     "input_tokens_details" => Dict{String,Any}("cached_tokens" => t.cached_tokens),
                     "output_tokens_details" => Dict{String,Any}("reasoning_tokens" => t.reasoning_tokens))
end
