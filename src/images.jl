# ============================================================================
# OpenAI Images API
# https://platform.openai.com/docs/api-reference/images
#
# The Images API generates images from text prompts using models like
# gpt-image-1.5. Supports configurable size, quality, background
# transparency, and output format.
# ============================================================================

using Base64

# ─── Request Type ─────────────────────────────────────────────────────────────

"""
    ImageGeneration(; prompt, model="gpt-image-1.5", kwargs...)

Configuration struct for an OpenAI Image Generation API request.

# Key Fields
- `model::String`: Model to use (default: `"gpt-image-1.5"`)
- `prompt::String`: A text description of the desired image
- `n::Union{Int,Nothing}`: Number of images to generate (1–10)
- `size::Union{String,Nothing}`: Size (`"1024x1024"`, `"1536x1024"`, `"1024x1536"`, `"auto"`)
- `quality::Union{String,Nothing}`: Quality level (`"low"`, `"medium"`, `"high"`, `"auto"`)
- `background::Union{String,Nothing}`: Background (`"transparent"`, `"opaque"`, `"auto"`)
- `output_format::Union{String,Nothing}`: File format (`"png"`, `"webp"`, `"jpeg"`)
- `output_compression::Union{Int,Nothing}`: Compression (0–100, for `"webp"` and `"jpeg"`)
- `user::Union{String,Nothing}`: End-user identifier

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
```
"""
@kwdef struct ImageGeneration
    service::Type{<:ServiceEndpoint} = OPENAIServiceEndpoint
    model::String = "gpt-image-1.5"
    prompt::String
    n::Union{Int,Nothing} = nothing
    size::Union{String,Nothing} = nothing
    quality::Union{String,Nothing} = nothing
    background::Union{String,Nothing} = nothing
    output_format::Union{String,Nothing} = nothing
    output_compression::Union{Int,Nothing} = nothing
    user::Union{String,Nothing} = nothing
end

function JSON.lower(ig::ImageGeneration)
    d = Dict{Symbol,Any}(:model => ig.model, :prompt => ig.prompt)
    for f in (:n, :size, :quality, :background, :output_format, :output_compression, :user)
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
- `b64_json::Union{String,Nothing}`: Base64-encoded image data
- `revised_prompt::Union{String,Nothing}`: The prompt as revised by the model
"""
@kwdef struct ImageObject
    b64_json::Union{String,Nothing} = nothing
    revised_prompt::Union{String,Nothing} = nothing
end

"""
    ImageResponse

Parsed response from the Image Generation API.

# Accessors
- `image_data(r)` — extract base64-encoded image data
- `r.created`, `r.data`, `r.usage` — basic fields
- `r.raw` — the complete raw JSON dict

# Fields
- `created::Int64`: Timestamp when the response was created
- `data::Vector{ImageObject}`: Generated images
- `usage::Union{Dict{String,Any},Nothing}`: Token usage information
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

HTTP-level failure from the Image Generation API. Contains the response body and status code.
"""
@kwdef struct ImageFailure <: LLMRequestResponse
    response::String
    status::Int
end

"""
    ImageCallError <: LLMRequestResponse

Exception-level error during an Image Generation API call (network, parsing, etc.).
"""
@kwdef struct ImageCallError <: LLMRequestResponse
    error::String
    status::Union{Int,Nothing} = nothing
end


# ─── Accessor Functions ──────────────────────────────────────────────────────

"""
    image_data(r::ImageResponse)::Vector{String}
    image_data(r::ImageSuccess)::Vector{String}

Extract base64-encoded image data from a response. Returns a vector of base64 strings,
one per generated image.

# Examples
```julia
result = generate_image("A sunset over mountains")
imgs = image_data(result)       # Vector{String}
length(imgs)                     # number of images generated
```
"""
function image_data(r::ImageResponse)::Vector{String}
    return [img.b64_json for img in r.data if !isnothing(img.b64_json)]
end

image_data(r::ImageSuccess) = image_data(r.response)
image_data(::ImageFailure) = String[]
image_data(::ImageCallError) = String[]

"""
    save_image(img_b64::String, filepath::String)

Decode a base64-encoded image and save it to a file.

# Examples
```julia
result = generate_image("A watercolor landscape")
if result isa ImageSuccess
    save_image(image_data(result)[1], "landscape.png")
end
```
"""
function save_image(img_b64::String, filepath::String)
    open(filepath, "w") do io
        write(io, base64decode(img_b64))
    end
    return filepath
end


# ─── Parsing ─────────────────────────────────────────────────────────────────

function parse_image_response(resp::HTTP.Response)::ImageResponse
    data = JSON.parse(resp.body; dicttype=Dict{String,Any})
    images = [
        ImageObject(
            b64_json=get(img, "b64_json", nothing),
            revised_prompt=get(img, "revised_prompt", nothing)
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
    generate_image(ig::ImageGeneration; retries=0)

Send a request to the OpenAI Image Generation API.

Returns [`ImageSuccess`](@ref), [`ImageFailure`](@ref), or [`ImageCallError`](@ref).

# Examples
```julia
ig = ImageGeneration(prompt="A cute robot learning Julia", quality="high")
result = generate_image(ig)
if result isa ImageSuccess
    println("Generated \$(length(result.response.data)) image(s)")
    save_image(image_data(result)[1], "robot.png")
end
```
"""
function generate_image(ig::ImageGeneration; retries::Int=0)
    res = ImageCallError(error="uninitialized", status=0)
    try
        body = JSON.json(ig)
        url = OPENAI_BASE_URL * IMAGES_GENERATIONS_PATH
        resp = HTTP.post(url, body=body, headers=auth_header(ig.service))

        if resp.status == 200
            return ImageSuccess(response=parse_image_response(resp))
        elseif resp.status in (500, 503)
            @warn "Request status: $(resp.status). Retrying in 1s..."
            sleep(1)
            if retries < 30
                return generate_image(ig; retries=retries + 1)
            else
                return ImageFailure(response=String(resp.body), status=resp.status)
            end
        else
            return ImageFailure(response=String(resp.body), status=resp.status)
        end
    catch e
        statuserror = hasproperty(e, :status) ? e.status : nothing
        res = ImageCallError(error=string(e), status=statuserror)
    end
    return res
end

"""
    generate_image(prompt::String; kwargs...)

Convenience method: create an [`ImageGeneration`](@ref) from a prompt + keyword arguments
and send it.

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

# Save to file
if result isa ImageSuccess
    save_image(image_data(result)[1], "logo.png")
end
```
"""
function generate_image(prompt::String; kwargs...)
    generate_image(ImageGeneration(; prompt=prompt, kwargs...))
end
