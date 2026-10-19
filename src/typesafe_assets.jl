# ============================================================================
# TypeSafe images — how the images of a System One request travel on TypeSafe's wire.
#
# TypeSafe reads an image wherever a string can stand (https://docs.typesafe.ai/images):
# the request lists each image once in its top-level `assets` object, keyed by a
# reference, and wherever that reference is written as a string the model reads the
# image instead. Each distinct ImageAttachment of a request gets a reference derived
# from its bytes and one multipart part, every place it stood is written as the
# reference, and every image is checked against TypeSafe's limits before anything is
# sent.
# ============================================================================

# TypeSafe's image limits (https://docs.typesafe.ai/images#limits): PNG and JPEG only, a
# file over 16 MiB is answered 413, a side outside 280–1568 pixels "Invalid image
# dimensions.", and a GIF, an animated PNG or a damaged file "Invalid image request."
# (both 400).
const _TYPESAFE_MAX_IMAGE_BYTES = 16 * 1024 * 1024
const _TYPESAFE_IMAGE_SIDES = 280:1568

# The reference of an image: the first 128 bits of the SHA-256 of its bytes, so the same
# image is one asset however often it occurs, and a request's text — and so its recording
# key — depends on the images, never on how they were built.
_typesafe_ref(digest::AbstractString)::String = string("<image:", first(digest, 32), ">")
const _TYPESAFE_REF = r"^<image:[0-9a-f]{32}>$"

# ─── Image headers ───────────────────────────────────────────────────────────
# The pixel size is read from the header, as the service reads it, without decoding
# an image: a size, or the reason there is none.

_be16(d::Vector{UInt8}, i::Int)::Int = Int(d[i]) << 8 | Int(d[i+1])
_be32(d::Vector{UInt8}, i::Int)::Int = Int(d[i]) << 24 | Int(d[i+1]) << 16 | Int(d[i+2]) << 8 | Int(d[i+3])

# A PNG's size is in its first chunk, IHDR. Its chunks are walked too: an `acTL` chunk
# before the first IDAT makes it an animated PNG, and a chunk that runs past the end of
# the file a damaged one.
function _png_size(d::Vector{UInt8})::Union{Tuple{Int,Int},String}
    n = length(d)
    n >= 33 && _be32(d, 9) == 13 && view(d, 13:16) == b"IHDR" || return "its PNG header cannot be read"
    pos, image_data = 9, false
    while pos <= n
        pos + 7 <= n || return "its PNG data is cut short"
        kind = view(d, pos+4:pos+7)
        next = pos + 12 + _be32(d, pos)
        next - 1 <= n || return "its PNG data is cut short"
        if kind == b"IDAT"
            image_data = true
        elseif kind == b"acTL" && !image_data
            return "it is an animated PNG; TypeSafe accepts still images only"
        elseif kind == b"IEND"
            break
        end
        pos = next
    end
    image_data ? (_be32(d, 17), _be32(d, 21)) : "its PNG holds no image data"
end

# A JPEG's size is in its frame header, the SOFn segment: every marker 0xC0–0xCF but DHT
# (C4), JPG (C8) and DAC (CC). The segments before it are skipped by their lengths.
_is_sof(m::UInt8)::Bool = 0xc0 <= m <= 0xcf && m != 0xc4 && m != 0xc8 && m != 0xcc

function _jpeg_size(d::Vector{UInt8})::Union{Tuple{Int,Int},String}
    n, pos = length(d), 3                                  # past the start-of-image marker
    while pos <= n
        d[pos] == 0xff || return "its JPEG markers cannot be read"
        while pos <= n && d[pos] == 0xff                   # fill bytes before a marker
            pos += 1
        end
        pos <= n || break
        m = d[pos]
        pos += 1
        (m == 0x01 || 0xd0 <= m <= 0xd7) && continue       # markers without a segment
        (m == 0xd9 || m == 0xda) && return "its JPEG frame header is missing"
        (m == 0x00 || m == 0xd8) && return "its JPEG markers cannot be read"
        pos + 1 <= n || break
        len = _be16(d, pos)
        len >= 2 && pos + len - 1 <= n || return "its JPEG data is cut short"
        _is_sof(m) && return len >= 7 ? (_be16(d, pos + 5), _be16(d, pos + 3)) : "its JPEG frame header cannot be read"
        pos += len
    end
    "its JPEG data is cut short"
end

function _typesafe_check_image(img::ImageAttachment, k::Int)::Nothing
    what = "image $(k) of the request ($(img))"
    img.mime in ("image/png", "image/jpeg") || throw(ArgumentError(
        "TypeSafe accepts PNG and JPEG images only; $(what) is neither. Convert it to PNG or JPEG"))
    length(img.data) <= _TYPESAFE_MAX_IMAGE_BYTES || throw(ArgumentError(
        "TypeSafe accepts image files of at most 16 MiB ($(_TYPESAFE_MAX_IMAGE_BYTES) bytes); $(what) has " *
        "$(length(img.data)) bytes. Resize it so neither side is over 1568 pixels, or save it compressed"))
    size = img.mime == "image/png" ? _png_size(img.data) : _jpeg_size(img.data)
    size isa String && throw(ArgumentError(
        "TypeSafe cannot take $(what): $(size). Export it again as a still PNG or JPEG"))
    w, h = size
    w in _TYPESAFE_IMAGE_SIDES && h in _TYPESAFE_IMAGE_SIDES || throw(ArgumentError(
        "TypeSafe accepts images of $(first(_TYPESAFE_IMAGE_SIDES)) to $(last(_TYPESAFE_IMAGE_SIDES)) pixels on " *
        "each side; $(what) is $(w)×$(h). Resize it, scaling up or padding a side that is too short"))
    nothing
end

# ─── Assets ──────────────────────────────────────────────────────────────────

# The images of one request on TypeSafe's wire: the reference of every image by the
# identity of its bytes, the `assets` object, the parts in first-occurrence order, and
# the line per part its recording key adds — the part's name and the SHA-256 of its
# bytes, never the bytes.
struct _TypeSafeAssets
    refs::IdDict{Vector{UInt8},String}
    json::JSON.Object{String,Any}
    parts::Vector{Pair{String,ImageAttachment}}
    digests::String
end

function _typesafe_assets(images::Vector{ImageAttachment})::_TypeSafeAssets
    refs = IdDict{Vector{UInt8},String}()
    json = JSON.Object{String,Any}()
    parts = Pair{String,ImageAttachment}[]
    lines = String[]
    for img in images
        haskey(refs, img.data) && continue
        digest = bytes2hex(sha256(img.data))
        ref = _typesafe_ref(digest)
        refs[img.data] = ref
        haskey(json, ref) && continue                      # the same bytes in another attachment
        name = string("asset-", length(parts))
        _typesafe_check_image(img, length(parts) + 1)
        json[ref] = JSON.Object{String,Any}("type" => "image", "part" => name)
        push!(parts, name => img)
        push!(lines, string(name, ' ', digest))
    end
    _TypeSafeAssets(refs, json, parts, join(lines, '\n'))
end

# Written in this style, every image of a request is its reference; everything else is
# written exactly as JSON.jl writes it by default.
struct _TypeSafeAssetStyle <: JSON.JSONStyle
    refs::IdDict{Vector{UInt8},String}
end

# An image the request's containers do not reach — held by a struct, say — was never
# made an asset, and written as anything else it would reach the model as text.
JSON.lower(s::_TypeSafeAssetStyle, img::ImageAttachment) = get(s.refs, img.data) do
    throw(ArgumentError("TypeSafe cannot send $(img), which the request holds inside a value that is not a " *
                        "dictionary, vector, tuple or NamedTuple: put the image in one of those"))
end

# The JSON part of a request with images: its System One JSON form with `assets` last.
# Like `_system_one_body`, a value JSON cannot carry is the caller's error, before any I/O.
function _typesafe_body(json::JSON.Object{String,Any}, assets::_TypeSafeAssets)::String
    json["assets"] = assets.json
    try
        JSON.json(json; style = _TypeSafeAssetStyle(assets.refs))
    catch e
        e isa InterruptException && rethrow()
        throw(ArgumentError("the System One request cannot be written as JSON, so nothing was sent: " *
                            first(split(sprint(showerror, e), '\n'))))
    end
end

# ─── Answers ─────────────────────────────────────────────────────────────────
# An answer that repeats the request's content — a Score's `legend` — repeats an image
# as its reference: each one there becomes the request's ImageAttachment again. The
# references are computed only when a legend holds one, so an answer without images
# costs nothing here.

_holds_ref(v::AbstractString)::Bool = sizeof(v) == 40 && startswith(v, "<image:") && occursin(_TYPESAFE_REF, v)
_holds_ref(v::AbstractVector)::Bool = any(_holds_ref, v)
_holds_ref(v::AbstractDict)::Bool = any(_holds_ref, values(v))
_holds_ref(_)::Bool = false

_unref(v::AbstractString, images::Dict{String,ImageAttachment}) = get(images, v, v)
_unref(v::AbstractVector, images::Dict{String,ImageAttachment}) = Any[_unref(x, images) for x in v]
_unref(v::AbstractDict, images::Dict{String,ImageAttachment}) =
    Dict{String,Any}(string(k) => _unref(x, images) for (k, x) in v)
_unref(v, ::Dict{String,ImageAttachment}) = v

function _images_by_ref(request::SystemOneRequest)::Dict{String,ImageAttachment}
    seen = IdDict{Vector{UInt8},Nothing}()
    out = Dict{String,ImageAttachment}()
    for img in _system_one_images(request)
        haskey(seen, img.data) && continue
        seen[img.data] = nothing
        get!(out, _typesafe_ref(bytes2hex(sha256(img.data))), img)
    end
    out
end

function _restore_images!(response::SystemOneResponse, request::SystemOneRequest)::SystemOneResponse
    images = nothing
    for a in values(response.answers)
        a isa ScoreAnswer || continue
        for level in collect(keys(a.legend))
            _holds_ref(a.legend[level]) || continue
            isnothing(images) && (images = _images_by_ref(request))
            a.legend[level] = _unref(a.legend[level], images)
        end
    end
    response
end
