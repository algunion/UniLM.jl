"""
    FunctionSignature(; name, description=nothing, parameters=nothing, strict=nothing)

Describes a function that can be called by the model in the Chat Completions API.

# Fields
- `name::String`: The name of the function.
- `description::Union{String,Nothing}`: A description of what the function does.
- `parameters::Union{AbstractDict,Nothing}`: JSON Schema object describing the function parameters.
- `strict::Union{Bool,Nothing}`: Enable strict schema adherence for function arguments
  (structured outputs). `nothing` (default) omits the field from the request — the API
  default, non-strict. `true` requires `parameters` to be a strict-valid schema
  (`additionalProperties: false` on every object, all properties `required`); UniLM does
  not validate this — the API rejects strict-invalid schemas with a 400.

# Example
```julia
sig = FunctionSignature(
    name="get_weather",
    description="Get the current weather in a given location",
    parameters=Dict(
        "type" => "object",
        "properties" => Dict(
            "location" => Dict("type" => "string", "description" => "The city")
        ),
        "required" => ["location"],
        "additionalProperties" => false
    ),
    strict=true
)
```
"""
@kwdef mutable struct FunctionSignature
    name::String
    description::Union{String,Nothing} = nothing
    parameters::Union{AbstractDict,Nothing} = nothing
    strict::Union{Bool,Nothing} = nothing
end

# pre-0.10.3 positional arity (@kwdef defaults apply only to the keyword constructor)
FunctionSignature(name, description, parameters) =
    FunctionSignature(name, description, parameters, nothing)

JSON.omit_null(::Type{FunctionSignature}) = true

struct GPTFunction
    name::String
    arguments::AbstractDict
end

function JSON.lower(x::GPTFunction)
    Dict(:name => x.name, :arguments => JSON.json(x.arguments))
end

"""
    ToolCall(; id, type="function", func)

Represents a tool call returned by the model. Contains the call `id` (used to match
results back), the tool `type`, and the [`GPTFunction`] with name and parsed arguments.

Also carries an optional `thought_signature::Union{Nothing,String}` field holding
Gemini-3's opaque tool-call signature, which must be echoed verbatim on the next turn;
it is set by the Gemini decoder, ignored (`nothing`) by every other provider, and
deliberately excluded from `JSON.lower` so OpenAI-wire serialization is unaffected.
"""
@kwdef struct ToolCall
    id::String
    type::String = "function"
    func::GPTFunction
    # Gemini-3 opaque function-calling signature; MUST be echoed verbatim on the
    # next turn or stateless multi-turn tool calls 400. Set only by the Gemini
    # decoder; ignored (nothing) by every other provider. Deliberately absent
    # from JSON.lower below, so OpenAI-wire serialization is byte-identical.
    thought_signature::Union{Nothing,String} = nothing
end

JSON.lower(x::ToolCall) = Dict(:id => x.id, :type => x.type, :function => x.func)

"""
    Tool(; type="function", func)

Wraps a [`FunctionSignature`](@ref) for use in the `tools` parameter of a [`Chat`](@ref).

# Example
```julia
tool = Tool(func=FunctionSignature(
    name="get_weather",
    description="Get the current weather",
    parameters=Dict("type" => "object", "properties" => Dict())
))
chat = Chat(tools=[tool], reasoning_effort="none")   # GPT-5.6 Chat tools need effort "none"
```
"""
@kwdef struct Tool
    type::String = "function"
    func::FunctionSignature
end

"""
    Tool(d::AbstractDict)

Construct a [`Tool`](@ref) from a dict. Accepts both the bare format
`{"name": ...}` and the wrapped OpenAI format `{"type": "function", "function": {"name": ...}}`.
"""
function Tool(d::AbstractDict)
    inner = haskey(d, "function") && d["function"] isa AbstractDict ? d["function"] : d
    strict = get(inner, "strict", nothing)
    strict isa Union{Bool,Nothing} ||
        throw(ArgumentError("tool \"strict\" must be a Bool or absent/null, got $(repr(strict))"))
    Tool(
        type=get(d, "type", "function"),
        func=FunctionSignature(
            name=inner["name"],
            description=get(inner, "description", nothing),
            parameters=get(inner, "parameters", nothing),
            strict=strict
        )
    )
end

JSON.lower(x::Tool) = Dict(:type => x.type, :function => x.func)


@kwdef struct GPTToolChoice
    type::String = "function"
    func::Union{String,Symbol}
end

JSON.lower(x::GPTToolChoice) = Dict(:type => x.type, :function => Dict(:name => x.func))


"""
    FunctionCallResult{T}

Holds the result of executing a function that was requested by the model via a tool call.

# Fields
- `name::Union{String,Symbol}`: The function name.
- `origincall::GPTFunction`: The original [`GPTFunction`] call from the model.
- `result::T`: The result of executing the function.
"""
struct FunctionCallResult{T}
    name::Union{String,Symbol}
    origincall::GPTFunction
    result::T
end

JSON.omit_null(::Type{<:FunctionCallResult}) = true
JSON.omit_empty(::Type{<:FunctionCallResult}) = true

# ─── Provider-neutral aliases ─────────────────────────────────────────────────
# The tool-surface structs above carry provider-neutral canonical names. These
# exported `GPT*` consts preserve the former names so existing user code
# (construction, dispatch, `isa`, field access, keyword constructors) keeps
# working unchanged — plain type aliases, no deprecation warning, retained until
# the 1.0 stability boundary. `GPTFunctionCallResult` stays a parametric
# `UnionAll`, so `GPTFunctionCallResult{T}(...)` still constructs.

"""
    GPTTool

Legacy alias for [`Tool`](@ref).
"""
const GPTTool = Tool

"""
    GPTToolCall

Legacy alias for [`ToolCall`](@ref).
"""
const GPTToolCall = ToolCall

"""
    GPTFunctionSignature

Legacy alias for [`FunctionSignature`](@ref).
"""
const GPTFunctionSignature = FunctionSignature

"""
    GPTFunctionCallResult

Legacy alias for [`FunctionCallResult`](@ref).
"""
const GPTFunctionCallResult = FunctionCallResult

"""
    RoleSystem

Role constant `"system"` — used for system-level instructions.
"""
const RoleSystem = "system"

"""
    RoleUser

Role constant `"user"` — used for user messages.
"""
const RoleUser = "user"

"""
    RoleAssistant

Role constant `"assistant"` — used for model-generated messages.
"""
const RoleAssistant = "assistant"

"""Role constant `"tool"` — used for tool/function call result messages."""
const RoleTool = "tool"

# The message roles of the chat wire.
const _MESSAGE_ROLES = (RoleSystem, RoleUser, RoleAssistant, RoleTool)

# Reasoning effort values the OpenAI API accepts; each model supports a subset.
const _OPENAI_REASONING_EFFORTS = ("none", "minimal", "low", "medium", "high", "xhigh", "max")

"""
    Model(name::String)

Named model identifier used internally for known model constants (e.g. `GPT5_2`).
"""
struct Model
    name::String
end

Base.show(io::IO, x::Model) = print(io, x.name)
Base.parse(::Type{Model}, s::String) = Model(s)

const GPT5_2 = Model("gpt-5.2")

const STOP = "stop"
const CONTENT_FILTER = "content_filter"
const TOOL_CALLS = "tool_calls"


"""
    ProviderContent(provider::Symbol, blocks::Vector{Any})

Provider-native assistant content captured verbatim at decode time and echoed
verbatim at encode time when the SAME provider encodes the turn again.

The neutral [`Message`](@ref) carries `content::String` + `tool_calls`, but some
providers attach blocks that must round-trip byte-faithfully for multi-turn
flows to work: Anthropic `thinking`/`redacted_thinking` blocks (their
`signature` must be echoed unmodified or tool round-trips on thinking models
are rejected with HTTP 400), Gemini text-part `thoughtSignature`s, DeepSeek
`reasoning_content` and Ollama `thinking`. `provider` tags the wire dialect
(`:anthropic`, `:gemini`, `:deepseek` or `:ollama`); `blocks` is the provider's
content/parts array exactly as decoded (String-keyed JSON) — for `:deepseek`,
`[Dict("reasoning_content" => text)]`, echoed only on requests that carry tools; for
`:ollama`, `[Dict("thinking" => text)]`, echoed as the assistant message's `thinking`.

Encoders ignore a `ProviderContent` tagged for a different provider — a
conversation moved across providers falls back to the neutral reconstruction
(the standard "other models drop thinking" semantics). The field never
serializes on the OpenAI wire (excluded from `JSON.lower(::Message)`).
"""
struct ProviderContent
    provider::Symbol
    blocks::Vector{Any}
end

"""
    Attachment

Abstract supertype of the media a user [`Message`](@ref) carries besides its text:
[`ImageAttachment`](@ref), [`AudioAttachment`](@ref) and [`DocumentAttachment`](@ref).
An attachment holds the media's bytes and the format read from their leading bytes (the
file signature), never from a file name, so its declared format always matches its content.

Endpoints on the OpenAI Chat Completions wire (OpenAI, Azure, the Gemini OpenAI-compatible
endpoint, DeepSeek, generic endpoints) send an image as an `image_url` part, with its
`detail` level when it sets one, a sound clip as an `input_audio` part and a document as a
`file` part, all base64: UniLM sends every kind as a content part on this wire, and the
server decides what to do with it. For the Responses API, [`input_file`](@ref) turns a
document into an `input_file` part. [`OllamaEndpoint`](@ref) sends images and clips in the
message's `images` list and refuses a document — a chat attachment, or an `input_file` part
for its Responses API, which documents no file inputs — and an image's detail level. The
native Anthropic encoder sends an image and a document as base64 `image` and `document` blocks
before the text, and refuses a sound clip and an image's detail level. Native Gemini sends
every kind as an `inlineData` part ahead of the text, an image's detail level as its per-part
media resolution (`"low"`, `"high"`; `"auto"` sends none; `"original"` is refused, as is any
level on the Gemini 2.5 and Gemma 4 models), and Gemini Interactions sends each as an `image`,
`audio` or `document` part of an [`InputMessage`](@ref)'s content. A System One request
carries images without a detail level, and refuses a document and a sound clip; an embedding
input carries an image or a clip without a level. Each refusal is an `ArgumentError` before
any network I/O.
"""
abstract type Attachment end

# File signatures of the accepted formats: the only source of an attachment's format.
function _sniff_image_mime(data::AbstractVector{UInt8})::Union{String,Nothing}
    n = length(data)
    n >= 8 && view(data, 1:8) == UInt8[0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a] && return "image/png"
    n >= 3 && view(data, 1:3) == UInt8[0xff, 0xd8, 0xff] && return "image/jpeg"
    n >= 6 && view(data, 1:6) in (b"GIF87a", b"GIF89a") && return "image/gif"
    n >= 12 && view(data, 1:4) == b"RIFF" && view(data, 9:12) == b"WEBP" && return "image/webp"
    nothing
end

# WAV is a RIFF/WAVE container; MP3 starts with an ID3v2 tag or an MPEG-1/2/2.5
# Layer III frame header (11 sync bits, then layer bits 01).
function _sniff_audio_format(data::AbstractVector{UInt8})::Union{String,Nothing}
    n = length(data)
    n >= 12 && view(data, 1:4) == b"RIFF" && view(data, 9:12) == b"WAVE" && return "wav"
    n >= 3 && view(data, 1:3) == b"ID3" && return "mp3"
    n >= 2 && data[1] == 0xff && data[2] & 0xe0 == 0xe0 && (data[2] >> 1) & 0x03 == 0x01 && return "mp3"
    nothing
end

# A PDF file begins with its header: `%PDF-`, then the version (ISO 32000-2, 7.5.2).
_sniff_document_mime(data::AbstractVector{UInt8})::Union{String,Nothing} =
    length(data) >= 5 && view(data, 1:5) == b"%PDF-" ? "application/pdf" : nothing

# The detail levels of an OpenAI image input, the Chat Completions `image_url.detail`
# (https://developers.openai.com/api/reference/resources/chat).
const _IMAGE_DETAILS = ("auto", "low", "high", "original")
_image_detail(::Nothing)::Nothing = nothing
_image_detail(level::AbstractString)::String = level in _IMAGE_DETAILS ? String(level) : _no_image_detail(level)
_image_detail(level) = _no_image_detail(level)
_no_image_detail(level) = throw(ArgumentError(
    "an image's detail level is one of $(join(repr.(_IMAGE_DETAILS), ", ")), or nothing for none (got $(repr(level)))"))

"""
    ImageAttachment(path::AbstractString; detail=nothing)
    ImageAttachment(data::AbstractVector{UInt8}; detail=nothing)

An image for a user [`Message`](@ref): PNG, JPEG, GIF or WebP, recognised from its
leading bytes. `mime` holds the media type (`"image/png"`, …). `detail` is the
level of detail an OpenAI model receives the image at — `"auto"`, `"low"`, `"high"`
or `"original"`, the image at its original dimensions, on the models that accept that
level, such as GPT-5.6
([image inputs](https://developers.openai.com/api/docs/guides/images-vision)) — or
`nothing`, which sends none and leaves the model's default. The OpenAI Chat
Completions wire sends it as the part's `detail`; an endpoint without an equivalent
refuses an image that sets one. Throws `ArgumentError` for empty data, bytes in any
other format, or any other `detail`.

```julia
msg = Message(Val(:user), "What colour is the shape?",
              ImageAttachment("shape.png"))
msg = Message(Val(:user), "Read the small print.",
              ImageAttachment("label.png"; detail="original"))
```
"""
struct ImageAttachment <: Attachment
    data::Vector{UInt8}
    mime::String
    detail::Union{String,Nothing}
    function ImageAttachment(data::AbstractVector{UInt8}; detail=nothing)
        mime = _sniff_image_mime(data)
        isnothing(mime) && throw(ArgumentError(
            "image data is not PNG, JPEG, GIF or WebP (checked by file signature; $(length(data)) bytes)"))
        new(Vector{UInt8}(data), mime, _image_detail(detail))
    end
end
ImageAttachment(path::AbstractString; detail=nothing) = ImageAttachment(read(_attachment_file(path)); detail)

"""
    AudioAttachment(path::AbstractString)
    AudioAttachment(data::AbstractVector{UInt8})

A sound clip for a user [`Message`](@ref): WAV or MP3, recognised from its leading
bytes. `format` holds `"wav"` or `"mp3"`, the names the OpenAI `input_audio` part
uses. Throws `ArgumentError` for empty data or bytes in any other format.

```julia
msg = Message(Val(:user), "Transcribe this.",
              AudioAttachment("clip.wav"))
```
"""
struct AudioAttachment <: Attachment
    data::Vector{UInt8}
    format::String
    function AudioAttachment(data::AbstractVector{UInt8})
        format = _sniff_audio_format(data)
        isnothing(format) && throw(ArgumentError(
            "audio data is not WAV or MP3 (checked by file signature; $(length(data)) bytes)"))
        new(Vector{UInt8}(data), format)
    end
end
AudioAttachment(path::AbstractString) = AudioAttachment(read(_attachment_file(path)))

"""
    DocumentAttachment(path::AbstractString)
    DocumentAttachment(data::AbstractVector{UInt8}; filename="document.pdf")

A PDF document for a user [`Message`](@ref), recognised from its leading bytes
(`%PDF-`). `mime` is `"application/pdf"`; `filename` is the name the provider is
given for it: the file's own name (`basename(path)`) or, for bytes, the keyword.
Endpoints on the OpenAI Chat Completions wire send it as a `file` part
([file inputs](https://developers.openai.com/api/docs/guides/file-inputs)); for the
Responses API, [`input_file`](@ref) turns it into an `input_file` part, and the native
Anthropic encoder sends it as a base64 `document` block. See
[`Attachment`](@ref) for the endpoints that refuse it. Throws `ArgumentError` for
bytes that are not PDF, whatever the file is called, and for a blank `filename`.

```julia
msg = Message(Val(:user), "What does the report conclude?",
              DocumentAttachment("report.pdf"))
```
"""
struct DocumentAttachment <: Attachment
    data::Vector{UInt8}
    mime::String
    filename::String
    function DocumentAttachment(data::AbstractVector{UInt8}; filename::AbstractString="document.pdf")
        mime = _sniff_document_mime(data)
        isnothing(mime) && throw(ArgumentError(
            "document data is not PDF (checked by file signature; $(length(data)) bytes)"))
        isempty(strip(filename)) && throw(ArgumentError(
            "a document's filename names it to the provider and must not be blank (got $(repr(String(filename))))"))
        new(Vector{UInt8}(data), mime, String(filename))
    end
end
DocumentAttachment(path::AbstractString) = DocumentAttachment(read(_attachment_file(path)); filename=basename(path))

# Attachments hold bytes, and not every provider fetches URLs (Ollama does not): a URL
# given where a file path is expected is named as such instead of failing as a missing file.
# A data URL (RFC 2397; URL schemes are case-insensitive) holds the bytes themselves, so it is
# named by its media type, not its payload.
function _attachment_file(path::AbstractString)
    data = match(r"^data:[^,]*"i, path)
    isnothing(data) || throw(ArgumentError(
        "attachments are read from a file or from bytes, not a data URL ($(repr(data.match * ",…"))): pass the " *
        "bytes it holds (for a `;base64` URL, `base64decode` of the text after its comma)"))
    occursin(r"^[A-Za-z][A-Za-z0-9+.-]*://", path) && throw(ArgumentError(
        "attachments are read from a file or from bytes, not a URL ($(repr(String(path)))); download it " *
        "first (e.g. with Downloads.download) and pass the file or its bytes"))
    path
end

# The bytes stay reachable through the fields; printing them would flood the REPL.
Base.show(io::IO, a::ImageAttachment) = print(io, "ImageAttachment(", a.mime, ", ", length(a.data), " bytes",
    isnothing(a.detail) ? "" : string(", detail=", repr(a.detail)), ")")
Base.show(io::IO, a::AudioAttachment) = print(io, "AudioAttachment(", a.format, ", ", length(a.data), " bytes)")
Base.show(io::IO, a::DocumentAttachment) =
    print(io, "DocumentAttachment(", a.mime, ", ", a.filename, ", ", length(a.data), " bytes)")
Base.:(==)(a::A, b::A) where {A<:Attachment} = all(f -> getfield(a, f) == getfield(b, f), fieldnames(A))
Base.hash(a::Attachment, h::UInt) = foldl((h, f) -> hash(getfield(a, f), h), fieldnames(typeof(a)); init=hash(typeof(a), h))

# The format of an audio reply by the file signature its bytes begin with: those an
# `AudioAttachment` is read by (WAV, MP3), FLAC's `fLaC` stream marker, an Ogg page whose
# first packet is the Opus ID header `OpusHead` (RFC 3533, RFC 7845), or an AAC ADTS frame
# header (sync word 0xFFF, layer 0). Raw samples, such as `pcm16`'s, begin with none.
function _audio_output_format(data::AbstractVector{UInt8})::Union{String,Nothing}
    known = _sniff_audio_format(data)
    isnothing(known) || return known
    n = length(data)
    n >= 4 && view(data, 1:4) == b"fLaC" && return "flac"
    if n >= 28 && view(data, 1:4) == b"OggS"
        packet = 28 + Int(data[27])          # after the 27-byte page header and its lacing values
        n >= packet + 7 && view(data, packet:packet + 7) == b"OpusHead" && return "opus"
    end
    n >= 2 && data[1] == 0xff && data[2] & 0xf6 == 0xf0 && return "aac"
    nothing
end

"""
    AudioOutput(; id, data, expires_at, transcript)

The audio an OpenAI model answers a Chat Completions turn with, on the assistant
[`Message`](@ref)'s `audio_output`. A [`Chat`](@ref) asks for it with
`modalities=["text", "audio"]` and `audio=Dict("voice" => …, "format" => …)` on a model
that speaks, such as `gpt-audio-1.5`
([Audio in Chat Completions](https://developers.openai.com/api/docs/guides/audio-chat-completions)).

- `id::String`: OpenAI's id for the audio, by which a later turn refers to it.
- `data::Vector{UInt8}`: the audio, decoded from base64, in the format the request asked for.
- `format::Union{String,Nothing}`: that format as the bytes' file signature shows it —
  `"wav"`, `"mp3"`, `"flac"`, `"opus"` (Ogg Opus) or `"aac"` (ADTS) — or `nothing` for
  bytes that begin with none, such as `"pcm16"`'s raw samples.
- `expires_at::Union{Int,Nothing}`: the Unix time, in seconds, until which OpenAI keeps the
  audio for later turns; after it, OpenAI answers a turn that sends the id back with HTTP
  400 `audio_not_found`, the answer observed for an id it never issued. `nothing` when it
  is unknown: the reply names none, or a streamed reply's expiry, which comes alone in its
  last update after the turn's finish, was lost to a connection that closed between the
  two; the turn is kept either way.
- `transcript::String`: what the audio says; the turn's `content` too, unless the turn
  holds other text.

The OpenAI-wire decoders read it from a reply's `message.audio`; a streamed reply
assembles it from its `delta.audio` chunks, and the transcript reaches the streaming
callback as text. When the conversation goes on, the OpenAI Chat encoder
(`OPENAIServiceEndpoint`) sends the turn back as
`{"role": "assistant", "content": content, "audio": {"id": id}}`, the turn's content
beside the id. Every other encoder — Azure, the Gemini OpenAI-compatible endpoint,
DeepSeek, generic endpoints, [`OllamaEndpoint`](@ref), native Anthropic and Gemini — sends
the turn's content alone, the neutral form it also sends for another provider's
[`ProviderContent`](@ref). Throws `ArgumentError` for an empty `id`.
"""
struct AudioOutput
    id::String
    data::Vector{UInt8}
    format::Union{String,Nothing}
    expires_at::Union{Int,Nothing}
    transcript::String
    function AudioOutput(id::AbstractString, data::AbstractVector{UInt8}, expires_at::Union{Integer,Nothing},
                         transcript::AbstractString)
        isempty(id) && throw(ArgumentError("an audio reply's id names it to OpenAI in later turns and cannot be empty"))
        new(String(id), Vector{UInt8}(data), _audio_output_format(data),
            isnothing(expires_at) ? nothing : Int(expires_at), String(transcript))
    end
end
AudioOutput(; id::AbstractString, data::AbstractVector{UInt8}, expires_at::Union{Integer,Nothing},
            transcript::AbstractString) = AudioOutput(id, data, expires_at, transcript)

Base.show(io::IO, a::AudioOutput) = print(io, "AudioOutput(", repr(a.id), ", ", something(a.format, "no file signature"),
    ", ", length(a.data), " bytes, expires_at=", a.expires_at, ", transcript=", repr(a.transcript), ")")
Base.:(==)(a::AudioOutput, b::AudioOutput) = all(f -> getfield(a, f) == getfield(b, f), fieldnames(AudioOutput))
Base.hash(a::AudioOutput, h::UInt) = foldl((h, f) -> hash(getfield(a, f), h), fieldnames(AudioOutput); init=hash(AudioOutput, h))

"""
    Message(; role, content=nothing, name=nothing,
              finish_reason=nothing, refusal_message=nothing,
              tool_calls=nothing, tool_call_id=nothing,
              provider_content=nothing, attachments=nothing, audio_output=nothing)

Represents a single message in a Chat Completions conversation.

# Fields
- `role::String`: One of [`RoleSystem`](@ref), [`RoleUser`](@ref), [`RoleAssistant`](@ref), or `RoleTool`.
- `content::Union{String,Nothing}`: The text content of the message.
- `name::Union{String,Nothing}`: Optional name for the participant.
- `finish_reason::Union{String,Nothing}`: Why the model stopped generating (e.g. `"stop"`, `"tool_calls"`); `nothing` when the provider reported none. Response-only: never sent in a request.
- `refusal_message::Union{String,Nothing}`: Refusal text when content is filtered; sent on the wire as `refusal`.
- `tool_calls::Union{Nothing,Vector{ToolCall}}`: Tool calls requested by the assistant.
- `tool_call_id::Union{String,Nothing}`: Required when `role` is `"tool"` — the ID of the tool call being responded to.
- `provider_content::Union{Nothing,ProviderContent}`: Provider-native content blocks captured for verbatim round-trip (see [`ProviderContent`](@ref)); set by the Anthropic, Gemini, DeepSeek and Ollama decoders (tags `:anthropic`, `:gemini`, `:deepseek`, `:ollama`), `nothing` otherwise. Never serialized on the OpenAI wire.
- `attachments::Union{Nothing,Vector{Attachment}}`: Images, sound clips and PDF documents sent with a user message ([`ImageAttachment`](@ref), [`AudioAttachment`](@ref), [`DocumentAttachment`](@ref)); `nothing` for none. See [`Attachment`](@ref) for how each endpoint sends or refuses them.
- `audio_output::Union{Nothing,AudioOutput}`: The audio an OpenAI model answered with ([`AudioOutput`](@ref)), set by the OpenAI-wire decoders; `nothing` otherwise. Its docstring says how each encoder sends the turn back.

# Validation
- `role` must be one of `"system"`, `"user"`, `"assistant"`, `"tool"`.
- At least one of `content`, `tool_calls`, `refusal_message`, `attachments` or `audio_output` must be given.
- `tool_call_id` is required when `role == "tool"`.
- `attachments` are accepted on user messages only; an empty vector is stored as `nothing`.
- `audio_output` is accepted on assistant messages only; with `content` unset or empty, the turn's content is its transcript.

# Convenience Constructors
```julia
Message(Val(:system), "You are a helpful assistant")
Message(Val(:user), "Hello!")
Message(Val(:user), "What colour is the shape?",
        ImageAttachment("shape.png"))
```
"""
@kwdef struct Message
    role::String
    content::Union{String,Nothing} = nothing
    name::Union{String,Nothing} = nothing
    finish_reason::Union{String,Nothing} = nothing
    refusal_message::Union{String,Nothing} = nothing
    tool_calls::Union{Nothing,Vector{ToolCall}} = nothing
    tool_call_id::Union{String,Nothing} = nothing
    provider_content::Union{Nothing,ProviderContent} = nothing
    attachments::Union{Nothing,Vector{Attachment}} = nothing
    audio_output::Union{Nothing,AudioOutput} = nothing
    function Message(role, content, name, finish_reason, refusal_message, tool_calls,
                     tool_call_id, provider_content=nothing, attachments=nothing, audio_output=nothing)
        role in _MESSAGE_ROLES || throw(ArgumentError(
            "message role must be one of $(join(_MESSAGE_ROLES, ", ")) (got $(repr(role)))"))
        isnothing(audio_output) || role == RoleAssistant || throw(ArgumentError(
            "audio_output is accepted on assistant messages only (got a :$role message)"))
        # An audio reply's text is its transcript, so the turn reads the same on every wire:
        # no content and empty content are the same no text, as the decoders read them.
        !isnothing(audio_output) && (isnothing(content) || isempty(content)) && (content = audio_output.transcript)
        isnothing(content) && isnothing(tool_calls) && isnothing(refusal_message) &&
            (isnothing(attachments) || isempty(attachments)) && throw(ArgumentError(
                "`content`, `tool_calls`, `refusal_message`, `attachments` and `audio_output` cannot all be empty"))
        role == RoleTool && isnothing(tool_call_id) && throw(ArgumentError("`tool_call_id` cannot be empty when role is `tool`"))
        isnothing(attachments) || !isempty(attachments) || (attachments = nothing)   # no media is no media list
        isnothing(attachments) || role == RoleUser || throw(ArgumentError(
            "attachments are accepted on user messages only (got a :$role message with $(length(attachments)))"))
        return new(role, content, name, finish_reason, refusal_message, tool_calls,
                   tool_call_id, provider_content, attachments, audio_output)
    end
end

Message(::Val{:system}, content) = Message(role=RoleSystem, content=content)
Message(::Val{:user}, content) = Message(role=RoleUser, content=content)
Message(::Val{:user}, content, attachment::Attachment, more::Attachment...) =
    Message(role=RoleUser, content=content, attachments=Attachment[attachment, more...])

const Conversation = Vector{Message}

JSON.omit_null(::Type{Message}) = true

# The request wire form. provider_content is a decode-side round-trip cache for
# provider-native blocks (same precedent as ToolCall.thought_signature) and
# finish_reason is a response-only field: neither is sent; audio_output goes back by
# its id on OpenAI's own wire only (`_wire_messages`). A refusal travels under
# its wire name, `refusal`; an assistant message needs `content` unless it carries
# tool calls, so a refusal turn sends it as null — the shape the API returns it in.
# Attachments turn `content` into an array of parts: the text (when there is any),
# then each attachment as the part its kind takes (`_wire_part`).
# Conditional insertion mirrors omit-null.
function JSON.lower(m::Message)
    d = Dict{Symbol,Any}(:role => m.role)
    (isnothing(m.content) && isnothing(m.refusal_message)) || (d[:content] = m.content)
    isnothing(m.attachments) || (d[:content] = _content_parts(m.content, m.attachments))
    isnothing(m.name)            || (d[:name] = m.name)
    isnothing(m.refusal_message) || (d[:refusal] = m.refusal_message)
    isnothing(m.tool_calls)      || (d[:tool_calls] = m.tool_calls)
    isnothing(m.tool_call_id)    || (d[:tool_call_id] = m.tool_call_id)
    d
end

# The OpenAI Chat Completions content part of each kind
# (https://developers.openai.com/api/reference/resources/chat): an image is an `image_url`
# data URL, with its `detail` level only when it sets one; a clip is `input_audio`; a
# document is a `file` part holding its name and a data URL, the form OpenAI's file-inputs
# guide gives for base64 PDFs.
_data_url(a::Union{ImageAttachment,DocumentAttachment})::String = string("data:", a.mime, ";base64,", base64encode(a.data))
function _wire_part(a::ImageAttachment)::Dict{Symbol,Any}
    image_url = Dict{Symbol,Any}(:url => _data_url(a))
    isnothing(a.detail) || (image_url[:detail] = a.detail)
    Dict{Symbol,Any}(:type => "image_url", :image_url => image_url)
end
_wire_part(a::AudioAttachment) = Dict{Symbol,Any}(:type => "input_audio",
    :input_audio => Dict(:data => base64encode(a.data), :format => a.format))
_wire_part(a::DocumentAttachment) = Dict{Symbol,Any}(:type => "file",
    :file => Dict(:filename => a.filename, :file_data => _data_url(a)))

# An attachment goes out only in the shape its wire gives it — an `image_url`, `input_audio`
# or `file` part, an Ollama `images` entry, a Gemini part, a System One asset — and every path
# that carries one maps it explicitly. Written by JSON.jl's struct lowering, its bytes would go
# out as an array of numbers that no service reads as media; reaching these methods means
# the attachment sits where nothing maps it — an OpenAI Responses input, a System One request
# that cannot carry its kind, a struct of the caller's own — so each names where its kind goes.
JSON.lower(a::ImageAttachment) = throw(ArgumentError(
    "an ImageAttachment cannot be written as JSON: send it in a user message's `attachments`, in an " *
    "`InputMessage`'s content on Gemini Interactions (`respond` on `GEMINIServiceEndpoint`), or in a System One " *
    "request's state, instructions or criteria (directly, or inside a Dict, vector, tuple or NamedTuple; not " *
    "inside a struct of your own). An OpenAI Responses input takes an image as " *
    "`input_image(\"data:$(a.mime);base64,…\")`, its bytes as a base64 data URL"))
JSON.lower(::AudioAttachment) = throw(ArgumentError(
    "an AudioAttachment cannot be written as JSON: send it in a user message's `attachments`, or in an " *
    "`InputMessage`'s content on Gemini Interactions (`respond` on `GEMINIServiceEndpoint`); a System One " *
    "request carries no audio"))
JSON.lower(::DocumentAttachment) = throw(ArgumentError(
    "a DocumentAttachment cannot be written as JSON: send it in a user message's `attachments`, or as " *
    "`input_file(document)` in a Responses input; a System One request carries no documents"))

function _content_parts(content::Union{String,Nothing}, attachments::Vector{Attachment})::Vector{Dict{Symbol,Any}}
    parts = Dict{Symbol,Any}[_wire_part(a) for a in attachments]
    isnothing(content) || isempty(content) || pushfirst!(parts, Dict{Symbol,Any}(:type => "text", :text => content))
    parts
end

"""
getcontent(m::Message)::Union{String,Nothing}

Get the content of the message.
"""
getcontent(m::Message) = m.content

"""
getrole(m::Message)::String
Get the role of the message.
"""
getrole(m::Message) = m.role

"""
iscall(m::Message)::Bool
Check if the message is a tool call.
"""
iscall(m::Message) = m.role == RoleTool

@kwdef struct JsonSchemaAPI
    name::String
    description::String
    schema::AbstractDict
    strict::Union{Bool,Nothing} = nothing
end

# pre-0.10.3 positional arity (@kwdef defaults apply only to the keyword constructor)
JsonSchemaAPI(name, description, schema) = JsonSchemaAPI(name, description, schema, nothing)

JSON.omit_null(::Type{JsonSchemaAPI}) = true

"""
    ResponseFormat(; type="json_object", json_schema=nothing)
    ResponseFormat(json_schema)

Specifies the output format for Chat Completions.

# Fields
- `type::String`: `"json_object"` or `"json_schema"`.
- `json_schema::Union{JsonSchemaAPI,AbstractDict,Nothing}`: Schema definition when `type` is `"json_schema"`.

# Examples
```julia
# Free-form JSON
fmt = ResponseFormat()

# Structured JSON via schema
fmt = ResponseFormat(UniLM.JsonSchemaAPI(
    name="result",
    description="A structured result",
    schema=Dict("type" => "object", "properties" => Dict())
))
```

!!! note
    `JsonSchemaAPI` and the convenience constructors `UniLM.json_object()` and
    `UniLM.json_schema(name, description, schema; strict=nothing)` are not exported:
    qualify them with `UniLM.`.
"""
@kwdef struct ResponseFormat
    type::String = "json_object"
    json_schema::Union{JsonSchemaAPI,AbstractDict,Nothing} = nothing
end

ResponseFormat(json_schema) = ResponseFormat("json_schema", json_schema)

JSON.omit_null(::Type{ResponseFormat}) = true

json_object() = ResponseFormat()
json_schema(schema) = ResponseFormat(schema)
json_schema(name::String, description::String, schema::AbstractDict; strict::Union{Bool,Nothing}=nothing) = ResponseFormat(JsonSchemaAPI(name, description, schema, strict))

"""
    ServiceEndpoint

Abstract supertype for LLM service backends. Subtypes control URL routing and authentication.

OpenAI-compatible backends subtype `OpenAIWireEndpoint` (itself a subtype of
`ServiceEndpoint`) to inherit the chat request/response encoding and SSE handling;
backends with a native wire (Anthropic, Gemini) subtype `ServiceEndpoint` directly
and implement the wire seam (`encode_request`/`decode_response`/`handle_sse_event!`)
themselves.

Built-in subtypes:
- `OPENAIServiceEndpoint` — OpenAI API (default)
- `AZUREServiceEndpoint` — Azure OpenAI Service
- `GEMINIOpenAIServiceEndpoint` — Google Gemini via OpenAI-compatible endpoint
- `GEMINIServiceEndpoint` — Google Gemini native generateContent API
- `ANTHROPICServiceEndpoint` — Anthropic (Claude) native Messages API
- `OllamaEndpoint` — a local Ollama server (native chat and embeddings API)
- `GenericOpenAIEndpoint` — any OpenAI-compatible provider (Mistral, vLLM, LM Studio, etc.)
"""
abstract type ServiceEndpoint end

"""
    OpenAIWireEndpoint <: ServiceEndpoint

Abstract supertype for backends that speak the OpenAI-compatible chat wire.
Subtypes inherit the OpenAI Chat Completions request/response encoding
(`encode_request`/`decode_response`) and the default SSE stream handling
(`handle_sse_event!`) for free, so a new OpenAI-compatible provider defines only
`get_url` and `auth_header` (plus, optionally, the `_api_base_url` pattern that
routes the Responses/agentic surface).

Backends that speak a genuinely different wire (Anthropic's Messages API, Gemini's
native `generateContent`) subtype `ServiceEndpoint` directly and additionally
implement `encode_request`, `decode_response`, and `handle_sse_event!`. A bare
`ServiceEndpoint` subtype that omits those methods fails with a `MethodError` at
call time rather than silently emitting OpenAI-shaped requests to a foreign API.

Built-in OpenAI-wire subtypes: `OPENAIServiceEndpoint`, `AZUREServiceEndpoint`,
`GEMINIOpenAIServiceEndpoint`, `GenericOpenAIEndpoint`, `DeepSeekEndpoint`, and
`OllamaEndpoint`, which overrides the chat seam with Ollama's native wire and keeps the
OpenAI-wire Responses and completions routes.
"""
abstract type OpenAIWireEndpoint <: ServiceEndpoint end

"""OpenAI API service endpoint (default). Requires `OPENAI_API_KEY` env variable."""
struct OPENAIServiceEndpoint <: OpenAIWireEndpoint end

"""Azure OpenAI Service endpoint. Requires `AZURE_OPENAI_BASE_URL`, `AZURE_OPENAI_API_KEY`, and `AZURE_OPENAI_API_VERSION` env variables."""
struct AZUREServiceEndpoint <: OpenAIWireEndpoint end

"""Google Gemini endpoint (OpenAI-compatible). Requires `GEMINI_API_KEY` env variable."""
struct GEMINIOpenAIServiceEndpoint <: OpenAIWireEndpoint end

"""Native Google Gemini `generateContent` API (`x-goog-api-key`; model in URL). Requires `GEMINI_API_KEY`."""
struct GEMINIServiceEndpoint <: ServiceEndpoint end

# The inference regions of a Messages API request
# (https://platform.claude.com/docs/en/manage-claude/data-residency).
const _ANTHROPIC_INFERENCE_GEOS = ("global", "us")

# A header value goes out as given: a blank one, or one holding a control character (CR and
# LF end a header line), is refused rather than sent.
function _anthropic_header_value(what::String, v)::String
    v isa AbstractString || throw(ArgumentError("an Anthropic $what is a string (got $(repr(v)))"))
    isempty(strip(v)) && throw(ArgumentError("an Anthropic $what must not be blank (got $(repr(v)))"))
    any(iscntrl, v) && throw(ArgumentError(
        "an Anthropic $what must not hold a control character such as CR or LF (got $(repr(v)))"))
    String(v)
end
_anthropic_optional_header(what::String, v)::Union{String,Nothing} = isnothing(v) ? nothing : _anthropic_header_value(what, v)

# The names an anthropic-beta header carries, comma-joined: each a header value, and none
# holding the comma that would split it in two.
function _anthropic_beta_names(betas::AbstractVector)::Vector{String}
    names = String[_anthropic_header_value("beta name", b) for b in betas]
    comma = findfirst(contains(','), names)
    isnothing(comma) || throw(ArgumentError("an Anthropic beta name cannot hold a comma, which separates " *
        "the names in the anthropic-beta header (got $(repr(names[comma])))"))
    names
end

"""
    ANTHROPICServiceEndpoint
    ANTHROPICServiceEndpoint(; betas=String[], user_profile_id=nothing, workspace_id=nothing,
                             inference_geo=nothing)

Anthropic (Claude) native Messages API endpoint. Requires the `ANTHROPIC_API_KEY` env variable.
Native wire format (content blocks, top-level `system`, `user`/`assistant` roles) — NOT OpenAI-compatible.

The type itself (`service=ANTHROPICServiceEndpoint`) is the endpoint with no options set. An
instance adds options that every request through it carries
([Messages API](https://platform.claude.com/docs/en/api/messages)):

- `betas`: beta feature names, sent comma-joined in the `anthropic-beta` header
  ([beta headers](https://platform.claude.com/docs/en/api/beta-headers)). UniLM sends the
  names and models no beta's request fields or response blocks; the one it reads is
  `"fast-mode-2026-02-01"`, which `service_tier="fast"` needs on Claude Opus 5 and Opus 4.8.
  [`list_models`](@ref) sends no `anthropic-beta` header: Anthropic's Models API reference
  marks it deprecated on `GET /v1/models`.
- `user_profile_id`: the `anthropic-user-profile-id` header, the `uprof_` id of the end
  customer the requests are made for
  ([user profiles](https://platform.claude.com/docs/en/manage-claude/user-profiles)).
  [`list_models`](@ref) does not send it: Anthropic's Models API reference names no such
  header for `GET /v1/models`.
- `workspace_id`: the `anthropic-workspace-id` header, the workspace of a credential that
  can act on more than one.
- `inference_geo`: `"global"` or `"us"`, the request's `inference_geo`, where inference
  runs ([data residency](https://platform.claude.com/docs/en/manage-claude/data-residency));
  `nothing` leaves the workspace default. The Claude 4.5 models
  (Opus 4.5, Sonnet 4.5, Haiku 4.5) and earlier answer it with HTTP 400, so a request for one
  of them throws `ArgumentError`.

[`estimated_cost`](@ref) bills US-only inference (a usage that reports `inference_geo` `"us"`)
at 1.1× every rate on Claude 4.6 and later; an earlier model is billed at the standard rates,
whatever region its usage reports
([data residency pricing](https://platform.claude.com/docs/en/about-claude/pricing#data-residency-pricing)).
A reply at Priority Tier (a usage that reports `service_tier` `"priority"`) is estimated at
the list rates, as at the standard tier: Priority Tier draws down an organization's
contracted capacity commitment ([service tiers](https://platform.claude.com/docs/en/api/service-tiers)).

The constructor throws `ArgumentError` for a blank value, a value that holds a control
character (CR, LF, …), a beta name that holds a comma, and an `inference_geo` other than
`"global"` or `"us"`. Instances with the same options are equal (`==`, `hash`).

The fields are read-only: to change an option, build a new endpoint. `betas` is copied at
construction but is a `Vector`, so it can still be changed in place; that changes the
endpoint's `hash` (a `Dict` or `Set` holding it no longer finds it), and the beta names
are checked again by the same rule when the header is built, so a name changed in place
that breaks it is refused before anything is sent (the verb returns the `ArgumentError`
as the `cause` of its call-error result).

```julia
claude = ANTHROPICServiceEndpoint(user_profile_id="uprof_011CZkZCu8hGbp5mYRQgUmz9", inference_geo="us")
chat = Chat(service=claude, model="claude-haiku-5-5")
```
"""
struct ANTHROPICServiceEndpoint <: ServiceEndpoint
    betas::Vector{String}
    user_profile_id::Union{String,Nothing}
    workspace_id::Union{String,Nothing}
    inference_geo::Union{String,Nothing}
    function ANTHROPICServiceEndpoint(betas::AbstractVector, user_profile_id, workspace_id, inference_geo)
        names = _anthropic_beta_names(betas)
        isnothing(inference_geo) || inference_geo in _ANTHROPIC_INFERENCE_GEOS || throw(ArgumentError(
            "Anthropic inference_geo is one of $(join(repr.(_ANTHROPIC_INFERENCE_GEOS), ", ")), or nothing for " *
            "the workspace default (got $(repr(inference_geo)))"))
        new(names, _anthropic_optional_header("user_profile_id", user_profile_id),
            _anthropic_optional_header("workspace_id", workspace_id),
            isnothing(inference_geo) ? nothing : String(inference_geo))
    end
end
ANTHROPICServiceEndpoint(; betas::AbstractVector=String[], user_profile_id=nothing, workspace_id=nothing,
                         inference_geo=nothing) =
    ANTHROPICServiceEndpoint(betas, user_profile_id, workspace_id, inference_geo)

Base.:(==)(a::ANTHROPICServiceEndpoint, b::ANTHROPICServiceEndpoint) =
    all(f -> getfield(a, f) == getfield(b, f), fieldnames(ANTHROPICServiceEndpoint))
Base.hash(e::ANTHROPICServiceEndpoint, h::UInt) =
    foldl((h, f) -> hash(getfield(e, f), h), fieldnames(ANTHROPICServiceEndpoint); init=hash(ANTHROPICServiceEndpoint, h))

# The keyword call that builds the endpoint, its unset options left out. It holds no
# credential (the key is read from the environment when a request is sent).
function Base.show(io::IO, e::ANTHROPICServiceEndpoint)
    set = isempty(e.betas) ? String[] : ["betas=" * repr(e.betas)]
    for f in (:user_profile_id, :workspace_id, :inference_geo)
        v = getfield(e, f)
        isnothing(v) || push!(set, string(f, "=", repr(v)))
    end
    print(io, "ANTHROPICServiceEndpoint(", join(set, ", "), ")")
end

"""
    GenericOpenAIEndpoint <: OpenAIWireEndpoint
    GenericOpenAIEndpoint(base_url, api_key, key_env="")

Configurable endpoint for any OpenAI-compatible API provider. Supports Chat Completions,
Embeddings, and (where the provider implements it) the Responses API.

# Fields
- `base_url::String`: Base URL without trailing slash (e.g., `"http://localhost:11434"`)
- `api_key::String`: API key for Bearer auth, sent as is when not empty.
- `key_env::String`: The environment variable that holds the key when `api_key` is empty,
  read each time a request is built, so a missing variable fails that request, naming it;
  `""` reads none.

An empty key, given or read, sends no `Authorization` header: use `""` for local servers
with no auth.

# Example
```julia
# vLLM (local)
chat = Chat(service=GenericOpenAIEndpoint("http://localhost:8000", ""),
            model="Qwen/Qwen3-8B")

# Mistral: GenericOpenAIEndpoint("https://api.mistral.ai", "", "MISTRAL_API_KEY")
chat = Chat(service=MistralEndpoint(), model="mistral-large-latest")
```
"""
struct GenericOpenAIEndpoint <: OpenAIWireEndpoint
    base_url::String
    api_key::String
    key_env::String
end
GenericOpenAIEndpoint(base_url, api_key) = GenericOpenAIEndpoint(base_url, api_key, "")

"""
    ServiceEndpointSpec

Type alias accepting both marker types (`OPENAIServiceEndpoint`) and instances
(`GenericOpenAIEndpoint(...)`). Used as the type of `service` fields.
"""
const ServiceEndpointSpec = Union{Type{<:ServiceEndpoint}, ServiceEndpoint}

# OpenAI-wire counterpart of `ServiceEndpointSpec`: the marker-type-or-instance
# domain of `OpenAIWireEndpoint`, used to type the OpenAI-wire seam defaults
# (chat: `encode_request`/`decode_response`/`handle_sse_event!`; agentic:
# `encode_agentic`/`decode_agentic`/`decode_agentic_stream`/`_agentic_url`).
# Internal — deliberately unexported.
const OpenAIWireEndpointSpec = Union{Type{<:OpenAIWireEndpoint}, OpenAIWireEndpoint}

"""
    OllamaOptions(; num_ctx=nothing, num_batch=nothing, num_gpu=nothing,
                  main_gpu=nothing, num_thread=nothing, use_mmap=nothing,
                  num_keep=nothing, top_k=nothing, min_p=nothing,
                  repeat_last_n=nothing, repeat_penalty=nothing)

The Ollama runtime and sampling options that [`Chat`](@ref) has no field for, sent as
the `options` object of an [`OllamaEndpoint`](@ref) request; unset options are
omitted, so the model's own defaults apply. Options `Chat` already names
(`temperature`, `top_p`, `seed`, `stop`, `max_tokens`, `presence_penalty`,
`frequency_penalty`) stay on the `Chat`, so no option can be set in two places.

- `num_ctx`: context window in tokens. Ollama loads the model with this window; a model with
  vision gets at least 2,048 tokens (`embeddinggemma-2:740m-q8_0` asked for 1,024 loaded with
  2,048 on Ollama 0.40.6 and 0.40.7; Ollama's server/sched.go). On the
  llama.cpp and ggml runners a request with another `num_ctx` reloads the model, and a
  prompt longer than the window is refused (HTTP 400) unless `truncate=true`; on the MLX
  runner (Apple Silicon) the window applies when the model is loaded, a loaded model
  keeps it when a later request asks for more, and it does not bound a chat prompt
  (verified with Ollama 0.40.6 on the llama.cpp and MLX runners; ggml untested).
- `num_batch`, `num_gpu` (layers offloaded to the GPU), `main_gpu`, `num_thread`,
  `use_mmap`: how the model runs on this machine.
- `num_keep`: tokens of the prompt kept when the context is shifted.
- `top_k`, `min_p`, `repeat_last_n`, `repeat_penalty`: sampling.

Every option is passed through unchanged; the effect of those other than `num_ctx` on the
MLX runner is not verified.

The constructor throws `ArgumentError` for a value out of range: `num_ctx`,
`num_batch` and `num_thread` must be positive, `top_k` non-negative (0 turns top-k off), `min_p` in
[0, 1], `repeat_penalty` non-negative, `repeat_last_n` and `num_keep` at least -1,
`num_gpu` at least -1 and `main_gpu` non-negative.
"""
@kwdef struct OllamaOptions
    num_ctx::Union{Int,Nothing} = nothing
    num_batch::Union{Int,Nothing} = nothing
    num_gpu::Union{Int,Nothing} = nothing
    main_gpu::Union{Int,Nothing} = nothing
    num_thread::Union{Int,Nothing} = nothing
    use_mmap::Union{Bool,Nothing} = nothing
    num_keep::Union{Int,Nothing} = nothing
    top_k::Union{Int,Nothing} = nothing
    min_p::Union{Float64,Nothing} = nothing
    repeat_last_n::Union{Int,Nothing} = nothing
    repeat_penalty::Union{Float64,Nothing} = nothing
    function OllamaOptions(num_ctx, num_batch, num_gpu, main_gpu, num_thread, use_mmap, num_keep,
                           top_k, min_p, repeat_last_n, repeat_penalty)
        _check(ok, what) = ok || throw(ArgumentError("Ollama option $what"))
        isnothing(num_ctx)        || _check(num_ctx > 0, "num_ctx must be positive (got $num_ctx)")
        isnothing(num_batch)      || _check(num_batch > 0, "num_batch must be positive (got $num_batch)")
        isnothing(num_gpu)        || _check(num_gpu >= -1, "num_gpu must be at least -1 (got $num_gpu)")
        isnothing(main_gpu)       || _check(main_gpu >= 0, "main_gpu must be non-negative (got $main_gpu)")
        isnothing(num_thread)     || _check(num_thread > 0, "num_thread must be positive (got $num_thread)")
        isnothing(num_keep)       || _check(num_keep >= -1, "num_keep must be at least -1 (got $num_keep)")
        isnothing(top_k)          || _check(top_k >= 0, "top_k must be non-negative (got $top_k)")
        isnothing(min_p)          || _check(0.0 <= min_p <= 1.0, "min_p must be in [0, 1] (got $min_p)")
        isnothing(repeat_last_n)  || _check(repeat_last_n >= -1, "repeat_last_n must be at least -1 (got $repeat_last_n)")
        isnothing(repeat_penalty) || _check(repeat_penalty >= 0.0, "repeat_penalty must be non-negative (got $repeat_penalty)")
        new(num_ctx, num_batch, num_gpu, main_gpu, num_thread, use_mmap, num_keep,
            top_k, min_p, repeat_last_n, repeat_penalty)
    end
end

# The set options, in field order, as the wire's `options` object.
_ollama_options(o::OllamaOptions)::Dict{String,Any} =
    Dict{String,Any}(String(f) => getfield(o, f) for f in fieldnames(OllamaOptions) if !isnothing(getfield(o, f)))

# `OLLAMA_HOST` → base URL, by the rules of Ollama's own clients: no scheme means
# http on port 11434 (https on 443 for ollama.com); an explicit http:// or https://
# without a port means 80 or 443; a missing host means 127.0.0.1; an IPv6 address may
# be bracketed or bare; a path is kept. Blank means the default server. A bind-all
# address (0.0.0.0, ::) tells the server to listen everywhere; a client reaches it on
# the loopback address, as the Ollama CLI does.
function _ollama_host_url(host::AbstractString)::String
    s = String(strip(host))
    isempty(s) && return "http://127.0.0.1:11434"
    # Groups that always take part in a match are asserted; only the port is optional.
    m = match(r"^([A-Za-z][A-Za-z0-9+.-]*)://(.*)$", s)
    given = isnothing(m) ? "http" : String(m.captures[1]::SubString{String})
    scheme = lowercase(given)
    scheme in ("http", "https") || throw(ArgumentError(
        "OLLAMA_HOST scheme must be http or https (got $(repr(given)))"))
    rest = isnothing(m) ? s : String(m.captures[2]::SubString{String})
    port = isnothing(m) ? "11434" : scheme == "https" ? "443" : "80"
    i = findfirst('/', rest)
    hostport, path = isnothing(i) ? (rest, "") : (rest[1:prevind(rest, i)], String(rstrip(rest[i:end], '/')))
    count(==(':'), hostport) > 1 && !startswith(hostport, '[') && (hostport = "[" * hostport * "]")  # bare IPv6
    if isnothing(m) && lowercase(hostport) == "ollama.com"                                  # Ollama's cloud
        scheme, port = "https", "443"
    end
    h = match(r"^(\[[^\]]*\]|[^:]*)(?::(\d*))?$", hostport)
    isnothing(h) && throw(ArgumentError("OLLAMA_HOST is not host[:port] (got $(repr(String(host))))"))
    given_host = String(h.captures[1]::SubString{String})
    name = isempty(given_host) || given_host == "0.0.0.0" ? "127.0.0.1" :
           given_host in ("[::]", "[0:0:0:0:0:0:0:0]") ? "[::1]" : given_host
    p = h.captures[2]
    if !isnothing(p) && !isempty(p)
        n = tryparse(Int, p)
        (isnothing(n) || !(0 < n <= 65535)) && throw(ArgumentError("OLLAMA_HOST port must be 1-65535 (got $(repr(String(p))))"))
        port = String(p)
    end
    string(scheme, "://", name, ":", port, path)
end

# The runners an Ollama server picks between for a model it can run more than one way.
const _OLLAMA_RUNNERS = (:mlx, :llamacpp, :ggml)

"""
    OllamaEndpoint <: OpenAIWireEndpoint
    OllamaEndpoint(; base_url=<OLLAMA_HOST or http://127.0.0.1:11434>,
                   keep_alive=nothing, truncate=false, shift=nothing,
                   runner=nothing, options...)

A local (or remote) [Ollama](https://ollama.com) server. Chat and [`Embeddings`](@ref)
requests use Ollama's native `/api/chat` and `/api/embed`, which carry what the
OpenAI-compatible routes ignore: the context window and the other
[`OllamaOptions`](@ref), how long the model stays loaded (`keep_alive`), and the
`think` and `truncate` fields. [`respond`](@ref), [`compact_response`](@ref),
[`FIMCompletion`](@ref) and [`transcribe`](@ref) use the OpenAI-compatible routes
(`/v1/responses`, `/v1/responses/compact`, `/v1/completions`, `/v1/audio/transcriptions`),
which carry none of the endpoint's `keep_alive`, [`OllamaOptions`](@ref) or `runner`.

- `base_url`: server root, without `/v1` or `/api`. Defaults to the `OLLAMA_HOST`
  environment variable, read with the rules of Ollama's own clients (`"gpu-box"` →
  `http://gpu-box:11434`), else `http://127.0.0.1:11434`.
- `keep_alive`: seconds the model stays loaded after each request (`0` unloads it at
  once, `Inf` keeps it loaded); `nothing` leaves the server's default (5 minutes
  unless `OLLAMA_KEEP_ALIVE` says otherwise).
- `truncate`: whether Ollama may shorten an input to fit the context window. In chat it
  can cut inside a single over-long message, keeping its first tokens and its tail
  (Ollama 0.40.6: a 5,407-token message cut to 2,050 tokens, the first 4 kept), so the
  start of a long document is lost without a word, not only the oldest messages; in
  embeddings it drops the end of an input. `false` (the default) makes an input that does
  not fit a failure — HTTP 400 "exceeds the available context size" — instead of an
  answer to a silently shortened prompt; raise `num_ctx`, or set `true` for Ollama's
  own behaviour. On the MLX runner (Apple Silicon) the window does not bound a chat
  prompt: a longer one is not refused (verified with Ollama 0.40.6).
- `shift`: whether generation that fills the context window may continue by dropping
  earlier context (`nothing`: the server's default, on). `false` ends such a reply with
  finish reason `"length"`. Ollama reloads the model when this changes.
- `runner`: the runner to serve a model with — `:mlx` (Apple Silicon), `:llamacpp` or `:ggml`;
  `nothing` (the default) leaves the choice to the server. Ollama matches it against the runner the
  model's manifest names, the one [`list_models`](@ref) reports as `OllamaModel.runner`. A model
  published for several runners is served by its variant for the one pinned (by Ollama's source;
  untested). A model published for one — every model measured — answers as if unpinned when that runner
  is pinned, and any other is the server's error, never a fallback: HTTP 400 `no compatible manifest
  found for runners: <runner>` (404 from [`model_info`](@ref), and so from [`load_model`](@ref)),
  even while the model is loaded (Ollama 0.40.6: `:llamacpp` on `kev:0.8b-q8_0` answered and
  [`running_models`](@ref) reported `:llamacpp`; `:mlx` and `:ggml` on it, and `:llamacpp` on
  `tev1:0.8b`, were that 400). Pin the runner the listing names, not the one `running_models`
  reports: a GGUF model listed as `:ggml` (`qwen2.5-coder:1.5b-base`, `embeddinggemma:latest`) answers
  pinned to `:ggml`, is served by llama.cpp, and is the 400 pinned to `:llamacpp`. The pin goes with
  every request of the endpoint, so a pinned endpoint fails for each model without a variant for its
  runner. Sent with chat, embeddings, System One, [`model_info`](@ref), [`load_model`](@ref),
  [`unload_model`](@ref) and [`pull_model`](@ref) requests; [`delete_model`](@ref), the listings and
  the OpenAI-compatible routes ([`respond`](@ref), [`compact_response`](@ref), FIM, transcription)
  take none.
- `options...`: keyword arguments of [`OllamaOptions`](@ref), e.g. `num_ctx=32_768`.
  An embeddings request for EmbeddingGemma 2 (and its [`load_model`](@ref)) carries its
  trained 8,192-token window, `num_ctx` and `num_batch` 8192, unless the endpoint sets
  either: at Ollama's default batch the llama.cpp builds measured refuse long inputs —
  `embeddinggemma-2:740m-q8_0` anything past about 2,037 tokens (Ollama 0.40.6).

No API key is sent. Local models cost nothing: [`estimated_cost`](@ref) reports `0.0`
for a chat, embeddings, `respond` or FIM result without a missing-price warning.
A cloud model reached through the local server — a model whose tag is `cloud` or ends in
`-cloud` (`clef:cloud`, `gpt-oss:120b-cloud`) — is billed by Ollama and is priced by the
name requested, never as a free local call: a System One, `respond` or FIM result whose
reply drops the tag (`gpt-oss:120b`) keeps the requested name, and the server's stays in
its `raw`, while a chat or embeddings result is priced by the model its request names; a
name missing from the price table logs the missing-price warning. A model is recognised as
cloud by that tag alone, so a cloud model copied under a name without it (`ollama cp`)
is priced as local, at `0.0`.

```julia
ollama = OllamaEndpoint(num_ctx=32_768, keep_alive=600)
chat = Chat(service=ollama, model="gemma4:e4b")
```
"""
struct OllamaEndpoint <: OpenAIWireEndpoint
    base_url::String
    keep_alive::Union{Float64,Nothing}
    truncate::Bool
    shift::Union{Bool,Nothing}
    runner::Union{Symbol,Nothing}
    options::OllamaOptions
    function OllamaEndpoint(base_url::AbstractString, keep_alive::Union{Real,Nothing}, truncate::Bool,
                            shift::Union{Bool,Nothing}, runner::Union{Symbol,Nothing}, options::OllamaOptions)
        url = String(rstrip(base_url, '/'))
        occursin(r"^https?://[^/]", url) || throw(ArgumentError(
            "Ollama base_url must start with http:// or https:// (got $(repr(String(base_url))))"))
        isnothing(keep_alive) || (!isnan(keep_alive) && 0 <= keep_alive && (isinf(keep_alive) || keep_alive <= 1e9)) ||
            throw(ArgumentError("keep_alive must be a number of seconds from 0 to 1e9, or Inf to keep " *
                                "the model loaded (got $keep_alive)"))
        isnothing(runner) || runner in _OLLAMA_RUNNERS || throw(ArgumentError(
            "Ollama runner must be one of $(join(repr.(_OLLAMA_RUNNERS), ", ")), or nothing for the server's choice " *
            "(got $(repr(runner)))"))
        new(url, isnothing(keep_alive) ? nothing : Float64(keep_alive), truncate, shift, runner, options)
    end
end

function OllamaEndpoint(; base_url::AbstractString=_ollama_host_url(get(ENV, OLLAMA_HOST, "")),
                        keep_alive::Union{Real,Nothing}=nothing, truncate::Bool=false,
                        shift::Union{Bool,Nothing}=nothing, runner::Union{Symbol,Nothing}=nothing, options...)
    unknown = setdiff(keys(options), fieldnames(OllamaOptions))
    isempty(unknown) || throw(ArgumentError("unknown OllamaEndpoint keyword(s) $(join(unknown, ", ")); " *
        "expected base_url, keep_alive, truncate, shift, runner or an OllamaOptions field: " *
        join(fieldnames(OllamaOptions), ", ")))
    OllamaEndpoint(base_url, keep_alive, truncate, shift, runner, OllamaOptions(; options...))
end

function Base.show(io::IO, e::OllamaEndpoint)
    print(io, "OllamaEndpoint(")
    show(io, e.base_url)
    set = Tuple{Symbol,Any}[(f, getfield(e.options, f)) for f in fieldnames(OllamaOptions)
                            if !isnothing(getfield(e.options, f))]
    isnothing(e.runner) || pushfirst!(set, (:runner, e.runner))
    isnothing(e.shift) || pushfirst!(set, (:shift, e.shift))
    e.truncate && pushfirst!(set, (:truncate, true))
    isnothing(e.keep_alive) || pushfirst!(set, (:keep_alive, e.keep_alive))
    isempty(set) || print(io, "; ", join(("$f=$(repr(v))" for (f, v) in set), ", "))
    print(io, ")")
end

"""
    MistralEndpoint(; api_key=nothing) -> GenericOpenAIEndpoint

Pre-configured endpoint for [Mistral AI](https://mistral.ai) API. A given `api_key` is sent
as is; without one the key is read from `MISTRAL_API_KEY` each time a request is built
(`GenericOpenAIEndpoint("https://api.mistral.ai", "", "MISTRAL_API_KEY")`), so building the
endpoint needs no key and a missing variable fails the request, naming it.
"""
MistralEndpoint(; api_key::Union{String,Nothing}=nothing) =
    isnothing(api_key) ? GenericOpenAIEndpoint("https://api.mistral.ai", "", MISTRAL_API_KEY) :
                         GenericOpenAIEndpoint("https://api.mistral.ai", api_key)

"""
    DeepSeekEndpoint <: OpenAIWireEndpoint
    DeepSeekEndpoint(; api_key=nothing)

Pre-configured endpoint for [DeepSeek](https://deepseek.com) API. Supports chat completions,
tool calling, FIM completion, and prefix completion.

`api_key::Union{String,Nothing}`: a given key is sent as is; `nothing` (the default) reads
`DEEPSEEK_API_KEY` each time a request is built, so building the endpoint needs no key and a
missing variable fails the request, naming it.

FIM and prefix completion use the beta base URL (`https://api.deepseek.com/beta`).
"""
struct DeepSeekEndpoint <: OpenAIWireEndpoint
    api_key::Union{String,Nothing}
end
DeepSeekEndpoint(; api_key::Union{String,Nothing}=nothing) = DeepSeekEndpoint(api_key)

# Render a stored API key as a short, non-reversible marker: a few leading
# characters (only when the key is long enough that those aren't the whole
# secret) followed by a fixed redaction tag. The tag is constant, so the key's
# length is never revealed, and the full key is never emitted. Endpoints keep
# their key in a struct field, so their `show` must redact it — otherwise the
# key surfaces wherever an endpoint is printed, including nested inside a Chat or
# a result value (Julia's default `show` recurses into fields via `show`).
function _redact_api_key(key::AbstractString)
    n = length(key)
    n == 0 && return ""                       # empty (e.g. local no-auth) — nothing to hide
    n > 6 ? string(first(key, 4), "…[redacted]") : "…[redacted]"
end

function Base.show(io::IO, e::GenericOpenAIEndpoint)
    print(io, "GenericOpenAIEndpoint(")
    show(io, e.base_url)
    print(io, ", ")
    show(io, _redact_api_key(e.api_key))
    isempty(e.key_env) || (print(io, ", "); show(io, e.key_env))
    print(io, ")")
end

# An endpoint that reads its key per request shows as the call that builds it, the variable named
# in a comment; a given key is redacted.
function Base.show(io::IO, e::DeepSeekEndpoint)
    key = e.api_key
    print(io, "DeepSeekEndpoint(")
    isnothing(key) ? print(io, "#= ", DEEPSEEK_API_KEY, " =#") : show(io, _redact_api_key(key))
    print(io, ")")
end


# Coerce the `tools` keyword to the stored `Vector{Tool}`. This fallback is
# the identity — a `Vector{Tool}` or `nothing` passes through unchanged. The
# method that unwraps a `Vector{<:CallableTool}` into its inner `Tool`s lives
# in tool_loop.jl, where `CallableTool` is defined (that file is `include`d
# after this one). Conversion happens only here at construction; the field type
# stays `Union{Vector{Tool},Nothing}`.
_chat_tools(tools) = tools

# The cache retentions providers document: Anthropic "5m" and "1h", OpenAI "30m" and "12h".
const _PROMPT_CACHE_TTLS = ("5m", "1h", "30m", "12h")

"""
    PromptCacheOptions(; mode=nothing, ttl=nothing, prewarm=nothing, comparison_response_id=nothing)

Prompt-cache controls (OpenAI: GPT-5.6 and later; Anthropic). `mode` is `"implicit"` or
`"explicit"`; `ttl`, how long a cached prefix is kept, is one of `"5m"` or `"1h"`
(Anthropic's lifetimes), `"30m"` or `"12h"` (OpenAI's) — any other value throws
`ArgumentError`. Each native endpoint takes its own subset and throws `ArgumentError`
naming it, before any request:

- `OPENAIServiceEndpoint`: Chat Completions takes `"30m"` (the API answers HTTP 400 for
  any other ttl); the Responses API ([`Respond`](@ref)) and OpenAI Decisions take `"30m"`
  and `"12h"`.
- `ANTHROPICServiceEndpoint`: a top-level `cache_control` (automatic caching) with ttl
  `"5m"` or `"1h"` and mode `"implicit"` or unset; `"explicit"`, `prewarm=true` and
  `comparison_response_id` throw.
- Native Gemini (Chat and Interactions) and [`OllamaEndpoint`](@ref) refuse
  `prompt_cache_options`.
- Azure, the Gemini OpenAI-compatible endpoint, DeepSeek and other OpenAI-compatible
  endpoints send the object unchanged: the server's answer is the check.

Explicit mode caches only prefixes marked with `prompt_cache_breakpoint` in input content
blocks (see [`input_text`](@ref) with `cache_breakpoint=true`).

Responses API only:
- `prewarm=true` prepares the prompt cache without generating output; send the
  real request afterwards with the same prefix and `prewarm` unset.
- `comparison_response_id` requests prompt-cache diagnostics against an earlier
  response; they come back in `r.response.raw["prompt_cache_diagnostics"]`.

Chat Completions accepts only `mode` and `ttl`: a [`Chat`](@ref) for
`OPENAIServiceEndpoint` with `prewarm` or `comparison_response_id` set throws
`ArgumentError` when encoded. [`Message`](@ref) content is a plain string, so a `Chat`
cannot mark cache breakpoints: with `mode="explicit"` its request has none, and per the
Chat Completions reference such a request does not use prompt caching. Unset fields are
omitted from the request. See
[OpenAI prompt caching](https://developers.openai.com/api/docs/guides/prompt-caching).
"""
@kwdef struct PromptCacheOptions
    mode::Union{String,Nothing} = nothing
    ttl::Union{String,Nothing} = nothing
    prewarm::Union{Bool,Nothing} = nothing
    comparison_response_id::Union{String,Nothing} = nothing
    function PromptCacheOptions(mode, ttl, prewarm=nothing, comparison_response_id=nothing)
        isnothing(mode) || mode in ("implicit", "explicit") ||
            throw(ArgumentError("prompt cache mode must be implicit or explicit"))
        isnothing(ttl) || ttl in _PROMPT_CACHE_TTLS || throw(ArgumentError(
            "prompt cache ttl must be one of $(join(repr.(_PROMPT_CACHE_TTLS), ", ")) (got $(repr(ttl)))"))
        new(mode, ttl, prewarm, comparison_response_id)
    end
end

function JSON.lower(options::PromptCacheOptions)
    d = Dict{Symbol,Any}()
    for f in (:mode, :ttl, :prewarm, :comparison_response_id)
        v = getfield(options, f)
        isnothing(v) || (d[f] = v)
    end
    d
end

"""
    ModerationConfig(; model, input_mode=nothing, output_mode=nothing)

OpenAI moderated completions for [`Chat`](@ref) and [`Respond`](@ref): the required
moderation `model` (e.g. `"omni-moderation-latest"`) checks the request input and the
generated output. The optional `input_mode` / `output_mode` set the policy per side:
`"score"` or `"block"`. Serialized as
`{"model": …, "policy": {"input": {"mode": …}, "output": {"mode": …}}}` with unset
policy parts omitted. A [`respond`](@ref) result carries the outcome in
`r.response.raw["moderation"]` (`"input"` and `"output"` moderation results);
[`LLMSuccess`](@ref) keeps no raw body, so Chat results do not expose it. See the
`moderation` parameter of
[Create a model response](https://developers.openai.com/api/reference/resources/responses/methods/create).
"""
@kwdef struct ModerationConfig
    model::String
    input_mode::Union{String,Nothing} = nothing
    output_mode::Union{String,Nothing} = nothing
    function ModerationConfig(model, input_mode, output_mode)
        isnothing(input_mode) || input_mode in ("score", "block") ||
            throw(ArgumentError("moderation input_mode must be score or block"))
        isnothing(output_mode) || output_mode in ("score", "block") ||
            throw(ArgumentError("moderation output_mode must be score or block"))
        new(model, input_mode, output_mode)
    end
end

function JSON.lower(m::ModerationConfig)
    d = Dict{Symbol,Any}(:model => m.model)
    policy = Dict{Symbol,Any}()
    isnothing(m.input_mode) || (policy[:input] = Dict(:mode => m.input_mode))
    isnothing(m.output_mode) || (policy[:output] = Dict(:mode => m.output_mode))
    isempty(policy) || (d[:policy] = policy)
    d
end

"""
    chat = Chat()

Creates a new `Chat` object with default settings:
- `model` is set to `gpt-6-sol` (the OpenAI default)
- `messages` is set to an empty `Vector{Message}`
- `history` is set to `true`

On the native OpenAI endpoint, `max_tokens` is sent as `max_completion_tokens` (setting
both throws `ArgumentError`), `user` throws (use `safety_identifier` or
`prompt_cache_key`), and `service_tier` must be one of `"auto"`, `"default"`, `"flex"`,
`"scale"`, `"priority"` or `"fast"`. Native Gemini and `GEMINIOpenAIServiceEndpoint` take
`service_tier` `"standard"`, `"flex"` or `"priority"`, and native Gemini sends `seed` (an
int32) and `store`; `ANTHROPICServiceEndpoint` takes `"auto"`, `"standard_only"` or
`"fast"` (fast mode). Each refuses another value with `ArgumentError` before any request.

OpenAI options, omitted from the request when unset:
- `prompt_cache_options::Union{PromptCacheOptions,Nothing}`: prompt-cache `mode` and
  `ttl` ([`PromptCacheOptions`](@ref)); supported for gpt-5.6 and later (gpt-5.4-mini
  answered HTTP 400 "prompt_cache_options is not supported on this model" on
  2026-09-22). A `Chat` sets no cache breakpoints (the API's `prompt_cache_breakpoint` on
  a message content part), so `mode="explicit"` means the request does not use prompt
  caching.
- `moderation::Union{ModerationConfig,Nothing}`: moderated completions
  ([`ModerationConfig`](@ref)).
- `stream_options::Union{AbstractDict,Nothing}`: streaming options; when unset, an
  `OPENAIServiceEndpoint` or `DeepSeekEndpoint` stream requests
  `{"include_usage": true}`, so the streamed result carries token usage.

The constructor validates ranges and throws `ArgumentError` on a violation; among
them, `n` must be `1` when set — a result carries a single choice — `top_logprobs`
must be in [0, 20], every `logit_bias` value (any `Real`) in [-100, 100], and
`reasoning_effort` one of `"none"`, `"minimal"`, `"low"`, `"medium"`, `"high"`,
`"xhigh"`, `"max"`. An empty `tools` vector is stored as `nothing`.
"""
@kwdef struct Chat
    service::ServiceEndpointSpec = OPENAIServiceEndpoint
    model::String = ""
    messages::Conversation = Message[]
    history::Bool = true
    tools::Union{Vector{Tool},Nothing} = nothing
    tool_choice::Union{String,GPTToolChoice,Nothing} = nothing # "auto" | "none" |
    parallel_tool_calls::Union{Bool,Nothing} = false
    temperature::Union{Float64,Nothing} = nothing # 0.0 - 2.0 - mutual exclusive with top_p
    top_p::Union{Float64,Nothing} = nothing # 0.0 - 1.0 - mutual exclusive with temperature
    n::Union{Int64,Nothing} = nothing # must be 1: results carry a single choice
    stream::Union{Bool,Nothing} = nothing
    stop::Union{Vector{String},String,Nothing} = nothing # max 4 sequences
    max_tokens::Union{Int64,Nothing} = nothing
    max_completion_tokens::Union{Int64,Nothing} = nothing # OpenAI's current name for max_tokens; other wires' encoders take it over max_tokens
    presence_penalty::Union{Float64,Nothing} = nothing # -2.0 - 2.0
    response_format::Union{ResponseFormat,Nothing} = nothing
    frequency_penalty::Union{Float64,Nothing} = nothing # -2.0 - 2.0
    logit_bias::Union{AbstractDict{String,<:Real},Nothing} = nothing # token id → bias in [-100, 100]
    user::Union{String,Nothing} = nothing
    seed::Union{Int64,Nothing} = nothing
    reasoning_effort::Union{String,Nothing} = nothing      # model-dependent reasoning effort
    stream_options::Union{AbstractDict,Nothing} = nothing  # e.g. {"include_usage": true}; see JSON.lower
    verbosity::Union{String,Nothing} = nothing             # low|medium|high
    store::Union{Bool,Nothing} = nothing
    metadata::Union{AbstractDict,Nothing} = nothing
    service_tier::Union{String,Nothing} = nothing          # per provider: see the docstring
    logprobs::Union{Bool,Nothing} = nothing
    top_logprobs::Union{Int64,Nothing} = nothing
    prediction::Union{AbstractDict,Nothing} = nothing
    modalities::Union{Vector{String},Nothing} = nothing
    audio::Union{AbstractDict,Nothing} = nothing
    web_search_options::Union{AbstractDict,Nothing} = nothing
    prompt_cache_key::Union{String,Nothing} = nothing
    safety_identifier::Union{String,Nothing} = nothing     # replaces deprecated `user`
    _cumulative_cost::Ref{Float64} = Ref(0.0)
    prompt_cache_options::Union{PromptCacheOptions,Nothing} = nothing  # gpt-5.6 and later
    moderation::Union{ModerationConfig,Nothing} = nothing
    function Chat(
        service,
        model,
        messages,
        history,
        tools,
        tool_choice,
        parallel_tool_calls,
        temperature,
        top_p,
        n,
        stream,
        stop,
        max_tokens,
        max_completion_tokens,
        presence_penalty,
        response_format,
        frequency_penalty,
        logit_bias,
        user,
        seed,
        reasoning_effort,
        stream_options,
        verbosity,
        store,
        metadata,
        service_tier,
        logprobs,
        top_logprobs,
        prediction,
        modalities,
        audio,
        web_search_options,
        prompt_cache_key,
        safety_identifier,
        _cumulative_cost,
        prompt_cache_options=nothing,
        moderation=nothing
    )
        model = _resolve_model(service, model)
        tools = _chat_tools(tools)  # accept a CallableTool vector, stored as Tools
        isnothing(tools) || !isempty(tools) || (tools = nothing)   # no tools is no tool list
        !isnothing(temperature) && !isnothing(top_p) && throw(ArgumentError("temperature and top_p are mutually exclusive"))
        !isnothing(temperature) && !(0.0 <= temperature <= 2.0) && throw(ArgumentError("temperature must be in [0.0, 2.0]"))
        !isnothing(top_p) && !(0.0 <= top_p <= 1.0) && throw(ArgumentError("top_p must be in [0.0, 1.0]"))
        # A result carries one choice: extra choices were dropped (non-streaming) or
        # merged into one garbled text (streaming).
        !isnothing(n) && n != 1 && throw(ArgumentError("n must be 1 (got $n): results carry a single choice"))
        !isnothing(max_tokens) && max_tokens < 1 && throw(ArgumentError("max_tokens must be >= 1"))
        !isnothing(max_completion_tokens) && max_completion_tokens < 1 && throw(ArgumentError("max_completion_tokens must be >= 1"))
        !isnothing(presence_penalty) && !(-2.0 <= presence_penalty <= 2.0) && throw(ArgumentError("presence_penalty must be in [-2.0, 2.0]"))
        !isnothing(frequency_penalty) && !(-2.0 <= frequency_penalty <= 2.0) && throw(ArgumentError("frequency_penalty must be in [-2.0, 2.0]"))
        !isnothing(top_logprobs) && !(0 <= top_logprobs <= 20) && throw(ArgumentError("top_logprobs must be in [0, 20]"))
        isnothing(logit_bias) || all(v -> -100 <= v <= 100, values(logit_bias)) ||
            throw(ArgumentError("logit_bias values must be in [-100, 100]"))
        isnothing(reasoning_effort) || reasoning_effort in _OPENAI_REASONING_EFFORTS || throw(ArgumentError(
            "reasoning_effort must be one of $(join(_OPENAI_REASONING_EFFORTS, ", ")) (got $(repr(reasoning_effort)))"))
        return new(
            service,
            model,
            messages,
            history,
            tools,
            tool_choice,
            !isnothing(tools) ? parallel_tool_calls : nothing,
            temperature,
            top_p,
            n,
            stream,
            stop,
            max_tokens,
            max_completion_tokens,
            presence_penalty,
            response_format,
            frequency_penalty,
            logit_bias,
            user,
            seed,
            reasoning_effort,
            stream_options,
            verbosity,
            store,
            metadata,
            service_tier,
            logprobs,
            top_logprobs,
            prediction,
            modalities,
            audio,
            web_search_options,
            prompt_cache_key,
            safety_identifier,
            _cumulative_cost,
            prompt_cache_options,
            moderation
        )
    end
end

function JSON.lower(chat::Chat)
    d = Dict{Symbol,Any}(:model => chat.model, :messages => _wire_messages(chat.service, chat.messages))
    for f in (:tools, :tool_choice, :parallel_tool_calls, :temperature, :top_p,
        :n, :stream, :stop, :max_tokens, :max_completion_tokens, :presence_penalty, :response_format,
        :frequency_penalty, :logit_bias, :user, :seed,
        :reasoning_effort, :stream_options, :verbosity, :store, :metadata, :service_tier,
        :logprobs, :top_logprobs, :prediction, :modalities, :audio, :web_search_options,
        :prompt_cache_key, :safety_identifier, :prompt_cache_options, :moderation)
        v = getfield(chat, f)
        !isnothing(v) && (d[f] = v)
    end
    # OpenAI, DeepSeek and Gemini's OpenAI-compatible endpoint report a stream's token usage
    # when asked through `stream_options.include_usage` (each documents it; Google's
    # compatibility page streams with `stream_options={'include_usage': True}`); ask for the
    # caller who set no stream_options, so a streamed turn is costed like a non-streamed one.
    # Not for other OpenAI-compatible servers, which may reject the field.
    chat.stream === true && isnothing(chat.stream_options) && _asks_stream_usage(chat.service) &&
        (d[:stream_options] = Dict(:include_usage => true))
    return d
end

_asks_stream_usage(::Type{OPENAIServiceEndpoint}) = true
_asks_stream_usage(::DeepSeekEndpoint) = true
_asks_stream_usage(::Type{GEMINIOpenAIServiceEndpoint}) = true
_asks_stream_usage(_) = false

# The conversation as an endpoint's Chat Completions wire carries it. OpenAI keeps the
# audio of an audio reply and takes it back by id, `"audio": {"id": …}` beside the turn's
# content (the assistant message of the Chat Completions reference); every other endpoint
# gets the content alone, the transcript unless the turn holds other text. A conversation
# without an audio reply goes out as it is.
_wire_messages(_, messages::Conversation) = messages
_wire_messages(::Type{OPENAIServiceEndpoint}, messages::Conversation) =
    any(m -> !isnothing(m.audio_output), messages) ? Any[_audio_turn(m) for m in messages] : messages

function _audio_turn(m::Message)
    audio = m.audio_output
    isnothing(audio) && return m
    d = JSON.lower(m)
    d[:audio] = Dict(:id => audio.id)
    d
end

Base.length(chat::Chat) = length(chat.messages)
Base.isempty(chat::Chat) = isempty(chat.messages)

"""
    LLMRequestResponse

Abstract supertype for all API call results. Pattern-match on subtypes to handle outcomes:

- [`LLMSuccess`](@ref) — successful response
- [`LLMFailure`](@ref) — HTTP-level failure (non-200 status)
- [`LLMCallError`](@ref) — exception during the call (network error, etc.)
- [`ResponseSuccess`](@ref) — successful Responses API result
- [`ResponseFailure`](@ref) — Responses API HTTP failure
- [`ResponseCallError`](@ref) — Responses API exception
"""
abstract type LLMRequestResponse end

"""
    TokenUsage(; prompt_tokens=0, completion_tokens=0, total_tokens=0, cached_tokens=0, reasoning_tokens=0,
               cache_write_tokens=0, cache_write_extended_tokens=0, service_tier=nothing, inference_geo=nothing,
               prompt_audio_tokens=0, completion_audio_tokens=0)
    TokenUsage(prompt_tokens, completion_tokens, total_tokens, cached_tokens, reasoning_tokens)

Token usage statistics returned by the API, in one shape for every provider.

`prompt_tokens` counts every input token, cache reads and cache writes included. Of them,
`cached_tokens` were read from the prompt cache and `cache_write_tokens` written to it;
`cache_write_extended_tokens` is the part of `cache_write_tokens` written at the extended
retention (OpenAI `"12h"`, Anthropic `"1h"`). `reasoning_tokens` is the part of
`completion_tokens` spent on hidden reasoning. `prompt_audio_tokens` is the part of
`prompt_tokens` that is audio input and `completion_audio_tokens` the part of
`completion_tokens` that is audio output, as OpenAI Chat Completions reports them for a
model that hears or speaks; `prompt_audio_tokens` also counts Gemini's audio input on the
native endpoint and Gemini Interactions. The counts are 0 when the provider omits them, as
`GEMINIOpenAIServiceEndpoint` does: its audio input is in `prompt_tokens` alone.
`service_tier` is the tier the provider reports
billing the call at (OpenAI's `service_tier`, for example) and `inference_geo` the inference
region it reports; each is `nothing` when the provider reports none. One exception: on
`GEMINIOpenAIServiceEndpoint`, whose replies name no billed tier, a Chat result's
`service_tier` is the tier the request asked for, so `"flex"` is priced at the Flex rates and
`"priority"` at the Priority rates — an upper bound, since Google bills a downgraded Priority
request at Standard and this endpoint does not report the downgrade.

[`estimated_cost`](@ref) bills `cached_tokens` at the discounted cached-input rate and the
rest at the standard rates, unless the provider's own pricing applies (see
[`DEFAULT_PRICING`](@ref)): OpenAI bills the audio tokens at the model's audio-token rates and
Gemini audio input at the rate its pricing page states for the model; any other audio leaves
the estimate at `0.0`. Audio a provider does not count apart (`GEMINIOpenAIServiceEndpoint`'s)
is priced with the text in `prompt_tokens`.
Positionally the fields go in the order listed, the five counts first; the fields left out
take their defaults.
"""
@kwdef struct TokenUsage
    prompt_tokens::Int = 0
    completion_tokens::Int = 0
    total_tokens::Int = 0
    cached_tokens::Int = 0
    reasoning_tokens::Int = 0
    cache_write_tokens::Int = 0
    cache_write_extended_tokens::Int = 0
    service_tier::Union{String,Nothing} = nothing
    inference_geo::Union{String,Nothing} = nothing
    prompt_audio_tokens::Int = 0
    completion_audio_tokens::Int = 0
    function TokenUsage(prompt_tokens, completion_tokens, total_tokens, cached_tokens, reasoning_tokens,
                        cache_write_tokens=0, cache_write_extended_tokens=0, service_tier=nothing,
                        inference_geo=nothing, prompt_audio_tokens=0, completion_audio_tokens=0)
        new(prompt_tokens, completion_tokens, total_tokens, cached_tokens, reasoning_tokens,
            cache_write_tokens, cache_write_extended_tokens, service_tier, inference_geo,
            prompt_audio_tokens, completion_audio_tokens)
    end
end

"""A per-token pricing row: `input` / `cached_input` / `output` USD per token. A provider's
surcharges (on cache writes, a service tier, long prompts) are not stored in the row: its
[`_billed_cost`](@ref) method applies them to these rates."""
const PriceRow = @NamedTuple{input::Float64, cached_input::Float64, output::Float64}

"""Build a [`PriceRow`](@ref) from per-1M-token USD figures (input, cached-input, output)."""
_price(i, c, o) = (input = i / 1_000_000, cached_input = c / 1_000_000, output = o / 1_000_000)

"""
    _billed_cost(service, model::String, u::TokenUsage, rates::PriceRow) -> Float64

The USD a provider bills for the usage `u` of `model`, whose price row is `rates`;
[`estimated_cost`](@ref) calls it once it has found the row. `service` is the endpoint the
result came from: a marker type such as `OPENAIServiceEndpoint`, an endpoint instance, or
`nothing` when the result records none.

The default method bills the row's standard rates: `(prompt_tokens - cached_tokens -
cache_write_tokens) * input + cached_tokens * cached_input + cache_write_tokens * input +
completion_tokens * output`. It leaves a usage that counts audio tokens (`prompt_audio_tokens`,
`completion_audio_tokens`) unpriced — `0.0`, with one warning per model and endpoint — since a
row has no audio-token rates. A provider whose bill differs adds a method for its endpoint
type in its own file, beside the pricing page it follows: a premium on cache writes
(`cache_write_tokens`, `cache_write_extended_tokens`), a multiplier for the reported
`service_tier` or `inference_geo`, a long-context tier. Such a method returns the whole
cost of the call, not a surcharge.
"""
function _billed_cost end

"""
    LLMSuccess(; message, self, usage=nothing, sse_dropped=0, model=nothing)

Successful Chat Completions API response.

# Fields
- `message::Message`: The assistant's reply message.
- `self::Chat`: The updated [`Chat`](@ref) object (with the new message appended if `history=true`).
- `usage::Union{TokenUsage, Nothing}`: Token usage statistics from the API.
- `sse_dropped::Int`: Undecodable SSE `data:` payloads dropped while assembling
  this streamed turn — `0` for a non-streamed call, and for a clean stream.
  Non-zero means the turn was built from an incomplete wire.
- `model::Union{String,Nothing}`: The model id the provider reports having served the turn
  (the reply's `model`; it differs from `self.model` when the provider routes the request to
  another model, or names the dated snapshot behind an alias, such as an id ending
  `-YYYY-MM-DD`, which is priced at the alias's row); `nothing` where the endpoint does not
  report it, and from an
  [`OllamaEndpoint`](@ref), whose cloud models are priced by the name requested.
  [`estimated_cost`](@ref) prices the call at this model when it is set, else at `self.model`.
"""
@kwdef struct LLMSuccess <: LLMRequestResponse
    message::Message
    self::Chat
    usage::Union{TokenUsage, Nothing} = nothing
    sse_dropped::Int = 0
    model::Union{String,Nothing} = nothing
end

# The provider's id for the request: `x-request-id` (OpenAI and most compatible
# servers), else `request-id` (Anthropic).
function _get_request_id(resp::HTTP.Response)::Union{Nothing,String}
    for name in ("x-request-id", "request-id")
        val = HTTP.header(resp, name, "")
        isempty(val) || return String(val)
    end
    nothing
end
_get_request_id(::Nothing) = nothing
function _get_request_id(e::Any)
    if hasproperty(e, :response) && e.response isa HTTP.Response
        return _get_request_id(e.response)
    elseif hasproperty(e, :message) && e.message isa HTTP.Response
        return _get_request_id(e.message)
    end
    return nothing
end

"""
    LLMFailure(; response, status, self, request_id=nothing, sse_dropped=0)

HTTP-level failure from the Chat Completions API. The server returned a non-200 status.

# Fields
- `response::String`: The raw response body.
- `status::Int`: The HTTP status code.
- `self::Chat`: The [`Chat`](@ref) object (unchanged).
- `request_id::Union{String, Nothing}`: The HTTP request ID from headers, if available.
- `sse_dropped::Int`: Undecodable SSE `data:` payloads dropped during a streamed
  attempt — `0` for a non-streamed call. On a truncated stream (HTTP 200, no
  terminal event) this is often the reason no message could be built.
"""
@kwdef struct LLMFailure <: LLMRequestResponse
    response::String
    status::Int
    self::Chat
    request_id::Union{String, Nothing} = nothing
    sse_dropped::Int = 0
end

"""
    LLMCallError(; error, status=nothing, self, request_id=nothing, cause=nothing)

Exception-level error during a Chat Completions API call (network failure, JSON parse error, timeout, etc.).

# Fields
- `error::String`: The stringified exception.
- `status::Union{Int,Nothing}`: HTTP status if available (`nothing` for timeouts — no fabricated statuses).
- `self::Chat`: The [`Chat`](@ref) object (unchanged).
- `request_id::Union{String, Nothing}`: The HTTP request ID from headers, if available.
- `cause::Union{Nothing,Exception}`: The underlying typed exception when one exists (e.g. a `UniLMTimeout` carrying phase/elapsed/limit).
"""
@kwdef struct LLMCallError <: LLMRequestResponse
    error::String
    status::Union{Int,Nothing} = nothing
    self::Chat
    request_id::Union{String, Nothing} = nothing
    cause::Union{Nothing,Exception} = nothing
end

# Every call-error result keeps the raw exception in `cause` — dispatching on it is
# the point of the field. Julia's default `show` recurses into it, though, and a
# transport wrapper can render as a full request dump, headers included, so printing
# the result would undo the redaction its `error` string already went through. Name the
# cause by TYPE instead: the object stays untouched and still reachable, it just
# stops printing its payload. Shared by the Chat/Embeddings/Responses/FIM results.
function _show_call_error(io::IO, name::AbstractString, err::AbstractString,
                          status, request_id, cause)
    print(io, name, "(error=")
    show(io, err)
    print(io, ", status=", repr(status))
    isnothing(request_id) || print(io, ", request_id=", repr(request_id))
    print(io, ", cause=")
    isnothing(cause) ? print(io, "nothing") : print(io, typeof(cause))
    print(io, ")")
end

Base.show(io::IO, r::LLMCallError) =
    _show_call_error(io, "LLMCallError", r.error, r.status, r.request_id, r.cause)

# ─── Result consumption ──────────────────────────────────────────────────────

"""
    issuccess(r::LLMRequestResponse) -> Bool

`true` when `r` is a success result (any `*Success` type), `false` for every
failure or call-error result. The generic method returns `false`; each concrete
`*Success` result type gets its own `true` method (registered once every result
type across the APIs is defined — see the bottom of `UniLM.jl`).

```julia
result = chatrequest!(chat)
issuccess(result) ? println(text(result)) : @warn "call did not succeed"
```
"""
issuccess(::LLMRequestResponse) = false

"""
    isfailure(r::LLMRequestResponse) -> Bool

Negation of [`issuccess`](@ref): `true` for any failure or call-error result.
"""
isfailure(r::LLMRequestResponse) = !issuccess(r)

"""
    text(r::LLMSuccess) -> Union{String,Nothing}

The assistant reply text — `r.message.content`. Can be `nothing` when the reply
carries only tool calls (no text). On a [`LLMFailure`](@ref) or [`LLMCallError`](@ref),
`text` throws an [`LLMResultError`](@ref); guard with [`issuccess`](@ref) /
[`isfailure`](@ref), or pattern-match the result type first.
"""
text(r::LLMSuccess) = r.message.content

"""
    LLMResultError <: Exception

Thrown by the result accessors — [`text`](@ref), [`output_text`](@ref),
[`embedding_vectors`](@ref), [`image_data`](@ref) and [`fim_text`](@ref) — when they
are called on a non-success result. Carries the offending `result`. `showerror`
prints only the status and a short (≤200-char) excerpt of the response body or error
message — never the conversation, the service endpoint, or the API key.
"""
struct LLMResultError <: Exception
    result::LLMRequestResponse
end

text(r::Union{LLMFailure,LLMCallError}) = throw(LLMResultError(r))

"""
    reasoning_text(m::Message) -> Union{String,Nothing}
    reasoning_text(r::LLMSuccess) -> Union{String,Nothing}

The reasoning ("thinking") text a provider returned with an assistant turn, read
from its [`ProviderContent`](@ref): Ollama's `thinking`, DeepSeek's
`reasoning_content`, Anthropic `thinking` blocks (separate blocks joined by a blank
line; redacted blocks carry no text) and Gemini thought parts. `nothing` when the
turn carries none — the provider returned no reasoning, or the model did not think.
Like [`text`](@ref), it throws an [`LLMResultError`](@ref) on a failure result.

```julia
chat = Chat(service=OllamaEndpoint(), model="gemma4:e4b",
            reasoning_effort="high")
push!(chat, Message(Val(:system), "Answer briefly."))
push!(chat, Message(Val(:user), "Is 391 prime?"))
r = chatrequest!(chat)
reasoning_text(r)   # "391 = 17 × 23, so …"
```
"""
function reasoning_text(m::Message)::Union{String,Nothing}
    pc = m.provider_content
    pc isa ProviderContent || return nothing
    parts = String[]
    for b in pc.blocks
        t = _reasoning_in(pc.provider, b)
        isnothing(t) || push!(parts, t)
    end
    isempty(parts) ? nothing : join(parts, pc.provider === :anthropic ? "\n\n" : "")
end
reasoning_text(r::LLMSuccess) = reasoning_text(r.message)
reasoning_text(r::Union{LLMFailure,LLMCallError}) = throw(LLMResultError(r))

# The reasoning text in one captured block, in the provider's own shape.
function _reasoning_in(provider::Symbol, b)::Union{String,Nothing}
    b isa AbstractDict || return nothing
    t = provider === :ollama    ? get(b, "thinking", nothing) :
        provider === :deepseek  ? get(b, "reasoning_content", nothing) :
        provider === :anthropic ? (get(b, "type", nothing) == "thinking" ? get(b, "thinking", nothing) : nothing) :
        provider === :gemini    ? (get(b, "thought", false) === true ? get(b, "text", nothing) : nothing) :
        nothing
    t isa AbstractString && !isempty(t) ? String(t) : nothing
end

# The status and the body/message text of a non-success result: every `*Failure`
# carries the response body in `response`, every `*CallError` its message in `error`.
_result_status(r::LLMRequestResponse) = hasproperty(r, :status) ? getproperty(r, :status) : nothing
function _result_text(r::LLMRequestResponse)::String
    for f in (:response, :error)
        hasproperty(r, f) && (v = getproperty(r, f)) isa AbstractString && return String(v)
    end
    ""
end

function Base.showerror(io::IO, e::LLMResultError)
    status = _result_status(e.result)
    print(io, "LLMResultError: no result data — the request did not succeed (status ",
          isnothing(status) ? "unknown" : status, "). ")
    body = _result_text(e.result)
    print(io, "Response: ", length(body) > 200 ? string(first(body, 200), "…") : body)
end

"""
    issendvalid(chat::Chat)::Bool

Check if the conversation is valid for sending to the API.

Returns `true` only when the conversation has at least two messages, the first
is a system message, the LAST is a user message, and no two adjacent messages
share a role. Note the last two clauses are stricter than
[`push!`](@ref Base.push!(::Chat, ::Message)), which permits consecutive `tool`
messages — this check has no such exemption, and a conversation ending in a tool
result is `false` here.

This is a heuristic, not a proof: it cannot catch every malformed shape (a
second system message in the middle passes the adjacency test). It never
throws — a `false` is a verdict, not an error.
"""
function issendvalid(chat::Chat)::Bool
    length(chat) > 1 &&
        chat.messages[begin].role == RoleSystem &&
        chat.messages[end].role == RoleUser &&
        all(chat.messages[i].role != chat.messages[i+1].role for i in 1:length(chat)-1)
end

"""
    push!(chat::Chat, msg::Message)

Add a message to the conversation, refusing any mutation that would make it invalid.

Throws [`InvalidConversationError`](@ref) when the mutation would produce an
invalid conversation: a non-system message pushed onto an empty conversation
(it must start with a system message), or a message whose role repeats the last
message's role (consecutive tool-result messages are the one exception).
`chat[i] = msg` and direct edits to `chat.messages` are not validated.

A system message pushed once the conversation has started changes the instructions
from that point on. Where it reaches the provider depends on the encoder:
- the OpenAI wire (OpenAI, Azure, DeepSeek and other OpenAI-compatible endpoints) and
  [`OllamaEndpoint`](@ref) send it in place;
- native Gemini joins it into `systemInstruction`;
- native Anthropic sends it in place on the models that accept
  [mid-conversation system messages](https://platform.claude.com/docs/en/build-with-claude/mid-conversation-system-messages),
  where the Messages API allows one (right after a user turn, and last or before an
  assistant turn), and elsewhere joins it into the top-level `system` prompt. Claude
  Fable 5.1, Opus 5.5, Sonnet 5.5 and Haiku 5.5 refuse an earlier turn's thinking blocks
  once the `system` prompt has changed, so on them — and on a Claude model UniLM has no
  rules for — one pushed after an assistant turn that cannot stay in place is sent right
  after the next user turn instead.

On native Anthropic a system message pushed after an assistant turn needs a user message
after it: until one follows, the request throws `ArgumentError`.
"""
function Base.push!(chat::Chat, msg::Message)
    if msg.role != RoleSystem && isempty(chat)
        throw(InvalidConversationError("conversation must start with a system message; got :$(msg.role)"))
    elseif !isempty(chat) && chat.messages[end].role == msg.role && msg.role != RoleTool
        throw(InvalidConversationError("conversation cannot contain consecutive messages from the same role; got two :$(msg.role) in a row"))
    end
    push!(chat.messages, msg)
    return chat
end

"""
    pop!(chat::Chat)

Remove the last message from the conversation.

Throws [`InvalidConversationError`](@ref) when the conversation is empty.
"""
function Base.pop!(chat::Chat)
    isempty(chat) && throw(InvalidConversationError("cannot pop! from an empty conversation"))
    pop!(chat.messages)
    return chat
end

"""
    last(chat::Chat)

    Get the last message in the conversation.
"""
Base.last(chat::Chat) = last(chat.messages)

"""
    update!(chat::Chat, msg::Message) -> Chat

Append `msg` to the conversation when `chat.history` is on; with `history=false` the
chat is left unchanged, as documented for that setting (logged at debug level only).
"""
function update!(chat::Chat, msg::Message)
    chat.history && push!(chat, msg)
    !chat.history && @debug "Cannot update chat with your message: chat history is disabled."
    return chat
end

"""
    Base.getindex(chat::Chat, i::Int)

    Get the message at index `i` in the conversation.
"""
Base.getindex(chat::Chat, i::Int) = chat.messages[i]

"""
    Base.setindex!(chat::Chat, msg::Message, i::Int)

Set the message at index `i` in the conversation. Unlike `push!`, this is not
validated: the caller keeps the conversation well-formed.
"""
Base.setindex!(chat::Chat, msg::Message, i::Int) = (chat.messages[i] = msg)

"""
    Base.lastindex(chat::Chat)

    Get the last index in the conversation.
"""
Base.lastindex(chat::Chat) = lastindex(chat.messages)

"""
    Base.firstindex(chat::Chat)

    Get the first index in the conversation.
"""
Base.firstindex(chat::Chat) = firstindex(chat.messages)


# _EMBEDDINGS_

const GPTTextEmbedding3Small = Model("text-embedding-3-small")

"""
    EmbeddingInput(text; title=nothing)
    EmbeddingInput(image::ImageAttachment)
    EmbeddingInput(audio::AudioAttachment)
    EmbeddingInput(text, media...)
    EmbeddingInput(; text=nothing, title=nothing, image=nothing, audio=nothing)

One item for [`Embeddings`](@ref), embedded as one vector: a text, an image, a sound clip, or a text
with an image or a sound clip — one of them per item on an [`OllamaEndpoint`](@ref), which refuses an
item holding both before any request; a service that embeds text only refuses media. `title` names the
document a text comes from; it is used by the `:document` task of models with task prompts
(EmbeddingGemma, `gemini-embedding-2`) and refused with any other task or model. Images and audio go
only to an `OllamaEndpoint` (Ollama's `/api/embed` takes them for models that see or hear, such as
`embeddinggemma-2:740m-q8_0`, which embedded a text with its picture or clip as one item on Ollama
0.40.6). Ollama passes an item's image and clip to the model in a random order, which is why an item
holding both is refused (on Ollama 0.40.6, ten requests of one such item to `embeddinggemma-2:740m-q8_0`
gave two different vectors): send the image and the audio as separate items.
Two items are `==` when their texts, titles and media are. An empty text beside an image or clip is
no text: the item holds the media alone (`text === nothing`), sent with no task prompt. An empty or
whitespace-only title is no title (`title === nothing`): with `task = :document` the text is sent as
`"title: none | text: …"`, the model card's prompt for a document without one.

Throws `ArgumentError` for an item with nothing to embed (no image or clip, and no text or an empty
one), a `title` without a non-empty text, more than one image or clip, an image with a detail level,
or an attachment of any other kind, such as a [`DocumentAttachment`](@ref): no embedding endpoint UniLM
sends to takes a level or a document (OpenAI's and its compatibles take text, Ollama's `/api/embed` an
image's or a clip's bytes alone).

```julia
EmbeddingInput("Refunds take 5 days."; title = "Refund policy")
EmbeddingInput("a red circle", ImageAttachment("red_circle.png"))
```
"""
struct EmbeddingInput
    text::Union{String,Nothing}
    title::Union{String,Nothing}
    image::Union{ImageAttachment,Nothing}
    audio::Union{AudioAttachment,Nothing}
    function EmbeddingInput(text::Union{AbstractString,Nothing}, title::Union{AbstractString,Nothing},
                            image::Union{ImageAttachment,Nothing}, audio::Union{AudioAttachment,Nothing})
        # An empty text is none: beside media the item is the media alone, sent with no bare task prompt.
        t = isnothing(text) || isempty(text) ? nothing : String(text)
        # A blank title is none too: the document is sent as the model card's untitled one ("title: none").
        h = isnothing(title) || isempty(strip(title)) ? nothing : String(title)
        isnothing(t) && isnothing(image) && isnothing(audio) && throw(ArgumentError(
            "an EmbeddingInput needs a non-empty text, an image or a sound clip to embed"))
        isnothing(h) || !isnothing(t) || throw(ArgumentError(
            "an EmbeddingInput's title names the document its text comes from: give it a non-empty text too"))
        isnothing(image) || isnothing(image.detail) || throw(ArgumentError(
            "an embedding reads an image's bytes alone, and no embedding endpoint takes a detail level: build " *
            "the ImageAttachment without `detail` (got $(repr(image.detail)))"))
        new(t, h, image, audio)
    end
end
EmbeddingInput(; text=nothing, title=nothing, image=nothing, audio=nothing) = EmbeddingInput(text, title, image, audio)
EmbeddingInput(text::AbstractString; title=nothing) = EmbeddingInput(text, title, nothing, nothing)
EmbeddingInput(image::ImageAttachment) = EmbeddingInput(nothing, nothing, image, nothing)
EmbeddingInput(audio::AudioAttachment) = EmbeddingInput(nothing, nothing, nothing, audio)
EmbeddingInput(a::Attachment) = EmbeddingInput("", a)   # another kind: refused by name below
# Any other value in a slot — a document as the image, a clip as the image — is refused by kind.
EmbeddingInput(text::Union{AbstractString,Nothing}, title::Union{AbstractString,Nothing}, image, audio) =
    throw(ArgumentError("an EmbeddingInput holds a text and a title (strings), an image (an ImageAttachment) and " *
        "a sound clip (an AudioAttachment), each optional; got text::$(typeof(text)), title::$(typeof(title)), " *
        "image::$(typeof(image)), audio::$(typeof(audio))"))
function EmbeddingInput(text::AbstractString, media::Attachment...)
    other = unique(typeof(m) for m in media if !(m isa Union{ImageAttachment,AudioAttachment}))
    isempty(other) || throw(ArgumentError(
        "an EmbeddingInput holds an image and/or a sound clip; got $(join(other, ", "))"))
    images = ImageAttachment[m for m in media if m isa ImageAttachment]
    clips = AudioAttachment[m for m in media if m isa AudioAttachment]
    length(images) <= 1 && length(clips) <= 1 || throw(ArgumentError(
        "an EmbeddingInput holds at most one image and one sound clip; embed the others as items of their own"))
    EmbeddingInput(text, nothing, isempty(images) ? nothing : only(images), isempty(clips) ? nothing : only(clips))
end

Base.show(io::IO, x::EmbeddingInput) = print(io, "EmbeddingInput(", join(filter(!isempty, [
    isnothing(x.text) ? "" : "text=" * repr(x.text), isnothing(x.title) ? "" : "title=" * repr(x.title),
    isnothing(x.image) ? "" : sprint(show, x.image), isnothing(x.audio) ? "" : sprint(show, x.audio)]), ", "), ")")
# Field-wise, so items holding equal media (compared by content) are equal.
Base.:(==)(a::EmbeddingInput, b::EmbeddingInput) = all(f -> getfield(a, f) == getfield(b, f), fieldnames(EmbeddingInput))
Base.hash(x::EmbeddingInput, h::UInt) =
    foldl((h, f) -> hash(getfield(x, f), h), fieldnames(EmbeddingInput); init=hash(EmbeddingInput, h))

# What a text is for, as the task prompts of a model that has them say it (EmbeddingGemma's model card;
# Gemini Embedding 2's guide gives the same prompts).
const _EMBEDDING_TASKS = (:search_query, :question_answering, :fact_checking, :code_retrieval, :document,
                          :classification, :clustering, :sentence_similarity, :none)
const _EMBEDDING_TASK_NAMES = Dict{Symbol,String}(:search_query => "search result",
    :question_answering => "question answering", :fact_checking => "fact checking",
    :code_retrieval => "code retrieval", :classification => "classification", :clustering => "clustering",
    :sentence_similarity => "sentence similarity")
const _EMBEDDINGGEMMA_DIMENSIONS = (768, 512, 256, 128)

# The task-prompt family of a model, from its name: EmbeddingGemma (v1 and v2, any tag or namespace),
# and gemini-embedding-2 by its exact id, whose guide recommends these prompts for text
# (https://ai.google.dev/gemini-api/docs/embeddings#task-types-embeddings-2) but needs none.
_embedding_prompt_family(model::AbstractString)::Union{Symbol,Nothing} =
    occursin("embeddinggemma", lowercase(model)) ? :embeddinggemma :
    model == "gemini-embedding-2" ? :gemini_embedding_2 : nothing

_embedding_titles(x::EmbeddingInput) = !isnothing(x.title)
_embedding_titles(::String) = false
_has_titles(input) = input isa AbstractVector ? any(_embedding_titles, input) : _embedding_titles(input)
_item_media(x::EmbeddingInput) = !isnothing(x.image) || !isnothing(x.audio)
_item_media(::String) = false
_has_media(emb) = emb.input isa AbstractVector ? any(_item_media, emb.input) : _item_media(emb.input)

function _check_embeddings(model::String, task::Union{Symbol,Nothing}, dimensions::Union{Int,Nothing}, input)::Nothing
    isnothing(task) || task in _EMBEDDING_TASKS || throw(ArgumentError(
        "embedding task must be one of $(join(repr.(_EMBEDDING_TASKS), ", ")) (got $(repr(task)))"))
    family = _embedding_prompt_family(model)
    if family === :embeddinggemma
        isnothing(task) && throw(ArgumentError(
            "$model expects a task prompt for what the text is for: pass task = " *
            join(repr.(filter(!=(:none), _EMBEDDING_TASKS)), ", ") *
            " — or task = :none to send the text as is (EmbeddingGemma then works with lower precision)"))
        isnothing(dimensions) || dimensions in _EMBEDDINGGEMMA_DIMENSIONS || throw(ArgumentError(
            "$model returns $(join(_EMBEDDINGGEMMA_DIMENSIONS, ", ")) dimensions (its trained Matryoshka sizes); " *
            "got dimensions = $dimensions"))
    end
    if isnothing(family)
        # No task sends a title here, so its error comes first: it names the only fix.
        _has_titles(input) && throw(ArgumentError(
            "$(repr(model)) has no task prompts, so a title has nowhere to go: drop it " *
            "(only task = :document of EmbeddingGemma and gemini-embedding-2 writes titles)"))
        isnothing(task) || task === :none || throw(ArgumentError(
            "UniLM knows the task prompts of EmbeddingGemma models and gemini-embedding-2 only, so " *
            "$(repr(model)) gets none: pass task = :none and write the model's own prefix into the text, if it has one"))
    else
        task !== :document && _has_titles(input) && throw(ArgumentError(
            "a title names the document a text comes from, which $model reads with task = :document only " *
            "(got task = $(repr(task))): pass task = :document, or drop the title"))
    end
    isnothing(dimensions) || dimensions > 0 || throw(ArgumentError("dimensions must be positive (got $dimensions)"))
    nothing
end

_embedding_item(x::AbstractString) = String(x)
_embedding_item(x::EmbeddingInput) = x
_embedding_item(x::Union{ImageAttachment,AudioAttachment}) = EmbeddingInput(x)
_embedding_item(x) = throw(ArgumentError(
    "an embedding input is a String, an EmbeddingInput, an ImageAttachment or an AudioAttachment (got $(typeof(x)))"))
# A vector of strings stays one; any other item makes every item an EmbeddingInput.
function _embedding_items(xs::AbstractVector)
    all(x -> x isa AbstractString, xs) && return String[x for x in xs]
    EmbeddingInput[(i = _embedding_item(x); i isa String ? EmbeddingInput(i) : i) for x in xs]
end

"""
    Embeddings(input::Union{AbstractString,EmbeddingInput,Attachment}; service=OPENAIServiceEndpoint,
               model="", dimensions=nothing, encoding_format=nothing, user=nothing, task=nothing)
    Embeddings(input::AbstractVector; service=OPENAIServiceEndpoint, model="", dimensions=nothing,
               encoding_format=nothing, user=nothing, task=nothing)

Create an embedding request for one or more texts. `model=""` resolves to the service's
default embedding model — OpenAI's `text-embedding-3-small` (1536 dimensions),
`gemini-embedding-2` for `GEMINIOpenAIServiceEndpoint` — and a service without one
(generic endpoints, DeepSeek) needs an explicit `model` (`ArgumentError` otherwise). Works
with any provider via the `service` parameter — Ollama, Gemini, Mistral, or any
OpenAI-compatible server.

An input is a text, an [`EmbeddingInput`](@ref), or an [`ImageAttachment`](@ref) or
[`AudioAttachment`](@ref) on its own; a vector of them embeds each item as a vector of its own.
A [`DocumentAttachment`](@ref), or an image with a detail level, is an `ArgumentError`, as in an
`EmbeddingInput`. A vector of strings is copied into a `Vector{String}` and sent as is, an empty
string included. In a vector holding any other item, every item becomes an `EmbeddingInput`, so an
empty string there is refused (`ArgumentError`: an item needs a non-empty text, an image or a sound clip).

The `embeddings` field is **pre-allocated** (`something(dimensions, 1536)` zeros per input),
filled in place by [`embeddingrequest!`](@ref), and resized to the length the model returns.
The result aliases this struct, so use one `Embeddings` per concurrent call.

# Fields
- `service::ServiceEndpointSpec`: LLM provider (default: `OPENAIServiceEndpoint`).
- `model::String`: The embedding model name.
- `input::Union{String,Vector{String},EmbeddingInput,Vector{EmbeddingInput}}`: Text(s) or item(s) to embed.
- `embeddings::Union{Vector{Float64},Vector{Vector{Float64}}}`: Pre-allocated embedding vector(s).
- `user::Union{String,Nothing}`: Optional end-user identifier.
- `dimensions::Union{Int,Nothing}`: Requested vector size, where the model supports it; a value
  that is not positive throws `ArgumentError`.
- `encoding_format::Union{String,Nothing}`: `"float"` or unset; any other format
  (e.g. `"base64"`) throws `ArgumentError`, since the result stores `Float64` vectors.
- `task::Union{Symbol,Nothing}`: What the texts are for, as a task prompt (see below); `nothing` for none.

# Task prompts (EmbeddingGemma, Gemini Embedding 2)
EmbeddingGemma models (`embeddinggemma`, `embeddinggemma-2`, on any service) and `gemini-embedding-2`
read a short prompt that says what a text is for, and neither Ollama nor the OpenAI-compatible request
adds it: pass `task` and UniLM writes the model card's prompt in front of each text — `:search_query`,
`:question_answering`, `:fact_checking`, `:code_retrieval` (the query side), `:document` (the corpus
side, with each item's `title`), `:classification`, `:clustering`, `:sentence_similarity` — or
`task = :none` to send the text as is. Without a task an EmbeddingGemma request is an `ArgumentError`;
`gemini-embedding-2` takes none, its text then sent as is (Google's
[embeddings guide](https://ai.google.dev/gemini-api/docs/embeddings#task-types-embeddings-2) recommends
the prompts for text); any other model takes `task = :none` or none. EmbeddingGemma's `dimensions` are
768 (default), 512, 256 or 128. Through an
[`OllamaEndpoint`](@ref), an EmbeddingGemma 2 request carries the model's trained 8,192-token window
(`num_ctx` and `num_batch`) unless the endpoint sets either (EmbeddingGemma v1 reads 2,048 tokens); with
`truncate=false`, the default, a longer input is an [`EmbeddingFailure`](@ref), never a vector of part of it
(measured on Ollama 0.40.6 with `740m-q8_0`, `740m-mxfp8`, `270m-q8_0-text` and v1).

# Example
```julia
emb = Embeddings("Julia is a great language")
embeddingrequest!(emb)
emb.embeddings  # => Float64[...] (1536 dims)

# With Ollama
emb = Embeddings("test"; service=OllamaEndpoint(), model="nomic-embed-text")
```
"""
struct Embeddings
    service::ServiceEndpointSpec
    model::String
    input::Union{String,Vector{String},EmbeddingInput,Vector{EmbeddingInput}}
    embeddings::Union{Vector{Float64},Vector{Vector{Float64}}}
    user::Union{String,Nothing}
    dimensions::Union{Int,Nothing}
    encoding_format::Union{String,Nothing}
    task::Union{Symbol,Nothing}
    function Embeddings(input::Union{String,Vector{String},EmbeddingInput,Vector{EmbeddingInput}},
                        service::ServiceEndpointSpec, model::String, dimensions::Union{Int,Nothing},
                        encoding_format::Union{String,Nothing}, user::Union{String,Nothing},
                        task::Union{Symbol,Nothing})
        input isa AbstractVector && isempty(input) && throw(ArgumentError("input must not be empty"))
        m = _embedding_model(service, model)
        _check_embeddings(m, task, dimensions, input)
        n = something(dimensions, 1536)
        buf = input isa AbstractVector ? [zeros(Float64, n) for _ in input] : zeros(Float64, n)
        new(service, m, input, buf, user, dimensions, _float_encoding(encoding_format), task)
    end
end

Embeddings(input::Union{AbstractString,EmbeddingInput,Attachment}; service::ServiceEndpointSpec=OPENAIServiceEndpoint,
           model::String="", dimensions::Union{Int,Nothing}=nothing, encoding_format::Union{String,Nothing}=nothing,
           user::Union{String,Nothing}=nothing, task::Union{Symbol,Nothing}=nothing) =
    Embeddings(_embedding_item(input), service, model, dimensions, encoding_format, user, task)
Embeddings(input::AbstractVector; service::ServiceEndpointSpec=OPENAIServiceEndpoint, model::String="",
           dimensions::Union{Int,Nothing}=nothing, encoding_format::Union{String,Nothing}=nothing,
           user::Union{String,Nothing}=nothing, task::Union{Symbol,Nothing}=nothing) =
    Embeddings(_embedding_items(input), service, model, dimensions, encoding_format, user, task)

# The embedding model: the given one, else the service's default.
function _embedding_model(service::ServiceEndpointSpec, model::String)::String
    isempty(model) || return model
    dm = default_embedding_model(service)
    isnothing(dm) && throw(ArgumentError(
        "model must be specified for embeddings with $(service isa Type ? nameof(service) : nameof(typeof(service)))"))
    dm
end

# The result stores Float64 vectors, so only the float encoding can be decoded.
_float_encoding(format::Union{String,Nothing}) =
    isnothing(format) || format == "float" ? format : throw(ArgumentError(
        "encoding_format must be \"float\" (got $(repr(format))): embeddings are stored as Float64 vectors"))

# The text a model is sent for `text` under `task`: the model card's prompt, or the text itself.
# Construction has checked that the model's family knows the task.
function _prompted(task::Union{Symbol,Nothing}, text::String, title::Union{String,Nothing})::String
    (isnothing(task) || task === :none) && return text
    task === :document && return string("title: ", something(title, "none"), " | text: ", text)
    string("task: ", _EMBEDDING_TASK_NAMES[task], " | query: ", text)
end

# The input on a text-only wire (the OpenAI Embeddings API and its compatibles): texts, prompted.
_wire_text(emb, x::String) = _prompted(emb.task, x, nothing)
function _wire_text(emb, x::EmbeddingInput)
    _item_media(x) && throw(ArgumentError(
        "the OpenAI Embeddings API takes text only (service: $(_service_name(emb.service))): send images and " *
        "sound clips with embeddingrequest! to an OllamaEndpoint, on Ollama's /api/embed (a model such as " *
        "embeddinggemma-2), or leave them out"))
    _prompted(emb.task, x.text::String, x.title)
end
_wire_texts(emb) = emb.input isa AbstractVector ? String[_wire_text(emb, x) for x in emb.input] : _wire_text(emb, emb.input)

function JSON.lower(emb::Embeddings)
    d = Dict{Symbol,Any}(:model => emb.model, :input => _wire_texts(emb))
    !isnothing(emb.user) && (d[:user] = emb.user)
    !isnothing(emb.dimensions) && (d[:dimensions] = emb.dimensions)
    !isnothing(emb.encoding_format) && (d[:encoding_format] = emb.encoding_format)
    return d
end

# Dispatch on the concrete buffer shape rather than branching on `emb.input isa String`.
# The inner constructor pairs a single input with a `Vector{Float64}` buffer and a vector
# input with a `Vector{Vector{Float64}}` buffer, so the buffer type alone
# selects the fill strategy — and each buffer limb has a matching method (no phantom
# `_store_embedding!(::Vector{Vector{Float64}}, …)` split on the `input`-typed branch).
update!(emb::Embeddings, data::AbstractVector) = _fill_embeddings!(emb.embeddings, data)

# The buffers start pre-zeroed, so any slot the response does not cover would stay a
# valid-looking all-zero vector — a silently corrupt embedding that still compares,
# normalizes and indexes. A response that does not cover every input exactly once is
# therefore an error, not a partial fill (`embeddingrequest!` reports it as an
# `EmbeddingCallError`).
function _fill_embeddings!(dst::Vector{Float64}, data::AbstractVector)
    length(data) == 1 || throw(ArgumentError(
        "embeddings response carries $(length(data)) rows for 1 input"))
    _store_embedding!(dst, _checked_embedding(_row_field(data[1], "embedding"), 0))
end

# A field of a reply row; a row that is not a JSON object has none, so it gets the row-named refusal of
# a missing field.
_row_field(item, key::String) = item isa AbstractDict ? get(item, key, nothing) : nothing

function _fill_embeddings!(dst::Vector{Vector{Float64}}, data::AbstractVector)
    n = length(dst)
    length(data) == n || throw(ArgumentError(
        "embeddings response carries $(length(data)) rows for $n inputs"))
    slots = _embedding_slots(data, n)
    for (slot, item) in zip(slots, data)
        _checked_embedding(_row_field(item, "embedding"), slot - 1)
    end
    # Every row is checked before any is stored, so a refused reply leaves the buffers as they were.
    for (slot, item) in zip(slots, data)
        _store_embedding!(dst[slot], item["embedding"])
    end
    return dst
end

# The input each reply row fills, 1-based. The OpenAI Embeddings API numbers its rows with a
# 0-based `index`, in any order. Gemini's OpenAI-compatible endpoint leaves the index out of its
# first row — proto3 JSON omits a field at its default value, 0 — and numbers the others (observed
# 2026-10-22 for gemini-embedding-2 and gemini-embedding-001: no index, then 1, 2), so a row
# without an index beside numbered ones is row 0. A reply that numbers no row is read in order.
# An index that is not an integer, out of range or repeated has no reading, so it is refused.
function _embedding_slots(data::AbstractVector, n::Int)::Vector{Int}
    indices = Any[_row_field(item, "index") for item in data]
    all(isnothing, indices) && return collect(1:n)
    seen = falses(n)
    slots = Vector{Int}(undef, length(indices))
    for (k, index) in enumerate(indices)
        i = something(index, 0)
        # A JSON true parses as a `Bool`, which is an `Integer` in Julia: it is no row index.
        i isa Integer && !(i isa Bool) || throw(ArgumentError(
            "embeddings response row index $(repr(i)) is not an integer"))
        checkbounds(Bool, seen, i + 1) || throw(ArgumentError(
            "embeddings response row index $i is out of range for $n inputs"))
        seen[i + 1] && throw(ArgumentError("embeddings response repeats row index $i" *
            (i == 0 && any(isnothing, indices) ? " (a row without an index is row 0)" : "")))
        seen[i + 1] = true
        slots[k] = i + 1
    end
    slots
end

# A vector that is empty, all zeros, or holds a value that is not a finite number is no embedding
# (Ollama has answered all-zero vectors with HTTP 200; a JSON `true` parses as a `Bool`, which is a
# `Real` in Julia): it is a malformed reply, refused with its row (the 0-based index the row fills),
# never stored as a vector that still compares and indexes.
function _checked_embedding(src::AbstractVector, row)
    all(x -> x isa Real && !(x isa Bool) && isfinite(x), src) || throw(ArgumentError(
        "the embeddings reply's row $row holds a value that is not a finite number"))
    !isempty(src) && any(!iszero, src) || throw(ArgumentError(
        "the embeddings reply's row $row is an empty or all-zero vector, which no model returns for real input"))
    src
end
# A row holding null, a number or a string has no vector at all.
_checked_embedding(src, row) = throw(ArgumentError(
    "the embeddings reply's row $row is not a vector (got $(isnothing(src) ? "null" : typeof(src)))"))

# Copy a checked embedding into the preallocated buffer, resizing when the model's actual
# dimension differs from the 1536 default.
function _store_embedding!(dst::Vector{Float64}, src::AbstractVector)
    length(dst) == length(src) || resize!(dst, length(src))
    @inbounds for i in eachindex(src)
        dst[i] = src[i]
    end
    return dst
end

"""
    EmbeddingSuccess(; embeddings, usage=nothing, raw)

Successful Embeddings API response. Vectors are in `embeddings.embeddings` (also filled
in place on the request struct); `embedding_vectors(r)` returns them.
"""
@kwdef struct EmbeddingSuccess <: LLMRequestResponse
    embeddings::Embeddings
    usage::Union{TokenUsage,Nothing} = nothing
    raw::Dict{String,Any}
end

"""
    EmbeddingFailure(; response, status)

HTTP-level failure from the Embeddings API (non-2xx).
"""
@kwdef struct EmbeddingFailure <: LLMRequestResponse
    response::String
    status::Int
end

"""
    EmbeddingCallError(; error, status=nothing, cause=nothing)

Exception-level error during an Embeddings API call (network, parse, timeout, etc.).
`cause` holds the underlying typed exception when one exists (e.g. a `UniLMTimeout`);
`status` stays `nothing` for timeouts — no fabricated HTTP statuses.
"""
@kwdef struct EmbeddingCallError <: LLMRequestResponse
    error::String
    status::Union{Int,Nothing} = nothing
    cause::Union{Nothing,Exception} = nothing
end

Base.show(io::IO, r::EmbeddingCallError) =
    _show_call_error(io, "EmbeddingCallError", r.error, r.status, nothing, r.cause)

"""
    embedding_vectors(r::EmbeddingSuccess)

Return the embedding vector(s): `Vector{Float64}` (single input) or `Vector{Vector{Float64}}` (batch).
On an [`EmbeddingFailure`](@ref) or [`EmbeddingCallError`](@ref) it throws an
[`LLMResultError`](@ref); guard with [`issuccess`](@ref).
"""
embedding_vectors(r::EmbeddingSuccess) = r.embeddings.embeddings
embedding_vectors(r::Union{EmbeddingFailure,EmbeddingCallError}) = throw(LLMResultError(r))
