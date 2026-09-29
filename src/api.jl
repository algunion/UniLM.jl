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

"""
    GPTImageContent(text, images)

Message content combining text with images. `text` is the prompt and `images`
is a vector of image URLs or base64-encoded image data.
"""
struct GPTImageContent
    text::String
    images::Vector{String}
end

function JSON.lower(x::GPTImageContent)
    d = Dict{Symbol,Any}[Dict{Symbol,Any}(:type => "text", :text => x.text)]
    for i in x.images
        push!(d, Dict{Symbol,Any}(:type => "image_url", :image_url => Dict(:url => i, :detail => "auto")))
    end
    return d
end

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
are rejected with HTTP 400), Gemini text-part `thoughtSignature`s, and DeepSeek
`reasoning_content`. `provider` tags the wire dialect (`:anthropic`, `:gemini` or
`:deepseek`); `blocks` is the provider's content/parts array exactly as decoded
(String-keyed JSON) — for `:deepseek`, `[Dict("reasoning_content" => text)]`, echoed
only on requests that carry tools.

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
[`ImageAttachment`](@ref) and [`AudioAttachment`](@ref). An attachment holds the
media's bytes and the format read from their leading bytes (the file signature),
never from a file name, so its declared format always matches its content.

Endpoints on the OpenAI Chat Completions wire send attachments as `image_url` and
`input_audio` content parts (base64); [`OllamaEndpoint`](@ref) sends them in the
message's `images` list. The native Anthropic and Gemini encoders do not map
attachments yet and throw `ArgumentError` before any network I/O.
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

"""
    ImageAttachment(path::AbstractString)
    ImageAttachment(data::AbstractVector{UInt8})

An image for a user [`Message`](@ref): PNG, JPEG, GIF or WebP, recognised from its
leading bytes. `mime` holds the media type (`"image/png"`, …). Throws `ArgumentError`
for empty data or bytes in any other format.

```julia
msg = Message(Val(:user), "What colour is the shape?", ImageAttachment("shape.png"))
```
"""
struct ImageAttachment <: Attachment
    data::Vector{UInt8}
    mime::String
    function ImageAttachment(data::AbstractVector{UInt8})
        mime = _sniff_image_mime(data)
        isnothing(mime) && throw(ArgumentError(
            "image data is not PNG, JPEG, GIF or WebP (checked by file signature; $(length(data)) bytes)"))
        new(Vector{UInt8}(data), mime)
    end
end
ImageAttachment(path::AbstractString) = ImageAttachment(read(path))

"""
    AudioAttachment(path::AbstractString)
    AudioAttachment(data::AbstractVector{UInt8})

A sound clip for a user [`Message`](@ref): WAV or MP3, recognised from its leading
bytes. `format` holds `"wav"` or `"mp3"`, the names the OpenAI `input_audio` part
uses. Throws `ArgumentError` for empty data or bytes in any other format.

```julia
msg = Message(Val(:user), "Transcribe this.", AudioAttachment("clip.wav"))
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
AudioAttachment(path::AbstractString) = AudioAttachment(read(path))

# The bytes stay reachable through the fields; printing them would flood the REPL.
Base.show(io::IO, a::ImageAttachment) = print(io, "ImageAttachment(", a.mime, ", ", length(a.data), " bytes)")
Base.show(io::IO, a::AudioAttachment) = print(io, "AudioAttachment(", a.format, ", ", length(a.data), " bytes)")
Base.:(==)(a::A, b::A) where {A<:Attachment} = all(f -> getfield(a, f) == getfield(b, f), fieldnames(A))
Base.hash(a::Attachment, h::UInt) = foldl((h, f) -> hash(getfield(a, f), h), fieldnames(typeof(a)); init=hash(typeof(a), h))

"""
    Message(; role, content=nothing, name=nothing, finish_reason=nothing, refusal_message=nothing, tool_calls=nothing, tool_call_id=nothing, provider_content=nothing, attachments=nothing)

Represents a single message in a Chat Completions conversation.

# Fields
- `role::String`: One of [`RoleSystem`](@ref), [`RoleUser`](@ref), [`RoleAssistant`](@ref), or `RoleTool`.
- `content::Union{String,Nothing}`: The text content of the message.
- `name::Union{String,Nothing}`: Optional name for the participant.
- `finish_reason::Union{String,Nothing}`: Why the model stopped generating (e.g. `"stop"`, `"tool_calls"`); `nothing` when the provider reported none. Response-only: never sent in a request.
- `refusal_message::Union{String,Nothing}`: Refusal text when content is filtered; sent on the wire as `refusal`.
- `tool_calls::Union{Nothing,Vector{ToolCall}}`: Tool calls requested by the assistant.
- `tool_call_id::Union{String,Nothing}`: Required when `role` is `"tool"` — the ID of the tool call being responded to.
- `provider_content::Union{Nothing,ProviderContent}`: Provider-native content blocks captured for verbatim round-trip (see [`ProviderContent`](@ref)); set by the Anthropic, Gemini and DeepSeek decoders (tags `:anthropic`, `:gemini`, `:deepseek`), `nothing` otherwise. Never serialized on the OpenAI wire.
- `attachments::Union{Nothing,Vector{Attachment}}`: Images and sound clips sent with a user message ([`ImageAttachment`](@ref), [`AudioAttachment`](@ref)); `nothing` for none. See [`Attachment`](@ref) for how each endpoint sends them.

# Validation
- `role` must be one of `"system"`, `"user"`, `"assistant"`, `"tool"`.
- At least one of `content`, `tool_calls`, or `refusal_message` must be non-`nothing`.
- `tool_call_id` is required when `role == "tool"`.
- `attachments` are accepted on user messages only; an empty vector is stored as `nothing`.

# Convenience Constructors
```julia
Message(Val(:system), "You are a helpful assistant")
Message(Val(:user), "Hello!")
Message(Val(:user), "What colour is the shape?", ImageAttachment("shape.png"))
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
    function Message(role, content, name, finish_reason, refusal_message, tool_calls,
                     tool_call_id, provider_content=nothing, attachments=nothing)
        role in _MESSAGE_ROLES || throw(ArgumentError(
            "message role must be one of $(join(_MESSAGE_ROLES, ", ")) (got $(repr(role)))"))
        isnothing(content) && isnothing(tool_calls) && isnothing(refusal_message) && throw(ArgumentError("`content`, `tool_calls`, and `refusal_message` cannot all be nothing"))
        role == RoleTool && isnothing(tool_call_id) && throw(ArgumentError("`tool_call_id` cannot be empty when role is `tool`"))
        isnothing(attachments) || !isempty(attachments) || (attachments = nothing)   # no media is no media list
        isnothing(attachments) || role == RoleUser || throw(ArgumentError(
            "attachments are accepted on user messages only (got a :$role message with $(length(attachments)))"))
        return new(role, content, name, finish_reason, refusal_message, tool_calls,
                   tool_call_id, provider_content, attachments)
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
# finish_reason is a response-only field: neither is sent. A refusal travels under
# its wire name, `refusal`; an assistant message needs `content` unless it carries
# tool calls, so a refusal turn sends it as null — the shape the API returns it in.
# Attachments turn `content` into an array of parts: the text (when there is any),
# then each attachment as an `image_url` data URL or an `input_audio` part.
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

_wire_part(a::ImageAttachment) = Dict{Symbol,Any}(:type => "image_url",
    :image_url => Dict(:url => string("data:", a.mime, ";base64,", base64encode(a.data))))
_wire_part(a::AudioAttachment) = Dict{Symbol,Any}(:type => "input_audio",
    :input_audio => Dict(:data => base64encode(a.data), :format => a.format))

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
- `GenericOpenAIEndpoint` — any OpenAI-compatible provider (Ollama, Mistral, vLLM, etc.)
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
`GEMINIOpenAIServiceEndpoint`, `GenericOpenAIEndpoint`, `DeepSeekEndpoint`.
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

"""Anthropic (Claude) native Messages API endpoint. Requires the `ANTHROPIC_API_KEY` env variable.
Native wire format (content blocks, top-level `system`, `user`/`assistant` roles) — NOT OpenAI-compatible."""
struct ANTHROPICServiceEndpoint <: ServiceEndpoint end

"""
    GenericOpenAIEndpoint <: OpenAIWireEndpoint

Configurable endpoint for any OpenAI-compatible API provider. Supports Chat Completions,
Embeddings, and (where the provider implements it) the Responses API.

# Fields
- `base_url::String`: Base URL without trailing slash (e.g., `"http://localhost:11434"`)
- `api_key::String`: API key for Bearer auth. Use `""` for local servers with no auth.

# Example
```julia
# Ollama (local)
chat = Chat(service=GenericOpenAIEndpoint("http://localhost:11434", ""), model="llama3.1")

# Mistral
chat = Chat(service=GenericOpenAIEndpoint("https://api.mistral.ai", ENV["MISTRAL_API_KEY"]),
            model="mistral-large-latest")
```
"""
struct GenericOpenAIEndpoint <: OpenAIWireEndpoint
    base_url::String
    api_key::String
end

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
    OllamaOptions(; num_ctx=nothing, num_batch=nothing, num_gpu=nothing, main_gpu=nothing,
                  num_thread=nothing, use_mmap=nothing, num_keep=nothing, top_k=nothing,
                  min_p=nothing, repeat_last_n=nothing, repeat_penalty=nothing)

The Ollama runtime and sampling options that [`Chat`](@ref) has no field for, sent as
the `options` object of an [`OllamaEndpoint`](@ref) request; unset options are
omitted, so the model's own defaults apply. Options `Chat` already names
(`temperature`, `top_p`, `seed`, `stop`, `max_tokens`, `presence_penalty`,
`frequency_penalty`) stay on the `Chat`, so no option can be set in two places.

- `num_ctx`: context window in tokens. Ollama loads the model with this window, so a
  change reloads it.
- `num_batch`, `num_gpu` (layers offloaded to the GPU), `main_gpu`, `num_thread`,
  `use_mmap`: how the model runs on this machine.
- `num_keep`: tokens of the prompt kept when the context is shifted.
- `top_k`, `min_p`, `repeat_last_n`, `repeat_penalty`: sampling.

The constructor throws `ArgumentError` for a value out of range: `num_ctx`,
`num_batch` and `num_thread` must be positive, `top_k` at least 1, `min_p` in
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
        isnothing(top_k)          || _check(top_k >= 1, "top_k must be at least 1 (got $top_k)")
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
# http on port 11434; an explicit http:// or https:// without a port means 80 or 443;
# a missing host means 127.0.0.1; a path is kept. Blank means the default server. A
# bind-all address (0.0.0.0, [::]) tells the server to listen everywhere; a client
# reaches it on the loopback address, as the Ollama CLI does.
function _ollama_host_url(host::AbstractString)::String
    s = strip(host)
    isempty(s) && return "http://127.0.0.1:11434"
    scheme, rest, port = "http", s, "11434"
    m = match(r"^([A-Za-z][A-Za-z0-9+.-]*)://(.*)$", s)
    if !isnothing(m)
        scheme = lowercase(m.captures[1])
        scheme in ("http", "https") || throw(ArgumentError(
            "OLLAMA_HOST scheme must be http or https (got $(repr(String(m.captures[1]))))"))
        rest = m.captures[2]
        port = scheme == "https" ? "443" : "80"
    end
    hostport, path = let i = findfirst('/', rest)
        isnothing(i) ? (rest, "") : (rest[1:prevind(rest, i)], rstrip(rest[i:end], '/'))
    end
    h = match(r"^(\[[^\]]*\]|[^:]*)(?::(\d*))?$", hostport)
    isnothing(h) && throw(ArgumentError("OLLAMA_HOST is not host[:port] (got $(repr(String(host))))"))
    name = isempty(h.captures[1]) || h.captures[1] == "0.0.0.0" ? "127.0.0.1" :
           h.captures[1] == "[::]" ? "[::1]" : h.captures[1]
    p = h.captures[2]
    if !isnothing(p) && !isempty(p)
        n = tryparse(Int, p)
        (isnothing(n) || !(0 < n <= 65535)) && throw(ArgumentError("OLLAMA_HOST port must be 1-65535 (got $(repr(p)))"))
        port = p
    end
    string(scheme, "://", name, ":", port, path)
end

"""
    OllamaEndpoint <: OpenAIWireEndpoint
    OllamaEndpoint(; base_url=<OLLAMA_HOST or http://127.0.0.1:11434>, keep_alive=nothing, options...)

A local (or remote) [Ollama](https://ollama.com) server. Chat requests use Ollama's
native `/api/chat` API, which carries what the OpenAI-compatible route cannot: the
context window and the other [`OllamaOptions`](@ref), how long the model stays
loaded, and thinking control. [`respond`](@ref), [`Embeddings`](@ref) and
[`FIMCompletion`](@ref) use Ollama's OpenAI-compatible routes.

- `base_url`: server root, without `/v1` or `/api`. Defaults to the `OLLAMA_HOST`
  environment variable, read with the rules of Ollama's own clients (`"gpu-box"` →
  `http://gpu-box:11434`), else `http://127.0.0.1:11434`.
- `keep_alive`: seconds the model stays loaded after each request (`0` unloads it at
  once, `Inf` keeps it loaded); `nothing` leaves the server's default (5 minutes
  unless `OLLAMA_KEEP_ALIVE` says otherwise).
- `truncate`: whether Ollama may drop the oldest messages (chat) or the end of an input
  (embeddings) to fit the context window. `false` (the default) makes an input that does
  not fit a failure — HTTP 400 "exceeds the available context size" — instead of an
  answer to a silently shortened prompt; raise `num_ctx`, or set `true` for Ollama's
  own behaviour.
- `shift`: whether generation that fills the context window may continue by dropping
  earlier context (`nothing`: the server's default, on). `false` ends such a reply with
  finish reason `"length"`. Ollama reloads the model when this changes.
- `options...`: keyword arguments of [`OllamaOptions`](@ref), e.g. `num_ctx=32_768`.

No API key is sent. Local models cost nothing: [`estimated_cost`](@ref) reports `0.0`
for them without a missing-price warning.

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
    options::OllamaOptions
    function OllamaEndpoint(base_url::AbstractString, keep_alive::Union{Real,Nothing}, truncate::Bool,
                            shift::Union{Bool,Nothing}, options::OllamaOptions)
        url = String(rstrip(base_url, '/'))
        occursin(r"^https?://[^/]", url) || throw(ArgumentError(
            "Ollama base_url must start with http:// or https:// (got $(repr(String(base_url))))"))
        isnothing(keep_alive) || (!isnan(keep_alive) && keep_alive >= 0) || throw(ArgumentError(
            "keep_alive must be a non-negative number of seconds, or Inf to keep the model loaded (got $keep_alive)"))
        new(url, isnothing(keep_alive) ? nothing : Float64(keep_alive), truncate, shift, options)
    end
end

function OllamaEndpoint(; base_url::AbstractString=_ollama_host_url(get(ENV, "OLLAMA_HOST", "")),
                        keep_alive::Union{Real,Nothing}=nothing, truncate::Bool=false,
                        shift::Union{Bool,Nothing}=nothing, options...)
    unknown = setdiff(keys(options), fieldnames(OllamaOptions))
    isempty(unknown) || throw(ArgumentError("unknown OllamaEndpoint keyword(s) $(join(unknown, ", ")); " *
        "expected base_url, keep_alive, truncate, shift or an OllamaOptions field: " *
        join(fieldnames(OllamaOptions), ", ")))
    OllamaEndpoint(base_url, keep_alive, truncate, shift, OllamaOptions(; options...))
end

function Base.show(io::IO, e::OllamaEndpoint)
    print(io, "OllamaEndpoint(")
    show(io, e.base_url)
    set = Tuple{Symbol,Any}[(f, getfield(e.options, f)) for f in fieldnames(OllamaOptions)
                            if !isnothing(getfield(e.options, f))]
    isnothing(e.shift) || pushfirst!(set, (:shift, e.shift))
    e.truncate && pushfirst!(set, (:truncate, true))
    isnothing(e.keep_alive) || pushfirst!(set, (:keep_alive, e.keep_alive))
    isempty(set) || print(io, "; ", join(("$f=$(repr(v))" for (f, v) in set), ", "))
    print(io, ")")
end

"""
    MistralEndpoint(; api_key=ENV["MISTRAL_API_KEY"]) -> GenericOpenAIEndpoint

Pre-configured endpoint for [Mistral AI](https://mistral.ai) API.
"""
MistralEndpoint(; api_key::String=ENV["MISTRAL_API_KEY"]) = GenericOpenAIEndpoint("https://api.mistral.ai", api_key)

"""
    DeepSeekEndpoint <: OpenAIWireEndpoint

Pre-configured endpoint for [DeepSeek](https://deepseek.com) API. Supports chat completions,
tool calling, FIM completion, and prefix completion.

FIM and prefix completion use the beta base URL (`https://api.deepseek.com/beta`).
"""
struct DeepSeekEndpoint <: OpenAIWireEndpoint
    api_key::String
end
DeepSeekEndpoint(; api_key::String=ENV["DEEPSEEK_API_KEY"]) = DeepSeekEndpoint(api_key)

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
    print(io, ")")
end

function Base.show(io::IO, e::DeepSeekEndpoint)
    print(io, "DeepSeekEndpoint(")
    show(io, _redact_api_key(e.api_key))
    print(io, ")")
end


# Coerce the `tools` keyword to the stored `Vector{Tool}`. This fallback is
# the identity — a `Vector{Tool}` or `nothing` passes through unchanged. The
# method that unwraps a `Vector{<:CallableTool}` into its inner `Tool`s lives
# in tool_loop.jl, where `CallableTool` is defined (that file is `include`d
# after this one). Conversion happens only here at construction; the field type
# stays `Union{Vector{Tool},Nothing}`.
_chat_tools(tools) = tools

"""
    PromptCacheOptions(; mode=nothing, ttl=nothing, prewarm=nothing, comparison_response_id=nothing)

OpenAI prompt-cache controls for GPT-5.6 and later. `mode` is `"implicit"` or
`"explicit"`; the supported `ttl` is `"30m"`. Explicit mode caches only prefixes
marked with `prompt_cache_breakpoint` in input content blocks (see
[`input_text`](@ref) with `cache_breakpoint=true`).

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
        isnothing(ttl) || ttl == "30m" || throw(ArgumentError("prompt cache ttl must be 30m"))
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
- `model` is set to `gpt-5.6-sol`
- `messages` is set to an empty `Vector{Message}`
- `history` is set to `true`

OpenAI options, omitted from the request when unset:
- `prompt_cache_options::Union{PromptCacheOptions,Nothing}`: prompt-cache `mode` and
  `ttl` ([`PromptCacheOptions`](@ref)); supported for gpt-5.6 and later (gpt-5.4-mini
  answered HTTP 400 "prompt_cache_options is not supported on this model" on
  2026-09-22). Chat messages cannot carry cache breakpoints, so `mode="explicit"` means
  the request does not use prompt caching.
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
    max_completion_tokens::Union{Int64,Nothing} = nothing # preferred over max_tokens; reasoning models reject max_tokens
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
    service_tier::Union{String,Nothing} = nothing          # auto|default|flex|scale|priority|fast
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
    d = Dict{Symbol,Any}(:model => chat.model, :messages => chat.messages)
    for f in (:tools, :tool_choice, :parallel_tool_calls, :temperature, :top_p,
        :n, :stream, :stop, :max_tokens, :max_completion_tokens, :presence_penalty, :response_format,
        :frequency_penalty, :logit_bias, :user, :seed,
        :reasoning_effort, :stream_options, :verbosity, :store, :metadata, :service_tier,
        :logprobs, :top_logprobs, :prediction, :modalities, :audio, :web_search_options,
        :prompt_cache_key, :safety_identifier, :prompt_cache_options, :moderation)
        v = getfield(chat, f)
        !isnothing(v) && (d[f] = v)
    end
    # OpenAI and DeepSeek report a stream's token usage when asked through
    # `stream_options.include_usage` (both document it); ask for the caller who set
    # no stream_options, so a streamed turn is costed like a non-streamed one. Not
    # for other OpenAI-compatible servers, which may reject the field.
    chat.stream === true && isnothing(chat.stream_options) && _asks_stream_usage(chat.service) &&
        (d[:stream_options] = Dict(:include_usage => true))
    return d
end

_asks_stream_usage(::Type{OPENAIServiceEndpoint}) = true
_asks_stream_usage(::DeepSeekEndpoint) = true
_asks_stream_usage(_) = false

# An encoder that cannot map `Message.attachments` refuses the request: dropping the
# media would ask the model about an image or a sound it never receives.
function _reject_attachments(chat::Chat, encoder::AbstractString)::Nothing
    any(m -> !isnothing(m.attachments), chat.messages) && throw(ArgumentError(
        "the $encoder encoder does not map message attachments (images, audio) yet; " *
        "send them through an OpenAI-compatible endpoint or OllamaEndpoint"))
    nothing
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
    TokenUsage(; prompt_tokens=0, completion_tokens=0, total_tokens=0, cached_tokens=0, reasoning_tokens=0)

Token usage statistics returned by the API.

`cached_tokens` (a subset of `prompt_tokens` served from the prompt cache) and
`reasoning_tokens` (a subset of `completion_tokens` spent on hidden reasoning) are
reported by newer models via the `*_tokens_details` objects; they default to 0 when
the provider omits those details. `estimated_cost` bills `cached_tokens` at the
discounted cached-input rate.
"""
@kwdef struct TokenUsage
    prompt_tokens::Int = 0
    completion_tokens::Int = 0
    total_tokens::Int = 0
    cached_tokens::Int = 0
    reasoning_tokens::Int = 0
end

"""
    LLMSuccess(; message, self, usage=nothing, sse_dropped=0)

Successful Chat Completions API response.

# Fields
- `message::Message`: The assistant's reply message.
- `self::Chat`: The updated [`Chat`](@ref) object (with the new message appended if `history=true`).
- `usage::Union{TokenUsage, Nothing}`: Token usage statistics from the API.
- `sse_dropped::Int`: Undecodable SSE `data:` payloads dropped while assembling
  this streamed turn — `0` for a non-streamed call, and for a clean stream.
  Non-zero means the turn was built from an incomplete wire.
"""
@kwdef struct LLMSuccess <: LLMRequestResponse
    message::Message
    self::Chat
    usage::Union{TokenUsage, Nothing} = nothing
    sse_dropped::Int = 0
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
chat = Chat(service=OllamaEndpoint(), model="gemma4:e4b", reasoning_effort="high")
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
(it must start with a system message), a system message pushed once the
conversation has already started, or a message whose role repeats the last
message's role (consecutive tool-result messages are the one exception).
`chat[i] = msg` and direct edits to `chat.messages` are not validated.
"""
function Base.push!(chat::Chat, msg::Message)
    if msg.role == RoleSystem
        isempty(chat) ||
            throw(InvalidConversationError("a system message is only valid as the first message; got :$(msg.role) after the conversation started"))
    elseif isempty(chat)
        throw(InvalidConversationError("conversation must start with a system message; got :$(msg.role)"))
    elseif chat.messages[end].role == msg.role && msg.role != RoleTool
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
    Embeddings(input::String; service=OPENAIServiceEndpoint, model="", dimensions=nothing,
               encoding_format=nothing, user=nothing)
    Embeddings(input::Vector{String}; service=OPENAIServiceEndpoint, model="", dimensions=nothing,
               encoding_format=nothing, user=nothing)

Create an embedding request for one or more texts. `model=""` resolves to the service's
default embedding model — OpenAI's `text-embedding-3-small` (1536 dimensions),
`gemini-embedding-001` for `GEMINIOpenAIServiceEndpoint` — and a service without one
(generic endpoints, DeepSeek) needs an explicit `model` (`ArgumentError` otherwise). Works
with any provider via the `service` parameter — Ollama, Gemini, Mistral, or any
OpenAI-compatible server.

The `embeddings` field is **pre-allocated** (`something(dimensions, 1536)` zeros per input),
filled in place by [`embeddingrequest!`](@ref), and resized to the length the model returns.
The result aliases this struct, so use one `Embeddings` per concurrent call.

# Fields
- `service::ServiceEndpointSpec`: LLM provider (default: `OPENAIServiceEndpoint`).
- `model::String`: The embedding model name.
- `input::Union{String,Vector{String}}`: Text(s) to embed.
- `embeddings::Union{Vector{Float64},Vector{Vector{Float64}}}`: Pre-allocated embedding vector(s).
- `user::Union{String,Nothing}`: Optional end-user identifier.
- `dimensions::Union{Int,Nothing}`: Requested vector size, where the model supports it.
- `encoding_format::Union{String,Nothing}`: `"float"` or unset; any other format
  (e.g. `"base64"`) throws `ArgumentError`, since the result stores `Float64` vectors.

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
    input::Union{String,Vector{String}}
    embeddings::Union{Vector{Float64},Vector{Vector{Float64}}}
    user::Union{String,Nothing}
    dimensions::Union{Int,Nothing}
    encoding_format::Union{String,Nothing}
    function Embeddings(input::String; service::ServiceEndpointSpec=OPENAIServiceEndpoint, model::String="",
        dimensions::Union{Int,Nothing}=nothing, encoding_format::Union{String,Nothing}=nothing,
        user::Union{String,Nothing}=nothing)
        return new(service, _embedding_model(service, model), input, zeros(Float64, something(dimensions, 1536)),
                   user, dimensions, _float_encoding(encoding_format))
    end
    function Embeddings(input::Vector{String}; service::ServiceEndpointSpec=OPENAIServiceEndpoint, model::String="",
        dimensions::Union{Int,Nothing}=nothing, encoding_format::Union{String,Nothing}=nothing,
        user::Union{String,Nothing}=nothing)
        isempty(input) && throw(ArgumentError("input must not be empty"))
        return new(service, _embedding_model(service, model), input,
                   [zeros(Float64, something(dimensions, 1536)) for _ in 1:length(input)], user, dimensions,
                   _float_encoding(encoding_format))
    end
end

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

function JSON.lower(emb::Embeddings)
    d = Dict{Symbol,Any}(:model => emb.model, :input => emb.input)
    !isnothing(emb.user) && (d[:user] = emb.user)
    !isnothing(emb.dimensions) && (d[:dimensions] = emb.dimensions)
    !isnothing(emb.encoding_format) && (d[:encoding_format] = emb.encoding_format)
    return d
end

# Dispatch on the concrete buffer shape rather than branching on `emb.input isa String`.
# The two inner constructors pair a String input with a `Vector{Float64}` buffer and a
# `Vector{String}` input with a `Vector{Vector{Float64}}` buffer, so the buffer type alone
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
    _store_embedding!(dst, data[1]["embedding"])
end

function _fill_embeddings!(dst::Vector{Vector{Float64}}, data::AbstractVector)
    n = length(dst)
    length(data) == n || throw(ArgumentError(
        "embeddings response carries $(length(data)) rows for $n inputs"))
    seen = falses(n)
    for item in data
        idx = item["index"] + 1  # API uses 0-based indexing
        checkbounds(Bool, dst, idx) || throw(ArgumentError(
            "embeddings response row index $(item["index"]) is out of range for $n inputs"))
        seen[idx] && throw(ArgumentError(
            "embeddings response repeats row index $(item["index"])"))
        seen[idx] = true
        _store_embedding!(dst[idx], item["embedding"])
    end
    return dst
end

# Copy an API-returned embedding into the preallocated buffer, resizing when the model's
# actual dimension differs from the 1536 default (e.g. text-embedding-3-large = 3072).
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
