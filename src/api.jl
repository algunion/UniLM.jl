"""
    GPTFunctionSignature(; name, description=nothing, parameters=nothing)

Describes a function that can be called by the model in the Chat Completions API.

# Fields
- `name::String`: The name of the function.
- `description::Union{String,Nothing}`: A description of what the function does.
- `parameters::Union{AbstractDict,Nothing}`: JSON Schema object describing the function parameters.

# Example
```julia
sig = GPTFunctionSignature(
    name="get_weather",
    description="Get the current weather in a given location",
    parameters=Dict(
        "type" => "object",
        "properties" => Dict(
            "location" => Dict("type" => "string", "description" => "The city")
        ),
        "required" => ["location"]
    )
)
```
"""
@kwdef mutable struct GPTFunctionSignature
    name::String
    description::Union{String,Nothing} = nothing
    parameters::Union{AbstractDict,Nothing} = nothing
end

JSON.omit_null(::Type{GPTFunctionSignature}) = true

"""
`text` is the prompt.\n
`images` is a vector of image urls (or base64 encoded images).
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
    GPTToolCall(; id, type="function", func)

Represents a tool call returned by the model. Contains the call `id` (used to match
results back), the tool `type`, and the [`GPTFunction`] with name and parsed arguments.
"""
@kwdef struct GPTToolCall
    id::String
    type::String = "function"
    func::GPTFunction
end

JSON.lower(x::GPTToolCall) = Dict(:id => x.id, :type => x.type, :function => x.func)

"""
    GPTTool(; type="function", func)

Wraps a [`GPTFunctionSignature`](@ref) for use in the `tools` parameter of a [`Chat`](@ref).

# Example
```julia
tool = GPTTool(func=GPTFunctionSignature(
    name="get_weather",
    description="Get the current weather",
    parameters=Dict("type" => "object", "properties" => Dict())
))
chat = Chat(tools=[tool])
```
"""
@kwdef struct GPTTool
    type::String = "function"
    func::GPTFunctionSignature
end

"""
    GPTTool(d::AbstractDict)

Construct a [`GPTTool`](@ref) from a dict. Accepts both the bare format
`{"name": ...}` and the wrapped OpenAI format `{"type": "function", "function": {"name": ...}}`.
"""
function GPTTool(d::AbstractDict)
    inner = haskey(d, "function") && d["function"] isa AbstractDict ? d["function"] : d
    GPTTool(
        type=get(d, "type", "function"),
        func=GPTFunctionSignature(
            name=inner["name"],
            description=get(inner, "description", nothing),
            parameters=get(inner, "parameters", nothing)
        )
    )
end

JSON.lower(x::GPTTool) = Dict(:type => x.type, :function => x.func)


@kwdef struct GPTToolChoice
    type::String = "function"
    func::Union{String,Symbol}
end

JSON.lower(x::GPTToolChoice) = Dict(:type => x.type, :function => Dict(:name => x.func))


"""
    GPTFunctionCallResult{T}

Holds the result of executing a function that was requested by the model via a tool call.

# Fields
- `name::Union{String,Symbol}`: The function name.
- `origincall::GPTFunction`: The original [`GPTFunction`] call from the model.
- `result::T`: The result of executing the function.
"""
struct GPTFunctionCallResult{T}
    name::Union{String,Symbol}
    origincall::GPTFunction
    result::T
end

JSON.omit_null(::Type{<:GPTFunctionCallResult}) = true
JSON.omit_empty(::Type{<:GPTFunctionCallResult}) = true

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

# to do: extend to all models/endpoints
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
    Message(; role, content=nothing, name=nothing, finish_reason=nothing, refusal_message=nothing, tool_calls=nothing, tool_call_id=nothing)

Represents a single message in a Chat Completions conversation.

# Fields
- `role::String`: One of [`RoleSystem`](@ref), [`RoleUser`](@ref), [`RoleAssistant`](@ref), or `RoleTool`.
- `content::Union{String,Nothing}`: The text content of the message.
- `name::Union{String,Nothing}`: Optional name for the participant.
- `finish_reason::Union{String,Nothing}`: Why the model stopped generating (e.g. `"stop"`, `"tool_calls"`).
- `refusal_message::Union{String,Nothing}`: Refusal text when content is filtered.
- `tool_calls::Union{Nothing,Vector{GPTToolCall}}`: Tool calls requested by the assistant.
- `tool_call_id::Union{String,Nothing}`: Required when `role` is `"tool"` — the ID of the tool call being responded to.

# Validation
- At least one of `content`, `tool_calls`, or `refusal_message` must be non-`nothing`.
- `tool_call_id` is required when `role == "tool"`.

# Convenience Constructors
```julia
Message(Val(:system), "You are a helpful assistant")
Message(Val(:user), "Hello!")
```
"""
@kwdef struct Message
    role::String
    content::Union{String,Nothing} = nothing
    name::Union{String,Nothing} = nothing
    finish_reason::Union{String,Nothing} = nothing
    refusal_message::Union{String,Nothing} = nothing
    tool_calls::Union{Nothing,Vector{GPTToolCall}} = nothing
    tool_call_id::Union{String,Nothing} = nothing
    function Message(role, content, name, finish_reason, refusal_message, tool_calls, tool_call_id)
        isnothing(content) && isnothing(tool_calls) && isnothing(refusal_message) && throw(ArgumentError("`content`, `tool_calls`, and `refusal_message` cannot all be nothing"))
        role == RoleTool && isnothing(tool_call_id) && throw(ArgumentError("`tool_call_id` cannot be empty when role is `tool`"))
        return new(role, content, name, finish_reason, refusal_message, tool_calls, tool_call_id)
    end
end

Message(::Val{:system}, content) = Message(role=RoleSystem, content=content)
Message(::Val{:user}, content) = Message(role=RoleUser, content=content)

const Conversation = Vector{Message}

JSON.omit_null(::Type{Message}) = true

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
end

# JsonSchemaAPI serializes as a regular struct (JSON.jl handles this automatically)

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
fmt = ResponseFormat(JsonSchemaAPI(
    name="result",
    description="A structured result",
    schema=Dict("type" => "object", "properties" => Dict())
))
```

!!! note
    Use the convenience constructors `json_object()` and `json_schema()` for cleaner code.
"""
@kwdef struct ResponseFormat
    type::String = "json_object"
    json_schema::Union{JsonSchemaAPI,AbstractDict,Nothing} = nothing
end

ResponseFormat(json_schema) = ResponseFormat("json_schema", json_schema)

JSON.omit_null(::Type{ResponseFormat}) = true

json_object() = ResponseFormat()
json_schema(schema) = ResponseFormat(schema)
json_schema(name::String, description::String, schema::AbstractDict) = ResponseFormat(JsonSchemaAPI(name, description, schema))

"""
    ServiceEndpoint

Abstract supertype for LLM service backends. Subtypes control URL routing and authentication.

Built-in subtypes:
- `OPENAIServiceEndpoint` — OpenAI API (default)
- `AZUREServiceEndpoint` — Azure OpenAI Service
- `GEMINIServiceEndpoint` — Google Gemini via OpenAI-compatible endpoint
- `GenericOpenAIEndpoint` — any OpenAI-compatible provider (Ollama, Mistral, vLLM, etc.)
"""
abstract type ServiceEndpoint end

"""OpenAI API service endpoint (default). Requires `OPENAI_API_KEY` env variable."""
struct OPENAIServiceEndpoint <: ServiceEndpoint end

"""Azure OpenAI Service endpoint. Requires `AZURE_OPENAI_BASE_URL`, `AZURE_OPENAI_API_KEY`, and `AZURE_OPENAI_API_VERSION` env variables."""
struct AZUREServiceEndpoint <: ServiceEndpoint end

"""Google Gemini endpoint (OpenAI-compatible). Requires `GEMINI_API_KEY` env variable."""
struct GEMINIServiceEndpoint <: ServiceEndpoint end

"""
    GenericOpenAIEndpoint <: ServiceEndpoint

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
struct GenericOpenAIEndpoint <: ServiceEndpoint
    base_url::String
    api_key::String
end

"""
    ServiceEndpointSpec

Type alias accepting both marker types (`OPENAIServiceEndpoint`) and instances
(`GenericOpenAIEndpoint(...)`). Used as the type of `service` fields.
"""
const ServiceEndpointSpec = Union{Type{<:ServiceEndpoint}, ServiceEndpoint}

"""
    OllamaEndpoint(; base_url="http://localhost:11434") -> GenericOpenAIEndpoint

Pre-configured endpoint for [Ollama](https://ollama.com) local server.
"""
OllamaEndpoint(; base_url::String="http://localhost:11434") = GenericOpenAIEndpoint(base_url, "")

"""
    MistralEndpoint(; api_key=ENV["MISTRAL_API_KEY"]) -> GenericOpenAIEndpoint

Pre-configured endpoint for [Mistral AI](https://mistral.ai) API.
"""
MistralEndpoint(; api_key::String=ENV["MISTRAL_API_KEY"]) = GenericOpenAIEndpoint("https://api.mistral.ai", api_key)

"""
    DeepSeekEndpoint <: ServiceEndpoint

Pre-configured endpoint for [DeepSeek](https://deepseek.com) API. Supports chat completions,
tool calling, FIM completion, and prefix completion.

FIM and prefix completion use the beta base URL (`https://api.deepseek.com/beta`).
"""
struct DeepSeekEndpoint <: ServiceEndpoint
    api_key::String
end
DeepSeekEndpoint(; api_key::String=ENV["DEEPSEEK_API_KEY"]) = DeepSeekEndpoint(api_key)


"""
    chat = Chat()

Creates a new `Chat` object with default settings:
- `model` is set to `gpt-5.2`
- `messages` is set to an empty `Vector{Message}`
- `history` is set to `true`
"""
@kwdef struct Chat
    service::ServiceEndpointSpec = OPENAIServiceEndpoint
    model::String = "gpt-5.2"
    messages::Conversation = Message[]
    history::Bool = true
    tools::Union{Vector{GPTTool},Nothing} = nothing
    tool_choice::Union{String,GPTToolChoice,Nothing} = nothing # "auto" | "none" |
    parallel_tool_calls::Union{Bool,Nothing} = false
    temperature::Union{Float64,Nothing} = nothing # 0.0 - 2.0 - mutual exclusive with top_p
    top_p::Union{Float64,Nothing} = nothing # 0.0 - 1.0 - mutual exclusive with temperature
    n::Union{Int64,Nothing} = nothing # 1 - 10
    stream::Union{Bool,Nothing} = nothing
    stop::Union{Vector{String},String,Nothing} = nothing # max 4 sequences
    max_tokens::Union{Int64,Nothing} = nothing
    presence_penalty::Union{Float64,Nothing} = nothing # -2.0 - 2.0
    response_format::Union{ResponseFormat,Nothing} = nothing
    frequency_penalty::Union{Float64,Nothing} = nothing # -2.0 - 2.0
    logit_bias::Union{AbstractDict{String,Float64},Nothing} = nothing
    user::Union{String,Nothing} = nothing
    seed::Union{Int64,Nothing} = nothing
    _cumulative_cost::Ref{Float64} = Ref(0.0)
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
        presence_penalty,
        response_format,
        frequency_penalty,
        logit_bias,
        user,
        seed,
        _cumulative_cost
    )
        !isnothing(temperature) && !isnothing(top_p) && throw(ArgumentError("temperature and top_p are mutually exclusive"))
        !isnothing(temperature) && !(0.0 <= temperature <= 2.0) && throw(ArgumentError("temperature must be in [0.0, 2.0]"))
        !isnothing(top_p) && !(0.0 <= top_p <= 1.0) && throw(ArgumentError("top_p must be in [0.0, 1.0]"))
        !isnothing(n) && !(1 <= n <= 10) && throw(ArgumentError("n must be in [1, 10]"))
        !isnothing(presence_penalty) && !(-2.0 <= presence_penalty <= 2.0) && throw(ArgumentError("presence_penalty must be in [-2.0, 2.0]"))
        !isnothing(frequency_penalty) && !(-2.0 <= frequency_penalty <= 2.0) && throw(ArgumentError("frequency_penalty must be in [-2.0, 2.0]"))
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
            presence_penalty,
            response_format,
            frequency_penalty,
            logit_bias,
            user,
            seed,
            _cumulative_cost
        )
    end
end

function JSON.lower(chat::Chat)
    d = Dict{Symbol,Any}(:model => chat.model, :messages => chat.messages)
    for f in (:tools, :tool_choice, :parallel_tool_calls, :temperature, :top_p,
        :n, :stream, :stop, :max_tokens, :presence_penalty, :response_format,
        :frequency_penalty, :logit_bias, :user, :seed)
        v = getfield(chat, f)
        !isnothing(v) && (d[f] = v)
    end
    return d
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
    TokenUsage(; prompt_tokens=0, completion_tokens=0, total_tokens=0)

Token usage statistics returned by the API.
"""
@kwdef struct TokenUsage
    prompt_tokens::Int = 0
    completion_tokens::Int = 0
    total_tokens::Int = 0
end

"""
    LLMSuccess(; message, self, usage=nothing)

Successful Chat Completions API response.

# Fields
- `message::Message`: The assistant's reply message.
- `self::Chat`: The updated [`Chat`](@ref) object (with the new message appended if `history=true`).
- `usage::Union{TokenUsage, Nothing}`: Token usage statistics from the API.
"""
@kwdef struct LLMSuccess <: LLMRequestResponse
    message::Message
    self::Chat
    usage::Union{TokenUsage, Nothing} = nothing
end

"""
    LLMFailure(; response, status, self)

HTTP-level failure from the Chat Completions API. The server returned a non-200 status.

# Fields
- `response::String`: The raw response body.
- `status::Int`: The HTTP status code.
- `self::Chat`: The [`Chat`](@ref) object (unchanged).
"""
@kwdef struct LLMFailure <: LLMRequestResponse
    response::String
    status::Int
    self::Chat
end

"""
    LLMCallError(; error, status=nothing, self)

Exception-level error during a Chat Completions API call (network failure, JSON parse error, etc.).

# Fields
- `error::String`: The stringified exception.
- `status::Union{Int,Nothing}`: HTTP status if available.
- `self::Chat`: The [`Chat`](@ref) object (unchanged).
"""
@kwdef struct LLMCallError <: LLMRequestResponse
    error::String
    status::Union{Int,Nothing} = nothing
    self::Chat
end



"""
    is_send_valid(chat::Chat)::Bool

    Check if the conversation is valid for sending to the API.

    This check employs a rough heuristic that works for practical purposes. 
            
    It checks if the conversation has at least two messages, the first message is from the system, the last message is from the user, and there are no consecutive messages from the same role. However, this is not a foolproof check and may not work in all cases (e.g. imagine that you passed another system message in the middle of the conversation).
"""
function issendvalid(chat::Chat)::Bool
    length(chat) > 1 &&
        chat.messages[begin].role == RoleSystem &&
        chat.messages[end].role == RoleUser &&
        all(chat.messages[i].role != chat.messages[i+1].role for i in 1:length(chat)-1)
end

"""
    push!(chat::Chat, msg::Message)

    Add a message to the conversation. The goal here is to make invalid conversations unrepresentable.
"""
function Base.push!(chat::Chat, msg::Message)
    inilen = length(chat)
    msg.role == RoleSystem && isempty(chat) && push!(chat.messages, msg)
    msg.role != RoleSystem && !isempty(chat) &&
        (chat.messages[end].role != msg.role || msg.role == RoleTool) &&
        push!(chat.messages, msg)
    length(chat) == inilen && @warn "Cannot add message $msg to conversation: $chat"
    return chat
end

"""
    pop!(chat::Chat)

    Remove the last message from the conversation.
"""
function Base.pop!(chat::Chat)
    inilength = length(chat)
    !isempty(chat) && pop!(chat.messages)
    length(chat) == inilength && @warn "Cannot remove last message from an empty conversation: $chat"
    return chat
end

"""
    last(chat::Chat)

    Get the last message in the conversation.
"""
Base.last(chat::Chat) = last(chat.messages)

"""
    update!(chat::Chat, msg::Message)

    Update the chat with a new message. 
"""
function update!(chat::Chat, msg::Message)
    chat.history && push!(chat, msg)
    !chat.history && @warn "Cannot update chat with your message: chat history is disabled."
    return chat
end

"""
    Base.getindex(chat::Chat, i::Int)

    Get the message at index `i` in the conversation.
"""
Base.getindex(chat::Chat, i::Int) = chat.messages[i]

"""
    Base.setindex!(chat::Chat, msg::Message, i::Int)

    Set the message at index `i` in the conversation.
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
    Embeddings(input::String)
    Embeddings(input::Vector{String})

Create an embedding request for one or more text inputs. Uses the `text-embedding-3-small`
model (1536-dimensional embeddings) by default.

The `embeddings` field is **pre-allocated** and filled in-place by [`embeddingrequest!`](@ref).

# Fields
- `model::String`: The embedding model name.
- `input::Union{String,Vector{String}}`: Text(s) to embed.
- `embeddings::Union{Vector{Float64},Vector{Vector{Float64}}}`: Pre-allocated embedding vector(s).
- `user::Union{String,Nothing}`: Optional end-user identifier.

# Example
```julia
emb = Embeddings("Julia is a great language")
embeddingrequest!(emb)
emb.embeddings  # => Float64[...] (1536 dims)
```
"""
struct Embeddings
    service::ServiceEndpointSpec
    model::String
    input::Union{String,Vector{String}}
    embeddings::Union{Vector{Float64},Vector{Vector{Float64}}}
    user::Union{String,Nothing}
    function Embeddings(input::String; service::ServiceEndpointSpec=OPENAIServiceEndpoint)
        return new(service, string(GPTTextEmbedding3Small), input, zeros(Float64, 1536), nothing)
    end
    function Embeddings(input::Vector{String}; service::ServiceEndpointSpec=OPENAIServiceEndpoint)
        isempty(input) && throw(ArgumentError("input must not be empty"))
        return new(service, string(GPTTextEmbedding3Small), input, [zeros(Float64, 1536) for _ in 1:length(input)], nothing)
    end
end

function JSON.lower(emb::Embeddings)
    d = Dict{Symbol,Any}(:model => emb.model, :input => emb.input)
    !isnothing(emb.user) && (d[:user] = emb.user)
    return d
end

function update!(emb::Embeddings, data::AbstractVector)
    if emb.input isa String
        copy!(emb.embeddings, data[1]["embedding"])
    else
        for item in data
            idx = item["index"] + 1  # API uses 0-based indexing
            copy!(emb.embeddings[idx], item["embedding"])
        end
    end
end