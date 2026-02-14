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

@kwdef struct GPTToolCall
    id::String
    type::String = "function"
    func::GPTFunction
end

JSON.lower(x::GPTToolCall) = Dict(:id => x.id, :type => x.type, :function => x.func)

@kwdef struct GPTTool
    type::String = "function"
    func::GPTFunctionSignature
end

JSON.lower(x::GPTTool) = Dict(:type => x.type, :function => x.func)


@kwdef struct GPTToolChoice
    type::String = "function"
    func::Union{String,Symbol}
end

JSON.lower(x::GPTToolChoice) = Dict(:type => x.type, :function => Dict(:name => x.func))


struct GPTFunctionCallResult{T}
    name::Union{String,Symbol}
    origincall::GPTFunction
    result::T
end

JSON.omit_null(::Type{<:GPTFunctionCallResult}) = true
JSON.omit_empty(::Type{<:GPTFunctionCallResult}) = true

const RoleSystem = "system"
const RoleUser = "user"
const RoleAssistant = "assistant"
const RoleTool = "tool"

# to do: extend to all models/endpoints
struct Model
    name::String
end

Base.show(io::IO, x::Model) = print(io, x.name)
Base.parse(::Type{Model}, s::String) = Model(s)

const GPT4o = Model("gpt-4o")
const GPT4oMini = Model("gpt-4o-mini")
const GPT4Turbo = Model("gpt-4-1106-preview")
const GPT4TurboVision = Model("gpt-4-vision-preview")

const STOP = "stop"
const CONTENT_FILTER = "content_filter"
const TOOL_CALLS = "tool_calls"


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

#string(m::Message) = getfield(m, :content) |> string

@kwdef struct JsonSchemaAPI
    name::String
    description::String
    schema::AbstractDict
end

# JsonSchemaAPI serializes as a regular struct (JSON.jl handles this automatically)

@kwdef struct ResponseFormat
    type::String = "json_object"
    json_schema::Union{JsonSchemaAPI,AbstractDict,Nothing} = nothing
end

ResponseFormat(json_schema) = ResponseFormat("json_schema", json_schema)

JSON.omit_null(::Type{ResponseFormat}) = true

json_object() = ResponseFormat()
json_schema(schema) = ResponseFormat(schema)
json_schema(name::String, description::String, schema::AbstractDict) = ResponseFormat(JsonSchemaAPI(name, description, schema))

abstract type ServiceEndpoint end
struct OPENAIServiceEndpoint <: ServiceEndpoint end
struct AZUREServiceEndpoint <: ServiceEndpoint end
struct GEMINIServiceEndpoint <: ServiceEndpoint end


"""
    chat = Chat()

Creates a new `Chat` object with default settings:
- `model` is set to `gpt-4o`
- `messages` is set to an empty `Vector{Message}`
- `history` is set to `true`
"""
@kwdef struct Chat
    service::Type{<:ServiceEndpoint} = OPENAIServiceEndpoint #AZUREServiceEndpoint #OPENAIServiceEndpoint
    model::String = "gpt-4o"
    messages::Conversation = Message[]
    history::Bool = true
    tools::Union{Vector{GPTTool},Nothing} = nothing
    tool_choice::Union{String,GPTToolChoice,Nothing} = nothing # "auto" | "none" |
    parallel_tool_calls::Union{Bool,Nothing} = false
    temperature::Union{Float64,Nothing} = nothing # 0.0 - 2.0 - mutual exclusive with top_p
    top_p::Union{Float64,Nothing} = nothing # 1 - 100 - mutual exclusive with temperature
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
        seed
    )
        !isnothing(temperature) && !isnothing(top_p) && throw(ArgumentError("temperature and top_p are mutually exclusive"))
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
            seed
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

abstract type LLMRequestResponse end
@kwdef struct LLMSuccess <: LLMRequestResponse
    message::Message
    self::Chat
end



@kwdef struct LLMFailure <: LLMRequestResponse
    response::String
    status::Int
    self::Chat
end



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
    msg.role != RoleSystem && chat.messages[end].role != msg.role && push!(chat.messages, msg)
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

const GPTTextEmbeddingAda002 = Model("text-embedding-ada-002")

# defaulting to text-embedding-ada-002 for now
# be aware of embedding size if changing model
struct Embeddings
    model::String
    input::Union{String,Vector{String}}
    embeddings::Union{Vector{Float64},Vector{Vector{Float64}}}
    user::Union{String,Nothing}
    function Embeddings(input::String)
        return new(string(GPTTextEmbeddingAda002), input, zeros(Float64, 1536), nothing)
    end
    function Embeddings(input::Vector{String})
        isempty(input) && throw(ArgumentError("input must not be empty"))
        return new(string(GPTTextEmbeddingAda002), input, [zeros(Float64, 1536) for _ in 1:length(input)], nothing)
    end
end

function JSON.lower(emb::Embeddings)
    d = Dict{Symbol,Any}(:model => emb.model, :input => emb.input)
    !isnothing(emb.user) && (d[:user] = emb.user)
    return d
end

update!(emb::Embeddings, embeddings) = copy!(emb.embeddings, embeddings)