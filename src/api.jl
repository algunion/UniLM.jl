@kwdef mutable struct GPTFunctionSignature
    name::String
    description::Union{String,Nothing} = nothing
    parameters::Union{AbstractDict,Nothing} = nothing
end

StructTypes.StructType(::Type{GPTFunctionSignature}) = StructTypes.Struct()
StructTypes.omitempties(::Type{GPTFunctionSignature}) = (:description, :parameters)

"""
`text` is the prompt.\n
`images` is a vector of image urls (or base64 encoded images).
"""
struct GPTImageContent
    text::String
    images::Vector{String}
end

StructTypes.StructType(::Type{GPTImageContent}) = StructTypes.CustomStruct()
function StructTypes.lower(x::GPTImageContent)
    d = [Dict(:type => "text", :text => x.text)]
    for i in x.images
        push!(d, Dict(:type => "image_url", :image_url => Dict(:url => i, :detail => "auto")))
    end
    return d
end

struct GPTFunction
    name::String
    arguments::AbstractDict{String,String}
end

StructTypes.StructType(::Type{GPTFunction}) = StructTypes.CustomStruct()
function StructTypes.lower(x::GPTFunction)
    Dict(:name => x.name, :arguments => JSON3.write(x.arguments))
end

@kwdef struct GPTToolCall
    id::String
    type::String = "function"
    func::GPTFunction
end

StructTypes.StructType(::Type{GPTToolCall}) = StructTypes.Struct()
StructTypes.names(::Type{GPTToolCall}) = ((:func, :function),)

@kwdef struct GPTTool
    type::String = "function"
    func::GPTFunctionSignature
end

StructTypes.StructType(::Type{GPTTool}) = StructTypes.Struct()
StructTypes.names(::Type{GPTTool}) = ((:func, :function),)


@kwdef struct GPTToolChoice
    type::String = "function"
    func::Union{String,Symbol}
end

StructTypes.StructType(::Type{GPTToolChoice}) = StructTypes.CustomStruct()
StructTypes.lower(x::GPTToolChoice) = Dict(:type => x.type, :function => Dict(:name => x.func))


struct GPTFunctionCallResult{T}
    name::Union{String,Symbol}
    origincall::GPTFunction
    result::T
end

StructTypes.StructType(::Type{GPTFunctionCallResult}) = StructTypes.Struct()
StructTypes.omitempties(::Type{GPTFunctionCallResult}) = (:name,)

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
        isnothing(content) && isnothing(tool_calls) && throw(ArgumentError("`content` and `tool_calls` cannot both be nothing"))
        role == RoleTool && isnothing(tool_call_id) && throw(ArgumentError("`tool_call_id` cannot be empty when role is `tool`"))
        return new(role, content, name, finish_reason, refusal_message, tool_calls, tool_call_id)
    end
end

const Conversation = Vector{Message}

StructTypes.StructType(::Type{Message}) = StructTypes.Struct()
StructTypes.omitempties(::Type{Message}) = (:name, :tool_calls, :tool_call_id) # content cannot be nothing when user generated

message(m::Message) = m.content
content(m::Message) = m.content

getrole(m::Message) = m.role
iscall(m::Message) = m.role == RoleTool

#string(m::Message) = getfield(m, :content) |> string

struct JsonSchemaAPI
    name::String
    description::String
    schema::AbstractDict
end

StructTypes.StructType(::Type{JsonSchemaAPI}) = StructTypes.Struct()

struct ResponseFormat
    type::String
    json_schema::Union{JsonSchemaAPI,AbstractDict,Nothing}
    ResponseFormat() = new("json_object", nothing)
    ResponseFormat(json_schema) = new("json_schema", json_schema)
end

StructTypes.StructType(::Type{ResponseFormat}) = StructTypes.Struct()
StructTypes.omitempties(::Type{ResponseFormat}) = (:json_schema,)

json_object() = ResponseFormat()
json_schema(schema) = ResponseFormat(schema)

abstract type ServiceEndpoint end
struct OPENAIServiceEndpoint <: ServiceEndpoint end
struct AZUREServiceEndpoint <: ServiceEndpoint end


"""
    chat = Chat()

Creates a new `Chat` object with default settings:
- `model` is set to `gpt-4o`
- `messages` is set to an empty `Vector{Message}`
- `history` is set to `true`
"""
@kwdef struct Chat
    service::Type{<:ServiceEndpoint} = AZUREServiceEndpoint #AZUREServiceEndpoint #OPENAIServiceEndpoint
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

StructTypes.StructType(::Type{Chat}) = StructTypes.Struct()
StructTypes.omitempties(::Type{Chat}) = true
StructTypes.excludes(::Type{Chat}) = (:history, :service)

Base.length(chat::Chat) = length(chat.messages)
Base.isempty(chat::Chat) = isempty(chat.messages)

abstract type LLMRequestResponse end
@kwdef struct LLMSuccess <: LLMRequestResponse
    message::Message
    self::Chat
end

StructTypes.StructType(::Type{LLMSuccess}) = StructTypes.Struct()

@kwdef struct LLMFailure <: LLMRequestResponse
    response::String
    status::Int
    self::Chat
end

StructTypes.StructType(::Type{LLMFailure}) = StructTypes.Struct()

@kwdef struct LLMCallError <: LLMRequestResponse
    error::String
    status::Union{Int,Nothing} = nothing
    self::Chat
end

StructTypes.StructType(::Type{LLMCallError}) = StructTypes.Struct()

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
        all([v.role != chat.messages[i+1].role for (i, v) in collect(enumerate(chat.messages))[1:end-1]])
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
    function Embeddings(input)
        if isa(input, String)
            return new(GPTTextEmbeddingAda002 |> string, input, zeros(Float64, 1536), nothing)
        elseif isa(input, Vector{String})
            return new(GPTTextEmbeddingAda002 |> string, input, zeros(Float64, 1536, length(input)), nothing)
        end
    end
end

StructTypes.StructType(::Type{Embeddings}) = StructTypes.Struct()
StructTypes.omitempties(::Type{Embeddings}) = (:user,)
StructTypes.excludes(::Type{Embeddings}) = (:embeddings,)

update!(emb::Embeddings, embeddings) = copy!(emb.embeddings, embeddings)