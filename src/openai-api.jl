@kwdef mutable struct GPTFunctionSignature
    name::String
    description::Union{String,Nothing} = nothing
    parameters::Union{JsonObject,Nothing} = nothing
end

StructTypes.StructType(::Type{GPTFunctionSignature}) = StructTypes.Struct()
StructTypes.omitempties(::Type{GPTFunctionSignature}) = (:description, :parameters)

@kwdef mutable struct GPTToolCall
    id::String
    type::String = "function"
    func::Dict{String,String}
end

StructTypes.StructType(::Type{GPTToolCall}) = StructTypes.Struct()
StructTypes.names(::Type{GPTToolCall}) = ((:func, :function))

@kwdef struct GPTTool
    type::String = "function"
    func::GTPFunctionSignature
end

StructTypes.StructType(::Type{GPTTool}) = StructTypes.Struct()
StructTypes.names(::Type{GPTTool}) = ((:func, :function))


# we only care about serialization here (syntactic sugar)
@kwdef struct GPTToolChoice
    type::String = "function"
    func::String
end

StructTypes.StructType(::Type{GPTToolChoice}) = StructTypes.CustomStruct()
StructTypes.lower(x::GPTToolChoice) = Dict(:type => x.type, :function => Dict(:name => x.func))


struct GPTFunctionCallResult{T}
    name::Union{String,Symbol}
    origincall::Dict{String,Any}
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

const GPT35Turbo = Model("gpt-3.5-turbo")
const GPT4 = Model("gpt-4")
const GPT4Turbo = Model("gpt-4-1106-preview")
const GPT4TurboVision = Model("gpt-4-vision-preview")
const GPT35Latest = ""
const GPT4Latest = ""


@kwdef struct Message
    role::String
    content::Union{String,Nothing} = nothing
    name::Union{String,Nothing} = nothing
    tool_calls::Union{Nothing,Vector{GPTToolCall}} = nothing
    tool_call_id::Union{String,Nothing} = nothing
    function Message(role, content, name, tool_calls, tool_call_id)
        isnothing(content) && isnothing(tool_calls) && throw(ArgumentError("`content` and `tool_calls` cannot both be nothing"))
        role == RoleTool && isnothing(tool_call_id) && throw(ArgumentError("`tool_call_id` cannot be empty when role is `tool`"))
        return new(role, content, name, function_call)
    end
end

const Conversation = Vector{Message}

StructTypes.StructType(::Type{Message}) = StructTypes.Struct()
StructTypes.omitempties(::Type{Message}) = (:name, :tool_calls, :tool_call_id) # content cannot be nothing when user generated

message(m::Message) = m.content
content(m::Message) = m.content


"""
    chat = Chat()

Creates a new `Chat` object with default settings:
- `model` is set to `gpt-3.5-turbo`
- `messages` is set to an empty `Vector{Message}`
- `history` is set to `true`
"""
@kwdef struct Chat
    model::String = "gpt-3.5-turbo"
    messages::Conversation = Message[]
    history::Bool = true
    tools::Union{Vector{GPTTool},Nothing} = nothing
    tool_choice::Union{String,Pair{String,String},Nothing} = nothing # "auto" | "none" | Dict("name" => "my_function")    
    temperature::Union{Float64,Nothing} = nothing # 0.0 - 2.0 - mutual exclusive with top_p
    top_p::Union{Float64,Nothing} = nothing # 1 - 100 - mutual exclusive with temperature
    n::Union{Int64,Nothing} = nothing # 1 - 10
    stream::Union{Bool,Nothing} = nothing
    stop::Union{Vector{String},String,Nothing} = nothing # max 4 sequences
    max_tokens::Union{Int64,Nothing} = nothing
    presence_penalty::Union{Float64,Nothing} = nothing # -2.0 - 2.0
    frequency_penalty::Union{Float64,Nothing} = nothing # -2.0 - 2.0
    logit_bias::Union{Dict{String,Float64},Nothing} = nothing
    user::Union{String,Nothing} = nothing
    response_format::Union{Dict{String, String}, Nothing} = nothing
    seed::Union{Int64,Nothing} = nothing
    function Chat(
        model,
        messages,
        history,
        functions,
        function_call,
        temperature,
        top_p,
        n,
        stream,
        stop,
        max_tokens,
        presence_penalty,
        frequency_penalty,
        logit_bias,
        user
    )
        !isnothing(temperature) && !isnothing(top_p) && throw(ArgumentError("temperature and top_p are mutually exclusive"))
        return new(
            model,
            messages,
            history,
            functions,
            function_call,
            temperature,
            top_p,
            n,
            stream,
            stop,
            max_tokens,
            presence_penalty,
            frequency_penalty,
            logit_bias,
            user
        )
    end
end

StructTypes.StructType(::Type{Chat}) = StructTypes.Struct()
StructTypes.omitempties(::Type{Chat}) = fieldnames(Chat)
StructTypes.excludes(::Type{Chat}) = (:history,)

Base.length(chat::Chat) = length(chat.messages)
Base.isempty(chat::Chat) = isempty(chat.messages)

"""
    is_send_valid(chat::Chat)::Bool

    Check if the conversation is valid for sending to the API.
"""
function issendvalid(chat::Chat)::Bool
    length(chat) > 1 &&
        chat.messages[begin].role == GPTSystem &&
        chat.messages[end].role == GPTUser &&
        all([v.role != chat.messages[i+1].role for (i, v) in collect(enumerate(chat.messages))[1:end-1]])
end

"""
    push!(chat::Chat, msg::Message)

    Add a message to the conversation. The goal here is to make invalid conversations unrepresentable.
"""
function Base.push!(chat::Chat, msg::Message)
    msg.role == GPTSystem && isempty(chat) && return push!(chat.messages, msg)
    msg.role != GPTSystem && chat.messages[end].role != msg.role && return push!(chat.messages, msg)
    throw(InvalidConversationError("Cannot add message $msg to conversation: $chat"))
end

"""
    pop!(chat::Chat)

    Remove the last message from the conversation.
"""
function Base.pop!(chat::Chat)
    !isempty(chat) && return pop!(chat.messages)
    throw(InvalidConversationError("Cannot remove message from an empty conversation and return it."))
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
    chat
end


"""
    replacelast!(chat::Chat, msg::Message)

    Replace the last message in the conversation with a new message.
"""
function replacelast!(chat::Chat, msg::Message)
    if !isempty(chat)
        chat.messages[end] = msg
        return chat
    end
end
# _EMBEDDINGS_

const GPTTextEmbeddingAda002 = Model("text-embedding-ada-002")

# defaulting to text-embedding-ada-002 for now
# be aware of embedding size if changing model
@kwdef struct Embedding
    model::String = "text-embedding-ada-002"
    input::Union{String,Vector{String}}
    embedding::Vector{Float64} = zeros(Float64, 1536)
    user::Union{String,Nothing} = nothing
end

StructTypes.StructType(::Type{Embedding}) = StructTypes.Struct()
StructTypes.omitempties(::Type{Embedding}) = (:user,)
StructTypes.excludes(::Type{Embedding}) = (:embedding,)

update!(emb::Embedding, embedding) = copy!(emb.embedding, embedding)

