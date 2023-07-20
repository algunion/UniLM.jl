const GPTSystem = "system"
const GPTUser = "user"
const GPTAssistant = "assistant"
const GPTFunction = "function"

# to do: extend to all models/endpoints
struct Model
    name::String
end

Base.show(io::IO, x::Model) = print(io, x.name)
Base.parse(::Type{Model}, s::String) = Model(s)

const GPT35Turbo = Model("gpt-3.5-turbo")
const GPT4 = Model("gpt-4")


@kwdef struct Message
    role::String
    content::Union{String,Nothing} = nothing
    name::Union{String,Nothing} = nothing
    function_call::Union{Nothing,Dict{String,Any}} = nothing
    function Message(role, content, name, function_call)
        isnothing(content) && isnothing(function_call) && throw(ArgumentError("content and function_call cannot both be nothing"))
        role == GPTFunction && isnothing(name) && throw(ArgumentError("name cannot be empty when role is GPTFunction"))
        return new(role, content, name, function_call)
    end
end

const Conversation = Vector{Message}

StructTypes.StructType(::Type{Message}) = StructTypes.Struct()
StructTypes.omitempties(::Type{Message}) = (:name, :function_call) # content cannot be nothing when user generated

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
    functions::Union{Vector{GPTFunctionSignature},Nothing} = nothing
    function_call::Union{String,Pair{String,String},Nothing} = nothing # "auto" | "none" | Dict("name" => "my_function")    
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
function is_send_valid(chat::Chat)::Bool
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
    InvalidConversationError("Cannot add message $msg to conversation: $chat")
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

