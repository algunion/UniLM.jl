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
    content::Union{String,Nothing}
    name::Union{String,Nothing} = nothing
    function_call::Union{Nothing,String} = nothing
end

StructTypes.StructType(::Type{Message}) = StructTypes.Struct()
StructTypes.omitempties(::Type{Message}) = (:name, :function_call) # content cannot be nothing when user generated

@kwdef struct GPTFunctionSignature
    name::String
    description::Union{String,Nothing} = nothing
    parameters::Union{Dict{String,Any},Nothing} = nothing
end

StructTypes.StructType(::Type{GPTFunctionSignature}) = StructTypes.Struct()
StructTypes.omitempties(::Type{GPTFunctionSignature}) = (:description, :parameters)

@kwdef struct ChatParams
    model::String = "gpt-3.5-turbo"
    messages::Vector{Message}= Message[]
    functions::Union{Vector{GPTFunctionSignature},Nothing} = nothing
    function_call::Union{String,Pair{String,String},Nothing} = nothing # "auto" | "none" | Dict("name" => "my_function")
    temperature::Union{Float64,Nothing} = 1.0 # 0.0 - 2.0 - mutual exclusive with top_p
    top_p::Union{Float64,Nothing} = nothing # 1 - 100 - mutual exclusive with temperature
    n::Union{Int64,Nothing} = nothing # 1 - 10
    stream::Union{Bool,Nothing} = nothing
    stop::Union{Vector{String},String,Nothing} = nothing # max 4 sequences
    max_tokens::Union{Int64,Nothing} = nothing
    presence_penalty::Union{Float64,Nothing} = nothing # -2.0 - 2.0
    frequency_penalty::Union{Float64,Nothing} = nothing # -2.0 - 2.0
    logit_bias::Union{Dict{String,Float64},Nothing} = nothing
    user::Union{String,Nothing} = nothing
end

StructTypes.StructType(::Type{ChatParams}) = StructTypes.Struct()
StructTypes.omitempties(::Type{ChatParams}) = fieldnames(ChatParams)


@kwdef struct Conversation
    history::Bool = false
    messages::Vector{Message} = Message[]
end

Base.length(conv::Conversation) = length(conv.messages)
Base.isempty(conv::Conversation) = isempty(conv.messages)




"""
    is_send_valid(conv::Conversation)::Bool

    Check if the conversation is valid for sending to the API.
"""
function is_send_valid(conv::Conversation)::Bool
    length(conv) > 1 &&
        conv.messages[begin].role == GPTSystem &&
        conv.messages[end].role == GPTUser &&
        all([v.role != conv.messages[i+1].role for (i, v) in collect(enumerate(conv.messages))[1:end-1]])
end

"""
    push!(conv::Conversation, msg::Message)

    Add a message to the conversation. The goal here is to make invalid conversations unrepresentable.
"""
function Base.push!(conv::Conversation, msg::Message)
    msg.role == GPTSystem && isempty(conv) && return push!(conv.messages, msg)
    msg.role != GPTSystem && conv.messages[end].role != msg.role && return push!(conv.messages, msg)
    InvalidConversationError("Cannot add message $msg to conversation: $conv")
end


