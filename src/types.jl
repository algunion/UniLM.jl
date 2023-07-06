const GPTSystem = "system"
const GPTUser = "user"
const GPTAssistant = "assistant"
const GPTFunction = "function"

@kwdef struct GPTFunctionSignature
    name::String
    description::Union{String, Nothing} = nothing
    parameters::Union{Dict{String, Any}, Nothing} = nothing
end

StructTypes.StructType(::Type{GPTFunctionSignature}) = StructTypes.Struct()
StructTypes.omitempties(::Type{GPTFunctionSignature}) = (:description, :parameters)

@kwdef struct ChatParams
    model::String="gpt-3.5-turbo"
    functions::Union{Vector{GPTFunctionSignature}, Nothing} = nothing
    function_call::Union{String, Dict{String, String}} = "auto" # "none" | Dict("name" => "my_function")
    temperature::Union{Float64, Nothing} = 1.0 # 0.0 - 2.0
    top_p::Union{Float64, Nothing} = nothing # 1 - 100
    n::Union{Int64, Nothing} = nothing # 1 - 10
    stream::Union{Bool,Nothing} = nothing
    stop::Union{Vector{String}, String, Nothing} = nothing # max 4 sequences
    max_tokens::Union{Int64, Nothing} = nothing
    presence_penalty::Union{Float64, Nothing} = nothing # -2.0 - 2.0
    frequency_penalty::Union{Float64, Nothing} = nothing # -2.0 - 2.0
    logit_bias::Union{Dict{String, Float64}, Nothing} = nothing
    user::Union{String, Nothing} = nothing
end

StructTypes.StructType(::Type{ChatParams}) = StructTypes.Struct()
StructTypes.omitempties(::Type{ChatParams}) = fieldnames(ChatParams)


struct Model
    name::String
end

Base.show(io::IO, x::Model) = print(io, x.name)
Base.parse(::Type{Model}, s::String) = Model(s)

const GPT35Turbo = Model("gpt-3.5-turbo")
const GPT4 = Model("gpt-4")

@kwdef struct Message
    role::String
    content::Union{String, Nothing}
    name::Union{String, Nothing} = nothing
    function_call::Union{Nothing,String} = nothing
end

StructTypes.StructType(::Type{Message}) = StructTypes.Struct()
StructTypes.omitempties(::Type{Message}) = (:name, :function_call) # content cannot be nothing when user generated

