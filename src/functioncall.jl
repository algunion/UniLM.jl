@kwdef struct GPTFunctionSignature
    name::String
    description::Union{String,Nothing} = nothing
    parameters::Union{Dict{String,Any},Nothing} = nothing
end

StructTypes.StructType(::Type{GPTFunctionSignature}) = StructTypes.Struct()
StructTypes.omitempties(::Type{GPTFunctionSignature}) = (:description, :parameters)

struct GPTFunctionCallResult{T}
    name::Union{String, Symbol}
    origincall::Dict{String, Any}
    result::T
end

StructTypes.StructType(::Type{GPTFunctionCallResult}) = StructTypes.Struct()
StructTypes.omitempties(::Type{GPTFunctionCallResult}) = (:name,)