abstract type JsonSchema end
@kwdef struct JsonString <: JsonSchema
    type::String = "string"
    description::Union{String,Nothing} = nothing
    enum::Union{Vector{String},Nothing} = nothing
    maxLength::Union{Int,Nothing} = nothing
    minLength::Union{Int,Nothing} = nothing
    pattern::Union{String,Nothing} = nothing
    format::Union{String,Nothing} = nothing
end

StructTypes.StructType(::Type{JsonString}) = StructTypes.Struct()
StructTypes.omitempties(::Type{JsonString}) = (:description, :enum, :maxLength, :minLength, :pattern, :format)

@kwdef struct JsonNumber <: JsonSchema
    type::String = "number"
    description::Union{String,Nothing} = nothing
    enum::Union{Vector{Float64},Nothing} = nothing
    maximum::Union{Float64,Nothing} = nothing
    minimum::Union{Float64,Nothing} = nothing
    exclusiveMaximum::Union{Bool,Nothing} = nothing
    exclusiveMinimum::Union{Bool,Nothing} = nothing
    multipleOf::Union{Float64,Nothing} = nothing
end

StructTypes.StructType(::Type{JsonNumber}) = StructTypes.Struct()
StructTypes.omitempties(::Type{JsonNumber}) = (:description, :enum, :maximum, :minimum, :exclusiveMaximum, :exclusiveMinimum, :multipleOf)

@kwdef struct JsonInteger <: JsonSchema
    type::String = "integer"
    description::Union{String,Nothing} = nothing
    enum::Union{Vector{Int},Nothing} = nothing
    maximum::Union{Int,Nothing} = nothing
    minimum::Union{Int,Nothing} = nothing
    exclusiveMaximum::Union{Bool,Nothing} = nothing
    exclusiveMinimum::Union{Bool,Nothing} = nothing
    multipleOf::Union{Int,Nothing} = nothing
end

StructTypes.StructType(::Type{JsonInteger}) = StructTypes.Struct()
StructTypes.omitempties(::Type{JsonInteger}) = (:description, :enum, :maximum, :minimum, :exclusiveMaximum, :exclusiveMinimum, :multipleOf)

@kwdef struct JsonBoolean <: JsonSchema
    type::String = "boolean"
    description::Union{String,Nothing} = nothing
end

StructTypes.StructType(::Type{JsonBoolean}) = StructTypes.Struct()
StructTypes.omitempties(::Type{JsonBoolean}) = (:description,)

@kwdef struct JsonNull <: JsonSchema
    type::String = "null"
    description::Union{String,Nothing} = nothing
end

StructTypes.StructType(::Type{JsonNull}) = StructTypes.Struct()
StructTypes.omitempties(::Type{JsonNull}) = (:description,)

@kwdef struct JsonArray <: JsonSchema
    type::String = "array"
    description::Union{String,Nothing} = nothing
    items::Union{JsonSchema,Nothing} = nothing
    maxItems::Union{Int,Nothing} = nothing
    minItems::Union{Int,Nothing} = nothing
    uniqueItems::Union{Bool,Nothing} = nothing
    prefixItems::Union{Vector{JsonSchema},Nothing} = nothing
    additionalItems::Union{JsonSchema,Nothing} = nothing
end

StructTypes.StructType(::Type{JsonArray}) = StructTypes.Struct()
StructTypes.omitempties(::Type{JsonArray}) = (:description, :items, :maxItems, :minItems, :uniqueItems, :prefixItems, :additionalItems)

@kwdef struct JsonAny <: JsonSchema
    description::Union{String,Nothing} = nothing
end

StructTypes.StructType(::Type{JsonAny}) = StructTypes.Struct()
StructTypes.omitempties(::Type{JsonAny}) = (:description,)

@kwdef struct JsonObject <: JsonSchema
    type::String = "object"
    description::Union{String,Nothing} = nothing
    properties::Dict{String,JsonSchema} = Dict()
    required::Vector{String} = String[]
end

StructTypes.StructType(::Type{JsonObject}) = StructTypes.Struct()
StructTypes.omitempties(::Type{JsonObject}) = (:description, :properties, :required)

function withdescription(schema::JsonSchema, description::String)::JsonSchema
    @set schema.description = description
end
