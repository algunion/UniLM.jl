# ============================================================================
# JSON Schema Generation from Julia Types
# Converts Julia type annotations to JSON Schema dictionaries for MCP tool
# parameter definitions.
# ============================================================================

"""
    _json_schema_type(::Type{T}) -> Dict{String,Any}

Convert a Julia type to its JSON Schema representation. Dispatches on type
to produce the correct schema. Used by `@mcp_tool` and `register_tool!` to
auto-generate `inputSchema` from function signatures.

# Supported types
- Primitives: `String`, `Int`, `Float64`, `Bool`, `Nothing`
- Containers: `Vector{T}`, `Dict{String,T}`
- Optionals: `Union{T, Nothing}` (makes field non-required)
- Fallback: `Any` → `{}`
"""
_json_schema_type(::Type{String}) = Dict{String,Any}("type" => "string")
_json_schema_type(::Type{Bool}) = Dict{String,Any}("type" => "boolean")
_json_schema_type(::Type{Nothing}) = Dict{String,Any}("type" => "null")
_json_schema_type(::Type{Any}) = Dict{String,Any}()

# All integer subtypes
_json_schema_type(::Type{T}) where {T<:Integer} = Dict{String,Any}("type" => "integer")

# All float subtypes
_json_schema_type(::Type{T}) where {T<:AbstractFloat} = Dict{String,Any}("type" => "number")

# Number supertype
_json_schema_type(::Type{Number}) = Dict{String,Any}("type" => "number")

# Arrays
_json_schema_type(::Type{Vector{T}}) where {T} = Dict{String,Any}("type" => "array", "items" => _json_schema_type(T))
_json_schema_type(::Type{Vector}) = Dict{String,Any}("type" => "array")

# Dicts with string keys
_json_schema_type(::Type{Dict{String,T}}) where {T} = Dict{String,Any}(
    "type" => "object", "additionalProperties" => _json_schema_type(T))
_json_schema_type(::Type{<:AbstractDict}) = Dict{String,Any}("type" => "object")

"""
    _is_optional(::Type{T}) -> (Bool, Type)

Check if a type is `Union{T, Nothing}`. Returns `(true, T)` if optional,
`(false, T)` if required.
"""
function _is_optional(::Type{T}) where {T}
    T === Nothing && return (true, Nothing)
    if T isa Union
        a, b = T.a, T.b
        a === Nothing && return (true, b)
        b === Nothing && return (true, a)
    end
    (false, T)
end

"""
    _function_schema(f::Function) -> Dict{String,Any}

Generate a JSON Schema `inputSchema` from a function's first method signature.
Extracts parameter names and types, producing an `object` schema with `properties`
and `required` arrays.

Returns a minimal `{"type": "object"}` if introspection fails.
"""
function _function_schema(f::Function)::Dict{String,Any}
    meths = methods(f)
    isempty(meths) && return Dict{String,Any}("type" => "object", "properties" => Dict{String,Any}())
    m = first(meths)
    sig = m.sig
    # sig is Tuple{typeof(f), arg1_type, arg2_type, ...}
    param_types = sig.parameters[2:end]  # skip the function type
    # Get parameter names from the method's slot names
    slot_names = Base.method_argnames(m)[2:end]  # skip the function name
    properties = Dict{String,Any}()
    required = String[]
    for (name, T) in zip(slot_names, param_types)
        name_str = string(name)
        name_str == "_" && continue  # skip unused params
        optional, inner = _is_optional(T)
        properties[name_str] = _json_schema_type(inner)
        !optional && push!(required, name_str)
    end
    schema = Dict{String,Any}("type" => "object", "properties" => properties)
    !isempty(required) && (schema["required"] = required)
    schema
end
