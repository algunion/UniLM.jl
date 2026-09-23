# ============================================================================
# JSON Schema Generation from Julia Types
# Converts Julia type annotations to JSON Schema dictionaries for MCP tool
# parameter definitions.
# ============================================================================

"""
    _json_schema_type(T) -> Dict{String,Any}

Convert a Julia type to its JSON Schema representation. Dispatches on type
to produce the correct schema. Used by `@mcp_tool` and `register_tool!` to
auto-generate `inputSchema` from function signatures. Never throws: a type
with no JSON mapping accepts any value.

# Supported types
- Primitives: `String`, `Symbol` (a string on the wire), `Int`, `Float64`, `Bool`, `Nothing`
- Containers: `Vector{T}`, `Dict{String,T}`, and any other `AbstractVector`/`AbstractDict`
- Unions: `anyOf` of their members (a `Union{T, Nothing}` parameter is optional
  instead, see [`_function_schema`](@ref))
- Type variables: their upper bound
- Fallback: `Any` and types with no JSON mapping → `{}`
"""
_json_schema_type(::Type{String}) = Dict{String,Any}("type" => "string")
_json_schema_type(::Type{Symbol}) = Dict{String,Any}("type" => "string")
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

# Everything the methods above do not name: unions, type variables, UnionAll containers
# (`Vector{<:Real}`), and types with no JSON mapping.
function _json_schema_type(@nospecialize(T))
    T isa TypeVar && return _json_schema_type(T.ub)
    T isa Union && return Dict{String,Any}("anyOf" => Any[_json_schema_type(S) for S in Base.uniontypes(T)])
    T isa Type && T <: AbstractVector && return Dict{String,Any}("type" => "array")
    Dict{String,Any}()
end

"""
    _is_optional(T) -> (Bool, Type)

Whether a declared type admits `nothing` — `Nothing` itself or a `Union` of any
arity containing it — and the type with `Nothing` removed. Returns `(true, T)`
if optional, `(false, T)` if required.
"""
function _is_optional(@nospecialize(T))
    T === Nothing && return (true, Nothing)
    T isa Union || return (false, T)
    members = Base.uniontypes(T)
    Nothing in members || return (false, T)
    (true, Union{filter(!=(Nothing), members)...})
end

"""A positional parameter of a handler's first method, as the schema and the by-name
binder see it: its name and its declared type (type variables replaced by their upper
bounds; a `Vararg` kept as is)."""
struct _HandlerParam
    name::String
    type::Any
end

# An ignored parameter (`_`, or an unnamed `::T`) has no name a caller could bind by.
_bound_by_name(p::_HandlerParam) = !isempty(p.name) && !startswith(p.name, '#')

"""Parameters of `f`'s first method, or `nothing` when `f` has no methods. A `where`
signature is a UnionAll: its body carries the parameters. A parameter that IS a type
variable takes its upper bound; one that merely contains one (`Vector{T}`) is re-bound
over the signature's variables."""
function _handler_params(f::Function)::Union{Nothing,Vector{_HandlerParam}}
    ms = methods(f)
    isempty(ms) && return nothing
    m = first(ms)
    params = Base.unwrap_unionall(m.sig).parameters[2:end]   # skip the function type
    [_HandlerParam(string(n), _declared_type(P, m.sig))
     for (n, P) in zip(Base.method_argnames(m)[2:end], params)]
end
_declared_type(P::TypeVar, sig) = _declared_type(P.ub, sig)
_declared_type(@nospecialize(P), sig) = Base.isvarargtype(P) ? P : Base.rewrap_unionall(P, sig)

"""
    _function_schema(f::Function) -> Dict{String,Any}

Generate a JSON Schema `inputSchema` from a function's first method signature:
an `object` schema with one property per named positional parameter, `required`
listing those whose type does not admit `nothing`. Never throws; a function with
no methods gets `{"type": "object", "properties": {}}`.
"""
function _function_schema(f::Function)::Dict{String,Any}
    properties = Dict{String,Any}()
    required = String[]
    for p in something(_handler_params(f), _HandlerParam[])
        _bound_by_name(p) || continue
        optional, inner = _is_optional(p.type)
        properties[p.name] = _json_schema_type(inner)
        optional || push!(required, p.name)
    end
    schema = Dict{String,Any}("type" => "object", "properties" => properties)
    isempty(required) || (schema["required"] = required)
    schema
end
