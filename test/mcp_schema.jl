# Tests for mcp_schema.jl — JSON Schema generation from Julia types

@testset "Primitive types" begin
    @test UniLM._json_schema_type(String) == Dict("type" => "string")
    @test UniLM._json_schema_type(Bool) == Dict("type" => "boolean")
    @test UniLM._json_schema_type(Nothing) == Dict("type" => "null")
    @test UniLM._json_schema_type(Any) == Dict{String,Any}()
    @test UniLM._json_schema_type(Int) == Dict("type" => "integer")
    @test UniLM._json_schema_type(Int64) == Dict("type" => "integer")
    @test UniLM._json_schema_type(Int32) == Dict("type" => "integer")
    @test UniLM._json_schema_type(Float64) == Dict("type" => "number")
    @test UniLM._json_schema_type(Float32) == Dict("type" => "number")
    @test UniLM._json_schema_type(Number) == Dict("type" => "number")
end

@testset "Container types" begin
    schema = UniLM._json_schema_type(Vector{String})
    @test schema["type"] == "array"
    @test schema["items"] == Dict("type" => "string")

    schema = UniLM._json_schema_type(Vector{Int})
    @test schema["type"] == "array"
    @test schema["items"] == Dict("type" => "integer")

    schema = UniLM._json_schema_type(Vector)
    @test schema == Dict("type" => "array")

    schema = UniLM._json_schema_type(Dict{String,Float64})
    @test schema["type"] == "object"
    @test schema["additionalProperties"] == Dict("type" => "number")

    schema = UniLM._json_schema_type(Dict{String,Any})
    @test schema["type"] == "object"

    # The AbstractDict fallback in mcp_schema.jl (non-String-keyed dicts): a bare object schema
    # with NO "additionalProperties" key. The file/module pin catches a stray definition
    # elsewhere or a test-local override; the method-identity check falsifies dispatch leaking
    # to the Dict{String,T} branch, which lives in the same file and so shares both.
    m = which(UniLM._json_schema_type, (Type{Dict{Symbol,Int}},))
    @test basename(String(m.file)) == "mcp_schema.jl"
    @test m.module === UniLM
    @test m !== which(UniLM._json_schema_type, (Type{Dict{String,Int}},))
    sym_schema = UniLM._json_schema_type(Dict{Symbol,Int})
    @test sym_schema == Dict{String,Any}("type" => "object")
    @test !haskey(sym_schema, "additionalProperties")
    # an Int-keyed dict lands on the same fallback
    @test UniLM._json_schema_type(Dict{Int,String}) == Dict{String,Any}("type" => "object")
end

@testset "_is_optional" begin
    opt, T = UniLM._is_optional(Union{String,Nothing})
    @test opt == true
    @test T == String

    opt, T = UniLM._is_optional(Union{Nothing,Int})
    @test opt == true
    @test T == Int

    opt, T = UniLM._is_optional(String)
    @test opt == false
    @test T == String

    opt, T = UniLM._is_optional(Nothing)
    @test opt == true
end

@testset "_is_optional — both 2-arg Union orderings + baselines" begin
    # Julia 1.12 normalizes 2-arg `Union{T, Nothing}` deterministically: Nothing lands in
    # `.a`, T in `.b`, REGARDLESS of source order. Empirically (verified here):
    #   Union{Int,Nothing}.a === Nothing  and  Union{Nothing,Int}.a === Nothing
    # So both spellings exercise src/mcp_schema.jl:53 (`a === Nothing && return (true, b)`);
    # the doubled values below assert the SAME (true, Int) result either way, which would
    # break if line 53's returned element (`b`) were swapped for `a`.
    @test UniLM._is_optional(Union{Int,Nothing}) == (true, Int)
    @test UniLM._is_optional(Union{Nothing,Int}) == (true, Int)
    # Pin the empirical ordering this conclusion rests on (line 54, the `b === Nothing`
    # branch, is therefore unreachable for any 2-arg Union{T,Nothing} on this Julia).
    @test Union{Int,Nothing}.a === Nothing
    @test Union{Nothing,Int}.a === Nothing
    @test Union{Int,Nothing}.b === Int
    # Non-optional baseline: a plain concrete type is required and returned unchanged.
    @test UniLM._is_optional(Int) == (false, Int)
    # Bare Nothing is optional with element Nothing (the T === Nothing fast-path, line 50).
    @test UniLM._is_optional(Nothing) == (true, Nothing)
end

@testset "_function_schema" begin
    f1(a::String, b::Int) = a * string(b)
    schema = UniLM._function_schema(f1)
    @test schema["type"] == "object"
    @test haskey(schema, "properties")
    @test haskey(schema["properties"], "a")
    @test haskey(schema["properties"], "b")
    @test schema["properties"]["a"] == Dict("type" => "string")
    @test schema["properties"]["b"] == Dict("type" => "integer")
    @test Set(schema["required"]) == Set(["a", "b"])

    # Function with optional arg (Union{T,Nothing})
    f2(x::Float64, y::Union{String,Nothing}) = string(x)
    schema2 = UniLM._function_schema(f2)
    @test "x" in schema2["required"]
    # y is Union{String,Nothing} — should not be required
    @test !("y" in schema2["required"])
end

@testset "_function_schema never throws on introspection" begin
    # A `where` signature is a UnionAll: its type variable maps to the upper bound.
    fint(x::T) where {T<:Integer} = x
    s = UniLM._function_schema(fint)
    @test s["properties"]["x"] == Dict{String,Any}("type" => "integer")
    @test s["required"] == ["x"]
    # A type variable nested in a container keeps the container's schema.
    fvec(x::T, ys::Vector{T}) where {T<:AbstractFloat} = x
    s = UniLM._function_schema(fvec)
    @test s["properties"]["x"] == Dict{String,Any}("type" => "number")
    @test s["properties"]["ys"]["type"] == "array"
    # Unions of any arity: `Nothing` makes the parameter optional, the rest is anyOf.
    fu(a::Union{Int,String,Nothing}, b::Union{Int,String}) = a
    s = UniLM._function_schema(fu)
    @test s["required"] == ["b"]
    int_or_str = Set([Dict{String,Any}("type" => "integer"), Dict{String,Any}("type" => "string")])
    @test Set(s["properties"]["a"]["anyOf"]) == int_or_str
    @test Set(s["properties"]["b"]["anyOf"]) == int_or_str
    @test UniLM._is_optional(Union{Int,String,Nothing}) == (true, Union{Int,String})
    # Symbol travels as a string; a type with no JSON mapping accepts any value.
    fs(k::Symbol, z::Complex{Float64}) = k
    s = UniLM._function_schema(fs)
    @test s["properties"]["k"] == Dict{String,Any}("type" => "string")
    @test s["properties"]["z"] == Dict{String,Any}()
    # Ignored parameters (`_`, an unnamed `::T`) are not advertised.
    fign(_, ::Int, n::Int) = n
    @test collect(keys(UniLM._function_schema(fign)["properties"])) == ["n"]
    # The type mapping itself: unions, UnionAll containers, Symbol, unknown types.
    @test UniLM._json_schema_type(Symbol) == Dict{String,Any}("type" => "string")
    @test UniLM._json_schema_type(Vector{<:Real}) == Dict{String,Any}("type" => "array")
    @test UniLM._json_schema_type(Complex{Float64}) == Dict{String,Any}()
    @test Set(UniLM._json_schema_type(Union{Int,Nothing})["anyOf"]) ==
          Set([Dict{String,Any}("type" => "integer"), Dict{String,Any}("type" => "null")])
end
