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
