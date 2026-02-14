if isdefined(@__MODULE__, :LanguageServer)
    include("../src/UniLM.jl")
end

using Test
using JET
using Pkg
using Aqua
using HTTP
using JSON

using UniLM

function get_pkg_version(name::AbstractString)
    for dep in values(Pkg.dependencies())
        if dep.name == name
            return dep.version
        end
    end
    return error("Dependency not available")
end

@testset "UniLM.jl" begin
    @testset "Aqua.jl quality checks" begin
        Aqua.test_all(UniLM; ambiguities=false)
    end

    @testset "Type Stability (JET.jl)" begin
        if VERSION >= v"1.12"
            @assert get_pkg_version("JET") >= v"0.11"
            JET.test_package(UniLM;
                target_modules=(UniLM,),
                ignore_missing_comparison=true)
        end
    end

    @testset "api.jl" begin
        include("api.jl")
    end

    @testset "requests.jl" begin
        include("requests.jl")
    end

    @testset "integration" begin
        include("integration.jl")
    end
end