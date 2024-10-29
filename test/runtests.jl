if isdefined(@__MODULE__, :LanguageServer)
    include("../src/UniLM.jl")
end

using Test
using Accessors
using JET
using Pkg

using LLMPowerTools

function get_pkg_version(name::AbstractString)
    for dep in values(Pkg.dependencies())
        if dep.name == name
            return dep.version
        end
    end
    return error("Dependency not available")
end

@testset "Type Stability (JET.jl)" begin
    if VERSION >= v"1.10"
        @assert get_pkg_version("JET") >= v"0.8.4"
        JET.test_package(LLMPowerTools;
            target_defined_modules=true,
            ignore_missing_comparison=true)
    end
end

@testset "api.jl" begin
    include("api.jl")
end