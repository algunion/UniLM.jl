if isdefined(@__MODULE__,:LanguageServer)
    include("../src/UniLM.jl")
end

using UniLM
using Aqua
using JET
using JSON3
using Pkg
using Test

function get_pkg_version(name::AbstractString)
    for dep in values(Pkg.dependencies())
        if dep.name == name
            return dep.version
        end
    end
    return error("Dependency not available")
end

@testset "Type Stability (JET.jl)" begin
    if VERSION >= v"1.9"
        @assert get_pkg_version("JET") >= v"0.8.4"
        JET.test_package(UniLM; target_defined_modules=true, ignore_missing_comparison=true)
    end
end

@testset verbose = true "Code quality (Aqua.jl)" begin
    Aqua.test_all(UniLM)
end

include("helpers.jl")
include("openai-api.jl")


@testset "UniLM.jl" begin

end
