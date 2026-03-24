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

@testset failfast=true "UniLM.jl" begin
    @testset "api.jl" begin
        include("api.jl")
    end

    @testset "requests.jl" begin
        include("requests.jl")
    end

    @testset "responses.jl" begin
        include("responses.jl")
    end

    @testset "images.jl" begin
        include("images.jl")
    end

    @testset "fork.jl" begin
        include("fork.jl")
    end

    @testset "accounting.jl" begin
        include("accounting.jl")
    end

    @testset "tool_loop" begin
        include("tool_loop.jl")
    end

    @testset "mcp_schema" begin
        include("mcp_schema.jl")
    end

    @testset "mcp_client" begin
        include("mcp_client.jl")
    end

    @testset "mcp_server" begin
        include("mcp_server.jl")
    end

    @testset "completions" begin
        include("completions.jl")
    end

    @testset "capabilities" begin
        include("capabilities.jl")
    end

    @testset "mock server" begin
        include("mock_server.jl")
    end

    @testset "integration" begin
        include("integration.jl")
    end

    @testset "integration — deepseek" begin
        include("integration_deepseek.jl")
    end

    # ── Slow tests run last ──────────────────────────────────────────────

    @testset "Aqua.jl quality checks" begin
        Aqua.test_all(UniLM; ambiguities=false)
    end

    @testset "Type Stability (JET.jl)" begin
        if VERSION >= v"1.12" && get(ENV, "UNILM_RUN_JET", "false") == "true"
            @assert get_pkg_version("JET") >= v"0.11"
            JET.test_package(UniLM;
                target_modules=(UniLM,),
                ignore_missing_comparison=true)
        else
            @info "JET.jl checks skipped (set UNILM_RUN_JET=true to enable)"
        end
    end

    @testset "integration — image generation" begin
        include("integration_images.jl")
    end
end