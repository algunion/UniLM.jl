# The precompilation workload, run as a function: what it leaves behind, and what it does
# when it has no temporary directory. (The build itself is the test that it leaves no task,
# timer or connection open: Julia does not finish precompiling a package that does.)

using Test
using UniLM

_plans_held() = (length(UniLM._nl_snapshot(UniLM._NL_PLANS)), length(UniLM._nl_snapshot(UniLM._NL_KEYED_PLANS)),
                 length(UniLM._nl_snapshot(UniLM._NL_TABLES)))

@testset "the workload replays every verb and leaves no file and no plan" begin
    parent = mktempdir()
    @test UniLM._precompile_workload(parent) === nothing
    @test isempty(readdir(parent))
    @test _plans_held() == (0, 0, 0)
end

@testset "without a temporary directory the workload skips the replayed verbs and fails nothing" begin
    root = mktempdir()
    @test UniLM._precompile_workload(joinpath(root, "missing", "parent")) === nothing
    @test !ispath(joinpath(root, "missing"))
    readonly = mktempdir()
    chmod(readonly, 0o555)
    try
        @test UniLM._precompile_workload(readonly) === nothing
        @test isempty(readdir(readonly))
    finally
        chmod(readonly, 0o755)
    end
    @test _plans_held() == (0, 0, 0)
end

@testset "the workload opens no connection pool" begin
    # In a process of its own: the suite's live tests may have opened the pool already here.
    script = """
    using UniLM
    UniLM._precompile_workload()
    print(UniLM._SYSTEM_ONE_CLIENT.value === nothing)
    """
    proj = dirname(dirname(pathof(UniLM)))
    @test read(`$(Base.julia_cmd()) --startup-file=no --project=$proj -e $script`, String) == "true"
end
