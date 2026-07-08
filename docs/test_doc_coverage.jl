include(joinpath(@__DIR__, "doc_coverage.jl"))
using Test

@testset "doc-coverage gate" begin
    exported   = Set(["Foo", "Bar", "Baz", "@mac"])
    documented = Set(["Foo", "@mac"])
    allow      = Set(["Bar"])

    @test missing_docs(exported, documented, allow) == ["Baz"]
    @test isempty(missing_docs(Set(["Foo"]), Set(["Foo"]), Set{String}()))
    @test stale_allow(exported, Set(["Bar", "Gone"])) == ["Gone"]
    @test resolved_allow(documented, Set(["Foo", "Bar"])) == ["Foo"]

    dir = mktempdir()
    write(joinpath(dir, "a.md"), """
    # Title
    ```@docs
    Foo
    UniLM.Bar
    @mac
    ```
    prose
    ```julia
    NotADocEntry
    ```
    """)
    got = parse_documented_symbols(dir)
    @test "Foo" in got
    @test "Bar" in got            # `UniLM.` prefix stripped
    @test "@mac" in got
    @test !("NotADocEntry" in got)  # plain ```julia fence ignored
end
