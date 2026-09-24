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

    # Only top-level @docs blocks count: Documenter renders none of these.
    write(joinpath(dir, "b.md"), """
    <!--
    ```@docs
    InComment
    ```
    -->
    <!-- ```@docs --> OneLineComment
    ````markdown
    ```@docs
    InLiteralFence
    ```
    ````
    ~~~
    ```@docs
    InTildeFence
    ```
    ~~~
    !!! note
        ```@docs
        InAdmonition
        ```
    > ```@docs
    > InQuote
    > ```
    ````@docs
    FourBackticks
    ````
    ```@docs; canonical=false
    InfoSuffix
    ```
    ```@docs
    AfterComments
    ```
    """)
    got = parse_documented_symbols(dir)
    @test !("InComment" in got)
    @test !("OneLineComment" in got)
    @test !("InLiteralFence" in got)
    @test !("InTildeFence" in got)
    @test !("InAdmonition" in got)
    @test !("InQuote" in got) && !("> InQuote" in got)
    @test "FourBackticks" in got   # a longer fence is still a top-level @docs block
    @test "InfoSuffix" in got      # the info string only has to start with @docs
    @test "AfterComments" in got   # comments and literal fences close; later blocks count
end
