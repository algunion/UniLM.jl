# The writer's own tests: the Markdoc serialisation of the page model, a build of the
# fixture manual in fixture/ (every construct the site supports), and builds of pages
# holding what the site cannot show. Keyless and offline:
#   julia --startup-file=no --project=docs docs/protocol/test/runtests.jl

using Documenter, Test

include(joinpath(@__DIR__, "..", "ProtocolWriter.jl"))
using .ProtocolWriter: ProtocolMDX, WriterError
const W = ProtocolWriter

module WriterFixture
"""
    greet(name::String)

Say hello to `name`; see [`WriterFixture.Greeting`](@ref).
"""
greet(name::String) = "Hello, $name"
"""
    greet(n::Int)

Say hello `n` times.

# Returns
A vector of greetings.
"""
greet(n::Int) = fill("Hello", n)
"A greeting."
struct Greeting end
"Documenter's HTML shows it as an image."
struct Picture end
Base.show(io::IO, ::MIME"image/png", ::Picture) = write(io, UInt8[0x89, 0x50, 0x4e, 0x47])
end

const FIXTURE_PAGES = ["Home" => "index.md",
                       "Guide" => ["Everything" => "guide.md", "Escaping" => "escaping.md"],
                       "Reference" => ["API" => "api.md"]]

build(root::String, site::String; pages = []) =
    makedocs(; root, source = "src", build = mktempdir(), sitename = "Fixture", format = ProtocolMDX(site),
             remotes = nothing, doctest = false, checkdocs = :none, modules = [WriterFixture], pages,
             warnonly = [:missing_docs, :cross_references])

"The error a one-page manual stops with (`nothing` if it builds)."
function build_error(markdown::String)
    root = mktempdir()
    mkpath(joinpath(root, "src"))
    write(joinpath(root, "src", "index.md"), markdown)
    try
        build(root, joinpath(root, "site"))
        nothing
    catch err
        err
    end
end

paragraph(s) = W.Paragraph(W.Inline[W.Text(s)])

@testset "ProtocolWriter" begin
    @testset "text keeps every character literal" begin
        @test W.md(W.Inline[W.Text("a*b_c {% d %} `e` \\ | <x> [y](z) #1 & ~s~")]) ==
              raw"a\*b\_c \{\% d \%\} \`e\` \\ \| \<x\> \[y\]\(z\) \#1 \& \~s\~"
        @test W.md(W.Inline[W.Emph([W.Text("e")]), W.Text(" "), W.Strong([W.Code("s")]), W.Text(" "),
                            W.Link("/api/#x", [W.Text("l")]), W.Text(" "), W.Image("/assets/i.png", "alt")]) ==
              "*e* **`s`** [l](/api/#x) ![alt](/assets/i.png)"
        @test_throws ArgumentError W.md(W.Inline[W.Text("two\nlines")])
        @test_throws ArgumentError W.md(W.Inline[W.Image("/assets/i.png", "a [b]")])
    end

    @testset "inline code, links and attributes" begin
        @test W.codespan("x", false) == "`x`"
        @test W.codespan("a`b", false) == "``a`b``"
        @test W.codespan("`x", false) == "`` `x ``"
        @test W.codespan(" a ", false) == "`  a  `"
        @test W.codespan("x|y", true) == raw"`x\|y`"
        @test_throws ArgumentError W.codespan("", false)
        @test W.destination("/api/#UniLM.Chat", false) == "/api/#UniLM.Chat"
        @test W.destination("/api/#f-Tuple{Dict{String, Any}}", false) == "</api/#f-Tuple{Dict{String, Any}}>"
        @test W.destination("/api/#a|b", true) == raw"/api/#a\|b"
        @test W.attribute("say \"hi\" \\ now\n") == "\"say \\\"hi\\\" \\\\ now\\n\""
        @test_throws ArgumentError W.attribute("bell\a")
    end

    @testset "blocks" begin
        @test W.lines(W.Heading(2, "Vector{T}", W.Inline[W.Text("Vector{T}")])) == [raw"## Vector\{T\} {% id=\"Vector{T}\" %}"]
        @test W.lines(W.Heading(4, nothing, W.Inline[W.Text("Returns")])) == ["#### Returns"]
        @test W.fence("julia", "print(\"```\")") == ["````julia", "print(\"```\")", "````"]
        @test W.fence("text", "{% raw %}") == ["```text {% process=false %}", "{% raw %}", "```"]
        @test W.lines(W.Output("")) == ["```output", "```"]
        @test W.lines(W.Callout(W.note, nothing, W.Block[paragraph("x")])) ==
              ["{% callout type=\"note\" %}", "", "x", "", "{% /callout %}"]
        @test W.lines(W.Callout(W.tip, "Mind it", W.Block[paragraph("x")]))[1] == "{% callout type=\"tip\" title=\"Mind it\" %}"
        @test W.lines(W.Details("More", W.Block[paragraph("x")]))[1] == "{% details summary=\"More\" %}"
        @test W.lines(W.Docstring("M.f", "M.f", "Function", [W.Block[paragraph("a")], W.Block[paragraph("b")]])) ==
              ["{% docstring id=\"M.f\" name=\"M.f\" kind=\"Function\" %}", "", "a", "", "---", "", "b", "", "{% /docstring %}"]
        nested = W.List(false, true, [W.Block[paragraph("a"), W.List(true, true, [W.Block[paragraph("b")]])], W.Block[paragraph("c")]])
        @test W.lines(nested) == ["- a", "  1. b", "- c"]
        loose = W.List(false, false, [W.Block[paragraph("a")], W.Block[paragraph("b"), W.CodeBlock("julia", "x\n\ny")]])
        @test W.lines(loose) == ["- a", "", "- b", "", "  ```julia", "  x", "", "  y", "  ```"]
        @test W.lines(W.Quote(W.Block[paragraph("a"), paragraph("b")])) == ["> a", ">", "> b"]
        table = W.Table([:left, :center, :right], [[W.Inline[W.Text("a|b")], W.Inline[W.Code("c|d")], W.Inline[]],
                                                   [W.Inline[W.Text("1")], W.Inline[W.Text("2")], W.Inline[W.Text("3")]]])
        @test W.lines(table) == [raw"| a\|b | `c\|d` |  |", "| :--- | :---: | ---: |", "| 1 | 2 | 3 |"]
    end

    @testset "frontmatter and description" begin
        page(description, body) = W.Page("p.md", "/p/", "A \"title\"", description, body, String[], String[])
        @test W.markdoc(page(nothing, W.Block[paragraph("x")])) ==
              "---\ntitle: \"A \\\"title\\\"\"\nnextjs:\n  metadata:\n    title: \"A \\\"title\\\"\"\n---\n\nx\n"
        @test occursin("\n    description: \"d\"\n---\n", W.markdoc(page("d", W.Block[])))
        @test W.description(W.Block[W.Paragraph(W.Inline[W.Image("/b.svg", "badge")]), paragraph("First  text.")]) == "First text."
        long = W.description(W.Block[paragraph(repeat("word ", 60))])
        @test length(long) <= 160 && endswith(long, "word…")
        @test W.description(W.Block[W.CodeBlock("julia", "x")]) === nothing
        err = try W.markdoc(page(nothing, W.Block[W.Paragraph(W.Inline[W.Code("")])])); nothing catch e; e end
        @test err isa WriterError && err.page == "p.md"
    end

    @testset "the fixture manual" begin
        site = mktempdir()
        app = joinpath(site, "src", "app")
        mkpath(joinpath(app, "gone"))
        write(joinpath(app, "gone", "page.md"), "a page an earlier build wrote")
        write(joinpath(app, "layout.tsx"), "the site's own file")
        build(joinpath(@__DIR__, "fixture"), site; pages = FIXTURE_PAGES)

        @test !ispath(joinpath(app, "gone"))
        @test read(joinpath(app, "layout.tsx"), String) == "the site's own file"
        @test sort([relpath(joinpath(d, f), app) for (d, _, fs) in walkdir(app) for f in fs if f == "page.md"]) ==
              ["api/page.md", "escaping/page.md", "guide/page.md", "page.md"]
        @test read(joinpath(site, "public", "assets", "logo.svg"), String) ==
              read(joinpath(@__DIR__, "fixture", "src", "assets", "logo.svg"), String)
        @test read(joinpath(site, "src", "navigation.json"), String) == """
            [
              {
                "title": "Introduction",
                "links": [
                  {
                    "title": "Home",
                    "href": "/"
                  }
                ]
              },
              {
                "title": "Guide",
                "links": [
                  {
                    "title": "Everything",
                    "href": "/guide/"
                  },
                  {
                    "title": "Escaping",
                    "href": "/escaping/"
                  }
                ]
              },
              {
                "title": "Reference",
                "links": [
                  {
                    "title": "API",
                    "href": "/api/"
                  }
                ]
              }
            ]
            """
        for page in ("page.md", "guide/page.md", "api/page.md", "escaping/page.md")
            expected = read(joinpath(@__DIR__, "expected", replace(page, "/" => "_")), String)
            @test read(joinpath(app, page), String) == expected
        end
    end

    @testset "what the site cannot show stops the build, naming the page and the node" begin
        @testset "$named" for (markdown, named) in [
            "```@raw html\n<b>x</b>\n```" => "Documenter.RawNode",
            "```@eval\nimport Markdown\nMarkdown.parse(\"evaluated\")\n```" => "Documenter.EvalNode",
            "```@repl\n1 + 1\n```" => "Documenter.MultiCodeBlock",
            "```@index\n```" => "Documenter.IndexNode",
            "```@contents\n```" => "Documenter.ContentsNode",
            "```math\nx^2\n```" => "MarkdownAST.DisplayMath",
            "Inline ``x^2`` math." => "MarkdownAST.InlineMath",
            "A remark[^1].\n\n[^1]: The remark." => "MarkdownAST.FootnoteLink",
            "!!! danger\n    Careful." => "`!!! danger`",
            "```@example\nMain.WriterFixture.Picture()\n```" => "image/png",
            "```output\nx\n```" => "the language `output`",
            "[nowhere](#nowhere)" => "a link to #nowhere",
            "[Missing section](@ref)" => "an unresolved link",
            "# Second title" => "a level-1 heading",
        ]
            err = build_error("# Page\n\n" * markdown * "\n")
            @test err isa WriterError && err.page == "index.md" && occursin(named, err.what)
        end
        err = build_error("Text before the title.\n\n# Page\n")
        @test err isa WriterError && occursin("does not open with its title", err.what)
    end
end
