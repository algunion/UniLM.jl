# julia --startup-file=no --project=docs docs/test_site_checks.jl
include(joinpath(@__DIR__, "site_checks.jl"))
using .SiteChecks, Test
using .SiteChecks: github_slug, github_ids

"A built site in a new temp dir: an `index.html` under each route of `pages` (`\"\"` is the home page)."
function fixture_out(; pages::Vector{String})::String
    out = mktempdir()
    for page in pages
        mkpath(joinpath(out, page))
        write(joinpath(out, page, "index.html"), "<!DOCTYPE html>")
    end
    out
end

"`text` in a new temp dir as the file `name`."
function fixture_file(text::String; name::String = "README.md")::String
    path = joinpath(mktempdir(), name)
    write(path, text)
    path
end

"The site's app directory in a new temp dir, with `text` as the page.md of `route`."
function fixture_app(text::String; route::String = "guide/x")::String
    app = mktempdir()
    mkpath(joinpath(app, route))
    write(joinpath(app, route, "page.md"), text)
    app
end

@testset "site checks" begin

@testset "site checks — a docs URL must name a built page and anchor" begin
    out = fixture_out(pages = ["providers/anthropic/"])
    readme = fixture_file("See https://algunion.github.io/UniLM.jl/dev/providers/anthropic/#anthropic_start, " *
                          "https://algunion.github.io/UniLM.jl/stable/gone/ and https://algunion.github.io/UniLM.jl/dev/providers/anthropic")
    probs = check_links(out, [readme], Dict("anthropic_start" => ["/providers/anthropic/"]))
    @test length(probs) == 2 && any(p -> occursin("stable/gone/ (no page)", p), probs) && any(p -> occursin("trailing slash", p), probs)
end
@testset "site checks — failure text in an example output is reported, prose is not" begin
    app = fixture_app("```output\nRequest failed — see result\nok ReplayMissError(x)\n```\n\nRequest failed in prose.\n")
    @test length(scan_outputs(app)) == 2
end

# Expected ids rendered by GitHub, as recorded by the github-slugger project's fixtures.
@testset "site checks — a heading's id is GitHub's slug; a repeated one gets -1, -2, …" begin
    @test github_slug("Tool / Function Calling") == "tool--function-calling"
    @test github_slug("heading with an _ underscore") == "heading-with-an-_-underscore"
    @test github_slug("exchange.bind_headers(exchange, routing [, bindCallback])") ==
          "exchangebind_headersexchange-routing--bindcallback"
    @test github_slug("apostrophe’s should be trimmed") == "apostrophes-should-be-trimmed"
    @test github_slug("I ♥ unicode") == "i--unicode"
    @test github_slug("😄_😄 unicode emoji") == "_-unicode-emoji"
    @test github_slug("Привет non-latin 你好") == "привет-non-latin-你好"
    @test github_slug("Ⓘ x² ½") == "ⓘ-x-"  # a letter-like symbol stays, other numbers go
    @test github_ids(["echo", "echo", "echo 1", "echo-1", "echo"]) ==
          Set(["echo", "echo-1", "echo-1-1", "echo-1-2", "echo-2"])
    @test github_ids(["foxtrot-1", "foxtrot", "foxtrot"]) == Set(["foxtrot-1", "foxtrot", "foxtrot-2"])
end
@testset "site checks — README's links to files and headings, llms.txt's URLs, the hero's hrefs" begin
    out = fixture_out(pages = ["", "guide/x/"])
    anchors = Dict("on_home" => ["/"], "café" => ["/guide/x/"])
    readme = fixture_file("""
        # UniLM.jl
        ## Setup
        ```julia
        # Not a heading
        [a link in code](missing.md)
        ```
        ## Setup
        [a](#setup-1), [b](#setup-2), [c](#not-a-heading), [d](#unilmjl), [e](CHANGELOG.md#v1), [f](gone.md), [g](https://example.com/gone.md), [i](#100%)
        https://algunion.github.io/UniLM.jl/dev/ https://algunion.github.io/UniLM.jl/stable https://algunion.github.io/UniLM.jl/dev/guide/x/#on_home.
        [h](https://algunion.github.io/UniLM.jl/dev/guide/x/#caf%C3%A9) <https://algunion.github.io/UniLM.jl/dev/guide/x/#%C3>
        """)
    write(joinpath(dirname(readme), "CHANGELOG.md"), "")
    @test check_links(out, [readme], anchors) == [
        "$readme:8: #setup-2 (no heading)", "$readme:8: #not-a-heading (no heading)", "$readme:8: gone.md (no file)",
        "$readme:8: #100% (does not decode)",
        "$readme:9: https://algunion.github.io/UniLM.jl/stable (no trailing slash)",
        "$readme:9: https://algunion.github.io/UniLM.jl/dev/guide/x/#on_home (no anchor \"on_home\" on /guide/x/)",
        "$readme:10: https://algunion.github.io/UniLM.jl/dev/guide/x/#%C3 (does not decode)"]
    llms = fixture_file("- [Gone](https://algunion.github.io/UniLM.jl/dev/gone/): [a file](gone.md)\n";
                        name = "llms.txt")
    @test check_links(out, [llms], anchors) == ["$llms:1: https://algunion.github.io/UniLM.jl/dev/gone/ (no page)"]
    hero = fixture_file("""
        <Button href="/guide/x/">Start</Button>
        <Button href="/gone/">Gone</Button> <a href="/guide/x">x</a>
        <a href="/#on_home">home</a> <a href="https://example.com/">out</a>
        """; name = "Hero.tsx")
    # the anchors as Redirects.read_anchors returns them, routes normalised
    @test check_links(out, [hero], Dict("on_home" => [""])) ==
          ["$hero:2: /gone/ (no page)", "$hero:2: /guide/x (no trailing slash)"]
    bad = fixture_file("https://algunion.github.io/UniLM.jl/dev/a//b/\n")
    @test_throws ErrorException "$bad:1" check_links(out, [bad], anchors)
    @test_throws ErrorException "notes.html" check_links(out, [fixture_file("x"; name = "notes.html")], anchors)
end
@testset "site checks — only outputs are scanned, from the fence's column; a longer fence holds a shorter one" begin
    app = fixture_app("""
        ```julia
        CallError("not an output")
        ```

        ````output
        a reply that quotes Request failed
        ```
        KeyError: key "x" not found
        ````
        CallError("after the fence")

        - a step

          ```output
          Request failed — in a list
          ```

        > ```output
        > CallError("quoted")
        > ```

        a ReplayMissError in prose
        """; route = "guide/deep")
    write(joinpath(app, "notes.md"), "```output\nRequest failed\n```\n")
    page = joinpath(app, "guide", "deep", "page.md")
    @test scan_outputs(app) == ["$page:8: KeyError: key \"x\" not found", "$page:15: Request failed — in a list",
                                "$page:19: CallError(\"quoted\")"]
    empty = mktempdir()
    @test_throws ErrorException empty scan_outputs(empty)
end
@testset "site checks — the CLI checks README.md, llms.txt, Hero.tsx and the outputs, and exits 1 on a problem" begin
    out = fixture_out(pages = String[])
    write(joinpath(out, "anchors.json"), "{}")
    app = fixture_app("```output\nRequest failed\n```\n")
    script = joinpath(@__DIR__, "site_checks.jl")
    err = IOBuffer()
    cmd = `$(Base.julia_cmd()) --startup-file=no --project=$(@__DIR__) $script $out $app`
    code = run(pipeline(ignorestatus(cmd); stdout = devnull, stderr = err)).exitcode
    msg = String(take!(err))
    @test code == 1
    @test all(s -> occursin(s, msg), ["README.md:", "llms.txt:", "Hero.tsx:", "page.md:2: Request failed"])
end

end
