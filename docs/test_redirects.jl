# julia --startup-file=no --project=docs docs/test_redirects.jl
# A stub's script is run by Node.js (`node` on PATH), which the site's build needs anyway.
include(joinpath(@__DIR__, "redirects.jl"))
using .Redirects, JSON, TOML, Test

"A built site in a new temp dir: an `index.html` under each route of `pages`, and `anchors.json` holding `anchors`."
function fixture_out(; pages::Vector{String}, anchors::Dict{String,Vector{String}})::String
    out = mktempdir()
    for page in pages
        mkpath(joinpath(out, page))
        write(joinpath(out, page, "index.html"), "<!DOCTYPE html>")
    end
    write(joinpath(out, "anchors.json"), JSON.json(anchors))
    out
end

"A `redirects.toml` in a new temp file holding `routes` and `anchors`."
function fixture_toml(; routes::Dict{String,String} = Dict{String,String}(),
                        anchors::Dict{String,String} = Dict{String,String}())::String
    path = tempname()
    open(io -> TOML.print(io, Dict("routes" => routes, "anchors" => anchors)), path, "w")
    path
end

# Runs the script read from stdin once per argument, as `location.hash`; prints where `location.replace` sent it.
const RUN_SCRIPT = """
const vm = require("node:vm"), script = require("node:fs").readFileSync(0, "utf8");
for (const hash of process.argv.slice(1)) {
  let url;
  vm.runInNewContext(script, { location: { hash, replace: (u) => { url = u; } } });
  console.log(url);
}
"""

"""
Where the stub `html` sends a reader arriving with each of `hashes`: its script, cut where an HTML parser ends it,
run by Node.js.
"""
function follow(html::String, hashes::Vector{String})::Vector{String}
    script = match(r"<script>(.*?)</script>"s, html)[1]
    readlines(IOBuffer(read(pipeline(`node -e $RUN_SCRIPT -- $hashes`; stdin = IOBuffer(script)))))
end

@testset "redirects" begin

@testset "redirects — a stub forwards to its target and keeps a known hash" begin
    out = fixture_out(pages = ["providers/openai/responses/"], anchors = Dict("responses_guide" => ["/providers/openai/responses/"]))
    toml = fixture_toml(routes = Dict("guide/responses_api" => "providers/openai/responses"))
    @test write_stubs(out, "/UniLM.jl/dev", toml) == [joinpath(out, "guide", "responses_api", "index.html")]
    html = read(joinpath(out, "guide", "responses_api", "index.html"), String)
    @test occursin("url=../../providers/openai/responses/", html) && occursin("location.hash", html)
    m = stub_map(out, "providers/openai/responses/", toml)
    @test m["responses_guide"] == "providers/openai/responses/#responses_guide" && !haskey(m, "no-such-id")  # unknown → top
end
@testset "redirects — refused: a stub over a page, a missing target, index.html" begin
    out = fixture_out(pages = ["guide/responses_api/", "x/"], anchors = Dict{String,Vector{String}}())
    @test_throws ErrorException write_stubs(out, "", fixture_toml(routes = Dict("guide/responses_api" => "x")))
    @test_throws ErrorException write_stubs(out, "", fixture_toml(routes = Dict("old" => "missing")))
    @test_throws ErrorException write_stubs(out, "", fixture_toml(routes = Dict("" => "x")))
end
@testset "redirects — every @id of v0.26.0 resolves through anchors.json or an alias" begin
    out = fixture_out(pages = ["a/"], anchors = Dict("kept" => ["/a/"], "new" => ["/a/"]))
    list = tempname(); write(list, "kept\nretired\n")
    @test check_old_anchors(out, fixture_toml(anchors = Dict("retired" => "new")), list) == String[]
    @test check_old_anchors(out, fixture_toml(), list) == ["retired"]
    write_stubs(out, "", fixture_toml(anchors = Dict("retired" => "new")))
    @test JSON.parsefile(joinpath(out, "anchor-aliases.json")) == Dict("retired" => "new")
end

@testset "redirects — an alias goes to its successor's page; an id on two routes, not the target, is left out" begin
    out = fixture_out(pages = ["a/", "b/", "c/"], anchors = Dict(
        "both" => ["/a/", "/b/"], "on_b" => ["/b/"], "on_b_c" => ["/b/", "/c/"], "home" => ["/"], "new" => ["/b/"]))
    @test stub_map(out, "/a", fixture_toml(anchors = Dict("retired" => "new"))) ==
          Dict("both" => "a/#both", "on_b" => "b/#on_b", "home" => "#home", "new" => "b/#new", "retired" => "b/#new")
end
@testset "redirects — the stub's links, and a script that decodes the hash and percent-encodes the id" begin
    out = fixture_out(pages = ["a/", "b/"], anchors = Dict(
        "x" => ["/a/"], "café" => ["/a/"], "</script>" => ["/a/"], "new" => ["/b/"]))
    toml = fixture_toml(routes = Dict("/old/deep/" => "/a"), anchors = Dict("retired" => "new"))
    @test write_stubs(out, "/UniLM.jl/dev", toml) == [joinpath(out, "old", "deep", "index.html")]
    html = read(joinpath(out, "old", "deep", "index.html"), String)
    @test occursin("""<link rel="canonical" href="/UniLM.jl/dev/a/">""", html)
    @test occursin("""<meta http-equiv="refresh" content="2; url=../../a/">""", html)
    @test follow(html, ["#x", "#caf%C3%A9", "#%3C%2Fscript%3E", "#retired", "#no-such-id", "#%E0%A4%A", ""]) ==
          ["../../a/#x", "../../a/#caf%C3%A9", "../../a/#%3C%2Fscript%3E", "../../b/#new",
           "../../a/", "../../a/", "../../a/"]
end
@testset "redirects — refused before writing: an unknown successor id, a route listed twice, a malformed toml" begin
    out = fixture_out(pages = ["a/"], anchors = Dict("kept" => ["/a/"]))
    stubs(; tables...) = write_stubs(out, "", fixture_toml(; tables...))
    @test_throws ErrorException "\"nope\"" stubs(routes = Dict("old" => "a"), anchors = Dict("retired" => "nope"))
    @test_throws ErrorException "\"other/\"" stubs(routes = Dict("old" => "a", "other" => "missing"))
    @test_throws ErrorException "\"old/\"" stubs(routes = Dict("old" => "a", "/old/" => "a"))
    @test readdir(out) == ["a", "anchors.json"]
    for body in ("[route]\n[anchors]\n", "routes = 1\n[anchors]\n", "[routes]\nold = 1\n[anchors]\n")
        toml = tempname(); write(toml, body)
        @test_throws ErrorException toml stub_map(out, "a/", toml)
    end
end
@testset "redirects — anchors.json missing or malformed is an error naming its path" begin
    out, toml, list = mktempdir(), fixture_toml(), tempname()
    write(list, "kept\n")
    path = joinpath(out, "anchors.json")
    for body in (nothing, "{bad", "[]",
                 """{"a": "/a/"}""", """{"a": {"b": "/a/"}}""", """{"a": []}""", """{"a": [1]}""")
        body === nothing ? rm(path; force = true) : write(path, body)
        @test_throws ErrorException path stub_map(out, "a/", toml)
        @test_throws ErrorException path write_stubs(out, "", toml)
        @test_throws ErrorException path check_old_anchors(out, toml, list)
    end
end
@testset "redirects — a route with an empty, . or .. segment is refused, naming its file and the route" begin
    for (old, new) in (("g//x", "g//r"), ("./x", "./a"), ("../escaped", "b/../a"))
        # the export in a folder of its own, so that a stub written at "../escaped" stays in a fresh temp dir
        out = mv(fixture_out(pages = ["a/", "b/", "g/r/"], anchors = Dict("kept" => ["/a/"])),
                 joinpath(mktempdir(), "out"))
        toml = fixture_toml(routes = Dict(old => "a"))
        @test_throws ErrorException [toml, old] write_stubs(out, "", toml)
        toml = fixture_toml(routes = Dict("old" => new))
        @test_throws ErrorException [toml, new] write_stubs(out, "", toml)
        @test_throws ErrorException new stub_map(out, new, fixture_toml())
        write(joinpath(out, "anchors.json"), JSON.json(Dict("kept" => ["/$new/"])))
        @test_throws ErrorException [joinpath(out, "anchors.json"), new] stub_map(out, "a/", fixture_toml())
    end
end
@testset "redirects — the CLI writes the stubs and exits 1 while an id of v0.26.0 resolves nowhere" begin
    ids = readlines(joinpath(@__DIR__, "anchors-v0.26.0.txt"))
    @test length(ids) == 139 && issorted(ids) && allunique(ids)
    out = fixture_out(pages = ["a/"], anchors = Dict(id => ["/a/"] for id in ids[2:end]))
    function cli(toml::String)::Tuple{Int,String}
        err = IOBuffer()
        script = joinpath(@__DIR__, "redirects.jl")
        cmd = `$(Base.julia_cmd()) --startup-file=no --project=$(@__DIR__) $script $out /UniLM.jl/dev $toml`
        (run(pipeline(ignorestatus(cmd); stdout = devnull, stderr = err)).exitcode, String(take!(err)))
    end
    code, err = cli(fixture_toml())
    @test code == 1 && occursin(ids[1], err)
    code, err = cli(fixture_toml(routes = Dict("old" => "a"), anchors = Dict(ids[1] => ids[2])))
    @test code == 0 && isfile(joinpath(out, "old", "index.html"))
end

end
