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

"The routes an earlier version served, one per line, in a new temp file."
function fixture_routes(routes::String...)::String
    path = tempname()
    write(path, join(routes, '\n') * "\n")
    path
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

# Runs the site's anchor forwarder (site/src/lib/anchors.ts, whose types Node.js 22.6 or later strips) as a page's
# head runs it: its text alone, in a context holding only what a page gives it, where `fetch` answers each anchor map
# with the JSON given ("null": a 404). Prints {"to": <the destination, or null to stay>, "errors": [<what it logged>]}.
const RUN_FORWARDER = """
const vm = require("node:vm"), { pathToFileURL } = require("node:url");
const [file, base, hash, path, aliases, anchors] = process.argv.slice(1);
import(pathToFileURL(file)).then(({ anchorDestination }) => {
  const errors = [], maps = { "anchor-aliases.json": aliases, "anchors.json": anchors };
  const respond = (body) => body === "null" ? { ok: false, status: 404 } :
    { ok: true, status: 200, json: () => Promise.resolve().then(() => JSON.parse(body)) };
  const page = {
    document: { getElementById: () => null },
    location: { hash, pathname: path },
    fetch: (url) => Promise.resolve(respond(maps[url.slice(base.length + 1)])),
    console: { error: (message) => errors.push(String(message)) },
  };
  const call = `(\${anchorDestination})(\${JSON.stringify(base)}, \${JSON.stringify(hash)}, \${JSON.stringify(path)})`;
  return vm.runInNewContext(call, page).then((to) => console.log(JSON.stringify({ to: to ?? null, errors })));
});
"""
const ANCHORS_TS = joinpath(@__DIR__, "..", "site", "src", "lib", "anchors.ts")

"""
Where the site's anchor forwarder sends a reader who opens `path`, under the base path `/base`, with `hash`, given
the JSON of anchor-aliases.json (`nothing`: no such file) and of anchors.json: the destination (`nothing` to stay)
and the errors it logged.
"""
function forward(hash::String, path::String; aliases::Union{String,Nothing}, anchors::String)
    args = [ANCHORS_TS, "/base", hash, path, something(aliases, "null"), anchors]
    cmd = addenv(`node --experimental-strip-types -e $RUN_FORWARDER -- $args`, "NODE_NO_WARNINGS" => "1")
    r = JSON.parse(read(cmd, String))
    (r["to"], r["errors"])
end

@testset "redirects" begin

@testset "redirects — a stub forwards to its target and keeps a known hash" begin
    out = fixture_out(pages = ["providers/openai/responses/"], anchors = Dict("responses_guide" => ["/providers/openai/responses/"]))
    toml = fixture_toml(routes = Dict("guide/responses_api" => "providers/openai/responses"))
    @test write_stubs(out, "/UniLM.jl/dev", toml, fixture_routes("/guide/responses_api/")) ==
          [joinpath(out, "guide", "responses_api", "index.html")]
    html = read(joinpath(out, "guide", "responses_api", "index.html"), String)
    @test occursin("url=../../providers/openai/responses/", html) && occursin("location.hash", html)
    m = stub_map(out, "providers/openai/responses/", toml)
    @test m["responses_guide"] == "providers/openai/responses/#responses_guide" && !haskey(m, "no-such-id")  # unknown → top
end
@testset "redirects — refused: a stub over a page, a missing target, index.html" begin
    out = fixture_out(pages = ["guide/responses_api/", "x/"], anchors = Dict{String,Vector{String}}())
    served = fixture_routes("/", "/guide/responses_api/", "/old/")
    @test_throws ErrorException write_stubs(out, "", fixture_toml(routes = Dict("guide/responses_api" => "x")), served)
    @test_throws ErrorException write_stubs(out, "", fixture_toml(routes = Dict("old" => "missing")), served)
    @test_throws ErrorException write_stubs(out, "", fixture_toml(routes = Dict("" => "x")), served)
end
@testset "redirects — every @id of v0.26.0 resolves through anchors.json or an alias" begin
    out = fixture_out(pages = ["a/"], anchors = Dict("kept" => ["/a/"], "new" => ["/a/"]))
    list = tempname(); write(list, "kept\nretired\n")
    @test check_old_anchors(out, fixture_toml(anchors = Dict("retired" => "new")), list) == String[]
    @test check_old_anchors(out, fixture_toml(), list) == ["retired"]
    write_stubs(out, "", fixture_toml(anchors = Dict("retired" => "new")), fixture_routes("/a/"))
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
    @test write_stubs(out, "/UniLM.jl/dev", toml, fixture_routes("/old/deep/")) == [joinpath(out, "old", "deep", "index.html")]
    html = read(joinpath(out, "old", "deep", "index.html"), String)
    @test occursin("""<link rel="canonical" href="/UniLM.jl/dev/a/">""", html)
    @test occursin("""<meta http-equiv="refresh" content="2; url=../../a/">""", html)
    @test follow(html, ["#x", "#caf%C3%A9", "#%3C%2Fscript%3E", "#retired", "#no-such-id", "#%E0%A4%A", ""]) ==
          ["../../a/#x", "../../a/#caf%C3%A9", "../../a/#%3C%2Fscript%3E", "../../b/#new",
           "../../a/", "../../a/", "../../a/"]
end
@testset "redirects — refused before writing: an unknown successor id, a route listed twice, a malformed toml" begin
    out = fixture_out(pages = ["a/"], anchors = Dict("kept" => ["/a/"]))
    stubs(; tables...) = write_stubs(out, "", fixture_toml(; tables...), fixture_routes("/old/", "/other/"))
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
        @test_throws ErrorException path write_stubs(out, "", toml, fixture_routes("/a/"))
        @test_throws ErrorException path check_old_anchors(out, toml, list)
    end
end
@testset "redirects — a route with an empty, . or .. segment is refused, naming its file and the route" begin
    for (old, new) in (("g//x", "g//r"), ("./x", "./a"), ("../escaped", "b/../a"))
        # the export in a folder of its own, so that a stub written at "../escaped" stays in a fresh temp dir
        out = mv(fixture_out(pages = ["a/", "b/", "g/r/"], anchors = Dict("kept" => ["/a/"])),
                 joinpath(mktempdir(), "out"))
        toml = fixture_toml(routes = Dict(old => "a"))
        @test_throws ErrorException [toml, old] write_stubs(out, "", toml, fixture_routes("/a/"))
        toml = fixture_toml(routes = Dict("old" => new))
        @test_throws ErrorException [toml, new] write_stubs(out, "", toml, fixture_routes("/old/"))
        @test_throws ErrorException new stub_map(out, new, fixture_toml())
        write(joinpath(out, "anchors.json"), JSON.json(Dict("kept" => ["/$new/"])))
        @test_throws ErrorException [joinpath(out, "anchors.json"), new] stub_map(out, "a/", fixture_toml())
    end
end
@testset "redirects — a stub goes only where an earlier version served a page" begin
    out = fixture_out(pages = ["", "a/"], anchors = Dict("kept" => ["/a/"]))
    served = fixture_routes("/", "/a/", "/guide/old/")
    @test_throws ErrorException ["never_served/", served] write_stubs(out, "", fixture_toml(routes = Dict("never_served" => "a")), served)
    @test !ispath(joinpath(out, "never_served")) && !isfile(joinpath(out, "anchor-aliases.json"))   # refused before writing
    @test check_old_routes(out, served) == ["/guide/old/"]                   # neither a page nor a stub
    write_stubs(out, "", fixture_toml(routes = Dict("guide/old" => "a")), served)
    @test check_old_routes(out, served) == String[]
end
@testset "redirects — a route of any list of served routes takes a stub, of none is refused; a # line is no route" begin
    out = fixture_out(pages = ["", "a/"], anchors = Dict("kept" => ["/a/"]))
    v026, earlier = fixture_routes("/", "/a/", "/guide/old/"), fixture_routes("# served before", "/guide/older/")
    @test check_old_routes(out, v026, earlier) == ["/guide/old/", "/guide/older/"]
    @test write_stubs(out, "", fixture_toml(routes = Dict("guide/old" => "a", "guide/older" => "a")), v026, earlier) ==
          [joinpath(out, "guide", "old", "index.html"), joinpath(out, "guide", "older", "index.html")]
    @test check_old_routes(out, v026, earlier) == String[]
    never_served = fixture_toml(routes = Dict("never_served" => "a"))
    @test_throws ErrorException ["\"never_served/\"", v026, earlier] write_stubs(out, "", never_served, v026, earlier)
end
@testset "redirects — a run overwrites the stubs an earlier run wrote, never a page" begin
    out = fixture_out(pages = ["a/", "b/"], anchors = Dict("kept" => ["/a/"]))
    served, stub = fixture_routes("/old/"), joinpath(out, "old", "index.html")
    for to in ("a", "b")                                                     # the second run moves the stub on
        @test write_stubs(out, "", fixture_toml(routes = Dict("old" => to)), served) == [stub]
        @test occursin("url=../$to/", read(stub, String))
    end
    write(stub, "<!DOCTYPE html>")                                           # a page where the stub was
    @test_throws ErrorException ["\"old/\"", "is a page"] write_stubs(out, "", fixture_toml(routes = Dict("old" => "a")), served)
    @test read(stub, String) == "<!DOCTYPE html>"
end
@testset "redirects — a stub an earlier run wrote is no page: a route never moves to one" begin
    # A rerun over an export that holds stubs must refuse what a fresh export refuses, where no stub is yet.
    out = fixture_out(pages = ["a/"], anchors = Dict("kept" => ["/a/"]))
    served = fixture_routes("/old/", "/older/")
    write_stubs(out, "", fixture_toml(routes = Dict("old" => "a")), served)          # a stub at old/
    @test_throws ErrorException ["\"older/\"", "moves to \"old/\"", "not a page"] write_stubs(
        out, "", fixture_toml(routes = Dict("older" => "old")), served)
    @test !ispath(joinpath(out, "older"))                                             # refused before writing
end
@testset "redirects — the CLI exits 1 while a route an earlier version served has neither a page nor a stub, or an id resolves nowhere; a rerun overwrites its stubs" begin
    ids = readlines(joinpath(@__DIR__, "anchors-v0.26.0.txt"))
    routes = readlines(joinpath(@__DIR__, "routes-v0.26.0.txt"))
    earlier = filter(!startswith('#'), readlines(joinpath(@__DIR__, "routes-earlier.txt")))
    @test length(ids) == 139 && issorted(ids) && allunique(ids)
    @test length(routes) == 54 && issorted(routes) && allunique(routes) && first(routes) == "/" &&
          all(r -> startswith(r, '/') && endswith(r, '/'), routes)
    @test !isempty(earlier) && issorted(earlier) && allunique(earlier) && isdisjoint(earlier, routes) &&
          all(r -> startswith(r, '/') && endswith(r, '/'), earlier)
    moved = "/guide/responses_api/"                                          # a page v0.26.0 served
    @test moved in routes
    pages = String[[chopprefix(r, "/") for r in routes if r != moved]; "providers/openai/responses/"]
    out = fixture_out(; pages, anchors = Dict(id => ["/llm/"] for id in ids[2:end]))
    function cli(toml::String)::Tuple{Int,String}
        err = IOBuffer()
        script = joinpath(@__DIR__, "redirects.jl")
        cmd = `$(Base.julia_cmd()) --startup-file=no --project=$(@__DIR__) $script $out /UniLM.jl/dev $toml`
        (run(pipeline(ignorestatus(cmd); stdout = devnull, stderr = err)).exitcode, String(take!(err)))
    end
    alias = Dict(ids[1] => ids[2])
    older = Dict{String,String}(r => "llm" for r in earlier)                 # a stub at each route of the earlier list
    code, err = cli(fixture_toml(anchors = alias))                           # the moved page has no stub, nor have they
    @test code == 1 && occursin(moved, err) && all(r -> occursin(r, err), earlier) && !occursin(ids[1], err)
    code, err = cli(fixture_toml(routes = Dict("guide/responses" => "providers/openai/responses"), anchors = alias))
    @test code == 1 && occursin("guide/responses/", err)                     # a route no version served
    code, err = cli(fixture_toml(routes = Dict("guide/responses_api" => "providers/openai/responses", older...)))
    @test code == 1 && occursin(ids[1], err) && !occursin(moved, err)        # an id resolves nowhere
    stub = joinpath(out, "guide", "responses_api", "index.html")             # that run wrote it before the check
    for to in ("providers/openai/responses", "llm")                          # each run overwrites the run before's
        code, err = cli(fixture_toml(routes = Dict("guide/responses_api" => to, older...), anchors = alias))
        @test code == 0 && occursin("url=../../$to/", read(stub, String))
    end
end
@testset "redirects — no [anchors] writes the aliases file {}, which the site's forwarder reads as no aliases" begin
    out = fixture_out(pages = ["a/"], anchors = Dict("kept" => ["/a/"]))
    write_stubs(out, "", fixture_toml(), fixture_routes("/a/"))
    aliases = read(joinpath(out, "anchor-aliases.json"), String)
    @test aliases == "{}"
    @test forward("#kept", "/base/b/"; aliases, anchors = """{"kept": ["/a/"]}""") ==
          (Dict("route" => "/a/", "id" => "kept", "here" => false), [])
end
@testset "redirects — the site's forwarder reads only a map's own keys: #constructor names no id until a map holds it" begin
    kept = """{"kept": ["/a/"]}"""
    @test forward("#constructor", "/base/b/"; aliases = "{}", anchors = kept) == (nothing, [])
    @test forward("#constructor", "/base/b/"; aliases = "{}", anchors = """{"constructor": ["/a/"]}""") ==
          (Dict("route" => "/a/", "id" => "constructor", "here" => false), [])
    @test forward("#constructor", "/base/b/"; aliases = """{"constructor": "kept"}""", anchors = kept) ==
          (Dict("route" => "/a/", "id" => "kept", "here" => false), [])
end

end
