# Keeps links into earlier versions of the documentation site working after pages and anchors move: redirect stubs
# at the old routes, the anchor aliases the site's pages read, and a check that every route an earlier version served
# and every anchor id of v0.26.0 resolves.
#
#   julia --startup-file=no --project=docs docs/redirects.jl <out> <base> [<toml>]
#
# <out> is the site's static export (site/out, holding anchors.json), <base> the path it is served under
# (/UniLM.jl/dev, /UniLM.jl/vX.Y.Z, or "" at the root), <toml> the redirect table (default docs/redirects.toml).
# Writes the stubs, over those an earlier run wrote into <out>, and <out>/anchor-aliases.json, then checks that each
# route of docs/routes-v0.26.0.txt and docs/routes-earlier.txt is a page or a stub of <out> and that each id of
# docs/anchors-v0.26.0.txt resolves; exits 1 on any problem.
module Redirects

using JSON, TOML

export stub_map, write_stubs, check_old_routes, check_old_anchors

"""
    route(r, from) -> String

The route `r` with no leading slash and one trailing slash: `"/guide/x"` is `"guide/x/"`, the home page `""`. A route
with an empty, `.` or `..` segment is an error naming `from`, the file or argument it came from: a stub's relative
links climb one `../` per segment, and its path must stay inside the export.
"""
function route(r::AbstractString, from::String)::String
    s = strip(r, '/')
    isempty(s) && return ""
    any(in(("", ".", "..")), eachsplit(s, '/')) && error("$from: the route $(repr(r)) has an empty, . or .. segment")
    string(s, '/')
end

"The routes of the file `list`, one per line (`/guide/x/`), in its order; a line that starts with `#` is a comment."
listed_routes(list::String)::Vector{String} = filter(!startswith('#'), readlines(list))

"The table `name` of `t`, the parsed file `toml`, of strings; anything else is an error naming the file and key."
function string_table(t::Dict{String,Any}, name::String, toml::String)::Dict{String,String}
    table = t[name]
    table isa Dict{String,Any} || error("$toml: $name is not a table")
    Dict(k => (v isa String ? v : error("$toml: [$name] $(repr(k)) maps to $(repr(v)), not a string"))
         for (k, v) in table)
end

"""
    Config(toml)

The redirect table `toml`, parsed: `routes` maps each old route to the route that replaced it, both normalised by
`route`, and `aliases` each retired anchor id to its successor id. The file holds exactly the tables `[routes]` and
`[anchors]`, of string values; anything else, a route that `route` refuses, or two old routes that normalise alike,
is an error.
"""
struct Config
    routes::Dict{String,String}
    aliases::Dict{String,String}
end
function Config(toml::String)
    t = TOML.parsefile(toml)
    issetequal(keys(t), ("routes", "anchors")) ||
        error("$toml: expected exactly the tables [routes] and [anchors], found $(sort!(collect(keys(t))))")
    routes = Dict{String,String}()
    for (old, new) in string_table(t, "routes", toml)
        r = route(old, toml)
        haskey(routes, r) && error("$toml: [routes] lists the route $(repr(r)) twice")
        routes[r] = route(new, toml)
    end
    Config(routes, string_table(t, "anchors", toml))
end

"""
    read_anchors(out) -> Dict{String,Vector{String}}

`<out>/anchors.json`: each anchor id of the built site with the routes that define it, normalised by `route`. A
missing file, one that is not a JSON object mapping each id to a non-empty array of routes, or a route that `route`
refuses, is an error naming its path.
"""
function read_anchors(out::String)::Dict{String,Vector{String}}
    path = joinpath(out, "anchors.json")
    a = try
        JSON.parsefile(path)
    catch e
        error("$path: $(sprint(showerror, e))")
    end
    a isa AbstractDict && all(rs -> rs isa AbstractVector && !isempty(rs) && all(r -> r isa String, rs), values(a)) ||
        error("$path: not a JSON object mapping each anchor id to the routes that define it")
    Dict{String,Vector{String}}(id => route.(rs, path) for (id, rs) in a)
end

"""
    stub_map(out, target, toml) -> Dict{String,String}

For each anchor id of `<out>/anchors.json` and each retired id of `toml`'s `[anchors]`, where a stub forwarding to
the route `target` sends a reader who arrives with that id as the hash: `"route#id"`, with the id resolved through
`[anchors]` first and the route `target` when it defines that id, else the one route that does. An id that `target`
does not define and that several routes or none define is left out: its reader goes to the top of `target`. Routes
are normalised by `route`.
"""
stub_map(out::String, target::String, toml::String)::Dict{String,String} =
    stub_map(read_anchors(out), Config(toml).aliases, route(target, "stub_map's target"))

"""
    stub_map(anchors, aliases, target) -> Dict{String,String}

The same map, from the parsed `anchors.json` and `[anchors]` table, for the normalised route `target`.
"""
function stub_map(anchors::Dict{String,Vector{String}}, aliases::Dict{String,String},
                  target::String)::Dict{String,String}
    dests = Dict{String,String}()
    for key in union(keys(anchors), keys(aliases))
        id = get(aliases, key, key)
        routes = get(anchors, id, String[])
        if target in routes
            dests[key] = string(target, '#', id)
        elseif length(routes) == 1
            dests[key] = string(only(routes), '#', id)
        end
    end
    dests
end

"""
`x` as a JavaScript literal inside an HTML `<script>`: its JSON, with `<` escaped, since an HTML parser ends the
script at the first `</script`, even one inside a string.
"""
js(x)::String = replace(JSON.json(x), '<' => "\\u003c")

"""
The tag in the head of every stub: HTML's `generator` metadata, naming the software that wrote the page. A later run
over the same export overwrites a file that holds it, and refuses to replace any other, a page of the site.
"""
const STUB_MARK = """<meta name="generator" content="UniLM.jl docs/redirects.jl">"""

"Whether the file `path` is a stub that an earlier run wrote: it holds `STUB_MARK`."
is_stub(path::String)::Bool = occursin(STUB_MARK, read(path, String))

"The HTML of the stub at route `old` for the route `new` that replaced it; `dests` is its `stub_map`."
function stub(old::String, new::String, base::String, dests::Dict{String,String})::String
    up = "../"^count('/', old)
    """
    <!DOCTYPE html>
    <html lang="en">
    <head>
    <meta charset="utf-8">
    $STUB_MARK
    <title>Moved</title>
    <link rel="canonical" href="$base/$new">
    <meta http-equiv="refresh" content="2; url=$up$new">
    <script>
    const dests = $(js(dests));
    let to = $(js(new));
    try {
      const id = decodeURIComponent(location.hash.slice(1));
      if (Object.hasOwn(dests, id)) to = dests[id];
    } catch {}  // a hash that does not decode names no anchor: the reader goes to the top
    location.replace($(js(up)) + to.replace(/#(.*)\$/s, (_, id) => "#" + encodeURIComponent(id)));
    </script>
    </head>
    <body><p><a href="$up$new">This page has moved.</a></p></body>
    </html>
    """
end

"""
    write_stubs(out, base, toml, lists...) -> Vector{String}

Writes `<out>/anchor-aliases.json` from `toml`'s `[anchors]` and, for each old route of its `[routes]`, a stub
`<out>/<old>/index.html` that moves its reader to the route that replaced it; returns the stubs' paths, sorted. The
stub's script sends a reader to the anchor `stub_map` gives for the hash they arrived with, else to the top of the
page; without scripts, a `<meta http-equiv="refresh">` sends them to the top after two seconds. Both use relative
URLs, so a reader stays in the folder they came from (dev/, stable/ or a version's); the canonical link is
`<base>/<new>/`.

Refuses, before writing anything, an old route that is the home page, that is a route of none of the files `lists` —
each lists routes an earlier version served (`listed_routes`), so a stub anywhere else forwards nobody — or where a
page exists, a new route that is not a page, and a successor id that `<out>/anchors.json` lacks. A stub an earlier
run wrote at an old route, which holds `STUB_MARK`, is no page: this run's stub replaces it.
"""
function write_stubs(out::String, base::String, toml::String, lists::String...)::Vector{String}
    c, anchors = Config(toml), read_anchors(out)
    served = Set(route(r, list) for list in lists for r in listed_routes(list))
    for (old, new) in c.aliases
        haskey(anchors, new) ||
            error("$toml: [anchors] $(repr(old)) names the successor $(repr(new)), which $out/anchors.json lacks")
    end
    for (old, new) in c.routes
        isempty(old) && error("$toml: [routes] moves the home page, which a stub would replace")
        old in served || error("$toml: [routes] $(repr(old)) is a route of none of $(join(lists, ", ")), which list " *
                               "the routes earlier versions served: a stub there forwards nobody")
        page = joinpath(out, old, "index.html")
        isfile(page) && !is_stub(page) &&
            error("$toml: [routes] $(repr(old)) is a page of $out, which a stub would replace")
        isfile(joinpath(out, new, "index.html")) ||
            error("$toml: [routes] $(repr(old)) moves to $(repr(new)), which is not a page of $out")
    end
    write(joinpath(out, "anchor-aliases.json"), JSON.json(c.aliases))
    map(sort!(collect(c.routes))) do (old, new)
        path = joinpath(out, old, "index.html")
        mkpath(dirname(path))
        write(path, stub(old, new, base, stub_map(anchors, c.aliases, new)))
        path
    end
end

"""
    check_old_routes(out, lists...) -> Vector{String}

The routes of the files `lists` (`listed_routes`), in their order, where the built site `out` has neither a page nor
a stub (`<out>/<route>/index.html`): a link to one of them finds nothing.
"""
check_old_routes(out::String, lists::String...)::Vector{String} =
    [r for list in lists for r in listed_routes(list) if !isfile(joinpath(out, route(r, list), "index.html"))]

"""
    check_old_anchors(out, toml, list) -> Vector{String}

The anchor ids of the file `list` (one per line), in its order, that are neither keys of `<out>/anchors.json` nor
retired ids of `toml`'s `[anchors]` whose successor is one: a link to such an id lands at the top of a page.
"""
function check_old_anchors(out::String, toml::String, list::String)::Vector{String}
    anchors, aliases = read_anchors(out), Config(toml).aliases
    filter(id -> !haskey(anchors, id) && !haskey(anchors, get(aliases, id, id)), readlines(list))
end

end # module

if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) in 2:3 ||
        error("usage: julia --startup-file=no --project=docs docs/redirects.jl <out> <base> [<toml>]")
    out, base, toml = ARGS[1], ARGS[2], get(ARGS, 3, joinpath(@__DIR__, "redirects.toml"))
    routes = joinpath.(@__DIR__, ("routes-v0.26.0.txt", "routes-earlier.txt"))
    stubs = Redirects.write_stubs(out, base, toml, routes...)
    gone = Redirects.check_old_routes(out, routes...)
    lost = Redirects.check_old_anchors(out, toml, joinpath(@__DIR__, "anchors-v0.26.0.txt"))
    isempty(gone) || println(stderr, "routes an earlier version served with neither a page nor a stub in $out (add ",
                             "one to [routes] of $toml):", join("\n  " .* gone))
    isempty(lost) || println(stderr, "anchor ids of v0.26.0 that neither $out/anchors.json defines nor an alias of ",
                             "$toml resolves:", join("\n  " .* lost))
    isempty(gone) && isempty(lost) || exit(1)
    println("wrote ", join([joinpath(out, "anchor-aliases.json"); stubs], ", "))
end
