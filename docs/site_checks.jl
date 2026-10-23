# Checks the documentation site once it is built: the links that README.md, llms.txt and the home page's hero give
# into it, and the outputs of the examples on its pages.
#
#   julia --startup-file=no --project=docs docs/site_checks.jl <out> <app>
#
# <out> is the site's static export (site/out, holding anchors.json), <app> the directory of its generated pages
# (site/src/app). Prints every link that does not resolve and every line of an example's output that reads as a
# failure; exits 1 when there is one.
module SiteChecks

include(joinpath(@__DIR__, "redirects.jl"))
using .Redirects: read_anchors, route

export check_links, scan_outputs

"""
A URL of the site's dev or stable build: `route` from the `/` after the version on (empty when the URL stops at the
version), `hash` after a `#`. It ends before a space, a quote, a bracket, a parenthesis, `<` or `>`, and before the
punctuation that a space or the end of the line follows.
"""
const SITE_URL = r"(?<link>https://algunion\.github\.io/UniLM\.jl/(?:dev|stable)(?<route>(?:/[^\s#()<>\[\]\"'`]*?)?)(?:#(?<hash>[^\s()<>\[\]\"'`]*?))?)(?=[?!.,:*_~]*(?!\S)|[()<>\[\]\"'`])"

"A JSX `href` naming a route of the site, and its `#` fragment."
const HREF = r"\bhref=\"(?<link>(?<route>/[^\"#]*)(?:#(?<hash>[^\"]*))?)\""

"""
A Markdown link or image whose destination has no URL scheme: a path (empty for a heading of its own file), then a
`?query` and a `#hash`.
"""
const MD_LINK = r"\]\((?<link>(?![A-Za-z][A-Za-z0-9+.-]*:)(?<path>[^\s)?#]*)[^\s)#]*(?:#(?<hash>[^\s)]*))?)"

"An ATX heading and its text, without the closing `#`s."
const ATX = r"^ {0,3}#{1,6}(?:[ \t]+|$)(.*?)(?:[ \t]+#+)?[ \t]*$"

"""
A line opening a backtick-fenced code block: the list item and block quote markers before the fence, the fence, and
the first word of its info string, which holds no backtick.
"""
const OPENING = r"^((?:[ >]|[-*+] |\d+[.)] )*)(`{3,})[ \t]*([^`\s]*)[^`]*$"

"A line of an example's output that reads as a failure."
const FAILURE = r"^Request failed|ReplayMissError|KeyError: key|CallError\("m

"A link at `line` of a file, `link` as written there, and why it does not resolve: `problem`, `nothing` when it does."
const Link = @NamedTuple{line::Int, link::SubString{String}, problem::Union{Nothing,String}}

"""
    github_slug(text) -> String

The id GitHub derives from a heading with the text `text`: lower-cased (by Unicode's simple case mapping, where GitHub
turns `İ` into `i̇`), then without each character that is not alphabetic, a mark, a decimal digit, connector
punctuation such as `_`, `-` or a space, then with each space turned into `-`. `Ⓐ`, a symbol Unicode counts as
alphabetic, stays; `½` and `♥` go.
"""
github_slug(text::AbstractString)::String =
    replace(lowercase(text), r"[^\p{Alphabetic}\p{M}\p{Nd}\p{Pc} -]" => "", ' ' => '-')

"""
    github_ids(headings) -> Set{String}

The ids GitHub gives the headings of a file whose texts, in order, are `headings`: each one's `github_slug`, or, when
an earlier heading has that id, the first of `slug-1`, `slug-2`, … that none has.
"""
github_ids(headings::Vector{String})::Set{String} =
    foldl((ids, h) -> push!(ids, unused(github_slug(h), ids)), headings; init = Set{String}())

"The first of `id`, `id-1`, `id-2`, … that `ids` lacks."
unused(id::String, ids::Set{String}, n::Int = 0)::String =
    (s = n == 0 ? id : "$id-$n") in ids ? unused(id, ids, n + 1) : s

"""
    decode(s) -> Union{Nothing,String}

`s` with its `%XX` escapes decoded as UTF-8, as JavaScript's `decodeURIComponent` decodes it; `nothing` where that
throws: a `%` without two hex digits after it, or escapes that are not UTF-8.
"""
function decode(s::AbstractString)::Union{Nothing,String}
    d = replace(s, r"(?:%[0-9A-Fa-f]{2})+" => e -> String(hex2bytes(replace(e, '%' => ""))))
    occursin(r"%(?![0-9A-Fa-f]{2})", s) || !isvalid(d) ? nothing : d
end

"""
The fence opening a code block: its number of backticks, the first word of its info string, and its column — the
width of the list item and block quote markers before it, where the lines of the block start.
"""
struct Fence
    ticks::Int
    info::String
    column::Int
end

"Whether the Markdown line `l` closes the block of `f`: from its column on, at least its backticks and nothing else."
closes(f::Fence, l::String)::Bool =
    (c = strip(chop(l; head = f.column, tail = 0)); length(c) >= f.ticks && all(==('`'), c))

"""
    fences(lines) -> Vector{Union{Nothing,Fence}}

For each of the Markdown `lines`, the fence of the backtick-fenced code block that holds it; `nothing` outside every
block and on the fences themselves. A block runs to the line that `closes` it, or to the end of the file.
"""
function fences(lines::Vector{String})::Vector{Union{Nothing,Fence}}
    found = Union{Nothing,Fence}[]
    open::Union{Nothing,Fence} = nothing
    for l in lines
        if open === nothing
            push!(found, nothing)
            m = match(OPENING, l)
            m === nothing || (open = Fence(length(m[2]), m[3], length(m[1])))
        elseif closes(open, l)
            push!(found, nothing)
            open = nothing
        else
            push!(found, open)
        end
    end
    found
end

"""
Why a link to the site's route `r`, as written (`/guide/x/`, `/` for the home page), and to the anchor `#hash` on it
(none when empty) does not resolve against the built site `out`; `nothing` when it does. `at` names where it stands.
"""
function site_problem(r::AbstractString, hash::AbstractString, at::String, out::String,
                      anchors::Dict{String,Vector{String}})::Union{Nothing,String}
    endswith(r, '/') || return "no trailing slash"
    page = route(r, at)
    isfile(joinpath(out, page, "index.html")) || return "no page"
    isempty(hash) && return nothing
    id = decode(hash)
    id === nothing ? "does not decode" : page in get(anchors, id, String[]) ? nothing : "no anchor $(repr(id)) on $r"
end

"""
Why a Markdown link to the file `path`, relative to `dir`, or — `path` empty — to the heading `#hash` of its own
file, whose headings have the ids `ids`, does not resolve; `nothing` when it does.
"""
function markdown_problem(path::AbstractString, hash::AbstractString, dir::String,
                          ids::Set{String})::Union{Nothing,String}
    target = decode(isempty(path) ? hash : path)
    target === nothing ? "does not decode" :
    isempty(path) ? (target in ids ? nothing : "no heading") :
    ispath(joinpath(dir, target)) ? nothing : "no file"
end

"The Markdown links of `lines` outside fenced code, to files relative to `dir` and to headings of their own file."
function markdown_links(dir::String, lines::Vector{String})::Vector{Link}
    prose = [(n, l) for (n, (l, f)) in enumerate(zip(lines, fences(lines))) if f === nothing]
    ids = github_ids(String[m[1] for (_, l) in prose for m in eachmatch(ATX, l)])
    Link[(line = n, link = m[:link], problem = markdown_problem(m[:path], something(m[:hash], ""), dir, ids))
         for (n, l) in prose for m in eachmatch(MD_LINK, l)]
end

"The links of `file` that do not resolve, as `check_links` reports them, in the order of their lines."
function link_problems(file::String, out::String, anchors::Dict{String,Vector{String}})::Vector{String}
    ext = last(splitext(file))
    ext in (".md", ".txt", ".tsx") || error("$file: not a .md, .txt or .tsx file")
    lines = readlines(file)
    found = Link[(line = n, link = m[:link],
                  problem = site_problem(m[:route], something(m[:hash], ""), "$file:$n", out, anchors))
                 for (n, l) in enumerate(lines) for m in eachmatch(ext == ".tsx" ? HREF : SITE_URL, l)]
    ext == ".md" && append!(found, markdown_links(dirname(file), lines))
    ["$file:$(f.line): $(f.link) ($(f.problem))" for f in sort!(found; by = f -> f.line) if f.problem !== nothing]
end

"""
    check_links(out, files, anchors) -> Vector{String}

The links of `files` that do not resolve, each as `"file:line: link (problem)"`. A file is read by its extension:

- `.md`, `.txt`: each URL of the site's dev or stable build, `https://algunion.github.io/UniLM.jl/dev/<route>/#id`;
- `.tsx`: each `href="/<route>/#id"`;
- `.md` also: each Markdown link outside fenced code to a file, which must exist relative to the file's directory,
  or to a heading of its own, whose ids are `github_ids`.

A route must end in `/` (`/` alone is the home page) and be a page of the built site `out` (`<out>/<route>/index.html`).
For an `#id`, decoded as `decodeURIComponent` decodes it, `anchors[id]` must list that route; `anchors` maps each
anchor id to the routes that define it, as `<out>/anchors.json` does. Routes are compared as `Redirects.route`
normalises them; a route it refuses is an error naming where it stands, and so is a file of another extension.
"""
function check_links(out::String, files::Vector{String}, anchors::Dict{String,Vector{String}})::Vector{String}
    pages = Dict(id => route.(rs, "check_links' anchors") for (id, rs) in anchors)
    String[p for file in files for p in link_problems(file, out, pages)]
end

"The lines of the example outputs among the Markdown `lines`: each line's number and its text from its block's column."
outputs(lines::Vector{String})::Vector{Tuple{Int,SubString{String}}} =
    [(n, chop(l; head = f.column, tail = 0)) for (n, (l, f)) in enumerate(zip(lines, fences(lines)))
     if f !== nothing && f.info == "output"]

"""
    scan_outputs(app) -> Vector{String}

The lines of the example outputs on the site's pages, `app/**/page.md`, that read as a failure — they start with
`Request failed`, or hold `ReplayMissError`, `KeyError: key` or `CallError(` — each as `"page.md:line: text"`, with
the page's path. An example output is a fenced code block whose info string starts with the word `output`, as the
site's writer writes it; in a list item or a block quote, a line's text starts at its fence's column. An `app` that
holds no page.md is an error.
"""
function scan_outputs(app::String)::Vector{String}
    pages = [joinpath(dir, f) for (dir, _, files) in walkdir(app) for f in files if f == "page.md"]
    isempty(pages) && error("$app: no page.md")
    String[string(page, ':', n, ": ", text) for page in pages for (n, text) in outputs(readlines(page))
           if occursin(FAILURE, text)]
end

end # module

if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) == 2 || error("usage: julia --startup-file=no --project=docs docs/site_checks.jl <out> <app>")
    out, app = ARGS
    files = joinpath.(dirname(@__DIR__), ["README.md", "llms.txt", joinpath("site", "src", "components", "Hero.tsx")])
    problems = [SiteChecks.check_links(out, files, SiteChecks.read_anchors(out)); SiteChecks.scan_outputs(app)]
    if !isempty(problems)
        println(stderr, "links that do not resolve against $out, and failures in the example outputs of $app:",
                join("\n  " .* problems))
        exit(1)
    end
    println("checked the links of ", join(files, ", "), " against $out, and the example outputs of $app")
end
