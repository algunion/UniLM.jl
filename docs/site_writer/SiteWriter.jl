# ============================================================================
# The documentation site's writer. Documenter stays the engine — it runs every
# example, expands `@docs`, resolves `@ref` and checks doc coverage — and this
# writer turns its processed pages into the Markdoc pages, the sidebar and the
# static assets of the Next.js site in `site/`.
#
# Two layers. Reading walks Documenter's processed MarkdownAST into a small page
# model: anchors, resolved links, docstrings and example outputs are settled
# there. Writing serialises that model as Markdoc (CommonMark plus `{% %}` tags).
# Anything the model cannot hold stops the build with a `WriterError` naming the
# page and the node: nothing is dropped or silently rewritten.
# ============================================================================
module SiteWriter

using Documenter: Documenter, MarkdownAST, Selectors
import JSON

"""
    WriterError <: Exception

A page holds something the site cannot show the way Documenter's HTML shows it — an
unsupported node, an output that is not text, a link to an anchor no page defines — or
would be served at the route of another page (`a.md` and `a/index.md`).
`page` is the page's path under `docs/src` (or ``make.jl `pages` `` for the
navigation), `what` names the node or the problem.
"""
struct WriterError <: Exception
    page::String
    what::String
end
Base.showerror(io::IO, e::WriterError) = print(io, "WriterError: ", e.page, ": ", e.what)

"""
    SiteMarkdoc(site)

Documenter output format (`makedocs(format = SiteMarkdoc(site))`) that writes the
manual into the site directory `site`:

- `src/app/<route>/page.md` for every page: `P.md` is served at `/P/`, `index.md`
  at `/` (the pages written by an earlier build are removed first);
- `src/navigation.json`: the sidebar, one group per section of `pages`, the
  top-level pages first as "Introduction";
- `public/assets/`: a copy of the manual's `assets/` directory.

Nothing is written unless every page converts.
"""
struct SiteMarkdoc <: Documenter.Writer
    site::String
    SiteMarkdoc(site::AbstractString) = new(abspath(site))
end

abstract type SiteFormat <: Documenter.FormatSelector end
Selectors.order(::Type{SiteFormat}) = 4.0
Selectors.matcher(::Type{SiteFormat}, fmt, _) = fmt isa SiteMarkdoc
Selectors.runner(::Type{SiteFormat}, fmt, doc) = Documenter.render(doc, fmt)

# ── The page model ──────────────────────────────────────────────────────────

abstract type Inline end
struct Text <: Inline; text::String end
struct Code <: Inline; code::String end
struct Emph <: Inline; content::Vector{Inline} end
struct Strong <: Inline; content::Vector{Inline} end
"`href`: `/route/`, `/route/#anchor`, `#anchor` on the same page, or an external URL."
struct Link <: Inline; href::String; content::Vector{Inline} end
"`src`: `/assets/…` for the manual's own images, else the external URL."
struct Image <: Inline; src::String; alt::String end

abstract type Block end
struct Paragraph <: Block; content::Vector{Inline} end
"A section (`id`: Documenter's anchor) or, inside a docstring, an unanchored heading."
struct Heading <: Block; level::Int; id::Union{String,Nothing}; content::Vector{Inline} end
struct CodeBlock <: Block; language::String; code::String end
"What an `@example` block printed or returned, as Documenter's HTML shows it."
struct Output <: Block; text::String end
@enum CalloutType note tip warning
const CALLOUT_TYPES = Dict("note" => note, "tip" => tip, "warning" => warning)
"`title` is `nothing` when the source kept the category's default title."
struct Callout <: Block; type::CalloutType; title::Union{String,Nothing}; body::Vector{Block} end
struct Details <: Block; summary::String; body::Vector{Block} end
"One `@docs` entry: `id` is what `@ref` resolves to; `bodies` are the binding's docstrings in order."
struct Docstring <: Block; id::String; name::String; kind::String; bodies::Vector{Vector{Block}} end
struct List <: Block; ordered::Bool; tight::Bool; items::Vector{Vector{Block}} end
struct Quote <: Block; body::Vector{Block} end
"`rows[1]` is the header row; `align` holds `:left`, `:center` or `:right` per column."
struct Table <: Block; align::Vector{Symbol}; rows::Vector{Vector{Vector{Inline}}} end
struct Rule <: Block end

"""
A page of the site. `ids` are the anchors it defines (headings, docstrings); `links`
the internal link targets it holds.
"""
struct Page
    key::String
    route::String
    title::String
    description::Union{String,Nothing}
    body::Vector{Block}
    ids::Vector{String}
    links::Vector{String}
end

# ── Reading Documenter's processed pages ────────────────────────────────────

struct Walk
    doc::Documenter.Document
    key::String                    # the page being read
    titles::Dict{String,String}    # page key => anchor of its title heading
    indocstring::Bool
    ids::Vector{String}
    links::Vector{String}
end
fail(w::Walk, what::AbstractString) = throw(WriterError(w.key, what))
unsupported(w::Walk, e) = fail(w, "unsupported node $(typeof(e))")

"Documenter's URL for a page key, kept as the site route: `guide/x.md` → `/guide/x/`, `index.md` → `/`."
function route(key::AbstractString)
    stem = basename(key) == "index.md" ? dirname(key) : first(splitext(key))
    isempty(stem) ? "/" : "/" * replace(stem, '\\' => '/') * "/"
end

"The page's title heading: the first node that renders anything, if it is a level-1 heading."
function title_node(page::Documenter.Page)
    nodes = collect(page.mdast.children)
    i = findfirst(n -> !(n.element isa Documenter.SetupNode), nodes)
    i !== nothing && nodes[i].element isa Documenter.AnchoredHeader && only(nodes[i].children).element.level == 1 ?
        nodes[i] : nothing
end

blocks(w::Walk, nodes) = Block[b for n in nodes for b in block(w, n.element, n)]

block(w::Walk, ::MarkdownAST.Paragraph, n) = [Paragraph(inlines(w, n))]
function block(w::Walk, e::Documenter.AnchoredHeader, n)
    h = only(n.children)
    2 <= h.element.level <= 4 ||
        fail(w, "a level-$(h.element.level) heading (the title is the page's one level-1 heading; sections use levels 2–4)")
    id = Documenter.anchor_label(e.anchor)
    push!(w.ids, id)
    [Heading(h.element.level, id, inlines(w, h))]
end
block(w::Walk, e::MarkdownAST.Heading, n) =
    w.indocstring ? [Heading(4, nothing, inlines(w, n))] : fail(w, "a level-$(e.level) heading nested inside a block")
block(w::Walk, e::MarkdownAST.CodeBlock, n) = [CodeBlock(language(w, e.info), e.code)]
block(w::Walk, ::Documenter.MultiOutput, n) = Block[example_part(w, c.element) for c in n.children]
block(w::Walk, ::Documenter.SetupNode, n) = Block[]
block(w::Walk, ::Documenter.DocsNodesBlock, n) = blocks(w, n.children)
function block(w::Walk, e::Documenter.DocsNode, n)
    push!(w.ids, e.anchor.id)
    inside = Walk(w.doc, w.key, w.titles, true, w.ids, w.links)
    [Docstring(e.anchor.id, Documenter.bindingstring(e.object.binding), Documenter.doccat(e.object),
               [blocks(inside, md.children) for md in e.mdasts])]
end
function block(w::Walk, e::MarkdownAST.Admonition, n)
    body = blocks(w, n.children)
    e.category == "details" && return [Details(e.title, body)]
    haskey(CALLOUT_TYPES, e.category) ||
        fail(w, "a `!!! $(e.category)` admonition (the site has note, tip, warning and details)")
    default = isempty(e.title) || e.title == uppercasefirst(e.category)
    [Callout(CALLOUT_TYPES[e.category], default ? nothing : e.title, body)]
end
block(w::Walk, e::MarkdownAST.List, n) = [List(e.type === :ordered, e.tight, [blocks(w, i.children) for i in n.children])]
block(w::Walk, ::MarkdownAST.BlockQuote, n) = [Quote(blocks(w, n.children))]
block(w::Walk, e::MarkdownAST.Table, n) =
    [Table(e.spec, [[inlines(w, cell) for cell in row.children] for row in MarkdownAST.tablerows(n)])]
block(w::Walk, ::MarkdownAST.ThematicBreak, n) = [Rule()]
block(w::Walk, e, n) = unsupported(w, e)

function language(w::Walk, info::AbstractString)
    lang = Documenter.codelang(info)
    lang == "output" && fail(w, "a code block in the language `output`, which the site keeps for example outputs")
    isempty(lang) ? "text" : lang
end

example_part(w::Walk, e::MarkdownAST.CodeBlock) = CodeBlock(language(w, e.info), e.code)
example_part(w::Walk, e::Documenter.MultiOutputElement) = Output(output_text(w, e.element))
example_part(w::Walk, e) = unsupported(w, e)

# The site shows an output as plain text, so examples run without ANSI colours.
Documenter.writer_supports_ansicolor(::SiteMarkdoc) = false

# Documenter's HTML shows the richest representation of a result (HTML, an image,
# LaTeX, Markdown) before plain text; the site shows text only.
function output_text(w::Walk, d::Dict{MIME,Any})
    rich = sort!([string(m) for m in keys(d) if m != MIME"text/plain"()])
    isempty(rich) || fail(w, "an @example output Documenter renders as $(join(rich, ", ")), not as text")
    text = d[MIME"text/plain"()]
    occursin('\e', text) && fail(w, "an @example output holding terminal escape codes")
    text
end
output_text(w::Walk, x) = fail(w, "an @example output of type $(typeof(x))")

inlines(w::Walk, n) = Inline[inline(w, c.element, c) for c in n.children]

isexternal(url::AbstractString) = startswith(url, r"https?://|mailto:")

inline(w::Walk, e::MarkdownAST.Text, n) = Text(e.text)
inline(w::Walk, e::MarkdownAST.Code, n) = Code(e.code)
# Julia's Markdown reads a double-backtick span (``x``) as inline LaTeX. The manual has no
# mathematics, so such a span is a code span written with two backticks, and is shown as code.
inline(w::Walk, e::MarkdownAST.InlineMath, n) = Code(e.math)
inline(w::Walk, ::MarkdownAST.Emph, n) = Emph(inlines(w, n))
inline(w::Walk, ::MarkdownAST.Strong, n) = Strong(inlines(w, n))
inline(w::Walk, e::MarkdownAST.Link, n) =
    isexternal(e.destination) ? Link(e.destination, inlines(w, n)) : fail(w, "an unresolved link to $(repr(e.destination))")
inline(w::Walk, e::Documenter.PageLink, n) = Link(href(w, e), inlines(w, n))
inline(w::Walk, e::MarkdownAST.Image, n) =
    isexternal(e.destination) ? Image(e.destination, alt(n)) : fail(w, "an image at $(repr(e.destination))")
inline(w::Walk, e::Documenter.LocalImage, n) =
    startswith(e.path, "assets/") ? Image("/" * e.path, alt(n)) :
    fail(w, "a local image outside assets/ ($(e.path)); the site serves only the copied assets")
inline(w::Walk, e, n) = unsupported(w, e)

alt(n) = Documenter.MDFlatten.mdflatten(n.children)

# A link to a page's title anchor points at the page itself: the title heading is
# the top of the page, and the site renders it without an anchor.
function href(w::Walk, e::Documenter.PageLink)
    key = Documenter.pagekey(w.doc, e.page)
    fragment = e.fragment == get(w.titles, key, nothing) ? "" : e.fragment
    target = key == w.key && !isempty(fragment) ? "#" * fragment :
             route(key) * (isempty(fragment) ? "" : "#" * fragment)
    push!(w.links, target)
    target
end

plain(xs::Vector{Inline}; images::Bool = true) = join(plain(x; images) for x in xs)
plain(x::Text; images::Bool = true) = x.text
plain(x::Code; images::Bool = true) = x.code
plain(x::Union{Emph,Strong,Link}; images::Bool = true) = plain(x.content; images)
plain(x::Image; images::Bool = true) = images ? x.alt : ""

"The first top-level paragraph's plain text, cut at a word to at most 160 characters."
function description(body::Vector{Block})
    for b in body
        b isa Paragraph || continue
        text = strip(replace(plain(b.content; images = false), r"\s+" => " "))
        isempty(text) && continue
        length(text) <= 160 && return String(text)
        cut = first(text, 159)
        space = findlast(' ', cut)
        return rstrip(space === nothing ? cut : cut[1:prevind(cut, space)]) * "…"
    end
    nothing
end

function read_page(doc::Documenter.Document, key::String, titles::Dict{String,String})
    w = Walk(doc, key, titles, false, String[], String[])
    page = doc.blueprint.pages[key]
    title = title_node(page)
    title === nothing && fail(w, "the page does not open with its title, a level-1 heading")
    body = blocks(w, (n for n in page.mdast.children if n !== title))
    Page(key, route(key), plain(inlines(w, only(title.children))), description(body), body, w.ids, w.links)
end

"Every page of the manual as the site's page model, its routes distinct and its internal links checked."
function read_site(doc::Documenter.Document)
    pagekeys = sort!(collect(keys(doc.blueprint.pages)))
    routes = Dict{String,String}()  # route => the first page served there
    for key in pagekeys
        first_key = get!(routes, route(key), key)
        first_key == key || throw(WriterError(key, "its route $(route(key)) is also the route of $first_key"))
    end
    titles = Dict{String,String}()
    for (key, page) in doc.blueprint.pages
        t = title_node(page)
        t === nothing || (titles[key] = Documenter.anchor_label(t.element.anchor))
    end
    pages = [read_page(doc, key, titles) for key in pagekeys]
    ids = Dict(p.route => Set(p.ids) for p in pages)
    for p in pages
        twice = unique(filter(id -> count(==(id), p.ids) > 1, p.ids))
        isempty(twice) || throw(WriterError(p.key, "anchors defined twice: $(join(twice, ", "))"))
        for target in p.links
            i = findfirst('#', target)
            r = i === nothing ? target : i == 1 ? p.route : target[1:i-1]
            haskey(ids, r) && (i === nothing || target[i+1:end] in ids[r]) ||
                throw(WriterError(p.key, "a link to $target, an anchor its page does not define"))
        end
    end
    pages
end

"The sidebar: the top-level pages as \"Introduction\", then one group per section of `pages`."
function navigation(doc::Documenter.Document, pages::Vector{Page})
    titles = Dict(p.key => p.title for p in pages)
    function link(nn::Documenter.NavNode)
        nn.page === nothing && throw(WriterError("make.jl `pages`", "a section nested in a section"))
        nn.visible || throw(WriterError("make.jl `pages`", "a hidden page, $(nn.page)"))
        (title = something(nn.title_override, titles[nn.page]), href = route(nn.page))
    end
    singles = [link(nn) for nn in doc.internal.navtree if nn.page !== nothing]
    sections = [(title = nn.title_override, links = [link(c) for c in nn.children])
                for nn in doc.internal.navtree if nn.page === nothing]
    isempty(singles) ? sections : [(title = "Introduction", links = singles); sections]
end

function Documenter.render(doc::Documenter.Document, w::SiteMarkdoc)
    @info "SiteWriter: writing the site's pages into $(w.site)."
    pages = read_site(doc)
    app = joinpath(w.site, "src", "app")
    files = [joinpath(app, split(p.route, '/'; keepempty = false)..., "page.md") => markdoc(p) for p in pages]
    nav = JSON.json(navigation(doc, pages); pretty = true) * "\n"
    if isdir(app)
        for (dir, _, names) in walkdir(app; topdown = false)
            "page.md" in names && rm(joinpath(dir, "page.md"))
            dir != app && isempty(readdir(dir)) && rm(dir)
        end
    end
    for (path, text) in files
        mkpath(dirname(path))
        write(path, text)
    end
    write(joinpath(mkpath(joinpath(w.site, "src")), "navigation.json"), nav)
    assets, public = joinpath(doc.user.root, doc.user.source, "assets"), joinpath(w.site, "public", "assets")
    rm(public; force = true, recursive = true)
    if isdir(assets)
        mkpath(dirname(public))
        cp(assets, public)
    end
    @info "SiteWriter: wrote $(length(files)) pages, the navigation and the assets."
    nothing
end

# ── Writing Markdoc ─────────────────────────────────────────────────────────
# Markdoc parses CommonMark (with GFM tables) plus `{% %}` tags. Every ASCII
# punctuation character in text is backslash-escaped, so no text can open a tag,
# emphasis, link, HTML or a list: text stays literal. A backslash inside a table
# row escapes the cell separator before inline parsing, so code and URLs in a
# cell write `|` as `\|`.

const ASCII_PUNCTUATION = r"[!-/:-@\[-`{-~]"

md(xs::Vector{Inline}, cell::Bool = false) = join(md(x, cell) for x in xs)
function md(x::Text, cell::Bool)
    occursin('\n', x.text) && throw(ArgumentError("a line break inside text $(repr(x.text))"))
    replace(x.text, ASCII_PUNCTUATION => s"\\\0")
end
md(x::Code, cell::Bool) = codespan(x.code, cell)
md(x::Emph, cell::Bool) = "*" * md(x.content, cell) * "*"
md(x::Strong, cell::Bool) = "**" * md(x.content, cell) * "**"
md(x::Link, cell::Bool) = "[" * md(x.content, cell) * "](" * destination(x.href, cell) * ")"
function md(x::Image, cell::Bool)
    # Markdoc takes an image's alt text from the raw label, escapes included.
    occursin(r"[\[\]\\\n]|\{%", x.alt) &&
        throw(ArgumentError("image alt text $(repr(x.alt)) holding a bracket, backslash, line break or `{%`"))
    "![" * (cell ? replace(x.alt, '|' => "\\|") : x.alt) * "](" * destination(x.src, cell) * ")"
end

longest_backtick_run(s::AbstractString) = maximum((length(m.match) for m in eachmatch(r"`+", s)); init = 0)

# A longer backtick run than the code holds; one space of padding on each side is
# stripped by the parser, so code that starts or ends with a backtick, or with a
# space at both ends, keeps its content.
function codespan(code::AbstractString, cell::Bool)
    isempty(code) && throw(ArgumentError("an empty inline code span"))
    occursin('\n', code) && throw(ArgumentError("a line break inside inline code $(repr(code))"))
    c = cell ? replace(code, '|' => "\\|") : code
    ticks = "`"^(longest_backtick_run(code) + 1)
    pad = startswith(c, '`') || endswith(c, '`') || (startswith(c, ' ') && endswith(c, ' ')) ? " " : ""
    string(ticks, pad, c, pad, ticks)
end

function destination(url::AbstractString, cell::Bool)
    any(iscntrl, url) && throw(ArgumentError("a link destination with a control character: $(repr(url))"))
    d = occursin(r"[\s()<>\\]", url) ? "<" * replace(url, r"[<>\\]" => s"\\\0") * ">" : url
    cell ? replace(d, '|' => "\\|") : d
end

"A Markdoc attribute string: its grammar escapes only `\"`, `\\`, `\\n`, `\\r` and `\\t`."
function attribute(s::AbstractString)
    any(c -> iscntrl(c) && !(c in "\n\r\t"), s) &&
        throw(ArgumentError("an attribute value with a control character: $(repr(s))"))
    '"' * replace(s, '\\' => "\\\\", '"' => "\\\"", '\n' => "\\n", '\r' => "\\r", '\t' => "\\t") * '"'
end

# Markdoc reads `{%` inside a fence as a tag unless the fence opts out.
function fence(language::AbstractString, code::AbstractString)
    occursin(r"^[A-Za-z0-9_+.#@-]+$", language) || throw(ArgumentError("a code block language $(repr(language))"))
    ticks = "`"^max(3, longest_backtick_run(code) + 1)
    info = occursin("{%", code) ? language * " {% process=false %}" : language
    [ticks * info; isempty(code) ? String[] : split(code, '\n'); ticks]
end

tag(name::String, attributes::Vector{Pair{String,String}}, body::Vector{Block}) =
    ["{% $name " * join(("$k=$(attribute(v))" for (k, v) in attributes), " ") * " %}"; ""; lines(body); ""; "{% /$name %}"]

function lines(bs::Vector{Block}; tight::Bool = false)
    out = String[]
    for b in bs
        isempty(out) || tight || push!(out, "")
        append!(out, lines(b))
    end
    out
end
lines(b::Paragraph) = [md(b.content)]
lines(b::Heading) = ["#"^b.level * " " * md(b.content) * (b.id === nothing ? "" : " {% id=$(attribute(b.id)) %}")]
lines(b::CodeBlock) = fence(b.language, b.code)
lines(b::Output) = fence("output", b.text)
lines(b::Callout) =
    tag("callout", ["type" => string(b.type); b.title === nothing ? Pair{String,String}[] : ["title" => b.title]], b.body)
lines(b::Details) = tag("details", ["summary" => b.summary], b.body)
lines(b::Docstring) =
    tag("docstring", ["id" => b.id, "name" => b.name, "kind" => b.kind], reduce((a, c) -> Block[a; Rule(); c], b.bodies))
lines(b::Quote) = [isempty(l) ? ">" : "> " * l for l in lines(b.body)]
lines(::Rule) = ["---"]
function lines(b::List)
    out = String[]
    for (i, item) in enumerate(b.items)
        marker = b.ordered ? "$i. " : "- "
        body = lines(item; tight = b.tight)
        isempty(body) && push!(body, "")
        i > 1 && !b.tight && push!(out, "")
        push!(out, marker * body[1])
        append!(out, (isempty(l) ? l : " "^length(marker) * l for l in body[2:end]))
    end
    out
end
function lines(b::Table)
    length(b.align) == length(first(b.rows)) ||
        throw(ArgumentError("a table with $(length(first(b.rows))) header cells for $(length(b.align)) columns"))
    row(cells) = "| " * join((md(c, true) for c in cells), " | ") * " |"
    rule = [a === :left ? ":---" : a === :center ? ":---:" : "---:" for a in b.align]
    [row(first(b.rows)); "| " * join(rule, " | ") * " |"; [row(r) for r in b.rows[2:end]]]
end

# The frontmatter's strings are JSON strings, which YAML reads as double-quoted scalars.
function frontmatter(p::Page)
    title = JSON.json(p.title)
    description = p.description === nothing ? String[] : ["    description: " * JSON.json(p.description)]
    ["---"; "title: $title"; "nextjs:"; "  metadata:"; "    title: $title"; description; "---"]
end

"The page as a Markdoc document."
function markdoc(p::Page)
    try
        join([frontmatter(p); ""; lines(p.body)], "\n") * "\n"
    catch err
        err isa ArgumentError ? throw(WriterError(p.key, err.msg)) : rethrow()
    end
end

end # module
