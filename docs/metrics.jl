"""
    DocMetrics

The acceptance numbers of the manual's provider entries, measured on the pages under
`docs/src`; `report` prints each beside its threshold.

- Other-provider share (`share`): of an entry's pages, of a reader's path (`reader_path`: the
  entry's pages and the concept pages they link to) and of each concept page, the fraction of
  prose lines (`prose_lines`) that hold a token of another provider (`provider_tokens`). The body
  of an Evidence box — a `!!! details` admonition whose title starts with "Evidence" — is no
  prose: it names the model and version a claim was measured on, provenance rather than
  guidance. Every other admonition is prose.
- Concept-page code (`concept_code_violations`): the endpoints and model ids named in code outside
  the binding lines of the page's binding block, and that block's breaches of its bounds.
- Placement ledgers (`check_ledger`): every passage of an audit placed exactly once, from its
  source — in a section whose body holds its witness, or deleted with a reason.

From the repository root:

    julia --startup-file=no --project=docs docs/metrics.jl [--baseline] [--ledgers DIR --atoms DIR]

exits 1 when a threshold is not met (`--baseline` prints the same numbers and exits 0) and 2 on
a usage error. `--atoms DIR` holds an audit's `<unit>.json` files; `--ledgers DIR` holds a
`<unit>.tsv` ledger for each of them with an `atoms` array. Without the two, the ledgers are not
checked.
"""
module DocMetrics

import JSON, UniLM
using Printf: @sprintf

isdefined(Main, :ProviderFacts) || Base.include(Main, joinpath(@__DIR__, "ProviderFacts.jl"))
using Main.ProviderFacts

export PageError, prose_lines, provider_tokens, share, reader_path, concept_code_violations, check_ledger, report

# The pages the command line measures.
const DOCSRC = joinpath(@__DIR__, "src")

"""
    PageError(file, line, message)

Thrown when a page cannot be measured as written — a fence never closed, an `@ref` that resolves
to nothing, an `@id` defined twice; `line` is the line at fault.
"""
struct PageError <: Exception
    file::String
    line::Int
    message::String
end
Base.showerror(io::IO, e::PageError) = print(io, "PageError: ", e.file, ":", e.line, ": ", e.message)

# ── Prose ────────────────────────────────────────────────────────────────────

const _FENCE = r"^\s*(`{3,}|~{3,})"

# The fenced blocks of `lines`, each the range from its opening to its closing fence line. A
# block opens on a run of three or more backticks or tildes at any indentation — so a fence in an
# admonition opens one too — and closes on a line that is that run and whitespace, as Julia's
# Markdown parser reads it; a block never closed is a `PageError`.
function _blocks(lines::AbstractVector{<:AbstractString}, file::String)::Vector{UnitRange{Int}}
    blocks = UnitRange{Int}[]
    open::Union{Tuple{String,Int},Nothing} = nothing        # the opening run and its line
    for (n, line) in pairs(lines)
        if isnothing(open)
            m = match(_FENCE, line)
            isnothing(m) || (open = (String(m[1]), n))
        elseif strip(line) == first(open)
            push!(blocks, last(open):n)
            open = nothing
        end
    end
    isnothing(open) || throw(PageError(file, last(open), "the fence opened here is never closed"))
    blocks
end

# `lines` with those of fenced blocks emptied, so every line keeps its number.
function _outside_code(lines::Vector{SubString{String}}, file::String)::Vector{SubString{String}}
    outside = copy(lines)
    foreach(block -> outside[block] .= SubString(""), _blocks(lines, file))
    outside
end
_outside_code(text::String, file::String)::Vector{SubString{String}} = _outside_code(split(text, '\n'), file)

const _EVIDENCE = r"^(\s*)!!!\s+details\s+\"Evidence"

# Whether each of `lines` is in the body of an Evidence box — a `!!! details` admonition outside
# code (`outside`: the lines with code emptied) whose title starts with "Evidence": the lines
# after its opening line that are blank or indented deeper than it.
function _in_evidence(lines::AbstractVector{<:AbstractString}, outside::AbstractVector{<:AbstractString})::BitVector
    body = falses(length(lines))
    depth::Union{Int,Nothing} = nothing                      # the open box's indentation
    for (n, line) in pairs(lines)
        if !isnothing(depth) && (isempty(strip(line)) || length(line) - length(lstrip(line)) > depth)
            body[n] = true
        else
            m = match(_EVIDENCE, outside[n])
            depth = isnothing(m) ? nothing : length(m[1])
        end
    end
    body
end

"""
    prose_lines(text::String; file::String = "string") -> Vector{Tuple{Int,String}}

The prose of a page, each line with its number: the non-blank lines outside fenced blocks, table
rows (lines starting with `|`) and the bodies of Evidence boxes (`!!! details` admonitions whose
title starts with "Evidence": provenance, not guidance). Headings and every other admonition's
body are prose. A fence opens on a run of three or more backticks or tildes at any indentation,
so a fence inside an admonition is code, and closes on the same run; one never closed is a
`PageError` naming `file` and the line.
"""
function prose_lines(text::String; file::String="string")::Vector{Tuple{Int,String}}
    lines = split(text, '\n')
    outside = _outside_code(lines, file)
    evidence = _in_evidence(lines, outside)
    [(n, String(l)) for (n, l) in pairs(outside) if !evidence[n] && !isempty(strip(l)) && !startswith(lstrip(l), '|')]
end

# ── Provider tokens ──────────────────────────────────────────────────────────

# Nothing of an identifier before a token.
const _BEFORE = "(?<![A-Za-z0-9_])"

# A token as one group of a pattern, and what may follow it: after a prefix — a token ending in
# `-` or `_`, such as `gpt-` — the rest of the identifier; after a token ending in a letter, an
# uppercase letter and the rest of a CamelCase identifier (`OllamaOptions`), or no part of an
# identifier (so `Jevons` holds no `Jev`); after any other token, no part of an identifier.
function _alternative(token::String)::String
    after = endswith(token, '-') || endswith(token, '_') ? "[A-Za-z0-9_]*" :
            isletter(last(token)) ? "(?:[A-Z][A-Za-z0-9_]*|(?![A-Za-z0-9_]))" : "(?![A-Za-z0-9_])"
    string("(", replace(token, r"[\\^$.|?*+()\[\]{}]" => s"\\\0"), ")", after)
end

# `tokens` longest first, and the alternation of their groups in that order, so the longest
# token wins where several start; group k matches the k-th token.
function _alternation(tokens::Vector{String})::Tuple{Vector{String},String}
    sorted = sort!(unique(tokens); by=t -> (-length(t), t))
    sorted, "(?:" * join(_alternative.(sorted), '|') * ")"
end

# A registry's tokens — each entry's `tokens` and `key_vars` — and the pattern finding them,
# built once per registry.
struct _Matcher
    tokens::Vector{String}               # longest first
    pattern::Regex                       # group k matches tokens[k]
    providers::Vector{Set{Symbol}}       # the providers of the entries listing tokens[k]
end
function _Matcher(registry::Vector{ProviderEntry})
    words(e::ProviderEntry) = [e.tokens; e.key_vars]
    tokens, alternation = _alternation([t for e in registry for t in words(e)])
    _Matcher(tokens, Regex(_BEFORE * alternation), [Set{Symbol}(e.provider for e in registry if t in words(e)) for t in tokens])
end

const _REGISTRY = _Matcher(REGISTRY)
const _NEUTRAL = Regex(_BEFORE * last(_alternation(NEUTRAL_TOKENS)))
const _DESTINATION = r"\]\([^)]*\)"

"""
    provider_tokens(line::String; registry = REGISTRY) -> Vector{Tuple{Set{Symbol},String}}

The provider tokens of `line` — the `tokens` and `key_vars` of the `registry` entries —, in order,
each with the providers of the entries that list it (more than one where entries of several
providers list it: `gpt-` is OpenAI's and Azure's). Link destinations — `](…)` — are dropped
first, then the `NEUTRAL_TOKENS`, matched like any token. A token matches case-sensitively where
no letter, digit or `_` precedes it (so `AZURE_OPENAI_API_KEY` holds no `OPENAI_API_KEY`). A token
ending in `-` or `_` is a prefix and takes in the rest of the identifier it starts (`gpt-` in
`gpt-5.4-mini`, `GPT-` in `GPT-6`); one ending in a letter takes in a CamelCase identifier it
starts (`Ollama` in `OllamaOptions`, `TypeSafe` in `TypeSafeProtocol`) and matches nowhere a
lowercase letter, digit or `_` follows it (`Jevons`); any other token matches only where none of
those follows it. Where several tokens start, the longest wins, and no span of the line is
counted twice: `Azure OpenAI` is one Azure token, and `OPENAIServiceEndpoint` keeps its
`ServiceEndpoint`.
"""
function provider_tokens(line::String; registry::Vector{ProviderEntry}=REGISTRY)::Vector{Tuple{Set{Symbol},String}}
    m = registry === REGISTRY ? _REGISTRY : _Matcher(registry)
    text = replace(replace(line, _DESTINATION => "]"), _NEUTRAL => " ")
    [(copy(m.providers[k]), m.tokens[k]) for k in (findfirst(!isnothing, x.captures)::Int for x in eachmatch(m.pattern, text))]
end

# ── Shares ───────────────────────────────────────────────────────────────────

# The provider of the entry `reader`; `nothing` for a concept page.
_provider(::Nothing, ::Vector{ProviderEntry})::Nothing = nothing
function _provider(reader::Symbol, registry::Vector{ProviderEntry})::Symbol
    i = findfirst(e -> e.key === reader, registry)
    isnothing(i) && throw(ArgumentError(
        "no provider entry :$reader; the entries are $(join((repr(e.key) for e in registry), ", "))"))
    registry[i].provider
end

# Of the prose lines of the files `pages`: how many hold a token none of whose providers is the
# reader's — on a concept page any token, as no provider is `nothing` —, and how many there are.
function _tally(pages::Vector{String}, reader::Union{Symbol,Nothing}, registry::Vector{ProviderEntry})::Tuple{Int,Int}
    own = _provider(reader, registry)
    lines = [l for p in pages for (_, l) in prose_lines(read(p, String); file=p)]
    count(l -> any(((providers, _),) -> own ∉ providers, provider_tokens(l; registry)), lines), length(lines)
end

"""
    share(pages::Vector{String}, reader::Union{Symbol,Nothing}; registry = REGISTRY) -> Float64

The other-provider share of the files `pages`: of their prose lines (`prose_lines`), the
fraction holding a token (`provider_tokens`) none of whose providers is the provider of the
entry `reader`. A line counts once however many tokens it holds, and an entry's tokens never
count against another entry of the same provider. For a concept page, `reader = nothing`: every
token counts. An `ArgumentError` when the pages hold no prose line.
"""
function share(pages::Vector{String}, reader::Union{Symbol,Nothing}; registry::Vector{ProviderEntry}=REGISTRY)::Float64
    other, total = _tally(pages, reader, registry)
    total > 0 || throw(ArgumentError("no prose line to measure in $(join(pages, ", "))"))
    other / total
end

# ── Reader paths ─────────────────────────────────────────────────────────────

const _CONCEPTS = Set{String}(p for pages in values(CONCEPT_PAGES) for p in pages)
const _HEADING = r"^(#{1,6})\s"
const _ID = r"\(@id\s+([^)\s]+)\s*\)"
# A link `[text](@ref)` or `[text](@ref target)`; text and target may wrap across lines.
const _REF = r"\[((?:[^\[\]]|\[[^\[\]]*\])*)\]\(@ref(?:\s+([^)\s][^)]*?))?\s*\)"

# The id a heading line defines with `(@id …)`; `nothing` for any other line.
function _heading_id(line::AbstractString)::Union{String,Nothing}
    occursin(_HEADING, line) || return nothing
    m = match(_ID, line)
    isnothing(m) ? nothing : String(m[1])
end

# Each id an `(@id …)` heading under `docsrc` defines, with its page relative to `docsrc`; an id
# defined twice is a `PageError` at its second heading.
function _anchors(docsrc::String)::Dict{String,String}
    anchors = Dict{String,String}()
    for (root, _, files) in walkdir(docsrc), f in files
        endswith(f, ".md") || continue
        file = joinpath(root, f)
        for (n, line) in pairs(_outside_code(read(file, String), file))
            id = _heading_id(line)
            isnothing(id) && continue
            haskey(anchors, id) && throw(PageError(file, n, "the id $id is already defined on $(anchors[id])"))
            anchors[id] = relpath(file, docsrc)
        end
    end
    anchors
end

# Whether `path` (`["UniLM", "Chat"]`) names a binding reachable from the module `m`.
_binding(m::Module, path::AbstractVector{<:AbstractString})::Bool =
    isdefined(m, Symbol(first(path))) &&
    (length(path) == 1 || (x = getfield(m, Symbol(first(path)))) isa Module && _binding(x, path[2:end]))

# Whether a reference's target names a docstring's binding as UniLM sees it — `Chat`,
# `UniLM.TypeSafeProtocol`, `@branch`; a signature `f(::T)` names `f`.
_docstring(target::AbstractString)::Bool = _binding(UniLM, split(first(split(target, '(')), '.'))

"""
    reader_path(e::ProviderEntry; docsrc::String = DOCSRC) -> Vector{String}

The pages a reader of the entry `e` reads, relative to `docsrc`: its pages, then each concept
page (`CONCEPT_PAGES`) they link to with `@ref`, once, in the order first linked. A link's target
is its `@ref` id — for a bare `(@ref)`, its text without backticks — and resolves to the page
whose heading defines it with `(@id …)`. A link to a docstring (a target naming a binding of
UniLM) or to a page that is no concept page adds nothing; one resolving to nothing is a
`PageError` naming the page and the line. Links in fenced blocks are code; links in table rows
count. Pages of `e` that do not exist are listed and not read.
"""
reader_path(e::ProviderEntry; docsrc::String=DOCSRC)::Vector{String} = _reader_path(e, docsrc, _anchors(docsrc))

function _reader_path(e::ProviderEntry, docsrc::String, anchors::Dict{String,String})::Vector{String}
    linked = String[]
    for page in e.pages
        file = joinpath(docsrc, page)
        isfile(file) || continue
        text = join(_outside_code(read(file, String), file), '\n')
        for x in eachmatch(_REF, text)
            target = something(x[2], strip(x[1], '`'))
            to = get(anchors, target, nothing)
            if !isnothing(to)
                to in _CONCEPTS && push!(linked, to)
            elseif !_docstring(target)
                line = 1 + count(==(UInt8('\n')), view(codeunits(text), 1:(x.offset - 1)))
                throw(PageError(file, line, "@ref $target: no (@id $target) heading under $docsrc and no binding of UniLM"))
            end
        end
    end
    unique!([e.pages; linked])
end

# ── Concept-page code ────────────────────────────────────────────────────────

const _ENDPOINT = r"\b(OPENAI|AZURE|GEMINI|GEMINIOpenAI|ANTHROPIC|TYPESAFE)ServiceEndpoint\b|\b(Ollama|DeepSeek|GenericOpenAI|Mistral)Endpoint\b"
# The registry's model-id prefixes: its tokens that end in `-`, and those of the shared prefix
# table, which holds `o1`, `o3` and `o4` too.
const _MODEL_PREFIXES = [t for e in REGISTRY for t in e.tokens if endswith(t, '-') || any(ps -> t in ps, ProviderFacts._MODEL_PREFIXES)]
# A quoted model id: a string that is an id with such a prefix, no whitespace in it.
const _MODEL_ID = Regex("\"" * last(_alternation(_MODEL_PREFIXES)) * "[^\"\\s]*\"")
const _PROVIDER_CODE = Regex(_ENDPOINT.pattern * "|" * _MODEL_ID.pattern)
const _BINDING = r"^\s*service, model ="
# A binding line: names assigned endpoints — called or not — and quoted ids, and nothing else.
const _VALUE = "(?:(?:" * _ENDPOINT.pattern * ")(?:\\((?:[^()\"]|\"[^\"]*\")*\\))?|\"[^\"\\s]+\")"
const _BINDING_LINE = Regex("^\\s*(?:const\\s+)?[A-Za-z_][A-Za-z0-9_]*(?:\\s*,\\s*[A-Za-z_][A-Za-z0-9_]*)*\\s*=\\s*" *
                            _VALUE * "(?:\\s*,\\s*" * _VALUE * ")*\\s*;?\\s*(?:#.*)?\$")

# The chapter of the concept page at the path `page`: the one whose `CONCEPT_PAGES` end the path;
# a language-model page's for any other path.
_chapter(page::String)::Chapter =
    any(p -> page == p || endswith(page, "/" * p), CONCEPT_PAGES[ProviderFacts.decisions]) ? ProviderFacts.decisions :
                                                                                             ProviderFacts.llm

# The breaches of a binding block's binding lines `bound` on `page`, each with its line: an
# endpoint beyond the one a language-model concept page binds, or the two a Decisions one binds;
# a registry-prefixed model id of a provider none of the bound endpoints serves.
function _breaches(page::String, lines::AbstractVector{<:AbstractString}, bound::Vector{Int})::Vector{Tuple{Int,String}}
    most, kind = _chapter(page) == ProviderFacts.decisions ? (2, "Decisions") : (1, "language-model")
    named = String[]                                         # the endpoints bound, in order
    found = Tuple{Int,String}[]
    for n in bound, x in eachmatch(_ENDPOINT, lines[n])
        x.match in named && continue
        push!(named, x.match)
        length(named) > most && push!(found, (n, "$page:$n: $(x.match): a $kind concept page's binding block binds " *
                                                 "at most $most endpoint$(most == 1 ? "" : "s")"))
    end
    providers(s::AbstractString) = Set{Symbol}(p for (ps, _) in provider_tokens(String(s)) for p in ps)
    served = reduce(union, providers.(named); init=Set{Symbol}())
    for n in bound, x in eachmatch(_MODEL_ID, lines[n])
        isdisjoint(providers(strip(x.match, '"')), served) &&
            push!(found, (n, "$page:$n: $(x.match): a model of no provider the binding block binds"))
    end
    found
end

"""
    concept_code_violations(page::String) -> Vector{String}

The provider code of the concept page at the path `page`, as `page:line: match` in line order.
The page's binding block is the first fenced block with a line starting `service, model =`; its
binding lines — assignments of endpoints, called or not, and quoted ids, such as
`llm, llm_model = OPENAIServiceEndpoint, "gpt-5.4-mini"` or `pinned = "jev-1.13.0"` — are exempt.
Every other line of code, before the binding block or after it or in it, `@setup` blocks
included, is checked for each endpoint it names (`OPENAIServiceEndpoint`, `MistralEndpoint`, …,
with or without a call) and each quoted model id with a registry model-id prefix
(`"claude-haiku-5-5"`; a string such as `"gpt-5.2 row: "` is no model id). The binding lines are
held to their own bounds: at most one endpoint on a language-model concept page and two on a
Decisions one (the chapter whose `CONCEPT_PAGES` end the path), and no registry-prefixed model id
of a provider no bound endpoint serves; each breach names its line. Prose is not checked here:
its provider tokens count in the page's share.
"""
function concept_code_violations(page::String)::Vector{String}
    lines = split(read(page, String), '\n')
    blocks = _blocks(lines, page)
    binding = findfirst(b -> any(n -> occursin(_BINDING, lines[n]), b), blocks)
    bound = isnothing(binding) ? Int[] : filter(n -> occursin(_BINDING_LINE, lines[n]), collect(blocks[binding]))
    found = Tuple{Int,String}[(n, "$page:$n: $(x.match)") for b in blocks for n in b if n ∉ bound
                              for x in eachmatch(_PROVIDER_CODE, lines[n])]
    last.(sort!(append!(found, _breaches(page, lines, bound)); by=first))
end

# ── Placement ledgers ────────────────────────────────────────────────────────

const _LEDGER_HEADER = "atom_id\tsource\tdestination\twitness"

# The fewest characters a witness has, after whitespace normalisation: enough to quote a phrase of
# the passage, more than a heading's few words.
const _WITNESS_FLOOR = 24

_normalized(s::AbstractString)::String = join(split(s), ' ')

# The body of the section of the page `file` whose heading defines `(@id id)`: the lines below
# that heading, up to the next heading of its level or a higher one, headings in fenced blocks
# aside; `nothing` when no heading defines `id`.
function _section(file::String, id::String)::Union{String,Nothing}
    text = read(file, String)
    outside = _outside_code(text, file)
    level(n) = (m = match(_HEADING, outside[n]); isnothing(m) ? 0 : length(m[1]))
    start = findfirst(n -> _heading_id(outside[n]) == id, eachindex(outside))
    isnothing(start) && return nothing
    stop = something(findnext(n -> 0 < level(n) <= level(start), eachindex(outside), start + 1), lastindex(outside) + 1)
    join(view(split(text, '\n'), (start + 1):(stop - 1)), '\n')
end

# What is wrong with placing a passage at `destination` with `witness`, or `nothing`.
function _placement(docsrc::String, destination::AbstractString, witness::AbstractString)::Union{String,Nothing}
    w = _normalized(witness)
    destination == "deleted" && return isempty(w) ? "deleted without a reason in the witness column" : nothing
    length(w) >= _WITNESS_FLOOR || return "a witness of $(length(w)) characters; a witness has at least $_WITNESS_FLOOR"
    m = match(r"^([^#]+\.md)#(.+)$", destination)
    isnothing(m) && return "the destination $(repr(destination)) is neither page.md#id nor deleted"
    file = joinpath(docsrc, m[1])
    isfile(file) || return "$(m[1]) is no page under $docsrc"
    section = _section(file, String(m[2]))
    isnothing(section) && return "no heading of $(m[1]) defines (@id $(m[2]))"
    occursin(w, _normalized(section)) ? nothing : "the witness is not in the body of $destination"
end

"""
    check_ledger(tsv::String, docsrc::String, expected::Vector{Pair{String,String}}) -> Vector{String}

The problems of the placement ledger at the path `tsv`, each naming the file and, for a row,
its line; empty when the ledger places every atom of `expected` — `id => location` pairs —
exactly once, from its location. The ledger is tab-separated: the header
`atom_id source destination witness`, then a row per atom — its id `<unit>:<index>` (the atom's
1-based index in the `atoms` array of `<unit>.json`), its source (the atom's `location`), and
where it went: `page.md#id` (a page under `docsrc` and the `@id` of one of its headings), whose
section body below the heading must hold the witness — at least 24 characters, compared with
whitespace runs made single spaces —, or `deleted`, with the reason in the witness column. A
row with an id not in `expected`, a source other than its atom's location, or other than four
columns is a problem.
"""
function check_ledger(tsv::String, docsrc::String, expected::Vector{Pair{String,String}})::Vector{String}
    lines = readlines(tsv)
    get(lines, 1, "") == _LEDGER_HEADER || return ["$tsv:1: the first line is not the header $(repr(_LEDGER_HEADER))"]
    locations = Dict(expected)
    problems = String[]
    rows = Dict{String,Vector{Int}}()                       # an atom id => the lines placing it
    for n in 2:lastindex(lines)
        fields = split(lines[n], '\t')
        if length(fields) != 4
            push!(problems, "$tsv:$n: $(length(fields)) columns, not the header's 4")
            continue
        end
        id, source, destination, witness = fields
        push!(get!(Vector{Int}, rows, id), n)
        location = get(locations, id, nothing)
        isnothing(location) && push!(problems, "$tsv:$n: $id is no atom of this ledger")
        isnothing(location) || source == location || push!(problems, "$tsv:$n: $id comes from $location, not $source")
        problem = _placement(docsrc, destination, witness)
        isnothing(problem) || push!(problems, "$tsv:$n: $id: $problem")
    end
    for (id, _) in expected
        at = get(rows, id, Int[])
        isempty(at) && push!(problems, "$tsv: $id has no row")
        length(at) > 1 && push!(problems, "$tsv: $id has $(length(at)) rows, at lines $(join(at, ", "))")
    end
    problems
end

# The location an audit atom names — its ledger row's source —; an `ArgumentError` naming the
# file and the atom when it has none.
function _location(atom, file::String, i::Int)::String
    location = get(atom, "location", nothing)
    location isa String || throw(ArgumentError("$file: atom $i has no location"))
    location
end

# The problems of the ledgers in the directory `ledgers` against the audit units in `atoms`: each
# `<unit>.json` with an `atoms` array expects `<unit>.tsv` to place `<unit>:1` … `<unit>:<n>`,
# each from its atom's `location`.
function _ledger_problems(ledgers::String, atoms::String, docsrc::String)::Vector{String}
    units = [(chopsuffix(f, ".json"), JSON.parsefile(joinpath(atoms, f))) for f in readdir(atoms) if endswith(f, ".json")]
    lists = [(unit, json["atoms"]::AbstractVector) for (unit, json) in units if haskey(json, "atoms")]
    isempty(lists) && return ["$atoms: no <unit>.json holds an atoms array"]
    problems = String[]
    for (unit, list) in lists
        tsv = joinpath(ledgers, unit * ".tsv")
        expected = ["$unit:$i" => _location(atom, "$unit.json", i) for (i, atom) in pairs(list)]
        append!(problems, isfile(tsv) ? check_ledger(tsv, docsrc, expected) :
                                        ["$tsv: missing; it places the $(length(list)) atoms of $unit.json"])
    end
    problems
end

# ── The report ───────────────────────────────────────────────────────────────

"""
    Threshold

What a share is held to: `label`, what it measures, and `at_most`, the largest fraction of
other-provider prose lines it accepts.
"""
struct Threshold
    label::String
    at_most::Rational{Int}
end

const ENTRY = Threshold("entry", 1 // 100)       # an entry's own pages
const PATH = Threshold("path", 5 // 100)         # a reader's path: an entry's pages and the concept pages they link
const CONCEPT = Threshold("concept", 2 // 100)   # one concept page, where every provider's token counts

"""
    Measurement

A share as the report measures it: the `threshold` it is held to; what it is `of` (an entry's
key or a concept page); its `pages`, relative to docs/src, and those of them `absent`; and of the
prose lines of the pages that exist, the `other` ones — holding another provider's token — and
the `total`.
"""
struct Measurement
    threshold::Threshold
    of::String
    pages::Vector{String}
    absent::Vector{String}
    other::Int
    total::Int
end

# Whether a measurement meets its threshold: every page exists, there is prose, and at most
# `at_most` of it names another provider. An absent page is never a 0% share.
holds(m::Measurement)::Bool = isempty(m.absent) && m.total > 0 && m.other // m.total <= m.threshold.at_most

function _measure(threshold::Threshold, of::String, pages::Vector{String}, reader::Union{Symbol,Nothing},
                  docsrc::String)::Measurement
    present = filter(p -> isfile(joinpath(docsrc, p)), pages)
    other, total = _tally([joinpath(docsrc, p) for p in present], reader, REGISTRY)
    Measurement(threshold, of, pages, setdiff(pages, present), other, total)
end

function _row(m::Measurement)::String
    measured = m.total > 0 ? @sprintf("%.2f%% (%d/%d)", 100 * m.other / m.total, m.other, m.total) :
               m.absent == m.pages ? "no pages" : "no prose lines"
    absent = isempty(m.absent) || m.absent == m.pages ? "" : "; missing " * join(m.absent, ", ")
    string(rpad(m.threshold.label, 9), rpad(m.of, 36), rpad(measured, 20), "≤ ", Int(100 * m.threshold.at_most), "%  ",
           holds(m) ? "ok" : "fails", absent)
end

"""
    report(io::IO; docsrc::String, ledgers = nothing, atoms = nothing, baseline::Bool = false) -> Bool

Prints to `io` the acceptance numbers of the pages under `docsrc`: a line per entry (`ENTRY`: at
most 1% of its prose lines name another provider), per reader's path (`PATH`: 5%) and per concept
page (`CONCEPT`: 2%, every provider counting), each with its share and threshold — an entry or a
page that does not exist yet shows "no pages" and fails —; then each endpoint and model id a
concept page names in code outside the binding lines of its binding block, and each breach of
that block's bounds (`concept_code_violations`); then the problems of the ledgers in the
directory `ledgers` against the audit units in the directory `atoms` (both or neither; without
them, "ledgers: not checked"). Returns whether every share holds, no concept page names provider
code and no ledger has a problem; with `baseline = true` it prints the same and returns `true`.
"""
function report(io::IO; docsrc::String, ledgers::Union{String,Nothing}=nothing, atoms::Union{String,Nothing}=nothing,
                baseline::Bool=false)::Bool
    isnothing(ledgers) == isnothing(atoms) ||
        throw(ArgumentError("the ledgers are checked against the atoms: pass both directories or neither"))
    anchors = _anchors(docsrc)
    concepts = [p for c in instances(Chapter) for p in CONCEPT_PAGES[c]]
    measurements = [[_measure(ENTRY, string(e.key), e.pages, e.key, docsrc) for e in REGISTRY];
                    [_measure(PATH, string(e.key), _reader_path(e, docsrc, anchors), e.key, docsrc) for e in REGISTRY];
                    [_measure(CONCEPT, p, [p], nothing, docsrc) for p in concepts]]
    code = [v for p in concepts if isfile(joinpath(docsrc, p)) for v in concept_code_violations(joinpath(docsrc, p))]
    problems = isnothing(ledgers) ? nothing : _ledger_problems(ledgers, atoms, docsrc)
    println(io, "Other-provider share of prose lines (non-blank lines outside fenced blocks and table rows)")
    foreach(m -> println(io, _row(m)), measurements)
    println(io, "Concept-page code outside the binding lines, and binding-block breaches: ", length(code))
    foreach(v -> println(io, "  ", v), code)
    println(io, "ledgers: ", isnothing(problems) ? "not checked" : "$(length(problems)) problems")
    foreach(p -> println(io, "  ", p), something(problems, String[]))
    pass = all(holds, measurements) && isempty(code) && isempty(something(problems, String[]))
    println(io, baseline ? "baseline: printed, not enforced" : pass ? "pass" : "fail")
    baseline || pass
end

# ── Command line ─────────────────────────────────────────────────────────────

const USAGE = "usage: julia --startup-file=no --project=docs docs/metrics.jl [--baseline] [--ledgers DIR --atoms DIR]"

# The options `args` give — each flag at most once, `--ledgers` and `--atoms` with a directory
# and together — or why they are not a usage of this script.
function _options(args::AbstractVector{String}, given::Dict{String,String}=Dict{String,String}())::Union{Dict{String,String},String}
    isempty(args) &&
        return haskey(given, "--ledgers") == haskey(given, "--atoms") ? given : "--ledgers and --atoms come together"
    flag = first(args)
    haskey(given, flag) && return "$flag is given twice"
    flag == "--baseline" && return _options(args[2:end], push!(given, flag => ""))
    flag in ("--ledgers", "--atoms") || return "unknown argument $(repr(flag))"
    length(args) >= 2 && isdir(args[2]) || return "$flag takes a directory"
    _options(args[3:end], push!(given, flag => args[2]))
end

"""
    main(args::Vector{String}) -> Int

The command line: prints the report of `docs/src` and returns the exit code — 0 when every
threshold holds or with `--baseline`, 1 when one does not, 2 on a usage error.
"""
function main(args::Vector{String})::Int
    options = _options(args)
    options isa String && (println(stderr, options, '\n', USAGE); return 2)
    report(stdout; docsrc=relpath(DOCSRC), ledgers=get(options, "--ledgers", nothing),
           atoms=get(options, "--atoms", nothing), baseline=haskey(options, "--baseline")) ? 0 : 1
end

end # module

if abspath(PROGRAM_FILE) == @__FILE__
    exit(DocMetrics.main(ARGS))
end
