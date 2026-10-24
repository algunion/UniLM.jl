"""
    DocMetrics

The acceptance numbers of the manual's provider entries, measured on the pages under
`docs/src`; `report` prints each beside its threshold.

- Other-provider share (`share`): of an entry's pages, of a reader's path (`reader_path`: the
  entry's pages and the concept pages they link to) and of each concept page, the fraction of
  prose lines (`prose_lines`) that hold a token of another provider (`provider_tokens`). A page is
  read as Julia's Markdown reads it — the stdlib parser Documenter runs on every page —: its
  fenced blocks (`fenced_blocks`) and its table rows are no prose, and a page the parser reads
  otherwise is refused with a `PageError`. The body of an Evidence box — a `!!! details`
  admonition whose title starts with "Evidence" — is no prose either: it names the model and
  version a claim was measured on, provenance rather than guidance; the report counts, per page,
  the lines it leaves out. Every other admonition is prose.
- Concept-page code (`concept_code_violations`): the endpoints and model ids named in code outside
  the binding lines of the page's binding block, and that block's breaches of its bounds.
- Placement ledgers (`check_ledger`): every passage of an audit placed exactly once, from its
  source — in a section whose body holds its witness, or deleted with a reason.

What the numbers cannot see:

- Tables: a table row names no provider in prose, whatever it holds (a line starting with `|`
  must be a row of a table Julia's Markdown reads; a row it reads without a leading `|` counts as
  prose).
- Generated text: an `@eval` block is code, and the table it returns is not in the page.
- Evidence boxes: what one holds is not measured (the factual reviews check that it holds
  measurements only). A box in a list item or an admonition, or indented below a block quote, is
  prose.
- Words that are no token: a model id without a registry prefix (`qwen3:4b`,
  `embeddinggemma-2:740m-q8_0`), a provider's API names ("Interactions API", "generateContent",
  "Responses API", "Messages API"), Claude's model families without "Claude" ("Opus", "Sonnet",
  "Haiku"), a name inside an identifier or after `_` (`_Claude_`), and a name in lower case
  ("openai").
- Indented code: four spaces of indentation outside a list item or an admonition make code to
  Julia and prose here.
- Placement: a witness proves that its text is in the section, not that it is the atom's passage.

The rules are fixed before the pages they measure are written: a page changes to meet them, never
the rules to meet a page.

From the repository root:

    julia --startup-file=no --project=docs docs/metrics.jl [--baseline] [--entry KEY] [--ledgers DIR --atoms DIR [--unit NAME]]

exits 1 when a threshold is not met (`--baseline` prints the same numbers and exits 0) and 2 on
a usage error. `--entry KEY` measures the entry `KEY` alone: its row and its reader's path. `--atoms
DIR` holds an audit's `<unit>.json` files; `--ledgers DIR` holds a `<unit>.tsv` ledger for each of
them with an `atoms` array, and `--unit NAME` checks the ledger of the unit `NAME` alone. Without
the two, the ledgers are not checked.
"""
module DocMetrics

import JSON, UniLM
using Markdown
using Printf: @sprintf

isdefined(Main, :ProviderFacts) || Base.include(Main, joinpath(@__DIR__, "ProviderFacts.jl"))
using Main.ProviderFacts

export PageError, fenced_blocks, outside_fences, prose_lines, provider_tokens, share, reader_path,
       concept_code_violations, check_ledger, report

# The pages the command line measures.
const DOCSRC = joinpath(@__DIR__, "src")

"""
    PageError(file, line, message)

Thrown when a page cannot be measured as written — a fence never closed, a fenced block or a
table row Julia's Markdown reads otherwise, an `@ref` that resolves to nothing, an `@id` defined
twice; `line` is the line at fault.
"""
struct PageError <: Exception
    file::String
    line::Int
    message::String
end
Base.showerror(io::IO, e::PageError) = print(io, "PageError: ", e.file, ":", e.line, ": ", e.message)

# ── Fenced blocks ────────────────────────────────────────────────────────────

# Spaces, tabs and carriage returns: what pads an opening fence (`startswith(stream, "```"; padding
# = true)`) and all a blank line holds (`blankline`), in Julia 1.13's stdlib Markdown/src/parse/
# util.jl:11-21, :59-63 and :70-80.
const _WHITESPACE = " \t\r"

# An opening fence: what pads it, its fence character, the length of its run and its info string.
const _Opener = @NamedTuple{padding::String, char::Char, run::Int, info::String}

# The opening fence `line` is, or `nothing`, as the julia flavor's `fencedcode` reads one (stdlib
# Markdown/src/GitHub/GitHub.jl:5-16): after padding, three backticks or three tildes; the run is
# those three and each fence character that starts the rest of the line once stripped, and the
# info string is that stripped rest without them. An info string that holds the fence character
# opens no block (`ch in flavor && return false`: inline code, to Julia).
function _opener(line::AbstractString)::Union{_Opener,Nothing}
    i = something(findfirst(!in(_WHITESPACE), line), ncodeunits(line) + 1)
    rest = SubString(line, i)
    startswith(rest, "```") || startswith(rest, "~~~") || return nothing
    char, trailing = first(rest), strip(SubString(rest, 4))
    info = lstrip(trailing, char)
    char in info && return nothing
    (padding=String(SubString(line, 1, prevind(line, i))), char=char, run=3 + length(trailing) - length(info),
     info=String(info))
end

# How many code units of `line` the closing fence of the block `o` opens takes, or `nothing` when
# `line` does not close it: the opening fence's padding, then exactly its run, no fence character
# after it (GitHub.jl:19-22). Julia closes on a run at the start of a line of the block's context —
# an admonition's body or a list item's, read without its indentation —, which the opening fence's
# padding stands for here. Text may follow the run: Julia reads it as the start of the next block.
function _closer(o::_Opener, line::AbstractString)::Union{Int,Nothing}
    fence = o.padding * string(o.char)^o.run
    startswith(line, fence) || return nothing
    n = ncodeunits(fence)
    n < ncodeunits(line) && codeunit(line, n + 1) == UInt8(o.char) ? nothing : n
end

"A fenced block: its lines, from the opening fence to the closing one; the code units of the last line the closing fence takes; its info string."
struct _Fenced
    lines::UnitRange{Int}
    closing::Int
    info::String
end

# The fenced blocks of `lines` by the rules of `_opener` and `_closer`; a fence never closed is a
# `PageError` (to Julia it opens no block, GitHub.jl:35, and the lines below it are text).
function _scan(lines::AbstractVector{<:AbstractString}, file::String)::Vector{_Fenced}
    blocks = _Fenced[]
    open::Union{Tuple{_Opener,Int},Nothing} = nothing        # the opening fence and its line
    for (n, line) in pairs(lines)
        if isnothing(open)
            o = _opener(line)
            isnothing(o) || (open = (o, n))
        else
            c = _closer(first(open), line)
            isnothing(c) || (push!(blocks, _Fenced(last(open):n, c, first(open).info)); open = nothing)
        end
    end
    isnothing(open) || throw(PageError(file, last(open), "the fence opened here is never closed"))
    blocks
end

const _PARSING = ReentrantLock()

# What Julia's Markdown reads in `text` — `Markdown.parse`, as Documenter reads a page
# (Documenter/src/documents.jl:54) —, in document order: the info string and code of each fenced
# block (a `math` block's formula), and the text of each table. The parser reports them itself:
# for the parse, the julia flavor's `fencedcode` and `github_table` (stdlib Markdown/src/Julia/
# Julia.jl:10-11) are wrapped to keep what they read, and every nested parse — an admonition's, a
# list item's, a block quote's, a footnote's — reads with that flavor.
function _julia_reads(text::String)::@NamedTuple{fences::Vector{Tuple{String,String}}, tables::Vector{String}}
    fences, tables = Tuple{String,String}[], String[]
    function fenced(stream::IO, md::Markdown.MD)::Bool
        Markdown.fencedcode(stream, md) || return false
        x = last(md.content)
        push!(fences, x isa Markdown.LaTeX ? ("math", x.formula) : (x.language, x.code))
        true
    end
    function table(stream::IO, md::Markdown.MD)::Bool
        start = position(stream)
        Markdown.github_table(stream, md) || return false
        stop = position(stream)
        push!(tables, String(read(seek(stream, start), stop - start)))     # which leaves it at `stop`
        true
    end
    flavor = Markdown.julia
    f = only(findall(==(Markdown.fencedcode), flavor.breaking))
    t = only(findall(==(Markdown.github_table), flavor.regular))
    @lock _PARSING try
        flavor.breaking[f], flavor.regular[t] = fenced, table
        Markdown.parse(text)
    finally
        flavor.breaking[f], flavor.regular[t] = Markdown.fencedcode, Markdown.github_table
    end
    (fences=fences, tables=tables)
end

_stripped(lines)::String = join(strip.(lines), '\n')

# How a fenced block of Julia's reads in a message: its info string and first line.
_described(block::Union{Tuple{String,String},Nothing})::String =
    isnothing(block) ? "none" : "```$(first(block)) beginning $(repr(first(split(last(block), '\n'))))"

# `blocks`, the fenced blocks of the page `text` (its `lines`; `outside`, the lines outside them),
# once Julia's Markdown reads the page alike: the same fenced blocks in order — info string and
# code, each line without its indentation —, and every line outside them that starts with `|` a row
# of a table it reads, in order. Else a `PageError` at the first line read otherwise.
function _checked(blocks::Vector{_Fenced}, lines::AbstractVector{<:AbstractString},
                  outside::AbstractVector{<:AbstractString}, text::String, file::String)::Vector{_Fenced}
    julia = _julia_reads(text)
    ours = [(b.info, _stripped(view(lines, (first(b.lines) + 1):(last(b.lines) - 1)))) for b in blocks]
    theirs = [(info, _stripped(split(code, '\n'))) for (info, code) in julia.fences]
    k = findfirst(i -> get(ours, i, nothing) != get(theirs, i, nothing), 1:max(length(ours), length(theirs)))
    if !isnothing(k)
        next = _described(get(theirs, k, nothing))
        k <= length(blocks) && throw(PageError(file, first(blocks[k].lines),
            "Julia's Markdown does not read the fenced block that opens here (its next fenced block: $next): " *
            "close a fence at its opening fence's indentation, outside a paragraph's hard line break"))
        from = k == 1 ? 1 : last(blocks[k - 1].lines) + 1
        throw(PageError(file, something(findnext(l -> occursin(r"```|~~~", l), lines, from), from),
            "Julia's Markdown reads a fenced block here that the metric does not ($next): in a block quote, " *
            "or closed at another indentation"))
    end
    rows = [strip(l) for t in julia.tables for l in split(t, '\n') if !isempty(strip(l))]
    i = 1
    for (n, l) in pairs(outside)
        startswith(lstrip(l), '|') || continue
        k = findnext(==(strip(l)), rows, i)
        isnothing(k) && throw(PageError(file, n, "a line starting with | that Julia's Markdown reads as no " *
            "table row: a table starts a block and holds a header, an alignment row and a body row"))
        i = k + 1
    end
    blocks
end

# A page's fenced blocks, checked against Julia's Markdown, and its lines outside them: those of
# each block emptied, but for the text after a closing fence, so every line keeps its number.
function _read(lines::AbstractVector{<:AbstractString}, file::String)::@NamedTuple{blocks::Vector{_Fenced},
                                                                                     outside::Vector{SubString{String}}}
    blocks = _scan(lines, file)
    outside = SubString{String}[SubString(String(l)) for l in lines]
    for b in blocks
        outside[b.lines] .= SubString("")
        outside[last(b.lines)] = SubString(String(lines[last(b.lines)]), b.closing + 1)
    end
    (blocks=_checked(blocks, lines, outside, join(lines, '\n'), file), outside=outside)
end

"""
    fenced_blocks(lines; file::String = "string") -> Vector{UnitRange{Int}}

The fenced code blocks of the Markdown page whose lines are `lines`, each the range from its
opening to its closing fence, as Julia's Markdown reads them — the stdlib parser Documenter runs
on every page (Julia 1.13, stdlib Markdown/src/GitHub/GitHub.jl:5-37):

- an opening fence is three or more backticks or tildes after padding of spaces, tabs and
  carriage returns; its run is its fence characters, and its info string, the rest of the line,
  holds no fence character;
- the block closes at the first later line that starts with the opening fence's padding and then
  exactly its run, no fence character after it; text may follow, and starts the next block.

The page is read by the stdlib parser too: a fenced block it reads otherwise — a closing fence at
another indentation, a fence in a block quote or in a paragraph's hard line break —, a fence
never closed, or a line starting with `|` that it reads as no table row, is a `PageError` naming
`file` and the line.
"""
fenced_blocks(lines::AbstractVector{<:AbstractString}; file::String="string")::Vector{UnitRange{Int}} =
    [b.lines for b in _read(lines, file).blocks]

"""
    outside_fences(lines; file::String = "string") -> Vector{SubString{String}}

`lines` with those of the page's fenced blocks (`fenced_blocks`) emptied, but for the text after
a closing fence, so every line keeps its number; the refusals are `fenced_blocks`'s.
"""
outside_fences(lines::AbstractVector{<:AbstractString}; file::String="string")::Vector{SubString{String}} =
    _read(lines, file).outside

_outside_code(text::String, file::String)::Vector{SubString{String}} = outside_fences(split(text, '\n'); file)

# ── Prose ────────────────────────────────────────────────────────────────────

# The title of an admonition (stdlib Markdown/src/Common/block.jl:211).
const _TITLED = r"^([a-z]+) \"(.*)\"$"

# Whether `line` opens an Evidence box: an admonition as Julia's Markdown reads one (Markdown/src/
# Common/block.jl:207-225) — `!!! ` where the line starts, then the rest, stripped, a category, one
# space and the title in double quotes —, of the category `details`, its title starting "Evidence".
# Indented, it opens none here: the block parser tries one where the line starts (parse/
# parse.jl:86-93; `skipblank` keeps a line's indentation, util.jl:26-36), except right after a block
# quote, whose parser takes up to three spaces of the next line (Common/block.jl:174): read as prose.
function _evidence_opener(line::AbstractString)::Bool
    startswith(line, "!!! ") || return false
    m = match(_TITLED, strip(SubString(line, 5)))
    !isnothing(m) && m[1] == "details" && startswith(m[2], "Evidence")
end

# Whether line `n` of `outside` follows a line ending in a backslash. After a paragraph's, a hard
# line break, Julia's paragraph takes line `n` without trying a block parser on it (Markdown/src/
# Common/block.jl:19-27), so no admonition opens there; after any other line too, here: its box is
# read as prose, never left out.
_after_break(outside::AbstractVector{<:AbstractString}, n::Int)::Bool =
    n > 1 && endswith(chopsuffix(outside[n - 1], "\r"), '\\')

# Whether each of `lines` is in the body of an Evidence box (its opening line found in `outside`,
# the lines outside fenced blocks): the lines after it that start with four spaces or a tab, or
# are blank — only spaces, tabs and carriage returns in them —, up to the first other line, as
# Julia's admonition parser takes its body (Markdown/src/Common/block.jl:227-237).
function _in_evidence(lines::AbstractVector{<:AbstractString}, outside::AbstractVector{<:AbstractString})::BitVector
    body = falses(length(lines))
    open = false
    for (n, line) in pairs(lines)
        if open && (startswith(line, "    ") || startswith(line, '\t') || all(in(_WHITESPACE), line))
            body[n] = true
        else
            open = _evidence_opener(outside[n]) && !_after_break(outside, n)
        end
    end
    body
end

# The prose lines of a page, each with its number, and how many lines the bodies of its Evidence
# boxes hold that would be prose lines otherwise.
function _prose(text::String, file::String)::Tuple{Vector{Tuple{Int,String}},Int}
    lines = split(text, '\n')
    outside = outside_fences(lines; file)
    evidence = _in_evidence(lines, outside)
    candidates = [(n, String(l)) for (n, l) in pairs(outside) if !isempty(strip(l)) && !startswith(lstrip(l), '|')]
    [(n, l) for (n, l) in candidates if !evidence[n]], count(((n, _),) -> evidence[n], candidates)
end

"""
    prose_lines(text::String; file::String = "string") -> Vector{Tuple{Int,String}}

The prose of a page, each line with its number: the non-blank lines outside fenced blocks
(`fenced_blocks`; the text after a closing fence is prose), outside table rows (lines starting
with `|`) and outside the bodies of Evidence boxes — provenance, not guidance. An Evidence box is
the admonition Julia's Markdown reads from a line `!!! details "Evidence…"`: it starts the line,
the title's quotes end it once the line is stripped, and the line above does not end in a
backslash; its body is the lines after it indented four spaces or a tab, and blank lines (only
spaces, tabs and carriage returns). A box nested in a list or another admonition is not
recognised: its body counts as prose. Headings and every other admonition's body are prose. The
refusals are `fenced_blocks`'s, naming `file` and the line.
"""
prose_lines(text::String; file::String="string")::Vector{Tuple{Int,String}} = first(_prose(text, file))

# How many lines the Evidence boxes of the page at the path `file` leave out of its prose.
_evidence_lines(file::String)::Int = last(_prose(read(file, String), file))

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

# The words that name an entry's provider: its `tokens`, its `key_vars` and its `env_vars` — of a
# variable written with `<…>` in it (`AZURE_OPENAI_DEPLOY_NAME_<MODEL>`), the prefix before.
_words(e::ProviderEntry)::Vector{String} = [e.tokens; e.key_vars; [String(first(split(v, '<'))) for v in e.env_vars]]

# A registry's words and the pattern finding them, built once per registry.
struct _Matcher
    tokens::Vector{String}               # longest first
    pattern::Regex                       # group k matches tokens[k]
    providers::Vector{Set{Symbol}}       # the providers of the entries listing tokens[k]
end
function _Matcher(registry::Vector{ProviderEntry})
    tokens, alternation = _alternation([t for e in registry for t in _words(e)])
    _Matcher(tokens, Regex(_BEFORE * alternation), [Set{Symbol}(e.provider for e in registry if t in _words(e)) for t in tokens])
end

const _REGISTRY = _Matcher(REGISTRY)
const _NEUTRAL = Regex(_BEFORE * last(_alternation(NEUTRAL_TOKENS)))
const _DESTINATION = r"\]\([^)]*\)"

# What follows a model id that carries an Ollama tag: the rest of the id, `:` and the tag. Ollama
# names a model it serves `model:tag` (https://github.com/ollama/ollama/blob/main/docs/api.md#model-names).
const _OLLAMA_TAG = r"^[A-Za-z0-9._-]*:[A-Za-z0-9]"

"""
    provider_tokens(line::String; registry = REGISTRY) -> Vector{Tuple{Set{Symbol},String}}

The provider tokens of `line` — the `tokens`, `key_vars` and `env_vars` of the `registry` entries
(of a variable written with `<MODEL>`, the prefix before it) —, in order, each with the providers
of the entries that list it (more than one where entries of several providers list it: `gpt-` is
OpenAI's and Azure's). Link destinations — `](…)` — are dropped first, then the `NEUTRAL_TOKENS`,
matched like any token. A token matches case-sensitively where no letter, digit or `_` precedes it
(so `AZURE_OPENAI_API_KEY` holds no `OPENAI_API_KEY`). A token ending in `-` or `_` is a prefix and
takes in the rest of the identifier it starts (`gpt-` in `gpt-5.4-mini`, `GPT-` in `GPT-6`); one
ending in a letter takes in a CamelCase identifier it starts (`Ollama` in `OllamaOptions`,
`TypeSafe` in `TypeSafeProtocol`) and matches nowhere a lowercase letter, digit or `_` follows it
(`Jevons`); any other token matches only where none of those follows it. Where several tokens
start, the longest wins, and no span of the line is counted twice: `Azure OpenAI` is one Azure
token, and `OPENAIServiceEndpoint` keeps its `ServiceEndpoint`. A model id with an Ollama tag
(`gpt-oss:120b-cloud`, `deepseek-r1:8b`) names a model Ollama serves: its prefix is Ollama's.
"""
function provider_tokens(line::String; registry::Vector{ProviderEntry}=REGISTRY)::Vector{Tuple{Set{Symbol},String}}
    m = registry === REGISTRY ? _REGISTRY : _Matcher(registry)
    text = replace(replace(line, _DESTINATION => "]"), _NEUTRAL => " ")
    found = Tuple{Set{Symbol},String}[]
    for x in eachmatch(m.pattern, text)
        k = findfirst(!isnothing, x.captures)::Int
        tagged = endswith(m.tokens[k], '-') && occursin(_OLLAMA_TAG, SubString(text, x.offset + ncodeunits(x.match)))
        push!(found, (tagged ? Set([:ollama]) : copy(m.providers[k]), m.tokens[k]))
    end
    found
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
# A binding line: names assigned endpoints — called or not — and quoted ids, and nothing else. A
# quoted string interpolates nothing (no `$`), so no call can hide in one.
const _VALUE = "(?:(?:" * _ENDPOINT.pattern * ")(?:\\((?:[^()\"]|\"[^\"\$]*\")*\\))?|\"[^\"\\s\$]+\")"
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
The page's binding block is the first fenced block (`fenced_blocks`) with a line starting
`service, model =`; its binding lines — assignments of endpoints, called or not, and quoted ids,
with nothing after them but a comment and no `\$` in a string, such as
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
    blocks = fenced_blocks(lines; file=page)
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

# The problems of the ledgers in the directory `ledgers` against the audit units in `atoms` — all
# of them, or the unit `unit` alone: each `<unit>.json` with an `atoms` array expects `<unit>.tsv`
# to place `<unit>:1` … `<unit>:<n>`, each from its atom's `location`. An `ArgumentError` when
# `atoms` holds no `<unit>.json` for the unit `unit`.
function _ledger_problems(ledgers::String, atoms::String, docsrc::String,
                          unit::Union{String,Nothing}=nothing)::Vector{String}
    isnothing(unit) || isfile(joinpath(atoms, unit * ".json")) ||
        throw(ArgumentError("$atoms holds no $unit.json: no audit unit $(repr(unit))"))
    files = isnothing(unit) ? filter(f -> endswith(f, ".json"), readdir(atoms)) : [unit * ".json"]
    units = [(chopsuffix(f, ".json"), JSON.parsefile(joinpath(atoms, f))) for f in files]
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
    report(io::IO; docsrc::String, ledgers = nothing, atoms = nothing, baseline::Bool = false,
           entry_key::Union{Symbol,Nothing} = nothing, unit::Union{String,Nothing} = nothing) -> Bool

Prints to `io` the acceptance numbers of the pages under `docsrc`: a line per entry (`ENTRY`: at
most 1% of its prose lines name another provider), per reader's path (`PATH`: 5%) and per concept
page (`CONCEPT`: 2%, every provider counting), each with its share and threshold — an entry or a
page that does not exist yet shows "no pages" and fails —; then, for each page measured whose
Evidence boxes hold lines, how many of them the shares leave out; then each endpoint and model id
a concept page names in code outside the binding lines of its binding block, and each breach of
that block's bounds (`concept_code_violations`); then the problems of the ledgers in the
directory `ledgers` against the audit units in the directory `atoms` (both or neither; without
them, "ledgers: not checked"). With `entry_key`, only that entry's line and its reader's path are
measured, and no concept page; with `unit`, only that audit unit's ledger is checked (an
`ArgumentError` when `atoms` holds no `<unit>.json`, or for an entry that does not exist).
Returns whether every share holds, no concept page names provider code and no ledger has a
problem; with `baseline = true` it prints the same and returns `true`.
"""
function report(io::IO; docsrc::String, ledgers::Union{String,Nothing}=nothing, atoms::Union{String,Nothing}=nothing,
                baseline::Bool=false, entry_key::Union{Symbol,Nothing}=nothing,
                unit::Union{String,Nothing}=nothing)::Bool
    isnothing(ledgers) == isnothing(atoms) ||
        throw(ArgumentError("the ledgers are checked against the atoms: pass both directories or neither"))
    isnothing(unit) || !isnothing(atoms) ||
        throw(ArgumentError("the unit $(repr(unit)) is one of the atoms: pass the ledgers and the atoms with it"))
    entries = isnothing(entry_key) ? REGISTRY : [entry(entry_key)]
    anchors = _anchors(docsrc)
    concepts = isnothing(entry_key) ? [p for c in instances(Chapter) for p in CONCEPT_PAGES[c]] : String[]
    measurements = [[_measure(ENTRY, string(e.key), e.pages, e.key, docsrc) for e in entries];
                    [_measure(PATH, string(e.key), _reader_path(e, docsrc, anchors), e.key, docsrc) for e in entries];
                    [_measure(CONCEPT, p, [p], nothing, docsrc) for p in concepts]]
    measured = unique!([p for m in measurements for p in m.pages if p ∉ m.absent])
    evidence = filter(((_, n),) -> n > 0, [p => _evidence_lines(joinpath(docsrc, p)) for p in measured])
    code = [v for p in concepts if isfile(joinpath(docsrc, p)) for v in concept_code_violations(joinpath(docsrc, p))]
    problems = isnothing(ledgers) ? nothing : _ledger_problems(ledgers, atoms, docsrc, unit)
    println(io, "Other-provider share of prose lines (non-blank lines outside fenced blocks, table rows and ",
            "Evidence-box bodies)")
    foreach(m -> println(io, _row(m)), measurements)
    println(io, "Evidence-box lines left out of the shares, per page: ", sum(last, evidence; init=0))
    foreach(((p, n),) -> println(io, "  ", p, ": ", n), evidence)
    println(io, "Concept-page code outside the binding lines, and binding-block breaches: ", length(code))
    foreach(v -> println(io, "  ", v), code)
    println(io, "ledgers: ", isnothing(problems) ? "not checked" : "$(length(problems)) problems")
    foreach(p -> println(io, "  ", p), something(problems, String[]))
    pass = all(holds, measurements) && isempty(code) && isempty(something(problems, String[]))
    println(io, baseline ? "baseline: printed, not enforced" : pass ? "pass" : "fail")
    baseline || pass
end

# ── Command line ─────────────────────────────────────────────────────────────

const USAGE = "usage: julia --startup-file=no --project=docs docs/metrics.jl [--baseline] [--entry KEY] " *
              "[--ledgers DIR --atoms DIR [--unit NAME]]"

# Why the options `given` are no usage of this script once every argument is read, or `nothing`:
# `--ledgers` and `--atoms` come together, `--unit` names a `<unit>.json` of `--atoms`, and
# `--entry` an entry of the registry.
function _misuse(given::Dict{String,String})::Union{String,Nothing}
    haskey(given, "--ledgers") == haskey(given, "--atoms") || return "--ledgers and --atoms come together"
    unit, atoms = get(given, "--unit", nothing), get(given, "--atoms", nothing)
    isnothing(unit) || !isnothing(atoms) || return "--unit names an audit unit of --atoms"
    isnothing(unit) || isfile(joinpath(atoms, unit * ".json")) || return "--unit: $atoms holds no $unit.json"
    key = get(given, "--entry", nothing)
    isnothing(key) || any(e -> string(e.key) == key, REGISTRY) ||
        return "--entry: no provider entry $key; the entries are $(join((string(e.key) for e in REGISTRY), ", "))"
    nothing
end

# The options `args` give — each flag at most once, `--ledgers` and `--atoms` with a directory,
# `--entry` and `--unit` with a name — or why they are not a usage of this script.
function _options(args::AbstractVector{String}, given::Dict{String,String}=Dict{String,String}())::Union{Dict{String,String},String}
    isempty(args) && return something(_misuse(given), given)
    flag = first(args)
    haskey(given, flag) && return "$flag is given twice"
    flag == "--baseline" && return _options(args[2:end], push!(given, flag => ""))
    if flag in ("--ledgers", "--atoms")
        length(args) >= 2 && isdir(args[2]) || return "$flag takes a directory"
    elseif flag in ("--entry", "--unit")
        length(args) >= 2 && !startswith(args[2], "--") || return "$flag takes a name"
    else
        return "unknown argument $(repr(flag))"
    end
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
    key = get(options, "--entry", nothing)
    report(stdout; docsrc=relpath(DOCSRC), ledgers=get(options, "--ledgers", nothing),
           atoms=get(options, "--atoms", nothing), baseline=haskey(options, "--baseline"),
           entry_key=isnothing(key) ? nothing : Symbol(key), unit=get(options, "--unit", nothing)) ? 0 : 1
end

end # module

if abspath(PROGRAM_FILE) == @__FILE__
    exit(DocMetrics.main(ARGS))
end
