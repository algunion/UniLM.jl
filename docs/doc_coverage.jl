# Doc-coverage gate. Every EXPORTED UniLM symbol must appear in an `@docs`
# block under docs/src, OR be listed in KNOWN_UNDOCUMENTED (docs/undocumented_allowlist.jl).
# Assumes explicit `@docs` listing; there are currently no `@autodocs` blocks
# (if one is added that splices a whole module, extend this parser).

"Exported names of `mod` as strings, excluding the module name itself."
exported_names(mod::Module)::Set{String} =
    Set(string(n) for n in names(mod) if n != nameof(mod))

# A fence line: up to 3 spaces of indentation keep it at the top level of the page, a
# run of 3+ backticks or tildes opens or closes it, and the rest is the info string.
const _FENCE = r"^( *)(`{3,}|~{3,})(.*)$"

"""
Symbol names listed in the top-level `@docs` / `@autodocs` blocks of `text`.

Only blocks Documenter expands count: a fence at the top level of the page (indented at
most 3 spaces — Documenter does not expand `@docs` nested in an admonition, list or
quote) whose info string starts with `@docs` or `@autodocs`, of any fence length. Text
inside an HTML comment block or inside any other fenced block (a ````markdown
example, a ~~~ block) is not parsed, so an `@docs` fence shown there is not counted.
"""
function _documented_in(text::AbstractString)::Set{String}
    documented = Set{String}()
    fence = nothing        # (fence char, run length, counts?) of the open fenced block
    comment = false        # inside an HTML comment block
    for line in eachline(IOBuffer(text))
        s = strip(line)
        if comment
            occursin("-->", line) && (comment = false)
        elseif fence !== nothing
            m = match(_FENCE, s)
            if m !== nothing && first(m[2]) == fence[1] && length(m[2]) >= fence[2] && isempty(strip(m[3]))
                fence = nothing
            elseif fence[3] && !isempty(s)
                (occursin('=', s) || occursin('[', s)) && continue  # skip @autodocs config lines
                push!(documented, replace(s, "UniLM." => ""))
            end
        elseif (m = match(_FENCE, line)) !== nothing
            info = strip(m[3])
            top = length(m[1]) <= 3
            fence = (first(m[2]), length(m[2]), top && (startswith(info, "@docs") || startswith(info, "@autodocs")))
        elseif startswith(s, "<!--") && length(line) - length(lstrip(line)) <= 3
            comment = !occursin("-->", s[5:end])
        end
    end
    return documented
end

"Symbol names referenced in the top-level `@docs` blocks of the Markdown under `docsrc` (recursive)."
function parse_documented_symbols(docsrc::AbstractString)::Set{String}
    documented = Set{String}()
    for (root, _, files) in walkdir(docsrc), f in files
        endswith(f, ".md") && union!(documented, _documented_in(read(joinpath(root, f), String)))
    end
    return documented
end

missing_docs(exported::Set{String}, documented::Set{String}, allow::Set{String})::Vector{String} =
    sort(collect(setdiff(exported, documented, allow)))

stale_allow(exported::Set{String}, allow::Set{String})::Vector{String} =
    sort(collect(setdiff(allow, exported)))

resolved_allow(documented::Set{String}, allow::Set{String})::Vector{String} =
    sort(collect(intersect(documented, allow)))

"""
    assert_doc_coverage(mod, docsrc, allow)

Error (failing the build) unless every exported symbol of `mod` is documented
in an `@docs` block under `docsrc` or listed in `allow`; also errors on stale or
already-resolved allow-list entries so the ledger stays honest.
"""
function assert_doc_coverage(mod::Module, docsrc::AbstractString, allow::Set{String})
    exported   = exported_names(mod)
    documented = parse_documented_symbols(docsrc)
    problems = String[]
    miss     = missing_docs(exported, documented, allow)
    stale    = stale_allow(exported, allow)
    resolved = resolved_allow(documented, allow)
    isempty(miss)     || push!(problems, "Undocumented exported symbols (add to an @docs block or KNOWN_UNDOCUMENTED):\n  " * join(miss, "\n  "))
    isempty(stale)    || push!(problems, "KNOWN_UNDOCUMENTED lists names no longer exported (remove them):\n  " * join(stale, "\n  "))
    isempty(resolved) || push!(problems, "KNOWN_UNDOCUMENTED lists names that are now documented (remove them):\n  " * join(resolved, "\n  "))
    isempty(problems) && return nothing
    error("Doc-coverage gate failed.\n\n" * join(problems, "\n\n"))
end
