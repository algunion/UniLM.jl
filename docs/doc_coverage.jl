# Doc-coverage gate. Every EXPORTED UniLM symbol must appear in an `@docs`
# block under docs/src, OR be listed in KNOWN_UNDOCUMENTED (docs/undocumented_allowlist.jl).
# Assumes explicit `@docs` listing; there are currently no `@autodocs` blocks
# (if one is added that splices a whole module, extend this parser).

"Exported names of `mod` as strings, excluding the module name itself."
exported_names(mod::Module)::Set{String} =
    Set(string(n) for n in names(mod) if n != nameof(mod))

"Symbol names referenced inside ```@docs``` fences under `docsrc` (recursive)."
function parse_documented_symbols(docsrc::AbstractString)::Set{String}
    documented = Set{String}()
    for (root, _, files) in walkdir(docsrc)
        for f in files
            endswith(f, ".md") || continue
            indocs = false
            for line in eachline(joinpath(root, f))
                s = strip(line)
                if startswith(s, "```@docs") || startswith(s, "```@autodocs")
                    indocs = true; continue
                elseif indocs && startswith(s, "```")
                    indocs = false; continue
                end
                if indocs && !isempty(s)
                    (occursin('=', s) || occursin('[', s)) && continue  # skip @autodocs config lines
                    push!(documented, replace(s, "UniLM." => ""))
                end
            end
        end
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
