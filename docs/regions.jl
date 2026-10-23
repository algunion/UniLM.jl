# Writes the generated regions of docs/src/llm.md and llms.txt from the provider registry, in place.
# With --check it changes nothing: it prints each stale region as `file: name` and exits 1 when
# there is one.
#
#   julia --startup-file=no --project=docs docs/regions.jl [--check]
isdefined(Main, :ProviderFacts) || include(joinpath(@__DIR__, "ProviderFacts.jl"))

const REGION_FILES = ["docs/src/llm.md", "llms.txt"]        # under the repository root

function main(args::Vector{String})::Int
    args in ([], ["--check"]) || (println(stderr, "usage: docs/regions.jl [--check]"); return 2)
    cd(dirname(@__DIR__)) do
        if isempty(args)
            texts = [f => ProviderFacts.apply_regions(read(f, String); file=f) for f in REGION_FILES]
            foreach(((f, text),) -> write(f, text), texts)  # every file is read and checked before one is written
            0
        else
            stale = [f * ": " * name for f in REGION_FILES for name in ProviderFacts.stale_regions(f)]
            foreach(println, stale)
            isempty(stale) ? 0 : 1
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(main(ARGS))
end
