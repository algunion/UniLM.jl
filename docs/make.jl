using Documenter
using UniLM

include(joinpath(@__DIR__, "doc_coverage.jl"))
include(joinpath(@__DIR__, "undocumented_allowlist.jl"))
include(joinpath(@__DIR__, "site_writer", "SiteWriter.jl"))
isdefined(Main, :ProviderFacts) || include(joinpath(@__DIR__, "ProviderFacts.jl"))

# The examples that call a service replay the answers committed in
# docs/recorded_answers: System One (`ask`, `list_models`) and the LLM verbs, streamed
# or not (`chatrequest!`, `respond`, `embeddingrequest!`, the tool loops, the Responses
# lifecycle calls, `fim_complete`, `prefix_complete`). A service
# does not answer a repeated request identically, and a build without keys would
# otherwise render only the error. The mode is chosen by flag, never by which keys
# are set:
#   no flag             → replay, even when keys are set; an example with no recording fails the build
#   UNILM_DOCS_RECORD=1 → replay what is recorded, record the rest live (with the keys
#                         of the providers whose examples have no recording)
#   UNILM_DOCS_LIVE=1   → no replay: every example calls its service live, and CI never
#                         deploys the result — the prose quotes the recorded outputs,
#                         which a live build does not reproduce
# Each flag is 1, 0 or unset, and at most one is 1. What the scope does not record —
# images, audio, files, MCP, … — goes to its service when recording or
# live. Replay hides every provider key, so a build without a flag never spends and
# renders what the keyless CI build renders, whatever the shell exports.
# After a recording run, commit the new files in docs/recorded_answers.
# Replay and record ignore TYPESAFE_DEFAULT_MODEL: an unpinned request names the
# default model, so an exported default would change every recording's key.
function docs_flag(name::String)::Bool
    value = get(ENV, name, "")
    value in ("", "0", "1") || error("$name must be 1, 0 or unset; got $(repr(value))")
    value == "1"
end
const RECORD = docs_flag("UNILM_DOCS_RECORD")
const LIVE = docs_flag("UNILM_DOCS_LIVE")
RECORD && LIVE && error("UNILM_DOCS_RECORD=1 and UNILM_DOCS_LIVE=1 exclude each other; accepted: " *
                        "neither (replay), UNILM_DOCS_RECORD=1 (record the missing) or UNILM_DOCS_LIVE=1 (live)")
const ANSWERS_MODE = LIVE ? nothing : RECORD ? :record_missing : :replay

const PROVIDER_KEYS = ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY", "DEEPSEEK_API_KEY",
                       "MISTRAL_API_KEY", "AZURE_OPENAI_API_KEY", "TYPESAFE_API_KEY", "OLLAMA_API_KEY")
# Replay contacts no server: the builder's OLLAMA_HOST must not show up in the examples' output.
hidden(mode::Symbol) = mode === :replay ? [k => nothing for k in (PROVIDER_KEYS..., "OLLAMA_HOST")] :
                                          Pair{String,Nothing}[]

with_answers(build, ::Nothing) = build()
with_answers(build, mode::Symbol) = withenv("TYPESAFE_DEFAULT_MODEL" => nothing, hidden(mode)...) do
    with_recorded_answers(build, joinpath(@__DIR__, "recorded_answers"); mode)
end

# docs/regions.jl writes these files' generated regions from the provider registry; a region that
# no longer matches the registry and the code stops the build before any page is built.
for p in (joinpath(@__DIR__, "src", "llm.md"), joinpath(dirname(@__DIR__), "llms.txt"))
    isempty(ProviderFacts.stale_regions(p)) || error("stale generated region in $p — run docs/regions.jl")
end

build_docs() = makedocs(;
    modules=[UniLM],
    authors="Marius Fersigan <marius.fersigan@gmail.com> and contributors",
    repo="https://github.com/algunion/UniLM.jl/blob/{commit}{path}#{line}",
    sitename="UniLM.jl",
    # Markdoc pages, the sidebar and the assets of the Next.js site in site/
    format=SiteWriter.SiteMarkdoc(joinpath(dirname(@__DIR__), "site")),
    pages=include(joinpath(@__DIR__, "pages.jl")),    # the sidebar
    warnonly=[:missing_docs, :cross_references],
)

with_answers(build_docs, ANSWERS_MODE)

assert_doc_coverage(UniLM, joinpath(@__DIR__, "src"), KNOWN_UNDOCUMENTED)

# The site is built from these pages (`npm run build` in site/) and deployed by docs/deploy.jl.
