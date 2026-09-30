using Documenter
using UniLM

include(joinpath(@__DIR__, "doc_coverage.jl"))
include(joinpath(@__DIR__, "undocumented_allowlist.jl"))
include(joinpath(@__DIR__, "site_writer", "SiteWriter.jl"))

# The examples that call a service replay the answers committed in
# docs/recorded_answers: System One (`ask`, `list_models`) and the non-streaming LLM
# verbs (`chatrequest!`, `respond`, `embeddingrequest!`, the tool loops). A service
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
# a streamed call, images, files, MCP, … — goes to its service when recording or
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
                       "MISTRAL_API_KEY", "AZURE_OPENAI_API_KEY", "TYPESAFE_API_KEY")
# Replay contacts no server: the builder's OLLAMA_HOST must not show up in the examples' output.
hidden(mode::Symbol) = mode === :replay ? [k => nothing for k in (PROVIDER_KEYS..., "OLLAMA_HOST")] :
                                          Pair{String,Nothing}[]

with_answers(build, ::Nothing) = build()
with_answers(build, mode::Symbol) = withenv("TYPESAFE_DEFAULT_MODEL" => nothing, hidden(mode)...) do
    with_recorded_answers(build, joinpath(@__DIR__, "recorded_answers"); mode)
end

build_docs() = makedocs(;
    modules=[UniLM],
    authors="Marius Fersigan <marius.fersigan@gmail.com> and contributors",
    repo="https://github.com/algunion/UniLM.jl/blob/{commit}{path}#{line}",
    sitename="UniLM.jl",
    # Markdoc pages, the sidebar and the assets of the Next.js site in site/
    format=SiteWriter.SiteMarkdoc(joinpath(dirname(@__DIR__), "site")),
    pages=[
        "Home" => "index.md",
        "LLM Reference" => "llm.md",
        "Getting Started" => "getting_started.md",
        "Versioning & Stability" => "stability.md",
        "Guide" => [
            "Chat Completions" => "guide/chat_completions.md",
            "Responses API" => "guide/responses_api.md",
            "Image Generation" => "guide/image_generation.md",
            "Embeddings" => "guide/embeddings.md",
            "Tool Calling" => "guide/tool_calling.md",
            "Retrieval & File Search" => "guide/retrieval.md",
            "Agentic Workflows" => "guide/agentic.md",
            "Streaming" => "guide/streaming.md",
            "Timeouts & Retries" => "guide/timeouts.md",
            "Concurrency, Tasks and Cancellation" => "guide/concurrency.md",
            "Structured Output" => "guide/structured_output.md",
            "Cost Tracking" => "guide/cost_tracking.md",
            "Multi-Backend" => "guide/multi_backend.md",
            "Local Models with Ollama" => "guide/ollama.md",
            "Custom Backends" => "guide/custom_backends.md",
            "MCP (Model Context Protocol)" => "guide/mcp.md",
            "FIM & Prefix Completion" => "guide/completions.md",
        ],
        "Decisions (Jev)" => [
            "Start Here: Decisions in Five Minutes" => "guide/jev_start.md",
            "Decisions with LLMs" => "guide/jev_with_llms.md",
            "Route and Decide" => "guide/system_one.md",
            "Dispatch on Meaning" => "guide/natural_language_dispatch.md",
            "Many Items at Once" => "guide/semantic_algorithms.md",
            "Test and Develop" => "guide/jev_testing.md",
        ],
        "API Reference" => [
            "Chat Types" => "api/chat.md",
            "Responses Types" => "api/responses.md",
            "Images" => "api/images.md",
            "Embeddings" => "api/embeddings.md",
            "Service Endpoints" => "api/endpoints.md",
            "Ollama" => "api/ollama.md",
            "Request Config & Timeouts" => "api/config.md",
            "Result Types" => "api/results.md",
            "Cost Tracking" => "api/accounting.md",
            "MCP Client & Server" => "api/mcp.md",
            "FIM Types" => "api/completions.md",
            "Provider Capabilities" => "api/capabilities.md",
            "Jev (TypeSafe System One)" => "api/system_one.md",
            "Natural-Language Dispatch" => "api/nl_dispatch.md",
            "Files" => "api/files.md",
            "Vector Stores" => "api/vector_stores.md",
            "Conversations" => "api/conversations.md",
            "Batch" => "api/batch.md",
            "Fine-tuning" => "api/fine_tuning.md",
            "Moderations" => "api/moderations.md",
            "Audio" => "api/audio.md",
            "Containers" => "api/containers.md",
            "Uploads" => "api/uploads.md",
            "Webhooks" => "api/webhooks.md",
            "Realtime" => "api/realtime.md",
        ],
    ],
    warnonly=[:missing_docs, :cross_references],
)

with_answers(build_docs, ANSWERS_MODE)

assert_doc_coverage(UniLM, joinpath(@__DIR__, "src"), KNOWN_UNDOCUMENTED)

# The site is built from these pages (`npm run build` in site/) and deployed by docs/deploy.jl.
