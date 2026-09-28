using Documenter
using UniLM

include(joinpath(@__DIR__, "doc_coverage.jl"))
include(joinpath(@__DIR__, "undocumented_allowlist.jl"))

# The System One examples (`ask`, `list_models`) replay the answers committed in
# docs/recorded_answers: the service does not answer a repeated request
# identically, and a build without a key would otherwise render only the error.
#   TYPESAFE_API_KEY unset                   → replay; an example with no recording fails the build
#   TYPESAFE_API_KEY and UNILM_DOCS_RECORD=1 → replay what is recorded, record the rest live
#   TYPESAFE_API_KEY alone                   → no replay: every example calls the service live
# After a recording run, commit the new files in docs/recorded_answers.
const HAS_TYPESAFE_KEY = !isempty(strip(get(ENV, "TYPESAFE_API_KEY", "")))
const RECORD_FLAG = get(ENV, "UNILM_DOCS_RECORD", "")
RECORD_FLAG in ("", "0", "1") || error("UNILM_DOCS_RECORD must be 1 or unset; got $(repr(RECORD_FLAG))")
RECORD_FLAG == "1" && !HAS_TYPESAFE_KEY &&
    error("UNILM_DOCS_RECORD=1 records answers from the live service and needs TYPESAFE_API_KEY")
const ANSWERS_MODE = RECORD_FLAG == "1" ? :record_missing : HAS_TYPESAFE_KEY ? nothing : :replay

with_answers(build, ::Nothing) = build()
with_answers(build, mode::Symbol) =
    with_recorded_answers(build, joinpath(@__DIR__, "recorded_answers"); mode)

build_docs() = makedocs(;
    modules=[UniLM],
    authors="Marius Fersigan <marius.fersigan@gmail.com> and contributors",
    repo="https://github.com/algunion/UniLM.jl/blob/{commit}{path}#{line}",
    sitename="UniLM.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://algunion.github.io/UniLM.jl",
        edit_link="main",
        assets=String[],
        sidebar_sitename=true,
        repolink="https://github.com/algunion/UniLM.jl",
    ),
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
            "Custom Backends" => "guide/custom_backends.md",
            "MCP (Model Context Protocol)" => "guide/mcp.md",
            "FIM & Prefix Completion" => "guide/completions.md",
        ],
        "System One (TypeSafe Jev)" => [
            "Typed Judgments with Jev" => "guide/system_one.md",
            "Multiple Dispatch on Natural Language" => "guide/natural_language_dispatch.md",
            "Semantic Programs" => "guide/semantic_programs.md",
            "Semantic Algorithms" => "guide/semantic_algorithms.md",
            "Developing and Testing with Jev" => "guide/jev_testing.md",
        ],
        "API Reference" => [
            "Chat Types" => "api/chat.md",
            "Responses Types" => "api/responses.md",
            "Images" => "api/images.md",
            "Embeddings" => "api/embeddings.md",
            "Service Endpoints" => "api/endpoints.md",
            "Request Config & Timeouts" => "api/config.md",
            "Result Types" => "api/results.md",
            "Cost Tracking" => "api/accounting.md",
            "MCP Client & Server" => "api/mcp.md",
            "FIM Types" => "api/completions.md",
            "Provider Capabilities" => "api/capabilities.md",
            "TypeSafe System One" => "api/system_one.md",
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

deploydocs(;
    repo="github.com/algunion/UniLM.jl",
    devbranch="main",
    versions=["stable" => "v^", "v#.#.#", "dev" => "dev"],
)
