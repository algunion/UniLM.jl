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
                       "MISTRAL_API_KEY", "AZURE_OPENAI_API_KEY", "TYPESAFE_API_KEY")
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
    # Chapter → group → page; a chapter holds groups only, so each "Start here" is a one-page group.
    # A page whose content moves to a new page stays listed right after that page until it is removed.
    pages=[
        "Home" => "index.md",
        "Install" => "getting_started.md",
        "Versioning & Stability" => "stability.md",
        "Language models" => [
            "Start here" => ["Overview" => "providers/index.md"],
            "OpenAI" => [
                "Start here" => "providers/openai/index.md",
                "Models & request rules" => "providers/openai/models.md",
                "Chat Completions" => "providers/openai/chat_completions.md",
                "Responses & hosted tools" => "providers/openai/responses.md",
                "Responses API" => "guide/responses_api.md",
                "Retrieval & file search" => "providers/openai/retrieval.md",
                "Retrieval & File Search" => "guide/retrieval.md",
                "Images, audio & realtime" => "providers/openai/media.md",
                "Platform APIs" => "providers/openai/platform.md",
            ],
            "Anthropic" => [
                "Start here" => "providers/anthropic/index.md",
                "Thinking, tools & output" => "providers/anthropic/requests.md",
                "Caching, cost & limits" => "providers/anthropic/caching.md",
            ],
            "Google Gemini" => [
                "Start here" => "providers/gemini/index.md",
                "The generateContent API" => "providers/gemini/generate_content.md",
                "The Interactions API" => "providers/gemini/interactions.md",
                "Images, speech, embeddings & the compatible endpoint" => "providers/gemini/media.md",
            ],
            "Local models" => [
                "Ollama" => "providers/ollama/index.md",
                "Local Models with Ollama" => "guide/ollama.md",
                "Embeddings" => "providers/ollama/embeddings.md",
                "Runtime & model management" => "providers/ollama/runtime.md",
                "vLLM, LM Studio & llama.cpp" => "providers/local_servers.md",
            ],
            "Other providers" => [
                "Azure OpenAI" => "providers/azure.md",
                "DeepSeek" => "providers/deepseek.md",
                "Mistral & OpenAI-compatible services" => "providers/openai_compatible.md",
            ],
            "Concepts (every provider)" => [
                "Conversations with Chat" => "guide/chat_completions.md",
                "Agentic respond" => "guide/agentic.md",
                "Tool Calling" => "guide/tool_calling.md",
                "Streaming" => "guide/streaming.md",
                "Structured Output" => "guide/structured_output.md",
                "Attachments" => "guide/attachments.md",
                "Embeddings" => "guide/embeddings.md",
                "Image Generation" => "guide/image_generation.md",
                "FIM & Prefix Completion" => "guide/completions.md",
                "Results" => "guide/results.md",
                "Cost Tracking" => "guide/cost_tracking.md",
                "Timeouts & Retries" => "guide/timeouts.md",
                "Concurrency, Tasks and Cancellation" => "guide/concurrency.md",
                "Testing and recordings" => "guide/testing.md",
                "MCP (Model Context Protocol)" => "guide/mcp.md",
            ],
            "Across providers" => [
                "One API across providers" => "guide/multi_backend.md",
                "Capabilities by provider" => "guide/capabilities.md",
                "Write your own backend" => "guide/custom_backends.md",
            ],
        ],
        "Decisions" => [
            "Start here" => ["Overview" => "decisions/index.md"],
            "Jev (TypeSafe)" => [
                "Start here" => "decisions/jev/index.md",
                "Start Here: Decisions in Five Minutes" => "guide/jev_start.md",
                "Images, Box, long context, structured guidance" => "decisions/jev/abilities.md",
                "Confidence, errors, cost & testing" => "decisions/jev/operations.md",
            ],
            "OpenAI Decisions API" => ["OpenAI Decisions API" => "decisions/openai.md"],
            "Local decision models" => ["Local decision models" => "decisions/ollama.md"],
            "Concepts (every decision provider)" => [
                "Route and Decide" => "guide/system_one.md",
                "Dispatch on Meaning" => "guide/natural_language_dispatch.md",
                "Many Items at Once" => "guide/semantic_algorithms.md",
                "Decisions with LLMs" => "guide/jev_with_llms.md",
                "Test and Develop" => "guide/jev_testing.md",
            ],
            "Across decision providers" => [
                "Choosing a decision provider" => "decisions/choosing.md",
                "Backends: Jev, OpenAI and Local Models" => "guide/system_one_backends.md",
                "Write your own decision provider" => "decisions/custom.md",
            ],
        ],
        "Reference" => [
            "API: language models" => [
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
                "Model Listings" => "api/models.md",
                "Audio" => "api/audio.md",
            ],
            "API: OpenAI platform" => [
                "Files" => "api/files.md",
                "Vector Stores" => "api/vector_stores.md",
                "Conversations" => "api/conversations.md",
                "Batch" => "api/batch.md",
                "Fine-tuning" => "api/fine_tuning.md",
                "Moderations" => "api/moderations.md",
                "Containers" => "api/containers.md",
                "Uploads" => "api/uploads.md",
                "Webhooks" => "api/webhooks.md",
                "Realtime" => "api/realtime.md",
            ],
            "API: decisions" => [
                "System One (Jev, OpenAI Decisions, Ollama)" => "api/system_one.md",
                "Natural-Language Dispatch" => "api/nl_dispatch.md",
            ],
            "For coding agents" => ["LLM Reference" => "llm.md"],
        ],
    ],
    warnonly=[:missing_docs, :cross_references],
)

with_answers(build_docs, ANSWERS_MODE)

assert_doc_coverage(UniLM, joinpath(@__DIR__, "src"), KNOWN_UNDOCUMENTED)

# The site is built from these pages (`npm run build` in site/) and deployed by docs/deploy.jl.
