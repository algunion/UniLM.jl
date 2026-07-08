using Documenter
using UniLM

include(joinpath(@__DIR__, "doc_coverage.jl"))
include(joinpath(@__DIR__, "undocumented_allowlist.jl"))

makedocs(;
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
        "Guide" => [
            "Chat Completions" => "guide/chat_completions.md",
            "Responses API" => "guide/responses_api.md",
            "Image Generation" => "guide/image_generation.md",
            "Embeddings" => "guide/embeddings.md",
            "Tool Calling" => "guide/tool_calling.md",
            "Agentic Workflows" => "guide/agentic.md",
            "Streaming" => "guide/streaming.md",
            "Structured Output" => "guide/structured_output.md",
            "Multi-Backend" => "guide/multi_backend.md",
            "MCP (Model Context Protocol)" => "guide/mcp.md",
            "FIM & Prefix Completion" => "guide/completions.md",
        ],
        "API Reference" => [
            "Chat Types" => "api/chat.md",
            "Responses Types" => "api/responses.md",
            "Images" => "api/images.md",
            "Embeddings" => "api/embeddings.md",
            "Service Endpoints" => "api/endpoints.md",
            "Result Types" => "api/results.md",
            "Cost Tracking" => "api/accounting.md",
            "MCP Client & Server" => "api/mcp.md",
            "FIM Types" => "api/completions.md",
            "Provider Capabilities" => "api/capabilities.md",
            "Files" => "api/files.md",
        ],
    ],
    warnonly=[:missing_docs, :cross_references],
)

assert_doc_coverage(UniLM, joinpath(@__DIR__, "src"), KNOWN_UNDOCUMENTED)

deploydocs(;
    repo="github.com/algunion/UniLM.jl",
    devbranch="main",
    push_preview=true,
    versions=["stable" => "v^", "v#.#.#", "dev" => "dev"],
)
