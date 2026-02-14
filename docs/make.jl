using Documenter
using UniLM

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
        "Getting Started" => "getting_started.md",
        "Guide" => [
            "Chat Completions" => "guide/chat_completions.md",
            "Responses API" => "guide/responses_api.md",
            "Embeddings" => "guide/embeddings.md",
            "Tool Calling" => "guide/tool_calling.md",
            "Streaming" => "guide/streaming.md",
            "Structured Output" => "guide/structured_output.md",
            "Multi-Backend" => "guide/multi_backend.md",
        ],
        "API Reference" => [
            "Chat Types" => "api/chat.md",
            "Responses Types" => "api/responses.md",
            "Embeddings" => "api/embeddings.md",
            "Service Endpoints" => "api/endpoints.md",
            "Result Types" => "api/results.md",
        ],
    ],
    warnonly=[:missing_docs, :cross_references],
)

deploydocs(;
    repo="github.com/algunion/UniLM.jl",
    devbranch="main",
    push_preview=true,
)
