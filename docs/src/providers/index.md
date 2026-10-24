# [Language models](@id llm_start)

```@setup llm
using UniLM
```

## [Choose your provider](@id llm_providers)

```@eval
Main.ProviderFacts.provider_table(Main.ProviderFacts.llm)
```

## [Concepts](@id llm_concepts)

- [Conversations with Chat](@ref chat_guide)
- [Agentic respond](@ref agentic_guide)
- [Tool Calling](@ref tools_guide)
- [Streaming](@ref streaming_guide)
- [Structured Output](@ref structured_guide)
- [Attachments](@ref attachments_guide)
- [Embeddings](@ref embeddings_guide)
- [Image Generation](@ref images_guide)
- [FIM & Prefix Completion](@ref completions_guide)
- [Results](@ref results_guide)
- [Cost Tracking](@ref cost_guide)
- [Timeouts & Retries](@ref timeouts_guide)
- [Concurrency, Tasks and Cancellation](@ref concurrency_guide)
- [Testing and recordings](@ref testing_guide)
- [MCP (Model Context Protocol)](@ref mcp_guide)

## [Across providers](@id llm_across)

- [One API across providers](@ref backend_guide)
- [Capabilities by provider](@ref capabilities_guide)
- [Write your own backend](@ref custom_backends_guide)
