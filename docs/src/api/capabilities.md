# [Provider Capabilities](@id capabilities_api)

Functions for querying and validating provider capabilities.

Each service endpoint declares which API features it supports. Request functions
validate capabilities before dispatching, giving clear errors instead of HTTP 404s.

## Functions

```@docs
provider_capabilities
has_capability
```

## Capability Symbols

| Symbol | Description |
|---|---|
| `:chat` | Chat Completions API |
| `:responses` | Responses API |
| `:agentic` | Agentic `respond` verb (OpenAI Responses / Gemini Interactions) |
| `:tools` | Tool / function calling |
| `:streaming` | Server-sent-event token streaming (native providers) |
| `:json_output` | JSON / structured output mode |
| `:embeddings` | Embeddings API |
| `:images` | Image generation |
| `:image_edits` | Image editing |
| `:fim` | FIM (fill-in-the-middle) completion |
| `:prefix_completion` | Prefix completion (continue a partial assistant message) |
| `:files` | Files API |
| `:vector_stores` | Vector Stores API |
| `:conversations` | Conversations API |
| `:moderation` | Moderations API |
| `:audio` | Audio (TTS / transcription / translation) |
| `:batch` | Batch API |
| `:fine_tuning` | Fine-tuning API |
| `:containers` | Containers API |
| `:uploads` | Resumable Uploads API |
| `:video` | Video generation (Sora) |
| `:realtime` | Realtime API |

## Capabilities by Provider

```@example capabilities
using UniLM

for (name, svc) in [
    ("OpenAI",                 OPENAIServiceEndpoint),
    ("Azure",                  AZUREServiceEndpoint),
    ("Gemini (native)",        GEMINIServiceEndpoint),
    ("Gemini (OpenAI-compat)", GEMINIOpenAIServiceEndpoint),
    ("Anthropic",              ANTHROPICServiceEndpoint),
    ("DeepSeek",               DeepSeekEndpoint("k")),
    ("Generic",                GenericOpenAIEndpoint("http://x", ""))
]
    caps = join(sort(collect(provider_capabilities(svc))), ", ")
    println("$name: $caps")
end
```
