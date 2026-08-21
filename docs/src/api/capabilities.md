# [Provider Capabilities](@id capabilities_api)

Functions for querying and validating provider capabilities.

Each service endpoint declares which API features it supports. Request functions
validate capabilities before dispatching, giving clear errors instead of HTTP 404s.

## What "validate before dispatch" means

Validation comes in two strengths, and which one applies depends on the verb:

- **Platform and lifecycle verbs** — files, vector stores, conversations,
  moderations, audio, batch, fine-tuning, containers, uploads, video, realtime,
  FIM, prefix completion, image *edits* — validate **strictly**. An endpoint that
  declares no capabilities at all is rejected, because these surfaces are
  OpenAI-shaped and a backend that has not declared them would simply 404.
- **The four primary verbs** — [`chatrequest!`](@ref) (`:chat`),
  [`embeddingrequest!`](@ref) (`:embeddings`), [`respond`](@ref) (`:responses`
  **or** `:agentic`, since the two agentic wires name the same surface
  differently), and [`generate_image`](@ref) (`:images`) — validate only endpoints
  that **declare** their capabilities. An endpoint with no
  `provider_capabilities` method passes through unvalidated.

That asymmetry is deliberate. Defining a `ServiceEndpoint` subtype is the
documented way to reach an OpenAI-compatible backend this package does not ship,
and such a backend cannot declare anything — refusing to dispatch it would be a
false negative about a server the package knows nothing about. "Undeclared" is
therefore *no applicable method*, never an empty capability set. Declaring
capabilities is opt-in strictness: once your endpoint declares, the four primary
verbs hold it to that declaration like any built-in.

A rejection throws `ArgumentError` naming the feature and listing what the
provider does support.

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
