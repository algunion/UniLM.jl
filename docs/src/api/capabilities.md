# [Provider Capabilities](@id capabilities_api)

Functions for querying and validating provider capabilities.

Each service endpoint declares which API features it supports. Request functions
validate capabilities before dispatching, giving clear errors instead of HTTP 404s.

## What "validate before dispatch" means

Validation comes in two strengths, and which one applies depends on the verb:

- **Platform and lifecycle verbs** — files, vector stores, conversations,
  moderations, audio, batch, fine-tuning, containers, uploads, realtime, FIM,
  prefix completion, System One, model listing — and the image verbs
  [`generate_image`](@ref) (`:images`) and [`edit_image`](@ref) (`:image_edits`)
  validate **strictly**. These surfaces are provider-specific, and a backend that
  has not declared them would simply 404, so an endpoint that declares no
  capabilities at all is not dispatched: the call throws an `ArgumentError` that
  names the endpoint and says it declares no capabilities (the System One verbs,
  [`ask`](@ref), [`ask_each`](@ref), [`with_system_one`](@ref) and the
  natural-language verbs, add that a backend declares `:system_one`).
- **The three primary verbs** — [`chatrequest!`](@ref) (`:chat`),
  [`embeddingrequest!`](@ref) (`:embeddings`) and [`respond`](@ref) (`:responses`
  **or** `:agentic`, since the two agentic wires name the same surface
  differently) — validate only endpoints that **declare** their capabilities. An
  endpoint with no `provider_capabilities` method passes through unvalidated.

That asymmetry is deliberate. Defining a `ServiceEndpoint` subtype is the
documented way to reach an OpenAI-compatible backend this package does not ship,
and such a backend cannot declare anything — refusing to dispatch it would be a
false negative about a server the package knows nothing about. "Undeclared" is
therefore *no applicable method*, never an empty capability set. Declaring
capabilities is opt-in strictness: once your endpoint declares, the three primary
verbs hold it to that declaration like any built-in.

A rejection of a declared endpoint throws `ArgumentError` before any request,
naming the feature and the endpoint and listing what it does support (for example
`Files API is not supported by ANTHROPICServiceEndpoint. Supported: chat,
json_output, models, streaming, tools`).

## Functions

```@docs
provider_capabilities
has_capability
```

## Capability Symbols

| Symbol | Description |
|---|---|
| `:chat` | Chat Completions API |
| `:responses` | Responses API wire: OpenAI, Ollama and generic servers |
| `:agentic` | The agentic [`respond`](@ref) verb, on either wire: the Responses API (OpenAI, Ollama, generic servers) or Gemini Interactions |
| `:tools` | Tool / function calling |
| `:streaming` | Server-sent-event token streaming (`stream=true`) |
| `:json_output` | JSON / structured output (`response_format`) |
| `:embeddings` | Embeddings API |
| `:images` | Image generation |
| `:image_edits` | Image editing |
| `:fim` | FIM (fill-in-the-middle) completion |
| `:prefix_completion` | Prefix completion (continue a partial assistant message) |
| `:files` | Files API |
| `:vector_stores` | Vector Stores API |
| `:conversations` | Conversations API |
| `:moderation` | Moderations API |
| `:audio` | OpenAI's Audio API route, which serves [`transcribe`](@ref): OpenAI, Ollama and generic servers (implies `:transcription`) |
| `:transcription` | Transcription ([`transcribe`](@ref)); implied by `:audio`, so an endpoint of your own may declare either |
| `:speech` | Speech ([`speak`](@ref)): an OpenAI Realtime session, or Gemini TTS on `GEMINIServiceEndpoint` |
| `:batch` | Batch API |
| `:fine_tuning` | Fine-tuning API |
| `:containers` | Containers API |
| `:uploads` | Resumable Uploads API |
| `:realtime` | Realtime API: [`realtime_connect`](@ref), [`translate`](@ref) and the Realtime session `speak` runs on OpenAI |
| `:system_one` | System One evaluation ([`ask`](@ref)): TypeSafe, OpenAI (its Decisions API) and Ollama |
| `:models` | Model listing ([`list_models`](@ref)): TypeSafe, OpenAI, Anthropic, Gemini and Ollama |

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
    ("Generic",                GenericOpenAIEndpoint("http://x", "")),
    ("Ollama",                 OllamaEndpoint()),
    ("TypeSafe (System One)",  TYPESAFEServiceEndpoint)
]
    caps = join(sort(collect(provider_capabilities(svc))), ", ")
    println("$name: $caps")
end
```
