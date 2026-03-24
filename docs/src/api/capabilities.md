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
| `:chat` | Chat Completions API (`/v1/chat/completions`) |
| `:embeddings` | Embeddings API (`/v1/embeddings`) |
| `:responses` | Responses API (`/v1/responses`) |
| `:images` | Image Generation API (`/v1/images/generations`) |
| `:tools` | Tool/function calling in chat |
| `:fim` | FIM Completion (`/v1/completions` with `suffix`) |
| `:prefix_completion` | Chat prefix completion |
| `:json_output` | JSON output mode (`response_format`) |

## Capabilities by Provider

```@example capabilities
using UniLM

for (name, svc) in [
    ("OpenAI",  OPENAIServiceEndpoint),
    ("Azure",   AZUREServiceEndpoint),
    ("Gemini",  GEMINIServiceEndpoint),
    ("DeepSeek", DeepSeekEndpoint("k")),
    ("Generic", GenericOpenAIEndpoint("http://x", ""))
]
    caps = join(sort(collect(provider_capabilities(svc))), ", ")
    println("$name: $caps")
end
```
