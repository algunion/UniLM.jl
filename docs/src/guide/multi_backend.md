# [Multi-Backend Support](@id backend_guide)

UniLM.jl is built around **neutral verbs**: the same `Chat` + [`chatrequest!`](@ref) — with
tools, streaming, and [cost accounting](@ref cost_guide) — run unchanged across every backend;
you only change the `service`. The agentic [`respond`](@ref) verb is neutral the same way
across OpenAI (Responses) and Gemini (Interactions); see [Agentic Workflows](@ref agentic_guide).
Native OpenAI, Anthropic, and Gemini are first-class backends with their own wire formats (each
exercised by live integration tests), not OpenAI-compatible shims.

Switching backends requires only changing the `service` parameter.

## Available Backends

| Backend          | Type                      | Env Variables                                                               |
| :--------------- | :------------------------ | :-------------------------------------------------------------------------- |
| OpenAI (default) | `OPENAIServiceEndpoint`   | `OPENAI_API_KEY`                                                            |
| Azure OpenAI     | `AZUREServiceEndpoint`    | `AZURE_OPENAI_BASE_URL`, `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_API_VERSION` |
| Google Gemini    | `GEMINIServiceEndpoint`   | `GEMINI_API_KEY`                                                            |
| Anthropic        | `ANTHROPICServiceEndpoint`| `ANTHROPIC_API_KEY`                                                         |
| DeepSeek         | `DeepSeekEndpoint`        | `DEEPSEEK_API_KEY`                                                          |
| Mistral          | `MistralEndpoint`         | `MISTRAL_API_KEY`                                                           |
| Ollama (local)   | `OllamaEndpoint`          | (none)                                                                      |
| Generic          | `GenericOpenAIEndpoint`   | (passed to constructor)                                                     |

## OpenAI (Default)

```@example backends
using UniLM
using JSON

# OpenAI is the default — no need to specify service
chat = Chat(model="gpt-5.2")
println("Service: ", chat.service)
println("Model: ", chat.model)
```

## Azure OpenAI

```julia
# Set environment variables
ENV["AZURE_OPENAI_BASE_URL"] = "https://your-resource.openai.azure.com"
ENV["AZURE_OPENAI_API_KEY"] = "your-key"
ENV["AZURE_OPENAI_API_VERSION"] = "2024-02-01"
ENV["AZURE_OPENAI_DEPLOY_NAME_GPT_5_2"] = "your-gpt52-deployment"

# Use Azure
chat = Chat(service=AZUREServiceEndpoint, model="gpt-5.2")
push!(chat, Message(Val(:system), "Hello from Azure!"))
push!(chat, Message(Val(:user), "Hi!"))
result = chatrequest!(chat)
```

### Custom Deployment Names

If your Azure deployment has a custom name:

```@example backends
UniLM.add_azure_deploy_name!("my-custom-model", "my-deployment-name")
println("Registered deployments: ", collect(keys(UniLM._MODEL_ENDPOINTS_AZURE_OPENAI)))
delete!(UniLM._MODEL_ENDPOINTS_AZURE_OPENAI, "my-custom-model")  # cleanup
```

## Google Gemini

!!! warning "Breaking change since v0.10.3"
    `GEMINIServiceEndpoint` now targets Google's **native `generateContent` API**
    (auth header `x-goog-api-key`, model in the URL, default model
    `gemini-3.5-flash`). The old **OpenAI-compatible** Gemini path is renamed
    [`GEMINIOpenAIServiceEndpoint`](@ref). Migrate code that relied on the
    OpenAI-compatible behavior — including `Embeddings(...; service=GEMINIServiceEndpoint)`,
    which the native endpoint does not support — to `GEMINIOpenAIServiceEndpoint`.

Native Gemini chat (real call, guarded so a failure never breaks the build):

```@example backends
gemini_chat = Chat(service=GEMINIServiceEndpoint)   # default model: gemini-3.5-flash
push!(gemini_chat, Message(Val(:system), "You are a helpful assistant."))
push!(gemini_chat, Message(Val(:user), "Say hello in one short sentence."))
result = chatrequest!(gemini_chat)
if result isa LLMSuccess
    println(result.message.content)
else
    println("Request failed — see result for details")
end
```

To keep using the OpenAI-compatible endpoint, switch the service type:

```julia
chat = Chat(service=GEMINIOpenAIServiceEndpoint, model="gemini-2.5-flash")
```

## Anthropic (native Messages API)

`ANTHROPICServiceEndpoint` calls Anthropic's native `/v1/messages` API
(`x-api-key` + `anthropic-version` headers). Default model `claude-opus-4-8`;
`max_tokens` is required on the wire and defaults to 4096 when you omit it.

```@example backends
claude_chat = Chat(service=ANTHROPICServiceEndpoint)  # default: claude-opus-4-8
push!(claude_chat, Message(Val(:system), "You are a helpful assistant."))
push!(claude_chat, Message(Val(:user), "Say hello in one short sentence."))
result = chatrequest!(claude_chat)
if result isa LLMSuccess
    println(result.message.content)
else
    println("Request failed — see result for details")
end
```

!!! note "Thinking models round-trip automatically"
    Claude models that emit thinking blocks (e.g. `claude-sonnet-5`) require
    those blocks — signatures intact — to be echoed verbatim on the next
    request of a tool-calling turn. UniLM captures the provider-native content
    on `Message.provider_content` at decode time (non-streaming and streaming)
    and echoes it automatically when the same provider encodes the
    conversation again, so multi-turn tool use works out of the box. Moving a
    conversation to a different provider falls back to the neutral
    text+tool_calls form (thinking is dropped, as other providers cannot
    verify another vendor's signatures).

## Responses API Backend

The Responses API also supports the `service` parameter:

```@example backends
r = Respond(
    service=UniLM.OPENAIServiceEndpoint,
    model="gpt-5.2",
    input="Hello!",
)
println("Service: ", r.service)
println("Model: ", r.model)
```

## OpenAI-Compatible Providers (Generic Endpoint)

Any provider that implements the OpenAI-compatible `/v1/chat/completions` endpoint can be
used with [`GenericOpenAIEndpoint`](@ref). This includes Ollama, vLLM, LM Studio, Mistral,
and many others.

### Ollama (local)

```@example backends
ep = OllamaEndpoint()  # defaults to http://localhost:11434
chat = Chat(service=ep, model="llama3.1")
println("URL: ", UniLM.get_url(chat))
```

### Mistral

```julia
chat = Chat(service=MistralEndpoint(), model="mistral-large-latest")
result = chatrequest!(chat)
```

### DeepSeek

```julia
chat = Chat(service=DeepSeekEndpoint(), model="deepseek-chat")       # V3.2
chat = Chat(service=DeepSeekEndpoint(), model="deepseek-reasoner")   # V3.2 thinking mode
```

### vLLM / LM Studio

```julia
# vLLM
chat = Chat(service=GenericOpenAIEndpoint("http://localhost:8000", ""), model="meta-llama/Llama-3.1-8B")

# LM Studio
chat = Chat(service=GenericOpenAIEndpoint("http://localhost:1234", ""), model="loaded-model")
```

### Anthropic (OpenAI-compatible shim)

Prefer the native `ANTHROPICServiceEndpoint` (the **Anthropic (native Messages
API)** section above). For evaluation only, Anthropic also exposes an
OpenAI-compatible endpoint — which Anthropic itself calls "not a long-term or
production-ready solution" (features like `response_format` and `strict` are
ignored):

```julia
chat = Chat(
    service=GenericOpenAIEndpoint("https://api.anthropic.com/v1", ENV["ANTHROPIC_API_KEY"]),
    model="claude-sonnet-4-6"
)
```

### Custom Provider

```@example backends
ep = GenericOpenAIEndpoint("https://my-llm-server.example.com", "sk-my-key")
chat = Chat(service=ep, model="my-model")
println("URL: ", UniLM.get_url(chat))
println("Has auth: ", any(p -> p.first == "Authorization", UniLM.auth_header(ep)))
```

### Embeddings with Generic Endpoint

Embeddings also support the `service` parameter:

```@example backends
emb = Embeddings("test"; service=OllamaEndpoint(), model="nomic-embed-text")
println("URL: ", UniLM.get_url(emb))
```

## API Compatibility Tiers

| API Surface | Standard Status | Supported Providers |
|---|---|---|
| Chat Completions | De facto standard | OpenAI, Azure, Gemini, Mistral, DeepSeek, Ollama, vLLM, LM Studio, Anthropic* |
| Embeddings | Widely adopted | OpenAI, Gemini, Mistral, Ollama, vLLM |
| Responses API | Emerging (Open Responses) | OpenAI, Ollama, vLLM, Amazon Bedrock |
| FIM Completion | Provider-specific | DeepSeek (beta), Ollama, vLLM |
| Image Generation | Limited | OpenAI, Gemini, Ollama |

*Anthropic compat layer is not production-recommended by Anthropic.

## Querying Provider Capabilities

Use [`has_capability`](@ref) to check what a provider supports before making requests:

```@example backends
for (name, svc) in [
    ("OpenAI", OPENAIServiceEndpoint),
    ("DeepSeek", DeepSeekEndpoint("k")),
    ("Ollama", OllamaEndpoint())
]
    caps = join(sort(collect(provider_capabilities(svc))), ", ")
    println("$name: $caps")
end
```

```@example backends
# Check specific capabilities
println("DeepSeek FIM: ", has_capability(DeepSeekEndpoint("k"), :fim))
println("OpenAI FIM: ", has_capability(OPENAIServiceEndpoint, :fim))
```

## See Also

- [`ServiceEndpoint`](@ref), [`GenericOpenAIEndpoint`](@ref) — endpoint types
- [`OllamaEndpoint`](@ref), [`MistralEndpoint`](@ref) — convenience constructors
- [`OPENAIServiceEndpoint`](@ref), [`AZUREServiceEndpoint`](@ref), [`GEMINIServiceEndpoint`](@ref) — built-in backends
