# [Multi-Backend Support](@id backend_guide)

UniLM.jl supports multiple LLM service backends through the `ServiceEndpoint`
type hierarchy. Switching backends requires only changing the `service` parameter.

## Available Backends

| Backend          | Type                    | Env Variables                                                               |
| :--------------- | :---------------------- | :-------------------------------------------------------------------------- |
| OpenAI (default) | `OPENAIServiceEndpoint` | `OPENAI_API_KEY`                                                            |
| Azure OpenAI     | `AZUREServiceEndpoint`  | `AZURE_OPENAI_BASE_URL`, `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_API_VERSION` |
| Google Gemini    | `GEMINIServiceEndpoint` | `GEMINI_API_KEY`                                                            |

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

```julia
ENV["GEMINI_API_KEY"] = "your-gemini-key"

chat = Chat(service=GEMINIServiceEndpoint, model="gemini-2.5-flash")
push!(chat, Message(Val(:system), "You are a helpful assistant."))
push!(chat, Message(Val(:user), "Hello!"))
result = chatrequest!(chat)
```

Available Gemini models:
- `"gemini-2.5-flash"`
- `"gemini-2.5-pro"`

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

### Anthropic (compatibility layer)

Anthropic provides an OpenAI-compatible endpoint for evaluation purposes.
Note: Anthropic considers this "not a long-term or production-ready solution" —
features like `response_format` and `strict` are ignored.

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
emb = Embeddings("test"; service=OllamaEndpoint())
println("URL: ", UniLM.get_url(emb))
```

## API Compatibility Tiers

| API Surface | Standard Status | Supported Providers |
|---|---|---|
| Chat Completions | De facto standard | OpenAI, Azure, Gemini, Mistral, Ollama, vLLM, LM Studio, Anthropic* |
| Embeddings | Widely adopted | OpenAI, Gemini, Mistral, Ollama, vLLM |
| Responses API | Emerging (Open Responses) | OpenAI, Ollama, vLLM, Amazon Bedrock |
| Image Generation | Limited | OpenAI, Gemini, Ollama |

*Anthropic compat layer is not production-recommended by Anthropic.

## See Also

- [`ServiceEndpoint`](@ref), [`GenericOpenAIEndpoint`](@ref) — endpoint types
- [`OllamaEndpoint`](@ref), [`MistralEndpoint`](@ref) — convenience constructors
- [`OPENAIServiceEndpoint`](@ref), [`AZUREServiceEndpoint`](@ref), [`GEMINIServiceEndpoint`](@ref) — built-in backends
