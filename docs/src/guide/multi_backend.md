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

```julia
# OpenAI is the default — no need to specify service
chat = Chat(model="gpt-4o")
result = chatrequest!(chat)

# Explicit:
chat = Chat(service=OPENAIServiceEndpoint, model="gpt-4o")
```

## Azure OpenAI

```julia
# Set environment variables
ENV["AZURE_OPENAI_BASE_URL"] = "https://your-resource.openai.azure.com"
ENV["AZURE_OPENAI_API_KEY"] = "your-key"
ENV["AZURE_OPENAI_API_VERSION"] = "2024-02-01"
ENV["AZURE_OPENAI_DEPLOY_NAME_GPT_4O"] = "your-gpt4o-deployment"

# Use Azure
chat = Chat(service=AZUREServiceEndpoint, model="gpt-4o")
push!(chat, Message(Val(:system), "Hello from Azure!"))
push!(chat, Message(Val(:user), "Hi!"))
result = chatrequest!(chat)
```

### Custom Deployment Names

If your Azure deployment has a custom name:

```julia
add_azure_deploy_name!("my-custom-model", "my-deployment-name")
chat = Chat(service=AZUREServiceEndpoint, model="my-custom-model")
```

## Google Gemini

```julia
ENV["GEMINI_API_KEY"] = "your-gemini-key"

chat = Chat(service=GEMINIServiceEndpoint, model="gemini-2.0-flash")
push!(chat, Message(Val(:system), "You are a helpful assistant."))
push!(chat, Message(Val(:user), "Hello!"))
result = chatrequest!(chat)
```

Available Gemini models:
- `"gemini-2.0-flash"`
- `"gemini-2.5-flash-preview-05-20"`
- `"gemini-1.5-ultra"`

## Responses API Backend

The Responses API also supports the `service` parameter:

```julia
r = Respond(
    service=OPENAIServiceEndpoint,  # only OpenAI for now
    model="gpt-4.1",
    input="Hello!",
)
result = respond(r)
```

## See Also

- `ServiceEndpoint` — backend type hierarchy
- `OPENAIServiceEndpoint`, `AZUREServiceEndpoint`, `GEMINIServiceEndpoint` — specific backends
