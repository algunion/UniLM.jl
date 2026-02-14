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

## See Also

- `ServiceEndpoint` — backend type hierarchy
- `OPENAIServiceEndpoint`, `AZUREServiceEndpoint`, `GEMINIServiceEndpoint` — specific backends
