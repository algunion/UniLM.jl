# Service Endpoints

Types for configuring **multi-backend** service endpoints.

## Abstract Type

```@docs
UniLM.ServiceEndpoint
```

## Built-in Endpoints

```@docs
UniLM.OPENAIServiceEndpoint
UniLM.AZUREServiceEndpoint
UniLM.GEMINIServiceEndpoint
```

## Generic Endpoint

```@docs
GenericOpenAIEndpoint
ServiceEndpointSpec
OllamaEndpoint
MistralEndpoint
DeepSeekEndpoint
```

## Configuration

Each endpoint reads its configuration from environment variables:

### OpenAI (default)

| Variable         | Description         |
| :--------------- | :------------------ |
| `OPENAI_API_KEY` | Your OpenAI API key |

### Azure OpenAI

| Variable                           | Description                             |
| :--------------------------------- | :-------------------------------------- |
| `AZURE_OPENAI_BASE_URL`            | Azure endpoint base URL                 |
| `AZURE_OPENAI_API_KEY`             | Azure API key                           |
| `AZURE_OPENAI_API_VERSION`         | API version (e.g. `2024-12-01-preview`) |
| `AZURE_OPENAI_DEPLOY_NAME_GPT_5_2` | Deployment name for gpt-5.2             |

### Google Gemini

| Variable         | Description         |
| :--------------- | :------------------ |
| `GEMINI_API_KEY` | Your Gemini API key |

## Azure Deployment Mapping

Azure requires model-to-deployment name mappings. Use `add_azure_deploy_name!` to register
custom mappings:

```@example endpoints
using UniLM

# Register a custom deployment for a specific model
UniLM.add_azure_deploy_name!("gpt-5.2", "my-gpt52-deploy")
println("Registered deployment: ", UniLM._MODEL_ENDPOINTS_AZURE_OPENAI["gpt-5.2"])
```

## Selecting a Backend

Pass the `service` keyword to any request constructor:

```@example endpoints
chat = Chat(service=UniLM.AZUREServiceEndpoint, model="gpt-5.2")
println("Service: ", chat.service)
println("Model: ", chat.model)
```
