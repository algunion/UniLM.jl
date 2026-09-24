# [Service Endpoints](@id endpoints_api)

Types for configuring **multi-backend** service endpoints, and the extension API a new
backend implements.

## Abstract Types

```@docs
UniLM.ServiceEndpoint
UniLM.OpenAIWireEndpoint
```

## Built-in Endpoints

```@docs
UniLM.OPENAIServiceEndpoint
UniLM.AZUREServiceEndpoint
UniLM.GEMINIServiceEndpoint
UniLM.GEMINIOpenAIServiceEndpoint
UniLM.ANTHROPICServiceEndpoint
UniLM.TYPESAFEServiceEndpoint
```

`TYPESAFEServiceEndpoint` is not a chat backend: it serves the
[TypeSafe System One API (Jev)](@ref system_one_api) and declares only
`:system_one` and `:models`, so the chat, Responses and embedding verbs reject
it up front.

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
| `AZURE_OPENAI_API_VERSION`         | API version (e.g. `2024-10-21`, the latest dated GA version) |
| `AZURE_OPENAI_DEPLOY_NAME_<MODEL>` | Deployment for a model, read at call time: the model id upper-cased, every character other than `A-Z`/`0-9` mapped to `_` (`AZURE_OPENAI_DEPLOY_NAME_GPT_5_2` for `gpt-5.2`) |

A registration made with [`add_azure_deploy_name!`](@ref) wins over the variable; a
model with neither fails the call with an `ArgumentError` (in the result's `cause`)
naming the variable to set. Azure's newer v1 API (no `api-version`) is not wrapped by
`AZUREServiceEndpoint`; see [Multi-Backend Support](@ref backend_guide) for reaching it
through `GenericOpenAIEndpoint` (untested).

### Google Gemini

| Variable         | Description         |
| :--------------- | :------------------ |
| `GEMINI_API_KEY` | Your Gemini API key |

Both `GEMINIServiceEndpoint` (native `generateContent`) and `GEMINIOpenAIServiceEndpoint`
(the OpenAI-compat shim) read `GEMINI_API_KEY`.

### Anthropic (Claude)

| Variable            | Description            |
| :------------------ | :--------------------- |
| `ANTHROPIC_API_KEY` | Your Anthropic API key |

### TypeSafe (System One / Jev)

| Variable                 | Description                                      |
| :----------------------- | :----------------------------------------------- |
| `TYPESAFE_API_KEY`       | Your TypeSafe API key                            |
| `TYPESAFE_BASE_URL`      | API root override (default `https://api.typesafe.ai`) |
| `TYPESAFE_DEFAULT_MODEL` | Model used when a call names none (default `jev-latest`) |

## Azure Deployment Mapping

Azure requires model-to-deployment name mappings. Use `add_azure_deploy_name!` to register
custom mappings:

```@docs
add_azure_deploy_name!
```

```@example endpoints
using UniLM

# Register a custom deployment for a specific model
UniLM.add_azure_deploy_name!("gpt-5.2", "my-gpt52-deploy")
println("Registered deployment: ", UniLM._MODEL_ENDPOINTS_AZURE_OPENAI["gpt-5.2"])
delete!(UniLM._MODEL_ENDPOINTS_AZURE_OPENAI, "gpt-5.2")  # cleanup
nothing # hide
```

## Selecting a Backend

Pass the `service` keyword to any request constructor:

```@example endpoints
chat = Chat(service=UniLM.AZUREServiceEndpoint, model="gpt-5.2")
println("Service: ", chat.service)
println("Model: ", chat.model)
```

## [Extension API](@id extension_api)

The functions a new backend adds methods to, and the stream-state types its handlers
mutate. They are declared `public` but not exported: qualify them (`UniLM.get_url`)
when adding methods. [Custom Backends](@ref custom_backends_guide) walks through both
cases — an OpenAI-compatible provider (routing and authentication only) and a provider
with its own wire (the chat and agentic seams). Pre-1.0, this contract can still change
in a minor release, under the CHANGELOG's **Breaking** heading.

### Routing, authentication and defaults

```@docs
UniLM.get_url
UniLM.auth_header
UniLM.default_model
```

[`provider_capabilities`](@ref) (exported) is the capability hook: an endpoint that
defines it opts into validation against that set.

### Chat wire seam

```@docs
UniLM.encode_request
UniLM.decode_response
UniLM.handle_sse_event!
UniLM.StreamState
```

### Agentic wire seam

```@docs
UniLM.encode_agentic
UniLM.decode_agentic
UniLM.decode_agentic_stream
UniLM.AgenticStreamState
```
