# [Ollama](@id ollama_api)

A local Ollama server: the endpoint, its runtime options, and the model-management
verbs. [Local Models with Ollama](@ref ollama_guide) shows them at work.

## Endpoint

```@docs
OllamaEndpoint
OllamaOptions
```

## Models

```@docs
model_info
pull_model
running_models
load_model
unload_model
OllamaModel
OllamaModelInfo
OllamaRunningModel
OllamaPullProgress
```

[`list_models`](@ref) lists a server's installed models when its `service` is an
`OllamaEndpoint`.

## Results

```@docs
OllamaSuccess
OllamaFailure
OllamaCallError
```
