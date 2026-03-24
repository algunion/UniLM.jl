# ============================================================================
# Provider Capability Routing
# Declares which API features each provider supports, and validates before
# dispatching requests.
# ============================================================================

"""
    provider_capabilities(service) -> Set{Symbol}

Return the set of capabilities supported by the given service endpoint.

Standard capability symbols:
- `:chat` — Chat Completions API (`/v1/chat/completions`)
- `:embeddings` — Embeddings API (`/v1/embeddings`)
- `:responses` — Responses API (`/v1/responses`)
- `:images` — Image Generation API (`/v1/images/generations`)
- `:tools` — Tool/function calling in chat
- `:fim` — FIM Completion (`/v1/completions` with `suffix`)
- `:prefix_completion` — Chat prefix completion (continue from partial assistant message)
- `:json_output` — JSON output mode (`response_format`)
"""
provider_capabilities(::Type{OPENAIServiceEndpoint})  = Set([:chat, :responses, :embeddings, :images, :tools, :json_output])
provider_capabilities(::Type{AZUREServiceEndpoint})   = Set([:chat, :tools])
provider_capabilities(::Type{GEMINIServiceEndpoint})  = Set([:chat, :embeddings, :tools, :json_output])
provider_capabilities(::DeepSeekEndpoint)              = Set([:chat, :tools, :fim, :prefix_completion, :json_output])
provider_capabilities(::GenericOpenAIEndpoint)          = Set([:chat, :embeddings, :fim, :tools, :responses])  # permissive default

"""
    has_capability(service, cap::Symbol) -> Bool

Check whether the service endpoint supports a given capability.
"""
has_capability(service, cap::Symbol)::Bool = cap in provider_capabilities(service)

"""
    validate_capability(service, cap::Symbol, feature_name::String)

Throw `ArgumentError` with a clear message if the provider does not support the feature.
Called at the top of request functions for early validation.
"""
function validate_capability(service, cap::Symbol, feature_name::String)
    has_capability(service, cap) && return
    caps = join(sort(collect(provider_capabilities(service))), ", ")
    throw(ArgumentError("$feature_name is not supported by $(typeof(service)). Supported: $caps"))
end
