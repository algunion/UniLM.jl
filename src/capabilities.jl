# ============================================================================
# Provider Capability Routing
# Declares which API features each provider supports, and validates before
# dispatching requests.
# ============================================================================

"""
    provider_capabilities(service) -> Set{Symbol}

Return the set of capabilities supported by the given service endpoint.

Standard capability symbols include:
- Core: `:chat`, `:responses`, `:agentic`, `:tools`, `:streaming`, `:json_output`
- Embeddings & images: `:embeddings`, `:images`, `:image_edits`
- Completions: `:fim`, `:prefix_completion`
- Platform APIs: `:files`, `:vector_stores`, `:conversations`, `:moderation`, `:audio`,
  `:batch`, `:fine_tuning`, `:containers`, `:uploads`, `:video`, `:realtime`
"""
provider_capabilities(::Type{OPENAIServiceEndpoint})  = Set([:chat, :responses, :agentic, :embeddings, :images, :tools, :json_output, :files, :vector_stores, :conversations, :moderation, :audio, :batch, :image_edits, :fine_tuning, :containers, :uploads, :video, :realtime])
provider_capabilities(::Type{AZUREServiceEndpoint})   = Set([:chat, :tools])
provider_capabilities(::Type{GEMINIOpenAIServiceEndpoint})  = Set([:chat, :embeddings, :tools, :json_output])
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

"""
    _capability_declared(service) -> Bool

Whether this endpoint type declares its capabilities at all.

`provider_capabilities` has no fallback method, so "undeclared" is exactly "no
applicable method" — never an empty set. That distinction is what lets a
capability check reject a KNOWN-incapable provider without also rejecting a
user-defined endpoint, which is the documented way to reach an OpenAI-compatible
backend this package does not ship and which cannot declare anything.
"""
_capability_declared(service)::Bool = applicable(provider_capabilities, service)

"""
    _validate_declared_capability(service, cap::Symbol, feature_name::String)

[`validate_capability`](@ref) restricted to endpoints that declared their
capabilities: an undeclared endpoint passes through untouched. Used by the request
verbs, where refusing to dispatch a custom backend would be a false negative — the
package has no basis for claiming what someone else's server does not support.
"""
_validate_declared_capability(service, cap::Symbol, feature_name::String) =
    _capability_declared(service) ? validate_capability(service, cap, feature_name) : nothing

"""
    _validate_agentic_capability(service)

Declaration-aware gate for [`respond`](@ref), which drives two wires that name the
same surface differently: the OpenAI wire declares `:responses` (Responses API) and
the Gemini native wire declares `:agentic` (Interactions). Either declaration
admits the verb; only an endpoint that declares its capabilities and lists neither
is rejected.
"""
function _validate_agentic_capability(service)
    _capability_declared(service) || return nothing
    (has_capability(service, :responses) || has_capability(service, :agentic)) && return nothing
    caps = join(sort(collect(provider_capabilities(service))), ", ")
    throw(ArgumentError("Responses API is not supported by $(typeof(service)). Supported: $caps"))
end

# ─── Default Model Resolution ──────────────────────────────────────────────

"""
    default_model(service) -> Union{String, Nothing}

Return the default chat/completions model for the given service endpoint.
Returns `nothing` for generic endpoints (model must be specified explicitly).
"""
default_model(::Type{OPENAIServiceEndpoint})  = "gpt-5.5"
default_model(::Type{AZUREServiceEndpoint})   = "gpt-5.2"
default_model(::Type{GEMINIOpenAIServiceEndpoint})  = "gemini-3.5-flash"
default_model(::DeepSeekEndpoint)              = "deepseek-chat"
default_model(::GenericOpenAIEndpoint)          = nothing

"""Default embedding model per provider."""
default_embedding_model(::Type{OPENAIServiceEndpoint})  = "text-embedding-3-small"
default_embedding_model(::Type{GEMINIOpenAIServiceEndpoint})  = "gemini-embedding-001"
default_embedding_model(::DeepSeekEndpoint)              = nothing
default_embedding_model(::GenericOpenAIEndpoint)          = nothing
default_embedding_model(_) = nothing

"""Default image generation model per provider."""
default_image_model(::Type{OPENAIServiceEndpoint}) = "gpt-image-2"
default_image_model(_) = nothing

"""Default FIM model per provider."""
default_fim_model(::DeepSeekEndpoint)      = "deepseek-chat"
default_fim_model(::GenericOpenAIEndpoint)  = nothing
default_fim_model(_) = nothing

"""Resolve model from sentinel (empty string) to service default, or throw if no default."""
function _resolve_model(service, model::String)
    !isempty(model) && return model
    dm = default_model(service)
    isnothing(dm) && throw(ArgumentError("model must be specified when using $(typeof(service))"))
    dm
end
