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
  `:batch`, `:fine_tuning`, `:containers`, `:uploads`, `:realtime`
- TypeSafe System One: `:system_one`, `:models`
"""
provider_capabilities(::Type{OPENAIServiceEndpoint})  = Set([:chat, :responses, :agentic, :embeddings, :images, :tools, :streaming, :json_output, :files, :vector_stores, :conversations, :moderation, :audio, :batch, :image_edits, :fine_tuning, :containers, :uploads, :realtime])
provider_capabilities(::Type{AZUREServiceEndpoint})   = Set([:chat, :tools, :streaming, :json_output])
provider_capabilities(::Type{GEMINIOpenAIServiceEndpoint})  = Set([:chat, :embeddings, :tools, :streaming, :json_output])
provider_capabilities(::DeepSeekEndpoint)              = Set([:chat, :tools, :streaming, :fim, :prefix_completion, :json_output])
provider_capabilities(::GenericOpenAIEndpoint)          = Set([:chat, :embeddings, :fim, :tools, :streaming, :json_output, :responses])  # permissive default

"""
    has_capability(service, cap::Symbol) -> Bool

Check whether the service endpoint supports a given capability.
"""
has_capability(service, cap::Symbol)::Bool = cap in provider_capabilities(service)

# An endpoint's name in messages: marker types are passed as the type itself, whose
# `typeof` is only `DataType`.
_service_name(service::Type) = nameof(service)
_service_name(service) = nameof(typeof(service))

"""
    validate_capability(service, cap::Symbol, feature_name::String)

Throw `ArgumentError` with a clear message if the provider does not support the feature.
Called at the top of request functions for early validation.
"""
function validate_capability(service, cap::Symbol, feature_name::String)
    has_capability(service, cap) && return
    caps = join(sort(collect(provider_capabilities(service))), ", ")
    throw(ArgumentError("$feature_name is not supported by $(_service_name(service)). Supported: $caps"))
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
    throw(ArgumentError("Responses API is not supported by $(_service_name(service)). Supported: $caps"))
end

# ─── Default Model Resolution ──────────────────────────────────────────────

"""
    default_model(service) -> Union{String, Nothing}

Return the default chat/completions model for the given service endpoint.
Returns `nothing` for generic and user-defined endpoints (model must be specified
explicitly; `Chat` then throws an `ArgumentError` when it is not). A backend adds
`default_model(::MyEndpoint)` to make `model=` optional for it.

Public extension API (not exported); see the Custom Backends guide.
"""
default_model(::Type{OPENAIServiceEndpoint})  = "gpt-5.6-sol"
default_model(::Type{AZUREServiceEndpoint})   = "gpt-5.2"
default_model(::Type{GEMINIOpenAIServiceEndpoint})  = "gemini-3.8-flash"
default_model(::DeepSeekEndpoint)              = "deepseek-flash"
default_model(::GenericOpenAIEndpoint)          = nothing
default_model(_) = nothing

"""Default embedding model per provider."""
default_embedding_model(::Type{OPENAIServiceEndpoint})  = "text-embedding-3-small"
default_embedding_model(::Type{GEMINIOpenAIServiceEndpoint})  = "gemini-embedding-001"
default_embedding_model(::DeepSeekEndpoint)              = nothing
default_embedding_model(::GenericOpenAIEndpoint)          = nothing
default_embedding_model(_) = nothing

"""Default image generation model per provider."""
default_image_model(::Type{OPENAIServiceEndpoint}) = "gpt-image-2"
default_image_model(_) = nothing

"""Default FIM model per provider. DeepSeek serves FIM (beta) on `deepseek-flash`."""
default_fim_model(::DeepSeekEndpoint)      = "deepseek-flash"
default_fim_model(::GenericOpenAIEndpoint)  = nothing
default_fim_model(_) = nothing

"""Resolve model from sentinel (empty string) to service default, or throw if no default."""
function _resolve_model(service, model::String)
    !isempty(model) && return model
    dm = default_model(service)
    isnothing(dm) && throw(ArgumentError("model must be specified when using $(_service_name(service))"))
    dm
end

_model_family(model::AbstractString, family::AbstractString) =
    model == family || startswith(model, family * "-")

# Model-specific restrictions belong to the native endpoint. A compatible
# server or Azure deployment may use the same name with a different contract.
# `sampling`: the names of the sampling-control and log-probability fields a request
# sets (`logprobs=false` requests none), so the error names them.
_set_fields(fields::Pair{String,Bool}...) = String[name for (name, set) in fields if set]

function _validate_astra(model::String, sampling::Vector{String}, effort)
    _model_family(model, "gpt-6-astra") || return nothing
    isempty(sampling) || throw(ArgumentError(
        "$model does not support sampling controls or log probabilities ($(join(sampling, ", ")))"))
    effort in ("none", "minimal") && throw(ArgumentError("$model requires at least low reasoning effort"))
    nothing
end

# GPT-6 Sol and Luna accept sampling controls and log probabilities only at
# reasoning effort "none". An omitted effort is the provider default, "medium".
function _validate_gpt6_sampling(model::String, effort, sampling::Vector{String}, fix::String)
    !isempty(sampling) && effort != "none" && any(f -> _model_family(model, f), ("gpt-6-sol", "gpt-6-luna")) ||
        return nothing
    shown = isnothing(effort) ? "\"medium (default)\"" : repr(effort)
    throw(ArgumentError("$model with reasoning effort $shown does not support sampling controls " *
        "or log probabilities ($(join(sampling, ", "))); set $fix or remove them"))
end

# GPT-6 Sol and Luna list reasoning efforts none, low, medium, high, xhigh and max;
# for a request that used "minimal", OpenAI's migration guide says to start with "low".
function _validate_gpt6_effort(model::String, effort)
    effort == "minimal" && any(f -> _model_family(model, f), ("gpt-6-sol", "gpt-6-luna")) &&
        throw(ArgumentError("$model does not support minimal reasoning effort; use \"low\""))
    nothing
end

# Chat Completions function calling on these families requires reasoning effort "none".
const _NO_REASONING_CHAT_TOOLS = ("gpt-5.6" => "GPT-5.6", "gpt-6-sol" => "GPT-6 Sol", "gpt-6-luna" => "GPT-6 Luna")

function encode_request(::Type{OPENAIServiceEndpoint}, chat::Chat)::String
    sampling = _set_fields("temperature" => !isnothing(chat.temperature), "top_p" => !isnothing(chat.top_p),
                           "logprobs" => chat.logprobs === true, "top_logprobs" => !isnothing(chat.top_logprobs))
    _validate_astra(chat.model, sampling, chat.reasoning_effort)
    _validate_gpt6_effort(chat.model, chat.reasoning_effort)
    _validate_gpt6_sampling(chat.model, chat.reasoning_effort, sampling, "reasoning_effort=\"none\"")
    if _model_family(chat.model, "gpt-6-astra") && !isnothing(chat.tools) && !isempty(chat.tools)
        throw(ArgumentError("GPT-6 Astra tool calling requires Respond and the Responses API"))
    end
    family = findfirst(p -> _model_family(chat.model, p.first), _NO_REASONING_CHAT_TOOLS)
    if !isnothing(family) && !isnothing(chat.tools) && !isempty(chat.tools) && chat.reasoning_effort != "none"
        throw(ArgumentError("$(_NO_REASONING_CHAT_TOOLS[family].second) Chat tools require reasoning_effort=\"none\"; use Respond for reasoning with tools"))
    end
    # The Chat Completions prompt_cache_options object has only mode and ttl.
    pco = chat.prompt_cache_options
    if !isnothing(pco) && (!isnothing(pco.prewarm) || !isnothing(pco.comparison_response_id))
        throw(ArgumentError("Chat Completions prompt_cache_options accepts only mode and ttl; prewarm and comparison_response_id are Responses-only"))
    end
    JSON.json(chat)
end

function encode_agentic(::Type{OPENAIServiceEndpoint}, r::Respond)::String
    effort = isnothing(r.reasoning) ? nothing : r.reasoning.effort
    logprobs_included = !isnothing(r.include) && "message.output_text.logprobs" in r.include
    sampling = _set_fields("temperature" => !isnothing(r.temperature), "top_p" => !isnothing(r.top_p),
                           "top_logprobs" => !isnothing(r.top_logprobs))
    _validate_astra(r.model, sampling, effort)
    _validate_gpt6_effort(r.model, effort)
    _validate_gpt6_sampling(r.model, effort, [sampling; logprobs_included ? ["include"] : String[]],
        "reasoning=Reasoning(effort=\"none\")")
    if (_model_family(r.model, "gpt-5.6") || _model_family(r.model, "gpt-6")) && !isnothing(r.prompt_cache_retention)
        throw(ArgumentError("$(r.model) uses prompt_cache_options=PromptCacheOptions(ttl=\"30m\"), not prompt_cache_retention"))
    end
    if _model_family(r.model, "gpt-6-astra") && !isnothing(r.include) && "message.output_text.logprobs" in r.include
        throw(ArgumentError("GPT-6 Astra does not support log probabilities"))
    end
    JSON.json(r)
end
