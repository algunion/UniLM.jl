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
- Transcription ([`transcribe`](@ref)): `:transcription` — implied by `:audio`; declared alone
  (Ollama), it allows transcription without speech or translation
- System One ([`ask`](@ref)): `:system_one` (TypeSafe; OpenAI, through its Decisions API; Ollama)
- Model listing ([`list_models`](@ref)): `:models` (TypeSafe System One, OpenAI, Anthropic, Gemini, Ollama)
"""
provider_capabilities(::Type{OPENAIServiceEndpoint})  = Set([:chat, :responses, :agentic, :embeddings, :images, :tools, :streaming, :json_output, :files, :vector_stores, :conversations, :moderation, :audio, :transcription, :batch, :image_edits, :fine_tuning, :containers, :uploads, :realtime, :system_one, :models])
provider_capabilities(::Type{AZUREServiceEndpoint})   = Set([:chat, :tools, :streaming, :json_output])
provider_capabilities(::Type{GEMINIOpenAIServiceEndpoint})  = Set([:chat, :embeddings, :tools, :streaming, :json_output])
provider_capabilities(::DeepSeekEndpoint)              = Set([:chat, :tools, :streaming, :fim, :prefix_completion, :json_output])
provider_capabilities(::GenericOpenAIEndpoint)          = Set([:chat, :embeddings, :fim, :tools, :streaming, :json_output, :responses, :transcription])  # permissive default

"""
    has_capability(service, cap::Symbol) -> Bool

Check whether the service endpoint supports a given capability. An endpoint that declares
`:audio` has `:transcription` too: transcription is part of the audio API.
"""
has_capability(service, cap::Symbol)::Bool =
    (caps = provider_capabilities(service); cap in caps || (cap === :transcription && :audio in caps))

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
default_model(::Type{OPENAIServiceEndpoint})  = "gpt-6-sol"
default_model(::Type{AZUREServiceEndpoint})   = "gpt-5.2"
default_model(::Type{GEMINIOpenAIServiceEndpoint})  = "gemini-3.8-flash"
default_model(::DeepSeekEndpoint)              = "deepseek-flash"
default_model(::GenericOpenAIEndpoint)          = nothing
default_model(_) = nothing

"""Default embedding model per provider."""
default_embedding_model(::Type{OPENAIServiceEndpoint})  = "text-embedding-3-small"
default_embedding_model(::Type{GEMINIOpenAIServiceEndpoint})  = "gemini-embedding-2"
default_embedding_model(::DeepSeekEndpoint)              = nothing
default_embedding_model(::GenericOpenAIEndpoint)          = nothing
default_embedding_model(_) = nothing

"""Default image generation model per provider."""
default_image_model(::Type{OPENAIServiceEndpoint}) = "gpt-image-2.5-sunburst"
default_image_model(::Type{GEMINIServiceEndpoint}) = "gemini-nano-banana-2.1"
default_image_model(_) = nothing

"""Default FIM model per provider. DeepSeek serves FIM (beta) on `deepseek-flash`."""
default_fim_model(::DeepSeekEndpoint)      = "deepseek-flash"
default_fim_model(::GenericOpenAIEndpoint)  = nothing
default_fim_model(_) = nothing

"""Resolve model from sentinel (empty string) to service default, or throw if no default."""
function _resolve_model(service, model::String)
    !isempty(model) && return model
    dm = default_model(service)
    isnothing(dm) && throw(ArgumentError("model must be specified when using $(_service_name(service))" *
                                         _model_hint(service)))
    dm
end

# What to name in the missing-model error, where an endpoint has more to say.
_model_hint(_) = ""

_model_family(model::AbstractString, family::AbstractString) =
    model == family || startswith(model, family * "-")

# ─── OpenAI model rules ───────────────────────────────────────────────────────
# Model-specific restrictions belong to the native endpoint. A compatible
# server or Azure deployment may use the same name with a different contract.

# What one OpenAI model accepts on the native endpoint, from its model page and OpenAI's
# GPT-6 guide (https://developers.openai.com/api/docs/guides/latest-model). `efforts` are
# the reasoning efforts the Responses API takes for it, `chat_efforts` those Chat
# Completions takes; the other `*_efforts` fields list the efforts at which the model takes
# a feature — `nothing` for any effort, empty for none — and a request that sets no effort
# runs at `default_effort`.
@kwdef struct _OpenAIModelRules
    efforts::Union{Nothing,Vector{String}} = nothing            # reasoning efforts on Responses
    chat_efforts::Union{Nothing,Vector{String}} = nothing       # reasoning efforts on Chat Completions
    sampling_efforts::Union{Nothing,Vector{String}} = nothing   # temperature, top_p, log probabilities
    chat_tool_efforts::Union{Nothing,Vector{String}} = nothing  # function tools on Chat Completions
    ultrafast::Bool = false                                     # service_tier "ultrafast" on Responses
    prompt_cache_retention::Bool = true                         # the deprecated prompt_cache_retention
    default_effort::String = "medium"
end

# One row per model; `_openai_row` finds a model's row. The effort lists are the API's answers:
# on 2026-10-22 Chat Completions refused "max" on every model below with HTTP 400 ("Supported
# values are: 'none', 'low', 'medium', 'high', and 'xhigh'"; without 'none' on Astra and 6.1
# Sol), though its reference lists "max"; the Responses API took "max" and refused "minimal".
const _OPENAI_MODEL_RULES = let
    # GPT-6 Astra and GPT-6.1 Sol reason at "low" or more, so they take no sampling control or
    # log probability, and tools only through the Responses API; both serve Ultrafast.
    top = _OpenAIModelRules(efforts=["low", "medium", "high", "xhigh", "max"],
                            chat_efforts=["low", "medium", "high", "xhigh"], sampling_efforts=String[],
                            chat_tool_efforts=String[], ultrafast=true, prompt_cache_retention=false)
    # GPT-6 Sol and Luna take sampling controls, log probabilities and Chat tools only at "none".
    sol_luna = _OpenAIModelRules(efforts=["none", "low", "medium", "high", "xhigh", "max"],
                                 chat_efforts=["none", "low", "medium", "high", "xhigh"],
                                 sampling_efforts=["none"], chat_tool_efforts=["none"],
                                 prompt_cache_retention=false)
    # GPT-5.6 takes Chat Completions tools only at "none". The `gpt-5.6` alias "routes requests
    # to GPT-5.6 Sol" (its model page), so it takes Sol's rules.
    gpt56 = _OpenAIModelRules(efforts=["none", "low", "medium", "high", "xhigh", "max"],
                              chat_efforts=["none", "low", "medium", "high", "xhigh"],
                              chat_tool_efforts=["none"], prompt_cache_retention=false)
    Dict{String,_OpenAIModelRules}("gpt-6-astra" => top, "gpt-6.1-sol" => top,
                                   "gpt-6-sol" => sol_luna, "gpt-6-luna" => sol_luna,
                                   "gpt-5.6-sol" => gpt56, "gpt-5.6-terra" => gpt56, "gpt-5.6-luna" => gpt56,
                                   "gpt-5.6" => gpt56)
end

# A model's row in one of OpenAI's per-model tables: its exact id, else the id without a
# dated snapshot suffix (OpenAI names snapshots `gpt-6-sol-2026-09-14`), else none. A row
# never passes to another generation (`gpt-6.1-sol` has its own), and a model without one
# gets no per-model rule: the server's answer is the check.
function _openai_row(table::Dict{String,T}, model::AbstractString)::Union{T,Nothing} where {T}
    row = get(table, model, nothing)
    isnothing(row) || return row
    undated = replace(model, r"-\d{4}-\d{2}-\d{2}$" => "")
    undated == model ? nothing : get(table, undated, nothing)
end

# `sampling`: the names of the sampling-control and log-probability fields a request
# sets (`logprobs=false` requests none), so the error names them.
_set_fields(fields::Pair{String,Bool}...) = String[name for (name, set) in fields if set]

# The two native OpenAI APIs a request reaches: the service tiers each accepts (Chat
# Completions and Responses references; "priority" is Fast mode's former name, still
# accepted), which effort list of a model's row each reads, and where an effort it refuses
# can go instead.
const _OPENAI_APIS = (
    chat=(name="Chat Completions", tiers=("auto", "default", "flex", "scale", "priority", "fast"),
          efforts=(rules::_OpenAIModelRules) -> rules.chat_efforts,
          elsewhere=(rules::_OpenAIModelRules) -> rules.efforts),
    responses=(name="Responses", tiers=("auto", "default", "flex", "scale", "priority", "fast", "ultrafast"),
               efforts=(rules::_OpenAIModelRules) -> rules.efforts,
               elsewhere=(::_OpenAIModelRules) -> nothing))

# A model without a row gets none of the per-model checks below.
_check_openai_effort(::String, ::Nothing, ::Union{String,Nothing}, ::NamedTuple) = nothing
_check_openai_sampling(::String, ::Nothing, ::Union{String,Nothing}, ::Vector{String}, ::Function) = nothing
_check_openai_chat_tools(::String, ::Nothing, ::Union{String,Nothing}, ::Bool) = nothing
_check_openai_retention(::String, ::Nothing, ::Union{String,Nothing}) = nothing

# An effort the model does not list on this API is refused, naming the next one up that it
# accepts — OpenAI's GPT-6 guide moves a "none" (Astra, 6.1 Sol) or "minimal" request to
# "low" — or, where none is left, Respond when the Responses API takes it ("max").
function _check_openai_effort(model::String, rules::_OpenAIModelRules, effort::Union{String,Nothing},
                              api::NamedTuple)::Nothing
    accepted = api.efforts(rules)
    (isnothing(accepted) || isnothing(effort) || effort in accepted) && return nothing
    rank = something(findfirst(==(effort), _OPENAI_REASONING_EFFORTS), 0)
    up = findfirst(in(accepted), _OPENAI_REASONING_EFFORTS[rank+1:end])
    elsewhere = api.elsewhere(rules)
    hint = !isnothing(up) ? "; use $(repr(_OPENAI_REASONING_EFFORTS[rank+up]))" :
           !isnothing(elsewhere) && effort in elsewhere ? "; the Responses API takes it: send the request with Respond" : ""
    throw(ArgumentError("$model on OpenAI $(api.name) does not take reasoning effort $(repr(effort)) " *
                        "(it takes $(join(repr.(accepted), ", ")))$hint"))
end

# GPT-6 guide: "When reasoning effort is not none, remove temperature, top_p, and
# top_logprobs", and the log probabilities. `setting(effort)` is how the request asks for
# an effort.
function _check_openai_sampling(model::String, rules::_OpenAIModelRules, effort::Union{String,Nothing},
                                set::Vector{String}, setting::Function)::Nothing
    allowed = rules.sampling_efforts
    (isnothing(allowed) || isempty(set) || something(effort, rules.default_effort) in allowed) && return nothing
    fields = join(set, ", ")
    isempty(allowed) && throw(ArgumentError("$model does not support sampling controls or log probabilities ($fields)"))
    shown = isnothing(effort) ? "\"$(rules.default_effort) (default)\"" : repr(effort)
    throw(ArgumentError("$model with reasoning effort $shown does not support sampling controls or log " *
                        "probabilities ($fields); set $(setting(first(allowed))) or remove them"))
end

# GPT-6 guide: "Use the Responses API for tool calling"; GPT-6 Sol and Luna (and GPT-5.6)
# take Chat Completions function calling "only with reasoning_effort: none".
function _check_openai_chat_tools(model::String, rules::_OpenAIModelRules, effort::Union{String,Nothing},
                                  tools::Bool)::Nothing
    allowed = rules.chat_tool_efforts
    (isnothing(allowed) || !tools || something(effort, rules.default_effort) in allowed) && return nothing
    isempty(allowed) && throw(ArgumentError("$model takes function tools only through the Responses API: " *
        "send them with Respond (Chat Completions serves $model without tools)"))
    throw(ArgumentError("$model Chat tools require reasoning_effort=$(repr(first(allowed))); " *
                        "use Respond for reasoning with tools"))
end

# The Responses reference marks prompt_cache_retention "Deprecated. Use
# prompt_cache_options.ttl instead." The API still answered it with HTTP 200 on gpt-6-luna
# (2026-10-22), so this refusal is UniLM's, on the models whose lifetime is set there.
function _check_openai_retention(model::String, rules::_OpenAIModelRules,
                                 retention::Union{String,Nothing})::Nothing
    (isnothing(retention) || rules.prompt_cache_retention) && return nothing
    throw(ArgumentError("$model takes its cache lifetime from prompt_cache_options, and OpenAI deprecates " *
        "prompt_cache_retention (\"Use prompt_cache_options.ttl instead\"): set " *
        "prompt_cache_options=PromptCacheOptions(ttl=\"30m\") or PromptCacheOptions(ttl=\"12h\")"))
end

# Ultrafast is a Responses tier of the models whose row serves it
# (https://developers.openai.com/api/docs/guides/ultrafast-mode); the API answered HTTP 400
# "Invalid service_tier argument" for gpt-5.6-sol and gpt-6-luna (2026-10-22).
const _OPENAI_ULTRAFAST_MODELS = sort!([model for (model, rules) in _OPENAI_MODEL_RULES if rules.ultrafast])
_serves_ultrafast(::Nothing) = false
_serves_ultrafast(rules::_OpenAIModelRules) = rules.ultrafast

function _check_openai_service_tier(tier::Union{String,Nothing}, api::NamedTuple, model::String,
                                    rules::Union{_OpenAIModelRules,Nothing})::Nothing
    isnothing(tier) && return nothing
    tier in api.tiers || throw(ArgumentError("OpenAI $(api.name) accepts service_tier " *
        "$(join(repr.(api.tiers), ", ")) (got $(repr(tier)))" * (tier == "ultrafast" ?
        "; \"ultrafast\" is a Responses tier: send it with Respond on $(join(_OPENAI_ULTRAFAST_MODELS, " or "))" : "")))
    tier == "ultrafast" && !_serves_ultrafast(rules) && throw(ArgumentError("$model has no ultrafast tier: " *
        "OpenAI serves service_tier=\"ultrafast\" for $(join(_OPENAI_ULTRAFAST_MODELS, " and ")) " *
        "(other models answer HTTP 400 \"Invalid service_tier argument\")"))
    nothing
end

# Chat Completions and Responses: `user` "is being replaced by safety_identifier and
# prompt_cache_key".
function _check_openai_user(user::Union{String,Nothing})::Nothing
    isnothing(user) || throw(ArgumentError("OpenAI replaces the deprecated user field with safety_identifier " *
        "(a stable id of the end user, at most 128 characters) and prompt_cache_key (cache routing): " *
        "set those instead of user"))
    nothing
end

# The prompt-cache retentions each OpenAI API accepts
# (https://developers.openai.com/api/docs/guides/prompt-caching): Chat Completions answers
# any other ttl with HTTP 400 "Supported values are: '30m'."; the Responses API also
# keeps a prefix for 12 hours; the Decisions API answered a "12h" request with HTTP 200.
const _OPENAI_CACHE_TTLS = (
    chat=(name="Chat Completions", ttls=("30m",), fix="; the Responses API (Respond) also takes \"12h\""),
    responses=(name="Responses", ttls=("30m", "12h"), fix=""),
    decisions=(name="Decisions API", ttls=("30m", "12h"), fix=""))

function _check_openai_cache_ttl(options::Union{PromptCacheOptions,Nothing}, api::NamedTuple)::Nothing
    ttl = isnothing(options) ? nothing : options.ttl
    isnothing(ttl) || ttl in api.ttls || throw(ArgumentError("OpenAI $(api.name) accepts prompt_cache_options " *
        "ttl $(join(repr.(api.ttls), " or ")) (got $(repr(ttl)))$(api.fix)"))
    nothing
end

# Chat Completions deprecates `max_tokens` "in favor of max_completion_tokens", which
# reasoning models require: the native endpoint gets the cap under its current name. A body
# without it is the Chat's own JSON, byte for byte.
function _openai_chat_json(chat::Chat)::String
    isnothing(chat.max_tokens) && return JSON.json(chat)
    isnothing(chat.max_completion_tokens) || throw(ArgumentError(
        "OpenAI sends max_tokens as max_completion_tokens, its current name; set one of them, not both"))
    body = JSON.lower(chat)
    body[:max_completion_tokens] = pop!(body, :max_tokens)
    JSON.json(body)
end

function encode_request(::Type{OPENAIServiceEndpoint}, chat::Chat)::String
    rules = _openai_row(_OPENAI_MODEL_RULES, chat.model)
    effort = chat.reasoning_effort
    _check_openai_effort(chat.model, rules, effort, _OPENAI_APIS.chat)
    _check_openai_sampling(chat.model, rules, effort,
        _set_fields("temperature" => !isnothing(chat.temperature), "top_p" => !isnothing(chat.top_p),
                    "logprobs" => chat.logprobs === true, "top_logprobs" => !isnothing(chat.top_logprobs)),
        e -> "reasoning_effort=$(repr(e))")
    _check_openai_chat_tools(chat.model, rules, effort, !isnothing(chat.tools))
    _check_openai_service_tier(chat.service_tier, _OPENAI_APIS.chat, chat.model, rules)
    _check_openai_user(chat.user)
    # The Chat Completions prompt_cache_options object has only mode and ttl.
    pco = chat.prompt_cache_options
    if !isnothing(pco) && (!isnothing(pco.prewarm) || !isnothing(pco.comparison_response_id))
        throw(ArgumentError("Chat Completions prompt_cache_options accepts only mode and ttl; prewarm and comparison_response_id are Responses-only"))
    end
    _check_openai_cache_ttl(pco, _OPENAI_CACHE_TTLS.chat)
    _openai_chat_json(chat)
end

function encode_agentic(::Type{OPENAIServiceEndpoint}, r::Respond)::String
    rules = _openai_row(_OPENAI_MODEL_RULES, r.model)
    effort = isnothing(r.reasoning) ? nothing : r.reasoning.effort
    logprobs = !isnothing(r.include) && "message.output_text.logprobs" in r.include
    _check_openai_effort(r.model, rules, effort, _OPENAI_APIS.responses)
    _check_openai_sampling(r.model, rules, effort,
        _set_fields("temperature" => !isnothing(r.temperature), "top_p" => !isnothing(r.top_p),
                    "top_logprobs" => !isnothing(r.top_logprobs), "include" => logprobs),
        e -> "reasoning=Reasoning(effort=$(repr(e)))")
    _check_openai_service_tier(r.service_tier, _OPENAI_APIS.responses, r.model, rules)
    _check_openai_retention(r.model, rules, r.prompt_cache_retention)
    _check_openai_user(r.user)
    _check_openai_cache_ttl(r.prompt_cache_options, _OPENAI_CACHE_TTLS.responses)
    _refuse_responses_seed(OPENAIServiceEndpoint, r)
    JSON.json(r)
end

# ─── OpenAI pricing ───────────────────────────────────────────────────────────

# How OpenAI bills a model beyond its standard row in DEFAULT_PRICING
# (https://developers.openai.com/api/docs/pricing, 2026-10-22). `tiers` maps a billed service
# tier to the multiple of every standard rate the page lists for prompts of at most 272K
# input tokens, `long_tiers` the same for longer prompts; a tier missing from them has no
# price there.
@kwdef struct _OpenAIPriceTerms
    cache_writes::Bool = false     # 30-minute writes cost 1.25× input, 12-hour writes 2.5×
    long_context::Bool = false     # over 272K input tokens: input and cache rates 2×, output 1.5×
    tiers::Dict{String,Float64} = Dict{String,Float64}()
    long_tiers::Dict{String,Float64} = Dict{String,Float64}()
end

const _OPENAI_PRICE_TERMS = let
    # GPT-6 and GPT-5.6 list cache-write and long-context rates, Flex at half and Fast at twice
    # the standard rates for prompts of every length, and Ultrafast at six times on GPT-6
    # Astra and GPT-6.1 Sol.
    current(more::Pair{String,Float64}...) = (t = Dict{String,Float64}("flex" => 0.5, "fast" => 2.0, more...);
        _OpenAIPriceTerms(cache_writes=true, long_context=true, tiers=t, long_tiers=t))
    Dict{String,_OpenAIPriceTerms}(
        "gpt-6-astra" => current("ultrafast" => 6.0), "gpt-6.1-sol" => current("ultrafast" => 6.0),
        "gpt-6-sol" => current(), "gpt-6-luna" => current(),
        "gpt-5.6-sol" => current(), "gpt-5.6-terra" => current(), "gpt-5.6-luna" => current(),
        # Earlier models bill no cache-write premium; Fast has no long-context price on the page.
        "gpt-5.5" => _OpenAIPriceTerms(long_context=true, tiers=Dict("flex" => 0.5, "fast" => 2.5),
                                       long_tiers=Dict("flex" => 0.5)),
        "gpt-5.5-pro" => _OpenAIPriceTerms(long_context=true, tiers=Dict("flex" => 0.5)),
        "gpt-5.4" => _OpenAIPriceTerms(long_context=true, tiers=Dict("flex" => 0.5, "fast" => 2.0),
                                       long_tiers=Dict("flex" => 0.5)),
        "gpt-5.4-pro" => _OpenAIPriceTerms(long_context=true, tiers=Dict("flex" => 0.5),
                                           long_tiers=Dict("flex" => 0.5)),
        "gpt-5.4-mini" => _OpenAIPriceTerms(tiers=Dict("flex" => 0.5, "fast" => 2.0)),
        "gpt-5.4-nano" => _OpenAIPriceTerms(tiers=Dict("flex" => 0.5)),           # shuts down 2027-04-01
        "gpt-5.2" => _OpenAIPriceTerms(tiers=Dict("flex" => 0.5, "fast" => 2.0)),
        "gpt-4.1" => _OpenAIPriceTerms(tiers=Dict("fast" => 1.75)),               # shuts down 2027-01-14
        "gpt-4.1-mini" => _OpenAIPriceTerms(tiers=Dict("fast" => 1.75)),          # shuts down 2027-01-14
        "o3" => _OpenAIPriceTerms(tiers=Dict("flex" => 0.5, "fast" => 1.75)))     # o3-2025-04-16: 2026-12-11
end

# The audio-token rates, USD per token, of gpt-audio-1.5, the Chat Completions audio model UniLM
# documents (https://developers.openai.com/api/docs/pricing, 2026-10-22, "Realtime and audio
# generation models"): `input` bills `prompt_audio_tokens`, `output` bills
# `completion_audio_tokens`, and the rest of the call bills at the model's text row in
# DEFAULT_PRICING. The page lists no cached-input rate for it, so a cache read bills as input.
const _OpenAIAudioRates = @NamedTuple{input::Float64, output::Float64}
const _OPENAI_AUDIO_RATES = Dict{String,_OpenAIAudioRates}(
    "gpt-audio-1.5" => (input = 32.0 / 1_000_000, output = 64.0 / 1_000_000))

# `audio_in` input and `audio_out` output audio tokens at the audio rates, each side scaled as
# the text rates on that side are. Assumes the billed tier and a long prompt scale the audio
# rates as they scale the text rates; re-check when an audio model gains price terms.
_audio_bill(::Nothing, ::Int, ::Int, ::Float64, ::Float64) = 0.0
_audio_bill(r::_OpenAIAudioRates, audio_in::Int, audio_out::Int, input::Float64, output::Float64) =
    input * r.input * audio_in + output * r.output * audio_out

# Prompts of at most this many input tokens bill at the short-context rates.
const _OPENAI_SHORT_CONTEXT = 272_000

_long_context(::Nothing, ::Int) = false
_long_context(terms::_OpenAIPriceTerms, prompt_tokens::Int) =
    terms.long_context && prompt_tokens > _OPENAI_SHORT_CONTEXT
_cache_write_multiples(::Nothing) = (1.0, 1.0)
_cache_write_multiples(terms::_OpenAIPriceTerms) = terms.cache_writes ? (1.25, 2.5) : (1.0, 1.0)

# The multiple of the standard rates the billed tier costs. GPT-5.6 and earlier report Fast
# mode as "priority", its former name. A tier the pricing page gives no number for is
# estimated at the standard rates, and says so once per model and tier.
function _openai_tier_multiple(terms::Union{_OpenAIPriceTerms,Nothing}, model::String,
                               reported::Union{String,Nothing}, long::Bool)::Float64
    (isnothing(reported) || reported == "default") && return 1.0
    tier = reported == "priority" ? "fast" : reported
    multiple = isnothing(terms) ? nothing : get(long ? terms.long_tiers : terms.tiers, tier, nothing)
    isnothing(multiple) || return multiple
    @warn "OpenAI's pricing page lists no $(long ? "long-context " : "")price for this service tier of this " *
          "model; its cost is estimated at the standard rates" model tier = reported _id = Symbol(
          "unilm_unpriced_tier_", model, "_", reported, long ? "_long" : "") maxlog = 1
    1.0
end

# OpenAI's bill for `u`: the row's rates with the model's price terms — cache writes at
# 1.25× input for the 30-minute lifetime and 2.5× for 12 hours
# (https://developers.openai.com/api/docs/guides/prompt-caching), a prompt over 272K input
# tokens at twice the input and cache rates and 1.5× the output rate for the whole request,
# and every rate times the billed tier's multiple. The row's rates bill the text tokens; the
# audio tokens bill at the model's audio-token rates, and are never billed at text rates
# (gpt-audio-1.5 bills audio input at 12.8 times and audio output at 6.4 times its text rates):
# a model without audio-token rates leaves a call that counts audio unpriced.
function _billed_cost(service::Type{OPENAIServiceEndpoint}, model::String, u::TokenUsage, rates::PriceRow)::Float64
    audio_rates = _openai_row(_OPENAI_AUDIO_RATES, model)
    _counts_audio(u) && isnothing(audio_rates) && return _unpriced_audio(service, model)
    terms = _openai_row(_OPENAI_PRICE_TERMS, model)
    long = _long_context(terms, u.prompt_tokens)
    tier = _openai_tier_multiple(terms, model, u.service_tier, long)
    input, output = tier * (long ? 2.0 : 1.0), tier * (long ? 1.5 : 1.0)
    w30, w12 = _cache_write_multiples(terms)
    audio_in = min(u.prompt_audio_tokens, u.prompt_tokens)
    audio_out = min(u.completion_audio_tokens, u.completion_tokens)
    text_in = u.prompt_tokens - audio_in
    # Assumes every cache read is text; re-check when an audio model gains a cached rate.
    cached = min(u.cached_tokens, text_in)
    extended = u.cache_write_extended_tokens
    fresh = text_in - cached - u.cache_write_tokens
    input * (rates.input * (fresh + w30 * (u.cache_write_tokens - extended) + w12 * extended) +
             rates.cached_input * cached) + output * rates.output * (u.completion_tokens - audio_out) +
        _audio_bill(audio_rates, audio_in, audio_out, input, output)
end
