# The provider drift checks: UniLM's model tables against a provider's model listing, as
# functions of the listed models that return what the tables lack. The live suite
# (integration_provider_drift.jl) runs them on today's listings, the keyless suite
# (model_listing.jl) on the listings of 2026-10-22 and on broken facts. Definitions only.

using Dates

# The id prefixes of each provider's models. An id UniLM names is checked against the listing of
# the provider whose prefix it carries; an id with none of them — a local Ollama model such as
# gemma4:e4b, a Jev model — is no listing's to check.
const _DRIFT_PREFIXES = (openai=("gpt-", "chatgpt-", "text-embedding-", "omni-moderation-", "o1", "o3", "o4"),
                         anthropic=("claude-",),
                         gemini=("gemini-", "gemma-"))

_drift_provider(id::AbstractString)::Union{Symbol,Nothing} =
    findfirst(prefixes -> any(p -> startswith(id, p), prefixes), _DRIFT_PREFIXES)

# A model id as source quotes it: lower-case words joined by single dots or hyphens.
const _DRIFT_QUOTED_ID = r"\"([a-z0-9]+(?:[.\-][a-z0-9]+)*)\""

# The models UniLM uses when a call names none: the default-model functions, the speech and
# Decisions defaults, and the keyword defaults its source writes as `model::String="…"`.
function _drift_defaults()::Vector{String}
    named = String[UniLM.default_model(OPENAIServiceEndpoint), UniLM.default_embedding_model(OPENAIServiceEndpoint),
                   UniLM.default_image_model(OPENAIServiceEndpoint), UniLM._default_transcription_model(OPENAIServiceEndpoint),
                   UniLM._REALTIME_SPEECH.model, UniLM._DECISIONS_MODEL, UniLM.default_model(ANTHROPICServiceEndpoint),
                   UniLM.default_model(GEMINIServiceEndpoint), UniLM.default_model(GEMINIOpenAIServiceEndpoint),
                   UniLM.default_embedding_model(GEMINIOpenAIServiceEndpoint), UniLM.default_image_model(GEMINIServiceEndpoint),
                   UniLM._GEMINI_SPEECH.model]
    for f in readdir(joinpath(pkgdir(UniLM), "src"); join=true)
        isfile(f) && endswith(f, ".jl") || continue
        for m in eachmatch(r"\bmodel::(?:Abstract)?String\s*=\s*\"([^\"]+)\"", read(f, String))
            push!(named, m[1])
        end
    end
    unique!(named)
end

# The model ids the live suites (test/integration_*.jl) quote.
function _drift_suite_ids()::Vector{String}
    ids = String[]
    for f in readdir(@__DIR__; join=true)
        isfile(f) && occursin(r"^integration_.*\.jl$", basename(f)) || continue
        for m in eachmatch(_DRIFT_QUOTED_ID, read(f, String))
            _drift_provider(m[1]) === nothing || push!(ids, m[1])
        end
    end
    unique!(ids)
end

# The models of `provider` among those UniLM names, the ones `models` does not list, and the
# listed ones whose retirement date has come.
function _drift_named_models(models::Vector{ProviderModel}, provider::Symbol;
                             named::Vector{String}=[_drift_defaults(); _drift_suite_ids()])
    listed = Dict(m.id => m for m in models)
    ids = unique!(filter(id -> _drift_provider(id) === provider, named))
    retired = [id => listed[id].retires for id in ids
               if haskey(listed, id) && !isnothing(listed[id].retires) && listed[id].retires <= today()]
    (; named=ids, unlisted=[id for id in ids if !haskey(listed, id)], retired)
end

# The ids among `ids` that the cost code's lookup prices at no row.
_unpriced(ids) = [id for id in ids if isnothing(UniLM._price_row(DEFAULT_PRICING, id))]

# OpenAI: GPT-6 and GPT-5.6 have rule rows; they and GPT-5.4 mini (and its dated snapshots) have
# price rows — the standard rates and the price terms.
_openai_ruled(id::AbstractString) = occursin(r"^gpt-(6|5\.6)", id)
_openai_priced(id::AbstractString) = _openai_ruled(id) || occursin(r"^gpt-5\.4-mini(-\d{4}-\d{2}-\d{2})?$", id)

function _openai_drift(models::Vector{ProviderModel})
    ids = [m.id for m in models]
    family = filter(_openai_priced, ids)
    (; family, unpriced=_unpriced(family),
     no_terms=[id for id in family if isnothing(UniLM._openai_row(UniLM._OPENAI_PRICE_TERMS, id))],
     unruled=[id for id in filter(_openai_ruled, ids) if isnothing(UniLM._openai_row(UniLM._OPENAI_MODEL_RULES, id))])
end

# The generation a Claude id names: claude-<line>-<major>[-<minor>] or the older
# claude-<major>[-<minor>]-<line>, either with an optional -YYYYMMDD snapshot; `nothing` for an id
# in neither form.
function _claude_generation(id::AbstractString)::Union{VersionNumber,Nothing}
    m = match(r"^claude-(?:[a-z]+-(\d+)(?:-(\d))?|(\d+)(?:-(\d))?-[a-z]+)(?:-\d{8})?$", id)
    isnothing(m) && return nothing
    VersionNumber(parse(Int, something(m[1], m[3])), parse(Int, something(m[2], m[4], "0")))
end

# What each Claude thinking mode (`_ClaudeFamily.thinking`, src/anthropic.jl) says the Messages API
# accepts: "adaptive", which UniLM sends with a reasoning effort on every mode but :manual (budget
# thinking only); "disabled", a 400 where thinking cannot be turned off (:always, :between) and
# accepted where thinking is off by default; "between_tools", the off setting of :between.
const _CLAUDE_THINKING_TYPES = Dict(
    :always  => (adaptive=true,  disabled=false, between_tools=false),
    :on      => (adaptive=true,  disabled=true,  between_tools=false),
    :between => (adaptive=true,  disabled=false, between_tools=true),
    :off     => (adaptive=true,  disabled=true,  between_tools=false),
    :manual  => (adaptive=false, disabled=true,  between_tools=false))

_listed_supported(c) = c isa AbstractDict && get(c, "supported", false) === true

# Where a Claude row and the model's listed capabilities disagree, one line each.
function _claude_disagreements(m::ProviderModel)::Vector{String}
    fam = UniLM._claude_family(m.id)
    isnothing(fam) && return String[]   # reported as a missing row
    thinking, effort = get(m.capabilities, "thinking", nothing), get(m.capabilities, "effort", nothing)
    thinking isa AbstractDict && effort isa AbstractDict || return ["no thinking or effort capabilities listed"]
    types = something(get(thinking, "types", nothing), Dict{String,Any}())
    out = String[]
    for (type, accepted) in pairs(_CLAUDE_THINKING_TYPES[fam.thinking])
        listed = _listed_supported(get(types, String(type), nothing))
        listed == accepted || push!(out, "thinking :$(fam.thinking) $(accepted ? "takes" : "refuses") $type, listed $listed")
    end
    levels = sort!([String(k) for (k, v) in effort if k != "supported" && _listed_supported(v)])
    Set(levels) == Set(fam.efforts) || push!(out, "efforts $(fam.efforts), listed $levels")
    out
end

# Anthropic: Claude 4.5 and later, by the generation the id names, and every id UniLM keeps a rule
# row for, whatever its name says. An id with neither is one whose generation cannot be read.
function _anthropic_drift(models::Vector{ProviderModel})
    ruled(m) = !isnothing(UniLM._claude_family(m.id))
    family = [m for m in models if ruled(m) || something(_claude_generation(m.id), v"0") >= v"4.5"]
    ids = [m.id for m in family]
    (; family=ids, unreadable=[m.id for m in models if !ruled(m) && isnothing(_claude_generation(m.id))],
     unpriced=_unpriced(ids), unruled=[m.id for m in family if !ruled(m)],
     disagreements=[m.id => d for m in family for d in _claude_disagreements(m)])
end

# The kinds of Gemini model that are not text models, each a segment of their ids: speech
# (gemini-3.8-flash-tts), transcription (gemini-3.5-transcribe), image generation (…-image) and the
# Live API's models (gemini-3.8-live).
const _GEMINI_NOT_TEXT = ("tts", "transcribe", "image", "live")

# Gemini 3.5 and later text models: a versioned id (gemini-<major>[.<minor>]-…) of no other kind
# that serves generateContent. Each has a price row; the ones that think also a row of thinking
# levels, which UniLM checks a reasoning effort against.
function _gemini_text_model(m::ProviderModel)::Bool
    v = match(r"^gemini-(\d+)(?:\.(\d+))?-", m.id)
    isnothing(v) && return false
    VersionNumber(parse(Int, something(v[1])), parse(Int, something(v[2], "0"))) >= v"3.5" &&
        !any(in(_GEMINI_NOT_TEXT), split(m.id, '-')) &&
        "generateContent" in something(get(m.capabilities, "supportedGenerationMethods", nothing), String[])
end

function _gemini_drift(models::Vector{ProviderModel})
    text = filter(_gemini_text_model, models)
    family = [m.id for m in text]
    thinking = [m.id for m in text if get(m.capabilities, "thinking", false) === true]
    (; family, thinking, unpriced=_unpriced(family),
     unruled=[id for id in thinking if isnothing(UniLM._gemini_thinking_levels(id))])
end
