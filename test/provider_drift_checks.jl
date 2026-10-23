# The provider drift checks: UniLM's model tables against a provider's model listing, as
# functions of the listed models that return what the tables lack. The live suite
# (integration_provider_drift.jl) runs them on today's listings, the keyless suite
# (model_listing.jl) on the listings of 2026-10-22 and on broken facts. Definitions only, in a
# module that each of those files loads once per session, so a run that includes both defines
# every method once:
#
#     isdefined(@__MODULE__, :ProviderDriftChecks) || include("provider_drift_checks.jl")
#     using .ProviderDriftChecks

module ProviderDriftChecks

using Dates
using UniLM

export _drift_provider, _drift_defaults, _drift_suite_ids, _drift_docs_ids, _drift_named_models,
       _openai_drift, _anthropic_drift, _gemini_drift

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

# The ids of a provider's models that `text` quotes.
_drift_quoted_ids(text::AbstractString)::Vector{String} =
    String[id for id in (something(m[1]) for m in eachmatch(_DRIFT_QUOTED_ID, text)) if !isnothing(_drift_provider(id))]

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
        isfile(f) && occursin(r"^integration_.*\.jl$", basename(f)) && append!(ids, _drift_quoted_ids(read(f, String)))
    end
    unique!(ids)
end

# A block a docs page fences as ```julia, ```@example or ```@repl, at any indentation (the blocks
# of an admonition are indented), and its code. Nothing else is read: not prose, where migration
# notes name retired models on purpose, and no other fence (```@setup, ```julia-repl, ```jldoctest,
# ```@docs, ```bash, …).
const _DRIFT_DOCS_CODE = r"^[ \t]*```(?:julia|@example|@repl)(?:[ \t][^\n]*)?\n(.*?)^[ \t]*```"ms

# The model ids the Julia code blocks of the docs pages under `dir` quote.
function _drift_docs_ids(dir::AbstractString=joinpath(pkgdir(UniLM), "docs", "src"))::Vector{String}
    ids = String[]
    for (root, _, files) in walkdir(dir), f in files
        endswith(f, ".md") || continue
        for block in eachmatch(_DRIFT_DOCS_CODE, read(joinpath(root, f), String))
            append!(ids, _drift_quoted_ids(something(block[1])))
        end
    end
    unique!(ids)
end

# The listed models by the ids a request may name them by: each listed id, and the alias of a
# dated snapshot (-YYYYMMDD) that the listing does not name itself — "for models before the 4.6
# generation, the alias is a convenience pointer that resolves to the dated ID"
# (https://platform.claude.com/docs/en/models/overview), as claude-haiku-4-5 resolves to the
# listed claude-haiku-4-5-20251001 (of two snapshots, to the later). A snapshot id is listed
# only as itself.
function _drift_listed(models::Vector{ProviderModel})::Dict{String,ProviderModel}
    listed = Dict{String,ProviderModel}()
    for m in sort(models; by=m -> m.id)
        alias = replace(m.id, r"-\d{8}$" => "")
        alias == m.id || (listed[alias] = m)
    end
    for m in models
        listed[m.id] = m
    end
    listed
end

# The models of `provider` among those UniLM names, the ones `models` does not list, and the
# listed ones whose retirement date has come, the shutdown day included.
function _drift_named_models(models::Vector{ProviderModel}, provider::Symbol;
                             named::Vector{String}=[_drift_defaults(); _drift_suite_ids(); _drift_docs_ids()])
    listed = _drift_listed(models)
    ids = unique!(filter(id -> _drift_provider(id) === provider, named))
    retired = [id => listed[id].retires for id in ids
               if haskey(listed, id) && !isnothing(listed[id].retires) && listed[id].retires <= today()]
    (; named=ids, unlisted=[id for id in ids if !haskey(listed, id)], retired)
end

# Models whose provider publishes no price, so no price row can hold one, by exact id: Claude
# Mythos Preview, a gated research preview ("Access is invitation-only",
# https://platform.claude.com/docs/en/release-notes/overview) that the pricing page
# (https://platform.claude.com/docs/en/about-claude/pricing.md) gives no rate.
const _NO_PUBLISHED_PRICE = ("claude-mythos-preview",)

# The ids among `ids` that the cost code's lookup prices at no row, those with no published
# price aside.
_unpriced(ids) = [id for id in ids if id ∉ _NO_PUBLISHED_PRICE && isnothing(UniLM._price_row(DEFAULT_PRICING, id))]

# OpenAI: GPT-6 and GPT-5.6 have rule rows; they and GPT-5.4 mini (and its dated snapshots) have
# price rows — the standard rates and the price terms.
_openai_ruled(id::AbstractString) = occursin(r"^gpt-(6|5\.6)", id)
_openai_priced(id::AbstractString) = _openai_ruled(id) || occursin(r"^gpt-5\.4-mini(-\d{4}-\d{2}-\d{2})?$", id)

# The kinds of OpenAI model that are not text models, each a segment of their ids: image
# generation (gpt-image-2, chatgpt-image-latest), audio (gpt-audio-1.5), the Realtime and Live
# APIs' models (gpt-realtime-2.1, gpt-live-1), transcription (gpt-transcribe) and speech
# (gpt-4o-mini-tts).
const _OPENAI_NOT_TEXT = ("image", "audio", "realtime", "live", "transcribe", "tts")

# An id of an older generation than the family: a GPT version before 5.6
# (gpt-<major>[.<minor>]…, gpt-4o reading 4), or the o1, o3 and o4 series.
function _openai_older(id::AbstractString)::Bool
    occursin(r"^o[134](-|$)", id) && return true
    v = match(r"^gpt-(\d+)(?:\.(\d+))?(?:o|-|$)", id)
    !isnothing(v) && VersionNumber(parse(Int, something(v[1])), parse(Int, something(v[2], "0"))) < v"5.6"
end

# The chat model ids — gpt-…, chatgpt-…, o<n>… — that no family, older generation or other known
# kind covers: a new generation or a name of a new form, logged by the live suite.
_openai_unclassified(ids) = [id for id in ids if occursin(r"^(gpt-|chatgpt-|o\d)", id) && !_openai_priced(id) &&
                             !_openai_older(id) && !any(in(_OPENAI_NOT_TEXT), split(id, '-'))]

function _openai_drift(models::Vector{ProviderModel})
    ids = [m.id for m in models]
    family = filter(_openai_priced, ids)
    (; family, unpriced=_unpriced(family),
     no_terms=[id for id in family if isnothing(UniLM._openai_row(UniLM._OPENAI_PRICE_TERMS, id))],
     unruled=[id for id in filter(_openai_ruled, ids) if isnothing(UniLM._openai_row(UniLM._OPENAI_MODEL_RULES, id))],
     unclassified=_openai_unclassified(ids))
end

# The generation a Claude id names: claude-<line>-<major>[-<minor>] or the older
# claude-<major>[-<minor>]-<line>, either with an optional -YYYYMMDD snapshot; `nothing` for an id
# in neither form.
function _claude_generation(id::AbstractString)::Union{VersionNumber,Nothing}
    m = match(r"^claude-(?:[a-z]+-(\d+)(?:-(\d))?|(\d+)(?:-(\d))?-[a-z]+)(?:-\d{8})?$", id)
    isnothing(m) && return nothing
    VersionNumber(parse(Int, something(m[1], m[3])), parse(Int, something(m[2], m[4], "0")))
end

# The thinking types the Messages API accepts on each Claude thinking mode (`_ClaudeFamily.thinking`,
# src/anthropic.jl), as a listing's `capabilities.thinking.types` shows them: "adaptive", which UniLM
# sends with a reasoning effort, on every mode but :manual (extended thinking only); "disabled" on
# every mode whose thinking can be turned off (:on, :off, :manual), a 400 on :always and :between;
# "between_tools", the off setting of :between, on :between alone. :on and :off accept the same
# types: they differ in whether thinking is on by default, which the listing does not show.
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
# (gemini-3.8-flash-tts), transcription (gemini-3.5-transcribe), image generation (…-image, and
# Nano Banana: gemini-nano-banana-2.1), video generation (Gemini Omni: gemini-omni-1.1-flash) and
# the Live API's models (gemini-3.8-live).
const _GEMINI_NOT_TEXT = ("tts", "transcribe", "image", "banana", "omni", "live")

# The generation a Gemini id names, gemini-<major>[.<minor>]-…; `nothing` for an id of another
# form (gemini-flash-latest, gemini-robotics-er-2-preview).
function _gemini_generation(id::AbstractString)::Union{VersionNumber,Nothing}
    v = match(r"^gemini-(\d+)(?:\.(\d+))?-", id)
    isnothing(v) ? nothing : VersionNumber(parse(Int, something(v[1])), parse(Int, something(v[2], "0")))
end

# A Gemini text model: a gemini-… id of no other kind that serves generateContent.
_gemini_text_model(m::ProviderModel)::Bool =
    startswith(m.id, "gemini-") && !any(in(_GEMINI_NOT_TEXT), split(m.id, '-')) &&
    "generateContent" in something(get(m.capabilities, "supportedGenerationMethods", nothing), String[])

# Gemini: the text models of 3.5 and later, by the generation the id names, and every text model
# UniLM keeps thinking levels for, whatever its name says — but the -latest aliases, which are
# priced at the model that served them. Each needs a price row, and the ones that think a row of
# thinking levels too, which UniLM checks a reasoning effort against. A text model of neither
# whose generation cannot be read is unclassified: a new name, logged by the live suite.
function _gemini_drift(models::Vector{ProviderModel})
    ruled(id) = !isnothing(UniLM._gemini_thinking_levels(id))
    text = [m for m in models if _gemini_text_model(m) && !endswith(m.id, "-latest")]
    members = [m for m in text if ruled(m.id) || something(_gemini_generation(m.id), v"0") >= v"3.5"]
    family = [m.id for m in members]
    thinking = [m.id for m in members if get(m.capabilities, "thinking", false) === true]
    (; family, thinking, unpriced=_unpriced(family), unruled=[id for id in thinking if !ruled(id)],
     unclassified=[m.id for m in text if !ruled(m.id) && isnothing(_gemini_generation(m.id))])
end

end
