# ─── Provider drift (live) ───────────────────────────────────────────────────
# UniLM's model tables against the models OpenAI, Anthropic and Gemini list today: every listed
# model of a family UniLM documents has a price row and, where UniLM keeps per-model rules for
# the family, a rule row; every model UniLM names as a default or in its live suites is listed
# and not past its retirement date; and the Claude rule rows agree with the thinking types and
# effort levels the Models API reports for each model. Model listings bill no tokens. Requires
# UNILM_LIVE=1 and each provider's key.

using Dates

if get(ENV, "UNILM_LIVE", "") != "1"
    @info "Skipping provider drift checks (set UNILM_LIVE=1 and OPENAI_API_KEY, ANTHROPIC_API_KEY, GEMINI_API_KEY to run live)"
else

# The provider of a model id UniLM names.
_drift_provider(id::AbstractString)::Symbol =
    startswith(id, "claude-") ? :anthropic : occursin(r"^(gemini|gemma)-", id) ? :gemini : :openai

# A provider model id quoted in source: OpenAI's, Anthropic's or Gemini's families of names.
const _DRIFT_QUOTED_ID = r"\"((?:gpt|chatgpt|text-embedding|omni-moderation|claude|gemini|gemma)-[a-z0-9.\-]+|o\d[a-z0-9.\-]*)\""

# The models UniLM uses when a call names none: the default-model functions, the speech and
# Decisions defaults, and the keyword defaults its source writes as `model::String="…"`.
function _drift_defaults()::Vector{String}
    named = String[UniLM.default_model(OPENAIServiceEndpoint), UniLM.default_embedding_model(OPENAIServiceEndpoint),
                   UniLM.default_image_model(OPENAIServiceEndpoint), UniLM._default_transcription_model(OPENAIServiceEndpoint),
                   UniLM._REALTIME_SPEECH.model, UniLM._DECISIONS_MODEL, UniLM.default_model(ANTHROPICServiceEndpoint),
                   UniLM.default_model(GEMINIServiceEndpoint), UniLM.default_model(GEMINIOpenAIServiceEndpoint),
                   UniLM.default_embedding_model(GEMINIOpenAIServiceEndpoint), UniLM.default_image_model(GEMINIServiceEndpoint),
                   UniLM._GEMINI_SPEECH.model]
    src = joinpath(pkgdir(UniLM), "src")
    for f in readdir(src; join=true), m in eachmatch(r"\bmodel::(?:Abstract)?String\s*=\s*\"([^\"]+)\"", read(f, String))
        push!(named, m[1])
    end
    unique!(named)
end

# The provider model ids quoted in the live suites (test/integration_*.jl).
_drift_suite_ids()::Vector{String} =
    unique!([String(m[1]) for f in readdir(@__DIR__; join=true) if occursin(r"^integration_.*\.jl$", basename(f))
             for m in eachmatch(_DRIFT_QUOTED_ID, read(f, String))])

# Every model of `provider` that UniLM names is listed and not past its retirement date.
function _drift_named_models(models::Vector{ProviderModel}, provider::Symbol)
    listed = Dict(m.id => m for m in models)
    named = filter(id -> _drift_provider(id) === provider, unique!([_drift_defaults(); _drift_suite_ids()]))
    @info "models UniLM names" provider named
    @test !isempty(named)
    @test isempty([id for id in named if !haskey(listed, id)])
    @test isempty([id => listed[id].retires for id in named
                   if haskey(listed, id) && !isnothing(listed[id].retires) && listed[id].retires <= today()])
end

# OpenAI: GPT-6 and GPT-5.6 have rule rows; they and GPT-5.4 mini (and its dated snapshots) have
# price rows — the standard rates and the price terms.
_openai_ruled(id::AbstractString) = occursin(r"^gpt-(6|5\.6)", id)
_openai_priced(id::AbstractString) = _openai_ruled(id) || occursin(r"^gpt-5\.4-mini(-\d{4}-\d{2}-\d{2})?$", id)

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

# Gemini 3.5 and later text models: a versioned id (gemini-<major>[.<minor>]-…) that serves
# generateContent and thinks — the models UniLM sends thinking levels and prices.
function _gemini_text_family(m::ProviderModel)::Bool
    v = match(r"^gemini-(\d+)(?:\.(\d+))?-", m.id)
    isnothing(v) && return false
    VersionNumber(parse(Int, something(v[1])), parse(Int, something(v[2], "0"))) >= v"3.5" &&
        "generateContent" in something(get(m.capabilities, "supportedGenerationMethods", nothing), String[]) &&
        get(m.capabilities, "thinking", false) === true
end

_unpriced(ids) = [id for id in ids if isnothing(UniLM._price_row(DEFAULT_PRICING, id))]

# The checks of each provider's listing `r`; a listing that failed fails them all.
function _openai_drift(r)
    @test r isa ModelListSuccess
    r isa ModelListSuccess || return
    ids = [m.id for m in r.models]
    priced = filter(_openai_priced, ids)
    @info "OpenAI listing" models = length(ids) pages = length(r.raw) priced
    @test UniLM.default_model(OPENAIServiceEndpoint) in priced
    @test isempty(_unpriced(priced))
    @test isempty([id for id in priced if isnothing(UniLM._openai_row(UniLM._OPENAI_PRICE_TERMS, id))])
    @test isempty([id for id in filter(_openai_ruled, ids) if isnothing(UniLM._openai_row(UniLM._OPENAI_MODEL_RULES, id))])
    _drift_named_models(r.models, :openai)
end

function _anthropic_drift(r)
    @test r isa ModelListSuccess
    r isa ModelListSuccess || return
    @test isempty([m.id for m in r.models if isnothing(_claude_generation(m.id))])
    family = [m for m in r.models if something(_claude_generation(m.id), v"0") >= v"4.5"]
    @info "Anthropic listing" models = length(r.models) pages = length(r.raw) family = [m.id for m in family]
    @test UniLM.default_model(ANTHROPICServiceEndpoint) in [m.id for m in family]
    @test isempty(_unpriced([m.id for m in family]))
    @test isempty([m.id for m in family if isnothing(UniLM._claude_family(m.id))])
    @test isempty([m.id => d for m in family for d in _claude_disagreements(m)])
    _drift_named_models(r.models, :anthropic)
end

function _gemini_drift(r)
    @test r isa ModelListSuccess
    r isa ModelListSuccess || return
    family = [m.id for m in r.models if _gemini_text_family(m)]
    @info "Gemini listing" models = length(r.models) pages = length(r.raw) family
    @test UniLM.default_model(GEMINIServiceEndpoint) in family
    @test isempty(_unpriced(family))
    @test isempty([id for id in family if isnothing(UniLM._gemini_thinking_levels(id))])
    _drift_named_models(r.models, :gemini)
end

if haskey(ENV, "OPENAI_API_KEY")
    @testset "OpenAI: GPT-6, GPT-5.6 and GPT-5.4 mini are priced and ruled; the models UniLM names are listed" begin
        _openai_drift(list_models(; service=OPENAIServiceEndpoint))
    end
else
    @info "Skipping OpenAI drift checks (set OPENAI_API_KEY)"
end

if haskey(ENV, "ANTHROPIC_API_KEY")
    @testset "Anthropic: Claude 4.5 and later are priced, with rule rows that agree with the Models API" begin
        _anthropic_drift(list_models(; service=ANTHROPICServiceEndpoint))
    end
else
    @info "Skipping Anthropic drift checks (set ANTHROPIC_API_KEY)"
end

if haskey(ENV, "GEMINI_API_KEY")
    @testset "Gemini: 3.5 and later text models are priced, with thinking-level rows; the models UniLM names are listed" begin
        _gemini_drift(list_models(; service=GEMINIServiceEndpoint))
    end
else
    @info "Skipping Gemini drift checks (set GEMINI_API_KEY)"
end

end
