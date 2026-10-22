# ─── Provider drift (live) ───────────────────────────────────────────────────
# UniLM's model tables against the models OpenAI, Anthropic and Gemini list today: every listed
# model of a family UniLM documents has a price row and, where UniLM keeps per-model rules for
# the family, a rule row; every model UniLM names as a default or in its live suites is listed
# and not past its retirement date; and the Claude rule rows agree with the thinking types and
# effort levels the Models API reports for each model. Model listings bill no tokens. Requires
# UNILM_LIVE=1 and each provider's key. The checks are in provider_drift_checks.jl, which the
# keyless suite runs on the listings of 2026-10-22.

include("provider_drift_checks.jl")

# Every model of `provider` that UniLM names is listed and not past its retirement date.
function _test_named_models(models::Vector{ProviderModel}, provider::Symbol)
    n = _drift_named_models(models, provider)
    @info "models UniLM names" provider named = n.named
    @test !isempty(n.named)
    @test isempty(n.unlisted)
    @test isempty(n.retired)
end

if get(ENV, "UNILM_LIVE", "") != "1"
    @info "Skipping provider drift checks (set UNILM_LIVE=1 and OPENAI_API_KEY, ANTHROPIC_API_KEY, GEMINI_API_KEY to run live)"
else

if haskey(ENV, "OPENAI_API_KEY")
    @testset "OpenAI: GPT-6, GPT-5.6 and GPT-5.4 mini are priced and ruled; the models UniLM names are listed" begin
        r = list_models(; service=OPENAIServiceEndpoint)
        @test r isa ModelListSuccess
        d = _openai_drift(r.models)
        @info "OpenAI listing" models = length(r.models) pages = length(r.raw) family = d.family
        @test UniLM.default_model(OPENAIServiceEndpoint) in d.family
        @test isempty(d.unpriced)
        @test isempty(d.no_terms)
        @test isempty(d.unruled)
        _test_named_models(r.models, :openai)
    end
else
    @info "Skipping OpenAI drift checks (set OPENAI_API_KEY)"
end

if haskey(ENV, "ANTHROPIC_API_KEY")
    @testset "Anthropic: Claude 4.5 and later are priced, with rule rows that agree with the Models API" begin
        r = list_models(; service=ANTHROPICServiceEndpoint)
        @test r isa ModelListSuccess
        d = _anthropic_drift(r.models)
        @info "Anthropic listing" models = length(r.models) pages = length(r.raw) family = d.family
        @test UniLM.default_model(ANTHROPICServiceEndpoint) in d.family
        @test isempty(d.unreadable)
        @test isempty(d.unpriced)
        @test isempty(d.unruled)
        @test isempty(d.disagreements)
        _test_named_models(r.models, :anthropic)
    end
else
    @info "Skipping Anthropic drift checks (set ANTHROPIC_API_KEY)"
end

if haskey(ENV, "GEMINI_API_KEY")
    @testset "Gemini: 3.5 and later text models are priced, those that think have thinking levels; the models UniLM names are listed" begin
        r = list_models(; service=GEMINIServiceEndpoint)
        @test r isa ModelListSuccess
        d = _gemini_drift(r.models)
        @info "Gemini listing" models = length(r.models) pages = length(r.raw) family = d.family thinking = d.thinking
        @test UniLM.default_model(GEMINIServiceEndpoint) in d.thinking
        @test isempty(d.unpriced)
        @test isempty(d.unruled)
        _test_named_models(r.models, :gemini)
    end
else
    @info "Skipping Gemini drift checks (set GEMINI_API_KEY)"
end

end
