# ─── Provider drift (live) ───────────────────────────────────────────────────
# UniLM's model tables against the models OpenAI, Anthropic and Gemini list today: every listed
# model of a family UniLM documents has a price row (unless its provider publishes none) and,
# where UniLM keeps per-model rules for the family, a rule row; every model UniLM names as a
# default, in its live suites or in the docs' Julia code is listed, and before its retirement
# date where the listing gives one — the check fails from the shutdown day on, and Gemini's
# listing carries no dates, so there it checks only that the model is listed; and the Claude
# rule rows agree with the thinking types and effort levels the Models API reports for each
# model. The OpenAI and Gemini models that no family, older generation or known kind covers — a
# new generation — are logged, not failed. Model listings bill no tokens. Requires UNILM_LIVE=1
# and each provider's key. The checks are in provider_drift_checks.jl, which the keyless suite
# runs on the listings of 2026-10-22.

isdefined(@__MODULE__, :ProviderDriftChecks) || include("provider_drift_checks.jl")
using .ProviderDriftChecks

# Every model of `provider` that UniLM names is listed, and before its retirement date where the
# listing gives one.
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
        # The unclassified ids as one string: in CI's 24-row log a vector prints as "N-element Vector{String}: …".
        @info "OpenAI listing" models = length(r.models) pages = length(r.raw) family = d.family unclassified = join(d.unclassified, ", ")
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
        # The unclassified ids as one string: in CI's 24-row log a vector prints as "N-element Vector{String}: …".
        @info "Gemini listing" models = length(r.models) pages = length(r.raw) family = d.family thinking = d.thinking unclassified = join(d.unclassified, ", ")
        @test UniLM.default_model(GEMINIServiceEndpoint) in d.thinking
        @test isempty(d.unpriced)
        @test isempty(d.unruled)
        _test_named_models(r.models, :gemini)
    end
else
    @info "Skipping Gemini drift checks (set GEMINI_API_KEY)"
end

end
