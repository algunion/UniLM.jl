# Provider model listings (`list_models` on OpenAI, Anthropic and Gemini): each listing decodes
# into ProviderModels — ids, names, token limits, capabilities, lifecycle dates — with every
# page fetched, against a local server replaying the listings each provider returned for the
# repository's keys on 2026-10-22 (test/fixtures/models/); and the provider drift checks of the
# live drift suite, on those listings and on broken facts. Offline.

using Dates
using Logging: with_logger, NullLogger
include("provider_drift_checks.jl")

_ml_fixture(name::AbstractString) = read(joinpath(@__DIR__, "fixtures", "models", name), String)
_ml_page(name::AbstractString) = JSON.parse(_ml_fixture(name); dicttype=Dict{String,Any})
const _ML_CFG = RequestConfig(max_attempts=1, total_deadline=30.0)

# An endpoint that lists through provider wire `W`, served by `_with_scripted`.
struct ListingProbe{W} <: UniLM.ServiceEndpoint end
UniLM._model_listing(::Type{ListingProbe{W}}) where {W} = W()
UniLM._models_url(::Type{<:ListingProbe}) = _url_probe_base[] * "/v1/models"
UniLM.auth_header(::Type{<:ListingProbe}) = ["Authorization" => "Bearer t", "Content-Type" => "application/json"]

const _ML_OPENAI = ListingProbe{UniLM._OpenAIModels}
const _ML_ANTHROPIC = ListingProbe{UniLM._AnthropicModels}
const _ML_GEMINI = ListingProbe{UniLM._GeminiModels}

# `list_models` on `service`, its n-th request answered by `respond(n, req)`: (result, requests).
_ml_list(service, respond::Function; config=_ML_CFG, cancel=nothing) =
    _with_scripted(() -> list_models(; service, config, cancel), respond)
_ml_serving(body::AbstractString) = (_, _) -> _json(200, body)
_ml_targets(seen) = [(s.method, s.target) for s in seen]

# Refuted if: a field is read from the wrong key or to the wrong type, a time is shifted from
# UTC, or an entry is dropped or reordered.
@testset "OpenAI's listing: ids, creation times and shutdown dates" begin
    page = _ml_page("openai.json")
    r, seen = _ml_list(_ML_OPENAI, _ml_serving(_ml_fixture("openai.json")))
    @test r isa ModelListSuccess && issuccess(r)
    @test _ml_targets(seen) == [("GET", "/v1/models")]
    @test r.raw == [page]
    @test [m.id for m in r.models] == [e["id"] for e in page["data"]] && length(r.models) == 133
    tts = r.models[1]
    @test (tts.id, tts.display_name, tts.created, tts.deprecated, tts.retires) ==
          ("tts-1", nothing, DateTime(2023, 4, 19, 21, 49, 11), nothing, Date(2027, 1, 6))
    @test (tts.input_token_limit, tts.output_token_limit, tts.capabilities) == (nothing, nothing, Dict{String,Any}())
    @test tts.raw == page["data"][1]
    sol = only(m for m in r.models if m.id == "gpt-6-sol")
    @test (sol.created, sol.retires) == (DateTime(2026, 9, 14, 17, 12, 17), nothing)
    @test count(m -> !isnothing(m.retires), r.models) == 69
    @test sprint(show, tts) == "ProviderModel(\"tts-1\", retires 2027-01-06)"
    @test sprint(show, sol) == "ProviderModel(\"gpt-6-sol\")"
end

@testset "Anthropic's listing: display names, token limits and capability objects" begin
    page = _ml_page("anthropic.json")
    r, seen = _ml_list(_ML_ANTHROPIC, _ml_serving(_ml_fixture("anthropic.json")))
    @test r isa ModelListSuccess
    @test _ml_targets(seen) == [("GET", "/v1/models")]
    @test r.raw == [page]
    @test [m.id for m in r.models] == [e["id"] for e in page["data"]] && length(r.models) == 13
    h = r.models[1]
    @test (h.id, h.display_name, h.created, h.input_token_limit, h.output_token_limit, h.deprecated, h.retires) ==
          ("claude-haiku-5-5", "Claude Haiku 5.5", DateTime(2026, 10, 7, 18), 1_000_000, 128_000, nothing, nothing)
    @test h.capabilities == page["data"][1]["capabilities"]
    @test h.capabilities["thinking"]["types"]["disabled"] == Dict("supported" => true)
    @test h.raw == page["data"][1]
    @test (r.models[3].id, r.models[3].created) == ("claude-opus-5-5", DateTime(2026, 9, 21, 16, 24))
    @test (r.models[12].display_name, r.models[12].input_token_limit, r.models[12].output_token_limit) ==
          ("Claude Opus 4.5", 200_000, 64_000)
    @test sprint(show, h) == "ProviderModel(\"claude-haiku-5-5\", \"Claude Haiku 5.5\")"
end

# Refuted if: an RFC 3339 instant is read in another zone than UTC, its fraction or offset is
# dropped, or the epoch Anthropic gives for an unknown release date reads as a real one.
@testset "Anthropic's lifecycle dates are the UTC dates of RFC 3339 instants" begin
    entry(id; kw...) = Dict{String,Any}("type" => "model", "id" => id, "display_name" => "Claude Example",
        "created_at" => "2026-01-05T00:00:00Z", "max_input_tokens" => nothing, "max_tokens" => nothing,
        "capabilities" => nothing, "lifecycle" => "active", "deprecated_at" => nothing, "retires_at" => nothing,
        (String(k) => v for (k, v) in kw)...)
    page = Dict{String,Any}("data" => [
            entry("claude-example-1"; lifecycle="deprecated", deprecated_at="2026-09-30T17:00:00.000Z",
                  retires_at="2026-11-30T00:00:00Z"),
            entry("claude-example-2"; retires_at="2026-12-01T01:00:00+02:00"),   # 2026-11-30T23:00Z
            entry("claude-example-3"; created_at="1970-01-01T00:00:00Z"),        # release date unknown
            entry("claude-example-4"; created_at="2019-12-27T18:11:19.117Z")],
        "has_more" => false, "first_id" => "claude-example-1", "last_id" => "claude-example-4")
    r, _ = _ml_list(_ML_ANTHROPIC, _ml_serving(JSON.json(page)))
    @test r isa ModelListSuccess
    a, b, c, d = r.models
    @test (a.created, a.deprecated, a.retires) == (DateTime(2026, 1, 5), Date(2026, 9, 30), Date(2026, 11, 30))
    @test (a.capabilities, a.input_token_limit, a.output_token_limit) == (Dict{String,Any}(), nothing, nothing)
    @test (b.deprecated, b.retires) == (nothing, Date(2026, 11, 30))
    @test c.created === nothing
    @test d.created == DateTime(2019, 12, 27, 18, 11, 19, 117)
    @test sprint(show, a) ==
          "ProviderModel(\"claude-example-1\", \"Claude Example\", deprecated 2026-09-30, retires 2026-11-30)"
end

# Refuted if: a date-time outside RFC 3339 (no offset, a space, a field out of range) or a date
# outside YYYY-MM-DD reads as some date instead of none.
@testset "date readers: RFC 3339 instants and calendar dates, nothing else" begin
    @test @inferred(Nothing, UniLM._rfc3339_utc("2026-11-30T20:00:00-05:00")) == DateTime(2026, 12, 1, 1)
    @test UniLM._rfc3339_utc("2026-10-07t18:00:00.123456z") == DateTime(2026, 10, 7, 18, 0, 0, 123)
    @test UniLM._rfc3339_utc("2026-10-07T18:00:00.1+00:30") == DateTime(2026, 10, 7, 17, 30, 0, 100)
    for bad in ("2026-11-30", "2026-11-30T00:00:00", "2026-11-30 00:00:00Z", "2026-13-01T00:00:00Z",
                "2026-02-30T00:00:00Z", "2026-11-30T24:00:00Z", "2026-11-30T00:00:60Z", "2026-11-30T00:00:00+2:00",
                "2026-11-30T00:00:00.Z", " 2026-11-30T00:00:00Z")
        @test UniLM._rfc3339_utc(bad) === nothing
    end
    @test @inferred(Nothing, UniLM._calendar_date("2027-01-06")) == Date(2027, 1, 6)
    for bad in ("2027-1-6", "2027/01/06", "2027-02-30", "20270106", "2027-01-06T00:00:00Z", "")
        @test UniLM._calendar_date(bad) === nothing
    end
end

@testset "Gemini's listing: ids without models/, token limits, generation methods and thinking" begin
    page = _ml_page("gemini.json")
    r, seen = _ml_list(_ML_GEMINI, _ml_serving(_ml_fixture("gemini.json")))
    @test r isa ModelListSuccess
    @test _ml_targets(seen) == [("GET", "/v1/models")]
    @test r.raw == [page]
    @test [m.id for m in r.models] == [chopprefix(e["name"], "models/") for e in page["models"]]
    @test length(r.models) == 63
    f = only(m for m in r.models if m.id == "gemini-3.8-flash")
    @test (f.display_name, f.created, f.input_token_limit, f.output_token_limit, f.deprecated, f.retires) ==
          ("Gemini 3.8 Flash", nothing, 1_048_576, 65_536, nothing, nothing)
    @test f.capabilities == Dict{String,Any}("thinking" => true,
        "supportedGenerationMethods" => ["generateContent", "countTokens", "createCachedContent", "batchGenerateContent"])
    @test f.raw == only(e for e in page["models"] if e["name"] == "models/gemini-3.8-flash")
    tts = only(m for m in r.models if m.id == "gemini-3.8-flash-tts")
    @test (tts.input_token_limit, tts.output_token_limit) == (8192, 16384)
    @test tts.capabilities ==
          Dict{String,Any}("supportedGenerationMethods" => ["generateContent", "countTokens", "batchGenerateContent"])
    @test sprint(show, f) == "ProviderModel(\"gemini-3.8-flash\", \"Gemini 3.8 Flash\")"
end

# Refuted if: a page after the first is skipped, fetched from the wrong cursor or with its cursor
# unescaped, or the pages' models are merged out of order.
@testset "pagination: every page is fetched, from the cursor the page before names" begin
    data = _ml_page("anthropic.json")["data"]
    p1 = Dict{String,Any}("data" => data[1:7], "has_more" => true, "first_id" => data[1]["id"], "last_id" => data[7]["id"])
    p2 = Dict{String,Any}("data" => data[8:end], "has_more" => false, "first_id" => data[8]["id"],
                          "last_id" => data[end]["id"])
    r, seen = _ml_list(_ML_ANTHROPIC, (n, _) -> _json(200, JSON.json(n == 1 ? p1 : p2)))
    @test r isa ModelListSuccess
    @test _ml_targets(seen) == [("GET", "/v1/models"), ("GET", "/v1/models?after_id=claude-fable-5")]
    @test [m.id for m in r.models] == [e["id"] for e in data]
    @test r.raw == [p1, p2]

    models = _ml_page("gemini.json")["models"]
    q1 = Dict{String,Any}("models" => models[1:50], "nextPageToken" => "Cg9tb2RlbHMvYXFh+/=")
    q2 = Dict{String,Any}("models" => models[51:end])
    r, seen = _ml_list(_ML_GEMINI, (n, _) -> _json(200, JSON.json(n == 1 ? q1 : q2)))
    @test r isa ModelListSuccess
    @test _ml_targets(seen) == [("GET", "/v1/models"), ("GET", "/v1/models?pageToken=Cg9tb2RlbHMvYXFh%2B%2F%3D")]
    @test [m.id for m in r.models] == [chopprefix(e["name"], "models/") for e in models]
    @test r.raw == [q1, q2]
    # An empty Gemini listing leaves `models` out: proto3 JSON omits an empty list.
    r, _ = _ml_list(_ML_GEMINI, _ml_serving("{}"))
    @test r isa ModelListSuccess && isempty(r.models) && r.raw == [Dict{String,Any}()]
end

# Refuted if: a listing that never ends — the same cursor again, or more pages promised with no
# cursor to fetch them from — loops, or returns the pages so far as the whole listing.
@testset "a cursor seen twice, or more pages without a cursor, is a call error" begin
    entry = _ml_page("anthropic.json")["data"][1]
    again = Dict{String,Any}("data" => [entry], "has_more" => true, "first_id" => entry["id"], "last_id" => entry["id"])
    r, seen = _ml_list(_ML_ANTHROPIC, _ml_serving(JSON.json(again)))
    @test r isa ModelListCallError && occursin("\"claude-haiku-5-5\"", r.error) && length(seen) == 2
    open = Dict{String,Any}("data" => [entry], "has_more" => true, "first_id" => nothing, "last_id" => nothing)
    r, seen = _ml_list(_ML_ANTHROPIC, _ml_serving(JSON.json(open)))
    @test r isa ModelListCallError && occursin("last_id", r.error) && length(seen) == 1
    r, seen = _ml_list(_ML_GEMINI, _ml_serving("""{"models":[],"nextPageToken":"t"}"""))
    @test r isa ModelListCallError && occursin("\"t\"", r.error) && length(seen) == 2
end

# Refuted if: an error status reads as a listing or loses its body, status or request id, or a
# failure on a later page leaves the earlier pages as the whole listing.
@testset "an error status on any page is the listing's failure" begin
    body = """{"type":"error","error":{"type":"authentication_error","message":"invalid x-api-key"}}"""
    r, seen = _ml_list(_ML_ANTHROPIC, (_, _) -> _json(401, body; headers=["request-id" => "req_011"]))
    @test r isa ModelListFailure && !issuccess(r)
    @test (r.status, r.response, r.request_id) == (401, body, "req_011") && length(seen) == 1
    data = _ml_page("anthropic.json")["data"]
    p1 = Dict{String,Any}("data" => data[1:2], "has_more" => true, "first_id" => data[1]["id"], "last_id" => data[2]["id"])
    r, seen = _ml_list(_ML_ANTHROPIC, (n, _) -> n == 1 ? _json(200, JSON.json(p1)) : _json(403, "{}"))
    @test r isa ModelListFailure && r.status == 403 && length(seen) == 2
end

# Refuted if: a reply that is not the documented listing decodes into a guess — a missing id, a
# number sent as text, a date in another form, a Gemini name outside models/{model}.
@testset "a reply that is not the documented listing is a call error naming what is wrong" begin
    cases = [(_ML_OPENAI, "[]", "not a JSON object"),
             (_ML_OPENAI, """{"object":"list"}""", "\"data\""),
             (_ML_OPENAI, """{"data":["gpt-6-luna"]}""", "entry"),
             (_ML_OPENAI, """{"data":[{"object":"model","created":1}]}""", "\"id\""),
             (_ML_OPENAI, """{"data":[{"id":""}]}""", "\"id\""),
             (_ML_OPENAI, """{"data":[{"id":"gpt-6-luna","created":"1789406102"}]}""", "\"created\""),
             (_ML_OPENAI, """{"data":[{"id":"gpt-6-luna","shutdown_date":"2027/01/06"}]}""", "\"shutdown_date\""),
             (_ML_ANTHROPIC, """{"data":[]}""", "\"has_more\""),
             (_ML_ANTHROPIC, """{"data":[{"id":"claude-haiku-5-5","max_tokens":"128000"}],"has_more":false}""",
              "\"max_tokens\""),
             (_ML_ANTHROPIC, """{"data":[{"id":"claude-haiku-5-5","retires_at":"2026-11-30"}],"has_more":false}""",
              "\"retires_at\""),
             (_ML_ANTHROPIC, """{"data":[{"id":"claude-haiku-5-5","capabilities":[]}],"has_more":false}""",
              "\"capabilities\""),
             (_ML_GEMINI, """{"models":[{"name":"gemini-3.8-flash"}]}""", "models/{model}"),
             (_ML_GEMINI, """{"models":[{"name":"models/"}]}""", "models/{model}"),
             (_ML_GEMINI, """{"models":[{"name":"models/gemini-3.8-flash","thinking":"yes"}]}""", "\"thinking\""),
             (_ML_GEMINI, """{"models":[{"name":"models/gemini-3.8-flash","supportedGenerationMethods":[1]}]}""",
              "\"supportedGenerationMethods\"")]
    for (service, body, says) in cases
        r, _ = _ml_list(service, _ml_serving(body))
        @test r isa ModelListCallError && occursin(says, r.error)
    end
end

# Refuted if: a provider endpoint lists from another provider's URL or through another wire, does
# not declare its listing, or a missing key surfaces as anything but a call error that names it.
# The Anthropic type and an instance are one endpoint.
@testset "OpenAI, Anthropic and Gemini: their listing URLs and wires, and a missing key named" begin
    anthropic = ("https://api.anthropic.com/v1/models", UniLM._AnthropicModels(), "ANTHROPIC_API_KEY")
    for (service, (url, wire, key)) in (
            OPENAIServiceEndpoint => ("https://api.openai.com/v1/models", UniLM._OpenAIModels(), "OPENAI_API_KEY"),
            ANTHROPICServiceEndpoint => anthropic, ANTHROPICServiceEndpoint() => anthropic,
            GEMINIServiceEndpoint => ("https://generativelanguage.googleapis.com/v1beta/models", UniLM._GeminiModels(),
                                      "GEMINI_API_KEY"))
        @test UniLM._models_url(service) == url
        @test UniLM._model_listing(service) === wire
        @test has_capability(service, :models)
        r = withenv(() -> list_models(; service, config=_ML_CFG), key => nothing)
        @test r isa ModelListCallError && r.cause isa KeyError && occursin(key, r.error)
    end
    # A System One scope on OpenAI lists OpenAI's models.
    r = withenv("OPENAI_API_KEY" => nothing) do
        with_system_one(() -> list_models(; config=_ML_CFG), OPENAIServiceEndpoint)
    end
    @test r isa ModelListCallError && occursin("OPENAI_API_KEY", r.error)
    # An endpoint with no listing of its own, whose System One protocol has none, is refused.
    for service in (GEMINIOpenAIServiceEndpoint, AZUREServiceEndpoint)
        @test_throws ArgumentError list_models(; service)
    end
end

# Refuted if: the listing goes out without the endpoint's headers, or is sent at all once a
# cancel or the deadline has ended the call.
@testset "the endpoint's headers; a cancel and the deadline end the listing before any request" begin
    auth = Ref("")
    r, _ = _ml_list(_ML_OPENAI, (_, req) -> (auth[] = HTTP.header(req, "Authorization");
                                             _json(200, """{"object":"list","data":[]}""")))
    @test r isa ModelListSuccess && isempty(r.models) && auth[] == "Bearer t"
    tok = CancelToken()
    cancel!(tok)
    r, seen = _ml_list(_ML_ANTHROPIC, _ml_serving("{}"); cancel=tok)
    @test r isa ModelListCallError && r.cause isa UniLMCancelled && isempty(seen)
    r, seen = _ml_list(_ML_GEMINI, _ml_serving("{}"); config=_TINY_DEADLINE)
    @test _seam_timeout(r, ModelListCallError) && isempty(seen)
end

# An Anthropic endpoint, the type or an instance, whose listing `_with_scripted` serves: its
# requests carry the headers that endpoint sends, on its listing and on its other requests.
struct _AnthropicListingAt <: UniLM.ServiceEndpoint
    endpoint::Union{Type{ANTHROPICServiceEndpoint},ANTHROPICServiceEndpoint}
end
UniLM._model_listing(::_AnthropicListingAt) = UniLM._AnthropicModels()
UniLM._models_url(::_AnthropicListingAt) = _url_probe_base[] * "/v1/models"
UniLM.auth_header(s::_AnthropicListingAt) = UniLM.auth_header(s.endpoint)
UniLM._models_headers(s::_AnthropicListingAt) = UniLM._models_headers(s.endpoint)

# https://platform.claude.com/docs/en/api/models/list (Headers) names `anthropic-version`,
# `anthropic-workspace-id` and `anthropic-beta`, deprecated on GET /v1/models, and no user-profile
# header. Refuted if: an Anthropic listing sends the endpoint's betas or its user profile, or
# leaves out its key, its API version or its workspace — from the type or from an instance.
@testset "an Anthropic listing sends the key, the version and the workspace, never the user profile or the betas" begin
    withenv("ANTHROPIC_API_KEY" => "sk-ant-local-0000") do
        base = ["x-api-key" => "sk-ant-local-0000", "anthropic-version" => "2026-10-21"]
        every = ANTHROPICServiceEndpoint(betas=["fast-mode-2026-02-01", "compact-2026-09-04"],
                                         user_profile_id="uprof_011CZkZCu8hGbp5mYRQgUmz9",
                                         workspace_id="wrkspc_011CZkZaBF1tNoB5wlCeusgy", inference_geo="us")
        options = ["anthropic-workspace-id" => "wrkspc_011CZkZaBF1tNoB5wlCeusgy"]
        profile = ANTHROPICServiceEndpoint(user_profile_id="uprof_011CZkZCu8hGbp5mYRQgUmz9")
        for (service, want) in ((ANTHROPICServiceEndpoint, base), (ANTHROPICServiceEndpoint(), base),
                                (ANTHROPICServiceEndpoint(betas=["fast-mode-2026-02-01"]), base), (profile, base),
                                (every, [base; options]))
            @test (@inferred UniLM._models_headers(service)) == want
            # On the wire, beside the headers HTTP.jl adds (Host, User-Agent, …): exactly these x- and
            # anthropic- headers, and no Content-Type.
            sent = Ref(Dict{String,String}())
            r, seen = _ml_list(_AnthropicListingAt(service), (_, req) -> (
                sent[] = Dict(lowercase(String(k)) => String(v) for (k, v) in req.headers);
                _json(200, _ml_fixture("anthropic.json"))))
            @test r isa ModelListSuccess && length(r.models) == 13 && length(seen) == 1
            @test filter(((k, _),) -> startswith(k, "x-") || startswith(k, "anthropic-"), sent[]) == Dict(want)
            @test !haskey(sent[], "content-type")
        end
        # The endpoint's other requests keep its betas and its user profile.
        @test Dict(UniLM.auth_header(every))["anthropic-beta"] == "fast-mode-2026-02-01,compact-2026-09-04"
        @test Dict(UniLM.auth_header(every))["anthropic-user-profile-id"] == "uprof_011CZkZCu8hGbp5mYRQgUmz9"
    end
end

# ─── The drift checks (provider_drift_checks.jl) ─────────────────────────────

_ml_models(name::AbstractString, wire) = UniLM._page_models(wire, _ml_page(name * ".json"))
_ml_model(id::AbstractString, capabilities::Dict{String,Any}=Dict{String,Any}()) =
    ProviderModel(id, nothing, nothing, nothing, nothing, capabilities, nothing, nothing, Dict{String,Any}())
# `m` with other capabilities or another retirement date.
_ml_with(m::ProviderModel; capabilities=m.capabilities, retires=m.retires) =
    ProviderModel(m.id, m.display_name, m.created, m.input_token_limit, m.output_token_limit, capabilities,
                  m.deprecated, retires, m.raw)
# The cost code warns once per process about each id it has no price row for, so whether a
# warning shows depends on which tests ran first; these tests assert the finding instead.
_ml_quiet(f::Function) = with_logger(f, NullLogger())

# Refuted if: the checks report a gap on the listings of 2026-10-22, which UniLM's tables cover,
# or read a family other than the one UniLM documents.
@testset "drift checks: the 2026-10-22 listings leave nothing to report" begin
    o = _openai_drift(_ml_models("openai", UniLM._OpenAIModels()))
    @test o.family == ["gpt-5.6-luna", "gpt-5.4-mini", "gpt-5.4-mini-2026-03-17", "gpt-5.6-terra", "gpt-5.6-sol",
                       "gpt-6.1-sol", "gpt-6-sol", "gpt-6-luna", "gpt-6-astra"]
    @test all(isempty, (o.unpriced, o.no_terms, o.unruled))
    an = _ml_models("anthropic", UniLM._AnthropicModels())
    a = _anthropic_drift(an)
    @test a.family == [m.id for m in an]
    @test all(isempty, (a.unreadable, a.unpriced, a.unruled, a.disagreements))
    g = _gemini_drift(_ml_models("gemini", UniLM._GeminiModels()))
    @test g.family == g.thinking == ["gemini-3.5-flash", "gemini-3.5-flash-lite", "gemini-3.6-flash",
                                     "gemini-3.7-flash", "gemini-3.8-flash", "gemini-4-argon"]
    @test all(isempty, (g.unpriced, g.unruled))
end

# Refuted if: a broken fact goes unreported, or a sound one is reported — each check gets one.
@testset "drift checks: each reports the broken fact it guards" begin
    oa = _ml_models("openai", UniLM._OpenAIModels())
    # A price row removed.
    row = pop!(DEFAULT_PRICING, "gpt-6-luna")
    try
        @test _ml_quiet(() -> _openai_drift(oa).unpriced) == ["gpt-6-luna"]
    finally
        DEFAULT_PRICING["gpt-6-luna"] = row
    end
    @test isempty(_openai_drift(oa).unpriced)
    # A newer GPT-6 model with no rows.
    d = _ml_quiet(() -> _openai_drift([oa; _ml_model("gpt-6.2-sol")]))
    @test (d.unpriced, d.no_terms, d.unruled) == (["gpt-6.2-sol"], ["gpt-6.2-sol"], ["gpt-6.2-sol"])

    an = _ml_models("anthropic", UniLM._AnthropicModels())
    # A listed capability flipped against its row.
    caps = deepcopy(an[1].capabilities)
    caps["thinking"]["types"]["disabled"]["supported"] = false
    caps["effort"]["xhigh"]["supported"] = false
    @test _anthropic_drift([_ml_with(an[1]; capabilities=caps); an[2:end]]).disagreements ==
          ["claude-haiku-5-5" => "thinking :on takes disabled, listed false",
           "claude-haiku-5-5" => "efforts [\"low\", \"medium\", \"high\", \"xhigh\", \"max\"], listed [\"high\", \"low\", \"max\", \"medium\"]"]
    # An id whose generation cannot be read and that no rule row names.
    d = _anthropic_drift([an; _ml_model("claude-nova")])
    @test d.unreadable == ["claude-nova"] && "claude-nova" ∉ d.family
    # An id a rule row names is in the family whatever its name says, its row checked.
    supported(b) = Dict{String,Any}("supported" => b)
    preview = Dict{String,Any}(
        "thinking" => Dict{String,Any}("supported" => true, "types" => Dict{String,Any}("adaptive" => supported(true),
            "disabled" => supported(false), "between_tools" => supported(false), "enabled" => supported(false))),
        "effort" => Dict{String,Any}("supported" => true, "low" => supported(true), "medium" => supported(true),
            "high" => supported(true), "xhigh" => supported(false), "max" => supported(true)))
    d = _ml_quiet(() -> _anthropic_drift([an; _ml_model("claude-mythos-preview", preview)]))
    @test "claude-mythos-preview" in d.family && all(isempty, (d.unreadable, d.unruled, d.disagreements))

    ge = _ml_models("gemini", UniLM._GeminiModels())
    serves = Dict{String,Any}("supportedGenerationMethods" => ["generateContent"])
    thinks = merge(serves, Dict{String,Any}("thinking" => true))
    d = _ml_quiet(() -> _gemini_drift([ge; _ml_model("gemini-3.9-flash", serves); _ml_model("gemini-3.9-pro", thinks);
                                       _ml_model("gemini-3.9-flash-tts", serves); _ml_model("gemini-3.9-flash-image", thinks)]))
    # A text model that does not say it thinks still needs a price row; one that thinks, its levels too.
    @test d.unpriced == ["gemini-3.9-flash", "gemini-3.9-pro"] && d.unruled == ["gemini-3.9-pro"]
    @test "gemini-3.9-flash" in d.family && "gemini-3.9-flash" ∉ d.thinking
    @test !any(in(d.family), ("gemini-3.9-flash-tts", "gemini-3.9-flash-image"))

    # The models UniLM names, by provider prefix: a named model the listing lacks, one whose
    # retirement date has come, and ids of no checked provider.
    n = _drift_named_models(oa, :openai; named=["gpt-6-sol", "gemma4:e4b", "jev-latest", "claude-haiku-5-5"])
    @test n.named == ["gpt-6-sol"] && isempty(n.unlisted) && isempty(n.retired)
    @test _drift_named_models(filter(m -> m.id != "gpt-6-sol", oa), :openai; named=["gpt-6-sol"]).unlisted == ["gpt-6-sol"]
    due = [_ml_with(m; retires=m.id == "gpt-6-sol" ? today() : m.retires) for m in oa]
    @test _drift_named_models(due, :openai; named=["gpt-6-sol"]).retired == ["gpt-6-sol" => today()]
    @test _drift_provider.(["gpt-6-sol", "o3", "omni-moderation-latest", "claude-haiku-5-5", "gemini-3.8-flash",
                            "gemma-4-31b-it", "gemma4:e4b", "jev-latest"]) ==
          [:openai, :openai, :openai, :anthropic, :gemini, :gemini, nothing, nothing]
    @test !isempty(_drift_suite_ids()) && all(id -> _drift_provider(id) !== nothing, _drift_suite_ids())
    @test UniLM.default_model(OPENAIServiceEndpoint) in _drift_defaults()
end
