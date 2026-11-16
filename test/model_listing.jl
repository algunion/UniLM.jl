# Provider model listings (`list_models` on OpenAI, Anthropic and Gemini): each listing decodes
# into ProviderModels — ids, names, token limits, capabilities, lifecycle dates — with every
# page fetched, against a local server replaying the listings each provider returned for the
# repository's keys on 2026-10-22 (test/fixtures/models/); and the provider drift checks of the
# live drift suite, on those listings and on broken facts, and the models UniLM uses by default on
# the listings of 2026-11-16 (the *-2026-11-16.json fixtures). Offline.

using Dates
using Logging: with_logger, NullLogger
isdefined(@__MODULE__, :ProviderDriftChecks) || include("provider_drift_checks.jl")
using .ProviderDriftChecks

_ml_fixture(name::AbstractString) = read(joinpath(@__DIR__, "fixtures", "models", name), String)
_ml_page(name::AbstractString) = JSON.parse(_ml_fixture(name); dicttype=Dict{String,Any})
const _ML_CFG = RequestConfig(max_attempts=1, total_deadline=30.0)

# `ListingProbe{W}` (platform_seam_fixtures.jl) lists through provider wire `W`, served by `_with_scripted`.

# An endpoint type of the caller's own that declares no capabilities: the documented way to reach an
# OpenAI-compatible server UniLM does not ship.
struct _MLUndeclared <: UniLM.ServiceEndpoint end

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
    @test _ml_targets(seen) == [("GET", "/v1/models?limit=1000")]
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
    @test _ml_targets(seen) == [("GET", "/v1/models?pageSize=1000")]
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

# Every page asks for the largest page the API serves: Anthropic's `limit` "Ranges from `1` to `1000`"
# (1001 was HTTP 400 on 2026-11-16), and Gemini returns "at most 1000 models per page", with "all other
# parameters" of a paged call matching the call that gave the page token.
# Refuted if: a page after the first is skipped, fetched from the wrong cursor or with its cursor
# unescaped, a page asks for another page size, or the pages' models are merged out of order.
@testset "pagination: every page is fetched, from the cursor the page before names" begin
    data = _ml_page("anthropic.json")["data"]
    p1 = Dict{String,Any}("data" => data[1:7], "has_more" => true, "first_id" => data[1]["id"], "last_id" => data[7]["id"])
    p2 = Dict{String,Any}("data" => data[8:end], "has_more" => false, "first_id" => data[8]["id"],
                          "last_id" => data[end]["id"])
    r, seen = _ml_list(_ML_ANTHROPIC, (n, _) -> _json(200, JSON.json(n == 1 ? p1 : p2)))
    @test r isa ModelListSuccess
    @test _ml_targets(seen) == [("GET", "/v1/models?limit=1000"), ("GET", "/v1/models?limit=1000&after_id=claude-fable-5")]
    @test [m.id for m in r.models] == [e["id"] for e in data]
    @test r.raw == [p1, p2]

    models = _ml_page("gemini.json")["models"]
    q1 = Dict{String,Any}("models" => models[1:50], "nextPageToken" => "Cg9tb2RlbHMvYXFh+/=")
    q2 = Dict{String,Any}("models" => models[51:end])
    r, seen = _ml_list(_ML_GEMINI, (n, _) -> _json(200, JSON.json(n == 1 ? q1 : q2)))
    @test r isa ModelListSuccess
    @test _ml_targets(seen) == [("GET", "/v1/models?pageSize=1000"),
                                ("GET", "/v1/models?pageSize=1000&pageToken=Cg9tb2RlbHMvYXFh%2B%2F%3D")]
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
    # An endpoint with no listing of its own, whose System One protocol has none, is refused by its own
    # name, in words that name no other provider's listing.
    for (service, name) in (GEMINIOpenAIServiceEndpoint => "GEMINIOpenAIServiceEndpoint",
                            AZUREServiceEndpoint => "AZUREServiceEndpoint", DeepSeekEndpoint() => "DeepSeekEndpoint",
                            GenericOpenAIEndpoint("http://127.0.0.1:9/v1", "") => "GenericOpenAIEndpoint",
                            MistralEndpoint() => "GenericOpenAIEndpoint", _MLUndeclared => "_MLUndeclared")
        e = try list_models(; service); nothing catch x; x end
        @test e isa ArgumentError && startswith(e.msg, "Model listing is not supported by $name") &&
              !occursin("TypeSafe", e.msg)
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
    endpoint::UniLM._AnthropicEndpoint
end
UniLM._model_listing(::_AnthropicListingAt) = UniLM._AnthropicModels()
UniLM._models_url(::_AnthropicListingAt) = _url_probe_base[] * "/v1/models"
UniLM.auth_header(s::_AnthropicListingAt) = UniLM.auth_header(s.endpoint)
UniLM._models_headers(s::_AnthropicListingAt) = UniLM._models_headers(s.endpoint)

# https://platform.claude.com/docs/en/api/models/list (Headers) names `anthropic-version` and
# `anthropic-workspace-id` for GET /v1/models, and no `anthropic-beta` or user-profile header.
# Refuted if: an Anthropic listing sends the endpoint's betas or its user profile, or
# leaves out its key, its API version or its workspace — from the type or from an instance.
@testset "an Anthropic listing sends the key, the version and the workspace, never the user profile or the betas" begin
    withenv("ANTHROPIC_API_KEY" => "sk-ant-local-0000") do
        base = ["Authorization" => "Bearer sk-ant-local-0000", "anthropic-version" => "2026-10-21"]
        every = ANTHROPICServiceEndpoint(betas=["fast-mode-2026-02-01", "compact-2026-09-04"],
                                         user_profile_id="uprof_011CZkZCu8hGbp5mYRQgUmz9",
                                         workspace_id="wrkspc_011CZkZaBF1tNoB5wlCeusgy", inference_geo="us")
        options = ["anthropic-workspace-id" => "wrkspc_011CZkZaBF1tNoB5wlCeusgy"]
        profile = ANTHROPICServiceEndpoint(user_profile_id="uprof_011CZkZCu8hGbp5mYRQgUmz9")
        for (service, want) in ((ANTHROPICServiceEndpoint, base), (ANTHROPICServiceEndpoint(), base),
                                (ANTHROPICServiceEndpoint(betas=["fast-mode-2026-02-01"]), base), (profile, base),
                                (every, [base; options]))
            @test (@inferred UniLM._models_headers(service)) == want
            # On the wire, beside the headers HTTP.jl adds (Host, User-Agent, …): exactly these
            # authorization, x- and anthropic- headers, and no Content-Type.
            sent = Ref(Dict{String,String}())
            r, seen = _ml_list(_AnthropicListingAt(service), (_, req) -> (
                sent[] = Dict(lowercase(String(k)) => String(v) for (k, v) in req.headers);
                _json(200, _ml_fixture("anthropic.json"))))
            @test r isa ModelListSuccess && length(r.models) == 13 && length(seen) == 1
            @test filter(((k, _),) -> k == "authorization" || startswith(k, "x-") || startswith(k, "anthropic-"),
                         sent[]) == Dict(lowercase(k) => v for (k, v) in want)
            @test !haskey(sent[], "content-type")
        end
        # The endpoint's other requests keep its betas and its user profile.
        @test Dict(UniLM.auth_header(every))["anthropic-beta"] == "fast-mode-2026-02-01,compact-2026-09-04"
        @test Dict(UniLM.auth_header(every))["anthropic-user-profile-id"] == "uprof_011CZkZCu8hGbp5mYRQgUmz9"
    end
end

# ─── One model by id (retrieve_model), and a fine-tuned model's deletion ─────

# The entries the APIs returned for one model on 2026-11-16: GET /v1/models/{model} on OpenAI and
# Anthropic (an alias resolves to the dated id), GET /v1beta/models/{model} on Gemini.
const _ML_ONE = Dict(
    UniLM._OpenAIModels => """{"id":"gpt-6-luna","object":"model","created":1789406102,"owned_by":"system","shutdown_date":null}""",
    UniLM._AnthropicModels => """{"type":"model","id":"claude-opus-4-5-20251101","display_name":"Claude Opus 4.5","created_at":"2025-11-24T00:00:00Z","max_input_tokens":200000,"max_tokens":64000,"capabilities":{"batch":{"supported":true}}}""",
    UniLM._GeminiModels => """{"name":"models/gemini-3.8-flash","version":"3.0","displayName":"Gemini 3.8 Flash","inputTokenLimit":1048576,"outputTokenLimit":65536,"supportedGenerationMethods":["generateContent","countTokens"],"thinking":true}""")
_ml_get(service, id, respond::Function; config=_ML_CFG, cancel=nothing) =
    _with_scripted(() -> retrieve_model(id; service, config, cancel), respond)

# Refuted if: a lookup goes to another URL or decodes otherwise than the same model in a listing, an id
# is sent unescaped, or the model the provider resolved an alias to is replaced by the id asked for.
@testset "retrieve_model: one model, decoded as its listing entry, at the listing's URL and the id" begin
    for (wire, body) in _ML_ONE
        service = ListingProbe{wire}
        r, seen = _ml_get(service, "gpt-6-luna", _ml_serving(body))
        @test r isa ModelSuccess && issuccess(r)
        @test _ml_targets(seen) == [("GET", "/v1/models/gpt-6-luna")]
        entry = JSON.parse(body; dicttype=Dict{String,Any})
        @test r.model.raw == entry
        want = UniLM._provider_model(wire(), entry)
        @test [getfield(r.model, f) for f in fieldnames(ProviderModel)] == [getfield(want, f) for f in fieldnames(ProviderModel)]
    end
    r, _ = _ml_get(_ML_ANTHROPIC, "claude-opus-4-5", _ml_serving(_ML_ONE[UniLM._AnthropicModels]))
    @test (r.model.id, r.model.display_name, r.model.input_token_limit) == ("claude-opus-4-5-20251101", "Claude Opus 4.5", 200000)
    r, _ = _ml_get(_ML_GEMINI, "gemini-3.8-flash", _ml_serving(_ML_ONE[UniLM._GeminiModels]))
    @test (r.model.id, r.model.capabilities["thinking"]) == ("gemini-3.8-flash", true)
    @test sprint(show, r) == "ModelSuccess(ProviderModel(\"gemini-3.8-flash\", \"Gemini 3.8 Flash\"))"
    _, seen = _ml_get(_ML_OPENAI, _HOSTILE_ID, _ml_serving(_ML_ONE[UniLM._OpenAIModels]))
    @test _ml_targets(seen) == [("GET", "/v1/models/" * _HOSTILE_ENC)]
    for (service, url) in ((OPENAIServiceEndpoint, "https://api.openai.com/v1/models/gpt-6-luna"),
                           (ANTHROPICServiceEndpoint(), "https://api.anthropic.com/v1/models/gpt-6-luna"),
                           (GEMINIServiceEndpoint, "https://generativelanguage.googleapis.com/v1beta/models/gpt-6-luna"))
        @test UniLM._model_url(service, "gpt-6-luna") == url
    end
end

# Refuted if: a 404 or another error status reads as a model or loses its body, status or request id; a
# reply that is not the documented model reads as one; a missing key, a cancel or the deadline sends anything.
@testset "retrieve_model: failures, malformed models, a missing key, a cancel and the deadline" begin
    body = """{"type":"error","error":{"type":"not_found_error","message":"model: claude-3-opus-20240229"}}"""
    r, seen = _ml_get(_ML_ANTHROPIC, "claude-3-opus-20240229", (_, _) -> _json(404, body; headers=["request-id" => "req_011"]))
    @test r isa ModelListFailure && !issuccess(r) && (r.status, r.response, r.request_id) == (404, body, "req_011")
    @test length(seen) == 1
    for (service, reply, says) in ((_ML_OPENAI, "[]", "not a JSON object"), (_ML_OPENAI, """{"object":"model"}""", "\"id\""),
                                   (_ML_ANTHROPIC, """{"id":"claude-haiku-5-5","max_tokens":"1"}""", "\"max_tokens\""),
                                   (_ML_GEMINI, """{"name":"gemini-3.8-flash"}""", "models/{model}"), (_ML_GEMINI, "nope", ""))
        r, seen = _ml_get(service, "m", _ml_serving(reply))
        @test r isa ModelListCallError && occursin(says, r.error) && length(seen) == 1
    end
    tok = CancelToken(); cancel!(tok)
    r, seen = _ml_get(_ML_GEMINI, "m", _ml_serving("{}"); cancel=tok)
    @test r isa ModelListCallError && r.cause isa UniLMCancelled && isempty(seen)
    r, seen = _ml_get(_ML_OPENAI, "m", _ml_serving("{}"); config=_TINY_DEADLINE)
    @test _seam_timeout(r, ModelListCallError) && isempty(seen)
    for (service, key) in ((OPENAIServiceEndpoint, "OPENAI_API_KEY"), (ANTHROPICServiceEndpoint, "ANTHROPIC_API_KEY"),
                           (GEMINIServiceEndpoint, "GEMINI_API_KEY"))
        r = withenv(() -> retrieve_model("m"; service, config=_ML_CFG), key => nothing)
        @test r isa ModelListCallError && r.cause isa KeyError && occursin(key, r.error)
    end
end

# Refuted if: an empty id, an endpoint with no lookup or one without the capability is sent anything, or
# is refused in words that name another provider's lookup.
@testset "retrieve_model: refused before any request where there is no lookup" begin
    e = try retrieve_model(""; service=_ML_OPENAI); nothing catch x; x end
    @test e isa ArgumentError && occursin("retrieve_model", e.msg)
    for (service, says) in ((TYPESAFEServiceEndpoint, "Model lookup is not supported by TYPESAFEServiceEndpoint"),
                            (OllamaEndpoint(), "Model lookup is not supported by OllamaEndpoint"),
                            (AZUREServiceEndpoint, "Model lookup is not supported by AZUREServiceEndpoint"),
                            (GEMINIOpenAIServiceEndpoint, "Model lookup is not supported by GEMINIOpenAIServiceEndpoint"),
                            (DeepSeekEndpoint(api_key="k"), "Model lookup is not supported by DeepSeekEndpoint"),
                            (_MLUndeclared, "Model lookup is not supported by _MLUndeclared"))
        e = try retrieve_model("m"; service); nothing catch x; x end
        @test e isa ArgumentError && startswith(e.msg, says)
    end
    e = try retrieve_model("m"; service=OllamaEndpoint()); nothing catch x; x end
    @test occursin("model_info", e.msg)
end

# DeleteModelResponse (OpenAI's OpenAPI spec) requires id, object and deleted; OpenAI answered the delete
# of a made-up fine-tuned model with HTTP 404 model_not_found on 2026-11-16.
# Refuted if: the delete goes to another URL or method, a reply without its id or its confirmation reads as
# a deletion, the requested id stands in for the reply's, or an endpoint without fine-tuning is sent anything.
@testset "a fine-tuned model's deletion on OpenAI: DELETE /v1/models/{model}, confirmed by the reply" begin
    del(id, respond::Function; config=_ML_CFG, cancel=nothing, service=_ML_OPENAI) =
        _with_scripted(() -> UniLM._delete_model(service, id, config, cancel), respond)
    id = "ft:gpt-4o-mini:org:suffix:abc123"
    r, seen = del(id, _ml_serving("""{"id":"$id","object":"model","deleted":true}"""))
    @test r isa ModelDeleteSuccess && issuccess(r) && (r.id, r.deleted) == (id, true)
    @test _ml_targets(seen) == [("DELETE", "/v1/models/ft%3Agpt-4o-mini%3Aorg%3Asuffix%3Aabc123")]
    for reply in ("""{"object":"model","deleted":true}""", """{"id":"","object":"model","deleted":true}""",
                  """{"id":"$id","object":"model"}""", """{"id":"$id","object":"model","deleted":false}""", "[]")
        r, seen = del(id, _ml_serving(reply))
        @test r isa ModelListCallError && length(seen) == 1
    end
    body = """{"error":{"message":"The model '$id' does not exist","code":"model_not_found"}}"""
    r, _ = del(id, (_, _) -> _json(404, body; headers=["x-request-id" => "req_1"]))
    @test r isa ModelListFailure && (r.status, r.response, r.request_id) == (404, body, "req_1")
    r, seen = del(id, _ml_serving("{}"); config=_TINY_DEADLINE)
    @test _seam_timeout(r, ModelListCallError) && isempty(seen)
    r = withenv(() -> UniLM._delete_model(OPENAIServiceEndpoint, id, _ML_CFG, nothing), "OPENAI_API_KEY" => nothing)
    @test r isa ModelListCallError && r.cause isa KeyError
    for (service, says) in ((ANTHROPICServiceEndpoint, "Model deletion is not supported by ANTHROPICServiceEndpoint"),
                            (GEMINIServiceEndpoint, "Model deletion is not supported by GEMINIServiceEndpoint"),
                            (SeamProbe, "Model deletion is not supported by SeamProbe"),
                            (_MLUndeclared, "Model deletion is not supported by _MLUndeclared"))
        e = try UniLM._delete_model(service, id, nothing, nothing); nothing catch x; x end
        @test e isa ArgumentError && startswith(e.msg, says)
    end
    e = try UniLM._delete_model(_ML_OPENAI, "", nothing, nothing); nothing catch x; x end
    @test e isa ArgumentError
    # The verb routes an OpenAI endpoint here, and refuses one without fine-tuning by name.
    r, seen = _with_scripted(() -> delete_model(id; service=_ML_OPENAI, config=_ML_CFG),
                             _ml_serving("""{"id":"$id","object":"model","deleted":true}"""))
    @test r isa ModelDeleteSuccess && _ml_targets(seen) == [("DELETE", "/v1/models/ft%3Agpt-4o-mini%3Aorg%3Asuffix%3Aabc123")]
    e = try delete_model(id; service=ANTHROPICServiceEndpoint); nothing catch x; x end
    @test e isa ArgumentError && startswith(e.msg, "Model deletion is not supported by ANTHROPICServiceEndpoint")
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
    # Gemini 3.5 and later, and the older text models UniLM keeps thinking levels for — not the
    # -latest aliases, nor gemini-3.1-flash-lite-image, an image model.
    @test g.family == g.thinking == ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.5-flash-lite",
                                     "gemini-3-flash-preview", "gemini-3.1-pro-preview", "gemini-3.1-flash-lite",
                                     "gemini-3.5-flash", "gemini-3.5-flash-lite", "gemini-3.6-flash", "gemini-3.7-flash",
                                     "gemini-3.8-flash", "gemini-4-argon", "gemini-robotics-er-2-preview"]
    @test g.unpriced == String[]
    @test isempty(g.unruled)
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
    # Claude Mythos Preview has no published price, and is exempt from the price check by its
    # exact id: a dated snapshot of it, should one be listed, is not.
    d = _anthropic_drift([an; _ml_model("claude-mythos-preview", preview)])
    @test "claude-mythos-preview" in d.family && all(isempty, (d.unreadable, d.unpriced, d.unruled, d.disagreements))
    d = _ml_quiet(() -> _anthropic_drift([an; _ml_model("claude-mythos-preview-20260609", preview)]))
    @test d.unpriced == ["claude-mythos-preview-20260609"] && isempty(d.unruled)

    ge = _ml_models("gemini", UniLM._GeminiModels())
    serves = Dict{String,Any}("supportedGenerationMethods" => ["generateContent"])
    thinks = merge(serves, Dict{String,Any}("thinking" => true))
    d = _ml_quiet(() -> _gemini_drift([ge; _ml_model("gemini-3.9-flash", serves); _ml_model("gemini-3.9-pro", thinks);
                                       _ml_model("gemini-3.9-flash-tts", serves); _ml_model("gemini-3.9-flash-image", thinks)]))
    # A text model that does not say it thinks still needs a price row; one that thinks, its levels too.
    @test d.unpriced == ["gemini-3.9-flash", "gemini-3.9-pro"] && d.unruled == ["gemini-3.9-pro"]
    @test "gemini-3.9-flash" in d.family && "gemini-3.9-flash" ∉ d.thinking
    @test !any(in(d.family), ("gemini-3.9-flash-tts", "gemini-3.9-flash-image"))
    # An older text model UniLM keeps thinking levels for needs a price row too.
    row = pop!(DEFAULT_PRICING, "gemini-2.5-flash")
    try
        @test _ml_quiet(() -> _gemini_drift(ge).unpriced) == ["gemini-2.5-flash"]
    finally
        DEFAULT_PRICING["gemini-2.5-flash"] = row
    end
    # A -latest alias, priced at the model that served it, and an image model need none.
    @test !any(in(_gemini_drift(ge).family), ("gemini-flash-latest", "gemini-pro-latest", "gemini-3.1-flash-lite-image"))

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
    @test !isempty(_drift_suite_ids())
    # Every default carries a provider's prefix, so the named-model check skips none of them.
    @test all(!isnothing ∘ _drift_provider, _drift_defaults())
end

# Refuted if: an alias Anthropic serves for a dated snapshot reads as unlisted, or its snapshot's
# retirement goes unseen, or a snapshot id the listing does not name reads as listed. "For models
# before the 4.6 generation, the alias is a convenience pointer that resolves to the dated ID"
# (https://platform.claude.com/docs/en/models/overview).
@testset "drift checks: a named alias is listed under its dated snapshot, a snapshot only as itself" begin
    an = _ml_models("anthropic", UniLM._AnthropicModels())
    named = ["claude-haiku-4-5", "claude-opus-4-5-20251101", "claude-haiku-5-5-20261007", "claude-sonnet-4-5"]
    n = _drift_named_models(an, :anthropic; named)
    @test n.named == named && n.unlisted == ["claude-haiku-5-5-20261007", "claude-sonnet-4-5"] && isempty(n.retired)
    due = [_ml_with(m; retires=m.id == "claude-haiku-4-5-20251001" ? today() : m.retires) for m in an]
    @test _drift_named_models(due, :anthropic; named).retired == ["claude-haiku-4-5" => today()]
end

# Refuted if: a chat model of a new generation or of a new name passes unseen on OpenAI or Gemini,
# or one of a documented family, an older generation or a known other kind is reported. The live
# suite logs the list: a listing may add models before UniLM has tables for them.
@testset "drift checks: OpenAI and Gemini models of no family, older generation or known kind are unclassified" begin
    oa = _ml_models("openai", UniLM._OpenAIModels())
    @test isempty(_openai_drift(oa).unclassified)
    new = ["gpt-7-sol", "o5-mini", "chatgpt-6-latest", "gpt-oss-120b"]
    known = ["gpt-4.5-preview", "o3-deep-research", "gpt-image-3", "gpt-realtime-3", "gpt-6.2-sol"]
    @test _ml_quiet(() -> _openai_drift([oa; _ml_model.([new; known])]).unclassified) == new
    ge = _ml_models("gemini", UniLM._GeminiModels())
    @test isempty(_gemini_drift(ge).unclassified)
    serves = Dict{String,Any}("supportedGenerationMethods" => ["generateContent"])
    new = ["gemini-ultra", "gemini-robotics-er-3-preview"]
    known = ["gemini-3.4-flash", "gemini-nano-banana-3", "gemini-omni-2-flash", "gemini-ultra-latest", "gemini-5-pro"]
    embeds = _ml_model("gemini-embedding-3", Dict{String,Any}("supportedGenerationMethods" => ["embedContent"]))
    d = _ml_quiet(() -> _gemini_drift([ge; [_ml_model(id, serves) for id in [new; known]]; embeds]))
    @test d.unclassified == new
end

# Refuted if: a model id in a docs page's Julia code — a ```julia, ```@example or ```@repl block, at
# any indentation — goes unchecked, or one in prose (migration notes name retired models on purpose)
# or in another block is checked, or a prefix that a `startswith` test names is taken for a model.
@testset "drift checks: the models the docs' Julia code names are checked, not those of their prose" begin
    mktempdir() do dir
        mkpath(joinpath(dir, "guide"))
        write(joinpath(dir, "guide", "page.md"), """
            Prose names "gpt-4o", retired, on purpose.

            ```julia
            chat = Chat(model="gpt-6-sol")
            ids = [m.id for m in ms if startswith(m.id, "gpt-6") || startswith(m.id, "gemini-3.8")]
            ```

            ```@example page
            Chat(service=ANTHROPICServiceEndpoint, model="claude-haiku-5-5")
            ```

            ```@repl page
            Chat(service=GEMINIServiceEndpoint, model="gemini-3.8-flash"); Chat(model="gemma4:e4b")
            ```

            ```@setup page
            Chat(model="gpt-5.5")
            ```

            ```bash
            curl -d '{"model": "gpt-5.4"}'
            ```

            ```julia-repl
            julia> Chat(model="gpt-5.2")
            ```

            !!! note "An admonition indents its blocks"
                ```julia
                Chat(model="gpt-6-luna")
                ```

                ```@setup page
                Chat(model="gpt-5.4-mini")
                ```
            """)
        write(joinpath(dir, "notes.txt"), "```julia\nChat(model=\"gpt-5.6-sol\")\n```\n")
        @test _drift_docs_ids(dir) == ["gpt-6-sol", "claude-haiku-5-5", "gemini-3.8-flash", "gpt-6-luna"]
    end
    docs = _drift_docs_ids()
    @test !isempty(docs)
    oa = _ml_models("openai", UniLM._OpenAIModels())
    @test issubset(filter(id -> _drift_provider(id) === :openai, docs), _drift_named_models(oa, :openai).named)
end

# Refuted if: a model id that only a Julia comment names — a `#` line comment or a `#= … =#` block
# comment, nested or not, in a docs code block or a live suite — is checked (a page prints a
# listing's output as comments: models the reader never runs), or an id the code names is lost
# because a string or a character literal on its line holds `#` or a quote.
@testset "drift checks: an id only a Julia comment names is not checked; a # inside a string is code" begin
    ids(code) = mktempdir() do dir
        write(joinpath(dir, "page.md"), "```julia\n$code\n```\n")
        _drift_docs_ids(dir)
    end
    @test isempty(ids("""ids = [m.id for m in list_models(ANTHROPICServiceEndpoint)]
                         #  ("claude-opus-4-5-20251101", "claude-haiku-4-5-20251001")"""))
    @test isempty(ids("""list_models()
                         #= ["gpt-5.4",
                             "gemini-2.5-flash"] =#"""))
    @test isempty(ids("""x = 1 #= outer #= "gpt-5.4" =# still a comment: "gpt-5.2" =# + 2"""))
    @test ids("""url = "https://x.org/a#b"; c = '#'; q = '"'; Chat(model="gpt-6-sol") # was "gpt-5.4" """) == ["gpt-6-sol"]
    @test ids("s = \"\"\"\n# kept: \"gpt-6-luna\"\n\"\"\"; Chat(model=\"claude-haiku-5-5\")") == ["gpt-6-luna", "claude-haiku-5-5"]
    # The live suites' sources are read by the same rule.
    @test ProviderDriftChecks._drift_quoted_ids("Chat(model=\"gpt-6-sol\")  # not \"gpt-5.4\"\n#= \"gpt-5.2\" =#") == ["gpt-6-sol"]
end

# Refuted if: a model UniLM uses when a call names none — translate's Realtime model and both
# Decisions models among them — is left out of the drift check's defaults, or is unlisted, or past
# its retirement date, on the listings of 2026-11-16.
@testset "drift checks: every default model is listed on 2026-11-16, before its retirement" begin
    for (provider, wire) in (:openai => UniLM._OpenAIModels(), :anthropic => UniLM._AnthropicModels(),
                             :gemini => UniLM._GeminiModels())
        n = _drift_named_models(_ml_models("$(provider)-2026-11-16", wire), provider; named=_drift_defaults())
        @test !isempty(n.named)
        @test (provider, n.unlisted, n.retired) == (provider, [], [])
    end
    @test UniLM._TRANSLATE_DEFAULT_MODEL in _drift_defaults()
    @test UniLM._DECISIONS_MODEL in _drift_defaults()
    @test UniLM._GEMINI_DECISIONS_MODEL in _drift_defaults()
end
