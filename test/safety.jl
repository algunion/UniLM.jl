# test/safety.jl — OpenAI's safety cases and alerts, and its content provenance checks. Local servers only.

# An endpoint at the scripted address that declares the safety and provenance APIs.
struct _SafetyProbe <: UniLM.ServiceEndpoint end
UniLM._resolve_base_url(::Type{_SafetyProbe}) = _url_probe_base[]
UniLM.auth_header(::Type{_SafetyProbe}) = ["Authorization" => "Bearer t", "Content-Type" => "application/json"]
UniLM.provider_capabilities(::Type{_SafetyProbe}) = Set([:safety, :content_provenance])

@testset "Safety — retrieve a safety case and a project safety alert" begin
    s = _SafetyProbe
    case = Dict("id" => "C-abc123", "object" => "safety.case", "created_at" => 1787659200,
                "entity_identifier" => "safety-id-123", "reason" => "cyber_abuse", "notice" => Dict("type" => "warning"))
    alert = Dict("id" => "alert_0123456789abcdef0123456789abcdef", "object" => "safety.alert", "created_at" => 1787659200,
                 "request_id" => "req_123", "response_id" => "resp_123", "model" => "gpt-6-astra", "request_paused" => true,
                 "error_type" => "potentially_unintended_data_access", "reason" => "Potentially unintended data access.")
    rs, seen = _with_scripted((n, _) -> _json(200, JSON.json(n == 1 ? case : alert))) do
        [retrieve_safety_case(_HOSTILE_ID; service=s), retrieve_safety_alert(_HOSTILE_ID; service=s)]
    end
    @test [(r.method, r.target) for r in seen] == [("GET", "/v1/safety/cases/$_HOSTILE_ENC"),
                                                    ("GET", "/v1/safety/alerts/$_HOSTILE_ENC")]
    c = rs[1]
    @test c isa SafetyCaseSuccess && (c.response.id, c.response.created_at, c.response.entity_identifier,
        c.response.reason, c.response.notice) == ("C-abc123", 1787659200, "safety-id-123", "cyber_abuse", "warning")
    a = rs[2]
    @test a isa SafetyAlertSuccess && (a.response.id, a.response.created_at, a.response.request_id, a.response.response_id,
        a.response.model, a.response.request_paused, a.response.error_type, a.response.reason) ==
        ("alert_0123456789abcdef0123456789abcdef", 1787659200, "req_123", "resp_123", "gpt-6-astra", true,
         "potentially_unintended_data_access", "Potentially unintended data access.")
    # `reason` is null for a zero-data-retention alert, and may be null on a case.
    zdr = _answered(() -> retrieve_safety_alert("alert_1"; service=s), 200; body=JSON.json(merge(alert, Dict("reason" => nothing))))
    @test zdr isa SafetyAlertSuccess && isnothing(zdr.response.reason)
    # The reference requires each field typed above: a reply without one is malformed.
    for key in ("id", "created_at", "entity_identifier", "notice")
        @test _answered(() -> retrieve_safety_case("C-1"; service=s), 200; body=JSON.json(delete!(copy(case), key))) isa
              SafetyCallError
    end
    for key in ("id", "created_at", "request_id", "response_id", "model", "request_paused", "error_type")
        @test _answered(() -> retrieve_safety_alert("alert_1"; service=s), 200; body=JSON.json(delete!(copy(alert), key))) isa
              SafetyCallError
    end
    r = _answered(() -> retrieve_safety_case("C-none"; service=s), 404; body="""{"error":{"message":"not found"}}""",
                  headers=["x-request-id" => "req_sc"])
    @test r isa SafetyFailure && r.status == 404 && r.request_id == "req_sc"
    for call in (kw -> retrieve_safety_case("C-1"; kw...), kw -> retrieve_safety_alert("alert_1"; kw...))
        @test _seam_timeout(call((; service=s, config=_TINY_DEADLINE)), SafetyCallError)
        err = try call((; service=AZUREServiceEndpoint)); nothing catch e; e end
        @test err isa ArgumentError && occursin("Safety API is not supported by", err.msg)
        tok = CancelToken(); cancel!(tok)
        r, seen = _with_scripted(() -> call((; service=s, cancel=tok)), (_, _) -> _json(200, "{}"))
        @test isempty(seen) && r isa SafetyCallError && r.cause isa UniLMCancelled
    end
    @test has_capability(OPENAIServiceEndpoint, :safety)
end

@testset "Safety — check a file for content provenance" begin
    s = _SafetyProbe
    png = tempname() * ".png"
    write(png, UInt8[0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a])
    checked = Dict("object" => "content_provenance_check", "created_at" => 1778000000, "results" => [
        Dict("type" => "c2pa", "outcome" => "detected", "validation_state" => "trusted", "issuer" => "OpenAI OpCo, LLC",
             "model" => "gpt-image", "generated_at" => "2026-07-27T18:34:12Z"),
        Dict("type" => "synthid", "outcome" => "not_detected", "model" => nothing, "generated_at" => nothing)])
    r, seen = _with_scripted((_, _) -> _json(200, JSON.json(checked))) do
        check_content_provenance(png; service=s)
    end
    req = only(seen)
    @test (req.method, req.target) == ("POST", "/v1/content_provenance_checks")
    # One multipart part, `file`, whose own Content-Type names the media: the reference takes no other field.
    @test occursin("name=\"file\"; filename=\"$(basename(png))\"", req.body) && occursin("Content-Type: image/png", req.body)
    @test count("Content-Disposition", req.body) == 1
    @test r isa ProvenanceCheckSuccess && r.response.created_at == 1778000000 && length(r.response.results) == 2
    c2pa, synthid = r.response.results
    @test (c2pa.type, c2pa.outcome, c2pa.validation_state, c2pa.issuer, c2pa.model, c2pa.generated_at) ==
          ("c2pa", "detected", "trusted", "OpenAI OpCo, LLC", "gpt-image", "2026-07-27T18:34:12Z")
    @test (synthid.type, synthid.outcome) == ("synthid", "not_detected") &&
          all(isnothing, (synthid.validation_state, synthid.issuer, synthid.model, synthid.generated_at))
    # An explicit media type wins over the one the extension names.
    _, seen = _with_scripted((_, _) -> _json(200, JSON.json(checked))) do
        check_content_provenance(png; mime_type="audio/aac", service=s)
    end
    @test occursin("Content-Type: audio/aac", only(seen).body) && !occursin("image/png", only(seen).body)
    for bad in (Dict("object" => "content_provenance_check", "created_at" => 1),
                Dict("object" => "content_provenance_check", "results" => []),
                merge(checked, Dict("results" => [Dict("type" => "c2pa")])))
        @test _answered(() -> check_content_provenance(png; service=s), 200; body=JSON.json(bad)) isa SafetyCallError
    end
    r = _answered(() -> check_content_provenance(png; service=s), 400; body="""{"error":{"message":"unsupported"}}""")
    @test r isa SafetyFailure && r.status == 400
    err = try check_content_provenance(png * ".missing"; service=s); nothing catch e; e end
    @test err isa ArgumentError && occursin("file not found", err.msg)
    @test _seam_timeout(check_content_provenance(png; service=s, config=_TINY_DEADLINE), SafetyCallError)
    err = try check_content_provenance(png; service=AZUREServiceEndpoint); nothing catch e; e end
    @test err isa ArgumentError && occursin("Content provenance API is not supported by", err.msg)
    tok = CancelToken(); cancel!(tok)
    r, seen = _with_scripted(() -> check_content_provenance(png; service=s, cancel=tok), (_, _) -> _json(200, "{}"))
    @test isempty(seen) && r isa SafetyCallError && r.cause isa UniLMCancelled
    @test has_capability(OPENAIServiceEndpoint, :content_provenance)
    rm(png)
end
