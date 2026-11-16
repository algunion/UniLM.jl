# test/safety.jl — OpenAI's safety cases and alerts. Local servers only.

# An endpoint at the scripted address that declares the safety API.
struct _SafetyProbe <: UniLM.ServiceEndpoint end
UniLM._resolve_base_url(::Type{_SafetyProbe}) = _url_probe_base[]
UniLM.auth_header(::Type{_SafetyProbe}) = ["Authorization" => "Bearer t", "Content-Type" => "application/json"]
UniLM.provider_capabilities(::Type{_SafetyProbe}) = Set([:safety])

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
        err = try call((; service=SeamProbe)); nothing catch e; e end
        @test err isa ArgumentError && occursin("Safety API is not supported by", err.msg)
        tok = CancelToken(); cancel!(tok)
        r, seen = _with_scripted(() -> call((; service=s, cancel=tok)), (_, _) -> _json(200, "{}"))
        @test isempty(seen) && r isa SafetyCallError && r.cause isa UniLMCancelled
    end
    @test has_capability(OPENAIServiceEndpoint, :safety)
end
