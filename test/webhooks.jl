@testset "Webhooks (unit)" begin
    # Canonical Standard-Webhooks vector (2021 timestamp → disable the replay window to test the signature).
    secret = "whsec_MfKQ9r8GKYqrTwjUPD8ILPZIo2LaLaSw"
    payload = "{\"test\": 2432232314}"
    headers = Dict(
        "webhook-id" => "msg_p5jXN8AQM9LWM0D4loKWxJek",
        "webhook-timestamp" => "1614265330",
        "webhook-signature" => "v1,g0hM9SsE+OTPJTGt/tmIKtSyZlE3uFJELVlNIOLJ1OE=")

    @testset "valid signature (replay check disabled)" begin
        @test verify_webhook(payload, headers, secret; tolerance_seconds=Inf) == true
        @test verify_webhook(payload, headers, "MfKQ9r8GKYqrTwjUPD8ILPZIo2LaLaSw"; tolerance_seconds=Inf) == true
        @test verify_webhook(payload, collect(headers), secret; tolerance_seconds=Inf) == true   # vector of pairs
    end

    @testset "rejects tampering / missing headers / wrong version label" begin
        @test verify_webhook(payload * " ", headers, secret; tolerance_seconds=Inf) == false
        @test verify_webhook(payload, headers, "whsec_AAAA"; tolerance_seconds=Inf) == false
        @test verify_webhook(payload, Dict("webhook-id" => "x"), secret) == false
        bad = merge(headers, Dict("webhook-signature" => "v2,g0hM9SsE+OTPJTGt/tmIKtSyZlE3uFJELVlNIOLJ1OE="))
        @test verify_webhook(payload, bad, secret; tolerance_seconds=Inf) == false   # right digest, wrong version
    end

    @testset "replay window" begin
        @test verify_webhook(payload, headers, secret) == false   # 2021 timestamp rejected by the default window
        # A recent timestamp passes the time check (signature still required → wrong sig ⇒ false).
        recent = string(round(Int, time()))
        @test verify_webhook(payload, merge(headers, Dict("webhook-timestamp" => recent,
            "webhook-signature" => "v1,d3Jvbmdfc2ln")), secret) == false
        # Non-numeric timestamp rejected (not thrown); non-String header values do not throw.
        @test verify_webhook(payload, merge(headers, Dict("webhook-timestamp" => "nope")), secret) == false
        @test verify_webhook(payload, Dict("webhook-id" => 1, "webhook-timestamp" => 2, "webhook-signature" => "v1,x"), secret; tolerance_seconds=Inf) == false
    end

    @testset "malformed secret base64 → ArgumentError (a config error, not a verdict)" begin
        # A FRESH timestamp passes the replay window, so control reaches base64decode.
        # The secret body is not valid base64: that is this endpoint's misconfiguration,
        # and answering `false` would silently reject every inbound webhook forever,
        # indistinguishable from an attacker's bad signature.
        recent = string(round(Int, time()))
        bad_headers = merge(headers, Dict("webhook-timestamp" => recent))
        @test_throws ArgumentError verify_webhook(payload, bad_headers, "whsec_!!!not-base64!!!")
        # Same malformed body without the whsec_ prefix (the secret[7:end] strip is bypassed,
        # base64decode still fails on the raw body).
        @test_throws ArgumentError verify_webhook(payload, bad_headers, "@@@not base64@@@")
        err = try; verify_webhook(payload, bad_headers, "whsec_!!!not-base64!!!"); catch e; e; end
        @test contains(sprint(showerror, err), "not valid base64")
        # A well-formed secret with a wrong signature is still a plain `false`: only
        # genuine verification failures return, so `false` keeps one meaning.
        @test verify_webhook(payload, bad_headers, secret) == false
        # `base64decode` skips characters outside the alphabet, so these decoded to another key
        # (`whsec_***` to the empty one) and every delivery verified `false`. RFC 4648 §3.3:
        # "Implementations MUST reject the encoded data if it contains characters outside the
        # base alphabet"; a space inside the secret counts as one.
        body = "MfKQ9r8GKYqrTwjUPD8ILPZIo2LaLaSw"
        for bad in ("whsec_***", "whsec_" * body * "*", "whsec_MfKQ 9r8GKYqrTwjUPD8ILPZIo2LaLaSw",
                    "whsec_MfKQ9r8G\nKYqrTwjUPD8ILPZIo2LaLaSw", "whsec_MfKQ9r8G-YqrTwjUPD8ILPZIo2LaLaSw", body * "_")
            err = try verify_webhook(payload, bad_headers, bad); nothing catch e; e end
            @test err isa ArgumentError && startswith(err.msg, "webhook signing secret is not valid base64")
        end
        # Whitespace around the secret is no part of it: a secret read with its line break from a file
        # or an env file verifies with the right key, as OpenAI's SDKs, whose decoders skip it, verify it.
        key = UniLM.base64decode(body)
        signed = Dict(bad_headers..., "webhook-signature" => "v1," * UniLM.base64encode(UniLM._hmac_sha256(key,
            Vector{UInt8}(string(bad_headers["webhook-id"], ".", bad_headers["webhook-timestamp"], ".", payload)))))
        for around in ("whsec_" * body * "\n", "whsec_" * body * "\r\n", " whsec_" * body * " ", "\t" * body * "\n")
            @test verify_webhook(payload, signed, around) == true
        end
        # Padding goes at the end only, and only as much as the last group lacks.
        for bad in ("whsec_YQ=", "whsec_YWJj=", "whsec_YQ===", "whsec_=YWJj", "whsec_YW=j", "whsec_Y", "whsec_YWJjZ")
            @test_throws ArgumentError verify_webhook(payload, bad_headers, bad)
        end
        # A last group of two or three characters may come without its padding, as the Standard
        # Webhooks reference library pads an unpadded secret before decoding it.
        for good in ("whsec_YQ", "whsec_YQ==", "whsec_YWI", "whsec_YWI=", "whsec_YWJj", "YWJj")
            @test verify_webhook(payload, bad_headers, good) == false
        end
    end

    @testset "an empty secret → ArgumentError, whatever the delivery" begin
        # Anyone can sign with an empty key, so under one a forged delivery verified `true` and a
        # genuine one `false`; the Standard Webhooks libraries refuse an empty secret when the
        # verifier is built ("webhook secret may not be empty"). Refused before any check: the
        # stale, unsigned and forged deliveries below each returned before.
        fresh = merge(headers, Dict("webhook-timestamp" => string(round(Int, time()))))
        forged = merge(fresh, Dict("webhook-signature" => "v1," * UniLM.base64encode(UniLM._hmac_sha256(UInt8[],
            Vector{UInt8}(string(fresh["webhook-id"], ".", fresh["webhook-timestamp"], ".", payload))))))
        for empty in ("", "   ", "\n", "whsec_", " whsec_ \n"), h in (forged, fresh, headers, Dict{String,String}())
            err = try verify_webhook(payload, h, empty); nothing catch e; e end
            @test err isa ArgumentError && startswith(err.msg, "webhook signing secret is empty")
        end
    end

    @testset "non-finite timestamps do not slip past the replay window" begin
        # `tryparse(Float64, "NaN")` is NaN, and every comparison with NaN is false —
        # `abs(time() - NaN) > tolerance` waved the frame through with the window on.
        # The signatures below are genuine (computed over the literal timestamp), so
        # only the freshness check can reject them.
        key = UniLM.base64decode("MfKQ9r8GKYqrTwjUPD8ILPZIo2LaLaSw")
        wid = headers["webhook-id"]
        for wts in ("NaN", "nan", "Inf", "-Inf", "inf")
            sig = UniLM.base64encode(UniLM._hmac_sha256(key, Vector{UInt8}(string(wid, ".", wts, ".", payload))))
            h = merge(headers, Dict("webhook-timestamp" => wts, "webhook-signature" => "v1,$sig"))
            @test verify_webhook(payload, h, secret) == false
            # With the window explicitly disabled the same frames verify, proving the
            # signatures are valid and the rejection above came from the time check.
            @test verify_webhook(payload, h, secret; tolerance_seconds=Inf) == true
        end
        # A finite, fresh timestamp with a genuine signature still passes.
        now = string(round(Int, time()))
        sig = UniLM.base64encode(UniLM._hmac_sha256(key, Vector{UInt8}(string(wid, ".", now, ".", payload))))
        @test verify_webhook(payload, merge(headers,
            Dict("webhook-timestamp" => now, "webhook-signature" => "v1,$sig")), secret) == true
    end

    @testset "parse_webhook" begin
        ev = parse_webhook("{\"id\":\"evt_1\",\"type\":\"response.completed\",\"data\":{\"id\":\"resp_1\"}}")
        @test ev.id == "evt_1"
        @test ev.type == "response.completed"
        @test ev.data["id"] == "resp_1"
        @test "response.completed" in WEBHOOK_EVENTS
    end
end

@testset "WEBHOOK_EVENTS are the events of OpenAI's webhooks reference" begin
    @test WEBHOOK_EVENTS == (
        "response.completed", "response.cancelled", "response.failed", "response.incomplete",
        "batch.completed", "batch.cancelled", "batch.expired", "batch.failed",
        "fine_tuning.job.succeeded", "fine_tuning.job.failed", "fine_tuning.job.cancelled",
        "realtime.call.incoming", "live.call.incoming", "live.transport.incoming", "live.transport.connected",
        "safety.warning_issued", "safety.deactivation_issued", "safety.alert.created", "safety.org_alert.created",
        "agent.environment.ready", "agent.environment.failed", "agent.environment.suspended", "agent.environment.expired",
        "agent.session.created", "agent.session.action_required", "agent.session.in_progress", "agent.session.idle",
        "agent.session.failed")
    # The Evals API shuts down on 2026-11-30, and its run events are not listed; the docstring says
    # so, since a filter on the list drops the events OpenAI still sends until then.
    @test !any(e -> startswith(e, "eval."), WEBHOOK_EVENTS)
    @test WEBHOOK_EVENTS isa NTuple{28,String}
    doc = string(@doc WEBHOOK_EVENTS)
    # The reference: live.transport.connected is "Sent when a direct Live WebRTC session is created or an Instant
    # Connect session activates".
    @test occursin("`live.transport.connected`", doc) && occursin("Instant Connect", doc)
    @test occursin("also lists `eval.run.succeeded`, `eval.run.failed` and", doc) &&
          occursin("`eval.run.canceled`, which OpenAI sends until its Evals API shuts down on November 30, 2026", doc)
    @test occursin("agent sessions", doc) && occursin("environments", doc)
    # The Videos API shut down on 2026-09-24, yet OpenAI's event-type list still names its two events: left out, and said.
    @test !any(e -> startswith(e, "video."), WEBHOOK_EVENTS)
    @test occursin("still names `video.completed` and", doc) &&
          occursin("`video.failed`, the events of its Videos API, which shut down on September 24, 2026", doc)
end

@testset "parse_webhook reads each event family's envelope" begin
    # One delivery per family, with the data OpenAI's reference shows for it: `id`, `type` and `created_at`
    # typed, the payload as `data`, the envelope's `object` in `raw`.
    sip = [Dict("name" => "From", "value" => "<sip:alice@example.com>;tag=abc123"),
           Dict("name" => "To", "value" => "<sip:recipient@example.com>")]
    fixtures = [
        ("agent.environment.ready", 1719168000, Dict("id" => "ccarenv_123")),
        ("agent.session.created", 1719168001, Dict("id" => "sess_123", "environment_id" => "ccarenv_123",
            "environment_type" => "self_hosted", "connect" => Dict("remote_url" => "https://api.openai.com/v1/agents/api/connect/example"))),
        ("batch.completed", 1719168002, Dict("id" => "batch_abc123")),
        ("fine_tuning.job.succeeded", 1719168003, Dict("id" => "ftjob_abc123")),
        ("live.transport.incoming", 1719168004, Dict("type" => "sip", "session_id" => "live_u0_479a2756",
            "sip_media_security" => "srtp", "sip_headers" => sip)),
        ("live.transport.connected", 1793815201, Dict("session_id" => "live_u0_abc123", "type" => "webrtc")),
        ("realtime.call.incoming", 1719168005, Dict("call_id" => "rtc_u0_479a2756", "sip_headers" => sip)),
        ("response.completed", 1719168006, Dict("id" => "resp_abc123")),
        ("safety.alert.created", 1787659200, Dict("id" => "alert_0123456789abcdef0123456789abcdef")),
        ("safety.warning_issued", 1787659201, Dict("id" => "C-abc123"))]
    for (type, created, data) in fixtures
        ev = parse_webhook(JSON.json(Dict("id" => "evt_123", "object" => "event", "created_at" => created,
                                          "type" => type, "data" => data)))
        @test (ev.id, ev.type, ev.created_at, ev.data) == ("evt_123", type, created, data)
        @test ev.created_at isa Int && ev.raw["object"] == "event" && ev.type in WEBHOOK_EVENTS
    end
end

# Webhook endpoint management at the scripted endpoint: an endpoint that declares `:webhooks`.
struct _WebhookProbe <: UniLM.ServiceEndpoint end
UniLM._resolve_base_url(::Type{_WebhookProbe}) = _url_probe_base[]
UniLM.auth_header(::Type{_WebhookProbe}) = ["Authorization" => "Bearer t", "Content-Type" => "application/json"]
UniLM.provider_capabilities(::Type{_WebhookProbe}) = Set([:webhooks])

@testset "Webhook endpoints — create, list, retrieve, update, rotate, test, delete, event types" begin
    s = _WebhookProbe
    ep = Dict{String,Any}("id" => "we_1", "object" => "webhook_endpoint", "created_at" => 1, "updated_at" => 2,
        "name" => "ops", "url" => "https://example.com/hook", "event_types" => ["batch.completed"],
        "signing_secret_hint" => "whsec_...abcd")
    secret = merge(ep, Dict("signing_secret" => "whsec_c2VjcmV0"))
    replies = [secret, Dict("object" => "list", "data" => [ep], "first_id" => "we_1", "last_id" => "we_1", "has_more" => true),
        ep, merge(ep, Dict("name" => "ops2")), secret, secret,
        Dict("object" => "webhook_endpoint.test", "webhook_endpoint_id" => "we_1", "event_type" => "batch.completed",
             "status_code" => 204, "success" => true),
        Dict("object" => "webhook_endpoint.deleted", "id" => "we_1", "deleted" => true),
        Dict("object" => "list", "data" => ["batch.completed", "agent.session.created"])]
    rs, seen = _with_scripted((n, _) -> _json(200, JSON.json(replies[n]))) do
        [create_webhook_endpoint(; name="ops", url="https://example.com/hook", event_types=["batch.completed"], service=s),
         list_webhook_endpoints(; limit=2, after=_HOSTILE_ID, service=s),
         retrieve_webhook_endpoint(_HOSTILE_ID; service=s),
         update_webhook_endpoint(_HOSTILE_ID; name="ops2", event_types=["batch.completed", "batch.failed"], service=s),
         rotate_webhook_endpoint_secret(_HOSTILE_ID; keep_old_secret_active_for_24_hours=true, service=s),
         rotate_webhook_endpoint_secret("we_1"; service=s),
         test_webhook_endpoint(_HOSTILE_ID, "batch.completed"; service=s),
         delete_webhook_endpoint(_HOSTILE_ID; service=s),
         list_webhook_event_types(; service=s)]
    end
    E = "/v1/webhook_endpoints"
    @test [(r.method, r.target) for r in seen] == [("POST", E), ("GET", "$E?limit=2&after=$_HOSTILE_ENC"),
        ("GET", "$E/$_HOSTILE_ENC"), ("POST", "$E/$_HOSTILE_ENC"), ("POST", "$E/$_HOSTILE_ENC/rotate_secret"),
        ("POST", "$E/we_1/rotate_secret"), ("POST", "$E/$_HOSTILE_ENC/test"), ("DELETE", "$E/$_HOSTILE_ENC"),
        ("GET", "/v1/webhook_event_types")]
    body(i) = JSON.parse(seen[i].body; dicttype=Dict{String,Any})
    @test body(1) == Dict("name" => "ops", "url" => "https://example.com/hook", "event_types" => ["batch.completed"])
    @test body(4) == Dict("name" => "ops2", "event_types" => ["batch.completed", "batch.failed"])
    @test body(5) == Dict("keep_old_secret_active_for_24_hours" => true) && body(6) == Dict()
    @test body(7) == Dict("event_type" => "batch.completed")
    c = rs[1]
    @test c isa WebhookEndpointSuccess && c.response.id == "we_1" && c.response.signing_secret == "whsec_c2VjcmV0"
    @test (c.response.name, c.response.url, c.response.event_types, c.response.created_at, c.response.updated_at,
           c.response.signing_secret_hint) == ("ops", "https://example.com/hook", ["batch.completed"], 1, 2, "whsec_...abcd")
    @test rs[2] isa WebhookEndpointListSuccess && rs[2].response.has_more && only(rs[2].response.data).id == "we_1"
    @test rs[3] isa WebhookEndpointSuccess && isnothing(rs[3].response.signing_secret)
    @test rs[4] isa WebhookEndpointSuccess && rs[4].response.name == "ops2"
    @test rs[5] isa WebhookEndpointSuccess && rs[5].response.signing_secret == "whsec_c2VjcmV0"
    @test rs[7] isa WebhookTestSuccess && (rs[7].response.webhook_endpoint_id, rs[7].response.event_type,
                                           rs[7].response.status_code) == ("we_1", "batch.completed", 204)
    @test rs[8] isa WebhookEndpointDeleteSuccess && rs[8].id == "we_1" && rs[8].deleted
    @test rs[9] isa WebhookEventTypesSuccess && rs[9].event_types == ["batch.completed", "agent.session.created"]
end

@testset "Webhook endpoints — a printed or logged endpoint shows no signing secret" begin
    # The signing secret verifies every delivery: a REPL echo or a logged create or rotation must not print it,
    # neither as the field nor inside `raw`, while the field still holds it.
    secret = "whsec_bGl2ZXNpZ25pbmdzZWNyZXQ="
    d = Dict{String,Any}("id" => "we_1", "object" => "webhook_endpoint", "created_at" => 1, "name" => "ops",
        "url" => "https://example.com/hook", "event_types" => ["batch.completed"], "signing_secret_hint" => "whsec_...cmV0",
        "signing_secret" => secret)
    ep = UniLM._parse_webhook_endpoint(d; secret=true)
    @test ep.signing_secret == secret && ep.raw["signing_secret"] == secret
    page = WebhookEndpointList(data=[ep], has_more=false, raw=Dict{String,Any}("object" => "list", "data" => [d]))
    for x in (ep, WebhookEndpointSuccess(response=ep), page, WebhookEndpointListSuccess(response=page), [ep])
        log = IOBuffer()
        Base.CoreLogging.with_logger(Base.CoreLogging.ConsoleLogger(log)) do
            @info "made" result = x
        end
        for s in (repr(x), sprint(show, x), repr(MIME"text/plain"(), x), string(x), takestring!(log))
            @test !occursin("bGl2ZXNpZ25pbmdzZWNyZXQ", s) && occursin("we_1", s)
        end
    end
    @test occursin("signing_secret=<not shown>", repr(ep))
    @test occursin("signing_secret=nothing", repr(UniLM._parse_webhook_endpoint(delete!(copy(d), "signing_secret"))))
end

@testset "Webhook endpoints — malformed replies, failures, refusals, seams" begin
    s = _WebhookProbe
    ep = Dict{String,Any}("id" => "we_1", "object" => "webhook_endpoint", "created_at" => 1, "name" => "ops",
        "url" => "https://example.com/hook", "event_types" => ["batch.completed"], "signing_secret_hint" => nothing)
    answer(call, reply) = _answered(call, 200; body=JSON.json(reply))
    create = () -> create_webhook_endpoint(; name="ops", url="https://example.com/hook", event_types=["batch.completed"], service=s)
    # A create or a rotation that returns no signing secret leaves the endpoint unverifiable.
    @test answer(create, ep) isa WebhookEndpointCallError
    @test answer(() -> rotate_webhook_endpoint_secret("we_1"; service=s), ep) isa WebhookEndpointCallError
    for key in ("id", "name", "url", "event_types", "created_at")
        @test answer(() -> retrieve_webhook_endpoint("we_1"; service=s), delete!(copy(ep), key)) isa WebhookEndpointCallError
    end
    @test answer(() -> retrieve_webhook_endpoint("we_1"; service=s), ep) isa WebhookEndpointSuccess
    for page in (Dict(), Dict("data" => [ep]), Dict("has_more" => false))
        @test answer(() -> list_webhook_endpoints(; service=s), page) isa WebhookEndpointCallError
    end
    for rec in (Dict("deleted" => true), Dict("id" => "we_1", "deleted" => false), Dict("id" => "we_1"),
                Dict("id" => "we_1", "deleted" => true), Dict("object" => "webhook_endpoint", "id" => "we_1", "deleted" => true))
        @test answer(() -> delete_webhook_endpoint("we_1"; service=s), rec) isa WebhookEndpointCallError
    end
    r = answer(() -> delete_webhook_endpoint("we_1"; service=s), Dict("object" => "webhook_endpoint", "id" => "we_1", "deleted" => true))
    @test r isa WebhookEndpointCallError && occursin("\"webhook_endpoint.deleted\"", r.error)
    @test answer(() -> test_webhook_endpoint("we_1", "batch.completed"; service=s),
                 Dict("webhook_endpoint_id" => "we_1", "event_type" => "batch.completed")) isa WebhookEndpointCallError
    @test answer(() -> list_webhook_event_types(; service=s), Dict("object" => "list")) isa WebhookEndpointCallError
    r = _answered(() -> retrieve_webhook_endpoint("we_x"; service=s), 404; body="""{"error":{"message":"not found"}}""",
                  headers=["x-request-id" => "req_we"])
    @test r isa WebhookEndpointFailure && r.status == 404 && r.request_id == "req_we"
    calls = (kw -> create_webhook_endpoint(; name="n", url="https://example.com", event_types=["batch.completed"], kw...),
             kw -> list_webhook_endpoints(; kw...), kw -> retrieve_webhook_endpoint("we_1"; kw...),
             kw -> update_webhook_endpoint("we_1"; name="n", kw...), kw -> delete_webhook_endpoint("we_1"; kw...),
             kw -> rotate_webhook_endpoint_secret("we_1"; kw...),
             kw -> test_webhook_endpoint("we_1", "batch.completed"; kw...), kw -> list_webhook_event_types(; kw...))
    for call in calls
        @test _seam_timeout(call((; service=s, config=_TINY_DEADLINE)), WebhookEndpointCallError)
        # An endpoint without `:webhooks` is refused before anything is sent.
        err = try call((; service=AZUREServiceEndpoint)); nothing catch e; e end
        @test err isa ArgumentError && occursin("Webhook endpoints API is not supported by", err.msg)
        tok = CancelToken(); cancel!(tok)
        r, seen = _with_scripted(() -> call((; service=s, cancel=tok)), (_, _) -> _json(200, "{}"))
        @test isempty(seen) && r isa WebhookEndpointCallError && r.cause isa UniLMCancelled
    end
    @test has_capability(OPENAIServiceEndpoint, :webhooks)
end
