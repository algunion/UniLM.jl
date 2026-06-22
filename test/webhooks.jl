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

    @testset "malformed secret base64 → false (not thrown)" begin
        # Reaches src/webhooks.jl:62: a FRESH timestamp passes the replay window (lines 55-57),
        # so control flows to base64decode (line 60). The secret body is not valid base64, so
        # base64decode THROWS and the catch returns false. If the try/catch were removed, this
        # would propagate the DecodeError instead of returning a Bool — so a thrown error here
        # (rather than `== false`) falsifies the guard.
        recent = string(round(Int, time()))
        bad_headers = merge(headers, Dict("webhook-timestamp" => recent))
        @test verify_webhook(payload, bad_headers, "whsec_!!!not-base64!!!") == false
        # Same malformed body without the whsec_ prefix (the secret[7:end] strip is bypassed,
        # base64decode still throws on the raw body) — also caught → false.
        @test verify_webhook(payload, bad_headers, "@@@not base64@@@") == false
    end

    @testset "parse_webhook" begin
        ev = parse_webhook("{\"id\":\"evt_1\",\"type\":\"response.completed\",\"data\":{\"id\":\"resp_1\"}}")
        @test ev.id == "evt_1"
        @test ev.type == "response.completed"
        @test ev.data["id"] == "resp_1"
        @test "response.completed" in WEBHOOK_EVENTS
    end
end
