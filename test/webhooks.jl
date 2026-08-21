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
