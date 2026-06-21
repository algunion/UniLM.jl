@testset "Webhooks (unit)" begin
    # Canonical Standard-Webhooks test vector.
    secret = "whsec_MfKQ9r8GKYqrTwjUPD8ILPZIo2LaLaSw"
    payload = "{\"test\": 2432232314}"
    headers = Dict(
        "webhook-id" => "msg_p5jXN8AQM9LWM0D4loKWxJek",
        "webhook-timestamp" => "1614265330",
        "webhook-signature" => "v1,g0hM9SsE+OTPJTGt/tmIKtSyZlE3uFJELVlNIOLJ1OE=")

    @testset "valid signature" begin
        @test verify_webhook(payload, headers, secret) == true
        # secret without the whsec_ prefix also works
        @test verify_webhook(payload, headers, "MfKQ9r8GKYqrTwjUPD8ILPZIo2LaLaSw") == true
        # header pairs (not a Dict) accepted too
        @test verify_webhook(payload, collect(headers), secret) == true
    end

    @testset "rejects tampering / missing headers" begin
        @test verify_webhook(payload * " ", headers, secret) == false
        @test verify_webhook(payload, headers, "whsec_" * "AAAA") == false
        @test verify_webhook(payload, Dict("webhook-id" => "x"), secret) == false
    end

    @testset "parse_webhook" begin
        ev = parse_webhook("{\"id\":\"evt_1\",\"type\":\"response.completed\",\"data\":{\"id\":\"resp_1\"}}")
        @test ev.id == "evt_1"
        @test ev.type == "response.completed"
        @test ev.data["id"] == "resp_1"
        @test "response.completed" in WEBHOOK_EVENTS
    end
end
