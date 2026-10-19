# ─── TypeSafe System One Integration Tests ───────────────────────────────────
# Requires UNILM_LIVE=1 and TYPESAFE_API_KEY (both, or the suite skips).
# Spend is capped at ten calls: one text evaluation, one models listing, and eight
# image, Box and long-context witnesses, two of which the service answers 400. The
# billed input is about 77k tokens, under $0.004 at the jev-latest price; most of it
# is the one long-context request.

if !haskey(ENV, "TYPESAFE_API_KEY") || get(ENV, "UNILM_LIVE", "") != "1"
    @info "Skipping TypeSafe integration tests (set UNILM_LIVE=1 and TYPESAFE_API_KEY to run live)"
else

@testset "TypeSafe System One — live evaluation" begin
    request = SystemOneRequest(
        "Help! My payouts have been failing for 3 days and nobody has replied to my emails.",
        ["department" => choice("Which team should handle this ticket?", (
             billing   = "Payments, invoicing, payouts, refunds",
             technical = "Bugs, outages, integrations",
             sales     = "Pricing, upgrades, new accounts")),
         "urgency" => score("How urgent is this ticket?",
             ["Can wait", "Needs attention this week", "Needs attention today"]),
         "is_frustrated" => noul("Is the customer frustrated?";
             yes = "The customer expresses frustration or impatience",
             no  = "The customer is neutral or satisfied")];
        model = "jev-latest")

    result = ask(request; config=RequestConfig(max_attempts=1, total_deadline=60.0))
    @test result isa SystemOneSuccess
    if result isa SystemOneSuccess
        r = result.response
        # The alias answers with a versioned id, which is what to pin.
        @test !isempty(r.model)
        @test sort(collect(keys(result))) == ["department", "is_frustrated", "urgency"]

        dept = result["department"]
        @test dept isa ChoiceAnswer
        @test dept.choice in ("billing", "technical", "sales")
        @test length(dept.probabilities) == 3
        @test sum(values(dept.probabilities)) ≈ 1.0 atol = 0.1
        @test 0.0 <= dept.confidence <= 1.0

        urg = result["urgency"]
        @test urg isa ScoreAnswer
        @test haskey(urg.legend, 0)
        @test haskey(urg.probabilities, 0)
        @test 0.0 <= urg.score <= 2.0

        frus = result["is_frustrated"]
        @test frus isa NoulAnswer
        @test 0.0 <= frus.noul <= 1.0

        @test r.usage.prompt_tokens > 0
        @info "TypeSafe usage" model = r.model request_id = r.request_id input = r.usage.prompt_tokens output = r.usage.completion_tokens cost = estimated_cost(result)
    else
        @info "TypeSafe evaluation did not succeed" result
    end
end

@testset "TypeSafe System One — live images, Box questions and long context" begin
    cfg = RequestConfig(max_attempts=1, total_deadline=120.0)
    images = joinpath(@__DIR__, "fixtures", "typesafe", "images")
    # A white 512×512 scene: a red rectangle (left 0.20, right 0.55, top 0.30, bottom
    # 0.70) and a blue disc (left 0.68, right 0.88, top 0.12, bottom 0.32).
    scene = ImageAttachment(joinpath(images, "scene-512.png"))
    scene_jpeg = ImageAttachment(joinpath(images, "scene-512.jpg"))
    red = ImageAttachment(joinpath(images, "red-300.png"))
    blue = ImageAttachment(joinpath(images, "blue-300.png"))
    spent = Ref(0.0)
    billed(r) = (r isa SystemOneSuccess && (spent[] += estimated_cost(r)); r)
    ok(r) = r isa SystemOneSuccess

    # A PNG as the whole state, and a Choice about it.
    r = billed(ask(scene, "colour" => choice("What colour is the rectangle in the image?", ["red", "green", "blue"]),
                   "red" => noul("Does the image contain a red shape?"); model="jev-latest", config=cfg))
    @test ok(r)
    ok(r) && @test r["colour"].choice == "red" && r["red"].noul > 0.5

    # A JPEG beside text in an object state, referred to by its key.
    r = billed(ask((photo = scene_jpeg, note = "A drawing of two shapes."), "disc" => noul("Is there a blue disc in `photo`?");
                   model="jev-latest", config=cfg))
    @test ok(r)
    ok(r) && @test r["disc"].noul > 0.5

    # The same image twice: one part, its reference twice.
    twice = SystemOneRequest([red, red], "same" => noul("Are the two images the same colour?"); model="jev-latest")
    @test length(UniLM.encode_system_one(UniLM.TypeSafeProtocol(), TYPESAFEServiceEndpoint, twice).parts) == 1
    r = billed(ask(twice; config=cfg))
    @test ok(r)
    ok(r) && @test r["same"].noul > 0.5

    # Options and levels defined by example pictures; the legend gives the pictures back.
    r = billed(ask(scene, "match" => choice("Which sample has the same colour as the rectangle in the state image?",
                                            (red_sample = red, blue_sample = blue)),
                   "closest" => score("Which level's sample is closest in colour to the rectangle in the state image?",
                                      [blue, red]); model="jev-latest", config=cfg))
    @test ok(r)
    if ok(r)
        @test r["match"].choice == "red_sample"
        @test r["closest"].legend[0] === blue && r["closest"].legend[1] === red
    end

    # Box on Jev 1.14: boxes of the documented shape, around the shapes drawn.
    r = billed(ask(scene, "rect" => box("the red rectangle"), "disc" => box("the blue disc"); model="jev-1.14.0", config=cfg))
    @test ok(r)
    if ok(r)
        @test startswith(r.response.model, "jev-1.14")
        for (name, truth) in (("rect", (0.30, 0.70, 0.20, 0.55)), ("disc", (0.12, 0.32, 0.68, 0.88)))
            b = r[name].box
            @test 0 <= b.top < b.bottom <= 1 && 0 <= b.left < b.right <= 1
            @test all(abs.((b.top, b.bottom, b.left, b.right) .- truth) .<= 0.1)
            @test r[name].confidence === nothing || 0 <= r[name].confidence <= 1
        end
        @info "TypeSafe Box" rect = r["rect"] disc = r["disc"]
    end

    # Box on Jev 1.13 — pinned, since the alias that names it moves: a 400 that names the models to use.
    f = ask(scene, "rect" => box("the red rectangle"); model="jev-1.13.0", config=cfg)
    @test f isa SystemOneFailure && f.status == 400
    f isa SystemOneFailure && @test contains(f.message, "Invalid request.") && contains(f.message, "jev-1.14.0")

    # Just over the default context: a 400 naming long_context, then the same request with it.
    long_state = repeat("The quick brown fox jumps over the lazy dog. ", 7000)     # about 70k tokens
    f = ask(long_state, "fox" => noul("Is the text about a fox?"); model="jev-latest", config=cfg)
    @test f isa SystemOneFailure && f.status == 400 && f.error_type == "max_tokens_exceeded"
    f isa SystemOneFailure && @test contains(f.message, "long_context = true")
    r = billed(ask(long_state, "fox" => noul("Is the text about a fox?"); model="jev-latest", long_context=true, config=cfg))
    @test ok(r)
    ok(r) && @test token_usage(r).prompt_tokens > 65_536 && r["fox"].noul > 0.5

    # What TypeSafe rejects is refused before any request — `ask` throws only before it
    # sends: a GIF, a WebP, and a PNG whose IHDR says 279 or 2000 pixels wide.
    gif = ImageAttachment(vcat(codeunits("GIF89a"), UInt8[0x2c, 0x01, 0x2c, 0x01, 0x00, 0x00, 0x00]))
    webp = ImageAttachment(vcat(codeunits("RIFF"), UInt8[0x1a, 0x00, 0x00, 0x00], codeunits("WEBPVP8 "), zeros(UInt8, 14)))
    wide(w) = ImageAttachment(vcat(red.data[1:16], reinterpret(UInt8, [hton(UInt32(w))]), red.data[21:end]))
    for img in (gif, webp, wide(279), wide(2000))
        @test_throws ArgumentError ask(img, "q" => noul("Is it red?"); model="jev-latest", config=cfg)
    end
    @info "TypeSafe images, Box and long context" cost_usd = spent[]
end

@testset "TypeSafe models — live listing" begin
    result = list_models(; config=RequestConfig(max_attempts=1, total_deadline=60.0))
    @test result isa TypeSafeModelsSuccess
    if result isa TypeSafeModelsSuccess
        @test length(result.models) >= 1
        @test all(m -> !isempty(m.name), result.models)
        @info "TypeSafe models" names = [m.name for m in result.models]
    else
        @info "TypeSafe models listing did not succeed" result
    end
end

end
