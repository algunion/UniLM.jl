# ── Image Generation Integration Tests ───────────────────────────────────────
# These are expensive (real API calls generating images), so they run last.

@testset "Image Generation — basic" begin
    r = generate_image(
        "A simple blue square on white background",
        size="1024x1024",
        quality="low"
    )
    @test r isa ImageSuccess
    @test !isnothing(r.response)
    @test length(r.response.data) >= 1
    @test !isnothing(r.response.data[1].b64_json)
end

@testset "Image Generation — image_data accessor" begin
    r = generate_image(
        "A small red circle",
        size="1024x1024",
        quality="low"
    )
    @test r isa ImageSuccess

    imgs = image_data(r)
    @test imgs isa Vector{String}
    @test length(imgs) >= 1
    @test length(imgs[1]) > 100  # non-trivial base64 data
end

@testset "Image Generation — save_image" begin
    r = generate_image(
        "A green triangle",
        size="1024x1024",
        quality="low"
    )
    @test r isa ImageSuccess

    tmpfile = tempname() * ".png"
    try
        result = save_image(image_data(r)[1], tmpfile)
        @test result == tmpfile
        @test isfile(tmpfile)
        @test filesize(tmpfile) > 0
    finally
        isfile(tmpfile) && rm(tmpfile)
    end
end
