# ── Image Generation Integration Tests ───────────────────────────────────────
# Real API calls that GENERATE images — billed per image, so they are opt-in.
# Enable with UNILM_RUN_IMAGE_TESTS=true (also needs OPENAI_API_KEY). Off in CI.

if get(ENV, "UNILM_RUN_IMAGE_TESTS", "false") != "true" || !haskey(ENV, "OPENAI_API_KEY")
    @info "Skipping image generation integration tests (set UNILM_RUN_IMAGE_TESTS=true and OPENAI_API_KEY to enable — these make paid image API calls)"
else

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

end
