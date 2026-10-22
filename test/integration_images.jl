# ── Image Generation Integration Tests ───────────────────────────────────────
# Real API calls that GENERATE images — billed per image, so they are opt-in.
# Enable with UNILM_LIVE=1 and UNILM_RUN_IMAGE_TESTS=true; the OpenAI tests also need
# OPENAI_API_KEY, the Gemini tests GEMINI_API_KEY. Off in CI. One image per call.

using Base64

# The width and height a PNG declares in its IHDR chunk, the first after the signature.
_png_size(b::Vector{UInt8}) = (Int(b[17]) << 24 | Int(b[18]) << 16 | Int(b[19]) << 8 | Int(b[20]),
                               Int(b[21]) << 24 | Int(b[22]) << 16 | Int(b[23]) << 8 | Int(b[24]))

# The MIME types Gemini declares for the image parts of its answer (thought parts excluded).
_gemini_image_mimes(r::ImageSuccess) =
    [p["inlineData"]["mimeType"] for p in r.response.raw["candidates"][1]["content"]["parts"]
     if haskey(p, "inlineData") && get(p, "thought", false) !== true]

if get(ENV, "UNILM_RUN_IMAGE_TESTS", "false") != "true" || get(ENV, "UNILM_LIVE", "") != "1"
    @info "Skipping image generation integration tests (set UNILM_LIVE=1 and UNILM_RUN_IMAGE_TESTS=true to enable — these make paid image API calls)"
else

if !haskey(ENV, "OPENAI_API_KEY")
    @info "Skipping OpenAI image generation tests (OPENAI_API_KEY is not set)"
else

@testset "OpenAI images — the default model, with a safety identifier" begin
    r = generate_image("a red circle on white"; size="1024x1024", quality="low", safety_identifier="unilm-test")
    @test r isa ImageSuccess
    # The reply states the identifier it used and the options it applied.
    raw = r.response.raw
    @info "OpenAI image live usage" usage = get(raw, "usage", nothing)
    @test raw["safety_identifier"] == "unilm-test"
    @test (raw["quality"], raw["size"], raw["output_format"]) == ("low", "1024x1024", "png")
    bytes = base64decode(only(image_data(r)))
    @test UniLM._sniff_image_mime(bytes) == "image/png"
    @test _png_size(bytes) == (1024, 1024)
    @test token_usage(r).completion_tokens > 0
    path = tempname() * ".png"
    try
        @test save_image(only(image_data(r)), path) == path
        @test read(path) == bytes
    finally
        rm(path; force=true)
    end
end

end

if !haskey(ENV, "GEMINI_API_KEY")
    @info "Skipping Gemini image generation tests (GEMINI_API_KEY is not set)"
else

@testset "Gemini images — gemini-nano-banana-2.1 draws a square 1K image" begin
    r = generate_image(ImageGeneration(service=GEMINIServiceEndpoint, prompt="a red circle on white",
                                       aspect_ratio="1:1", resolution="1K", safety_identifier="unilm-test"))
    @test r isa ImageSuccess
    @info "Gemini image live usage" usage = get(r.response.raw, "usageMetadata", nothing)
    @test startswith(r.response.raw["modelVersion"], "gemini-nano-banana-2.1")
    bytes = base64decode(only(image_data(r)))
    @test _gemini_image_mimes(r) == ["image/png"]
    @test UniLM._sniff_image_mime(bytes) == "image/png"
    @test _png_size(bytes) == (1024, 1024)               # 1:1 at 1K, per Gemini's size table
    # usageMetadata is the result's usage, counted as on Gemini Chat; a 1K image is 1,120 tokens.
    @test token_usage(r) == UniLM._gemini_usage(r.response.raw["usageMetadata"])
    @test token_usage(r).completion_tokens >= 1120
end

@testset "Gemini image edits — gemini-nano-banana-2.1 edits an input image" begin
    fixture = joinpath(@__DIR__, "fixtures", "ollama", "red_circle.png")
    r = edit_image(fixture, "make the circle blue"; service=GEMINIServiceEndpoint)
    @test r isa ImageSuccess
    @info "Gemini image edit live usage" usage = get(r.response.raw, "usageMetadata", nothing)
    bytes = base64decode(only(image_data(r)))
    @test _gemini_image_mimes(r) == ["image/png"]
    @test UniLM._sniff_image_mime(bytes) == "image/png"
    @test bytes != read(fixture)
    @test token_usage(r) == UniLM._gemini_usage(r.response.raw["usageMetadata"])
    @test token_usage(r).prompt_tokens > 0                # the input image is counted as input
end

end

end
