# ── Image Generation Integration Tests ───────────────────────────────────────
# Real API calls that GENERATE images — billed per image, so they are opt-in.
# Enable with UNILM_LIVE=1 and UNILM_RUN_IMAGE_TESTS=true; the OpenAI tests also need
# OPENAI_API_KEY, the Gemini tests GEMINI_API_KEY. Off in CI. One image per call.

using Base64

# The width and height a PNG declares in its IHDR chunk, the first after the signature.
_png_size(b::Vector{UInt8}) = (Int(b[17]) << 24 | Int(b[18]) << 16 | Int(b[19]) << 8 | Int(b[20]),
                               Int(b[21]) << 24 | Int(b[22]) << 16 | Int(b[23]) << 8 | Int(b[24]))

# The width and height a JPEG declares in its frame header (the first SOFn segment).
function _jpeg_size(b::Vector{UInt8})
    i = 3                                                 # the first segment after SOI (FF D8)
    while i + 8 <= length(b) && b[i] == 0xff
        b[i+1] in (0xc0, 0xc1, 0xc2, 0xc3, 0xc5, 0xc6, 0xc7, 0xc9, 0xca, 0xcb, 0xcd, 0xce, 0xcf) &&
            return (Int(b[i+7]) << 8 | Int(b[i+8]), Int(b[i+5]) << 8 | Int(b[i+6]))
        i += 2 + (Int(b[i+2]) << 8 | Int(b[i+3]))
    end
    nothing
end

# The MIME types Gemini declares for the image parts of its answer (thought parts excluded).
_gemini_image_mimes(r::ImageSuccess) =
    [p["inlineData"]["mimeType"] for p in r.response.raw["candidates"][1]["content"]["parts"]
     if haskey(p, "inlineData") && get(p, "thought", false) !== true]

# The token counts of a usage: an image result's usage has the Images API shape, which carries
# no billed tier (the reply's own tier stays in `raw`).
_gemini_image_counts(u::TokenUsage) = (u.prompt_tokens, u.completion_tokens, u.total_tokens, u.cached_tokens,
                                       u.reasoning_tokens)

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

@testset "OpenAI images — gpt-image-2.6-flare at complexity low, the smallest size" begin
    # 1024x640 is 655,360 pixels, the fewest a custom size may have; 450 output tokens on 2026-11-16.
    r = generate_image("a red circle on white"; model="gpt-image-2.6-flare", size="1024x640", quality="low",
                       complexity="low")
    @test r isa ImageSuccess
    @info "OpenAI image (complexity) live usage" usage = get(r.response.raw, "usage", nothing)
    @test (r.response.raw["size"], r.response.raw["output_format"]) == ("1024x640", "png")
    bytes = base64decode(only(image_data(r)))
    @test UniLM._sniff_image_mime(bytes) == "image/png" && _png_size(bytes) == (1024, 640)
    @test token_usage(r).completion_tokens > 0 && isnothing(r.response.svg_usage)
end

@testset "OpenAI images — a 2-frame GIF on gpt-image-2.6-flare" begin
    # The fewest frames at the smallest size: 1,584 output tokens on 2026-11-16. An organization without the GIF
    # preview gets an HTTP 400, "Requires access to the GIF preview for the selected model".
    r = generate_image("a red ball bouncing once on white"; model="gpt-image-2.6-flare", size="1024x640", quality="low",
                       output_format="gif", frame_count=2, frame_duration_ms=200)
    @test r isa ImageSuccess
    @info "OpenAI image (GIF) live usage" usage = get(r.response.raw, "usage", nothing)
    @test r.response.raw["output_format"] == "gif"
    @test String(base64decode(only(image_data(r)))[1:6]) == "GIF89a"
    @test token_usage(r).completion_tokens > 0
end

@testset "OpenAI images — the Responses image generation tool at complexity low runs gpt-image-2.6-sunburst" begin
    # On 2026-11-16 the reply repeated the tool with "model":"gpt-image-2.6-sunburst" and "complexity":"low"; its
    # tool_usage reported 824 image output tokens.
    r = respond(Respond(model="gpt-6-luna", input="Draw a red circle on a white background.", store=false,
                        reasoning=Reasoning(effort="none"), tool_choice=tool_choice_hosted("image_generation"),
                        tools=[image_generation_tool(quality="low", size="1024x1024", complexity="low")]))
    @test r isa ResponseSuccess
    tool = only(t for t in r.response.raw["tools"] if t["type"] == "image_generation")
    @test (tool["model"], tool["complexity"]) == ("gpt-image-2.6-sunburst", "low")
    @test length(image_generation_results(r)) == 1
    used = tool_usage(r)
    @info "image tool live usage" used cost = estimated_cost(r)
    @test used.image_output_tokens > 0
    u = token_usage(r)
    @test isapprox(estimated_cost(r), (u.prompt_tokens * 0.10 + u.completion_tokens * 0.50 +
        (used.image_input_tokens - used.image_input_image_tokens) * 5.0 + used.image_input_image_tokens * 8.0 +
        used.image_output_tokens * 30.0) / 1e6; rtol=1e-9)
end

@testset "OpenAI images — an SVG on gpt-image-2.6-flare, authored by gpt-6-luna, reports svg_usage" begin
    # An organization with SVG model selection must name the author ("SVG generation requires 'svg.model'.",
    # HTTP 400 without it, 2026-11-16); the reference image cost 450 output tokens, the author 1,364 tokens.
    r = generate_image(ImageGeneration(prompt="a simple flat red circle icon", model="gpt-image-2.6-flare", size="1024x640",
                                       quality="low", output_format="svg", n=1, svg=SvgOptions(model="gpt-6-luna", effort="low")))
    @test r isa ImageSuccess
    @info "OpenAI image (SVG) live usage" usage = get(r.response.raw, "usage", nothing) svg_usage = r.response.svg_usage
    @test r.response.raw["output_format"] == "svg"
    @test startswith(String(base64decode(only(image_data(r)))), "<svg")
    s = r.response.svg_usage
    @test s isa Dict{String,Any} && startswith(s["model"], "gpt-6-luna") && s["total_tokens"] > 0
    rates = Dict("gpt-image-2.6-flare" => UniLM._price(5.0, 1.25, 30.0))
    @test (@test_logs (:warn, r"SVG authoring tokens") estimated_cost(r; model="gpt-image-2.6-flare", pricing=rates)) > 0
end

end

if !haskey(ENV, "GEMINI_API_KEY")
    @info "Skipping Gemini image generation tests (GEMINI_API_KEY is not set)"
else

@testset "Gemini images — gemini-nano-banana-2.1 draws a square 1K image" begin
    # Gemini's recitation filter refused "a red circle on white" again and again (finishReason
    # IMAGE_RECITATION, which Google does not bill: "You will not be charged for this request"); this
    # prompt drew 3 images in 3 runs on 2 November 2026. A recitation refusal is asked again, up to three
    # times in all.
    r = nothing
    for attempt in 1:3
        r = generate_image(ImageGeneration(service=GEMINIServiceEndpoint,
                                           prompt="three small blue squares in a row on a light grey background",
                                           aspect_ratio="1:1", resolution="1K", safety_identifier="unilm-test"))
        r isa ImageCallError && occursin("IMAGE_RECITATION", r.error) || break
        @info "Gemini image: a recitation refusal" attempt error = r.error
    end
    @test r isa ImageSuccess
    if r isa ImageSuccess
        @info "Gemini image live usage" usage = get(r.response.raw, "usageMetadata", nothing)
        @test startswith(r.response.raw["modelVersion"], "gemini-nano-banana-2.1")
        bytes = base64decode(only(image_data(r)))
        # Gemini answers JPEG (observed), and says so in the part's mimeType.
        @test _gemini_image_mimes(r) == ["image/jpeg"]
        @test UniLM._sniff_image_mime(bytes) == "image/jpeg"
        @test _jpeg_size(bytes) == (1024, 1024)              # 1:1 at 1K, per Gemini's size table
        # usageMetadata is the result's usage, counted as on Gemini Chat; a 1K image is 1,120 tokens.
        @test _gemini_image_counts(token_usage(r)) == _gemini_image_counts(UniLM._gemini_usage(r.response.raw["usageMetadata"]))
        @test token_usage(r).completion_tokens >= 1120
    end
end

@testset "Gemini image edits — gemini-nano-banana-2.1 edits an input image" begin
    fixture = joinpath(@__DIR__, "fixtures", "ollama", "red_circle.png")
    r = edit_image(fixture, "make the circle blue"; service=GEMINIServiceEndpoint)
    @test r isa ImageSuccess
    @info "Gemini image edit live usage" usage = get(r.response.raw, "usageMetadata", nothing)
    bytes = base64decode(only(image_data(r)))
    @test _gemini_image_mimes(r) == ["image/jpeg"]
    @test UniLM._sniff_image_mime(bytes) == "image/jpeg"
    @test _gemini_image_counts(token_usage(r)) == _gemini_image_counts(UniLM._gemini_usage(r.response.raw["usageMetadata"]))
    @test token_usage(r).prompt_tokens > 0                # the input image is counted as input
end

end

end
