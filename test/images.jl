@testset "ImageGeneration" begin
    @testset "minimal creation" begin
        ig = ImageGeneration(prompt="A cute robot")
        @test ig.model == ""  # resolved to "gpt-image-2" during serialization
        @test JSON.parse(JSON.json(ig))["model"] == "gpt-image-2"
        @test ig.prompt == "A cute robot"
        @test ig.service == UniLM.OPENAIServiceEndpoint
        @test isnothing(ig.n)
        @test isnothing(ig.size)
        @test isnothing(ig.quality)
        @test isnothing(ig.background)
        @test isnothing(ig.output_format)
        @test isnothing(ig.output_compression)
        @test isnothing(ig.user)
    end

    @testset "full creation" begin
        ig = ImageGeneration(
            prompt="A logo",
            model="gpt-image-1.5",
            n=2,
            size="1024x1024",
            quality="high",
            background="transparent",
            output_format="png",
            output_compression=nothing,
            user="user_123"
        )
        @test ig.prompt == "A logo"
        @test ig.n == 2
        @test ig.size == "1024x1024"
        @test ig.quality == "high"
        @test ig.background == "transparent"
        @test ig.output_format == "png"
        @test ig.user == "user_123"
    end

    @testset "JSON serialization" begin
        ig = ImageGeneration(prompt="test", quality="high", size="1024x1024")
        lowered = JSON.lower(ig)
        @test lowered[:model] == "gpt-image-2"
        @test lowered[:prompt] == "test"
        @test lowered[:quality] == "high"
        @test lowered[:size] == "1024x1024"
        # nil fields excluded
        @test !haskey(lowered, :service)
        @test !haskey(lowered, :n)
        @test !haskey(lowered, :background)
        @test !haskey(lowered, :output_format)
        @test !haskey(lowered, :output_compression)
        @test !haskey(lowered, :user)
    end

    @testset "JSON round-trip" begin
        ig = ImageGeneration(prompt="A sunset", n=2, size="1536x1024", background="opaque")
        json_str = JSON.json(ig)
        parsed = JSON.parse(json_str)
        @test parsed["model"] == "gpt-image-2"
        @test parsed["prompt"] == "A sunset"
        @test parsed["n"] == 2
        @test parsed["size"] == "1536x1024"
        @test parsed["background"] == "opaque"
        @test !haskey(parsed, "service")
        @test !haskey(parsed, "quality")
    end
end

@testset "ImageObject" begin
    @testset "defaults" begin
        io = ImageObject()
        @test isnothing(io.b64_json)
        @test isnothing(io.revised_prompt)
    end

    @testset "with data" begin
        io = ImageObject(b64_json="aGVsbG8=", revised_prompt="A cute robot in watercolor")
        @test io.b64_json == "aGVsbG8="
        @test io.revised_prompt == "A cute robot in watercolor"
    end
end

@testset "ImageResponse" begin
    ir = ImageResponse(
        created=1713833628,
        data=[ImageObject(b64_json="aGVsbG8="), ImageObject(b64_json="d29ybGQ=")],
        usage=Dict{String,Any}("total_tokens" => 100),
        raw=Dict{String,Any}("created" => 1713833628)
    )
    @test ir.created == 1713833628
    @test length(ir.data) == 2
    @test ir.usage["total_tokens"] == 100
end

@testset "image_data" begin
    @testset "extract base64 data" begin
        ir = ImageResponse(
            created=1,
            data=[ImageObject(b64_json="abc123"), ImageObject(b64_json="def456")],
            raw=Dict{String,Any}()
        )
        imgs = image_data(ir)
        @test length(imgs) == 2
        @test imgs[1] == "abc123"
        @test imgs[2] == "def456"
    end

    @testset "skip nil entries" begin
        ir = ImageResponse(
            created=1,
            data=[ImageObject(b64_json="abc123"), ImageObject()],
            raw=Dict{String,Any}()
        )
        imgs = image_data(ir)
        @test length(imgs) == 1
    end

    @testset "on ImageSuccess" begin
        ir = ImageResponse(
            created=1,
            data=[ImageObject(b64_json="xyz")],
            raw=Dict{String,Any}()
        )
        s = ImageSuccess(response=ir)
        @test s isa UniLM.LLMRequestResponse
        @test image_data(s) == ["xyz"]
    end
end

@testset "Result types" begin
    @testset "ImageSuccess" begin
        ir = ImageResponse(created=1, data=ImageObject[], raw=Dict{String,Any}())
        s = ImageSuccess(response=ir)
        @test s isa UniLM.LLMRequestResponse
    end

    @testset "ImageFailure" begin
        f = ImageFailure(response="error body", status=400)
        @test f isa UniLM.LLMRequestResponse
        @test f.response == "error body"
        @test f.status == 400
    end

    @testset "ImageCallError" begin
        e = ImageCallError(error="timeout")
        @test e isa UniLM.LLMRequestResponse
        @test e.error == "timeout"
        @test isnothing(e.status)

        e2 = ImageCallError(error="server error", status=503)
        @test e2.status == 503
    end
end

@testset "parse_image_response" begin
    function make_response(body::Dict; status=200)
        body_bytes = Vector{UInt8}(JSON.json(body))
        HTTP.Response(status, [], body_bytes)
    end

    @testset "basic response" begin
        body = Dict(
            "created" => 1713833628,
            "data" => [
                Dict("b64_json" => "aGVsbG8="),
                Dict("b64_json" => "d29ybGQ=")
            ],
            "usage" => Dict(
                "total_tokens" => 100,
                "input_tokens" => 50,
                "output_tokens" => 50
            )
        )
        resp = make_response(body)
        ir = UniLM.parse_image_response(resp)
        @test ir.created == 1713833628
        @test length(ir.data) == 2
        @test ir.data[1].b64_json == "aGVsbG8="
        @test ir.data[2].b64_json == "d29ybGQ="
        @test ir.usage["total_tokens"] == 100
    end

    @testset "response with revised_prompt" begin
        body = Dict(
            "created" => 1713833628,
            "data" => [
                Dict("b64_json" => "abc", "revised_prompt" => "A cute baby sea otter swimming")
            ]
        )
        resp = make_response(body)
        ir = UniLM.parse_image_response(resp)
        @test ir.data[1].revised_prompt == "A cute baby sea otter swimming"
    end

    @testset "empty data" begin
        body = Dict("created" => 1, "data" => [])
        resp = make_response(body)
        ir = UniLM.parse_image_response(resp)
        @test isempty(ir.data)
    end
end

@testset "save_image" begin
    @testset "saves decoded data to file" begin
        # "hello" base64-encoded is "aGVsbG8="
        tmpfile = tempname()
        try
            save_image("aGVsbG8=", tmpfile)
            @test isfile(tmpfile)
            @test read(tmpfile, String) == "hello"
        finally
            isfile(tmpfile) && rm(tmpfile)
        end
    end

    @testset "returns filepath" begin
        tmpfile = tempname()
        try
            result = save_image("dGVzdA==", tmpfile)
            @test result == tmpfile
        finally
            isfile(tmpfile) && rm(tmpfile)
        end
    end
end

@testset "generate_image error handling" begin
    @testset "generate_image(::ImageGeneration) with invalid API key" begin
        withenv("OPENAI_API_KEY" => "sk-invalid-test-key") do
            ig = ImageGeneration(prompt="test")
            result = generate_image(ig)
            @test result isa ImageCallError || result isa ImageFailure
        end
    end

    @testset "generate_image(prompt; kwargs...) convenience" begin
        withenv("OPENAI_API_KEY" => "sk-invalid-test-key") do
            result = generate_image("test prompt", size="1024x1024")
            @test result isa ImageCallError || result isa ImageFailure
        end
    end
end

@testset "Constants - images path" begin
    @test UniLM.IMAGES_GENERATIONS_PATH == "/v1/images/generations"
end

@testset "images.jl — config seam wiring" begin
    ig = ImageGeneration(prompt="probe", model="seam-probe-image", service=SeamProbe)
    @test _reached_seam(generate_image(ig; config=_TINY_DEADLINE), ImageCallError)
    @test _reached_seam(generate_image("probe"; model="seam-probe-image", service=SeamProbe,
                                       config=_TINY_DEADLINE), ImageCallError)

    imgpath = tempname() * ".png"
    write(imgpath, UInt8[0x89, 0x50, 0x4e, 0x47])   # PNG magic; content is irrelevant (never sent)
    try
        e = ImageEdit(image=imgpath, prompt="probe", model="seam-probe-image", service=SeamProbe)
        @test _reached_seam(edit_image(e; config=_TINY_DEADLINE), ImageCallError)
        @test _reached_seam(edit_image(imgpath, "probe"; model="seam-probe-image",
                                       service=SeamProbe, config=_TINY_DEADLINE), ImageCallError)
    finally
        rm(imgpath; force=true)
    end

    # retries kwarg is removed (hard cut, no shim)
    @test_throws MethodError generate_image(ig; retries=1)
    @test_throws MethodError edit_image(ImageEdit(image=imgpath, prompt="p", model="m", service=SeamProbe); retries=1)
end

using Sockets

# Local image-edit target whose base URL is chosen after the listener binds.
struct ImagesRetryProbe <: UniLM.ServiceEndpoint end
const _images_probe_base = Ref("")
UniLM._resolve_base_url(::Type{ImagesRetryProbe}) = _images_probe_base[]
UniLM.auth_header(::Type{ImagesRetryProbe}) =
    ["Authorization" => "Bearer t", "Content-Type" => "application/json"]
UniLM.provider_capabilities(::Type{ImagesRetryProbe}) = Set([:image_edits])

# Probe an ephemeral port, then serve on it; the close-then-rebind window can race.
function _images_retry_server(handler)
    for _ in 1:5
        tcp = Sockets.listen(Sockets.localhost, 0)
        port = Int(Sockets.getsockname(tcp)[2])
        close(tcp)
        try
            return HTTP.serve!(handler, "127.0.0.1", port; verbose=false), "http://127.0.0.1:$port"
        catch e
            e isa Base.IOError || rethrow()
        end
    end
    error("could not bind an ephemeral port for the image-edit retry fixture")
end

@testset "edit_image: a retried attempt sends a complete multipart body" begin
    # Same rewind gap as the file-upload path: the seam owns retries and disables
    # HTTP.jl's, so a single Form is consumed by attempt 1 and attempt 2 would put
    # an empty body on the wire. Every attempt must build a fresh Form.
    marker = "image-retry-marker-" * repeat("z", 256)
    imgpath = tempname() * ".png"
    write(imgpath, marker)
    seen = Vector{Int}()          # body length per attempt, in order
    complete = Ref(false)
    server, base = _images_retry_server(req -> begin
        body = String(copy(req.body))
        push!(seen, sizeof(body))
        if length(seen) == 1
            return HTTP.Response(503, ["Retry-After" => "0"], Vector{UInt8}("{}"))
        end
        complete[] = occursin(marker, body) && occursin("a prompt", body)
        return HTTP.Response(200, ["Content-Type" => "application/json"],
                             Vector{UInt8}(JSON.json(Dict("created" => 1,
                                                          "data" => [Dict("b64_json" => "aGVsbG8=")]))))
    end)
    _images_probe_base[] = base
    try
        cfg = UniLM.RequestConfig(max_attempts=2, total_deadline=Inf)
        r = edit_image(imgpath, "a prompt"; model="probe-edit", service=ImagesRetryProbe, config=cfg)
        @test length(seen) == 2                 # the 503 was actually retried
        @test seen[2] >= seen[1]                # attempt 2 is not a truncated replay
        @test complete[]                        # ...and carries the whole image + prompt
        @test r isa ImageSuccess
        @test image_data(r)[1] == "aGVsbG8="
    finally
        close(server)
        rm(imgpath; force=true)
    end
end

@testset "edit_image: the optional mask reaches the wire only when supplied" begin
    # The mask is optional and the body factory decides per attempt whether to add
    # its part, so both shapes have to be pinned: supplied, the part carries the
    # file's own name and bytes; omitted, no mask part is emitted at all.
    img_marker = "edit-image-marker-" * repeat("i", 64)
    mask_marker = "edit-mask-marker-" * repeat("m", 64)
    imgpath = tempname() * ".png"
    maskpath = tempname() * ".png"
    write(imgpath, img_marker)
    write(maskpath, mask_marker)
    bodies = Vector{String}()
    server, base = _images_retry_server(req -> begin
        push!(bodies, String(copy(req.body)))
        return HTTP.Response(200, ["Content-Type" => "application/json"],
                             Vector{UInt8}(JSON.json(Dict("created" => 1,
                                                          "data" => [Dict("b64_json" => "aGVsbG8=")]))))
    end)
    _images_probe_base[] = base
    try
        cfg = UniLM.RequestConfig(max_attempts=1, total_deadline=Inf)
        with_mask = edit_image(imgpath, "a prompt"; model="probe-edit", service=ImagesRetryProbe,
                               mask=maskpath, config=cfg)
        @test with_mask isa ImageSuccess
        @test occursin("name=\"mask\"", bodies[end])
        @test occursin("filename=\"$(basename(maskpath))\"", bodies[end])
        @test occursin(mask_marker, bodies[end])

        without_mask = edit_image(imgpath, "a prompt"; model="probe-edit",
                                  service=ImagesRetryProbe, config=cfg)
        @test without_mask isa ImageSuccess
        @test !occursin("name=\"mask\"", bodies[end])   # nothing mask-shaped slipped in
        @test occursin(img_marker, bodies[end])         # ...and the image still did
    finally
        close(server)
        rm(imgpath; force=true)
        rm(maskpath; force=true)
    end
end

# A user-defined endpoint that declares NO capabilities — the documented way to reach
# an OpenAI-compatible backend this package does not ship, and which therefore cannot
# say what it supports.
struct ImagesUndeclaredProbe <: UniLM.ServiceEndpoint
    base_url::String
end
UniLM._resolve_base_url(e::ImagesUndeclaredProbe) = e.base_url
UniLM.auth_header(::ImagesUndeclaredProbe) =
    ["Authorization" => "Bearer t", "Content-Type" => "application/json"]

@testset "edit_image validates capabilities only where they were declared" begin
    imgpath = tempname() * ".png"
    write(imgpath, UInt8[0x89, 0x50, 0x4e, 0x47])
    try
        # Declared and lacking :image_edits → refused before any I/O, naming the feature.
        err = try
            edit_image(ImageEdit(image=imgpath, prompt="p", model="m", service=AZUREServiceEndpoint))
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("Image Edits API is not supported", err.msg)

        # Undeclared → nothing to check against, so the call must reach the wire. It
        # used to die on a raw MethodError from provider_capabilities instead.
        hits = Ref(0)
        server, base = _images_retry_server(_ -> begin
            hits[] += 1
            HTTP.Response(200, ["Content-Type" => "application/json"],
                          Vector{UInt8}(JSON.json(Dict("created" => 1,
                                                       "data" => [Dict("b64_json" => "aGVsbG8=")]))))
        end)
        try
            r = edit_image(imgpath, "p"; model="m", service=ImagesUndeclaredProbe(base),
                           config=UniLM.RequestConfig(max_attempts=1, total_deadline=10.0))
            @test r isa ImageSuccess
            @test image_data(r)[1] == "aGVsbG8="
            @test hits[] == 1
        finally
            close(server)
        end
    finally
        rm(imgpath; force=true)
    end
end

using Base64

@testset "save_image decodes before it opens the destination" begin
    # `open(path, "w")` truncates on entry, so decoding inside the block meant a
    # malformed payload destroyed whatever already lived there and left a 0-byte
    # stub. A decode that cannot succeed must cost the caller nothing.
    path = tempname() * ".png"
    original = "PRECIOUS ORIGINAL BYTES"
    write(path, original)
    try
        @test_throws ArgumentError save_image("!!!not base64!!!", path)
        @test isfile(path)
        @test read(path, String) == original      # untouched, not a 0-byte stub

        # The happy path still writes, and still overwrites.
        payload = base64encode("new image bytes")
        @test save_image(payload, path) == path
        @test read(path, String) == "new image bytes"
    finally
        rm(path; force=true)
    end
end
