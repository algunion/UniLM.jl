@testset "ImageGeneration" begin
    @testset "minimal creation" begin
        ig = ImageGeneration(prompt="A cute robot")
        @test ig.model == ""  # resolved to "gpt-image-2.5-sunburst" during serialization
        @test JSON.parse(JSON.json(ig))["model"] == "gpt-image-2.5-sunburst"
        @test UniLM.default_image_model(OPENAIServiceEndpoint) == "gpt-image-2.5-sunburst"
        @test ig.prompt == "A cute robot"
        @test ig.service == UniLM.OPENAIServiceEndpoint
        @test isnothing(ig.n)
        @test isnothing(ig.size)
        @test isnothing(ig.quality)
        @test isnothing(ig.background)
        @test isnothing(ig.output_format)
        @test isnothing(ig.output_compression)
        @test isnothing(ig.safety_identifier)
        @test isnothing(ig.aspect_ratio)
        @test isnothing(ig.resolution)
    end

    @testset "full creation" begin
        ig = ImageGeneration(
            prompt="A logo",
            model="gpt-image-2.5-flare",
            n=2,
            size="1024x1024",
            quality="high",
            background="transparent",
            output_format="png",
            output_compression=nothing,
            safety_identifier="user_123"
        )
        @test ig.prompt == "A logo"
        @test ig.n == 2
        @test ig.size == "1024x1024"
        @test ig.quality == "high"
        @test ig.background == "transparent"
        @test ig.output_format == "png"
        @test ig.safety_identifier == "user_123"
    end

    @testset "JSON serialization" begin
        ig = ImageGeneration(prompt="test", quality="high", size="1024x1024")
        lowered = JSON.lower(ig)
        @test lowered[:model] == "gpt-image-2.5-sunburst"
        @test lowered[:prompt] == "test"
        @test lowered[:quality] == "high"
        @test lowered[:size] == "1024x1024"
        # nil fields excluded
        @test !haskey(lowered, :service)
        @test !haskey(lowered, :n)
        @test !haskey(lowered, :background)
        @test !haskey(lowered, :output_format)
        @test !haskey(lowered, :output_compression)
        @test !haskey(lowered, :safety_identifier)
        @test !haskey(lowered, :user)
    end

    @testset "JSON round-trip" begin
        ig = ImageGeneration(prompt="A sunset", n=2, size="1536x1024", background="opaque")
        json_str = JSON.json(ig)
        parsed = JSON.parse(json_str)
        @test parsed["model"] == "gpt-image-2.5-sunburst"
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

@testset "generate_image: a rejected key is an ImageFailure, from either form" begin
    rejected = "{\"error\":{\"message\":\"Incorrect API key provided.\",\"type\":\"invalid_request_error\"," *
               "\"param\":null,\"code\":\"invalid_api_key\"}}"
    one = UniLM.RequestConfig(max_attempts=1)
    results, seen = _with_scripted((_, _) -> _json(401, rejected)) do
        (generate_image(ImageGeneration(prompt="test", model="m", service=URLProbe); config=one),
         generate_image("test prompt"; model="m", service=URLProbe, size="1024x1024", config=one))
    end
    @test all(r -> r isa ImageFailure && r.status == 401 && r.response == rejected, results)
    @test [(s.method, s.target) for s in seen] == fill(("POST", "/v1/images/generations"), 2)
    @test [JSON.parse(s.body; dicttype=Dict{String,Any}) for s in seen] ==
          [Dict{String,Any}("model" => "m", "prompt" => "test"),
           Dict{String,Any}("model" => "m", "prompt" => "test prompt", "size" => "1024x1024")]
end

@testset "Constants - images path" begin
    @test UniLM.IMAGES_GENERATIONS_PATH == "/v1/images/generations"
end

@testset "images.jl — config seam wiring" begin
    ig = ImageGeneration(prompt="probe", model="seam-probe-image", service=SeamProbe)
    @test _seam_timeout(generate_image(ig; config=_TINY_DEADLINE), ImageCallError)
    @test _seam_timeout(generate_image("probe"; model="seam-probe-image", service=SeamProbe,
                                       config=_TINY_DEADLINE), ImageCallError)

    imgpath = tempname() * ".png"
    write(imgpath, UInt8[0x89, 0x50, 0x4e, 0x47])   # PNG magic; content is irrelevant (never sent)
    try
        e = ImageEdit(image=imgpath, prompt="probe", model="seam-probe-image", service=SeamProbe)
        @test _seam_timeout(edit_image(e; config=_TINY_DEADLINE), ImageCallError)
        @test _seam_timeout(edit_image(imgpath, "probe"; model="seam-probe-image",
                                       service=SeamProbe, config=_TINY_DEADLINE), ImageCallError)
    finally
        rm(imgpath; force=true)
    end

    # retries kwarg is removed (hard cut, no shim)
    @test_throws MethodError generate_image(ig; retries=1)
    @test_throws MethodError edit_image(ImageEdit(image=imgpath, prompt="p", model="m", service=SeamProbe); retries=1)
end

@testset "images.jl — URL-delivered images are decoded; failures carry no image data" begin
    body = Dict("created" => 1, "data" => [Dict("url" => "https://example.com/1.png", "revised_prompt" => "p"),
                                           Dict("b64_json" => "aGVsbG8=")])
    ir = UniLM.parse_image_response(HTTP.Response(200, [], Vector{UInt8}(JSON.json(body))))
    @test ir.data[1].url == "https://example.com/1.png" && isnothing(ir.data[1].b64_json)
    @test isnothing(ir.data[2].url) && isnothing(ImageObject().url)
    @test image_data(ImageSuccess(response=ir)) == ["https://example.com/1.png", "aGVsbG8="]
    # A call that did not succeed has no images: asking for them throws, as `text` does,
    # instead of answering with an empty list that reads as "zero images generated".
    @test_throws LLMResultError image_data(ImageFailure(response="{\"error\":{}}", status=400))
    @test_throws LLMResultError image_data(ImageCallError(error="connect refused"))
end

@testset "images.jl — failures keep the request id and the underlying exception" begin
    one = UniLM.RequestConfig(max_attempts=1, total_deadline=30.0)
    probe = ImageGeneration(prompt="p", model="m", service=URLProbe)
    f = _answered(() -> generate_image(probe; config=one), 400; headers=["x-request-id" => "req_img"])
    @test f isa ImageFailure && f.request_id == "req_img"
    # A 200 that is not an images envelope: the reply existed, so its id is kept too.
    c = _answered(() -> generate_image(probe; config=one), 200; body="{}",
                  headers=["x-request-id" => "req_img_200"])
    @test c isa ImageCallError
    @test c.request_id == "req_img_200"
    @test c.cause isa KeyError
    @test isnothing(ImageCallError(error="x").request_id) && isnothing(ImageCallError(error="x").cause)
end

@testset "images.jl — cancel: a cancelled token sends nothing, a mid-request cancel returns at once" begin
    cfg = UniLM.RequestConfig(max_attempts=3, total_deadline=60.0)
    probe = ImageGeneration(prompt="p", model="m", service=URLProbe)
    imgpath = tempname() * ".png"
    write(imgpath, UInt8[0x89, 0x50, 0x4e, 0x47])
    try
        tok = cancel!(CancelToken())
        calls = (() -> generate_image(probe; config=cfg, cancel=tok),
                 () -> generate_image("p"; model="m", service=URLProbe, config=cfg, cancel=tok),
                 () -> edit_image(ImageEdit(image=imgpath, prompt="p", model="m", service=URLProbe);
                                  config=cfg, cancel=tok),
                 () -> edit_image(imgpath, "p"; model="m", service=URLProbe, config=cfg, cancel=tok),
                 () -> with_cancel(() -> generate_image(probe; config=cfg), tok))   # ambient token
        results, seen = _with_scripted(() -> map(f -> f(), calls), (_, _) -> _json(200, "{}"))
        @test all(r -> r isa ImageCallError && r.cause isa UniLMCancelled, results)
        @test isempty(seen)
    finally
        rm(imgpath; force=true)
    end

    # The server would hold its reply for 10 s; the cancel lands 0.3 s into the exchange.
    arrived, release = Threads.Atomic{Bool}(false), Threads.Atomic{Bool}(false)
    hold = (_, _) -> (arrived[] = true; timedwait(() -> release[], 10.0); _json(200, "{}"))
    (r, after), seen = _with_scripted(hold) do
        tok = CancelToken()
        t = Threads.@spawn (generate_image(probe; config=cfg, cancel=tok), time_ns())
        timedwait(() -> arrived[], 25.0)
        sleep(0.3)
        cancelled_at = time_ns()
        cancel!(tok)
        finished = timedwait(() -> istaskdone(t), 25.0) === :ok
        release[] = true
        finished || return (nothing, Inf)
        res, done_at = fetch(t)
        (res, (done_at - cancelled_at) / 1e9)
    end
    @test r isa ImageCallError && r.cause isa UniLMCancelled && r.cause.source === :token
    @test after < 1.0
    @test length(seen) == 1   # the cancelled exchange is never retried
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
        # Attempt totals are not compared: each rebuild draws a fresh random
        # boundary. A truncated or empty replay still cannot reach the size of
        # the payload it must carry.
        @test seen[2] >= sizeof(marker)
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

@testset "edit_image refuses an input image or mask that is no file before anything is sent" begin
    # As upload_file, transcribe and translate refuse a path that is no file.
    png = tempname() * ".png"
    write(png, UInt8[0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a])
    try
        for (call, words) in ((() -> edit_image("/no/such/image.png", "p"; service=URLProbe, model="m"), "image not found"),
                              (() -> edit_image([png, "/no/such/b.png"], "p"; service=URLProbe, model="m"), "image not found"),
                              (() -> edit_image(png, "p"; mask="/no/such/mask.png", service=URLProbe, model="m"), "mask not found"))
            err, seen = _with_scripted((_, _) -> _json(200, "{}")) do
                try call(); nothing catch e; e end
            end
            @test err isa ArgumentError && occursin(words, err.msg) && isempty(seen)
        end
    finally
        rm(png; force=true)
    end
end

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

# ── OpenAI: safety_identifier replaces the deprecated user field ────────────────

_image_json(x) = JSON.parse(JSON.json(x); dicttype=Dict{String,Any})
_refusal(f) = try f(); nothing catch e; e end

@testset "OpenAI images — safety_identifier replaces the deprecated user field" begin
    @test _image_json(ImageGeneration(prompt="p", safety_identifier="u1")) ==
          Dict{String,Any}("model" => "gpt-image-2.5-sunburst", "prompt" => "p", "safety_identifier" => "u1")
    @test !hasfield(ImageGeneration, :user) && !hasfield(ImageEdit, :user)
    @test_throws MethodError ImageGeneration(prompt="p", user="u1")
    # OpenAI's safety guide: at most 128 characters, counted as characters, not bytes.
    @test _image_json(ImageGeneration(prompt="p", safety_identifier="é"^128))["safety_identifier"] == "é"^128
    # Longer is refused; so is "", which the API reads as no identifier at all.
    for bad in ("a"^129, "é"^129, "")
        err = _refusal(() -> JSON.json(ImageGeneration(prompt="p", safety_identifier=bad)))
        @test err isa ArgumentError
        @test occursin("OpenAI", err.msg) && occursin("safety_identifier", err.msg) && occursin("128", err.msg)
    end
end

@testset "OpenAI images — Gemini's sizing fields are refused, naming OpenAI and the field" begin
    for (field, value) in (:aspect_ratio => "16:9", :resolution => "2K")
        err = _refusal(() -> JSON.json(ImageGeneration(; prompt="p", field => value)))
        @test err isa ArgumentError
        @test occursin("OpenAI", err.msg) && occursin(String(field), err.msg) && occursin("size", err.msg)
        e = ImageEdit(; image="never-read.png", prompt="p", field => value)
        err = _refusal(() -> UniLM._check_image_request(OPENAIServiceEndpoint, e))
        @test err isa ArgumentError && occursin("OpenAI", err.msg) && occursin(String(field), err.msg)
    end
end

@testset "OpenAI images — a refused request sends nothing" begin
    path = tempname() * ".png"
    write(path, UInt8[0x89, 0x50, 0x4e, 0x47])
    long = "a"^129
    try
        calls = (() -> generate_image(ImageGeneration(prompt="p", model="m", service=URLProbe, safety_identifier=long)),
                 () -> generate_image("p"; model="m", service=URLProbe, aspect_ratio="16:9"),
                 () -> edit_image(ImageEdit(image=path, prompt="p", model="m", service=URLProbe, safety_identifier=long)),
                 () -> edit_image(path, "p"; model="m", service=URLProbe, resolution="2K"))
        thrown, seen = _with_scripted((_, _) -> _json(200, "{}")) do
            map(_refusal, calls)
        end
        @test all(e -> e isa ArgumentError, thrown)
        @test isempty(seen)
    finally
        rm(path; force=true)
    end
end

@testset "OpenAI image edits — safety_identifier is a form field; the default model is gpt-image-2.5-sunburst" begin
    path = tempname() * ".png"
    write(path, UInt8[0x89, 0x50, 0x4e, 0x47])
    ok = JSON.json(Dict("created" => 1, "data" => [Dict("b64_json" => "aGVsbG8=")]))
    try
        r, seen = _with_scripted((_, _) -> _json(200, ok)) do
            edit_image(path, "p"; model="m", service=URLProbe, safety_identifier="u1",
                       config=UniLM.RequestConfig(max_attempts=1))
        end
        @test r isa ImageSuccess
        @test occursin("name=\"safety_identifier\"\r\n\r\nu1\r\n", only(seen).body)
        @test !occursin("name=\"user\"", only(seen).body)
        url, _, body = withenv(() -> UniLM._image_request(OPENAIServiceEndpoint, ImageEdit(image=path, prompt="p")),
                               "OPENAI_API_KEY" => "sk-test")
        @test url == "https://api.openai.com/v1/images/edits"
        @test occursin("name=\"model\"\r\n\r\ngpt-image-2.5-sunburst\r\n", String(read(body.build())))
    finally
        rm(path; force=true)
    end
end

# ── Gemini (Nano Banana) through generateContent ────────────────────────────────

const _PNG_SIGNATURE = UInt8[0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]

_gemini_wire(r) = withenv(() -> UniLM._image_request(GEMINIServiceEndpoint, r), "GEMINI_API_KEY" => "test-key")

# The body of a text-to-image request for `prompt` with no image controls set.
_gemini_plain(prompt) = Dict{String,Any}(
    "contents" => Any[Dict{String,Any}("role" => "user", "parts" => Any[Dict{String,Any}("text" => prompt)])],
    "generationConfig" => Dict{String,Any}("responseModalities" => Any["IMAGE"]))

@testset "Gemini images — generateContent on gemini-nano-banana-2.1, image output requested" begin
    @test UniLM.default_image_model(GEMINIServiceEndpoint) == "gemini-nano-banana-2.1"
    ig = ImageGeneration(service=GEMINIServiceEndpoint, prompt="p")
    @test ig.model == ""
    url, headers, body = _gemini_wire(ig)
    @test url == "$(UniLM.GEMINI_NATIVE_BASE)/models/gemini-nano-banana-2.1:generateContent"
    @test Dict(headers) == Dict("x-goog-api-key" => "test-key", "Content-Type" => "application/json")
    @test JSON.parse(body; dicttype=Dict{String,Any}) == _gemini_plain("p")
    @test _image_json(ig) == _gemini_plain("p")       # JSON.json shows the body that is sent
    # Gemini's image controls, and the end-user id under Google's documented request label.
    sized = ImageGeneration(service=GEMINIServiceEndpoint, prompt="p", n=1, aspect_ratio="16:9",
                            resolution="2K", safety_identifier="user_42")
    @test _image_json(sized) == Dict{String,Any}(
        "contents" => _gemini_plain("p")["contents"],
        "generationConfig" => Dict{String,Any}("responseModalities" => Any["IMAGE"],
            "imageConfig" => Dict{String,Any}("aspectRatio" => "16:9", "imageSize" => "2K")),
        "labels" => Dict{String,Any}("safety_identifier" => "user_42"))
    # Gemini's label rule, not OpenAI's 128 characters: 63 lowercase letters, numerals, _ or -,
    # international ones included.
    intl = ImageGeneration(service=GEMINIServiceEndpoint, prompt="p", safety_identifier="usuário_日本-٤٢")
    @test _image_json(intl)["labels"] == Dict{String,Any}("safety_identifier" => "usuário_日本-٤٢")
    for bad in ("User_42", "a"^64)
        err = _refusal(() -> JSON.json(ImageGeneration(service=GEMINIServiceEndpoint, prompt="p", safety_identifier=bad)))
        @test err isa ArgumentError && occursin("Gemini", err.msg) && occursin("63", err.msg)
    end
    # The model is one path segment: separators in it cannot re-shape the URL.
    hostile = ImageGeneration(service=GEMINIServiceEndpoint, model=_HOSTILE_ID, prompt="p")
    @test first(_gemini_wire(hostile)) == "$(UniLM.GEMINI_NATIVE_BASE)/models/$(_HOSTILE_ENC):generateContent"
end

@testset "Gemini images — OpenAI-only fields are refused, naming Gemini and the field" begin
    for (field, value) in (:size => "1024x1024", :quality => "high", :background => "opaque",
                           :output_format => "png", :output_compression => 50, :moderation => "low")
        err = _refusal(() -> JSON.json(ImageGeneration(; service=GEMINIServiceEndpoint, prompt="p", field => value)))
        @test err isa ArgumentError && startswith(err.msg, "Gemini image generation does not take $field,")
    end
    err = _refusal(() -> JSON.json(ImageGeneration(service=GEMINIServiceEndpoint, prompt="p", size="1024x1024")))
    @test occursin("aspect_ratio", err.msg) && occursin("resolution", err.msg)   # the fix: Gemini's own controls
    for (field, value) in (:mask => "mask.png", :input_fidelity => "high", :size => "1024x1024",
                           :quality => "high", :background => "opaque", :output_format => "png")
        e = ImageEdit(; service=GEMINIServiceEndpoint, image="never-read.png", prompt="p", field => value)
        err = _refusal(() -> UniLM._check_image_request(GEMINIServiceEndpoint, e))
        @test err isa ArgumentError && startswith(err.msg, "Gemini image editing does not take $field,")
    end
    # Gemini returns one image per request: n=1 is that image, any other count has no Gemini form.
    for n in (0, 2, 10)
        err = _refusal(() -> JSON.json(ImageGeneration(service=GEMINIServiceEndpoint, prompt="p", n=n)))
        @test err isa ArgumentError && occursin("Gemini", err.msg) && occursin("n must be 1", err.msg)
    end
end

@testset "Gemini images — aspect_ratio and resolution take Gemini's documented values" begin
    config(; kw...) = _image_json(ImageGeneration(; service=GEMINIServiceEndpoint, prompt="p", kw...))["generationConfig"]["imageConfig"]
    for ratio in ("1:1", "1:4", "4:1", "1:8", "8:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9")
        @test config(aspect_ratio=ratio) == Dict{String,Any}("aspectRatio" => ratio)
    end
    for size in ("1K", "2K", "4K")
        @test config(resolution=size) == Dict{String,Any}("imageSize" => size)
    end
    for kw in ((aspect_ratio="16x9",), (aspect_ratio="16:10",), (aspect_ratio="",),
               (resolution="8K",), (resolution="1k",), (resolution="1024x1024",))
        err = _refusal(() -> config(; kw...))
        @test err isa ArgumentError && occursin("Gemini", err.msg) && occursin(String(only(keys(kw))), err.msg)
    end
    # Nano Banana 2.1 renders at 1K, 2K or 4K; Gemini documents 512 for an earlier model.
    err = _refusal(() -> config(resolution="512"))
    @test err isa ArgumentError && occursin("gemini-nano-banana-2.1", err.msg) && occursin("512", err.msg)
    @test err == _refusal(() -> config(model="gemini-nano-banana-2.1", resolution="512"))
    # A model without its own row (a near-miss name, a later model) takes every size Gemini
    # documents: the server's answer is the check.
    for model in ("gemini-nano-banana-2.1-preview", "gemini-nano-banana-3")
        @test config(model=model, resolution="512") == Dict{String,Any}("imageSize" => "512")
    end
end

@testset "Gemini images — an inlineData image part is an ImageObject; thought parts are not" begin
    png = base64encode(_PNG_SIGNATURE)
    reply = Dict(
        "candidates" => [Dict("content" => Dict("role" => "model", "parts" => [
                Dict("text" => "Sketching a circle.", "thought" => true),
                Dict("inlineData" => Dict("mimeType" => "image/png", "data" => base64encode(UInt8[1, 2, 3])),
                     "thought" => true),
                Dict("text" => "A red circle "),
                Dict("text" => "on white."),
                Dict("inlineData" => Dict("mimeType" => "image/png", "data" => png))]),
            "finishReason" => "STOP")],
        "usageMetadata" => Dict("promptTokenCount" => 9, "candidatesTokenCount" => 1120,
                                "thoughtsTokenCount" => 40, "totalTokenCount" => 1169),
        "modelVersion" => "gemini-nano-banana-2.1")
    ir = UniLM._image_reply(GEMINIServiceEndpoint, HTTP.Response(200, [], Vector{UInt8}(JSON.json(reply))))
    @test ir.data == [ImageObject(b64_json=png, revised_prompt="A red circle on white.")]
    @test ir.raw == _image_json(reply)
    @test abs(ir.created - time()) < 60          # Gemini reports no creation time: when it arrived
    s = ImageSuccess(response=ir)
    @test base64decode(only(image_data(s))) == _PNG_SIGNATURE
    # usageMetadata as the image result's usage: thoughts are output, as on Gemini Chat.
    @test token_usage(s) == TokenUsage(prompt_tokens=9, completion_tokens=1160, total_tokens=1169, reasoning_tokens=40)
    @test token_usage(s) == UniLM._gemini_usage(reply["usageMetadata"])
    # No text: no revised prompt. Two images: two objects, in order. No usage: none reported.
    two = Dict("candidates" => [Dict("finishReason" => "STOP", "content" => Dict("parts" => [
        Dict("inlineData" => Dict("mimeType" => "image/png", "data" => png)),
        Dict("inlineData" => Dict("mimeType" => "image/jpeg", "data" => "/9j/"))]))])
    ir2 = UniLM._image_reply(GEMINIServiceEndpoint, HTTP.Response(200, [], Vector{UInt8}(JSON.json(two))))
    @test ir2.data == [ImageObject(b64_json=png), ImageObject(b64_json="/9j/")]
    @test isnothing(ir2.usage) && token_usage(ImageSuccess(response=ir2)) == TokenUsage()
end

@testset "Gemini images — a reply without an image is an error naming why" begin
    why(d) = sprint(showerror, _refusal(() ->
        UniLM._image_reply(GEMINIServiceEndpoint, HTTP.Response(200, [], Vector{UInt8}(JSON.json(d))))))
    filtered = Dict("candidates" => [Dict("content" => Dict("parts" => [Dict("text" => "I can't draw that.")]),
        "finishReason" => "IMAGE_SAFETY", "finishMessage" => "The image was blocked.")])
    m = why(filtered)
    @test occursin("no image", m) && occursin("IMAGE_SAFETY", m) && occursin("The image was blocked.", m) &&
          occursin("I can't draw that.", m)
    @test occursin("NO_IMAGE", why(Dict("candidates" => [Dict("finishReason" => "NO_IMAGE")])))
    # Only a thought image: the final image never came.
    thought_only = Dict("candidates" => [Dict("finishReason" => "MAX_TOKENS", "content" => Dict("parts" => [
        Dict("inlineData" => Dict("mimeType" => "image/png", "data" => "AA=="), "thought" => true)]))])
    @test occursin("MAX_TOKENS", why(thought_only))
    # No candidate: the prompt was blocked, and Gemini's feedback says why.
    @test occursin("PROHIBITED_CONTENT", why(Dict("promptFeedback" => Dict("blockReason" => "PROHIBITED_CONTENT"))))
end

@testset "Gemini image edits — each input image is an inline part ahead of the prompt" begin
    paths = [tempname() * ".png", tempname() * ".jpg"]
    jpeg = UInt8[0xff, 0xd8, 0xff, 0xe0]
    write(paths[1], _PNG_SIGNATURE)
    write(paths[2], jpeg)
    try
        e = ImageEdit(service=GEMINIServiceEndpoint, image=paths, prompt="make the circle blue", aspect_ratio="1:1")
        url, headers, body = _gemini_wire(e)
        @test url == "$(UniLM.GEMINI_NATIVE_BASE)/models/gemini-nano-banana-2.1:generateContent"
        @test Dict(headers) == Dict("x-goog-api-key" => "test-key", "Content-Type" => "application/json")
        @test JSON.parse(body; dicttype=Dict{String,Any}) == Dict{String,Any}(
            "contents" => Any[Dict{String,Any}("role" => "user", "parts" => Any[
                Dict{String,Any}("inlineData" => Dict{String,Any}("mimeType" => "image/png", "data" => base64encode(_PNG_SIGNATURE))),
                Dict{String,Any}("inlineData" => Dict{String,Any}("mimeType" => "image/jpeg", "data" => base64encode(jpeg))),
                Dict{String,Any}("text" => "make the circle blue")])],
            "generationConfig" => Dict{String,Any}("responseModalities" => Any["IMAGE"],
                "imageConfig" => Dict{String,Any}("aspectRatio" => "1:1")))
        # The format is read from the file's bytes, never from its name.
        write(paths[2], "not an image")
        err = _refusal(() -> _gemini_wire(e))
        @test err isa ArgumentError && occursin("not PNG, JPEG, GIF or WebP", err.msg)
    finally
        rm.(paths; force=true)
    end
end

# Gemini's image wire against a local server: GEMINIServiceEndpoint's own checks, request
# and decoder, with only the host replaced — the redirect recorded_answers.jl uses for chat.
struct GeminiImagesHere <: UniLM.ServiceEndpoint end
UniLM._check_image_request(::Type{GeminiImagesHere}, r::Union{ImageGeneration,ImageEdit}) =
    UniLM._check_image_request(GEMINIServiceEndpoint, r)
for T in (ImageGeneration, ImageEdit)
    @eval function UniLM._image_request(::Type{GeminiImagesHere}, r::$T)
        url, headers, body = UniLM._image_request(GEMINIServiceEndpoint, r)
        (replace(url, r"^https://[^/]+" => _url_probe_base[]), headers, body)
    end
end
UniLM._image_reply(::Type{GeminiImagesHere}, resp::HTTP.Response) = UniLM._image_reply(GEMINIServiceEndpoint, resp)

@testset "Gemini images — generate_image and edit_image over the wire" begin
    png = base64encode(_PNG_SIGNATURE)
    ok = JSON.json(Dict("candidates" => [Dict("finishReason" => "STOP", "content" => Dict("parts" => [
        Dict("inlineData" => Dict("mimeType" => "image/png", "data" => png))]))]))
    one = UniLM.RequestConfig(max_attempts=1)
    path = tempname() * ".png"
    write(path, _PNG_SIGNATURE)
    try
        withenv("GEMINI_API_KEY" => "test-key") do
            (g, e), seen = _with_scripted((_, _) -> _json(200, ok)) do
                (generate_image("a red circle"; service=GeminiImagesHere, config=one),
                 edit_image(path, "make the circle blue"; service=GeminiImagesHere, config=one))
            end
            @test g isa ImageSuccess && image_data(g) == [png]
            @test e isa ImageSuccess && image_data(e) == [png]
            @test [s.target for s in seen] == fill("/v1beta/models/gemini-nano-banana-2.1:generateContent", 2)
            @test JSON.parse(seen[1].body; dicttype=Dict{String,Any}) == _gemini_plain("a red circle")
            @test JSON.parse(seen[2].body; dicttype=Dict{String,Any})["contents"] == Any[Dict{String,Any}(
                "role" => "user", "parts" => Any[
                    Dict{String,Any}("inlineData" => Dict{String,Any}("mimeType" => "image/png", "data" => png)),
                    Dict{String,Any}("text" => "make the circle blue")])]
            # A 200 without an image is a call error naming the finish reason.
            blocked = JSON.json(Dict("candidates" => [Dict("finishReason" => "IMAGE_SAFETY")]))
            r, _ = _with_scripted((_, _) -> _json(200, blocked)) do
                generate_image("a red circle"; service=GeminiImagesHere, config=one)
            end
            @test r isa ImageCallError && occursin("IMAGE_SAFETY", r.error) && isnothing(r.status)
            # A missing input image is refused before anything is sent, as on OpenAI.
            missing_input, none = _with_scripted((_, _) -> _json(200, ok)) do
                _refusal(() -> edit_image("/no/such/image.png", "p"; service=GeminiImagesHere, config=one))
            end
            @test missing_input isa ArgumentError && occursin("image not found", missing_input.msg) && isempty(none)
            # Refused before any request: OpenAI-only fields, a size Nano Banana 2.1 does not render.
            thrown, none = _with_scripted((_, _) -> _json(200, ok)) do
                map(_refusal, (() -> generate_image("p"; service=GeminiImagesHere, quality="high"),
                               () -> generate_image("p"; service=GeminiImagesHere, resolution="512"),
                               () -> edit_image(path, "p"; service=GeminiImagesHere, mask=path),
                               () -> edit_image(path, "p"; service=GeminiImagesHere, input_fidelity="high")))
            end
            @test all(x -> x isa ArgumentError, thrown) && isempty(none)
        end
    finally
        rm(path; force=true)
    end
end

@testset "an image edit with no input image is refused before any request, on both wires" begin
    # Sent anyway, Gemini would read the prompt alone as a text-to-image request.
    for service in (URLProbe, GeminiImagesHere)
        thrown, seen = withenv("GEMINI_API_KEY" => "test-key") do
            _with_scripted((_, _) -> _json(200, "{}")) do
                map(_refusal, (() -> edit_image(ImageEdit(service=service, image=String[], prompt="make it blue")),
                               () -> edit_image(String[], "make it blue"; service=service, model="m")))
            end
        end
        @test all(e -> e isa ArgumentError && occursin("at least one input image", e.msg), thrown)
        @test isempty(seen)
    end
end
