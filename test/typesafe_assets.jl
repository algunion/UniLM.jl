# test/typesafe_assets.jl — TypeSafe images, Box questions and long context, zero-spend.
#
# The images under test/fixtures/typesafe/images/ were accepted by the live service,
# and the fixtures 40–47 are verbatim captures of the live witnesses of 2026-10-20 (the
# JSON part as sent, the response body byte for byte); 30, 32 and 50–56 are error
# bodies the live service sent on 2026-10-19. Local mocks only: nothing here reaches
# a network.
#
# Requires test/typesafe.jl (its mock server and fixture readers) to be included first.

using Base64

const _TA_IMAGES = joinpath(@__DIR__, "fixtures", "typesafe", "images")
# A white 512×512 scene holding a red rectangle (left 0.20, right 0.55, top 0.30,
# bottom 0.70) and a blue disc, as a PNG and as a JPEG; and two 300×300 swatches.
const _TA_SCENE = ImageAttachment(joinpath(_TA_IMAGES, "scene-512.png"))
const _TA_SCENE_JPEG = ImageAttachment(joinpath(_TA_IMAGES, "scene-512.jpg"))
const _TA_RED = ImageAttachment(joinpath(_TA_IMAGES, "red-300.png"))
const _TA_BLUE = ImageAttachment(joinpath(_TA_IMAGES, "blue-300.png"))

_ta_digest(img::ImageAttachment) = bytes2hex(UniLM.SHA.sha256(img.data))
_ta_ref(img::ImageAttachment) = "<image:" * first(_ta_digest(img), 32) * ">"
_ta_wire(r::SystemOneRequest) = UniLM.encode_system_one(UniLM.TypeSafeProtocol(), TYPESAFEServiceEndpoint, r)
_ta_caught(f::Function) = try f(); nothing catch e; e end

# Image bytes made to order. The header checks read chunk and segment structure, not
# pixels or checksums, so a PNG here is its chunks with zero CRCs and a JPEG its markers.
_ta_be32(x::Integer) = reinterpret(UInt8, [hton(UInt32(x))])
_ta_chunk(kind::String, data::Vector{UInt8}=UInt8[]) = vcat(_ta_be32(length(data)), codeunits(kind), data, zeros(UInt8, 4))
_ta_png_bytes(w::Integer, h::Integer; before::Vector{UInt8}=UInt8[], idat::Bool=true, pad::Integer=0) = vcat(
    UInt8[0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a],
    _ta_chunk("IHDR", vcat(_ta_be32(w), _ta_be32(h), UInt8[0x08, 0x02, 0x00, 0x00, 0x00])),
    before,
    idat ? _ta_chunk("IDAT", UInt8[0x78, 0x01, 0x01, 0x00]) : UInt8[],
    pad > 0 ? _ta_chunk("prVt", zeros(UInt8, pad)) : UInt8[],
    _ta_chunk("IEND"))
_ta_png(w::Integer, h::Integer; kw...) = ImageAttachment(_ta_png_bytes(w, h; kw...))
_ta_jpeg(w::Integer, h::Integer; sof::UInt8=0xc0) = ImageAttachment(UInt8[
    0xff, 0xd8,
    0xff, 0xe0, 0x00, 0x10, 0x4a, 0x46, 0x49, 0x46, 0x00, 0x01, 0x01, 0x00, 0x00, 0x01, 0x00, 0x01, 0x00, 0x00,
    0xff, sof, 0x00, 0x11, 0x08, UInt8(h >> 8), UInt8(h & 0xff), UInt8(w >> 8), UInt8(w & 0xff),
    0x03, 0x01, 0x22, 0x00, 0x02, 0x11, 0x01, 0x03, 0x11, 0x01,
    0xff, 0xda, 0x00, 0x08, 0x01, 0x01, 0x00, 0x00, 0x3f, 0x00, 0x12, 0x34,
    0xff, 0xd9])
const _TA_GIF = ImageAttachment(vcat(codeunits("GIF89a"), UInt8[0x2c, 0x01, 0x2c, 0x01, 0x00, 0x00, 0x00]))
const _TA_WEBP = ImageAttachment(vcat(codeunits("RIFF"), UInt8[0x1a, 0x00, 0x00, 0x00], codeunits("WEBPVP8 "),
                                      zeros(UInt8, 14)))

"The parts of a recorded multipart request, as HTTP.jl parses them."
_ta_form(seen::AbstractDict) = HTTP.parse_multipart_form(seen["headers"]["content-type"], Vector{UInt8}(seen["body"]))

# An image held by a struct, which no container of a request reaches.
struct _TAHolder
    img::ImageAttachment
end

@testset "TypeSafe images — sizes are read from the PNG and JPEG headers, and what TypeSafe rejects is refused" begin
    @test UniLM._png_size(_TA_SCENE.data) == (512, 512)
    @test UniLM._png_size(_TA_RED.data) == (300, 300)
    @test UniLM._jpeg_size(_TA_SCENE_JPEG.data) == (512, 512)       # past its JFIF and Exif segments
    @test UniLM._png_size(_ta_png(1568, 280).data) == (1568, 280)
    @test UniLM._jpeg_size(_ta_jpeg(300, 1200).data) == (300, 1200)
    @test UniLM._jpeg_size(_ta_jpeg(640, 480; sof=0xc2).data) == (640, 480)   # progressive
    check(img) = UniLM._typesafe_check_image(img, 1)
    # Each side 280 to 1568 pixels, both ends included.
    for (w, h) in ((280, 280), (1568, 1568), (280, 1568), (1568, 280))
        @test check(_ta_png(w, h)) === nothing && check(_ta_jpeg(w, h)) === nothing
    end
    for (w, h) in ((279, 300), (300, 279), (1569, 800), (800, 1569), (2000, 200))
        for img in (_ta_png(w, h), _ta_jpeg(w, h))
            e = _ta_caught(() -> check(img))
            @test e isa ArgumentError && contains(e.msg, "TypeSafe") && contains(e.msg, "280 to 1568") &&
                  contains(e.msg, "$(w)×$(h)")
        end
    end
    # PNG and JPEG only.
    for img in (_TA_GIF, _TA_WEBP)
        e = _ta_caught(() -> check(img))
        @test e isa ArgumentError && contains(e.msg, "PNG and JPEG images only") && contains(e.msg, img.mime)
    end
    # A still image only: an animation control chunk before the image data is an animated PNG.
    apng = _ta_png(300, 300; before=_ta_chunk("acTL", zeros(UInt8, 8)))
    e = _ta_caught(() -> check(apng))
    @test e isa ArgumentError && contains(e.msg, "animated PNG")
    # A file that is cut short, or whose header cannot be read.
    half = ImageAttachment(_TA_RED.data[1:div(end, 2)])
    for (img, needle) in ((half, "cut short"),
                          (ImageAttachment(_TA_RED.data[1:20]), "header cannot be read"),
                          (_ta_png(300, 300; idat=false), "holds no image data"),
                          (ImageAttachment(_TA_SCENE_JPEG.data[1:30]), "cut short"),
                          (ImageAttachment(UInt8[0xff, 0xd8, 0xff, 0xda, 0x00, 0x02, 0xff, 0xd9]), "frame header is missing"),
                          (ImageAttachment(UInt8[0xff, 0xd8, 0xff, 0xe0, 0x00, 0x10, 0x00]), "cut short"))
        e = _ta_caught(() -> check(img))
        @test e isa ArgumentError && contains(e.msg, "TypeSafe cannot take") && contains(e.msg, needle)
    end
    # A JPEG ends with its end-of-image marker (FF D9), as the one the live service accepted
    # does: one cut short after its frame header, or carrying bytes after the image, is refused.
    jpg = _TA_SCENE_JPEG.data
    @test jpg[end-1:end] == [0xff, 0xd9]
    sos = findfirst(i -> jpg[i] == 0xff && jpg[i+1] == 0xda, 1:length(jpg)-1)
    for img in (ImageAttachment(jpg[1:sos+20]), ImageAttachment(jpg[1:end-1]), ImageAttachment([jpg; 0x00; 0x00]))
        e = _ta_caught(() -> check(img))
        @test e isa ArgumentError && contains(e.msg, "TypeSafe cannot take") && contains(e.msg, "end-of-image marker")
    end
    # At most 16 MiB a file, exactly.
    base = length(_ta_png_bytes(300, 300))
    at_limit = _ta_png(300, 300; pad=16 * 1024 * 1024 - base - 12)
    @test length(at_limit.data) == 16 * 1024 * 1024 && check(at_limit) === nothing
    over = _ta_png(300, 300; pad=16 * 1024 * 1024 - base - 11)
    e = _ta_caught(() -> check(over))
    @test e isa ArgumentError && contains(e.msg, "16 MiB") && contains(e.msg, "16777217 bytes")
end

@testset "TypeSafe images — each distinct image is one part, and every place it stood holds its reference" begin
    req = SystemOneRequest((photo = _TA_SCENE, note = "x", again = [_TA_SCENE]),
        ["match" => choice("Which sample?", (red_sample = _TA_RED, blue_sample = _TA_BLUE, neither = nothing)),
         "closest" => score("Closest?", [_TA_BLUE, (summary = "red", example = _TA_RED)]),
         "same" => noul(_TA_SCENE; yes = _TA_RED)]; model = "jev-latest")
    w = _ta_wire(req)
    sent = JSON.parse(w.body)
    @test collect(keys(sent)) == ["state", "model", "questions", "assets"]   # `assets` last
    scene, red, blue = _ta_ref(_TA_SCENE), _ta_ref(_TA_RED), _ta_ref(_TA_BLUE)
    @test occursin(r"^<image:[0-9a-f]{32}>$", scene)
    @test sent["state"] == Dict("photo" => scene, "note" => "x", "again" => [scene])
    @test sent["questions"]["match"]["criteria"] == Dict("red_sample" => red, "blue_sample" => blue, "neither" => nothing)
    @test sent["questions"]["closest"]["criteria"] == [blue, Dict("summary" => "red", "example" => red)]
    @test sent["questions"]["same"]["instructions"] == scene && sent["questions"]["same"]["criteria"]["true"] == red
    # The assets in first-occurrence order, one part each, named asset-0, asset-1, …
    @test collect(keys(sent["assets"])) == [scene, red, blue]
    @test [a["part"] for a in values(sent["assets"])] == ["asset-0", "asset-1", "asset-2"]
    @test all(a -> a["type"] == "image" && length(a) == 2, values(sent["assets"]))
    @test w.parts == ["asset-0" => _TA_SCENE, "asset-1" => _TA_RED, "asset-2" => _TA_BLUE]
    # Keyed on the digests of the parts, never on their bytes, after the JSON part as sent.
    @test w.key_extra == "asset-0 $(_ta_digest(_TA_SCENE))\nasset-1 $(_ta_digest(_TA_RED))\nasset-2 $(_ta_digest(_TA_BLUE))"
    @test w.record_body == w.body
    @test w.headers == ["TypeSafe-Danger-Zone-Images" => "true"]
    @test w.path == "/v1/systemone" && w.method == "POST"

    # The same bytes in two attachments are one image.
    twin = ImageAttachment(copy(_TA_RED.data))
    w2 = _ta_wire(SystemOneRequest([_TA_RED, twin], "same" => noul("Are the two images the same colour?"); model="m"))
    @test length(w2.parts) == 1 && JSON.parse(w2.body)["state"] == [red, red]

    # The captured requests are rebuilt byte for byte.
    rebuilt = [
        "40-img-png-state" => SystemOneRequest(_TA_SCENE,
            ["colour" => choice("What colour is the rectangle in the image?", ["red", "green", "blue"]),
             "red" => noul("Does the image contain a red shape?")]; model="jev-latest"),
        "41-img-jpeg-state" => SystemOneRequest((photo = _TA_SCENE_JPEG, note = "A drawing of two shapes."),
            ["disc" => noul("Is there a blue disc in `photo`?")]; model="jev-latest"),
        "42-img-same-image-twice" => SystemOneRequest([_TA_RED, _TA_RED],
            ["same" => noul("Are the two images the same colour?")]; model="jev-latest"),
        "43-img-criteria" => SystemOneRequest(_TA_SCENE,
            ["match" => choice("Which sample has the same colour as the rectangle in the state image?",
                               (red_sample = _TA_RED, blue_sample = _TA_BLUE)),
             "closest" => score("Which level's sample is closest in colour to the rectangle in the state image?",
                                [_TA_BLUE, _TA_RED])]; model="jev-latest"),
        "44-box-jev-1.14.0" => SystemOneRequest(_TA_SCENE,
            ["rect" => box("the red rectangle"), "disc" => box("the blue disc")]; model="jev-1.14.0")]
    for (name, r) in rebuilt
        @test _ta_wire(r).body == _ts_fixture("$(name).request.json")
    end
    @test length(_ta_wire(last(rebuilt[3])).parts) == 1

    # An image no container of the request reaches cannot be made an asset: refused, not sent as text.
    e = _ta_caught(() -> _ta_wire(SystemOneRequest([_TA_RED, (held = _TAHolder(_TA_BLUE),)], noul("q?"); model="m")))
    @test e isa ArgumentError && contains(e.msg, "TypeSafe cannot send") && contains(e.msg, "NamedTuple")
end

@testset "TypeSafe images — a request without images, Box or options is sent and keyed as it always was" begin
    p = UniLM.TypeSafeProtocol()
    for r in (_ts_request_02(),
              SystemOneRequest(Dict("b" => [1, 2], "a" => (x = "y",)), ["c" => choice(["x", "y"])]; model="jev-1.13.0"),
              SystemOneRequest("s", "r" => RawQuestion(Dict("type" => "rank", "items" => ["a"])); model="m"),
              SystemOneRequest("s", "q" => noul("Is it?"); model="m", options=(long_context = false,)))
        w = _ta_wire(r)
        @test w.body == UniLM._system_one_body(UniLM._system_one_json(r, UniLM._system_one_model(p, TYPESAFEServiceEndpoint, r)))
        @test isempty(w.parts) && isempty(w.headers) && w.key_extra == ""
        @test !contains(w.body, "assets")
    end
    # A Box question locates something in an image; with none in the request there is nothing
    # to locate in, and it is refused before anything is sent.
    e = _ta_caught(() -> _ta_wire(SystemOneRequest("A white page with a red rectangle.",
                                                   "rect" => box("the red rectangle"); model="jev-1.14.0")))
    @test e isa ArgumentError && contains(e.msg, "carries none")
end

@testset "TypeSafe long context — one option, a header, and a key of its own" begin
    q = "fox" => noul("Is the text about a fox?")
    on = _ta_wire(SystemOneRequest("text", q; model="m", options=(long_context = true,)))
    off = _ta_wire(SystemOneRequest("text", q; model="m"))
    @test on.headers == ["TypeSafe-Danger-Zone-Long-Context" => "true"] && on.key_extra == "long_context=true"
    @test on.body == off.body && isempty(off.headers) && off.key_extra == ""
    both = _ta_wire(SystemOneRequest(_TA_RED, q; model="m", options=(long_context = true,)))
    @test both.headers == ["TypeSafe-Danger-Zone-Images" => "true", "TypeSafe-Danger-Zone-Long-Context" => "true"]
    @test both.key_extra == "long_context=true\nasset-0 $(_ta_digest(_TA_RED))"
    # Any other option, or a long_context that is not a Bool, is refused by name.
    for (options, needle) in (((confg = 1,), "`confg` (accepted: long_context)"),
                              ((a = 1, b = 2), "options `a`, `b` (accepted: long_context)"),
                              ((long_context = 1,), "`true` or `false`; got 1"),
                              ((long_context = "true",), "`true` or `false`; got \"true\""))
        e = _ta_caught(() -> _ta_wire(SystemOneRequest("text", q; model="m", options)))
        @test e isa ArgumentError && contains(e.msg, "TypeSafe") && contains(e.msg, needle)
    end
end

# A value nested `n` levels deep: `n` arrays around a string.
_ta_nest(n::Int) = foldl((a, _) -> Any[a], 1:n; init="x")

@testset "TypeSafe request rules — nesting and body size are checked before anything is sent" begin
    q = "q" => noul("Is it?")
    # 64 levels in the state pass — the state object is the first — and 65 do not.
    @test _ta_wire(SystemOneRequest(Dict("n" => _ta_nest(63)), q; model="m")) isa UniLM.SystemOneWire
    for (r, where) in ((SystemOneRequest(Dict("n" => _ta_nest(64)), q; model="m"), "the state"),
                       (SystemOneRequest(_ta_nest(65), q; model="m"), "the state"),
                       (SystemOneRequest("s", "c" => choice(_ta_nest(65), ["a"]); model="m"), "the instructions of \"c\""),
                       (SystemOneRequest("s", "c" => choice(["a" => _ta_nest(64)]); model="m"), "the criteria of \"c\""),
                       (SystemOneRequest("s", "r" => score([_ta_nest(64)]); model="m"), "the criteria of \"r\""),
                       (SystemOneRequest("s", "n" => noul("q?"; yes=_ta_nest(64)); model="m"), "the criteria of \"n\""),
                       (SystemOneRequest(_TA_RED, "b" => box(_ta_nest(65)); model="m"), "the instructions of \"b\""),
                       (SystemOneRequest("s", "x" => RawQuestion(Dict("type" => "rank", "criteria" => _ta_nest(65))); model="m"),
                        "the criteria of \"x\""))
        e = _ta_caught(() -> _ta_wire(r))
        @test e isa ArgumentError && contains(e.msg, "TypeSafe accepts at most 64 levels") && contains(e.msg, where)
    end
    @test _ta_wire(SystemOneRequest("s", "c" => choice(["a" => _ta_nest(63)]); model="m")) isa UniLM.SystemOneWire
    # A cyclic value stops at the limit instead of recursing forever.
    cyclic = Dict{String,Any}(); cyclic["self"] = cyclic
    @test _ta_caught(() -> _ta_wire(SystemOneRequest(cyclic, q; model="m"))) isa ArgumentError

    # A body of 32 MiB passes, one byte more does not.
    limit = 32 * 1024 * 1024
    overhead = sizeof(_ta_wire(SystemOneRequest("", q; model="m")).body)
    @test sizeof(_ta_wire(SystemOneRequest("x"^(limit - overhead), q; model="m")).body) == limit
    e = _ta_caught(() -> _ta_wire(SystemOneRequest("x"^(limit - overhead + 1), q; model="m")))
    @test e isa ArgumentError && contains(e.msg, "32 MiB") && contains(e.msg, "$(limit + 1) bytes") &&
          contains(e.msg, "Shorten the state")
    # A multipart body counts its framing: two images of 16 MiB each do not fit.
    base = length(_ta_png_bytes(300, 300))
    full(i) = _ta_png(300, 300; before=_ta_chunk("prVu", UInt8[i]), pad=16 * 1024 * 1024 - base - 25)
    a, b = full(0x01), full(0x02)
    @test length(a.data) == length(b.data) == 16 * 1024 * 1024
    e = _ta_caught(() -> _ta_wire(SystemOneRequest([a, b], q; model="m")))
    @test e isa ArgumentError && contains(e.msg, "32 MiB") && contains(e.msg, "Send fewer or smaller images")
    # The size measured is the size sent.
    w = _ta_wire(SystemOneRequest([_TA_SCENE, _TA_RED], q; model="m"))
    @test UniLM._typesafe_body_size(w) == length(read(UniLM._system_one_form(w))) > sizeof(w.body) + 2347 + 685
end

@testset "TypeSafe images — ask sends one multipart request: the JSON part first, then the image parts" begin
    req = SystemOneRequest(_TA_SCENE,
        ["match" => choice("Which sample has the same colour as the rectangle in the state image?",
                           (red_sample = _TA_RED, blue_sample = _TA_BLUE)),
         "closest" => score("Which level's sample is closest in colour to the rectangle in the state image?",
                            [_TA_BLUE, _TA_RED])]; model="jev-latest")
    r, seen = _ts_mock(; status=200, body=_ts_fixture("43-img-criteria.response.json"),
                       headers=["Content-Type" => "application/json", "x-typesafe-request-id" => "req_img"]) do
        ask(req; config=_TS_MOCK_CFG)
    end
    @test r isa SystemOneSuccess && r.response.request_id == "req_img" && length(seen) == 1
    s = only(seen)
    @test s["method"] == "POST" && s["target"] == "/v1/systemone"
    @test startswith(s["headers"]["content-type"], "multipart/form-data; boundary=")
    @test s["headers"]["typesafe-danger-zone-images"] == "true"
    @test !haskey(s["headers"], "typesafe-danger-zone-long-context")
    @test startswith(s["headers"]["authorization"], "Bearer ")
    form = _ta_form(s)
    @test [p.name for p in form] == ["request", "asset-0", "asset-1", "asset-2"]
    @test form[1].filename === nothing && form[1].contenttype == "application/json"
    @test String(read(form[1].data)) == _ts_fixture("43-img-criteria.request.json")
    @test [p.filename for p in form[2:end]] == ["asset-0", "asset-1", "asset-2"]
    @test all(p -> p.contenttype == "image/png", form[2:end])
    @test [read(p.data) for p in form[2:end]] == [_TA_SCENE.data, _TA_RED.data, _TA_BLUE.data]
    # The answers: the choice by option name, and a legend that repeats the levels' images
    # as the request's own attachments, never as references.
    @test r["match"].choice == "red_sample"
    @test r["closest"].legend[0] === _TA_BLUE && r["closest"].legend[1] === _TA_RED
    @test r["closest"].raw["legend"]["0"] == _ta_ref(_TA_BLUE)        # the raw answer is kept as sent

    # A JPEG part carries its own content type; the keyword form and long_context reach the wire.
    r2, seen2 = _ts_mock(; status=200, body=_ts_fixture("41-img-jpeg-state.response.json")) do
        ask((photo = _TA_SCENE_JPEG, note = "A drawing of two shapes."), "disc" => noul("Is there a blue disc in `photo`?");
            model="jev-latest", long_context=true, config=_TS_MOCK_CFG)
    end
    @test r2 isa SystemOneSuccess && r2["disc"].noul == 0.94
    h = only(seen2)["headers"]
    @test h["typesafe-danger-zone-images"] == "true" && h["typesafe-danger-zone-long-context"] == "true"
    form2 = _ta_form(only(seen2))
    @test [p.name for p in form2] == ["request", "asset-0"] && form2[2].contenttype == "image/jpeg"
    @test read(form2[2].data) == _TA_SCENE_JPEG.data

    # Legends with structured levels keep their structure, an image inside them restored.
    legend = """{"model":"jev-1.13.0","answers":{"s":{"type":"score","score":0.0,"confidence":1.0,
                 "legend":{"0":{"summary":"red","example":"$(_ta_ref(_TA_RED))"},"1":["$(_ta_ref(_TA_BLUE))","text"]},
                 "probabilities":{"0":1.0,"1":0.0}}},"usage":{"input_tokens":1,"output_tokens":1}}"""
    r3, _ = _ts_mock(; status=200, body=legend) do
        ask(_TA_SCENE, "s" => score([(summary = "red", example = _TA_RED), [_TA_BLUE, "text"]]); config=_TS_MOCK_CFG)
    end
    @test r3["s"].legend[0] == Dict{String,Any}("summary" => "red", "example" => _TA_RED)
    @test r3["s"].legend[1] == Any[_TA_BLUE, "text"] && r3["s"].legend[1][1] === _TA_BLUE
end

@testset "TypeSafe images — what TypeSafe rejects is refused locally, and nothing is sent" begin
    q = "q" => noul("Is it red?")
    base = length(_ta_png_bytes(300, 300))
    refused, seen = _ts_mock(; status=200, body=_ts_fixture("40-img-png-state.response.json")) do
        [_ta_caught(() -> ask(_TA_GIF, q; config=_TS_MOCK_CFG)),
         _ta_caught(() -> ask(_TA_WEBP, q; config=_TS_MOCK_CFG)),
         _ta_caught(() -> ask(_ta_png(279, 300), q; config=_TS_MOCK_CFG)),                     # too small
         _ta_caught(() -> ask(_ta_png(2000, 200), q; config=_TS_MOCK_CFG)),                    # too large
         _ta_caught(() -> ask("s", "c" => choice(["a" => _ta_jpeg(1569, 800), "b" => _TA_RED]); config=_TS_MOCK_CFG)),
         _ta_caught(() -> ask(_ta_png(300, 300; before=_ta_chunk("acTL", zeros(UInt8, 8))), q; config=_TS_MOCK_CFG)),
         _ta_caught(() -> ask(ImageAttachment(_TA_RED.data[1:div(end, 2)]), q; config=_TS_MOCK_CFG)),
         _ta_caught(() -> ask(_ta_png(300, 300; pad=16 * 1024 * 1024 - base - 11), q; config=_TS_MOCK_CFG)),
         _ta_caught(() -> ask(Dict("n" => _ta_nest(64)), q; config=_TS_MOCK_CFG)),
         _ta_caught(() -> ask("s", q; confg=1, config=_TS_MOCK_CFG))]
    end
    @test isempty(seen)
    @test all(e -> e isa ArgumentError && contains(e.msg, "TypeSafe"), refused)
    @test contains(refused[1].msg, "image/gif") && contains(refused[2].msg, "image/webp")
    @test contains(refused[3].msg, "279×300") && contains(refused[4].msg, "2000×200")
    @test contains(refused[5].msg, "image 1 of the request") && contains(refused[5].msg, "1569×800")
    @test contains(refused[6].msg, "animated") && contains(refused[7].msg, "cut short")
    @test contains(refused[8].msg, "16 MiB") && contains(refused[9].msg, "64 levels")
    @test contains(refused[10].msg, "accepted: long_context")
end

@testset "TypeSafe images — a recording keys on the JSON part and the digests, holds no image, replays keyless" begin
    dir = mktempdir()
    req = SystemOneRequest([_TA_RED, _TA_RED], ["same" => noul("Are the two images the same colour?")]; model="jev-latest")
    live, seen = _ts_mock(; status=200, body=_ts_fixture("42-img-same-image-twice.response.json"),
                          headers=["Content-Type" => "application/json", "x-typesafe-request-id" => "req_rec"]) do
        with_recorded_answers(() -> ask(req; config=_TS_MOCK_CFG), dir; mode=:record)
    end
    @test live isa SystemOneSuccess && length(seen) == 1
    file = only(readdir(dir; join=true))
    text = read(file, String)
    rec = JSON.parse(text)
    digests = "asset-0 $(_ta_digest(_TA_RED))"
    @test rec["request"]["path"] == "/v1/systemone" && rec["request"]["key_extra"] == digests
    @test JSON.json(rec["request"]["body"]) == _ts_fixture("42-img-same-image-twice.request.json")
    @test basename(file) == bytes2hex(UniLM.SHA.sha256(
        "POST /v1/systemone\n" * _ts_fixture("42-img-same-image-twice.request.json") * "\n" * digests)) * ".json"
    @test isnothing(findfirst(_TA_RED.data, read(file))) && !contains(text, base64encode(_TA_RED.data))
    # Replayed with no server and no key; another image is another request.
    withenv(UniLM.TYPESAFE_API_KEY => nothing) do
        replayed = with_recorded_answers(() -> ask(req), dir)
        @test replayed isa SystemOneSuccess && replayed.response.raw == live.response.raw
        @test replayed.response.request_id == "req_rec"
        other = with_recorded_answers(() -> _ta_caught(() -> ask([_TA_RED, _TA_BLUE], "same" => noul("Are the two images the same colour?");
                                                                  model="jev-latest")), dir)
        @test other isa ReplayMissError && endswith(other.body, "asset-0 $(_ta_digest(_TA_RED))\nasset-1 $(_ta_digest(_TA_BLUE))")
        # long_context keys apart; `false` is the default and keys as no option at all.
        with(long_context) = SystemOneRequest(req.state, req.questions, req.model, (; long_context))
        lc = with_recorded_answers(() -> _ta_caught(() -> ask(with(true))), dir)
        @test lc isa ReplayMissError && endswith(lc.body, "\nlong_context=true\n" * digests)
        @test with_recorded_answers(() -> ask(with(false)), dir) isa SystemOneSuccess
    end
end

@testset "TypeSafe — Box answers decode into boxes, their confidence a number or nothing" begin
    req = SystemOneRequest(_TA_SCENE, ["rect" => box("the red rectangle"), "disc" => box("the blue disc")];
                           model="jev-1.14.0")
    r, seen = _ts_mock(; status=200, body=_ts_fixture("44-box-jev-1.14.0.response.json")) do
        ask(req; config=_TS_MOCK_CFG)
    end
    @test r isa SystemOneSuccess && r.response.model == "jev-1.14.0"
    @test r["rect"].box == BoxCoordinates(0.29, 0.71, 0.18, 0.56) && r["rect"].confidence == 0.85
    @test r["disc"].box == BoxCoordinates(0.11, 0.32, 0.66, 0.87) && r["disc"].confidence == 0.82
    @test r["rect"].raw["box"]["top"] == 0.29
    @test only(seen)["headers"]["typesafe-danger-zone-images"] == "true"
    # null is a missing confidence; an absent one, a missing edge or a non-number is malformed.
    answer(a) = """{"model":"jev-1.14.0","answers":{"rect":$(a),"disc":{"type":"box","box":{"top":0.1,"bottom":0.3,""" *
                """"left":0.6,"right":0.9},"confidence":0.8}},"usage":{"input_tokens":1,"output_tokens":1}}"""
    edges = """{"top":0.29,"bottom":0.71,"left":0.18,"right":0.56}"""
    ok, _ = _ts_mock(; status=200, body=answer("""{"type":"box","box":$(edges),"confidence":null}""")) do
        ask(req; config=_TS_MOCK_CFG)
    end
    @test ok isa SystemOneSuccess && ok["rect"].confidence === nothing
    for bad in ("""{"type":"box","box":$(edges)}""",
                """{"type":"box","box":{"top":0.29,"bottom":0.71,"left":0.18},"confidence":0.5}""",
                """{"type":"box","box":{"top":"0.29","bottom":0.71,"left":0.18,"right":0.56},"confidence":0.5}""",
                """{"type":"box","box":$(edges),"confidence":"high"}""",
                """{"type":"box","box":[0.29,0.71,0.18,0.56],"confidence":0.5}""",
                """{"type":"box","box":{"top":0.29,"bottom":1.71,"left":0.18,"right":0.56},"confidence":0.5}""",
                """{"type":"noul","noul":0.5}""")
        res, _ = _ts_mock(; status=200, body=answer(bad)) do
            ask(req; config=_TS_MOCK_CFG)
        end
        @test res isa SystemOneCallError
    end
end

@testset "TypeSafe — the new error bodies become typed failures, and the terse ones get a hint" begin
    box_req = SystemOneRequest(_TA_SCENE, ["rect" => box("the red rectangle"), "disc" => box("the blue disc")];
                               model="jev-latest")
    long = SystemOneRequest("text", "fox" => noul("Is the text about a fox?"); model="jev-latest")
    long_on = SystemOneRequest("text", "fox" => noul("Is the text about a fox?"); model="jev-latest",
                               options=(long_context = true,))
    cases = [
        # (request, fixture, status, error_type, message)
        (_ts_request_02(), "30-choice-256-options.response.json", 422, nothing,
            "questions.pick.choice.criteria: Dictionary should have at most 255 items after validation, not 256"),
        (_ts_request_02(), "32-score-11-levels.response.json", 422, nothing,
            "questions.rate.score.criteria: List should have at most 10 items after validation, not 11"),
        (box_req, "50-err-image-over-16MiB.response.json", 413, "api_usage_error", "Request exceeds size limits."),
        (box_req, "51-err-body-over-32MiB.response.json", 413, "api_usage_error",
            "Request body of at least 33555590 bytes exceeds the hard maximum of 33554432 bytes."),
        (_ts_request_02(), "52-err-content-encoding.response.json", 415, "api_usage_error",
            "Unsupported Content-Encoding 'br'; supported: gzip, zstd."),
        (box_req, "53-err-invalid-image.response.json", 400, "api_usage_error", "Invalid image request."),
        (box_req, "54-err-image-dimensions.response.json", 400, "api_usage_error", "Invalid image dimensions."),
        (box_req, "55-err-images-not-turned-on.response.json", 400, "api_usage_error",
            "Images are experimental. Send the header TypeSafe-Danger-Zone-Images: true to use them."),
        (_ts_request_02(), "56-err-nesting.response.json", 400, "api_usage_error",
            "Request JSON nesting exceeds the maximum depth of 64."),
        # The bare "Invalid request." a Box question draws from Jev 1.13 names the models that answer one…
        (box_req, "45-box-jev-latest.response.json", 400, "api_usage_error",
            "Invalid request. Box questions need a Jev model that supports them, 1.14 or later, such as " *
            "\"jev-1.14.0\" or \"jev-preview\""),
        # …and only when the request holds a Box question.
        (_ts_request_02(), "45-box-jev-latest.response.json", 400, "api_usage_error", "Invalid request."),
        # A context overflow names long_context and the limits, or, with it on, the long limit.
        (long, "46-long-context-off.response.json", 400, "max_tokens_exceeded",
            "max_tokens_exceeded. The request is longer than the context of 64k tokens per request and 32k for " *
            "the state with any one question; `long_context = true` raises both to 262k at the same price per token"),
        (long_on, "46-long-context-off.response.json", 400, "max_tokens_exceeded",
            "max_tokens_exceeded. The request is longer than even the long context of 262k tokens per request " *
            "and per question: shorten the state, or split the questions across requests"),
    ]
    for (req, fixture, status, error_type, message) in cases
        body = _ts_fixture(fixture)
        # A budget of three attempts: none of these statuses may spend a second one.
        r, seen = _ts_mock(; status, body, headers=["Content-Type" => "application/json", "x-typesafe-request-id" => "req_e"]) do
            ask(req; config=UniLM.RequestConfig(max_attempts=3, total_deadline=30.0))
        end
        @test r isa SystemOneFailure
        @test r.status == status && r.error_type == error_type && r.message == message
        @test r.response == body && r.request_id == "req_e" && isnothing(r.retry_after)
        @test length(seen) == 1
    end
end
