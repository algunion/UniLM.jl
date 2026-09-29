# Message attachments: format recognition, validation, and how each encoder sends
# (or refuses) them. Offline.

const _PNG = UInt8[0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 0x00, 0x00, 0x00, 0x0d]
const _JPEG = UInt8[0xff, 0xd8, 0xff, 0xe0, 0x00, 0x10]
const _GIF = Vector{UInt8}(b"GIF89a\x01\x00")
const _WEBP = Vector{UInt8}(b"RIFF\x24\x00\x00\x00WEBPVP8 ")
const _WAV = Vector{UInt8}(b"RIFF\x24\x00\x00\x00WAVEfmt ")
const _MP3_ID3 = Vector{UInt8}(b"ID3\x04\x00\x00")
const _MP3_FRAME = UInt8[0xff, 0xfb, 0x90, 0x64]      # MPEG-1 Layer III
const _AAC_ADTS = UInt8[0xff, 0xf1, 0x50, 0x80]       # sync bits, layer 00: not MP3

_att_chat(service; model="m", kws...) = begin
    c = Chat(; service, model, kws...)
    push!(c, Message(Val(:system), "sys"))
    push!(c, Message(Val(:user), "What is shown?", ImageAttachment(_PNG), AudioAttachment(_WAV)))
    c
end

@testset "formats come from the leading bytes" begin
    @test ImageAttachment(_PNG).mime == "image/png"
    @test ImageAttachment(_JPEG).mime == "image/jpeg"
    @test ImageAttachment(_GIF).mime == "image/gif"
    @test ImageAttachment(_WEBP).mime == "image/webp"
    @test AudioAttachment(_WAV).format == "wav"
    @test AudioAttachment(_MP3_ID3).format == "mp3"
    @test AudioAttachment(_MP3_FRAME).format == "mp3"
    @test_throws ArgumentError AudioAttachment(_AAC_ADTS)
    @test_throws ArgumentError ImageAttachment(UInt8[])
    @test_throws ArgumentError AudioAttachment(UInt8[])
    @test_throws ArgumentError ImageAttachment(_WAV)      # a sound is not an image
    @test_throws ArgumentError AudioAttachment(_WEBP)     # RIFF, but WEBP, not WAVE
    @test_throws ArgumentError ImageAttachment(Vector{UInt8}("not an image at all"))
    # The bytes are copied: mutating the caller's buffer cannot change the attachment.
    buf = copy(_PNG); a = ImageAttachment(buf); buf[1] = 0x00
    @test a.data == _PNG
    mktempdir() do dir
        path = joinpath(dir, "shape.jpg")     # the name says JPEG; the bytes say PNG
        write(path, _PNG)
        @test ImageAttachment(path).mime == "image/png"
        wav = joinpath(dir, "clip.bin"); write(wav, _WAV)
        @test AudioAttachment(wav).format == "wav"
    end
    @test_throws SystemError ImageAttachment(joinpath(tempdir(), "unilm-no-such-file.png"))
    # A URL is not a file: the error says to download it.
    for url in ("https://example.com/cat.png", "http://h/x.wav", "file:///tmp/x.png")
        err = try ImageAttachment(url); nothing catch x; x end
        @test err isa ArgumentError && occursin("not a URL", err.msg)
        @test_throws ArgumentError AudioAttachment(url)
    end
end

@testset "show and equality" begin
    @test sprint(show, ImageAttachment(_PNG)) == "ImageAttachment(image/png, $(length(_PNG)) bytes)"
    @test sprint(show, AudioAttachment(_WAV)) == "AudioAttachment(wav, $(length(_WAV)) bytes)"
    @test ImageAttachment(_PNG) == ImageAttachment(copy(_PNG))
    @test hash(ImageAttachment(_PNG)) == hash(ImageAttachment(copy(_PNG)))
    @test ImageAttachment(_PNG) != ImageAttachment(_JPEG)
    @test ImageAttachment(_PNG) != AudioAttachment(_WAV)
end

@testset "attachments belong to user messages" begin
    img = ImageAttachment(_PNG)
    m = Message(Val(:user), "look", img)
    @test m.attachments == [img]
    @test Message(Val(:user), "look", img, AudioAttachment(_WAV)).attachments isa Vector{Attachment}
    @test isnothing(Message(role=RoleUser, content="x", attachments=Attachment[]).attachments)
    @test isnothing(Message(Val(:user), "plain").attachments)
    for role in (RoleSystem, RoleAssistant)
        @test_throws ArgumentError Message(; role, content="x", attachments=[img])
    end
    @test_throws ArgumentError Message(role=UniLM.RoleTool, content="x", tool_call_id="c", attachments=[img])
end

@testset "OpenAI wire: content becomes parts" begin
    img, aud = ImageAttachment(_PNG), AudioAttachment(_WAV)
    d = JSON.lower(Message(Val(:user), "What is shown?", img, aud))
    parts = d[:content]
    @test [p[:type] for p in parts] == ["text", "image_url", "input_audio"]
    @test parts[1][:text] == "What is shown?"
    @test parts[2][:image_url][:url] == "data:image/png;base64," * UniLM.Base64.base64encode(_PNG)
    @test parts[3][:input_audio] == Dict(:data => UniLM.Base64.base64encode(_WAV), :format => "wav")
    # No text part for empty text; a message without attachments keeps its string.
    @test [p[:type] for p in JSON.lower(Message(role=RoleUser, content="", attachments=[img]))[:content]] == ["image_url"]
    @test JSON.lower(Message(Val(:user), "plain"))[:content] == "plain"
    # Every OpenAI-wire endpoint sends the parts.
    for ep in (OPENAIServiceEndpoint, UniLM.AZUREServiceEndpoint, GEMINIOpenAIServiceEndpoint,
               GenericOpenAIEndpoint("http://127.0.0.1:8000", ""), DeepSeekEndpoint("k"))
        wire = JSON.parse(UniLM.encode_request(ep, _att_chat(ep; model=ep === OPENAIServiceEndpoint ? "gpt-5.4-mini" : "m")))
        @test [p["type"] for p in wire["messages"][2]["content"]] == ["text", "image_url", "input_audio"]
    end
end

@testset "encoders that cannot send attachments refuse them before any I/O" begin
    for (ep, model) in ((ANTHROPICServiceEndpoint, "claude-opus-5-5"), (GEMINIServiceEndpoint, "gemini-3.8-flash"))
        err = try
            UniLM.encode_request(ep, _att_chat(ep; model))
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("attachments", err.msg)
        # chatrequest! encodes before any network I/O, so the refusal surfaces as a throw.
        @test_throws ArgumentError chatrequest!(_att_chat(ep; model))
    end
end

@testset "fork copies attachments independently" begin
    c = _att_chat(GenericOpenAIEndpoint("http://127.0.0.1:8000", ""))
    f = fork(c)
    @test f.messages[2].attachments == c.messages[2].attachments
    f.messages[2].attachments[1].data[1] = 0x00
    @test c.messages[2].attachments[1].data == _PNG
end

# A prefix-completion endpoint whose URL points at a local capture server.
const _ATT_URL = Ref("http://127.0.0.1:0")
struct _AttPrefixMock <: UniLM.OpenAIWireEndpoint end
UniLM.provider_capabilities(::Type{_AttPrefixMock}) = Set([:chat, :prefix_completion])
UniLM.auth_header(::Type{_AttPrefixMock}) = ["Content-Type" => "application/json"]
UniLM._prefix_complete_url(::Type{_AttPrefixMock}) = _ATT_URL[]

@testset "prefix_complete sends each message in its wire form" begin
    # The body is built from JSON.lower(::Message), so attachments reach the wire.
    body = Ref("")
    server = HTTP.serve!("127.0.0.1", 0; listenany=true) do req
        body[] = String(copy(req.body))
        HTTP.Response(200, ["Content-Type" => "application/json"], Vector{UInt8}(JSON.json(Dict(
            "choices" => [Dict("message" => Dict("role" => "assistant", "content" => "done"),
                               "finish_reason" => "stop")]))))
    end
    try
        _ATT_URL[] = "http://127.0.0.1:$(HTTP.port(server))/v1/chat/completions"
        c = _att_chat(_AttPrefixMock)
        push!(c, Message(role=RoleAssistant, content="It shows"))
        r = prefix_complete(c; config=RequestConfig(max_attempts=1, request_timeout=10.0))
        @test r isa LLMSuccess
        wire = JSON.parse(body[])
        @test [p["type"] for p in wire["messages"][2]["content"]] == ["text", "image_url", "input_audio"]
        @test wire["messages"][end] == Dict("role" => "assistant", "content" => "It shows", "prefix" => true)
    finally
        close(server)
    end
end
