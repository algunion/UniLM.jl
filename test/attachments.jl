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
# A one-page PDF whose only text is "The secret word is PELICAN." (Helvetica, exact xref offsets).
const _PDF_PATH = joinpath(@__DIR__, "fixtures", "documents", "secret_word.pdf")
const _PDF = read(_PDF_PATH)
const _PDF_URL = "data:application/pdf;base64," * UniLM.Base64.base64encode(_PDF)
_att_caught(f) = try f(); nothing catch e; e end

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
        # Every kind is refused, a document or an image with a detail level alone included.
        for att in (DocumentAttachment(_PDF_PATH), ImageAttachment(_PNG; detail="high"))
            c = Chat(; service=ep, model)
            push!(c, Message(Val(:system), "sys"))
            push!(c, Message(Val(:user), "Read this.", att))
            e = _att_caught(() -> UniLM.encode_request(ep, c))
            @test e isa ArgumentError && occursin("(images, audio and documents)", e.msg)
        end
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

@testset "documents: a PDF, recognised by its leading bytes" begin
    doc = DocumentAttachment(_PDF_PATH)
    @test doc isa Attachment && fieldnames(DocumentAttachment) == (:data, :mime, :filename)
    @test doc.mime == "application/pdf" && doc.filename == "secret_word.pdf" && doc.data == _PDF
    @test DocumentAttachment(_PDF).filename == "document.pdf"                   # bytes: the default name
    @test DocumentAttachment(_PDF; filename="report.pdf").filename == "report.pdf"
    v = DocumentAttachment(view(_PDF, :); filename=SubString("a report.pdf", 3))
    @test v.data isa Vector{UInt8} && v.data == _PDF && v.filename isa String && v.filename == "report.pdf"
    # The bytes are copied: mutating the caller's buffer cannot change the document.
    buf = copy(_PDF); d = DocumentAttachment(buf); buf[1] = 0x00
    @test d.data == _PDF
    # The signature decides, never the name: a header alone is PDF by its signature.
    @test DocumentAttachment(Vector{UInt8}(b"%PDF-1.7\n")).mime == "application/pdf"
    for bad in (b"hello", b"", b"%PDF", b"%PDF1.4", b" %PDF-1.4", b"%pdf-1.4")
        e = _att_caught(() -> DocumentAttachment(Vector{UInt8}(bad)))
        @test e isa ArgumentError && e.msg == "document data is not PDF (checked by file signature; $(length(bad)) bytes)"
    end
    @test _att_caught(() -> DocumentAttachment(_PNG)) isa ArgumentError         # an image is not a document
    @test _att_caught(() -> ImageAttachment(_PDF)) isa ArgumentError            # nor a document an image
    mktempdir() do dir
        named = joinpath(dir, "notes.txt"); write(named, _PDF)
        @test DocumentAttachment(named).filename == "notes.txt"
        fake = joinpath(dir, "fake.pdf"); write(fake, "hello")
        @test occursin("not PDF", _att_caught(() -> DocumentAttachment(fake)).msg)
    end
    @test_throws SystemError DocumentAttachment(joinpath(tempdir(), "unilm-no-such-file.pdf"))
    for url in ("https://example.com/report.pdf", "file:///tmp/report.pdf")      # a URL is not a file
        e = _att_caught(() -> DocumentAttachment(url))
        @test e isa ArgumentError && occursin("not a URL", e.msg)
    end
    for blank in ("", "  ")                                                   # the name a provider is given
        e = _att_caught(() -> DocumentAttachment(_PDF; filename=blank))
        @test e isa ArgumentError && occursin("filename", e.msg)
    end
    # A user message carries it; no other role does.
    @test Message(Val(:user), "What is the secret word?", doc).attachments == Attachment[doc]
    @test_throws ArgumentError Message(role=RoleAssistant, content="x", attachments=Attachment[doc])
    # Written as JSON anywhere else it is refused, naming where a document goes.
    e = _att_caught(() -> JSON.json(InputMessage(role="user", content=[input_text("Read it."), doc])))
    @test e isa ArgumentError && occursin("attachments", e.msg) && occursin("input_file", e.msg)
end

@testset "an image's detail level" begin
    @test isnothing(ImageAttachment(_PNG).detail) && fieldnames(ImageAttachment) == (:data, :mime, :detail)
    for level in ("auto", "low", "high", "original")
        @test ImageAttachment(_PNG; detail=level).detail == level
    end
    @test ImageAttachment(_PNG; detail=SubString("ultra-high", 7)).detail isa String
    mktempdir() do dir
        path = joinpath(dir, "shape.png"); write(path, _PNG)
        @test ImageAttachment(path; detail="low").detail == "low" && isnothing(ImageAttachment(path).detail)
    end
    for bad in ("medium", "", "HIGH", :high, 1)
        e = _att_caught(() -> ImageAttachment(_PNG; detail=bad))
        @test e isa ArgumentError && all(level -> occursin(repr(level), e.msg), ("auto", "low", "high", "original"))
    end
end

@testset "show, equality and hash of documents and detailed images" begin
    doc = DocumentAttachment(_PDF_PATH)
    @test sprint(show, doc) == "DocumentAttachment(application/pdf, secret_word.pdf, $(length(_PDF)) bytes)"
    same = DocumentAttachment(copy(_PDF); filename="secret_word.pdf")
    @test doc == same && isequal(doc, same) && hash(doc) == hash(same)
    @test doc != DocumentAttachment(_PDF)                                     # another name
    @test doc != DocumentAttachment(Vector{UInt8}(b"%PDF-1.7\n"); filename="secret_word.pdf")
    img = ImageAttachment(_PNG; detail="original")
    @test sprint(show, img) == "ImageAttachment(image/png, $(length(_PNG)) bytes, detail=\"original\")"
    @test sprint(show, ImageAttachment(_PNG)) == "ImageAttachment(image/png, $(length(_PNG)) bytes)"
    twin = ImageAttachment(copy(_PNG); detail="original")
    @test img == twin && hash(img) == hash(twin)
    @test img != ImageAttachment(_PNG) && img != ImageAttachment(_PNG; detail="high")
    @test length(Set([doc, same, img, twin, ImageAttachment(_PNG)])) == 3
end

@testset "OpenAI wire: a document as a file part, an image with its detail level" begin
    doc = DocumentAttachment(_PDF_PATH)
    png_url = "data:image/png;base64," * UniLM.Base64.base64encode(_PNG)
    @test UniLM._wire_part(doc) == Dict(:type => "file", :file => Dict(:filename => "secret_word.pdf", :file_data => _PDF_URL))
    @test (@inferred UniLM._wire_part(doc)) isa Dict{Symbol,Any}
    @test UniLM._wire_part(ImageAttachment(_PNG; detail="original")) ==
          Dict(:type => "image_url", :image_url => Dict(:url => png_url, :detail => "original"))
    @test UniLM._wire_part(ImageAttachment(_PNG)) == Dict(:type => "image_url", :image_url => Dict(:url => png_url))
    # An image without a detail level writes no `detail` key: its part is the URL alone, byte for byte.
    @test JSON.json(Message(Val(:user), "What is shown?", ImageAttachment(_PNG), AudioAttachment(_WAV))) ==
          """{"content":[{"text":"What is shown?","type":"text"},{"image_url":{"url":"data:image/png;base64,""" *
          """iVBORw0KGgoAAAAN"},"type":"image_url"},{"input_audio":{"data":"UklGRiQAAABXQVZFZm10IA==","format":"wav"},""" *
          """"type":"input_audio"}],"role":"user"}"""
    # Every OpenAI-wire endpoint sends both; whether a compatible server reads them is its answer.
    for ep in (OPENAIServiceEndpoint, UniLM.AZUREServiceEndpoint, GEMINIOpenAIServiceEndpoint,
               GenericOpenAIEndpoint("http://127.0.0.1:8000", ""), DeepSeekEndpoint("k"))
        c = Chat(service=ep, model=ep === OPENAIServiceEndpoint ? "gpt-6-luna" : "m")
        push!(c, Message(Val(:system), "sys"))
        push!(c, Message(Val(:user), "Read both.", ImageAttachment(_PNG; detail="high"), doc))
        parts = JSON.parse(UniLM.encode_request(ep, c))["messages"][2]["content"]
        @test parts == [Dict("type" => "text", "text" => "Read both."),
                        Dict("type" => "image_url", "image_url" => Dict("url" => png_url, "detail" => "high")),
                        Dict("type" => "file", "file" => Dict("filename" => "secret_word.pdf", "file_data" => _PDF_URL))]
    end
    # A document alone is a message's whole content.
    @test JSON.lower(Message(role=RoleUser, attachments=Attachment[doc]))[:content] == [UniLM._wire_part(doc)]
    # A recording names the document by its digest, never its bytes.
    c = Chat(service=GenericOpenAIEndpoint("http://127.0.0.1:8000", ""), model="m")
    push!(c, Message(Val(:system), "sys"))
    push!(c, Message(Val(:user), "Read it.", doc))
    keyed = UniLM._redacted(UniLM.encode_request(c.service, c))
    @test occursin("data:application/pdf;sha256,", keyed) && !occursin(UniLM.Base64.base64encode(_PDF), keyed)
end

# A local server counting the requests that reach it: a refusal before any I/O leaves it at 0.
function _att_count(f::Function)::Int
    hits = Threads.Atomic{Int}(0)
    server = HTTP.serve!("127.0.0.1", 0; listenany=true) do req
        Threads.atomic_add!(hits, 1)
        HTTP.Response(500, ["Content-Type" => "application/json"], Vector{UInt8}("{}"))
    end
    try
        f("http://127.0.0.1:$(HTTP.port(server))")
    finally
        close(server)
    end
    hits[]
end

@testset "Ollama refuses a document or an image's detail level, before any I/O" begin
    doc, img = DocumentAttachment(_PDF_PATH), ImageAttachment(_PNG; detail="low")
    chat(ep, att...) = (c = Chat(service=ep, model="gemma4:e4b"); push!(c, Message(Val(:system), "s"));
                        push!(c, Message(Val(:user), "Read this.", att...)); c)
    for (att, word) in (((doc,), "documents"), ((img,), "detail"), ((ImageAttachment(_PNG), doc), "documents"))
        e = _att_caught(() -> UniLM.encode_request(OllamaEndpoint(), chat(OllamaEndpoint(), att...)))
        @test e isa ArgumentError && occursin("Ollama", e.msg) && occursin(word, e.msg)
    end
    # Images without a level and clips still go in `images`, in order.
    wire = JSON.parse(UniLM.encode_request(OllamaEndpoint(),
                                           chat(OllamaEndpoint(), ImageAttachment(_PNG), AudioAttachment(_WAV))))
    @test wire["messages"][2]["images"] == [UniLM.Base64.base64encode(_PNG), UniLM.Base64.base64encode(_WAV)]
    # Its Responses API documents no file inputs: an input_file part is refused, by data, URL or
    # file id, in an InputMessage or a plain input item.
    asking(ep, items) = Respond(service=ep, model="gemma4:e4b", input=items)
    message(part) = [InputMessage(role="user", content=[input_text("Read this."), part])]
    for items in (message(input_file(doc)), message(input_file(url="https://example.com/report.pdf")),
                  message(input_file(id="file-1")),
                  [Dict("role" => "user", "content" => [input_text("Read this."), input_file(doc)])])
        e = _att_caught(() -> UniLM.encode_agentic(OllamaEndpoint(), asking(OllamaEndpoint(), items)))
        @test e isa ArgumentError && occursin("Ollama", e.msg) && occursin("input_file", e.msg) &&
              occursin("input_text", e.msg)
    end
    @test JSON.parse(UniLM.encode_agentic(OllamaEndpoint(), asking(OllamaEndpoint(), message(input_text("No file.")))))["input"][1]["content"] ==
          [Dict("type" => "input_text", "text" => "Read this."), Dict("type" => "input_text", "text" => "No file.")]
    seen = _att_count() do url
        for att in (doc, img)
            @test_throws ArgumentError chatrequest!(chat(OllamaEndpoint(base_url=url), att);
                                                    config=RequestConfig(max_attempts=1))
        end
        @test_throws ArgumentError respond(asking(OllamaEndpoint(base_url=url), message(input_file(doc)));
                                           config=RequestConfig(max_attempts=1))
    end
    @test seen == 0
end

@testset "embeddings take no document and no image detail level" begin
    doc, img = DocumentAttachment(_PDF_PATH), ImageAttachment(_PNG; detail="high")
    ollama = (service=OllamaEndpoint(), model="nomic-embed-text")
    for f in (() -> EmbeddingInput(img), () -> EmbeddingInput("a picture", img), () -> EmbeddingInput(; image=img),
              () -> Embeddings(img; ollama...), () -> Embeddings(["a", img]; ollama...))
        e = _att_caught(f)
        @test e isa ArgumentError && occursin("detail", e.msg)
    end
    for f in (() -> EmbeddingInput("a report", doc), () -> Embeddings(doc; ollama...), () -> Embeddings(["a", doc]; ollama...))
        e = _att_caught(f)
        @test e isa ArgumentError && occursin("DocumentAttachment", e.msg)
    end
    @test EmbeddingInput(ImageAttachment(_PNG)).image == ImageAttachment(_PNG)   # without a level: an item
end

# System One endpoints whose URL is the counting server: one speaks TypeSafe's wire (the
# protocol fallback), the other OpenAI's Decisions API.
const _ATT_S1_URL = Ref("http://127.0.0.1:0")
struct _AttTypeSafeMock <: UniLM.ServiceEndpoint end
struct _AttDecisionsMock <: UniLM.ServiceEndpoint end
UniLM.system_one_protocol(::Type{_AttDecisionsMock}) = UniLM.OpenAIDecisionsProtocol()
UniLM._resolve_base_url(::Type{_AttTypeSafeMock}) = _ATT_S1_URL[]
UniLM._resolve_base_url(::Type{_AttDecisionsMock}) = _ATT_S1_URL[]
UniLM.auth_header(::Type{_AttTypeSafeMock}) = ["Content-Type" => "application/json"]
UniLM.auth_header(::Type{_AttDecisionsMock}) = ["Content-Type" => "application/json"]
UniLM.provider_capabilities(::Type{_AttTypeSafeMock}) = Set([:system_one])
UniLM.provider_capabilities(::Type{_AttDecisionsMock}) = Set([:system_one])

@testset "System One carries no document and no image detail level, on any backend, before any I/O" begin
    doc, img = DocumentAttachment(_PDF_PATH), ImageAttachment(_PNG; detail="high")
    # Nested in the state, as the state, or in a question's guidance.
    cases = [(Dict("report" => doc), noul("Is it signed?"), "documents"),
             (["Read this.", doc], noul("Is it signed?"), "documents"),
             ((note = "see", files = (doc,)), noul("Is it signed?"), "documents"),
             (img, noul("Is it red?"), "detail"),
             ([img, "What is it?"], noul("Is it red?"), "detail"),
             ("a ticket", choice(["a" => img, "b" => "the other one"]), "detail"),
             ("a ticket", noul(Dict("look" => img)), "detail"),
             ("a ticket", RawQuestion(Dict("type" => "rank", "items" => [doc])), "documents")]
    seen = _att_count() do url
        _ATT_S1_URL[] = url
        for ep in (_AttTypeSafeMock, _AttDecisionsMock, OllamaEndpoint(base_url=url)), (state, q, word) in cases
            e = _att_caught(() -> ask(state, "q" => q; service=ep, model="m", config=RequestConfig(max_attempts=1)))
            @test e isa ArgumentError && occursin(string(UniLM._service_name(ep)), e.msg) && occursin(word, e.msg)
        end
    end
    @test seen == 0
    # A state that holds itself is walked once: its document is still found, on every backend.
    cyclic = Dict{String,Any}("report" => doc); cyclic["self"] = cyclic
    for ep in (_AttTypeSafeMock, _AttDecisionsMock, OllamaEndpoint())
        e = _att_caught(() -> UniLM._system_one_call(SystemOneRequest(cyclic, "q" => noul("Is it signed?"); model="m"), ep))
        @test e isa ArgumentError && occursin("documents", e.msg)
    end
    # Without them the same requests encode; a document is no state or entry at all.
    for ep in (_AttTypeSafeMock, _AttDecisionsMock, OllamaEndpoint())
        @test UniLM._system_one_call(SystemOneRequest(Dict("report" => "signed"), "q" => noul("Is it signed?");
                                                      model="m"), ep) isa UniLM._SystemOneCall
    end
    @test UniLM._system_one_call(SystemOneRequest(ImageAttachment(_PNG), "q" => noul("Is it red?"); model="m"),
                                 OllamaEndpoint()) isa UniLM._SystemOneCall
    @test_throws ArgumentError SystemOneRequest(doc, "q" => noul("Is it signed?"))
    @test_throws ArgumentError noul(doc)
end
