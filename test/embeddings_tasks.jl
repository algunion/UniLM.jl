# Embeddings: task prompts (EmbeddingGemma), EmbeddingInput items, construction rules. Offline.

const _ET_FIX = joinpath(@__DIR__, "fixtures", "ollama")
const _ET_IMG = ImageAttachment(joinpath(_ET_FIX, "red_circle.png"))
const _ET_WAV = AudioAttachment(joinpath(_ET_FIX, "secret_word.wav"))
const _ET_EP = OllamaEndpoint(base_url="http://box:11434")
const _ET_EG = "embeddinggemma-2:740m"
_et_err(f) = try f(); nothing catch e; e end
struct _ETVideo <: Attachment; data::Vector{UInt8}; end     # media an EmbeddingInput cannot carry

@testset "EmbeddingInput: one item and its invariants" begin
    @test EmbeddingInput("a").text == "a" && isnothing(EmbeddingInput("a").title)
    @test EmbeddingInput("a"; title="T").title == "T"
    @test EmbeddingInput(_ET_IMG).image === _ET_IMG && isnothing(EmbeddingInput(_ET_IMG).text)
    @test EmbeddingInput(_ET_WAV).audio === _ET_WAV
    x = EmbeddingInput("caption", _ET_IMG, _ET_WAV)
    @test x.text == "caption" && x.image === _ET_IMG && x.audio === _ET_WAV
    @test _et_err(() -> EmbeddingInput()) isa ArgumentError                          # nothing to embed
    @test _et_err(() -> EmbeddingInput(; title="T", image=_ET_IMG)) isa ArgumentError  # a title needs text
    @test _et_err(() -> EmbeddingInput("t", _ET_IMG, _ET_IMG)) isa ArgumentError       # one image per item
    @test _et_err(() -> EmbeddingInput("t", _ET_WAV, _ET_WAV)) isa ArgumentError       # one clip per item
    e = _et_err(() -> EmbeddingInput("caption", _ET_IMG, _ETVideo(UInt8[1, 2, 3])))    # never dropped silently
    @test e isa ArgumentError && occursin("_ETVideo", e.msg) && occursin("an image and/or a sound clip", e.msg)
    @test _et_err(() -> EmbeddingInput("")) isa ArgumentError                          # empty text, no media
    @test _et_err(() -> EmbeddingInput(; text="", title="T")) isa ArgumentError
    @test EmbeddingInput("", _ET_IMG).image === _ET_IMG                                  # the image is embedded
    img2 = ImageAttachment(copy(_ET_IMG.data))                                         # equal, not identical
    @test EmbeddingInput("x", _ET_IMG) == EmbeddingInput("x", img2) &&
          hash(EmbeddingInput("x", _ET_IMG)) == hash(EmbeddingInput("x", img2))
    @test length(Set([EmbeddingInput("x", _ET_IMG), EmbeddingInput("x", img2)])) == 1
    @test EmbeddingInput("x"; title="T") != EmbeddingInput("x") != EmbeddingInput("y") != EmbeddingInput(_ET_IMG)
    @test EmbeddingInput("x", _ET_IMG) != EmbeddingInput("x", _ET_WAV)                  # the media alone differ
    shown = sprint(show, x)
    @test occursin("caption", shown) && occursin("bytes", shown) && length(shown) < 200   # no byte dump
end

@testset "Embeddings task: checked when the request is built" begin
    @test Embeddings("q"; service=_ET_EP, model=_ET_EG, task=:search_query).task === :search_query
    e = _et_err(() -> Embeddings("q"; service=_ET_EP, model=_ET_EG))
    @test e isa ArgumentError && occursin("task", e.msg) && occursin(":none", e.msg) && occursin(":search_query", e.msg)
    @test Embeddings("q"; service=_ET_EP, model="EmbeddingGemma", task=:none).task === :none   # any case
    @test Embeddings("q"; service=_ET_EP, model="hf.co/ggml-org/embeddinggemma-2-GGUF:Q8_0", task=:document).task === :document
    @test _et_err(() -> Embeddings("q"; service=_ET_EP, model=_ET_EG, task=:retrieval)) isa ArgumentError
    e = _et_err(() -> Embeddings("q"; service=_ET_EP, model="nomic-embed-text", task=:search_query))
    @test e isa ArgumentError && occursin("nomic-embed-text", e.msg) && occursin(":none", e.msg)
    @test isnothing(Embeddings("q"; service=_ET_EP, model="nomic-embed-text").task)          # unchanged
    @test Embeddings("q"; service=_ET_EP, model="nomic-embed-text", task=:none).task === :none
    @test isnothing(Embeddings("q").task)                                                    # OpenAI default unchanged
    @test _et_err(() -> Embeddings([EmbeddingInput("t"; title="T")]; service=_ET_EP, model=_ET_EG,
                                   task=:search_query)) isa ArgumentError                   # titles: :document only
    titled = EmbeddingInput("t"; title="T")
    for task in (:search_query, :none)              # single-item path; the message names the model and both fixes
        e = _et_err(() -> Embeddings(titled; service=_ET_EP, model=_ET_EG, task))
        @test e isa ArgumentError && occursin(_ET_EG, e.msg) && occursin("task = :document", e.msg) &&
              occursin("drop the title", e.msg)
    end
    @test Embeddings(titled; service=_ET_EP, model=_ET_EG, task=:document).input == titled
    for task in (nothing, :none, :document)         # no task prompts: every way out says to drop the title
        e = _et_err(() -> Embeddings([titled]; service=_ET_EP, model="nomic-embed-text", task))
        @test e isa ArgumentError && occursin("\"nomic-embed-text\"", e.msg) && occursin("drop it", e.msg)
    end
    e = _et_err(() -> Embeddings(titled))                                                   # OpenAI default model
    @test e isa ArgumentError && occursin("text-embedding-3-small", e.msg) && occursin("drop it", e.msg)
    for d in (768, 512, 256, 128)
        @test Embeddings("q"; service=_ET_EP, model=_ET_EG, task=:none, dimensions=d).dimensions == d
    end
    e = _et_err(() -> Embeddings("q"; service=_ET_EP, model=_ET_EG, task=:none, dimensions=300))
    @test e isa ArgumentError && occursin("768, 512, 256", e.msg)
    @test _et_err(() -> Embeddings("q"; service=_ET_EP, model="nomic-embed-text", dimensions=0)) isa ArgumentError
    @test Embeddings("q"; service=_ET_EP, model="nomic-embed-text", dimensions=300).dimensions == 300
    @test _et_err(() -> Embeddings(EmbeddingInput[]; service=_ET_EP, model=_ET_EG, task=:none)) isa ArgumentError
end

@testset "Embeddings inputs: strings stay strings; any other item makes items" begin
    @test Embeddings(["a", "b"]; service=_ET_EP, model=_ET_EG, task=:document).input == ["a", "b"]
    m = Embeddings(["a", _ET_IMG, EmbeddingInput("c"; title="C")]; service=_ET_EP, model=_ET_EG, task=:document)
    @test m.input isa Vector{EmbeddingInput} && m.input[1].text == "a" && m.input[2].image === _ET_IMG
    @test length(m.embeddings) == 3 && m.embeddings isa Vector{Vector{Float64}}
    s = Embeddings(_ET_IMG; service=_ET_EP, model=_ET_EG, task=:none)
    @test s.input isa EmbeddingInput && s.embeddings isa Vector{Float64}
    @test _et_err(() -> Embeddings(Any["a", 3]; service=_ET_EP, model=_ET_EG, task=:none)) isa ArgumentError
    video = _ETVideo(UInt8[1, 2, 3])                 # an attachment no item holds: refused by name, alone or listed
    for input in (video, ["a", video])
        e = _et_err(() -> Embeddings(input; service=_ET_EP, model=_ET_EG, task=:none))
        @test e isa ArgumentError && occursin("_ETVideo", e.msg)
    end
    @test UniLM._has_media(m) && !UniLM._has_media(Embeddings(["a"]; service=_ET_EP, model=_ET_EG, task=:none))
    @test UniLM._has_media(s) && !UniLM._has_media(Embeddings(EmbeddingInput("a"); service=_ET_EP, model=_ET_EG, task=:none))
    @test !UniLM._has_media(Embeddings("a"; service=_ET_EP, model=_ET_EG, task=:none))
    @test Embeddings(""; service=_ET_EP, model="nomic-embed-text").input == ""            # a plain String: as before
end

const _ET_PROMPTS = (search_query = "task: search result | query: ",
                     question_answering = "task: question answering | query: ",
                     fact_checking = "task: fact checking | query: ", code_retrieval = "task: code retrieval | query: ",
                     classification = "task: classification | query: ", clustering = "task: clustering | query: ",
                     sentence_similarity = "task: sentence similarity | query: ")
_et_body(e) = JSON.parse(UniLM._encode_embeddings(e.service, e))

@testset "EmbeddingGemma prompts, byte for byte, on every wire" begin
    for (task, prefix) in pairs(_ET_PROMPTS)
        @test _et_body(Embeddings("How long do refunds take?"; service=_ET_EP, model=_ET_EG, task))["input"] ==
              prefix * "How long do refunds take?"
    end
    docs = Embeddings([EmbeddingInput("Refunds take 5 days."; title="Refund policy"), "Shipping is free."];
                      service=_ET_EP, model=_ET_EG, task=:document)
    @test _et_body(docs)["input"] == ["title: Refund policy | text: Refunds take 5 days.",
                                      "title: none | text: Shipping is free."]
    @test _et_body(Embeddings("raw"; service=_ET_EP, model=_ET_EG, task=:none))["input"] == "raw"
    @test _et_body(Embeddings("x | y"; service=_ET_EP, model="embeddinggemma", task=:clustering))["input"] ==
          "task: clustering | query: x | y"                        # v1 too; the separator in a text passes verbatim
    g = GenericOpenAIEndpoint("http://vllm:8000", "")
    @test JSON.parse(JSON.json(Embeddings(["a"]; service=g, model="google/embeddinggemma-2",
                                          task=:search_query)))["input"] == ["task: search result | query: a"]
    gw(input, task) = _et_body(Embeddings(input; service=g, model="google/embeddinggemma-2", task))["input"]
    @test gw(EmbeddingInput("a"), :search_query) == "task: search result | query: a"            # one item
    @test gw([EmbeddingInput("a"), EmbeddingInput("b")], :none) == ["a", "b"]                    # a vector of items
    @test gw(EmbeddingInput("Refunds take 5 days."; title="T"), :document) == "title: T | text: Refunds take 5 days."
    ow(input, task) = _et_body(Embeddings(input; service=_ET_EP, model=_ET_EG, task))["input"]
    for wire in (gw, ow)                                        # a title's own " | " passes verbatim on both wires
        @test wire([EmbeddingInput("body"; title="Refunds | Returns")], :document) == ["title: Refunds | Returns | text: body"]
    end
    @test _et_body(Embeddings(["a"]; service=_ET_EP, model="nomic-embed-text"))["input"] == ["a"]   # unchanged
    for e in (Embeddings(["", "a"]), Embeddings(["", "a"]; service=_ET_EP, model="nomic-embed-text"))
        @test _et_body(e)["input"] == ["", "a"]                         # strings stay strings, an empty one too
    end
end

@testset "media: Ollama items; text-only wires refuse them; recordings hold digests" begin
    e = Embeddings(["a red circle", _ET_IMG, EmbeddingInput("caption", _ET_IMG), _ET_WAV];
                   service=_ET_EP, model=_ET_EG, task=:document)
    b = _et_body(e)["input"]
    b64 = UniLM.Base64.base64encode
    @test b[1] == "title: none | text: a red circle"
    @test b[2] == Dict("image" => b64(_ET_IMG.data))                       # media take no prefix
    @test b[3] == Dict("text" => "title: none | text: caption", "image" => b64(_ET_IMG.data))
    @test b[4] == Dict("audio" => b64(_ET_WAV.data))
    k = JSON.parse(UniLM._record_embeddings_body(_ET_EP, e, UniLM._encode_embeddings(_ET_EP, e)))["input"]
    @test k[2]["image"] == UniLM._media_digest(_ET_IMG) && k[4]["audio"] == UniLM._media_digest(_ET_WAV)
    t = Embeddings(["a"]; service=_ET_EP, model="nomic-embed-text")
    @test UniLM._record_embeddings_body(_ET_EP, t, UniLM._encode_embeddings(_ET_EP, t)) ==
          UniLM._encode_embeddings(_ET_EP, t)                                   # text bodies key as before
    dead = GenericOpenAIEndpoint("http://127.0.0.1:1", "")        # nothing listens: I/O would return a call error
    err = _et_err(() -> embeddingrequest!(Embeddings(_ET_IMG; service=dead, model="google/embeddinggemma-2", task=:none)))
    @test err isa ArgumentError                              # thrown before any I/O, never returned as a call error
    @test occursin("service: GenericOpenAIEndpoint", err.msg)                            # the backend is named
    for (svc, name) in ((OPENAIServiceEndpoint, "OPENAIServiceEndpoint"),
                        (GEMINIOpenAIServiceEndpoint, "GEMINIOpenAIServiceEndpoint"),
                        (GenericOpenAIEndpoint("http://vllm:8000", ""), "GenericOpenAIEndpoint"))
        err = _et_err(() -> UniLM._encode_embeddings(svc, Embeddings([_ET_IMG]; service=svc,
                                                                     model="google/embeddinggemma-2", task=:none)))
        @test err isa ArgumentError && startswith(err.msg, "the OpenAI Embeddings API takes text only") &&
              occursin("service: $name", err.msg)                        # the wire refuses; the backend is named
    end
    err = _et_err(() -> JSON.json(Embeddings([_ET_IMG]; service=_ET_EP, model=_ET_EG, task=:none)))
    @test err isa ArgumentError && startswith(err.msg, "the OpenAI Embeddings API takes text only") &&
          occursin("service: OllamaEndpoint", err.msg) && occursin("embeddingrequest!", err.msg)  # lowered for the wrong wire
end

@testset "an empty text beside media is no text: the media alone are sent, with no bare prompt" begin
    x = EmbeddingInput("", _ET_IMG)
    @test isnothing(x.text) && x.image === _ET_IMG && x == EmbeddingInput(_ET_IMG)
    @test isnothing(EmbeddingInput(; text="", audio=_ET_WAV).text) && isnothing(EmbeddingInput("", _ET_IMG, _ET_WAV).text)
    img = UniLM.Base64.base64encode(_ET_IMG.data)
    for task in (:document, :search_query, :none)
        @test _et_body(Embeddings(x; service=_ET_EP, model=_ET_EG, task))["input"] == Dict("image" => img)
    end
    @test _et_body(Embeddings(["a", x]; service=_ET_EP, model=_ET_EG, task=:document))["input"] ==
          ["title: none | text: a", Dict("image" => img)]
    e = _et_err(() -> EmbeddingInput(; text="", title="T", image=_ET_IMG))        # a title needs a text
    @test e isa ArgumentError && occursin("title names the document its text comes from", e.msg)
    @test _et_err(() -> EmbeddingInput("")) isa ArgumentError                     # alone: nothing to embed
    @test JSON.parse(JSON.json(Embeddings("")))["input"] == "" && Embeddings(["", "a"]).input == ["", "a"]  # strings: as before
end

@testset "driver: a media embedding records by digest and replays offline; a 400 stays a failure" begin
    server, url, bodies = _ollama_mock((n, body) -> n == 1 ?
        (200, [JSON.json(Dict("model" => _ET_EG, "embeddings" => [[0.6, 0.8]], "prompt_eval_count" => 3))]) :
        (400, [JSON.json(Dict("error" => "dimensions 300 not supported by this model; valid values are [128 256 512 768]"))]))
    try
        dir = mktempdir()
        r = with_recorded_answers(dir; mode=:record) do
            embeddingrequest!(Embeddings(_ET_IMG; service=OllamaEndpoint(base_url=url), model=_ET_EG, task=:none))
        end
        @test r isa EmbeddingSuccess
        sent = only(bodies)                                       # the server gets the bytes; only the file the digest
        @test occursin(UniLM.Base64.base64encode(_ET_IMG.data), sent) && !occursin(UniLM._media_digest(_ET_IMG), sent)
        rec = read(only(readdir(dir; join=true)), String)
        @test !occursin(UniLM.Base64.base64encode(_ET_IMG.data), rec) && occursin(UniLM._media_digest(_ET_IMG), rec)
        off = Embeddings(_ET_IMG; service=OllamaEndpoint(base_url="http://127.0.0.1:1"), model=_ET_EG, task=:none)
        @test embedding_vectors(with_recorded_answers(() -> embeddingrequest!(off), dir; mode=:replay)) == [0.6, 0.8]
        f = embeddingrequest!(Embeddings("x"; service=OllamaEndpoint(base_url=url), model="nomic-embed-text",
                                         dimensions=300); config=RequestConfig(max_attempts=1))
        @test f isa EmbeddingFailure && f.status == 400 && occursin("not supported", f.response)
    finally
        close(server)
    end
end
