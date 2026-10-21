# Embeddings: task prompts (EmbeddingGemma), EmbeddingInput items, construction rules. Offline.

const _ET_FIX = joinpath(@__DIR__, "fixtures", "ollama")
const _ET_IMG = ImageAttachment(joinpath(_ET_FIX, "red_circle.png"))
const _ET_WAV = AudioAttachment(joinpath(_ET_FIX, "secret_word.wav"))
const _ET_EP = OllamaEndpoint(base_url="http://box:11434")
const _ET_EG = "embeddinggemma-2:740m"
_et_err(f) = try f(); nothing catch e; e end

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
    @test UniLM._has_media(m) && !UniLM._has_media(Embeddings(["a"]; service=_ET_EP, model=_ET_EG, task=:none))
end
