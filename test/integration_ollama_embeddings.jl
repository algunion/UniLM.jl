# ─── EmbeddingGemma live witnesses (Ollama) ─────────────────────────────────
# Requires UNILM_LIVE_OLLAMA=1 and a reachable Ollama server with every model in
# UNILM_OLLAMA_EMBED_MODELS installed (a missing one fails). Local models: no spend. Each
# witness names the outcome that refutes it.

const _OE_EP = get(ENV, "UNILM_LIVE_OLLAMA", "") == "1" ? OllamaEndpoint() : nothing
# The test environment has no LinearAlgebra.
_oe_dot(a, b) = sum(a .* b)
_oe_norm(v) = sqrt(sum(abs2, v))
# Names as `String`s: `Embeddings` takes its model as one.
const _OE_MODELS = String.(split(get(ENV, "UNILM_OLLAMA_EMBED_MODELS",
    "embeddinggemma-2:740m-q8_0,embeddinggemma-2:740m-mxfp8,embeddinggemma-2:270m-q8_0-text,embeddinggemma"), ','))
const _OE_FIX = joinpath(@__DIR__, "fixtures", "ollama")

if isnothing(_OE_EP)
    @info "Skipping EmbeddingGemma live witnesses (set UNILM_LIVE_OLLAMA=1)"
else

const _OE_INFO = Dict(m.name => m for m in list_models(; service=_OE_EP).response)
# Ollama lists a model pulled without a tag under `latest` (`embeddinggemma` as `embeddinggemma:latest`).
_oe_listed(m) = occursin(':', last(split(m, '/'))) ? m : m * ":latest"
_oe_vec(m, x; task=:none, kw...) = embedding_vectors(embeddingrequest!(Embeddings(x; service=_OE_EP, model=m, task, kw...)))

@testset "EmbeddingGemma live — $m" for m in _OE_MODELS
    @test haskey(_OE_INFO, _oe_listed(m))        # refuted: a listed model is not installed
    haskey(_OE_INFO, _oe_listed(m)) || continue
    caps = _OE_INFO[_oe_listed(m)].capabilities

    @testset "unit vectors at every Matryoshka size" begin
        v = _oe_vec(m, "a short text"; task=:search_query)
        @test length(v) == 768 && abs(_oe_norm(v) - 1) <= 1e-3      # refuted: other length or not unit-length
        for d in (512, 256, 128)
            w = _oe_vec(m, "a short text"; task=:search_query, dimensions=d)
            @test length(w) == d && abs(_oe_norm(w) - 1) <= 1e-3
        end
    end

    @testset "same input, same vector; a batch keeps its order" begin
        a, b = "cats purr on sofas", "stock markets fell"
        v1 = _oe_vec(m, a; task=:clustering); v2 = _oe_vec(m, a; task=:clustering)
        @test maximum(abs, v1 .- v2) <= 1e-6
        ab = _oe_vec(m, [a, b]; task=:clustering); ba = _oe_vec(m, [b, a]; task=:clustering)
        @test maximum(abs, ab[1] .- ba[2]) <= 1e-5 && maximum(abs, ab[2] .- ba[1]) <= 1e-5
    end

    @testset "task prompts do not lose to raw text on the labelled set" begin
        set = JSON.parse(read(joinpath(_OE_FIX, "retrieval_set.json"), String))
        docs, qs = set["documents"], set["queries"]
        ids = [d["id"] for d in docs]
        function score(prompted::Bool)
            D = prompted ? _oe_vec(m, [EmbeddingInput(d["text"]; title=d["title"]) for d in docs]; task=:document) :
                           _oe_vec(m, [d["text"] for d in docs]; task=:none)
            Q = _oe_vec(m, [q["text"] for q in qs]; task=prompted ? :search_query : :none)
            ranks = [findfirst(==(q["gold"]), ids[sortperm([_oe_dot(qv, dv) for dv in D]; rev=true)]) for (q, qv) in zip(qs, Q)]
            (top1 = count(==(1), ranks) / length(ranks), mrr = sum(1 ./ ranks) / length(ranks))
        end
        p, r = score(true), score(false)
        @info "retrieval on $m" prompted = p raw = r
        @test p.top1 >= r.top1 || p.mrr >= r.mrr                 # refuted: prompts worse on both metrics
    end

    if :vision in caps
        @testset "image ↔ text (cross-modal)" begin
            img = ImageAttachment(joinpath(_OE_FIX, "red_circle.png"))
            vi = _oe_vec(m, img; task=:document)
            vt = _oe_vec(m, ["a red circle", "a blue square"]; task=:search_query)
            @test _oe_dot(vi, vt[1]) > _oe_dot(vi, vt[2])                 # refuted: the blue square is closer
        end
    end
    if :audio in caps
        @testset "audio ↔ text (cross-modal)" begin
            wav = AudioAttachment(joinpath(_OE_FIX, "secret_word.wav"))
            va = _oe_vec(m, wav; task=:document)
            vt = _oe_vec(m, ["banana", "apple"]; task=:search_query)
            @test _oe_dot(va, vt[1]) > _oe_dot(va, vt[2])                 # refuted: apple is closer
        end
    end

    @testset "an over-long input is a failure, never a silent vector" begin
        long = repeat("The warehouse shipped copper wire on schedule. ", m == "embeddinggemma" ? 400 : 1300)
        f = embeddingrequest!(Embeddings(long; service=_OE_EP, model=m, task=:none); config=RequestConfig(max_attempts=1))
        @test f isa EmbeddingFailure && f.status in (400, 413)   # refuted: a vector
    end

    @testset "local refusals send nothing; cost 0; record once, replay offline" begin
        @test_throws ArgumentError Embeddings("x"; service=_OE_EP, model=m)                      # no task
        @test_throws ArgumentError Embeddings("x"; service=_OE_EP, model=m, task=:none, dimensions=300)
        r = embeddingrequest!(Embeddings("x"; service=_OE_EP, model=m, task=:search_query))
        @test (@test_logs estimated_cost(r)) == 0.0 && r.usage.prompt_tokens > 0
        dir = mktempdir()
        rec = with_recorded_answers(dir; mode=:record) do
            embedding_vectors(embeddingrequest!(Embeddings("record me"; service=_OE_EP, model=m, task=:search_query)))
        end
        off = Embeddings("record me"; service=OllamaEndpoint(base_url="http://127.0.0.1:1"), model=m, task=:search_query)
        @test with_recorded_answers(() -> embedding_vectors(embeddingrequest!(off)), dir; mode=:replay) == rec
    end
end

end # live guard
