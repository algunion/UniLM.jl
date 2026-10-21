# ─── EmbeddingGemma live witnesses (Ollama) ─────────────────────────────────
# Requires UNILM_LIVE_OLLAMA=1 and a reachable Ollama server with every model in
# UNILM_OLLAMA_EMBED_MODELS installed (a missing one fails, and so does a server that does not
# list its models). Local models: no spend, and a cloud model named in the environment is
# refused before any request. Each witness names the outcome that refutes it.

const _OE_EP = get(ENV, "UNILM_LIVE_OLLAMA", "") == "1" ? OllamaEndpoint() : nothing
# The test environment has no LinearAlgebra.
_oe_dot(a, b) = sum(a .* b)
_oe_norm(v) = sqrt(sum(abs2, v))
# Names as `String`s: `Embeddings` takes its model as one. The 740m GGUF's text path is the 270m text
# model: embeddinggemma-2:740m-q8_0 and embeddinggemma-2:270m-q8_0-text return identical text
# embeddings (maximum difference 0.0, observed on Ollama 0.40.6).
const _OE_MODELS = String.(split(get(ENV, "UNILM_OLLAMA_EMBED_MODELS",
    "embeddinggemma-2:740m-q8_0,embeddinggemma-2:740m-mxfp8,embeddinggemma-2:270m-q8_0-text,embeddinggemma"), ','))
const _OE_FIX = joinpath(@__DIR__, "fixtures", "ollama")
_oe_v2(m) = occursin(r"embeddinggemma-?2(?!\d)"i, m)    # EmbeddingGemma 2 (trained on 8,192 tokens), not v1 (2,048)
# Zero spend: a cloud model (a tag of `cloud`, or one ending in `-cloud`) runs on Ollama's servers and bills
# its account, so a name from the environment that is one is refused before any request.
for m in _OE_MODELS
    UniLM._ollama_cloud_model(m) && error("EmbeddingGemma live witnesses: UNILM_OLLAMA_EMBED_MODELS names $m, " *
                                          "a cloud model, which bills Ollama's account: name local models")
end

if isnothing(_OE_EP)
    @info "Skipping EmbeddingGemma live witnesses (set UNILM_LIVE_OLLAMA=1)"
else

const _OE_INFO = let r = list_models(; service=_OE_EP, config=RequestConfig(max_attempts=1, request_timeout=5.0))
    r isa OllamaSuccess || error("EmbeddingGemma live witnesses: listing the installed models failed: $(r)")
    Dict(m.name => m for m in r.response)
end
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

    # Pre-registered: task prompts do not lose to raw text on both top-1 and MRR. Observed on Ollama 0.40.6
    # with the 24-document set (12 queries; prompted top-1 / MRR vs raw top-1 / MRR):
    #   embeddinggemma-2:740m-q8_0        0.5    / 0.7222  vs  0.5    / 0.7361   tie on top-1
    #   embeddinggemma-2:740m-mxfp8       0.5    / 0.7222  vs  0.5    / 0.7361   tie on top-1
    #   embeddinggemma-2:270m-q8_0-text   0.5    / 0.7222  vs  0.5    / 0.7361   tie on top-1
    #   embeddinggemma                    0.5833 / 0.7917  vs  0.6667 / 0.8333   refuted, by one query
    # At 12 queries a one-query difference cannot separate the hypotheses: whether the task prompts help
    # needs a separate, powered evaluation, not this fixture. Here the prompted run is a floor: a scrambled,
    # degenerate or mis-ordered batch ranks near chance (1/24 ≈ 0.04), far below 0.4. It does not detect a
    # wrong or missing prompt: on EmbeddingGemma 2 a missing query prompt or title scores 0.4167, unprompted
    # documents or the question-answering query task 0.5 (Ollama 0.40.6); the byte-exact offline tests pin
    # the prompts. Both runs are logged.
    @testset "prompted retrieval clears a wiring floor on the labelled set" begin
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
        @test p.top1 >= 0.4                       # refuted: prompted top-1 below 0.4 on the 24-document set
    end

    if :vision in caps
        @testset "image ↔ text (cross-modal)" begin
            img = ImageAttachment(joinpath(_OE_FIX, "red_circle.png"))
            vi = _oe_vec(m, img; task=:document)
            vt = _oe_vec(m, ["a red circle", "a blue square"]; task=:search_query)
            red_circle, blue_square = _oe_dot(vi, vt[1]), _oe_dot(vi, vt[2])
            @info "image ↔ text on $m" red_circle blue_square
            @test red_circle > blue_square                                 # refuted: the blue square is closer
        end
    end
    if :audio in caps
        @testset "audio ↔ text (cross-modal)" begin
            wav = AudioAttachment(joinpath(_OE_FIX, "secret_word.wav"))
            va = _oe_vec(m, wav; task=:document)
            vt = _oe_vec(m, ["banana", "apple"]; task=:search_query)
            banana, apple = _oe_dot(va, vt[1]), _oe_dot(va, vt[2])
            @info "audio ↔ text on $m" banana apple
            @test banana > apple                                           # refuted: apple is closer
        end
    end

    # A text and its media sent as one /api/embed item ({"text", "image"} or {"text", "audio"}). Refuted by: a
    # failure (Ollama's 400), a vector that is not finite or not unit-length, or a vector equal (max |Δ| < 1e-6)
    # to the text-only or the media-only embedding of the same parts — the server dropped a part. The same text
    # and task go into all three requests, so a dropped part reproduces the other request exactly.
    for (cap, label, media, words) in ((:vision, "picture", () -> ImageAttachment(joinpath(_OE_FIX, "red_circle.png")), "a red circle"),
                                       (:audio, "sound clip", () -> AudioAttachment(joinpath(_OE_FIX, "secret_word.wav")), "banana"))
        cap in caps || continue
        @testset "a text and its $label as one item: both parts reach the vector" begin
            x = media()
            item = x isa ImageAttachment ? EmbeddingInput(; text=words, image=x) : EmbeddingInput(; text=words, audio=x)
            r = embeddingrequest!(Embeddings(item; service=_OE_EP, model=m, task=:document);
                                  config=RequestConfig(max_attempts=1))
            @test r isa EmbeddingSuccess                                   # refuted: a 400 (or any failure)
            if r isa EmbeddingSuccess
                vm = embedding_vectors(r)
                vt, vx = _oe_vec(m, words; task=:document), _oe_vec(m, x; task=:document)
                @test all(isfinite, vm) && abs(_oe_norm(vm) - 1) <= 1e-3     # refuted: not finite or not unit-length
                @test maximum(abs, vm .- vt) >= 1e-6                         # refuted: the media part was dropped
                @test maximum(abs, vm .- vx) >= 1e-6                         # refuted: the text part was dropped
                @info "text + $label item on $m" mixed_text = _oe_dot(vm, vt) mixed_media = _oe_dot(vm, vx) text_media = _oe_dot(vt, vx) max_diff_text = maximum(abs, vm .- vt) max_diff_media = maximum(abs, vm .- vx)
            end
        end
    end

    # Token counts measured with prompt_eval_count on Ollama 0.40.6: 240 reports are 5,991 tokens on all
    # three EmbeddingGemma 2 builds; 330 reports are 8,331 (counted on the 740m GGUF with a 16,384-token
    # window). v1's 400 repeats are 3,203 tokens at the measured 8 a repeat (200 repeats are 1,603).
    @testset "the trained window: an input inside it is embedded whole, one past it is a failure" begin
        reports(n) = join("Report $k: the warehouse in district $k shipped $(3k) crates of copper wire on schedule. " for k in 1:n)
        embed(x) = embeddingrequest!(Embeddings(x; service=_OE_EP, model=m, task=:none); config=RequestConfig(max_attempts=1))
        if _oe_v2(m)
            inside = embed(reports(240))                                         # 5,991 tokens
            @test inside isa EmbeddingSuccess && inside.usage.prompt_tokens >= 5_900   # refuted: refused, or read short
            past = embed(reports(330))                                           # 8,331 tokens
            @test past isa EmbeddingFailure && past.status in (400, 413)         # refuted: a vector
        else
            past = embed(repeat("The warehouse shipped copper wire on schedule. ", 400))   # 3,203 tokens
            @test past isa EmbeddingFailure && past.status in (400, 413)         # refuted: a vector
        end
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
