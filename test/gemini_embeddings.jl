# Native Gemini embeddings: embeddingrequest! on models.embedContent (one input) and models.batchEmbedContents
# (several), src/gemini_embeddings.jl. The replies are the shapes the API returned on 2026-11-17.

const _GE = GEMINIServiceEndpoint
_ge_body(e) = JSON.parse(UniLM._encode_embeddings(_GE, e))
_ge_text(text) = Dict("parts" => [Dict("text" => text)])

@testset "Gemini embeddings: the native endpoint embeds, gemini-embedding-2 by default" begin
    @test has_capability(_GE, :embeddings)
    e = Embeddings("hello"; service=_GE)
    @test e.model == "gemini-embedding-2"
    @test UniLM.get_url(e) == "https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-2:embedContent"
    v = Embeddings(["a", "b"]; service=_GE, model="gemini-embedding-001")
    @test UniLM.get_url(v) == "https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-001:batchEmbedContents"
end

# gemini-embedding-2 "cannot use the task_type field": a taskType or title sent with it left its vector unchanged
# (2026-11-17), so its task is the prompt in the text, as on the OpenAI-compatible endpoint. gemini-embedding-001
# takes the task as its taskType, a document's title beside it.
# Refuted if: gemini-embedding-2 gets a taskType or a title field, gemini-embedding-001 gets a prompt in its text or
# no taskType, a field goes in embedContentConfig (which gemini-embedding-001 ignored), or a request of a
# batchEmbedContents names no model.
@testset "Gemini embeddings: the request bodies" begin
    @test _ge_body(Embeddings("hello"; service=_GE)) == Dict("content" => _ge_text("hello"))
    @test _ge_body(Embeddings("hello"; service=_GE, task=:search_query, dimensions=768)) ==
          Dict("content" => _ge_text("task: search result | query: hello"), "outputDimensionality" => 768)
    doc = EmbeddingInput("Refunds take 5 days."; title="Refund policy")
    @test _ge_body(Embeddings(doc; service=_GE, task=:document)) ==
          Dict("content" => _ge_text("title: Refund policy | text: Refunds take 5 days."))
    @test _ge_body(Embeddings(doc; service=_GE, model="gemini-embedding-001", task=:document, dimensions=128)) ==
          Dict("content" => _ge_text("Refunds take 5 days."), "taskType" => "RETRIEVAL_DOCUMENT", "title" => "Refund policy",
               "outputDimensionality" => 128)
    for (task, type) in (:search_query => "RETRIEVAL_QUERY", :question_answering => "QUESTION_ANSWERING",
                         :fact_checking => "FACT_VERIFICATION", :code_retrieval => "CODE_RETRIEVAL_QUERY",
                         :classification => "CLASSIFICATION", :clustering => "CLUSTERING",
                         :sentence_similarity => "SEMANTIC_SIMILARITY", :document => "RETRIEVAL_DOCUMENT")
        @test _ge_body(Embeddings("x"; service=_GE, model="gemini-embedding-001", task)) ==
              Dict("content" => _ge_text("x"), "taskType" => type)
    end
    @test _ge_body(Embeddings("x"; service=_GE, model="gemini-embedding-001", task=:none)) == Dict("content" => _ge_text("x"))
    @test _ge_body(Embeddings(["a", "b"]; service=_GE, model="gemini-embedding-001", task=:clustering)) == Dict("requests" => [
        Dict("model" => "models/gemini-embedding-001", "content" => _ge_text(t), "taskType" => "CLUSTERING") for t in ("a", "b")])
    @test _ge_body(Embeddings(["a", EmbeddingInput("b")]; service=_GE, task=:clustering, dimensions=64)) == Dict("requests" => [
        Dict("model" => "models/gemini-embedding-2", "content" => _ge_text("task: clustering | query: $t"),
             "outputDimensionality" => 64) for t in ("a", "b")])
    @test _ge_body(Embeddings("x"; service=_GE, encoding_format="float")) == Dict("content" => _ge_text("x"))
end

@testset "Gemini embeddings: refused before any request" begin
    # gemini-embedding-001 answers a title beside any task type but RETRIEVAL_DOCUMENT with HTTP 400.
    @test_throws ArgumentError Embeddings(EmbeddingInput("x"; title="t"); service=_GE, model="gemini-embedding-001",
                                          task=:search_query)
    # The OpenAI-compatible endpoint has no task type: there the task is refused, and the message names the endpoint
    # that takes it.
    err = try Embeddings("x"; service=GEMINIOpenAIServiceEndpoint, model="gemini-embedding-001", task=:search_query); nothing
          catch e; e end
    @test err isa ArgumentError && occursin("GEMINIServiceEndpoint", err.msg)
    @test_throws ArgumentError Embeddings("x"; service=_GE, model="text-embedding-004", task=:clustering)
    img = ImageAttachment(joinpath(@__DIR__, "fixtures", "ollama", "red_circle.png"))
    for e in (Embeddings(EmbeddingInput("a red circle", img); service=_GE), Embeddings(img; service=_GE),
              Embeddings("x"; service=_GE, user="u-1"), Embeddings(fill("w", 101); service=_GE))
        @test_throws ArgumentError UniLM._encode_embeddings(_GE, e)
        @test_throws ArgumentError embeddingrequest!(e)       # thrown before any request: no key is set here
    end
    many = try UniLM._encode_embeddings(_GE, Embeddings(fill("w", 101); service=_GE)); nothing catch e; e end
    @test occursin("at most 100", many.msg) && occursin("101", many.msg)
    @test length(_ge_body(Embeddings(fill("w", 100); service=_GE))["requests"]) == 100
end

@testset "Gemini embeddings: the replies" begin
    dec(d) = UniLM._decode_embeddings(_GE, d)
    rows, usage = dec(Dict{String,Any}("embedding" => Dict("values" => [0.1, 0.2]), "usageMetadata" => Dict(
        "promptTokenCount" => 1, "promptTokenDetails" => [Dict("modality" => "TEXT", "tokenCount" => 1)])))
    @test rows == [Dict("embedding" => [0.1, 0.2], "index" => 0)]
    @test (usage.prompt_tokens, usage.completion_tokens, usage.total_tokens) == (1, 0, 1)
    rows2, usage2 = dec(Dict{String,Any}("embeddings" => [Dict("values" => [0.1]), Dict("values" => [0.3])]))  # gemini-embedding-001
    @test [r["index"] for r in rows2] == [0, 1] && [r["embedding"] for r in rows2] == [[0.1], [0.3]] && isnothing(usage2)
    for bad in (Dict{String,Any}(), Dict{String,Any}("embedding" => Dict("values" => [0.1]), "embeddings" => []),
                Dict{String,Any}("embeddings" => Dict()), Dict{String,Any}("embeddings" => [3]),
                Dict{String,Any}("embedding" => Dict("values" => [0.1]), "usageMetadata" => Dict("promptTokenCount" => "1")),
                Dict{String,Any}("embedding" => Dict("values" => [0.1]), "usageMetadata" => []))
        @test_throws ArgumentError dec(bad)
    end
    # Priced at gemini-embedding-2's $0.20 per million text tokens, as on the OpenAI-compatible endpoint.
    r = EmbeddingSuccess(embeddings=Embeddings("x"; service=_GE), usage=TokenUsage(prompt_tokens=1000, total_tokens=1000),
                         raw=Dict{String,Any}())
    @test estimated_cost(r) ≈ 1000 * 0.20 / 1e6
end

# Gemini's embeddings wire at the scripted endpoint.
struct GeminiEmbedProbe <: UniLM.ServiceEndpoint end
UniLM._embeddings_wire(::Type{GeminiEmbedProbe}) = UniLM._GeminiEmbeddings()
UniLM.get_url(::Type{GeminiEmbedProbe}, e::Embeddings) = UniLM._gemini_embed_url(_url_probe_base[] * "/v1beta", e)
UniLM.auth_header(::Type{GeminiEmbedProbe}) = ["x-goog-api-key" => "t", "Content-Type" => "application/json"]
UniLM.provider_capabilities(::Type{GeminiEmbedProbe}) = Set([:embeddings])

# Refuted if: a request goes to another path or method, a reply's vectors are not filled in order, the usage is
# lost, or a reply short of a vector fills zeros instead of being a call error.
@testset "embeddingrequest! on Gemini's wire: one embedContent, or one batchEmbedContents, filled and priced" begin
    one = Dict("embedding" => Dict("values" => [0.5, -0.5, 0.25]),
               "usageMetadata" => Dict("promptTokenCount" => 1, "promptTokenDetails" => [Dict("modality" => "TEXT", "tokenCount" => 1)]))
    e = Embeddings("hello"; service=GeminiEmbedProbe, model="gemini-embedding-2", dimensions=3)
    r, seen = _with_scripted((_, _) -> _json(200, JSON.json(one))) do
        embeddingrequest!(e)
    end
    @test r isa EmbeddingSuccess && e.embeddings == [0.5, -0.5, 0.25] && embedding_vectors(r) === e.embeddings
    @test r.usage.prompt_tokens == 1 && estimated_cost(r) ≈ 0.20 / 1e6
    @test [(s.method, s.target) for s in seen] == [("POST", "/v1beta/models/gemini-embedding-2:embedContent")]
    @test JSON.parse(only(seen).body) == Dict("content" => _ge_text("hello"), "outputDimensionality" => 3)
    many = Dict("embeddings" => [Dict("values" => [0.1, 0.2]), Dict("values" => [0.3, 0.4])])
    v = Embeddings(["a", "b"]; service=GeminiEmbedProbe, model="gemini-embedding-001", task=:clustering)
    r2, seen2 = _with_scripted((_, _) -> _json(200, JSON.json(many))) do
        embeddingrequest!(v)
    end
    @test r2 isa EmbeddingSuccess && v.embeddings == [[0.1, 0.2], [0.3, 0.4]] && isnothing(r2.usage)
    @test [(s.method, s.target) for s in seen2] == [("POST", "/v1beta/models/gemini-embedding-001:batchEmbedContents")]
    @test [q["taskType"] for q in JSON.parse(only(seen2).body)["requests"]] == ["CLUSTERING", "CLUSTERING"]
    short = _answered(() -> embeddingrequest!(Embeddings(["a", "b"]; service=GeminiEmbedProbe, model="gemini-embedding-2")),
                      200; body=JSON.json(Dict("embeddings" => [Dict("values" => [0.1])])))
    @test short isa EmbeddingCallError && short.cause isa ArgumentError
    novec = _answered(() -> embeddingrequest!(Embeddings("a"; service=GeminiEmbedProbe, model="gemini-embedding-2")),
                      200; body="{\"embedding\":{}}")
    @test novec isa EmbeddingCallError && occursin("not a vector", novec.error)
    gone = _answered(() -> embeddingrequest!(Embeddings("a"; service=GeminiEmbedProbe, model="gemini-embedding-2")), 404;
                     body="{\"error\":{\"code\":404,\"message\":\"not found\",\"status\":\"NOT_FOUND\"}}")
    @test gone isa EmbeddingFailure && gone.status == 404
end
