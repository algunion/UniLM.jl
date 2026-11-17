# ============================================================================
# Gemini embeddings — embeddingrequest! on the Gemini API's own embedding methods
# (https://ai.google.dev/api/embeddings): models.embedContent for one input, models.batchEmbedContents for
# several. Each shape below is one the API returned on 2026-11-17.
# ============================================================================

_embeddings_wire(::Type{GEMINIServiceEndpoint}) = _GeminiEmbeddings()

# The model is a path segment; the method follows it: one input is one embedContent, a vector one
# batchEmbedContents.
_gemini_embed_url(base::String, emb::Embeddings)::String =
    base * "/models/" * _uripart(emb.model) * (emb.input isa AbstractVector ? ":batchEmbedContents" : ":embedContent")
get_url(::Type{GEMINIServiceEndpoint}, emb::Embeddings)::String = _gemini_embed_url(GEMINI_NATIVE_BASE, emb)

# "at most 100 requests can be in one batch" (the API's HTTP 400 to 101 of them, 2026-11-17).
const _GEMINI_EMBED_BATCH_MAX = 100

# Task types (EmbedContentRequest.taskType) by the task each names.
const _GEMINI_TASK_TYPES = Dict{Symbol,String}(:search_query => "RETRIEVAL_QUERY", :document => "RETRIEVAL_DOCUMENT",
    :question_answering => "QUESTION_ANSWERING", :fact_checking => "FACT_VERIFICATION",
    :code_retrieval => "CODE_RETRIEVAL_QUERY", :classification => "CLASSIFICATION", :clustering => "CLUSTERING",
    :sentence_similarity => "SEMANTIC_SIMILARITY")

# The text an item is sent as. UniLM sends Gemini text alone: an image or a clip is refused, as on the
# OpenAI-compatible endpoint.
_gemini_embed_text(::Embeddings, x::String) = x
function _gemini_embed_text(emb::Embeddings, x::EmbeddingInput)
    _item_media(x) && throw(ArgumentError(
        "embeddingrequest!: UniLM sends text to Gemini's embedContent (service: $(_service_name(emb.service))): " *
        "send images and sound clips with embeddingrequest! to an OllamaEndpoint, or leave them out"))
    x.text::String
end

# One EmbedContentRequest. A task reaches the model as its task type on gemini-embedding-001, with a
# document's title beside it, and as the task prompt in the text on the models that read one. Its fields go
# at the request's top level: the API documents them as moved into `embedContentConfig`, which
# gemini-embedding-001 ignored (a task type, a title and outputDimensionality 128 there gave its default
# 3072-dimension vector, 2026-11-17), while both models honour them at the top.
function _gemini_embed_request(emb::Embeddings, x::Union{String,EmbeddingInput}; model::Bool=false)::Dict{String,Any}
    text = _gemini_embed_text(emb, x)
    title = x isa EmbeddingInput ? x.title : nothing
    d = Dict{String,Any}()
    model && (d["model"] = "models/" * emb.model)
    if _embedding_task_family(emb.service, emb.model) === :gemini_task_type
        d["content"] = Dict("parts" => [Dict("text" => text)])
        if !isnothing(emb.task) && emb.task !== :none
            d["taskType"] = _GEMINI_TASK_TYPES[emb.task]
            isnothing(title) || (d["title"] = title)
        end
    else
        d["content"] = Dict("parts" => [Dict("text" => _prompted(emb.task, text, title))])
    end
    isnothing(emb.dimensions) || (d["outputDimensionality"] = emb.dimensions)
    d
end

# Options of the OpenAI Embeddings API that Gemini's has no field for. "float", the one encoding a result
# holds, is what Gemini returns.
function _gemini_embed_options(emb::Embeddings)::Nothing
    isnothing(emb.user) || throw(ArgumentError(
        "embeddingrequest!: Gemini's embedContent takes no end-user id (user); unset user"))
    nothing
end

function _encode_embeddings(::_GeminiEmbeddings, _, emb::Embeddings)::String
    _gemini_embed_options(emb)
    emb.input isa AbstractVector || return JSON.json(_gemini_embed_request(emb, emb.input))
    n = length(emb.input)
    n <= _GEMINI_EMBED_BATCH_MAX || throw(ArgumentError("embeddingrequest!: Gemini's batchEmbedContents takes at " *
        "most $_GEMINI_EMBED_BATCH_MAX inputs (got $n): send them in requests of $_GEMINI_EMBED_BATCH_MAX, or as " *
        "a batch (create_batch)"))
    # Every request of a batchEmbedContents names its model ("model is not specified" otherwise).
    JSON.json(Dict("requests" => [_gemini_embed_request(emb, x; model=true) for x in emb.input]))
end

# The vector of a ContentEmbedding, as the OpenAI-shaped row the fill reads: `values`, which the fill refuses
# when it is no vector of finite numbers.
_gemini_embedding_row(e::AbstractDict, i::Int) = Dict{String,Any}("embedding" => get(e, "values", nothing), "index" => i)
_gemini_embedding_row(e, i::Int) =
    throw(ArgumentError("the Gemini embeddings reply's row $i is a JSON $(_json_type(e)), not a ContentEmbedding"))

# The usage of an embedding: `usageMetadata.promptTokenCount`, the input's tokens (proto3 JSON omits a zero).
# gemini-embedding-001 reports none (2026-11-17), and its estimate then says so.
function _gemini_embed_usage(d::AbstractDict)::Union{TokenUsage,Nothing}
    u = get(d, "usageMetadata", nothing)
    isnothing(u) && return nothing
    u isa AbstractDict || throw(ArgumentError("the Gemini embeddings reply's usageMetadata is a JSON $(_json_type(u))"))
    n = get(u, "promptTokenCount", 0)
    n isa Integer && !(n isa Bool) && n >= 0 || throw(ArgumentError(
        "the Gemini embeddings reply's promptTokenCount is $(repr(n)), not a token count"))
    TokenUsage(prompt_tokens=n, total_tokens=n)
end

# An EmbedContentResponse holds one `embedding`, a BatchEmbedContentsResponse its `embeddings` in the order of
# the requests; a reply with neither (or both) is no embedding.
function _decode_embeddings(::_GeminiEmbeddings, _, data::Dict{String,Any})
    one, many = get(data, "embedding", nothing), get(data, "embeddings", nothing)
    isnothing(one) == isnothing(many) && throw(ArgumentError("the Gemini embeddings reply holds " *
        (isnothing(one) ? "neither an embedding nor embeddings" : "both an embedding and embeddings")))
    many isa Union{Nothing,AbstractVector} || throw(ArgumentError(
        "the Gemini embeddings reply's embeddings is a JSON $(_json_type(many)), not a list"))
    rows = isnothing(one) ? [_gemini_embedding_row(e, i - 1) for (i, e) in enumerate(many)] : [_gemini_embedding_row(one, 0)]
    rows, _gemini_embed_usage(data)
end
