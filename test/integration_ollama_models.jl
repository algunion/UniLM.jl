# ─── Ollama model management live witnesses ──────────────────────────────────
# Requires UNILM_LIVE_OLLAMA=1 and a reachable Ollama server (OLLAMA_HOST, else
# 127.0.0.1:11434) with the embedding model UNILM_OLLAMA_EMBED_MODEL (default
# embeddinggemma) and the decision model kev:0.8b-q8_0 installed; no chat model is needed.
# Live, nothing is skipped for want of a server or a model: a server that does not list its
# models is an error, and a witness whose model is not installed fails, naming the model and
# the fix. The pull and delete witness downloads UNILM_OLLAMA_SCRATCH_MODEL (default
# all-minilm:22m, about 46 MB) and deletes it again; it is skipped when that model is
# installed already, so it never deletes a model it did not pull. Local models: no spend.
# Each witness names the outcome that would refute it.

const _OMM_EMBED = get(ENV, "UNILM_OLLAMA_EMBED_MODEL", "embeddinggemma")
const _OMM_SCRATCH = get(ENV, "UNILM_OLLAMA_SCRATCH_MODEL", "all-minilm:22m")
const _OMM_DECISION = "kev:0.8b-q8_0"
const _OMM_RUNNERS = (:mlx, :llamacpp, :ggml)

# Built only when the witnesses run: an unusual OLLAMA_HOST must not break the offline suite.
const _OMM_EP = get(ENV, "UNILM_LIVE_OLLAMA", "") == "1" ? OllamaEndpoint() : nothing
function _omm_installed()::Set{String}
    r = list_models(; service=_OMM_EP, config=RequestConfig(max_attempts=1, request_timeout=5.0))
    r isa OllamaSuccess || error("Ollama model management live witnesses: listing the installed models failed: $(r)")
    Set(m.name for m in r.response)
end
const _OMM_TAGS = isnothing(_OMM_EP) ? nothing : _omm_installed()

# The installed name a model name stands for (a bare name means its `latest` tag), or nothing.
_omm_tag(tags, m) = m in tags ? m : (m * ":latest") in tags ? m * ":latest" : nothing
# The installed name of a model a witness needs; a model that is not installed is an error naming it and the fix.
function _omm_need(m::String, fix::String)::String
    tag = _omm_tag(_OMM_TAGS, m)
    isnothing(tag) ? error("Ollama model management live witnesses: $m is not installed; $fix") : tag
end
const _OMM_EMBED_FIX = "pull it or set UNILM_OLLAMA_EMBED_MODEL to an installed embedding model"
# The server's loaded models: name => context length.
_omm_ps() = Dict(m.name => m.context_length for m in running_models(; service=_OMM_EP).response)

if isnothing(_OMM_EP)
    @info "Skipping Ollama model-management live witnesses (set UNILM_LIVE_OLLAMA=1, run an Ollama server and " *
          "pull $_OMM_EMBED)"
else

@testset "Ollama model management live — $_OMM_EMBED" begin

@testset "listing and metadata of an embedding model" begin
    # Refuted by: the embedding model listed without :embedding, a listing and model_info that disagree on its
    # capabilities or format, a runner or format Ollama does not name, or a missing model that is not a 404.
    tag = _omm_need(_OMM_EMBED, _OMM_EMBED_FIX)
    listed = only(m for m in list_models(; service=_OMM_EP).response if m.name == tag)
    @test :embedding in listed.capabilities && listed.runner in _OMM_RUNNERS
    info = model_info(_OMM_EMBED; service=_OMM_EP)
    @test info isa OllamaSuccess{OllamaModelInfo}
    @test info.response.capabilities == listed.capabilities && info.response.format == listed.format
    @test info.response.format in (:gguf, :safetensors)
    # /api/show names no runner for some models (embeddinggemma:latest on Ollama 0.40.6): nothing, never a guess.
    @test isnothing(info.response.runner) || info.response.runner in _OMM_RUNNERS
    missing_model = model_info("no-such:tag"; service=_OMM_EP)
    @test missing_model isa OllamaFailure && missing_model.status == 404
end

@testset "an embedding model and a decision model load and unload" begin
    # Refuted by: a load refused for a model that cannot chat, a cold load whose /api/ps context is not the
    # endpoint's num_ctx, a runner /api/ps does not name, or a model still loaded 10 s after its unload.
    tag = _omm_need(_OMM_EMBED, _OMM_EMBED_FIX)
    unload_model(_OMM_EMBED; service=_OMM_EP)                       # cold: a loaded model may keep its context
    @test timedwait(() -> !haskey(_omm_ps(), tag), 10.0) == :ok
    small = OllamaEndpoint(num_ctx=1024)
    @test load_model(_OMM_EMBED; service=small) isa OllamaSuccess{Nothing}
    loaded = only(m for m in running_models(; service=_OMM_EP).response if m.name == tag)
    @test loaded.context_length == 1024 && loaded.runner in _OMM_RUNNERS
    @test unload_model(_OMM_EMBED; service=small) isa OllamaSuccess{Nothing}
    @test timedwait(() -> !haskey(_omm_ps(), tag), 10.0) == :ok
    dtag = _omm_need(_OMM_DECISION, "pull it")
    unload_model(dtag; service=_OMM_EP)
    @test timedwait(() -> !haskey(_omm_ps(), dtag), 10.0) == :ok
    @test load_model(dtag; service=_OMM_EP) isa OllamaSuccess{Nothing}
    @test haskey(_omm_ps(), dtag)
    @test unload_model(dtag; service=_OMM_EP) isa OllamaSuccess{Nothing}
    @test timedwait(() -> !haskey(_omm_ps(), dtag), 10.0) == :ok
end

@testset "a runner the model has no variant for: model_info and load_model get Ollama's 404" begin
    # Refuted by: a show or a load through an endpoint pinned to a runner the model is not published for
    # answering 200 (a silent fallback), or loading the model. kev:0.8b-q8_0 is published for llama.cpp only.
    dtag = _omm_need(_OMM_DECISION, "pull it")
    other = OllamaEndpoint(runner=:mlx)
    i = model_info(dtag; service=other)
    @test i isa OllamaFailure && i.status == 404 && occursin("no compatible manifest found for runners: mlx", i.response)
    l = load_model(dtag; service=other)
    @test l isa OllamaFailure && l.status == 404 && !haskey(_omm_ps(), dtag)
    @test model_info(dtag; service=OllamaEndpoint(runner=:llamacpp)) isa OllamaSuccess{OllamaModelInfo}
end

@testset "pull and delete a scratch model" begin
    # Refuted by: a pull whose progress does not run from "pulling manifest" to "success", a delete that fails or
    # leaves the model listed, a second delete that is not a 404, or an unknown tag pulled as a success.
    if !isnothing(_omm_tag(_omm_installed(), _OMM_SCRATCH))
        @info "Skipping the pull and delete witness: $_OMM_SCRATCH is installed already, and the witness deletes " *
              "only a model it pulled"
    else
        seen = OllamaPullProgress[]
        pulled = pull_model(_OMM_SCRATCH; service=_OMM_EP, progress=p -> push!(seen, p))
        # Deleted at once, and only after its own successful pull: no failed check below can leave it installed.
        deleted = pulled isa OllamaSuccess ? delete_model(_OMM_SCRATCH; service=_OMM_EP) : nothing
        @test pulled isa OllamaSuccess{Nothing}
        @test !isempty(seen) && first(seen).status == "pulling manifest" && last(seen).status == "success"
        @test deleted isa OllamaSuccess{Nothing}
        @test isnothing(_omm_tag(_omm_installed(), _OMM_SCRATCH))
        again = delete_model(_OMM_SCRATCH; service=_OMM_EP)
        @test again isa OllamaFailure && again.status == 404
    end
    @test pull_model("gemma4:no-such-tag"; service=_OMM_EP) isa OllamaCallError
end

end # Ollama model management live
end # guard
