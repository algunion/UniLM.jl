# ─── Ollama model management live witnesses ──────────────────────────────────
# Requires UNILM_LIVE_OLLAMA=1 and a reachable Ollama server (OLLAMA_HOST, else
# 127.0.0.1:11434) with the embedding model UNILM_OLLAMA_EMBED_MODEL (default
# embeddinggemma); no chat model is needed. The pull and delete witness downloads
# UNILM_OLLAMA_SCRATCH_MODEL (default all-minilm:22m, about 46 MB) and deletes it again; it
# is skipped when that model is installed already, so it never deletes a model it did not
# pull. The decision-model load witness runs when kev:0.8b-q8_0 is installed. Local models:
# no spend. Each witness names the outcome that would refute it.

const _OMM_EMBED = get(ENV, "UNILM_OLLAMA_EMBED_MODEL", "embeddinggemma")
const _OMM_SCRATCH = get(ENV, "UNILM_OLLAMA_SCRATCH_MODEL", "all-minilm:22m")
const _OMM_DECISION = "kev:0.8b-q8_0"
const _OMM_RUNNERS = (:mlx, :llamacpp, :ggml)

# Built only when the witnesses run: an unusual OLLAMA_HOST must not break the offline suite.
const _OMM_EP = get(ENV, "UNILM_LIVE_OLLAMA", "") == "1" ? OllamaEndpoint() : nothing
_omm_installed() = (r = list_models(; service=_OMM_EP, config=RequestConfig(max_attempts=1, request_timeout=5.0));
                    r isa OllamaSuccess ? Set(m.name for m in r.response) : nothing)
const _OMM_TAGS = isnothing(_OMM_EP) ? nothing : _omm_installed()

# The installed name a model name stands for (a bare name means its `latest` tag), or nothing.
_omm_tag(tags, m) = m in tags ? m : (m * ":latest") in tags ? m * ":latest" : nothing
# The server's loaded models: name => context length.
_omm_ps() = Dict(m.name => m.context_length for m in running_models(; service=_OMM_EP).response)

if isnothing(_OMM_TAGS) || isnothing(_omm_tag(_OMM_TAGS, _OMM_EMBED))
    @info "Skipping Ollama model-management live witnesses (set UNILM_LIVE_OLLAMA=1, run an Ollama server and " *
          "pull $_OMM_EMBED)"
else

@testset "Ollama model management live — $_OMM_EMBED" begin

@testset "listing and metadata of an embedding model" begin
    # Refuted by: the embedding model listed without :embedding, a listing and model_info that disagree on its
    # capabilities or format, a runner or format Ollama does not name, or a missing model that is not a 404.
    tag = _omm_tag(_OMM_TAGS, _OMM_EMBED)
    listed = only(m for m in list_models(; service=_OMM_EP).response if m.name == tag)
    @test :embedding in listed.capabilities && listed.runner in _OMM_RUNNERS
    info = model_info(_OMM_EMBED; service=_OMM_EP)
    @test info isa OllamaSuccess{OllamaModelInfo}
    @test info.response.capabilities == listed.capabilities && info.response.format == listed.format
    @test info.response.format in ("gguf", "safetensors")
    # /api/show names no runner for some models (embeddinggemma:latest on Ollama 0.40.6): nothing, never a guess.
    @test isnothing(info.response.runner) || info.response.runner in _OMM_RUNNERS
    missing_model = model_info("no-such:tag"; service=_OMM_EP)
    @test missing_model isa OllamaFailure && missing_model.status == 404
end

@testset "an embedding model and a decision model load and unload" begin
    # Refuted by: a load refused for a model that cannot chat, a cold load whose /api/ps context is not the
    # endpoint's num_ctx, a runner /api/ps does not name, or a model still loaded 10 s after its unload.
    tag = _omm_tag(_OMM_TAGS, _OMM_EMBED)
    unload_model(_OMM_EMBED; service=_OMM_EP)                       # cold: a loaded model may keep its context
    @test timedwait(() -> !haskey(_omm_ps(), tag), 10.0) == :ok
    small = OllamaEndpoint(num_ctx=1024)
    @test load_model(_OMM_EMBED; service=small) isa OllamaSuccess{Nothing}
    loaded = only(m for m in running_models(; service=_OMM_EP).response if m.name == tag)
    @test loaded.context_length == 1024 && loaded.runner in _OMM_RUNNERS
    @test unload_model(_OMM_EMBED; service=small) isa OllamaSuccess{Nothing}
    @test timedwait(() -> !haskey(_omm_ps(), tag), 10.0) == :ok
    if _OMM_DECISION in _OMM_TAGS
        unload_model(_OMM_DECISION; service=_OMM_EP)
        @test timedwait(() -> !haskey(_omm_ps(), _OMM_DECISION), 10.0) == :ok
        @test load_model(_OMM_DECISION; service=_OMM_EP) isa OllamaSuccess{Nothing}
        @test haskey(_omm_ps(), _OMM_DECISION)
        @test unload_model(_OMM_DECISION; service=_OMM_EP) isa OllamaSuccess{Nothing}
        @test timedwait(() -> !haskey(_omm_ps(), _OMM_DECISION), 10.0) == :ok
    else
        @info "Skipping the decision-model load witness: $_OMM_DECISION is not installed"
    end
end

@testset "pull and delete a scratch model" begin
    # Refuted by: a pull whose progress does not run from "pulling manifest" to "success", a delete that fails or
    # leaves the model listed, a second delete that is not a 404, or an unknown tag pulled as a success.
    installed = _omm_installed()
    if isnothing(installed) || !isnothing(_omm_tag(installed, _OMM_SCRATCH))
        @info "Skipping the pull and delete witness: $_OMM_SCRATCH is installed already (or the server did not " *
              "list its models), and the witness deletes only a model it pulled"
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
