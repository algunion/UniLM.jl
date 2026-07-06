# ─── Base URLs & API Key Env Var Names ────────────────────────────────────────

const OPENAI_BASE_URL::String = "https://api.openai.com"
const OPENAI_API_KEY::String = "OPENAI_API_KEY"
const AZURE_OPENAI_BASE_URL::String = "AZURE_OPENAI_BASE_URL"
const AZURE_OPENAI_API_KEY::String = "AZURE_OPENAI_API_KEY"
const AZURE_OPENAI_API_VERSION::String = "AZURE_OPENAI_API_VERSION"
const GEMINI_API_KEY::String = "GEMINI_API_KEY"

# ─── OpenAI API Endpoint Paths ────────────────────────────────────────────────
# Endpoints are determined by request type, not model name.
# Any model compatible with a given API can be used at its endpoint.

"""OpenAI Chat Completions API path."""
const CHAT_COMPLETIONS_PATH::String = "/v1/chat/completions"

"""OpenAI Embeddings API path."""
const EMBEDDINGS_PATH::String = "/v1/embeddings"

"""OpenAI Responses API path."""
const RESPONSES_PATH::String = "/v1/responses"

"""OpenAI Image Generation API path."""
const IMAGES_GENERATIONS_PATH::String = "/v1/images/generations"

"""Legacy Completions API path (used for FIM by DeepSeek, Ollama, vLLM)."""
const COMPLETIONS_PATH::String = "/v1/completions"

"""OpenAI Files API path."""
const FILES_PATH::String = "/v1/files"

"""OpenAI Vector Stores API path."""
const VECTOR_STORES_PATH::String = "/v1/vector_stores"

"""OpenAI Conversations API path."""
const CONVERSATIONS_PATH::String = "/v1/conversations"

"""OpenAI Moderations API path."""
const MODERATIONS_PATH::String = "/v1/moderations"

"""OpenAI Audio API paths."""
const AUDIO_SPEECH_PATH::String = "/v1/audio/speech"
const AUDIO_TRANSCRIPTIONS_PATH::String = "/v1/audio/transcriptions"
const AUDIO_TRANSLATIONS_PATH::String = "/v1/audio/translations"

"""OpenAI Batch API path."""
const BATCHES_PATH::String = "/v1/batches"

"""OpenAI Image Edits API path."""
const IMAGES_EDITS_PATH::String = "/v1/images/edits"

"""OpenAI Fine-tuning API path."""
const FINE_TUNING_PATH::String = "/v1/fine_tuning/jobs"

"""OpenAI Containers API path."""
const CONTAINERS_PATH::String = "/v1/containers"

"""OpenAI Uploads API path (resumable large-file uploads)."""
const UPLOADS_PATH::String = "/v1/uploads"

"""OpenAI Videos API path (Sora)."""
const VIDEOS_PATH::String = "/v1/videos"

"""OpenAI Realtime API paths."""
const REALTIME_CLIENT_SECRETS_PATH::String = "/v1/realtime/client_secrets"
const REALTIME_CALLS_PATH::String = "/v1/realtime/calls"
const REALTIME_WS_URL::String = "wss://api.openai.com/v1/realtime"

# ─── Azure OpenAI Deployment Mapping ─────────────────────────────────────────
# Azure routes requests via deployment names, so a model→deployment mapping is required.

"""
    _MODEL_ENDPOINTS_AZURE_OPENAI

Maps model names to Azure deployment paths. Populated from environment variables
`AZURE_OPENAI_DEPLOY_NAME_GPT_5_2` and `AZURE_OPENAI_DEPLOY_NAME_GPT_5_2_MINI` at
load time. Use [`add_azure_deploy_name!`](@ref) to register additional deployments.
"""
const _MODEL_ENDPOINTS_AZURE_OPENAI::Dict{String,String} = let
    d = Dict{String,String}()

    if haskey(ENV, "AZURE_OPENAI_DEPLOY_NAME_GPT_5_2")
        d["gpt-5.2"] = "/openai/deployments/" * ENV["AZURE_OPENAI_DEPLOY_NAME_GPT_5_2"]
    end



    d
end

"""
    add_azure_deploy_name!(model::String, deploy_name::String)

Register an Azure OpenAI deployment for a given model name.

# Example
```julia
add_azure_deploy_name!("gpt-5.2", "my-gpt52-deployment")
```
"""
function add_azure_deploy_name!(model::String, deploy_name::String)
    _MODEL_ENDPOINTS_AZURE_OPENAI[model] = "/openai/deployments/" * deploy_name
end

# ─── Gemini (OpenAI-compatible) ───────────────────────────────────────────────

"""DeepSeek API base URL."""
const DEEPSEEK_BASE_URL::String = "https://api.deepseek.com"

"""DeepSeek beta API base URL (required for FIM and prefix completion)."""
const DEEPSEEK_BETA_BASE_URL::String = "https://api.deepseek.com/beta"

"""Google Gemini OpenAI-compatible base URL (replaces `https://api.openai.com` in paths)."""
const GEMINI_OPENAI_BASE::String = "https://generativelanguage.googleapis.com/v1beta/openai"

"""Google Gemini chat completions URL (OpenAI-compatible endpoint).
Note: Gemini uses `/chat/completions` without the `/v1` prefix."""
const GEMINI_CHAT_URL::String = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"

# ─── Anthropic (Claude) native Messages API ──────────────────────────────────

"""Anthropic API key env var name."""
const ANTHROPIC_API_KEY::String = "ANTHROPIC_API_KEY"

"""Anthropic API base URL."""
const ANTHROPIC_BASE_URL::String = "https://api.anthropic.com"

"""Anthropic Messages API path."""
const ANTHROPIC_MESSAGES_PATH::String = "/v1/messages"

"""Required `anthropic-version` request header value (stable since 2024)."""
const ANTHROPIC_VERSION::String = "2023-06-01"

"""Moderate, overridable default for Anthropic's REQUIRED `max_tokens` when the
caller leaves it unset. Not the model ceiling — a ceiling-sized cap invites
runaway output; unused headroom is not billed. Raise `max_tokens` explicitly for
long generations."""
const _ANTHROPIC_DEFAULT_MAX_TOKENS::Int = 4096