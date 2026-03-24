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

"""Google Gemini OpenAI-compatible base URL (replaces `https://api.openai.com` in paths)."""
const GEMINI_OPENAI_BASE::String = "https://generativelanguage.googleapis.com/v1beta/openai"

"""Google Gemini chat completions URL (OpenAI-compatible endpoint).
Note: Gemini uses `/chat/completions` without the `/v1` prefix."""
const GEMINI_CHAT_URL::String = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"