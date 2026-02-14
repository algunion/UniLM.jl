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

# ─── Azure OpenAI Deployment Mapping ─────────────────────────────────────────
# Azure routes requests via deployment names, so a model→deployment mapping is required.

"""
    _MODEL_ENDPOINTS_AZURE_OPENAI

Maps model names to Azure deployment paths. Populated from environment variables
`AZURE_OPENAI_DEPLOY_NAME_GPT_4O` and `AZURE_OPENAI_DEPLOY_NAME_GPT_4O_MINI` at
load time. Use [`add_azure_deploy_name!`](@ref) to register additional deployments.
"""
const _MODEL_ENDPOINTS_AZURE_OPENAI::Dict{String,String} = let
    d = Dict{String,String}()

    if haskey(ENV, "AZURE_OPENAI_DEPLOY_NAME_GPT_4O")
        d["gpt-4o"] = "/openai/deployments/" * ENV["AZURE_OPENAI_DEPLOY_NAME_GPT_4O"]
    end

    if haskey(ENV, "AZURE_OPENAI_DEPLOY_NAME_GPT_4O_MINI")
        d["gpt-4o-mini"] = "/openai/deployments/" * ENV["AZURE_OPENAI_DEPLOY_NAME_GPT_4O_MINI"]
    end

    d
end

"""
    add_azure_deploy_name!(model::String, deploy_name::String)

Register an Azure OpenAI deployment for a given model name.

# Example
```julia
add_azure_deploy_name!("gpt-4.1", "my-gpt41-deployment")
```
"""
function add_azure_deploy_name!(model::String, deploy_name::String)
    _MODEL_ENDPOINTS_AZURE_OPENAI[model] = "/openai/deployments/" * deploy_name
end

# ─── Gemini (OpenAI-compatible) ───────────────────────────────────────────────

"""Google Gemini chat completions URL (OpenAI-compatible endpoint)."""
const GEMINI_CHAT_URL::String = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"