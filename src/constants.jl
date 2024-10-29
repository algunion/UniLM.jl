

const OPENAI_API_KEY::String =
    let
        get(ENV, "OPENAI_API_KEY", "")
    end

const AZURE_OPENAI_API_KEY::String =
    let
        get(ENV, "AZURE_OPENAI_API_KEY", "")
    end

const OPENAI_BASE_URL::String = "https://api.openai.com"

const AZURE_OPENAI_BASE_URL::String =
    let
        get(ENV, "AZURE_OPENAI_BASE_URL", "")
    end

const AZURE_OPENAI_DEPLOY_NAME::String =
    let
        get(ENV, "AZURE_OPENAI_DEPLOY_NAME", "")
    end

"""
    Convenience mapping of model names to their respective endpoints.
    
"""
const _MODEL_ENDPOINTS_OPENAI::Dict{String,String} = Dict(
    "gpt-4-1106-preview" => "/v1/chat/completions",
    "gpt-4-vision-preview" => "/v1/chat/completions",
    "gpt-4" => "/v1/chat/completions",
    "gpt-4o" => "/v1/chat/completions",
    "gpt-4o-2024-08-06" => "/v1/chat/completions",
    "gpt-4o-mini" => "/v1/chat/completions",
    "gpt-4-0613" => "/v1/chat/completions",
    "gpt-4-32k" => "/v1/chat/completions",
    "gpt-4-32k-0613" => "/v1/chat/completions",
    "gpt-3.5-turbo" => "/v1/chat/completions",
    "gpt-3.5-turbo-0613" => "/v1/chat/completions",
    "gpt-3.5-turbo-16k" => "/v1/chat/completions",
    "gpt-3.5-turbo-16k-0613" => "/v1/chat/completions",
    "text-davinci-003" => "/v1/completions",
    "text-davinci-002" => "/v1/completions",
    "text-curie-001" => "/v1/completions",
    "text-babbage-001" => "/v1/completions",
    "text-ada-001" => "/v1/completions",
    "text-davinci-edit-001" => "/v1/edits",
    "code-davinci-edit-001" => "/v1/edits",
    "whisper-1" => "/v1/audio/transcriptions",
    "whisper-1" => "/v1/audio/translations",
    "davinci" => "/v1/fine-tunes",
    "curie" => "/v1/fine-tunes",
    "babbage" => "/v1/fine-tunes",
    "ada" => "/v1/fine-tunes",
    "text-embedding-ada-002" => "/v1/embeddings",
    "text-search-ada-doc-001" => "/v1/embeddings",
    "text-moderation-stable" => "/v1/moderations",
    "text-moderation-latest" => "/v1/moderations"
)