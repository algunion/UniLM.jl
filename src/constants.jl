# FROM: https://platform.openai.com/docs/models/model-endpoint-compatibility
# /v1/chat/completions	gpt-4, gpt-4-0613, gpt-4-32k, gpt-4-32k-0613, gpt-3.5-turbo, gpt-3.5-turbo-0613, gpt-3.5-turbo-16k, gpt-3.5-turbo-16k-0613
# /v1/completions	text-davinci-003, text-davinci-002, text-curie-001, text-babbage-001, text-ada-001
# /v1/edits	text-davinci-edit-001, code-davinci-edit-001
# /v1/audio/transcriptions	whisper-1
# /v1/audio/translations	whisper-1
# /v1/fine-tunes	davinci, curie, babbage, ada
# /v1/embeddings	text-embedding-ada-002, text-search-ada-doc-001
# /v1/moderations	text-moderation-stable, text-moderation-latest

const OPENAI_API_KEY =
    let
        @info "Loading OpenAI API key from environment variable OPENAI_API_KEY"
        get(ENV, "OPENAI_API_KEY", nothing)
    end

const OPENAI_BASE_URL = "https://api.openai.com"



const _CHAT_COMPLETIONS_MODELS = [
    "gpt-4",
    "gpt-4-1106-preview",
    "gpt-4-vision-preview",
    "gpt-4-0613",
    "gpt-4-32k",
    "gpt-4-32k-0613",
    "gpt-3.5-turbo-1106",
    "gpt-3.5-turbo",
    "gpt-3.5-turbo-0613",
    "gpt-3.5-turbo-16k",
    "gpt-3.5-turbo-16k-0613"
]

const _COMPLETIONS_MODELS = [
    "text-davinci-003",
    "text-davinci-002",
    "text-curie-001",
    "text-babbage-001",
    "text-ada-001"
]

const _EDITS_MODELS = [
    "text-davinci-edit-001",
    "code-davinci-edit-001"
]

const _AUDIO_TTS_MODELS = [
    "tts-1",
    "tts-1-hd"
]

const _AUDIO_TRANSCRIPTIONS_MODELS = [
    "whisper-1"
]

const _AUDIO_TRANSLATIONS_MODELS = [
    "whisper-1"
]

const _FINE_TUNES_MODELS = [
    "davinci",
    "curie",
    "babbage",
    "ada"
]

const _EMBEDDINGS_MODELS = [
    "text-embedding-ada-002",
    "text-search-ada-doc-001"
]

const _MODERATIONS_MODELS = [
    "text-moderation-stable",
    "text-moderation-latest"
]

const _ENDPOINTS = [
    "/v1/chat/completions",
    "/v1/completions",
    "/v1/edits",
    "/v1/audio/speech",
    "/v1/audio/transcriptions",
    "/v1/audio/translations",
    "/v1/fine-tunes",
    "/v1/embeddings",
    "/v1/moderations"
]

# dict of model names to endpoint names
const _MODEL_ENDPOINTS = Dict(
    "gpt-4-1106-preview" => "/v1/chat/completions",
    "gpt-4-vision-preview" => "/v1/chat/completions",
    "gpt-4" => "/v1/chat/completions",
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
