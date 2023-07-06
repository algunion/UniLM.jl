# /v1/chat/completions	gpt-4, gpt-4-0613, gpt-4-32k, gpt-4-32k-0613, gpt-3.5-turbo, gpt-3.5-turbo-0613, gpt-3.5-turbo-16k, gpt-3.5-turbo-16k-0613
# /v1/completions	text-davinci-003, text-davinci-002, text-curie-001, text-babbage-001, text-ada-001
# /v1/edits	text-davinci-edit-001, code-davinci-edit-001
# /v1/audio/transcriptions	whisper-1
# /v1/audio/translations	whisper-1
# /v1/fine-tunes	davinci, curie, babbage, ada
# /v1/embeddings	text-embedding-ada-002, text-search-ada-doc-001
# /v1/moderations	text-moderation-stable, text-moderation-latest

const OPENAI_API_KEY::String = ENV["OPENAI_API_KEY"]

const _chat_completions = [
    "gpt-4",
    "gpt-4-0613",
    "gpt-4-32k",
    "gpt-4-32k-0613",
    "gpt-3.5-turbo",
    "gpt-3.5-turbo-0613",
    "gpt-3.5-turbo-16k",
    "gpt-3.5-turbo-16k-0613"
]

const _chat_completions_syn = [
    :GPT4,
    :GPT40613,
    :GPT432k,
    :GPT432k0613,
    :GPT35turbo,
    :GPT35turbo0613,
    :GPT35turbo16k,
    :GPT35turbo16k0613
]

const _chat_complettions_dict_syn = Dict(
    :GPT4 => "gpt-4",
    :GPT40613 => "gpt-4-0613",
    :GPT432k => "gpt-4-32k",
    :GPT432k0613 => "gpt-4-32k-0613",
    :GPT35turbo => "gpt-3.5-turbo",
    :GPT35turbo0613 => "gpt-3.5-turbo-0613",
    :GPT35turbo16k => "gpt-3.5-turbo-16k",
    :GPT35turbo16k0613 => "gpt-3.5-turbo-16k-0613"    
)

const _completions = [
    "text-davinci-003",
    "text-davinci-002",
    "text-curie-001",
    "text-babbage-001",
    "text-ada-001"
]

const _edits = [
    "text-davinci-edit-001",
    "code-davinci-edit-001"
]

const _audio_transcriptions = [
    "whisper-1"
]

const _audio_translations = [
    "whisper-1"
]

const _fine_tunes = [
    "davinci",
    "curie",
    "babbage",
    "ada"
]

const _embeddings = [
    "text-embedding-ada-002",
    "text-search-ada-doc-001"
]

const _moderations = [
    "text-moderation-stable",
    "text-moderation-latest"
]

const _endpoints = [ 
    "/v1/chat/completions",
    "/v1/completions",
    "/v1/edits",
    "/v1/audio/transcriptions",
    "/v1/audio/translations",
    "/v1/fine-tunes",
    "/v1/embeddings",
    "/v1/moderations"    
]

# sintactic sugar helper
const _endpoints_syn = [
    :chat,
    :completions,
    :edits,
    :transcriptions,
    :translations,
    :finetunes,
    :embeddings,
    :moderations
]

const _endpoints_dict_syn = Dict(
    :chat => "/v1/chat/completions",
    :completions => "/v1/completions",
    :edits => "/v1/edits",
    :transcriptions => "/v1/audio/transcriptions",
    :translations => "/v1/audio/translations",
    :finetunes => "/v1/fine-tunes",
    :embeddings => "/v1/embeddings",
    :moderations => "/v1/moderations"
)








