module UniLM
using HTTP
using JSON3
using StructTypes
using Accessors

include("constants.jl")
include("exceptions.jl")
include("api.jl")
include("requests.jl")

export
    Chat,
    Message,
    RoleSystem,
    RoleUser,
    RoleAssistant,
    GPTTool,
    GPTToolCall,
    GPTFunctionSignature,
    GPTFunctionCallResult,
    InvalidConversationError,
    issendvalid,
    chatrequest!,
    Embeddings,
    embeddingrequest!,
    update!,
    ResponseFormat,
    LLMRequestResponse,
    LLMSuccess,
    LLMFailure,
    LLMCallError
end
