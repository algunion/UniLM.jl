module UniLM
using HTTP
using JSON3
using StructTypes
using Accessors
using DotEnv

DotEnv.load!()

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
    update!
end
