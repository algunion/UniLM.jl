module UniLM
using HTTP
using JSON3
using StructTypes
using Accessors

include("constants.jl")
include("exceptions.jl")
include("helpers.jl")
include("functioncall.jl")
include("openai-api.jl")
include("metaprog.jl")
include("requests.jl")
include("testutils.jl")

export
    Chat,
    Message,
    GPTSystem,
    GPTUser,
    GPTAssistant,
    GPTFunction,
    GPTFunctionSignature,
    GPTFunctionCallResult,
    InvalidConversationError,
    issendvalid,
    chatrequest!,
    Embedding,
    embeddingrequest!,
    update!
end
