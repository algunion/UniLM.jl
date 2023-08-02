module UniLM
using HTTP
using JSON3
using StructTypes
using Accessors

include("constants.jl")
include("exceptions.jl")
include("helpers.jl")
include("jsonschema.jl")
include("openai-api.jl")
include("functioncall.jl")
include("requests.jl")
include("testutils.jl")

export
    Chat,
    Message,
    GPTSystem,
    GPTUser,
    GPTAssistant,
    GPTFunction,
    JsonSchema,
    JsonArray,
    JsonObject,
    JsonBoolean,
    JsonNull,
    JsonNumber,
    JsonString,
    JsonInteger,
    GPTFunctionSignature,
    withdescription,
    GPTFunctionCallResult,
    InvalidConversationError,
    issendvalid,
    chatrequest!,
    Embedding,
    embeddingrequest!,
    update!
end
