module UniLM
using HTTP
using JSON3
using StructTypes
using Accessors

import MethodSchemas

include("constants.jl")
include("exceptions.jl")
include("helpers.jl")
include("openai-api.jl")
include("convertors.jl")
include("functioncall.jl")
include("requests.jl")
include("testutils.jl")

export
    Chat,
    Message,
    RoleSystem,
    RoleUser,
    RoleAssistant,
    RoleFunction,
    GPTTool,
    GPTToolCall,
    JsonSchema,
    JsonArray,
    JsonObject,
    JsonBoolean,
    JsonNull,
    JsonNumber,
    JsonString,
    JsonInteger,
    JsonAny,
    GPTFunctionSignature,
    withdescription,
    GPTFunctionCallResult,
    InvalidConversationError,
    issendvalid,
    chatrequest!,
    Embeddings,
    embeddingrequest!,
    update!,
    MethodSchemas
end
