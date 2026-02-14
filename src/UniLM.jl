module UniLM
using HTTP
using JSON

include("constants.jl")
include("exceptions.jl")
include("api.jl")
include("requests.jl")
include("responses.jl")

# ─── Chat Completions API ─────────────────────────────────────────────────────
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

# ─── Responses API ────────────────────────────────────────────────────────────
export
    # Core types
    Respond,
    InputMessage,
    ResponseTool,
    FunctionTool,
    WebSearchTool,
    FileSearchTool,
    TextConfig,
    TextFormatSpec,
    Reasoning,
    ResponseObject,
    # Result types
    ResponseSuccess,
    ResponseFailure,
    ResponseCallError,
    # Request functions
    respond,
    get_response,
    delete_response,
    list_input_items,
    # Accessor functions
    output_text,
    function_calls,
    # Input helpers
    input_text,
    input_image,
    input_file,
    # Tool constructors
    function_tool,
    web_search,
    file_search,
    # Format constructors
    text_format,
    json_schema_format,
    json_object_format

end
