"""
    UniLM

A Julia interface for OpenAI's language models, supporting both the **Chat Completions API**
and the newer **Responses API**.

UniLM.jl provides:
- **Chat Completions** via [`Chat`](@ref) and [`chatrequest!`](@ref) — stateful conversations
  with tool calling, streaming, and structured output.
- **Responses API** via [`Respond`](@ref) and [`respond`](@ref) — the newer, more flexible
  API with built-in tools (web search, file search), multi-turn chaining via `previous_response_id`,
  and reasoning support for O-series models.
- **Embeddings** via [`Embeddings`](@ref) and [`embeddingrequest!`](@ref).
- **Multi-backend support**: OpenAI, Azure OpenAI, and Google Gemini.

# Quick Start
```julia
using UniLM

# Chat Completions
chat = Chat(model=\"gpt-4o\")
push!(chat, Message(Val(:system), \"You are a helpful assistant\"))
push!(chat, Message(Val(:user), \"Hello!\"))
result = chatrequest!(chat)

# Responses API
result = respond(\"Tell me a joke\")
println(output_text(result))
```

See the [documentation](https://algunion.github.io/UniLM.jl/) for full details.
"""
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
