"""
    UniLM

A Julia interface for **LLM providers** via the OpenAI-compatible API standard.

UniLM.jl provides:
- **Chat Completions** via [`Chat`](@ref) and [`chatrequest!`](@ref) — stateful conversations
  with tool calling, streaming, and structured output.
- **Responses API** via [`Respond`](@ref) and [`respond`](@ref) — the newer, more flexible
  API with built-in tools (web search, file search), multi-turn chaining via `previous_response_id`,
  and reasoning support for O-series models.
- **Image Generation** via [`ImageGeneration`](@ref) and [`generate_image`](@ref) — create
  images from text prompts using `gpt-image-1.5`.
- **Embeddings** via [`Embeddings`](@ref) and [`embeddingrequest!`](@ref).
- **MCP** via [`MCPSession`](@ref) and [`MCPServer`](@ref) — Model Context Protocol client and server.
- **Multi-provider support**: OpenAI, Azure, Gemini, Mistral, Ollama, vLLM, and any OpenAI-compatible provider via [`GenericOpenAIEndpoint`](@ref).

# Quick Start
```julia
using UniLM

# Chat Completions
chat = Chat(model=\"gpt-5.2\")
push!(chat, Message(Val(:system), \"You are a helpful assistant\"))
push!(chat, Message(Val(:user), \"Hello!\"))
result = chatrequest!(chat)

# Responses API
result = respond(\"Tell me a joke\")
println(output_text(result))

# Image Generation
result = generate_image(\"A watercolor painting of a Julia butterfly\")
if result isa ImageSuccess
    save_image(image_data(result)[1], \"butterfly.png\")
end
```

See the [documentation](https://algunion.github.io/UniLM.jl/) for full details.
"""
module UniLM
using HTTP
using JSON
using Base64

include("constants.jl")
include("exceptions.jl")
include("api.jl")
include("requests.jl")
include("fork.jl")
include("responses.jl")
include("images.jl")
include("accounting.jl")
include("tool_loop.jl")
include("mcp_schema.jl")
include("mcp_client.jl")
include("mcp_server.jl")

# ─── Service Endpoints ────────────────────────────────────────────────────────
export
    ServiceEndpoint,
    ServiceEndpointSpec,
    OPENAIServiceEndpoint,
    AZUREServiceEndpoint,
    GEMINIServiceEndpoint,
    GenericOpenAIEndpoint,
    OllamaEndpoint,
    MistralEndpoint,
    DeepSeekEndpoint,
    add_azure_deploy_name!

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
    LLMCallError,
    # Token usage and accounting
    TokenUsage,
    token_usage,
    estimated_cost,
    cumulative_cost,
    DEFAULT_PRICING,
    # Fork
    fork,
    # Tool loop
    CallableTool,
    ToolCallOutcome,
    ToolLoopResult,
    tool_loop!,
    tool_loop,
    to_tool

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
    cancel_response,
    compact_response,
    count_input_tokens,
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
    mcp_tool,
    computer_use,
    image_generation_tool,
    code_interpreter,
    # Tool types
    MCPTool,
    ComputerUseTool,
    ImageGenerationTool,
    CodeInterpreterTool,
    # Format constructors
    text_format,
    json_schema_format,
    json_object_format

# ─── Image Generation API ─────────────────────────────────────────────────────────
export
    # Core types
    ImageGeneration,
    ImageObject,
    ImageResponse,
    # Result types
    ImageSuccess,
    ImageFailure,
    ImageCallError,
    # Functions
    generate_image,
    image_data,
    save_image

# ─── MCP Client ──────────────────────────────────────────────────────────────
export
    MCPSession,
    MCPToolInfo,
    MCPResourceInfo,
    MCPPromptInfo,
    MCPServerCapabilities,
    MCPTransport,
    StdioTransport,
    HTTPTransport,
    MCPError,
    mcp_connect,
    mcp_disconnect!,
    mcp_tools,
    mcp_tools_respond,
    list_tools!,
    list_resources!,
    list_prompts!,
    call_tool,
    read_resource,
    get_prompt,
    ping

# ─── MCP Server ──────────────────────────────────────────────────────────────
export
    MCPServer,
    MCPServerTool,
    MCPServerResource,
    MCPServerResourceTemplate,
    MCPServerPrompt,
    MCPServerPrimitive,
    register_tool!,
    register_resource!,
    register_resource_template!,
    register_prompt!,
    serve,
    @mcp_tool,
    @mcp_resource,
    @mcp_prompt

end
