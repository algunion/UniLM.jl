"""
    UniLM

A Julia interface for **LLM providers** with first-class native backends (OpenAI, Anthropic, Gemini) plus any OpenAI-compatible provider.

UniLM.jl provides:
- **Chat Completions** via [`Chat`](@ref) and [`chatrequest!`](@ref) — stateful conversations
  with tool calling, streaming, and structured output.
- **Responses API & agentic verb** via [`Respond`](@ref) and [`respond`](@ref) — built-in tools
  (web search, file search), multi-turn chaining via `previous_response_id`, reasoning support,
  and a cross-provider `respond` that also drives Google's Gemini Interactions.
- **Image Generation** via [`ImageGeneration`](@ref) and [`generate_image`](@ref) — create
  images from text prompts using `gpt-image-2`.
- **Embeddings** via [`Embeddings`](@ref) and [`embeddingrequest!`](@ref).
- **MCP** via [`MCPSession`](@ref) and [`MCPServer`](@ref) — Model Context Protocol client and server.
- **Cost accounting** via [`estimated_cost`](@ref) and [`cumulative_cost`](@ref) — token-usage and USD cost estimation.
- **Multi-provider support**: native OpenAI, Anthropic, and Gemini backends, plus Azure, DeepSeek, Mistral, Ollama, vLLM, and any OpenAI-compatible provider via [`GenericOpenAIEndpoint`](@ref).

# Quick Start
```julia
using UniLM

# Chat Completions
chat = Chat(model=\"gpt-5.5\")
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
using SHA

include("constants.jl")
include("exceptions.jl")
include("config.jl")
include("deadline.jl")
include("api.jl")
include("requests.jl")
include("sse.jl")
include("fork.jl")
include("responses.jl")
include("images.jl")
include("capabilities.jl")
include("anthropic.jl")
include("gemini.jl")
include("interactions.jl")
include("completions.jl")
include("accounting.jl")
include("files.jl")
include("vector_stores.jl")
include("conversations.jl")
include("moderations.jl")
include("audio.jl")
include("batch.jl")
include("fine_tuning.jl")
include("webhooks.jl")
include("containers.jl")
include("uploads.jl")
include("videos.jl")
include("realtime.jl")
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
    GEMINIOpenAIServiceEndpoint,
    ANTHROPICServiceEndpoint,
    GenericOpenAIEndpoint,
    OllamaEndpoint,
    MistralEndpoint,
    DeepSeekEndpoint,
    add_azure_deploy_name!

# ─── Request Configuration & Timeouts ─────────────────────────────────────────
export
    RequestConfig,
    current_config,
    with_request_config,
    set_default_config!,
    UniLMTimeout,
    MCPTimeoutError

# ─── Chat Completions API ─────────────────────────────────────────────────────
export
    Chat,
    Message,
    ProviderContent,
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
    EmbeddingSuccess,
    EmbeddingFailure,
    EmbeddingCallError,
    embedding_vectors,
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
    reasoning_summaries,
    reasoning_items,
    refusals,
    url_citations,
    web_search_results,
    file_search_results,
    image_generation_results,
    code_interpreter_outputs,
    mcp_call_outputs,
    mcp_approval_requests,
    response_status,
    incomplete_details,
    usage_details,
    # Input helpers
    input_text,
    input_image,
    input_file,
    # Tool constructors
    function_tool,
    tool_result,
    web_search,
    file_search,
    mcp_tool,
    mcp_approval_response,
    computer_use,
    computer_tool,
    image_generation_tool,
    code_interpreter,
    local_shell,
    shell,
    apply_patch_tool,
    custom_tool,
    # Gemini native hosted tools
    gemini_google_search,
    gemini_code_execution,
    gemini_url_context,
    # Tool types
    MCPTool,
    ComputerUseTool,
    ComputerTool,
    ImageGenerationTool,
    CodeInterpreterTool,
    LocalShellTool,
    ShellTool,
    ApplyPatchTool,
    CustomTool,
    # Format constructors
    text_format,
    json_schema_format,
    json_object_format,
    # tool_choice builders
    tool_choice_function,
    tool_choice_hosted,
    tool_choice_mcp,
    tool_choice_custom,
    tool_choice_allowed

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
    save_image,
    ImageEdit,
    edit_image

# ─── FIM / Completions API ──────────────────────────────────────────────────
export
    FIMCompletion,
    FIMChoice,
    FIMResponse,
    FIMSuccess,
    FIMFailure,
    FIMCallError,
    fim_complete,
    fim_text,
    prefix_complete

# ─── Files API ───────────────────────────────────────────────────────────────
export
    FileUpload,
    FileObject,
    FileList,
    FileSuccess,
    FileListSuccess,
    FileContentSuccess,
    FileDeleteSuccess,
    FileFailure,
    FileCallError,
    upload_file,
    list_files,
    retrieve_file,
    delete_file,
    file_content,
    save_file_content

# ─── Vector Stores API ────────────────────────────────────────────────────────
export
    VectorStoreObject,
    VectorStoreFileObject,
    VectorStoreFileBatch,
    VectorStoreList,
    VectorStoreSuccess,
    VectorStoreListSuccess,
    VectorStoreFileSuccess,
    VectorStoreBatchSuccess,
    VectorStoreDeleteSuccess,
    VectorStoreFailure,
    VectorStoreCallError,
    vector_store_id,
    create_vector_store,
    retrieve_vector_store,
    list_vector_stores,
    delete_vector_store,
    add_vector_store_file,
    create_file_batch,
    retrieve_file_batch,
    poll_file_batch

# ─── Conversations API ────────────────────────────────────────────────────────
export
    ConversationObject,
    ConversationItem,
    ConversationItemList,
    ConversationSuccess,
    ConversationItemListSuccess,
    ConversationItemSuccess,
    ConversationDeleteSuccess,
    ConversationFailure,
    ConversationCallError,
    conversation_id,
    create_conversation,
    retrieve_conversation,
    update_conversation,
    delete_conversation,
    add_conversation_items,
    list_conversation_items,
    delete_conversation_item

# ─── Moderations API ──────────────────────────────────────────────────────────
export
    ModerationResult,
    ModerationResponse,
    ModerationSuccess,
    ModerationFailure,
    ModerationCallError,
    moderate,
    is_flagged

# ─── Audio API ────────────────────────────────────────────────────────────────
export
    SpeechRequest,
    SpeechSuccess,
    TranscriptionRequest,
    TranscriptionSuccess,
    AudioFailure,
    AudioCallError,
    speak,
    save_audio,
    transcribe,
    translate,
    transcript_text

# ─── Batch API ────────────────────────────────────────────────────────────────
export
    BatchObject,
    BatchList,
    BatchSuccess,
    BatchListSuccess,
    BatchFailure,
    BatchCallError,
    create_batch,
    retrieve_batch,
    cancel_batch,
    list_batches,
    poll_batch

# ─── Fine-tuning API ──────────────────────────────────────────────────────────
export
    FineTuningJob,
    FineTuningList,
    FineTuningSuccess,
    FineTuningListSuccess,
    FineTuningFailure,
    FineTuningCallError,
    create_fine_tuning_job,
    retrieve_fine_tuning_job,
    cancel_fine_tuning_job,
    list_fine_tuning_jobs,
    list_fine_tuning_events,
    list_fine_tuning_checkpoints

# ─── Webhooks ─────────────────────────────────────────────────────────────────
export
    WebhookEvent,
    WEBHOOK_EVENTS,
    verify_webhook,
    parse_webhook

# ─── Containers API ───────────────────────────────────────────────────────────
export
    ContainerObject,
    ContainerList,
    ContainerSuccess,
    ContainerListSuccess,
    ContainerDeleteSuccess,
    ContainerFailure,
    ContainerCallError,
    create_container,
    retrieve_container,
    list_containers,
    delete_container,
    add_container_file

# ─── Uploads API ──────────────────────────────────────────────────────────────
export
    UploadObject,
    UploadPartObject,
    UploadSuccess,
    UploadPartSuccess,
    UploadFailure,
    UploadCallError,
    create_upload,
    add_upload_part,
    complete_upload,
    cancel_upload

# ─── Videos API ───────────────────────────────────────────────────────────────
export
    VideoObject,
    VideoList,
    VideoSuccess,
    VideoListSuccess,
    VideoContentSuccess,
    VideoFailure,
    VideoCallError,
    create_video,
    retrieve_video,
    list_videos,
    video_content

# ─── Realtime API ─────────────────────────────────────────────────────────────
export
    RealtimeSession,
    RealtimeSecretSuccess,
    RealtimeFailure,
    RealtimeCallError,
    mint_realtime_secret,
    realtime_connect,
    realtime_send,
    realtime_receive,
    realtime_event,
    session_update,
    input_audio_append,
    response_create

# ─── Provider Capabilities ──────────────────────────────────────────────────
export
    has_capability,
    provider_capabilities

# ─── MCP Client ──────────────────────────────────────────────────────────────
export
    MCPSession,
    MCPToolInfo,
    MCPToolResult,
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
