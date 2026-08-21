# UniLM.jl — LLM Reference

> **Single-file reference for LLM code-generation systems.** This is a **Julia** package.
> Julia ≥ 1.12 · Deps: `HTTP.jl`, `JSON.jl`, `Base64`
> Repo: <https://github.com/algunion/UniLM.jl>

## Installation

```julia
using Pkg
Pkg.add("UniLM")
using UniLM
```

## Environment Variables

| Variable                           | Required by                       | Description                                     |
| ---------------------------------- | --------------------------------- | ----------------------------------------------- |
| `OPENAI_API_KEY`                   | `OPENAIServiceEndpoint` (default) | OpenAI API key                                  |
| `AZURE_OPENAI_BASE_URL`            | `AZUREServiceEndpoint`            | Azure deployment base URL                       |
| `AZURE_OPENAI_API_KEY`             | `AZUREServiceEndpoint`            | Azure API key                                   |
| `AZURE_OPENAI_API_VERSION`         | `AZUREServiceEndpoint`            | Azure API version string                        |
| `AZURE_OPENAI_DEPLOY_NAME_GPT_5_2` | `AZUREServiceEndpoint`            | Auto-registers Azure deployment for `"gpt-5.2"` |
| `GEMINI_API_KEY`                   | `GEMINIServiceEndpoint`           | Google Gemini API key                           |
| `ANTHROPIC_API_KEY`                | `ANTHROPICServiceEndpoint`        | Anthropic (Claude) API key                      |
| `DEEPSEEK_API_KEY`                 | `DeepSeekEndpoint`                | DeepSeek API key                                |
| `MISTRAL_API_KEY`                  | `MistralEndpoint`                 | Mistral AI API key                              |

## Four APIs

UniLM.jl wraps four OpenAI API surfaces plus FIM completion:

1. **Chat Completions** (`Chat` + `chatrequest!`) — stateful, message-based conversations with tool calling, streaming, structured output. Supports OpenAI, Azure, Gemini (native), Anthropic (native), DeepSeek, Ollama, Mistral, and any OpenAI-compatible provider.
2. **Responses API** (`Respond` + `respond`) — newer, more flexible API with built-in tools (web search, file search), multi-turn chaining via `previous_response_id`, reasoning support for O-series models, structured output. OpenAI Responses by default; the unified `respond` verb also targets Google's Gemini Interactions via `service=GEMINIServiceEndpoint` (see the Agentic Workflows guide).
3. **Image Generation** (`ImageGeneration` + `generate_image`) — text-to-image with `gpt-image-2`. OpenAI only.
4. **Embeddings** (`Embeddings` + `embeddingrequest!`) — vector embeddings. Multi-provider via `service` parameter.
5. **FIM Completion** (`FIMCompletion` + `fim_complete`) — code infilling. DeepSeek, Ollama, vLLM.

**Which API to use:**
- **Chat Completions** — best for multi-turn conversations; broadest provider support. Use for chat, tool calling, or streaming across any supported backend.
- **Responses API** — simpler for single-shot or chained requests; built-in web search, file search, MCP, computer use tools. OpenAI Responses plus Google's Gemini Interactions via the unified `respond` verb (see the Agentic Workflows guide).
- **FIM Completion** — code infilling between prefix and suffix. DeepSeek, Ollama, vLLM only.

---

## Service Endpoints

```julia
abstract type ServiceEndpoint end
abstract type OpenAIWireEndpoint <: ServiceEndpoint end   # OpenAI-wire backends inherit encode/decode/SSE
struct OPENAIServiceEndpoint <: OpenAIWireEndpoint end   # default — uses OPENAI_API_KEY
struct AZUREServiceEndpoint  <: OpenAIWireEndpoint end   # uses AZURE_OPENAI_* env vars
struct GEMINIServiceEndpoint <: ServiceEndpoint end       # native generateContent — GEMINI_API_KEY
struct GEMINIOpenAIServiceEndpoint <: OpenAIWireEndpoint end # Gemini via OpenAI-compat shim — GEMINI_API_KEY
struct ANTHROPICServiceEndpoint <: ServiceEndpoint end    # native Messages API — ANTHROPIC_API_KEY
struct GenericOpenAIEndpoint <: OpenAIWireEndpoint    # any OpenAI-compatible provider
    base_url::String
    api_key::String
end
# DeepSeekEndpoint <: OpenAIWireEndpoint  (constructor below)

# Convenience constructors
OllamaEndpoint(; base_url="http://localhost:11434")   # Ollama local
MistralEndpoint(; api_key=ENV["MISTRAL_API_KEY"])     # Mistral AI
DeepSeekEndpoint(; api_key=ENV["DEEPSEEK_API_KEY"])   # DeepSeek

# Type alias for service fields — accepts both marker types and instances:
const ServiceEndpointSpec = Union{Type{<:ServiceEndpoint}, ServiceEndpoint}
# Built-in types: Chat(service=OPENAIServiceEndpoint)      — passed as the type
# Instance types: Chat(service=DeepSeekEndpoint())          — passed as a constructed value
```

### Provider Compatibility

UniLM talks to **native** provider APIs where it implements them, and rides the **OpenAI-compatible** standard elsewhere:

| Access path | Providers |
|---|---|
| **Native backends** (own wire format) | OpenAI (Chat + Responses), Anthropic (Messages), Gemini (generateContent + agentic Interactions) |
| Chat Completions (OpenAI-compat) | OpenAI, Azure, DeepSeek, Mistral, Ollama, vLLM, LM Studio, Gemini (compat shim) |
| Embeddings (OpenAI-compat) | OpenAI, Gemini (compat shim), Mistral, Ollama, vLLM |
| Responses API | OpenAI, Ollama, vLLM, Amazon Bedrock (emerging Open Responses) |
| Image Generation | OpenAI |

Anthropic and native Gemini use their **own** APIs here (`ANTHROPICServiceEndpoint` / `GEMINIServiceEndpoint`) — not an OpenAI-compat shim — and that is the recommended path. The OpenAI-compat Gemini shim (`GEMINIOpenAIServiceEndpoint`) exists for embeddings and drop-in compatibility.

Register additional Azure deployments at runtime:

```julia
add_azure_deploy_name!(model::String, deploy_name::String)
# e.g. add_azure_deploy_name!("gpt-5.2", "my-deployment")
```

Pass the backend via the `service` keyword on `Chat`, `Respond`, or `ImageGeneration`:

```julia
Chat(service=AZUREServiceEndpoint, model="gpt-5.2")
Respond(service=OPENAIServiceEndpoint, input="Hello")
```

---

## Chat Completions API

### Chat

```julia
@kwdef struct Chat
    service::ServiceEndpointSpec = OPENAIServiceEndpoint
    model::String = "gpt-5.5"
    messages::Vector{Message} = Message[]
    history::Bool = true
    tools::Union{Vector{Tool},Nothing} = nothing
    tool_choice::Union{String,GPTToolChoice,Nothing} = nothing
    parallel_tool_calls::Union{Bool,Nothing} = false
    temperature::Union{Float64,Nothing} = nothing       # 0.0–2.0, mutually exclusive with top_p
    top_p::Union{Float64,Nothing} = nothing              # 0.0–1.0, mutually exclusive with temperature
    n::Union{Int64,Nothing} = nothing
    stream::Union{Bool,Nothing} = nothing
    stop::Union{Vector{String},String,Nothing} = nothing # max 4 sequences
    max_tokens::Union{Int64,Nothing} = nothing
    max_completion_tokens::Union{Int64,Nothing} = nothing # preferred over max_tokens; reasoning models reject max_tokens
    presence_penalty::Union{Float64,Nothing} = nothing   # -2.0 to 2.0
    response_format::Union{ResponseFormat,Nothing} = nothing
    frequency_penalty::Union{Float64,Nothing} = nothing  # -2.0 to 2.0
    logit_bias::Union{AbstractDict{String,Float64},Nothing} = nothing
    user::Union{String,Nothing} = nothing
    seed::Union{Int64,Nothing} = nothing
    reasoning_effort::Union{String,Nothing} = nothing    # none|minimal|low|medium|high|xhigh
    stream_options::Union{AbstractDict,Nothing} = nothing # e.g. Dict("include_usage" => true)
    verbosity::Union{String,Nothing} = nothing           # low|medium|high
    store::Union{Bool,Nothing} = nothing
    metadata::Union{AbstractDict,Nothing} = nothing
    service_tier::Union{String,Nothing} = nothing        # auto|default|flex|scale|priority
    logprobs::Union{Bool,Nothing} = nothing
    top_logprobs::Union{Int64,Nothing} = nothing
    prediction::Union{AbstractDict,Nothing} = nothing
    modalities::Union{Vector{String},Nothing} = nothing
    audio::Union{AbstractDict,Nothing} = nothing
    web_search_options::Union{AbstractDict,Nothing} = nothing
    prompt_cache_key::Union{String,Nothing} = nothing
    safety_identifier::Union{String,Nothing} = nothing   # replaces deprecated `user`
end
```

A 35th field, `_cumulative_cost::Ref{Float64}`, is internal bookkeeping for
[`cumulative_cost`](@ref) — read it through that accessor, never directly.

- **Model defaults**: the declared default is the sentinel `""`; the constructor resolves it, so `Chat().model` reads back `"gpt-5.5"`. Per provider: `"gpt-5.5"` for OpenAI, `"gpt-5.2"` for Azure, `"gemini-3.5-flash"` for Gemini (native and OpenAI-compat), `"claude-opus-4-8"` for native Anthropic, `"deepseek-chat"` for DeepSeek. For `GenericOpenAIEndpoint` / `OllamaEndpoint` there is no default — an unset model throws `ArgumentError` at construction.
- `history=true`: responses are automatically appended to `messages`.
- `temperature` and `top_p` are mutually exclusive (constructor throws `ArgumentError`).
- `parallel_tool_calls` is auto-set to `nothing` when `tools` is `nothing`.
- **Parameter validation**: the constructor validates ranges at construction time — `temperature` ∈ [0.0, 2.0], `top_p` ∈ [0.0, 1.0], `n` ∈ [1, 10], `presence_penalty` ∈ [-2.0, 2.0], `frequency_penalty` ∈ [-2.0, 2.0]. Out-of-range values throw `ArgumentError`.

### Message

```julia
@kwdef struct Message
    role::String                                          # RoleSystem, RoleUser, RoleAssistant, or "tool"
    content::Union{String,Nothing} = nothing
    name::Union{String,Nothing} = nothing
    finish_reason::Union{String,Nothing} = nothing        # "stop", "tool_calls", "content_filter"
    refusal_message::Union{String,Nothing} = nothing
    tool_calls::Union{Nothing,Vector{ToolCall}} = nothing
    tool_call_id::Union{String,Nothing} = nothing         # required when role == "tool"
    provider_content::Union{Nothing,ProviderContent} = nothing
end
```

**Validation**: at least one of `content`, `tool_calls`, or `refusal_message` must be non-`nothing`. `tool_call_id` is required when `role == "tool"`.

`provider_content` carries provider-native blocks (e.g. Anthropic thinking) captured for verbatim round-trip; it never serializes on the OpenAI wire.

**Convenience constructors**:

```julia
Message(Val(:system), "You are a helpful assistant")
Message(Val(:user), "Hello!")
```

**Role constants**: `RoleSystem = "system"`, `RoleUser = "user"`, `RoleAssistant = "assistant"`.

### chatrequest!

```julia
# Mutating form — sends chat.messages, appends response when history=true
chatrequest!(chat::Chat; config=nothing, callback=nothing, on_tool_call=nothing) -> LLMSuccess | LLMFailure | LLMCallError | Task

# Keyword-argument convenience form — builds a Chat internally
chatrequest!(; service=OPENAIServiceEndpoint, model="gpt-5.5",
    systemprompt, userprompt, messages=Message[], history=true,
    tools=nothing, tool_choice=nothing, temperature=nothing, ...) -> same
```

- Non-streaming: returns `LLMSuccess`, `LLMFailure`, or `LLMCallError`.
- Streaming (`stream=true`): returns a `Task`. Pass a `callback(chunk::Union{String,Message}, close::Ref{Bool})` — text deltas arrive as `String`s (verbatim, in order), then the assembled `Message` at end-of-stream.
- Streaming tool calls: pass `on_tool_call(tc::ToolCall)` to be notified once per completed streamed tool call, as calls finish (see the [Streaming guide](@ref streaming_guide)).
- Retries transient statuses (408/429/500/502/503/504/529) with exponential backoff and jitter under the resolved [`RequestConfig`](@ref) — `max_attempts` (default 3) and `total_deadline` bound the attempts; `Retry-After` is honored, but a retry whose backoff would exceed the remaining `total_deadline` is not attempted — the call fails immediately with the last real response rather than sleeping past the deadline. Timeouts surface as `LLMCallError` with `status=nothing` and the `UniLMTimeout` in `.cause`.
- Streaming retry boundary: transient failures (including the in-band `overloaded_error`, the documented 529 equivalent) are retried inside the task only until the first `callback`/`on_tool_call` invocation; afterwards failures surface typed. A user `InterruptException` propagates — `fetch` throws a `TaskFailedException` instead of returning a result value.

### Conversation Management

```julia
push!(chat, message)       # append a Message (the FIRST message must be a system Message)
pop!(chat)                 # remove last message
update!(chat, message)     # append if history=true
issendvalid(chat) -> Bool  # check conversation rules (see below)
length(chat)               # number of messages
isempty(chat)              # true if no messages
chat[i]                    # index into messages
```

`issendvalid` is `true` only when ALL of these hold — note it is stricter than
`push!`, with no tool exemption:

1. at least 2 messages;
2. the first message is a system message;
3. the **last** message is a user message;
4. no two adjacent messages share a role — including `tool`, which `push!` does
   permit consecutively.

**Important:** A `Chat` must begin with a system message. `push!` throws
[`InvalidConversationError`](@ref) on a non-system message pushed onto an empty `Chat`,
on a system message pushed once the conversation has started, and on consecutive
same-role messages (except `tool`), so
`chat = Chat(); push!(chat, Message(Val(:user), "…"))` raises rather than silently
leaving the chat empty. `pop!` on an empty `Chat` likewise throws
`InvalidConversationError` rather than returning `nothing`. Use `respond(input=…)`
for a single turn without a system prompt.

### Tool Calling Types

```julia
# Define a function the model can call
@kwdef struct FunctionSignature
    name::String
    description::Union{String,Nothing} = nothing
    parameters::Union{AbstractDict,Nothing} = nothing   # JSON Schema dict
    strict::Union{Bool,Nothing} = nothing               # strict function calling; nothing = omit (API default)
end

# Wrap it for the tools parameter
@kwdef struct Tool
    type::String = "function"
    func::FunctionSignature
end
Tool(d::AbstractDict)   # construct from dict with keys "name", "description", "parameters", "strict"

# Returned by model when it wants to call a function
@kwdef struct ToolCall
    id::String
    type::String = "function"
    func::GPTFunction       # has .name::String and .arguments::AbstractDict
    # Gemini-3's opaque tool-call signature. Set only by the Gemini decoder
    # (nothing on every other provider) and echoed verbatim on the next turn;
    # deliberately excluded from JSON.lower, so OpenAI-wire bytes are unchanged.
    thought_signature::Union{Nothing,String} = nothing
end

# Your result after executing the function
struct FunctionCallResult{T}
    name::Union{String,Symbol}
    origincall::GPTFunction
    result::T
end
```

The pre-rename names `GPTTool`, `GPTToolCall`, `GPTFunctionSignature`, and
`GPTFunctionCallResult` remain exported as aliases of these types, so existing
code keeps working unchanged.

### ResponseFormat (Structured Output)

```julia
@kwdef struct ResponseFormat
    type::String = "json_object"                               # "json_object" or "json_schema"
    json_schema::Union{JsonSchemaAPI,AbstractDict,Nothing} = nothing
end
ResponseFormat(json_schema)  # shorthand, sets type="json_schema"

@kwdef struct JsonSchemaAPI   # not exported — construct as UniLM.JsonSchemaAPI(...)
    name::String
    description::String
    schema::AbstractDict
    strict::Union{Bool,Nothing} = nothing   # strict schema adherence; nothing = omit (API default)
end
UniLM.json_schema(name, description, schema; strict=nothing)  # → ResponseFormat(JsonSchemaAPI(...)); not exported
```

### Chat Completions Example

```julia
using UniLM

# Build conversation
chat = Chat(model="gpt-5.2")
push!(chat, Message(Val(:system), "You are a helpful assistant."))
push!(chat, Message(Val(:user), "What is the capital of France?"))

result = chatrequest!(chat)
if result isa LLMSuccess
    println(result.message.content)     # "Paris..."
    # chat.messages already has the response appended (history=true)
end

# One-shot via keywords
result = chatrequest!(
    systemprompt="You are a translator.",
    userprompt="Translate 'hello' to French.",
    model="gpt-5.2"
)
```

### Tool Calling Example (Chat)

```julia
weather_tool = Tool(func=FunctionSignature(
    name="get_weather",
    description="Get current weather",
    parameters=Dict(
        "type" => "object",
        "properties" => Dict("location" => Dict("type" => "string")),
        "required" => ["location"]
    )
))

chat = Chat(model="gpt-5.2", tools=[weather_tool])
push!(chat, Message(Val(:system), "You help with weather."))
push!(chat, Message(Val(:user), "Weather in Paris?"))

result = chatrequest!(chat)
if result isa LLMSuccess && result.message.finish_reason == "tool_calls"
    for tc in result.message.tool_calls
        # tc.func.name == "get_weather", tc.func.arguments == Dict("location" => "Paris")
        answer = "22°C, sunny"  # your function result
        push!(chat, Message(role="tool", content=answer, tool_call_id=tc.id))
    end
    result2 = chatrequest!(chat)
    println(result2.message.content)
end
```

### Streaming Example (Chat)

```julia
chat = Chat(model="gpt-5.2", stream=true)
push!(chat, Message(Val(:system), "You are helpful."))
push!(chat, Message(Val(:user), "Tell me a story."))

task = chatrequest!(chat) do chunk, close_ref
    if chunk isa String
        print(chunk)            # partial text delta
    elseif chunk isa Message
        println("\n[Done]")     # final assembled message
        # close_ref[] = true    # to stop early
    end
end

result = fetch(task)  # LLMSuccess when complete
```

---

## Responses API

### Respond

```julia
@kwdef struct Respond
    service::ServiceEndpointSpec = OPENAIServiceEndpoint
    model::String = "gpt-5.5"                                # declared as "" and resolved by the constructor
    input::Union{String, Vector}                             # String, Vector{InputMessage}, or Vector{Dict}
    instructions::Union{String,Nothing} = nothing
    tools::Union{Vector,Nothing} = nothing                  # untyped: accepts ResponseTool, CallableTool, and Dict
    tool_choice::Union{String,AbstractDict,Nothing} = nothing  # "auto"/"none"/"required", or a tool_choice_* Dict (see below)
    parallel_tool_calls::Union{Bool,Nothing} = nothing
    temperature::Union{Float64,Nothing} = nothing           # 0.0–2.0, mutually exclusive with top_p
    top_p::Union{Float64,Nothing} = nothing                 # 0.0–1.0
    max_output_tokens::Union{Int64,Nothing} = nothing
    stream::Union{Bool,Nothing} = nothing
    text::Union{TextConfig,Nothing} = nothing               # output format
    reasoning::Union{Reasoning,Nothing} = nothing           # O-series models
    truncation::Union{String,Nothing} = nothing             # "auto" or "disabled"
    store::Union{Bool,Nothing} = nothing                    # store for later retrieval
    metadata::Union{AbstractDict,Nothing} = nothing
    previous_response_id::Union{String,Nothing} = nothing   # multi-turn chaining
    user::Union{String,Nothing} = nothing
    background::Union{Bool,Nothing} = nothing
    include::Union{Vector{String},Nothing} = nothing
    max_tool_calls::Union{Int64,Nothing} = nothing
    service_tier::Union{String,Nothing} = nothing           # "auto","default","flex","priority"
    top_logprobs::Union{Int64,Nothing} = nothing            # 0–20
    prompt::Union{AbstractDict,Nothing} = nothing
    prompt_cache_key::Union{String,Nothing} = nothing
    prompt_cache_retention::Union{String,Nothing} = nothing  # "in-memory","24h"
    safety_identifier::Union{String,Nothing} = nothing
    conversation::Union{Any,Nothing} = nothing              # String or Dict; the declared type collapses to Any
    context_management::Union{Vector,Nothing} = nothing
    stream_options::Union{AbstractDict,Nothing} = nothing
end
```

!!! warning "Gemini Interactions rejects unmappable fields"
    With `service=GEMINIServiceEndpoint`, a set `Respond` field the Interactions
    wire has no equivalent for throws `ArgumentError` at encode time rather than
    being silently dropped, so a request never goes out quietly ignoring what you
    asked for. That wire maps `model`, `input`, `instructions`, `tools`,
    `tool_choice`, `temperature`, `top_p`, `max_output_tokens`, `stream`, `store`,
    `previous_response_id`, and `background`; leave every other field unset, or
    send the request to an OpenAI Responses service.

### Input Helpers

```julia
# Structured input messages
InputMessage(role="user", content="Hello")
InputMessage(role="user", content=[input_text("Describe:"), input_image("https://...")])

# Content part constructors
input_text(text::String)                                              # → Dict(:type=>"input_text", :text=>...)
input_image(url=nothing; detail=nothing, file_id=nothing)             # → Dict(:type=>"input_image", ...); pass url OR file_id, detail: "auto","low","high"
input_file(; url=nothing, id=nothing, file_data=nothing, filename=nothing)  # → Dict(:type=>"input_file", ...); pass one of url / id / file_data (base64)
```

`input_image` and `input_file` throw `ArgumentError` when none of their
source arguments is given.

### Tool Types

```julia
abstract type ResponseTool end

@kwdef struct FunctionTool <: ResponseTool
    name::String
    description::Union{String,Nothing} = nothing
    parameters::Union{AbstractDict,Nothing} = nothing
    strict::Union{Bool,Nothing} = nothing
end

@kwdef struct WebSearchTool <: ResponseTool
    type::String = "web_search"                             # GA; legacy "web_search_preview"
    search_context_size::String = "medium"                  # "low","medium","high"
    user_location::Union{AbstractDict,Nothing} = nothing
    filters::Union{AbstractDict,Nothing} = nothing          # "allowed_domains"/"blocked_domains" (GA only)
end

@kwdef struct FileSearchTool <: ResponseTool
    vector_store_ids::Vector{String}
    max_num_results::Union{Int,Nothing} = nothing
    ranking_options::Union{AbstractDict,Nothing} = nothing
    filters::Union{AbstractDict,Nothing} = nothing
end
```

```julia
# Reach a remote MCP server (server_url), an OpenAI connector (connector_id, e.g.
# "connector_googledrive"), or a Secure MCP Tunnel (tunnel_id) — hence no single
# required target field beyond server_label.
@kwdef struct MCPTool <: ResponseTool
    server_label::String
    server_url::Union{String, Nothing} = nothing
    connector_id::Union{String, Nothing} = nothing
    authorization::Union{String, Nothing} = nothing         # OAuth access token
    server_description::Union{String, Nothing} = nothing
    require_approval::Union{String, AbstractDict, Nothing} = "never"
    allowed_tools::Union{Vector{String}, AbstractDict, Nothing} = nothing
    headers::Union{AbstractDict, Nothing} = nothing
    tunnel_id::Union{String, Nothing} = nothing
end

@kwdef struct ComputerUseTool <: ResponseTool
    display_width::Int = 1024
    display_height::Int = 768
    environment::Union{String, Nothing} = nothing
end

@kwdef struct ImageGenerationTool <: ResponseTool
    background::Union{String, Nothing} = nothing
    output_format::Union{String, Nothing} = nothing
    output_compression::Union{Int, Nothing} = nothing
    quality::Union{String, Nothing} = nothing
    size::Union{String, Nothing} = nothing
end

@kwdef struct CodeInterpreterTool <: ResponseTool
    container::Union{AbstractDict, Nothing} = nothing
    file_ids::Union{Vector{String}, Nothing} = nothing
end
```

**Convenience constructors**:

```julia
function_tool(name, description=nothing; parameters=nothing, strict=nothing)
function_tool(d::AbstractDict)           # from dict with keys "name", "description", "parameters"
web_search(; context_size="medium", location=nothing, type="web_search", filters=nothing)
file_search(store_ids; max_results=nothing, ranking=nothing, filters=nothing)
mcp_tool(label, url=nothing; require_approval="never", allowed_tools=nothing, headers=nothing,
         connector_id=nothing, authorization=nothing, server_description=nothing, tunnel_id=nothing)
computer_use(; display_width=1024, display_height=768, environment=nothing)
image_generation_tool(; kwargs...)
code_interpreter(; container=nothing, file_ids=nothing)
```

### Text Format / Structured Output

```julia
@kwdef struct TextFormatSpec
    type::String = "text"                                   # "text","json_object","json_schema"
    name::Union{String,Nothing} = nothing
    description::Union{String,Nothing} = nothing
    schema::Union{AbstractDict,Nothing} = nothing
    strict::Union{Bool,Nothing} = nothing
end

@kwdef struct TextConfig
    format::TextFormatSpec = TextFormatSpec()
    verbosity::Union{String,Nothing} = nothing              # "low","medium","high" (gpt-5.x)
end
```

**Convenience constructors**:

```julia
text_format(; verbosity=nothing, kwargs...)                  # kwargs go to TextFormatSpec; verbosity to TextConfig
json_schema_format(name, description, schema; strict=nothing) # JSON Schema output
json_schema_format(d::AbstractDict)                          # from dict with keys "name", "description", "schema"
json_object_format()                                         # unstructured JSON
```

### Reasoning (O-series models)

```julia
@kwdef struct Reasoning
    effort::Union{String,Nothing} = nothing                 # "none","low","medium","high"
    generate_summary::Union{String,Nothing} = nothing       # "auto","concise","detailed"
    summary::Union{String,Nothing} = nothing                # deprecated alias
end
```

```julia
Respond(input="Hard math problem", model="o3", reasoning=Reasoning(effort="high"))
```

### respond

```julia
# Struct form
respond(r::Respond; config=nothing, callback=nothing) -> ResponseSuccess | ResponseFailure | ResponseCallError | Task

# Convenience — builds Respond internally
respond(input; kwargs...) -> same

# do-block streaming — auto-sets stream=true
respond(callback::Function, input; kwargs...) -> Task
```

- Streaming callback signature: `callback(chunk::Union{String, ResponseObject}, close::Ref{Bool})`
- Retries retryable statuses (408/429/500/502/503/504/529) up to `config.max_attempts` (default 3) with full-jitter backoff bounded by `config.total_deadline`; honors `Retry-After`, but a retry whose backoff would exceed the remaining deadline is not attempted — it fails immediately with the last real response instead of sleeping past it. Every attempt is time-bounded — a silent peer fails with a typed timeout inside `ResponseCallError` (`status = nothing`, `cause::UniLMTimeout`), never a hang.
- **Parameter validation**: `temperature` ∈ [0.0, 2.0], `top_p` ∈ [0.0, 1.0], `max_output_tokens` ≥ 1, `top_logprobs` ∈ [0, 20]. Out-of-range values throw `ArgumentError`.

### Response Accessors

```julia
output_text(result::ResponseSuccess)::String                # concatenated text output
output_text(result::ResponseFailure)::String                # error message
output_text(result::ResponseCallError)::String              # error message

function_calls(result::ResponseSuccess)::Vector{Dict{String,Any}}
# Each dict has: "id", "call_id", "name", "arguments" (JSON string), "status"
```

### Response Management Functions

```julia
get_response(id::String; service=OPENAIServiceEndpoint)           -> ResponseSuccess | ResponseFailure | ResponseCallError
delete_response(id::String; service=OPENAIServiceEndpoint)        -> Dict | ResponseFailure | ResponseCallError
list_input_items(id::String; limit=20, order="desc", after=nothing, service=OPENAIServiceEndpoint) -> Dict | ...
cancel_response(id::String; service=OPENAIServiceEndpoint)        -> ResponseSuccess | ...
compact_response(; model="gpt-5.5", input, service=OPENAIServiceEndpoint) -> Dict | ...
count_input_tokens(; model="gpt-5.5", input, instructions=nothing, tools=nothing, service=OPENAIServiceEndpoint) -> Dict | ...
```

### ResponseObject

```julia
@kwdef struct ResponseObject
    id::String
    status::String
    model::String
    output::Vector{Any}
    usage::Union{Dict{String,Any},Nothing} = nothing
    error::Union{Dict{String,Any}, String, Nothing} = nothing   # provider error object, or a plain message
    metadata::Union{Dict{String,Any},Nothing} = nothing
    raw::Dict{String,Any}
end
```

A generation whose terminal status is `"failed"` is reported as a
`ResponseFailure`, not a success — on both agentic wires, streamed and
non-streamed alike. The failure carries the wire body verbatim, so the response's
own `error` and `metadata` are preserved on `.response`. Non-streamed, only
`"failed"` flips: `cancelled`, `expired`, `in_progress`, `queued` and
`requires_action` are legitimate terminals of the background and tool-action
flows and stay `ResponseSuccess`, inspectable via [`response_status`](@ref) and
[`incomplete_details`](@ref). Streamed, a terminal `response.incomplete` event
also resolves to `ResponseFailure`, so check the result type rather than
assuming a stream that produced deltas ended in success.

### Responses API Examples

```julia
using UniLM

# Basic
result = respond("Tell me a joke")
if result isa ResponseSuccess
    println(output_text(result))
end

# With instructions
result = respond("Hello", instructions="You are a pirate. Respond in pirate speak.")

# Multi-turn via chaining
r1 = respond("Tell me a joke")
r2 = respond("Tell me another", previous_response_id=r1.response.id)

# Structured output
schema = Dict(
    "type" => "object",
    "properties" => Dict(
        "name" => Dict("type" => "string"),
        "age" => Dict("type" => "integer")
    ),
    "required" => ["name", "age"],
    "additionalProperties" => false
)
result = respond("Extract: John is 30 years old",
    text=json_schema_format("person", "A person", schema, strict=true))
parsed = JSON.parse(output_text(result))

# Web search
result = respond("Latest Julia language news", tools=[web_search()])

# Function calling
tool = function_tool("get_weather", "Get weather",
    parameters=Dict(
        "type" => "object",
        "properties" => Dict("location" => Dict("type" => "string")),
        "required" => ["location"]
    ))
result = respond("Weather in NYC?", tools=ResponseTool[tool])
for call in function_calls(result)
    println(call["name"], ": ", call["arguments"])
end

# Streaming (do-block)
respond("Tell me a story") do chunk, close_ref
    if chunk isa String
        print(chunk)
    elseif chunk isa ResponseObject
        println("\nDone: ", chunk.status)
    end
end

# Reasoning (O-series)
result = respond("Prove that √2 is irrational", model="o3",
    reasoning=Reasoning(effort="high", generate_summary="concise"))

# Multimodal input
result = respond([
    InputMessage(role="user", content=[
        input_text("What's in this image?"),
        input_image("https://example.com/photo.jpg")
    ])
])

# Count tokens without generating
tokens = count_input_tokens(model="gpt-5.2", input="Hello world")
println(tokens["input_tokens"])
```

---

## Image Generation API

### ImageGeneration

```julia
@kwdef struct ImageGeneration
    service::ServiceEndpointSpec = OPENAIServiceEndpoint
    model::String = ""                                      # sentinel — see below
    prompt::String
    n::Union{Int,Nothing} = nothing                         # 1–10
    size::Union{String,Nothing} = nothing                   # "1024x1024","1536x1024","1024x1536","auto"
    quality::Union{String,Nothing} = nothing                # "low","medium","high","auto"
    background::Union{String,Nothing} = nothing             # "transparent","opaque","auto"
    output_format::Union{String,Nothing} = nothing          # "png","webp","jpeg"
    output_compression::Union{Int,Nothing} = nothing        # 0–100 (webp/jpeg only)
    user::Union{String,Nothing} = nothing
    input_fidelity::Union{String,Nothing} = nothing         # how closely to follow an input image
    moderation::Union{String,Nothing} = nothing             # content-moderation strictness
end
```

Unlike `Chat` and `Respond`, `ImageGeneration` does **not** resolve its model at
construction: the field stays `""` and resolves at serialization time, to
`"gpt-image-2"` for OpenAI. So `ImageGeneration(prompt="…").model` reads back as
the empty string — pass `model=` explicitly if you need to read it, or inspect
`JSON.json(ig)` to see what will go on the wire. A service with no default image
model throws `ArgumentError` at serialization.

### generate_image

```julia
generate_image(ig::ImageGeneration; config=nothing) -> ImageSuccess | ImageFailure | ImageCallError
generate_image(prompt::String; kwargs...)       -> same   # convenience
```

Retries transient statuses (408/429/500/502/503/504/529) inside the request budget (`RequestConfig.max_attempts`, default 3; `Retry-After` respected) — a retry whose backoff would exceed the remaining `total_deadline` is not attempted, so the call fails immediately with the last real response rather than sleeping past it. Bound per call with `config=RequestConfig(...)`.

### Response Types

```julia
struct ImageObject
    b64_json::Union{String,Nothing}
    revised_prompt::Union{String,Nothing}
end

struct ImageResponse
    created::Int64
    data::Vector{ImageObject}
    usage::Union{Dict{String,Any},Nothing}
    raw::Dict{String,Any}
end
```

### Accessors

```julia
image_data(result::ImageSuccess)::Vector{String}       # base64-encoded image strings
image_data(result::ImageFailure)::String[]              # empty
image_data(result::ImageCallError)::String[]            # empty
save_image(img_b64::String, filepath::String)           # decode + write to disk, returns filepath
```

### Image Generation Example

```julia
using UniLM

result = generate_image("A watercolor painting of a Julia butterfly",
    size="1024x1024", quality="high")

if result isa ImageSuccess
    imgs = image_data(result)
    save_image(imgs[1], "butterfly.png")
    println("Saved! Revised prompt: ", result.response.data[1].revised_prompt)
end

# Multiple images with transparent background
result = generate_image("Minimalist logo",
    n=3, background="transparent", output_format="png")
```

---

## Embeddings API

### Embeddings

```julia
struct Embeddings
    service::ServiceEndpointSpec     # default: OPENAIServiceEndpoint
    model::String                    # default resolved per provider at construction
    input::Union{String,Vector{String}}
    embeddings::Union{Vector{Float64},Vector{Vector{Float64}}}   # pre-allocated, filled in place
    user::Union{String,Nothing}
    dimensions::Union{Int,Nothing}          # requested output dimensionality
    encoding_format::Union{String,Nothing}  # e.g. "float", "base64"
end

Embeddings(input::String; service=OPENAIServiceEndpoint, model="",
           dimensions=nothing, encoding_format=nothing, user=nothing)
Embeddings(input::Vector{String}; service=OPENAIServiceEndpoint, model="",
           dimensions=nothing, encoding_format=nothing, user=nothing)
```

Model defaults: `"text-embedding-3-small"` for OpenAI, `"gemini-embedding-001"` for Gemini (OpenAI-compat shim). For generic/DeepSeek endpoints, model must be specified explicitly or the constructor throws `ArgumentError`. An empty `Vector{String}` input also throws.

The `embeddings` buffer is pre-allocated to `something(dimensions, 1536)` zeros per input and filled in place. Because a pre-zeroed slot would be indistinguishable from a real vector, a response that does not cover every input exactly once — a missing or duplicated row index — is reported as an `EmbeddingCallError` rather than left as zeros.

### embeddingrequest!

```julia
embeddingrequest!(emb::Embeddings; config=nothing) -> EmbeddingSuccess | EmbeddingFailure | EmbeddingCallError
```

Returns an `EmbeddingSuccess`/`EmbeddingFailure`/`EmbeddingCallError` (a `<: LLMRequestResponse`). Fills `emb.embeddings` in-place; `embedding_vectors(result)` returns the vectors. Retries transient statuses (408/429/500/502/503/504/529) with backoff and jitter under the resolved [`RequestConfig`](@ref) (`max_attempts`, `total_deadline`; `Retry-After` honored), but a retry whose backoff would exceed the remaining deadline is not attempted — it fails immediately with the last real response rather than sleeping past it. Timeouts surface as `EmbeddingCallError` with `status=nothing` and the `UniLMTimeout` in `.cause`.

### Embeddings Example

```julia
using UniLM, LinearAlgebra

emb = Embeddings("What is Julia?")
embeddingrequest!(emb)
println(emb.embeddings[1:5])  # first 5 dimensions

# Batch + cosine similarity
emb = Embeddings(["cat", "dog", "airplane"])
embeddingrequest!(emb)
similarity = dot(emb.embeddings[1], emb.embeddings[2]) /
    (norm(emb.embeddings[1]) * norm(emb.embeddings[2]))
```

---

## Cost Tracking

### TokenUsage

```julia
@kwdef struct TokenUsage
    prompt_tokens::Int = 0
    completion_tokens::Int = 0
    total_tokens::Int = 0
    cached_tokens::Int = 0      # subset of prompt_tokens served from the prompt cache
    reasoning_tokens::Int = 0   # subset of completion_tokens spent on hidden reasoning
end
```

`cached_tokens` and `reasoning_tokens` come from the providers' `*_tokens_details`
objects and default to `0` when the provider omits them. `cached_tokens` is
load-bearing for cost: it is billed at the discounted cached-input rate.

### Functions

```julia
token_usage(result::LLMRequestResponse)::TokenUsage   # never Nothing — zero usage for a failure
estimated_cost(result; model=nothing, pricing=DEFAULT_PRICING)   # per-call cost estimate (Float64)
cumulative_cost(chat::Chat)::Float64                             # running total for a Chat instance

DEFAULT_PRICING   # Dict{String, PriceRow} where PriceRow = @NamedTuple{input, cached_input, output}
```

- **`token_usage` returns a `TokenUsage`, never `nothing`**, for every result of a
  token-billed API — chat, Responses, embeddings, and their image counterparts.
  Failures of those APIs report all-zero usage.
- **Both functions throw `ArgumentError` for result types outside the token-billed
  APIs** — audio, batch, container, conversation, file, fine-tuning, moderation,
  upload, vector-store, video, realtime. Those calls carry no token usage at all,
  and a `0.0` would be indistinguishable from a genuinely free call.
- **`DEFAULT_PRICING` values are USD *per token*, not per 1M tokens** — provider
  list prices divided by `1_000_000` (e.g. `"gpt-5.5"` is
  `(input = 5.0e-6, cached_input = 5.0e-7, output = 3.0e-5)`). A `pricing=`
  dict you supply must use the same per-token convention, or your estimate is
  off by a factor of a million.
- `estimated_cost` returns `0.0` for any model **not** in the pricing dict — pass
  `pricing=` to price custom models. The formula bills
  `min(cached_tokens, prompt_tokens)` at `cached_input`, the remaining prompt
  tokens at `input`, and `completion_tokens` at `output` (reasoning tokens are
  already counted within completion tokens).

### Cost Tracking Example

```julia
chat = Chat(model="gpt-5.2")
push!(chat, Message(Val(:system), "You are helpful."))
push!(chat, Message(Val(:user), "Hello!"))

result = chatrequest!(chat)
if result isa LLMSuccess
    usage = token_usage(result)
    cost = estimated_cost(result)
    println("Tokens: $(usage.total_tokens), Cost: \$$(round(cost; digits=6))")
    println("Cumulative: \$$(round(cumulative_cost(chat); digits=6))")
end
```

---

## Conversation Forking

```julia
fork(chat::Chat)::Chat          # deep-copy a Chat; cumulative cost is copied by value (independent Ref)
fork(chat::Chat, n::Int)::Vector{Chat}  # create n independent forks
```

### Fork Example

```julia
chat = Chat(model="gpt-5.2")
push!(chat, Message(Val(:system), "You are a creative writer."))
push!(chat, Message(Val(:user), "Start a story about a robot."))
chatrequest!(chat)

# Fork into 3 independent continuations
forks = fork(chat, 3)
for (i, f) in enumerate(forks)
    push!(f, Message(Val(:user), "Continue the story with ending $i."))
    chatrequest!(f)
end
```

---

## Tool Loop

Automated tool dispatch for both APIs. Wraps a tool schema with a callable function.

### CallableTool

```julia
struct CallableTool{T}
    tool::T              # Tool or FunctionTool
    callable::Function   # (name::String, args::Dict{String,Any}) -> String
end
```

### to_tool

```julia
to_tool(x)  # identity for Tool, FunctionTool, CallableTool; converts AbstractDict to Tool
```

### ToolCallOutcome / ToolLoopResult

```julia
# Per-call record
struct ToolCallOutcome
    tool_name::String
    arguments::Dict{String,Any}
    result::Union{FunctionCallResult,Nothing}
    success::Bool
    error::Union{String,Nothing}
end

# Loop result
struct ToolLoopResult
    response::LLMRequestResponse
    tool_calls::Vector{ToolCallOutcome}
    turns_used::Int
    completed::Bool
    llm_error::Union{String,Nothing}
end
```

### tool_loop! (Chat Completions)

```julia
tool_loop!(chat, dispatcher; max_turns=10, config=nothing, callback=nothing, on_tool_call=nothing) -> ToolLoopResult
tool_loop!(chat; tools::Vector{<:CallableTool}, kwargs...) -> ToolLoopResult   # kwargs as above
```

`callback` and `on_tool_call` are the streaming hooks from `chatrequest!`, applied
to every turn of the loop.

### tool_loop (Responses API)

```julia
tool_loop(r::Respond, dispatcher; max_turns=10, config=nothing) -> ToolLoopResult
tool_loop(r::Respond; max_turns=10, config=nothing) -> ToolLoopResult   # extracts callables from r.tools
tool_loop(input, dispatcher; tools, kwargs...) -> ToolLoopResult    # convenience form
```

---

## MCP Client

Native MCP client (JSON-RPC 2.0 over stdio or HTTP, spec 2025-11-25).

### Types

```julia
MCPSession            # live connection — manages transport + cached tools/resources/prompts
MCPToolInfo           # tool definition from tools/list
MCPToolResult         # typed tools/call result: content, structured, is_error, parts
MCPResourceInfo       # resource definition from resources/list
MCPPromptInfo         # prompt definition from prompts/list
MCPServerCapabilities # capabilities from initialize
MCPTransport          # abstract (subtypes: StdioTransport, HTTPTransport)
MCPError <: Exception # JSON-RPC error (code, message, data)
MCPCrashError <: Exception # stdio server process exited / pipe broke (exitcode, termsignal, cause)
```

### Lifecycle

```julia
mcp_connect(command::Cmd; config=nothing, auto_respawn=false, ...) -> MCPSession        # stdio subprocess
mcp_connect(url::String; headers=[], config=nothing, auto_respawn=false, ...) -> MCPSession  # HTTP
mcp_connect(f::Function, args...; ...)            # do-block, auto-disconnect
mcp_disconnect!(session)
```

### Discovery

```julia
list_tools!(session; timeout=nothing) -> Vector{MCPToolInfo}
list_resources!(session) -> Vector{MCPResourceInfo}
list_prompts!(session) -> Vector{MCPPromptInfo}
```

When the server sends `notifications/tools/list_changed`, `session.tools_stale` is set to `true` to flag the cached tool list as out of date; call `list_tools!(session)` to refresh it (which clears the flag).

### Operations

```julia
call_tool(session, name, arguments; timeout=nothing) -> MCPToolResult
read_resource(session, uri) -> String
get_prompt(session, name, arguments) -> Vector{Dict}
ping(session)
```

`call_tool` and `get_prompt` accept any `AbstractDict` for `arguments`, so the
natural `Dict("path" => "/tmp/x")` form works without conversion.

Timeouts surface as the exported `MCPTimeoutError` (`phase` `:connect`/`:request`);
a stdio request timeout closes the session (opt-in `auto_respawn` respawns on the
next call), an HTTP request timeout does not. A stdio server that exits or whose
pipe breaks surfaces the exported `MCPCrashError` (the session closes with cause
`:crash`; `auto_respawn` respawns the server on the next call).

### Session Concurrency (concurrency-1)

An `MCPSession` serializes everything: each call's liveness check, id allocation
and request/response exchange runs under one session lock, so a concurrent caller
waits for the exchange in progress.

- The queued caller's `mcp_request_timeout` is measured **from the moment it takes
  the lock**, not from when it asked — waiting behind another call never counts
  against its own bound, and the wait stays bounded transitively because the call
  ahead runs under that same per-exchange bound.
- `mcp_disconnect!` takes the lock too, so a disconnect racing a call in flight
  **waits** for that exchange to finish instead of tearing the transport down
  under its reader.
- Interleaved server → client frames are handled in place: notifications are
  skipped and server `ping` requests answered.

For genuine parallelism, open one session per concurrent worker.

### Tool Result

`call_tool` returns a typed result — a tool-execution error (`isError: true`) is
carried on `is_error`, **not** thrown; only JSON-RPC protocol errors throw (`MCPError`).

```julia
struct MCPToolResult
    content::String                              # text parts joined by "\n"; non-text parts JSON-encoded
    structured::Union{Nothing,Dict{String,Any}}  # server's structuredContent verbatim, or nothing
    is_error::Bool                               # true when the tool reported an execution error (isError)
    parts::Vector{Any}                           # raw content array verbatim
end
```

The `mcp_tools` / `mcp_tools_respond` bridges surface `content` to the model on
success (falling back to `JSON.json(structured)` when `content` is empty), and
raise `content` as an error when `is_error` is set.

### Tool Bridge

```julia
mcp_tools(session) -> Vector{CallableTool{Tool}}         # for tool_loop!
mcp_tools_respond(session) -> Vector{CallableTool{FunctionTool}}  # for tool_loop
```

### Client Example

```julia
session = mcp_connect(`npx -y @modelcontextprotocol/server-filesystem /tmp`)
tools = mcp_tools(session)
chat = Chat(model="gpt-5.2", tools=map(t -> t.tool, tools))
push!(chat, Message(Val(:system), "You are a helpful assistant with filesystem access."))
push!(chat, Message(Val(:user), "List files"))
result = tool_loop!(chat; tools)
mcp_disconnect!(session)
```

---

## MCP Server

Build MCP servers that expose tools, resources, and prompts.

### Types

```julia
MCPServer(name, version; description=nothing)
MCPServerPrimitive    # abstract (MCPServerTool, MCPServerResource, MCPServerResourceTemplate, MCPServerPrompt)
```

### Registration

```julia
register_tool!(server, name, description, schema, handler)
register_tool!(server, name, description, handler)           # auto-schema from signature
register_tool!(server, ct::CallableTool{Tool})            # bridge from Chat API
register_tool!(server, ct::CallableTool{FunctionTool})       # bridge from Responses API
register_resource!(server, uri, name, handler; mime_type="text/plain", description=nothing)
register_resource_template!(server, uri_template, name, handler; ...)
register_prompt!(server, name, handler; description=nothing, arguments=[])
```

### Macros

```julia
@mcp_tool server function name(args...) body end
@mcp_resource server uri function(args...) body end
@mcp_prompt server name function(args...) body end
```

### Serving

```julia
serve(server; transport=:stdio)                          # default — stdio
serve(server; transport=:http, host="127.0.0.1", port=8080)  # HTTP — blocks until closed
```

The HTTP transport blocks until the server is closed (like `HTTP.serve`); pass
`block=false` to get the running server handle back and `close` it yourself. It
also validates the `Origin` header (DNS-rebinding defense): requests with no
`Origin` header and localhost origins pass, any other origin gets 403 unless
listed in `allowed_origins`.

```julia
handle = serve(server; transport=:http, port=8080, block=false,
               allowed_origins=["https://app.example.com"])
close(handle)
```

### Error and Limit Contracts

- **A throwing tool handler is a tool result, not a protocol error.** The
  exception's message is returned to the client as tool content with
  `isError: true`, which is exactly what lets a model see and correct its own
  mistake. Write handlers accordingly: raise with a message you are willing to
  show the model *and* the client, since it is relayed verbatim.
- **Dispatch-layer errors are generic.** Any unhandled error below the handler
  answers JSON-RPC `-32603` with the message `"Internal error"` — the exception
  and backtrace go to the server's own logs, not to the peer, because an
  exception string can carry file paths and argument values a remote client has
  no business reading. A single bad frame never takes the transport down.
- **Frames and request bodies are capped at 16 MiB** on both transports. An
  oversized frame is answered with `-32600` rather than parsed, since parsing an
  attacker-sized payload allocates a multiple of it — an out-of-memory kill
  rather than a protocol error.
- The server tolerates spec-legal parameter shapes: `params: null` and positional
  `params` do not crash it.

### Server Example

```julia
server = MCPServer("calc", "1.0.0")
@mcp_tool server function add(a::Float64, b::Float64)::String
    string(a + b)
end
serve(server)
```

---

## FIM Completion

Fill-in-the-Middle: generate text between a `prompt` (prefix) and `suffix`.
Supported by DeepSeek (beta), Ollama, vLLM.

```julia
@kwdef struct FIMCompletion
    service::ServiceEndpointSpec
    model::String = ""    # sentinel: resolves at serialization to "deepseek-chat" for DeepSeek
    prompt::String
    suffix::Union{String,Nothing} = nothing
    max_tokens::Union{Int,Nothing} = 128
    # temperature, top_p, stream, stop, echo, logprobs, frequency_penalty, presence_penalty
end

struct FIMChoice; text, index, finish_reason; end
struct FIMResponse; choices, usage, model, raw; end
struct FIMSuccess <: LLMRequestResponse; response::FIMResponse; end
struct FIMFailure <: LLMRequestResponse; response, status, request_id; end
struct FIMCallError <: LLMRequestResponse; error, status, request_id, cause; end

fim_complete(fim::FIMCompletion; config=nothing) -> LLMRequestResponse
fim_complete(prompt; suffix=nothing, kwargs...) -> LLMRequestResponse  # convenience
fim_text(result) -> String  # extract generated text
```

Unlike `Chat`, `FIMCompletion` does not resolve its model at construction — the
field stays `""` until serialization, where it becomes `"deepseek-chat"` for
`DeepSeekEndpoint`. Endpoints without a default FIM model must set `model=`
explicitly, or serialization throws `ArgumentError`.

### FIM Example

```julia
result = fim_complete("def fib(a):",
    service=DeepSeekEndpoint(), suffix="    return fib(a-1) + fib(a-2)",
    max_tokens=128, stop=["\n\n"])
println(fim_text(result))
```

---

## Chat Prefix Completion

Continue from a partial assistant message. The model generates text continuing
from the assistant's prefix. DeepSeek beta feature.

```julia
prefix_complete(chat::Chat; config=nothing) -> LLMRequestResponse
# Last message must be role=assistant with the prefix text
```

### Prefix Example

```julia
chat = Chat(service=DeepSeekEndpoint(), model="deepseek-chat")
push!(chat, Message(Val(:system), "You are a coding assistant."))
push!(chat, Message(Val(:user), "Write quicksort in Python"))
push!(chat, Message(role=RoleAssistant, content="```python\n"))
result = prefix_complete(chat)
```

---

## Provider Capabilities

Each endpoint declares supported features. Request functions validate before
dispatch, so an unsupported feature is an `ArgumentError` naming what the provider
does support, not an HTTP 404.

```julia
provider_capabilities(service) -> Set{Symbol}
has_capability(service, cap::Symbol) -> Bool
```

### Capabilities by Provider

| Provider | Capabilities |
|---|---|
| OpenAI | `:chat`, `:responses`, `:agentic`, `:embeddings`, `:images`, `:image_edits`, `:tools`, `:json_output`, `:files`, `:vector_stores`, `:conversations`, `:moderation`, `:audio`, `:batch`, `:fine_tuning`, `:containers`, `:uploads`, `:video`, `:realtime` |
| Azure | `:chat`, `:tools` |
| Gemini (native) | `:chat`, `:tools`, `:streaming`, `:agentic` |
| Gemini (OpenAI-compat) | `:chat`, `:embeddings`, `:tools`, `:json_output` |
| Anthropic (native) | `:chat`, `:tools`, `:json_output`, `:streaming` |
| DeepSeek | `:chat`, `:tools`, `:fim`, `:prefix_completion`, `:json_output` |
| Generic | `:chat`, `:embeddings`, `:fim`, `:tools`, `:responses` |

Validation comes in two strengths, and the difference matters if you write your
own endpoint:

- **Platform and lifecycle verbs** (files, vector stores, conversations,
  moderations, audio, batch, fine-tuning, containers, uploads, video, realtime,
  FIM, prefix completion, image *edits*) validate **strictly**: an endpoint with
  no declaration at all is rejected, because those surfaces are OpenAI-shaped and
  a non-OpenAI backend would just 404.
- **The four primary verbs** — `chatrequest!` (`:chat`), `embeddingrequest!`
  (`:embeddings`), `respond` (`:responses` **or** `:agentic`), and
  `generate_image` (`:images`) — validate only endpoints that **declare** their
  capabilities. A custom endpoint that defines no `provider_capabilities` method
  passes through unvalidated: the package has no basis for claiming what someone
  else's OpenAI-compatible server cannot do, and refusing to dispatch it would be
  a false negative. Declaring capabilities is therefore opt-in strictness — once
  you declare, you are held to the declaration.

Either way you never need to check capabilities manually before a request.

---

## Request Configuration & Timeouts

Every network operation is bounded. One struct carries all knobs — all time
fields are seconds, `Inf` disables that bound:

```julia
Base.@kwdef struct RequestConfig
    connect_timeout::Float64     = 10.0
    request_timeout::Float64     = 600.0
    stream_idle_timeout::Float64 = 120.0
    total_deadline::Float64      = 900.0
    max_attempts::Int            = 3
    mcp_connect_timeout::Float64 = 120.0
    mcp_request_timeout::Float64 = 120.0
end
```

- `connect_timeout` — per-attempt connection establishment. `request_timeout` — per-attempt whole exchange (non-stream). `stream_idle_timeout` — byte-gap between raw stream chunks. `total_deadline` — across ALL attempts including backoff (streams: until first byte). `max_attempts` — wire attempts (`1` disables retries). `mcp_connect_timeout` / `mcp_request_timeout` — MCP handshake / per-exchange bounds.
- **Validation**: every `Float64` field rejects `NaN` and values `≤ 0` with `ArgumentError` (`Inf` = disabled); `max_attempts ≥ 1`.
- **Copy-with-overrides**: `RequestConfig(base::RequestConfig; kwargs...)`.
- **Four channels, struct-wise precedence** (a channel supplies a complete struct):
  1. per-call `config::Union{Nothing,RequestConfig}` keyword on request verbs;
  2. dynamic scope: `with_request_config(f; kwargs...)` — merges kwargs over `current_config()` at entry; propagates into `Threads.@spawn`;
  3. process default: `set_default_config!(cfg)` or `set_default_config!(; kwargs...)` (merges over the current default) — the channel for REPL/notebook sessions;
  4. the built-in defaults above.
  `current_config()` returns the ambient struct (active scope, else process default).

```julia
with_request_config(request_timeout=30.0, max_attempts=1) do
    chatrequest!(chat)          # bounded, no retries, inside this scope only
end
set_default_config!(total_deadline=120.0)   # process-wide default
```

### Which verbs retry

`max_attempts` applies to the inference verbs only: `chatrequest!`,
`embeddingrequest!`, `respond`, `fim_complete`, `prefix_complete`,
`generate_image`, `edit_image`, `upload_file`, and the `tool_loop` family
(streams retry only before the first callback fires). Platform and lifecycle verbs
— batch, container, conversation, file, fine-tuning, moderation, upload,
vector-store, video, audio, realtime, and the Responses lifecycle operations —
make a **single bounded attempt**, and `max_attempts` has no effect on them.

### Sharp edges

- **`connect_timeout = Inf` is unsupported on the HTTP 1.x major.** A breached
  task-mode watchdog abandons its worker rather than killing it, relying on the
  attempt's native bound to end it; on 1.x `connect_timeout` is the only native
  bound covering connection acquisition, so disabling it can leave a worker that
  never self-terminates. Safe on the 2.x major, whose request bound also covers
  acquisition.
- **Streams, pre-first-byte, on HTTP 2.x** are additionally bounded by
  `stream_idle_timeout` (that major caps the response-header wait with the
  read-idle timer), so the effective bound is
  `min(total_deadline, request_timeout, stream_idle_timeout)` and a breach reports
  phase `:stream_idle` even though no byte arrived.
- **A completed turn is never discarded or re-sent.** Once a stream records its
  terminal state, teardown noise — idle breach, transport reset, truncated read —
  finalizes it as a success rather than failing or retrying it, on every provider
  and both stream drivers. Re-POSTing would bill a second generation.
- **Realtime**: `realtime_connect` bounds the open phase with `connect_timeout`;
  `realtime_receive` bounds the read with `stream_idle_timeout` and surfaces a
  breach within `[limit, limit + ~5 s]` (the WebSocket close waits for the peer's
  acknowledgement). An open session's lifetime is deliberately unbounded.

### Concurrency

- **One `Chat` per in-flight call.** `messages` and the cumulative-cost `Ref` are
  unsynchronized by design; use `fork(chat)` / `fork(chat, n)` to fan out.
  `respond`, `embeddingrequest!` and `generate_image` take a fresh request struct
  per call and need no such care.
- **An `MCPSession` is concurrency-1** (whole-exchange lock; a queued caller's
  bound starts when it takes the lock; `mcp_disconnect!` waits for the in-flight
  exchange). One session per concurrent worker.
- **HTTP 1.x pools connections process-globally across hosts**, capped at
  `max(16, 4 × Threads.nthreads())`; HTTP 2.x pools per host with no default cap.
  Prefer HTTP 2.x for high fan-out.

---

## Result Type Hierarchy

All 66 API call result types inherit from `LLMRequestResponse`, in 16 per-API
families:

```
LLMRequestResponse   (abstract parent of every result type below)
│
├─ Chat Completions   LLMSuccess · LLMFailure · LLMCallError
├─ Responses API      ResponseSuccess · ResponseFailure · ResponseCallError
├─ Embeddings         EmbeddingSuccess · EmbeddingFailure · EmbeddingCallError
├─ Image Generation   ImageSuccess · ImageFailure · ImageCallError
├─ FIM Completion     FIMSuccess · FIMFailure · FIMCallError
├─ Files              FileSuccess · FileListSuccess · FileContentSuccess · FileDeleteSuccess · FileFailure · FileCallError
├─ Vector Stores      VectorStoreSuccess · VectorStoreListSuccess · VectorStoreFileSuccess · VectorStoreBatchSuccess · VectorStoreDeleteSuccess · VectorStoreFailure · VectorStoreCallError
├─ Conversations      ConversationSuccess · ConversationItemSuccess · ConversationItemListSuccess · ConversationDeleteSuccess · ConversationFailure · ConversationCallError
├─ Moderations        ModerationSuccess · ModerationFailure · ModerationCallError
├─ Audio              SpeechSuccess · TranscriptionSuccess · AudioFailure · AudioCallError
├─ Batch              BatchSuccess · BatchListSuccess · BatchFailure · BatchCallError
├─ Fine-tuning        FineTuningSuccess · FineTuningListSuccess · FineTuningFailure · FineTuningCallError
├─ Containers         ContainerSuccess · ContainerListSuccess · ContainerDeleteSuccess · ContainerFailure · ContainerCallError
├─ Uploads            UploadSuccess · UploadPartSuccess · UploadFailure · UploadCallError
├─ Videos             VideoSuccess · VideoListSuccess · VideoContentSuccess · VideoFailure · VideoCallError
└─ Realtime           RealtimeSecretSuccess · RealtimeFailure · RealtimeCallError
```

Every `*Success` wraps a parsed response object, every `*Failure` carries the HTTP
status and body, and every `*CallError` wraps a transport/exception. Fields of the
five primary families:

```julia
LLMSuccess        (.message::Message, .self::Chat, .usage::Union{TokenUsage,Nothing})
LLMFailure        (.response::String, .status::Int, .self::Chat, .request_id::Union{String,Nothing})
LLMCallError      (.error::String, .status::Union{Int,Nothing}, .self::Chat, .request_id::Union{String,Nothing}, .cause::Union{Nothing,Exception})
ResponseSuccess   (.response::ResponseObject)
ResponseFailure   (.response::String, .status::Int, .request_id::Union{String,Nothing})
ResponseCallError (.error::String, .status::Union{Int,Nothing}, .request_id::Union{String,Nothing}, .cause::Union{Nothing,Exception})
EmbeddingSuccess  (.embeddings::Embeddings, .usage::Union{TokenUsage,Nothing}, .raw::Dict{String,Any})
EmbeddingFailure  (.response::String, .status::Int)
EmbeddingCallError(.error::String, .status::Union{Int,Nothing}, .cause::Union{Nothing,Exception})
ImageSuccess      (.response::ImageResponse)
ImageFailure      (.response::String, .status::Int)
ImageCallError    (.error::String, .status::Union{Int,Nothing})
FIMSuccess        (.response::FIMResponse)
FIMFailure        (.response::String, .status::Int, .request_id::Union{String,Nothing})
FIMCallError      (.error::String, .status::Union{Int,Nothing}, .request_id::Union{String,Nothing}, .cause::Union{Nothing,Exception})
```

`request_id` carries the provider's `x-request-id` header for support escalation.
It exists on the six Chat / Responses / FIM failure and call-error types above;
the image and embedding limbs do not carry it.

**Credentials never reach a result value.** API keys and request bodies are
redacted from the `error` strings these types carry, and a `*CallError`'s
`cause` is named by type rather than dumped when the result is displayed.

**Standard pattern-matching idiom**:

```julia
result = chatrequest!(chat)
if result isa LLMSuccess
    println(result.message.content)
elseif result isa LLMFailure
    @error "HTTP $(result.status): $(result.response)"
elseif result isa LLMCallError
    @error "Exception: $(result.error)"
end

result = respond("Hello")
if result isa ResponseSuccess
    println(output_text(result))
elseif result isa ResponseFailure
    @error "HTTP $(result.status)"
elseif result isa ResponseCallError
    @error result.error
end

result = generate_image("A cat")
if result isa ImageSuccess
    save_image(image_data(result)[1], "cat.png")
elseif result isa ImageFailure
    @error "HTTP $(result.status)"
elseif result isa ImageCallError
    @error result.error
end
```

---

## API Constants

```julia
const OPENAI_BASE_URL = "https://api.openai.com"
const CHAT_COMPLETIONS_PATH = "/v1/chat/completions"
const EMBEDDINGS_PATH = "/v1/embeddings"
const RESPONSES_PATH = "/v1/responses"
const IMAGES_GENERATIONS_PATH = "/v1/images/generations"
const COMPLETIONS_PATH = "/v1/completions"                     # FIM endpoint
const DEEPSEEK_BASE_URL = "https://api.deepseek.com"
const DEEPSEEK_BETA_BASE_URL = "https://api.deepseek.com/beta"
const GEMINI_CHAT_URL = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
```

## Exceptions

```julia
struct InvalidConversationError <: Exception
    reason::String
end
```

Thrown by `push!` (a non-system message onto an empty `Chat`, a system message after the conversation started, or a same-role repeat other than `tool`) and by `pop!` on an empty `Chat`. `issendvalid` never throws — it returns a `Bool`.

```julia
struct LLMResultError <: Exception
    result::Union{LLMFailure,LLMCallError}
end
```

Thrown by `text(result)` when the result is not a success. `showerror` prints only the status and a trimmed (≤200-char) response excerpt — never the conversation, endpoint, or API key.

```julia
struct UniLMTimeout <: Exception
    phase::Symbol        # :connect | :request | :stream_idle | :deadline
    elapsed::Float64     # seconds, monotonic
    limit::Float64
end

struct MCPTimeoutError <: Exception
    phase::Symbol        # :connect | :request
    elapsed::Float64
    limit::Float64
    msg::String          # names the applicable timeout override
end

struct MCPCrashError <: Exception
    msg::String                          # recovery guidance (embeds exit info)
    exitcode::Union{Int,Nothing}         # exit code if the server exited and was reaped in time
    termsignal::Union{Int,Nothing}       # signal number if the server was signal-killed
    cause::Union{Exception,Nothing}      # underlying transport exception; nothing on clean read-EOF
end
```

Raised when a `RequestConfig` bound is exceeded. Value-returning surfaces
(chat, embeddings, responses) deliver `UniLMTimeout` inside their error
results; the MCP surface throws `MCPTimeoutError`. A stdio MCP server that exits
or whose pipe breaks throws `MCPCrashError` instead (the session closes with
cause `:crash`).

---

## Complete Exports List

Every exported symbol (`names(UniLM)`), grouped by area:

**Chat Completions**: `Chat`, `Message`, `ProviderContent`, `RoleSystem`, `RoleUser`, `RoleAssistant`, `Tool`, `ToolCall`, `FunctionSignature`, `FunctionCallResult`, `ResponseFormat`, `InvalidConversationError`, `issendvalid`, `chatrequest!`, `update!`, `fork`
- *Legacy aliases* (pre-rename names, exported and non-breaking, retained until 1.0): `GPTTool` → `Tool`, `GPTToolCall` → `ToolCall`, `GPTFunctionSignature` → `FunctionSignature`, `GPTFunctionCallResult` → `FunctionCallResult`

**Responses API & Agentic**: `Respond`, `InputMessage`, `ResponseObject`, `ResponseSuccess`, `ResponseFailure`, `ResponseCallError`, `Reasoning`, `TextConfig`, `TextFormatSpec`, `respond`, `get_response`, `delete_response`, `cancel_response`, `list_input_items`, `compact_response`, `count_input_tokens`, `text_format`, `json_schema_format`, `json_object_format`
- *Input builders*: `input_text`, `input_image`, `input_file`
- *Tool types*: `ResponseTool`, `FunctionTool`, `WebSearchTool`, `FileSearchTool`, `MCPTool`, `ComputerUseTool`, `ComputerTool`, `ImageGenerationTool`, `CodeInterpreterTool`, `LocalShellTool`, `ShellTool`, `ApplyPatchTool`, `CustomTool`
- *Tool constructors*: `function_tool`, `web_search`, `file_search`, `mcp_tool`, `computer_use`, `computer_tool`, `image_generation_tool`, `code_interpreter`, `local_shell`, `shell`, `apply_patch_tool`, `custom_tool`, `tool_result`, `mcp_approval_response`
- *Hosted Gemini tools*: `gemini_google_search`, `gemini_code_execution`, `gemini_url_context`
- *tool_choice builders*: `tool_choice_function`, `tool_choice_hosted`, `tool_choice_mcp`, `tool_choice_custom`, `tool_choice_allowed`
- *Result accessors*: `output_text`, `function_calls`, `refusals`, `reasoning_summaries`, `reasoning_items`, `url_citations`, `web_search_results`, `file_search_results`, `image_generation_results`, `code_interpreter_outputs`, `mcp_call_outputs`, `mcp_approval_requests`, `response_status`, `incomplete_details`, `usage_details`

**Image Generation & Edits**: `ImageGeneration`, `ImageEdit`, `ImageObject`, `ImageResponse`, `ImageSuccess`, `ImageFailure`, `ImageCallError`, `generate_image`, `edit_image`, `image_data`, `save_image`

**Embeddings**: `Embeddings`, `embeddingrequest!`, `embedding_vectors`, `EmbeddingSuccess`, `EmbeddingFailure`, `EmbeddingCallError`

**Cost Tracking**: `TokenUsage`, `token_usage`, `estimated_cost`, `cumulative_cost`, `DEFAULT_PRICING`

**Service Endpoints**: `ServiceEndpoint`, `OpenAIWireEndpoint`, `ServiceEndpointSpec`, `OPENAIServiceEndpoint`, `AZUREServiceEndpoint`, `GEMINIServiceEndpoint`, `GEMINIOpenAIServiceEndpoint`, `ANTHROPICServiceEndpoint`, `GenericOpenAIEndpoint`, `OllamaEndpoint`, `MistralEndpoint`, `DeepSeekEndpoint`, `add_azure_deploy_name!`

**Provider Capabilities**: `provider_capabilities`, `has_capability`

**Tool Loop**: `CallableTool`, `ToolCallOutcome`, `ToolLoopResult`, `tool_loop!`, `tool_loop`, `to_tool`

**Forking**: `fork`

**MCP Client**: `MCPSession`, `MCPToolInfo`, `MCPToolResult`, `MCPResourceInfo`, `MCPPromptInfo`, `MCPServerCapabilities`, `MCPTransport`, `StdioTransport`, `HTTPTransport`, `MCPError`, `MCPCrashError`, `mcp_connect`, `mcp_disconnect!`, `mcp_tools`, `mcp_tools_respond`, `list_tools!`, `list_resources!`, `list_prompts!`, `call_tool`, `read_resource`, `get_prompt`, `ping`

**MCP Server**: `MCPServer`, `MCPServerTool`, `MCPServerResource`, `MCPServerResourceTemplate`, `MCPServerPrompt`, `MCPServerPrimitive`, `register_tool!`, `register_resource!`, `register_resource_template!`, `register_prompt!`, `serve`, `@mcp_tool`, `@mcp_resource`, `@mcp_prompt`

**FIM / Completions**: `FIMCompletion`, `FIMChoice`, `FIMResponse`, `FIMSuccess`, `FIMFailure`, `FIMCallError`, `fim_complete`, `fim_text`, `prefix_complete`

**Files**: `FileUpload`, `FileObject`, `FileList`, `FileSuccess`, `FileListSuccess`, `FileContentSuccess`, `FileDeleteSuccess`, `FileFailure`, `FileCallError`, `upload_file`, `list_files`, `retrieve_file`, `delete_file`, `file_content`, `save_file_content`

**Vector Stores**: `VectorStoreObject`, `VectorStoreFileObject`, `VectorStoreFileBatch`, `VectorStoreList`, `VectorStoreSuccess`, `VectorStoreListSuccess`, `VectorStoreFileSuccess`, `VectorStoreBatchSuccess`, `VectorStoreDeleteSuccess`, `VectorStoreFailure`, `VectorStoreCallError`, `create_vector_store`, `retrieve_vector_store`, `list_vector_stores`, `delete_vector_store`, `add_vector_store_file`, `create_file_batch`, `retrieve_file_batch`, `poll_file_batch`, `vector_store_id`

**Conversations**: `ConversationObject`, `ConversationItem`, `ConversationItemList`, `ConversationSuccess`, `ConversationItemSuccess`, `ConversationItemListSuccess`, `ConversationDeleteSuccess`, `ConversationFailure`, `ConversationCallError`, `create_conversation`, `retrieve_conversation`, `update_conversation`, `delete_conversation`, `add_conversation_items`, `list_conversation_items`, `delete_conversation_item`, `conversation_id`

**Moderations**: `ModerationResponse`, `ModerationResult`, `ModerationSuccess`, `ModerationFailure`, `ModerationCallError`, `moderate`, `is_flagged`

**Audio**: `SpeechRequest`, `TranscriptionRequest`, `SpeechSuccess`, `TranscriptionSuccess`, `AudioFailure`, `AudioCallError`, `speak`, `save_audio`, `transcribe`, `translate`, `transcript_text`

**Batch**: `BatchObject`, `BatchList`, `BatchSuccess`, `BatchListSuccess`, `BatchFailure`, `BatchCallError`, `create_batch`, `retrieve_batch`, `cancel_batch`, `list_batches`, `poll_batch`

**Fine-tuning**: `FineTuningJob`, `FineTuningList`, `FineTuningSuccess`, `FineTuningListSuccess`, `FineTuningFailure`, `FineTuningCallError`, `create_fine_tuning_job`, `retrieve_fine_tuning_job`, `cancel_fine_tuning_job`, `list_fine_tuning_jobs`, `list_fine_tuning_events`, `list_fine_tuning_checkpoints`

**Containers**: `ContainerObject`, `ContainerList`, `ContainerSuccess`, `ContainerListSuccess`, `ContainerDeleteSuccess`, `ContainerFailure`, `ContainerCallError`, `create_container`, `retrieve_container`, `list_containers`, `delete_container`, `add_container_file`

**Uploads**: `UploadObject`, `UploadPartObject`, `UploadSuccess`, `UploadPartSuccess`, `UploadFailure`, `UploadCallError`, `create_upload`, `add_upload_part`, `complete_upload`, `cancel_upload`

**Videos**: `VideoObject`, `VideoList`, `VideoSuccess`, `VideoListSuccess`, `VideoContentSuccess`, `VideoFailure`, `VideoCallError`, `create_video`, `retrieve_video`, `list_videos`, `video_content`

**Webhooks**: `WebhookEvent`, `WEBHOOK_EVENTS`, `verify_webhook`, `parse_webhook`

**Realtime**: `RealtimeSession`, `RealtimeSecretSuccess`, `RealtimeFailure`, `RealtimeCallError`, `mint_realtime_secret`, `realtime_connect`, `realtime_send`, `realtime_receive`, `realtime_event`, `session_update`, `input_audio_append`, `response_create`

**Result Types (base)**: `LLMRequestResponse`, `LLMSuccess`, `LLMFailure`, `LLMCallError`, `LLMResultError`, `issuccess`, `isfailure`, `text`

**Request Config & Timeouts**: `RequestConfig`, `current_config`, `with_request_config`, `set_default_config!`, `UniLMTimeout`, `MCPTimeoutError`
