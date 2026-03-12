# UniLM.jl — LLM Reference

> **Single-file reference for LLM code-generation systems.**
> UniLM.jl v0.7.0 · Julia ≥ 1.12 · Deps: `HTTP.jl`, `JSON.jl`, `Base64`
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

## Four APIs

UniLM.jl wraps four OpenAI API surfaces:

1. **Chat Completions** (`Chat` + `chatrequest!`) — stateful, message-based conversations with tool calling, streaming, structured output. Supports OpenAI, Azure, and Gemini backends.
2. **Responses API** (`Respond` + `respond`) — newer, more flexible API with built-in tools (web search, file search), multi-turn chaining via `previous_response_id`, reasoning support for O-series models, structured output. OpenAI only.
3. **Image Generation** (`ImageGeneration` + `generate_image`) — text-to-image with `gpt-image-1.5`. OpenAI only.
4. **Embeddings** (`Embeddings` + `embeddingrequest!`) — vector embeddings with `text-embedding-3-small`.

---

## Service Endpoints

```julia
abstract type ServiceEndpoint end
struct OPENAIServiceEndpoint <: ServiceEndpoint end   # default — uses OPENAI_API_KEY
struct AZUREServiceEndpoint  <: ServiceEndpoint end   # uses AZURE_OPENAI_* env vars
struct GEMINIServiceEndpoint <: ServiceEndpoint end   # uses GEMINI_API_KEY
```

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
    service::Type{<:ServiceEndpoint} = OPENAIServiceEndpoint
    model::String = "gpt-5.2"
    messages::Vector{Message} = Message[]
    history::Bool = true
    tools::Union{Vector{GPTTool},Nothing} = nothing
    tool_choice::Union{String,GPTToolChoice,Nothing} = nothing
    parallel_tool_calls::Union{Bool,Nothing} = false
    temperature::Union{Float64,Nothing} = nothing       # 0.0–2.0, mutually exclusive with top_p
    top_p::Union{Float64,Nothing} = nothing              # 0.0–1.0, mutually exclusive with temperature
    n::Union{Int64,Nothing} = nothing
    stream::Union{Bool,Nothing} = nothing
    stop::Union{Vector{String},String,Nothing} = nothing # max 4 sequences
    max_tokens::Union{Int64,Nothing} = nothing
    presence_penalty::Union{Float64,Nothing} = nothing   # -2.0 to 2.0
    response_format::Union{ResponseFormat,Nothing} = nothing
    frequency_penalty::Union{Float64,Nothing} = nothing  # -2.0 to 2.0
    logit_bias::Union{AbstractDict{String,Float64},Nothing} = nothing
    user::Union{String,Nothing} = nothing
    seed::Union{Int64,Nothing} = nothing
end
```

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
    tool_calls::Union{Nothing,Vector{GPTToolCall}} = nothing
    tool_call_id::Union{String,Nothing} = nothing         # required when role == "tool"
end
```

**Validation**: at least one of `content`, `tool_calls`, or `refusal_message` must be non-`nothing`. `tool_call_id` is required when `role == "tool"`.

**Convenience constructors**:

```julia
Message(Val(:system), "You are a helpful assistant")
Message(Val(:user), "Hello!")
```

**Role constants**: `RoleSystem = "system"`, `RoleUser = "user"`, `RoleAssistant = "assistant"`.

### chatrequest!

```julia
# Mutating form — sends chat.messages, appends response when history=true
chatrequest!(chat::Chat; retries::Int=0, callback=nothing) -> LLMSuccess | LLMFailure | LLMCallError | Task

# Keyword-argument convenience form — builds a Chat internally
chatrequest!(; service=OPENAIServiceEndpoint, model="gpt-5.2",
    systemprompt, userprompt, messages=Message[], history=true,
    tools=nothing, tool_choice=nothing, temperature=nothing, ...) -> same
```

- Non-streaming: returns `LLMSuccess`, `LLMFailure`, or `LLMCallError`.
- Streaming (`stream=true`): returns a `Task`. Pass a `callback(chunk::Union{String,Message}, close::Ref{Bool})`.
- Auto-retries on HTTP 429/500/503 with exponential backoff and jitter (up to 30 attempts). Respects `Retry-After` headers on 429 responses.

### Conversation Management

```julia
push!(chat, message)       # append a Message
pop!(chat)                 # remove last message
update!(chat, message)     # append if history=true
issendvalid(chat) -> Bool  # check conversation rules (≥1 message, system first if present, etc.)
length(chat)               # number of messages
isempty(chat)              # true if no messages
chat[i]                    # index into messages
```

### Tool Calling Types

```julia
# Define a function the model can call
@kwdef struct GPTFunctionSignature
    name::String
    description::Union{String,Nothing} = nothing
    parameters::Union{AbstractDict,Nothing} = nothing   # JSON Schema dict
end

# Wrap it for the tools parameter
@kwdef struct GPTTool
    type::String = "function"
    func::GPTFunctionSignature
end
GPTTool(d::AbstractDict)   # construct from dict with keys "name", "description", "parameters"

# Returned by model when it wants to call a function
@kwdef struct GPTToolCall
    id::String
    type::String = "function"
    func::GPTFunction       # has .name::String and .arguments::AbstractDict
end

# Your result after executing the function
struct GPTFunctionCallResult{T}
    name::Union{String,Symbol}
    origincall::GPTFunction
    result::T
end
```

### ResponseFormat (Structured Output)

```julia
@kwdef struct ResponseFormat
    type::String = "json_object"                               # "json_object" or "json_schema"
    json_schema::Union{JsonSchemaAPI,AbstractDict,Nothing} = nothing
end
ResponseFormat(json_schema)  # shorthand, sets type="json_schema"
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
weather_tool = GPTTool(func=GPTFunctionSignature(
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
    service::Type{<:ServiceEndpoint} = OPENAIServiceEndpoint
    model::String = "gpt-5.2"
    input::Union{String, Vector}                             # String or Vector{InputMessage}
    instructions::Union{String,Nothing} = nothing
    tools::Union{Vector,Nothing} = nothing                  # Vector of ResponseTool subtypes
    tool_choice::Union{String,Nothing} = nothing            # "auto", "none", "required"
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
    conversation::Union{Any,Nothing} = nothing
    context_management::Union{Vector,Nothing} = nothing
    stream_options::Union{AbstractDict,Nothing} = nothing
end
```

### Input Helpers

```julia
# Structured input messages
InputMessage(role="user", content="Hello")
InputMessage(role="user", content=[input_text("Describe:"), input_image("https://...")])

# Content part constructors
input_text(text::String)                                    # → Dict(:type=>"input_text", :text=>...)
input_image(url::String; detail=nothing)                    # → Dict(:type=>"input_image", ...) detail: "auto","low","high"
input_file(; url=nothing, id=nothing)                       # → Dict(:type=>"input_file", ...) provide url or file id
```

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
    search_context_size::String = "medium"                  # "low","medium","high"
    user_location::Union{AbstractDict,Nothing} = nothing
end

@kwdef struct FileSearchTool <: ResponseTool
    vector_store_ids::Vector{String}
    max_num_results::Union{Int,Nothing} = nothing
    ranking_options::Union{AbstractDict,Nothing} = nothing
    filters::Union{AbstractDict,Nothing} = nothing
end
```

```julia
@kwdef struct MCPTool <: ResponseTool
    server_label::String
    server_url::String
    require_approval::Union{String, AbstractDict, Nothing} = "never"
    allowed_tools::Union{Vector{String}, Nothing} = nothing
    headers::Union{AbstractDict, Nothing} = nothing
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
web_search(; context_size="medium", location=nothing)
file_search(store_ids; max_results=nothing, ranking=nothing, filters=nothing)
mcp_tool(label, url; require_approval="never", allowed_tools=nothing, headers=nothing)
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
end
```

**Convenience constructors**:

```julia
text_format(; kwargs...)                                     # generic TextConfig
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
respond(r::Respond; retries=0, callback=nothing) -> ResponseSuccess | ResponseFailure | ResponseCallError | Task

# Convenience — builds Respond internally
respond(input; kwargs...) -> same

# do-block streaming — auto-sets stream=true
respond(callback::Function, input; kwargs...) -> Task
```

- Streaming callback signature: `callback(chunk::Union{String, ResponseObject}, close::Ref{Bool})`
- Auto-retries on HTTP 429/500/503 with exponential backoff and jitter (up to 30 attempts). Respects `Retry-After` headers.
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
compact_response(; model="gpt-5.2", input, service=OPENAIServiceEndpoint) -> Dict | ...
count_input_tokens(; model="gpt-5.2", input, instructions=nothing, tools=nothing, service=OPENAIServiceEndpoint) -> Dict | ...
```

### ResponseObject

```julia
@kwdef struct ResponseObject
    id::String
    status::String
    model::String
    output::Vector{Any}
    usage::Union{Dict{String,Any},Nothing} = nothing
    error::Union{Any,Nothing} = nothing
    metadata::Union{Dict{String,Any},Nothing} = nothing
    raw::Dict{String,Any}
end
```

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
    service::Type{<:ServiceEndpoint} = OPENAIServiceEndpoint
    model::String = "gpt-image-1.5"
    prompt::String
    n::Union{Int,Nothing} = nothing                         # 1–10
    size::Union{String,Nothing} = nothing                   # "1024x1024","1536x1024","1024x1536","auto"
    quality::Union{String,Nothing} = nothing                # "low","medium","high","auto"
    background::Union{String,Nothing} = nothing             # "transparent","opaque","auto"
    output_format::Union{String,Nothing} = nothing          # "png","webp","jpeg"
    output_compression::Union{Int,Nothing} = nothing        # 0–100 (webp/jpeg only)
    user::Union{String,Nothing} = nothing
end
```

### generate_image

```julia
generate_image(ig::ImageGeneration; retries=0) -> ImageSuccess | ImageFailure | ImageCallError
generate_image(prompt::String; kwargs...)       -> same   # convenience
```

Auto-retries on 429/500/503 with exponential backoff and jitter (up to 30 attempts). Respects `Retry-After` headers.

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
    model::String             # "text-embedding-3-small" (1536 dims)
    input::Union{String,Vector{String}}
    embeddings::Union{Vector{Float64},Vector{Vector{Float64}}}
    user::Union{String,Nothing}
end

Embeddings(input::String)           # single input, pre-allocates 1536-dim vector
Embeddings(input::Vector{String})   # batch input, pre-allocates one vector per input
```

### embeddingrequest!

```julia
embeddingrequest!(emb::Embeddings; retries=0) -> (response_dict, emb) | nothing
```

Fills `emb.embeddings` in-place. Auto-retries on 429/500/503 with exponential backoff and jitter (up to 30 attempts). Respects `Retry-After` headers.

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
end
```

### Functions

```julia
token_usage(result::LLMSuccess)::Union{TokenUsage, Nothing}      # extract TokenUsage from a Chat result
token_usage(result::ResponseSuccess)::Union{TokenUsage, Nothing}  # extract from Responses API result

estimated_cost(result; model=nothing, pricing=DEFAULT_PRICING)     # per-call cost estimate (Float64)
cumulative_cost(chat::Chat)::Float64                               # running total for a Chat instance

DEFAULT_PRICING   # Dict{String, Tuple{Float64, Float64}} — model → (input_price, output_price) per token
```

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
fork(chat::Chat)::Chat          # deep-copy a Chat, resetting cumulative cost
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

## Result Type Hierarchy

All API call results inherit from `LLMRequestResponse`:

```
LLMRequestResponse (abstract)
├── LLMSuccess          — Chat Completions success (.message::Message, .self::Chat)
├── LLMFailure          — Chat Completions HTTP error (.response::String, .status::Int, .self::Chat)
├── LLMCallError        — Chat Completions exception (.error::String, .status, .self::Chat)
├── ResponseSuccess     — Responses API success (.response::ResponseObject)
├── ResponseFailure     — Responses API HTTP error (.response::String, .status::Int)
├── ResponseCallError   — Responses API exception (.error::String, .status)
├── ImageSuccess        — Image Gen success (.response::ImageResponse)
├── ImageFailure        — Image Gen HTTP error (.response::String, .status::Int)
└── ImageCallError      — Image Gen exception (.error::String, .status)
```

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
const GEMINI_CHAT_URL = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
```

## Exceptions

```julia
struct InvalidConversationError <: Exception
    reason::String
end
```

Thrown by `issendvalid` / internal validation when conversation structure is invalid (e.g., missing system message position, consecutive same-role messages).

---

## Complete Exports List

**Chat Completions**: `Chat`, `Message`, `RoleSystem`, `RoleUser`, `RoleAssistant`, `GPTTool`, `GPTToolCall`, `GPTFunctionSignature`, `GPTFunctionCallResult`, `InvalidConversationError`, `issendvalid`, `chatrequest!`, `update!`, `ResponseFormat`

**Responses API**: `Respond`, `InputMessage`, `ResponseTool`, `FunctionTool`, `WebSearchTool`, `FileSearchTool`, `MCPTool`, `ComputerUseTool`, `ImageGenerationTool`, `CodeInterpreterTool`, `TextConfig`, `TextFormatSpec`, `Reasoning`, `ResponseObject`, `ResponseSuccess`, `ResponseFailure`, `ResponseCallError`, `respond`, `get_response`, `delete_response`, `list_input_items`, `cancel_response`, `compact_response`, `count_input_tokens`, `output_text`, `function_calls`, `input_text`, `input_image`, `input_file`, `function_tool`, `web_search`, `file_search`, `mcp_tool`, `computer_use`, `image_generation_tool`, `code_interpreter`, `text_format`, `json_schema_format`, `json_object_format`

**Image Generation**: `ImageGeneration`, `ImageObject`, `ImageResponse`, `ImageSuccess`, `ImageFailure`, `ImageCallError`, `generate_image`, `image_data`, `save_image`

**Embeddings**: `Embeddings`, `embeddingrequest!`

**Cost Tracking**: `TokenUsage`, `token_usage`, `estimated_cost`, `cumulative_cost`, `DEFAULT_PRICING`

**Forking**: `fork`

**Result Types**: `LLMRequestResponse`, `LLMSuccess`, `LLMFailure`, `LLMCallError`
