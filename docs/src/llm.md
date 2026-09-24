# UniLM.jl — LLM Reference

> **Single-file reference for LLM code-generation systems.** This is a **Julia** package.
> Julia ≥ 1.13 · Deps: `HTTP.jl` (≥ 2.7.1), `JSON.jl`, `Base64`, `SHA`
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
| `AZURE_OPENAI_DEPLOY_NAME_<MODEL>` | `AZUREServiceEndpoint`            | Deployment for a model, read at call time: model id upper-cased, other characters → `_` (`AZURE_OPENAI_DEPLOY_NAME_GPT_5_2`) |
| `GEMINI_API_KEY`                   | `GEMINIServiceEndpoint`           | Google Gemini API key                           |
| `ANTHROPIC_API_KEY`                | `ANTHROPICServiceEndpoint`        | Anthropic (Claude) API key                      |
| `DEEPSEEK_API_KEY`                 | `DeepSeekEndpoint`                | DeepSeek API key                                |
| `MISTRAL_API_KEY`                  | `MistralEndpoint`                 | Mistral AI API key                              |
| `TYPESAFE_API_KEY`                 | `TYPESAFEServiceEndpoint`         | TypeSafe System One (Jev) API key               |
| `TYPESAFE_BASE_URL`                | `TYPESAFEServiceEndpoint`         | Optional API root override (default `https://api.typesafe.ai`) |
| `TYPESAFE_DEFAULT_MODEL`           | `TYPESAFEServiceEndpoint`         | Model used when a call names none (default `jev-latest`) |

## The APIs

UniLM.jl wraps these surfaces:

1. **Chat Completions** (`Chat` + `chatrequest!`) — stateful, message-based conversations with tool calling, streaming, structured output. Supports OpenAI, Azure, Gemini (native), Anthropic (native), DeepSeek, Ollama, Mistral, and any OpenAI-compatible provider.
2. **Responses API** (`Respond` + `respond`) — newer, more flexible API with built-in tools (web search, file search), multi-turn chaining via `previous_response_id`, reasoning support for O-series models, structured output. OpenAI Responses by default; the unified `respond` verb also targets Google's Gemini Interactions via `service=GEMINIServiceEndpoint` (see the Agentic Workflows guide).
3. **Image Generation** (`ImageGeneration` + `generate_image`) — text-to-image with `gpt-image-2`. OpenAI only.
4. **Embeddings** (`Embeddings` + `embeddingrequest!`) — vector embeddings. Multi-provider via `service` parameter.
5. **FIM Completion** (`FIMCompletion` + `fim_complete`) — code infilling. DeepSeek (beta), Mistral, Ollama.
6. **System One** (`ask`, plus `nl_dispatch` and `@branch`) — TypeSafe's Jev: enumerated questions about a piece of `state`, answered with a typed value and a calibrated probability distribution instead of generated text. `service=TYPESAFEServiceEndpoint` only; not a chat backend.

**Which API to use:**
- **Chat Completions** — best for multi-turn conversations; broadest provider support. Use for chat, tool calling, or streaming across any supported backend.
- **Responses API** — simpler for single-shot or chained requests; built-in web search, file search, MCP, computer use tools. OpenAI Responses plus Google's Gemini Interactions via the unified `respond` verb (see the Agentic Workflows guide).
- **FIM Completion** — code infilling between prefix and suffix. DeepSeek (beta), Mistral and Ollama (vLLM's completions endpoint does not accept `suffix`).
- **System One** — classify, rank, screen or route unstructured input when the outcomes are a fixed set you can enumerate and you want a probability, not prose. `ask` returns typed results; `nl_dispatch` turns the answer into ordinary Julia multiple dispatch and `@branch` into an inline switch, both throwing rather than guessing a branch.

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
struct TYPESAFEServiceEndpoint <: ServiceEndpoint end     # TypeSafe System One (Jev) — TYPESAFE_API_KEY
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
    model::String = ""
    messages::Vector{Message} = Message[]
    history::Bool = true
    tools::Union{Vector{Tool},Nothing} = nothing
    tool_choice::Union{String,GPTToolChoice,Nothing} = nothing
    parallel_tool_calls::Union{Bool,Nothing} = false
    temperature::Union{Float64,Nothing} = nothing       # 0.0–2.0, mutually exclusive with top_p
    top_p::Union{Float64,Nothing} = nothing              # 0.0–1.0, mutually exclusive with temperature
    n::Union{Int64,Nothing} = nothing                    # must be 1: a result carries one choice
    stream::Union{Bool,Nothing} = nothing
    stop::Union{Vector{String},String,Nothing} = nothing # max 4 sequences
    max_tokens::Union{Int64,Nothing} = nothing
    max_completion_tokens::Union{Int64,Nothing} = nothing # preferred over max_tokens; reasoning models reject max_tokens
    presence_penalty::Union{Float64,Nothing} = nothing   # -2.0 to 2.0
    response_format::Union{ResponseFormat,Nothing} = nothing
    frequency_penalty::Union{Float64,Nothing} = nothing  # -2.0 to 2.0
    logit_bias::Union{AbstractDict{String,<:Real},Nothing} = nothing  # values in [-100, 100]
    user::Union{String,Nothing} = nothing
    seed::Union{Int64,Nothing} = nothing
    reasoning_effort::Union{String,Nothing} = nothing    # none|minimal|low|medium|high|xhigh|max (per-model subset)
    stream_options::Union{AbstractDict,Nothing} = nothing # unset: OpenAI/DeepSeek streams send {"include_usage": true}
    verbosity::Union{String,Nothing} = nothing           # low|medium|high
    store::Union{Bool,Nothing} = nothing
    metadata::Union{AbstractDict,Nothing} = nothing
    service_tier::Union{String,Nothing} = nothing        # auto|default|flex|scale|priority|fast
    logprobs::Union{Bool,Nothing} = nothing
    top_logprobs::Union{Int64,Nothing} = nothing
    prediction::Union{AbstractDict,Nothing} = nothing
    modalities::Union{Vector{String},Nothing} = nothing
    audio::Union{AbstractDict,Nothing} = nothing
    web_search_options::Union{AbstractDict,Nothing} = nothing
    prompt_cache_key::Union{String,Nothing} = nothing
    safety_identifier::Union{String,Nothing} = nothing   # replaces deprecated `user`
    _cumulative_cost::Ref{Float64} = Ref(0.0)            # internal
    prompt_cache_options::Union{PromptCacheOptions,Nothing} = nothing  # gpt-5.6 and later
    moderation::Union{ModerationConfig,Nothing} = nothing
end
```

`_cumulative_cost` is internal bookkeeping for [`cumulative_cost`](@ref) — read it
through that accessor, never directly.

- **Model defaults**: the declared default is the sentinel `""`; the constructor resolves it, so `Chat().model` reads back `"gpt-5.6-sol"`. Per provider: `"gpt-5.6-sol"` for OpenAI, `"gpt-5.2"` for Azure, `"gemini-3.8-flash"` for Gemini (native and OpenAI-compat), `"claude-opus-5-5"` for native Anthropic (whose `max_tokens` defaults to 16000), `"deepseek-flash"` for DeepSeek. For `GenericOpenAIEndpoint` / `OllamaEndpoint` / `MistralEndpoint` and user-defined endpoints without a `UniLM.default_model` method there is no default — an unset model throws `ArgumentError` at construction.
- `history=true`: responses are automatically appended to `messages`.
- `temperature` and `top_p` are mutually exclusive (constructor throws `ArgumentError`).
- `parallel_tool_calls` is auto-set to `nothing` when `tools` is `nothing`.
- **Parameter validation**: the constructor validates at construction time — `temperature` ∈ [0.0, 2.0], `top_p` ∈ [0.0, 1.0], `n` must be `1` (a result carries a single choice), `max_tokens` / `max_completion_tokens` ≥ 1, `presence_penalty` ∈ [-2.0, 2.0], `frequency_penalty` ∈ [-2.0, 2.0], `top_logprobs` ∈ [0, 20], every `logit_bias` value ∈ [-100, 100], `reasoning_effort` ∈ {`"none"`, `"minimal"`, `"low"`, `"medium"`, `"high"`, `"xhigh"`, `"max"`}. Violations throw `ArgumentError`. An empty `tools` vector is stored as `nothing`; a `Vector{<:CallableTool}` is accepted and stored as its `Tool`s.
- **Provider/model limits** (GPT-5.6/GPT-6 tool and sampling rules, the Anthropic and native-Gemini field mappings) are checked by the provider's encoder: `chatrequest!` throws their `ArgumentError` before any network I/O.
- `prompt_cache_options` is supported for gpt-5.6 and later (gpt-5.4-mini answered HTTP 400 "prompt_cache_options is not supported on this model" on 2026-09-22); Chat Completions takes only its `mode` and `ttl` — `chatrequest!` on a native OpenAI `Chat` with `prewarm` or `comparison_response_id` set throws `ArgumentError` before any request. `Message` content is a plain string, so Chat messages cannot carry cache breakpoints, and the Chat Completions reference says an explicit-mode request without breakpoints "does not use prompt caching": `mode="explicit"` on a `Chat` turns prompt caching off. `moderation` takes a [`ModerationConfig`](@ref).

### Message

```julia
@kwdef struct Message
    role::String                                          # "system", "user", "assistant" or "tool" (else ArgumentError)
    content::Union{String,Nothing} = nothing
    name::Union{String,Nothing} = nothing
    finish_reason::Union{String,Nothing} = nothing        # response-only: "stop", "tool_calls", "length", "content_filter", …; nothing if none reported
    refusal_message::Union{String,Nothing} = nothing
    tool_calls::Union{Nothing,Vector{ToolCall}} = nothing
    tool_call_id::Union{String,Nothing} = nothing         # required when role == "tool"
    provider_content::Union{Nothing,ProviderContent} = nothing
end
```

**Validation**: `role` must be one of the four roles; at least one of `content`, `tool_calls`, or `refusal_message` must be non-`nothing`; `tool_call_id` is required when `role == "tool"`.

On the wire a message never sends `finish_reason` (response-only) or `provider_content`, and a refusal travels as `refusal` (with `content: null`). A tool-call turn reads `finish_reason == "tool_calls"` when the provider finished it with `"stop"` or reported none, and keeps any other reason (`"length"`, `"content_filter"`, …).

`provider_content` carries provider-native blocks captured for verbatim round-trip — Anthropic thinking blocks (`:anthropic`), Gemini parts with thought signatures (`:gemini`), DeepSeek `reasoning_content` (`:deepseek`, echoed on requests that carry tools); it never serializes on the OpenAI wire.

**Convenience constructors**:

```julia
Message(Val(:system), "You are a helpful assistant")
Message(Val(:user), "Hello!")
```

**Role constants**: `RoleSystem = "system"`, `RoleUser = "user"`, `RoleAssistant = "assistant"`.

### chatrequest!

```julia
# Mutating form — sends chat.messages, appends response when history=true
chatrequest!(chat::Chat; config=nothing, callback=nothing, on_tool_call=nothing, cancel=nothing)
    -> LLMSuccess | LLMFailure | LLMCallError | Task

# Keyword-argument convenience form — builds a Chat internally
chatrequest!(; messages, config=nothing, cancel=nothing, chat_kwargs...) -> same
chatrequest!(; systemprompt, userprompt, config=nothing, cancel=nothing, chat_kwargs...) -> same
```

- Non-streaming: returns `LLMSuccess`, `LLMFailure`, or `LLMCallError`.
- Streaming (`stream=true`): returns a `Task` whose `fetch` yields the same typed results. Pass a `callback(chunk::Union{String,Message}, close::Ref{Bool})` — text deltas arrive as `String`s (verbatim, in order), then the assembled `Message` at end-of-stream.
- Streaming tool calls: pass `on_tool_call(tc::ToolCall)` to be notified at most once per completed streamed tool call, as calls finish (a call whose arguments do not parse is skipped with a warning; see the [Streaming guide](@ref streaming_guide)).
- Local validation throws before any network I/O, streaming or not: `ArgumentError` when the service declares capabilities without `:chat` or the provider's encoder rejects the request; `InvalidConversationError` when `history=true` and the conversation ends with an assistant message (the reply could not be appended).
- The keyword form takes EITHER `messages` (copied, never mutated) OR both `systemprompt` and `userprompt` (a `String` or a `Message` each); anything else throws `ArgumentError`. Every other keyword is a `Chat` field; `history` controls whether the reply is appended, not what is sent.
- Retries transient statuses (408/429/500/502/503/504/529) with exponential backoff and jitter under the resolved [`RequestConfig`](@ref) — `max_attempts` (default 3) and `total_deadline` bound the attempts. `Retry-After` is a floor under the jitter (at most half the budget left after the floor is spread above it), so a rate-limited fan-out does not retry in lockstep; a retry whose wait would exceed the remaining `total_deadline` is not attempted — the call returns the last real response rather than sleeping past the deadline. Timeouts surface as `LLMCallError` with `status=nothing` and the `UniLMTimeout` in `.cause`.
- `cancel::Union{Nothing,CancelToken}` (default: the ambient [`with_cancel`](@ref) token): a cancel before connecting, during the header wait, mid-stream or during a backoff ends the call with `LLMCallError(status=nothing, cause=UniLMCancelled(:token, …))` — never retried, nothing appended, no terminal callback; a pre-cancelled token sends nothing. A TCP connect / TLS handshake in progress finishes (or reaches `connect_timeout`) first.
- Streaming stop: `close[] = true` (in the callback or from any task) ends the stream at once with `LLMCallError(status=nothing, cause=UniLMCancelled(:callback, …))`, unless the terminal event was already recorded (the turn then stands). An exception thrown by `callback`/`on_tool_call` ends the call with that exception in `.cause` (never retried, nothing appended). Time inside callbacks is not counted by `stream_idle_timeout`.
- Streaming retry boundary: transient failures (including the in-band `overloaded_error`, the documented 529 equivalent) are retried inside the task only until the first `callback`/`on_tool_call` invocation; afterwards failures surface typed. A user `InterruptException` propagates — `fetch` throws a `TaskFailedException` instead of returning a result value.
- Streams use HTTP/1.1 (one connection per stream); OpenAI and DeepSeek streams request `include_usage` when `stream_options` is unset, so streamed turns are costed.

### Conversation Management

```julia
push!(chat, message)       # append a Message (the FIRST message must be a system Message)
pop!(chat)                 # remove last message
update!(chat, message)     # append if history=true
issendvalid(chat) -> Bool  # check conversation rules (see below)
length(chat)               # number of messages
isempty(chat)              # true if no messages
chat[i]                    # index into messages
chat[i] = msg              # replace a message — NOT validated (nor are direct edits to chat.messages)
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
@kwdef mutable struct FunctionSignature
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

task = chatrequest!(chat; callback = function (chunk, close_ref)
    if chunk isa String
        print(chunk)            # partial text delta
    elseif chunk isa Message
        println("\n[Done]")     # final assembled message
        # close_ref[] = true    # to stop early → LLMCallError, cause = UniLMCancelled(:callback, …)
    end
end)

result = fetch(task)  # LLMSuccess when complete, else a typed failure
```

---

## Responses API

### Respond

```julia
@kwdef struct Respond
    service::ServiceEndpointSpec = OPENAIServiceEndpoint
    model::String = ""                                # declared as "" and resolved by the constructor
    input::Union{String, Vector}                             # String, Vector{InputMessage}, or Vector{Dict}
    instructions::Union{String,Nothing} = nothing
    tools::Union{Vector,Nothing} = nothing                  # untyped: ResponseTool, CallableTool, Dict; a Chat Tool becomes a FunctionTool
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
    service_tier::Union{String,Nothing} = nothing           # "auto","default","flex","scale","priority","fast","ultrafast"
    top_logprobs::Union{Int64,Nothing} = nothing            # 0–20
    prompt::Union{AbstractDict,Nothing} = nothing           # v1/prompts shuts down on November 30, 2026
    prompt_cache_key::Union{String,Nothing} = nothing
    prompt_cache_retention::Union{String,Nothing} = nothing  # "in_memory","24h" (older models)
    safety_identifier::Union{String,Nothing} = nothing
    conversation::Union{Any,Nothing} = nothing              # String or Dict (not with previous_response_id); the declared type collapses to Any
    context_management::Union{Vector,Nothing} = nothing
    stream_options::Union{AbstractDict,Nothing} = nothing
    prompt_cache_options::Union{PromptCacheOptions,Nothing} = nothing
    moderation::Union{ModerationConfig,Nothing} = nothing
end
```

!!! warning "Gemini Interactions rejects unmappable fields"
    With `service=GEMINIServiceEndpoint`, a set `Respond` field the Interactions
    wire has no equivalent for makes `respond` throw `ArgumentError` before any
    request rather than being silently dropped, so a request never goes out quietly ignoring what you
    asked for. That wire maps `model`, `input`, `instructions`, `tools`,
    `tool_choice`, `temperature`, `top_p`, `max_output_tokens`, `stream`, `text`
    (its format becomes the top-level `response_format`; `verbosity` is
    rejected), `store`, `previous_response_id`, `background`, and `reasoning`
    (effort and automatic summaries only). Every other field — `moderation`,
    `prompt_cache_options`, and the rest of the OpenAI-only options — is
    rejected; leave it unset, or send the request to an OpenAI Responses service.
    A `FunctionTool` with `async`, `allowed_callers`, `defer_loading`, or
    `output_schema` set is rejected the same way.

### Input Helpers

```julia
# Structured input messages
InputMessage(role="user", content="Hello")
InputMessage(role="user", content=[input_text("Describe:"), input_image("https://...")])

# Content part constructors
input_text(text::String; cache_breakpoint=false)                      # → Dict(:type=>"input_text", :text=>...); cache_breakpoint=true adds prompt_cache_breakpoint={"mode":"explicit"}
input_image(url=nothing; detail=nothing, file_id=nothing)             # → Dict(:type=>"input_image", ...); pass url OR file_id, detail: "auto","low","high"
input_file(; url=nothing, id=nothing, file_data=nothing, filename=nothing)  # → Dict(:type=>"input_file", ...); pass one of url / id / file_data (base64)

# Input item: change reasoning effort mid-conversation (GPT-6) without rewriting the cached prefix
configuration_update(; effort)   # → Dict(:type=>"configuration_update", :reasoning=>Dict(:effort=>...)); effort: "none","minimal","low","medium","high","xhigh","max"
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
    async::Union{Bool,Nothing} = nothing                    # GPT-6 async tool calling
    allowed_callers::Union{Vector{String},Nothing} = nothing  # "direct" and/or "programmatic"; anything else throws
    defer_loading::Union{Bool,Nothing} = nothing            # loaded via tool search
    output_schema::Union{AbstractDict,Nothing} = nothing    # JSON Schema of the string output
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

# action, moderation, partial_images, and input_fidelity are validated at construction
# (ArgumentError); quality, background, and output_format pass through unchecked
# (the accepted quality values depend on the model).
@kwdef struct ImageGenerationTool <: ResponseTool
    background::Union{String, Nothing} = nothing
    output_format::Union{String, Nothing} = nothing
    output_compression::Union{Int, Nothing} = nothing
    quality::Union{String, Nothing} = nothing               # "low","medium","high","auto"; 2.5 models add "xhigh","max"
    size::Union{String, Nothing} = nothing
    model::Union{String, Nothing} = nothing                 # e.g. "gpt-image-2.5-flare"
    action::Union{String, Nothing} = nothing                # "generate","edit","auto"
    moderation::Union{String, Nothing} = nothing            # "auto","low"
    partial_images::Union{Int, Nothing} = nothing           # 0–3
    input_fidelity::Union{String, Nothing} = nothing        # "high","low"
    input_image_mask::Union{AbstractDict, Nothing} = nothing  # "file_id" or "image_url"
end

@kwdef struct CodeInterpreterTool <: ResponseTool
    container::Union{AbstractDict, Nothing} = nothing
    file_ids::Union{Vector{String}, Nothing} = nothing
end
```

**Convenience constructors**:

```julia
function_tool(name, description=nothing; parameters=nothing, strict=nothing, async=nothing,
              allowed_callers=nothing, defer_loading=nothing, output_schema=nothing)
function_tool(d::AbstractDict)           # from a dict with "name" and optionally "description", "parameters", "strict",
                                         # "async", "allowed_callers", "defer_loading", "output_schema"
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

### Reasoning controls

```julia
@kwdef struct Reasoning
    effort::Union{String,Nothing} = nothing                 # "none","minimal","low","medium","high","xhigh","max" (per-model subset)
    generate_summary::Union{String,Nothing} = nothing       # deprecated alias, serialized as summary
    summary::Union{String,Nothing} = nothing                # "auto","concise","detailed"
    context::Union{String,Nothing} = nothing                # "auto","current_turn","all_turns"
    mode::Union{String,Nothing} = nothing                   # "standard","pro"
end
```

```julia
Respond(input="Hard math problem", model="gpt-5.4-mini", reasoning=Reasoning(effort="high"))
```

GPT-5.6 and later use typed prompt-cache options on `Respond` and `Chat`:

```julia
@kwdef struct PromptCacheOptions
    mode::Union{String,Nothing} = nothing                    # "implicit" or "explicit"
    ttl::Union{String,Nothing} = nothing                     # "30m"
    prewarm::Union{Bool,Nothing} = nothing                   # Responses only: prepare the cache, no output
    comparison_response_id::Union{String,Nothing} = nothing  # Responses only: request cache diagnostics
end
Respond(input="Hello", model="gpt-5.6-luna",
        reasoning=Reasoning(effort="low", context="current_turn", mode="standard"),
        prompt_cache_options=PromptCacheOptions(mode="explicit", ttl="30m"))
```

Use `prompt_cache_options` in place of legacy `prompt_cache_retention` for these
models. Explicit mode requires cache breakpoints in input content to create cache
writes — mark them with `input_text(text; cache_breakpoint=true)`, a Responses input
part; `Chat` messages cannot carry one, so explicit mode on a `Chat` means no prompt
caching. Chat Completions takes only `mode` and `ttl` (a native OpenAI `Chat` with `prewarm` or
`comparison_response_id` set throws `ArgumentError`). Diagnostics requested with
`comparison_response_id` come back in `r.response.raw["prompt_cache_diagnostics"]`.
Unset fields are omitted.
[OpenAI prompt caching](https://developers.openai.com/api/docs/guides/prompt-caching).

### Moderation

```julia
@kwdef struct ModerationConfig
    model::String                                 # required, e.g. "omni-moderation-latest"
    input_mode::Union{String,Nothing} = nothing   # "score" or "block"
    output_mode::Union{String,Nothing} = nothing  # "score" or "block"
end
Respond(input="Hello", moderation=ModerationConfig(model="omni-moderation-latest"))
Chat(moderation=ModerationConfig(model="omni-moderation-latest", input_mode="block"))
```

The `moderation` field on `Respond` and `Chat` runs a moderation model over the
request input and the generated output. `model` is required, as in both the
Responses and the Chat Completions API references: `ModerationConfig()` without it
throws `UndefKeywordError`, and a mode other than `"score"`/`"block"` throws
`ArgumentError`. It serializes as
`{"model": …, "policy": {"input": {"mode": …}, "output": {"mode": …}}}` with unset
policy parts omitted. A `respond` result carries the outcome in
`r.response.raw["moderation"]` (`"input"` and `"output"` moderation results);
`LLMSuccess` keeps no raw body, so Chat results do not expose it.

### respond

```julia
# Struct form
respond(r::Respond; config=nothing, callback=nothing, cancel=nothing)
    -> ResponseSuccess | ResponseFailure | ResponseCallError | Task

# Convenience — builds Respond internally
respond(input; kwargs...) -> same

# do-block streaming — auto-sets stream=true
respond(callback::Function, input; kwargs...) -> Task
```

- Streaming callback signature: `callback(chunk::Union{String, ResponseObject}, close::Ref{Bool})`; `close[] = true` (or a `cancel!` on the call's token) stops the stream with `ResponseCallError(status=nothing, cause=UniLMCancelled(...))` unless the terminal event was already recorded. A callback exception ends the call with it in `.cause`.
- Local validation throws `ArgumentError` before any network I/O: a service that declares capabilities with neither `:responses` nor `:agentic`, or an encoder rejection (a field the wire or model cannot express).
- Retries retryable statuses (408/429/500/502/503/504/529) up to `config.max_attempts` (default 3) with full-jitter backoff bounded by `config.total_deadline`; `Retry-After` is a floor under the jitter, and a retry whose wait would exceed the remaining deadline is not attempted — it returns the last real response instead of sleeping past it. Every attempt is time-bounded — a silent peer fails with a typed timeout inside `ResponseCallError` (`status = nothing`, `cause::UniLMTimeout`), never a hang. Every exception that ends a call (including a cancellation, `cause::UniLMCancelled`) is in `.cause`.
- **Parameter validation**: `temperature` ∈ [0.0, 2.0], `top_p` ∈ [0.0, 1.0], `max_output_tokens` ≥ 1, `top_logprobs` ∈ [0, 20]; `conversation` cannot be combined with `previous_response_id`, `background=true` requires `store` not `false`, and `prompt_cache_retention` cannot be combined with `prompt_cache_options`; `Reasoning(effort=…)` must be one of `none`, `minimal`, `low`, `medium`, `high`, `xhigh`, `max`. Violations throw `ArgumentError`.

### Response Accessors

```julia
output_text(result::ResponseSuccess)::String                # concatenated text output ("\n" between parts)
output_text(result::ResponseFailure)                        # throws LLMResultError
output_text(result::ResponseCallError)                      # throws LLMResultError

function_calls(result::ResponseSuccess)::Vector{Dict{String,Any}}   # empty on a failure result
# Each dict has: "id", "call_id", "name", "arguments" (JSON string), "status"
```

### Response Management Functions

```julia
get_response(id::String; service=OPENAIServiceEndpoint, config=nothing)    -> ResponseSuccess | ResponseFailure | ResponseCallError
delete_response(id::String; service=OPENAIServiceEndpoint, config=nothing) -> Dict | ResponseFailure | ResponseCallError
list_input_items(id::String; limit=20, order="desc", after=nothing, service=OPENAIServiceEndpoint, config=nothing) -> Dict | ...
cancel_response(id::String; service=OPENAIServiceEndpoint, config=nothing) -> ResponseSuccess | ...
compact_response(; model="gpt-5.6-sol", input, service=OPENAIServiceEndpoint, config=nothing) -> Dict | ...
count_input_tokens(; model="gpt-5.6-sol", input, instructions=nothing, tools=nothing, service=OPENAIServiceEndpoint, config=nothing) -> Dict | ...
```

Each makes a single bounded attempt (no retries) and observes the ambient
[`with_cancel`](@ref) token; every call error carries its exception in `.cause`.

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
own `error` and `metadata` are preserved on `.response`. Only `"failed"` flips:
`incomplete`, `cancelled`, `expired`, `in_progress`, `queued` and
`requires_action` are legitimate terminals of the capped, background and
tool-action flows and stay `ResponseSuccess`, inspectable via
[`response_status`](@ref) and [`incomplete_details`](@ref). An `incomplete`
generation — one cut short by a token or output cap — therefore arrives as a
`ResponseSuccess` carrying its partial output, `.response.status ==
"incomplete"`, and the wire's `incomplete_details` verbatim. A terminal
`response.incomplete` event decodes exactly as the non-streamed body does, so
`issuccess` reflects what happened rather than how the call was made.

### Responses API Examples

```julia
using UniLM, JSON

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

# Reasoning
result = respond("Prove that √2 is irrational", model="gpt-5.4-mini",
    reasoning=Reasoning(effort="high", summary="concise"))

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
    size::Union{String,Nothing} = nothing                   # "1024x1024","1536x1024","1024x1536","auto"; gpt-image-2+: any WIDTHxHEIGHT, multiples of 16, up to 3840x2160
    quality::Union{String,Nothing} = nothing                # "low","medium","high","auto"; gpt-image-2.5-flare/-sunburst add "xhigh","max"
    background::Union{String,Nothing} = nothing             # "transparent","opaque","auto"
    output_format::Union{String,Nothing} = nothing          # "png","webp","jpeg"
    output_compression::Union{Int,Nothing} = nothing        # 0–100 (webp/jpeg only)
    user::Union{String,Nothing} = nothing
    moderation::Union{String,Nothing} = nothing             # "auto","low"
end
```

`input_fidelity` is an edit-only parameter: it lives on `ImageEdit`, and
`ImageGeneration(...; input_fidelity=...)` is a `MethodError`.

Unlike `Chat` and `Respond`, `ImageGeneration` does **not** resolve its model at
construction: the field stays `""` and resolves at serialization time, to
`"gpt-image-2"` for OpenAI. So `ImageGeneration(prompt="…").model` reads back as
the empty string — pass `model=` explicitly if you need to read it, or inspect
`JSON.json(ig)` to see what will go on the wire. A service with no default image
model throws `ArgumentError` at serialization.

### generate_image

```julia
generate_image(ig::ImageGeneration; config=nothing, cancel=nothing) -> ImageSuccess | ImageFailure | ImageCallError
generate_image(prompt::String; config=nothing, cancel=nothing, kwargs...) -> same   # convenience
edit_image(e::ImageEdit; config=nothing, cancel=nothing)                 -> same
edit_image(image, prompt; mask=nothing, service=OPENAIServiceEndpoint, config=nothing, cancel=nothing, kwargs...)
```

Retries transient statuses (408/429/500/502/503/504/529) inside the request budget (`RequestConfig.max_attempts`, default 3; `Retry-After` is a floor under the jitter) — a retry whose wait would exceed the remaining `total_deadline` is not attempted, so the call returns the last real response rather than sleeping past it. Bound per call with `config=RequestConfig(...)`; a cancelled call returns `ImageCallError` with `cause::UniLMCancelled`.

### Response Types

```julia
@kwdef struct ImageObject
    b64_json::Union{String,Nothing} = nothing
    revised_prompt::Union{String,Nothing} = nothing
    url::Union{String,Nothing} = nothing          # set when the image was delivered by URL
end

@kwdef struct ImageResponse
    created::Int64
    data::Vector{ImageObject}
    usage::Union{Dict{String,Any},Nothing} = nothing
    raw::Dict{String,Any}
end
```

### Accessors

```julia
image_data(result::ImageSuccess)::Vector{String}       # per image: its base64 data, else its URL
image_data(result::ImageFailure)                        # throws LLMResultError
image_data(result::ImageCallError)                      # throws LLMResultError
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
    encoding_format::Union{String,Nothing}  # "float" or nothing; anything else throws ArgumentError
end

Embeddings(input::String; service=OPENAIServiceEndpoint, model="",
           dimensions=nothing, encoding_format=nothing, user=nothing)
Embeddings(input::Vector{String}; service=OPENAIServiceEndpoint, model="",
           dimensions=nothing, encoding_format=nothing, user=nothing)
```

Model defaults: `"text-embedding-3-small"` for OpenAI, `"gemini-embedding-001"` for Gemini (OpenAI-compat shim). For generic/DeepSeek endpoints, model must be specified explicitly or the constructor throws `ArgumentError` naming the service. An empty `Vector{String}` input also throws.

The `embeddings` buffer is pre-allocated to `something(dimensions, 1536)` zeros per input, filled in place, and resized to the length the model returns (a 3072-dimension model needs no adjustment). Because a pre-zeroed slot would be indistinguishable from a real vector, a response that does not cover every input exactly once — a missing or duplicated row index — is reported as an `EmbeddingCallError` rather than left as zeros. The result aliases the request (`result.embeddings === emb`), so use one `Embeddings` per concurrent call.

### embeddingrequest!

```julia
embeddingrequest!(emb::Embeddings; config=nothing, cancel=nothing) -> EmbeddingSuccess | EmbeddingFailure | EmbeddingCallError
```

Returns an `EmbeddingSuccess`/`EmbeddingFailure`/`EmbeddingCallError` (a `<: LLMRequestResponse`). Fills `emb.embeddings` in-place; `embedding_vectors(result)` returns the vectors (and throws `LLMResultError` on a failure). Retries transient statuses (408/429/500/502/503/504/529) with backoff and jitter under the resolved [`RequestConfig`](@ref) (`max_attempts`, `total_deadline`; `Retry-After` is a floor under the jitter), but a retry whose wait would exceed the remaining deadline is not attempted — it returns the last real response rather than sleeping past it. Timeouts surface as `EmbeddingCallError` with `status=nothing` and the `UniLMTimeout` in `.cause`; a cancelled call carries `UniLMCancelled` there. A service that declares capabilities without `:embeddings` throws `ArgumentError` before any request.

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

DEFAULT_PRICING   # lock-guarded AbstractDict{String, PriceRow} (not a Dict)
                  # where PriceRow = @NamedTuple{input::Float64, cached_input::Float64, output::Float64}
```

- **`token_usage` returns a `TokenUsage`, never `nothing`**, for every result of a
  token-billed API — chat, Responses, embeddings, image generation, FIM and System
  One. Failures of those APIs report all-zero usage.
- **Both functions throw `ArgumentError` for result types outside the token-billed
  APIs** — audio, batch, container, conversation, file, fine-tuning, moderation,
  upload, vector-store, realtime. Those calls carry no token usage at all,
  and a `0.0` would be indistinguishable from a genuinely free call.
- The model is inferred from the result (`self.model` for Chat, `response.model` for
  Responses, FIM and System One, `embeddings.model` for Embeddings). An image result
  reports its usage but `estimated_cost` returns `0.0` for it unless `model=` names a
  row in `pricing` (image models have no default rows).
- **`DEFAULT_PRICING` is safe to share across tasks**: add or replace a row with
  `DEFAULT_PRICING[model] = row` while requests run; `get`, `haskey`, `delete!`,
  `pop!`, `empty!`, `keys` and `length` work as on a `Dict`, iteration walks a
  snapshot, and `copy` / `merge` / `filter` return a plain `Dict`. `pricing=` accepts
  any `AbstractDict{String, PriceRow}`.
- **`DEFAULT_PRICING` values are USD *per token*, not per 1M tokens** — provider
  list prices divided by `1_000_000` (e.g. `"gpt-5.6-sol"` is
  `(input = 4.0e-6, cached_input = 4.0e-7, output = 2.0e-5)`). A `pricing=`
  dict you supply must use the same per-token convention, or your estimate is
  off by a factor of a million.
- Lookup: the exact id, then the id without a dated snapshot suffix (OpenAI
  `-YYYY-MM-DD`, Anthropic `-YYYYMMDD`), and a versioned `jev-X.Y.Z` id without its
  own row is priced at the `jev-latest` row. Other unpriced models return `0.0` and
  log one warning per model id — pass `pricing=` (or add a row) to price custom
  models. The formula bills
  `min(cached_tokens, prompt_tokens)` at `cached_input`, the remaining prompt
  tokens at `input`, and `completion_tokens` at `output` (reasoning tokens are
  already counted within completion tokens).

Current OpenAI, Gemini, and TypeSafe rows were verified September 22, 2026 (Anthropic
and DeepSeek rows September 24, 2026; the DeepSeek rows are its peak rates, and
off-peak hours bill half). These are estimates for standard short-context text
requests, excluding cache-write premiums, long-context surcharges, nonstandard service
tiers, multimodal rates, and hosted-tool fees. Anthropic `prompt_tokens` include cache
reads and writes (`cached_tokens` is the read share), and DeepSeek cache hits
(`prompt_cache_hit_tokens`) are billed at the cached rate. A streamed Chat turn accrues
cost when its stream reports usage: OpenAI and DeepSeek streams request it
automatically, native Anthropic and Gemini report it, and other OpenAI-compatible
servers need `stream_options=Dict("include_usage" => true)`.
Gemini 3.8/3.7/3.6 Flash introductory rates expire December 31, 2026; refresh pricing
before estimating later calls.

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
fork(chat::Chat)::Chat          # deep-copy every field but `service` (shared); cumulative cost copied by value (fresh Ref)
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
    callable::Function   # (name::String, args::Dict{String,Any}) -> String (any other value is sent JSON-encoded)
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
    response::LLMRequestResponse        # the last response received (on max_turns: the last tool-call turn)
    tool_calls::Vector{ToolCallOutcome}
    turns_used::Int
    completed::Bool                     # true only for a final text answer
    llm_error::Union{String,Nothing}    # why it stopped, when not completed
end
```

### tool_loop! (Chat Completions)

```julia
tool_loop!(chat, dispatcher; max_turns=10, config=nothing, callback=nothing, on_tool_call=nothing,
           cancel=nothing, tool_concurrency=1) -> ToolLoopResult
tool_loop!(chat; tools::Vector{<:CallableTool}, kwargs...) -> ToolLoopResult   # kwargs as above
```

`callback` and `on_tool_call` are the streaming hooks from `chatrequest!`, applied
to every turn of the loop. Rules:

- `chat.history` must be `true` (else `ArgumentError` before any request); `max_turns < 1`
  and `tool_concurrency < 1` throw `ArgumentError`.
- Calls run only on a turn whose `finish_reason == "tool_calls"`. A turn with calls that
  finished otherwise (`"length"`, `"content_filter"`, …) runs none: `completed=false`,
  `llm_error` names the reason, and the unanswered assistant turn is removed from `chat`.
- A dispatcher's `String` result is sent as is, any other value JSON-encoded; a throwing
  dispatcher sends `"Error: <message>"` and records a failed `ToolCallOutcome`; an
  `InterruptException` propagates.
- On `max_turns` exhaustion `response` is the last real response, `completed=false`,
  `llm_error = "max turns (N) exhausted"`.
- `cancel` (default: the ambient token) scopes every turn and dispatch; a cancelled loop
  returns `completed=false` with an `LLMCallError` whose `cause` is `UniLMCancelled`
  (a turn cancelled between dispatches is removed from `chat`).
- `tool_concurrency = n > 1` runs up to `n` of a turn's calls at once on spawned tasks
  (thread-safe dispatcher required); results are sent in call order.

### tool_loop (Responses API)

```julia
tool_loop(r::Respond, dispatcher; max_turns=10, config=nothing, cancel=nothing, tool_concurrency=1) -> ToolLoopResult
tool_loop(r::Respond; max_turns=10, config=nothing, cancel=nothing, tool_concurrency=1) -> ToolLoopResult   # callables from r.tools
tool_loop(input, dispatcher; max_turns=10, config=nothing, cancel=nothing, tool_concurrency=1, respond_kwargs...)
tool_loop(input::String; tools, max_turns=10, config=nothing, cancel=nothing, tool_concurrency=1, respond_kwargs...)
```

Turns chain through `previous_response_id`, or through `conversation` alone when the
`Respond` sets it. Function calls run only on a `completed` or `requires_action` turn;
another status stops the loop (`completed=false`, `llm_error` naming the status and the
`incomplete_details` reason). Call `arguments` that are not a JSON object are answered
with `"Error: invalid arguments: …"` and the loop continues. A turn requesting a
client-side action the loop cannot run (`custom_tool_call`, `apply_patch_call`,
`local_shell_call`, `computer_call`, a `shell_call` without a `shell_call_output` in the
same output, `mcp_approval_request`) stops it with `completed=false`, naming the type,
and runs none of the turn's calls. Result encoding,
`max_turns`, `cancel` and `tool_concurrency` behave as in `tool_loop!` (a cancelled loop's
`response` is a `ResponseCallError`).

---

## MCP Client

Native MCP client (JSON-RPC 2.0 over stdio or Streamable HTTP). Client and server
negotiate revisions 2025-11-25 (preferred), 2025-06-18 and 2025-03-26; the stateless
2026-07-28 revision is not supported yet, so a 2026-07-28-only peer cannot connect.

### Types

```julia
MCPSession            # live connection — manages transport + cached tools/resources/prompts
MCPToolInfo           # tool definition from tools/list
MCPToolResult         # typed tools/call result: content, structured, is_error, parts
MCPResourceInfo       # resource definition from resources/list
MCPPromptInfo         # prompt definition from prompts/list
MCPServerCapabilities # capabilities from initialize
MCPTransport          # abstract (subtypes: StdioTransport, HTTPTransport)
MCPError <: Exception # JSON-RPC error (code, message, data — any JSON value)
MCPCrashError <: Exception # stdio server process exited / pipe broke (exitcode, termsignal, cause)
MCPSessionClosedError <: Exception # call on a closed / not-connected session (cause :disconnected|:timeout|:crash, msg)
```

### Lifecycle

```julia
mcp_connect(command::Cmd; stderr=nothing, client_name="UniLM.jl", client_version=<package version>,
            protocol_version="2025-11-25", config=nothing, auto_respawn=false) -> MCPSession  # stdio subprocess
mcp_connect(url::String; headers=Pair{String,String}[], kwargs...) -> MCPSession            # Streamable HTTP
mcp_connect(transport::MCPTransport; kwargs...) -> MCPSession      # a custom transport must bound its own IO
mcp_connect(f::Function, args...; kwargs...)                       # do-block, auto-disconnect
mcp_disconnect!(session)
```

`stderr` is where a stdio server's stderr goes — an `IO` (e.g. `devnull`) or a file path
(appended to); `nothing` inherits this process's. A stdio server runs in its own process
group: disconnecting or a fatal timeout tears it down (stdin EOF, SIGTERM, then SIGKILL
of the group), the group is killed when its leader exits, and servers still running at
exit are torn down by an exit hook.

### Discovery

```julia
list_tools!(session; timeout=nothing) -> Vector{MCPToolInfo}
list_resources!(session) -> Vector{MCPResourceInfo}
list_prompts!(session) -> Vector{MCPPromptInfo}
```

When the client reads a `notifications/tools/list_changed`, `session.tools_stale` is set to `true` to flag the cached tool list as out of date; call `list_tools!(session)` to refresh it (which clears the flag). Server frames are read only during an exchange: a stdio notification sent between calls is seen on the next call, and over HTTP (no standing listener) only notifications inside a response body are seen.

### Operations

```julia
call_tool(session, name, arguments; timeout=nothing) -> MCPToolResult
read_resource(session, uri) -> String
get_prompt(session, name, arguments) -> Vector{Dict}
ping(session)
```

`call_tool` and `get_prompt` accept any `AbstractDict` for `arguments`, so the
natural `Dict("path" => "/tmp/x")` form works without conversion.

Timeouts surface as the exported `MCPTimeoutError` (`phase` `:connect`, `:queue` or
`:request`). A stdio request timeout closes the session — killing the server is the
only way to release a read blocked on it; the error surfaces at about the bound and a
reply that raced the teardown is discarded. An HTTP request timeout does not close the
session, and the timed-out request gets a best-effort `notifications/cancelled`. A
stdio server that exits or whose pipe breaks surfaces the exported `MCPCrashError`
(the session closes with cause `:crash`). With `auto_respawn=true` the next call on a
session closed by a timeout or a crash respawns the server; otherwise, and after
`mcp_disconnect!`, every call throws `MCPSessionClosedError`.

A `CancelToken` (ambient `with_cancel`) reaches MCP calls over HTTP: a cancel throws
`UniLMCancelled` at once, sends a best-effort `notifications/cancelled`, and leaves the
session open. A stdio exchange does not observe the token (bounded by
`mcp_request_timeout`). Teardown requests (the disconnect `DELETE`, a cancellation
notice) are sent even inside a cancelled scope.

### Session Concurrency

An `MCPSession` runs one call at a time: each call's liveness check, id allocation
and request/response exchange holds the session, and waiting callers are served in
arrival order (FIFO).

- A call's per-call bound (`timeout`, default `mcp_request_timeout`) also bounds its
  **wait** for the session: a call that cannot acquire it in time throws
  `MCPTimeoutError(:queue, …)` without touching it. Once held, the exchange gets the
  full bound, measured from acquisition, so a waiter never cuts it short.
- `mcp_disconnect!` waits its turn too, so a disconnect racing a call in flight
  **waits** for that exchange to finish instead of tearing the transport down
  under its reader.
- Interleaved server → client frames are handled in place: notifications are
  skipped, server `ping` requests answered, other server requests answered `-32601`,
  and a stdio line that is not JSON skipped with a warning.

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

Bridged tool names are provider-safe: any character outside `[A-Za-z0-9_-]` becomes `_`
and names are cut to 128 characters (the callable still calls the MCP name); two tools
mapping to one name raise `ArgumentError`.

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
register_prompt!(server, name, handler; description=nothing, arguments=Dict{String,Any}[])
```

The inferred-schema `register_tool!(server, name, description, handler)` binds each
`tools/call` `arguments` object to the handler's positional parameters **by name**, as
`@mcp_tool` does: a parameter whose type admits `nothing` is optional, every other one
required, and a value must have its parameter's JSON type (a `Symbol` binds from a
string; typed `Vector`/`Dict` parameters convert element-wise). A violation is answered
with an `isError: true` tool result naming the argument. A handler taking one `Dict` or
varargs throws `ArgumentError` — pass an explicit schema for those. Registration is
synchronized: it may run while `serve` is dispatching. `@mcp_tool` registers the tool
without a description.

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
listed in `allowed_origins`. A request other than `initialize` whose
`MCP-Protocol-Version` header names an unsupported revision gets 400; `initialize`
answers the requested revision when supported, otherwise 2025-11-25.

**Concurrency:** over HTTP, `tools/call`, `resources/read` and `prompts/get` handlers
run concurrently, one task per request on the default thread pool — handlers must be
thread-safe; protocol requests (`initialize`, lists, `ping`) are answered inline. Over
stdio handlers run one at a time, and the process `stdout` points at `stderr` while
serving on it, so a printing handler cannot corrupt the frame stream.

```julia
handle = serve(server; transport=:http, port=8080, block=false,
               allowed_origins=["https://app.example.com"])
close(handle)
```

### Error and Limit Contracts

- **A throwing tool handler is a tool result, not a protocol error.** The client
  receives `isError: true` with the text `Error: <showerror text>`, which is exactly
  what lets a model see and correct its own mistake. Write handlers accordingly:
  raise with a message you are willing to show the model *and* the client.
- **Resource, prompt and dispatch-layer errors are generic.** An exception from a
  resource or prompt handler, or any unhandled error below the handler, answers
  JSON-RPC `-32603` with the message `"Internal error"` — the exception and
  backtrace go to the server's own logs, not to the peer, because an exception
  string can carry file paths and argument values a remote client has no business
  reading. A single bad frame never takes the transport down. An
  `InterruptException` is never converted: it propagates.
- **Frames and request bodies are capped at 16 MiB.** stdio answers an oversized
  frame with `-32600`, HTTP an oversized body with `413 Payload Too Large`, rather
  than parsing it, since parsing an attacker-sized payload allocates a multiple of
  it — an out-of-memory kill rather than a protocol error.
- The server tolerates spec-legal parameter shapes: `params: null` and positional
  `params` do not crash it, and a JSON-RPC response sent to it is accepted without
  an answer (HTTP `202`). Non-object `arguments`, a missing tool `name` and an unknown
  tool answer `-32602`.

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
Supported by DeepSeek (beta), Mistral (routed to `/v1/fim/completions`), and Ollama;
vLLM's completions endpoint rejects `suffix`.

```julia
@kwdef struct FIMCompletion
    service::ServiceEndpointSpec
    model::String = ""    # sentinel: resolves at serialization to "deepseek-flash" for DeepSeek
    prompt::String
    suffix::Union{String,Nothing} = nothing
    max_tokens::Union{Int,Nothing} = 128
    temperature::Union{Float64,Nothing} = nothing
    top_p::Union{Float64,Nothing} = nothing
    stream::Union{Bool,Nothing} = nothing          # stream=true throws ArgumentError (no streaming path)
    stop::Union{Vector{String},String,Nothing} = nothing
    echo::Union{Bool,Nothing} = nothing
    logprobs::Union{Int,Nothing} = nothing
    frequency_penalty::Union{Float64,Nothing} = nothing
    presence_penalty::Union{Float64,Nothing} = nothing
end

@kwdef struct FIMChoice
    text::String
    index::Int = 0
    finish_reason::Union{String,Nothing} = nothing
end

@kwdef struct FIMResponse
    choices::Vector{FIMChoice}
    usage::Union{TokenUsage,Nothing} = nothing
    model::String = ""
    raw::Dict{String,Any} = Dict{String,Any}()
end

@kwdef struct FIMSuccess <: LLMRequestResponse
    response::FIMResponse
end

@kwdef struct FIMFailure <: LLMRequestResponse
    response::String
    status::Int
    request_id::Union{String,Nothing} = nothing
end

@kwdef struct FIMCallError <: LLMRequestResponse
    error::String
    status::Union{Int,Nothing} = nothing
    request_id::Union{String,Nothing} = nothing
    cause::Union{Nothing,Exception} = nothing
end

fim_complete(fim::FIMCompletion; config=nothing, cancel=nothing) -> LLMRequestResponse
fim_complete(prompt; suffix=nothing, config=nothing, cancel=nothing, kwargs...) -> LLMRequestResponse  # convenience
fim_text(result) -> String  # generated text; throws LLMResultError on a FIMFailure/FIMCallError
```

Unlike `Chat`, `FIMCompletion` does not resolve its model at construction — the
field stays `""` until serialization, where it becomes `"deepseek-flash"` for
`DeepSeekEndpoint`. Endpoints without a default FIM model must set `model=`
explicitly, or `fim_complete` throws `ArgumentError`. Local validation (capability —
strict, so an endpoint that declares nothing throws `MethodError` — model resolution,
routing) throws before any request; every later failure, including a 200 body that is
not a completions response, is a `FIMCallError` whose `cause` holds the exception.
Retries follow the shared policy; `FIMSuccess` results report usage and cost.

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
prefix_complete(chat::Chat; config=nothing, cancel=nothing) -> LLMRequestResponse
# Last message must be role=assistant with the prefix text
```

The result's `message` is the continuation the API returns; with `chat.history` the
conversation keeps the whole assistant turn, prefix followed by continuation. A
cancelled call returns `LLMCallError` (`cause::UniLMCancelled`) and leaves `chat`
untouched.

### Prefix Example

```julia
chat = Chat(service=DeepSeekEndpoint())   # deepseek-flash
push!(chat, Message(Val(:system), "You are a coding assistant."))
push!(chat, Message(Val(:user), "Write quicksort in Python"))
push!(chat, Message(role=RoleAssistant, content="```python\n"))
result = prefix_complete(chat)
```

---

## TypeSafe System One API (Jev)

A System One model answers enumerated questions about a piece of `state` with a
probability distribution over the outcomes you named — no generated text, no
parsing. `TYPESAFE_API_KEY` is required; the default model is `jev-latest` and
pinning `"jev-1.13.0"` keeps tuned confidence thresholds meaningful.

```julia
# What a piece of guidance may be on the wire: string | object | array
const SystemOneEntry = Union{AbstractString, AbstractDict, AbstractVector, NamedTuple}
# What `state` may be. No `nothing`: the server reports a null state as missing.
const SystemOneState = Union{AbstractString, AbstractDict, AbstractVector, Tuple, NamedTuple}

abstract type SystemOneQuestion end

struct ChoiceQuestion <: SystemOneQuestion          # 1..255 options, insertion order kept
    instructions::Union{Nothing,SystemOneEntry}
    criteria::JSON.Object{String,Any}               # option name => description (or nothing)
end

struct ScoreQuestion <: SystemOneQuestion           # 1..10 levels, criteria[1] is level 0
    instructions::Union{Nothing,SystemOneEntry}
    criteria::Vector{Any}
end

struct NoulCriteria                                 # wire keys "true" / "false"
    yes::Union{Nothing,SystemOneEntry}
    no::Union{Nothing,SystemOneEntry}
end

struct NoulQuestion <: SystemOneQuestion            # needs instructions or one criterion
    instructions::Union{Nothing,SystemOneEntry}
    criteria::Union{Nothing,NoulCriteria}
end

struct SystemOneRequest
    state::SystemOneState
    questions::Vector{Pair{String,SystemOneQuestion}}   # ordered, unique non-empty names
    model::String
end

abstract type SystemOneAnswer end

struct NoulAnswer <: SystemOneAnswer                # no confidence: the value is the belief
    noul::Float64
    raw::Dict{String,Any}
end

struct ChoiceAnswer <: SystemOneAnswer
    choice::String
    confidence::Float64
    probabilities::Dict{String,Float64}             # keyed by option name, unordered
    raw::Dict{String,Any}
end

struct ScoreAnswer <: SystemOneAnswer
    score::Float64                                  # expectation; may fall between levels
    confidence::Float64
    legend::Dict{Int,Any}                           # 0-based level number => description
    probabilities::Dict{Int,Float64}                # same keys as legend
    raw::Dict{String,Any}
end

struct UnknownAnswer <: SystemOneAnswer             # forward-compatible, never dropped
    type::String
    raw::Dict{String,Any}
end

struct SystemOneResponse
    model::String                                   # the versioned id that answered
    answers::Dict{String,SystemOneAnswer}           # keyed by question name
    usage::TokenUsage                               # prompt = input, completion = output
    request_id::Union{Nothing,String}               # x-typesafe-request-id header
    raw::Dict{String,Any}
end

struct SystemOneSuccess <: LLMRequestResponse
    response::SystemOneResponse
end

@kwdef struct SystemOneFailure <: LLMRequestResponse   # HTTP non-2xx
    response::String
    status::Int
    request_id::Union{Nothing,String} = nothing
    error_type::Union{Nothing,String} = nothing     # "authentication_error", "api_usage_error"
    message::String = ""
    retry_after::Union{Nothing,Float64} = nothing   # seconds; retry-after-ms, else Retry-After
end

@kwdef struct SystemOneCallError <: LLMRequestResponse # no usable response
    error::String
    status::Union{Int,Nothing} = nothing
    request_id::Union{Nothing,String} = nothing
    cause::Union{Nothing,Exception} = nothing       # e.g. a UniLMTimeout or UniLMCancelled
end

struct SystemOneError <: Exception                  # thrown by accessors on a non-success
    result::Union{SystemOneFailure,SystemOneCallError}
end

struct TypeSafeModelCard
    name::String
    description::String
    release_date::String                            # opaque string (live returns RFC 3339)
    raw::Dict{String,Any}
end

struct TypeSafeModelsSuccess <: LLMRequestResponse
    models::Vector{TypeSafeModelCard}
    raw::Dict{String,Any}
end
```

### Constructors, verbs and accessors

```julia
choice(instructions, criteria) -> ChoiceQuestion    # criteria: NamedTuple | Dict |
choice(criteria) -> ChoiceQuestion                  #   Vector{Pair} | Vector{String}
score(instructions, levels::AbstractVector) -> ScoreQuestion
score(levels::AbstractVector) -> ScoreQuestion
noul(instructions; yes=nothing, no=nothing) -> NoulQuestion
noul(; yes=nothing, no=nothing) -> NoulQuestion

SystemOneRequest(state, questions; model=UniLM.default_typesafe_model())

ask(request::SystemOneRequest; service=TYPESAFEServiceEndpoint, config=nothing, cancel=nothing)
ask(state, questions...; model=UniLM.default_typesafe_model(), service=TYPESAFEServiceEndpoint,
    config=nothing, cancel=nothing)
    # -> SystemOneSuccess | SystemOneFailure | SystemOneCallError

list_models(; service=TYPESAFEServiceEndpoint, config=nothing, cancel=nothing)
    # -> TypeSafeModelsSuccess | SystemOneFailure | SystemOneCallError

answers(result) -> Dict{String,SystemOneAnswer}     # throws SystemOneError on a non-success
answer(result, name)                                # name::Union{AbstractString,Symbol}
result[name]; haskey(result, name); keys(result)    # same, on Success or Response
```

`questions` accepts `name => question` pairs (vector, tuple, `NamedTuple`, or
`AbstractDict`), or bare questions auto-named `"q1"`, `"q2"`, … in order. Mixing
the two, repeating a name, or passing none is an `ArgumentError`. Both verbs ride
the shared retry seam, so `RequestConfig.max_attempts` applies to 408, 429, 500,
502, 503, 504 and 529, and the final `SystemOneFailure` carries `retry_after` —
the wait the service asked for in seconds, or `nothing` when it sent no hint.

### Example

```julia
r = ask("Help! My payouts have been failing for 3 days.",
        "department" => choice("Which team should handle this ticket?",
            (billing="Payments, invoicing, payouts, refunds",
             technical="Bugs, outages, integrations",
             sales="Pricing, upgrades, new accounts")),
        "urgency" => score("How urgent is this ticket?",
            ["Can wait", "Needs attention this week", "Needs attention today"]),
        "is_frustrated" => noul("Is the customer frustrated?"; yes="Frustrated", no="Neutral"))
issuccess(r) && println(r["department"].choice, " ", r["urgency"].score, " ", r["is_frustrated"].noul)
```

### Natural-language control flow

A Choice answer is already a branch decision, so two constructs compile ordinary
control flow into ONE System One request. `@branch` is a natural-language switch
that evaluates only the selected body; `nl_dispatch` sends one Choice per
`Meaning` slot and then hands the resolved meanings to Julia's own dispatch, so
the remaining arguments still dispatch on their types.

```julia
struct Meaning{S} end                     # S::Symbol IS the description
Meaning(text::AbstractString)             # the instance; nl"..." is the TYPE

struct LowConfidenceError <: Exception    # raised when confidence < min_confidence
    question::String
    answer::ChoiceAnswer                  # the full distribution, for diagnostics
    min_confidence::Float64
end

@branch state [key = value ...] begin
    "option name"                  => expression   # the name is what the model reads
    ("option name", "description") => expression   # the only way to add a description
    _                              => expression   # requires min_confidence
end
# keys: model, min_confidence, instructions, service, config, cancel
# -> the selected body's value; LowConfidenceError or SystemOneError otherwise

nl_dispatch(f, args...; model=nothing, service=TYPESAFEServiceEndpoint, config=nothing,
            cancel=nothing, min_confidence=0.0, fallback=nothing, instructions=nothing, state=nothing)
    # -> f(resolved meanings spliced into their positions, args...)
meanings(f) -> Dict{Int,Vector{String}}   # slot position => options, in send order
```

Methods without a concrete `Meaning` argument (including wildcard `::Meaning`
ones) are ordinary methods and are not part of the natural-language interface.
Every natural-language method of `f` must agree on arity and on which positions
are slots. Without `state`, the state is a `Dict{String,Any}` keyed by the
ordinary argument names of the first such method. A resolved combination no
method covers raises Julia's own `MethodError`.

```julia
ticket = "My package arrived crushed and the screen is cracked. I want my money back."

action = @branch ticket min_confidence=0.6 begin
    "the customer wants a refund"           => :refund
    "the customer reports a bug in the app" => :bug
    _                                       => :escalate
end                                         # => :refund

route(::nl"the customer wants a refund", t)           = (:refund, t)
route(::nl"the customer reports a bug in the app", t) = (:bug, t)
nl_dispatch(route, ticket)                            # => (:refund, ticket)
route(nl"the customer wants a refund"(), ticket)      # direct call, no request
```

---

## Provider Capabilities

Each endpoint declares supported features. Request functions validate before
dispatch, so an unsupported feature is an `ArgumentError` naming the endpoint and what
it does support, not an HTTP 404.

```julia
provider_capabilities(service) -> Set{Symbol}
has_capability(service, cap::Symbol) -> Bool
```

### Capabilities by Provider

| Provider | Capabilities |
|---|---|
| OpenAI | `:chat`, `:responses`, `:agentic`, `:embeddings`, `:images`, `:image_edits`, `:tools`, `:streaming`, `:json_output`, `:files`, `:vector_stores`, `:conversations`, `:moderation`, `:audio`, `:batch`, `:fine_tuning`, `:containers`, `:uploads`, `:realtime` |
| Azure | `:chat`, `:tools`, `:streaming`, `:json_output` |
| Gemini (native) | `:chat`, `:tools`, `:json_output`, `:streaming`, `:agentic` |
| Gemini (OpenAI-compat) | `:chat`, `:embeddings`, `:tools`, `:streaming`, `:json_output` |
| Anthropic (native) | `:chat`, `:tools`, `:json_output`, `:streaming` |
| TypeSafe (System One) | `:system_one`, `:models` |
| DeepSeek | `:chat`, `:tools`, `:streaming`, `:fim`, `:prefix_completion`, `:json_output` |
| Generic | `:chat`, `:embeddings`, `:fim`, `:tools`, `:streaming`, `:json_output`, `:responses` |

Validation comes in two strengths, and the difference matters if you write your
own endpoint:

- **Platform and lifecycle verbs** (files, vector stores, conversations,
  moderations, audio, batch, fine-tuning, containers, uploads, realtime, FIM,
  prefix completion, System One) validate **strictly**: an endpoint with no
  declaration at all is not dispatched — the call throws `MethodError` (no
  `provider_capabilities` method) — because those surfaces are provider-specific
  and an undeclared backend would just 404.
- **The four primary verbs** — `chatrequest!` (`:chat`), `embeddingrequest!`
  (`:embeddings`), `respond` (`:responses` **or** `:agentic`), and
  `generate_image` (`:images`) — plus `edit_image` (`:image_edits`) validate only
  endpoints that **declare** their capabilities. A custom endpoint that defines no `provider_capabilities` method
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
- **Validation**: every `Float64` field rejects `NaN`, values `≤ 0` and finite values above `1e9` s with `ArgumentError` (`Inf` = disabled); `max_attempts ≥ 1`.
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
`generate_image`, `edit_image`, `ask`, `list_models`, and the `tool_loop` family
(streams retry only before the first callback fires). Platform and lifecycle verbs
— batch, container, conversation, file (`upload_file` included: a create is never
retried), fine-tuning, moderation, upload, vector-store, audio, realtime, and the
Responses lifecycle operations — make a **single bounded attempt**, and
`max_attempts` has no effect on them. `poll_batch` / `poll_file_batch` poll through
retryable statuses and per-attempt timeouts until their own wall-clock `timeout`
(`interval` and `timeout` must be > 0); a poll that times out returns the family's
call error with `cause = UniLMTimeout(:deadline, …)` and the last object seen in
`last_observed`. Both take `cancel`.

### Cancellation

```julia
tok = CancelToken()                 # level-triggered: once cancelled, stays cancelled
cancel!(tok)                        # from any task; idempotent; runs every registered hook
iscancelled(tok)::Bool              # iscancelled(nothing) == false
with_cancel(f, tok)                 # tok is the ambient token inside f (and tasks spawned there)
```

HTTP verbs take `cancel::Union{Nothing,CancelToken}=nothing` (`chatrequest!`,
`respond`, `embeddingrequest!`, `generate_image`, `edit_image`, `fim_complete`,
`prefix_complete`, `ask`, `list_models`, `poll_batch`, `poll_file_batch`, the tool
loops, `nl_dispatch`, `@branch`); `nothing` resolves the ambient token, and every other
HTTP verb observes the ambient token. A cancelled call returns its call-error result
with `status = nothing` and `cause = UniLMCancelled(source, elapsed)` (`source` is
`:token`, or `:callback` for a stream stopped with `close[] = true`), is never retried,
commits nothing; a pre-cancelled token sends nothing. Limits: a TCP connect / TLS
handshake in progress is not interrupted (it completes or hits `connect_timeout`), a
running callback finishes first, and the Realtime WebSocket and MCP stdio exchanges do
not observe the token. See the Concurrency, Tasks and Cancellation guide.

### Sharp edges

- **Streams, pre-first-byte,** are additionally bounded by `stream_idle_timeout`
  (HTTP.jl caps the response-header wait with the read-idle timer), so the effective
  bound is `min(total_deadline, request_timeout, stream_idle_timeout)` and a breach
  reports phase `:stream_idle` even though no byte arrived.
- **The idle bound measures wire idleness only.** Time a stream spends inside the
  user's `callback` / `on_tool_call` is not counted, so a slow consumer never
  idle-kills a healthy stream — and no bound covers a callback that never returns.
- **Timeouts land on time.** A non-streaming attempt's bound fires within a few ms of
  its limit, and a successful call has no detection latency.
- **A completed turn is never discarded or re-sent.** Once a stream records its
  terminal state, teardown noise — idle breach, transport reset, truncated read —
  finalizes it as a success rather than failing or retrying it, on every provider
  and both stream drivers. Re-POSTing would bill a second generation.
- **Realtime**: `realtime_connect` (OpenAI only: any other `service` throws
  `ArgumentError` before any I/O) bounds the open phase with `connect_timeout`, surfaces
  an open that failed for another reason as that error, and closes an upgrade that
  completes after the timeout without running the handler; `realtime_receive` bounds
  the read with `stream_idle_timeout` and surfaces a breach within
  `[limit, limit + ~5 s]` (the WebSocket close waits for the peer's acknowledgement).
  An open session's lifetime is deliberately unbounded. `mint_realtime_secret` returns
  `RealtimeCallError` for a 200 without a non-empty secret.

### Concurrency

- **One `Chat` per in-flight call.** `messages` and the cumulative-cost `Ref` are
  unsynchronized by design; use `fork(chat)` / `fork(chat, n)` to fan out. An
  `Embeddings` is filled in place (and aliased by its result): one per concurrent
  call. `Respond`, `ImageGeneration`, `FIMCompletion` and `SystemOneRequest` are not
  mutated by their verbs and may be shared.
- **An `MCPSession` runs one call at a time**, FIFO; a call's `timeout` bounds its
  wait for the session too (`MCPTimeoutError(:queue)`), and `mcp_disconnect!` waits
  for the in-flight exchange. One session per concurrent worker. `serve(:http)` runs
  handlers concurrently (thread-safe handlers required); stdio runs them one at a time.
- **Streams use one HTTP/1.1 connection each**; non-streaming calls may multiplex over
  HTTP/2. HTTP.jl sets no per-host connection cap by default. `Retry-After` is a floor
  under jitter, so a rate-limited batch does not retry in lockstep.
- **Thread 1** (the main task, and the libuv event loop) must not be blocked: a task
  that computes there without yielding delays every timer. Keep callbacks short.
  `tool_concurrency = n` runs a turn's tool calls on spawned tasks.

---

## Result Type Hierarchy

All 65 API call result types inherit from `LLMRequestResponse`, in 16 per-API
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
├─ Realtime           RealtimeSecretSuccess · RealtimeFailure · RealtimeCallError
└─ System One         SystemOneSuccess · TypeSafeModelsSuccess · SystemOneFailure · SystemOneCallError
```

Every `*Success` wraps a parsed response object, every `*Failure` carries the HTTP
status and body (every family but Embeddings and Realtime also the `request_id`), and
every `*CallError` carries the rendered `error` and, in
`cause`, the exception behind it when there is one (a `UniLMTimeout` for a timeout, a
`UniLMCancelled` for a cancellation). Fields of the five primary families:

```julia
LLMSuccess        (.message::Message, .self::Chat, .usage::Union{TokenUsage,Nothing}, .sse_dropped::Int)
LLMFailure        (.response::String, .status::Int, .self::Chat, .request_id::Union{String,Nothing}, .sse_dropped::Int)
LLMCallError      (.error::String, .status::Union{Int,Nothing}, .self::Chat, .request_id::Union{String,Nothing}, .cause::Union{Nothing,Exception})
ResponseSuccess   (.response::ResponseObject, .sse_dropped::Int)
ResponseFailure   (.response::String, .status::Int, .request_id::Union{String,Nothing}, .sse_dropped::Int)
ResponseCallError (.error::String, .status::Union{Int,Nothing}, .request_id::Union{String,Nothing}, .cause::Union{Nothing,Exception})
EmbeddingSuccess  (.embeddings::Embeddings, .usage::Union{TokenUsage,Nothing}, .raw::Dict{String,Any})
EmbeddingFailure  (.response::String, .status::Int)
EmbeddingCallError(.error::String, .status::Union{Int,Nothing}, .cause::Union{Nothing,Exception})
ImageSuccess      (.response::ImageResponse)
ImageFailure      (.response::String, .status::Int, .request_id::Union{String,Nothing})
ImageCallError    (.error::String, .status::Union{Int,Nothing}, .request_id::Union{String,Nothing}, .cause::Union{Nothing,Exception})
FIMSuccess        (.response::FIMResponse)
FIMFailure        (.response::String, .status::Int, .request_id::Union{String,Nothing})
FIMCallError      (.error::String, .status::Union{Int,Nothing}, .request_id::Union{String,Nothing}, .cause::Union{Nothing,Exception})
```

`request_id` carries the provider's request id for support escalation — the
`x-request-id` header, else `request-id` (Anthropic). It exists on the Chat /
Responses / FIM / image failure and call-error types above, on the System One results
and on every platform `*Failure` except `RealtimeFailure` (a `RealtimeCallError`
carries one); the embedding limbs do not carry it.

`sse_dropped` (default `0`) counts the undecodable SSE `data:` payloads dropped
while assembling **that** streamed turn. It is `0` for a non-streamed call and
for a clean stream; anything higher means the result was built from an incomplete
wire — see [Streaming](@ref streaming_guide).

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

Thrown by `push!` (a non-system message onto an empty `Chat`, a system message after the conversation started, or a same-role repeat other than `tool`), by `pop!` on an empty `Chat`, and by `chatrequest!` before any request when `history=true` and the conversation ends with an assistant message. `issendvalid` never throws — it returns a `Bool`.

```julia
struct LLMResultError <: Exception
    result::LLMRequestResponse
end
```

Thrown by the result accessors `text`, `output_text`, `embedding_vectors`, `image_data` and `fim_text` when the result is not a success. `showerror` prints only the status and a trimmed (≤200-char) excerpt of the response body or error message — never the conversation, endpoint, or API key.

```julia
struct UniLMTimeout <: Exception
    phase::Symbol        # :connect | :request | :stream_idle | :deadline
    elapsed::Float64     # seconds, monotonic
    limit::Float64
end

struct MCPTimeoutError <: Exception
    phase::Symbol        # :connect | :queue (never acquired its session) | :request
    elapsed::Float64
    limit::Float64
    msg::String          # names the applicable timeout override
end

struct UniLMCancelled <: Exception
    source::Symbol       # :token (cancel! on a CancelToken) | :callback (stream close[] = true)
    elapsed::Float64     # seconds since the operation started
end

struct MCPSessionClosedError <: Exception
    cause::Symbol        # :disconnected | :timeout | :crash
    msg::String          # recovery guidance
end

struct MCPCrashError <: Exception
    msg::String                          # recovery guidance (embeds exit info)
    exitcode::Union{Int,Nothing}         # exit code if the server exited and was reaped in time
    termsignal::Union{Int,Nothing}       # signal number if the server was signal-killed
    cause::Union{Exception,Nothing}      # underlying transport exception; nothing on clean read-EOF
end
```

Raised when a `RequestConfig` bound is exceeded. Value-returning surfaces
(chat, embeddings, responses, images, FIM, System One, platform verbs) deliver
`UniLMTimeout` — and a cancellation's `UniLMCancelled` — inside their call-error
results; the MCP surface throws `MCPTimeoutError` (and `UniLMCancelled` for a cancelled
HTTP call). A stdio MCP server that exits or whose pipe breaks throws `MCPCrashError`
instead (the session closes with cause `:crash`), and a call on a closed session throws
`MCPSessionClosedError`.

---

## Public Extension API (not exported)

Declared `public`, not exported — qualify them (`UniLM.get_url`). These are the
functions a new backend adds methods to and the stream-state types its handlers
mutate (see the Custom Backends guide): `get_url`, `auth_header`, `default_model`,
`encode_request`, `decode_response`, `handle_sse_event!`, `StreamState`,
`encode_agentic`, `decode_agentic`, `decode_agentic_stream`, `AgenticStreamState`.

---

## Complete Exports List

Every exported symbol, grouped by area:

**Chat Completions**: `Chat`, `Message`, `ProviderContent`, `RoleSystem`, `RoleUser`, `RoleAssistant`, `Tool`, `ToolCall`, `FunctionSignature`, `FunctionCallResult`, `ResponseFormat`, `InvalidConversationError`, `issendvalid`, `chatrequest!`, `update!`, `fork`
- *Legacy aliases* (pre-rename names, exported and non-breaking, retained until 1.0): `GPTTool` → `Tool`, `GPTToolCall` → `ToolCall`, `GPTFunctionSignature` → `FunctionSignature`, `GPTFunctionCallResult` → `FunctionCallResult`

**Responses API & Agentic**: `Respond`, `InputMessage`, `ResponseObject`, `ResponseSuccess`, `ResponseFailure`, `ResponseCallError`, `Reasoning`, `PromptCacheOptions`, `ModerationConfig`, `TextConfig`, `TextFormatSpec`, `respond`, `get_response`, `delete_response`, `cancel_response`, `list_input_items`, `compact_response`, `count_input_tokens`, `text_format`, `json_schema_format`, `json_object_format`
- *Input builders*: `input_text`, `input_image`, `input_file`, `configuration_update`
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

**MCP Client**: `MCPSession`, `MCPToolInfo`, `MCPToolResult`, `MCPResourceInfo`, `MCPPromptInfo`, `MCPServerCapabilities`, `MCPTransport`, `StdioTransport`, `HTTPTransport`, `MCPError`, `MCPCrashError`, `MCPSessionClosedError`, `mcp_connect`, `mcp_disconnect!`, `mcp_tools`, `mcp_tools_respond`, `list_tools!`, `list_resources!`, `list_prompts!`, `call_tool`, `read_resource`, `get_prompt`, `ping`

**MCP Server**: `MCPServer`, `MCPServerTool`, `MCPServerResource`, `MCPServerResourceTemplate`, `MCPServerPrompt`, `MCPServerPrimitive`, `register_tool!`, `register_resource!`, `register_resource_template!`, `register_prompt!`, `serve`, `@mcp_tool`, `@mcp_resource`, `@mcp_prompt`

**FIM / Completions**: `FIMCompletion`, `FIMChoice`, `FIMResponse`, `FIMSuccess`, `FIMFailure`, `FIMCallError`, `fim_complete`, `fim_text`, `prefix_complete`

**Files**: `FileUpload`, `FileObject`, `FileList`, `FileSuccess`, `FileListSuccess`, `FileContentSuccess`, `FileDeleteSuccess`, `FileFailure`, `FileCallError`, `upload_file`, `list_files`, `retrieve_file`, `delete_file`, `file_content`, `save_file_content`

**Vector Stores**: `VectorStoreObject`, `VectorStoreFileObject`, `VectorStoreFileBatch`, `VectorStoreList`, `VectorStoreSuccess`, `VectorStoreListSuccess`, `VectorStoreFileSuccess`, `VectorStoreBatchSuccess`, `VectorStoreDeleteSuccess`, `VectorStoreFailure`, `VectorStoreCallError`, `create_vector_store`, `retrieve_vector_store`, `list_vector_stores`, `delete_vector_store`, `add_vector_store_file`, `create_file_batch`, `retrieve_file_batch`, `poll_file_batch`, `vector_store_id`

**Conversations**: `ConversationObject`, `ConversationItem`, `ConversationItemList`, `ConversationSuccess`, `ConversationItemSuccess`, `ConversationItemListSuccess`, `ConversationDeleteSuccess`, `ConversationFailure`, `ConversationCallError`, `create_conversation`, `retrieve_conversation`, `update_conversation`, `delete_conversation`, `add_conversation_items`, `list_conversation_items`, `delete_conversation_item`, `conversation_id`

**Moderations**: `ModerationResponse`, `ModerationResult`, `ModerationSuccess`, `ModerationFailure`, `ModerationCallError`, `moderate`, `is_flagged`

**TypeSafe System One (Jev)**: `TYPESAFEServiceEndpoint`, `SystemOneQuestion`, `ChoiceQuestion`, `ScoreQuestion`, `NoulQuestion`, `NoulCriteria`, `choice`, `score`, `noul`, `SystemOneRequest`, `ask`, `SystemOneAnswer`, `ChoiceAnswer`, `ScoreAnswer`, `NoulAnswer`, `UnknownAnswer`, `SystemOneResponse`, `SystemOneSuccess`, `SystemOneFailure`, `SystemOneCallError`, `SystemOneError`, `answers`, `answer`, `TypeSafeModelCard`, `TypeSafeModelsSuccess`, `list_models`
- *Natural-language control flow*: `Meaning`, `@nl_str`, `@branch`, `nl_dispatch`, `meanings`, `LowConfidenceError`

**Audio**: `SpeechRequest`, `TranscriptionRequest`, `SpeechSuccess`, `TranscriptionSuccess`, `AudioFailure`, `AudioCallError`, `speak`, `save_audio`, `transcribe`, `translate`, `transcript_text`

**Batch**: `BatchObject`, `BatchList`, `BatchSuccess`, `BatchListSuccess`, `BatchFailure`, `BatchCallError`, `create_batch`, `retrieve_batch`, `cancel_batch`, `list_batches`, `poll_batch`

**Fine-tuning**: `FineTuningJob`, `FineTuningList`, `FineTuningSuccess`, `FineTuningListSuccess`, `FineTuningFailure`, `FineTuningCallError`, `create_fine_tuning_job`, `retrieve_fine_tuning_job`, `cancel_fine_tuning_job`, `list_fine_tuning_jobs`, `list_fine_tuning_events`, `list_fine_tuning_checkpoints`

**Containers**: `ContainerObject`, `ContainerList`, `ContainerSuccess`, `ContainerListSuccess`, `ContainerDeleteSuccess`, `ContainerFailure`, `ContainerCallError`, `create_container`, `retrieve_container`, `list_containers`, `delete_container`, `add_container_file`

**Uploads**: `UploadObject`, `UploadPartObject`, `UploadSuccess`, `UploadPartSuccess`, `UploadFailure`, `UploadCallError`, `create_upload`, `add_upload_part`, `complete_upload`, `cancel_upload`

**Webhooks**: `WebhookEvent`, `WEBHOOK_EVENTS`, `verify_webhook`, `parse_webhook`

**Realtime**: `RealtimeSession`, `RealtimeSecretSuccess`, `RealtimeFailure`, `RealtimeCallError`, `mint_realtime_secret`, `realtime_connect`, `realtime_send`, `realtime_receive`, `realtime_event`, `session_update`, `input_audio_append`, `response_create`

**Result Types (base)**: `LLMRequestResponse`, `LLMSuccess`, `LLMFailure`, `LLMCallError`, `LLMResultError`, `issuccess`, `isfailure`, `text`

**Request Config, Timeouts & Cancellation**: `RequestConfig`, `current_config`, `with_request_config`, `set_default_config!`, `UniLMTimeout`, `MCPTimeoutError`, `UniLMCancelled`, `CancelToken`, `cancel!`, `iscancelled`, `with_cancel`
