# ============================================================================
# FIM Completion & Chat Prefix Completion
# FIM uses /v1/completions with prompt + suffix (fill-in-the-middle); Mistral
# serves it at /v1/fim/completions.
# Prefix Completion uses /v1/chat/completions with an assistant prefix message.
# Both are beta features on DeepSeek; also supported by Ollama and vLLM.
# ============================================================================

# ─── FIM Types ──────────────────────────────────────────────────────────────

"""
    FIMCompletion(; service, model, prompt, suffix=nothing, max_tokens=128, ...)

A Fill-in-the-Middle completion request. The model generates text between `prompt`
(prefix) and `suffix`.

Supported by [`DeepSeekEndpoint`](@ref) (beta), Mistral, Ollama, vLLM. `stream=true`
throws an `ArgumentError`: [`fim_complete`](@ref) has no streaming path.

# Example
```julia
fim = FIMCompletion(service=DeepSeekEndpoint(), prompt="def fib(a):",
    suffix="    return fib(a-1) + fib(a-2)", max_tokens=128)
result = fim_complete(fim)
println(fim_text(result))
```
"""
@kwdef struct FIMCompletion
    service::ServiceEndpointSpec
    model::String = ""
    prompt::String
    suffix::Union{String,Nothing} = nothing
    max_tokens::Union{Int,Nothing} = 128
    temperature::Union{Float64,Nothing} = nothing
    top_p::Union{Float64,Nothing} = nothing
    stream::Union{Bool,Nothing} = nothing
    stop::Union{Vector{String},String,Nothing} = nothing
    echo::Union{Bool,Nothing} = nothing
    logprobs::Union{Int,Nothing} = nothing
    frequency_penalty::Union{Float64,Nothing} = nothing
    presence_penalty::Union{Float64,Nothing} = nothing
    function FIMCompletion(service, model, prompt, suffix, max_tokens, temperature, top_p, stream,
                           stop, echo, logprobs, frequency_penalty, presence_penalty)
        stream === true && throw(ArgumentError(
            "FIMCompletion does not support stream=true: fim_complete has no streaming path"))
        new(service, model, prompt, suffix, max_tokens, temperature, top_p, stream, stop, echo,
            logprobs, frequency_penalty, presence_penalty)
    end
end

function JSON.lower(fim::FIMCompletion)
    model = fim.model
    if isempty(model)
        dm = default_fim_model(fim.service)
        isnothing(dm) && throw(ArgumentError("model must be specified for FIM with $(_service_name(fim.service))"))
        model = dm
    end
    d = Dict{Symbol,Any}(:model => model, :prompt => fim.prompt)
    for f in (:suffix, :max_tokens, :temperature, :top_p, :stream, :stop,
              :echo, :logprobs, :frequency_penalty, :presence_penalty)
        v = getfield(fim, f)
        !isnothing(v) && (d[f] = v)
    end
    d
end

# ─── FIM Result Types ──────────────────────────────────────────────────────

"""
    FIMChoice

A single completion choice from a FIM response.

# Fields
- `text::String`: The generated text
- `index::Int`: Choice index (default 0)
- `finish_reason::Union{String,Nothing}`: Why generation stopped (e.g. `"stop"`, `"length"`)
"""
@kwdef struct FIMChoice
    text::String
    index::Int = 0
    finish_reason::Union{String,Nothing} = nothing
end

"""
    FIMResponse

Parsed FIM completion response containing choices, usage, and raw data.

# Fields
- `choices::Vector{FIMChoice}`: Generated completions
- `usage::Union{TokenUsage,Nothing}`: Token usage statistics
- `model::String`: Model that generated the response
- `raw::Dict{String,Any}`: Complete raw JSON response
"""
@kwdef struct FIMResponse
    choices::Vector{FIMChoice}
    usage::Union{TokenUsage,Nothing} = nothing
    model::String = ""
    raw::Dict{String,Any} = Dict{String,Any}()
end

"""Successful FIM completion result."""
@kwdef struct FIMSuccess <: LLMRequestResponse
    response::FIMResponse
end

"""HTTP-level failure from FIM completion."""
@kwdef struct FIMFailure <: LLMRequestResponse
    response::String
    status::Int
    request_id::Union{String, Nothing} = nothing
end

"""Exception-level error during FIM completion."""
@kwdef struct FIMCallError <: LLMRequestResponse
    error::String
    status::Union{Int,Nothing} = nothing
    request_id::Union{String, Nothing} = nothing
    # Underlying exception when the failure is a timeout/transport error rather
    # than an HTTP status (status stays `nothing` — no fabricated HTTP code).
    cause::Union{Nothing,Exception} = nothing
end

Base.show(io::IO, r::FIMCallError) =
    _show_call_error(io, "FIMCallError", r.error, r.status, r.request_id, r.cause)

# ─── FIM Accessors ─────────────────────────────────────────────────────────

"""
    fim_text(result) -> String

Extract the generated text from a FIM completion result. On a [`FIMFailure`](@ref) or
[`FIMCallError`](@ref) it throws an [`LLMResultError`](@ref), as [`text`](@ref) does on a
failed Chat result: a failed call has no text, and `""` would read as an empty
completion. Guard with [`issuccess`](@ref).
"""
fim_text(r::FIMResponse)::String = isempty(r.choices) ? "" : r.choices[1].text
fim_text(r::FIMSuccess)::String = fim_text(r.response)
fim_text(r::Union{FIMFailure,FIMCallError}) = throw(LLMResultError(r))
_llm_result_status(r::FIMFailure)   = r.status
_llm_result_status(r::FIMCallError) = r.status
_llm_result_body(r::FIMFailure)     = r.response
_llm_result_body(r::FIMCallError)   = r.error

# ─── FIM URL Routing ──────────────────────────────────────────────────────

get_url(s::DeepSeekEndpoint, ::FIMCompletion)::String = DEEPSEEK_BETA_BASE_URL * COMPLETIONS_PATH
# Mistral (MistralEndpoint is a GenericOpenAIEndpoint on its API host) serves FIM at its
# own path; other OpenAI-compatible servers (Ollama, vLLM) use the completions path.
function get_url(s::GenericOpenAIEndpoint, ::FIMCompletion)::String
    base = rstrip(s.base_url, '/')
    base * (lowercase(HTTP.URI(base).host) == MISTRAL_API_HOST ? MISTRAL_FIM_PATH : COMPLETIONS_PATH)
end
# FIM is an OpenAI-compatible-only verb (DeepSeek beta + GenericOpenAIEndpoint); any other
# endpoint is rejected up front by `validate_capability(:fim)`. These fail-loud fallbacks
# give the router total coverage so `get_url(fim.service, fim)` types as `String` for any
# `fim.service::ServiceEndpointSpec` instead of leaving the marker-type limb methodless.
get_url(s::ServiceEndpoint, ::FIMCompletion) = throw(ArgumentError("FIM completion is not supported by $(_service_name(s))"))
get_url(::Type{<:ServiceEndpoint}, ::FIMCompletion) = throw(ArgumentError("FIM completion is not supported by this endpoint type"))

# ─── FIM Response Parsing ─────────────────────────────────────────────────

# Throws on a body that is not a completions response; fim_complete reports that as a
# FIMCallError, never as a success with invented empty text.
function _parse_fim_response(resp::HTTP.Response)::FIMResponse
    data = JSON.parse(resp.body; dicttype=Dict{String,Any})
    choices = data isa AbstractDict ? get(data, "choices", nothing) : nothing
    choices isa AbstractVector && !isempty(choices) ||
        throw(ArgumentError("FIM response carries no choices array"))
    usage = get(data, "usage", nothing)
    FIMResponse(
        choices=[FIMChoice(text=_fim_choice_text(c), index=get(c, "index", 0),
                           finish_reason=get(c, "finish_reason", nothing)) for c in choices],
        usage=usage isa AbstractDict ? _token_usage_from(usage) : nothing,
        model=something(get(data, "model", nothing), ""), raw=data)
end

# Completions choices carry `text`; Mistral's /v1/fim/completions answers in the chat
# shape, `message.content` (https://docs.mistral.ai/api/endpoint/fim).
function _fim_choice_text(c)::String
    c isa AbstractDict || throw(ArgumentError("FIM choice is not an object"))
    t = get(c, "text", nothing)
    t isa AbstractString && return t
    m = get(c, "message", nothing)
    t = m isa AbstractDict ? get(m, "content", nothing) : nothing
    t isa AbstractString || throw(ArgumentError("FIM choice carries no text"))
    t
end

# ─── FIM Request ──────────────────────────────────────────────────────────

"""
    fim_complete(fim::FIMCompletion; config=nothing, cancel=nothing) -> LLMRequestResponse

Execute a FIM (Fill-in-the-Middle) completion request. Returns [`FIMSuccess`](@ref),
[`FIMFailure`](@ref), or [`FIMCallError`](@ref).

Per-call `config::RequestConfig` overrides timeouts and the retry budget; a silent
peer fails with a typed timeout inside the [`FIMCallError`](@ref) result
(`cause::UniLMTimeout`). Local validation (capability, model resolution, routing)
throws an `ArgumentError` before any request; every later failure — transport, a
200 body that is not a completions response — is a `FIMCallError` whose `cause`
holds the exception.

`cancel::Union{Nothing,CancelToken}` (default: the ambient [`with_cancel`](@ref) token)
makes the call cancellable: once the token is cancelled it returns a `FIMCallError`
whose `cause` is a [`UniLMCancelled`](@ref) — without sending anything when the token
was already cancelled, at once when mid-request (never retried).
"""
function fim_complete(fim::FIMCompletion; config::Union{Nothing,RequestConfig}=nothing,
                      cancel::Union{Nothing,CancelToken}=nothing)::LLMRequestResponse
    validate_capability(fim.service, :fim, "FIM Completion")
    body = JSON.json(fim)
    url = get_url(fim.service, fim)::String
    cfg = _resolve_config(config); tok = _resolve_cancel(cancel); t0 = time_ns()
    local resp
    try
        resp = _http_with_retries(cfg, t0, "POST", url, auth_header(fim.service), body; cancel=tok)
        if resp.status == 200
            return FIMSuccess(response=_parse_fim_response(resp))
        else
            # Retry/backoff already ran inside the shared loop; the last real
            # response is the truthful outcome (a budget-exhausted 429 is a 429,
            # not a fabricated timeout).
            return FIMFailure(response=String(resp.body), status=resp.status, request_id=_get_request_id(resp))
        end
    catch e
        e isa InterruptException && rethrow()
        e isa UniLMTimeout && return FIMCallError(error=sprint(showerror, e), status=nothing, cause=e)
        statuserror = hasproperty(e, :status) ? e.status : nothing
        req_id = @isdefined(resp) ? _get_request_id(resp) : _get_request_id(e)
        return FIMCallError(error=_error_text(e), status=statuserror, request_id=req_id,
                            cause=e isa Exception ? e : nothing)
    end
end

"""
    fim_complete(prompt::String; suffix=nothing, kwargs...) -> LLMRequestResponse

Convenience form: creates a [`FIMCompletion`](@ref) and executes it. `config` and
`cancel` go to the request; every other keyword to the constructor.
"""
function fim_complete(prompt::String; suffix::Union{String,Nothing}=nothing, kwargs...)
    kws = Dict{Symbol,Any}(kwargs)
    config = pop!(kws, :config, nothing)
    cancel = pop!(kws, :cancel, nothing)
    fim_complete(FIMCompletion(; prompt, suffix, kws...); config, cancel)
end

# ─── Chat Prefix Completion ──────────────────────────────────────────────

_prefix_complete_url(s::DeepSeekEndpoint) = DEEPSEEK_BETA_BASE_URL * CHAT_COMPLETIONS_PATH
_prefix_complete_url(s::GenericOpenAIEndpoint) = rstrip(s.base_url, '/') * CHAT_COMPLETIONS_PATH
_prefix_complete_url(s) = get_url(s, Chat())  # fallback for other endpoints

"""
    prefix_complete(chat::Chat; config=nothing, cancel=nothing) -> LLMRequestResponse

Chat prefix completion: the model continues from a partial assistant message.
The last message in `chat` must be `role=assistant` containing the text prefix
to continue from. The result's `message` is the continuation the API returns; with
`chat.history` the conversation keeps the whole assistant turn, prefix followed by
continuation.

Supported by [`DeepSeekEndpoint`](@ref) (beta).

Per-call `config::RequestConfig` overrides timeouts and the retry budget; a silent
peer fails with a typed timeout inside the [`LLMCallError`](@ref) result
(`cause::UniLMTimeout`). Local validation throws an `ArgumentError` before any
request; every later failure is an `LLMCallError` whose `cause` holds the exception.
`cancel::Union{Nothing,CancelToken}` makes the call cancellable as for
[`fim_complete`](@ref): a cancelled call returns an `LLMCallError` whose `cause` is a
[`UniLMCancelled`](@ref), and `chat` is left untouched.

# Example
```julia
chat = Chat(service=DeepSeekEndpoint(), model="deepseek-flash")
push!(chat, Message(Val(:system), "You are a coding assistant."))
push!(chat, Message(Val(:user), "Write a quicksort in Python"))
push!(chat, Message(role=RoleAssistant, content="```python\\n"))
result = prefix_complete(chat)
```
"""
function prefix_complete(chat::Chat; config::Union{Nothing,RequestConfig}=nothing,
                         cancel::Union{Nothing,CancelToken}=nothing)::LLMRequestResponse
    validate_capability(chat.service, :prefix_completion, "Chat Prefix Completion")
    isempty(chat) && throw(ArgumentError("Chat must not be empty for prefix completion"))
    last(chat).role != RoleAssistant && throw(ArgumentError("Last message must be role=assistant for prefix completion"))
    body_dict = JSON.lower(chat)
    # Convert messages to mutable dicts so we can inject the prefix flag
    msgs = map(body_dict[:messages]) do m
        d = Dict{Symbol,Any}(:role => m.role)
        !isnothing(m.content) && (d[:content] = m.content)
        !isnothing(m.name) && (d[:name] = m.name)
        !isnothing(m.tool_calls) && (d[:tool_calls] = m.tool_calls)
        !isnothing(m.tool_call_id) && (d[:tool_call_id] = m.tool_call_id)
        d
    end
    msgs[end][:prefix] = true
    body_dict[:messages] = msgs
    body = JSON.json(body_dict)
    url = _prefix_complete_url(chat.service)
    cfg = _resolve_config(config); tok = _resolve_cancel(cancel); t0 = time_ns()
    local resp
    try
        resp = _http_with_retries(cfg, t0, "POST", url, auth_header(chat.service), body; cancel=tok)

        if resp.status == 200
            extracted = extract_message(resp)
            continuation = extracted.message
            # The API returns only the continuation: keep the whole turn in history.
            chat.history && (chat.messages[end] = _message_with(continuation, :content,
                something(last(chat).content, "") * something(continuation.content, "")))
            return LLMSuccess(message=continuation, self=chat, usage=extracted.usage)
        else
            # Retry/backoff already ran inside the shared loop; the last real
            # response is the truthful outcome.
            return LLMFailure(status=resp.status, response=String(resp.body), self=chat, request_id=_get_request_id(resp))
        end
    catch e
        e isa InterruptException && rethrow()
        e isa UniLMTimeout && return LLMCallError(error=sprint(showerror, e), self=chat, status=nothing, cause=e)
        statuserror = hasproperty(e, :status) ? e.status : nothing
        req_id = @isdefined(resp) ? _get_request_id(resp) : _get_request_id(e)
        return LLMCallError(error=_error_text(e), self=chat, status=statuserror, request_id=req_id,
                            cause=e isa Exception ? e : nothing)
    end
end
