# ============================================================================
# FIM Completion & Chat Prefix Completion
# FIM uses /v1/completions with prompt + suffix (fill-in-the-middle).
# Prefix Completion uses /v1/chat/completions with an assistant prefix message.
# Both are beta features on DeepSeek; also supported by Ollama and vLLM.
# ============================================================================

# ─── FIM Types ──────────────────────────────────────────────────────────────

"""
    FIMCompletion(; service, model, prompt, suffix=nothing, max_tokens=128, ...)

A Fill-in-the-Middle completion request. The model generates text between `prompt`
(prefix) and `suffix`.

Supported by [`DeepSeekEndpoint`](@ref) (beta), Ollama, vLLM.

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
end

function JSON.lower(fim::FIMCompletion)
    model = fim.model
    if isempty(model)
        dm = default_fim_model(fim.service)
        isnothing(dm) && throw(ArgumentError("model must be specified for FIM with $(typeof(fim.service))"))
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
end

"""Exception-level error during FIM completion."""
@kwdef struct FIMCallError <: LLMRequestResponse
    error::String
    status::Union{Int,Nothing} = nothing
end

# ─── FIM Accessors ─────────────────────────────────────────────────────────

"""
    fim_text(result) -> String

Extract the generated text from a FIM completion result.
"""
fim_text(r::FIMResponse)::String = isempty(r.choices) ? "" : r.choices[1].text
fim_text(r::FIMSuccess)::String = fim_text(r.response)
fim_text(::FIMFailure)::String = ""
fim_text(::FIMCallError)::String = ""

# ─── FIM URL Routing ──────────────────────────────────────────────────────

get_url(s::DeepSeekEndpoint, ::FIMCompletion) = DEEPSEEK_BETA_BASE_URL * COMPLETIONS_PATH
get_url(s::GenericOpenAIEndpoint, ::FIMCompletion) = rstrip(s.base_url, '/') * COMPLETIONS_PATH

# ─── FIM Response Parsing ─────────────────────────────────────────────────

function _parse_fim_response(resp::HTTP.Response)::FIMResponse
    data = JSON.parse(resp.body; dicttype=Dict{String,Any})
    choices = [FIMChoice(
        text=get(c, "text", ""),
        index=get(c, "index", 0),
        finish_reason=get(c, "finish_reason", nothing)
    ) for c in get(data, "choices", Any[])]
    usage_raw = get(data, "usage", nothing)
    usage = isnothing(usage_raw) ? nothing : TokenUsage(
        prompt_tokens=get(usage_raw, "prompt_tokens", 0),
        completion_tokens=get(usage_raw, "completion_tokens", 0),
        total_tokens=get(usage_raw, "total_tokens", 0)
    )
    FIMResponse(choices=choices, usage=usage, model=get(data, "model", ""), raw=data)
end

# ─── FIM Request ──────────────────────────────────────────────────────────

"""
    fim_complete(fim::FIMCompletion; retries=0) -> LLMRequestResponse

Execute a FIM (Fill-in-the-Middle) completion request. Returns [`FIMSuccess`](@ref),
[`FIMFailure`](@ref), or [`FIMCallError`](@ref).
"""
function fim_complete(fim::FIMCompletion; retries::Int=0)::LLMRequestResponse
    validate_capability(fim.service, :fim, "FIM Completion")
    try
        body = JSON.json(fim)
        url = get_url(fim.service, fim)
        resp = HTTP.post(url, body=body, headers=auth_header(fim.service); status_exception=false)
        if resp.status == 200
            return FIMSuccess(response=_parse_fim_response(resp))
        elseif _is_retryable(resp.status)
            if retries < _RETRY_MAX_ATTEMPTS
                delay = _retry_delay(retries, resp)
                @warn "Request status: $(resp.status). Retrying in $(round(delay; digits=2))s..."
                sleep(delay)
                return fim_complete(fim; retries=retries + 1)
            else
                return FIMFailure(response=String(resp.body), status=resp.status)
            end
        else
            return FIMFailure(response=String(resp.body), status=resp.status)
        end
    catch e
        e isa ArgumentError && rethrow()  # re-throw validation errors
        statuserror = hasproperty(e, :status) ? e.status : nothing
        return FIMCallError(error=string(e), status=statuserror)
    end
end

"""
    fim_complete(prompt::String; suffix=nothing, kwargs...) -> LLMRequestResponse

Convenience form: creates a [`FIMCompletion`](@ref) and executes it.
"""
function fim_complete(prompt::String; suffix::Union{String,Nothing}=nothing, kwargs...)
    kws = Dict{Symbol,Any}(kwargs)
    retries = pop!(kws, :retries, 0)
    fim_complete(FIMCompletion(; prompt, suffix, kws...); retries)
end

# ─── Chat Prefix Completion ──────────────────────────────────────────────

_prefix_complete_url(s::DeepSeekEndpoint) = DEEPSEEK_BETA_BASE_URL * CHAT_COMPLETIONS_PATH
_prefix_complete_url(s::GenericOpenAIEndpoint) = rstrip(s.base_url, '/') * CHAT_COMPLETIONS_PATH
_prefix_complete_url(s) = get_url(s, Chat())  # fallback for other endpoints

"""
    prefix_complete(chat::Chat; retries=0) -> LLMRequestResponse

Chat prefix completion: the model continues from a partial assistant message.
The last message in `chat` must be `role=assistant` containing the text prefix
to continue from.

Supported by [`DeepSeekEndpoint`](@ref) (beta).

# Example
```julia
chat = Chat(service=DeepSeekEndpoint(), model="deepseek-chat")
push!(chat, Message(Val(:user), "Write a quicksort in Python"))
push!(chat, Message(role=RoleAssistant, content="```python\\n"))
result = prefix_complete(chat)
```
"""
function prefix_complete(chat::Chat; retries::Int=0)::LLMRequestResponse
    validate_capability(chat.service, :prefix_completion, "Chat Prefix Completion")
    isempty(chat) && throw(ArgumentError("Chat must not be empty for prefix completion"))
    last(chat).role != RoleAssistant && throw(ArgumentError("Last message must be role=assistant for prefix completion"))

    try
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
        resp = HTTP.post(url, body=body, headers=auth_header(chat.service); status_exception=false)

        if resp.status == 200
            extracted = extract_message(resp)
            update!(chat, extracted.message)
            return LLMSuccess(message=extracted.message, self=chat, usage=extracted.usage)
        elseif _is_retryable(resp.status)
            if retries < _RETRY_MAX_ATTEMPTS
                delay = _retry_delay(retries, resp)
                @warn "Request status: $(resp.status). Retrying in $(round(delay; digits=2))s..."
                sleep(delay)
                return prefix_complete(chat; retries=retries + 1)
            else
                return LLMFailure(status=resp.status, response=String(resp.body), self=chat)
            end
        else
            return LLMFailure(status=resp.status, response=String(resp.body), self=chat)
        end
    catch e
        e isa ArgumentError && rethrow()
        return LLMCallError(error=string(e), self=chat)
    end
end
