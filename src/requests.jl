# ─── Retry Infrastructure ────────────────────────────────────────────────────

const _RETRY_BASE = 1.0
const _RETRY_FACTOR = 2.0
const _RETRY_MAX_DELAY = 60.0
const _RETRY_MAX_ATTEMPTS = 30

_is_retryable(status::Integer)::Bool = status in (408, 429, 500, 502, 503, 504, 529)

function _retry_delay(retry::Integer, resp::HTTP.Response)::Float64
    computed = min(_RETRY_BASE * _RETRY_FACTOR^retry, _RETRY_MAX_DELAY)
    delay = rand() * computed  # full jitter
    ra = HTTP.header(resp, "Retry-After", "")
    if !isempty(ra)
        parsed = tryparse(Int, ra)
        !isnothing(parsed) && parsed > 0 && (delay = max(Float64(parsed), delay))
    end
    delay
end

# ─── URL Dispatch ─────────────────────────────────────────────────────────────
# Endpoints are determined by (ServiceEndpoint, RequestType), not model name.

get_url(chat::Chat) = get_url(chat.service, chat)
get_url(emb::Embeddings) = get_url(emb.service, emb)

get_url(::Type{OPENAIServiceEndpoint}, ::Chat) = OPENAI_BASE_URL * CHAT_COMPLETIONS_PATH
get_url(::Type{AZUREServiceEndpoint}, chat::Chat) = ENV[AZURE_OPENAI_BASE_URL] * _MODEL_ENDPOINTS_AZURE_OPENAI[chat.model] * "/chat/completions?api-version=$(ENV[AZURE_OPENAI_API_VERSION])"
get_url(::Type{GEMINIOpenAIServiceEndpoint}, ::Chat) = GEMINI_CHAT_URL

get_url(::Type{OPENAIServiceEndpoint}, ::Embeddings) = OPENAI_BASE_URL * EMBEDDINGS_PATH
get_url(::Type{GEMINIOpenAIServiceEndpoint}, ::Embeddings) = GEMINI_OPENAI_BASE * "/embeddings"

_api_base_url(::Type{OPENAIServiceEndpoint}) = OPENAI_BASE_URL
_api_base_url(::Type{AZUREServiceEndpoint}) = throw(ArgumentError("Responses API is only supported with OPENAIServiceEndpoint"))
_api_base_url(::Type{GEMINIOpenAIServiceEndpoint}) = throw(ArgumentError("Responses API is only supported with OPENAIServiceEndpoint"))

# ─── GenericOpenAIEndpoint dispatch ──────────────────────────────────────────

get_url(s::GenericOpenAIEndpoint, ::Chat) = rstrip(s.base_url, '/') * CHAT_COMPLETIONS_PATH
get_url(s::GenericOpenAIEndpoint, ::Embeddings) = rstrip(s.base_url, '/') * EMBEDDINGS_PATH
_api_base_url(s::GenericOpenAIEndpoint) = rstrip(s.base_url, '/')

function auth_header(s::GenericOpenAIEndpoint)
    hdrs = ["Content-Type" => "application/json"]
    !isempty(s.api_key) && pushfirst!(hdrs, "Authorization" => "Bearer $(s.api_key)")
    hdrs
end

# ─── DeepSeekEndpoint dispatch ───────────────────────────────────────────────

get_url(s::DeepSeekEndpoint, ::Chat) = DEEPSEEK_BASE_URL * CHAT_COMPLETIONS_PATH
get_url(s::DeepSeekEndpoint, ::Embeddings) = DEEPSEEK_BASE_URL * EMBEDDINGS_PATH
_api_base_url(s::DeepSeekEndpoint) = DEEPSEEK_BASE_URL

function auth_header(s::DeepSeekEndpoint)
    ["Authorization" => "Bearer $(s.api_key)", "Content-Type" => "application/json"]
end

# ─── Built-in endpoint auth ─────────────────────────────────────────────────

function auth_header(::Type{OPENAIServiceEndpoint})
    [
        "Authorization" => "Bearer $(ENV[OPENAI_API_KEY])",
        "Content-Type" => "application/json"
    ]
end

function auth_header(::Type{AZUREServiceEndpoint})
    [
        "api-key" => "$(ENV[AZURE_OPENAI_API_KEY])",
        "Content-Type" => "application/json"
    ]
end

function auth_header(::Type{GEMINIOpenAIServiceEndpoint})
    [
        "Authorization" => "Bearer $(ENV[GEMINI_API_KEY])",
        "Content-Type" => "application/json"
    ]
end



# Multipart auth: strip the JSON Content-Type so HTTP.Form can set its own
# multipart/form-data boundary header (a stray application/json corrupts the upload).
auth_header_multipart(s) = filter(p -> lowercase(String(first(p))) != "content-type", auth_header(s))

# Stub: overridden by accounting.jl after include
_accumulate_cost!(::Chat, ::LLMRequestResponse) = nothing

"""
    _token_usage_from(u::AbstractDict; prompt_key, completion_key, prompt_details, completion_details)

Build a [`TokenUsage`](@ref) from a raw `usage` dict, pulling the cached/reasoning
subtotals from the detail objects when present. Chat Completions and the Responses API
name these differently (`prompt_tokens`/`input_tokens`, `prompt_tokens_details`/
`input_tokens_details`, …), so the keys are parameterized.
"""
function _token_usage_from(u::AbstractDict;
    prompt_key::String="prompt_tokens", completion_key::String="completion_tokens",
    prompt_details::String="prompt_tokens_details", completion_details::String="completion_tokens_details")
    pd = get(u, prompt_details, nothing)
    cd = get(u, completion_details, nothing)
    _i(x) = x isa Integer ? Int(x) : 0   # tolerate JSON null / missing / non-int → 0
    TokenUsage(
        prompt_tokens=_i(get(u, prompt_key, 0)),
        completion_tokens=_i(get(u, completion_key, 0)),
        total_tokens=_i(get(u, "total_tokens", 0)),
        cached_tokens=(pd isa AbstractDict ? _i(get(pd, "cached_tokens", 0)) : 0),
        reasoning_tokens=(cd isa AbstractDict ? _i(get(cd, "reasoning_tokens", 0)) : 0),
    )
end

function _parse_usage(data::Dict{String,Any})::Union{TokenUsage, Nothing}
    haskey(data, "usage") || return nothing
    u = data["usage"]
    u isa AbstractDict || return nothing
    _token_usage_from(u)
end

function extract_message(resp::HTTP.Response)
    received_message = JSON.parse(resp.body; dicttype=Dict{String,Any})
    choices = get(received_message, "choices", [])
    isempty(choices) && error("API returned empty choices array")
    finish_reason = choices[1]["finish_reason"]
    message = choices[1]["message"]
    usage = _parse_usage(received_message)
    msg = if finish_reason == TOOL_CALLS && haskey(message, "tool_calls")
        tcalls = GPTToolCall[]
        for x in message["tool_calls"]
            fdict = x["function"]
            args = JSON.parse(fdict["arguments"]; dicttype=Dict{String,Any})
            gptfunc = GPTFunction(fdict["name"], args)
            tc = GPTToolCall(id=x["id"], func=gptfunc)
            push!(tcalls, tc)
        end
        Message(role=RoleAssistant, tool_calls=tcalls, finish_reason=TOOL_CALLS)
    elseif haskey(message, "content") && !isnothing(message["content"])
        # Preserve content for ANY finish_reason (incl. "length"/truncated) — never discard partial output.
        Message(role=RoleAssistant, content=message["content"], finish_reason=finish_reason)
    elseif haskey(message, "refusal") && !isnothing(message["refusal"])
        # A refusal may arrive with finish_reason "content_filter" OR "stop" — capture it regardless.
        Message(role=RoleAssistant, refusal_message=message["refusal"], finish_reason=finish_reason)
    else
        Message(role=RoleAssistant, content="No response from the model.", finish_reason=finish_reason)
    end
    (; message=msg, usage)
end

"""Mutable accumulator for streaming Chat Completions chunks."""
@kwdef mutable struct StreamState
    content::IOBuffer = IOBuffer()
    refusal::IOBuffer = IOBuffer()
    tool_calls::Dict{Int, Dict{String,Any}} = Dict{Int, Dict{String,Any}}()
    finish_reason::Union{String, Nothing} = nothing
    usage::Union{TokenUsage, Nothing} = nothing
    # ── streaming-machine additions (0.11.3) ──
    # Text deltas collected by handlers, not yet forwarded to the callback;
    # the driver take!s and forwards verbatim (kills the take!/re-print churn).
    pending_delta::IOBuffer = IOBuffer()
    # In-band terminal stream error (e.g. Anthropic `error` event on HTTP 200);
    # non-nothing → the driver returns LLMFailure/LLMCallError, never LLMSuccess.
    error::Union{Nothing, Dict{String,Any}} = nothing
    # on_tool_call fire-once-per-index guard (final sweep at stream end).
    fired_tool_calls::Set{Int} = Set{Int}()
end

function _build_stream_message(state::StreamState)::Message
    content = String(take!(state.content))
    refusal = String(take!(state.refusal))
    if !isempty(state.tool_calls)
        tcalls = GPTToolCall[]
        for idx in sort!(collect(keys(state.tool_calls)))
            tc_data = state.tool_calls[idx]
            fdict = tc_data["function"]
            args = _parse_tool_arguments(fdict["arguments"])   # "" → Dict{String,Any}() (zero-arg tool call)
            push!(tcalls, GPTToolCall(id=tc_data["id"], func=GPTFunction(fdict["name"], args),
                thought_signature=get(tc_data, "thought_signature", nothing)))
        end
        # Keep accumulated text ALONGSIDE the tool calls: providers emit
        # both in one turn and the non-streaming decoders already preserve both.
        Message(role=RoleAssistant, content=(isempty(content) ? nothing : content),
                tool_calls=tcalls, finish_reason=TOOL_CALLS)
    elseif isempty(content) && !isempty(refusal)
        Message(role=RoleAssistant, refusal_message=refusal, finish_reason=something(state.finish_reason, STOP))
    else
        Message(role=RoleAssistant, content=content, finish_reason=something(state.finish_reason, STOP))
    end
end

# ─── Wire-translation seam ───────────────────────────────────────────────────
# Generics translate between the neutral Chat/Message IR and a provider's wire
# format. The untyped-`service` methods are the OpenAI-wire defaults — DeepSeek,
# Azure, the Gemini OpenAI-compat shim, and GenericOpenAIEndpoint all speak this
# format. Providers with a different wire (Anthropic, native Gemini) override
# them. `chatrequest!`/`_chatrequeststream` call ONLY these generics plus the
# streaming seam `handle_sse_event!` (src/sse.jl), so the retry/HTTP/cost/
# tool-loop/streaming orchestration stays provider-agnostic.

"""
    encode_request(service, chat::Chat) -> String

Serialize `chat` into the provider's request body. Default: OpenAI Chat Completions JSON.
"""
encode_request(service, chat::Chat) = JSON.json(chat)

"""
    decode_response(service, resp::HTTP.Response)

Parse a provider's 200 response into `(; message::Message, usage::Union{TokenUsage,Nothing})`.
Default: OpenAI Chat Completions (`extract_message`).
"""
decode_response(service, resp::HTTP.Response) = extract_message(resp)

# ─── Streaming driver helpers ────────────────────────────────────────────────

"""
    _flush_delta!(callback, state::StreamState, close_ref) -> Nothing

Forward collected-but-unsent text deltas to the streaming callback verbatim.
Handlers collect deltas on `state.pending_delta`; the driver forwards them
directly — no `take!`/re-print churn, no byte-offset diffing (which broke on
multibyte boundaries and was O(n²)).
"""
function _flush_delta!(callback, state::StreamState, close_ref)::Nothing
    delta = String(take!(state.pending_delta))
    isnothing(callback) || isempty(delta) || callback(delta, close_ref)
    nothing
end

"""
    _fire_tool_calls!(on_tool_call, state::StreamState, stream_done::Bool) -> Nothing

Provider-agnostic `on_tool_call` completion detection. A
tool call at index `i` is complete when (a) `i` is no longer the max index (a
later call started), OR (b) its entry carries `"complete" => true` (Anthropic
`content_block_stop` / Gemini whole-part functionCall), OR (c) the stream is
done (`stream_done` — the final sweep). Fires at most once per index
(`state.fired_tool_calls`). Empty accumulated arguments parse as
`Dict{String,Any}()` via `_parse_tool_arguments`.
"""
function _fire_tool_calls!(on_tool_call, state::StreamState, stream_done::Bool)::Nothing
    (isnothing(on_tool_call) || isempty(state.tool_calls)) && return nothing
    maxidx = maximum(keys(state.tool_calls))
    for idx in sort!(collect(keys(state.tool_calls)))
        idx in state.fired_tool_calls && continue
        tc_data = state.tool_calls[idx]
        stream_done || idx < maxidx || get(tc_data, "complete", false) === true || continue
        fdict = tc_data["function"]
        args = try
            _parse_tool_arguments(fdict["arguments"])
        catch e
            # A COMPLETE call's arguments cannot improve later: warn once, never retry.
            push!(state.fired_tool_calls, idx)
            @warn "on_tool_call: undecodable tool-call arguments; not firing" index = idx exception = e
            continue
        end
        push!(state.fired_tool_calls, idx)   # before the user callback: a throwing callback must not re-fire
        try
            on_tool_call(GPTToolCall(id=tc_data["id"], func=GPTFunction(fdict["name"], args),
                thought_signature=get(tc_data, "thought_signature", nothing)))
        catch e
            @warn "on_tool_call callback error" exception = e
        end
    end
    nothing
end

"""
    _stream_error_result(chat, err::Dict{String,Any}, request_id)

Map an in-band SSE `error` payload (`state.error`) to a typed non-success
result: `overloaded_error` is the documented
529-equivalent → `LLMFailure(status=529)` (status-keyed policies see it);
any other in-band error type → `LLMCallError` (no fabricated HTTP status).
"""
function _stream_error_result(chat::Chat, err::Dict{String,Any}, request_id)
    inner = get(err, "error", nothing)
    etype = inner isa AbstractDict ? get(inner, "type", "") : ""
    etype == "overloaded_error" ?
        LLMFailure(status=529, response=JSON.json(err), self=chat, request_id=request_id) :
        LLMCallError(error=JSON.json(err), self=chat, status=nothing, request_id=request_id)
end

function _chatrequeststream(chat, body, callback=nothing; on_tool_call=nothing)
    Threads.@spawn begin
        io_ref = Ref{Union{HTTP.Stream,Nothing}}(nothing)
        try
            m = Ref{Union{Message,Nothing}}(nothing)
            stream_usage = Ref{Union{TokenUsage,Nothing}}(nothing)
            stream_error = Ref{Union{Dict{String,Any},Nothing}}(nothing)
            raw_buffer = IOBuffer()  # wire bytes for non-200/truncation reporting (streamed resp.body is empty under HTTP 2.x)
            # SSE must reach the parser uncompressed. Some providers (e.g. Anthropic) gzip even
            # streamed responses, and HTTP.jl's streaming read loop does NOT auto-decompress on the
            # 1.x major — raw gzip bytes hit the SSE parser, every chunk fails to decode, and no
            # message is built (→ LLMFailure). Request identity encoding + disable decompression so
            # `data:` lines arrive verbatim on both HTTP majors.
            stream_headers = push!(copy(auth_header(chat.service)), "Accept-Encoding" => "identity")
            resp = HTTP.open("POST", get_url(chat), stream_headers; status_exception=false, decompress=false) do io
                io_ref[] = io
                state = StreamState()
                carry = IOBuffer()                 # layer-1 partial-line carry
                current_event = Ref("")            # layer-2 sticky event name
                close_ref = Ref(false)
                status = :continue
                write(io, body)
                HTTP.closewrite(io)
                HTTP.startread(io)
                while !eof(io) && !close_ref[] && status === :continue
                    raw = String(readavailable(io))
                    write(raw_buffer, raw)
                    status = _sse_dispatch!(chat.service, carry, current_event, raw, state)
                    _fire_tool_calls!(on_tool_call, state, false)
                    _flush_delta!(callback, state, close_ref)
                end
                if status === :continue && !close_ref[]
                    # EOF flush: a final line the server never '\n'-terminated
                    # (e.g. `data: [DONE]` as the very last bytes) is still one
                    # complete line — dispatch it before finalizing.
                    tail = String(take!(carry))
                    if !isempty(tail)
                        status = _sse_dispatch!(chat.service, carry, current_event, tail * "\n", state)
                        _fire_tool_calls!(on_tool_call, state, false)
                        _flush_delta!(callback, state, close_ref)
                    end
                end
                stream_error[] = state.error
                # Terminal contract: `:done` is the sentinel EOS
                # ([DONE] / message_stop). Gemini has NO sentinel — its handler
                # never returns :done; its stream ends at EOF with finishReason
                # recorded. EOF with NEITHER signal = truncated/garbage stream
                # (or a non-200 error body) → no message → LLMFailure below.
                finished = status === :done ||
                           (status === :continue && !isnothing(state.finish_reason))
                if isnothing(state.error) && !close_ref[] && finished
                    _fire_tool_calls!(on_tool_call, state, true)   # final sweep BEFORE the terminal callback
                    m[] = _build_stream_message(state)
                    stream_usage[] = state.usage
                    !isnothing(callback) && callback(m[], close_ref)
                end
                close_ref[] && @info "stream closed by user"
                HTTP.closeread(io)
            end
            if !isnothing(stream_error[])
                # In-band `error` event on an HTTP-200 stream: never LLMSuccess.
                _stream_error_result(chat, stream_error[], _get_request_id(resp))
            elseif resp.status == 200 && !isnothing(m[])
                msg = m[]::Message
                update!(chat, msg)
                result = LLMSuccess(message=msg, self=chat, usage=stream_usage[])
                _accumulate_cost!(chat, result)
                result
            else
                LLMFailure(status=resp.status, response=String(take!(raw_buffer)), self=chat, request_id=_get_request_id(resp))
            end
        catch e
            statuserror = hasproperty(e, :status) ? e.status : nothing
            req_id = !isnothing(io_ref[]) ? _get_request_id(io_ref[]) : _get_request_id(e)
            LLMCallError(error=string(e), self=chat, status=statuserror, request_id=req_id)
        end
    end
end


"""
    chatrequest!(chat::Chat, retries=0, callback=nothing)

Send a request to the OpenAI API to generate a response to the messages in `conv`.

The `callback` function is called for each chunk of the response. The `close` Ref is also passed to the callback function, which can be used to close the stream (for example when not satisfied with the intermediate results and want to stop the stream).
    
    The signature of the callback function is:
        `callback(chunk::Union{String, Message}, close::Ref{Bool})`
"""
function chatrequest!(chat::Chat; retries::Int=0, callback=nothing, on_tool_call=nothing)
    res = LLMCallError(error="uninitialized", status=0, self=chat)
    local resp
    try
        body = encode_request(chat.service, chat)
        if chat.stream !== true
            resp = HTTP.post(get_url(chat), body=body, headers=auth_header(chat.service); status_exception=false)
            if resp.status == 200
                extracted = decode_response(chat.service, resp)
                update!(chat, extracted.message)
                result = LLMSuccess(message=extracted.message, self=chat, usage=extracted.usage)
                _accumulate_cost!(chat, result)
                return result
            elseif _is_retryable(resp.status)
                if retries < _RETRY_MAX_ATTEMPTS
                    delay = _retry_delay(retries, resp)
                    @warn "Request status: $(resp.status). Retrying in $(round(delay; digits=2))s..."
                    sleep(delay)
                    return chatrequest!(chat; retries=retries + 1, callback, on_tool_call)
                else
                    return LLMFailure(status=resp.status, response=String(resp.body), self=chat, request_id=_get_request_id(resp))
                end
            else
                @error "Request status: $(resp.status)"
                return LLMFailure(status=resp.status, response=String(resp.body), self=chat, request_id=_get_request_id(resp))
            end
        else
            task = _chatrequeststream(chat, body, callback; on_tool_call)
            return task
        end
    catch e
        @info "Error: $e"
        statuserror = hasproperty(e, :status) ? e.status : nothing
        req_id = @isdefined(resp) ? _get_request_id(resp) : _get_request_id(e)
        res = LLMCallError(error=string(e), self=chat, status=statuserror, request_id=req_id)
    end
    return res
end

"""
# Flexible keyword arguments usage
chatrequest!(; kwargs...)
Send a request to the OpenAI API to generate a response to the messages in `conv`.

# Keyword Arguments
- `service::Type{<:ServiceEndpoint} = AZUREServiceEndpoint`: The service endpoint to use (e.g., `AZUREServiceEndpoint`, `OPENAIServiceEndpoint`).
- `model::String = "gpt-5.5"`: The model to use for the chat completion.
- `systemprompt::Union{Message,String}`: The system prompt message.
- `userprompt::Union{Message,String}`: The user prompt message.
- `messages::Conversation = Message[]`: The conversation history or the system/prompt messages.
- `history::Bool = true`: Whether to include the conversation history in the request.
- `tools::Union{Vector{GPTTool},Nothing} = nothing`: A list of tools the model may call.
- `tool_choice::Union{String,GPTToolChoice,Nothing} = nothing`: Controls which (if any) function is called by the model. e.g. "auto", "none", `GPTToolChoice`.
- `parallel_tool_calls::Union{Bool,Nothing} = false`: Whether to enable parallel function calling.
- `temperature::Union{Float64,Nothing} = nothing`: Sampling temperature (0.0-2.0). Higher values make output more random. Mutually exclusive with `top_p`.
- `top_p::Union{Float64,Nothing} = nothing`: Nucleus sampling parameter (0.0-1.0). Mutually exclusive with `temperature`.
- `n::Union{Int64,Nothing} = nothing`: How many chat completion choices to generate for each input message (1-10).
- `stream::Union{Bool,Nothing} = nothing`: If set, partial message deltas will be sent, like in ChatGPT.
- `stop::Union{Vector{String},String,Nothing} = nothing`: Up to 4 sequences where the API will stop generating further tokens.
- `max_tokens::Union{Int64,Nothing} = nothing`: The maximum number of tokens to generate in the chat completion.
- `presence_penalty::Union{Float64,Nothing} = nothing`: Number between -2.0 and 2.0. Positive values penalize new tokens based on whether they appear in the text so far.
- `response_format::Union{ResponseFormat,Nothing} = nothing`: An object specifying the format that the model must output. e.g., `ResponseFormat(type="json_object")`.
- `frequency_penalty::Union{Float64,Nothing} = nothing`: Number between -2.0 and 2.0. Positive values penalize new tokens based on their existing frequency in the text so far.
- `logit_bias::Union{AbstractDict{String,Float64},Nothing} = nothing`: Modify the likelihood of specified tokens appearing in the completion.
- `user::Union{String,Nothing} = nothing`: A unique identifier representing your end-user, which can help OpenAI to monitor and detect abuse.
- `seed::Union{Int64,Nothing} = nothing`: This feature is in Beta. If specified, the system will make a best effort to sample deterministically.
"""
function chatrequest!(; kws...)
    filteredkws = filter(x -> x[1] != :messages && x[1] != :userprompt && x[1] != :systemprompt, kws)
    !haskey(kws, :messages) && (!haskey(kws, :userprompt) || !haskey(kws, :systemprompt)) && return LLMFailure(response="No messages and/or systemprompt/userprompt provided.", status=499, self=Chat(; filteredkws...))
    messages = get(kws, :messages, Message[])
    if haskey(kws, :userprompt) && haskey(kws, :systemprompt)
        empty!(messages)
        if kws[:systemprompt] isa AbstractString
            push!(messages, Message(role=RoleSystem, content=kws[:systemprompt]))
        else
            push!(messages, kws[:systemprompt])
        end
        if kws[:userprompt] isa AbstractString
            push!(messages, Message(role=RoleUser, content=kws[:userprompt]))
        else
            push!(messages, kws[:userprompt])
        end
    end
    chatrequest!(Chat(; messages=messages, filteredkws...))
end


"""
    embeddingrequest!(emb::Embeddings; retries=0) -> LLMRequestResponse

Send an Embeddings API request for the `input` in `emb`. Returns `EmbeddingSuccess`,
`EmbeddingFailure` (non-2xx), or `EmbeddingCallError` (network/parse). The resulting
vectors are filled into `emb.embeddings` in place and are also reachable via
`embedding_vectors(result)`.
"""
function embeddingrequest!(emb::Embeddings; retries::Int=0)
    try
        body = JSON.json(emb)
        resp = HTTP.post(get_url(emb), body=body, headers=auth_header(emb.service); status_exception=false)
        if resp.status == 200
            data = JSON.parse(resp.body; dicttype=Dict{String,Any})
            update!(emb, data["data"])
            return EmbeddingSuccess(embeddings=emb, usage=_parse_usage(data), raw=data)
        elseif _is_retryable(resp.status)
            if retries < _RETRY_MAX_ATTEMPTS
                delay = _retry_delay(retries, resp)
                @warn "Request status: $(resp.status). Retrying in $(round(delay; digits=2))s..."
                sleep(delay)
                return embeddingrequest!(emb; retries=retries + 1)
            else
                return EmbeddingFailure(response=String(resp.body), status=resp.status)
            end
        else
            return EmbeddingFailure(response=String(resp.body), status=resp.status)
        end
    catch e
        statuserror = hasproperty(e, :status) ? e.status : nothing
        return EmbeddingCallError(error=string(e), status=statuserror)
    end
end