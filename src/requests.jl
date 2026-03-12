# ─── Retry Infrastructure ────────────────────────────────────────────────────

const _RETRY_BASE = 1.0
const _RETRY_FACTOR = 2.0
const _RETRY_MAX_DELAY = 60.0
const _RETRY_MAX_ATTEMPTS = 30

_is_retryable(status::Integer)::Bool = status in (429, 500, 503)

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
get_url(::Embeddings) = OPENAI_BASE_URL * EMBEDDINGS_PATH

get_url(::Type{OPENAIServiceEndpoint}, ::Chat) = OPENAI_BASE_URL * CHAT_COMPLETIONS_PATH
get_url(::Type{AZUREServiceEndpoint}, chat::Chat) = ENV[AZURE_OPENAI_BASE_URL] * _MODEL_ENDPOINTS_AZURE_OPENAI[chat.model] * "/chat/completions?api-version=$(ENV[AZURE_OPENAI_API_VERSION])"
get_url(::Type{GEMINIServiceEndpoint}, ::Chat) = GEMINI_CHAT_URL

_api_base_url(::Type{OPENAIServiceEndpoint}) = OPENAI_BASE_URL
_api_base_url(::Type{AZUREServiceEndpoint}) = throw(ArgumentError("Responses API is only supported with OPENAIServiceEndpoint"))
_api_base_url(::Type{GEMINIServiceEndpoint}) = throw(ArgumentError("Responses API is only supported with OPENAIServiceEndpoint"))


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

function auth_header(::Type{GEMINIServiceEndpoint})
    [
        "Authorization" => "Bearer $(ENV[GEMINI_API_KEY])",
        "Content-Type" => "application/json"
    ]
end



# Stub: overridden by accounting.jl after include
_accumulate_cost!(::Chat, ::LLMRequestResponse) = nothing

function _parse_usage(data::Dict{String,Any})::Union{TokenUsage, Nothing}
    haskey(data, "usage") || return nothing
    u = data["usage"]
    u isa Dict || return nothing
    TokenUsage(
        prompt_tokens=get(u, "prompt_tokens", 0),
        completion_tokens=get(u, "completion_tokens", 0),
        total_tokens=get(u, "total_tokens", 0)
    )
end

function extract_message(resp::HTTP.Response)
    received_message = JSON.parse(resp.body; dicttype=Dict{String,Any})
    finish_reason = received_message["choices"][1]["finish_reason"]
    message = received_message["choices"][1]["message"]
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
    elseif finish_reason == STOP && haskey(message, "content") && !isnothing(message["content"])
        Message(role=RoleAssistant, content=message["content"], finish_reason=STOP)
    elseif finish_reason == CONTENT_FILTER && haskey(message, "refusal") && !isnothing(message["refusal"])
        Message(role=RoleAssistant, refusal_message=message["refusal"], finish_reason=CONTENT_FILTER)
    else
        Message(role=RoleAssistant, content="No response from the model.", finish_reason=finish_reason)
    end
    (; message=msg, usage)
end

"""Mutable accumulator for streaming Chat Completions chunks."""
@kwdef mutable struct StreamState
    content::IOBuffer = IOBuffer()
    tool_calls::Dict{Int, Dict{String,Any}} = Dict{Int, Dict{String,Any}}()
    finish_reason::Union{String, Nothing} = nothing
    usage::Union{TokenUsage, Nothing} = nothing
end

# There is a variable chunk format (which is also incomplete from time to time) that can be received from the stream.
# There are also multiple complaints about this on the OpenAI API forum.
# I'll attempt a robust parsing approach here that can handle the variable chunk format.
function _parse_chunk(chunk::String, state::StreamState, failbuff)
    lines = strip.(split(chunk, "\n"))
    lines = filter(!isempty, lines)
    isempty(lines) && return (; eos=false)
    eos = lines[end] == "data: [DONE]"
    eos && length(lines) == 1 && return (; eos=true)
    for line in lines[1:end-(eos ? 1 : 0)]
        try
            parsed = JSON.parse(line[6:end]; dicttype=Dict{String,Any})
            # Capture usage if present (e.g. stream_options.include_usage)
            if haskey(parsed, "usage") && parsed["usage"] isa Dict
                u = parsed["usage"]
                state.usage = TokenUsage(
                    prompt_tokens=get(u, "prompt_tokens", 0),
                    completion_tokens=get(u, "completion_tokens", 0),
                    total_tokens=get(u, "total_tokens", 0)
                )
            end
            cho = parsed["choices"][1]
            fr = cho["finish_reason"]
            if !isnothing(fr)
                state.finish_reason = fr
                fr == "stop" && (eos = true)
            end
            delta = get(cho, "delta", Dict{String,Any}())
            # Text content delta
            if haskey(delta, "content") && !isnothing(delta["content"])
                print(state.content, delta["content"])
            end
            # Tool call deltas — accumulate by index
            if haskey(delta, "tool_calls")
                for tc_delta in delta["tool_calls"]
                    idx = tc_delta["index"]
                    if !haskey(state.tool_calls, idx)
                        state.tool_calls[idx] = Dict{String,Any}(
                            "id" => get(tc_delta, "id", ""),
                            "type" => get(tc_delta, "type", "function"),
                            "function" => Dict{String,Any}("name" => "", "arguments" => "")
                        )
                    end
                    entry = state.tool_calls[idx]
                    haskey(tc_delta, "id") && !isempty(tc_delta["id"]) && (entry["id"] = tc_delta["id"])
                    if haskey(tc_delta, "function")
                        fd = tc_delta["function"]
                        haskey(fd, "name") && !isnothing(fd["name"]) && (entry["function"]["name"] *= fd["name"])
                        haskey(fd, "arguments") && !isnothing(fd["arguments"]) && (entry["function"]["arguments"] *= fd["arguments"])
                    end
                end
            end
        catch e
            print(failbuff, line)
            continue
        end
    end
    (; eos)
end

function _build_stream_message(state::StreamState)::Message
    if state.finish_reason == TOOL_CALLS && !isempty(state.tool_calls)
        tcalls = GPTToolCall[]
        for idx in sort(collect(keys(state.tool_calls)))
            tc_data = state.tool_calls[idx]
            fdict = tc_data["function"]
            args = JSON.parse(fdict["arguments"]; dicttype=Dict{String,Any})
            gptfunc = GPTFunction(fdict["name"], args)
            push!(tcalls, GPTToolCall(id=tc_data["id"], func=gptfunc))
        end
        Message(role=RoleAssistant, tool_calls=tcalls, finish_reason=TOOL_CALLS)
    else
        content = String(take!(state.content))
        Message(role=RoleAssistant, content=content, finish_reason=something(state.finish_reason, STOP))
    end
end

function _chatrequeststream(chat, body, callback=nothing; on_tool_call=nothing)
    Threads.@spawn begin
        try
            m = Ref{Union{Message,Nothing}}(nothing)
            stream_usage = Ref{Union{TokenUsage,Nothing}}(nothing)
            resp = HTTP.open("POST", get_url(chat), auth_header(chat.service)) do io
                state = StreamState()
                callback_buf = IOBuffer()  # tracks already-emitted text
                fail_buffer = IOBuffer()
                done = Ref(false)
                close_ref = Ref(false)
                prev_tc_count = 0
                write(io, body)
                HTTP.closewrite(io)
                HTTP.startread(io)
                while !eof(io) && !close_ref[] && !done[]
                    chunk::String = join((String(take!(fail_buffer)), String(readavailable(io))))
                    streamstatus = _parse_chunk(chunk, state, fail_buffer)
                    # Fire on_tool_call for newly completed tool calls
                    if !isnothing(on_tool_call) && length(state.tool_calls) > prev_tc_count
                        for idx in sort(collect(keys(state.tool_calls)))
                            idx < prev_tc_count && continue
                            tc_data = state.tool_calls[idx]
                            fdict = tc_data["function"]
                            # A tool call is "complete" when we have a non-empty name and parseable arguments
                            if !isempty(fdict["name"]) && !isempty(fdict["arguments"])
                                try
                                    args = JSON.parse(fdict["arguments"]; dicttype=Dict{String,Any})
                                    gptfunc = GPTFunction(fdict["name"], args)
                                    tc = GPTToolCall(id=tc_data["id"], func=gptfunc)
                                    on_tool_call(tc)
                                catch; end
                            end
                        end
                        prev_tc_count = length(state.tool_calls)
                    end
                    if streamstatus.eos
                        done[] = true
                        m[] = _build_stream_message(state)
                        stream_usage[] = state.usage
                        !isnothing(callback) && callback(m[], close_ref)
                    elseif !isnothing(callback)
                        # Emit only newly-arrived text
                        full = String(take!(state.content))
                        emitted = String(take!(callback_buf))
                        if sizeof(full) > sizeof(emitted)
                            delta_text = full[nextind(full, sizeof(emitted)):end]
                            callback(delta_text, close_ref)
                        end
                        print(state.content, full)
                        print(callback_buf, full)
                    end
                end
                close_ref[] && @info "stream closed by user"
                HTTP.closeread(io)
            end
            if resp.status == 200 && !isnothing(m[])
                msg = m[]::Message
                update!(chat, msg)
                result = LLMSuccess(message=msg, self=chat, usage=stream_usage[])
                _accumulate_cost!(chat, result)
                result
            else
                LLMFailure(status=resp.status, response=String(resp.body), self=chat)
            end
        catch e
            statuserror = hasproperty(e, :status) ? e.status : nothing
            LLMCallError(error=string(e), self=chat, status=statuserror)
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
    try
        body = JSON.json(chat)
        if chat.stream !== true
            resp = HTTP.post(get_url(chat), body=body, headers=auth_header(chat.service); status_exception=false)
            if resp.status == 200
                extracted = extract_message(resp)
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
                    return LLMFailure(status=resp.status, response=String(resp.body), self=chat)
                end
            else
                @error "Request status: $(resp.status)"
                return LLMFailure(status=resp.status, response=String(resp.body), self=chat)
            end
        else
            task = _chatrequeststream(chat, body, callback; on_tool_call)
            return task
        end
    catch e
        @info "Error: $e"
        statuserror = hasproperty(e, :status) ? e.status : nothing
        res = LLMCallError(error=string(e), self=chat, status=statuserror)
    end
    return res
end

"""
# Flexible keyword arguments usage
chatrequest!(; kwargs...)
Send a request to the OpenAI API to generate a response to the messages in `conv`.

# Keyword Arguments
- `service::Type{<:ServiceEndpoint} = AZUREServiceEndpoint`: The service endpoint to use (e.g., `AZUREServiceEndpoint`, `OPENAIServiceEndpoint`).
- `model::String = "gpt-5.2"`: The model to use for the chat completion.
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
    embeddingrequest!(emb::Embedding)

    Send a request to the OpenAI API to generate an embedding for the `input` in `emb`.
    
    Resulting embedding is stored in the preallocated `embedding` field.  

    @kwdef struct Embedding
        model::String = "text-embedding-3-small"
        input::Union{String,Vector{String}}
        embedding::Vector{Float64} = zeros(Float64, 1536)
        user::Union{String,Nothing} = nothing
    end

"""
function embeddingrequest!(emb::Embeddings; retries::Int=0)
    body = JSON.json(emb)
    try
        resp = HTTP.post(get_url(emb), body=body, headers=auth_header(OPENAIServiceEndpoint); status_exception=false)
        if resp.status == 200
            embedding = JSON.parse(resp.body; dicttype=Dict{String,Any})
            update!(emb, embedding["data"])
            return (embedding, emb)
        elseif _is_retryable(resp.status)
            if retries < _RETRY_MAX_ATTEMPTS
                delay = _retry_delay(retries, resp)
                @warn "Request status: $(resp.status). Retrying in $(round(delay; digits=2))s..."
                sleep(delay)
                return embeddingrequest!(emb; retries=retries + 1)
            else
                @error "Max retries ($(_RETRY_MAX_ATTEMPTS)) exceeded for embedding request."
                return nothing
            end
        else
            @error "Request status: $(resp.status)"
            return nothing
        end
    catch e
        @error e
        return nothing
    end
end