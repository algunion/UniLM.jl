get_url(params::Chat) = get_url(params.service, params.model)
get_url(emb::Embeddings) = get_url(OPENAIServiceEndpoint, emb.model)
get_url(::Type{OPENAIServiceEndpoint}, model::String) = OPENAI_BASE_URL * _MODEL_ENDPOINTS_OPENAI[model]
get_url(::Type{OPENAIServiceEndpoint}, emb::Embeddings) = get_url(emb)
get_url(::Type{AZUREServiceEndpoint}, model::String) = ENV[AZURE_OPENAI_BASE_URL] * _MODEL_ENDPOINTS_AZURE_OPENAI[model] * "/chat/completions?api-version=$(ENV[AZURE_OPENAI_API_VERSION])"


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

function extract_message(resp::HTTP.Response)
    received_message = JSON3.read(resp.body, Dict)
    finish_reason = received_message["choices"][1]["finish_reason"]
    message = received_message["choices"][1]["message"]
    if finish_reason == TOOL_CALLS && haskey(message, "tool_calls")
        tcalls = GPTToolCall[]
        for x in message["tool_calls"]
            fdict = x["function"]
            args = JSON3.read(fdict["arguments"], Dict)
            gptfunc = GPTFunction(fdict["name"], args)
            tc = GPTToolCall(id=x["id"], func=gptfunc)
            push!(tcalls, tc)
        end

        return Message(role=RoleAssistant, tool_calls=tcalls, finish_reason=TOOL_CALLS)
    elseif finish_reason == STOP && haskey(message, "content") && !isnothing(message["content"])
        return Message(role=RoleAssistant, content=message["content"], finish_reason=STOP)
    elseif finish_reason == CONTENT_FILTER && haskey(message, "refusal") && !isnothing(message["refusal"])
        return Message(role=RoleAssistant, refusal_message=message["refusal"], finish_reason=CONTENT_FILTER)
    else
        return Message(role=RoleAssistant, content="No response from the model.", finish_reason=finish_reason)
    end
end

# There is a variable chunk format (which is also incomplete from time to time) that can be received from the stream. 
# There are also multiple complaints about this on the OpenAI API forum. 
# I'll attempt a robust parsind approach here that can handle the variable chunk format.
function _parse_chunk(chunk::String, iobuff, failbuff)
    # check if chunk contains new line
    # if occursin("\n", chunk)
    #     @info "Newline in chunk $chunk"
    # end

    lines = strip.(split(chunk, "\n"))
    lines = filter(!isempty, lines)
    eos = lines[end] == "data: [DONE]"
    eos && length(lines) == 1 && return (; eos=true)
    for line in lines[1:end-(eos ? 1 : 0)]
        try
            parsed = JSON3.read(line[6:end], Vector)
            cho = parsed["choices"][1]
            if cho["finish_reason"] == "stop"
                eos = true
            else
                print(iobuff, cho["delta"]["content"])
            end
        catch e
            #@warn "JSON parsing failed for line: $line"
            #@info "Chunk was: $chunk"
            print(failbuff, line)
            continue
        end
    end
    (; eos=eos)
end

function _chatrequeststream(chat, body, callback=nothing)
    Threads.@spawn begin
        m = Ref{Union{Message,Nothing}}(nothing)
        resp = HTTP.open("POST", get_url(chat), auth_header(chat.service)) do io
            tmp = IOBuffer()
            chunk_buffer = IOBuffer()
            fail_buffer = IOBuffer()
            done = Ref(false)
            close = Ref(false)
            write(io, body)
            HTTP.closewrite(io)
            HTTP.startread(io)
            while !eof(io) && !close[] && !done[]
                chunk::String = join((String(take!(fail_buffer)), String(readavailable(io)))) # JET doesn't like * operator 
                streamstatus = _parse_chunk(chunk, chunk_buffer, fail_buffer)
                parsed = String(take!(chunk_buffer))
                if streamstatus.eos
                    done[] = true
                    m[] = Message(role=RoleAssistant, content=String(take!(tmp)))
                    !isnothing(callback) && callback(m[], close)
                else
                    !isnothing(callback) && callback(parsed, close)
                    print(tmp, parsed)
                end
            end
            close[] && @info "stream closed by user"
            #@info "Stream closing and returning"
            HTTP.closeread(io)
        end
        #@info "Finishing streaming with stutus: $(resp.status)"
        resp.status == 200 ? (m[], chat) : nothing
    end
end


"""
    chatrequest!(chat::Chat, retries=0, callback=nothing)

Send a request to the OpenAI API to generate a response to the messages in `conv`.

The `callback` function is called for each chunk of the response. The `close` Ref is also passed to the callback function, which can be used to close the stream (for example when not satisfied with the intermediate results and want to stop the stream).
    
    The signature of the callback function is:
        `callback(chunk::Union{String, Message}, close::Ref{Bool})`
"""
function chatrequest!(chat::Chat, retries=0, callback=nothing)
    res = LLMCallError(error="uninitialized", status=0, self=chat)
    try
        body = JSON3.write(chat)
        if isnothing(chat.stream) || !something(chat.stream)
            resp = HTTP.post(get_url(chat), body=body, headers=auth_header(chat.service))
            if resp.status == 200
                m = extract_message(resp)
                update!(chat, m)
                return LLMSuccess(message=m, self=chat)
            elseif resp.status == 500 || resp.status == 503
                @warn "Request status: $(resp.status)."
                @info "Retrying... in 1s"
                sleep(1)
                if retries < 30
                    return chatrequest!(chat, retries + 1, callback)
                else
                    return LLMFailure(status=resp.status, response=String(resp.body), self=chat)
                end
            else
                @error "Request status: $(resp.status)"
                return LLMFailure(status=resp.status, response=String(resp.body), self=chat)
            end
        else
            task = _chatrequeststream(chat, body, callback)
            return task
        end
    catch e
        @info "Error: $e"
        statuserror = hasfield(typeof(e), :status) ? getfield(e, :status) : nothing
        res = LLMCallError(error=e |> string, self=chat, status=statuserror)
    end
    @info "Returning from chatrequest!"
    return res
end

"""
# Flexible keyword arguments usage
chatrequest!(; kwargs...)
Send a request to the OpenAI API to generate a response to the messages in `conv`.

# Keyword Arguments
- `service::Type{<:ServiceEndpoint} = AZUREServiceEndpoint`: The service endpoint to use (e.g., `AZUREServiceEndpoint`, `OPENAIServiceEndpoint`).
- `model::String = "gpt-4o"`: The model to use for the chat completion.
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
    !haskey(kws, :messages) && (!haskey(kws, :userprompt) || !haskey(kws, :systemprompt)) && return LLMFailure(response="No messages and/or systemprompt/userprompt provided.", status=499, self=Chat(; kws...))
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
    nkws = filter(x -> x[1] != :messages && x[1] != :userprompt && x[1] != :systemprompt, kws)
    chatrequest!(Chat(; messages=messages, nkws...))
end


"""
    embeddingrequest!(emb::Embedding)

    Send a request to the OpenAI API to generate an embedding for the `input` in `emb`.
    
    Resulting embedding is stored in the preallocated `embedding` field.  

    @kwdef struct Embedding
        model::String = "text-embedding-ada-002"
        input::Union{String,Vector{String}}
        embedding::Vector{Float64} = zeros(Float64, 1536)
        user::Union{String,Nothing} = nothing
    end

"""
function embeddingrequest!(emb::Embeddings)
    body = JSON3.write(emb)
    try
        resp = HTTP.post(get_url(emb), body=body, headers=auth_header(OPENAIServiceEndpoint)) # default to OpenAI service for now
        if resp.status == 200
            # headers info
            # @info "Request headers: $(resp.headers)"
            embedding = JSON3.read(resp.body, Dict)
            update!(emb, embedding["data"][1]["embedding"])
            return (embedding, emb)
        elseif resp.status == 500 || resp.status == 503
            @warn "Request status: $(resp.status). Retrying... in 1s"
            sleep(1) # not the best way to handle this, but it's a start
            return embeddingrequest!(emb)
        else
            @error "Request status: $(resp.status)"
            return nothing
        end
    catch e
        @error e
        return nothing
    end
end