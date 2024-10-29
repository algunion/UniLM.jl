get_url(model::String) = OPENAI_BASE_URL * _MODEL_ENDPOINTS[model]
get_url(params::Chat) = params.service == OPENAIServiceEndpoint ? get_url(params.model) : get_url(AZUREServiceEndpoint)
get_url(emb::Embeddings) = get_url(emb.model)
get_url(::Type{OPENAIServiceEndpoint}, params::Chat) = get_url(params)
get_url(::Type{OPENAIServiceEndpoint}, emb::Embeddings) = get_url(emb)
get_url(::Type{OPENAIServiceEndpoint}, model::String) = get_url(model)
function get_url(::Type{AZUREServiceEndpoint}, params::Chat)
    AZURE_OPENAI_BASE_URL * "/openai/deployments/" * AZURE_OPENAI_DEPLOY_NAME * "/chat/completions?api-version=$(getfield(params, :api_version))"
end

function auth_header(api_key::String=OPENAI_API_KEY)
    [
        "Authorization" => "Bearer $api_key",
        "Content-Type" => "application/json"
    ]
end

function auth_header(::Type{OPENAIServiceEndpoint})
    [
        "Authorization" => "Bearer $OPENAI_API_KEY",
        "Content-Type" => "application/json"
    ]
end

function auth_header(::Type{AZUREServiceEndpoint})
    @info "Using Azure OpenAI API with key: $AZURE_OPENAI_API_KEY"
    [
        "api-key" => "$AZURE_OPENAI_API_KEY",
        "Content-Type" => "application/json"
    ]
end

function extract_message(resp::HTTP.Response)::Message
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
        return Message(role=RoleAssistant, content=message["content"])
    elseif finish_reason == CONTENT_FILTER && haskey(message, "refusal") && !isnothing(message["refusal"])
        return Message(role=RoleAssistant, refusal_message=message["refusal"])
    else
        return Message(role=RoleAssistant, content="No response from the model.")
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
    chatrequest!(chat::Chat; callback=nothing)

Send a request to the OpenAI API to generate a response to the messages in `conv`.

The `callback` function is called for each chunk of the response. The `close` Ref is also passed to the callback function, which can be used to close the stream (for example when not satisfied with the intermediate results and want to stop the stream).
    
    The signature of the callback function is:
        `callback(chunk::Union{String, Message}, close::Ref{Bool})`
"""
function chatrequest!(chat::Chat; retries=0, callback=nothing)::Union{Task,LLMCallError,LLMFailure,LLMSuccess}
    res = LLMCallError(error="uninitialized", status=0, self=chat)
    @info "Sending chat request to $(chat.service)..."
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
                    return chatrequest!(chat; retries=retries + 1, callback=callback)
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
        statuserror = e isa HTTP.Exception ? e.status : nothing
        @info "HTTP status error: $statuserror"
        res = LLMCallError(error=e |> string, self=chat, status=statuserror)
    end
    @info "Returning from chatrequest!"
    return res
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
        resp = HTTP.post(get_url(emb), body=body, headers=auth_header())
        if resp.status == 200
            # headers info
            # @info "Request headers: $(resp.headers)"
            embedding = JSON3.read(resp.body, Dict)
            update!(emb, embedding["data"][1]["embedding"])
            return (embedding, emb)
        elseif resp.status == 500 || resp.status == 503
            @warn "Request status: $(resp.status). Retrying... in 1s"
            sleep(1)
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