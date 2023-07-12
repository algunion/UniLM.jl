get_url(model::String) = OPENAI_BASE_URL * _MODEL_ENDPOINTS[model]
get_url(params::ChatParams) = get_url(params.model)

function auth_header(api_key::String=OPENAI_API_KEY)
    [
        "Authorization" => "Bearer $api_key",
        "Content-Type" => "application/json"
    ]
end

function extract_message(resp::HTTP.Response)
    received_message = JSON3.read(resp.body, Dict)
    content = received_message["choices"][1]["message"]["content"]
    Message(role=GPTAssistant, content=content)
end

# There is a variable chunk format (which is also incomplete from time to time) that can be received from the stream. There are also multiple complaints about this on the OpenAI API forum. I'll attempt a robust parsind approach here that can handle the variable chunk format.
function _parse_chunk(chunk::String, iobuff)    
    lines = strip.(split(chunk, "\n"))    
    lines = filter(!isempty, lines)
    eos = lines[end] == "data: [DONE]"
    eos && length(lines) == 1 && return (;eos=true)
    for line in lines[1:end-(eos ? 1 : 0)]
        try
            parsed = JSON3.read(line[6:end])["choices"][1]
            if parsed["finish_reason"] == "stop"
                eos = true
            else
                print(iobuff, parsed["delta"]["content"])
            end            
        catch e
            @warn "JSON parsing failed for line: $line"
            continue
        end
    end    
    (;eos=eos)
end

function _chat_request_stream(params, body, callback=nothing)    
    Threads.@spawn begin         
        m = Ref{Union{Message, Nothing}}(nothing)
        resp = HTTP.open("POST", get_url(params), auth_header()) do io
            tmp = IOBuffer()
            chunk_buffer = IOBuffer()
            done = Ref(false)
            close = Ref(false)
            write(io, body)
            HTTP.closewrite(io)
            HTTP.startread(io)        
            while !eof(io) && !close[] && !done[]
                chunk = String(readavailable(io))  
                @info "chunk: $chunk"
                streamstatus = _parse_chunk(chunk, chunk_buffer)
                parsed = String(take!(chunk_buffer))
                if streamstatus.eos 
                    done[] = true
                    m[] = Message(role=GPTAssistant, content=String(take!(tmp)))
                    !isnothing(callback) && callback(m[], close)
                else
                    !isnothing(callback) && callback(parsed, close)
                    print(tmp, parsed)
                end            
            end
            close[] && @info "stream closed by user"
            @info "Stream closing and returning"        
            HTTP.closeread(io)             
        end
        resp.status == 200 ? m[] : @error "Request staus: " resp.status; nothing
    end        
end


"""
    chat_request(conv::Conversation; callback=nothing, params::ChatParams=ChatParams())

Send a request to the OpenAI API to generate a response to the messages in `conv`.

The `callback` function is called for each chunk of the response. The `close` Ref is also passed to the callback function, which can be used to close the stream (for example when not satisfied with the intermediate results and want to stop the stream).
    
    The signature of the callback function is:
        `callback(chunk::Union{String, Message}, close::Ref{Bool})`
"""
function chat_request(conv::Conversation; callback=nothing, params::ChatParams=ChatParams())::Union{Message, Task, Nothing}
    push!(params.messages, conv.messages...)
    body = JSON3.write(params)
    if isnothing(params.stream) || !something(params.stream)
        @info "chat_request: request/no-stream"
        resp = HTTP.post(get_url(params), body=body, headers=auth_header())
        m = resp.status == 200 ? extract_message(resp) : @error "Request staus: " resp.status; nothing
        m !== nothing && conv.history && push!(conv.messages, m)
        @info "chat_request: $m"
        return m
    else
        @info "chat_request: stream"
        task = _chat_request_stream(params, body, callback)
        @info "chat_request: stream task launched"
        task
    end
end

