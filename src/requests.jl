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

function _parse_chunk(chunk::String)
    strip(chunk)
end

"""
    _chat_request_stream(params::ChatParams, body::String, callback=nothing)::HTTP.Response

    Send a chat request to the OpenAI API and return the streamed response.

    The `callback` function is called for each chunk of the response. The `close` Ref is also passed to the callback function, which can be used to close the stream (for example when not satisfied with the intermediate results and want to stop the stream).
    
    The signature of the callback function is:
        `callback(chunk::String, close::Ref{Bool})::Nothing`
"""
function _chat_request_stream(params, body, callback=nothing)    
    Threads.@spawn HTTP.open("POST", get_url(params), auth_header()) do io
        done = Ref(false)
        close = Ref(false)
        write(io, body)
        HTTP.closewrite(io)
        HTTP.startread(io)
        @info "stream reading started"
        while !eof(io) && !close[] && !done[]
            chunk = String(readavailable(io))
            parsed = _parse_chunk(chunk)
            @info "parsed: $parsed"
            !isnothing(callback) && callback(chunk, close)
            if endswith(strip(chunk), "data: [DONE]") 
                done[] = true
            end            
        end
        close[] && @info "stream closed by user"
        @info "stream reading done"
        HTTP.closeread(io)
    end        
end

function chat_request(conv::Conversation; callback=nothing, params::ChatParams=ChatParams())
    push!(params.messages, conv.messages...)
    body = JSON3.write(params)
    if isnothing(params.stream)
        @info "chat_request: request/no-stream"
        resp = HTTP.post(get_url(params), body=body, headers=auth_header())
        m = resp.status == 200 ? extract_message(resp) : nothing
        m !== nothing && conv.history && push!(conv.messages, m)
        @info "chat_request: $m"
        return m
    else
        @info "chat_request: stream"
        _chat_request_stream(params, body, callback)
        @info "chat_request: stream task launched"
    end
end

