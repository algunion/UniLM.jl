function auth_header(api_key::String)
    [
        "Authorization" => "Bearer $api_key",
        "Content-Type" => "application/json"
    ]
end

function chat_request(conv::Conversation, params::ChatParams=ChatParams())    
    push!(params.messages, conv.messages...)
    body = JSON3.write(params)
    @info "Request body: $body"
    #req = HTTP.post(url, body=body, headers=Dict("Authorization" => "Bearer $key", "Content-Type" => "application/json"))
end

