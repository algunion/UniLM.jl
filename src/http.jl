function auth_header(api_key::String)
    [
        "Authorization" => "Bearer $api_key",
        "Content-Type" => "application/json"
    ]
end

