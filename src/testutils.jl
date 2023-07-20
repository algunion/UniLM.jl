function get_current_weather(;location, unit="fahrenheit")
    Dict(
        "location" => location,
        "temperature" => "72",
        "unit" => unit,
        "forecast" => ["sunny", "windy"]
    )    
end