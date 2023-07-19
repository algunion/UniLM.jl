function get_current_weather(;location, unit="fahrenheit")
    weather_info = Dict(
        "location" => location,
        "temperature" => "72",
        "unit" => unit,
        "forecast" => ["sunny", "windy"]
    )
    return JSON3.write(weather_info)
end