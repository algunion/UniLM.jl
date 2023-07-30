@testset "jsonschema.jl" begin
    jsonschema = UniLM.JsonObject(
        properties=Dict(
            "location" => UniLM.JsonString(description="The city and state, e.g. San Francisco, CA"),
            "unit" => UniLM.JsonString(enum=["celsius", "fahrenheit"])
        ),
        required=["location"]
    )

    newdesc = "Getting the current weather"

    updated_schema = UniLM.withdescription(jsonschema, newdesc)

    @test updated_schema.description == newdesc


end