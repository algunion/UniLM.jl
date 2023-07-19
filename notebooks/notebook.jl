### A Pluto.jl notebook ###
# v0.19.27

using Markdown
using InteractiveUtils

# ╔═╡ 9ae84052-264c-11ee-2e7d-1dc9f61fa7c8
begin
	using Pkg
	using JSON3
	Pkg.add(url="https://github.com/algunion/UniLM.jl.git")
	using UniLM;
end

# ╔═╡ d24af788-59ef-4bd0-8fd1-b4cce5106f2d
get_current_weather_schema = Dict(
        "name" => "get_current_weather",
        "description" => "Get the current weather in a given location",
        "parameters" => Dict(
            "type" => "object",
            "properties" => Dict(
                "location" => Dict(
                    "type" => "string",
                    "description" => "The city and state, e.g. San Francisco, CA"
                ),
                "unit" => Dict(
                    "type" => "string",
                    "enum" => ["celsius", "fahrenheit"]
                )
            ),
            "required" => ["location"]
        )
    )

# ╔═╡ 23e1b1fa-64d4-4c18-a740-8378c24d00e6
gptfsig = UniLM.GPTFunctionSignature(name=get_current_weather_schema["name"], description=get_current_weather_schema["description"], parameters=get_current_weather_schema["parameters"])

# ╔═╡ 015895ad-ddfc-40ce-b43c-515a95df0ca3
begin
	funchat = UniLM.Chat(functions=[gptfsig], function_call="auto")
    
    push!(funchat, UniLM.Message(role=UniLM.GPTSystem, content="Act as a helpful AI agent."))
    push!(funchat, UniLM.Message(role=UniLM.GPTUser, content="What's the weather like in Boston?"))

	(m, _) = UniLM.chat_request!(funchat)    
end

# ╔═╡ 39993fba-21a8-4515-b3af-b40f046337b5
m

# ╔═╡ 86151d49-73f4-48d9-87d1-02149a9d42dd
funchat

# ╔═╡ 6f6fe9e6-1b72-46ac-a6dc-527ef0f8f6fa
r = UniLM.evalcall(m)

# ╔═╡ 3811d45f-6ec2-4190-9431-79caa560c80e
typeof(r)

# ╔═╡ dc80bb85-bada-4751-9228-2f1bf7c4212a
fname = m.function_call["name"]

# ╔═╡ 508bec95-b761-4a01-a550-4f23ec1f1974
funmsg = Message(role=GPTFunction,name=fname, content=r)

# ╔═╡ 15bbdad9-1f13-4b71-8a5e-7d07380cc8fa
update!(funchat, funmsg)

# ╔═╡ daf74b90-3cbe-4f62-ac27-836d01d77ce6
funchat.messages[3].function_call["arguments"] = funchat.messages[3].function_call["arguments"] |> JSON3.write

# ╔═╡ c936aae8-2840-409e-b3a7-3dc83668fede
funchat

# ╔═╡ 8635143e-a224-488e-b6d2-94d54520c695
(m2, _) = UniLM.chat_request!(funchat)   

# ╔═╡ 7272c665-21a8-416c-a4a7-12b408f06bfb
funchat

# ╔═╡ 4aa470a5-981d-4b54-b357-2844b01d95b4


# ╔═╡ Cell order:
# ╠═9ae84052-264c-11ee-2e7d-1dc9f61fa7c8
# ╟─d24af788-59ef-4bd0-8fd1-b4cce5106f2d
# ╠═23e1b1fa-64d4-4c18-a740-8378c24d00e6
# ╠═015895ad-ddfc-40ce-b43c-515a95df0ca3
# ╠═39993fba-21a8-4515-b3af-b40f046337b5
# ╠═86151d49-73f4-48d9-87d1-02149a9d42dd
# ╠═6f6fe9e6-1b72-46ac-a6dc-527ef0f8f6fa
# ╠═3811d45f-6ec2-4190-9431-79caa560c80e
# ╠═dc80bb85-bada-4751-9228-2f1bf7c4212a
# ╠═508bec95-b761-4a01-a550-4f23ec1f1974
# ╠═15bbdad9-1f13-4b71-8a5e-7d07380cc8fa
# ╠═daf74b90-3cbe-4f62-ac27-836d01d77ce6
# ╠═c936aae8-2840-409e-b3a7-3dc83668fede
# ╠═8635143e-a224-488e-b6d2-94d54520c695
# ╠═7272c665-21a8-416c-a4a7-12b408f06bfb
# ╠═4aa470a5-981d-4b54-b357-2844b01d95b4
