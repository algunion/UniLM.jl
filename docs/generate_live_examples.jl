#!/usr/bin/env julia
# ============================================================================
# Generate Live Examples for UniLM.jl Documentation
#
# Runs real API calls and captures output to markdown snippets.
# These snippets are included in the documentation to show real results.
#
# Requirements: OPENAI_API_KEY must be set in the environment
# Usage: julia --project=docs/ docs/generate_live_examples.jl
# ============================================================================

using Pkg
Pkg.activate(joinpath(@__DIR__))
Pkg.develop(path=joinpath(@__DIR__, ".."))
Pkg.instantiate()

using UniLM
using JSON
using LinearAlgebra

const OUTPUT_DIR = joinpath(@__DIR__, "src", "_generated")
mkpath(OUTPUT_DIR)

function write_example(filename, content)
    path = joinpath(OUTPUT_DIR, filename)
    open(path, "w") do io
        write(io, content)
    end
    println("  ✓ $filename")
end

const CHAT_MODEL = "gpt-4o-mini"

println("═══════════════════════════════════════════════════════")
println("  UniLM.jl — Live Example Generation")
println("═══════════════════════════════════════════════════════\n")

# ── 1. Responses API — Basic text ────────────────────────────────────────────
println("1/16 Responses API — Basic text")
r = respond("Explain Julia's multiple dispatch in 2-3 sentences.")
if r isa ResponseSuccess
    write_example(
        "respond_basic.md",
        """
```julia
julia> result = respond("Explain Julia's multiple dispatch in 2-3 sentences.")

julia> output_text(result)
"$(escape_string(output_text(r)))"

julia> result.response.id
"$(r.response.id)"

julia> result.response.status
"$(r.response.status)"

julia> result.response.model
"$(r.response.model)"

julia> result.response.usage
$(JSON.json(r.response.usage, 2))
```
"""
    )
else
    @warn "FAILED" r
end

# ── 2. Responses API — With instructions ─────────────────────────────────────
println("2/16 Responses API — Instructions")
r = respond(
    "Translate to French: The quick brown fox jumps over the lazy dog.",
    instructions="You are a professional translator. Respond only with the translation."
)
if r isa ResponseSuccess
    write_example(
        "respond_instructions.md",
        """
```julia
julia> result = respond(
           "Translate to French: The quick brown fox jumps over the lazy dog.",
           instructions="You are a professional translator. Respond only with the translation."
       )

julia> output_text(result)
"$(escape_string(output_text(r)))"
```
"""
    )
else
    @warn "FAILED" r
end

# ── 3. Responses API — Web search ────────────────────────────────────────────
println("3/16 Responses API — Web search")
r = respond(
    "What is the latest stable release of the Julia programming language?",
    tools=[web_search()]
)
if r isa ResponseSuccess
    text = output_text(r)
    display_text = length(text) > 500 ? text[1:500] * "…" : text
    write_example(
        "respond_websearch.md",
        """
```julia
julia> result = respond(
           "What is the latest stable release of the Julia programming language?",
           tools=[web_search()]
       )

julia> output_text(result)
"$(escape_string(display_text))"
```
"""
    )
else
    @warn "FAILED" r
end

# ── 4. Responses API — Multi-turn ────────────────────────────────────────────
println("4/16 Responses API — Multi-turn")
r1 = respond("Tell me a one-liner programming joke.", instructions="Be concise.")
if r1 isa ResponseSuccess
    r2 = respond("Explain why that's funny, in one sentence.", previous_response_id=r1.response.id)
    if r2 isa ResponseSuccess
        write_example(
            "respond_multiturn.md",
            """
```julia
julia> r1 = respond("Tell me a one-liner programming joke.", instructions="Be concise.")

julia> output_text(r1)
"$(escape_string(output_text(r1)))"

julia> r2 = respond("Explain why that's funny, in one sentence.", previous_response_id=r1.response.id)

julia> output_text(r2)
"$(escape_string(output_text(r2)))"
```
"""
        )
    end
end

# ── 5. Responses API — Structured output ─────────────────────────────────────
println("5/16 Responses API — Structured output")
fmt = json_schema_format(
    "languages", "A list of programming languages",
    Dict(
        "type" => "object",
        "properties" => Dict(
            "languages" => Dict(
                "type" => "array",
                "items" => Dict(
                    "type" => "object",
                    "properties" => Dict(
                        "name" => Dict("type" => "string"),
                        "year" => Dict("type" => "integer"),
                        "paradigm" => Dict("type" => "string")
                    ),
                    "required" => ["name", "year", "paradigm"],
                    "additionalProperties" => false
                )
            )
        ),
        "required" => ["languages"],
        "additionalProperties" => false
    ),
    strict=true
)
r = respond("List Julia, Python, and Rust with their release year and primary paradigm.", text=fmt)
if r isa ResponseSuccess
    pretty = JSON.json(JSON.parse(output_text(r)), 2)
    write_example(
        "respond_structured.md",
        """
```julia
julia> fmt = json_schema_format(
           "languages", "A list of programming languages",
           Dict(
               "type" => "object",
               "properties" => Dict(
                   "languages" => Dict(
                       "type" => "array",
                       "items" => Dict(
                           "type" => "object",
                           "properties" => Dict(
                               "name" => Dict("type" => "string"),
                               "year" => Dict("type" => "integer"),
                               "paradigm" => Dict("type" => "string")
                           ),
                           "required" => ["name", "year", "paradigm"],
                           "additionalProperties" => false
                       )
                   )
               ),
               "required" => ["languages"],
               "additionalProperties" => false
           ),
           strict=true
       )

julia> result = respond(
           "List Julia, Python, and Rust with their release year and primary paradigm.",
           text=fmt
       )

julia> JSON.parse(output_text(result))
$pretty
```
"""
    )
else
    @warn "FAILED" r
end

# ── 6. Responses API — Function calling ──────────────────────────────────────
println("6/16 Responses API — Function calling")
weather_tool = function_tool(
    "get_weather",
    "Get the current weather for a given location",
    parameters=Dict(
        "type" => "object",
        "properties" => Dict(
            "location" => Dict("type" => "string", "description" => "City name"),
            "unit" => Dict("type" => "string", "enum" => ["celsius", "fahrenheit"])
        ),
        "required" => ["location", "unit"],
        "additionalProperties" => false
    ),
    strict=true
)
r = respond("What's the weather in Tokyo? Use celsius.", tools=[weather_tool])
if r isa ResponseSuccess
    calls = function_calls(r)
    if !isempty(calls)
        c = calls[1]
        args_pretty = JSON.json(JSON.parse(c["arguments"]), 2)
        write_example(
            "respond_function_call.md",
            """
```julia
julia> weather_tool = function_tool(
           "get_weather", "Get the current weather for a given location",
           parameters=Dict(
               "type" => "object",
               "properties" => Dict(
                   "location" => Dict("type" => "string", "description" => "City name"),
                   "unit" => Dict("type" => "string", "enum" => ["celsius", "fahrenheit"])
               ),
               "required" => ["location", "unit"],
               "additionalProperties" => false
           ),
           strict=true
       )

julia> result = respond("What's the weather in Tokyo? Use celsius.", tools=[weather_tool])

julia> calls = function_calls(result)

julia> calls[1]["name"]
"$(c["name"])"

julia> JSON.parse(calls[1]["arguments"])
$args_pretty
```
"""
        )
    end
else
    @warn "FAILED" r
end

# ── 7. Chat Completions — Basic ──────────────────────────────────────────────
println("7/16 Chat Completions — Basic")
chat = Chat(model=CHAT_MODEL)
push!(chat, Message(Val(:system), "You are a concise Julia programming tutor."))
push!(chat, Message(Val(:user), "What is multiple dispatch? Answer in 2-3 sentences."))
r = chatrequest!(chat)
if r isa LLMSuccess
    write_example(
        "chat_basic.md",
        """
```julia
julia> chat = Chat(model="$CHAT_MODEL")

julia> push!(chat, Message(Val(:system), "You are a concise Julia programming tutor."))

julia> push!(chat, Message(Val(:user), "What is multiple dispatch? Answer in 2-3 sentences."))

julia> result = chatrequest!(chat)

julia> result.message.content
"$(escape_string(r.message.content))"

julia> result.message.finish_reason
"$(r.message.finish_reason)"

julia> length(chat)  # system + user + assistant
$(length(chat))
```
"""
    )
else
    @warn "FAILED" r
end

# ── 8. Chat Completions — Multi-turn ─────────────────────────────────────────
println("8/16 Chat Completions — Multi-turn")
push!(chat, Message(Val(:user), "Give a short Julia code example of it."))
r = chatrequest!(chat)
if r isa LLMSuccess
    write_example(
        "chat_multiturn.md",
        """
```julia
julia> push!(chat, Message(Val(:user), "Give a short Julia code example of it."))

julia> result = chatrequest!(chat)

julia> println(result.message.content)
$(r.message.content)

julia> length(chat)  # system + user + assistant + user + assistant
$(length(chat))
```
"""
    )
else
    @warn "FAILED" r
end

# ── 9. Chat Completions — JSON output ────────────────────────────────────────
println("9/16 Chat Completions — JSON output")
json_chat = Chat(model=CHAT_MODEL, response_format=ResponseFormat())
push!(json_chat, Message(Val(:system), "You always respond in valid JSON."))
push!(json_chat, Message(Val(:user), "List 3 programming languages with their year of creation. Keys: name, year."))
r = chatrequest!(json_chat)
if r isa LLMSuccess
    pretty = JSON.json(JSON.parse(r.message.content), 2)
    write_example(
        "chat_json.md",
        """
```julia
julia> chat = Chat(model="$CHAT_MODEL", response_format=ResponseFormat())

julia> push!(chat, Message(Val(:system), "You always respond in valid JSON."))

julia> push!(chat, Message(Val(:user), "List 3 programming languages with their year of creation."))

julia> result = chatrequest!(chat)

julia> JSON.parse(result.message.content)
$pretty
```
"""
    )
else
    @warn "FAILED" r
end

# ── 10. Chat Completions — JSON Schema ───────────────────────────────────────
println("10/16 Chat Completions — JSON Schema")
schema_rf = UniLM.json_schema("weather", "Weather data", Dict(
    "type" => "object",
    "properties" => Dict(
        "location" => Dict("type" => "string"),
        "temperature" => Dict("type" => "number"),
        "unit" => Dict("type" => "string", "enum" => ["celsius", "fahrenheit"]),
        "conditions" => Dict("type" => "string")
    ),
    "required" => ["location", "temperature", "unit", "conditions"],
    "additionalProperties" => false
))
schema_chat = Chat(model=CHAT_MODEL, response_format=schema_rf)
push!(schema_chat, Message(Val(:system), "Respond with weather data in structured JSON."))
push!(schema_chat, Message(Val(:user), "What's the weather like in Tokyo?"))
r = chatrequest!(schema_chat)
if r isa LLMSuccess
    pretty = JSON.json(JSON.parse(r.message.content), 2)
    write_example(
        "chat_json_schema.md",
        """
```julia
julia> schema = UniLM.json_schema("weather", "Weather data", Dict(
           "type" => "object",
           "properties" => Dict(
               "location" => Dict("type" => "string"),
               "temperature" => Dict("type" => "number"),
               "unit" => Dict("type" => "string"),
               "conditions" => Dict("type" => "string")
           ),
           "required" => ["location", "temperature", "unit", "conditions"],
           "additionalProperties" => false
       ))

julia> chat = Chat(model="$CHAT_MODEL", response_format=schema)

julia> push!(chat, Message(Val(:system), "Respond with weather data in structured JSON."))

julia> push!(chat, Message(Val(:user), "What's the weather like in Tokyo?"))

julia> result = chatrequest!(chat)

julia> JSON.parse(result.message.content)
$pretty
```
"""
    )
else
    @warn "FAILED" r
end

# ── 11. Chat Completions — Function calling ──────────────────────────────────
println("11/16 Chat Completions — Function calling")
gptfsig = GPTFunctionSignature(
    name="get_weather",
    description="Get the current weather for a location",
    parameters=Dict(
        "type" => "object",
        "properties" => Dict(
            "location" => Dict("type" => "string", "description" => "City name"),
            "unit" => Dict("type" => "string", "enum" => ["celsius", "fahrenheit"])
        ),
        "required" => ["location"]
    )
)
tc_chat = Chat(
    model="gpt-4o",
    tools=[GPTTool(func=gptfsig)],
    tool_choice=UniLM.GPTToolChoice(func=:get_weather)
)
push!(tc_chat, Message(Val(:system), "Use the provided tools to answer."))
push!(tc_chat, Message(Val(:user), "What's the weather in Paris?"))
r = chatrequest!(tc_chat)
if r isa LLMSuccess && r.message.finish_reason == "tool_calls"
    tc = r.message.tool_calls[1]
    write_example(
        "chat_function_call.md",
        """
```julia
julia> sig = GPTFunctionSignature(
           name="get_weather",
           description="Get the current weather for a location",
           parameters=Dict(
               "type" => "object",
               "properties" => Dict(
                   "location" => Dict("type" => "string", "description" => "City name"),
                   "unit" => Dict("type" => "string", "enum" => ["celsius", "fahrenheit"])
               ),
               "required" => ["location"]
           )
       )

julia> chat = Chat(
           model="$CHAT_MODEL",
           tools=[GPTTool(func=sig)],
           tool_choice=UniLM.GPTToolChoice(func=:get_weather)
       )

julia> push!(chat, Message(Val(:system), "Use the provided tools to answer."))

julia> push!(chat, Message(Val(:user), "What's the weather in Paris?"))

julia> result = chatrequest!(chat)

julia> result.message.finish_reason
"tool_calls"

julia> tc = result.message.tool_calls[1]

julia> tc.func.name
"$(tc.func.name)"

julia> tc.func.arguments
$(JSON.json(tc.func.arguments, 2))
```
"""
    )
else
    @warn "FAILED or unexpected" r
end

# ── 12. Chat — Keyword arguments ─────────────────────────────────────────────
println("12/16 Chat Completions — Keywords")
r = chatrequest!(
    systemprompt="You are a calculator. Respond only with the number.",
    userprompt="What is 42 * 17?",
    model=CHAT_MODEL,
    temperature=0.0
)
if r isa LLMSuccess
    write_example(
        "chat_kwargs.md",
        """
```julia
julia> result = chatrequest!(
           systemprompt="You are a calculator. Respond only with the number.",
           userprompt="What is 42 * 17?",
           model="$CHAT_MODEL",
           temperature=0.0
       )

julia> result.message.content
"$(escape_string(r.message.content))"
```
"""
    )
else
    @warn "FAILED" r
end

# ── 13. Embeddings ───────────────────────────────────────────────────────────
println("13/16 Embeddings")
emb = Embeddings("Julia is a high-performance programming language for technical computing.")
embeddingrequest!(emb)
first5 = round.(emb.embeddings[1:5], digits=6)
norm_val = round(sqrt(sum(x^2 for x in emb.embeddings)), digits=4)
write_example(
    "embeddings.md",
    """
```julia
julia> emb = Embeddings("Julia is a high-performance programming language for technical computing.")

julia> embeddingrequest!(emb)

julia> emb.embeddings[1:5]  # first 5 dimensions
5-element Vector{Float64}:
$(join(["  " * string(v) for v in first5], "\n"))

julia> sqrt(sum(x^2 for x in emb.embeddings))  # L2 norm ≈ 1.0
$norm_val
```
"""
)

# ── 14. Image Generation ────────────────────────────────────────────────────
println("14/16 Image Generation")
r = generate_image(
    "A watercolor painting of a friendly robot reading a Julia programming book, soft pastel colors, whimsical style",
    size="1024x1024",
    quality="medium"
)
if r isa ImageSuccess
    img_dir = joinpath(@__DIR__, "src", "assets")
    mkpath(img_dir)
    img_path = joinpath(img_dir, "generated_robot.png")
    save_image(image_data(r)[1], img_path)

    n_images = length(r.response.data)
    b64_len = length(image_data(r)[1])
    revised = r.response.data[1].revised_prompt
    usage = r.response.usage

    write_example(
        "image_generation.md",
        """
```julia
julia> result = generate_image(
           "A watercolor painting of a friendly robot reading a Julia programming book",
           size="1024x1024",
           quality="medium"
       )

julia> result isa ImageSuccess
true

julia> length(image_data(result))
$n_images

julia> length(image_data(result)[1])  # base64 string length
$b64_len

julia> result.response.data[1].revised_prompt
$(isnothing(revised) ? "nothing" : "\"$(escape_string(string(revised)))\"")

julia> result.response.usage
$(isnothing(usage) ? "nothing" : JSON.json(usage, 2))

julia> save_image(image_data(result)[1], "robot_julia.png")
"robot_julia.png"
```

**Generated image:**

![A watercolor painting of a friendly robot reading a Julia programming book](assets/generated_robot.png)
"""
    )
    println("  ✓ Image saved to docs/src/assets/generated_robot.png")
else
    @warn "FAILED" r
end

# ── 15. Responses API — Streaming ────────────────────────────────────────────
println("15/16 Responses API — Streaming")
task = respond("Write a haiku about Julia programming.") do chunk, close
end
r = fetch(task)
if r isa ResponseSuccess
    text = output_text(r)
    write_example(
        "respond_streaming.md",
        """
```julia
julia> task = respond("Write a haiku about Julia programming.") do chunk, close
           if chunk isa String
               print(chunk)  # tokens stream in real-time
           elseif chunk isa ResponseObject
               println("\\nDone! Status: ", chunk.status)
           end
       end
$(text)
Done! Status: completed

julia> result = fetch(task)

julia> output_text(result)
"$(escape_string(text))"
```
"""
    )
else
    @warn "FAILED" r
end

# ── 16. Chat Completions — Streaming ─────────────────────────────────────────
println("16/16 Chat Completions — Streaming")
stream_chat = Chat(model=CHAT_MODEL, stream=true)
push!(stream_chat, Message(Val(:system), "You are a poet."))
push!(stream_chat, Message(Val(:user), "Write a very short 2-line poem about coding."))
try
    task = chatrequest!(stream_chat, callback=function (chunk, close)
    end)
    result = fetch(task)
    if result isa LLMSuccess
        msg = result.message
        write_example(
            "chat_streaming.md",
            """
```julia
julia> chat = Chat(model="$CHAT_MODEL", stream=true)

julia> push!(chat, Message(Val(:system), "You are a poet."))

julia> push!(chat, Message(Val(:user), "Write a very short 2-line poem about coding."))

julia> task = chatrequest!(chat, callback=function(chunk, close)
           if chunk isa String
               print(chunk)  # tokens stream in real-time
           elseif chunk isa Message
               println("\\n--- done ---")
           end
       end)
$(msg.content)
--- done ---

julia> result = fetch(task)

julia> result.message.content
"$(escape_string(msg.content))"
```
"""
        )
    end
catch e
    @warn "Streaming failed" e
end

# ── Summary ──────────────────────────────────────────────────────────────────
println("\n═══════════════════════════════════════════════════════")
generated = sort(readdir(OUTPUT_DIR))
println("  Generated $(length(generated)) files in docs/src/_generated/")
for f in generated
    println("    • $f")
end
println("═══════════════════════════════════════════════════════")
