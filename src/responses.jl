# ============================================================================
# OpenAI Responses API
# https://platform.openai.com/docs/api-reference/responses
#
# The Responses API is a newer, more flexible alternative to Chat Completions.
# Key differences:
#   - `input` instead of `messages` (string or structured items)
#   - `instructions` instead of system message
#   - `previous_response_id` for multi-turn conversations (stateful or stateless)
#   - Built-in tools: web_search, file_search, code_interpreter
#   - Function tools have `name` at top-level (not nested under `function`)
#   - `reasoning` parameter for O-series models
#   - `text.format` for output formatting
# ============================================================================


# ─── Input Types ──────────────────────────────────────────────────────────────

"""
    InputMessage(; role, content)

A structured input message for the Responses API.

# Fields
- `role::String`: `"user"`, `"assistant"`, `"system"`, or `"developer"`
- `content::Any`: String or a vector of content parts (see `input_text`, `input_image`, `input_file`)

# Examples
```julia
InputMessage(role="user", content="What is 2+2?")
InputMessage(role="user", content=[input_text("Describe this:"), input_image("https://example.com/img.png")])
```
"""
@kwdef struct InputMessage
    role::String
    content::Any
end

JSON.lower(m::InputMessage) = Dict{Symbol,Any}(:role => m.role, :content => m.content)

"""
    input_text(text::String)

Create an `input_text` content part for multimodal input messages.
"""
input_text(text::String) = Dict{Symbol,Any}(:type => "input_text", :text => text)

"""
    input_image(url::String; detail=nothing)

Create an `input_image` content part. `detail` can be `"auto"`, `"low"`, or `"high"`.
"""
function input_image(url::String; detail::Union{String,Nothing}=nothing)
    d = Dict{Symbol,Any}(:type => "input_image", :image_url => url)
    !isnothing(detail) && (d[:detail] = detail)
    return d
end

"""
    input_file(; url=nothing, id=nothing)

Create an `input_file` content part. Provide either a `url` or a `file_id`.
"""
function input_file(; url::Union{String,Nothing}=nothing, id::Union{String,Nothing}=nothing)
    isnothing(url) && isnothing(id) && throw(ArgumentError("Either `url` or `id` must be provided"))
    d = Dict{Symbol,Any}(:type => "input_file")
    !isnothing(url) && (d[:file_url] = url)
    !isnothing(id) && (d[:file_id] = id)
    return d
end


# ─── Tool Types ───────────────────────────────────────────────────────────────

"""
    ResponseTool

Abstract supertype for Responses API tools. Subtypes:
- [`FunctionTool`](@ref)
- [`WebSearchTool`](@ref)
- [`FileSearchTool`](@ref)
"""
abstract type ResponseTool end

"""
    FunctionTool(; name, description=nothing, parameters=nothing, strict=nothing)

A function tool for the Responses API.

# Examples
```julia
FunctionTool(
    name="get_weather",
    description="Get current weather for a location",
    parameters=Dict(
        "type" => "object",
        "properties" => Dict(
            "location" => Dict("type" => "string", "description" => "City name")
        ),
        "required" => ["location"]
    )
)
```
"""
@kwdef struct FunctionTool <: ResponseTool
    name::String
    description::Union{String,Nothing} = nothing
    parameters::Union{AbstractDict,Nothing} = nothing
    strict::Union{Bool,Nothing} = nothing
end

function JSON.lower(t::FunctionTool)
    d = Dict{Symbol,Any}(:type => "function", :name => t.name)
    !isnothing(t.description) && (d[:description] = t.description)
    !isnothing(t.parameters) && (d[:parameters] = t.parameters)
    !isnothing(t.strict) && (d[:strict] = t.strict)
    return d
end

"""
    WebSearchTool(; search_context_size="medium", user_location=nothing)

A web search tool for the Responses API. Allows the model to search the web.

- `search_context_size`: `"low"`, `"medium"`, or `"high"`
- `user_location`: Dict with keys like `"country"`, `"city"`, `"region"`, `"timezone"`
"""
@kwdef struct WebSearchTool <: ResponseTool
    search_context_size::String = "medium"
    user_location::Union{AbstractDict,Nothing} = nothing
end

function JSON.lower(t::WebSearchTool)
    d = Dict{Symbol,Any}(:type => "web_search_preview", :search_context_size => t.search_context_size)
    !isnothing(t.user_location) && (d[:user_location] = t.user_location)
    return d
end

"""
    FileSearchTool(; vector_store_ids, max_num_results=nothing, ranking_options=nothing, filters=nothing)

A file search tool for the Responses API. Searches over uploaded vector stores.
"""
@kwdef struct FileSearchTool <: ResponseTool
    vector_store_ids::Vector{String}
    max_num_results::Union{Int,Nothing} = nothing
    ranking_options::Union{AbstractDict,Nothing} = nothing
    filters::Union{AbstractDict,Nothing} = nothing
end

function JSON.lower(t::FileSearchTool)
    d = Dict{Symbol,Any}(:type => "file_search", :vector_store_ids => t.vector_store_ids)
    !isnothing(t.max_num_results) && (d[:max_num_results] = t.max_num_results)
    !isnothing(t.ranking_options) && (d[:ranking_options] = t.ranking_options)
    !isnothing(t.filters) && (d[:filters] = t.filters)
    return d
end

# Convenience constructors

"""
    function_tool(name, description=nothing; parameters=nothing, strict=nothing)

Shorthand constructor for [`FunctionTool`](@ref).
"""
function_tool(name::String, description::Union{String,Nothing}=nothing;
    parameters::Union{AbstractDict,Nothing}=nothing,
    strict::Union{Bool,Nothing}=nothing) =
    FunctionTool(name=name, description=description, parameters=parameters, strict=strict)

"""
    web_search(; context_size="medium", location=nothing)

Shorthand constructor for [`WebSearchTool`](@ref).
"""
web_search(; context_size::String="medium",
    location::Union{AbstractDict,Nothing}=nothing) =
    WebSearchTool(search_context_size=context_size, user_location=location)

"""
    file_search(store_ids::Vector{String}; max_results=nothing, ranking=nothing, filters=nothing)

Shorthand constructor for [`FileSearchTool`](@ref).
"""
file_search(store_ids::Vector{String};
    max_results::Union{Int,Nothing}=nothing,
    ranking::Union{AbstractDict,Nothing}=nothing,
    filters::Union{AbstractDict,Nothing}=nothing) =
    FileSearchTool(vector_store_ids=store_ids, max_num_results=max_results,
        ranking_options=ranking, filters=filters)


# ─── Configuration Types ─────────────────────────────────────────────────────

"""
    TextFormatSpec(; type="text", name=nothing, description=nothing, schema=nothing, strict=nothing)

Output text format specification.

- `type`: `"text"` (default), `"json_object"`, or `"json_schema"`
- For `"json_schema"`: provide `name`, `description`, `schema`, and optionally `strict`
"""
@kwdef struct TextFormatSpec
    type::String = "text"
    name::Union{String,Nothing} = nothing
    description::Union{String,Nothing} = nothing
    schema::Union{AbstractDict,Nothing} = nothing
    strict::Union{Bool,Nothing} = nothing
end

function JSON.lower(t::TextFormatSpec)
    d = Dict{Symbol,Any}(:type => t.type)
    !isnothing(t.name) && (d[:name] = t.name)
    !isnothing(t.description) && (d[:description] = t.description)
    !isnothing(t.schema) && (d[:schema] = t.schema)
    !isnothing(t.strict) && (d[:strict] = t.strict)
    return d
end

"""
    TextConfig(; format=TextFormatSpec())

Wrapper for the `text` field in the Responses API request body.
"""
@kwdef struct TextConfig
    format::TextFormatSpec = TextFormatSpec()
end

# Convenience constructors

"""
    text_format(; kwargs...)

Create a [`TextConfig`](@ref) with the given format options.
"""
text_format(; kwargs...) = TextConfig(format=TextFormatSpec(; kwargs...))

"""
    json_schema_format(name, description, schema; strict=nothing)

Create a JSON Schema output format for structured output.
"""
json_schema_format(name::String, description::String, schema::AbstractDict;
    strict::Union{Bool,Nothing}=nothing) =
    TextConfig(format=TextFormatSpec(type="json_schema", name=name, description=description, schema=schema, strict=strict))

"""
    json_object_format()

Create a JSON object output format (unstructured).
"""
json_object_format() = TextConfig(format=TextFormatSpec(type="json_object"))


"""
    Reasoning(; effort=nothing, summary=nothing)

Reasoning configuration for O-series models (o3, o4-mini, etc.).

- `effort`: `"low"`, `"medium"`, or `"high"`
- `summary`: `"auto"`, `"concise"`, or `"detailed"`
"""
@kwdef struct Reasoning
    effort::Union{String,Nothing} = nothing
    summary::Union{String,Nothing} = nothing
end

function JSON.lower(r::Reasoning)
    d = Dict{Symbol,Any}()
    !isnothing(r.effort) && (d[:effort] = r.effort)
    !isnothing(r.summary) && (d[:summary] = r.summary)
    return d
end


# ─── Main Request Type ────────────────────────────────────────────────────────

"""
    Respond(; model="gpt-5.2", input, kwargs...)

Configuration struct for an OpenAI Responses API request.

# Key Fields
- `model::String`: Model to use (default: `"gpt-5.2"`)
- `input::Any`: A `String` or `Vector{InputMessage}` — the prompt input
- `instructions::String`: System-level instructions
- `tools::Vector`: Available tools (`FunctionTool`, `WebSearchTool`, `FileSearchTool`)
- `previous_response_id::String`: Chain to a previous response for multi-turn
- `reasoning::Reasoning`: Reasoning config for O-series models
- `text::TextConfig`: Output format (text, json, json_schema)
- `temperature::Float64`: Sampling temperature (0.0–2.0), mutually exclusive with `top_p`
- `top_p::Float64`: Nucleus sampling (0.0–1.0), mutually exclusive with `temperature`
- `max_output_tokens::Int64`: Max tokens in the response
- `stream::Bool`: Enable streaming
- `truncation::String`: `"auto"` or `"disabled"`
- `store::Bool`: Whether to store the response for later retrieval
- `metadata::AbstractDict`: Arbitrary metadata to attach
- `user::String`: End-user identifier

# Examples
```julia
# Simple text
Respond(input="Tell me a joke")

# With instructions
Respond(input="Translate to French: Hello", instructions="You are a translator")

# Multi-turn via chaining
Respond(input="Tell me more", previous_response_id="resp_abc123")

# With tools
Respond(
    input="What's the weather in NYC?",
    tools=ResponseTool[function_tool("get_weather", "Get weather", parameters=Dict("type"=>"object", "properties"=>Dict("location"=>Dict("type"=>"string"))))]
)

# Reasoning (O-series models)
Respond(input="Solve this math problem...", model="o3", reasoning=Reasoning(effort="high"))
```
"""
@kwdef struct Respond
    service::Type{<:ServiceEndpoint} = OPENAIServiceEndpoint
    model::String = "gpt-5.2"
    input::Any  # String or Vector{InputMessage}
    instructions::Union{String,Nothing} = nothing
    tools::Union{Vector,Nothing} = nothing
    tool_choice::Union{String,Nothing} = nothing      # "auto", "none", "required"
    parallel_tool_calls::Union{Bool,Nothing} = nothing
    temperature::Union{Float64,Nothing} = nothing
    top_p::Union{Float64,Nothing} = nothing
    max_output_tokens::Union{Int64,Nothing} = nothing
    stream::Union{Bool,Nothing} = nothing
    text::Union{TextConfig,Nothing} = nothing
    reasoning::Union{Reasoning,Nothing} = nothing
    truncation::Union{String,Nothing} = nothing        # "auto", "disabled"
    store::Union{Bool,Nothing} = nothing
    metadata::Union{AbstractDict,Nothing} = nothing
    previous_response_id::Union{String,Nothing} = nothing
    user::Union{String,Nothing} = nothing
    function Respond(service, model, input, instructions, tools, tool_choice,
        parallel_tool_calls, temperature, top_p, max_output_tokens,
        stream, text, reasoning, truncation, store, metadata,
        previous_response_id, user)
        !isnothing(temperature) && !isnothing(top_p) && throw(ArgumentError("temperature and top_p are mutually exclusive"))
        new(service, model, input, instructions, tools, tool_choice,
            parallel_tool_calls, temperature, top_p, max_output_tokens,
            stream, text, reasoning, truncation, store, metadata,
            previous_response_id, user)
    end
end

function JSON.lower(r::Respond)
    d = Dict{Symbol,Any}(:model => r.model, :input => r.input)
    for f in (:instructions, :tools, :tool_choice, :parallel_tool_calls,
        :temperature, :top_p, :max_output_tokens, :stream, :text,
        :reasoning, :truncation, :store, :metadata, :previous_response_id, :user)
        v = getfield(r, f)
        !isnothing(v) && (d[f] = v)
    end
    return d
end


# ─── Response Object ─────────────────────────────────────────────────────────

"""
    ResponseObject

Parsed response from the Responses API.

# Accessors
- `output_text(r)` — extract concatenated text output
- `function_calls(r)` — extract function call outputs
- `r.id`, `r.status`, `r.model` — basic metadata
- `r.output` — full output array (raw dicts)
- `r.usage` — token usage info
- `r.raw` — the complete raw JSON dict
"""
@kwdef struct ResponseObject
    id::String
    status::String
    model::String
    output::Vector{Any}
    usage::Union{Dict{String,Any},Nothing} = nothing
    error::Union{Any,Nothing} = nothing
    metadata::Union{Dict{String,Any},Nothing} = nothing
    raw::Dict{String,Any}
end


# ─── Result Types ─────────────────────────────────────────────────────────────

"""
    ResponseSuccess <: LLMRequestResponse

Successful response from the Responses API. Access the parsed response via `.response`.
"""
@kwdef struct ResponseSuccess <: LLMRequestResponse
    response::ResponseObject
end

"""
    ResponseFailure <: LLMRequestResponse

HTTP-level failure from the Responses API. Contains the response body and status code.
"""
@kwdef struct ResponseFailure <: LLMRequestResponse
    response::String
    status::Int
end

"""
    ResponseCallError <: LLMRequestResponse

Exception-level error during a Responses API call (network, parsing, etc.).
"""
@kwdef struct ResponseCallError <: LLMRequestResponse
    error::String
    status::Union{Int,Nothing} = nothing
end


# ─── Accessor Functions ──────────────────────────────────────────────────────

"""
    output_text(r::ResponseObject)::String
    output_text(r::ResponseSuccess)::String

Extract the concatenated text output from a response.

# Examples
```julia
result = respond("Hello!")
output_text(result)  # => "Hi there! How can I help?"
```
"""
function output_text(r::ResponseObject)::String
    texts = String[]
    for item in r.output
        if item isa Dict && get(item, "type", "") == "message"
            for content in get(item, "content", [])
                if content isa Dict && get(content, "type", "") == "output_text"
                    push!(texts, get(content, "text", ""))
                end
            end
        end
    end
    return join(texts, "\n")
end

output_text(r::ResponseSuccess) = output_text(r.response)
output_text(r::ResponseFailure) = "Error (HTTP $(r.status)): $(r.response)"
output_text(r::ResponseCallError) = "Error: $(r.error)"

"""
    function_calls(r::ResponseObject)::Vector{Dict{String,Any}}
    function_calls(r::ResponseSuccess)::Vector{Dict{String,Any}}

Extract function call outputs from a response.

Each dict contains: `"id"`, `"call_id"`, `"name"`, `"arguments"` (JSON string), `"status"`.

# Examples
```julia
result = respond("What's the weather?", tools=[function_tool("get_weather", ...)])
for call in function_calls(result)
    name = call["name"]
    args = JSON.parse(call["arguments"])
    # dispatch to your function...
end
```
"""
function function_calls(r::ResponseObject)
    calls = Dict{String,Any}[]
    for item in r.output
        if item isa Dict && get(item, "type", "") == "function_call"
            push!(calls, item)
        end
    end
    return calls
end

function_calls(r::ResponseSuccess) = function_calls(r.response)
function_calls(::ResponseFailure) = Dict{String,Any}[]
function_calls(::ResponseCallError) = Dict{String,Any}[]


# ─── Parsing ─────────────────────────────────────────────────────────────────

function parse_response(resp::HTTP.Response)::ResponseObject
    data = JSON.parse(resp.body; dicttype=Dict{String,Any})
    ResponseObject(
        id=data["id"],
        status=data["status"],
        model=data["model"],
        output=get(data, "output", Any[]),
        usage=get(data, "usage", nothing),
        error=get(data, "error", nothing),
        metadata=get(data, "metadata", nothing),
        raw=data
    )
end


# ─── Streaming ───────────────────────────────────────────────────────────────

function _parse_response_stream_chunk(chunk::String, textbuff::IOBuffer, failbuff::IOBuffer)
    lines = strip.(split(chunk, "\n"))
    lines = filter(!isempty, lines)
    isempty(lines) && return (; done=false, event="", data=nothing)

    last_event = ""
    for line in lines
        if startswith(line, "event: ")
            last_event = strip(line[8:end])
        elseif startswith(line, "data: ")
            try
                payload = JSON.parse(line[7:end]; dicttype=Dict{String,Any})
                if last_event == "response.output_text.delta"
                    delta = get(payload, "delta", "")
                    print(textbuff, delta)
                elseif last_event == "response.completed"
                    return (; done=true, event=last_event, data=payload)
                end
            catch e
                print(failbuff, line)
                continue
            end
        end
    end
    return (; done=false, event=last_event, data=nothing)
end

function _respond_stream(r::Respond, body::String, callback=nothing)
    Threads.@spawn begin
        try
            result = Ref{Union{ResponseObject,Nothing}}(nothing)
            url = OPENAI_BASE_URL * RESPONSES_PATH
            resp = HTTP.open("POST", url, auth_header(r.service)) do io
                text_buffer = IOBuffer()
                fail_buffer = IOBuffer()
                done = Ref(false)
                close_ref = Ref(false)
                write(io, body)
                HTTP.closewrite(io)
                HTTP.startread(io)
                while !eof(io) && !close_ref[] && !done[]
                    chunk = join((String(take!(fail_buffer)), String(readavailable(io))))
                    status = _parse_response_stream_chunk(chunk, text_buffer, fail_buffer)
                    if status.done && !isnothing(status.data)
                        rdata = status.data["response"]
                        result[] = ResponseObject(
                            id=rdata["id"],
                            status=rdata["status"],
                            model=rdata["model"],
                            output=get(rdata, "output", Any[]),
                            usage=get(rdata, "usage", nothing),
                            error=get(rdata, "error", nothing),
                            metadata=get(rdata, "metadata", nothing),
                            raw=rdata
                        )
                        done[] = true
                        !isnothing(callback) && callback(result[], close_ref)
                    else
                        parsed_text = String(take!(text_buffer))
                        if !isempty(parsed_text) && !isnothing(callback)
                            callback(parsed_text, close_ref)
                        end
                    end
                end
                close_ref[] && @info "Response stream closed by user"
                HTTP.closeread(io)
            end
            if resp.status == 200 && !isnothing(result[])
                ResponseSuccess(response=result[]::ResponseObject)
            else
                ResponseFailure(response=String(resp.body), status=resp.status)
            end
        catch e
            statuserror = hasproperty(e, :status) ? e.status : nothing
            ResponseCallError(error=string(e), status=statuserror)
        end
    end
end


# ─── Request Functions ───────────────────────────────────────────────────────

"""
    respond(r::Respond; retries=0, callback=nothing)

Send a request to the OpenAI Responses API.

Returns `ResponseSuccess`, `ResponseFailure`, or `ResponseCallError`.

For streaming, set `stream=true` and pass a `callback`:
```julia
callback(chunk::Union{String, ResponseObject}, close::Ref{Bool})
```

# Examples
```julia
r = Respond(input="Tell me a joke")
result = respond(r)
if result isa ResponseSuccess
    println(output_text(result))
end
```
"""
function respond(r::Respond; retries::Int=0, callback=nothing)
    res = ResponseCallError(error="uninitialized", status=0)
    try
        body = JSON.json(r)

        # Streaming path
        if !isnothing(r.stream) && r.stream
            return _respond_stream(r, body, callback)
        end

        url = OPENAI_BASE_URL * RESPONSES_PATH
        resp = HTTP.post(url, body=body, headers=auth_header(r.service))

        if resp.status == 200
            return ResponseSuccess(response=parse_response(resp))
        elseif resp.status in (500, 503)
            @warn "Request status: $(resp.status). Retrying in 1s..."
            sleep(1)
            if retries < 30
                return respond(r; retries=retries + 1, callback=callback)
            else
                return ResponseFailure(response=String(resp.body), status=resp.status)
            end
        else
            return ResponseFailure(response=String(resp.body), status=resp.status)
        end
    catch e
        statuserror = hasproperty(e, :status) ? e.status : nothing
        res = ResponseCallError(error=string(e), status=statuserror)
    end
    return res
end

"""
    respond(input; kwargs...)

Convenience method: create a [`Respond`](@ref) from `input` + keyword arguments and send it.

# Examples
```julia
# Simple text
result = respond("Tell me a joke")

# With instructions and model
result = respond("Translate: Hello", instructions="You are a translator", model="gpt-5.2")

# With tools
result = respond("Search for Julia news", tools=[web_search()])

# Multi-turn
r1 = respond("Tell me a joke")
r2 = respond("Tell me another", previous_response_id=r1.response.id)

# Streaming
respond("Tell me a story", stream=true) do chunk, close
    if chunk isa String
        print(chunk)  # partial text delta
    end
end
```
"""
function respond(input; kwargs...)
    kws = Dict{Symbol,Any}(kwargs)
    callback = pop!(kws, :callback, nothing)
    retries = pop!(kws, :retries, 0)
    respond(Respond(; input=input, kws...); retries=retries, callback=callback)
end

"""
    respond(callback::Function, input; kwargs...)

`do`-block form for streaming. Automatically sets `stream=true`.

# Examples
```julia
respond("Tell me a story") do chunk, close
    if chunk isa String
        print(chunk)
    elseif chunk isa ResponseObject
        println("\\nDone! Status: ", chunk.status)
    end
end
```
"""
function respond(callback::Function, input; kwargs...)
    respond(input; stream=true, callback=callback, kwargs...)
end


"""
    get_response(response_id::String; service=OPENAIServiceEndpoint)

Retrieve an existing response by its ID.

# Examples
```julia
result = get_response("resp_abc123")
if result isa ResponseSuccess
    println(output_text(result))
end
```
"""
function get_response(response_id::String; service::Type{<:ServiceEndpoint}=OPENAIServiceEndpoint)
    try
        url = OPENAI_BASE_URL * RESPONSES_PATH * "/" * response_id
        resp = HTTP.get(url, headers=auth_header(service))
        if resp.status == 200
            return ResponseSuccess(response=parse_response(resp))
        else
            return ResponseFailure(response=String(resp.body), status=resp.status)
        end
    catch e
        statuserror = hasproperty(e, :status) ? e.status : nothing
        return ResponseCallError(error=string(e), status=statuserror)
    end
end

"""
    delete_response(response_id::String; service=OPENAIServiceEndpoint)

Delete a stored response by its ID. Returns a Dict with `"id"`, `"object"`, `"deleted"` keys.

# Examples
```julia
result = delete_response("resp_abc123")
result["deleted"]  # => true
```
"""
function delete_response(response_id::String; service::Type{<:ServiceEndpoint}=OPENAIServiceEndpoint)
    try
        url = OPENAI_BASE_URL * RESPONSES_PATH * "/" * response_id
        resp = HTTP.request("DELETE", url, headers=auth_header(service))
        if resp.status == 200
            return JSON.parse(resp.body; dicttype=Dict{String,Any})
        else
            return ResponseFailure(response=String(resp.body), status=resp.status)
        end
    catch e
        statuserror = hasproperty(e, :status) ? e.status : nothing
        return ResponseCallError(error=string(e), status=statuserror)
    end
end

"""
    list_input_items(response_id::String; limit=20, order="desc", after=nothing, service=OPENAIServiceEndpoint)

List input items for a stored response. Returns a Dict with `"data"`, `"first_id"`, `"last_id"`, `"has_more"`.

# Examples
```julia
items = list_input_items("resp_abc123")
for item in items["data"]
    println(item["type"], ": ", item)
end
```
"""
function list_input_items(response_id::String;
    limit::Int=20,
    order::String="desc",
    after::Union{String,Nothing}=nothing,
    service::Type{<:ServiceEndpoint}=OPENAIServiceEndpoint)

    try
        url = OPENAI_BASE_URL * RESPONSES_PATH * "/" * response_id * "/input_items"
        params = ["limit=$limit", "order=$order"]
        !isnothing(after) && push!(params, "after=$after")
        url *= "?" * join(params, "&")

        resp = HTTP.get(url, headers=auth_header(service))
        if resp.status == 200
            return JSON.parse(resp.body; dicttype=Dict{String,Any})
        else
            return ResponseFailure(response=String(resp.body), status=resp.status)
        end
    catch e
        statuserror = hasproperty(e, :status) ? e.status : nothing
        return ResponseCallError(error=string(e), status=statuserror)
    end
end
