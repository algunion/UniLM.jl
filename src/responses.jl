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
    input_image(url=nothing; detail=nothing, file_id=nothing)

Create an `input_image` content part. Provide either an image `url` or a `file_id`.
`detail` can be `"auto"`, `"low"`, or `"high"`.
"""
function input_image(url::Union{String,Nothing}=nothing; detail::Union{String,Nothing}=nothing,
        file_id::Union{String,Nothing}=nothing)
    isnothing(url) && isnothing(file_id) && throw(ArgumentError("Either `url` or `file_id` must be provided"))
    d = Dict{Symbol,Any}(:type => "input_image")
    !isnothing(url) && (d[:image_url] = url)
    !isnothing(file_id) && (d[:file_id] = file_id)
    !isnothing(detail) && (d[:detail] = detail)
    return d
end

"""
    input_file(; url=nothing, id=nothing, file_data=nothing, filename=nothing)

Create an `input_file` content part. Provide a `url`, a file `id`, or inline `file_data`
(base64). `filename` is recommended when passing `file_data`.
"""
function input_file(; url::Union{String,Nothing}=nothing, id::Union{String,Nothing}=nothing,
        file_data::Union{String,Nothing}=nothing, filename::Union{String,Nothing}=nothing)
    isnothing(url) && isnothing(id) && isnothing(file_data) &&
        throw(ArgumentError("One of `url`, `id`, or `file_data` must be provided"))
    d = Dict{Symbol,Any}(:type => "input_file")
    !isnothing(url) && (d[:file_url] = url)
    !isnothing(id) && (d[:file_id] = id)
    !isnothing(file_data) && (d[:file_data] = file_data)
    !isnothing(filename) && (d[:filename] = filename)
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
    WebSearchTool(; type="web_search", search_context_size="medium", user_location=nothing, filters=nothing)

A web search tool for the Responses API. Allows the model to search the web.

- `type`: `"web_search"` (GA, default) or the legacy `"web_search_preview"`
- `search_context_size`: `"low"`, `"medium"`, or `"high"`
- `user_location`: Dict with keys like `"country"`, `"city"`, `"region"`, `"timezone"`
- `filters`: Dict with `"allowed_domains"` / `"blocked_domains"` (GA only)

Fetch sources/results back via `include=["web_search_call.results", "web_search_call.action.sources"]`.
"""
@kwdef struct WebSearchTool <: ResponseTool
    type::String = "web_search"
    search_context_size::String = "medium"
    user_location::Union{AbstractDict,Nothing} = nothing
    filters::Union{AbstractDict,Nothing} = nothing
end

function JSON.lower(t::WebSearchTool)
    d = Dict{Symbol,Any}(:type => t.type, :search_context_size => t.search_context_size)
    !isnothing(t.user_location) && (d[:user_location] = t.user_location)
    !isnothing(t.filters) && (d[:filters] = t.filters)
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

"""
    MCPTool(; server_label, server_url=nothing, connector_id=nothing, authorization=nothing,
            server_description=nothing, require_approval="never", allowed_tools=nothing,
            headers=nothing, tunnel_id=nothing)

A Model Context Protocol (MCP) tool for the Responses API. Connect the model to a remote
MCP server (`server_url`), an OpenAI connector (`connector_id`, e.g. `"connector_googledrive"`,
`"connector_gmail"`, `"connector_dropbox"`), or a Secure MCP Tunnel (`tunnel_id`). Use
`authorization` for an OAuth access token. `require_approval`/`allowed_tools` accept the
string or object forms.
"""
@kwdef struct MCPTool <: ResponseTool
    server_label::String
    server_url::Union{String, Nothing} = nothing
    connector_id::Union{String, Nothing} = nothing
    authorization::Union{String, Nothing} = nothing
    server_description::Union{String, Nothing} = nothing
    require_approval::Union{String, AbstractDict, Nothing} = "never"
    allowed_tools::Union{Vector{String}, AbstractDict, Nothing} = nothing
    headers::Union{AbstractDict, Nothing} = nothing
    tunnel_id::Union{String, Nothing} = nothing
end

function JSON.lower(t::MCPTool)
    d = Dict{Symbol,Any}(:type => "mcp", :server_label => t.server_label)
    !isnothing(t.server_url) && (d[:server_url] = t.server_url)
    !isnothing(t.connector_id) && (d[:connector_id] = t.connector_id)
    !isnothing(t.authorization) && (d[:authorization] = t.authorization)
    !isnothing(t.server_description) && (d[:server_description] = t.server_description)
    !isnothing(t.require_approval) && (d[:require_approval] = t.require_approval)
    !isnothing(t.allowed_tools) && (d[:allowed_tools] = t.allowed_tools)
    !isnothing(t.headers) && (d[:headers] = t.headers)
    !isnothing(t.tunnel_id) && (d[:tunnel_id] = t.tunnel_id)
    return d
end

"""
    ComputerUseTool(; display_width=1024, display_height=768, environment=nothing)

A computer use tool for the Responses API. Allows the model to interact with a
virtual display via screenshots, mouse, and keyboard.
"""
@kwdef struct ComputerUseTool <: ResponseTool
    display_width::Int = 1024
    display_height::Int = 768
    environment::Union{String, Nothing} = nothing
end

function JSON.lower(t::ComputerUseTool)
    d = Dict{Symbol,Any}(:type => "computer_use_preview", :display_width => t.display_width, :display_height => t.display_height)
    !isnothing(t.environment) && (d[:environment] = t.environment)
    return d
end

"""
    ImageGenerationTool(; background=nothing, output_format=nothing, output_compression=nothing, quality=nothing, size=nothing)

An image generation tool for the Responses API. Allows the model to generate
images inline during a response.
"""
@kwdef struct ImageGenerationTool <: ResponseTool
    background::Union{String, Nothing} = nothing
    output_format::Union{String, Nothing} = nothing
    output_compression::Union{Int, Nothing} = nothing
    quality::Union{String, Nothing} = nothing
    size::Union{String, Nothing} = nothing
end

function JSON.lower(t::ImageGenerationTool)
    d = Dict{Symbol,Any}(:type => "image_generation")
    !isnothing(t.background) && (d[:background] = t.background)
    !isnothing(t.output_format) && (d[:output_format] = t.output_format)
    !isnothing(t.output_compression) && (d[:output_compression] = t.output_compression)
    !isnothing(t.quality) && (d[:quality] = t.quality)
    !isnothing(t.size) && (d[:size] = t.size)
    return d
end

"""
    CodeInterpreterTool(; container=nothing, file_ids=nothing)

A code interpreter tool for the Responses API. Allows the model to execute
code in a sandboxed environment.
"""
@kwdef struct CodeInterpreterTool <: ResponseTool
    container::Union{AbstractDict, Nothing} = nothing
    file_ids::Union{Vector{String}, Nothing} = nothing
end

function JSON.lower(t::CodeInterpreterTool)
    d = Dict{Symbol,Any}(:type => "code_interpreter")
    !isnothing(t.container) && (d[:container] = t.container)
    !isnothing(t.file_ids) && (d[:file_ids] = t.file_ids)
    return d
end

# ─── Newer hosted tools (GA computer, shell, local_shell, apply_patch, custom) ──

"""
    ComputerTool(; environment=nothing)

GA computer-use tool (`type:"computer"`) for newer models. Unlike [`ComputerUseTool`](@ref)
(`computer_use_preview`) it carries no `display_width`/`display_height`.
"""
@kwdef struct ComputerTool <: ResponseTool
    environment::Union{String, Nothing} = nothing
end
function JSON.lower(t::ComputerTool)
    d = Dict{Symbol,Any}(:type => "computer")
    !isnothing(t.environment) && (d[:environment] = t.environment)
    return d
end

"""
    LocalShellTool()

Local shell tool (`type:"local_shell"`): the model emits shell commands you run on your
own runtime (codex-style models).
"""
struct LocalShellTool <: ResponseTool end
JSON.lower(::LocalShellTool) = Dict{Symbol,Any}(:type => "local_shell")

"""
    ShellTool(; environment=nothing)

Hosted shell tool (`type:"shell"`).
"""
@kwdef struct ShellTool <: ResponseTool
    environment::Union{AbstractDict, Nothing} = nothing
end
function JSON.lower(t::ShellTool)
    d = Dict{Symbol,Any}(:type => "shell")
    !isnothing(t.environment) && (d[:environment] = t.environment)
    return d
end

"""
    ApplyPatchTool()

Structured file-edit tool (`type:"apply_patch"`).
"""
struct ApplyPatchTool <: ResponseTool end
JSON.lower(::ApplyPatchTool) = Dict{Symbol,Any}(:type => "apply_patch")

"""
    CustomTool(; name, description=nothing, format=nothing)

Custom tool (`type:"custom"`) with free-form text input, or a grammar-constrained input via
`format = Dict("type"=>"grammar", "syntax"=>"lark"|"regex", "definition"=>...)`.
"""
@kwdef struct CustomTool <: ResponseTool
    name::String
    description::Union{String, Nothing} = nothing
    format::Union{String, AbstractDict, Nothing} = nothing
end
function JSON.lower(t::CustomTool)
    d = Dict{Symbol,Any}(:type => "custom", :name => t.name)
    !isnothing(t.description) && (d[:description] = t.description)
    !isnothing(t.format) && (d[:format] = t.format)
    return d
end

"Construct a [`ComputerTool`](@ref) (GA `computer` tool); for the preview variant with display dimensions use [`computer_use`](@ref)."
computer_tool(; environment::Union{String,Nothing}=nothing) = ComputerTool(environment=environment)
"Construct a [`LocalShellTool`](@ref) (`local_shell`): the model emits commands you run on your own runtime."
local_shell() = LocalShellTool()
"Construct a [`ShellTool`](@ref) (hosted `shell` tool)."
shell(; environment::Union{AbstractDict,Nothing}=nothing) = ShellTool(environment=environment)
"Construct an [`ApplyPatchTool`](@ref) (structured `apply_patch` file-edit tool)."
apply_patch_tool() = ApplyPatchTool()
"Construct a [`CustomTool`](@ref) (`custom` tool); `format` optionally constrains input to a grammar."
custom_tool(name::String; description::Union{String,Nothing}=nothing,
    format::Union{String,AbstractDict,Nothing}=nothing) =
    CustomTool(name=name, description=description, format=format)

# Convenience constructors

"""
    mcp_tool(label, url; require_approval="never", allowed_tools=nothing, headers=nothing)

Shorthand constructor for [`MCPTool`](@ref).
"""
mcp_tool(label::String, url::Union{String,Nothing}=nothing;
    require_approval::Union{String, AbstractDict, Nothing}="never",
    allowed_tools::Union{Vector{String}, AbstractDict, Nothing}=nothing,
    headers::Union{AbstractDict, Nothing}=nothing,
    connector_id::Union{String, Nothing}=nothing,
    authorization::Union{String, Nothing}=nothing,
    server_description::Union{String, Nothing}=nothing,
    tunnel_id::Union{String, Nothing}=nothing) =
    MCPTool(server_label=label, server_url=url, require_approval=require_approval,
        allowed_tools=allowed_tools, headers=headers, connector_id=connector_id,
        authorization=authorization, server_description=server_description, tunnel_id=tunnel_id)

"""
    mcp_approval_response(approval_request_id, approve; reason=nothing)

Build an `mcp_approval_response` input item to approve/deny a pending MCP tool call.
Pass it back as an element of the next request's `input`.
"""
function mcp_approval_response(approval_request_id::String, approve::Bool; reason::Union{String,Nothing}=nothing)
    d = Dict{Symbol,Any}(:type => "mcp_approval_response", :approval_request_id => approval_request_id, :approve => approve)
    !isnothing(reason) && (d[:reason] = reason)
    return d
end

"""
    computer_use(; display_width=1024, display_height=768, environment=nothing)

Shorthand constructor for [`ComputerUseTool`](@ref).
"""
computer_use(; display_width::Int=1024, display_height::Int=768,
    environment::Union{String, Nothing}=nothing) =
    ComputerUseTool(display_width=display_width, display_height=display_height, environment=environment)

"""
    image_generation_tool(; kwargs...)

Shorthand constructor for [`ImageGenerationTool`](@ref).
"""
image_generation_tool(; kwargs...) = ImageGenerationTool(; kwargs...)

"""
    code_interpreter(; container=nothing, file_ids=nothing)

Shorthand constructor for [`CodeInterpreterTool`](@ref).
"""
code_interpreter(; container::Union{AbstractDict, Nothing}=nothing,
    file_ids::Union{Vector{String}, Nothing}=nothing) =
    CodeInterpreterTool(container=container, file_ids=file_ids)

# Convenience constructors (existing)

"""
    function_tool(name, description=nothing; parameters=nothing, strict=nothing)

Shorthand constructor for [`FunctionTool`](@ref).
"""
function_tool(name::String, description::Union{String,Nothing}=nothing;
    parameters::Union{AbstractDict,Nothing}=nothing,
    strict::Union{Bool,Nothing}=nothing) =
    FunctionTool(name=name, description=description, parameters=parameters, strict=strict)

"""
    function_tool(d::AbstractDict)

Construct a [`FunctionTool`](@ref) from a dict. Accepts both the bare format
`{"name": ...}` and the wrapped format `{"type": "function", "function": {"name": ...}}`.
"""
function function_tool(d::AbstractDict)
    inner = haskey(d, "function") && d["function"] isa AbstractDict ? d["function"] : d
    FunctionTool(
        name=inner["name"],
        description=get(inner, "description", nothing),
        parameters=get(inner, "parameters", nothing),
        strict=get(inner, "strict", nothing)
    )
end

"""
    tool_result(call_id, name, output) -> Dict

Neutral multi-turn tool-result input item for the agentic verb. Feed a function's
output back via `respond(previous_response_id=id, input=[tool_result(...)])` or through
[`tool_loop`](@ref). Wire-neutral: OpenAI serializes it as `function_call_output`
(ignoring `name`); the Gemini encoder translates it to `function_result` (which requires
`name`). `output` is the function's return value as a string.
"""
tool_result(call_id::AbstractString, name::AbstractString, output::AbstractString) =
    Dict{String,Any}("type" => "function_call_output", "call_id" => call_id,
                     "name" => name, "output" => output)

"""
    web_search(; context_size="medium", location=nothing)

Shorthand constructor for [`WebSearchTool`](@ref).
"""
web_search(; context_size::String="medium",
    location::Union{AbstractDict,Nothing}=nothing,
    type::String="web_search",
    filters::Union{AbstractDict,Nothing}=nothing) =
    WebSearchTool(type=type, search_context_size=context_size, user_location=location, filters=filters)

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


# ─── tool_choice helpers ──────────────────────────────────────────────────────
# Beyond the string forms ("auto"/"none"/"required"), `Respond.tool_choice` accepts a
# Dict to force a specific tool. These builders produce those Dicts.

"""
    tool_choice_function(name)

Force the model to call a specific function tool: `{type:"function", name}`.
"""
tool_choice_function(name::String) = Dict{Symbol,Any}(:type => "function", :name => name)

"""
    tool_choice_hosted(type)

Force a specific hosted tool, e.g. `tool_choice_hosted("file_search")`, `"image_generation"`,
`"code_interpreter"`. Note: the hosted web-search selector is `"web_search_preview"` (not `"web_search"`).
"""
tool_choice_hosted(type::String) = Dict{Symbol,Any}(:type => type)

"""
    tool_choice_mcp(server_label; name=nothing)

Force a specific MCP server (optionally a specific tool): `{type:"mcp", server_label, name?}`.
"""
function tool_choice_mcp(server_label::String; name::Union{String,Nothing}=nothing)
    d = Dict{Symbol,Any}(:type => "mcp", :server_label => server_label)
    !isnothing(name) && (d[:name] = name)
    return d
end

"""
    tool_choice_custom(name)

Force a specific custom tool: `{type:"custom", name}`.
"""
tool_choice_custom(name::String) = Dict{Symbol,Any}(:type => "custom", :name => name)

"""
    tool_choice_allowed(mode, tools)

Constrain the model to a subset of tools: `{type:"allowed_tools", mode, tools}`.
`mode` is `"auto"` or `"required"`; `tools` is a vector of tool-reference dicts.
"""
tool_choice_allowed(mode::String, tools::Vector) =
    Dict{Symbol,Any}(:type => "allowed_tools", :mode => mode, :tools => tools)


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
    verbosity::Union{String,Nothing} = nothing   # "low" | "medium" | "high" (gpt-5.x)
end

function JSON.lower(t::TextConfig)
    d = Dict{Symbol,Any}(:format => t.format)
    !isnothing(t.verbosity) && (d[:verbosity] = t.verbosity)
    return d
end

# Convenience constructors

"""
    text_format(; kwargs...)

Create a [`TextConfig`](@ref) with the given format options.
"""
text_format(; verbosity::Union{String,Nothing}=nothing, kwargs...) =
    TextConfig(format=TextFormatSpec(; kwargs...), verbosity=verbosity)

"""
    json_schema_format(name, description, schema; strict=nothing)

Create a JSON Schema output format for structured output.
"""
json_schema_format(name::String, description::String, schema::AbstractDict;
    strict::Union{Bool,Nothing}=nothing) =
    TextConfig(format=TextFormatSpec(type="json_schema", name=name, description=description, schema=schema, strict=strict))

"""
    json_schema_format(d::AbstractDict)

Construct a JSON Schema [`TextConfig`](@ref) from a dict with keys `"name"`, `"description"`, and `"schema"`.
"""
json_schema_format(d::AbstractDict) = TextConfig(
    format=TextFormatSpec(
        type="json_schema",
        name=d["name"],
        description=get(d, "description", nothing),
        schema=d["schema"],
        strict=get(d, "strict", nothing)
    )
)

"""
    json_object_format()

Create a JSON object output format (unstructured).
"""
json_object_format() = TextConfig(format=TextFormatSpec(type="json_object"))


"""
    Reasoning(; effort=nothing, summary=nothing, generate_summary=nothing)

Reasoning configuration for reasoning models (gpt-5.x, o-series).

- `effort`: `"none"`, `"minimal"`, `"low"`, `"medium"`, `"high"`, or `"xhigh"` (supported values are
  model-dependent; passed through verbatim).
- `summary`: `"auto"`, `"concise"`, or `"detailed"` — request a reasoning summary in the output.
- `generate_summary`: deprecated alias of `summary`; prefer `summary`.
"""
@kwdef struct Reasoning
    effort::Union{String,Nothing} = nothing
    generate_summary::Union{String,Nothing} = nothing
    summary::Union{String,Nothing} = nothing
end

function JSON.lower(r::Reasoning)
    d = Dict{Symbol,Any}()
    !isnothing(r.effort) && (d[:effort] = r.effort)
    !isnothing(r.generate_summary) && (d[:generate_summary] = r.generate_summary)
    !isnothing(r.summary) && (d[:summary] = r.summary)
    return d
end


# ─── Main Request Type ────────────────────────────────────────────────────────

"""
    Respond(; model="gpt-5.5", input, kwargs...)

Configuration struct for an OpenAI Responses API request.

# Key Fields
- `model::String`: Model to use (default: `"gpt-5.5"`)
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
    service::ServiceEndpointSpec = OPENAIServiceEndpoint
    model::String = ""
    input::Union{String, Vector}  # String, Vector{InputMessage}, or Vector{Dict}
    instructions::Union{String,Nothing} = nothing
    tools::Union{Vector,Nothing} = nothing  # Untyped Vector: accepts ResponseTool, CallableTool, and Dict
    tool_choice::Union{String,AbstractDict,Nothing} = nothing  # "auto"/"none"/"required" or a {type:...} object
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
    background::Union{Bool,Nothing} = nothing
    include::Union{Vector{String},Nothing} = nothing
    max_tool_calls::Union{Int64,Nothing} = nothing
    service_tier::Union{String,Nothing} = nothing      # "auto", "default", "flex", "priority"
    top_logprobs::Union{Int64,Nothing} = nothing       # 0-20
    prompt::Union{AbstractDict,Nothing} = nothing
    prompt_cache_key::Union{String,Nothing} = nothing
    prompt_cache_retention::Union{String,Nothing} = nothing  # "in-memory", "24h"
    safety_identifier::Union{String,Nothing} = nothing
    conversation::Union{Any,Nothing} = nothing         # String or Dict
    context_management::Union{Vector,Nothing} = nothing
    stream_options::Union{AbstractDict,Nothing} = nothing
    function Respond(service, model, input, instructions, tools, tool_choice,
        parallel_tool_calls, temperature, top_p, max_output_tokens,
        stream, text, reasoning, truncation, store, metadata,
        previous_response_id, user, background, include, max_tool_calls,
        service_tier, top_logprobs, prompt, prompt_cache_key,
        prompt_cache_retention, safety_identifier, conversation,
        context_management, stream_options)
        model = _resolve_model(service, model)
        !isnothing(temperature) && !isnothing(top_p) && throw(ArgumentError("temperature and top_p are mutually exclusive"))
        !isnothing(temperature) && !(0.0 <= temperature <= 2.0) && throw(ArgumentError("temperature must be in [0.0, 2.0]"))
        !isnothing(top_p) && !(0.0 <= top_p <= 1.0) && throw(ArgumentError("top_p must be in [0.0, 1.0]"))
        !isnothing(max_output_tokens) && max_output_tokens < 1 && throw(ArgumentError("max_output_tokens must be >= 1"))
        !isnothing(top_logprobs) && !(0 <= top_logprobs <= 20) && throw(ArgumentError("top_logprobs must be in [0, 20]"))
        new(service, model, input, instructions, tools, tool_choice,
            parallel_tool_calls, temperature, top_p, max_output_tokens,
            stream, text, reasoning, truncation, store, metadata,
            previous_response_id, user, background, include, max_tool_calls,
            service_tier, top_logprobs, prompt, prompt_cache_key,
            prompt_cache_retention, safety_identifier, conversation,
            context_management, stream_options)
    end
end

function JSON.lower(r::Respond)
    d = Dict{Symbol,Any}(:model => r.model, :input => r.input)
    for f in (:instructions, :tools, :tool_choice, :parallel_tool_calls,
        :temperature, :top_p, :max_output_tokens, :stream, :text,
        :reasoning, :truncation, :store, :metadata, :previous_response_id,
        :user, :background, :include, :max_tool_calls, :service_tier,
        :top_logprobs, :prompt, :prompt_cache_key, :prompt_cache_retention,
        :safety_identifier, :conversation, :context_management, :stream_options)
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
    error::Union{Dict{String,Any}, String, Nothing} = nothing
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

HTTP-level failure from the Responses API. Contains the response body, status code, and optional request ID.
"""
@kwdef struct ResponseFailure <: LLMRequestResponse
    response::String
    status::Int
    request_id::Union{String, Nothing} = nothing
end

"""
    ResponseCallError <: LLMRequestResponse

Exception-level error during a Responses API call (network, parsing, etc.).
"""
@kwdef struct ResponseCallError <: LLMRequestResponse
    error::String
    status::Union{Int,Nothing} = nothing
    request_id::Union{String, Nothing} = nothing
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
            for content in _as_iter(get(item, "content", ()))
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


# ─── Additional typed accessors over the output array ─────────────────────────

# Tolerate JSON null / missing / non-array where the API may return an array (e.g. content: null).
_as_iter(x) = x isa AbstractVector ? x : ()

# Collect output items of a given "type".
_output_items(r::ResponseObject, typ::String) =
    Dict{String,Any}[item for item in r.output if item isa Dict && get(item, "type", "") == typ]

"""
    reasoning_summaries(r) -> Vector{String}

Reasoning-summary text from each `reasoning` output item.
"""
function reasoning_summaries(r::ResponseObject)
    out = String[]
    for item in _output_items(r, "reasoning"), s in _as_iter(get(item, "summary", ()))
        s isa Dict && haskey(s, "text") && push!(out, s["text"])
    end
    return out
end

"""
    refusals(r) -> Vector{String}

Refusal messages from any `refusal` content part of the output messages.
"""
function refusals(r::ResponseObject)
    out = String[]
    for item in r.output
        item isa Dict && get(item, "type", "") == "message" || continue
        for c in _as_iter(get(item, "content", ()))
            c isa Dict && get(c, "type", "") == "refusal" && haskey(c, "refusal") && push!(out, c["refusal"])
        end
    end
    return out
end

"""
    url_citations(r) -> Vector{Dict{String,Any}}

URL-citation annotations on output_text parts (from web_search).
"""
function url_citations(r::ResponseObject)
    out = Dict{String,Any}[]
    for item in r.output
        item isa Dict && get(item, "type", "") == "message" || continue
        for c in _as_iter(get(item, "content", ())), a in _as_iter(c isa Dict ? get(c, "annotations", ()) : ())
            a isa Dict && get(a, "type", "") == "url_citation" && push!(out, a)
        end
    end
    return out
end

"""
    image_generation_results(r) -> Vector{String}

Base64 image results from `image_generation_call` output items.
"""
function image_generation_results(r::ResponseObject)
    out = String[]
    for item in _output_items(r, "image_generation_call")
        v = get(item, "result", nothing)
        v isa String && push!(out, v)
    end
    return out
end

"Raw `web_search_call` output items (request results via `include`)."
web_search_results(r::ResponseObject)       = _output_items(r, "web_search_call")
"Raw `file_search_call` output items."
file_search_results(r::ResponseObject)      = _output_items(r, "file_search_call")
"Raw `code_interpreter_call` output items."
code_interpreter_outputs(r::ResponseObject) = _output_items(r, "code_interpreter_call")
"Raw `mcp_call` output items."
mcp_call_outputs(r::ResponseObject)         = _output_items(r, "mcp_call")
"Raw `mcp_approval_request` output items (feed back via [`mcp_approval_response`](@ref))."
mcp_approval_requests(r::ResponseObject)    = _output_items(r, "mcp_approval_request")
"Raw `reasoning` output items."
reasoning_items(r::ResponseObject)          = _output_items(r, "reasoning")

"The response's lifecycle status string (e.g. `completed`, `requires_action`); `failed`/`error` for non-success results."
response_status(r::ResponseObject)    = r.status
"Why a response is `incomplete` (the API's `incomplete_details` object), or `nothing`."
incomplete_details(r::ResponseObject) = get(r.raw, "incomplete_details", nothing)
"The response's raw token-usage dict; see [`token_usage`](@ref) for the typed [`TokenUsage`](@ref)."
usage_details(r::ResponseObject)      = r.usage

# ResponseSuccess forwarders + empty/typed defaults for the non-success results.
# String-returning vs Dict-returning accessors get correctly-typed empty defaults (not Vector{Any}).
for f in (:reasoning_summaries, :refusals, :image_generation_results)
    @eval $f(r::ResponseSuccess) = $f(r.response)
    @eval $f(::ResponseFailure) = String[]
    @eval $f(::ResponseCallError) = String[]
end
for f in (:url_citations, :web_search_results, :file_search_results, :code_interpreter_outputs,
          :mcp_call_outputs, :mcp_approval_requests, :reasoning_items)
    @eval $f(r::ResponseSuccess) = $f(r.response)
    @eval $f(::ResponseFailure) = Dict{String,Any}[]
    @eval $f(::ResponseCallError) = Dict{String,Any}[]
end
response_status(r::ResponseSuccess) = response_status(r.response)
response_status(::ResponseFailure) = "failed"
response_status(::ResponseCallError) = "error"
incomplete_details(r::ResponseSuccess) = incomplete_details(r.response)
incomplete_details(::ResponseFailure) = nothing
incomplete_details(::ResponseCallError) = nothing
usage_details(r::ResponseSuccess) = usage_details(r.response)
usage_details(::ResponseFailure) = nothing
usage_details(::ResponseCallError) = nothing


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

function _parse_response_stream_chunk(chunk::String, textbuff::IOBuffer, failbuff::IOBuffer,
                                      last_event::Ref{String}=Ref(""))
    # Layers 1–2 of the shared SSE machine (src/sse.jl): `failbuff` is the
    # partial-line carry (verbatim, never stripped), `last_event` the sticky
    # event name. Event dispatch below is unchanged. A COMPLETE line whose
    # payload fails to parse is logged + counted + dropped — never re-queued.
    for (ev, payload) in _sse_events!(failbuff, last_event, chunk)
        try
            data = JSON.parse(payload; dicttype=Dict{String,Any})
            if ev == "response.output_text.delta"
                print(textbuff, get(data, "delta", ""))
            elseif ev == "response.completed"
                return (; done=true, event=ev, data, terminal=:completed)
            elseif ev == "response.failed"
                return (; done=true, event=ev, data, terminal=:failed)
            elseif ev == "response.incomplete"
                return (; done=true, event=ev, data, terminal=:incomplete)
            elseif ev == "error"
                return (; done=true, event=ev, data, terminal=:error)
            end
            # Every other event (created/in_progress/queued, output_item.*,
            # content_part.*, refusal.*, function_call_arguments.*, reasoning*,
            # hosted-tool progress) degrades gracefully — as do unknown types.
        catch e
            Threads.atomic_add!(_SSE_DROPPED_LINES, 1)
            @debug "Responses SSE: dropped undecodable data payload" event = ev payload = String(payload) exception = e
        end
    end
    return (; done=false, event=last_event[], data=nothing, terminal=:none)
end

function _respond_stream(r::Respond, body::String, callback=nothing)
    Threads.@spawn begin
        io_ref = Ref{Union{HTTP.Stream,Nothing}}(nothing)
        try
            result = Ref{Union{ResponseObject,Nothing}}(nothing)
            terminal_error = Ref{Union{Dict{String,Any},Nothing}}(nothing)  # structured failed/incomplete/error payload
            raw_buffer = IOBuffer()  # wire bytes for non-200 reporting (streamed resp.body is empty under HTTP 2.x)
            url = get_url(r.service, r)
            # SSE must reach the parser uncompressed: some providers (Gemini Interactions)
            # gzip even streamed responses, and HTTP.jl's 1.x streaming read does NOT
            # auto-decompress the body — raw gzip bytes hit the SSE parser, every line fails
            # to decode, and no text/output is built. Request identity encoding + disable
            # decompression so `data:` lines arrive verbatim (mirrors _chatrequeststream).
            stream_headers = push!(copy(auth_header(r.service)), "Accept-Encoding" => "identity")
            resp = HTTP.open("POST", url, stream_headers; status_exception=false, decompress=false) do io
                io_ref[] = io
                text_buffer = IOBuffer()
                fail_buffer = IOBuffer()
                last_event = Ref("")
                done = Ref(false)
                close_ref = Ref(false)
                callback_buf = IOBuffer()  # tracks already-emitted text (emitted-length delta)
                write(io, body)
                HTTP.closewrite(io)
                HTTP.startread(io)
                while !eof(io) && !close_ref[] && !done[]
                    chunk = String(readavailable(io))
                    write(raw_buffer, chunk)
                    status = decode_agentic_stream(r.service, chunk, text_buffer, fail_buffer, last_event)
                    if status.terminal == :completed && status.data isa AbstractDict && haskey(status.data, "response")
                        # Flush residual text buffer to callback before building the ResponseObject
                        full = String(take!(text_buffer))
                        emitted = String(take!(callback_buf))
                        if sizeof(full) > sizeof(emitted) && !isnothing(callback)
                            callback(full[nextind(full, sizeof(emitted)):end], close_ref)
                        end
                        print(text_buffer, full)
                        print(callback_buf, full)

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
                    elseif status.terminal in (:failed, :incomplete, :error) && !isnothing(status.data)
                        # Structured terminal failure mid-stream (HTTP itself may be 200): keep the
                        # response's own error/incomplete details instead of dropping them.
                        terminal_error[] = status.data
                        done[] = true
                    else
                        # Retain full accumulated text (emitted-length tracking, like the chat
                        # stream) so a provider whose TERMINAL event omits the output — Gemini's
                        # `interaction.completed` carries no steps — can rebuild it from the
                        # deltas. OpenAI ignores this at assembly (its completed event has output).
                        full = String(take!(text_buffer))
                        emitted = String(take!(callback_buf))
                        if sizeof(full) > sizeof(emitted) && !isnothing(callback)
                            callback(full[nextind(full, sizeof(emitted)):end], close_ref)
                        end
                        print(text_buffer, full)
                        print(callback_buf, full)
                    end
                end
                close_ref[] && @info "Response stream closed by user"
                HTTP.closeread(io)
            end
            if resp.status == 200 && !isnothing(result[])
                ResponseSuccess(response=result[]::ResponseObject)
            elseif !isnothing(terminal_error[])
                te = terminal_error[]
                req_id = !isnothing(io_ref[]) ? _get_request_id(io_ref[]) : nothing
                if haskey(te, "response")            # response.failed / response.incomplete
                    ResponseFailure(response=JSON.json(te["response"]), status=resp.status, request_id=req_id)
                else                                  # bare `error` event
                    ResponseCallError(error=get(te, "message", JSON.json(te)),
                        status=(resp.status == 200 ? nothing : resp.status), request_id=req_id)
                end
            else
                ResponseFailure(response=String(take!(raw_buffer)), status=resp.status, request_id=_get_request_id(resp))
            end
        catch e
            statuserror = hasproperty(e, :status) ? e.status : nothing
            req_id = !isnothing(io_ref[]) ? _get_request_id(io_ref[]) : _get_request_id(e)
            ResponseCallError(error=string(e), status=statuserror, request_id=req_id)
        end
    end
end


# ─── Request Functions ───────────────────────────────────────────────────────

# ─── Agentic wire-translation seam ───────────────────────────────────────────
# Parallel to the chat seam (src/requests.jl:252-273): three generics dispatched
# on `service` translate between the neutral Respond/ResponseObject IR and a
# provider's agentic wire. The untyped-`service` methods below are the OpenAI
# Responses defaults; a provider with a different surface (Gemini Interactions)
# overrides them. `respond`/`_respond_stream` call ONLY these generics,
# so retry/HTTP/cost/streaming orchestration stays provider-agnostic.
# NB: named `*_agentic`, NOT `decode_response` — that would collide with the chat
# seam's `decode_response(service, ::HTTP.Response)` (identical argument types).

get_url(r::Respond) = get_url(r.service, r)
_agentic_url(service) = _api_base_url(service) * RESPONSES_PATH
get_url(service, r::Respond) = _agentic_url(service)

encode_agentic(service, r::Respond)::String = JSON.json(r)

decode_agentic(service, resp::HTTP.Response)::ResponseObject = parse_response(resp)

decode_agentic_stream(service, chunk::String, textbuff::IOBuffer, failbuff::IOBuffer,
                      last_event::Ref{String}) =
    _parse_response_stream_chunk(chunk, textbuff, failbuff, last_event)

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
    local resp
    try
        body = encode_agentic(r.service, r)

        # Streaming path
        if !isnothing(r.stream) && r.stream
            return _respond_stream(r, body, callback)
        end

        url = get_url(r.service, r)
        resp = HTTP.post(url, body=body, headers=auth_header(r.service); status_exception=false)

        if resp.status == 200
            return ResponseSuccess(response=decode_agentic(r.service, resp))
        elseif _is_retryable(resp.status)
            if retries < _RETRY_MAX_ATTEMPTS
                delay = _retry_delay(retries, resp)
                @warn "Request status: $(resp.status). Retrying in $(round(delay; digits=2))s..."
                sleep(delay)
                return respond(r; retries=retries + 1, callback=callback)
            else
                return ResponseFailure(response=String(resp.body), status=resp.status, request_id=_get_request_id(resp))
            end
        else
            return ResponseFailure(response=String(resp.body), status=resp.status, request_id=_get_request_id(resp))
        end
    catch e
        statuserror = hasproperty(e, :status) ? e.status : nothing
        req_id = @isdefined(resp) ? _get_request_id(resp) : _get_request_id(e)
        res = ResponseCallError(error=string(e), status=statuserror, request_id=req_id)
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
result = respond("Translate: Hello", instructions="You are a translator", model="gpt-5.5")

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
function get_response(response_id::String; service::ServiceEndpointSpec=OPENAIServiceEndpoint)
    local resp
    try
        url = _agentic_url(service) * "/" * response_id
        resp = HTTP.get(url, headers=auth_header(service); status_exception=false)
        if resp.status == 200
            return ResponseSuccess(response=decode_agentic(service, resp))
        else
            return ResponseFailure(response=String(resp.body), status=resp.status, request_id=_get_request_id(resp))
        end
    catch e
        statuserror = hasproperty(e, :status) ? e.status : nothing
        req_id = @isdefined(resp) ? _get_request_id(resp) : _get_request_id(e)
        return ResponseCallError(error=string(e), status=statuserror, request_id=req_id)
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
function delete_response(response_id::String; service::ServiceEndpointSpec=OPENAIServiceEndpoint)
    local resp
    try
        url = _agentic_url(service) * "/" * response_id
        resp = HTTP.request("DELETE", url, headers=auth_header(service); status_exception=false)
        if resp.status == 200
            return JSON.parse(resp.body; dicttype=Dict{String,Any})
        else
            return ResponseFailure(response=String(resp.body), status=resp.status, request_id=_get_request_id(resp))
        end
    catch e
        statuserror = hasproperty(e, :status) ? e.status : nothing
        req_id = @isdefined(resp) ? _get_request_id(resp) : _get_request_id(e)
        return ResponseCallError(error=string(e), status=statuserror, request_id=req_id)
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
    service::ServiceEndpointSpec=OPENAIServiceEndpoint)

    local resp
    try
        url = _agentic_url(service) * "/" * response_id * "/input_items"
        params = ["limit=$limit", "order=$order"]
        !isnothing(after) && push!(params, "after=$after")
        url *= "?" * join(params, "&")

        resp = HTTP.get(url, headers=auth_header(service); status_exception=false)
        if resp.status == 200
            return JSON.parse(resp.body; dicttype=Dict{String,Any})
        else
            return ResponseFailure(response=String(resp.body), status=resp.status, request_id=_get_request_id(resp))
        end
    catch e
        statuserror = hasproperty(e, :status) ? e.status : nothing
        req_id = @isdefined(resp) ? _get_request_id(resp) : _get_request_id(e)
        return ResponseCallError(error=string(e), status=statuserror, request_id=req_id)
    end
end


"""
    cancel_response(response_id::String; service=OPENAIServiceEndpoint)

Cancel an in-progress response by its ID. Returns `ResponseSuccess` on success.

# Examples
```julia
# Start a background response, then cancel it
result = respond("Write a very long essay", background=true)
cancel_result = cancel_response(result.response.id)
if cancel_result isa ResponseSuccess
    println("Cancelled: ", cancel_result.response.status)
end
```
"""
function cancel_response(response_id::String; service::ServiceEndpointSpec=OPENAIServiceEndpoint)
    local resp
    try
        url = _agentic_url(service) * "/" * response_id * "/cancel"
        resp = HTTP.post(url, headers=auth_header(service); status_exception=false)
        if resp.status == 200
            return ResponseSuccess(response=decode_agentic(service, resp))
        else
            return ResponseFailure(response=String(resp.body), status=resp.status, request_id=_get_request_id(resp))
        end
    catch e
        statuserror = hasproperty(e, :status) ? e.status : nothing
        req_id = @isdefined(resp) ? _get_request_id(resp) : _get_request_id(e)
        return ResponseCallError(error=string(e), status=statuserror, request_id=req_id)
    end
end


"""
    compact_response(; model, input, kwargs...)

Compact a conversation by running a compaction pass. Returns opaque, encrypted items
that can be passed as input to subsequent requests, reducing token usage in long conversations.

# Fields
- `model::String`: Model to use for compaction
- `input::Any`: The conversation items to compact (typically the full conversation history)

Returns a Dict with `"id"`, `"object"`, `"output"`, and `"usage"` keys.

# Examples
```julia
compacted = compact_response(model="gpt-5.5", input=[
    InputMessage(role="user", content="Hello"),
    Dict("type" => "message", "role" => "assistant", "status" => "completed",
         "content" => [Dict("type" => "output_text", "text" => "Hi there!")])
])
# Use compacted["output"] as input to the next request
```
"""
function compact_response(; model::String="gpt-5.5",
    input::Any,
    service::ServiceEndpointSpec=OPENAIServiceEndpoint)

    local resp
    try
        url = _api_base_url(service) * RESPONSES_PATH * "/compact"
        body = JSON.json(Dict{Symbol,Any}(:model => model, :input => input))
        resp = HTTP.post(url, body=body, headers=auth_header(service); status_exception=false)
        if resp.status == 200
            return JSON.parse(resp.body; dicttype=Dict{String,Any})
        else
            return ResponseFailure(response=String(resp.body), status=resp.status, request_id=_get_request_id(resp))
        end
    catch e
        statuserror = hasproperty(e, :status) ? e.status : nothing
        req_id = @isdefined(resp) ? _get_request_id(resp) : _get_request_id(e)
        return ResponseCallError(error=string(e), status=statuserror, request_id=req_id)
    end
end


"""
    count_input_tokens(; model, input, kwargs...)

Count the number of input tokens a request would use without actually generating a response.
Useful for estimating costs or checking whether input fits within the context window.

Returns a Dict with `"object"` (`"response.input_tokens"`) and `"input_tokens"` keys.

# Examples
```julia
result = count_input_tokens(model="gpt-5.5", input="Tell me a joke")
println("Input tokens: ", result["input_tokens"])
```
"""
function count_input_tokens(; model::String="gpt-5.5",
    input::Any,
    instructions::Union{String,Nothing}=nothing,
    tools::Union{Vector,Nothing}=nothing,
    service::ServiceEndpointSpec=OPENAIServiceEndpoint)

    local resp
    try
        url = _api_base_url(service) * RESPONSES_PATH * "/input_tokens"
        d = Dict{Symbol,Any}(:model => model, :input => input)
        !isnothing(instructions) && (d[:instructions] = instructions)
        !isnothing(tools) && (d[:tools] = tools)
        body = JSON.json(d)
        resp = HTTP.post(url, body=body, headers=auth_header(service); status_exception=false)
        if resp.status == 200
            return JSON.parse(resp.body; dicttype=Dict{String,Any})
        else
            return ResponseFailure(response=String(resp.body), status=resp.status, request_id=_get_request_id(resp))
        end
    catch e
        statuserror = hasproperty(e, :status) ? e.status : nothing
        req_id = @isdefined(resp) ? _get_request_id(resp) : _get_request_id(e)
        return ResponseCallError(error=string(e), status=statuserror, request_id=req_id)
    end
end
