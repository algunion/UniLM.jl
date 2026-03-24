# ============================================================================
# MCP Server — Build MCP servers using UniLM.jl
# Register tools, resources, and prompts, then serve over stdio or HTTP.
#
# Protocol: JSON-RPC 2.0 over stdio or Streamable HTTP (spec 2025-11-25)
# ============================================================================

# ─── Server Primitive Types ──────────────────────────────────────────────────

"""
    MCPServerPrimitive

Abstract supertype for MCP server-side primitives (tools, resources, prompts).
"""
abstract type MCPServerPrimitive end

"""
    MCPServerTool <: MCPServerPrimitive

A tool registered on an MCP server. The `handler` receives `Dict{String,Any}` arguments
and returns any value (converted to text content by the server).

# Fields
- `name::String`: Unique tool name
- `description::Union{String,Nothing}`: Human-readable description
- `input_schema::Dict{String,Any}`: JSON Schema for the input parameters
- `handler::Function`: `(args::Dict{String,Any}) -> Any`
"""
struct MCPServerTool <: MCPServerPrimitive
    name::String
    description::Union{String,Nothing}
    input_schema::Dict{String,Any}
    handler::Function
end

"""
    MCPServerResource <: MCPServerPrimitive

A static resource registered on an MCP server.

# Fields
- `uri::String`: Resource URI
- `name::String`: Human-readable name
- `description::Union{String,Nothing}`: Description
- `mime_type::String`: MIME type (default `"text/plain"`)
- `handler::Function`: `() -> Union{String, Vector{UInt8}}`
"""
struct MCPServerResource <: MCPServerPrimitive
    uri::String
    name::String
    description::Union{String,Nothing}
    mime_type::String
    handler::Function
end

"""
    MCPServerResourceTemplate <: MCPServerPrimitive

A URI-templated resource. Template variables like `{path}` are extracted and
passed to the handler.

# Fields
- `uri_template::String`: URI template (e.g., `"file://{path}"`)
- `name::String`: Human-readable name
- `description::Union{String,Nothing}`: Description
- `mime_type::String`: MIME type
- `handler::Function`: `(params::Dict{String,String}) -> Union{String, Vector{UInt8}}`
- `_pattern::Regex`: Compiled regex from template
- `_param_names::Vector{String}`: Extracted parameter names
"""
struct MCPServerResourceTemplate <: MCPServerPrimitive
    uri_template::String
    name::String
    description::Union{String,Nothing}
    mime_type::String
    handler::Function
    _pattern::Regex
    _param_names::Vector{String}
end

"""
    MCPServerPrompt <: MCPServerPrimitive

A prompt template registered on an MCP server.

# Fields
- `name::String`: Unique prompt name
- `description::Union{String,Nothing}`: Description
- `arguments::Vector{Dict{String,Any}}`: Argument definitions
- `handler::Function`: `(args::Dict{String,Any}) -> Vector{Dict{String,Any}}`
"""
struct MCPServerPrompt <: MCPServerPrimitive
    name::String
    description::Union{String,Nothing}
    arguments::Vector{Dict{String,Any}}
    handler::Function
end

# ─── URI Template Compilation ────────────────────────────────────────────────

"""Compile a URI template like `"file://{path}"` into a regex and param name list."""
function _compile_uri_template(template::String)
    param_names = String[]
    pattern = replace(template, r"\{(\w+)\}" => s -> begin
        name = s[2:end-1]
        push!(param_names, name)
        "(?P<$name>[^/]+)"
    end)
    (Regex("^" * pattern * "\$"), param_names)
end

# ─── MCPServer ───────────────────────────────────────────────────────────────

"""
    MCPServer(name, version; description=nothing)

An MCP server that can host tools, resources, and prompts.

Register primitives via [`register_tool!`](@ref), [`register_resource!`](@ref),
[`register_prompt!`](@ref), or the `@mcp_tool`, `@mcp_resource`, `@mcp_prompt` macros.

Start serving via [`serve`](@ref).

# Example
```julia
server = MCPServer("my-server", "1.0.0")
register_tool!(server, "add", "Add two numbers",
    Dict("type"=>"object", "properties"=>Dict("a"=>Dict("type"=>"number"),"b"=>Dict("type"=>"number")), "required"=>["a","b"]),
    args -> string(args["a"] + args["b"]))
serve(server)  # stdio by default
```
"""
mutable struct MCPServer
    name::String
    version::String
    description::Union{String,Nothing}
    tools::Dict{String,MCPServerTool}
    resources::Dict{String,MCPServerResource}
    resource_templates::Vector{MCPServerResourceTemplate}
    prompts::Dict{String,MCPServerPrompt}
    _initialized::Bool
end

function MCPServer(name::String, version::String; description::Union{String,Nothing}=nothing)
    MCPServer(name, version, description,
        Dict{String,MCPServerTool}(),
        Dict{String,MCPServerResource}(),
        MCPServerResourceTemplate[],
        Dict{String,MCPServerPrompt}(),
        false)
end

# ─── Registration ────────────────────────────────────────────────────────────

"""
    register_tool!(server, name, description, input_schema, handler)

Register a tool on the MCP server with an explicit JSON Schema.
"""
function register_tool!(server::MCPServer, name::String,
                        description::Union{String,Nothing},
                        input_schema::Dict{String,Any},
                        handler::Function)
    server.tools[name] = MCPServerTool(name, description, input_schema, handler)
    server
end

"""
    register_tool!(server, name, description, handler)

Register a tool with schema auto-inferred from the handler's type signature.
"""
function register_tool!(server::MCPServer, name::String,
                        description::Union{String,Nothing},
                        handler::Function)
    schema = _function_schema(handler)
    register_tool!(server, name, description, schema, handler)
end

"""
    register_tool!(server, ct::CallableTool{GPTTool})

Register a `CallableTool{GPTTool}` on the MCP server, bridging from UniLM's
Chat Completions tool type.
"""
function register_tool!(server::MCPServer, ct::CallableTool{GPTTool})
    name = ct.tool.func.name
    desc = ct.tool.func.description
    schema = something(ct.tool.func.parameters,
        Dict{String,Any}("type" => "object", "properties" => Dict{String,Any}()))
    handler = (args::Dict{String,Any}) -> ct.callable(name, args)
    register_tool!(server, name, desc, schema, handler)
end

"""
    register_tool!(server, ct::CallableTool{FunctionTool})

Register a `CallableTool{FunctionTool}` on the MCP server, bridging from UniLM's
Responses API tool type.
"""
function register_tool!(server::MCPServer, ct::CallableTool{FunctionTool})
    name = ct.tool.name
    desc = ct.tool.description
    schema = something(ct.tool.parameters,
        Dict{String,Any}("type" => "object", "properties" => Dict{String,Any}()))
    handler = (args::Dict{String,Any}) -> ct.callable(name, args)
    register_tool!(server, name, desc, schema, handler)
end

"""
    register_resource!(server, uri, name, handler; mime_type="text/plain", description=nothing)

Register a static resource on the MCP server.
"""
function register_resource!(server::MCPServer, uri::String, name::String,
                            handler::Function;
                            mime_type::String="text/plain",
                            description::Union{String,Nothing}=nothing)
    server.resources[uri] = MCPServerResource(uri, name, description, mime_type, handler)
    server
end

"""
    register_resource_template!(server, uri_template, name, handler; mime_type="text/plain", description=nothing)

Register a URI-templated resource on the MCP server.
"""
function register_resource_template!(server::MCPServer, uri_template::String,
                                     name::String, handler::Function;
                                     mime_type::String="text/plain",
                                     description::Union{String,Nothing}=nothing)
    pattern, param_names = _compile_uri_template(uri_template)
    push!(server.resource_templates,
        MCPServerResourceTemplate(uri_template, name, description, mime_type,
                                  handler, pattern, param_names))
    server
end

"""
    register_prompt!(server, name, handler; description=nothing, arguments=Dict{String,Any}[])

Register a prompt template on the MCP server.
"""
function register_prompt!(server::MCPServer, name::String, handler::Function;
                          description::Union{String,Nothing}=nothing,
                          arguments::Vector{Dict{String,Any}}=Dict{String,Any}[])
    server.prompts[name] = MCPServerPrompt(name, description, arguments, handler)
    server
end

# ─── Content Formatting ─────────────────────────────────────────────────────

"""Format a tool handler result into MCP content array."""
_format_tool_result(result::AbstractString) = [Dict{String,Any}("type" => "text", "text" => String(result))]
function _format_tool_result(result::AbstractDict)
    haskey(result, "type") ? [result] : [Dict{String,Any}("type" => "text", "text" => JSON.json(result))]
end
_format_tool_result(result::AbstractVector) = result  # pre-formatted content array
_format_tool_result(result) = [Dict{String,Any}("type" => "text", "text" => string(result))]

"""Format a resource handler result into MCP resource content."""
function _format_resource_content(uri::String, mime_type::String, result)
    d = Dict{String,Any}("uri" => uri, "mimeType" => mime_type)
    if result isa Vector{UInt8}
        d["blob"] = Base64.base64encode(result)
    else
        d["text"] = string(result)
    end
    d
end

# ─── JSON-RPC Dispatch ──────────────────────────────────────────────────────

"""Build a JSON-RPC success response."""
_jsonrpc_result(id, result) = Dict{String,Any}(
    "jsonrpc" => _JSONRPC_VERSION, "id" => id, "result" => result)

"""Build a JSON-RPC error response."""
_jsonrpc_error(id, code::Int, message::String; data=nothing) = begin
    err = Dict{String,Any}("code" => code, "message" => message)
    !isnothing(data) && (err["data"] = data)
    Dict{String,Any}("jsonrpc" => _JSONRPC_VERSION, "id" => id, "error" => err)
end

function _handle_initialize(server::MCPServer, id, params::Dict{String,Any})
    caps = Dict{String,Any}()
    !isempty(server.tools) && (caps["tools"] = Dict{String,Any}())
    (!isempty(server.resources) || !isempty(server.resource_templates)) && (caps["resources"] = Dict{String,Any}())
    !isempty(server.prompts) && (caps["prompts"] = Dict{String,Any}())
    server._initialized = true
    server_info = Dict{String,Any}("name" => server.name, "version" => server.version)
    !isnothing(server.description) && (server_info["description"] = server.description)
    _jsonrpc_result(id, Dict{String,Any}(
        "protocolVersion" => _MCP_PROTOCOL_VERSION,
        "capabilities" => caps,
        "serverInfo" => server_info
    ))
end

function _handle_tools_list(server::MCPServer, id, params::Dict{String,Any})
    tools_list = [begin
        d = Dict{String,Any}("name" => t.name, "inputSchema" => t.input_schema)
        !isnothing(t.description) && (d["description"] = t.description)
        d
    end for t in values(server.tools)]
    _jsonrpc_result(id, Dict{String,Any}("tools" => tools_list))
end

function _handle_tools_call(server::MCPServer, id, params::Dict{String,Any})
    name = params["name"]
    args = get(params, "arguments", Dict{String,Any}())
    tool = get(server.tools, name, nothing)
    isnothing(tool) && return _jsonrpc_error(id, -32602, "Unknown tool: $name")
    try
        result = tool.handler(args)
        _jsonrpc_result(id, Dict{String,Any}(
            "content" => _format_tool_result(result), "isError" => false))
    catch e
        _jsonrpc_result(id, Dict{String,Any}(
            "content" => [Dict{String,Any}("type" => "text", "text" => "Error: $(sprint(showerror, e))")],
            "isError" => true))
    end
end

function _handle_resources_list(server::MCPServer, id, params::Dict{String,Any})
    resources_list = [begin
        d = Dict{String,Any}("uri" => r.uri, "name" => r.name, "mimeType" => r.mime_type)
        !isnothing(r.description) && (d["description"] = r.description)
        d
    end for r in values(server.resources)]
    _jsonrpc_result(id, Dict{String,Any}("resources" => resources_list))
end

function _handle_resources_templates_list(server::MCPServer, id, params::Dict{String,Any})
    templates_list = [begin
        d = Dict{String,Any}("uriTemplate" => t.uri_template, "name" => t.name)
        !isnothing(t.description) && (d["description"] = t.description)
        d
    end for t in server.resource_templates]
    _jsonrpc_result(id, Dict{String,Any}("resourceTemplates" => templates_list))
end

function _handle_resources_read(server::MCPServer, id, params::Dict{String,Any})
    uri = params["uri"]
    # Check static resources first
    if haskey(server.resources, uri)
        r = server.resources[uri]
        try
            result = r.handler()
            content = _format_resource_content(uri, r.mime_type, result)
            return _jsonrpc_result(id, Dict{String,Any}("contents" => [content]))
        catch e
            return _jsonrpc_error(id, -32603, "Resource read error: $(sprint(showerror, e))")
        end
    end
    # Check templates
    for tmpl in server.resource_templates
        m = match(tmpl._pattern, uri)
        if !isnothing(m)
            params_dict = Dict{String,String}(name => m[name] for name in tmpl._param_names)
            try
                result = tmpl.handler(params_dict)
                content = _format_resource_content(uri, tmpl.mime_type, result)
                return _jsonrpc_result(id, Dict{String,Any}("contents" => [content]))
            catch e
                return _jsonrpc_error(id, -32603, "Resource read error: $(sprint(showerror, e))")
            end
        end
    end
    _jsonrpc_error(id, -32002, "Resource not found: $uri")
end

function _handle_prompts_list(server::MCPServer, id, params::Dict{String,Any})
    prompts_list = [begin
        d = Dict{String,Any}("name" => p.name)
        !isnothing(p.description) && (d["description"] = p.description)
        !isempty(p.arguments) && (d["arguments"] = p.arguments)
        d
    end for p in values(server.prompts)]
    _jsonrpc_result(id, Dict{String,Any}("prompts" => prompts_list))
end

function _handle_prompts_get(server::MCPServer, id, params::Dict{String,Any})
    name = params["name"]
    args = get(params, "arguments", Dict{String,Any}())
    prompt = get(server.prompts, name, nothing)
    isnothing(prompt) && return _jsonrpc_error(id, -32602, "Unknown prompt: $name")
    try
        messages = prompt.handler(args)
        _jsonrpc_result(id, Dict{String,Any}("messages" => messages))
    catch e
        _jsonrpc_error(id, -32603, "Prompt error: $(sprint(showerror, e))")
    end
end

"""Route a parsed JSON-RPC request to the appropriate handler."""
function _dispatch_mcp(server::MCPServer, parsed::Dict{String,Any})
    id = get(parsed, "id", nothing)
    method = get(parsed, "method", "")
    params = get(parsed, "params", Dict{String,Any}())
    # Notifications (no id) — handle silently
    isnothing(id) && return nothing
    handlers = Dict{String,Function}(
        "initialize" => (id, p) -> _handle_initialize(server, id, p),
        "tools/list" => (id, p) -> _handle_tools_list(server, id, p),
        "tools/call" => (id, p) -> _handle_tools_call(server, id, p),
        "resources/list" => (id, p) -> _handle_resources_list(server, id, p),
        "resources/templates/list" => (id, p) -> _handle_resources_templates_list(server, id, p),
        "resources/read" => (id, p) -> _handle_resources_read(server, id, p),
        "prompts/list" => (id, p) -> _handle_prompts_list(server, id, p),
        "prompts/get" => (id, p) -> _handle_prompts_get(server, id, p),
        "ping" => (id, _) -> _jsonrpc_result(id, Dict{String,Any}()),
    )
    handler = get(handlers, method, nothing)
    isnothing(handler) && return _jsonrpc_error(id, -32601, "Method not found: $method")
    handler(id, params)
end

# ─── Transports ──────────────────────────────────────────────────────────────

"""
    _serve_stdio(server::MCPServer; input=stdin, output=stdout)

Run the MCP server over stdio. Reads JSON-RPC messages from `input` (one per line),
dispatches them, and writes responses to `output`. Diagnostic logs go to stderr.
"""
function _serve_stdio(server::MCPServer; input::IO=stdin, output::IO=stdout)
    while !eof(input)
        line = readline(input)
        isempty(strip(line)) && continue
        parsed = try
            JSON.parse(line; dicttype=Dict{String,Any})
        catch e
            response = _jsonrpc_error(nothing, -32700, "Parse error: $(sprint(showerror, e))")
            println(output, JSON.json(response))
            flush(output)
            continue
        end
        response = _dispatch_mcp(server, parsed)
        # Notifications produce no response
        isnothing(response) && continue
        println(output, JSON.json(response))
        flush(output)
    end
end

"""
    _serve_http(server::MCPServer; host="127.0.0.1", port=8080)

Run the MCP server over HTTP. POST requests contain JSON-RPC messages.
"""
function _serve_http(server::MCPServer; host::String="127.0.0.1", port::Int=8080)
    HTTP.serve!(host, port) do req
        if req.method == "POST"
            body = String(req.body)
            parsed = try
                JSON.parse(body; dicttype=Dict{String,Any})
            catch
                return HTTP.Response(400, JSON.json(_jsonrpc_error(nothing, -32700, "Parse error")))
            end
            response = _dispatch_mcp(server, parsed)
            if isnothing(response)
                return HTTP.Response(202, "")
            end
            HTTP.Response(200, ["Content-Type" => "application/json"], JSON.json(response))
        elseif req.method == "DELETE"
            HTTP.Response(200, "")
        else
            HTTP.Response(405, "Method Not Allowed")
        end
    end
end

"""
    serve(server::MCPServer; transport=:stdio, kwargs...)

Start the MCP server using the specified transport.

# Transports
- `:stdio` (default): Read from stdin, write to stdout. For Claude Desktop/CLI integration.
- `:http`: HTTP server. Accepts `host` (default `"127.0.0.1"`) and `port` (default `8080`).

# Examples
```julia
serve(server)                            # stdio (default)
serve(server; transport=:http, port=3000)  # HTTP on port 3000
```
"""
function serve(server::MCPServer; transport::Symbol=:stdio, kwargs...)
    if transport == :stdio
        _serve_stdio(server; kwargs...)
    elseif transport == :http
        _serve_http(server; kwargs...)
    else
        throw(ArgumentError("Unknown transport: $transport. Use :stdio or :http"))
    end
end

# ─── Macros ──────────────────────────────────────────────────────────────────

"""
    @mcp_tool server function name(args...)::ReturnType body end

Register a tool on `server` with auto-generated JSON Schema from the function signature.

# Example
```julia
server = MCPServer("calc", "1.0.0")
@mcp_tool server function add(a::Float64, b::Float64)::String
    string(a + b)
end
```
"""
macro mcp_tool(server, func_expr)
    func_expr.head in (:function, :(=)) || error("@mcp_tool expects a function definition")
    call_expr = func_expr.args[1]
    body = func_expr.args[2]
    actual_call = call_expr isa Expr && call_expr.head == :(::) ? call_expr.args[1] : call_expr
    fname = actual_call.args[1]
    raw_args = actual_call.args[2:end]
    name_str = string(fname)
    # Extract arg names and types
    arg_info = [(string(a isa Expr && a.head == :(::) ? a.args[1] : a),
                 a isa Expr && a.head == :(::) ? a.args[2] : :Any)
                for a in raw_args]
    required = [n for (n, T) in arg_info if T !== :Any]
    # Build schema and handler with full esc to avoid hygiene issues
    prop_exprs = [:($(n) => UniLM._json_schema_type($(T))) for (n, T) in arg_info]
    unpack = [:(UniLM._mcp_convert($(T), get(_d_, $(n), nothing))) for (n, T) in arg_info]
    quote
        # Define the function in caller's scope
        function $(esc(fname))($(map(esc, raw_args)...))
            $(esc(body))
        end
        # Register with schema and dict-unpacking wrapper
        UniLM.register_tool!($(esc(server)), $name_str, nothing,
            Dict{String,Any}("type" => "object",
                "properties" => Dict{String,Any}($(prop_exprs...)),
                "required" => $required),
            function(_d_::Dict{String,Any})
                $(esc(fname))($(unpack...))
            end)
    end
end

"""Convert a value from Dict{String,Any} to the expected Julia type."""
_mcp_convert(::Type{String}, v) = string(v)
_mcp_convert(::Type{T}, v) where {T<:Integer} = convert(T, v isa AbstractFloat ? round(T, v) : v)
_mcp_convert(::Type{T}, v) where {T<:AbstractFloat} = convert(T, v)
_mcp_convert(::Type{Bool}, v) = convert(Bool, v)
_mcp_convert(::Type{T}, v) where {T} = v  # fallback: pass through

"""
    @mcp_resource server uri_or_template function(args...) body end

Register a resource or resource template on `server`.
If the URI contains `{...}` placeholders, it is registered as a template.

# Examples
```julia
@mcp_resource server "config://app" function()
    read("config.toml", String)
end

@mcp_resource server "file://{path}" function(path::String)
    read(path, String)
end
```
"""
macro mcp_resource(server, uri, func_expr)
    func_expr.head in (:function, :(=)) || error("@mcp_resource expects a function definition")
    body = func_expr.args[2]
    is_template = occursin(r"\{.*\}", string(uri))
    if is_template
        quote
            UniLM.register_resource_template!($(esc(server)), $(esc(uri)), $(esc(uri)),
                function(_p_::Dict{String,String}); $(esc(body)); end)
        end
    else
        quote
            UniLM.register_resource!($(esc(server)), $(esc(uri)), $(esc(uri)),
                function(); $(esc(body)); end)
        end
    end
end

"""
    @mcp_prompt server name function(args...) body end

Register a prompt on `server`. The handler should return a Vector of message Dicts.

# Example
```julia
@mcp_prompt server "review" function(code::String)
    [Dict("role" => "user", "content" => Dict("type" => "text", "text" => "Review: \$code"))]
end
```
"""
macro mcp_prompt(server, name, func_expr)
    func_expr.head in (:function, :(=)) || error("@mcp_prompt expects a function definition")
    call_expr = func_expr.args[1]
    body = func_expr.args[2]
    actual_call = call_expr isa Expr && call_expr.head == :(::) ? call_expr.args[1] : call_expr
    raw_args = actual_call.args[2:end]
    arg_defs = Dict{String,Any}[]
    for arg in raw_args
        n = arg isa Expr && arg.head == :(::) ? string(arg.args[1]) : string(arg)
        req = arg isa Expr && arg.head == :(::)
        push!(arg_defs, Dict{String,Any}("name" => n, "required" => req))
    end
    # Build handler that unpacks dict to local vars
    arg_names = [arg isa Expr && arg.head == :(::) ? arg.args[1] : arg for arg in raw_args]
    name_strs = [string(a) for a in arg_names]
    unpack = [:($(esc(a)) = get(_d_, $(n), nothing)) for (a, n) in zip(arg_names, name_strs)]
    quote
        UniLM.register_prompt!($(esc(server)), $(esc(name)),
            function(_d_::Dict{String,Any})
                $(unpack...)
                $(esc(body))
            end;
            arguments=$(arg_defs))
    end
end

# ─── Bridge: MCPServerTool ↔ UniLM tool types ───────────────────────────────

"""Convert an MCPServerTool to a FunctionTool for use with the Responses API."""
to_tool(t::MCPServerTool) = FunctionTool(
    name=t.name, description=t.description, parameters=t.input_schema)
