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
    # Split on {param} placeholders, escape literal parts for regex safety
    parts = split(template, r"\{\w+\}"; keepempty=true)
    params = [m.match for m in eachmatch(r"\{(\w+)\}", template)]
    buf = IOBuffer()
    for (i, part) in enumerate(parts)
        write(buf, replace(part, r"([.+*?^\$|\\()\[\]{}])" => s"\\\1"))
        if i <= length(params)
            name = params[i][2:end-1]
            push!(param_names, name)
            write(buf, "(?P<$name>[^/]+)")
        end
    end
    (Regex("^" * takestring!(buf) * "\$"), param_names)
end

# ─── MCPServer ───────────────────────────────────────────────────────────────

"""
    MCPServer(name, version; description=nothing)

An MCP server that can host tools, resources, and prompts.

Register primitives via [`register_tool!`](@ref), [`register_resource!`](@ref),
[`register_prompt!`](@ref), or the `@mcp_tool`, `@mcp_resource`, `@mcp_prompt` macros.
Registration is thread-safe: it may run while [`serve`](@ref) is dispatching.

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
    # Guards the four registries above: registration may run while requests are
    # dispatched on other threads (a Dict rehash under a concurrent read corrupts the
    # heap). Held only to read or write a registry, never across a handler call.
    const lock::ReentrantLock
end

function MCPServer(name::String, version::String; description::Union{String,Nothing}=nothing)
    MCPServer(name, version, description,
        Dict{String,MCPServerTool}(),
        Dict{String,MCPServerResource}(),
        MCPServerResourceTemplate[],
        Dict{String,MCPServerPrompt}(),
        false,
        ReentrantLock())
end

# ─── Registration ────────────────────────────────────────────────────────────

"""
    register_tool!(server, name, description, input_schema, handler)

Register a tool on the MCP server with an explicit JSON Schema.

`handler(args::Dict{String,Any})` receives the client's `arguments` object (a
`tools/call` whose `arguments` is not an object is answered with JSON-RPC `-32602`
and never reaches it). An exception it raises is NOT a protocol error: the message is
relayed to the client as tool content with `isError: true`, so a model can see the
failure and correct itself. Write handlers with that in mind — raise with a message
you are willing to show both the model and the client, and never one carrying secrets
or internals. Errors below the handler (in dispatch itself) answer a generic JSON-RPC
`-32603` instead, with the detail going to the server's logs. An `InterruptException`
is never converted: it propagates.

Over HTTP, handlers run concurrently — one task per request, on the default thread
pool — so a handler must be thread-safe. Over stdio they run one at a time, with the
process `stdout` pointed at `stderr` (see [`serve`](@ref)).
"""
function register_tool!(server::MCPServer, name::String,
                        description::Union{String,Nothing},
                        input_schema::Dict{String,Any},
                        handler::Function)
    tool = MCPServerTool(name, description, input_schema, handler)
    @lock server.lock server.tools[name] = tool
    server
end

"""
    register_tool!(server, name, description, handler)

Register a tool whose input schema is inferred from `handler`'s positional parameters
(names and declared types of its first method). Each `tools/call` binds the client's
`arguments` object to those parameters BY NAME — the binding [`@mcp_tool`](@ref)
generates: a parameter whose type admits `nothing` (`Union{T,Nothing}`) is optional
and binds `nothing` when omitted, every other parameter is required, and a value must
have its parameter's JSON type (a string for `String`, an integral number for an
integer type, a number for a float type, a boolean for `Bool`). A call that violates
this never reaches the handler: it is answered with a tool result carrying
`isError: true` and a text naming the argument, the MCP report for an input
validation error, so the model can correct its call.

A handler taking one dictionary is the explicit-schema calling convention, and a
variadic handler has no parameter names to bind: both are rejected with an
`ArgumentError` — register them with an explicit `input_schema`.

# Example
```julia
register_tool!(server, "add", "Add two integers", (a::Int, b::Int) -> a + b)
```
"""
function register_tool!(server::MCPServer, name::String,
                        description::Union{String,Nothing},
                        handler::Function)
    params = something(_handler_params(handler), _HandlerParam[])
    if length(params) == 1 && params[1].type isa Type && params[1].type <: AbstractDict
        throw(ArgumentError("register_tool!(server, \"$name\", description, handler): the " *
            "handler takes its arguments as one dictionary, but a schema inferred from its " *
            "signature binds arguments by name to positional parameters. Pass the schema " *
            "explicitly: register_tool!(server, name, description, input_schema, handler)."))
    end
    any(p -> Base.isvarargtype(p.type), params) &&
        throw(ArgumentError("register_tool!(server, \"$name\", description, handler): a " *
            "variadic handler has no parameter names to bind arguments to. Pass an " *
            "input_schema: register_tool!(server, name, description, input_schema, handler)."))
    register_tool!(server, name, description, _function_schema(handler),
                   _by_name_handler(handler, params))
end

"""
    register_tool!(server, ct::CallableTool{Tool})

Register a `CallableTool{Tool}` on the MCP server, bridging from UniLM's
Chat Completions tool type.
"""
function register_tool!(server::MCPServer, ct::CallableTool{Tool})
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
    resource = MCPServerResource(uri, name, description, mime_type, handler)
    @lock server.lock server.resources[uri] = resource
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
    template = MCPServerResourceTemplate(uri_template, name, description, mime_type,
                                         handler, pattern, param_names)
    @lock server.lock push!(server.resource_templates, template)
    server
end

"""
    register_prompt!(server, name, handler; description=nothing, arguments=Dict{String,Any}[])

Register a prompt template on the MCP server.
"""
function register_prompt!(server::MCPServer, name::String, handler::Function;
                          description::Union{String,Nothing}=nothing,
                          arguments::Vector{Dict{String,Any}}=Dict{String,Any}[])
    prompt = MCPServerPrompt(name, description, arguments, handler)
    @lock server.lock server.prompts[name] = prompt
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

# Registry reads take a snapshot under `server.lock`; entries are built and handlers run
# outside it, so a slow handler never blocks registration or other requests.

function _handle_initialize(server::MCPServer, id, params::Dict{String,Any})
    caps = Dict{String,Any}()
    @lock server.lock begin
        isempty(server.tools) || (caps["tools"] = Dict{String,Any}())
        (isempty(server.resources) && isempty(server.resource_templates)) ||
            (caps["resources"] = Dict{String,Any}())
        isempty(server.prompts) || (caps["prompts"] = Dict{String,Any}())
        server._initialized = true
    end
    server_info = Dict{String,Any}("name" => server.name, "version" => server.version)
    !isnothing(server.description) && (server_info["description"] = server.description)
    # Lifecycle: answer the requested revision when it is supported, else the latest one.
    requested = get(params, "protocolVersion", nothing)
    _jsonrpc_result(id, Dict{String,Any}(
        "protocolVersion" => requested in _MCP_SUPPORTED_PROTOCOL_VERSIONS ? requested : _MCP_PROTOCOL_VERSION,
        "capabilities" => caps,
        "serverInfo" => server_info
    ))
end

function _handle_tools_list(server::MCPServer, id, params::Dict{String,Any})
    tools_list = [begin
        d = Dict{String,Any}("name" => t.name, "inputSchema" => t.input_schema)
        !isnothing(t.description) && (d["description"] = t.description)
        d
    end for t in @lock(server.lock, collect(values(server.tools)))]
    _jsonrpc_result(id, Dict{String,Any}("tools" => tools_list))
end

function _handle_tools_call(server::MCPServer, id, params::Dict{String,Any})
    name = get(params, "name", nothing)
    isnothing(name) && return _jsonrpc_error(id, -32602, "Missing required parameter: name")
    args = get(params, "arguments", Dict{String,Any}())
    args isa AbstractDict ||
        return _jsonrpc_error(id, -32602, "Invalid params: `arguments` must be an object")
    tool = @lock server.lock get(server.tools, name, nothing)
    isnothing(tool) && return _jsonrpc_error(id, -32602, "Unknown tool: $name")
    try
        result = tool.handler(_mcp_arguments(args))
        _jsonrpc_result(id, Dict{String,Any}(
            "content" => _format_tool_result(result), "isError" => false))
    catch e
        e isa InterruptException && rethrow()
        # A tool execution error — the handler's own, or an argument the by-name binding
        # rejected (MCP 2025-11-25 server/tools classes input validation errors as tool
        # execution errors) — is a result the model can read and correct its call from.
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
    end for r in @lock(server.lock, collect(values(server.resources)))]
    _jsonrpc_result(id, Dict{String,Any}("resources" => resources_list))
end

function _handle_resources_templates_list(server::MCPServer, id, params::Dict{String,Any})
    templates_list = [begin
        d = Dict{String,Any}("uriTemplate" => t.uri_template, "name" => t.name)
        !isnothing(t.description) && (d["description"] = t.description)
        d
    end for t in @lock(server.lock, copy(server.resource_templates))]
    _jsonrpc_result(id, Dict{String,Any}("resourceTemplates" => templates_list))
end

# A resource or prompt handler that throws is a server fault, not a result: it propagates
# to `_dispatch_guarded`, which answers a generic -32603 and logs the detail locally.
function _handle_resources_read(server::MCPServer, id, params::Dict{String,Any})
    uri = get(params, "uri", nothing)
    isnothing(uri) && return _jsonrpc_error(id, -32602, "Missing required parameter: uri")
    resource, templates = @lock server.lock (get(server.resources, uri, nothing),
                                             copy(server.resource_templates))
    # Static resources first, then the templates in registration order.
    if !isnothing(resource)
        content = _format_resource_content(uri, resource.mime_type, resource.handler())
        return _jsonrpc_result(id, Dict{String,Any}("contents" => [content]))
    end
    for tmpl in templates
        m = match(tmpl._pattern, uri)
        isnothing(m) && continue
        params_dict = Dict{String,String}(name => m[name] for name in tmpl._param_names)
        content = _format_resource_content(uri, tmpl.mime_type, tmpl.handler(params_dict))
        return _jsonrpc_result(id, Dict{String,Any}("contents" => [content]))
    end
    _jsonrpc_error(id, -32002, "Resource not found: $uri")
end

function _handle_prompts_list(server::MCPServer, id, params::Dict{String,Any})
    prompts_list = [begin
        d = Dict{String,Any}("name" => p.name)
        !isnothing(p.description) && (d["description"] = p.description)
        !isempty(p.arguments) && (d["arguments"] = p.arguments)
        d
    end for p in @lock(server.lock, collect(values(server.prompts)))]
    _jsonrpc_result(id, Dict{String,Any}("prompts" => prompts_list))
end

function _handle_prompts_get(server::MCPServer, id, params::Dict{String,Any})
    name = get(params, "name", nothing)
    isnothing(name) && return _jsonrpc_error(id, -32602, "Missing required parameter: name")
    args = get(params, "arguments", Dict{String,Any}())
    prompt = @lock server.lock get(server.prompts, name, nothing)
    isnothing(prompt) && return _jsonrpc_error(id, -32602, "Unknown prompt: $name")
    _jsonrpc_result(id, Dict{String,Any}("messages" => prompt.handler(args)))
end

"""
Normalize the `params` member of a JSON-RPC request into the by-name `Dict` every
handler takes. `params` is optional and may be `null` (both mean "no arguments"),
and JSON-RPC also permits a positional array — which no method here accepts, so an
array yields `nothing` and the caller answers Invalid params instead of failing to
dispatch.
"""
_mcp_named_params(::Nothing) = Dict{String,Any}()
_mcp_named_params(p::Dict{String,Any}) = p
_mcp_named_params(p::AbstractDict) = Dict{String,Any}(p)
_mcp_named_params(::Any) = nothing

"""Route a parsed JSON-RPC request to the appropriate handler."""
function _dispatch_mcp(server::MCPServer, parsed::Dict{String,Any})
    id = get(parsed, "id", nothing)
    # A notification (no id) and a response (an id but no method: the reply to a
    # server-initiated request, which this server never sends) are accepted silently.
    (isnothing(id) || !haskey(parsed, "method")) && return nothing
    method = parsed["method"]
    params = _mcp_named_params(get(parsed, "params", nothing))
    isnothing(params) && return _jsonrpc_error(id, -32602,
        "Invalid params: expected an object of by-name arguments")
    if method == "initialize"
        _handle_initialize(server, id, params)
    elseif method == "tools/list"
        _handle_tools_list(server, id, params)
    elseif method == "tools/call"
        _handle_tools_call(server, id, params)
    elseif method == "resources/list"
        _handle_resources_list(server, id, params)
    elseif method == "resources/templates/list"
        _handle_resources_templates_list(server, id, params)
    elseif method == "resources/read"
        _handle_resources_read(server, id, params)
    elseif method == "prompts/list"
        _handle_prompts_list(server, id, params)
    elseif method == "prompts/get"
        _handle_prompts_get(server, id, params)
    elseif method == "ping"
        _jsonrpc_result(id, Dict{String,Any}())
    else
        _jsonrpc_error(id, -32601, "Method not found: $method")
    end
end

# ─── Transports ──────────────────────────────────────────────────────────────

"""
Dispatch one request, converting any unhandled error into an Internal error
response so a single bad frame cannot take the transport down with it. The client
gets a generic message — an exception string can carry file paths, argument values
and other server internals that a remote peer has no business reading — while the
exception and its backtrace are logged locally.

Handler-raised tool errors are NOT routed here: those reach the client as tool
results (`isError`), which is what lets a model see and correct its own mistake.
"""
function _dispatch_guarded(server::MCPServer, parsed::Dict{String,Any})
    try
        _dispatch_mcp(server, parsed)
    catch e
        e isa InterruptException && rethrow()
        @error "MCP request dispatch failed" method=get(parsed, "method", "") exception=(e, catch_backtrace())
        id = get(parsed, "id", nothing)
        isnothing(id) ? nothing : _jsonrpc_error(id, -32603, "Internal error")
    end
end

# Largest frame accepted on either transport. JSON.parse of an attacker-sized
# payload allocates a multiple of the payload itself, so an unbounded frame is an
# out-of-memory kill, not a protocol error. 16 MiB sits far above any legitimate
# message (the biggest realistic frame is a base64 resource blob) and far below a
# heap-exhausting one.
const _MCP_MAX_FRAME_BYTES = 16 * 1024 * 1024

"""
Read one newline-delimited frame from `input`, keeping at most `limit` bytes.
Bytes past the limit are drained and discarded rather than buffered, so an
oversized frame costs a bounded amount of memory instead of its own size.
Returns the line (trailing CRLF removed, as `readline` does) and whether it
overflowed the limit.
"""
function _read_frame(input::IO, limit::Int)
    buf = IOBuffer()
    kept = 0
    overflow = false
    while !eof(input)
        b = read(input, UInt8)
        b == UInt8('\n') && break
        if kept < limit
            write(buf, b)
            kept += 1
        else
            overflow = true
        end
    end
    line = takestring!(buf)
    endswith(line, '\r') && (line = line[1:end-1])
    (line, overflow)
end

"""
    _serve_stdio(server::MCPServer; input=stdin, output=stdout)

Run the MCP server over stdio. Reads JSON-RPC messages from `input` (one per line),
dispatches them, and writes responses to `output`. Diagnostic logs go to stderr.

The loop survives every malformed frame: parse errors, wrong-shape frames,
oversized frames and internal dispatch failures are all answered as JSON-RPC
errors and serving continues.

The stdio transport forbids anything but protocol messages on stdout, yet a handler
that prints (`println`, `@show`, a library's progress output, C code) writes to the
process stdout. When `output` is the process stdout, it is kept as the protocol
stream and the process stdout points at stderr while serving.
"""
function _serve_stdio(server::MCPServer; input::IO=stdin, output::IO=stdout)
    output === stdout || return _serve_frames(server, input, output)
    redirect_stdout(() -> _serve_frames(server, input, output), stderr)
end

function _serve_frames(server::MCPServer, input::IO, output::IO)
    while !eof(input)
        line, overflow = _read_frame(input, _MCP_MAX_FRAME_BYTES)
        if overflow
            response = _jsonrpc_error(nothing, -32600, "Invalid Request: frame exceeds $(_MCP_MAX_FRAME_BYTES) bytes")
            println(output, JSON.json(response))
            flush(output)
            continue
        end
        isempty(strip(line)) && continue
        parsed = try
            JSON.parse(line; dicttype=Dict{String,Any})
        catch e
            response = _jsonrpc_error(nothing, -32700, "Parse error: $(sprint(showerror, e))")
            println(output, JSON.json(response))
            flush(output)
            continue
        end
        # JSON-RPC messages are single objects (the MCP spec removed batch
        # arrays); any other JSON value is answerable only as Invalid Request.
        if !(parsed isa Dict{String,Any})
            response = _jsonrpc_error(nothing, -32600, "Invalid Request: expected a single JSON-RPC object")
            println(output, JSON.json(response))
            flush(output)
            continue
        end
        response = _dispatch_guarded(server, parsed)
        # Notifications produce no response
        isnothing(response) && continue
        println(output, JSON.json(response))
        flush(output)
    end
end

# Localhost origins (any port, either scheme, IPv4/IPv6/name) are always
# allowed: the Origin check exists to stop DNS-rebinding from foreign web
# origins, and pages served from the developer's own machine are not that
# threat. Matching is anchored, so lookalikes ("localhost.evil.example") fail.
const _MCP_LOCALHOST_ORIGIN = r"^https?://(localhost|127\.0\.0\.1|\[::1\])(:\d+)?$"i

"""True when `origin` is a localhost origin or an exact member of `allowed_origins`."""
_mcp_origin_allowed(origin::AbstractString, allowed_origins::Vector{String})::Bool =
    origin in allowed_origins || occursin(_MCP_LOCALHOST_ORIGIN, origin)

# Payload byte count of a request body already buffered by the transport. HTTP.jl
# hands a request handler a body object whose `length` is the payload size —
# `sizeof` would measure that wrapper struct instead of the bytes. A request that
# carried no payload gets a separate zero-length representation that answers no
# `length` at all, so it needs its own arm; without one a bodyless POST fails to
# dispatch and takes the exchange down with a transport error instead of a
# JSON-RPC reply.
_request_body_bytes(::HTTP.EmptyBody)::Int = 0
_request_body_bytes(body::HTTP.AbstractBody)::Int = length(body)

# Streamable HTTP: every request after `initialize` carries the negotiated revision in
# `MCP-Protocol-Version`, and a server receiving an unsupported one MUST answer 400. A
# request without the header is accepted (the spec's backwards-compatible default).
function _unsupported_protocol_version(req::HTTP.Request)::Union{HTTP.Response,Nothing}
    v = HTTP.header(req, "MCP-Protocol-Version", "")
    (isempty(v) || v in _MCP_SUPPORTED_PROTOCOL_VERSIONS) && return nothing
    HTTP.Response(400, "Bad Request: unsupported MCP-Protocol-Version \"$v\" " *
        "(supported: $(join(_MCP_SUPPORTED_PROTOCOL_VERSIONS, ", ")))")
end

# Requests that run a registered handler. HTTP.jl runs each request handler on its
# :interactive pool, whose first thread also drives the event loop: a busy handler
# there stalls every other request, Timer and IO completion in the process. These run
# on the default pool instead; protocol requests (initialize, lists, ping) stay inline,
# so they answer promptly even while handlers occupy the default pool.
const _MCP_HANDLER_METHODS = ("tools/call", "resources/read", "prompts/get")

function _dispatch_http(server::MCPServer, parsed::Dict{String,Any})
    get(parsed, "method", nothing) in _MCP_HANDLER_METHODS || return _dispatch_guarded(server, parsed)
    task = Threads.@spawn :default _dispatch_guarded(server, parsed)
    try
        fetch(task)
    catch e
        e isa InterruptException && rethrow()
        throw(_unwrap_task_failure(e))   # the handler's own exception, as when run inline
    end
end

"""
    _serve_http(server::MCPServer; host="127.0.0.1", port=8080, allowed_origins=String[], block=true)

Run the MCP server over HTTP. POST requests contain JSON-RPC messages.
Requests carrying a non-localhost, non-allowlisted `Origin` header get 403; a request
other than `initialize` whose `MCP-Protocol-Version` header names an unsupported
revision gets 400.

Blocks until the server is closed. With `block=false`, returns the running
server immediately; close it with `close`.
"""
function _serve_http(server::MCPServer; host::String="127.0.0.1", port::Int=8080,
                     allowed_origins::Vector{String}=String[], block::Bool=true)
    # Request-handler form (HTTP.jl `stream=false`, the default): the handler receives a
    # buffered `HTTP.Request` and returns an `HTTP.Response`. Annotating the argument keeps
    # `HTTP.header(req, …)` — defined for `Request`/`Response`, not `Stream` — well-typed.
    http_server = HTTP.serve!(host, port) do req::HTTP.Request
        # Streamable HTTP requires Origin validation (DNS-rebinding defense).
        # Requests without an Origin header are not browser cross-origin
        # requests and pass; browser requests must come from localhost or an
        # allowlisted origin. Checked before any method dispatch.
        origin = HTTP.header(req, "Origin", "")
        if !isempty(origin) && !_mcp_origin_allowed(origin, allowed_origins)
            return HTTP.Response(403, "Forbidden: Origin not allowed")
        end
        if req.method == "POST"
            # Reject an oversized body before it is copied into a String and parsed
            # (see _MCP_MAX_FRAME_BYTES and _request_body_bytes). The transport
            # buffers the request before the handler runs, so this bounds the parse
            # rather than the read.
            if _request_body_bytes(req.body) > _MCP_MAX_FRAME_BYTES
                return HTTP.Response(413, "Payload Too Large")
            end
            body = String(req.body)
            parsed = try
                JSON.parse(body; dicttype=Dict{String,Any})
            catch
                return HTTP.Response(400, JSON.json(_jsonrpc_error(nothing, -32700, "Parse error")))
            end
            if !(parsed isa Dict{String,Any})
                return HTTP.Response(400, JSON.json(_jsonrpc_error(nothing, -32600, "Invalid Request: expected a single JSON-RPC object")))
            end
            # `initialize` negotiates its revision in the body, so its header is not checked.
            if get(parsed, "method", nothing) != "initialize"
                refusal = _unsupported_protocol_version(req)
                isnothing(refusal) || return refusal
            end
            response = _dispatch_http(server, parsed)
            if isnothing(response)
                return HTTP.Response(202, "")
            end
            HTTP.Response(200, ["Content-Type" => "application/json"], JSON.json(response))
        else
            refusal = _unsupported_protocol_version(req)
            isnothing(refusal) || return refusal
            req.method == "DELETE" ? HTTP.Response(200, "") : HTTP.Response(405, "Method Not Allowed")
        end
    end
    block || return http_server
    # Same shape as HTTP.serve: block on the running server and close it even if
    # the wait is interrupted (e.g. Ctrl-C).
    try
        wait(http_server)
    finally
        close(http_server)
    end
    nothing
end

"""
    serve(server::MCPServer; transport=:stdio, kwargs...)

Start the MCP server using the specified transport.

# Transports
- `:stdio` (default): Read from stdin, write to stdout. For Claude Desktop/CLI
  integration. Accepts `input`/`output` IO overrides; returns after EOF on `input`.
  Handlers run one at a time. While serving on the process stdout, the process
  `stdout` points at `stderr`, so a handler that prints cannot corrupt the frame
  stream (only protocol messages may appear on a stdio server's stdout).
- `:http`: HTTP server. Accepts `host` (default `"127.0.0.1"`), `port`
  (default `8080`), `allowed_origins` (extra allowed `Origin` header values —
  localhost origins and requests without an `Origin` are always accepted;
  anything else gets 403), and `block` (default `true`: block until the server
  is closed; `block=false` returns the running server — close it with `close`).
  Handlers run concurrently, one task per request on the default thread pool, so
  they must be thread-safe; protocol requests (`initialize`, lists, `ping`) are
  answered inline and stay responsive while handlers are busy. A request other than
  `initialize` carrying an unsupported `MCP-Protocol-Version` header gets 400.

`initialize` answers the protocol revision the client requested when it is supported,
otherwise the latest supported one.

# Robustness contract
Both transports cap an incoming frame (stdio) or request body (HTTP) at 16 MiB and
answer an oversized one with JSON-RPC `-32600` rather than parsing it — parsing an
attacker-sized payload allocates a multiple of it, which is an out-of-memory kill
rather than a protocol error. Any unhandled error while dispatching a request —
including an exception from a resource or prompt handler — is answered with a
generic `-32603` "Internal error" and logged locally, so one bad frame cannot take
the transport down and no exception text (file paths, argument values) reaches the
peer. Tool-handler exceptions are excluded: they reach the client as tool results —
see [`register_tool!`](@ref). An `InterruptException` is never converted into an
answer: it propagates. A JSON-RPC response sent to the server is accepted without
an answer (HTTP: 202).

# Examples
```julia
serve(server)                                    # stdio (default)
serve(server; transport=:http, port=3000)        # HTTP; blocks until closed
h = serve(server; transport=:http, block=false)  # HTTP; returns the server
close(h)
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

# Macro-expansion-time helpers (operate on `Expr`, not runtime values).

"""
    _mcp_arg_name(arg) -> Symbol

Name of a declared argument node. `:a` → `:a`; `Expr(:(::), :a, T)` → `:a`.
"""
_mcp_arg_name(arg::Symbol) = arg
function _mcp_arg_name(arg::Expr)
    arg.head === :(::) || error("@mcp_* unsupported argument form: $(arg)")
    arg.args[1]::Symbol
end

"""
    _mcp_arg_type(arg) -> type expression

Declared type of an argument node. Untyped (`:a`) → `:Any`; typed
`Expr(:(::), :a, T)` → `T`. "required"/typedness is `type !== :Any`, matching the
pre-existing macro logic.
"""
_mcp_arg_type(arg::Symbol) = :Any
_mcp_arg_type(arg::Expr) = (arg.head === :(::) ? arg.args[2] : :Any)

"""
    _mcp_sig_args(sig) -> Vector

Extract the declared-argument nodes from a function signature `Expr`
(`func_expr.args[1]`), independent of whether the function was written in named
or anonymous form. Handles every shape Julia produces:

  - Named `f(a, b)`            → `Expr(:call, :f, args...)` ⇒ `args[2:end]` (drop name).
  - Anonymous multi `(a, b)`   → `Expr(:tuple, args...)`    ⇒ all of `.args`.
  - Anonymous single `(a)`     → bare `Symbol` `:a` OR typed `Expr(:(::), :a, T)`
                                  ⇒ the whole node is the single arg (`[sig]`).

A trailing return-type annotation (`… :: RetType`) is unwrapped first: it appears
as `Expr(:(::), realsig, RetType)` whose inner `realsig` is itself a `:call`
(named) or `:tuple` (anonymous). A top-level `Expr(:(::), :sym, T)` with a *Symbol*
inner node is instead a typed single anonymous arg and is NOT unwrapped.
"""
function _mcp_sig_args(sig)
    # Unwrap a trailing return type only when the inner node is a real signature
    # (`:call`/`:tuple`), never when it is the typed-single-arg `a::T` case.
    if sig isa Expr && sig.head === :(::) && sig.args[1] isa Expr &&
       sig.args[1].head in (:call, :tuple)
        sig = sig.args[1]
    end
    if sig isa Expr && sig.head === :call
        return sig.args[2:end]            # named: drop the function name
    elseif sig isa Expr && sig.head === :tuple
        return collect(sig.args)          # anonymous (zero or more args)
    else
        return Any[sig]                   # anonymous single arg: bare Symbol or `a::T`
    end
end

"""
    @mcp_tool server function name(args...)::ReturnType body end

Register a tool on `server` with auto-generated JSON Schema from the function signature.

Each `tools/call` binds the `arguments` object to the parameters BY NAME. A typed
parameter is required; an untyped one is optional and binds `nothing` when omitted. A
value must have its parameter's JSON type — a string for `String`, an integral number
for an integer type (`5.0` binds `5`), a number for a float type, a boolean for `Bool`;
other declared types pass through unconverted. A missing required argument or a value
of the wrong type never reaches the function: it is answered with a tool result
carrying `isError: true` whose text names the argument.

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
    # @mcp_tool registers under (and redefines) the function NAME, so it requires a
    # named signature `f(args...)`. Reject the anonymous form loudly instead of
    # registering a tool named after the first argument node.
    (actual_call isa Expr && actual_call.head === :call) ||
        error("@mcp_tool requires a NAMED function `function name(args...) … end`")
    fname = actual_call.args[1]
    raw_args = _mcp_sig_args(call_expr)
    name_str = string(fname)
    # Extract arg names and types (shared, AST-shape-robust extractor).
    arg_info = [(string(_mcp_arg_name(a)), _mcp_arg_type(a)) for a in raw_args]
    required = [n for (n, T) in arg_info if T !== :Any]
    # Build schema and handler with full esc to avoid hygiene issues
    prop_exprs = [:($(n) => UniLM._json_schema_type($(T))) for (n, T) in arg_info]
    names = String[n for (n, _) in arg_info]
    types = Expr(:tuple, (T for (_, T) in arg_info)...)
    typed = Bool[T !== :Any for (_, T) in arg_info]
    quote
        # Define the function in caller's scope
        function $(esc(fname))($(map(esc, raw_args)...))
            $(esc(body))
        end
        # Register with schema and the by-name binding wrapper
        UniLM.register_tool!($(esc(server)), $name_str, nothing,
            Dict{String,Any}("type" => "object",
                "properties" => Dict{String,Any}($(prop_exprs...)),
                "required" => $required),
            function(_d_::Dict{String,Any})
                $(esc(fname))(UniLM._mcp_bind(_d_, $names, $types, $typed)...)
            end)
    end
end

"""A `tools/call` whose arguments violate the tool's by-name binding — a missing
required argument or a value of the wrong JSON type. An input validation error, which
MCP reports as a tool execution error: a result with `isError: true` whose text
names the argument."""
struct _MCPInvalidArguments <: Exception
    msg::String
end
Base.showerror(io::IO, e::_MCPInvalidArguments) = print(io, "Invalid arguments: ", e.msg)

"""Bind a `tools/call` `arguments` object to positional parameters BY NAME, converting
each value to its declared type. An absent argument binds `nothing` unless it is
required; a parameter whose name starts with `#` (ignored: `_`, unnamed) binds `nothing`."""
function _mcp_bind(args::AbstractDict, names::Vector{String}, types::Tuple,
                   required::Vector{Bool})::Tuple
    ntuple(length(names)) do i
        name = names[i]
        startswith(name, '#') && return nothing
        haskey(args, name) || return required[i] ?
            throw(_MCPInvalidArguments("missing required argument `$name`")) : nothing
        _mcp_convert(types[i], args[name], name)
    end
end

"""The by-name wrapper the inferred-schema `register_tool!` registers for `handler`:
the same binding [`@mcp_tool`](@ref) generates, with `required` taken from the inferred
schema (a parameter is optional when its type admits `nothing`)."""
function _by_name_handler(handler::Function, params::Vector{_HandlerParam})::Function
    names = [_bound_by_name(p) ? p.name : "#" for p in params]
    types = Tuple(p.type for p in params)
    required = [_bound_by_name(p) && !first(_is_optional(p.type)) for p in params]
    (args::Dict{String,Any}) -> handler(_mcp_bind(args, names, types, required)...)
end

_invalid_arg(name::String, expected::String, v) =
    throw(_MCPInvalidArguments("argument `$name` must be $expected, got $(_json_kind(v))"))
_json_kind(v) = v === nothing ? "null" : v isa Bool ? "a boolean" : v isa Number ? "a number" :
                v isa AbstractString ? "a string" : v isa AbstractVector ? "an array" : "an object"

"""Convert one JSON argument to its parameter's declared type. Validating, not coercing:
42 is not a `String`, 2.5 is not an `Int`, 1 is not a `Bool`."""
_mcp_convert(::Type{String}, v, name::String) =
    v isa AbstractString ? String(v) : _invalid_arg(name, "a string", v)
function _mcp_convert(::Type{T}, v, name::String) where {T<:Integer}
    (v isa Real && !(v isa Bool) && isinteger(v)) || _invalid_arg(name, "an integer", v)
    try
        convert(T, v)
    catch e
        e isa InexactError || rethrow()
        _invalid_arg(name, "an integer representable as $T", v)
    end
end
_mcp_convert(::Type{T}, v, name::String) where {T<:AbstractFloat} =
    v isa Real && !(v isa Bool) ? convert(T, v) : _invalid_arg(name, "a number", v)
_mcp_convert(::Type{Bool}, v, name::String) = v isa Bool ? v : _invalid_arg(name, "a boolean", v)
# Any other declared type: a value of that type binds as is, `nothing` binds an optional
# parameter, and anything else passes through for the function's own method to accept.
function _mcp_convert(@nospecialize(T), v, name::String)
    v isa T && return v
    optional, inner = _is_optional(T)
    (optional && inner !== T) || return v
    v === nothing ? nothing : _mcp_convert(inner, v, name)
end

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
        # Bind each declared arg from the matched path params `_p_`, keyed by the
        # arg's NAME (which must equal the URI `{param}` name, per the docstring).
        # esc the binding LHS so it is visible to the esc'd user body. A declared
        # name with no matching URI param raises a clear KeyError at read time.
        raw_args = _mcp_sig_args(func_expr.args[1])
        unpack = [:($(esc(_mcp_arg_name(a))) = _p_[$(string(_mcp_arg_name(a)))]) for a in raw_args]
        quote
            UniLM.register_resource_template!($(esc(server)), $(esc(uri)), $(esc(uri)),
                function(_p_::Dict{String,String})
                    $(unpack...)
                    $(esc(body))
                end)
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
    body = func_expr.args[2]
    # Shared, AST-shape-robust arg extraction: works for the anonymous
    # `function(x) … end` form (the docstring's shape) AND the named form.
    raw_args = _mcp_sig_args(func_expr.args[1])
    arg_names = [_mcp_arg_name(a) for a in raw_args]
    name_strs = [string(n) for n in arg_names]
    # required ⇔ the arg is typed (type !== :Any), matching the prior logic.
    arg_defs = [Dict{String,Any}("name" => s, "required" => _mcp_arg_type(a) !== :Any)
                for (a, s) in zip(raw_args, name_strs)]
    # Build handler that unpacks the request dict into local (esc'd) vars.
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
