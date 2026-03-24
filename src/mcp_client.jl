# ============================================================================
# MCP Client — Model Context Protocol client for UniLM.jl
# Connects to MCP servers (stdio or HTTP), discovers tools, and bridges them
# into CallableTool for seamless tool_loop! / tool_loop integration.
#
# Protocol: JSON-RPC 2.0 over stdio or Streamable HTTP (spec 2025-11-25)
# ============================================================================

# ─── JSON-RPC 2.0 Framing (internal) ────────────────────────────────────────

const _JSONRPC_VERSION = "2.0"
const _MCP_PROTOCOL_VERSION = "2025-11-25"

struct _JSONRPCRequest
    id::Union{Int,String}
    method::String
    params::Union{Dict{String,Any},Nothing}
end

function _jsonrpc_serialize(r::_JSONRPCRequest)::String
    d = Dict{String,Any}("jsonrpc" => _JSONRPC_VERSION, "id" => r.id, "method" => r.method)
    !isnothing(r.params) && (d["params"] = r.params)
    JSON.json(d)
end

struct _JSONRPCNotification
    method::String
    params::Union{Dict{String,Any},Nothing}
end

function _jsonrpc_serialize(n::_JSONRPCNotification)::String
    d = Dict{String,Any}("jsonrpc" => _JSONRPC_VERSION, "method" => n.method)
    !isnothing(n.params) && (d["params"] = n.params)
    JSON.json(d)
end

struct _JSONRPCResponse
    id::Union{Int,String,Nothing}
    result::Union{Dict{String,Any},Nothing}
    error::Union{Dict{String,Any},Nothing}
end

function _JSONRPCResponse(d::Dict{String,Any})
    _JSONRPCResponse(get(d, "id", nothing), get(d, "result", nothing), get(d, "error", nothing))
end

# ─── MCP Error ───────────────────────────────────────────────────────────────

"""
    MCPError <: Exception

Error from MCP protocol operations. Contains the JSON-RPC error code, message,
and optional data from the server.
"""
struct MCPError <: Exception
    code::Int
    message::String
    data::Union{Dict{String,Any},Nothing}
end

MCPError(d::Dict{String,Any}) = MCPError(d["code"], d["message"], get(d, "data", nothing))

function Base.showerror(io::IO, e::MCPError)
    print(io, "MCPError($(e.code)): $(e.message)")
    !isnothing(e.data) && print(io, " data=", e.data)
end

# ─── MCP Types ───────────────────────────────────────────────────────────────

"""
    MCPServerCapabilities

Capabilities declared by an MCP server during initialization.
"""
struct MCPServerCapabilities
    tools::Union{Dict{String,Any},Nothing}
    resources::Union{Dict{String,Any},Nothing}
    prompts::Union{Dict{String,Any},Nothing}
    logging::Union{Dict{String,Any},Nothing}
end

function MCPServerCapabilities(d::Dict{String,Any})
    MCPServerCapabilities(
        get(d, "tools", nothing),
        get(d, "resources", nothing),
        get(d, "prompts", nothing),
        get(d, "logging", nothing)
    )
end

MCPServerCapabilities() = MCPServerCapabilities(nothing, nothing, nothing, nothing)

"""
    MCPToolInfo

A tool definition received from an MCP server via `tools/list`.
Convert to `CallableTool` via [`mcp_tools`](@ref) or [`mcp_tools_respond`](@ref).
"""
struct MCPToolInfo
    name::String
    description::Union{String,Nothing}
    input_schema::Union{Dict{String,Any},Nothing}
    output_schema::Union{Dict{String,Any},Nothing}
end

function MCPToolInfo(d::Dict{String,Any})
    MCPToolInfo(
        d["name"],
        get(d, "description", nothing),
        get(d, "inputSchema", nothing),
        get(d, "outputSchema", nothing)
    )
end

"""
    MCPResourceInfo

A resource definition received from an MCP server via `resources/list`.
"""
struct MCPResourceInfo
    uri::String
    name::String
    description::Union{String,Nothing}
    mime_type::Union{String,Nothing}
end

function MCPResourceInfo(d::Dict{String,Any})
    MCPResourceInfo(d["uri"], d["name"], get(d, "description", nothing), get(d, "mimeType", nothing))
end

"""
    MCPPromptInfo

A prompt definition received from an MCP server via `prompts/list`.
"""
struct MCPPromptInfo
    name::String
    description::Union{String,Nothing}
    arguments::Union{Vector{Dict{String,Any}},Nothing}
end

function MCPPromptInfo(d::Dict{String,Any})
    MCPPromptInfo(d["name"], get(d, "description", nothing), get(d, "arguments", nothing))
end

# ─── Transport Abstraction ───────────────────────────────────────────────────

"""
    MCPTransport

Abstract type for MCP transport implementations. Subtypes must implement:
- `_transport_connect!(t)` — establish connection
- `_transport_send!(t, msg::String)::String` — send JSON-RPC message, return response
- `_transport_notify!(t, msg::String)` — send notification (no response expected)
- `_transport_disconnect!(t)` — close connection
- `_transport_isconnected(t)::Bool` — check if connected
"""
abstract type MCPTransport end

"""
    StdioTransport <: MCPTransport

Stdio transport: launches a subprocess and communicates via stdin/stdout.
Messages are newline-delimited JSON-RPC 2.0.
"""
mutable struct StdioTransport <: MCPTransport
    command::Cmd
    process::Union{Base.Process,Nothing}
    input::Union{IO,Nothing}
    output::Union{IO,Nothing}
    lock::ReentrantLock
    StdioTransport(command::Cmd) = new(command, nothing, nothing, nothing, ReentrantLock())
end

function _transport_connect!(t::StdioTransport)
    proc = open(t.command, read=true, write=true)
    t.process = proc
    t.input = proc.in
    t.output = proc.out
    nothing
end

function _transport_send!(t::StdioTransport, msg::String)::String
    inp = t.input
    isnothing(inp) && error("StdioTransport not connected")
    lock(t.lock) do
        write(inp, msg, "\n")
        flush(inp)
    end
    out = t.output
    isnothing(out) && error("StdioTransport not connected")
    line = readline(out)
    isempty(line) && error("MCP server closed connection")
    line
end

function _transport_notify!(t::StdioTransport, msg::String)
    inp = t.input
    isnothing(inp) && error("StdioTransport not connected")
    lock(t.lock) do
        write(inp, msg, "\n")
        flush(inp)
    end
    nothing
end

function _transport_disconnect!(t::StdioTransport)
    proc = t.process
    if !isnothing(proc) && process_running(proc)
        inp = t.input
        !isnothing(inp) && close(inp)
        timedwait(() -> !process_running(proc), 5.0)
        process_running(proc) && kill(proc)
    end
    t.process = nothing
    t.input = nothing
    t.output = nothing
    nothing
end

function _transport_isconnected(t::StdioTransport)::Bool
    proc = t.process
    !isnothing(proc) && process_running(proc)
end

"""
    HTTPTransport <: MCPTransport

Streamable HTTP transport: communicates via POST requests to an MCP endpoint.
Handles `Mcp-Session-Id` header for session management.
"""
mutable struct HTTPTransport <: MCPTransport
    url::String
    headers::Vector{Pair{String,String}}
    session_id::Union{String,Nothing}
    connected::Bool
    lock::ReentrantLock
    function HTTPTransport(url::String; headers::Vector{Pair{String,String}}=Pair{String,String}[])
        new(url, headers, nothing, false, ReentrantLock())
    end
end

_transport_connect!(t::HTTPTransport) = (t.connected = true; nothing)

function _transport_send!(t::HTTPTransport, msg::String)::String
    hdrs = copy(t.headers)
    push!(hdrs, "Content-Type" => "application/json")
    push!(hdrs, "Accept" => "application/json, text/event-stream")
    push!(hdrs, "Mcp-Protocol-Version" => _MCP_PROTOCOL_VERSION)
    !isnothing(t.session_id) && push!(hdrs, "Mcp-Session-Id" => t.session_id)
    resp = HTTP.post(t.url; body=msg, headers=hdrs, status_exception=false)
    # Capture session ID from response
    sid = HTTP.header(resp, "Mcp-Session-Id", "")
    !isempty(sid) && (t.session_id = sid)
    resp.status == 200 || error("MCP HTTP request failed with status $(resp.status): $(String(resp.body))")
    ct = HTTP.header(resp, "Content-Type", "")
    if startswith(ct, "text/event-stream")
        # Parse SSE: extract last data line as the response
        _parse_sse_response(String(resp.body))
    else
        String(resp.body)
    end
end

function _transport_notify!(t::HTTPTransport, msg::String)
    hdrs = copy(t.headers)
    push!(hdrs, "Content-Type" => "application/json")
    push!(hdrs, "Mcp-Protocol-Version" => _MCP_PROTOCOL_VERSION)
    !isnothing(t.session_id) && push!(hdrs, "Mcp-Session-Id" => t.session_id)
    HTTP.post(t.url; body=msg, headers=hdrs, status_exception=false)
    nothing
end

function _transport_disconnect!(t::HTTPTransport)
    if t.connected && !isnothing(t.session_id)
        hdrs = copy(t.headers)
        push!(hdrs, "Mcp-Session-Id" => t.session_id)
        try
            HTTP.request("DELETE", t.url; headers=hdrs, status_exception=false)
        catch e
            @debug "MCP HTTP disconnect failed" exception=e
        end
    end
    t.connected = false
    t.session_id = nothing
    nothing
end

_transport_isconnected(t::HTTPTransport) = t.connected

"""Parse SSE response body to extract the JSON-RPC response data."""
function _parse_sse_response(body::String)::String
    last_data = ""
    for line in split(body, "\n")
        stripped = strip(line)
        startswith(stripped, "data: ") && (last_data = stripped[7:end])
    end
    isempty(last_data) && error("No data found in SSE response")
    last_data
end

# ─── MCPSession ──────────────────────────────────────────────────────────────

"""
    MCPSession

A live connection to an MCP server. Manages lifecycle, transport, and cached
tool/resource/prompt lists.

Create via [`mcp_connect`](@ref). Disconnect via [`mcp_disconnect!`](@ref).
"""
mutable struct MCPSession
    transport::MCPTransport
    server_capabilities::MCPServerCapabilities
    server_info::Dict{String,Any}
    tools::Vector{MCPToolInfo}
    resources::Vector{MCPResourceInfo}
    prompts::Vector{MCPPromptInfo}
    protocol_version::String
    _id_counter::Int
    status::Symbol  # :disconnected, :initializing, :ready, :closed
end

function _next_id!(session::MCPSession)::Int
    session._id_counter += 1
    session._id_counter
end

"""Send a JSON-RPC request and return the parsed response, throwing MCPError on failure."""
function _mcp_request!(session::MCPSession, method::String, params::Union{Dict{String,Any},Nothing}=nothing)::Dict{String,Any}
    id = _next_id!(session)
    req = _JSONRPCRequest(id, method, params)
    raw = _transport_send!(session.transport, _jsonrpc_serialize(req))
    parsed = JSON.parse(raw; dicttype=Dict{String,Any})
    resp = _JSONRPCResponse(parsed)
    !isnothing(resp.error) && throw(MCPError(resp.error))
    resp.id != id && @warn "Response ID mismatch" expected=id got=resp.id
    something(resp.result, Dict{String,Any}())
end

"""Send a JSON-RPC notification (no response expected)."""
function _mcp_notify!(session::MCPSession, method::String, params::Union{Dict{String,Any},Nothing}=nothing)
    notif = _JSONRPCNotification(method, params)
    _transport_notify!(session.transport, _jsonrpc_serialize(notif))
end

# ─── Lifecycle ───────────────────────────────────────────────────────────────

"""
    mcp_connect(command::Cmd; client_name="UniLM.jl", protocol_version="2025-11-25") -> MCPSession

Connect to an MCP server via stdio transport (subprocess).

# Example
```julia
session = mcp_connect(`npx -y @modelcontextprotocol/server-filesystem /tmp`)
tools = mcp_tools(session)
# ... use tools with tool_loop! ...
mcp_disconnect!(session)
```
"""
function mcp_connect(command::Cmd; kwargs...)::MCPSession
    mcp_connect(StdioTransport(command); kwargs...)
end

"""
    mcp_connect(url::String; headers=[], kwargs...) -> MCPSession

Connect to an MCP server via HTTP transport.

# Example
```julia
session = mcp_connect("https://mcp.example.com/mcp";
    headers=["Authorization" => "Bearer token"])
```
"""
function mcp_connect(url::String; headers::Vector{Pair{String,String}}=Pair{String,String}[], kwargs...)::MCPSession
    mcp_connect(HTTPTransport(url; headers); kwargs...)
end

"""
    mcp_connect(transport::MCPTransport; client_name="UniLM.jl", protocol_version="2025-11-25") -> MCPSession

Connect to an MCP server via the given transport. Performs initialization handshake
and populates tool cache.
"""
function mcp_connect(transport::MCPTransport;
                     client_name::String="UniLM.jl",
                     client_version::String="0.8.0",
                     protocol_version::String=_MCP_PROTOCOL_VERSION)::MCPSession
    _transport_connect!(transport)
    session = MCPSession(
        transport, MCPServerCapabilities(), Dict{String,Any}(),
        MCPToolInfo[], MCPResourceInfo[], MCPPromptInfo[],
        protocol_version, 0, :initializing
    )
    # Initialize handshake
    init_result = _mcp_request!(session, "initialize", Dict{String,Any}(
        "protocolVersion" => protocol_version,
        "capabilities" => Dict{String,Any}(),
        "clientInfo" => Dict{String,Any}("name" => client_name, "version" => client_version)
    ))
    session.server_info = get(init_result, "serverInfo", Dict{String,Any}())
    caps = get(init_result, "capabilities", Dict{String,Any}())
    session.server_capabilities = MCPServerCapabilities(caps)
    session.protocol_version = get(init_result, "protocolVersion", protocol_version)
    # Send initialized notification
    _mcp_notify!(session, "notifications/initialized")
    # Auto-populate tool cache if server supports tools
    !isnothing(session.server_capabilities.tools) && list_tools!(session)
    !isnothing(session.server_capabilities.resources) && list_resources!(session)
    !isnothing(session.server_capabilities.prompts) && list_prompts!(session)
    session.status = :ready
    session
end

"""
    mcp_connect(f::Function, args...; kwargs...)

Do-block form: automatically disconnects after the block executes.

# Example
```julia
mcp_connect(`npx server`) do session
    tools = mcp_tools(session)
    chat = Chat(tools=map(t -> t.tool, tools))
    push!(chat, Message(Val(:user), "List files"))
    tool_loop!(chat; tools)
end
```
"""
function mcp_connect(f::Function, args...; kwargs...)
    session = mcp_connect(args...; kwargs...)
    try
        f(session)
    finally
        mcp_disconnect!(session)
    end
end

"""
    mcp_disconnect!(session::MCPSession)

Gracefully disconnect from the MCP server.
"""
function mcp_disconnect!(session::MCPSession)
    _transport_disconnect!(session.transport)
    session.status = :closed
    nothing
end

# ─── Discovery ───────────────────────────────────────────────────────────────

"""
    list_tools!(session::MCPSession) -> Vector{MCPToolInfo}

Fetch the tool list from the MCP server. Handles pagination via cursor.
Stores result in `session.tools`.
"""
function list_tools!(session::MCPSession)::Vector{MCPToolInfo}
    all_tools = MCPToolInfo[]
    cursor = nothing
    pages = 0
    while true
        (pages += 1) > 1000 && error("MCP pagination exceeded 1000 pages")
        params = isnothing(cursor) ? Dict{String,Any}() : Dict{String,Any}("cursor" => cursor)
        result = _mcp_request!(session, "tools/list", params)
        for t in get(result, "tools", [])
            push!(all_tools, MCPToolInfo(t))
        end
        cursor = get(result, "nextCursor", nothing)
        isnothing(cursor) && break
    end
    session.tools = all_tools
    all_tools
end

"""
    list_resources!(session::MCPSession) -> Vector{MCPResourceInfo}

Fetch the resource list from the MCP server. Handles pagination.
"""
function list_resources!(session::MCPSession)::Vector{MCPResourceInfo}
    all_resources = MCPResourceInfo[]
    cursor = nothing
    pages = 0
    while true
        (pages += 1) > 1000 && error("MCP pagination exceeded 1000 pages")
        params = isnothing(cursor) ? Dict{String,Any}() : Dict{String,Any}("cursor" => cursor)
        result = _mcp_request!(session, "resources/list", params)
        for r in get(result, "resources", [])
            push!(all_resources, MCPResourceInfo(r))
        end
        cursor = get(result, "nextCursor", nothing)
        isnothing(cursor) && break
    end
    session.resources = all_resources
    all_resources
end

"""
    list_prompts!(session::MCPSession) -> Vector{MCPPromptInfo}

Fetch the prompt list from the MCP server. Handles pagination.
"""
function list_prompts!(session::MCPSession)::Vector{MCPPromptInfo}
    all_prompts = MCPPromptInfo[]
    cursor = nothing
    pages = 0
    while true
        (pages += 1) > 1000 && error("MCP pagination exceeded 1000 pages")
        params = isnothing(cursor) ? Dict{String,Any}() : Dict{String,Any}("cursor" => cursor)
        result = _mcp_request!(session, "prompts/list", params)
        for p in get(result, "prompts", [])
            push!(all_prompts, MCPPromptInfo(p))
        end
        cursor = get(result, "nextCursor", nothing)
        isnothing(cursor) && break
    end
    session.prompts = all_prompts
    all_prompts
end

# ─── Tool Operations ────────────────────────────────────────────────────────

"""
    call_tool(session::MCPSession, name::String, arguments::Dict{String,Any}) -> String

Call a tool on the MCP server and return the result as a string.
Concatenates text content parts; non-text content is JSON-encoded.
"""
function call_tool(session::MCPSession, name::String, arguments::Dict{String,Any}=Dict{String,Any}())::String
    result = _mcp_request!(session, "tools/call", Dict{String,Any}(
        "name" => name, "arguments" => arguments
    ))
    content = get(result, "content", [])
    is_error = get(result, "isError", false)
    parts = String[]
    for part in content
        if part isa Dict
            ptype = get(part, "type", "")
            if ptype == "text"
                push!(parts, part["text"])
            else
                push!(parts, JSON.json(part))
            end
        else
            push!(parts, string(part))
        end
    end
    text = join(parts, "\n")
    is_error && error("MCP tool '$name' returned error: $text")
    text
end

"""
    read_resource(session::MCPSession, uri::String) -> String

Read a resource from the MCP server.
"""
function read_resource(session::MCPSession, uri::String)::String
    result = _mcp_request!(session, "resources/read", Dict{String,Any}("uri" => uri))
    contents = get(result, "contents", [])
    parts = String[]
    for c in contents
        if haskey(c, "text")
            push!(parts, c["text"])
        elseif haskey(c, "blob")
            push!(parts, c["blob"])
        end
    end
    join(parts, "\n")
end

"""
    get_prompt(session::MCPSession, name::String, arguments::Dict{String,Any}=Dict()) -> Vector{Dict{String,Any}}

Get a rendered prompt from the MCP server. Returns the messages array.
"""
function get_prompt(session::MCPSession, name::String, arguments::Dict{String,Any}=Dict{String,Any}())::Vector{Dict{String,Any}}
    result = _mcp_request!(session, "prompts/get", Dict{String,Any}(
        "name" => name, "arguments" => arguments
    ))
    get(result, "messages", Dict{String,Any}[])
end

"""
    ping(session::MCPSession)

Send a ping to the MCP server. Throws on error.
"""
function ping(session::MCPSession)
    _mcp_request!(session, "ping")
    nothing
end

# ─── Tool Bridge ─────────────────────────────────────────────────────────────

"""
    mcp_tools(session::MCPSession) -> Vector{CallableTool{GPTTool}}

Convert all tools from an MCP session into `CallableTool{GPTTool}` instances
that work directly with [`tool_loop!`](@ref) (Chat Completions API).

Each tool's callable invokes `call_tool(session, name, args)` under the hood.

# Example
```julia
session = mcp_connect(`npx server`)
tools = mcp_tools(session)
chat = Chat(model="gpt-5.2", tools=map(t -> t.tool, tools))
push!(chat, Message(Val(:user), "Do something"))
result = tool_loop!(chat; tools)
```
"""
function mcp_tools(session::MCPSession)::Vector{CallableTool{GPTTool}}
    map(session.tools) do info
        schema = GPTTool(func=GPTFunctionSignature(
            name=info.name,
            description=info.description,
            parameters=info.input_schema
        ))
        # Capture session and info.name in closure
        sref = session
        tname = info.name
        callable = (_::String, args::Dict{String,Any}) -> call_tool(sref, tname, args)
        CallableTool(schema, callable)
    end
end

"""
    mcp_tools_respond(session::MCPSession) -> Vector{CallableTool{FunctionTool}}

Convert all tools from an MCP session into `CallableTool{FunctionTool}` instances
that work directly with [`tool_loop`](@ref) (Responses API).

# Example
```julia
session = mcp_connect("https://mcp.example.com/mcp")
tools = mcp_tools_respond(session)
result = tool_loop("Do something"; tools=tools)
```
"""
function mcp_tools_respond(session::MCPSession)::Vector{CallableTool{FunctionTool}}
    map(session.tools) do info
        schema = FunctionTool(
            name=info.name,
            description=info.description,
            parameters=info.input_schema
        )
        sref = session
        tname = info.name
        callable = (_::String, args::Dict{String,Any}) -> call_tool(sref, tname, args)
        CallableTool(schema, callable)
    end
end

# Extend to_tool protocol
to_tool(info::MCPToolInfo) = GPTTool(func=GPTFunctionSignature(
    name=info.name, description=info.description, parameters=info.input_schema
))
