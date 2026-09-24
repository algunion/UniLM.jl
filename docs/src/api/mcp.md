# [MCP Client & Server](@id mcp_api)

Types and functions for the **Model Context Protocol** — connecting to MCP servers
and building MCP servers in Julia. Both sides negotiate MCP revisions 2025-11-25
(preferred), 2025-06-18 and 2025-03-26; the 2026-07-28 revision is not supported yet.
A session runs one call at a time in arrival order, and a call's `timeout` bounds its
wait for the session too ([`MCPTimeoutError`](@ref) phase `:queue`); over HTTP,
[`serve`](@ref) runs handlers concurrently. See the [MCP guide](@ref mcp_guide) and
[Concurrency, Tasks and Cancellation](@ref concurrency_guide).

## Client Types

```@docs
MCPSession
MCPToolInfo
MCPToolResult
MCPResourceInfo
MCPPromptInfo
MCPServerCapabilities
MCPTransport
StdioTransport
HTTPTransport
MCPError
MCPCrashError
MCPSessionClosedError
```

## Client Functions

### Lifecycle

```@docs
mcp_connect
mcp_disconnect!
```

### Discovery

```@docs
list_tools!
list_resources!
list_prompts!
```

### Operations

```@docs
call_tool
read_resource
get_prompt
ping
```

### Tool Bridge

```@docs
mcp_tools
mcp_tools_respond
```

## Server Types

```@docs
MCPServer
MCPServerPrimitive
MCPServerTool
MCPServerResource
MCPServerResourceTemplate
MCPServerPrompt
```

## Server Functions

### Registration

```@docs
register_tool!
register_resource!
register_resource_template!
register_prompt!
```

### Serving

```@docs
serve
```

### Macros

```@docs
@mcp_tool
@mcp_resource
@mcp_prompt
```

## Example

```@example mcp_api
using UniLM
using JSON

# Construct client info types
info = MCPToolInfo(Dict{String,Any}(
    "name" => "read_file",
    "description" => "Read a file from disk",
    "inputSchema" => Dict{String,Any}(
        "type" => "object",
        "properties" => Dict{String,Any}(
            "path" => Dict{String,Any}("type" => "string")
        )
    )
))
println("Tool: ", info.name, " — ", info.description)
```

```@example mcp_api
# Build and populate a server
server = MCPServer("demo", "1.0.0")
register_tool!(server, "add", "Add two numbers",
    Dict{String,Any}(
        "type" => "object",
        "properties" => Dict{String,Any}(
            "a" => Dict{String,Any}("type" => "number"),
            "b" => Dict{String,Any}("type" => "number")
        ),
        "required" => ["a", "b"]
    ),
    args -> string(args["a"] + args["b"]))
println("Server: ", server.name, " v", server.version)
println("Tools: ", collect(keys(server.tools)))
```
