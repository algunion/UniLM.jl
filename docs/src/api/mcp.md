# [MCP Client & Server](@id mcp_api)

Types and functions for the **Model Context Protocol** — connecting to MCP servers
and building MCP servers in Julia.

## Client Types

```@docs
MCPSession
MCPToolInfo
MCPResourceInfo
MCPPromptInfo
MCPServerCapabilities
MCPTransport
StdioTransport
HTTPTransport
MCPError
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
