# [Model Context Protocol (MCP)](@id mcp_guide)

UniLM.jl provides native MCP support — both as a **client** (connect to MCP servers) and a
**server** (build your own). MCP tools integrate seamlessly with [`tool_loop!`](@ref) and
[`tool_loop`](@ref) via the [`CallableTool`](@ref) bridge.

Protocol: JSON-RPC 2.0 over stdio or Streamable HTTP (MCP spec 2025-11-25). Zero external dependencies.

```@setup mcp
using UniLM
using JSON
```

---

## MCP Client

### Transports

Two transport types are available:

- **[`StdioTransport`](@ref)** — launches a subprocess, communicates via stdin/stdout (newline-delimited JSON-RPC)
- **[`HTTPTransport`](@ref)** — communicates via POST requests with `Mcp-Session-Id` session management

```@example mcp
# Transport types are constructed but not connected until mcp_connect
t1 = StdioTransport(`echo hello`)
println("Stdio transport for: ", t1.command)

t2 = HTTPTransport("https://mcp.example.com/mcp";
    headers=["Authorization" => "Bearer token"])
println("HTTP transport for: ", t2.url)
```

### Connecting to an MCP Server

Use [`mcp_connect`](@ref) with a `Cmd` (stdio), URL string (HTTP), or transport object:

```julia
# Stdio — launches subprocess
session = mcp_connect(`npx -y @modelcontextprotocol/server-filesystem /tmp`)

# HTTP
session = mcp_connect("https://mcp.example.com/mcp";
    headers=["Authorization" => "Bearer token"])

# Custom transport
session = mcp_connect(StdioTransport(`my-server`))
```

The do-block form automatically disconnects when done:

```julia
mcp_connect(`npx server`) do session
    tools = mcp_tools(session)
    # ... use tools ...
end  # session is disconnected here
```

### Discovering Tools, Resources, and Prompts

After connecting, the session auto-populates tool/resource/prompt caches. You can also
refresh them manually:

```julia
tools    = list_tools!(session)     # -> Vector{MCPToolInfo}
resources = list_resources!(session) # -> Vector{MCPResourceInfo}
prompts  = list_prompts!(session)   # -> Vector{MCPPromptInfo}
```

```@example mcp
# MCPToolInfo fields
info = MCPToolInfo(Dict{String,Any}(
    "name" => "read_file",
    "description" => "Read a file from disk",
    "inputSchema" => Dict{String,Any}(
        "type" => "object",
        "properties" => Dict{String,Any}(
            "path" => Dict{String,Any}("type" => "string", "description" => "File path")
        ),
        "required" => ["path"]
    )
))
println("Tool: ", info.name)
println("Description: ", info.description)
println("Schema: ", JSON.json(info.input_schema, 2))
```

### Calling Tools Directly

```julia
result = call_tool(session, "read_file", Dict{String,Any}("path" => "/tmp/data.txt"))
content = read_resource(session, "config://app")
messages = get_prompt(session, "review", Dict{String,Any}("code" => "x + 1"))
ping(session)
```

### Bridging to tool_loop! (Chat Completions)

[`mcp_tools`](@ref) converts MCP tools into `Vector{CallableTool{GPTTool}}` for use with
[`tool_loop!`](@ref):

```julia
session = mcp_connect(`npx server`)
tools = mcp_tools(session)

chat = Chat(model="gpt-5.2", tools=map(t -> t.tool, tools))
push!(chat, Message(Val(:user), "List files in /tmp"))
result = tool_loop!(chat; tools)

mcp_disconnect!(session)
```

### Bridging to tool_loop (Responses API)

[`mcp_tools_respond`](@ref) converts MCP tools into `Vector{CallableTool{FunctionTool}}`
for use with [`tool_loop`](@ref):

```julia
session = mcp_connect("https://mcp.example.com/mcp")
tools = mcp_tools_respond(session)
result = tool_loop("List files in /tmp"; tools=tools)
mcp_disconnect!(session)
```

---

## MCP Server

### Creating a Server

```@example mcp
server = MCPServer("calc", "1.0.0"; description="A calculator server")
println("Server: ", server.name, " v", server.version)
```

### Registering Tools

Register tools with explicit JSON Schema or auto-inferred schema:

```@example mcp
# Explicit schema
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

println("Registered tools: ", collect(keys(server.tools)))
```

```@example mcp
# Auto-inferred schema from function signature
register_tool!(server, "greet", "Greet someone",
    (args::Dict{String,Any}) -> "Hello, $(args["name"])!")

println("Tools now: ", collect(keys(server.tools)))
```

You can also register existing [`CallableTool`](@ref) instances:

```julia
# From Chat Completions tool
register_tool!(server, my_callable_gpt_tool)

# From Responses API tool
register_tool!(server, my_callable_function_tool)
```

### Registering Resources

```@example mcp
# Static resource
register_resource!(server, "config://app", "App Config",
    () -> "{\"debug\": true}";
    mime_type="application/json",
    description="Application configuration")

println("Resources: ", collect(keys(server.resources)))
```

```@example mcp
# URI-templated resource
register_resource_template!(server, "file://{path}", "File Reader",
    (params::Dict{String,String}) -> "Contents of $(params["path"])";
    description="Read files by path")

println("Templates: ", length(server.resource_templates))
```

### Registering Prompts

```@example mcp
register_prompt!(server, "review", (args::Dict{String,Any}) ->
    [Dict{String,Any}("role" => "user",
        "content" => Dict{String,Any}("type" => "text",
            "text" => "Review this code:\n$(args["code"])"))];
    description="Code review prompt",
    arguments=[Dict{String,Any}("name" => "code", "required" => true)])

println("Prompts: ", collect(keys(server.prompts)))
```

### Macros

The `@mcp_tool`, `@mcp_resource`, and `@mcp_prompt` macros provide a more ergonomic
registration API with automatic JSON Schema generation from Julia type annotations:

```julia
server = MCPServer("calc", "1.0.0")

@mcp_tool server function add(a::Float64, b::Float64)::String
    string(a + b)
end

@mcp_resource server "config://app" function()
    read("config.toml", String)
end

@mcp_resource server "file://{path}" function(path::String)
    read(path, String)
end

@mcp_prompt server "review" function(code::String)
    [Dict("role" => "user",
        "content" => Dict("type" => "text", "text" => "Review: $code"))]
end
```

### Serving

Start the server with [`serve`](@ref):

```julia
serve(server)                             # stdio (default) — for Claude Desktop/CLI
serve(server; transport=:http, port=3000)  # HTTP on port 3000
```

---

## MCP Tool in Responses API

Separately from the client/server above, OpenAI's Responses API has a built-in
[`MCPTool`](@ref) type for server-side MCP integration. This tells the model to connect
to an external MCP server during response generation:

```@example mcp
tool = mcp_tool("my-server", "https://mcp.example.com/sse";
    require_approval="never",
    allowed_tools=["read_file", "list_dir"])
println("Type: ", typeof(tool))
println("Label: ", tool.server_label)
println("URL: ", tool.server_url)
println("JSON: ", JSON.json(JSON.lower(tool)))
```

This is distinct from the UniLM.jl MCP client — `MCPTool` delegates tool execution to
OpenAI's servers, while `mcp_connect` runs tools locally.

---

## See Also

- [Tool Calling Guide](@ref tools_guide) — function tools and automated tool loop
- [MCP API Reference](@ref mcp_api) — full type and function reference
- [`CallableTool`](@ref), [`tool_loop!`](@ref), [`tool_loop`](@ref) — tool loop integration
