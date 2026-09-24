# [Model Context Protocol (MCP)](@id mcp_guide)

UniLM.jl provides native MCP support — both as a **client** (connect to MCP servers) and a
**server** (build your own). MCP tools integrate seamlessly with [`tool_loop!`](@ref) and
[`tool_loop`](@ref) via the [`CallableTool`](@ref) bridge.

Protocol: JSON-RPC 2.0 over stdio or Streamable HTTP. The client and the server
negotiate MCP revisions 2025-11-25 (preferred), 2025-06-18 and 2025-03-26. The
stateless 2026-07-28 revision is not supported yet, so a peer that speaks only
2026-07-28 cannot interoperate: `mcp_connect` fails naming the revision such a server
returned, `serve` answers such a client's `initialize` with 2025-11-25, and over HTTP
a request whose `MCP-Protocol-Version` header names an unsupported revision gets
`400`. Zero external dependencies.

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
# Stdio — launches subprocess (stderr=devnull or a log-file path silences its stderr)
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

!!! note "A session runs one call at a time, first come first served"
    Every call on an [`MCPSession`](@ref) — its liveness check, id allocation and
    request/response exchange — holds the session, and waiting callers are served
    in arrival order. A call's per-call bound (`timeout`, default
    `mcp_request_timeout`) also covers its wait: a call that cannot acquire the
    session in time throws [`MCPTimeoutError`](@ref) with phase `:queue` without
    touching it. Once a call holds the session, its exchange gets the full bound,
    measured from that moment, so a waiter's clock never tears down the exchange in
    progress. [`mcp_disconnect!`](@ref) waits its turn too, so a disconnect racing a
    call in flight waits for that exchange to finish instead of tearing the
    transport down under its reader. For real parallelism, open one session per
    concurrent worker — see [Concurrency, Tasks and Cancellation](@ref concurrency_guide).

A call on a session that is closed — by [`mcp_disconnect!`](@ref), by a stdio
timeout or crash, or because its transport is not connected — throws the typed
[`MCPSessionClosedError`](@ref), whose `cause` is `:disconnected`, `:timeout` or
`:crash` (see [Timeouts](@ref mcp_timeouts) for `auto_respawn`).

### Discovering Tools, Resources, and Prompts

After connecting, the session auto-populates tool/resource/prompt caches. You can also
refresh them manually:

```julia
tools    = list_tools!(session)     # -> Vector{MCPToolInfo}
resources = list_resources!(session) # -> Vector{MCPResourceInfo}
prompts  = list_prompts!(session)   # -> Vector{MCPPromptInfo}
```

!!! note "Server notifications"
    MCP servers may send notifications interleaved with responses; the client
    skips them transparently, answers server `ping` requests, and answers any
    other server request with `-32601`. It reads the server's frames only during an
    exchange — a stdio notification sent between calls is read with the next call,
    and over HTTP there is no standing listener, so only notifications carried in a
    response body (before or after the response itself) are seen. When a
    `notifications/tools/list_changed` is read, the session's cached tool list is
    marked stale — check `session.tools_stale` and call `list_tools!` to refresh.
    A stdio line that is not JSON (a stray log line on the server's stdout) is
    skipped with a warning.

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

### [Timeouts](@id mcp_timeouts)

Every `mcp_connect` handshake and every `call_tool` / `list_tools!` exchange runs
under a bound — set per session at connect time, or overridden for one call:

```julia
session = mcp_connect(`npx server`;
    config=RequestConfig(current_config(); mcp_connect_timeout=10.0, mcp_request_timeout=30.0),
    auto_respawn=true)

call_tool(session, "read_file", Dict{String,Any}("path" => "/tmp/data.txt"); timeout=5.0)
```

A stdio request timeout closes the session: a read blocked on an unresponsive
server can be released only by killing the server. The watchdog closes the pipe, so
the typed [`MCPTimeoutError`](@ref) surfaces at about the bound, and then tears the
server's process group down (stdin EOF, SIGTERM, then SIGKILL); a reply that raced
the teardown is discarded. With `auto_respawn=true` the next call transparently
respawns the server (same command, fresh handshake; in-memory server state is lost
and tools are refetched); without it, the next call raises
[`MCPSessionClosedError`](@ref) naming the opt-in. An HTTP request timeout does not
close the session — each exchange is an independent POST — and the timed-out request
is cancelled on the server with a best-effort `notifications/cancelled`.

`auto_respawn` covers **hangs and crashes** alike: a server that dies abruptly —
killed, crashed, or its stdio pipe broken — closes the session too, surfacing the
typed [`MCPCrashError`](@ref) (carrying the exit code or signal when known) on the
call in flight, and the next call respawns it when `auto_respawn=true`. Stdio
servers still running when Julia exits are torn down by an exit hook, and when a
server's process-group leader exits (an `npx` wrapper, say) the rest of its group is
killed at once.

A [`CancelToken`](@ref) reaches MCP calls over HTTP: a cancel ends the call at once
with `UniLMCancelled`, sends a best-effort `notifications/cancelled`, and leaves the
session open. A stdio exchange does not observe the token; it is bounded by
`mcp_request_timeout` alone. Teardown — `mcp_disconnect!`'s `DELETE`, a
cancellation notice — is sent even inside a cancelled `with_cancel` scope.

### Bridging to tool_loop! (Chat Completions)

[`mcp_tools`](@ref) converts MCP tools into `Vector{CallableTool{Tool}}` for use with
[`tool_loop!`](@ref). Tool names are advertised provider-safe: OpenAI and Anthropic
accept only `^[a-zA-Z0-9_-]{1,128}$`, so any other character of an MCP tool name
(the dots in `admin.tools.list`, say) becomes `_`, cut to 128 characters — the
bridged callable still calls the tool by its MCP name. Two tools that map to the same
name raise an `ArgumentError`.

```julia
session = mcp_connect(`npx server`)
tools = mcp_tools(session)

chat = Chat(model="gpt-5.2", tools=tools)
push!(chat, Message(Val(:system), "You are a helpful assistant."))
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

With the inferred form, the handler's positional parameters *are* the schema: each
`tools/call` binds the `arguments` object to them **by name**. A parameter whose type
admits `nothing` (`Union{T,Nothing}`) is optional; every other one is required, and
a value must have its parameter's JSON type (a string for `String`, an integral
number for an integer, a number for a float, a boolean for `Bool`; a `Symbol` binds
from a string, and a typed `Vector` or `Dict` element-wise). A call that violates
this never reaches the handler: the client gets a tool result with `isError: true`
naming the argument, so the model can correct its call. A handler that takes one
`Dict` (the explicit-schema convention) or varargs has no names to bind and is
rejected with an `ArgumentError`.

```@example mcp
# Auto-inferred schema from function signature: `name` is a required string
register_tool!(server, "greet", "Greet someone", (name::String) -> "Hello, $name!")

println("Tools now: ", collect(keys(server.tools)))
println("greet schema: ", JSON.json(server.tools["greet"].input_schema))
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
registration API with automatic JSON Schema generation from Julia type annotations.
The examples below are executed at doc-build time and call the registered handlers
directly to prove the wiring works.

#### `@mcp_tool` — typed args become JSON Schema

A named typed function becomes a tool whose `inputSchema` is inferred from the argument
types (typed args are `required`, untyped ones optional). The generated handler binds
the incoming `arguments` object to your function's parameters by name, with the same
type checks as the inferred-schema `register_tool!`; a missing or wrong-typed
argument is answered with an `isError: true` tool result naming it. The tool is
registered without a description.

!!! note "Contract"
    `@mcp_tool` requires a **named** function (`function name(args…)`). An anonymous
    `function(x) … end` now raises a clear error instead of registering a tool named
    after its first argument.

```@example mcp
srv = MCPServer("calc", "1.0.0")

@mcp_tool srv function add(a::Float64, b::Float64)::String
    string(a + b)
end

# Generated schema, inferred from the signature:
println("schema: ", JSON.json(srv.tools["add"].input_schema))
# Call the registered handler with raw JSON-shaped args:
println("add(2, 3) = ", srv.tools["add"].handler(Dict{String,Any}("a" => 2.0, "b" => 3.0)))
```

#### `@mcp_resource` — static and URI-templated

A `function()` registers a static resource; a URI with `{param}` placeholders registers
a template whose handler arguments are bound from the matched path params.

!!! note "Contract"
    For template resources, each handler argument name must match a URI `{param}` name —
    it is bound from the matched params (e.g. `{id}` ⇒ `function(id::String)`).

```@example mcp
@mcp_resource srv "config://app" function()
    "{\"debug\": true}"
end

@mcp_resource srv "note://{id}" function(id::String)
    "Note #$id"
end

# Read the templated resource via JSON-RPC; the {id} param is bound into `id`:
req = Dict{String,Any}("jsonrpc" => "2.0", "id" => 1, "method" => "resources/read",
    "params" => Dict{String,Any}("uri" => "note://42"))
resp = UniLM._dispatch_mcp(srv, req)
println("note://42 -> ", resp["result"]["contents"][1]["text"])
```

#### `@mcp_prompt` — args bound from the request

The anonymous `function(arg::String) … end` form registers a prompt; declared args are
bound from the `prompts/get` request arguments.

```@example mcp
@mcp_prompt srv "review" function(code::String)
    [Dict("role" => "user",
        "content" => Dict("type" => "text", "text" => "Review: $code"))]
end

# Dispatch prompts/get; `code` is bound from the request arguments:
greq = Dict{String,Any}("jsonrpc" => "2.0", "id" => 2, "method" => "prompts/get",
    "params" => Dict{String,Any}("name" => "review",
        "arguments" => Dict{String,Any}("code" => "x + 1")))
gresp = UniLM._dispatch_mcp(srv, greq)
println("review -> ", gresp["result"]["messages"][1]["content"]["text"])
```

### Serving

Start the server with [`serve`](@ref):

```julia
serve(server)                              # stdio (default) — for Claude Desktop/CLI
serve(server; transport=:http, port=3000)  # HTTP on port 3000 — blocks until closed
```

The HTTP transport blocks until the server is closed (like `HTTP.serve`). To
keep working in the same session, pass `block=false` and close the returned
server yourself:

```julia
handle = serve(server; transport=:http, port=3000, block=false)
# ... interact with the running server ...
close(handle)
```

The HTTP transport validates the `Origin` header (a DNS-rebinding defense the
MCP spec requires): requests without an `Origin` header (curl, SDK clients)
and requests from localhost origins always pass; any other browser origin gets
403 unless you allowlist it:

```julia
serve(server; transport=:http, port=3000,
      allowed_origins=["https://app.example.com"])
```

**Handler concurrency.** Over HTTP, `tools/call`, `resources/read` and `prompts/get`
handlers run concurrently — one task per request, on the default thread pool — so
handlers must be thread-safe (protocol requests such as `initialize`, the lists and
`ping` are answered inline and stay prompt while handlers are busy). Over stdio,
handlers run one at a time, and while serving on the process `stdout` it is pointed
at `stderr`, so a handler that prints cannot corrupt the protocol stream.
Registration is synchronized: tools, resources and prompts may be registered while
the server is serving.

**What reaches the client.** A tool handler's exception is a tool result, not a
protocol error: the client receives `isError: true` with the text
`Error: <showerror text>`, which lets a model correct itself — raise with a message you
are willing to show it. An exception from a resource or prompt handler, or from
dispatch itself, is answered with a generic JSON-RPC `-32603` `"Internal error"` and
logged on the server, so no exception text (file paths, argument values) reaches the
peer. An `InterruptException` is never converted: it propagates. Frames (stdio) and
request bodies (HTTP) are capped at 16 MiB: stdio answers an oversized frame with
`-32600`, HTTP with `413 Payload Too Large`. `initialize` answers the revision the
client requested when it is supported, otherwise 2025-11-25; a JSON-RPC response sent
to the server is accepted without an answer (HTTP `202`).

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
- [Timeouts & Retries](@ref timeouts_guide) — MCP bounds, the `:queue` phase and `auto_respawn`
- [Concurrency, Tasks and Cancellation](@ref concurrency_guide) — sessions under fan-out, server handler concurrency
- [`CallableTool`](@ref), [`tool_loop!`](@ref), [`tool_loop`](@ref) — tool loop integration
