# Tests for mcp_client.jl — MCP Client types, JSON-RPC framing, tool bridge

@testset "JSON-RPC framing" begin
    @testset "Request serialization" begin
        req = UniLM._JSONRPCRequest(1, "initialize", Dict{String,Any}("protocolVersion" => "2025-11-25"))
        json = UniLM._jsonrpc_serialize(req)
        parsed = JSON.parse(json)
        @test parsed["jsonrpc"] == "2.0"
        @test parsed["id"] == 1
        @test parsed["method"] == "initialize"
        @test parsed["params"]["protocolVersion"] == "2025-11-25"
    end

    @testset "Request without params" begin
        req = UniLM._JSONRPCRequest(2, "ping", nothing)
        json = UniLM._jsonrpc_serialize(req)
        parsed = JSON.parse(json)
        @test !haskey(parsed, "params")
        @test parsed["method"] == "ping"
    end

    @testset "Notification serialization" begin
        notif = UniLM._JSONRPCNotification("notifications/initialized", nothing)
        json = UniLM._jsonrpc_serialize(notif)
        parsed = JSON.parse(json)
        @test parsed["jsonrpc"] == "2.0"
        @test parsed["method"] == "notifications/initialized"
        @test !haskey(parsed, "id")
    end

    @testset "Response parsing" begin
        d = Dict{String,Any}("id" => 1, "result" => Dict{String,Any}("protocolVersion" => "2025-11-25"))
        resp = UniLM._JSONRPCResponse(d)
        @test resp.id == 1
        @test !isnothing(resp.result)
        @test isnothing(resp.error)
        @test resp.result["protocolVersion"] == "2025-11-25"
    end

    @testset "Error response parsing" begin
        d = Dict{String,Any}("id" => 1, "error" => Dict{String,Any}("code" => -32601, "message" => "Method not found"))
        resp = UniLM._JSONRPCResponse(d)
        @test !isnothing(resp.error)
        @test isnothing(resp.result)
        @test resp.error["code"] == -32601
    end
end

@testset "MCPError" begin
    e = MCPError(42, "test error", nothing)
    @test e.code == 42
    @test e.message == "test error"
    @test isnothing(e.data)
    buf = IOBuffer()
    showerror(buf, e)
    @test contains(String(take!(buf)), "MCPError(42)")

    # From dict
    d = Dict{String,Any}("code" => -32700, "message" => "Parse error", "data" => Dict{String,Any}("details" => "bad json"))
    e2 = MCPError(d)
    @test e2.code == -32700
    @test !isnothing(e2.data)
end

@testset "MCP types" begin
    @testset "MCPServerCapabilities" begin
        caps = MCPServerCapabilities(Dict{String,Any}(
            "tools" => Dict{String,Any}(),
            "resources" => Dict{String,Any}("subscribe" => true)
        ))
        @test !isnothing(caps.tools)
        @test !isnothing(caps.resources)
        @test isnothing(caps.prompts)
        @test isnothing(caps.logging)
    end

    @testset "MCPToolInfo" begin
        d = Dict{String,Any}(
            "name" => "get_weather",
            "description" => "Get weather",
            "inputSchema" => Dict{String,Any}("type" => "object", "properties" => Dict{String,Any}(
                "location" => Dict{String,Any}("type" => "string")
            ))
        )
        info = MCPToolInfo(d)
        @test info.name == "get_weather"
        @test info.description == "Get weather"
        @test !isnothing(info.input_schema)
        @test isnothing(info.output_schema)
    end

    @testset "MCPResourceInfo" begin
        d = Dict{String,Any}("uri" => "file:///tmp", "name" => "tmp", "mimeType" => "text/plain")
        info = MCPResourceInfo(d)
        @test info.uri == "file:///tmp"
        @test info.name == "tmp"
        @test info.mime_type == "text/plain"
    end

    @testset "MCPPromptInfo" begin
        d = Dict{String,Any}("name" => "review", "description" => "Code review")
        info = MCPPromptInfo(d)
        @test info.name == "review"
        @test info.description == "Code review"
        @test isnothing(info.arguments)
    end
end

@testset "Tool bridge" begin
    @testset "MCPToolInfo → GPTTool via to_tool" begin
        info = MCPToolInfo("calc", "Calculator", Dict{String,Any}("type" => "object"), nothing)
        tool = to_tool(info)
        @test tool isa GPTTool
        @test tool.func.name == "calc"
        @test tool.func.description == "Calculator"
    end

    @testset "mcp_tools builds CallableTool{GPTTool}" begin
        # Create a minimal session with mock tools
        session = MCPSession(
            StdioTransport(`echo`),  # won't be used
            MCPServerCapabilities(),
            Dict{String,Any}(),
            [MCPToolInfo("add", "Add numbers",
                Dict{String,Any}("type" => "object", "properties" => Dict{String,Any}(
                    "a" => Dict{String,Any}("type" => "number"),
                    "b" => Dict{String,Any}("type" => "number")
                )), nothing)],
            MCPResourceInfo[],
            MCPPromptInfo[],
            "2025-11-25", 0, :ready
        )
        tools = mcp_tools(session)
        @test length(tools) == 1
        @test tools[1] isa CallableTool{GPTTool}
        @test tools[1].tool.func.name == "add"
    end

    @testset "mcp_tools_respond builds CallableTool{FunctionTool}" begin
        session = MCPSession(
            StdioTransport(`echo`),
            MCPServerCapabilities(),
            Dict{String,Any}(),
            [MCPToolInfo("sub", "Subtract", Dict{String,Any}("type" => "object"), nothing)],
            MCPResourceInfo[],
            MCPPromptInfo[],
            "2025-11-25", 0, :ready
        )
        tools = mcp_tools_respond(session)
        @test length(tools) == 1
        @test tools[1] isa CallableTool{FunctionTool}
        @test tools[1].tool.name == "sub"
    end
end

@testset "SSE response parsing" begin
    body = "event: message\ndata: {\"jsonrpc\":\"2.0\",\"id\":1,\"result\":{}}\n\n"
    result = UniLM._parse_sse_response(body)
    parsed = JSON.parse(result)
    @test parsed["id"] == 1
    @test haskey(parsed, "result")
end

@testset "Transport construction" begin
    @testset "StdioTransport" begin
        t = StdioTransport(`echo hello`)
        @test t.command == `echo hello`
        @test isnothing(t.process)
        @test !UniLM._transport_isconnected(t)
    end

    @testset "HTTPTransport" begin
        t = HTTPTransport("https://example.com/mcp";
            headers=["Authorization" => "Bearer token"])
        @test t.url == "https://example.com/mcp"
        @test length(t.headers) == 1
        @test isnothing(t.session_id)
        @test !UniLM._transport_isconnected(t)
    end
end

# ─── In-process HTTP MCP server: drive the CLIENT against a REAL UniLM server ──
# We mount a real `MCPServer` (with a tool, resource, resource template, prompt)
# behind a tiny HTTP handler that delegates to `UniLM._dispatch_mcp`, then drive
# the HTTPTransport client end-to-end. This exercises the previously-untested
# client operation layer (handshake, list/call/read/get, ping, errors, SSE).

using Sockets

"Pick a free ephemeral localhost port (bind→read→close so the port is free again)."
_free_port() = let s = Sockets.listen(Sockets.localhost, 0)
    p = Int(Sockets.getsockname(s)[2])
    close(s)
    p
end

"""
Build a real `MCPServer` populated with one tool, one static resource, one
resource template, and one prompt — all with deterministic, assertable outputs.
`captured` (a Ref) records the LAST tool-call arguments seen server-side so a
test can prove the client actually transmitted them on the wire.
"""
function _build_mcp_test_server(captured::Ref{Any})
    server = UniLM.MCPServer("unilm-test-mcp", "9.9.9"; description="probe server")
    # Tool: result is a deterministic function of args → exact-value assertable.
    UniLM.register_tool!(server, "concat", "Concatenate a and b with a pipe",
        Dict{String,Any}("type" => "object",
            "properties" => Dict{String,Any}(
                "a" => Dict{String,Any}("type" => "string"),
                "b" => Dict{String,Any}("type" => "string")),
            "required" => ["a", "b"]),
        function (args::Dict{String,Any})
            captured[] = args
            string(args["a"]) * "|" * string(args["b"])
        end)
    # Tool whose handler THROWS → server returns isError content (not a JSON-RPC error).
    UniLM.register_tool!(server, "boom", "Always throws",
        Dict{String,Any}("type" => "object", "properties" => Dict{String,Any}()),
        function (_::Dict{String,Any})
            error("kaboom")
        end)
    # Static resource: fixed text content.
    UniLM.register_resource!(server, "probe://greeting", "greeting",
        () -> "hello-from-resource"; mime_type="text/plain",
        description="a canned greeting")
    # Resource template: content derived from the captured path segment.
    UniLM.register_resource_template!(server, "probe://echo/{word}", "echo",
        (p::Dict{String,String}) -> "echo:" * p["word"]; mime_type="text/plain")
    # Prompt: returns a known messages array derived from args.
    UniLM.register_prompt!(server, "salute",
        function (args::Dict{String,Any})
            who = get(args, "who", "world")
            [Dict{String,Any}("role" => "user",
                "content" => Dict{String,Any}("type" => "text", "text" => "Hello, $(who)!"))]
        end;
        description="Greet someone",
        arguments=Dict{String,Any}[Dict{String,Any}("name" => "who", "required" => true)])
    server
end

"Serve `server` over HTTP via `_dispatch_mcp`; JSON responses. Returns (httpserver, baseurl)."
function _serve_mcp_json(server::UniLM.MCPServer)
    port = _free_port()
    httpserver = HTTP.serve!("127.0.0.1", port; verbose=false) do req
        req.method == "DELETE" && return HTTP.Response(200, "")
        req.method == "POST" || return HTTP.Response(405, "Method Not Allowed")
        parsed = JSON.parse(String(req.body); dicttype=Dict{String,Any})
        resp = UniLM._dispatch_mcp(server, parsed)
        isnothing(resp) && return HTTP.Response(202, "")  # notification
        HTTP.Response(200, ["Content-Type" => "application/json"], JSON.json(resp))
    end
    (httpserver, "http://127.0.0.1:$port")
end

@testset "MCP client ↔ in-process HTTP server" begin
    captured = Ref{Any}(nothing)
    server = _build_mcp_test_server(captured)
    httpserver, url = _serve_mcp_json(server)
    try
        @testset "mcp_connect handshake auto-populates caches" begin
            session = mcp_connect(url)
            try
                @test session.status == :ready
                # serverInfo carries the EXACT registered name/version.
                @test session.server_info["name"] == "unilm-test-mcp"
                @test session.server_info["version"] == "9.9.9"
                # Server advertised tools+resources+prompts (all registered, non-empty).
                @test !isnothing(session.server_capabilities.tools)
                @test !isnothing(session.server_capabilities.resources)
                @test !isnothing(session.server_capabilities.prompts)
                @test isnothing(session.server_capabilities.logging)
                @test session.protocol_version == UniLM._MCP_PROTOCOL_VERSION
                # Auto-populated caches contain EXACTLY the registered names.
                @test Set(t.name for t in session.tools) == Set(["concat", "boom"])
                @test Set(r.uri for r in session.resources) == Set(["probe://greeting"])
                @test [p.name for p in session.prompts] == ["salute"]
            finally
                mcp_disconnect!(session)
            end
            @test session.status == :closed
        end

        @testset "list_tools! returns exact tool defs" begin
            session = mcp_connect(url)
            try
                tools = list_tools!(session)
                concat = only(filter(t -> t.name == "concat", tools))
                @test concat.description == "Concatenate a and b with a pipe"
                # The input schema field must round-trip from the server registration.
                @test concat.input_schema["type"] == "object"
                @test haskey(concat.input_schema["properties"], "a")
                @test concat.input_schema["properties"]["a"]["type"] == "string"
                @test "a" in concat.input_schema["required"]
            finally
                mcp_disconnect!(session)
            end
        end

        @testset "list_resources! returns exact resource defs" begin
            session = mcp_connect(url)
            try
                resources = list_resources!(session)
                r = only(resources)
                @test r.uri == "probe://greeting"
                @test r.name == "greeting"
                @test r.description == "a canned greeting"
                @test r.mime_type == "text/plain"
            finally
                mcp_disconnect!(session)
            end
        end

        @testset "list_prompts! returns exact prompt defs" begin
            session = mcp_connect(url)
            try
                prompts = list_prompts!(session)
                p = only(prompts)
                @test p.name == "salute"
                @test p.description == "Greet someone"
                @test !isnothing(p.arguments)
                @test p.arguments[1]["name"] == "who"
                @test p.arguments[1]["required"] == true
            finally
                mcp_disconnect!(session)
            end
        end

        @testset "call_tool returns exact computed result and sends args" begin
            session = mcp_connect(url)
            try
                captured[] = nothing
                out = call_tool(session, "concat",
                    Dict{String,Any}("a" => "foo", "b" => "bar"))
                @test out == "foo|bar"  # exact deterministic handler output
                # Prove the args reached the server over the wire.
                @test captured[] == Dict{String,Any}("a" => "foo", "b" => "bar")
                # Different args → different exact result (falsifies a constant-return bug).
                @test call_tool(session, "concat",
                    Dict{String,Any}("a" => "x", "b" => "yz")) == "x|yz"
            finally
                mcp_disconnect!(session)
            end
        end

        @testset "read_resource (static) returns exact content" begin
            session = mcp_connect(url)
            try
                @test read_resource(session, "probe://greeting") == "hello-from-resource"
            finally
                mcp_disconnect!(session)
            end
        end

        @testset "read_resource (template) returns templated content" begin
            session = mcp_connect(url)
            try
                # Template handler returns "echo:" * captured path segment.
                @test read_resource(session, "probe://echo/banana") == "echo:banana"
            finally
                mcp_disconnect!(session)
            end
        end

        @testset "get_prompt returns exact rendered messages" begin
            session = mcp_connect(url)
            try
                msgs = get_prompt(session, "salute", Dict{String,Any}("who" => "Ada"))
                @test length(msgs) == 1
                @test msgs[1]["role"] == "user"
                @test msgs[1]["content"]["type"] == "text"
                @test msgs[1]["content"]["text"] == "Hello, Ada!"
                # Default-arg path: handler falls back to "world".
                msgs2 = get_prompt(session, "salute", Dict{String,Any}())
                @test msgs2[1]["content"]["text"] == "Hello, world!"
            finally
                mcp_disconnect!(session)
            end
        end

        @testset "ping succeeds and returns nothing" begin
            session = mcp_connect(url)
            try
                @test ping(session) === nothing
            finally
                mcp_disconnect!(session)
            end
        end

        @testset "do-block form runs and auto-disconnects" begin
            ran = Ref(false)
            captured_session = Ref{Union{MCPSession,Nothing}}(nothing)
            ret = mcp_connect(url) do session
                captured_session[] = session
                @test session.status == :ready
                ran[] = true
                call_tool(session, "concat", Dict{String,Any}("a" => "p", "b" => "q"))
            end
            @test ran[]                       # block executed
            @test ret == "p|q"                # block's value is returned
            @test captured_session[].status == :closed  # auto-disconnected after block
        end

        @testset "error path: unknown tool raises MCPError(-32602)" begin
            session = mcp_connect(url)
            try
                err = nothing
                try
                    call_tool(session, "does_not_exist", Dict{String,Any}())
                catch e
                    err = e
                end
                @test err isa MCPError
                @test err.code == -32602
                @test contains(err.message, "Unknown tool: does_not_exist")
            finally
                mcp_disconnect!(session)
            end
        end

        @testset "tool handler that throws → isError → ErrorException" begin
            # Server CATCHES the handler error and returns isError=true content;
            # the client's call_tool then raises a plain ErrorException (NOT MCPError).
            session = mcp_connect(url)
            try
                @test_throws ErrorException call_tool(session, "boom", Dict{String,Any}())
            finally
                mcp_disconnect!(session)
            end
        end

        @testset "unknown resource URI raises MCPError(-32002)" begin
            session = mcp_connect(url)
            try
                err = nothing
                try
                    read_resource(session, "probe://nope")
                catch e
                    err = e
                end
                @test err isa MCPError
                @test err.code == -32002
                @test contains(err.message, "Resource not found")
            finally
                mcp_disconnect!(session)
            end
        end
    finally
        close(httpserver)
    end
end

@testset "MCP client ↔ HTTP server framed as SSE" begin
    # Same dispatch, but responses are wrapped as Server-Sent-Events so the
    # HTTPTransport must route through `_parse_sse_response`. Single canned
    # frame per request (deterministic, not chunked streaming).
    captured = Ref{Any}(nothing)
    server = _build_mcp_test_server(captured)
    port = _free_port()
    httpserver = HTTP.serve!("127.0.0.1", port; verbose=false) do req
        req.method == "DELETE" && return HTTP.Response(200, "")
        req.method == "POST" || return HTTP.Response(405, "Method Not Allowed")
        parsed = JSON.parse(String(req.body); dicttype=Dict{String,Any})
        resp = UniLM._dispatch_mcp(server, parsed)
        isnothing(resp) && return HTTP.Response(202, "")
        sse = "event: message\ndata: " * JSON.json(resp) * "\n\n"
        HTTP.Response(200, ["Content-Type" => "text/event-stream"], sse)
    end
    url = "http://127.0.0.1:$port"
    try
        session = mcp_connect(url)  # handshake itself must parse SSE frames
        try
            @test session.status == :ready
            @test session.server_info["name"] == "unilm-test-mcp"
            # Operations still return correct values through the SSE parser.
            @test call_tool(session, "concat",
                Dict{String,Any}("a" => "se", "b" => "se")) == "se|se"
            @test read_resource(session, "probe://greeting") == "hello-from-resource"
        finally
            mcp_disconnect!(session)
        end
    finally
        close(httpserver)
    end
end

@testset "MCP client HTTPTransport DELETE-on-disconnect path" begin
    # Server echoes back an Mcp-Session-Id; the client captures it, so
    # mcp_disconnect! takes the branch that issues a DELETE to the endpoint.
    captured = Ref{Any}(nothing)
    server = _build_mcp_test_server(captured)
    delete_hit = Ref(false)
    port = _free_port()
    httpserver = HTTP.serve!("127.0.0.1", port; verbose=false) do req
        if req.method == "DELETE"
            delete_hit[] = true
            return HTTP.Response(200, "")
        end
        req.method == "POST" || return HTTP.Response(405, "Method Not Allowed")
        parsed = JSON.parse(String(req.body); dicttype=Dict{String,Any})
        resp = UniLM._dispatch_mcp(server, parsed)
        isnothing(resp) && return HTTP.Response(202,
            ["Mcp-Session-Id" => "sess-123"], "")
        HTTP.Response(200,
            ["Content-Type" => "application/json", "Mcp-Session-Id" => "sess-123"],
            JSON.json(resp))
    end
    url = "http://127.0.0.1:$port"
    try
        session = mcp_connect(url)
        # The transport must have captured the server-assigned session id.
        @test session.transport.session_id == "sess-123"
        mcp_disconnect!(session)
        @test session.status == :closed
        @test delete_hit[]  # DELETE branch in _transport_disconnect! ran
    finally
        close(httpserver)
    end
end
