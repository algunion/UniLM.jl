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

@testset "SSE frame extraction" begin
    body = "event: message\ndata: {\"jsonrpc\":\"2.0\",\"id\":1,\"result\":{}}\n\n"
    frames = UniLM._parse_sse_frames(body)
    @test length(frames) == 1
    parsed = JSON.parse(frames[1])
    @test parsed["id"] == 1
    @test haskey(parsed, "result")

    # Multi-frame bodies come back complete and in arrival order.
    multi = "data: {\"a\":1}\n\nevent: message\ndata: {\"b\":2}\n\n"
    @test UniLM._parse_sse_frames(multi) == ["{\"a\":1}", "{\"b\":2}"]
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

        @testset "call_tool text-only result is a typed MCPToolResult, sends args" begin
            session = mcp_connect(url)
            try
                captured[] = nothing
                out = call_tool(session, "concat",
                    Dict{String,Any}("a" => "foo", "b" => "bar"))
                # Text-only result: typed struct; content is the joined text; no
                # structuredContent; not an error; raw content array preserved verbatim.
                @test out isa MCPToolResult
                @test out.content == "foo|bar"    # exact deterministic handler output
                @test out.is_error == false
                @test out.structured === nothing
                @test out.parts == Any[Dict{String,Any}("type" => "text", "text" => "foo|bar")]
                # Prove the args reached the server over the wire.
                @test captured[] == Dict{String,Any}("a" => "foo", "b" => "bar")
                # Different args → different exact result (falsifies a constant-return bug).
                @test call_tool(session, "concat",
                    Dict{String,Any}("a" => "x", "b" => "yz")).content == "x|yz"
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
            @test ret.content == "p|q"        # block's value (an MCPToolResult) is returned
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

        @testset "tool execution error is returned typed, not thrown" begin
            # Server CATCHES the handler error and returns isError=true content.
            # call_tool must RETURN it as data (is_error=true, faithful content) —
            # NOT throw — so a tool-execution error is distinguishable from a
            # JSON-RPC protocol error (which still throws MCPError, tested above).
            session = mcp_connect(url)
            try
                res = call_tool(session, "boom", Dict{String,Any}())
                @test res isa MCPToolResult
                @test res.is_error == true
                @test res.content == "Error: kaboom"   # server's showerror text, verbatim
                @test res.structured === nothing
                @test res.parts == Any[Dict{String,Any}("type" => "text", "text" => "Error: kaboom")]
                # The tool-loop bridge surfaces the faithful content by RAISING it,
                # so the loop records an unsuccessful outcome (not a silent success).
                boom = only(filter(t -> t.tool.func.name == "boom", mcp_tools(session)))
                err = try
                    boom.callable("boom", Dict{String,Any}())
                    nothing
                catch e
                    e
                end
                @test err isa ErrorException
                @test err.msg == "Error: kaboom"       # faithful content on the raised error
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
    # HTTPTransport must route through `_parse_sse_frames`. Single canned
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
                Dict{String,Any}("a" => "se", "b" => "se")).content == "se|se"
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

# ─── StdioTransport: drive the CLIENT against a REAL subprocess MCP server ─────
# The most common MCP deployment is a child process speaking newline-delimited
# JSON-RPC over stdin/stdout. We spawn a Julia child that `using UniLM` and runs
# `UniLM._serve_stdio(server)` (which FLUSHES stdout per response → the parent's
# `readline` never deadlocks). This is the ONLY way to exercise
# `_transport_connect!/_transport_send!/_transport_notify!/_transport_disconnect!`
# for StdioTransport and the `mcp_connect(::Cmd)` entry point — all of which the
# HTTP path cannot reach.
#
# DETERMINISM: stdio is synchronous request/response (no timing races). A watchdog
# (`timedwait` + kill-by-unique-marker) guarantees a pathologically hung child can
# never block the suite, and `finally` always tears the child down even on an
# assertion failure (`failfast` semantics in CI).
@testset "MCP client ↔ subprocess stdio server" begin
    # Unique marker embedded in the child program so we can pkill it by name even
    # if we never obtain its Process handle (e.g. a hung handshake).
    marker = "UNILMSTDIOSRV" * string(rand(UInt64); base=16)
    child_src = """
    # $marker
    using UniLM
    server = UniLM.MCPServer("stdio-probe-mcp", "7.7.7"; description="stdio probe")
    UniLM.register_tool!(server, "concat", "Concatenate a and b with a pipe",
        Dict{String,Any}("type" => "object",
            "properties" => Dict{String,Any}(
                "a" => Dict{String,Any}("type" => "string"),
                "b" => Dict{String,Any}("type" => "string")),
            "required" => ["a", "b"]),
        function (args::Dict{String,Any})
            string(args["a"]) * "|" * string(args["b"])
        end)
    UniLM._serve_stdio(server)
    """
    childfile, childio = mktemp()
    write(childio, child_src)
    close(childio)
    proj = dirname(dirname(pathof(UniLM)))
    cmd = `$(Base.julia_cmd()) --startup-file=no --project=$proj $childfile`

    # Connect on a worker task guarded by a watchdog so a hung child cannot hang
    # the suite. On success the worker stores the live MCPSession.
    box = Ref{Any}(nothing)
    worker = @async (box[] = try mcp_connect(cmd) catch e; e end)
    connected = timedwait(() -> istaskdone(worker), 90.0)
    try
        if connected !== :ok
            # Pathological hang: kill the child by marker, fail loud, do not block.
            try; run(pipeline(`pkill -f $marker`; stderr=devnull)); catch; end
            @test connected === :ok  # falsifies: handshake completed within budget
        else
            session = box[]
            @test session isa MCPSession           # not a thrown exception
            if session isa MCPSession
                try
                    # Handshake completed over a real pipe.
                    @test session.status == :ready
                    @test UniLM._transport_isconnected(session.transport) == true
                    @test session.transport isa StdioTransport
                    # serverInfo carries the EXACT values the child registered.
                    @test session.server_info["name"] == "stdio-probe-mcp"
                    @test session.server_info["version"] == "7.7.7"
                    # A real round-trip request → exact deterministic handler output.
                    @test call_tool(session, "concat",
                        Dict{String,Any}("a" => "foo", "b" => "bar")).content == "foo|bar"
                    # Different args → different exact result (falsifies constant-return).
                    @test call_tool(session, "concat",
                        Dict{String,Any}("a" => "x", "b" => "yz")).content == "x|yz"
                finally
                    mcp_disconnect!(session)
                end
                # Teardown actually closed the subprocess.
                @test session.status == :closed
                @test UniLM._transport_isconnected(session.transport) == false
            end
        end
    finally
        # Belt-and-suspenders: ensure no child of ours survives, whatever happened.
        try; run(pipeline(`pkill -f $marker`; stderr=devnull)); catch; end
        rm(childfile; force=true)
    end
end

@testset "MCP client HTTPTransport non-200 → error with status" begin
    # The server returns HTTP 500 for the initialize POST. `_transport_send!`
    # must raise (NOT silently return), and the message must name the status —
    # this is the `resp.status == 200 || error(...)` branch (mcp_client.jl:255).
    port = _free_port()
    httpserver = HTTP.serve!("127.0.0.1", port; verbose=false) do req
        req.method == "DELETE" && return HTTP.Response(200, "")
        HTTP.Response(500, "upstream exploded")
    end
    url = "http://127.0.0.1:$port"
    try
        err = nothing
        try
            mcp_connect(url)  # handshake's first POST hits the 500
        catch e
            err = e
        end
        @test err isa ErrorException                    # raised, not swallowed
        @test contains(err.msg, "MCP HTTP request failed with status 500")
        @test contains(err.msg, "upstream exploded")    # body echoed into the error
    finally
        close(httpserver)
    end
end

@testset "MCP client list_* pagination follows nextCursor" begin
    # A stateful handler hand-crafts JSON-RPC responses: the FIRST tools/list (and
    # resources/list, prompts/list) returns ONE item + a `nextCursor`; the SECOND
    # (carrying that cursor) returns the remaining item and NO cursor. Merged
    # results containing items from BOTH pages prove the cursor loop iterated past
    # its first `break` (mcp_client.jl:473/496/519). `initialize` advertises NO
    # capabilities so `mcp_connect` does NOT auto-list — we drive list_*! ourselves.
    calls = Dict{String,Int}("tools/list" => 0, "resources/list" => 0, "prompts/list" => 0)
    seen_cursor = Dict{String,Any}("tools/list" => :none, "resources/list" => :none, "prompts/list" => :none)
    port = _free_port()
    httpserver = HTTP.serve!("127.0.0.1", port; verbose=false) do req
        req.method == "DELETE" && return HTTP.Response(200, "")
        req.method == "POST" || return HTTP.Response(405, "Method Not Allowed")
        parsed = JSON.parse(String(req.body); dicttype=Dict{String,Any})
        method = get(parsed, "method", "")
        id = get(parsed, "id", nothing)
        isnothing(id) && return HTTP.Response(202, "")  # notification (initialized)
        local result::Dict{String,Any}
        if method == "initialize"
            # No capabilities → no handshake auto-list; we call list_*! explicitly.
            result = Dict{String,Any}(
                "protocolVersion" => UniLM._MCP_PROTOCOL_VERSION,
                "capabilities" => Dict{String,Any}(),
                "serverInfo" => Dict{String,Any}("name" => "pager", "version" => "1.0"))
        elseif method == "ping"
            result = Dict{String,Any}()
        elseif haskey(calls, method)
            calls[method] += 1
            n = calls[method]
            params = get(parsed, "params", Dict{String,Any}())
            seen_cursor[method] = get(params, "cursor", :none)
            key = split(method, "/")[1]  # "tools" | "resources" | "prompts"
            if key == "tools"
                if n == 1
                    result = Dict{String,Any}("tools" => [Dict{String,Any}("name" => "page1tool",
                        "inputSchema" => Dict{String,Any}("type" => "object"))],
                        "nextCursor" => "CUR-tools")
                else
                    result = Dict{String,Any}("tools" => [Dict{String,Any}("name" => "page2tool",
                        "inputSchema" => Dict{String,Any}("type" => "object"))])
                end
            elseif key == "resources"
                if n == 1
                    result = Dict{String,Any}("resources" => [Dict{String,Any}(
                        "uri" => "probe://p1", "name" => "page1res")], "nextCursor" => "CUR-res")
                else
                    result = Dict{String,Any}("resources" => [Dict{String,Any}(
                        "uri" => "probe://p2", "name" => "page2res")])
                end
            else  # prompts
                if n == 1
                    result = Dict{String,Any}("prompts" => [Dict{String,Any}("name" => "page1prompt")],
                        "nextCursor" => "CUR-prompts")
                else
                    result = Dict{String,Any}("prompts" => [Dict{String,Any}("name" => "page2prompt")])
                end
            end
        else
            return HTTP.Response(200, ["Content-Type" => "application/json"],
                JSON.json(Dict{String,Any}("jsonrpc" => "2.0", "id" => id,
                    "error" => Dict{String,Any}("code" => -32601, "message" => "Method not found"))))
        end
        HTTP.Response(200, ["Content-Type" => "application/json"],
            JSON.json(Dict{String,Any}("jsonrpc" => "2.0", "id" => id, "result" => result)))
    end
    url = "http://127.0.0.1:$port"
    try
        session = mcp_connect(url)
        try
            @test isnothing(session.server_capabilities.tools)  # confirm: no auto-list

            tools = list_tools!(session)
            @test [t.name for t in tools] == ["page1tool", "page2tool"]  # BOTH pages merged
            @test calls["tools/list"] == 2                # loop made a second request
            @test seen_cursor["tools/list"] == "CUR-tools"  # 2nd request carried the cursor
            @test session.tools == tools                  # cache stored the merge

            resources = list_resources!(session)
            @test [r.uri for r in resources] == ["probe://p1", "probe://p2"]
            @test [r.name for r in resources] == ["page1res", "page2res"]
            @test calls["resources/list"] == 2
            @test seen_cursor["resources/list"] == "CUR-res"

            prompts = list_prompts!(session)
            @test [p.name for p in prompts] == ["page1prompt", "page2prompt"]
            @test calls["prompts/list"] == 2
            @test seen_cursor["prompts/list"] == "CUR-prompts"
        finally
            mcp_disconnect!(session)
        end
    finally
        close(httpserver)
    end
end

@testset "MCP client call_tool non-text & non-dict content parts" begin
    # `tools/call` content has THREE part shapes the client must render distinctly:
    #   - {"type":"text", "text":...}   → push the raw text
    #   - a Dict with a non-"text" type → JSON-encode the whole part
    #   - a non-Dict element (bare string) → `string(part)`
    # call_tool joins them with "\n" in order (rendering preserved verbatim from
    # before the MCPToolResult change), while `parts` keeps the raw array verbatim.
    port = _free_port()
    httpserver = HTTP.serve!("127.0.0.1", port; verbose=false) do req
        req.method == "DELETE" && return HTTP.Response(200, "")
        req.method == "POST" || return HTTP.Response(405, "Method Not Allowed")
        parsed = JSON.parse(String(req.body); dicttype=Dict{String,Any})
        id = get(parsed, "id", nothing)
        method = get(parsed, "method", "")
        isnothing(id) && return HTTP.Response(202, "")
        result = if method == "initialize"
            Dict{String,Any}("protocolVersion" => UniLM._MCP_PROTOCOL_VERSION,
                "capabilities" => Dict{String,Any}(),
                "serverInfo" => Dict{String,Any}("name" => "mixed", "version" => "1.0"))
        elseif method == "tools/call"
            Dict{String,Any}("content" => Any[
                Dict{String,Any}("type" => "text", "text" => "alpha"),
                Dict{String,Any}("type" => "image", "data" => "QUJD", "mimeType" => "image/png"),
                "bare-string-part"])
        else
            Dict{String,Any}()
        end
        HTTP.Response(200, ["Content-Type" => "application/json"],
            JSON.json(Dict{String,Any}("jsonrpc" => "2.0", "id" => id, "result" => result)))
    end
    url = "http://127.0.0.1:$port"
    try
        session = mcp_connect(url)
        try
            out = call_tool(session, "whatever", Dict{String,Any}())
            @test out isa MCPToolResult
            @test out.is_error == false
            @test out.structured === nothing
            lines = split(out.content, "\n")
            @test length(lines) == 3
            @test lines[1] == "alpha"                                  # text branch
            # Non-text Dict part → JSON-encoded whole part. Re-parse to assert exact
            # fields (key order in JSON is unspecified, so compare the parsed dict).
            img = JSON.parse(lines[2]; dicttype=Dict{String,Any})
            @test img == Dict{String,Any}("type" => "image", "data" => "QUJD", "mimeType" => "image/png")
            @test lines[3] == "bare-string-part"                       # non-Dict → string(part)
            # The raw content array is preserved verbatim in `parts`.
            @test out.parts == Any[
                Dict{String,Any}("type" => "text", "text" => "alpha"),
                Dict{String,Any}("type" => "image", "data" => "QUJD", "mimeType" => "image/png"),
                "bare-string-part"]
        finally
            mcp_disconnect!(session)
        end
    finally
        close(httpserver)
    end
end

@testset "MCP client call_tool surfaces structuredContent" begin
    # A tools/call result may carry `structuredContent` (a typed object) alongside
    # textual content or on its own. call_tool must capture it verbatim in
    # `structured`. The tool-loop bridge returns `content` when present and falls
    # back to a JSON encoding of `structured` only when `content` is empty.
    structured_obj = Dict{String,Any}("temperature" => 21, "unit" => "C",
        "nested" => Dict{String,Any}("ok" => true))
    port = _free_port()
    httpserver = HTTP.serve!("127.0.0.1", port; verbose=false) do req
        req.method == "DELETE" && return HTTP.Response(200, "")
        req.method == "POST" || return HTTP.Response(405, "Method Not Allowed")
        parsed = JSON.parse(String(req.body); dicttype=Dict{String,Any})
        id = get(parsed, "id", nothing)
        method = get(parsed, "method", "")
        isnothing(id) && return HTTP.Response(202, "")
        result = if method == "initialize"
            Dict{String,Any}("protocolVersion" => UniLM._MCP_PROTOCOL_VERSION,
                "capabilities" => Dict{String,Any}(),
                "serverInfo" => Dict{String,Any}("name" => "structured", "version" => "1.0"))
        elseif method == "tools/call"
            tname = get(get(parsed, "params", Dict{String,Any}()), "name", "")
            if tname == "with_text"
                # structuredContent alongside a text part.
                Dict{String,Any}(
                    "content" => Any[Dict{String,Any}("type" => "text", "text" => "summary")],
                    "structuredContent" => structured_obj,
                    "isError" => false)
            else
                # "structured_only": structuredContent, empty textual content.
                Dict{String,Any}(
                    "content" => Any[],
                    "structuredContent" => structured_obj,
                    "isError" => false)
            end
        else
            Dict{String,Any}()
        end
        HTTP.Response(200, ["Content-Type" => "application/json"],
            JSON.json(Dict{String,Any}("jsonrpc" => "2.0", "id" => id, "result" => result)))
    end
    url = "http://127.0.0.1:$port"
    try
        session = mcp_connect(url)
        try
            # (1) structuredContent alongside text: structured captured verbatim
            # (including the nested object), text still rendered, not an error.
            withtext = call_tool(session, "with_text", Dict{String,Any}())
            @test withtext isa MCPToolResult
            @test withtext.is_error == false
            @test withtext.content == "summary"
            @test withtext.structured == structured_obj
            @test withtext.structured !== nothing
            # Bridge returns the textual content when content is present (structured
            # does NOT override a non-empty content).
            @test UniLM._mcp_tool_dispatch(withtext) == "summary"

            # (2) structured-only: content renders empty, structured captured verbatim.
            sonly = call_tool(session, "structured_only", Dict{String,Any}())
            @test sonly.content == ""
            @test sonly.structured == structured_obj
            @test sonly.is_error == false
            # Bridge falls back to a JSON encoding of `structured` when content is empty.
            @test JSON.parse(UniLM._mcp_tool_dispatch(sonly); dicttype=Dict{String,Any}) == structured_obj
        finally
            mcp_disconnect!(session)
        end
    finally
        close(httpserver)
    end
end

@testset "MCP client read_resource blob content branch" begin
    # resources/read may return `blob` contents (base64) instead of `text`. The
    # client must push the blob string (mcp_client.jl:568-569). A mixed list
    # (text + blob) proves both branches and the "\n" join order.
    port = _free_port()
    httpserver = HTTP.serve!("127.0.0.1", port; verbose=false) do req
        req.method == "DELETE" && return HTTP.Response(200, "")
        req.method == "POST" || return HTTP.Response(405, "Method Not Allowed")
        parsed = JSON.parse(String(req.body); dicttype=Dict{String,Any})
        id = get(parsed, "id", nothing)
        method = get(parsed, "method", "")
        isnothing(id) && return HTTP.Response(202, "")
        result = if method == "initialize"
            Dict{String,Any}("protocolVersion" => UniLM._MCP_PROTOCOL_VERSION,
                "capabilities" => Dict{String,Any}(),
                "serverInfo" => Dict{String,Any}("name" => "blobby", "version" => "1.0"))
        elseif method == "resources/read"
            Dict{String,Any}("contents" => Any[
                Dict{String,Any}("uri" => "probe://x", "text" => "plain-text"),
                Dict{String,Any}("uri" => "probe://x", "blob" => "YmluYXJ5LWJsb2I=")])
        else
            Dict{String,Any}()
        end
        HTTP.Response(200, ["Content-Type" => "application/json"],
            JSON.json(Dict{String,Any}("jsonrpc" => "2.0", "id" => id, "result" => result)))
    end
    url = "http://127.0.0.1:$port"
    try
        session = mcp_connect(url)
        try
            # text part then blob part, joined by "\n" in order.
            @test read_resource(session, "probe://x") == "plain-text\nYmluYXJ5LWJsb2I="
        finally
            mcp_disconnect!(session)
        end
    finally
        close(httpserver)
    end
end

@testset "MCP client HTTPTransport disconnect-failure is swallowed" begin
    # If the DELETE on disconnect raises (server already gone), `_transport_disconnect!`
    # must catch it and still mark the session closed (mcp_client.jl:280-281 @debug).
    # We give the client a session_id (so the DELETE branch is taken), then CLOSE the
    # server before disconnecting so the DELETE request throws a connection error.
    port = _free_port()
    httpserver = HTTP.serve!("127.0.0.1", port; verbose=false) do req
        req.method == "DELETE" && return HTTP.Response(200, "")
        req.method == "POST" || return HTTP.Response(405, "Method Not Allowed")
        parsed = JSON.parse(String(req.body); dicttype=Dict{String,Any})
        id = get(parsed, "id", nothing)
        isnothing(id) && return HTTP.Response(202, ["Mcp-Session-Id" => "sid-9"], "")
        result = Dict{String,Any}("protocolVersion" => UniLM._MCP_PROTOCOL_VERSION,
            "capabilities" => Dict{String,Any}(),
            "serverInfo" => Dict{String,Any}("name" => "doomed", "version" => "1.0"))
        HTTP.Response(200,
            ["Content-Type" => "application/json", "Mcp-Session-Id" => "sid-9"],
            JSON.json(Dict{String,Any}("jsonrpc" => "2.0", "id" => id, "result" => result)))
    end
    url = "http://127.0.0.1:$port"
    session = mcp_connect(url)
    @test session.transport.session_id == "sid-9"   # DELETE branch will be taken
    close(httpserver)                                # kill the endpoint → DELETE will throw
    # Must NOT propagate the connection error; must still close cleanly.
    @test mcp_disconnect!(session) === nothing
    @test session.status == :closed
    @test UniLM._transport_isconnected(session.transport) == false
end

# ─── Interleaved server→client frames & the session lock ─────────────────────
# All deterministic: in-memory IOBuffers on a StdioTransport (no subprocess,
# no timing). `t.output` is pre-loaded with the exact frames "the server"
# sends; `t.input` captures what the client writes back.

"Fresh :ready session over IOBuffers pre-loaded with `server_frames`."
function _iobuf_session(server_frames::String)
    t = UniLM.StdioTransport(`cat`)   # command is never spawned
    t.input = IOBuffer()
    t.output = IOBuffer(server_frames)
    session = UniLM.MCPSession(t, UniLM.MCPServerCapabilities(), Dict{String,Any}(),
        UniLM.MCPToolInfo[], UniLM.MCPResourceInfo[], UniLM.MCPPromptInfo[],
        UniLM._MCP_PROTOCOL_VERSION, 0, :ready)
    session, t
end

@testset "client answers server-initiated ping inline, then reads on" begin
    session, t = _iobuf_session(
        """{"jsonrpc":"2.0","id":"srv-ping-1","method":"ping"}\n""" *
        """{"jsonrpc":"2.0","id":1,"result":{"ok":true}}\n""")
    res = UniLM._mcp_request!(session, "tools/list")
    @test res == Dict{String,Any}("ok" => true)
    # The client wrote TWO frames: our request, then the ping answer.
    written = split(String(take!(t.input)), "\n"; keepempty=false)
    @test length(written) == 2
    req = JSON.parse(written[1])
    @test req["method"] == "tools/list"
    @test req["id"] == 1
    pong = JSON.parse(written[2])
    @test pong["id"] == "srv-ping-1"
    @test pong["result"] == Dict{String,Any}()
    @test !haskey(pong, "error")
end

@testset "client rejects unknown server-initiated requests with -32601" begin
    session, t = _iobuf_session(
        """{"jsonrpc":"2.0","id":"srv-req-9","method":"sampling/createMessage","params":{}}\n""" *
        """{"jsonrpc":"2.0","id":1,"result":{}}\n""")
    res = UniLM._mcp_request!(session, "ping")
    @test res == Dict{String,Any}()
    written = split(String(take!(t.input)), "\n"; keepempty=false)
    @test length(written) == 2
    reply = JSON.parse(written[2])
    @test reply["id"] == "srv-req-9"
    @test reply["error"]["code"] == -32601
end

@testset "notifications/tools/list_changed marks the tool cache stale" begin
    session, _ = _iobuf_session(
        """{"jsonrpc":"2.0","method":"notifications/tools/list_changed"}\n""" *
        """{"jsonrpc":"2.0","id":1,"result":{}}\n""" *
        """{"jsonrpc":"2.0","id":2,"result":{"tools":[{"name":"fresh","inputSchema":{"type":"object"}}]}}\n""")
    @test session.tools_stale == false
    UniLM._mcp_request!(session, "ping")
    @test session.tools_stale == true
    # Refreshing the list stores the new tools and clears the flag.
    tools = list_tools!(session)
    @test [tl.name for tl in tools] == ["fresh"]
    @test session.tools_stale == false
end

@testset "unrelated notifications are skipped without side effects" begin
    session, _ = _iobuf_session(
        """{"jsonrpc":"2.0","method":"notifications/resources/list_changed"}\n""" *
        """{"jsonrpc":"2.0","method":"notifications/message","params":{"level":"info","data":"hi"}}\n""" *
        """{"jsonrpc":"2.0","id":1,"result":{"ok":1}}\n""")
    res = UniLM._mcp_request!(session, "ping")
    @test res == Dict{String,Any}("ok" => 1)
    @test session.tools_stale == false
end

@testset "stale response frames with the wrong id are skipped" begin
    session, _ = _iobuf_session(
        """{"jsonrpc":"2.0","id":99,"result":{"stale":true}}\n""" *
        """{"jsonrpc":"2.0","id":1,"result":{"fresh":true}}\n""")
    res = @test_logs (:warn, r"unexpected id") UniLM._mcp_request!(session, "ping")
    @test res == Dict{String,Any}("fresh" => true)
end

@testset "id-null error response aborts the exchange loudly" begin
    session, _ = _iobuf_session(
        """{"jsonrpc":"2.0","id":null,"error":{"code":-32600,"message":"Invalid Request"}}\n""")
    err = try
        UniLM._mcp_request!(session, "ping")
        nothing
    catch e
        e
    end
    @test err isa MCPError
    @test err.code == -32600
end

# Minimal MCPTransport implementing only what the lock probe needs.
struct _LockProbeTransport <: UniLM.MCPTransport
    on_send::Function
    reply::String
end
UniLM._transport_send!(t::_LockProbeTransport, msg::String;
                       cfg::UniLM.RequestConfig=UniLM.current_config()) = (t.on_send(msg); t.reply)

@testset "whole exchange runs under the session lock" begin
    # The probe records whether session._lock is held while the request is on
    # the wire. Serialization of concurrent exchanges follows by construction
    # (one lock, whole exchange inside it); the interleaving itself is a
    # timing race and is deliberately not reproduced here.
    session_box = Ref{Any}(nothing)
    held_during_send = Ref(false)
    probe = _LockProbeTransport(
        _ -> (held_during_send[] = islocked(session_box[]._lock)),
        """{"jsonrpc":"2.0","id":1,"result":{}}""")
    session = UniLM.MCPSession(probe, UniLM.MCPServerCapabilities(), Dict{String,Any}(),
        UniLM.MCPToolInfo[], UniLM.MCPResourceInfo[], UniLM.MCPPromptInfo[],
        UniLM._MCP_PROTOCOL_VERSION, 0, :ready)
    session_box[] = session
    UniLM._mcp_request!(session, "ping")
    @test held_during_send[]
    # The id allocated under that same lock made it onto the wire.
    @test session._id_counter == 1
end

# ─── HTTP transport: multi-frame SSE bodies ──────────────────────────────────

@testset "HTTP SSE body: notification AFTER the response is not mistaken for it" begin
    captured = Ref{Any}(nothing)
    server = _build_mcp_test_server(captured)
    port = _free_port()
    httpserver = HTTP.serve!("127.0.0.1", port; verbose=false) do req
        req.method == "DELETE" && return HTTP.Response(200, "")
        req.method == "POST" || return HTTP.Response(405, "Method Not Allowed")
        parsed = JSON.parse(String(req.body); dicttype=Dict{String,Any})
        resp = UniLM._dispatch_mcp(server, parsed)
        isnothing(resp) && return HTTP.Response(202, "")
        # Response frame FIRST, then a trailing notification on the same stream.
        sse = "event: message\ndata: " * JSON.json(resp) * "\n\n" *
              "event: message\ndata: {\"jsonrpc\":\"2.0\",\"method\":\"notifications/message\",\"params\":{}}\n\n"
        HTTP.Response(200, ["Content-Type" => "text/event-stream"], sse)
    end
    url = "http://127.0.0.1:$port"
    try
        session = mcp_connect(url)
        try
            @test session.server_info["name"] == "unilm-test-mcp"
            @test call_tool(session, "concat",
                Dict{String,Any}("a" => "L", "b" => "R")).content == "L|R"
        finally
            mcp_disconnect!(session)
        end
    finally
        close(httpserver)
    end
end

@testset "HTTP SSE body: notification BEFORE the response is skipped (stale flag set)" begin
    captured = Ref{Any}(nothing)
    server = _build_mcp_test_server(captured)
    port = _free_port()
    httpserver = HTTP.serve!("127.0.0.1", port; verbose=false) do req
        req.method == "DELETE" && return HTTP.Response(200, "")
        req.method == "POST" || return HTTP.Response(405, "Method Not Allowed")
        parsed = JSON.parse(String(req.body); dicttype=Dict{String,Any})
        resp = UniLM._dispatch_mcp(server, parsed)
        isnothing(resp) && return HTTP.Response(202, "")
        sse = "data: {\"jsonrpc\":\"2.0\",\"method\":\"notifications/tools/list_changed\"}\n\n" *
              "data: " * JSON.json(resp) * "\n\n"
        HTTP.Response(200, ["Content-Type" => "text/event-stream"], sse)
    end
    url = "http://127.0.0.1:$port"
    try
        session = mcp_connect(url)
        try
            # Isolate the exchange under test: connect's own list_resources!/
            # list_prompts! responses were also prefixed with list_changed, so
            # reset the flag via a fresh tools listing first.
            list_tools!(session)
            @test session.tools_stale == false
            @test call_tool(session, "concat",
                Dict{String,Any}("a" => "x", "b" => "y")).content == "x|y"
            # The list_changed preceding that response marked the cache stale.
            @test session.tools_stale == true
        finally
            mcp_disconnect!(session)
        end
    finally
        close(httpserver)
    end
end

# ─── Protocol-version negotiation & HTTP session recovery ─────────────────────
# The client must validate the server's negotiated protocol version, carry the
# negotiated value on every request after initialize, and re-initialize a single
# time when the HTTP session expires (404). Each mock is a tiny in-process HTTP
# handler (no real MCPServer needed) so the exact wire behavior is scripted.

@testset "unsupported negotiated protocol version aborts the connection" begin
    # The server returns a protocolVersion this client cannot speak. mcp_connect
    # must refuse it and leave the transport CLOSED (no leaked connection),
    # naming both the requested and the returned versions in the error.
    port = _free_port()
    httpserver = HTTP.serve!("127.0.0.1", port; verbose=false) do req
        req.method == "DELETE" && return HTTP.Response(200, "")
        req.method == "POST" || return HTTP.Response(405, "Method Not Allowed")
        parsed = JSON.parse(String(req.body); dicttype=Dict{String,Any})
        id = get(parsed, "id", nothing)
        isnothing(id) && return HTTP.Response(202, "")
        result = Dict{String,Any}("protocolVersion" => "1999-01-01",  # unsupported
            "capabilities" => Dict{String,Any}(),
            "serverInfo" => Dict{String,Any}("name" => "ancient", "version" => "0.1"))
        HTTP.Response(200, ["Content-Type" => "application/json"],
            JSON.json(Dict{String,Any}("jsonrpc" => "2.0", "id" => id, "result" => result)))
    end
    url = "http://127.0.0.1:$port"
    transport = HTTPTransport(url)  # held so we can inspect it after the throw
    leaked = nothing
    try
        msg = try
            leaked = mcp_connect(transport)
            ""  # no throw (old behavior) → empty message fails the asserts cleanly
        catch e
            e isa ErrorException ? e.msg : sprint(showerror, e)
        end
        @test contains(msg, "1999-01-01")                    # the rejected server version
        @test contains(msg, UniLM._MCP_PROTOCOL_VERSION)     # the version we requested
        @test UniLM._transport_isconnected(transport) == false  # transport closed on exit
    finally
        isnothing(leaked) || mcp_disconnect!(leaked)
        close(httpserver)
    end
end

@testset "older negotiated protocol version is honored on later request headers" begin
    # The server negotiates DOWN to an older but still-supported revision. The
    # session must work, and every request AFTER initialize must carry that
    # negotiated version in Mcp-Protocol-Version — while the initialize request
    # itself advertised the client's latest supported version.
    negotiated = "2025-06-18"
    seen = Dict{String,String}()  # method => Mcp-Protocol-Version header sent
    port = _free_port()
    httpserver = HTTP.serve!("127.0.0.1", port; verbose=false) do req
        req.method == "DELETE" && return HTTP.Response(200, "")
        req.method == "POST" || return HTTP.Response(405, "Method Not Allowed")
        parsed = JSON.parse(String(req.body); dicttype=Dict{String,Any})
        method = get(parsed, "method", "")
        seen[method] = HTTP.header(req, "Mcp-Protocol-Version", "")
        id = get(parsed, "id", nothing)
        isnothing(id) && return HTTP.Response(202, "")
        result = method == "initialize" ?
            Dict{String,Any}("protocolVersion" => negotiated,
                "capabilities" => Dict{String,Any}(),
                "serverInfo" => Dict{String,Any}("name" => "downgrader", "version" => "1.0")) :
            Dict{String,Any}()
        HTTP.Response(200, ["Content-Type" => "application/json"],
            JSON.json(Dict{String,Any}("jsonrpc" => "2.0", "id" => id, "result" => result)))
    end
    url = "http://127.0.0.1:$port"
    session = mcp_connect(url)
    try
        @test session.protocol_version == negotiated       # stored on the session
        @test ping(session) === nothing                    # session still works
        @test seen["initialize"] == UniLM._MCP_PROTOCOL_VERSION  # init: client's latest
        @test seen["ping"] == negotiated                   # after init: negotiated value
    finally
        mcp_disconnect!(session)
    end
end

@testset "expired HTTP session is re-initialized and the request replayed" begin
    # The first post-initialize request 404s (session expired), then the server
    # succeeds. The client must re-initialize once (obtaining a fresh session id)
    # and replay the request — the call returns normally, initialize ran twice.
    init_count = Ref(0)
    ping_count = Ref(0)
    port = _free_port()
    httpserver = HTTP.serve!("127.0.0.1", port; verbose=false) do req
        req.method == "DELETE" && return HTTP.Response(200, "")
        req.method == "POST" || return HTTP.Response(405, "Method Not Allowed")
        parsed = JSON.parse(String(req.body); dicttype=Dict{String,Any})
        method = get(parsed, "method", "")
        id = get(parsed, "id", nothing)
        isnothing(id) && return HTTP.Response(202, "")
        if method == "initialize"
            init_count[] += 1
            return HTTP.Response(200,
                ["Content-Type" => "application/json", "Mcp-Session-Id" => "sess-$(init_count[])"],
                JSON.json(Dict{String,Any}("jsonrpc" => "2.0", "id" => id, "result" => Dict{String,Any}(
                    "protocolVersion" => UniLM._MCP_PROTOCOL_VERSION,
                    "capabilities" => Dict{String,Any}(),
                    "serverInfo" => Dict{String,Any}("name" => "expiry", "version" => "1.0")))))
        elseif method == "ping"
            ping_count[] += 1
            ping_count[] == 1 && return HTTP.Response(404, "session expired")  # first: gone
            return HTTP.Response(200, ["Content-Type" => "application/json"],
                JSON.json(Dict{String,Any}("jsonrpc" => "2.0", "id" => id, "result" => Dict{String,Any}())))
        end
        HTTP.Response(200, ["Content-Type" => "application/json"],
            JSON.json(Dict{String,Any}("jsonrpc" => "2.0", "id" => id, "result" => Dict{String,Any}())))
    end
    url = "http://127.0.0.1:$port"
    session = mcp_connect(url)
    try
        @test init_count[] == 1                          # one initialize at connect
        @test session.transport.session_id == "sess-1"
        @test ping(session) === nothing                  # 404 → re-init → replay ok
        @test init_count[] == 2                          # re-initialized exactly once
        @test ping_count[] == 2                          # first 404, replay succeeded
        @test session.transport.session_id == "sess-2"   # fresh session id now in use
    finally
        mcp_disconnect!(session)
    end
end

@testset "second HTTP session expiry after re-initialization aborts" begin
    # If the replayed request ALSO 404s, the client gives up — no retry loop. It
    # re-initializes exactly once, then the second expiry raises.
    init_count = Ref(0)
    port = _free_port()
    httpserver = HTTP.serve!("127.0.0.1", port; verbose=false) do req
        req.method == "DELETE" && return HTTP.Response(200, "")
        req.method == "POST" || return HTTP.Response(405, "Method Not Allowed")
        parsed = JSON.parse(String(req.body); dicttype=Dict{String,Any})
        method = get(parsed, "method", "")
        id = get(parsed, "id", nothing)
        isnothing(id) && return HTTP.Response(202, "")
        if method == "initialize"
            init_count[] += 1
            return HTTP.Response(200,
                ["Content-Type" => "application/json", "Mcp-Session-Id" => "sess-$(init_count[])"],
                JSON.json(Dict{String,Any}("jsonrpc" => "2.0", "id" => id, "result" => Dict{String,Any}(
                    "protocolVersion" => UniLM._MCP_PROTOCOL_VERSION,
                    "capabilities" => Dict{String,Any}(),
                    "serverInfo" => Dict{String,Any}("name" => "always-expired", "version" => "1.0")))))
        end
        HTTP.Response(404, "session expired")  # every request reports the session gone
    end
    url = "http://127.0.0.1:$port"
    session = mcp_connect(url)
    try
        @test_throws ErrorException ping(session)  # 404 → re-init → 404 again → give up
        @test init_count[] == 2                    # re-initialized once, then stopped
    finally
        mcp_disconnect!(session)
    end
end

@testset "HTTP 401 explains authentication is passed via headers" begin
    # The server rejects with 401. The error must name the status and point the
    # caller at the `headers` kwarg of mcp_connect (no built-in auth flow).
    port = _free_port()
    httpserver = HTTP.serve!("127.0.0.1", port; verbose=false) do req
        req.method == "DELETE" && return HTTP.Response(200, "")
        HTTP.Response(401, "Unauthorized")
    end
    url = "http://127.0.0.1:$port"
    try
        err = nothing
        try
            mcp_connect(url)  # the initialize POST is rejected
        catch e
            err = e
        end
        @test err isa ErrorException
        @test contains(err.msg, "401")          # names the status
        @test contains(err.msg, "headers")      # points at the headers kwarg…
        @test contains(err.msg, "mcp_connect")  # …of mcp_connect
    finally
        close(httpserver)
    end
end

@testset "WS4 mcp_connect captures resolved config + auto_respawn" begin
    captured = Ref{Any}(nothing)
    server = _build_mcp_test_server(captured)
    httpserver, url = _serve_mcp_json(server)
    try
        cfg = RequestConfig(current_config(); mcp_request_timeout = 7.0, mcp_connect_timeout = 11.0)
        session = mcp_connect(url; config = cfg, auto_respawn = true)
        try
            # Captured verbatim on the session.
            @test session.config.mcp_request_timeout == 7.0
            @test session.config.mcp_connect_timeout == 11.0
            @test session.auto_respawn == true
            @test session._closed_by_timeout == false
        finally
            mcp_disconnect!(session)
        end
        # Default channel: no config kwarg ⇒ current_config() captured, auto_respawn OFF.
        s2 = mcp_connect(url)
        try
            @test s2.config isa RequestConfig
            @test s2.auto_respawn == false
        finally
            mcp_disconnect!(s2)
        end
    finally
        close(httpserver)
    end
end

# ─── WS4: StdioTransport teardown ladder — no child survivors ─────────────────
# MCP spec: a compliant stdio server exits when its stdin reaches EOF, so
# `_kill_transport!` closes stdin first and escalates (SIGTERM, then an
# UNCONDITIONAL process-group SIGKILL) only as needed. The child leads its own
# process group (spawned detach=true), so the final group-SIGKILL by the
# spawn-captured pgid reaps a grandchild a wrapper forked and then orphaned by
# exiting politely on EOF. Two shapes are covered: a well-behaved child (graceful
# path — asserts the wiring: process/io handles AND pgid nulled) and a wrapper
# that forks a group-resident grandchild (OS-level leak falsifier: pgrep finds
# zero survivors after teardown).

# Shared WS4 stdio fixture. A raw JSON-RPC child (only `using JSON`, so it starts
# fast — no UniLM precompile): answers the handshake, advertises hang/incr/
# notify_then_ok, and exits cleanly when stdin reaches EOF. `marker` is embedded in
# a leading comment so the parent can pgrep/pkill it. No signal manipulation — it is
# reaped by the ladder's stdin-EOF (graceful) or SIGTERM (hung-in-handler) step.
function _ws4_raw_child_src(marker::String)
    ver = UniLM._MCP_PROTOCOL_VERSION
    """
    # $marker
    import JSON
    counter = Ref(0)
    while !eof(stdin)
        line = readline(stdin)
        isempty(line) && continue
        msg = JSON.parse(line)
        id = get(msg, "id", nothing)
        method = get(msg, "method", "")
        if method == "initialize"
            println(stdout, JSON.json(Dict("jsonrpc"=>"2.0","id"=>id,"result"=>Dict(
                "protocolVersion"=>"$ver",
                "capabilities"=>Dict("tools"=>Dict()),
                "serverInfo"=>Dict("name"=>"ws4-fixture","version"=>"1.0")))))
            flush(stdout)
        elseif method == "notifications/initialized"
            # no response
        elseif method == "tools/list"
            println(stdout, JSON.json(Dict("jsonrpc"=>"2.0","id"=>id,"result"=>Dict(
                "tools"=>[Dict("name"=>"hang","inputSchema"=>Dict("type"=>"object")),
                          Dict("name"=>"incr","inputSchema"=>Dict("type"=>"object")),
                          Dict("name"=>"notify_then_ok","inputSchema"=>Dict("type"=>"object"))]))))
            flush(stdout)
        elseif method == "tools/call"
            name = get(get(msg,"params",Dict()), "name", "")
            if name == "incr"
                counter[] += 1
                println(stdout, JSON.json(Dict("jsonrpc"=>"2.0","id"=>id,"result"=>Dict(
                    "content"=>[Dict("type"=>"text","text"=>string(counter[]))]))))
                flush(stdout)
            elseif name == "notify_then_ok"
                println(stdout, JSON.json(Dict("jsonrpc"=>"2.0","method"=>"notifications/message",
                    "params"=>Dict("level"=>"info","data"=>"working"))))
                flush(stdout)
                sleep(0.2)
                println(stdout, JSON.json(Dict("jsonrpc"=>"2.0","id"=>id,"result"=>Dict(
                    "content"=>[Dict("type"=>"text","text"=>"done")]))))
                flush(stdout)
            else  # hang: emit a notification first, then never respond
                println(stdout, JSON.json(Dict("jsonrpc"=>"2.0","method"=>"notifications/message",
                    "params"=>Dict("level"=>"info","data"=>"before-hang"))))
                flush(stdout)
                while true; sleep(3600); end
            end
        elseif !isnothing(id)
            println(stdout, JSON.json(Dict("jsonrpc"=>"2.0","id"=>id,"result"=>Dict())))
            flush(stdout)
        end
    end
    """
end

@testset "WS4 stdio disconnect routes through the ladder (handles + pgid cleared)" begin
    # Portable, deterministic wiring guard: mcp_disconnect! must delegate to
    # _kill_transport!, which nulls process/input/output AND the spawn-captured pgid.
    # RED on the wave base: the original _transport_disconnect! nulls the io handles
    # but not `pgid` (the field/ladder do not exist yet). GREEN once disconnect routes
    # through _kill_transport!. The actual group-SIGKILL of an orphaned grandchild is
    # pinned by the hang_matrix pin flipped in Step 5 (cross-platform process detection
    # is genuinely OS-specific — pgrep -f does not see an sh -c body on macOS — so that
    # OS-verified fixture is owned by the red suite, not duplicated here).
    marker = "UNILMDISC" * string(rand(UInt64); base=16)
    proj = dirname(dirname(pathof(UniLM)))
    childfile, io = mktemp(); write(io, _ws4_raw_child_src(marker)); close(io)
    jl = Base.julia_cmd()
    cmd = `$(jl) --startup-file=no --project=$proj $childfile`
    try
        session = mcp_connect(cmd)
        @test session.status == :ready
        @test UniLM._transport_isconnected(session.transport) == true
        @test session.transport.pgid !== nothing            # pgid captured at spawn (detached: child leads its own group)
        mcp_disconnect!(session)
        @test session.status == :closed
        @test UniLM._transport_isconnected(session.transport) == false
        @test session.transport.process === nothing
        @test session.transport.pgid === nothing            # ladder cleared it (new behavior)
    finally
        try; run(pipeline(`pkill -f $marker`; stderr=devnull)); catch; end
        rm(childfile; force=true)
    end
end

# WS4 wrapper fixture: forks a long-lived grandchild that INHERITS this child's
# process group (this child leads its own group under the parent's detach=true
# spawn), then answers only `initialize` and reads until stdin EOF. On disconnect
# the ladder closes stdin (this child exits, ORPHANING the grandchild) and then
# unconditionally group-SIGKILLs by the spawn-captured pgid — reaping the orphan.
# The grandchild marker rides in the `sh -c` argv so the parent can pgrep/pkill it.
function _ws4_wrapper_child_src(marker::String, gc_marker::String)
    ver = UniLM._MCP_PROTOCOL_VERSION
    """
    # $marker
    import JSON
    run(pipeline(`sh -c 'while :; do sleep 1; done # $(gc_marker)'`); wait=false)
    while !eof(stdin)
        line = readline(stdin)
        isempty(line) && continue
        msg = JSON.parse(line)
        id = get(msg, "id", nothing)
        method = get(msg, "method", "")
        if method == "initialize"
            println(stdout, JSON.json(Dict("jsonrpc"=>"2.0","id"=>id,"result"=>Dict(
                "protocolVersion"=>"$ver",
                "capabilities"=>Dict(),
                "serverInfo"=>Dict("name"=>"ws4-wrapper","version"=>"1.0")))))
            flush(stdout)
        end
    end
    """
end

@testset "WS4 stdio teardown group-kills an orphaned grandchild (OS leak falsifier)" begin
    # Falsifies "a wrapper's grandchild survives teardown". The child forks a
    # long-lived grandchild sharing its process group, then exits on stdin EOF —
    # orphaning it. The ladder's UNCONDITIONAL final rung group-SIGKILLs by the
    # spawn-captured pgid, reaping the orphan even though the direct child already
    # exited politely. LEAK FALSIFIER: after disconnect `pgrep -f gc_marker` must find
    # zero — a survivor keeps `timedwait` from ever reaching :ok and fails the @test.
    # The `pkill` in `finally` is a safety net, never the assertion.
    marker = "UNILMWRAP" * string(rand(UInt64); base=16)
    gc_marker = "UNILMGC" * string(rand(UInt64); base=16)
    proj = dirname(dirname(pathof(UniLM)))
    childfile, io = mktemp(); write(io, _ws4_wrapper_child_src(marker, gc_marker)); close(io)
    jl = Base.julia_cmd()
    cmd = `$(jl) --startup-file=no --project=$proj $childfile`
    _alive(m) = success(pipeline(`pgrep -f $m`; stdout=devnull, stderr=devnull))
    try
        session = mcp_connect(cmd)
        @test session.status == :ready
        @test session.transport.pgid !== nothing
        # The grandchild came up and shares the group before we tear down.
        @test timedwait(() -> _alive(gc_marker), 10.0) === :ok
        mcp_disconnect!(session)
        @test session.status == :closed
        # The orphaned grandchild is gone: the unconditional group SIGKILL reaped it.
        @test timedwait(() -> !_alive(gc_marker), 5.0) === :ok
    finally
        try; run(pipeline(`pkill -f $gc_marker`; stderr=devnull)); catch; end
        try; run(pipeline(`pkill -f $marker`; stderr=devnull)); catch; end
        rm(childfile; force=true)
    end
end

@testset "WS4 HTTP request timeout is typed and non-fatal" begin
    hit = Ref(0)
    port = _free_port()
    httpserver = HTTP.serve!("127.0.0.1", port; verbose=false) do req
        req.method == "DELETE" && return HTTP.Response(200, "")
        req.method == "POST" || return HTTP.Response(405, "Method Not Allowed")
        parsed = JSON.parse(String(req.body); dicttype=Dict{String,Any})
        id = get(parsed, "id", nothing); method = get(parsed, "method", "")
        isnothing(id) && return HTTP.Response(202, "")
        if method == "initialize"
            return HTTP.Response(200, ["Content-Type" => "application/json"],
                JSON.json(Dict{String,Any}("jsonrpc"=>"2.0","id"=>id,"result"=>Dict{String,Any}(
                    "protocolVersion"=>UniLM._MCP_PROTOCOL_VERSION,
                    "capabilities"=>Dict{String,Any}(),
                    "serverInfo"=>Dict{String,Any}("name"=>"slow","version"=>"1.0")))))
        elseif method == "tools/call"
            if (hit[] += 1) == 1
                sleep(3.0)   # first call: silent past the client's short bound
            end
            return HTTP.Response(200, ["Content-Type" => "application/json"],
                JSON.json(Dict{String,Any}("jsonrpc"=>"2.0","id"=>id,"result"=>Dict{String,Any}(
                    "content"=>[Dict{String,Any}("type"=>"text","text"=>"ok")]))))
        end
        HTTP.Response(200, ["Content-Type" => "application/json"],
            JSON.json(Dict{String,Any}("jsonrpc"=>"2.0","id"=>id,"result"=>Dict{String,Any}())))
    end
    url = "http://127.0.0.1:$port"
    try
        session = mcp_connect(url; config = RequestConfig(current_config(); mcp_request_timeout = 0.5))
        try
            err = nothing
            try
                call_tool(session, "slowtool", Dict{String,Any}())
            catch e
                err = e
            end
            @test err isa MCPTimeoutError
            @test err.phase === :request
            @test err.limit == 0.5
            @test session.status == :ready          # HTTP timeout is NOT session-fatal
            # Session survives: the retried call reaches the now-responsive server.
            @test call_tool(session, "slowtool", Dict{String,Any}()).content == "ok"
        finally
            mcp_disconnect!(session)
        end
    finally
        close(httpserver)
    end
end
