# Tests for mcp_server.jl — MCP Server types, registration, JSON-RPC routing, transport

@testset "MCPServer construction" begin
    server = MCPServer("test-server", "1.0.0")
    @test server.name == "test-server"
    @test server.version == "1.0.0"
    @test isnothing(server.description)
    @test isempty(server.tools)
    @test isempty(server.resources)
    @test isempty(server.prompts)
    @test !server._initialized

    server2 = MCPServer("s2", "2.0.0"; description="A test server")
    @test server2.description == "A test server"
end

@testset "Tool registration" begin
    server = MCPServer("test", "1.0.0")

    @testset "Explicit schema" begin
        schema = Dict{String,Any}("type" => "object", "properties" => Dict{String,Any}(
            "a" => Dict{String,Any}("type" => "number"),
            "b" => Dict{String,Any}("type" => "number")
        ), "required" => ["a", "b"])
        register_tool!(server, "add", "Add two numbers", schema, args -> args["a"] + args["b"])
        @test haskey(server.tools, "add")
        @test server.tools["add"].name == "add"
        @test server.tools["add"].description == "Add two numbers"
        # Test handler works
        @test server.tools["add"].handler(Dict{String,Any}("a" => 3, "b" => 5)) == 8
    end

    @testset "Inferred schema" begin
        register_tool!(server, "greet", "Greet someone", (args) -> "Hello $(args["name"])!")
        @test haskey(server.tools, "greet")
    end

    @testset "CallableTool{Tool} bridge" begin
        gpt_tool = Tool(func=FunctionSignature(
            name="multiply", description="Multiply",
            parameters=Dict{String,Any}("type" => "object", "properties" => Dict{String,Any}(
                "x" => Dict{String,Any}("type" => "number")))))
        ct = CallableTool(gpt_tool, (name, args) -> string(args["x"] * 2))
        register_tool!(server, ct)
        @test haskey(server.tools, "multiply")
        @test server.tools["multiply"].handler(Dict{String,Any}("x" => 5)) == "10"
    end

    @testset "CallableTool{FunctionTool} bridge" begin
        ft = FunctionTool(name="divide", description="Divide",
            parameters=Dict{String,Any}("type" => "object"))
        ct = CallableTool(ft, (name, args) -> string(args["a"] / args["b"]))
        register_tool!(server, ct)
        @test haskey(server.tools, "divide")
    end
end

@testset "Resource registration" begin
    server = MCPServer("test", "1.0.0")

    @testset "Static resource" begin
        register_resource!(server, "config://app", "App Config", () -> "key=value";
            description="Application config")
        @test haskey(server.resources, "config://app")
        @test server.resources["config://app"].handler() == "key=value"
    end

    @testset "Resource template" begin
        register_resource_template!(server, "file://{path}", "Files",
            params -> "content of $(params["path"])";
            mime_type="text/plain")
        @test length(server.resource_templates) == 1
        @test server.resource_templates[1].uri_template == "file://{path}"
    end
end

@testset "Prompt registration" begin
    server = MCPServer("test", "1.0.0")
    register_prompt!(server, "review", args -> [Dict{String,Any}(
        "role" => "user",
        "content" => Dict{String,Any}("type" => "text", "text" => "Review: $(args["code"])")
    )]; description="Code review", arguments=[Dict{String,Any}("name" => "code", "required" => true)])
    @test haskey(server.prompts, "review")
    msgs = server.prompts["review"].handler(Dict{String,Any}("code" => "x = 1"))
    @test length(msgs) == 1
    @test msgs[1]["role"] == "user"
end

@testset "URI template compilation" begin
    pattern, names = UniLM._compile_uri_template("file://{path}")
    @test "path" in names
    m = match(pattern, "file://foo.txt")
    @test !isnothing(m)
    @test m["path"] == "foo.txt"

    pattern2, names2 = UniLM._compile_uri_template("db://{schema}/{table}")
    @test Set(names2) == Set(["schema", "table"])
    m2 = match(pattern2, "db://public/users")
    @test m2["schema"] == "public"
    @test m2["table"] == "users"
end

@testset "Content formatting" begin
    @test UniLM._format_tool_result("hello") == [Dict{String,Any}("type" => "text", "text" => "hello")]
    @test UniLM._format_tool_result(42) == [Dict{String,Any}("type" => "text", "text" => "42")]

    # Dict with type key passes through
    d = Dict{String,Any}("type" => "image", "data" => "base64data")
    @test UniLM._format_tool_result(d) == [d]

    # Dict without type key becomes JSON text
    d2 = Dict{String,Any}("key" => "value")
    result = UniLM._format_tool_result(d2)
    @test result[1]["type"] == "text"

    # Vector passes through as-is
    v = [Dict{String,Any}("type" => "text", "text" => "a")]
    @test UniLM._format_tool_result(v) === v
end

@testset "JSON-RPC dispatch" begin
    server = MCPServer("test-server", "1.0.0")
    register_tool!(server, "echo", "Echo input", Dict{String,Any}(
        "type" => "object", "properties" => Dict{String,Any}("msg" => Dict{String,Any}("type" => "string"))
    ), args -> args["msg"])

    @testset "Initialize" begin
        req = Dict{String,Any}(
            "jsonrpc" => "2.0", "id" => 1, "method" => "initialize",
            "params" => Dict{String,Any}(
                "protocolVersion" => "2025-11-25",
                "capabilities" => Dict{String,Any}(),
                "clientInfo" => Dict{String,Any}("name" => "test", "version" => "1.0.0")
            ))
        resp = UniLM._dispatch_mcp(server, req)
        @test resp["id"] == 1
        @test haskey(resp, "result")
        @test resp["result"]["protocolVersion"] == "2025-11-25"
        @test haskey(resp["result"]["capabilities"], "tools")
        @test resp["result"]["serverInfo"]["name"] == "test-server"
        @test server._initialized
    end

    @testset "Tools list" begin
        req = Dict{String,Any}("jsonrpc" => "2.0", "id" => 2, "method" => "tools/list",
            "params" => Dict{String,Any}())
        resp = UniLM._dispatch_mcp(server, req)
        @test resp["id"] == 2
        tools = resp["result"]["tools"]
        @test length(tools) == 1
        @test tools[1]["name"] == "echo"
    end

    @testset "Tools call — success" begin
        req = Dict{String,Any}("jsonrpc" => "2.0", "id" => 3, "method" => "tools/call",
            "params" => Dict{String,Any}("name" => "echo", "arguments" => Dict{String,Any}("msg" => "hello")))
        resp = UniLM._dispatch_mcp(server, req)
        @test resp["id"] == 3
        @test resp["result"]["isError"] == false
        @test resp["result"]["content"][1]["text"] == "hello"
    end

    @testset "Tools call — unknown tool" begin
        req = Dict{String,Any}("jsonrpc" => "2.0", "id" => 4, "method" => "tools/call",
            "params" => Dict{String,Any}("name" => "nonexistent", "arguments" => Dict{String,Any}()))
        resp = UniLM._dispatch_mcp(server, req)
        @test haskey(resp, "error")
        @test resp["error"]["code"] == -32602
    end

    @testset "Tools call — handler error" begin
        register_tool!(server, "fail", "Always fails", Dict{String,Any}("type" => "object"),
            args -> error("intentional"))
        req = Dict{String,Any}("jsonrpc" => "2.0", "id" => 5, "method" => "tools/call",
            "params" => Dict{String,Any}("name" => "fail", "arguments" => Dict{String,Any}()))
        resp = UniLM._dispatch_mcp(server, req)
        @test resp["result"]["isError"] == true
        @test contains(resp["result"]["content"][1]["text"], "intentional")
    end

    @testset "Ping" begin
        req = Dict{String,Any}("jsonrpc" => "2.0", "id" => 6, "method" => "ping", "params" => Dict{String,Any}())
        resp = UniLM._dispatch_mcp(server, req)
        @test resp["id"] == 6
        @test haskey(resp, "result")
    end

    @testset "Unknown method" begin
        req = Dict{String,Any}("jsonrpc" => "2.0", "id" => 7, "method" => "unknown/method",
            "params" => Dict{String,Any}())
        resp = UniLM._dispatch_mcp(server, req)
        @test haskey(resp, "error")
        @test resp["error"]["code"] == -32601
    end

    @testset "Notification (no id) returns nothing" begin
        req = Dict{String,Any}("jsonrpc" => "2.0", "method" => "notifications/initialized")
        resp = UniLM._dispatch_mcp(server, req)
        @test isnothing(resp)
    end
end

@testset "Stdio transport round-trip" begin
    server = MCPServer("stdio-test", "1.0.0")
    register_tool!(server, "upper", "Uppercase",
        Dict{String,Any}("type" => "object", "properties" => Dict{String,Any}("s" => Dict{String,Any}("type" => "string"))),
        args -> uppercase(args["s"]))

    # Simulate stdio with IOBuffers
    input = IOBuffer()
    output = IOBuffer()

    # Write init request + tools/list + tools/call + EOF
    init_req = JSON.json(Dict("jsonrpc" => "2.0", "id" => 1, "method" => "initialize",
        "params" => Dict("protocolVersion" => "2025-11-25", "capabilities" => Dict(),
            "clientInfo" => Dict("name" => "test", "version" => "1.0.0"))))
    initialized_notif = JSON.json(Dict("jsonrpc" => "2.0", "method" => "notifications/initialized"))
    tools_req = JSON.json(Dict("jsonrpc" => "2.0", "id" => 2, "method" => "tools/list", "params" => Dict()))
    call_req = JSON.json(Dict("jsonrpc" => "2.0", "id" => 3, "method" => "tools/call",
        "params" => Dict("name" => "upper", "arguments" => Dict("s" => "hello"))))

    write(input, init_req, "\n", initialized_notif, "\n", tools_req, "\n", call_req, "\n")
    seekstart(input)

    UniLM._serve_stdio(server; input, output)

    seekstart(output)
    lines = filter(!isempty, split(String(take!(output)), "\n"))

    # Should have 3 response lines (init, tools/list, tools/call) — notification produces none
    @test length(lines) == 3

    # Parse init response
    init_resp = JSON.parse(lines[1])
    @test init_resp["result"]["serverInfo"]["name"] == "stdio-test"

    # Parse tools/list response
    tools_resp = JSON.parse(lines[2])
    @test length(tools_resp["result"]["tools"]) == 1

    # Parse tools/call response
    call_resp = JSON.parse(lines[3])
    @test call_resp["result"]["content"][1]["text"] == "HELLO"
    @test call_resp["result"]["isError"] == false
end

@testset "Resource operations via dispatch" begin
    server = MCPServer("res-test", "1.0.0")
    register_resource!(server, "config://app", "Config", () -> "debug=true")
    register_resource_template!(server, "file://{name}", "Files", p -> "content:$(p["name"])")

    # resources/list
    req = Dict{String,Any}("jsonrpc" => "2.0", "id" => 1, "method" => "resources/list",
        "params" => Dict{String,Any}())
    resp = UniLM._dispatch_mcp(server, req)
    @test length(resp["result"]["resources"]) == 1
    @test resp["result"]["resources"][1]["uri"] == "config://app"

    # resources/read — static
    req2 = Dict{String,Any}("jsonrpc" => "2.0", "id" => 2, "method" => "resources/read",
        "params" => Dict{String,Any}("uri" => "config://app"))
    resp2 = UniLM._dispatch_mcp(server, req2)
    @test resp2["result"]["contents"][1]["text"] == "debug=true"

    # resources/read — template
    req3 = Dict{String,Any}("jsonrpc" => "2.0", "id" => 3, "method" => "resources/read",
        "params" => Dict{String,Any}("uri" => "file://readme.md"))
    resp3 = UniLM._dispatch_mcp(server, req3)
    @test resp3["result"]["contents"][1]["text"] == "content:readme.md"

    # resources/read — not found
    req4 = Dict{String,Any}("jsonrpc" => "2.0", "id" => 4, "method" => "resources/read",
        "params" => Dict{String,Any}("uri" => "unknown://x"))
    resp4 = UniLM._dispatch_mcp(server, req4)
    @test haskey(resp4, "error")
    @test resp4["error"]["code"] == -32002
end

@testset "Prompt operations via dispatch" begin
    server = MCPServer("prompt-test", "1.0.0")
    register_prompt!(server, "greet", args -> [Dict{String,Any}(
        "role" => "user",
        "content" => Dict{String,Any}("type" => "text", "text" => "Hi $(args["name"])!")
    )]; arguments=[Dict{String,Any}("name" => "name", "required" => true)])

    # prompts/list
    req = Dict{String,Any}("jsonrpc" => "2.0", "id" => 1, "method" => "prompts/list",
        "params" => Dict{String,Any}())
    resp = UniLM._dispatch_mcp(server, req)
    @test length(resp["result"]["prompts"]) == 1
    @test resp["result"]["prompts"][1]["name"] == "greet"

    # prompts/get
    req2 = Dict{String,Any}("jsonrpc" => "2.0", "id" => 2, "method" => "prompts/get",
        "params" => Dict{String,Any}("name" => "greet", "arguments" => Dict{String,Any}("name" => "World")))
    resp2 = UniLM._dispatch_mcp(server, req2)
    @test resp2["result"]["messages"][1]["content"]["text"] == "Hi World!"
end

@testset "to_tool(MCPServerTool)" begin
    st = MCPServerTool("calc", "Calculator",
        Dict{String,Any}("type" => "object"), x -> x)
    ft = to_tool(st)
    @test ft isa FunctionTool
    @test ft.name == "calc"
    @test ft.description == "Calculator"
end

@testset "@mcp_tool macro" begin
    server = MCPServer("macro-test", "1.0.0")
    @mcp_tool server function add(a::Float64, b::Float64)::String
        string(a + b)
    end
    @test haskey(server.tools, "add")
    @test server.tools["add"].handler(Dict{String,Any}("a" => 3.0, "b" => 4.0)) == "7.0"
    @test server.tools["add"].input_schema["properties"]["a"] == Dict{String,Any}("type" => "number")
    @test server.tools["add"].input_schema["properties"]["b"] == Dict{String,Any}("type" => "number")
end

@testset "@mcp_tool macro — zero arguments" begin
    server = MCPServer("zero-arg", "1.0.0")
    @mcp_tool server function hello()::String
        "world"
    end
    @test haskey(server.tools, "hello")
    @test server.tools["hello"].handler(Dict{String,Any}()) == "world"
    @test isempty(server.tools["hello"].input_schema["properties"])
end

# ─── Error & edge paths for resources/prompts/stdio (coverage: src lines 272, 362, 375, 402, 422, 451–454) ───

@testset "resources/read — binary blob (base64, no text key)" begin
    # src/mcp_server.jl:272 — Vector{UInt8} handler result → base64 "blob" not "text".
    server = MCPServer("blob-test", "1.0.0")
    payload = UInt8[0x00, 0x01, 0xff, 0xfe, 0x42, 0x00, 0x80]  # includes non-UTF8/null bytes
    register_resource!(server, "bytes://raw", "Raw Bytes", () -> payload;
        mime_type="application/octet-stream")

    req = Dict{String,Any}("jsonrpc" => "2.0", "id" => 11, "method" => "resources/read",
        "params" => Dict{String,Any}("uri" => "bytes://raw"))
    resp = UniLM._dispatch_mcp(server, req)

    @test resp["id"] == 11
    content = resp["result"]["contents"][1]
    @test content["uri"] == "bytes://raw"
    @test content["mimeType"] == "application/octet-stream"
    # The blob branch must be taken: a "blob" key present, NO "text" key.
    @test haskey(content, "blob")
    @test !haskey(content, "text")
    # The base64 must decode to the EXACT original bytes.
    @test UniLM.Base64.base64decode(content["blob"]) == payload
end

# Resource and prompt handler failures follow `serve`'s robustness contract: the peer gets
# a generic -32603 (exception text can carry paths and argument values) and the detail is
# logged locally. Only tool-handler errors are relayed, as `isError` tool results.

@testset "resources/read — static handler throws → generic -32603, detail logged" begin
    server = MCPServer("res-err", "1.0.0")
    register_resource!(server, "boom://static", "Boom", () -> error("static handler exploded"))

    req = Dict{String,Any}("jsonrpc" => "2.0", "id" => 12, "method" => "resources/read",
        "params" => Dict{String,Any}("uri" => "boom://static"))
    resp = @test_logs (:error,) match_mode=:any UniLM._dispatch_guarded(server, req)

    @test resp["id"] == 12
    @test !haskey(resp, "result")
    @test resp["error"]["code"] == -32603
    @test resp["error"]["message"] == "Internal error"
    @test !contains(JSON.json(resp), "exploded")
end

@testset "resources/read — template handler throws → generic -32603, detail logged" begin
    server = MCPServer("tmpl-err", "1.0.0")
    register_resource_template!(server, "boom://{id}", "BoomTmpl",
        p -> error("template handler exploded for $(p["id"])"))

    req = Dict{String,Any}("jsonrpc" => "2.0", "id" => 13, "method" => "resources/read",
        "params" => Dict{String,Any}("uri" => "boom://42"))
    resp = @test_logs (:error,) match_mode=:any UniLM._dispatch_guarded(server, req)

    @test resp["id"] == 13
    @test !haskey(resp, "result")
    @test resp["error"]["code"] == -32603
    @test resp["error"]["message"] == "Internal error"
    @test !contains(JSON.json(resp), "exploded")
end

@testset "prompts/get — handler throws → generic -32603, detail logged" begin
    server = MCPServer("prompt-err", "1.0.0")
    register_prompt!(server, "explode", args -> error("prompt handler exploded");
        arguments=[Dict{String,Any}("name" => "x", "required" => false)])

    req = Dict{String,Any}("jsonrpc" => "2.0", "id" => 14, "method" => "prompts/get",
        "params" => Dict{String,Any}("name" => "explode", "arguments" => Dict{String,Any}()))
    resp = @test_logs (:error,) match_mode=:any UniLM._dispatch_guarded(server, req)

    @test resp["id"] == 14
    @test !haskey(resp, "result")
    @test resp["error"]["code"] == -32603
    @test resp["error"]["message"] == "Internal error"
    @test !contains(JSON.json(resp), "exploded")
end

@testset "resources/templates/list dispatch" begin
    # src/mcp_server.jl:422 — method "resources/templates/list" routes to templates-list handler.
    server = MCPServer("tmpl-list", "1.0.0")
    register_resource_template!(server, "file://{path}", "Files",
        p -> "content of $(p["path"])"; description="File reader")

    req = Dict{String,Any}("jsonrpc" => "2.0", "id" => 15, "method" => "resources/templates/list",
        "params" => Dict{String,Any}())
    resp = UniLM._dispatch_mcp(server, req)

    @test resp["id"] == 15
    @test haskey(resp["result"], "resourceTemplates")
    templates = resp["result"]["resourceTemplates"]
    @test length(templates) == 1
    @test templates[1]["uriTemplate"] == "file://{path}"
    @test templates[1]["name"] == "Files"
    @test templates[1]["description"] == "File reader"
end

@testset "stdio transport — malformed JSON → parse error (-32700)" begin
    # src/mcp_server.jl:451–454 — malformed line in _serve_stdio → JSON-RPC -32700 with null id.
    server = MCPServer("stdio-parse-err", "1.0.0")
    input = IOBuffer()
    output = IOBuffer()

    # First line is invalid JSON; second is a valid ping so we can confirm the loop continues.
    write(input, "{ this is not valid json", "\n",
        JSON.json(Dict("jsonrpc" => "2.0", "id" => 99, "method" => "ping", "params" => Dict())), "\n")
    seekstart(input)

    UniLM._serve_stdio(server; input, output)

    seekstart(output)
    lines = filter(!isempty, split(String(take!(output)), "\n"))
    @test length(lines) == 2  # parse-error response + ping response

    err_resp = JSON.parse(lines[1])
    @test err_resp["jsonrpc"] == "2.0"
    @test err_resp["id"] === nothing  # parse error responses correlate to id=null per JSON-RPC
    @test err_resp["error"]["code"] == -32700
    @test contains(err_resp["error"]["message"], "Parse error")

    # Loop survived the bad line and serviced the next request.
    ping_resp = JSON.parse(lines[2])
    @test ping_resp["id"] == 99
    @test haskey(ping_resp, "result")
end

# ─── HTTP transport (_serve_http, src 469–488) + serve dispatcher (506–513) ───
# These drive the REAL `serve(...; transport=:http,...)` over the wire (raw HTTP.post
# with status_exception=false so non-2xx is inspectable, not thrown).

using Sockets

"Bind→read→close an ephemeral localhost port so it is free for the server to claim."
_mcp_free_port() = let s = Sockets.listen(Sockets.localhost, 0)
    p = Int(Sockets.getsockname(s)[2])
    close(s)
    p
end

"A server with one registered tool/resource/prompt and deterministic, assertable outputs."
function _build_http_server()
    server = MCPServer("http-xport", "3.2.1"; description="http transport probe")
    register_tool!(server, "shout", "Uppercase the input",
        Dict{String,Any}("type" => "object",
            "properties" => Dict{String,Any}("s" => Dict{String,Any}("type" => "string")),
            "required" => ["s"]),
        args -> uppercase(string(args["s"])))
    register_resource!(server, "probe://greeting", "greeting", () -> "hi-from-http")
    register_prompt!(server, "salute",
        args -> [Dict{String,Any}("role" => "user",
            "content" => Dict{String,Any}("type" => "text", "text" => "Hello, $(get(args, "who", "x"))!"))];
        arguments=[Dict{String,Any}("name" => "who", "required" => true)])
    server
end

@testset "HTTP transport — valid JSON-RPC POST → 200 + result body (src 471–478,482)" begin
    server = _build_http_server()
    port = _mcp_free_port()
    httpserver = serve(server; transport=:http, host="127.0.0.1", port=port, block=false)
    try
        # initialize: drives _dispatch_mcp → _handle_initialize through the HTTP handler.
        body = JSON.json(Dict("jsonrpc" => "2.0", "id" => 1, "method" => "initialize",
            "params" => Dict("protocolVersion" => "2025-11-25",
                "capabilities" => Dict(),
                "clientInfo" => Dict("name" => "t", "version" => "1"))))
        resp = HTTP.post("http://127.0.0.1:$port", ["Content-Type" => "application/json"], body;
            status_exception=false)
        @test resp.status == 200
        # The handler sets a JSON content type (line 482).
        @test any(lowercase(k) == "content-type" && occursin("application/json", lowercase(v))
                  for (k, v) in resp.headers)
        parsed = JSON.parse(String(resp.body))
        @test parsed["jsonrpc"] == "2.0"
        @test parsed["id"] == 1
        @test parsed["result"]["protocolVersion"] == UniLM._MCP_PROTOCOL_VERSION
        @test parsed["result"]["serverInfo"]["name"] == "http-xport"
        @test parsed["result"]["serverInfo"]["version"] == "3.2.1"
        # Tool+resource+prompt registered → all three capability buckets advertised.
        @test haskey(parsed["result"]["capabilities"], "tools")
        @test haskey(parsed["result"]["capabilities"], "resources")
        @test haskey(parsed["result"]["capabilities"], "prompts")

        # tools/call: prove the handler actually ran the tool and returned its content.
        call_body = JSON.json(Dict("jsonrpc" => "2.0", "id" => 2, "method" => "tools/call",
            "params" => Dict("name" => "shout", "arguments" => Dict("s" => "echo"))))
        call_resp = HTTP.post("http://127.0.0.1:$port", ["Content-Type" => "application/json"],
            call_body; status_exception=false)
        @test call_resp.status == 200
        cparsed = JSON.parse(String(call_resp.body))
        @test cparsed["id"] == 2
        @test cparsed["result"]["isError"] == false
        @test cparsed["result"]["content"][1]["text"] == "ECHO"
    finally
        close(httpserver)
    end
end

@testset "HTTP transport — malformed JSON → 400 + parse error -32700 (src 475–476)" begin
    server = _build_http_server()
    port = _mcp_free_port()
    httpserver = serve(server; transport=:http, port=port, block=false)  # default host
    try
        resp = HTTP.post("http://127.0.0.1:$port", ["Content-Type" => "application/json"],
            "{ not valid json at all"; status_exception=false)
        @test resp.status == 400
        parsed = JSON.parse(String(resp.body))
        @test parsed["jsonrpc"] == "2.0"
        @test parsed["id"] === nothing       # parse-error id is null per JSON-RPC
        @test parsed["error"]["code"] == -32700
        @test parsed["error"]["message"] == "Parse error"
    finally
        close(httpserver)
    end
end

@testset "HTTP transport — notification (no id) → 202 empty (src 479–480)" begin
    server = _build_http_server()
    port = _mcp_free_port()
    httpserver = serve(server; transport=:http, port=port, block=false)
    try
        # Well-formed JSON-RPC with NO "id" → _dispatch_mcp returns nothing → 202.
        notif = JSON.json(Dict("jsonrpc" => "2.0", "method" => "notifications/initialized"))
        resp = HTTP.post("http://127.0.0.1:$port", ["Content-Type" => "application/json"],
            notif; status_exception=false)
        @test resp.status == 202
        @test isempty(String(resp.body))
    finally
        close(httpserver)
    end
end

@testset "HTTP transport — DELETE → 200 (src 483–484); GET → 405 (src 485–486)" begin
    server = _build_http_server()
    port = _mcp_free_port()
    httpserver = serve(server; transport=:http, port=port, block=false)
    try
        del = HTTP.request("DELETE", "http://127.0.0.1:$port"; status_exception=false)
        @test del.status == 200
        @test isempty(String(del.body))

        # Any non-POST/non-DELETE method hits the 405 else-branch.
        getr = HTTP.request("GET", "http://127.0.0.1:$port"; status_exception=false)
        @test getr.status == 405
        @test String(getr.body) == "Method Not Allowed"
    finally
        close(httpserver)
    end
end

@testset "serve dispatcher — :stdio routes to _serve_stdio (src 506–508)" begin
    # serve(...; transport=:stdio, input=, output=) must forward kwargs to _serve_stdio
    # and produce a real JSON-RPC response on the output buffer.
    server = MCPServer("dispatch-stdio", "1.0.0")
    register_tool!(server, "id", "identity",
        Dict{String,Any}("type" => "object",
            "properties" => Dict{String,Any}("v" => Dict{String,Any}("type" => "string"))),
        args -> args["v"])
    input = IOBuffer()
    output = IOBuffer()
    write(input, JSON.json(Dict("jsonrpc" => "2.0", "id" => 7, "method" => "tools/call",
        "params" => Dict("name" => "id", "arguments" => Dict("v" => "ok")))), "\n")
    seekstart(input)

    serve(server; transport=:stdio, input=input, output=output)

    seekstart(output)
    lines = filter(!isempty, split(String(take!(output)), "\n"))
    @test length(lines) == 1
    resp = JSON.parse(lines[1])
    @test resp["id"] == 7
    @test resp["result"]["content"][1]["text"] == "ok"
    @test resp["result"]["isError"] == false
end

@testset "serve dispatcher — unknown transport → ArgumentError (src 511–512)" begin
    server = MCPServer("dispatch-bogus", "1.0.0")
    err = try
        serve(server; transport=:bogus)
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("Unknown transport", err.msg)
    @test occursin("bogus", err.msg)
end

# ─── @mcp_tool _mcp_convert coverage (src 564–568) ────────────────────────────
# Each macro-unpacked arg type forces a distinct _mcp_convert method.

@testset "@mcp_tool — _mcp_convert per-type dispatch (src 564–568)" begin
    server = MCPServer("convert-test", "1.0.0")
    # String (564), Int<:Integer (565), Float64<:AbstractFloat (566), Bool (567),
    # Vector{Int} → fallback pass-through (568).
    @mcp_tool server function mixt(s::String, n::Int, f::Float64, b::Bool, xs::Vector{Int})::String
        # Assert the converted Julia types are exactly right (not just stringly-equal).
        string(s, "|", n, "|", n isa Int, "|", f, "|", f isa Float64, "|",
            b, "|", b isa Bool, "|", sum(xs), "|", xs isa Vector{Int})
    end
    @test haskey(server.tools, "mixt")
    sch = server.tools["mixt"].input_schema
    @test sch["properties"]["s"] == Dict{String,Any}("type" => "string")
    @test sch["properties"]["n"] == Dict{String,Any}("type" => "integer")
    @test sch["properties"]["f"] == Dict{String,Any}("type" => "number")
    @test sch["properties"]["b"] == Dict{String,Any}("type" => "boolean")
    @test sch["properties"]["xs"] == Dict{String,Any}("type" => "array", "items" => Dict{String,Any}("type" => "integer"))
    @test Set(sch["required"]) == Set(["s", "n", "f", "b", "xs"])

    # Feed values as they arrive from JSON: numbers may be Float (JSON has one number type).
    # An integral 5.0 binds an Int parameter as 5; an integer binds a Float64 as 2.0.
    out = server.tools["mixt"].handler(Dict{String,Any}(
        "s" => "42",          # String parameter: a JSON string
        "n" => 5.0,           # Int parameter: an integral number → 5
        "f" => 2,             # Float64 parameter: any number → 2.0
        "b" => true,          # Bool parameter: a JSON boolean
        "xs" => [3, 4]))      # other declared types pass through unconverted
    @test out == "42|5|true|2.0|true|true|true|7|true"

    # Integer branch with an already-integer value exercises the non-float arm of line 565.
    @mcp_tool server function inc(n::Int)::Int
        n + 1
    end
    @test server.tools["inc"].handler(Dict{String,Any}("n" => 41)) == 42
end

# ─── @mcp_resource macro — template (src 593–594) and static (src 598–599) ────

@testset "@mcp_resource — template branch registers a template (param-free body)" begin
    server = MCPServer("res-macro-tmpl", "1.0.0")
    # Template branch with a CONSTANT (param-free) body: exercises registration +
    # dispatch without depending on a bound param. The declared arg name matches the
    # URI {param}; the fixed macro binds it (then the body ignores it). (Param binding
    # itself is covered by the "@mcp_resource — template binds matched path param" sets.)
    @mcp_resource server "doc://{name}" function(name::String)
        "doc-body-constant"
    end
    # Template branch (URI has {...}) → goes to resource_templates, NOT static resources.
    @test isempty(server.resources)
    @test length(server.resource_templates) == 1
    @test server.resource_templates[1].uri_template == "doc://{name}"
    # The registered handler runs through dispatch and matches the templated URI.
    req = Dict{String,Any}("jsonrpc" => "2.0", "id" => 1, "method" => "resources/read",
        "params" => Dict{String,Any}("uri" => "doc://readme"))
    resp = UniLM._dispatch_mcp(server, req)
    @test resp["result"]["contents"][1]["uri"] == "doc://readme"
    @test resp["result"]["contents"][1]["text"] == "doc-body-constant"
end

@testset "@mcp_resource — static branch registers a static resource (src 598–599)" begin
    server = MCPServer("res-macro-static", "1.0.0")
    @mcp_resource server "config://app" function()
        "k=v"
    end
    # No {...} → static branch → resources dict, NOT templates.
    @test isempty(server.resource_templates)
    @test haskey(server.resources, "config://app")
    @test server.resources["config://app"].handler() == "k=v"
    req = Dict{String,Any}("jsonrpc" => "2.0", "id" => 1, "method" => "resources/read",
        "params" => Dict{String,Any}("uri" => "config://app"))
    resp = UniLM._dispatch_mcp(server, req)
    @test resp["result"]["contents"][1]["text"] == "k=v"
end

# ─── @mcp_prompt macro (src 632–638) ──────────────────────────────────────────

@testset "@mcp_prompt — NAMED form registers prompt, unpacks args, builds schema" begin
    # NAMED `function f(args...)` form regression guard: the shared extractor must
    # drop the fn name and keep all declared args. (The anonymous `function(x) … end`
    # form is covered by the "@mcp_prompt — anonymous …" regression sets above.)
    server = MCPServer("prompt-macro", "1.0.0")
    # One typed (required) arg + one untyped (optional) arg exercises arg_defs both ways.
    @mcp_prompt server "review" function _ignored(code::String, lang)
        [Dict{String,Any}("role" => "user",
            "content" => Dict{String,Any}("type" => "text",
                "text" => "Review $(lang) code: $(code)"))]
    end
    @test haskey(server.prompts, "review")
    p = server.prompts["review"]
    # arg_defs: code typed → required=true; lang untyped → required=false (src 622–627).
    code_arg = only(filter(a -> a["name"] == "code", p.arguments))
    lang_arg = only(filter(a -> a["name"] == "lang", p.arguments))
    @test code_arg["required"] == true
    @test lang_arg["required"] == false
    @test length(p.arguments) == 2

    # Drive through dispatch so the generated unpacking handler (src 634–636) runs and
    # binds BOTH args from the request into the escaped body.
    req = Dict{String,Any}("jsonrpc" => "2.0", "id" => 1, "method" => "prompts/get",
        "params" => Dict{String,Any}("name" => "review",
            "arguments" => Dict{String,Any}("code" => "x=1", "lang" => "julia")))
    resp = UniLM._dispatch_mcp(server, req)
    @test resp["result"]["messages"][1]["role"] == "user"
    @test resp["result"]["messages"][1]["content"]["text"] == "Review julia code: x=1"
end

# ─── REGRESSION: macro arg-binding bugs (fix-mcp-macro-arg-binding) ────────────
# These lock the DOCUMENTED behavior that was broken on the pre-fix source:
#  (1) @mcp_resource template form did not bind matched path params into the
#      declared arg (esc'd body referenced an UNDEFINED var → -32603 UndefVarError).
#  (2) @mcp_prompt / @mcp_tool used args[2:end] (assumes a NAMED :call signature),
#      silently dropping the FIRST declared arg of the ANONYMOUS `function(x)…end`
#      form → arg never in the schema and never bound in the handler.
# Each assertion below FAILS on the unfixed source (proven: dispatch returns an
# `error` -32603 / empty `arguments`, never the asserted `result`/text).

@testset "@mcp_resource — template binds matched path param into declared arg (docstring form)" begin
    server = MCPServer("res-tmpl-bind", "1.0.0")
    # The docstring's own shape: a templated URI + an arg whose NAME equals the
    # {param} name, referenced in the body. Must bind "word" from the matched URI.
    @mcp_resource server "echo://{word}" function(word::String)
        "got:" * word
    end
    @test length(server.resource_templates) == 1
    @test server.resource_templates[1].uri_template == "echo://{word}"

    req = Dict{String,Any}("jsonrpc" => "2.0", "id" => 1, "method" => "resources/read",
        "params" => Dict{String,Any}("uri" => "echo://hello"))
    resp = UniLM._dispatch_mcp(server, req)
    # On the UNFIXED source this is an `error` (-32603 UndefVarError: `word`), so
    # demanding a `result` with the exact text is a falsifiable regression assertion.
    @test haskey(resp, "result")
    @test !haskey(resp, "error")
    @test resp["result"]["contents"][1]["uri"] == "echo://hello"
    @test resp["result"]["contents"][1]["text"] == "got:hello"
end

@testset "@mcp_resource — template binds MULTIPLE matched path params (order-independent)" begin
    server = MCPServer("res-tmpl-multi", "1.0.0")
    # Two params; body references BOTH and in the opposite order to the URI so the
    # test is sensitive to per-name (not positional) binding.
    @mcp_resource server "db://{schema}/{table}" function(schema::String, table::String)
        table * "@" * schema
    end
    req = Dict{String,Any}("jsonrpc" => "2.0", "id" => 1, "method" => "resources/read",
        "params" => Dict{String,Any}("uri" => "db://public/users"))
    resp = UniLM._dispatch_mcp(server, req)
    @test haskey(resp, "result")
    @test resp["result"]["contents"][1]["text"] == "users@public"
end

@testset "@mcp_prompt — anonymous `function(x)` registers arg AND binds it (docstring form)" begin
    server = MCPServer("prompt-anon", "1.0.0")
    # Exactly the docstring shape: a single typed anonymous arg.
    @mcp_prompt server "review" function(code::String)
        [Dict{String,Any}("role" => "user",
            "content" => Dict{String,Any}("type" => "text", "text" => "Review: " * code))]
    end
    @test haskey(server.prompts, "review")
    p = server.prompts["review"]
    # On the unfixed source `arguments` is EMPTY (the only arg was dropped by args[2:end]).
    @test length(p.arguments) == 1
    @test p.arguments[1]["name"] == "code"
    @test p.arguments[1]["required"] == true   # typed ⇒ required

    req = Dict{String,Any}("jsonrpc" => "2.0", "id" => 1, "method" => "prompts/get",
        "params" => Dict{String,Any}("name" => "review",
            "arguments" => Dict{String,Any}("code" => "x = 1")))
    resp = UniLM._dispatch_mcp(server, req)
    @test haskey(resp, "result")
    @test !haskey(resp, "error")
    @test resp["result"]["messages"][1]["role"] == "user"
    @test resp["result"]["messages"][1]["content"]["text"] == "Review: x = 1"
end

@testset "@mcp_prompt — anonymous MULTI-arg (typed+untyped mix): both registered & bound" begin
    server = MCPServer("prompt-anon-multi", "1.0.0")
    @mcp_prompt server "diff" function(a::String, b)
        [Dict{String,Any}("role" => "user",
            "content" => Dict{String,Any}("type" => "text", "text" => a * "→" * string(b)))]
    end
    p = server.prompts["diff"]
    @test length(p.arguments) == 2
    a_arg = only(filter(x -> x["name"] == "a", p.arguments))
    b_arg = only(filter(x -> x["name"] == "b", p.arguments))
    @test a_arg["required"] == true    # typed
    @test b_arg["required"] == false   # untyped

    req = Dict{String,Any}("jsonrpc" => "2.0", "id" => 1, "method" => "prompts/get",
        "params" => Dict{String,Any}("name" => "diff",
            "arguments" => Dict{String,Any}("a" => "lhs", "b" => "rhs")))
    resp = UniLM._dispatch_mcp(server, req)
    @test haskey(resp, "result")
    @test resp["result"]["messages"][1]["content"]["text"] == "lhs→rhs"
end

@testset "@mcp_prompt — NAMED form still works (regression guard for the shared extractor)" begin
    # The named `function f(args...)` form was correct before; it MUST stay correct
    # after the extractor refactor. First arg here is the NAME, not a param.
    server = MCPServer("prompt-named", "1.0.0")
    @mcp_prompt server "greet" function _named(name::String)
        [Dict{String,Any}("role" => "user",
            "content" => Dict{String,Any}("type" => "text", "text" => "Hi " * name))]
    end
    p = server.prompts["greet"]
    @test length(p.arguments) == 1
    @test p.arguments[1]["name"] == "name"
    @test p.arguments[1]["required"] == true
    req = Dict{String,Any}("jsonrpc" => "2.0", "id" => 1, "method" => "prompts/get",
        "params" => Dict{String,Any}("name" => "greet",
            "arguments" => Dict{String,Any}("name" => "Ada")))
    resp = UniLM._dispatch_mcp(server, req)
    @test resp["result"]["messages"][1]["content"]["text"] == "Hi Ada"
end

@testset "@mcp_tool — NAMED single typed arg: schema + handler (locks shared extractor)" begin
    # @mcp_tool registers under the function NAME, so its supported form is named.
    # A single-arg named signature is the case most likely to break under a naive
    # extractor; lock it with exact schema + handler-result assertions.
    server = MCPServer("tool-named-1arg", "1.0.0")
    @mcp_tool server function square(n::Int)::String
        string(n * n)
    end
    @test haskey(server.tools, "square")
    @test server.tools["square"].input_schema["properties"]["n"] == Dict{String,Any}("type" => "integer")
    @test server.tools["square"].input_schema["required"] == ["n"]
    @test server.tools["square"].handler(Dict{String,Any}("n" => 6)) == "36"
end

@testset "@mcp_tool — NAMED mixed typed/untyped args: required reflects typedness" begin
    server = MCPServer("tool-named-mix", "1.0.0")
    @mcp_tool server function pair(a::String, b)::String
        string(a, "/", b)
    end
    sch = server.tools["pair"].input_schema
    @test Set(keys(sch["properties"])) == Set(["a", "b"])
    # `a` typed ⇒ required; `b` untyped (:Any) ⇒ NOT required (matches pre-fix logic).
    @test sch["required"] == ["a"]
    @test server.tools["pair"].handler(Dict{String,Any}("a" => "x", "b" => "y")) == "x/y"
end

# ─── Non-object JSON-RPC frames (valid JSON, wrong shape) ─────────────────────

@testset "stdio transport — non-object frames → -32600, loop continues" begin
    server = MCPServer("stdio-invalid-req", "1.0.0")
    input = IOBuffer()
    output = IOBuffer()
    # Three valid-JSON non-object frames (array, string, number), then a valid
    # ping proving the loop survived all of them.
    write(input,
        """[{"jsonrpc":"2.0","id":1,"method":"ping"}]""", "\n",
        "\"just a string\"", "\n",
        "42", "\n",
        JSON.json(Dict("jsonrpc" => "2.0", "id" => 5, "method" => "ping", "params" => Dict())), "\n")
    seekstart(input)

    UniLM._serve_stdio(server; input, output)

    seekstart(output)
    lines = filter(!isempty, split(String(take!(output)), "\n"))
    @test length(lines) == 4  # three -32600 responses + one ping result

    for i in 1:3
        err = JSON.parse(lines[i])
        @test err["jsonrpc"] == "2.0"
        @test err["id"] === nothing          # unattributable frame → id null
        @test err["error"]["code"] == -32600
        @test contains(err["error"]["message"], "Invalid Request")
        @test !haskey(err, "result")
    end

    ping_resp = JSON.parse(lines[4])
    @test ping_resp["id"] == 5
    @test haskey(ping_resp, "result")
end

@testset "HTTP transport — non-object frame → 400 + -32600 (not 500)" begin
    server = _build_http_server()
    port = _mcp_free_port()
    httpserver = serve(server; transport=:http, port=port, block=false)
    try
        for bad in ("""[{"jsonrpc":"2.0","id":1,"method":"ping"}]""", "\"str\"", "42")
            resp = HTTP.post("http://127.0.0.1:$port", ["Content-Type" => "application/json"],
                bad; status_exception=false)
            @test resp.status == 400
            parsed = JSON.parse(String(resp.body))
            @test parsed["id"] === nothing
            @test parsed["error"]["code"] == -32600
            @test contains(parsed["error"]["message"], "Invalid Request")
        end
        # The server is still alive and answers a valid request afterwards.
        ok = HTTP.post("http://127.0.0.1:$port", ["Content-Type" => "application/json"],
            JSON.json(Dict("jsonrpc" => "2.0", "id" => 9, "method" => "ping",
                "params" => Dict())); status_exception=false)
        @test ok.status == 200
        @test JSON.parse(String(ok.body))["id"] == 9
    finally
        close(httpserver)
    end
end

# ─── Origin validation (DNS-rebinding defense) ────────────────────────────────

@testset "HTTP transport — Origin validation (localhost default + allowlist)" begin
    server = _build_http_server()
    port = _mcp_free_port()
    httpserver = serve(server; transport=:http, port=port,
        allowed_origins=["https://app.example.com"], block=false)
    ping_body = JSON.json(Dict("jsonrpc" => "2.0", "id" => 1, "method" => "ping",
        "params" => Dict()))
    post(origin) = HTTP.post("http://127.0.0.1:$port",
        ["Content-Type" => "application/json", "Origin" => origin], ping_body;
        status_exception=false)
    try
        # No Origin header (non-browser client): allowed.
        no_origin = HTTP.post("http://127.0.0.1:$port",
            ["Content-Type" => "application/json"], ping_body; status_exception=false)
        @test no_origin.status == 200

        # Localhost origins: always allowed — any port, either scheme, any case.
        for o in ("http://localhost:5173", "http://127.0.0.1",
                  "https://LOCALHOST:8443", "http://[::1]:3000")
            @test post(o).status == 200
        end

        # Allowlisted origin: allowed (exact match).
        @test post("https://app.example.com").status == 200

        # Everything else: 403 — including lookalike hosts and opaque origins.
        for o in ("https://evil.example", "http://localhost.evil.example",
                  "http://mylocalhost", "null",
                  "https://app.example.com.evil.example")
            @test post(o).status == 403
        end

        # The gate runs before method dispatch: DELETE with a bad Origin is 403.
        del = HTTP.request("DELETE", "http://127.0.0.1:$port",
            ["Origin" => "https://evil.example"]; status_exception=false)
        @test del.status == 403
    finally
        close(httpserver)
    end
end

@testset "HTTP transport — Origin default (no allowlist): localhost only" begin
    server2 = _build_http_server()
    port2 = _mcp_free_port()
    httpserver2 = serve(server2; transport=:http, port=port2, block=false)
    ping_body = JSON.json(Dict("jsonrpc" => "2.0", "id" => 1, "method" => "ping",
        "params" => Dict()))
    try
        okr = HTTP.post("http://127.0.0.1:$port2",
            ["Content-Type" => "application/json", "Origin" => "http://localhost:3000"],
            ping_body; status_exception=false)
        @test okr.status == 200
        badr = HTTP.post("http://127.0.0.1:$port2",
            ["Content-Type" => "application/json", "Origin" => "https://app.example.com"],
            ping_body; status_exception=false)
        @test badr.status == 403   # not allowlisted here
    finally
        close(httpserver2)
    end
end

# ─── serve(:http) blocking semantics ─────────────────────────────────────────

@testset "serve(:http) — block=false returns the running server handle" begin
    server = _build_http_server()
    port = _mcp_free_port()
    handle = serve(server; transport=:http, port=port, block=false)
    try
        @test !isnothing(handle)
        resp = HTTP.post("http://127.0.0.1:$port", ["Content-Type" => "application/json"],
            JSON.json(Dict("jsonrpc" => "2.0", "id" => 1, "method" => "ping",
                "params" => Dict())); status_exception=false)
        @test resp.status == 200
    finally
        close(handle)
    end
end

@testset "serve(:http) — blocks by default until closed" begin
    server = _build_http_server()
    port = _mcp_free_port()
    ping_body = JSON.json(Dict("jsonrpc" => "2.0", "id" => 1, "method" => "ping",
        "params" => Dict()))
    serve_task = @async serve(server; transport=:http, port=port)

    # Bounded readiness poll: the listener is up once a request round-trips.
    ready = timedwait(10.0) do
        try
            HTTP.post("http://127.0.0.1:$port", ["Content-Type" => "application/json"],
                ping_body; status_exception=false).status == 200
        catch
            false
        end
    end
    @test ready === :ok

    # The server answers requests, yet serve has NOT returned: it blocks.
    @test !istaskdone(serve_task)

    # Interrupt the blocked wait; serve must close the listener on the way out.
    schedule(serve_task, InterruptException(); error=true)
    @test timedwait(() -> istaskdone(serve_task), 10.0) === :ok
    @test istaskfailed(serve_task)   # the interrupt propagated out of serve

    # The cleanup ran: the port stops accepting (bounded check).
    gone = timedwait(10.0) do
        try
            HTTP.post("http://127.0.0.1:$port", ["Content-Type" => "application/json"],
                ping_body; status_exception=false)
            false
        catch
            true
        end
    end
    @test gone === :ok
end

# ─── Spec-legal `params` shapes: omitted / null / positional ──────────────────
# Every `_handle_*` takes `params::Dict{String,Any}`. JSON-RPC lets `params` be
# omitted or null (no arguments) and permits a positional array, so the raw member
# has to be normalized before dispatch — otherwise those frames reach the handlers
# as `nothing`/`Vector` and fail to dispatch at all.

"A server with one tool, for frames that must reach a real `_handle_*`."
function _build_params_server()
    server = MCPServer("params-shapes", "1.0.0")
    register_tool!(server, "shout", "Uppercase",
        Dict{String,Any}("type" => "object",
            "properties" => Dict{String,Any}("s" => Dict{String,Any}("type" => "string"))),
        args -> uppercase(string(get(args, "s", ""))))
    server
end

@testset "dispatch — omitted/null params are empty params; positional is -32602" begin
    server = _build_params_server()
    # `"params": null` and an omitted `params` both mean "no arguments".
    for req in (Dict{String,Any}("jsonrpc" => "2.0", "id" => 1, "method" => "tools/list",
                    "params" => nothing),
                Dict{String,Any}("jsonrpc" => "2.0", "id" => 1, "method" => "tools/list"))
        resp = UniLM._dispatch_mcp(server, req)
        @test !haskey(resp, "error")
        @test length(resp["result"]["tools"]) == 1
    end
    # A positional array cannot name this server's arguments → Invalid params, and
    # the request keeps its id (it is answerable, unlike an unparsable frame).
    for positional in (Any[1, 2], Any[], "str", 42, true)
        resp = UniLM._dispatch_mcp(server, Dict{String,Any}(
            "jsonrpc" => "2.0", "id" => 4, "method" => "tools/list", "params" => positional))
        @test resp["id"] == 4
        @test resp["error"]["code"] == -32602
        @test contains(resp["error"]["message"], "Invalid params")
        @test !haskey(resp, "result")
    end
    # A notification (no id) is still answered with nothing whatever params holds.
    @test isnothing(UniLM._dispatch_mcp(server, Dict{String,Any}(
        "jsonrpc" => "2.0", "method" => "notifications/initialized", "params" => nothing)))
    @test isnothing(UniLM._dispatch_mcp(server, Dict{String,Any}(
        "jsonrpc" => "2.0", "method" => "notifications/initialized", "params" => Any[1])))
    # Non-`Dict{String,Any}` mappings are accepted by value, not rejected by type.
    resp = UniLM._dispatch_mcp(server, Dict{String,Any}(
        "jsonrpc" => "2.0", "id" => 5, "method" => "tools/call",
        "params" => Dict{String,String}("name" => "shout")))
    @test resp["result"]["isError"] == false
end

@testset "stdio — a null-params frame is answered and the loop keeps serving" begin
    server = _build_params_server()
    input = IOBuffer()
    output = IOBuffer()
    # Frame 2 carries the spec-legal `"params": null`. Reaching a handler typed
    # `::Dict{String,Any}` as `nothing` raised a MethodError from inside the read
    # loop, which had no guard — the loop exited and frames 3-5 were never read.
    write(input,
        JSON.json(Dict("jsonrpc" => "2.0", "id" => 1, "method" => "tools/list", "params" => Dict())), "\n",
        """{"jsonrpc":"2.0","id":2,"method":"tools/list","params":null}""", "\n",
        JSON.json(Dict("jsonrpc" => "2.0", "id" => 3, "method" => "tools/list", "params" => Dict())), "\n",
        JSON.json(Dict("jsonrpc" => "2.0", "id" => 4, "method" => "tools/list", "params" => Dict())), "\n",
        JSON.json(Dict("jsonrpc" => "2.0", "id" => 5, "method" => "tools/call",
            "params" => Dict("name" => "shout", "arguments" => Dict("s" => "ok")))), "\n")
    seekstart(input)

    UniLM._serve_stdio(server; input, output)

    seekstart(output)
    lines = filter(!isempty, split(String(take!(output)), "\n"))
    @test length(lines) == 5                       # all five frames answered
    @test eof(input)                               # the loop consumed the whole stream
    ids = [JSON.parse(l)["id"] for l in lines]
    @test ids == [1, 2, 3, 4, 5]
    @test !any(l -> haskey(JSON.parse(l), "error"), lines)
    @test length(JSON.parse(lines[2])["result"]["tools"]) == 1   # null params ⇒ empty params
    @test JSON.parse(lines[5])["result"]["content"][1]["text"] == "OK"
end

@testset "stdio — positional params → -32602, loop keeps serving" begin
    server = _build_params_server()
    input = IOBuffer()
    output = IOBuffer()
    write(input, """{"jsonrpc":"2.0","id":1,"method":"tools/list","params":[1,2]}""", "\n",
        JSON.json(Dict("jsonrpc" => "2.0", "id" => 2, "method" => "tools/list", "params" => Dict())), "\n")
    seekstart(input)

    UniLM._serve_stdio(server; input, output)

    seekstart(output)
    lines = filter(!isempty, split(String(take!(output)), "\n"))
    @test length(lines) == 2
    err = JSON.parse(lines[1])
    @test err["id"] == 1
    @test err["error"]["code"] == -32602
    @test JSON.parse(lines[2])["id"] == 2          # the loop survived and served on
    @test haskey(JSON.parse(lines[2]), "result")
end

@testset "HTTP transport — null/positional params answered as JSON-RPC, not 500" begin
    server = _build_params_server()
    port = _mcp_free_port()
    httpserver = serve(server; transport=:http, port=port, block=false)
    try
        # Un-normalized, this MethodError escaped to a bare empty-body 500.
        nul = HTTP.post("http://127.0.0.1:$port", ["Content-Type" => "application/json"],
            """{"jsonrpc":"2.0","id":1,"method":"tools/list","params":null}""";
            status_exception=false)
        @test nul.status == 200
        @test length(JSON.parse(String(nul.body))["result"]["tools"]) == 1

        pos = HTTP.post("http://127.0.0.1:$port", ["Content-Type" => "application/json"],
            """{"jsonrpc":"2.0","id":2,"method":"tools/list","params":[1,2]}""";
            status_exception=false)
        # An answerable request carrying a JSON-RPC-level error: 200 with the error body.
        @test pos.status == 200
        parsed = JSON.parse(String(pos.body))
        @test parsed["id"] == 2
        @test parsed["error"]["code"] == -32602

        ok = HTTP.post("http://127.0.0.1:$port", ["Content-Type" => "application/json"],
            JSON.json(Dict("jsonrpc" => "2.0", "id" => 3, "method" => "ping", "params" => Dict()));
            status_exception=false)
        @test ok.status == 200                     # still serving
    finally
        close(httpserver)
    end
end

# ─── Dispatch-layer faults: contained, generic, logged ────────────────────────

"A `resources/read` frame whose non-string `uri` reaches `match(::Regex, uri)` in the
template scan — a client-triggerable internal fault that is not a tool error."
_internal_fault_request(id) = Dict{String,Any}("jsonrpc" => "2.0", "id" => id,
    "method" => "resources/read", "params" => Dict{String,Any}("uri" => 42))

@testset "dispatch fault → generic -32603, internals logged not returned" begin
    server = MCPServer("fault-probe", "1.0.0")
    register_resource_template!(server, "file://{name}", "Files", p -> "c")
    # The fault is real: unguarded dispatch throws rather than returning a response.
    @test_throws MethodError UniLM._dispatch_mcp(server, _internal_fault_request(77))

    resp = @test_logs (:error,) match_mode=:any UniLM._dispatch_guarded(server, _internal_fault_request(77))
    @test resp["id"] == 77
    @test resp["error"]["code"] == -32603
    @test resp["error"]["message"] == "Internal error"     # generic, no exception text
    body = JSON.json(resp)
    for internal in ("MethodError", "match", "Regex", ".jl", "mcp_server", "Stacktrace")
        @test !contains(body, internal)
    end
    @test !haskey(resp["error"], "data")
    # The same frame without an id is a notification: dispatch returns before the
    # handler runs, so there is nothing to fault on and nothing to answer — a
    # notification never gets a response, error or otherwise.
    @test isnothing(UniLM._dispatch_guarded(server,
        Dict{String,Any}("jsonrpc" => "2.0", "method" => "resources/read",
            "params" => Dict{String,Any}("uri" => 42))))
end

@testset "dispatch fault does not stop either transport" begin
    server = MCPServer("fault-transport", "1.0.0")
    register_resource_template!(server, "file://{name}", "Files", p -> "c")
    input = IOBuffer()
    output = IOBuffer()
    write(input, JSON.json(_internal_fault_request(1)), "\n",
        JSON.json(Dict("jsonrpc" => "2.0", "id" => 2, "method" => "ping", "params" => Dict())), "\n")
    seekstart(input)
    @test_logs (:error,) match_mode=:any UniLM._serve_stdio(server; input, output)
    seekstart(output)
    lines = filter(!isempty, split(String(take!(output)), "\n"))
    @test length(lines) == 2
    @test JSON.parse(lines[1])["error"]["code"] == -32603
    @test haskey(JSON.parse(lines[2]), "result")   # loop alive

    port = _mcp_free_port()
    httpserver = serve(server; transport=:http, port=port, block=false)
    try
        r = HTTP.post("http://127.0.0.1:$port", ["Content-Type" => "application/json"],
            JSON.json(_internal_fault_request(3)); status_exception=false)
        @test r.status == 200                      # a JSON-RPC error body, not a bare 500
        parsed = JSON.parse(String(r.body))
        @test parsed["id"] == 3
        @test parsed["error"]["code"] == -32603
        @test parsed["error"]["message"] == "Internal error"
        ok = HTTP.post("http://127.0.0.1:$port", ["Content-Type" => "application/json"],
            JSON.json(Dict("jsonrpc" => "2.0", "id" => 4, "method" => "ping", "params" => Dict()));
            status_exception=false)
        @test ok.status == 200
    finally
        close(httpserver)
    end
end

@testset "tool-handler errors still reach the client as tool results" begin
    # The guard must not swallow this path: a failing tool is reported as an
    # `isError` result carrying the message, which is how a model self-corrects.
    server = MCPServer("tool-error", "1.0.0")
    register_tool!(server, "boom", "Always fails", Dict{String,Any}("type" => "object"),
        args -> error("tool detail the model needs"))
    resp = UniLM._dispatch_guarded(server, Dict{String,Any}("jsonrpc" => "2.0", "id" => 1,
        "method" => "tools/call", "params" => Dict{String,Any}("name" => "boom")))
    @test !haskey(resp, "error")                   # not a protocol-level error
    @test resp["result"]["isError"] == true
    @test contains(resp["result"]["content"][1]["text"], "tool detail the model needs")
end

# ─── Frame size cap ───────────────────────────────────────────────────────────

@testset "_read_frame bounds what it buffers" begin
    cap = UniLM._MCP_MAX_FRAME_BYTES
    @test cap == 16 * 1024 * 1024
    # Bytes past the cap are drained, not buffered: the returned line stays at the cap.
    line, overflow = UniLM._read_frame(IOBuffer(repeat('a', cap + 1) * "\n"), cap)
    @test overflow
    @test sizeof(line) == cap
    # Frames at or under the cap are returned whole, with no overflow flag.
    small, ov = UniLM._read_frame(IOBuffer("{\"a\":1}\n{\"b\":2}\n"), cap)
    @test (small, ov) == ("{\"a\":1}", false)
    # A CRLF line ending is trimmed like `readline` does, and framing is per-line.
    crlf, _ = UniLM._read_frame(IOBuffer("{\"a\":1}\r\n"), cap)
    @test crlf == "{\"a\":1}"
    # A final line without a newline is still a frame; then EOF.
    io = IOBuffer("tail")
    @test UniLM._read_frame(io, cap) == ("tail", false)
    @test eof(io)
end

@testset "stdio — over-cap frame → -32600 unparsed, loop keeps serving" begin
    server = MCPServer("frame-cap", "1.0.0")
    input = IOBuffer()
    output = IOBuffer()
    # Not valid JSON either: if the cap did not short-circuit, JSON.parse would run
    # on the whole payload and answer -32700 instead of -32600.
    write(input, repeat('a', UniLM._MCP_MAX_FRAME_BYTES + 1), "\n",
        JSON.json(Dict("jsonrpc" => "2.0", "id" => 9, "method" => "ping", "params" => Dict())), "\n")
    seekstart(input)

    UniLM._serve_stdio(server; input, output)

    seekstart(output)
    lines = filter(!isempty, split(String(take!(output)), "\n"))
    @test length(lines) == 2
    err = JSON.parse(lines[1])
    @test err["id"] === nothing                    # unparsed ⇒ no id to correlate
    @test err["error"]["code"] == -32600
    @test contains(err["error"]["message"], "exceeds")
    @test JSON.parse(lines[2])["id"] == 9          # the next frame is served normally
end

@testset "HTTP transport — over-cap body → 413 before parsing, server alive" begin
    server = _build_http_server()
    port = _mcp_free_port()
    httpserver = serve(server; transport=:http, port=port, block=false)
    try
        over = HTTP.post("http://127.0.0.1:$port", ["Content-Type" => "application/json"],
            repeat('a', UniLM._MCP_MAX_FRAME_BYTES + 1); status_exception=false)
        @test over.status == 413                   # not the 400 a parse attempt would give
        under = HTTP.post("http://127.0.0.1:$port", ["Content-Type" => "application/json"],
            JSON.json(Dict("jsonrpc" => "2.0", "id" => 1, "method" => "ping", "params" => Dict()));
            status_exception=false)
        @test under.status == 200                  # still serving
        @test JSON.parse(String(under.body))["id"] == 1
    finally
        close(httpserver)
    end
end

@testset "HTTP transport — bodyless POST → 400 parse error, server alive" begin
    # A POST carrying no payload is legal HTTP and reaches the handler like any
    # other. HTTP.jl does not hand it the same body object as a POST with bytes:
    # a zero-length request has its own representation, which answers no size
    # question. The size guard must still measure it, so an empty payload comes
    # back as the JSON-RPC parse error it is (-32700) rather than as a transport
    # failure that takes the exchange down.
    server = _build_http_server()
    port = _mcp_free_port()
    httpserver = serve(server; transport=:http, port=port, block=false)
    try
        empty = HTTP.post("http://127.0.0.1:$port", ["Content-Type" => "application/json"], "";
            status_exception=false)
        @test empty.status == 400
        @test JSON.parse(String(empty.body))["error"]["code"] == -32700
        ok = HTTP.post("http://127.0.0.1:$port", ["Content-Type" => "application/json"],
            JSON.json(Dict("jsonrpc" => "2.0", "id" => 4, "method" => "ping", "params" => Dict()));
            status_exception=false)
        @test ok.status == 200                     # still serving
        @test JSON.parse(String(ok.body))["id"] == 4
    finally
        close(httpserver)
    end
end

# ─── Concurrency: registries and HTTP handler scheduling ─────────────────────

"Serve `server` over HTTP on an OS-assigned ephemeral port, re-probing when the port is
taken between the probe and the bind. Returns (handle, port)."
function _mcp_serve_ephemeral(server::MCPServer; kwargs...)
    for attempt in 1:5
        port = _mcp_free_port()
        try
            return serve(server; transport=:http, port=port, block=false, kwargs...), port
        catch e
            e isa InterruptException && rethrow()
            attempt == 5 && rethrow()
        end
    end
end

_rpc(id, method, params=Dict{String,Any}()) = Dict{String,Any}("jsonrpc" => "2.0", "id" => id,
    "method" => method, "params" => params)

@testset "registration during dispatch is safe (data-race regression test)" begin
    # Data-race regression test. The registries are a plain Dict/Vector, and `register_*!`
    # while `serve` dispatched on other threads corrupted the heap (a Dict rehash under a
    # concurrent iteration aborted the process). Each round, three dispatch loops race a
    # fourth task registering 1000 tools (and resources, templates, prompts) on a fresh
    # server, so every registry grows — and rehashes — while it is being read. Every
    # response must be well formed and every registration must land.
    list = _rpc(1, "tools/list")
    others = [_rpc(2, "resources/list"), _rpc(3, "resources/templates/list"), _rpc(4, "prompts/list"),
              _rpc(5, "resources/read", Dict{String,Any}("uri" => "race://seed")),
              _rpc(6, "resources/read", Dict{String,Any}("uri" => "nomatch://x")),
              _rpc(7, "initialize", Dict{String,Any}("protocolVersion" => UniLM._MCP_PROTOCOL_VERSION))]
    reqs = collect(Iterators.flatten([list, list, list, o] for o in others))
    well_formed(r) = haskey(r, "result") || r["error"]["code"] == -32002   # nomatch → not found
    function race_round()
        server = MCPServer("race", "1.0.0")
        register_resource!(server, "race://seed", "seed", () -> "seed")
        foreach(req -> UniLM._dispatch_mcp(server, req), reqs)   # compiled before the race starts
        stop = Threads.Atomic{Bool}(false)
        served, anomalies = Threads.Atomic{Int}(0), Threads.Atomic{Int}(0)
        readers = [Threads.@spawn begin
            while !stop[]
                for req in reqs
                    ok = try
                        well_formed(UniLM._dispatch_mcp(server, req))
                    catch e
                        e isa InterruptException && rethrow()
                        false
                    end
                    Threads.atomic_add!(ok ? served : anomalies, 1)
                end
            end
        end for _ in 1:3]
        writer = Threads.@spawn for i in 1:1000
            register_tool!(server, "t$i", nothing, Dict{String,Any}("type" => "object"), _ -> "ok")
            if i % 10 == 0
                register_resource!(server, "race://r$i", "r$i", () -> "r")
                register_resource_template!(server, "race$i://{x}", "tmpl$i", p -> "t")
                register_prompt!(server, "p$i", _ -> Dict{String,Any}[])
            end
        end
        finished = timedwait(() -> istaskdone(writer), 10.0)
        stop[] = true
        readers_done = timedwait(() -> all(istaskdone, readers), 25.0)
        (; finished, readers_done, served = served[], anomalies = anomalies[],
           sizes = (length(server.tools), length(server.resources),
                    length(server.resource_templates), length(server.prompts)))
    end
    t0 = time()
    rounds = [race_round()]
    while length(rounds) < 8 && time() - t0 < 6.0
        push!(rounds, race_round())
    end
    @test all(r -> r.finished === :ok && r.readers_done === :ok, rounds)
    @test sum(r -> r.anomalies, rounds) == 0
    @test all(r -> r.served > 0, rounds)
    @test all(r -> r.sizes == (1000, 101, 100, 100), rounds)
end

"""POST `body` to 127.0.0.1:`port` over a fresh raw TCP connection; returns the raw
response. Bypasses the HTTP.jl client, which writes a request body from a task on the
default pool — a pool this test deliberately saturates."""
function _raw_post(port::Int, body::String)::String
    sock = Sockets.connect(Sockets.localhost, port)
    try
        write(sock, "POST / HTTP/1.1\r\nHost: 127.0.0.1:$port\r\nContent-Type: application/json\r\n" *
                    "Content-Length: $(sizeof(body))\r\nConnection: close\r\n\r\n", body)
        read(sock, String)
    finally
        close(sock)
    end
end

@testset "serve(:http) — handlers run on the default pool; ping stays prompt" begin
    # HTTP.jl runs each request handler on the :interactive pool, whose first thread also
    # drives the event loop. A CPU-bound tool handler there blocked every other request —
    # and every Timer and IO completion in the process — until it finished. Handlers now
    # run on the default pool, one task per request; protocol requests such as ping are
    # answered inline and stay prompt while handlers keep the default pool busy.
    if Threads.nthreads(:default) < 4 || Threads.nthreads(:interactive) < 1
        @test_skip "needs --threads=4,1 or more"
    else
        server = MCPServer("spin", "1.0.0")
        started = Threads.Atomic{Int}(0)
        register_tool!(server, "spin", "CPU-bound for 1 s", Dict{String,Any}("type" => "object"),
            function (_)
                Threads.atomic_add!(started, 1)
                t0 = time()
                while time() - t0 < 1.0 end        # no yield point: holds its thread
                "spun"
            end)
        register_tool!(server, "warm", nothing, Dict{String,Any}("type" => "object"), _ -> "ok")
        httpserver, port = _mcp_serve_ephemeral(server)
        post(body) = HTTP.post("http://127.0.0.1:$port", ["Content-Type" => "application/json"],
            JSON.json(body); status_exception=false)
        try
            # Compile the request paths first: the timings below are about scheduling.
            post(_rpc(0, "tools/call", Dict{String,Any}("name" => "warm")))
            _raw_post(port, JSON.json(_rpc(0, "ping")))
            t0 = time()
            calls = [Threads.@spawn :interactive (post(_rpc(i, "tools/call",
                Dict{String,Any}("name" => "spin"))), time()) for i in 1:4]
            @test timedwait(() -> started[] >= 2, 25.0) === :ok
            tp = time()
            pinger = Threads.@spawn :interactive (_raw_post(port, JSON.json(_rpc(99, "ping"))), time() - tp)
            @test timedwait(() -> istaskdone(pinger) && all(istaskdone, calls), 25.0) === :ok
            pong, ping_s = fetch(pinger)
            total_s = maximum(last(fetch(c)) for c in calls) - t0
            (ping_s < 0.5 && total_s < 2.5) || @warn "HTTP handler scheduling" ping_s total_s
            @test startswith(pong, "HTTP/1.1 200") && occursin("\"id\":99", pong)
            @test ping_s < 0.5                     # not queued behind a spinning handler
            @test total_s < 2.5                    # the four 1 s handlers ran concurrently
            @test all(c -> JSON.parse(String(first(fetch(c)).body))["result"]["content"][1]["text"] == "spun", calls)
        finally
            close(httpserver)
        end
    end
end

# ─── Inferred-schema registration binds arguments by name ─────────────────────

@testset "inferred-schema register_tool! binds arguments by name" begin
    server = MCPServer("infer", "1.0.0")
    register_tool!(server, "add", "Add two integers", (a::Int, b::Int) -> a + b)
    sch = server.tools["add"].input_schema
    @test sch["properties"] == Dict{String,Any}("a" => Dict{String,Any}("type" => "integer"),
                                               "b" => Dict{String,Any}("type" => "integer"))
    @test Set(sch["required"]) == Set(["a", "b"])
    call(name, args) = UniLM._dispatch_guarded(server, _rpc(1, "tools/call",
        Dict{String,Any}("name" => name, "arguments" => args)))
    # A schema-conforming call reaches the handler with its positional arguments.
    r = call("add", Dict{String,Any}("a" => 2, "b" => 3))
    @test r["result"]["isError"] == false
    @test r["result"]["content"][1]["text"] == "5"
    # A `Union{T,Nothing}` parameter is optional: omitted, it binds `nothing`.
    register_tool!(server, "greet", nothing,
        (name::String, title::Union{String,Nothing}) -> isnothing(title) ? "hi $name" : "hi $title $name")
    @test server.tools["greet"].input_schema["required"] == ["name"]
    @test call("greet", Dict{String,Any}("name" => "Ada"))["result"]["content"][1]["text"] == "hi Ada"
    @test call("greet", Dict{String,Any}("name" => "Ada", "title" => "Dr"))["result"]["content"][1]["text"] == "hi Dr Ada"
    # Calls that violate the advertised schema are invalid params, not handler errors.
    @test call("add", Dict{String,Any}("a" => 2))["error"]["code"] == -32602
    @test call("add", Dict{String,Any}("a" => 2, "b" => 2.5))["error"]["code"] == -32602
    # A handler taking ONE dictionary is the explicit-schema calling convention: inferring a
    # schema from it would advertise a single required `args` object. Rejected loudly.
    err = try
        register_tool!(server, "dict", "d", (args::Dict{String,Any}) -> args)
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test err isa ArgumentError && occursin("input_schema", err.msg)
    @test !haskey(server.tools, "dict")
end

# ─── Handler failure modes ────────────────────────────────────────────────────

@testset "@mcp_tool binding never fabricates values: violations are -32602" begin
    server = MCPServer("bind", "1.0.0")
    @mcp_tool server function typed(s::String, n::Int, f::Float64, b::Bool)::String
        string(s, n, f, b)
    end
    call(args) = UniLM._dispatch_guarded(server, _rpc(1, "tools/call",
        Dict{String,Any}("name" => "typed", "arguments" => args)))
    good = Dict{String,Any}("s" => "x", "n" => 1, "f" => 1.5, "b" => false)
    @test call(good)["result"]["content"][1]["text"] == "x11.5false"
    for (key, bad) in (("s", 42), ("n", 2.5), ("n", "3"), ("n", true), ("f", "1.0"), ("b", 1))
        args = merge(good, Dict{String,Any}(key => bad))
        resp = call(args)
        @test get(get(resp, "error", Dict()), "code", nothing) == -32602
        @test occursin(key, get(get(resp, "error", Dict()), "message", ""))
    end
    # A missing required argument is not the string "nothing".
    missing_s = call(Dict{String,Any}("n" => 1, "f" => 1.5, "b" => false))
    @test get(get(missing_s, "error", Dict()), "code", nothing) == -32602
    @test occursin("s", get(get(missing_s, "error", Dict()), "message", ""))
    # `arguments` must be an object (CallToolRequest schema).
    for notobj in (Any[1, 2], "str", 5, nothing)
        resp = call(notobj)
        @test get(get(resp, "error", Dict()), "code", nothing) == -32602
    end
end

@testset "InterruptException from any handler propagates" begin
    # A user interrupt must abort dispatch, not be reported to the peer as a failure.
    server = MCPServer("int", "1.0.0")
    register_tool!(server, "t", nothing, Dict{String,Any}("type" => "object"), _ -> throw(InterruptException()))
    register_resource!(server, "int://x", "x", () -> throw(InterruptException()))
    register_resource_template!(server, "intt://{x}", "tx", _ -> throw(InterruptException()))
    register_prompt!(server, "p", _ -> throw(InterruptException()))
    for (m, p) in (("tools/call", Dict{String,Any}("name" => "t")),
                   ("resources/read", Dict{String,Any}("uri" => "int://x")),
                   ("resources/read", Dict{String,Any}("uri" => "intt://y")),
                   ("prompts/get", Dict{String,Any}("name" => "p")))
        @test_throws InterruptException UniLM._dispatch_guarded(server, _rpc(1, m, p))
    end
end

@testset "a JSON-RPC response sent to the server is accepted without an answer" begin
    # A response (id, no method) answers a server-initiated request; it is not a request.
    server = _build_params_server()
    @test isnothing(UniLM._dispatch_mcp(server, Dict{String,Any}("jsonrpc" => "2.0", "id" => 7,
        "result" => Dict{String,Any}())))
    @test isnothing(UniLM._dispatch_mcp(server, Dict{String,Any}("jsonrpc" => "2.0", "id" => 8,
        "error" => Dict{String,Any}("code" => -32601, "message" => "Method not found"))))
    input, output = IOBuffer(), IOBuffer()
    write(input, """{"jsonrpc":"2.0","id":7,"result":{}}""", "\n", JSON.json(_rpc(9, "ping")), "\n")
    seekstart(input)
    UniLM._serve_stdio(server; input, output)
    lines = filter(!isempty, split(String(take!(output)), "\n"))
    @test length(lines) == 1 && JSON.parse(lines[1])["id"] == 9
    httpserver, port = _mcp_serve_ephemeral(server)
    try
        r = HTTP.post("http://127.0.0.1:$port", ["Content-Type" => "application/json"],
            """{"jsonrpc":"2.0","id":7,"result":{}}"""; status_exception=false)
        @test r.status == 202
        @test isempty(String(r.body))
    finally
        close(httpserver)
    end
end

# ─── Protocol version negotiation (lifecycle) and the HTTP version header ────

@testset "initialize answers the requested protocol version when supported" begin
    # Lifecycle: a server supporting the requested version MUST answer with it; otherwise
    # it answers another version it supports (the latest).
    server = MCPServer("negotiate", "1.0.0")
    negotiated(v) = UniLM._dispatch_mcp(server, _rpc(1, "initialize", Dict{String,Any}(
        "protocolVersion" => v, "capabilities" => Dict{String,Any}(),
        "clientInfo" => Dict{String,Any}("name" => "t", "version" => "1"))))["result"]["protocolVersion"]
    for v in UniLM._MCP_SUPPORTED_PROTOCOL_VERSIONS
        @test negotiated(v) == v
    end
    @test negotiated("1999-01-01") == UniLM._MCP_PROTOCOL_VERSION
    @test negotiated(42) == UniLM._MCP_PROTOCOL_VERSION
end

@testset "HTTP transport — unsupported MCP-Protocol-Version header → 400" begin
    server = _build_http_server()
    httpserver, port = _mcp_serve_ephemeral(server)
    post(body, hdrs...) = HTTP.post("http://127.0.0.1:$port",
        ["Content-Type" => "application/json", hdrs...], JSON.json(body); status_exception=false)
    try
        @test post(_rpc(1, "ping"), "MCP-Protocol-Version" => "1999-01-01").status == 400
        @test post(_rpc(2, "ping"), "MCP-Protocol-Version" => "2025-06-18").status == 200
        @test post(_rpc(3, "ping")).status == 200          # absent: backwards-compatible default
        # initialize negotiates in its body, so an unknown header value there is not refused.
        init = post(_rpc(4, "initialize", Dict{String,Any}("protocolVersion" => "2099-01-01",
            "capabilities" => Dict{String,Any}(), "clientInfo" => Dict{String,Any}("name" => "t", "version" => "1"))),
            "MCP-Protocol-Version" => "2099-01-01")
        @test init.status == 200
        @test JSON.parse(String(init.body))["result"]["protocolVersion"] == UniLM._MCP_PROTOCOL_VERSION
        del = HTTP.request("DELETE", "http://127.0.0.1:$port", ["MCP-Protocol-Version" => "1999-01-01"];
            status_exception=false)
        @test del.status == 400
    finally
        close(httpserver)
    end
end

# ─── stdio: the protocol owns stdout ──────────────────────────────────────────

@testset "serve(:stdio) keeps handler output off the protocol stream" begin
    # stdio framing: the server MUST NOT write anything to stdout that is not an MCP
    # message, but a handler that prints (println, @show, a progress bar) writes to the
    # process stdout. `serve` keeps the protocol stream and points stdout at stderr while
    # it serves, so the noise lands on stderr and every stdout line stays a frame.
    proj = dirname(dirname(pathof(UniLM)))
    src = """
    using UniLM
    server = MCPServer("noisy", "1.0.0")
    register_tool!(server, "noisy", nothing, Dict{String,Any}("type" => "object"),
        _ -> (println("HANDLER-NOISE"); ccall(:puts, Cint, (Cstring,), "C-LEVEL-NOISE");
              Base.Libc.flush_cstdio(); "quiet-result"))
    serve(server)
    """
    srcfile, sio = mktemp(); write(sio, src); close(sio)
    infile, iio = mktemp()
    write(iio, JSON.json(_rpc(1, "initialize", Dict{String,Any}("protocolVersion" => UniLM._MCP_PROTOCOL_VERSION,
        "capabilities" => Dict{String,Any}(), "clientInfo" => Dict{String,Any}("name" => "t", "version" => "1")))), "\n",
        JSON.json(_rpc(2, "tools/call", Dict{String,Any}("name" => "noisy"))), "\n")
    close(iio)
    outfile, oio = mktemp(); close(oio)
    errfile, eio = mktemp(); close(eio)
    try
        p = run(pipeline(`$(Base.julia_cmd()) --startup-file=no --project=$proj $srcfile`;
            stdin=infile, stdout=outfile, stderr=errfile); wait=false)
        @test timedwait(() -> process_exited(p), 120.0) === :ok
        lines = filter(!isempty, readlines(outfile))
        @test length(lines) == 2
        @test all(l -> startswith(l, "{"), lines)
        @test !any(l -> occursin("NOISE", l), lines)
        @test JSON.parse(lines[end])["result"]["content"][1]["text"] == "quiet-result"
        errtext = read(errfile, String)
        @test occursin("HANDLER-NOISE", errtext)
        @test occursin("C-LEVEL-NOISE", errtext)
    finally
        try; run(pipeline(`pkill -f $srcfile`; stderr=devnull)); catch; end
        foreach(f -> rm(f; force=true), (srcfile, infile, outfile, errfile))
    end
end
