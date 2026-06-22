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

    @testset "CallableTool{GPTTool} bridge" begin
        gpt_tool = GPTTool(func=GPTFunctionSignature(
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

@testset "resources/read — static handler throws (-32603)" begin
    # src/mcp_server.jl:362 — static resource handler error → JSON-RPC -32603.
    server = MCPServer("res-err", "1.0.0")
    register_resource!(server, "boom://static", "Boom", () -> error("static handler exploded"))

    req = Dict{String,Any}("jsonrpc" => "2.0", "id" => 12, "method" => "resources/read",
        "params" => Dict{String,Any}("uri" => "boom://static"))
    resp = UniLM._dispatch_mcp(server, req)

    @test resp["id"] == 12
    @test !haskey(resp, "result")
    @test resp["error"]["code"] == -32603
    @test contains(resp["error"]["message"], "Resource read error")
    @test contains(resp["error"]["message"], "static handler exploded")
end

@testset "resources/read — template handler throws (-32603)" begin
    # src/mcp_server.jl:375 — TEMPLATE branch (distinct from static 362) handler error → -32603.
    server = MCPServer("tmpl-err", "1.0.0")
    register_resource_template!(server, "boom://{id}", "BoomTmpl",
        p -> error("template handler exploded for $(p["id"])"))

    req = Dict{String,Any}("jsonrpc" => "2.0", "id" => 13, "method" => "resources/read",
        "params" => Dict{String,Any}("uri" => "boom://42"))
    resp = UniLM._dispatch_mcp(server, req)

    @test resp["id"] == 13
    @test !haskey(resp, "result")
    @test resp["error"]["code"] == -32603
    @test contains(resp["error"]["message"], "Resource read error")
    # Confirms the template branch ran (param interpolated into the thrown message).
    @test contains(resp["error"]["message"], "template handler exploded for 42")
end

@testset "prompts/get — handler throws (-32603)" begin
    # src/mcp_server.jl:402 — prompt handler error → JSON-RPC -32603 "Prompt error".
    server = MCPServer("prompt-err", "1.0.0")
    register_prompt!(server, "explode", args -> error("prompt handler exploded");
        arguments=[Dict{String,Any}("name" => "x", "required" => false)])

    req = Dict{String,Any}("jsonrpc" => "2.0", "id" => 14, "method" => "prompts/get",
        "params" => Dict{String,Any}("name" => "explode", "arguments" => Dict{String,Any}()))
    resp = UniLM._dispatch_mcp(server, req)

    @test resp["id"] == 14
    @test !haskey(resp, "result")
    @test resp["error"]["code"] == -32603
    @test contains(resp["error"]["message"], "Prompt error")
    @test contains(resp["error"]["message"], "prompt handler exploded")
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
    httpserver = serve(server; transport=:http, host="127.0.0.1", port=port)
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
    httpserver = serve(server; transport=:http, port=port)  # default host
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
    httpserver = serve(server; transport=:http, port=port)
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
    httpserver = serve(server; transport=:http, port=port)
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
    # _mcp_convert(Int, 5.0) must round → Int(5); String conv on a number → "42"; Bool passes.
    out = server.tools["mixt"].handler(Dict{String,Any}(
        "s" => 42,            # _mcp_convert(String, 42) → "42"  (line 564)
        "n" => 5.0,           # _mcp_convert(Int, 5.0)   → 5     (line 565, AbstractFloat branch)
        "f" => 2,             # _mcp_convert(Float64, 2) → 2.0   (line 566)
        "b" => true,          # _mcp_convert(Bool, true) → true  (line 567)
        "xs" => [3, 4]))      # _mcp_convert(Vector{Int}, …) passthrough (line 568)
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
