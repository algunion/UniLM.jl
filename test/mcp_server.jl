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
