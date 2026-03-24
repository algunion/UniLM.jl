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
