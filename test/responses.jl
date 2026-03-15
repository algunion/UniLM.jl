@testset "InputMessage" begin
    @testset "basic creation" begin
        m = InputMessage(role="user", content="Hello")
        @test m.role == "user"
        @test m.content == "Hello"
    end

    @testset "multimodal content" begin
        m = InputMessage(role="user", content=[
            input_text("Describe this image:"),
            input_image("https://example.com/img.png")
        ])
        @test m.role == "user"
        @test length(m.content) == 2
    end

    @testset "JSON serialization" begin
        m = InputMessage(role="user", content="hi")
        lowered = JSON.lower(m)
        @test lowered[:role] == "user"
        @test lowered[:content] == "hi"
    end
end

@testset "Content part helpers" begin
    @testset "input_text" begin
        p = input_text("hello")
        @test p[:type] == "input_text"
        @test p[:text] == "hello"
    end

    @testset "input_image" begin
        p = input_image("https://example.com/img.png")
        @test p[:type] == "input_image"
        @test p[:image_url] == "https://example.com/img.png"
        @test !haskey(p, :detail)

        p2 = input_image("https://example.com/img.png", detail="high")
        @test p2[:detail] == "high"
    end

    @testset "input_file with url" begin
        p = input_file(url="https://example.com/doc.pdf")
        @test p[:type] == "input_file"
        @test p[:file_url] == "https://example.com/doc.pdf"
        @test !haskey(p, :file_id)
    end

    @testset "input_file with id" begin
        p = input_file(id="file-abc123")
        @test p[:type] == "input_file"
        @test p[:file_id] == "file-abc123"
        @test !haskey(p, :file_url)
    end

    @testset "input_file requires url or id" begin
        @test_throws ArgumentError input_file()
    end
end

@testset "ResponseTool hierarchy" begin
    @test FunctionTool <: UniLM.ResponseTool
    @test WebSearchTool <: UniLM.ResponseTool
    @test FileSearchTool <: UniLM.ResponseTool
    @test MCPTool <: UniLM.ResponseTool
    @test ComputerUseTool <: UniLM.ResponseTool
    @test ImageGenerationTool <: UniLM.ResponseTool
    @test CodeInterpreterTool <: UniLM.ResponseTool
end

@testset "MCPTool" begin
    @testset "minimal creation" begin
        t = MCPTool(server_label="my-server", server_url="https://mcp.example.com/sse")
        @test t.server_label == "my-server"
        @test t.server_url == "https://mcp.example.com/sse"
        @test t.require_approval == "never"
        @test isnothing(t.allowed_tools)
        @test isnothing(t.headers)
    end

    @testset "full creation" begin
        t = MCPTool(
            server_label="tools",
            server_url="https://mcp.example.com/sse",
            require_approval="always",
            allowed_tools=["search", "read"],
            headers=Dict("Authorization" => "Bearer token")
        )
        @test t.require_approval == "always"
        @test t.allowed_tools == ["search", "read"]
        @test t.headers["Authorization"] == "Bearer token"
    end

    @testset "JSON serialization" begin
        t = MCPTool(server_label="srv", server_url="https://example.com")
        lowered = JSON.lower(t)
        @test lowered[:type] == "mcp"
        @test lowered[:server_label] == "srv"
        @test lowered[:server_url] == "https://example.com"
        @test lowered[:require_approval] == "never"
        @test !haskey(lowered, :allowed_tools)
        @test !haskey(lowered, :headers)
    end

    @testset "mcp_tool convenience" begin
        t = mcp_tool("my-server", "https://example.com"; allowed_tools=["fn1"])
        @test t isa MCPTool
        @test t.server_label == "my-server"
        @test t.allowed_tools == ["fn1"]
    end
end

@testset "ComputerUseTool" begin
    @testset "defaults" begin
        t = ComputerUseTool()
        @test t.display_width == 1024
        @test t.display_height == 768
        @test isnothing(t.environment)
    end

    @testset "custom" begin
        t = ComputerUseTool(display_width=1920, display_height=1080, environment="browser")
        @test t.display_width == 1920
        @test t.environment == "browser"
    end

    @testset "JSON serialization" begin
        t = ComputerUseTool()
        lowered = JSON.lower(t)
        @test lowered[:type] == "computer_use_preview"
        @test lowered[:display_width] == 1024
        @test lowered[:display_height] == 768
        @test !haskey(lowered, :environment)
    end

    @testset "computer_use convenience" begin
        t = computer_use(display_width=800, display_height=600)
        @test t isa ComputerUseTool
        @test t.display_width == 800
    end
end

@testset "ImageGenerationTool" begin
    @testset "defaults" begin
        t = ImageGenerationTool()
        @test isnothing(t.background)
        @test isnothing(t.output_format)
        @test isnothing(t.output_compression)
        @test isnothing(t.quality)
        @test isnothing(t.size)
    end

    @testset "full creation" begin
        t = ImageGenerationTool(
            background="transparent",
            output_format="png",
            output_compression=80,
            quality="high",
            size="1024x1024"
        )
        @test t.background == "transparent"
        @test t.output_compression == 80
    end

    @testset "JSON serialization" begin
        t = ImageGenerationTool(quality="high", size="1024x1024")
        lowered = JSON.lower(t)
        @test lowered[:type] == "image_generation"
        @test lowered[:quality] == "high"
        @test lowered[:size] == "1024x1024"
        @test !haskey(lowered, :background)
        @test !haskey(lowered, :output_format)
    end

    @testset "image_generation_tool convenience" begin
        t = image_generation_tool(quality="medium")
        @test t isa ImageGenerationTool
        @test t.quality == "medium"
    end
end

@testset "CodeInterpreterTool" begin
    @testset "defaults" begin
        t = CodeInterpreterTool()
        @test isnothing(t.container)
        @test isnothing(t.file_ids)
    end

    @testset "with options" begin
        t = CodeInterpreterTool(
            container=Dict("type" => "auto"),
            file_ids=["file-1", "file-2"]
        )
        @test t.container["type"] == "auto"
        @test length(t.file_ids) == 2
    end

    @testset "JSON serialization" begin
        t = CodeInterpreterTool(file_ids=["file-1"])
        lowered = JSON.lower(t)
        @test lowered[:type] == "code_interpreter"
        @test lowered[:file_ids] == ["file-1"]
        @test !haskey(lowered, :container)
    end

    @testset "code_interpreter convenience" begin
        t = code_interpreter(file_ids=["f1"])
        @test t isa CodeInterpreterTool
        @test t.file_ids == ["f1"]
    end
end

@testset "FunctionTool" begin
    @testset "minimal creation" begin
        t = FunctionTool(name="my_fn")
        @test t.name == "my_fn"
        @test isnothing(t.description)
        @test isnothing(t.parameters)
        @test isnothing(t.strict)
    end

    @testset "full creation" begin
        params = Dict("type" => "object", "properties" => Dict("x" => Dict("type" => "string")))
        t = FunctionTool(name="fn", description="A test", parameters=params, strict=true)
        @test t.description == "A test"
        @test t.parameters == params
        @test t.strict == true
    end

    @testset "JSON serialization" begin
        t = FunctionTool(name="fn", description="desc")
        lowered = JSON.lower(t)
        @test lowered[:type] == "function"
        @test lowered[:name] == "fn"
        @test lowered[:description] == "desc"
        @test !haskey(lowered, :parameters)
        @test !haskey(lowered, :strict)
    end

    @testset "function_tool shorthand" begin
        t = function_tool("get_weather", "Get weather")
        @test t isa FunctionTool
        @test t.name == "get_weather"
        @test t.description == "Get weather"
    end

    @testset "function_tool from bare dict" begin
        d = Dict("name" => "bare_fn", "description" => "bare", "strict" => true)
        t = function_tool(d)
        @test t isa FunctionTool
        @test t.name == "bare_fn"
        @test t.description == "bare"
        @test t.strict == true
    end

    @testset "function_tool from wrapped dict" begin
        d = Dict("type" => "function", "function" => Dict(
            "name" => "wrapped_fn", "description" => "wrapped",
            "parameters" => Dict("type" => "object")))
        t = function_tool(d)
        @test t isa FunctionTool
        @test t.name == "wrapped_fn"
        @test t.description == "wrapped"
        @test t.parameters == Dict("type" => "object")
    end
end

@testset "WebSearchTool" begin
    @testset "default creation" begin
        t = WebSearchTool()
        @test t.search_context_size == "medium"
        @test isnothing(t.user_location)
    end

    @testset "custom creation" begin
        loc = Dict("country" => "US", "city" => "NYC")
        t = WebSearchTool(search_context_size="high", user_location=loc)
        @test t.search_context_size == "high"
        @test t.user_location == loc
    end

    @testset "JSON serialization" begin
        t = WebSearchTool()
        lowered = JSON.lower(t)
        @test lowered[:type] == "web_search_preview"
        @test lowered[:search_context_size] == "medium"
        @test !haskey(lowered, :user_location)
    end

    @testset "web_search shorthand" begin
        t = web_search(context_size="low")
        @test t isa WebSearchTool
        @test t.search_context_size == "low"
    end
end

@testset "FileSearchTool" begin
    @testset "creation" begin
        t = FileSearchTool(vector_store_ids=["vs_123", "vs_456"])
        @test t.vector_store_ids == ["vs_123", "vs_456"]
        @test isnothing(t.max_num_results)
    end

    @testset "JSON serialization" begin
        t = FileSearchTool(vector_store_ids=["vs_123"], max_num_results=10)
        lowered = JSON.lower(t)
        @test lowered[:type] == "file_search"
        @test lowered[:vector_store_ids] == ["vs_123"]
        @test lowered[:max_num_results] == 10
        @test !haskey(lowered, :ranking_options)
    end

    @testset "file_search shorthand" begin
        t = file_search(["vs_abc"], max_results=20)
        @test t isa FileSearchTool
        @test t.vector_store_ids == ["vs_abc"]
        @test t.max_num_results == 20
    end
end

@testset "TextFormatSpec" begin
    @testset "default" begin
        f = UniLM.TextFormatSpec()
        @test f.type == "text"
        @test isnothing(f.name)
    end

    @testset "json_schema" begin
        schema = Dict("type" => "object")
        f = UniLM.TextFormatSpec(type="json_schema", name="my_schema", description="desc", schema=schema, strict=true)
        @test f.type == "json_schema"
        @test f.name == "my_schema"
    end

    @testset "JSON serialization omits nulls" begin
        f = UniLM.TextFormatSpec()
        lowered = JSON.lower(f)
        @test lowered[:type] == "text"
        @test !haskey(lowered, :name)
        @test !haskey(lowered, :schema)
    end
end

@testset "TextConfig & format constructors" begin
    @testset "text_format" begin
        tc = text_format()
        @test tc.format.type == "text"
    end

    @testset "json_schema_format" begin
        schema = Dict("type" => "object", "properties" => Dict("x" => Dict("type" => "string")))
        tc = json_schema_format("my_schema", "A schema", schema)
        @test tc.format.type == "json_schema"
        @test tc.format.name == "my_schema"
        @test tc.format.description == "A schema"
        @test tc.format.schema == schema
    end

    @testset "json_object_format" begin
        tc = json_object_format()
        @test tc.format.type == "json_object"
    end
end

@testset "Reasoning" begin
    @testset "creation" begin
        r = Reasoning(effort="high", summary="concise")
        @test r.effort == "high"
        @test r.summary == "concise"
    end

    @testset "defaults" begin
        r = Reasoning()
        @test isnothing(r.effort)
        @test isnothing(r.summary)
    end

    @testset "JSON serialization omits nulls" begin
        r = Reasoning(effort="medium")
        lowered = JSON.lower(r)
        @test lowered[:effort] == "medium"
        @test !haskey(lowered, :summary)
    end
end

@testset "Respond" begin
    @testset "minimal creation" begin
        r = Respond(input="Tell me a joke")
        @test r.model == "gpt-5.2"
        @test r.input == "Tell me a joke"
        @test r.service == UniLM.OPENAIServiceEndpoint
        @test isnothing(r.instructions)
        @test isnothing(r.tools)
        @test isnothing(r.temperature)
        @test isnothing(r.top_p)
        @test isnothing(r.stream)
        @test isnothing(r.reasoning)
        @test isnothing(r.previous_response_id)
    end

    @testset "full creation" begin
        r = Respond(
            input="Hello",
            model="gpt-5.2",
            instructions="Be helpful",
            temperature=0.5,
            max_output_tokens=100,
            store=true,
            metadata=Dict("key" => "value"),
            truncation="auto"
        )
        @test r.model == "gpt-5.2"
        @test r.instructions == "Be helpful"
        @test r.temperature == 0.5
        @test r.max_output_tokens == 100
        @test r.store == true
        @test r.metadata == Dict("key" => "value")
        @test r.truncation == "auto"
    end

    @testset "temperature and top_p mutual exclusion" begin
        @test_throws ArgumentError Respond(input="test", temperature=0.5, top_p=0.9)
    end

    @testset "with structured input" begin
        r = Respond(input=[InputMessage(role="user", content="Hello")])
        @test r.input isa Vector
        @test length(r.input) == 1
    end

    @testset "with tools" begin
        tools = [function_tool("fn"), web_search()]
        r = Respond(input="test", tools=tools)
        @test length(r.tools) == 2
    end

    @testset "with previous_response_id" begin
        r = Respond(input="Follow-up", previous_response_id="resp_abc123")
        @test r.previous_response_id == "resp_abc123"
    end

    @testset "with reasoning" begin
        r = Respond(input="Solve", model="o3-mini", reasoning=Reasoning(effort="high"))
        @test r.reasoning.effort == "high"
    end

    @testset "JSON serialization" begin
        r = Respond(input="Hello", instructions="Be nice", temperature=0.7)
        lowered = JSON.lower(r)
        @test lowered[:model] == "gpt-5.2"
        @test lowered[:input] == "Hello"
        @test lowered[:instructions] == "Be nice"
        @test lowered[:temperature] == 0.7
        # service excluded, nil fields excluded
        @test !haskey(lowered, :service)
        @test !haskey(lowered, :top_p)
        @test !haskey(lowered, :tools)
        @test !haskey(lowered, :stream)
        @test !haskey(lowered, :reasoning)
        @test !haskey(lowered, :previous_response_id)
    end

    @testset "JSON round-trip" begin
        params = Dict("type" => "object", "properties" => Dict("q" => Dict("type" => "string")))
        r = Respond(
            input=[InputMessage(role="user", content="test")],
            tools=[function_tool("search", "Search tool", parameters=params)],
            text=json_schema_format("output", "desc", Dict("type" => "object")),
            reasoning=Reasoning(effort="low")
        )
        json_str = JSON.json(r)
        parsed = JSON.parse(json_str)

        @test parsed["model"] == "gpt-5.2"
        @test parsed["input"] isa Vector
        @test parsed["input"][1]["role"] == "user"
        @test parsed["tools"] isa Vector
        @test parsed["tools"][1]["type"] == "function"
        @test parsed["tools"][1]["name"] == "search"
        @test parsed["text"]["format"]["type"] == "json_schema"
        @test parsed["reasoning"]["effort"] == "low"
    end
end

@testset "ResponseObject" begin
    raw = Dict{String,Any}(
        "id" => "resp_123",
        "status" => "completed",
        "model" => "gpt-5.2",
        "output" => Any[
            Dict{String,Any}(
            "type" => "message",
            "id" => "msg_1",
            "status" => "completed",
            "role" => "assistant",
            "content" => Any[
                Dict{String,Any}(
                "type" => "output_text",
                "text" => "Hello there!",
                "annotations" => Any[]
            )
            ]
        )
        ],
        "usage" => Dict{String,Any}("input_tokens" => 10, "output_tokens" => 5, "total_tokens" => 15)
    )

    ro = ResponseObject(
        id="resp_123",
        status="completed",
        model="gpt-5.2",
        output=raw["output"],
        usage=raw["usage"],
        raw=raw
    )

    @test ro.id == "resp_123"
    @test ro.status == "completed"
    @test ro.model == "gpt-5.2"
    @test length(ro.output) == 1
end

@testset "output_text" begin
    ro = ResponseObject(
        id="resp_1",
        status="completed",
        model="gpt-5.2",
        output=Any[
            Dict{String,Any}(
            "type" => "message",
            "content" => Any[
                Dict{String,Any}("type" => "output_text", "text" => "Hello!")
            ]
        )
        ],
        raw=Dict{String,Any}()
    )
    @test output_text(ro) == "Hello!"
end

@testset "output_text multiple messages" begin
    ro = ResponseObject(
        id="resp_1",
        status="completed",
        model="gpt-5.2",
        output=Any[
            Dict{String,Any}(
                "type" => "message",
                "content" => Any[
                    Dict{String,Any}("type" => "output_text", "text" => "Line 1")
                ]
            ),
            Dict{String,Any}(
                "type" => "message",
                "content" => Any[
                    Dict{String,Any}("type" => "output_text", "text" => "Line 2")
                ]
            )
        ],
        raw=Dict{String,Any}()
    )
    @test output_text(ro) == "Line 1\nLine 2"
end

@testset "output_text empty output" begin
    ro = ResponseObject(
        id="resp_1", status="completed", model="gpt-5.2",
        output=Any[], raw=Dict{String,Any}()
    )
    @test output_text(ro) == ""
end

@testset "function_calls" begin
    ro = ResponseObject(
        id="resp_1",
        status="completed",
        model="gpt-5.2",
        output=Any[
            Dict{String,Any}(
                "type" => "function_call",
                "id" => "fc_1",
                "call_id" => "call_abc",
                "name" => "get_weather",
                "arguments" => "{\"location\":\"NYC\"}",
                "status" => "completed"
            ),
            Dict{String,Any}(
                "type" => "message",
                "content" => Any[
                    Dict{String,Any}("type" => "output_text", "text" => "result")
                ]
            )
        ],
        raw=Dict{String,Any}()
    )

    calls = function_calls(ro)
    @test length(calls) == 1
    @test calls[1]["name"] == "get_weather"
    @test calls[1]["call_id"] == "call_abc"
end

@testset "function_calls empty" begin
    ro = ResponseObject(
        id="resp_1", status="completed", model="gpt-5.2",
        output=Any[
            Dict{String,Any}("type" => "message", "content" => Any[])
        ],
        raw=Dict{String,Any}()
    )
    @test isempty(function_calls(ro))
end

@testset "Result types" begin
    @testset "ResponseSuccess" begin
        ro = ResponseObject(
            id="resp_1", status="completed", model="gpt-5.2",
            output=Any[], raw=Dict{String,Any}()
        )
        s = ResponseSuccess(response=ro)
        @test s isa UniLM.LLMRequestResponse
        @test s.response.id == "resp_1"
        @test output_text(s) == ""
        @test isempty(function_calls(s))
    end

    @testset "ResponseFailure" begin
        f = ResponseFailure(response="error body", status=400)
        @test f isa UniLM.LLMRequestResponse
        @test f.response == "error body"
        @test f.status == 400
    end

    @testset "ResponseCallError" begin
        e = ResponseCallError(error="timeout")
        @test e isa UniLM.LLMRequestResponse
        @test e.error == "timeout"
        @test isnothing(e.status)

        e2 = ResponseCallError(error="server error", status=503)
        @test e2.status == 503
    end
end

@testset "parse_response" begin
    function make_response(body::Dict; status=200)
        body_bytes = Vector{UInt8}(JSON.json(body))
        HTTP.Response(status, [], body_bytes)
    end

    @testset "text response" begin
        body = Dict(
            "id" => "resp_test",
            "object" => "response",
            "status" => "completed",
            "model" => "gpt-5.2",
            "output" => [Dict(
                "type" => "message",
                "id" => "msg_1",
                "status" => "completed",
                "role" => "assistant",
                "content" => [Dict(
                    "type" => "output_text",
                    "text" => "Hello!",
                    "annotations" => []
                )]
            )],
            "usage" => Dict(
                "input_tokens" => 10,
                "output_tokens" => 5,
                "total_tokens" => 15
            ),
            "error" => nothing,
            "metadata" => Dict()
        )
        resp = make_response(body)
        ro = UniLM.parse_response(resp)
        @test ro.id == "resp_test"
        @test ro.status == "completed"
        @test ro.model == "gpt-5.2"
        @test output_text(ro) == "Hello!"
        @test ro.usage["total_tokens"] == 15
    end

    @testset "function call response" begin
        body = Dict(
            "id" => "resp_fn",
            "object" => "response",
            "status" => "completed",
            "model" => "gpt-5.2",
            "output" => [Dict(
                "type" => "function_call",
                "id" => "fc_1",
                "call_id" => "call_xyz",
                "name" => "get_weather",
                "arguments" => "{\"location\":\"Tokyo\"}",
                "status" => "completed"
            )],
            "usage" => Dict("input_tokens" => 20, "output_tokens" => 10, "total_tokens" => 30),
            "error" => nothing,
            "metadata" => Dict()
        )
        resp = make_response(body)
        ro = UniLM.parse_response(resp)
        @test ro.id == "resp_fn"
        calls = function_calls(ro)
        @test length(calls) == 1
        @test calls[1]["name"] == "get_weather"
        parsed_args = JSON.parse(calls[1]["arguments"])
        @test parsed_args["location"] == "Tokyo"
    end
end

@testset "_parse_response_stream_chunk" begin
    @testset "text delta" begin
        chunk = """event: response.output_text.delta
data: {"type":"response.output_text.delta","delta":"Hello"}"""
        textbuff = IOBuffer()
        failbuff = IOBuffer()
        result = UniLM._parse_response_stream_chunk(chunk, textbuff, failbuff)
        @test result.done == false
        @test String(take!(textbuff)) == "Hello"
    end

    @testset "completed event" begin
        resp_data = Dict(
            "response" => Dict(
                "id" => "resp_1",
                "status" => "completed",
                "model" => "gpt-5.2",
                "output" => [],
                "usage" => Dict("input_tokens" => 1, "output_tokens" => 1, "total_tokens" => 2)
            )
        )
        chunk = "event: response.completed\ndata: $(JSON.json(resp_data))"
        textbuff = IOBuffer()
        failbuff = IOBuffer()
        result = UniLM._parse_response_stream_chunk(chunk, textbuff, failbuff)
        @test result.done == true
        @test !isnothing(result.data)
        @test result.data["response"]["id"] == "resp_1"
    end

    @testset "empty chunk" begin
        textbuff = IOBuffer()
        failbuff = IOBuffer()
        result = UniLM._parse_response_stream_chunk("", textbuff, failbuff)
        @test result.done == false
    end

    @testset "malformed JSON goes to failbuff" begin
        chunk = "event: response.output_text.delta\ndata: {invalid json"
        textbuff = IOBuffer()
        failbuff = IOBuffer()
        result = UniLM._parse_response_stream_chunk(chunk, textbuff, failbuff)
        @test result.done == false
        @test !isempty(take!(failbuff))
    end
end

@testset "respond() error handling" begin
    @testset "respond(::Respond) with invalid API key" begin
        withenv("OPENAI_API_KEY" => "sk-invalid-test-key") do
            r = Respond(input="Hello")
            result = respond(r)
            @test result isa ResponseCallError || result isa ResponseFailure
        end
    end

    @testset "respond(input; kwargs...) convenience method" begin
        withenv("OPENAI_API_KEY" => "sk-invalid-test-key") do
            result = respond("Hello", model="gpt-5.2")
            @test result isa ResponseCallError || result isa ResponseFailure
        end
    end

    @testset "respond(callback, input; kwargs...) do-block form" begin
        withenv("OPENAI_API_KEY" => "sk-invalid-test-key") do
            result = respond("Hello") do chunk, close
                # callback - won't be called because request fails
            end
            # With an invalid key, the streaming request will either
            # return an error or a Task that fails
            @test result isa ResponseCallError || result isa Task
        end
    end
end

@testset "get_response error handling" begin
    @testset "with invalid API key" begin
        withenv("OPENAI_API_KEY" => "sk-invalid-test-key") do
            result = get_response("resp_nonexistent")
            @test result isa ResponseCallError || result isa ResponseFailure
        end
    end
end

@testset "delete_response error handling" begin
    @testset "with invalid API key" begin
        withenv("OPENAI_API_KEY" => "sk-invalid-test-key") do
            result = delete_response("resp_nonexistent")
            @test result isa ResponseCallError || result isa ResponseFailure
        end
    end
end

@testset "list_input_items error handling" begin
    @testset "with invalid API key" begin
        withenv("OPENAI_API_KEY" => "sk-invalid-test-key") do
            result = list_input_items("resp_nonexistent")
            @test result isa ResponseCallError || result isa ResponseFailure
        end
    end

    @testset "with after parameter" begin
        withenv("OPENAI_API_KEY" => "sk-invalid-test-key") do
            result = list_input_items("resp_nonexistent"; limit=10, order="asc", after="item_abc")
            @test result isa ResponseCallError || result isa ResponseFailure
        end
    end
end

@testset "ResponseObject accessors" begin
    @testset "output_text with non-message items" begin
        ro = ResponseObject(
            id="resp_1", status="completed", model="gpt-5.2",
            output=Any[
                Dict{String,Any}("type" => "function_call", "name" => "fn"),
                Dict{String,Any}(
                    "type" => "message",
                    "content" => Any[
                        Dict{String,Any}("type" => "other_type", "data" => "ignored"),
                        Dict{String,Any}("type" => "output_text", "text" => "actual")
                    ]
                )
            ],
            raw=Dict{String,Any}()
        )
        @test output_text(ro) == "actual"
    end

    @testset "function_calls on ResponseSuccess" begin
        ro = ResponseObject(
            id="resp_1", status="completed", model="gpt-5.2",
            output=Any[
                Dict{String,Any}(
                "type" => "function_call",
                "id" => "fc_1",
                "call_id" => "call_1",
                "name" => "fn1",
                "arguments" => "{}",
                "status" => "completed"
            )
            ],
            raw=Dict{String,Any}()
        )
        s = ResponseSuccess(response=ro)
        @test length(function_calls(s)) == 1
        @test function_calls(s)[1]["name"] == "fn1"
    end
end

@testset "ResponseObject optional fields" begin
    ro = ResponseObject(
        id="resp_1", status="completed", model="gpt-5.2",
        output=Any[], raw=Dict{String,Any}(),
        error=Dict{String,Any}("code" => "error"),
        metadata=Dict{String,Any}("key" => "value")
    )
    @test ro.error["code"] == "error"
    @test ro.metadata["key"] == "value"
end

@testset "TextConfig JSON serialization" begin
    tc = TextConfig(format=UniLM.TextFormatSpec(type="json_schema", name="test", description="d", schema=Dict("type" => "object"), strict=true))
    lowered = JSON.lower(tc.format)
    @test lowered[:type] == "json_schema"
    @test lowered[:name] == "test"
    @test lowered[:strict] == true
end

@testset "FileSearchTool with all options" begin
    ranking = Dict("ranker" => "auto")
    filters = Dict("type" => "eq", "key" => "author", "value" => "test")
    t = FileSearchTool(vector_store_ids=["vs_1"], max_num_results=5, ranking_options=ranking, filters=filters)
    lowered = JSON.lower(t)
    @test lowered[:ranking_options] == ranking
    @test lowered[:filters] == filters
end

@testset "WebSearchTool with location" begin
    loc = Dict("country" => "DE", "city" => "Berlin")
    t = WebSearchTool(user_location=loc)
    lowered = JSON.lower(t)
    @test lowered[:user_location] == loc
end

@testset "FunctionTool with strict and parameters" begin
    params = Dict("type" => "object", "properties" => Dict())
    t = FunctionTool(name="fn", parameters=params, strict=true)
    lowered = JSON.lower(t)
    @test lowered[:parameters] == params
    @test lowered[:strict] == true
end

@testset "function_tool shorthand with kwargs" begin
    params = Dict("type" => "object")
    t = function_tool("fn", "desc"; parameters=params, strict=true)
    @test t.parameters == params
    @test t.strict == true
end

@testset "web_search shorthand with location" begin
    loc = Dict("country" => "US")
    t = web_search(context_size="high", location=loc)
    @test t.search_context_size == "high"
    @test t.user_location == loc
end

@testset "file_search shorthand with all options" begin
    ranking = Dict("ranker" => "auto")
    filters = Dict("type" => "eq")
    t = file_search(["vs_1"]; max_results=10, ranking=ranking, filters=filters)
    @test t.max_num_results == 10
    @test t.ranking_options == ranking
    @test t.filters == filters
end

@testset "input_file with both url and id" begin
    p = input_file(url="https://example.com/f.pdf", id="file-123")
    @test p[:file_url] == "https://example.com/f.pdf"
    @test p[:file_id] == "file-123"
end

@testset "Respond JSON serialization with all fields" begin
    r = Respond(
        input="test",
        instructions="inst",
        tools=[function_tool("fn")],
        tool_choice="auto",
        parallel_tool_calls=true,
        temperature=0.5,
        max_output_tokens=100,
        stream=false,
        text=text_format(),
        reasoning=Reasoning(effort="high"),
        truncation="auto",
        store=true,
        metadata=Dict("k" => "v"),
        previous_response_id="resp_prev",
        user="user_123"
    )
    lowered = JSON.lower(r)
    @test lowered[:instructions] == "inst"
    @test lowered[:tool_choice] == "auto"
    @test lowered[:parallel_tool_calls] == true
    @test lowered[:max_output_tokens] == 100
    @test lowered[:stream] == false
    @test haskey(lowered, :text)
    @test haskey(lowered, :reasoning)
    @test lowered[:truncation] == "auto"
    @test lowered[:store] == true
    @test lowered[:metadata] == Dict("k" => "v")
    @test lowered[:previous_response_id] == "resp_prev"
    @test lowered[:user] == "user_123"
end

# ─── New Respond fields (API coverage) ────────────────────────────────────────

@testset "Respond new fields" begin
    @testset "background field" begin
        r = Respond(input="test", background=true)
        @test r.background == true
        lowered = JSON.lower(r)
        @test lowered[:background] == true
    end

    @testset "background defaults to nothing" begin
        r = Respond(input="test")
        @test isnothing(r.background)
        lowered = JSON.lower(r)
        @test !haskey(lowered, :background)
    end

    @testset "include field" begin
        r = Respond(input="test", include=["file_search_call.results", "message.output_text.logprobs"])
        @test r.include == ["file_search_call.results", "message.output_text.logprobs"]
        lowered = JSON.lower(r)
        @test lowered[:include] == ["file_search_call.results", "message.output_text.logprobs"]
    end

    @testset "max_tool_calls field" begin
        r = Respond(input="test", max_tool_calls=Int64(10))
        @test r.max_tool_calls == 10
        lowered = JSON.lower(r)
        @test lowered[:max_tool_calls] == 10
    end

    @testset "service_tier field" begin
        r = Respond(input="test", service_tier="flex")
        @test r.service_tier == "flex"
        lowered = JSON.lower(r)
        @test lowered[:service_tier] == "flex"
    end

    @testset "top_logprobs field" begin
        r = Respond(input="test", top_logprobs=Int64(5))
        @test r.top_logprobs == 5
        lowered = JSON.lower(r)
        @test lowered[:top_logprobs] == 5
    end

    @testset "prompt field" begin
        p = Dict("id" => "prompt_abc", "version" => "1")
        r = Respond(input="test", prompt=p)
        @test r.prompt == p
        lowered = JSON.lower(r)
        @test lowered[:prompt] == p
    end

    @testset "prompt_cache_key field" begin
        r = Respond(input="test", prompt_cache_key="cache_key_123")
        @test r.prompt_cache_key == "cache_key_123"
        lowered = JSON.lower(r)
        @test lowered[:prompt_cache_key] == "cache_key_123"
    end

    @testset "prompt_cache_retention field" begin
        r = Respond(input="test", prompt_cache_retention="24h")
        @test r.prompt_cache_retention == "24h"
        lowered = JSON.lower(r)
        @test lowered[:prompt_cache_retention] == "24h"
    end

    @testset "safety_identifier field" begin
        r = Respond(input="test", safety_identifier="user_hash_abc")
        @test r.safety_identifier == "user_hash_abc"
        lowered = JSON.lower(r)
        @test lowered[:safety_identifier] == "user_hash_abc"
    end

    @testset "conversation field (string)" begin
        r = Respond(input="test", conversation="conv_abc")
        @test r.conversation == "conv_abc"
        lowered = JSON.lower(r)
        @test lowered[:conversation] == "conv_abc"
    end

    @testset "context_management field" begin
        cm = [Dict("type" => "truncation", "compact_threshold" => 0.8)]
        r = Respond(input="test", context_management=cm)
        @test r.context_management == cm
        lowered = JSON.lower(r)
        @test lowered[:context_management] == cm
    end

    @testset "stream_options field" begin
        so = Dict("include_obfuscation" => true)
        r = Respond(input="test", stream=true, stream_options=so)
        @test r.stream_options == so
        lowered = JSON.lower(r)
        @test lowered[:stream_options] == so
    end

    @testset "all new fields in JSON round-trip" begin
        r = Respond(
            input="test",
            background=true,
            include=["file_search_call.results"],
            max_tool_calls=Int64(5),
            service_tier="auto",
            top_logprobs=Int64(3),
            prompt=Dict("id" => "p1"),
            prompt_cache_key="key",
            prompt_cache_retention="in-memory",
            safety_identifier="safe_123",
            conversation="conv_1",
            context_management=[Dict("type" => "truncation")],
            stream_options=Dict("include_obfuscation" => false)
        )
        json_str = JSON.json(r)
        parsed = JSON.parse(json_str)

        @test parsed["background"] == true
        @test parsed["include"] == ["file_search_call.results"]
        @test parsed["max_tool_calls"] == 5
        @test parsed["service_tier"] == "auto"
        @test parsed["top_logprobs"] == 3
        @test parsed["prompt"]["id"] == "p1"
        @test parsed["prompt_cache_key"] == "key"
        @test parsed["prompt_cache_retention"] == "in-memory"
        @test parsed["safety_identifier"] == "safe_123"
        @test parsed["conversation"] == "conv_1"
        @test parsed["context_management"][1]["type"] == "truncation"
        @test parsed["stream_options"]["include_obfuscation"] == false
    end
end

@testset "Reasoning generate_summary" begin
    @testset "creation with generate_summary" begin
        r = Reasoning(effort="high", generate_summary="concise")
        @test r.effort == "high"
        @test r.generate_summary == "concise"
        @test isnothing(r.summary)
    end

    @testset "JSON serialization includes generate_summary" begin
        r = Reasoning(effort="medium", generate_summary="detailed")
        lowered = JSON.lower(r)
        @test lowered[:effort] == "medium"
        @test lowered[:generate_summary] == "detailed"
        @test !haskey(lowered, :summary)
    end

    @testset "both generate_summary and summary" begin
        r = Reasoning(effort="low", generate_summary="auto", summary="concise")
        lowered = JSON.lower(r)
        @test lowered[:generate_summary] == "auto"
        @test lowered[:summary] == "concise"
    end
end

@testset "cancel_response error handling" begin
    @testset "with invalid API key" begin
        withenv("OPENAI_API_KEY" => "sk-invalid-test-key") do
            result = cancel_response("resp_nonexistent")
            @test result isa ResponseCallError || result isa ResponseFailure
        end
    end
end

@testset "compact_response error handling" begin
    @testset "with invalid API key" begin
        withenv("OPENAI_API_KEY" => "sk-invalid-test-key") do
            result = compact_response(model="gpt-5.2", input="test")
            @test result isa ResponseCallError || result isa ResponseFailure
        end
    end
end

@testset "count_input_tokens error handling" begin
    @testset "with invalid API key" begin
        withenv("OPENAI_API_KEY" => "sk-invalid-test-key") do
            result = count_input_tokens(model="gpt-5.2", input="test")
            @test result isa ResponseCallError || result isa ResponseFailure
        end
    end

    @testset "with instructions and tools" begin
        withenv("OPENAI_API_KEY" => "sk-invalid-test-key") do
            result = count_input_tokens(
                model="gpt-5.2",
                input="test",
                instructions="Be helpful",
                tools=[function_tool("fn")]
            )
            @test result isa ResponseCallError || result isa ResponseFailure
        end
    end
end

# ─── Expanded coverage: Respond field edge cases ──────────────────────────────

@testset "Respond with top_p only (no temperature)" begin
    r = Respond(input="test", top_p=0.9)
    @test r.top_p == 0.9
    @test isnothing(r.temperature)
    lowered = JSON.lower(r)
    @test lowered[:top_p] == 0.9
    @test !haskey(lowered, :temperature)
end

@testset "Respond with user field" begin
    r = Respond(input="test", user="user_abc123")
    @test r.user == "user_abc123"
    lowered = JSON.lower(r)
    @test lowered[:user] == "user_abc123"
end

@testset "Respond with stream=true" begin
    r = Respond(input="test", stream=true)
    @test r.stream == true
    lowered = JSON.lower(r)
    @test lowered[:stream] == true
end

@testset "Respond with conversation as Dict" begin
    conv = Dict("type" => "persistent", "id" => "conv_abc")
    r = Respond(input="test", conversation=conv)
    @test r.conversation == conv
    lowered = JSON.lower(r)
    @test lowered[:conversation] == conv
end

@testset "Respond edge cases: zero values" begin
    r = Respond(input="test", top_logprobs=Int64(0), max_tool_calls=Int64(0))
    @test r.top_logprobs == 0
    @test r.max_tool_calls == 0
    lowered = JSON.lower(r)
    @test lowered[:top_logprobs] == 0
    @test lowered[:max_tool_calls] == 0
end

@testset "Respond with prompt_cache_retention in-memory" begin
    r = Respond(input="test", prompt_cache_retention="in-memory")
    @test r.prompt_cache_retention == "in-memory"
    lowered = JSON.lower(r)
    @test lowered[:prompt_cache_retention] == "in-memory"
end

@testset "Respond with all service_tier values" begin
    for tier in ["auto", "default", "flex", "priority"]
        r = Respond(input="test", service_tier=tier)
        @test r.service_tier == tier
        lowered = JSON.lower(r)
        @test lowered[:service_tier] == tier
    end
end

@testset "Respond with all fields combined (old + new)" begin
    r = Respond(
        input=[InputMessage(role="user", content="test")],
        model="gpt-5.2",
        instructions="Be helpful",
        tools=[function_tool("fn"), web_search()],
        tool_choice="auto",
        parallel_tool_calls=true,
        temperature=0.5,
        max_output_tokens=200,
        stream=false,
        text=json_object_format(),
        reasoning=Reasoning(effort="high", generate_summary="concise"),
        truncation="auto",
        store=true,
        metadata=Dict("env" => "test"),
        previous_response_id="resp_prev",
        user="user_123",
        background=false,
        include=["file_search_call.results"],
        max_tool_calls=Int64(10),
        service_tier="flex",
        top_logprobs=Int64(5),
        prompt=Dict("id" => "prompt_1"),
        prompt_cache_key="cache_1",
        prompt_cache_retention="24h",
        safety_identifier="safe_1",
        conversation="conv_1",
        context_management=[Dict("type" => "truncation")],
        stream_options=Dict("include_usage" => true)
    )
    lowered = JSON.lower(r)
    @test lowered[:model] == "gpt-5.2"
    @test lowered[:instructions] == "Be helpful"
    @test lowered[:tool_choice] == "auto"
    @test lowered[:parallel_tool_calls] == true
    @test lowered[:temperature] == 0.5
    @test lowered[:max_output_tokens] == 200
    @test lowered[:stream] == false
    @test haskey(lowered, :text)
    @test haskey(lowered, :reasoning)
    @test lowered[:truncation] == "auto"
    @test lowered[:store] == true
    @test lowered[:metadata] == Dict("env" => "test")
    @test lowered[:previous_response_id] == "resp_prev"
    @test lowered[:user] == "user_123"
    @test lowered[:background] == false
    @test lowered[:include] == ["file_search_call.results"]
    @test lowered[:max_tool_calls] == 10
    @test lowered[:service_tier] == "flex"
    @test lowered[:top_logprobs] == 5
    @test lowered[:prompt] == Dict("id" => "prompt_1")
    @test lowered[:prompt_cache_key] == "cache_1"
    @test lowered[:prompt_cache_retention] == "24h"
    @test lowered[:safety_identifier] == "safe_1"
    @test lowered[:conversation] == "conv_1"
    @test lowered[:context_management] == [Dict("type" => "truncation")]
    @test lowered[:stream_options] == Dict("include_usage" => true)

    # No service in JSON
    @test !haskey(lowered, :service)

    # JSON round-trip
    json_str = JSON.json(r)
    parsed = JSON.parse(json_str)
    @test parsed["background"] == false
    @test parsed["service_tier"] == "flex"
    @test parsed["safety_identifier"] == "safe_1"
    @test parsed["prompt_cache_retention"] == "24h"
    @test parsed["reasoning"]["generate_summary"] == "concise"
end

# ─── Expanded coverage: TextConfig ────────────────────────────────────────────

@testset "TextConfig JSON serialization" begin
    @testset "text_format JSON round-trip" begin
        tc = text_format()
        json_str = JSON.json(tc)
        parsed = JSON.parse(json_str)
        @test parsed["format"]["type"] == "text"
        @test !haskey(parsed["format"], "name")
    end

    @testset "json_object_format JSON round-trip" begin
        tc = json_object_format()
        json_str = JSON.json(tc)
        parsed = JSON.parse(json_str)
        @test parsed["format"]["type"] == "json_object"
    end

    @testset "json_schema_format with strict" begin
        schema = Dict("type" => "object", "properties" => Dict("x" => Dict("type" => "string")))
        tc = json_schema_format("my_schema", "desc", schema, strict=true)
        @test tc.format.strict == true
        json_str = JSON.json(tc)
        parsed = JSON.parse(json_str)
        @test parsed["format"]["strict"] == true
        @test parsed["format"]["name"] == "my_schema"
        @test parsed["format"]["schema"]["type"] == "object"
    end
end

# ─── Expanded coverage: Reasoning edge cases ─────────────────────────────────

@testset "Reasoning edge cases" begin
    @testset "only generate_summary (no effort)" begin
        r = Reasoning(generate_summary="detailed")
        @test r.generate_summary == "detailed"
        @test isnothing(r.effort)
        lowered = JSON.lower(r)
        @test lowered[:generate_summary] == "detailed"
        @test !haskey(lowered, :effort)
    end

    @testset "all effort values" begin
        for effort in ["none", "low", "medium", "high"]
            r = Reasoning(effort=effort)
            @test r.effort == effort
        end
    end

    @testset "all generate_summary values" begin
        for gs in ["auto", "concise", "detailed"]
            r = Reasoning(generate_summary=gs)
            @test r.generate_summary == gs
        end
    end

    @testset "JSON round-trip with all fields" begin
        r = Reasoning(effort="high", generate_summary="concise", summary="auto")
        json_str = JSON.json(r)
        parsed = JSON.parse(json_str)
        @test parsed["effort"] == "high"
        @test parsed["generate_summary"] == "concise"
        @test parsed["summary"] == "auto"
    end
end

# ─── Expanded coverage: parse_response edge cases ────────────────────────────

@testset "parse_response edge cases" begin
    function make_response(body::Dict; status=200)
        body_bytes = Vector{UInt8}(JSON.json(body))
        HTTP.Response(status, [], body_bytes)
    end

    @testset "minimal response (no optional fields)" begin
        body = Dict(
            "id" => "resp_min",
            "status" => "completed",
            "model" => "gpt-5.2",
            "output" => []
        )
        ro = UniLM.parse_response(make_response(body))
        @test ro.id == "resp_min"
        @test ro.status == "completed"
        @test ro.model == "gpt-5.2"
        @test isempty(ro.output)
        @test isnothing(ro.usage)
        @test isnothing(ro.error)
        @test isnothing(ro.metadata)
    end

    @testset "response with error field" begin
        body = Dict(
            "id" => "resp_err",
            "status" => "failed",
            "model" => "gpt-5.2",
            "output" => [],
            "error" => Dict("code" => "rate_limit_exceeded", "message" => "Rate limited")
        )
        ro = UniLM.parse_response(make_response(body))
        @test ro.status == "failed"
        @test ro.error["code"] == "rate_limit_exceeded"
    end

    @testset "response with metadata" begin
        body = Dict(
            "id" => "resp_meta",
            "status" => "completed",
            "model" => "gpt-5.2",
            "output" => [],
            "metadata" => Dict("request_id" => "req_123", "env" => "test")
        )
        ro = UniLM.parse_response(make_response(body))
        @test ro.metadata["request_id"] == "req_123"
        @test ro.metadata["env"] == "test"
    end

    @testset "response with multiple output items" begin
        body = Dict(
            "id" => "resp_multi",
            "status" => "completed",
            "model" => "gpt-5.2",
            "output" => [
                Dict("type" => "function_call", "id" => "fc_1", "call_id" => "call_1",
                    "name" => "fn1", "arguments" => "{}", "status" => "completed"),
                Dict("type" => "function_call", "id" => "fc_2", "call_id" => "call_2",
                    "name" => "fn2", "arguments" => "{\"x\":1}", "status" => "completed"),
                Dict("type" => "message", "content" => [
                    Dict("type" => "output_text", "text" => "Done")
                ])
            ]
        )
        ro = UniLM.parse_response(make_response(body))
        @test length(ro.output) == 3
        calls = function_calls(ro)
        @test length(calls) == 2
        @test calls[1]["name"] == "fn1"
        @test calls[2]["name"] == "fn2"
        @test output_text(ro) == "Done"
    end
end

# ─── Expanded coverage: ResponseObject creation & accessors ──────────────────

@testset "ResponseObject usage accessor" begin
    ro = ResponseObject(
        id="resp_u", status="completed", model="gpt-5.2",
        output=Any[],
        usage=Dict{String,Any}("input_tokens" => 100, "output_tokens" => 50, "total_tokens" => 150),
        raw=Dict{String,Any}()
    )
    @test ro.usage["input_tokens"] == 100
    @test ro.usage["output_tokens"] == 50
    @test ro.usage["total_tokens"] == 150
end

@testset "ResponseObject with nil usage" begin
    ro = ResponseObject(
        id="resp_nu", status="completed", model="gpt-5.2",
        output=Any[], raw=Dict{String,Any}()
    )
    @test isnothing(ro.usage)
end

@testset "output_text with mixed content types" begin
    ro = ResponseObject(
        id="resp_mx", status="completed", model="gpt-5.2",
        output=Any[
            Dict{String,Any}(
            "type" => "message",
            "content" => Any[
                Dict{String,Any}("type" => "refusal", "refusal" => "Cannot comply"),
                Dict{String,Any}("type" => "output_text", "text" => "Allowed text"),
                Dict{String,Any}("type" => "annotation", "data" => "ignored")
            ]
        )
        ],
        raw=Dict{String,Any}()
    )
    @test output_text(ro) == "Allowed text"
end

@testset "function_calls with multiple calls" begin
    ro = ResponseObject(
        id="resp_fc", status="completed", model="gpt-5.2",
        output=Any[
            Dict{String,Any}("type" => "function_call", "id" => "fc_1",
                "call_id" => "c1", "name" => "fn_a", "arguments" => "{}", "status" => "completed"),
            Dict{String,Any}("type" => "function_call", "id" => "fc_2",
                "call_id" => "c2", "name" => "fn_b", "arguments" => "{\"x\":1}", "status" => "completed"),
            Dict{String,Any}("type" => "function_call", "id" => "fc_3",
                "call_id" => "c3", "name" => "fn_c", "arguments" => "{\"y\":2}", "status" => "completed")
        ],
        raw=Dict{String,Any}()
    )
    calls = function_calls(ro)
    @test length(calls) == 3
    @test calls[1]["name"] == "fn_a"
    @test calls[2]["name"] == "fn_b"
    @test calls[3]["name"] == "fn_c"
end

# ─── Expanded coverage: Result type accessors ────────────────────────────────

@testset "ResponseSuccess output_text and function_calls" begin
    ro = ResponseObject(
        id="resp_s", status="completed", model="gpt-5.2",
        output=Any[
            Dict{String,Any}("type" => "message", "content" => Any[
                Dict{String,Any}("type" => "output_text", "text" => "response text")
            ]),
            Dict{String,Any}("type" => "function_call", "id" => "fc_1",
                "call_id" => "c1", "name" => "fn_x", "arguments" => "{}", "status" => "completed")
        ],
        raw=Dict{String,Any}()
    )
    s = ResponseSuccess(response=ro)
    @test output_text(s) == "response text"
    @test length(function_calls(s)) == 1
    @test function_calls(s)[1]["name"] == "fn_x"
end

@testset "ResponseFailure output_text" begin
    f = ResponseFailure(response="bad request body", status=422)
    @test output_text(f) == "Error (HTTP 422): bad request body"
end

@testset "ResponseCallError output_text" begin
    e = ResponseCallError(error="connection refused", status=nothing)
    @test output_text(e) == "Error: connection refused"
    @test isnothing(e.status)
end

# ─── Expanded coverage: InputMessage variations ──────────────────────────────

@testset "InputMessage with file content" begin
    m = InputMessage(role="user", content=[
        input_text("Analyze this file:"),
        input_file(url="https://example.com/data.csv")
    ])
    @test length(m.content) == 2
    @test m.content[1][:type] == "input_text"
    @test m.content[2][:type] == "input_file"
    @test m.content[2][:file_url] == "https://example.com/data.csv"
end

@testset "InputMessage with image detail" begin
    m = InputMessage(role="user", content=[
        input_text("What's in this image?"),
        input_image("https://example.com/photo.jpg", detail="low")
    ])
    @test m.content[2][:detail] == "low"
end

@testset "InputMessage JSON round-trip with multimodal" begin
    m = InputMessage(role="user", content=[
        input_text("Describe:"),
        input_image("https://example.com/img.png", detail="high")
    ])
    json_str = JSON.json(m)
    parsed = JSON.parse(json_str)
    @test parsed["role"] == "user"
    @test length(parsed["content"]) == 2
    @test parsed["content"][1]["type"] == "input_text"
    @test parsed["content"][2]["type"] == "input_image"
    @test parsed["content"][2]["detail"] == "high"
end

@testset "InputMessage with all roles" begin
    for role in ["user", "assistant", "system", "developer"]
        m = InputMessage(role=role, content="test")
        @test m.role == role
        lowered = JSON.lower(m)
        @test lowered[:role] == role
    end
end

# ─── Expanded coverage: Tool serialization round-trips ───────────────────────

@testset "FunctionTool JSON round-trip" begin
    params = Dict(
        "type" => "object",
        "properties" => Dict("q" => Dict("type" => "string", "description" => "Query")),
        "required" => ["q"],
        "additionalProperties" => false
    )
    t = FunctionTool(name="search", description="Search tool", parameters=params, strict=true)
    json_str = JSON.json(t)
    parsed = JSON.parse(json_str)
    @test parsed["type"] == "function"
    @test parsed["name"] == "search"
    @test parsed["description"] == "Search tool"
    @test parsed["strict"] == true
    @test parsed["parameters"]["required"] == ["q"]
end

@testset "WebSearchTool JSON round-trip" begin
    loc = Dict("country" => "US", "city" => "San Francisco", "timezone" => "America/Los_Angeles")
    t = WebSearchTool(search_context_size="high", user_location=loc)
    json_str = JSON.json(t)
    parsed = JSON.parse(json_str)
    @test parsed["type"] == "web_search_preview"
    @test parsed["search_context_size"] == "high"
    @test parsed["user_location"]["city"] == "San Francisco"
end

@testset "FileSearchTool JSON round-trip" begin
    ranking = Dict("ranker" => "auto", "score_threshold" => 0.5)
    filters = Dict("type" => "eq", "key" => "category", "value" => "science")
    t = FileSearchTool(
        vector_store_ids=["vs_1", "vs_2"],
        max_num_results=25,
        ranking_options=ranking,
        filters=filters
    )
    json_str = JSON.json(t)
    parsed = JSON.parse(json_str)
    @test parsed["type"] == "file_search"
    @test length(parsed["vector_store_ids"]) == 2
    @test parsed["max_num_results"] == 25
    @test parsed["ranking_options"]["score_threshold"] == 0.5
    @test parsed["filters"]["key"] == "category"
end

# ─── Expanded coverage: Respond with explicit model override ─────────────────

@testset "Respond with explicit model override" begin
    r = Respond(input="test", model="o3-mini")
    @test r.model == "o3-mini"
    lowered = JSON.lower(r)
    @test lowered[:model] == "o3-mini"
end

@testset "Respond convenience with new kwargs" begin
    withenv("OPENAI_API_KEY" => "sk-invalid-test-key") do
        result = respond("Hello",
            background=true,
            service_tier="flex",
            include=["file_search_call.results"]
        )
        @test result isa ResponseCallError || result isa ResponseFailure
    end
end

@testset "compact_response with default model" begin
    withenv("OPENAI_API_KEY" => "sk-invalid-test-key") do
        result = compact_response(input="test")
        @test result isa ResponseCallError || result isa ResponseFailure
    end
end

@testset "count_input_tokens with default model" begin
    withenv("OPENAI_API_KEY" => "sk-invalid-test-key") do
        result = count_input_tokens(input="test")
        @test result isa ResponseCallError || result isa ResponseFailure
    end
end

# ─── Expanded coverage: _parse_response_stream_chunk edge cases ──────────────

@testset "_parse_response_stream_chunk edge cases" begin
    @testset "multiple deltas in one chunk" begin
        chunk = """event: response.output_text.delta
data: {"type":"response.output_text.delta","delta":"Hello"}
event: response.output_text.delta
data: {"type":"response.output_text.delta","delta":" World"}"""
        textbuff = IOBuffer()
        failbuff = IOBuffer()
        result = UniLM._parse_response_stream_chunk(chunk, textbuff, failbuff)
        @test result.done == false
        @test String(take!(textbuff)) == "Hello World"
    end

    @testset "non-text event type (ignored)" begin
        chunk = """event: response.created
data: {"type":"response.created","response":{"id":"resp_1"}}"""
        textbuff = IOBuffer()
        failbuff = IOBuffer()
        result = UniLM._parse_response_stream_chunk(chunk, textbuff, failbuff)
        @test result.done == false
        @test isempty(take!(textbuff))
    end

    @testset "whitespace-only chunk" begin
        textbuff = IOBuffer()
        failbuff = IOBuffer()
        result = UniLM._parse_response_stream_chunk("  \n  \n  ", textbuff, failbuff)
        @test result.done == false
    end
end

# ─── Phase 3B: Respond parameter validation ──────────────────────────────────

@testset "Respond parameter validation" begin
    @testset "temperature out of range" begin
        @test_throws ArgumentError Respond(input="test", temperature=-0.1)
        @test_throws ArgumentError Respond(input="test", temperature=2.1)
        @test_throws ArgumentError Respond(input="test", temperature=3.0)
    end

    @testset "temperature boundary values accepted" begin
        @test Respond(input="test", temperature=0.0).temperature == 0.0
        @test Respond(input="test", temperature=2.0).temperature == 2.0
        @test Respond(input="test", temperature=1.0).temperature == 1.0
    end

    @testset "top_p out of range" begin
        @test_throws ArgumentError Respond(input="test", top_p=-0.1)
        @test_throws ArgumentError Respond(input="test", top_p=1.1)
        @test_throws ArgumentError Respond(input="test", top_p=2.0)
    end

    @testset "top_p boundary values accepted" begin
        @test Respond(input="test", top_p=0.0).top_p == 0.0
        @test Respond(input="test", top_p=1.0).top_p == 1.0
        @test Respond(input="test", top_p=0.5).top_p == 0.5
    end

    @testset "max_output_tokens out of range" begin
        @test_throws ArgumentError Respond(input="test", max_output_tokens=Int64(0))
        @test_throws ArgumentError Respond(input="test", max_output_tokens=Int64(-1))
        @test_throws ArgumentError Respond(input="test", max_output_tokens=Int64(-100))
    end

    @testset "max_output_tokens boundary values accepted" begin
        @test Respond(input="test", max_output_tokens=Int64(1)).max_output_tokens == 1
        @test Respond(input="test", max_output_tokens=Int64(100)).max_output_tokens == 100
    end

    @testset "top_logprobs out of range" begin
        @test_throws ArgumentError Respond(input="test", top_logprobs=Int64(-1))
        @test_throws ArgumentError Respond(input="test", top_logprobs=Int64(21))
        @test_throws ArgumentError Respond(input="test", top_logprobs=Int64(100))
    end

    @testset "top_logprobs boundary values accepted" begin
        @test Respond(input="test", top_logprobs=Int64(0)).top_logprobs == 0
        @test Respond(input="test", top_logprobs=Int64(20)).top_logprobs == 20
        @test Respond(input="test", top_logprobs=Int64(10)).top_logprobs == 10
    end

    @testset "nothing values still accepted" begin
        r = Respond(input="test")
        @test isnothing(r.temperature)
        @test isnothing(r.top_p)
        @test isnothing(r.max_output_tokens)
        @test isnothing(r.top_logprobs)
    end
end

# ─── Phase 3A: Respond.input type tightening ─────────────────────────────────

@testset "Respond.input type" begin
    @testset "String input accepted" begin
        r = Respond(input="Hello")
        @test r.input == "Hello"
    end

    @testset "Vector{InputMessage} accepted" begin
        msgs = [InputMessage(role="user", content="Hello")]
        r = Respond(input=msgs)
        @test r.input == msgs
    end

    @testset "Vector{Dict} accepted" begin
        dicts = [Dict("role" => "user", "content" => "Hello")]
        r = Respond(input=dicts)
        @test r.input == dicts
    end

    @testset "Mixed Vector{Any} accepted" begin
        mixed = Any[InputMessage(role="user", content="Hi"), Dict("role" => "assistant", "content" => "Hey")]
        r = Respond(input=mixed)
        @test r.input == mixed
    end

    @testset "non-String non-Vector rejected" begin
        @test_throws MethodError Respond(input=42)
        @test_throws MethodError Respond(input=nothing)
        @test_throws MethodError Respond(input=3.14)
    end
end
