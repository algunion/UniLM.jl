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
        @test r.model == "gpt-4.1"
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
            model="gpt-4.1-mini",
            instructions="Be helpful",
            temperature=0.5,
            max_output_tokens=100,
            store=true,
            metadata=Dict("key" => "value"),
            truncation="auto"
        )
        @test r.model == "gpt-4.1-mini"
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
        @test lowered[:model] == "gpt-4.1"
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

        @test parsed["model"] == "gpt-4.1"
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
        "model" => "gpt-4.1",
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
        model="gpt-4.1",
        output=raw["output"],
        usage=raw["usage"],
        raw=raw
    )

    @test ro.id == "resp_123"
    @test ro.status == "completed"
    @test ro.model == "gpt-4.1"
    @test length(ro.output) == 1
end

@testset "output_text" begin
    ro = ResponseObject(
        id="resp_1",
        status="completed",
        model="gpt-4.1",
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
        model="gpt-4.1",
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
        id="resp_1", status="completed", model="gpt-4.1",
        output=Any[], raw=Dict{String,Any}()
    )
    @test output_text(ro) == ""
end

@testset "function_calls" begin
    ro = ResponseObject(
        id="resp_1",
        status="completed",
        model="gpt-4.1",
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
        id="resp_1", status="completed", model="gpt-4.1",
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
            id="resp_1", status="completed", model="gpt-4.1",
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
            "model" => "gpt-4.1",
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
        @test ro.model == "gpt-4.1"
        @test output_text(ro) == "Hello!"
        @test ro.usage["total_tokens"] == 15
    end

    @testset "function call response" begin
        body = Dict(
            "id" => "resp_fn",
            "object" => "response",
            "status" => "completed",
            "model" => "gpt-4.1",
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
                "model" => "gpt-4.1",
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
