@testset "Constants" begin
    @test UniLM.RoleSystem == "system"
    @test UniLM.RoleUser == "user"
    @test UniLM.RoleAssistant == "assistant"
    @test UniLM.RoleTool == "tool"
    @test UniLM.STOP == "stop"
    @test UniLM.CONTENT_FILTER == "content_filter"
    @test UniLM.TOOL_CALLS == "tool_calls"
    @test UniLM.OPENAI_BASE_URL == "https://api.openai.com"

    @testset "Endpoint path constants" begin
        @test UniLM.CHAT_COMPLETIONS_PATH == "/v1/chat/completions"
        @test UniLM.EMBEDDINGS_PATH == "/v1/embeddings"
        @test UniLM.RESPONSES_PATH == "/v1/responses"
        @test UniLM.GEMINI_CHAT_URL == "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
    end

    @testset "API key env var name constants" begin
        @test UniLM.OPENAI_API_KEY == "OPENAI_API_KEY"
        @test UniLM.AZURE_OPENAI_API_KEY == "AZURE_OPENAI_API_KEY"
        @test UniLM.AZURE_OPENAI_BASE_URL == "AZURE_OPENAI_BASE_URL"
        @test UniLM.AZURE_OPENAI_API_VERSION == "AZURE_OPENAI_API_VERSION"
        @test UniLM.GEMINI_API_KEY == "GEMINI_API_KEY"
    end

    @testset "Azure deployment mapping is a Dict" begin
        @test UniLM._MODEL_ENDPOINTS_AZURE_OPENAI isa Dict{String,String}
    end
end

@testset "Model type" begin
    m = UniLM.Model("gpt-4o")
    @test string(m) == "gpt-4o"
    @test Base.parse(UniLM.Model, "gpt-4o") == UniLM.Model("gpt-4o")

    @test UniLM.GPT5_2 == UniLM.Model("gpt-5.2")
    @test UniLM.GPTTextEmbedding3Small == UniLM.Model("text-embedding-3-small")
end

@testset "InvalidConversationError" begin
    e = InvalidConversationError("test reason")
    @test e isa Exception
    @test e.reason == "test reason"
end

@testset "GPTFunctionSignature" begin
    @testset "minimal creation" begin
        sig = GPTFunctionSignature(name="test_fn")
        @test sig.name == "test_fn"
        @test isnothing(sig.description)
        @test isnothing(sig.parameters)
    end

    @testset "full creation" begin
        params = Dict("type" => "object", "properties" => Dict("x" => Dict("type" => "string")))
        sig = GPTFunctionSignature(name="test_fn", description="A test", parameters=params)
        @test sig.name == "test_fn"
        @test sig.description == "A test"
        @test sig.parameters == params
    end

    @testset "JSON.jl config" begin
        @test JSON.omit_null(GPTFunctionSignature) == true
    end

    @testset "serialization omit_null" begin
        sig = GPTFunctionSignature(name="fn")
        json = JSON.json(sig)
        parsed = JSON.parse(json)
        @test parsed["name"] == "fn"
        @test !haskey(parsed, "description")
        @test !haskey(parsed, "parameters")
    end
end

@testset "GPTImageContent" begin
    ic = UniLM.GPTImageContent("hello", ["http://img1.png", "http://img2.png"])
    @test ic.text == "hello"
    @test length(ic.images) == 2

    lowered = JSON.lower(ic)
    @test length(lowered) == 3
    @test lowered[1][:type] == "text"
    @test lowered[1][:text] == "hello"
    @test lowered[2][:type] == "image_url"
    @test lowered[2][:image_url][:url] == "http://img1.png"
    @test lowered[2][:image_url][:detail] == "auto"
    @test lowered[3][:image_url][:url] == "http://img2.png"
end

@testset "GPTFunction" begin
    args = Dict("location" => "NYC", "count" => 3)
    f = UniLM.GPTFunction("get_weather", args)
    @test f.name == "get_weather"
    @test f.arguments["location"] == "NYC"
    @test f.arguments["count"] == 3  # non-string values allowed

    lowered = JSON.lower(f)
    @test lowered[:name] == "get_weather"
    @test lowered[:arguments] isa String  # arguments serialized to JSON string
    parsed_args = JSON.parse(lowered[:arguments])
    @test parsed_args["location"] == "NYC"
end

@testset "GPTToolCall" begin
    func = UniLM.GPTFunction("test_fn", Dict("a" => "b"))
    tc = GPTToolCall(id="call_123", func=func)
    @test tc.id == "call_123"
    @test tc.type == "function"
    @test tc.func.name == "test_fn"

    lowered = JSON.lower(tc)
    @test haskey(lowered, :function)
    @test !haskey(lowered, :func)
    @test lowered[:id] == "call_123"
    @test lowered[:type] == "function"
end

@testset "GPTTool" begin
    sig = GPTFunctionSignature(name="my_tool")
    tool = GPTTool(func=sig)
    @test tool.type == "function"
    @test tool.func.name == "my_tool"

    lowered = JSON.lower(tool)
    @test haskey(lowered, :function)
    @test !haskey(lowered, :func)
    @test lowered[:type] == "function"

    @testset "from bare dict" begin
        d = Dict("name" => "bare_fn", "description" => "A bare function",
            "parameters" => Dict("type" => "object"))
        t = GPTTool(d)
        @test t.type == "function"
        @test t.func.name == "bare_fn"
        @test t.func.description == "A bare function"
        @test t.func.parameters == Dict("type" => "object")
    end

    @testset "from wrapped dict" begin
        d = Dict("type" => "function", "function" => Dict(
            "name" => "wrapped_fn", "description" => "Wrapped",
            "parameters" => Dict("type" => "object", "properties" => Dict())))
        t = GPTTool(d)
        @test t.type == "function"
        @test t.func.name == "wrapped_fn"
        @test t.func.description == "Wrapped"
    end
end

@testset "GPTToolChoice" begin
    tc = UniLM.GPTToolChoice(func=:my_function)
    @test tc.type == "function"
    @test tc.func == :my_function

    lowered = JSON.lower(tc)
    @test lowered[:type] == "function"
    @test lowered[:function][:name] == :my_function
end

@testset "GPTFunctionCallResult" begin
    func = UniLM.GPTFunction("test_fn", Dict("x" => "1"))
    fcr = GPTFunctionCallResult("test_fn", func, "result_value")
    @test fcr.name == "test_fn"
    @test fcr.origincall === func
    @test fcr.result == "result_value"

    @test JSON.omit_null(GPTFunctionCallResult{String}) == true
    @test JSON.omit_empty(GPTFunctionCallResult{String}) == true
end

@testset "Message" begin
    @testset "basic creation" begin
        m = Message(role=UniLM.RoleUser, content="Hello")
        @test m.role == "user"
        @test m.content == "Hello"
        @test isnothing(m.name)
        @test isnothing(m.finish_reason)
        @test isnothing(m.refusal_message)
        @test isnothing(m.tool_calls)
        @test isnothing(m.tool_call_id)
    end

    @testset "Val constructors" begin
        sys = Message(Val(:system), "You are helpful.")
        @test sys.role == UniLM.RoleSystem
        @test sys.content == "You are helpful."

        usr = Message(Val(:user), "Hi")
        @test usr.role == UniLM.RoleUser
        @test usr.content == "Hi"
    end

    @testset "validation: content and tool_calls and refusal_message all nothing" begin
        @test_throws ArgumentError Message(role=UniLM.RoleUser)
    end

    @testset "validation: tool role requires tool_call_id" begin
        @test_throws ArgumentError Message(role=UniLM.RoleTool, content="result")
    end

    @testset "tool message with tool_call_id" begin
        m = Message(role=UniLM.RoleTool, content="result", tool_call_id="call_abc")
        @test m.role == UniLM.RoleTool
        @test m.content == "result"
        @test m.tool_call_id == "call_abc"
    end

    @testset "message with tool_calls" begin
        func = UniLM.GPTFunction("fn", Dict("a" => "b"))
        tc = GPTToolCall(id="call_1", func=func)
        m = Message(role=UniLM.RoleAssistant, tool_calls=[tc], finish_reason=UniLM.TOOL_CALLS)
        @test length(m.tool_calls) == 1
        @test isnothing(m.content)
    end

    @testset "message with refusal_message (content_filter)" begin
        m = Message(role=UniLM.RoleAssistant, refusal_message="Content filtered", finish_reason=UniLM.CONTENT_FILTER)
        @test m.refusal_message == "Content filtered"
        @test isnothing(m.content)
        @test isnothing(m.tool_calls)
    end

    @testset "helpers" begin
        m = Message(role=UniLM.RoleUser, content="test")
        @test UniLM.getcontent(m) == "test"
        @test UniLM.getrole(m) == "user"
        @test UniLM.iscall(m) == false

        tool_m = Message(role=UniLM.RoleTool, content="result", tool_call_id="call_x")
        @test UniLM.iscall(tool_m) == true
    end

    @testset "JSON serialization" begin
        @test JSON.omit_null(Message) == true

        m = Message(role=UniLM.RoleUser, content="hi")
        json = JSON.json(m)
        parsed = JSON.parse(json)
        @test parsed["role"] == "user"
        @test parsed["content"] == "hi"
        @test !haskey(parsed, "name")
        @test !haskey(parsed, "tool_calls")
    end
end

@testset "ResponseFormat" begin
    @testset "json_object default" begin
        rf = ResponseFormat()
        @test rf.type == "json_object"
        @test isnothing(rf.json_schema)
    end

    @testset "json_object via helper" begin
        rf = UniLM.json_object()
        @test rf.type == "json_object"
    end

    @testset "json_schema via helper" begin
        schema = Dict("type" => "object", "properties" => Dict("x" => Dict("type" => "string")))
        rf = UniLM.json_schema("test", "desc", schema)
        @test rf.type == "json_schema"
        @test rf.json_schema isa UniLM.JsonSchemaAPI
        @test rf.json_schema.name == "test"
        @test rf.json_schema.description == "desc"
        @test rf.json_schema.schema == schema
    end

    @testset "json_schema with dict" begin
        d = Dict("name" => "test", "schema" => Dict())
        rf = UniLM.json_schema(d)
        @test rf.type == "json_schema"
        @test rf.json_schema == d
    end

    @testset "ResponseFormat constructor with positional" begin
        jsa = UniLM.JsonSchemaAPI(name="n", description="d", schema=Dict("type" => "object"))
        rf = ResponseFormat(jsa)
        @test rf.type == "json_schema"
    end

    @testset "serialization omit_null" begin
        rf = UniLM.json_object()
        json = JSON.json(rf)
        parsed = JSON.parse(json)
        @test parsed["type"] == "json_object"
        @test !haskey(parsed, "json_schema")
    end
end

@testset "JsonSchemaAPI" begin
    schema = Dict("type" => "object")
    jsa = UniLM.JsonSchemaAPI(name="weather", description="Get weather", schema=schema)
    @test jsa.name == "weather"
    @test jsa.description == "Get weather"
    json = JSON.json(jsa)
    parsed = JSON.parse(json)
    @test parsed["name"] == "weather"
    @test parsed["description"] == "Get weather"
end

@testset "ServiceEndpoint types" begin
    @test UniLM.OPENAIServiceEndpoint <: UniLM.ServiceEndpoint
    @test UniLM.AZUREServiceEndpoint <: UniLM.ServiceEndpoint
    @test UniLM.GEMINIServiceEndpoint <: UniLM.ServiceEndpoint
end

@testset "Chat" begin
    @testset "default creation" begin
        chat = Chat()
        @test chat.model == "gpt-5.2"
        @test isempty(chat.messages)
        @test chat.history == true
        @test isnothing(chat.tools)
        @test isnothing(chat.tool_choice)
        @test isnothing(chat.parallel_tool_calls)  # nil because tools is nothing
        @test isnothing(chat.temperature)
        @test isnothing(chat.top_p)
        @test isnothing(chat.n)
        @test isnothing(chat.stream)
        @test isnothing(chat.stop)
        @test isnothing(chat.max_tokens)
        @test isnothing(chat.presence_penalty)
        @test isnothing(chat.response_format)
        @test isnothing(chat.frequency_penalty)
        @test isnothing(chat.logit_bias)
        @test isnothing(chat.user)
        @test isnothing(chat.seed)
        @test chat.service == UniLM.OPENAIServiceEndpoint
        @test chat._cumulative_cost[] == 0.0
    end

    @testset "custom creation" begin
        chat = Chat(model="gpt-4o-mini", temperature=0.5, max_tokens=100, seed=42)
        @test chat.model == "gpt-4o-mini"
        @test chat.temperature == 0.5
        @test chat.max_tokens == 100
        @test chat.seed == 42
    end

    @testset "temperature and top_p mutual exclusion" begin
        @test_throws ArgumentError Chat(temperature=0.2, top_p=0.5)
    end

    @testset "parallel_tool_calls nil when no tools" begin
        chat = Chat(parallel_tool_calls=true)
        @test isnothing(chat.parallel_tool_calls)  # reset to nothing since tools is nothing
    end

    @testset "parallel_tool_calls preserved with tools" begin
        sig = GPTFunctionSignature(name="fn")
        chat = Chat(tools=[GPTTool(func=sig)], parallel_tool_calls=true)
        @test chat.parallel_tool_calls == true
    end

    @testset "length and isempty" begin
        chat = Chat()
        @test length(chat) == 0
        @test isempty(chat)

        push!(chat, Message(role=UniLM.RoleSystem, content="sys"))
        @test length(chat) == 1
        @test !isempty(chat)
    end

    @testset "push! operations" begin
        chat = Chat()
        sys = Message(role=UniLM.RoleSystem, content="system prompt")
        usr = Message(role=UniLM.RoleUser, content="hello")

        # System as first message
        push!(chat, sys)
        @test length(chat) == 1
        @test chat.messages[1] == sys

        # User after system
        push!(chat, usr)
        @test length(chat) == 2

        # Consecutive same-role rejected
        push!(chat, usr)
        @test length(chat) == 2  # unchanged

        # System not allowed after conversation started
        push!(chat, sys)
        @test length(chat) == 2  # unchanged

        # Assistant after user
        asst = Message(role=UniLM.RoleAssistant, content="response")
        push!(chat, asst)
        @test length(chat) == 3
    end

    @testset "push! allows consecutive tool messages" begin
        chat = Chat()
        push!(chat, Message(role=UniLM.RoleSystem, content="sys"))
        push!(chat, Message(role=UniLM.RoleUser, content="q"))

        func = UniLM.GPTFunction("fn", Dict("a" => "b"))
        tc1 = GPTToolCall(id="call_1", func=func)
        tc2 = GPTToolCall(id="call_2", func=func)
        asst = Message(role=UniLM.RoleAssistant, tool_calls=[tc1, tc2], finish_reason=UniLM.TOOL_CALLS)
        push!(chat, asst)
        @test length(chat) == 3

        tool1 = Message(role=UniLM.RoleTool, content="result1", tool_call_id="call_1")
        tool2 = Message(role=UniLM.RoleTool, content="result2", tool_call_id="call_2")
        push!(chat, tool1)
        @test length(chat) == 4
        push!(chat, tool2)
        @test length(chat) == 5  # consecutive tool messages allowed
    end

    @testset "pop!" begin
        chat = Chat()
        push!(chat, Message(role=UniLM.RoleSystem, content="sys"))
        push!(chat, Message(role=UniLM.RoleUser, content="q"))
        @test length(chat) == 2

        pop!(chat)
        @test length(chat) == 1

        pop!(chat)
        @test length(chat) == 0

        # Pop from empty - should warn but not error
        pop!(chat)
        @test length(chat) == 0
    end

    @testset "last" begin
        chat = Chat()
        push!(chat, Message(role=UniLM.RoleSystem, content="sys"))
        push!(chat, Message(role=UniLM.RoleUser, content="q"))
        @test last(chat).content == "q"
    end

    @testset "getindex / setindex! / firstindex / lastindex" begin
        chat = Chat()
        push!(chat, Message(role=UniLM.RoleSystem, content="sys"))
        push!(chat, Message(role=UniLM.RoleUser, content="q"))

        @test chat[1].role == UniLM.RoleSystem
        @test chat[2].role == UniLM.RoleUser
        @test firstindex(chat) == 1
        @test lastindex(chat) == 2

        new_msg = Message(role=UniLM.RoleUser, content="new_q")
        chat[2] = new_msg
        @test chat[2].content == "new_q"
    end

    @testset "update! with history" begin
        chat = Chat(history=true)
        push!(chat, Message(role=UniLM.RoleSystem, content="sys"))
        push!(chat, Message(role=UniLM.RoleUser, content="q"))
        asst = Message(role=UniLM.RoleAssistant, content="a")
        update!(chat, asst)
        @test length(chat) == 3
        @test last(chat) == asst
    end

    @testset "update! without history" begin
        chat = Chat(history=false)
        push!(chat, Message(role=UniLM.RoleSystem, content="sys"))
        push!(chat, Message(role=UniLM.RoleUser, content="q"))
        asst = Message(role=UniLM.RoleAssistant, content="a")
        update!(chat, asst)
        @test length(chat) == 2  # unchanged because history=false
    end

    @testset "issendvalid" begin
        # Valid: system + user
        chat = Chat()
        push!(chat, Message(role=UniLM.RoleSystem, content="sys"))
        push!(chat, Message(role=UniLM.RoleUser, content="q"))
        @test issendvalid(chat) == true

        # Invalid: only system
        chat2 = Chat()
        push!(chat2, Message(role=UniLM.RoleSystem, content="sys"))
        @test issendvalid(chat2) == false

        # Invalid: empty
        @test issendvalid(Chat()) == false

        # Invalid: ends with system
        chat3 = Chat()
        push!(chat3, Message(role=UniLM.RoleSystem, content="sys"))
        push!(chat3, Message(role=UniLM.RoleUser, content="q"))
        chat3[end] = Message(role=UniLM.RoleSystem, content="sys2")
        @test issendvalid(chat3) == false

        # Valid: system + user + assistant + user
        chat4 = Chat()
        push!(chat4, Message(role=UniLM.RoleSystem, content="sys"))
        push!(chat4, Message(role=UniLM.RoleUser, content="q1"))
        push!(chat4, Message(role=UniLM.RoleAssistant, content="a1"))
        push!(chat4, Message(role=UniLM.RoleUser, content="q2"))
        @test issendvalid(chat4) == true

        # Invalid: starts with user
        chat5 = Chat()
        push!(chat5.messages, Message(role=UniLM.RoleUser, content="q"))
        push!(chat5.messages, Message(role=UniLM.RoleAssistant, content="a"))
        @test issendvalid(chat5) == false
    end

    @testset "JSON serialization" begin
        chat = Chat(temperature=0.7)
        push!(chat, Message(role=UniLM.RoleSystem, content="sys"))
        lowered = JSON.lower(chat)
        @test !haskey(lowered, :history)
        @test !haskey(lowered, :service)
        @test lowered[:temperature] == 0.7
        @test !haskey(lowered, :top_p)  # nothing fields omitted

        json = JSON.json(chat)
        parsed = JSON.parse(json)
        @test parsed["model"] == "gpt-5.2"
        @test parsed["temperature"] == 0.7
        @test !haskey(parsed, "history")
        @test !haskey(parsed, "service")
        @test !haskey(parsed, "top_p")
        @test !haskey(parsed, "_cumulative_cost")
    end
end

@testset "LLMRequestResponse types" begin
    chat = Chat()

    @testset "LLMSuccess" begin
        m = Message(role=UniLM.RoleAssistant, content="hello")
        s = LLMSuccess(message=m, self=chat)
        @test s isa UniLM.LLMRequestResponse
        @test s.message == m
        @test s.self === chat
        @test isnothing(s.usage)
    end

    @testset "LLMSuccess with usage" begin
        m = Message(role=UniLM.RoleAssistant, content="hello")
        u = TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        s = LLMSuccess(message=m, self=chat, usage=u)
        @test s.usage === u
        @test s.usage.prompt_tokens == 10
    end

    @testset "LLMFailure" begin
        f = LLMFailure(response="error", status=500, self=chat)
        @test f isa UniLM.LLMRequestResponse
        @test f.response == "error"
        @test f.status == 500
    end

    @testset "LLMCallError" begin
        e = LLMCallError(error="timeout", self=chat)
        @test e isa UniLM.LLMRequestResponse
        @test e.error == "timeout"
        @test isnothing(e.status)

        e2 = LLMCallError(error="err", status=503, self=chat)
        @test e2.status == 503
        @test e2.error == "err"
    end
end

@testset "Embeddings" begin
    @testset "String input" begin
        emb = UniLM.Embeddings("hello world")
        @test emb.model == "text-embedding-3-small"
        @test emb.input == "hello world"
        @test emb.embeddings isa Vector{Float64}
        @test length(emb.embeddings) == 1536
        @test all(x -> x == 0.0, emb.embeddings)
        @test isnothing(emb.user)
    end

    @testset "Vector{String} input" begin
        emb = UniLM.Embeddings(["hello", "world"])
        @test emb.input == ["hello", "world"]
        @test emb.embeddings isa Vector{Vector{Float64}}
        @test length(emb.embeddings) == 2
        @test length(emb.embeddings[1]) == 1536
    end

    @testset "empty Vector{String} error" begin
        @test_throws ArgumentError UniLM.Embeddings(String[])
    end

    @testset "JSON serialization" begin
        emb = UniLM.Embeddings("test")
        lowered = JSON.lower(emb)
        @test !haskey(lowered, :embeddings)
        @test !haskey(lowered, :user)
        @test lowered[:model] == "text-embedding-3-small"
        @test lowered[:input] == "test"

        json = JSON.json(emb)
        parsed = JSON.parse(json)
        @test parsed["model"] == "text-embedding-3-small"
        @test parsed["input"] == "test"
        @test !haskey(parsed, "embeddings")
        @test !haskey(parsed, "user")
    end

    @testset "update! single input" begin
        emb = UniLM.Embeddings("test")
        new_vals = rand(1536)
        data = [Dict{String,Any}("index" => 0, "embedding" => new_vals)]
        update!(emb, data)
        @test emb.embeddings ≈ new_vals
    end

    @testset "update! batch input" begin
        emb = UniLM.Embeddings(["hello", "world", "foo"])
        vecs = [rand(1536) for _ in 1:3]
        # Simulate API response with potentially out-of-order indices
        data = [
            Dict{String,Any}("index" => 2, "embedding" => vecs[3]),
            Dict{String,Any}("index" => 0, "embedding" => vecs[1]),
            Dict{String,Any}("index" => 1, "embedding" => vecs[2]),
        ]
        update!(emb, data)
        @test emb.embeddings[1] ≈ vecs[1]
        @test emb.embeddings[2] ≈ vecs[2]
        @test emb.embeddings[3] ≈ vecs[3]
    end
end

@testset "Chat JSON serialization - all optional fields" begin
    sig = GPTFunctionSignature(name="fn")
    chat = Chat(
        model="gpt-4o",
        temperature=0.7,
        tools=[GPTTool(func=sig)],
        tool_choice="auto",
        parallel_tool_calls=true,
        n=2,
        stream=true,
        stop=["END"],
        max_tokens=100,
        presence_penalty=0.5,
        response_format=ResponseFormat(),
        frequency_penalty=0.3,
        logit_bias=Dict("100" => 1.0),
        user="user_1",
        seed=42
    )
    lowered = JSON.lower(chat)
    @test lowered[:model] == "gpt-4o"
    @test lowered[:tools] isa Vector
    @test lowered[:tool_choice] == "auto"
    @test lowered[:parallel_tool_calls] == true
    @test lowered[:n] == 2
    @test lowered[:stream] == true
    @test lowered[:stop] == ["END"]
    @test lowered[:max_tokens] == 100
    @test lowered[:presence_penalty] == 0.5
    @test haskey(lowered, :response_format)
    @test lowered[:frequency_penalty] == 0.3
    @test lowered[:logit_bias] == Dict("100" => 1.0)
    @test lowered[:user] == "user_1"
    @test lowered[:seed] == 42
    # service and history not serialized
    @test !haskey(lowered, :service)
    @test !haskey(lowered, :history)
end

@testset "Chat with different service endpoints" begin
    @testset "Azure endpoint" begin
        chat = Chat(service=UniLM.AZUREServiceEndpoint, model="gpt-4o")
        @test chat.service == UniLM.AZUREServiceEndpoint
    end

    @testset "Gemini endpoint" begin
        chat = Chat(service=UniLM.GEMINIServiceEndpoint, model="gemini-2.0-flash")
        @test chat.service == UniLM.GEMINIServiceEndpoint
    end
end

@testset "Chat with top_p (no temperature)" begin
    chat = Chat(top_p=0.9)
    @test chat.top_p == 0.9
    @test isnothing(chat.temperature)
end

@testset "InvalidConversationError detailed" begin
    e = InvalidConversationError("bad conversation")
    @test sprint(showerror, e) == "InvalidConversationError(\"bad conversation\")"
end

@testset "Chat parameter validation" begin
    @testset "temperature out of range" begin
        @test_throws ArgumentError Chat(temperature=-0.1)
        @test_throws ArgumentError Chat(temperature=2.1)
        @test_throws ArgumentError Chat(temperature=3.0)
    end

    @testset "temperature boundary values accepted" begin
        @test Chat(temperature=0.0).temperature == 0.0
        @test Chat(temperature=2.0).temperature == 2.0
        @test Chat(temperature=1.0).temperature == 1.0
    end

    @testset "top_p out of range" begin
        @test_throws ArgumentError Chat(top_p=-0.1)
        @test_throws ArgumentError Chat(top_p=1.1)
        @test_throws ArgumentError Chat(top_p=2.0)
    end

    @testset "top_p boundary values accepted" begin
        @test Chat(top_p=0.0).top_p == 0.0
        @test Chat(top_p=1.0).top_p == 1.0
        @test Chat(top_p=0.5).top_p == 0.5
    end

    @testset "n out of range" begin
        @test_throws ArgumentError Chat(n=0)
        @test_throws ArgumentError Chat(n=-1)
        @test_throws ArgumentError Chat(n=11)
    end

    @testset "n boundary values accepted" begin
        @test Chat(n=1).n == 1
        @test Chat(n=10).n == 10
        @test Chat(n=5).n == 5
    end

    @testset "presence_penalty out of range" begin
        @test_throws ArgumentError Chat(presence_penalty=-2.1)
        @test_throws ArgumentError Chat(presence_penalty=2.1)
    end

    @testset "presence_penalty boundary values accepted" begin
        @test Chat(presence_penalty=-2.0).presence_penalty == -2.0
        @test Chat(presence_penalty=2.0).presence_penalty == 2.0
        @test Chat(presence_penalty=0.0).presence_penalty == 0.0
    end

    @testset "frequency_penalty out of range" begin
        @test_throws ArgumentError Chat(frequency_penalty=-2.1)
        @test_throws ArgumentError Chat(frequency_penalty=2.1)
    end

    @testset "frequency_penalty boundary values accepted" begin
        @test Chat(frequency_penalty=-2.0).frequency_penalty == -2.0
        @test Chat(frequency_penalty=2.0).frequency_penalty == 2.0
        @test Chat(frequency_penalty=0.0).frequency_penalty == 0.0
    end

    @testset "nothing values still accepted" begin
        chat = Chat()
        @test isnothing(chat.temperature)
        @test isnothing(chat.top_p)
        @test isnothing(chat.n)
        @test isnothing(chat.presence_penalty)
        @test isnothing(chat.frequency_penalty)
    end
end