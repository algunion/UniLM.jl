import InteractiveUtils

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

@testset "FunctionSignature" begin
    @testset "minimal creation" begin
        sig = FunctionSignature(name="test_fn")
        @test sig.name == "test_fn"
        @test isnothing(sig.description)
        @test isnothing(sig.parameters)
    end

    @testset "full creation" begin
        params = Dict("type" => "object", "properties" => Dict("x" => Dict("type" => "string")))
        sig = FunctionSignature(name="test_fn", description="A test", parameters=params)
        @test sig.name == "test_fn"
        @test sig.description == "A test"
        @test sig.parameters == params
    end

    @testset "JSON.jl config" begin
        @test JSON.omit_null(FunctionSignature) == true
    end

    @testset "serialization omit_null" begin
        sig = FunctionSignature(name="fn")
        json = JSON.json(sig)
        parsed = JSON.parse(json)
        @test parsed["name"] == "fn"
        @test !haskey(parsed, "description")
        @test !haskey(parsed, "parameters")
    end

    @testset "strict field" begin
        sig = FunctionSignature(name="fn")
        @test sig.strict === nothing

        @test FunctionSignature(name="fn", strict=true).strict === true
        @test FunctionSignature(name="fn", strict=false).strict === false
    end

    @testset "positional constructor back-compat" begin
        # pre-0.10.3 3-arg arity must keep working (patch release, non-breaking)
        sig = FunctionSignature("fn", "desc", Dict("type" => "object"))
        @test sig.name == "fn"
        @test sig.description == "desc"
        @test sig.strict === nothing
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

@testset "ToolCall" begin
    func = UniLM.GPTFunction("test_fn", Dict("a" => "b"))
    tc = ToolCall(id="call_123", func=func)
    @test tc.id == "call_123"
    @test tc.type == "function"
    @test tc.func.name == "test_fn"

    lowered = JSON.lower(tc)
    @test haskey(lowered, :function)
    @test !haskey(lowered, :func)
    @test lowered[:id] == "call_123"
    @test lowered[:type] == "function"
end

@testset "Tool" begin
    sig = FunctionSignature(name="my_tool")
    tool = Tool(func=sig)
    @test tool.type == "function"
    @test tool.func.name == "my_tool"

    lowered = JSON.lower(tool)
    @test haskey(lowered, :function)
    @test !haskey(lowered, :func)
    @test lowered[:type] == "function"

    @testset "from bare dict" begin
        d = Dict("name" => "bare_fn", "description" => "A bare function",
            "parameters" => Dict("type" => "object"))
        t = Tool(d)
        @test t.type == "function"
        @test t.func.name == "bare_fn"
        @test t.func.description == "A bare function"
        @test t.func.parameters == Dict("type" => "object")
    end

    @testset "from wrapped dict" begin
        d = Dict("type" => "function", "function" => Dict(
            "name" => "wrapped_fn", "description" => "Wrapped",
            "parameters" => Dict("type" => "object", "properties" => Dict())))
        t = Tool(d)
        @test t.type == "function"
        @test t.func.name == "wrapped_fn"
        @test t.func.description == "Wrapped"
    end

    @testset "strict serialization (F1-F3)" begin
        params = Dict("type" => "object",
            "properties" => Dict("x" => Dict("type" => "string")),
            "required" => ["x"], "additionalProperties" => false)

        # F1: strict=true is transmitted INSIDE the "function" object of the
        # actual request body (Chat Completions nesting, not Responses top-level)
        tool = Tool(func=FunctionSignature(name="fn", description="d",
            parameters=params, strict=true))
        chat = Chat(model="gpt-test", tools=[tool])
        body = JSON.parse(JSON.json(chat))
        @test body["tools"][1]["function"]["strict"] === true
        @test !haskey(body["tools"][1], "strict")

        # F2: default (no strict) → no strict key anywhere; the function object's
        # key SET matches pre-strict UniLM (structural pin — JSON dict key order
        # is not deterministic, so a byte-level pin would be brittle)
        tool_default = Tool(func=FunctionSignature(name="fn", description="d",
            parameters=params))
        parsed = JSON.parse(JSON.json(tool_default))
        @test Set(keys(parsed["function"])) == Set(["name", "description", "parameters"])
        @test !haskey(parsed, "strict")

        # F3: strict=false is transmitted explicitly
        tool_false = Tool(func=FunctionSignature(name="fn", strict=false))
        @test JSON.parse(JSON.json(tool_false))["function"]["strict"] === false
    end

    @testset "strict parse-back (F4)" begin
        # wrapped OpenAI format: strict inside "function"
        d = Dict("type" => "function", "function" => Dict(
            "name" => "fn", "strict" => true,
            "parameters" => Dict("type" => "object")))
        @test Tool(d).func.strict === true

        # bare format
        @test Tool(Dict("name" => "fn", "strict" => false)).func.strict === false

        # absent → nothing (API default, non-strict)
        @test Tool(Dict("name" => "fn")).func.strict === nothing

        # malformed non-Bool strict fails loud with a diagnostic error
        @test_throws ArgumentError Tool(Dict("name" => "fn", "strict" => "true"))
        @test_throws ArgumentError Tool(Dict("type" => "function",
            "function" => Dict("name" => "fn", "strict" => 1)))

        # round-trip: serialize → parse → reconstruct preserves strict
        for s in (true, false)
            t = Tool(func=FunctionSignature(name="rt", strict=s))
            @test Tool(JSON.parse(JSON.json(t))).func.strict === s
        end
        t = Tool(func=FunctionSignature(name="rt"))
        @test Tool(JSON.parse(JSON.json(t))).func.strict === nothing
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

@testset "FunctionCallResult" begin
    func = UniLM.GPTFunction("test_fn", Dict("x" => "1"))
    fcr = FunctionCallResult("test_fn", func, "result_value")
    @test fcr.name == "test_fn"
    @test fcr.origincall === func
    @test fcr.result == "result_value"

    @test JSON.omit_null(FunctionCallResult{String}) == true
    @test JSON.omit_empty(FunctionCallResult{String}) == true

    @testset "serialization honors omit_empty" begin
        # Observable consequence of api.jl:142 (omit_empty=true): an EMPTY result field is dropped,
        # a populated one is kept. Falsifies a regression where omit_empty stopped applying.
        fcr_empty = FunctionCallResult("get_weather", func, String[])
        parsed_empty = JSON.parse(JSON.json(fcr_empty))
        @test parsed_empty["name"] == "get_weather"
        @test !haskey(parsed_empty, "result")           # empty vector omitted
        fcr_val = FunctionCallResult("get_weather", func, "sunny")
        parsed_val = JSON.parse(JSON.json(fcr_val))
        @test parsed_val["result"] == "sunny"            # non-empty kept
    end
end

@testset "legacy GPT* aliases construct, dispatch, and field-access (back-compat pin)" begin
    # Pins the const-alias contract: pre-rename user code that constructs via the
    # GPT* names, dispatches with `isa`, and reads fields must keep working after
    # the structs were given provider-neutral canonical names. Silent aliases
    # (no deprecation warning) until 1.0.

    # Each alias binds to exactly the canonical type object.
    @test GPTTool === Tool
    @test GPTToolCall === ToolCall
    @test GPTFunctionSignature === FunctionSignature
    @test GPTFunctionCallResult === FunctionCallResult

    # Construct via the old names (keyword ctor + nested old name).
    tool = GPTTool(func=GPTFunctionSignature(name="f"))
    @test tool isa Tool                       # canonical type
    @test tool isa GPTTool                     # and the alias binding
    @test tool.func.name == "f"

    # Dict constructor reached through the alias.
    @test GPTTool(Dict("name" => "g")).func.name == "g"

    # ToolCall via the old name: isa + field access.
    tc = GPTToolCall(id="c1", func=UniLM.GPTFunction("f", Dict{String,Any}("a" => 1)))
    @test tc isa ToolCall
    @test tc isa GPTToolCall
    @test tc.id == "c1"

    # The parametric alias must stay a UnionAll: `GPTFunctionCallResult{Int}`
    # constructs positionally exactly like the canonical parametric type.
    fcr = GPTFunctionCallResult{Int}("fn", UniLM.GPTFunction("fn", Dict{String,Any}()), 42)
    @test fcr isa FunctionCallResult{Int}
    @test fcr isa GPTFunctionCallResult{Int}
    @test fcr.result === 42
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
        tc = ToolCall(id="call_1", func=func)
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

    @testset "provider-native content side-channel" begin
        blocks = Any[Dict{String,Any}("type" => "thinking",
                                      "thinking" => "check the weather",
                                      "signature" => "sig==")]
        pc = ProviderContent(:anthropic, blocks)
        @test pc.provider === :anthropic
        @test pc.blocks === blocks

        # Default: absent.
        m0 = Message(role=UniLM.RoleAssistant, content="hi")
        @test isnothing(m0.provider_content)

        # Wire isolation: provider_content NEVER serializes — byte-identical JSON.
        # (GPTFunction is unexported; this file qualifies it everywhere.)
        tc = [ToolCall(id="c1", func=UniLM.GPTFunction("f", Dict{String,Any}("a" => 1)))]
        m_plain = Message(role=UniLM.RoleAssistant, content="ok", tool_calls=tc)
        m_pc = Message(role=UniLM.RoleAssistant, content="ok", tool_calls=tc,
                       provider_content=pc)
        @test JSON.json(m_plain) == JSON.json(m_pc)
        @test !occursin("provider_content", JSON.json(m_pc))

        # All of today's wire fields still serialize (omit-null preserved).
        full = Message(role=UniLM.RoleTool, content="r", name="n",
                       finish_reason="stop", tool_call_id="c1")
        parsed = JSON.parse(JSON.json(full))
        @test parsed["role"] == "tool" && parsed["content"] == "r" &&
              parsed["name"] == "n" && parsed["finish_reason"] == "stop" &&
              parsed["tool_call_id"] == "c1"
        @test !haskey(parsed, "tool_calls") && !haskey(parsed, "refusal_message")

        # Chat-level: no leak through the full request body either.
        chat = Chat(model="gpt-5.5")
        push!(chat, Message(Val(:system), "s"))
        push!(chat, Message(Val(:user), "u"))
        push!(chat, m_pc)
        body = JSON.parse(JSON.json(chat))
        @test length(body["messages"]) == 3   # the assistant turn actually landed
        @test all(msg -> !haskey(msg, "provider_content"), body["messages"])

        # Positional compatibility: the 7-arg positional form still constructs
        # (kwdef defaults do not extend positional constructors).
        m7 = Message(UniLM.RoleAssistant, "hi", nothing, nothing, nothing, nothing, nothing)
        @test m7.content == "hi" && isnothing(m7.provider_content)

        # Validation still enforced with the new field present.
        @test_throws ArgumentError Message(role=UniLM.RoleAssistant, provider_content=pc)
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

    @testset "strict" begin
        # default: nothing, omitted from the wire (existing bodies unchanged)
        # (the omit_null line itself is a known Julia coverage quirk: bare
        # `= true` method bodies never instrument — value asserted here)
        @test JSON.omit_null(UniLM.JsonSchemaAPI) == true
        @test jsa.strict === nothing
        @test !haskey(parsed, "strict")

        js = UniLM.JsonSchemaAPI(name="n", description="d", schema=schema, strict=true)
        @test JSON.parse(JSON.json(js))["strict"] === true

        # full response_format body: strict inside the "json_schema" object
        rf = UniLM.json_schema("n", "d", schema; strict=true)
        prf = JSON.parse(JSON.json(rf))
        @test prf["type"] == "json_schema"
        @test prf["json_schema"]["strict"] === true

        # helper without strict stays strict-free
        rf0 = UniLM.json_schema("n", "d", schema)
        @test !haskey(JSON.parse(JSON.json(rf0))["json_schema"], "strict")
    end

    @testset "positional constructor back-compat" begin
        # pre-0.10.3 3-arg arity must keep working (patch release, non-breaking)
        js = UniLM.JsonSchemaAPI("n", "d", schema)
        @test js.name == "n"
        @test js.strict === nothing
    end
end

@testset "ServiceEndpoint types" begin
    @test UniLM.OPENAIServiceEndpoint <: UniLM.ServiceEndpoint
    @test UniLM.AZUREServiceEndpoint <: UniLM.ServiceEndpoint
    @test UniLM.GEMINIOpenAIServiceEndpoint <: UniLM.ServiceEndpoint

    @testset "DeepSeekEndpoint keyword constructor" begin
        # api.jl:379 — the kwarg ctor. Passing api_key explicitly means the ENV["DEEPSEEK_API_KEY"]
        # default is never evaluated (zero-spend), and the key must round-trip into the struct.
        ds = DeepSeekEndpoint(api_key="explicit-key")
        @test ds isa UniLM.DeepSeekEndpoint
        @test ds.api_key == "explicit-key"
    end
end

@testset "Chat" begin
    @testset "default creation" begin
        chat = Chat()
        @test chat.model == "gpt-5.6-sol"
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
        sig = FunctionSignature(name="fn")
        chat = Chat(tools=[Tool(func=sig)], parallel_tool_calls=true)
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

        # Consecutive same-role rejected — throws, conversation left unchanged
        @test_throws InvalidConversationError push!(chat, usr)
        @test length(chat) == 2  # unchanged

        # System not allowed after conversation started — throws, unchanged
        @test_throws InvalidConversationError push!(chat, sys)
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
        tc1 = ToolCall(id="call_1", func=func)
        tc2 = ToolCall(id="call_2", func=func)
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

        # Pop from empty throws (fail loud, not a silent no-op)
        @test_throws InvalidConversationError pop!(chat)
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

    @testset "update! without history is a documented no-op (not a caller error)" begin
        chat = Chat(history=false)
        push!(chat, Message(role=UniLM.RoleSystem, content="sys"))
        push!(chat, Message(role=UniLM.RoleUser, content="q"))
        asst = Message(role=UniLM.RoleAssistant, content="a")
        @test update!(chat, asst) === chat   # returns the chat; does NOT throw
        @test length(chat) == 2              # unchanged because history=false
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
        @test parsed["model"] == "gpt-5.6-sol"
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

    @testset "call errors carry a typed cause" begin
        root = ArgumentError("root failure")
        e = LLMCallError(error="wrapped", self=chat, cause=root)
        @test e.cause === root
        @test isnothing(LLMCallError(error="x", self=chat).cause)   # additive default

        ec = EmbeddingCallError(error="net", cause=root)
        @test ec.cause === root
        @test isnothing(EmbeddingCallError(error="net").cause)
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

    @testset "dimensions / encoding_format / resize" begin
        emb = UniLM.Embeddings("x"; model="text-embedding-3-large", dimensions=256, encoding_format="float", user="u1")
        lowered = JSON.lower(emb)
        @test lowered[:dimensions] == 256
        @test lowered[:encoding_format] == "float"
        @test lowered[:user] == "u1"
        @test length(emb.embeddings) == 256          # buffer pre-sized to the requested dimension
        # defaults omit both; buffer falls back to 1536
        e2 = UniLM.Embeddings("x"; model="text-embedding-3-small")
        @test !haskey(JSON.lower(e2), :dimensions)
        @test !haskey(JSON.lower(e2), :encoding_format)
        @test length(e2.embeddings) == 1536
        # update! tolerates a 3072-dim response even when the buffer started at 1536
        e3 = UniLM.Embeddings("x"; model="text-embedding-3-large")
        update!(e3, [Dict{String,Any}("index" => 0, "embedding" => collect(1.0:3072.0))])
        @test length(e3.embeddings) == 3072
        @test e3.embeddings[3072] == 3072.0
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

    @testset "update! rejects a response that does not cover every input" begin
        # The buffers are pre-zeroed, so an uncovered slot stays an all-zero vector
        # that still normalizes, compares and indexes — a corrupt embedding no caller
        # can distinguish from a real one. Any response that does not fill each slot
        # exactly once is an error.
        row(i, v) = Dict{String,Any}("index" => i, "embedding" => fill(v, 4))

        # Fewer rows than inputs (the silent zero-vector case).
        @test_throws ArgumentError update!(UniLM.Embeddings(["a", "b", "c"]), [row(0, 1.0), row(1, 2.0)])
        # More rows than inputs, and a duplicated or out-of-range index.
        @test_throws ArgumentError update!(UniLM.Embeddings(["a", "b"]),
            [row(0, 1.0), row(1, 2.0), row(1, 3.0)])
        @test_throws ArgumentError update!(UniLM.Embeddings(["a", "b"]), [row(0, 1.0), row(0, 2.0)])
        @test_throws ArgumentError update!(UniLM.Embeddings(["a", "b"]), [row(0, 1.0), row(7, 2.0)])
        @test_throws ArgumentError update!(UniLM.Embeddings(["a", "b"]), [row(0, 1.0), row(-1, 2.0)])
        # Single-input requests take the same contract: exactly one row.
        @test_throws ArgumentError update!(UniLM.Embeddings("a"), Dict{String,Any}[])
        @test_throws ArgumentError update!(UniLM.Embeddings("a"), [row(0, 1.0), row(1, 2.0)])

        # The message names the mismatch rather than failing on an index later.
        err = try; update!(UniLM.Embeddings(["a", "b", "c"]), [row(0, 1.0), row(1, 2.0)]); catch e; e; end
        @test contains(sprint(showerror, err), "2 rows for 3 inputs")

        # A complete response still fills every slot, in any row order.
        ok = UniLM.Embeddings(["a", "b", "c"])
        update!(ok, [row(2, 3.0), row(0, 1.0), row(1, 2.0)])
        @test [v[1] for v in ok.embeddings] == [1.0, 2.0, 3.0]
        @test all(v -> length(v) == 4, ok.embeddings)
    end

end

@testset "Chat JSON serialization - all optional fields" begin
    sig = FunctionSignature(name="fn")
    chat = Chat(
        model="gpt-4o",
        temperature=0.7,
        tools=[Tool(func=sig)],
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

@testset "Chat max_completion_tokens" begin
    c = Chat(model="gpt-5.5", max_completion_tokens=64)
    l = JSON.lower(c)
    @test l[:max_completion_tokens] == 64
    @test !haskey(l, :max_tokens)
    # legacy max_tokens still serializes independently
    @test JSON.lower(Chat(model="gpt-5.5", max_tokens=64))[:max_tokens] == 64
end

@testset "Chat parity params (Phase C)" begin
    c = Chat(model="gpt-5.5",
        reasoning_effort="high", stream_options=Dict("include_usage" => true), verbosity="low",
        store=true, metadata=Dict("k" => "v"), service_tier="flex", logprobs=true, top_logprobs=3,
        prediction=Dict("type" => "content", "content" => "x"), modalities=["text"],
        audio=Dict("voice" => "alloy", "format" => "mp3"),
        web_search_options=Dict("search_context_size" => "low"),
        prompt_cache_key="ck", safety_identifier="sid")
    l = JSON.lower(c)
    @test l[:reasoning_effort] == "high"
    @test l[:stream_options]["include_usage"] == true
    @test l[:verbosity] == "low"
    @test l[:store] == true
    @test l[:metadata]["k"] == "v"
    @test l[:service_tier] == "flex"
    @test l[:logprobs] == true
    @test l[:top_logprobs] == 3
    @test l[:modalities] == ["text"]
    @test l[:audio]["voice"] == "alloy"
    @test l[:web_search_options]["search_context_size"] == "low"
    @test l[:prompt_cache_key] == "ck"
    @test l[:safety_identifier] == "sid"
    @test haskey(l, :prediction)
    # all omitted when unset
    l2 = JSON.lower(Chat(model="gpt-5.5"))
    @test !haskey(l2, :reasoning_effort) && !haskey(l2, :store) && !haskey(l2, :audio)
end

@testset "Chat with different service endpoints" begin
    @testset "Azure endpoint" begin
        chat = Chat(service=UniLM.AZUREServiceEndpoint, model="gpt-4o")
        @test chat.service == UniLM.AZUREServiceEndpoint
    end

    @testset "Gemini endpoint" begin
        chat = Chat(service=UniLM.GEMINIOpenAIServiceEndpoint, model="gemini-2.0-flash")
        @test chat.service == UniLM.GEMINIOpenAIServiceEndpoint
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

@testset "ToolCall.thought_signature (Gemini-3 opaque echo)" begin
    tc = ToolCall(id="fc_1", func=UniLM.GPTFunction("f", Dict("x" => 1)))
    @test isnothing(tc.thought_signature)                      # optional, defaults nothing
    tc2 = ToolCall(id="fc_2", func=UniLM.GPTFunction("f", Dict()), thought_signature="SIG")
    @test tc2.thought_signature == "SIG"
    # MUST NOT leak into OpenAI wire serialization:
    lowered = JSON.lower(tc2)
    @test !haskey(lowered, :thoughtSignature) && !haskey(lowered, :thought_signature)
    @test Set(keys(lowered)) == Set([:id, :type, :function])
end

@testset "Chat ctor accepts a CallableTool vector (stores unwrapped Tools)" begin
    # Ergonomics: `Chat(tools=mcp_tools(session))` must work without a manual
    # `map(t -> t.tool, tools)`. The keyword accepts a CallableTool vector and
    # the constructor unwraps each wrapper's inner Tool; the field stays
    # `Vector{Tool}`.
    sig = FunctionSignature(name="lookup", description="Look something up",
        parameters=Dict{String,Any}("type" => "object", "properties" => Dict{String,Any}()))
    ct = CallableTool(Tool(func=sig), (name, args) -> "ok")
    chat = Chat(model="gpt-test", tools=[ct])
    @test chat.tools isa Vector{Tool}
    @test length(chat.tools) == 1
    @test chat.tools[1] === ct.tool           # the exact inner Tool, unwrapped
    @test chat.tools[1].func.name == "lookup"
    # The unwrapped vector is non-empty, so parallel_tool_calls is preserved.
    chat2 = Chat(model="gpt-test", tools=[ct], parallel_tool_calls=true)
    @test chat2.parallel_tool_calls == true
    # A plain Tool vector still passes through unchanged.
    gtool = Tool(func=sig)
    @test Chat(model="gpt-test", tools=[gtool]).tools[1] === gtool
end

@testset "fail-loud conversation mutations (throw, not warn)" begin
    sys  = Message(role=UniLM.RoleSystem, content="s")
    usr  = Message(role=UniLM.RoleUser, content="u")
    asst = Message(role=UniLM.RoleAssistant, content="a")

    @testset "empty conversation must start with a system message" begin
        chat = Chat()
        @test_throws InvalidConversationError push!(chat, usr)
        @test_throws InvalidConversationError push!(chat, asst)
        @test isempty(chat)                     # refused inputs never mutate the chat
    end

    @testset "a system message is only valid as the first message" begin
        chat = Chat()
        push!(chat, sys); push!(chat, usr)
        @test_throws InvalidConversationError push!(chat, sys)
        @test length(chat) == 2
    end

    @testset "no consecutive same-role (non-tool) messages" begin
        chat = Chat()
        push!(chat, sys); push!(chat, usr)
        @test_throws InvalidConversationError push!(chat, usr)
    end

    @testset "pop! on empty throws" begin
        @test_throws InvalidConversationError pop!(Chat())
    end

    @testset "thrown message names the rule + role, leaks no content/endpoint/key" begin
        chat = Chat(service=GenericOpenAIEndpoint("https://api.example.com", "sk-live-secret123"),
                    model="mock")
        err = try
            push!(chat, Message(role=UniLM.RoleUser, content="TOP-SECRET-PROMPT"))
            nothing
        catch e
            e
        end
        @test err isa InvalidConversationError
        shown = sprint(showerror, err)
        @test occursin("system", shown)             # states the violated rule
        @test occursin("user", shown)               # names the offending role
        @test !occursin("TOP-SECRET-PROMPT", shown) # never the message content
        @test !occursin("secret123", shown)         # never the api key
        @test !occursin("api.example.com", shown)   # never the endpoint
    end
end

@testset "endpoint api_key redaction in show" begin
    ep = GenericOpenAIEndpoint("https://api.example.com", "sk-live-secret123")
    s = sprint(show, ep)
    @test occursin("[redacted]", s)
    @test !occursin("secret123", s)
    @test !occursin("sk-live-secret123", s)
    @test occursin("https://api.example.com", s)      # non-secret fields still render

    ds = DeepSeekEndpoint(api_key="ds-live-secret999")
    sd = sprint(show, ds)
    @test occursin("[redacted]", sd)
    @test !occursin("secret999", sd)

    # empty key (local no-auth server) is not a secret: no marker, no crash
    @test !occursin("[redacted]", sprint(show, OllamaEndpoint()))

    # nested: endpoint inside a Chat inside an LLMFailure must not leak the raw key
    chat = Chat(service=ep, model="mock",
                messages=[Message(Val(:system), "s"), Message(Val(:user), "u")])
    fail = LLMFailure(response="upstream body", status=500, self=chat)
    sf = sprint(show, fail)
    @test occursin("[redacted]", sf)
    @test !occursin("secret123", sf)
    @test !occursin("sk-live-secret123", sf)
end

@testset "result consumption sugar (issuccess/isfailure/text/LLMResultError)" begin
    chat = Chat(model="gpt-5.5",
                messages=[Message(Val(:system), "s"), Message(Val(:user), "u")])
    okmsg   = Message(role=UniLM.RoleAssistant, content="hello", finish_reason="stop")
    ok      = LLMSuccess(message=okmsg, self=chat)
    fail    = LLMFailure(response="boom", status=500, self=chat)
    callerr = LLMCallError(error="network down", self=chat)

    @testset "issuccess / isfailure" begin
        @test issuccess(ok) === true
        @test isfailure(ok) === false
        @test issuccess(fail) === false
        @test isfailure(fail) === true
        @test issuccess(callerr) === false
        @test isfailure(callerr) === true
    end

    @testset "issuccess spot-checks across result families (incl. platform API)" begin
        @test issuccess(EmbeddingSuccess(embeddings=UniLM.Embeddings("x"), raw=Dict{String,Any}())) === true
        @test issuccess(FileDeleteSuccess(id="f", deleted=true)) === true   # a platform-API success
        @test isfailure(EmbeddingFailure(response="e", status=400)) === true
    end

    @testset "text() returns content on success; nothing for tool-calls-only" begin
        @test text(ok) == "hello"
        tc = ToolCall(id="c1", func=UniLM.GPTFunction("fn", Dict("a" => "b")))
        toolmsg = Message(role=UniLM.RoleAssistant, tool_calls=[tc], finish_reason=UniLM.TOOL_CALLS)
        @test text(LLMSuccess(message=toolmsg, self=chat)) === nothing
    end

    @testset "text() on failure/callerror throws a typed LLMResultError" begin
        @test_throws LLMResultError text(fail)
        @test_throws LLMResultError text(callerr)
    end

    @testset "LLMResultError.showerror shows status + trimmed excerpt, never chat/service/key" begin
        chat2 = Chat(service=GenericOpenAIEndpoint("https://api.example.com", "sk-live-secret123"),
                     model="mock",
                     messages=[Message(Val(:system), "s"), Message(Val(:user), "u")])
        bigbody = "E" * repeat("x", 500)
        f2 = LLMFailure(response=bigbody, status=503, self=chat2)
        err = try text(f2); catch e; e; end
        @test err isa LLMResultError
        se = sprint(showerror, err)
        @test occursin("503", se)                # status shown
        @test occursin("…", se)                  # body was trimmed (500 chars > 200)
        @test !occursin(bigbody, se)             # full body not present
        @test !occursin("secret123", se)         # never the api key
        @test !occursin("api.example.com", se)   # never the endpoint
        @test length(se) < 350                   # a short excerpt, not the whole payload
    end

    @testset "every LLMRequestResponse subtype is classified by its *Success name" begin
        fallback = which(issuccess, Tuple{LLMRequestResponse})
        n_success = 0
        for T in InteractiveUtils.subtypes(LLMRequestResponse)
            isconcretetype(T) || continue
            if endswith(String(nameof(T)), "Success")
                @test which(issuccess, Tuple{T}) !== fallback   # an explicit `= true` method exists
                n_success += 1
            else
                @test which(issuccess, Tuple{T}) === fallback   # falls through to the `false` default
            end
        end
        @test n_success >= 30    # ~34 success types across chat/embeddings/platform APIs
    end
end
# Stand-in for a transport wrapper whose default show dumps the request it carries
# (the HTTP.jl 1.x RequestError shape). Top level: structs cannot live in a testset.
struct _CauseDumpError <: Exception; dump::String; end

@testset "call-error shows name the cause instead of dumping it" begin
    # `cause` keeps the raw exception on purpose — callers dispatch on it. Julia's
    # default show recurses into it, though, so a transport wrapper that renders as
    # a request dump (HTTP.jl 1.x) put the credential back into the printed result
    # after `.error` had already been redacted.
    token = "sk-ant-SECRETVALUE0123456789"
    dumping = _CauseDumpError("HTTP.Request:\nPOST /v1/messages\r\nx-api-key: $token\r\n\r\n{}")
    chat = Chat(model="gpt-5.5", messages=[Message(Val(:system), "s"), Message(Val(:user), "u")])
    results = (LLMCallError(error="transport failed", self=chat, status=nothing, cause=dumping),
               EmbeddingCallError(error="transport failed", cause=dumping),
               ResponseCallError(error="transport failed", request_id="req_9", cause=dumping),
               FIMCallError(error="transport failed", cause=dumping))
    for r in results
        s = sprint(show, r)
        @test !occursin(token, s)
        @test !occursin("SECRETVALUE", s)
        @test occursin("transport failed", s)        # the redacted message still shows
        @test occursin("_CauseDumpError", s)           # the cause is named by TYPE
        @test r.cause === dumping                    # …and still reachable
    end
    @test occursin("request_id=\"req_9\"", sprint(show, results[3]))
    # No cause: the field renders as nothing, not as an empty type name.
    @test occursin("cause=nothing", sprint(show, LLMCallError(error="x", self=chat)))
end
