# ─── CallableTool ────────────────────────────────────────────────────────────

@testset "CallableTool" begin
    @testset "construction from Tool" begin
        sig = FunctionSignature(name="add", description="Add numbers",
            parameters=Dict("type"=>"object", "properties"=>Dict("a"=>Dict("type"=>"number"))))
        tool = Tool(func=sig)
        ct = CallableTool(tool, (name, args) -> "42")
        @test ct.tool === tool
        @test ct.callable isa Function
        @test UniLM._tool_name(ct) == "add"
    end

    @testset "construction from FunctionTool" begin
        ft = FunctionTool(name="search", description="Search")
        ct = CallableTool(ft, (name, args) -> "found")
        @test ct.tool === ft
        @test UniLM._tool_name(ct) == "search"
    end

    @testset "JSON serialization delegates to inner Tool" begin
        sig = FunctionSignature(name="fn1", description="desc")
        tool = Tool(func=sig)
        ct = CallableTool(tool, (n, a) -> "ok")

        lowered_tool = JSON.lower(tool)
        lowered_ct = JSON.lower(ct)
        @test lowered_ct == lowered_tool
        @test lowered_ct[:type] == "function"
        @test lowered_ct[:function] isa FunctionSignature
    end

    @testset "JSON serialization delegates to inner FunctionTool" begin
        ft = FunctionTool(name="fn2", description="desc2")
        ct = CallableTool(ft, (n, a) -> "ok")

        lowered_ft = JSON.lower(ft)
        lowered_ct = JSON.lower(ct)
        @test lowered_ct == lowered_ft
        @test lowered_ct[:type] == "function"
        @test lowered_ct[:name] == "fn2"
    end
end

# ─── to_tool ─────────────────────────────────────────────────────────────────

@testset "to_tool" begin
    @testset "identity for Tool" begin
        tool = Tool(func=FunctionSignature(name="t"))
        @test to_tool(tool) === tool
    end

    @testset "identity for FunctionTool" begin
        ft = FunctionTool(name="t")
        @test to_tool(ft) === ft
    end

    @testset "identity for CallableTool" begin
        tool = Tool(func=FunctionSignature(name="t"))
        ct = CallableTool(tool, (n, a) -> "")
        @test to_tool(ct) === ct
    end

    @testset "dict conversion to Tool (bare)" begin
        d = Dict("name" => "myfn", "description" => "a fn",
            "parameters" => Dict("type" => "object"))
        result = to_tool(d)
        @test result isa Tool
        @test result.func.name == "myfn"
        @test result.func.description == "a fn"
    end

    @testset "dict conversion to Tool (wrapped)" begin
        d = Dict("type" => "function", "function" => Dict(
            "name" => "wrapped_fn", "description" => "wrapped"))
        result = to_tool(d)
        @test result isa Tool
        @test result.func.name == "wrapped_fn"
    end
end

# ─── _dispatch_tool ──────────────────────────────────────────────────────────

@testset "_dispatch_tool" begin
    @testset "successful dispatch" begin
        outcome = UniLM._dispatch_tool("add", Dict{String,Any}("a" => 3, "b" => 5),
            (name, args) -> string(args["a"] + args["b"]))
        @test outcome.success
        @test outcome.tool_name == "add"
        @test outcome.arguments == Dict{String,Any}("a" => 3, "b" => 5)
        @test !isnothing(outcome.result)
        @test outcome.result.name == "add"
        @test outcome.result.result == "8"
        @test outcome.result.origincall.name == "add"
        @test isnothing(outcome.error)
    end

    @testset "dispatch exception" begin
        outcome = UniLM._dispatch_tool("bad", Dict{String,Any}(),
            (n, a) -> error("boom"))
        @test !outcome.success
        @test outcome.tool_name == "bad"
        @test isnothing(outcome.result)
        @test contains(outcome.error, "boom")
    end

    @testset "result is stringified" begin
        outcome = UniLM._dispatch_tool("num", Dict{String,Any}(),
            (n, a) -> 42)
        @test outcome.success
        @test outcome.result.result == "42"
    end
end

# ─── Faithful tool-error messages (no exception-constructor noise) ────────────

@testset "tool errors reach the model unwrapped" begin
    @testset "ErrorException stores its raw message, not the constructor form" begin
        # `error("kaboom")` throws ErrorException("kaboom"). The stored error must
        # be the raw "kaboom" — NOT `string(e)` == "ErrorException(\"kaboom\")",
        # whose type-constructor noise would reach the model verbatim once the loop
        # prepends its single "Error: " prefix.
        outcome = UniLM._dispatch_tool("explode", Dict{String,Any}(),
            (n, a) -> error("kaboom"))
        @test !outcome.success
        @test outcome.error == "kaboom"
        @test !occursin("ErrorException", outcome.error)
    end

    @testset "non-ErrorException stores the readable showerror form" begin
        # Any other exception renders through `showerror` (human-readable), never
        # the bare `string(e)` constructor form: KeyError(:x) becomes
        # "KeyError: key :x not found", not "KeyError(:x)".
        outcome = UniLM._dispatch_tool("lookup", Dict{String,Any}(),
            (n, a) -> throw(KeyError(:x)))
        @test !outcome.success
        @test outcome.error == sprint(showerror, KeyError(:x))
        @test outcome.error != string(KeyError(:x))
    end
end

# ─── ToolCallOutcome construction ────────────────────────────────────────────

@testset "ToolCallOutcome" begin
    @testset "success outcome" begin
        gf = UniLM.GPTFunction("fn", Dict{String,Any}("x" => 1))
        fcr = FunctionCallResult("fn", gf, "ok")
        o = ToolCallOutcome("fn", Dict{String,Any}("x" => 1), fcr, true, nothing)
        @test o.success
        @test o.tool_name == "fn"
        @test o.result === fcr
        @test isnothing(o.error)
    end

    @testset "failure outcome" begin
        o = ToolCallOutcome("fn", Dict{String,Any}(), nothing, false, "oops")
        @test !o.success
        @test isnothing(o.result)
        @test o.error == "oops"
    end
end

# ─── ToolLoopResult construction ─────────────────────────────────────────────

@testset "ToolLoopResult" begin
    chat = Chat()
    msg = Message(role=UniLM.RoleAssistant, content="done")
    success = LLMSuccess(message=msg, self=chat)
    r = ToolLoopResult(success, ToolCallOutcome[], 1, true, nothing)
    @test r.completed
    @test r.turns_used == 1
    @test isempty(r.tool_calls)
    @test isnothing(r.llm_error)
    @test r.response === success
end

# ─── _next_respond ───────────────────────────────────────────────────────────

@testset "_next_respond" begin
    r = Respond(input="hello", model="gpt-4o", temperature=0.5, stream=true)
    r2 = UniLM._next_respond(r; input="new input", previous_response_id="resp_123")

    @test r2.input == "new input"
    @test r2.previous_response_id == "resp_123"
    @test isnothing(r2.stream)  # streaming disabled in tool loop
    @test r2.model == "gpt-4o"
    @test r2.temperature == 0.5
end

# ─── tool_loop(::String; tools) convenience method ───────────────────────────

@testset "tool_loop(::String; tools) exists and routes kwargs" begin
    ct = CallableTool(FunctionTool(name="noop", description="does nothing"),
                      (name, args) -> "ok")

    # Exists, and `max_turns` is routed to the LOOP (not the Respond ctor, not
    # swallowed): the loop rejects max_turns=0 before any HTTP call, so this stays
    # offline. On code without the String method this line is a MethodError.
    @test_throws ArgumentError "max_turns" tool_loop("hi"; tools=[ct], max_turns=0)

    # `tools` is a required keyword.
    @test_throws UndefKeywordError tool_loop("hi")

    # A Respond-bound keyword is forwarded to the Respond constructor, which
    # validates it there (temperature must be in [0, 2]).
    @test_throws ArgumentError tool_loop("hi"; tools=[ct], temperature=5.0)

    # An unknown keyword is not silently swallowed — it reaches (and is rejected
    # by) the Respond constructor.
    @test_throws MethodError tool_loop("hi"; tools=[ct], not_a_real_kwarg=1)
end

# ─── Scripted endpoints for the loop tests ───────────────────────────────────
# Every request goes to the scripted local server of platform_seam_fixtures.jl
# (`_with_scripted`, OS-assigned port; `_url_probe_base[]` is its origin).

# OpenAI-wire endpoint whose decoder hands the loop the next scripted Message, so a
# test controls the exact finish_reason / tool_calls pair the loop receives.
struct _TLFixture <: UniLM.OpenAIWireEndpoint
    replies::Vector{Message}
end
UniLM.get_url(::_TLFixture, ::Chat) = _url_probe_base[] * "/v1/chat/completions"
UniLM.auth_header(::_TLFixture) = ["Content-Type" => "application/json"]
UniLM.decode_response(s::_TLFixture, ::HTTP.Response) = (; message=popfirst!(s.replies), usage=nothing)

# A sendable Chat on `service`: system + user.
function _tl_chat(service; kw...)
    chat = Chat(; service, model="mock", kw...)
    push!(chat, Message(Val(:system), "s"))
    push!(chat, Message(Val(:user), "u"))
    chat
end

# An assistant turn requesting the zero-arg calls `id => name`, finished with `finish`.
_tl_calls(finish, calls::Pair...) = Message(role=UniLM.RoleAssistant, finish_reason=finish,
    tool_calls=[ToolCall(id=id, func=UniLM.GPTFunction(name, Dict{String,Any}())) for (id, name) in calls])
_tl_reply(text) = Message(role=UniLM.RoleAssistant, content=text, finish_reason="stop")

# `f()` against the scripted server answering `{}` to every request (the fixture
# decoder ignores the body); returns `(f(), requests)`.
_tl_scripted(f) = _with_scripted(f, (_, _) -> _json(200, "{}"))

# Responses-wire pieces for the scripted server.
function _tl_resp(id, output; status="completed", reason=nothing)
    d = Dict{String,Any}("id" => id, "status" => status, "model" => "mock", "output" => output)
    isnothing(reason) || (d["incomplete_details"] = Dict("reason" => reason))
    JSON.json(d)
end
_tl_fcall(call_id, name, args="{}") = Dict("type" => "function_call", "id" => "fc_" * call_id,
    "call_id" => call_id, "name" => name, "arguments" => args, "status" => "completed")
_tl_text(s) = Dict("type" => "message", "role" => "assistant", "status" => "completed",
    "content" => [Dict("type" => "output_text", "text" => s)])
_tl_respond(; kw...) = Respond(; service=GenericOpenAIEndpoint(_url_probe_base[], ""),
                               model="mock", input="go", kw...)
_tl_body(req) = JSON.parse(req.body; dicttype=Dict{String,Any})

# ─── Stop conditions ─────────────────────────────────────────────────────────

@testset "tool calls run only on a tool_calls finish; any other reason stops the loop" begin
    # A zero-arg destructive call on a turn the model did not finish as a tool
    # request (truncated, filtered, provider-specific) must never execute.
    for reason in ("length", "content_filter", "MALFORMED_FUNCTION_CALL")
        ran = Ref(0)
        chat = _tl_chat(_TLFixture([_tl_calls(reason, "call_1" => "wipe_disk")]))
        res, seen = _tl_scripted() do
            tool_loop!(chat, (name, args) -> (ran[] += 1; "wiped"))
        end
        @test ran[] == 0
        @test !res.completed && res.turns_used == 1 && length(seen) == 1
        @test isempty(res.tool_calls)
        @test occursin(reason, res.llm_error)
        @test res.response isa LLMSuccess && res.response.message.finish_reason == reason
        # The Chat stays sendable: the unanswered assistant tool-call turn is gone.
        @test length(chat) == 2 && last(chat).role == UniLM.RoleUser
    end
end

@testset "tool_loop! refuses a Chat without history before any request" begin
    # With history=false the assistant tool-call turn is never recorded, so the
    # follow-up request would carry tool results that answer nothing.
    chat = _tl_chat(GenericOpenAIEndpoint("http://127.0.0.1:1", ""); history=false)
    cfg = RequestConfig(max_attempts=1)
    @test_throws ArgumentError tool_loop!(chat, (name, args) -> "x"; config=cfg)
    @test_throws ArgumentError tool_loop!(chat; tools=CallableTool[], config=cfg)
end

@testset "Responses: a non-completed turn stops the loop, names its reason, runs no call" begin
    for reason in ("max_output_tokens", "content_filter")
        ran = Ref(0)
        body = _tl_resp("resp_1", [_tl_fcall("call_1", "wipe_disk")]; status="incomplete", reason)
        res, seen = _with_scripted((_, _) -> _json(200, body)) do
            tool_loop(_tl_respond(), (name, args) -> (ran[] += 1; "wiped"))
        end
        @test ran[] == 0
        @test !res.completed && res.turns_used == 1 && length(seen) == 1
        @test occursin("status=incomplete", res.llm_error) && occursin(reason, res.llm_error)
    end
end

@testset "max_turns exhaustion keeps the last real response" begin
    chat = _tl_chat(_TLFixture([_tl_calls(UniLM.TOOL_CALLS, "call_$i" => "noop") for i in 1:2]))
    res, seen = _tl_scripted() do
        tool_loop!(chat, (name, args) -> "ok"; max_turns=2)
    end
    @test !res.completed && res.turns_used == 2 && length(res.tool_calls) == 2 && length(seen) == 2
    @test res.llm_error == "max turns (2) exhausted"
    @test res.response isa LLMSuccess && only(res.response.message.tool_calls).id == "call_2"
    @test last(chat).role == UniLM.RoleTool && last(chat).tool_call_id == "call_2"

    body(n) = _tl_resp("resp_$n", [_tl_fcall("call_$n", "noop")])
    res, seen = _with_scripted((n, _) -> _json(200, body(n))) do
        tool_loop(_tl_respond(), (name, args) -> "ok"; max_turns=2)
    end
    @test !res.completed && res.turns_used == 2 && length(res.tool_calls) == 2 && length(seen) == 2
    @test res.llm_error == "max turns (2) exhausted"
    @test res.response isa ResponseSuccess && res.response.response.id == "resp_2"
end

@testset "max_turns below 1 is rejected before any request" begin
    dead = GenericOpenAIEndpoint("http://127.0.0.1:1", "")
    for n in (0, -1)
        @test_throws ArgumentError "max_turns" tool_loop!(_tl_chat(dead), (a, b) -> "x"; max_turns=n)
        @test_throws ArgumentError "max_turns" tool_loop(
            Respond(service=dead, model="mock", input="x"), (a, b) -> "x"; max_turns=n)
    end
end
