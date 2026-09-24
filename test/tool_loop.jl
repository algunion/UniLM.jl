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

    @testset "Strings pass through; any other result is JSON-encoded" begin
        # The model reads the result as text: never a Julia repr such as
        # `Dict{String, Any}("temp" => 21.5)` or `nothing`.
        for (ret, sent) in (("plain text", "plain text"), (SubString("abcd", 1, 2), "ab"),
                            (Dict("temp" => 21.5), "{\"temp\":21.5}"), (nothing, "null"),
                            (42, "42"), ([1, 2], "[1,2]"), (true, "true"))
            outcome = UniLM._dispatch_tool("t", Dict{String,Any}(), (n, a) -> ret)
            @test outcome.success && outcome.result.result == sent
        end
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

@testset "a text turn completes the Chat loop only when it finished with stop (or none)" begin
    # A filtered, paused or otherwise unfinished text turn is not a completed answer.
    for reason in ("content_filter", "pause_turn", "malformed_function_call")
        chat = _tl_chat(_TLFixture([Message(role=UniLM.RoleAssistant, content="partial",
                                            finish_reason=reason)]))
        res, seen = _tl_scripted() do
            tool_loop!(chat, (name, args) -> "x")
        end
        @test !res.completed && res.turns_used == 1 && length(seen) == 1
        @test occursin(reason, res.llm_error)
        @test res.response isa LLMSuccess && res.response.message.content == "partial"
    end
    # "stop", none, and a "tool_calls" finish that carries no calls complete; "length" is
    # reported as the truncation it is.
    for (reason, completed, err) in (("stop", true, nothing), (nothing, true, nothing),
                                     (UniLM.TOOL_CALLS, true, nothing),
                                     ("length", false, "Model output was truncated by the token limit"))
        chat = _tl_chat(_TLFixture([Message(role=UniLM.RoleAssistant, content="done",
                                            finish_reason=reason)]))
        res, _ = _tl_scripted() do
            tool_loop!(chat, (name, args) -> "x")
        end
        @test res.completed == completed && res.llm_error == err
    end
end

@testset "tool_loop! refuses a Chat without history before any request" begin
    # With history=false the assistant tool-call turn is never recorded, so the
    # follow-up request would carry tool results that answer nothing.
    chat = _tl_chat(GenericOpenAIEndpoint("http://127.0.0.1:1", ""); history=false)
    cfg = RequestConfig(max_attempts=1)
    @test_throws ArgumentError "history=true" tool_loop!(chat, (name, args) -> "x"; config=cfg)
    @test_throws ArgumentError "history=true" tool_loop!(chat; tools=CallableTool[], config=cfg)
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

@testset "a structured tool result reaches the model as JSON" begin
    chat = _tl_chat(_TLFixture([_tl_calls(UniLM.TOOL_CALLS, "call_1" => "weather"), _tl_reply("done")]))
    res, _ = _tl_scripted() do
        tool_loop!(chat, (name, args) -> Dict("temp" => 21.5))
    end
    @test res.completed
    @test only(filter(m -> m.role == UniLM.RoleTool, chat.messages)).content == "{\"temp\":21.5}"

    turn(n) = n == 1 ? _tl_resp("resp_1", [_tl_fcall("call_1", "weather")]) :
                       _tl_resp("resp_2", [_tl_text("done")])
    res, seen = _with_scripted((n, _) -> _json(200, turn(n))) do
        tool_loop(_tl_respond(), (name, args) -> nothing)
    end
    @test res.completed && length(seen) == 2
    @test only(_tl_body(seen[2])["input"])["output"] == "null"
end

# ─── Responses loop chaining and unrunnable calls ─────────────────────────────

@testset "Responses: a conversation chains through conversation alone" begin
    # The API rejects previous_response_id together with conversation; the
    # conversation already holds the previous turn's items.
    turn(n) = n == 1 ? _tl_resp("resp_1", [_tl_fcall("call_1", "noop")]) :
                       _tl_resp("resp_2", [_tl_text("done")])
    for conversation in ("conv_1", Dict("id" => "conv_1"))
        res, seen = _with_scripted((n, _) -> _json(200, turn(n))) do
            tool_loop(_tl_respond(; conversation), (name, args) -> "ok")
        end
        @test res.completed && length(seen) == 2
        follow = _tl_body(seen[2])
        @test follow["conversation"] == conversation
        @test !haskey(follow, "previous_response_id")
        @test only(follow["input"])["call_id"] == "call_1"
    end
    # Without a conversation the follow-up chains through previous_response_id.
    res, seen = _with_scripted((n, _) -> _json(200, turn(n))) do
        tool_loop(_tl_respond(), (name, args) -> "ok")
    end
    @test res.completed && _tl_body(seen[2])["previous_response_id"] == "resp_1"
end

@testset "Responses: malformed call arguments go back to the model as that call's output" begin
    calls = [_tl_fcall("call_bad", "lookup", "{\"city\": "), _tl_fcall("call_arr", "lookup", "[1, 2]"),
             _tl_fcall("call_ok", "lookup", "{\"city\": \"Oslo\"}")]
    turn(n) = n == 1 ? _tl_resp("resp_1", calls) : _tl_resp("resp_2", [_tl_text("done")])
    ran = String[]
    res, seen = _with_scripted((n, _) -> _json(200, turn(n))) do
        tool_loop(_tl_respond(), (name, args) -> (push!(ran, args["city"]); "sunny"))
    end
    @test res.completed && res.turns_used == 2
    @test ran == ["Oslo"]                  # the malformed calls never reach the dispatcher
    @test [o.success for o in res.tool_calls] == [false, false, true]
    @test all(o -> occursin("invalid arguments", o.error), res.tool_calls[1:2])
    outs = _tl_body(seen[2])["input"]
    @test [o["call_id"] for o in outs] == ["call_bad", "call_arr", "call_ok"]
    @test all(o -> startswith(o["output"], "Error: invalid arguments"), outs[1:2])
    @test outs[3]["output"] == "sunny"
end

@testset "Responses: a pending client-side call the loop cannot run ends it incomplete" begin
    # A local shell_call is the client's to run; an mcp_approval_request waits for an
    # mcp_approval_response. Either one left pending makes the next request a 400.
    for typ in ("custom_tool_call", "apply_patch_call", "local_shell_call", "computer_call",
                "shell_call", "mcp_approval_request")
        ran = Ref(0)
        item = Dict("type" => typ, "id" => "item_1", "call_id" => "call_1", "status" => "completed")
        # Alone, and next to a function call whose output could not be sent without it.
        for output in ([item], [_tl_fcall("call_2", "noop"), item])
            res, seen = _with_scripted((_, _) -> _json(200, _tl_resp("resp_1", output))) do
                tool_loop(_tl_respond(), (name, args) -> (ran[] += 1; "ok"))
            end
            @test !res.completed && res.turns_used == 1 && length(seen) == 1
            @test occursin(typ, res.llm_error)
        end
        @test ran[] == 0
    end
end

@testset "Responses: a hosted shell call the platform answered does not stop the loop" begin
    # A shell tool in a container environment runs on the platform, which emits the
    # call's shell_call_output in the same output: nothing is left for the client.
    shell = Dict("type" => "shell_call", "id" => "sh_1", "call_id" => "call_sh", "status" => "completed",
                 "environment" => Dict("type" => "container_reference", "container_id" => "cntr_1"))
    answer = Dict("type" => "shell_call_output", "id" => "sho_1", "call_id" => "call_sh", "output" => Any[])
    replies = [_tl_resp("resp_1", [shell, answer, _tl_fcall("call_2", "noop")]),
               _tl_resp("resp_2", [_tl_text("done")])]
    ran = Ref(0)
    res, seen = _with_scripted((n, _) -> _json(200, replies[n])) do
        tool_loop(_tl_respond(), (name, args) -> (ran[] += 1; "ok"))
    end
    @test res.completed && res.turns_used == 2 && length(seen) == 2 && ran[] == 1
end

@testset "Responses: a CallableTool wrapping a Chat Tool goes out in the Responses function shape" begin
    # The Chat shape nests the function under `function`, which the Responses API
    # rejects. Respond converts a bare Chat Tool; one wrapped in a CallableTool — the
    # tools a Chat loop takes — converts the same way and keeps its callable.
    params = Dict("type" => "object", "properties" => Dict("city" => Dict("type" => "string")))
    ct = CallableTool(Tool(func=FunctionSignature(name="weather", description="Get weather",
                                                  parameters=params)), (name, args) -> "sunny")
    replies = [_tl_resp("resp_1", [_tl_fcall("call_1", "weather", "{\"city\":\"Oslo\"}")]),
               _tl_resp("resp_2", [_tl_text("done")])]
    (r, res), seen = _with_scripted((n, _) -> _json(200, replies[n])) do
        r = _tl_respond(; tools=[ct])
        r, tool_loop(r)
    end
    @test r.tools[1] isa CallableTool{FunctionTool} && r.tools[1].callable === ct.callable
    @test res.completed && only(res.tool_calls).success && only(res.tool_calls).result.result == "sunny"
    for req in seen
        wire = only(_tl_body(req)["tools"])
        @test wire["type"] == "function" && wire["name"] == "weather" && !haskey(wire, "function")
        @test wire["parameters"] == params
    end
end

# ─── Cancellation ────────────────────────────────────────────────────────────

@testset "cancel: a cancel between tool dispatches stops the Chat loop before the next call" begin
    tok = CancelToken()
    ran, scopes = String[], Any[]
    chat = _tl_chat(_TLFixture([_tl_calls(UniLM.TOOL_CALLS, "c1" => "first", "c2" => "second",
                                          "c3" => "third")]))
    res, seen = _tl_scripted() do
        tool_loop!(chat, (name, args) -> (push!(ran, name); push!(scopes, UniLM._current_cancel());
                                          cancel!(tok); "ok"); cancel=tok)
    end
    @test ran == ["first"]                          # no dispatch after the cancel
    @test only(scopes) === tok                      # the turn runs inside the token's scope
    @test !res.completed && res.turns_used == 1 && length(seen) == 1
    @test res.response isa LLMCallError && res.response.cause isa UniLMCancelled
    @test res.llm_error == res.response.error
    @test only(res.tool_calls).tool_name == "first"
    @test length(chat) == 2                         # the interrupted turn is rolled back
end

@testset "cancel: a cancel between tool dispatches stops the Responses loop" begin
    tok = CancelToken()
    ran, scopes = String[], Any[]
    body = _tl_resp("resp_1", [_tl_fcall("c1", "first"), _tl_fcall("c2", "second")])
    res, seen = _with_scripted((_, _) -> _json(200, body)) do
        tool_loop(_tl_respond(), (name, args) -> (push!(ran, name); push!(scopes, UniLM._current_cancel());
                                                  cancel!(tok); "ok"); cancel=tok)
    end
    @test ran == ["first"] && only(scopes) === tok
    @test !res.completed && res.turns_used == 1 && length(seen) == 1
    @test res.response isa ResponseCallError && res.response.cause isa UniLMCancelled
    @test res.llm_error == res.response.error
end

@testset "cancel: the ambient with_cancel token is honoured (CallableTool variant)" begin
    tok = CancelToken()
    ran = String[]
    tools = [CallableTool(Tool(func=FunctionSignature(name=n)),
                          (name, args) -> (push!(ran, name); cancel!(tok); "ok")) for n in ("first", "second")]
    chat = _tl_chat(_TLFixture([_tl_calls(UniLM.TOOL_CALLS, "c1" => "first", "c2" => "second")]))
    res, _ = _tl_scripted() do
        with_cancel(() -> tool_loop!(chat; tools), tok)
    end
    @test ran == ["first"]
    @test !res.completed && res.response isa LLMCallError && res.response.cause isa UniLMCancelled
end

@testset "cancel: a cancel mid-request returns promptly with the typed cancellation result" begin
    tok = CancelToken()
    hits = Threads.Atomic{Int}(0)
    release = Threads.Atomic{Bool}(false)
    held = (n, _) -> begin
        Threads.atomic_add!(hits, 1)
        n >= 2 && timedwait(() -> release[], 30.0)   # the follow-up reply is held for 30 s
        _json(200, "{}")
    end
    chat = _tl_chat(_TLFixture([_tl_calls(UniLM.TOOL_CALLS, "c1" => "noop"), _tl_reply("late")]))
    ran = Ref(0)
    (res, after_cancel), _ = _with_scripted(held) do
        t = Threads.@spawn tool_loop!(chat, (name, args) -> (ran[] += 1; "ok"); cancel=tok)
        in_flight = timedwait(() -> hits[] >= 2 || istaskdone(t), 25.0)
        sleep(0.2)
        cancelled_at = time_ns()
        cancel!(tok)
        done = timedwait(() -> istaskdone(t), 25.0)
        after = (time_ns() - cancelled_at) / 1e9
        release[] = true
        @test in_flight === :ok && done === :ok
        (fetch(t), after)
    end
    @test after_cancel < 2.0
    @test !res.completed && res.turns_used == 2 && ran[] == 1
    @test res.response isa LLMCallError && res.response.cause isa UniLMCancelled
end

@testset "cancel: every loop form forwards the token; a cancelled one sends nothing" begin
    tok = cancel!(CancelToken())
    ct = CallableTool(FunctionTool(name="noop"), (n, a) -> "ok")
    chat_ct = CallableTool(Tool(func=FunctionSignature(name="noop")), (n, a) -> "ok")
    results, seen = _tl_scripted() do
        svc = GenericOpenAIEndpoint(_url_probe_base[], "")
        [tool_loop!(_tl_chat(svc), (n, a) -> "ok"; cancel=tok),
         tool_loop!(_tl_chat(svc); tools=[chat_ct], cancel=tok),
         tool_loop(_tl_respond(), (n, a) -> "ok"; cancel=tok),
         tool_loop(_tl_respond(; tools=[ct]); cancel=tok),
         tool_loop("go", (n, a) -> "ok"; service=svc, model="mock", cancel=tok),
         tool_loop("go"; tools=[ct], service=svc, model="mock", cancel=tok)]
    end
    @test isempty(seen)
    @test all(r -> !r.completed && r.turns_used == 1 && occursin("cancelled", r.llm_error), results)
end

# ─── Concurrent dispatch ─────────────────────────────────────────────────────
# Call i sleeps (1.3 - 0.1i) s: ≥ 1 s each, finishing in reverse call order. The
# tools record their own start/end, so the asserted span covers the dispatch alone.

function _tl_timed_dispatcher()
    spans, tasks, lk = Tuple{UInt64,UInt64}[], Task[], ReentrantLock()
    dispatcher = (name, args) -> begin
        t = time_ns()
        sleep(1.3 - 0.1 * parse(Int, name[2:end]))
        @lock lk (push!(spans, (t, time_ns())); push!(tasks, current_task()))
        "result of $name"
    end
    span() = (maximum(last, spans) - minimum(first, spans)) / 1e9
    (; dispatcher, span, tasks)
end

@testset "tool_concurrency runs a turn's calls at once and keeps call order" begin
    runs = map((1, 3)) do n
        timed = _tl_timed_dispatcher()
        chat = _tl_chat(_TLFixture([_tl_calls(UniLM.TOOL_CALLS, ("c$i" => "t$i" for i in 1:3)...),
                                    _tl_reply("done")]))
        (res, caller), seen = _tl_scripted() do
            r = n == 1 ? tool_loop!(chat, timed.dispatcher) :
                         tool_loop!(chat, timed.dispatcher; tool_concurrency=n)
            (r, current_task())
        end
        (; res, chat, seen, span=timed.span(), on_caller=all(t -> t === caller, timed.tasks))
    end
    sequential, concurrent = runs
    @test sequential.span >= 3.0 && sequential.on_caller    # default: one at a time, in this task
    @test concurrent.span < 2.0 && !concurrent.on_caller
    for r in runs
        @test r.res.completed && r.res.turns_used == 2
        @test [o.tool_name for o in r.res.tool_calls] == ["t1", "t2", "t3"]
        tool_msgs = filter(m -> m.role == UniLM.RoleTool, r.chat.messages)
        @test [m.tool_call_id for m in tool_msgs] == ["c1", "c2", "c3"]
        @test [m.content for m in tool_msgs] == ["result of t$i" for i in 1:3]
    end
    @test sequential.seen[2].body == concurrent.seen[2].body   # the next request is deterministic
end

@testset "tool_concurrency on the Responses loop keeps call order" begin
    timed = _tl_timed_dispatcher()
    turn(n) = n == 1 ? _tl_resp("resp_1", [_tl_fcall("c$i", "t$i") for i in 1:3]) :
                       _tl_resp("resp_2", [_tl_text("done")])
    res, seen = _with_scripted((n, _) -> _json(200, turn(n))) do
        tool_loop(_tl_respond(), timed.dispatcher; tool_concurrency=3)
    end
    @test timed.span() < 2.0
    @test res.completed && [o.tool_name for o in res.tool_calls] == ["t1", "t2", "t3"]
    outs = _tl_body(seen[2])["input"]
    @test [o["call_id"] for o in outs] == ["c1", "c2", "c3"]
    @test [o["output"] for o in outs] == ["result of t$i" for i in 1:3]
end

@testset "tool_concurrency: an InterruptException from any dispatch propagates" begin
    interrupting = (name, args) -> name == "boom" ? throw(InterruptException()) : "ok"
    for n in (1, 3)
        chat = _tl_chat(_TLFixture([_tl_calls(UniLM.TOOL_CALLS, "c1" => "fine", "c2" => "boom",
                                              "c3" => "fine")]))
        _tl_scripted() do
            @test_throws InterruptException tool_loop!(chat, interrupting; tool_concurrency=n)
        end
        # The interrupted turn is rolled back with the results that did arrive: tool calls
        # left unanswered would make the next request a provider 400.
        @test [m.role for m in chat.messages] == [UniLM.RoleSystem, UniLM.RoleUser]
        @test issendvalid(chat)
    end
    body = _tl_resp("resp_1", [_tl_fcall("c1", "fine"), _tl_fcall("c2", "boom")])
    _with_scripted((_, _) -> _json(200, body)) do
        @test_throws InterruptException tool_loop(_tl_respond(), interrupting; tool_concurrency=2)
    end
end

@testset "tool_concurrency below 1 is rejected before any request" begin
    dead = GenericOpenAIEndpoint("http://127.0.0.1:1", "")
    @test_throws ArgumentError "tool_concurrency" tool_loop!(_tl_chat(dead), (a, b) -> "x"; tool_concurrency=0)
    @test_throws ArgumentError "tool_concurrency" tool_loop(
        Respond(service=dead, model="mock", input="x"), (a, b) -> "x"; tool_concurrency=0)
end

@testset "tool_concurrency: a cancel stops the hand-out of further calls" begin
    tok = CancelToken()
    ran, lk = String[], ReentrantLock()
    t2_started = Threads.Atomic{Bool}(false)
    dispatcher = (name, args) -> begin
        @lock lk push!(ran, name)
        if name == "t1"                        # cancel once t2 is in flight too
            timedwait(() -> t2_started[], 25.0)
            cancel!(tok)
        else
            t2_started[] = true
            sleep(0.2)
        end
        "ok"
    end
    chat = _tl_chat(_TLFixture([_tl_calls(UniLM.TOOL_CALLS, ("c$i" => "t$i" for i in 1:4)...)]))
    res, seen = _tl_scripted() do
        tool_loop!(chat, dispatcher; cancel=tok, tool_concurrency=2)
    end
    @test sort(ran) == ["t1", "t2"]            # t3 and t4 are never handed out
    @test [o.tool_name for o in res.tool_calls] == ["t1", "t2"]
    @test !res.completed && res.response isa LLMCallError && res.response.cause isa UniLMCancelled
    @test length(chat) == 2 && length(seen) == 1
end

# ─── CallableTool dispatch ───────────────────────────────────────────────────

@testset "CallableTool dispatch routes by name; unknown names are tool errors" begin
    tools = [CallableTool(FunctionTool(name="add"), (n, a) -> a["x"] + 1), FunctionTool(name="plain")]
    d = UniLM._callable_dispatcher(tools)
    @test d("add", Dict{String,Any}("x" => 1)) == 2
    @test_throws ErrorException "Unknown tool: plain" d("plain", Dict{String,Any}())
    @test_throws ArgumentError "No CallableTool" tool_loop(Respond(input="x", model="mock"))
    @test_throws ArgumentError "No CallableTool" tool_loop(
        Respond(input="x", model="mock", tools=[FunctionTool(name="plain")]))

    turn(n) = n == 1 ? _tl_resp("resp_1", [_tl_fcall("c1", "add", "{\"x\": 41}")]) :
                       _tl_resp("resp_2", [_tl_text("42")])
    res, _ = _with_scripted((n, _) -> _json(200, turn(n))) do
        tool_loop(_tl_respond(; tools=tools))
    end
    @test res.completed && only(res.tool_calls).result.result == "42"
end
