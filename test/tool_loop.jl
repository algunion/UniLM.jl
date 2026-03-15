# ─── CallableTool ────────────────────────────────────────────────────────────

@testset "CallableTool" begin
    @testset "construction from GPTTool" begin
        sig = GPTFunctionSignature(name="add", description="Add numbers",
            parameters=Dict("type"=>"object", "properties"=>Dict("a"=>Dict("type"=>"number"))))
        tool = GPTTool(func=sig)
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

    @testset "JSON serialization delegates to inner GPTTool" begin
        sig = GPTFunctionSignature(name="fn1", description="desc")
        tool = GPTTool(func=sig)
        ct = CallableTool(tool, (n, a) -> "ok")

        lowered_tool = JSON.lower(tool)
        lowered_ct = JSON.lower(ct)
        @test lowered_ct == lowered_tool
        @test lowered_ct[:type] == "function"
        @test lowered_ct[:function] isa GPTFunctionSignature
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
    @testset "identity for GPTTool" begin
        tool = GPTTool(func=GPTFunctionSignature(name="t"))
        @test to_tool(tool) === tool
    end

    @testset "identity for FunctionTool" begin
        ft = FunctionTool(name="t")
        @test to_tool(ft) === ft
    end

    @testset "identity for CallableTool" begin
        tool = GPTTool(func=GPTFunctionSignature(name="t"))
        ct = CallableTool(tool, (n, a) -> "")
        @test to_tool(ct) === ct
    end

    @testset "dict conversion to GPTTool (bare)" begin
        d = Dict("name" => "myfn", "description" => "a fn",
            "parameters" => Dict("type" => "object"))
        result = to_tool(d)
        @test result isa GPTTool
        @test result.func.name == "myfn"
        @test result.func.description == "a fn"
    end

    @testset "dict conversion to GPTTool (wrapped)" begin
        d = Dict("type" => "function", "function" => Dict(
            "name" => "wrapped_fn", "description" => "wrapped"))
        result = to_tool(d)
        @test result isa GPTTool
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

# ─── ToolCallOutcome construction ────────────────────────────────────────────

@testset "ToolCallOutcome" begin
    @testset "success outcome" begin
        gf = UniLM.GPTFunction("fn", Dict{String,Any}("x" => 1))
        fcr = GPTFunctionCallResult("fn", gf, "ok")
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
