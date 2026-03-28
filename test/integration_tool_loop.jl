# ─── Tool Loop Integration Tests ─────────────────────────────────────────────
# Requires OPENAI_API_KEY environment variable

if !haskey(ENV, "OPENAI_API_KEY")
    @info "Skipping tool loop integration tests (OPENAI_API_KEY not set)"
else

# Shared tool schema: add(a, b) -> a + b
const _ADD_TOOL_SCHEMA = Dict(
    "type" => "object",
    "properties" => Dict(
        "a" => Dict("type" => "number", "description" => "First number"),
        "b" => Dict("type" => "number", "description" => "Second number")
    ),
    "required" => ["a", "b"],
    "additionalProperties" => false
)

const _ADD_DISPATCHER = (name::String, args::Dict{String,Any}) -> string(args["a"] + args["b"])

# ── Chat Completions ─────────────────────────────────────────────────────────

@testset "tool_loop! — Chat Completions — dispatcher" begin
    add_tool = GPTTool(func=GPTFunctionSignature(
        name="add", description="Add two numbers", parameters=_ADD_TOOL_SCHEMA))

    chat = Chat(model="gpt-5.4-nano", temperature=0.0, tools=[add_tool])
    push!(chat, Message(Val(:system), "You are a calculator. Always use the add tool. Give only the number in your final answer."))
    push!(chat, Message(Val(:user), "What is 3 + 5?"))

    result = tool_loop!(chat, _ADD_DISPATCHER; max_turns=5)
    @test result isa ToolLoopResult
    @test result.completed
    @test result.turns_used >= 2
    @test !isempty(result.tool_calls)
    @test result.tool_calls[1].success
    @test result.tool_calls[1].tool_name == "add"
    @test occursin("8", result.response.message.content)
end

@testset "tool_loop! — Chat Completions — CallableTool" begin
    add_gpt = GPTTool(func=GPTFunctionSignature(
        name="add", description="Add two numbers", parameters=_ADD_TOOL_SCHEMA))
    ct = CallableTool(add_gpt, _ADD_DISPATCHER)

    chat = Chat(model="gpt-5.4-nano", temperature=0.0, tools=[add_gpt])
    push!(chat, Message(Val(:system), "You are a calculator. Always use the add tool. Give only the number in your final answer."))
    push!(chat, Message(Val(:user), "What is 3 + 5?"))

    result = tool_loop!(chat; tools=[ct], max_turns=5)
    @test result isa ToolLoopResult
    @test result.completed
    @test !isempty(result.tool_calls)
    @test result.tool_calls[1].success
end

# ── Responses API ────────────────────────────────────────────────────────────

@testset "tool_loop — Responses API — dispatcher" begin
    ft = function_tool("add", "Add two numbers",
        parameters=_ADD_TOOL_SCHEMA, strict=true)

    r = Respond(
        model="gpt-5.4-nano",
        input="What is 3 + 5? Use the add tool. Give only the number in your final answer.",
        tools=[ft],
        temperature=0.0
    )

    result = tool_loop(r, _ADD_DISPATCHER; max_turns=5)
    @test result isa ToolLoopResult
    @test result.completed
    @test !isempty(result.tool_calls)
    @test result.tool_calls[1].success
    @test result.tool_calls[1].tool_name == "add"
    @test occursin("8", output_text(result.response))
end

@testset "tool_loop — Responses API — CallableTool" begin
    ft = function_tool("add", "Add two numbers",
        parameters=_ADD_TOOL_SCHEMA, strict=true)
    ct = CallableTool(ft, _ADD_DISPATCHER)

    r = Respond(
        model="gpt-5.4-nano",
        input="What is 3 + 5? Use the add tool. Give only the number in your final answer.",
        tools=[ct],
        temperature=0.0
    )

    result = tool_loop(r; max_turns=5)
    @test result isa ToolLoopResult
    @test result.completed
    @test !isempty(result.tool_calls)
    @test result.tool_calls[1].success
end

@testset "tool_loop — Responses API — convenience" begin
    ft = function_tool("add", "Add two numbers",
        parameters=_ADD_TOOL_SCHEMA, strict=true)

    result = tool_loop(
        "What is 3 + 5? Use the add tool. Give only the number in your final answer.",
        _ADD_DISPATCHER;
        model="gpt-5.4-nano", tools=[ft], temperature=0.0, max_turns=5
    )
    @test result isa ToolLoopResult
    @test result.completed
    @test !isempty(result.tool_calls)
end

# ── Error handling ───────────────────────────────────────────────────────────

@testset "tool_loop! — dispatcher error recovery" begin
    add_tool = GPTTool(func=GPTFunctionSignature(
        name="add", description="Add two numbers", parameters=_ADD_TOOL_SCHEMA))

    failing_dispatcher = (name::String, args::Dict{String,Any}) -> error("tool crashed")

    chat = Chat(model="gpt-5.4-nano", temperature=0.0, tools=[add_tool])
    push!(chat, Message(Val(:system), "You are a calculator. Always use the add tool."))
    push!(chat, Message(Val(:user), "What is 3 + 5?"))

    result = tool_loop!(chat, failing_dispatcher; max_turns=5)
    @test result isa ToolLoopResult
    @test !isempty(result.tool_calls)
    @test !result.tool_calls[1].success
    @test !isnothing(result.tool_calls[1].error)
end

end  # if OPENAI_API_KEY
