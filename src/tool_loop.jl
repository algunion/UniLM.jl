# ============================================================================
# Tool-Calling Loop Integration
# Provides generic tool dispatch loops for both Chat Completions and Responses API.
# ============================================================================

# ─── Types ──────────────────────────────────────────────────────────────────

"""
    CallableTool{T}(tool, callable)

Wraps a tool schema `T` ([`GPTTool`](@ref) or [`FunctionTool`](@ref)) with a callable.
JSON serialization delegates to the inner tool, preserving backward compatibility.

# Fields
- `tool::T`: The tool schema.
- `callable::Function`: `(name::String, args::Dict{String,Any}) -> String`

# Example
```julia
tool = GPTTool(func=GPTFunctionSignature(name="add", description="Add two numbers",
    parameters=Dict("type"=>"object","properties"=>Dict("a"=>Dict("type"=>"number"),"b"=>Dict("type"=>"number")))))
ct = CallableTool(tool, (name, args) -> string(args["a"] + args["b"]))
```
"""
struct CallableTool{T}
    tool::T
    callable::Function
end

JSON.lower(ct::CallableTool) = JSON.lower(ct.tool)

_tool_name(t::GPTTool) = t.func.name
_tool_name(t::FunctionTool) = t.name
_tool_name(ct::CallableTool) = _tool_name(ct.tool)

"""
    to_tool(x)

Overloadable conversion protocol. Identity for GPTTool, FunctionTool, CallableTool.
Converts AbstractDict to GPTTool. Package extensions can add methods for other types.
"""
to_tool(x::GPTTool) = x
to_tool(x::FunctionTool) = x
to_tool(x::CallableTool) = x
to_tool(d::AbstractDict) = GPTTool(d)

"""
    ToolCallOutcome

Per-call record from a tool dispatch.

# Fields
- `tool_name::String`: Name of the tool that was called.
- `arguments::Dict{String,Any}`: Arguments passed to the tool.
- `result::Union{GPTFunctionCallResult,Nothing}`: The result wrapper, or `nothing` on failure.
- `success::Bool`: Whether the dispatch succeeded.
- `error::Union{String,Nothing}`: Error message on failure.
"""
struct ToolCallOutcome
    tool_name::String
    arguments::Dict{String,Any}
    result::Union{GPTFunctionCallResult,Nothing}
    success::Bool
    error::Union{String,Nothing}
end

"""
    ToolLoopResult

Result of a tool dispatch loop.

# Fields
- `response::LLMRequestResponse`: The final API response.
- `tool_calls::Vector{ToolCallOutcome}`: History of all tool dispatches.
- `turns_used::Int`: Number of API round-trips.
- `completed::Bool`: Whether the loop terminated normally (text response).
- `llm_error::Union{String,Nothing}`: Error message if not completed.
"""
struct ToolLoopResult
    response::LLMRequestResponse
    tool_calls::Vector{ToolCallOutcome}
    turns_used::Int
    completed::Bool
    llm_error::Union{String,Nothing}
end

# ─── Dispatch Helper ────────────────────────────────────────────────────────

"""
    _dispatch_tool(name, args, dispatcher) -> ToolCallOutcome

Call `dispatcher(name, args)`, wrap result in [`GPTFunctionCallResult`](@ref),
return a [`ToolCallOutcome`](@ref). Catches exceptions as error outcomes.
"""
function _dispatch_tool(name::String, args::Dict{String,Any}, dispatcher::Function)::ToolCallOutcome
    try
        result_str = string(dispatcher(name, args))
        gptfunc = GPTFunction(name, args)
        fcr = GPTFunctionCallResult(name, gptfunc, result_str)
        ToolCallOutcome(name, args, fcr, true, nothing)
    catch e
        ToolCallOutcome(name, args, nothing, false, string(e))
    end
end

# ─── Chat Completions Loop ──────────────────────────────────────────────────

"""
    tool_loop!(chat::Chat, dispatcher::Function; max_turns=10, retries=0, callback=nothing, on_tool_call=nothing) -> ToolLoopResult

Run a tool-calling loop on a [`Chat`](@ref). Repeatedly calls [`chatrequest!`](@ref),
dispatches tool calls via `dispatcher(name, args)`, pushes tool-role messages back,
and repeats until a text response, API error, or `max_turns`.

# Arguments
- `dispatcher`: `(name::String, args::Dict{String,Any}) -> String`
- `max_turns`: Maximum API round-trips (default 10).
- `retries`: Retry count passed to `chatrequest!`.
- `callback`: Streaming callback passed to `chatrequest!`.
- `on_tool_call`: Tool call notification callback passed to `chatrequest!`.

# Example
```julia
chat = Chat(model="gpt-5-mini", tools=[tool])
push!(chat, Message(Val(:system), "You are a calculator"))
push!(chat, Message(Val(:user), "What is 3+5?"))
result = tool_loop!(chat, (name, args) -> string(args["a"] + args["b"]))
```
"""
function tool_loop!(chat::Chat, dispatcher::Function;
                    max_turns::Int=10, retries::Int=0,
                    callback=nothing, on_tool_call=nothing)::ToolLoopResult
    all_outcomes = ToolCallOutcome[]
    turns = 0

    while turns < max_turns
        turns += 1
        raw = chatrequest!(chat; retries, callback, on_tool_call)
        result = raw isa Task ? fetch(raw) : raw

        if result isa LLMFailure
            return ToolLoopResult(result, all_outcomes, turns, false, result.response)
        elseif result isa LLMCallError
            return ToolLoopResult(result, all_outcomes, turns, false, result.error)
        end

        msg = result.message

        if msg.finish_reason != TOOL_CALLS || isnothing(msg.tool_calls)
            return ToolLoopResult(result, all_outcomes, turns, true, nothing)
        end

        for tc in msg.tool_calls
            outcome = _dispatch_tool(tc.func.name, tc.func.arguments, dispatcher)
            push!(all_outcomes, outcome)
            content = outcome.success ? string(outcome.result.result) : "Error: $(outcome.error)"
            push!(chat, Message(role=RoleTool, content=content, tool_call_id=tc.id))
        end
    end

    ToolLoopResult(
        LLMCallError(error="max turns ($max_turns) exhausted", self=chat),
        all_outcomes, turns, false, "max turns ($max_turns) exhausted"
    )
end

"""
    tool_loop!(chat::Chat; tools::Vector{<:CallableTool}, kwargs...) -> ToolLoopResult

No-dispatcher variant: builds a dispatcher from [`CallableTool`](@ref) entries.
"""
function tool_loop!(chat::Chat; tools::Vector{<:CallableTool},
                    max_turns::Int=10, retries::Int=0,
                    callback=nothing, on_tool_call=nothing)::ToolLoopResult
    tool_map = Dict{String,Function}(_tool_name(ct) => ct.callable for ct in tools)
    dispatcher = (name, args) -> begin
        fn = get(tool_map, name, nothing)
        isnothing(fn) && error("Unknown tool: $name")
        fn(name, args)
    end
    tool_loop!(chat, dispatcher; max_turns, retries, callback, on_tool_call)
end

# ─── Responses API Loop ─────────────────────────────────────────────────────

"""Reconstruct a [`Respond`](@ref) with new `input` and `previous_response_id`, copying all other fields.
Streaming is always disabled in the tool loop."""
function _next_respond(r::Respond; input, previous_response_id=nothing)
    kwargs = Dict{Symbol,Any}()
    for field in fieldnames(Respond)
        field in (:input, :previous_response_id, :stream) && continue
        kwargs[field] = getfield(r, field)
    end
    Respond(; input, previous_response_id, stream=nothing, kwargs...)
end

"""
    tool_loop(r::Respond, dispatcher::Function; max_turns=10, retries=0) -> ToolLoopResult

Run a tool-calling loop on a [`Respond`](@ref) request. Dispatches function calls
via `dispatcher(name, args)`, builds `function_call_output` input items, and chains
via `previous_response_id`.
"""
function tool_loop(r::Respond, dispatcher::Function;
                   max_turns::Int=10, retries::Int=0)::ToolLoopResult
    all_outcomes = ToolCallOutcome[]
    turns = 0
    input = r.input
    prev_id = r.previous_response_id

    while turns < max_turns
        turns += 1
        req = _next_respond(r; input, previous_response_id=prev_id)
        raw = respond(req; retries)
        result = raw isa Task ? fetch(raw) : raw

        if result isa ResponseFailure
            return ToolLoopResult(result, all_outcomes, turns, false, result.response)
        elseif result isa ResponseCallError
            return ToolLoopResult(result, all_outcomes, turns, false, result.error)
        end

        calls = function_calls(result)

        if isempty(calls)
            return ToolLoopResult(result, all_outcomes, turns, true, nothing)
        end

        output_items = Any[]
        for call in calls
            name = call["name"]
            args = JSON.parse(call["arguments"]; dicttype=Dict{String,Any})
            outcome = _dispatch_tool(name, args, dispatcher)
            push!(all_outcomes, outcome)
            content = outcome.success ? string(outcome.result.result) : "Error: $(outcome.error)"
            push!(output_items, Dict{String,Any}(
                "type" => "function_call_output",
                "call_id" => call["call_id"],
                "output" => content
            ))
        end

        input = output_items
        prev_id = result.response.id
    end

    ToolLoopResult(
        ResponseCallError(error="max turns ($max_turns) exhausted"),
        all_outcomes, turns, false, "max turns ($max_turns) exhausted"
    )
end

"""
    tool_loop(r::Respond; max_turns=10, retries=0) -> ToolLoopResult

No-dispatcher variant: extracts callables from [`CallableTool`](@ref) entries in `r.tools`.
"""
function tool_loop(r::Respond; max_turns::Int=10, retries::Int=0)::ToolLoopResult
    callables = Dict{String,Function}()
    if !isnothing(r.tools)
        for t in r.tools
            t isa CallableTool && (callables[_tool_name(t)] = t.callable)
        end
    end
    isempty(callables) && throw(ArgumentError("No CallableTool entries found in tools"))
    dispatcher = (name, args) -> begin
        fn = get(callables, name, nothing)
        isnothing(fn) && error("Unknown tool: $name")
        fn(name, args)
    end
    tool_loop(r, dispatcher; max_turns, retries)
end

"""
    tool_loop(input, dispatcher::Function; tools, kwargs...) -> ToolLoopResult

Convenience form: creates a [`Respond`](@ref) and runs the tool loop.
"""
function tool_loop(input, dispatcher::Function; kwargs...)
    kws = Dict{Symbol,Any}(kwargs)
    retries = pop!(kws, :retries, 0)
    max_turns = pop!(kws, :max_turns, 10)
    r = Respond(; input, kws...)
    tool_loop(r, dispatcher; max_turns, retries)
end
