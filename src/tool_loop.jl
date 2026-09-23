# ============================================================================
# Tool-Calling Loop Integration
# Provides generic tool dispatch loops for both Chat Completions and Responses API.
# ============================================================================

# ─── Types ──────────────────────────────────────────────────────────────────

"""
    CallableTool{T}(tool, callable)

Wraps a tool schema `T` ([`Tool`](@ref) or [`FunctionTool`](@ref)) with a callable.
JSON serialization delegates to the inner tool, preserving backward compatibility.

# Fields
- `tool::T`: The tool schema.
- `callable::Function`: `(name::String, args::Dict{String,Any}) -> String`

# Example
```julia
tool = Tool(func=FunctionSignature(name="add", description="Add two numbers",
    parameters=Dict("type"=>"object","properties"=>Dict("a"=>Dict("type"=>"number"),"b"=>Dict("type"=>"number")))))
ct = CallableTool(tool, (name, args) -> string(args["a"] + args["b"]))
```
"""
struct CallableTool{T}
    tool::T
    callable::Function
end

JSON.lower(ct::CallableTool) = JSON.lower(ct.tool)

# CallableTool unwraps to its inner tool for the Gemini Interactions encoder (mirrors the
# JSON.lower unwrap the OpenAI wire uses). Defined here — _interactions_tool lives in
# interactions.jl (loaded before tool_loop.jl), CallableTool is defined just above.
_interactions_tool(ct::CallableTool) = _interactions_tool(ct.tool)

_tool_name(t::Tool) = t.func.name
_tool_name(t::FunctionTool) = t.name
_tool_name(ct::CallableTool) = _tool_name(ct.tool)

"""
    to_tool(x)

Overloadable conversion protocol. Identity for Tool, FunctionTool, CallableTool.
Converts AbstractDict to Tool. Package extensions can add methods for other types.
"""
to_tool(x::Tool) = x
to_tool(x::FunctionTool) = x
to_tool(x::CallableTool) = x
to_tool(d::AbstractDict) = Tool(d)

# Chat stores `Vector{Tool}` but accepts a `Vector{<:CallableTool}` at
# construction (e.g. `Chat(tools=mcp_tools(session))`) by unwrapping each
# wrapper's inner tool — no manual `map(t -> t.tool, tools)`. This completes the
# `_chat_tools` fallback declared in api.jl: `CallableTool` is defined here, in a
# file `include`d after api.jl. A wrapper whose `.tool` is not a `Tool` (e.g.
# a `FunctionTool` from `mcp_tools_respond`) fails the conversion.
_chat_tools(tools::Vector{<:CallableTool}) = Tool[ct.tool for ct in tools]

"""
    ToolCallOutcome

Per-call record from a tool dispatch.

# Fields
- `tool_name::String`: Name of the tool that was called.
- `arguments::Dict{String,Any}`: Arguments passed to the tool.
- `result::Union{FunctionCallResult,Nothing}`: The result wrapper, or `nothing` on failure.
- `success::Bool`: Whether the dispatch succeeded.
- `error::Union{String,Nothing}`: Error message on failure.
"""
struct ToolCallOutcome
    tool_name::String
    arguments::Dict{String,Any}
    result::Union{FunctionCallResult,Nothing}
    success::Bool
    error::Union{String,Nothing}
end

"""
    ToolLoopResult

Result of a tool dispatch loop.

# Fields
- `response::LLMRequestResponse`: The last response the loop received: the final
  text turn, the failure that ended the loop, the turn it stopped on, or — when
  `max_turns` ran out — the last tool-call turn (whose calls were dispatched).
- `tool_calls::Vector{ToolCallOutcome}`: History of all tool dispatches.
- `turns_used::Int`: Number of API round-trips.
- `completed::Bool`: Whether the loop terminated normally (text response).
  Truncated output, a turn stopped before its calls ran, a pending server action,
  or `max_turns` exhaustion leaves this `false`.
- `llm_error::Union{String,Nothing}`: Why the loop stopped when not completed
  (e.g. `"max turns (3) exhausted"`).
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

Call `dispatcher(name, args)`, wrap result in [`FunctionCallResult`](@ref),
return a [`ToolCallOutcome`](@ref). Catches exceptions as error outcomes.
"""
function _dispatch_tool(name::String, args::Dict{String,Any}, dispatcher::Function)::ToolCallOutcome
    try
        result_str = string(dispatcher(name, args))
        gptfunc = GPTFunction(name, args)
        fcr = FunctionCallResult(name, gptfunc, result_str)
        ToolCallOutcome(name, args, fcr, true, nothing)
    catch e
        # A user Ctrl-C (InterruptException) must abort the loop, not be recorded
        # as a tool failure and swallowed — propagate it before any conversion.
        e isa InterruptException && rethrow()
        # Store the tool's own message faithfully: `error("x")` carries it verbatim
        # in `.msg`, and any other error renders through `showerror` (its
        # human-readable form, e.g. `KeyError: key "x" not found`). `string(e)` would
        # instead leak the constructor form (`ErrorException("x")`) — exception-type
        # noise the model would otherwise have to see through.
        ToolCallOutcome(name, args, nothing, false,
                        e isa ErrorException ? e.msg : sprint(showerror, e))
    end
end

_check_max_turns(n::Int) =
    ispositive(n) || throw(ArgumentError("max_turns must be >= 1 (got $n)"))

# ─── Chat Completions Loop ──────────────────────────────────────────────────

"""
    tool_loop!(chat::Chat, dispatcher::Function; max_turns=10, config=nothing, callback=nothing, on_tool_call=nothing) -> ToolLoopResult

Run a tool-calling loop on a [`Chat`](@ref). Repeatedly calls [`chatrequest!`](@ref),
dispatches tool calls via `dispatcher(name, args)`, pushes tool-role messages back,
and repeats until a text response, API error, or `max_turns`.

Tool calls run only on a turn whose `finish_reason` is `"tool_calls"`. A turn that
carries tool calls but finished for any other reason (`"length"`, `"content_filter"`,
a provider-specific value) may hold partial calls: none runs, the loop stops with
`completed=false` and an `llm_error` naming the reason, and that unanswered assistant
turn is removed from `chat`, so the conversation stays sendable.

`chat.history` must be `true` (else `ArgumentError` before any request): each
follow-up request carries the tool results together with the assistant turn that
requested them.

# Arguments
- `dispatcher`: `(name::String, args::Dict{String,Any}) -> String`
- `max_turns`: Maximum API round-trips (default 10; `< 1` throws `ArgumentError`).
  When they run out, the result keeps the last response, with `completed=false` and
  `llm_error = "max turns (N) exhausted"`.
- `config`: Per-request [`RequestConfig`](@ref) passed to [`chatrequest!`](@ref) — each turn gets its own attempt/deadline budget.
- `callback`: Streaming callback passed to `chatrequest!`.
- `on_tool_call`: Tool call notification callback passed to `chatrequest!`.

# Example
```julia
chat = Chat(model="gpt-5.4-mini", tools=[tool])
push!(chat, Message(Val(:system), "You are a calculator"))
push!(chat, Message(Val(:user), "What is 3+5?"))
result = tool_loop!(chat, (name, args) -> string(args["a"] + args["b"]))
```
"""
function tool_loop!(chat::Chat, dispatcher::Function;
                    max_turns::Int=10, config::Union{Nothing,RequestConfig}=nothing,
                    callback=nothing, on_tool_call=nothing)::ToolLoopResult
    _check_max_turns(max_turns)
    chat.history || throw(ArgumentError("tool_loop! needs a Chat with history=true: " *
        "each follow-up request must carry the assistant turn its tool results answer"))
    all_outcomes = ToolCallOutcome[]
    turns = 0
    local latest::LLMSuccess

    while turns < max_turns
        turns += 1
        before = length(chat)
        raw = chatrequest!(chat; config, callback, on_tool_call)
        result = raw isa Task ? fetch(raw) : raw

        if result isa LLMFailure
            return ToolLoopResult(result, all_outcomes, turns, false, result.response)
        elseif result isa LLMCallError
            return ToolLoopResult(result, all_outcomes, turns, false, result.error)
        end

        latest = result
        msg = result.message
        calls = something(msg.tool_calls, ToolCall[])

        if isempty(calls)
            msg.finish_reason == "length" && return ToolLoopResult(result, all_outcomes, turns,
                false, "Model output was truncated by the token limit")
            return ToolLoopResult(result, all_outcomes, turns, true, nothing)
        end

        if msg.finish_reason != TOOL_CALLS
            resize!(chat.messages, before)   # no results will answer this turn
            return ToolLoopResult(result, all_outcomes, turns, false,
                "turn finished with finish_reason=$(repr(msg.finish_reason)); " *
                "its $(length(calls)) tool call(s) were not executed")
        end

        for tc in calls
            outcome = _dispatch_tool(tc.func.name, tc.func.arguments, dispatcher)
            push!(all_outcomes, outcome)
            content = outcome.success ? string(outcome.result.result) : "Error: $(outcome.error)"
            push!(chat, Message(role=RoleTool, content=content, tool_call_id=tc.id))
        end
    end

    ToolLoopResult(latest, all_outcomes, turns, false, "max turns ($max_turns) exhausted")
end

"""
    tool_loop!(chat::Chat; tools::Vector{<:CallableTool}, kwargs...) -> ToolLoopResult

No-dispatcher variant: builds a dispatcher from [`CallableTool`](@ref) entries.
"""
function tool_loop!(chat::Chat; tools::Vector{<:CallableTool},
                    max_turns::Int=10, config::Union{Nothing,RequestConfig}=nothing,
                    callback=nothing, on_tool_call=nothing)::ToolLoopResult
    tool_map = Dict{String,Function}(_tool_name(ct) => ct.callable for ct in tools)
    dispatcher = (name, args) -> begin
        fn = get(tool_map, name, nothing)
        isnothing(fn) && error("Unknown tool: $name")
        fn(name, args)
    end
    tool_loop!(chat, dispatcher; max_turns, config, callback, on_tool_call)
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
    tool_loop(r::Respond, dispatcher::Function; max_turns=10, config=nothing) -> ToolLoopResult

Run a tool-calling loop on a [`Respond`](@ref) request. Dispatches function calls
via `dispatcher(name, args)`, builds `function_call_output` input items, and chains
via `previous_response_id`.

Function calls run only on a `completed` or `requires_action` turn. Any other status
(e.g. `incomplete`, whose calls may be partial) stops the loop with `completed=false`
and an `llm_error` naming the status and the `incomplete_details` reason.

`max_turns` (default 10; `< 1` throws `ArgumentError`) bounds the round-trips; when
they run out, the result keeps the last response, with `completed=false` and
`llm_error = "max turns (N) exhausted"`.

Per-call `config::RequestConfig` overrides timeouts/retry budget.
"""
function tool_loop(r::Respond, dispatcher::Function;
                   max_turns::Int=10, config::Union{Nothing,RequestConfig}=nothing)::ToolLoopResult
    _check_max_turns(max_turns)
    all_outcomes = ToolCallOutcome[]
    turns = 0
    input = r.input
    prev_id = r.previous_response_id
    local latest::ResponseSuccess

    while turns < max_turns
        turns += 1
        req = _next_respond(r; input, previous_response_id=prev_id)
        raw = respond(req; config)
        result = raw isa Task ? fetch(raw) : raw

        if result isa ResponseFailure
            return ToolLoopResult(result, all_outcomes, turns, false, result.response)
        elseif result isa ResponseCallError
            return ToolLoopResult(result, all_outcomes, turns, false, result.error)
        end

        latest = result
        status = result.response.status
        if status ∉ ("completed", "requires_action")
            details = incomplete_details(result)
            reason = details isa AbstractDict ? get(details, "reason", nothing) : nothing
            return ToolLoopResult(result, all_outcomes, turns, false, "Response did not complete " *
                "(status=$status" * (isnothing(reason) ? ")" : ", reason=$reason)"))
        end

        calls = function_calls(result)

        if isempty(calls)
            completed = status == "completed"
            return ToolLoopResult(result, all_outcomes, turns, completed,
                completed ? nothing : "Response requires an action this tool loop cannot perform")
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
                "name" => name,
                "output" => content
            ))
        end

        input = output_items
        prev_id = result.response.id
    end

    ToolLoopResult(latest, all_outcomes, turns, false, "max turns ($max_turns) exhausted")
end

"""
    tool_loop(r::Respond; max_turns=10, config=nothing) -> ToolLoopResult

No-dispatcher variant: extracts callables from [`CallableTool`](@ref) entries in `r.tools`.

Per-call `config::RequestConfig` overrides timeouts/retry budget.
"""
function tool_loop(r::Respond; max_turns::Int=10, config::Union{Nothing,RequestConfig}=nothing)::ToolLoopResult
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
    tool_loop(r, dispatcher; max_turns, config)
end

"""
    tool_loop(input, dispatcher::Function; tools, kwargs...) -> ToolLoopResult

Convenience form: creates a [`Respond`](@ref) and runs the tool loop.

Per-call `config::RequestConfig` overrides timeouts/retry budget.
"""
function tool_loop(input, dispatcher::Function; kwargs...)
    kws = Dict{Symbol,Any}(kwargs)
    config = pop!(kws, :config, nothing)
    max_turns = pop!(kws, :max_turns, 10)
    r = Respond(; input, kws...)
    tool_loop(r, dispatcher; max_turns, config)
end

"""
    tool_loop(input::String; tools, max_turns=10, config=nothing, kwargs...) -> ToolLoopResult

No-dispatcher convenience form of the Responses-API tool loop for a plain-string prompt.
Wraps `input` and `tools` in a [`Respond`](@ref) and delegates to
[`tool_loop(::Respond)`](@ref), which dispatches each model-requested function call to the
matching [`CallableTool`](@ref) callable.

Keyword routing is explicit: `max_turns` and `config` drive the loop (`config::RequestConfig`
overrides timeouts/retry budget), while every other
keyword is forwarded verbatim to the [`Respond`](@ref) constructor — an unknown keyword
raises there rather than being silently dropped. `tools` is required and must hold
[`CallableTool`](@ref) entries (e.g. from [`mcp_tools_respond`](@ref)).

# Example
```julia
session = mcp_connect("https://mcp.example.com/mcp")
tools = mcp_tools_respond(session)
result = tool_loop("List files in /tmp"; tools=tools)
```
"""
function tool_loop(input::String; tools, max_turns::Int=10, config::Union{Nothing,RequestConfig}=nothing, kwargs...)::ToolLoopResult
    r = Respond(; input, tools, kwargs...)
    tool_loop(r; max_turns, config)
end
