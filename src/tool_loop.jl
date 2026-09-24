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
  `max_turns` ran out — the last tool-call turn (whose calls were dispatched). A loop
  cancelled between tool dispatches holds a call error whose `cause` is
  [`UniLMCancelled`](@ref).
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
The model reads the result as text: a `String` is passed through, any other value
is JSON-encoded (`JSON.json`), never sent as its Julia `repr`.
"""
function _dispatch_tool(name::String, args::Dict{String,Any}, dispatcher::Function)::ToolCallOutcome
    try
        out = dispatcher(name, args)
        result_str = out isa AbstractString ? String(out) : JSON.json(out)
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

# A dispatcher routing each call to the `CallableTool` of that name among `tools`;
# a name with no tool is a tool error the model sees ("Unknown tool: …").
function _callable_dispatcher(tools)::Function
    table = Dict{String,Function}(_tool_name(t) => t.callable for t in tools if t isa CallableTool)
    (name, args) -> haskey(table, name) ? table[name](name, args) : error("Unknown tool: $name")
end

_check_max_turns(n::Int) =
    ispositive(n) || throw(ArgumentError("max_turns must be >= 1 (got $n)"))
_check_concurrency(n::Int) =
    ispositive(n) || throw(ArgumentError("tool_concurrency must be >= 1 (got $n)"))

# The text a tool-call outcome sends back to the model.
_tool_output(o::ToolCallOutcome)::String = o.success ? string(o.result.result) : "Error: $(o.error)"

# Dispatch `calls` and hand each `(call, outcome)` to `emit` in call order; `false` when
# a cancelled `tok` stopped the hand-out of calls first. `n == 1` runs them one at a time
# in this task. `n > 1` runs up to `n` at once on spawned tasks and emits after all have
# finished, so the next request does not depend on completion order. An exception
# escaping a dispatch (`_dispatch_tool` lets only an InterruptException through)
# propagates once the in-flight dispatches are done.
function _run_calls(emit::Function, dispatch::Function, calls::AbstractVector,
                    tok::Union{Nothing,CancelToken}, n::Int)::Bool
    if n == 1
        for c in calls
            iscancelled(tok) && return false
            emit(c, dispatch(c))
        end
        return true
    end
    slots = Vector{Union{Nothing,ToolCallOutcome}}(nothing, length(calls))
    next = Threads.Atomic{Int}(1)
    halt = Threads.Atomic{Bool}(false)
    # A worker leaves once no call is left to hand out, on a cancel, or on a failure;
    # in each case no other worker should take a further call.
    worker() = try
        while !(halt[] || iscancelled(tok))
            i = Threads.atomic_add!(next, 1)
            i > length(calls) && break
            slots[i] = dispatch(calls[i])
        end
    finally
        halt[] = true
    end
    workers = [Threads.@spawn(worker()) for _ in 1:min(n, length(calls))]
    try
        waitall(workers; failfast=false, throw=false)
    finally
        halt[] = true   # an interrupt delivered to this task stops the hand-out as well
    end
    failed = findfirst(istaskfailed, workers)
    isnothing(failed) || throw(workers[failed].exception)
    ran = something(findfirst(isnothing, slots), length(slots) + 1) - 1   # a prefix ran
    foreach(i -> emit(calls[i], slots[i]::ToolCallOutcome), 1:ran)
    return ran == length(calls)
end

# Run `f` with `tok` as the ambient token: a turn's request and the tools it dispatches
# (including tasks they spawn) observe it. No token: run `f` as is.
_in_cancel_scope(f::Function, tok::CancelToken) = with_cancel(f, tok)
_in_cancel_scope(f::Function, ::Nothing) = f()

# The cause for a loop cancelled between tool dispatches (`t0`: the loop's start); the
# loop wraps it in the call-error type a cancelled request of that API returns.
_cancelled_since(t0::UInt64) = UniLMCancelled(:token, _elapsed_s(t0))

# ─── Chat Completions Loop ──────────────────────────────────────────────────

"""
    tool_loop!(chat::Chat, dispatcher::Function; max_turns=10, config=nothing, callback=nothing, on_tool_call=nothing, cancel=nothing, tool_concurrency=1) -> ToolLoopResult

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
- `dispatcher`: `(name::String, args::Dict{String,Any}) -> String`; any other
  return value is sent to the model JSON-encoded.
- `max_turns`: Maximum API round-trips (default 10; `< 1` throws `ArgumentError`).
  When they run out, the result keeps the last response, with `completed=false` and
  `llm_error = "max turns (N) exhausted"`.
- `config`: Per-request [`RequestConfig`](@ref) passed to [`chatrequest!`](@ref) — each turn gets its own attempt/deadline budget.
- `callback`: Streaming callback passed to `chatrequest!`.
- `on_tool_call`: Tool call notification callback passed to `chatrequest!`.
- `cancel`: A [`CancelToken`](@ref); `nothing` (default) uses the ambient token of
  [`with_cancel`](@ref). Every turn — its request and its tool dispatches — runs inside
  `with_cancel(cancel)`, and the token is checked before each dispatch. A cancelled loop
  returns promptly with `completed=false` and dispatches nothing further; `response` is
  the turn's typed cancellation result: the cancelled request's `LLMCallError`, or, when
  the cancel lands between dispatches, an `LLMCallError` whose `cause` is
  [`UniLMCancelled`](@ref). A turn cancelled between dispatches is removed from `chat`
  (the calls it did run stay in `tool_calls`), so the conversation stays sendable.
- `tool_concurrency`: How many of one turn's tool calls may run at once (default 1:
  one at a time, in the calling task; `< 1` throws `ArgumentError`). Above 1, up to that
  many run concurrently on spawned tasks (`Threads.@spawn`), so `dispatcher` must be
  thread-safe; their results are appended in call order once all have finished, so the
  next request does not depend on completion order. An `InterruptException` from any
  dispatch propagates.

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
                    callback=nothing, on_tool_call=nothing,
                    cancel::Union{Nothing,CancelToken}=nothing,
                    tool_concurrency::Int=1)::ToolLoopResult
    _check_max_turns(max_turns)
    _check_concurrency(tool_concurrency)
    chat.history || throw(ArgumentError("tool_loop! needs a Chat with history=true: " *
        "each follow-up request must carry the assistant turn its tool results answer"))
    tok = _resolve_cancel(cancel)
    t0 = time_ns()
    _in_cancel_scope(tok) do
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

            ran = _run_calls(tc -> _dispatch_tool(tc.func.name, tc.func.arguments, dispatcher),
                             calls, tok, tool_concurrency) do tc, outcome
                push!(all_outcomes, outcome)
                push!(chat, Message(role=RoleTool, content=_tool_output(outcome), tool_call_id=tc.id))
            end
            if !ran
                resize!(chat.messages, before)   # its remaining calls will never be answered
                c = _cancelled_since(t0)
                err = LLMCallError(error=sprint(showerror, c), self=chat, cause=c)
                return ToolLoopResult(err, all_outcomes, turns, false, err.error)
            end
        end

        ToolLoopResult(latest, all_outcomes, turns, false, "max turns ($max_turns) exhausted")
    end
end

"""
    tool_loop!(chat::Chat; tools::Vector{<:CallableTool}, kwargs...) -> ToolLoopResult

No-dispatcher variant: builds a dispatcher from [`CallableTool`](@ref) entries.
"""
function tool_loop!(chat::Chat; tools::Vector{<:CallableTool},
                    max_turns::Int=10, config::Union{Nothing,RequestConfig}=nothing,
                    callback=nothing, on_tool_call=nothing,
                    cancel::Union{Nothing,CancelToken}=nothing,
                    tool_concurrency::Int=1)::ToolLoopResult
    tool_loop!(chat, _callable_dispatcher(tools);
               max_turns, config, callback, on_tool_call, cancel, tool_concurrency)
end

# ─── Responses API Loop ─────────────────────────────────────────────────────

# Output items the client must execute or answer, which this loop cannot: a turn holding
# one cannot be continued with function-call outputs alone. An `mcp_approval_request`
# waits for an `mcp_approval_response`. A `shell_call` is the client's only when the
# platform did not run it: a hosted shell (a container environment) answers its own
# call with a `shell_call_output` in the same output, a local one does not.
const _CLIENT_CALL_TYPES = ("custom_tool_call", "apply_patch_call", "local_shell_call", "computer_call",
                            "shell_call", "mcp_approval_request")

# The client-side call types a Responses turn leaves pending, each named once.
function _pending_client_calls(output)::Vector{String}
    items = [item for item in output if item isa AbstractDict]
    answered = Set(get(item, "call_id", nothing) for item in items if get(item, "type", "") == "shell_call_output")
    unique!(String[item["type"] for item in items if get(item, "type", "") in _CLIENT_CALL_TYPES &&
                   !(item["type"] == "shell_call" && get(item, "call_id", nothing) in answered)])
end

# One Responses function call → its outcome. `arguments` that do not parse to a JSON
# object are the model's error to correct, so they come back as a failed outcome (sent
# to the model as that call's output) instead of escaping the loop.
function _dispatch_call(call::AbstractDict, dispatcher::Function)::ToolCallOutcome
    name = call["name"]
    args = try
        JSON.parse(call["arguments"]; dicttype=Dict{String,Any})
    catch e
        e isa InterruptException && rethrow()
        return ToolCallOutcome(name, Dict{String,Any}(), nothing, false,
                               "invalid arguments: " * sprint(showerror, e))
    end
    args isa Dict{String,Any} || return ToolCallOutcome(name, Dict{String,Any}(), nothing, false,
        "invalid arguments: expected a JSON object, got $(repr(call["arguments"]))")
    _dispatch_tool(name, args, dispatcher)
end

"""Reconstruct a [`Respond`](@ref) with new `input` and `previous_response_id`, copying all other fields.
Streaming is always disabled in the tool loop."""
function _next_respond(r::Respond; input, previous_response_id=nothing)
    kept = (f => getfield(r, f) for f in fieldnames(Respond)
            if f ∉ (:input, :previous_response_id, :stream))
    Respond(; input, previous_response_id, stream=nothing, kept...)
end

"""
    tool_loop(r::Respond, dispatcher::Function; max_turns=10, config=nothing, cancel=nothing, tool_concurrency=1) -> ToolLoopResult

Run a tool-calling loop on a [`Respond`](@ref) request. Dispatches function calls
via `dispatcher(name, args)` (a non-`String` return is sent JSON-encoded), builds
`function_call_output` input items, and chains via `previous_response_id` — or, when
`r.conversation` is set, through the conversation alone (the API rejects the two
together). A call whose `arguments` are not a JSON object is answered with an
`"Error: invalid arguments: …"` output, like a dispatcher error, and the loop goes on.

Function calls run only on a `completed` or `requires_action` turn. Any other status
(e.g. `incomplete`, whose calls may be partial) stops the loop with `completed=false`
and an `llm_error` naming the status and the `incomplete_details` reason. A turn that
requests a client-side call this loop cannot execute (`custom_tool_call`,
`apply_patch_call`, `local_shell_call`, `computer_call`, a `shell_call` the platform
did not run itself, or an `mcp_approval_request`) stops it the same way, with the
pending call type in `llm_error`; none of that turn's calls run.

`max_turns` (default 10; `< 1` throws `ArgumentError`) bounds the round-trips; when
they run out, the result keeps the last response, with `completed=false` and
`llm_error = "max turns (N) exhausted"`.

`cancel` works as in [`tool_loop!`](@ref): every turn runs inside `with_cancel(cancel)`
(`nothing` uses the ambient token), the token is checked before each dispatch, and a
cancelled loop returns promptly with `completed=false` and the turn's typed
cancellation result — the cancelled request's `ResponseCallError`, or, between
dispatches, a `ResponseCallError` whose `cause` is [`UniLMCancelled`](@ref).
`tool_concurrency` (default 1) works as in [`tool_loop!`](@ref): above 1, up to that
many of a turn's calls run at once and their outputs are sent in call order.

Per-call `config::RequestConfig` overrides timeouts/retry budget.
"""
function tool_loop(r::Respond, dispatcher::Function;
                   max_turns::Int=10, config::Union{Nothing,RequestConfig}=nothing,
                   cancel::Union{Nothing,CancelToken}=nothing,
                   tool_concurrency::Int=1)::ToolLoopResult
    _check_max_turns(max_turns)
    _check_concurrency(tool_concurrency)
    tok = _resolve_cancel(cancel)
    t0 = time_ns()
    _in_cancel_scope(tok) do
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

            pending = _pending_client_calls(result.response.output)
            isempty(pending) || return ToolLoopResult(result, all_outcomes, turns, false,
                "Response requests $(join(pending, ", ")), which this tool loop cannot execute")

            calls = function_calls(result)

            if isempty(calls)
                completed = status == "completed"
                return ToolLoopResult(result, all_outcomes, turns, completed,
                    completed ? nothing : "Response requires an action this tool loop cannot perform")
            end

            output_items = Any[]
            ran = _run_calls(call -> _dispatch_call(call, dispatcher), calls, tok,
                             tool_concurrency) do call, outcome
                push!(all_outcomes, outcome)
                push!(output_items, Dict{String,Any}(
                    "type" => "function_call_output",
                    "call_id" => call["call_id"],
                    "name" => call["name"],
                    "output" => _tool_output(outcome)
                ))
            end
            if !ran
                c = _cancelled_since(t0)
                err = ResponseCallError(error=sprint(showerror, c), cause=c)
                return ToolLoopResult(err, all_outcomes, turns, false, err.error)
            end

            input = output_items
            prev_id = isnothing(r.conversation) ? result.response.id : nothing
        end

        ToolLoopResult(latest, all_outcomes, turns, false, "max turns ($max_turns) exhausted")
    end
end

"""
    tool_loop(r::Respond; max_turns=10, config=nothing, cancel=nothing, tool_concurrency=1) -> ToolLoopResult

No-dispatcher variant: extracts callables from [`CallableTool`](@ref) entries in `r.tools`.

Per-call `config::RequestConfig` overrides timeouts/retry budget.
"""
function tool_loop(r::Respond; max_turns::Int=10, config::Union{Nothing,RequestConfig}=nothing,
                   cancel::Union{Nothing,CancelToken}=nothing,
                   tool_concurrency::Int=1)::ToolLoopResult
    tools = something(r.tools, [])
    any(t -> t isa CallableTool, tools) || throw(ArgumentError("No CallableTool entries found in tools"))
    tool_loop(r, _callable_dispatcher(tools); max_turns, config, cancel, tool_concurrency)
end

"""
    tool_loop(input, dispatcher::Function; tools, kwargs...) -> ToolLoopResult

Convenience form: creates a [`Respond`](@ref) and runs the tool loop.

Per-call `config::RequestConfig` overrides timeouts/retry budget; `max_turns`, `cancel`
and `tool_concurrency` drive the loop as in [`tool_loop(::Respond, ::Function)`](@ref);
every other keyword goes to the [`Respond`](@ref) constructor.
"""
function tool_loop(input, dispatcher::Function; max_turns::Int=10,
                   config::Union{Nothing,RequestConfig}=nothing,
                   cancel::Union{Nothing,CancelToken}=nothing, tool_concurrency::Int=1,
                   kwargs...)::ToolLoopResult
    tool_loop(Respond(; input, kwargs...), dispatcher; max_turns, config, cancel, tool_concurrency)
end

"""
    tool_loop(input::String; tools, max_turns=10, config=nothing, cancel=nothing, tool_concurrency=1, kwargs...) -> ToolLoopResult

No-dispatcher convenience form of the Responses-API tool loop for a plain-string prompt.
Wraps `input` and `tools` in a [`Respond`](@ref) and delegates to
[`tool_loop(::Respond)`](@ref), which dispatches each model-requested function call to the
matching [`CallableTool`](@ref) callable.

Keyword routing is explicit: `max_turns`, `config`, `cancel` and `tool_concurrency`
drive the loop (`config::RequestConfig` overrides timeouts/retry budget), while every other
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
function tool_loop(input::String; tools, max_turns::Int=10, config::Union{Nothing,RequestConfig}=nothing,
                   cancel::Union{Nothing,CancelToken}=nothing, tool_concurrency::Int=1,
                   kwargs...)::ToolLoopResult
    r = Respond(; input, tools, kwargs...)
    tool_loop(r; max_turns, config, cancel, tool_concurrency)
end
