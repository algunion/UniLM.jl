# ─── Typed timeouts + watchdog primitives ────────────────────────────────────
# Blocked reads (eof/readavailable/readline) never return control to an
# in-loop check, so polling cannot bound them. The primitive here is CLOSE:
# closing the guarded resource unblocks the read with an IOError (and killing
# a process delivers EOF). A guard resolves EXACTLY ONCE — :armed → :done |
# :fired via a single atomic CAS — so the winner between completion and breach
# is always well-defined and the close side effect never doubles.

"""
    UniLMTimeout <: Exception

A configured UniLM time bound was exceeded.

# Fields
- `phase::Symbol`: which bound fired — `:connect`, `:request`, `:stream_idle`,
  or `:deadline` (the total budget across attempts).
- `elapsed::Float64`: seconds elapsed when the timeout surfaced (monotonic clock).
- `limit::Float64`: the configured bound in seconds.

Configure the bounds via [`RequestConfig`](@ref). Value-returning surfaces
(chat, embeddings, responses) deliver this inside their error results rather
than throwing; see each surface's documentation.
"""
struct UniLMTimeout <: Exception
    phase::Symbol
    elapsed::Float64
    limit::Float64
end

Base.showerror(io::IO, e::UniLMTimeout) =
    print(io, "UniLMTimeout: ", e.phase, " phase exceeded its ", e.limit,
        " s limit (elapsed ", round(e.elapsed; digits=3), " s)")

"""
    MCPTimeoutError <: Exception

An MCP operation exceeded its configured time bound. The MCP surface is
throw-based, so this surfaces as an exception rather than a failure value.

# Fields
- `phase::Symbol`: `:connect` (spawn → `initialize` handshake) or `:request`
  (one MCP exchange).
- `elapsed::Float64`: seconds elapsed when the timeout fired (monotonic clock).
- `limit::Float64`: the configured bound in seconds.
- `msg::String`: human-readable message naming the applicable override
  (the `mcp_connect_timeout`/`mcp_request_timeout` field or a per-call
  `timeout` keyword).
"""
struct MCPTimeoutError <: Exception
    phase::Symbol
    elapsed::Float64
    limit::Float64
    msg::String
end

Base.showerror(io::IO, e::MCPTimeoutError) =
    print(io, "MCPTimeoutError: ", e.msg, " (", e.phase, " phase, limit ",
        e.limit, " s, elapsed ", round(e.elapsed; digits=3), " s)")

# Watchdog task-mode delivery payload: thrown INTO the worker task on breach,
# then mapped to UniLMTimeout by the wrapper. Never escapes the seam.
struct _DeadlineBreach <: Exception
    phase::Symbol
    limit::Float64
end

_elapsed_s(t0::UInt64)::Float64 = (time_ns() - t0) / 1e9

_remaining_s(cfg::RequestConfig, t0::UInt64)::Float64 =
    cfg.total_deadline == Inf ? Inf : max(cfg.total_deadline - _elapsed_s(t0), 0.0)

# Walk an exception's wrapping chain looking for the first exception matching
# `pred`. An exception delivered into a task mid-request can surface
# arbitrarily nested (TaskFailedException, CompositeException, and the HTTP
# majors' cause-carrying wrappers expose .error/.cause); matching on the chain
# keeps classification independent of which layer caught first.
function _find_exception(pred::Function, e)
    e isa Exception && pred(e) && return e
    if e isa TaskFailedException
        inner = e.task.exception
        if inner isa Exception
            found = _find_exception(pred, inner)
            found !== nothing && return found
        end
    elseif e isa CompositeException
        for inner in e.exceptions
            found = _find_exception(pred, inner)
            found !== nothing && return found
        end
    end
    for name in (:error, :cause)
        if e isa Exception && hasproperty(e, name)
            inner = getproperty(e, name)
            if inner isa Exception
                found = _find_exception(pred, inner)
                found !== nothing && return found
            end
        end
    end
    return nothing
end

# Strip TaskFailedException layers so callers see the worker's own exception type.
function _unwrap_task_failure(e)
    while e isa TaskFailedException && e.task.exception isa Exception
        e = e.task.exception
    end
    return e
end

# Connection-level transport failure shapes on both HTTP majors. Excluded:
# status-carrying errors (a response is an outcome, not a transport failure)
# and native timeout errors (those ride the UniLMTimeout channel via the
# seam's mapping, which carries phase attribution).
function _transport_shaped(x)::Bool
    x isa Base.IOError && return true
    x isa EOFError && return true
    x isa HTTP.HTTPError || return false
    x isa HTTP.StatusError && return false
    x isa HTTP.TimeoutError && return false
    return true
end

"""
    _is_transport_error(e) -> Bool

True when `e` is a connection-level IO failure worth another attempt
(IOError/EOFError/DNS/connect-shaped), unwrapped across `TaskFailedException`,
`CompositeException`, and both HTTP majors' cause-carrying wrappers. Always
false for `InterruptException` (user intent wins, even when nested beside a
transport error), `_DeadlineBreach` and `UniLMTimeout` (timeouts are policy,
classified by phase — never blanket-retried here), and status-carrying errors
(a response is an outcome, not a transport failure).
"""
function _is_transport_error(e)::Bool
    _find_exception(x -> x isa InterruptException, e) !== nothing && return false
    _find_exception(x -> x isa _DeadlineBreach, e) !== nothing && return false
    _find_exception(x -> x isa UniLMTimeout, e) !== nothing && return false
    return _find_exception(_transport_shaped, e) !== nothing
end
