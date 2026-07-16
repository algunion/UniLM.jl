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

# The guard's entire shared state is one atomic Symbol. Timer, clock origin,
# and phase stay locals in the wrapper, so the CAS is the only cross-task
# communication and the :done/:fired winner is decided exactly once.
mutable struct _DeadlineGuard
    @atomic state::Symbol   # :armed → :done | :fired
end

"""
    _with_deadline(f, close!, limit, phase)

Run `f()` under a hard deadline with HANDLE-mode enforcement: if `limit`
seconds elapse first, `close!()` runs exactly once to close the guarded
resource, which unblocks any read stuck inside `f` (blocked reads never
return to a polling check; closing is the only universal unblocker). Returns
`f()`'s value. `limit == Inf` calls `f()` directly with zero overhead.

Resolution is exactly-once via one atomic CAS: completion swaps
`:armed → :done`; the timer swaps `:armed → :fired`, and only the winner
acts. If `f` returns after the guard fired (completion ≈ breach race) the
result is real and is returned — the close side effect stands, and any
long-lived state so closed must remain closed. If `f` throws:
`InterruptException` always rethrows first; with the guard `:fired` the error
is the echo of our own close and `UniLMTimeout(phase, …)` is thrown instead;
otherwise the original error rethrows. A throwing `close!` is debug-logged,
never propagated. The timer is always closed on exit.
"""
function _with_deadline(f::Function, close!::Function, limit::Float64, phase::Symbol)
    limit == Inf && return f()
    t0 = time_ns()
    guard = _DeadlineGuard(:armed)
    timer = Timer(limit) do _
        (@atomicreplace guard.state :armed => :fired).success || return
        try
            close!()
        catch e
            @debug "deadline close! failed" phase exception = (e, catch_backtrace())
        end
    end
    try
        result = f()
        @atomicreplace guard.state :armed => :done
        # A lost swap means the guard fired as f completed; the result is
        # real either way — return it.
        return result
    catch e
        e isa InterruptException && rethrow()
        if (@atomicreplace guard.state :armed => :done).success
            rethrow()   # real failure; the guard resolved :done
        else
            # :fired won — the caught error is the echo of our own close.
            throw(UniLMTimeout(phase, _elapsed_s(t0), limit))
        end
    finally
        close(timer)
    end
end

"""
    _with_deadline_task(f, limit, phase)

Run `f` under a hard deadline with TASK-mode enforcement, for opaque calls
that expose no closeable handle: `f` runs in its own task, and on breach the
guard delivers a private breach exception into it via
`schedule(task, exc; error=true)` (IO-blocked tasks are parked at a yield
point, so delivery unblocks them). The wrapper fetches the task and maps a
breach-caused failure to `UniLMTimeout(phase, …)`. `InterruptException`
rethrows first, whether it hit the wrapper or the worker. Other worker
failures rethrow the worker's own exception (`TaskFailedException`
unwrapped). `limit == Inf` calls `f()` directly. The timer is always closed
on exit.

Known limit: a worker inside an uninterruptible foreign call is unblocked
only at its waiter; the foreign call itself ends on the OS's schedule.
"""
function _with_deadline_task(f::Function, limit::Float64, phase::Symbol)
    limit == Inf && return f()
    t0 = time_ns()
    guard = _DeadlineGuard(:armed)
    task = Threads.@spawn f()
    timer = Timer(limit) do _
        (@atomicreplace guard.state :armed => :fired).success || return
        try
            schedule(task, _DeadlineBreach(phase, limit); error=true)
        catch e
            # Undeliverable (worker already finishing, or not yet parked at a
            # yield point): the fetch below settles it — a real result is
            # returned as real.
            @debug "deadline breach delivery failed" phase exception = (e, catch_backtrace())
        end
    end
    try
        result = fetch(task)
        @atomicreplace guard.state :armed => :done
        return result
    catch e
        e isa InterruptException && rethrow()
        interrupted = _find_exception(x -> x isa InterruptException, e)
        interrupted === nothing || throw(interrupted)
        if _find_exception(x -> x isa _DeadlineBreach, e) !== nothing
            throw(UniLMTimeout(phase, _elapsed_s(t0), limit))
        end
        @atomicreplace guard.state :armed => :done
        throw(_unwrap_task_failure(e))
    finally
        close(timer)
    end
end

# Stream idle guard: a byte-gap watchdog. ONE periodic timer per stream (no
# per-chunk allocation); _touch! stamps the monotonic clock on every raw
# chunk, so SSE comments and keep-alive pings reset the clock by
# construction. Breach = gap > limit, checked every period = min(limit/4, 5) s
# — it fires within [limit, limit + period]. Resolution shares the
# exactly-once CAS discipline: only the :armed → :fired winner runs close!().
mutable struct _IdleGuard
    @atomic state::Symbol       # :armed → :fired | :disarmed
    @atomic last_byte::UInt64   # time_ns() of the most recent raw chunk
    @atomic fired_gap::Float64  # the byte gap recorded at breach time (s)
    const limit::Float64
    timer::Union{Timer,Nothing}
end

"""
    _idle_guard(close!, limit) -> guard

Arm a byte-gap idle watchdog: once more than `limit` seconds pass without a
[`_touch!`](@ref), `close!()` runs exactly once and `_idle_fired(guard)`
turns true, with the breaching gap frozen for `_idle_gap_s(guard)` (an idle
timeout is ABOUT the gap, so the gap — not whole-call time — is what error
reporting surfaces as elapsed). Returns `nothing` when `limit == Inf` (all
guard operations no-op on `nothing`). Call `_touch!(guard)` after every raw
chunk and `_disarm!(guard)` on every exit path.
"""
function _idle_guard(close!::Function, limit::Float64)
    limit == Inf && return nothing
    guard = _IdleGuard(:armed, time_ns(), 0.0, limit, nothing)
    period = min(limit / 4, 5.0)
    guard.timer = Timer(period; interval=period) do timer
        gap = (time_ns() - @atomic(guard.last_byte)) / 1e9
        gap > guard.limit || return
        # Record the gap BEFORE resolving: any reader that observes :fired
        # then observes the recorded gap (a losing write is unobservable —
        # the accessor gates on :fired).
        @atomic guard.fired_gap = gap
        (@atomicreplace guard.state :armed => :fired).success || return
        try
            close!()
        catch e
            @debug "idle-guard close! failed" exception = (e, catch_backtrace())
        end
        close(timer)   # resolved; stop the periodic ticks
    end
    return guard
end

_touch!(::Nothing) = nothing
_touch!(guard::_IdleGuard)::Nothing = (@atomic guard.last_byte = time_ns(); nothing)

_idle_fired(::Nothing) = false
_idle_fired(guard::_IdleGuard)::Bool = (@atomic guard.state) === :fired

_idle_gap_s(::Nothing) = 0.0
function _idle_gap_s(guard::_IdleGuard)::Float64
    (@atomic guard.state) === :fired && return @atomic guard.fired_gap
    return (time_ns() - @atomic(guard.last_byte)) / 1e9
end

_disarm!(::Nothing) = nothing
function _disarm!(guard::_IdleGuard)::Nothing
    # Losing this CAS to :fired is fine — the resolution stands; disarming is
    # only a promise that the guard will never fire in the FUTURE.
    @atomicreplace guard.state :armed => :disarmed
    timer = guard.timer
    timer === nothing || close(timer)
    return nothing
end
