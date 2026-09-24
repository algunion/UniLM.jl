# ─── Cooperative cancellation ────────────────────────────────────────────────
# A CancelToken is a level-triggered, idempotent flag plus the hooks that turn
# its flip into action (cancelling a request context, waking a waiter).
# Operations take an explicit `cancel` keyword or inherit the ambient token set
# by `with_cancel` — a ScopedValue, the same resolution idiom as RequestConfig.

"""
    UniLMCancelled <: Exception

An operation was cancelled before it completed.

# Fields
- `source::Symbol`: `:token` (a [`cancel!`](@ref) on the operation's
  [`CancelToken`](@ref)) or `:callback` (a streaming callback set `close[] = true`).
- `elapsed::Float64`: seconds since the operation started (monotonic clock).

```julia
julia> sprint(showerror, UniLMCancelled(:token, 0.25))
"UniLMCancelled: cancelled by token after 0.25 s"
```
"""
struct UniLMCancelled <: Exception
    source::Symbol
    elapsed::Float64
    function UniLMCancelled(source::Symbol, elapsed::Real)
        source in (:token, :callback) ||
            throw(ArgumentError("UniLMCancelled source must be :token or :callback (got :$source)"))
        return new(source, Float64(elapsed))
    end
end

Base.showerror(io::IO, e::UniLMCancelled) =
    print(io, "UniLMCancelled: cancelled by ", e.source, " after ", round(e.elapsed; digits=3), " s")

"""
    CancelToken()

A cooperative cancellation token: make it ambient with [`with_cancel`](@ref) (or
pass it where an operation takes a `cancel` keyword), then call [`cancel!`](@ref)
from any task. Cancellation is level-triggered — a cancelled token stays
cancelled, and every later request that sees it stops before any network I/O —
so use a fresh token per unit of work you may want to abandon.

A token reaches what goes through the package's HTTP layer: every HTTP verb,
streams included, and the pauses between retries and polls. The Realtime WebSocket
([`realtime_connect`](@ref), [`realtime_receive`](@ref)) and MCP stdio exchanges do
not observe it.

The Julia 1.14 development branch adds task cancellation built on scoped cancellation
tokens (`Base.CancellationTokenSource` / `Base.CANCEL_TOKEN`) with a `cancel` keyword
on blocking operations; this API has the same shape so a later version can bridge
to it.

```julia
tok = CancelToken()
t = Threads.@spawn with_cancel(() -> chatrequest!(chat), tok)
cancel!(tok)          # e.g. the user pressed "stop"
```
"""
mutable struct CancelToken
    @atomic cancelled::Bool
    const lock::ReentrantLock
    const hooks::Vector{Any}   # one Ref per registration: its identity is the handle
    CancelToken() = new(false, ReentrantLock(), Any[])
end

"""
    cancel!(tok::CancelToken) -> CancelToken

Cancel `tok`. The first call flips the flag and then runs every registered hook
exactly once, outside the token's lock. A hook that throws is `@debug`-logged and the
rest still run; an `InterruptException` from a hook is rethrown once every hook has
run (the first one, if several). Later calls are no-ops.

```julia
tok = CancelToken()
cancel!(tok); cancel!(tok)   # idempotent
iscancelled(tok)             # true
```
"""
function cancel!(tok::CancelToken)::CancelToken
    (@atomicreplace tok.cancelled false => true).success || return tok
    hooks = @lock tok.lock splice!(tok.hooks, eachindex(tok.hooks))
    # The hooks have left the token: one skipped here would never run, so an interrupt
    # waits until they all have.
    interrupt = nothing
    for h in hooks
        try
            _run_cancel_hook(h[])
        catch e
            e isa InterruptException || rethrow()
            interrupt = something(interrupt, e)
        end
    end
    isnothing(interrupt) || throw(interrupt)
    return tok
end

"""
    iscancelled(tok::Union{Nothing,CancelToken}) -> Bool

Whether `tok` has been cancelled; `false` for `nothing` (no token).

```julia
iscancelled(CancelToken())   # false
iscancelled(nothing)         # false
```
"""
iscancelled(tok::CancelToken)::Bool = @atomic tok.cancelled
iscancelled(::Nothing) = false

const _CANCEL_TOKEN = ScopedValue{Union{Nothing,CancelToken}}(nothing)

"""
    with_cancel(f, tok::CancelToken)

Run `f()` with `tok` as the ambient cancellation token and return `f()`'s value.
What `f` sends through the package's HTTP layer observes `tok` — every HTTP verb,
streams included, and the pauses between retries and polls — including requests made
in tasks spawned inside `f` (scoped values propagate into `Threads.@spawn`); the
innermost scope wins. The Realtime WebSocket and MCP stdio exchanges do not observe
it. The Julia 1.14 development branch adds task cancellation built on scoped
cancellation tokens (`Base.CancellationTokenSource` / `Base.CANCEL_TOKEN`) with a
`cancel` keyword on blocking operations; this API has the same shape so a later
version can bridge to it.

```julia
tok = CancelToken()
with_cancel(tok) do
    chatrequest!(chat)       # cancellable through tok
end
```
"""
with_cancel(f::Function, tok::CancelToken) = with(f, _CANCEL_TOKEN => tok)

# The ambient token: the innermost active `with_cancel` scope, else nothing.
_current_cancel()::Union{Nothing,CancelToken} = _CANCEL_TOKEN[]

# Per-call resolution: an explicit token wins over the ambient one.
_resolve_cancel(kw::Union{Nothing,CancelToken})::Union{Nothing,CancelToken} =
    kw === nothing ? _current_cancel() : kw

function _run_cancel_hook(f)::Nothing
    try
        f()
    catch e
        e isa InterruptException && rethrow()
        @debug "cancel hook failed" exception = (e, catch_backtrace())
    end
    return nothing
end

# Register `f` to run once when `tok` is cancelled; returns the handle for
# `_off_cancel`. Already cancelled: `f` runs now, in the caller, and the handle
# is `nothing`. The flag is re-read under the lock `cancel!` takes to collect the
# hooks, so a registration racing `cancel!` either lands in its batch or sees the
# flag and runs here — exactly once either way.
function _on_cancel(f::Function, tok::CancelToken)
    hook = Ref{Function}(f)
    registered = @lock tok.lock (iscancelled(tok) ? false : (push!(tok.hooks, hook); true))
    registered && return hook
    _run_cancel_hook(f)
    return nothing
end
_on_cancel(f::Function, ::Nothing) = nothing

# Idempotent deregistration by identity; every seam registration is undone in a
# `finally`, so a token reused across many calls never accumulates hooks.
function _off_cancel(tok::CancelToken, handle)::Nothing
    handle === nothing && return nothing
    @lock tok.lock begin
        i = findfirst(h -> h === handle, tok.hooks)
        i === nothing || deleteat!(tok.hooks, i)
    end
    return nothing
end
_off_cancel(::Nothing, _) = nothing

# Sleep up to `seconds`, waking at once on cancel; true iff `tok` is cancelled.
# Event-driven (a one-shot timer and a cancel hook race to notify), never polled.
# The timer is closed off the caller's path: `close(::Timer)` waits for the event
# loop's close handshake, which a busy loop thread can stall.
function _cancel_sleep(tok::CancelToken, seconds::Real)::Bool
    iscancelled(tok) && return true
    wake = Base.Event()
    timer = Timer(_ -> notify(wake), seconds; spawn=true)
    handle = _on_cancel(() -> notify(wake), tok)
    try
        wait(wake)
    finally
        _off_cancel(tok, handle)
        errormonitor(Threads.@spawn :default close(timer))
    end
    return iscancelled(tok)
end
_cancel_sleep(::Nothing, seconds::Real)::Bool = (sleep(seconds); false)
