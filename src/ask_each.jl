# ============================================================================
# ask_each — the same questions about many states, or many requests, sent
# concurrently and returned in order. Each backend's transport (its timeouts,
# retry header and default batch width) is set by its protocol's hooks, in its
# own file; the hooks and the connection pool are in system_one.jl.
# ============================================================================

# ─── Shared backoff ──────────────────────────────────────────────────────────

# The backoff the requests of one batch share. When the service answers any of them with
# 429 or 529 — an attempt the retry seam goes on to retry included — no attempt of the
# batch starts until the wait that response asked for has passed: `retry-after-ms`, else
# `Retry-After` (read as `SystemOneFailure.retry_after` reads them), else the longest
# backoff the seam draws for that attempt. Requests already in flight finish; the others
# would otherwise go on sending to a service that has just said it is overloaded. The
# retry seam waits it out before each attempt within what remains of the request's own
# `total_deadline`, and reports a deadline that runs out while waiting as it reports any.
mutable struct _SharedBackoff
    @atomic until::UInt64       # the time_ns() before which no attempt starts; 0 for none
end
_SharedBackoff() = _SharedBackoff(UInt64(0))

function _held_for(b::_SharedBackoff)::Float64
    until = @atomic b.until
    now = time_ns()
    until > now ? (until - now) / 1e9 : 0.0
end

function _back_off!(b::_SharedBackoff, attempt::Int, resp::HTTP.Response)::Nothing
    resp.status == 429 || resp.status == 529 || return nothing
    asked = something(_retry_after(resp), min(_RETRY_BASE * _RETRY_FACTOR^(attempt - 1), _RETRY_MAX_DELAY))
    @atomic b.until max time_ns() + round(UInt64, min(asked, _MAX_FINITE_TIMEOUT) * 1e9)
    nothing
end

# ─── ask_each ────────────────────────────────────────────────────────────────

const _SystemOneResult = Union{SystemOneSuccess,SystemOneFailure,SystemOneCallError}

# One batch in flight: its items, resolved and checked but not encoded, the slot each
# result lands in, the token that stops it, its shared backoff, the caller's `on_result`
# and the lock its calls take turns under, the next item to claim, and the first exception
# that must reach the caller once every worker has stopped.
mutable struct _EachBatch
    const items::Vector{_SystemOneResolved}
    const results::Vector{_SystemOneResult}
    const config::RequestConfig
    const token::CancelToken
    const backoff::_SharedBackoff
    const on_result::Any
    const reporting::ReentrantLock
    @atomic next::Int
    @atomic failure::Union{Nothing,Some{Any}}
end

"""
    ask_each(states::AbstractVector, questions...; model=nothing, service=nothing, config=nothing,
             cancel=nothing, ntasks=nothing, on_result=nothing, options...)
    ask_each(requests::AbstractVector{<:SystemOneRequest}; service=nothing, config=nothing,
             cancel=nothing, ntasks=nothing, on_result=nothing)

Ask the same questions about every state — or send every request — concurrently,
and return a `Vector{Union{SystemOneSuccess,SystemOneFailure,SystemOneCallError}}`
whose `i`-th entry is the result for item `i`: what [`ask`](@ref) returns for
that item alone, with the same answers. Each item is a request of its own;
nothing is packed together.

`questions`, `model`, `service`, `config` and any other keyword — a provider
option — mean what they mean for [`ask`](@ref), and the backend and model
resolve as they do there, an enclosing [`with_system_one`](@ref) scope included.
Every request is built and checked by its backend's wrapper before the first one
is sent: an item that cannot be sent is an `ArgumentError` naming its position,
and nothing is sent. The checked request is not kept: the worker that sends an
item encodes it again, so a batch holds no more request bodies — images and all —
than it has requests in flight.

At most `ntasks` requests are in flight at once — any `Integer` of at least 1;
the default is the backend's — 16 on TypeSafe's wire (a TypeSafe-compatible
gateway included) and OpenAI's Decisions API, 1 on Ollama, 8 on a protocol of
your own — a default, not a limit. The requests
share one connection pool, and when the service answers any of them with 429 or
529, a retried attempt included, no attempt of the batch starts until the wait
that response asked for (`retry-after-ms`, else `Retry-After`, else the retry
seam's own backoff) has passed; requests already in flight finish. A request
waits at most what remains of its own `total_deadline`, whose clock starts when a
worker takes it: one whose budget runs out while it waits is not sent, and its
result is a [`SystemOneCallError`](@ref) whose `cause` is a
[`UniLMTimeout`](@ref) of phase `:deadline`.

`on_result(i, result)`, when given — any callable of an index and each kind of
result, checked before anything is sent — runs in the worker task as each result
lands, in the order they complete, one call at a time.

`cancel` (default: the ambient [`with_cancel`](@ref) token) stops the batch: the
requests in flight end at once, the rest are not sent, and each item still
unanswered becomes a [`SystemOneCallError`](@ref) whose `cause` is a
[`UniLMCancelled`](@ref), as for `ask`. A [`ReplayMissError`](@ref) or
[`RecordingWriteError`](@ref) inside [`with_recorded_answers`](@ref), an
`InterruptException`, or an exception thrown by `on_result` stops it as well and
is rethrown — the first one — once every worker has stopped.

```julia
tickets = ["My order arrived broken, I want a refund.", "The app crashes when I log in.",
           "Do you ship to Canada?"]
results = ask_each(tickets, "team" => choice(["billing", "technical", "shipping"]))
[issuccess(r) ? r["team"].choice : nothing for r in results]
```
"""
function ask_each(states::AbstractVector, question, questions...;
                  model::Union{Nothing,AbstractString}=nothing,
                  service::Union{Nothing,ServiceEndpointSpec}=nothing,
                  config::Union{Nothing,RequestConfig}=nothing,
                  cancel::Union{Nothing,CancelToken}=nothing,
                  ntasks::Union{Nothing,Integer}=nothing,
                  on_result=nothing,
                  options...)
    qs = _normalize_varargs((question, questions...))
    _check_question_names(qs)
    m, opts, n = _model_name(model), values(options), length(states)
    requests = SystemOneRequest[_each_item(() -> SystemOneRequest(s, qs, m, opts), i, n)
                                for (i, s) in enumerate(states)]
    ask_each(requests; service, config, cancel, ntasks, on_result)
end

function ask_each(requests::AbstractVector{<:SystemOneRequest};
                  service::Union{Nothing,ServiceEndpointSpec}=nothing,
                  config::Union{Nothing,RequestConfig}=nothing,
                  cancel::Union{Nothing,CancelToken}=nothing,
                  ntasks::Union{Nothing,Integer}=nothing,
                  on_result=nothing)
    width = _each_width(ntasks)
    _each_check_on_result(on_result)
    svc = first(_system_one_target(service, nothing))
    _check_system_one_service(svc)
    n = length(requests)
    items = _SystemOneResolved[_each_item(() -> _each_checked(r, service), i, n) for (i, r) in enumerate(requests)]
    results = Vector{_SystemOneResult}(undef, n)
    n == 0 && return results
    p = system_one_protocol(svc)::SystemOneProtocol
    batch = _EachBatch(items, results, _resolve_config(config, _system_one_default_config(p, svc)), CancelToken(),
                       _SharedBackoff(), on_result, ReentrantLock(), 0, nothing)
    _run_each(batch, _resolve_cancel(cancel), min(something(width, _system_one_concurrency(p, svc)), n))
end

# The batch width a call names: any Integer of at least 1, as an Int. A Bool is an Integer
# in Julia, but `ntasks = true` names no width: it is refused rather than read as 1.
_each_width(::Nothing)::Nothing = nothing
_each_width(n::Bool) = throw(ArgumentError(
    "`ntasks` is how many requests ask_each keeps in flight, an Integer of at least 1; got $n"))
_each_width(n::Integer)::Int = 1 <= n <= typemax(Int) ? Int(n) : throw(ArgumentError(
    "`ntasks` is how many requests ask_each keeps in flight, an Integer of at least 1; got $n"))

# `on_result` is called with an item's index and its result, whichever kind it is: one
# that cannot take each would fail only once a result had been paid for.
function _each_check_on_result(f)::Nothing
    isnothing(f) || all(T -> hasmethod(f, Tuple{Int,T}), (SystemOneSuccess, SystemOneFailure, SystemOneCallError)) ||
        throw(ArgumentError("`on_result` must be `nothing` or callable as `on_result(i, result)` with an index and " *
                            "any result — a SystemOneSuccess, SystemOneFailure or SystemOneCallError; got $(typeof(f))"))
    nothing
end

# An item resolved and encoded — through its backend's checks — before anything is sent,
# and kept without its wire, which the worker that sends it encodes again.
function _each_checked(r::SystemOneRequest, service::Union{Nothing,ServiceEndpointSpec})::_SystemOneResolved
    resolved = _system_one_resolve(r, service)
    _system_one_call(resolved)
    resolved
end

# An item's request, built — and encoded, through its backend's checks — before anything
# is sent; a refusal names the item.
function _each_item(f::Function, i::Int, n::Int)
    try
        f()
    catch e
        e isa ArgumentError || rethrow()
        throw(ArgumentError("item $i of $n cannot be sent, so ask_each sent nothing: $(e.msg)"))
    end
end

function _run_each(b::_EachBatch, cancel::Union{Nothing,CancelToken}, workers::Int)::Vector{_SystemOneResult}
    link = _on_cancel(() -> cancel!(b.token), cancel)     # the caller's token stops the batch
    tasks = Task[]
    try
        for _ in 1:workers
            push!(tasks, Threads.@spawn :default _each_worker(b))
        end
        foreach(wait, tasks)
    catch e
        # Interrupted while waiting: stop the batch, and let every worker end first.
        _stop!(b, e)
        foreach(_wait_ended, tasks)
    finally
        _off_cancel(cancel, link)
    end
    failure = @atomic b.failure
    isnothing(failure) || throw(something(failure))
    b.results
end

function _wait_ended(t::Task)::Nothing
    try
        wait(t)
    catch e
        e isa TaskFailedException || rethrow()      # a second interrupt is the caller's to have
    end
    nothing
end

function _each_worker(b::_EachBatch)::Nothing
    hooks = (hold=() -> _held_for(b.backoff), observe=(k::Int, r::HTTP.Response) -> _back_off!(b.backoff, k, r))
    on_result = b.on_result
    while isnothing(@atomic b.failure)
        i = @atomic b.next += 1
        i > length(b.items) && return nothing
        # The item's budget runs from here, so the batch's backoff — waited out before each
        # attempt — holds it no longer than its own total_deadline.
        t0 = time_ns()
        r = try
            _system_one_exchange(_system_one_call(b.items[i]), b.config, b.token; t0, hooks...)
        catch e
            return _stop!(b, e)
        end
        b.results[i] = r
        if !isnothing(on_result)
            try
                # One call at a time; none once the batch has stopped.
                @lock b.reporting (isnothing(@atomic b.failure) && on_result(i, r))
            catch e
                return _stop!(b, e)
            end
        end
    end
    nothing
end

# The first exception to stop the batch is the one the caller gets; the token ends the
# requests in flight, and the workers claim no further item.
function _stop!(b::_EachBatch, e)::Nothing
    @atomicreplace b.failure nothing => Some{Any}(e)
    cancel!(b.token)
    nothing
end
