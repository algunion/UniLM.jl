# ─── Bounded HTTP seam ───────────────────────────────────────────────────────
# Every provider HTTP exchange routes through _http (one attempt),
# _http_with_retries (the one retry loop) or _http_open (streaming). The seam
# translates RequestConfig bounds into HTTP.jl's native timeout kwargs — they
# fire earlier, with better phase attribution — AND arms an outer watchdog at
# the same bound as the guarantee of last resort. Native timeout exceptions map
# to UniLMTimeout; all other transport exceptions propagate unchanged.

const _RETRY_BASE = 1.0
const _RETRY_FACTOR = 2.0
const _RETRY_MAX_DELAY = 60.0

_is_retryable(status::Integer)::Bool = status in (408, 429, 500, 502, 503, 504, 529)

# RFC 7231 IMF-fixdate, the one HTTP-date form every sender must emit. Always GMT.
const _IMF_FIXDATE = r"^[A-Za-z]{3}, (\d{2}) ([A-Za-z]{3}) (\d{4}) (\d{2}):(\d{2}):(\d{2}) GMT$"
const _IMF_MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun",
                     "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")

# Unix seconds for a UTC civil date-time (days-from-civil, era-based Gregorian).
# Written out rather than pulled from `Dates` — that stdlib is not a dependency
# of this package and one header form does not justify making it one.
function _utc_epoch_seconds(y::Int, mo::Int, d::Int, h::Int, mi::Int, s::Int)::Float64
    yr = y - (mo <= 2)
    era = fld(yr, 400)
    yoe = yr - era * 400
    doy = div(153 * (mo + (mo > 2 ? -3 : 9)) + 2, 5) + d - 1
    doe = yoe * 365 + div(yoe, 4) - div(yoe, 100) + doy
    return (era * 146097 + doe - 719468) * 86400.0 + h * 3600 + mi * 60 + s
end

# Read capture group `i` of an _IMF_FIXDATE match as an integer. Every group in
# that pattern is mandatory, so a successful match fills all six; a capture is
# nullable in general, and reading one straight into `parse` would leave that
# invariant asserted nowhere and surface a violation as a MethodError inside
# `parse` instead of naming the group that came back empty.
function _fixdate_int(m::RegexMatch, i::Int)::Int
    cap = m[i]
    isnothing(cap) && throw(ArgumentError("IMF-fixdate group $i did not capture"))
    return parse(Int, cap)
end

"""
    _retry_after_seconds(resp) -> Union{Nothing,Float64}

The wait `Retry-After` asks for, in seconds, or `nothing` when the header is
absent or unparseable. RFC 7231 defines TWO forms and servers behind CDNs send
both: delta-seconds, and an HTTP-date — reading only the first under-waits a 429
storm by whatever the date form was asking for. A date already in the past
yields `0.0`. Malformed values return `nothing` so the caller keeps its default
backoff: a server sending garbage must never make a client throw.
"""
function _retry_after_seconds(resp::HTTP.Response)::Union{Nothing,Float64}
    ra = strip(HTTP.header(resp, "Retry-After", ""))
    isempty(ra) && return nothing
    secs = tryparse(Int, ra)
    isnothing(secs) || return max(0.0, Float64(secs))
    m = match(_IMF_FIXDATE, ra)
    isnothing(m) && return nothing
    mo = findfirst(==(m[2]), _IMF_MONTHS)
    isnothing(mo) && return nothing
    due = _utc_epoch_seconds(_fixdate_int(m, 3), mo, _fixdate_int(m, 1),
                             _fixdate_int(m, 4), _fixdate_int(m, 5), _fixdate_int(m, 6))
    return max(0.0, due - time())
end

# Full-jitter backoff; a Retry-After header is a FLOOR under it, not a
# replacement: clients that all receive the same header would otherwise wake at
# the same instant and retry in lockstep. The spread above the floor is capped at
# the budget left after it, so jitter never pushes a floor that fits past
# `remaining`.
function _retry_delay(retry::Integer, resp::HTTP.Response, remaining::Float64=Inf)::Float64
    computed = min(_RETRY_BASE * _RETRY_FACTOR^retry, _RETRY_MAX_DELAY)
    ra = _retry_after_seconds(resp)
    isnothing(ra) && return rand() * computed
    return ra + rand() * min(computed, max(remaining - ra, 0.0))
end

"""
    _retry_pause(cfg, t0, attempt, resp) -> (action::Symbol, delay::Float64)

Shared retry-budget arithmetic — the single implementation used by the non-stream
retry loop and the stream driver. Computes the full-jitter backoff for `attempt`
(1-based); a `Retry-After` header, when a response carries one, is the floor the
jitter spreads above. Returns `(:sleep, delay)` when the pause fits the remaining
total deadline, else `(:budget, delay)` — which, with a header, happens only when
the header's own wait does not fit: fail NOW with the last real outcome — sleeping
less and attempting with ~zero budget is a guaranteed mid-flight breach, and
sleeping past the deadline breaks the bound.
"""
function _retry_pause(cfg::RequestConfig, t0::UInt64, attempt::Int,
                      resp::Union{HTTP.Response,Nothing})::Tuple{Symbol,Float64}
    remaining = _remaining_s(cfg, t0)
    delay = _retry_delay(attempt - 1, isnothing(resp) ? HTTP.Response(0) : resp, remaining)
    delay > remaining ? (:budget, delay) : (:sleep, delay)
end

"""
    _unwrap_exception(e)

Peel task/transport wrappers to the root cause so timeout and interrupt classification
works regardless of how the failure was wrapped: `TaskFailedException` (internal
request tasks), `CompositeException`, and wrapper exceptions exposing their cause as
an `error::Exception` field.
"""
function _unwrap_exception(e)
    while true
        if e isa TaskFailedException
            e = e.task.exception
        elseif e isa CompositeException && !isempty(e.exceptions)
            e = first(e.exceptions)
        elseif e isa Exception && hasproperty(e, :error) && getproperty(e, :error) isa Exception
            e = getproperty(e, :error)
        else
            return e
        end
    end
end

# Header names whose VALUE is a credential. HTTP.jl masks only Authorization,
# Proxy-Authorization and Cookie when it renders a request, so a provider that
# authenticates with its own header — Anthropic `x-api-key`, Gemini native
# `x-goog-api-key`, Azure `api-key` — has its key printed verbatim inside the
# request dump that some transport exceptions carry in their message. Matches the
# wire form (`name: value`) and the Julia pair form (`"name" => "value"`). The
# leading `\b` holds a name to header position, so a longer word merely ENDING in
# one (`reauthorization:`) is left alone; the alternation stays longest-first so the
# leftmost match of `x-api-key` starts at its own `x`, not at its `api-key` tail.
const _AUTH_HEADER_PATTERN =
    r"(?i)\b(x-goog-api-key|x-api-key|api-key|proxy-authorization|authorization)(\"?\s*(?::|=>|=)\s*\"?)([^\r\n\"]*)"

# Replace every auth-shaped header value with the same short, non-reversible
# marker the endpoint `show` methods use.
_mask_auth_headers(s::AbstractString)::String =
    replace(s, _AUTH_HEADER_PATTERN => function (hit)
        m = match(_AUTH_HEADER_PATTERN, hit)
        # `hit` is a substring the same pattern just matched, so this re-match
        # always succeeds; `match` is still typed `Union{Nothing,RegexMatch}`, and
        # a `nothing` here would crash the one renderer every *CallError goes
        # through. Hand the text back unchanged instead of indexing `nothing`.
        m === nothing && return hit
        # Every capture group is statically `Union{Nothing,SubString}` because any
        # group may go unmatched. Group 3 is a `*` group, so it always
        # participates at runtime — and an absent value would be an empty one,
        # with nothing to redact.
        string(m[1], m[2], _redact_api_key(something(m[3], "")))
    end)

"""
    _error_text(e) -> String

The one renderer for the user-visible `error::String` of every `*CallError` result.

Prefers the root cause's `showerror` text over `string(e)`: the wrapper layers add
no diagnostic value, and `string` renders a wrapper's raw fields rather than a
message. Then masks auth-shaped header values as defense in depth, so a credential
cannot reach a result value (or a log line, or a bug report) no matter which
library layer produced the text.
"""
function _error_text(e)::String
    u = _unwrap_exception(e)
    txt = try
        u isa Exception ? sprint(showerror, u) : string(u)
    catch e
        # A user interrupt that lands while rendering is still the user's intent.
        e isa InterruptException && rethrow()
        # A showerror that itself throws must not replace a typed failure with a
        # crash: name the type and move on.
        string(typeof(u))
    end
    _mask_auth_headers(txt)
end

# HTTP.jl's timeout kwargs take ::Real seconds but require them finite: Inf
# must be translated to the documented "off" value 0 BEFORE the call.
_native_seconds_real(x::Float64)::Float64 = x == Inf ? 0.0 : x

# Native kwargs for a non-streaming attempt. connect_timeout is passed
# EXPLICITLY: omitting it selects a 30 s library default, not "off".
_native_timeout_kwargs(cfg::RequestConfig, bound::Float64) =
    (connect_timeout = _native_seconds_real(cfg.connect_timeout),
     request_timeout = _native_seconds_real(bound))

# Native kwargs for a streaming attempt: the connect bound, plus
# read_idle_timeout, which resets on every read — byte-gap idle semantics — so
# it rides along as a fast path. No whole-exchange bound is armed: it would kill
# long healthy streams.
# INVARIANT: connect and (finite idle bound) read-idle are the ONLY native
# timers armed here WHILE THE IDLE BOUND IS FINITE.
# `_classify_stream_timeout` attributes every non-connect native timeout on a
# streaming attempt to the read-idle timer BY ELIMINATION, because HTTP.jl
# surfaces a read-idle breach with the literal operation="request" —
# indistinguishable by label from any other request-phase timeout. The
# header-wait cap below is armed ONLY when the idle bound is disabled, exactly
# so the elimination stays sound (that classifier requires
# `stream_idle_timeout < Inf`); adding any other streaming timer requires
# revisiting the classifier first.
function _native_stream_kwargs(cfg::RequestConfig, bound::Float64=Inf)
    kw = (connect_timeout   = _native_seconds_real(cfg.connect_timeout),
          read_idle_timeout = _native_seconds_real(cfg.stream_idle_timeout))
    # Idle bound disabled: nothing above bounds the response-header wait, and the
    # driver's request-phase watchdog cannot help either — closing the client
    # stream cannot reach the connection until `startread` returns (HTTP.jl hands
    # the stream its connection together with the response headers) — so a mute
    # peer would stall forever. Cap the header wait natively at the same request
    # bound. ONLY in this branch: with a finite idle bound read_idle_timeout
    # already bounds that wait (HTTP.jl waits min(response_header_timeout,
    # read_idle_timeout)), and a second native non-connect timer would break the
    # by-elimination attribution in `_classify_stream_timeout`.
    return (cfg.stream_idle_timeout == Inf && bound < Inf) ?
        (kw..., response_header_timeout = bound) : kw
end

# The kwargs the seam imposes on HTTP.jl, after any caller kwargs so they win.
# Every call is ONE attempt (the retry budget lives in _http_with_retries; the
# library's own retry layer would multiply wire attempts behind its back), and
# callers branch on the status instead of catching StatusError.
_request_kwargs(cfg::RequestConfig, bound::Float64) =
    (status_exception = false, retry = false, _native_timeout_kwargs(cfg, bound)...)

# Streams also pin HTTP/1.1. HTTP.jl negotiates HTTP/2 for https and then
# multiplexes every concurrent call to a host over ONE connection with shared
# flow-control windows, so a stream whose consumer applies backpressure starves
# every other stream on that connection. One connection per stream isolates them
# (the official OpenAI and Anthropic SDKs also stream over HTTP/1.1 by default).
# Non-streaming requests keep protocol negotiation (:auto).
_open_kwargs(cfg::RequestConfig, bound::Float64) =
    (status_exception = false, retry = false, protocol = :h1, _native_stream_kwargs(cfg, bound)...)

# Phase attribution for a native TimeoutError operation label.
_timeout_phase(operation::AbstractString)::Symbol =
    operation == "connect" || operation == "tls_handshake" ? :connect : :request

# Map a native HTTP.jl timeout exception (possibly nested in wrapper layers)
# to UniLMTimeout; return nothing when `e` is not a timeout — the caller then
# rethrows the original, so non-timeout transport errors propagate unchanged.
function _map_native_timeout(e, cfg::RequestConfig, bound::Float64, t0::UInt64)::Union{Nothing,UniLMTimeout}
    native = _find_exception(x -> x isa HTTP.TimeoutError, e)
    native === nothing && return nothing
    phase = _timeout_phase(native.operation)
    return UniLMTimeout(phase, _elapsed_s(t0), phase === :connect ? cfg.connect_timeout : bound)
end

"""
    _classify_stream_timeout(e, idle, cfg, t0) -> Union{Nothing,UniLMTimeout}

Classify a caught streaming-attempt exception as a byte-gap idle breach —
`UniLMTimeout(:stream_idle, …)` — or return `nothing` for the caller's
fallthrough (connect mapping / transport handling / rethrow).

A breach is recognized from TWO timing-independent facts, never from whether
the in-driver idle guard happened to be armed yet:

1. `_idle_fired(idle)`: our own guard closed the socket — the caught error is
   the echo of that close.
2. A native `HTTP.TimeoutError` whose operation is NOT connect/TLS, while the
   read-idle fast path is armed (`cfg.stream_idle_timeout < Inf`). The
   streaming seam arms no other native non-connect timer (see
   `_native_stream_kwargs`), so such a timeout IS the read-idle timer by
   elimination — regardless of where in the exchange it fired. In particular,
   HTTP.jl bounds the response-header wait by
   `min(response_header_timeout, read_idle_timeout)`, so the byte-gap bound
   can breach BEFORE the first byte arrives — before the idle guard exists.
   Deciding from the armed-timer set keeps the phase deterministic across
   that arming-order race (observed flipping with HTTP 2.6.x server-side
   task-scheduling changes).

`elapsed` reports the measured byte gap where the guard measured one; for a
pre-first-byte breach the attempt's own elapsed time is the honest "no bytes
for this long" bound.
"""
function _classify_stream_timeout(e, idle, cfg::RequestConfig, t0::UInt64)::Union{Nothing,UniLMTimeout}
    native = _find_exception(x -> x isa HTTP.TimeoutError, e)
    native_idle = native !== nothing && cfg.stream_idle_timeout < Inf &&
                  _timeout_phase(native.operation) !== :connect
    (_idle_fired(idle) || native_idle) || return nothing
    gap = idle === nothing ? _elapsed_s(t0) : _idle_gap_s(idle)
    return UniLMTimeout(:stream_idle, gap, cfg.stream_idle_timeout)
end

"""
    _stream_teardown_noise(e, breach) -> Bool

True when `e` is the TEARDOWN of a stream rather than its outcome: an already
classified byte-gap breach (`breach`, from [`_classify_stream_timeout`] — the
echo of our own guard's close OR the native read-idle timer, whichever won the
race), a connection-level failure, or a typed bound. Both stream drivers consult
this once a terminal result is recorded — teardown noise must then neither
discard a billed generation nor re-POST it — while anything else (a throwing
user callback, a decoding bug) still surfaces as a failure.

Taking the CLASSIFIED breach rather than the guard handle is load-bearing: on
HTTP 2.x the native read-idle timer can fire before our own guard, and it
surfaces as an `HTTP.TimeoutError`, which is deliberately neither
transport-shaped nor a `UniLMTimeout`. Only the classifier recognises it.
"""
_stream_teardown_noise(e, breach::Union{Nothing,UniLMTimeout})::Bool =
    breach !== nothing || _is_transport_error(e) ||
    _find_exception(x -> x isa UniLMTimeout, e) !== nothing

"""
    _exit_breach(idle, bound, cfg) -> Union{Nothing,UniLMTimeout}

The typed breach owed by a read loop that exited WITHOUT an exception. Closing
the socket is how a guard unblocks a blocked read, so a close landing while the
driver is inside a user callback truncates the read into a clean EOF and the
loop simply ends — with nothing to classify, a killed stream would be reported
as a 200 carrying partial bytes. Returns the byte-gap breach when the idle guard
fired, else the recorded request-phase bound, else `nothing` (a clean EOF really
is the end of the stream).

Reads guard STATE here, unlike `_stream_teardown_noise`, which needs the
classifier: this path exists only because OUR guard's `close` turns a blocked
read into a clean EOF. The 2.x native read-idle timer never reaches here — it
raises from the read, so it always takes the throwing path.
"""
_exit_breach(idle, bound::Ref{Union{Nothing,UniLMTimeout}},
             cfg::RequestConfig)::Union{Nothing,UniLMTimeout} =
    _idle_fired(idle) ?
        UniLMTimeout(:stream_idle, _idle_gap_s(idle), cfg.stream_idle_timeout) : bound[]

"""
    _with_recorded_deadline(f, close!, limit, phase, slot) -> f()

[`_with_deadline`](@ref) with the typed outcome RECORDED into `slot` before the
exception unwinds any further: on a breach (or any `UniLMTimeout` escaping
`f`), `slot[]` is set to that timeout and the exception rethrows unchanged.

Why recording matters: the streaming drivers run the deadline block INSIDE
`HTTP.open`'s handler, and what escapes `HTTP.open` is not always what the
handler raised — once the attempt's request context is cancelled, HTTP.jl
rethrows any handler exception as its own `HTTP.CanceledError`. The recorded
value lets the driver's catch restore the typed cause; transport errors with NO
recorded bound are untouched and classify as today. `InterruptException`
rethrows unrecorded.

The completion race is recorded and raised the same way: when `f` returns a real
value but the timer already won the resolution CAS, the guarded socket has been
closed even though nothing threw. Nothing else would report that — the next read
would raise a bare IOError on the closed socket, which classifies as a RETRYABLE
transport failure (an extra billed wire attempt, with the phase lost) — so the
bound that actually fired is recorded and thrown here instead.
"""
function _with_recorded_deadline(f::Function, close!::Function, limit::Float64,
                                 phase::Symbol, slot::Ref{Union{Nothing,UniLMTimeout}})
    t0 = time_ns()
    result, fired = try
        _with_deadline_reported(f, close!, limit, phase)
    catch e
        e isa UniLMTimeout && (slot[] = e)
        rethrow()
    end
    fired || return result
    bound = UniLMTimeout(phase, _elapsed_s(t0), limit)
    slot[] = bound
    throw(bound)
end

# Retry-loop predicate: a per-attempt timeout (:connect/:request — a timeout
# with budget left is worth another try) or a pure transport failure per
# _is_transport_error. :deadline (budget spent), :stream_idle, interrupts,
# and status-carrying errors all fall through to false via the classifier's
# always-false set — ONE classifier, composed here, never duplicated. A
# cancellation anywhere in the chain is never retried.
_retryable_exception(e)::Bool =
    _find_exception(_cancel_shaped, e) === nothing &&
    ((e isa UniLMTimeout && (e.phase === :connect || e.phase === :request)) ||
     _is_transport_error(e))

"""
    _BodyFactory(build)

A request body that must be rebuilt for every attempt. The seam owns retries and
therefore passes `retry=false`, which also disables HTTP.jl's own mark/reset body
rewind — nothing rewinds a consumable body between attempts. A multipart
`HTTP.Form` read to EOF by attempt 1 would put a zero-length body on the wire on
attempt 2, turning a transient 429/503 into a hard protocol failure. Wrapping the
body in a factory makes each attempt construct a fresh, fully readable one;
`build()` re-reads from its source rather than retaining a buffered copy.
"""
struct _BodyFactory{F}
    build::F
end

_attempt_body(body) = body
_attempt_body(f::_BodyFactory) = f.build()

"""
    _http(method, url, headers=[], body=UInt8[]; cfg, remaining=Inf, cancel, kwargs...) -> HTTP.Response

One bounded, cancellable HTTP attempt. The per-attempt bound is
`min(cfg.request_timeout, remaining)`; `remaining <= 0` throws
`UniLMTimeout(:deadline, …)` without touching the network. HTTP.jl's native
timeout kwargs are the fast path; a task-mode watchdog at the same bound is
the guarantee of last resort. Native timeout exceptions map to
[`UniLMTimeout`](@ref) (`:connect` where attributable, else `:request`);
other transport exceptions propagate unchanged.

`cancel` (default: the ambient token of [`with_cancel`](@ref)) already cancelled
at entry throws [`UniLMCancelled`](@ref) with no network I/O. Each attempt runs
under its own `HTTP.RequestContext`, and a cancel during the attempt cancels that
context (aborting the exchange) and releases the watchdog's waiter; any failure
surfacing while the token is cancelled is reported as `UniLMCancelled(:token, …)`.

Always imposes `status_exception=false` (callers branch on `resp.status`)
and `retry=false`: this is ONE attempt — the retry budget lives in
`_http_with_retries`, and the library's internal retry layer would multiply
wire attempts behind the budget's back. `body` is a passthrough positional:
`String`, `Vector{UInt8}`, and `HTTP.Form` are all handed to `HTTP.request`
unconverted (callers keep their existing body shapes); a [`_BodyFactory`](@ref)
is built here instead, so each attempt gets its own body. Remaining kwargs pass
through to `HTTP.request` (e.g. `decompress=false`).
"""
function _http(method::AbstractString, url::AbstractString,
               headers=Pair{String,String}[], body=UInt8[];
               cfg::RequestConfig=current_config(), remaining::Float64=Inf,
               cancel::Union{Nothing,CancelToken}=_current_cancel(),
               kwargs...)::HTTP.Response
    t0 = time_ns()
    iscancelled(cancel) && throw(UniLMCancelled(:token, _elapsed_s(t0)))
    ispositive(remaining) ||
        throw(UniLMTimeout(:deadline, max(cfg.total_deadline - remaining, 0.0), cfg.total_deadline))
    bound = min(cfg.request_timeout, remaining)
    ctx = HTTP.RequestContext()
    try
        return _with_deadline_task(bound, :request; cancel,
                                   on_cancel=() -> HTTP.cancel!(ctx; message="cancelled")) do
            HTTP.request(method, url, headers, _attempt_body(body);
                         kwargs..., _request_kwargs(cfg, bound)..., context=ctx)
        end
    catch e
        e isa InterruptException && rethrow()
        e isa UniLMCancelled && rethrow()
        # Cancelled mid-attempt: whatever surfaced (HTTP.CanceledError from the
        # aborted exchange, or a racing timeout) is the cancellation's echo.
        iscancelled(cancel) && throw(UniLMCancelled(:token, _elapsed_s(t0)))
        e isa UniLMTimeout && rethrow()
        mapped = _map_native_timeout(e, cfg, bound, t0)
        mapped === nothing ? rethrow() : throw(mapped)
    end
end

"""
    _http_with_retries(cfg, t0, method, url, headers=[], body=UInt8[]; cancel, kwargs...) -> HTTP.Response

The one retry loop. `t0` is the verb-entry monotonic origin (`time_ns()`).
Runs at most `cfg.max_attempts` attempts inside `cfg.total_deadline`
(breaching it throws `UniLMTimeout(:deadline, …)`); each attempt gets
`min(cfg.request_timeout, remaining)` via [`_http`](@ref).

Retryable = retryable status (408/429/500/502/503/504/529) ∪ per-attempt
`UniLMTimeout(:connect|:request)` ∪ transport-level IO exceptions — never
`InterruptException`. When the backoff (`Retry-After`/jitter) exceeds the
remaining budget, fail NOW with the last real outcome: sleeping less and
attempting with ~zero budget is a guaranteed mid-flight breach, and a
budget-exhausted 429 is a 429, not a fabricated timeout. Intermediate
retries log at debug; the final failure warns once.

`cancel` (default: the ambient token) is checked before every attempt, backoff
sleeps wake at once on a cancel, and a cancelled call is never retried: each
surfaces as [`UniLMCancelled`](@ref) with `elapsed` measured from `t0`.
"""
function _http_with_retries(cfg::RequestConfig, t0::UInt64,
                            method::AbstractString, url::AbstractString,
                            headers=Pair{String,String}[], body=UInt8[];
                            cancel::Union{Nothing,CancelToken}=_current_cancel(),
                            kwargs...)::HTTP.Response
    cancelled() = UniLMCancelled(:token, _elapsed_s(t0))
    for attempt in 1:cfg.max_attempts
        iscancelled(cancel) && throw(cancelled())
        remaining = _remaining_s(cfg, t0)
        ispositive(remaining) || throw(UniLMTimeout(:deadline, _elapsed_s(t0), cfg.total_deadline))
        final = attempt == cfg.max_attempts
        resp = try
            _http(method, url, headers, body; cfg, remaining, cancel, kwargs...)
        catch e
            e isa InterruptException && rethrow()
            e isa UniLMCancelled && throw(cancelled())   # never retried
            (_retryable_exception(e) && !final) || rethrow()
            action, delay = _retry_pause(cfg, t0, attempt, nothing)
            if action === :budget
                @warn "transport failure is retryable but the backoff exceeds the remaining total_deadline budget; giving up" attempt delay
                rethrow()
            end
            # Log the ROOT CAUSE, never the wrapper chain: the wrapper layers add
            # no diagnostic value. Twin of the stream driver below.
            @debug "retrying after transport failure" attempt delay exception = (_unwrap_exception(e), catch_backtrace())
            _cancel_sleep(cancel, delay) && throw(cancelled())
            continue
        end
        if _is_retryable(resp.status) && !final
            action, delay = _retry_pause(cfg, t0, attempt, resp)
            if action === :budget
                @warn "status is retryable but the backoff (Retry-After/jitter) exceeds the remaining total_deadline budget; returning the last response" status = resp.status attempt delay
                return resp
            end
            @debug "retrying after retryable status" status = resp.status attempt delay
            _cancel_sleep(cancel, delay) && throw(cancelled())
            continue
        end
        _is_retryable(resp.status) &&
            @warn "request failed with a retryable status after exhausting max_attempts" status = resp.status attempts = cfg.max_attempts
        return resp
    end
    # Unreachable: max_attempts >= 1 and the final attempt never continues.
    error("retry loop exited without an outcome")
end

"""
    _http_open(f, method, url, headers; cfg, t0, cancel, kwargs...) -> HTTP.Response

Streaming attempt seam: wraps `HTTP.open` with `status_exception=false`,
`retry=false`, `protocol=:h1` (one connection per stream: HTTP/2 multiplexing
would let one backpressured stream starve its siblings), and the native stream
kwargs (the connect bound and the per-read `read_idle_timeout` byte-gap fast
path — a whole-exchange native bound would kill long healthy streams — plus,
where the idle bound is disabled, a native cap on the response-header wait at
`min(request_timeout, remaining)`). `f(io)` receives the raw stream untouched:
the first-byte deadline and the idle guard are the calling driver's job, because
only the driver knows when the request body is written and the response headers
arrive. `t0` is the driver's monotonic origin, accepted here so drivers
thread one origin through the seam.

`cancel` (default: the ambient token) already cancelled at entry throws
[`UniLMCancelled`](@ref) with no network I/O. The attempt runs under its own
`HTTP.RequestContext`, cancelled by a cancel for the attempt's whole duration:
HTTP.jl checks it before acquiring a connection and aborts the connection once
acquired, which unblocks the response-header wait and body reads parked inside
`f` (a TCP/TLS connect already in progress finishes, or hits its connect bound,
first). Whatever then escapes `HTTP.open` (HTTP.jl reports a cancelled context as
`HTTP.CanceledError`) propagates unchanged — mapping it to a typed result is the
driver's job.
"""
function _http_open(f::Function, method::AbstractString, url::AbstractString, headers;
                    cfg::RequestConfig, t0::UInt64,
                    cancel::Union{Nothing,CancelToken}=_current_cancel(),
                    kwargs...)::HTTP.Response
    iscancelled(cancel) && throw(UniLMCancelled(:token, _elapsed_s(t0)))
    ctx = HTTP.RequestContext()
    handle = _on_cancel(() -> HTTP.cancel!(ctx; message="cancelled"), cancel)
    try
        return HTTP.open(method, url, headers;
                         kwargs..., _open_kwargs(cfg, min(cfg.request_timeout, _remaining_s(cfg, t0)))...,
                         context=ctx) do io
            f(io)
        end
    finally
        _off_cancel(cancel, handle)
    end
end

# ─── URL Dispatch ─────────────────────────────────────────────────────────────
# Endpoints are determined by (ServiceEndpoint, RequestType), not model name.

# Assert `::String` at the request-URL entry points: the per-endpoint `get_url` methods
# each build a `String`, but the abstract `service`-typed dispatch can lose that precision
# under coverage/`--check-bounds=yes` (which suppress the constant-folding that resolves it),
# widening to `Union{Missing,…}`. The assertion is inference-independent and holds under any
# Julia flags, keeping every `url` a concrete `String` at the HTTP seam.
get_url(chat::Chat)::String = get_url(chat.service, chat)::String
get_url(emb::Embeddings)::String = get_url(emb.service, emb)::String

get_url(::Type{OPENAIServiceEndpoint}, ::Chat)::String = OPENAI_BASE_URL * CHAT_COMPLETIONS_PATH
get_url(::Type{AZUREServiceEndpoint}, chat::Chat)::String = ENV[AZURE_OPENAI_BASE_URL] * _azure_deployment_path(chat.model) * "/chat/completions?api-version=$(ENV[AZURE_OPENAI_API_VERSION])"
get_url(::Type{GEMINIOpenAIServiceEndpoint}, ::Chat)::String = GEMINI_CHAT_URL

get_url(::Type{OPENAIServiceEndpoint}, ::Embeddings)::String = OPENAI_BASE_URL * EMBEDDINGS_PATH
get_url(::Type{GEMINIOpenAIServiceEndpoint}, ::Embeddings)::String = GEMINI_OPENAI_BASE * "/embeddings"

# Single typed entry over the per-endpoint `_resolve_base_url` dispatch, whose method set is
# wider than inference will union at an abstract `service::ServiceEndpointSpec` call (the
# `Type{<:ServiceEndpoint}` limb exceeds the union-split budget → `Any`). A DECLARED
# `::String` return (not a body typeassert) is a method-signature guarantee that holds under
# that widening AND under coverage instrumentation, keeping every caller's `url` a `String`.
_api_base_url(service::ServiceEndpointSpec)::String = _resolve_base_url(service)

_resolve_base_url(::Type{OPENAIServiceEndpoint}) = OPENAI_BASE_URL
_resolve_base_url(::Type{AZUREServiceEndpoint}) = throw(ArgumentError("Responses API is only supported with OPENAIServiceEndpoint"))
_resolve_base_url(::Type{GEMINIOpenAIServiceEndpoint}) = throw(ArgumentError("Responses API is only supported with OPENAIServiceEndpoint"))
# Fail-loud total coverage: any endpoint without a specific base URL (native providers,
# user-defined subtypes) is not an OpenAI-wire platform endpoint. Total coverage keeps the
# abstract dispatch free of latent MethodErrors (nothing for JET to flag as a missing method).
_resolve_base_url(s::ServiceEndpoint) = throw(ArgumentError("base URL is not defined for $(typeof(s)); this endpoint is not an OpenAI-wire platform endpoint"))
_resolve_base_url(::Type{<:ServiceEndpoint}) = throw(ArgumentError("base URL is not defined for this endpoint type; this endpoint is not an OpenAI-wire platform endpoint"))

# ─── GenericOpenAIEndpoint dispatch ──────────────────────────────────────────

get_url(s::GenericOpenAIEndpoint, ::Chat)::String = rstrip(s.base_url, '/') * CHAT_COMPLETIONS_PATH
get_url(s::GenericOpenAIEndpoint, ::Embeddings)::String = rstrip(s.base_url, '/') * EMBEDDINGS_PATH
_resolve_base_url(s::GenericOpenAIEndpoint) = String(rstrip(s.base_url, '/'))

function auth_header(s::GenericOpenAIEndpoint)::Vector{Pair{String,String}}
    hdrs = ["Content-Type" => "application/json"]
    !isempty(s.api_key) && pushfirst!(hdrs, "Authorization" => "Bearer $(s.api_key)")
    hdrs
end

# ─── DeepSeekEndpoint dispatch ───────────────────────────────────────────────

get_url(s::DeepSeekEndpoint, ::Chat)::String = DEEPSEEK_BASE_URL * CHAT_COMPLETIONS_PATH
get_url(s::DeepSeekEndpoint, ::Embeddings)::String = DEEPSEEK_BASE_URL * EMBEDDINGS_PATH
_resolve_base_url(s::DeepSeekEndpoint) = DEEPSEEK_BASE_URL

function auth_header(s::DeepSeekEndpoint)::Vector{Pair{String,String}}
    ["Authorization" => "Bearer $(s.api_key)", "Content-Type" => "application/json"]
end

# ─── Built-in endpoint auth ─────────────────────────────────────────────────

function auth_header(::Type{OPENAIServiceEndpoint})::Vector{Pair{String,String}}
    [
        "Authorization" => "Bearer $(ENV[OPENAI_API_KEY])",
        "Content-Type" => "application/json"
    ]
end

function auth_header(::Type{AZUREServiceEndpoint})::Vector{Pair{String,String}}
    [
        "api-key" => "$(ENV[AZURE_OPENAI_API_KEY])",
        "Content-Type" => "application/json"
    ]
end

function auth_header(::Type{GEMINIOpenAIServiceEndpoint})::Vector{Pair{String,String}}
    [
        "Authorization" => "Bearer $(ENV[GEMINI_API_KEY])",
        "Content-Type" => "application/json"
    ]
end



# Multipart auth: strip the JSON Content-Type so HTTP.Form can set its own
# multipart/form-data boundary header (a stray application/json corrupts the upload).
auth_header_multipart(s) = filter(p -> lowercase(String(first(p))) != "content-type", auth_header(s))

# Stub: overridden by accounting.jl after include
_accumulate_cost!(::Chat, ::LLMRequestResponse) = nothing

"""
    _token_usage_from(u::AbstractDict; prompt_key, completion_key, prompt_details, completion_details)

Build a [`TokenUsage`](@ref) from a raw `usage` dict, pulling the cached/reasoning
subtotals from the detail objects when present. Chat Completions and the Responses API
name these differently (`prompt_tokens`/`input_tokens`, `prompt_tokens_details`/
`input_tokens_details`, …), so the keys are parameterized.
"""
function _token_usage_from(u::AbstractDict;
    prompt_key::String="prompt_tokens", completion_key::String="completion_tokens",
    prompt_details::String="prompt_tokens_details", completion_details::String="completion_tokens_details")
    pd = get(u, prompt_details, nothing)
    cd = get(u, completion_details, nothing)
    _i(x) = x isa Integer ? Int(x) : 0   # tolerate JSON null / missing / non-int → 0
    TokenUsage(
        prompt_tokens=_i(get(u, prompt_key, 0)),
        completion_tokens=_i(get(u, completion_key, 0)),
        total_tokens=_i(get(u, "total_tokens", 0)),
        cached_tokens=(pd isa AbstractDict ? _i(get(pd, "cached_tokens", 0)) : 0),
        reasoning_tokens=(cd isa AbstractDict ? _i(get(cd, "reasoning_tokens", 0)) : 0),
    )
end

function _parse_usage(data::Dict{String,Any})::Union{TokenUsage, Nothing}
    haskey(data, "usage") || return nothing
    u = data["usage"]
    u isa AbstractDict || return nothing
    _token_usage_from(u)
end

"""
    extract_message(resp::HTTP.Response) -> (; message::Message, usage)

Decode the first choice of an OpenAI-wire Chat Completions reply. No part of the
choice is dropped for its shape: tool calls are decoded whenever present, with any
text kept alongside; content given as an array of parts is joined from its text
parts; a refusal is kept; `""` tool arguments decode as an empty object. A choice
that carries nothing is the empty turn it is (`content = ""` — a reasoning model can
spend the whole budget on thought tokens). The finish reason is the wire's, except
that a tool-call turn finished with `"stop"` or none reads `"tool_calls"`.
"""
function extract_message(resp::HTTP.Response)
    received = JSON.parse(resp.body; dicttype=Dict{String,Any})
    choices = get(received, "choices", nothing)
    (choices isa AbstractVector && !isempty(choices)) || error("API returned empty choices array")
    choice = first(choices)
    message = choice["message"]
    finish = get(choice, "finish_reason", nothing)
    text = _content_text(get(message, "content", nothing))
    refusal = get(message, "refusal", nothing)
    refusal isa AbstractString || (refusal = nothing)
    raw_calls = get(message, "tool_calls", nothing)
    msg = if raw_calls isa AbstractVector && !isempty(raw_calls)
        Message(role=RoleAssistant, content=(isnothing(text) || isempty(text) ? nothing : text),
                tool_calls=_decode_tool_calls(raw_calls), refusal_message=refusal,
                finish_reason=_tool_finish_reason(finish))
    else
        Message(role=RoleAssistant, content=(isnothing(text) && !isnothing(refusal) ? nothing : something(text, "")),
                refusal_message=refusal, finish_reason=finish)
    end
    (; message=msg, usage=_parse_usage(received))
end

# Message content as text: a string as is; an array of content parts (the form
# requests use, which OpenAI-compatible servers may echo) joined from its text parts.
_content_text(s::AbstractString)::String = String(s)
_content_text(parts::AbstractVector)::String =
    join(p["text"] for p in parts
         if p isa AbstractDict && get(p, "type", "text") == "text" && get(p, "text", nothing) isa AbstractString)
_content_text(::Nothing) = nothing

# The finish reason of a turn that carries tool calls: the wire's own, with
# "tool_calls" standing in for none or "stop" (providers close tool turns with
# either). A turn cut at "length" or filtered keeps that reason, so no tool loop
# dispatches its partial calls.
_tool_finish_reason(reason::Union{Nothing,AbstractString})::String =
    isnothing(reason) || reason == STOP ? TOOL_CALLS : String(reason)

# OpenAI-wire `tool_calls` array → the neutral ToolCall vector.
_decode_tool_calls(raw::AbstractVector)::Vector{ToolCall} =
    [ToolCall(id=x["id"], func=GPTFunction(x["function"]["name"], _parse_tool_arguments(x["function"]["arguments"])))
     for x in raw]

"""Mutable accumulator for streaming Chat Completions chunks."""
@kwdef mutable struct StreamState
    content::IOBuffer = IOBuffer()
    refusal::IOBuffer = IOBuffer()
    tool_calls::Dict{Int, Dict{String,Any}} = Dict{Int, Dict{String,Any}}()
    finish_reason::Union{String, Nothing} = nothing
    usage::Union{TokenUsage, Nothing} = nothing
    # ── streaming-machine additions (0.11.3) ──
    # Text deltas collected by handlers, not yet forwarded to the callback;
    # the driver take!s and forwards verbatim (kills the take!/re-print churn).
    pending_delta::IOBuffer = IOBuffer()
    # In-band terminal stream error (e.g. Anthropic `error` event on HTTP 200);
    # non-nothing → the driver returns LLMFailure/LLMCallError, never LLMSuccess.
    error::Union{Nothing, Dict{String,Any}} = nothing
    # on_tool_call fire-once-per-index guard (final sweep at stream end).
    fired_tool_calls::Set{Int} = Set{Int}()
    # Provider-native content blocks captured verbatim during streaming so the
    # assistant turn can round-trip (e.g. Anthropic thinking blocks whose
    # signatures must be echoed unmodified). `raw_pending` holds in-flight
    # blocks by content-block index; `raw_json` accumulates partial tool-input
    # JSON per index; finalized blocks land in `raw_blocks` in arrival order.
    # `raw_provider` tags the dialect for ProviderContent construction.
    raw_blocks::Vector{Any} = Any[]
    raw_pending::Dict{Int,Dict{String,Any}} = Dict{Int,Dict{String,Any}}()
    raw_json::Dict{Int,String} = Dict{Int,String}()
    raw_provider::Union{Symbol,Nothing} = nothing
    # Undecodable `data:` payloads dropped while assembling THIS stream (the
    # process-global `_SSE_DROPPED_LINES` cannot be attributed to one request).
    # Rides the result as `sse_dropped`, so a turn built from a truncated wire is
    # distinguishable from a clean one.
    sse_dropped::Int = 0
    # Streamed tool-call (name, arguments) fragments by index: appended in
    # O(fragment) and joined onto `tool_calls` once, by `_tool_function!` —
    # re-concatenating a String per fragment is quadratic in the argument length.
    tool_fragments::Dict{Int,NTuple{2,IOBuffer}} = Dict{Int,NTuple{2,IOBuffer}}()
end

# The (name, arguments) fragment buffers of streamed tool call `idx`.
_tool_fragments!(state::StreamState, idx::Int)::NTuple{2,IOBuffer} =
    get!(() -> (IOBuffer(), IOBuffer()), state.tool_fragments, idx)

# The function dict of streamed tool call `idx`, with its buffered fragments joined
# on. The buffers are consumed, so reading a call again returns the same strings.
function _tool_function!(state::StreamState, idx::Int)::Dict{String,Any}
    fdict = state.tool_calls[idx]["function"]
    frags = pop!(state.tool_fragments, idx, nothing)
    if !isnothing(frags)
        fdict["name"] = fdict["name"] * takestring!(frags[1])
        fdict["arguments"] = fdict["arguments"] * takestring!(frags[2])
    end
    fdict
end

function _build_stream_message(state::StreamState)::Message
    content = String(take!(state.content))
    refusal = String(take!(state.refusal))
    # Echo complete captures only: a block still pending (its stop line was
    # dropped as malformed) means the capture is incomplete — fall back to
    # neutral reconstruction rather than echo a partial turn.
    # Bind the provider tag to a local so its `Symbol` type is carried past the
    # `isnothing` guard into the ProviderContent constructor (a re-read of the
    # mutable field would stay `Union{Symbol,Nothing}`).
    provider = state.raw_provider
    pc = (isnothing(provider) || isempty(state.raw_blocks) ||
          !isempty(state.raw_pending)) ? nothing :
         ProviderContent(provider, state.raw_blocks)
    # The finish reason is reported as it arrived: none when the provider sent none.
    if !isempty(state.tool_calls)
        tcalls = map(sort!(collect(keys(state.tool_calls)))) do idx
            tc_data = state.tool_calls[idx]
            fdict = _tool_function!(state, idx)
            args = _parse_tool_arguments(fdict["arguments"])   # "" → Dict{String,Any}() (zero-arg tool call)
            ToolCall(id=tc_data["id"], func=GPTFunction(fdict["name"], args),
                     thought_signature=get(tc_data, "thought_signature", nothing))
        end
        # Keep accumulated text ALONGSIDE the tool calls: providers emit
        # both in one turn and the non-streaming decoders already preserve both.
        Message(role=RoleAssistant, content=(isempty(content) ? nothing : content),
                tool_calls=tcalls, finish_reason=_tool_finish_reason(state.finish_reason),
                provider_content=pc)
    elseif isempty(content) && !isempty(refusal)
        Message(role=RoleAssistant, refusal_message=refusal,
                finish_reason=state.finish_reason, provider_content=pc)
    else
        Message(role=RoleAssistant, content=content,
                finish_reason=state.finish_reason, provider_content=pc)
    end
end

# ─── Wire-translation seam ───────────────────────────────────────────────────
# Generics translate between the neutral Chat/Message IR and a provider's wire
# format. The methods below are the OpenAI-wire defaults, dispatched on
# `OpenAIWireEndpointSpec` — inherited by OPENAI, Azure, the Gemini OpenAI-compat
# shim, GenericOpenAIEndpoint, and DeepSeek. Providers with a different wire
# (Anthropic, native Gemini) subtype `ServiceEndpoint` directly and override them;
# a bare `ServiceEndpoint` subtype with no override fails loud (MethodError) here
# rather than emitting OpenAI-shaped requests at a foreign API.
# `chatrequest!`/`_chatrequeststream` call ONLY these generics plus the streaming
# seam `handle_sse_event!` (src/sse.jl), so the retry/HTTP/cost/tool-loop/streaming
# orchestration stays provider-agnostic.

"""
    encode_request(service, chat::Chat) -> String

Serialize `chat` into the provider's request body. The `OpenAIWireEndpoint`
default emits OpenAI Chat Completions JSON.
"""
encode_request(service::OpenAIWireEndpointSpec, chat::Chat) = JSON.json(chat)

"""
    decode_response(service, resp::HTTP.Response)

Parse a provider's 200 response into `(; message::Message, usage::Union{TokenUsage,Nothing})`.
The `OpenAIWireEndpoint` default reads OpenAI Chat Completions (`extract_message`).
"""
decode_response(service::OpenAIWireEndpointSpec, resp::HTTP.Response) = extract_message(resp)

# ─── Streaming driver helpers ────────────────────────────────────────────────

"""
    _CloseRef(stop::CancelToken) <: Ref{Bool}

The `close` flag handed to streaming callbacks. Setting it `true` — from the
callback or from any other task — also cancels `stop`, the token the call's
request context is aborted on, so a stop takes effect at once rather than at the
next chunk. The flag is atomic, so a write from another task is well-defined.
"""
mutable struct _CloseRef <: Ref{Bool}
    @atomic flag::Bool
    const stop::CancelToken
    _CloseRef(stop::CancelToken) = new(false, stop)
end
Base.getindex(r::_CloseRef)::Bool = @atomic r.flag
function Base.setindex!(r::_CloseRef, v)
    flag = convert(Bool, v)
    @atomic r.flag = flag
    flag && cancel!(r.stop)
    return r
end

# An exception raised by user code (a streaming callback or `on_tool_call`). The
# field is named neither `error` nor `cause`: the transport and teardown classifiers
# walk those, and a failing callback is never connection noise.
struct _UserCallbackError <: Exception
    thrown::Any
end

# Per-call streaming control, shared by every attempt of one call: the caller's
# token (a stop from it reports source `:token`); the stop token every attempt's
# request context is aborted on — cancelled by the caller's token and by the
# callback's close flag (source `:callback`); whether user code has run (a retry
# after that would replay output); and the first exception user code raised,
# recorded before HTTP.jl can replace it with its rendering of the aborted exchange.
mutable struct _StreamCtl
    const token::Union{Nothing,CancelToken}
    const stop::CancelToken
    const close::_CloseRef
    fired::Bool
    failure::Union{Nothing,_UserCallbackError}
end
function _StreamCtl(token::Union{Nothing,CancelToken})
    stop = CancelToken()
    _StreamCtl(token, stop, _CloseRef(stop), false, nothing)
end

# The typed stop a stopped call reports; the caller's token wins over the close flag.
function _stop_cause(ctl::_StreamCtl, t0::UInt64)::UniLMCancelled
    c = UniLMCancelled(iscancelled(ctl.token) ? :token : :callback, _elapsed_s(t0))
    @debug "stream stopped" source = c.source elapsed = c.elapsed
    c
end

# Run user code from a stream driver. It is user-visible output from its first
# instruction (no retry after it), its duration is not wire idle time, and what it
# throws is recorded and raised as `_UserCallbackError` — except an interrupt, which
# is the user's intent and propagates unchanged.
function _user_call(f, ctl::_StreamCtl, idle, args...)
    ctl.fired = true
    _enter_user!(idle)
    try
        f(args...)
    catch e
        e isa InterruptException && rethrow()
        err = _UserCallbackError(e)
        isnothing(ctl.failure) && (ctl.failure = err)
        throw(err)
    finally
        _exit_user!(idle)
    end
end

"""
    _flush_delta!(callback, state::StreamState, close_ref) -> Nothing

Forward collected-but-unsent text deltas to the streaming callback verbatim.
Handlers collect deltas on `state.pending_delta`; the driver forwards them
directly — no `take!`/re-print churn, no byte-offset diffing (which broke on
multibyte boundaries and was O(n²)).
"""
function _flush_delta!(callback, state::StreamState, close_ref)::Nothing
    delta = takestring!(state.pending_delta)
    isnothing(callback) || isempty(delta) || callback(delta, close_ref)
    nothing
end

"""
    _fire_tool_calls!(on_tool_call, state::StreamState, stream_done::Bool) -> Nothing

Provider-agnostic `on_tool_call` completion detection. A
tool call at index `i` is complete when (a) `i` is no longer the max index (a
later call started), OR (b) its entry carries `"complete" => true` (Anthropic
`content_block_stop` / Gemini whole-part functionCall), OR (c) the stream is
done (`stream_done` — the final sweep). Fires at most once per index
(`state.fired_tool_calls`), marked before the callback runs, so what
`on_tool_call` throws propagates without a re-fire (the drivers pass it wrapped
in `_user_call`). Empty accumulated arguments parse as `Dict{String,Any}()` via
`_parse_tool_arguments`.
"""
function _fire_tool_calls!(on_tool_call, state::StreamState, stream_done::Bool)::Nothing
    (isnothing(on_tool_call) || isempty(state.tool_calls)) && return nothing
    maxidx = maximum(keys(state.tool_calls))
    for idx in sort!(collect(keys(state.tool_calls)))
        idx in state.fired_tool_calls && continue
        tc_data = state.tool_calls[idx]
        stream_done || idx < maxidx || get(tc_data, "complete", false) === true || continue
        fdict = _tool_function!(state, idx)
        args = try
            _parse_tool_arguments(fdict["arguments"])
        catch e
            e isa InterruptException && rethrow()
            # A COMPLETE call's arguments cannot improve later: warn once, never retry.
            push!(state.fired_tool_calls, idx)
            @warn "on_tool_call: undecodable tool-call arguments; not firing" index = idx exception = e
            continue
        end
        push!(state.fired_tool_calls, idx)   # before the user callback: a throwing callback must not re-fire
        on_tool_call(ToolCall(id=tc_data["id"], func=GPTFunction(fdict["name"], args),
                              thought_signature=get(tc_data, "thought_signature", nothing)))
    end
    nothing
end

"""
    _stream_error_result(chat, err::Dict{String,Any}, request_id, sse_dropped=0)

Map an in-band SSE `error` payload (`state.error`) to a typed non-success
result: a numeric error code (Gemini; OpenAI-compatible servers such as vLLM)
preserves the reported status;
`overloaded_error` is the documented
529-equivalent → `LLMFailure(status=529)` (status-keyed policies see it);
any other in-band error type → `LLMCallError` (no fabricated HTTP status, and
no drop count — `LLMCallError` describes an exception, not a decoded stream).
"""
function _stream_error_result(chat::Chat, err::Dict{String,Any}, request_id, sse_dropped::Int=0)
    code = get(err, "code", nothing)
    if code isa Integer && 400 <= code <= 599
        return LLMFailure(status=Int(code), response=JSON.json(err), self=chat,
                          request_id=request_id, sse_dropped=sse_dropped)
    end
    inner = get(err, "error", nothing)
    etype = inner isa AbstractDict ? get(inner, "type", "") : ""
    etype == "overloaded_error" ?
        LLMFailure(status=529, response=JSON.json(err), self=chat, request_id=request_id,
                   sse_dropped=sse_dropped) :
        LLMCallError(error=JSON.json(err), self=chat, status=nothing, request_id=request_id)
end

# Shared stream finalization: final tool-call sweep, message assembly, terminal
# callback. Used by the normal end-of-stream path and by the teardown-recovery
# rule. AT MOST ONCE per attempt, structurally: `m` records the assembled
# message and is itself the guard, because assembly `take!`s the accumulation
# buffers — a second call would deliver ANOTHER terminal callback carrying an
# EMPTY message and commit that empty turn.
function _finalize_stream_message!(state::StreamState, callback, on_tool_call,
                                   close_ref::Ref{Bool}, m::Ref{Union{Message,Nothing}})
    recorded = m[]
    isnothing(recorded) || return (; msg=recorded, usage=state.usage)
    _fire_tool_calls!(on_tool_call, state, true)   # final sweep BEFORE the terminal callback
    msg = _build_stream_message(state)
    m[] = msg          # record BEFORE the callback: re-entry must find the turn, not rebuild it
    !isnothing(callback) && callback(msg, close_ref)
    (; msg, usage=state.usage)
end

# Commit a completed streamed turn: history update, typed success, cost accrual.
# Shared by the clean end-of-stream path and the teardown-recovery path, so a
# turn recovered from teardown noise is committed exactly like a clean one.
function _stream_success(chat::Chat, msg::Message, usage::Union{TokenUsage,Nothing},
                         sse_dropped::Int)
    update!(chat, msg)
    result = LLMSuccess(message=msg, self=chat, usage=usage, sse_dropped=sse_dropped)
    _accumulate_cost!(chat, result)
    return result
end

"""
    _stream_attempt(chat, body, callback, on_tool_call, cfg, t0, io_ref, ctl) -> (; result, resp)

ONE streaming connection attempt with fresh accumulation state. User code
(`callback`, `on_tool_call`) runs through [`_user_call`](@ref). Returns the typed result
plus the `HTTP.Response` (`nothing` when the turn was recovered from teardown noise or a
late stop) so the caller can honor `Retry-After`; throws on connect/first-byte timeout
and transport failures — retry classification is the caller's job — on a stop
(`UniLMCancelled`) and on a user-code failure (`_UserCallbackError`). A breach of the
byte-gap idle bound throws `UniLMTimeout(:stream_idle, …)` wherever in the attempt it
fires (the native read-idle timer also bounds the response-header wait, so it can
undercut the request-phase bound; see `_classify_stream_timeout`). `StreamState`, the
SSE line carry, and the raw byte log are all locals: a retried attempt cannot inherit
partial SSE state.
"""
function _stream_attempt(chat::Chat, body, callback, on_tool_call,
                         cfg::RequestConfig, t0::UInt64, io_ref, ctl::_StreamCtl)
    state = StreamState()
    m = Ref{Union{Message,Nothing}}(nothing)
    raw_buffer = IOBuffer()  # wire bytes for non-200/truncation reporting (a streamed resp.body is empty)
    idle = Ref{Union{Nothing,_IdleGuard}}(nothing)   # armed at the first byte; stays nothing when disabled
    # Request-phase bound, recorded before it unwinds through HTTP.jl (see
    # `_with_recorded_deadline`): the catch restores it as the surfaced cause
    # when the library's teardown of the bound-closed socket replaces it.
    bound = Ref{Union{Nothing,UniLMTimeout}}(nothing)
    cb = isnothing(callback) ? nothing : (x, c) -> _user_call(callback, ctl, idle[], x, c)
    otc = isnothing(on_tool_call) ? nothing : tc -> _user_call(on_tool_call, ctl, idle[], tc)
    # SSE must reach the parser uncompressed: some providers (e.g. Anthropic) gzip even
    # streamed responses, and raw gzip bytes fail every line's decode, so no message is
    # built (→ LLMFailure). Request identity encoding and disable decompression so
    # `data:` lines arrive verbatim.
    stream_headers = push!(copy(auth_header(chat.service)), "Accept-Encoding" => "identity")
    try
        # Seam-routed: _http_open applies the native stream timeout kwargs plus
        # status_exception=false and retry=false (HTTP.jl's internal retries would
        # silently multiply the attempt budget) and aborts the exchange on `ctl.stop`;
        # decompress=false passes through.
        resp = _http_open("POST", get_url(chat), stream_headers; cfg, t0, cancel=ctl.stop,
                          decompress=false) do io
            io_ref[] = io
            carry = IOBuffer()                 # layer-1 partial-line carry
            current_event = Ref("")            # layer-2 sticky event name
            status = :continue
            # First byte = response headers received. The request-phase deadline guards
            # the whole send/first-byte exchange; the total deadline governs a stream
            # only up to this point — after it, only the idle guard runs (a long
            # healthy stream is not a failure).
            _with_recorded_deadline(() -> begin
                    write(io, body)
                    HTTP.closewrite(io)
                    HTTP.startread(io)
                end, () -> close(io),
                min(_remaining_s(cfg, t0), cfg.request_timeout), :request, bound)
            idle[] = _idle_guard(() -> close(io), cfg.stream_idle_timeout)
            # `eof` first: after `[DONE]` it consumes the body's end, which keeps the
            # connection reusable. A stop aborts the connection, so it never blocks here.
            while !eof(io) && !iscancelled(ctl.stop) && status === :continue
                raw = String(readavailable(io))
                _touch!(idle[])
                write(raw_buffer, raw)
                status = _sse_dispatch!(chat.service, carry, current_event, raw, state)
                _fire_tool_calls!(otc, state, false)
                _flush_delta!(cb, state, ctl.close)
            end
            if status === :continue && !iscancelled(ctl.stop)
                # EOF flush: a final line the server never '\n'-terminated
                # (e.g. `data: [DONE]` as the very last bytes) is still one
                # complete line — dispatch it before finalizing.
                tail = takestring!(carry)
                if !isempty(tail)
                    status = _sse_dispatch!(chat.service, carry, current_event, tail * "\n", state)
                    _fire_tool_calls!(otc, state, false)
                    _flush_delta!(cb, state, ctl.close)
                end
            end
            # A stop ends the attempt here, before finalization and without draining
            # the body; the catch decides whether a completed turn still stands.
            iscancelled(ctl.stop) && throw(_stop_cause(ctl, t0))
            # Terminal contract: `:done` is the sentinel EOS
            # ([DONE] / message_stop). Gemini has NO sentinel — its handler
            # never returns :done; its stream ends at EOF with finishReason
            # recorded. EOF with NEITHER signal = truncated/garbage stream
            # (or a non-200 error body) → no message → LLMFailure below.
            finished = status === :done ||
                       (status === :continue && !isnothing(state.finish_reason))
            if isnothing(state.error)
                if finished
                    _finalize_stream_message!(state, cb, otc, ctl.close, m)
                else
                    # No terminal, no exception: a guard's close landing as the driver
                    # entered a user callback truncates the read into a clean EOF, so
                    # the loop just ends. Raise what the throwing path would have
                    # raised, or the kill reads as a 200 with partial bytes (or, worse,
                    # as a truncated success).
                    breach = _exit_breach(idle[], bound, cfg)
                    breach === nothing || throw(breach)
                end
            end
            # A stop from the terminal callback leaves the recorded turn standing; the
            # aborted body is not drained.
            iscancelled(ctl.stop) || HTTP.closeread(io)
        end
        serr = state.error
        if !isnothing(serr)
            # In-band `error` event on an HTTP-200 stream: never LLMSuccess.
            return (; result=_stream_error_result(chat, serr, _get_request_id(resp),
                                                  state.sse_dropped), resp)
        elseif resp.status == 200 && !isnothing(m[])
            return (; result=_stream_success(chat, m[]::Message, state.usage,
                                             state.sse_dropped), resp)
        else
            return (; result=LLMFailure(status=resp.status, response=takestring!(raw_buffer),
                                        self=chat, request_id=_get_request_id(resp),
                                        sse_dropped=state.sse_dropped), resp)
        end
    catch e
        # Chain-walk: an interrupt nested inside a wrapper must still surface first.
        _find_exception(x -> x isa InterruptException, e) !== nothing && rethrow()
        # User code failed: its exception is the outcome — never teardown noise and
        # never retried, whatever HTTP.jl surfaced while the exchange unwound.
        failure = ctl.failure
        isnothing(failure) || throw(failure)
        # Byte-gap idle breach: classified by `_classify_stream_timeout` from the
        # seam's armed-timer set (our guard's close echo, or the native read-idle
        # timer — which can fire before the first byte too, while the
        # response-header wait is still in progress). Never a retryable
        # transport failure, regardless of when in the attempt it fired.
        breach = _classify_stream_timeout(e, idle[], cfg, t0)
        stopped = iscancelled(ctl.stop)
        # Teardown of an exchange that already produced its answer. The turn is
        # complete when EITHER the terminal message is recorded (`m[]`) or the
        # provider's own completion marker is (`state.finish_reason` — the only
        # signal an EOF-less provider gives, and all that survives when the
        # connection dies on the read that would have carried the sentinel). The
        # generation is billed and its deltas are already delivered, so teardown
        # noise — or a stop arriving after the terminal — must neither discard it
        # nor re-POST it (the caller's retry limbs would bill a second generation).
        # Failures that are NOT teardown-shaped — a decoding bug — still surface
        # below. Finalization is at-most-once, so this never doubles the terminal
        # callback.
        if isnothing(state.error) && (!isnothing(m[]) || !isnothing(state.finish_reason)) &&
           (stopped || _stream_teardown_noise(e, breach))
            fin = _finalize_stream_message!(state, cb, otc, ctl.close, m)
            return (; result=_stream_success(chat, fin.msg, fin.usage, state.sse_dropped), resp=nothing)
        end
        # Stopped before the turn completed: whatever surfaced (HTTP.jl's rendering
        # of the aborted exchange, a racing bound) is the stop's echo.
        stopped && throw(_stop_cause(ctl, t0))
        breach === nothing || throw(breach)
        # Recorded request-phase bound: the typed cause that initiated the
        # teardown takes precedence over whatever the library surfaced while
        # unwinding it (e.g. EPIPE from writing to the socket the bound
        # closed). With no displacement this rethrows the same timeout the
        # attempt already threw; it never masks an error that arrived with no
        # bound fired.
        bt = bound[]
        bt === nothing || throw(bt)
        # Remaining native timeouts are connect-phase (connect/TLS labels); map them
        # to the same phase-attributed UniLMTimeout the non-stream seam produces so a
        # raw HTTP.TimeoutError never leaks as the failure cause.
        mapped = _map_native_timeout(e, cfg, min(_remaining_s(cfg, t0), cfg.request_timeout), t0)
        mapped === nothing || throw(mapped)
        rethrow()
    finally
        _disarm!(idle[])
        # One statement per attempt, on every exit — a retried attempt starts from a
        # fresh StreamState, so each connection reports only what IT dropped.
        _warn_sse_drops(state.sse_dropped, chat.model, "chat stream")
    end
end

"""
    _stream_drive(chat, body, callback, on_tool_call, cfg, t0, token) -> LLMRequestResponse

Streaming request loop. Retry is legal only while NO user code has run — a retried
attempt after user-visible output would replay or reorder it. Retryable pre-callback
outcomes: retryable HTTP status, the in-band `overloaded_error` (documented 529
equivalent — `_stream_error_result` maps it to `LLMFailure(status=529)`, so the status
rule covers both twins), a connect/request-phase `UniLMTimeout`, and transport IO
failures. Backoff/budget arithmetic is `_retry_pause` — identical to the non-stream
loop — and a backoff wakes at once on a stop. A stop (`token`, or the callback's close
flag) ends the call with `UniLMCancelled` in `cause`, never retried; an exception from
user code ends it with that exception in `cause`. `InterruptException` always
rethrows; every other failure becomes a typed result value.
"""
function _stream_drive(chat::Chat, body, callback, on_tool_call, cfg::RequestConfig, t0::UInt64,
                       token::Union{Nothing,CancelToken})
    io_ref = Ref{Union{HTTP.Stream,Nothing}}(nothing)
    ctl = _StreamCtl(token)
    link = _on_cancel(() -> cancel!(ctl.stop), token)   # the caller's token stops the call
    attempt = 0
    try
        while true
            attempt += 1
            _remaining_s(cfg, t0) <= 0.0 &&
                throw(UniLMTimeout(:deadline, _elapsed_s(t0), cfg.total_deadline))
            outcome = try
                _stream_attempt(chat, body, callback, on_tool_call, cfg, t0, io_ref, ctl)
            catch e
                _find_exception(x -> x isa InterruptException, e) !== nothing && rethrow()
                u = _unwrap_exception(e)
                (_retryable_exception(u) && !ctl.fired && attempt < cfg.max_attempts) || rethrow()
                action, delay = _retry_pause(cfg, t0, attempt, nothing)
                if action !== :sleep
                    @warn "stream retry abandoned: backoff exceeds the remaining deadline" attempt delay
                    rethrow()
                end
                @debug "stream attempt failed; retrying" attempt exception = u
                _cancel_sleep(ctl.stop, delay) && throw(_stop_cause(ctl, t0))
                continue
            end
            result = outcome.result
            if result isa LLMFailure && _is_retryable(result.status) &&
               !ctl.fired && attempt < cfg.max_attempts
                action, delay = _retry_pause(cfg, t0, attempt, outcome.resp)
                if action === :sleep
                    @debug "retryable streamed status; retrying" attempt status = result.status
                    _cancel_sleep(ctl.stop, delay) && throw(_stop_cause(ctl, t0))
                    continue
                end
                # Budget cut: the last REAL outcome is the truthful answer — never a
                # fabricated timeout. Stated once.
                @warn "returning last streamed failure: backoff exceeds the remaining deadline" attempt status = result.status delay
            end
            return result
        end
    catch e
        _find_exception(x -> x isa InterruptException, e) !== nothing && rethrow()
        u = _unwrap_exception(e)
        req_id = !isnothing(io_ref[]) ? _get_request_id(io_ref[]) : _get_request_id(e)
        u isa _UserCallbackError &&
            return LLMCallError(error=_error_text(u.thrown), self=chat, status=nothing,
                                request_id=req_id, cause=u.thrown isa Exception ? u.thrown : nothing)
        u isa UniLMTimeout &&
            return LLMCallError(error=sprint(showerror, u), self=chat, status=nothing,
                                request_id=req_id, cause=u)
        statuserror = hasproperty(u, :status) ? u.status : nothing
        return LLMCallError(error=_error_text(e), self=chat, status=statuserror,
                            request_id=req_id, cause=u isa Exception ? u : nothing)
    finally
        _off_cancel(token, link)
    end
end

# With history on, the reply is appended to the conversation; one that ends with an
# assistant message could not take it (`push!` refuses consecutive assistant turns).
# Refuse that before the billed call rather than after it.
function _validate_reply_slot(chat::Chat)::Nothing
    chat.history && !isempty(chat) && last(chat).role == RoleAssistant && throw(InvalidConversationError(
        "the conversation ends with an assistant message, so the reply could not be appended; " *
        "add a user message first, or set history=false to send it as a prefill"))
    nothing
end

"""
    _chatrequeststream(chat, body, callback=nothing; on_tool_call=nothing,
                       cfg=_resolve_config(nothing), t0=time_ns(), cancel=_current_cancel()) -> Task

Spawn the streaming request task. `cfg`, `t0` and the cancellation token are
resolved/stamped at call entry — BEFORE the spawn — and the task closes over them, so
a running stream is immune to later changes of the process-default configuration.

The returned task throws only for a user `InterruptException` (every other failure is a
typed result value), so `fetch` then raises a `TaskFailedException` whose
`task.exception` is the `InterruptException` — callers catching interrupts around
`fetch` must unwrap it.
"""
function _chatrequeststream(chat::Chat, body, callback=nothing; on_tool_call=nothing,
                            cfg::RequestConfig=_resolve_config(nothing),
                            t0::UInt64=time_ns(),
                            cancel::Union{Nothing,CancelToken}=_current_cancel())
    Threads.@spawn _stream_drive(chat, body, callback, on_tool_call, cfg, t0, cancel)
end


"""
    chatrequest!(chat::Chat; config=nothing, callback=nothing, on_tool_call=nothing, cancel=nothing)

Send `chat` to its provider and return a typed result.

Non-streaming (`chat.stream !== true`): returns `LLMSuccess`, `LLMFailure`, or
`LLMCallError`. Transient statuses (408/429/500/502/503/504/529) are retried with
backoff and jitter under the resolved [`RequestConfig`](@ref) (`max_attempts`,
`total_deadline`; `Retry-After` honored). Timeouts surface as `LLMCallError` with
`status = nothing` and the `UniLMTimeout` in `cause` — no fabricated HTTP statuses.

Streaming (`chat.stream === true`): returns a `Task` whose `fetch` yields the same
typed results. `callback(chunk::Union{String,Message}, close::Ref{Bool})` receives
text deltas then the final assembled `Message`; `on_tool_call(tc::ToolCall)` fires
once per completed streamed tool call. Setting `close[] = true` — in the callback or
from any other task — stops the stream at once: the call ends with
`LLMCallError(status=nothing, cause=UniLMCancelled(:callback, …))`, unless the
provider's terminal event was already recorded, in which case the turn stands (a
success, committed). An exception thrown by `callback` or `on_tool_call` ends the
call with that exception in `cause`: never retried, nothing committed, and no
callback runs after it. Time spent in these callbacks does not count toward
`stream_idle_timeout`, which bounds only the gap between bytes off the socket. A
user `InterruptException` is never converted into a result value: it propagates, so
`fetch` on the streaming task throws a `TaskFailedException` whose `task.exception`
is the `InterruptException`.

`config::Union{Nothing,RequestConfig}`: per-call timeout/retry budget; `nothing`
resolves the ambient configuration (`with_request_config` scope, else the process
default set via `set_default_config!`).

`cancel::Union{Nothing,CancelToken}`: a [`CancelToken`](@ref); `nothing` resolves the
ambient token of [`with_cancel`](@ref), at call entry. A cancel at any point — before
connecting, during the response-header wait, mid-stream, or during a retry backoff —
ends the call with `LLMCallError(status=nothing, cause=UniLMCancelled(:token, …))`:
never retried, nothing committed to `chat`, no terminal callback. A pre-cancelled token
sends nothing. A TCP connect or TLS handshake already in progress cannot be interrupted
(HTTP.jl 2.7.1), so a cancel during one takes effect when it completes or reaches
`connect_timeout`.

Local validation throws before any network I/O, streaming or not: `ArgumentError`
when `chat.service` is an endpoint type that declares its capabilities and does not
list `:chat` (a custom endpoint declares none and is dispatched unvalidated), or when
the provider's encoder rejects the request (e.g. an option the provider or model does
not support); `InvalidConversationError` when `chat.history` is on and the
conversation ends with an assistant message, so the reply could not be appended.
"""
function chatrequest!(chat::Chat; config::Union{Nothing,RequestConfig}=nothing,
                      callback=nothing, on_tool_call=nothing,
                      cancel::Union{Nothing,CancelToken}=nothing)
    _validate_declared_capability(chat.service, :chat, "Chat Completions API")
    _validate_reply_slot(chat)
    body = encode_request(chat.service, chat)
    cfg = _resolve_config(config)
    tok = _resolve_cancel(cancel)
    t0 = time_ns()
    chat.stream === true && return _chatrequeststream(chat, body, callback; on_tool_call, cfg, t0, cancel=tok)
    local resp
    try
        resp = _http_with_retries(cfg, t0, "POST", get_url(chat),
                                  auth_header(chat.service), body; cancel=tok)
        if resp.status == 200
            extracted = decode_response(chat.service, resp)
            update!(chat, extracted.message)
            result = LLMSuccess(message=extracted.message, self=chat, usage=extracted.usage)
            _accumulate_cost!(chat, result)
            return result
        else
            # Retry/backoff already happened inside the shared loop; the last
            # real response is the truthful outcome (a budget-exhausted 429 is
            # a 429, not a fabricated timeout).
            return LLMFailure(status=resp.status, response=String(resp.body),
                              self=chat, request_id=_get_request_id(resp))
        end
    catch e
        e isa InterruptException && rethrow()
        e isa UniLMTimeout && return LLMCallError(error=sprint(showerror, e), self=chat,
                                                  status=nothing, cause=e)
        statuserror = hasproperty(e, :status) ? e.status : nothing
        req_id = @isdefined(resp) ? _get_request_id(resp) : _get_request_id(e)
        return LLMCallError(error=_error_text(e), self=chat, status=statuserror,
                            request_id=req_id, cause=e isa Exception ? e : nothing)
    end
end

"""
    chatrequest!(; messages, config=nothing, cancel=nothing, kwargs...)
    chatrequest!(; systemprompt, userprompt, config=nothing, cancel=nothing, kwargs...)

Build a [`Chat`](@ref) from keyword arguments, send it with
`chatrequest!(chat; config, cancel)`, and return that result (the reply is appended
to `result.self`).

The conversation is EITHER `messages` — copied, so the caller's vector is never
mutated — OR `systemprompt` and `userprompt`, each a `String` or a [`Message`](@ref).
Passing neither, only one prompt, or `messages` together with a prompt throws
`ArgumentError` before any network I/O. Every other keyword is a [`Chat`](@ref)
field, for example:

- `service::ServiceEndpointSpec = OPENAIServiceEndpoint`: The provider endpoint type or instance.
- `model::String`: The model; the service's default when omitted.
- `history::Bool = true`: Whether the reply is appended to the conversation. It does not change what is sent.
- `tools::Union{Vector{Tool},Nothing} = nothing`: A list of tools the model may call.
- `tool_choice::Union{String,GPTToolChoice,Nothing} = nothing`: Controls which (if any) function is called by the model. e.g. "auto", "none", `GPTToolChoice`.
- `parallel_tool_calls::Union{Bool,Nothing} = false`: Whether to enable parallel function calling.
- `temperature::Union{Float64,Nothing} = nothing`: Sampling temperature (0.0-2.0). Higher values make output more random. Mutually exclusive with `top_p`.
- `top_p::Union{Float64,Nothing} = nothing`: Nucleus sampling parameter (0.0-1.0). Mutually exclusive with `temperature`.
- `n::Union{Int64,Nothing} = nothing`: Number of choices; must be 1 — a result carries a single choice.
- `stream::Union{Bool,Nothing} = nothing`: If `true`, the call returns a `Task` and streams deltas to `callback` (see `chatrequest!(chat)`).
- `stop::Union{Vector{String},String,Nothing} = nothing`: Up to 4 sequences where the API will stop generating further tokens.
- `max_tokens::Union{Int64,Nothing} = nothing`: The maximum number of tokens to generate in the chat completion.
- `presence_penalty::Union{Float64,Nothing} = nothing`: Number between -2.0 and 2.0. Positive values penalize new tokens based on whether they appear in the text so far.
- `response_format::Union{ResponseFormat,Nothing} = nothing`: An object specifying the format that the model must output. e.g., `ResponseFormat(type="json_object")`.
- `frequency_penalty::Union{Float64,Nothing} = nothing`: Number between -2.0 and 2.0. Positive values penalize new tokens based on their existing frequency in the text so far.
- `logit_bias::Union{AbstractDict{String,Float64},Nothing} = nothing`: Modify the likelihood of specified tokens appearing in the completion.
- `user::Union{String,Nothing} = nothing`: A unique identifier representing your end-user, which can help OpenAI to monitor and detect abuse.
- `seed::Union{Int64,Nothing} = nothing`: This feature is in Beta. If specified, the system will make a best effort to sample deterministically.
- `config::Union{Nothing,RequestConfig} = nothing`: Per-call timeout/retry budget; `nothing` resolves the ambient configuration (scoped, else process default).
- `cancel::Union{Nothing,CancelToken} = nothing`: Cancellation token; `nothing` resolves the ambient token (see `chatrequest!(chat)`).
"""
function chatrequest!(; kws...)
    has_messages = haskey(kws, :messages)
    has_messages && (haskey(kws, :systemprompt) || haskey(kws, :userprompt)) && throw(ArgumentError(
        "chatrequest!: pass either `messages` or `systemprompt` and `userprompt`, not both"))
    has_messages || (haskey(kws, :systemprompt) && haskey(kws, :userprompt)) || throw(ArgumentError(
        "chatrequest!: pass `messages`, or both `systemprompt` and `userprompt`"))
    messages = has_messages ? Vector{Message}(kws[:messages]) :
        [_prompt_message(RoleSystem, kws[:systemprompt]), _prompt_message(RoleUser, kws[:userprompt])]
    chatkws = filter(x -> x[1] ∉ (:messages, :userprompt, :systemprompt, :config, :cancel), kws)
    chatrequest!(Chat(; messages, chatkws...); config=get(kws, :config, nothing),
                 cancel=get(kws, :cancel, nothing))
end

_prompt_message(role::String, prompt::AbstractString)::Message = Message(; role, content=String(prompt))
_prompt_message(::String, prompt::Message)::Message = prompt


"""
    embeddingrequest!(emb::Embeddings; config=nothing, cancel=nothing) -> LLMRequestResponse

Send an Embeddings API request for the `input` in `emb`. Returns `EmbeddingSuccess`,
`EmbeddingFailure` (non-2xx), or `EmbeddingCallError` (network/parse/timeout). The
resulting vectors are filled into `emb.embeddings` in place and are also reachable via
`embedding_vectors(result)`.

Transient statuses (408/429/500/502/503/504/529) are retried with backoff and jitter
under the resolved [`RequestConfig`](@ref) (`config === nothing` resolves the ambient
configuration). Timeouts surface as `EmbeddingCallError` with `status = nothing` and
the `UniLMTimeout` in `cause`.

`cancel::Union{Nothing,CancelToken}` (`nothing`: the ambient token of
[`with_cancel`](@ref)): a cancel ends the call with `EmbeddingCallError(status=nothing,
cause=UniLMCancelled(:token, …))`, never retried; a pre-cancelled token sends nothing.
A TCP connect or TLS handshake already in progress finishes (or reaches
`connect_timeout`) before the cancel takes effect.

Throws `ArgumentError` before any network I/O when `emb.service` is an endpoint type
that declares its capabilities and does not list `:embeddings`.
"""
function embeddingrequest!(emb::Embeddings; config::Union{Nothing,RequestConfig}=nothing,
                           cancel::Union{Nothing,CancelToken}=nothing)
    _validate_declared_capability(emb.service, :embeddings, "Embeddings API")
    cfg = _resolve_config(config)
    tok = _resolve_cancel(cancel)
    t0 = time_ns()
    try
        body = JSON.json(emb)
        resp = _http_with_retries(cfg, t0, "POST", get_url(emb),
                                  auth_header(emb.service), body; cancel=tok)
        if resp.status == 200
            data = JSON.parse(resp.body; dicttype=Dict{String,Any})
            update!(emb, data["data"])
            return EmbeddingSuccess(embeddings=emb, usage=_parse_usage(data), raw=data)
        else
            return EmbeddingFailure(response=String(resp.body), status=resp.status)
        end
    catch e
        e isa InterruptException && rethrow()
        e isa UniLMTimeout && return EmbeddingCallError(error=sprint(showerror, e),
                                                        status=nothing, cause=e)
        statuserror = hasproperty(e, :status) ? e.status : nothing
        return EmbeddingCallError(error=_error_text(e), status=statuserror,
                                  cause=e isa Exception ? e : nothing)
    end
end
