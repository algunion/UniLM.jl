# ─── Request configuration ───────────────────────────────────────────────────
# One immutable struct carries every timeout/retry knob (seconds; Inf means
# "disabled"). Resolution is struct-wise: whichever channel supplies the
# config supplies ALL fields — per-call kwarg > dynamic scope > process
# default > built-in defaults.

using Base.ScopedValues: ScopedValue, with

# Validate a timeout/deadline value. NaN is rejected EXPLICITLY: NaN compares
# false against every bound, so a plain `x <= 0` range check would accept it
# and the configured timeout would silently never fire — an unbounded wait.
function _validated_timeout(name::Symbol, v::Real)::Float64
    x = Float64(v)
    isnan(x) && throw(ArgumentError("$name must not be NaN"))
    x <= 0 && throw(ArgumentError("$name must be > 0 seconds (got $x); use Inf to disable"))
    return x
end

"""
    RequestConfig(; kwargs...)
    RequestConfig(base::RequestConfig; kwargs...)

Timeout and retry budget for every UniLM network operation. All time fields
are seconds (`Float64`); `Inf` disables that bound.

# Fields
- `connect_timeout::Float64 = 10.0`: per-attempt connection-establishment bound.
- `request_timeout::Float64 = 600.0`: per-attempt bound on a whole non-streaming exchange.
- `stream_idle_timeout::Float64 = 120.0`: maximum byte-gap between raw chunks of a stream.
- `total_deadline::Float64 = 900.0`: bound across ALL attempts including backoff; for streams it applies until the first byte.
- `max_attempts::Int = 3`: maximum wire attempts (`1` disables retries).
- `mcp_connect_timeout::Float64 = 120.0`: MCP spawn → `initialize` handshake bound.
- `mcp_request_timeout::Float64 = 120.0`: per MCP exchange bound.

The constructor throws `ArgumentError` for `NaN` or non-positive time values
(`NaN` is rejected explicitly because it compares false against every bound
and would silently disable the timeout), and for `max_attempts < 1`.

The two-argument form copies `base` with the named fields overridden, under
the same validation.

See also [`with_request_config`](@ref), [`set_default_config!`](@ref),
[`current_config`](@ref).
"""
Base.@kwdef struct RequestConfig
    connect_timeout::Float64     = 10.0
    request_timeout::Float64     = 600.0
    stream_idle_timeout::Float64 = 120.0
    total_deadline::Float64      = 900.0
    max_attempts::Int            = 3
    mcp_connect_timeout::Float64 = 120.0
    mcp_request_timeout::Float64 = 120.0

    function RequestConfig(connect_timeout::Real, request_timeout::Real,
                           stream_idle_timeout::Real, total_deadline::Real,
                           max_attempts::Integer, mcp_connect_timeout::Real,
                           mcp_request_timeout::Real)
        max_attempts >= 1 ||
            throw(ArgumentError("max_attempts must be >= 1 (got $max_attempts)"))
        new(_validated_timeout(:connect_timeout, connect_timeout),
            _validated_timeout(:request_timeout, request_timeout),
            _validated_timeout(:stream_idle_timeout, stream_idle_timeout),
            _validated_timeout(:total_deadline, total_deadline),
            Int(max_attempts),
            _validated_timeout(:mcp_connect_timeout, mcp_connect_timeout),
            _validated_timeout(:mcp_request_timeout, mcp_request_timeout))
    end
end

function RequestConfig(base::RequestConfig; kwargs...)
    fields = NamedTuple{fieldnames(RequestConfig)}(
        ntuple(i -> getfield(base, i), Val(fieldcount(RequestConfig))))
    return RequestConfig(; merge(fields, values(kwargs))...)
end
