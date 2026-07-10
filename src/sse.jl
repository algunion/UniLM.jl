# ============================================================================
# Shared SSE machine — Decision 1 of the wave-1 architecture spec
# (docs/superpowers/specs/2026-07-10-wave1-p0-architecture.md). Three layers:
#   1. _sse_complete_lines!  — line assembly across arbitrary read boundaries
#   2. _sse_events!          — SSE field framing → (event, data-payload) pairs
#   3. handle_sse_event!     — per-provider event handlers (the seam), driven
#      by _sse_dispatch!, which enforces the drop-don't-requeue policy.
# Replaces `_parse_chunk` and the `decode_stream_chunk` seam (removed 0.11.3).
# `_parse_response_stream_chunk` (Responses) and the Interactions
# `decode_agentic_stream` reuse layers 1–2 and keep their own event dispatch.
# ============================================================================

"""
Count of complete SSE lines whose payload failed to decode. Such lines are
`@debug`-logged and DROPPED — never re-queued into the carry buffer
(re-queueing a failed line without its newline is the Azure/proxy
stream-poisoning mechanism the old parsers shared). Observability hook for
tests and debugging; monotonically increasing, process-global.
"""
const _SSE_DROPPED_LINES = Threads.Atomic{Int}(0)

"""
    _sse_complete_lines!(carry::IOBuffer, chunk::String) -> Vector{SubString{String}}

Layer 1 of the SSE machine: line assembly across arbitrary read boundaries.
Prepends the stashed carry, splits at the LAST `'\\n'`, and stashes the tail
**verbatim** (never `strip` — stripping eats whitespace inside JSON strings
split at a read boundary). Complete lines are split on `'\\n'`, a single
trailing `'\\r'` is dropped per line (CRLF tolerance), and empty lines are
dropped. A line is returned (and later parsed) exactly once.
"""
function _sse_complete_lines!(carry::IOBuffer, chunk::String)::Vector{SubString{String}}
    data = string(String(take!(carry)), chunk)
    lines = SubString{String}[]
    last_nl = findlast('\n', data)
    if isnothing(last_nl)
        print(carry, data)          # no complete line yet — stash verbatim
        return lines
    end
    # Partial line after the LAST '\n': stash verbatim for the next read.
    last_nl < lastindex(data) && print(carry, SubString(data, nextind(data, last_nl)))
    for line in eachsplit(SubString(data, 1, last_nl), '\n')
        line = chopsuffix(line, "\r")
        isempty(line) || push!(lines, line)
    end
    lines
end

"""
    _sse_events!(carry::IOBuffer, current_event::Ref{String}, chunk::String)
        -> Vector{Tuple{String,SubString{String}}}

Layers 1+2: line assembly plus SSE field framing. `event:` updates
`current_event` (sticky until the next `event:` line — blank-line reset
semantics are not needed by any supported provider and blank lines are
dropped by layer 1). Each `data:` line emits `(current_event[], payload)`
where payload is the remainder after the colon minus AT MOST one leading
space — both `data:foo` and `data: foo` are accepted per the SSE spec (the
old Responses parser wrongly REQUIRED the space). `:` comment lines
(keep-alives) and all other fields (`id:`, `retry:`, unknown) are ignored.
"""
function _sse_events!(carry::IOBuffer, current_event::Ref{String}, chunk::String)
    events = Tuple{String,SubString{String}}[]
    for line in _sse_complete_lines!(carry, chunk)
        first(line) == ':' && continue                    # SSE comment (": keep-alive")
        colon = findfirst(':', line)
        field = isnothing(colon) ? line : SubString(line, 1, prevind(line, colon))
        value = isnothing(colon) ? SubString(line, nextind(line, lastindex(line))) :
                                   SubString(line, nextind(line, colon))
        startswith(value, ' ') && (value = SubString(value, nextind(value, 1)))
        if field == "event"
            current_event[] = String(value)
        elseif field == "data"
            push!(events, (current_event[], value))
        end
    end
    events
end

"""
    handle_sse_event!(service, event::AbstractString, payload::AbstractString,
                      state::StreamState) -> Symbol

Per-provider chat-streaming event handler — the seam that replaces
`decode_stream_chunk` (removed in 0.11.3). Called by [`_sse_dispatch!`](@ref)
once per complete `data:` payload, with the current SSE event name. Mutates
`state` and returns
- `:continue` — keep reading;
- `:done`     — sentinel end-of-stream (`[DONE]`, Anthropic `message_stop`);
- `:error`    — terminal in-band error; the handler MUST store the decoded
  payload in `state.error` (the driver then returns a non-success result).
Text deltas are appended to BOTH `state.content` and `state.pending_delta`
(the driver forwards the latter to the streaming callback). Throwing on an
undecodable payload is safe: the dispatcher logs, counts, and drops the line.
Unexported but documented: provider packages/tests may add methods for their
own service types (dispatch is on the first argument).
"""
function handle_sse_event! end

"""
    _sse_dispatch!(service, carry::IOBuffer, current_event::Ref{String},
                   chunk::String, state::StreamState) -> Symbol

Layer-3 glue: frames `chunk` via [`_sse_events!`](@ref) and feeds each data
payload to [`handle_sse_event!`](@ref). Returns the first non-`:continue`
status, else `:continue`. A COMPLETE line whose handler throws is
`@debug`-logged, counted in `_SSE_DROPPED_LINES`, and DROPPED — never
re-queued.
"""
function _sse_dispatch!(service, carry::IOBuffer, current_event::Ref{String},
                        chunk::String, state::StreamState)::Symbol
    for (event, payload) in _sse_events!(carry, current_event, chunk)
        status = try
            handle_sse_event!(service, event, payload, state)
        catch e
            Threads.atomic_add!(_SSE_DROPPED_LINES, 1)
            @debug "SSE: dropped undecodable data payload" event payload = String(payload) exception = (e, catch_backtrace())
            :continue
        end
        status === :continue || return status
    end
    :continue
end
