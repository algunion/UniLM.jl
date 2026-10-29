# ============================================================================
# Shared SSE machine. Three layers:
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

Process-global means no caller can attribute a drop to its OWN request. Every
drop is therefore ALSO counted on the stream state that saw it, and that
per-stream count rides the result (`sse_dropped` on `LLMSuccess`/`LLMFailure`/
`ResponseSuccess`/`ResponseFailure`).
"""
const _SSE_DROPPED_LINES = Threads.Atomic{Int}(0)

"""
    _warn_sse_drops(n::Int, model, surface::AbstractString) -> Nothing

State a stream's dropped-payload count ONCE, at finalize. A drop means the
provider sent a `data:` payload this parser could not read, so the turn was
assembled from an incomplete wire — anomalous by construction, never routine,
hence `@warn` rather than the per-line `@debug`. Silent when nothing was dropped.
"""
function _warn_sse_drops(n::Int, model, surface::AbstractString)::Nothing
    n == 0 || @warn "SSE: undecodable data payloads dropped during the stream" count = n model surface
    nothing
end

"""
    _sse_complete_lines!(carry::IOBuffer, chunk::String) -> Vector{SubString{String}}

Layer 1 of the SSE machine: line assembly across arbitrary read boundaries.
Only `chunk` is scanned for its LAST `'\\n'` — the carry never holds one — so a
line spread over many reads costs O(line), not O(line²). A chunk without a
newline is appended to the carry; otherwise the carry and the chunk's complete
lines are joined once and the tail after the last `'\\n'` becomes the new carry.
The tail is stashed **verbatim** (never `strip` — stripping eats whitespace
inside JSON strings split at a read boundary). Complete lines are split on
`'\\n'`, a single trailing `'\\r'` is dropped per line (CRLF tolerance), and
empty lines are dropped. A line is returned (and later parsed) exactly once.
"""
function _sse_complete_lines!(carry::IOBuffer, chunk::String)::Vector{SubString{String}}
    lines = SubString{String}[]
    last_nl = findlast('\n', chunk)
    if isnothing(last_nl)
        write(carry, chunk)         # no complete line yet — stash verbatim
        return lines
    end
    head = SubString(chunk, 1, last_nl)
    data = iszero(position(carry)) ? head : (write(carry, head); SubString(takestring!(carry)))
    # Partial line after the LAST '\n': stash verbatim for the next read.
    last_nl < ncodeunits(chunk) && write(carry, SubString(chunk, last_nl + 1))
    for line in eachsplit(data, '\n')
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
`decode_stream_chunk` (removed in 0.11.3). Called by the stream driver
once per complete `data:` payload, with the current SSE event name. Mutates
`state` and returns
- `:continue` — keep reading;
- `:done`     — sentinel end-of-stream (`[DONE]`, Anthropic `message_stop`);
- `:error`    — terminal in-band error; the handler MUST store the decoded
  payload in `state.error` (the driver then returns a non-success result).
Text deltas are appended to BOTH `state.content` and `state.pending_delta`
(the driver forwards the latter to the streaming callback). Throwing on an
undecodable payload is safe: the dispatcher logs the payload, counts it in the
result's `sse_dropped` and drops it, the rest of the stream is read, and a stream
that ends without its turn is an `LLMFailure`.
The `OpenAIWireEndpoint` default reads OpenAI Chat Completions chunks and ends at
`[DONE]`; a provider without a sentinel (native Gemini) never returns `:done`, and
its stream ends at EOF with `state.finish_reason` recorded.

Public extension API (not exported): a backend adds a method for its own service
type (dispatch is on the first argument); see the Custom Backends guide. A streamed
[`chatrequest!`](@ref) on an endpoint with no method throws the `MethodError` for this
function before anything is sent, as a missing [`encode_request`](@ref) does, and so does
a streamed [`respond`](@ref) on one with no [`decode_agentic_stream`](@ref).
"""
function handle_sse_event! end

# The stream handler a streamed Chat reads its reply with, checked before anything is sent. The
# dispatcher drops a payload whose handler throws, so on an endpoint without one every payload of
# a paid reply would be dropped as undecodable: it is refused with the `MethodError` the handler's
# call raises, as an endpoint without an encoder is.
function _require_stream_handler(service)::Nothing
    hasmethod(handle_sse_event!, Tuple{Core.Typeof(service),String,SubString{String},StreamState}) ||
        throw(MethodError(handle_sse_event!, (service, "", SubString(""), StreamState())))
    nothing
end

# How a provider frames its stream into (event, payload) pairs: SSE for every
# built-in wire but Ollama's, whose lines are each one JSON payload (src/ollama.jl).
_stream_frames(service, carry::IOBuffer, current_event::Ref{String}, chunk::String) =
    _sse_events!(carry, current_event, chunk)

"""
    _sse_dispatch!(service, carry::IOBuffer, current_event::Ref{String},
                   chunk::String, state::StreamState) -> Symbol

Layer-3 glue: frames `chunk` via [`_sse_events!`](@ref) and feeds each data
payload to [`handle_sse_event!`](@ref). Returns the first non-`:continue`
status, else `:continue`. A COMPLETE line whose handler throws is
`@debug`-logged, counted in `_SSE_DROPPED_LINES` **and in `state.sse_dropped`**,
and DROPPED — never re-queued. An `InterruptException` is the user's, not the
payload's: it propagates.
"""
function _sse_dispatch!(service, carry::IOBuffer, current_event::Ref{String},
                        chunk::String, state::StreamState)::Symbol
    for (event, payload) in _stream_frames(service, carry, current_event, chunk)
        status = try
            handle_sse_event!(service, event, payload, state)
        catch e
            e isa InterruptException && rethrow()
            Threads.atomic_add!(_SSE_DROPPED_LINES, 1)
            state.sse_dropped += 1
            @debug "SSE: dropped undecodable data payload" event payload = String(payload) exception = (e, catch_backtrace())
            :continue
        end
        status === :continue || return status
    end
    :continue
end

"""
    _parse_tool_arguments(args::AbstractString) -> Dict{String,Any}

Parse a streamed tool call's accumulated `arguments` JSON string. A zero-arg
tool call may stream `arguments` as `""` (Anthropic sends no
`input_json_delta` for `{}` input) — that parses as an empty object instead
of destroying the turn. Anything else must be a JSON object; a
non-object payload throws a loud `ArgumentError`. Shared by
`_build_stream_message` and the driver's `on_tool_call` firing.
"""
function _parse_tool_arguments(args::AbstractString)::Dict{String,Any}
    isempty(strip(args)) && return Dict{String,Any}()
    parsed = JSON.parse(String(args); dicttype=Dict{String,Any})
    parsed isa Dict{String,Any} || throw(ArgumentError(
        "streamed tool-call arguments must be a JSON object, got $(typeof(parsed))"))
    parsed
end

# ─── OpenAI-wire default handler (Chat Completions SSE) ─────────────────────
# Dispatched on `OpenAIWireEndpointSpec`: OPENAI, DeepSeek, Azure, the Gemini
# OpenAI-compat shim, and GenericOpenAIEndpoint all inherit this wire. Providers
# with a different SSE dialect (Anthropic, native Gemini) override it. Contract:
# `[DONE]` is the ONLY end-of-stream — `finish_reason` is recorded but never
# terminal (the stream_options.include_usage chunk trails it); `choices` may
# be empty (usage-only chunks, Azure prompt-filter preambles) — iterate, never
# index [1]; `usage` is captured from any chunk. OpenAI names the served `model` and
# the billed `service_tier` on every chunk; the usage takes the stream's tier
# (`_chunk_tier!`: the latest one named), so a usage chunk without one still carries
# it. `delta.audio` updates assemble an audio reply (`_audio_delta!`), and a `delta.content` given
# as a list of chunks is read chunk by chunk (`_content_delta!`). A `{"error": …}` payload is
# terminal (`:error`): vLLM and OpenAI-compatible proxies report a mid-stream
# failure that way on the HTTP-200 stream, usually followed by `[DONE]`, so the
# partial text must not finalize. An error object is recorded as is — its numeric
# `code`, where one is given, is the server's status for the failure.
function handle_sse_event!(service::OpenAIWireEndpointSpec, event::AbstractString,
                           payload::AbstractString, state::StreamState)::Symbol
    payload == "[DONE]" && return (_finalize_thinking!(state); :done)
    parsed = JSON.parse(payload; dicttype=Dict{String,Any})
    parsed isa AbstractDict || return :continue
    err = get(parsed, "error", nothing)
    if !isnothing(err)
        state.error = err isa Dict{String,Any} ? err : parsed
        return :error
    end
    model = _wire_string(parsed, "model")
    isnothing(model) || (state.model = model)
    tier = _wire_string(parsed, "service_tier")
    isnothing(tier) || _chunk_tier!(service, state, tier)
    u = get(parsed, "usage", nothing)
    u isa AbstractDict && (state.usage = _token_usage_from(u; service_tier=state.service_tier))
    choices = get(parsed, "choices", nothing)
    choices isa AbstractVector || return :continue
    for cho in choices
        cho isa AbstractDict || continue
        delta = get(cho, "delta", nothing)
        delta isa AbstractDict && _content_delta!(state, get(delta, "content", nothing))
        fr = get(cho, "finish_reason", nothing)
        fr isa AbstractString && (state.finish_reason = fr; _finalize_thinking!(state))
        delta isa AbstractDict || continue
        r = get(delta, "refusal", nothing)
        r isa AbstractString && print(state.refusal, r)
        audio = get(delta, "audio", nothing)
        audio isa AbstractDict && _audio_delta!(state, audio)
        tcs = get(delta, "tool_calls", nothing)
        tcs isa AbstractVector || continue
        for tc_delta in tcs
            tc_delta isa AbstractDict || continue
            # Google's OpenAI-compatible stream sends each call whole, without the `index` an
            # OpenAI stream gives every call fragment (gemini-3.8-flash, 29 October 2026): such a
            # call takes the next index and is complete as it arrives.
            whole = !haskey(tc_delta, "index")
            idx = whole ? (isempty(state.tool_calls) ? 0 : maximum(keys(state.tool_calls)) + 1) : tc_delta["index"]
            entry = get!(state.tool_calls, idx) do
                Dict{String,Any}("id" => "", "type" => get(tc_delta, "type", "function"),
                    "function" => Dict{String,Any}("name" => "", "arguments" => ""))
            end
            whole && (entry["complete"] = true)
            id = get(tc_delta, "id", nothing)
            id isa AbstractString && !isempty(id) && (entry["id"] = id)
            sig = _google_signature(tc_delta)
            isnothing(sig) || (entry["thought_signature"] = sig)
            fd = get(tc_delta, "function", nothing)
            if fd isa AbstractDict
                name, args = _tool_fragments!(state, idx)
                n = get(fd, "name", nothing)
                n isa AbstractString && print(name, n)
                a = get(fd, "arguments", nothing)
                a isa AbstractString && print(args, a)
            end
        end
    end
    :continue
end

# The tier a chunk names becomes the stream's, which the usage takes. An endpoint whose
# response head names the tier first (`_stream_head!`) may keep that one.
_chunk_tier!(_, state::StreamState, tier::String)::Nothing = (state.service_tier = tier; nothing)

# A streamed `delta.content`. A string is the answer's next text. A list of chunks is the shape
# Mistral's reasoning models stream: "1. Thinking phase: `delta.content` is a list containing a
# `ThinkChunk` 2. Transition: a single list containing both a closing `ThinkChunk` and the first
# `TextChunk` 3. Answer phase: `delta.content` is a plain string"
# (https://docs.mistral.ai/studio/conversations/reasoning). Its text chunks are the answer, in
# order, read as the unstreamed reply's are (`_content_text`); the text chunks inside a thinking
# chunk are the reasoning, kept apart in one pending ThinkChunk (`_thinking_buffer!`).
_content_delta!(state::StreamState, text::AbstractString)::Nothing =
    (print(state.content, text); print(state.pending_delta, text); nothing)
function _content_delta!(state::StreamState, chunks::AbstractVector)::Nothing
    for chunk in chunks
        chunk isa AbstractDict || continue
        kind = get(chunk, "type", "text")
        if kind == "text"
            t = get(chunk, "text", nothing)
            t isa AbstractString && _content_delta!(state, t)
        elseif kind == "thinking"
            parts = get(chunk, "thinking", nothing)
            for p in (parts isa AbstractVector ? parts : Any[])
                t = p isa AbstractDict && get(p, "type", "text") == "text" ? get(p, "text", nothing) : nothing
                t isa AbstractString && print(_thinking_buffer!(state), t)
            end
        end
    end
    nothing
end
_content_delta!(::StreamState, _)::Nothing = nothing

# The pending ThinkChunk a stream's thinking text accumulates in. The finish reason or `[DONE]`
# closes it (`_finalize_thinking!`) into Mistral's own shape, `{"type": "thinking", "thinking":
# [{"type": "text", "text": …}]}`, which the message carries as `ProviderContent(:mistral, …)` for
# `reasoning_text`; a block still pending marks the capture incomplete, so a stream cut before then
# keeps no partial reasoning.
function _thinking_buffer!(state::StreamState)::IOBuffer
    blk = get!(state.raw_pending, 0) do
        state.raw_provider = :mistral
        Dict{String,Any}("type" => "thinking", "thinking" => IOBuffer())
    end
    blk["thinking"]::IOBuffer
end

# Close the pending ThinkChunk, if this stream holds one; another wire's pending capture (DeepSeek's
# `reasoning_content`, whose handler runs this one first) is that handler's to close.
function _finalize_thinking!(state::StreamState)::Nothing
    state.raw_provider === :mistral || return nothing
    blk = pop!(state.raw_pending, 0, nothing)
    isnothing(blk) || push!(state.raw_blocks, Dict{String,Any}("type" => "thinking",
        "thinking" => Any[Dict{String,Any}("type" => "text", "text" => takestring!(blk["thinking"]))]))
    nothing
end

# One update of a streamed audio reply: "Audio chunks may carry an ID, base64 data, or
# transcript text; the final audio update contains only its expiry timestamp" (Chat
# Completions reference, the chunk's `delta.audio`). The transcript is the turn's text, so
# it also goes to the streaming callback; an expiry that is no integer is a malformed
# update, as it is unstreamed (`_audio_expiry`).
function _audio_delta!(state::StreamState, a::AbstractDict)::Nothing
    s = _streamed_audio!(state)
    id = get(a, "id", nothing)
    id isa AbstractString && !isempty(id) && (s.id = String(id))
    data = get(a, "data", nothing)
    data isa AbstractString && print(s.base64, data)
    t = get(a, "transcript", nothing)
    if t isa AbstractString
        print(s.transcript, t)
        print(state.pending_delta, t)
    end
    e = get(a, "expires_at", nothing)
    isnothing(e) || (s.expires_at = _audio_expiry(e))
    nothing
end

_streamed_audio!(state::StreamState)::_StreamedAudio =
    (s = state.audio; s isa _StreamedAudio ? s : (state.audio = _StreamedAudio()))
