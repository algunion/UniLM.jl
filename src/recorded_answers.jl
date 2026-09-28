# ============================================================================
# Recorded System One answers: record a real answer once, replay it offline.
#
# The service does not answer a repeated request identically — probabilities
# move between byte-identical calls, and on an ambiguous input the winner can
# flip — so a test or a docs build that calls it live cannot be reproduced.
# Inside `with_recorded_answers` the HTTP exchange of `ask` and `list_models`
# (and so of `nl_dispatch` and `@branch`) passes through a directory of
# recordings on its way to the network.
#
# A recording is keyed by the exact request bytes, never a canonical form: the
# order of Choice options and of the state's keys moves the answer, so a
# reordered request is a different request.
# ============================================================================

"""
    ReplayMissError <: Exception

Thrown by [`ask`](@ref) and [`list_models`](@ref) — and so out of
[`nl_dispatch`](@ref) and [`@branch`](@ref) — inside a
[`with_recorded_answers`](@ref) scope when a request cannot be replayed: in
`:replay` mode the directory holds no recording of it, or in `:replay` or
`:record_missing` mode its recording file cannot be read as one. It is thrown
rather than returned as a [`SystemOneCallError`](@ref): a gap in the recordings
is not a service failure, and a caller's `r isa SystemOneSuccess || fallback()`
path must not absorb it as if it were one.

# Fields
- `dir::String`: the recordings directory that was searched.
- `key::String`: the recording key; the recording is the file `<dir>/<key>.json`.
- `method::String`, `path::String`: the request line, e.g. `"POST"` and `"/v1/systemone"`.
- `body::String`: the exact request body (empty for `GET /v1/models`).
- `reason::String`: `"no recording"`, or `"unreadable recording: …"` followed by
  what is wrong with the file — not JSON, no `response` object, a status other
  than 200, a `body` that is not a JSON object, or a `request_id` that is neither
  a string nor null.

`showerror` names the request line, the start of the key, the questions the
request asked, the directory or (for an unreadable recording) the file, and how to
record the answer: an unreadable file must be deleted first, because
`:record_missing` never overwrites one.
"""
struct ReplayMissError <: Exception
    dir::String
    key::String
    method::String
    path::String
    body::String
    reason::String
end

const _NO_RECORDING = "no recording"

function Base.showerror(io::IO, e::ReplayMissError)
    print(io, "ReplayMissError: ", e.method, " ", e.path, " (key ", first(e.key, 12), "…)")
    parsed = _typesafe_parse_body(e.body)
    asked = parsed isa AbstractDict ? get(parsed, "questions", nothing) : nothing
    asked isa AbstractDict && !isempty(asked) &&
        print(io, " asking ", join(sort!([repr(string(k)) for k in keys(asked)]), ", "))
    record = "run the same code with TYPESAFE_API_KEY set and `mode = :record_missing`."
    e.reason == _NO_RECORDING ?
        print(io, ": ", e.reason, " in ", e.dir, ". To record it, ", record) :
        print(io, ": ", e.reason, ". Delete ", joinpath(e.dir, e.key * ".json"),
              " and record it again: ", record)
end

const _ANSWER_MODES = (:replay, :record, :record_missing)

# One active scope. `parent` is the enclosing scope, which serves as this scope's
# service; the chain ends at `nothing`, the network.
struct _AnswerScope
    dir::String
    mode::Symbol
    parent::Union{Nothing,_AnswerScope}
end

const _ANSWER_SCOPE = ScopedValue{Union{Nothing,_AnswerScope}}(nothing)

"""
    with_recorded_answers(f, dir::AbstractString; mode::Symbol = :replay)

Run `f()` with every System One exchange inside it — [`ask`](@ref), and so
[`nl_dispatch`](@ref) and [`@branch`](@ref), and [`list_models`](@ref) — passing
through the recordings in `dir`, and return `f()`'s value.

The service does not answer a repeated request identically, so a test or a docs
build that calls it live cannot be reproduced; one recorded answer, replayed, can.

- `:replay` (the default): nothing reaches the network and no API key is needed.
  A request with no recording, or with a recording file that cannot be read as
  one, throws [`ReplayMissError`](@ref) out of `ask` itself. `dir` must exist.
- `:record`: every request goes to the service; each HTTP 200 is written to
  `dir`, replacing an earlier recording of the same request, readable or not. A
  non-200 result is returned as usual and never recorded.
- `:record_missing`: replay what `dir` holds, and send the rest to the service,
  recording their 200s. An unreadable recording throws `ReplayMissError` here
  too instead of being overwritten: delete the file to record it again.

A recording is written after the service has answered, and billed, so both
recording modes create `dir` if needed and prove it writable — a file is created
in it and removed — before `f` runs: a `dir` that is a file, or a directory that
refuses a file, is an `ArgumentError` naming the path and the cause, and nothing
is sent. A recording that still cannot be written once an answer has arrived —
the directory was removed or replaced, the disk is full — throws that I/O error
out of `ask` and `list_models` (and so out of `nl_dispatch` and `@branch`)
instead of returning a `SystemOneCallError`: the answer was paid for, and a
caller's `r isa SystemOneSuccess || fallback()` path must not absorb the lost
write as a service failure. A `tool_loop` dispatcher is the exception: the loop
reports a dispatcher's I/O error to the model as a tool error (only
`ReplayMissError` and interrupts escape it), so a write lost there surfaces as the
missing recording on the next replay.

A recording is `<dir>/<key>.json`, where `key` is the lowercase hex SHA-256 of
`"<METHOD> <path>\\n"` followed by the exact request body. The key is never a
canonical form of the request: the order of Choice options and of the state's
keys moves the answer, so a reordered request is a different request. The file
holds the request, the response body with its `x-typesafe-request-id`, and the
time of recording, pretty-printed so a diff is readable; the API key and the
headers are never written. Files are written atomically (a temporary file in
`dir`, then a rename), so concurrent tasks can record at once. A replayed answer
goes through the same decoding as a live 200 and has the same type and fields.
A cancelled token ends a call before any recording is read, exactly as it ends
one before anything is sent.

The scope is a `ScopedValue`, so it covers the tasks started inside `f`
(`Threads.@spawn`, `asyncmap`). Scopes nest: an inner scope's service is the
enclosing scope, so a `:record` scope inside a `:replay` scope copies the
replayed answers without touching the network. A `mode` other than the three
above is an `ArgumentError`, as is `:replay` with a `dir` that does not exist.

```julia
const ANSWERS = joinpath(@__DIR__, "recorded_answers")

with_recorded_answers(ANSWERS) do          # replay: no key, no network
    r = ask("My payouts have been failing for 3 days.",
            "urgency" => score("How urgent is this ticket?",
                               ["Can wait", "Needs attention this week", "Needs attention today"]))
    r["urgency"].score                     # the recorded answer, every run
end
```

A request that has no recording yet is recorded by running the same code once
with `TYPESAFE_API_KEY` set and `mode = :record_missing`.
"""
function with_recorded_answers(f::Function, dir::AbstractString; mode::Symbol=:replay)
    mode in _ANSWER_MODES || throw(ArgumentError(
        "mode must be :replay, :record or :record_missing; got $(repr(mode))"))
    root = abspath(dir)
    mode === :replay && !isdir(root) && throw(ArgumentError(
        "no recordings directory at $(repr(root)); record into it first with mode = :record_missing"))
    mode === :replay || _writable_dir(root)
    with(f, _ANSWER_SCOPE => _AnswerScope(root, mode, _ANSWER_SCOPE[]))
end

# Created, then a file made and removed in it the way `_record` makes one: a directory
# that cannot take a recording fails here, before `f` sends a request to pay for.
function _writable_dir(root::String)::Nothing
    ispath(root) && !isdir(root) && throw(ArgumentError(
        "cannot record answers into $(repr(root)): it exists and is not a directory"))
    try
        mkpath(root)
        tmp, io = mktemp(root; cleanup=false)
        close(io)
        rm(tmp)
    catch err
        err isa InterruptException && rethrow()
        throw(ArgumentError("cannot record answers into $(repr(root)): " *
                            first(split(sprint(showerror, err), '\n'))))
    end
    nothing
end

# A recording that could not be written once the service had answered. The answer
# was paid for, so `ask` and `list_models` throw the `cause` itself rather than
# returning it as a call error, which a fallback would absorb as a service failure.
struct _UnwrittenRecording <: Exception
    cause::Exception
end

_answer_key(method::String, path::String, body::String)::String =
    bytes2hex(sha256(string(method, ' ', path, '\n', body)))

# The HTTP exchange behind `ask` and `list_models`: `live()` is the real call, and
# every active scope stands between it and the caller, innermost first. The
# cancel check mirrors the seam's first one, so a cancelled call reads no
# recording, just as it sends nothing.
function _recorded_exchange(live::Function, method::String, path::String, body::String,
                            t0::UInt64, tok::Union{Nothing,CancelToken})::HTTP.Response
    iscancelled(tok) && throw(UniLMCancelled(:token, _elapsed_s(t0)))
    _exchange(_ANSWER_SCOPE[], method, path, body, live)
end

_exchange(::Nothing, ::String, ::String, ::String, live::Function)::HTTP.Response = live()

# A file that is not a readable recording is as loud as a missing one: returned as
# a call error, it would read as a service failure. `:record` never reads it, so
# recording over it is how it gets replaced.
function _exchange(s::_AnswerScope, method::String, path::String, body::String,
                   live::Function)::HTTP.Response
    key = _answer_key(method, path, body)
    file = joinpath(s.dir, key * ".json")
    miss(reason::String) = ReplayMissError(s.dir, key, method, path, body, reason)
    if s.mode !== :record && isfile(file)
        return try
            _replayed(file)
        catch err
            err isa InterruptException && rethrow()
            # The first line only: a JSON parse error goes on to quote the text around it.
            why = String(first(split(err isa ArgumentError ? err.msg : sprint(showerror, err), '\n')))
            throw(miss("unreadable recording: " * why))
        end
    end
    s.mode === :replay && throw(miss(_NO_RECORDING))
    resp = _exchange(s.parent, method, path, body, live)
    resp.status == 200 && _record(file, method, path, body, resp)
    resp
end

# A recording rebuilt as the 200 it captured, for the caller to decode as a live
# one. Only the recording's own shape is checked here: whether the body is a
# usable answer is the decoder's call, exactly as for a live 200.
function _replayed(file::String)::HTTP.Response
    rec = JSON.parse(read(file, String))
    res = rec isa AbstractDict ? get(rec, "response", nothing) : nothing
    res isa AbstractDict || throw(ArgumentError("it has no \"response\" object"))
    status = get(res, "status", nothing)
    status == 200 || throw(ArgumentError("its status is $(repr(status)), and only HTTP 200s are recorded"))
    answer = get(res, "body", nothing)
    answer isa AbstractDict || throw(ArgumentError("its \"body\" is not a JSON object"))
    id = get(res, "request_id", nothing)
    id isa Union{Nothing,AbstractString} || throw(ArgumentError(
        "its \"request_id\" is neither a string nor null"))
    HTTP.Response(200, isnothing(id) ? Pair{String,String}[] : ["x-typesafe-request-id" => String(id)],
                  Vector{UInt8}(JSON.json(answer)))
end

function _record(file::String, method::String, path::String, body::String, resp::HTTP.Response)::Nothing
    rec = JSON.Object{String,Any}(
        "request" => JSON.Object{String,Any}(
            "method" => method, "path" => path, "body" => isempty(body) ? nothing : JSON.parse(body)),
        # Parsed from the bytes, which stay unconsumed for the caller's own decoding.
        "response" => JSON.Object{String,Any}(
            "status" => 200, "request_id" => _typesafe_request_id(resp), "body" => JSON.parse(resp.body)),
        "recorded_at" => _rfc3339_utc(time()))
    try
        _write_recording(file, rec)
    catch err
        err isa InterruptException && rethrow()
        throw(_UnwrittenRecording(err))
    end
    nothing
end

function _write_recording(file::String, rec::JSON.Object{String,Any})::Nothing
    dir = dirname(file)
    mkpath(dir)
    tmp, io = mktemp(dir; cleanup=false)
    try
        try
            JSON.json(io, rec; pretty=true)
            println(io)
        finally
            close(io)
        end
        chmod(tmp, 0o644)           # `mktemp` makes it owner-only; a recording is meant to be read and committed
        mv(tmp, file; force=true)   # a rename: a reader sees the old file or the new, never a torn one
    catch
        rm(tmp; force=true)
        rethrow()
    end
    nothing
end

# RFC 3339 UTC timestamp of the Unix time `t`, to the second: civil-from-days, the
# inverse of `_utc_epoch_seconds`.
function _rfc3339_utc(t::Real)::String
    days, secs = fldmod(floor(Int, t), 86400)
    z = days + 719468
    era = fld(z, 146097)
    doe = z - era * 146097
    yoe = div(doe - div(doe, 1460) + div(doe, 36524) - div(doe, 146096), 365)
    doy = doe - (365 * yoe + div(yoe, 4) - div(yoe, 100))
    mp = div(5 * doy + 2, 153)
    m = mp < 10 ? mp + 3 : mp - 9
    d = doy - div(153 * mp + 2, 5) + 1
    two(n) = lpad(n, 2, '0')
    string(lpad(yoe + era * 400 + (m <= 2), 4, '0'), '-', two(m), '-', two(d), 'T',
           two(div(secs, 3600)), ':', two(div(rem(secs, 3600), 60)), ':', two(rem(secs, 60)), 'Z')
end
