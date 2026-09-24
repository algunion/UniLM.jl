# ============================================================================
# MCP Client — Model Context Protocol client for UniLM.jl
# Connects to MCP servers (stdio or HTTP), discovers tools, and bridges them
# into CallableTool for seamless tool_loop! / tool_loop integration.
#
# Protocol: JSON-RPC 2.0 over stdio or Streamable HTTP (spec 2025-11-25)
# ============================================================================

# ─── JSON-RPC 2.0 Framing (internal) ────────────────────────────────────────

const _JSONRPC_VERSION = "2.0"

# Protocol revisions this package negotiates, as client and as server, preferred
# (latest) first. 2024-11-05 is excluded: it predates Streamable HTTP and used
# the separate HTTP+SSE dual-endpoint transport this package does not implement.
const _MCP_SUPPORTED_PROTOCOL_VERSIONS = ("2025-11-25", "2025-06-18", "2025-03-26")

# The version the client prefers and advertises: its latest supported revision.
const _MCP_PROTOCOL_VERSION = first(_MCP_SUPPORTED_PROTOCOL_VERSIONS)

struct _JSONRPCRequest
    id::Union{Int,String}
    method::String
    params::Union{Dict{String,Any},Nothing}
end

function _jsonrpc_serialize(r::_JSONRPCRequest)::String
    d = Dict{String,Any}("jsonrpc" => _JSONRPC_VERSION, "id" => r.id, "method" => r.method)
    !isnothing(r.params) && (d["params"] = r.params)
    JSON.json(d)
end

struct _JSONRPCNotification
    method::String
    params::Union{Dict{String,Any},Nothing}
end

function _jsonrpc_serialize(n::_JSONRPCNotification)::String
    d = Dict{String,Any}("jsonrpc" => _JSONRPC_VERSION, "method" => n.method)
    !isnothing(n.params) && (d["params"] = n.params)
    JSON.json(d)
end

struct _JSONRPCResponse
    id::Union{Int,String,Nothing}
    result::Union{Dict{String,Any},Nothing}
    error::Union{Dict{String,Any},Nothing}
end

function _JSONRPCResponse(d::Dict{String,Any})
    _JSONRPCResponse(get(d, "id", nothing), get(d, "result", nothing), get(d, "error", nothing))
end

# ─── MCP Error ───────────────────────────────────────────────────────────────

"""
    MCPError <: Exception

Error from MCP protocol operations. Contains the JSON-RPC error code, message,
and the optional `data` member from the server — any JSON value, `nothing` when
absent.
"""
struct MCPError <: Exception
    code::Int
    message::String
    data::Any
end

MCPError(d::Dict{String,Any}) = MCPError(d["code"], d["message"], get(d, "data", nothing))

function Base.showerror(io::IO, e::MCPError)
    print(io, "MCPError($(e.code)): $(e.message)")
    !isnothing(e.data) && print(io, " data=", e.data)
end

"""
    MCPCrashError <: Exception

The stdio MCP server process exited or its stdio pipe broke. The session is
closed with cause `:crash`; with `auto_respawn=true` the next call respawns the
server, otherwise it errors naming the opt-in.

# Fields
- `msg::String`: human-readable message with recovery guidance.
- `exitcode::Union{Int,Nothing}`: exit code when the server exited on its own
  and had been reaped in time; else `nothing`.
- `termsignal::Union{Int,Nothing}`: signal number when the server was
  signal-killed; else `nothing`.
- `cause::Union{Exception,Nothing}`: underlying transport exception
  (`nothing` for a clean read-EOF).

Best-effort diagnostics: both `exitcode` and `termsignal` are `nothing` when the
process had not been reaped within a brief settle window after the failure
surfaced (or was still alive with a broken pipe).
"""
struct MCPCrashError <: Exception
    msg::String
    exitcode::Union{Int,Nothing}
    termsignal::Union{Int,Nothing}
    cause::Union{Exception,Nothing}
end

Base.showerror(io::IO, e::MCPCrashError) = print(io, "MCPCrashError: ", e.msg)

function _crash_exit_info(exitcode::Union{Int,Nothing}, termsignal::Union{Int,Nothing})::String
    termsignal !== nothing && return " (killed by signal $termsignal)"
    exitcode !== nothing && return " (exit code $exitcode)"
    ""
end

_crash_msg(context::String, exitcode::Union{Int,Nothing}, termsignal::Union{Int,Nothing})::String =
    "MCP server crashed during $context: the server process exited or its stdio " *
    "pipe broke$(_crash_exit_info(exitcode, termsignal)). The session is closed. " *
    "Reconnect explicitly, or pass auto_respawn=true to mcp_connect so the next " *
    "call transparently respawns the server (its in-memory state is lost and " *
    "tools are refetched)."

"""
    MCPSessionClosedError <: Exception

The session is closed, or its transport is not connected, so the call never reached
the server.

# Fields
- `cause::Symbol`: `:disconnected` (closed by [`mcp_disconnect!`](@ref) or a failed
  connect, or a transport that is not connected), `:timeout` (a stdio request or
  connect timeout closed it), or `:crash` (the stdio server process died).
- `msg::String`: human-readable message with recovery guidance.

A stdio session closed by `:timeout` or `:crash` respawns its server instead of
raising this when it was connected with `auto_respawn=true`.
"""
struct MCPSessionClosedError <: Exception
    cause::Symbol
    msg::String
    function MCPSessionClosedError(cause::Symbol, msg::String)
        cause in (:disconnected, :timeout, :crash) ||
            throw(ArgumentError("MCPSessionClosedError cause must be :disconnected, :timeout " *
                                "or :crash (got :$cause)"))
        new(cause, msg)
    end
end

Base.showerror(io::IO, e::MCPSessionClosedError) =
    print(io, "MCPSessionClosedError(:", e.cause, "): ", e.msg)

_not_connected(transport::String) = MCPSessionClosedError(:disconnected,
    "MCP $transport transport is not connected. Reconnect with mcp_connect.")

"""
    _MCPSessionExpired <: Exception

Internal signal that an HTTP transport received `404 Not Found` while holding a
session id — the server dropped the session. Caught by
[`_mcp_request_recover!`](@ref), which re-initializes and replays the request
once; never surfaced to callers.
"""
struct _MCPSessionExpired <: Exception
    body::String
end

# ─── MCP Types ───────────────────────────────────────────────────────────────

"""
    MCPServerCapabilities

Capabilities declared by an MCP server during initialization.
"""
struct MCPServerCapabilities
    tools::Union{Dict{String,Any},Nothing}
    resources::Union{Dict{String,Any},Nothing}
    prompts::Union{Dict{String,Any},Nothing}
    logging::Union{Dict{String,Any},Nothing}
end

function MCPServerCapabilities(d::Dict{String,Any})
    MCPServerCapabilities(
        get(d, "tools", nothing),
        get(d, "resources", nothing),
        get(d, "prompts", nothing),
        get(d, "logging", nothing)
    )
end

MCPServerCapabilities() = MCPServerCapabilities(nothing, nothing, nothing, nothing)

"""
    MCPToolInfo

A tool definition received from an MCP server via `tools/list`.
Convert to `CallableTool` via [`mcp_tools`](@ref) or [`mcp_tools_respond`](@ref).
"""
struct MCPToolInfo
    name::String
    description::Union{String,Nothing}
    input_schema::Union{Dict{String,Any},Nothing}
    output_schema::Union{Dict{String,Any},Nothing}
end

function MCPToolInfo(d::Dict{String,Any})
    MCPToolInfo(
        d["name"],
        get(d, "description", nothing),
        get(d, "inputSchema", nothing),
        get(d, "outputSchema", nothing)
    )
end

"""
    MCPResourceInfo

A resource definition received from an MCP server via `resources/list`.
"""
struct MCPResourceInfo
    uri::String
    name::String
    description::Union{String,Nothing}
    mime_type::Union{String,Nothing}
end

function MCPResourceInfo(d::Dict{String,Any})
    MCPResourceInfo(d["uri"], d["name"], get(d, "description", nothing), get(d, "mimeType", nothing))
end

"""
    MCPPromptInfo

A prompt definition received from an MCP server via `prompts/list`.
"""
struct MCPPromptInfo
    name::String
    description::Union{String,Nothing}
    arguments::Union{Vector{Dict{String,Any}},Nothing}
end

function MCPPromptInfo(d::Dict{String,Any})
    MCPPromptInfo(d["name"], get(d, "description", nothing), get(d, "arguments", nothing))
end

# ─── Transport Abstraction ───────────────────────────────────────────────────

"""
    MCPTransport

Abstract type for MCP transport implementations. Subtypes must implement:
- `_transport_connect!(t)` — establish connection
- `_transport_send!(t, msg::String; cfg)::String` — send JSON-RPC message, return response
- `_transport_read!(t)::String` — read the next incoming JSON-RPC frame
- `_transport_notify!(t, msg::String; cfg)` — send notification (no response expected)
- `_transport_disconnect!(t; cfg)` — close connection
- `_transport_isconnected(t)::Bool` — check if connected

A custom transport must bound its own IO: the connect and per-call bounds are enforced
by the stdio transport's watchdog and inside the HTTP transport's requests, and a
custom transport's connect handshake and exchanges run without either — an unbounded
read in one blocks its caller, and every caller queued behind it, indefinitely.
"""
abstract type MCPTransport end

"""Internal sentinel: a stdio pipe operation failed because the server process
exited or the pipe broke. Thrown ONLY by the three stdio transport primitives;
mapped to the exported `MCPCrashError` at the session boundaries. `cause` is the
underlying exception, or `nothing` for the read-EOF case. Never leaks to users:
every producer sits beneath one of the two mapping boundaries."""
struct _TransportClosed <: Exception
    cause::Union{Exception,Nothing}
end

"""
    StdioTransport(command::Cmd; stderr=nothing) <: MCPTransport

Stdio transport: launches a subprocess and communicates via stdin/stdout.
Messages are newline-delimited JSON-RPC 2.0. `stderr` is where the server's own
stderr goes — an `IO` (e.g. `devnull`) or a file path, appended to; `nothing` inherits
this process's stderr.
"""
mutable struct StdioTransport <: MCPTransport
    command::Cmd
    const stderr::Union{Nothing,IO,AbstractString}
    process::Union{Base.Process,Nothing}
    input::Union{IO,Nothing}
    output::Union{IO,Nothing}
    # Process-group id captured at spawn. Under detach=true the child is its own
    # group leader, so its pgid == its pid. Retained so teardown can group-SIGKILL
    # even after the direct child is reaped (when getpid(process) would fail). A
    # group id stays reserved only while the group has members, so it is taken — by
    # the teardown ladder's final rung or by the leader watcher, whichever runs first
    # (see _track_live!) — under the live-transport lock.
    pgid::Union{Int32,Nothing}
    StdioTransport(command::Cmd; stderr::Union{Nothing,IO,AbstractString}=nothing) =
        new(command, stderr, nothing, nothing, nothing, nothing)
end

# Live stdio transports, torn down at process exit: servers run detached (their own
# process group), so nothing else reaps them when this process exits, and a server busy
# in a handler never sees stdin EOF. The same lock makes taking a transport's pgid
# exactly-once between the teardown ladder and the leader watcher. An atexit hook (not
# a finalizer: teardown does IO) is registered on first use.
const _LIVE_STDIO = Base.Lockable(Base.IdSet{StdioTransport}())
const _REAPER_ARMED = Threads.Atomic{Bool}(false)

"""Register a freshly spawned transport: live registry, the atexit reaper, and a watcher
that reaps the group the moment its leader exits. A dead leader can leave members behind,
and once the group is empty its id can be reused — group-killing a stale id later could
hit an unrelated process group."""
function _track_live!(t::StdioTransport, proc::Base.Process)
    @lock _LIVE_STDIO push!(_LIVE_STDIO[], t)
    Threads.atomic_cas!(_REAPER_ARMED, false, true) || atexit(_reap_live_stdio!)
    errormonitor(Threads.@spawn (wait(proc); _group_kill(_take_group!(t))))
    nothing
end

"""Take `t`'s process-group id and drop `t` from the live registry — exactly once across
the teardown ladder's final rung and the leader watcher, so only one of them signals it."""
function _take_group!(t::StdioTransport)::Union{Int32,Nothing}
    @lock _LIVE_STDIO begin
        delete!(_LIVE_STDIO[], t)
        pgid, t.pgid = t.pgid, nothing
        pgid
    end
end

# SIGKILL a whole process group. The `pgid > 0` guard is defense-in-depth: kill(-0, …)
# would signal the CALLER's own group. ESRCH (the group is already empty) is expected.
_group_kill(::Nothing) = nothing
function _group_kill(pgid::Int32)
    pgid > 0 || return nothing
    if @ccall(kill((-pgid)::Cint, 9::Cint)::Cint) != 0
        err = Libc.errno()
        err == Libc.ESRCH || @debug "MCP group SIGKILL failed" errno = err
    end
    nothing
end

"""atexit hook: run every live stdio transport's teardown ladder, concurrently, so exit
waits for one ladder rather than their sum."""
function _reap_live_stdio!()
    live = @lock _LIVE_STDIO collect(_LIVE_STDIO[])
    @sync for t in live
        Threads.@spawn _kill_transport!(t)
    end
end

function _transport_connect!(t::StdioTransport)
    # Spawn in the child's own process group (detach=true) so the teardown ladder's
    # final rung can group-SIGKILL grandchildren a wrapper forks (e.g. the node
    # process `npx` launches) that would otherwise survive holding our stdio pipe.
    # Teardown ladder: _kill_transport!.
    # A stderr file is appended to, so a respawn keeps the log of the server it replaces.
    cmd = Cmd(t.command; detach=true)
    isnothing(t.stderr) || (cmd = pipeline(cmd; stderr=t.stderr, append=t.stderr isa AbstractString))
    proc = open(cmd, read=true, write=true)
    t.process = proc
    t.pgid = getpid(proc)   # == the child's pgid under detach; capture while alive
    t.input = proc.in
    t.output = proc.out
    _track_live!(t, proc)
end

# A request is one frame out, then frames in until its response; the whole exchange is
# bounded by the watchdog in _mcp_request!.
function _transport_send!(t::StdioTransport, msg::String;
                          cfg::RequestConfig=current_config())::String
    _transport_notify!(t, msg; cfg)
    _transport_read!(t)
end

"""Read the next frame. Only a read at EOF comes back empty — a blank line keeps its
newline — so EOF is told apart from blank and whitespace-only lines, which carry no
frame and are skipped."""
function _transport_read!(t::StdioTransport)::String
    out = t.output
    isnothing(out) && throw(_not_connected("stdio"))
    while true
        line = try
            readline(out; keep=true)
        catch e
            e isa InterruptException && rethrow()
            throw(_TransportClosed(e))
        end
        isempty(line) && throw(_TransportClosed(nothing))
        isempty(strip(line)) || return String(chomp(line))
    end
end

function _transport_notify!(t::StdioTransport, msg::String;
                            cfg::RequestConfig=current_config())
    inp = t.input
    isnothing(inp) && throw(_not_connected("stdio"))
    try
        write(inp, msg, "\n")
        flush(inp)
    catch e
        e isa InterruptException && rethrow()
        throw(_TransportClosed(e))
    end
    nothing
end

"""
Tear a stdio transport down with a graceful escalation ladder, then null its
handles. MCP spec: a compliant server exits when its stdin reaches EOF, so we close
stdin first; if the process lingers we escalate SIGTERM. The FINAL rung is
UNCONDITIONAL — a group-directed SIGKILL by the pgid captured at spawn (the child
leads its own group, spawned detach=true) — so a grandchild the leader orphaned by
exiting on stdin EOF cannot survive holding our pipe. The pgid is taken exactly once
(see [`_take_group!`](@ref)): when the leader watcher already reaped the group, the
rung has nothing left to signal; ESRCH (empty group) is the expected no-op.
`grace_term`/`grace_kill` are exposed for suite-time control; production defaults are
fixed. Best-effort and idempotent: safe from a disconnect, from the request/connect
watchdog and from the atexit reaper.
"""
function _kill_transport!(t::StdioTransport;
                          grace_term::Float64=5.0, grace_kill::Float64=2.0)::Nothing
    proc = t.process
    if !isnothing(proc)
        _close_quietly(t.input)                            # stdin EOF: compliant servers exit
        if process_running(proc)
            if timedwait(() -> !process_running(proc), grace_term) !== :ok
                try; kill(proc); catch; end                  # SIGTERM
                timedwait(() -> !process_running(proc), grace_kill)
            end
        end
    end
    _group_kill(_take_group!(t))                           # final rung, unconditional
    t.process = nothing
    t.input = nothing
    t.output = nothing
    nothing
end

# Teardown closes streams that may already be closed or broken; it must still run to its
# last rung.
function _close_quietly(io::Union{IO,Nothing})
    isnothing(io) && return nothing
    try
        close(io)
    catch e
        e isa InterruptException && rethrow()
        @debug "MCP stdio close failed during teardown" exception = e
    end
    nothing
end

"""Request-watchdog teardown. Closing our end of the server's stdout FIRST releases the
read blocked in the exchange at once; the kill ladder alone releases it only when the
server exits, up to the ladder's whole grace later."""
function _abort_exchange!(t::StdioTransport)
    out = t.output
    t.output = nothing
    _close_quietly(out)
    _kill_transport!(t)
end

# Graceful disconnect uses the same ladder. `cfg` is accepted for signature parity
# with the HTTP transport (which needs it for the DELETE) and ignored here.
_transport_disconnect!(t::StdioTransport; cfg::Union{Nothing,RequestConfig}=nothing) =
    _kill_transport!(t)

# Connected while the read end is open and the server runs: the request watchdog drops
# the read end before its kill ladder finishes.
function _transport_isconnected(t::StdioTransport)::Bool
    proc = t.process
    !isnothing(t.output) && !isnothing(proc) && process_running(proc)
end

"""
    HTTPTransport <: MCPTransport

Streamable HTTP transport: communicates via POST requests to an MCP endpoint.
Handles `Mcp-Session-Id` header for session management.
"""
mutable struct HTTPTransport <: MCPTransport
    url::String
    headers::Vector{Pair{String,String}}
    session_id::Union{String,Nothing}
    # Value sent in the `Mcp-Protocol-Version` header. Starts at the client's
    # preferred revision (used for the initialize request) and is updated to the
    # server-negotiated revision once the handshake succeeds.
    protocol_version::String
    connected::Bool
    pending::Vector{String}  # frames from the last response body, not yet consumed
    function HTTPTransport(url::String; headers::Vector{Pair{String,String}}=Pair{String,String}[])
        new(url, headers, nothing, _MCP_PROTOCOL_VERSION, false, String[])
    end
end

_transport_connect!(t::HTTPTransport) = (t.connected = true; nothing)

# Caller-supplied headers are where MCP credentials live (the documented usage is
# `"Authorization" => "Bearer <token>"`), and a default field dump prints them in
# full at the REPL, under @show, and into any log that renders the transport. Show
# the header NAMES and redact every auth-shaped value, matching the policy the
# service-endpoint shows already apply to a stored API key.
function _redacted_headers(headers)::Vector{Pair{String,String}}
    [lowercase(k) in ("authorization", "proxy-authorization", "x-api-key",
                      "x-goog-api-key", "api-key", "cookie") ?
     (k => _redact_api_key(v)) : (k => v) for (k, v) in headers]
end

function Base.show(io::IO, t::HTTPTransport)
    print(io, "HTTPTransport(")
    show(io, t.url)
    print(io, ", headers=")
    show(io, _redacted_headers(t.headers))
    print(io, ", connected=", t.connected,
          ", protocol_version=", repr(t.protocol_version),
          ", session_id=", isnothing(t.session_id) ? "nothing" : "<set>", ")")
end

"""Headers carried by EVERY MCP Streamable-HTTP POST: the JSON body type, the dual
Accept the transport requires (the server may answer with a JSON body or an SSE
stream — for a notification too), the protocol revision currently in force, and the
session id once the server assigned one. One header set for requests and
notifications alike: a notification that advertises less than a request is one a
spec-strict server may refuse."""
function _mcp_post_headers(t::HTTPTransport)::Vector{Pair{String,String}}
    hdrs = copy(t.headers)
    push!(hdrs, "Content-Type" => "application/json")
    push!(hdrs, "Accept" => "application/json, text/event-stream")
    push!(hdrs, "Mcp-Protocol-Version" => t.protocol_version)
    !isnothing(t.session_id) && push!(hdrs, "Mcp-Session-Id" => t.session_id)
    hdrs
end

"""Fail loud on an HTTP status the MCP transport cannot use. `401`/`403` name the
mechanism that supplies credentials (this client implements no authentication flow —
credentials travel as request headers); any other status is reported with its body.
Never returns."""
function _mcp_http_status_error(resp::HTTP.Response)
    if resp.status == 401 || resp.status == 403
        error("MCP HTTP request rejected with status $(resp.status). The server " *
              "requires authentication; pass credentials via the `headers` kwarg " *
              "of mcp_connect (e.g. headers=[\"Authorization\" => \"Bearer <token>\"]).")
    end
    error("MCP HTTP request failed with status $(resp.status): $(String(resp.body))")
end

function _transport_send!(t::HTTPTransport, msg::String;
                          cfg::RequestConfig=current_config())::String
    t.connected || throw(_not_connected("HTTP"))
    resp = _http("POST", t.url, _mcp_post_headers(t), msg; cfg=cfg, remaining=Inf)
    # Capture session ID from response
    sid = HTTP.header(resp, "Mcp-Session-Id", "")
    !isempty(sid) && (t.session_id = sid)
    # A 404 while holding a session id means the server expired the session;
    # signal the request layer to re-initialize (Streamable HTTP session
    # lifecycle) rather than failing the call.
    if resp.status == 404 && !isnothing(t.session_id)
        throw(_MCPSessionExpired(String(resp.body)))
    end
    resp.status == 200 || _mcp_http_status_error(resp)
    ct = HTTP.header(resp, "Content-Type", "")
    empty!(t.pending)  # frames left over from a previous exchange are stale
    if startswith(ct, "text/event-stream")
        # An SSE body may carry several frames (notifications/requests around
        # the response). Queue them in arrival order; hand back the first.
        frames = _parse_sse_frames(String(resp.body))
        isempty(frames) && error("No data found in SSE response")
        append!(t.pending, frames[2:end])
        frames[1]
    else
        String(resp.body)
    end
end

function _transport_read!(t::HTTPTransport)::String
    isempty(t.pending) &&
        error("MCP HTTP response body ended before a response to the pending request")
    popfirst!(t.pending)
end

# Frames queued after the one that answered the request: the rest of an SSE body, still
# server messages (a list_changed or a server request may follow the response).
_transport_trailing!(::MCPTransport) = String[]
_transport_trailing!(t::HTTPTransport) = splice!(t.pending, eachindex(t.pending))

function _transport_notify!(t::HTTPTransport, msg::String;
                            cfg::RequestConfig=current_config())
    t.connected || throw(_not_connected("HTTP"))
    resp = _http("POST", t.url, _mcp_post_headers(t), msg; cfg=cfg, remaining=Inf)
    # A notification carries no response frame to demux, but its STATUS is the server's
    # only channel for refusing it (202 Accepted is the usual acceptance). Dropping the
    # status turns a rejected `notifications/initialized` into a session the client
    # believes is initialized and the server does not. A 404 stays in this loud bucket
    # rather than signalling session expiry: the expiry path re-initializes and replays
    # a REQUEST, and re-entering it from a notification would recurse through the
    # handshake's own notification.
    200 <= resp.status < 300 || _mcp_http_status_error(resp)
    nothing
end

function _transport_disconnect!(t::HTTPTransport; cfg::RequestConfig=current_config())
    if t.connected && !isnothing(t.session_id)
        # Like every request after initialize, the DELETE carries the negotiated revision.
        hdrs = copy(t.headers)
        push!(hdrs, "Mcp-Session-Id" => t.session_id, "Mcp-Protocol-Version" => t.protocol_version)
        try
            _http("DELETE", t.url, hdrs; cfg=cfg, remaining=Inf)
        catch e
            @debug "MCP HTTP disconnect failed" exception=e
        end
    end
    t.connected = false
    t.session_id = nothing
    nothing
end

_transport_isconnected(t::HTTPTransport) = t.connected

# The negotiated protocol-version header and session reset only apply to
# transports that carry HTTP session state; stdio transports ignore them.
_set_protocol_version!(::MCPTransport, ::AbstractString) = nothing
_set_protocol_version!(t::HTTPTransport, v::AbstractString) = (t.protocol_version = v; nothing)

_reset_session!(::MCPTransport) = nothing
_reset_session!(t::HTTPTransport) = (t.session_id = nothing; nothing)

"""Split an SSE response body into its JSON-RPC frames, in arrival order — ONE frame
per event. The body is cut into events at blank lines (the SSE event delimiter, which
the streaming machine's layers do not see: layer 1 drops blank lines because no
supported provider needs the boundary), then each event's fields are framed by the
shared [`_sse_events!`](@ref). That buys the spec's two rules the transport needs: the
space after `data:` is OPTIONAL, and an event carrying several `data:` lines is ONE
payload, its lines joined with `\\n`. An event whose data is empty is not dispatched
(SSE): that is the priming event — an id with empty data — a Streamable HTTP server
opens its streams with."""
function _parse_sse_frames(body::String)::Vector{String}
    frames = String[]
    carry, event = IOBuffer(), Ref("")
    for block in eachsplit(body, r"\r?\n\r?\n")
        # Layer 1 emits only lines it has seen terminated; the appended newline
        # completes an event whose last line ends at the delimiter (or at EOF).
        payloads = _sse_events!(carry, event, block * "\n")
        frame = join((payload for (_, payload) in payloads), "\n")
        isempty(frame) || push!(frames, frame)
    end
    frames
end

# ─── Session lock ────────────────────────────────────────────────────────────

"""
FIFO hand-off lock serializing a session's calls. Re-entrant for its owner (a
notification sent inside an exchange re-enters it). Release hands ownership straight to
the longest waiter BEFORE waking it, so a caller looping on the session cannot barge
ahead of a queued one. A waiter that gives up — its bound passed, or it was interrupted —
leaves the queue, and passes the lock on if it had already been handed to it, so no
caller behind it is stranded on a free lock.
"""
mutable struct _SessionLock <: Base.AbstractLock
    const guard::ReentrantLock                 # held only for the O(1) state changes below
    owner::Union{Task,Nothing}
    depth::Int
    const queue::Vector{Pair{Task,Base.Event}}
    _SessionLock() = new(ReentrantLock(), nothing, 0, Pair{Task,Base.Event}[])
end

Base.islocked(l::_SessionLock) = (@lock l.guard l.owner) !== nothing
Base.lock(l::_SessionLock) = (_acquire!(l, Inf); nothing)

function Base.unlock(l::_SessionLock)
    @lock l.guard begin
        l.owner === current_task() ||
            error("unlock of an MCP session lock by a task that does not hold it")
        (l.depth -= 1) == 0 && _hand_off!(l)
    end
    nothing
end

# Caller holds `l.guard`. Ownership moves before the wakeup, so the lock is never free
# while a waiter is queued.
function _hand_off!(l::_SessionLock)
    if isempty(l.queue)
        l.owner, l.depth = nothing, 0
    else
        task, ev = popfirst!(l.queue)
        l.owner, l.depth = task, 1
        notify(ev)
    end
    nothing
end

"""Acquire `l` within `limit` seconds (`Inf`: unbounded); `false` when the bound passed
first — the caller then never held the lock."""
function _acquire!(l::_SessionLock, limit::Float64)::Bool
    me = current_task()
    ev = @lock l.guard begin
        if l.owner === me
            l.depth += 1
            return true
        elseif l.owner === nothing
            l.owner, l.depth = me, 1
            return true
        end
        e = Base.Event()
        push!(l.queue, me => e)
        e
    end
    timer = isfinite(limit) ? Timer(_ -> notify(ev), limit) : nothing
    try
        wait(ev)
    catch
        _abandon!(l, me)
        rethrow()
    finally
        isnothing(timer) || close(timer)
    end
    @lock l.guard begin
        l.owner === me && return true
        deleteat!(l.queue, findfirst(p -> first(p) === me, l.queue))   # timed out: leave
        false
    end
end

# A waiter interrupted while queued leaves the queue; one interrupted after the lock was
# handed to it passes the lock on instead.
function _abandon!(l::_SessionLock, me::Task)
    @lock l.guard begin
        if l.owner === me
            _hand_off!(l)
        else
            i = findfirst(p -> first(p) === me, l.queue)
            isnothing(i) || deleteat!(l.queue, i)
        end
    end
    nothing
end

# ─── MCPSession ──────────────────────────────────────────────────────────────

"""
    MCPSession

A live connection to an MCP server. Manages lifecycle, transport, and cached
tool/resource/prompt lists.

Create via [`mcp_connect`](@ref). Disconnect via [`mcp_disconnect!`](@ref).

Requests are serialized: each call (its liveness/respawn check, id allocation and
request/response exchange) holds the session, and callers are served in arrival
order. Interleaved server → client frames are handled in place (notifications
skipped, server `ping` requests answered). A caller's per-call bound (`timeout`,
default `mcp_request_timeout`) also covers its wait for the session: a caller that
cannot acquire it in time raises [`MCPTimeoutError`](@ref) with phase `:queue`
without touching it. Once acquired, the exchange gets its full bound, measured from
that moment — a waiter's clock never tears down the exchange in progress. After the
server sends `notifications/tools/list_changed`, `tools_stale` is `true` until the
next [`list_tools!`](@ref).
"""
mutable struct MCPSession
    transport::MCPTransport
    server_capabilities::MCPServerCapabilities
    server_info::Dict{String,Any}
    tools::Vector{MCPToolInfo}
    resources::Vector{MCPResourceInfo}
    prompts::Vector{MCPPromptInfo}
    protocol_version::String
    _id_counter::Int
    status::Symbol  # :disconnected, :initializing, :ready, :closed
    _lock::_SessionLock
    tools_stale::Bool
    # The exact `initialize` params (protocolVersion, capabilities, clientInfo)
    # retained so an expired HTTP session can be re-initialized transparently.
    _init_params::Dict{String,Any}
    # Timeout configuration resolved and captured at mcp_connect. Connect/initialize
    # bounds come from here; the per-request bound resolves at call time (see call_tool).
    config::RequestConfig
    # When true, the next call on a session closed by a stdio request timeout or a
    # server crash respawns the server (same command, fresh handshake) instead of
    # erroring. Default OFF: a silent respawn fabricates session continuity and
    # in-memory server state is lost.
    auto_respawn::Bool
    # Why the session reached :closed — :none (live, or a normal disconnect),
    # :timeout (request/connect watchdog), :crash (server process died or stdio
    # pipe broke). Gates the respawn-or-error decision on the next call.
    _close_cause::Symbol
end

# Sessions start with a fresh lock, a fresh (not stale) tool cache, default config,
# respawn OFF, and not-timed-out. New keyword args default so existing positional
# call sites (tests, handshake) construct unchanged.
function MCPSession(transport::MCPTransport, caps::MCPServerCapabilities,
                    server_info::Dict{String,Any}, tools::Vector{MCPToolInfo},
                    resources::Vector{MCPResourceInfo}, prompts::Vector{MCPPromptInfo},
                    protocol_version::String, id_counter::Int, status::Symbol;
                    init_params::Dict{String,Any}=Dict{String,Any}(),
                    config::RequestConfig=RequestConfig(),
                    auto_respawn::Bool=false)
    MCPSession(transport, caps, server_info, tools, resources, prompts,
               protocol_version, id_counter, status, _SessionLock(), false, init_params,
               config, auto_respawn, :none)
end

# A session reaches the credential through its transport, so a default field dump
# prints the caller's token as surely as the transport's own would. Render the
# transport through its (redacting) show and summarize the rest: what a user wants
# from printing a session is where it points, whether it is live, and what it found.
function Base.show(io::IO, s::MCPSession)
    print(io, "MCPSession(")
    show(io, s.transport)
    print(io, ", status=:", s.status,
          ", protocol_version=", repr(s.protocol_version),
          ", tools=", length(s.tools), s.tools_stale ? " (stale)" : "",
          ", resources=", length(s.resources),
          ", prompts=", length(s.prompts), ")")
end

"""Best-effort exit diagnostics, captured BEFORE the teardown ladder (which reaps
and nulls the process). Bounded settle: EOF/EPIPE can surface a beat before libuv
reaps the child, so wait briefly for `process_exited` — for the pipe-broken-but-
alive server this adds at most 2 s to a terminal error path."""
function _exit_diagnostics(t::StdioTransport)::Tuple{Union{Int,Nothing},Union{Int,Nothing}}
    proc = t.process
    isnothing(proc) && return (nothing, nothing)
    process_exited(proc) || timedwait(() -> process_exited(proc), 2.0)
    process_exited(proc) || return (nothing, nothing)
    proc.termsignal > 0 && return (nothing, Int(proc.termsignal))
    (Int(proc.exitcode), nothing)
end

"""Classify a stdio transport failure as a server crash: capture diagnostics
(pre-ladder), tear the transport down (also reaps a pipe-broken-but-alive server),
close the session with cause `:crash`, and throw the typed error."""
function _crash_close!(session::MCPSession, t::StdioTransport, tc::_TransportClosed,
                       context::String)
    exitcode, termsignal = _exit_diagnostics(t)
    _kill_transport!(t)
    session.status = :closed
    session._close_cause = :crash
    throw(MCPCrashError(_crash_msg(context, exitcode, termsignal), exitcode, termsignal, tc.cause))
end

"""Allocate the next request id. Callers must hold `session._lock`."""
function _next_id!(session::MCPSession)::Int
    session._id_counter += 1
    session._id_counter
end

"""Reject a per-call MCP timeout that would reintroduce an unbounded wait.
NaN must be rejected explicitly (NaN ≤ 0 is false); Inf disables the bound."""
function _validate_mcp_timeout(t::Float64)::Float64
    (isnan(t) || t <= 0) &&
        throw(ArgumentError("MCP timeout must be a positive number of seconds " *
                            "(got $t); Inf disables the bound."))
    t
end

"""Resolve the per-exchange MCP request bound: explicit kwarg > ambient scoped
config's `mcp_request_timeout` (when a scope is set) > the session-captured config.
Bridged tool closures pass no kwarg, so an ambient `with_request_config` reaches
them through the scope leg."""
function _resolve_mcp_request_timeout(session::MCPSession, timeout::Union{Nothing,Float64})::Float64
    timeout !== nothing && return _validate_mcp_timeout(timeout)
    amb = _REQUEST_CONFIG[]
    amb !== nothing ? amb.mcp_request_timeout : session.config.mcp_request_timeout
end

"""Select the bound for ONE exchange. Connect-phase exchanges — handshake discovery
re-entering [`_mcp_request!`](@ref) from [`_finalize_connect!`](@ref) while the session
is `:initializing` — belong to the connect phase and are governed by
`mcp_connect_timeout`, the bound the connect-timeout message names; every other
exchange takes the resolved per-call request bound. An explicit per-call `timeout` is
a deliberate override and wins in either phase (and is validated in both)."""
function _mcp_exchange_bound(session::MCPSession, timeout::Union{Nothing,Float64})::Float64
    requested = _resolve_mcp_request_timeout(session, timeout)
    timeout === nothing && session.status === :initializing &&
        return session.config.mcp_connect_timeout
    requested
end

const _MCP_TIMEOUT_OVERRIDES =
    "Raise it for one call with call_tool(session, name, args; timeout=<seconds>), for " *
    "a dynamic scope with with_request_config(; mcp_request_timeout=<seconds>), or per " *
    "session with mcp_connect(...; config=RequestConfig(current_config(); " *
    "mcp_request_timeout=<seconds>))."

_request_timeout_msg(limit::Float64)::String =
    "MCP request exceeded the $(limit)s request timeout. " * _MCP_TIMEOUT_OVERRIDES

_queue_timeout_msg(limit::Float64)::String =
    "MCP call waited $(limit)s (its request timeout) for the session without acquiring " *
    "it: calls on one session run one at a time, and the calls ahead held it. " *
    _MCP_TIMEOUT_OVERRIDES

# Bound on the best-effort cancellation notice sent for a timed-out request: it must not
# stretch the timeout it follows by more than this.
const _MCP_CANCEL_BOUND = 2.0

"""Tell the server to stop working on request `id`, which timed out on this side
(lifecycle: the sender SHOULD cancel a request it no longer waits for). Best effort:
bounded by [`_MCP_CANCEL_BOUND`](@ref), and a failure is logged at debug level — the
timeout it follows stands either way."""
function _send_cancelled!(session::MCPSession, id::Int, excfg::RequestConfig)
    notice = _JSONRPCNotification("notifications/cancelled",
        Dict{String,Any}("requestId" => id, "reason" => "the request timed out on the client"))
    try
        _transport_notify!(session.transport, _jsonrpc_serialize(notice);
            cfg=RequestConfig(excfg; request_timeout=min(excfg.request_timeout, _MCP_CANCEL_BOUND)))
    catch e
        e isa InterruptException && rethrow()
        @debug "MCP cancellation notice not delivered" request_id = id exception = e
    end
    nothing
end

"""Handle one incoming frame of an exchange; a response is returned for the caller to
match, every other frame yields `nothing`. Server notifications are skipped —
`notifications/tools/list_changed` marks the tool cache stale; a server `ping` request is
answered with an empty result and any other server request with `-32601` (this client
offers no server-callable capabilities). A line that is not JSON is skipped with a
(bounded) warning: stdio servers must write only protocol messages to stdout, but a stray
log line must not fail the exchange it lands in."""
function _serve_frame!(session::MCPSession, raw::String,
                       excfg::RequestConfig)::Union{_JSONRPCResponse,Nothing}
    parsed = try
        JSON.parse(raw; dicttype=Dict{String,Any})
    catch e
        e isa InterruptException && rethrow()
        @warn "Skipping a non-JSON line from the MCP server" line = first(raw, 200) maxlog = 10
        return nothing
    end
    parsed isa Dict{String,Any} || error("MCP server sent a non-object JSON-RPC frame: $raw")
    haskey(parsed, "method") || return _JSONRPCResponse(parsed)
    frame_id = get(parsed, "id", nothing)
    if isnothing(frame_id)
        parsed["method"] == "notifications/tools/list_changed" && (session.tools_stale = true)
    else
        reply = parsed["method"] == "ping" ? _jsonrpc_result(frame_id, Dict{String,Any}()) :
            _jsonrpc_error(frame_id, -32601, "Method not found: $(parsed["method"])")
        _transport_notify!(session.transport, JSON.json(reply); cfg=excfg)
    end
    nothing
end

"""
Send a JSON-RPC request and return the parsed response, throwing MCPError on failure.

The whole exchange — id allocation, request write, and reading frames until
the response with the matching id arrives — runs under `session._lock`, so
concurrent callers are serialized and cannot interleave reads. Frames are handled
by [`_serve_frame!`](@ref): server notifications and requests before the response
are served in place, and so are the frames that follow it in the same body (the
rest of an HTTP SSE stream). Among responses:

- A response with a `null` id carrying an `error` aborts the exchange with
  [`MCPError`](@ref) (the server could not attribute the request).
- Any other non-matching response frame is skipped with a warning.

A request that times out in the transport (HTTP) is cancelled with a best-effort
`notifications/cancelled` before the timeout propagates; `initialize` is never cancelled.
"""
function _mcp_request_once!(session::MCPSession, method::String,
                            params::Union{Dict{String,Any},Nothing}=nothing;
                            excfg::RequestConfig=session.config)::Dict{String,Any}
    @lock session._lock begin
        t = session.transport
        id = _next_id!(session)
        raw = try
            _transport_send!(t, _jsonrpc_serialize(_JSONRPCRequest(id, method, params)); cfg=excfg)
        catch e
            e isa InterruptException && rethrow()
            e isa UniLMTimeout && e.phase === :request && method != "initialize" &&
                _send_cancelled!(session, id, excfg)
            rethrow()
        end
        while true
            resp = _serve_frame!(session, raw, excfg)
            if resp !== nothing && resp.id == id
                for extra in _transport_trailing!(t)
                    late = _serve_frame!(session, extra, excfg)
                    late === nothing || @warn "Skipping response with unexpected id" expected=id got=late.id
                end
                isnothing(resp.error) || throw(MCPError(resp.error))
                return something(resp.result, Dict{String,Any}())
            end
            if resp !== nothing
                isnothing(resp.id) && !isnothing(resp.error) && throw(MCPError(resp.error))
                @warn "Skipping response with unexpected id" expected=id got=resp.id
            end
            raw = _transport_read!(t)
        end
    end
end

"""
    _mcp_request_recover!(session, method, params=nothing; excfg=session.config) -> Dict{String,Any}

Run a request, transparently recovering from an expired HTTP session. On a 404
carrying a live session id the client re-initializes once — obtaining a fresh
session id — and replays the request a single time (MCP Streamable HTTP: the
client re-initializes when the server reports the session gone). A second
expiry aborts with an error; there is no retry loop. Every other outcome,
including [`MCPError`](@ref), passes through unchanged.
"""
function _mcp_request_recover!(session::MCPSession, method::String,
                               params::Union{Dict{String,Any},Nothing}=nothing;
                               excfg::RequestConfig=session.config)::Dict{String,Any}
    try
        return _mcp_request_once!(session, method, params; excfg=excfg)
    catch e
        e isa _MCPSessionExpired || rethrow()
        # Re-initialize at the connect bound (a fresh handshake is a connect action).
        _mcp_reinitialize!(session;
            excfg=RequestConfig(session.config; request_timeout=session.config.mcp_connect_timeout))
        try
            return _mcp_request_once!(session, method, params; excfg=excfg)
        catch e2
            e2 isa _MCPSessionExpired || rethrow()
            error("MCP HTTP session expired again immediately after " *
                  "re-initialization; aborting.")
        end
    end
end

"""Run `f()` holding the session. The wait for it is bounded by the caller's per-call
bound (`timeout`, else the ambient/session `mcp_request_timeout`): a caller that cannot
acquire the session in time raises `MCPTimeoutError(:queue)` without touching it. The
holder's exchange is never cut short by a waiter's clock. Re-entrant."""
function _with_session(f::Function, session::MCPSession, timeout::Union{Nothing,Float64})
    limit = _resolve_mcp_request_timeout(session, timeout)
    t0 = time_ns()
    _acquire!(session._lock, limit) ||
        throw(MCPTimeoutError(:queue, _elapsed_s(t0), limit, _queue_timeout_msg(limit)))
    try
        f()
    finally
        unlock(session._lock)
    end
end

"""
Guarded request entry point used by every discovery/operation verb. Holds the session
for the whole call — liveness check, any auto-respawn it triggers, and the exchange —
acquired within the per-call bound (see [`_with_session`](@ref)); resolves the
per-exchange bound (call-time), routes through the 404-recovery wrapper, and maps a
per-exchange timeout to a typed [`MCPTimeoutError`](@ref). The stdio branch arms the
whole-exchange watchdog only once the session is HELD, so the exchange gets its full
bound and a waiter's clock never tears down the exchange in progress. A stdio timeout
is session-fatal — killing the server is the only way to release a read blocked on
it — and once the watchdog fired the call is a timeout, even if a reply raced the
teardown. EXCEPT during connect-phase discovery (status `:initializing`), which runs
directly beneath the enclosing connect deadline in [`_establish!`](@ref) instead of
arming a second, tighter watchdog. HTTP timeouts are NOT session-fatal
(request/response correlation is per-POST).
"""
function _mcp_request!(session::MCPSession, method::String,
                       params::Union{Dict{String,Any},Nothing}=nothing;
                       timeout::Union{Nothing,Float64}=nothing)::Dict{String,Any}
    # ONE acquisition spans liveness check → respawn → exchange. The liveness check is
    # check-then-act on session state (status, transport, id counter): unsynchronized,
    # two callers on a closed session both respawn, the loser's overwritten transport
    # orphans its server process beyond every kill ladder, and the two handshakes
    # interleave on the shared id counter. The exchange below re-locks re-entrantly.
    _with_session(session, timeout) do
        # A closed session is never reused: respawn (opt-in, stdio timeout or crash) or
        # raise MCPSessionClosedError before touching the transport. Respawn's fresh
        # handshake avoids this guard (it runs through _establish!), and its list_tools!
        # re-enters here while status is :initializing, so the guard no-ops — no
        # re-entrant respawn.
        _ensure_live!(session)
        bound = _mcp_exchange_bound(session, timeout)
        excfg = RequestConfig(session.config; request_timeout=bound)
        if session.transport isa StdioTransport
            t = session.transport
            # Connect-phase rule: discovery (list_tools!/list_resources!/list_prompts!)
            # re-enters here from _finalize_connect! while the session is :initializing,
            # already beneath the enclosing :connect deadline in _establish!. Arming the
            # per-exchange request watchdog here would double-arm the exchange, and its
            # completion-race tail (which closes the session on a ~deadline race) could
            # tear a healthy connect down. Under :initializing, run the exchange directly
            # and let the connect deadline bound it.
            session.status === :initializing &&
                return _mcp_request_recover!(session, method, params; excfg=excfg)
            t0 = time_ns()
            try
                # ONE deadline for the whole exchange, armed once the session is held: a
                # burst of pre-response notifications cannot reset it, and time spent
                # waiting for the session cannot consume it — a bound armed before the
                # acquisition would expire during the HOLDER's healthy exchange and
                # group-kill its server, surfacing to the holder as a crash. On breach
                # the watchdog releases the in-flight readline, then its ladder
                # group-kills the server — which ends the session.
                result, fired = _with_deadline_reported(
                    () -> _mcp_request_recover!(session, method, params; excfg=excfg),
                    () -> _abort_exchange!(t), bound, :request)
                fired || return result
                # The watchdog won the completion race: decided by GUARD STATE (the
                # :armed→:fired CAS, set before its teardown begins). The teardown is
                # closing the transport, so a reply that made it through is late — the
                # call timed out.
                throw(UniLMTimeout(:request, _elapsed_s(t0), bound))
            catch e
                e isa InterruptException && rethrow()
                if e isa UniLMTimeout && e.phase === :request
                    session.status = :closed
                    session._close_cause = :timeout
                    throw(MCPTimeoutError(:request, e.elapsed, e.limit, _request_timeout_msg(e.limit)))
                end
                # Crash classification AFTER the timeout branch: when the watchdog fired,
                # the surfaced error is the timeout even though its teardown also broke
                # the pipe. _find_exception sees through task wrapping.
                tc = _find_exception(x -> x isa _TransportClosed, e)
                tc !== nothing && _crash_close!(session, t, tc, "an MCP exchange")
                rethrow()
            end
        else
            # HTTP arms no exchange watchdog: `excfg` bounds each POST inside _http, so
            # the bound selected above IS the enforcement — including the connect-phase
            # substitution, which needs no `:initializing` bypass here (there is no
            # second watchdog to double-arm). A breach while :initializing IS a connect
            # timeout and must name that override, not the per-exchange one.
            try
                return _mcp_request_recover!(session, method, params; excfg=excfg)
            catch e
                e isa InterruptException && rethrow()
                if e isa UniLMTimeout && e.phase in (:request, :connect)
                    session.status === :initializing && throw(MCPTimeoutError(
                        :connect, e.elapsed, e.limit, _connect_timeout_msg(e.limit)))
                    throw(MCPTimeoutError(:request, e.elapsed, e.limit, _request_timeout_msg(e.limit)))
                end
                rethrow()
            end
        end
    end
end

"""Send a JSON-RPC notification (no response expected). Runs under `session._lock`:
a notification travels over the same transport an exchange may be holding (the HTTP
endpoint, the stdio pipe) and touches the same transport state, so it is serialized
WITH exchanges rather than interleaved into one. Re-entrant — the handshake already
holds the lock when an expired HTTP session is re-initialized mid-request."""
function _mcp_notify!(session::MCPSession, method::String,
                      params::Union{Dict{String,Any},Nothing}=nothing;
                      excfg::RequestConfig=session.config)
    notif = _JSONRPCNotification(method, params)
    @lock session._lock _transport_notify!(session.transport, _jsonrpc_serialize(notif); cfg=excfg)
end

# ─── Lifecycle ───────────────────────────────────────────────────────────────

"""
    mcp_connect(command::Cmd; stderr=nothing, client_name="UniLM.jl",
                protocol_version="2025-11-25", config=nothing, auto_respawn=false) -> MCPSession

Connect to an MCP server via stdio transport (subprocess). `stderr` is where the
server's stderr goes — an `IO` such as `devnull`, or a file path (appended to);
`nothing` (the default) inherits this process's stderr. A respawned server keeps the
same setting.

`config::Union{Nothing,RequestConfig}` is resolved (`config` if given, else the
ambient/process default) and captured on the session: `config.mcp_connect_timeout`
bounds the spawn→initialize handshake, `config.mcp_request_timeout` is the default
per-call bound (it covers the wait for the session and then the exchange). A stdio
request timeout is session-fatal — killing the server is the only way to release a
read blocked on an unresponsive one — as is a server crash; with `auto_respawn=true`
the next call respawns the server (same command, fresh handshake — in-memory server
state is lost), otherwise it raises [`MCPSessionClosedError`](@ref).

The server runs in its own process group. Disconnecting (or a fatal timeout) tears it
down: stdin EOF, then SIGTERM, then a SIGKILL of the whole group. If its group leader
exits on its own, the rest of the group is killed at once; and servers still running
when this process exits are torn down by an exit hook.

# Example
```julia
session = mcp_connect(`npx -y @modelcontextprotocol/server-filesystem /tmp`)
tools = mcp_tools(session)
# ... use tools with tool_loop! ...
mcp_disconnect!(session)
```
"""
function mcp_connect(command::Cmd; stderr::Union{Nothing,IO,AbstractString}=nothing,
                     kwargs...)::MCPSession
    mcp_connect(StdioTransport(command; stderr); kwargs...)
end

"""
    mcp_connect(url::String; headers=[], client_name="UniLM.jl", protocol_version="2025-11-25",
                config=nothing, auto_respawn=false) -> MCPSession

Connect to an MCP server via HTTP transport.

`config::Union{Nothing,RequestConfig}` is resolved (`config` if given, else the
ambient/process default) and captured on the session: `config.mcp_connect_timeout`
bounds the connect step, `config.mcp_request_timeout` is the default per-call bound.
A per-call timeout is not session-fatal over HTTP; the timed-out request is cancelled
with a best-effort `notifications/cancelled`.

# Example
```julia
session = mcp_connect("https://mcp.example.com/mcp";
    headers=["Authorization" => "Bearer token"])
```
"""
function mcp_connect(url::String; headers::Vector{Pair{String,String}}=Pair{String,String}[], kwargs...)::MCPSession
    mcp_connect(HTTPTransport(url; headers); kwargs...)
end

"""
Run the MCP `initialize` handshake on `session` from its stored `_init_params`,
validate the server's protocol version, and send `notifications/initialized`.
Returns the raw `initialize` result.

The initialize request advertises the client's preferred protocol version in the
`Mcp-Protocol-Version` header; once the server's version is accepted it becomes
the header value for every subsequent request. If the server returns a version
this client does not support, the transport is disconnected and an error naming
both the requested and returned versions is thrown (MCP spec: the client SHOULD
terminate the connection when it cannot support the negotiated version).
"""
function _mcp_handshake!(session::MCPSession; excfg::RequestConfig=session.config)::Dict{String,Any}
    requested = get(session._init_params, "protocolVersion", _MCP_PROTOCOL_VERSION)
    # The initialize request advertises the client's preferred (latest) revision.
    _set_protocol_version!(session.transport, _MCP_PROTOCOL_VERSION)
    init_result = _mcp_request_once!(session, "initialize", session._init_params; excfg=excfg)
    negotiated = get(init_result, "protocolVersion", requested)
    if !(negotiated in _MCP_SUPPORTED_PROTOCOL_VERSIONS)
        _transport_disconnect!(session.transport; cfg=excfg)
        session.status = :closed
        supported = join(_MCP_SUPPORTED_PROTOCOL_VERSIONS, ", ")
        error("MCP server returned unsupported protocol version \"$(negotiated)\" " *
              "(client requested \"$(requested)\"; supported: $(supported)). " *
              "Connection closed.")
    end
    session.protocol_version = negotiated
    # Every request after initialize carries the negotiated version.
    _set_protocol_version!(session.transport, negotiated)
    _mcp_notify!(session, "notifications/initialized"; excfg=excfg)
    init_result
end

"""
Re-establish an expired HTTP session: drop the stale session id so the server
issues a fresh one, then re-run the `initialize` handshake. Used by
[`_mcp_request_recover!`](@ref) when a request 404s.
"""
function _mcp_reinitialize!(session::MCPSession; excfg::RequestConfig=session.config)
    _reset_session!(session.transport)
    _mcp_handshake!(session; excfg=excfg)
    nothing
end

# ─── Connect handshake watchdog ──────────────────────────────────────────────

_connect_timeout_msg(limit::Float64)::String =
    "MCP server did not complete the initialize handshake within $(limit)s. " *
    "Cold-starting a server (for example via `npx`) can legitimately take longer; " *
    "raise the bound with mcp_connect(...; config=RequestConfig(current_config(); " *
    "mcp_connect_timeout=<seconds>))."

# stdio: the whole spawn+handshake+discovery runs under one connect watchdog (kill on
# breach). Returns whether the watchdog fired on a lost completion race (f returned a
# real result as the timer fired). Every other transport runs unguarded and bounds its
# own IO — HTTP bounds each initialize/list POST inside _http; a custom transport must
# do the same (see MCPTransport) — so its connect guard never fires.
_connect_guard(f::Function, t::StdioTransport, bound::Float64)::Bool =
    last(_with_deadline_reported(f, () -> _kill_transport!(t), bound, :connect))
_connect_guard(f::Function, ::MCPTransport, ::Float64)::Bool = (f(); false)

"""Spawn (stdio) / mark connected (http), run the initialize handshake, and run
tool/resource/prompt discovery — the whole spawn → ready sequence — under ONE
connect deadline. On success the session is `:ready`. A breach surfaces as
`UniLMTimeout(:connect)`; the caller maps it to `MCPTimeoutError(:connect)`.

Discovery (`list_tools!`/`list_resources!`/`list_prompts!`) runs here, inside the
connect guard, so a cold server slow to answer `tools/list` is bounded by
`mcp_connect_timeout` — the phase and bound the connect-timeout message promises —
not the tighter per-exchange `mcp_request_timeout`. Those discovery calls re-enter
`_mcp_request!` while the session is `:initializing`; that path runs the exchange
directly beneath this deadline (see [`_mcp_request!`](@ref))."""
function _establish!(session::MCPSession)
    t = session.transport
    cfg = session.config
    bound = cfg.mcp_connect_timeout
    connect_excfg = RequestConfig(cfg; request_timeout=bound)
    t0 = time_ns()
    fired = _connect_guard(t, bound) do
        _transport_connect!(t)
        init_result = _mcp_handshake!(session; excfg=connect_excfg)
        _finalize_connect!(session, init_result)
    end
    _guard_connect_completion!(session, fired, t0, bound)
end

"""Connect-completion symmetry (mirror of the request-path completion-race tail): the
connect guard can return a real result while its timer fired at ~completion and ran
the kill ladder. Detected by GUARD STATE (`fired`, the :armed→:fired CAS set
atomically before close! begins) — not by nulled handles, which _kill_transport!
nulls only at the END of its grace ladder, a settling window in which the check would
miss the kill. On a fired connect the session is `:ready` over a dying transport and
must not escape: reflect the close and surface the connect timeout naming
`mcp_connect_timeout`. `fired` is only ever true for stdio (the HTTP connect path has
no watchdog)."""
function _guard_connect_completion!(session::MCPSession, fired::Bool, t0::UInt64, bound::Float64)
    if fired
        session.status = :closed
        session._close_cause = :timeout
        throw(MCPTimeoutError(:connect, _elapsed_s(t0), bound, _connect_timeout_msg(bound)))
    end
    nothing
end

"""Populate server info/capabilities, auto-list advertised primitives, then mark the
session ready. Runs inside the connect deadline (see [`_establish!`](@ref)), so its
discovery is governed by `mcp_connect_timeout`. Shared by initial connect and
auto-respawn."""
function _finalize_connect!(session::MCPSession, init_result::Dict{String,Any})
    session.server_info = get(init_result, "serverInfo", Dict{String,Any}())
    session.server_capabilities = MCPServerCapabilities(get(init_result, "capabilities", Dict{String,Any}()))
    !isnothing(session.server_capabilities.tools) && list_tools!(session)
    !isnothing(session.server_capabilities.resources) && list_resources!(session)
    !isnothing(session.server_capabilities.prompts) && list_prompts!(session)
    session.status = :ready
    nothing
end

_closed_session_msg()::String =
    "MCP stdio session was closed by a request timeout and cannot be reused: killing " *
    "the server is the only way to release a read blocked on an unresponsive one, and " *
    "that ends the session. Reconnect explicitly, or pass auto_respawn=true to " *
    "mcp_connect so the next call transparently respawns the server (its in-memory " *
    "state is lost and tools are refetched)."

_crashed_session_msg()::String =
    "MCP stdio session was closed by a server crash and cannot be reused. " *
    "Reconnect explicitly, or pass auto_respawn=true to mcp_connect so the next " *
    "call transparently respawns the server (its in-memory state is lost and " *
    "tools are refetched)."

_disconnected_session_msg()::String =
    "MCP session is not connected: it was closed (by mcp_disconnect!, or by a failed " *
    "connect). Reconnect with mcp_connect."

"""Tear down a stdio transport whose connect sequence failed, then let the failure
escape. UNCONDITIONAL by design: once the server is spawned, ANY failure before the
session is established — an `initialize` error frame, a rejected protocol version, a
non-object frame — leaves a live child holding our pipes, so
teardown must not be a per-exception-shape branch that the next new failure mode
escapes. Crash-shaped failures route through [`_crash_close!`](@ref), which captures
exit diagnostics, tears down and throws the typed error; everything else is torn down
here and the session marked closed. A no-op for HTTP transports (no child process)."""
function _abort_connect!(session::MCPSession, e, context::String)
    t = session.transport
    t isa StdioTransport || return nothing
    tc = _find_exception(x -> x isa _TransportClosed, e)
    tc === nothing || _crash_close!(session, t, tc, context)   # throws
    _kill_transport!(t)   # idempotent: a watchdog breach already ran the ladder
    session.status = :closed
    nothing
end

"""Respawn a stdio session closed by a timeout or a server crash: fresh transport (same command),
captured config, fresh handshake. In-memory server state is lost and tools are
refetched. Throws `MCPTimeoutError(:connect)` if the respawned server does not
hand-shake in time, or `MCPCrashError` if it dies during the respawn handshake.
Callers must hold `session._lock` (see [`_ensure_live!`](@ref))."""
function _respawn!(session::MCPSession)
    session.transport isa StdioTransport ||
        error("MCP auto-respawn is only supported for stdio sessions.")
    old = session.transport
    recorded = session._close_cause
    reason = recorded === :crash ? "a server crash" : "a request timeout"
    @warn "MCP stdio session was closed by $reason; respawning the server. \
           In-memory server state is lost and tools are refetched." command=old.command
    session.transport = StdioTransport(old.command; stderr=old.stderr)
    session._id_counter = 0
    session.tools_stale = false
    session._close_cause = :none
    session.status = :initializing
    try
        _establish!(session)
    catch e
        # Tear the half-established server down whatever went wrong (_abort_connect!
        # throws for crash-shaped failures, which ARE a fresh diagnosis).
        _abort_connect!(session, e, "an auto-respawn attempt")
        if e isa UniLMTimeout
            session._close_cause = :timeout
            throw(MCPTimeoutError(:connect, e.elapsed, e.limit, _connect_timeout_msg(e.limit)))
        end
        # The attempt is cleared to :none above, so a non-:none cause here was diagnosed
        # BY the attempt (the connect-completion tail records :timeout) and stands.
        # Otherwise the attempt learned nothing about why the session closed: keep the
        # recorded cause, so the next call still reports the reason that closed it.
        session._close_cause === :none && (session._close_cause = recorded)
        rethrow()
    end
    nothing
end

"""Guard against reusing a closed session: a stdio session closed by a request timeout
or a server crash respawns when opted in; every other closed session raises
[`MCPSessionClosedError`](@ref) with its cause and recovery guidance (`:disconnected`
for a normal close). A no-op for live sessions. Check and respawn run under
`session._lock` so they cannot interleave: concurrent callers on one closed session
respawn exactly ONE server, and the callers that lose the race find it already live.
Re-entrant — [`_mcp_request!`](@ref) already holds the lock across the whole call."""
function _ensure_live!(session::MCPSession)
    @lock session._lock begin
        session.status === :closed || return nothing
        cause = session._close_cause
        cause !== :none && session.auto_respawn && return _respawn!(session)
        throw(MCPSessionClosedError(cause === :none ? :disconnected : cause,
            cause === :crash ? _crashed_session_msg() :
            cause === :timeout ? _closed_session_msg() : _disconnected_session_msg()))
    end
end

"""
    mcp_connect(transport::MCPTransport; client_name="UniLM.jl",
                client_version=string(pkgversion(UniLM)), protocol_version="2025-11-25",
                config=nothing, auto_respawn=false) -> MCPSession

Connect to an MCP server via the given transport. Performs initialization handshake
and populates tool cache; `client_name`/`client_version` are sent as `clientInfo`.
`config` (a [`RequestConfig`](@ref), default: the ambient configuration) is resolved
and captured on the session — its `mcp_connect_timeout` bounds this handshake and
`mcp_request_timeout` bounds each later call (the wait for the session, then the
exchange). `auto_respawn=true` lets a stdio session closed by a request timeout or a
server crash respawn its server (same command, captured config) and retry the next
call once. A custom `MCPTransport` must bound its own IO (see [`MCPTransport`](@ref)).
"""
function mcp_connect(transport::MCPTransport;
                     client_name::String="UniLM.jl",
                     client_version::String=string(pkgversion(@__MODULE__)),
                     protocol_version::String=_MCP_PROTOCOL_VERSION,
                     config::Union{Nothing,RequestConfig}=nothing,
                     auto_respawn::Bool=false)::MCPSession
    cfg = _resolve_config(config)
    init_params = Dict{String,Any}(
        "protocolVersion" => protocol_version,
        "capabilities" => Dict{String,Any}(),
        "clientInfo" => Dict{String,Any}("name" => client_name, "version" => client_version)
    )
    session = MCPSession(
        transport, MCPServerCapabilities(), Dict{String,Any}(),
        MCPToolInfo[], MCPResourceInfo[], MCPPromptInfo[],
        protocol_version, 0, :initializing; init_params=init_params,
        config=cfg, auto_respawn=auto_respawn
    )
    # The whole spawn → initialize → notifications/initialized → discovery sequence
    # runs under one connect deadline (see _establish!): a cold server slow to
    # hand-shake OR to answer tools/list is governed by mcp_connect_timeout, and a
    # breach surfaces as UniLMTimeout mapped to a naming MCPTimeoutError. A synchronous
    # spawn failure (a nonexistent command) throws inside the guard before the timer
    # fires and rethrows unchanged, so command-not-found surfaces immediately rather
    # than riding the connect timer.
    try
        _establish!(session)
    catch e
        # Never leave a spawned server behind: the teardown runs for every failure
        # shape (see _abort_connect!), which also throws the typed crash error when
        # the failure was the server dying.
        _abort_connect!(session, e, "the connect handshake")
        e isa UniLMTimeout &&
            throw(MCPTimeoutError(:connect, e.elapsed, e.limit, _connect_timeout_msg(e.limit)))
        rethrow()
    end
    session
end

"""
    mcp_connect(f::Function, args...; kwargs...)

Do-block form: automatically disconnects after the block executes.

# Example
```julia
mcp_connect(`npx server`) do session
    tools = mcp_tools(session)
    chat = Chat(model="gpt-5.4-mini", tools=tools)
    push!(chat, Message(Val(:system), "You can use the server's tools."))
    push!(chat, Message(Val(:user), "List files"))
    tool_loop!(chat; tools)
end
```
"""
function mcp_connect(f::Function, args...; kwargs...)
    session = mcp_connect(args...; kwargs...)
    try
        f(session)
    finally
        mcp_disconnect!(session)
    end
end

"""
    mcp_disconnect!(session::MCPSession)

Gracefully disconnect from the MCP server.

Takes the session lock, so a disconnect racing a call in flight WAITS for that
exchange to finish instead of tearing the transport down under its reader — the same
concurrency-1 semantics every other call obeys. The wait stays bounded transitively:
the exchange ahead runs under its own `mcp_request_timeout`. Every later call on the
session raises [`MCPSessionClosedError`](@ref) with cause `:disconnected`.
"""
function mcp_disconnect!(session::MCPSession)
    @lock session._lock begin
        _transport_disconnect!(session.transport;
            cfg=RequestConfig(session.config; request_timeout=session.config.mcp_request_timeout))
        session.status = :closed
        # User intent wins: an explicit disconnect is a normal close, not a timeout, so
        # the next call must never respawn or cite auto_respawn — even if this session
        # was closed by a request timeout or a server crash before the caller
        # disconnected it.
        session._close_cause = :none
    end
    nothing
end

# ─── Discovery ───────────────────────────────────────────────────────────────

"""Every page of a paginated list request, following `nextCursor` (at most 1000 pages):
the entries under `key`, each converted with `T`."""
function _paginate(session::MCPSession, method::String, key::String, ::Type{T};
                   timeout::Union{Nothing,Float64}=nothing)::Vector{T} where {T}
    items = T[]
    cursor = nothing
    for _ in 1:1000
        params = isnothing(cursor) ? Dict{String,Any}() : Dict{String,Any}("cursor" => cursor)
        result = _mcp_request!(session, method, params; timeout)
        append!(items, (T(d) for d in get(result, key, [])))
        cursor = get(result, "nextCursor", nothing)
        isnothing(cursor) && return items
    end
    error("MCP pagination exceeded 1000 pages")
end

"""
    list_tools!(session::MCPSession) -> Vector{MCPToolInfo}

Fetch the tool list from the MCP server. Handles pagination via cursor.
Stores result in `session.tools`.

`timeout::Union{Nothing,Float64}` overrides the per-call bound for this call
(kwarg > ambient [`with_request_config`](@ref) > session-captured config; Inf
disables, NaN/≤0 rejected). The whole listing — every page and the cache update —
holds the session once, so a `list_changed` another caller receives meanwhile is
recorded after the refresh and leaves `tools_stale` set.
"""
function list_tools!(session::MCPSession; timeout::Union{Nothing,Float64}=nothing)::Vector{MCPToolInfo}
    _with_session(session, timeout) do
        session.tools = _paginate(session, "tools/list", "tools", MCPToolInfo; timeout)
        session.tools_stale = false
        session.tools
    end
end

"""
    list_resources!(session::MCPSession) -> Vector{MCPResourceInfo}

Fetch the resource list from the MCP server. Handles pagination.
"""
list_resources!(session::MCPSession)::Vector{MCPResourceInfo} =
    session.resources = _paginate(session, "resources/list", "resources", MCPResourceInfo)

"""
    list_prompts!(session::MCPSession) -> Vector{MCPPromptInfo}

Fetch the prompt list from the MCP server. Handles pagination.
"""
list_prompts!(session::MCPSession)::Vector{MCPPromptInfo} =
    session.prompts = _paginate(session, "prompts/list", "prompts", MCPPromptInfo)

# ─── Tool Operations ────────────────────────────────────────────────────────

"""
    MCPToolResult

The typed result of a [`call_tool`](@ref) call, mirroring an MCP `tools/call`
result. A tool-*execution* error (`isError: true` on the wire) is reported here
as data (`is_error == true`), not thrown, so callers can distinguish it from a
JSON-RPC *protocol* error (which still throws [`MCPError`](@ref)).

# Fields
- `content::String`: the content parts rendered to text — `text` parts joined
  with `\\n`, each non-text part JSON-encoded.
- `structured::Union{Nothing,Dict{String,Any}}`: the server's `structuredContent`
  object verbatim when present, otherwise `nothing`.
- `is_error::Bool`: `true` when the server flagged the call as a tool-execution
  error (`isError`); the detail is carried in `content`.
- `parts::Vector{Any}`: the raw `content` array exactly as received, before it is
  rendered into `content::String`.

The [`mcp_tools`](@ref) / [`mcp_tools_respond`](@ref) bridges surface this to a
tool-calling loop: `content` on success (falling back to a JSON encoding of
`structured` when `content` is empty), or a raised error carrying `content` when
`is_error`.
"""
struct MCPToolResult
    content::String
    structured::Union{Nothing,Dict{String,Any}}
    is_error::Bool
    parts::Vector{Any}
end

# JSON-RPC params are objects with string keys, so any AbstractDict a caller
# writes is accepted and normalized here. Without this the natural literal
# `Dict("path" => "/x")` — which infers Dict{String,String} — is a MethodError.
_mcp_arguments(d::AbstractDict)::Dict{String,Any} =
    Dict{String,Any}(string(k) => v for (k, v) in d)
_mcp_arguments(d::Dict{String,Any})::Dict{String,Any} = d

"""
    call_tool(session::MCPSession, name::String, arguments::AbstractDict) -> MCPToolResult

Call a tool on the MCP server and return its result as an [`MCPToolResult`](@ref).

`content` concatenates text content parts (non-text parts JSON-encoded);
`structured` carries the server's `structuredContent` verbatim; `parts` is the raw
content array. A tool-execution error (`isError: true`) is returned with
`is_error == true` — it is **not** thrown. JSON-RPC protocol errors still throw
[`MCPError`](@ref).

`timeout::Union{Nothing,Float64}` overrides the per-call bound for this call
(kwarg > ambient [`with_request_config`](@ref) > session-captured config; Inf
disables, NaN/≤0 rejected). It bounds the wait for the session — calls on one session
run one at a time, in arrival order; a call that cannot acquire it in time raises
[`MCPTimeoutError`](@ref) with phase `:queue` — and then, from acquisition, the exchange
itself (phase `:request`).
"""
function call_tool(session::MCPSession, name::String,
                   arguments::AbstractDict=Dict{String,Any}();
                   timeout::Union{Nothing,Float64}=nothing)::MCPToolResult
    result = _mcp_request!(session, "tools/call", Dict{String,Any}(
        "name" => name, "arguments" => _mcp_arguments(arguments)); timeout=timeout)
    content = get(result, "content", Any[])
    is_error = get(result, "isError", false) === true
    text = join((part isa Dict ? (get(part, "type", "") == "text" ? part["text"] : JSON.json(part)) :
                 string(part) for part in content), "\n")
    sc = get(result, "structuredContent", nothing)
    structured = sc isa Dict{String,Any} ? sc : nothing
    MCPToolResult(text, structured, is_error, content)
end

"""
    read_resource(session::MCPSession, uri::String) -> String

Read a resource from the MCP server.
"""
function read_resource(session::MCPSession, uri::String)::String
    result = _mcp_request!(session, "resources/read", Dict{String,Any}("uri" => uri))
    join((haskey(c, "text") ? c["text"] : c["blob"]
          for c in get(result, "contents", []) if haskey(c, "text") || haskey(c, "blob")), "\n")
end

"""
    get_prompt(session::MCPSession, name::String, arguments::AbstractDict=Dict()) -> Vector{Dict{String,Any}}

Get a rendered prompt from the MCP server. Returns the messages array.
"""
function get_prompt(session::MCPSession, name::String, arguments::AbstractDict=Dict{String,Any}())::Vector{Dict{String,Any}}
    result = _mcp_request!(session, "prompts/get", Dict{String,Any}(
        "name" => name, "arguments" => _mcp_arguments(arguments)
    ))
    get(result, "messages", Dict{String,Any}[])
end

"""
    ping(session::MCPSession)

Send a ping to the MCP server. Throws on error.
"""
function ping(session::MCPSession)
    _mcp_request!(session, "ping")
    nothing
end

# ─── Tool Bridge ─────────────────────────────────────────────────────────────

# Surface an `MCPToolResult` to a tool-loop's string dispatcher. On success return
# the rendered content, falling back to a JSON encoding of `structuredContent` when
# the content is empty but structured data is present. A tool-execution error is
# raised so the loop records an unsuccessful outcome carrying the faithful content.
function _mcp_tool_dispatch(r::MCPToolResult)::String
    r.is_error && error(r.content)
    (isempty(r.content) && !isnothing(r.structured)) ? JSON.json(r.structured) : r.content
end

# Function names OpenAI and Anthropic accept. MCP tool names may also contain dots
# (e.g. `admin.tools.list`) — or, from servers predating the naming guidance, anything.
const _PROVIDER_TOOL_NAME = r"^[a-zA-Z0-9_-]{1,128}$"

"""Provider-safe aliases for MCP tool names, position for position: a conforming name is
kept; any other has each non-conforming character replaced with `_` and is cut to 128
characters. Two tools that map to one alias raise an `ArgumentError` naming both — a
bridged tool set cannot tell them apart."""
function _provider_tool_aliases(names::Vector{String})::Vector{String}
    aliases = [occursin(_PROVIDER_TOOL_NAME, n) ? n :
               first(replace(isempty(n) ? "_" : n, r"[^a-zA-Z0-9_-]" => "_"), 128) for n in names]
    owner = Dict{String,String}()
    for (n, a) in zip(names, aliases)
        prev = get!(owner, a, n)
        prev == n || throw(ArgumentError("MCP tools \"$prev\" and \"$n\" both map to the " *
            "provider-safe tool name \"$a\"; a bridged tool set needs distinct names."))
    end
    aliases
end

# The tool loop dispatches by the advertised alias; the call goes out under the MCP name.
_mcp_bridge_callable(session::MCPSession, name::String) =
    (_::String, args::Dict{String,Any}) -> _mcp_tool_dispatch(call_tool(session, name, args))

"""
    mcp_tools(session::MCPSession) -> Vector{CallableTool{Tool}}

Convert all tools from an MCP session into `CallableTool{Tool}` instances
that work directly with [`tool_loop!`](@ref) (Chat Completions API).

Each tool's callable invokes `call_tool(session, name, args)` under the hood. Tool
names are advertised provider-safe: OpenAI and Anthropic accept only
`^[a-zA-Z0-9_-]{1,128}\$`, so any other character of an MCP name (e.g. the dots in
`admin.tools.list`) becomes `_` — the callable still calls the tool by its MCP name.
Two tools whose names map to the same alias raise an `ArgumentError`.

# Example
```julia
session = mcp_connect(`npx server`)
tools = mcp_tools(session)
chat = Chat(model="gpt-5.4-mini", tools=tools)
push!(chat, Message(Val(:system), "You can use the server's tools."))
push!(chat, Message(Val(:user), "Do something"))
result = tool_loop!(chat; tools)
```
"""
function mcp_tools(session::MCPSession)::Vector{CallableTool{Tool}}
    infos = session.tools
    [CallableTool(Tool(func=FunctionSignature(name=alias, description=info.description,
                                              parameters=info.input_schema)),
                  _mcp_bridge_callable(session, info.name))
     for (info, alias) in zip(infos, _provider_tool_aliases([i.name for i in infos]))]
end

"""
    mcp_tools_respond(session::MCPSession) -> Vector{CallableTool{FunctionTool}}

Convert all tools from an MCP session into `CallableTool{FunctionTool}` instances
that work directly with [`tool_loop`](@ref) (Responses API). Tool names are
advertised provider-safe, as in [`mcp_tools`](@ref).

# Example
```julia
session = mcp_connect("https://mcp.example.com/mcp")
tools = mcp_tools_respond(session)
result = tool_loop("Do something"; tools=tools)
```
"""
function mcp_tools_respond(session::MCPSession)::Vector{CallableTool{FunctionTool}}
    infos = session.tools
    [CallableTool(FunctionTool(name=alias, description=info.description,
                               parameters=info.input_schema),
                  _mcp_bridge_callable(session, info.name))
     for (info, alias) in zip(infos, _provider_tool_aliases([i.name for i in infos]))]
end

# Extend to_tool protocol
to_tool(info::MCPToolInfo) = Tool(func=FunctionSignature(
    name=info.name, description=info.description, parameters=info.input_schema
))
