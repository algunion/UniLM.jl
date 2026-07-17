# ============================================================================
# MCP Client — Model Context Protocol client for UniLM.jl
# Connects to MCP servers (stdio or HTTP), discovers tools, and bridges them
# into CallableTool for seamless tool_loop! / tool_loop integration.
#
# Protocol: JSON-RPC 2.0 over stdio or Streamable HTTP (spec 2025-11-25)
# ============================================================================

# ─── JSON-RPC 2.0 Framing (internal) ────────────────────────────────────────

const _JSONRPC_VERSION = "2.0"

# Protocol revisions this client can negotiate over Streamable HTTP, preferred
# (latest) first. 2024-11-05 is excluded: it predates Streamable HTTP and used
# the separate HTTP+SSE dual-endpoint transport this client does not implement.
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
and optional data from the server.
"""
struct MCPError <: Exception
    code::Int
    message::String
    data::Union{Dict{String,Any},Nothing}
end

MCPError(d::Dict{String,Any}) = MCPError(d["code"], d["message"], get(d, "data", nothing))

function Base.showerror(io::IO, e::MCPError)
    print(io, "MCPError($(e.code)): $(e.message)")
    !isnothing(e.data) && print(io, " data=", e.data)
end

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
- `_transport_send!(t, msg::String)::String` — send JSON-RPC message, return response
- `_transport_read!(t)::String` — read the next incoming JSON-RPC frame
- `_transport_notify!(t, msg::String)` — send notification (no response expected)
- `_transport_disconnect!(t)` — close connection
- `_transport_isconnected(t)::Bool` — check if connected
"""
abstract type MCPTransport end

"""
    StdioTransport <: MCPTransport

Stdio transport: launches a subprocess and communicates via stdin/stdout.
Messages are newline-delimited JSON-RPC 2.0.
"""
mutable struct StdioTransport <: MCPTransport
    command::Cmd
    process::Union{Base.Process,Nothing}
    input::Union{IO,Nothing}
    output::Union{IO,Nothing}
    # Process-group id captured at spawn. Under detach=true the child is its own
    # group leader, so its pgid == its pid. Retained so teardown can group-SIGKILL
    # even after the direct child is reaped (when getpid(process) would fail). POSIX
    # reserves a pid while it still names a live group, so this cannot target a
    # recycled pid.
    pgid::Union{Int32,Nothing}
    lock::ReentrantLock
    StdioTransport(command::Cmd) = new(command, nothing, nothing, nothing, nothing, ReentrantLock())
end

function _transport_connect!(t::StdioTransport)
    # Spawn in the child's own process group (detach=true) so the teardown ladder's
    # final rung can group-SIGKILL grandchildren a wrapper forks (e.g. the node
    # process `npx` launches) that would otherwise survive holding our stdio pipe.
    # Teardown ladder: _kill_transport!.
    proc = open(Cmd(t.command; detach=true), read=true, write=true)
    t.process = proc
    t.pgid = getpid(proc)   # == the child's pgid under detach; capture while alive
    t.input = proc.in
    t.output = proc.out
    nothing
end

function _transport_send!(t::StdioTransport, msg::String;
                          cfg::RequestConfig=current_config())::String
    inp = t.input
    isnothing(inp) && error("StdioTransport not connected")
    lock(t.lock) do
        write(inp, msg, "\n")
        flush(inp)
    end
    _transport_read!(t)   # bounded by the whole-exchange watchdog in _mcp_request!
end

function _transport_read!(t::StdioTransport)::String
    out = t.output
    isnothing(out) && error("StdioTransport not connected")
    line = readline(out)
    isempty(line) && error("MCP server closed connection")
    line
end

function _transport_notify!(t::StdioTransport, msg::String;
                            cfg::RequestConfig=current_config())
    inp = t.input
    isnothing(inp) && error("StdioTransport not connected")
    lock(t.lock) do
        write(inp, msg, "\n")
        flush(inp)
    end
    nothing
end

"""
Tear a stdio transport down with a graceful escalation ladder, then null its
handles. MCP spec: a compliant server exits when its stdin reaches EOF, so we close
stdin first; if the process lingers we escalate SIGTERM. The FINAL rung is
UNCONDITIONAL — a group-directed SIGKILL by the pgid captured at spawn (the child
leads its own group, spawned detach=true) — so a grandchild the leader orphaned by
exiting on stdin EOF cannot survive holding our pipe. POSIX reserves a pid while it
still names a live process group, so the unconditional kill cannot hit a recycled
pid even after the direct child was reaped; ESRCH (empty group) is the expected
no-op. `grace_term`/`grace_kill` are exposed for suite-time control; production
defaults are fixed. Best-effort and idempotent: safe from a disconnect and from the
request/connect watchdog.
"""
function _kill_transport!(t::StdioTransport;
                          grace_term::Float64=5.0, grace_kill::Float64=2.0)::Nothing
    proc = t.process
    pgid = t.pgid
    if !isnothing(proc)
        inp = t.input
        !isnothing(inp) && (try; close(inp); catch; end)   # stdin EOF: compliant servers exit
        if process_running(proc)
            if timedwait(() -> !process_running(proc), grace_term) !== :ok
                try; kill(proc); catch; end                  # SIGTERM
                timedwait(() -> !process_running(proc), grace_kill)
            end
        end
    end
    # Final rung, UNCONDITIONAL: SIGKILL the whole process group by the spawn-captured
    # pgid (getpid on a reaped Process may fail, so use the stored value). Reaps a
    # grandchild the leader orphaned by exiting on stdin EOF. The `pgid > 0` guard is
    # defense-in-depth: `kill(-0, 9)` (POSIX) would signal the CALLER's own process
    # group, so a zero pgid must never reach the group kill even though no reachable
    # spawn path produces one.
    if !isnothing(pgid) && pgid > 0
        rc = ccall(:kill, Cint, (Cint, Cint), -pgid, 9)
        if rc != 0
            e = Base.Libc.errno()
            e == Base.Libc.ESRCH || @debug "MCP group SIGKILL failed" errno=e
        end
    end
    t.process = nothing
    t.input = nothing
    t.output = nothing
    t.pgid = nothing
    nothing
end

# Graceful disconnect uses the same ladder. `cfg` is accepted for signature parity
# with the HTTP transport (which needs it for the DELETE) and ignored here.
_transport_disconnect!(t::StdioTransport; cfg::Union{Nothing,RequestConfig}=nothing) =
    _kill_transport!(t)

function _transport_isconnected(t::StdioTransport)::Bool
    proc = t.process
    !isnothing(proc) && process_running(proc)
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
    lock::ReentrantLock
    pending::Vector{String}  # frames from the last response body, not yet consumed
    function HTTPTransport(url::String; headers::Vector{Pair{String,String}}=Pair{String,String}[])
        new(url, headers, nothing, _MCP_PROTOCOL_VERSION, false, ReentrantLock(), String[])
    end
end

_transport_connect!(t::HTTPTransport) = (t.connected = true; nothing)

function _transport_send!(t::HTTPTransport, msg::String;
                          cfg::RequestConfig=current_config())::String
    hdrs = copy(t.headers)
    push!(hdrs, "Content-Type" => "application/json")
    push!(hdrs, "Accept" => "application/json, text/event-stream")
    push!(hdrs, "Mcp-Protocol-Version" => t.protocol_version)
    !isnothing(t.session_id) && push!(hdrs, "Mcp-Session-Id" => t.session_id)
    resp = _http("POST", t.url, hdrs, msg; cfg=cfg, remaining=Inf)
    # Capture session ID from response
    sid = HTTP.header(resp, "Mcp-Session-Id", "")
    !isempty(sid) && (t.session_id = sid)
    # A 404 while holding a session id means the server expired the session;
    # signal the request layer to re-initialize (Streamable HTTP session
    # lifecycle) rather than failing the call.
    if resp.status == 404 && !isnothing(t.session_id)
        throw(_MCPSessionExpired(String(resp.body)))
    end
    # This client implements no authentication flow; credentials travel as
    # request headers, so point the caller at the mechanism that supplies them.
    if resp.status == 401 || resp.status == 403
        error("MCP HTTP request rejected with status $(resp.status). The server " *
              "requires authentication; pass credentials via the `headers` kwarg " *
              "of mcp_connect (e.g. headers=[\"Authorization\" => \"Bearer <token>\"]).")
    end
    resp.status == 200 || error("MCP HTTP request failed with status $(resp.status): $(String(resp.body))")
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

function _transport_notify!(t::HTTPTransport, msg::String;
                            cfg::RequestConfig=current_config())
    hdrs = copy(t.headers)
    push!(hdrs, "Content-Type" => "application/json")
    push!(hdrs, "Mcp-Protocol-Version" => t.protocol_version)
    !isnothing(t.session_id) && push!(hdrs, "Mcp-Session-Id" => t.session_id)
    _http("POST", t.url, hdrs, msg; cfg=cfg, remaining=Inf)
    nothing
end

function _transport_disconnect!(t::HTTPTransport; cfg::RequestConfig=current_config())
    if t.connected && !isnothing(t.session_id)
        hdrs = copy(t.headers)
        push!(hdrs, "Mcp-Session-Id" => t.session_id)
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

"""Split an SSE response body into its `data:` payloads, in arrival order."""
function _parse_sse_frames(body::String)::Vector{String}
    frames = String[]
    for line in split(body, "\n")
        stripped = strip(line)
        startswith(stripped, "data: ") && push!(frames, stripped[7:end])
    end
    frames
end

# ─── MCPSession ──────────────────────────────────────────────────────────────

"""
    MCPSession

A live connection to an MCP server. Manages lifecycle, transport, and cached
tool/resource/prompt lists.

Create via [`mcp_connect`](@ref). Disconnect via [`mcp_disconnect!`](@ref).

Requests are serialized: each request/response exchange (including its id
allocation) runs under an internal session lock, and interleaved server →
client frames are handled in place (notifications skipped, server `ping`
requests answered). After the server sends
`notifications/tools/list_changed`, `tools_stale` is `true` until the next
[`list_tools!`](@ref).
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
    _lock::ReentrantLock
    tools_stale::Bool
    # The exact `initialize` params (protocolVersion, capabilities, clientInfo)
    # retained so an expired HTTP session can be re-initialized transparently.
    _init_params::Dict{String,Any}
    # Timeout configuration resolved and captured at mcp_connect. Connect/initialize
    # bounds come from here; the per-request bound resolves at call time (see call_tool).
    config::RequestConfig
    # When true, the next call on a session closed by a stdio request timeout respawns
    # the server (same command, fresh handshake) instead of erroring. Default OFF: a
    # silent respawn fabricates session continuity and in-memory server state is lost.
    auto_respawn::Bool
    # True only when `status == :closed` was reached via a request timeout (not a
    # normal disconnect); gates the respawn-or-error decision on the next call.
    _closed_by_timeout::Bool
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
               protocol_version, id_counter, status, ReentrantLock(), false, init_params,
               config, auto_respawn, false)
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

_request_timeout_msg(limit::Float64)::String =
    "MCP request exceeded the $(limit)s request timeout. Raise it for one call with " *
    "call_tool(session, name, args; timeout=<seconds>), for a dynamic scope with " *
    "with_request_config(; mcp_request_timeout=<seconds>), or per session with " *
    "mcp_connect(...; config=RequestConfig(current_config(); mcp_request_timeout=<seconds>))."

"""
Send a JSON-RPC request and return the parsed response, throwing MCPError on failure.

The whole exchange — id allocation, request write, and reading frames until
the response with the matching id arrives — runs under `session._lock`, so
concurrent callers are serialized and cannot interleave reads. Frames received
before the matching response are handled in place:

- Server notifications (no `id`) are skipped; `notifications/tools/list_changed`
  additionally sets `session.tools_stale = true` (refresh via [`list_tools!`](@ref)).
- Server-initiated requests (`id` + `method`): `ping` is answered with an empty
  result; anything else with `-32601` (this client offers no server-callable
  capabilities).
- A response with a `null` id carrying an `error` aborts the exchange with
  [`MCPError`](@ref) (the server could not attribute the request).
- Any other non-matching response frame is skipped with a warning.
"""
function _mcp_request_once!(session::MCPSession, method::String,
                            params::Union{Dict{String,Any},Nothing}=nothing;
                            excfg::RequestConfig=session.config)::Dict{String,Any}
    lock(session._lock) do
        id = _next_id!(session)
        req = _JSONRPCRequest(id, method, params)
        raw = _transport_send!(session.transport, _jsonrpc_serialize(req); cfg=excfg)
        while true
            parsed = JSON.parse(raw; dicttype=Dict{String,Any})
            parsed isa Dict{String,Any} ||
                error("MCP server sent a non-object JSON-RPC frame: $raw")
            if haskey(parsed, "method")
                frame_id = get(parsed, "id", nothing)
                if isnothing(frame_id)
                    # Server → client notification: never "the response".
                    parsed["method"] == "notifications/tools/list_changed" &&
                        (session.tools_stale = true)
                elseif parsed["method"] == "ping"
                    # Server-initiated ping: answer with an empty result.
                    _transport_notify!(session.transport,
                        JSON.json(_jsonrpc_result(frame_id, Dict{String,Any}())); cfg=excfg)
                else
                    # Server-initiated request this client cannot serve.
                    _transport_notify!(session.transport,
                        JSON.json(_jsonrpc_error(frame_id, -32601,
                            "Method not found: $(parsed["method"])")); cfg=excfg)
                end
                raw = _transport_read!(session.transport)
                continue
            end
            resp = _JSONRPCResponse(parsed)
            if resp.id != id
                if isnothing(resp.id) && !isnothing(resp.error)
                    throw(MCPError(resp.error))
                end
                @warn "Skipping response with unexpected id" expected=id got=resp.id
                raw = _transport_read!(session.transport)
                continue
            end
            !isnothing(resp.error) && throw(MCPError(resp.error))
            return something(resp.result, Dict{String,Any}())
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

"""
Guarded request entry point used by every discovery/operation verb. Resolves the
per-exchange bound (call-time), routes through the 404-recovery wrapper, and maps
a per-exchange timeout to a typed [`MCPTimeoutError`](@ref). HTTP timeouts are NOT
session-fatal (request/response correlation is per-POST). The stdio branch is
extended to the whole-exchange watchdog in a later change.
"""
function _mcp_request!(session::MCPSession, method::String,
                       params::Union{Dict{String,Any},Nothing}=nothing;
                       timeout::Union{Nothing,Float64}=nothing)::Dict{String,Any}
    bound = _resolve_mcp_request_timeout(session, timeout)
    excfg = RequestConfig(session.config; request_timeout=bound)
    if session.transport isa StdioTransport
        t = session.transport
        result = try
            # ONE deadline for the whole lock-to-response exchange (armed once at
            # exchange start): a burst of pre-response notifications cannot reset it.
            # On breach the escalation ladder group-kills the server (unblocking the
            # in-flight readline); stdio framing has no id demux, so a late reply
            # could misdeliver — the timeout is therefore session-fatal.
            _with_deadline(() -> _mcp_request_recover!(session, method, params; excfg=excfg),
                           () -> _kill_transport!(t), bound, :request)
        catch e
            if e isa UniLMTimeout && e.phase === :request
                session.status = :closed
                session._closed_by_timeout = true
                throw(MCPTimeoutError(:request, e.elapsed, e.limit, _request_timeout_msg(e.limit)))
            end
            rethrow()
        end
        # Exactly-once race: _with_deadline returns a real result even if the timer
        # fired at ~completion. If the ladder already tore the transport down, the
        # session cannot continue — reflect the close truthfully (spec: it stays closed).
        if !_transport_isconnected(t)
            session.status = :closed
            session._closed_by_timeout = true
        end
        return result
    else
        try
            return _mcp_request_recover!(session, method, params; excfg=excfg)
        catch e
            if e isa UniLMTimeout && e.phase in (:request, :connect)
                throw(MCPTimeoutError(:request, e.elapsed, e.limit, _request_timeout_msg(e.limit)))
            end
            rethrow()
        end
    end
end

"""Send a JSON-RPC notification (no response expected)."""
function _mcp_notify!(session::MCPSession, method::String,
                      params::Union{Dict{String,Any},Nothing}=nothing;
                      excfg::RequestConfig=session.config)
    notif = _JSONRPCNotification(method, params)
    _transport_notify!(session.transport, _jsonrpc_serialize(notif); cfg=excfg)
end

# ─── Lifecycle ───────────────────────────────────────────────────────────────

"""
    mcp_connect(command::Cmd; client_name="UniLM.jl", protocol_version="2025-11-25") -> MCPSession

Connect to an MCP server via stdio transport (subprocess).

# Example
```julia
session = mcp_connect(`npx -y @modelcontextprotocol/server-filesystem /tmp`)
tools = mcp_tools(session)
# ... use tools with tool_loop! ...
mcp_disconnect!(session)
```
"""
function mcp_connect(command::Cmd; kwargs...)::MCPSession
    mcp_connect(StdioTransport(command); kwargs...)
end

"""
    mcp_connect(url::String; headers=[], kwargs...) -> MCPSession

Connect to an MCP server via HTTP transport.

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

"""
    mcp_connect(transport::MCPTransport; client_name="UniLM.jl", protocol_version="2025-11-25") -> MCPSession

Connect to an MCP server via the given transport. Performs initialization handshake
and populates tool cache.
"""
function mcp_connect(transport::MCPTransport;
                     client_name::String="UniLM.jl",
                     client_version::String="0.8.0",
                     protocol_version::String=_MCP_PROTOCOL_VERSION,
                     config::Union{Nothing,RequestConfig}=nothing,
                     auto_respawn::Bool=false)::MCPSession
    cfg = _resolve_config(config)
    _transport_connect!(transport)
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
    # Initialize handshake: validates the negotiated protocol version, stores it
    # on the session, and sends notifications/initialized.
    connect_excfg = RequestConfig(session.config; request_timeout=session.config.mcp_connect_timeout)
    init_result = _mcp_handshake!(session; excfg=connect_excfg)
    session.server_info = get(init_result, "serverInfo", Dict{String,Any}())
    caps = get(init_result, "capabilities", Dict{String,Any}())
    session.server_capabilities = MCPServerCapabilities(caps)
    # Auto-populate caches for whatever the server advertises.
    !isnothing(session.server_capabilities.tools) && list_tools!(session)
    !isnothing(session.server_capabilities.resources) && list_resources!(session)
    !isnothing(session.server_capabilities.prompts) && list_prompts!(session)
    session.status = :ready
    session
end

"""
    mcp_connect(f::Function, args...; kwargs...)

Do-block form: automatically disconnects after the block executes.

# Example
```julia
mcp_connect(`npx server`) do session
    tools = mcp_tools(session)
    chat = Chat(tools=map(t -> t.tool, tools))
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
"""
function mcp_disconnect!(session::MCPSession)
    _transport_disconnect!(session.transport;
        cfg=RequestConfig(session.config; request_timeout=session.config.mcp_request_timeout))
    session.status = :closed
    nothing
end

# ─── Discovery ───────────────────────────────────────────────────────────────

"""
    list_tools!(session::MCPSession) -> Vector{MCPToolInfo}

Fetch the tool list from the MCP server. Handles pagination via cursor.
Stores result in `session.tools`.
"""
function list_tools!(session::MCPSession; timeout::Union{Nothing,Float64}=nothing)::Vector{MCPToolInfo}
    all_tools = MCPToolInfo[]
    cursor = nothing
    pages = 0
    while true
        (pages += 1) > 1000 && error("MCP pagination exceeded 1000 pages")
        params = isnothing(cursor) ? Dict{String,Any}() : Dict{String,Any}("cursor" => cursor)
        result = _mcp_request!(session, "tools/list", params; timeout=timeout)
        for t in get(result, "tools", [])
            push!(all_tools, MCPToolInfo(t))
        end
        cursor = get(result, "nextCursor", nothing)
        isnothing(cursor) && break
    end
    session.tools = all_tools
    session.tools_stale = false
    all_tools
end

"""
    list_resources!(session::MCPSession) -> Vector{MCPResourceInfo}

Fetch the resource list from the MCP server. Handles pagination.
"""
function list_resources!(session::MCPSession)::Vector{MCPResourceInfo}
    all_resources = MCPResourceInfo[]
    cursor = nothing
    pages = 0
    while true
        (pages += 1) > 1000 && error("MCP pagination exceeded 1000 pages")
        params = isnothing(cursor) ? Dict{String,Any}() : Dict{String,Any}("cursor" => cursor)
        result = _mcp_request!(session, "resources/list", params)
        for r in get(result, "resources", [])
            push!(all_resources, MCPResourceInfo(r))
        end
        cursor = get(result, "nextCursor", nothing)
        isnothing(cursor) && break
    end
    session.resources = all_resources
    all_resources
end

"""
    list_prompts!(session::MCPSession) -> Vector{MCPPromptInfo}

Fetch the prompt list from the MCP server. Handles pagination.
"""
function list_prompts!(session::MCPSession)::Vector{MCPPromptInfo}
    all_prompts = MCPPromptInfo[]
    cursor = nothing
    pages = 0
    while true
        (pages += 1) > 1000 && error("MCP pagination exceeded 1000 pages")
        params = isnothing(cursor) ? Dict{String,Any}() : Dict{String,Any}("cursor" => cursor)
        result = _mcp_request!(session, "prompts/list", params)
        for p in get(result, "prompts", [])
            push!(all_prompts, MCPPromptInfo(p))
        end
        cursor = get(result, "nextCursor", nothing)
        isnothing(cursor) && break
    end
    session.prompts = all_prompts
    all_prompts
end

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

"""
    call_tool(session::MCPSession, name::String, arguments::Dict{String,Any}) -> MCPToolResult

Call a tool on the MCP server and return its result as an [`MCPToolResult`](@ref).

`content` concatenates text content parts (non-text parts JSON-encoded);
`structured` carries the server's `structuredContent` verbatim; `parts` is the raw
content array. A tool-execution error (`isError: true`) is returned with
`is_error == true` — it is **not** thrown. JSON-RPC protocol errors still throw
[`MCPError`](@ref).
"""
function call_tool(session::MCPSession, name::String,
                   arguments::Dict{String,Any}=Dict{String,Any}();
                   timeout::Union{Nothing,Float64}=nothing)::MCPToolResult
    result = _mcp_request!(session, "tools/call", Dict{String,Any}(
        "name" => name, "arguments" => arguments); timeout=timeout)
    content = get(result, "content", Any[])
    is_error = get(result, "isError", false) === true
    rendered = String[]
    for part in content
        if part isa Dict
            ptype = get(part, "type", "")
            if ptype == "text"
                push!(rendered, part["text"])
            else
                push!(rendered, JSON.json(part))
            end
        else
            push!(rendered, string(part))
        end
    end
    text = join(rendered, "\n")
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
    contents = get(result, "contents", [])
    parts = String[]
    for c in contents
        if haskey(c, "text")
            push!(parts, c["text"])
        elseif haskey(c, "blob")
            push!(parts, c["blob"])
        end
    end
    join(parts, "\n")
end

"""
    get_prompt(session::MCPSession, name::String, arguments::Dict{String,Any}=Dict()) -> Vector{Dict{String,Any}}

Get a rendered prompt from the MCP server. Returns the messages array.
"""
function get_prompt(session::MCPSession, name::String, arguments::Dict{String,Any}=Dict{String,Any}())::Vector{Dict{String,Any}}
    result = _mcp_request!(session, "prompts/get", Dict{String,Any}(
        "name" => name, "arguments" => arguments
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

"""
    mcp_tools(session::MCPSession) -> Vector{CallableTool{GPTTool}}

Convert all tools from an MCP session into `CallableTool{GPTTool}` instances
that work directly with [`tool_loop!`](@ref) (Chat Completions API).

Each tool's callable invokes `call_tool(session, name, args)` under the hood.

# Example
```julia
session = mcp_connect(`npx server`)
tools = mcp_tools(session)
chat = Chat(model="gpt-5.5", tools=map(t -> t.tool, tools))
push!(chat, Message(Val(:user), "Do something"))
result = tool_loop!(chat; tools)
```
"""
function mcp_tools(session::MCPSession)::Vector{CallableTool{GPTTool}}
    map(session.tools) do info
        schema = GPTTool(func=GPTFunctionSignature(
            name=info.name,
            description=info.description,
            parameters=info.input_schema
        ))
        # Capture session and info.name in closure
        sref = session
        tname = info.name
        callable = (_::String, args::Dict{String,Any}) -> _mcp_tool_dispatch(call_tool(sref, tname, args))
        CallableTool(schema, callable)
    end
end

"""
    mcp_tools_respond(session::MCPSession) -> Vector{CallableTool{FunctionTool}}

Convert all tools from an MCP session into `CallableTool{FunctionTool}` instances
that work directly with [`tool_loop`](@ref) (Responses API).

# Example
```julia
session = mcp_connect("https://mcp.example.com/mcp")
tools = mcp_tools_respond(session)
result = tool_loop("Do something"; tools=tools)
```
"""
function mcp_tools_respond(session::MCPSession)::Vector{CallableTool{FunctionTool}}
    map(session.tools) do info
        schema = FunctionTool(
            name=info.name,
            description=info.description,
            parameters=info.input_schema
        )
        sref = session
        tname = info.name
        callable = (_::String, args::Dict{String,Any}) -> _mcp_tool_dispatch(call_tool(sref, tname, args))
        CallableTool(schema, callable)
    end
end

# Extend to_tool protocol
to_tool(info::MCPToolInfo) = GPTTool(func=GPTFunctionSignature(
    name=info.name, description=info.description, parameters=info.input_schema
))
