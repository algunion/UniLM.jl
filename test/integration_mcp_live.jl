# ─── Live MCP Integration Tests (real third-party stdio servers) ──────────────
# Witnesses UniLM's MCP timeout/teardown machinery against REAL Model Context
# Protocol servers from the reference ecosystem, launched locally via `npx`:
#   - @modelcontextprotocol/server-filesystem  (a scratch dir you create + clean)
#   - @modelcontextprotocol/server-everything   (schema-variety probe)
#
# These run third-party server code (node), spawn/kill real OS processes, and are
# therefore OPT-IN: they execute only when UNILM_LIVE_MCP=1 AND `npx` is on PATH.
# In CI and default runs the file skips loudly and records nothing.
#
# Discipline: bounded observation everywhere (a test that would hang is not a
# test — every hangable op runs on a task watched by a wall-clock `timedwait`),
# every server process group is reaped on every path (finally), and each server
# generation is identified by a narrow marker (its scratch-dir path in argv).

if !(get(ENV, "UNILM_LIVE_MCP", "") == "1" && Sys.which("npx") !== nothing)
    @info "Skipping live MCP integration tests (set UNILM_LIVE_MCP=1 and put npx on PATH to enable)"
else

# ─── Signals + process helpers ───────────────────────────────────────────────
# Raw POSIX signals by number. SIGKILL is 9 everywhere; SIGSTOP/SIGCONT differ
# between Darwin (17/19) and Linux (19/18). Group signalling uses a negative pid.
const _LIVE_SIGKILL = 9
const _LIVE_SIGSTOP = Sys.isapple() ? 17 : 19
const _LIVE_SIGCONT = Sys.isapple() ? 19 : 18
_live_signal(pid::Integer, s::Integer) = ccall(:kill, Cint, (Cint, Cint), pid % Cint, s % Cint)

# Count live processes whose argv contains `marker`. A server generation's scratch
# directory appears in both the `npm exec` wrapper and its `node` child argv, so
# the scratch-dir basename is a unique, narrow marker for that generation.
_live_survivors(marker::AbstractString)::Int = try
    out = read(pipeline(`pgrep -f $marker`; stderr = devnull), String)
    count(!isempty, split(strip(out), '\n'))
catch
    0
end

# Bounded observation: run `f()` on a task, observe with a wall-clock bound so a
# hang can never block the suite. Returns (:ok, value) | (:threw, exc) | (:timeout, nothing).
function _live_bounded(f; bound::Float64 = 25.0)
    t = Threads.@spawn f()
    timedwait(() -> istaskdone(t), bound) === :ok || return (:timeout, nothing)
    try
        (:ok, fetch(t))
    catch e
        (:threw, e)
    end
end

# The shape of an abrupt server death observed by the client: the next stdio
# exchange fails on a broken pipe (write leg) or a closed stream (read leg). Not
# a typed timeout — a crash is a transport failure, distinct from a hang.
_live_transport_death(e)::Bool =
    e isa Base.IOError || e isa EOFError ||
    (e isa ErrorException && (occursin("closed connection", e.msg) ||
                              occursin("broken pipe", lowercase(e.msg)) ||
                              occursin("EPIPE", e.msg)))

# Best-effort reap of a server generation: group-SIGKILL by the spawn-captured
# pgid (reaches the node grandchild even after the wrapper is gone), then pkill by
# marker as a backstop. A stopped process dies to SIGKILL, so no SIGCONT is needed.
function _live_reap(marker::AbstractString; pgid = nothing)
    pgid === nothing || (try; _live_signal(-pgid, _LIVE_SIGKILL); catch; end)
    try; run(pipeline(`pkill -9 -f $marker`; stderr = devnull)); catch; end
    nothing
end

_fs_cmd(dir::AbstractString) = `npx -y @modelcontextprotocol/server-filesystem $dir`

# Pre-warm a package so its first-run download never pollutes a timed observation.
# Spawn the server, close its stdin (a compliant MCP stdio server exits on EOF),
# wait bounded for exit, force-kill any straggler. Not timed/asserted.
function _live_prewarm(cmd::Cmd)
    try
        p = open(cmd, "r+")
        try; close(p.in); catch; end
        timedwait(() -> !process_running(p), 90.0)
        try; kill(p); catch; end
        try; close(p); catch; end
    catch e
        @warn "live MCP pre-warm failed (first timed test may absorb a download)" exception = e
    end
    nothing
end

let warm = mktempdir(; prefix = "mcplive_warm_")
    try
        _live_prewarm(_fs_cmd(warm))
        _live_prewarm(`npx -y @modelcontextprotocol/server-everything`)
    finally
        rm(warm; recursive = true, force = true)
    end
end

# ─── 1. Happy path: real handshake, tool round-trip, clean teardown ──────────
@testset "live filesystem MCP: handshake, write/read round-trip, teardown" begin
    scratch = mktempdir(; prefix = "mcplive_happy_")
    marker = basename(scratch)
    session = nothing
    try
        t0 = time()
        session = mcp_connect(_fs_cmd(scratch))
        handshake_s = time() - t0
        @test session.status === :ready

        tools = list_tools!(session)
        @test !isempty(tools)
        @test any(t -> t.name == "write_file", tools)
        @test any(t -> t.name == "read_file", tools)

        path = joinpath(scratch, "hello.txt")
        wt0 = time()
        w = call_tool(session, "write_file",
            Dict{String,Any}("path" => path, "content" => "hi from live mcp"))
        write_s = time() - wt0
        @test w isa MCPToolResult
        @test w.is_error === false

        rt0 = time()
        r = call_tool(session, "read_file", Dict{String,Any}("path" => path))
        read_s = time() - rt0
        @test r isa MCPToolResult
        @test r.is_error === false
        @test r.content == "hi from live mcp"

        # Real-latency observations vs the 120 s connect / 120 s per-request defaults.
        println("  [live-mcp] handshake:      ", round(handshake_s; digits = 3),
                " s (mcp_connect_timeout default 120 s)")
        println("  [live-mcp] write_file RTT: ", round(write_s; digits = 3),
                " s (mcp_request_timeout default 120 s)")
        println("  [live-mcp] read_file RTT:  ", round(read_s; digits = 3), " s")

        dt0 = time()
        mcp_disconnect!(session)
        disconnect_s = time() - dt0
        @test session.status === :closed
        println("  [live-mcp] disconnect:     ", round(disconnect_s; digits = 3), " s")

        @test timedwait(() -> _live_survivors(marker) == 0, 8.0) === :ok
        @test _live_survivors(marker) == 0
    finally
        _live_reap(marker; pgid = session === nothing ? nothing : session.transport.pgid)
        rm(scratch; recursive = true, force = true)
    end
end

# ─── 2. Killing the transport handle does NOT kill the server ────────────────
# `session.transport.process` is the `npm exec` WRAPPER, not the server: it forks
# a `node` grandchild that inherits the stdio pipes and is the real MCP server.
# Killing only the wrapper orphans the grandchild, which keeps serving — this is
# exactly why teardown spawns detached and group-kills by the captured pgid.
@testset "live MCP: killing the wrapper leaves the real server (node grandchild) serving" begin
    scratch = mktempdir(; prefix = "mcplive_wrap_")
    marker = basename(scratch)
    session = nothing
    try
        session = mcp_connect(_fs_cmd(scratch); auto_respawn = false)
        pgid = session.transport.pgid
        # Warm-up call succeeds against the freshly spawned server.
        warm = _live_bounded(bound = 12.0) do
            call_tool(session, "list_allowed_directories", Dict{String,Any}())
        end
        @test warm[1] === :ok

        before = _live_survivors(marker)
        @test before >= 2                       # wrapper + node child both carry the marker
        _live_signal(getpid(session.transport.process), _LIVE_SIGKILL)  # kill the wrapper only
        sleep(0.6)
        survived = _live_survivors(marker)
        @test survived >= 1                      # the node grandchild is still alive

        # The orphaned real server still answers over the inherited pipes.
        after = _live_bounded(bound = 12.0) do
            try
                (:returned, call_tool(session, "list_allowed_directories", Dict{String,Any}()))
            catch e
                (:threw, e)
            end
        end
        @test after[1] === :ok
        if after[1] === :ok
            @test after[2][1] === :returned
            after[2][1] === :returned && @test after[2][2].is_error === false
        end
        println("  [live-mcp] wrapper killed; marker procs ", before, " -> ", survived,
                "; orphaned node still served the next call")
    finally
        _live_reap(marker; pgid = session === nothing ? nothing : session.transport.pgid)
        rm(scratch; recursive = true, force = true)
    end
end

# ─── 3. Abrupt server death (kill -9 the group) surfaces a raw transport error ─
# FINDING pin: the session-fatal + auto_respawn machinery is keyed on a request
# TIMEOUT (the watchdog nulling the transport handles), NOT on abrupt death. When
# the real server is killed, the next exchange fails with a raw broken-pipe /
# closed-stream IOError; the session is neither respawned nor marked closed-by-
# timeout — even with auto_respawn=true. auto_respawn covers hangs, not crashes.
@testset "live MCP: killing the server group surfaces a raw transport error, no respawn" begin
    scratch = mktempdir(; prefix = "mcplive_kill_")
    marker = basename(scratch)
    session = nothing
    try
        # auto_respawn=true is the STRONGER setting: if respawn does not fire here,
        # it certainly does not with the default auto_respawn=false either.
        session = mcp_connect(_fs_cmd(scratch); auto_respawn = true)
        pgid = session.transport.pgid
        warm = _live_bounded(bound = 12.0) do
            call_tool(session, "list_allowed_directories", Dict{String,Any}())
        end
        @test warm[1] === :ok

        _live_signal(-pgid, _LIVE_SIGKILL)       # kill the whole server group
        @test timedwait(() -> _live_survivors(marker) == 0, 8.0) === :ok

        outcome = _live_bounded(bound = 12.0) do
            try
                (:returned, call_tool(session, "list_allowed_directories", Dict{String,Any}()))
            catch e
                (:threw, e)
            end
        end
        @test outcome[1] === :ok
        if outcome[1] === :ok
            @test outcome[2][1] === :threw       # NOT respawned-and-returned
            if outcome[2][1] === :threw
                e = outcome[2][2]
                @test _live_transport_death(e)
                @test !(e isa MCPTimeoutError)
                @test !occursin("auto_respawn", sprint(showerror, e))
                println("  [live-mcp] post-crash error: ", sprint(showerror, e))
            end
        end
        # The crash left the session neither closed-by-timeout nor respawned.
        @test session._close_cause === :none
    finally
        _live_reap(marker; pgid = session === nothing ? nothing : session.transport.pgid)
        rm(scratch; recursive = true, force = true)
    end
end

# ─── 4. Real hang via SIGSTOP → whole-exchange timeout + group-kill of a stopped process ─
# A SIGSTOP'd server is frozen: it never answers and never reacts to stdin EOF or
# SIGTERM. The whole-exchange watchdog must still terminate the call with a typed
# MCPTimeoutError and the teardown ladder's UNCONDITIONAL final rung (group SIGKILL)
# must reap the stopped process. The read only unblocks once that SIGKILL lands, so
# the user-visible latency is timeout + grace_term(5 s) + grace_kill(2 s): bounded,
# but materially larger than the timeout alone for a truly frozen server.
@testset "live MCP: SIGSTOP hang yields MCPTimeoutError and the stopped server is SIGKILL-reaped" begin
    scratch = mktempdir(; prefix = "mcplive_stop_")
    marker = basename(scratch)
    session = nothing
    pgid = nothing
    try
        session = mcp_connect(_fs_cmd(scratch); auto_respawn = false)
        pgid = session.transport.pgid
        warm = _live_bounded(bound = 12.0) do
            call_tool(session, "list_allowed_directories", Dict{String,Any}())
        end
        @test warm[1] === :ok

        _live_signal(-pgid, _LIVE_SIGSTOP)       # freeze the whole server group

        # Per-call timeout override (2 s); the observation bound (25 s) leaves ample
        # room for the kill ladder so a genuine hang would still surface as :timeout.
        outcome = _live_bounded(bound = 25.0) do
            c0 = time()
            err = try
                call_tool(session, "list_allowed_directories", Dict{String,Any}(); timeout = 2.0)
                nothing
            catch e
                e
            end
            (err, time() - c0)
        end
        @test outcome[1] === :ok
        if outcome[1] === :ok
            err, wall = outcome[2]
            @test err isa MCPTimeoutError
            if err isa MCPTimeoutError
                @test err.phase === :request
                @test err.limit == 2.0            # the call-time override was applied
            end
            # Bounded: at least the timeout, at most timeout + the fixed ladder graces
            # (5 s + 2 s) plus scheduler slack. Proves termination, not an infinite wait.
            @test wall >= 2.0
            @test wall <= 13.0
            println("  [live-mcp] SIGSTOP hang: MCPTimeoutError after ",
                    round(wall; digits = 2), " s wall (limit 2.0 s + kill-ladder graces)")
        end
        @test session.status === :closed
        @test session._close_cause === :timeout

        # The final unconditional SIGKILL rung reaps even a stopped process.
        @test timedwait(() -> _live_survivors(marker) == 0, 8.0) === :ok
        @test _live_survivors(marker) == 0

        # A stdio timeout is session-fatal: the next call names auto_respawn (default off).
        nexterr = try
            call_tool(session, "list_allowed_directories", Dict{String,Any}())
            nothing
        catch e
            e
        end
        @test nexterr isa Exception
        @test nexterr !== nothing && occursin("auto_respawn", sprint(showerror, nexterr))
    finally
        pgid === nothing || (try; _live_signal(-pgid, _LIVE_SIGCONT); catch; end)
        _live_reap(marker; pgid = pgid)
        rm(scratch; recursive = true, force = true)
    end
end

# ─── 5. SIGSTOP timeout + auto_respawn=true → the real server is respawned ────
# With auto_respawn=true a stdio session closed by a request timeout respawns the
# server (same command, fresh handshake) on the NEXT call. The timed-out call
# still throws; the following call transparently respawns the REAL server, emits
# the documented @warn, and succeeds against the fresh process.
@testset "live MCP: auto_respawn respawns the real server after a timeout" begin
    scratch = mktempdir(; prefix = "mcplive_respawn_")
    marker = basename(scratch)
    session = nothing
    pgid = nothing
    try
        session = mcp_connect(_fs_cmd(scratch); auto_respawn = true)
        pgid = session.transport.pgid
        warm = _live_bounded(bound = 12.0) do
            call_tool(session, "list_allowed_directories", Dict{String,Any}())
        end
        @test warm[1] === :ok

        _live_signal(-pgid, _LIVE_SIGSTOP)

        # Call 1: hits the frozen server, times out, closes the session.
        first_outcome = _live_bounded(bound = 25.0) do
            try
                (:returned, call_tool(session, "list_allowed_directories",
                                      Dict{String,Any}(); timeout = 2.0))
            catch e
                (:threw, e)
            end
        end
        @test first_outcome[1] === :ok
        if first_outcome[1] === :ok
            @test first_outcome[2][1] === :threw
            first_outcome[2][1] === :threw && @test first_outcome[2][2] isa MCPTimeoutError
        end
        @test session.status === :closed

        # Call 2: transparently respawns the real server and succeeds. Capture logs
        # to confirm the respawn @warn fired.
        logbuf = IOBuffer()
        second_outcome = Base.CoreLogging.with_logger(Base.CoreLogging.SimpleLogger(logbuf)) do
            _live_bounded(bound = 25.0) do
                try
                    (:returned, call_tool(session, "list_allowed_directories", Dict{String,Any}()))
                catch e
                    (:threw, e)
                end
            end
        end
        logs = String(take!(logbuf))
        @test second_outcome[1] === :ok
        if second_outcome[1] === :ok
            @test second_outcome[2][1] === :returned
            if second_outcome[2][1] === :returned
                @test second_outcome[2][2] isa MCPToolResult
                @test second_outcome[2][2].is_error === false
            end
        end
        @test session.status === :ready
        @test occursin("respawning the server", logs)
        println("  [live-mcp] auto_respawn: server respawned after timeout; next call succeeded")
    finally
        # Both generations share the same command (scratch-dir marker); reap by marker.
        pgid === nothing || (try; _live_signal(-pgid, _LIVE_SIGCONT); catch; end)
        if session !== nothing
            try; mcp_disconnect!(session); catch; end
            try; _live_signal(-session.transport.pgid, _LIVE_SIGKILL); catch; end
        end
        _live_reap(marker; pgid = pgid)
        rm(scratch; recursive = true, force = true)
    end
end

# ─── 6. stdin-EOF politeness: a real server exits on the first teardown rung ──
# The teardown ladder closes stdin first; MCP spec says a compliant server exits
# on stdin EOF. Measure how long the real server takes to exit, and validate the
# shipped grace_term=5 s first rung against reality: a disconnect that returns in
# well under 5 s means the server exited on stdin EOF (before any SIGTERM). If a
# real server ever needed SIGTERM this assertion would fail and surface it.
@testset "live MCP: real server exits on stdin EOF within the grace_term first rung" begin
    scratch = mktempdir(; prefix = "mcplive_eof_")
    marker = basename(scratch)
    session = nothing
    try
        session = mcp_connect(_fs_cmd(scratch))
        @test session.status === :ready
        @test _live_survivors(marker) >= 1

        d0 = time()
        mcp_disconnect!(session)                 # closes stdin, then the escalation ladder
        exit_s = time() - d0
        @test session.status === :closed
        # grace_term (first rung) is 5 s; a clean stdin-EOF exit returns far under it.
        @test exit_s < 5.0
        @test timedwait(() -> _live_survivors(marker) == 0, 6.0) === :ok
        @test _live_survivors(marker) == 0
        println("  [live-mcp] stdin-EOF exit: ", round(exit_s; digits = 3),
                " s (reaped by the stdin-EOF first rung; grace_term 5 s)")
    finally
        _live_reap(marker; pgid = session === nothing ? nothing : session.transport.pgid)
        rm(scratch; recursive = true, force = true)
    end
end

# ─── 7. Schema-variety probe against the everything server ───────────────────
# The everything server advertises tools with richer JSON-Schema inputs (nested
# $schema/properties/required). Assert they decode into the session's tool
# registry without error and that a real tool call round-trips.
@testset "live MCP: everything server schemas decode and a tool call round-trips" begin
    marker = "mcp-server-everything"
    session = nothing
    try
        session = mcp_connect(`npx -y @modelcontextprotocol/server-everything`)
        @test session.status === :ready
        tools = list_tools!(session)
        @test !isempty(tools)
        # Every advertised tool decoded into a typed MCPToolInfo with an object schema.
        @test all(t -> t isa UniLM.MCPToolInfo, tools)
        @test all(t -> t.input_schema === nothing || t.input_schema isa Dict{String,Any}, tools)
        @test any(t -> t.name == "echo", tools)

        r = call_tool(session, "echo", Dict{String,Any}("message" => "live probe"))
        @test r isa MCPToolResult
        @test r.is_error === false
        @test occursin("live probe", r.content)
        println("  [live-mcp] everything server: ", length(tools),
                " tools decoded; echo round-trip ok")

        mcp_disconnect!(session)
        @test session.status === :closed
    finally
        if session !== nothing
            try; mcp_disconnect!(session); catch; end
        end
        try; run(pipeline(`pkill -9 -f $marker`; stderr = devnull)); catch; end
    end
end

end  # if UNILM_LIVE_MCP
