# ============================================================================
# Hang matrix — bounded-timeout falsification suite. Every operation in UniLM
# (provider request, stream, MCP exchange) must complete or fail with a TYPED
# error within a configurable, bounded time. Each testset states the FIXED
# contract wrapped in @test_broken; a timeout fix flips @test_broken → @test.
#
# BOUNDED OBSERVATION: a test that would hang is not a test. Every pin either
# fails fast on a not-yet-implemented symbol/kwarg, or runs the operation in a
# task observed with a short `timedwait` bound. Fully offline (zero-spend):
# all servers are 127.0.0.1 and no provider key is read.
# ============================================================================

using Sockets

# Run `f()` on a task; observe it with a short wall-clock bound so a hang can
# never block the suite. Returns:
#   (:ok, value)       — finished within `bound`; `fetch` returned `value`
#   (:threw, exc)      — finished within `bound`; the task threw `exc`
#   (:timeout, nothing)— did not finish within `bound`
# The guarded `fetch` is what lets a task that threw a not-yet-defined symbol
# (UndefVarError/MethodError on the base) surface as (:threw, …) → the pin's
# ok-expression stays `false` → Broken, never Error.
function _hm_bounded(f; bound::Float64 = 10.0)
    t = Threads.@spawn f()
    if timedwait(() -> istaskdone(t), bound) != :ok
        return (:timeout, nothing)
    end
    try
        (:ok, fetch(t))
    catch e
        (:threw, e)
    end
end

# ─── RequestConfig channels & validation (server-free unit contracts) ────────

@testset "config: constructor rejects NaN and non-positive; Inf disables" begin
    # FIXED contract: the inner constructor rejects NaN and ≤ 0 on every Float64
    # field with ArgumentError (NaN checked explicitly — `NaN < 0` is false, a
    # silently-reintroduced infinite hang), rejects max_attempts < 1, and ACCEPTS
    # Inf (Inf = "disabled", not an error).
    _rejects(f) = try
        f()
        false
    catch e
        e isa ArgumentError
    end
    ok = try
        _rejects(() -> UniLM.RequestConfig(connect_timeout = NaN)) &&
        _rejects(() -> UniLM.RequestConfig(request_timeout = 0.0)) &&
        _rejects(() -> UniLM.RequestConfig(total_deadline = -1.0)) &&
        _rejects(() -> UniLM.RequestConfig(mcp_request_timeout = NaN)) &&
        _rejects(() -> UniLM.RequestConfig(max_attempts = 0)) &&
        (UniLM.RequestConfig(request_timeout = Inf) isa UniLM.RequestConfig) &&
        (UniLM.RequestConfig(stream_idle_timeout = Inf) isa UniLM.RequestConfig)
    catch
        false
    end
    @test_broken ok
end

@testset "config: channel precedence kwarg > scope > process default; streams capture at spawn" begin
    # FIXED contract: resolution precedence is per-call config > dynamic scope >
    # process default; and a config resolved INSIDE a scope is immune to a later
    # mutation of the process default (streams close over the resolved struct at
    # spawn time). `_resolve_config(nothing)` = current_config().
    ok = try
        # process default (lowest precedence)
        UniLM.set_default_config!(UniLM.RequestConfig(max_attempts = 7))
        default_seen = UniLM.current_config().max_attempts == 7
        # dynamic scope overrides the process default
        scoped_seen = UniLM.with_request_config(max_attempts = 3) do
            UniLM.current_config().max_attempts == 3
        end
        # per-call config overrides the scope
        kwarg_wins = UniLM.with_request_config(max_attempts = 3) do
            UniLM._resolve_config(UniLM.RequestConfig(max_attempts = 2)).max_attempts == 2
        end
        # capture-at-spawn: a value resolved in-scope is unaffected by a later
        # mutation of the process default
        captured = UniLM.with_request_config(max_attempts = 5) do
            snap = UniLM._resolve_config(nothing)
            UniLM.set_default_config!(UniLM.RequestConfig(max_attempts = 9))
            snap.max_attempts == 5
        end
        # The config machinery above satisfies precedence + snapshot on its own,
        # so this pin would pass the moment the core config workstream merges —
        # but its flip only lands with the requests migration that adds the
        # `config` kwarg to chatrequest! (the channel a running stream resolves
        # through). Gate on that kwarg so the pin cannot pass before its flip
        # owner ships. `hasmethod(f, Tuple{Chat}, (:config,))` is false while the
        # kwarg is absent and true once the migration adds it.
        hasmethod(chatrequest!, Tuple{Chat}, (:config,)) &&
            default_seen && scoped_seen && kwarg_wins && captured
    catch
        false
    finally
        # restore the process default so this unit does not leak into other tests
        try
            UniLM.set_default_config!(UniLM.RequestConfig())
        catch
        end
    end
    @test_broken ok
end
