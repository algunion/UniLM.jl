# Unit tests for RequestConfig and its resolution channels.

@testset "constructor rejects NaN and non-positive; Inf disables" begin
    float_fields = (:connect_timeout, :request_timeout, :stream_idle_timeout,
                    :total_deadline, :mcp_connect_timeout, :mcp_request_timeout)
    for f in float_fields, bad in (NaN, 0.0, -1.0)
        @test_throws ArgumentError RequestConfig(; (f => bad,)...)
    end
    for f in float_fields
        cfg = RequestConfig(; (f => Inf,)...)
        @test getfield(cfg, f) == Inf
    end
    @test_throws ArgumentError RequestConfig(max_attempts=0)
    @test_throws ArgumentError RequestConfig(max_attempts=-3)
    @test RequestConfig(max_attempts=1).max_attempts == 1
    # defaults are themselves valid and match the documented values
    d = RequestConfig()
    @test d.connect_timeout == 10.0
    @test d.request_timeout == 600.0
    @test d.stream_idle_timeout == 120.0
    @test d.total_deadline == 900.0
    @test d.max_attempts == 3
    @test d.mcp_connect_timeout == 120.0
    @test d.mcp_request_timeout == 120.0
end

@testset "copy-with-overrides revalidates and touches only named fields" begin
    base = RequestConfig()
    c = RequestConfig(base; request_timeout=5.0, max_attempts=7)
    @test c.request_timeout == 5.0
    @test c.max_attempts == 7
    for f in fieldnames(RequestConfig)
        f in (:request_timeout, :max_attempts) && continue
        @test getfield(c, f) == getfield(base, f)
    end
    @test_throws ArgumentError RequestConfig(base; connect_timeout=NaN)
    @test_throws ArgumentError RequestConfig(base; total_deadline=0.0)
    @test_throws ArgumentError RequestConfig(base; max_attempts=0)
    @test_throws MethodError RequestConfig(base; not_a_field=1.0)
end

@testset "channel precedence: kwarg > scope > process default" begin
    initial = UniLM.current_config()   # outside any scope = the process default
    try
        set_default_config!(request_timeout=111.0)
        @test current_config().request_timeout == 111.0
        with_request_config(request_timeout=222.0) do
            @test current_config().request_timeout == 222.0
            # channel 1: an explicit per-call config beats the active scope
            @test UniLM._resolve_config(RequestConfig(request_timeout=333.0)).request_timeout == 333.0
            # nothing falls through to the scope
            @test UniLM._resolve_config(nothing).request_timeout == 222.0
        end
        # scope exited → process default governs again
        @test current_config().request_timeout == 111.0
        @test UniLM._resolve_config(nothing).request_timeout == 111.0
        # with_request_config returns f's value
        @test with_request_config(() -> 42) == 42
        # the scope pins a COMPLETE struct merged at entry: unmentioned fields
        # come from the entry-time ambient config
        with_request_config(max_attempts=9) do
            @test current_config().request_timeout == 111.0
            @test current_config().max_attempts == 9
        end
    finally
        set_default_config!(initial)
    end
end

@testset "set_default_config! merges over the process default, not the scope" begin
    initial = UniLM.current_config()
    try
        set_default_config!(connect_timeout=5.0)
        with_request_config(connect_timeout=77.0) do
            newdef = set_default_config!(max_attempts=9)
            # kwargs merged over the PROCESS default (5.0), not the scope (77.0)
            @test newdef.connect_timeout == 5.0
            @test newdef.max_attempts == 9
            # the active scope still governs ambient resolution
            @test current_config().connect_timeout == 77.0
        end
        @test current_config().connect_timeout == 5.0
        @test current_config().max_attempts == 9
        # whole-struct form returns and installs exactly the given struct
        cfg = RequestConfig(request_timeout=42.0)
        @test set_default_config!(cfg) == cfg
        @test current_config() == cfg
        # validation applies through the merge form too
        @test_throws ArgumentError set_default_config!(request_timeout=NaN)
    finally
        set_default_config!(initial)
    end
end

# A timeout value that parks its own conversion. `_validated_timeout` calls
# `Float64` on every field, and that happens strictly between the process
# default's read and its write — so this pins one kwarg call inside its
# read-modify-write window with no sleeps and no polling.
mutable struct GatedSeconds <: Real
    v::Float64
    entered::Base.Event
    release::Base.Event
end
GatedSeconds(v) = GatedSeconds(v, Base.Event(), Base.Event())
Base.Float64(g::GatedSeconds) = (notify(g.entered); wait(g.release); g.v)

@testset "kwarg set_default_config! composes and never loses a concurrent field" begin
    initial = UniLM.current_config()
    try
        # Sequential composition: call 2 must not clobber call 1's field.
        set_default_config!(RequestConfig())
        set_default_config!(connect_timeout=11.0)
        set_default_config!(request_timeout=22.0)
        @test current_config().connect_timeout == 11.0
        @test current_config().request_timeout == 22.0

        # Deterministic lost-update: one call is held inside its merge window while
        # a second completes end to end. A plain read-modify-write then writes a
        # snapshot that predates the second call and silently drops its field.
        set_default_config!(RequestConfig())
        gate = GatedSeconds(11.0)
        held = Threads.@spawn set_default_config!(connect_timeout=gate)
        wait(gate.entered)                      # the held call has read, not yet written
        set_default_config!(request_timeout=22.0)
        notify(gate.release)                    # let it finish its write
        wait(held)
        @test current_config().connect_timeout == 11.0
        @test current_config().request_timeout == 22.0

        # Bounded concurrent hammer: each task owns one field and only ever raises
        # it, so every field must be non-decreasing over time. A decrease is a
        # write built from a stale snapshot.
        fields = (:connect_timeout, :request_timeout, :stream_idle_timeout, :mcp_request_timeout)
        set_default_config!(RequestConfig(; (f => 1.0 for f in fields)...))
        regressions = Threads.Atomic{Int}(0)
        @sync for i in eachindex(fields)
            Threads.@spawn begin
                high = fill(1.0, length(fields))     # task-local high-water marks
                for k in 2:2000
                    set_default_config!(; (fields[i] => Float64(k),)...)
                    cur = current_config()
                    for j in eachindex(fields)
                        v = getfield(cur, fields[j])
                        v < high[j] && Threads.atomic_add!(regressions, 1)
                        high[j] = max(high[j], v)
                    end
                end
            end
        end
        @test regressions[] == 0
        @test all(getfield(current_config(), f) == 2000.0 for f in fields)
    finally
        set_default_config!(initial)
    end
end

@testset "scope propagates into spawned tasks and is immune to default mutation" begin
    initial = UniLM.current_config()
    try
        with_request_config(request_timeout=222.0) do
            t = Threads.@spawn begin
                sleep(0.2)   # let the process default mutate before reading
                UniLM.current_config().request_timeout
            end
            set_default_config!(request_timeout=999.0)
            @test fetch(t) == 222.0                            # task carries the scope
            @test current_config().request_timeout == 222.0    # scope pinned at entry
        end
        @test current_config().request_timeout == 999.0
    finally
        set_default_config!(initial)
    end
end
