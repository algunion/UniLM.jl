# Unit tests for RequestConfig and its resolution channels.

using HTTP, JSON

# The process default exactly as found — `nothing` while none has been set, which lets a
# System One backend's own default apply — put back by every testset that sets one, so
# the process is left as it was found.
_saved_default() = @atomic UniLM._PROCESS_DEFAULT_CONFIG.cfg
_restore_default!(saved) = (@atomic UniLM._PROCESS_DEFAULT_CONFIG.cfg = saved; nothing)

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

@testset "a finite bound too large for the timers is rejected; the ceiling itself is valid" begin
    # A finite bound becomes a native timer deadline (HTTP.jl nanoseconds overflow
    # above ~9.2e9 s, a Timer's milliseconds above ~1.8e16 s): a value past what the
    # timers hold must fail at construction, not when a guard is armed mid-call.
    float_fields = (:connect_timeout, :request_timeout, :stream_idle_timeout,
                    :total_deadline, :mcp_connect_timeout, :mcp_request_timeout)
    for f in float_fields, big in (1.0e9 + 1, 1e17, floatmax(Float64))
        @test_throws "use Inf" RequestConfig(; (f => big,)...)
    end
    for f in float_fields
        @test getfield(RequestConfig(; (f => 1.0e9,)...), f) == 1.0e9
    end
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
    initial = _saved_default()
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
        _restore_default!(initial)
    end
end

@testset "set_default_config! merges over the process default, not the scope" begin
    initial = _saved_default()
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
        _restore_default!(initial)
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
    initial = _saved_default()
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
        _restore_default!(initial)
    end
end

@testset "scope propagates into spawned tasks and is immune to default mutation" begin
    initial = _saved_default()
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
        _restore_default!(initial)
    end
end

# ─── A backend's own default (System One) ────────────────────────────────────

@testset "a backend's own default ranks below every channel the user sets, above the field defaults" begin
    backend = RequestConfig(request_timeout=60.0, total_deadline=150.0)
    percall = RequestConfig(request_timeout=7.0)
    initial = _saved_default()
    try
        _restore_default!(nothing)                               # no process default set
        @test current_config() == RequestConfig()                # what every other verb resolves: as before
        @test UniLM._resolve_config(nothing) == RequestConfig()
        @test UniLM._resolve_config(nothing, backend) == backend
        @test UniLM._resolve_config(nothing, nothing) == RequestConfig()
        @test UniLM._resolve_config(percall, backend) === percall
        with_request_config(max_attempts=1) do
            # A scope is a whole struct, merged over the field defaults: the backend's is replaced.
            @test UniLM._resolve_config(nothing, backend) == RequestConfig(max_attempts=1)
            @test UniLM._resolve_config(percall, backend) === percall
        end
        # A process default the user sets — even the field defaults — replaces the backend's.
        set_default_config!(RequestConfig())
        @test UniLM._resolve_config(nothing, backend) == RequestConfig()
        _restore_default!(nothing)
        merged = set_default_config!(request_timeout=5.0)        # merged over the field defaults
        @test merged == RequestConfig(request_timeout=5.0)
        @test UniLM._resolve_config(nothing, backend) == merged
        @test current_config() == merged
        with_request_config(request_timeout=3.0) do
            @test UniLM._resolve_config(nothing, backend) == RequestConfig(request_timeout=3.0)
        end
    finally
        _restore_default!(initial)
    end
end

# A System One backend whose protocol brings a default of 0.4 s per attempt, and an
# OpenAI-wire chat endpoint, both answered by one server after 1 s.
struct _CfgProbeProtocol <: UniLM.SystemOneProtocol end
struct _CfgProbe <: UniLM.ServiceEndpoint end
const _CFG_BASE = Ref("")
UniLM._resolve_base_url(::Type{_CfgProbe}) = _CFG_BASE[]
UniLM.auth_header(::Type{_CfgProbe}) = ["Authorization" => "Bearer c", "Content-Type" => "application/json"]
UniLM.provider_capabilities(::Type{_CfgProbe}) = Set([:system_one])
UniLM.system_one_protocol(::Type{_CfgProbe}) = _CfgProbeProtocol()
UniLM.encode_system_one(::_CfgProbeProtocol, s, r::SystemOneRequest) =
    UniLM.encode_system_one(UniLM.TypeSafeProtocol(), s, r)
UniLM.decode_system_one(::_CfgProbeProtocol, s, r::SystemOneRequest, resp::HTTP.Response) =
    UniLM.decode_system_one(UniLM.TypeSafeProtocol(), s, r, resp)
UniLM._system_one_default_config(::_CfgProbeProtocol, _) = RequestConfig(request_timeout=0.4, max_attempts=1)

_cfg_slow_server() = HTTP.serve!("127.0.0.1", 0; verbose=false) do req
    sleep(1.0)
    body = String(req.target) == "/v1/systemone" ?
        """{"model":"jev-1.13.0","answers":{"q":{"type":"noul","noul":0.5}},"usage":{"input_tokens":1,"output_tokens":0}}""" :
        """{"choices":[{"index":0,"finish_reason":"stop","message":{"role":"assistant","content":"ok"}}]}"""
    HTTP.Response(200, ["Content-Type" => "application/json"], Vector{UInt8}(body))
end

@testset "end to end: a System One backend's default governs its calls only, below the user's channels" begin
    server = _cfg_slow_server()
    _CFG_BASE[] = "http://127.0.0.1:$(HTTP.port(server))"
    limit(r) = r isa SystemOneCallError && r.cause isa UniLMTimeout ? r.cause.limit : nothing
    probe(; kw...) = ask("s", "q" => noul("x"); service=_CfgProbe, kw...)
    initial = _saved_default()
    try
        _restore_default!(nothing)
        @test limit(probe()) == 0.4                                    # the backend's own default
        # A chat to the same server, with no process default set either: the field defaults
        # bound it, so it is answered — a System One backend's default reaches no other verb.
        chat = Chat(service=GenericOpenAIEndpoint(_CFG_BASE[], ""), model="mock")
        push!(chat, Message(Val(:system), "s"))
        push!(chat, Message(Val(:user), "u"))
        @test chatrequest!(chat) isa LLMSuccess
        @test with_request_config(() -> limit(probe()); request_timeout=0.2, max_attempts=1) == 0.2
        @test limit(probe(config=RequestConfig(request_timeout=0.3, max_attempts=1))) == 0.3
        set_default_config!(RequestConfig(request_timeout=0.25, max_attempts=1))
        @test limit(probe()) == 0.25
        reset_default_config!()                                         # cleared: the backend's own again
        @test limit(probe()) == 0.4
        set_default_config!(RequestConfig())                            # the field defaults, set
        @test probe() isa SystemOneSuccess
    finally
        _restore_default!(initial)
        close(server)
    end
end

# ─── A model's own default (Gemini 4) ───────────────────────────────────────

@testset "a Gemini 4 call defaults to 20-minute bounds, below every channel the user sets" begin
    # Google: "Set your HTTP read timeout to 20 minutes" for Gemini 4 Argon, whose non-streaming
    # calls the server closes after 10 minutes and streams after 20.
    long = RequestConfig(request_timeout=1200.0, stream_idle_timeout=1200.0, total_deadline=1200.0)
    for svc in (GEMINIServiceEndpoint, GEMINIOpenAIServiceEndpoint), model in ("gemini-4-argon", "gemini-4-flash")
        @test UniLM._model_default_config(svc, model) == long
    end
    # A Gemini 3 model, a near-miss id or another provider's endpoint brings no default of its own.
    for (svc, model) in ((GEMINIServiceEndpoint, "gemini-3.8-flash"), (GEMINIServiceEndpoint, "gemini-40-flash"),
                         (GEMINIServiceEndpoint, "gemini-4"), (OPENAIServiceEndpoint, "gemini-4-argon"),
                         (ANTHROPICServiceEndpoint, "claude-opus-5-5"))
        @test isnothing(UniLM._model_default_config(svc, model))
    end
    initial = _saved_default()
    try
        _restore_default!(nothing)
        model_default = UniLM._model_default_config(GEMINIServiceEndpoint, "gemini-4-argon")
        @test UniLM._resolve_config(nothing, model_default) == long
        percall = RequestConfig(request_timeout=7.0)
        @test UniLM._resolve_config(percall, model_default) === percall
        # A scope or a process default the user sets replaces it as a whole, as for a System One backend.
        with_request_config(max_attempts=1) do
            @test UniLM._resolve_config(nothing, model_default) == RequestConfig(max_attempts=1)
        end
        set_default_config!(RequestConfig())
        @test UniLM._resolve_config(nothing, model_default) == RequestConfig()
    finally
        _restore_default!(initial)
    end
end

# An OpenAI-wire endpoint whose model "slow-thinker" brings a bound of its own, answered by the
# slow server above.
struct _ModelCfgProbe <: UniLM.OpenAIWireEndpoint end
UniLM.get_url(::Type{_ModelCfgProbe}, ::Chat) = _CFG_BASE[] * "/v1/chat/completions"
UniLM._agentic_url(::Type{_ModelCfgProbe}) = _CFG_BASE[] * "/v1/responses"
UniLM.auth_header(::Type{_ModelCfgProbe}) = ["Authorization" => "Bearer c", "Content-Type" => "application/json"]
UniLM._model_default_config(::Type{_ModelCfgProbe}, model::AbstractString) =
    model == "slow-thinker" ? RequestConfig(request_timeout=0.4, max_attempts=1) : nothing

@testset "end to end: chatrequest! and respond take a model's own default, below the user's channels" begin
    server = _cfg_slow_server()
    _CFG_BASE[] = "http://127.0.0.1:$(HTTP.port(server))"
    limit(r) = r isa Union{LLMCallError,ResponseCallError} && r.cause isa UniLMTimeout ? r.cause.limit : nothing
    chat(model; kw...) = Chat(; service=_ModelCfgProbe, model, messages=[Message(Val(:system), "s"), Message(Val(:user), "u")], kw...)
    asked(model) = Respond(service=_ModelCfgProbe, model=model, input="u")
    initial = _saved_default()
    try
        _restore_default!(nothing)
        @test limit(chatrequest!(chat("slow-thinker"))) == 0.4
        @test limit(fetch(chatrequest!(chat("slow-thinker"; stream=true)))) == 0.4
        @test limit(respond(asked("slow-thinker"))) == 0.4
        @test chatrequest!(chat("gpt-6-luna")) isa LLMSuccess            # no default of its own: answered
        @test limit(chatrequest!(chat("slow-thinker"); config=RequestConfig(request_timeout=0.3, max_attempts=1))) == 0.3
        @test with_request_config(() -> limit(respond(asked("slow-thinker"))); request_timeout=0.2, max_attempts=1) == 0.2
        set_default_config!(RequestConfig(request_timeout=0.25, max_attempts=1))
        @test limit(chatrequest!(chat("slow-thinker"))) == 0.25
        @test limit(respond(asked("slow-thinker"))) == 0.25
        set_default_config!(nothing)                                    # cleared: the model's own again
        @test limit(chatrequest!(chat("slow-thinker"))) == 0.4
        @test limit(respond(asked("slow-thinker"))) == 0.4
    finally
        _restore_default!(initial)
        close(server)
    end
end

# ─── Clearing the process default ───────────────────────────────────────────

@testset "after a clear, calls resolve exactly as if no default had ever been set" begin
    long = RequestConfig(request_timeout=1200.0, stream_idle_timeout=1200.0, total_deadline=1200.0)
    hosted = RequestConfig(request_timeout=60.0, total_deadline=150.0)
    backend = RequestConfig(request_timeout=0.4, max_attempts=1)
    set = RequestConfig(request_timeout=5.0, max_attempts=1)
    initial = _saved_default()
    try
        for clear! in (() -> set_default_config!(nothing), reset_default_config!)
            set_default_config!(set)
            @test UniLM._resolve_config(nothing, backend) === set              # set, it replaces a backend's own
            @test clear!() === nothing
            @test current_config() == RequestConfig()                          # outside every scope
            @test UniLM._resolve_config(nothing, backend) === backend
            # What the seams pass: Gemini 4's 20 minutes, a hosted System One service's 60 s and 150 s.
            for svc in (GEMINIServiceEndpoint, GEMINIOpenAIServiceEndpoint)
                @test UniLM._resolve_config(nothing, UniLM._model_default_config(svc, "gemini-4-argon")) == long
            end
            for (p, svc) in ((UniLM.TypeSafeProtocol(), TYPESAFEServiceEndpoint),
                             (UniLM.OpenAIDecisionsProtocol(), OPENAIServiceEndpoint))
                @test UniLM._resolve_config(nothing, UniLM._system_one_default_config(p, svc)) == hosted
            end
            # The keyword form merges over the field defaults, not over the default cleared.
            @test set_default_config!(; total_deadline=120.0) == RequestConfig(total_deadline=120.0)
            @test current_config() == RequestConfig(total_deadline=120.0)
            clear!(); clear!()                                                 # clearing none leaves none
            @test current_config() == RequestConfig() && UniLM._resolve_config(nothing, backend) === backend
        end
    finally
        _restore_default!(initial)
    end
end

@testset "a clear racing the keyword form's compare-and-swap loses no write" begin
    # Property: a keyword set never installs a merge of what it read before a clear that landed
    # first. Its compare-and-swap misses, and it merges again over what the clear left, the
    # field defaults. So once every task is done and a final keyword set has run, the default
    # holds the tasks' fields and that set's over the field defaults: no task's field is lost,
    # and no value set before the clear survives it (`stale` sets fields no task sets).
    fields = (:request_timeout, :stream_idle_timeout, :mcp_connect_timeout, :mcp_request_timeout)
    stale = RequestConfig(connect_timeout=7.0, max_attempts=9)
    initial = _saved_default()
    try
        # Each task is held inside its merge window, having read `stale`, while the clear lands;
        # released, the tasks race one another's compare-and-swap.
        for clear! in (() -> set_default_config!(nothing), reset_default_config!)
            set_default_config!(stale)
            gates = [GatedSeconds(10.0i) for i in eachindex(fields)]
            held = [Threads.@spawn set_default_config!(; (fields[i] => gates[i],)...) for i in eachindex(fields)]
            foreach(g -> wait(g.entered), gates)
            clear!()
            foreach(g -> notify(g.release), gates)
            foreach(wait, held)
            set_default_config!(; total_deadline=120.0)
            @test current_config() == RequestConfig(; (fields[i] => 10.0i for i in eachindex(fields))..., total_deadline=120.0)
        end
    finally
        _restore_default!(initial)
    end
end
