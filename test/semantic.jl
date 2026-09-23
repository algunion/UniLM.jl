# test/semantic.jl — natural-language control flow contracts, zero-spend.
#
# Everything here runs against a local mock that answers Choice questions from a
# lookup table, so no key and no network egress are involved. The mock is the
# oracle for the wire: what @branch and nl_dispatch put on it (question names,
# criteria order, state, model) is asserted on the recorded request, and what
# they do with an answer is asserted on the returned value.
#
# The dispatch fixtures are defined at file scope on purpose: `include` runs
# them at module level, which is what makes them real methods that `methods(f)`
# can see, and their source order is the option order under test.

using Sockets

# ─── Mock endpoint ───────────────────────────────────────────────────────────

struct SemanticMock <: UniLM.ServiceEndpoint end

const _semantic_base = Ref("")

UniLM._resolve_base_url(::Type{SemanticMock}) = _semantic_base[]
UniLM.auth_header(::Type{SemanticMock}) =
    ["Authorization" => "Bearer t", "Content-Type" => "application/json"]
UniLM.provider_capabilities(::Type{SemanticMock}) = Set([:system_one, :models])

# One attempt is enough for every status served here, and a short deadline keeps
# a bound on the suite if a bind ever goes wrong.
const _SEM_CFG = UniLM.RequestConfig(max_attempts=1, total_deadline=30.0)

"""
Run `f()` against a local System One mock and return `(f(), recorded_requests)`.

Every Choice question is answered with `pick[question_name]` when the table has
an entry and with the question's FIRST criteria key otherwise, at probability
1.0 against 0.0 for the rest. Request bodies are recorded parsed with
`JSON.parse`, which keeps object key order, so criteria order is observable.
A `status` other than 200 serves `body` verbatim instead.
"""
function _with_semantic_mock(f::Function; pick::AbstractDict=Dict{String,String}(),
                             confidence::Real=0.9, status::Int=200,
                             body::Union{Nothing,AbstractString}=nothing)
    recorded = Dict{String,Any}[]
    # A request with no body at all arrives as a sentinel that supports no byte access.
    bodytext(req) = applicable(copy, req.body) ? String(copy(req.body)) : ""
    handler = function (req)
        raw = bodytext(req)
        parsed = isempty(raw) ? JSON.Object{String,Any}() : JSON.parse(raw)
        push!(recorded, Dict{String,Any}(
            "method" => String(req.method),
            "target" => String(req.target),
            "body" => parsed,
            "headers" => Dict{String,String}(
                lowercase(String(k)) => String(v) for (k, v) in req.headers)))
        status == 200 || return HTTP.Response(status, ["Content-Type" => "application/json"],
                                              Vector{UInt8}(isnothing(body) ? "" : body))
        answered = JSON.Object{String,Any}()
        for (name, q) in get(parsed, "questions", JSON.Object{String,Any}())
            (q isa AbstractDict && get(q, "type", "") == "choice") || continue
            options = collect(keys(q["criteria"]))
            selected = get(pick, name, first(options))
            probabilities = JSON.Object{String,Any}()
            for o in options
                probabilities[o] = o == selected ? 1.0 : 0.0
            end
            a = JSON.Object{String,Any}()
            a["type"] = "choice"
            a["choice"] = selected
            a["confidence"] = Float64(confidence)
            a["probabilities"] = probabilities
            answered[name] = a
        end
        usage = JSON.Object{String,Any}()
        usage["input_tokens"] = 10
        usage["output_tokens"] = 1
        payload = JSON.Object{String,Any}()
        payload["model"] = "jev-1.13.0"
        payload["answers"] = answered
        payload["usage"] = usage
        HTTP.Response(200, ["Content-Type" => "application/json",
                            "x-typesafe-request-id" => "req_mock"],
                      Vector{UInt8}(JSON.json(payload)))
    end
    # Probe an ephemeral port, then serve on it; the close-then-rebind window can race.
    for _ in 1:5
        tcp = Sockets.listen(Sockets.localhost, 0)
        port = Int(Sockets.getsockname(tcp)[2])
        close(tcp)
        server = try
            HTTP.serve!(handler, "127.0.0.1", port; verbose=false)
        catch e
            e isa Base.IOError || rethrow()
            continue
        end
        _semantic_base[] = "http://127.0.0.1:$port"
        try
            return f(), recorded
        finally
            close(server)
        end
    end
    error("could not bind an ephemeral port for the semantic mock server")
end

# ─── Dispatch fixtures (definition order is the option order under test) ─────

route(::nl"the customer wants a refund", t) = (:refund, t)
route(::nl"the customer reports a bug", t) = (:bug, t)
route(::Meaning, t) = :wildcard          # wildcard: not a slot, must be ignored
route(x::Int) = x                        # no Meaning at all: must be ignored

route2(intent::nl"a", t) = (intent, t)   # a NAMED slot names the question

pair(::nl"x1", ::nl"y1") = 11
pair(::nl"x1", ::nl"y2") = 12
pair(::nl"x2", ::nl"y1") = 21
pair(::nl"x2", ::nl"y2") = 22

handle(::nl"greet", who::String) = "hi " * who
handle(::nl"greet", n::Int) = n + 1

half(::nl"p1", ::nl"q1") = 1             # only 2 of the 4 combinations exist
half(::nl"p2", ::nl"q2") = 2

bad_slots(::nl"a", x) = 1                # the slot moves between methods
bad_slots(x, ::nl"b") = 2

bad_arity(::nl"a", x) = 1                # the arity changes between methods
bad_arity(::nl"b", x, y) = 2

no_meanings(x) = x

va_route(::nl"a", rest...) = :va         # varargs cannot carry a meaning

# Two REPL entries, defined as [2] and then [10]: definition order puts [2] first,
# while the string order of their source labels puts "REPL[10]" first.
include_string(@__MODULE__, "repl_route(::nl\"from the second prompt\", t) = 2", "REPL[2]")
include_string(@__MODULE__, "repl_route(::nl\"from the tenth prompt\", t) = 10", "REPL[10]")

# The binding exists up front; its natural-language method is defined while a call is
# already running, so it lives in a newer world than the caller.
late_route(x::Int) = x
function _define_late_route_then_dispatch()
    @eval late_route(::nl"defined at run time", t) = (:late, t)
    nl_dispatch(late_route, "x"; service=SemanticMock, config=_SEM_CFG)
end

# ─── 1. Meaning types ────────────────────────────────────────────────────────

@testset "semantic — nl\"...\" is a type, and the description round-trips" begin
    @test nl"abc" === Meaning{Symbol("abc")}
    @test Meaning("abc") isa nl"abc"
    @test nl"abc"() isa nl"abc"
    @test contains(sprint(show, nl"abc"), "nl\"abc\"")
    @test contains(sprint(show, nl"abc"()), "nl\"abc\"()")
    @test UniLM._description(nl"abc") == "abc"
    @test UniLM._description(nl"abc"()) == "abc"
    # Identical text is the same type; different text is not.
    @test nl"abc" === Meaning{Symbol("abc")} !== nl"abd"
    @test UniLM._description(nl"the customer wants a refund") == "the customer wants a refund"
end

# ─── 2. @branch happy path ───────────────────────────────────────────────────

@testset "semantic — @branch sends one Choice and runs only the selected body" begin
    hits = Dict("a" => 0, "b" => 0, "c" => 0)
    evaluations = Ref(0)
    ticket() = (evaluations[] += 1; "ticket")

    out, seen = _with_semantic_mock(; pick=Dict("branch" => "option b")) do
        @branch ticket() service=SemanticMock config=_SEM_CFG begin
            "option a"              => (hits["a"] += 1; :a)
            "option b"              => (hits["b"] += 1; :b)
            string("option ", "c")  => (hits["c"] += 1; :c)
        end
    end
    @test out === :b
    @test hits == Dict("a" => 0, "b" => 1, "c" => 0)   # unselected bodies never ran
    @test evaluations[] == 1                           # state evaluated exactly once

    @test length(seen) == 1
    @test seen[1]["method"] == "POST"
    @test seen[1]["target"] == "/v1/systemone"
    body = seen[1]["body"]
    @test body["state"] == "ticket"
    @test body["model"] == UniLM.default_typesafe_model()
    @test collect(keys(body["questions"])) == ["branch"]
    q = body["questions"]["branch"]
    @test q["type"] == "choice"
    # Source order, and a bare option name carries a null description.
    @test collect(keys(q["criteria"])) == ["option a", "option b", "option c"]
    @test all(isnothing, values(q["criteria"]))
    @test q["instructions"] == "Select the option that best describes the provided state."

    # `model`, `instructions` and the ("name", "description") tuple all reach the wire.
    out2, seen2 = _with_semantic_mock(; pick=Dict("branch" => "described")) do
        @branch "s" model="jev-1.13.0" instructions="Which situation applies?" service=SemanticMock config=_SEM_CFG begin
            "plain"                            => :plain
            ("described", "a longer account")  => :described
        end
    end
    @test out2 === :described
    body2 = seen2[1]["body"]
    @test body2["model"] == "jev-1.13.0"
    @test body2["questions"]["branch"]["instructions"] == "Which situation applies?"
    @test body2["questions"]["branch"]["criteria"]["described"] == "a longer account"
    @test isnothing(body2["questions"]["branch"]["criteria"]["plain"])
end

# ─── 3. @branch confidence gate and expansion-time errors ────────────────────

@testset "semantic — @branch gates on confidence and rejects bad spellings" begin
    out, _ = _with_semantic_mock(; pick=Dict("branch" => "beta"), confidence=0.5) do
        @branch "s" min_confidence=0.95 service=SemanticMock config=_SEM_CFG begin
            "alpha" => :alpha
            "beta"  => :beta
            _       => :fallback
        end
    end
    @test out === :fallback

    # The same answer with no `_` line is a loud failure, not a silent route.
    err = try
        _with_semantic_mock(; pick=Dict("branch" => "beta"), confidence=0.5) do
            @branch "s" min_confidence=0.95 service=SemanticMock config=_SEM_CFG begin
                "alpha" => :alpha
                "beta"  => :beta
            end
        end
        nothing
    catch e
        e
    end
    @test err isa LowConfidenceError
    text = sprint(showerror, err)
    @test contains(text, "beta")
    @test contains(text, "0.5")
    @test contains(text, "0.95")

    # A well-formed answer above the threshold still routes normally.
    ok, _ = _with_semantic_mock(; pick=Dict("branch" => "beta"), confidence=0.99) do
        @branch "s" min_confidence=0.95 service=SemanticMock config=_SEM_CFG begin
            "alpha" => :alpha
            "beta"  => :beta
            _       => :fallback
        end
    end
    @test ok === :beta

    # Every malformed spelling fails while the macro expands, so it surfaces when
    # the surrounding code is loaded rather than when the branch is first reached.
    @test_throws LoadError Core.eval(@__MODULE__, :(@branch s begin _ => 1 end))
    @test_throws LoadError Core.eval(@__MODULE__, :(@branch s nope=1 begin "a" => 1 end))
    @test_throws LoadError Core.eval(@__MODULE__, :(@branch s begin end))
    @test_throws LoadError Core.eval(@__MODULE__, :(@branch s begin 1 + 1 end))
    @test_throws LoadError Core.eval(@__MODULE__,
        :(@branch s min_confidence=0.5 begin "a" => 1; _ => 2; _ => 3 end))
    @test_throws LoadError Core.eval(@__MODULE__,
        :(@branch s min_confidence=0.5 begin _ => 2; "a" => 1 end))
    @test_throws LoadError Core.eval(@__MODULE__, :(@branch s ["a" => 1]))
end

# ─── 4. @branch on a failed call ─────────────────────────────────────────────

@testset "semantic — @branch throws rather than pick a branch on a failed call" begin
    err = try
        _with_semantic_mock(; status=500, body="""{"detail":"boom"}""") do
            @branch "s" service=SemanticMock config=UniLM.RequestConfig(max_attempts=1) begin
                "alpha" => :alpha
                "beta"  => :beta
            end
        end
        nothing
    catch e
        e
    end
    @test err isa SystemOneError
    @test contains(sprint(showerror, err), "500")
end

# ─── 5. nl_dispatch, one slot ────────────────────────────────────────────────

@testset "semantic — nl_dispatch resolves one slot and hands over to Julia" begin
    @test meanings(route) ==
          Dict(1 => ["the customer wants a refund", "the customer reports a bug"])

    out, seen = _with_semantic_mock(;
            pick=Dict("meaning_1" => "the customer reports a bug")) do
        nl_dispatch(route, "it crashes"; service=SemanticMock, config=_SEM_CFG)
    end
    @test out == (:bug, "it crashes")
    @test length(seen) == 1
    body = seen[1]["body"]
    @test body["state"] == Dict("t" => "it crashes")
    @test collect(keys(body["questions"])) == ["meaning_1"]
    q = body["questions"]["meaning_1"]
    @test q["type"] == "choice"
    @test collect(keys(q["criteria"])) ==
          ["the customer wants a refund", "the customer reports a bug"]
    @test all(isnothing, values(q["criteria"]))

    # A named slot names its question.
    _, named = _with_semantic_mock() do
        nl_dispatch(route2, "z"; service=SemanticMock, config=_SEM_CFG)
    end
    @test collect(keys(named[1]["body"]["questions"])) == ["intent"]

    # An explicit state replaces the Dict built from the ordinary arguments.
    _, overridden = _with_semantic_mock() do
        nl_dispatch(route, "it crashes"; service=SemanticMock, config=_SEM_CFG,
                    state="override")
    end
    @test overridden[1]["body"]["state"] == "override"
end

# ─── 6. nl_dispatch, two slots ───────────────────────────────────────────────

@testset "semantic — nl_dispatch asks one question per slot in one request" begin
    @test meanings(pair) == Dict(1 => ["x1", "x2"], 2 => ["y1", "y2"])

    # No ordinary arguments means nothing describes the state, so one is required.
    @test_throws ArgumentError nl_dispatch(pair; service=SemanticMock, config=_SEM_CFG)

    out, seen = _with_semantic_mock(;
            pick=Dict("meaning_1" => "x2", "meaning_2" => "y1")) do
        nl_dispatch(pair; service=SemanticMock, config=_SEM_CFG, state="s")
    end
    @test out == 21
    @test length(seen) == 1
    questions = seen[1]["body"]["questions"]
    @test collect(keys(questions)) == ["meaning_1", "meaning_2"]
    @test all(q -> q["type"] == "choice", values(questions))
    @test collect(keys(questions["meaning_1"]["criteria"])) == ["x1", "x2"]
    @test collect(keys(questions["meaning_2"]["criteria"])) == ["y1", "y2"]
end

# ─── 7. Composition with ordinary dispatch ───────────────────────────────────

@testset "semantic — a resolved meaning still dispatches on the other arguments" begin
    @test meanings(handle) == Dict(1 => ["greet"])
    out1, _ = _with_semantic_mock() do
        nl_dispatch(handle, "world"; service=SemanticMock, config=_SEM_CFG)
    end
    @test out1 == "hi world"
    out2, _ = _with_semantic_mock() do
        nl_dispatch(handle, 41; service=SemanticMock, config=_SEM_CFG)
    end
    @test out2 == 42
end

# ─── 8. nl_dispatch failure modes ────────────────────────────────────────────

@testset "semantic — nl_dispatch refuses what it cannot resolve" begin
    @test_throws ArgumentError meanings(bad_slots)
    @test_throws ArgumentError meanings(bad_arity)
    @test_throws ArgumentError meanings(no_meanings)
    @test_throws ArgumentError meanings(va_route)
    @test_throws ArgumentError nl_dispatch(no_meanings, 1; service=SemanticMock, config=_SEM_CFG)
    # Wrong number of ordinary arguments for the slot layout.
    @test_throws ArgumentError nl_dispatch(route, "a", "b"; service=SemanticMock, config=_SEM_CFG)
    # `instructions` must carry one entry per slot when it is a vector.
    @test_throws ArgumentError nl_dispatch(route, "a"; service=SemanticMock, config=_SEM_CFG,
                                           instructions=["one", "two"])

    # Below the threshold and with no fallback: a typed error, not a guess.
    err = try
        _with_semantic_mock(; confidence=0.1) do
            nl_dispatch(route, "it crashes"; service=SemanticMock, config=_SEM_CFG,
                        min_confidence=0.9)
        end
        nothing
    catch e
        e
    end
    @test err isa LowConfidenceError
    @test contains(sprint(showerror, err), "meaning_1")

    # With a fallback, the caller's arguments are handed to it unchanged.
    received = Ref{Any}(nothing)
    out, _ = _with_semantic_mock(; confidence=0.1) do
        nl_dispatch(route, "it crashes"; service=SemanticMock, config=_SEM_CFG,
                    min_confidence=0.9, fallback=(a...) -> (received[] = a; :fb))
    end
    @test out === :fb
    @test received[] == ("it crashes",)

    # A failed call throws instead of resolving to a method.
    failed = try
        _with_semantic_mock(; status=500, body="""{"detail":"boom"}""") do
            nl_dispatch(route, "x"; service=SemanticMock,
                        config=UniLM.RequestConfig(max_attempts=1))
        end
        nothing
    catch e
        e
    end
    @test failed isa SystemOneError

    # A combination no method covers is Julia's own MethodError, not swallowed.
    gap = try
        _with_semantic_mock(; pick=Dict("meaning_1" => "p1", "meaning_2" => "q2")) do
            nl_dispatch(half; service=SemanticMock, config=_SEM_CFG, state="s")
        end
        nothing
    catch e
        e
    end
    @test gap isa MethodError
end

@testset "semantic — a cancelled token throws SystemOneError before any request" begin
    tok = cancel!(CancelToken())
    caught(f) = try f(); nothing catch e; e end
    errs, seen = _with_semantic_mock() do
        [caught(() -> nl_dispatch(route, "x"; service=SemanticMock, config=_SEM_CFG, cancel=tok)),
         caught(() -> @branch "s" service=SemanticMock config=_SEM_CFG cancel=tok begin
             "alpha" => :alpha
             "beta"  => :beta
         end),
         caught(() -> with_cancel(tok) do   # ambient token
             nl_dispatch(route, "x"; service=SemanticMock, config=_SEM_CFG)
         end)]
    end
    @test all(e -> e isa SystemOneError && e.result isa SystemOneCallError &&
                   e.result.cause isa UniLMCancelled, errs)
    @test isempty(seen)
end

# ─── 9. Definition order and run-time methods ────────────────────────────────

@testset "semantic — options follow definition order, not source-label order" begin
    @test meanings(repl_route) == Dict(1 => ["from the second prompt", "from the tenth prompt"])
    _, seen = _with_semantic_mock() do
        nl_dispatch(repl_route, "x"; service=SemanticMock, config=_SEM_CFG)
    end
    @test collect(keys(seen[1]["body"]["questions"]["meaning_1"]["criteria"])) ==
          ["from the second prompt", "from the tenth prompt"]
end

@testset "semantic — a method defined at run time is offered AND callable" begin
    # `methods` sees the newest world, so the run-time method is sent (and billed) as an
    # option; calling the chosen one must therefore also happen in the newest world.
    out, _ = _with_semantic_mock() do
        try
            _define_late_route_then_dispatch()
        catch e
            e
        end
    end
    @test out == (:late, "x")
end

# ─── 10. No API in the direct path ───────────────────────────────────────────

@testset "semantic — a meaning method is callable directly, with no service" begin
    _semantic_base[] = "http://127.0.0.1:1"   # nothing is listening
    @test route(nl"the customer wants a refund"(), "x") == (:refund, "x")
    @test route(nl"the customer reports a bug"(), "x") == (:bug, "x")
    @test route(Meaning("the customer wants a refund"), "x") == (:refund, "x")
    @test route(nl"anything else"(), "x") === :wildcard
    @test pair(nl"x2"(), nl"y2"()) == 22
end
