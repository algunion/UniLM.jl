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

# A conversation state machine: which meanings a turn can resolve to depends on the
# state the conversation is in.
abstract type Phase end
struct Waiting <: Phase end
struct Confirming <: Phase end
step(::Waiting, ::nl"gives an order number", msg) = :to_confirming
step(::Confirming, ::nl"confirms", msg) = :confirmed
step(::Confirming, ::nl"declines", msg) = :declined
step(::Phase, ::nl"asks for a human", msg) = :human

patched(::nl"u1", ::nl"v1") = 11         # a gap, closed by a method wild in every slot
patched(::nl"u2", ::nl"v2") = 22
patched(::Meaning, ::Meaning) = :backstop

ambiguous(::nl"m1", x::Int, y) = 1       # an (Int, Int) call matches both equally well
ambiguous(::nl"m1", x, y::Int) = 2

struct Ticket                            # a constructor dispatches on Type{Ticket}
    kind::Symbol
end
Ticket(::nl"urgent", text) = Ticket(:urgent)
kinds(::nl"k", ::Type{Int}) = :int       # a type argument dispatches on Type{Int}

const _counted = Ref(0)                  # how many times a `counted` method ran
counted(::nl"c1", t) = (_counted[] += 1; :c1)
counted(::nl"c2", t) = (_counted[] += 1; :c2)

# Two ordinary arguments whose position order is neither their sorted order nor, on
# Julia 1.13, the iteration order of a Dict holding the same keys.
ship(::nl"asks where the parcel is", order, customer) = (order, customer)

# A method that takes keywords records its unnamed positions under the empty name.
kw_route(::nl"kw first", ::String; k=1) = (:first, k)
kw_route(::nl"kw second", ::String; k=1) = (:second, k)

# Two slots: ["n1", "o1"] is ambiguous for (Int, Int); ["n1", "o2"] and ["n2", "o1"] have no method.
mixed(::nl"n1", ::nl"o1", x::Int, y) = 1
mixed(::nl"n1", ::nl"o1", x, y::Int) = 2
mixed(::nl"n2", ::nl"o2", x, y) = 3

# A partial wildcard pins one slot and leaves the other as `::Meaning`, defined after
# the concrete method and before it.
partial_last(::nl"a complaint", ::nl"a calm tone", msg) = :apologise
partial_last(::nl"a question", ::Meaning, msg) = :answer
partial_first(::nl"a question", ::Meaning, msg) = :answer
partial_first(::nl"a complaint", ::nl"a calm tone", msg) = :apologise

const _pair_calls = Ref(0)               # how many times a `counted_pair` method ran
counted_pair(::nl"c1", ::nl"d1") = (_pair_calls[] += 1; 11)
counted_pair(::nl"c2", ::nl"d2") = (_pair_calls[] += 1; 22)
counted_pair(::Meaning, ::Meaning) = (_pair_calls[] += 1; 0)

cached_route(::nl"the first meaning", t) = (:first, t)   # gains a method during its test

struct Router                            # a callable struct: its methods belong to its type
    label::Symbol
end
(r::Router)(::nl"to the first desk", t) = (r.label, :first)
(r::Router)(::nl"to the second desk", t) = (r.label, :second)

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

    # A combination no method covers is refused before the request: an answer landing
    # on it would be billed and then end in a MethodError.
    gap, billed = _with_semantic_mock(; pick=Dict("meaning_1" => "p1", "meaning_2" => "q2")) do
        try
            nl_dispatch(half; service=SemanticMock, config=_SEM_CFG, state="s")
            nothing
        catch e
            e
        end
    end
    @test gap isa ArgumentError
    @test isempty(billed)
    gaptext = sprint(showerror, gap)
    @test contains(gaptext, repr(["p1", "q2"]))
    @test contains(gaptext, repr(["p2", "q1"]))
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

# ─── 11. Options follow the ordinary argument types ──────────────────────────

@testset "semantic — a call offers only the meanings whose method accepts its arguments" begin
    @test meanings(step) ==
          Dict(2 => ["gives an order number", "confirms", "declines", "asks for a human"])
    @test meanings(step, Tuple{Waiting,String}) == Dict(2 => ["gives an order number", "asks for a human"])
    @test meanings(step, Tuple{Confirming,String}) == Dict(2 => ["confirms", "declines", "asks for a human"])
    @test_throws ArgumentError meanings(step, Tuple{String})              # one ordinary type short
    @test_throws ArgumentError meanings(step, Tuple{Vararg{String}})      # no definite length
    @test_throws ArgumentError meanings(step, Tuple{Int,String})          # no method accepts an Int

    out, seen = _with_semantic_mock(; pick=Dict("meaning_2" => "asks for a human")) do
        nl_dispatch(step, Confirming(), "yes"; service=SemanticMock, config=_SEM_CFG)
    end
    @test out === :human
    @test length(seen) == 1
    @test collect(keys(seen[1]["body"]["questions"]["meaning_2"]["criteria"])) ==
          ["confirms", "declines", "asks for a human"]

    # No natural-language method accepts these types: refused before anything is sent.
    err, sent = _with_semantic_mock() do
        try
            nl_dispatch(step, 1, "x"; service=SemanticMock, config=_SEM_CFG)
            nothing
        catch e
            e
        end
    end
    @test err isa ArgumentError
    @test isempty(sent)
    text = sprint(showerror, err)
    @test contains(text, "step")
    @test contains(text, "Tuple{Int64, String}")
    @test contains(text, "asks for a human")

    # A constructor and a type argument dispatch on Type{T}, not on DataType.
    @test meanings(kinds, Tuple{Type{Int}}) == Dict(1 => ["k"])
    typed, _ = _with_semantic_mock() do
        (nl_dispatch(Ticket, "it is on fire"; service=SemanticMock, config=_SEM_CFG),
         nl_dispatch(kinds, Int; service=SemanticMock, config=_SEM_CFG, state="s"))
    end
    @test typed == (Ticket(:urgent), :int)
end

@testset "semantic — the state lists the ordinary arguments in position order" begin
    _, seen = _with_semantic_mock() do
        nl_dispatch(ship, "A-17", "ada"; service=SemanticMock, config=_SEM_CFG)
    end
    @test collect(keys(seen[1]["body"]["state"])) == ["order", "customer"]
end

# ─── 12. Coverage before the request ─────────────────────────────────────────

@testset "semantic — meaning_gaps lists the offered combinations no method covers" begin
    @test meaning_gaps(half, Tuple{}) == [["p1", "q2"], ["p2", "q1"]]
    @test meaning_gaps(half, Tuple{}) isa Vector{Vector{String}}
    @test isempty(meaning_gaps(pair, Tuple{}))
    @test isempty(meaning_gaps(patched, Tuple{}))                 # the wild method covers both
    @test isempty(meaning_gaps(step, Tuple{Confirming,String}))
    @test meaning_gaps(ambiguous, Tuple{Int,Int}) == [["m1"]]     # ambiguous is a gap too
    @test isempty(meaning_gaps(ambiguous, Tuple{Int,String}))
    @test_throws ArgumentError meaning_gaps(half, Tuple{String})

    # The backstop is not an option, so the call still offers only the defined meanings.
    out, seen = _with_semantic_mock(; pick=Dict("meaning_1" => "u1", "meaning_2" => "v2")) do
        nl_dispatch(patched; service=SemanticMock, config=_SEM_CFG, state="s")
    end
    @test out === :backstop
    @test collect(keys(seen[1]["body"]["questions"]["meaning_1"]["criteria"])) == ["u1", "u2"]
end

@testset "semantic — a gap error tells a missing method from an ambiguous one" begin
    caught(f) = try f(); nothing catch e; e end
    refused(call) = _with_semantic_mock(() -> caught(call))

    # Missing: define it with every slot pinned, or one method wild in EVERY slot.
    err, sent = refused(() -> nl_dispatch(half; service=SemanticMock, config=_SEM_CFG, state="s"))
    @test err isa ArgumentError && isempty(sent)
    text = sprint(showerror, err)
    @test contains(text, "No method covers [\"p1\", \"q2\"], [\"p2\", \"q1\"]")
    @test contains(text, "wild in EVERY slot, such as half(::Meaning, ::Meaning)")
    @test contains(text, "A method wild in only some slots is not supported")
    @test !contains(text, "ambiguous")

    # Ambiguous: a backstop is less specific than the colliding methods, so the advice
    # names them and their intersection instead.
    err, sent = refused(() -> nl_dispatch(ambiguous, 1, 1; service=SemanticMock, config=_SEM_CFG))
    @test err isa ArgumentError && isempty(sent)
    text = sprint(showerror, err)
    @test contains(text, "[\"m1\"] is ambiguous between")
    @test all(m -> contains(text, string(m)), methods(ambiguous))
    @test contains(text, "ambiguous(::nl\"m1\", ::Int64, ::Int64)")
    @test !contains(text, "::Meaning") && !contains(text, "No method covers")

    # Both kinds in one call: each gap gets the advice that closes it.
    @test meaning_gaps(mixed, Tuple{Int,Int}) == [["n1", "o1"], ["n1", "o2"], ["n2", "o1"]]
    err, sent = refused(() -> nl_dispatch(mixed, 1, 1; service=SemanticMock, config=_SEM_CFG))
    @test err isa ArgumentError && isempty(sent)
    text = sprint(showerror, err)
    @test contains(text, "3 of the 4 combinations")
    @test contains(text, "No method covers [\"n1\", \"o2\"], [\"n2\", \"o1\"]:")
    @test contains(text, "mixed(::Meaning, ::Meaning, _, _)")
    @test contains(text, "[\"n1\", \"o1\"] is ambiguous between")
    @test contains(text, "mixed(::nl\"n1\", ::nl\"o1\", ::Int64, ::Int64)")
end

@testset "semantic — a partial wildcard is refused as one" begin
    for f in (partial_last, partial_first)
        err = try meanings(f); nothing catch e; e end
        @test err isa ArgumentError
        text = sprint(showerror, err)
        @test contains(text, "partial wildcard") && contains(text, "not supported")
        @test contains(text, "wild in EVERY slot")
    end
    # A slot that merely moves is not a wildcard, and is not reported as one.
    @test !contains(sprint(showerror, try meanings(bad_slots); nothing catch e; e end), "wildcard")
end

# ─── 12b. Unnamed positions ──────────────────────────────────────────────────

@testset "semantic — unnamed positions of a method with keywords are labelled by index" begin
    out, seen = _with_semantic_mock(; pick=Dict("meaning_1" => "kw second")) do
        nl_dispatch(kw_route, "the ticket"; service=SemanticMock, config=_SEM_CFG)
    end
    @test out == (:second, 1)
    @test collect(keys(seen[1]["body"]["questions"])) == ["meaning_1"]
    @test seen[1]["body"]["state"] == Dict("arg2" => "the ticket")
end

# ─── 13. Decision policy ─────────────────────────────────────────────────────

@testset "semantic — nl_dispatch acts on the decision policy's verdict" begin
    caught(f) = try f(); nothing catch e; e end

    # An offered meaning that is not the argmax is dispatched, and the policy saw the answer.
    saw = Ref{Any}(nothing)
    out, _ = _with_semantic_mock(; pick=Dict("meaning_1" => "the customer wants a refund")) do
        nl_dispatch(route, "t"; service=SemanticMock, config=_SEM_CFG,
                    decide=a -> (saw[] = a; "the customer reports a bug"))
    end
    @test out == (:bug, "t")
    @test saw[] isa ChoiceAnswer && saw[].choice == "the customer wants a refund"

    # A decline runs the fallback with the caller's arguments.
    received = Ref{Any}(nothing)
    out, _ = _with_semantic_mock() do
        nl_dispatch(route, "it crashes"; service=SemanticMock, config=_SEM_CFG,
                    decide=_ -> nothing, fallback=(a...) -> (received[] = a; :fb))
    end
    @test out === :fb
    @test received[] == ("it crashes",)

    # A decline without a fallback is a typed error carrying the question and the answer.
    err, _ = _with_semantic_mock(; pick=Dict("meaning_1" => "the customer reports a bug")) do
        caught(() -> nl_dispatch(route, "x"; service=SemanticMock, config=_SEM_CFG,
                                 decide=_ -> nothing))
    end
    @test err isa DecisionDeclinedError
    @test err.question == "meaning_1"
    @test err.answer isa ChoiceAnswer && err.answer.choice == "the customer reports a bug"
    text = sprint(showerror, err)
    @test contains(text, "meaning_1")
    @test contains(text, "the customer reports a bug")
    @test contains(text, "0.9")

    # A verdict that is neither an offered meaning nor `nothing` runs neither f nor fallback.
    for verdict in ("not offered", :c1, 1)
        _counted[] = 0
        fell_back = Ref(0)
        bad, _ = _with_semantic_mock() do
            caught(() -> nl_dispatch(counted, "x"; service=SemanticMock, config=_SEM_CFG,
                                     decide=_ -> verdict, fallback=(a...) -> (fell_back[] += 1)))
        end
        @test bad isa ArgumentError
        @test _counted[] == 0
        @test fell_back[] == 0
    end

    # One policy per slot.
    out, _ = _with_semantic_mock(; pick=Dict("meaning_1" => "x1", "meaning_2" => "y1")) do
        nl_dispatch(pair; service=SemanticMock, config=_SEM_CFG, state="s",
                    decide=[_ -> "x2", a -> a.choice])
    end
    @test out == 21

    # Every slot is judged before any verdict is acted on: a decline in slot 1 does not
    # hide an invalid verdict in slot 2...
    _pair_calls[] = 0
    fell_back = Ref(0)
    bad, sent = _with_semantic_mock() do
        caught(() -> nl_dispatch(counted_pair; service=SemanticMock, config=_SEM_CFG, state="s",
                                 decide=[_ -> nothing, _ -> "zzz"], fallback=(a...) -> (fell_back[] += 1)))
    end
    @test bad isa ArgumentError && contains(bad.msg, "meaning_2") && contains(bad.msg, "zzz")
    @test _pair_calls[] == 0 && fell_back[] == 0 && length(sent) == 1
    # ...and a decline in slot 2 alone is reported for slot 2.
    second, _ = _with_semantic_mock(; pick=Dict("meaning_1" => "c2", "meaning_2" => "d2")) do
        caught(() -> nl_dispatch(counted_pair; service=SemanticMock, config=_SEM_CFG, state="s",
                                 decide=[a -> a.choice, _ -> nothing]))
    end
    @test second isa DecisionDeclinedError && second.question == "meaning_2"
    @test second.answer.choice == "d2" && _pair_calls[] == 0

    # Refused before any request: two policies at once, a wrong-length vector, and
    # something that cannot be called on an answer.
    refused, sent = _with_semantic_mock() do
        [caught(() -> nl_dispatch(route, "x"; service=SemanticMock, config=_SEM_CFG,
                                  decide=a -> a.choice, min_confidence=0.5)),
         caught(() -> nl_dispatch(pair; service=SemanticMock, config=_SEM_CFG, state="s",
                                  decide=[a -> a.choice])),
         caught(() -> nl_dispatch(route, "x"; service=SemanticMock, config=_SEM_CFG,
                                  decide="the customer wants a refund"))]
    end
    @test all(e -> e isa ArgumentError, refused)
    @test isempty(sent)
end

@testset "semantic — @branch acts on the decision policy's verdict" begin
    out, _ = _with_semantic_mock(; pick=Dict("branch" => "alpha")) do
        @branch "s" decide=(_ -> "beta") service=SemanticMock config=_SEM_CFG begin
            "alpha" => :alpha
            "beta"  => :beta
        end
    end
    @test out === :beta

    # `decide` alone makes a `_` line reachable: a decline takes it.
    out, _ = _with_semantic_mock() do
        @branch "s" decide=(_ -> nothing) service=SemanticMock config=_SEM_CFG begin
            "alpha" => :alpha
            "beta"  => :beta
            _       => :fallback
        end
    end
    @test out === :fallback

    # A decline with no `_` line is a typed error.
    declined = try
        _with_semantic_mock() do
            @branch "s" decide=(_ -> nothing) service=SemanticMock config=_SEM_CFG begin
                "alpha" => :alpha
                "beta"  => :beta
            end
        end
        nothing
    catch e
        e
    end
    @test declined isa DecisionDeclinedError
    @test declined.question == "branch"
    @test declined.answer.choice == "alpha"

    # A verdict that is not an option name runs no body at all.
    ran = Ref(0)
    bad = try
        _with_semantic_mock() do
            @branch "s" decide=(_ -> "gamma") service=SemanticMock config=_SEM_CFG begin
                "alpha" => (ran[] += 1; :alpha)
                "beta"  => (ran[] += 1; :beta)
                _       => (ran[] += 1; :fallback)
            end
        end
        nothing
    catch e
        e
    end
    @test bad isa ArgumentError
    @test ran[] == 0

    # Two policies at once fail while the macro expands.
    both = try
        Core.eval(@__MODULE__, :(@branch s decide=identity min_confidence=0.5 begin "a" => 1 end))
        nothing
    catch e
        e
    end
    @test both isa LoadError && both.error isa ArgumentError

    # A policy that cannot decline when the branch runs leaves `_` unreachable: refused
    # before the request.
    caught(f) = try f(); nothing catch e; e end
    unreachable, sent = _with_semantic_mock() do
        [caught(() -> @branch "s" decide=nothing service=SemanticMock config=_SEM_CFG begin
             "alpha" => :alpha
             _       => :fallback
         end),
         caught(() -> @branch "s" min_confidence=0 service=SemanticMock config=_SEM_CFG begin
             "alpha" => :alpha
             _       => :fallback
         end)]
    end
    @test all(e -> e isa ArgumentError && contains(e.msg, "could never run"), unreachable)
    @test isempty(sent)
end

# ─── 14. The per-call plan ───────────────────────────────────────────────────

@testset "semantic — nl_dispatch reuses its plan until the method table changes" begin
    criteria(seen) = collect(keys(seen[1]["body"]["questions"]["meaning_1"]["criteria"]))
    dispatch(pick) = _with_semantic_mock(; pick=Dict("meaning_1" => pick)) do
        nl_dispatch(cached_route, "x"; service=SemanticMock, config=_SEM_CFG)
    end
    first_out, first_seen = dispatch("the first meaning")
    again_out, again_seen = dispatch("the first meaning")
    @test first_out == again_out == (:first, "x")
    @test again_seen[1]["body"] == first_seen[1]["body"]

    # A method defined after a cached call is offered, and callable, on the next one.
    @eval cached_route(::nl"a later meaning", t) = (:later, t)
    later_out, later_seen = dispatch("a later meaning")
    @test criteria(later_seen) == ["the first meaning", "a later meaning"]
    @test later_out == (:later, "x")

    # The cached plan is the uncached one, and within one world it is computed once.
    cases = [(route, Any[String]), (pair, Any[]), (step, Any[Confirming, String]),
             (step, Any[Waiting, String]), (half, Any[]), (ambiguous, Any[Int, Int]),
             (Ticket, Any[String]), (kinds, Any[Type{Int}]), (ship, Any[String, String]),
             (kw_route, Any[String]), (cached_route, Any[String])]
    for (f, types) in cases
        plan = UniLM._nl_plan(f, types)
        @test UniLM._nl_plan(f, types) === plan
        @test plan == UniLM._nl_plan_uncached(f, types)
    end
    # Concurrent calls get the same plans.
    plans = fetch.([Threads.@spawn UniLM._nl_plan(f, types) for (f, types) in repeat(cases, 20)])
    @test all(p == UniLM._nl_plan_uncached(f, types) for ((f, types), p) in zip(repeat(cases, 20), plans))

    # The key is the TYPE of `f`: two values of a callable struct share one plan, and
    # each call still reaches its own value.
    outs, _ = _with_semantic_mock() do
        [nl_dispatch(Router(label), "x"; service=SemanticMock, config=_SEM_CFG) for label in (:a, :b)]
    end
    @test outs == [(:a, :first), (:b, :first)]
    @test count(k -> k <: Tuple{Router,Vararg}, @lock(UniLM._NL_PLANS, collect(keys(UniLM._NL_PLANS[])))) == 1
end

# ─── 15. Keyed dispatch: the sentences in a table, the keys in the signatures ─

# The same meanings written twice: as `nl"..."` in the signatures, and as keys whose
# sentences a table holds. Both must put the same bytes on the wire.
lit_route(::nl"A", ticket) = (:a, ticket)
lit_route(::nl"B", ticket) = (:b, ticket)
key_route(::Val{:a}, ticket) = (:a, ticket)
key_route(::Val{:b}, ticket) = (:b, ticket)
const KEY_AB = (a = "A", b = "B")

lit_pair(::nl"x1", ::nl"y1", t) = 11
lit_pair(::nl"x1", ::nl"y2", t) = 12
lit_pair(::nl"x2", ::nl"y1", t) = 21
lit_pair(::nl"x2", ::nl"y2", t) = 22
key_pair(::Val{:x1}, ::Val{:y1}, t) = 11
key_pair(::Val{:x1}, ::Val{:y2}, t) = 12
key_pair(::Val{:x2}, ::Val{:y1}, t) = 21
key_pair(::Val{:x2}, ::Val{:y2}, t) = 22
const KEY_X = (x1 = "x1", x2 = "x2")
const KEY_Y = (y1 = "y1", y2 = "y2")

# The call shape the documentation shows.
const INTENT = (refund = "the customer wants a refund",
                bug    = "the customer reports a bug in the app",
                other  = "anything else")
intent_route(::Val{:refund}, t) = (:refund, t)
intent_route(::Val{:bug}, t)    = (:bug, t)
intent_route(::Val{:other}, t)  = (:other, t)

# A taxonomy as a type hierarchy: the table names the leaves, and ordinary dispatch
# picks the most specific method for each — with instances as keys, and with types.
abstract type Intent end
abstract type Billing <: Intent end
abstract type Technical <: Intent end
struct Refund <: Billing end
struct DoubleCharge <: Billing end
struct Crash <: Technical end
struct Other <: Intent end
desk(::Refund, t) = :refund
desk(::Billing, t) = :billing
desk(::Intent, t) = :intent
const LEAVES = [Refund() => "the customer wants a refund",
                DoubleCharge() => "the customer was charged twice",
                Crash() => "the app crashes",
                Other() => "anything else"]
type_desk(::Type{Refund}, t) = :refund
type_desk(::Type{<:Billing}, t) = :billing
type_desk(::Type{<:Intent}, t) = :intent
const LEAF_TYPES = [Refund => "the customer wants a refund",
                    DoubleCharge => "the customer was charged twice",
                    Crash => "the app crashes",
                    Other => "anything else"]

# The state machine of section 11, keyed: which keys a turn can resolve to depends on
# the phase the conversation is in.
kstep(::Waiting, ::Val{:order}, msg) = :to_confirming
kstep(::Confirming, ::Val{:yes}, msg) = :confirmed
kstep(::Confirming, ::Val{:no}, msg) = :declined
kstep(::Phase, ::Val{:human}, msg) = :human
const TURN = (order = "gives an order number", yes = "confirms", no = "declines",
              human = "asks for a human")

@enum Tone calm_tone angry_tone          # enum values are keys, dispatched on their type
tone_reply(tone::Tone, msg) = (tone, msg)
const TONES = [calm_tone => "a calm tone", angry_tone => "an angry tone"]

val_route(::Val{:v1}, t) = :v1           # Val instances are keys as written
val_route(::Val{:v2}, t) = :v2
const VALS = [Val(:v1) => "the first", Val(:v2) => "the second"]

catch_route(::Val{:a}, t) = :a           # a catch-all is the caller's explicit backstop
catch_route(k, t) = (:caught, k)

mutable struct MutableKey end            # an identity no signature can name
both_val(::Val{:a}, mode::Val) = 1       # a table of Val keys fits both positions
typed_route(::Val{:a}, t::Int) = t       # takes only an Int beside its key
two_arity(::Val{:a}, t) = 1              # the same key at two arities
two_arity(::Val{:a}, t, u) = 2

key_half(::Val{:p1}, ::Val{:q1}) = 1     # only 2 of the 4 combinations exist
key_half(::Val{:p2}, ::Val{:q2}) = 2
const P_TABLE = (p1 = "p1", p2 = "p2")
const Q_TABLE = (q1 = "q1", q2 = "q2")

key_patched(::Val{:p1}, ::Val{:q1}) = 11 # the same gaps, closed by a catch-all
key_patched(::Val{:p2}, ::Val{:q2}) = 22
key_patched(p, q) = :backstop

# Two keyed positions tied by a `where` clause: an entry is offered when some entry of the
# other table completes a call, which for "vb" is not the first one tried.
tied(::Val{S}, ::Type{Val{S}}, t) where {S} = S
const TIED = ([Val(:a) => "va", Val(:b) => "vb"], [Val{:a} => "ta", Val{:b} => "tb"])

# ["n1", "o1"] is ambiguous for (Int, Int); ["n1", "o2"] and ["n2", "o1"] have no method.
key_mixed(::Val{:n1}, ::Val{:o1}, x::Int, y) = 1
key_mixed(::Val{:n1}, ::Val{:o1}, x, y::Int) = 2
key_mixed(::Val{:n2}, ::Val{:o2}, x, y) = 3
const N_TABLE = (n1 = "n1", n2 = "n2")
const O_TABLE = (o1 = "o1", o2 = "o2")

const _key_calls = Ref(0)                # how many times a `key_counted` method ran
key_counted(::Val{:c1}, t) = (_key_calls[] += 1; :c1)
key_counted(::Val{:c2}, t) = (_key_calls[] += 1; :c2)
const C_TABLE = (c1 = "the first c", c2 = "the second c")

key_cached(::Val{:a}, t::String) = (:a, t)   # gains a String method for :b during its test
key_cached(::Val{:b}, t::Int) = (:b, t)

@testset "semantic — keyed dispatch sends only the sentences and calls the key's method" begin
    out, seen = _with_semantic_mock(; pick=Dict("meaning_1" => "the customer reports a bug in the app")) do
        nl_dispatch(intent_route, "it crashes"; service=SemanticMock, config=_SEM_CFG, texts=INTENT)
    end
    @test out == (:bug, "it crashes")
    @test length(seen) == 1
    body = seen[1]["body"]
    @test collect(keys(body["questions"])) == ["meaning_1"]
    q = body["questions"]["meaning_1"]
    @test collect(keys(q["criteria"])) == collect(values(INTENT))   # the sentences, in table order
    @test all(isnothing, values(q["criteria"]))
    @test body["state"] == Dict("t" => "it crashes")
    wire = JSON.json(body)                                          # no key reaches the model
    @test !any(k -> contains(wire, "\"$(k)\""), keys(INTENT)) && !contains(wire, "Val")

    first_out, _ = _with_semantic_mock() do                         # the argmax is the first option
        nl_dispatch(intent_route, "my money back"; service=SemanticMock, config=_SEM_CFG, texts=INTENT)
    end
    @test first_out == (:refund, "my money back")

    # A catch-all takes every key: the caller's backstop, not a key no method takes.
    backstop, _ = _with_semantic_mock(; pick=Dict("meaning_1" => "Z")) do
        nl_dispatch(catch_route, "t"; service=SemanticMock, config=_SEM_CFG, texts=(a = "A", z = "Z"))
    end
    @test backstop == (:caught, Val(:z))
end

@testset "semantic — keyed dispatch puts the literal request on the wire, byte for byte" begin
    cases = [(lit_route, key_route, KEY_AB, Dict("meaning_1" => "B"), (:b, "the ticket")),
             (lit_pair, key_pair, (KEY_X, KEY_Y), Dict("meaning_1" => "x2", "meaning_2" => "y1"), 21)]
    for (lit, keyed, texts, pick, expected) in cases
        mktempdir() do dir
            outs, seen = _with_semantic_mock(; pick) do
                with_recorded_answers(dir; mode=:record) do
                    (nl_dispatch(lit, "the ticket"; service=SemanticMock, config=_SEM_CFG),
                     nl_dispatch(keyed, "the ticket"; service=SemanticMock, config=_SEM_CFG, texts))
                end
            end
            @test outs == (expected, expected)
            @test length(seen) == 2
            @test JSON.json(seen[1]["body"]) == JSON.json(seen[2]["body"])
            # A recording is keyed by the SHA-256 of the exact request bytes: one file, one body.
            @test length(readdir(dir)) == 1
        end
    end
end

@testset "semantic — a keyed call refuses a bad table, or keys its methods do not take, before any request" begin
    caught(f) = try f(); nothing catch e; e end
    dispatch(f, args...; kw...) = () -> nl_dispatch(f, args...; service=SemanticMock, config=_SEM_CFG, kw...)
    classify(texts; kw...) = () -> nl_classify("s", texts; service=SemanticMock, config=_SEM_CFG, kw...)
    many = [Symbol("k", i) => "sentence $(i)" for i in 1:256]
    refusals = [
        "a Dict"                       => (dispatch(key_route, "t"; texts=Dict(:a => "A", :b => "B")), "collect(pairs(d))"),
        "an empty table"               => (dispatch(key_route, "t"; texts=NamedTuple()), "at least one"),
        "more than 255 entries"        => (dispatch(key_route, "t"; texts=many), "at most 255"),
        "a sentence that is no string" => (dispatch(key_route, "t"; texts=(a = 1,)), "non-empty string"),
        "a blank sentence"             => (dispatch(key_route, "t"; texts=(a = "  ",)), "non-empty string"),
        "a repeated sentence"          => (dispatch(key_route, "t"; texts=[:a => "same", :b => "same"]), "could not tell them apart"),
        "the same key twice"           => (dispatch(key_route, "t"; texts=[:a => "A", Val(:a) => "B"]), "are the same key"),
        "a Tuple of pairs"             => (dispatch(key_route, "t"; texts=(:a => "A", :b => "B")), "pass one table as a vector"),
        "an empty Tuple"               => (dispatch(key_route, "t"; texts=()), "empty"),
        "an entry that is no pair"     => (dispatch(key_route, "t"; texts=[:a => "A", "B"]), "`key => sentence` pairs"),
        "a String key"                 => (dispatch(key_route, "t"; texts=["a" => "A"]), "\"a\""),
        "an Int key"                   => (dispatch(key_route, "t"; texts=[1 => "A"]), "Int64"),
        "a mutable instance key"       => (dispatch(key_route, "t"; texts=[MutableKey() => "A"]), "MutableKey"),
        "keys no method declares"      => (dispatch(key_route, "t"; texts=(zz = "Z",)), "Val{:zz}"),
        "keys declared twice"          => (dispatch(both_val, Val(:fast); texts=(a = "A",)), "positions 1 and 2"),
        "two tables at one position"   => (dispatch(key_route; texts=((a = "A",), (b = "B",)), state="s"), "each table needs its own position"),
        "meanings and keys mixed"      => (dispatch(route, "t"; texts=(a = "A",)), "not both"),
        "a key no method takes"        => (dispatch(key_route, "t"; texts=(a = "A", b = "B", c = "C")), "key_route(::Val{:c}, _)"),
        "no key for these types"       => (dispatch(typed_route, "t"; texts=(a = "A",)), "Tuple{String}"),
        "no method of the arity"       => (dispatch(key_route, "t", "u"; texts=KEY_AB), "3 positional arguments"),
        "a gap"                        => (dispatch(key_half; texts=(P_TABLE, Q_TABLE), state="s"), "No method covers"),
        "a bad on_response"            => (dispatch(key_route, "t"; texts=KEY_AB, on_response=42), "on_response"),
        "decide with min_confidence"   => (dispatch(key_route, "t"; texts=KEY_AB, decide=a -> a.choice, min_confidence=0.5), "not both"),
        "nl_classify: a tuple of tables"  => (classify((KEY_X, KEY_Y)), "one table"),
        "nl_classify: a Tuple of pairs"   => (classify((:a => "A", :b => "B")), "pass one table as a vector"),
        "nl_classify: a Dict"             => (classify(Dict(:a => "A")), "no order"),
        "nl_classify: a bad key"          => (classify(["a" => "A"]), "\"a\""),
        "nl_classify: a bad on_response"  => (classify(INTENT; on_response="audit"), "on_response"),
        "nl_classify: decide with min_confidence" => (classify(INTENT; decide=a -> a.choice, min_confidence=0.5), "not both"),
        "nl_classify: a decide that is no callable" => (classify(INTENT; decide=[a -> a.choice]), "decide"),
    ]
    results, sent = _with_semantic_mock() do
        [caught(call) for (_, (call, _)) in refusals]
    end
    @test isempty(sent)                                              # nothing was billed
    for ((label, (_, fragment)), err) in zip(refusals, results)
        @testset "refused: $(label)" begin
            @test err isa ArgumentError
            @test contains(sprint(showerror, err), fragment)
        end
    end

    # The drift error names the key and the method that would take it.
    drift = results[findfirst(r -> first(r) == "a key no method takes", refusals)]
    @test contains(drift.msg, ":c") && contains(drift.msg, "key_route(::Val{:c}, _)")

    # @branch checks its hook before its request too.
    branch_err, branch_sent = _with_semantic_mock() do
        caught(() -> @branch "s" service=SemanticMock config=_SEM_CFG on_response=42 begin
            "alpha" => :alpha
            "beta"  => :beta
        end)
    end
    @test branch_err isa ArgumentError && contains(branch_err.msg, "on_response") && isempty(branch_sent)
end

@testset "semantic — a table of leaves dispatches through the type hierarchy" begin
    for (f, table) in ((desk, LEAVES), (type_desk, LEAF_TYPES))
        @test meanings(f; texts=table) == Dict(1 => last.(table))
        @test meanings(f, Tuple{String}; texts=table) == Dict(1 => last.(table))   # all four offered
        @test isempty(meaning_gaps(f, Tuple{String}; texts=table))
        for (sentence, expected) in zip(last.(table), (:refund, :billing, :intent, :intent))
            out, seen = _with_semantic_mock(; pick=Dict("meaning_1" => sentence)) do
                nl_dispatch(f, "a ticket"; service=SemanticMock, config=_SEM_CFG, texts=table)
            end
            @test out === expected
            @test collect(keys(seen[1]["body"]["questions"]["meaning_1"]["criteria"])) == last.(table)
        end
    end
end

@testset "semantic — keyed options follow the argument types, in table order" begin
    @test meanings(kstep; texts=TURN) ==
          Dict(2 => ["gives an order number", "confirms", "declines", "asks for a human"])
    @test meanings(kstep, Tuple{Waiting,String}; texts=TURN) == Dict(2 => ["gives an order number", "asks for a human"])
    @test meanings(kstep, Tuple{Confirming,String}; texts=TURN) == Dict(2 => ["confirms", "declines", "asks for a human"])
    # Table order, not definition order.
    backwards = (human = "asks for a human", no = "declines", yes = "confirms", order = "gives an order number")
    @test meanings(kstep, Tuple{Confirming,String}; texts=backwards) ==
          Dict(2 => ["asks for a human", "declines", "confirms"])
    @test_throws ArgumentError meanings(kstep, Tuple{Int,String}; texts=TURN)   # no key for an Int phase
    @test isempty(meaning_gaps(kstep, Tuple{Waiting,String}; texts=TURN))

    for (phase, pick, expected) in ((Waiting(), "gives an order number", :to_confirming),
                                    (Confirming(), "declines", :declined),
                                    (Confirming(), "asks for a human", :human))
        out, seen = _with_semantic_mock(; pick=Dict("meaning_2" => pick)) do
            nl_dispatch(kstep, phase, "A-17"; service=SemanticMock, config=_SEM_CFG, texts=TURN)
        end
        @test out === expected
        # The preview is the wire.
        @test collect(keys(seen[1]["body"]["questions"]["meaning_2"]["criteria"])) ==
              meanings(kstep, Tuple{typeof(phase),String}; texts=TURN)[2]
        @test collect(keys(seen[1]["body"]["state"])) == ["arg1", "msg"]
    end
end

@testset "semantic — enum values and Val instances are keys" begin
    out, seen = _with_semantic_mock(; pick=Dict("tone" => "an angry tone")) do
        nl_dispatch(tone_reply, "WHY IS IT BROKEN"; service=SemanticMock, config=_SEM_CFG, texts=TONES)
    end
    @test out == (angry_tone, "WHY IS IT BROKEN")
    @test collect(keys(seen[1]["body"]["questions"])) == ["tone"]      # a named slot names its question
    @test collect(keys(seen[1]["body"]["questions"]["tone"]["criteria"])) == ["a calm tone", "an angry tone"]
    out, _ = _with_semantic_mock(; pick=Dict("meaning_1" => "the second")) do
        nl_dispatch(val_route, "x"; service=SemanticMock, config=_SEM_CFG, texts=VALS)
    end
    @test out === :v2
end

@testset "semantic — keyed previews, and the arity they take from a call" begin
    @test meanings(intent_route; texts=INTENT) == Dict(1 => collect(values(INTENT)))
    @test meanings(intent_route, Tuple{String}; texts=INTENT) == Dict(1 => collect(values(INTENT)))
    # Two arities declare the key: only argument types can say which call is meant.
    err = try meanings(two_arity; texts=(a = "A",)); nothing catch e; e end
    @test err isa ArgumentError && contains(err.msg, "argtypes")
    @test meanings(two_arity, Tuple{String}; texts=(a = "A",)) == Dict(1 => ["A"])
    @test meanings(two_arity, Tuple{String,Int}; texts=(a = "A",)) == Dict(1 => ["A"])
    # Without `texts`, the previews are those of the literal pathway.
    @test_throws ArgumentError meanings(intent_route)
end

@testset "semantic — several tables: one per keyed position, every combination checked first" begin
    caught(f) = try f(); nothing catch e; e end
    refused(call) = _with_semantic_mock(() -> caught(call))
    @test meanings(key_pair, Tuple{String}; texts=(KEY_X, KEY_Y)) == Dict(1 => ["x1", "x2"], 2 => ["y1", "y2"])
    @test meanings(key_pair, Tuple{String}; texts=(KEY_Y, KEY_X)) == Dict(1 => ["x1", "x2"], 2 => ["y1", "y2"])

    # Each key lands in its own position, whatever the order of the tables.
    out, seen = _with_semantic_mock(; pick=Dict("meaning_1" => "x2", "meaning_2" => "y2")) do
        nl_dispatch(key_pair, "t"; service=SemanticMock, config=_SEM_CFG, texts=(KEY_Y, KEY_X))
    end
    @test out == 22
    @test collect(keys(seen[1]["body"]["questions"])) == ["meaning_1", "meaning_2"]

    # Missing combinations: define each, or one catch-all.
    @test meaning_gaps(key_half, Tuple{}; texts=(P_TABLE, Q_TABLE)) == [["p1", "q2"], ["p2", "q1"]]
    err, sent = refused(() -> nl_dispatch(key_half; service=SemanticMock, config=_SEM_CFG, state="s",
                                          texts=(P_TABLE, Q_TABLE)))
    @test err isa ArgumentError && isempty(sent)
    text = sprint(showerror, err)
    @test contains(text, "No method covers [\"p1\", \"q2\"], [\"p2\", \"q1\"]")
    @test contains(text, "key_half(::Val{:p1}, ::Val{:q2})")
    @test contains(text, "key_half(_, _)")
    @test !contains(text, "Meaning") && !contains(text, "ambiguous")
    @test isempty(meaning_gaps(key_patched, Tuple{}; texts=(P_TABLE, Q_TABLE)))
    out, _ = _with_semantic_mock(; pick=Dict("meaning_1" => "p1", "meaning_2" => "q2")) do
        nl_dispatch(key_patched; service=SemanticMock, config=_SEM_CFG, state="s", texts=(P_TABLE, Q_TABLE))
    end
    @test out === :backstop

    # Positions tied by a `where` clause are searched as whole calls.
    @test meanings(tied, Tuple{String}; texts=TIED) == Dict(1 => ["va", "vb"], 2 => ["ta", "tb"])
    @test meaning_gaps(tied, Tuple{String}; texts=TIED) == [["va", "tb"], ["vb", "ta"]]

    # Ambiguous and missing combinations in one call: each gets the advice that closes it.
    @test meaning_gaps(key_mixed, Tuple{Int,Int}; texts=(N_TABLE, O_TABLE)) == [["n1", "o1"], ["n1", "o2"], ["n2", "o1"]]
    err, sent = refused(() -> nl_dispatch(key_mixed, 1, 1; service=SemanticMock, config=_SEM_CFG,
                                          texts=(N_TABLE, O_TABLE)))
    @test err isa ArgumentError && isempty(sent)
    text = sprint(showerror, err)
    @test contains(text, "3 of the 4 combinations")
    @test contains(text, "No method covers [\"n1\", \"o2\"], [\"n2\", \"o1\"]:")
    @test contains(text, "[\"n1\", \"o1\"] is ambiguous between")
    @test contains(text, "key_mixed(::Val{:n1}, ::Val{:o1}, ::Int64, ::Int64)")
end

@testset "semantic — a keyed decision names a sentence, a key as written, or nothing" begin
    outcome(f) = try f() catch e; e end                  # the value, or what was thrown
    keyed(f, args...; mock=(;), kw...) = first(_with_semantic_mock(; mock...) do    # argmax: the first option
        outcome(() -> nl_dispatch(f, args...; service=SemanticMock, config=_SEM_CFG, kw...))
    end)
    @test keyed(intent_route, "t"; texts=INTENT, decide=_ -> :bug) == (:bug, "t")
    @test keyed(intent_route, "t"; texts=INTENT, decide=_ -> "anything else") == (:other, "t")
    @test keyed(desk, "t"; texts=LEAVES, decide=_ -> Crash()) === :intent
    @test keyed(type_desk, "t"; texts=LEAF_TYPES, decide=_ -> DoubleCharge) === :billing
    @test keyed(tone_reply, "t"; texts=TONES, decide=_ -> angry_tone) == (angry_tone, "t")
    @test keyed(val_route, "t"; texts=VALS, decide=_ -> Val(:v2)) === :v2

    # A decline runs the fallback with the caller's arguments, or is a typed error.
    received = Ref{Any}(nothing)
    @test keyed(intent_route, "t"; texts=INTENT, decide=_ -> nothing, fallback=(a...) -> (received[] = a; :fb)) === :fb
    @test received[] == ("t",)
    declined = keyed(intent_route, "t"; texts=INTENT, decide=_ -> nothing)
    @test declined isa DecisionDeclinedError && declined.question == "meaning_1"
    @test declined.answer.choice == "the customer wants a refund"
    low = keyed(intent_route, "t"; mock=(; confidence=0.1), texts=INTENT, min_confidence=0.9)
    @test low isa LowConfidenceError && low.question == "meaning_1"

    # Anything else runs neither f nor the fallback: a key's dispatch value is not the key
    # as written, and a key the table does not hold is not an option.
    for verdict in (Val(:c2), 1, "not offered", :c3)
        _key_calls[] = 0
        fell_back = Ref(0)
        bad = keyed(key_counted, "x"; texts=C_TABLE, decide=_ -> verdict, fallback=(a...) -> (fell_back[] += 1))
        @test bad isa ArgumentError
        @test _key_calls[] == 0 && fell_back[] == 0
    end
    # Nor is a key of the table that this call does not offer.
    not_offered = keyed(kstep, Waiting(), "x"; texts=TURN, decide=_ -> :yes)
    @test not_offered isa ArgumentError && contains(not_offered.msg, ":yes")
end

@testset "semantic — nl_classify returns the chosen key as written in the table" begin
    key, seen = _with_semantic_mock(; pick=Dict("classify" => "the customer reports a bug in the app")) do
        nl_classify("the app crashes on start", INTENT; service=SemanticMock, config=_SEM_CFG)
    end
    @test key === :bug                                   # the Symbol as written, not Val(:bug)
    @test length(seen) == 1
    body = seen[1]["body"]
    @test body["state"] == "the app crashes on start"    # passed as given
    @test body["model"] == UniLM.default_typesafe_model()
    @test collect(keys(body["questions"])) == ["classify"]
    q = body["questions"]["classify"]
    @test collect(keys(q["criteria"])) == collect(values(INTENT)) && all(isnothing, values(q["criteria"]))
    @test q["instructions"] == "Select the option that best describes the provided state."

    for (table, sentence, expected) in ((LEAF_TYPES, "the app crashes", Crash),
                                        (LEAVES, "anything else", Other()),
                                        (TONES, "an angry tone", angry_tone),
                                        (VALS, "the second", Val(:v2)))
        got, _ = _with_semantic_mock(; pick=Dict("classify" => sentence)) do
            nl_classify("s", table; service=SemanticMock, config=_SEM_CFG)
        end
        @test got === expected
    end

    # A structured state goes out as given, and so do the model and the instructions.
    state = JSON.Object{String,Any}("ticket" => "it crashes", "customer" => "ada")
    _, seen = _with_semantic_mock() do
        nl_classify(state, INTENT; service=SemanticMock, config=_SEM_CFG, model="jev-1.13.0",
                    instructions="Which intent does the ticket show?")
    end
    @test collect(keys(seen[1]["body"]["state"])) == ["ticket", "customer"]
    @test seen[1]["body"]["model"] == "jev-1.13.0"
    @test seen[1]["body"]["questions"]["classify"]["instructions"] == "Which intent does the ticket show?"
end

@testset "semantic — nl_classify acts on the decision policy" begin
    outcome(f) = try f() catch e; e end                  # the value, or what was thrown
    classify(; mock=(;), kw...) = first(_with_semantic_mock(; mock...) do
        outcome(() -> nl_classify("the ticket", INTENT; service=SemanticMock, config=_SEM_CFG, kw...))
    end)
    low = classify(; mock=(; confidence=0.1), min_confidence=0.9)
    @test low isa LowConfidenceError && low.question == "classify"
    got = Ref{Any}(nothing)
    @test classify(; mock=(; confidence=0.1), min_confidence=0.9, fallback=s -> (got[] = s; :fb)) === :fb
    @test got[] == "the ticket"                                              # fallback(state)
    @test classify(; decide=_ -> :other) === :other                          # a key as written
    @test classify(; decide=_ -> "the customer reports a bug in the app") === :bug   # a sentence
    declined = classify(; decide=_ -> nothing)
    @test declined isa DecisionDeclinedError && declined.question == "classify"
    @test classify(; decide=_ -> nothing, fallback=s -> (:fb, s)) == (:fb, "the ticket")
    fell_back = Ref(0)
    for verdict in (Val(:bug), "not offered", 2)
        @test classify(; decide=_ -> verdict, fallback=_ -> (fell_back[] += 1)) isa ArgumentError
    end
    @test fell_back[] == 0
    @test classify(; mock=(; status=500, body="""{"detail":"boom"}""")) isa SystemOneError
end

@testset "semantic — on_response sees the success once, before the decision policy" begin
    branch(hook, decide) = @branch "x" service=SemanticMock config=_SEM_CFG on_response=hook decide=decide begin
        "alpha" => :alpha
        "beta"  => :beta
    end
    calls = [
        "nl_dispatch" => (hook, decide) -> nl_dispatch(route, "x"; service=SemanticMock, config=_SEM_CFG,
                                                       on_response=hook, decide),
        "keyed nl_dispatch" => (hook, decide) -> nl_dispatch(intent_route, "x"; service=SemanticMock,
                                                             config=_SEM_CFG, texts=INTENT, on_response=hook, decide),
        "nl_classify" => (hook, decide) -> nl_classify("x", INTENT; service=SemanticMock, config=_SEM_CFG,
                                                       on_response=hook, decide),
        "@branch" => branch,
    ]
    for (label, call) in calls
        @testset "$(label)" begin
            log = Any[]
            _, sent = _with_semantic_mock() do
                call(r -> push!(log, r), a -> (push!(log, :decide); a.choice))
            end
            @test length(sent) == 1
            @test length(log) == 2 && log[2] === :decide
            @test log[1] isa SystemOneSuccess
            @test log[1].response.request_id == "req_mock" && log[1].response.model == "jev-1.13.0"
        end
    end

    caught(f) = try f(); nothing catch e; e end
    hooked = Ref(0)
    hook = _ -> (hooked[] += 1)
    # A failed call throws SystemOneError, which carries the result; the hook never runs.
    failed, _ = _with_semantic_mock(; status=500, body="""{"detail":"boom"}""") do
        [caught(() -> nl_dispatch(route, "x"; service=SemanticMock, config=_SEM_CFG, on_response=hook)),
         caught(() -> nl_dispatch(intent_route, "x"; service=SemanticMock, config=_SEM_CFG, texts=INTENT,
                                  on_response=hook)),
         caught(() -> nl_classify("x", INTENT; service=SemanticMock, config=_SEM_CFG, on_response=hook)),
         caught(() -> @branch "x" service=SemanticMock config=_SEM_CFG on_response=hook begin
             "alpha" => :alpha
             "beta"  => :beta
         end)]
    end
    @test all(e -> e isa SystemOneError, failed) && hooked[] == 0
    # Nothing sent, nothing to see.
    tok = cancel!(CancelToken())
    unsent, sent = _with_semantic_mock() do
        [caught(() -> nl_dispatch(intent_route, "x"; service=SemanticMock, config=_SEM_CFG, texts=INTENT,
                                  on_response=hook, cancel=tok)),
         caught(() -> nl_classify("x", INTENT; service=SemanticMock, config=_SEM_CFG, on_response=hook, cancel=tok)),
         caught(() -> nl_dispatch(key_half; service=SemanticMock, config=_SEM_CFG, state="s",
                                  texts=(P_TABLE, Q_TABLE), on_response=hook))]
    end
    @test isempty(sent) && hooked[] == 0
    @test unsent[1] isa SystemOneError && unsent[2] isa SystemOneError && unsent[3] isa ArgumentError

    # An exception from the hook propagates, and nothing after it runs: no method, no body,
    # no fallback — even where the policy would have declined.
    boom = _ -> error("the audit store is down")
    _key_calls[] = 0
    _counted[] = 0
    ran = Ref(0)
    raised, sent = _with_semantic_mock(; confidence=0.1) do
        [caught(() -> nl_dispatch(key_counted, "x"; service=SemanticMock, config=_SEM_CFG, texts=C_TABLE,
                                  on_response=boom)),
         caught(() -> nl_dispatch(key_counted, "x"; service=SemanticMock, config=_SEM_CFG, texts=C_TABLE,
                                  on_response=boom, min_confidence=0.9, fallback=(a...) -> (ran[] += 1))),
         caught(() -> nl_dispatch(counted, "x"; service=SemanticMock, config=_SEM_CFG, on_response=boom)),
         caught(() -> nl_classify("x", C_TABLE; service=SemanticMock, config=_SEM_CFG, on_response=boom,
                                  min_confidence=0.9, fallback=_ -> (ran[] += 1))),
         caught(() -> @branch "x" service=SemanticMock config=_SEM_CFG on_response=boom min_confidence=0.9 begin
             "alpha" => (ran[] += 1)
             _       => (ran[] += 1)
         end)]
    end
    @test length(sent) == 5
    @test all(e -> e isa ErrorException && contains(e.msg, "audit store"), raised)
    @test _key_calls[] == 0 && _counted[] == 0 && ran[] == 0
end

@testset "semantic — a keyed plan follows the table's contents and the method table" begin
    criteria(seen) = collect(keys(seen[1]["body"]["questions"]["meaning_1"]["criteria"]))
    dispatch(texts; pick=Dict{String,String}()) = _with_semantic_mock(; pick) do
        nl_dispatch(key_cached, "x"; service=SemanticMock, config=_SEM_CFG, texts)
    end
    # Only :a has a String method so far.
    table = (a = "the first sentence", b = "the second sentence")
    @test criteria(last(dispatch(table))) == ["the first sentence"]
    table = (a = "a reworded first sentence", b = "the second sentence")    # rebound, new contents
    @test criteria(last(dispatch(table))) == ["a reworded first sentence"]

    # A vector mutated in place after a call does not reach the plan made from it.
    v = [:a => "the first sentence", :b => "the second sentence"]
    @test criteria(last(dispatch(v))) == ["the first sentence"]
    v[1] = :a => "a mutated sentence"
    @test criteria(last(dispatch(v))) == ["a mutated sentence"]
    stored = @lock UniLM._NL_KEYED_PLANS collect(keys(UniLM._NL_KEYED_PLANS[]))
    @test all(k -> last(k) isa Tuple && all(t -> t isa UniLM._NLTable, last(k)), stored)

    # A method defined after a cached call is offered, and callable, on the next one.
    @eval key_cached(::Val{:b}, t::String) = (:b, t)
    out, seen = dispatch(v; pick=Dict("meaning_1" => "the second sentence"))
    @test criteria(seen) == ["a mutated sentence", "the second sentence"]
    @test out == (:b, "x")

    # The cached plan is the uncached one, and within one world it is computed once.
    cases = [(intent_route, Any[String], INTENT), (kstep, Any[Waiting, String], TURN),
             (kstep, Any[Confirming, String], TURN), (desk, Any[String], LEAVES),
             (type_desk, Any[String], LEAF_TYPES), (key_pair, Any[String], (KEY_Y, KEY_X)),
             (key_half, Any[], (P_TABLE, Q_TABLE)), (key_mixed, Any[Int, Int], (N_TABLE, O_TABLE))]
    for (f, types, texts) in cases
        tables = UniLM._nl_tables(texts)
        plan = UniLM._nl_keyed_plan(f, types, tables)
        @test UniLM._nl_keyed_plan(f, types, UniLM._nl_tables(texts)) === plan
        @test plan == UniLM._nl_keyed_plan_uncached(f, types, tables)
    end
end
