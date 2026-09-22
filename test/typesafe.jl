# test/typesafe.jl — TypeSafe System One (Jev) contracts, zero-spend.
#
# The goldens under test/fixtures/typesafe/ are verbatim captures of the live
# service (request bodies, 200 responses, and one body per error shape). They are
# the oracle: what this client serializes must parse equal to the captured
# request, and what it decodes must come out of the captured response.
#
# Requires test/platform_seam_fixtures.jl (SeamProbe, _TINY_DEADLINE,
# _reached_seam) to be included first.

using Sockets

_ts_fixture(name::AbstractString) =
    read(joinpath(@__DIR__, "fixtures", "typesafe", name), String)
_ts_json(name::AbstractString) = JSON.parse(_ts_fixture(name))

# Bounded budget for the mock-server testsets: one attempt is enough for the
# statuses served here (none is retryable), and a short deadline keeps a bound
# on the suite if a bind ever goes wrong.
const _TS_MOCK_CFG = UniLM.RequestConfig(max_attempts=1, total_deadline=30.0)

"""
Run `f()` against a local HTTP server that answers every request with the given
status/body/headers, and return `(result_of_f, recorded_requests)`. The key and
base URL are supplied through the environment, so the call under test goes
through the real `TYPESAFEServiceEndpoint` auth and base-URL resolution.
"""
function _ts_mock(f::Function; status::Int=200, body::AbstractString="{}",
                  headers::Vector{Pair{String,String}}=["Content-Type" => "application/json"])
    recorded = Dict{String,Any}[]
    # HTTP.jl 1.x hands the handler a byte vector; 2.x hands it a body object and
    # uses a distinct sentinel for a request with no body at all, which supports
    # no byte access. "Can it be copied" separates the two without naming either.
    bodytext(req) = applicable(copy, req.body) ? String(copy(req.body)) : ""
    handler = function (req)
        push!(recorded, Dict{String,Any}(
            "method" => String(req.method),
            "target" => String(req.target),
            "body" => bodytext(req),
            "headers" => Dict{String,String}(
                lowercase(String(k)) => String(v) for (k, v) in req.headers)))
        HTTP.Response(status, headers, Vector{UInt8}(body))
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
        try
            out = withenv(UniLM.TYPESAFE_API_KEY => "ts-test-key",
                          UniLM.TYPESAFE_BASE_URL_ENV => "http://127.0.0.1:$port") do
                f()
            end
            return out, recorded
        finally
            close(server)
        end
    end
    error("could not bind an ephemeral port for the TypeSafe mock server")
end

# The Julia spelling of live/02-basic-three-questions.request.json.
_ts_request_02() = SystemOneRequest(
    "Help! My payouts have been failing for 3 days and nobody has replied to my emails.",
    ["department" => choice("Which team should handle this ticket?", (
         billing   = "Payments, invoicing, payouts, refunds",
         technical = "Bugs, outages, integrations",
         sales     = "Pricing, upgrades, new accounts")),
     "urgency" => score("How urgent is this ticket?",
         ["Can wait", "Needs attention this week", "Needs attention today"]),
     "is_frustrated" => noul("Is the customer frustrated?";
         yes = "The customer expresses frustration or impatience",
         no  = "The customer is neutral or satisfied")];
    model = "jev-latest")

@testset "TypeSafe — request goldens serialize to the captured wire" begin
    @test JSON.parse(JSON.json(_ts_request_02())) ==
          _ts_json("02-basic-three-questions.request.json")

    # Fixture 03: structured instructions and criteria, object state.
    state03 = Dict{String,Any}(
        "ticket" => Dict{String,Any}(
            "subject" => "Duplicate charge on order A-104",
            "messages" => [
                Dict{String,Any}("from" => "customer",
                    "text" => "I was charged twice for order A-104. Please refund the duplicate today."),
                Dict{String,Any}("from" => "support", "text" => "We are checking the charges.")]),
        "order" => Dict{String,Any}("id" => "A-104", "charges" => [
            Dict{String,Any}("amount_usd" => 49, "status" => "captured"),
            Dict{String,Any}("amount_usd" => 49, "status" => "captured")]),
        "refund_policy" => "Duplicate charges are eligible for a refund.")
    req03 = SystemOneRequest(state03, (
        department = choice(
            (question = "Which team should handle the `ticket`?",
             focus = "Classify the customer's primary request, not every topic mentioned."),
            (billing = (what = "Charges, invoices, refunds, or subscriptions",
                        not_for = "Order tracking or account access",
                        examples = ["I was charged twice", "Where is my refund?"]),
             orders  = (what = "Order status, delivery, cancellation, or returns",
                        not_for = "Charges or account access",
                        examples = ["Where is my package?", "Cancel my order"]),
             account = (what = "Login, password, profile, or security",
                        not_for = "Charges or delivery",
                        examples = ["I can't log in", "Change my email"]))),
        refund_size = score(
            (field = (name = "refund_amount", type = "number", unit = "USD",
                      description = "The duplicate amount to refund."),
             question = "How large is the duplicate charge in `order`?"),
            [(summary = "Under \$25",
              signals = ["A small charge", "Below the manual review threshold"]),
             (summary = "\$25 to \$100", signals = ["A typical single-item order"]),
             (summary = "Over \$100", signals = ["A large order", "Needs supervisor approval"])]),
        policy_allows_refund = noul(
            (question = "Does `refund_policy` allow refunding the duplicate charge in `order`?",
             inspect = "refund_policy",
             focus = "Only consider the written policy, not goodwill exceptions.");
            yes = (what = "The written policy covers this charge",
                   examples = ["Duplicate charges are refundable"]),
            no  = (what = "The written policy does not cover this charge",
                   examples = ["Final sale, no refunds"])),
    ); model = "jev-latest")
    @test JSON.parse(JSON.json(req03)) == _ts_json("03-structured-questions.request.json")

    # Fixture 19: array state, bare option names (descriptions are null).
    req19 = SystemOneRequest(
        ["Hi", "My customer number is TS1337.", "My card was charged twice."],
        ["topic" => choice("What is this conversation about?", ["billing", "shipping", "other"])];
        model = "jev-latest")
    @test JSON.parse(JSON.json(req19)) == _ts_json("19-array-state.request.json")

    # Only `state`, `model` and `questions` ever reach the wire: the server
    # rejects any other top-level key with 400.
    @test sort(collect(keys(JSON.parse(JSON.json(_ts_request_02()))))) ==
          ["model", "questions", "state"]

    # Choice option order on the wire is the caller's insertion order, not
    # alphabetical and not hash order.
    s = JSON.json(_ts_request_02())
    pos(k) = first(findfirst(k, s))
    @test pos("\"billing\"") < pos("\"technical\"") < pos("\"sales\"")
    s2 = JSON.json(SystemOneRequest("x",
        ["q" => choice(["zeta", "alpha", "mid"])]; model="jev-latest"))
    pos2(k) = first(findfirst(k, s2))
    @test pos2("\"zeta\"") < pos2("\"alpha\"") < pos2("\"mid\"")
end

@testset "TypeSafe — construction rejects what the server would reject" begin
    # Choice: 1..255 options, non-empty names, known criteria shapes.
    @test length(choice(["a"]).criteria) == 1
    names255 = ["opt$(lpad(i, 3, '0'))" for i in 0:254]
    @test length(choice("pick", names255).criteria) == 255
    @test_throws ArgumentError choice("pick", [names255; "opt255"])          # 256
    @test_throws ArgumentError choice("pick", String[])                      # 0
    @test_throws ArgumentError choice("pick", Dict{String,Any}())
    @test_throws ArgumentError choice("pick", ["" => "blank name"])
    @test_throws ArgumentError choice("pick", ["a" => "x", "a" => "y"])      # duplicate option
    @test_throws ArgumentError choice("pick", "not a criteria map")
    @test_throws ArgumentError choice("pick", (a = 42,))                     # number is not an entry

    # Symbol and String keys both land as String, in order.
    @test collect(keys(choice((a = "A", b = "B")).criteria)) == ["a", "b"]
    @test collect(keys(choice(["b" => "B", :a => "A"]).criteria)) == ["b", "a"]

    # Score: 1..10 levels, never `nothing`.
    @test length(score(["only"]).criteria) == 1
    ten = ["level $i" for i in 0:9]
    @test length(score("rate", ten).criteria) == 10
    @test_throws ArgumentError score("rate", [ten; "level 10"])               # 11
    @test_throws ArgumentError score("rate", [])                             # 0
    @test_throws ArgumentError score("rate", ["ok", nothing])
    @test_throws ArgumentError score("rate", ["ok", 3])
    # The public constructors enforce the same rules as the builders.
    @test_throws ArgumentError UniLM.ScoreQuestion("rate", Any["ok", nothing])
    @test_throws ArgumentError UniLM.ChoiceQuestion("pick",
        (o = JSON.Object{String,Any}(); o[""] = "blank"; o))

    # Noul: instructions or at least one criterion.
    @test noul("Is it urgent?") isa NoulQuestion
    @test noul(; yes = "it is") isa NoulQuestion
    @test noul(; no = "it is not") isa NoulQuestion
    @test_throws ArgumentError noul()
    @test_throws ArgumentError NoulQuestion(nothing, nothing)
    @test_throws ArgumentError NoulQuestion(nothing, NoulCriteria(nothing, nothing))
    @test isnothing(noul("bare").criteria)                                   # no empty criteria object

    # Question naming.
    q = noul("Is it urgent?")
    @test first.(SystemOneRequest("s", [q, q, q]).questions) == ["q1", "q2", "q3"]
    @test first.(SystemOneRequest("s", ("a" => q, :b => q)).questions) == ["a", "b"]
    @test first.(SystemOneRequest("s", (alpha = q, beta = q)).questions) == ["alpha", "beta"]
    @test first.(SystemOneRequest("s", Dict("solo" => q)).questions) == ["solo"]
    @test first.(SystemOneRequest("s", q).questions) == ["q1"]
    @test_throws ArgumentError SystemOneRequest("s", [q, "named" => q])       # mixed bare/named
    @test_throws ArgumentError SystemOneRequest("s", ["dup" => q, "dup" => q])
    @test_throws ArgumentError SystemOneRequest("s", ["" => q])
    @test_throws ArgumentError SystemOneRequest("s", SystemOneQuestion[])
    @test_throws ArgumentError SystemOneRequest("s", (;))
    @test_throws ArgumentError SystemOneRequest("s", ["ok" => "not a question"])
    @test_throws ArgumentError SystemOneRequest("s", 42)

    # Model.
    @test_throws ArgumentError SystemOneRequest("s", q; model = "")
    @test_throws ArgumentError SystemOneRequest("s", q; model = "   ")

    # `state` has no `nothing` member: an unset state is a type error here, not
    # a 422 after a round trip.
    @test_throws Union{MethodError,ArgumentError} SystemOneRequest(nothing, q)

    # Default model comes from the environment, read at call time.
    @test SystemOneRequest("s", q).model == "jev-latest"
    withenv(UniLM.TYPESAFE_DEFAULT_MODEL_ENV => "jev-1.13.0") do
        @test UniLM.default_typesafe_model() == "jev-1.13.0"
        @test SystemOneRequest("s", q).model == "jev-1.13.0"
    end
    withenv(UniLM.TYPESAFE_DEFAULT_MODEL_ENV => "   ") do
        @test UniLM.default_typesafe_model() == "jev-latest"   # blank is not a choice
    end
end

@testset "TypeSafe — response goldens decode into typed answers" begin
    r = UniLM._decode_systemone(_ts_fixture("02-basic-three-questions.response.json"), nothing)
    @test r.model == "jev-1.13.0"
    @test r.usage.prompt_tokens == 445
    @test r.usage.completion_tokens == 72
    @test r.usage.total_tokens == 517
    @test sort(collect(keys(r))) == ["department", "is_frustrated", "urgency"]

    dept = r["department"]
    @test dept isa ChoiceAnswer
    @test dept.choice == "billing"
    @test dept.confidence == 1.0
    @test dept.probabilities == Dict("billing" => 1.0, "sales" => 0.0, "technical" => 0.0)
    @test sprint(show, dept) == "ChoiceAnswer(\"billing\", confidence=1.0)"

    urg = r["urgency"]
    @test urg isa ScoreAnswer
    @test urg.score == 1.98
    @test urg.confidence == 0.96
    @test urg.legend isa Dict{Int,Any} && length(urg.legend) == 3
    @test urg.legend[0] == "Can wait"
    @test urg.probabilities isa Dict{Int,Float64}
    @test argmax(urg.probabilities) == 2
    @test sprint(show, urg) == "ScoreAnswer(1.98, confidence=0.96)"

    frus = r["is_frustrated"]
    @test frus isa NoulAnswer
    @test frus.noul == 0.98
    @test sprint(show, frus) == "NoulAnswer(0.98)"
    # A Noul answer carries no confidence and no distribution — the value is both.
    @test !hasproperty(frus, :confidence)

    # Lookup is by name, through either spelling, on the response or the success.
    ok = SystemOneSuccess(r)
    @test answer(ok, :department).choice == "billing"
    @test ok["urgency"].score == 1.98
    @test haskey(ok, "urgency") && !haskey(ok, "nope")
    @test answers(ok) === r.answers
    @test_throws KeyError ok["nope"]

    # Structured rubrics come back as the original objects, not strings.
    r3 = UniLM._decode_systemone(_ts_fixture("03-structured-questions.response.json"), nothing)
    @test all(v -> v isa AbstractDict, values(r3["refund_size"].legend))
    @test r3["refund_size"].legend[0]["summary"] == "Under \$25"

    # 255 options: every option keeps a probability, and they sum to ~1.
    r29 = UniLM._decode_systemone(_ts_fixture("29-choice-255-options.response.json"), nothing)
    @test length(r29["pick"].probabilities) == 255
    @test sum(values(r29["pick"].probabilities)) ≈ 1.0 atol = 0.01

    # 10 levels: keys are the 0-based level numbers, as Ints.
    r31 = UniLM._decode_systemone(_ts_fixture("31-score-10-levels.response.json"), nothing)
    rate = r31["rate"]
    @test length(rate.probabilities) == 10
    @test length(rate.legend) == 10
    @test argmax(rate.probabilities) isa Int
    @test argmax(rate.probabilities) == 9
    @test rate.legend[argmax(rate.probabilities)] == "level 9"

    # An unrecognised answer type is preserved, never dropped.
    unknown = UniLM._decode_systemone(
        """{"model":"jev-9","answers":{"q1":{"type":"rank","rank":["a","b"]}},
            "usage":{"input_tokens":1,"output_tokens":2}}""", nothing)
    @test unknown["q1"] isa UnknownAnswer
    @test unknown["q1"].type == "rank"
    @test unknown["q1"].raw["rank"] == ["a", "b"]

    # A required field that is absent is malformed provider output, not a zero.
    broken = """{"model":"jev-1.13.0","answers":{"department":{"type":"choice","choice":"billing",
                 "probabilities":{"billing":1.0}}},"usage":{"input_tokens":1,"output_tokens":1}}"""
    @test_throws ArgumentError UniLM._decode_systemone(broken, nothing)
    err = try UniLM._decode_systemone(broken, nothing) catch e; e end
    @test contains(sprint(showerror, err), "confidence")
    for missing_field in ("""{"type":"noul"}""",
                          """{"type":"score","score":1.0,"confidence":0.5,"legend":{"0":"a"}}""",
                          """{"type":"choice","confidence":1.0,"probabilities":{"a":1.0}}""")
        body = """{"model":"m","answers":{"q":$missing_field},"usage":{}}"""
        @test_throws ArgumentError UniLM._decode_systemone(body, nothing)
    end
    # usage is relaxed on read, as the official SDKs relax it.
    relaxed = UniLM._decode_systemone(
        """{"model":"m","answers":{"q":{"type":"noul","noul":0.5}}}""", nothing)
    @test relaxed.usage.prompt_tokens == 0 && relaxed.usage.total_tokens == 0
    # A broken 200 becomes a call error, never a success.
    bad, _ = _ts_mock(; status=200, body=broken) do
        ask(_ts_request_02(); config=_TS_MOCK_CFG)
    end
    @test bad isa SystemOneCallError
    @test contains(bad.error, "confidence")
end

@testset "TypeSafe — ask over a mock server puts the captured request on the wire" begin
    result, seen = _ts_mock(; status=200,
        body=_ts_fixture("02-basic-three-questions.response.json"),
        headers=["Content-Type" => "application/json",
                 "x-typesafe-request-id" => "req_test"]) do
        ask(_ts_request_02(); config=_TS_MOCK_CFG)
    end
    @test result isa SystemOneSuccess
    @test issuccess(result)
    @test result.response.request_id == "req_test"
    @test result["department"].choice == "billing"
    @test result.response.model == "jev-1.13.0"

    @test length(seen) == 1
    req = seen[1]
    @test req["method"] == "POST"
    @test req["target"] == "/v1/systemone"
    @test JSON.parse(req["body"]) == _ts_json("02-basic-three-questions.request.json")

    h = req["headers"]
    @test startswith(h["authorization"], "Bearer ")
    @test h["accept"] == "application/json"
    @test h["content-type"] == "application/json"
    @test startswith(h["user-agent"], "UniLM.jl/")
    @test startswith(h["x-typesafe-sdk"], "UniLM.jl/")
    @test startswith(h["x-typesafe-runtime"], "julia/")
    # No retry-count header is ever sent: this client's retries are the package's
    # shared seam, which does not label attempts.
    @test !haskey(h, "x-typesafe-retry-count")

    # The keyword form builds the same request.
    result2, seen2 = _ts_mock(; status=200,
        body=_ts_fixture("02-basic-three-questions.response.json")) do
        ask("Help! My payouts have been failing for 3 days and nobody has replied to my emails.",
            "department" => choice("Which team should handle this ticket?", (
                billing   = "Payments, invoicing, payouts, refunds",
                technical = "Bugs, outages, integrations",
                sales     = "Pricing, upgrades, new accounts")),
            "urgency" => score("How urgent is this ticket?",
                ["Can wait", "Needs attention this week", "Needs attention today"]),
            "is_frustrated" => noul("Is the customer frustrated?";
                yes = "The customer expresses frustration or impatience",
                no  = "The customer is neutral or satisfied");
            model = "jev-latest", config = _TS_MOCK_CFG)
    end
    @test result2 isa SystemOneSuccess
    @test JSON.parse(seen2[1]["body"]) == _ts_json("02-basic-three-questions.request.json")
    @test isnothing(result2.response.request_id)   # header absent → nothing, not ""
end

@testset "TypeSafe — every error shape becomes a typed failure" begin
    cases = [
        # (fixture, status, expected error_type, a substring of the message)
        ("06-err-no-auth.response.json", 403, "authentication_error", "Must supply an API key!"),
        ("07-err-invalid-key.response.json", 401, "authentication_error", "Cannot authenticate with the server."),
        ("08-err-choice-missing-criteria.response.json", 422, nothing,
            "questions.department.choice.criteria: Field required"),
        ("10-err-empty-questions.response.json", 422, nothing,
            "questions: Dictionary should have at least 1 item"),
        ("09-err-unknown-model.response.json", 400, "api_usage_error", "Unknown model: not-a-real-model"),
        ("18-instructions-omitted.response.json", 400, nothing,
            "Noul question must have criteria or instructions: bare_noul"),
        ("30-choice-256-options.response.json", 400, nothing,
            "Too many choices. Must have at most 255 choices."),
        ("22-err-unknown-path.response.json", 404, nothing, "Not Found"),
    ]
    for (fixture, status, error_type, needle) in cases
        body = _ts_fixture(fixture)
        r, seen = _ts_mock(; status=status, body=body,
            headers=["Content-Type" => "application/json",
                     "x-typesafe-request-id" => "req_err"]) do
            ask(_ts_request_02(); config=_TS_MOCK_CFG)
        end
        @test r isa SystemOneFailure
        @test r.status == status
        @test r.error_type == error_type
        @test contains(r.message, needle)
        @test r.response == body            # the raw body is kept verbatim
        @test r.request_id == "req_err"
        @test length(seen) == 1             # none of these statuses is retried
        @test !issuccess(r)

        # A failed call has no answers, and must not read as a call that answered none.
        @test_throws UniLM.SystemOneError answers(r)
        @test_throws UniLM.SystemOneError answer(r, "department")
        @test_throws UniLM.SystemOneError r["department"]
        e = try answers(r) catch err; err end
        @test e isa UniLM.SystemOneError
        @test contains(sprint(showerror, e), string(status))
        @test contains(sprint(showerror, e), needle)
    end

    # A non-JSON body still yields a message: the raw text, truncated.
    r, _ = _ts_mock(; status=502, body="<html>bad gateway</html>",
                    headers=["Content-Type" => "text/html"]) do
        ask(_ts_request_02(); config=_TS_MOCK_CFG)
    end
    @test r isa SystemOneFailure && r.status == 502
    @test r.message == "<html>bad gateway</html>"
    @test isnothing(r.error_type)
    @test length(UniLM._typesafe_error_message("x"^500)) == 200

    # A call error has no answers either.
    ce = SystemOneCallError(error="connect refused")
    @test_throws UniLM.SystemOneError answers(ce)
    @test contains(sprint(showerror, UniLM.SystemOneError(ce)), "connect refused")
end

@testset "TypeSafe — verbs honour the config/timeout seam" begin
    @test _reached_seam(ask("x", noul("q?"); service=SeamProbe, config=_TINY_DEADLINE),
                        UniLM.SystemOneCallError)
    @test _reached_seam(list_models(; service=SeamProbe, config=_TINY_DEADLINE),
                        UniLM.SystemOneCallError)
end

@testset "TypeSafe — capability routing refuses the wrong surface up front" begin
    @test UniLM.provider_capabilities(TYPESAFEServiceEndpoint) == Set([:system_one, :models])
    @test has_capability(TYPESAFEServiceEndpoint, :system_one)
    @test !has_capability(TYPESAFEServiceEndpoint, :chat)

    # TypeSafe verbs reject a provider that does not serve them, before any request.
    @test_throws ArgumentError ask("x", noul("q?"); service=OPENAIServiceEndpoint)
    @test_throws ArgumentError list_models(; service=OPENAIServiceEndpoint)
    err = try ask("x", noul("q?"); service=OPENAIServiceEndpoint) catch e; e end
    @test contains(sprint(showerror, err), "TypeSafe System One API is not supported")

    # And the chat/agentic/embedding verbs reject this endpoint, loudly, with no
    # network: a System One model answers questions, it does not generate text.
    chat = Chat(model="jev-latest", service=TYPESAFEServiceEndpoint)
    push!(chat, Message(Val(:system), "sys"))
    push!(chat, Message(Val(:user), "hello"))
    @test_throws ArgumentError chatrequest!(chat)
    @test_throws ArgumentError respond("hi"; model="jev-latest", service=TYPESAFEServiceEndpoint)
    @test_throws ArgumentError embeddingrequest!(
        Embeddings("hi"; model="jev-latest", service=TYPESAFEServiceEndpoint))
    @test_throws ArgumentError moderate("hi"; service=TYPESAFEServiceEndpoint)
    # Whatever they do, it is never a success.
    for f in (() -> chatrequest!(chat),
              () -> respond("hi"; model="jev-latest", service=TYPESAFEServiceEndpoint),
              () -> embeddingrequest!(Embeddings("hi"; model="jev-latest", service=TYPESAFEServiceEndpoint)))
        out = try f() catch e; e end
        @test !(out isa LLMSuccess)
        @test out isa Exception || !issuccess(out)
    end
end

@testset "TypeSafe — configuration comes from the environment" begin
    withenv(UniLM.TYPESAFE_API_KEY => nothing) do
        r = ask("x", noul("q?"))
        @test r isa SystemOneCallError
        @test contains(r.error, "TYPESAFE_API_KEY")
        m = list_models()
        @test m isa SystemOneCallError
        @test contains(m.error, "TYPESAFE_API_KEY")
        # The throw is an ArgumentError, not the KeyError a bare ENV lookup raises.
        @test_throws ArgumentError UniLM.auth_header(TYPESAFEServiceEndpoint)
    end
    withenv(UniLM.TYPESAFE_API_KEY => "   ") do
        @test_throws ArgumentError UniLM.auth_header(TYPESAFEServiceEndpoint)
    end
    withenv(UniLM.TYPESAFE_API_KEY => "  ts-key  ") do
        h = Dict(UniLM.auth_header(TYPESAFEServiceEndpoint))
        @test h["Authorization"] == "Bearer ts-key"     # surrounding whitespace stripped
        @test h["X-TypeSafe-SDK"] == h["User-Agent"]
    end

    withenv(UniLM.TYPESAFE_BASE_URL_ENV => nothing) do
        @test UniLM._resolve_base_url(TYPESAFEServiceEndpoint) == "https://api.typesafe.ai"
    end
    withenv(UniLM.TYPESAFE_BASE_URL_ENV => "http://127.0.0.1:1/") do
        @test UniLM._resolve_base_url(TYPESAFEServiceEndpoint) == "http://127.0.0.1:1"
    end
    withenv(UniLM.TYPESAFE_BASE_URL_ENV => "  ") do
        @test UniLM._resolve_base_url(TYPESAFEServiceEndpoint) == "https://api.typesafe.ai"
    end
end

@testset "TypeSafe — usage and cost accounting" begin
    ok = SystemOneSuccess(
        UniLM._decode_systemone(_ts_fixture("02-basic-three-questions.response.json"), nothing))
    u = token_usage(ok)
    @test u.prompt_tokens == 445
    @test u.completion_tokens == 72
    @test u.total_tokens == 517
    # Only input tokens are billed; the output rate is 0.0.
    @test estimated_cost(ok) == 445 * 0.042 / 1e6
    @test estimated_cost(ok; model="jev-latest") == estimated_cost(ok)
    @test estimated_cost(ok; model="not-priced") == 0.0
    @test token_usage(SystemOneFailure(response="{}", status=500)) == TokenUsage()
    @test estimated_cost(SystemOneFailure(response="{}", status=500)) == 0.0
    @test token_usage(SystemOneCallError(error="boom")) == TokenUsage()
end

@testset "TypeSafe — models listing" begin
    result, seen = _ts_mock(; status=200, body=_ts_fixture("01-models-list.response.json")) do
        list_models(; config=_TS_MOCK_CFG)
    end
    @test result isa TypeSafeModelsSuccess
    @test issuccess(result)
    @test length(result.models) == 2
    @test [m.name for m in result.models] == ["jev-latest", "jev-preview"]
    @test startswith(result.models[1].description, "The latest iteration")
    # release_date is kept opaque: live returns RFC 3339, the docs promise a date.
    @test result.models[1].release_date == "2026-09-10T18:38:01.391457+00:00"
    @test result.models[1].raw["name"] == "jev-latest"
    @test seen[1]["method"] == "GET"
    @test seen[1]["target"] == "/v1/models"
    @test isempty(seen[1]["body"])

    # No auth → the same typed failure as the evaluation endpoint.
    fail, _ = _ts_mock(; status=403, body=_ts_fixture("06-err-no-auth.response.json")) do
        list_models(; config=_TS_MOCK_CFG)
    end
    @test fail isa SystemOneFailure
    @test fail.status == 403
    @test fail.error_type == "authentication_error"

    # A 200 that is not a models envelope is a call error, not an empty list.
    bad, _ = _ts_mock(; status=200, body="""{"data":[]}""") do
        list_models(; config=_TS_MOCK_CFG)
    end
    @test bad isa SystemOneCallError
end
