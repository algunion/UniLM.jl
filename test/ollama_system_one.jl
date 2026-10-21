# test/ollama_system_one.jl — Ollama's /v1/systemone as a System One backend, zero-spend.
#
# The wrapper's wire, its local refusals (each with nothing sent), the two replies
# captured live from Ollama 0.40.6 (tev1:0.8b, kev:0.8b-q8_0), the surface-defined
# confidence, failures, recordings, cost and the natural-language verbs — against an
# in-process server that applies the route's own rules (decision/systemone.go of Ollama
# 0.40.6: ValidateLimits and compileField) and answers as its generic encodings do.

using Sockets

const _OS1_FIXTURES = joinpath(@__DIR__, "fixtures", "ollama")
const _OS1_CFG = RequestConfig(max_attempts=1, total_deadline=30.0)
const _OS1_P = UniLM.OllamaSystemOneProtocol()
const _OS1_MODELS = Dict("tev1:0.8b" => "systemone_tev1.json", "kev:0.8b-q8_0" => "systemone_kev.json")

# The request the live replies answer: three questions about one ticket.
const _OS1_TICKET = "Help! My payouts have been failing for 3 days."
_os1_ticket() = ["department" => choice("Which team should handle this ticket?",
                                        (billing="Payments, invoicing, payouts, refunds",
                                         technical="Bugs, outages, integrations",
                                         sales="Pricing, upgrades, new accounts")),
                 "urgency" => score("How urgent is this ticket?",
                                    ["Can wait", "Needs attention this week", "Needs attention today"]),
                 "is_frustrated" => noul("Is the customer frustrated?")]

_os1_caught(f::Function) = try f(); nothing catch e; e end
_os1_text(req) = applicable(copy, req.body) ? String(copy(req.body)) : ""
_os1_json(status::Int, body::AbstractString) =
    HTTP.Response(status, ["Content-Type" => "application/json; charset=utf-8"], Vector{UInt8}(body))
_os1_error(status::Int, msg::AbstractString) = _os1_json(status, JSON.json(Dict("error" => msg)))

# The route's own validation (Ollama 0.40.6, generic encodings), as the message it answers
# with; `nothing` for a request it compiles.
function _os1_invalid(b::AbstractDict)::Union{Nothing,String}
    qs = get(b, "questions", nothing)
    qs isa AbstractDict && 1 <= length(qs) <= 64 || return "questions must contain 1–64 fields"
    st = get(b, "state", nothing)
    st isa AbstractString && isempty(strip(st)) && return "state must not be empty"
    for (name, q) in qs
        c = get(q, "criteria", nothing)
        q["type"] == "score" && !(c isa AbstractVector && 2 <= length(c) <= 10) &&
            return "question \"$name\": score criteria must be an array of 2–10 levels"
    end
    for (name, q) in qs
        ins = get(q, "instructions", nothing)
        ins isa AbstractString && !isempty(strip(ins)) || ins isa AbstractDict || ins isa AbstractVector ||
            return "question \"$name\": instructions must be a nonempty string, object, or array"
        t, c = q["type"], get(q, "criteria", nothing)
        n = if t == "noul"
            isnothing(c) || all(kv -> kv[1] in ("true", "false") && kv[2] isa AbstractString, c) ||
                return "question \"$name\": noul descriptions must be strings"
            2
        elseif t == "choice"
            all(v -> isnothing(v) || v isa AbstractString, values(c)) ||
                return "question \"$name\": choice criteria must map option keys to descriptions or null"
            length(c)
        elseif t == "score"
            all(v -> v isa AbstractString, c) || return "question \"$name\": score descriptions must be strings"
            length(c)
        else
            return "question \"$name\": type must be choice, noul, or score"
        end
        2 <= n <= 26 || return "question \"$name\": criteria must contain 2–26 candidates"
    end
    nothing
end

# Answer() of the generic encodings: a softmax over the candidates' logits — here 0, −1,
# −2, …, so the first candidate is the likeliest — and confidence 1 − H(p)/ln n.
_os1_softmax(n::Int) = (w = [exp(1.0 - j) for j in 1:n]; w ./ sum(w))
_os1_entropy_confidence(p) = clamp(1 + sum(x * log(x) for x in p if x > 0) / log(length(p)), 0.0, 1.0)

function _os1_answer(q::AbstractDict)
    t = q["type"]
    t == "noul" && return Dict{String,Any}("type" => "noul", "noul" => _os1_softmax(2)[2])
    if t == "choice"
        names = collect(keys(q["criteria"]))
        p = _os1_softmax(length(names))
        return Dict{String,Any}("type" => "choice", "choice" => first(names), "confidence" => _os1_entropy_confidence(p),
                                "probabilities" => Dict{String,Any}(zip(names, p)))
    end
    levels = q["criteria"]
    p = _os1_softmax(length(levels))
    Dict{String,Any}("type" => "score", "score" => sum((j - 1) * p[j] for j in eachindex(p)),
                     "confidence" => _os1_entropy_confidence(p),
                     "legend" => Dict{String,Any}(string(j - 1) => levels[j] for j in eachindex(levels)),
                     "probabilities" => Dict{String,Any}(string(j - 1) => p[j] for j in eachindex(p)))
end

# The route: the live reply for the ticket's three questions, a computed one otherwise.
function _os1_route(req)
    target = String(req.target)
    target == "/api/tags" && return _os1_json(200, read(joinpath(_OS1_FIXTURES, "tags.json"), String))
    target == "/v1/systemone" && req.method == "POST" || return HTTP.Response(404, "404 page not found")
    b = JSON.parse(_os1_text(req))
    model = get(b, "model", nothing)
    model isa AbstractString && !isempty(model) || return _os1_error(400, "model is required")
    haskey(_OS1_MODELS, model) || return _os1_error(404, "model '$model' not found")
    invalid = _os1_invalid(b)
    isnothing(invalid) || return _os1_error(400, invalid)
    qs = b["questions"]
    Set(keys(qs)) == Set(["department", "urgency", "is_frustrated"]) &&
        return _os1_json(200, read(joinpath(_OS1_FIXTURES, _OS1_MODELS[model]), String))
    _os1_json(200, JSON.json(Dict{String,Any}(
        "model" => model, "answers" => Dict{String,Any}(name => _os1_answer(q) for (name, q) in qs),
        "usage" => Dict("input_tokens" => 100, "output_tokens" => 0), "prompt_eval_cached_count" => 40)))
end

const _Os1Seen = @NamedTuple{method::String, target::String, body::String}

"Run `f(endpoint, requests)` against a local Ollama answering with `respond(n, req)`."
function _os1_serve(f::Function; respond::Function=(_, req) -> _os1_route(req))
    seen = _Os1Seen[]
    guard = ReentrantLock()
    handler = function (req)
        n = @lock guard begin
            push!(seen, (method=String(req.method), target=String(req.target), body=_os1_text(req)))
            length(seen)
        end
        respond(n, req)
    end
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
            return f(OllamaEndpoint(base_url="http://127.0.0.1:$port"), seen)
        finally
            close(server)
        end
    end
    error("could not bind an ephemeral port for the Ollama System One mock")
end

# TypeSafe's published confidence (docs.typesafe.ai/confidence), transcribed from its
# Python: the oracle the wrapper's own helpers are checked against.
_os1_published_choice(ps) = (n = length(ps); (maximum(ps) - 1 / n) / (1 - 1 / n))
function _os1_published_score(ps)
    n = length(ps)
    m = findfirst(==(maximum(ps)), ps) - 1
    spread = sum(p * abs(i - m) for (i, p) in zip(0:n-1, ps))
    even = sum(abs(i - (n - 1) / 2) for i in 0:n-1) / n
    max(0.0, 1 - spread / even)
end
_os1_levels(d::AbstractDict) = Float64[d[l] for l in 0:length(d)-1]

# ─── Trait and wire ──────────────────────────────────────────────────────────

@testset "Ollama System One — the trait, the capability and the hooks" begin
    ep = OllamaEndpoint(base_url="http://box:11434")
    @test :system_one in provider_capabilities(ep)
    @test UniLM.system_one_protocol(ep) === _OS1_P
    @test UniLM._system_one_namespace(_OS1_P) == "ollama"
    @test UniLM._allows_raw(_OS1_P) && !UniLM._allows_refusal(_OS1_P)
    @test isnothing(UniLM._system_one_request_id(_OS1_P, HTTP.Response(200)))
    @test UniLM._probability_tolerance(_OS1_P, 3) == 3 * 0.5e-4 + 1e-6
    # list_models with Ollama keeps listing the installed models, also as the scope's backend.
    _os1_serve() do ep, seen
        r = with_system_one(() -> list_models(; config=_OS1_CFG), ep; model="tev1:0.8b")
        @test r isa OllamaSuccess && only(seen).target == "/api/tags"
    end
end

@testset "Ollama System One — TypeSafe's JSON form at {host}/v1/systemone, keep_alive last" begin
    req = SystemOneRequest(_OS1_TICKET, _os1_ticket(); model="tev1:0.8b")
    w = withenv(() -> UniLM.encode_system_one(_OS1_P, OllamaEndpoint(), req), "OLLAMA_HOST" => "gpu-box:9999")
    @test (w.method, w.url, w.path) == ("POST", "http://gpu-box:9999/v1/systemone", "/v1/systemone")
    @test isempty(w.parts) && isempty(w.headers) && w.key_extra == ""
    # The same lowering code writes both bodies, byte for byte.
    @test w.body == UniLM.encode_system_one(UniLM.TypeSafeProtocol(), TYPESAFEServiceEndpoint, req).body
    @test collect(keys(JSON.parse(w.body))) == ["state", "model", "questions"]
    @test UniLM.auth_header(OllamaEndpoint()) == ["Content-Type" => "application/json"]   # no key

    body(ep, opts) = UniLM.encode_system_one(_OS1_P, ep, SystemOneRequest("s", noul("Is it?"); model="m", options=opts)).body
    plain = OllamaEndpoint(base_url="http://box:1")
    @test endswith(body(plain, (keep_alive=600,)), "},\"keep_alive\":600}")
    @test JSON.parse(body(plain, (keep_alive=Inf,)))["keep_alive"] == -1          # stay loaded
    @test JSON.parse(body(plain, (keep_alive=0,)))["keep_alive"] == 0             # unload at once
    @test JSON.parse(body(plain, (keep_alive=2.5,)))["keep_alive"] == 2.5
    @test !haskey(JSON.parse(body(plain, (;))), "keep_alive")                     # the server's default
    # The endpoint's keep_alive, which its chat requests carry, applies unless the call sets one.
    held = OllamaEndpoint(base_url="http://box:1", keep_alive=Inf)
    @test JSON.parse(body(held, (;)))["keep_alive"] == -1
    @test JSON.parse(body(held, (keep_alive=30,)))["keep_alive"] == 30
    for bad in (-1, NaN, -Inf, 1e9 + 1, true, "5m", nothing)
        e = _os1_caught(() -> body(plain, (keep_alive=bad,)))
        @test e isa ArgumentError && contains(e.msg, "keep_alive") && contains(e.msg, "Inf")
    end
    e = _os1_caught(() -> body(plain, (runner="mlx", keep_alive=1, think=false)))
    @test e isa ArgumentError && contains(e.msg, "`keep_alive`") && contains(e.msg, "runner, think")
end

@testset "Ollama System One — keep_alive is sent, and left out of what a recording is keyed on" begin
    req = SystemOneRequest("state", ["q" => noul("Is it?")]; model="tev1:0.8b")
    w1 = UniLM.encode_system_one(_OS1_P, OllamaEndpoint(base_url="http://box:1"), req)
    w2 = UniLM.encode_system_one(_OS1_P, OllamaEndpoint(base_url="http://box:1", keep_alive=300), req)
    @test !contains(w1.body, "keep_alive") && endswith(w2.body, ",\"keep_alive\":300}")
    @test w1.record_body == w2.record_body == w1.body
    mktempdir() do dir
        q = "is_urgent" => noul("Does this need attention today?")
        live, sent = _os1_serve() do ep, seen
            r = with_recorded_answers(dir; mode=:record) do
                ask(_OS1_TICKET, q; service=ep, model="tev1:0.8b", config=_OS1_CFG, keep_alive=0)
            end
            (r, only(seen).body)
        end
        @test live isa SystemOneSuccess && JSON.parse(sent)["keep_alive"] == 0
        rec = JSON.parse(read(only(readdir(dir; join=true)), String))
        @test !haskey(rec["request"]["body"], "keep_alive")
        # Replayed with no server whatever the keep_alive: another, the endpoint's own, or none.
        for (ep, kw) in ((OllamaEndpoint(base_url="http://127.0.0.1:1"), (keep_alive=600,)),
                         (OllamaEndpoint(base_url="http://127.0.0.1:1", keep_alive=Inf), (;)),
                         (OllamaEndpoint(base_url="http://127.0.0.1:1"), (;)))
            r = with_recorded_answers(() -> ask(_OS1_TICKET, q; service=ep, model="tev1:0.8b", kw...), dir)
            @test r isa SystemOneSuccess && r.response.raw == live.response.raw
        end
    end
end

# ─── Local refusals ──────────────────────────────────────────────────────────

struct _Os1Odd <: SystemOneQuestion end

@testset "Ollama System One — what Ollama rejects is refused before any I/O" begin
    png = read(joinpath(_OS1_FIXTURES, "red_circle.png"))
    img = ImageAttachment(png)
    q = noul("Is it urgent?")
    many(n) = ["o$i" => "option $i" for i in 1:n]
    rungs(n) = ["level $i" for i in 1:n]
    _os1_serve() do ep, seen
        a(state, qs...; kw...) = () -> ask(state, qs...; service=ep, model="tev1:0.8b", config=_OS1_CFG, kw...)
        cases = [
            "no model"              => (() -> ask("s", q; service=ep), "`model = \"tev1:0.8b\"`"),
            "no model in the scope" => (() -> with_system_one(() -> ask("s", q), ep), "with_system_one(OllamaEndpoint(); model = …)"),
            "another scope's model" => (() -> with_system_one(() -> ask("s", q; service=ep), TYPESAFEServiceEndpoint;
                                                              model="jev-1.13.0"), "no default System One model"),
            "65 questions"          => (a("s", ["q$i" => q for i in 1:65]), "at most 64 questions"),
            "no instructions"       => (a("s", choice(["a", "b"])), "requires `instructions`"),
            "criteria-only noul"    => (a("s", noul(; yes="It is")), "requires `instructions`"),
            "blank instructions"    => (a("s", choice("  ", ["a", "b"])), "are blank"),
            "image instructions"    => (a("s", score(img, ["low", "high"])), "as a string, object or array"),
            "object description"    => (a("s", choice("Which?", ["a" => Dict("d" => "A"), "b" => nothing])), "string or `nothing`"),
            "tuple description"     => (a("s", choice("Which?", (a=(d="A",), b="B"))), "string or `nothing`"),
            "array description"     => (a("s", choice("Which?", ["a" => ["A"], "b" => "B"])), "string or `nothing`"),
            "object noul criterion" => (a("s", noul("Is it?"; no=Dict("d" => "no"))), "Noul criterion as a string"),
            "object score level"    => (a("s", score("How?", ["low", Dict("d" => "high")])), "Score level as a string"),
            "one option"            => (a("s", choice("Which?", ["a"])), "2 to 26 options"),
            "27 options"            => (a("s", choice("Which?", many(27))), "2 to 26 options"),
            "one level"             => (a("s", score("How?", ["only"])), "2 to 10 levels"),
            "11 levels"             => (a("s", score("How?", rungs(11))), "2 to 10 levels"),
            "blank state"           => (a(" \n ", q), "blank `state`"),
            "image state"           => (a(img, q), "not supported by UniLM's Ollama wrapper yet"),
            "image in instructions" => (a("s", noul(["Is this", img, "a circle?"])), "not supported by UniLM's Ollama wrapper yet"),
            "image in a raw field"  => (a("s", RawQuestion(Dict("type" => "noul", "instructions" => img))), "not supported by UniLM's Ollama wrapper yet"),
            "Box"                   => (a(img, box("the circle")), "Box questions are not supported by Ollama"),
            "unknown kind"          => (a("s", "odd" => _Os1Odd()), "Ollama cannot send \"odd\""),
            "unknown option"        => (a("s", q; num_ctx=4096), "one System One option, `keep_alive`"),
        ]
        @testset "$what" for (what, (f, needle)) in cases
            e = _os1_caught(f)
            @test e isa ArgumentError && contains(e.msg, needle)
        end
        @test isempty(seen)                               # nothing reached the server
    end
    # What the wrapper refuses for its shape, the route rejects too (its own rules, above):
    # the local check saves the round trip, it does not narrow what Ollama accepts.
    wire(qs...; state="s") = JSON.parse(JSON.json(UniLM._system_one_json(
        SystemOneRequest(state, UniLM._normalize_varargs(qs)), "tev1:0.8b")))
    for b in (wire(["q$i" => q for i in 1:65]...), wire(choice(["a", "b"])), wire(noul(; yes="It is")),
              wire(choice("  ", ["a", "b"])), wire(choice("Which?", ["a" => Dict("d" => "A"), "b" => nothing])),
              wire(noul("Is it?"; no=Dict("d" => "no"))), wire(score("How?", ["low", Dict("d" => "high")])),
              wire(choice("Which?", ["a"])), wire(choice("Which?", many(27))), wire(score("How?", ["only"])),
              wire(score("How?", rungs(11))), wire(q; state=" \n "))
        @test !isnothing(_os1_invalid(b))
    end
end

@testset "Ollama System One — the edges Ollama accepts are sent and answered" begin
    _os1_serve() do ep, seen
        rs = [ask("s", choice("Which?", ["a" => "A", "b" => nothing]); service=ep, model="tev1:0.8b", config=_OS1_CFG),
              ask("s", choice("Which?", ["o$i" => "option $i" for i in 1:26]); service=ep, model="kev:0.8b-q8_0", config=_OS1_CFG),
              ask("s", score(Dict("ask" => "How urgent?"), ["level $i" for i in 1:10]); service=ep, model="tev1:0.8b", config=_OS1_CFG),
              ask(Dict("ticket" => "x"), noul(["Is", "it?"]; yes="It is", no="It is not"); service=ep, model="kev:0.8b-q8_0", config=_OS1_CFG),
              ask("s", ["q$i" => noul("Is $i odd?") for i in 1:64]; service=ep, model="tev1:0.8b", config=_OS1_CFG)]
        @test all(r -> r isa SystemOneSuccess, rs)
        @test length(seen) == 5 && all(s -> s.target == "/v1/systemone" && s.method == "POST", seen)
        @test length(answers(rs[5])) == 64
    end
end

# ─── Decoding ────────────────────────────────────────────────────────────────

@testset "Ollama System One — the live replies decode, with the published confidence" begin
    _os1_serve() do ep, seen
        for (model, file) in _OS1_MODELS
            r = ask(_OS1_TICKET, _os1_ticket()...; service=ep, model, config=_OS1_CFG)
            @test r isa SystemOneSuccess
            fixture = JSON.parse(read(joinpath(_OS1_FIXTURES, file), String))
            res = r.response
            @test res.model == model && isnothing(res.request_id) && res.service === ep
            @test res.raw == fixture
            d, u, f = r["department"], r["urgency"], r["is_frustrated"]
            @test d isa ChoiceAnswer && d.choice == "billing" && d.probabilities == fixture["answers"]["department"]["probabilities"]
            @test u isa ScoreAnswer && u.score == fixture["answers"]["urgency"]["score"]
            @test u.legend == Dict(0 => "Can wait", 1 => "Needs attention this week", 2 => "Needs attention today")
            @test f isa NoulAnswer && f.noul == fixture["answers"]["is_frustrated"]["noul"]
            # The surface's confidence, from the probabilities; the server's stays in raw.
            @test d.confidence ≈ _os1_published_choice(collect(values(d.probabilities))) atol = 1e-12
            @test u.confidence ≈ _os1_published_score(_os1_levels(u.probabilities)) atol = 1e-12
            @test d.raw["confidence"] == fixture["answers"]["department"]["confidence"]
            @test u.raw["confidence"] == fixture["answers"]["urgency"]["confidence"]
            if startswith(model, "kev")
                # Kev reports TypeSafe's formula itself: ours reproduces it.
                @test abs(d.confidence - d.raw["confidence"]) <= 1e-6
                @test abs(u.confidence - u.raw["confidence"]) <= 1e-6
                @test token_usage(r) == TokenUsage(prompt_tokens=126, total_tokens=126)
            else
                # tev1 reports 1 − H(p)/ln n (0.497 for .760/.240/.0001); the surface's is 0.640.
                @test d.raw["confidence"] ≈ _os1_entropy_confidence(collect(values(d.probabilities))) atol = 1e-12
                @test abs(d.confidence - d.raw["confidence"]) > 0.1
                @test token_usage(r) == TokenUsage(prompt_tokens=416, total_tokens=416, cached_tokens=126)
            end
        end
        @test length(seen) == 2
    end
end

@testset "Ollama System One — TypeSafe's published confidence formulas" begin
    C, S = UniLM._choice_confidence, UniLM._score_confidence
    # The worked examples of docs.typesafe.ai/confidence.
    @test S([0.0, 0.57, 0.43]) ≈ 0.355 atol = 1e-12
    @test S([0.0, 0.5, 0.5]) ≈ 0.25 atol = 1e-12
    @test S([0.5, 0.0, 0.5]) == 0.0                     # torn between the ends: floored
    @test C([0.6, 0.3, 0.1]) ≈ 0.4 atol = 1e-12          # only the top probability counts
    @test C([0.6, 0.2, 0.2]) ≈ 0.4 atol = 1e-12
    @test C([0.0, 0.5, 0.5]) ≈ 0.25 atol = 1e-12        # the Choice formula reads both alike
    @test C(fill(0.25, 4)) == 0.0 && C([1.0, 0.0]) == 1.0 && S([0.0, 1.0, 0.0]) == 1.0
    @test C([1.0]) == 1.0 && S([1.0]) == 1.0           # one outcome, as TypeSafe answers it
    @test C([0.3333, 0.3333, 0.3333]) == 0.0            # rounded below an even split: clamped
    # A tie for the most likely level is the lowest of them, as the published code picks it.
    @test S([0.45, 0.45, 0.1, 0.0, 0.0]) ≈ 1 - 0.65 / 1.2 atol = 1e-12
    for ps in ([0.1, 0.2, 0.7], [0.05, 0.15, 0.3, 0.5], [0.2, 0.2, 0.2, 0.2, 0.2], [0.9, 0.05, 0.03, 0.01, 0.01])
        @test C(ps) ≈ _os1_published_choice(ps) atol = 1e-12
        @test S(ps) ≈ _os1_published_score(ps) atol = 1e-12
    end
    # Against every Choice and Score answer TypeSafe recorded (121): it computes on unrounded
    # probabilities and reports both them and the confidence to two decimals, so ours, from
    # the rounded ones, lands within 0.02 of what it reported (0.88 against 0.86 at worst,
    # for a four-level Score whose two-level-away probability carries the rounding twice).
    worst, checked = 0.0, 0
    recordings = [JSON.parse(read(f, String)) for f in readdir(joinpath(pkgdir(UniLM), "docs", "recorded_answers"); join=true)]
    bodies = Any[r["response"]["body"] for r in recordings if r["request"]["path"] == "/v1/systemone"]
    append!(bodies, [JSON.parse(read(f, String)) for f in readdir(joinpath(@__DIR__, "fixtures", "typesafe"); join=true)
                     if endswith(f, ".response.json")])
    for b in bodies, a in values(get(b, "answers", Dict()))
        a isa AbstractDict && get(a, "type", "") in ("choice", "score") || continue
        ours = a["type"] == "choice" ? C(Float64[values(a["probabilities"])...]) :
               S(Float64[a["probabilities"][string(l)] for l in 0:length(a["probabilities"])-1])
        worst = max(worst, abs(ours - a["confidence"]))
        checked += 1
    end
    @test checked >= 100
    @test worst <= 0.02 + 1e-12                         # 0.88 − 0.86 is 0.02 plus Float64's 1.8e-17
end

@testset "Ollama System One — probabilities may carry Kev's four-decimal rounding, not more" begin
    reply(ps) = (_, _) -> _os1_json(200, JSON.json(Dict(
        "model" => "kev:0.8b-q8_0", "usage" => Dict("input_tokens" => 9, "output_tokens" => 0),
        "answers" => Dict("pick" => Dict("type" => "choice", "choice" => "a", "confidence" => 0.5,
                                         "probabilities" => Dict(zip(["a", "b", "c"], ps)))))))
    ask3(ep) = ask("s", "pick" => choice("Which?", ["a", "b", "c"]); service=ep, model="kev:0.8b-q8_0", config=_OS1_CFG)
    # Each rounded to four decimals, the three fall 1e-4 short of 1: within 3·0.5e-4.
    r = _os1_serve(; respond=reply([0.5001, 0.2999, 0.1999])) do ep, _
        ask3(ep)
    end
    @test r isa SystemOneSuccess
    @test r["pick"].confidence ≈ (3 * 0.5001 - 1) / 2 atol = 1e-12
    # Beyond the rounding of three probabilities, the answer is malformed.
    bad = _os1_serve(; respond=reply([0.5002, 0.2999, 0.1997 - 1.6e-4])) do ep, _
        ask3(ep)
    end
    @test bad isa SystemOneCallError && contains(bad.error, "not 1") && isnothing(bad.request_id)
end

@testset "Ollama System One — a raw question goes out verbatim and its answer gets the surface's confidence" begin
    raw = RawQuestion(JSON.Object{String,Any}("type" => "choice", "instructions" => "Which colour?",
                                              "criteria" => JSON.Object{String,Any}("red" => nothing, "blue" => "the sky")))
    _os1_serve() do ep, seen
        r = ask("A red apple.", "colour" => raw; service=ep, model="tev1:0.8b", config=_OS1_CFG)
        @test JSON.parse(only(seen).body)["questions"]["colour"] == raw.fields
        a = r["colour"]
        @test a isa ChoiceAnswer
        @test a.confidence ≈ _os1_published_choice(collect(values(a.probabilities))) atol = 1e-12
        @test a.raw["confidence"] ≈ _os1_entropy_confidence(collect(values(a.probabilities))) atol = 1e-12
        # A kind the route does not know is the server's to refuse.
        f = ask("s", "r" => RawQuestion(Dict("type" => "rank", "instructions" => "Rank", "criteria" => ["a", "b"]));
                service=ep, model="tev1:0.8b", config=_OS1_CFG)
        @test f isa SystemOneFailure && f.status == 400
        @test f.message == "question \"r\": type must be choice, noul, or score"
    end
    # A Score answer to a raw question whose levels are no rubric has no confidence to compute.
    odd = _os1_serve(; respond=(_, _) -> _os1_json(200, JSON.json(Dict(
            "model" => "tev1:0.8b", "usage" => Dict("input_tokens" => 1, "output_tokens" => 0),
            "answers" => Dict("r" => Dict("type" => "score", "score" => 1.0, "confidence" => 0.5,
                                          "legend" => Dict("0" => "a", "2" => "b"),
                                          "probabilities" => Dict("0" => 0.5, "2" => 0.5))))))) do ep, _
        ask("s", "r" => RawQuestion(Dict("type" => "score", "instructions" => "How?", "criteria" => ["a", "b"]));
            service=ep, model="tev1:0.8b", config=_OS1_CFG)
    end
    @test odd isa SystemOneCallError && contains(odd.error, "not 0 to 1")
end

@testset "Ollama System One — a reply that does not fit the request is a call error" begin
    for (what, answers_) in (
            "a question unanswered" => Dict{String,Any}(),
            "a refusal"             => Dict("q" => Dict("type" => "refusal", "reason" => "no")),
            "the wrong kind"        => Dict("q" => Dict("type" => "noul", "noul" => 0.5)),
            "an unoffered option"   => Dict("q" => Dict("type" => "choice", "choice" => "z", "confidence" => 1.0,
                                                        "probabilities" => Dict("a" => 0.5, "b" => 0.5))))
        r = _os1_serve(; respond=(_, _) -> _os1_json(200, JSON.json(Dict(
                "model" => "tev1:0.8b", "answers" => answers_, "usage" => Dict("input_tokens" => 1, "output_tokens" => 0))))) do ep, _
            ask("s", "q" => choice("Which?", ["a", "b"]); service=ep, model="tev1:0.8b", config=_OS1_CFG)
        end
        @test r isa SystemOneCallError
    end
    # A cached count that is no token count is accounting: it reads as 0, with a warning.
    r = _os1_serve(; respond=(_, _) -> _os1_json(200, JSON.json(Dict(
            "model" => "tev1:0.8b", "answers" => Dict("q" => Dict("type" => "noul", "noul" => 0.9)),
            "usage" => Dict("input_tokens" => 5, "output_tokens" => 0), "prompt_eval_cached_count" => -3)))) do ep, _
        @test_logs (:warn, r"prompt_eval_cached_count") ask("s", "q" => noul("Is it?"); service=ep, model="tev1:0.8b", config=_OS1_CFG)
    end
    @test r isa SystemOneSuccess && token_usage(r) == TokenUsage(prompt_tokens=5, total_tokens=5)
end

# ─── Failures, recordings, cost ──────────────────────────────────────────────

@testset "Ollama System One — failures carry Ollama's message and the fix" begin
    _os1_serve() do ep, seen
        f = ask("s", noul("Is it?"); service=ep, model="nope:1b", config=_OS1_CFG)
        @test f isa SystemOneFailure && f.status == 404 && isnothing(f.request_id) && isnothing(f.error_type)
        @test startswith(f.message, "model 'nope:1b' not found — pull it first: `ollama pull nope:1b`")
        @test f.response == "{\"error\":\"model 'nope:1b' not found\"}"
    end
    for (status, body, message) in ((500, "{\"error\":\"llama runner process has terminated\"}", "llama runner process has terminated"),
                                    (404, "404 page not found", "404 page not found"),       # a server without the route
                                    (413, "{\"error\":\"text and schema must not exceed 64 KiB\"}", "text and schema must not exceed 64 KiB"))
        f = _os1_serve(; respond=(_, _) -> HTTP.Response(status, body)) do ep, _
            ask("s", noul("Is it?"); service=ep, model="tev1:0.8b", config=_OS1_CFG)
        end
        @test f isa SystemOneFailure && f.status == status && f.message == message && f.response == body
    end
    # No server: the call error says how to start one.
    e = ask("s", noul("Is it?"); service=OllamaEndpoint(base_url="http://127.0.0.1:1"), model="tev1:0.8b", config=_OS1_CFG)
    @test e isa SystemOneCallError && contains(e.error, "no Ollama server answered at http://127.0.0.1:1")
end

struct _Os1Gateway <: UniLM.ServiceEndpoint end      # a TypeSafe-compatible gateway, at a dead port
UniLM._resolve_base_url(::Type{_Os1Gateway}) = "http://127.0.0.1:1"
UniLM.auth_header(::Type{_Os1Gateway}) = ["Content-Type" => "application/json"]
UniLM.provider_capabilities(::Type{_Os1Gateway}) = Set([:system_one])

@testset "Ollama System One — recordings are filed under the \"ollama\" namespace" begin
    mktempdir() do dir
        q = "is_urgent" => noul("Does this need attention today?")
        live, seen = _os1_serve() do ep, seen
            r = with_recorded_answers(dir; mode=:record) do
                ask(_OS1_TICKET, q; service=ep, model="tev1:0.8b", config=_OS1_CFG)
            end
            (r, seen)
        end
        @test live isa SystemOneSuccess
        sent = only(seen).body
        @test readdir(dir) == [bytes2hex(UniLM.SHA.sha256("POST /ollama/v1/systemone\n" * sent)) * ".json"]
        @test JSON.parse(read(joinpath(dir, only(readdir(dir))), String))["request"]["path"] == "/ollama/v1/systemone"
        # Replayed with no server and no key.
        again = with_recorded_answers(dir) do
            ask(_OS1_TICKET, q; service=OllamaEndpoint(base_url="http://127.0.0.1:1"), model="tev1:0.8b")
        end
        @test again isa SystemOneSuccess && again["is_urgent"].noul == live["is_urgent"].noul
        # The same body sent over TypeSafe's wire is another backend's request: no replay.
        @test UniLM.encode_system_one(UniLM.TypeSafeProtocol(), _Os1Gateway,
                                      SystemOneRequest(_OS1_TICKET, [q]; model="tev1:0.8b")).body == sent
        @test_throws ReplayMissError with_recorded_answers(dir) do
            ask(_OS1_TICKET, q; service=_Os1Gateway, model="tev1:0.8b")
        end
    end
end

@testset "Ollama System One — a local answer costs nothing, without a price warning" begin
    r = _os1_serve() do ep, _
        ask("s", noul("Is it?"); service=ep, model="tev1:0.8b", config=_OS1_CFG)
    end
    @test (@test_logs estimated_cost(r)) == 0.0
    # The rule is the answering endpoint's locality and the model's tag: an id no price row knows
    # (and no cloud tag marks) warns from elsewhere and stays silent from Ollama.
    hand(service) = SystemOneSuccess(SystemOneResponse("unpriced-local-witness", Dict{String,SystemOneAnswer}(),
                                                       TokenUsage(prompt_tokens=10, total_tokens=10), nothing,
                                                       Dict{String,Any}(), service))
    @test (@test_logs estimated_cost(hand(OllamaEndpoint()))) == 0.0
    @test (@test_logs (:warn, r"no price row") estimated_cost(hand(nothing))) == 0.0
    @test estimated_cost(r; model="jev-latest") > 0.0           # a `model` given is priced as named
end

# ─── Natural-language verbs ──────────────────────────────────────────────────

_os1_intent(::nl"the customer wants a refund", t) = (:refund, t)
_os1_intent(::nl"the customer reports a bug", t) = (:bug, t)
const _OS1_INTENT = (refund = "the customer wants a refund", bug = "the customer reports a bug in the app")
_os1_keyed(::Val{:refund}, t) = (:refund, t)
_os1_keyed(::Val{:bug}, t) = (:bug, t)

@testset "Ollama System One — nl_classify, nl_dispatch and @branch run on a local model" begin
    # The mock's first option is the likeliest: refund everywhere.
    verbs(; kw...) = (nl_classify("my money back", _OS1_INTENT; config=_OS1_CFG, kw...),
                      nl_dispatch(_os1_intent, "my money back"; config=_OS1_CFG, kw...),
                      nl_dispatch(_os1_keyed, "my money back"; texts=_OS1_INTENT, config=_OS1_CFG, kw...))
    _os1_serve() do ep, seen
        expected = (:refund, (:refund, "my money back"), (:refund, "my money back"))
        @test verbs(; service=ep, model="tev1:0.8b") == expected
        @test with_system_one(() -> verbs(), ep; model="kev:0.8b-q8_0") == expected
        b1 = @branch "my money back" service=ep model="tev1:0.8b" config=_OS1_CFG begin
            "the customer wants a refund"                              => :refund
            ("the customer reports a bug", "A defect in the software") => :bug
        end
        b2 = with_system_one(ep; model="kev:0.8b-q8_0") do
            @branch "my money back" config=_OS1_CFG min_confidence=0.2 begin
                "the customer wants a refund" => :refund
                "the customer reports a bug"  => :bug
                _                             => :unsure
            end
        end
        @test b1 == :refund && b2 == :refund
        sent = [JSON.parse(s.body) for s in seen]
        @test length(seen) == 8 && all(s -> s.target == "/v1/systemone", seen)
        @test [b["model"] for b in sent] == [fill("tev1:0.8b", 3); fill("kev:0.8b-q8_0", 3); "tev1:0.8b"; "kev:0.8b-q8_0"]
        @test all(b -> all(q -> !isempty(q["instructions"]), values(b["questions"])), sent)
        # A scope without a model sends nothing and says how to name one.
        e = _os1_caught(() -> with_system_one(() -> nl_classify("x", _OS1_INTENT), ep))
        @test e isa ArgumentError && contains(e.msg, "`model = \"tev1:0.8b\"`") && length(seen) == 8
    end
end

@testset "runner goes on the wire and into the recording key" begin
    e = OllamaEndpoint(base_url="http://box:11434", runner=:mlx)
    w = UniLM.encode_system_one(_OS1_P, e, SystemOneRequest("The sky is blue.", ["q" => noul("Is the sky blue?")];
                                                            model="tev1:0.8b"))
    @test JSON.parse(w.body)["runner"] == "mlx"
    @test JSON.parse(w.record_body)["runner"] == "mlx"     # it changes the probabilities: keyed
end
