# ─── Ollama System One live witnesses ────────────────────────────────────────
# Requires UNILM_LIVE_OLLAMA=1 and a reachable Ollama server, 0.35 or later (OLLAMA_HOST,
# else 127.0.0.1:11434), with the decision models tev1:0.8b and kev:0.8b-q8_0 installed.
# Local models: no spend. Every request goes through an in-process relay that counts the
# System One requests reaching the server, so a witness can say how many were sent. The
# models are 0.8B: states are clear-cut, and assertions are on shapes, ranges and those
# routes only. Each witness names the outcome that would refute it.

using Sockets

const _OS1L_MODELS = ("tev1:0.8b", "kev:0.8b-q8_0")
const _OS1L_EP = get(ENV, "UNILM_LIVE_OLLAMA", "") == "1" ? OllamaEndpoint() : nothing
const _OS1L_TAGS = isnothing(_OS1L_EP) ? nothing : let r = list_models(; service=_OS1L_EP,
                                                                        config=RequestConfig(max_attempts=1, request_timeout=5.0))
    r isa OllamaSuccess ? Set(m.name for m in r.response) : nothing
end

if isnothing(_OS1L_TAGS) || !all(in(_OS1L_TAGS), _OS1L_MODELS)
    @info "Skipping Ollama System One live witnesses (set UNILM_LIVE_OLLAMA=1, run an Ollama server and pull " *
          join(_OS1L_MODELS, " and ") * ")"
else

# The relay: forwards every request to the server through the package's own request seam
# and counts those to /v1/systemone.
function _os1l_relay(f::Function, upstream::String)
    sent = Threads.Atomic{Int}(0)
    cfg = RequestConfig(request_timeout=120.0, total_deadline=120.0, max_attempts=1)
    handler = function (req)
        target = String(req.target)
        target == "/v1/systemone" && Threads.atomic_add!(sent, 1)
        body = applicable(copy, req.body) ? Vector{UInt8}(copy(req.body)) : UInt8[]
        r = UniLM._http(String(req.method), upstream * target, ["Content-Type" => "application/json"], body; cfg)
        HTTP.Response(r.status, ["Content-Type" => "application/json"], r.body)
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
            return f(OllamaEndpoint(base_url="http://127.0.0.1:$port"), sent)
        finally
            close(server)
        end
    end
    error("could not bind an ephemeral port for the Ollama relay")
end

# TypeSafe's published confidence (docs.typesafe.ai/confidence), transcribed, and the
# 1 − H(p)/ln n that Ollama's generic decision encodings report.
_os1l_choice(ps) = (n = length(ps); (maximum(ps) - 1 / n) / (1 - 1 / n))
function _os1l_score(ps)
    n = length(ps)
    m = findfirst(==(maximum(ps)), ps) - 1
    max(0.0, 1 - sum(p * abs(i - m) for (i, p) in zip(0:n-1, ps)) / (sum(abs(i - (n - 1) / 2) for i in 0:n-1) / n))
end
_os1l_entropy(ps) = clamp(1 + sum(p * log(p) for p in ps if p > 0) / log(length(ps)), 0.0, 1.0)

const _OS1L_INTENT = (refund  = "the customer wants a refund",
                      bug     = "the customer reports a software bug",
                      pricing = "the customer asks about prices")
const _OS1L_REFUND = "I was charged twice for my order. Please give me my money back."
const _OS1L_BUG = "The app crashes every time I press the save button."
const _OS1L_PRICE = "How much does the premium plan cost per month?"
_os1l_route(::nl"the customer wants a refund", t) = :refund
_os1l_route(::nl"the customer reports a software bug", t) = :bug
_os1l_route(::nl"the customer asks about prices", t) = :pricing
_os1l_keyed(::Val{:refund}, t) = :refund
_os1l_keyed(::Val{:bug}, t) = :bug
_os1l_keyed(::Val{:pricing}, t) = :pricing

const _OS1L_LOG = Tuple{String,String,Int,Float64}[]          # model, witness, requests, seconds
const _OS1L_DRIFT = Ref(0.0)                                  # largest |Σp − 1| seen

function _os1l_timed(f::Function, model::String, witness::String, sent::Threads.Atomic{Int})
    before = sent[]
    t = @elapsed out = f()
    push!(_OS1L_LOG, (model, witness, sent[] - before, t))
    out
end

_os1l_drift!(ps) = (_OS1L_DRIFT[] = max(_OS1L_DRIFT[], abs(sum(ps) - 1)); ps)

@testset "Ollama System One live — $model" for model in _OS1L_MODELS
_os1l_relay(_OS1L_EP.base_url) do ep, sent

@testset "choice, score and noul in one call; the confidence is the published one" begin
    # Refuted by: more than one request, a non-success, a value out of range, the wrong team
    # for a payouts ticket, or a confidence that is not TypeSafe's formula.
    r = _os1l_timed(model, "ask: choice + score + noul", sent) do
        ask("Help! My payouts have been failing for 3 days.",
            "department" => choice("Which team should handle this ticket?",
                                   (billing="Payments, invoicing, payouts, refunds",
                                    technical="Bugs, outages, integrations",
                                    sales="Pricing, upgrades, new accounts")),
            "urgency" => score("How urgent is this ticket?",
                               ["Can wait", "Needs attention this week", "Needs attention today"]),
            "is_frustrated" => noul("Is the customer frustrated?"); service=ep, model)
    end
    @test r isa SystemOneSuccess && last(_OS1L_LOG)[3] == 1
    res = r.response
    @test res.model == model && isnothing(res.request_id) && res.service === ep
    d, u, f = r["department"], r["urgency"], r["is_frustrated"]
    @test d.choice == "billing"
    ps = _os1l_drift!(Float64[d.probabilities[k] for k in ("billing", "technical", "sales")])
    ls = _os1l_drift!(Float64[u.probabilities[l] for l in 0:2])
    @test 0 <= u.score <= 2 && 0 <= f.noul <= 1 && all(p -> 0 <= p <= 1, [ps; ls])
    @test d.confidence ≈ _os1l_choice(ps) atol = 1e-12
    @test u.confidence ≈ _os1l_score(ls) atol = 1e-12
    if model == "tev1:0.8b"
        # tev1 reports 1 − H(p)/ln n; the surface's confidence replaces it, the raw keeps it.
        @test d.raw["confidence"] ≈ _os1l_entropy(ps) atol = 1e-9
        @test abs(d.confidence - d.raw["confidence"]) > 1e-3
    else
        # Kev reports TypeSafe's formula itself.
        @test abs(d.confidence - d.raw["confidence"]) <= 1e-6
        @test abs(u.confidence - u.raw["confidence"]) <= 1e-6
    end
    u_ = token_usage(r)
    @test u_.prompt_tokens > 0 && u_.total_tokens == u_.prompt_tokens + u_.completion_tokens
    @test (@test_logs estimated_cost(r)) == 0.0          # local: free, and no price warning
end

@testset "the natural-language verbs, by service and model and by scope" begin
    # Refuted by: a route other than the clear-cut one, or a verb sending more than one request.
    by = (; service=ep, model)
    branch_by() = @branch _OS1L_REFUND service=ep model=model begin
        "the customer wants a refund"         => :refund
        "the customer reports a software bug" => :bug
    end
    branch_scoped() = @branch _OS1L_REFUND begin
        "the customer wants a refund"         => :refund
        "the customer reports a software bug" => :bug
    end
    for (verb, route, direct, scoped) in (
            ("nl_classify", :refund, () -> nl_classify(_OS1L_REFUND, _OS1L_INTENT; by...),
                                     () -> nl_classify(_OS1L_REFUND, _OS1L_INTENT)),
            ("nl_dispatch literal", :bug, () -> nl_dispatch(_os1l_route, _OS1L_BUG; by...),
                                          () -> nl_dispatch(_os1l_route, _OS1L_BUG)),
            ("nl_dispatch keyed", :pricing, () -> nl_dispatch(_os1l_keyed, _OS1L_PRICE; texts=_OS1L_INTENT, by...),
                                            () -> nl_dispatch(_os1l_keyed, _OS1L_PRICE; texts=_OS1L_INTENT)),
            ("@branch", :refund, branch_by, branch_scoped))
        @test _os1l_timed(direct, model, "$verb (service, model)", sent) == route
        @test last(_OS1L_LOG)[3] == 1
        @test _os1l_timed(() -> with_system_one(scoped, ep; model), model, "$verb (with_system_one)", sent) == route
        @test last(_OS1L_LOG)[3] == 1
    end
end

@testset "what Ollama rejects is refused here, with no request" begin
    # Refuted by: a request reaching the server, or anything but an ArgumentError naming Ollama.
    for (witness, call) in (
            "27-option Choice" => () -> ask("s", "pick" => choice("Which?", ["o$i" => "option $i" for i in 1:27]); service=ep, model),
            "object criterion" => () -> ask("s", "pick" => choice("Which?", ["a" => Dict("d" => "A"), "b" => nothing]); service=ep, model),
            "no model"         => () -> ask("s", noul("Is it?"); service=ep))
        e = _os1l_timed(model, witness, sent) do
            try call(); nothing catch x; x end
        end
        @test e isa ArgumentError && contains(e.msg, "Ollama")
        @test last(_OS1L_LOG)[3] == 0
    end
    # 26 options, the route's bound, are sent. A model may cap lower and says so itself:
    # tev1 answers 400 "Tev1 supports at most 24 candidates" (Ollama 0.40.6), kev answers.
    r = _os1l_timed(model, "26-option Choice", sent) do
        ask("Option 7 is the right one.", "pick" => choice("Which option is the right one?",
                                                            ["o$i" => "option $i" for i in 1:26]); service=ep, model)
    end
    @test last(_OS1L_LOG)[3] == 1
    if model == "tev1:0.8b"
        @test r isa SystemOneFailure && r.status == 400 && contains(r.message, "at most 24 candidates")
    else
        @test r isa SystemOneSuccess && length(r["pick"].probabilities) == 26
        r isa SystemOneSuccess && _os1l_drift!(collect(values(r["pick"].probabilities)))
    end
end

@testset "a model that is not installed says to pull it" begin
    # Refuted by: anything but a 404 failure whose message names `ollama pull`.
    f = _os1l_timed(model, "missing model", sent) do
        ask("s", noul("Is it?"); service=ep, model="no-such-decision-model:0b")
    end
    @test f isa SystemOneFailure && f.status == 404 && contains(f.message, "ollama pull no-such-decision-model:0b")
end

@testset "keep_alive reaches the server" begin
    # Refuted by: the model still loaded after keep_alive = 0, or not kept after keep_alive = Inf.
    loaded() = Dict(m.name => m.expires_at for m in running_models(; service=_OS1L_EP).response)
    q = "q" => noul("Is the sky blue?")
    r = _os1l_timed(() -> ask("The sky is blue.", q; service=ep, model, keep_alive=Inf), model, "keep_alive = Inf", sent)
    @test r isa SystemOneSuccess && r["q"].noul > 0.5
    @test parse(Int, first(loaded()[model], 4)) > 2100       # Ollama's "forever"
    r = _os1l_timed(() -> ask("The sky is blue.", q; service=ep, model, keep_alive=0), model, "keep_alive = 0", sent)
    @test r isa SystemOneSuccess
    gone = false
    for _ in 1:50                                            # the unload happens after the reply
        gone = !haskey(loaded(), model)
        gone && break
        sleep(0.2)
    end
    @test gone
end

end # relay
end # model testset

@info "Ollama System One live witnesses: requests and latency (relay included)" *
      join(("\n  $(rpad(m, 14)) $(rpad(w, 36)) requests=$(n)  $(round(t; digits=3)) s" for (m, w, n, t) in _OS1L_LOG)) *
      "\n  largest |Σp − 1| over the answers: $(_OS1L_DRIFT[])"

end # live guard
