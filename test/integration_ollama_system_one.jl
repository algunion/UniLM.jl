# ─── Ollama System One live witnesses ────────────────────────────────────────
# Requires UNILM_LIVE_OLLAMA=1 and a reachable Ollama server, 0.35 or later (OLLAMA_HOST,
# else 127.0.0.1:11434), with the decision models UNILM_OLLAMA_DECISION_MODELS lists
# (comma-separated; by default the nine below) installed: a listed model that is not
# installed fails. Local models: no spend. Every request goes through an in-process relay
# that counts the System One requests reaching the server and the requests in flight at
# once, so a witness can say how many were sent. The models have 0.6B to 9B parameters:
# states are clear-cut, and assertions are on shapes, ranges and those routes only; the caps
# and messages asserted were measured on Ollama 0.40.6. Each witness names the outcome that
# would refute it.

using Sockets

const _OS1L_MODELS = String.(strip.(split(get(ENV, "UNILM_OLLAMA_DECISION_MODELS",
    "tev1:0.8b,tev1:4b,kev:0.8b-q8_0,kev:9b-q4_K_M,strands-decider:2b,nimble:9b,d1-omni:600m,d1:3b,clef-flash:9b"), ',')))
const _OS1L_EP = get(ENV, "UNILM_LIVE_OLLAMA", "") == "1" ? OllamaEndpoint() : nothing
const _OS1L_INFO = isnothing(_OS1L_EP) ? nothing :
    let r = list_models(; service=_OS1L_EP, config=RequestConfig(max_attempts=1, request_timeout=5.0))
        r isa OllamaSuccess || error("Ollama System One live witnesses: listing the installed models failed: $(r)")
        Dict(m.name => m for m in r.response)
    end

if isnothing(_OS1L_EP)
    @info "Skipping Ollama System One live witnesses (set UNILM_LIVE_OLLAMA=1 and run an Ollama server with " *
          join(_OS1L_MODELS, ", ") * " installed)"
else

const _OS1L_INFLIGHT = Threads.Atomic{Int}(0)                 # requests inside the relay now
const _OS1L_PEAK = Threads.Atomic{Int}(0)                     # the most at once since the last reset
# The last /v1/systemone exchange through the relay: the bytes received, the status and the reply.
const _OS1L_LAST = Ref((request=UInt8[], status=0, response=UInt8[]))
const _OS1L_LAST_LOCK = ReentrantLock()
_os1l_last() = @lock _OS1L_LAST_LOCK _OS1L_LAST[]

# The relay: forwards every request to the server through the package's own request seam,
# counts those to /v1/systemone, keeps the last of them, and the peak of requests in flight.
function _os1l_relay(f::Function, upstream::String)
    sent = Threads.Atomic{Int}(0)
    cfg = RequestConfig(request_timeout=120.0, total_deadline=120.0, max_attempts=1)
    handler = function (req)
        Threads.atomic_max!(_OS1L_PEAK, Threads.atomic_add!(_OS1L_INFLIGHT, 1) + 1)
        try
            target = String(req.target)
            target == "/v1/systemone" && Threads.atomic_add!(sent, 1)
            body = applicable(copy, req.body) ? Vector{UInt8}(copy(req.body)) : UInt8[]
            received = copy(body)
            r = UniLM._http(String(req.method), upstream * target, ["Content-Type" => "application/json"], body; cfg)
            target == "/v1/systemone" &&
                @lock _OS1L_LAST_LOCK (_OS1L_LAST[] = (request=received, status=Int(r.status), response=copy(r.body)))
            HTTP.Response(r.status, ["Content-Type" => "application/json"], r.body)
        finally
            Threads.atomic_sub!(_OS1L_INFLIGHT, 1)
        end
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
const _OS1L_27 = Pair{String,String}[]                        # model => what a 27-option Choice got
const _OS1L_PARITY = Tuple{String,Int,Float64}[]              # model, choices that differ, largest |Δp|

function _os1l_timed(f::Function, model::String, witness::String, sent::Threads.Atomic{Int})
    before = sent[]
    t = @elapsed out = f()
    push!(_OS1L_LOG, (model, witness, sent[] - before, t))
    out
end

_os1l_drift!(ps) = (_OS1L_DRIFT[] = max(_OS1L_DRIFT[], abs(sum(ps) - 1)); ps)
_os1l_outcome(r) = r isa SystemOneSuccess ? "answered, $(length(r["pick"].probabilities)) probabilities" :
                   r isa SystemOneFailure ? "HTTP $(r.status): $(r.message)" : "call error: $(r.error)"

const _OS1L_TICKETS = JSON.parse(read(joinpath(@__DIR__, "fixtures", "ollama", "parity_tickets.json"), String))
const _OS1L_IMAGE = joinpath(@__DIR__, "fixtures", "ollama", "red_circle.png")

const _OS1L_BLUE = joinpath(@__DIR__, "fixtures", "ollama", "blue_square.png")   # a blue square on white

# Vision models whose answers about what an image shows — a yes/no about it, its colour — are
# model-quality claims this suite does not assert for them: their questions are still sent, the
# wiring is asserted, and the values are reported. Observed with Ollama 0.40.6 on an Apple M4 Max:
# d1-omni:600m answers noul 5.97e-7 to "Is there a red shape in the image?" for the red-circle image
# while its own colour choice on the same request says red 0.996; for a 300-pixel red square its
# colour choice says blue 0.954 (red 0.0396, green 0.00661) while that noul says 0.999; for a
# 300-pixel blue square, blue 0.51 against green 0.42 (red 0.0695), and "Is there a blue shape in the
# image?" 0.00125. d1:3b and clef-flash:9b answer the noul and the colours of all three correctly.
const _OS1L_NOUL_ON_IMAGE_EXCEPT = ("d1-omni:600m",)
const _OS1L_COLOUR_ON_IMAGE_EXCEPT = ("d1-omni:600m",)

@testset "Ollama System One live — $model" for model in _OS1L_MODELS
@test haskey(_OS1L_INFO, model)                               # listed, so it must be installed
haskey(_OS1L_INFO, model) || continue
info = _OS1L_INFO[model]
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
    if info.runner === :mlx
        # Ollama's MLX decision encodings report 1 − H(p)/ln n; the surface's confidence
        # replaces it, the raw keeps it (for tev1:0.8b's split on this ticket, measurably).
        @test d.raw["confidence"] ≈ _os1l_entropy(ps) atol = 1e-9
        model == "tev1:0.8b" && @test abs(d.confidence - d.raw["confidence"]) > 1e-3
    else
        # Kev and the models on the llama.cpp path report TypeSafe's formula themselves.
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

@testset "route bounds: refused locally, or the model's own answer" begin
    # Refuted by: a request reaching the server, or anything but an ArgumentError naming Ollama.
    for (witness, call) in (
            "256-option Choice" => () -> ask("s", "pick" => choice("Which?", ["o$i" => "option $i" for i in 1:256]); service=ep, model),
            "object criterion"  => () -> ask("s", "pick" => choice("Which?", ["a" => Dict("d" => "A"), "b" => nothing]); service=ep, model),
            "no model"          => () -> ask("s", noul("Is it?"); service=ep))
        e = _os1l_timed(model, witness, sent) do
            try call(); nothing catch x; x end
        end
        @test e isa ArgumentError && contains(e.msg, "Ollama")
        @test last(_OS1L_LOG)[3] == 0
    end
    # Up to 255 options, the route's bound, are sent. A model may cap lower and says so itself, as
    # measured on Ollama 0.40.6: on MLX, the generic encoder (tev1, nimble) and Clef's answer 400
    # "criteria must contain 2–26 candidates", and Tev1's prompt, built after that check, 400 "Tev1
    # supports at most 24 candidates"; Strands' encoder and the llama.cpp path (kev, d1) take 27.
    cap = startswith(model, "tev1:") ? 24 : startswith(model, "nimble:") || startswith(model, "clef-flash:") ? 26 : 255
    for n in (26, 27)
        r = _os1l_timed(model, "$n-option Choice", sent) do
            ask("Option 7 is the right one.", "pick" => choice("Which option is the right one?",
                                                                ["o$i" => "option $i" for i in 1:n]); service=ep, model)
        end
        @test last(_OS1L_LOG)[3] == 1
        n == 27 && push!(_OS1L_27, model => _os1l_outcome(r))
        if n <= cap
            @test r isa SystemOneSuccess && length(r["pick"].probabilities) == n
            r isa SystemOneSuccess && _os1l_drift!(collect(values(r["pick"].probabilities)))
        else
            @test r isa SystemOneFailure && r.status == 400 &&
                  contains(r.message, n > 26 ? "2–26 candidates" : "at most 24 candidates")
        end
    end
end

@testset "a model that is not installed says to pull it" begin
    # Refuted by: anything but a 404 failure whose message names `ollama pull`.
    f = _os1l_timed(model, "missing model", sent) do
        ask("s", noul("Is it?"); service=ep, model="no-such-decision-model:0b")
    end
    @test f isa SystemOneFailure && f.status == 404 && contains(f.message, "ollama pull no-such-decision-model:0b")
end

@testset "wrapper parity: the encoder's bytes are sent, and the answer is the server's reply" begin
    # Refuted by: request bytes other than encode_system_one's reaching the server, or a decoded
    # answer that differs in any way from the reply the relay captured for that request.
    set = _OS1L_TICKETS
    q = choice(set["instructions"], [k => set["descriptions"][k] for k in set["order"]])
    differ, spread = 0, 0.0
    for t in set["items"]
        req = SystemOneRequest(t["text"], ["intent" => q]; model)
        wire = UniLM.encode_system_one(UniLM.OllamaSystemOneProtocol(), ep, req)
        r = ask(req; service=ep)
        x = _os1l_last()
        @test String(copy(x.request)) == wire.body && x.status == 200
        a = JSON.parse(String(copy(x.response)))["answers"]["intent"]
        @test r isa SystemOneSuccess && r["intent"].choice == a["choice"]
        r isa SystemOneSuccess || continue
        ps, want = r["intent"].probabilities, a["probabilities"]
        @test Set(keys(ps)) == Set(keys(want)) && all(ps[k] == want[k] for k in keys(want))
        differ += r["intent"].choice != a["choice"]
        spread = max(spread, maximum(abs(ps[k] - want[k]) for k in keys(want)))
    end
    push!(_OS1L_PARITY, (model, differ, spread))
end

if :vision in info.capabilities
    @testset "vision: an image state answers; an image after text is refused locally" begin
        # Refuted by: a non-success or a malformed answer to an image (any vision model); outside the
        # documented exceptions, a colour choice that does not tell the red circle from the blue
        # square, or a yes/no that misses the red shape; or an image after a text reaching the server.
        img = ImageAttachment(_OS1L_IMAGE)
        colours = ["red" => "red", "green" => "green", "blue" => "blue"]
        for (file, colour) in (_OS1L_IMAGE => "red", _OS1L_BLUE => "blue")
            qs = colour == "red" ? ("red" => noul("Is there a red shape in the image?"),
                                    "colour" => choice("Which colour is the shape?", colours)) :
                                   ("colour" => choice("Which colour is the shape?", colours),)
            r = _os1l_timed(model, "image ($colour): $(join(first.(qs), " + "))", sent) do
                ask(ImageAttachment(file), qs...; service=ep, model)
            end
            # The wiring, for any vision model: one request, answered (HTTP 200), in shape and range.
            @test r isa SystemOneSuccess && last(_OS1L_LOG)[3] == 1 && _os1l_last().status == 200
            r isa SystemOneSuccess || continue
            c = r["colour"]
            @test c isa ChoiceAnswer && c.choice in first.(colours) && Set(keys(c.probabilities)) == Set(first.(colours)) &&
                  all(p -> 0 <= p <= 1, values(c.probabilities)) && 0 <= c.confidence <= 1
            _os1l_drift!(collect(values(c.probabilities)))
            colour == "red" && @test r["red"] isa NoulAnswer && 0 <= r["red"].noul <= 1
            if model in _OS1L_COLOUR_ON_IMAGE_EXCEPT
                @info "$model on the $colour image (colour not asserted): colour = $(c.choice), " *
                      "probabilities = $(c.probabilities)"
            else
                @test c.choice == colour && c.probabilities[colour] > 0.5     # refuted: the model does not see the colour
            end
            colour == "red" || continue
            if model in _OS1L_NOUL_ON_IMAGE_EXCEPT
                @info "$model on the red-circle image (noul not asserted): noul = $(r["red"].noul)"
            else
                @test r["red"].noul > 0.5                                 # refuted: its yes/no misses the red shape
            end
        end
        r2 = ask([img, "A product photo from a customer."], "red" => noul("Is there a red shape?"); service=ep, model)
        @test r2 isa SystemOneSuccess
        e = _os1l_timed(model, "image after text", sent) do
            try ask(["Photo:", img], "red" => noul("Is it red?"); service=ep, model); nothing catch x; x end
        end
        @test e isa ArgumentError && last(_OS1L_LOG)[3] == 0
    end
else
    @testset "no vision: an image is the model's own 400" begin
        # Refuted by: a model without vision answering an image, or a refusal sent as anything
        # but its HTTP 400.
        r = _os1l_timed(model, "image to a text-only model", sent) do
            ask(ImageAttachment(_OS1L_IMAGE), "red" => noul("Is it red?"); service=ep, model)
        end
        @test r isa SystemOneFailure && r.status == 400 && contains(r.message, r"vision|image"i) && last(_OS1L_LOG)[3] == 1
    end
end

if model == "kev:0.8b-q8_0"
    @testset "ask_each: each slot is its own text's answer; one request in flight, two with ntasks=2" begin
        # Refuted by: a non-success, a slot whose choice is not that of its own text asked alone (a
        # reversal included), more than one request in flight by default (Ollama's width is 1), or
        # never two with ntasks = 2.
        set = _OS1L_TICKETS
        q = choice(set["instructions"], [k => set["descriptions"][k] for k in set["order"]])
        items = set["items"][1:6]
        texts = String[t["text"] for t in items]
        refunds = findall(t -> t["kind"] == "refund", items)
        choices(rs) = [r isa SystemOneSuccess ? r["intent"].choice : nothing for r in rs]
        want = choices([ask(x, "intent" => q; service=ep, model) for x in texts])
        @test length(refunds) == 2 && all(i -> want[i] == "refund", refunds)
        @test !any(isnothing, want) && want != reverse(want)            # the slots tell a reversal apart
        for (ntasks, peak) in ((nothing, 1), (2, 2))
            Threads.atomic_xchg!(_OS1L_PEAK, 0)
            rs = _os1l_timed(model, "ask_each: 6 tickets, ntasks=$(something(ntasks, "default"))", sent) do
                ask_each(texts, "intent" => q; service=ep, model, ntasks)
            end
            @test length(rs) == 6 && all(r -> r isa SystemOneSuccess, rs) && last(_OS1L_LOG)[3] == 6
            @test choices(rs) == want
            @test _OS1L_PEAK[] == peak
        end
    end
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
      join(("\n  $(rpad(m, 18)) $(rpad(w, 36)) requests=$(n)  $(round(t; digits=3)) s" for (m, w, n, t) in _OS1L_LOG)) *
      "\n  largest |Σp − 1| over the answers: $(_OS1L_DRIFT[])" *
      "\n  27-option Choice, per model:" * join(("\n  $(rpad(m, 18)) $(o)" for (m, o) in _OS1L_27)) *
      "\n  wrapper parity, per model (choices that differ, largest |Δp|):" *
      join(("\n  $(rpad(m, 18)) $(d)  $(g)" for (m, d, g) in _OS1L_PARITY))

end # live guard
