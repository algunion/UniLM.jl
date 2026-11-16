# test/gemini_decisions.jl — Gemini's Decisions API as a System One backend, zero-spend.
#
# The wrapper's encoding (the requests captured live in fixtures/gemini_decisions, rebuilt from public
# constructors; every form of state; every refusal before I/O, each tied to the capture of the HTTP 400
# the API answers it with), its decoding (the captured replies, a blocked input in both documented shapes,
# malformed replies, error bodies), the confidence it recomputes, the float tolerance the captures hold to,
# recording, the natural-language verbs, and cost. Local mocks only; nothing here reaches a network.

using Sockets, Logging, Base64

const _GD_CFG = UniLM.RequestConfig(max_attempts=1, total_deadline=30.0)
const _GD_FIXTURES = joinpath(@__DIR__, "fixtures", "gemini_decisions")
# The picture the captures sent (docs/src/assets/red_circle.png, 266 bytes).
const _GD_PNG = ImageAttachment(joinpath(pkgdir(UniLM), "docs", "src", "assets", "red_circle.png"))
const _GD_JPEG = ImageAttachment(UInt8[0xff, 0xd8, 0xff, 0xe0, 0x0d, 0x0a, 0x2d, 0x2d, 0x00, 0x10])
const _GD_TICKET = "The package arrived with a broken screen."

"A live capture: its request as sent, status, headers and reply body, as recorded on 2026-11-16."
_gd_fixture(tag::String) = JSON.parse(read(joinpath(_GD_FIXTURES, tag * ".json"), String))

_gd_caught(f::Function) = try f(); nothing catch e; e end

# An endpoint that speaks the Decisions wire to a local server.
struct _GDMock <: UniLM.ServiceEndpoint end
const _GD_BASE = Ref("")
UniLM._resolve_base_url(::Type{_GDMock}) = _GD_BASE[]
UniLM.auth_header(::Type{_GDMock}) = ["x-goog-api-key" => "gd", "Content-Type" => "application/json"]
UniLM.provider_capabilities(::Type{_GDMock}) = Set([:system_one])
UniLM.system_one_protocol(::Type{_GDMock}) = UniLM.GeminiDecisionsProtocol()

const _GDSeen = @NamedTuple{method::String, target::String, headers::Vector{Pair{String,String}}, body::String}

"Run `f()` against a local server answering request `n` with `respond(n, req)`; return `(f(), requests)`."
function _gd_serve(f::Function, respond::Function)
    seen = _GDSeen[]
    guard = ReentrantLock()
    handler = function (req)
        n = @lock guard begin
            push!(seen, (method=String(req.method), target=String(req.target),
                         headers=Pair{String,String}[String(k) => String(v) for (k, v) in req.headers],
                         body=applicable(copy, req.body) ? String(copy(req.body)) : ""))
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
        _GD_BASE[] = "http://127.0.0.1:$port"
        try
            return f(), seen
        finally
            close(server)
        end
    end
    error("could not bind an ephemeral port for the Gemini Decisions mock")
end

_gd_reply(status::Int, body::AbstractString; headers::Vector{Pair{String,String}}=Pair{String,String}[]) =
    HTTP.Response(status, Pair{String,String}["Content-Type" => "application/json"; headers], Vector{UInt8}(body))

const _GD_USAGE = Dict{String,Any}("total_tokens" => 120, "total_input_tokens" => 120, "total_output_tokens" => 0,
    "input_tokens_by_modality" => [Dict{String,Any}("modality" => "text", "tokens" => 120)])

# A valid reply to every question of a Decisions body: the first option, the lowest level, an even noul, an
# echo of a raw question's type.
function _gd_answers(body::AbstractDict)::Dict{String,Any}
    answers = Dict{String,Any}()
    for (name, q) in body["questions"]
        t = q["type"]
        answers[name] = if t == "noul"
            Dict{String,Any}("type" => t, "noul" => 0.5, "confidence" => 0.5)
        elseif t == "choice"
            options = collect(keys(q["criteria"]))
            Dict{String,Any}("type" => t, "choice" => options[1], "confidence" => 1.0,
                             "probabilities" => Dict{String,Any}(o => i == 1 ? 1.0 : 0.0 for (i, o) in enumerate(options)))
        elseif t == "score"
            n = length(q["criteria"])
            Dict{String,Any}("type" => t, "score" => 0.0, "confidence" => 1.0,
                             "legend" => Dict{String,Any}(string(i - 1) => q["criteria"][i] for i in 1:n),
                             "probabilities" => Dict{String,Any}(string(i - 1) => i == 1 ? 1.0 : 0.0 for i in 1:n))
        else
            Dict{String,Any}("type" => t, "echo" => true)
        end
    end
    Dict{String,Any}("answers" => answers, "usage" => _GD_USAGE)
end

_gd_body(req) = JSON.parse(String(copy(req.body)))
_gd_body(s::_GDSeen) = JSON.parse(s.body)
_gd_valid(n, req) = _gd_reply(200, JSON.json(_gd_answers(_gd_body(req))))
# A blocked input, as the API reference shapes it: no answers, and the reason in `prompt_feedback`.
_gd_blocked(reason="prohibited_content") = (n, req) -> _gd_reply(200, JSON.json(Dict{String,Any}(
    "prompt_feedback" => Dict{String,Any}("block_reason" => reason, "block_reason_message" => "Blocked by policy."),
    "usage" => _GD_USAGE)))

# ─── The trait and its hooks ─────────────────────────────────────────────────

@testset "Gemini Decisions — GEMINIServiceEndpoint speaks the Decisions protocol" begin
    req = SystemOneRequest(_GD_TICKET, "damaged" => noul("Does the customer report a damaged item?"))
    withenv("GEMINI_API_KEY" => nothing) do
        # A scope over an empty recordings directory shows what would be sent, and where it would be filed,
        # without a key or a server: the default body is the one the API answered (W7 sent it to /v1).
        miss = with_recorded_answers(mktempdir()) do
            _gd_caught(() -> ask(req; service=GEMINIServiceEndpoint))
        end
        @test miss isa ReplayMissError
        @test miss.path == "/gemini-decisions/v1beta/decisions" && miss.body == _gd_fixture("W7")["request_bytes"]
        # Live with no key: a call error that names it, before anything is sent.
        r = ask(req; service=GEMINIServiceEndpoint, config=_GD_CFG)
        @test r isa SystemOneCallError && contains(r.error, "GEMINI_API_KEY") && r.request_id === nothing
        pinned = with_recorded_answers(mktempdir()) do
            with_system_one(() -> _gd_caught(() -> ask(req)), GEMINIServiceEndpoint; model="models/gemini-decision-1")
        end
        @test JSON.parse(pinned.body)["model"] == "models/gemini-decision-1"
    end
    p = UniLM.GeminiDecisionsProtocol()
    @test UniLM.system_one_protocol(GEMINIServiceEndpoint) === p
    @test has_capability(GEMINIServiceEndpoint, :system_one) && !has_capability(GEMINIOpenAIServiceEndpoint, :system_one)
    e = _gd_caught(() -> ask(req; service=GEMINIOpenAIServiceEndpoint))
    @test e isa ArgumentError && contains(e.msg, "System One API is not supported by GEMINIOpenAIServiceEndpoint")
    @test UniLM._system_one_default_model(p, GEMINIServiceEndpoint) == "gemini-decision-1"
    @test UniLM._system_one_namespace(p) == "gemini-decisions"
    @test UniLM._allows_refusal(p) && UniLM._allows_raw(p)
    @test UniLM._system_one_default_config(p, GEMINIServiceEndpoint) === UniLM._HOSTED_SYSTEM_ONE_CONFIG
    @test UniLM._system_one_concurrency(p, GEMINIServiceEndpoint) == 8
    @test UniLM._system_one_request_id(p, HTTP.Response(200)) === nothing
    wire = UniLM.encode_system_one(p, GEMINIServiceEndpoint, req)
    @test wire.method == "POST" && wire.url == "https://generativelanguage.googleapis.com/v1beta/decisions"
    @test wire.path == "/v1beta/decisions" && isempty(wire.parts) && isempty(wire.headers) && wire.key_extra == ""
    @test wire.record_body == wire.body
    # The listing of the endpoint is Gemini's models listing: with no key, a call error that names it.
    r = withenv(() -> list_models(; service=GEMINIServiceEndpoint, config=_GD_CFG), "GEMINI_API_KEY" => nothing)
    @test r isa ModelListCallError && contains(r.error, "GEMINI_API_KEY")
end

const _GD = UniLM.GeminiDecisionsProtocol()

"The body `request` is sent as to GEMINIServiceEndpoint."
_gd_wire(request::SystemOneRequest; service=GEMINIServiceEndpoint) = UniLM.encode_system_one(_GD, service, request)
_gd_sent(state, questions...; model=nothing) =
    JSON.parse(_gd_wire(SystemOneRequest(state, collect(Pair{String,SystemOneQuestion}, questions); model)).body)

# ─── Encoding ────────────────────────────────────────────────────────────────

const _GD_TEAM = (billing = "Payments, invoices, and refunds.", technical = "Problems using the product.",
                  shipping = "Delivery and tracking.")

# Each capture the API answered with HTTP 200, as a request built from public constructors.
const _GD_REBUILT = [
    "W1" => (state = "Customer returned the sweater because they preferred green over blue.",
             questions = ["is_defect" => noul("Is the customer reporting a product defect?";
                                              yes = "The product itself is broken or faulty.",
                                              no = "Preference, sizing, shipping, or pricing complaint.")]),
    "W2" => (state = "I was charged twice for my order.",
             questions = ["department" => choice("Which department should handle this request?", (_GD_TEAM..., other = nothing))]),
    "W3" => (state = "Export fails in Safari but works in Chrome.",
             questions = ["severity" => score("How severe is this issue for the user?",
                                              ["Cosmetic: appearance only, no lost functionality.",
                                               "Workaround available: a task fails, but another way works.",
                                               "Fully blocked: a task fails with no workaround."])]),
    "W4" => (state = "Export fails in Safari but works in Chrome. I need this fixed before Friday's board meeting.",
             questions = ["severity" => score("How severe is this issue?", ["Cosmetic", "Workaround available", "Fully blocked"]),
                          "department" => choice("Which department should handle this complaint?",
                                                 (_GD_TEAM..., other = "Requests outside these categories.")),
                          "deadline" => noul("Does the customer mention a deadline?")]),
    "W5" => (state = Any["Inspect this image.", ImageAttachment(_GD_PNG.data; detail = "low")],
             questions = ["red" => noul("Is the shape in the image red?"),
                          "shape" => choice("Which shape does the image show?", ["circle", "square", "triangle"])]),
    "W13" => (state = _GD_TICKET, questions = ["frustrated" => noul(["Is the customer ", "upset about the damage?"])]),
    "W14" => (state = _GD_PNG, questions = ["red" => noul("Is the shape in the image red?")]),
    "W23" => (state = _GD_TICKET, questions = ["c" => choice("Which word fits best?", ["needs review", "café", "x"^128])]),
    "W27" => (state = _GD_TICKET, questions = ["d" => noul("Is the item damaged?"; yes = "It is broken.")])]

_gd_request(tag::String)::SystemOneRequest = (r = Dict(_GD_REBUILT)[tag];
    SystemOneRequest(r.state, collect(Pair{String,SystemOneQuestion}, r.questions)))

@testset "Gemini Decisions — the captured requests rebuild from public constructors, byte for byte" begin
    for (tag, _) in _GD_REBUILT
        fx = _gd_fixture(tag)
        @test fx["status"] == 200 && fx["path"] == "/v1beta/decisions"
        @test _gd_wire(_gd_request(tag)).body == fx["request_bytes"]
    end
    # The prefixed model the API answered too, sent as written.
    w6 = _gd_fixture("W6")
    @test w6["status"] == 200 && _gd_wire(SystemOneRequest(_GD_TICKET, "damaged" => noul("Does the customer report a damaged item?");
                                                           model = "models/gemini-decision-1")).body == w6["request_bytes"]
end

@testset "Gemini Decisions — the state is the input: text, JSON text, or content blocks of text and images" begin
    input(state) = _gd_sent(state, "q" => noul("Is it?"))["input"]
    @test input("plain text") == "plain text" && input(" padded ") == " padded "
    @test input(Dict("b" => 1, "a" => [1, 2])) == """{"a":[1,2],"b":1}"""      # a plain Dict in sorted key order
    @test input(JSON.Object{String,Any}("b" => 1, "a" => 2)) == """{"b":1,"a":2}"""
    @test input((message = "I want my money back",)) == """{"message":"I want my money back"}"""
    @test input(["a", "b"]) == """["a","b"]"""                                 # strings alone are JSON
    @test input(("a", 1, nothing)) == """["a",1,null]"""
    @test input(Dict{String,Any}()) == "{}"
    b64 = base64encode(_GD_PNG.data)
    @test input(_GD_PNG) == [Dict("type" => "image", "mime_type" => "image/png", "data" => b64)]
    mixed = input(Any["first", _GD_PNG, "\n", _GD_JPEG])
    @test [b["type"] for b in mixed] == ["text", "image", "text", "image"] && mixed[3]["text"] == "\n"
    @test mixed[4]["mime_type"] == "image/jpeg" && mixed[4]["data"] == base64encode(_GD_JPEG.data)
    @test input((_GD_PNG, "What is it?")) == input(Any[_GD_PNG, "What is it?"])  # a tuple is taken as a vector
    # The detail level is the block's resolution; "auto" is the API's default and sets none.
    for (detail, resolution) in (("low", "low"), ("high", "high"), ("auto", nothing))
        block = only(input(ImageAttachment(_GD_PNG.data; detail)))
        @test get(block, "resolution", nothing) == resolution
        @test collect(keys(block)) == (isnothing(resolution) ? ["type", "mime_type", "data"] : ["type", "mime_type", "data", "resolution"])
    end
end

# Every request here is refused before any I/O: none reaches the local server.
function _gd_refused(calls::Vector)
    refusals, seen = _gd_serve(_gd_valid) do
        [_gd_caught(f) for f in calls]
    end
    @test isempty(seen)
    @test all(e -> e isa ArgumentError, refusals)
    refusals
end

@testset "Gemini Decisions — each limit the API answers with HTTP 400 is refused before any I/O" begin
    on(state, qs...; kw...) = () -> ask(state, qs...; service=_GDMock, config=_GD_CFG, kw...)
    q = "damaged" => noul("Does the customer report a damaged item?")
    cases = [
        "W8a" => (on(_GD_TICKET, "bad.key" => noul("Is it?")), "takes a question's name as 1 to 64 ASCII letters, digits, `_` and `-`"),
        "W8b" => (on(_GD_TICKET, "one" => choice("Which?", ["only"])), "a choice offers 2 to 255 options, and it has 1. Add one"),
        "W8c" => (on(_GD_TICKET, "one" => score("Rate.", ["only"])), "a score has 2 to 255 levels, and it has 1. Add a level"),
        "W8d" => (on(_GD_TICKET, "dept" => choice("Which?", ["billing ", "technical"])), "begins or ends with whitespace"),
        "W9"  => (on("", q), "takes no empty input"),
        "W15" => (on(_GD_TICKET, "d" => noul(; yes = "Damaged.")), "every question needs `instructions`"),
        "W18" => (on(_GD_TICKET, ["q$i" => noul("Is it?") for i in 0:1000]...), "at most 1000 questions per request; this one has 1001"),
        "W19" => (on(Any["", _GD_PNG], q), "takes no empty text block, and text block 1 of the state is empty"),
        "W20" => (on(_GD_TICKET, "d" => noul("")), "its instructions are empty"),
        "W21" => (on(_GD_TICKET, "d" => noul(String[])), "its instructions hold no text"),
        "W22" => (on("   ", q), "takes no empty input"),
        "W24" => (on(_GD_TICKET, "c" => choice("Which word fits best?", ["tab\there", "ok"])), "holds a control or format character"),
        "W25" => (on(_GD_TICKET, "c" => choice("Which word fits best?", ["no\u00a0break", "ok"])), "a space other than the ASCII one"),
        "W26" => (on(_GD_TICKET, "c" => choice("Which word fits best?", ["x"^129, "ok"])), "is 129 characters (129 bytes)"),
        "W28" => (on(_GD_TICKET, "s" => score("Rate.", ["low", ""])), "its level 1 must be a non-empty string")]
    refusals = _gd_refused([f for (_, (f, _)) in cases])
    for ((tag, (_, words)), e) in zip(cases, refusals)
        @testset "$tag" begin
            fx = _gd_fixture(tag)
            # The API refuses the same request with HTTP 400 naming the rule.
            @test fx["status"] == 400 && JSON.parse(fx["body_text"])["error"]["code"] == "invalid_request"
            @test e isa ArgumentError && contains(e.msg, "Gemini's Decisions API") && contains(e.msg, words)
        end
    end
    # A kind the API does not know is sent as written, and the API's 400 comes back typed (W10).
    w10 = _gd_fixture("W10")
    raw = RawQuestion(JSON.Object{String,Any}("type" => "rank", "instructions" => "Rank them.", "items" => ["a", "b"]))
    @test _gd_wire(SystemOneRequest(_GD_TICKET, "rank" => raw)).body == w10["request_bytes"]
    r, seen = _gd_serve((_, _) -> _gd_reply(400, w10["body_text"])) do
        ask(_GD_TICKET, "rank" => raw; service=_GDMock, config=_GD_CFG)
    end
    @test length(seen) == 1 && r isa SystemOneFailure && r.status == 400 && r.error_type == "invalid_request"
    @test r.message == JSON.parse(w10["body_text"])["error"]["message"]
end

@testset "Gemini Decisions — what the wire has no form for is refused before any I/O" begin
    on(state, qs...; kw...) = () -> ask(state, qs...; service=_GDMock, config=_GD_CFG, kw...)
    q = "q" => noul("Is it?")
    msg = "Hello, I want a refund."
    conversation = [Message(Val(:user), msg), Message(role = RoleAssistant, content = "On its way.")]
    wav = AudioAttachment(joinpath(@__DIR__, "fixtures", "ollama", "secret_word.wav"))
    pdf = DocumentAttachment(vcat(Vector{UInt8}("%PDF-1.4\n"), zeros(UInt8, 8)))
    cases = [
        # the state
        (on(Any[], q), "takes no empty input, and this state is an empty Array"),
        (on((), q), "takes no empty input, and this state is an empty Tuple"),
        (on(conversation, q), "takes no conversation as its input"),
        (on(Any["text", conversation[1]], q), "this state holds a Message. Pass the conversation's text"),
        (on(Dict("photo" => _GD_PNG), q), "holds one nested in JSON"),
        (on(Any[Dict("photo" => _GD_PNG), "text"], q), "holds one nested in JSON"),
        (on((photo = _GD_PNG,), q), "holds one nested in JSON"),
        (on(ImageAttachment(_GD_PNG.data; detail = "original"), q), "no media resolution for an image at its own dimensions"),
        (on(Any["Listen.", wav], q), "UniLM does not send audio in a decision's input"),
        (on(Dict("clip" => wav), q), "UniLM does not send audio in a decision's input"),
        (on(Any["Read.", pdf], q), "not documents"),
        # the questions
        (on(msg, "c" => choice(["a", "b"])), "every question needs `instructions`"),
        (on(msg, "c" => choice(Dict("intent" => "refund"), ["a", "b"])), "the API takes instructions as text only"),
        (on(msg, "c" => choice(_GD_PNG, ["a", "b"])), "An image belongs in the state"),
        (on(msg, "c" => noul(["Is it ", _GD_PNG])), "An image belongs in the state"),
        (on(msg, "c" => noul(["Is it ", ""])), "part 2 of the instructions of \"c\" is empty"),
        (on(msg, "c" => choice("Which?", ["a" => _GD_PNG, "b" => nothing])), "the description of its option \"a\" must be a string"),
        (on(msg, "c" => choice("Which?", ["a" => (why = "x",), "b" => nothing])), "Write it as text"),
        (on(msg, "c" => choice("Which?", ["o$i" for i in 1:256])), "a choice offers 2 to 255 options, and it has 256. Split it"),
        (on(msg, "c" => choice("Which?", ["é"^128 * "x", "b"])), "is 129 characters"),
        (on(msg, "c" => choice("Which?", ["é"^127 * "x", "b"])), nothing),     # 128 characters, 255 bytes: sent
        (on(msg, "c" => choice("Which?", ["line\u2028break", "b"])), "holds a control or format character"),
        (on(msg, "c" => choice("Which?", ["\u2028line", "b"])), "holds a control or format character"),
        (on(msg, "c" => choice("Which?", ["zero\u200bwidth", "b"])), "holds a control or format character"),
        (on(msg, "s" => score("Rate.", ["l$i" for i in 1:256])), "a score has 2 to 255 levels, and it has 256. Merge"),
        (on(msg, "s" => score("Rate.", ["low", (level = 1,)])), "its level 1 must be a non-empty string, its description, not"),
        (on(msg, "n" => noul("Is it?"; yes = Dict("a" => 1))), "its `yes` criterion must be a string"),
        (on(msg, "n" => noul("Is it?"; no = _GD_PNG)), "its `no` criterion must be a string"),
        (on(msg, "b" => bounding_box("the dent")), "it is a Box question"),
        (on(msg, "r" => RawQuestion(Dict("type" => "noul", "instructions" => "x", "img" => _GD_PNG))), "it holds an image"),
        (on(msg, "café" => noul("Is it?")), "\"café\" is not one"),
        (on(msg, "x"^65 => noul("Is it?")), "takes a question's name as 1 to 64"),
        # the options
        (on(msg, q; long_context = true), "takes no provider options"),
        (on(msg, q; prompt_cache_key = "k", safety_identifier = "u"), "got `prompt_cache_key`, `safety_identifier`. Drop them")]
    refused = filter(c -> !isnothing(last(c)), cases)
    refusals = _gd_refused([f for (f, _) in refused])
    for ((_, words), e) in zip(refused, refusals)
        @test e isa ArgumentError && contains(e.msg, words)
    end
    @test all(e -> startswith(e.msg, "Gemini's Decisions API") || contains(e.msg, "GEMINIServiceEndpoint") ||
                   contains(e.msg, "_GDMock") || contains(e.msg, "Gemini has no media resolution"), refusals)
    sent, seen = _gd_serve(_gd_valid) do
        [f() for (f, words) in cases if isnothing(words)]
    end
    @test all(issuccess, sent) && length(seen) == 1
    # A scope refuses its options when it opens, before f runs; a scope without options works.
    ran = Ref(false)
    e = _gd_caught(() -> with_system_one(() -> (ran[] = true), GEMINIServiceEndpoint; keep_alive = 5))
    @test e isa ArgumentError && contains(e.msg, "takes no provider options") && !ran[]
    @test with_system_one(() -> :ok, GEMINIServiceEndpoint) === :ok
    # Inline data: the images' bytes against "up to 100 MB per request", read as MiB.
    @test isnothing(UniLM._check_gemini_inline(100 * 1024^2))
    e = _gd_caught(() -> UniLM._check_gemini_inline(100 * 1024^2 + 1))
    @test e isa ArgumentError && contains(e.msg, "at most 104857600 bytes of inline data") && contains(e.msg, "104857601")
    e = _gd_caught(() -> UniLM._check_gemini_inline(length(_GD_PNG.data); limit = length(_GD_PNG.data) - 1))
    @test e isa ArgumentError
end

@testset "Gemini Decisions — questions are a map keyed by name; instructions as text or text blocks" begin
    body = _gd_sent("Help! My payouts have been failing for 3 days.",
        "dept" => choice("Which team?", (billing = "Payments, invoicing, payouts, refunds", technical = nothing)),
        "urg" => score("How urgent?", ["Can wait", "This week", "Today"]),
        "frus" => noul(("Is the customer ", "frustrated?")),
        "def" => noul("Is it a defect?"; no = "A preference"),
        "bare" => NoulQuestion("Is it bare?", NoulCriteria(nothing, nothing)),
        "rank" => RawQuestion(JSON.Object{String,Any}("type" => "rank", "items" => ["b", "a"])))
    @test collect(keys(body)) == ["model", "input", "questions"] && body["model"] == "gemini-decision-1"
    @test JSON.json(body["questions"]) == "{" * join([
        """"dept":{"type":"choice","instructions":"Which team?","criteria":{"billing":"Payments, invoicing, payouts, refunds","technical":""}}""",
        """"urg":{"type":"score","instructions":"How urgent?","criteria":["Can wait","This week","Today"]}""",
        """"frus":{"type":"noul","instructions":[{"type":"text","text":"Is the customer "},{"type":"text","text":"frustrated?"}]}""",
        """"def":{"type":"noul","instructions":"Is it a defect?","criteria":{"false_description":"A preference"}}""",
        """"bare":{"type":"noul","instructions":"Is it bare?"}""",
        """"rank":{"type":"rank","items":["b","a"]}"""], ",") * "}"
    # Option order is sent as given; the API sorts the keys itself before evaluating them.
    @test collect(keys(_gd_sent("s", "c" => choice("Which?", ["zebra", "apple"]))["questions"]["c"]["criteria"])) == ["zebra", "apple"]
end

# ─── Decoding ────────────────────────────────────────────────────────────────

_gd_decode(request::SystemOneRequest, fx; service=GEMINIServiceEndpoint) =
    UniLM.decode_system_one(_GD, service, request, HTTP.Response(fx["status"], Pair{String,String}[k => v for (k, v) in fx["headers"]],
                                                     Vector{UInt8}(fx["body_text"])))

@testset "Gemini Decisions — the captured replies decode, with TypeSafe's published confidence" begin
    for (tag, _) in _GD_REBUILT
        fx = _gd_fixture(tag)
        r = _gd_decode(_gd_request(tag), fx)
        @test r isa SystemOneSuccess && r.response.model == "gemini-decision-1" && r.response.request_id === nothing
        @test r.response.service === GEMINIServiceEndpoint && r.response.raw == JSON.parse(fx["body_text"])
        wire = JSON.parse(fx["body_text"])["answers"]
        for (name, a) in r.response.answers
            @test a.raw == wire[name]                          # Gemini's answer, its confidence included, kept
            if a isa ChoiceAnswer
                @test a.confidence == UniLM._choice_confidence(collect(values(a.probabilities))) && a.confidence < a.raw["confidence"]
            elseif a isa ScoreAnswer
                @test a.confidence == UniLM._score_confidence([a.probabilities[l] for l in 0:length(a.probabilities)-1])
            else
                @test a isa NoulAnswer && a.noul == wire[name]["noul"]
            end
        end
    end
    # W2: billing at 0.988 of four options; Gemini's confidence is that probability, TypeSafe's (4·0.988 − 1)/3.
    r = _gd_decode(_gd_request("W2"), _gd_fixture("W2"))
    a = r["department"]
    @test a.choice == "billing" && a.raw["confidence"] == a.probabilities["billing"] == 0.9876832703286654
    @test a.confidence ≈ (4 * 0.9876832703286654 - 1) / 3
    s = _gd_decode(_gd_request("W3"), _gd_fixture("W3"))["severity"]
    @test s.score == 1.121459528316996 && s.legend[1] == "Workaround available: a task fails, but another way works."
    @test s.raw["confidence"] == s.probabilities[1] && s.confidence < s.raw["confidence"]
    # A model the call names is the response's: the reply names none.
    w6 = _gd_fixture("W6")
    r = _gd_decode(SystemOneRequest(_GD_TICKET, "damaged" => noul("Does the customer report a damaged item?");
                                    model = "models/gemini-decision-1"), w6)
    @test r.response.model == "models/gemini-decision-1"
    # The same request three times: the same answer, in another key order (W2, W11a–c).
    @test length(unique(_gd_decode(_gd_request("W2"), _gd_fixture(t))["department"].probabilities
                        for t in ("W2", "W11a", "W11b", "W11c"))) == 1
end

@testset "Gemini Decisions — usage: input by modality, output 0, a bad count read as 0 with a warning" begin
    @test token_usage(_gd_decode(_gd_request("W4"), _gd_fixture("W4"))) == TokenUsage(prompt_tokens=241, total_tokens=241)
    # An image counts as image tokens, in the input total: 256 at resolution "low" (W5), 1,089 without (W14).
    @test token_usage(_gd_decode(_gd_request("W5"), _gd_fixture("W5"))) == TokenUsage(prompt_tokens=398, total_tokens=398)
    @test token_usage(_gd_decode(_gd_request("W14"), _gd_fixture("W14"))).prompt_tokens == 1181
    with_usage(usage) = (fx = _gd_fixture("W1"); b = JSON.parse(fx["body_text"]);
                         isnothing(usage) ? delete!(b, "usage") : (b["usage"] = usage); fx["body_text"] = JSON.json(b);
                         _gd_decode(_gd_request("W1"), fx))
    audio = Dict("total_input_tokens" => 900, "total_output_tokens" => 0, "total_cached_tokens" => 0,
                 "input_tokens_by_modality" => [Dict("modality" => "text", "tokens" => 100), Dict("modality" => "audio", "tokens" => 800)])
    @test token_usage(with_usage(audio)) == TokenUsage(prompt_tokens=900, total_tokens=900, prompt_audio_tokens=800)
    @test token_usage(@test_logs min_level=Logging.Warn with_usage(nothing)) == TokenUsage()
    for bad in (Dict("total_input_tokens" => -1), Dict("total_input_tokens" => 1.5), "lots")
        r = @test_logs (:warn, r"System One usage") min_level=Logging.Warn with_usage(bad)
        @test r isa SystemOneSuccess && token_usage(r).prompt_tokens == 0
    end
end

@testset "Gemini Decisions — probabilities arrive unrounded: the float tolerance, and the docs' rounded examples refused" begin
    # Every capture's distribution sums to 1 within 2.3e-16, with 15 or more decimals: the pre-registered rule
    # tightens the allowance from two-decimal rounding to the float tolerance of every unrounded backend.
    deviations = Float64[]
    for (tag, _) in _GD_REBUILT, (_, a) in _gd_decode(_gd_request(tag), _gd_fixture(tag)).response.answers
        a isa Union{ChoiceAnswer,ScoreAnswer} && push!(deviations, abs(sum(values(a.probabilities)) - 1))
    end
    @test length(deviations) >= 6 && maximum(deviations) <= 1e-15
    @test UniLM._probability_tolerance(_GD, 4) == UniLM._probability_slack(_GD) == UniLM._FLOAT_PROBABILITY_TOLERANCE
    # The guide's own examples, rounded to two or three decimals, are no distribution at that tolerance.
    choice_reply = """{"answers":{"department":{"type":"choice","choice":"billing","confidence":0.99,
        "probabilities":{"billing":0.99,"shipping":0.004,"other":0.003,"technical":0.002}}}}"""
    req = SystemOneRequest("I was charged twice for my order.",
                           "department" => choice("Which department?", ["billing", "technical", "shipping", "other"]))
    e = _gd_caught(() -> UniLM.decode_system_one(_GD, GEMINIServiceEndpoint, req, _gd_reply(200, choice_reply)))
    @test e isa ArgumentError && contains(e.msg, "summing to 0.999")
    score_reply = """{"answers":{"severity":{"type":"score","score":1.12,"confidence":0.83,"legend":{"0":"a","1":"b","2":"c"},
        "probabilities":{"0":0.03,"1":0.83,"2":0.15}}}}"""
    e = _gd_caught(() -> UniLM.decode_system_one(_GD, GEMINIServiceEndpoint,
        SystemOneRequest("s", "severity" => score("How severe?", ["a", "b", "c"])), _gd_reply(200, score_reply)))
    @test e isa ArgumentError && contains(e.msg, "not 1")
end

@testset "Gemini Decisions — a blocked input is a refusal of every question, in either documented shape" begin
    q = ["team" => choice("Which team?", ["billing", "other"]), "frustrated" => noul("Is the customer frustrated?")]
    r, seen = _gd_serve(_gd_blocked("prohibited_content")) do
        ask(_GD_TICKET, q...; service=_GDMock, config=_GD_CFG)
    end
    @test length(seen) == 1 && r isa SystemOneSuccess
    @test all(name -> r[name] isa RefusalAnswer, ("team", "frustrated"))
    @test r["team"].raw == Dict("block_reason" => "prohibited_content", "block_reason_message" => "Blocked by policy.")
    @test r.response.raw["prompt_feedback"]["block_reason"] == "prohibited_content" && token_usage(r).prompt_tokens == 120
    # A reply that blocks and still answers, or names its reason in another JSON type, is no answer.
    for odd in ("""{"prompt_feedback":{"block_reason":"safety"},"answers":{"team":{"type":"noul","noul":0.5}}}""",
                """{"prompt_feedback":{"block_reason":3}}""", """{"prompt_feedback":"safety"}""")
        r, _ = _gd_serve((_, _) -> _gd_reply(200, odd)) do
            ask(_GD_TICKET, q...; service=_GDMock, config=_GD_CFG)
        end
        @test r isa SystemOneCallError
    end
    # A prompt_feedback without a reason blocks nothing.
    r, _ = _gd_serve((n, req) -> (b = _gd_answers(_gd_body(req)); b["prompt_feedback"] = Dict{String,Any}();
                                  _gd_reply(200, JSON.json(b)))) do
        ask(_GD_TICKET, q...; service=_GDMock, config=_GD_CFG)
    end
    @test r isa SystemOneSuccess && r["team"] isa ChoiceAnswer
    # The guide's shape: HTTP 400 with a string code, which becomes the failure's error type.
    for code in ("safety", "prohibited_content", "blocklist", "image_safety", "content_blocked")
        body = """{"error":{"code":"$code","message":"Request blocked due to safety violations (harmful content). Please modify your input and retry."}}"""
        r, _ = _gd_serve((_, _) -> _gd_reply(400, body)) do
            ask(_GD_TICKET, q...; service=_GDMock, config=_GD_CFG)
        end
        @test r isa SystemOneFailure && r.status == 400 && r.error_type == code && startswith(r.message, "Request blocked")
    end
end

@testset "Gemini Decisions — a non-200 is a failure with the error's code, else its status, and message" begin
    for (tag, error_type) in (("W8a", "invalid_request"), ("W8d", "invalid_request"), ("W9", "invalid_request"),
                              ("W18", "invalid_request"), ("W17", "not_found"), ("W16", "INVALID_ARGUMENT"))
        fx = _gd_fixture(tag)
        r = _gd_decode(SystemOneRequest(_GD_TICKET, "q" => noul("Is it?")), fx)
        @test r isa SystemOneFailure && r.status == fx["status"] && r.error_type == error_type
        @test r.message == JSON.parse(fx["body_text"])["error"]["message"] && r.response == fx["body_text"]
        @test r.request_id === nothing && r.retry_after === nothing && !issuccess(r)
    end
    # The v1 path has no Decisions route: an empty 404.
    r = _gd_decode(SystemOneRequest(_GD_TICKET, "q" => noul("Is it?")), _gd_fixture("W7"))
    @test r isa SystemOneFailure && r.status == 404 && r.error_type === nothing && r.message == ""
    r, _ = _gd_serve((_, _) -> _gd_reply(429, """{"error":{"code":429,"message":"Resource exhausted.","status":"RESOURCE_EXHAUSTED"}}""";
                                         headers=["retry-after" => "2"])) do
        ask(_GD_TICKET, "q" => noul("Is it?"); service=_GDMock, config=_GD_CFG)
    end
    @test r isa SystemOneFailure && r.error_type == "RESOURCE_EXHAUSTED" && r.retry_after == 2.0
    @test contains(sprint(showerror, SystemOneError(r)), "HTTP 429 — Resource exhausted.")
    r, _ = _gd_serve((_, _) -> _gd_reply(503, "<html>upstream unavailable</html>")) do
        ask(_GD_TICKET, "q" => noul("Is it?"); service=_GDMock, config=_GD_CFG)
    end
    @test r isa SystemOneFailure && r.error_type === nothing && r.message == "<html>upstream unavailable</html>"
end

@testset "Gemini Decisions — a reply that does not fit the request is a call error, never a success" begin
    w4 = JSON.parse(_gd_fixture("W4")["body_text"])
    edit(f) = (b = deepcopy(w4); f(b); JSON.json(b))
    malformed = [
        "answers is no object"        => edit(b -> b["answers"] = collect(values(b["answers"]))),
        "no answers"                  => edit(b -> delete!(b, "answers")),
        "an answer is no object"      => edit(b -> b["answers"]["deadline"] = 0.9),
        "a question unanswered"       => edit(b -> delete!(b["answers"], "deadline")),
        "an unasked answer"           => edit(b -> b["answers"]["extra"] = Dict("type" => "noul", "noul" => 0.5)),
        "no type"                     => edit(b -> delete!(b["answers"]["deadline"], "type")),
        "a choice answering a noul"   => edit(b -> b["answers"]["deadline"]["type"] = "choice"),
        "an unknown type for a noul"  => edit(b -> b["answers"]["deadline"]["type"] = "verdict"),
        "a noul out of range"         => edit(b -> b["answers"]["deadline"]["noul"] = 1.2),
        "a noul as text"              => edit(b -> b["answers"]["deadline"]["noul"] = "0.9"),
        "a choice not offered"        => edit(b -> b["answers"]["department"]["choice"] = "marketing"),
        "an offered option missing"   => edit(b -> delete!(b["answers"]["department"]["probabilities"], "billing")),
        "a choice that is not the argmax" => edit(b -> b["answers"]["department"]["choice"] = "other"),
        "a sum off by 1e-5"           => edit(b -> b["answers"]["department"]["probabilities"]["billing"] += 1e-5),
        "a level beyond the rubric"   => edit(b -> b["answers"]["severity"]["probabilities"]["3"] = 0.0),
        "a level renamed"             => edit(b -> b["answers"]["severity"]["legend"]["0"] = "WRONG"),
        "a score its probabilities contradict" => edit(b -> b["answers"]["severity"]["score"] = 1.5),
        "not an object"               => "[]"]
    req = _gd_request("W4")
    for (label, body) in malformed
        @testset "$label" begin
            r, sent = _gd_serve((_, _) -> _gd_reply(200, body)) do
                ask(req; service=_GDMock, config=_GD_CFG)
            end
            @test r isa SystemOneCallError && length(sent) == 1 && r.request_id === nothing && r.cause isa Exception
        end
    end
    # A raw question's answer of any type is accepted, one of a known type typed, with the surface's confidence.
    raw = SystemOneRequest("s", ["rank" => RawQuestion(Dict("type" => "rank", "items" => ["a", "b"])),
                                 "pick" => RawQuestion(Dict("type" => "choice", "instructions" => "Which?",
                                                            "criteria" => Dict("a" => "", "b" => "")))])
    r, seen = _gd_serve(_gd_valid) do
        ask(raw; service=_GDMock, config=_GD_CFG)
    end
    @test r isa SystemOneSuccess && r["rank"] isa UnknownAnswer && r["rank"].type == "rank"
    @test r["pick"] isa ChoiceAnswer && r["pick"].confidence == 1.0
    @test only(seen).target == "/v1beta/decisions"
    @test any(h -> lowercase(first(h)) == "x-goog-api-key" && last(h) == "gd", only(seen).headers)
end

# ─── Recording ───────────────────────────────────────────────────────────────

@testset "Gemini Decisions — recorded under its own namespace, an image by its digest, replayed with no server" begin
    dir = mktempdir()
    q = "damaged" => noul("Does the customer report a damaged item?")
    live, seen = _gd_serve(_gd_valid) do
        with_recorded_answers(() -> ask(_GD_TICKET, q; service=_GDMock, config=_GD_CFG), dir; mode=:record)
    end
    @test live isa SystemOneSuccess && length(seen) == 1
    rec = JSON.parse(read(only(readdir(dir; join=true)), String))
    @test rec["request"]["path"] == "/gemini-decisions/v1beta/decisions" && !haskey(rec["request"], "key_extra")
    @test rec["response"]["request_id"] === nothing && JSON.json(rec["request"]["body"]) == only(seen).body
    replayed = with_recorded_answers(() -> ask(_GD_TICKET, q; service=_GDMock), dir)
    @test replayed isa SystemOneSuccess && replayed.response.raw == live.response.raw && replayed["damaged"].noul == 0.5

    images = mktempdir()
    low = ImageAttachment(_GD_PNG.data; detail = "low")
    live, seen = _gd_serve(_gd_valid) do
        with_recorded_answers(() -> ask(Any["Inspect this.", low], q; service=_GDMock, config=_GD_CFG), images; mode=:record)
    end
    b64 = base64encode(_GD_PNG.data)
    @test live isa SystemOneSuccess && contains(only(seen).body, b64)          # the service gets the image
    file = only(readdir(images; join=true))
    text = read(file, String)
    digest = "data:image/png;sha256," * bytes2hex(UniLM.SHA.sha256(b64))
    @test !contains(text, b64) && JSON.parse(text)["request"]["body"]["input"][2]["data"] == digest
    keyed = replace(only(seen).body, b64 => digest)
    @test basename(file) == bytes2hex(UniLM.SHA.sha256("POST /gemini-decisions/v1beta/decisions\n" * keyed)) * ".json"
    @test with_recorded_answers(() -> ask(Any["Inspect this.", low], q; service=_GDMock), images).response.raw == live.response.raw
    # Another resolution, or another image, is another request, reported by its digest.
    for other in (ImageAttachment(_GD_PNG.data; detail = "high"), _GD_JPEG)
        miss = with_recorded_answers(() -> _gd_caught(() -> ask(Any["Inspect this.", other], q; service=_GDMock)), images)
        @test miss isa ReplayMissError && contains(miss.body, ";sha256,") && !contains(miss.body, base64encode(other.data))
    end
end

@testset "Gemini Decisions — ask_each sends each request as a single ask sends it" begin
    states = [Any["Inspect image $i.", _GD_PNG] for i in 1:3]
    q = "damaged" => noul("Is anything damaged?")
    each, seen_each = _gd_serve(_gd_valid) do
        ask_each(states, q; service=_GDMock, config=_GD_CFG, ntasks=2)
    end
    alone, seen_alone = _gd_serve(_gd_valid) do
        [ask(s, q; service=_GDMock, config=_GD_CFG) for s in states]
    end
    @test all(issuccess, each) && all(issuccess, alone)
    @test sort([s.body for s in seen_each]) == sort([s.body for s in seen_alone]) && length(unique(s.body for s in seen_each)) == 3
end

# ─── The natural-language verbs ──────────────────────────────────────────────

_gd_route(::nl"the customer wants a refund", ticket) = (:refund, ticket)
_gd_route(::nl"the customer reports a bug", ticket) = (:bug, ticket)
const _GD_INTENT = (refund = "the customer wants a refund", bug = "the customer reports a bug")

@testset "Gemini Decisions — the natural-language verbs, a blocked input as a decline" begin
    ticket = "My package arrived crushed and the screen is cracked. I want my money back."
    answered, seen = _gd_serve(_gd_valid) do
        (nl_classify(ticket, _GD_INTENT; service=_GDMock, config=_GD_CFG),
         nl_dispatch(_gd_route, ticket; service=_GDMock, config=_GD_CFG),
         (@branch ticket service=_GDMock config=_GD_CFG begin
             "the customer wants a refund" => :refund
             "the customer reports a bug"  => :bug
         end))
    end
    @test answered == (:refund, (:refund, ticket), :refund) && length(seen) == 3
    @test [only(keys(_gd_body(s)["questions"])) for s in seen] == ["classify", "meaning_1", "branch"]
    out, seen = _gd_serve(_gd_blocked()) do
        [nl_classify(ticket, _GD_INTENT; service=_GDMock, config=_GD_CFG, fallback = s -> :fallback),
         _gd_caught(() -> nl_dispatch(_gd_route, ticket; service=_GDMock, config=_GD_CFG)),
         _gd_caught(() -> @branch ticket service=_GDMock config=_GD_CFG begin
             "the customer wants a refund" => :refund
             "the customer reports a bug"  => :bug
         end)]
    end
    @test length(seen) == 3 && out[1] === :fallback
    @test all(e -> e isa DecisionDeclinedError && e.answer isa RefusalAnswer, out[2:3])
    # A sentence over an option name's 128 characters is refused before any I/O, with the fix.
    long = (refund = "the customer wants a refund " * "of the order "^9, bug = "the customer reports a bug")
    e = only(_gd_refused([() -> nl_classify(ticket, long; service=_GDMock, config=_GD_CFG)]))
    @test contains(e.msg, "an option's name takes at most 128 characters") && contains(e.msg, "as its description")
end

# ─── Cost ────────────────────────────────────────────────────────────────────

@testset "Gemini Decisions — estimated_cost bills the input tokens at \$0.10 per 1M, and no output" begin
    @test DEFAULT_PRICING["gemini-decision-1"] == UniLM._price(0.10, 0.10, 0.0)
    r = _gd_decode(_gd_request("W4"), _gd_fixture("W4"))
    @test estimated_cost(r) ≈ 241 * 0.10 / 1_000_000
    image = _gd_decode(_gd_request("W5"), _gd_fixture("W5"))
    @test estimated_cost(image) ≈ 398 * 0.10 / 1_000_000                       # every modality at one rate
    by_hand = SystemOneSuccess(SystemOneResponse("gemini-decision-1", Dict{String,SystemOneAnswer}(),
        TokenUsage(prompt_tokens=1000, completion_tokens=0, total_tokens=1000), nothing, Dict{String,Any}(), GEMINIServiceEndpoint))
    @test estimated_cost(by_hand) ≈ 1000 * 0.10 / 1_000_000
    # The prefixed id the API also answers has no row of its own: the bare id is the one priced.
    prefixed = _gd_decode(SystemOneRequest(_GD_TICKET, "damaged" => noul("Does the customer report a damaged item?");
                                           model = "models/gemini-decision-1"), _gd_fixture("W6"))
    @test (@test_logs (:warn, r"no price row") estimated_cost(prefixed)) == 0.0
    @test estimated_cost(prefixed; model = "gemini-decision-1") ≈ 98 * 0.10 / 1_000_000
end
