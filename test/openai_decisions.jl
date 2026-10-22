# test/openai_decisions.jl — OpenAI's Decisions API as a System One backend, zero-spend.
#
# The wrapper's encoding (the requests captured live in fixtures/openai_decisions,
# rebuilt from public constructors; every form of state; every refusal before I/O), its
# decoding (the captured replies, malformed ones, error bodies), the tolerance its
# two-decimal rounding needs, the natural-language verbs on a backend that refuses,
# recording, and cost. Local mocks only; nothing here reaches a network.

using Sockets, Logging, Base64

const _OD = UniLM.OpenAIDecisionsProtocol()
const _OD_CFG = UniLM.RequestConfig(max_attempts=1, total_deadline=30.0)
const _OD_FIXTURES = joinpath(@__DIR__, "fixtures", "openai_decisions")
const _OD_PNG = ImageAttachment(read(joinpath(@__DIR__, "fixtures", "ollama", "red_circle.png")))
const _OD_JPEG = ImageAttachment(UInt8[0xff, 0xd8, 0xff, 0xe0, 0x0d, 0x0a, 0x2d, 0x2d, 0x00, 0x10])

"A live capture: its request, status, headers and reply body, as recorded."
_od_fixture(tag::String) = JSON.parse(read(joinpath(_OD_FIXTURES, tag * ".json"), String))

_od_caught(f::Function) = try f(); nothing catch e; e end

# An endpoint that speaks the Decisions wire to a local server.
struct _ODMock <: UniLM.ServiceEndpoint end
const _OD_BASE = Ref("")
UniLM._resolve_base_url(::Type{_ODMock}) = _OD_BASE[]
UniLM.auth_header(::Type{_ODMock}) = ["Authorization" => "Bearer od", "Content-Type" => "application/json"]
UniLM.provider_capabilities(::Type{_ODMock}) = Set([:system_one])
UniLM.system_one_protocol(::Type{_ODMock}) = UniLM.OpenAIDecisionsProtocol()

const _ODSeen = @NamedTuple{method::String, target::String, headers::Vector{Pair{String,String}}, body::String}

"Run `f()` against a local server answering request `n` with `respond(n, req)`; return `(f(), requests)`."
function _od_serve(f::Function, respond::Function)
    seen = _ODSeen[]
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
        _OD_BASE[] = "http://127.0.0.1:$port"
        try
            return f(), seen
        finally
            close(server)
        end
    end
    error("could not bind an ephemeral port for the Decisions mock")
end

_od_reply(status::Int, body::AbstractString; headers::Vector{Pair{String,String}}=Pair{String,String}[]) =
    HTTP.Response(status, Pair{String,String}["Content-Type" => "application/json"; headers], Vector{UInt8}(body))

const _OD_USAGE = Dict{String,Any}("input_tokens" => 120, "output_tokens" => 0, "total_tokens" => 120,
    "input_tokens_details" => Dict{String,Any}("cached_tokens" => 0, "cache_write_tokens" => 0),
    "output_tokens_details" => Dict{String,Any}("reasoning_tokens" => 0))

# A valid reply to every question of a Decisions body, in order: the first option, the
# lowest level, an even predicate, an echo of a raw question's type; `refuse` names the
# questions refused instead.
function _od_answers(body::AbstractDict; refuse=String[])::Dict{String,Any}
    answers = Any[]
    for q in body["questions"]
        name, t = q["name"], q["type"]
        push!(answers, if name in refuse
            Dict{String,Any}("type" => "refusal", "name" => name)
        elseif t == "predicate"
            Dict{String,Any}("type" => t, "name" => name, "probability" => 0.5)
        elseif t == "choice"
            values = [c["value"] for c in q["choices"]]
            Dict{String,Any}("type" => t, "name" => name, "choice" => values[1], "confidence" => 1.0,
                             "probabilities" => [Dict{String,Any}("value" => v, "probability" => i == 1 ? 1.0 : 0.0)
                                                 for (i, v) in enumerate(values)])
        elseif t == "score"
            Dict{String,Any}("type" => t, "name" => name, "score" => 0.0, "confidence" => 1.0,
                             "probabilities" => [Dict{String,Any}("value" => i - 1, "label" => l["label"],
                                                                  "probability" => i == 1 ? 1.0 : 0.0)
                                                 for (i, l) in enumerate(q["levels"])])
        else
            Dict{String,Any}("type" => t, "name" => name, "echo" => true)
        end)
    end
    Dict{String,Any}("model" => "gpt-6-luna", "answers" => answers, "usage" => _OD_USAGE)
end

_od_body(req) = JSON.parse(String(copy(req.body)))
_od_body(s::_ODSeen) = JSON.parse(s.body)
_od_valid(n, req) = _od_reply(200, JSON.json(_od_answers(_od_body(req))); headers=["x-request-id" => "req_$n"])
_od_refusing(names...) = (n, req) -> _od_reply(200, JSON.json(_od_answers(_od_body(req); refuse=collect(names)));
                                              headers=["x-request-id" => "req_$n"])

_od_header(s::_ODSeen, name::String) = [last(h) for h in s.headers if lowercase(first(h)) == lowercase(name)]

"The body `request` is sent as, parsed in order."
_od_sent(request::SystemOneRequest; service=_ODMock) =
    JSON.parse(UniLM.encode_system_one(_OD, service, request).body)

# ─── The trait and its hooks ─────────────────────────────────────────────────

@testset "OpenAI Decisions — OPENAIServiceEndpoint speaks the Decisions protocol" begin
    @test UniLM.system_one_protocol(OPENAIServiceEndpoint) === _OD
    @test has_capability(OPENAIServiceEndpoint, :system_one)
    @test UniLM._system_one_default_model(_OD, OPENAIServiceEndpoint) == "gpt-6-luna"
    @test UniLM._system_one_namespace(_OD) == "openai-decisions"
    @test UniLM._allows_refusal(_OD) && UniLM._allows_raw(_OD)

    req = SystemOneRequest("The package arrived with a broken screen.",
                           "damaged" => noul("Does the customer report a damaged item?"))
    wire = UniLM.encode_system_one(_OD, OPENAIServiceEndpoint, req)
    @test wire.method == "POST" && wire.url == "https://api.openai.com/v1/decisions" && wire.path == "/v1/decisions"
    @test isempty(wire.parts) && isempty(wire.headers) && wire.key_extra == ""
    # The request OpenAI's reference shows, byte for byte, with the default model.
    @test wire.body == """{"model":"gpt-6-luna","input":"The package arrived with a broken screen.",""" *
                       """"questions":[{"type":"predicate","name":"damaged","instructions":"Does the customer report a damaged item?"}]}"""

    withenv("OPENAI_API_KEY" => nothing) do
        # A scope over an empty recordings directory shows what would be sent, and where it
        # would be filed, without a key or a server.
        miss = with_recorded_answers(mktempdir()) do
            with_system_one(OPENAIServiceEndpoint) do
                _od_caught(() -> ask(req))
            end
        end
        @test miss isa ReplayMissError && miss.path == "/openai-decisions/v1/decisions" && miss.body == wire.body
        pinned = with_recorded_answers(mktempdir()) do
            with_system_one(() -> _od_caught(() -> ask(req)), OPENAIServiceEndpoint; model="gpt-6-luna-next")
        end
        @test JSON.parse(pinned.body)["model"] == "gpt-6-luna-next"
        # Live with no key: a call error that names it, before anything is sent.
        r = ask(req; service=OPENAIServiceEndpoint, config=_OD_CFG)
        @test r isa SystemOneCallError && contains(r.error, "OPENAI_API_KEY") && r.request_id === nothing
    end
    # No models listing on this protocol: refused by name, before any request.
    e = _od_caught(() -> list_models(; service=OPENAIServiceEndpoint))
    @test e isa ArgumentError && contains(e.msg, "OpenAIDecisionsProtocol")
end

# ─── Encoding ────────────────────────────────────────────────────────────────

# The typed request a captured body was made from, rebuilt with public constructors only.
function _od_rebuild(body::AbstractDict)::SystemOneRequest
    qs = Pair{String,SystemOneQuestion}[]
    for q in body["questions"]
        t, ins = q["type"], q["instructions"]
        push!(qs, q["name"] => t == "predicate" ? noul(ins) :
                               t == "choice" ? choice(ins, [c["value"] => get(c, "description", nothing) for c in q["choices"]]) :
                               score(ins, [l["label"] for l in q["levels"]]))
    end
    SystemOneRequest(body["input"], qs; model=body["model"])
end

@testset "OpenAI Decisions — the captured requests rebuild from public constructors, byte for byte" begin
    # Whole bodies: a predicate (D01) and the seven-intent choice over a JSON text input
    # (D11–D19), descriptions included.
    for tag in ("D01", ("D$(i)" for i in 11:19)...)
        sent = _od_fixture(tag)["request"]
        @test JSON.json(_od_sent(_od_rebuild(sent); service=OPENAIServiceEndpoint)) == JSON.json(sent)
    end
    # Single questions where a capture holds what no typed question sends (an unnamed
    # question, boolean choice values, level descriptions).
    question(name, q) = JSON.json(only(_od_sent(SystemOneRequest("x", name => q))["questions"]))
    d02 = _od_fixture("D02")["request"]["questions"]
    @test question("department", choice(d02[2]["instructions"],
                   [c["value"] => c["description"] for c in d02[2]["choices"]])) == JSON.json(d02[2])
    @test question("angry", noul(d02[4]["instructions"])) == JSON.json(d02[4])
    d03 = _od_fixture("D03")["request"]["questions"][2]
    @test question("bare", choice(d03["instructions"], [c["value"] for c in d03["choices"]])) == JSON.json(d03)
    # An image: the user message the capture sent (its bytes are elided there).
    d04 = _od_fixture("D04")["request"]
    sent = _od_sent(SystemOneRequest(Any["Inspect this image.", _OD_PNG],
                                     [q["name"] => (q["type"] == "predicate" ? noul(q["instructions"]) :
                                                    choice(q["instructions"], [c["value"] for c in q["choices"]]))
                                      for q in d04["questions"]]))
    @test JSON.json(sent["questions"]) == JSON.json(d04["questions"])
    message = only(sent["input"])
    @test collect(keys(message)) == ["role", "content"] && message["role"] == "user"
    @test message["content"][1] == JSON.parse("""{"type":"input_text","text":"Inspect this image."}""")
    @test collect(keys(message["content"][2])) == ["type", "image_url"] && message["content"][2]["type"] == "input_image"
    @test message["content"][2]["image_url"] == "data:image/png;base64," * base64encode(_OD_PNG.data)
end

@testset "OpenAI Decisions — the state is the input: text, JSON text, or one message of text and images" begin
    input(state) = _od_sent(SystemOneRequest(state, "q" => noul("Is it?")))["input"]
    @test input("plain text") == "plain text"
    @test input("") == ""
    @test input(Dict("b" => 1, "a" => [1, 2])) == """{"a":[1,2],"b":1}"""      # a plain Dict in sorted key order
    @test input(JSON.Object{String,Any}("b" => 1, "a" => 2)) == """{"b":1,"a":2}"""
    @test input((message = "I want my money back",)) == """{"message":"I want my money back"}"""
    @test input(["a", "b"]) == """["a","b"]"""                                 # strings alone are JSON
    @test input(("a", 1, nothing)) == """["a",1,null]"""
    @test input(Any[]) == "[]"
    parts(state) = only(input(state))["content"]
    @test parts(_OD_PNG) == [JSON.parse("""{"type":"input_image","image_url":"data:image/png;base64,$(base64encode(_OD_PNG.data))"}""")]
    mixed = parts(Any["first", _OD_PNG, "", _OD_JPEG])
    @test [p["type"] for p in mixed] == ["input_text", "input_image", "input_text", "input_image"]
    @test mixed[1]["text"] == "first" && mixed[3]["text"] == ""
    @test startswith(mixed[2]["image_url"], "data:image/png;base64,") && startswith(mixed[4]["image_url"], "data:image/jpeg;base64,")
    @test length(parts(fill(_OD_PNG, 128))) == 128

    # An image anywhere but there has no form on the wire: refused before any I/O.
    refusals, seen = _od_serve(_od_valid) do
        [_od_caught(() -> ask(Dict("photo" => _OD_PNG), "q" => noul("Is it?"); service=_ODMock)),
         _od_caught(() -> ask(Any[Dict("photo" => _OD_PNG), "text"], "q" => noul("Is it?"); service=_ODMock)),
         _od_caught(() -> ask((_OD_PNG, "text"), "q" => noul("Is it?"); service=_ODMock)),
         _od_caught(() -> ask((photo = _OD_PNG,), "q" => noul("Is it?"); service=_ODMock)),
         _od_caught(() -> ask(Any["text", [_OD_PNG]], "q" => noul("Is it?"); service=_ODMock)),
         _od_caught(() -> ask(fill(_OD_PNG, 129), "q" => noul("Is it?"); service=_ODMock)),
         _od_caught(() -> ask("s", "q" => choice(_OD_PNG, ["a", "b"]); service=_ODMock)),
         _od_caught(() -> ask("s", "q" => choice("Which?", ["a" => _OD_PNG, "b" => nothing]); service=_ODMock)),
         _od_caught(() -> ask("s", "q" => RawQuestion(Dict("type" => "predicate", "instructions" => "x", "img" => _OD_PNG));
                              service=_ODMock)),
         _od_caught(() -> ask("x"^10_485_761, "q" => noul("Is it?"); service=_ODMock)),
         _od_caught(() -> ask(Dict("x" => NaN), "q" => noul("Is it?"); service=_ODMock)),
         _od_caught(() -> ask("s", "q" => score("Rate.", ["low", _OD_PNG]); service=_ODMock))]
    end
    @test isempty(seen)
    @test all(e -> e isa ArgumentError, refusals)
    @test all(e -> contains(e.msg, "OpenAI's Decisions API") && contains(e.msg, "Vector of strings and ImageAttachments"),
              refusals[1:5])
    @test contains(refusals[6].msg, "at most 128 images") && contains(refusals[6].msg, "129")
    @test contains(refusals[7].msg, "must be a string") && contains(refusals[8].msg, "description of its option \"a\"")
    @test all(e -> contains(e.msg, "an image belongs in the state") &&
                   contains(e.msg, "Vector of strings and ImageAttachments"), refusals[[7, 8, 12]])
    @test contains(refusals[12].msg, "level 1 must be a string")
    @test contains(refusals[9].msg, "images go in the state only") && contains(refusals[9].msg, "Vector of strings")
    @test contains(refusals[10].msg, "10485760 characters") && contains(refusals[10].msg, "10485761")
    @test contains(refusals[11].msg, "JSON")
    @test length(input("x"^10_485_760)) == 10_485_760
end

@testset "OpenAI Decisions — questions map onto predicate, choice and score; what has no form is refused" begin
    sent = _od_sent(SystemOneRequest("Help! My payouts have been failing for 3 days.", [
        "dept" => choice("Which team?", (billing = "Payments, invoicing, payouts, refunds", technical = nothing)),
        "urg" => score("How urgent?", ["Can wait", "This week", "Today"]),
        "frus" => noul("Is the customer frustrated?"),
        "bare" => NoulQuestion("Is it bare?", NoulCriteria(nothing, nothing)),
        "rank" => RawQuestion(JSON.Object{String,Any}("type" => "rank", "items" => ["b", "a"])),
        "named" => RawQuestion(Dict("type" => "predicate", "name" => "named", "instructions" => "Is it?"))]))
    @test JSON.json(sent["questions"]) == "[" * join([
        """{"type":"choice","name":"dept","instructions":"Which team?","choices":[{"value":"billing","description":"Payments, invoicing, payouts, refunds"},{"value":"technical"}]}""",
        """{"type":"score","name":"urg","instructions":"How urgent?","levels":[{"label":"Can wait"},{"label":"This week"},{"label":"Today"}]}""",
        """{"type":"predicate","name":"frus","instructions":"Is the customer frustrated?"}""",
        """{"type":"predicate","name":"bare","instructions":"Is it bare?"}""",
        """{"type":"rank","items":["b","a"],"name":"rank"}""",
        """{"instructions":"Is it?","name":"named","type":"predicate"}"""], ",") * "]"
    @test collect(keys(sent)) == ["model", "input", "questions"]
    # The bounds are accepted.
    @test length(only(_od_sent(SystemOneRequest("s", "c" => choice("Pick.", ["o$i" for i in 1:255])))["questions"])["choices"]) == 255
    @test length(only(_od_sent(SystemOneRequest("s", "c" => choice("Pick.", ["a", "b"])))["questions"])["choices"]) == 2
    @test length(only(_od_sent(SystemOneRequest("s", "r" => score("Rate.", ["l$i" for i in 1:10])))["questions"])["levels"]) == 10
    @test length(_od_sent(SystemOneRequest("s", ["q$i" => noul("Is it $i?") for i in 1:200]))["questions"]) == 200

    refused, seen = _od_serve(_od_valid) do
        one(q) = _od_caught(() -> ask("s", "q" => q; service=_ODMock))
        [one(noul("Is the customer frustrated?"; yes = "Impatient", no = "Calm")),     # 1
         one(noul(; yes = "Impatient")),                                                # 2
         one(choice(["a", "b"])),                                                       # 3
         one(choice(Dict("text" => "Which?"), ["a", "b"])),                             # 4
         one(choice("Pick.", ["only"])),                                                # 5
         one(choice("Pick.", ["o$i" for i in 1:256])),                                  # 6
         one(choice("Pick.", ["a" => (why = "structured",), "b" => nothing])),         # 7
         one(score("Rate.", ["single"])),                                               # 8
         one(score("Rate.", ["l$i" for i in 1:11])),                                    # 9
         one(score("Rate.", ["low", (label = "high",)])),                               # 10
         one(score(["low", "high"])),                                                   # 11
         one(box("the dent")),                                                          # 12
         one(RawQuestion(Dict("type" => "predicate", "name" => "other", "instructions" => "x"))),   # 13
         _od_caught(() -> ask("s", ["q$i" => noul("Is it $i?") for i in 1:201]; service=_ODMock))]  # 14
    end
    @test isempty(seen)                                                   # nothing was sent
    @test all(e -> e isa ArgumentError && contains(e.msg, "OpenAI's Decisions API"), refused)
    @test all(e -> contains(e.msg, "the question \"q\""), refused[1:13])
    @test contains(refused[1].msg, "true/false criteria") && contains(refused[2].msg, "true/false criteria")
    @test contains(refused[3].msg, "needs `instructions`") && contains(refused[4].msg, "must be a string")
    @test contains(refused[5].msg, "2 to 255 options, and it has 1") && contains(refused[5].msg, "Add one")
    @test contains(refused[6].msg, "2 to 255 options, and it has 256") && contains(refused[6].msg, "Split it")
    @test contains(refused[7].msg, "description of its option \"a\" must be a string or `nothing`")
    @test contains(refused[8].msg, "2 to 10 levels, and it has 1") && contains(refused[9].msg, "it has 11")
    @test contains(refused[10].msg, "level 1 must be a string") && contains(refused[11].msg, "needs `instructions`")
    @test contains(refused[12].msg, "Box question")
    @test contains(refused[13].msg, "\"name\" field is \"other\"")
    @test contains(refused[14].msg, "at most 200 questions") && contains(refused[14].msg, "201")
end

@testset "OpenAI Decisions — the provider options, written in one order whatever the caller's" begin
    q = "q" => noul("Is it?")
    a = UniLM.encode_system_one(_OD, _ODMock, SystemOneRequest("s", q; options=(safety_identifier = "user-1", prompt_cache_key = "tickets")))
    b = UniLM.encode_system_one(_OD, _ODMock, SystemOneRequest("s", q; options=(prompt_cache_key = "tickets", safety_identifier = "user-1")))
    @test a.body == b.body && a.key_extra == "" && b.key_extra == ""
    @test endswith(a.body, ""","prompt_cache_key":"tickets","safety_identifier":"user-1"}""")
    # Keyed and recorded without them: the body the same request has with no options.
    @test a.record_body == b.record_body == UniLM.encode_system_one(_OD, _ODMock, SystemOneRequest("s", q)).body
    @test collect(keys(JSON.parse(UniLM.encode_system_one(_OD, _ODMock,
        SystemOneRequest("s", q; options=(safety_identifier = "u",))).body))) == ["model", "input", "questions", "safety_identifier"]
    # Through ask, the keywords are the options, sent as fields of the body.
    r, seen = _od_serve(_od_valid) do
        ask("s", q; service=_ODMock, config=_OD_CFG, prompt_cache_key = "k"^64, safety_identifier = "u"^128)
    end
    @test r isa SystemOneSuccess && _od_body(seen[1])["prompt_cache_key"] == "k"^64

    refused, seen = _od_serve(_od_valid) do
        [_od_caught(() -> ask("s", q; service=_ODMock, temperature = 0)),
         _od_caught(() -> ask("s", q; service=_ODMock, prompt_cache_key = "k"^65)),
         _od_caught(() -> ask("s", q; service=_ODMock, safety_identifier = "u"^129)),
         _od_caught(() -> ask("s", q; service=_ODMock, safety_identifier = 7)),
         _od_caught(() -> ask("s", q; service=_ODMock, confg = _OD_CFG))]
    end
    @test isempty(seen) && all(e -> e isa ArgumentError, refused)
    @test contains(refused[1].msg, "prompt_cache_key (a String of at most 64 characters), safety_identifier " *
                                   "(a String of at most 128 characters) and prompt_cache_options (a PromptCacheOptions " *
                                   "of mode and ttl \"30m\" or \"12h\"); got `temperature`")
    @test contains(refused[2].msg, "`prompt_cache_key` as a String of at most 64 characters; got 65 characters")
    @test contains(refused[3].msg, "at most 128 characters; got 129")
    @test contains(refused[4].msg, "got a Int64")
    @test contains(refused[5].msg, "`confg`")
end

@testset "OpenAI Decisions — prompt_cache_options: mode and ttl, sent and left out of the recording" begin
    q = "q" => noul("Is it?")
    plain = UniLM.encode_system_one(_OD, _ODMock, SystemOneRequest("s", q))
    pco = PromptCacheOptions(mode="implicit", ttl="12h")
    wire = UniLM.encode_system_one(_OD, _ODMock, SystemOneRequest("s", q; options=(prompt_cache_options=pco,)))
    @test wire.body == chop(plain.body) * ""","prompt_cache_options":{"mode":"implicit","ttl":"12h"}}"""
    @test wire.record_body == plain.body && wire.key_extra == ""          # keyed and recorded without it
    # After the other options, in one order whatever the caller's.
    every = UniLM.encode_system_one(_OD, _ODMock, SystemOneRequest("s", q; options=(
        prompt_cache_options=PromptCacheOptions(ttl="30m"), safety_identifier="u", prompt_cache_key="k")))
    @test every.body == chop(plain.body) *
        ""","prompt_cache_key":"k","safety_identifier":"u","prompt_cache_options":{"ttl":"30m"}}"""
    @test every.record_body == plain.body
    # Explicit mode is sent as asked: the request marks no breakpoint, so OpenAI caches none of it.
    @test UniLM.encode_system_one(_OD, _ODMock, SystemOneRequest("s", q;
        options=(prompt_cache_options=PromptCacheOptions(mode="explicit"),))).body ==
        chop(plain.body) * ""","prompt_cache_options":{"mode":"explicit"}}"""
    # Through ask, the keyword is the option, and the service receives it.
    r, seen = _od_serve(_od_valid) do
        ask("s", q; service=_ODMock, config=_OD_CFG, prompt_cache_options=pco)
    end
    @test r isa SystemOneSuccess && _od_body(only(seen))["prompt_cache_options"] == Dict("mode" => "implicit", "ttl" => "12h")
    # The same question with it replays the recording made without it, sending nothing.
    dir = mktempdir()
    _, recorded = _od_serve(_od_valid) do
        with_recorded_answers(() -> ask("s", q; service=_ODMock, config=_OD_CFG), dir; mode=:record)
    end
    replayed, unsent = _od_serve(_od_valid) do
        with_recorded_answers(() -> ask("s", q; service=_ODMock, prompt_cache_options=pco), dir)
    end
    @test length(recorded) == 1 && isempty(unsent) && replayed isa SystemOneSuccess
    # Refused before any request: another API's ttl, the Responses-only fields, another type.
    refused, seen = _od_serve(_od_valid) do
        [_od_caught(() -> ask("s", q; service=_ODMock, prompt_cache_options=PromptCacheOptions(ttl="1h"))),
         _od_caught(() -> ask("s", q; service=_ODMock, prompt_cache_options=PromptCacheOptions(ttl="5m"))),
         _od_caught(() -> ask("s", q; service=_ODMock, prompt_cache_options=PromptCacheOptions(prewarm=true))),
         _od_caught(() -> ask("s", q; service=_ODMock,
                              prompt_cache_options=PromptCacheOptions(comparison_response_id="resp_1"))),
         _od_caught(() -> ask("s", q; service=_ODMock, prompt_cache_options=Dict("ttl" => "12h")))]
    end
    @test isempty(seen) && all(e -> e isa ArgumentError, refused)
    @test refused[1].msg == "OpenAI Decisions API accepts prompt_cache_options ttl \"30m\" or \"12h\" (got \"1h\")"
    @test refused[2].msg == "OpenAI Decisions API accepts prompt_cache_options ttl \"30m\" or \"12h\" (got \"5m\")"
    @test refused[3].msg == refused[4].msg == "OpenAI's Decisions API takes prompt_cache_options mode and ttl only; " *
                                              "prewarm and comparison_response_id are fields of the Responses API (Respond)"
    @test refused[5].msg == "OpenAI's Decisions API takes `prompt_cache_options` as a PromptCacheOptions; " *
                            "got a Dict{String, String}"
end

# ─── Decoding ────────────────────────────────────────────────────────────────

const _OD_REQ = SystemOneRequest("Export fails in Safari but works in Chrome.", [
    "severity" => score("How severe is this issue?", ["Cosmetic", "Workaround available", "Fully blocked"]),
    "department" => choice("Which department should handle this complaint?", ["billing", "technical", "shipping", "other"]),
    "deadline" => noul("Does the customer mention a deadline?"),
    "angry" => noul("Is the customer angry?")])

# D02's reply, with the name its unnamed question would have had.
function _od_d02(f::Function=identity)
    body = _od_fixture("D02")["body"]
    body["answers"][3]["name"] = "deadline"
    f(body)
    JSON.json(body)
end

"`ask(_OD_REQ)` over a mock answering `body` once; returns (result, requests sent)."
function _od_ask_with(body::AbstractString; status::Int=200, headers=Pair{String,String}["x-request-id" => "req_od"])
    r, seen = _od_serve((_, _) -> _od_reply(status, body; headers)) do
        ask(_OD_REQ; service=_ODMock, config=_OD_CFG)
    end
    r, length(seen)
end

@testset "OpenAI Decisions — a reply is decoded, matched by name, and its usage read" begin
    r, sent = _od_ask_with(_od_d02())
    @test r isa SystemOneSuccess && sent == 1
    @test r.response.model == "gpt-6-luna" && r.response.request_id == "req_od" && r.response.service === _ODMock
    sev = r["severity"]
    @test sev isa ScoreAnswer && sev.score == 0.97 && sev.confidence == 0.96
    @test sev.legend == Dict(0 => "Cosmetic", 1 => "Workaround available", 2 => "Fully blocked")
    @test sev.probabilities == Dict(0 => 0.03, 1 => 0.97, 2 => 0.0)
    dept = r["department"]
    @test dept isa ChoiceAnswer && dept.choice == "technical" && dept.confidence == 0.99
    @test dept.probabilities == Dict("billing" => 0.0, "technical" => 0.99, "shipping" => 0.01, "other" => 0.0)
    @test r["deadline"] isa NoulAnswer && r["deadline"].noul == 1.0 && r["angry"].noul == 0.44
    @test r["angry"].raw == Dict{String,Any}("type" => "predicate", "name" => "angry", "probability" => 0.44)
    @test token_usage(r) == TokenUsage(prompt_tokens=589, completion_tokens=0, total_tokens=589)

    # Cached and reasoning tokens are read from their details.
    cached = _od_d02() do body
        body["usage"] = Dict("input_tokens" => 2048, "input_tokens_details" => Dict("cached_tokens" => 1024, "cache_write_tokens" => 0),
                             "output_tokens" => 3, "output_tokens_details" => Dict("reasoning_tokens" => 2), "total_tokens" => 2051)
    end
    @test token_usage(first(_od_ask_with(cached))) ==
          TokenUsage(prompt_tokens=2048, completion_tokens=3, total_tokens=2051, cached_tokens=1024, reasoning_tokens=2)
    # Usage is accounting: a value that is no token count reads as 0, with one warning; a
    # missing one reads as 0, silently.
    bad = _od_d02(body -> body["usage"]["input_tokens_details"]["cached_tokens"] = -1)
    r = @test_logs (:warn, r"System One usage") min_level=Logging.Warn first(_od_ask_with(bad))
    @test r isa SystemOneSuccess && token_usage(r).cached_tokens == 0 && token_usage(r).prompt_tokens == 589
    for usage in ("lots", Dict("input_tokens" => 1.5))
        r = @test_logs (:warn, r"System One usage") min_level=Logging.Warn first(_od_ask_with(_od_d02(b -> b["usage"] = usage)))
        @test r isa SystemOneSuccess && token_usage(r) == TokenUsage()
    end
    # The details are objects in the schema: a count where the object should be is no count.
    for details in (5, "lots", [1024], nothing)
        odd = _od_d02(body -> body["usage"]["input_tokens_details"] = details)
        r = @test_logs (:warn, r"System One usage") min_level=Logging.Warn first(_od_ask_with(odd))
        @test r isa SystemOneSuccess && token_usage(r).cached_tokens == 0 && token_usage(r).prompt_tokens == 589
    end
    r = @test_logs (:warn, r"System One usage") min_level=Logging.Warn first(_od_ask_with(
        _od_d02(body -> body["usage"]["output_tokens_details"] = 2)))
    @test r isa SystemOneSuccess && token_usage(r).reasoning_tokens == 0
    r = @test_logs min_level=Logging.Warn first(_od_ask_with(_od_d02(b -> delete!(b, "usage"))))
    @test r isa SystemOneSuccess && token_usage(r) == TokenUsage()
    r = @test_logs min_level=Logging.Warn first(_od_ask_with(_od_d02(b -> b["usage"] = Dict("input_tokens" => 7))))
    @test token_usage(r) == TokenUsage(prompt_tokens=7, total_tokens=7)
    # Cache writes are read as on the Responses API, the 12-hour ones apart, with the tier the
    # reply reports beside its usage.
    written = _od_d02() do body
        body["usage"] = Dict("input_tokens" => 3000, "output_tokens" => 0, "total_tokens" => 3000,
            "input_tokens_details" => Dict("cached_tokens" => 1000, "cache_write_tokens" => 1500,
                "cache_write_tokens_details" => Dict("ttl_30m_tokens" => 300, "ttl_12h_tokens" => 1200)),
            "output_tokens_details" => Dict("reasoning_tokens" => 0))
        body["service_tier"] = "default"
    end
    @test token_usage(first(_od_ask_with(written))) ==
          TokenUsage(prompt_tokens=3000, total_tokens=3000, cached_tokens=1000, cache_write_tokens=1500,
                     cache_write_extended_tokens=1200, service_tier="default")
    odd = _od_d02(body -> body["usage"]["input_tokens_details"]["cache_write_tokens_details"] = 7)
    r = @test_logs (:warn, r"System One usage") min_level=Logging.Warn first(_od_ask_with(odd))
    @test token_usage(r) == TokenUsage(prompt_tokens=589, total_tokens=589)

    # A boolean choice value — which the API types apart from the string — reads as text.
    d03 = _od_fixture("D03")
    req = SystemOneRequest(d03["request"]["input"], ["positive" => choice("Is the review positive?", ["true", "false"]),
                                                     "bare" => choice("Which product is reviewed?", ["blender", "toaster", "kettle"])])
    r, _ = _od_serve((_, _) -> _od_reply(200, JSON.json(d03["body"]))) do
        ask(req; service=_ODMock, config=_OD_CFG)
    end
    @test r isa SystemOneSuccess && r["positive"].choice == "true" && r["positive"].probabilities == Dict("true" => 1.0, "false" => 0.0)
    @test r.response.request_id === nothing                                  # no x-request-id header was sent

    # A refusal is an answer of its own; the others are still answered.
    refused = _od_d02(body -> body["answers"][2] = Dict("type" => "refusal", "name" => "department"))
    r = first(_od_ask_with(refused))
    @test r isa SystemOneSuccess && r["department"] isa RefusalAnswer && r["department"].raw["type"] == "refusal"
    @test r["severity"] isa ScoreAnswer
    # A raw question's answer of any type is accepted; one of a known type comes back typed.
    raw = SystemOneRequest("s", ["rank" => RawQuestion(Dict("type" => "rank", "items" => ["a", "b"])),
                                 "p" => RawQuestion(Dict("type" => "predicate", "instructions" => "Is it?"))])
    r, seen = _od_serve(_od_valid) do
        ask(raw; service=_ODMock, config=_OD_CFG)
    end
    @test r isa SystemOneSuccess && r["rank"] isa UnknownAnswer && r["rank"].type == "rank" && r["p"] isa NoulAnswer
    @test only(_od_header(seen[1], "authorization")) == "Bearer od" && only(_od_header(seen[1], "content-type")) == "application/json"
    @test seen[1].target == "/v1/decisions"
end

@testset "OpenAI Decisions — a reply that does not fit the request is a call error, never a success" begin
    answer_at(i, f) = _od_d02(body -> f(body["answers"][i]))
    malformed = [
        "answers is no array"         => _od_d02(b -> b["answers"] = Dict("angry" => b["answers"][4])),
        "an answer is no object"      => _od_d02(b -> b["answers"][4] = "angry"),
        "a null name"                 => answer_at(4, a -> a["name"] = nothing),
        "no name"                     => answer_at(4, a -> delete!(a, "name")),
        "a name twice"                => answer_at(4, a -> a["name"] = "deadline"),
        "a question unanswered"       => _od_d02(b -> pop!(b["answers"])),
        "an unasked answer"           => _od_d02(b -> push!(b["answers"], Dict("type" => "predicate", "name" => "extra", "probability" => 0.5))),
        "answers out of order"        => _od_d02(b -> reverse!(b["answers"])),
        "no type"                     => answer_at(4, a -> delete!(a, "type")),
        "an unknown type for a predicate" => answer_at(4, a -> a["type"] = "verdict"),
        "a choice answering a predicate" => answer_at(4, a -> a["type"] = "choice"),
        "a probability out of range"  => answer_at(4, a -> a["probability"] = 1.2),
        "a probability as text"       => answer_at(4, a -> a["probability"] = "0.4"),
        "a choice not offered"        => answer_at(2, a -> a["choice"] = "marketing"),
        "a choice value as a number"  => answer_at(2, a -> a["choice"] = 3),
        "choice probabilities as an object" => answer_at(2, a -> a["probabilities"] = Dict("technical" => 1.0)),
        "a choice entry no object"    => answer_at(2, a -> a["probabilities"][1] = 0.0),
        "an offered option missing"   => answer_at(2, a -> deleteat!(a["probabilities"], 1)),
        "an option twice"             => answer_at(2, a -> a["probabilities"][1]["value"] = "technical"),
        "a choice entry without probability" => answer_at(2, a -> delete!(a["probabilities"][3], "probability")),
        "a choice that is not the argmax" => answer_at(2, a -> a["choice"] = "shipping"),
        "a confidence out of range"   => answer_at(2, a -> a["confidence"] = 1.5),
        "a sum far from 1"            => answer_at(2, a -> a["probabilities"][2]["probability"] = 0.9),
        "a level as a float"          => answer_at(1, a -> a["probabilities"][2]["value"] = 1.0),
        "a level as text"             => answer_at(1, a -> a["probabilities"][2]["value"] = "1"),
        "a level as a boolean"        => answer_at(1, a -> a["probabilities"][2]["value"] = true),
        "a negative level"            => answer_at(1, a -> a["probabilities"][1]["value"] = -1),
        "a level twice"               => answer_at(1, a -> a["probabilities"][3]["value"] = 1),
        "a level beyond the rubric"   => answer_at(1, a -> a["probabilities"][3]["value"] = 7),
        "a level without a label"     => answer_at(1, a -> delete!(a["probabilities"][1], "label")),
        "a label that is not the level asked" => answer_at(1, a -> a["probabilities"][1]["label"] = "WRONG"),
        "a score beyond the rubric"   => answer_at(1, a -> a["score"] = 2.5),
        "a score its probabilities contradict" => answer_at(1, a -> a["score"] = 2.0),
        "no model"                    => _od_d02(b -> delete!(b, "model")),
        "not an object"               => "[]",
    ]
    for (label, body) in malformed
        @testset "$(label)" begin
            r, sent = _od_ask_with(body)
            @test r isa SystemOneCallError
            @test sent == 1
            r isa SystemOneCallError && @test r.request_id == "req_od" && r.cause isa Exception
        end
    end
    e = first(_od_ask_with(_od_d02(b -> reverse!(b["answers"]))))
    @test contains(e.error, "not in the order asked") && contains(e.error, "\"angry\", \"deadline\"")
end

@testset "OpenAI Decisions — a non-2xx reply is a failure with the error's code, message and request id" begin
    for tag in ("D05", "D06", "D07", "D08", "D09", "D10")
        fx = _od_fixture(tag)
        r, sent = _od_ask_with(JSON.json(fx["body"]); status=fx["status"], headers=["x-request-id" => fx["headers"]["x-request-id"]])
        err = fx["body"]["error"]
        @test r isa SystemOneFailure && sent == 1
        @test r.status == fx["status"] && r.request_id == fx["headers"]["x-request-id"]
        @test r.error_type == err["code"] && r.message == err["message"] && r.response == JSON.json(fx["body"])
        @test r.retry_after === nothing && !issuccess(r)
    end
    # `type` stands in for a null `code`; an error naming neither has no error type.
    r, _ = _od_ask_with("""{"error":{"message":"Bad.","type":"invalid_request_error","param":null,"code":null}}"""; status=400)
    @test r.error_type == "invalid_request_error" && r.message == "Bad."
    r, _ = _od_ask_with("""{"error":{"message":"Invalid runtime agent assertion.","type":null,"code":"invalid_agent_instance_assertion","param":null},"status":401}"""; status=401)
    @test r.status == 401 && r.error_type == "invalid_agent_instance_assertion" && r.message == "Invalid runtime agent assertion."
    r, _ = _od_ask_with("""{"error":{"message":"No kind."}}"""; status=400)
    @test r.error_type === nothing && r.message == "No kind."
    # A body that is not an OpenAI error is kept, and its text is the message.
    r, _ = _od_ask_with("<html>upstream unavailable</html>"; status=503)
    @test r isa SystemOneFailure && r.status == 503 && r.error_type === nothing && r.message == "<html>upstream unavailable</html>"
    # A rate limit's wait, the millisecond header first.
    r, _ = _od_ask_with("""{"error":{"message":"Slow down.","type":"requests","code":"rate_limit_exceeded"}}"""; status=429,
                        headers=["x-request-id" => "req_429", "retry-after-ms" => "1500", "retry-after" => "2"])
    @test r.error_type == "rate_limit_exceeded" && r.retry_after == 1.5 && r.request_id == "req_429"
    @test contains(sprint(showerror, SystemOneError(r)), "HTTP 429 — Slow down.")
end

# ─── The rounding tolerance ──────────────────────────────────────────────────

# The questions a captured 200 answered, for its answers to be checked against; the state
# plays no part in decoding.
function _od_asked(fx)::SystemOneRequest
    qs = Pair{String,SystemOneQuestion}[]
    for (i, q) in enumerate(fx["request"]["questions"])
        name = get(q, "name", "unnamed_$i")
        push!(qs, name => q["type"] == "predicate" ? noul(q["instructions"]) :
                         q["type"] == "choice" ? choice(q["instructions"], [string(c["value"]) for c in q["choices"]]) :
                         score(q["instructions"], [l["label"] for l in q["levels"]]))
    end
    SystemOneRequest("state", qs; model=fx["request"]["model"])
end

function _od_decoded(fx)
    body = fx["body"]
    for (i, a) in enumerate(body["answers"])
        isnothing(a["name"]) && (a["name"] = "unnamed_$i")
    end
    UniLM.decode_system_one(_OD, _ODMock, _od_asked(fx),
                            HTTP.Response(200, ["x-request-id" => fx["headers"]["x-request-id"]], Vector{UInt8}(JSON.json(body))))
end

_od_dist(ps...) = Dict{String,Float64}("o$i" => p for (i, p) in enumerate(ps))

"Check a Choice answer over options o1… against the Decisions tolerance."
function _od_choice_ok(chosen::Int, ps...)
    q = choice("Pick.", ["o$i" for i in eachindex(ps)])
    a = ChoiceAnswer("o$chosen", 0.5, _od_dist(ps...), Dict{String,Any}())
    req = SystemOneRequest("s", "c" => q)
    resp = SystemOneResponse("gpt-6-luna", Dict{String,SystemOneAnswer}("c" => a), TokenUsage(), nothing, Dict{String,Any}())
    _od_caught(() -> UniLM._check_answers(_OD, req, resp)) === nothing
end

function _od_score_ok(score_value, ps...)
    req = SystemOneRequest("s", "r" => score("Rate.", ["l$i" for i in eachindex(ps)]))
    a = ScoreAnswer(score_value, 0.5, Dict{Int,Any}(i - 1 => "l$i" for i in eachindex(ps)),
                    Dict{Int,Float64}(i - 1 => p for (i, p) in enumerate(ps)), Dict{String,Any}())
    resp = SystemOneResponse("gpt-6-luna", Dict{String,SystemOneAnswer}("r" => a), TokenUsage(), nothing, Dict{String,Any}())
    _od_caught(() -> UniLM._check_answers(_OD, req, resp)) === nothing
end

@testset "OpenAI Decisions — the rounding tolerance holds every captured answer, and no more" begin
    tol(n) = UniLM._probability_tolerance(_OD, n)
    @test tol(2) ≈ 0.01 + 1e-9 && tol(4) ≈ 0.02 + 1e-9 && tol(7) ≈ 0.035 + 1e-9
    # The slack for the choice and the score is one rounding step whatever the number of
    # options: 255 of them widen the sum's allowance, never the winner's.
    @test UniLM._probability_slack(_OD) ≈ 0.01 + 1e-9 && tol(255) > 1
    # Every 200 captured live decodes and passes the checks.
    deviations = Float64[]
    for tag in ("D01", "D02", "D03", "D04", ("D$(i)" for i in 11:20)...)
        fx = _od_fixture(tag)
        r = _od_decoded(fx)
        @test r isa SystemOneSuccess
        for (_, a) in r.response.answers
            a isa Union{ChoiceAnswer,ScoreAnswer} && push!(deviations, abs(sum(values(a.probabilities)) - 1))
        end
    end
    @info "Decisions captures: $(length(deviations)) distributions, largest |Σp − 1| = $(maximum(deviations))"
    @test length(deviations) >= 15

    # The sum: worst-case rounding at the edge passes, one more hundredth fails.
    @test _od_choice_ok(1, 0.25, 0.25, 0.25, 0.23)          # Σ 0.98: 4 options rounded down by 0.005 each
    @test _od_choice_ok(1, 0.26, 0.26, 0.26, 0.24)          # Σ 1.02
    @test !_od_choice_ok(1, 0.25, 0.25, 0.25, 0.22)         # Σ 0.97
    @test !_od_choice_ok(1, 0.27, 0.26, 0.26, 0.24)         # Σ 1.03
    @test _od_choice_ok(1, 0.50, 0.49)                       # two options: Σ 0.99
    @test !_od_choice_ok(1, 0.50, 0.48)                      # Σ 0.98
    @test _od_choice_ok(1, 0.17, 0.14, 0.14, 0.13, 0.13, 0.13, 0.13)    # seven: Σ 0.97, within 0.035
    @test !_od_choice_ok(1, 0.17, 0.14, 0.14, 0.13, 0.13, 0.13, 0.12)   # Σ 0.96
    # The winner: rounding may leave it one hundredth below another, not two.
    @test _od_choice_ok(2, 0.50, 0.50) && _od_choice_ok(1, 0.50, 0.51)
    @test !_od_choice_ok(1, 0.49, 0.51)
    # The score: within its range by the slack.
    @test _od_score_ok(2.01, 0.0, 0.0, 1.0) && _od_score_ok(-0.01, 1.0, 0.0, 0.0)
    @test !_od_score_ok(2.02, 0.0, 0.0, 1.0) && !_od_score_ok(-0.02, 1.0, 0.0, 0.0)
    # A backend that normalises in floating point (Ollama's generic decision models) is held
    # far tighter: the same rounded answer is malformed there.
    req = SystemOneRequest("s", "c" => choice("Pick.", ["o1", "o2", "o3", "o4"]))
    resp = SystemOneResponse("m", Dict{String,SystemOneAnswer}("c" => ChoiceAnswer("o1", 0.5, _od_dist(0.25, 0.25, 0.25, 0.23),
                                                                                 Dict{String,Any}())),
                             TokenUsage(), nothing, Dict{String,Any}())
    @test_throws ArgumentError UniLM._check_answers(UniLM.OllamaSystemOneProtocol(), req, resp)
end

# ─── Recording ───────────────────────────────────────────────────────────────

@testset "OpenAI Decisions — recorded under its own namespace, replayed with no server" begin
    dir = mktempdir()
    state, q = "The package arrived with a broken screen.", "damaged" => noul("Does the customer report a damaged item?")
    live, seen = _od_serve(_od_valid) do
        with_recorded_answers(() -> ask(state, q; service=_ODMock, config=_OD_CFG), dir; mode=:record)
    end
    @test live isa SystemOneSuccess && length(seen) == 1
    rec = JSON.parse(read(only(readdir(dir; join=true)), String))
    @test rec["request"]["path"] == "/openai-decisions/v1/decisions" && !haskey(rec["request"], "key_extra")
    @test rec["response"]["request_id"] == "req_1" && rec["response"]["request_id_header"] == "x-request-id"
    @test JSON.json(rec["request"]["body"]) == seen[1].body
    replayed = with_recorded_answers(() -> ask(state, q; service=_ODMock), dir)
    @test replayed isa SystemOneSuccess && replayed.response.raw == live.response.raw
    @test replayed.response.request_id == "req_1" && replayed["damaged"].noul == 0.5
    # The options are sent, and left out of the key: neither changes an answer, so the same
    # question with them replays the recording made without them.
    other, unsent = _od_serve(_od_valid) do
        with_recorded_answers(() -> ask(state, q; service=_ODMock, prompt_cache_key="k", safety_identifier="u"), dir)
    end
    @test other isa SystemOneSuccess && other.response.raw == live.response.raw && isempty(unsent)
    with_options = mktempdir()
    _, sent = _od_serve(_od_valid) do
        with_recorded_answers(() -> ask(state, q; service=_ODMock, config=_OD_CFG, prompt_cache_key="k"),
                              with_options; mode=:record)
    end
    @test _od_body(only(sent))["prompt_cache_key"] == "k"                       # the service receives them
    @test readdir(with_options) == readdir(dir)                                 # filed under the same key
    @test !haskey(JSON.parse(read(only(readdir(with_options; join=true)), String))["request"]["body"], "prompt_cache_key")
end

@testset "OpenAI Decisions — an image is recorded by its digest, never by its bytes" begin
    dir = mktempdir()
    q = "red" => noul("Is the circle red?")
    live, seen = _od_serve(_od_valid) do
        with_recorded_answers(() -> ask(_OD_PNG, q; service=_ODMock, config=_OD_CFG), dir; mode=:record)
    end
    b64 = base64encode(_OD_PNG.data)
    @test live isa SystemOneSuccess && contains(only(seen).body, b64)              # the service gets the image
    file = only(readdir(dir; join=true))
    text = read(file, String)
    @test !contains(text, b64) && !contains(text, "base64,")
    digest = "data:image/png;sha256," * bytes2hex(UniLM.SHA.sha256(b64))
    @test only(JSON.parse(text)["request"]["body"]["input"])["content"][1]["image_url"] == digest
    keyed = replace(only(seen).body, "data:image/png;base64," * b64 => digest)
    @test basename(file) == bytes2hex(UniLM.SHA.sha256("POST /openai-decisions/v1/decisions\n" * keyed)) * ".json"
    # Replayed with no server; another image is another request, reported by its digest too.
    @test with_recorded_answers(() -> ask(_OD_PNG, q; service=_ODMock), dir).response.raw == live.response.raw
    miss = with_recorded_answers(() -> _od_caught(() -> ask(_OD_JPEG, q; service=_ODMock)), dir)
    @test miss isa ReplayMissError && contains(miss.body, "data:image/jpeg;sha256,") && !contains(miss.body, "base64,")
end

@testset "OpenAI Decisions — ask_each sends each image request as a single ask sends it, byte for byte" begin
    states = [Any["Inspect image $i.", _OD_PNG] for i in 1:3]
    q = "damaged" => noul("Is anything damaged?")
    each, seen_each = _od_serve(_od_valid) do
        ask_each(states, q; service=_ODMock, config=_OD_CFG, ntasks=2)
    end
    alone, seen_alone = _od_serve(_od_valid) do
        [ask(s, q; service=_ODMock, config=_OD_CFG) for s in states]
    end
    @test all(issuccess, each) && all(issuccess, alone)
    @test sort([s.body for s in seen_each]) == sort([s.body for s in seen_alone])
    @test length(unique(s.body for s in seen_each)) == 3
end

# ─── The natural-language verbs on a backend that refuses ───────────────────

_od_route(::nl"the customer wants a refund", ticket) = (:refund, ticket)
_od_route(::nl"the customer reports a bug", ticket) = (:bug, ticket)
_od_keyed(::Val{:refund}, ticket) = :refund
_od_keyed(::Val{:bug}, ticket) = :bug
_od_pair(::nl"a refund", ::nl"urgent", ticket) = (:refund, :urgent)
_od_pair(::nl"a refund", ::nl"not urgent", ticket) = (:refund, :calm)
_od_pair(::nl"a bug", ::nl"urgent", ticket) = (:bug, :urgent)
_od_pair(::nl"a bug", ::nl"not urgent", ticket) = (:bug, :calm)
const _OD_INTENT = (refund = "the customer wants a refund", bug = "the customer reports a bug")

@testset "OpenAI Decisions — the natural-language verbs take a refusal as a decline" begin
    ticket = "My package arrived crushed and the screen is cracked. I want my money back."
    heard, decided = Ref(0), Ref(0)
    hook(r) = (heard[] += 1; nothing)
    policy(a) = (decided[] += 1; a.choice)
    # Answered: the mock picks the first option, and the policy sees it.
    answered, _ = _od_serve(_od_valid) do
        (nl_classify(ticket, _OD_INTENT; service=_ODMock, config=_OD_CFG, decide=policy),
         nl_dispatch(_od_route, ticket; service=_ODMock, config=_OD_CFG))
    end
    @test answered == (:refund, (:refund, ticket)) && decided[] == 1
    decided[] = 0

    out, seen = _od_serve(_od_refusing("classify", "meaning_1", "branch")) do
        [nl_classify(ticket, _OD_INTENT; service=_ODMock, config=_OD_CFG, decide=policy, on_response=hook,
                     fallback = s -> :fallback),
         _od_caught(() -> nl_classify(ticket, _OD_INTENT; service=_ODMock, config=_OD_CFG, min_confidence=0.5)),
         nl_dispatch(_od_route, ticket; service=_ODMock, config=_OD_CFG, decide=policy, on_response=hook,
                     fallback = t -> (:fallback, t)),
         _od_caught(() -> nl_dispatch(_od_route, ticket; service=_ODMock, config=_OD_CFG)),
         nl_dispatch(_od_keyed, ticket; texts=_OD_INTENT, service=_ODMock, config=_OD_CFG, fallback = t -> :keyed_fallback),
         _od_caught(() -> nl_dispatch(_od_keyed, ticket; texts=_OD_INTENT, service=_ODMock, config=_OD_CFG, decide=policy)),
         (@branch ticket service=_ODMock config=_OD_CFG decide=policy on_response=hook begin
             "the customer wants a refund" => :refund
             "the customer reports a bug"  => :bug
             _                             => :branch_fallback
         end),
         _od_caught(() -> @branch ticket service=_ODMock config=_OD_CFG begin
             "the customer wants a refund" => :refund
             "the customer reports a bug"  => :bug
         end)]
    end
    @test length(seen) == 8
    @test out[1] === :fallback && out[3] == (:fallback, ticket) && out[5] === :keyed_fallback && out[7] === :branch_fallback
    @test decided[] == 0                     # no policy saw a refusal
    @test heard[] == 3                       # and the hook ran before each decline
    for (i, question) in ((2, "classify"), (4, "meaning_1"), (6, "meaning_1"), (8, "branch"))
        e = out[i]
        @test e isa DecisionDeclinedError && e.question == question && e.answer isa RefusalAnswer
        @test sprint(showerror, e) == "DecisionDeclinedError: the backend declined to answer $(repr(question)). " *
                                      "Add a fallback, or rephrase the question or the state."
    end

    # Several slots: one refused is a decline of the call, and the first declined slot is reported.
    two, _ = _od_serve(_od_refusing("meaning_2")) do
        [nl_dispatch(_od_pair, ticket; service=_ODMock, config=_OD_CFG, fallback = t -> :fallback),
         _od_caught(() -> nl_dispatch(_od_pair, ticket; service=_ODMock, config=_OD_CFG)),
         _od_caught(() -> nl_dispatch(_od_pair, ticket; service=_ODMock, config=_OD_CFG,
                                      decide=[a -> nothing, a -> a.choice]))]
    end
    @test two[1] === :fallback
    @test two[2] isa DecisionDeclinedError && two[2].question == "meaning_2" && two[2].answer isa RefusalAnswer
    @test two[3] isa DecisionDeclinedError && two[3].question == "meaning_1" && two[3].answer isa ChoiceAnswer
    @test contains(sprint(showerror, two[3]), "the decision policy declined the answer to \"meaning_1\", which selected")
end

# ─── Cost ────────────────────────────────────────────────────────────────────

@testset "OpenAI Decisions — estimated_cost bills input tokens alone, cached ones at the input rate" begin
    result(usage; model="gpt-6-luna", service=OPENAIServiceEndpoint) = SystemOneSuccess(SystemOneResponse(
        model, Dict{String,SystemOneAnswer}(), usage, nothing, Dict{String,Any}(), service))
    rate = DEFAULT_PRICING["gpt-6-luna"].input
    @test rate == 0.10 / 1_000_000                                   # the guide's $0.10 per 1M input tokens
    plain = result(TokenUsage(prompt_tokens=1000, total_tokens=1000))
    cached = result(TokenUsage(prompt_tokens=1000, cached_tokens=600, completion_tokens=5, total_tokens=1005))
    @test estimated_cost(plain) ≈ 1000 * rate
    @test estimated_cost(cached) ≈ 1000 * rate                       # no cache discount, no output charge
    @test UniLM._system_one_cost(_OD, cached, nothing, DEFAULT_PRICING) ≈ 1000 * rate
    # Cache writes, at either lifetime, are input tokens like the rest: no write premium.
    written = result(TokenUsage(prompt_tokens=1000, cached_tokens=200, cache_write_tokens=500,
                                cache_write_extended_tokens=400, completion_tokens=3, total_tokens=1003))
    @test estimated_cost(written) ≈ 1000 * rate
    @test @inferred(UniLM._system_one_cost(_OD, written, nothing, DEFAULT_PRICING)) ≈ 1000 * rate
    # Decisions has rates of its own: a tier a reply names prices nothing differently.
    @test estimated_cost(result(TokenUsage(prompt_tokens=1000, total_tokens=1000, service_tier="flex"))) ≈ 1000 * rate
    @test estimated_cost(result(TokenUsage(prompt_tokens=272_000))) ≈ 272_000 * rate
    @test estimated_cost(result(TokenUsage(prompt_tokens=272_001))) ≈ 2 * 272_001 * rate   # long context: twice
    # `model` and `pricing` keep their meaning.
    custom = Dict{String,UniLM.PriceRow}("my-luna" => UniLM._price(1.0, 0.0, 0.0))
    @test estimated_cost(plain; model="my-luna", pricing=custom) ≈ 1000 / 1_000_000
    @test (@test_logs (:warn, r"no price row") estimated_cost(result(TokenUsage(prompt_tokens=10); model="unlisted-decider"))) == 0.0
    # The same usage from another backend is priced the general way, cached tokens at the cached rate.
    elsewhere = result(TokenUsage(prompt_tokens=1000, cached_tokens=600); service=TYPESAFEServiceEndpoint)
    @test estimated_cost(elsewhere) ≈ 400 * rate + 600 * DEFAULT_PRICING["gpt-6-luna"].cached_input
    @test estimated_cost(SystemOneSuccess(SystemOneResponse("gpt-6-luna", Dict{String,SystemOneAnswer}(),
        TokenUsage(prompt_tokens=1000, cached_tokens=600), nothing, Dict{String,Any}()))) ≈
          400 * rate + 600 * DEFAULT_PRICING["gpt-6-luna"].cached_input            # built by hand: no backend
end
