# test/system_one.jl — the generic System One surface, zero-spend.
#
# The protocol seam (trait, encoder, decoder, wire), the answer checks every backend's
# answers pass, the ambient backend, the multipart send path, and the byte-identity
# gate: every TypeSafe request recorded in docs/recorded_answers, rebuilt from public
# constructors, must reproduce its key and its body through the seam and replay its
# answers with no key set. Local mocks only; nothing here reaches a network.
#
# Requires test/platform_seam_fixtures.jl (SeamProbe) to be included first.

using Sockets, Logging, Base64

const _S1_CFG = UniLM.RequestConfig(max_attempts=1, total_deadline=30.0)
const _S1_RECORDINGS = joinpath(pkgdir(UniLM), "docs", "recorded_answers")
const _S1_FIXTURES = joinpath(@__DIR__, "fixtures", "typesafe")

# A System One endpoint that declares no protocol, as a TypeSafe-compatible gateway
# would: it speaks TypeSafe's wire through the trait's fallback.
struct _S1Mock <: UniLM.ServiceEndpoint end
const _S1_BASE = Ref("")
UniLM._resolve_base_url(::Type{_S1Mock}) = _S1_BASE[]
UniLM.auth_header(::Type{_S1Mock}) = ["Authorization" => "Bearer s1", "Content-Type" => "application/json"]
UniLM.provider_capabilities(::Type{_S1Mock}) = Set([:system_one, :models])

const _S1Seen = @NamedTuple{method::String, target::String, headers::Vector{Pair{String,String}}, body::Vector{UInt8}}

"Run `f()` against a local server answering request `n` with `respond(n, req)`; return `(f(), requests)`."
function _s1_serve(f::Function, respond::Function)
    seen = _S1Seen[]
    guard = ReentrantLock()
    handler = function (req)
        n = @lock guard begin
            push!(seen, (method=String(req.method), target=String(req.target),
                         headers=Pair{String,String}[String(k) => String(v) for (k, v) in req.headers],
                         body=applicable(copy, req.body) ? Vector{UInt8}(copy(req.body)) : UInt8[]))
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
        _S1_BASE[] = "http://127.0.0.1:$port"
        try
            return f(), seen
        finally
            close(server)
        end
    end
    error("could not bind an ephemeral port for the System One mock")
end

_s1_reply(status::Int, body::AbstractString; id::Union{Nothing,String}=nothing) =
    HTTP.Response(status, Pair{String,String}["Content-Type" => "application/json";
                                              isnothing(id) ? Pair{String,String}[] : ["x-typesafe-request-id" => id]],
                  Vector{UInt8}(body))

# A valid answer to every question of a System One JSON body: the first option, the
# lowest level, an even noul, and an answer of the question's own type to a raw one.
function _s1_answers(body::AbstractDict)::Dict{String,Any}
    out = Dict{String,Any}()
    for (name, q) in body["questions"]
        t = q["type"]
        out[name] = if t == "choice"
            opts = collect(keys(q["criteria"]))
            Dict{String,Any}("type" => t, "choice" => first(opts), "confidence" => 1.0,
                             "probabilities" => Dict{String,Any}(o => (i == 1 ? 1.0 : 0.0) for (i, o) in enumerate(opts)))
        elseif t == "score"
            n = length(q["criteria"])
            Dict{String,Any}("type" => t, "score" => 0.0, "confidence" => 1.0,
                             "legend" => Dict{String,Any}(string(i - 1) => q["criteria"][i] for i in 1:n),
                             "probabilities" => Dict{String,Any}(string(i - 1) => (i == 1 ? 1.0 : 0.0) for i in 1:n))
        elseif t == "noul"
            Dict{String,Any}("type" => t, "noul" => 0.5)
        else
            Dict{String,Any}("type" => t, "echo" => true)
        end
    end
    Dict{String,Any}("model" => "jev-1.13.0", "answers" => out,
                     "usage" => Dict{String,Any}("input_tokens" => 10, "output_tokens" => 1))
end

_s1_text(r::_S1Seen) = String(copy(r.body))
_s1_sent(r::_S1Seen) = JSON.parse(_s1_text(r))

"Every request answered validly; the models listing too."
_s1_valid(n, req) = String(req.target) == "/v1/models" ?
    _s1_reply(200, """{"models":[{"name":"mock-model","description":"listed","release_date":"2026-10-19"}]}""") :
    _s1_reply(200, JSON.json(_s1_answers(JSON.parse(String(copy(req.body))))); id="req_$n")

_s1_caught(f::Function) = try f(); nothing catch e; e end

# The documented recording key, computed here from its definition.
_s1_key(method::AbstractString, path::AbstractString, body::AbstractString) =
    bytes2hex(UniLM.SHA.sha256(string(method, " ", path, "\n", body)))

# ─── Byte identity ───────────────────────────────────────────────────────────

# The typed request a recorded body was made from, rebuilt with public constructors only.
function _s1_rebuild(body::AbstractDict)::SystemOneRequest
    qs = Pair{String,SystemOneQuestion}[]
    for (name, q) in body["questions"]
        ins = get(q, "instructions", nothing)
        t = q["type"]
        question = if t == "choice"
            choice(ins, [k => v for (k, v) in q["criteria"]])
        elseif t == "score"
            score(ins, collect(q["criteria"]))
        elseif t == "noul"
            c = get(q, "criteria", nothing)
            NoulQuestion(ins, isnothing(c) ? nothing : NoulCriteria(get(c, "true", nothing), get(c, "false", nothing)))
        else
            RawQuestion(q)
        end
        push!(qs, name => question)
    end
    SystemOneRequest(body["state"], qs; model=body["model"])
end

# The recorded answer a typed answer was decoded from, field by field.
_s1_same(a::ChoiceAnswer, raw::AbstractDict) = a.choice == raw["choice"] && a.confidence == raw["confidence"] &&
    a.probabilities == Dict{String,Float64}(k => v for (k, v) in raw["probabilities"])
_s1_same(a::ScoreAnswer, raw::AbstractDict) = a.score == raw["score"] && a.confidence == raw["confidence"] &&
    a.probabilities == Dict{Int,Float64}(parse(Int, k) => v for (k, v) in raw["probabilities"]) &&
    a.legend == Dict{Int,Any}(parse(Int, k) => v for (k, v) in raw["legend"])
_s1_same(a::NoulAnswer, raw::AbstractDict) = a.noul == raw["noul"]
_s1_same(::SystemOneAnswer, ::AbstractDict) = false

@testset "System One — every recorded TypeSafe request rebuilds to its key and body, and replays keyless" begin
    corpus = Tuple{String,Any}[]
    for f in sort!(readdir(_S1_RECORDINGS; join=true))
        endswith(f, ".json") || continue
        rec = JSON.parse(read(f, String))
        # A recording keyed on more than its JSON body cannot be rebuilt from the body alone.
        rec["request"]["path"] == "/v1/systemone" && !haskey(rec["request"], "key_extra") || continue
        push!(corpus, (splitext(basename(f))[1], rec))
    end
    keys_ok = bodies_ok = replays_ok = 0
    withenv(UniLM.TYPESAFE_API_KEY => nothing, UniLM.TYPESAFE_DEFAULT_MODEL_ENV => nothing,
            UniLM.TYPESAFE_BASE_URL_ENV => nothing) do
        with_recorded_answers(_S1_RECORDINGS) do
            for (key, rec) in corpus
                req = _s1_rebuild(rec["request"]["body"])
                wire = UniLM.encode_system_one(UniLM.TypeSafeProtocol(), TYPESAFEServiceEndpoint, req)
                # (i) the key: TypeSafe keeps the plain path and keys nothing beyond the body.
                keys_ok += wire.path == "/v1/systemone" && isempty(wire.key_extra) && isempty(wire.parts) &&
                           _s1_key("POST", wire.path, wire.body) == key
                # (ii) the body: a recording stores it parsed, so it is compared parsed, key order included.
                bodies_ok += JSON.json(JSON.parse(wire.body)) == JSON.json(rec["request"]["body"])
                # (iii) the replay: the same success, field by field, with no key set.
                r = ask(req)
                recorded = rec["response"]["body"]
                replays_ok += r isa SystemOneSuccess && r.response.raw == recorded &&
                              r.response.model == recorded["model"] &&
                              r.response.request_id == rec["response"]["request_id"] &&
                              r.response.service === TYPESAFEServiceEndpoint &&
                              sort!(collect(keys(r))) == sort!(collect(keys(recorded["answers"]))) &&
                              all(name -> _s1_same(r[name], recorded["answers"][name]), keys(r))
            end
        end
    end
    n = length(corpus)
    @info "System One byte-identity gate: $(keys_ok)/$(n) keys, $(bodies_ok)/$(n) bodies, $(replays_ok)/$(n) keyless replays"
    @test n >= 90                    # the committed corpus: a gate over nothing would pass vacuously
    @test keys_ok == n
    @test bodies_ok == n
    @test replays_ok == n            # so no committed recording is rejected by the answer checks
end

@testset "System One — TypeSafe's probability tolerance is at least ten times the largest deviation it has shown" begin
    # Every Choice and Score distribution the repository holds: the committed recordings
    # and the live captures kept as fixtures.
    bodies = Any[]
    for f in readdir(_S1_RECORDINGS; join=true)
        rec = JSON.parse(read(f, String); dicttype=Dict{String,Any})
        get(rec["request"], "path", "") == "/v1/systemone" && push!(bodies, rec["response"]["body"])
    end
    for f in readdir(_S1_FIXTURES; join=true)
        endswith(f, ".response.json") && push!(bodies, JSON.parse(read(f, String); dicttype=Dict{String,Any}))
    end
    deviations = Tuple{Int,Float64}[]
    for b in bodies, (_, a) in get(b, "answers", Dict{String,Any}())
        get(a, "type", "") in ("choice", "score") || continue
        p = Float64[v for v in values(a["probabilities"])]
        push!(deviations, (length(p), abs(sum(p) - 1)))
    end
    worst = maximum(last, deviations)
    @info "TypeSafe probability sums: $(length(deviations)) distributions, largest |Σp − 1| = $(worst)"
    @test length(deviations) >= 100
    @test all(((n, d),) -> UniLM._probability_tolerance(UniLM.TypeSafeProtocol(), n) >= 10 * worst, deviations)
    @test UniLM._probability_tolerance(UniLM.TypeSafeProtocol(), 1) >= 10 * worst
end

# ─── The protocol seam ───────────────────────────────────────────────────────

# A second backend on the same route: its own protocol, filed under the namespace "x".
struct _XProtocol <: UniLM.SystemOneProtocol end
struct _XEndpoint <: UniLM.ServiceEndpoint end
UniLM.provider_capabilities(::Type{_XEndpoint}) = Set([:system_one])
UniLM.system_one_protocol(::Type{_XEndpoint}) = _XProtocol()
UniLM.auth_header(::Type{_XEndpoint}) = ["Content-Type" => "application/json"]
UniLM._system_one_namespace(::_XProtocol) = "x"
UniLM._system_one_default_model(::_XProtocol, _) = "x-default"
UniLM.encode_system_one(p::_XProtocol, service, r::SystemOneRequest) = UniLM.SystemOneWire(
    "POST", _S1_BASE[] * "/v1/systemone", "/v1/systemone",
    UniLM._system_one_body(UniLM._system_one_json(r, UniLM._system_one_model(p, service, r))))
UniLM.decode_system_one(::_XProtocol, service, r::SystemOneRequest, resp::HTTP.Response) =
    UniLM.decode_system_one(UniLM.TypeSafeProtocol(), service, r, resp)

# A protocol with nothing of its own, and an endpoint that declares no capabilities.
struct _BareProtocol <: UniLM.SystemOneProtocol end
struct _BareEndpoint <: UniLM.ServiceEndpoint end
UniLM.provider_capabilities(::Type{_BareEndpoint}) = Set([:system_one, :models])
UniLM.system_one_protocol(::Type{_BareEndpoint}) = _BareProtocol()
struct _S1Undeclared <: UniLM.ServiceEndpoint end

@testset "System One — the trait keeps every endpoint without a protocol on TypeSafe's wire" begin
    @test UniLM.system_one_protocol(TYPESAFEServiceEndpoint) === UniLM.TypeSafeProtocol()
    @test UniLM.system_one_protocol(SeamProbe) === UniLM.TypeSafeProtocol()
    @test UniLM.system_one_protocol(_S1Mock) === UniLM.TypeSafeProtocol()
    isdefined(@__MODULE__, :SemanticMock) &&
        @test UniLM.system_one_protocol(getfield(@__MODULE__, :SemanticMock)) === UniLM.TypeSafeProtocol()
    @test UniLM.system_one_protocol(_XEndpoint) === _XProtocol()

    # The mock gateway reaches TypeSafe's path with TypeSafe's body and the endpoint's own auth.
    r, seen = _s1_serve(_s1_valid) do
        ask("s", "q" => noul("Is it?"); service=_S1Mock, model="jev-1.13.0", config=_S1_CFG)
    end
    @test r isa SystemOneSuccess && r.response.request_id == "req_1" && r.response.service === _S1Mock
    @test seen[1].target == "/v1/systemone"
    @test _s1_text(seen[1]) == """{"state":"s","model":"jev-1.13.0","questions":{"q":{"type":"noul","instructions":"Is it?"}}}"""
    @test any(h -> lowercase(first(h)) == "authorization" && last(h) == "Bearer s1", seen[1].headers)

    # Hooks the protocols define, and the fallbacks that fail closed.
    ts = UniLM.TypeSafeProtocol()
    @test UniLM._system_one_namespace(ts) == "" && UniLM._system_one_namespace(_BareProtocol()) == "_BareProtocol"
    @test !UniLM._allows_refusal(ts) && UniLM._allows_raw(ts)
    @test !UniLM._allows_refusal(_BareProtocol()) && !UniLM._allows_raw(_BareProtocol())
    withenv(UniLM.TYPESAFE_DEFAULT_MODEL_ENV => nothing) do
        @test UniLM._system_one_default_model(ts, TYPESAFEServiceEndpoint) == "jev-latest"
    end
    @test_throws ArgumentError UniLM._system_one_default_model(_BareProtocol(), _BareEndpoint)
    for call in (() -> ask("s", noul("q?"); service=_BareEndpoint),
                 () -> list_models(; service=_BareEndpoint),
                 () -> list_models(; service=_XEndpoint))
        e = _s1_caught(call)
        @test e isa ArgumentError && (contains(e.msg, "_Bare") || contains(e.msg, "_XEndpoint"))
    end
    # An endpoint that declares nothing is refused by name, never with a MethodError.
    for call in (() -> ask("s", noul("q?"); service=_S1Undeclared), () -> with_system_one(() -> 1, _S1Undeclared))
        e = _s1_caught(call)
        @test e isa ArgumentError && contains(e.msg, "_S1Undeclared")
    end
    e = _s1_caught(() -> ask("s", noul("q?"); service=OPENAIServiceEndpoint))
    @test e isa ArgumentError && contains(e.msg, "System One API is not supported by OPENAIServiceEndpoint")
end

@testset "System One — another backend's namespace never replays TypeSafe's recordings" begin
    dir = mktempdir()
    # The same bytes on both backends: only the namespace can tell their recordings apart.
    req = SystemOneRequest("Where is my parcel?", ["team" => choice("Which team?", ["billing", "shipping"])];
                           model="jev-1.13.0")
    # Recorded through TypeSafe's protocol.
    _, seen_a = _s1_serve(_s1_valid) do
        with_recorded_answers(() -> ask(req; service=_S1Mock, config=_S1_CFG), dir; mode=:record)
    end
    @test length(seen_a) == 1 && length(readdir(dir)) == 1
    # The same request to the other backend: a miss in :replay, a live call in :record_missing.
    miss = with_recorded_answers(() -> _s1_caught(() -> ask(req; service=_XEndpoint, config=_S1_CFG)), dir)
    @test miss isa ReplayMissError && miss.path == "/x/v1/systemone"
    @test miss.body == JSON.json(req)                         # byte for byte the body TypeSafe's recording has
    @test miss.key == _s1_key("POST", "/x/v1/systemone", miss.body)
    @test only(readdir(dir)) == _s1_key("POST", "/v1/systemone", miss.body) * ".json"
    # A request that names no model takes the other backend's default.
    unnamed = with_recorded_answers(() -> _s1_caught(() -> ask("s", noul("q?"); service=_XEndpoint)), dir)
    @test unnamed isa ReplayMissError && JSON.parse(unnamed.body)["model"] == "x-default"
    msg = sprint(showerror, miss)
    @test contains(msg, "asking \"team\"") && !contains(msg, "TYPESAFE_API_KEY") && contains(msg, "access to the provider")
    r, seen_b = _s1_serve(_s1_valid) do
        with_recorded_answers(() -> ask(req; service=_XEndpoint, config=_S1_CFG), dir; mode=:record_missing)
    end
    @test r isa SystemOneSuccess && r.response.service === _XEndpoint && length(seen_b) == 1
    paths = sort!([JSON.parse(read(f, String))["request"]["path"] for f in readdir(dir; join=true)])
    @test paths == ["/v1/systemone", "/x/v1/systemone"]
    # The committed recordings answer TypeSafe's requests only.
    docs = withenv(UniLM.TYPESAFE_API_KEY => nothing) do
        with_recorded_answers(_S1_RECORDINGS) do
            first_recorded = first(f for f in readdir(_S1_RECORDINGS; join=true)
                                   if JSON.parse(read(f, String))["request"]["path"] == "/v1/systemone")
            body = JSON.parse(read(first_recorded, String))["request"]["body"]
            (ask(_s1_rebuild(body)), _s1_caught(() -> ask(_s1_rebuild(body); service=_XEndpoint)))
        end
    end
    @test docs[1] isa SystemOneSuccess && docs[2] isa ReplayMissError
end

@testset "System One — a wire names a keyed path, and its parts, safely" begin
    W = UniLM.SystemOneWire
    @test W("POST", "http://h/v1/systemone", "/v1/systemone", "{}").key_extra == ""
    @test isempty(W("POST", "u", "/p", "{}").parts) && isempty(W("POST", "u", "/p", "{}").headers)
    @test_throws ArgumentError W("POST", "u", "v1/systemone", "{}")
    @test_throws ArgumentError W("POST", "u", "/v1/systemone?key=k", "{}")
    @test_throws ArgumentError W("", "u", "/p", "{}")
    png = ImageAttachment(UInt8[0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 0x01])
    @test_throws ArgumentError W("POST", "u", "/p", "{}"; parts=["request" => png])
    @test_throws ArgumentError W("POST", "u", "/p", "{}"; parts=["a" => png, "a" => png])
    @test_throws ArgumentError W("POST", "u", "/p", "{}"; parts=[" " => png])
    # A protocol without an encoder or a decoder fails by name.
    e = _s1_caught(() -> UniLM.encode_system_one(_BareProtocol(), _BareEndpoint, SystemOneRequest("s", noul("q?"))))
    @test e isa ArgumentError && contains(e.msg, "_BareProtocol")
    e = _s1_caught(() -> UniLM.decode_system_one(_BareProtocol(), _BareEndpoint, SystemOneRequest("s", noul("q?")),
                                                 HTTP.Response(200, "{}")))
    @test e isa ArgumentError && contains(e.msg, "_BareProtocol")
end

# ─── Multipart ───────────────────────────────────────────────────────────────

# A backend that sends the images of a request's `images` option as parts, keyed by
# their digests: the multipart path of the exchange, exercised without TypeSafe.
struct _PartsProtocol <: UniLM.SystemOneProtocol end
struct _PartsEndpoint <: UniLM.ServiceEndpoint end
UniLM.provider_capabilities(::Type{_PartsEndpoint}) = Set([:system_one])
UniLM.system_one_protocol(::Type{_PartsEndpoint}) = _PartsProtocol()
UniLM.auth_header(::Type{_PartsEndpoint}) = ["Authorization" => "Bearer parts", "Content-Type" => "application/json"]
UniLM._system_one_namespace(::_PartsProtocol) = "parts"
UniLM._system_one_default_model(::_PartsProtocol, _) = "parts-model"
function UniLM.encode_system_one(p::_PartsProtocol, service, r::SystemOneRequest)
    images = r.options.images
    parts = Pair{String,ImageAttachment}["asset-$(i - 1)" => img for (i, img) in enumerate(images)]
    UniLM.SystemOneWire("POST", _S1_BASE[] * "/v1/systemone", "/v1/systemone",
        UniLM._system_one_body(UniLM._system_one_json(r, UniLM._system_one_model(p, service, r)));
        parts, headers=["X-Gate" => "on"],
        key_extra=join(("$(name) $(bytes2hex(UniLM.SHA.sha256(img.data)))" for (name, img) in parts), '\n'))
end
UniLM.decode_system_one(::_PartsProtocol, service, r::SystemOneRequest, resp::HTTP.Response) =
    UniLM.decode_system_one(UniLM.TypeSafeProtocol(), service, r, resp)

const _S1_PNG = ImageAttachment(UInt8[0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 0x00, 0x0d, 0x0a, 0xff, 0x01, 0x02])
const _S1_JPEG = ImageAttachment(UInt8[0xff, 0xd8, 0xff, 0xe0, 0x0d, 0x0a, 0x2d, 0x2d, 0x00, 0x10])

_s1_header(r::_S1Seen, name::String) = [last(h) for h in r.headers if lowercase(first(h)) == lowercase(name)]

"The JSON part and the image parts of a recorded multipart request, as HTTP.jl parses them."
function _s1_form(r::_S1Seen)
    ctype = only(_s1_header(r, "content-type"))
    parts = HTTP.parse_multipart_form(ctype, r.body)
    (ctype = ctype, parts = parts)
end

_s1_multipart_ok(n, req) = begin
    form = HTTP.parse_multipart_form(HTTP.header(req, "Content-Type"), Vector{UInt8}(copy(req.body)))
    _s1_reply(200, JSON.json(_s1_answers(JSON.parse(String(read(form[1].data))))); id="req_$n")
end

@testset "System One — a wire with parts is sent as multipart/form-data, rebuilt for every attempt" begin
    q = "q" => noul("Is the shape red?")
    images = [_S1_PNG, _S1_JPEG]
    expected_body = """{"state":"Describe the pictures.","model":"parts-model","questions":{"q":{"type":"noul","instructions":"Is the shape red?"}}}"""
    # The first attempt meets a 503; the retry must send a complete form again.
    script(n, req) = n == 1 ? _s1_reply(503, """{"detail":"overloaded"}""") : _s1_multipart_ok(n, req)
    r, seen = _s1_serve(script) do
        ask("Describe the pictures.", q; service=_PartsEndpoint, images,
            config=UniLM.RequestConfig(max_attempts=2, total_deadline=30.0))
    end
    @test r isa SystemOneSuccess && r["q"].noul == 0.5
    @test length(seen) == 2
    for s in seen
        f = _s1_form(s)
        @test startswith(f.ctype, "multipart/form-data; boundary=")
        @test length(_s1_header(s, "content-type")) == 1            # no JSON Content-Type beside the form's
        @test only(_s1_header(s, "authorization")) == "Bearer parts" && only(_s1_header(s, "x-gate")) == "on"
        @test [p.name for p in f.parts] == ["request", "asset-0", "asset-1"]
        @test f.parts[1].filename === nothing && f.parts[1].contenttype == "application/json"
        @test String(read(f.parts[1].data)) == expected_body
        @test [p.filename for p in f.parts[2:3]] == ["asset-0", "asset-1"]
        @test [p.contenttype for p in f.parts[2:3]] == ["image/png", "image/jpeg"]
        @test [read(p.data) for p in f.parts[2:3]] == [_S1_PNG.data, _S1_JPEG.data]
    end

    # Recorded under the JSON part and the digests: never the image bytes.
    dir = mktempdir()
    live, seen_rec = _s1_serve(_s1_multipart_ok) do
        with_recorded_answers(() -> ask("Describe the pictures.", q; service=_PartsEndpoint, images, config=_S1_CFG),
                              dir; mode=:record)
    end
    @test live isa SystemOneSuccess && length(seen_rec) == 1
    file = only(readdir(dir; join=true))
    text = read(file, String)
    rec = JSON.parse(text)
    digests = "asset-0 $(bytes2hex(UniLM.SHA.sha256(_S1_PNG.data)))\nasset-1 $(bytes2hex(UniLM.SHA.sha256(_S1_JPEG.data)))"
    @test rec["request"]["path"] == "/parts/v1/systemone" && rec["request"]["key_extra"] == digests
    @test JSON.json(rec["request"]["body"]) == expected_body
    @test basename(file) == _s1_key("POST", "/parts/v1/systemone", expected_body * "\n" * digests) * ".json"
    bytes = read(file)
    for img in images
        @test isnothing(findfirst(img.data, bytes))                  # no image byte sequence in the file
        @test !contains(text, base64encode(img.data))
    end
    # And it replays with no server at all.
    replayed = with_recorded_answers(() -> ask("Describe the pictures.", q; service=_PartsEndpoint, images), dir)
    @test replayed isa SystemOneSuccess && replayed.response.raw == live.response.raw
    # Another image is another request.
    other = with_recorded_answers(() -> _s1_caught(() -> ask("Describe the pictures.", q; service=_PartsEndpoint,
                                                             images=[_S1_JPEG, _S1_PNG])), dir)
    @test other isa ReplayMissError && endswith(other.body, "\n" * "asset-0 $(bytes2hex(UniLM.SHA.sha256(_S1_JPEG.data)))\n" *
                                                 "asset-1 $(bytes2hex(UniLM.SHA.sha256(_S1_PNG.data)))")

    # Without parts the body is sent as it is, with the endpoint's JSON Content-Type.
    plain, seen_plain = _s1_serve(_s1_valid) do
        ask("s", q; service=_PartsEndpoint, images=ImageAttachment[], config=_S1_CFG)
    end
    @test plain isa SystemOneSuccess && only(_s1_header(seen_plain[1], "content-type")) == "application/json"
    @test _s1_text(seen_plain[1]) == """{"state":"s","model":"parts-model","questions":{"q":{"type":"noul","instructions":"Is the shape red?"}}}"""
end

# ─── The ambient backend ─────────────────────────────────────────────────────

@testset "System One — with_system_one sets the backend and model of every call that names none" begin
    q = "q" => noul("Is it?")
    models(seen) = [_s1_sent(s)["model"] for s in seen if s.target == "/v1/systemone"]
    out, seen = _s1_serve(_s1_valid) do
        withenv(UniLM.TYPESAFE_DEFAULT_MODEL_ENV => nothing) do
            with_system_one(_S1Mock; model="m-scope") do
                [ask("s", q; config=_S1_CFG),                                   # the scope's backend and model
                 ask("s", q; model="m-call", config=_S1_CFG),                   # the call's model wins
                 ask("s", q; service=_S1Mock, config=_S1_CFG),                  # a named service: its default
                 fetch(Threads.@spawn ask("s", q; config=_S1_CFG)),             # spawned tasks inherit the scope
                 with_system_one(() -> ask("s", q; config=_S1_CFG), _S1Mock; model="m-inner"),   # innermost wins
                 with_system_one(() -> ask("s", q; config=_S1_CFG), _S1Mock),   # no model: the backend's default
                 list_models(; config=_S1_CFG)]                                 # the listing follows the scope
            end
        end
    end
    @test all(issuccess, out)
    @test models(seen) == ["m-scope", "m-call", "jev-latest", "m-scope", "m-inner", "jev-latest"]
    @test seen[end].target == "/v1/models" && out[end] isa TypeSafeModelsSuccess
    @test all(r -> r.response.service === _S1Mock, out[1:end-1])

    # Outside every scope: TypeSafe, which a keyless process cannot reach.
    withenv(UniLM.TYPESAFE_API_KEY => nothing) do
        r = ask("s", q; config=_S1_CFG)
        @test r isa SystemOneCallError && contains(r.error, "TYPESAFE_API_KEY")
    end

    # Refused before f runs.
    ran = Ref(false)
    for (service, kw) in ((OPENAIServiceEndpoint, (;)), (_S1Mock, (model="  ",)), ("typesafe", (;)))
        e = _s1_caught(() -> with_system_one(() -> (ran[] = true), service; kw...))
        @test e isa ArgumentError
    end
    @test !ran[]
end

_s1_route(::Val{:refund}, ticket) = :refund
_s1_route(::Val{:bug}, ticket) = :bug

@testset "System One — the natural-language verbs follow with_system_one when they name no service" begin
    intent = (refund = "the customer wants a refund", bug = "the customer reports a bug")
    picked, seen = _s1_serve(_s1_valid) do
        with_system_one(_S1Mock; model="m-nl") do
            classified = nl_classify("I want my money back", intent; config=_S1_CFG)
            dispatched = nl_dispatch(_s1_route, "I want my money back"; texts=intent, config=_S1_CFG)
            branched = @branch "I want my money back" config=_S1_CFG begin
                "a refund" => :refund
                "a bug"    => :bug
            end
            (classified, dispatched, branched)
        end
    end
    @test picked == (:refund, :refund, :refund)                # the mock answers with the first option
    @test length(seen) == 3 && all(s -> s.target == "/v1/systemone" && _s1_sent(s)["model"] == "m-nl", seen)
end

@testset "System One — the default model is resolved when the request is sent" begin
    req = SystemOneRequest("s", "q" => noul("Is it?"))          # built with no model
    @test req.model === nothing
    _, seen = _s1_serve(_s1_valid) do
        withenv(UniLM.TYPESAFE_DEFAULT_MODEL_ENV => "jev-preview") do
            ask(req; service=_S1Mock, config=_S1_CFG)
        end
        withenv(UniLM.TYPESAFE_DEFAULT_MODEL_ENV => nothing) do
            ask(req; service=_S1Mock, config=_S1_CFG)
        end
    end
    @test [_s1_sent(s)["model"] for s in seen] == ["jev-preview", "jev-latest"]
    @test req.model === nothing                                  # the request itself is unchanged
    # The System One JSON form writes a model only when the request names one.
    @test JSON.json(req) == """{"state":"s","questions":{"q":{"type":"noul","instructions":"Is it?"}}}"""
end

# ─── Construction and local refusals ─────────────────────────────────────────

@testset "System One — the types hold only universal invariants; TypeSafe's limits are refused before sending" begin
    # No upper bound on the types.
    @test length(choice("pick", ["o$i" for i in 1:300]).criteria) == 300
    @test length(score("rate", ["l$i" for i in 1:14]).criteria) == 14
    # The universal ones stay.
    @test_throws ArgumentError choice("pick", String[])
    @test_throws ArgumentError choice("pick", ["  "])
    @test_throws ArgumentError choice("pick", [" " => "blank after strip"])
    @test_throws ArgumentError score("rate", [])
    @test_throws ArgumentError score("rate", ["ok", nothing])
    @test_throws ArgumentError SystemOneRequest("s", [" " => noul("q?")])
    @test_throws ArgumentError SystemOneRequest("s", ["a" => noul("q?")]; model=" ")
    # A vector of pairs of one question type is accepted as it is.
    @test first.(SystemOneRequest("s", ["a" => noul("x?"), "b" => noul("y?")], "m").questions) == ["a", "b"]
    @test SystemOneRequest("s", Pair{String,NoulQuestion}["a" => noul("x?")], "m").model == "m"

    at = ["o$i" for i in 1:255]
    sent, seen = _s1_serve(_s1_valid) do
        [ask("s", "c" => choice("pick", at); service=_S1Mock, config=_S1_CFG),
         ask("s", "r" => score("rate", ["l$i" for i in 1:10]); service=_S1Mock, config=_S1_CFG)]
    end
    @test all(r -> r isa SystemOneSuccess, sent) && length(seen) == 2
    refused, seen = _s1_serve(_s1_valid) do
        c = choice("pick", ["a", "b"])
        for i in 1:300; c.criteria["o$i"] = nothing; end               # grown after it was built
        dup = SystemOneRequest("s", ["q" => noul("x?"), "p" => noul("y?")])
        push!(dup.questions, "q" => noul("again?"))                     # a repeated name, pushed in later
        emptied = choice("pick", ["a"]); delete!(emptied.criteria, "a")
        [_s1_caught(() -> ask("s", "c" => choice("pick", [at; "o256"]); service=_S1Mock)),
         _s1_caught(() -> ask("s", "r" => score("rate", ["l$i" for i in 1:11]); service=_S1Mock)),
         _s1_caught(() -> ask("s", "c" => c; service=_S1Mock)),
         _s1_caught(() -> ask(dup; service=_S1Mock)),
         _s1_caught(() -> ask("s", "c" => emptied; service=_S1Mock))]
    end
    @test isempty(seen)                                                 # nothing was sent
    @test all(e -> e isa ArgumentError, refused)
    @test contains(refused[1].msg, "TypeSafe") && contains(refused[1].msg, "255") && contains(refused[1].msg, "256")
    @test contains(refused[2].msg, "TypeSafe") && contains(refused[2].msg, "10") && contains(refused[2].msg, "11")
    @test contains(refused[3].msg, "302")
    @test contains(refused[4].msg, "duplicate question name \"q\"")
    @test contains(refused[5].msg, "at least one option")
end

@testset "System One — malformed local input is an ArgumentError before any I/O" begin
    q = noul("Is it?")
    table = (refund = "the customer wants a refund", other = "anything else")
    errors, seen = _s1_serve(_s1_valid) do
        [_s1_caught(() -> ask(42, q; service=_S1Mock)),
         _s1_caught(() -> ask(nothing, q; service=_S1Mock)),
         _s1_caught(() -> nl_classify(42, table; service=_S1Mock)),
         _s1_caught(() -> ask(Dict("x" => NaN), q; service=_S1Mock)),
         _s1_caught(() -> ask("s", "c" => choice("pick", (a = (weight = Inf,),)); service=_S1Mock)),
         _s1_caught(() -> ask("s", " " => q; service=_S1Mock)),
         _s1_caught(() -> ask("s", q; service=_S1Mock, confg=_S1_CFG)),       # a misspelt keyword is an option
         _s1_caught(() -> ask("s", "w" => box("the dent"); service=_S1Mock)),
         _s1_caught(() -> ask(_S1_PNG, q; service=_S1Mock)),
         _s1_caught(() -> ask("s", "c" => choice("pick", ["a" => _S1_PNG, "b" => nothing]); service=_S1Mock))]
    end
    @test isempty(seen)
    @test all(e -> e isa ArgumentError, errors)
    @test contains(errors[1].msg, "Int64") && contains(errors[2].msg, "nothing")
    @test contains(errors[4].msg, "JSON") && contains(errors[5].msg, "JSON")
    @test contains(errors[6].msg, "blank")
    @test contains(errors[7].msg, "accepted: none") && contains(errors[7].msg, "confg")
    @test contains(errors[8].msg, "Box") && contains(errors[8].msg, "TypeSafe")
    @test contains(errors[9].msg, "images") && contains(errors[10].msg, "images")
end

@testset "System One — a plain Dict is taken in sorted key order, any other dictionary in its own" begin
    @test collect(keys(choice(Dict("b" => nothing, "a" => nothing, "c" => nothing)).criteria)) == ["a", "b", "c"]
    @test collect(keys(choice(Dict(:zeta => "Z", :alpha => "A")).criteria)) == ["alpha", "zeta"]
    ordered = JSON.Object{String,Any}("zeta" => nothing, "alpha" => nothing)
    @test collect(keys(choice(ordered).criteria)) == ["zeta", "alpha"]
    q = noul("Is it?")
    @test first.(SystemOneRequest("s", Dict("zeta" => q, "alpha" => q, "mid" => q)).questions) == ["alpha", "mid", "zeta"]
    @test first.(SystemOneRequest("s", JSON.Object{String,Any}("zeta" => q, "alpha" => q)).questions) == ["zeta", "alpha"]
    @test collect(keys(RawQuestion(Dict("type" => "rank", "b" => 1, "a" => 2)).fields)) == ["a", "b", "type"]
    @test_throws ArgumentError choice(Dict(:a => "x", "a" => "y"))         # the same name twice
end

@testset "System One — a raw question is sent verbatim and accepts any answer" begin
    raw = RawQuestion(JSON.Object{String,Any}("type" => "rank", "items" => ["b", "a"], "top" => 1))
    @test_throws ArgumentError RawQuestion(Dict("items" => [1]))
    @test_throws ArgumentError RawQuestion(Dict("type" => "  "))
    @test_throws ArgumentError RawQuestion(Dict("type" => :rank))
    @test_throws ArgumentError RawQuestion(Dict(1 => "x", "type" => "rank"))
    @test_throws ArgumentError RawQuestion(Dict(:type => "a", "type" => "b"))
    unchecked = Dict{String,Any}("type" => "choice", "choice" => "zz", "confidence" => 7,
                                 "probabilities" => Dict("zz" => 3.0))
    reply(n, req) = _s1_reply(200, JSON.json(Dict("model" => "jev-1.13.0", "answers" => Dict(
        "rank" => Dict("type" => "rank", "order" => ["a", "b"]), "loose" => unchecked),
        "usage" => Dict("input_tokens" => 1, "output_tokens" => 0))))
    r, seen = _s1_serve(reply) do
        ask("s", "rank" => raw, "loose" => RawQuestion(Dict("type" => "choice", "criteria" => Dict("a" => nothing)));
            service=_S1Mock, model="jev-1.13.0", config=_S1_CFG)
    end
    @test contains(_s1_text(seen[1]), "\"rank\":{\"type\":\"rank\",\"items\":[\"b\",\"a\"],\"top\":1}")
    @test r isa SystemOneSuccess
    @test r["rank"] isa UnknownAnswer && r["rank"].raw["order"] == ["a", "b"]
    @test r["loose"] isa ChoiceAnswer && r["loose"].choice == "zz"     # a raw question's answer is not checked
    # A backend that does not allow raw questions refuses them before sending.
    e = _s1_caught(() -> UniLM._check_request(_BareProtocol(), _BareEndpoint, SystemOneRequest("s", "r" => raw)))
    @test e isa ArgumentError && contains(e.msg, "raw questions are not supported by _BareEndpoint")
end

# ─── Answer checks ───────────────────────────────────────────────────────────

const _S1_REQ = SystemOneRequest("Help! My payouts have been failing for 3 days.",
    ["dept" => choice("Which team?", ["billing", "technical", "sales"]),
     "urg"  => score("How urgent?", ["Can wait", "This week", "Today"]),
     "frus" => noul("Is the customer frustrated?")]; model="jev-latest")

const _S1_GOOD = Dict(
    "dept" => """{"type":"choice","choice":"billing","confidence":1.0,"probabilities":{"billing":1.0,"technical":0.0,"sales":0.0}}""",
    "urg"  => """{"type":"score","score":1.98,"confidence":0.96,"legend":{"0":"Can wait","1":"This week","2":"Today"},"probabilities":{"0":0.0,"1":0.02,"2":0.98}}""",
    "frus" => """{"type":"noul","noul":0.98}""")

_s1_body(over::Dict=Dict(); usage="""{"input_tokens":445,"output_tokens":72}""", drop=String[],
         model="\"model\":\"jev-1.13.0\",") =
    "{" * model * "\"answers\":{" *
    join(["\"$k\":$(get(over, k, _S1_GOOD[k]))" for k in ("dept", "urg", "frus") if !(k in drop)], ",") *
    "},\"usage\":$usage}"

"`ask(_S1_REQ)` over a mock answering `body` once; returns (result, requests sent)."
function _s1_ask_with(body::AbstractString)
    r, seen = _s1_serve((n, req) -> _s1_reply(200, body; id="req_bad")) do
        ask(_S1_REQ; service=_S1Mock, config=_S1_CFG)
    end
    r, length(seen)
end

@testset "System One — a reply whose answers do not fit the request is a call error, never a success" begin
    @test first(_s1_ask_with(_s1_body())) isa SystemOneSuccess
    # Floating-point drift far beyond what TypeSafe has shown is still an answer.
    drift = _s1_body(Dict("dept" => """{"type":"choice","choice":"billing","confidence":0.9,"probabilities":{"billing":0.900000001,"technical":0.1,"sales":0.0}}"""))
    @test first(_s1_ask_with(drift)) isa SystemOneSuccess
    malformed = [
        "a sum of 1.001"           => _s1_body(Dict("dept" => """{"type":"choice","choice":"billing","confidence":0.9,"probabilities":{"billing":0.901,"technical":0.1,"sales":0.0}}""")),
        # every asked question answered once, and nothing else
        "answers empty"            => """{"model":"jev-1.13.0","answers":{},"usage":{"input_tokens":1,"output_tokens":0}}""",
        "a question unanswered"    => _s1_body(; drop=["urg"]),
        "an unasked answer"        => replace(_s1_body(), "},\"usage\"" => ",\"extra\":{\"type\":\"noul\",\"noul\":0.5}},\"usage\""),
        # the question's kind
        "a Noul answering a Choice" => _s1_body(Dict("dept" => """{"type":"noul","noul":0.9}""")),
        "a box answering a Noul"   => _s1_body(Dict("frus" => """{"type":"box","box":{"top":0.1,"bottom":0.4,"left":0.2,"right":0.5},"confidence":null}""")),
        # values in range
        "noul -0.5"                => _s1_body(Dict("frus" => """{"type":"noul","noul":-0.5}""")),
        "noul 1.7"                 => _s1_body(Dict("frus" => """{"type":"noul","noul":1.7}""")),
        "noul 1e400"               => _s1_body(Dict("frus" => """{"type":"noul","noul":1e400}""")),
        "noul NaN"                 => _s1_body(Dict("frus" => """{"type":"noul","noul":NaN}""")),
        "confidence 2.0"           => _s1_body(Dict("dept" => """{"type":"choice","choice":"billing","confidence":2.0,"probabilities":{"billing":1.0,"technical":0.0,"sales":0.0}}""")),
        "confidence null"          => _s1_body(Dict("dept" => """{"type":"choice","choice":"billing","confidence":null,"probabilities":{"billing":1.0,"technical":0.0,"sales":0.0}}""")),
        "a negative probability"   => _s1_body(Dict("dept" => """{"type":"choice","choice":"billing","confidence":1.0,"probabilities":{"billing":1.1,"technical":-0.1,"sales":0.0}}""")),
        "probabilities as booleans" => _s1_body(Dict("dept" => """{"type":"choice","choice":"billing","confidence":true,"probabilities":{"billing":true,"technical":false,"sales":false}}""")),
        # a distribution over exactly the options offered, won by the choice
        "a choice not offered"     => _s1_body(Dict("dept" => """{"type":"choice","choice":"marketing","confidence":1.0,"probabilities":{"billing":1.0,"technical":0.0,"sales":0.0}}""")),
        "an offered option missing" => _s1_body(Dict("dept" => """{"type":"choice","choice":"billing","confidence":0.85,"probabilities":{"billing":0.9,"technical":0.1}}""")),
        "an unoffered option"      => _s1_body(Dict("dept" => """{"type":"choice","choice":"billing","confidence":0.7,"probabilities":{"billing":0.8,"technical":0.1,"sales":0.0,"marketing":0.1}}""")),
        "a sum of 0.9"             => _s1_body(Dict("dept" => """{"type":"choice","choice":"billing","confidence":0.7,"probabilities":{"billing":0.8,"technical":0.1,"sales":0.0}}""")),
        "a choice that is not the argmax" => _s1_body(Dict("dept" => """{"type":"choice","choice":"sales","confidence":0.5,"probabilities":{"billing":0.6,"technical":0.0,"sales":0.4}}""")),
        # score levels exactly 0 … n-1, in canonical decimal
        "level -1"                 => _s1_body(Dict("urg" => """{"type":"score","score":1.0,"confidence":0.5,"legend":{"0":"a","1":"b","2":"c"},"probabilities":{"-1":0.2,"0":0.3,"1":0.5}}""")),
        "levels 01 and 1"          => _s1_body(Dict("urg" => """{"type":"score","score":1.0,"confidence":0.5,"legend":{"0":"a","1":"b","2":"c"},"probabilities":{"0":0.0,"01":0.4,"1":0.6}}""")),
        "level +1"                 => _s1_body(Dict("urg" => """{"type":"score","score":1.0,"confidence":0.5,"legend":{"0":"a","1":"b","2":"c"},"probabilities":{"0":0.0,"+1":1.0,"2":0.0}}""")),
        "level ' 2'"               => _s1_body(Dict("urg" => """{"type":"score","score":1.0,"confidence":0.5,"legend":{"0":"a","1":"b","2":"c"},"probabilities":{"0":0.0,"1":0.0," 2":1.0}}""")),
        "a level beyond the rubric" => _s1_body(Dict("urg" => """{"type":"score","score":1.0,"confidence":0.5,"legend":{"0":"a","1":"b","2":"c"},"probabilities":{"7":1.0}}""")),
        "an overflowing level"     => _s1_body(Dict("urg" => """{"type":"score","score":1.0,"confidence":0.5,"legend":{"0":"a","1":"b","2":"c"},"probabilities":{"99999999999999999999":1.0}}""")),
        "a legend short of a level" => _s1_body(Dict("urg" => """{"type":"score","score":1.0,"confidence":0.5,"legend":{"0":"a","1":"b"},"probabilities":{"0":0.0,"1":1.0,"2":0.0}}""")),
        "a score beyond the rubric" => _s1_body(Dict("urg" => """{"type":"score","score":2.5,"confidence":0.5,"legend":{"0":"a","1":"b","2":"c"},"probabilities":{"0":0.0,"1":0.0,"2":1.0}}""")),
        # the response names its model
        "no model"                 => _s1_body(; model=""),
        "a model that is no string" => _s1_body(; model="\"model\":7,"),
    ]
    for (label, body) in malformed
        @testset "$(label)" begin
            r, sent = _s1_ask_with(body)
            @test r isa SystemOneCallError
            @test sent == 1
            r isa SystemOneCallError && @test r.request_id == "req_bad" && r.cause isa Exception
        end
    end
    # A malformed reply reaches the natural-language verbs as the typed error a failed call is.
    stray = """{"model":"jev-1.13.0","answers":{"classify":{"type":"choice","choice":"a pony","confidence":1.0,"probabilities":{"a pony":1.0}}},"usage":{}}"""
    e, _ = _s1_serve((_, _) -> _s1_reply(200, stray)) do
        _s1_caught(() -> nl_classify("s", (refund = "a refund", other = "anything else"); service=_S1Mock, config=_S1_CFG))
    end
    @test e isa SystemOneError && e.result isa SystemOneCallError && contains(e.result.error, "a pony")
end

@testset "System One — usage that is not a token count reads as 0 with a warning, never failing the answers" begin
    for (usage, tokens, warns) in [("""{"input_tokens":1e300,"output_tokens":0}""", 0, true),
                                   ("""{"input_tokens":-5,"output_tokens":0}""", 0, true),
                                   ("""{"input_tokens":true,"output_tokens":false}""", 0, true),
                                   ("""{"input_tokens":2.5,"output_tokens":0}""", 0, true),
                                   ("""{"input_tokens":null,"output_tokens":0}""", 0, true),
                                   ("\"lots\"", 0, true),
                                   ("""{"input_tokens":445.0,"output_tokens":72}""", 445, false),
                                   ("""{"output_tokens":3}""", 0, false)]
        body = _s1_body(; usage)
        if warns
            r = @test_logs (:warn, r"System One usage") min_level=Logging.Warn first(_s1_ask_with(body))
        else
            r = @test_logs min_level=Logging.Warn first(_s1_ask_with(body))
        end
        @test r isa SystemOneSuccess && token_usage(r).prompt_tokens == tokens
    end
    # Usage left out entirely is the official SDKs' relaxed reading: 0, silently.
    no_usage = """{"model":"jev-1.13.0","answers":{"q":{"type":"noul","noul":0.5}}}"""
    @test (@test_logs min_level=Logging.Warn UniLM._decode_systemone(no_usage, nothing)).usage == TokenUsage()
end

# A backend that may decline single questions.
struct _RefusingProtocol <: UniLM.SystemOneProtocol end
UniLM._allows_refusal(::_RefusingProtocol) = true

@testset "System One — the checks of Box answers and refusals" begin
    req = SystemOneRequest(_S1_PNG, ["where" => box("the dent"), "ok" => noul("Is it fine?")]; model="m")
    resp(a; ok=NoulAnswer(0.5, Dict{String,Any}())) =
        SystemOneResponse("m", Dict{String,SystemOneAnswer}("where" => a, "ok" => ok), TokenUsage(), nothing, Dict{String,Any}())
    ts = UniLM.TypeSafeProtocol()
    check(p, a; kw...) = UniLM._check_answers(p, req, resp(a; kw...))
    @test check(ts, BoxAnswer(BoxCoordinates(0.1, 0.5, 0.2, 0.6), 0.9, Dict{String,Any}())) === nothing
    @test check(ts, BoxAnswer(BoxCoordinates(0.0, 1.0, 0.0, 1.0), nothing, Dict{String,Any}())) === nothing
    @test_throws ArgumentError check(ts, BoxAnswer(BoxCoordinates(-0.1, 0.5, 0.2, 0.6), 0.9, Dict{String,Any}()))
    @test_throws ArgumentError check(ts, BoxAnswer(BoxCoordinates(0.1, 1.5, 0.2, 0.6), 0.9, Dict{String,Any}()))
    @test_throws ArgumentError check(ts, BoxAnswer(BoxCoordinates(0.1, 0.5, NaN, 0.6), 0.9, Dict{String,Any}()))
    @test_throws ArgumentError check(ts, BoxAnswer(BoxCoordinates(0.1, 0.5, 0.2, 0.6), 1.5, Dict{String,Any}()))
    @test_throws ArgumentError check(ts, NoulAnswer(0.5, Dict{String,Any}()))
    @test_throws ArgumentError check(ts, UnknownAnswer("box", Dict{String,Any}()))
    # A refusal: only from a backend that declines questions.
    refusal = RefusalAnswer(Dict{String,Any}("type" => "refusal"))
    @test_throws ArgumentError check(ts, BoxAnswer(BoxCoordinates(0.1, 0.5, 0.2, 0.6), 0.9, Dict{String,Any}()); ok=refusal)
    @test check(_RefusingProtocol(), BoxAnswer(BoxCoordinates(0.1, 0.5, 0.2, 0.6), 0.9, Dict{String,Any}()); ok=refusal) === nothing
    # Shown compactly.
    @test sprint(show, BoxCoordinates(0.29, 0.71, 0.18, 0.56)) == "BoxCoordinates(top=0.29, bottom=0.71, left=0.18, right=0.56)"
    @test sprint(show, BoxAnswer(BoxCoordinates(0.29, 0.71, 0.18, 0.56), nothing, Dict{String,Any}())) ==
          "BoxAnswer(BoxCoordinates(top=0.29, bottom=0.71, left=0.18, right=0.56), confidence=nothing)"
    @test sprint(show, refusal) == "RefusalAnswer(1 fields)"
    # A Box question names its target.
    @test_throws ArgumentError box(nothing)
    @test_throws ArgumentError box("  ")
    @test_throws ArgumentError box(42)
    @test box((what = "the dent", where = "front")) isa BoxQuestion && box(_S1_PNG).instructions === _S1_PNG
end

@testset "System One — the images of a request, depth first in document order" begin
    a, b, c, d = (ImageAttachment(UInt8[0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, i]) for i in 0x01:0x04)
    req = SystemOneRequest(Dict("z" => a, "m" => [b, "text"]),
        ["c" => choice((look = c,), ["x" => d, "y" => nothing]),
         "r" => RawQuestion(JSON.Object{String,Any}("type" => "rank", "items" => (b, a)))]; model="m")
    @test UniLM._system_one_images(req) == [b, a, c, d, b, a]    # "m" sorts before "z" in a plain Dict
    @test isempty(UniLM._system_one_images(SystemOneRequest("s", noul("q?"))))
end

@testset "System One — the response records the backend that answered" begin
    r, _ = _s1_serve(_s1_valid) do
        ask("s", noul("q?"); service=_S1Mock, model="m", config=_S1_CFG)
    end
    @test r.response.service === _S1Mock
    @test SystemOneResponse("m", Dict{String,SystemOneAnswer}(), TokenUsage(), nothing, Dict{String,Any}()).service === nothing
    decoded = UniLM._decode_systemone(read(joinpath(_S1_FIXTURES, "02-basic-three-questions.response.json"), String), nothing)
    @test decoded.service === nothing
    # Provider options key a request apart, in one canonical form.
    @test UniLM._options_key((b = 1, a = "x")) == "a=\"x\"\nb=1"
    @test UniLM._options_key((a = "x", b = 1)) == UniLM._options_key((b = 1, a = "x"))
    @test UniLM._options_key((;)) == ""
    @test_throws ArgumentError UniLM._options_key((a = NaN,))
end
