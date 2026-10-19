# ============================================================================
# Precompilation workload.
#
# Runs once, while the package image is built, so that a session's first System
# One call — `ask`, `ask_each`, `list_models`, `nl_dispatch`, `nl_classify` or
# `@branch`, on TypeSafe, OpenAI's Decisions API or Ollama — costs what the
# network costs, not what compiling its path costs: building and encoding the
# request (images and multipart parts included), the exchange and its recording
# seam, decoding the reply or an error body, checking the answers, and planning a
# natural-language call. Nothing leaves the machine: each exchange is replayed
# from a recording written to a temporary directory, with an answer synthesized
# from the request itself, and the directory is removed. No connection pool,
# task, timer or file outlives the workload, and no plan it made ships in the
# image.
# ============================================================================

using PrecompileTools: @compile_workload

# Natural-language functions for the workload: methods must be global to be found.
_workload_route(::nl"the customer wants a refund", msg) = :refund
_workload_route(::nl"anything else", msg) = :other
_workload_keyed(::Val{:refund}, msg) = :refund
_workload_keyed(::Val{:other}, msg) = :other

const _WORKLOAD_INTENT = (refund = "the customer wants a refund", other = "anything else")
const _WORKLOAD_TICKET = "My payouts have been failing for 3 days and nobody has replied to my emails."

# Image headers of 320 × 320 pixels, a PNG (IHDR, an empty IDAT, IEND) and a JPEG (SOI,
# then a baseline frame header): enough for the local checks, which read the size from the
# header. Nothing decodes them, and nothing sends them.
const _WORKLOAD_PNG = UInt8[0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a,
                            0x00, 0x00, 0x00, 0x0d, 0x49, 0x48, 0x44, 0x52, 0x00, 0x00, 0x01, 0x40,
                            0x00, 0x00, 0x01, 0x40, 0x08, 0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                            0x00, 0x00, 0x00, 0x00, 0x49, 0x44, 0x41, 0x54, 0x00, 0x00, 0x00, 0x00,
                            0x00, 0x00, 0x00, 0x00, 0x49, 0x45, 0x4e, 0x44, 0x00, 0x00, 0x00, 0x00]
const _WORKLOAD_JPEG = UInt8[0xff, 0xd8, 0xff, 0xc0, 0x00, 0x11, 0x08, 0x01, 0x40, 0x01, 0x40, 0x03,
                             0x01, 0x22, 0x00, 0x02, 0x11, 0x01, 0x03, 0x11, 0x01]

# ─── Synthesized replies ─────────────────────────────────────────────────────

# The answer a question of the System One JSON form gets: certain of its first option or
# its top level, a likely yes, a box in the middle of the image; a kind this version does
# not know echoes its type.
function _workload_answer(q::AbstractDict)::JSON.Object{String,Any}
    kind = q["type"]::String
    a = JSON.Object{String,Any}("type" => kind)
    if kind == "choice"
        names = collect(String, keys(q["criteria"]::AbstractDict))
        a["choice"] = first(names)
        a["confidence"] = 1.0
        a["probabilities"] = JSON.Object{String,Any}(n => (i == 1 ? 1.0 : 0.0) for (i, n) in enumerate(names))
    elseif kind == "score"
        levels = q["criteria"]::AbstractVector
        n = length(levels)
        a["score"] = n - 1.0
        a["confidence"] = 1.0
        a["legend"] = JSON.Object{String,Any}(string(i - 1) => levels[i] for i in 1:n)
        a["probabilities"] = JSON.Object{String,Any}(string(i - 1) => (i == n ? 1.0 : 0.0) for i in 1:n)
    elseif kind == "noul"
        a["noul"] = 0.9
    elseif kind == "box"
        a["box"] = JSON.Object{String,Any}("top" => 0.25, "bottom" => 0.75, "left" => 0.25, "right" => 0.75)
        a["confidence"] = 0.8
    end
    a
end

# The same on the Decisions wire: an array of answers in question order, each naming its
# question, a choice's and a score's probabilities as arrays.
function _workload_decision(q::AbstractDict)::JSON.Object{String,Any}
    kind = q["type"]::String
    a = JSON.Object{String,Any}("type" => kind, "name" => q["name"])
    if kind == "choice"
        values = Any[c["value"] for c in q["choices"]::AbstractVector]
        a["choice"] = first(values)
        a["confidence"] = 1.0
        a["probabilities"] = Any[JSON.Object{String,Any}("value" => v, "probability" => i == 1 ? 1.0 : 0.0)
                                 for (i, v) in enumerate(values)]
    elseif kind == "score"
        levels = q["levels"]::AbstractVector
        n = length(levels)
        a["score"] = n - 1.0
        a["confidence"] = 1.0
        a["probabilities"] = Any[JSON.Object{String,Any}("value" => i - 1, "label" => levels[i]["label"],
                                                         "probability" => i == n ? 1.0 : 0.0) for i in 1:n]
    else
        a["probability"] = 0.9
    end
    a
end

# The path a backend's exchange is keyed on: its namespace, then the wire's path. (Asserted:
# `p` is abstract where the workload loops over backends.)
_workload_path(p::SystemOneProtocol, path::String)::String =
    (ns = _system_one_namespace(p)::String; isempty(ns) ? path : "/" * ns * path)

# The reply to the request `body` sent to the keyed `path`: a models listing, a Decisions
# reply, or a reply in the System One JSON form (Ollama's with its cache count).
function _workload_reply(path::String, body::AbstractString)::JSON.Object{String,Any}
    path == TYPESAFE_MODELS_PATH && return JSON.Object{String,Any}("models" => Any[JSON.Object{String,Any}(
        "name" => TYPESAFE_DEFAULT_MODEL, "description" => "", "release_date" => "")])
    request = JSON.parse(body)::AbstractDict
    usage = JSON.Object{String,Any}("input_tokens" => 1, "output_tokens" => 0)
    reply = JSON.Object{String,Any}("model" => request["model"])
    if path == _workload_path(OpenAIDecisionsProtocol(), _DECISIONS_PATH)
        reply["answers"] = Any[_workload_decision(q) for q in request["questions"]::AbstractVector]
    else
        reply["answers"] = JSON.Object{String,Any}(String(k) => _workload_answer(q)
                                                    for (k, q) in request["questions"]::AbstractDict)
        path == _workload_path(OllamaSystemOneProtocol(), SYSTEMONE_PATH) && (reply["prompt_eval_cached_count"] = 0)
    end
    reply["usage"] = usage
    reply
end

# The header a backend sends its request id in, by keyed path; Ollama sends none.
_workload_id_header(path::String)::Union{Nothing,String} =
    path == _workload_path(OllamaSystemOneProtocol(), SYSTEMONE_PATH) ? nothing :
    path == _workload_path(OpenAIDecisionsProtocol(), _DECISIONS_PATH) ? "x-request-id" : "x-typesafe-request-id"

_workload_headers(path::String)::Vector{Pair{String,String}} =
    (h = _workload_id_header(path); isnothing(h) ? Pair{String,String}[] : [h => "req_precompile"])

# ─── Requests ────────────────────────────────────────────────────────────────

# Questions in their common spellings, each a kind every backend here carries.
_workload_questions() = (
    "department" => choice("Which team should handle this ticket?",
                           (billing = "Payments, invoicing, payouts, refunds", technical = "Bugs, outages, integrations")),
    "urgency" => score("How urgent is this ticket?", ["Can wait", "Needs attention this week", "Needs attention today"]),
    "is_frustrated" => noul("Is the customer frustrated?"))

# The requests a backend's codec is exercised on: the three common kinds about a string,
# about JSON and with the backend's own option, and what only it carries — images, a Box
# and a raw question on TypeSafe; an image state on the Decisions API.
function _workload_requests(p::SystemOneProtocol)::Vector{SystemOneRequest}
    qs = collect(Pair{String,SystemOneQuestion}, _workload_questions())
    picks = ["billing" => "Payments and refunds", "other" => nothing]
    out = [SystemOneRequest(_WORKLOAD_TICKET, qs),
           SystemOneRequest(Dict{String,Any}("ticket" => _WORKLOAD_TICKET, "plan" => "pro"),
                            ["pick" => choice("Pick one", picks), "level" => score("How bad?", ["low", "high"])])]
    if p isa TypeSafeProtocol
        png, jpeg = ImageAttachment(_WORKLOAD_PNG), ImageAttachment(_WORKLOAD_JPEG)
        mixed = ["upset" => noul(; yes = "Upset", no = "Calm"), "named" => choice(["billing", "other"]),
                 "rank" => RawQuestion(Dict("type" => "rank", "items" => ["a", "b"]))]
        push!(out, SystemOneRequest(Dict("text" => _WORKLOAD_TICKET), mixed; options = (long_context = true,)),
              SystemOneRequest(png, ["dent" => box("the dent on the bumper"),
                                     "same" => choice("Is it this one?", ["yes" => jpeg, "no" => nothing]),
                                     "shade" => score("How dark?", ["light", png])]))
    elseif p isa OpenAIDecisionsProtocol
        push!(out, SystemOneRequest(Any["What is shown?", ImageAttachment(_WORKLOAD_PNG)],
                                    ["damaged" => noul("Is anything damaged?")]; options = (prompt_cache_key = "precompile",)))
    else
        push!(out, SystemOneRequest(_WORKLOAD_TICKET, qs; options = (keep_alive = 0,)))
    end
    out
end

# Error bodies in each backend's shapes, with the headers that carry a request id or a
# wait; two TypeSafe failures get a hint.
function _workload_errors(p::SystemOneProtocol)::Vector{Tuple{Int,Vector{Pair{String,String}},String}}
    retry = ["retry-after" => "1", "retry-after-ms" => "500"]
    if p isa TypeSafeProtocol
        id = ["x-typesafe-request-id" => "req_precompile"]
        [(401, id, """{"detail":{"error_type":"authentication_error","message":"Cannot authenticate with the server."}}"""),
         (422, id, """{"detail":[{"type":"missing","loc":["body","questions","q","criteria"],"msg":"Field required"}]}"""),
         (404, id, """{"detail":"Not Found"}"""),
         (400, id, """{"detail":{"error_type":"max_tokens_exceeded"}}"""),
         (400, id, """{"detail":"Invalid request."}"""),
         (529, [id; retry], """{"error":"Overloaded"}"""),
         (502, Pair{String,String}[], "upstream connect error")]
    elseif p isa OpenAIDecisionsProtocol
        id = ["x-request-id" => "req_precompile"]
        [(404, id, """{"error":{"message":"The model does not exist.","type":"invalid_request_error","param":null,"code":"model_not_found"}}"""),
         (429, [id; retry], """{"error":{"message":"Rate limit reached.","type":"requests","param":null,"code":null}}"""),
         (500, Pair{String,String}[], "upstream connect error")]
    else
        [(404, Pair{String,String}[], """{"error":"model 'tev1:0.8b' not found"}"""),
         (400, Pair{String,String}[], """{"error":"too many options"}"""),
         (503, retry, "server busy")]
    end
end

# The backends of the workload, each with a model to name: the default where there is one.
_workload_backends() = ((TYPESAFEServiceEndpoint, nothing), (OPENAIServiceEndpoint, nothing),
                        (OllamaEndpoint(; base_url = "http://127.0.0.1:11434"), "tev1:0.8b"))

# ─── Codecs ──────────────────────────────────────────────────────────────────

# Every backend's encoder and decoder, called directly: no file, no exchange. A request is
# encoded, its synthesized reply decoded and checked, its error bodies read; the multipart
# form of a request with images is built, as each attempt that sends one builds it.
function _workload_codecs()::Nothing
    for (service, model) in _workload_backends()
        p = system_one_protocol(service)::SystemOneProtocol
        requests = SystemOneRequest[SystemOneRequest(r.state, r.questions, model, r.options) for r in _workload_requests(p)]
        for request in requests
            wire = encode_system_one(p, service, request)::SystemOneWire
            isempty(wire.parts) || _system_one_form(wire)
            path = _workload_path(p, wire.path)
            reply = HTTP.Response(200, _workload_headers(path),
                                  Vector{UInt8}(JSON.json(_workload_reply(path, wire.body))))
            decode_system_one(p, service, request, reply)::SystemOneSuccess
        end
        for (status, headers, body) in _workload_errors(p)
            decode_system_one(p, service, last(requests), HTTP.Response(status, headers, Vector{UInt8}(body)))::SystemOneFailure
        end
    end
    nothing
end

# ─── Verbs ───────────────────────────────────────────────────────────────────

# The recording a replay of the missed request needs, written where its scope looks.
function _workload_record!(e::ReplayMissError)::Nothing
    header = _workload_id_header(e.path)
    response = JSON.Object{String,Any}("status" => 200, "request_id" => isnothing(header) ? nothing : "req_precompile")
    isnothing(header) || (response["request_id_header"] = header)
    # The keyed body is the request body, then a line of what it is keyed on besides.
    response["body"] = _workload_reply(e.path, first(split(e.body, '\n'; limit = 2)))
    write(joinpath(e.dir, e.key * ".json"), JSON.json(JSON.Object{String,Any}("response" => response)))
    nothing
end

# `f()` replayed from `dir`, each request it misses recorded first: a batch of `n`
# requests misses at most `n` times before it replays whole.
function _workload_replay(f::Function, dir::String)
    for _ in 1:8
        try
            return with_recorded_answers(f, dir)
        catch e
            e isa ReplayMissError || rethrow()
            _workload_record!(e)
        end
    end
    error("the precompilation workload's replay of $(f) did not converge")
end

# The verbs every backend answers, with the keywords `kw` — none inside a scope, the
# backend (and model) where a call names its own: a request built two ways, a batch, the
# literal and keyed dispatch, and a classification.
function _workload_verbs(dir::String, kw::NamedTuple)::Nothing
    q = _workload_questions()
    _workload_replay(() -> ask(_WORKLOAD_TICKET, q...; kw...), dir)
    _workload_replay(() -> ask(_WORKLOAD_TICKET, last(q); kw...), dir)
    _workload_replay(() -> ask_each([_WORKLOAD_TICKET, "The app crashes when I log in."], first(q); kw...), dir)
    _workload_replay(() -> nl_dispatch(_workload_route, _WORKLOAD_TICKET; kw...), dir)
    _workload_replay(() -> nl_dispatch(_workload_keyed, _WORKLOAD_TICKET; texts = _WORKLOAD_INTENT, kw...), dir)
    _workload_replay(() -> nl_classify(_WORKLOAD_TICKET, _WORKLOAD_INTENT; kw...), dir)
    nothing
end

# One backend through a `with_system_one` scope — the verbs, a request built as a value, a
# vector table and two branches — then through the keywords of each call.
function _workload_backend(dir::String, service, model::Union{Nothing,String})::Nothing
    with_system_one(service; model) do
        _workload_verbs(dir, (;))
        _workload_replay(() -> ask(SystemOneRequest(Dict{String,Any}("ticket" => _WORKLOAD_TICKET),
                                                    ["pick" => choice("Pick one", ["a", "b"])])), dir)
        _workload_replay(() -> nl_dispatch(_workload_keyed, _WORKLOAD_TICKET; texts = collect(pairs(_WORKLOAD_INTENT))), dir)
        _workload_replay(() -> (@branch _WORKLOAD_TICKET begin
                                    "the customer wants a refund" => 1
                                    "anything else" => 2
                                end), dir)
        _workload_replay(() -> (@branch _WORKLOAD_TICKET min_confidence = 0.5 begin
                                    "the customer wants a refund" => 1
                                    ("anything else", "Any other request") => 2
                                    _ => 0
                                end), dir)
    end
    _workload_verbs(dir, isnothing(model) ? (; service) : (; service, model))
end

# What only TypeSafe answers, outside any scope as most calls are: its models listing, a
# request with images and a Box question, its one option, and a pinned model.
function _workload_typesafe(dir::String)::Nothing
    _workload_replay(() -> list_models(), dir)
    _workload_replay(() -> ask(ImageAttachment(_WORKLOAD_PNG), "dent" => box("the dent on the bumper")), dir)
    _workload_replay(() -> ask(_WORKLOAD_TICKET, _workload_questions()...; long_context = true), dir)
    _workload_replay(() -> ask(_WORKLOAD_TICKET, _workload_questions()...; model = "jev-1.13.0"), dir)
    nothing
end

# A directory for the recordings, or `nothing` when none can be made: a missing or
# read-only temporary directory must not fail the package's build, which then goes
# without the replayed verbs — they compile at their first call instead.
function _workload_dir(parent::AbstractString)::Union{Nothing,String}
    try
        mktempdir(parent)
    catch e
        e isa InterruptException && rethrow()
        @debug "no temporary directory for the precompilation workload; the replayed verbs are skipped" exception = e
        nothing
    end
end

_workload_forget!(cache::_NLCache{K,V}) where {K,V} = (@atomic cache.entries = Dict{K,V}(); nothing)

# The workload: every backend's codecs, then — in a directory made in `parent` and removed
# afterwards — every verb replayed on every backend. The plan caches are emptied last: a
# plan belongs to the world it was made in, and the tables are the workload's own.
function _precompile_workload(parent::AbstractString = tempdir())::Nothing
    try
        _workload_codecs()
        dir = _workload_dir(parent)
        isnothing(dir) || try
            for (service, model) in _workload_backends()
                _workload_backend(dir, service, model)
            end
            _workload_typesafe(dir)
        finally
            rm(dir; recursive = true, force = true)
        end
    finally
        _workload_forget!(_NL_PLANS)
        _workload_forget!(_NL_KEYED_PLANS)
        _workload_forget!(_NL_TABLES)
    end
    nothing
end

@compile_workload _precompile_workload()
