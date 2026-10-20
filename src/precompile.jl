# ============================================================================
# Precompilation workload.
#
# Runs once, while the package image is built, so that a session's first System
# One call — `ask`, `ask_each`, `nl_dispatch`, `nl_classify` or `@branch`, on
# TypeSafe, OpenAI's Decisions API or Ollama — costs what the
# network costs, not what compiling its path costs: building and encoding the
# request (images and multipart parts included), the exchange and its recording
# seam, decoding the reply or an error body, checking the answers, and planning a
# natural-language call. Nothing leaves the machine: each exchange is replayed
# from a recording written to a temporary directory, with an answer synthesized
# from the request itself, and the directory is removed. No connection pool,
# task, timer or file outlives the workload, and no plan it made ships in the
# image. It calls the public verbs and seam functions only, and stays small:
# everything it compiles is loaded with the package.
# ============================================================================

using PrecompileTools: @compile_workload

# Natural-language functions for the workload: methods must be global to be found.
_workload_route(::nl"the customer wants a refund", msg) = :refund
_workload_route(::nl"anything else", msg) = :other
_workload_keyed(::Val{:refund}, msg) = :refund
_workload_keyed(::Val{:other}, msg) = :other

const _WORKLOAD_INTENT = (refund = "the customer wants a refund", other = "anything else")
const _WORKLOAD_TICKET = "My payouts have been failing for 3 days and nobody has replied to my emails."

# Image headers of 320 × 320 pixels, a PNG (IHDR, an empty IDAT, IEND) and a JPEG (SOI, a
# baseline frame header, EOI): enough for the local checks, which read the size from the
# header and require a JPEG to end with its end-of-image marker. Nothing decodes them, and
# nothing sends them.
const _WORKLOAD_PNG = UInt8[0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a,
                            0x00, 0x00, 0x00, 0x0d, 0x49, 0x48, 0x44, 0x52, 0x00, 0x00, 0x01, 0x40,
                            0x00, 0x00, 0x01, 0x40, 0x08, 0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                            0x00, 0x00, 0x00, 0x00, 0x49, 0x44, 0x41, 0x54, 0x00, 0x00, 0x00, 0x00,
                            0x00, 0x00, 0x00, 0x00, 0x49, 0x45, 0x4e, 0x44, 0x00, 0x00, 0x00, 0x00]
const _WORKLOAD_JPEG = UInt8[0xff, 0xd8, 0xff, 0xc0, 0x00, 0x11, 0x08, 0x01, 0x40, 0x01, 0x40, 0x03,
                             0x01, 0x22, 0x00, 0x02, 0x11, 0x01, 0x03, 0x11, 0x01, 0xff, 0xd9]

# ─── Synthesized replies ─────────────────────────────────────────────────────
# Written with loops over `JSON.Object{String,Any}`, the type the wire is read and written
# in anyway: the workload's own code adds as little as it can to the image.

# The answer a question of the System One JSON form gets: certain of its first option or
# its top level, a likely yes, a box in the middle of the image; a kind this version does
# not know echoes its type.
function _workload_answer(q::AbstractDict)::JSON.Object{String,Any}
    kind = q["type"]::String
    a = JSON.Object{String,Any}()
    a["type"] = kind
    if kind == "choice"
        probabilities = JSON.Object{String,Any}()
        for name in keys(q["criteria"]::AbstractDict)
            isempty(probabilities) && (a["choice"] = name)
            probabilities[name] = isempty(probabilities) ? 1.0 : 0.0
        end
        a["confidence"] = 1.0
        a["probabilities"] = probabilities
    elseif kind == "score"
        levels = q["criteria"]::AbstractVector
        legend, probabilities = JSON.Object{String,Any}(), JSON.Object{String,Any}()
        for i in eachindex(levels)
            legend[string(i - 1)] = levels[i]
            probabilities[string(i - 1)] = i == lastindex(levels) ? 1.0 : 0.0
        end
        a["score"] = length(levels) - 1.0
        a["confidence"] = 1.0
        a["legend"] = legend
        a["probabilities"] = probabilities
    elseif kind == "noul"
        a["noul"] = 0.9
    elseif kind == "box"
        box = JSON.Object{String,Any}()
        box["top"] = 0.25; box["bottom"] = 0.75; box["left"] = 0.25; box["right"] = 0.75
        a["box"] = box
        a["confidence"] = 0.8
    end
    a
end

# The same on the Decisions wire: an array of answers in question order, each naming its
# question, a choice's and a score's probabilities as arrays.
function _workload_decision(q::AbstractDict)::JSON.Object{String,Any}
    kind = q["type"]::String
    a = JSON.Object{String,Any}()
    a["type"] = kind
    a["name"] = q["name"]
    if kind == "predicate"
        a["probability"] = 0.9
        return a
    end
    outcomes = (kind == "choice" ? q["choices"] : q["levels"])::AbstractVector
    probabilities = Any[]
    for i in eachindex(outcomes)
        p = JSON.Object{String,Any}()
        if kind == "choice"
            p["value"] = outcomes[i]["value"]
            i == firstindex(outcomes) && (a["choice"] = p["value"])
        else
            p["value"] = i - 1
            p["label"] = outcomes[i]["label"]
        end
        p["probability"] = (kind == "choice" ? i == firstindex(outcomes) : i == lastindex(outcomes)) ? 1.0 : 0.0
        push!(probabilities, p)
    end
    kind == "score" && (a["score"] = length(outcomes) - 1.0)
    a["confidence"] = 1.0
    a["probabilities"] = probabilities
    a
end

# Which wire a keyed path belongs to: a recording is filed under its backend's namespace
# (`/openai-decisions/…`, `/ollama/…`; TypeSafe's under the plain path).
_workload_decisions(path::AbstractString)::Bool = endswith(path, _DECISIONS_PATH)
_workload_ollama(path::AbstractString)::Bool = startswith(path, "/ollama/")

# The reply to the request `body` sent to the keyed `path`: a models listing, a Decisions
# reply, or a reply in the System One JSON form (Ollama's with its cache count).
function _workload_reply(path::String, body::String)::JSON.Object{String,Any}
    reply = JSON.Object{String,Any}()
    if path == TYPESAFE_MODELS_PATH
        card = JSON.Object{String,Any}()
        card["name"] = TYPESAFE_DEFAULT_MODEL; card["description"] = ""; card["release_date"] = ""
        reply["models"] = Any[card]
        return reply
    end
    request = JSON.parse(body)::AbstractDict
    reply["model"] = request["model"]
    if _workload_decisions(path)
        answers = Any[]
        for q in request["questions"]::AbstractVector
            push!(answers, _workload_decision(q::AbstractDict))
        end
    else
        answers = JSON.Object{String,Any}()
        for (name, q) in request["questions"]::AbstractDict
            answers[String(name)] = _workload_answer(q::AbstractDict)
        end
        _workload_ollama(path) && (reply["prompt_eval_cached_count"] = 0)
    end
    reply["answers"] = answers
    usage = JSON.Object{String,Any}()
    usage["input_tokens"] = 1; usage["output_tokens"] = 0
    reply["usage"] = usage
    reply
end

# The header a backend sends its request id in, by keyed path; Ollama sends none.
_workload_id_header(path::AbstractString)::Union{Nothing,String} =
    _workload_ollama(path) ? nothing : _workload_decisions(path) ? "x-request-id" : "x-typesafe-request-id"

_workload_headers(path::AbstractString)::Vector{Pair{String,String}} =
    (h = _workload_id_header(path); isnothing(h) ? Pair{String,String}[] : [h => "req_precompile"])

# ─── Requests ────────────────────────────────────────────────────────────────

# Questions in their common spellings, each a kind every backend here carries; a Noul with
# `yes`/`no` criteria where the backend reads them (OpenAI's predicates have none).
_workload_questions(p::SystemOneProtocol) = (
    "department" => choice("Which team should handle this ticket?",
                           (billing = "Payments, invoicing, payouts, refunds", technical = "Bugs, outages, integrations")),
    "urgency" => score("How urgent is this ticket?", ["Can wait", "Needs attention this week", "Needs attention today"]),
    "is_frustrated" => (p isa OpenAIDecisionsProtocol ? noul("Is the customer frustrated?") :
                        noul("Is the customer frustrated?"; yes = "Impatient or upset", no = "Neutral or satisfied")))

# What only TypeSafe carries: images — the state and an option's description, sent as the
# parts of a multipart body — and a Box question.
_workload_images() = (ImageAttachment(_WORKLOAD_PNG),
                      ("dent" => box("the dent on the bumper"),
                       "same" => choice("Is it this one?", ["yes" => ImageAttachment(_WORKLOAD_JPEG), "no" => nothing])))

# The backends of the workload, each with a model to name: the default where there is one.
_workload_backends() = ((TYPESAFEServiceEndpoint, nothing), (OPENAIServiceEndpoint, nothing),
                        (OllamaEndpoint(; base_url = "http://127.0.0.1:11434"), "tev1:0.8b"))

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

# ─── Codecs ──────────────────────────────────────────────────────────────────

# Every backend's error bodies, read by its decoder: a replay only ever returns a 200.
function _workload_failures()::Nothing
    for (service, model) in _workload_backends()
        p = system_one_protocol(service)::SystemOneProtocol
        request = SystemOneRequest(_WORKLOAD_TICKET, collect(Pair{String,SystemOneQuestion}, _workload_questions(p)); model)
        for (status, headers, body) in _workload_errors(p)
            decode_system_one(p, service, request, HTTP.Response(status, headers, Vector{UInt8}(body)))::SystemOneFailure
        end
    end
    nothing
end

# Every backend's encoder and decoder called directly, for a build that cannot replay: a
# request is encoded — TypeSafe's encoder builds the multipart form of one with images, to
# measure it — and its synthesized reply decoded and checked.
function _workload_codecs()::Nothing
    for (service, model) in _workload_backends()
        p = system_one_protocol(service)::SystemOneProtocol
        requests = [SystemOneRequest(_WORKLOAD_TICKET, collect(Pair{String,SystemOneQuestion}, _workload_questions(p)); model)]
        if p isa TypeSafeProtocol
            image, questions = _workload_images()
            push!(requests, SystemOneRequest(image, collect(Pair{String,SystemOneQuestion}, questions)))
        end
        for request in requests
            wire = encode_system_one(p, service, request)::SystemOneWire
            path = p isa OllamaSystemOneProtocol ? "/ollama" * wire.path : wire.path
            reply = HTTP.Response(200, _workload_headers(path), Vector{UInt8}(JSON.json(_workload_reply(path, wire.body))))
            decode_system_one(p, service, request, reply)::SystemOneSuccess
        end
    end
    nothing
end

# ─── Verbs ───────────────────────────────────────────────────────────────────

# The recording a replay of the missed request needs, written where its scope looks.
function _workload_record!(e::ReplayMissError)::Nothing
    header = _workload_id_header(e.path)
    response = JSON.Object{String,Any}()
    response["status"] = 200
    response["request_id"] = isnothing(header) ? nothing : "req_precompile"
    isnothing(header) || (response["request_id_header"] = header)
    # The keyed body is the request body, then a line of what it is keyed on besides; read as
    # a `String`, the type every body is parsed as elsewhere.
    response["body"] = _workload_reply(e.path, String(first(split(e.body, '\n'; limit = 2))))
    recording = JSON.Object{String,Any}()
    recording["response"] = response
    write(joinpath(e.dir, e.key * ".json"), JSON.json(recording))
    nothing
end

# Every replayed call, as one callable type holding its function abstractly: the replay
# scope is compiled once for the workload, not once for each of its calls.
struct _WorkloadCall <: Function
    f::Function
end
(c::_WorkloadCall)() = c.f()

# `f()` replayed from `dir`, each request it misses recorded first: a batch of `n`
# requests misses at most `n` times before it replays whole.
function _workload_replay(@nospecialize(f::Function), dir::String)
    call = _WorkloadCall(f)
    for _ in 1:8
        try
            return with_recorded_answers(call, dir)
        catch e
            e isa ReplayMissError || rethrow()
            _workload_record!(e)
        end
    end
    error("a call of the precompilation workload kept missing its recordings")
end

# Every verb on one backend, the backend of a `with_system_one` scope: a request with a
# question of each kind, a batch, the literal and keyed dispatch, a classification and a
# branch.
function _workload_verbs(dir::String, service, model::Union{Nothing,String})::Nothing
    with_system_one(service; model) do
        q = _workload_questions(system_one_protocol(service)::SystemOneProtocol)
        _workload_replay(() -> ask(_WORKLOAD_TICKET, q...), dir)
        _workload_replay(() -> ask_each([_WORKLOAD_TICKET, "The app crashes when I log in."], first(q)), dir)
        _workload_replay(() -> nl_dispatch(_workload_route, _WORKLOAD_TICKET), dir)
        _workload_replay(() -> nl_dispatch(_workload_keyed, _WORKLOAD_TICKET; texts = _WORKLOAD_INTENT), dir)
        _workload_replay(() -> nl_classify(_WORKLOAD_TICKET, _WORKLOAD_INTENT), dir)
        _workload_replay(() -> (@branch _WORKLOAD_TICKET min_confidence = 0.5 begin
                                    "the customer wants a refund" => 1
                                    ("anything else", "Any other request") => 2
                                    _ => 0
                                end), dir)
    end
    nothing
end

# What only TypeSafe answers, outside any scope as most calls are: a request with images and
# a Box question. Its models listing is left out: compiled here it pushed package load past
# its budget (+25.6% over 0.23.0, bound +25%), and a listing is no call a hot path makes
# first — it compiles on first use instead (about 0.5 s).
function _workload_typesafe(dir::String)::Nothing
    image, questions = _workload_images()
    _workload_replay(() -> ask(image, questions...), dir)
    nothing
end

# A directory for the recordings, or `nothing` when none can be made: a missing or
# read-only temporary directory must not fail the package's build, which then compiles
# the encoders and decoders without the exchange.
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

# The workload: every verb replayed on every backend in a directory made in `parent` and
# removed afterwards — or, with no directory, every backend's codecs alone — then every
# backend's error bodies. The plan caches are emptied last: a plan belongs to the world it
# was made in, and the tables are the workload's own.
function _precompile_workload(parent::AbstractString = tempdir())::Nothing
    try
        dir = _workload_dir(parent)
        if isnothing(dir)
            _workload_codecs()
        else
            try
                for (service, model) in _workload_backends()
                    _workload_verbs(dir, service, model)
                end
                _workload_typesafe(dir)
            finally
                rm(dir; recursive = true, force = true)
            end
        end
        _workload_failures()
    finally
        _workload_forget!(_NL_PLANS)
        _workload_forget!(_NL_KEYED_PLANS)
        _workload_forget!(_NL_TABLES)
    end
    nothing
end

@compile_workload _precompile_workload()
