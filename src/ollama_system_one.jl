# ============================================================================
# Ollama's System One route (https://docs.ollama.com/api/systemone): POST
# {host}/v1/systemone, served by local decision models (Ollama 0.35 and later) in
# TypeSafe's wire format. The body is the System One JSON form TypeSafe's wrapper
# writes and the reply is read by TypeSafe's decoder; what lives here is what
# differs — Ollama's stricter rules for a question, the top-level `images` (the
# state's images, raw base64) and `runner`, its one option `keep_alive`, its error
# bodies, and `confidence`, which Ollama's generic decision encodings report as
# 1 − H(p)/ln n and this wrapper recomputes from the probabilities with TypeSafe's
# published formulas, so a threshold means the same on every backend.
# ============================================================================

"""
    OllamaSystemOneProtocol <: SystemOneProtocol

Ollama's System One wire, the protocol [`OllamaEndpoint`](@ref) speaks:
`POST {base_url}/v1/systemone` (Ollama 0.35 and later) with the System One JSON
form TypeSafe's wrapper writes — `state`, `model`, `questions`, then `runner`
when the endpoint sets one (part of a recording's key: it can change an answer),
`images` when the state holds any, and `keep_alive` when one is set — answered
by a local decision model such as `tev1:0.8b` and decoded as a TypeSafe reply
is. No key is sent, no request id comes back, and recordings are filed under the
namespace `"ollama"`, keyed without `keep_alive`, which changes how long the
model stays loaded but never an answer.

There is no default model: the call names one (`model = "tev1:0.8b"`), or a
[`with_system_one`](@ref) scope does. Before any I/O it refuses what Ollama's
System One route rejects for every model and runner (Ollama 0.40.6) — more than
64 questions; `instructions` given as anything but a string, an object or an
array; a Choice of fewer than 2 or more than 255 options, or an option described
by anything but a string or `nothing`; a Score of fewer than 2 or more than 10
levels, or a level that is not a string; a Noul criterion that is not a string;
a blank text `state` without images; a Box question, which Ollama does not
answer — and, a refusal of UniLM's own, blank (whitespace-only) `instructions`,
which Ollama would read as none. The route accepts a question without
`instructions`, but most models do not (measured on Ollama 0.40.6): `tev1:4b`,
`nimble:9b` and `strands-decider:2b` answer HTTP 400 `instructions must be a
nonempty string, object, or array`, `kev:0.8b-q8_0` answers 400
`questions.intent: "instructions" must be provided` (for a question named
`intent`), and `clef-flash:9b` answers the question — so pass `instructions` with
local models. A model that takes fewer options (Tev1: at most 24; Nimble and Clef
Flash: 26) says so in its own HTTP 400 too, which comes back as a
[`SystemOneFailure`](@ref). Ollama also caps a local request's text and schema,
images excluded, at 64 KiB: that limit is the server's to count, by its own
measure, and is not checked here; a request over it comes back as a
[`SystemOneFailure`](@ref) carrying Ollama's 413, `text and schema must not exceed
64 KiB`. A [`RawQuestion`](@ref) is sent verbatim, for the server to judge. Its
one option is `keep_alive`: the seconds the model stays loaded after the request
(`0` unloads it at once, `Inf` keeps it loaded), by default the endpoint's own.

Images go in the state. Ollama takes a request's images as one list, `images`
(raw base64), shared by all its questions and shown to the model before the
text, so the state is an image, or a vector or tuple of images then texts
(`[image, "What is in it?"]`). With images, the state's other entries must be
texts; they become `state` — none: `""`, one: that string, several: an array —
and a blank one is allowed, as Ollama 0.40.6 answers a blank `state` sent with
images. An image after a text, inside a JSON object or array, or in a question
could only be moved, never sent where it was put, and is refused before any
I/O, as is a state of images and anything but texts. A model without vision
answers its own 400, in its own words (Ollama 0.40.6: `tev1:0.8b` `image inputs
are not supported by this decision model`, `kev:0.8b-q8_0`
`registry.ollama.ai/library/kev:0.8b-q8_0 does not support vision`), which comes
back as a [`SystemOneFailure`](@ref).
Recordings key an image by its digest, `data:<mime>;sha256,<hex>` (the SHA-256
of its base64 text), never its bytes.

`confidence` is TypeSafe's published measure, computed here from the returned
probabilities: `(n·p_max − 1)/(n − 1)` clamped to `0 … 1` for a Choice, and
`1 − Σᵢ pᵢ·|i − m| / MAD_unif` floored at `0` for a Score, `m` its most likely
level ([Confidence](https://docs.typesafe.ai/confidence)). Measured on Ollama
0.40.6, the five MLX models (`tev1:0.8b`, `tev1:4b`, `nimble:9b`,
`strands-decider:2b`, `clef-flash:9b`) report `1 − H(p)/ln n`, and Kev and the
models on the llama.cpp path (`kev`, `d1`, `d1-omni`) TypeSafe's formulas; the
server's value stays in each answer's `raw`.
Usage counts `prompt_eval_cached_count` as cached tokens where the server
reports it (the MLX path); on the llama.cpp path the count is not reported and
reads as 0. A cloud model whose reply drops its tag (Ollama's cloud answers
`clef` for `clef:cloud`) is reported under the requested name, so it is priced
as the cloud model; the server's `model` stays in `raw`.

Public extension API (not exported).
"""
struct OllamaSystemOneProtocol <: SystemOneProtocol end

system_one_protocol(::OllamaEndpoint) = OllamaSystemOneProtocol()

_system_one_default_model(::OllamaSystemOneProtocol, service)::String = throw(ArgumentError(
    "$(_service_name(service)) has no default System One model: name an installed decision model, e.g. " *
    "`model = \"tev1:0.8b\"`, or set one for a scope with `with_system_one(OllamaEndpoint(); model = …) do … end`"))
_system_one_namespace(::OllamaSystemOneProtocol)::String = "ollama"
_allows_refusal(::OllamaSystemOneProtocol)::Bool = false
_allows_raw(::OllamaSystemOneProtocol)::Bool = true
# A local Ollama server's decision runners answer one request at a time, so a queue of
# local requests only adds timeouts; its config stays the field defaults (the fallback's
# `nothing`), since loading a model on a CPU can take minutes.
_system_one_concurrency(::OllamaSystemOneProtocol, service)::Int = 1
# Ollama's generic decision encodings normalise a softmax in Float64, as llama.cpp's
# server does for GGUF decision models: the replies of tev1 and kev captured live sum to 1
# within 4.4e-16. Kev models on Ollama's own runner round every probability to four
# decimals (decision/kev.go, `roundKev`), so n of them may sum up to n·0.5e-4 away from 1;
# that bound is admitted, with the Float64 margin on top — and one four-decimal step,
# 1e-4, between the chosen option and the most probable.
_probability_tolerance(::OllamaSystemOneProtocol, n::Int)::Float64 = n * 0.5e-4 + _FLOAT_PROBABILITY_TOLERANCE
_probability_slack(::OllamaSystemOneProtocol)::Float64 = 1.0e-4 + _FLOAT_PROBABILITY_TOLERANCE

# ─── Encoding ────────────────────────────────────────────────────────────────

# The rules of Ollama's System One route, for every model and runner (Ollama 0.40.6: 1–64 questions,
# score levels 2–10, Choice options up to 255 — the documented maximum; `instructions` optional). A model
# that takes less says so in its own HTTP 400, which reaches the caller as a SystemOneFailure — measured on
# Ollama 0.40.6: Tev1 takes 24 options, Nimble and Clef Flash 26; tev1, nimble, strands-decider and kev
# refuse a question without `instructions`, clef-flash answers it.
const _OLLAMA_MAX_QUESTIONS = 64
const _OLLAMA_MAX_OPTIONS = 255
const _OLLAMA_MAX_LEVELS = 10

function _ollama_instructions(name::String, instructions)::Nothing
    isnothing(instructions) && return nothing
    instructions isa AbstractString && isempty(strip(instructions)) && throw(ArgumentError(
        "Ollama reads blank `instructions` as none; those of $(repr(name)) are blank: say what it asks, or pass none"))
    instructions isa Union{AbstractString,AbstractDict,AbstractVector,NamedTuple} || throw(ArgumentError(
        "Ollama reads `instructions` as a string, object or array; $(repr(name)) has $(typeof(instructions))"))
    nothing
end

_ollama_candidates(name::String, n::Int, hi::Int, what::String, fix::String)::Nothing = 2 <= n <= hi ? nothing :
    throw(ArgumentError("Ollama accepts 2 to $(hi) $(what) per question; $(repr(name)) has $(n). " *
                        (n < 2 ? "Offer at least two" : fix)))

function _ollama_carries(name::String, q::ChoiceQuestion)::Nothing
    _ollama_instructions(name, q.instructions)
    for (option, d) in q.criteria
        isnothing(d) || d isa AbstractString || throw(ArgumentError(
            "Ollama reads a Choice option's description as a string or `nothing`; option $(repr(option)) of " *
            "$(repr(name)) has $(typeof(d)). Describe it in words, or pass `nothing`"))
    end
    _ollama_candidates(name, length(q.criteria), _OLLAMA_MAX_OPTIONS, "options", "Split it, or narrow the options")
end

function _ollama_carries(name::String, q::ScoreQuestion)::Nothing
    _ollama_instructions(name, q.instructions)
    for (i, level) in pairs(q.criteria)
        level isa AbstractString || throw(ArgumentError(
            "Ollama reads a Score level as a string; level $(i - 1) of $(repr(name)) is $(typeof(level))"))
    end
    _ollama_candidates(name, length(q.criteria), _OLLAMA_MAX_LEVELS, "levels", "Merge levels into a coarser rubric")
end

function _ollama_carries(name::String, q::NoulQuestion)::Nothing
    _ollama_instructions(name, q.instructions)
    c = q.criteria
    isnothing(c) && return nothing
    for (side, v) in (("yes", c.yes), ("no", c.no))
        isnothing(v) || v isa AbstractString || throw(ArgumentError(
            "Ollama reads a Noul criterion as a string; the `$(side)` criterion of $(repr(name)) is $(typeof(v))"))
    end
    nothing
end

_ollama_carries(::String, ::RawQuestion)::Nothing = nothing
_ollama_carries(name::String, ::BoxQuestion) = throw(ArgumentError(
    "Box questions are not supported by Ollama, whose System One route answers Choice, Score and Noul " *
    "questions; $(repr(name)) is one"))
_ollama_carries(name::String, q::SystemOneQuestion) = throw(ArgumentError(
    "Ollama cannot send $(repr(name)), a $(nameof(typeof(q))): its System One route answers Choice, Score and " *
    "Noul questions, and this wrapper sends a RawQuestion verbatim"))

# The endpoint's own keep_alive, the one its chat and embeddings requests carry.
_ollama_endpoint_keep_alive(s::OllamaEndpoint)::Union{Nothing,Float64} = s.keep_alive
_ollama_endpoint_keep_alive(_)::Union{Nothing,Float64} = nothing

function _ollama_system_one_keep_alive(options::NamedTuple, service)::Union{Nothing,Float64}
    other = Symbol[k for k in keys(options) if k !== :keep_alive]
    isempty(other) || throw(ArgumentError(
        "UniLM's Ollama wrapper accepts one System One option, `keep_alive`; got " * join(other, ", ")))
    haskey(options, :keep_alive) || return _ollama_endpoint_keep_alive(service)
    ka = options.keep_alive
    ka isa Real && !(ka isa Bool) && !isnan(ka) && 0 <= ka && (isinf(ka) || ka <= 1e9) || throw(ArgumentError(
        "Ollama's `keep_alive` is a number of seconds from 0 to 1e9, or Inf to keep the model loaded; got $(repr(ka))"))
    Float64(ka)
end

# The text state and the images an Ollama System One request sends. Ollama takes images as a list
# shared by all questions and shows them before the text, so the forms that keep that order are an
# image, or a vector or tuple whose images all come before its texts; the texts become the state
# (none: "", one: that string, several: an array). An image anywhere else could only be moved, never
# sent where the caller put it, and is refused. nl_dispatch sends its arguments inside an object, so
# the fix it is given is its `state` keyword.
function _ollama_state_images(state)::Tuple{Any,Vector{ImageAttachment}}
    state isa ImageAttachment && return ("", ImageAttachment[state])
    if state isa Union{AbstractVector,Tuple} && any(x -> x isa ImageAttachment, state)
        all(x -> x isa Union{ImageAttachment,AbstractString}, state) || throw(ArgumentError(
            "Ollama takes a state of images then texts, and this one mixes images with " *
            "$(join(unique(string(typeof(x)) for x in state if !(x isa Union{ImageAttachment,AbstractString})), ", ")): " *
            "write the other values as text, after the images"))
        first_text = findfirst(x -> x isa AbstractString, collect(state))
        isnothing(first_text) || all(x -> x isa AbstractString, collect(state)[first_text:end]) || throw(ArgumentError(
            "Ollama shows a request's images before its text, so an image after a text cannot be sent where it is: " *
            "put the images first, e.g. [image, \"What is in it?\"]"))
        texts = String[x for x in state if x isa AbstractString]
        return (isempty(texts) ? "" : length(texts) == 1 ? only(texts) : texts,
                ImageAttachment[x for x in state if x isa ImageAttachment])
    end
    isempty(_collect_images!(ImageAttachment[], state)) || throw(ArgumentError(
        "Ollama takes images as the state itself (an image, or a vector of images then texts), not inside a " *
        "JSON object or array. Pass them so: `[image, …texts]` as the state, or `state = [image, …texts]` to " *
        "nl_dispatch, which sends its arguments inside an object"))
    (state, ImageAttachment[])
end

function encode_system_one(p::OllamaSystemOneProtocol, service, request::SystemOneRequest)::SystemOneWire
    _check_request(p, service, request)
    n = length(request.questions)
    n <= _OLLAMA_MAX_QUESTIONS || throw(ArgumentError(
        "Ollama accepts at most $(_OLLAMA_MAX_QUESTIONS) questions per request; this one has $(n). " *
        "Split them across requests"))
    for (name, q) in request.questions
        _ollama_carries(name, q)
        isempty(_question_images!(ImageAttachment[], q)) || throw(ArgumentError(
            "Ollama takes images in the state only, shared by all questions; $(repr(name)) holds one. " *
            "Move it into the state, e.g. [image, \"…\"]"))
    end
    state, images = _ollama_state_images(request.state)
    isempty(images) && state isa AbstractString && isempty(strip(state)) && throw(ArgumentError(
        "Ollama refuses a blank `state` without images: it must hold the content the questions are about"))
    keep_alive = _ollama_system_one_keep_alive(request.options, service)
    o = _system_one_json(request, _system_one_model(p, service, request))
    o["state"] = state
    service isa OllamaEndpoint && _ollama_runner!(o, service)
    url = _api_base_url(service) * SYSTEMONE_PATH
    b64 = base64encode.(i.data for i in images)
    # The recording's key: images as digests, without keep_alive (it changes no answer).
    isempty(images) || (o["images"] = [_media_digest(i.mime, b) for (i, b) in zip(images, b64)])
    keyed = _system_one_body(o)
    isempty(images) && isnothing(keep_alive) && return SystemOneWire("POST", url, SYSTEMONE_PATH, keyed)
    isempty(images) || (o["images"] = b64)
    isnothing(keep_alive) || (o["keep_alive"] = _ollama_keep_alive(keep_alive))
    SystemOneWire("POST", url, SYSTEMONE_PATH, _system_one_body(o); record_body=keyed)
end

# ─── Decoding ────────────────────────────────────────────────────────────────

# TypeSafe's published confidence (https://docs.typesafe.ai/confidence), from the
# probabilities alone. A Choice: how far the top probability stands above an even split,
# 0 at an even split and 1 at certainty. A Score: the probability-weighted distance from
# the most likely level `m` — the lowest of a tie, as the published code picks it — against
# the same distance for an even spread measured from the middle, floored at 0. One outcome
# leaves nothing to doubt: 1, as TypeSafe answers a one-option Choice.
_choice_confidence(ps::AbstractVector{Float64})::Float64 =
    (n = length(ps); n == 1 ? 1.0 : clamp((n * maximum(ps) - 1) / (n - 1), 0.0, 1.0))

function _score_confidence(ps::AbstractVector{Float64})::Float64
    n = length(ps)
    n == 1 && return 1.0
    m = argmax(ps)
    spread = sum(p * abs(i - m) for (i, p) in pairs(ps))
    even = sum(abs(i - (n + 1) / 2) for i in 1:n) / n
    max(0.0, 1 - spread / even)
end

_surface_confidence(::String, a::SystemOneAnswer)::SystemOneAnswer = a
function _surface_confidence(name::String, a::ChoiceAnswer)::ChoiceAnswer
    isempty(a.probabilities) && throw(ArgumentError("System One answer $(repr(name)) gives no probabilities"))
    ChoiceAnswer(a.choice, _choice_confidence(collect(values(a.probabilities))), a.probabilities, a.raw)
end
# The answer to a typed Score question has passed `_check_answers`, so its levels are 0 to
# n − 1; only the answer to a RawQuestion can fail this, and a Score whose levels are not a
# rubric has no confidence to compute.
function _surface_confidence(name::String, a::ScoreAnswer)::ScoreAnswer
    n = length(a.probabilities)
    n > 0 && all(l -> haskey(a.probabilities, l), 0:n-1) || throw(ArgumentError(
        "System One answer $(repr(name)) has the levels $(sort!(collect(keys(a.probabilities)))), not 0 to $(n - 1)"))
    ScoreAnswer(a.score, _score_confidence(Float64[a.probabilities[l] for l in 0:n-1]), a.legend,
                a.probabilities, a.raw)
end

# `prompt_eval_cached_count`, top level of the reply, is the prompt tokens served from the
# cache. Like the other counts it is accounting: a value that is no token count reads as 0.
function _ollama_system_one_usage(r::SystemOneResponse)::TokenUsage
    v = get(r.raw, "prompt_eval_cached_count", 0)
    cached = _token_count(v)
    isnothing(cached) &&
        @warn "Ollama's prompt_eval_cached_count is not a non-negative whole token count; it reads as 0" cached = v
    u = r.usage
    TokenUsage(prompt_tokens=u.prompt_tokens, completion_tokens=u.completion_tokens, total_tokens=u.total_tokens,
               cached_tokens=something(cached, 0))
end

function decode_system_one(p::OllamaSystemOneProtocol, service, request::SystemOneRequest,
                           response::HTTP.Response)::Union{SystemOneSuccess,SystemOneFailure}
    response.status == 200 || return _ollama_system_one_failure(request, response)
    decoded = _decode_systemone(String(response.body), nothing, service)
    _check_answers(p, request, decoded)
    answers = Dict{String,SystemOneAnswer}(name => _surface_confidence(name, a) for (name, a) in decoded.answers)
    # The route documents the reply's `model` as the request's, but a cloud model may answer
    # without its tag (Ollama's published example: "clef" for "clef:cloud"); the requested name
    # is kept, so the answer is priced as the cloud model it is. A request that names no model
    # (`ask` always names one) keeps the reply's.
    model = _ollama_billed_model(request.model, decoded.model)
    SystemOneSuccess(SystemOneResponse(model, answers, _ollama_system_one_usage(decoded), nothing, decoded.raw, service))
end

# Ollama's error body is `{"error": "…"}`. A model that is not installed is a 404 that
# says so ("model 'x' not found"); the fix is to pull it.
function _ollama_system_one_failure(request::SystemOneRequest, resp::HTTP.Response)::SystemOneFailure
    body = String(resp.body)
    message = _typesafe_error_message(body)
    if resp.status in (400, 404) && occursin(r"\bmodel\b.*\bnot found"i, message)
        model = something(request.model, "<model>")
        message *= " — pull it first: `ollama pull $(model)`, or `pull_model($(repr(model)))`"
    end
    SystemOneFailure(response=body, status=resp.status, message=message, retry_after=_retry_after_seconds(resp))
end
