# ============================================================================
# TypeSafe System One API (Jev) — the TypeSafe wrapper of the System One surface.
#
# Two endpoints: POST /v1/systemone evaluates questions about a piece of state,
# GET /v1/models lists the model names the account may name in `model`. The
# endpoint, its credentials and base URL, TypeSafe's protocol (its limits, its
# wire, its decoder and error bodies) and its models listing live here; the
# questions, answers, verbs and answer checks are the generic surface in
# system_one.jl.
# ============================================================================

"""
    TYPESAFEServiceEndpoint <: ServiceEndpoint

TypeSafe System One API endpoint (model family Jev). Requires the
`TYPESAFE_API_KEY` environment variable; `TYPESAFE_BASE_URL` overrides the API
root and `TYPESAFE_DEFAULT_MODEL` the default model name. It speaks
[`TypeSafeProtocol`](@ref UniLM.TypeSafeProtocol), and it is the System One
backend a call uses when it names none and no [`with_system_one`](@ref) scope
sets another.

This endpoint speaks neither the OpenAI chat wire nor any other chat wire: it
declares only `:system_one` and `:models`, so [`ask`](@ref) and
[`list_models`](@ref) accept it while the verbs — `chatrequest!`, `respond`,
`embeddingrequest!`, `moderate` and the other OpenAI platform verbs — reject it
up front with an `ArgumentError` rather than posting a request no server here
would answer. A `Chat` or an `Embeddings` that names this endpoint can be built
only with an explicit `model=` (without one the constructor throws `ArgumentError`,
since there is no default model to resolve); sending it is refused.
"""
struct TYPESAFEServiceEndpoint <: ServiceEndpoint end

provider_capabilities(::Type{TYPESAFEServiceEndpoint}) = Set([:system_one, :models])

# A chat/agentic verb resolves its model before any capability check runs, so an
# endpoint with no `default_model` method would surface the refusal as a
# MethodError from deep inside the constructor. Refuse in the same words the
# capability check uses, so `respond(...; service=TYPESAFEServiceEndpoint)` reads
# the same with or without a `model=`.
default_model(::Type{TYPESAFEServiceEndpoint}) = throw(ArgumentError(
    "Chat and agentic APIs are not supported by TYPESAFEServiceEndpoint. Supported: " *
    join(sort(collect(provider_capabilities(TYPESAFEServiceEndpoint))), ", ") *
    ". `ask` is the System One verb."))

# Trailing slashes are stripped before a path is appended, matching the
# official SDKs — an override of "https://host/" must not produce "//v1/...".
# Blank environment values are ignored (an exported-but-empty variable is not a
# configuration choice).
function _resolve_base_url(::Type{TYPESAFEServiceEndpoint})::String
    override = strip(get(ENV, TYPESAFE_BASE_URL_ENV, ""))
    String(rstrip(isempty(override) ? TYPESAFE_BASE_URL : override, '/'))
end

"Client identity sent as both `User-Agent` and `X-TypeSafe-SDK`, fixed when the package loads."
const _TYPESAFE_AGENT = string("UniLM.jl/", something(pkgversion(@__MODULE__), v"0.0.0"))

function auth_header(::Type{TYPESAFEServiceEndpoint})::Vector{Pair{String,String}}
    key = strip(get(ENV, TYPESAFE_API_KEY, ""))
    # Fail with a named ArgumentError, not the KeyError `ENV[...]` would raise:
    # the verbs catch it into a typed call error, so a missing key surfaces as a
    # result that says which variable to set.
    isempty(key) && throw(ArgumentError("$TYPESAFE_API_KEY is not set"))
    [
        "Authorization" => "Bearer $key",
        "Content-Type" => "application/json",
        "Accept" => "application/json",
        "User-Agent" => _TYPESAFE_AGENT,
        "X-TypeSafe-SDK" => _TYPESAFE_AGENT,
        "X-TypeSafe-Runtime" => "julia/$(VERSION) ($(Sys.KERNEL); $(Sys.ARCH))",
    ]
end

"""
    default_typesafe_model() -> String

The model [`ask`](@ref) names on TypeSafe when a call does not:
`TYPESAFE_DEFAULT_MODEL` when it is set and non-blank, otherwise `"jev-latest"`.
Read when the request is sent, so exporting the variable after the package
loads — or after a [`SystemOneRequest`](@ref) without a model was built — still
takes effect.
"""
function default_typesafe_model()::String
    v = strip(get(ENV, TYPESAFE_DEFAULT_MODEL_ENV, ""))
    isempty(v) ? TYPESAFE_DEFAULT_MODEL : String(v)
end

# ─── Protocol ────────────────────────────────────────────────────────────────

"""
    TypeSafeProtocol <: SystemOneProtocol

TypeSafe's System One wire: `POST {base}/v1/systemone` with the JSON body
`{"state", "model", "questions"}` in the System One JSON form, answers decoded
from `{"model", "answers", "usage"}`, the request id read from the
`x-typesafe-request-id` header, and failures from TypeSafe's three error-body
shapes. Recordings are keyed under the plain path (no namespace). Before any
I/O it refuses what TypeSafe's wire cannot carry in this version: more than 255
options in a Choice, more than 10 levels in a Score, a Box question, an image,
and any provider option. It sends a [`RawQuestion`](@ref) verbatim and never
expects a refusal.

The untyped fallback of [`system_one_protocol`](@ref UniLM.system_one_protocol)
returns it, so [`TYPESAFEServiceEndpoint`](@ref) and every endpoint that
declares `:system_one` without a protocol of its own speak it.

Public extension API (not exported).
"""
struct TypeSafeProtocol <: SystemOneProtocol end

system_one_protocol(::Type{TYPESAFEServiceEndpoint}) = TypeSafeProtocol()
system_one_protocol(service) = TypeSafeProtocol()

_system_one_default_model(::TypeSafeProtocol, service)::String = default_typesafe_model()
# TypeSafe's recordings are filed under the plain path, as they always were, so every
# recording made before backends had namespaces keeps its key.
_system_one_namespace(::TypeSafeProtocol)::String = ""
_system_one_request_id(::TypeSafeProtocol, resp::HTTP.Response)::Union{Nothing,String} = _typesafe_request_id(resp)
# TypeSafe's probabilities sum to 1 exactly as decimals: over every answer recorded in
# docs/recorded_answers, kept in test/fixtures/typesafe and captured live on 2026-10-19
# (142 Choice and Score distributions, up to 255 options), the largest Float64 deviation
# of a sum from 1 was 2.2e-16 — one unit in the last place. The generic tolerance,
# ten units of rounding per outcome, is at least ten times that for any number of outcomes.
_probability_tolerance(::TypeSafeProtocol, n::Int)::Float64 = 10 * n * eps(1.0)
_allows_refusal(::TypeSafeProtocol)::Bool = false
_allows_raw(::TypeSafeProtocol)::Bool = true

# ─── Encoding ────────────────────────────────────────────────────────────────

# TypeSafe's Choice and Score limits: 256 options or 11 levels are rejected by the
# service, so they are refused here, before a round trip.
const _TYPESAFE_MAX_OPTIONS = 255
const _TYPESAFE_MAX_LEVELS = 10

_typesafe_carries(name::String, q::ChoiceQuestion) = length(q.criteria) <= _TYPESAFE_MAX_OPTIONS ||
    throw(ArgumentError("TypeSafe accepts at most $(_TYPESAFE_MAX_OPTIONS) options per Choice question; " *
                        "$(repr(name)) has $(length(q.criteria)). Split it, or narrow the options"))
_typesafe_carries(name::String, q::ScoreQuestion) = length(q.criteria) <= _TYPESAFE_MAX_LEVELS ||
    throw(ArgumentError("TypeSafe accepts at most $(_TYPESAFE_MAX_LEVELS) levels per Score question; " *
                        "$(repr(name)) has $(length(q.criteria)). Merge levels into a coarser rubric"))
_typesafe_carries(::String, ::Union{NoulQuestion,RawQuestion}) = true
_typesafe_carries(name::String, ::BoxQuestion) = throw(ArgumentError(
    "Box questions are not supported by this version's TypeSafe wrapper yet; $(repr(name)) is one"))
_typesafe_carries(name::String, q::SystemOneQuestion) = throw(ArgumentError(
    "TypeSafe cannot send $(repr(name)), a $(nameof(typeof(q))): its wire carries Choice, Score and Noul " *
    "questions, and a RawQuestion verbatim"))

function encode_system_one(p::TypeSafeProtocol, service, request::SystemOneRequest)::SystemOneWire
    _check_request(p, service, request)
    for (name, q) in request.questions
        _typesafe_carries(name, q)
    end
    isempty(_system_one_images(request)) || throw(ArgumentError(
        "images in a System One request are not supported by this version's TypeSafe wrapper yet; " *
        "send the request without them"))
    isempty(request.options) || throw(ArgumentError(
        "TypeSafe accepts no request options in this version (accepted: none); got " *
        join(map(string, collect(keys(request.options))), ", ")))
    # Exactly three top-level keys: the service rejects any other with 400, so nothing
    # unknown is ever written even though unknown fields are ignored on read.
    body = _system_one_body(_system_one_json(request, _system_one_model(p, service, request)))
    SystemOneWire("POST", _api_base_url(service) * SYSTEMONE_PATH, SYSTEMONE_PATH, body)
end

# ─── Decoding ────────────────────────────────────────────────────────────────
# Permissive on read (unknown fields ignored, an unknown answer type preserved),
# but a REQUIRED field that is absent, or of the wrong JSON type, is malformed
# provider output, not a defaultable zero: it throws, and `ask` turns the throw
# into a SystemOneCallError so no success is ever built from a broken payload.

function _answer_field(d::AbstractDict, key::AbstractString, name::AbstractString)
    haskey(d, key) || throw(ArgumentError(
        "System One answer $(repr(name)) has no \"$key\" field"))
    d[key]
end

# A JSON number: `true` and `false` are not numbers, though Julia's `Bool` is a `Real`.
_json_number(v)::Bool = v isa Real && !(v isa Bool)

function _answer_number(d::AbstractDict, key::AbstractString, name::AbstractString)::Float64
    v = _answer_field(d, key, name)
    _json_number(v) || throw(ArgumentError(
        "System One answer $(repr(name)) field \"$key\" is not a number (got $(typeof(v)))"))
    Float64(v)
end

function _answer_string(d::AbstractDict, key::AbstractString, name::AbstractString)::String
    v = _answer_field(d, key, name)
    v isa AbstractString || throw(ArgumentError(
        "System One answer $(repr(name)) field \"$key\" is not a string (got $(typeof(v)))"))
    String(v)
end

function _answer_map(d::AbstractDict, key::AbstractString, name::AbstractString)::AbstractDict
    v = _answer_field(d, key, name)
    v isa AbstractDict || throw(ArgumentError(
        "System One answer $(repr(name)) field \"$key\" is not an object (got $(typeof(v)))"))
    v
end

# A level key is a 0-based level number in canonical decimal: "-1", "+1", " 2" and "01"
# are not levels, and reading "01" as 1 would merge two wire entries into one.
function _level_key(k, name::AbstractString)::Int
    s = string(k)
    occursin(r"^(0|[1-9][0-9]*)$", s) || throw(ArgumentError(
        "System One score answer $(repr(name)) has the level key $(repr(s)); levels are 0-based integers " *
        "rendered as canonical decimal strings"))
    lvl = tryparse(Int, s)
    isnothing(lvl) && throw(ArgumentError(
        "System One score answer $(repr(name)) has the level key $(repr(s)), too large to be a level"))
    lvl
end

function _probability(v, name::AbstractString)::Float64
    _json_number(v) || throw(ArgumentError(
        "System One answer $(repr(name)) has a non-numeric probability (got $(typeof(v)))"))
    Float64(v)
end

function _named_probabilities(d::AbstractDict, name::AbstractString)::Dict{String,Float64}
    out = Dict{String,Float64}()
    for (k, v) in _answer_map(d, "probabilities", name)
        out[string(k)] = _probability(v, name)
    end
    out
end

function _put_level!(out::AbstractDict{Int}, k, v, name::AbstractString)
    lvl = _level_key(k, name)
    haskey(out, lvl) && throw(ArgumentError(
        "System One score answer $(repr(name)) gives level $(lvl) twice"))
    out[lvl] = v
    out
end

function _level_probabilities(d::AbstractDict, name::AbstractString)::Dict{Int,Float64}
    out = Dict{Int,Float64}()
    for (k, v) in _answer_map(d, "probabilities", name)
        _put_level!(out, k, _probability(v, name), name)
    end
    out
end

function _legend(d::AbstractDict, name::AbstractString)::Dict{Int,Any}
    out = Dict{Int,Any}()
    for (k, v) in _answer_map(d, "legend", name)
        _put_level!(out, k, v, name)
    end
    out
end

function _decode_answer(name::AbstractString, d::AbstractDict)::SystemOneAnswer
    raw = Dict{String,Any}(string(k) => v for (k, v) in d)
    kind = _answer_string(d, "type", name)
    if kind == "noul"
        NoulAnswer(_answer_number(d, "noul", name), raw)
    elseif kind == "choice"
        ChoiceAnswer(_answer_string(d, "choice", name), _answer_number(d, "confidence", name),
                     _named_probabilities(d, name), raw)
    elseif kind == "score"
        ScoreAnswer(_answer_number(d, "score", name), _answer_number(d, "confidence", name),
                    _legend(d, name), _level_probabilities(d, name), raw)
    else
        UnknownAnswer(kind, raw)
    end
end

# A token count: a JSON number that is a whole, non-negative value an `Int` holds.
_token_count(v)::Union{Nothing,Int} =
    _json_number(v) && isfinite(v) && isinteger(v) && 0 <= v < 2.0^63 ? Int(v) : nothing

# The spec makes both usage fields required; the official SDKs relax them, so a
# missing one reads as 0 rather than failing an otherwise complete evaluation. A value
# that is present but no token count reads as 0 too, with one warning: usage is
# accounting, never a reason to discard answers that were paid for.
function _typesafe_usage(d::AbstractDict)::TokenUsage
    haskey(d, "usage") || return TokenUsage()
    u = d["usage"]
    if !(u isa AbstractDict)
        @warn "System One usage is not an object; its token counts read as 0" usage = u
        return TokenUsage()
    end
    i = _token_count(get(u, "input_tokens", 0))
    o = _token_count(get(u, "output_tokens", 0))
    (isnothing(i) || isnothing(o)) &&
        @warn "System One usage holds a value that is not a non-negative whole token count; it reads as 0" usage = u
    i, o = something(i, 0), something(o, 0)
    TokenUsage(prompt_tokens=i, completion_tokens=o, total_tokens=i + o)
end

function _decode_systemone(body::AbstractString, request_id::Union{Nothing,String},
                           service::Union{Nothing,ServiceEndpointSpec}=nothing)::SystemOneResponse
    parsed = JSON.parse(body; dicttype=Dict{String,Any})
    parsed isa AbstractDict || throw(ArgumentError("System One response body is not a JSON object"))
    raw = Dict{String,Any}(string(k) => v for (k, v) in parsed)
    amap = get(raw, "answers", nothing)
    amap isa AbstractDict || throw(ArgumentError(
        "System One response carries no \"answers\" object"))
    out = Dict{String,SystemOneAnswer}()
    for (k, v) in amap
        name = string(k)
        v isa AbstractDict || throw(ArgumentError(
            "System One answer $(repr(name)) is not a JSON object"))
        out[name] = _decode_answer(name, v)
    end
    # The versioned id that answered is what a tuned threshold and a price are about: a
    # response that does not name it is malformed, not an answer from the model "".
    model = get(raw, "model", nothing)
    model isa AbstractString || throw(ArgumentError(
        "System One response carries no \"model\" string naming the model that answered"))
    SystemOneResponse(String(model), out, _typesafe_usage(raw), request_id, raw, service)
end

function decode_system_one(p::TypeSafeProtocol, service, request::SystemOneRequest,
                           response::HTTP.Response)::Union{SystemOneSuccess,SystemOneFailure}
    response.status == 200 || return _typesafe_failure(response)
    decoded = _decode_systemone(String(response.body), _typesafe_request_id(response), service)
    _check_answers(p, request, decoded)
    SystemOneSuccess(decoded)
end

# ─── Error bodies ────────────────────────────────────────────────────────────
# Three `detail` shapes exist: a validation list (422), an object carrying
# `error_type`/`message` (400 and both auth failures), and a plain string (404,
# 405, and the semantic 400s). `{"error": …}` / `{"message": …}` are handled too,
# as the official SDKs do, even though the live service never produced them.

const _TYPESAFE_MAX_MESSAGE = 200

_truncate_body(s::AbstractString)::String =
    (t = strip(s); length(t) <= _TYPESAFE_MAX_MESSAGE ? String(t) : String(first(t, _TYPESAFE_MAX_MESSAGE)))

function _typesafe_parse_body(body::AbstractString)
    try
        JSON.parse(body; dicttype=Dict{String,Any})
    catch e
        e isa InterruptException && rethrow()
        nothing   # not JSON: the caller falls back to the raw text
    end
end

function _validation_path(loc)::String
    loc isa AbstractVector || return ""
    parts = String[]
    for (i, seg) in enumerate(loc)
        # FastAPI prefixes every location with the request part it validated;
        # "body" is noise in a message about a field of the body.
        i == 1 && seg isa AbstractString && seg == "body" && continue
        push!(parts, seg isa AbstractString ? String(seg) : string(seg))
    end
    join(parts, ".")
end

_detail_message(v::AbstractString)::Union{Nothing,String} = String(v)
_detail_message(v::AbstractDict)::Union{Nothing,String} =
    (m = get(v, "message", nothing); m isa AbstractString ? String(m) : nothing)
function _detail_message(v::AbstractVector)::Union{Nothing,String}
    parts = String[]
    for e in v
        e isa AbstractDict || continue
        msg = get(e, "msg", nothing)
        msg isa AbstractString || continue
        path = _validation_path(get(e, "loc", nothing))
        push!(parts, isempty(path) ? String(msg) : string(path, ": ", msg))
    end
    isempty(parts) ? nothing : join(parts, "; ")
end
_detail_message(::Any)::Union{Nothing,String} = nothing

"""
    _typesafe_error_message(body) -> String

The human-readable message inside a TypeSafe error body, whichever shape it
used. Validation lists become `"<dotted loc>: <msg>"` entries joined by `"; "`.
With no message string anywhere in the body, an object `detail`'s `error_type`
stands in for one. A body that is not JSON, or that carries neither, is returned
as raw text truncated to 200 characters.
"""
function _typesafe_error_message(body::AbstractString)::String
    parsed = _typesafe_parse_body(body)
    parsed isa AbstractDict || return _truncate_body(body)
    for key in ("detail", "error", "message")
        haskey(parsed, key) || continue
        m = _detail_message(parsed[key])
        isnothing(m) || return m
    end
    # A `detail` object with no message still names its `error_type` (a live 400 sends
    # `{"detail":{"error_type":"max_tokens_exceeded"}}`), which says more than raw JSON;
    # it comes last, because a message string anywhere in the body is written for a reader.
    something(_typesafe_error_type(body), _truncate_body(body))
end

"`detail.error_type` when the body uses the object shape, else `nothing`."
function _typesafe_error_type(body::AbstractString)::Union{Nothing,String}
    parsed = _typesafe_parse_body(body)
    parsed isa AbstractDict || return nothing
    detail = get(parsed, "detail", nothing)
    detail isa AbstractDict || return nothing
    t = get(detail, "error_type", nothing)
    t isa AbstractString ? String(t) : nothing
end

_typesafe_request_id(resp::HTTP.Response)::Union{Nothing,String} =
    (v = HTTP.header(resp, "x-typesafe-request-id", ""); isempty(v) ? nothing : String(v))
_typesafe_request_id(::Nothing) = nothing

"""
    _typesafe_retry_after(resp) -> Union{Nothing,Float64}

The wait a rate-limited or overloaded response asked for, in seconds.

`retry-after-ms` is read first and `retry-after` second: a server that sends both
means the millisecond form, and reading only the coarse one rounds a sub-second
hint up to a whole second of idle client. `retry-after` itself takes either
delta-seconds or an HTTP-date, both left to `_retry_after_seconds` — the seam's
own reader, so the two forms are not parsed twice by two rules that could drift
apart. `nothing` when neither header arrived or neither parsed: a value
that is absent, malformed or not a finite number must leave the client's own
backoff in charge, never make it throw or sleep forever.
"""
function _typesafe_retry_after(resp::HTTP.Response)::Union{Nothing,Float64}
    ms = strip(HTTP.header(resp, "retry-after-ms", ""))
    if !isempty(ms)
        v = tryparse(Float64, ms)
        if !isnothing(v) && isfinite(v)
            return max(0.0, v / 1000)
        end
    end
    return _retry_after_seconds(resp)
end

_typesafe_failure(resp::HTTP.Response)::SystemOneFailure = begin
    body = String(resp.body)
    SystemOneFailure(response=body, status=resp.status, request_id=_typesafe_request_id(resp),
                     error_type=_typesafe_error_type(body), message=_typesafe_error_message(body),
                     retry_after=_typesafe_retry_after(resp))
end

_typesafe_call_error(e, resp::Union{Nothing,HTTP.Response})::SystemOneCallError =
    _callerr(SystemOneCallError, e; request_id=_typesafe_request_id(resp))

# ─── Models ──────────────────────────────────────────────────────────────────

"""
    TypeSafeModelCard

One entry of `GET /v1/models`: a model `name` accepted by
[`SystemOneRequest`](@ref)`.model`, a human-readable `description`, and
`release_date`.

`release_date` is kept as an opaque `String`: the documented shape is
`YYYY-MM-DD` but the service returns an RFC 3339 timestamp, so parsing it into a
date here would either fail or guess. `raw` holds the unparsed JSON card.

The list contains aliases only. A pinned version such as `"jev-1.13.0"` is a
valid `model` even though it does not appear here, and the list is scoped to the
authenticated account — do not hard-code names read from someone else's list.
"""
struct TypeSafeModelCard
    name::String
    description::String
    release_date::String
    raw::Dict{String,Any}
end

"Successful [`list_models`](@ref) result: the [`TypeSafeModelCard`](@ref)s and the unparsed body."
struct TypeSafeModelsSuccess <: LLMRequestResponse
    models::Vector{TypeSafeModelCard}
    raw::Dict{String,Any}
end

function _model_card(d::AbstractDict)::TypeSafeModelCard
    raw = Dict{String,Any}(string(k) => v for (k, v) in d)
    name = get(raw, "name", nothing)
    name isa AbstractString || throw(ArgumentError("a model card has no \"name\" field"))
    text(key) = (v = get(raw, key, ""); v isa AbstractString ? String(v) : string(v))
    TypeSafeModelCard(String(name), text("description"), text("release_date"), raw)
end

function _system_one_models(::TypeSafeProtocol, service, config::Union{Nothing,RequestConfig},
                            cancel::Union{Nothing,CancelToken})
    validate_capability(service, :models, "TypeSafe models listing")
    cfg = _resolve_config(config); tok = _resolve_cancel(cancel); t0 = time_ns()
    local resp
    try
        resp = _recorded_exchange("GET", TYPESAFE_MODELS_PATH, "", t0, tok) do
            _http_with_retries(cfg, t0, "GET", _api_base_url(service) * TYPESAFE_MODELS_PATH,
                               auth_header(service); cancel=tok)
        end
        resp.status == 200 || return _typesafe_failure(resp)
        parsed = JSON.parse(String(resp.body); dicttype=Dict{String,Any})
        parsed isa AbstractDict || throw(ArgumentError("models listing body is not a JSON object"))
        raw = Dict{String,Any}(string(k) => v for (k, v) in parsed)
        listed = get(raw, "models", nothing)
        listed isa AbstractVector || throw(ArgumentError(
            "models listing carries no \"models\" array"))
        cards = TypeSafeModelCard[]
        for c in listed
            c isa AbstractDict || throw(ArgumentError("a model card is not a JSON object"))
            push!(cards, _model_card(c))
        end
        TypeSafeModelsSuccess(cards, raw)
    catch e
        e isa Union{InterruptException,ReplayMissError,RecordingWriteError} && rethrow()
        _typesafe_call_error(e, @isdefined(resp) ? resp : nothing)
    end
end
