# ============================================================================
# TypeSafe System One API (Jev) — typed judgments instead of generated text.
#
# Two endpoints: POST /v1/systemone evaluates questions about a piece of state,
# GET /v1/models lists the model names the account may name in `model`.
# A System One model returns a distribution over the outcomes the caller
# enumerated (a Choice option, a Score level, or a yes/no probability), so the
# answer is already machine-readable — there is no text to parse and no
# streaming or batch surface to drive.
# ============================================================================

"""
    TYPESAFEServiceEndpoint <: ServiceEndpoint

TypeSafe System One API endpoint (model family Jev). Requires the
`TYPESAFE_API_KEY` environment variable; `TYPESAFE_BASE_URL` overrides the API
root and `TYPESAFE_DEFAULT_MODEL` the default model name.

This endpoint speaks neither the OpenAI chat wire nor any other chat wire: it
declares only `:system_one` and `:models`, so [`ask`](@ref) and
[`list_models`](@ref) accept it while the verbs — `chatrequest!`, `respond`,
`embeddingrequest!`, `moderate` and the other OpenAI platform verbs — reject it
up front with an `ArgumentError` rather than posting a request no server here
would answer. Building a `Chat` or an `Embeddings` that names this endpoint is
allowed; only sending one is refused.
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

"Client identity sent as both `User-Agent` and `X-TypeSafe-SDK`."
_typesafe_agent()::String = string("UniLM.jl/", something(pkgversion(@__MODULE__), v"0.0.0"))

function auth_header(::Type{TYPESAFEServiceEndpoint})::Vector{Pair{String,String}}
    key = strip(get(ENV, TYPESAFE_API_KEY, ""))
    # Fail with a named ArgumentError, not the KeyError `ENV[...]` would raise:
    # the verbs catch it into a typed call error, so a missing key surfaces as a
    # result that says which variable to set.
    isempty(key) && throw(ArgumentError("$TYPESAFE_API_KEY is not set"))
    agent = _typesafe_agent()
    [
        "Authorization" => "Bearer $key",
        "Content-Type" => "application/json",
        "Accept" => "application/json",
        "User-Agent" => agent,
        "X-TypeSafe-SDK" => agent,
        "X-TypeSafe-Runtime" => "julia/$(VERSION) ($(Sys.KERNEL); $(Sys.ARCH))",
    ]
end

"""
    default_typesafe_model() -> String

The model [`ask`](@ref) names when a call does not: `TYPESAFE_DEFAULT_MODEL`
when it is set and non-blank, otherwise `"jev-latest"`. Read at call time, so
exporting the variable after the package loads still takes effect.
"""
function default_typesafe_model()::String
    v = strip(get(ENV, TYPESAFE_DEFAULT_MODEL_ENV, ""))
    isempty(v) ? TYPESAFE_DEFAULT_MODEL : String(v)
end

# ─── Questions ───────────────────────────────────────────────────────────────

"""
    SystemOneEntry

What the wire accepts wherever a piece of guidance appears (`instructions`, a
Choice option description, a Score level, a Noul criterion): a string, a JSON
object, or a JSON array. Numbers and booleans are not entries — the model reads
words, and a bare number carries none.
"""
const SystemOneEntry = Union{AbstractString, AbstractDict, AbstractVector, NamedTuple}

"""
    SystemOneQuestion

Abstract supertype of the three question primitives: [`ChoiceQuestion`](@ref)
(pick one of up to 255 named options), [`ScoreQuestion`](@ref) (place the state
on an ordered rubric of up to 10 levels), and [`NoulQuestion`](@ref) (the
probability that a statement is true). Build them with [`choice`](@ref),
[`score`](@ref) and [`noul`](@ref).
"""
abstract type SystemOneQuestion end

_entry(v) = isnothing(v) || v isa SystemOneEntry ? v : throw(ArgumentError(
    "a System One entry must be a string, object, or array (or `nothing`); got $(typeof(v))"))

_score_level(v) = isnothing(v) ?
    throw(ArgumentError("Score levels may not be `nothing`; every level needs a description")) :
    _entry(v)

"""
    ChoiceQuestion(instructions, criteria::JSON.Object{String,Any})

One question that picks a single named option. `criteria` maps each option name
to its description, in insertion order (the order is sent on the wire; the
answer's `probabilities` map is unordered). A description of `nothing` tells the
model to interpret the option by its name alone.

Constructed through [`choice`](@ref), which normalizes the many ways to spell a
criteria map. The server accepts 1–255 options and rejects 256, so that bound is
checked here, before the round trip.
"""
struct ChoiceQuestion <: SystemOneQuestion
    instructions::Union{Nothing,SystemOneEntry}
    criteria::JSON.Object{String,Any}
    function ChoiceQuestion(instructions, criteria::JSON.Object{String,Any})
        n = length(criteria)
        1 <= n <= 255 || throw(ArgumentError(
            "a Choice question needs 1 to 255 options (the server rejects more than 255); got $n"))
        # Checked here and not only in the normalizer: this constructor is public,
        # and an option the model cannot name is not a describable outcome.
        any(isempty, keys(criteria)) && throw(ArgumentError(
            "Choice option names must be non-empty"))
        new(_entry(instructions), criteria)
    end
end

"""
    ScoreQuestion(instructions, criteria::Vector{Any})

One question that places the state on an ordered rubric. `criteria` lists the
levels lowest first; `criteria[1]` is level `0` on the 0-based scale the answer
reports, so a rubric of three levels yields a `score` in `0.0 … 2.0`.

Constructed through [`score`](@ref). The server accepts 1–10 levels and rejects
11, so that bound is checked here. A level may not be `nothing`: an unnamed
level is a rung the model cannot recognise.
"""
struct ScoreQuestion <: SystemOneQuestion
    instructions::Union{Nothing,SystemOneEntry}
    criteria::Vector{Any}
    function ScoreQuestion(instructions, criteria::Vector{Any})
        n = length(criteria)
        1 <= n <= 10 || throw(ArgumentError(
            "a Score question needs 1 to 10 levels (the server rejects more than 10); got $n"))
        new(_entry(instructions), Any[_score_level(v) for v in criteria])
    end
end

"""
    NoulCriteria(yes, no)

The two optional criteria of a [`NoulQuestion`](@ref): what makes the statement
true (`yes`, sent as the wire key `"true"`) and what makes it false (`no`, sent
as `"false"`). Either may be `nothing`, in which case that key is omitted
entirely rather than sent as null.
"""
struct NoulCriteria
    yes::Union{Nothing,SystemOneEntry}
    no::Union{Nothing,SystemOneEntry}
    NoulCriteria(yes, no) = new(_entry(yes), _entry(no))
end

"""
    NoulQuestion(instructions, criteria)

One question answered by a single probability in `0 … 1` — the model's belief
that the statement holds. There is no confidence and no distribution: the value
*is* the distribution, and `0.5` is maximal uncertainty.

Constructed through [`noul`](@ref). The server rejects a noul carrying neither
`instructions` nor a criterion, so that rule is checked here.
"""
struct NoulQuestion <: SystemOneQuestion
    instructions::Union{Nothing,SystemOneEntry}
    criteria::Union{Nothing,NoulCriteria}
    function NoulQuestion(instructions, criteria::Union{Nothing,NoulCriteria})
        told = !isnothing(criteria) && (!isnothing(criteria.yes) || !isnothing(criteria.no))
        (!isnothing(instructions) || told) || throw(ArgumentError(
            "a Noul question needs `instructions` or at least one of the `yes`/`no` criteria"))
        new(_entry(instructions), criteria)
    end
end

_option_name(k::AbstractString)::String = String(k)
_option_name(k::Symbol)::String = String(k)
_option_name(k) = throw(ArgumentError(
    "Choice option names must be a String or a Symbol; got $(typeof(k))"))

function _put_option!(obj::JSON.Object{String,Any}, k, v)
    name = _option_name(k)
    isempty(name) && throw(ArgumentError("Choice option names must be non-empty"))
    haskey(obj, name) && throw(ArgumentError(
        "duplicate Choice option name $(repr(name)); the wire is a map, so a repeat would silently drop one"))
    obj[name] = _entry(v)
    obj
end

_put_entry!(obj::JSON.Object{String,Any}, e::Pair) = _put_option!(obj, first(e), last(e))
_put_entry!(obj::JSON.Object{String,Any}, e::AbstractString) = _put_option!(obj, e, nothing)
_put_entry!(obj::JSON.Object{String,Any}, e::Symbol) = _put_option!(obj, e, nothing)
_put_entry!(::JSON.Object{String,Any}, e) = throw(ArgumentError(
    "a Choice criteria vector holds `name => description` pairs or bare option names; got $(typeof(e))"))

function _choice_criteria(nt::NamedTuple)::JSON.Object{String,Any}
    obj = JSON.Object{String,Any}()
    for (k, v) in pairs(nt); _put_option!(obj, k, v); end
    obj
end
function _choice_criteria(d::AbstractDict)::JSON.Object{String,Any}
    obj = JSON.Object{String,Any}()
    for (k, v) in d; _put_option!(obj, k, v); end
    obj
end
function _choice_criteria(v::AbstractVector)::JSON.Object{String,Any}
    obj = JSON.Object{String,Any}()
    for e in v; _put_entry!(obj, e); end
    obj
end
_choice_criteria(x) = throw(ArgumentError(
    "Choice `criteria` must be a NamedTuple, an AbstractDict, a vector of `name => description` " *
    "pairs, or a vector of option names; got $(typeof(x))"))

"""
    choice(instructions, criteria) -> ChoiceQuestion
    choice(criteria) -> ChoiceQuestion

Build a Choice question: the model picks exactly one of the named options and
reports a probability for every one of them.

`criteria` may be a `NamedTuple`, an `AbstractDict`, a vector of
`name => description` pairs, or a vector of bare option names (each described by
its name alone). Names may be `String`s or `Symbol`s and are stored as `String`s;
a description may be a string, an object, an array, or `nothing`.

**Option order is the insertion order of what you pass.** A `NamedTuple`, a
vector of pairs, and an ordered mapping such as `JSON.Object` all preserve it; a
plain `Dict` does not, so pass one of the ordered forms when the wire order
matters. Order never changes the answer's meaning — answers are keyed by name —
but it is what the model reads.

```julia
choice("Which team should handle this ticket?", (
    billing   = "Payments, invoicing, payouts, refunds",
    technical = "Bugs, outages, integrations",
    sales     = "Pricing, upgrades, new accounts",
))

choice(["billing", "shipping", "other"])   # descriptions are the names themselves
```
"""
choice(instructions, criteria) = ChoiceQuestion(instructions, _choice_criteria(criteria))
choice(criteria) = ChoiceQuestion(nothing, _choice_criteria(criteria))

"""
    score(instructions, levels::AbstractVector) -> ScoreQuestion
    score(levels::AbstractVector) -> ScoreQuestion

Build a Score question over an ordered rubric, lowest level first. `levels[1]`
is level `0`: the answer's `score` and its `probabilities` keys live on that
same 0-based scale.

Each level is a string, an object, or an array — never `nothing`. The server
accepts 1 to 10 levels, and TypeSafe advises at least two: a one-level rubric
gives the model nothing to place the state against.

```julia
score("How urgent is this ticket?",
      ["Can wait", "Needs attention this week", "Needs attention today"])
```

The returned `score` is a probability-weighted expectation, so it falls between
integer levels. Threshold it; do not read a magnitude into the fractional part.
"""
score(instructions, levels::AbstractVector) = ScoreQuestion(instructions, Any[v for v in levels])
score(levels::AbstractVector) = ScoreQuestion(nothing, Any[v for v in levels])

_noul_criteria(yes, no) = isnothing(yes) && isnothing(no) ? nothing : NoulCriteria(yes, no)

"""
    noul(instructions; yes=nothing, no=nothing) -> NoulQuestion
    noul(; yes=nothing, no=nothing) -> NoulQuestion

Build a Noul question — a single probability that the statement is true. `yes`
and `no` describe the two sides and are sent as the wire keys `"true"` and
`"false"`; either may be omitted.

At least one of `instructions`, `yes`, or `no` must be present: the server
rejects a bare noul, because nothing in it says what is being asked.

```julia
noul("Is the customer frustrated?";
     yes = "The customer expresses frustration or impatience",
     no  = "The customer is neutral or satisfied")
```

A Noul answer carries no confidence — the probability itself expresses it — and
`P(q) + P(not q)` is not guaranteed to be 1, so do not derive one from the other.
"""
noul(instructions; yes=nothing, no=nothing) = NoulQuestion(instructions, _noul_criteria(yes, no))
noul(; yes=nothing, no=nothing) = NoulQuestion(nothing, _noul_criteria(yes, no))

# ─── Question serialization ─────────────────────────────────────────────────
# `instructions` is written only when present: the server distinguishes an
# absent key from an explicit null for the Noul "instructions or criteria" rule,
# and an ordered JSON.Object keeps Choice options in the order they were given.

function JSON.lower(q::ChoiceQuestion)
    o = JSON.Object{String,Any}()
    o["type"] = "choice"
    isnothing(q.instructions) || (o["instructions"] = q.instructions)
    o["criteria"] = q.criteria
    o
end

function JSON.lower(q::ScoreQuestion)
    o = JSON.Object{String,Any}()
    o["type"] = "score"
    isnothing(q.instructions) || (o["instructions"] = q.instructions)
    o["criteria"] = q.criteria
    o
end

function JSON.lower(c::NoulCriteria)
    o = JSON.Object{String,Any}()
    isnothing(c.yes) || (o["true"] = c.yes)
    isnothing(c.no) || (o["false"] = c.no)
    o
end

function JSON.lower(q::NoulQuestion)
    o = JSON.Object{String,Any}()
    o["type"] = "noul"
    isnothing(q.instructions) || (o["instructions"] = q.instructions)
    isnothing(q.criteria) || (o["criteria"] = q.criteria)
    o
end

# ─── Request ─────────────────────────────────────────────────────────────────

"""
    SystemOneState

What `state` may be: a string, a JSON object, or a JSON array (a `Tuple` and a
`NamedTuple` serialize as those). `nothing` is deliberately absent — the server
reports a null state as a missing field (422), so an unset state is a type
error here instead of a wasted round trip.
"""
const SystemOneState = Union{AbstractString, AbstractDict, AbstractVector, Tuple, NamedTuple}

_question_name(k::AbstractString)::String = String(k)
_question_name(k::Symbol)::String = String(k)
_question_name(k) = throw(ArgumentError("question names must be a String or a Symbol; got $(typeof(k))"))

_question_value(q::SystemOneQuestion) = q
_question_value(q) = throw(ArgumentError(
    "a question must be a ChoiceQuestion, ScoreQuestion, or NoulQuestion; got $(typeof(q))"))

function _questions_from_items(items)::Vector{Pair{String,SystemOneQuestion}}
    named = any(x -> x isa Pair, items)
    bare = any(x -> x isa SystemOneQuestion, items)
    named && bare && throw(ArgumentError(
        "questions must be either all named (`\"name\" => question`) or all bare (auto-named " *
        "\"q1\", \"q2\", …); mixing the two would make the answer keys depend on argument position"))
    out = Pair{String,SystemOneQuestion}[]
    for (i, x) in enumerate(items)
        if x isa Pair
            push!(out, _question_name(first(x)) => _question_value(last(x)))
        elseif x isa SystemOneQuestion
            push!(out, "q$i" => x)
        else
            throw(ArgumentError(
                "each question must be a SystemOneQuestion or a `name => question` pair; got $(typeof(x))"))
        end
    end
    out
end

_normalize_questions(nt::NamedTuple)::Vector{Pair{String,SystemOneQuestion}} =
    Pair{String,SystemOneQuestion}[String(k) => _question_value(v) for (k, v) in pairs(nt)]
_normalize_questions(d::AbstractDict)::Vector{Pair{String,SystemOneQuestion}} =
    Pair{String,SystemOneQuestion}[_question_name(k) => _question_value(v) for (k, v) in d]
_normalize_questions(v::AbstractVector) = _questions_from_items(v)
_normalize_questions(t::Tuple) = _questions_from_items(t)
_normalize_questions(p::Pair) = _questions_from_items((p,))
_normalize_questions(q::SystemOneQuestion) = _questions_from_items((q,))
_normalize_questions(x) = throw(ArgumentError(
    "`questions` must be a SystemOneQuestion, a `name => question` pair, or a collection of " *
    "either (vector, tuple, NamedTuple, or Dict); got $(typeof(x))"))

# One question given positionally is the collection itself (a vector, NamedTuple
# or Dict of questions); several are the items of one.
_normalize_varargs(items::Tuple) =
    length(items) == 1 ? _normalize_questions(items[1]) : _questions_from_items(items)

"""
    SystemOneRequest(state, questions; model=default_typesafe_model())

One System One evaluation: every question in `questions` is answered about the
same `state`, in a single round trip. The state is ingested once no matter how
many questions ride along, which is why batching is far cheaper than one call
per question.

# Fields
- `state::SystemOneState`: a string, object, or array — the content every
  question refers to. A question points into it by naming a key in backticks,
  e.g. ``"Does `refund_policy` cover this charge?"``.
- `questions::Vector{Pair{String,SystemOneQuestion}}`: ordered, with unique
  non-empty names. Answers come back keyed by these names.
- `model::String`: `"jev-latest"`, `"jev-preview"`, or a pinned version such as
  `"jev-1.13.0"`. Aliases move; pin the version if you have tuned thresholds
  against it, and read [`SystemOneResponse`](@ref)`.model` to see which version
  answered.

`questions` accepts `name => question` pairs (vector, tuple, `NamedTuple`, or
`AbstractDict`), or bare questions in a vector or tuple, which are auto-named
`"q1"`, `"q2"`, … in order. Mixing bare and named questions, repeating a name,
passing none, or passing a blank model is an `ArgumentError`.
"""
struct SystemOneRequest
    state::SystemOneState
    questions::Vector{Pair{String,SystemOneQuestion}}
    model::String
    function SystemOneRequest(state::SystemOneState,
                              questions::Vector{Pair{String,SystemOneQuestion}},
                              model::AbstractString)
        isempty(questions) && throw(ArgumentError(
            "a System One request needs at least one question (the server rejects an empty map)"))
        name_of_model = String(model)
        isempty(strip(name_of_model)) && throw(ArgumentError(
            "`model` must be a non-empty model name such as \"jev-latest\" or \"jev-1.13.0\""))
        seen = Set{String}()
        for (name, _) in questions
            isempty(name) && throw(ArgumentError("question names must be non-empty"))
            name in seen && throw(ArgumentError(
                "duplicate question name $(repr(name)); answers are keyed by name, so names must be unique"))
            push!(seen, name)
        end
        new(state, questions, name_of_model)
    end
end

SystemOneRequest(state, questions; model::AbstractString=default_typesafe_model()) =
    SystemOneRequest(state, _normalize_questions(questions), model)

# Exactly three top-level keys: the server rejects any other with 400, so
# nothing unknown is ever written even though unknown fields are ignored on read.
function JSON.lower(r::SystemOneRequest)
    o = JSON.Object{String,Any}()
    o["state"] = r.state
    o["model"] = r.model
    qs = JSON.Object{String,Any}()
    for (name, q) in r.questions
        qs[name] = q
    end
    o["questions"] = qs
    o
end

# ─── Answers ─────────────────────────────────────────────────────────────────

"""
    SystemOneAnswer

Abstract supertype of the answers a System One evaluation returns:
[`ChoiceAnswer`](@ref), [`ScoreAnswer`](@ref), [`NoulAnswer`](@ref), and
[`UnknownAnswer`](@ref) for a `type` this version does not recognise.
"""
abstract type SystemOneAnswer end

"""
    NoulAnswer(noul, raw)

The probability in `0 … 1` that the statement is true. There is no confidence
field: the value itself carries the uncertainty, and `0.5` is maximal doubt.
`raw` is the unparsed JSON answer.
"""
struct NoulAnswer <: SystemOneAnswer
    noul::Float64
    raw::Dict{String,Any}
end

"""
    ChoiceAnswer(choice, confidence, probabilities, raw)

The option with the highest probability (`choice`), a `confidence` in `0 … 1`
summarising how concentrated the distribution is, and `probabilities` — one
entry per option of the question, keyed by option name and summing to
approximately 1. The map is unordered: the server does not return the request's
option order.

Read `probabilities` when the ranking matters; `choice` is only its argmax.
`raw` is the unparsed JSON answer.
"""
struct ChoiceAnswer <: SystemOneAnswer
    choice::String
    confidence::Float64
    probabilities::Dict{String,Float64}
    raw::Dict{String,Any}
end

"""
    ScoreAnswer(score, confidence, legend, probabilities, raw)

`score` is the probability-weighted expectation over the rubric levels, so it
may fall between them; `confidence` is in `0 … 1`. `legend` and `probabilities`
are keyed by the **0-based level number** (parsed from the wire's `"0"`, `"1"`,
…), so `argmax(a.probabilities)` is a level on the same scale as `score` and
`legend[argmax(a.probabilities)]` is that level's description — a string, or the
original object/array when the rubric was structured.

`raw` is the unparsed JSON answer.
"""
struct ScoreAnswer <: SystemOneAnswer
    score::Float64
    confidence::Float64
    legend::Dict{Int,Any}
    probabilities::Dict{Int,Float64}
    raw::Dict{String,Any}
end

"""
    UnknownAnswer(type, raw)

An answer whose `type` this version does not know, kept verbatim in `raw`
instead of being dropped. A new answer primitive on the service therefore
surfaces as data you can inspect rather than as a silently missing key.
"""
struct UnknownAnswer <: SystemOneAnswer
    type::String
    raw::Dict{String,Any}
end

Base.show(io::IO, a::NoulAnswer) = print(io, "NoulAnswer(", a.noul, ")")
Base.show(io::IO, a::ChoiceAnswer) =
    print(io, "ChoiceAnswer(", repr(a.choice), ", confidence=", a.confidence, ")")
Base.show(io::IO, a::ScoreAnswer) =
    print(io, "ScoreAnswer(", a.score, ", confidence=", a.confidence, ")")
Base.show(io::IO, a::UnknownAnswer) =
    print(io, "UnknownAnswer(", repr(a.type), ", ", length(a.raw), " fields)")

"""
    SystemOneResponse

A decoded 200 from `POST /v1/systemone`.

# Fields
- `model::String`: the **versioned** id that answered (e.g. `"jev-1.13.0"`),
  which may differ from the alias the request named.
- `answers::Dict{String,SystemOneAnswer}`: keyed by question name. The key is
  the contract — the wire's key order is not.
- `usage::TokenUsage`: `prompt_tokens` = `input_tokens`, `completion_tokens` =
  `output_tokens`, `total_tokens` their sum. Only input tokens are billed.
- `request_id::Union{Nothing,String}`: the `x-typesafe-request-id` response
  header, the id to quote in a support report. It travels in the header only —
  the body has no id field.
- `raw::Dict{String,Any}`: the unparsed JSON body.
"""
struct SystemOneResponse
    model::String
    answers::Dict{String,SystemOneAnswer}
    usage::TokenUsage
    request_id::Union{Nothing,String}
    raw::Dict{String,Any}
end

"Successful [`ask`](@ref) result wrapping a [`SystemOneResponse`](@ref)."
struct SystemOneSuccess <: LLMRequestResponse
    response::SystemOneResponse
end

"""
    SystemOneFailure

A TypeSafe call that reached the service and came back non-2xx.

# Fields
- `response::String`: the raw body, kept verbatim.
- `status::Int`: the HTTP status. 401 (invalid key) and 403 (no key) are both
  auth failures; 400 and 422 are request problems; 408, 429, 500, 502, 503, 504
  and 529 are the retryable band.
- `request_id::Union{Nothing,String}`: `x-typesafe-request-id`, present on error
  responses too.
- `error_type::Union{Nothing,String}`: `detail.error_type` when the body uses the
  object shape (`"authentication_error"`, `"api_usage_error"`); `nothing` for the
  string and validation-list shapes.
- `message::String`: the human-readable message extracted from whichever of the
  three `detail` shapes the service used.
- `retry_after::Union{Nothing,Float64}`: how long the service asked the caller to
  wait, in seconds, read off the **final** response — the one being returned,
  once the retry seam has spent `max_attempts`. `retry-after-ms` wins over
  `retry-after` when both arrive, and an HTTP-date becomes a delay from now
  floored at `0.0`. The seam's own backoff between attempts reads `Retry-After`
  alone, so on a response that sends only the millisecond header this field is the
  finer figure of the two. `nothing` when no usable hint was sent, which is every
  status that is not rate-limiting and, as the service documents the header as
  optional, some that are.
"""
@kwdef struct SystemOneFailure <: LLMRequestResponse
    response::String
    status::Int
    request_id::Union{Nothing,String} = nothing
    error_type::Union{Nothing,String} = nothing
    message::String = ""
    retry_after::Union{Nothing,Float64} = nothing
end

"""
    SystemOneCallError

A TypeSafe call that never produced an HTTP response: a timeout, a transport
failure, a missing `TYPESAFE_API_KEY`, or a 200 whose body could not be decoded
into answers. `status` is filled in only when the underlying exception carried one.
"""
@kwdef struct SystemOneCallError <: LLMRequestResponse
    error::String
    status::Union{Int,Nothing} = nothing
end

"""
    SystemOneError(result)

Thrown by [`answers`](@ref), [`answer`](@ref) and `getindex` when the result is
a [`SystemOneFailure`](@ref) or [`SystemOneCallError`](@ref).

A failed call has no answers, and returning an empty map instead would let
`r["is_unsafe"].noul > 0.9` read as "safe" on a call that never happened. Check
`issuccess` first, or handle the throw. `showerror` prints the status, the
extracted message and the `retry_after` hint when the service sent one — never
the request or the API key.
"""
struct SystemOneError <: Exception
    result::Union{SystemOneFailure,SystemOneCallError}
end

function Base.showerror(io::IO, e::SystemOneError)
    r = e.result
    if r isa SystemOneFailure
        print(io, "SystemOneError: the System One call failed with HTTP ", r.status)
        isempty(r.message) || print(io, " — ", r.message)
        isnothing(r.retry_after) ||
            print(io, " (the service asked to retry after ", r.retry_after, "s)")
    else
        print(io, "SystemOneError: the System One call did not complete (", r.error, ")")
    end
end

# ─── Decoding ────────────────────────────────────────────────────────────────
# Permissive on read (unknown fields ignored, an unknown answer type preserved),
# but a REQUIRED field that is absent is malformed provider output, not a
# defaultable zero: it throws, and `ask` turns the throw into a
# SystemOneCallError so no success is ever built from a broken payload.

function _answer_field(d::AbstractDict, key::AbstractString, name::AbstractString)
    haskey(d, key) || throw(ArgumentError(
        "System One answer $(repr(name)) has no \"$key\" field"))
    d[key]
end

function _answer_number(d::AbstractDict, key::AbstractString, name::AbstractString)::Float64
    v = _answer_field(d, key, name)
    v isa Real || throw(ArgumentError(
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

function _level_key(k, name::AbstractString)::Int
    k isa Integer && return Int(k)
    s = string(k)
    lvl = tryparse(Int, s)
    isnothing(lvl) && throw(ArgumentError(
        "System One score answer $(repr(name)) has the non-numeric level key $(repr(s)); " *
        "levels are 0-based integers rendered as decimal strings"))
    lvl
end

function _probability(v, name::AbstractString)::Float64
    v isa Real || throw(ArgumentError(
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

function _level_probabilities(d::AbstractDict, name::AbstractString)::Dict{Int,Float64}
    out = Dict{Int,Float64}()
    for (k, v) in _answer_map(d, "probabilities", name)
        out[_level_key(k, name)] = _probability(v, name)
    end
    out
end

function _legend(d::AbstractDict, name::AbstractString)::Dict{Int,Any}
    out = Dict{Int,Any}()
    for (k, v) in _answer_map(d, "legend", name)
        out[_level_key(k, name)] = v
    end
    out
end

function _decode_answer(name::AbstractString, d::AbstractDict)::SystemOneAnswer
    raw = Dict{String,Any}(string(k) => v for (k, v) in d)
    t = get(d, "type", nothing)
    kind = t isa AbstractString ? String(t) : string(t)
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

# The spec makes both usage fields required; the official SDKs relax them, so a
# missing one reads as 0 rather than failing an otherwise complete evaluation.
function _typesafe_usage(d::AbstractDict)::TokenUsage
    u = get(d, "usage", nothing)
    u isa AbstractDict || return TokenUsage()
    input = get(u, "input_tokens", 0)
    output = get(u, "output_tokens", 0)
    i = input isa Real ? round(Int, input) : 0
    o = output isa Real ? round(Int, output) : 0
    TokenUsage(prompt_tokens=i, completion_tokens=o, total_tokens=i + o)
end

function _decode_systemone(body::AbstractString, request_id::Union{Nothing,String})::SystemOneResponse
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
    model = get(raw, "model", "")
    SystemOneResponse(model isa AbstractString ? String(model) : string(model),
                      out, _typesafe_usage(raw), request_id, raw)
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
    catch
        nothing
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
function _detail_message(v::AbstractDict)::Union{Nothing,String}
    m = get(v, "message", nothing)
    m isa AbstractString ? String(m) : nothing
end
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
A body that is not JSON, or that carries no recognisable message, is returned as
raw text truncated to 200 characters.
"""
function _typesafe_error_message(body::AbstractString)::String
    parsed = _typesafe_parse_body(body)
    parsed isa AbstractDict || return _truncate_body(body)
    for key in ("detail", "error", "message")
        haskey(parsed, key) || continue
        m = _detail_message(parsed[key])
        isnothing(m) || return m
    end
    _truncate_body(body)
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

_typesafe_call_error(e)::SystemOneCallError =
    SystemOneCallError(error=_error_text(e), status=(hasproperty(e, :status) ? e.status : nothing))

# ─── Verbs ───────────────────────────────────────────────────────────────────

"""
    ask(request::SystemOneRequest; service=TYPESAFEServiceEndpoint, config=nothing)
    ask(state, questions...; model=default_typesafe_model(), service=TYPESAFEServiceEndpoint, config=nothing)

Evaluate every question against `state` in one `POST /v1/systemone` call and
return exactly one of [`SystemOneSuccess`](@ref), [`SystemOneFailure`](@ref) (a
non-2xx response) or [`SystemOneCallError`](@ref) (no response, or a 200 whose
body was not a usable set of answers).

`questions` takes the same forms as [`SystemOneRequest`](@ref): `name => question`
pairs, a `NamedTuple` or `AbstractDict` of them, or bare questions auto-named
`"q1"`, `"q2"`, …

```julia
r = ask("Help! My payouts have been failing for 3 days.",
        "department" => choice("Which team should handle this ticket?",
                               (billing="Payments, invoicing, payouts, refunds",
                                technical="Bugs, outages, integrations",
                                sales="Pricing, upgrades, new accounts")),
        "urgency" => score("How urgent is this ticket?",
                           ["Can wait", "Needs attention this week", "Needs attention today"]))
issuccess(r) && println(r["department"].choice, " ", r["urgency"].score)
```

Pass `config::Union{Nothing,RequestConfig}` to override the timeout and retry
budget for this call. The request rides the package's shared retry seam, so
`max_attempts` applies: a retryable status (408, 429, 500, 502, 503, 504, 529)
or a transport failure is retried within `total_deadline`, honouring
`Retry-After`. Other statuses — 400, 401, 403, 404, 422 — are returned as they
came, because a second identical request cannot fix them.
"""
function ask(request::SystemOneRequest; service::ServiceEndpointSpec=TYPESAFEServiceEndpoint,
             config::Union{Nothing,RequestConfig}=nothing)
    validate_capability(service, :system_one, "TypeSafe System One API")
    cfg = _resolve_config(config); t0 = time_ns()
    try
        body = JSON.json(request)
        resp = _http_with_retries(cfg, t0, "POST", _api_base_url(service) * SYSTEMONE_PATH,
                                  auth_header(service), body)
        resp.status == 200 ?
            SystemOneSuccess(_decode_systemone(String(resp.body), _typesafe_request_id(resp))) :
            _typesafe_failure(resp)
    catch e
        e isa InterruptException && rethrow()
        _typesafe_call_error(e)
    end
end

function ask(state, question, questions...;
             model::AbstractString=default_typesafe_model(),
             service::ServiceEndpointSpec=TYPESAFEServiceEndpoint,
             config::Union{Nothing,RequestConfig}=nothing)
    ask(SystemOneRequest(state, _normalize_varargs((question, questions...)), model);
        service, config)
end

# ─── Accessors ───────────────────────────────────────────────────────────────

"""
    answers(result) -> Dict{String,SystemOneAnswer}

The answers of a [`SystemOneSuccess`](@ref) (or a [`SystemOneResponse`](@ref)),
keyed by question name.

A [`SystemOneFailure`](@ref) or [`SystemOneCallError`](@ref) **throws**
[`SystemOneError`](@ref): a call that produced no answers must not read as a
call that answered with none. [`answer`](@ref), `getindex`, `haskey` and `keys`
throw the same error on those two results — a call that never answered has no
key set to query either.
"""
answers(r::SystemOneResponse)::Dict{String,SystemOneAnswer} = r.answers
answers(r::SystemOneSuccess)::Dict{String,SystemOneAnswer} = r.response.answers
answers(r::SystemOneFailure) = throw(SystemOneError(r))
answers(r::SystemOneCallError) = throw(SystemOneError(r))

"""
    answer(result, name) -> SystemOneAnswer

The answer to one question by name (a `String` or `Symbol`). A name that was not
asked throws a `KeyError`; a failed call throws [`SystemOneError`](@ref), as
[`answers`](@ref) does.
"""
answer(r::Union{SystemOneSuccess,SystemOneResponse}, name::AbstractString) = answers(r)[String(name)]
answer(r::Union{SystemOneSuccess,SystemOneResponse}, name::Symbol) = answers(r)[String(name)]
answer(r::Union{SystemOneFailure,SystemOneCallError}, ::Union{AbstractString,Symbol}) =
    throw(SystemOneError(r))

Base.getindex(r::Union{SystemOneSuccess,SystemOneResponse}, name::Union{AbstractString,Symbol}) =
    answer(r, name)
Base.getindex(r::Union{SystemOneFailure,SystemOneCallError}, ::Union{AbstractString,Symbol}) =
    throw(SystemOneError(r))
Base.haskey(r::Union{SystemOneSuccess,SystemOneResponse}, name::Union{AbstractString,Symbol}) =
    haskey(answers(r), String(name))
Base.haskey(r::Union{SystemOneFailure,SystemOneCallError}, ::Union{AbstractString,Symbol}) =
    throw(SystemOneError(r))
Base.keys(r::Union{SystemOneSuccess,SystemOneResponse}) = keys(answers(r))
Base.keys(r::Union{SystemOneFailure,SystemOneCallError}) = throw(SystemOneError(r))

# ─── Accounting ──────────────────────────────────────────────────────────────
# These live here rather than in accounting.jl because that file is included
# first; the price rows themselves are registered there, in DEFAULT_PRICING.

token_usage(r::SystemOneSuccess)::TokenUsage = r.response.usage
token_usage(::SystemOneFailure)::TokenUsage = TokenUsage()
token_usage(::SystemOneCallError)::TokenUsage = TokenUsage()

"""
    estimated_cost(result::SystemOneSuccess; model=nothing, pricing=DEFAULT_PRICING) -> Float64

Estimate the USD cost of one System One evaluation. Only input tokens are
billed — output tokens are currently free, which the shipped `jev-*` price rows
encode as an output rate of `0.0`. The model is taken from the response (the
versioned id that answered) unless `model` overrides it. An id with a dated snapshot
suffix uses its base id's row; a versioned `jev-X.Y.Z` id without its own row is
priced at the `jev-latest` row (assuming a later Jev keeps the price TypeSafe lists
for Jev 1.13); any other unpriced name returns `0.0`.
"""
function estimated_cost(result::SystemOneSuccess; model::Union{String,Nothing}=nothing,
                        pricing::Dict{String,PriceRow}=DEFAULT_PRICING)::Float64
    u = token_usage(result)
    rates = _price_row(pricing, isnothing(model) ? result.response.model : model)
    isnothing(rates) && return 0.0
    cached = min(u.cached_tokens, u.prompt_tokens)
    fresh = u.prompt_tokens - cached
    fresh * rates.input + cached * rates.cached_input + u.completion_tokens * rates.output
end

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

"""
    list_models(; service=TYPESAFEServiceEndpoint, config=nothing)

List the models and aliases the authenticated account may name in
[`SystemOneRequest`](@ref)`.model` (`GET /v1/models`). Returns
[`TypeSafeModelsSuccess`](@ref), [`SystemOneFailure`](@ref), or
[`SystemOneCallError`](@ref).

Pass `config::Union{Nothing,RequestConfig}` to override the timeout and retry
budget; like [`ask`](@ref), this rides the shared retry seam, so `max_attempts`
applies to the retryable statuses.
"""
function list_models(; service::ServiceEndpointSpec=TYPESAFEServiceEndpoint,
                     config::Union{Nothing,RequestConfig}=nothing)
    validate_capability(service, :models, "TypeSafe models listing")
    cfg = _resolve_config(config); t0 = time_ns()
    try
        resp = _http_with_retries(cfg, t0, "GET", _api_base_url(service) * TYPESAFE_MODELS_PATH,
                                  auth_header(service))
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
        e isa InterruptException && rethrow()
        _typesafe_call_error(e)
    end
end
