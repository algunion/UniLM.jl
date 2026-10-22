# ============================================================================
# System One — typed judgments instead of generated text, from any backend.
#
# A System One model reads a piece of state and answers questions the caller
# enumerated with a distribution over the outcomes the caller named (a Choice
# option, a Score level, a yes/no probability, a box), so the answer is already
# machine-readable: there is no text to parse.
#
# This file is the backend-neutral surface: the question and answer types, the
# request, the verbs, the checks every answer passes, the ambient backend, and
# the protocol seam a backend implements (`system_one_protocol`,
# `encode_system_one`, `decode_system_one`). TypeSafe's wrapper is typesafe.jl.
# Types carry only the invariants every backend shares; a provider's own limits
# live in its wrapper and are checked before anything is sent.
# ============================================================================

# ─── Questions ───────────────────────────────────────────────────────────────

"""
    SystemOneEntry

What may stand wherever a piece of guidance appears (`instructions`, a Choice
option description, a Score level, a Noul criterion): a string, a JSON object,
a JSON array, or an [`ImageAttachment`](@ref) — an image is carried only by a
backend whose wrapper sends images, and refused before any request by the
others. Numbers and booleans are not entries: the model reads words, and a bare
number carries none. No backend sends an image's detail level or a
[`DocumentAttachment`](@ref): either, anywhere in a request, is an
`ArgumentError` before any request.
"""
const SystemOneEntry = Union{AbstractString, AbstractDict, AbstractVector, NamedTuple, ImageAttachment}

"""
    SystemOneQuestion

Abstract supertype of the questions a System One request asks:
[`ChoiceQuestion`](@ref) (pick one of the named options), [`ScoreQuestion`](@ref)
(place the state on an ordered rubric), [`NoulQuestion`](@ref) (the probability
that a statement is true), [`BoxQuestion`](@ref) (where something is in an
image), and [`RawQuestion`](@ref), the hatch for a question kind this version
does not know. Build them with [`choice`](@ref), [`score`](@ref),
[`noul`](@ref) and [`box`](@ref).

The types hold only what every backend requires; how many options or levels a
backend accepts is its wrapper's rule, checked when the request is sent.
"""
abstract type SystemOneQuestion end

_entry(v) = isnothing(v) || v isa SystemOneEntry ? v : throw(ArgumentError(
    "a System One entry must be a string, object, array or image (or `nothing`); got $(typeof(v))"))

_score_level(v) = isnothing(v) ?
    throw(ArgumentError("Score levels may not be `nothing`; every level needs a description")) :
    _entry(v)

# The invariants of each kind, shared by the constructors and by the check every request
# passes again when it is sent: the containers a question holds are mutable, so a question
# valid when built can be emptied or given an invalid entry before it goes out.
function _check_choice_criteria(criteria::JSON.Object{String,Any})::Nothing
    isempty(criteria) && throw(ArgumentError("a Choice question needs at least one option"))
    for (name, description) in criteria
        isempty(strip(name)) && throw(ArgumentError("Choice option names must not be blank; got $(repr(name))"))
        _entry(description)
    end
    nothing
end

function _check_score_levels(levels::Vector{Any})::Nothing
    isempty(levels) && throw(ArgumentError("a Score question needs at least one level"))
    foreach(_score_level, levels)
    nothing
end

"""
    ChoiceQuestion(instructions, criteria::JSON.Object{String,Any})

One question that picks a single named option. `criteria` maps each option name
to its description, in insertion order (the order is sent on the wire; the
answer's `probabilities` map is unordered). A description of `nothing` tells the
model to interpret the option by its name alone.

Constructed through [`choice`](@ref), which normalizes the many ways to spell a
criteria map. At least one option is required and option names must not be
blank; there is no upper bound here, because backends differ — TypeSafe accepts
at most 255 options, a limit its wrapper checks before the request is sent.
"""
struct ChoiceQuestion <: SystemOneQuestion
    instructions::Union{Nothing,SystemOneEntry}
    criteria::JSON.Object{String,Any}
    function ChoiceQuestion(instructions, criteria::JSON.Object{String,Any})
        # Checked here and not only in the normalizer: this constructor is public, an
        # option the model cannot name is not a describable outcome, and a description
        # must be an entry exactly as `choice` requires.
        _check_choice_criteria(criteria)
        new(_entry(instructions), criteria)
    end
end

"""
    ScoreQuestion(instructions, criteria::Vector{Any})

One question that places the state on an ordered rubric. `criteria` lists the
levels lowest first; `criteria[1]` is level `0` on the 0-based scale the answer
reports, so a rubric of three levels yields a `score` in `0.0 … 2.0`.

Constructed through [`score`](@ref). At least one level is required and a level
may not be `nothing`: an unnamed level is a rung the model cannot recognise.
There is no upper bound here — TypeSafe accepts at most 10 levels, a limit its
wrapper checks before the request is sent.
"""
struct ScoreQuestion <: SystemOneQuestion
    instructions::Union{Nothing,SystemOneEntry}
    criteria::Vector{Any}
    function ScoreQuestion(instructions, criteria::Vector{Any})
        levels = Any[v for v in criteria]      # its own copy: the caller's vector may change
        _check_score_levels(levels)
        new(_entry(instructions), levels)
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

_noul_told(instructions, criteria::Union{Nothing,NoulCriteria})::Bool =
    !isnothing(instructions) || (!isnothing(criteria) && (!isnothing(criteria.yes) || !isnothing(criteria.no)))

"""
    NoulQuestion(instructions, criteria)

One question answered by a single probability in `0 … 1` — the model's belief
that the statement holds. There is no confidence and no distribution: the value
*is* the distribution, and `0.5` is maximal uncertainty.

Constructed through [`noul`](@ref). A noul carrying neither `instructions` nor a
criterion asks nothing, so that is an `ArgumentError`.
"""
struct NoulQuestion <: SystemOneQuestion
    instructions::Union{Nothing,SystemOneEntry}
    criteria::Union{Nothing,NoulCriteria}
    function NoulQuestion(instructions, criteria::Union{Nothing,NoulCriteria})
        _noul_told(instructions, criteria) || throw(ArgumentError(
            "a Noul question needs `instructions` or at least one of the `yes`/`no` criteria"))
        new(_entry(instructions), criteria)
    end
end

function _check_box_instructions(instructions)::Nothing
    isnothing(instructions) && throw(ArgumentError(
        "a Box question needs `instructions` naming what to locate: a box with no target is not a question"))
    instructions isa AbstractString && isempty(strip(instructions)) && throw(ArgumentError(
        "a Box question needs `instructions` naming what to locate; got a blank string"))
    _entry(instructions)
    nothing
end

"""
    BoxQuestion(instructions)

One question that locates something in an image of the state: the answer is a
[`BoxAnswer`](@ref), a box whose edges are fractions of the image's height and
width. `instructions` says what to locate and is required — a string that is not
blank, an object, an array or an image; a box with no target is not a question.

Constructed through [`box`](@ref). Not every backend answers Box questions;
those that cannot refuse the request with an `ArgumentError` before it is sent.
"""
struct BoxQuestion <: SystemOneQuestion
    instructions::SystemOneEntry
    function BoxQuestion(instructions)
        _check_box_instructions(instructions)
        new(instructions)
    end
end

_raw_field(k::AbstractString)::String = String(k)
_raw_field(k::Symbol)::String = String(k)
_raw_field(k) = throw(ArgumentError("raw question field names must be a String or a Symbol; got $(typeof(k))"))

function _check_raw_fields(fields::JSON.Object{String,Any})::Nothing
    t = get(fields, "type", nothing)
    t isa AbstractString && !isempty(strip(t)) || throw(ArgumentError(
        "a raw question needs a non-blank String \"type\" naming its kind; got $(repr(t))"))
    nothing
end

# A plain `Dict` iterates in hash order, which changes between Julia versions (the
# string hash changed in 1.13), and the order of what is sent is part of what the model
# reads: its entries are taken in sorted key order, the order JSON.jl writes a plain
# `Dict` in. Any other dictionary keeps its own iteration order.
_dict_pairs(d::Dict) = sort!(collect(d); by = p -> string(first(p)))
_dict_pairs(d::AbstractDict) = d

"""
    RawQuestion(fields::AbstractDict)

A question of a kind this version has no type for, kept as the JSON object
`fields` and sent verbatim by the backends that allow raw questions — TypeSafe,
OpenAI's Decisions API (which adds the question's name as `"name"`) and Ollama
all do; a backend that does not refuses it before the request is sent. `fields`
is copied into a `JSON.Object{String,Any}` with `String` keys — a plain `Dict`
in sorted key order, any other dictionary in its own order — and must hold a
non-blank String `"type"`.

Its answer is accepted whatever kind it is: an answer the decoder recognises
comes back typed, any other as an [`UnknownAnswer`](@ref).

```julia
r = ask(state, "rank" => RawQuestion(JSON.Object("type" => "rank", "items" => ["a", "b"])))
```
"""
struct RawQuestion <: SystemOneQuestion
    fields::JSON.Object{String,Any}
    function RawQuestion(fields::AbstractDict)
        obj = JSON.Object{String,Any}()
        for (k, v) in _dict_pairs(fields)
            name = _raw_field(k)
            haskey(obj, name) && throw(ArgumentError(
                "duplicate raw question field $(repr(name)); the wire is an object, so a repeat would drop one"))
            obj[name] = v
        end
        _check_raw_fields(obj)
        new(obj)
    end
end

_option_name(k::AbstractString)::String = String(k)
_option_name(k::Symbol)::String = String(k)
_option_name(k) = throw(ArgumentError(
    "Choice option names must be a String or a Symbol; got $(typeof(k))"))

function _put_option!(obj::JSON.Object{String,Any}, k, v)
    name = _option_name(k)
    isempty(strip(name)) && throw(ArgumentError("Choice option names must not be blank; got $(repr(name))"))
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
    for (k, v) in _dict_pairs(d); _put_option!(obj, k, v); end
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
a description may be a string, an object, an array, an image, or `nothing`.

**Option order is the order of what you pass**: a `NamedTuple`, a vector of
pairs, and an ordered mapping such as `JSON.Object` keep theirs. A plain `Dict`
has none of its own — its hash order changes between Julia versions — so its
options are taken in sorted name order. Order never changes the answer's meaning
— answers are keyed by name — but it is what the model reads.

How many options a question may offer is the backend's rule, checked when the
request is sent: TypeSafe accepts up to 255.

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

Each level is a string, an object, an array or an image — never `nothing`. How
many levels a rubric may have is the backend's rule, checked when the request is
sent: TypeSafe accepts 1 to 10 and advises at least two, since a one-level
rubric gives the model nothing to place the state against.

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

At least one of `instructions`, `yes`, or `no` must be present: a bare noul
says nothing about what is being asked.

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

"""
    box(instructions) -> BoxQuestion

Build a Box question: where in an image of the state is what `instructions`
names. The answer is a [`BoxAnswer`](@ref) — a [`BoxCoordinates`](@ref) of
fractions of the image's height and width, with a `confidence` when the backend
reports one. `instructions` is required: a string that is not blank, an object,
an array or an image.

```julia
box("the dent on the front bumper")
```
"""
box(instructions) = BoxQuestion(instructions)

# ─── The System One JSON form ────────────────────────────────────────────────
# The object form the System One wire is written in (TypeSafe's published wire, which
# other servers of the same route speak too). `instructions` is written only when
# present: a backend may distinguish an absent key from an explicit null, and an ordered
# JSON.Object keeps Choice options in the order they were given. A Box question always
# has instructions and never criteria; a backend that cannot answer one refuses it first.

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

JSON.lower(q::BoxQuestion) = JSON.Object{String,Any}("type" => "box", "instructions" => q.instructions)

JSON.lower(q::RawQuestion) = q.fields

# ─── Request ─────────────────────────────────────────────────────────────────

"""
    SystemOneState

What `state` may be: a string, a JSON object, a JSON array (a `Tuple` and a
`NamedTuple` serialize as those), or an [`ImageAttachment`](@ref) — images,
here or nested anywhere in the content, are carried only by a backend whose
wrapper sends them, and never with a detail level. A [`DocumentAttachment`](@ref)
is no state, and one nested in the content is an `ArgumentError` before any
request, as is an image with a detail level: no backend sends either. `nothing`
is deliberately absent: a request about nothing is a type error here instead of
a wasted round trip.
"""
const SystemOneState = Union{AbstractString, AbstractDict, AbstractVector, Tuple, NamedTuple, ImageAttachment}

_question_name(k::AbstractString)::String = String(k)
_question_name(k::Symbol)::String = String(k)
_question_name(k) = throw(ArgumentError("question names must be a String or a Symbol; got $(typeof(k))"))

_question_value(q::SystemOneQuestion) = q
_question_value(q) = throw(ArgumentError(
    "a question must be a SystemOneQuestion (built with choice, score, noul or box, or a RawQuestion); " *
    "got $(typeof(q))"))

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
    Pair{String,SystemOneQuestion}[_question_name(k) => _question_value(v) for (k, v) in _dict_pairs(d)]
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

function _check_question_names(questions::Vector{Pair{String,SystemOneQuestion}})::Nothing
    isempty(questions) && throw(ArgumentError("a System One request needs at least one question"))
    seen = Set{String}()
    for (name, _) in questions
        isempty(strip(name)) && throw(ArgumentError("question names must not be blank; got $(repr(name))"))
        name in seen && throw(ArgumentError(
            "duplicate question name $(repr(name)); answers are keyed by name, so names must be unique"))
        push!(seen, name)
    end
    nothing
end

_model_name(::Nothing) = nothing
_model_name(m::AbstractString)::String = isempty(strip(m)) ? throw(ArgumentError(
    "`model` must be a non-empty model name such as \"jev-latest\", or `nothing` for the backend's default")) :
    String(m)
_model_name(m) = throw(ArgumentError("`model` must be a model name or `nothing`; got $(typeof(m))"))

"""
    SystemOneRequest(state, questions; model=nothing, options=(;))

One System One evaluation: every question in `questions` is answered about the
same `state`, in a single round trip. The state is ingested once no matter how
many questions ride along, which is why batching is far cheaper than one call
per question.

# Fields
- `state::SystemOneState`: a string, object, array or image — the content every
  question refers to. A question points into it by naming a key in backticks,
  e.g. "Does \\`refund_policy\\` cover this charge?".
- `questions::Vector{Pair{String,SystemOneQuestion}}`: ordered, with unique
  names that are not blank. Answers come back keyed by these names.
- `model::Union{Nothing,String}`: the model to ask, such as `"jev-latest"` or a
  pinned `"jev-1.13.0"` on TypeSafe; `nothing` leaves it to the backend that
  sends the request and is resolved only then — to the model of the enclosing
  [`with_system_one`](@ref) scope when the call names no `service`, otherwise to
  the backend's default (TypeSafe: `UniLM.default_typesafe_model()`, read at
  send time). Aliases move; pin the version if you have tuned thresholds
  against it, and read [`SystemOneResponse`](@ref)`.model` to see which version
  answered.
- `options::NamedTuple`: provider options, checked by the backend's wrapper when
  the request is sent; keyword arguments of [`ask`](@ref) become these.
  TypeSafe accepts one, `long_context::Bool`; OpenAI's Decisions API
  `prompt_cache_key`, `safety_identifier` and `prompt_cache_options`; Ollama `keep_alive`.

`questions` accepts `name => question` pairs (vector, tuple, `NamedTuple`, or
`AbstractDict`), or bare questions in a vector or tuple, which are auto-named
`"q1"`, `"q2"`, … in order; a plain `Dict` of questions is taken in sorted name
order. A `state` of any other type, mixing bare and named questions, repeating a
name, a blank name, passing none, or passing a blank model is an
`ArgumentError`.
"""
struct SystemOneRequest
    state::SystemOneState
    questions::Vector{Pair{String,SystemOneQuestion}}
    model::Union{Nothing,String}
    options::NamedTuple
    function SystemOneRequest(state, questions, model, options)
        state isa SystemOneState || throw(ArgumentError(
            "a System One `state` must be a string, object, array, tuple, NamedTuple or image; got " *
            (isnothing(state) ? "`nothing` (a request about nothing asks nothing)" : string(typeof(state)))))
        options isa NamedTuple || throw(ArgumentError(
            "System One request `options` must be a NamedTuple; got $(typeof(options))"))
        qs = _normalize_questions(questions)
        _check_question_names(qs)
        new(state, qs, _model_name(model), options)
    end
end

SystemOneRequest(state, questions, model) = SystemOneRequest(state, questions, model, (;))
SystemOneRequest(state, questions; model=nothing, options=(;)) = SystemOneRequest(state, questions, model, options)

# The request in the System One JSON form, with `model` written when there is one:
# `state`, `model`, `questions`, in that order. A wrapper sends it with the model it
# resolved; `JSON.json(request)` writes the request as built.
function _system_one_json(r::SystemOneRequest, model::Union{Nothing,String})::JSON.Object{String,Any}
    o = JSON.Object{String,Any}()
    o["state"] = r.state
    isnothing(model) || (o["model"] = model)
    qs = JSON.Object{String,Any}()
    for (name, q) in r.questions
        qs[name] = q
    end
    o["questions"] = qs
    o
end

JSON.lower(r::SystemOneRequest) = _system_one_json(r, r.model)

# The bytes of a request body. JSON has no NaN or Inf, and JSON.jl throws while writing
# one; that is the caller's content, reported as an ArgumentError before any I/O, never
# as a call error that would read as the service's.
function _system_one_body(x)::String
    try
        JSON.json(x)
    catch e
        e isa InterruptException && rethrow()
        throw(ArgumentError("the System One request cannot be written as JSON, so nothing was sent: " *
                            first(split(sprint(showerror, e), '\n'))))
    end
end

# ─── Images ──────────────────────────────────────────────────────────────────

# Every image a request carries, depth first in document order: the state, then each
# question's instructions and criteria (or levels, or raw fields) — the order the System
# One JSON form writes them in, a plain `Dict` in sorted key order as JSON.jl writes it.
# A wrapper that sends images reads them from here; one that cannot refuses when this is
# not empty.
function _system_one_images(r::SystemOneRequest)::Vector{ImageAttachment}
    out = ImageAttachment[]
    _collect_images!(out, r.state)
    for (_, q) in r.questions
        _question_images!(out, q)
    end
    out
end

# Where a question holds content, in the order the System One JSON form writes it: its
# instructions, then its criteria (a Choice's options, a Score's levels, a Noul's two sides),
# or a raw question's fields.
_question_content(q::Union{ChoiceQuestion,ScoreQuestion}) = (q.instructions, q.criteria)
_question_content(q::NoulQuestion) =
    (c = q.criteria; (q.instructions, isnothing(c) ? nothing : c.yes, isnothing(c) ? nothing : c.no))
_question_content(q::BoxQuestion) = (q.instructions,)
_question_content(q::RawQuestion) = (q.fields,)
_question_content(::SystemOneQuestion) = ()

_question_images!(out::Vector{ImageAttachment}, q::SystemOneQuestion) = _collect_images!(out, _question_content(q))

_collect_images!(out::Vector{ImageAttachment}, x::ImageAttachment) = push!(out, x)
function _collect_images!(out::Vector{ImageAttachment}, x::AbstractDict)
    for (_, v) in _dict_pairs(x)
        _collect_images!(out, v)
    end
    out
end
function _collect_images!(out::Vector{ImageAttachment}, x::Union{AbstractVector,Tuple,NamedTuple})
    for v in x
        _collect_images!(out, v)
    end
    out
end
_collect_images!(out::Vector{ImageAttachment}, _) = out

# Each attachment of any kind in `x`, through the containers `_collect_images!` walks. A mutable
# container is entered once, so a value that holds itself ends this walk instead of recursing
# forever; what the wrapper does with it is unchanged (TypeSafe's nesting rule refuses it).
_each_attachment(f, a::Attachment, ::Base.IdSet{Any}) = (f(a); nothing)
function _each_attachment(f, x::AbstractDict, seen::Base.IdSet{Any})
    _first_visit!(seen, x) || return nothing
    for (_, v) in _dict_pairs(x)
        _each_attachment(f, v, seen)
    end
end
function _each_attachment(f, x::Union{AbstractVector,Tuple,NamedTuple}, seen::Base.IdSet{Any})
    _first_visit!(seen, x) || return nothing
    for v in x
        _each_attachment(f, v, seen)
    end
end
_each_attachment(_, _, ::Base.IdSet{Any}) = nothing
_first_visit!(seen::Base.IdSet{Any}, x)::Bool = !ismutable(x) || (x ∉ seen && (push!(seen, x); true))

# ─── Answers ─────────────────────────────────────────────────────────────────

"""
    SystemOneAnswer

Abstract supertype of the answers a System One evaluation returns:
[`ChoiceAnswer`](@ref), [`ScoreAnswer`](@ref), [`NoulAnswer`](@ref),
[`BoxAnswer`](@ref), [`RefusalAnswer`](@ref) when a backend declined one
question, and [`UnknownAnswer`](@ref) for a `type` this version does not
recognise.
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
entry per option of the question, keyed by option name and summing to 1 within
the backend's rounding. The map is unordered: a backend need not return the
request's option order.

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
`legend[argmax(a.probabilities)]` is that level's description — a string, the
original object/array when the rubric was structured, or the level's own
[`ImageAttachment`](@ref) when it was an image.

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
    BoxCoordinates(top, bottom, left, right)

A box in an image, each edge a fraction of the image's size: `top` and `bottom`
of its height measured from the top edge, `left` and `right` of its width
measured from the left edge, so `(0, 1, 0, 1)` is the whole image. Shown
compactly, edge by edge.
"""
struct BoxCoordinates
    top::Float64
    bottom::Float64
    left::Float64
    right::Float64
end

"""
    BoxAnswer(box, confidence, raw)

The answer to a [`BoxQuestion`](@ref): where the thing the instructions named
is, as a [`BoxCoordinates`](@ref), and the backend's `confidence` in `0 … 1` —
`nothing` when the backend reports none. `raw` is the unparsed JSON answer.
"""
struct BoxAnswer <: SystemOneAnswer
    box::BoxCoordinates
    confidence::Union{Nothing,Float64}
    raw::Dict{String,Any}
end

"""
    RefusalAnswer(raw)

A backend declined to answer one question — OpenAI's Decisions API does this
per question — so there is no value to read; `raw` is the unparsed JSON answer.
It is accepted only from a backend that documents refusals: from any other, a
refusal makes the whole result a [`SystemOneCallError`](@ref).
"""
struct RefusalAnswer <: SystemOneAnswer
    raw::Dict{String,Any}
end

"""
    UnknownAnswer(type, raw)

An answer whose `type` this version does not know, kept verbatim in `raw`
instead of being dropped. A new answer primitive on a backend therefore surfaces
as data you can inspect rather than as a silently missing key — to a
[`RawQuestion`](@ref); for a question of a kind this version does know, an
answer of an unknown type is malformed, and the result is a
[`SystemOneCallError`](@ref).
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
Base.show(io::IO, b::BoxCoordinates) =
    print(io, "BoxCoordinates(top=", b.top, ", bottom=", b.bottom, ", left=", b.left, ", right=", b.right, ")")
Base.show(io::IO, a::BoxAnswer) = print(io, "BoxAnswer(", a.box, ", confidence=", repr(a.confidence), ")")
Base.show(io::IO, a::RefusalAnswer) = print(io, "RefusalAnswer(", length(a.raw), " fields)")
Base.show(io::IO, a::UnknownAnswer) =
    print(io, "UnknownAnswer(", repr(a.type), ", ", length(a.raw), " fields)")

# ─── Results ─────────────────────────────────────────────────────────────────

"""
    SystemOneResponse

A decoded answer to a System One request.

# Fields
- `model::String`: the id that answered as the backend reports it — on TypeSafe
  the **versioned** id (e.g. `"jev-1.13.0"`), which may differ from the alias the
  request named.
- `answers::Dict{String,SystemOneAnswer}`: keyed by question name, one per
  question asked. The key is the contract — the wire's key order is not.
- `usage::TokenUsage`: `prompt_tokens` = input tokens, `completion_tokens` =
  output tokens, `total_tokens` their sum. A usage value that is not a
  non-negative whole number reads as 0, with a warning, and never fails the
  answers.
- `request_id::Union{Nothing,String}`: the id the backend sent with the answer,
  the one to quote in a support report — on TypeSafe the
  `x-typesafe-request-id` response header; `nothing` when it sent none.
- `raw::Dict{String,Any}`: the unparsed JSON body.
- `service::Union{Nothing,ServiceEndpointSpec}`: the backend that produced the
  answers, exactly as the call resolved it (a type or an instance); `nothing`
  for a response built by hand, which the five-argument constructor
  `SystemOneResponse(model, answers, usage, request_id, raw)` makes.
"""
struct SystemOneResponse
    model::String
    answers::Dict{String,SystemOneAnswer}
    usage::TokenUsage
    request_id::Union{Nothing,String}
    raw::Dict{String,Any}
    service::Union{Nothing,ServiceEndpointSpec}
end

SystemOneResponse(model, answers, usage, request_id, raw) =
    SystemOneResponse(model, answers, usage, request_id, raw, nothing)

"Successful [`ask`](@ref) result wrapping a [`SystemOneResponse`](@ref)."
struct SystemOneSuccess <: LLMRequestResponse
    response::SystemOneResponse
end

"""
    SystemOneFailure

A System One call that reached its backend and came back with a status other
than 200 — the status TypeSafe, OpenAI's Decisions API and Ollama answer a
success with.

# Fields
- `response::String`: the raw body, kept verbatim.
- `status::Int`: the HTTP status. On TypeSafe, 401 (invalid key) and 403 (no
  key) are both auth failures; 400, 413 (over a size limit), 415 (an unsupported
  `Content-Encoding`) and 422 are request problems, never retried; 408, 429, 500,
  502–504, 520–524 and 529 are the retryable band.
- `request_id::Union{Nothing,String}`: the backend's request id
  (TypeSafe: `x-typesafe-request-id`), present on error responses too.
- `error_type::Union{Nothing,String}`: the machine-readable error kind when the
  body carries one (TypeSafe: `detail.error_type` of the object shape, such as
  `"authentication_error"` or `"api_usage_error"`; `nothing` for its string and
  validation-list shapes).
- `message::String`: the human-readable message extracted from the body.
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

A System One call that produced no usable answer: a timeout, a transport
failure, a missing API key (`TYPESAFE_API_KEY` on TypeSafe), or a 200 whose body
could not be decoded into answers or whose answers do not fit the request — a
question unanswered or one answered that was not asked, an answer of the wrong
kind, an option that was not offered, probabilities that are not a distribution,
a value outside its range, a Score whose `score` its probabilities contradict or
whose legend renames a level asked, or no model named. `status` is filled in
only when the underlying exception carried one.

`cause` is that underlying exception — a [`UniLMTimeout`](@ref) for a timeout —
and `request_id` is the backend's request id of the 200 that could not be used
(TypeSafe: `x-typesafe-request-id`; `nothing` when no reply arrived).
"""
@kwdef struct SystemOneCallError <: LLMRequestResponse
    error::String
    status::Union{Int,Nothing} = nothing
    request_id::Union{Nothing,String} = nothing
    cause::Union{Nothing,Exception} = nothing
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

# ─── Provider protocol ───────────────────────────────────────────────────────
# A backend is reached through its protocol, a Holy trait of its endpoint: the
# protocol encodes a request into the exchange it sends and decodes the reply, and the
# exchange itself — recording, retries, cancellation, typed results — is shared.

"""
    SystemOneProtocol

Abstract supertype of the System One wire protocols, the trait value
[`system_one_protocol`](@ref UniLM.system_one_protocol) returns for an endpoint.
A protocol is a singleton type; its methods of
[`encode_system_one`](@ref UniLM.encode_system_one) and
[`decode_system_one`](@ref UniLM.decode_system_one) are the whole of a backend's
wire. [`TypeSafeProtocol`](@ref UniLM.TypeSafeProtocol) is the first.

Public extension API (not exported).
"""
abstract type SystemOneProtocol end

_no_parts() = Pair{String,ImageAttachment}[]
_no_headers() = Pair{String,String}[]

"""
    SystemOneWire(method, url, path, body, parts, headers, key_extra[, record_body])
    SystemOneWire(method, url, path, body; parts=[], headers=[], key_extra="", record_body=body)

One System One exchange as a protocol encoded it — what is sent and what its
recording is keyed by. Credentials are not part of it: the endpoint's
`auth_header` is read only when the request really goes out, so a replay needs
no key.

# Fields
- `method::String`, `url::String`: the request line, sent as is.
- `path::String`: the path the recording key uses — never derived from `url`, so
  no host, proxy prefix or query ever moves a key. Must start with `/` and hold
  no `?` or `#`.
- `body::String`: the JSON body exactly as sent (the `request` part when the
  exchange is multipart).
- `parts::Vector{Pair{String,ImageAttachment}}`: images sent as parts after the
  body. Empty: the body is sent as it is. Not empty: the request is
  `multipart/form-data` — first a part named `request` (`application/json`, no
  filename) holding `body`, then one part per `name => image`, in order, whose
  filename is its name and whose content type is the image's. Part names must not
  be blank, be `"request"`, or repeat.
- `headers::Vector{Pair{String,String}}`: extra headers that are not credentials,
  such as a feature gate; sent after the endpoint's `auth_header`.
- `key_extra::String`: appended to what the recording key is computed over, after
  a newline, when it is not empty — the canonical form of the request's options,
  digests of its parts — so requests that differ only there are recorded apart.
- `record_body::String`: the body as a recording is keyed on and writes it —
  `body` itself unless the protocol leaves out fields that change how a request
  is served but never what it is answered (Ollama's `keep_alive`; OpenAI's
  `prompt_cache_key`, `safety_identifier` and `prompt_cache_options`), so requests that differ only
  there replay one recording. What is sent is `body`.

Public extension API (not exported).
"""
struct SystemOneWire
    method::String
    url::String
    path::String
    body::String
    parts::Vector{Pair{String,ImageAttachment}}
    headers::Vector{Pair{String,String}}
    key_extra::String
    record_body::String
    function SystemOneWire(method::AbstractString, url::AbstractString, path::AbstractString, body::AbstractString,
                           parts::AbstractVector, headers::AbstractVector, key_extra::AbstractString,
                           record_body::AbstractString)
        isempty(method) && throw(ArgumentError("a System One wire needs an HTTP method"))
        startswith(path, '/') && !occursin(r"[?#]", path) || throw(ArgumentError(
            "a System One wire's keyed path must start with \"/\" and hold no query or fragment; got $(repr(path))"))
        seen = Set{String}()
        for (name, _) in parts
            (isempty(strip(name)) || name == "request" || name in seen) && throw(ArgumentError(
                "System One part names must not be blank, be \"request\" (the JSON part's name), or repeat; " *
                "got $(repr(name))"))
            push!(seen, name)
        end
        new(method, url, path, body, parts, headers, key_extra, record_body)
    end
end

SystemOneWire(method::AbstractString, url::AbstractString, path::AbstractString, body::AbstractString,
              parts::AbstractVector, headers::AbstractVector, key_extra::AbstractString) =
    SystemOneWire(method, url, path, body, parts, headers, key_extra, body)

SystemOneWire(method, url, path, body; parts=_no_parts(), headers=_no_headers(), key_extra="", record_body=body) =
    SystemOneWire(method, url, path, body, parts, headers, key_extra, record_body)

"""
    system_one_protocol(service) -> SystemOneProtocol

The System One protocol `service` speaks — a Holy trait over endpoints. A backend
adds a method following [`provider_capabilities`](@ref)' convention: on
`::Type{MyEndpoint}` for a singleton endpoint, on the instance for one with
fields. The untyped fallback returns [`TypeSafeProtocol`](@ref UniLM.TypeSafeProtocol)`()`,
so every endpoint that declares `:system_one` and no protocol — a
TypeSafe-compatible gateway or proxy, a test mock — speaks TypeSafe's wire.

Public extension API (not exported).
"""
function system_one_protocol end

"""
    encode_system_one(p::SystemOneProtocol, service, request::SystemOneRequest) -> SystemOneWire

The exchange `request` is sent as on `service`. A protocol checks here, before
any I/O, everything its backend cannot carry — its limits, the question kinds it
does not know, content such as images it cannot send, the options it does not
accept — and throws an `ArgumentError` naming the backend, the rule and the fix;
it resolves `request.model === nothing` to its default. [`ask`](@ref) calls it
outside the exchange's error handling, so its `ArgumentError` reaches the caller
and nothing is sent.

Public extension API (not exported).
"""
encode_system_one(p::SystemOneProtocol, service, ::SystemOneRequest) = throw(ArgumentError(
    "$(nameof(typeof(p))) has no `encode_system_one` method: a System One protocol must define how " *
    "$(_service_name(service)) is sent a request"))

"""
    decode_system_one(p::SystemOneProtocol, service, request::SystemOneRequest, response::HTTP.Response)
        -> Union{SystemOneSuccess,SystemOneFailure}

The typed result of `response`, the reply `service` sent to `request`: a
[`SystemOneSuccess`](@ref) whose answers have passed the checks every backend's
answers pass — one per question asked, of the question's kind, each value in its
range (see [`ask`](@ref)) — or a [`SystemOneFailure`](@ref) for a status other
than 200. A 200 that is not a usable answer throws, and [`ask`](@ref)
returns the throw as a [`SystemOneCallError`](@ref) carrying the reply's request
id.

Public extension API (not exported).
"""
decode_system_one(p::SystemOneProtocol, service, ::SystemOneRequest, ::HTTP.Response) = throw(ArgumentError(
    "$(nameof(typeof(p))) has no `decode_system_one` method: a System One protocol must define how " *
    "$(_service_name(service))'s reply is read"))

# Internal hooks of a protocol, each with a fallback that fails closed. The model a
# request names when it names none: a protocol without a default needs one named.
_system_one_default_model(p::SystemOneProtocol, service)::String = throw(ArgumentError(
    "$(_service_name(service)) has no default System One model: pass `model = …`, or set one for a scope " *
    "with `with_system_one(service; model = …) do … end`"))
# What a backend's recordings are filed under, as a first path segment of the keyed
# path. Empty keeps the path as it is (TypeSafe's recordings); any other backend's
# requests must never replay another's answers, so the fallback is the protocol's name.
_system_one_namespace(p::SystemOneProtocol)::String = String(nameof(typeof(p)))
# The request id a backend sends with a reply, from its headers.
_system_one_request_id(::SystemOneProtocol, ::HTTP.Response)::Union{Nothing,String} = nothing
# Two allowances for how a backend writes its numbers, each a protocol's to set. The
# tolerance: how far probabilities over `n` outcomes may sum from 1 — a backend that
# rounds each value to `d` decimals can be off by n·0.5·10⁻ᵈ in all. The slack: how far
# the chosen option may fall below the most probable one, and a Score's expectation
# outside its levels — a single rounding step, since two rounded values swap order by at
# most that, however many outcomes there are. A backend that normalises in floating point
# drifts by units in the last place; 1e-6 admits that with orders of magnitude to spare,
# and a defect an allowance could hide — a missing, extra or out-of-range probability —
# is caught by the key-set and range checks on their own.
const _FLOAT_PROBABILITY_TOLERANCE = 1.0e-6
_probability_tolerance(::SystemOneProtocol, n::Int)::Float64 = _FLOAT_PROBABILITY_TOLERANCE
_probability_slack(::SystemOneProtocol)::Float64 = _FLOAT_PROBABILITY_TOLERANCE
# Whether a backend may decline single questions (`RefusalAnswer`), and send raw questions.
_allows_refusal(::SystemOneProtocol)::Bool = false
_allows_raw(::SystemOneProtocol)::Bool = false
# The models listing of a backend's protocol, for `list_models`.
_system_one_models(p::SystemOneProtocol, service, ::Union{Nothing,RequestConfig}, ::Union{Nothing,CancelToken}) =
    throw(ArgumentError("listing models is not supported by $(_service_name(service)), whose System One " *
                        "protocol is $(nameof(typeof(p)))"))

# The model a protocol sends `request` with: the request's, or the protocol's default,
# read now — at send time — so a default changed after the request was built applies.
_system_one_model(p::SystemOneProtocol, service, request::SystemOneRequest)::String =
    isnothing(request.model) ? _system_one_default_model(p, service) : request.model

"""
    _options_key(options::NamedTuple) -> String

The canonical form of a request's provider options for its recording key: one
`name=value` line per option, sorted by name, each value written as JSON — so
the same options in any order key alike, and no options add nothing.
"""
_options_key(options::NamedTuple)::String =
    join((string(k, '=', _system_one_body(v)) for (k, v) in sort!(collect(pairs(options)); by=first)), '\n')

# ─── Validation ──────────────────────────────────────────────────────────────

# The invariants every request meets, checked again when it is sent: a question's
# containers are mutable and so is the question list, so a request valid when built can
# reach the wire empty, with a blank or repeated name, or with too few options.
_check_question(q::ChoiceQuestion) = _check_choice_criteria(q.criteria)
_check_question(q::ScoreQuestion) = _check_score_levels(q.criteria)
_check_question(q::NoulQuestion) = _noul_told(q.instructions, q.criteria) ? nothing : throw(ArgumentError(
    "a Noul question needs `instructions` or at least one of the `yes`/`no` criteria"))
_check_question(q::BoxQuestion) = _check_box_instructions(q.instructions)
_check_question(q::RawQuestion) = _check_raw_fields(q.fields)
_check_question(::SystemOneQuestion) = nothing

function _check_request(p::SystemOneProtocol, service, r::SystemOneRequest)::Nothing
    _check_question_names(r.questions)
    for (name, q) in r.questions
        _check_question(q)
        q isa RawQuestion && !_allows_raw(p) && throw(ArgumentError(
            "raw questions are not supported by $(_service_name(service)): $(repr(name)) is a RawQuestion " *
            "of type $(repr(q.fields["type"]))"))
    end
    _check_media(service, r)
end

# No System One wire carries a document or an image's detail level: TypeSafe's assets, the
# Decisions API's `input_image` parts and Ollama's `images` list take an image's bytes alone,
# and none of them a file. Either is refused, naming where it stands, before anything is sent.
function _check_media(service, r::SystemOneRequest)::Nothing
    seen = Base.IdSet{Any}()
    _each_attachment(a -> _check_medium(service, a, "the state"), r.state, seen)
    for (name, q) in r.questions
        _each_attachment(a -> _check_medium(service, a, "the question $(repr(name))"), _question_content(q), seen)
    end
    nothing
end

_check_medium(service, a::DocumentAttachment, place::String) = throw(ArgumentError(
    "$(_service_name(service)) cannot send $(a), which $(place) holds: a System One request carries text and " *
    "images, not documents. Put the document's text in the request, or ask about the document in a Chat on " *
    "an OpenAI-compatible endpoint"))
function _check_medium(service, a::ImageAttachment, place::String)::Nothing
    isnothing(a.detail) || throw(ArgumentError(
        "$(_service_name(service)) sends a System One request's images without a detail level, so $(a) in " *
        "$(place) cannot be sent as asked: build the ImageAttachment without `detail`"))
    nothing
end
_check_medium(_, ::Attachment, ::String) = nothing

# A Box question locates something in an image. Asked about a request that carries none, a
# backend can still answer — TypeSafe returns a box at near-zero confidence — but nothing it
# answers can mean anything, so a wrapper that carries Box questions refuses such a request
# before it is paid for (after its own checks: a wire with no Box at all says so first).
function _check_box_image(r::SystemOneRequest)::Nothing
    box = findfirst(((_, q),) -> q isa BoxQuestion, r.questions)
    isnothing(box) || !isempty(_system_one_images(r)) || throw(ArgumentError(
        "the Box question $(repr(first(r.questions[box]))) locates something in an image, and this request " *
        "carries none: put an `ImageAttachment` in the state"))
    nothing
end

_unit(x::Real)::Bool = 0 <= x <= 1      # false for NaN and ±Inf

function _listed(xs)::String
    s::Vector{String} = sort!(collect(String, xs))
    shown = join(String[repr(x) for x in first(s, 5)], ", ")
    length(s) > 5 ? string(shown, " and ", length(s) - 5, " more") : shown
end

function _check_distribution(name::String, probabilities::AbstractDict, tol::Float64)::Nothing
    for (k, v) in probabilities
        _unit(v) || throw(ArgumentError(
            "System One answer $(repr(name)) has the probability $(v), outside 0 … 1, for " *
            (k isa Integer ? "level $(k)" : "option $(repr(k))")))
    end
    s = sum(values(probabilities))
    abs(s - 1) <= tol || throw(ArgumentError(
        "System One answer $(repr(name)) has probabilities summing to $(s), not 1 (tolerance $(tol))"))
    nothing
end

function _check_answers(p::SystemOneProtocol, request::SystemOneRequest, response::SystemOneResponse)::Nothing
    # The id that answered is what a tuned threshold and a price are about: a blank one
    # names no model.
    isempty(strip(response.model)) && throw(ArgumentError(
        "the System One reply names no model: its \"model\" is $(repr(response.model))"))
    got = response.answers
    asked = request.questions
    missed = String[name for (name, _) in asked if !haskey(got, name)]
    isempty(missed) || throw(ArgumentError(
        "the System One reply leaves $(_listed(missed)) unanswered"))
    if length(got) != length(asked)       # every asked name is answered, so the rest were never asked
        stray = setdiff(keys(got), (name for (name, _) in asked))
        throw(ArgumentError("the System One reply answers $(_listed(stray)), which was not asked"))
    end
    for (name, q) in asked
        _check_answer(p, name, q, got[name])
    end
    nothing
end

function _check_answer(p::SystemOneProtocol, name::String, q::SystemOneQuestion, a::SystemOneAnswer)::Nothing
    if a isa RefusalAnswer
        _allows_refusal(p) || throw(ArgumentError(
            "System One answer $(repr(name)) is a refusal, which $(nameof(typeof(p))) does not send"))
        return nothing
    end
    q isa RawQuestion && return nothing   # a kind this version does not know: any answer is its answer
    _check_kind(p, name, q, a)
end

_kind(::ChoiceQuestion) = "Choice"
_kind(::ScoreQuestion) = "Score"
_kind(::NoulQuestion) = "Noul"
_kind(::BoxQuestion) = "Box"
_kind(q::SystemOneQuestion) = string(nameof(typeof(q)))

_answer_kind(a::UnknownAnswer) = "an answer of the unknown type $(repr(a.type))"
_answer_kind(a::SystemOneAnswer) = "a $(nameof(typeof(a)))"

_check_kind(::SystemOneProtocol, name::String, q::SystemOneQuestion, a::SystemOneAnswer) = throw(ArgumentError(
    "System One answer $(repr(name)) is $(_answer_kind(a)), but $(repr(name)) asks a $(_kind(q)) question"))

function _check_kind(p::SystemOneProtocol, name::String, q::ChoiceQuestion, a::ChoiceAnswer)::Nothing
    # A JSON.Object finds a key by a linear scan, and the names are looked up once per
    # answered option — quadratic over 255 options — so they go into a Set first.
    offered = Set{String}(keys(q.criteria))
    a.choice in offered || throw(ArgumentError(
        "System One answer $(repr(name)) chose $(repr(a.choice)), which was not one of the options offered"))
    unoffered = String[k for k in keys(a.probabilities) if !(k in offered)]
    isempty(unoffered) || throw(ArgumentError(
        "System One answer $(repr(name)) gives probabilities to $(_listed(unoffered)), which was not offered"))
    unscored = String[k for k in offered if !haskey(a.probabilities, k)]
    isempty(unscored) || throw(ArgumentError(
        "System One answer $(repr(name)) gives no probability to the offered $(_listed(unscored))"))
    tol = _probability_tolerance(p, length(offered))
    _check_distribution(name, a.probabilities, tol)
    a.probabilities[a.choice] >= maximum(values(a.probabilities)) - _probability_slack(p) || throw(ArgumentError(
        "System One answer $(repr(name)) chose $(repr(a.choice)) at probability $(a.probabilities[a.choice]), " *
        "below another option's $(maximum(values(a.probabilities)))"))
    _unit(a.confidence) || throw(ArgumentError(
        "System One answer $(repr(name)) has the confidence $(a.confidence), outside 0 … 1"))
    nothing
end

function _check_kind(p::SystemOneProtocol, name::String, q::ScoreQuestion, a::ScoreAnswer)::Nothing
    n = length(q.criteria)
    for (what, levels) in (("probabilities", keys(a.probabilities)), ("legend", keys(a.legend)))
        length(levels) == n && all(l -> 0 <= l < n, levels) || throw(ArgumentError(
            "System One answer $(repr(name)) has the $(what) levels $(sort!(collect(levels))), not the " *
            "$(n) levels 0 to $(n - 1) of the rubric asked"))
    end
    tol = _probability_tolerance(p, n)
    _check_distribution(name, a.probabilities, tol)
    slack = _probability_slack(p)
    -slack <= a.score <= n - 1 + slack || throw(ArgumentError(
        "System One answer $(repr(name)) has the score $(a.score), outside 0 … $(n - 1)"))
    # The score is the expected level. Probabilities that err by the sum's tolerance in all
    # move that expectation by at most n − 1 times it, and a rounded score is off by one step.
    expected = sum(l * pr for (l, pr) in a.probabilities)
    allowed = (n - 1) * tol + slack
    abs(a.score - expected) <= allowed || throw(ArgumentError(
        "System One answer $(repr(name)) has the score $(a.score), but its probabilities put the expected level at " *
        "$(expected) (allowed difference $(allowed))"))
    # A level asked as a string comes back in the legend as it was asked. A level that is an
    # object or an array is not compared: a backend may write it back re-serialized (another
    # key order, another number format). Nor is an image: TypeSafe echoes its reference.
    for (i, level) in enumerate(q.criteria)
        level isa AbstractString && a.legend[i-1] != level && throw(ArgumentError(
            "System One answer $(repr(name)) describes level $(i - 1) as $(repr(a.legend[i-1])), not as the level " *
            "asked, $(repr(level))"))
    end
    _unit(a.confidence) || throw(ArgumentError(
        "System One answer $(repr(name)) has the confidence $(a.confidence), outside 0 … 1"))
    nothing
end

_check_kind(::SystemOneProtocol, name::String, ::NoulQuestion, a::NoulAnswer)::Nothing = _unit(a.noul) ? nothing :
    throw(ArgumentError("System One answer $(repr(name)) has the probability $(a.noul), outside 0 … 1"))

function _check_kind(::SystemOneProtocol, name::String, ::BoxQuestion, a::BoxAnswer)::Nothing
    b = a.box
    for (edge, v) in (("top", b.top), ("bottom", b.bottom), ("left", b.left), ("right", b.right))
        _unit(v) || throw(ArgumentError(
            "System One answer $(repr(name)) has the $(edge) edge at $(v), outside 0 … 1 of the image"))
    end
    # Edges are measured from the top and from the left, so a box's top lies above its bottom
    # and its left before its right; a box turned inside out is not a location.
    b.top <= b.bottom && b.left <= b.right || throw(ArgumentError(
        "System One answer $(repr(name)) has a box turned inside out (top $(b.top), bottom $(b.bottom), " *
        "left $(b.left), right $(b.right))"))
    isnothing(a.confidence) || _unit(a.confidence) || throw(ArgumentError(
        "System One answer $(repr(name)) has the confidence $(a.confidence), outside 0 … 1"))
    nothing
end

# ─── Ambient backend ─────────────────────────────────────────────────────────

# The backend, the model and the provider options a System One verb uses when the call
# names no service.
struct _SystemOneTarget
    service::ServiceEndpointSpec
    model::Union{Nothing,String}
    options::NamedTuple
end

const _SYSTEM_ONE_TARGET = ScopedValue{Union{Nothing,_SystemOneTarget}}(nothing)

# An endpoint must declare `:system_one` to be sent a System One request; one that
# declares nothing has no basis for it either.
function _check_system_one_service(service)::Nothing
    _capability_declared(service) || throw(ArgumentError(
        "System One API is not supported by $(_service_name(service)): it declares no capabilities " *
        "(a System One backend declares :system_one in `provider_capabilities`)"))
    validate_capability(service, :system_one, "System One API")
end

"""
    with_system_one(f, service; model=nothing, options...)

Run `f()` with `service` as the System One backend of every verb inside it that
names none — [`ask`](@ref), [`list_models`](@ref), [`nl_dispatch`](@ref),
[`nl_classify`](@ref) and [`@branch`](@ref) — `model`, when given, as the
model of those calls that name none, and any other keyword as
the provider options of those calls that name none (see [`ask`](@ref); checked
by the backend's wrapper when each request is sent); return `f()`'s value.
`service` must declare `:system_one` and `model` must not be blank, or this is
an `ArgumentError` before `f` runs.

A call resolves its backend, model and options this way:

- with `service`: that backend, and `model` and options if the call names them,
  else the backend's default model and no options — the scope's model and
  options belong to the scope's backend and are never applied to another;
- without `service`: the scope's backend, and the call's `model` if it names
  one, else the scope's; the call's options if it names any, else the scope's;
- outside every scope: [`TYPESAFEServiceEndpoint`](@ref) and its default model.

The scope is a `ScopedValue`, so it covers the tasks started inside `f`
(`Threads.@spawn`, `asyncmap`), and scopes nest: the innermost applies.

```julia
with_system_one(TYPESAFEServiceEndpoint; model = "jev-1.13.0", long_context = true) do
    nl_classify(ticket, INTENT)          # asks jev-1.13.0, with TypeSafe's long context
end
```
"""
function with_system_one(f::Function, service; model::Union{Nothing,AbstractString}=nothing, options...)
    service isa ServiceEndpointSpec || throw(ArgumentError(
        "with_system_one takes an endpoint (a ServiceEndpoint type or instance); got $(typeof(service))"))
    _check_system_one_service(service)
    with(f, _SYSTEM_ONE_TARGET => _SystemOneTarget(service, _model_name(model), values(options)))
end

# The backend, model and options of a call: the call's own, then the ambient scope's —
# whose model and options apply only with its own backend — then TypeSafe and its default.
# (No return annotation: a tuple holding a type has the element type `DataType`, not `Type{T}`.)
function _system_one_target(service::Union{Nothing,ServiceEndpointSpec}, model::Union{Nothing,String},
                            options::NamedTuple=(;))
    isnothing(service) || return (service, model, options)
    t = _SYSTEM_ONE_TARGET[]
    isnothing(t) && return (TYPESAFEServiceEndpoint, model, options)
    (t.service, isnothing(model) ? t.model : model, isempty(options) ? t.options : options)
end

# ─── Verbs ───────────────────────────────────────────────────────────────────

# A request ready to send: its backend, that backend's protocol, the request with the
# model and options the call resolved, and the exchange it encodes to. Building one sends
# nothing, and every local refusal happens here.
struct _SystemOneCall
    protocol::SystemOneProtocol
    service::ServiceEndpointSpec
    request::SystemOneRequest
    wire::SystemOneWire
end

# The same before it is encoded: what `ask_each` keeps of each item, so a batch holds no
# more request bodies than it has requests in flight.
const _SystemOneResolved = @NamedTuple{protocol::SystemOneProtocol, service::ServiceEndpointSpec,
                                       request::SystemOneRequest}

function _system_one_resolve(request::SystemOneRequest,
                             service::Union{Nothing,ServiceEndpointSpec})::_SystemOneResolved
    svc, model, options = _system_one_target(service, request.model, request.options)
    _check_system_one_service(svc)
    p = system_one_protocol(svc)::SystemOneProtocol
    req = model === request.model && options === request.options ? request :
          SystemOneRequest(request.state, request.questions, model, options)
    (protocol=p, service=svc, request=req)
end

_system_one_call(r::_SystemOneResolved)::_SystemOneCall =
    _SystemOneCall(r.protocol, r.service, r.request, encode_system_one(r.protocol, r.service, r.request)::SystemOneWire)

_system_one_call(request::SystemOneRequest, service::Union{Nothing,ServiceEndpointSpec})::_SystemOneCall =
    _system_one_call(_system_one_resolve(request, service))

# A multipart body is consumed by the attempt that sends it, so each attempt builds its
# own: the JSON part first, named `request`, then the parts in order.
function _system_one_form(wire::SystemOneWire)::HTTP.Form
    parts = Pair{String,Any}["request" => HTTP.Multipart(nothing, IOBuffer(wire.body), "application/json")]
    for (name, img) in wire.parts
        push!(parts, name => HTTP.Multipart(name, IOBuffer(img.data), img.mime))
    end
    HTTP.Form(parts)
end

# The connection pool of every System One exchange. HTTP.jl's default client keeps two
# idle connections per host, so concurrent requests to one HTTP/1.1 server — a local
# backend, a gateway — close nearly every connection they open: 64 workers in a closed
# loop opened over 16,000 connections within seconds and ran out of ephemeral ports. This
# pool keeps an idle connection for each request in flight (a backend that negotiates
# HTTP/2 multiplexes them over one), for 30 s — long enough for a batch, or a loop, to
# find its connections warm, and well under the idle timeouts of the NATs and load
# balancers on a typical path. HTTP.jl hands a pooled connection to the next request
# without checking that the server has not closed it meanwhile, and that is acceptable
# here: the stale connection fails as the request is written or its reply read, a reset
# the retry seam retries on a fresh connection like any transport failure, and a System
# One request is safe to repeat — it asks for an evaluation, changes nothing, and was
# never read by the server that had closed the connection. (MCP, whose tool calls may
# act on the world, bounds idle reuse to a second instead.) The pool is created by the
# first live request, once per process — never during precompilation, and never by a
# replayed call.
const _SYSTEM_ONE_CLIENT = Base.OncePerProcess{HTTP.Client{Nothing}}() do
    HTTP.Client(; transport=HTTP.Transport(; proxy=HTTP.ProxyFromEnvironment(),
        max_idle_per_host=256, max_idle_total=1024, idle_timeout_ns=30_000_000_000))
end
_system_one_client()::HTTP.Client{Nothing} = _SYSTEM_ONE_CLIENT()

# A backend's transport, through internal hooks of its protocol whose fallbacks keep what
# every backend had before it could bring its own: the RequestConfig its calls use when no
# channel the user controls sets one (`nothing`: the field defaults), the header its
# retries are numbered in, and the requests `ask_each` keeps in flight when the call names
# no `ntasks` — a default, not a limit.
_system_one_default_config(::SystemOneProtocol, service)::Union{Nothing,RequestConfig} = nothing
_system_one_retry_header(::SystemOneProtocol)::Union{Nothing,String} = nothing
_system_one_concurrency(::SystemOneProtocol, service)::Int = 8

# The bounds of a hosted System One service. TypeSafe answers a typical request in about
# 0.3 s and the slowest one measured — 262k tokens of long context — in 5.3 s; OpenAI's
# Decisions API answers in about 0.5 s with a tail near 4 s. 60 s per attempt is over ten
# times the slowest, and 150 s bounds a call with its retries, where the field defaults
# would hold a stalled call for 10 minutes per attempt and 15 in all.
const _HOSTED_SYSTEM_ONE_CONFIG = RequestConfig(request_timeout=60.0, total_deadline=150.0)

# The live exchange. Credentials are read here and only here, so a replay needs none, and
# so is the connection pool, so a replay opens no connection. A multipart request carries
# no JSON Content-Type: HTTP.jl sets the form's own, with its boundary, and a second one
# would corrupt the upload. `hooks` are the retry seam's `hold` and `observe`, through
# which the requests of one `ask_each` back off together.
function _system_one_send(cfg::RequestConfig, t0::UInt64, protocol::SystemOneProtocol, service::ServiceEndpointSpec,
                          wire::SystemOneWire, tok::Union{Nothing,CancelToken}; hooks...)::HTTP.Response
    auth = auth_header(service)
    headers = isempty(wire.headers) ? auth : [auth; wire.headers]
    seam = (; client=_system_one_client(), retry_header=_system_one_retry_header(protocol), hooks...)
    isempty(wire.parts) &&
        return _http_with_retries(cfg, t0, wire.method, wire.url, headers, wire.body; cancel=tok, seam...)
    _http_with_retries(cfg, t0, wire.method, wire.url,
                       filter(h -> lowercase(String(first(h))) != "content-type", headers),
                       _BodyFactory(() -> _system_one_form(wire)); cancel=tok, seam...)
end

# One call's exchange: through any recorded-answers scope (keyed by the protocol's
# namespace, the wire's path, record_body and key_extra) to the live send of its body,
# then the protocol's decoding. A missing or unwritable recording and an interrupt
# propagate; anything else that goes wrong is the call's error, with the reply's request
# id when one arrived. The backend's own default config applies when no channel the user
# controls sets one, and its `total_deadline` runs from `t0`.
function _system_one_exchange(call::_SystemOneCall, config::Union{Nothing,RequestConfig},
                              cancel::Union{Nothing,CancelToken}; t0::UInt64=time_ns(), hooks...)
    (; protocol, service, request, wire) = call
    cfg = _resolve_config(config, _system_one_default_config(protocol, service))
    tok = _resolve_cancel(cancel)
    # Asserted: `protocol` is abstract here, and once a hook has more methods than inference
    # splits over, its result would be inferred as `Any` however each method is annotated.
    ns = _system_one_namespace(protocol)::String
    local resp
    try
        resp = _recorded_exchange(wire.method, isempty(ns) ? wire.path : "/" * ns * wire.path, wire.record_body, t0,
                                  tok; key_extra=wire.key_extra) do
            _system_one_send(cfg, t0, protocol, service, wire, tok; hooks...)
        end
        decode_system_one(protocol, service, request, resp)::Union{SystemOneSuccess,SystemOneFailure}
    catch e
        # A missing recording is not a service failure: it must reach the caller. Nor
        # is a paid answer whose recording could not be written.
        e isa Union{InterruptException,ReplayMissError,RecordingWriteError} && rethrow()
        _callerr(SystemOneCallError, e; error=_error_text(service, e),
                 request_id=@isdefined(resp) ? _system_one_request_id(protocol, resp) : nothing)
    end
end

"""
    ask(request::SystemOneRequest; service=nothing, config=nothing, cancel=nothing)
    ask(state, questions...; model=nothing, service=nothing, config=nothing, cancel=nothing, options...)

Evaluate every question against `state` in one request to a System One backend
and return one of [`SystemOneSuccess`](@ref), [`SystemOneFailure`](@ref) (a
response with a status other than 200) or [`SystemOneCallError`](@ref) (no
response, or a 200 that was not a usable set of answers). Inside a [`with_recorded_answers`](@ref) scope,
a request the scope cannot replay — no recording in `:replay` mode, or a
recording file that cannot be read — throws [`ReplayMissError`](@ref) instead of
returning: a gap in the recordings is not a service failure.

The backend is `service` when given, else the one of the enclosing
[`with_system_one`](@ref) scope, else [`TYPESAFEServiceEndpoint`](@ref). The
model is `model` (or the request's) when given; else, with no `service`, the
scope's; else the backend's default, read when the request is sent (TypeSafe:
`TYPESAFE_DEFAULT_MODEL`, or `"jev-latest"`).

`questions` takes the same forms as [`SystemOneRequest`](@ref): `name => question`
pairs, a `NamedTuple` or `AbstractDict` of them, or bare questions auto-named
`"q1"`, `"q2"`, … Any other keyword is a provider option and becomes the
request's `options`, which the backend's wrapper checks before anything is sent:
TypeSafe accepts `long_context::Bool` (`true` raises its context limits to 262k
tokens), OpenAI's Decisions API `prompt_cache_key` and `safety_identifier`
(strings) and `prompt_cache_options` (a [`PromptCacheOptions`](@ref)), Ollama
`keep_alive` (the seconds the model stays loaded), so any other
keyword — a misspelt one included — is an `ArgumentError` that lists the options
the backend accepts.

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

Malformed input is an `ArgumentError` before anything is sent: a `state` that is
not a string, object, array or image; a value JSON cannot carry (`NaN`, `Inf`);
a request the backend cannot carry — TypeSafe takes at most 255 options per
Choice and 10 levels per Score, PNG and JPEG images of at most 16 MiB with 280
to 1568 pixels on each side (a JPEG ending with its end-of-image marker), 64
levels of nesting in the state, instructions and criteria, and a body of at most
32 MiB; a `service` that does not declare `:system_one`.

Every answer is checked against the request before a success is built: each
question asked is answered once and nothing else is; each answer is of its
question's kind (an answer to a [`RawQuestion`](@ref) may be any); a Choice picks
an offered option that is the most probable and gives a probability to exactly
the options offered; a Score gives one to exactly the levels `0` to `n − 1`, its
`score` lies between them and is the expectation of its probabilities within the
backend's rounding, and its `legend` repeats every level asked as a string;
probabilities lie in `0 … 1` and sum to 1 within the backend's rounding;
confidences, Noul probabilities and box edges lie in `0 … 1`; the response names
the model that answered, by a name that is not blank. A reply that fails any of
these is a `SystemOneCallError` — never a success built from a malformed answer.

Pass `config::Union{Nothing,RequestConfig}` to override the timeout and retry
budget for this call. With none — no `config`, no [`with_request_config`](@ref)
scope, no process default set — the backend's own default applies: on TypeSafe
and OpenAI's Decisions API, 60 s per attempt and 150 s in all. On TypeSafe a
retry carries its number in `X-TypeSafe-Retry-Count`, as the official SDKs send
it. The request rides the package's shared retry seam, so
`max_attempts` applies: a retryable status (408, 429, 500, 502–504, 520–524, 529)
or a transport failure is retried within `total_deadline`, honouring
`Retry-After`. Other statuses — 400, 401, 403, 404, 413, 415, 422 — are returned
as they came, because a second identical request cannot fix them.

Pass `cancel::Union{Nothing,CancelToken}` (default: the ambient [`with_cancel`](@ref)
token) to make the call cancellable: once the token is cancelled it returns a
`SystemOneCallError` whose `cause` is a [`UniLMCancelled`](@ref) — without sending
anything when the token was already cancelled, at once when mid-request (never retried).
"""
function ask(request::SystemOneRequest; service::Union{Nothing,ServiceEndpointSpec}=nothing,
             config::Union{Nothing,RequestConfig}=nothing,
             cancel::Union{Nothing,CancelToken}=nothing)
    _system_one_exchange(_system_one_call(request, service), config, cancel)
end

function ask(state, question, questions...;
             model::Union{Nothing,AbstractString}=nothing,
             service::Union{Nothing,ServiceEndpointSpec}=nothing,
             config::Union{Nothing,RequestConfig}=nothing,
             cancel::Union{Nothing,CancelToken}=nothing,
             options...)
    ask(SystemOneRequest(state, _normalize_varargs((question, questions...)), model, values(options));
        service, config, cancel)
end

"""
    list_models(; service=nothing, config=nothing, cancel=nothing)

List the models and aliases the authenticated account may name in
[`SystemOneRequest`](@ref)`.model` — on TypeSafe (`GET /v1/models`) a
[`TypeSafeModelsSuccess`](@ref), a [`SystemOneFailure`](@ref), or a
[`SystemOneCallError`](@ref). The backend is `service` when given, else the one
of the enclosing [`with_system_one`](@ref) scope, else
[`TYPESAFEServiceEndpoint`](@ref); a backend whose System One protocol has no
listing is an `ArgumentError`. Inside a [`with_recorded_answers`](@ref) scope that
cannot replay the listing it throws [`ReplayMissError`](@ref) instead, as
[`ask`](@ref) does.

Pass `config::Union{Nothing,RequestConfig}` to override the timeout and retry
budget; like [`ask`](@ref), this rides the shared retry seam, so `max_attempts`
applies to the retryable statuses. `cancel::Union{Nothing,CancelToken}` makes the
call cancellable as for [`ask`](@ref): a cancelled call returns a
`SystemOneCallError` whose `cause` is a [`UniLMCancelled`](@ref).

With an [`OllamaEndpoint`](@ref) as `service`, it lists the models installed on that
Ollama server instead (`GET /api/tags`) and returns an [`OllamaSuccess`](@ref) holding a
`Vector{`[`OllamaModel`](@ref)`}`, an [`OllamaFailure`](@ref), or an
[`OllamaCallError`](@ref); `config` and `cancel` apply as above, and the listing is not
recorded by [`with_recorded_answers`](@ref).

```julia
r = list_models(service=OllamaEndpoint())
issuccess(r) && [m.name for m in r.response]   # ["gemma4:e4b", …]
```
"""
list_models(; service::Union{Nothing,ServiceEndpointSpec}=nothing,
            config::Union{Nothing,RequestConfig}=nothing,
            cancel::Union{Nothing,CancelToken}=nothing) =
    _list_models(first(_system_one_target(service, nothing)), config, cancel)

# The listing of the service that serves one: its System One protocol's here, Ollama's
# own in ollama_models.jl.
_list_models(service::ServiceEndpointSpec, config::Union{Nothing,RequestConfig}, cancel::Union{Nothing,CancelToken}) =
    _system_one_models(system_one_protocol(service)::SystemOneProtocol, service, config, cancel)

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

Estimate the USD cost of one System One evaluation from its token usage and the
price row of the model that answered (`result.response.model`), unless `model`
overrides it. TypeSafe bills input tokens only — output tokens are currently
free, which the shipped `jev-*` price rows encode as an output rate of `0.0`. An
id with a dated snapshot suffix uses its base id's row; a versioned `jev-X.Y.Z`
id without its own row is priced at the `jev-latest` row (assuming a later Jev
keeps the price TypeSafe lists for Jev 1.13); any other unpriced name returns
`0.0`. An answer from a local model through an [`OllamaEndpoint`](@ref) costs
`0.0`, without a missing-price warning, unless `model` is passed; a cloud model
reached through it (a tag of `cloud` or ending in `-cloud`) is priced by the name
requested, which an answer whose reply drops the tag keeps.
A backend that bills otherwise is priced its own way: an
answer from OpenAI's Decisions API
([`OpenAIDecisionsProtocol`](@ref UniLM.OpenAIDecisionsProtocol)) costs its input
tokens alone, cached ones included, at the row's input rate — twice that rate
above 272K input tokens.
"""
function estimated_cost(result::SystemOneSuccess; model::Union{String,Nothing}=nothing,
                        pricing::AbstractDict{String,PriceRow}=DEFAULT_PRICING)::Float64
    service = result.response.service
    isnothing(model) && _is_local(service, result.response.model) && return 0.0
    _system_one_cost(isnothing(service) ? nothing : system_one_protocol(service)::SystemOneProtocol,
                     result, model, pricing)
end

# The price of one evaluation on the backend that answered it, by its protocol: input,
# cached input and output tokens at the rates of the model's row, unless the protocol
# bills otherwise. A response built by hand names no backend and is priced this way.
function _system_one_cost(::Union{Nothing,SystemOneProtocol}, result::SystemOneSuccess,
                          model::Union{String,Nothing}, pricing::AbstractDict{String,PriceRow})::Float64
    u = token_usage(result)
    rates = _price_row(pricing, isnothing(model) ? result.response.model : model)
    isnothing(rates) && return 0.0
    cached = min(u.cached_tokens, u.prompt_tokens)
    fresh = u.prompt_tokens - cached
    fresh * rates.input + cached * rates.cached_input + u.completion_tokens * rates.output
end
