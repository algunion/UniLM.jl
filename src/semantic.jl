# ============================================================================
# Natural-language control flow on top of the System One client.
#
# A Choice question already returns a machine-readable branch decision: one of
# the option names the caller enumerated, plus a confidence. That is exactly the
# shape a `switch` needs, so the two constructs here compile ordinary Julia
# control flow into ONE Choice request and let Julia do the rest.
#
# `@branch` is the switch: the option names are the criteria keys, the selected
# index picks which body to evaluate, and the unselected bodies never run.
#
# `nl_dispatch` is multiple dispatch: `Meaning{S}` is a singleton type whose
# parameter carries a natural-language description, so `nl"..."` is usable in an
# ordinary method signature. One Choice per Meaning slot turns the state into a
# concrete `Meaning` instance per slot, and Julia's own dispatch then selects the
# method — including on the types of the remaining, ordinary arguments.
# ============================================================================

# The instruction sent when the caller names none. It is deliberately generic:
# the option names carry the semantics, and a Choice question with no
# instructions would make the model guess what the enumeration is *for*.
const _NL_INSTRUCTIONS = "Select the option that best describes the provided state."

# ─── Meanings ────────────────────────────────────────────────────────────────

"""
    Meaning{S}

A natural-language description lifted into the type domain: `S` is a `Symbol`
holding the description itself, and `Meaning{S}` is a zero-field singleton, so
it costs nothing at run time and can appear in an ordinary method signature.

Write the type with the [`@nl_str`](@ref) string macro — `nl"the customer wants
a refund"` *is* `Meaning{Symbol("the customer wants a refund")}` — and the
instance with `nl"..."()` or [`Meaning`](@ref)`(text)`.

```julia
route(::nl"the customer wants a refund", ticket) = refund!(ticket)
route(::nl"the customer reports a bug", ticket)  = file_bug!(ticket)

route(nl"the customer wants a refund"(), ticket)   # direct call, no API involved
nl_dispatch(route, ticket)                         # one Choice request picks the method
```

Dispatching on a `Meaning` is ordinary Julia dispatch: nothing about the type
reaches the network. Only [`nl_dispatch`](@ref) talks to the service, and only
to turn a piece of state into the `Meaning` instance to call with.
"""
struct Meaning{S} end

Meaning(text::AbstractString) = Meaning{Symbol(text)}()

# The description is the type parameter; both spellings (the type, as written in
# a signature, and the instance, as passed to a call) answer the same question.
_description(::Type{Meaning{S}}) where {S} = String(S)
_description(::Meaning{S}) where {S} = String(S)

Base.show(io::IO, ::Meaning{S}) where {S} = print(io, "nl\"", S, "\"()")
Base.show(io::IO, ::Type{Meaning{S}}) where {S} = print(io, "nl\"", S, "\"")

"""
    nl"some natural-language description"

The [`Meaning`](@ref) **type** for that description, so it reads as a type
wherever one is expected:

```julia
const Refund = nl"the customer wants a refund"   # a type alias
route(::nl"the customer wants a refund", ticket) = refund!(ticket)
```

Append `()` for the instance: `nl"the customer wants a refund"()`. The text is
interned as the type parameter, so two identical descriptions are the same type
and `===` compares them.

The macro performs no interpolation and no escaping — the description is taken
verbatim, which is what the model is shown.
"""
macro nl_str(text)
    :(Meaning{$(QuoteNode(Symbol(text)))})
end

# ─── Confidence gate ─────────────────────────────────────────────────────────

"""
    LowConfidenceError(question, answer, min_confidence)

Thrown when a natural-language branch or dispatch resolved to an option whose
`confidence` fell below the threshold the caller required, and no fallback was
given.

A Choice answer always names a winner, even when the distribution is nearly
flat; `confidence` is what separates "the state clearly matches this option"
from "this option won by a nose". Raising the threshold therefore converts a
silent mis-route into a loud failure. `answer` is the full
[`ChoiceAnswer`](@ref), so `probabilities` is available for a diagnostic or a
second-best policy.

Supply a `_` fallback line to [`@branch`](@ref), or the `fallback` keyword of
[`nl_dispatch`](@ref), to handle the case instead of raising.
"""
struct LowConfidenceError <: Exception
    question::String
    answer::ChoiceAnswer
    min_confidence::Float64
end

function Base.showerror(io::IO, e::LowConfidenceError)
    print(io, "LowConfidenceError: the answer to ", repr(e.question), " selected ",
          repr(e.answer.choice), " with confidence ", e.answer.confidence,
          ", below the required ", e.min_confidence,
          ". Lower the threshold, add a fallback, or make the options more distinct.")
end

# ─── @branch ─────────────────────────────────────────────────────────────────

const _BRANCH_KEYWORDS = (:model, :min_confidence, :instructions, :service, :config)

"""
    _branch_select(state, names, descriptions; kwargs...) -> Int

The one request behind [`@branch`](@ref): ask which of `names` describes
`state`, and return the 1-based index of the winner, or `0` when the answer was
below `min_confidence` and the caller has a fallback body.

`descriptions` is parallel to `names`; `nothing` means the option is described
by its name alone. A non-success result throws [`SystemOneError`](@ref) rather
than resolving to a branch — a call that did not happen must not pick one.
"""
function _branch_select(state, names::Vector{String}, descriptions::Vector{Any};
                        model::Union{Nothing,AbstractString}=nothing,
                        min_confidence::Real=0.0,
                        instructions=nothing,
                        service::ServiceEndpointSpec=TYPESAFEServiceEndpoint,
                        config::Union{Nothing,RequestConfig}=nothing,
                        has_fallback::Bool=false)::Int
    isempty(names) && throw(ArgumentError("a branch needs at least one option"))
    any(isempty, names) && throw(ArgumentError("branch option names must be non-empty"))
    length(unique(names)) == length(names) || throw(ArgumentError(
        "branch option names must be unique; the wire is a map, so a repeat would drop one: $(names)"))
    question = choice(isnothing(instructions) ? _NL_INSTRUCTIONS : instructions,
                      [names[i] => descriptions[i] for i in eachindex(names)])
    result = isnothing(model) ? ask(state, "branch" => question; service, config) :
                                ask(state, "branch" => question; model, service, config)
    result isa SystemOneSuccess || throw(SystemOneError(result))
    a = answer(result, "branch")
    a isa ChoiceAnswer || throw(ArgumentError(
        "a branch needs a Choice answer; the service answered with a $(typeof(a))"))
    idx = findfirst(==(a.choice), names)
    isnothing(idx) && throw(ArgumentError(
        "model selected an option that was not offered: $(repr(a.choice)) is not one of $(names)"))
    if a.confidence < min_confidence
        has_fallback && return 0
        throw(LowConfidenceError("branch", a, Float64(min_confidence)))
    end
    return idx
end

"""
    @branch state begin
        "option name"                  => expression
        ("option name", "description") => expression
        _                              => expression
    end
    @branch state key = value ... begin ... end

A natural-language `switch`. The option names are sent as the criteria of ONE
[`choice`](@ref) question about `state`; the selected option's expression is the
only one evaluated, and its value is the value of the macro.

Each line is `option => expression`. A bare left-hand side is the option **name**
— the text the model actually reads and the text the answer returns. The
2-tuple form `("name", "description")` attaches a longer description to that
name; it is the only way to do so. Left-hand sides are ordinary expressions
evaluated once, in source order, so an option name may be computed.

A final `_ => expression` line is the fallback taken when the winning option's
confidence is below `min_confidence`. It requires `min_confidence`: without a
threshold nothing is ever low-confidence and the line would be unreachable, so
that spelling is rejected at macro-expansion time.

Keywords go between the state and the block, written `key = value`:

| Keyword | Meaning |
| :--- | :--- |
| `model` | model name; omitted, the client default applies |
| `min_confidence` | threshold in `0 … 1`; below it, take `_` or throw [`LowConfidenceError`](@ref) |
| `instructions` | the question's instructions (default: `"Select the option that best describes the provided state."`) |
| `service` | endpoint type (default `TYPESAFEServiceEndpoint`) |
| `config` | `RequestConfig` for this call |

An unknown keyword, a block that is not `begin ... end`, a line that is not
`option => expression`, no options at all, two `_` lines, or a `_` that is not
last is an `ArgumentError` raised while the macro expands, so it surfaces when
the surrounding code is loaded rather than when the branch is first reached. A
non-success call throws [`SystemOneError`](@ref); the branch is never guessed.

```julia
ticket = "My package arrived crushed and the screen is cracked. I want my money back."

action = @branch ticket min_confidence=0.6 begin
    "the customer wants a refund"                                  => refund!(ticket)
    ("the customer reports a bug", "A defect in the software")     => file_bug!(ticket)
    "the customer asks a pricing question"                         => quote_price(ticket)
    _                                                              => escalate(ticket)
end
```
"""
macro branch(args...)
    length(args) >= 2 || throw(ArgumentError(
        "@branch takes a state expression and a `begin ... end` block of `option => expression` lines"))
    state = args[1]
    block = args[end]
    (block isa Expr && block.head === :block) || throw(ArgumentError(
        "@branch needs a `begin ... end` block of `option => expression` lines; got $(repr(block))"))

    kwnames = Symbol[]
    kwvalues = Any[]
    for a in args[2:end-1]
        (a isa Expr && a.head === :(=) && length(a.args) == 2 && a.args[1] isa Symbol) ||
            throw(ArgumentError(
                "@branch keywords are written `key = value` between the state and the block; got $(repr(a))"))
        k = a.args[1]::Symbol
        k in _BRANCH_KEYWORDS || throw(ArgumentError(
            "unknown @branch keyword $(repr(k)); allowed: " * join(string.(_BRANCH_KEYWORDS), ", ")))
        k in kwnames && throw(ArgumentError("@branch keyword $(repr(k)) is given twice"))
        push!(kwnames, k)
        push!(kwvalues, a.args[2])
    end

    lines = Any[x for x in block.args if !(x isa LineNumberNode)]
    option_names = Any[]
    option_descriptions = Any[]
    bodies = Any[]
    has_fallback = false
    fallback_body = nothing
    for (i, line) in enumerate(lines)
        (line isa Expr && line.head === :call && length(line.args) == 3 && line.args[1] === :(=>)) ||
            throw(ArgumentError("every @branch line is `option => expression`; got $(repr(line))"))
        lhs = line.args[2]
        rhs = line.args[3]
        if lhs === :_
            has_fallback && throw(ArgumentError("@branch accepts at most one `_` fallback line"))
            i == length(lines) || throw(ArgumentError("the `_` fallback must be the last @branch line"))
            :min_confidence in kwnames || throw(ArgumentError(
                "a `_` fallback needs `min_confidence = <threshold>`; without one no answer is ever " *
                "low-confidence and the fallback could never run"))
            has_fallback = true
            fallback_body = rhs
        elseif lhs isa Expr && lhs.head === :tuple
            length(lhs.args) == 2 || throw(ArgumentError(
                "a tuple @branch option is `(\"name\", \"description\") => expression`; got $(repr(lhs))"))
            push!(option_names, lhs.args[1])
            push!(option_descriptions, lhs.args[2])
            push!(bodies, rhs)
        else
            push!(option_names, lhs)
            push!(option_descriptions, nothing)
            push!(bodies, rhs)
        end
    end
    isempty(option_names) && throw(ArgumentError(
        "@branch needs at least one `option => expression` line"))

    statev = gensym("state")
    namesv = gensym("names")
    descv = gensym("descriptions")
    idxv = gensym("idx")

    setup = Any[:(local $statev = $(esc(state))),
                :(local $namesv = String[]),
                :(local $descv = Any[])]
    for i in eachindex(option_names)
        push!(setup, :(push!($namesv, $(esc(option_names[i])))))
        d = option_descriptions[i]
        push!(setup, :(push!($descv, $(isnothing(d) ? nothing : esc(d)))))
    end

    params = Expr(:parameters, Expr(:kw, :has_fallback, has_fallback))
    for (k, v) in zip(kwnames, kwvalues)
        push!(params.args, Expr(:kw, k, esc(v)))
    end
    push!(setup, :(local $idxv = $(Expr(:call, :_branch_select, params, statev, namesv, descv))))

    # Only the selected body is evaluated: the chain is built from the tail up,
    # with index 0 (the fallback) as the final `else`. With no fallback
    # `_branch_select` can only return 1..n, so the last option's body is the
    # final `else` and no unreachable arm is emitted.
    n = length(bodies)
    chain = has_fallback ? esc(fallback_body) : esc(bodies[n])
    for i in (has_fallback ? n : n - 1):-1:1
        chain = Expr(:if, :($idxv == $i), esc(bodies[i]), chain)
    end
    Expr(:block, setup..., chain)
end

# ─── Multiple dispatch on natural language ───────────────────────────────────

# The positional parameter types of a method, `self` dropped. A signature that
# is not a DataType after unwrapping carries no positional parameters we can
# read, so it contributes no slots.
function _nl_params(m::Method)::Vector{Any}
    sig = Base.unwrap_unionall(m.sig)
    sig isa DataType || return Any[]
    Any[sig.parameters[i] for i in 2:length(sig.parameters)]
end

# A slot is a position pinned to ONE concrete meaning. A bare `::Meaning` is a
# UnionAll and `::Meaning{S} where S` leaves a free type variable, so neither is
# concrete: wildcard methods are ordinary methods and are ignored here.
_is_meaning_slot(p)::Bool =
    p isa DataType && p <: Meaning && isconcretetype(p) && p.parameters[1] isa Symbol

_nl_slots(params::Vector{Any})::Vector{Int} =
    Int[i for i in eachindex(params) if _is_meaning_slot(params[i])]

"""
    _nl_methods(f) -> (methods, slots, arity)

The methods of `f` that carry natural-language arguments, sorted by definition
site so the option order is deterministic, together with the shared slot
positions and positional arity.

Methods with no slot are ignored (they are ordinary methods, including wildcard
`::Meaning` ones). Every slotted method must agree on arity and on which
positions are slots, because one set of questions has to serve all of them.
"""
function _nl_methods(f)
    slotted = Method[]
    for m in methods(f)
        isempty(_nl_slots(_nl_params(m))) && continue
        m.isva && throw(ArgumentError("varargs methods cannot carry natural-language arguments"))
        push!(slotted, m)
    end
    isempty(slotted) && throw(ArgumentError(
        "$(f) has no methods with natural-language (Meaning) arguments"))
    sort!(slotted; by = m -> (string(m.file), m.line))

    reference = _nl_params(slotted[1])
    slots = _nl_slots(reference)
    arity = length(reference)
    for m in slotted
        params = _nl_params(m)
        length(params) == arity || throw(ArgumentError(
            "every natural-language method of $(f) must take the same number of positional " *
            "arguments: $(slotted[1]) takes $(arity), $(m) takes $(length(params))"))
        found = _nl_slots(params)
        found == slots || throw(ArgumentError(
            "every natural-language method of $(f) must put its Meaning arguments in the same " *
            "positions: $(slotted[1]) uses $(slots), $(m) uses $(found)"))
    end
    return (slotted, slots, arity)
end

# The options offered for each slot: every distinct description defined for that
# position, in definition order. A description repeated across methods (because
# another slot varies) is offered once.
function _nl_options(slotted::Vector{Method}, slots::Vector{Int})::Vector{Vector{String}}
    options = [String[] for _ in slots]
    for m in slotted
        params = _nl_params(m)
        for k in eachindex(slots)
            d = _description(params[slots[k]])
            d in options[k] || push!(options[k], d)
        end
    end
    options
end

# Julia records an unnamed positional argument as `#unused#`, and gensymed names
# also start with `#`; neither is a name a caller wrote, so such a position is
# labelled by its index instead.
_nl_anonymous(n::Symbol)::Bool = n === Symbol("#unused#") || startswith(String(n), "#")

function _nl_argnames(m::Method, arity::Int)::Vector{Symbol}
    recorded = Base.method_argnames(m)
    Symbol[length(recorded) >= i + 1 ? recorded[i+1] : Symbol("#unused#") for i in 1:arity]
end

_nl_question_name(n::Symbol, pos::Int)::String = _nl_anonymous(n) ? "meaning_$(pos)" : String(n)
_nl_state_key(n::Symbol, pos::Int)::String = _nl_anonymous(n) ? "arg$(pos)" : String(n)

function _nl_instructions(instructions, n::Int)::Vector{Any}
    isnothing(instructions) && return Any[_NL_INSTRUCTIONS for _ in 1:n]
    instructions isa AbstractString && return Any[String(instructions) for _ in 1:n]
    instructions isa AbstractVector || throw(ArgumentError(
        "`instructions` must be `nothing`, a String, or a Vector with one entry per natural-language " *
        "slot; got $(typeof(instructions))"))
    length(instructions) == n || throw(ArgumentError(
        "`instructions` must carry one entry per natural-language slot ($(n)); got $(length(instructions))"))
    Any[x for x in instructions]
end

"""
    meanings(f) -> Dict{Int,Vector{String}}

The natural-language options `f` dispatches on, keyed by positional argument
index. Each vector lists the distinct descriptions defined for that slot, in the
order [`nl_dispatch`](@ref) sends them to the model — definition order, so the
listing is a faithful preview of the request.

Methods of `f` without a concrete [`Meaning`](@ref) argument are not part of the
natural-language interface and do not appear. `f` with no such method at all is
an `ArgumentError`.

```julia
route(::nl"the customer wants a refund", ticket) = :refund
route(::nl"the customer reports a bug", ticket)  = :bug

meanings(route)   # Dict(1 => ["the customer wants a refund", "the customer reports a bug"])
```
"""
function meanings(f)::Dict{Int,Vector{String}}
    slotted, slots, _ = _nl_methods(f)
    options = _nl_options(slotted, slots)
    Dict{Int,Vector{String}}(slots[k] => options[k] for k in eachindex(slots))
end

"""
    nl_dispatch(f, args...; model=nothing, service=TYPESAFEServiceEndpoint, config=nothing,
                min_confidence=0.0, fallback=nothing, instructions=nothing, state=nothing)

Resolve the natural-language arguments of `f` against a piece of state and call
the method Julia's own dispatch selects.

`f` is an ordinary generic function whose methods pin one or more positions to a
concrete [`Meaning`](@ref) — written `nl"..."` in the signature. One
[`choice`](@ref) question per such slot goes out in a **single** request; each
answer becomes a `Meaning` instance, those instances are spliced back into the
positions they came from, `args` fill the rest in order, and `f` is called. The
remaining arguments still dispatch normally, so a natural-language slot composes
with ordinary type-based dispatch.

The state is `state` when given, otherwise a `Dict{String,Any}` built from
`args` and keyed by the argument names of the first natural-language method (an
unnamed position becomes `"arg<index>"`). With no ordinary arguments there is
nothing to describe, so `state` is then required.

Each slot's question is named after that argument (an unnamed one becomes
`"meaning_<index>"`) and carries `instructions`: `nothing` for a generic
default, a `String` for every slot, or a `Vector` with one entry per slot.

`min_confidence` gates the result. Below it, `fallback` is called with the
caller's `args` if given, and otherwise [`LowConfidenceError`](@ref) is thrown.
A non-success call throws [`SystemOneError`](@ref). A resolved combination of
meanings that no method covers raises Julia's own `MethodError` — it is a gap in
the method table, not a service failure, and is not swallowed.

[`meanings`](@ref) previews the options exactly as they will be sent.

```julia
using UniLM

route(::nl"the customer wants a refund", ticket)        = (:refund, ticket)
route(::nl"the customer reports a bug in the app", t)   = (:bug, t)
route(::nl"the customer asks a pricing question", t)    = (:pricing, t)

ticket = "My package arrived crushed and the screen is cracked. I want my money back."

meanings(route)                     # Dict(1 => [...three descriptions, in definition order...])
nl_dispatch(route, ticket)          # one request, then route(nl"the customer wants a refund"(), ticket)

# Gate on confidence, with a fallback instead of an exception.
nl_dispatch(route, ticket; min_confidence = 0.7, fallback = (t,) -> (:escalate, t))

# The method is reachable without any network call at all.
route(nl"the customer wants a refund"(), ticket)
```
"""
function nl_dispatch(f, args...;
                     model::Union{Nothing,AbstractString}=nothing,
                     service::ServiceEndpointSpec=TYPESAFEServiceEndpoint,
                     config::Union{Nothing,RequestConfig}=nothing,
                     min_confidence::Real=0.0,
                     fallback=nothing,
                     instructions=nothing,
                     state=nothing)
    slotted, slots, arity = _nl_methods(f)
    options = _nl_options(slotted, slots)
    argnames = _nl_argnames(slotted[1], arity)
    ordinary = Int[p for p in 1:arity if !(p in slots)]
    length(args) == length(ordinary) || throw(ArgumentError(
        "$(f) takes $(length(ordinary)) ordinary argument(s) beside its $(length(slots)) " *
        "natural-language slot(s); got $(length(args))"))

    payload = _nl_state(f, state, args, ordinary, argnames)
    names = String[_nl_question_name(argnames[p], p) for p in slots]
    texts = _nl_instructions(instructions, length(slots))
    questions = [names[k] => choice(texts[k], [o => nothing for o in options[k]])
                 for k in eachindex(slots)]

    result = isnothing(model) ? ask(payload, questions...; service, config) :
                                ask(payload, questions...; model, service, config)
    result isa SystemOneSuccess || throw(SystemOneError(result))

    picked = ChoiceAnswer[]
    for k in eachindex(slots)
        a = answer(result, names[k])
        a isa ChoiceAnswer || throw(ArgumentError(
            "the answer to $(repr(names[k])) is a $(typeof(a)), not a ChoiceAnswer"))
        a.choice in options[k] || throw(ArgumentError(
            "the model chose $(repr(a.choice)) for $(repr(names[k])), which is not one of the " *
            "offered meanings: $(options[k])"))
        push!(picked, a)
    end
    for k in eachindex(slots)
        picked[k].confidence < min_confidence || continue
        isnothing(fallback) && throw(LowConfidenceError(names[k], picked[k], Float64(min_confidence)))
        return fallback(args...)
    end

    callargs = Any[]
    slot_i = 1
    arg_i = 1
    for p in 1:arity
        if p in slots
            push!(callargs, Meaning{Symbol(picked[slot_i].choice)}())
            slot_i += 1
        else
            push!(callargs, args[arg_i])
            arg_i += 1
        end
    end
    return f(callargs...)
end

# Values are passed through untouched: JSON.jl lowers what it knows, and
# anything it cannot encode fails loudly at serialization rather than being
# stringified into something the model would silently misread.
function _nl_state(f, state, args::Tuple, ordinary::Vector{Int}, argnames::Vector{Symbol})
    isnothing(state) || return state
    isempty(ordinary) && throw(ArgumentError(
        "$(f) has no ordinary arguments to describe the state; pass `state = ...` to nl_dispatch"))
    payload = Dict{String,Any}()
    for k in eachindex(ordinary)
        payload[_nl_state_key(argnames[ordinary[k]], ordinary[k])] = args[k]
    end
    payload
end
