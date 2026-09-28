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
#
# With a table of `key => sentence` entries (`texts`), the methods dispatch on the
# keys instead: the request offers the sentences exactly as `nl"..."` would, the
# chosen sentence maps back to its key locally, and `nl_classify` returns that key
# without dispatching.
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

# ─── Confidence gate and decision policy ─────────────────────────────────────

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
A threshold is one decision policy; the `decide` keyword of both takes any other.
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

"""
    DecisionDeclinedError(question, answer)

Thrown when the `decide` policy of [`nl_dispatch`](@ref) or [`@branch`](@ref)
returned `nothing` for an answer — it declined to act on it — and no fallback
was given. `question` names the question (`"branch"` for `@branch`); `answer` is
the full [`ChoiceAnswer`](@ref) the policy saw.

Supply a `_` fallback line to `@branch` or the `fallback` keyword of
`nl_dispatch`, or return an offered option from the policy, to handle the case.
A `min_confidence` threshold raises [`LowConfidenceError`](@ref) instead.
"""
struct DecisionDeclinedError <: Exception
    question::String
    answer::ChoiceAnswer
end

function Base.showerror(io::IO, e::DecisionDeclinedError)
    print(io, "DecisionDeclinedError: the decision policy declined the answer to ", repr(e.question),
          ", which selected ", repr(e.answer.choice), " with confidence ", e.answer.confidence,
          ". Add a fallback, or handle the case in the policy.")
end

# The policy a `min_confidence` threshold stands for: the argmax, declined below it.
_nl_gate(min_confidence::Real) = (a::ChoiceAnswer) -> a.confidence < min_confidence ? nothing : a.choice

# Checked before the request, so a `decide` that cannot take the answer fails
# before one is billed rather than after.
_nl_is_decider(d)::Bool = hasmethod(d, Tuple{ChoiceAnswer})

const _NL_BOTH_POLICIES = "pass `decide` or `min_confidence`, not both: a threshold is itself the policy " *
                          "`a -> a.confidence >= τ ? a.choice : nothing`"

# The policy of a single question: the confidence gate, or `decide`.
function _nl_one_policy(decide, min_confidence::Real)
    isnothing(decide) && return _nl_gate(min_confidence)
    min_confidence == 0 || throw(ArgumentError(_NL_BOTH_POLICIES))
    _nl_is_decider(decide) || throw(ArgumentError(
        "`decide` must be `nothing` or a callable of a ChoiceAnswer; got $(typeof(decide))"))
    decide
end

# Checked before the request too: a hook that cannot take the success would fail only
# after the call was billed.
function _nl_check_hook(on_response)::Nothing
    isnothing(on_response) || hasmethod(on_response, Tuple{SystemOneSuccess}) || throw(ArgumentError(
        "`on_response` must be `nothing` or a callable of a SystemOneSuccess; got $(typeof(on_response))"))
    nothing
end

# A fallback too: one that cannot take what it is called with — values of the types in
# `argtypes`, as dispatch sees them — would fail only after the declined answer was billed.
function _nl_check_fallback(fallback, argtypes::Vector{Any}, call::String)::Nothing
    isnothing(fallback) || hasmethod(fallback, Tuple{argtypes...}) || throw(ArgumentError(
        "`fallback` must be `nothing` or callable as `$(call)`, with arguments of types " *
        "$(Tuple{argtypes...}); got $(typeof(fallback))"))
    nothing
end

# A policy's verdict on one answer: the index of the offered option it picked, or
# `nothing` when it declined. Any other value would be a guess at what was meant.
function _nl_verdict(v, offered::Vector{String}, question::String)::Union{Nothing,Int}
    isnothing(v) && return nothing
    i = v isa AbstractString ? findfirst(==(v), offered) : nothing
    isnothing(i) && throw(ArgumentError(
        "the decision for $(repr(question)) must be one of the offered options $(offered), or " *
        "`nothing` to decline; got $(repr(v))"))
    i
end

# A keyed policy may also name an offered sentence by its key, as the table writes it:
# `:refund`, not the `Val(:refund)` a method receives.
function _nl_keyed_verdict(v, offered::Vector{String}, keys::Vector{Any}, question::String)::Union{Nothing,Int}
    isnothing(v) && return nothing
    i = v isa AbstractString ? findfirst(==(v), offered) : findfirst(k -> isequal(k, v), keys)
    isnothing(i) && throw(ArgumentError(
        "the decision for $(repr(question)) must be one of the offered sentences $(offered), the key of " *
        "one as the table writes it ($(join(map(repr, keys), ", "))), or `nothing` to decline; got $(repr(v))"))
    i
end

# A decline with no fallback: the threshold keeps its own error type.
_nl_declined(question::String, a::ChoiceAnswer, decide, min_confidence::Real)::Exception =
    isnothing(decide) ? LowConfidenceError(question, a, Float64(min_confidence)) :
                        DecisionDeclinedError(question, a)

# ─── @branch ─────────────────────────────────────────────────────────────────

const _BRANCH_KEYWORDS = (:model, :min_confidence, :decide, :instructions, :service, :config, :cancel,
                          :on_response)

"""
    _branch_select(state, names, descriptions; kwargs...) -> Int

The one request behind [`@branch`](@ref): ask which of `names` describes
`state`, and return the 1-based index of the option the policy picked — the
winner, unless `decide` picked another — or `0` when the policy declined and the
caller has a fallback body.

`descriptions` is parallel to `names`; `nothing` means the option is described
by its name alone. A non-success result throws [`SystemOneError`](@ref) rather
than resolving to a branch — a call that did not happen must not pick one.
"""
function _branch_select(state, names::Vector{String}, descriptions::Vector{Any};
                        model::Union{Nothing,AbstractString}=nothing,
                        min_confidence::Real=0.0,
                        decide=nothing,
                        instructions=nothing,
                        service::ServiceEndpointSpec=TYPESAFEServiceEndpoint,
                        config::Union{Nothing,RequestConfig}=nothing,
                        cancel::Union{Nothing,CancelToken}=nothing,
                        on_response=nothing,
                        has_fallback::Bool=false)::Int
    isempty(names) && throw(ArgumentError("a branch needs at least one option"))
    any(isempty, names) && throw(ArgumentError("branch option names must be non-empty"))
    length(unique(names)) == length(names) || throw(ArgumentError(
        "branch option names must be unique; the wire is a map, so a repeat would drop one: $(names)"))
    isnothing(decide) || _nl_is_decider(decide) || throw(ArgumentError(
        "@branch `decide` must be callable on a ChoiceAnswer; got $(typeof(decide))"))
    _nl_check_hook(on_response)
    has_fallback && isnothing(decide) && min_confidence <= 0 && throw(ArgumentError(
        "a `_` fallback needs a policy that can decline, but `decide` is nothing and `min_confidence` " *
        "is $(min_confidence): no answer is ever declined and the fallback could never run"))
    question = choice(isnothing(instructions) ? _NL_INSTRUCTIONS : instructions,
                      [names[i] => descriptions[i] for i in eachindex(names)])
    result = isnothing(model) ? ask(state, "branch" => question; service, config, cancel) :
                                ask(state, "branch" => question; model, service, config, cancel)
    result isa SystemOneSuccess || throw(SystemOneError(result))
    isnothing(on_response) || on_response(result)
    a = answer(result, "branch")
    a isa ChoiceAnswer || throw(ArgumentError(
        "a branch needs a Choice answer; the service answered with a $(typeof(a))"))
    a.choice in names || throw(ArgumentError(
        "model selected an option that was not offered: $(repr(a.choice)) is not one of $(names)"))
    policy = isnothing(decide) ? _nl_gate(min_confidence) : decide
    idx = _nl_verdict(policy(a), names, "branch")
    isnothing(idx) || return idx
    has_fallback && return 0
    throw(_nl_declined("branch", a, decide, min_confidence))
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

A final `_ => expression` line is the fallback taken when the decision policy
declines: the winner's confidence is below `min_confidence`, or `decide`
returned `nothing`. It requires one of the two: without a policy nothing is ever
declined and the line would be unreachable, so that spelling is rejected at
macro-expansion time. A policy that cannot decline when the branch runs —
`decide = nothing`, or a `min_confidence` of 0 or less — leaves the line just as
unreachable, and is an `ArgumentError` before the request.

Keywords go between the state and the block, written `key = value`:

| Keyword | Meaning |
| :--- | :--- |
| `model` | model name; omitted, the client default applies |
| `min_confidence` | threshold in `0 … 1`; below it, take `_` or throw [`LowConfidenceError`](@ref) |
| `decide` | a policy called with the [`ChoiceAnswer`](@ref): return an option name — any of them, not only the winner — or `nothing` to take `_` or throw [`DecisionDeclinedError`](@ref) |
| `instructions` | the question's instructions (default: `"Select the option that best describes the provided state."`) |
| `service` | endpoint type (default `TYPESAFEServiceEndpoint`) |
| `config` | `RequestConfig` for this call |
| `cancel` | [`CancelToken`](@ref) for this call (default: the ambient [`with_cancel`](@ref) token) |
| `on_response` | called once with the [`SystemOneSuccess`](@ref) before the policy runs, and never for a failed call — audits get `request_id`, `model` and `raw` from its `response`, which `decide` cannot see; its return value is ignored, and an exception from it propagates and runs no body |

`confidence` is computed over the options offered — `(n·p_max − 1)/(n − 1)`,
clamped to `0 … 1`, for `n` of them
([Confidence](https://docs.typesafe.ai/confidence)) — so the same
`min_confidence` is a different bar whenever the option list changes. `decide`
replaces the threshold with any policy over the full distribution; a threshold
is itself the policy `a -> a.confidence >= τ ? a.choice : nothing`, so giving
both is an error.

An unknown keyword, a block that is not `begin ... end`, a line that is not
`option => expression`, no options at all, two `_` lines, a `_` that is not
last, or `decide` together with `min_confidence` is an `ArgumentError` raised
while the macro expands, so it surfaces when the surrounding code is loaded
rather than when the branch is first reached. A `decide` that returns anything
but an option name or `nothing` is an `ArgumentError`, and no body runs. A
non-success call throws [`SystemOneError`](@ref); the branch is never guessed. A
cancelled call is one of those: its `result` is a `SystemOneCallError` whose `cause`
is a [`UniLMCancelled`](@ref), and no body runs.

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
    :decide in kwnames && :min_confidence in kwnames && throw(ArgumentError(
        "@branch takes `decide` or `min_confidence`, not both: a threshold is itself the policy " *
        "`a -> a.confidence >= τ ? a.choice : nothing`"))

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
            (:min_confidence in kwnames || :decide in kwnames) || throw(ArgumentError(
                "a `_` fallback needs `min_confidence = <threshold>` or `decide = <policy>`; without " *
                "one no answer is ever declined and the fallback could never run"))
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

# ─── Keyed tables ────────────────────────────────────────────────────────────

"""
    _NLTable

A validated table of `key => sentence` entries, in table order. The sentences are
what a request offers; the keys never leave the process. Both fields are tuples,
so a table is a value: it keys the plan cache by its contents, and a caller who
later mutates the vector they passed cannot reach a plan made from it.
"""
struct _NLTable
    keys::Tuple
    texts::Tuple{Vararg{String}}
end

# What a position receives for a key: a Symbol is lifted to `Val(k)`, so a method can
# pin it as `::Val{:refund}`; every other key is already something dispatch can see.
_nl_value(k) = k isa Symbol ? Val(k) : k

_nl_values(t::_NLTable)::Vector{Any} = Any[_nl_value(k) for k in t.keys]

# A key must be something a signature can name: a Symbol (as `::Val{:k}`), a singleton
# instance such as `Val(:k)` or `Refund()` (by its type), a type (as `::Type{T}`), or an
# enum value (by its enum type). A string, a number or a mutable instance is none of these.
_nl_is_key(k)::Bool = k isa Union{Symbol,Type,Enum} || Base.issingletontype(typeof(k))

const _NL_TUPLE_OF_PAIRS =
    "a Tuple of `key => sentence` pairs reads as a tuple of several tables: pass one table as a " *
    "vector, `[key => sentence, ...]`, or as a NamedTuple"

"""
    _nl_table(t) -> _NLTable

The one validator of a table, for every entry point: a `NamedTuple` of sentences or
a vector of `key => sentence` pairs, holding 1 to 255 entries (the options one
Choice question takes), each key one a signature can name, each sentence a string
that is not blank, and no sentence or dispatched value twice. Anything else is an
`ArgumentError`, raised before any request.
"""
_nl_table(t::NamedTuple)::_NLTable = _nl_entries(Pair{Any,Any}[k => v for (k, v) in pairs(t)])
function _nl_table(t::AbstractVector)::_NLTable
    for e in t
        e isa Pair || throw(ArgumentError(
            "a table of sentences holds `key => sentence` pairs; got $(repr(e))"))
    end
    _nl_entries(Pair{Any,Any}[e for e in t])
end
_nl_table(::AbstractDict) = throw(ArgumentError(
    "a table of sentences cannot be a Dict: a Dict has no order, and option order is part of what the " *
    "model reads. Pass a NamedTuple or a vector of `key => sentence` pairs, e.g. `collect(pairs(d))` " *
    "for an ordered dict"))
_nl_table(::Tuple) = throw(ArgumentError(_NL_TUPLE_OF_PAIRS))
_nl_table(t) = throw(ArgumentError(
    "a table of sentences is a NamedTuple of strings or a vector of `key => sentence` pairs; got $(typeof(t))"))

function _nl_entries(entries::Vector{Pair{Any,Any}})::_NLTable
    isempty(entries) && throw(ArgumentError("a table of sentences needs at least one `key => sentence` entry"))
    length(entries) <= 255 || throw(ArgumentError(
        "a table of sentences holds at most 255 entries, the options one Choice question takes; " *
        "got $(length(entries))"))
    said = Dict{String,Any}()     # sentence => its key
    # Dispatched value => its key: `:a` and `Val(:a)` collide, and so do two keys that are
    # equal without being identical (`Vector` and `Array{S,1} where S`), which a hash of
    # identity would tell apart. A scan: a table holds at most 255 entries.
    held = Pair{Any,Any}[]
    for (k, s) in entries
        # Singletons, but each already means something here: a policy returns `nothing`
        # to decline, so a `nothing` key could never be decided on, and `missing`
        # compares as `missing` rather than `true` or `false`.
        k isa Union{Nothing,Missing} && throw(ArgumentError(
            "$(repr(k)) cannot be a table key: a decision policy returns `nothing` to decline, and " *
            "`missing` does not compare as true or false. Use a Symbol or a type of your own"))
        _nl_is_key(k) || throw(ArgumentError(
            "the table key $(repr(k)) is a $(typeof(k)); a key must be a Symbol, a singleton instance such " *
            "as `Val(:k)`, a type, or an enum value, so that a method signature can name it"))
        s isa AbstractString && !isempty(strip(s)) || throw(ArgumentError(
            "the sentence for $(repr(k)) must be a non-empty string; got $(repr(s))"))
        haskey(said, s) && throw(ArgumentError(
            "the sentence $(repr(s)) is given for both $(repr(said[s])) and $(repr(k)): the model could " *
            "not tell them apart"))
        v = _nl_value(k)
        j = findfirst(h -> isequal(first(h), v), held)
        isnothing(j) || throw(ArgumentError(
            "$(repr(last(held[j]))) and $(repr(k)) are the same key: both dispatch as $(repr(v))"))
        said[s] = k
        push!(held, v => k)
    end
    _NLTable(Tuple(first(e) for e in entries), Tuple(String(last(e)) for e in entries))
end

# The tables of a keyed call: one table, or a Tuple with one table per keyed argument.
function _nl_tables(texts)::Vector{_NLTable}
    texts isa Tuple || return _NLTable[_nl_table(texts)]
    isempty(texts) && throw(ArgumentError(
        "`texts` is an empty tuple; pass one table, or a tuple with one table per keyed argument"))
    any(x -> x isa Pair, texts) && throw(ArgumentError(_NL_TUPLE_OF_PAIRS))
    _NLTable[_nl_table(t) for t in texts]
end

_nl_listed(xs::Vector{String}, n::Int, sep::String)::String =
    join(first(xs, n), sep) * (length(xs) > n ? "$(sep)and $(length(xs) - n) more" : "")

_nl_keylist(t::_NLTable)::String = _nl_listed(String[repr(k) for k in t.keys], 10, ", ")

# A method signature as advice: `types[p]` at the positions it names, `_` elsewhere.
_nl_spelled(f, arity::Int, types::Dict{Int,Any})::String =
    string(f, "(", join((haskey(types, p) ? "::$(types[p])" : "_" for p in 1:arity), ", "), ")")

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

# A position that takes any meaning rather than pinning one: `::Meaning`,
# `::Meaning{S} where S`, or a union of meanings.
_is_meaning_wildcard(p)::Bool = p isa Type && p <: Meaning && !_is_meaning_slot(p)

# Definition order, as `_nl_methods` explains it.
_nl_definition_order(m::Method) = (m.primary_world, string(m.file), m.line)

"""
    _nl_methods(f) -> (methods, slots, arity)

The methods of `f` that carry natural-language arguments, in definition order,
together with the shared slot positions and positional arity.

Definition order is the world age each method was defined in (`primary_world`):
source labels do not order REPL input (`"REPL[10]"` sorts before `"REPL[2]"`), and
`methods` returns its own order. Methods that share a world — a precompiled
package activates its methods together — fall back to source file path, then line:
when one function's methods span several files, that is path order, not `include`
order.

Methods with no slot are ignored (they are ordinary methods, including wildcard
`::Meaning` ones). Every slotted method must agree on arity and on which
positions are slots, because one set of questions has to serve all of them; a
partial wildcard — a method that pins some slots and leaves another as
`::Meaning` — breaks that agreement and is refused as such.
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
    sort!(slotted; by = _nl_definition_order)

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
            "positions: $(slotted[1]) uses $(slots), $(m) uses $(found)" *
            (any(p -> _is_meaning_wildcard(reference[p]) || _is_meaning_wildcard(params[p]),
                 symdiff(slots, found)) ?
                ". A method that pins some slots and leaves another as `::Meaning` is a partial " *
                "wildcard, which is not supported: define the missing concrete methods, or one " *
                "method wild in EVERY slot" : "")))
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

# The positional list of the call `nl_dispatch` makes — `at_slots` in the slot
# positions, `at_ordinary` in the others, each in order — as values or as types.
function _nl_splice(slots::Vector{Int}, ordinary::Vector{Int}, at_slots::AbstractVector,
                    at_ordinary::AbstractVector)::Vector{Any}
    out = Vector{Any}(undef, length(slots) + length(ordinary))
    out[slots] = at_slots
    out[ordinary] = at_ordinary
    out
end

# A method accepts ordinary arguments of types `argtypes` when the call made with
# its own meanings and those types is a subtype of its signature; `<:` against
# `m.sig` honours `where` clauses. `Core.Typeof` is the type dispatch sees for `f`:
# `Type{T}` for a constructor, where `typeof` would say `DataType`.
_nl_accepts(f, m::Method, slots::Vector{Int}, ordinary::Vector{Int}, argtypes::Vector{Any})::Bool =
    Tuple{Core.Typeof(f), _nl_splice(slots, ordinary, _nl_params(m)[slots], argtypes)...} <: m.sig

# The element types of a caller-supplied `argtypes`. A Vararg, Union or UnionAll
# tuple type has no fixed list of entries to line up with the ordinary positions.
function _nl_argtypes(argtypes::Type{<:Tuple})::Vector{Any}
    argtypes isa DataType && !Base.isvatuple(argtypes) ? Any[argtypes.parameters...] :
        throw(ArgumentError(
            "`argtypes` must be a tuple type with one entry per ordinary argument, such as " *
            "Tuple{String}; got $(argtypes)"))
end

const _NLOffer = @NamedTuple{slotted::Vector{Method}, slots::Vector{Int}, ordinary::Vector{Int},
                             arity::Int, options::Vector{Vector{String}}}

# The keyed counterpart (see `_nl_keyed_offer`): `slotted` holds every method of the
# call's arity, and each slot's offered sentences come with the table they are drawn
# from and their indices in it.
const _NLKeyedOffer = @NamedTuple{slotted::Vector{Method}, slots::Vector{Int}, ordinary::Vector{Int},
                                  arity::Int, options::Vector{Vector{String}}, tables::Vector{_NLTable},
                                  offered::Vector{Vector{Int}}}

"""
    _nl_offer(f, argtypes) -> (; slotted, slots, ordinary, arity, options)

What a call of [`nl_dispatch`](@ref) with ordinary arguments of types `argtypes`
offers: the slot layout of the whole method table, and each slot's options drawn
only from the natural-language methods that accept those types. A wrong number
of types, or types no natural-language method accepts, is an `ArgumentError`.
"""
function _nl_offer(f, argtypes::Vector{Any})::_NLOffer
    slotted, slots, arity = _nl_methods(f)
    ordinary = Int[p for p in 1:arity if !(p in slots)]
    length(argtypes) == length(ordinary) || throw(ArgumentError(
        "$(f) takes $(length(ordinary)) ordinary argument(s) beside its $(length(slots)) " *
        "natural-language slot(s); got $(length(argtypes))"))
    fits = filter(m -> _nl_accepts(f, m, slots, ordinary, argtypes), slotted)
    isempty(fits) && throw(ArgumentError(
        "no natural-language method of $(f) accepts ordinary arguments of types " *
        "$(Tuple{argtypes...}); its natural-language methods are " *
        join(string.(first(slotted, 5)), "; ") *
        (length(slotted) > 5 ? "; and $(length(slotted) - 5) more" : "")))
    (; slotted, slots, ordinary, arity, options = _nl_options(fits, slots))
end

# Every combination of one option per slot, the first slot varying slowest.
function _nl_combinations(options::AbstractVector{Vector{String}})::Vector{Vector{String}}
    isempty(options) && return [String[]]
    tails = _nl_combinations(options[2:end])
    [String[o; t] for o in options[1] for t in tails]
end

# The positional argument types of the call made when the answers are `combo`. A
# keyed answer is a sentence, and the call receives its key's dispatched value.
_nl_call_types(offer::_NLOffer, combo::Vector{String}, argtypes::Vector{Any})::Vector{Any} =
    _nl_splice(offer.slots, offer.ordinary, Any[Meaning{Symbol(d)} for d in combo], argtypes)
_nl_call_types(offer::_NLKeyedOffer, combo::Vector{String}, argtypes::Vector{Any})::Vector{Any} =
    _nl_splice(offer.slots, offer.ordinary,
               Any[Core.Typeof(_nl_value(_nl_key(offer.tables[k], combo[k]))) for k in eachindex(combo)], argtypes)

# The key of the sentence `s`; a table holds each sentence once.
_nl_key(t::_NLTable, s::String) = t.keys[something(findfirst(==(s), t.texts))]

# The offered combinations with no method to call. `hasmethod` is false for an
# ambiguous call as well as a missing one, and Julia could call neither. It looks
# in the newest world, as `methods(f)` and the final `invokelatest` do; without
# `world` it would use the caller's, blind to a method defined after the call began.
_nl_gaps(f, offer::Union{_NLOffer,_NLKeyedOffer}, argtypes::Vector{Any})::Vector{Vector{String}} =
    filter(c -> !hasmethod(f, Tuple{_nl_call_types(offer, c, argtypes)...}; world = Base.get_world_counter()),
           _nl_combinations(offer.options))

# The methods an ambiguous call of signature `sig` collides on: those that accept it
# and that no other accepting method is more specific than. Empty when none accepts
# it. `methods(f, types)` cannot tell: it lists nothing at all for an ambiguous call.
function _nl_colliding(f, sig::DataType)::Vector{Method}
    accepting = Method[m for m in methods(f) if sig <: m.sig]
    filter(m -> !any(o -> o !== m && Base.morespecific(o, m), accepting), accepting)
end

# The method that settles an ambiguity, spelled as a signature: the intersection of
# the colliding methods, or the call itself when that intersection needs a `where`.
function _nl_intersection(f, colliding::Vector{Method}, sig::DataType)::String
    both = reduce(typeintersect, (m.sig for m in colliding))
    fix = both isa DataType && !Base.isvatuple(both) ? both : sig
    string(f, "(", join(("::" * string(fix.parameters[i]) for i in 2:length(fix.parameters)), ", "), ")")
end

# A method wild in every slot is not more specific than the methods an ambiguous call
# collides on, so it closes a missing combination and not an ambiguous one: each kind
# of gap gets the advice that closes it.
function _nl_gap_error(f, offer::Union{_NLOffer,_NLKeyedOffer}, argtypes::Vector{Any},
                       gaps::Vector{Vector{String}})::ArgumentError
    uncovered, ambiguous = Vector{String}[], String[]
    for g in gaps
        sig = Tuple{Core.Typeof(f), _nl_call_types(offer, g, argtypes)...}
        colliding = _nl_colliding(f, sig)
        isempty(colliding) ? push!(uncovered, g) : push!(ambiguous,
            "$(repr(g)) is ambiguous between $(join(string.(colliding), " and ")): define their " *
            "intersection, $(_nl_intersection(f, colliding, sig))")
    end
    keyed = offer isa _NLKeyedOffer
    ArgumentError(
        "$(length(gaps)) of the $(prod(length, offer.options)) combinations of " *
        "$(keyed ? "sentences" : "meanings") $(f) offers " *
        "for ordinary arguments of types $(Tuple{argtypes...}) have no method to call; an answer " *
        "landing on one would be billed and then end in a MethodError, so nothing was sent." *
        (isempty(uncovered) ? "" :
            " No method covers $(_nl_listed(repr.(uncovered), 10, ", ")): " *
            _nl_cover_advice(f, offer, argtypes, first(uncovered))) *
        (isempty(ambiguous) ? "" :
            " $(_nl_listed(ambiguous, 3, "; ")). $(keyed ? "A catch-all" : "A method wild in every slot") " *
            "does not settle an ambiguity: it is not more specific than the methods that collide."))
end

function _nl_cover_advice(f, offer::_NLOffer, ::Vector{Any}, ::Vector{String})::String
    backstop = string(f, "(", join((p in offer.slots ? "::Meaning" : "_" for p in 1:offer.arity), ", "), ")")
    "define each with every slot pinned to a meaning, or one method wild in EVERY slot, such as " *
    "$(backstop), which is not an option and covers every combination that has no method. A method " *
    "wild in only some slots is not supported."
end

# In keyed mode a method may take any key at some positions and pin others: that is
# ordinary dispatch, so one catch-all is advice and not a requirement.
function _nl_cover_advice(f, offer::_NLKeyedOffer, argtypes::Vector{Any}, gap::Vector{String})::String
    types = _nl_call_types(offer, gap, argtypes)
    "define a method for each, such as " *
    "$(_nl_spelled(f, offer.arity, Dict{Int,Any}(p => types[p] for p in offer.slots))), or one catch-all " *
    "such as $(_nl_spelled(f, offer.arity, Dict{Int,Any}())), which covers every combination that has no method."
end

# Julia records an unnamed positional argument as `#unused#` — or, in a method that
# takes keywords, as the empty name — and gensymed names also start with `#`; none
# is a name a caller wrote, so such a position is labelled by its index instead.
_nl_anonymous(n::Symbol)::Bool = (s = String(n); isempty(s) || startswith(s, "#"))

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

# One decider per slot, in the shape `instructions` takes: `nothing` is the
# confidence gate, one callable serves every slot, a vector carries one per slot.
function _nl_policy(decide, min_confidence::Real, n::Int)::Vector{Any}
    isnothing(decide) && return Any[_nl_gate(min_confidence) for _ in 1:n]
    min_confidence == 0 || throw(ArgumentError(_NL_BOTH_POLICIES))
    deciders = decide isa AbstractVector ? Any[d for d in decide] : Any[decide for _ in 1:n]
    length(deciders) == n || throw(ArgumentError(
        "`decide` must carry one callable per natural-language slot ($(n)); got $(length(deciders))"))
    all(_nl_is_decider, deciders) || throw(ArgumentError(
        "`decide` must be `nothing`, a callable of a ChoiceAnswer, or a Vector with one such callable " *
        "per natural-language slot; got $(decide isa AbstractVector ? map(typeof, deciders) : typeof(decide))"))
    deciders
end

# What a call with ordinary arguments of types `argtypes` offers, the argument names
# its questions and state keys come from, and its gaps.
const _NLPlan = @NamedTuple{offer::_NLOffer, argnames::Vector{Symbol}, gaps::Vector{Vector{String}}}

function _nl_plan_uncached(f, argtypes::Vector{Any})::_NLPlan
    offer = _nl_offer(f, argtypes)
    (; offer, argnames = _nl_argnames(offer.slotted[1], offer.arity), gaps = _nl_gaps(f, offer, argtypes))
end

# Plans by call signature, each with the world it was computed in. Reflection and the
# gap check cost microseconds per offered combination, and a plan stays right for as
# long as the method table does not change, which every method definition marks by
# moving to a newer world. The key holds the TYPE of `f`, which owns the methods: a
# callable struct keyed by value would add an entry per instance.
const _NL_PLANS = Base.Lockable(Dict{Type,Tuple{UInt,_NLPlan}}())

# The plan under `key` in `cache`, made by `make()` unless one made in the current
# world is there. A cache that reaches `limit` entries starts over: that costs the
# next calls a re-plan, never a wrong plan.
function _nl_cached(make::Function, cache::Base.Lockable, key; limit::Int=typemax(Int))
    world = Base.get_world_counter()   # read first: a method defined while planning makes the entry stale
    cached = @lock cache get(cache[], key, nothing)
    !isnothing(cached) && first(cached) == world && return last(cached)
    plan = make()
    @lock cache begin
        length(cache[]) >= limit && !haskey(cache[], key) && empty!(cache[])
        cache[][key] = (world, plan)
    end
    plan
end

_nl_plan(f, argtypes::Vector{Any})::_NLPlan =
    _nl_cached(() -> _nl_plan_uncached(f, argtypes), _NL_PLANS, Tuple{Core.Typeof(f), argtypes...})

# ─── Keyed dispatch ──────────────────────────────────────────────────────────
# With `texts`, the methods of `f` dispatch on keys, and a table maps each key to the
# sentence the model reads. Nothing in a signature is sent: a table fills the position
# whose declared type its keys have, and a chosen sentence resolves to its key before
# Julia's own dispatch runs.

# The type a method accepts at positional argument `p`, as a one-element tuple type in
# the method's own `where` clauses: `f(::Type{T}, t) where {T<:Billing}` accepts
# `Tuple{Type{Refund}}` there, which the bare `Type{T}`, its `T` unbound, does not.
_nl_at(m::Method, p::Int) = Base.rewrap_unionall(Tuple{_nl_params(m)[p]}, m.sig)

_nl_takes(m::Method, p::Int, v)::Bool = Tuple{Core.Typeof(v)} <: _nl_at(m, p)

# A method pins a table at `p` when it declares a type there that the value of some key
# has, other than a type that takes everything: `::Val{:refund}`, `::Refund`,
# `::Type{<:Billing}` and `::Val` pin; `::Any` and an unconstrained `x::T where T` do not.
function _nl_pins(m::Method, p::Int, values::Vector{Any})::Bool
    at = _nl_at(m, p)
    !(Tuple{Any} <: at) && any(v -> Tuple{Core.Typeof(v)} <: at, values)
end

# The position a table fills: the one at which some method pins it. With none, or with
# several, where its keys go would be a guess.
function _nl_slot(f, slotted::Vector{Method}, t::_NLTable, arity::Int)::Int
    values = _nl_values(t)
    pinned = Pair{Int,Method}[]
    for p in 1:arity
        i = findfirst(m -> _nl_pins(m, p, values), slotted)
        i === nothing || push!(pinned, p => slotted[i])
    end
    isempty(pinned) && throw(ArgumentError(
        "no method of $(f) taking $(arity) positional arguments declares an argument for the keys " *
        "$(_nl_keylist(t)), which dispatch as " *
        "$(_nl_listed(unique(String[string(Core.Typeof(v)) for v in values]), 5, ", ")): a table fills " *
        "the position whose declared type its keys have, such as `::$(Core.Typeof(first(values)))`"))
    length(pinned) == 1 || throw(ArgumentError(
        "the keys $(_nl_keylist(t)) are declared at positions $(join(first.(pinned), ", ", " and ")) of " *
        "$(f) (" * join(("position $(p) by $(m)" for (p, m) in pinned), "; ") * "): a table fills one " *
        "position, so give the other arguments types its keys do not have"))
    first(only(pinned))
end

const _NLLayout = @NamedTuple{slotted::Vector{Method}, slots::Vector{Int}, ordinary::Vector{Int},
                              arity::Int, tables::Vector{_NLTable}}

"""
    _nl_keyed_layout(f, arity, tables) -> (; slotted, slots, ordinary, arity, tables)

The methods of `f` a keyed call of `arity` positional arguments can reach, in
definition order, and the position each table fills; tables and positions come
back in position order. Refused, before any request: no method of that arity; a
method pinning a `Meaning` (a call uses meanings in signatures or `texts`, not
both); a table with no position or with several; two tables in one position; and
a key no method takes at its position whatever the other arguments are — an
answer naming it would be billed and then end in a `MethodError`. A catch-all at
the position takes every key: that backstop is the caller's to write.
"""
function _nl_keyed_layout(f, arity::Int, tables::Vector{_NLTable})::_NLLayout
    slotted = sort!(Method[m for m in methods(f) if !m.isva && length(_nl_params(m)) == arity];
                    by = _nl_definition_order)
    isempty(slotted) && throw(ArgumentError(
        "$(f) has no method taking $(arity) positional arguments: the ordinary arguments and one per " *
        "table of `texts`"))
    mixed = findfirst(m -> any(_is_meaning_slot, _nl_params(m)), slotted)
    mixed === nothing || throw(ArgumentError(
        "$(slotted[mixed]) takes a natural-language meaning: a call uses meanings in signatures or " *
        "`texts`, not both"))
    found = Int[_nl_slot(f, slotted, t, arity) for t in tables]
    for j in eachindex(found), i in 1:j-1
        found[i] == found[j] && throw(ArgumentError(
            "the tables with the keys $(_nl_keylist(tables[i])) and $(_nl_keylist(tables[j])) are both " *
            "declared at position $(found[i]) of $(f): each table needs its own position"))
    end
    order = sortperm(found)
    slots, placed = found[order], tables[order]
    for (slot, t) in zip(slots, placed)
        values = _nl_values(t)
        drifted = Int[i for i in eachindex(values) if !any(m -> _nl_takes(m, slot, values[i]), slotted)]
        isempty(drifted) || throw(ArgumentError(
            "no method of $(f) takes $(_nl_listed(String[repr(t.keys[i]) for i in drifted], 10, ", ")) at " *
            "position $(slot), whatever the other arguments are: an answer naming one would be billed and " *
            "then end in a MethodError, so nothing was sent. Define " *
            _nl_listed(String[_nl_spelled(f, arity, Dict{Int,Any}(slot => Core.Typeof(values[i])))
                              for i in drifted], 3, ", ") *
            ", or remove $(length(drifted) == 1 ? "it" : "them") from the table"))
    end
    (; slotted, slots, ordinary = Int[p for p in 1:arity if !(p in slots)], arity, tables = placed)
end

# The first combination — an entry index per keyed position, drawn from `candidates` —
# whose call `m` accepts, or `nothing`; recursive over the positions, the first outermost.
function _nl_first_call(f, m::Method, layout::_NLLayout, argtypes::Vector{Any}, types::Vector{Vector{Any}},
                        candidates::Vector{Vector{Int}}, chosen::Vector{Int})::Union{Nothing,Vector{Int}}
    k = length(chosen) + 1
    if k > length(candidates)
        call = _nl_splice(layout.slots, layout.ordinary, Any[types[j][chosen[j]] for j in eachindex(chosen)], argtypes)
        return Tuple{Core.Typeof(f), call...} <: m.sig ? chosen : nothing
    end
    for i in candidates[k]
        found = _nl_first_call(f, m, layout, argtypes, types, candidates, Int[chosen; i])
        found === nothing || return found
    end
    nothing
end

# The entries a call with ordinary arguments of types `argtypes` offers, as indices into
# each table: those some method accepts at their position together with those types
# and, at the other keyed positions, some entry of their tables. A `where` clause can tie
# positions together, so a candidate is confirmed on a whole call; the other positions
# are drawn from the entries the method takes there on their own, which settles a method
# whose positions are not tied on its first combination.
function _nl_offered(f, layout::_NLLayout, argtypes::Vector{Any})::Vector{Vector{Int}}
    types = [Any[Core.Typeof(v) for v in _nl_values(t)] for t in layout.tables]
    offered = [falses(length(t.keys)) for t in layout.tables]
    for m in layout.slotted
        all(j -> Tuple{argtypes[j]} <: _nl_at(m, layout.ordinary[j]), eachindex(argtypes)) || continue
        takes = [Int[i for i in eachindex(types[k]) if Tuple{types[k][i]} <: _nl_at(m, layout.slots[k])]
                 for k in eachindex(types)]
        for k in eachindex(takes), i in takes[k]
            offered[k][i] && continue
            combo = _nl_first_call(f, m, layout, argtypes, types,
                                   Vector{Int}[j == k ? [i] : takes[j] for j in eachindex(takes)], Int[])
            combo === nothing && continue
            for j in eachindex(combo)
                offered[j][combo[j]] = true
            end
        end
    end
    Vector{Int}[findall(o) for o in offered]
end

"""
    _nl_keyed_offer(f, argtypes, tables) -> (; slotted, slots, ordinary, arity, options, tables, offered)

What a keyed call of [`nl_dispatch`](@ref) with ordinary arguments of types
`argtypes` offers: the layout of its tables, and at each keyed position the
sentences of the entries the call can reach, in table order, with their indices
into the table in `offered`. Types that reach no entry are an `ArgumentError`, as
they are for meanings.
"""
function _nl_keyed_offer(f, argtypes::Vector{Any}, tables::Vector{_NLTable})::_NLKeyedOffer
    layout = _nl_keyed_layout(f, length(argtypes) + length(tables), tables)
    offered = _nl_offered(f, layout, argtypes)
    any(isempty, offered) && throw(ArgumentError(
        "no method of $(f) accepts ordinary arguments of types $(Tuple{argtypes...}) together with a key " *
        "of `texts`; its methods taking $(layout.arity) positional arguments are " *
        _nl_listed(string.(layout.slotted), 5, "; ")))
    options = [String[layout.tables[k].texts[i] for i in offered[k]] for k in eachindex(offered)]
    (; layout.slotted, layout.slots, layout.ordinary, layout.arity, options, layout.tables, offered)
end

const _NLKeyedPlan = @NamedTuple{offer::_NLKeyedOffer, argnames::Vector{Symbol}, gaps::Vector{Vector{String}}}

function _nl_keyed_plan_uncached(f, argtypes::Vector{Any}, tables::Vector{_NLTable})::_NLKeyedPlan
    offer = _nl_keyed_offer(f, argtypes, tables)
    (; offer, argnames = _nl_argnames(_nl_namer(offer), offer.arity), gaps = _nl_gaps(f, offer, argtypes))
end

# The method the questions and the state keys are named from: the first, in definition
# order, that pins a table at its position — as the literal path names them from its
# first slotted method. A catch-all pins none, so its names never reach the request.
# Some method pins every table: that is how its position was found.
function _nl_namer(offer::_NLKeyedOffer)::Method
    pins(m::Method) = any(k -> _nl_pins(m, offer.slots[k], _nl_values(offer.tables[k])), eachindex(offer.slots))
    offer.slotted[something(findfirst(pins, offer.slotted))]
end

# Keyed plans depend on the tables too, so their key adds the tables' contents: each
# `_NLTable` is an immutable copy, never the vector a caller passed and may mutate.
# Method definitions bound the literal cache; sentences do not bound this one — a
# table built per call, with a sentence templated from the input, adds an entry per
# call — so it is capped.
const _NL_KEYED_PLANS = Base.Lockable(Dict{Tuple{Type,Tuple{Vararg{_NLTable}}},Tuple{UInt,_NLKeyedPlan}}())
const _NL_KEYED_PLANS_LIMIT = 1024

_nl_keyed_plan(f, argtypes::Vector{Any}, tables::Vector{_NLTable})::_NLKeyedPlan =
    _nl_cached(() -> _nl_keyed_plan_uncached(f, argtypes, tables), _NL_KEYED_PLANS,
               (Tuple{Core.Typeof(f), argtypes...}, Tuple(tables)); limit=_NL_KEYED_PLANS_LIMIT)

"""
    meanings(f; texts=nothing) -> Dict{Int,Vector{String}}

The natural-language options `f` dispatches on, keyed by positional argument
index. Each vector lists the distinct descriptions defined for that slot in
definition order, the order in which [`nl_dispatch`](@ref) sends them.

Definition order is the order in which the methods were defined as the code ran:
at the REPL, in a script, or in a package loaded from source
(`--compiled-modules=no`). A precompiled package defines all of its methods at
once when it loads, and those are ordered by source file path and then line —
which is not `include` order when one function's meanings span several files.
Keep each function's meanings in one file.

Methods of `f` without a concrete [`Meaning`](@ref) argument are not part of the
natural-language interface and do not appear. `f` with no such method at all is
an `ArgumentError`.

This is the union over every natural-language method. A call offers only the
meanings whose method accepts its ordinary arguments; `meanings(f, argtypes)`
previews that.

With `texts` — a table of `key => sentence` entries, or a tuple of them, as
`nl_dispatch` takes it — the options are sentences: each table's position maps to
every sentence of the table, in table order. With no call to take an arity from,
the arity is the one at which the methods of `f` declare arguments for the keys;
when several arities do, pass `argtypes`. The tables and their positions are
checked as a call checks them — each table is validated as `nl_dispatch`
validates it, its keys must fill one position of its own, no method of that arity
may pin a `Meaning`, and some method must take every key at its position — and a
failure is an `ArgumentError`. The combinations of keys are not checked:
[`meaning_gaps`](@ref)`(f, argtypes; texts)` lists those with no method, which a
call refuses before its request.

```julia
route(::nl"the customer wants a refund", ticket) = :refund
route(::nl"the customer reports a bug", ticket)  = :bug

meanings(route)   # Dict(1 => ["the customer wants a refund", "the customer reports a bug"])
```
"""
function meanings(f; texts=nothing)::Dict{Int,Vector{String}}
    isnothing(texts) || return _nl_keyed_meanings(f, _nl_tables(texts))
    slotted, slots, _ = _nl_methods(f)
    options = _nl_options(slotted, slots)
    Dict{Int,Vector{String}}(slots[k] => options[k] for k in eachindex(slots))
end

# With no call, the arity is the one at which the methods of `f` declare arguments for
# the keys; every entry that passes the checks of such a call is listed.
function _nl_keyed_meanings(f, tables::Vector{_NLTable})::Dict{Int,Vector{String}}
    values = map(_nl_values, tables)
    declares(m::Method) = !m.isva && any(p -> any(v -> _nl_pins(m, p, v), values), 1:length(_nl_params(m)))
    arities = sort!(unique(Int[length(_nl_params(m)) for m in methods(f) if declares(m)]))
    isempty(arities) && throw(ArgumentError(
        "no method of $(f) declares an argument for the keys " * join(map(_nl_keylist, tables), "; ")))
    length(arities) == 1 || throw(ArgumentError(
        "methods of $(f) taking $(join(arities, ", ", " and ")) positional arguments declare arguments " *
        "for the keys of `texts`: pass the types of the ordinary arguments, `meanings(f, argtypes; texts)`"))
    layout = _nl_keyed_layout(f, only(arities), tables)
    Dict{Int,Vector{String}}(layout.slots[k] => collect(layout.tables[k].texts) for k in eachindex(layout.slots))
end

"""
    meanings(f, argtypes::Type{<:Tuple}; texts=nothing) -> Dict{Int,Vector{String}}

The options [`nl_dispatch`](@ref) offers when its ordinary arguments have the
types in `argtypes` — one entry per ordinary argument, in order, such as
`Tuple{String}`, or `Tuple{}` when every argument is a meaning. A slot's options
are the distinct descriptions of the natural-language methods whose signature
accepts those types, in definition order as `meanings(f)` defines it; an
abstract type admits only the methods that accept all of it.

A tuple type of the wrong length is an `ArgumentError`, and so are types no
natural-language method accepts — `nl_dispatch` refuses such a call before any
request.

With `texts`, a keyed position offers the sentences whose keys a method accepts
there together with those types — and, at other keyed positions, some entry of
their tables — in table order: exactly the options
`nl_dispatch(f, args...; texts)` sends. A key accepted only with other types is
not offered, and is no error.

```julia
abstract type Phase end
struct Waiting <: Phase end
struct Confirming <: Phase end

step(::Waiting, ::nl"gives an order number", msg) = :to_confirming
step(::Confirming, ::nl"confirms", msg)           = :confirmed
step(::Confirming, ::nl"declines", msg)           = :declined
step(::Phase, ::nl"asks for a human", msg)        = :human

meanings(step, Tuple{Waiting,String})      # Dict(2 => ["gives an order number", "asks for a human"])
meanings(step, Tuple{Confirming,String})   # Dict(2 => ["confirms", "declines", "asks for a human"])
```
"""
function meanings(f, argtypes::Type{<:Tuple}; texts=nothing)::Dict{Int,Vector{String}}
    types = _nl_argtypes(argtypes)
    offer = isnothing(texts) ? _nl_offer(f, types) : _nl_keyed_offer(f, types, _nl_tables(texts))
    Dict{Int,Vector{String}}(offer.slots[k] => offer.options[k] for k in eachindex(offer.slots))
end

"""
    meaning_gaps(f, argtypes::Type{<:Tuple}; texts=nothing) -> Vector{Vector{String}}

The combinations of meanings [`nl_dispatch`](@ref) would offer for ordinary
arguments of the types in `argtypes` — as [`meanings`](@ref)`(f, argtypes)`
lists them — that no method of `f` covers. Each gap holds one description per
slot, slots in position order; gaps are listed with the first slot varying
slowest. A combination whose call is ambiguous is a gap too: Julia could not
call it either.

`nl_dispatch` runs this check before its request and sends nothing while a gap
exists, because an answer landing on one would be billed and then end in a
`MethodError`; its error tells the two kinds apart. Close a missing combination
by defining its method, with every slot pinned to a meaning, or add a method
that is wild in every slot, such as `f(::Meaning, ::Meaning, x)`: it is not an
option, and it covers every combination that has no method. It does not settle
an ambiguity, not being more specific than the methods that collide: close an
ambiguous combination with a method for their intersection, which the error
spells out.

With `texts`, a gap holds one sentence per keyed position. A missing combination
is closed by a method for its keys, or by a catch-all such as `f(_, _, x)`; a
method may also take any key at one position and pin a key at another, as
ordinary dispatch allows.

```julia
reply(::nl"a complaint", ::nl"a calm tone", msg)   = :apologise
reply(::nl"a complaint", ::nl"an angry tone", msg) = :escalate
reply(::nl"a question", ::nl"a calm tone", msg)    = :answer

meaning_gaps(reply, Tuple{String})   # [["a question", "an angry tone"]]

reply(::Meaning, ::Meaning, msg) = :triage
meaning_gaps(reply, Tuple{String})   # empty: the backstop covers the combination with no method
```
"""
function meaning_gaps(f, argtypes::Type{<:Tuple}; texts=nothing)::Vector{Vector{String}}
    types = _nl_argtypes(argtypes)
    _nl_gaps(f, isnothing(texts) ? _nl_offer(f, types) : _nl_keyed_offer(f, types, _nl_tables(texts)), types)
end

"""
    nl_dispatch(f, args...; model=nothing, service=TYPESAFEServiceEndpoint, config=nothing,
                cancel=nothing, min_confidence=0.0, decide=nothing, fallback=nothing,
                instructions=nothing, state=nothing, texts=nothing, on_response=nothing)

Resolve the natural-language arguments of `f` against a piece of state and call
the method Julia's own dispatch selects.

`f` is an ordinary generic function whose methods pin one or more positions to a
concrete [`Meaning`](@ref) — written `nl"..."` in the signature. One
[`choice`](@ref) question per such slot goes out in a **single** request; each
answer becomes a `Meaning` instance, those instances are spliced back into the
positions they came from, `args` fill the rest in order, and `f` is called. The
remaining arguments still dispatch normally, so a natural-language slot composes
with ordinary type-based dispatch.

Only meanings the call can reach are offered: a slot's options come from the
natural-language methods that accept the types of `args`, and
[`meanings`](@ref)`(f, argtypes)` previews them exactly as they will be sent.
Types no such method accepts are an `ArgumentError`. So is a gap — a combination
of offered meanings with no method, or with an ambiguous one, which
[`meaning_gaps`](@ref) lists — because an answer landing on it would be billed
and then end in a `MethodError`. Define the missing methods, or add a method
that is wild in every slot (`f(::Meaning, ::Meaning, x)`), which is not an
option and covers every combination that has no method; an ambiguous
combination needs a method for the intersection of the methods it collides on,
which the error names. Both are raised before any request. The options and
this check are worked out once per type of `f`, types of `args` and state of
the method table, so a method defined later is offered from the next call on.

The state is `state` when given, otherwise a `JSON.Object{String,Any}` built from
`args` in argument order, keyed by the argument names of the first
natural-language method (an unnamed position becomes `"arg<index>"`), so the
same arguments always make the same request. With no ordinary arguments there is
nothing to describe, so `state` is then required.

Each slot's question is named after that argument (an unnamed one becomes
`"meaning_<index>"`) and carries `instructions`: `nothing` for a generic
default, a `String` for every slot, or a `Vector` with one entry per slot.

A decision policy turns the answers into meanings. The default is the argmax
gated by `min_confidence`: below it, `fallback` is called with the caller's
`args` if given, and otherwise [`LowConfidenceError`](@ref) is thrown. A
`fallback` that cannot be called with arguments of the types of `args` is an
`ArgumentError` before any request.
`confidence` is computed over the options offered — `(n·p_max − 1)/(n − 1)`,
clamped to `0 … 1`, for `n` of them
([Confidence](https://docs.typesafe.ai/confidence)) — so the same threshold is a
different bar whenever the option list changes, including with the types of
`args`. `decide` replaces the gate: one callable for every slot, or a `Vector`
with one per slot, each called with that slot's [`ChoiceAnswer`](@ref) and
returning an offered meaning — any of them, not only the argmax — or `nothing`
to decline. Every slot is decided before anything is called; if any declined,
`fallback` runs, or [`DecisionDeclinedError`](@ref) is thrown for the first
declined slot. Any other return is an `ArgumentError`, and neither `f` nor
`fallback` runs. `decide` with a nonzero `min_confidence` is an `ArgumentError`
before any request: a threshold is itself the policy
`a -> a.confidence >= τ ? a.choice : nothing`.

`on_response` is called once with the [`SystemOneSuccess`](@ref) as soon as the
request has succeeded, before the policy runs: audits get `request_id`, `model`
and `raw` from its `response` — which `decide` cannot see. Its return value is
ignored; an exception from it propagates, and neither `f` nor `fallback` runs. It
is not called when nothing was sent, nor for a failed call, whose result the
thrown [`SystemOneError`](@ref) carries. A value that cannot be called on a
`SystemOneSuccess` is an `ArgumentError` before the request.

A non-success call throws [`SystemOneError`](@ref) — including one cancelled
through `cancel::Union{Nothing,CancelToken}` (default: the ambient
[`with_cancel`](@ref) token), whose `result` is a `SystemOneCallError` with a
[`UniLMCancelled`](@ref) `cause`; `f` is then never called.

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

# Or decide on the whole distribution: a likely refund is worth acting on even
# when it is not the argmax, and a spread-out answer goes to the fallback.
refund_first(a) = a.probabilities["the customer wants a refund"] >= 0.3 ?
                  "the customer wants a refund" : (a.confidence >= 0.7 ? a.choice : nothing)
nl_dispatch(route, ticket; decide = refund_first, fallback = (t,) -> (:escalate, t))

# The method is reachable without any network call at all.
route(nl"the customer wants a refund"(), ticket)
```

# Keys and a table of sentences

With `texts`, the methods of `f` dispatch on short keys, and a table maps each key
to the sentence the model reads. Only the sentences are sent — the request is byte
for byte the one the same sentences written as `nl"..."` would make — and the
chosen sentence maps back to its key locally, so no key reaches the model:

```julia
const INTENT = (refund = "the customer wants a refund",
                bug    = "the customer reports a bug in the app",
                other  = "anything else")
route(::Val{:refund}, t) = refund!(t)
route(::Val{:bug}, t)    = file_bug!(t)
route(::Val{:other}, t)  = escalate(t)

nl_dispatch(route, ticket; texts = INTENT)   # sends the 3 sentences; calls route(Val(:refund), ticket)
```

A table is a `NamedTuple` of sentences or a vector of `key => sentence` pairs,
with at most 255 entries and no sentence or key twice — two keys are the same when
their dispatched values are `isequal`, as `:a` and `Val(:a)` are. A `Dict` is
refused: it has no order, and option order is part of what the model reads.
`nothing` and `missing` cannot be keys: a policy returns `nothing` to decline, and
`missing` compares as neither true nor false. A `Symbol` key is
passed as `Val(key)`; a `Val` or other singleton instance (`Refund()`), a type
(`Refund`) or an enum value is passed as written, so methods on `::Refund`,
`::Type{<:Billing}` or an enum type select through ordinary dispatch, and a table
of leaf keys reaches the most specific method of a type hierarchy. A table fills
the one position whose declared type its keys have (`::Any` declares none);
several keyed positions take a `Tuple` of tables, one per position. The call has
`length(args)` positional arguments plus one per table, and only the methods of
that arity take part; one of them pinning a `Meaning` is an `ArgumentError`, since
a call uses meanings in signatures or `texts`, not both. Each keyed position is a
slot as above — one question, and one entry of an `instructions` or `decide`
vector, in position order. The question names and the state's keys come from the
first of those methods, in definition order, that pins a table (declares a type its
keys have), as they come from the first natural-language method for meanings: a
catch-all `route(k, t)` defined before it names nothing.

Every key must be taken at its position by some method whatever the other
arguments are: a key none takes is an `ArgumentError` before any request, naming
the method that would take it, such as `route(::Val{:shipping}, _)` — a catch-all
`route(k, t)` takes every key. Options then follow the types of `args` as above,
but in **table order**, where meanings in signatures are offered in definition
order; gaps are refused the same way. `decide` may also return the key of an
offered sentence as the table writes it (`:refund`, not `Val(:refund)`). Plans are
cached by the tables' contents as well. [`meanings`](@ref) and
[`meaning_gaps`](@ref) take the same `texts`, and [`nl_classify`](@ref) returns
the chosen key without dispatching.
"""
function nl_dispatch(f, args...;
                     model::Union{Nothing,AbstractString}=nothing,
                     service::ServiceEndpointSpec=TYPESAFEServiceEndpoint,
                     config::Union{Nothing,RequestConfig}=nothing,
                     cancel::Union{Nothing,CancelToken}=nothing,
                     min_confidence::Real=0.0,
                     decide=nothing,
                     fallback=nothing,
                     instructions=nothing,
                     state=nothing,
                     texts=nothing,
                     on_response=nothing)
    _nl_check_hook(on_response)
    # `Core.Typeof` is the type dispatch sees: `Type{Int}` for the argument `Int`.
    argtypes = Any[Core.Typeof(a) for a in args]
    _nl_check_fallback(fallback, argtypes, "fallback(args...)")
    (; offer, argnames, gaps) = isnothing(texts) ? _nl_plan(f, argtypes) :
                                                   _nl_keyed_plan(f, argtypes, _nl_tables(texts))
    isempty(gaps) || throw(_nl_gap_error(f, offer, argtypes, gaps))
    slots, options = offer.slots, offer.options
    policy = _nl_policy(decide, min_confidence, length(slots))

    payload = _nl_state(f, state, args, offer.ordinary, argnames)
    names = String[_nl_question_name(argnames[p], p) for p in slots]
    guidance = _nl_instructions(instructions, length(slots))
    questions = [names[k] => choice(guidance[k], [o => nothing for o in options[k]])
                 for k in eachindex(slots)]
    picked = _nl_ask(payload, questions, options, _nl_noun(offer); model, service, config, cancel, on_response)

    # Every slot is decided before anything runs, so an invalid verdict in a later
    # slot cannot follow a fallback or a call already made on an earlier one.
    chosen = Union{Nothing,Int}[_nl_verdict(policy[k](picked[k]), offer, k, names[k])
                                for k in eachindex(slots)]
    declined = findfirst(isnothing, chosen)
    if !isnothing(declined)
        isnothing(fallback) || return fallback(args...)
        throw(_nl_declined(names[declined], picked[declined], decide, min_confidence))
    end

    resolved = Any[_nl_resolved(offer, k, chosen[k]::Int) for k in eachindex(slots)]
    # `methods(f)` lists methods of the newest world, including ones defined after this
    # call began (an `@eval` at run time); the call must see the same method table as
    # the options that were offered, or a billed choice ends in a MethodError.
    return Base.invokelatest(f, _nl_splice(slots, offer.ordinary, resolved, Any[args...])...)
end

# The one request behind `nl_dispatch` and `nl_classify`, and the Choice answer to each
# of its questions. A failed call throws rather than resolving to anything, and
# `on_response` sees the success before any policy runs on it. `offered` names the
# options in a message: meanings, or a table's sentences.
function _nl_ask(payload, questions::Vector{Pair{String,ChoiceQuestion}}, options::Vector{Vector{String}},
                 offered::String; model::Union{Nothing,AbstractString}, service::ServiceEndpointSpec,
                 config::Union{Nothing,RequestConfig}, cancel::Union{Nothing,CancelToken},
                 on_response)::Vector{ChoiceAnswer}
    result = isnothing(model) ? ask(payload, questions...; service, config, cancel) :
                                ask(payload, questions...; model, service, config, cancel)
    result isa SystemOneSuccess || throw(SystemOneError(result))
    isnothing(on_response) || on_response(result)
    ChoiceAnswer[_nl_choice(result, first(questions[k]), options[k], offered) for k in eachindex(questions)]
end

function _nl_choice(result::SystemOneSuccess, name::String, options::Vector{String}, offered::String)::ChoiceAnswer
    a = answer(result, name)
    a isa ChoiceAnswer || throw(ArgumentError(
        "the answer to $(repr(name)) is a $(typeof(a)), not a ChoiceAnswer"))
    a.choice in options || throw(ArgumentError(
        "the model chose $(repr(a.choice)) for $(repr(name)), which is not one of the " *
        "offered $(offered): $(options)"))
    a
end

# What a message calls a call's options: the meanings in its signatures, or its table's sentences.
_nl_noun(::_NLOffer)::String = "meanings"
_nl_noun(::_NLKeyedOffer)::String = "sentences"

# A verdict for slot `k`, and what the slot then receives: a meaning is its own value;
# a keyed slot may also be decided by a key as the table writes it, and receives the
# key's dispatched value.
_nl_verdict(v, offer::_NLOffer, k::Int, question::String) = _nl_verdict(v, offer.options[k], question)
_nl_verdict(v, offer::_NLKeyedOffer, k::Int, question::String) =
    _nl_keyed_verdict(v, offer.options[k], Any[offer.tables[k].keys[i] for i in offer.offered[k]], question)

_nl_resolved(offer::_NLOffer, k::Int, i::Int) = Meaning{Symbol(offer.options[k][i])}()
_nl_resolved(offer::_NLKeyedOffer, k::Int, i::Int) = _nl_value(offer.tables[k].keys[offer.offered[k][i]])

"""
    nl_classify(state, texts; min_confidence=0.0, decide=nothing, fallback=nothing,
                instructions=nothing, model=nothing, service=TYPESAFEServiceEndpoint,
                config=nothing, cancel=nothing, on_response=nothing)

Classify `state` against ONE table of `key => sentence` entries and return the
chosen key as the table writes it: `:refund` for a `NamedTuple` table — the
`Symbol`, not `Val(:refund)` — and the instance, type or enum value otherwise. It
is the keyed form of [`nl_dispatch`](@ref) without the dispatch.

One [`ask`](@ref) goes out, with one [`choice`](@ref) question named
`"classify"` whose options are every sentence of the table, in table order, with
no descriptions and the default instructions of `nl_dispatch` unless
`instructions`, a `String`, is given; `state` is sent as given. The keys never
reach the model. The table is checked as `nl_dispatch` checks it — a `Dict`, an
empty table or one of more than 255 entries, a blank sentence, a sentence or key
given twice, or a key no signature could name is an `ArgumentError` before the
request — and a tuple of tables is an `ArgumentError` too.

The decision policy is that of `nl_dispatch`: the argmax gated by
`min_confidence`, or `decide`, called with the [`ChoiceAnswer`](@ref) and
returning an offered sentence, a key as the table writes it, or `nothing` to
decline — not both. A declined answer calls `fallback(state)` when given, and
otherwise throws [`LowConfidenceError`](@ref) or [`DecisionDeclinedError`](@ref)
for the question `"classify"`; any other verdict is an `ArgumentError`, and
`fallback` does not run. A `fallback` that cannot be called with `state` is an
`ArgumentError` before the request. A non-success call throws
[`SystemOneError`](@ref).
`on_response` is called once with the [`SystemOneSuccess`](@ref) before the
policy runs — audits get `request_id`, `model` and `raw` from its `response`,
which `decide` cannot see — as it is for `nl_dispatch`.

```julia
const INTENT = (refund = "the customer wants a refund",
                bug    = "the customer reports a bug in the app",
                other  = "anything else")

ticket = "My package arrived crushed and the screen is cracked. I want my money back."

nl_classify(ticket, INTENT)                                            # :refund
nl_classify(ticket, INTENT; min_confidence = 0.7, fallback = t -> :other)
```
"""
function nl_classify(state, texts;
                     min_confidence::Real=0.0,
                     decide=nothing,
                     fallback=nothing,
                     instructions::Union{Nothing,AbstractString}=nothing,
                     model::Union{Nothing,AbstractString}=nothing,
                     service::ServiceEndpointSpec=TYPESAFEServiceEndpoint,
                     config::Union{Nothing,RequestConfig}=nothing,
                     cancel::Union{Nothing,CancelToken}=nothing,
                     on_response=nothing)
    texts isa Tuple && !any(x -> x isa Pair, texts) && throw(ArgumentError(
        "nl_classify takes one table; got a tuple of $(length(texts)). A tuple of tables is for " *
        "nl_dispatch, one per keyed argument"))
    table = _nl_table(texts)
    policy = _nl_one_policy(decide, min_confidence)
    _nl_check_hook(on_response)
    _nl_check_fallback(fallback, Any[Core.Typeof(state)], "fallback(state)")
    options = collect(table.texts)
    question = "classify" => choice(String(something(instructions, _NL_INSTRUCTIONS)),
                                    [o => nothing for o in options])
    a = only(_nl_ask(state, [question], [options], "sentences"; model, service, config, cancel, on_response))
    keys = collect(Any, table.keys)
    i = _nl_keyed_verdict(policy(a), options, keys, "classify")
    isnothing(i) || return keys[i]
    isnothing(fallback) || return fallback(state)
    throw(_nl_declined("classify", a, decide, min_confidence))
end

# Values are passed through untouched: JSON.jl lowers what it knows, and
# anything it cannot encode fails loudly at serialization rather than being
# stringified into something the model would silently misread. The keys follow
# argument order, so the same arguments make byte-identical requests: a `Dict`
# iterates in hash order, which may change between Julia versions, and key order
# is part of what the model reads.
function _nl_state(f, state, args::Tuple, ordinary::Vector{Int}, argnames::Vector{Symbol})
    isnothing(state) || return state
    isempty(ordinary) && throw(ArgumentError(
        "$(f) has no ordinary arguments to describe the state; pass `state = ...` to nl_dispatch"))
    payload = JSON.Object{String,Any}()
    for k in eachindex(ordinary)
        payload[_nl_state_key(argnames[ordinary[k]], ordinary[k])] = args[k]
    end
    payload
end
