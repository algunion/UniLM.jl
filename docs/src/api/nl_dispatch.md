# [Natural-Language Dispatch](@id nl_dispatch_api)

[Dispatch on Meaning](@ref nl_dispatch_guide) is the how-to. `nl_classify`,
`nl_dispatch` and `@branch` each send one [`ask`](@ref) request, whose answers,
errors and limits are documented in [TypeSafe System One API (Jev)](@ref
system_one_api).

[`@branch`](@ref) is a `switch` whose cases are written in plain language. The
option names become the criteria of one [`choice`](@ref) question about the
state; the winning option selects which expression is evaluated and the other
bodies never run. However many options a branch lists, it costs a single
request.

[`nl_dispatch`](@ref) lifts the same idea into the method table, in two
spellings that send the same request. With `texts = TABLE`, methods dispatch on
short keys — `Val{:billing}`, your own types or instances, enum values — and the
ordered table holds the sentence the model reads for each key; the keys never
leave the process. With `nl"..."`, the sentence is itself a [`Meaning`](@ref)
type written in the signature. Either way `nl_dispatch` sends one Choice
question per such position — again in a single request — turns each answer back
into the key or the `Meaning` instance and calls the function, so Julia's own
dispatch selects the method and the remaining arguments still dispatch on their
types. [`nl_classify`](@ref) asks the same question over a table and returns the
key itself, without dispatching. All three act on an answer through a decision
policy, and all three take `on_response`, which sees the whole
[`SystemOneSuccess`](@ref) — request id, model, raw body — before the policy
runs. The default gates on `confidence`: below `min_confidence`
`@branch` takes its `_` line and `nl_dispatch` calls `fallback`, and with
neither they raise [`LowConfidenceError`](@ref) rather than act on a near-tie.
`decide` replaces the gate with any function of the [`ChoiceAnswer`](@ref) that
returns an offered option — not only the winner — or `nothing` to decline,
which takes the same fallback or raises [`DecisionDeclinedError`](@ref).
[`meanings`](@ref)`(f)` is the union of the options of every natural-language
method of `f`; `meanings(f, Tuple{…})` lists the ones a call with ordinary
arguments of those types sends, in the order it sends them. Both take the same
`texts` a keyed call does.

```@docs
Meaning
@nl_str
@branch
nl_dispatch
nl_classify
meanings
meaning_gaps
LowConfidenceError
DecisionDeclinedError
```

## Usage

```julia
using UniLM

ticket = "My package arrived crushed and the screen is cracked. I want my money back."

action = @branch ticket min_confidence=0.6 begin
    "the customer wants a refund"                                => :refund
    ("the customer reports a bug", "A defect in the software")   => :bug
    "the customer asks a pricing question"                       => :pricing
    _                                                            => :escalate
end                                                              # => :refund

# The same decision as multiple dispatch: `nl"..."` is a type, so the meanings
# live in the signatures and Julia selects the method once they are resolved.
route(::nl"the customer wants a refund", t)           = (:refund, t)
route(::nl"the customer reports a bug in the app", t) = (:bug, t)
route(::nl"the customer asks a pricing question", t)  = (:pricing, t)

meanings(route)                     # Dict(1 => [...the three descriptions...])
nl_dispatch(route, ticket)          # one request, then route(nl"..."(), ticket)
route(nl"the customer wants a refund"(), ticket)   # direct call, no request at all
```

A non-success call raises [`SystemOneError`](@ref) instead of resolving to a
branch, and a combination of offered meanings that no method covers is an
`ArgumentError` raised before any request — [`meaning_gaps`](@ref) lists such
gaps without a request.
