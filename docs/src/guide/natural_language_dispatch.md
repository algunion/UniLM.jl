# [Multiple Dispatch on Natural Language](@id nl_dispatch_guide)

Julia selects a method from the types of *all* the arguments, not just the
first. With Jev one of those positions can hold a natural-language **meaning**
instead of a data type: `nl"the customer wants a refund"` is an ordinary Julia
type, so `route(::nl"the customer wants a refund", ticket)` is an ordinary
method with an ordinary signature. [`nl_dispatch`](@ref) asks Jev which of the
meanings declared across the method table fits the actual input, turns the
answer back into a value, and lets Julia's own dispatch pick the method — on
that meaning and on the types of every other argument at the same time. However
many meaning slots the signature has, one call is one request: each slot rides
along as an independent [`choice`](@ref) question over the same state. The model
returns a calibrated choice over the options you declared and never returns
text, so the decision arrives already typed, with a probability for every option
and a `confidence` to gate on. It is cheap — only input tokens are billed
([Models](https://docs.typesafe.ai/models)) — and a System One model is built to
judge in one shot rather than to generate ([System
One](https://docs.typesafe.ai/concepts/system-one)). Contrast that with LLM tool
calling, where a generative model writes a call into text that you prompt for
and then parse, a malformed call costs a retry, and nothing in the reply tells
you how close the runner-up was. Here the option list *is* the method table,
restricted to the methods the call can reach; the answer is one of its entries by
construction; and a combination no method covers is refused before anything is
sent, instead of surfacing as a `MethodError` after the request was billed.

Every block that calls Jev runs when this page is built, so what it prints is a
real answer — a live call, or a recorded live answer when the build has no API
key. The numbers in the prose are measurements, not what a block prints.

## The sixty-second version

Write one method per meaning. [`meanings`](@ref) shows exactly what will be
offered to the model, and needs no network at all:

```@example nldispatch
using UniLM

route(::nl"the customer wants a refund", ticket)           = :refund
route(::nl"the customer reports a bug in the app", ticket) = :bug
route(::nl"anything else", ticket)                         = :other

for (position, options) in sort(collect(meanings(route)))
    println(position, " => ", options)
end
```

Then dispatch on a real ticket. This is the only line that talks to the service:

```@example nldispatch
nl_dispatch(route, "My package arrived crushed and the screen is cracked. I want my money back.")
```

One request went out, it carried one Choice question over the three meanings,
and the winning meaning became the first argument of an ordinary Julia call.

## How it works

**`Meaning` is a type.** [`Meaning`](@ref)`{S}` is a zero-field singleton whose
only type parameter is a `Symbol` holding the description. The
[`@nl_str`](@ref) macro writes the **type**, so it reads as one wherever a type
is expected — in a signature, in a `const` alias, in an `isa` test. Append `()`
for the instance, or build it from a runtime string with `Meaning(text)`:

```@example nldispatch
const Refund = nl"the customer wants a refund"    # a type alias
(Refund, Refund(), Meaning("the customer wants a refund") isa Refund)
```

The text is interned as the type parameter, so two identical descriptions are
the same type and `===` compares them. Nothing about the type reaches the
network on its own; only [`nl_dispatch`](@ref) makes a call, and only to turn a
piece of state into the instance to call with.

**What `nl_dispatch` collects.** It reads `methods(f)` and keeps the methods that
pin at least one positional argument to a *concrete* meaning, in definition
order: the world age each method was defined in. The options therefore follow
the order in which the methods were defined — in a file, a script or a REPL
session — and [`meanings`](@ref) previews it faithfully. Methods loaded from a
package image share one world and fall back to source file path, then line —
not `include` order — so a function whose meanings span several files offers
them in a different order from the image than with `--compiled-modules=no`; keep
one function's meanings in one file. Redefining a method, by editing its body in
a running session for instance, gives it a new world age and moves its meaning
to the end. The order is part of what the model reads ([Definition order is an
input](@ref nl_dispatch_order)). A method with no concrete meaning anywhere in
its signature — a wildcard `f(::Meaning, x)`, or a method with no `Meaning` at
all — is an ordinary method and is skipped. Every method that *is* collected
must agree with the others on positional arity and on which positions are
meaning slots, because one set of questions has to serve all of them; a
disagreement is an `ArgumentError` naming both methods, raised before any
request goes out. Varargs cannot carry a meaning.

**How the request is built.** One [`choice`](@ref) question per slot, all in a
single [`ask`](@ref):

- The **question name** is the slot's argument name, or `meaning_<position>`
  when the slot is unnamed (`::nl"..."` with nothing in front of the `::`).
  Question names key the answers and are never sent to the model.
- The **option names are the meanings themselves**, with no description
  attached: each slot offers the distinct meanings of the methods that accept
  the call's ordinary arguments ([Composing with ordinary
  dispatch](@ref nl_dispatch_composing)). There is no other text for the model to
  read, so the description you write in the signature *is* the prompt for that
  option.
- The **state** is a JSON object keyed by the names of the remaining, ordinary
  arguments (`arg<position>` for an unnamed one), in argument order, so the same
  arguments always make the same request. Pass `state = ...` to send something
  else instead; with no ordinary arguments at all there is nothing to describe,
  so `state` is then required.
- The **instructions** default to a generic *"Select the option that best
  describes the provided state."* — the options carry the semantics.
- The **model** is the client default (`jev-latest`, or whatever
  `TYPESAFE_DEFAULT_MODEL` names) unless `model =` overrides it for this call.

For the sixty-second example above, that is exactly this body — captured off the
wire by pointing the call at a local server instead of `api.typesafe.ai`:

```json
{
  "state": {
    "ticket": "My package arrived crushed and the screen is cracked. I want my money back."
  },
  "model": "jev-latest",
  "questions": {
    "meaning_1": {
      "type": "choice",
      "instructions": "Select the option that best describes the provided state.",
      "criteria": {
        "the customer wants a refund": null,
        "the customer reports a bug in the app": null,
        "anything else": null
      }
    }
  }
}
```

The model sees the state, the instructions and the option names. It does **not**
see the function's name, the method bodies, the return types, the question name
`meaning_1`, or the fact that this is dispatch at all — as far as Jev is
concerned this is one Choice question like any other
([Choice](https://docs.typesafe.ai/primitives/choice)).

**How the answer comes back.** Each answer is a [`ChoiceAnswer`](@ref) whose
`choice` is one of the option names by construction. A decision policy picks the
meaning to act on — by default the winner, gated by `min_confidence`
([Confidence and control](@ref nl_dispatch_control)). `nl_dispatch` turns it
back into a `Meaning` instance, splices it into the position it came from,
fills the remaining positions with your arguments in order, and makes a normal
Julia call. From there nothing is special: the value it returns is whatever the
selected method returns.

## Several meaning slots, one request

A signature may pin more than one position. Four methods over two slots enumerate
four combinations:

```@example nldispatch
reply(::nl"a complaint", ::nl"a calm tone", msg)  = :apologise
reply(::nl"a complaint", ::nl"an angry tone", msg) = :escalate
reply(::nl"a question", ::nl"a calm tone", msg)   = :answer
reply(::nl"a question", ::nl"an angry tone", msg) = :answer_and_apologise

for (position, options) in sort(collect(meanings(reply)))
    println(position, " => ", options)
end
```

```@example nldispatch
nl_dispatch(reply, "This is the third time I have written about my broken order.")
```

Both slots resolve in **one** request. The two questions are independent — they
are asked about the same state, and neither answer becomes context for the
other — so the cost of a second slot is the tokens of its option list, not a
second ingestion of the state. That is the same economics as batching questions
into one [`ask`](@ref) ([parallel
questions](https://docs.typesafe.ai/cookbooks/parallel_questions)).

Four methods cover all four combinations here. When they do not, an answer can
land on a combination that has no method, and the request would already be
billed by the time Julia raised its `MethodError`. So `nl_dispatch` checks every
combination of the offered meanings against the method table first, and a gap is
an `ArgumentError` naming it, with nothing sent. [`meaning_gaps`](@ref) runs the
same check without a call — in a unit test, for instance — for ordinary
arguments of the types you name. A catch-all meaning added for only one of the
tones opens a gap:

```@example nldispatch
reply(::nl"anything else", ::nl"a calm tone", msg) = :triage

meaning_gaps(reply, Tuple{String})   # the message is a String
```

```@example nldispatch
try
    nl_dispatch(reply, "This is the third time I have written about my broken order.")
catch err
    println(sprint(showerror, err))
end
```

Two ways to close a gap:

```@example nldispatch
# 1. Define the missing combination. A catch-all meaning IS offered to the model,
#    so the model can say "none of these" instead of forcing its probability onto
#    an option that does not fit.
reply(::nl"anything else", ::nl"an angry tone", msg) = :triage

meaning_gaps(reply, Tuple{String})
```

```@example nldispatch
# 2. A method wild in every slot. `::Meaning` is not a concrete meaning, so this
#    method is never collected and never offered to the model — it is a pure
#    Julia backstop that ordinary dispatch reaches when nothing more specific
#    matches.
reply(::Meaning, ::Meaning, msg) = :unclassified

reply(nl"a rant"(), nl"a calm tone"(), "…")   # no request; ordinary dispatch
```

Prefer the first: a backstop tells you nothing about *why* it fired, while a
catch-all meaning comes back with a name and a probability. A backstop also
covers every combination that has no method, so once it exists `meaning_gaps`
cannot report the method you forgot. It does not settle an *ambiguous*
combination, which two methods match with neither more specific: the error
names both, and a method for their intersection settles it. Note what the
wildcard is not allowed to be — a method that pins some slots and leaves others
wild, such as `reply(::nl"a complaint", ::Meaning, msg)`, disagrees with the
other methods about which positions are slots, and [`meanings`](@ref) and
[`nl_dispatch`](@ref) both reject it with an `ArgumentError` rather than send a
question list that does not match the method table. Wildcard *every* slot or
none of them.

## [Composing with ordinary dispatch](@id nl_dispatch_composing)

A resolved meaning is an ordinary argument, so the remaining positions keep
dispatching on their types exactly as they always did — and their types decide
which meanings are offered at all:

```@example nldispatch
handle(::nl"a greeting", who::String) = "hello, " * who
handle(::nl"a greeting", n::Int)      = "hello, customer #" * string(n)
handle(::nl"a complaint", who::String) = "sorry, " * who

println(meanings(handle))                  # every meaning of every method
println(meanings(handle, Tuple{String}))   # what nl_dispatch(handle, "ada") offers
println(meanings(handle, Tuple{Int}))      # what nl_dispatch(handle, 41) offers
println(handle(nl"a greeting"(), 41))
```

`meanings(handle)` is the union over the method table, and lists `"a greeting"`
once even though two methods declare it: the options are the *distinct*
descriptions per slot. A call offers only the meanings whose method accepts the
concrete types of its ordinary arguments, and `meanings(f, Tuple{…})` lists them
for the types you name, exactly as they will be sent. `nl_dispatch(handle, 41)`
never offers `"a complaint"`: no method could act on that answer with an `Int`.
The model picks among the meanings the call can reach; Julia picks between the
`String` and the `Int` method afterwards, with no extra request. A call whose
argument types no natural-language method accepts, such as
`nl_dispatch(handle, 2.5)`, is an `ArgumentError` before any request.

That is what turns a method table into a state machine: make the conversation's
state a typed, ordinary argument, and each state offers the model only its own
transitions. [Semantic Programs with Jev](@ref jev_programs_guide) builds a
conversation state machine this way. Because the number of options now depends
on the argument types, so does what a fixed `min_confidence` means ([Confidence
and control](@ref nl_dispatch_control)).

This also makes the interface open. The option list is the method table, and a
method table is extensible from anywhere: a package that does not own `handle`
can add a method on an ordinary argument type it owns —
`handle(::nl"a refund request", t::MyTicket)` — and from the next call with a
`MyTicket` on, it is one more option the model may choose. Nothing central
enumerates the meanings, so adding a case never means editing a list in two
places. A method on types the package does not own —
`handle(::nl"a refund request", who::String)` added to someone else's `handle` —
is type piracy: it changes what every caller with a `String` is offered, and two
packages that add the same one collide silently, the one loaded last winning.
Coordinate such a case with the owner of the function.

## [Confidence and control](@id nl_dispatch_control)

A Choice answer always names a winner, even when the distribution is nearly
flat, so the default gate is `confidence` rather than `choice`. `confidence`
measures how concentrated the distribution is — all the mass on one option gives
1.0, a flat spread gives a low number. The probabilities are calibrated across
groups of answers, which is what makes a gate worth tuning, but that is not a
guarantee about one answer: a high confidence says the distribution is peaked,
not that the winner is *correct*
([Confidence](https://docs.typesafe.ai/confidence)).

```@example nldispatch
unclear = "Your update deleted my saved cards and now I've been billed for a plan I cancelled."

# Below the threshold, `fallback` is called with the caller's own arguments.
nl_dispatch(route, unclear; min_confidence = 0.7, fallback = t -> :needs_a_human)
```

With no `fallback`, an answer below the threshold is a loud failure instead of a
route: [`LowConfidenceError`](@ref), which holds the whole
[`ChoiceAnswer`](@ref), so a handler can read `probabilities` for a diagnostic
or a second-best policy.

For a Choice over `n` options, `confidence` is `clamp((n·p_max − 1)/(n − 1), 0, 1)`:
the top probability `p_max`, rescaled so that a uniform answer scores 0. The
formula fits every answer we measured within 0.02 (jev-1.13.0, September 2026,
on our own labeled sets). A fixed `min_confidence` is therefore a different bar
on the top probability for every number of options — 0.5 means `p_max ≥ 0.75`
with 2 options, 0.6 with 5 and 0.55 with 10 — and adding a method, or calling
with argument types that reach fewer methods, moves it.

`decide` replaces the gate with any policy over the full answer. It is called
with the [`ChoiceAnswer`](@ref) and returns an offered meaning — any of them,
not only the winner — or `nothing` to decline, which runs `fallback` or throws
[`DecisionDeclinedError`](@ref). A bar on the top probability itself stays where
you put it, whatever the number of options:

```@example nldispatch
sure(a) = a.probabilities[a.choice] >= 0.8 ? a.choice : nothing

[nl_dispatch(route, ticket; decide = sure, fallback = t -> :needs_a_human)
 for ticket in ("My package arrived crushed and the screen is cracked. I want my money back.", unclear)]
```

With three options, `min_confidence = 0.7` sets about the same bar; a fourth
meaning would lower it to `p_max ≥ 0.775` and leave `sure` at 0.8. A policy can
also prefer a cheaper meaning to a slightly likelier one: the loss-matrix
section of [Semantic Programs with Jev](@ref jev_programs_guide) derives such a
policy from what each mistake costs. `decide` together with a nonzero
`min_confidence` is an `ArgumentError` before any request, because a threshold
is itself the policy `a -> a.confidence >= τ ? a.choice : nothing`. Every slot
is decided before anything runs, and a policy that returns anything but an
offered meaning or `nothing` is an `ArgumentError`: neither `f` nor `fallback`
runs. The keywords:

| Keyword | Meaning |
| :--- | :--- |
| `min_confidence` | threshold in `0 … 1`; below it, `fallback` runs or [`LowConfidenceError`](@ref) is thrown |
| `decide` | a policy called with the slot's [`ChoiceAnswer`](@ref) — one callable for every slot, or a `Vector` with one per slot — returning an offered meaning, or `nothing` to decline: `fallback` runs or [`DecisionDeclinedError`](@ref) is thrown. Excludes a nonzero `min_confidence` (in `@branch`, any `min_confidence`) |
| `fallback` | called with the caller's `args...` when the policy declines |
| `instructions` | a `String` for every slot, or a `Vector` with one entry per slot; default is the generic instruction above |
| `state` | send this instead of the object built from the ordinary arguments |
| `model` | pin a version, e.g. `"jev-1.13.0"`, instead of the `jev-latest` alias |
| `service` | endpoint type; defaults to [`TYPESAFEServiceEndpoint`](@ref UniLM.TYPESAFEServiceEndpoint) |
| `config` | a [`RequestConfig`](@ref) governing timeouts and the retry budget for this call |
| `cancel` | a [`CancelToken`](@ref) that can cancel the call from another task; `nothing` (the default) uses the ambient [`with_cancel`](@ref) token. A cancelled call throws `SystemOneError` like any failed call |

Pin the model before you tune a threshold. A threshold is a claim about one
version, and `jev-latest` moves when a new one ships
([Models](https://docs.typesafe.ai/models)); [`SystemOneResponse`](@ref)`.model`
reports the version that actually answered. And read `meanings(f)` first — a
threshold that keeps tripping is usually an option list with two overlapping
meanings in it, not a number that needs lowering.

A call that failed outright throws [`SystemOneError`](@ref) rather than
resolving to a method, so a timeout or a 500 can never be mistaken for a
routing decision.

## [Definition order is an input](@id nl_dispatch_order)

Definition order is the order the model reads the options in, and on ambiguous
inputs it moves the probabilities. Here one request asks the same question twice
about a ticket that fits two meanings — once with the options as defined, once
reversed:

```@example nldispatch
triage(::nl"the customer wants their money back", ticket)                                     = :refund
triage(::nl"the customer was charged twice or charged an amount they did not expect", ticket) = :billing_error
triage(::nl"the customer asks for an invoice, receipt, or billing document", ticket)          = :invoice
triage(::nl"the app crashes, freezes, or shows an error", ticket)                             = :crash
triage(::nl"the customer cannot sign in to their account", ticket)                            = :login
triage(::nl"the order is late or has not arrived", ticket)                                    = :late
triage(::nl"the order arrived damaged or broken", ticket)                                     = :damaged
triage(::nl"anything else", ticket)                                                           = :other

defined = meanings(triage)[1]    # what nl_dispatch(triage, ticket) offers, in this order
question(options) = choice("Select the option that best describes the provided state.", options)

r = ask((ticket = "There is a charge from your company on my card that I don't recognize at all.",),
        "as defined" => question(defined), "reversed" => question(reverse(defined)))
for order in ("as defined", "reversed")
    p = r[order].probabilities
    println(rpad(order, 12), "money back ", p[defined[1]], "   charged twice or unexpectedly ", p[defined[2]])
end
```

Measured on jev-1.13.0 (September 2026) on our own labeled sets, in the form
`nl_dispatch` sends: across seven rotations of the option order, the routed
method changed for 4 of 33 ambiguous tickets, and the first-listed meaning
gained about 3 percentage points on average; clear-cut inputs did not move
(≈ 0.01). When we measured the ticket above, reversing the order took the
money-back meaning from P = 0.55 to 0.04 and gave "charged twice or
unexpectedly" 0.92. A flip can land on a confident answer: a confidence gate
declines only the shaky side of it. The defences are in the option list itself:

- fix ambiguity in the meanings, not in the order — distinct, non-overlapping
  meanings leave the order nothing to decide;
- choose the order deliberately and keep it stable once thresholds are tuned: a
  new method, or a redefined one, changes it.

## `@branch`: the inline form

When the decision is local and nobody needs to extend it, [`@branch`](@ref) is
the same request without the method table. It is a `switch` whose cases are
written in plain language:

```@example nldispatch
ticket = "My package arrived crushed and the screen is cracked. I want my money back."

action = @branch ticket min_confidence=0.6 begin
    "the customer wants a refund"                              => :refund
    ("the customer reports a bug", "A defect in the software") => :bug
    "the customer asks a pricing question"                     => :pricing
    _                                                          => :escalate
end
```

Every line is `option => expression`:

- A bare left-hand side is the option **name** — the text the model reads and the
  text the answer returns. Left-hand sides are ordinary expressions, evaluated
  once in source order, so a name may be computed.
- The 2-tuple form `("name", "description")` attaches a longer description to
  that name. It is the only way to do so.
- A final `_ => expression` is the fallback taken when the decision policy
  declines: the winner's confidence is below `min_confidence`, or `decide`
  returned `nothing`. It *requires* a policy: without one nothing is ever
  declined and the line could never run. A block that names neither keyword is
  rejected while the macro expands; one whose keyword sets no policy when it
  runs — `decide = nothing`, or a `min_confidence` that is not positive — is
  refused then, before any request.

Keywords go between the state and the block, written `key = value`: `model`,
`min_confidence`, `decide`, `instructions`, `service`, `config` and `cancel`,
each meaning what it means for [`nl_dispatch`](@ref), except that `decide`
excludes any `min_confidence`, zero included. Here `decide` is one callable, and
it returns an option name — any of them, not only the winner — or `nothing` to
take `_` (with no `_` line, [`DecisionDeclinedError`](@ref) is thrown).

The whole block compiles to a **single** Choice request whose question is named
`branch` and whose criteria are the option names, in source order. Only the
selected body is evaluated — the others are not merely discarded, they never
run — and the macro's value is that body's value. The macro checks its syntax
while it expands, so these mistakes surface when the surrounding code is loaded
rather than the first time the branch is reached: an unknown keyword, a block
that is not `begin ... end`, a line that is not `option => expression`, no
options at all, two `_` lines, a `_` that is not last, or `decide` together with
`min_confidence`. The option names are values, evaluated when the branch runs,
so a duplicate or empty name — even a literal one — is refused then, still
before any request. A non-success call throws [`SystemOneError`](@ref); the
branch is never guessed.

**Which to reach for.** `@branch` when the decision belongs to one call site and
the arms are three lines of code — it keeps the options and their handling in
one place, and it needs no function at all. [`nl_dispatch`](@ref) when the
decision is an interface: the arms are real methods that can be tested and
called directly, other modules can add cases, the ordinary arguments keep
dispatching on their types, and [`meanings`](@ref) gives you the option list as
data.

## Testing without the API

The point of putting the meanings in the signature is that the methods stay
ordinary. A unit test calls them directly, resolving nothing and spending
nothing:

```@example nldispatch
t = "My package arrived crushed and the screen is cracked. I want my money back."

(route(nl"the customer wants a refund"(), t),
 route(nl"the customer reports a bug in the app"(), t),
 route(Meaning("anything else"), t))
```

That covers the method bodies. Coverage needs no service either:
[`meaning_gaps`](@ref) is the check `nl_dispatch` runs before its request. The
*resolution* — question names, option order, the state on the wire, the policy —
needs one real answer: record it once (`mode = :record_missing`, with a key) and
replay it in the test with [`with_recorded_answers`](@ref):

```julia
using Test, UniLM

@testset "routing" begin
    t = "My package arrived crushed and the screen is cracked. I want my money back."
    @test route(nl"the customer wants a refund"(), t) === :refund   # the method
    @test isempty(meaning_gaps(reply, Tuple{String}))               # the coverage
    with_recorded_answers(joinpath(@__DIR__, "answers")) do          # the resolution, replayed
        @test nl_dispatch(route, t) === :refund
    end
end
```

A replayed answer is the one the service gave to exactly those request bytes,
and a request with no recording throws [`ReplayMissError`](@ref) instead of
reaching the network. [Developing and Testing with Jev](@ref jev_testing_guide)
covers recording, and crafted edge cases — a low confidence, a failed call —
answered by a local mock server.

## Designing meanings

- **Write the meaning as a description, not a label.** It is the option name and
  it is the only text the model gets for that branch. `nl"the customer wants a
  refund"` works; `nl"refund"` asks the model to guess what you meant by the
  word. Measured on jev-1.13.0 (September 2026) on our own labeled sets, bare
  labels were 10 points less accurate than sentences (0.877 against 0.981). A
  short key with the sentence attached as its description was confidently wrong
  on keyword traps (an example is in the [design note](@ref nl_dispatch_design)
  below), and 10% of the short-key answers at confidence 0.8 or more were wrong,
  against 0% for sentences. Structured descriptions did not beat plain sentences
  on ordinary traffic.
- **An edited sentence is a new method.** Edit a meaning's sentence in a running
  session and the method with the old sentence still exists, so `meanings(f)`
  offers both until you delete it or restart Julia. Read the list: a duplicate
  meaning split one ticket's probability from 1.0 to 0.51 in our measurements.
- **Keep them mutually exclusive.** Two meanings that overlap split the
  probability between them, which shows up as low confidence rather than as a
  wrong answer — check `meanings(f)` and read the list as the model will.
- **Always include an "anything else".** Without one, an input outside the
  enumeration has nowhere to put its probability except onto a meaning that does
  not fit.
- **Keep the state to what matters.** Accuracy falls as unrelated material grows
  around the decision, so filter and retrieve in code and pass only the fields
  the decision needs — or pass `state = ...` explicitly.
- **The model reads literally and does no arithmetic.** Negations, scoping words
  and implied conditions are taken at face value, and counting, numeric magnitude
  and date ordering are unreliable — keep those in Julia and let the meanings
  carry only the semantic judgment ([Jev 1.13
  jaggedness](https://docs.typesafe.ai/model-jaggedness/jev-1.13)).
- **Methods are free; requests are not.** Ten methods and three methods cost the
  same single request, and the extra cost of a meaning is the tokens of its own
  description. Only input tokens are billed
  ([Models](https://docs.typesafe.ai/models)).
- **255 meanings per slot is the ceiling.** That is the server's Choice limit,
  and [`choice`](@ref) checks it locally before the round trip
  ([Choice](https://docs.typesafe.ai/primitives/choice)).

## [Design note: meanings are sentences, not keys](@id nl_dispatch_design)

The question every Julia reader asks: why not `route(::Val{:refund}, t)`, with
the text kept in `Dict(:refund => "the customer wants a refund")`? A key
separates a meaning's identity from its wording, stays short while the wording
changes, and dispatches as fast as any type. Three findings decided against it.

**The model reads the option name.** A Choice sends each option's name, and the
model reads it along with any description, so a key sent as the name is part of
the prompt. Here the same eight descriptions are offered once as the option
names and once under short keys (`defined` and `question` come from [Definition
order is an input](@ref nl_dispatch_order)):

```@example nldispatch
short = ["refund", "double_charge", "invoice", "crash", "login", "late", "damaged", "other"]

r = ask((ticket = "Is your refund policy the same for EU and US customers?",),
        "sentences" => question(defined),             # what nl_dispatch sends
        "keys"      => question(short .=> defined))   # a key per option, the sentence as its description
[(form, r[form].choice, r[form].confidence) for form in ("sentences", "keys")]
```

When we measured this ticket, the sentences answered "anything else" at
confidence 0.97 and the keys answered `refund` at confidence 0.6: the word in
the question matched the key. On keyword traps in general, short keys were wrong
more often, and confidently (the numbers are under Designing meanings above).

**A separate text registry drifts from the methods.** With `Val` keys the
wording lives in a `Dict` beside the method table, and nothing keeps the two in
step: a method can lack an entry, and an entry can outlive its method. A package
that adds entries to another module's `Dict` at top level does so while it
precompiles, and the writes are lost when the package is loaded from its cache.
A sentence in the signature has no second table to drift from.

Method collisions are the same for both spellings. `Val{:other}` and
`nl"anything else"` are each one type in every loaded package, so two packages
that both add `handle(::nl"anything else", ::String)` to a third package's
`handle` define one method twice, and the one loaded last wins, silently. A
module that owns neither the function nor an argument type is committing type
piracy with either spelling. The remedy for both is a type the module owns.

**Speed is not a reason.** Measured: resolving the options and dispatching took
≈ 39 µs of local work, uncached, against ≈ 300 ms for the request (0.013%), and
dispatch on a `Val` and on a `Meaning` cost the same, ≈ 0.12 µs. `nl_dispatch`
caches that plan — the options, argument names and gaps — per world, so after
the first call it is looked up, not recomputed.

Identity by sentence is content-addressed: two modules that write the same
sentence mean the same thing, and an edited sentence is a new meaning. When you
want short handles and a hierarchy, use module-scoped types — `Refund` in one
package and `Refund` in another are different types — and send the sentence as
the option name; the taxonomy section of [Semantic Programs with
Jev](@ref jev_programs_guide) builds one.

## See also

- [Typed Judgments with Jev (TypeSafe System One)](@ref system_one_guide) — the
  `ask` verb, the three primitives, and the setup this page assumes
- [Semantic Programs with Jev](@ref jev_programs_guide) — decision policies,
  taxonomies and conversation state machines built on these answers
- [Semantic Algorithms with Jev](@ref jev_algorithms_guide) — many judgments per
  request over collections: routing, ranking, search and joins
- [Developing and Testing with Jev](@ref jev_testing_guide) — recorded answers,
  coverage checks and mock servers for tests
- [TypeSafe System One API (Jev)](@ref system_one_api) — every type and verb,
  including [`Meaning`](@ref), [`@branch`](@ref) and [`nl_dispatch`](@ref)
- [TypeSafe documentation](https://docs.typesafe.ai) — the service's own
  reference for [Choice](https://docs.typesafe.ai/primitives/choice),
  [confidence](https://docs.typesafe.ai/confidence) and
  [models](https://docs.typesafe.ai/models)
