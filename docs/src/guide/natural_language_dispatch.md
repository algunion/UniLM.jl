# [Dispatch on Meaning](@id nl_dispatch_guide)

A Julia method runs by the types of its arguments. With Jev, one argument can be
chosen by what a text means: write one method per meaning, and one request picks
the method whose meaning fits the text. The model only ever chooses among the
sentences you wrote, and every choice comes with its probabilities.

## [Pick a pathway](@id nl_dispatch_paths)

| I want to… | Use | You get | Section |
| :--- | :--- | :--- | :--- |
| turn a text into one of my own values | [`nl_classify`](@ref)`(text, TEAM)` | the key, e.g. `:billing` | [Keys and tables](@ref nl_dispatch_keyed) |
| run the right method, short keys in the signatures | [`nl_dispatch`](@ref)`(route, text; texts = TEAM)` | what `route(Val(:billing), text)` returns | [Keys and tables](@ref nl_dispatch_keyed) |
| run the right method, the sentence in the signature | [`nl_dispatch`](@ref)`(route, text)` | what the method whose sentence fits returns | [Sentences in signatures](@ref nl_dispatch_literal) |
| branch once, inline | [`@branch`](@ref)` text begin … end` | the value of the chosen line | [`@branch`](@ref nl_dispatch_branch) |

**Which one?** Keys and sentences are two spellings of one request; they differ
in where the sentences live.

| | Keys: `texts = TEAM` | Sentences: `nl"…"` |
| :--- | :--- | :--- |
| Where a sentence lives | in one table, apart from the methods | in its method's signature |
| Option order | table order | definition order |
| Editing a sentence | edit the table; the methods stay as they are | defines a new method; the old one stays until you restart Julia |
| Typed keys, hierarchies | Symbols, your own types or instances, enum values; keys can form a type hierarchy | none: each sentence is a type of its own |
| What the model reads | the sentences | the same sentences, byte for byte |

## The sixty-second version

- **The job:** send each support message to the team that handles it.
- **Without Jev:** keyword rules that miss rephrasings, a classifier to train and
  retrain, or an LLM reply to parse and validate.
- **With Jev:** one table of what each team handles, one method per team, one
  request per message.

```@example nldispatch
using UniLM

const TEAM = (billing   = "payments, charges, invoices or refunds",
              technical = "the app or the website does not work as expected",
              shipping  = "a parcel that is late, lost or arrived damaged",
              other     = "anything else")

message = "I was charged twice for order #4471. Please refund the duplicate payment."

nl_classify(message, TEAM)
```

The model read the four sentences, never the keys, and the key of the sentence it
chose came back. To run code instead, write one method per key:

```@example nldispatch
route(::Val{:billing}, message)   = "opened a payment review"
route(::Val{:technical}, message) = "filed a bug report"
route(::Val{:shipping}, message)  = "opened a claim with the carrier"
route(::Val{:other}, message)     = "forwarded to the front desk"

nl_dispatch(route, message; texts = TEAM)
```

The same router with each sentence in its signature, and no table:

```@example nldispatch
route_by_sentence(::nl"payments, charges, invoices or refunds", message)           = "opened a payment review"
route_by_sentence(::nl"the app or the website does not work as expected", message) = "filed a bug report"
route_by_sentence(::nl"a parcel that is late, lost or arrived damaged", message)    = "opened a claim with the carrier"
route_by_sentence(::nl"anything else", message)                                    = "forwarded to the front desk"

nl_dispatch(route_by_sentence, message)
```

Same answer: both calls put the same request on the wire, byte for byte (How it
works, below, shows it), so the model was asked the same question.

**Tune it**

- Write each sentence as a description of the case, not a one-word label
  ([Designing meanings](@ref "Designing meanings")).
- Keep an "anything else" option, so a message that fits no team has somewhere
  to go.
- Hand unclear messages to a person with `min_confidence` and `fallback`
  ([Confidence and control](@ref nl_dispatch_control)).

## [Keys and tables](@id nl_dispatch_keyed)

- **The job:** keep what the model reads in one table that a reviewer or a
  translator can read at a glance, while the code dispatches on short names.
- **Without Jev:** a dictionary from label to handler, and a prompt that lists
  the labels, kept in step by hand.
- **With Jev:** pass the table as `texts`; the request is built from it, the keys
  never leave your process, and a key with no method is refused before anything
  is sent.

A table is ordered, and its order is the order the model reads the options in:

- a `NamedTuple` of sentences, like `TEAM`: each `Symbol` key reaches its method
  as `Val(key)`, so `route(::Val{:billing}, message)` takes `:billing`;
- a vector of `key => sentence` pairs whose keys are your own values — an enum
  value, a singleton instance such as `Refund()`, or a type such as `Refund` —
  each passed as written, so a method on the enum type, on `::Refund` or on
  `::Type{<:Billing}` takes it through ordinary dispatch
  ([Hierarchies](@ref nl_dispatch_hierarchy)).

A `Dict` is refused, because it has no order. A table holds 1 to 255 entries, the
options one Choice question takes, with no sentence and no key twice. Enum values
make [`nl_classify`](@ref) return one of your own values:

```@example nldispatch
@enum Team billing technical shipping other

nl_classify(message, [billing   => TEAM.billing,
                      technical => TEAM.technical,
                      shipping  => TEAM.shipping,
                      other     => TEAM.other])
```

`nl_classify` returns the key as the table writes it — `:billing` for a
`NamedTuple`, not `Val(:billing)` — and sends the text as given, in one question
named `"classify"`. Below a `min_confidence` it calls `fallback(text)` when you
pass one, and throws otherwise, as `nl_dispatch` does.

**Preview.** [`meanings`](@ref) takes the same table, with no request:
`meanings(f; texts)` lists every sentence per position, and
`meanings(f, argtypes; texts)` what a call with arguments of those types offers:

```@example nldispatch
meanings(route, Tuple{String}; texts = TEAM)[1]   # position 1, for a String message
```

**Where a table goes.** A table fills the one position whose declared type its
keys have — `::Val{:billing}` above; `::Any` declares none — so
`nl_dispatch(route, message; texts = TEAM)` calls a method of two positional
arguments: one per table, plus the arguments you pass. Several keyed positions
take a `Tuple` of tables, one per position. A call uses keys or sentences, not
both: a method of the same arity with an `nl"…"` in its signature makes a keyed
call an `ArgumentError`, which is why the sentence router above has a name of its
own.

**The drift check.** Every key must reach a method at its position, whatever the
other arguments are. A key that no method takes is refused before any request,
with the method to write:

```@example nldispatch
RETURNS = (; TEAM..., returns = "the customer wants to send an item back")   # a new team, no method yet

try
    nl_dispatch(route, message; texts = RETURNS)
catch err
    println(sprint(showerror, err))
end
```

A catch-all method, `route(key, message)`, takes every key; write one only when
you want that backstop. A `decide` policy may return an offered sentence, the key
of one as the table writes it (`:billing`, not `Val(:billing)`), or `nothing` to
decline ([Confidence and control](@ref nl_dispatch_control)).

**Tune it**

- Order the table deliberately, and keep the order once a threshold is tuned: it
  is the option order.
- Check `meanings(f, argtypes; texts)` and `meaning_gaps(f, argtypes; texts)` in
  a unit test; neither sends a request.
- Add a key together with its method: the drift check refuses the table until
  the method exists.

## [Sentences in signatures](@id nl_dispatch_literal)

- **The job:** keep each sentence right next to the code it triggers.
- **Without Jev:** a dictionary from sentence to handler, kept in step with the
  handlers by hand.
- **With Jev:** `nl"…"` is a Julia type, so the sentence goes into the signature
  itself, and there is nothing else to keep in step.

`nl"the sentence"` is the type [`Meaning`](@ref)`{Symbol("the sentence")}`, a
singleton with no fields, so it reads as a type anywhere one is expected: in a
signature, a `const` alias or an `isa` test. Append `()` for the instance, or
build one from a runtime string with `Meaning(text)`:

```@example nldispatch
const Payments = nl"payments, charges, invoices or refunds"    # a type alias

(Payments, Payments(), Meaning(TEAM.billing) isa Payments)
```

Two identical sentences are the same type, so `===` compares them.
[`nl_dispatch`](@ref) collects the methods that pin at least one position to one
concrete sentence; a wildcard `f(::Meaning, x)`, or a method with no meaning, is
an ordinary method and is skipped. The collected methods must agree on their
number of positional arguments and on which positions hold meanings, because one
set of questions serves them all: a disagreement is an `ArgumentError` naming
both methods, before any request. Varargs cannot carry a meaning.

### [Definition order is an input](@id nl_dispatch_order)

The options follow definition order: the order in which the methods were defined
in a file, a script or a REPL session, which [`meanings`](@ref) previews. Three
things follow from it:

- Redefining a method, by editing its body in a running session for instance,
  moves its sentence to the end.
- An edited sentence is a new method. The method with the old sentence still
  exists, so `meanings(f)` offers both until you delete it or restart Julia, and
  two sentences that mean the same split the probability between them.
- A package loaded from its precompiled image defines its methods together, and
  they are ordered by source file path, then line — not `include` order. Keep
  one function's sentences in one file.

Order matters because the model reads it, and on an ambiguous text it moves the
probabilities. Here one request asks the same question twice about a ticket that
fits two meanings, once with the options as defined and once reversed:

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

Reversing the order took "money back" from 0.52 to 0.03 and gave "charged twice
or unexpectedly" 0.93. A flip can land on a confident answer, so a confidence gate
declines only the shaky side of it. A table's order moves answers the same way.

!!! details "Evidence"
    Measured on jev-1.13.0 (September 2026) on our own labeled sets, in the form
    `nl_dispatch` sends: across seven rotations of the option order, the routed
    method changed for 4 of 33 ambiguous tickets, and the first-listed meaning
    gained about 3 percentage points on average; clear-cut inputs did not move
    (≈ 0.01). When we measured the ticket above, reversing the order took the
    money-back meaning from P = 0.55 to 0.04 and gave "charged twice or
    unexpectedly" 0.92. A duplicate meaning split one ticket's probability from
    1.0 to 0.51.

**Tune it**

- Fix ambiguity in the sentences, not in the order: distinct sentences that do
  not overlap leave the order nothing to decide.
- Choose the order deliberately, and keep it stable once thresholds are tuned: a
  new method, or a redefined one, changes it.
- Read `meanings(f)` after editing a sentence, and restart Julia to drop the old
  method.

## How it works

Both pathways build the same request. For the sixty-second version it is exactly
this body, indented here — captured off the wire by pointing both calls at a
local server instead of `api.typesafe.ai`; the keyed call and the sentence call
sent the same bytes:

```json
{
  "state": {
    "message": "I was charged twice for order #4471. Please refund the duplicate payment."
  },
  "model": "jev-latest",
  "questions": {
    "meaning_1": {
      "type": "choice",
      "instructions": "Select the option that best describes the provided state.",
      "criteria": {
        "payments, charges, invoices or refunds": null,
        "the app or the website does not work as expected": null,
        "a parcel that is late, lost or arrived damaged": null,
        "anything else": null
      }
    }
  }
}
```

One [`choice`](@ref) question per keyed or meaning position, all in one
[`ask`](@ref):

- The **question name** is the position's argument name, or `meaning_<position>`
  when it is unnamed. Question names key the answers and are never sent to the
  model.
- The **options** are the sentences, with no description attached: the sentence
  you write is the whole prompt for its option. Only the options the call can
  reach are offered ([Composing with ordinary dispatch](@ref nl_dispatch_composing)),
  in table order or definition order.
- The **state** is a JSON object keyed by the names of the other arguments
  (`arg<position>` for an unnamed one), in argument order, so the same arguments
  always make the same request. `state = ...` sends something else instead, and
  is required when there are no other arguments.
- The **instructions** default to *"Select the option that best describes the
  provided state."*; the options carry the meaning. `instructions = ...`
  overrides them.
- The **model** is the client default (`jev-latest`, or whatever
  `TYPESAFE_DEFAULT_MODEL` names) unless `model = ...` pins one for the call.

The model sees the state, the instructions and the sentences. It does **not** see
the keys, the function's name, the method bodies, the question name, or the fact
that this is dispatch at all: to Jev it is one Choice question like any other
([Choice](https://docs.typesafe.ai/primitives/choice)).

**The answer.** Each answer is a [`ChoiceAnswer`](@ref) whose `choice` is one of
the offered sentences by construction. A decision policy picks the sentence to
act on — by default the winner, gated by `min_confidence` ([Confidence and
control](@ref nl_dispatch_control)). The sentence becomes its key's value
(`Val(:billing)` for the key `:billing`) or its `Meaning` instance, goes back into
the position it came from, the other arguments fill the rest in order, and Julia
makes an ordinary call. From there nothing is special: the value is whatever the
selected method returns.

## [Composing with ordinary dispatch](@id nl_dispatch_composing)

- **The job:** a support chat that, at each step, only considers the replies
  that make sense at that step.
- **Without Jev:** a state machine whose transitions are keyword rules, or an
  intent classifier plus a hand-kept list of which intents each state allows.
- **With Jev:** each state is a type and each transition a method on the state
  and a key; `nl_dispatch` offers the model only the keys whose method accepts
  the current state.

A chosen key is an ordinary argument, so the other positions keep dispatching on
their types, and their types decide which options are offered at all. Keys suit
a state machine: each line reads *state, reply → next state*, and the six
sentences the model reads sit together in one table.

```@example nldispatch
abstract type Conversation end
struct AwaitingOrderId <: Conversation end
struct ConfirmingRefund <: Conversation
    order::String
end
struct Closed <: Conversation
    outcome::Symbol
end
# A state prints as its constructor call, e.g. Closed(:refunded).
Base.show(io::IO, s::Conversation) =
    print(io, nameof(typeof(s)), "(", join((repr(getfield(s, f)) for f in fieldnames(typeof(s))), ", "), ")")

asked(::AwaitingOrderId) = "Which order would you like refunded?"
asked(s::ConfirmingRefund) = "Shall I refund order $(s.order)? Please confirm."
asked(::Closed) = "Anything else I can help with?"
order_number(msg) = match(r"\d{4,}", msg).match      # the value is read by code, not by the model

const REPLY = (order = "The customer gives an order number",
               yes   = "The customer confirms the refund",
               no    = "The customer declines or changes their mind",
               bye   = "The customer says goodbye or thanks",
               human = "The customer asks for a human agent",
               other = "Something else")

transition(::AwaitingOrderId,  ::Val{:order}, msg) = ConfirmingRefund(order_number(msg))
transition(::ConfirmingRefund, ::Val{:yes}, msg)   = Closed(:refunded)
transition(::ConfirmingRefund, ::Val{:no}, msg)    = Closed(:kept)
transition(::Closed,           ::Val{:bye}, msg)   = Closed(:done)
transition(::Conversation,     ::Val{:human}, msg) = Closed(:handoff)
transition(s::Conversation,    ::Val{:other}, msg) = s

println(length(REPLY), " replies in all; offered while confirming a refund:")
foreach(println, meanings(transition, Tuple{ConfirmingRefund, String}; texts = REPLY)[2])
```

Each turn is one request whose state is what the assistant asked and what the
customer replied:

```@example nldispatch
context(s, msg) = (assistant_asked = asked(s), customer_replied = msg)

let s = AwaitingOrderId()
    for msg in ("It's 48213.", "thanks, bye", "yes please", "no, that's all")
        s = nl_dispatch(transition, s, msg; texts = REPLY, state = context(s, msg))
        println(rpad(repr(msg), 18), "→ ", s)
    end
end
```

A call offers only the options whose method accepts the types of its other
arguments, and `meanings(f, Tuple{…}; texts)` lists them for the types you name,
exactly as they will be sent; types that no method accepts are an
`ArgumentError` before any request. The model picks among the reachable options,
and Julia picks the method afterwards, with no extra request. Values the model
must not guess are read by code: the model decides that the customer gave an
order number, and `order_number` parses it.

!!! details "Evidence"
    Measured on 30 live turns of a larger version of this machine (nine meanings):
    offering every meaning in every state ended 7 turns in a `MethodError` — 3 of
    them at confidence 1.00 — and got 22 transitions right; offering only the
    meanings valid in the current state removed every `MethodError` and got 29
    right.

**Tune it**

- Put the replies every state accepts on the abstract type (`::Conversation`
  above), and keep an "anything else" reply that leaves the state as it is.
- The number of options now depends on the state, and so does what a fixed
  `min_confidence` means ([Confidence and control](@ref nl_dispatch_control)).
- Another module can add a state type of its own, with methods for the keys in
  the table (see ownership in the [design note](@ref nl_dispatch_design)).

## [Hierarchies](@id nl_dispatch_hierarchy)

- **The job:** send each ticket to the most specific desk the model is sure of:
  the refund flow for a clear refund, the billing desk for a billing problem of
  unclear kind, a person when nothing is clear.
- **Without Jev:** one classifier per level of the tree, or one flat classifier
  plus a hand-kept map from each case to its desk.
- **With Jev:** the tree is a Julia type hierarchy, the table names its leaves,
  and each desk is a method on a node; Julia picks the most specific one.

```@example nldispatch
abstract type Intent end
abstract type Billing <: Intent end
abstract type Technical <: Intent end
struct Refund <: Billing end
struct DoubleCharge <: Billing end
struct Crash <: Technical end
struct Login <: Technical end
struct Other <: Intent end

# The leaves the model chooses between, in the order it reads them.
const LEAVES = [Refund       => "the customer wants their money back",
                DoubleCharge => "the customer was charged twice or charged an amount they did not expect",
                Crash        => "the app crashes, freezes, or shows an error",
                Login        => "the customer cannot sign in to their account",
                Other        => "anything else"]

# A desk for any node of the tree; Julia picks the most specific one.
desk(::Type{<:Intent}, ticket)    = :human_triage
desk(::Type{<:Billing}, ticket)   = :billing_desk
desk(::Type{<:Technical}, ticket) = :tech_desk
desk(::Type{Refund}, ticket)      = :refund_flow

nl_dispatch(desk, "Please refund my annual plan, I cancelled on day two."; texts = LEAVES)
```

No table maps leaves to desks. A double charge has no desk of its own, so it
reaches `desk(::Type{<:Billing}, ticket)` by Julia's specificity rules, and
"anything else" reaches a person. The keys are the types themselves, not
instances, so that a branch — an abstract type, which has no instance — can be
handed to the same desks.

**When the leaf is unsure but its branch is not.** Add each leaf's probability to
every node above it, and act on the most specific node that reaches the bar. A
`decide` policy can return only a key the table offered, and a branch is not one,
so this rule reads the answer itself — from [`ask`](@ref), with the options and
instructions `nl_dispatch` sends — and hands the node to `desk`:

```@example nldispatch
const INTENT = choice("Select the option that best describes the provided state.", last.(LEAVES) .=> nothing)

# The most specific node whose leaves reach the bar together, and their probability.
function most_specific(answer::ChoiceAnswer; bar = 0.7)
    mass(node) = sum(answer.probabilities[text] for (leaf, text) in LEAVES if leaf <: node)
    nodes = [first.(LEAVES); Billing; Technical]              # most specific first
    i = findfirst(node -> mass(node) >= bar, nodes)
    node = isnothing(i) ? Intent : nodes[i]
    (node, round(mass(node); digits = 2))
end

for ticket in ("Please refund my annual plan, I cancelled on day two.",
               "Two subscriptions renewed on my card. Please cancel one and return that payment.",
               "Since the update I can't get past the login screen, it just freezes.",
               "There is a charge from your company on my card that I don't recognize at all.")
    node, p = most_specific(ask((ticket = ticket,), "intent" => INTENT)["intent"])
    println(rpad(nameof(node), 13), rpad(p, 6), rpad(desk(node, ticket), 14), ticket)
end
```

For the last two tickets no single leaf reached the bar, but a branch did: they
went to the tech desk and the billing desk, where a bar on the leaf alone would
have sent them to a person.

!!! details "Evidence"
    Measured at threshold 0.7 on 65 labeled tickets, each asked with the leaves in
    four orders (260 decisions; correct / escalated / wrong): this rule
    250 / 7 / 3; flat leaf routing with the same threshold 238 / 19 / 3; top-down
    classification with one request per level 238 / 7 / 15. Summing up the tree
    turned 12 escalations into coarse routes, all 12 correct, and added no wrong
    one; asking level by level escalated as rarely but was wrong five times as
    often.

**Tune it**

- `bar` trades escalations for coarse routes; set it on labeled tickets of your
  own.
- A leaf's own desk is used only when that leaf alone reaches the bar;
  otherwise the ticket goes to its branch's desk, or to a person when no branch
  reaches the bar either.
- Add a desk at any level with one method, such as
  `desk(::Type{Login}, ticket) = :password_reset`.

## Several slots, one request

- **The job:** decide two things about one message — what it is and in what
  tone — at once.
- **Without Jev:** two classifier calls, or one LLM prompt that returns JSON you
  validate.
- **With Jev:** a signature with two meaning positions; one request carries one
  question per position.

```@example nldispatch
reply(::nl"a complaint", ::nl"a calm tone", msg)  = :apologise
reply(::nl"a complaint", ::nl"an angry tone", msg) = :escalate
reply(::nl"a question", ::nl"a calm tone", msg)   = :answer
reply(::nl"a question", ::nl"an angry tone", msg) = :answer_and_apologise

nl_dispatch(reply, "This is the third time I have written about my broken order.")
```

Both positions resolve in **one** request, as independent questions about the
same state: a second position costs the tokens of its options, not a second read
of the message ([parallel
questions](https://docs.typesafe.ai/cookbooks/parallel_questions)). With keys,
pass a `Tuple` with one table per keyed position, `texts = (KIND, TONE)`: each
table goes to the position whose declared type its keys have.

**Every combination needs a method.** An answer can land on any combination of
the offered options, and one with no method would be billed before Julia raised
its `MethodError`. So `nl_dispatch` checks every combination first and refuses a
gap with nothing sent; [`meaning_gaps`](@ref) runs the same check with no call,
in a unit test for instance. A catch-all added for one tone only opens a gap:

```@example nldispatch
reply(::nl"anything else", ::nl"a calm tone", msg) = :triage   # no method yet for an angry "anything else"

println(meaning_gaps(reply, Tuple{String}))                    # the message is a String
try
    nl_dispatch(reply, "This is the third time I have written about my broken order.")
catch err
    println(sprint(showerror, err))
end
```

Close it with the missing method:

```@example nldispatch
reply(::nl"anything else", ::nl"an angry tone", msg) = :triage

meaning_gaps(reply, Tuple{String})
```

A method wild in every position, `reply(::Meaning, ::Meaning, msg)`, is the
other way out: it is never offered, and ordinary dispatch reaches it for every
combination that has no method. Prefer the catch-all sentence: it comes back with
a name and a probability, while a backstop cannot tell you why it fired, and once
it exists `meaning_gaps` cannot report the method you forgot. A method wild in
only some positions is refused with an `ArgumentError`: wildcard every position
or none. A combination that two methods match with neither more specific is a
gap too; the error names both and the method for their intersection that settles
it. With keys, a catch-all method such as `reply(kind, tone, msg)` is the
backstop.

**Tune it**

- Keep each position's options few and distinct: the combinations multiply.
- Run `meaning_gaps(f, argtypes)` in your tests.

## [Confidence and control](@id nl_dispatch_control)

- **The job:** act on the messages the model is sure about, and hand the rest to
  a person.
- **Without Jev:** a classifier score you calibrate yourself, or an LLM that
  sounds just as sure when it is guessing.
- **With Jev:** every answer carries probabilities; a threshold or a policy of
  your own decides, and a fallback takes the rest.

A day's inbox, routed with a threshold:

```@example nldispatch
const MESSAGES = [
    "I was charged twice for order #4471. Please refund the duplicate payment.",
    "The app crashes every time I open my order history. iPhone 15, latest version.",
    "My parcel was due last Monday and the tracking hasn't moved in six days.",
    "Do you ship to Norway, and how much does delivery cost?",
    "The phone I received has a cracked screen and the box was crushed. I want my money back.",
    "Your update deleted my saved addresses and now I can't check out. Third time this month!",
]

for m in MESSAGES
    action = nl_dispatch(route, m; texts = TEAM, min_confidence = 0.7, fallback = m -> "held for a person")
    println(rpad(action, 33), m)
end
```

Five messages went to a team. The question about shipping to Norway fell below
the threshold, so `fallback` ran with the caller's own arguments and held it for
a person. With no `fallback`, the call throws [`LowConfidenceError`](@ref), which
holds the whole [`ChoiceAnswer`](@ref), so a handler can read `probabilities`.

A Choice always names a winner, even when the probabilities are nearly flat, so
the gate is `confidence` rather than `choice`. For `n` options, `confidence` is
`clamp((n·p_max − 1)/(n − 1), 0, 1)`: the top probability `p_max`, rescaled so
that a flat answer scores 0 and a certain one 1. A fixed `min_confidence` is
therefore a different bar on the top probability for every number of options —
0.5 means `p_max ≥ 0.75` with 2 options, 0.6 with 5 and 0.55 with 10 — and adding
a method, or calling with argument types that reach fewer, moves it. The
probabilities are calibrated across groups of answers, which is what makes a
threshold worth tuning, but a high confidence says the distribution is peaked,
not that the winner is correct ([Confidence](https://docs.typesafe.ai/confidence)).

!!! details "Evidence"
    The confidence formula fits every answer we measured within 0.02 (jev-1.13.0,
    September 2026, on our own labeled sets).

`decide` replaces the threshold with any policy over the whole answer. It is
called with the [`ChoiceAnswer`](@ref) and returns an offered sentence — any of
them, not only the winner — or, with a table, the key of one; or `nothing` to
decline, which runs `fallback` or throws [`DecisionDeclinedError`](@ref). A bar on
the top probability itself stays where you put it:

```@example nldispatch
sure(a) = a.probabilities[a.choice] >= 0.8 ? a.choice : nothing

[nl_dispatch(route, m; texts = TEAM, decide = sure, fallback = m -> "held for a person") for m in MESSAGES]
```

On this inbox both policies hold the same message. With the four teams,
`min_confidence = 0.7` is the bar `p_max ≥ 0.775`; `sure` stays at 0.8 whatever
the number of options. A policy can also prefer a cheaper action to a slightly
likelier one: [When mistakes have different prices](@ref jev_loss_matrix)
derives such a policy from what each mistake costs. `decide` together with a nonzero `min_confidence` is an
`ArgumentError` before any request, because a threshold is itself the policy
`a -> a.confidence >= τ ? a.choice : nothing`. Every position is decided before
anything runs, and a policy that returns anything else is an `ArgumentError`:
neither the method nor `fallback` runs.

`on_response` is for audits. It is called once with the
[`SystemOneSuccess`](@ref), before the policy runs, and gets what `decide` cannot
see: the request id to quote in a support report, the model version that
answered, and the raw body.

```@example nldispatch
audit(result) = println("audit: request ", result.response.request_id, ", answered by ", result.response.model)

println(nl_dispatch(route, message; texts = TEAM, on_response = audit))
```

A call that failed throws [`SystemOneError`](@ref) instead, and `on_response` is
not called: a timeout or a 500 can never be mistaken for a routing decision. The
keywords of [`nl_dispatch`](@ref):

| Keyword | Meaning |
| :--- | :--- |
| `texts` | a table of `key => sentence` entries, or a `Tuple` of them, one per keyed position ([Keys and tables](@ref nl_dispatch_keyed)) |
| `min_confidence` | threshold in `0 … 1`; below it, `fallback` runs or [`LowConfidenceError`](@ref) is thrown |
| `decide` | a policy called with the position's [`ChoiceAnswer`](@ref) — one callable for every position, or a `Vector` with one per position — returning an offered sentence, the key of one, or `nothing` to decline: `fallback` runs or [`DecisionDeclinedError`](@ref) is thrown. Excludes a nonzero `min_confidence` |
| `fallback` | called with the caller's `args...` when the policy declines |
| `on_response` | called once with the [`SystemOneSuccess`](@ref) before the policy runs; never for a failed call. Its return value is ignored; an exception from it propagates, and neither the method nor `fallback` runs |
| `instructions` | a `String` for every position, or a `Vector` with one entry per position; default is the generic instruction above |
| `state` | send this instead of the object built from the other arguments |
| `model` | pin a version, e.g. `"jev-1.13.0"`, instead of the `jev-latest` alias |
| `service` | endpoint type; defaults to [`TYPESAFEServiceEndpoint`](@ref UniLM.TYPESAFEServiceEndpoint) |
| `config` | a [`RequestConfig`](@ref) governing timeouts and the retry budget for this call |
| `cancel` | a [`CancelToken`](@ref) that can cancel the call from another task; `nothing` (the default) uses the ambient [`with_cancel`](@ref) token. A cancelled call throws `SystemOneError` like any failed call |

**Tune it**

- Pin the model before you tune a threshold: a threshold is a claim about one
  version, `jev-latest` moves when a new one ships
  ([Models](https://docs.typesafe.ai/models)), and
  [`SystemOneResponse`](@ref)`.model` reports the version that answered.
- Read `meanings(f, argtypes; texts)` first: a threshold that keeps tripping is
  usually two overlapping options, not a number that needs lowering.
- When mistakes cost different amounts, price them in a `decide` policy instead
  of tuning one threshold.

## [`@branch`: the inline form](@id nl_dispatch_branch)

- **The job:** make one decision at one place in the code, without defining a
  function.
- **Without Jev:** an `if`/`else` chain over keyword matches.
- **With Jev:** [`@branch`](@ref) is a `switch` whose cases are sentences: one
  request, and only the chosen line runs.

```@example nldispatch
ticket = "My package arrived crushed and the screen is cracked. I want my money back."

action = @branch ticket min_confidence=0.6 begin
    "the customer wants a refund"                              => :refund
    ("the customer reports a bug", "A defect in the software") => :bug
    "the customer asks a pricing question"                     => :pricing
    _                                                          => :escalate
end
```

Every line is `option => expression`. A bare left-hand side is the option name,
the text the model reads; the form `("name", "description")` attaches a
description to it. A final `_ => expression` runs when the policy declines, so it
needs `min_confidence` or `decide`. Keywords go between the state and the block,
written `key = value`: `model`, `min_confidence`, `decide`, `instructions`,
`service`, `config`, `cancel` and `on_response`, as for [`nl_dispatch`](@ref),
except that `decide` returns an option name and excludes any `min_confidence`,
zero included.

The block is a single Choice request whose question is named `branch`, with the
options in source order; only the selected expression runs, and its value is the
macro's value. Mistakes in the block's shape — an unknown keyword, a line that is
not `option => expression`, two `_` lines, a `_` that is not last, `decide`
together with `min_confidence` — are errors when the surrounding code is loaded.
A duplicate or empty option name, or a `_` whose policy cannot decline
(`decide = nothing`, or a `min_confidence` of 0 or less), is refused when the
branch runs, still before any request. A failed call throws
[`SystemOneError`](@ref); the branch is never guessed.

**`@branch` or `nl_dispatch`?** `@branch` when the decision belongs to one call
site and each arm is a line of code: the options and their handling sit in one
place, and no function is needed. [`nl_dispatch`](@ref) when the decision is an
interface: the arms are real methods that can be tested and called directly,
other modules can add cases, the other arguments keep dispatching on their types,
and [`meanings`](@ref) gives you the option list as data.

## Designing meanings

The advice holds for both pathways. "The sentence" is the text the model reads:
the table entry, or the `nl"…"` in the signature.

- **Write the sentence as a description, not a label.** It is the only text the
  model gets for that option. "the customer wants a refund" works; "refund" asks
  the model to guess what you meant, and bare labels were 10 points less accurate
  than sentences ([design note](@ref nl_dispatch_design)). A key is never read,
  so a short key costs nothing.
- **Keep them mutually exclusive.** Two sentences that overlap split the
  probability between them, which shows up as low confidence rather than as a
  wrong answer: read `meanings(f)` as the model will.
- **Always include an "anything else".** Without one, a text outside the list has
  nowhere to put its probability except on a sentence that does not fit
  ([Let Jev say none of these](@ref jev_none_of_these)).
- **Keep the state to what matters.** Accuracy falls as unrelated material grows
  around the decision, so filter and retrieve in code and pass only the fields
  the decision needs, or pass `state = ...` explicitly.
- **The model reads literally and does no arithmetic.** Negations, scoping words
  and implied conditions are taken at face value, and counting, numeric
  magnitude and date ordering are unreliable: keep those in Julia and let the
  sentences carry only the judgment ([Jev 1.13
  jaggedness](https://docs.typesafe.ai/model-jaggedness/jev-1.13)).
- **Methods are free; requests are not.** Ten methods and three methods cost the
  same single request, and the extra cost of an option is the tokens of its
  sentence. Only input tokens are billed
  ([Models](https://docs.typesafe.ai/models)).
- **255 options per position is the ceiling.** That is the server's Choice limit,
  and [`choice`](@ref) checks it locally before the round trip
  ([Choice](https://docs.typesafe.ai/primitives/choice)).

## [Design note: what the model reads](@id nl_dispatch_design)

Both pathways put each sentence on the wire as an option name, with nothing
else: the model reads the sentence, and only the sentence, for that option. That
is the most accurate encoding we measured.

!!! details "Evidence"
    Measured on jev-1.13.0 (September 2026) on our own labeled set, accuracy by
    what each option sends: the sentence as the option name 0.981; a short key
    with the sentence as its description 0.965; an opaque id with the sentence as
    its description 0.969; a bare label 0.877; a short key with a structured
    description 0.977. Structured descriptions did not beat plain sentences on
    ordinary traffic.

**Why the key must not be the option name.** A key sent as the option name is
part of the prompt, and a word in the text can match it. Here the same eight
sentences are offered once as the option names and once under short keys, with
each sentence as its key's description (`defined` and `question` come from
[Definition order is an input](@ref nl_dispatch_order)):

```@example nldispatch
short = ["refund", "double_charge", "invoice", "crash", "login", "late", "damaged", "other"]

r = ask((ticket = "Is your refund policy the same for EU and US customers?",),
        "sentences" => question(defined),             # what nl_dispatch sends
        "keys"      => question(short .=> defined))   # a key per option, the sentence as its description
[(form, r[form].choice, r[form].confidence) for form in ("sentences", "keys")]
```

As sentences, the question about a refund *policy* went to "anything else" at
confidence 0.98; under keys it went to `refund` at confidence 0.6, because the
word in the question matched the key. A `texts` table never sends its keys: the
chosen sentence maps back to its key locally, after the answer.

!!! details "Evidence"
    When we measured this ticket, the sentences answered "anything else" at
    confidence 0.97 and the keys answered `refund` at confidence 0.6. On keyword
    traps, a short key with the sentence attached as its description was
    confidently wrong: 10% of the short-key answers at confidence 0.8 or more
    were wrong, against 0% for sentences.

**What a key buys.** An identity that stays put while the wording changes: edit a
sentence, and the methods, tests and logs that name `:billing` stay as they are.
One table to review or translate, which is also the whole list of options: a
method alone adds none. An explicit option order. Typed values — your own types,
instances and enum values — and type hierarchies.

**What a sentence in the signature buys.** One place for everything, with nothing
to keep in step. Identity by content: two modules that write the same sentence
mean the same thing, and an edited sentence is a new meaning. An open list: the
options are the method table, so another module adds a case with one method.

**Drift.** A table can name a key that no method takes; `nl_dispatch` checks that
before every request and refuses it ([Keys and tables](@ref nl_dispatch_keyed)).
A method whose key is missing from the table is simply never offered. A sentence
in a signature cannot drift: there is no second place for it to live.

**Ownership.** Method collisions are the same for both spellings. `Val{:other}`
and `nl"anything else"` are each one type in every loaded package, so two
packages that both add `handle(::nl"anything else", ::String)` to a third
package's `handle` define one method twice, and the one loaded last wins,
silently. A module that owns neither the function nor an argument type is
committing type piracy with either spelling. The remedy for both is a type the
module owns: a package that does not own `handle` can add
`handle(::nl"a refund request", t::MyTicket)` for a `MyTicket` of its own, and
from the next call with a `MyTicket` on it is one more option the model may
choose.

**Speed.** Not a reason to pick either. Both pathways cache their plan — the
options, the argument names and the gaps — per state of the method table (and,
with a table, per its contents), so after the first call it is looked up, not
recomputed; measured with sentences in signatures, even the uncached work was a
rounding error next to the request.

!!! details "Evidence"
    Measured with sentences in signatures: resolving the options and dispatching
    took ≈ 39 µs of local work, uncached, against ≈ 300 ms for the request
    (0.013%), and dispatch on a `Val` and on a `Meaning` cost the same,
    ≈ 0.12 µs.

## Testing without the API

The methods stay ordinary: `route(Val(:billing), message)` or
`route_by_sentence(nl"anything else"(), message)` runs one with no request.
Coverage needs no service either: [`meaning_gaps`](@ref) is the check
`nl_dispatch` runs before its request. The resolution — question names, option
order, the state on the wire, the policy — needs one real answer, recorded once
and replayed in the test by [`with_recorded_answers`](@ref), which throws
[`ReplayMissError`](@ref) instead of reaching the network for a request it has no
recording of. [Test and Develop](@ref jev_testing_guide) walks through recording,
replaying, and crafting edge cases with a local server.

## See also

- [Start Here: Jev in Five Minutes](@ref jev_start) — which pathway, in one table
- [Jev with LLMs](@ref jev_llm_guide) — route before an LLM call, check after it
- [Route and Decide](@ref system_one_guide) — [`ask`](@ref), the three
  primitives, and decision policies built on the answers
- [Many Items at Once](@ref jev_algorithms_guide) — many judgments per request
  over collections: routing, ranking, search and joins
- [Test and Develop](@ref jev_testing_guide) — recorded answers, coverage checks
  and mock servers for tests
- [Natural-Language Dispatch](@ref nl_dispatch_api) — the API reference of
  [`Meaning`](@ref), [`nl_classify`](@ref), [`nl_dispatch`](@ref) and
  [`@branch`](@ref); [TypeSafe System One API (Jev)](@ref system_one_api) has
  the rest
- [TypeSafe documentation](https://docs.typesafe.ai) — the service's own
  reference for [Choice](https://docs.typesafe.ai/primitives/choice),
  [confidence](https://docs.typesafe.ai/confidence) and
  [models](https://docs.typesafe.ai/models)
