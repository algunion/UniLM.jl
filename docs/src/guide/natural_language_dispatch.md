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
you how close the runner-up was. Here the option list *is* the method table, the
answer is one of its entries by construction, and a combination no method covers
surfaces as Julia's own `MethodError` instead of a silent mis-route.

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

```julia
nl_dispatch(route, "My package arrived crushed and the screen is cracked. I want my money back.")
# => :refund
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
pin at least one positional argument to a *concrete* meaning. Those methods are
sorted by source file and line, so within one file the option order is the order
you wrote them in, and [`meanings`](@ref) previews it faithfully. A method with
no concrete meaning anywhere in its signature — a wildcard `f(::Meaning, x)`, or
a method with no `Meaning` at all — is an ordinary method and is skipped. Every
method that *is* collected must agree with the others on positional arity and on
which positions are meaning slots, because one set of questions has to serve all
of them; a disagreement is an `ArgumentError` naming both methods, raised before
any request goes out. Varargs cannot carry a meaning.

**How the request is built.** One [`choice`](@ref) question per slot, all in a
single [`ask`](@ref):

- The **question name** is the slot's argument name, or `meaning_<position>`
  when the slot is unnamed (`::nl"..."` with nothing in front of the `::`).
  Question names key the answers and are never sent to the model.
- The **option names are the meanings themselves**, with no description
  attached. There is no other text for the model to read, so the description you
  write in the signature *is* the prompt for that option.
- The **state** is a JSON object keyed by the names of the remaining, ordinary
  arguments (`arg<position>` for an unnamed one). Pass `state = ...` to send
  something else instead; with no ordinary arguments at all there is nothing to
  describe, so `state` is then required.
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
`choice` is one of the option names by construction. `nl_dispatch` turns it back
into `Meaning{Symbol(choice)}()`, splices it into the position it came from,
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

```julia
nl_dispatch(reply, "This is the third time I have written about my broken order.")
# one request, two questions; the resolved pair selects one of the four methods
```

Both slots resolve in **one** request. The two questions are independent — they
are asked about the same state, and neither answer becomes context for the
other — so the cost of a second slot is the tokens of its option list, not a
second ingestion of the state. That is the same economics as batching questions
into one [`ask`](@ref) ([parallel
questions](https://docs.typesafe.ai/cookbooks/parallel_questions)).

Four methods cover all four combinations here. When they do not, the resolved
pair lands on a gap in the method table and Julia raises its own `MethodError` —
`nl_dispatch` does not catch it, because a missing method is your bug and not a
service failure. Two ways to close the gap:

```@example nldispatch
# 1. A wildcard method. `::Meaning` is not a concrete meaning, so this method is
#    never collected and never offered to the model — it is a pure Julia
#    backstop that ordinary dispatch reaches when nothing more specific matches.
reply(::Meaning, ::Meaning, msg) = :unclassified

reply(nl"a rant"(), nl"a calm tone"(), "…")   # no request; ordinary dispatch
```

```@example nldispatch
# 2. An explicit catch-all meaning, which IS offered to the model, so the model
#    can say "none of these" instead of forcing its probability onto an option
#    that does not fit.
reply(::nl"anything else", ::nl"a calm tone", msg)  = :triage
reply(::nl"anything else", ::nl"an angry tone", msg) = :triage

for (position, options) in sort(collect(meanings(reply)))
    println(position, " => ", options)
end
```

Prefer the second: a wildcard tells you nothing about *why* it fired, while a
catch-all meaning comes back with a name and a probability. Note what the
wildcard is not allowed to be — a method that pins some slots and leaves others
wild, such as `reply(::nl"a complaint", ::Meaning, msg)`, disagrees with the
other methods about which positions are slots, and [`meanings`](@ref) and
[`nl_dispatch`](@ref) both reject it with an `ArgumentError` rather than send a
question list that does not match the method table. Wildcard *every* slot or
none of them.

## Composing with ordinary dispatch

A resolved meaning is an ordinary argument, so the remaining positions keep
dispatching on their types exactly as they always did:

```@example nldispatch
handle(::nl"a greeting", who::String) = "hello, " * who
handle(::nl"a greeting", n::Int)      = "hello, customer #" * string(n)
handle(::nl"a complaint", who::String) = "sorry, " * who

println(meanings(handle))
println(handle(nl"a greeting"(), "ada"))
println(handle(nl"a greeting"(), 41))
```

`meanings(handle)` lists `"a greeting"` once even though two methods declare it:
the options are the *distinct* descriptions per slot. The model picks the
meaning; Julia picks between the `String` and the `Int` method afterwards, with
no extra request.

This also makes the interface open. The option list is the method table, and a
method table is extensible from anywhere — a downstream module or a package that
does not own `handle` can add `handle(::nl"a refund request", who::String)`, and
from the next call on it is one more option the model may choose. Nothing
central enumerates the meanings, so adding a case never means editing a list in
two places.

## Confidence and control

A Choice answer always names a winner, even when the distribution is nearly
flat, so the gate is `confidence` rather than `choice`. `confidence` measures how
concentrated the distribution is — all the mass on one option gives 1.0, a flat
spread gives a low number — and says nothing about whether the winner is
*correct* ([Confidence](https://docs.typesafe.ai/confidence)).

```julia
# Below the threshold, `fallback` is called with the caller's own arguments.
nl_dispatch(route, ticket; min_confidence = 0.7, fallback = t -> :needs_a_human)

# With no fallback, the same answer is a loud failure instead of a route.
nl_dispatch(route, ticket; min_confidence = 0.7)
# => throws LowConfidenceError, whose `.answer` carries the full probabilities
```

[`LowConfidenceError`](@ref) holds the whole [`ChoiceAnswer`](@ref), so a handler
can read `probabilities` for a diagnostic or a second-best policy. The other
keywords:

| Keyword | Meaning |
| :--- | :--- |
| `min_confidence` | threshold in `0 … 1`; below it, `fallback` runs or [`LowConfidenceError`](@ref) is thrown |
| `fallback` | called with the caller's `args...` when the gate trips |
| `instructions` | a `String` for every slot, or a `Vector` with one entry per slot; default is the generic instruction above |
| `state` | send this instead of the object built from the ordinary arguments |
| `model` | pin a version, e.g. `"jev-1.13.0"`, instead of the `jev-latest` alias |
| `service` | endpoint type; defaults to [`TYPESAFEServiceEndpoint`](@ref UniLM.TYPESAFEServiceEndpoint) |
| `config` | a [`RequestConfig`](@ref) governing timeouts and the retry budget for this call |

Pin the model before you tune a threshold. A threshold is a claim about one
version, and `jev-latest` moves when a new one ships
([Models](https://docs.typesafe.ai/models)); [`SystemOneResponse`](@ref)`.model`
reports the version that actually answered. And read `meanings(f)` first — a
threshold that keeps tripping is usually an option list with two overlapping
meanings in it, not a number that needs lowering.

A call that failed outright throws [`SystemOneError`](@ref) rather than
resolving to a method, so a timeout or a 500 can never be mistaken for a
routing decision.

## `@branch`: the inline form

When the decision is local and nobody needs to extend it, [`@branch`](@ref) is
the same request without the method table. It is a `switch` whose cases are
written in plain language:

```julia
ticket = "My package arrived crushed and the screen is cracked. I want my money back."

action = @branch ticket min_confidence=0.6 begin
    "the customer wants a refund"                              => :refund
    ("the customer reports a bug", "A defect in the software") => :bug
    "the customer asks a pricing question"                     => :pricing
    _                                                          => :escalate
end
# => :refund
```

Every line is `option => expression`:

- A bare left-hand side is the option **name** — the text the model reads and the
  text the answer returns. Left-hand sides are ordinary expressions, evaluated
  once in source order, so a name may be computed.
- The 2-tuple form `("name", "description")` attaches a longer description to
  that name. It is the only way to do so.
- A final `_ => expression` is the fallback taken when the winner's confidence is
  below `min_confidence`. It *requires* `min_confidence`: without a threshold
  nothing is ever low-confidence and the line could never run, so that spelling
  is rejected.

Keywords go between the state and the block, written `key = value`: `model`,
`min_confidence`, `instructions`, `service` and `config`, each meaning what it
means for [`nl_dispatch`](@ref).

The whole block compiles to a **single** Choice request whose question is named
`branch` and whose criteria are the option names, in source order. Only the
selected body is evaluated — the others are not merely discarded, they never
run — and the macro's value is that body's value. Everything that can be checked
statically is checked while the macro expands, so it surfaces when the
surrounding code is loaded rather than the first time the branch is reached: an
unknown keyword, a block that is not `begin ... end`, a line that is not
`option => expression`, no options at all, two `_` lines, or a `_` that is not
last. A non-success call throws [`SystemOneError`](@ref); the branch is never
guessed.

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

That covers the method bodies. To cover the *resolution* — question names,
option order, the state that goes on the wire, the confidence gate — point
`TYPESAFE_BASE_URL` at a local server and answer the Choice yourself:

```julia
using UniLM, HTTP, JSON

canned = """
{"model": "jev-1.13.0",
 "answers": {"meaning_1": {"type": "choice",
                           "choice": "the customer wants a refund",
                           "confidence": 0.97,
                           "probabilities": {"the customer wants a refund": 0.97,
                                             "the customer reports a bug in the app": 0.02,
                                             "anything else": 0.01}}},
 "usage": {"input_tokens": 96, "output_tokens": 3}}
"""

seen = String[]
server = HTTP.serve!("127.0.0.1", 8123; verbose=false) do request
    push!(seen, String(copy(request.body)))          # assert on the request you sent
    HTTP.Response(200, ["Content-Type" => "application/json"], Vector{UInt8}(canned))
end

withenv("TYPESAFE_BASE_URL" => "http://127.0.0.1:8123", "TYPESAFE_API_KEY" => "test") do
    @assert nl_dispatch(route, t) === :refund
end
close(server)
```

The response shape is the whole contract: a `model`, an `answers` object keyed by
the question names the client sent (`meaning_1` here, or the slot's argument name
when it has one), each answer a Choice with `choice`, `confidence` and
`probabilities`, plus a `usage` object. The numbers are yours to pick — that is
the point of a canned answer. Make `confidence` low to exercise `min_confidence`,
`fallback` and [`LowConfidenceError`](@ref); return a non-200 to exercise
[`SystemOneError`](@ref); answer with a meaning whose combination no method
covers to exercise the `MethodError`.

## Designing meanings

- **Write the meaning as a description, not a label.** It is the option name and
  it is the only text the model gets for that branch. `nl"the customer wants a
  refund"` works; `nl"refund"` asks the model to guess what you meant by the
  word.
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

## See also

- [Typed Judgments with Jev (TypeSafe System One)](@ref system_one_guide) — the
  `ask` verb, the three primitives, and the setup this page assumes
- [TypeSafe System One API (Jev)](@ref system_one_api) — every type and verb,
  including [`Meaning`](@ref), [`@branch`](@ref) and [`nl_dispatch`](@ref)
- [TypeSafe documentation](https://docs.typesafe.ai) — the service's own
  reference for [Choice](https://docs.typesafe.ai/primitives/choice),
  [confidence](https://docs.typesafe.ai/confidence) and
  [models](https://docs.typesafe.ai/models)
