# [Start Here: Decisions in Five Minutes](@id jev_start)

An LLM writes text. Jev reads a text and decides: which team handles it, how
urgent it is, whether a reply is safe to send, whether it matches the source. It
answers only the questions you list, with a probability for each answer, in one
quick request billed only for the text it reads ([prices](https://docs.typesafe.ai/models)).

**Setup:** `export TYPESAFE_API_KEY=…` in your shell, then `using UniLM` in
Julia ([Setup](@ref jev_setup) lists the optional variables).

!!! note "Providers"
    Decisions run on **Jev**, TypeSafe's System One model
    ([`TYPESAFEServiceEndpoint`](@ref), default model `jev-latest`). OpenAI
    announced a Decisions API on 29 September 2026, in limited preview; it has no
    public API reference yet, and UniLM does not support it yet. Support is
    planned once that reference is published.

## Which path?

| I want to… | Use | You get | Section |
| :--- | :--- | :--- | :--- |
| label a text with one of my own values | [`nl_classify`](@ref)`(text, TEAM)` | the key of the sentence that fits, e.g. `:billing` | [Dispatch on Meaning](@ref nl_dispatch_keyed) |
| ask several things about one text at once | one [`ask`](@ref) with a [`choice`](@ref), [`score`](@ref) or [`noul`](@ref) per question | every answer from one request, each with its probabilities | [Route and Decide](@ref jev_many_questions) |
| act only when Jev is sure enough, or weigh what mistakes cost | `min_confidence` and `fallback`, or a `decide` policy | the answer when it clears your bar, a hand-over otherwise | [Route and Decide](@ref jev_act_on_answer) |
| run the right function for a text | [`nl_dispatch`](@ref) with `texts = TEAM`, or `nl"…"` in the signatures | what the method of the chosen meaning returns | [Dispatch on Meaning](@ref nl_dispatch_paths) |
| decide before an LLM call, or check its output after | a Jev question on each side of [`respond`](@ref) | the team, the model to call, a draft checked before it is sent | [Decisions with LLMs](@ref jev_llm_guide) |
| judge a whole list at once | one [`ask`](@ref), one question per item | every item judged in one request | [Many Items at Once](@ref jev_algorithms_guide) |
| test without the network | [`with_recorded_answers`](@ref) | real answers recorded once, replayed with no key | [Test and Develop](@ref jev_testing_recorded) |

## Your first decision

One plain sentence per team, and a customer message:

```@example jevstart
using UniLM

const TEAM = (billing   = "payments, charges, invoices or refunds",
              technical = "the app or the website does not work as expected",
              shipping  = "a parcel that is late, lost or arrived damaged",
              other     = "anything else")

nl_classify("I was charged twice for order #4471. Please refund the duplicate payment.", TEAM)
```

Jev read the message and the four sentences, never the keys, and the key of the
sentence that fits came back: a value your program can act on.

## Five-minute tour

The support desk of a small online shop that sells phones and accessories,
ships parcels and has a mobile app. Decisions with LLMs and Test and Develop
build the same desk; the other Decisions pages borrow from it where it fits.
Without Jev, each step below would be keyword rules, a classifier trained on
labelled messages, or an LLM whose free-text answer you parse. Every example
runs when this manual is built, from answers recorded once and replayed ([Test
and Develop](@ref jev_testing_manual)).

**Label the inbox.**

```@example jevstart
const MESSAGES = [
    "I was charged twice for order #4471. Please refund the duplicate payment.",
    "The app crashes every time I open my order history. iPhone 15, latest version.",
    "My parcel was due last Monday and the tracking hasn't moved in six days.",
    "Do you ship to Norway, and how much does delivery cost?",
    "The phone I received has a cracked screen and the box was crushed. I want my money back.",
    "Your update deleted my saved addresses and now I can't check out. Third time this month!",
]

for message in MESSAGES
    println(rpad(nl_classify(message, TEAM), 11), message)
end
```

Every message gets a team. The Norway question is the unclear one: Jev's best
guess is `:billing`, at a confidence of only 0.49, with the probability split
between billing (0.62) and anything else (0.38).

**Hand the unsure ones to a person.**

```@example jevstart
nl_classify(MESSAGES[4], TEAM; min_confidence = 0.6, fallback = message -> :person)
```

Below `min_confidence`, the call returns what `fallback` returns; without a
`fallback`, it throws [`LowConfidenceError`](@ref). Set the bar by what a wrong
team costs against a person's time, and when some mistakes cost more than
others, let a `decide` policy weigh them ([When mistakes have different
prices](@ref jev_loss_matrix)).

**Run the right function.** One method per team, on the team's key; with
`texts = TEAM`, the method whose sentence fits runs:

```@example jevstart
route(::Val{:billing}, message)   = "opened a payment review"
route(::Val{:technical}, message) = "filed a bug report"
route(::Val{:shipping}, message)  = "opened a claim with the carrier"
route(::Val{:other}, message)     = "forwarded to the front desk"

nl_dispatch(route, MESSAGES[2]; texts = TEAM)
```

A key in the table with no method is refused before any request is sent.
[Dispatch on Meaning](@ref nl_dispatch_literal) shows the other spelling: each
sentence in its method's signature, `nl"…"`, and no table.

**Check an LLM's draft before the customer sees it.** The LLM writes the reply;
Jev reads it, and your code decides whether it goes out:

```@example jevstart
reply = output_text(respond(MESSAGES[5]; model = "gpt-5.4-mini",
                            instructions = "You answer customers of a small online shop, in at most three sentences."))
println(reply)

promises = ask(reply, "refund" => noul("Does this reply promise the customer a refund?"))["refund"].noul
# Cut low: a refund promised without approval costs more than a colleague's look.
println("P(promises a refund) = ", promises, promises >= 0.3 ? ": a person approves it first" : ": send it")
```

The draft ends with "we’ll help arrange a refund right away". Jev puts the
probability that it promises a refund at 0.81, above the cut, so a person
approves the reply before the customer sees it. [Decisions with LLMs](@ref
jev_llm_check_draft) checks one draft for a leak, a promise the policy does not
allow and rudeness in one request, and checks its claims against the source.

## What Jev is not for

- **Writing.** Jev answers the questions you list; it never writes text. A reply
  is an LLM's job, and [Decisions with LLMs](@ref jev_llm_guide) combines the two.
- **Arithmetic, counting and dates.** Adding amounts, counting items and
  comparing dates are unreliable: compute them in Julia, and ask Jev only for
  the judgment.
- **Long documents with unrelated material.** Accuracy falls as unrelated text
  grows around the decision: pass only what the decision needs.

TypeSafe documents these limits for each model version: [Jev 1.13
jaggedness](https://docs.typesafe.ai/model-jaggedness/jev-1.13).

## Where next

- [Decisions with LLMs](@ref jev_llm_guide) — route before an LLM call, check the
  draft after it, pick the model, approve tool calls
- [Route and Decide](@ref system_one_guide) — [`ask`](@ref) and its three
  question types, a struct filled in one request, decision rules on the answers
- [Dispatch on Meaning](@ref nl_dispatch_guide) — [`nl_classify`](@ref),
  [`nl_dispatch`](@ref) with a table or with `nl"…"` in the signatures, and
  [`@branch`](@ref)
- [Many Items at Once](@ref jev_algorithms_guide) — a batch in one request,
  ranking, the first line where something happens, matching records
- [Test and Develop](@ref jev_testing_guide) — recorded answers for Jev and LLM
  calls, tests that need no key, an audit trail
- [TypeSafe System One API (Jev)](@ref system_one_api) — setup, models, limits,
  errors and cost, then every type and function
