# [Decisions](@id decisions_start)

```@setup decisions
using UniLM
```

A decision model reads a piece of state — a message, a record, a picture — and answers the questions you list
about it with a probability for each answer you allowed: which of these options, where on this scale, whether
this statement holds. It writes no text, so there is nothing to parse: [`ask`](@ref) returns typed answers your
code branches on, and their probabilities say whether to act on an answer or to hand the case to a person. Use
a decision model rather than a language model when the outcome is one of a set you can name and your code acts
on it — the team that handles a ticket, how urgent it is, whether a draft is safe to send — and a language model
when the output is text a person reads; [Decisions with LLMs](@ref jev_llm_guide) puts a decision on each side
of a language model's call.

## [Choose your provider](@id decisions_providers)

```@eval
Main.ProviderFacts.provider_table(Main.ProviderFacts.decisions)
```

Every provider answers [`ask`](@ref) and the verbs built on it with the same result types, though not every
provider takes every question kind or option. A call names its provider with
`service = …`, a [`with_system_one`](@ref) scope names one for every call inside it, and a call that names
neither goes to Jev. What each provider takes, and why a confidence threshold is tuned for each, is on
[Choosing a decision provider](@ref system_one_backends).

## [Concepts](@id decisions_concepts)

- [Route and Decide](@ref system_one_guide) — [`ask`](@ref) and the question kinds, reading the answers, and
  decision rules on their probabilities: a threshold, a price for each mistake, a catch-all option.
- [Dispatch on Meaning](@ref nl_dispatch_guide) — [`nl_classify`](@ref), [`nl_dispatch`](@ref) and
  [`@branch`](@ref): the answer chooses the code that runs.
- [Many Items at Once](@ref jev_algorithms_guide) — [`ask_each`](@ref), many items in one request, ranking,
  finding the first match, matching records across two lists.
- [Decisions with LLMs](@ref jev_llm_guide) — a decision before a language model's call, to route it, and after
  it, to check what it wrote.
- [Test and Develop](@ref jev_testing_guide) — previews of what a call sends, recorded answers, tests that need
  no key, and an audit trail.

## [Across providers](@id decisions_across)

- [Choosing a decision provider](@ref system_one_backends) — the same call on each provider, per call or
  per scope, what each one takes, and a threshold tuned per provider.
- [Write your own decision provider](@ref decisions_custom) — a gateway that speaks TypeSafe's wire, or a
  protocol of your own.
