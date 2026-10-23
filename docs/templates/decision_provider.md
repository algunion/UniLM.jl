# [<Provider>](@id <entry>_start)

> Template: The start page of a Decisions entry, the first of its `REGISTRY` `pages`. The H1 is the provider's name with the entry's start id, an inherited one as `<id>__next` (`jev_start__next`); another page of the entry opens with `# [<Title>](@id <entry>_<stem>_page)` and this `@setup` block, then holds the sections moved to it.

```@setup <entry>
using UniLM
```

> Template: Identity line: one sentence on which decision models <Provider> serves and which endpoint reaches them.

```@eval
Main.ProviderFacts.at_a_glance(:<entry>)
```

| I want to… | Use | You get |
| :--- | :--- | :--- |
| [<task>](@ref <entry>_<feature>) | <call> | <result> |

> Template: One row per thing a reader comes to do, its first cell linking the section that shows it: in this entry, or in the entry's counterpart in the Language models chapter.

> Template: Keep the H2 of each feature `REGISTRY` lists for `<entry>`, with the id the README's Ids give it, and delete the others. A feature section holds <Provider>'s rules for it, an example and a link to its concept page.

## [Setup](@id <entry>_setup)

> Template: What the reader does once: the key or the server, and how a call reaches the provider (`service = …`, or `with_system_one` around the code).

## [First decision](@id <entry>_first_decision)

> Template: The smallest complete decision, executed (`@example <entry>`): one `ask`, its answer and its probability.

## [Models](@id <entry>_models)

> Template: The models to name, the default, how to list them, what each one takes.

## [Questions](@id <entry>_questions)

> Template: The question kinds answered (`choice`, `score`, `noul`, `box`, `RawQuestion`) and the limits of each: options, levels, instructions.

## [Images](@id <entry>_images)

> Template: Pictures in the state: formats, sizes, the models that see them.

> Template: Options and fields only <Provider> has, belonging to no feature: a section `## [Options](@id <entry>_options)` here.

## [Limits and refusals](@id <entry>_limits)

> Template: State, context and request limits, and every refusal before the request (`ArgumentError`) with its reason.

## [Confidence and tuning](@id <entry>_confidence)

> Template: What a probability means on this provider, how a threshold or a `decide` policy is set, and the model and version it was measured on.

## [Cost](@id <entry>_cost)

> Template: How a decision is priced and what its usage counts. The price table below is kept by `jev` and deleted by the other entries.

```@eval
Main.ProviderFacts.price_table(:<entry>)
```

## [Testing and recordings](@id <entry>_testing)

> Template: Recorded answers (`with_recorded_answers`): the namespace this provider's recordings are filed under, and what a replay needs.

## [Throughput](@id <entry>_throughput)

> Template: Many decisions at once (`ask_each`, concurrent calls): rate limits, and the throughput measured, with its model, date and machine.

## [When something fails](@id <entry>_failures)

> Template: One row per failure a reader can meet: what UniLM returns or throws, with its message; why; the fix.

| What you see | Why | What to do |
| :--- | :--- | :--- |
| <failure> | <cause> | <fix> |

> Template: Provenance: one line naming the model, the date and the versions (UniLM; the provider's API or server) the page's recordings were made with.

> Template: See also: one line linking the API reference of the entry's types and the entry's counterpart in the Language models chapter.
