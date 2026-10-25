# [<Provider>](@id <entry>_start)

> Template: The start page of a Decisions entry, the first of its `REGISTRY` `pages`. The H1 is the provider's name with the entry's start id, an inherited one as `<id>__next` (`jev_start__next`); another page of the entry opens with `# [<Title>](@id <entry>_<stem>_page)` and this `@setup` block, then holds the sections moved to it. An entry of several pages keeps Setup, Options, First decision, Models, Questions and Limits and refusals on its start page, puts Images and the sections of what only <Provider> takes on an abilities page, and Confidence and tuning to the end on an operations page, as `decisions/jev/` does, each id on one page (README, Placement); where its pages divide otherwise, each section goes to the page whose subject it is.

```@setup <entry>
using UniLM
```

> Template: Identity line: a sentence each on what <Provider> is, which decision models it serves and which of its APIs the endpoint speaks; on the verbs that refuse the endpoint, if any, pointing at At a glance for the capabilities it lacks instead of listing them; and on the parts of the provider's API that UniLM does not wrap. An entry that does not record also says that its examples are not run (README, Executed or quoted).

```@eval
Main.ProviderFacts.at_a_glance(:<entry>)
```

| I want to… | Use | You get |
| :--- | :--- | :--- |
| [<task>](@ref <entry>_<feature>) | <call> | <result> |

> Template: One row per thing a reader comes to do, its first cell linking the section that shows it: in this entry, or in the entry's counterpart in the Language models chapter.

> Template: Keep the H2 of each feature `REGISTRY` lists for `<entry>`, with the id the README's Ids give it, and delete the others. A feature section holds <Provider>'s rules for it, an example and a link to its concept page.

> Template: A refusal of a request by <Provider>'s endpoint, an `ArgumentError` thrown before it is sent (a missing capability aside: At a glance), is stated once, as a row of Limits and refusals: every other section that meets it names what is refused and links Limits and refusals, since a table row has no anchor. A check the endpoint's constructor runs is stated once too, in Options, which a section that meets it links. A refusal every decision provider makes (no question, a blank or repeated question name) stays on its concept page, which Limits and refusals links.

> Template: A sentence about behaviour says what the code does and the condition under which it does it, no broader: "returns a `SystemOneCallError`" where nothing is thrown, "before anything is sent" only where no byte reaches the provider. A rule UniLM applies per model is a generated table's (README, Generated facts).

> Template: A page other than the start page may open, under its H1, with a paragraph and a table of the sections it holds, each row linking its section, as `decisions/jev/abilities.md` opens. The paragraph and the table are the page's index: they state nothing that the sections the table links do not.

## [Setup](@id <entry>_setup)

> Template: What the reader does once before the first decision: set the key (a `bash` block), start a server or pull a model; then how a call reaches the provider (`service = …`, or `with_system_one` around the code) and the model it gets when it names none.

> Template: The endpoint's options, the arguments of its constructor that the requests made through an instance carry, and the provider options `ask` takes as keywords (`options = (…)` on the natural-language verbs, keywords of `with_system_one` for a scope) that no other section covers: what each sends and what is checked, in a section `## [Options](@id <entry>_options)` here.

## [First decision](@id <entry>_first_decision)

> Template: The smallest complete decision, on the default model or, where the entry has none, a model it names; executed (`@example <entry>`) when the entry records: one `ask`, its answer and its probability, and the model that answered.

## [Models](@id <entry>_models)

> Template: The models to name, the default and what it resolves to, how to list them, what each one takes, and how to pin a version against an alias that moves.

## [Questions](@id <entry>_questions)

> Template: The question kinds answered (`choice`, `score`, `noul`, `box`, `RawQuestion`), what of a question reaches the model (its name, its options' names), and the limits of each, named and linked to Limits and refusals.

## [Images](@id <entry>_images)

> Template: Pictures in a request: where they may stand, formats and sizes (a refused one named and linked to Limits and refusals), how they are sent and billed, the models that see them.

> Template: What only <Provider> takes, belonging to no feature (a question kind of its own, a request option with a subject of its own): a section each, `## [<Section>](@id <entry>_<section>)`, here.

## [Limits and refusals](@id <entry>_limits)

> Template: The state, context and request limits a request meets, and which of them UniLM checks (rate limits go to the failures section); then one row per refusal before the request is sent (`ArgumentError`): what is refused; the Why cell opens with words of its message as a code span, without quote marks, so that the message's underscores render as written — a refusal with several messages gives each its own span, comma-separated — and then gives the reason; the fix.

| Refused | Why: the message's words | What to do |
| :--- | :--- | :--- |
| <refusal> | `<words>`: <cause> | <fix> |

## [Confidence and tuning](@id <entry>_confidence)

> Template: What a probability and a `confidence` mean on this provider (who computes them, how they are rounded, what `ask` accepts), how a threshold or a `decide` policy is set for it, and the model and version each measurement was made on (an Evidence note).

## [Cost](@id <entry>_cost)

> Template: How a decision is priced (`estimated_cost`) and what its usage counts. The price table below is kept by `jev` and deleted by the other entries.

```@eval
Main.ProviderFacts.price_table(:<entry>)
```

## [Testing and recordings](@id <entry>_testing)

> Template: Recorded answers (`with_recorded_answers`): the path this provider's recordings are filed under, what is keyed besides the body and what is left out, what recording needs and what a replay needs; how to test what a recording cannot produce.

## [Throughput](@id <entry>_throughput)

> Template: Many decisions at once (`ask_each`, concurrent calls): the requests in flight by default, the rate limits, and the throughput measured, with its model and date.

## [When something fails](@id <entry>_failures)

> Template: How a failed call is returned and which failures UniLM sends again, rate limits and overload among them; then one row per failure a reader can meet: what UniLM returns or throws, with its message; why; the fix.

| What you see | Why | What to do |
| :--- | :--- | :--- |
| <failure> | <cause> | <fix> |

> Template: Provenance: the last line of each page with recordings, above See also on the start page, stating how they were made and nothing else, as `decisions/jev/index.md` does: "Recorded on" the date their files' `recorded_at` gives (UTC); each model they were made on; UniLM's version, or, for code not yet released, "UniLM's development branch after" the last release; and the provider's API or server at its version. A page that replays recordings of several dates names each date with the models and the UniLM version of its recordings. An entry that does not record puts its provenance line (README, Executed or quoted) last on its start page, above See also; output quoted from a live run is dated as that section says.

> Template: See also: the start page's last line, linking the API reference of the entry's types and, where the entry has one, its counterpart in the Language models chapter.
