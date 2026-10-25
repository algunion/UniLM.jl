# [<Provider>](@id <entry>_start)

> Template: The start page of a Language models entry, the first of its `REGISTRY` `pages`. The H1 is the provider's name with the entry's start id, an inherited one as `<id>__next` (`ollama_guide__next`); another page of the entry opens with `# [<Title>](@id <entry>_<stem>_page)` and this `@setup` block, then holds the sections moved to it. An entry of several pages keeps Setup to Streaming on its start page and puts Tools to Limits and refusals on a requests page and Prompt caching to the end on a cost page, as `providers/anthropic/` does, each id on one page (README, Placement); where its pages divide otherwise (a page per API, a media page), each section goes to the page whose subject it is.

```@setup <entry>
using UniLM
```

> Template: Identity line: a sentence each on what <Provider> is and which of its APIs the endpoint speaks; on the verbs that refuse the endpoint, if any, pointing at At a glance for the capabilities it lacks instead of listing them; and on the parts of the provider's API that UniLM does not wrap, beyond what At a glance lists. An entry that does not record also says that its examples are not run (README, Executed or quoted).

```@eval
Main.ProviderFacts.at_a_glance(:<entry>)
```

| I want to… | Use | You get |
| :--- | :--- | :--- |
| [<task>](@ref <entry>_<feature>) | <call> | <result> |

> Template: One row per thing a reader comes to do, its first cell linking the section that shows it: in this entry, or in the entry's counterpart in the Decisions chapter.

> Template: Keep the H2 of each feature `REGISTRY` lists for `<entry>`, with the id the README's Ids give it, and delete the others (At a glance lists what the provider lacks). A feature section holds <Provider>'s rules for it, an example and a link to its concept page.

> Template: A refusal of a request by <Provider>'s endpoint, an `ArgumentError` thrown before it is sent (a missing capability aside: At a glance), is stated once, as a row of Limits and refusals: every other section that meets it names what is refused and links Limits and refusals, since a table row has no anchor. A check the endpoint's constructor runs is stated once too, in Options, which a section that meets it links.

> Template: A sentence about behaviour says what the code does and the condition under which it does it, no broader: "returns an `LLMCallError`" where nothing is thrown, "once every block arrived whole" where a stream cut short keeps no blocks. A rule UniLM applies per model is a generated table's (README, Generated facts).

## [Setup](@id <entry>_setup)

> Template: What the reader does once before the first call: set the key (a `bash` block), give the endpoint its arguments, start a server or pull a model.

> Template: The endpoint's options, the arguments of its constructor that the requests made through an instance carry, belonging to no feature: what each sends and what the constructor checks, in a section `## [Options](@id <entry>_options)` here.

## [First call](@id <entry>_first_call)

> Template: The smallest complete call, on the default model or, where the entry has none, a model it names; executed (`@example <entry>`) when the entry records; then its usage and cost.

## [Conversations](@id <entry>_conversations)

> Template: A multi-turn `Chat`: what goes back with each turn, where the system message goes, what is refused.

## [Streaming](@id <entry>_streaming)

> Template: A streamed call, and what the stream carries: text, thinking, tool calls, usage.

> Template: A requests page may open, under its H1, with a paragraph and a table of the `Chat` fields: each field, what the provider's API takes for it, and a link to the section with its rules, as `providers/anthropic/requests.md` opens. The paragraph and the table are the page's index: they state nothing that the sections the table links do not.

## [Tools](@id <entry>_tools)

> Template: Tool calls and `tool_loop!`, with the `tool_choice`, `parallel_tool_calls` and `strict` values taken or refused.

## [Structured output](@id <entry>_structured_output)

> Template: `response_format`: what the provider enforces, and what is refused before the request.

## [Reasoning](@id <entry>_reasoning)

> Template: The thinking settings taken, `reasoning_text`, and what thinking spends.

## [Attachments](@id <entry>_media)

> Template: The images, audio and documents a message may carry: formats, sizes, the models that read them.

## [Embeddings](@id <entry>_embeddings)

> Template: Embedding requests: models, dimensions, task prompts, what is refused.

## [Image generation](@id <entry>_images)

> Template: Generating and editing images: models, sizes, formats, what is refused.

## [Audio](@id <entry>_audio)

> Template: Speech, transcription and spoken replies: models, formats, what is refused.

## [FIM and prefix completion](@id <entry>_completions)

> Template: `fim_complete` and `prefix_complete`: the models that take them, what is refused.

> Template: The request fields no feature section covers, `Chat` fields with a rule of <Provider>'s and fields only <Provider> has: what each sends, in a section `## [Request options](@id <entry>_request_options)` here.

## [Limits and refusals](@id <entry>_limits)

> Template: The token and size limits a request meets, and which of them UniLM checks (rate limits go to the failures section); then one row per refusal before the request is sent (`ArgumentError`): what is refused; the Why cell opens with words of its message in quotes — a refusal with several messages quotes each by its own words, comma-separated — and then gives the reason; the fix.

| Refused | Why: the message's words | What to do |
| :--- | :--- | :--- |
| <refusal> | "<words>": <cause> | <fix> |

## [Prompt caching](@id <entry>_caching)

> Template: How caching is turned on, what a cached token costs, where usage reports it.

## [Cost and usage](@id <entry>_cost)

> Template: How a result is priced (`estimated_cost`) and what its usage counts. The price table below is kept by `openai`, `anthropic`, `gemini` and `deepseek`, and deleted by the other entries.

```@eval
Main.ProviderFacts.price_table(:<entry>)
```

## [When something fails](@id <entry>_failures)

> Template: How a failed call is returned and which failures UniLM sends again, rate limits and overload among them; then one row per failure a reader can meet: what UniLM returns or throws, with its message; why; the fix.

| What you see | Why | What to do |
| :--- | :--- | :--- |
| <failure> | <cause> | <fix> |

## [Models](@id <entry>_models)

> Template: The models to name, how to list them, the rules of each (a generated model table: README, Generated facts).

> Template: An endpoint the entry documents but does not recommend, one `REGISTRY` does not list among the entry's `endpoints`, such as the provider's OpenAI-compatible one: a section `## [The OpenAI-compatible endpoint](@id <entry>_compatible)` here, with what the provider serves it for, how a `Chat` reaches it, and what of this entry does not hold there.

> Template: Provenance: the last line of each page with recordings, above See also on the start page, stating how they were made and nothing else, as `providers/anthropic/index.md` does: "Recorded on" the date their files' `recorded_at` gives (UTC), or the first and the last where those dates differ; each model they were made on; UniLM's version, or, for code not yet released, "UniLM's development branch after" the last release; and the provider's API or server at its version. An entry that does not record puts its provenance line (README, Executed or quoted) last on its start page, above See also; output quoted from a live run is dated as that section says.

> Template: See also: the start page's last line, linking the API reference of the entry's types and, where the entry has one, its counterpart in the Decisions chapter.
