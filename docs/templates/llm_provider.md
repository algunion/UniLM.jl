# [<Provider>](@id <entry>_start)

> Template: The start page of a Language models entry, the first of its `REGISTRY` `pages`. The H1 is the provider's name with the entry's start id, an inherited one as `<id>__next` (`ollama_guide__next`); another page of the entry opens with `# [<Title>](@id <entry>_<stem>_page)` and this `@setup` block, then holds the sections moved to it.

```@setup <entry>
using UniLM
```

> Template: Identity line: one sentence on what <Provider> is and which of its APIs the endpoint speaks; an entry that does not record also says that its examples are not run (README, Executed or quoted).

```@eval
Main.ProviderFacts.at_a_glance(:<entry>)
```

| I want to… | Use | You get |
| :--- | :--- | :--- |
| [<task>](@ref <entry>_<feature>) | <call> | <result> |

> Template: One row per thing a reader comes to do, its first cell linking the section that shows it: in this entry, or in the entry's counterpart in the Decisions chapter.

> Template: Keep the H2 of each feature `REGISTRY` lists for `<entry>`, with the id the README's Ids give it, and delete the others (At a glance lists what the provider lacks). A feature section holds <Provider>'s rules for it, an example and a link to its concept page.

## [Setup](@id <entry>_setup)

> Template: What the reader does once before the first call: set the key (a `bash` block), give the endpoint its arguments, start a server or pull a model.

## [First call](@id <entry>_first_call)

> Template: The smallest complete call, on the default model or, where the entry has none, a model it names; executed (`@example <entry>`) when the entry records; then its usage and cost.

## [Conversations](@id <entry>_conversations)

> Template: A multi-turn `Chat`: what goes back with each turn, where the system message goes, what is refused.

## [Streaming](@id <entry>_streaming)

> Template: A streamed call, and what the stream carries: text, thinking, tool calls, usage.

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

## [Prompt caching](@id <entry>_caching)

> Template: How caching is turned on, what a cached token costs, where usage reports it.

> Template: Options and fields only <Provider> has, belonging to no feature: a section `## [Options](@id <entry>_options)` here.

## [Cost and usage](@id <entry>_cost)

> Template: How a result is priced (`estimated_cost`) and what its usage counts. The price table below is kept by `openai`, `anthropic`, `gemini` and `deepseek`, and deleted by the other entries.

```@eval
Main.ProviderFacts.price_table(:<entry>)
```

## [Limits and refusals](@id <entry>_limits)

> Template: Token, size and rate limits, and every refusal before the request (`ArgumentError`) with its reason.

## [When something fails](@id <entry>_failures)

> Template: One row per failure a reader can meet: what UniLM returns or throws, with its message; why; the fix.

| What you see | Why | What to do |
| :--- | :--- | :--- |
| <failure> | <cause> | <fix> |

## [Models](@id <entry>_models)

> Template: The models to name, how to list them, the rules of each. A generated model table (`openai_model_rules()`, `claude_family_table()`, `gemini_thinking_table()`) goes in an `@eval` block in the section whose rules it shows.

> Template: Provenance: one line naming the model, the date and the versions (UniLM; the provider's API or server) the page's recordings were made with, or, on an entry that does not record, the versions its code was checked against.

> Template: See also: one line linking the API reference of the entry's types and the entry's counterpart in the Decisions chapter.
