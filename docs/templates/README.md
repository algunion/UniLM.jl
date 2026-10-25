# Page templates

The pages of the Language models and Decisions chapters start from these templates. They live outside
`docs/src`, so the manual never builds them.

| Template | Copied to |
| :--- | :--- |
| `llm_provider.md` | the pages of a Language models entry, under `docs/src/providers/` |
| `decision_provider.md` | the pages of a Decisions entry, under `docs/src/decisions/` |
| `concept.md` | a concept page: a mechanism every provider shares |
| `chapter.md` | a chapter's start page, `docs/src/providers/index.md` or `docs/src/decisions/index.md` |

An entry is a `REGISTRY` row of `docs/ProviderFacts.jl`, which stores its key, name, `pages` (start page
first), start id and feature ids.

## Filling a template

A line that starts with `> Template:` is an instruction: follow it, then delete the line, which holds all of it.
Julia's Markdown has no HTML, so an HTML comment would show on the page; instructions are block quotes with this
prefix instead, and `grep -rn '^> Template:' docs/src` must print nothing. Every placeholder is replaced:

| Placeholder | Stands for |
| :--- | :--- |
| `<entry>` | the entry's `REGISTRY` key (`anthropic`, `local_servers`, `jev`), also the page's example sandbox |
| `<Provider>` | the entry's `REGISTRY` name |
| `<feature>` | a feature key (`streaming`, `first_decision`, …) |
| `<chapter>`, `<Chapter>` | `llm` or `decisions`, as `ProviderFacts.Chapter` names them; the chapter's title |
| `<stem>`, `<Title>` | the page file's name without `.md`; the page's title |
| `<Endpoint>`, `<model>` | an endpoint as code names it (`OPENAIServiceEndpoint`, `DeepSeekEndpoint()`); a model id |
| `<names>`, `<values>` | further names a binding block binds (`llm, llm_model`); their values |
| `<section>`, `<Section>` | a section's id part (`options` in `ollama_options`); its title |
| `<id>` | an id that exists elsewhere today, written `<id>__next` (Ids) |
| `<task>`, `<call>`, `<result>`, `<failure>`, `<refusal>`, `<cause>`, `<fix>` | the cells of a table row; the table takes one row per item |
| `<words>` | words of a refusal's `ArgumentError` message, as the message writes them |

A section title may say what the provider offers ("Images and audio") or what the section holds ("Rate limits,
overload and failures"); its id does not change.

## Ids

Every heading carries its id, `[Title](@id id)`: generated tables and other pages link by id, so a section
keeps its links when its title changes or it moves.

- Page titles (H1): an entry's start page, the entry's start id; any other page of an entry,
  `<entry>_<stem>_page` (`anthropic_caching_page`); the chapter start pages, `llm_start` and `decisions_start`;
  `decisions/custom.md`, `decisions_custom`; a new concept page, or one whose title has no id, `<stem>_guide`.
- Sections: a feature, the id `REGISTRY` stores for it, which the generated tables link to — `<entry>_<feature>`,
  or the id it inherits (`jev_errors` for Jev's failures; Inherited ids below); a section that is no feature,
  `<entry>_<section>` (`ollama_options`).
- Inherited ids: a page or section that takes over the content of one moving into the entry inherits its id,
  feature or not, written `<id>__next` while the old one exists and renamed to `<id>`, with the links to it, in
  the commit that deletes the old one; `REGISTRY` stores the final ids (`ollama_first`, `jev_errors`). The
  inherited page titles: `providers/ollama/index.md` `ollama_guide__next`, `decisions/jev/index.md`
  `jev_start__next`, `providers/openai/responses.md` `responses_guide__next`, `providers/openai/retrieval.md`
  `retrieval_guide__next`, `decisions/choosing.md` `system_one_backends__next`.

## Placement

| Content | Goes to |
| :--- | :--- |
| a fact of one provider | that provider's entry, and nowhere else |
| a mechanism every provider shares | a concept page |
| a comparison of providers | the chapter's across-providers group |
| a signature | the API reference |
| the digest for coding agents | the generated regions of `docs/src/llm.md` and `llms.txt` |

- A new entry's start page holds the H2 of every feature of the entry, inherited ids as `<id>__next`. A section
  moves to another page of the same entry when its content belongs there, and its id moves with it: the id test
  in `docs/test_provider_facts.jl` asks for each id once, on one of the entry's pages.
- A section that is no feature (`<entry>_options`, `ollama_options`, `jev_box`) arrives with its content: a new one
  with the change that writes it, an inherited one as `<id>__next` with the change that moves it. It comes empty,
  ahead of its content, only where a later commit on the same branch writes it, before the branch merges; none
  ships empty.
- A page being dissolved stays in the sidebar, right after the page that replaces it, until the commit that
  deletes it adds its redirect to `docs/redirects.toml`.
- A fact a generated table shows is not restated in prose: restated, it goes stale when the code changes, while
  every docs build rebuilds the table from the code.
- A page explains with public, technical reasons, and names no internal planning document, task or finding.

## Generated facts

An `@eval` block runs in a bare module, so it calls a function below as `Main.ProviderFacts.f(…)`, a chapter
as `Main.ProviderFacts.llm` or `Main.ProviderFacts.decisions`, and returns a `Markdown.MD` or `nothing`.

| Call | Shows | Placed |
| :--- | :--- | :--- |
| `provider_table(chapter)` | the chapter's entries, each linked to its start page | on the chapter's start page |
| `at_a_glance(:<entry>)` | endpoint, key variables, default model, what the entry supports and, for an LLM entry, lacks | under the entry's identity line |
| `capability_matrix()` | every capability of every LLM entry | on `guide/capabilities.md` |
| `on_your_provider(chapter, :<feature>)` | the chapter's entries with the feature, each linked to its section | at the end of a concept page |
| `openai_model_rules()`, `claude_family_table()`, `gemini_thinking_table()` | the request rules of OpenAI's models; Claude's families; Gemini's thinking levels | in that entry's section whose rules it shows, or, where several sections need its columns, in one of them |
| `price_table(table)` | the prices of `:openai`, `:anthropic`, `:gemini`, `:deepseek` or `:jev` | in that entry's cost section |
| `claude_price_rules()` | the Claude models `estimated_cost` does not price at their row's rates alone: a long-prompt multiplier, or `0.0` for a model of the family table without a price row | in the Anthropic entry's cost section |

A per-model fact that UniLM's code holds reaches a page through a tested generator in `docs/ProviderFacts.jl`,
never through an inline `@eval` that reads UniLM internals. A generated table has no title: prose names it by
what it holds or where it stands ("the price table", "the table at the top of this page") and a column by its
header, and another section links the section the table is placed in. That section explains each column of a
model table once: in a clause, linking the section that holds the column's rule where that is another one; or,
where a column's cell texts each stand for a different request, in a table keyed on those texts as the generator
writes them (the Thinking cells on `providers/anthropic/requests.md`).

## Executed or quoted

An executed block (`@example`, `@repl`, `@setup`, `@eval`) reaches a service only through a verb the recordings
replay: `chatrequest!`, `respond`, `embeddingrequest!`, the tool loops, streamed calls, the six Responses lifecycle
verbs (`count_input_tokens`, `compact_response`, `get_response`, `list_input_items`, `delete_response`,
`cancel_response`), `fim_complete`, `prefix_complete`, `ask`, `ask_each` and the verbs built on them
(`nl_classify`, `nl_dispatch`, …), and TypeSafe's `list_models`. `docs/test_executed_verbs.jl` holds this list
and fails on an executed call to any other verb that reaches a service.

- Everything else (image generation, speech and transcription, files, batches, moderation, provider model
  listings, model pulls, realtime) is a `julia` block whose output is quoted from a live run, as a comment
  naming its date and versions; never an `output` block, a language the site keeps for executed output.
- No executed polling loop: execute `background=true` and `cancel_response`, and quote the poll.
- An entry without a key to record with (Azure, the OpenAI-compatible services, local servers when no local
  model runs) shows its calls as `julia` blocks without output and says so in its identity line; its provenance
  line names the versions its code was checked against.
- The keyless build replays every executed call and fails on one without a recording; `UNILM_DOCS_RECORD=1`
  records the missing ones (the top of `docs/make.jl`). A recording is never edited: delete it and record again.
- New recordings are checked against the page before they are committed: each field its prose relies on is in
  them (a cached call's cache tokens, a thinking example's thinking text, a stream's several deltas), and two calls
  whose difference the page shows, such as a cache write and then a read, are two recording files. A recording is
  filed under its request's key, the method, path and body as `with_recorded_answers` keys them (credentials
  redacted; on a System One backend, without the options that change only how a request is served): in a
  recording run, a later call with the key of a recorded request replays it and never reaches the service.
- A chain that passes a stored id on (a `store=true` response read back by `get_response`, …): changing any of
  its calls deletes the chain's first recording, found by its request text in `docs/recorded_answers`, in the
  same commit, so the next recording run makes the chain again from a new id.

## The `service, model` binding

A concept page's binding block, its first visible code block, binds `service, model = <Endpoint>, "<model>"` and
every further name its examples need; every other example uses only those names. `service` and `model` are the
provider and model the examples run on (on a Decisions page, the decision provider); a reader binds their own there.

- The pair is the chapter's: `OPENAIServiceEndpoint, "gpt-5.4-mini"` on a Language models page,
  `TYPESAFEServiceEndpoint, "jev-latest"` on a Decisions page.
- Only where the chapter's pair cannot serve the page's mechanism without provider-specific arguments (a concept
  page carries no provider caveat) does counting choose the pair: among the pairs that serve it without them, the
  one most of the page's examples call (counting blocks), preferring pairs that record, and where none records,
  among all. Today: embeddings `OPENAIServiceEndpoint, "text-embedding-3-small"`; image generation
  `OPENAIServiceEndpoint` with OpenAI's default image model; FIM and prefix completion
  `DeepSeekEndpoint(), "deepseek-flash"`.
- Further names: an LLM pair a Decisions page also needs (`llm, llm_model = OPENAIServiceEndpoint, "gpt-5.4-mini"`),
  the models a page chooses between (`strong_model, cheap_model = "gpt-5.6-sol", "gpt-5.4-mini"`) and a pinned
  version a page contrasts with the default (`pinned = "jev-1.13.0"`).
- An example of another provider moves to that provider's entry (the Gemini calls of `guide/agentic.md` go to the
  Gemini entry). No code outside the binding block names an endpoint or a model id (`docs/metrics.jl` counts each
  one). Prose may refer to the bound names; a fact about one provider goes to that provider's entry (Placement);
  the model a measurement ran on appears only as a bound name or in an Evidence note (a `!!! details "Evidence…"`
  box), whose body the page's 2% share of provider lines leaves out: `docs/metrics.jl` reads it as provenance, not
  prose.
- An example whose request bytes change is recorded again in the same commit, and a change in a stored-id chain
  deletes the chain's first recording.
