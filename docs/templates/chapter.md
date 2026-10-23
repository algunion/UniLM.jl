# [<Chapter>](@id <chapter>_start)

> Template: A chapter's start page, `docs/src/providers/index.md` (`llm`) or `docs/src/decisions/index.md` (`decisions`): the one page of the chapter's "Start here" group, with no provider code.

```@setup <chapter>
using UniLM
```

> Template: One paragraph: what the chapter's models do, and when a reader wants them rather than the other chapter's.

## [Choose your provider](@id <chapter>_providers)

```@eval
Main.ProviderFacts.provider_table(Main.ProviderFacts.<chapter>)
```

## [Concepts](@id <chapter>_concepts)

> Template: The concept map: one list item per page of the chapter's concepts group, linked by the page's H1 id (`[<Title>](@ref …)`), with what it explains.

## [Across providers](@id <chapter>_across)

> Template: One list item per page of the chapter's across-providers group, as above.

> Template: Sidebar: in `docs/make.jl` the chapter is `"<Chapter>" => [groups…]`, a chapter holding groups only (the site writer refuses a page beside a group), in this order: "Start here" (this page alone), the provider groups (an entry's pages together, start page first), the concepts group, the across-providers group.
