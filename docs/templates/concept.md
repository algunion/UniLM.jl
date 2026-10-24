# [<Title>](@id <stem>_guide)

> Template: A concept page: a mechanism every provider shares, with no provider caveat (a provider's rule goes to its entry). A page rewritten in place keeps its ids; a new page, or a title without an id, takes `<stem>_guide`.

```@setup <stem>
using UniLM
```

> Template: One paragraph: what the mechanism does and when a reader needs it.

```@example <stem>
service, model = <Endpoint>, "<model>"
<names> = <values>
nothing # hide
```

The examples use only the names bound above. `service` and `model` are the provider and model they run on (on a
Decisions page, the decision provider): bind your own there. The table [at the end](@ref <stem>_providers) links
each provider's rules.

> Template: The binding block (README, The `service, model` binding): `<Endpoint>`, `"<model>"` are the chapter's pair (`OPENAIServiceEndpoint`, `"gpt-5.4-mini"`; on a Decisions page `TYPESAFEServiceEndpoint`, `"jev-latest"`) or, where it cannot serve the mechanism without provider-specific arguments, the pair counting chooses; `<names> = <values>` binds a further name, one line each (`llm, llm_model` for an LLM a Decisions page also needs, the models a choice is between, a pinned version), or is deleted. No code outside the block names an endpoint or a model id; prose may refer to the bound names, and the model a measurement ran on appears only as a bound name or in an Evidence note (a `!!! details "Evidence…"` box), whose body the page's 2% share of provider lines leaves out.

## [<Section>](@id <stem>_<section>)

> Template: One H2 per part of the mechanism; each example uses only the binding block's names (executed or quoted as the README says).

## [On your provider](@id <stem>_providers)

> Template: `:<feature>` is the feature the page explains; a page that explains none calls `Main.ProviderFacts.provider_table(Main.ProviderFacts.<chapter>)` instead.

```@eval
Main.ProviderFacts.on_your_provider(Main.ProviderFacts.<chapter>, :<feature>)
```
