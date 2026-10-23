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

The examples use only the names bound above; `service` and `model` are the provider and model this page is
about: bind your own provider's there to run the examples on it. The table [at the end](@ref <stem>_providers)
links each provider's rules.

> Template: The binding block (README, The `service, model` binding): `<Endpoint>`, `"<model>"` are the pair the page is about (on a Language models page an LLM's, on a Decisions page a decision provider's), one that serves the mechanism without provider-specific arguments; `<names> = <values>` binds a further name it needs, one line each (`llm, llm_model` for an LLM a Decisions page also needs, the models a choice is between), or is deleted. After the block no code names an endpoint or a model id; prose about a model may, within the page's 2% share of provider lines.

## [<Section>](@id <stem>_<section>)

> Template: One H2 per part of the mechanism; each example uses only the binding block's names (executed or quoted as the README says).

## [On your provider](@id <stem>_providers)

> Template: `:<feature>` is the feature the page explains; a page that explains none calls `Main.ProviderFacts.provider_table(Main.ProviderFacts.<chapter>)` instead.

```@eval
Main.ProviderFacts.on_your_provider(Main.ProviderFacts.<chapter>, :<feature>)
```
