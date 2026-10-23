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

The examples use the names bound above: bind your own provider's endpoint and model to run them there. The
table [at the end](@ref <stem>_providers) links each provider's rules.

> Template: The binding block (README, The `service, model` binding): `<Endpoint>`, `"<model>"` are the endpoint and model that serve the mechanism; `<names> = <values>` binds a further name it needs, one line each (a second pair, the models a choice is between), or is deleted. After the block, no example names an endpoint or a model id.

## [<Section>](@id <stem>_<section>)

> Template: One H2 per part of the mechanism; each example uses only the binding block's names (executed or quoted as the README says).

## [On your provider](@id <stem>_providers)

> Template: `:<feature>` is the feature the page explains; a page that explains none calls `Main.ProviderFacts.provider_table(Main.ProviderFacts.<chapter>)` instead.

```@eval
Main.ProviderFacts.on_your_provider(Main.ProviderFacts.<chapter>, :<feature>)
```
