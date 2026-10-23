# [<Title>](@id <stem>_guide)

> Template: A concept page: a mechanism every provider shares, with no provider caveat (a provider's rule goes to its entry). A page rewritten in place keeps its ids; a new page, or a title without an id, takes `<stem>_guide`.

```@setup <stem>
using UniLM
```

> Template: One paragraph: what the mechanism does and when a reader needs it.

```@example <stem>
service, model = <Endpoint>, "<model>"
nothing # hide
```

The examples pass `service` and `model`, bound above: bind your own provider's endpoint and model to run
them there. The table [at the end](@ref <stem>_providers) links each provider's rules.

> Template: `<Endpoint>`, `"<model>"`: the endpoint and model that serve the mechanism (README, The `service, model` binding). After the binding, no example names an endpoint or a model id.

## [<Section>](@id <stem>_<section>)

> Template: One H2 per part of the mechanism; each example passes `service` and `model` (executed or quoted as the README says).

## [On your provider](@id <stem>_providers)

> Template: `:<feature>` is the feature the page explains; a page that explains none calls `Main.ProviderFacts.provider_table(Main.ProviderFacts.<chapter>)` instead.

```@eval
Main.ProviderFacts.on_your_provider(Main.ProviderFacts.<chapter>, :<feature>)
```
