# [Capabilities by provider](@id capabilities_guide)

```@setup capabilities
using UniLM
```

Each endpoint declares what it serves as a set of capability symbols — `:chat`, `:tools`, `:embeddings` and so
on —, and a verb checks the endpoint's set before anything is sent: on an endpoint whose declared set lacks the
capability the verb needs, it throws `ArgumentError`, naming the feature and the endpoint and listing what the
endpoint supports ([Provider Capabilities](@ref capabilities_api) says which verbs check which symbol). Read the
table below to see what a provider serves before you choose it, or before you move code to it.

```@example capabilities
service, model = OPENAIServiceEndpoint, "gpt-5.4-mini"
nothing # hide
```

The examples use only the names bound above. `service` and `model` are the provider and model they run on: bind
your own there. The table [at the end](@ref capabilities_providers) links each provider's start page.

## [Every provider's capabilities](@id capabilities_matrix)

```@eval
Main.ProviderFacts.capability_matrix()
```

Each row is a capability symbol, as [`has_capability`](@ref) takes it
([Provider Capabilities](@ref capabilities_api)), and "yes" says that `has_capability` holds for some
endpoint of the entry. "sent unchecked" marks an entry whose endpoint is a `GenericOpenAIEndpoint`:
UniLM sends those requests without knowing whether the server serves them. No symbol names speech: the
row speech (`speak`) says where `speak` reaches. The table leaves out `:system_one`, serving decision
models through [`ask`](@ref), which the entries of the [Decisions](@ref decisions_start) chapter list.

## [Check a provider in code](@id capabilities_check)

[`has_capability`](@ref) answers for one symbol, and [`provider_capabilities`](@ref) gives the endpoint's
whole set:

```@example capabilities
has_capability(service, :tools), has_capability(service, :fim)
```

```@example capabilities
sort!(collect(provider_capabilities(service)))
```

An endpoint of your own declares a set by adding a method to `provider_capabilities`; one that declares
none is not checked by [`chatrequest!`](@ref), [`respond`](@ref), [`embeddingrequest!`](@ref) and the
image verbs, and the other verbs refuse it ([Write your own backend](@ref custom_backends_guide)).

## [On your provider](@id capabilities_providers)

```@eval
Main.ProviderFacts.provider_table(Main.ProviderFacts.llm)
```
