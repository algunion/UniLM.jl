# [Capabilities by provider](@id capabilities_guide)

```@setup capabilities
using UniLM
```

```@eval
Main.ProviderFacts.capability_matrix()
```

Each row is a capability symbol, as [`has_capability`](@ref) takes it
([Provider Capabilities](@ref capabilities_api)), and "yes" says that `has_capability` holds for some
endpoint of the entry. "sent unchecked" marks an entry whose endpoint is a `GenericOpenAIEndpoint`:
UniLM sends those requests without knowing whether the server serves them. No symbol names speech: the
row speech (`speak`) says where `speak` reaches.
