# [Decisions](@id decisions_start)

```@setup decisions
using UniLM
```

## [Choose your provider](@id decisions_providers)

```@eval
Main.ProviderFacts.provider_table(Main.ProviderFacts.decisions)
```

## [Concepts](@id decisions_concepts)

- [Route and Decide](@ref system_one_guide)
- [Dispatch on Meaning](@ref nl_dispatch_guide)
- [Many Items at Once](@ref jev_algorithms_guide)
- [Decisions with LLMs](@ref jev_llm_guide)
- [Test and Develop](@ref jev_testing_guide)

## [Across providers](@id decisions_across)

- [Choosing a decision provider](@ref system_one_backends__next)
- [Write your own decision provider](@ref decisions_custom)
