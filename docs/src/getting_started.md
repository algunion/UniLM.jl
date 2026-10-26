# [Install](@id getting_started)

## [Add the package](@id getting_started_package)

UniLM requires Julia 1.13 or later and is registered in Julia's General registry:

```julia
using Pkg
Pkg.add("UniLM")
```

Or from the Pkg REPL:

```
pkg> add UniLM
```

To track the latest unreleased changes, install from GitHub instead:

```julia
Pkg.add(url="https://github.com/algunion/UniLM.jl")
```

## [Set up your provider](@id getting_started_keys)

A hosted provider reads its API key from an environment variable; set it in the environment Julia starts
from, for example:

```bash
export OPENAI_API_KEY="sk-..."
```

A model you run yourself — through Ollama, vLLM, LM Studio or llama.cpp — needs no key unless its server
was started with one: the setup is where the server listens. Each provider's Setup section names what it
reads and where its requests go; the provider's name below opens it.

```@eval
Main.ProviderFacts.on_your_provider(Main.ProviderFacts.llm, :setup)
```

The decision providers:

```@eval
Main.ProviderFacts.on_your_provider(Main.ProviderFacts.decisions, :setup)
```

## [Make a first call](@id getting_started_first_call)

Each provider's first call is a short example on its entry:

```@eval
Main.ProviderFacts.on_your_provider(Main.ProviderFacts.llm, :first_call)
```

The first decision on each decision provider:

```@eval
Main.ProviderFacts.on_your_provider(Main.ProviderFacts.decisions, :first_decision)
```

## [Handle the result](@id getting_started_results)

A request to a model returns its outcome as a typed result rather than throwing it: a success holding the
reply, a failure holding what the provider answered with, or a call error holding what ended the call.
[Results](@ref results_guide) shows the types each verb returns and how to read them; four of the Responses
lifecycle verbs — `delete_response`, `list_input_items`, `compact_response` and `count_input_tokens` — return
the provider's answer as a `Dict` on success.

## [Next steps](@id getting_started_next)

- [Language models](@ref llm_start) and [Decisions](@ref decisions_start) — every provider of each kind, and the
  concepts they share.
- [One API across providers](@ref backend_guide) — what every language-model provider shares, how to switch, and
  where they differ.
- [Choosing a decision provider](@ref system_one_backends) — the same choice for decision models.
