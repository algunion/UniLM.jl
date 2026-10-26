# UniLM.jl

*One Julia API for language models and decision models.*

[![CI](https://github.com/algunion/UniLM.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/algunion/UniLM.jl/actions/workflows/CI.yml)
[![codecov](https://codecov.io/gh/algunion/UniLM.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/algunion/UniLM.jl)
[![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)
[![](https://img.shields.io/badge/%F0%9F%9B%A9%EF%B8%8F_tested_with-JET.jl-233f9a)](https://github.com/aviatesk/JET.jl)
[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://algunion.github.io/UniLM.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://algunion.github.io/UniLM.jl/dev/)

UniLM.jl gives Julia one set of types and verbs for two kinds of model. A language model writes: an answer,
a call of one of your functions, JSON that fits a schema. A decision model reads a piece of state — a message,
a record, a picture — and answers the questions you list about it with a probability for each answer you
allowed, writing no text. Each kind has its chapter, and each provider an entry of its own: its setup, its first
call, its options, limits and failures, on its own pages.

## [Language models](@id home_llm)

A [`Chat`](@ref) holds a conversation and [`chatrequest!`](@ref) sends it; [`respond`](@ref) sends one request
of a provider's agentic API; embeddings, images, speech, transcription and fill-in-the-middle completion have
verbs of their own, on the providers that serve them. Choose your provider below: its name opens its entry.
[Language models](@ref llm_start) maps the concepts every provider shares.

```@eval
Main.ProviderFacts.provider_table(Main.ProviderFacts.llm)
```

## [Decisions](@id home_decisions)

[`ask`](@ref) sends the questions — which of these options, where on this scale, whether this statement holds —
and returns typed answers your code branches on, and [`nl_classify`](@ref) and [`nl_dispatch`](@ref) turn an
answer into the key or the method that runs. Choose your provider below: its name opens its entry.
[Decisions](@ref decisions_start) maps the concepts every decision provider shares, and
[Decisions with LLMs](@ref jev_llm_guide) puts a decision on each side of a language model's call.

```@eval
Main.ProviderFacts.provider_table(Main.ProviderFacts.decisions)
```

## [One API across providers](@id home_across)

The provider is one argument: `service` names its endpoint and `model` its model, so code written for one
provider runs on another once those two change: the same types and verbs, the same typed results, the same
cost accounting and the same [`RequestConfig`](@ref). [One API across providers](@ref backend_guide) says what every provider
shares, how to switch, and where the providers differ — which speak their own API and which the
OpenAI-compatible one; [Capabilities by provider](@ref capabilities_guide) shows what each provider serves; and
[Write your own backend](@ref custom_backends_guide) adds a provider UniLM does not ship. For decision models,
the same choice is [Choosing a decision provider](@ref system_one_backends).

## [Install](@id home_install)

UniLM requires Julia 1.13 or later and is registered in Julia's General registry:

```julia
using Pkg
Pkg.add("UniLM")
```

[Install](@ref getting_started) leads on to your provider's setup and its first call.
