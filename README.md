# UniLM.jl

[![CI](https://github.com/algunion/UniLM.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/algunion/UniLM.jl/actions/workflows/CI.yml)
[![codecov](https://codecov.io/gh/algunion/UniLM.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/algunion/UniLM.jl)
[![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)
[![](https://img.shields.io/badge/%F0%9F%9B%A9%EF%B8%8F_tested_with-JET.jl-233f9a)](https://github.com/aviatesk/JET.jl)
[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://algunion.github.io/UniLM.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://algunion.github.io/UniLM.jl/dev/)

One Julia API for two kinds of model. A **language model** writes: an answer, a call of one of your functions,
JSON that fits a schema. A **decision model** reads a piece of state — a message, a record, a picture — and
answers the questions you list about it with a probability for each answer, writing no text.

The provider is one argument: `service` names its endpoint and `model` its model, so code written for one
provider runs on another once those two change. Every provider gets the same types and verbs, the same typed
results — a success, a failure or a call error, never an exception for what the provider answered —, the same
token and cost accounting, and the same `RequestConfig` of timeouts and retries. OpenAI, Anthropic, Google
Gemini and Ollama are reached through their own APIs, so what a provider adds to its replies — Anthropic's
thinking blocks and their signatures, Gemini's thought signatures — goes back to it on the next turn; the rest
through the OpenAI-compatible API they serve. UniLM also speaks MCP, as a client and as a server, and records a
call once to replay it in tests with no key.
[One API across providers](https://algunion.github.io/UniLM.jl/dev/guide/multi_backend/) says what every
provider shares, how to switch, and where the providers differ.

## Install

UniLM requires Julia 1.13 or later and is registered in Julia's General registry:

```julia
using Pkg
Pkg.add("UniLM")
```

## Language models

A `Chat` holds a conversation and `chatrequest!` sends it; `respond` sends one request of a provider's agentic
API; embeddings, images, speech, transcription and fill-in-the-middle completion have verbs of their own, on the
providers that serve them. Each provider has an entry of its own — its setup, first call, options, limits and
failures —, which its name opens:

| Provider | `service` | Reads | Default model |
| :--- | :--- | :--- | :--- |
| [OpenAI](https://algunion.github.io/UniLM.jl/dev/providers/openai/) | `OPENAIServiceEndpoint` | `OPENAI_API_KEY` | `gpt-6-sol` |
| [Anthropic](https://algunion.github.io/UniLM.jl/dev/providers/anthropic/) | `ANTHROPICServiceEndpoint` | `ANTHROPIC_API_KEY` | `claude-opus-5-5` |
| [Google Gemini](https://algunion.github.io/UniLM.jl/dev/providers/gemini/) | `GEMINIServiceEndpoint` | `GEMINI_API_KEY` | `gemini-3.8-flash` |
| [Ollama](https://algunion.github.io/UniLM.jl/dev/providers/ollama/) | `OllamaEndpoint()` | `OLLAMA_HOST`, `OLLAMA_API_KEY` | none: a call names one |
| [vLLM, LM Studio & llama.cpp](https://algunion.github.io/UniLM.jl/dev/providers/local_servers/) | `GenericOpenAIEndpoint("http://localhost:8000", "")` | no variable | none: a call names one |
| [Azure OpenAI](https://algunion.github.io/UniLM.jl/dev/providers/azure/) | `AZUREServiceEndpoint` | `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_BASE_URL`, `AZURE_OPENAI_API_VERSION`, `AZURE_OPENAI_DEPLOY_NAME_<MODEL>` | `gpt-5.2` |
| [DeepSeek](https://algunion.github.io/UniLM.jl/dev/providers/deepseek/) | `DeepSeekEndpoint()` | `DEEPSEEK_API_KEY` | `deepseek-flash` |
| [Mistral & OpenAI-compatible services](https://algunion.github.io/UniLM.jl/dev/providers/openai_compatible/) | `MistralEndpoint()` | `MISTRAL_API_KEY` | none: a call names one |

The first call of the OpenAI entry, with the reply it recorded:

```julia
using UniLM

chat = Chat(service=OPENAIServiceEndpoint)
push!(chat, Message(Val(:system), "You are a concise assistant."))
push!(chat, Message(Val(:user), "In one sentence: what is a prime number?"))
result = chatrequest!(chat)
text(result)
# "A prime number is a whole number greater than 1 that has exactly two positive divisors: 1 and itself."
```

On Anthropic and Google Gemini only `service` changes, and each answers with its default model:

```julia
chat = Chat(service=ANTHROPICServiceEndpoint)
push!(chat, Message(Val(:system), "You are a concise assistant."))
push!(chat, Message(Val(:user), "In one sentence: what is a prime number?"))
result = chatrequest!(chat)
text(result)
# "A prime number is a whole number greater than 1 whose only positive divisors are 1 and itself, such as 2, 3, 5, or 7."

chat = Chat(service=GEMINIServiceEndpoint)
push!(chat, Message(Val(:system), "You are a concise assistant."))
push!(chat, Message(Val(:user), "In one sentence: what is a prime number?"))
result = chatrequest!(chat)
text(result)
# "A prime number is a whole number greater than 1 whose only factors are 1 and itself."
```

Ollama runs a model on your own machine, with no key and at no cost. It has no default model, so the call names
an installed one (`ollama pull gemma4:e4b`):

```julia
chat = Chat(service=OllamaEndpoint(), model="gemma4:e4b", reasoning_effort="none")
push!(chat, Message(Val(:system), "You are a concise assistant."))
push!(chat, Message(Val(:user), "In one sentence: why run a language model locally?"))
result = chatrequest!(chat)
text(result)
# "Running a language model locally provides greater privacy, reduces reliance on internet connectivity, and allows for fine-tuning or modification of the model without external service constraints."
```

[Language models](https://algunion.github.io/UniLM.jl/dev/providers/) maps the concepts every provider shares,
from conversations, tools and streaming to cost, timeouts, testing and MCP.

## Decisions

`ask` sends the questions — which of these options, where on this scale, whether this statement holds — and
returns typed answers your code branches on; `nl_classify` and `nl_dispatch` turn an answer into the key or the
method that runs. Use a decision model when the outcome is one of a set you can name and your code acts on it,
and a language model when the output is text a person reads. Each decision provider has an entry of its own:

| Provider | `service` | Reads | Default model |
| :--- | :--- | :--- | :--- |
| [Jev (TypeSafe)](https://algunion.github.io/UniLM.jl/dev/decisions/jev/) | `TYPESAFEServiceEndpoint` | `TYPESAFE_API_KEY`, `TYPESAFE_BASE_URL`, `TYPESAFE_DEFAULT_MODEL` | `jev-latest` |
| [OpenAI Decisions API](https://algunion.github.io/UniLM.jl/dev/decisions/openai/) | `OPENAIServiceEndpoint` | `OPENAI_API_KEY` | `gpt-6-luna` |
| [Local decision models](https://algunion.github.io/UniLM.jl/dev/decisions/ollama/) | `OllamaEndpoint()` | `OLLAMA_HOST`, `OLLAMA_API_KEY` | none: a call names one |

The same decision on each, from the first decision of each entry, with the answers they recorded:

```julia
using UniLM

const TEAM = (billing   = "payments, charges, invoices or refunds",
              technical = "the app or the website does not work as expected",
              shipping  = "a parcel that is late, lost or arrived damaged",
              other     = "anything else")
message = "I was charged twice for order #4471. Please refund the duplicate payment."

nl_classify(message, TEAM)                                                     # Jev, the default
# :billing
nl_classify(message, TEAM; service = OPENAIServiceEndpoint)                    # OpenAI's Decisions API
# :billing
nl_classify(message, TEAM; service = OllamaEndpoint(), model = "tev1:0.8b")    # a local decision model
# :billing
```

[Decisions](https://algunion.github.io/UniLM.jl/dev/decisions/) maps the concepts every decision provider
shares — the question kinds and decision rules, dispatch on meaning, many items at once, decisions around a
language model's call, and testing —, and
[Choosing a decision provider](https://algunion.github.io/UniLM.jl/dev/decisions/choosing/) compares the
providers.

## Documentation

**[https://algunion.github.io/UniLM.jl/dev/](https://algunion.github.io/UniLM.jl/dev/)**

- [Install](https://algunion.github.io/UniLM.jl/dev/getting_started/) — the package, your provider's setup, and
  a first call
- [Language models](https://algunion.github.io/UniLM.jl/dev/providers/) and
  [Decisions](https://algunion.github.io/UniLM.jl/dev/decisions/) — every provider of each kind, and the concepts
  they share
- [One API across providers](https://algunion.github.io/UniLM.jl/dev/guide/multi_backend/) — what every provider
  shares, how to switch, and where they differ
- [Capabilities by provider](https://algunion.github.io/UniLM.jl/dev/guide/capabilities/) — what each provider
  serves
- [Write your own backend](https://algunion.github.io/UniLM.jl/dev/guide/custom_backends/) and
  [Write your own decision provider](https://algunion.github.io/UniLM.jl/dev/decisions/custom/) — a provider
  UniLM does not ship
- [LLM Reference](https://algunion.github.io/UniLM.jl/dev/llm/) and [llms.txt](llms.txt) — for coding agents

## Versioning & Stability

UniLM is pre-1.0. While on `0.x`, **MINOR** releases (e.g. `0.13 → 0.14`) may carry breaking changes — each is listed under a **Breaking** heading in the [CHANGELOG](CHANGELOG.md) with migration notes — while **PATCH** releases never break. Breaking changes are batched into infrequent minors rather than dribbled across releases, and renamed identifiers keep working as aliases until at least `1.0`. [Versioning & Stability](https://algunion.github.io/UniLM.jl/dev/stability/) has the full contract.
