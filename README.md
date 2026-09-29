# UniLM.jl

[![CI](https://github.com/algunion/UniLM.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/algunion/UniLM.jl/actions/workflows/CI.yml)
[![codecov](https://codecov.io/gh/algunion/UniLM.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/algunion/UniLM.jl)
[![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)
[![](https://img.shields.io/badge/%F0%9F%9B%A9%EF%B8%8F_tested_with-JET.jl-233f9a)](https://github.com/aviatesk/JET.jl)
[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://algunion.github.io/UniLM.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://algunion.github.io/UniLM.jl/dev/)

A **Julian**, type-safe interface to **LLM providers** with **first-class native backends** — OpenAI (Chat Completions + Responses), Anthropic (Messages), Google Gemini (generateContent + agentic Interactions), and local models through Ollama's native API — plus any **OpenAI-compatible** provider (Azure, DeepSeek, Mistral, vLLM, LM Studio). Covers the **Chat Completions** & **Responses** APIs, a cross-provider agentic **`respond`** verb, **Image Generation/Edits**, **Embeddings**, **Files/Vector Stores**, **Conversations**, **Audio**, **Batch**, **Moderations**, **Fine-tuning**, **Webhooks**, **Realtime**, and **MCP** (client & server) — with built-in token/cost accounting and illegal states made unrepresentable.

## When to choose UniLM

UniLM speaks each provider's own wire API, not just the OpenAI-compatible protocol. Reach for it when you need:

- **Native Anthropic and Gemini backends** — each speaks the provider's own format (Anthropic Messages, Gemini `generateContent`) rather than an OpenAI-compat shim, and round-trips provider-verbatim content, so reasoning state such as Anthropic thinking signatures and Gemini thought signatures survives across turns.
- **An MCP client _and_ an MCP server in one package** — connect to external MCP servers (`MCPSession`) with tool-loop integration, and expose your own Julia functions as tools over MCP (`MCPServer`).
- **Typed results with fail-loud invariants** — every call resolves to a concrete `LLMSuccess` / `LLMFailure` / `LLMCallError` result (with matching `Response…` types for the Responses API); a timeout or a cancellation comes back as a typed call-error result carrying `UniLMTimeout` / `UniLMCancelled` in `cause`, never a silent default; local validation (an option the provider cannot express, a conversation the reply could not be appended to) throws before any request; and invalid conversation mutations raise a typed `InvalidConversationError` rather than corrupting the conversation.
- **Built-in per-conversation cost accounting** — provider token counts are normalized to one shape, and each `Chat` accumulates a running USD estimate you read with `cumulative_cost`.
- **Broad OpenAI platform-API coverage** — well beyond chat: Responses, Images, Embeddings, Files, Vector Stores, Conversations, Audio, Batch, Moderations, Fine-tuning, Webhooks, and Realtime.

## Features

- **Chat Completions** — stateful conversations with automatic history management
- **Responses API & Agentic Verb** — OpenAI's Responses API with built-in tools, multi-turn chaining, and reasoning; the unified `respond` verb also drives Google's Gemini Interactions
- **Image Generation & Edits** — create and edit images with `gpt-image-2`
- **Tool/Function Calling** — first-class support for function tools in both APIs, with automated `tool_loop`
- **MCP (Model Context Protocol)** — connect to MCP servers or build your own, with seamless tool loop integration
- **Jev (TypeSafe System One)** — decisions about a text instead of generated text: `nl_classify` returns one of your own keys, and `ask` answers `choice` / `score` / `noul` questions with calibrated probabilities, many questions in one request
- **Dispatch on Meaning** — `nl_dispatch` runs the method whose sentence fits the text: short keys with a table of sentences (`texts = TEAM`, methods on `Val{:billing}`), or `nl"..."` in the signature; `@branch` is the same decision inline
- **Jev with LLMs** — route a message before an LLM call and check the draft after it; `with_recorded_answers` records Jev's answers and the non-streaming LLM calls once and replays them, so tests run without keys
- **Embeddings** — text embedding generation with `text-embedding-3-small`
- **Files, Vector Stores & Conversations** — upload files, build vector stores for `file_search`, and manage server-side conversation state
- **Audio, Batch & Moderations** — TTS/transcription, async 50%-off bulk jobs, and free safety classification
- **Realtime, Fine-tuning, Webhooks, Containers & Uploads** — WebSocket realtime, custom models, signed-webhook verification, and more. See provider availability limits for [fine-tuning](docs/src/api/fine_tuning.md).
- **Not wrapped in this release: OpenAI Agents API** — public beta since September 10, 2026; UniLM has no wrapper for it yet.
- **Not wrapped in this release: GPT-Live sessions** — `v1/live/sessions` (GPT-Live 1); the Realtime wrappers do not cover it.
- **Streaming** — real-time token streaming with `do`-block syntax
- **Structured Output** — JSON Schema–constrained generation
- **Multi-Backend** — OpenAI, Azure, Gemini, Anthropic, DeepSeek, Mistral, vLLM, LM Studio, and any OpenAI-compatible provider
- **Local models** — Ollama's native API with Gemma 4: thinking, tools, structured output, images, audio, embeddings, a context window of your choosing, model management — at no cost
- **Type Safety** — invalid states are unrepresentable; tested with [JET.jl](https://github.com/aviatesk/JET.jl) and [Aqua.jl](https://github.com/JuliaTesting/Aqua.jl)

## Installation

UniLM requires **Julia 1.13+** and is registered in Julia's General registry:

```julia
using Pkg
Pkg.add("UniLM")
```

Or in the Pkg REPL:

```
pkg> add UniLM
```

For the latest unreleased changes, install directly from GitHub:

```julia
Pkg.add(url="https://github.com/algunion/UniLM.jl")
```

## Quick Start

> 💡 **Costs & free local option:** hosted API calls bill your provider key. To try UniLM for free with no key, run a local model with Ollama (`ollama pull gemma4:e4b`, then `service=OllamaEndpoint()`) — see [Local Models with Ollama](https://algunion.github.io/UniLM.jl/dev/guide/ollama/).

Set your API key:

```bash
export OPENAI_API_KEY="sk-..."
```

### Responses API (recommended for new code)

```julia
julia> using UniLM, JSON

julia> result = respond("Explain Julia's multiple dispatch in 2-3 sentences.")

julia> output_text(result)
"Julia's multiple dispatch means a function can have many method definitions, and Julia chooses which one to run based on the types of *all* arguments in a call (not just the first). This makes it easy to write generic code while still getting specialized, high-performance behavior for specific type combinations."

julia> result.response.model
"gpt-5.6-sol"
```

### Chat Completions

```julia
julia> chat = Chat(model="gpt-5.4-mini")

julia> push!(chat, Message(Val(:system), "You are a concise Julia programming tutor. Answer in plain text, without Markdown."))

julia> push!(chat, Message(Val(:user), "What is multiple dispatch? Answer in 2-3 sentences."))

julia> result = chatrequest!(chat)

julia> println(text(result))
Multiple dispatch is a feature in programming languages, including Julia, that allows the selection of a method to execute based on the types of all its arguments, rather than just the first one. This enables more flexible and expressive code, as it can define different behaviors for a function depending on the combination of argument types. It supports polymorphism, making it easier to write generic code that works with multiple types.

julia> length(chat)  # system + user + assistant
3
```

Use `issuccess(result)` / `isfailure(result)` to branch on the outcome; `text(result)` returns the reply text and throws `LLMResultError` on a failed call.

### One-Shot Convenience

```julia
julia> result = chatrequest!(
           systemprompt="You are a calculator. Respond only with the number.",
           userprompt="What is 42 * 17?",
           model="gpt-5.4-mini",
           temperature=0.0
       )

julia> println(text(result))
714
```

### Image Generation

```julia
julia> result = generate_image(
           "A watercolor painting of a friendly robot reading a Julia programming book",
           size="1024x1024", quality="medium"
       )

julia> result isa ImageSuccess
true

julia> save_image(image_data(result)[1], "robot_julia.png")
"robot_julia.png"
```

### Embeddings

```julia
julia> emb = Embeddings("Julia is a high-performance programming language for technical computing.")

julia> embeddingrequest!(emb)

julia> emb.embeddings[1:5]
5-element Vector{Float64}:
  -0.039474
  -0.009283
   0.001706
  -0.028087
   0.063363
```

### Streaming

```julia
julia> task = respond("Write a haiku about Julia programming.") do chunk, close
           if chunk isa String
               print(chunk)
           elseif chunk isa ResponseObject
               println("\nDone! Status: ", chunk.status)
           end
       end
Multiple dispatch sings,
Types align in swift fusion—
Loops bloom into speed.
Done! Status: completed
```

### Structured Output

```julia
julia> fmt = json_schema_format(
           "languages", "A list of programming languages",
           Dict(
               "type" => "object",
               "properties" => Dict(
                   "languages" => Dict(
                       "type" => "array",
                       "items" => Dict(
                           "type" => "object",
                           "properties" => Dict(
                               "name" => Dict("type" => "string"),
                               "year" => Dict("type" => "integer"),
                               "paradigm" => Dict("type" => "string")
                           ),
                           "required" => ["name", "year", "paradigm"],
                           "additionalProperties" => false
                       )
                   )
               ),
               "required" => ["languages"],
               "additionalProperties" => false
           ),
           strict=true
       )

julia> result = respond("List Julia, Python, and Rust with their release year and primary paradigm.", text=fmt)

julia> JSON.parse(output_text(result))
{
  "languages": [
    {"name": "Julia", "year": 2012, "paradigm": "Multi-paradigm (scientific/numerical, functional, concurrent)"},
    {"name": "Python", "year": 1991, "paradigm": "Multi-paradigm (object-oriented, imperative, functional)"},
    {"name": "Rust", "year": 2010, "paradigm": "Multi-paradigm (systems programming, functional, imperative)"}
  ]
}
```

### Tool / Function Calling

**Responses API:**

```julia
julia> weather_tool = function_tool(
           "get_weather", "Get the current weather for a given location",
           parameters=Dict(
               "type" => "object",
               "properties" => Dict(
                   "location" => Dict("type" => "string", "description" => "City name"),
                   "unit" => Dict("type" => "string", "enum" => ["celsius", "fahrenheit"])
               ),
               "required" => ["location", "unit"],
               "additionalProperties" => false
           ),
           strict=true
       )

julia> result = respond("What's the weather in Tokyo? Use celsius.", tools=[weather_tool])

julia> calls = function_calls(result)

julia> calls[1]["name"]
"get_weather"

julia> JSON.parse(calls[1]["arguments"])
{"location": "Tokyo", "unit": "celsius"}
```

**Web Search:**

```julia
julia> result = respond(
           "What is the latest stable release of the Julia programming language? Answer in one plain-text sentence.",
           tools=[web_search()]
       )

julia> output_text(result)
"The latest **stable** release of the Julia programming language is **Julia v1.12.5**."
```

### Multi-Turn Conversations

**Responses API** (via `previous_response_id`):

```julia
julia> r1 = respond("Tell me a one-liner programming joke.", instructions="Be concise.")

julia> output_text(r1)
"There are only 10 kinds of people in the world: those who understand binary and those who don't."

julia> r2 = respond("Explain why that's funny, in one sentence.", previous_response_id=r1.response.id)

julia> output_text(r2)
"It's funny because \"10\" looks like ten in decimal but equals two in binary, so it sets up a nerdy misdirection that only people who know binary immediately get."
```

## Multi-Backend Support

UniLM.jl is built around **neutral verbs**: the same `Chat` + `chatrequest!` (tools, streaming, and cost accounting included — cost needs a price row, so models outside the built-in table estimate at \$0 with a warning) run unchanged across OpenAI, Anthropic, Gemini, DeepSeek, and any OpenAI-compatible provider — you only change `service`. The agentic `respond` verb is neutral the same way across OpenAI (Responses) and Gemini (Interactions). Native OpenAI/Anthropic/Gemini are first-class backends with their own wire formats (each exercised by live integration tests), not OpenAI-compatible shims. Switch via the `service` parameter:

| Backend          | Type                             | Env Variables                                                               |
| :--------------- | :------------------------------- | :-------------------------------------------------------------------------- |
| OpenAI (default) | `OPENAIServiceEndpoint`          | `OPENAI_API_KEY`                                                            |
| Azure OpenAI     | `AZUREServiceEndpoint`           | `AZURE_OPENAI_BASE_URL`, `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_API_VERSION` |
| Google Gemini    | `GEMINIServiceEndpoint`          | `GEMINI_API_KEY`                                                            |
| Anthropic        | `ANTHROPICServiceEndpoint`       | `ANTHROPIC_API_KEY`                                                         |
| DeepSeek         | `DeepSeekEndpoint()`             | `DEEPSEEK_API_KEY`                                                          |
| Ollama (local)   | `OllamaEndpoint()`               | none                                                                        |
| Mistral          | `MistralEndpoint()`              | `MISTRAL_API_KEY`                                                           |
| Any OpenAI-compat | `GenericOpenAIEndpoint(url, key)` | custom                                                                      |

```julia
# Azure
chat = Chat(service=AZUREServiceEndpoint, model="gpt-5.2")

# Gemini (native generateContent)
chat = Chat(service=GEMINIServiceEndpoint)          # default: gemini-3.8-flash

# Anthropic (native Messages API)
chat = Chat(service=ANTHROPICServiceEndpoint)       # default: claude-opus-5-5

# DeepSeek
chat = Chat(service=DeepSeekEndpoint())             # default: deepseek-flash

# Ollama (local, native API): context window, keep-alive, thinking, images and audio
chat = Chat(service=OllamaEndpoint(num_ctx=32_768), model="gemma4:e4b")
```

TypeSafe's System One endpoint (`TYPESAFEServiceEndpoint`, `TYPESAFE_API_KEY`) is deliberately absent from that table: it answers enumerated questions rather than generating text, so `chatrequest!`, `respond`, `embeddingrequest!` and the other platform verbs reject it up front with an `ArgumentError` (naming it on a `Chat` or an `Embeddings` is allowed only with an explicit `model=` — omitting it throws `ArgumentError` — and sending the request is not). It has its own section below.

## Jev: Decisions About Text (TypeSafe System One)

An LLM writes text; [TypeSafe](https://docs.typesafe.ai)'s System One model **Jev** reads a text and decides — which team, how urgent, is it safe to send, does it match the source — answering only the questions you list, with a probability for each answer, in one request billed only for the text it reads ([Models](https://docs.typesafe.ai/models)).
Set `export TYPESAFE_API_KEY="..."` and pick a path:

| I want to… | Use | You get | Read |
| :--- | :--- | :--- | :--- |
| label a text with one of my own values | `nl_classify(text, TEAM)` | the key of the sentence that fits, e.g. `:billing` | [Start Here: Jev in Five Minutes](https://algunion.github.io/UniLM.jl/dev/guide/jev_start/) |
| run the right function for a text | `nl_dispatch` with a table of sentences, or `nl"..."` in the signatures | what the method of the chosen meaning returns | [Dispatch on Meaning](https://algunion.github.io/UniLM.jl/dev/guide/natural_language_dispatch/) |
| ask several things about one text, and act only when Jev is sure | `ask` with `choice` / `score` / `noul`, then `min_confidence` or a `decide` policy | every answer from one request, with its probabilities | [Route and Decide](https://algunion.github.io/UniLM.jl/dev/guide/system_one/) |
| decide before an LLM call, or check its output after | a Jev question on each side of `respond` | the team, the model to call, a draft checked before it is sent | [Jev with LLMs](https://algunion.github.io/UniLM.jl/dev/guide/jev_with_llms/) |

**Label a text** with one of your own keys. Jev reads the sentences, never the keys:

```julia
using UniLM

const TEAM = (billing   = "payments, charges, invoices or refunds",
              technical = "the app or the website does not work as expected",
              shipping  = "a parcel that is late, lost or arrived damaged",
              other     = "anything else")

nl_classify("I was charged twice for order #4471. Please refund the duplicate payment.", TEAM)
# => :billing

# Unsure goes to a person, not to a guess (Jev's confidence here: 0.49, below the 0.6 asked for):
nl_classify("Do you ship to Norway, and how much does delivery cost?", TEAM;
            min_confidence = 0.6, fallback = message -> :person)
# => :person
```

**Run the right function**, one method per key; `texts = TEAM` supplies the sentences, and a key with no method is refused before any request:

```julia
route(::Val{:billing}, message)   = "opened a payment review"
route(::Val{:technical}, message) = "filed a bug report"
route(::Val{:shipping}, message)  = "opened a claim with the carrier"
route(::Val{:other}, message)     = "forwarded to the front desk"

nl_dispatch(route, "The app crashes every time I open my order history. iPhone 15, latest version."; texts = TEAM)
# => "filed a bug report"
```

**Or put each sentence in the signature**: `nl"..."` is a Julia type, so the router needs no table; for the same message both forms send the same request, byte for byte. `@branch` is the same decision inline, for one call site:

```julia
route_by_sentence(::nl"payments, charges, invoices or refunds", message)           = "opened a payment review"
route_by_sentence(::nl"the app or the website does not work as expected", message) = "filed a bug report"
route_by_sentence(::nl"a parcel that is late, lost or arrived damaged", message)    = "opened a claim with the carrier"
route_by_sentence(::nl"anything else", message)                                    = "forwarded to the front desk"

nl_dispatch(route_by_sentence, "I was charged twice for order #4471. Please refund the duplicate payment.")
# => "opened a payment review"
```

**Check an LLM's draft** before the customer sees it: the LLM writes, Jev reads, your code decides:

```julia
reply = output_text(respond("The phone I received has a cracked screen and the box was crushed. I want my money back.";
                            model = "gpt-5.4-mini",
                            instructions = "You answer customers of a small online shop, in at most three sentences."))
# => "I’m sorry your order arrived damaged. Please send us a photo of the cracked screen and the crushed box along with your order number, and we’ll help arrange a refund right away."

ask(reply, "refund" => noul("Does this reply promise the customer a refund?"))["refund"].noul
# => 0.81 — above a cut of 0.3, so a person approves the reply before it is sent
```

The Jev guides, most of them built around one small shop's support desk: [Start Here](https://algunion.github.io/UniLM.jl/dev/guide/jev_start/), [Jev with LLMs](https://algunion.github.io/UniLM.jl/dev/guide/jev_with_llms/), [Route and Decide](https://algunion.github.io/UniLM.jl/dev/guide/system_one/), [Dispatch on Meaning](https://algunion.github.io/UniLM.jl/dev/guide/natural_language_dispatch/), [Many Items at Once](https://algunion.github.io/UniLM.jl/dev/guide/semantic_algorithms/) and [Test and Develop](https://algunion.github.io/UniLM.jl/dev/guide/jev_testing/).

## Chat Completions vs Responses (OpenAI)

UniLM speaks each provider's own API (see [Multi-Backend Support](#multi-backend-support)). For **OpenAI**, you can use either of two conversational APIs. **Chat Completions** (`Chat` + `chatrequest!`) is the portable path — it's also how the native Anthropic and Gemini backends and every OpenAI-compatible provider work. **Responses** (`respond`) is OpenAI's newer API, and the basis for the cross-provider agentic verb (which also targets Gemini Interactions). They map like this:

| Feature                |       Chat Completions       |            Responses API            |
| :--------------------- | :--------------------------: | :---------------------------------: |
| Stateful conversations |       `Chat` + `push!`       |       `previous_response_id`        |
| System prompt          | `Message(Val(:system), ...)` |        `instructions` kwarg         |
| Tool calling           |  `Tool` / `ToolCall`   |  `FunctionTool` / `function_tool`   |
| Web search             |    `web_search_options`      |           `WebSearchTool`           |
| File search            |              —               |          `FileSearchTool`           |
| Streaming              |   `stream=true` + callback   |          `do`-block syntax          |
| Structured output      |       `ResponseFormat`       | `TextConfig` / `json_schema_format` |
| Reasoning              |     `reasoning_effort`       |             `Reasoning`             |
| Automated tool loop    |       `tool_loop!`           |          `tool_loop`                |
| MCP integration        |    `mcp_tools` bridge        |   `MCPTool` / `mcp_tool`            |

## Documentation

Full documentation with guides and API reference: **[https://algunion.github.io/UniLM.jl/dev/](https://algunion.github.io/UniLM.jl/dev/)**

- [Getting Started](https://algunion.github.io/UniLM.jl/dev/getting_started/) — setup and first requests
- [Chat Completions Guide](https://algunion.github.io/UniLM.jl/dev/guide/chat_completions/) — `Chat` and `chatrequest!`
- [Responses API Guide](https://algunion.github.io/UniLM.jl/dev/guide/responses_api/) — the newer Responses API
- [Image Generation Guide](https://algunion.github.io/UniLM.jl/dev/guide/image_generation/) — create images from text
- [Tool Calling Guide](https://algunion.github.io/UniLM.jl/dev/guide/tool_calling/) — function calling
- [Agentic Workflows Guide](https://algunion.github.io/UniLM.jl/dev/guide/agentic/) — cross-provider `respond` (OpenAI Responses + Gemini Interactions)
- [Streaming Guide](https://algunion.github.io/UniLM.jl/dev/guide/streaming/) — real-time streaming
- [Structured Output Guide](https://algunion.github.io/UniLM.jl/dev/guide/structured_output/) — JSON Schema output
- [Multi-Backend Guide](https://algunion.github.io/UniLM.jl/dev/guide/multi_backend/) — Azure, Gemini, DeepSeek, Ollama, and more
- [Local Models with Ollama](https://algunion.github.io/UniLM.jl/dev/guide/ollama/) — Gemma 4 on your own machine
- [MCP Guide](https://algunion.github.io/UniLM.jl/dev/guide/mcp/) — MCP client/server
- [Start Here: Jev in Five Minutes](https://algunion.github.io/UniLM.jl/dev/guide/jev_start/) — what Jev decides, a first decision, and which Jev page answers your question
- [Jev with LLMs](https://algunion.github.io/UniLM.jl/dev/guide/jev_with_llms/) — route a message before an LLM call and check the draft after it
- [Route and Decide](https://algunion.github.io/UniLM.jl/dev/guide/system_one/) — `ask` with `choice` / `score` / `noul`, calibrated answers, and decision rules on them
- [Dispatch on Meaning](https://algunion.github.io/UniLM.jl/dev/guide/natural_language_dispatch/) — `nl_classify`, `nl_dispatch` with a table of sentences or `nl"..."` in the signatures, `@branch`
- [Many Items at Once](https://algunion.github.io/UniLM.jl/dev/guide/semantic_algorithms/) — many items in one request, ranking, finding an event in a long sequence, matching records, stopping early
- [Test and Develop](https://algunion.github.io/UniLM.jl/dev/guide/jev_testing/) — recorded answers for Jev and LLM calls (`with_recorded_answers`), tests that need no key, an audit trail
- [Timeouts & Retries Guide](https://algunion.github.io/UniLM.jl/dev/guide/timeouts/) — bounds, typed failures, retry contracts
- [Concurrency, Tasks and Cancellation](https://algunion.github.io/UniLM.jl/dev/guide/concurrency/) — sharing rules, fan-out, streaming into a `Channel`, `CancelToken`

## Timeouts & Concurrency

Every network operation waits on a peer only under a bounded, configurable limit,
and reports a breach as a typed error. All bounds live on one `RequestConfig`,
resolved per call:

```julia
# Per call
chatrequest!(chat; config=RequestConfig(request_timeout=60.0, max_attempts=1))

# For a block of calls (propagates into spawned tasks)
with_request_config(request_timeout=30.0) do
    chatrequest!(chat)
    embeddingrequest!(emb)
end

set_default_config!(stream_idle_timeout=300.0)   # process-wide, for notebooks
```

A timeout surfaces as the call's usual error result with `status = nothing` and a
`UniLMTimeout` (`phase`, `elapsed`, `limit`) on `.cause` — every `*CallError` carries
`cause` — never a hang and never a fabricated HTTP status. `max_attempts` (default 3)
applies to the inference verbs (`chatrequest!`, `embeddingrequest!`, `respond`,
`generate_image`, `edit_image`, `fim_complete`, `prefix_complete`, `ask`,
`list_models`); platform and lifecycle verbs, `upload_file` included, make a single
bounded attempt. A `Retry-After` header is a floor under the jittered backoff, so a
rate-limited batch does not retry in lockstep.

Any HTTP call can be cancelled cooperatively from another task (the Realtime WebSocket
and MCP stdio exchanges ignore the token):

```julia
tok = CancelToken()
task = Threads.@spawn chatrequest!(chat; cancel=tok)   # or with_cancel(tok) do … end
cancel!(tok)                                           # e.g. the user pressed "stop"
fetch(task)          # still in flight when cancelled: an LLMCallError whose cause is UniLMCancelled
```

Two concurrency rules are worth knowing before you fan out:

- **One `Chat` per in-flight call.** A `Chat` is unsynchronized mutable state, so
  use `fork(chat)` / `fork(chat, n)` to fan out rather than sharing one. The same
  holds for an `Embeddings`, which `embeddingrequest!` fills in place. `respond` and
  `generate_image` do not mutate their request, so one may be shared.
- **An `MCPSession` runs one call at a time**, first come first served, and a call's
  `timeout` bounds its wait for the session — open one session per parallel worker.

Each stream uses its own HTTP/1.1 connection, so a slow consumer on one stream cannot
stall another. The [Timeouts & Retries guide](https://algunion.github.io/UniLM.jl/dev/guide/timeouts/)
has the full bound contract and the
[Concurrency, Tasks and Cancellation guide](https://algunion.github.io/UniLM.jl/dev/guide/concurrency/)
the sharing, fan-out and cancellation rules.

## Versioning & Stability

UniLM is pre-1.0. While on `0.x`, **MINOR** releases (e.g. `0.13 → 0.14`) may carry breaking changes — each is listed under a **Breaking** heading in the [CHANGELOG](CHANGELOG.md) with migration notes — while **PATCH** releases never break. Breaking changes are batched into infrequent minors rather than dribbled across releases, and renamed identifiers keep working as aliases until at least `1.0`. See the [Versioning & Stability policy](https://algunion.github.io/UniLM.jl/dev/stability/) for the full contract.
