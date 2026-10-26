# [One API across providers](@id backend_guide)

```@setup multi_backend
using UniLM
```

UniLM reaches every language-model provider through one set of types and verbs. The provider is one argument:
`service` names its endpoint and `model` its model, so code written for one provider runs on another once those
two change. This page says what every provider shares, how to switch, and where the providers differ; each
provider's own rules are on its entry, which the table at the end links.

```@example multi_backend
service, model = OPENAIServiceEndpoint, "gpt-5.4-mini"
nothing # hide
```

The examples use only the names bound above. `service` and `model` are the provider and model they run on: bind
your own there. The table [at the end](@ref multi_backend_providers) links each provider's start page.

## [What every provider shares](@id multi_backend_shared)

A function written once takes the provider as its argument:

```@example multi_backend
function explain(service, model)
    chat = Chat(service=service, model=model)
    push!(chat, Message(Val(:system), "You are a concise Julia programming tutor. Answer in plain text, without Markdown."))
    push!(chat, Message(Val(:user), "What is multiple dispatch? Answer in 2-3 sentences."))
    result = chatrequest!(chat)
    result isa LLMSuccess || error("the request failed: $result")
    result
end

result = explain(service, model)
text(result)
```

```@example multi_backend
u = token_usage(result)
(; u.prompt_tokens, u.completion_tokens, cost = round(estimated_cost(result); digits=6))
```

Whatever `service` names:

- **The types and verbs.** A [`Chat`](@ref) holds a conversation, with its tools, streaming, structured
  output and attachments, and [`chatrequest!`](@ref) sends it ([Conversations with Chat](@ref chat_guide)); a
  [`Respond`](@ref) is one request of a provider's agentic API, which [`respond`](@ref) sends where the
  provider has one ([Agentic respond](@ref agentic_guide)). Embeddings, images, FIM and audio have verbs of
  their own, on the providers that serve them.
- **The results.** A call returns a typed success, failure or call error rather than throwing for what the
  provider answers ([Results](@ref results_guide)); its usage is a [`TokenUsage`](@ref), which
  [`estimated_cost`](@ref) prices ([Cost Tracking](@ref cost_guide)).
- **Time, retries and cancellation.** A [`RequestConfig`](@ref) sets a call's timeouts and retries, and a
  [`CancelToken`](@ref) stops it ([Timeouts & Retries](@ref timeouts_guide),
  [Concurrency, Tasks and Cancellation](@ref concurrency_guide)); where a provider's models start from
  other bounds, its entry says so.
- **Recordings.** [`with_recorded_answers`](@ref) records a call of the verbs it covers — chat, `respond`,
  embeddings, FIM and prefix completion, the Responses lifecycle — once and replays it with no key and
  no network ([Testing and recordings](@ref testing_guide)).
- **Refusals before sending.** A verb refuses an endpoint whose declared capabilities lack it, and the
  request's encoder refuses what the provider's rules name, each with an `ArgumentError` before anything is
  sent ([Capabilities by provider](@ref capabilities_guide)).

## [Switch providers](@id multi_backend_switch)

- **The endpoint and its key.** Bind `service` to the provider's endpoint, and set the variable it reads its
  key or host from: each provider's start page names both, and its default model. A `Chat` or a `Respond`
  that names no `model` gets that default, and throws `ArgumentError` on a provider that has none.
- **The model and its options.** Model ids, and the options each model takes — reasoning efforts, sampling
  controls, service tiers, caching —, are the provider's. UniLM checks a request against the rules its
  provider's entry lists and throws `ArgumentError` before sending for what they refuse; a request those
  rules let through goes to the provider, whose answer is the check. After a switch, an option the new
  provider refuses by such a rule fails at the first call, before anything is sent. The table below links
  each provider's rules.
- **A conversation.** Its messages move with it: `Chat(service=…, model=…, messages=chat.messages)`
  continues a conversation on another provider. What a provider adds to its own replies — thinking text,
  and the signatures that come with it — goes back only to that provider ([`ProviderContent`](@ref)); the
  next provider gets each turn's text and tool calls.
- **The price.** [`estimated_cost`](@ref) prices a result at its model's row of [`DEFAULT_PRICING`](@ref)
  ([Cost Tracking](@ref cost_guide)), and each provider's entry says which models its rows cover and what it
  prices otherwise.

```@eval
Main.ProviderFacts.on_your_provider(Main.ProviderFacts.llm, :limits)
```

## [Where the providers differ](@id multi_backend_compare)

- **What each serves.** [Capabilities by provider](@ref capabilities_guide) shows the capabilities each
  provider's endpoints declare — chat, tools, embeddings, images, the platform APIs —, and which requests
  UniLM sends to a server without knowing whether it serves them.
- **The API its endpoint speaks.** Some endpoints speak their provider's native API, and others the
  OpenAI wire that many providers serve; the table below says which, and what UniLM's release gate tests
  live before every release. A server with an OpenAI-compatible API is reached with a
  `GenericOpenAIEndpoint`, and any other with an endpoint of your own
  ([Write your own backend](@ref custom_backends_guide)).
- **Its models.** Each entry names the models UniLM holds rules and prices for, and the next section says
  how UniLM keeps up with what the providers list.

| Provider | The API its endpoint speaks | Live tests before every release |
| :--- | :--- | :--- |
| OpenAI | its own: Chat Completions and Responses | against the API |
| Anthropic | its own: Messages | against the API |
| Google Gemini | its own: generateContent and Interactions; `GEMINIOpenAIServiceEndpoint`, Google's OpenAI-compatible endpoint | against the API |
| Ollama | its own, and its OpenAI-compatible routes for `respond`, compaction, FIM and transcription | decision models, embeddings and model management, against a local server; not chat, `respond`, FIM or transcription |
| vLLM, LM Studio & llama.cpp | OpenAI-compatible | — |
| Azure OpenAI | Azure OpenAI's Chat Completions, in the dated `api-version` form | — |
| DeepSeek | OpenAI-compatible, with its beta FIM and prefix completion | — |
| Mistral & OpenAI-compatible services | OpenAI-compatible | — |

## [Keep up with provider models](@id backends_provider_models)

Providers add models, deprecate them and shut them down on their own schedules. Where an endpoint declares
`:models`, [`list_models`](@ref) lists the models it serves your key, or holds, today
([Model Listings](@ref models_api)), and that provider's Models section shows what its listing gives.

UniLM checks a request against its rules for the model the request names, and prices the result at the row
of [`DEFAULT_PRICING`](@ref) of the model the reply names as having served it, else of the model requested:
the row of the id, else of the id without a dated snapshot suffix such as `-2026-09-14` or `-20251101`. A
model with no rule row of its own meets only the rules its provider sets for every model, and the provider's
answer checks the rest; a model with no price row is estimated at `0.0`
([Unpriced models cost \$0](@ref unpriced-zero)). Before you adopt a newly listed model, look for its price
row as the cost code does:

```@example multi_backend
priced(id) = haskey(DEFAULT_PRICING, id) ||
             haskey(DEFAULT_PRICING, replace(id, r"-(\d{4}-\d{2}-\d{2}|\d{8})$" => ""))
priced(model), priced(model * "-2026-09-14"), priced("my-model")
```

UniLM's release gate holds its rule and price tables to three providers' listings before every release:
every listed model of a family UniLM documents needs a price row, and a rule row where the table below says so;
every model UniLM names — as a default, in its live tests or in the Julia code of these pages — must be listed,
and listed before its shutdown date where the listing gives one.

| Provider | A price row for each listed | A rule row for each listed | A listed model of no family |
| :--- | :--- | :--- | :--- |
| OpenAI | GPT-6, GPT-5.6 and GPT-5.4 mini model, dated snapshots included, with a row of OpenAI's price terms | GPT-6 and GPT-5.6 model | left out when of an older generation or not a text model; else logged, not failed |
| Anthropic | Claude model of 4.5 or later, and model with a rule row, but Claude Mythos Preview, which Anthropic's model pricing table does not list | model of the same set, its thinking types and effort levels agreeing with the listing's ([Models](@ref anthropic_models)) | left out when older than 4.5; fails the gate when its id gives no generation |
| Google Gemini | text model of 3.5 or later, and text model with thinking levels, but the `-latest` aliases, which UniLM prices as the models that serve them | model of the same set that thinks: its thinking levels | left out when older than 3.5; logged, not failed, when its id gives no generation |

## [See also](@id multi_backend_see_also)

- [Capabilities by provider](@ref capabilities_guide) — what each provider serves
- [Write your own backend](@ref custom_backends_guide) — a provider UniLM does not ship
- [Choosing a decision provider](@ref system_one_backends) — the same choice for decision models
- [Service Endpoints](@ref endpoints_api) — every built-in endpoint and its constructor
- [`ServiceEndpoint`](@ref), [`GenericOpenAIEndpoint`](@ref) — the endpoint types

## [On your provider](@id multi_backend_providers)

```@eval
Main.ProviderFacts.provider_table(Main.ProviderFacts.llm)
```
