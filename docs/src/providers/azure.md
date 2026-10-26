# [Azure OpenAI](@id azure_start)

```@setup azure
using UniLM
```

Azure OpenAI serves the GPT models from an Azure resource of yours, each through a deployment you
create. `AZUREServiceEndpoint` speaks its Chat Completions API in the dated form — a deployment's URL
and an `api-version` — through `chatrequest!` and `tool_loop!`. The agentic verbs — `respond` and the
`tool_loop` that takes an input or a `Respond` — refuse this endpoint before sending anything, and no
verb of a capability the table below lists as not available sends a request through it. Azure's
newer v1 API, which takes no `api-version`, is reachable through a generic endpoint
([The v1 API](@ref azure_compatible)). No key of an Azure resource was at hand for this page: its
examples are not run, and their code stands without output.

```@eval
Main.ProviderFacts.at_a_glance(:azure)
```

| I want to… | Use | You get |
| :--- | :--- | :--- |
| [point UniLM at my resource and deployments](@ref azure_setup) | the `AZURE_OPENAI_*` variables, `add_azure_deploy_name!` | requests to the deployment of each model |
| [make a first call](@ref azure_first_call) | `chatrequest!(chat)` | an `LLMSuccess`: the reply and its usage, priced by `estimated_cost` |
| [hold a conversation](@ref azure_conversations) | `push!` a message, then `chatrequest!` again | each reply appended to the `Chat` and sent back with the next turn |
| [stream a reply](@ref azure_streaming) | `stream=true`, a `callback` and `stream_options` | the text as it arrives, then the whole message and its usage |
| [let the model call my functions](@ref azure_tools) | `tools` and `tool_loop!` | the calls run and answered until the model replies |
| [get JSON that fits a schema](@ref azure_structured_output) | a JSON Schema `response_format` | a reply constrained to the schema |
| [control reasoning](@ref azure_reasoning) | `reasoning_effort`, `max_completion_tokens` | that effort, and a cap on the reasoning and the reply together |
| [reuse a cached prompt](@ref azure_caching) | nothing, or `prompt_cache_key` | cache hits in `usage.cached_tokens`, billed at the cached rate |
| [know what a call cost](@ref azure_cost) | `estimated_cost`, `cumulative_cost` | USD estimates at the model's list rates |
| [use the v1 API](@ref azure_compatible) | `GenericOpenAIEndpoint` | requests without an `api-version` |
| [handle a failure](@ref azure_failures) | the result's type | the status or the exception, and what to do |

## [Setup](@id azure_setup)

Set the resource, its key, the API version and a deployment for each model you call:

```bash
export AZURE_OPENAI_BASE_URL="https://your-resource.openai.azure.com"
export AZURE_OPENAI_API_KEY="..."
export AZURE_OPENAI_API_VERSION="2024-10-21"
export AZURE_OPENAI_DEPLOY_NAME_GPT_5_2="your-gpt52-deployment"
```

UniLM reads these variables each time a request is built, so they may be set after the package
loads. A `Chat` goes to
`$AZURE_OPENAI_BASE_URL/openai/deployments/<deployment>/chat/completions?api-version=$AZURE_OPENAI_API_VERSION`
with the key in the `api-key` header, the form of Azure's dated API versions; `2024-10-21` is the
dated GA release Azure's REST reference documents
([REST API reference](https://learn.microsoft.com/en-us/azure/foundry/openai/reference)). A field
Azure added later needs a later version: `reasoning_effort` came in `2024-12-01-preview`
([API versions](https://learn.microsoft.com/en-us/azure/foundry/openai/api-version-lifecycle)), and
audio replies in `2025-01-01-preview`
([audio generation](https://learn.microsoft.com/en-us/azure/foundry/openai/audio-completions-quickstart)). A call
without one of the first three variables returns a call-error result whose `cause` is the `KeyError`
naming it. UniLM sends the key only: Azure's other way in, a Microsoft Entra ID token in an
`Authorization: Bearer` header, is not this endpoint's.

Azure routes a request to a *deployment*, and UniLM finds the deployment of the model a `Chat` names.
A deployment registered with [`add_azure_deploy_name!`](@ref) wins; otherwise UniLM reads
`AZURE_OPENAI_DEPLOY_NAME_<MODEL>` when the request is built, where `<MODEL>` is the model id
upper-cased with every character other than `A-Z` and `0-9` made `_` (`gpt-5.2` →
`AZURE_OPENAI_DEPLOY_NAME_GPT_5_2`, `gpt-6-luna` → `AZURE_OPENAI_DEPLOY_NAME_GPT_6_LUNA`). A model
with neither makes the call return an `LLMCallError` whose `cause` is an `ArgumentError` naming the
variable to set. The deployment's name is sent as one path segment, percent-encoded. A registration
holds for the session, and a second one for the same model replaces it:

```julia
add_azure_deploy_name!("gpt-5.2", "my-gpt52-deployment")
chat = Chat(service=AZUREServiceEndpoint, model="gpt-5.2")   # sent to my-gpt52-deployment
```

A `Chat` takes the endpoint as the type, `service=AZUREServiceEndpoint`, and `AZUREServiceEndpoint()`
returns the type too; with no `model`, it takes the default model named at the top of this page. The
model id goes in the request body as well, and it is the id UniLM prices a reply at when Azure's reply
names none ([Cost and usage](@ref azure_cost)).

## [First call](@id azure_first_call)

A `Chat` starts with its system message; `chatrequest!` sends it and returns a typed result, and
`text` reads the reply — or throws, on a failure:

```julia
chat = Chat(service=AZUREServiceEndpoint)
push!(chat, Message(Val(:system), "You are a concise assistant."))
push!(chat, Message(Val(:user), "In one sentence: what is a prime number?"))
result = chatrequest!(chat)
text(result)
result.model, result.message.finish_reason
u = result.usage
(; u.prompt_tokens, u.completion_tokens, u.reasoning_tokens, u.cached_tokens, cost = estimated_cost(result))
```

The reply is read as a Chat Completions reply: `result.model` is the model Azure reports, and
`finish_reason` is Azure's, `content_filter` "to signify that some of the completion was filtered"
([content filtering](https://learn.microsoft.com/en-us/azure/foundry-classic/foundry-models/concepts/content-filter)).
`prompt_tokens` counts the input, of which `cached_tokens` came from the prompt cache
([Prompt caching](@ref azure_caching)), and `completion_tokens` the output, of which
`reasoning_tokens` is the reasoning ([Reasoning](@ref azure_reasoning)). The result types are on
[Results](@ref results_guide).

## [Conversations](@id azure_conversations)

A `Chat` keeps the conversation: each reply is appended to it, and the next `chatrequest!` sends the
whole conversation again.

```julia
push!(chat, Message(Val(:user), "And a composite number?"))
text(chatrequest!(chat))
[m.role for m in chat.messages], cumulative_cost(chat)
```

A system message pushed after the conversation started is sent in place. Azure's reasoning models take
system messages; "You should not use both a developer message and a system message in the same API
request" ([reasoning models](https://learn.microsoft.com/en-us/azure/foundry/openai/how-to/reasoning)),
and UniLM sends no developer message. What `push!` refuses is the same on every provider
([Chat Completions](@ref chat_guide)).

A reply that holds audio (`audio_output`) goes back to Azure by the audio's id, `"audio": {"id": …}`
beside the turn's transcript, as Azure's quickstart adds "a history message referencing the previous
turn's audio by ID"
([audio generation](https://learn.microsoft.com/en-us/azure/foundry/openai/audio-completions-quickstart)),
until a minute before the reply's `expires_at`, and from then on as the transcript alone
([Audio replies](@ref chat_audio_output)).

## [Streaming](@id azure_streaming)

With `stream=true`, `chatrequest!` returns a `Task` at once, and `fetch` gives the same typed result as
an unstreamed call; the `callback` receives the text as it arrives, then the assembled `Message`. A
`Chat` on this endpoint sends no `stream_options` unless it sets them, so a streamed turn reports no
usage, and adds `0.0` to `cumulative_cost`, unless the `Chat` asks for it:

```julia
chat = Chat(service=AZUREServiceEndpoint, stream=true, stream_options=Dict("include_usage" => true))
push!(chat, Message(Val(:system), "You are a concise assistant."))
push!(chat, Message(Val(:user), "List the planets of the solar system in order, one per line."))
task = chatrequest!(chat; callback=(chunk, close) -> chunk isa String && print(chunk))
result = fetch(task)
result.usage
```

Azure added `stream_options` and `include_usage` in the API version `2024-09-01-preview`
([API versions](https://learn.microsoft.com/en-us/azure/foundry/openai/api-version-lifecycle)). The
stream is the Chat Completions one: a chunk whose `choices` list is empty, such as the usage chunk,
whose "`choices` field will always be an empty array"
([chat REST reference](https://learn.microsoft.com/en-us/rest/api/microsoft-foundry/azureopenai/chat?view=rest-microsoft-foundry-2024-10-21&preserve-view=true)),
gives its usage and nothing else; the turn is final at `data: [DONE]`, or, without
one, when the HTTP body ends after the finish reason; `on_tool_call(tc)` runs once per call whose
arguments decode, as the next call begins or at the end of the stream; and setting `close[] = true`
stops it ([Stopping a Stream Early](@ref streaming_stop)). The concepts every provider shares are on
[Streaming](@ref streaming_guide).

## [Tools](@id azure_tools)

Each `Tool` in `tools` goes out as a function tool, and `tool_choice`, `parallel_tool_calls` and a
tool's `strict` go as they are set; a `Chat` with tools sends `parallel_tool_calls: false` unless it
sets another value, as Azure asks with strict tools: "Structured outputs are not supported with
parallel function calls"
([structured outputs](https://learn.microsoft.com/en-us/azure/foundry/openai/how-to/structured-outputs)).
`tool_loop!` runs the calls and sends their results back until the model answers in text:

```julia
weather = UniLM.CallableTool(
    Tool(func=FunctionSignature(name="get_weather", description="Current weather in a city",
        parameters=Dict("type" => "object", "required" => ["city"],
                        "properties" => Dict("city" => Dict("type" => "string"))))),
    (name, args) -> "18°C and sunny in $(args["city"])")
chat = Chat(service=AZUREServiceEndpoint, tools=[weather])
push!(chat, Message(Val(:system), "You are concise."))
push!(chat, Message(Val(:user), "What is the weather in Lyon? Use the tool, then answer in one sentence."))
loop = tool_loop!(chat; tools=[weather])
text(loop.response)
```

On the GPT-5.6 models, Azure's Chat Completions API refuses function tools at any effort but `none`:
"The `gpt-5.6` models support the Chat Completions API and tools, but can't combine reasoning with
tools on Chat Completions", and "The request fails even when you don't send `reasoning_effort`,
because these models default to `medium`"; "If you must stay on Chat Completions, set
`reasoning_effort` to `none` on every request that sends `tools`"
([reasoning models](https://learn.microsoft.com/en-us/azure/foundry/openai/how-to/reasoning)). UniLM
does not check this on Azure: the refusal is Azure's HTTP 400. The loop every provider shares is on
[Tool Calling](@ref tools_guide).

## [Structured output](@id azure_structured_output)

A JSON Schema `response_format` goes out as `{"type": "json_schema", "json_schema": {…}}`, with
`"strict": true` when it sets `strict`, and `UniLM.json_object()` as `{"type": "json_object"}`. Azure
constrains the reply: "For Chat Completions, define the schema in `response_format`", and "API version
`2024-08-01-preview` is the first version that supports structured outputs"
([structured outputs](https://learn.microsoft.com/en-us/azure/foundry/openai/how-to/structured-outputs)),
so `AZURE_OPENAI_API_VERSION` must name one that does:

```julia
using JSON
schema = Dict("type" => "object", "additionalProperties" => false, "required" => ["name", "year"],
              "properties" => Dict("name" => Dict("type" => "string"), "year" => Dict("type" => "integer")))
chat = Chat(service=AZUREServiceEndpoint,
            response_format=UniLM.json_schema("person", "A person", schema; strict=true))
push!(chat, Message(Val(:system), "You extract facts."))
push!(chat, Message(Val(:user), "Extract the person: Ada Lovelace, born 1815."))
JSON.parse(text(chatrequest!(chat)))
```

The formats every provider shares are on [Structured Output](@ref structured_guide).

## [Reasoning](@id azure_reasoning)

A `Chat`'s `reasoning_effort` goes out as it is set; Azure's values "vary by model and include `none`,
`minimal`, `low`, `medium`, `high`, `xhigh`, and `max`. Defaults vary by model as well", and "`max`
works only with GPT-6 or GPT-5.6 models and the Responses API"
([reasoning models](https://learn.microsoft.com/en-us/azure/foundry/openai/how-to/reasoning)), which
this endpoint does not speak. The field needs an `api-version` from `2024-12-01-preview` on
([Setup](@ref azure_setup)). Cap a
reasoning model's output with `max_completion_tokens`: "Reasoning models will only work with the
`max_completion_tokens` parameter when using the Chat Completions API", and the cap covers "reasoning
tokens, visible output tokens, and formatting tokens". On this endpoint `max_tokens` keeps its name, so
a request to a reasoning model sets `max_completion_tokens` instead. "Reasoning tokens are billed as
output tokens": they count in `completion_tokens`, `reasoning_tokens` among them, and Chat Completions
returns no reasoning text, so [`reasoning_text`](@ref) gives `nothing`.

```julia
chat = Chat(service=AZUREServiceEndpoint, reasoning_effort="low", max_completion_tokens=2000)
```

## [Attachments](@id azure_media)

A user message's attachments go out as content parts: an image as an `image_url` part, with its
`detail` when one is set, a sound clip as an `input_audio` part, and a document as a `file` part
holding its name and a data URL; the deployment's model decides what it reads, and UniLM checks none of
it before sending. The attachments every provider shares are on [Attachments](@ref attachments_guide).

## [Limits and refusals](@id azure_limits)

A deployment's quota sets its tokens and requests per minute
([quotas and limits](https://learn.microsoft.com/en-us/azure/foundry/openai/quotas-limits)), and a
request past them meets HTTP 429 ([When something fails](@ref azure_failures)). UniLM checks no limit
of Azure's, and none of the per-model rules it applies on its native endpoint for the GPT models:
every field a `Chat` sets goes out as it is — `max_tokens` keeps its name, `user` is sent,
`service_tier` and `reasoning_effort` are not checked — and Azure's answer is the check. It refuses
nothing of Azure's own before sending, only the checks every `Chat` gets, such as a `temperature` from
0 to 2 or an `n` of 1; the verbs of the capabilities the table at the top of this page lists as not
available refuse the endpoint.

## [Prompt caching](@id azure_caching)

Azure caches the prompts of the models that support it on its own: "cache reads are billed at a
discount on input token pricing for Standard deployment types and up to 100% discount on input
tokens for Provisioned deployment types"
([prompt caching](https://learn.microsoft.com/en-us/azure/foundry/openai/how-to/prompt-caching)).
The reply's `prompt_tokens_details.cached_tokens` becomes `usage.cached_tokens`, billed at the cached
rate ([Cost and usage](@ref azure_cost)). The cache fields go out as they are set, and Azure answers
them:

- `prompt_cache_key`: on the GPT-5.6 models and later, "reuse the same key for requests that share
  long, common prompt prefixes".
- `prompt_cache_options`: its `ttl` "`30m` value is the default and the only supported value", and
  "Models before the GPT-5.6 family don't support `prompt_cache_options` or `prompt_cache_breakpoint`.
  Requests that include these parameters return a `400` error".
- `prompt_cache_retention`: `in_memory` or `24h` on the models before GPT-5.6, where it "selects a
  maximum-retention policy".

## [Cost and usage](@id azure_cost)

[`estimated_cost`](@ref) prices an Azure result at the model's row of [`DEFAULT_PRICING`](@ref),
looked up by the model Azure reports, without its dated suffix, else the model the `Chat` names; the
rows hold OpenAI's list prices, and Azure's own are not in UniLM. It bills the standard rates:
input, cache reads at the cached rate, output, and a reported cache write at the input rate. No
service-tier, long-context or cache-write multiple applies to an Azure result, though Azure bills
GPT-6 and GPT-5.6 models in "separate pricing categories for short-context and long-context requests"
([models](https://learn.microsoft.com/en-us/azure/foundry/foundry-models/concepts/models-sold-directly-by-azure)).
A model without a row is estimated at `0.0`, with one warning per model id, and a result whose usage
counts audio tokens at `0.0`, with one warning per model and endpoint, since a row holds no audio
rates. [`cumulative_cost`](@ref) adds up a `Chat`'s turns; how a call is priced on every provider is on
[Cost Tracking](@ref cost_guide).

## [When something fails](@id azure_failures)

`chatrequest!` returns what Azure or the network did as a typed result, and does not throw it
([Results](@ref results_guide)). A failure holds Azure's HTTP `status` and, as `response`, the body
that came with it; a call error holds the text of an exception as `error` and the exception as
`cause`. UniLM sends a request again after a transient status — 408, 429, 500, 502 to 504, 520 to 524
and 529 — or a failed connection, as often as the call's [`RequestConfig`](@ref) allows, and then
returns the last answer ([Timeouts & Retries](@ref timeouts_guide)); a streamed request is sent again
only while none of its callbacks has run.

| What you see | Why | What to do |
| :--- | :--- | :--- |
| an `LLMCallError` whose `cause` is a `KeyError` | `AZURE_OPENAI_BASE_URL`, `AZURE_OPENAI_API_KEY` or `AZURE_OPENAI_API_VERSION` is not set | set it ([Setup](@ref azure_setup)) |
| an `LLMCallError` whose `cause` is an `ArgumentError` naming `AZURE_OPENAI_DEPLOY_NAME_<MODEL>` | the model has no deployment | set the variable, or `add_azure_deploy_name!` |
| a failure with `status` 401 | Azure did not accept the key | the key |
| a failure with `status` 400 whose error `code` is `content_filter` | "Prompts that are classified at a filtered category and severity level return an HTTP 400 error" ([content filtering](https://learn.microsoft.com/en-us/azure/foundry-classic/foundry-models/concepts/content-filter)) | another prompt |
| a failure with `status` 400, 404 or another status not listed above | Azure refused the request — a deployment, an `api-version`, a field or a value it does not take ([Limits and refusals](@ref azure_limits)) — and UniLM does not send it again | what Azure's message says |
| a reply whose `finish_reason` is `content_filter` | the filter withheld output; "In rare cases with longer responses, a partial result can be returned" | another prompt |
| a failure with `status` 429 | the deployment's quota was reached, which Azure can answer "even when token usage metrics appear below your quota" ([quotas and limits](https://learn.microsoft.com/en-us/azure/foundry/openai/quotas-limits)); sent again until the attempts or the deadline run out | fewer requests, or more quota |
| a failure with `status` 500, 503 or another transient status | a fault or an overload, sent again until the attempts or the deadline run out | the request again, later |
| an `LLMFailure` with `status` 200 from a stream | the stream ended before a finish reason; `response` holds what arrived, and nothing is appended | the turn again |
| a call error with `status` `nothing` and a `UniLMTimeout` as `cause` | a bound of the call's `RequestConfig` ran out | a longer bound, or a stream for a long reply ([Timeouts & Retries](@ref timeouts_guide)) |
| a call error whose `cause` is a `UniLMCancelled` | a stop: the stream's `close[] = true` or a cancelled `CancelToken` | nothing, if the stop was yours |
| a call error whose `cause` is the exception your `callback` or `on_tool_call` threw | the exception ended the stream; it is not sent again, and nothing is appended; a thrown value that is no `Exception` leaves `cause` `nothing`, its text in `error`; an `InterruptException`, or a recorded call's `ReplayMissError` or `RecordingWriteError`, is thrown at `fetch` instead | the callback |
| `LLMResultError` from `text` | the result is a failure, which holds no reply | check `issuccess(result)` first |

## [Models](@id azure_models)

A `Chat` names a model id, and the model's deployment serves it ([Setup](@ref azure_setup)). Azure's
catalog lists the GPT-6.1, GPT-6, GPT-5.6, GPT-5.5, GPT-5.4 and GPT-5.2 series among others
([models](https://learn.microsoft.com/en-us/azure/foundry/foundry-models/concepts/models-sold-directly-by-azure)).

```julia
chat = Chat(service=AZUREServiceEndpoint, model="gpt-6-luna")   # AZURE_OPENAI_DEPLOY_NAME_GPT_6_LUNA
```

## [The v1 API](@id azure_compatible)

Azure's v1 API "removes the need for dated `api-version` parameters"; its base URL is the resource
with `/openai/v1` appended, and a request names the deployment as its `model`
([v1 API](https://learn.microsoft.com/en-us/azure/foundry/openai/api-version-lifecycle)).
`AZUREServiceEndpoint` does not speak it. A `GenericOpenAIEndpoint` whose base URL is the resource's
`/openai` path sends `POST …/openai/v1/chat/completions`, with the key as `Authorization: Bearer`:

```julia
v1 = GenericOpenAIEndpoint("https://your-resource.openai.azure.com/openai", "", "AZURE_OPENAI_API_KEY")
chat = Chat(service=v1, model="your-gpt52-deployment")
```

Azure's REST example of the v1 API sends a key in an `api-key` header and a Microsoft Entra ID token
as `Authorization: Bearer`; a key sent as `Authorization: Bearer`, as here, is untested, and so is this
route. Through it, a request has the generic endpoint's rules: no default model and no deployment
lookup, the `Chat`'s `model` sent as the deployment's name and the reply priced by the model Azure
reports, and the capabilities of a generic endpoint, sent unchecked
([OpenAI-compatible services](@ref openai_compatible_start)).

Checked against Azure OpenAI's documentation of 25 October 2026 — the dated API version `2024-10-21`
and the v1 API — with UniLM's development branch after 0.26.0; nothing on this page was run.

See also: [`AZUREServiceEndpoint`](@ref), [`add_azure_deploy_name!`](@ref), [`Chat`](@ref),
[`chatrequest!`](@ref) and [`LLMSuccess`](@ref) in the API reference.
