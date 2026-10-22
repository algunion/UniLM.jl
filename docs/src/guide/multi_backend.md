# [Multi-Backend Support](@id backend_guide)

UniLM.jl is built around **neutral verbs**: the same `Chat` + [`chatrequest!`](@ref) — with
tools, streaming, and [cost accounting](@ref cost_guide) — run unchanged across every backend;
you only change the `service`. Cost accounting needs a price row for the model: the built-in
table covers the OpenAI, Anthropic, Gemini, DeepSeek and TypeSafe models it lists, a local
Ollama model costs `0.0` (a cloud model reached through Ollama, tagged `cloud` or `…-cloud`,
is priced by the requested name), and any other model (Mistral, custom) is estimated at `0.0` with a
one-time warning unless you supply `pricing=`. The agentic [`respond`](@ref) verb is neutral the same way
across OpenAI (Responses) and Gemini (Interactions); see [Agentic Workflows](@ref agentic_guide).
Native OpenAI, Anthropic, and Gemini are first-class backends with their own wire formats (each
exercised by live integration tests that the release gate runs against the real APIs before every
release), not OpenAI-compatible shims.

Select the backend with `service`. Model names and supported generation controls
remain provider-specific. A request option the chosen provider (or model) cannot
express throws `ArgumentError` before any request is sent — native Anthropic and
Gemini fail closed on unmapped `Chat` fields rather than dropping them.

## Available Backends

| Backend          | Type                      | Env Variables                                                               |
| :--------------- | :------------------------ | :-------------------------------------------------------------------------- |
| OpenAI (default) | `OPENAIServiceEndpoint`   | `OPENAI_API_KEY`                                                            |
| Azure OpenAI     | `AZUREServiceEndpoint`    | `AZURE_OPENAI_BASE_URL`, `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_API_VERSION` |
| Google Gemini    | `GEMINIServiceEndpoint`   | `GEMINI_API_KEY`                                                            |
| Anthropic        | `ANTHROPICServiceEndpoint`| `ANTHROPIC_API_KEY`                                                         |
| DeepSeek         | `DeepSeekEndpoint`        | `DEEPSEEK_API_KEY`                                                          |
| Mistral          | `MistralEndpoint`         | `MISTRAL_API_KEY`                                                           |
| Ollama (local)   | `OllamaEndpoint`          | (none)                                                                      |
| Generic          | `GenericOpenAIEndpoint`   | (passed to constructor)                                                     |

## OpenAI (Default)

```@example backends
using UniLM
using JSON

# OpenAI is the default — no need to specify service
chat = Chat(model="gpt-5.2")
println("Service: ", chat.service)
println("Model: ", chat.model)
```

## Azure OpenAI

```julia
# Set environment variables
ENV["AZURE_OPENAI_BASE_URL"] = "https://your-resource.openai.azure.com"
ENV["AZURE_OPENAI_API_KEY"] = "your-key"
ENV["AZURE_OPENAI_API_VERSION"] = "2024-10-21"   # the latest dated GA api-version
ENV["AZURE_OPENAI_DEPLOY_NAME_GPT_5_2"] = "your-gpt52-deployment"

# Use Azure (default model: gpt-5.2)
chat = Chat(service=AZUREServiceEndpoint, model="gpt-5.2")
push!(chat, Message(Val(:system), "Hello from Azure!"))
push!(chat, Message(Val(:user), "Hi!"))
result = chatrequest!(chat)
```

Azure routes each request to a *deployment*. For every model the deployment is read
**at call time** from `AZURE_OPENAI_DEPLOY_NAME_<MODEL>`, where `<MODEL>` is the model
id upper-cased with every character other than `A-Z` and `0-9` mapped to `_`
(`gpt-5.2` → `AZURE_OPENAI_DEPLOY_NAME_GPT_5_2`, `gpt-6-luna` →
`AZURE_OPENAI_DEPLOY_NAME_GPT_6_LUNA`), unless a deployment was registered with
[`add_azure_deploy_name!`](@ref), which wins. A model with neither makes the call
return an `LLMCallError` whose `cause` is an `ArgumentError` naming the variable to
set. `AZUREServiceEndpoint` declares `:chat`, `:tools`, `:streaming` and
`:json_output`.

!!! note "Azure OpenAI v1 API"
    `AZUREServiceEndpoint` speaks the dated `api-version` form of the Azure OpenAI
    API. Azure's newer v1 API, which needs no `api-version`, has the OpenAI-compatible
    path shape, so it should be reachable through a generic endpoint — **untested**:
    `GenericOpenAIEndpoint("https://your-resource.openai.azure.com/openai", key)`
    sends `POST …/openai/v1/chat/completions` with an `Authorization: Bearer <key>`
    header.

### Custom Deployment Names

If your Azure deployment has a custom name:

```@example backends
UniLM.add_azure_deploy_name!("my-custom-model", "my-deployment-name")
println("Registered deployments: ", collect(keys(UniLM._MODEL_ENDPOINTS_AZURE_OPENAI)))
delete!(UniLM._MODEL_ENDPOINTS_AZURE_OPENAI, "my-custom-model")  # cleanup
nothing # hide
```

## Google Gemini

!!! warning "Breaking change since v0.10.3"
    `GEMINIServiceEndpoint` now targets Google's **native `generateContent` API**
    (auth header `x-goog-api-key`, model in the URL, default model
    `gemini-3.8-flash`). The old **OpenAI-compatible** Gemini path is renamed
    [`GEMINIOpenAIServiceEndpoint`](@ref). Migrate code that relied on the
    OpenAI-compatible behavior — including `Embeddings(...; service=GEMINIServiceEndpoint)`,
    which the native endpoint does not support — to `GEMINIOpenAIServiceEndpoint`.

Native Gemini chat (real call, guarded so a failure never breaks the build):

```@example backends
gemini_chat = Chat(service=GEMINIServiceEndpoint, model="gemini-3.8-flash",
                   reasoning_effort="low", max_tokens=1024)
push!(gemini_chat, Message(Val(:system), "You are a helpful assistant."))
push!(gemini_chat, Message(Val(:user), "Say hello in one short sentence."))
result = chatrequest!(gemini_chat)
if result isa LLMSuccess
    println(result.message.content)
else
    println("Request failed — see result for details")
end
```

To keep using the OpenAI-compatible endpoint, switch the service type:

```julia
chat = Chat(service=GEMINIOpenAIServiceEndpoint, model="gemini-3.8-flash",
            reasoning_effort="low")
```

Google calls `generateContent` legacy — still fully supported — and launches new models
and features on the Interactions API, which [`respond`](@ref) speaks with the same
`service=GEMINIServiceEndpoint` (see [Agentic Workflows](@ref agentic_guide) and Google's
[Interactions overview](https://ai.google.dev/gemini-api/docs/interactions-overview)).

### Images, audio and documents

Native Gemini sends a user message's [`ImageAttachment`](@ref), [`AudioAttachment`](@ref) and
[`DocumentAttachment`](@ref) as `inlineData` parts — the bytes in base64, under the MIME type read
from them — ahead of the message's text, as Google advises for an image with a prompt:

```julia
chat = Chat(service=GEMINIServiceEndpoint, model="gemini-3.8-flash", reasoning_effort="low",
            messages=[Message(Val(:user), "What colour is the circle, and what is the note's secret word?",
                              ImageAttachment("circle.png"), DocumentAttachment("note.pdf"))])
result = chatrequest!(chat)
```

An image's `detail` is its per-part media resolution
([media resolution](https://ai.google.dev/gemini-api/docs/generate-content/media-resolution)): `"low"`
and `"high"` are Gemini's `MEDIA_RESOLUTION_LOW` and `MEDIA_RESOLUTION_HIGH` (on `gemini-3.8-flash`,
`"low"` took a 1,089-token image to 256 tokens), `"auto"` leaves the model's default, and
`"original"`, an OpenAI level, throws `ArgumentError`. Google documents a per-part resolution for
Gemini 4 and Gemini 3 models, so any level on the earlier Gemini 2.5 and Gemma 4 models throws too
(inferred from that list; not probed on those models). Inline media has no file name on this API,
so a document's `filename` is not sent. On [`respond`](@ref) the same attachments go in an
`InputMessage`'s content (see [Agentic Workflows](@ref agentic_guide)).

### Current model controls

`reasoning_effort` maps to `thinkingConfig.thinkingLevel` on native Chat (and a
`Reasoning` effort to `generation_config.thinking_level` on `respond`), checked against
the model's levels in Google's [thinking table](https://ai.google.dev/gemini-api/docs/thinking):
Gemini 4 Argon and Gemini 3.8 Flash take `"low"`, `"medium"` or `"high"`, so `"minimal"`
throws `ArgumentError` before any request (the API answers it with HTTP 400), as it does for
`gemini-pro-latest` and `gemini-flash-latest`, which take the levels of the models behind
them. A row matches its exact model id; an id the table does not list is left to the server.
The completion cap includes thinking tokens, so a very small cap can produce an empty answer.

Gemini ignores `temperature` and `top_p` on Gemini 3.6 Flash, 3.5 Flash-Lite and every later
model ([what changed in Gemini 3.6](https://ai.google.dev/gemini-api/docs/whats-new-gemini-3.6)),
so native Chat, `respond` (Interactions) and `GEMINIOpenAIServiceEndpoint` throw `ArgumentError`
for them on every model except the earlier ones that honour them, such as `gemini-2.5-flash`,
`gemini-3-flash-preview` and `gemini-3.1-pro-preview`. From the same models on, a request may not
end with a model turn: a prefilled answer (a `Chat` whose last message is the assistant's, sent with
`history=false`) throws `ArgumentError` on native Chat, since the API answers it with HTTP 400, and
`GEMINIOpenAIServiceEndpoint` returns that 400 as an `LLMFailure`. On `respond`, Gemini 4 Argon
takes a `max_output_tokens` above 65,536 only with `background=true`
([long decode](https://ai.google.dev/gemini-api/docs/long-decode)).
`prompt_cache_options` throws on native Chat and `respond`: Gemini caches repeated prompt
prefixes [implicitly](https://ai.google.dev/gemini-api/docs/caching), and UniLM does not wrap the
explicit caching (cached contents) that `generateContent` also offers on Gemini 3 and 2.5 models;
`GEMINIOpenAIServiceEndpoint` answers the field with HTTP 400.

Google routes the deprecated `gemini-3.7-flash` to `gemini-3.8-flash` and `gemini-3.5-flash` to
`gemini-3.6-flash`. Such a request takes the rules of the model that serves it, and the reply names
that model — `result.model` from `modelVersion` on Chat, `response.model` on `respond` — which is
the model [`estimated_cost`](@ref) prices: an alias or a routed id is billed at its target's rates.
Native Gemini tool declarations use `parametersJsonSchema`, preserving standard
JSON Schema constraints such as `additionalProperties`.

`response_format` is mapped natively: JSON object and JSON Schema output become
`generationConfig.responseFormat` (see [Structured Output](@ref structured_guide)).
`safety_identifier` is mapped natively, on Chat and on `respond`, to the request label
`safety_identifier`; label values allow at most 63 characters of lowercase letters
(international characters allowed), numeric characters, underscores, and dashes, so a value
such as an email address or a 64-character SHA-256 hex digest throws `ArgumentError` at encode
time. `seed` is `generationConfig.seed` (and `generation_config.seed` on `respond`), an int32,
so a seed outside -2,147,483,648 to 2,147,483,647 throws `ArgumentError` before any request; `store`
is the request's `store`, which sets whether Google logs it over the project's setting.
Native Chat returns one candidate; options `generateContent` has no equivalent for, such as
`metadata`, `presence_penalty` and `stream_options`, fail explicitly.
Gemini determines parallel tool use; the inherited `parallel_tool_calls` setting
does not constrain native Gemini.

`service_tier` is Gemini's ([optimization](https://ai.google.dev/gemini-api/docs/optimization)):
`"standard"`, `"flex"` (half the standard rates, answers in minutes) or `"priority"` (1.8 times
the rates) on native Chat, sent as `serviceTier`, and on `respond` also `"deferred"`, sent as
`service_tier`; any other value throws `ArgumentError`. `GEMINIOpenAIServiceEndpoint` takes
`"standard"`, `"flex"` and `"priority"`, the values Google's
[compatibility page](https://ai.google.dev/gemini-api/docs/openai#flex-and-priority-inference) names,
and throws for any other. The tier that billed the call is `usage.service_tier` — the reply's
`x-gemini-service-tier` header, which Google names as the sign of a downgrade (and the only place
Gemini 4 Argon names the tier), else its `usageMetadata.serviceTier`, and an interaction's
`service_tier` — so a Priority request Google downgrades reports and is priced as standard. A
`GEMINIOpenAIServiceEndpoint` reply is read header first too, but replies to flex and priority
requests there named their tier in neither the head nor the body (observed 2026-10-22), so a
Chat result there records the tier the request asked for: `"flex"` is priced at the Flex rates
and `"priority"` at the Priority rates — an upper bound, since Google bills a downgraded
Priority request at Standard and this endpoint does not report the downgrade. [`estimated_cost`](@ref)
bills input and output at half the standard rates on Flex and at 1.8 times on Priority, as the
[pricing page](https://ai.google.dev/gemini-api/docs/pricing) lists them, and a cache hit at the
same multiple except where a model's table lists another cache-hit price: on Flex the standard
one for 3.1 Pro Preview, 3 Flash Preview and the 2.5 models, and \$0.02 (Flex) and \$0.05
(Priority) per 1M tokens for 3.5 Flash-Lite, whose standard rate is \$0.03; `"deferred"` has no
price there, so it is estimated at the standard rates with a warning. A prompt over 200K tokens
on 3.1 Pro Preview (and its `-customtools` endpoint) or 2.5 Pro bills at the page's long-context
rates, the whole request: twice the input and cache-hit rates and 1.5 times the output rate, at
the tier's multiple. Audio input bills at the audio rate the page states for the model, and is
unpriced where it states none (see [Cost Tracking](@ref cost_estimate)). A Flex request can wait
in a queue for minutes — Google advises client timeouts of 10 minutes or more
([Flex inference](https://ai.google.dev/gemini-api/docs/flex-inference)) — and a stream's
wait for its headers counts against `stream_idle_timeout` (120 s by default), so give a Flex
call a longer bound, such as `config=RequestConfig(request_timeout=900, stream_idle_timeout=900)`.

Gemini 4 Argon writes past one request's output cap — 32,768 tokens on a non-streamed call and
65,536 on a stream, Google documents — in hops ([long decode](https://ai.google.dev/gemini-api/docs/generate-content/long-decode)):
a reply that ends with `CONTINUATION` is continued inside [`chatrequest!`](@ref) — the same request
again with the latest `continuationToken`, in as many requests as the output budget
(`max_completion_tokens`, else `max_tokens`) needs at that cap, plus one, and at least 8 — and the
result is one assistant turn, its usage the hops' summed (Google bills every hop; each hop's prompt
includes the output so far, most of it as cache hits) and priced at the tier the last hop
reported. Each request has the whole `RequestConfig` budget; a retry sends its request again from
the start, since a stream that drops after a checkpoint token is not resumed from it. A turn still
unfinished after the last of those requests, or whose hop continues without having produced an
output token (thinking included), returns with `finish_reason == "continuation"`; a hop that fails
ends the call with nothing committed, as does a stop or cancel before the last hop's finish reason
(a `CONTINUATION` is not the turn's completion marker), and the hops already answered — one whose
`CONTINUATION` chunk a stop or a throwing callback landed on included — are added to
`cumulative_cost`, with the usage the hop in flight had reported. A streamed turn streams every
hop into the same callback. Continuation is
exercised against a local mock; a live CONTINUATION reply was not observed.
`GEMINIOpenAIServiceEndpoint` has no long decode.
A Gemini 4 call (a `gemini-4-…` model) with no `RequestConfig` of yours in any channel waits up to
20 minutes per attempt, between stream bytes and in all, since Google ends such a call after 10
minutes, a stream after 20, and advises a 20-minute read timeout (see
[Timeouts & Retries](@ref timeouts_guide)).
See Google's [Gemini 4 Argon guide](https://ai.google.dev/gemini-api/docs/latest-model) for the
current model.

Native Gemini reports `finishReason` truthfully on both the streamed and the
non-streamed path: `STOP` is `"stop"` (`"tool_calls"` when the turn carries function
calls), `MAX_TOKENS` is `"length"`, the safety filters — `SAFETY`, `RECITATION`,
`BLOCKLIST`, `PROHIBITED_CONTENT`, `SPII` and the image filters `IMAGE_SAFETY`,
`IMAGE_PROHIBITED_CONTENT` and `IMAGE_RECITATION` — are `"content_filter"`, and any
other value passes through lowercased (for example `"malformed_function_call"` or
`"no_image"`). A candidate without a `finishReason` reports `nothing`,
and function calls under a reason other than `STOP` keep that reason, so a tool loop
does not run them. A filtered turn's `refusal_message` is the reply's `finishMessage`,
Google's account of the block, when it carries one. A model turn with neither text nor
function calls (a refusal, or a turn spent entirely on thinking) is left out of the next
request. Usage counts a tool-use prompt (`toolUsePromptTokenCount`) as prompt tokens, billed
at the input rate.

OpenAI defaults to `gpt-6-sol` for `Chat` and `Respond`. OpenAI's tool path is the
Responses API: GPT-5.6, GPT-6 Sol, and GPT-6 Luna take Chat Completions tools only with an
explicit `reasoning_effort="none"`, and GPT-6 Astra and GPT-6.1 Sol take tools only through
`Respond`. GPT-6 Sol and Luna also reject `temperature`, `top_p`, and log probabilities
unless reasoning effort is `"none"` (an omitted effort is the provider default,
`"medium"`). GPT-6 Sol, GPT-6 Luna and the GPT-5.6 models refuse `"minimal"` reasoning effort;
use `"low"` instead. GPT-6 Astra and GPT-6.1 Sol reason at `"low"` or more, so they reject
sampling controls and log probabilities at every effort. Chat Completions takes effort up to
`"xhigh"` on all of them — it answered `"max"` with HTTP 400 (2026-10-22), though its reference
lists it — while Responses takes `"max"`. These rules hold on the native OpenAI endpoint for
`gpt-6-sol`, `gpt-6.1-sol`, `gpt-6-astra`, `gpt-6-luna`, `gpt-5.6` and
`gpt-5.6-sol`/`-terra`/`-luna`, by exact id or a dated snapshot (`gpt-6-sol-2026-09-14`), and are
checked before any request.
For inexpensive testing, use `gpt-6-luna` with `reasoning_effort="none"`.
`service_tier` takes `"fast"` (OpenAI still accepts `"priority"` as an alias), `"flex"`,
`"scale"`, `"default"` and `"auto"`; `"ultrafast"` is a Responses tier of `gpt-6-astra` and
`gpt-6.1-sol`. [`estimated_cost`](@ref) bills OpenAI's cache writes (1.25× input for 30
minutes, 2.5× for 12 hours), prompts over 272K input tokens and the billed tier at the
pricing page's rates. On `OPENAIServiceEndpoint`, `max_tokens` goes out as
`max_completion_tokens` and `user` throws `ArgumentError` (use `safety_identifier` and
`prompt_cache_key`). See [OpenAI's migration guide](https://developers.openai.com/api/docs/guides/latest-model).

## Anthropic (native Messages API)

`ANTHROPICServiceEndpoint` calls Anthropic's native `/v1/messages` API
(`x-api-key` header, `anthropic-version: 2026-10-21`). Default model `claude-opus-5-5`;
`max_tokens` is required on the wire and defaults to 16000 when you omit it (it caps
thinking plus text, and current Claude models think by default — raise it for long
generations or `xhigh`/`max` effort). The endpoint declares `:chat`, `:tools`,
`:streaming`, `:json_output` and `:models` ([`list_models`](@ref)).

The bare type is the endpoint with no options. An instance,
`ANTHROPICServiceEndpoint(; betas, user_profile_id, workspace_id, inference_geo)`, adds
them to every request made through it: `betas` comma-joined in the `anthropic-beta` header,
`user_profile_id` (the `uprof_` id of the end customer the requests are for) in
`anthropic-user-profile-id`, `workspace_id` in `anthropic-workspace-id`, and `inference_geo`
(`"global"` or `"us"`) in the request body. [`list_models`](@ref) sends only the workspace
of these, the one header besides `anthropic-version` that Anthropic's Models API reference
names (it marks `anthropic-beta` deprecated on `GET /v1/models`). A blank value, a control character, a comma in
a beta name or another region throws `ArgumentError`. UniLM sends a beta's name only: the
request fields and response blocks of beta features are not modelled. Either form goes
wherever a `service` goes, and instances with the same options are equal. The options are
read-only: to change one, build a new endpoint (beta names changed in place are checked
again when the header is built, and one that breaks the rules is refused before anything is
sent).

```julia
claude = ANTHROPICServiceEndpoint(user_profile_id="uprof_011CZkZCu8hGbp5mYRQgUmz9", inference_geo="us")
chat = Chat(service=claude, model="claude-haiku-5-5")
```

How a neutral `Chat` maps onto the Messages API:

| `Chat` field | Messages API |
|---|---|
| leading system messages | the top-level `system` prompt (joined with blank lines) |
| later system messages | kept in place as `role: "system"` on models that accept mid-conversation system messages, where the placement is valid (after a user turn, and last or followed by an assistant turn); otherwise joined into the top-level `system` prompt — except after an assistant turn on Claude Fable 5.1, Opus 5.5, Sonnet 5.5, Haiku 5.5 and any id no row names (below) |
| user message attachments | an [`ImageAttachment`](@ref) becomes a base64 `image` block and a [`DocumentAttachment`](@ref) a base64 `document` block (`application/pdf`), placed before the message's text, as Anthropic's vision and PDF guides recommend; empty text is left out. An [`AudioAttachment`](@ref) or an image `detail` level throws `ArgumentError`: the Messages API has no audio input and no detail level |
| `tool` messages / assistant `tool_calls` | `tool_result` blocks (consecutive results in one user turn) / `tool_use` blocks; an assistant turn with no text, no tool calls and no Claude content blocks is left out. A captured Claude turn goes back without its empty text blocks (the API refuses empty text), and a turn holding nothing else is left out |
| `max_completion_tokens`, else `max_tokens` | `max_tokens` (default 16000) |
| `tools` (`FunctionSignature.strict`) | `tools` with `input_schema` (and a top-level `strict`) |
| `tool_choice` | `"auto"` → `auto`, `"none"` → `none`, `"required"` → `any`, a `GPTToolChoice` → `tool`; any other string throws |
| `parallel_tool_calls = false` (the `Chat` default when `tools` is set) | `tool_choice.disable_parallel_tool_use = true` — so a tool request allows one tool call per turn unless you set `parallel_tool_calls = true` |
| `response_format` JSON Schema | `output_config.format = {type: "json_schema", schema}` (structured outputs; the schema `name`, `description` and `strict` have no counterpart); `json_object` throws — there is no schema-less JSON mode |
| `reasoning_effort` | `output_config.effort` plus adaptive `thinking` with `display: "summarized"`, per model (below) |
| `prompt_cache_options` | a top-level `cache_control` — [automatic caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching#automatic-caching), the breakpoint on the last cacheable block: `{type: "ephemeral"}`, with its `ttl` when set to `"5m"` or `"1h"`; `mode` may be `"implicit"` or unset. `mode="explicit"` (a `Chat` marks no breakpoints), `prewarm=true`, `comparison_response_id` and the OpenAI ttls `"30m"` and `"12h"` throw `ArgumentError`. A prompt shorter than the model's minimum (512 tokens on the 5.5 models) is not cached and the API raises no error: the usage reports no cache write |
| `safety_identifier`, `user`, `metadata["user_id"]` | `metadata.user_id` — set one, or set them equal; `metadata` may carry only `user_id` |
| `service_tier` | `"auto"` or `"standard_only"`; `"fast"` is [fast mode](https://platform.claude.com/docs/en/build-with-claude/fast-mode), `speed: "fast"` and no `service_tier`, on Claude Opus 5.5 — and on Opus 5 and Opus 4.8, in research preview, only through an endpoint whose `betas` hold `"fast-mode-2026-02-01"`; it throws on other models |
| `stop`, `stream` | `stop_sequences`, `stream` |
| `n` | must be `1` |
| `temperature`, `top_p` | none: API version `2026-10-21` removed them, and answers HTTP 400 for either on every model, even at its default value — setting one throws `ArgumentError` |
| `moderation` (OpenAI-only), `seed`, `logprobs`, `top_logprobs`, `presence_penalty`, `frequency_penalty`, `logit_bias`, `verbosity`, `store`, `prompt_cache_key`, `stream_options`, `prediction`, `modalities`, `audio`, `web_search_options` | no counterpart: setting any of them throws `ArgumentError` naming the fields |

`reasoning_effort` per model — the model id decides, or the id with a dated snapshot
suffix (`claude-opus-4-5-20251101` takes the Opus 4.5 row); no model takes another's
row by a shared prefix (`claude-sonnet-5-5` is not `claude-sonnet-5`):

| Models | `"low"` … `"max"` | `"none"` |
|---|---|---|
| Opus 5.5, Fable 5.1, Fable 5, Mythos 5.1, Mythos 5 (thinking always on) | effort + adaptive thinking | throws: thinking cannot be disabled |
| Sonnet 5.5 (adaptive thinking on by default; `"disabled"` is a 400) | effort + adaptive thinking | `thinking = {type: "between_tools"}` (up-front thinking off), no effort |
| Haiku 5.5, Opus 5, Sonnet 5 (adaptive thinking on by default) | effort + adaptive thinking | `thinking = {type: "disabled"}` |
| Opus 4.8, Opus 4.7, Opus 4.6, Sonnet 4.6 (thinking off by default) | effort + adaptive thinking (so the request reasons); the 4.6 models have no `xhigh` | nothing sent |
| Opus 4.5 | effort `low`/`medium`/`high` only, no adaptive thinking | nothing sent |
| Haiku 4.5 | throws: no effort parameter | throws |
| an id no row names (a newer model) | effort + adaptive thinking, unvalidated | `thinking = {type: "disabled"}` |

Adaptive thinking goes out with `display: "summarized"`: the current Claude models return
their thinking blocks with the text omitted unless asked
([thinking display](https://platform.claude.com/docs/en/build-with-claude/thinking#controlling-thinking-display)),
so [`reasoning_text`](@ref) holds the summarized thinking of a request that sets
`reasoning_effort`; without it the thinking text comes back omitted, or the model does not
think, and `reasoning_text` is `nothing`.

`"minimal"` throws on every Claude model (use `"low"`). Other requests Claude answers
with HTTP 400 are refused locally, before the round trip: `temperature` or `top_p` on
every model; a forced `tool_choice` (`"required"` or a named function) on Opus 5.5,
Sonnet 5.5, Fable 5.1 and Mythos 5.1 (use `"auto"` with strict tools, or a JSON Schema
`response_format`); a conversation that ends with an assistant turn (response
prefill) from the 4.6 generation on, Sonnet 5.5 and Haiku 5.5 included; `service_tier="fast"`
on a model without fast mode (Opus 4.6, which would run the request at standard speed,
included), or on Opus 5 and Opus 4.8 without the `fast-mode-2026-02-01` beta; and an
endpoint's `inference_geo` on the Claude 4.5 models (Opus 4.5, Haiku 4.5). An id no row names gets
only the rules that hold for every model, and append-only system placement.

Claude Fable 5.1, Opus 5.5, Sonnet 5.5 and Haiku 5.5 accept a thinking block only while
everything before it — the top-level `system` prompt, the `tools` and the earlier
messages — is unchanged
([preserved thinking](https://platform.claude.com/docs/en/build-with-claude/preserved-thinking)).
On these models, and on a Claude model UniLM has no row for (Anthropic asks every
integration to be append-only), UniLM never edits the top-level `system` prompt once an
assistant turn is in the conversation: a system message pushed after a reply that cannot
stay in place is sent right after the next user turn, so each request extends the one
before it. On every model, a system message pushed after a reply needs a user turn
after it — a user message, or, after a reply that calls tools, the tool results: until one
follows, the request throws `ArgumentError` naming which (sent as it stands, it would be a
response prefill), also when a turn left out of the request (a refusal, an empty reply)
comes after the system message. Push your own messages and send the replies back as they
came; editing earlier turns in `chat.messages` breaks the same check.

```julia
chat = Chat(service=ANTHROPICServiceEndpoint, model="claude-haiku-5-5")
push!(chat, Message(Val(:system), "You are a helpful assistant."))
push!(chat, Message(Val(:user), "Is 391 prime?"))
chatrequest!(chat)                                         # reply appended
push!(chat, Message(Val(:system), "From now on, answer in French."))
push!(chat, Message(Val(:user), "And 397?"))
chatrequest!(chat)   # system: "You are a helpful assistant."; the new instruction follows "And 397?"
```

Decoding: `stop_reason` maps to `finish_reason` — `end_turn`/`stop_sequence` →
`"stop"`, `tool_use` → `"tool_calls"`, `max_tokens` and
`model_context_window_exceeded` → `"length"`, `refusal` → `"content_filter"`. A
refusal carries no content or tool calls (partial output before it is discarded;
deltas already streamed to your callback cannot be recalled), and `refusal_message`
holds `stop_details.explanation` or a default text. A `200` without a content array or
a `stop_reason` is an `LLMCallError`. `result.model` is the model that served the
reply. In `usage`, `prompt_tokens` counts cache reads and writes with the uncached
input (`cached_tokens` is the cache-read share, `cache_write_tokens` the writes and
`cache_write_extended_tokens` the 1-hour ones), `reasoning_tokens` comes from
`output_tokens_details.thinking_tokens`, `service_tier` is the reported tier —
`"fast"` for a request served in fast mode — and `inference_geo` the region inference
ran in; a stream reports the same usage. The request id is read from the `request-id`
header.

[`estimated_cost`](@ref) bills Anthropic's surcharges on the model's price row: cache
writes at 1.25× the input rate (5-minute) or 2× (1-hour), every rate 2× in fast mode
and 1.1× for US-only inference (`inference_geo == "us"`) on Claude 4.6 and later models
(the 4.5 models take no `inference_geo` and bill at the standard rates), and every rate 5×
for a Claude Haiku 5.5 prompt over 100,000 tokens (cache reads and writes included); the
multipliers stack.

```@example backends
claude_chat = Chat(service=ANTHROPICServiceEndpoint, model="claude-haiku-5-5")  # native Messages API
push!(claude_chat, Message(Val(:system), "You are a helpful assistant."))
push!(claude_chat, Message(Val(:user), "Say hello in one short sentence."))
result = chatrequest!(claude_chat)
if result isa LLMSuccess
    println(result.message.content)
else
    println("Request failed — see result for details")
end
```

A PDF, a prompt cached for an hour, and the summarized thinking:

```julia
chat = Chat(service=ANTHROPICServiceEndpoint, model="claude-haiku-5-5", reasoning_effort="low",
            prompt_cache_options=PromptCacheOptions(ttl="1h"))
push!(chat, Message(Val(:system), "You summarize reports."))
push!(chat, Message(Val(:user), "What does the report conclude?", DocumentAttachment("report.pdf")))
result = chatrequest!(chat)
reasoning_text(result)                       # the thinking, summarized
result.usage.cache_write_extended_tokens     # tokens written to the 1-hour cache
```

!!! note "Thinking models round-trip automatically"
    Claude models that emit thinking blocks (e.g. `claude-sonnet-5`) require
    those blocks — signatures intact — to be echoed verbatim on the next
    request of a tool-calling turn. UniLM captures the provider-native content
    on `Message.provider_content` at decode time (non-streaming and streaming)
    and echoes it automatically when the same provider encodes the
    conversation again, so multi-turn tool use works out of the box; a turn cut
    at `max_tokens` that holds only thinking blocks goes back too. A
    conversation moved to another Claude model keeps the blocks: the API reads
    the ones that model can and drops the others without an error. Moving a
    conversation to a different provider falls back to the neutral
    text+tool_calls form (thinking is dropped, as other providers cannot
    verify another vendor's signatures).

## Responses API Backend

The Responses API also supports the `service` parameter:

```@example backends
r = Respond(
    service=UniLM.OPENAIServiceEndpoint,
    model="gpt-5.2",
    input="Hello!",
)
println("Service: ", r.service)
println("Model: ", r.model)
```

## Ollama (local)

[`OllamaEndpoint`](@ref) speaks Ollama's native API — the context window, keep-alive,
`think` and `truncate` fields the OpenAI-compatible route ignores — and needs no key. [Local
Models with Ollama](@ref ollama_guide) covers it in full.

```@example backends
ep = OllamaEndpoint(num_ctx=32_768)   # OLLAMA_HOST, else http://127.0.0.1:11434
chat = Chat(service=ep, model="gemma4:e4b")
println("URL: ", UniLM.get_url(chat))
```

## OpenAI-Compatible Providers (Generic Endpoint)

Any provider that implements the OpenAI-compatible `/v1/chat/completions` endpoint can be
used with [`GenericOpenAIEndpoint`](@ref). This includes vLLM, LM Studio, Mistral, and
many others.

### Mistral

```julia
chat = Chat(service=MistralEndpoint(), model="mistral-large-latest")
result = chatrequest!(chat)
```

### DeepSeek

```julia
chat = Chat(service=DeepSeekEndpoint())                        # default: deepseek-flash
chat = Chat(service=DeepSeekEndpoint(), model="deepseek-v4-pro")
```

The default model for chat and FIM is `deepseek-flash`. DeepSeek models think by
default and return their chain of thought as `reasoning_content`; UniLM keeps it on
the assistant `Message` as `ProviderContent(:deepseek, …)` (streamed or not) and sends
it back on those assistant turns whenever the request carries `tools` — DeepSeek's
thinking mode requires that for multi-turn tool use and ignores it otherwise. Streams
request `include_usage` automatically, and cache hits (`prompt_cache_hit_tokens`) are
priced at the cached rate. The price rows use DeepSeek's peak rates; off-peak hours
bill half, so estimates for off-peak calls run high. FIM and prefix completion go to
the beta base URL (see [FIM & Prefix Completion](@ref completions_guide)).

### vLLM / LM Studio

```julia
# vLLM
chat = Chat(service=GenericOpenAIEndpoint("http://localhost:8000", ""), model="meta-llama/Llama-3.1-8B")

# LM Studio
chat = Chat(service=GenericOpenAIEndpoint("http://localhost:1234", ""), model="loaded-model")
```

### Anthropic (OpenAI-compatible shim)

Prefer the native `ANTHROPICServiceEndpoint` (the **Anthropic (native Messages
API)** section above). For evaluation only, Anthropic also exposes an
OpenAI-compatible endpoint — which Anthropic itself calls "not a long-term or
production-ready solution" (features like `response_format` and `strict` are
ignored). `GenericOpenAIEndpoint` appends `/v1/chat/completions` to its base URL, so
the base is the bare host:

```julia
chat = Chat(
    service=GenericOpenAIEndpoint("https://api.anthropic.com", ENV["ANTHROPIC_API_KEY"]),
    model="claude-sonnet-4-6"
)
```

### Custom Provider

```@example backends
ep = GenericOpenAIEndpoint("https://my-llm-server.example.com", "sk-my-key")
chat = Chat(service=ep, model="my-model")
println("URL: ", UniLM.get_url(chat))
println("Has auth: ", any(p -> p.first == "Authorization", UniLM.auth_header(ep)))
```

### Embeddings with Generic Endpoint

Embeddings also support the `service` parameter:

```@example backends
emb = Embeddings("test"; service=GenericOpenAIEndpoint("http://127.0.0.1:8000", ""), model="bge-m3")
println("URL: ", UniLM.get_url(emb))
```

## API Compatibility Tiers

| API Surface | Standard Status | Supported Providers |
|---|---|---|
| Chat Completions | De facto standard | OpenAI, Azure, Gemini, Mistral, DeepSeek, vLLM, LM Studio, Anthropic* (Ollama: native `/api/chat`) |
| Embeddings | Widely adopted | OpenAI, Gemini, Mistral, vLLM (Ollama: native `/api/embed`) |
| Responses API | Emerging (Open Responses) | OpenAI, Ollama (stateless), vLLM, Amazon Bedrock |
| FIM Completion | Provider-specific | DeepSeek (beta), Mistral (`/v1/fim/completions`), Ollama |
| Image Generation | Limited | OpenAI |

*Anthropic compat layer is not production-recommended by Anthropic.

## Querying Provider Capabilities

Use [`has_capability`](@ref) to check what a provider supports before making requests:

```@example backends
for (name, svc) in [
    ("OpenAI", OPENAIServiceEndpoint),
    ("DeepSeek", DeepSeekEndpoint("k")),
    ("Ollama", OllamaEndpoint())
]
    caps = join(sort(collect(provider_capabilities(svc))), ", ")
    println("$name: $caps")
end
```

```@example backends
# Check specific capabilities
println("DeepSeek FIM: ", has_capability(DeepSeekEndpoint("k"), :fim))
println("OpenAI FIM: ", has_capability(OPENAIServiceEndpoint, :fim))
```

## [Keep up with provider models](@id backends_provider_models)

Providers add models, deprecate them and shut them down on their own schedules.
[`list_models`](@ref) shows what OpenAI, Anthropic and Gemini serve your key today, with each
model's token limits, capabilities and lifecycle dates ([Model Listings](@ref models_api)):

```julia
using Dates

r = list_models(service=ANTHROPICServiceEndpoint)
r isa ModelListSuccess || error("the listing failed: $r")
for m in r.models
    !isnothing(m.retires) && m.retires <= today() + Day(90) && println(m.id, " retires on ", m.retires)
end
```

UniLM checks a request against its rules for the model it names and prices the result at the
model's [`DEFAULT_PRICING`](@ref) row. A request for a model with no rule row of its own is
still held to the provider-wide rules — no Claude request sets `temperature`, for one — and the
provider's answer checks the rest; a model with no price row costs \$0 ([Unpriced models cost
\$0](@ref unpriced-zero)). Before you adopt a newly listed model, look for its price row as the
cost code does — the id, else the id without its dated snapshot suffix (`-YYYY-MM-DD` on OpenAI,
`-YYYYMMDD` on Anthropic) — or add one:

```julia
priced(id) = haskey(DEFAULT_PRICING, id) ||
             haskey(DEFAULT_PRICING, replace(id, r"-(\d{4}-\d{2}-\d{2}|\d{8})$" => ""))
[m.id for m in r.models if !priced(m.id)]
```

UniLM's release gate runs the same listings against its tables
(`test/integration_provider_drift.jl`): every listed model of the families UniLM documents —
GPT-6, GPT-5.6 and GPT-5.4 mini; Claude 4.5 and later, and every Claude model UniLM has rules
for; Gemini 3.5 and later text models, and every Gemini text model UniLM has thinking levels
for but the `-latest` aliases, priced at the model that serves them — must have a price row,
unless its provider publishes no price (Claude Mythos Preview), and a rule row too unless it is
GPT-5.4 mini or a Gemini model that does not think; every model UniLM uses by default, in its
live tests or in the Julia code of these pages must be listed — an alias such as
`claude-haiku-4-5` as its dated snapshot — and the gate must run before its shutdown date,
where the listing gives one (Gemini's gives none); and each Claude rule row must agree with the
thinking types and effort levels Anthropic lists for the model. The gate logs the OpenAI and
Gemini models that no family, older generation or other known kind covers, such as a new
generation.

## See Also

- [`ServiceEndpoint`](@ref), [`GenericOpenAIEndpoint`](@ref) — endpoint types
- [`OllamaEndpoint`](@ref), [`MistralEndpoint`](@ref) — convenience constructors
- [`OPENAIServiceEndpoint`](@ref), [`AZUREServiceEndpoint`](@ref), [`GEMINIServiceEndpoint`](@ref) — built-in backends
