# [Multi-Backend Support](@id backend_guide)

UniLM.jl is built around **neutral verbs**: the same `Chat` + [`chatrequest!`](@ref) — with
tools, streaming, and [cost accounting](@ref cost_guide) — run unchanged across every backend;
you only change the `service`. Cost accounting needs a price row for the model: the built-in
table covers the OpenAI, Anthropic, Gemini, DeepSeek and TypeSafe models it lists, and any
other model (Ollama, Mistral, custom) is estimated at `\$0` with a one-time warning unless you
supply `pricing=`. The agentic [`respond`](@ref) verb is neutral the same way
across OpenAI (Responses) and Gemini (Interactions); see [Agentic Workflows](@ref agentic_guide).
Native OpenAI, Anthropic, and Gemini are first-class backends with their own wire formats (each
exercised by live integration tests), not OpenAI-compatible shims.

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
(`gpt-5.2` → `AZURE_OPENAI_DEPLOY_NAME_GPT_5_2`, `gpt-4o-mini` →
`AZURE_OPENAI_DEPLOY_NAME_GPT_4O_MINI`), unless a deployment was registered with
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

### Current model controls

As of September 7, 2026, Gemini 3.8 Flash supports `reasoning_effort="low"`,
`"medium"`, or `"high"`. Native Chat maps this to `thinkingConfig.thinkingLevel`.
The completion cap includes thinking tokens, so a very small cap can produce
an empty answer. Gemini 3.8 rejects `temperature`, `top_p`, and minimal thinking.
On the Interactions API (`respond`), `temperature` and `top_p` stay mapped to
`generation_config` for models that accept them (`gemini-3.7-flash` accepted both
on September 22, 2026).
Native Gemini tool declarations use `parametersJsonSchema`, preserving standard
JSON Schema constraints such as `additionalProperties`.

`response_format` is mapped natively: JSON object and JSON Schema output become
`generationConfig.responseFormat` (see [Structured Output](@ref structured_guide)).
`safety_identifier` is mapped natively to the request label `safety_identifier`;
label values allow at most 63 characters of lowercase letters (international characters
allowed), numeric characters, underscores, and dashes, so a value such as an email address
or a 64-character SHA-256 hex digest throws `ArgumentError` at encode time.
Native Chat returns one candidate and does not implement OpenAI-specific options
such as `seed`, `metadata`, and `stream_options`; those options fail explicitly.
Gemini determines parallel tool use; the inherited `parallel_tool_calls` setting
does not constrain native Gemini.
See [Google's migration guide](https://ai.google.dev/gemini-api/docs/latest-model).

Native Gemini reports `finishReason` truthfully on both the streamed and the
non-streamed path: `STOP` is `"stop"` (`"tool_calls"` when the turn carries function
calls), `MAX_TOKENS` is `"length"`, the safety filters — `SAFETY`, `RECITATION`,
`BLOCKLIST`, `PROHIBITED_CONTENT`, `SPII` and the image filters `IMAGE_SAFETY`,
`IMAGE_PROHIBITED_CONTENT` and `IMAGE_RECITATION` — are `"content_filter"`, and any
other value passes through lowercased (for example `"malformed_function_call"` or
`"no_image"`). A candidate without a `finishReason` reports `nothing`,
and function calls under a reason other than `STOP` keep that reason, so a tool loop
does not run them. A model turn with neither text nor function calls (a refusal, or a
turn spent entirely on thinking) is left out of the next request.

OpenAI defaults to `gpt-5.6-sol`. GPT-5.6, GPT-6 Sol, and GPT-6 Luna Chat
Completions tool calling requires an explicit `reasoning_effort="none"`; use
Responses for reasoning with tools. GPT-6 Sol and Luna also reject `temperature`,
`top_p`, and log probabilities unless reasoning effort is `"none"` (an omitted
effort is the provider default, `"medium"`).
On the native OpenAI endpoint, GPT-6 Sol and Luna also refuse `"minimal"`
reasoning effort; use `"low"` instead.
For inexpensive testing, use `gpt-5.6-luna` and low or no reasoning.
Use `Respond(model="gpt-6-astra", ...)` for Astra tool workflows: Astra tools require
Responses, and Astra rejects sampling controls, log probabilities, and reasoning effort
below `"low"` (`"none"`, `"minimal"`).
The existing `service_tier` field accepts `"fast"`; OpenAI still accepts `"priority"`
as an alias. See [OpenAI's migration guide](https://developers.openai.com/api/docs/guides/latest-model).

## Anthropic (native Messages API)

`ANTHROPICServiceEndpoint` calls Anthropic's native `/v1/messages` API
(`x-api-key` + `anthropic-version` headers). Default model `claude-opus-5-5`;
`max_tokens` is required on the wire and defaults to 16000 when you omit it (it caps
thinking plus text, and current Claude models think by default — raise it for long
generations or `xhigh`/`max` effort). The endpoint declares `:chat`, `:tools`,
`:streaming` and `:json_output`.

How a neutral `Chat` maps onto the Messages API:

| `Chat` field | Messages API |
|---|---|
| leading system messages | the top-level `system` prompt (joined with blank lines) |
| later system messages | kept in place as `role: "system"` on models that accept mid-conversation system messages, where the placement is valid (after a user turn, and last or followed by an assistant turn); hoisted into the top-level `system` prompt otherwise |
| `tool` messages / assistant `tool_calls` | `tool_result` blocks (consecutive results in one user turn) / `tool_use` blocks; an assistant turn with neither text nor tool calls is left out |
| `max_completion_tokens`, else `max_tokens` | `max_tokens` (default 16000) |
| `tools` (`FunctionSignature.strict`) | `tools` with `input_schema` (and a top-level `strict`) |
| `tool_choice` | `"auto"` → `auto`, `"none"` → `none`, `"required"` → `any`, a `GPTToolChoice` → `tool`; any other string throws |
| `parallel_tool_calls = false` (the `Chat` default when `tools` is set) | `tool_choice.disable_parallel_tool_use = true` — so a tool request allows one tool call per turn unless you set `parallel_tool_calls = true` |
| `response_format` JSON Schema | `output_config.format = {type: "json_schema", schema}` (structured outputs; the schema `name`, `description` and `strict` have no counterpart); `json_object` throws — there is no schema-less JSON mode |
| `reasoning_effort` | `output_config.effort` plus adaptive `thinking`, per model family (below) |
| `safety_identifier`, `user`, `metadata["user_id"]` | `metadata.user_id` — set one, or set them equal; `metadata` may carry only `user_id` |
| `service_tier` | `"auto"` or `"standard_only"` |
| `stop`, `temperature`, `top_p`, `stream` | `stop_sequences`, `temperature` (0–1), `top_p`, `stream` |
| `n` | must be `1` |
| `moderation`, `prompt_cache_options` (OpenAI-only), `seed`, `logprobs`, `top_logprobs`, `presence_penalty`, `frequency_penalty`, `logit_bias`, `verbosity`, `store`, `prompt_cache_key`, `stream_options`, `prediction`, `modalities`, `audio`, `web_search_options` | no counterpart: setting any of them throws `ArgumentError` naming the fields |

`reasoning_effort` per family (the longest matching model-id prefix decides):

| Models | `"low"` … `"max"` | `"none"` |
|---|---|---|
| Opus 5.5, Fable 5.x, Mythos 5.x, Mythos Preview (thinking always on) | effort + adaptive thinking; Mythos Preview has no `xhigh` | throws: thinking cannot be disabled |
| Opus 5, Sonnet 5 (adaptive thinking on by default) | effort + adaptive thinking | `thinking = {type: "disabled"}` |
| Opus 4.8, Opus 4.7, Opus 4.6, Sonnet 4.6 (thinking off by default) | effort + adaptive thinking (so the request reasons); the 4.6 models have no `xhigh` | nothing sent |
| Opus 4.5 | effort `low`/`medium`/`high` only, no adaptive thinking | nothing sent |
| Sonnet 4.5, Haiku 4.5 and older | throws: no effort parameter | throws |
| an id no row covers (a newer model) | effort + adaptive thinking, unvalidated | `thinking = {type: "disabled"}` |

`"minimal"` throws on every Claude model (use `"low"`). Other requests Claude answers
with HTTP 400 are refused locally, before the round trip: `temperature` outside
[0, 1] on every model; any `top_p`, or a `temperature` other than the default 1.0, on
Opus 4.7 and later (Opus 4.8, Sonnet 5, Opus 5, Opus 5.5, Fable, Mythos); a forced
`tool_choice` (`"required"` or a named function) on Opus 5.5, Fable 5.1 and Mythos 5.1;
and a conversation that ends with an assistant turn (response prefill) from the 4.6
generation on.

Decoding: `stop_reason` maps to `finish_reason` — `end_turn`/`stop_sequence` →
`"stop"`, `tool_use` → `"tool_calls"`, `max_tokens` and
`model_context_window_exceeded` → `"length"`, `refusal` → `"content_filter"`. A
refusal carries no content or tool calls (partial output before it is discarded;
deltas already streamed to your callback cannot be recalled), and `refusal_message`
holds `stop_details.explanation` or a default text. A `200` without a content array or
a `stop_reason` is an `LLMCallError`. `usage.prompt_tokens` counts cache reads and
writes with the uncached input (`cached_tokens` is the cache-read share), and
`reasoning_tokens` comes from `output_tokens_details.thinking_tokens`. The request id
is read from the `request-id` header.

```@example backends
claude_chat = Chat(service=ANTHROPICServiceEndpoint, model="claude-haiku-4-5")  # native Messages API
push!(claude_chat, Message(Val(:system), "You are a helpful assistant."))
push!(claude_chat, Message(Val(:user), "Say hello in one short sentence."))
result = chatrequest!(claude_chat)
if result isa LLMSuccess
    println(result.message.content)
else
    println("Request failed — see result for details")
end
```

!!! note "Thinking models round-trip automatically"
    Claude models that emit thinking blocks (e.g. `claude-sonnet-5`) require
    those blocks — signatures intact — to be echoed verbatim on the next
    request of a tool-calling turn. UniLM captures the provider-native content
    on `Message.provider_content` at decode time (non-streaming and streaming)
    and echoes it automatically when the same provider encodes the
    conversation again, so multi-turn tool use works out of the box. Moving a
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

## OpenAI-Compatible Providers (Generic Endpoint)

Any provider that implements the OpenAI-compatible `/v1/chat/completions` endpoint can be
used with [`GenericOpenAIEndpoint`](@ref). This includes Ollama, vLLM, LM Studio, Mistral,
and many others.

### Ollama (local)

```@example backends
ep = OllamaEndpoint()  # defaults to http://localhost:11434
chat = Chat(service=ep, model="llama3.1")
println("URL: ", UniLM.get_url(chat))
```

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
emb = Embeddings("test"; service=OllamaEndpoint(), model="nomic-embed-text")
println("URL: ", UniLM.get_url(emb))
```

## API Compatibility Tiers

| API Surface | Standard Status | Supported Providers |
|---|---|---|
| Chat Completions | De facto standard | OpenAI, Azure, Gemini, Mistral, DeepSeek, Ollama, vLLM, LM Studio, Anthropic* |
| Embeddings | Widely adopted | OpenAI, Gemini, Mistral, Ollama, vLLM |
| Responses API | Emerging (Open Responses) | OpenAI, Ollama, vLLM, Amazon Bedrock |
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

## See Also

- [`ServiceEndpoint`](@ref), [`GenericOpenAIEndpoint`](@ref) — endpoint types
- [`OllamaEndpoint`](@ref), [`MistralEndpoint`](@ref) — convenience constructors
- [`OPENAIServiceEndpoint`](@ref), [`AZUREServiceEndpoint`](@ref), [`GEMINIServiceEndpoint`](@ref) — built-in backends
