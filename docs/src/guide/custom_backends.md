# [Write your own backend](@id custom_backends_guide)

```@setup custom_backends
using UniLM
```

UniLM's provider surface is a typed, open extension point: you add a backend by defining a
[`ServiceEndpoint`](@ref) subtype and the handful of methods that route and — when the wire
differs — translate its requests. The request driver, retries, cost accounting, tool loop,
and streaming are not provider-specific; they call a small **wire seam** through multiple
dispatch, so a new backend plugs in without touching that machinery.

```@example custom_backends
service, model = OPENAIServiceEndpoint, "gpt-5.4-mini"
nothing # hide
```

The examples use only the names bound above and the endpoint types they define. `service` and `model` are a
built-in provider and model: bind your own there. The table [at the end](@ref custom_backends_providers) links
each provider's start page, which says what wire its endpoint speaks.

The functions you add methods to — `get_url`, `auth_header`, `default_model`, the chat
seam `encode_request` / `decode_response` / `handle_sse_event!` and the agentic seam
`encode_agentic` / `decode_agentic` / `decode_agentic_stream` — and the stream-state types
`StreamState` / `AgenticStreamState` are UniLM's public extension API: declared `public`,
not exported, so qualify them (`UniLM.get_url`). Each is documented in the
[Extension API](@ref extension_api) reference.

There are two cases, depending on whether your provider speaks the OpenAI wire.

## [Case 1 — an OpenAI-compatible provider](@id custom_backends_openai_wire)

If the provider speaks the OpenAI wire — `/v1/chat/completions`, and optionally the
`/v1/responses` and `/v1/embeddings` surfaces —, subtype [`OpenAIWireEndpoint`](@ref).
Subtypes inherit the OpenAI wire's request and response encoding and its SSE stream handling;
you define only **routing** (`get_url`) and **authentication** (`auth_header`).

```julia
using UniLM

# A fixed-URL, OpenAI-compatible provider.
struct AcmeEndpoint <: OpenAIWireEndpoint
    api_key::String
end

# Routing: where a chat-completions request POSTs to.
UniLM.get_url(::AcmeEndpoint, ::Chat) = "https://api.acme.ai/v1/chat/completions"

# Authentication: the headers attached to every request.
UniLM.auth_header(e::AcmeEndpoint) =
    ["Authorization" => "Bearer $(e.api_key)", "Content-Type" => "application/json"]

chat = Chat(service=AcmeEndpoint(ENV["ACME_API_KEY"]), model="acme-large")
push!(chat, Message(Val(:system), "You are helpful."))
push!(chat, Message(Val(:user), "Hi!"))
result = chatrequest!(chat)   # encode_request / decode_response / SSE are all inherited
```

Embeddings route the same way, through `get_url(::MyEndpoint, ::Embeddings)`, and
[`respond`](@ref) through `get_url(::MyEndpoint, ::Respond)`. The Responses lifecycle
operations (`get_response`, `cancel_response`, `compact_response`, …) build their URLs
from one base instead, the internal hook `UniLM._api_base_url(service)` (the Responses
URL is that base plus `"/v1/responses"`). Defining it is the one step here outside the
public extension API — an underscore name that may change without notice.
`GenericOpenAIEndpoint` is the built-in reference for this shape:

```julia
UniLM._api_base_url(s::MyEndpoint) = rstrip(s.base_url, '/')   # internal hook
UniLM.get_url(s::MyEndpoint, ::Chat)       = UniLM._api_base_url(s) * "/v1/chat/completions"
UniLM.get_url(s::MyEndpoint, ::Embeddings) = UniLM._api_base_url(s) * "/v1/embeddings"
```

For a provider that is a plain OpenAI-compatible server at a custom URL you need no new type at all —
[`GenericOpenAIEndpoint`](@ref)`(base_url, api_key)` already does exactly this. Define a new
`OpenAIWireEndpoint` subtype when you want a named type, provider-specific defaults, or
custom routing/auth:

- **`default_model`** — without a method for your endpoint, `model=` is required:
  `Chat(service=AcmeEndpoint(key))` throws `ArgumentError: model must be specified when
  using AcmeEndpoint` at construction. `UniLM.default_model(::AcmeEndpoint) = "acme-large"`
  makes it optional.
- **`provider_capabilities`** — an endpoint that defines no method is not checked by
  `chatrequest!`, `respond`, `embeddingrequest!`, `generate_image` and `edit_image`, and every
  other verb refuses it: most with a `MethodError` for `provider_capabilities`, the decision
  verbs with an `ArgumentError` (see [Provider Capabilities](@ref capabilities_api));
  declaring a set opts into validation against it.
- **Streamed usage** — an endpoint of your own does not ask a stream to report its token usage
  (`stream_options.include_usage`). Set `stream_options=Dict("include_usage" => true)` on a
  streamed `Chat` when your server supports it; otherwise a streamed turn carries usage only when
  your server sends it unasked. Which built-in endpoints ask is on their entries, in the
  streaming section or the section of the endpoint it concerns.

An endpoint passed as its type (`service=MyEndpoint`, as the built-in singletons are) dispatches
on the type — `get_url(::Type{MyEndpoint}, ::Chat)`; one passed as an instance dispatches on the
instance, as above.

## [Case 2 — a provider with a different wire](@id custom_backends_native)

If the provider speaks a genuinely different wire (its own request shape, response shape, or
SSE dialect), subtype [`ServiceEndpoint`](@ref) directly and additionally implement the three
chat wire-seam methods. They translate between the neutral `Chat`/`Message` types and the
provider's wire:

| Method | Signature | Return contract |
| :-- | :-- | :-- |
| `encode_request` | `encode_request(service, chat::Chat)` | `String` — the request body to POST |
| `decode_response` | `decode_response(service, resp::HTTP.Response)` | `(; message::Message, usage::Union{TokenUsage,Nothing}, model::Union{String,Nothing})` — the parsed 200 response; `model`, the model the provider reports having served, is optional |
| `handle_sse_event!` | `handle_sse_event!(service, event, payload, state::StreamState)` | `Symbol` — `:continue` \| `:done` \| `:error`; mutates `state` in place |

`handle_sse_event!` is called once per streamed SSE event (`event`/`payload` are strings):
it appends text and tool-call deltas onto `state`, records the usage and the served model
when the provider reports them (`state.usage`, `state.model`), and returns `:done` at the
provider's end-of-stream sentinel, `:error` on an in-band error event, or `:continue`
otherwise. The served model becomes the result's `model`, which [`estimated_cost`](@ref)
prices before the model requested; a decoder that returns only `(; message, usage)` and a
handler that leaves `state.model` unset report none.

The native backends in UniLM's `src/` are complete worked examples. `src/anthropic.jl`
implements all three methods, and captures the provider's own content blocks for a verbatim
round-trip on the next turn; `src/gemini.jl` is a second example, whose stream ends at EOF
rather than at a sentinel. Read them rather than copying them here.

To also serve the agentic [`respond`](@ref) verb on a native wire, override the agentic seam
the same way — `encode_agentic` / `decode_agentic` / `decode_agentic_stream` (the stream
handler mutates an `AgenticStreamState`) — and route it with the internal hook
`UniLM._agentic_url(service)`, which `respond` and the lifecycle operations read (an
underscore name outside the public extension API); `src/interactions.jl` is the reference.

Both seams split validation from failure the same way. An encoder runs before any network
I/O and what it throws propagates from the verb, so throw `ArgumentError` for an option your
wire cannot express. An exception thrown by a decoder becomes the verb's typed call-error
result (`LLMCallError` / `ResponseCallError`) with the exception in `cause`. Their stream
handlers differ. `handle_sse_event!` is called once per payload, and an exception it throws
drops that payload, counted in the result's `sse_dropped`. `decode_agentic_stream` is called once
per raw read, which can hold several payloads, and an exception it throws ends the streamed
`respond` as a `ResponseCallError` with the exception in `cause`: to drop an undecodable payload,
skip it in the handler and count it in `state.sse_dropped`, as `src/interactions.jl` does.

## [Fail-loud contract](@id custom_backends_fail_loud)

The wire seam has **no permissive fallback**. Subtyping `OpenAIWireEndpoint` is the explicit
opt-in to the OpenAI wire; subtyping `ServiceEndpoint` is the commitment to implement the seam
yourself. A built-in endpoint shows which it chose:

```@example custom_backends
(service isa Type ? service : typeof(service)) <: OpenAIWireEndpoint
```

`true` says the bound endpoint opted into the OpenAI wire, and `false` that it implements the seam
itself. A `ServiceEndpoint` subtype that defines no encoder sends
nothing: `chatrequest!` throws a `MethodError` for `encode_request` before any request, rather
than sending OpenAI-wire JSON to a provider that does not speak it.

```@example custom_backends
struct BareEndpoint <: ServiceEndpoint end
UniLM.get_url(::BareEndpoint, ::Chat) = "https://llm.example.invalid/v1/chat"
UniLM.auth_header(::BareEndpoint) = ["Content-Type" => "application/json"]

chat = Chat(service=BareEndpoint(), model=model)
push!(chat, Message(Val(:system), "You are helpful."))
push!(chat, Message(Val(:user), "Hi!"))
err = try
    chatrequest!(chat)
catch e
    e
end
err isa MethodError, err.f
```

A missing decoder fails the call too, once the reply has arrived: the result is an `LLMCallError`
whose `cause` is the `MethodError` for `decode_response`. A missing stream handler is refused as a
missing encoder is, before anything is sent, alike on both seams: a streamed `chatrequest!` throws the
`MethodError` for `handle_sse_event!`, and a streamed `respond` the one for `decode_agentic_stream`.

## [See also](@id custom_backends_see_also)

- [`OpenAIWireEndpoint`](@ref), [`ServiceEndpoint`](@ref) — the two extension supertypes
- [Extension API](@ref extension_api) — the public functions and types a backend implements
- [One API across providers](@ref backend_guide) — using the built-in backends
- [`GenericOpenAIEndpoint`](@ref) — the built-in configurable OpenAI-compatible endpoint
- [Write your own decision provider](@ref decisions_custom) — the same for decision models

## [On your provider](@id custom_backends_providers)

```@eval
Main.ProviderFacts.provider_table(Main.ProviderFacts.llm)
```
