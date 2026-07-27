# [Custom Backends](@id custom_backends_guide)

UniLM's provider surface is a typed, open extension point: you add a backend by defining a
[`ServiceEndpoint`](@ref) subtype and the handful of methods that route and — when the wire
differs — translate its requests. The request driver, retries, cost accounting, tool loop,
and streaming are not provider-specific; they call a small **wire seam** through multiple
dispatch, so a new backend plugs in without touching that machinery.

There are two cases, depending on whether your provider speaks the OpenAI wire.

## Case 1 — an OpenAI-compatible provider

If the provider implements OpenAI's `/v1/chat/completions` (and, optionally, the
`/v1/responses` and `/v1/embeddings` surfaces), subtype [`OpenAIWireEndpoint`](@ref).
Subtypes inherit the OpenAI request/response encoding and the SSE stream handling for free;
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

If your provider also serves the Responses/agentic surface or embeddings from one host,
define the `_api_base_url` pattern instead of a per-verb URL — it is the single base that
the Responses URL (`_api_base_url(service) * "/v1/responses"`) and the embeddings URL are
built from. `GenericOpenAIEndpoint` is the built-in reference for this shape:

```julia
UniLM._api_base_url(s::MyEndpoint) = rstrip(s.base_url, '/')
UniLM.get_url(s::MyEndpoint, ::Chat)       = UniLM._api_base_url(s) * "/v1/chat/completions"
UniLM.get_url(s::MyEndpoint, ::Embeddings) = UniLM._api_base_url(s) * "/v1/embeddings"
```

For a provider that is a plain OpenAI endpoint at a custom URL you need no new type at all —
[`GenericOpenAIEndpoint`](@ref)`(base_url, api_key)` already does exactly this. Define a new
`OpenAIWireEndpoint` subtype when you want a named type, provider-specific defaults
(`default_model`, `provider_capabilities`), or custom routing/auth.

Singleton endpoints (no fields) dispatch on the type — `get_url(::Type{MyEndpoint}, ::Chat)`;
field-bearing endpoints dispatch on the instance, as above.

## Case 2 — a provider with a different wire

If the provider speaks a genuinely different wire (its own request shape, response shape, or
SSE dialect), subtype [`ServiceEndpoint`](@ref) directly and additionally implement the three
chat wire-seam methods. They translate between the neutral `Chat`/`Message` types and the
provider's wire:

| Method | Signature | Return contract |
| :-- | :-- | :-- |
| `encode_request` | `encode_request(service, chat::Chat)` | `String` — the request body to POST |
| `decode_response` | `decode_response(service, resp::HTTP.Response)` | `(; message::Message, usage::Union{TokenUsage,Nothing})` — the parsed 200 response |
| `handle_sse_event!` | `handle_sse_event!(service, event, payload, state::StreamState)` | `Symbol` — `:continue` \| `:done` \| `:error`; mutates `state` in place |

`handle_sse_event!` is called once per streamed SSE event (`event`/`payload` are strings):
it appends text and tool-call deltas onto `state`, and returns `:done` at the provider's
end-of-stream sentinel, `:error` on an in-band error event, or `:continue` otherwise.

The in-repo reference implementation is the **Anthropic backend** (`src/anthropic.jl`):
`ANTHROPICServiceEndpoint` subtypes `ServiceEndpoint` and implements all three methods
against Anthropic's native Messages API — including capturing provider-native content blocks
for verbatim multi-turn round-trip. Read it as a complete worked example rather than copying
it here. The native Gemini backend (`src/gemini.jl`) is a second example, with a different
end-of-stream rule (EOF, not a sentinel).

To also serve the agentic [`respond`](@ref) verb on a native wire, override the agentic seam
the same way — `encode_agentic` / `decode_agentic` / `decode_agentic_stream`, plus
`_agentic_url` for routing; the Gemini Interactions backend (`src/interactions.jl`) is the
reference.

## Fail-loud contract

The wire seam has **no permissive fallback**. A bare `ServiceEndpoint` subtype that omits a
seam method fails with a `MethodError` at call time, rather than silently emitting
OpenAI-shaped requests to a provider that does not speak that wire. Subtyping
`OpenAIWireEndpoint` is the explicit opt-in to the OpenAI wire; subtyping `ServiceEndpoint`
is the commitment to implement the seam yourself.

```@example custom_backends
using UniLM
# The OpenAI-wire opt-in is a real supertype relationship:
println(OPENAIServiceEndpoint  <: OpenAIWireEndpoint)   # true  — inherits the wire
println(ANTHROPICServiceEndpoint <: OpenAIWireEndpoint) # false — native wire, own seam
```

## See Also

- [`OpenAIWireEndpoint`](@ref), [`ServiceEndpoint`](@ref) — the two extension supertypes
- [Multi-Backend Support](@ref backend_guide) — using the built-in backends
- [`GenericOpenAIEndpoint`](@ref) — the built-in configurable OpenAI-compatible endpoint
