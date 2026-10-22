# [Model Listings](@id models_api)

[`list_models`](@ref) asks OpenAI, Anthropic or Gemini which models it serves your key, and
returns each as a [`ProviderModel`](@ref): the id a request names, the provider's display
name, the token limits, the capabilities the provider reports, and the lifecycle dates —
when the model was released, when it was deprecated, and when it retires or shuts down.
Anthropic and Gemini answer in pages, and `list_models` fetches every page. A listing bills
no tokens.

```julia
r = list_models(service=OPENAIServiceEndpoint)
if r isa ModelListSuccess
    for m in r.models
        isnothing(m.retires) || println(m.id, " shuts down on ", m.retires)
    end
end
```

## What each provider reports

| Field | OpenAI (`GET /v1/models`) | Anthropic (`GET /v1/models`) | Gemini (`GET /v1beta/models`) |
|:--|:--|:--|:--|
| `id` | `id` | `id` | `name` without its `models/` prefix |
| `display_name` | — | `display_name` | `displayName` |
| `created` | `created` (Unix seconds) | `created_at` | — |
| `input_token_limit` | — | `max_input_tokens` | `inputTokenLimit` |
| `output_token_limit` | — | `max_tokens` | `outputTokenLimit` |
| `capabilities` | — | `capabilities`: thinking types, effort levels, inputs, tools, … | `supportedGenerationMethods`, `thinking` |
| `deprecated` | — | `deprecated_at` | — |
| `retires` | `shutdown_date` | `retires_at` | — |

A dash reads as `nothing`, or an empty `Dict` for `capabilities`. Times are in UTC, and
Anthropic's timestamps give their UTC date for `deprecated` and `retires`; the Unix epoch,
which Anthropic gives for an unknown release date, reads as no `created` at all. Each
entry keeps the provider's object in `raw`, and the success keeps every page's body in
its own `raw`.

A listing is the provider's view for the key that asks, so two accounts can see different
models. Anthropic lists active and deprecated models, not retired ones — its `lifecycle`
field, in `raw`, says which — and may still revise a `retires_at` it lists. OpenAI's
listing showed models whose `shutdown_date` had passed on 22 October 2026, so a listed
model is not always one that still answers: compare `retires` with today's date before
relying on it.

The Anthropic capabilities are the ones UniLM's request checks encode for each Claude
model — the thinking types it accepts (`adaptive`, `disabled`, `between_tools`) and its
effort levels — so they show what a request may set on a model UniLM has no rule for.

## Results

```@docs
ProviderModel
ModelListSuccess
ModelListFailure
ModelListCallError
```

A missing key, a network failure, a timeout or a cancel gives a `ModelListCallError`, and
so does a reply that is not the documented listing; an error status on any page gives a
`ModelListFailure`. [`list_models`](@ref) is documented with the System One verbs: on
TypeSafe it lists the Jev models and aliases, on an [`OllamaEndpoint`](@ref) the models a
server has installed.
