# [Model Listings](@id models_api)

[`list_models`](@ref) asks OpenAI, Anthropic or Gemini which models it serves your key, and
returns each as a [`ProviderModel`](@ref): the id a request names, the provider's display
name, the token limits, the capabilities the provider reports, and the lifecycle dates —
when the model was released, when it was deprecated, and when it retires or shuts down.
Anthropic and Gemini answer in pages, and `list_models` fetches every page, each asking for
1000 models, the most either API serves on one page. A listing bills no tokens.

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
field, in `raw`, says which — and may still revise a `retires_at` it lists. UniLM sends no
`lifecycle` filter: the listing without one is the active and the deprecated models (the
same body as `lifecycle[]=active&lifecycle[]=deprecated`), `ProviderModel.deprecated` tells
them apart, and a retired model answers no request (on 16 November 2026 Anthropic listed no
retired model for UniLM's key, and answered a retired id with HTTP 404). Nor does it send an
endpoint's `betas` to the Models API: the listing and the lookup came back the same with each
beta tried and without one, and a beta name Anthropic does not know turns them into an error. OpenAI keeps
listing a model after its `shutdown_date`, so a listed model is not always one that still
answers: compare `retires` with today's date before relying on it.

The Anthropic capabilities are the ones UniLM's request checks encode for each Claude
model — the thinking types it accepts (`adaptive`, `disabled`, `between_tools`) and its
effort levels — so they show what a request may set on a model UniLM has no rule for.

## [One model by id](@id models_lookup)

[`retrieve_model`](@ref) asks for one model by its id — `GET /v1/models/{model}` on OpenAI and
Anthropic, `GET /v1beta/models/{model}` on Gemini — and returns a [`ModelSuccess`](@ref) whose
`model` is the `ProviderModel` the listing holds for it, read the same way. Anthropic resolves
an alias to the dated id it names: `"claude-opus-4-5"` comes back as
`"claude-opus-4-5-20251101"`. A model the key cannot use, a retired one among them, is a
`ModelListFailure` with status 404. TypeSafe and Ollama look up no model by id
([`model_info`](@ref) shows an installed Ollama model), and an endpoint without the `:models`
capability has no models to look up: both throw `ArgumentError` before any request. A lookup
bills no tokens and is not recorded, so its output here is quoted from a live run:

```julia
r = retrieve_model("gemini-3.8-flash"; service=GEMINIServiceEndpoint)
r isa ModelSuccess && (r.model.input_token_limit, r.model.output_token_limit)
# Looked up on 16 November 2026: (1048576, 65536)
```

```@docs
retrieve_model
```

## Results

```@docs
ProviderModel
ModelListSuccess
ModelSuccess
ModelDeleteSuccess
ModelListFailure
ModelListCallError
```

A missing key, a network failure, a timeout or a cancel gives a `ModelListCallError`, and
so does a reply that is not the documented listing or model; an error status on any page,
or on a lookup, gives a `ModelListFailure`. A `ModelDeleteSuccess` is the result of a
fine-tuned model's deletion on OpenAI (`DELETE /v1/models/{model}`), which holds the id the
reply names once the reply confirms the deletion.

[`list_models`](@ref) is documented with the System One verbs: on TypeSafe it lists the Jev
models and aliases, on an [`OllamaEndpoint`](@ref) the models a server has installed.
