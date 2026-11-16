# [Token Counting](@id token_counting_api)

[`count_tokens`](@ref) asks a provider how many input tokens a request would bill, without
running it: the request object the generation verb takes, encoded as that verb encodes it.

| Provider | Request | Route | Total |
|:--|:--|:--|:--|
| OpenAI | a [`Respond`](@ref) | `POST /v1/responses/input_tokens` | `input_tokens` |
| Anthropic | a [`Chat`](@ref) | `POST /v1/messages/count_tokens` | `input_tokens` |
| Gemini | a `Chat` on `GEMINIServiceEndpoint` | `POST /v1beta/models/{model}:countTokens` | `totalTokens` |

```julia
chat = Chat(service=ANTHROPICServiceEndpoint, model="claude-haiku-5-5", messages=[Message(Val(:user), "Hello")])
r = count_tokens(chat)
r isa TokenCountSuccess && r.input_tokens
# Counted on 16 November 2026: 11
```

Each count sends the generation body without the fields the counting route does not take, which
bound or deliver the output, name the end user, or pick a tier, a region or a stored copy, and so
change no input token. On Anthropic those are `max_tokens`, `stop_sequences`, `stream`,
`metadata`, `service_tier` and `inference_geo`; an endpoint with `betas` counts on the beta route,
its betas and `token-counting-2024-11-01` in the `anthropic-beta` header. On Gemini the count takes
the whole `generateContent` body, tools, system instruction and generation config included. On
OpenAI it takes the input, the instructions, the tools and the tool choice, the previous response
or conversation, the reasoning, the text format and the truncation; a `Respond` that sets
`context_management` or `image_context_compression`, each of which changes the input the model
reads, throws `ArgumentError`. OpenAI counts no `Chat`, and Gemini's Interactions API (a `Respond`
on Gemini) has no count: either throws `ArgumentError` before any request, as an endpoint without
the `:token_count` capability does.

Anthropic's token counting "is free to use"
([Token counting](https://platform.claude.com/docs/en/build-with-claude/token-counting)); OpenAI and
Google publish no price for theirs, and no count reports a usage. A count is not recorded by
[`with_recorded_answers`](@ref): inside a scope it goes to the provider. Gemini's reply also
gives the tokens per modality (`promptTokensDetails`), which stay in `raw`. [`count_input_tokens`](@ref) counts on OpenAI's route by keyword and returns
the reply as a `Dict`.

```@docs
count_tokens
TokenCountSuccess
TokenCountFailure
TokenCountCallError
```
