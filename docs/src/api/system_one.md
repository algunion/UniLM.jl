# [TypeSafe System One API (Jev)](@id system_one_api)

A System One model does not write text. It reads a piece of `state` and answers
the questions you enumerate with a probability distribution over the outcomes
you named, already typed — a `String` option, a `Float64` score, a probability —
so there is nothing to parse. The answer says *what*, the distribution says
*whether to act on it*. [Route and Decide](@ref system_one_guide) is the how-to;
this page starts with setup, models, limits, errors and cost, then documents
every type and verb; [`nl_classify`](@ref), [`nl_dispatch`](@ref) and
[`@branch`](@ref) have a page of their own, [Natural-Language
Dispatch](@ref nl_dispatch_api).

## [Setup](@id jev_setup)

```bash
export TYPESAFE_API_KEY="..."
```

| Variable | Default | Meaning |
| :--- | :--- | :--- |
| `TYPESAFE_API_KEY` | — | Required. A missing key is a [`SystemOneCallError`](@ref), not a thrown `KeyError`. |
| `TYPESAFE_BASE_URL` | `https://api.typesafe.ai` | API root override, for a proxy or a mock server. |
| `TYPESAFE_DEFAULT_MODEL` | `jev-latest` | The model [`ask`](@ref) names when a call does not. |

All three are read at call time. `jev-latest` is an alias that moves when a new
version ships, which suits building; once a confidence threshold is tuned
against a version, pin its id — `ask(...; model="jev-1.13.0")` or
`TYPESAFE_DEFAULT_MODEL=jev-1.13.0` — and move on your own schedule
([Models](https://docs.typesafe.ai/models)). `TYPESAFE_DEFAULT_MODEL` changes
every request that names no model, and with it the key its recorded answer is
filed under: in code whose answers you record, pin with `model =`, or record
with the same environment you replay with ([Test and Develop](@ref
jev_testing_guide)).

[`TYPESAFEServiceEndpoint`](@ref UniLM.TYPESAFEServiceEndpoint) ([Service
Endpoints](endpoints.md)) is not a chat backend: it declares only `:system_one`
and `:models`, so [`chatrequest!`](@ref), [`respond`](@ref),
[`embeddingrequest!`](@ref) and the other platform verbs refuse it with an
`ArgumentError` before sending anything. A `Chat` or `Embeddings` naming it can
be built, with an explicit `model=`, but not sent ([Provider
Capabilities](@ref capabilities_api)).

## [Models and versions](@id jev_models)

[`list_models`](@ref) returns the names the account may put in `model`:

```@setup jev
using UniLM
```

```@example jev
m = list_models()
if m isa TypeSafeModelsSuccess
    for card in m.models
        println(card.name, "  ", card.release_date, "  ", card.description)
    end
else
    println("Request failed — ", m)
end
```

The listing holds **aliases**: `jev-latest`, the most recent stable release and
the default here, and `jev-preview`, the most recent release of any kind, ahead
of `jev-latest` when a preview build exists. A versioned id such as `jev-1.13.0`
is a valid `model` whether or not it is listed, and the listing is scoped to the
account ([Models](https://docs.typesafe.ai/models)).
[`SystemOneResponse`](@ref)`.model` reports the **versioned** id that answered:
log it, since an alias moving is otherwise invisible, and a threshold tuned
against one version is a claim about that version only.

## [Limits](@id jev_limits)

| Limit | Value |
| :--- | :--- |
| Choice options | 1–255 per question ([Choice](https://docs.typesafe.ai/primitives/choice)) |
| Score levels | 1–10 per question, TypeSafe advises at least two ([Score](https://docs.typesafe.ai/primitives/score)) |
| Context | 64k tokens per request; 32k for `state` plus the single longest question ([Models](https://docs.typesafe.ai/models)) |
| Rate limits | 250,000 tokens/second and 1,200 requests/minute, over either → 429; TypeSafe documents these as subject to change without notice ([Models](https://docs.typesafe.ai/models)) |

[`ask`](@ref) refuses a Choice of 256 options or a Score of 11 levels with an
`ArgumentError` naming TypeSafe's limit, before the round trip. The question
types themselves set no upper bound — other System One backends have other
limits, which their wrappers check the same way — so [`choice`](@ref) and
[`score`](@ref) build such questions; only sending one to TypeSafe is refused.

## [Errors, timeouts and retries](@id jev_errors)

A call that reaches the service returns a [`SystemOneSuccess`](@ref), a
[`SystemOneFailure`](@ref) (non-2xx) or a [`SystemOneCallError`](@ref) (no usable
response), and `issuccess` separates them; a wrong `service` or a malformed
request is an `ArgumentError` before anything is sent. A 200 whose answers do not
fit the request — a question unanswered, an answer of another kind, an option
that was not offered, probabilities that are not a distribution over the offered
outcomes, a value outside `0 … 1` — is a `SystemOneCallError` too, carrying the
request id: no success is built from a malformed answer. The service writes errors
in three `detail` shapes: `message` is extracted from whichever arrived, and
`error_type` is filled in only by the object shape.

| Status | What it means | `error_type` / `message` |
| :--- | :--- | :--- |
| 400 | Unknown model, or a request the schema accepted but the service rejected (a bare noul, more than 255 Choice options, more than 10 Score levels) | `"api_usage_error"` with `"Unknown model: …"`, or `nothing` with the plain-string reason |
| 401 | The API key was present but not valid | `"authentication_error"` |
| 403 | No `Authorization` header reached the service | `"authentication_error"` |
| 404 | Unknown path | `nothing`, message `"Not Found"` |
| 405 | Wrong method for the path | `nothing`, message `"Method Not Allowed"` |
| 422 | Schema validation failed | `nothing`; the message joins each validation entry as `"<field path>: <reason>"`, e.g. `questions.department.choice.criteria: Field required` |
| 429 | Rate limit — 250,000 tokens/second or 1,200 requests/minute at the time of writing | retried by the seam up to `max_attempts` honouring `Retry-After`; the final failure carries `retry_after`, the wait the service asked for in seconds |
| 500, 502, 503, 504, 520–524, 529 | Service-side failure, including the non-standard `529 Overloaded` and the origin errors 520–524 of a service behind Cloudflare | retried by the request seam, 520–524 like 502 and 504; a captured 529 used the object shape with `error_type = "system_overloaded"` and sent no retry header |

[`ask`](@ref) and [`list_models`](@ref) ride the shared retry seam, so
[`RequestConfig`](@ref) governs them as it governs a chat call: 408, 429, 500,
502, 503, 504, 520–524 and 529 are retried up to `max_attempts` within
`total_deadline`, honouring `Retry-After`; 400, 401, 403, 404 and 422 come back
as they are, because an identical retry cannot fix them.

```@example jev
state     = "Help! My payouts have been failing for 3 days."
questions = ["urgency" => score("How urgent is this ticket?",
                                ["Can wait", "Needs attention this week", "Needs attention today"])]

# Per call.
r = ask(state, questions; config = RequestConfig(request_timeout = 20.0, max_attempts = 5))

# Or for a whole scope, including tasks spawned inside it.
scoped = with_request_config(request_timeout = 20.0, max_attempts = 1) do
    ask(state, questions)
end

for res in (r, scoped)
    if res isa SystemOneSuccess
        println(answer(res, "urgency"))
    else
        println("Request failed — ", res)
    end
end
```

A timeout, a transport failure or a cancellation — through `cancel=` or the
ambient [`with_cancel`](@ref) token, with a [`UniLMCancelled`](@ref) as the
`cause` — is a `SystemOneCallError`, never a partial success ([Timeouts &
Retries](@ref timeouts_guide), [Concurrency, Tasks and Cancellation](@ref
concurrency_guide)).

## [Cost](@id jev_cost)

Only **input** tokens are billed; output tokens are currently free
([Models](https://docs.typesafe.ai/models)). Every question in one call shares
one ingestion of the `state`, so many questions in one [`ask`](@ref) cost far
less than one call per question.

```@example jev
r = ask("Help! My payouts have been failing for 3 days.",
        "urgency" => score("How urgent is this ticket?",
                           ["Can wait", "Needs attention this week", "Needs attention today"]))

println(answer(r, "urgency"))
u = token_usage(r)                     # prompt_tokens = input, completion_tokens = output
println(u.prompt_tokens, " billable tokens")
println("USD ", estimated_cost(r))     # priced against r.response.model
```

[`estimated_cost`](@ref) prices the **versioned** id in `r.response.model`, not
the alias the request named: a `jev-X.Y.Z` id without its own row at the
`jev-latest` row, and any other name that is not a key in
[`DEFAULT_PRICING`](@ref) at `0.0` ([Cost Tracking](@ref cost_guide)).

## Questions

```@docs
SystemOneQuestion
choice
score
noul
box
ChoiceQuestion
ScoreQuestion
NoulQuestion
NoulCriteria
BoxQuestion
RawQuestion
```

## Request & Verb

```@docs
SystemOneRequest
ask
with_system_one
```

## Answers

```@docs
SystemOneAnswer
ChoiceAnswer
ScoreAnswer
NoulAnswer
BoxCoordinates
BoxAnswer
RefusalAnswer
UnknownAnswer
```

## Result Types

```@docs
SystemOneResponse
SystemOneSuccess
SystemOneFailure
SystemOneCallError
SystemOneError
```

## Models

```@docs
list_models
TypeSafeModelCard
TypeSafeModelsSuccess
```

## Accessors

```@docs
answers
answer
```

`getindex`, `haskey` and `keys` also work on a [`SystemOneSuccess`](@ref) and a
[`SystemOneResponse`](@ref), so `result["urgency"]` is `answer(result,
"urgency")`. On a [`SystemOneFailure`](@ref) or [`SystemOneCallError`](@ref) all
of them throw [`SystemOneError`](@ref) instead of returning an empty map.

## [Provider protocol](@id system_one_protocol_api)

Every System One verb reaches its backend through a protocol, a Holy trait of
the endpoint: [`system_one_protocol`](@ref UniLM.system_one_protocol) maps an
endpoint to a [`SystemOneProtocol`](@ref UniLM.SystemOneProtocol), whose
[`encode_system_one`](@ref UniLM.encode_system_one) turns a request into a
[`SystemOneWire`](@ref UniLM.SystemOneWire) — refusing, before any I/O, what its
backend cannot carry — and whose [`decode_system_one`](@ref UniLM.decode_system_one)
turns the reply into a typed result, its answers checked against the request.
The rest is shared by every backend: the backend and model a call resolves
(through [`with_system_one`](@ref)), recording and replay, retries and
cancellation, and credentials, which `auth_header` supplies only when a request
really goes out, so a replay needs no key.

An endpoint that declares `:system_one` and no protocol of its own speaks
[`TypeSafeProtocol`](@ref UniLM.TypeSafeProtocol), TypeSafe's wire — a
TypeSafe-compatible gateway or proxy needs nothing more. A protocol other than
TypeSafe's files its recordings under a namespace of its own
(`"/<namespace>/v1/systemone"`), so two backends never replay each other's
answers, while every TypeSafe recording keeps its key.

```@docs
UniLM.SystemOneProtocol
UniLM.TypeSafeProtocol
UniLM.system_one_protocol
UniLM.encode_system_one
UniLM.decode_system_one
UniLM.SystemOneWire
```

## Recorded answers

A service does not answer a repeated request identically, so a test or a docs
build that calls it live cannot be reproduced. [`with_recorded_answers`](@ref)
records a real answer once and replays it offline. Inside its scope, `ask` and
`list_models` (and so `nl_dispatch`, `nl_classify` and `@branch`) and the non-streaming
[`chatrequest!`](@ref), [`respond`](@ref) and [`embeddingrequest!`](@ref) (and
so the tool loops) exchange through a directory of recordings keyed by the exact
request bytes, with any credential the body carries redacted; a streamed call and
every other verb reach the network as usual. In the default `:replay` mode no
recorded verb reaches the network and none needs a key,
and a request with no recording, or an unreadable one, throws
[`ReplayMissError`](@ref); `:record` and `:record_missing` call the service,
with its key, for the answers they write. A build of this documentation without
a recording flag replays every such example, System One and LLM alike.

```@docs
with_recorded_answers
ReplayMissError
RecordingWriteError
```

## [OpenAI Decisions](@id system_one_openai)

OpenAI's Decisions API (`POST /v1/decisions`, a public beta) answers predicate,
choice and score questions, and [`OPENAIServiceEndpoint`](@ref) speaks it: name
the endpoint as the `service` of [`ask`](@ref) or of a [`with_system_one`](@ref)
scope, and set `OPENAI_API_KEY`. The model is `gpt-6-luna`, the only one the API
serves, unless the call names another ([Decisions
guide](https://developers.openai.com/api/docs/guides/decisions)).

```julia
r = ask("Export fails in Safari but works in Chrome.",
        "severity" => score("How severe is this issue?",
                            ["Cosmetic", "Workaround available", "Fully blocked"]),
        "angry"    => noul("Is the customer angry?");
        service = OPENAIServiceEndpoint)

with_system_one(OPENAIServiceEndpoint) do
    nl_classify(ticket, INTENT)          # every verb inside asks OpenAI
end
```

| Generic surface | Decisions wire |
| :--- | :--- |
| `state`: a string, or a JSON value | `input`: the string, or the value's JSON text |
| `state`: an image, or a `Vector` of strings and images | one user message of `input_text` and `input_image` parts, in order; at most 128 images |
| [`noul`](@ref) | `predicate` |
| [`choice`](@ref) | `choice` of 2–255 options: the name is the `value`, a string description the `description` |
| [`score`](@ref) | `score` of 2–10 levels, each a string: its `label` |
| [`RawQuestion`](@ref) | sent verbatim, with the question's name added as `name` |
| keywords of [`ask`](@ref) | `prompt_cache_key` (at most 64 characters), `safety_identifier` (at most 128) |

Every question needs string `instructions`, and at most 200 go in one request.
What has no form on this wire — a Box question, a description or level that is
not a string, a Noul's `yes`/`no` criteria, an image outside the state, any other
keyword — is an `ArgumentError` before anything is sent. The answers come back
typed as on any backend, and a question the API declined comes back as a
[`RefusalAnswer`](@ref): [`nl_dispatch`](@ref), [`nl_classify`](@ref) and
[`@branch`](@ref) take it as a decline — the fallback, or
[`DecisionDeclinedError`](@ref). Probabilities and confidences arrive rounded to
two decimals, so a distribution over `n` outcomes is accepted when it sums to 1
within `0.005n`. A failure's `error_type` is the error's `code`, and the request
id is the `x-request-id` header. [`estimated_cost`](@ref) charges what the guide
prices: input tokens only, cached ones at the full input rate, twice that rate
for a request of more than 272K input tokens.

```@docs
UniLM.OpenAIDecisionsProtocol
```
