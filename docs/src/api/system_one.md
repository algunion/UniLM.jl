# [TypeSafe System One API (Jev)](@id system_one_api)

A System One model does not write text. It reads a piece of `state` and answers
the questions you enumerate with a probability distribution over the outcomes
you named, already typed — a `String` option, a `Float64` score, a probability —
so there is nothing to parse. The answer says *what*, the distribution says
*whether to act on it*. [Route and Decide](@ref system_one_guide) is the how-to;
this page starts with setup, models, limits, errors and cost, then documents
every type and verb.

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

[`choice`](@ref) with 256 options and [`score`](@ref) with 11 levels throw an
`ArgumentError` locally, before the round trip.

## [Errors, timeouts and retries](@id jev_errors)

A call that reaches the service returns a [`SystemOneSuccess`](@ref), a
[`SystemOneFailure`](@ref) (non-2xx) or a [`SystemOneCallError`](@ref) (no usable
response), and `issuccess` separates them; a wrong `service` or a malformed
request is an `ArgumentError` before anything is sent. The service writes errors
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
ChoiceQuestion
ScoreQuestion
NoulQuestion
NoulCriteria
```

## Request & Verb

```@docs
SystemOneRequest
ask
```

## Answers

```@docs
SystemOneAnswer
ChoiceAnswer
ScoreAnswer
NoulAnswer
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

## Natural-Language Control Flow

[`@branch`](@ref) is a `switch` whose cases are written in plain language. The
option names become the criteria of one [`choice`](@ref) question about the
state; the winning option selects which expression is evaluated and the other
bodies never run. However many options a branch lists, it costs a single
request.

[`nl_dispatch`](@ref) lifts the same idea into the method table. `nl"..."` is a
[`Meaning`](@ref) type, so a meaning is writable in an ordinary signature;
`nl_dispatch` sends one Choice question per `Meaning` position — again in a
single request — turns each answer back into a `Meaning` instance and calls the
function, so Julia's own dispatch selects the method and the remaining arguments
still dispatch on their types. Both constructs act on an answer through a
decision policy. The default gates on `confidence`: below `min_confidence`
`@branch` takes its `_` line and `nl_dispatch` calls `fallback`, and with
neither they raise [`LowConfidenceError`](@ref) rather than act on a near-tie.
`decide` replaces the gate with any function of the [`ChoiceAnswer`](@ref) that
returns an offered option — not only the winner — or `nothing` to decline,
which takes the same fallback or raises [`DecisionDeclinedError`](@ref).
[`meanings`](@ref)`(f)` is the union of the options of every natural-language
method of `f`; `meanings(f, Tuple{…})` lists the ones a call with ordinary
arguments of those types sends, in the order it sends them.

```@docs
Meaning
@nl_str
@branch
nl_dispatch
nl_classify
meanings
meaning_gaps
LowConfidenceError
DecisionDeclinedError
```

### Usage

```julia
using UniLM

ticket = "My package arrived crushed and the screen is cracked. I want my money back."

action = @branch ticket min_confidence=0.6 begin
    "the customer wants a refund"                                => :refund
    ("the customer reports a bug", "A defect in the software")   => :bug
    "the customer asks a pricing question"                       => :pricing
    _                                                            => :escalate
end                                                              # => :refund

# The same decision as multiple dispatch: `nl"..."` is a type, so the meanings
# live in the signatures and Julia selects the method once they are resolved.
route(::nl"the customer wants a refund", t)           = (:refund, t)
route(::nl"the customer reports a bug in the app", t) = (:bug, t)
route(::nl"the customer asks a pricing question", t)  = (:pricing, t)

meanings(route)                     # Dict(1 => [...the three descriptions...])
nl_dispatch(route, ticket)          # one request, then route(nl"..."(), ticket)
route(nl"the customer wants a refund"(), ticket)   # direct call, no request at all
```

A non-success call raises [`SystemOneError`](@ref) instead of resolving to a
branch, and a combination of offered meanings that no method covers is an
`ArgumentError` raised before any request — [`meaning_gaps`](@ref) lists such
gaps without a request.

## Recorded answers

A service does not answer a repeated request identically, so a test or a docs
build that calls it live cannot be reproduced. [`with_recorded_answers`](@ref)
records a real answer once and replays it offline. Inside its scope, `ask` and
`list_models` (and so `nl_dispatch` and `@branch`) and the non-streaming
[`chatrequest!`](@ref), [`respond`](@ref) and [`embeddingrequest!`](@ref) (and
so the tool loops) exchange through a directory of recordings keyed by the exact
request bytes; a streamed call and every other verb reach the network as usual.
In the default `:replay` mode nothing reaches the network and no key is needed,
and a request with no recording, or an unreadable one, throws
[`ReplayMissError`](@ref); `:record` and `:record_missing` call the service,
with its key, for the answers they write. A build of this documentation without
a recording flag replays every such example, System One and LLM alike.

```@docs
with_recorded_answers
ReplayMissError
RecordingWriteError
```
