# [TypeSafe System One API (Jev)](@id system_one_api)

A System One model does not write text. It reads a piece of `state` and answers
the questions you enumerate with a probability distribution over the outcomes
you named: which of these options, where on this rubric, how likely is this
statement. The answer arrives already typed — a `String` option, a `Float64`
score, a probability — so there is nothing to parse and nothing to re-prompt
when the shape comes back wrong. Every answer also carries the full
distribution, which is what makes confidence-gated routing possible: the answer
says *what*, the distribution says *whether to act on it*.

This complements a text model rather than replacing one. Jev classifies,
ranks, screens and routes; a generative model writes. Set `TYPESAFE_API_KEY` to
use it. The default model is `jev-latest`, an alias that moves when a new
version ships — [`SystemOneResponse`](@ref)`.model` reports the versioned id
that actually answered, and pinning `"jev-1.13.0"` keeps a tuned confidence
threshold meaningful across releases. `TYPESAFE_BASE_URL` overrides the API root
and `TYPESAFE_DEFAULT_MODEL` the default model name. The endpoint type is
[`TYPESAFEServiceEndpoint`](@ref UniLM.TYPESAFEServiceEndpoint), documented on
the [Service Endpoints](endpoints.md) page.

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

## Errors

A call that reached the service and came back non-2xx is a
[`SystemOneFailure`](@ref). The service uses three different `detail` body
shapes; `message` is extracted from whichever one arrived, and `error_type` is
filled in only by the object shape.

| Status | What it means | `error_type` / `message` |
| :--- | :--- | :--- |
| 400 | Unknown model, or a request the schema accepted but the service rejected (a bare noul, more than 255 Choice options, more than 10 Score levels) | `"api_usage_error"` with `"Unknown model: …"`, or `nothing` with the plain-string reason |
| 401 | The API key was present but not valid | `"authentication_error"` |
| 403 | No `Authorization` header reached the service | `"authentication_error"` |
| 404 | Unknown path | `nothing`, message `"Not Found"` |
| 405 | Wrong method for the path | `nothing`, message `"Method Not Allowed"` |
| 422 | Schema validation failed | `nothing`; the message joins each validation entry as `"<field path>: <reason>"`, e.g. `questions.department.choice.criteria: Field required` |
| 429 | Rate limit (tokens/second or requests/minute) | `nothing` unless the service sends the object shape; retried by the request seam |
| 500, 502, 503, 504, 529 | Service-side failure, including the non-standard `529 Overloaded` | retried by the request seam |

[`ask`](@ref) and [`list_models`](@ref) ride the package's shared retry seam, so
`RequestConfig.max_attempts` applies: 408, 429, 500, 502, 503, 504 and 529 are
retried within `total_deadline`, honouring `Retry-After`. The rest — 400, 401,
403, 404, 422 — come back as they are, because repeating an identical request
cannot fix them. A call that never produced a response at all (a timeout, a
transport failure, a missing key, or a 200 whose body was not a usable set of
answers) is a [`SystemOneCallError`](@ref).

Any call that reaches the service returns one of the three result types, and
`issuccess` separates them; a wrong `service` or a malformed request (duplicate
or blank question names, invalid criteria) is an `ArgumentError` raised before
any request is sent.

## Usage

```julia
using UniLM

result = ask(
    "Help! My payouts have been failing for 3 days and nobody has replied to my emails.",
    "department" => choice("Which team should handle this ticket?", (
        billing   = "Payments, invoicing, payouts, refunds",
        technical = "Bugs, outages, integrations",
        sales     = "Pricing, upgrades, new accounts")),
    "urgency" => score("How urgent is this ticket?",
        ["Can wait", "Needs attention this week", "Needs attention today"]),
    "is_frustrated" => noul("Is the customer frustrated?";
        yes = "The customer expresses frustration or impatience",
        no  = "The customer is neutral or satisfied"))

if issuccess(result)
    println(result["department"].choice)              # "billing"
    println(result["department"].probabilities)       # every option, not just the winner
    println(result["urgency"].score)                  # 1.98 on the 0..2 rubric
    println(result["is_frustrated"].noul)             # 0.98
    println(result.response.model)                    # "jev-1.13.0" — the version that answered
    println(estimated_cost(result))                   # only input tokens are billed

    # Confidence is the second axis: act when the distribution is concentrated,
    # hand the rest to a human or a larger model.
    d = result["department"]
    d.confidence >= 0.8 ? route(d.choice) : escalate()
end

# The models the account may name in `model`.
models = list_models()
issuccess(models) && println([m.name for m in models.models])
```

All questions in one call share one ingestion of the `state`, so batching many
small questions into a single [`ask`](@ref) costs far less than one call per
question — and speculative questions you may not use are cheap to include.

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
still dispatch on their types. Both constructs gate on the answer's
`confidence`: below the threshold `@branch` takes its `_` line and `nl_dispatch`
calls `fallback`, and with neither they raise
[`LowConfidenceError`](@ref) rather than act on a near-tie.
[`meanings`](@ref) previews the options exactly as they will be sent.

```@docs
Meaning
@nl_str
@branch
nl_dispatch
meanings
LowConfidenceError
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
branch, and a combination of meanings that no method covers raises Julia's own
`MethodError` — a gap in the method table is not a service failure and is not
swallowed.
