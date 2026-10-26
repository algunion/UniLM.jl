# [Confidence, errors, cost & testing](@id jev_operations_page)

```@setup jev
using UniLM, JSON
```

## [Confidence and tuning](@id jev_confidence)

TypeSafe trains Jev for calibrated decisions: its probabilities are optimized against outcomes, and calibration
is measured across groups of answers, so it does not guarantee that any single answer is correct
([System One](https://docs.typesafe.ai/concepts/system-one)). TypeSafe computes the `confidence` of every
Choice and Score answer from its probabilities, 1 when all of it is on one outcome and 0 when it is spread
evenly, and a Noul answer carries none: its probability is the whole answer. TypeSafe's
[Confidence](https://docs.typesafe.ai/confidence) page gives the formulas of both; the Choice formula is also on
[Reading answers](@ref Reading-answers).

In the replies recorded for this manual, TypeSafe rounds every probability to two decimals, so a distribution
need not sum to exactly 1: `ask` accepts a sum within 0.005 per outcome of 1, and a chosen option at most 0.01
below the most probable one, and returns a reply further off as a [`SystemOneCallError`](@ref).

Answers vary between identical calls: a decision near a threshold can flip from one call to the next. Log what
a decision was based on — the answer's `raw`, `r.response.request_id` and `r.response.model` — and record the
answers a test relies on ([Testing and recordings](@ref jev_testing)).

!!! details "Evidence"
    A live reply over four options summed to 0.99 on 20 October 2026. Measured on jev-1.13.0 in September 2026,
    identical requests differed in 60% of repeated pairs, by 0.011 in a probability on average and by up to
    0.11.

Jev answers every question it is asked: TypeSafe's answer types hold no refusal
([System One](https://docs.typesafe.ai/api/system-one)), so a case is handed over only by a bar you set. Below
`min_confidence`, a natural-language verb returns what `fallback` returns, and with no `fallback` it throws
`LowConfidenceError`. A support desk's inbox, one team per message:

```@example jev
const TEAM = (billing   = "payments, charges, invoices or refunds",
              technical = "the app or the website does not work as expected",
              shipping  = "a parcel that is late, lost or arrived damaged",
              other     = "anything else")
const MESSAGES = [
    "I was charged twice for order #4471. Please refund the duplicate payment.",
    "The app crashes every time I open my order history. iPhone 15, latest version.",
    "My parcel was due last Monday and the tracking hasn't moved in six days.",
    "Do you ship to Norway, and how much does delivery cost?",
    "The phone I received has a cracked screen and the box was crushed. I want my money back.",
    "Your update deleted my saved addresses and now I can't check out. Third time this month!",
]

for message in MESSAGES
    println(rpad(nl_classify(message, TEAM), 11), message)
end
```

Every message gets a team. The question about shipping to Norway is the unclear one, and with a bar it goes to a
person:

```@example jev
seen = SystemOneSuccess[]
team = nl_classify(MESSAGES[4], TEAM; min_confidence = 0.6, fallback = message -> :person,
                   on_response = r -> push!(seen, r))
a = only(seen)["classify"]
(team, findfirst(==(a.choice), TEAM), a.confidence)
```

Jev's best guess, `billing`, came with a confidence below the 0.6 asked for. Set the bar by what a wrong team
costs against a person's time; where some mistakes cost more than others, price each one and let a `decide`
policy choose the cheapest action ([When mistakes have different prices](@ref jev_loss_matrix)). Tune a bar on
messages of your own, with the version pinned: a bar tuned against one version is a claim about that version
([Models](@ref jev_models)). The measurements below were made on Jev 1.13, and on 25 October 2026
`jev-latest` points at Jev 1.14. The decision rules on every provider are on
[Act on the answer](@ref jev_act_on_answer).

!!! details "Evidence"
    Measured on 20 October 2026 on our 129 labelled support tickets, one Choice of seven intents per ticket, one
    call at a time, on `jev-1.13.0`: accuracy 0.969, expected calibration error 0.016 (10 bins), Brier score
    0.052, mean top probability 0.96; 115 tickets at a confidence of 0.8 or more, of which 0.991 were right;
    latency 0.25 s at the median and 0.33 s at the 95th percentile. Limits: one labelled set, one run. In
    September 2026, also on jev-1.13.0, two other sets of ours measured higher errors: 0.047 on 100 items built
    on the model's documented failure modes, and 0.071 on 74 Noul answers, which were under-confident.

## [Cost](@id jev_cost)

TypeSafe bills the input tokens of a request — the state, the questions and the pictures — and output tokens
are free ([Models](https://docs.typesafe.ai/models), [Images](https://docs.typesafe.ai/images)). The state is
read once for every question of a request, so many questions in one `ask` cost far less than a request each.
The rows are TypeSafe's prices as checked on 25 October 2026; prices change, so check the Models page before
relying on an estimate.

```@eval
Main.ProviderFacts.price_table(:jev)
```

```@example jev
message = MESSAGES[1]
r = ask(message, "team" => choice("Which team should handle this message?", TEAM))
u = token_usage(r)
(u.prompt_tokens, u.completion_tokens, r.response.model, haskey(DEFAULT_PRICING, r.response.model),
 estimated_cost(r))
```

`prompt_tokens` is TypeSafe's `input_tokens` and `completion_tokens` its `output_tokens`.
[`estimated_cost`](@ref) prices the versioned id that answered, `r.response.model`, not the alias the request
named: here `jev-1.14.0`, which has a row of its own in the table above, so `haskey` is `true`. A `jev-X.Y.Z` id
with no row of its own, such as a release newer than the table, is priced at the `jev-latest` row, and any other
name with no row at `0.0`, with a warning. What a picture costs is on [Images](@ref jev_images);
totals across calls and prices of your own are on [Cost Tracking](@ref cost_guide).

## [Testing and recordings](@id jev_testing)

[`with_recorded_answers`](@ref) records a TypeSafe exchange once and replays it with no key and no network:
`ask`, `ask_each` and the verbs built on them, and TypeSafe's [`list_models`](@ref), the one model listing it
records. A recording is filed under the request's own path, `/v1/systemone` or `/v1/models`, with no namespace
in front of it, and keyed on the exact request: its method, its path and its body, which names the model.
`long_context = true` is part of the key too; the pictures of a request are keyed by the SHA-256 of their bytes,
and the file holds those digests, never the pictures.

```@example jev
recordings = mktempdir()      # in a package: a directory committed next to the tests
recorded = with_recorded_answers(recordings; mode = :record) do
    ask(message, "team" => choice("Which team should handle this message?", TEAM))
end
JSON.parsefile(only(readdir(recordings; join = true)))["request"]["path"]
```

```@example jev
replayed = withenv("TYPESAFE_API_KEY" => nothing) do
    with_recorded_answers(recordings) do
        ask(message, "team" => choice("Which team should handle this message?", TEAM))
    end
end
replayed["team"].probabilities == recorded["team"].probabilities
```

Recording takes `TYPESAFE_API_KEY`, and replaying takes nothing. In this manual's build the recording scope
above replays the manual's own recordings, so the example reaches no service either
([How this manual runs its examples](@ref jev_testing_manual)).

- **Pin the model.** A request's key holds the model it names. An unpinned request names the default model, an
  alias ([at the top of the start page](@ref jev_start)), so its recording keeps answering with the version
  it was recorded on after the alias moves; a pinned one misses once
  you move the pin, until it is recorded again. An unpinned request names `TYPESAFE_DEFAULT_MODEL` instead when
  that is set: record with the environment you replay with, or pin.
- **What a recording cannot hold.** To test what your code does with an answer the model rarely gives — a
  near-tie, a failed call, an option it was never offered — point `TYPESAFE_BASE_URL` at a local server that
  answers what you wrote. Run such tests outside `with_recorded_answers`: in a replay scope the request is
  replayed or misses, and never reaches the server, and a recording scope would file the canned answers as real
  ones. This block is
  not run when the manual is built:

```julia
using UniLM, HTTP, JSON, Test

struct Email
    subject::String
    body::String
end
route(::nl"the customer wants a refund", msg)           = :refunds
route(::nl"the customer reports a bug in the app", msg) = :engineering
route(::nl"an automatic out-of-office reply", ::Email)  = :drop
route(::nl"anything else", msg)                         = :frontline

canned = Ref{Any}()                  # (status, body) the server answers next
sent = String[]                      # every request body it received
server = HTTP.serve!("127.0.0.1", 8123; verbose = false) do req
    push!(sent, String(copy(req.body)))
    status, body = canned[]
    HTTP.Response(status, ["Content-Type" => "application/json"], JSON.json(body))
end

const OPTIONS = meanings(route, Tuple{String})[1]     # what a String message is offered

# A 200 in the service's shape: route's one question is "meaning_1" (its slot is unnamed).
# `option` gets probability p, the others share the rest, and confidence is (n·p − 1)/(n − 1).
function choosing(option, p)
    n = length(OPTIONS)
    probabilities = Dict(o => o == option ? p : (1 - p) / (n - 1) for o in OPTIONS)
    (200, (model = "jev-1.13.0", usage = (input_tokens = 96, output_tokens = 3),
           answers = (meaning_1 = (type = "choice", choice = option, probabilities,
                                   confidence = clamp((n * p - 1) / (n - 1), 0, 1)),)))
end

withenv("TYPESAFE_BASE_URL" => "http://127.0.0.1:8123", "TYPESAFE_API_KEY" => "test") do
    @testset "route: resolution edge cases" begin
        canned[] = choosing("the customer wants a refund", 0.97)
        @test nl_dispatch(route, "I want my money back.") === :refunds
        wire = JSON.parse(last(sent))                     # the request, in wire order
        @test wire["state"] == Dict("msg" => "I want my money back.")
        @test collect(keys(wire["questions"]["meaning_1"]["criteria"])) == OPTIONS

        canned[] = choosing("the customer wants a refund", 0.40)        # confidence 0.1
        @test nl_dispatch(route, "Hm."; min_confidence = 0.7, fallback = m -> :human) === :human
        @test_throws LowConfidenceError nl_dispatch(route, "Hm."; min_confidence = 0.7)

        canned[] = (503, (detail = "Service overloaded",))
        @test_throws SystemOneError nl_dispatch(route, "Hm."; config = RequestConfig(max_attempts = 1))

        canned[] = choosing("an automatic out-of-office reply", 0.97)   # never offered for a String
        @test_throws SystemOneError nl_dispatch(route, "Hm.")
    end
end
close(server)
```

The response shape is the whole contract: a `model`, an `answers` object keyed by the question names the client
sent, each a Choice with `choice`, `confidence` and `probabilities`, and a `usage` object. An answer outside the
offered options makes the call a `SystemOneCallError`, so `nl_dispatch` throws a `SystemOneError`, and no method
runs.

- **A pipeline with a language model.** A test that also calls a language model records each request with its
  own provider's key and replays with none ([Test a Jev + LLM pipeline offline](@ref jev_testing_pipeline)).

Previews of what a call sends, tests of the methods and of the judgment, and an audit trail are on
[Test and Develop](@ref jev_testing_guide).

## [Throughput](@id jev_throughput)

[`ask_each`](@ref) sends one request per item and returns the results in input order; on Jev it keeps at most 16
requests in flight unless its `ntasks` says otherwise. TypeSafe sets rate limits per organization, in tokens and
in requests per second, shows them in its developer console, and answers a request over either with 429
([Models](https://docs.typesafe.ai/models)); higher limits come with its custom and enterprise plans. How a
batch backs off when the service is busy is on [Many requests at once](@ref jev_items_each).

```@example jev
results = ask_each(MESSAGES, "team" => choice("Which team should handle this message?", collect(TEAM)))
[(findfirst(==(r["team"].choice), TEAM), r["team"].confidence) for r in results]
```

!!! details "Evidence"
    Measured live on 20 October 2026 (`jev-1.13.0`): 50 support tickets, three questions each, took 3.28 s
    through `ask_each` with its default of 16 in flight, and 13.21 s as 50 `ask` calls one after another. The
    batch used a single connection (HTTP/2).

## [When something fails](@id jev_errors)

A call that reaches TypeSafe returns a [`SystemOneSuccess`](@ref); a [`SystemOneFailure`](@ref) for a status
other than 200, with the `status`, TypeSafe's `message`, its `error_type` where TypeSafe sends one, and the
`request_id`; or a [`SystemOneCallError`](@ref) when no usable response came. A 200 whose answers do not fit the
request — a question unanswered, an option that was not offered, probabilities that are no distribution within
the rounding above — is a `SystemOneCallError` too: no success is built from a malformed answer. Reading an answer
of a failed call (`r["name"]`, [`answer`](@ref), [`answers`](@ref)) throws a [`SystemOneError`](@ref).

UniLM sends a request again after a 408, 429, 500, 502, 503, 504, 520 to 524 or 529, up to `max_attempts` times
within `total_deadline`, waiting at least what the reply's `retry-after-ms` header asks for, else its
`Retry-After`, as the shared backoff of [`ask_each`](@ref) does ([Many requests at once](@ref jev_items_each)). Each retry carries `X-TypeSafe-Retry-Count`, `1` on the second
attempt. Any other status comes back as it is, since an identical
request cannot fix it. With no [`RequestConfig`](@ref) of your own, a call to Jev is bounded by 60 s per attempt
and 150 s in all; one of your own replaces both, per call or for a scope:

```@example jev
state     = "Help! My payouts have been failing for 3 days."
questions = ["urgency" => score("How urgent is this ticket?",
                                ["Can wait", "Needs attention this week", "Needs attention today"])]

r = ask(state, questions; config = RequestConfig(request_timeout = 20.0, max_attempts = 5))
scoped = with_request_config(request_timeout = 20.0, max_attempts = 1) do
    ask(state, questions)
end
(r["urgency"].score, issuccess(scoped))
```

| What you see | Why | What to do |
| :--- | :--- | :--- |
| a `SystemOneCallError` whose `error` reads `TYPESAFE_API_KEY is not set` | no key in the environment; nothing was sent | set it ([Setup](@ref jev_setup)) |
| a `SystemOneFailure` with status 401 and `error_type` `"authentication_error"` | the key is not valid | a key from the TypeSafe console |
| status 400, `"Unknown model: …"` | a model name TypeSafe does not know | a name `list_models` returns, or a versioned id |
| status 400, `error_type` `"max_tokens_exceeded"`, and a message naming the limits, and `long_context` when the request did not set it | the request is longer than its context | shorten the state, split the questions, or `long_context = true` |
| status 400, `"Invalid request."`, and a hint naming `jev-1.14.0` and `jev-preview` | a Box question to Jev 1.13 | `model = "jev-1.14.0"` |
| status 413 | over a size limit UniLM cannot check before sending: all images as TypeSafe re-encodes them, the request with each reference replaced by its picture, or the references' total length ([Images](https://docs.typesafe.ai/images)) | fewer or smaller pictures, and a picture several questions need in the state once |
| status 422, a message of `"<field path>: <reason>"` entries | TypeSafe's schema refuses what UniLM's checks let through, such as a `RawQuestion` | the field the message names |
| status 429 after the retries, with `retry_after`, the wait TypeSafe asked for | over the organization's rate limits | fewer requests in flight (`ntasks`), or higher limits |
| status 529, or 408, 500, 502 to 504, 520 to 524, after the retries | TypeSafe overloaded, or a failure on its side | a larger `max_attempts` or `total_deadline` |
| status 404, `"Not Found"` | `TYPESAFE_BASE_URL` points at a server without TypeSafe's paths | the API root, with no path |
| a `SystemOneCallError` whose `cause` is a `UniLMTimeout` or a `UniLMCancelled` | a timeout, or a cancelled token | a `RequestConfig` of your own ([Timeouts & Retries](@ref timeouts_guide)) |
| a `SystemOneCallError` after a 200 | an answer that does not fit the request | keep `request_id` (the reply's `x-typesafe-request-id`) to quote in a support report |
| a `ReplayMissError`, thrown | in a replay scope, a request with no recording | record it with `TYPESAFE_API_KEY` ([Testing and recordings](@ref jev_testing)) |
| a `SystemOneError`, thrown | an answer read from a failed call | test `issuccess(r)` first |

!!! details "Evidence"
    The messages and error types in the table were captured from TypeSafe's live API: the 401, the unknown
    model's 400, a 422 and the 404 on 22 September 2026, the Box's 400 from Jev 1.13 in October 2026. TypeSafe's
    [Request rules and errors](https://docs.typesafe.ai/api/errors) page lists these statuses but the 404, and none
    of these messages.

Recorded on 28 September 2026 on `jev-1.13.0` with UniLM's development branch after 0.21.0 (after 0.20.1 for the
`RequestConfig` example), and on 25 October 2026 on `jev-1.14.0` with UniLM's development branch after 0.26.0,
through TypeSafe's API at `/v1`.
