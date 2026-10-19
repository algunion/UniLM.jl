# [System One API (Jev, OpenAI Decisions, Ollama)](@id system_one_api)

A System One model does not write text. It reads a piece of `state` and answers
the questions you enumerate with a probability distribution over the outcomes
you named, already typed — a `String` option, a `Float64` score, a probability,
a box — so there is nothing to parse. The answer says *what*, the distribution
says *whether to act on it*.

UniLM has one System One surface — [`ask`](@ref), [`ask_each`](@ref), the
question builders and the typed answers below — over three backends: TypeSafe's
Jev (the default), OpenAI's Decisions API (a public beta) and local decision
models served by Ollama. [Route and Decide](@ref system_one_guide) is the how-to
and [Backends: Jev, OpenAI and Local Models](@ref system_one_backends) compares
the three. This page gives what every backend shares, then each backend's setup,
rules, errors and cost, then every type and verb; [`nl_classify`](@ref),
[`nl_dispatch`](@ref) and [`@branch`](@ref) have a page of their own,
[Natural-Language Dispatch](@ref nl_dispatch_api).

## [Backends](@id system_one_backend_api)

| Backend | `service` | Model when a call names none | Credentials | Reference |
| :--- | :--- | :--- | :--- | :--- |
| TypeSafe's Jev (the default) | `TYPESAFEServiceEndpoint` | `TYPESAFE_DEFAULT_MODEL`, else `jev-latest` | `TYPESAFE_API_KEY` | [TypeSafe's Jev](@ref system_one_typesafe) |
| OpenAI's Decisions API (public beta) | `OPENAIServiceEndpoint` | `gpt-6-luna` | `OPENAI_API_KEY` | [OpenAI Decisions](@ref system_one_openai) |
| A decision model on Ollama | `OllamaEndpoint()` | none: name an installed one, e.g. `tev1:0.8b` | none | [Ollama](@ref system_one_ollama) |

A call takes its backend from `service = …`; without one, from the enclosing
[`with_system_one`](@ref) scope, which can also set the model of the calls that
name none; outside every scope, from TypeSafe. [`ask`](@ref),
[`ask_each`](@ref), [`list_models`](@ref) and the natural-language verbs resolve
it alike. Any other endpoint that declares `:system_one` is a backend too, and
one without a protocol of its own speaks TypeSafe's wire ([Provider
protocol](@ref system_one_protocol_api)).

The question types hold only what every backend requires. What a backend's wire
cannot carry — more options or levels than it takes, a question kind, an image
or an option it does not know — its wrapper refuses before anything is sent,
with an `ArgumentError` naming the backend, the rule and the fix: [`choice`](@ref)
builds a Choice of 256 options, and only sending it to a backend that takes
fewer is refused. Every reply is checked against its request alike on every
backend (see [`ask`](@ref)), with the backend's own allowance for how it rounds
probabilities; a reply that does not fit is a [`SystemOneCallError`](@ref).

`confidence` is one measure on every backend — `(n·p_max − 1)/(n − 1)` for a
Choice over `n` options, `1 − Σᵢ pᵢ·|i − m| / MAD_unif` for a Score whose most
likely level is `m` ([Confidence](https://docs.typesafe.ai/confidence)): TypeSafe
and OpenAI report it, and the Ollama wrapper computes it from the probabilities;
over a single option it is 1.0, as TypeSafe answers a one-option Choice. How far
a confidence can be trusted is a property of the model, not of the formula.

!!! warning "Tune `min_confidence` per backend"
    A threshold tuned on one backend does not carry over to another. On the same
    129 labelled support tickets with 7 options (2026-10-20), Jev (`jev-1.13.0`)
    chose the right option for 96.9% of them, with an expected calibration error
    of 0.016, and OpenAI's `gpt-6-luna` for 94.6%, with 0.080 — in every bin
    above 0.5 its accuracy was at least its mean top probability: it is
    under-confident. Tune a threshold on the backend, and the model version, it
    will run on ([Backends: Jev, OpenAI and Local Models](@ref
    system_one_backends)).

While you set no [`RequestConfig`](@ref) of your own — no `config`, no
[`with_request_config`](@ref) scope, no [`set_default_config!`](@ref) — a backend
may bound its calls itself: TypeSafe's wire and OpenAI's by 60 s per attempt and
150 s in all, while Ollama brings no bound of its own, so the field defaults
apply (loading a model on a CPU can take minutes). [`ask_each`](@ref) keeps at
most 16 requests in flight on TypeSafe's wire and OpenAI's, 1 on Ollama and 8 on
a protocol of your own, unless its `ntasks` says otherwise. Every System One
exchange goes through one connection pool per process, which keeps up to 256
idle connections per host for 30 s, so a batch or a loop finds its connections
open.

[`with_recorded_answers`](@ref) files each backend's exchanges apart: TypeSafe's
under the request's path (`/v1/systemone`), OpenAI's under
`/openai-decisions/v1/decisions`, Ollama's under `/ollama/v1/systemone`. A
recording never replays for another backend; an endpoint that speaks TypeSafe's
wire shares TypeSafe's recordings, since its requests are TypeSafe's.

## [TypeSafe's Jev](@id system_one_typesafe)

[TypeSafe](https://docs.typesafe.ai)'s System One model, Jev, answers through
[`TYPESAFEServiceEndpoint`](@ref UniLM.TYPESAFEServiceEndpoint) — the backend of
every call that names none outside a [`with_system_one`](@ref) scope — on
TypeSafe's wire, [`TypeSafeProtocol`](@ref UniLM.TypeSafeProtocol).

### [Setup](@id jev_setup)

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

### [Models and versions](@id jev_models)

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

In October 2026 `jev-latest` answers as `jev-1.13.0` and `jev-preview` as
`jev-1.14.0` (Jev 1.14, available from October 14, 2026). Both read images; a
[`box`](@ref) question needs Jev 1.14 or later — `model = "jev-1.14.0"` or
`"jev-preview"`. Jev 1.13 answers a request with one with HTTP 400 and the bare
message `"Invalid request."`, which [`ask`](@ref) returns with a hint naming those
models. `long_context = true` works on both ([Limits](@ref jev_limits)).

### [Limits](@id jev_limits)

| Limit | Value |
| :--- | :--- |
| Choice options | 1–255 per question ([Choice](https://docs.typesafe.ai/primitives/choice)) |
| Score levels | 1–10 per question, TypeSafe advises at least two ([Score](https://docs.typesafe.ai/primitives/score)) |
| Context | 64k tokens per request; 32k for `state` plus the single longest question; 262k for both with `long_context = true` ([Models](https://docs.typesafe.ai/models)) |
| Images | PNG or JPEG, still (no animated PNG); at most 16 MiB a file; 280 to 1568 pixels on each side ([Images](https://docs.typesafe.ai/images)) |
| Request body | 32 MiB (33,554,432 bytes), multipart framing included ([API](https://docs.typesafe.ai/api)) |
| Nesting | 64 levels of objects and arrays in each of `state`, `instructions` and `criteria` ([API](https://docs.typesafe.ai/api)) |
| Rate limits | set per organization, in tokens per second and requests per second, and shown in TypeSafe's developer console; a request over either → 429 ([Models](https://docs.typesafe.ai/models)) |

[`ask`](@ref) checks every limit but the context and the rate limits before the
round trip, and refuses a request over one with an `ArgumentError` naming
TypeSafe's limit and the value — a Choice of 256 options, a Score of 11 levels, a
GIF, an image 2000 pixels wide, a file cut short, a state nested 65 levels deep.
The question types themselves set no upper bound — other System One backends have
other limits, which their wrappers check the same way — so [`choice`](@ref) and
[`score`](@ref) build such questions; only sending one to TypeSafe is refused. A
[`box`](@ref) question needs an image in the request to locate in, so one in a
request without an image is refused as well: TypeSafe would answer it with a box
at a confidence near 0.05 that means nothing. A Box answer turned inside out
(`top > bottom` or `left > right`) is a [`SystemOneCallError`](@ref).

An [`ImageAttachment`](@ref) may stand anywhere in the `state`, the instructions
or the criteria — a Choice whose options are example pictures, a Score whose
levels are. TypeSafe marks images and Box questions experimental: their request
format, answers and limits may change between releases
([Images](https://docs.typesafe.ai/images)). Each distinct image is sent once, as
a part of a `multipart/form-data` request, and every place it stood holds its
reference, `<image:…>` with the first 32 hex digits of the SHA-256 of its bytes;
every use is billed, so an image several questions need belongs in the `state`.
A request with an image or a Box question carries the header
`TypeSafe-Danger-Zone-Images: true` TypeSafe requires: passing one is the opt-in.
A Score level that is an image comes back in the answer's `legend` as that same
`ImageAttachment`. Images are numbered in an error message as they first occur:
the `state`, then each question's instructions and criteria. TypeSafe applies no
EXIF orientation, so rotate a photo's pixels upright before sending it. A
recording of a request with images ([`with_recorded_answers`](@ref)) is keyed on
its JSON part and the SHA-256 of each image, and holds no image.

`long_context = true` — `ask(state, questions...; long_context = true)` — sends
`TypeSafe-Danger-Zone-Long-Context: true`, which raises both context limits to
262k tokens at the same price per token; TypeSafe documents it as slower and less
accurate than the default limits, and the answer does not say which ran. A
request over its context comes back as a 400 whose `error_type` is
`"max_tokens_exceeded"`, and its `message` names `long_context` and the limits.

### [Errors, timeouts and retries](@id jev_errors)

A call that reaches the service returns a [`SystemOneSuccess`](@ref), a
[`SystemOneFailure`](@ref) (non-2xx) or a [`SystemOneCallError`](@ref) (no usable
response), and `issuccess` separates them; a wrong `service` or a malformed
request is an `ArgumentError` before anything is sent. A 200 whose answers do not
fit the request — a question unanswered, an answer of another kind, an option
that was not offered, probabilities that are not a distribution over the offered
outcomes, a value outside `0 … 1` — is a `SystemOneCallError` too, carrying the
request id: no success is built from a malformed answer. TypeSafe's probabilities
are accepted when they sum to 1 within 0.005 per outcome, and the chosen option
when it lies at most 0.01 below the most probable: one live reply, a Choice of
four options, summed to 0.99 — the mark of values rounded to two decimals. The
service writes errors in three `detail` shapes: `message` is extracted from
whichever arrived, and `error_type` is filled in only by the object shape.

| Status | What it means | `error_type` / `message` |
| :--- | :--- | :--- |
| 400 | Unknown model, or a request the schema accepted but the service rejected: a bare noul; a Box question on a Jev before 1.14; an image it cannot read or whose size is out of range; nesting deeper than 64 levels | `"api_usage_error"` with `"Unknown model: …"`, `"Invalid request."` (on a Box question, with a hint naming `jev-1.14.0` and `jev-preview`), `"Invalid image request."` or `"Invalid image dimensions."`; or `nothing` with the plain-string reason |
| 400 | The request is longer than its context | `"max_tokens_exceeded"`, and no message from the service: the `message` is that type with a hint naming `long_context = true` and the limits |
| 401 | The API key was present but not valid | `"authentication_error"` |
| 403 | No `Authorization` header reached the service | `"authentication_error"` |
| 404 | Unknown path | `nothing`, message `"Not Found"` |
| 405 | Wrong method for the path | `nothing`, message `"Method Not Allowed"` |
| 413 | Over a size limit: an image file over 16 MiB, a body over 32 MiB (after decompression too), references too long | `"api_usage_error"` with `"Request exceeds size limits."` or `"Request body of at least … bytes exceeds the hard maximum of 33554432 bytes."` |
| 415 | An unsupported `Content-Encoding` — never one [`ask`](@ref) sends, which compresses nothing; a proxy's | `"api_usage_error"` with `"Unsupported Content-Encoding '…'; supported: gzip, zstd."` |
| 422 | Schema validation failed, including more than 255 Choice options or 10 Score levels | `nothing`; the message joins each validation entry as `"<field path>: <reason>"`, e.g. `questions.department.choice.criteria: Field required` or `questions.pick.choice.criteria: Dictionary should have at most 255 items after validation, not 256` |
| 429 | Over the organization's rate limit, in tokens or requests per second (shown in TypeSafe's developer console) | `"api_usage_error"` with `"Rate limit exceeded. Please retry shortly."`, sent live (2026-10-19) with both `retry-after` and `retry-after-ms` headers; retried by the seam up to `max_attempts` honouring `Retry-After`; the final failure carries `retry_after`, the wait the service asked for in seconds |
| 500, 502, 503, 504, 520–524, 529 | Service-side failure, including the non-standard `529 Overloaded` and the origin errors 520–524 of a service behind Cloudflare | retried by the request seam, 520–524 like 502 and 504; a captured 529 used the object shape with `error_type = "system_overloaded"` and sent no retry header |

[`ask`](@ref) and [`list_models`](@ref) ride the shared retry seam, so
[`RequestConfig`](@ref) governs them as it governs a chat call: 408, 429, 500,
502, 503, 504, 520–524 and 529 are retried up to `max_attempts` within
`total_deadline`, honouring `Retry-After`; 400, 401, 403, 404, 413, 415 and 422
come back as they are, because an identical retry cannot fix them. While you set
no `RequestConfig` of your own, a call is bounded by 60 s per attempt and 150 s
in all: TypeSafe answers a typical request in about 0.3 s, and the slowest call
measured — 262k tokens of long context — took 5.3 s. A retry carries
`X-TypeSafe-Retry-Count`, `1` on the second attempt, as TypeSafe's official SDKs
send it.

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

### [Cost](@id jev_cost)

Only **input** tokens are billed; output tokens are currently free
([Models](https://docs.typesafe.ai/models)). Every question in one call shares
one ingestion of the `state`, so many questions in one [`ask`](@ref) cost far
less than one call per question. An image is billed on every use: on 2026-10-19
the input tokens of a W×H image fit 2·⌈W/28⌉·⌈H/28⌉, plus about 4 per use — an
observation, not a published rate.

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

## [OpenAI Decisions](@id system_one_openai)

OpenAI's Decisions API (`POST /v1/decisions`, a public beta since October 6,
2026) answers predicate, choice and score questions, and
[`OPENAIServiceEndpoint`](@ref) speaks it: name the endpoint as the `service` of
[`ask`](@ref) or of a [`with_system_one`](@ref) scope, and set `OPENAI_API_KEY`.
The model is `gpt-6-luna`, the only one the API serves, unless the call names
another ([Decisions guide](https://developers.openai.com/api/docs/guides/decisions)).

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
within `0.005n`, and the chosen option when it lies at most 0.01 below the most
probable. A failure's `error_type` is the error's `code`, and the request id is
the `x-request-id` header. [`estimated_cost`](@ref) charges what the guide
prices: input tokens only — \$0.10 per million, the `gpt-6-luna` row's input rate —
cached ones at the full input rate, twice that rate for a request of more than
272K input tokens.

The calls are bounded as TypeSafe's are (60 s per attempt, 150 s in all, while
you set no `RequestConfig` of your own), [`ask_each`](@ref) keeps 16 of them in
flight, and [`list_models`](@ref) has no listing here: it is an `ArgumentError`.
On the 129 support tickets of the [Backends](@ref system_one_backend_api) section,
OpenAI answered in 0.48 s at the median and 2.44 s at the 95th percentile (Jev:
0.25 s and 0.33 s), at \$0.0000296 a ticket (Jev: \$0.0000225).

```@docs
UniLM.OpenAIDecisionsProtocol
```

## [Ollama](@id system_one_ollama)

An [`OllamaEndpoint`](@ref) answers System One requests from a local decision
model — Tev1, Nimble, Kev and the others Ollama lists as [decision
models](https://ollama.com/search?c=decision) — through Ollama's
TypeSafe-compatible route `POST /v1/systemone` (Ollama 0.35 or later). The
verbs, the questions and the result types are the ones below; no key is sent and
nothing is billed. Start the server and pull a decision model first, such as
`ollama pull tev1:0.8b`.

```julia
ollama = OllamaEndpoint()                     # OLLAMA_HOST, else http://127.0.0.1:11434

r = ask("Help! My payouts have been failing for 3 days.",
        "department" => choice("Which team should handle this ticket?",
                               (billing = "Payments, invoicing, payouts, refunds",
                                technical = "Bugs, outages, integrations")),
        "is_frustrated" => noul("Is the customer frustrated?");
        service = ollama, model = "tev1:0.8b")

INTENT = (refund = "the customer wants a refund", bug = "the customer reports a software bug")
with_system_one(ollama; model = "kev:0.8b-q8_0") do     # every verb inside asks Kev
    nl_classify("I was charged twice. Please give me my money back.", INTENT)   # :refund
end
```

- **Model.** There is no default: name an installed decision model with
  `model = …`, or set one for a scope with [`with_system_one`](@ref); a call
  without one is an `ArgumentError` before anything is sent. A model that is not
  installed comes back as a 404 [`SystemOneFailure`](@ref) whose message says to
  `ollama pull` it, and a server that does not answer as a
  [`SystemOneCallError`](@ref) that says how to start one.
- **Refused before any request.** Ollama's decision encodings are stricter than
  TypeSafe's wire; [`ask`](@ref) checks their rules first and throws an
  `ArgumentError` naming Ollama and the rule.

| Rule | Ollama |
| :--- | :--- |
| Questions per request | 1–64 |
| `instructions` | required on every Choice, Score and Noul question: a string that is not blank, an object or an array |
| Choice options | 2–26, each described by a string or `nothing` |
| Score levels | 2–10, each a string |
| Noul criteria | strings |
| `state` | not a blank string |
| Box questions | refused: the route answers Choice, Score and Noul |
| Images | not sent yet: no vision decision model has been verified with this wrapper |
| Options | `keep_alive` only |

- **A model's own limits.** A model may cap lower than the route, and its 400
  says so: `tev1:0.8b` takes at most 24 options (Ollama 0.40.6). A
  [`RawQuestion`](@ref) is sent verbatim, for the server to judge.
- **`keep_alive`.** `ask(...; keep_alive = 600)` keeps the model loaded for 600
  seconds after the request, `0` unloads it at once and `Inf` keeps it loaded;
  without it the endpoint's own `keep_alive` applies, else the server's default.
  The endpoint's other settings — `truncate`, `shift` and the
  [`OllamaOptions`](@ref) such as `num_ctx` — have no field on this route and
  are not sent.
- **Confidence.** `confidence` is TypeSafe's published measure, computed from the
  returned probabilities — `(n·p_max − 1)/(n − 1)` for a Choice and
  `1 − Σᵢ pᵢ·|i − m| / MAD_unif`, floored at 0, for a Score
  ([Confidence](https://docs.typesafe.ai/confidence)) — so a threshold means the
  same on every backend. Ollama's generic decision models report
  `1 − H(p)/ln n` instead (Kev models report TypeSafe's); the server's value stays
  in each answer's `raw`. Probabilities are accepted when they sum to 1 within
  0.00005 per outcome — Kev models round each to four decimals on Ollama's own
  runner — and the chosen option when it lies at most 0.0001 below the most
  probable.
- **Speed.** Measured live on 2026-10-20 (Apple M4 Max, Ollama 0.40.6): a warm
  request took about 17 ms on `kev:0.8b-q8_0` and 25–130 ms on `tev1:0.8b`, and
  the first call, which loads the model, 0.9–2.6 s. The server's decision runners
  answer one request at a time, so [`ask_each`](@ref) keeps one in flight unless
  its `ntasks` says otherwise.
- **Results.** `request_id` is `nothing`, as Ollama sends none; `usage` counts
  `prompt_eval_cached_count` as cached tokens; [`estimated_cost`](@ref) is `0.0`.
  [`with_recorded_answers`](@ref) files the exchanges under
  `/ollama/v1/systemone`, apart from TypeSafe's.
- **Known upstream issues**, open at Ollama 0.40.6: a model may score a request's
  questions one after another, so its latency grows with their number
  ([#18750](https://github.com/ollama/ollama/issues/18750)); one over-long request
  left a llama.cpp runner answering later ones with HTTP 200 and wrong
  probabilities until the model was unloaded
  ([#19090](https://github.com/ollama/ollama/issues/19090)); a vision decision
  runner returned repeated HTTP 500s after many sequential requests
  ([#19062](https://github.com/ollama/ollama/issues/19062)). `keep_alive = 0`
  unloads a model whose answers stop making sense.

```@docs
UniLM.OllamaSystemOneProtocol
```

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

## Request & Verbs

```@docs
SystemOneRequest
ask
ask_each
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
cancellation, the connection pool, and credentials, which `auth_header` supplies
only when a request really goes out, so a replay needs no key. Three protocols
ship: [`TypeSafeProtocol`](@ref UniLM.TypeSafeProtocol) above, and
[`OpenAIDecisionsProtocol`](@ref UniLM.OpenAIDecisionsProtocol) and
[`OllamaSystemOneProtocol`](@ref UniLM.OllamaSystemOneProtocol) in their
backends' sections.

An endpoint that declares `:system_one` and no protocol of its own speaks
[`TypeSafeProtocol`](@ref UniLM.TypeSafeProtocol), TypeSafe's wire — a
TypeSafe-compatible gateway or proxy needs no protocol code ([Service
Endpoints](@ref extension_api) shows one). A protocol other than TypeSafe's
files its recordings under a namespace of its own, before the request's path
(`/ollama/v1/systemone`), so two backends never replay each other's answers,
while every TypeSafe recording keeps its key.

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
[`ReplayMissError`](@ref); `:record` and `:record_missing` call the service —
with its key, or its local server running — for the answers they write. A build
of this documentation without a recording flag replays every such example,
System One and LLM alike.

```@docs
with_recorded_answers
ReplayMissError
RecordingWriteError
```
