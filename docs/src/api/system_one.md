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

| Backend | `service` | Model when a call names none | Credentials |
| :--- | :--- | :--- | :--- |
| [TypeSafe's Jev](@ref system_one_typesafe) (the default) | `TYPESAFEServiceEndpoint` | `TYPESAFE_DEFAULT_MODEL`, else `jev-latest` | `TYPESAFE_API_KEY` |
| [OpenAI's Decisions API](@ref system_one_openai) (public beta) | `OPENAIServiceEndpoint` | `gpt-6-luna` | `OPENAI_API_KEY` |
| [A decision model on Ollama](@ref system_one_ollama) | `OllamaEndpoint()` | none: name an installed one, e.g. `tev1:4b` (`clef-flash:9b` for pictures) | none |

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

!!! warning "Tune the confidence threshold per backend"
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
wire shares TypeSafe's recordings, since its requests are TypeSafe's. An option
that changes how a request is served but never what it is answered — OpenAI's
`prompt_cache_key`, `safety_identifier` and `prompt_cache_options`, Ollama's `keep_alive` — is sent and
left out of the recording, key and file alike, so the same question replays
whatever it is; TypeSafe's `long_context` is part of the key.

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
Endpoints](@ref endpoints_api)) is not a chat backend: it declares only `:system_one`
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
GIF, an image 2000 pixels wide, a file cut short within its header, a JPEG that
does not end with its end-of-image marker (`FF D9`: cut short, or carrying bytes
after the image), a state nested 65 levels deep.
The question types themselves set no upper bound — other System One backends have
other limits, which their wrappers check the same way — so [`choice`](@ref) and
[`score`](@ref) build such questions; only sending one to TypeSafe is refused. A
[`box`](@ref) question needs an image in the request to locate in, so one in a
request without an image is refused as well: TypeSafe answers it anyway, with
HTTP 200 and a box at a low confidence — 0.05 to 0.10 in probes on 2026-10-19
and 2026-10-20 — that locates nothing. A Box answer turned inside out
(`top > bottom` or `left > right`) is a [`SystemOneCallError`](@ref).

An [`ImageAttachment`](@ref) may stand anywhere in the `state`, the instructions
or the criteria — a Choice whose options are example pictures, a Score whose
levels are. TypeSafe marks images and Box questions "Dreamer-ready", not yet ready for
production: their request format, answers and limits "can change between releases"
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
[`SystemOneFailure`](@ref) (a status other than 200) or a
[`SystemOneCallError`](@ref) (no usable response), and `issuccess` separates
them; a wrong `service` or a malformed request is an `ArgumentError` before
anything is sent. A 200 whose answers do not fit the request — a question
unanswered, an answer of another kind, an option that was not offered,
probabilities that are not a distribution over the offered outcomes, a value
outside `0 … 1`, a Score whose `score` its probabilities contradict or whose
`legend` renames a level asked as a string, no model named — is a
`SystemOneCallError` too, carrying the request id: no success is built from a
malformed answer. TypeSafe's probabilities
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
| 429 | Over the organization's rate limit, in tokens or requests per second (shown in TypeSafe's developer console) | `"api_usage_error"` with `"Rate limit exceeded. Please retry shortly."`, sent live (2026-10-19) with both `retry-after` and `retry-after-ms` headers; retried by the seam up to `max_attempts` honouring `retry-after-ms`, else `Retry-After`; the final failure carries `retry_after`, the wait the service asked for in seconds |
| 500, 502, 503, 504, 520–524, 529 | Service-side failure, including the non-standard `529 Overloaded` and the origin errors 520–524 of a service behind Cloudflare | retried by the request seam, 520–524 like 502 and 504; a captured 529 used the object shape with `error_type = "system_overloaded"` and sent no retry header |

[`ask`](@ref) and [`list_models`](@ref) ride the shared retry seam, so
[`RequestConfig`](@ref) governs them as it governs a chat call: 408, 429, 500,
502, 503, 504, 520–524 and 529 are retried up to `max_attempts` within
`total_deadline`, honouring `retry-after-ms`, else `Retry-After`; 400, 401, 403, 404, 413, 415 and 422
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

OpenAI's Decisions API (`POST /v1/decisions`, a public beta since
October 6, 2026) answers predicate, choice and score questions, and
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
| keywords of [`ask`](@ref) | `prompt_cache_key` (at most 64 characters), `safety_identifier` (at most 128), `prompt_cache_options` (a [`PromptCacheOptions`](@ref) of `mode` and `ttl`) |

Every question needs string `instructions`, and at most 200 go in one request.
`prompt_cache_options` takes a `ttl` of `"30m"` or `"12h"` — OpenAI's prompt-caching
guide names `"12h"` for the Responses API, and the Decisions API answered a `"12h"`
request with HTTP 200 — and a `mode`; with `mode="explicit"` the request marks no
breakpoint, so OpenAI caches none of it. What has no form on this wire — a Box
question, a description or level that is not a string, a Noul's `yes`/`no`
criteria, an image outside the state, another `ttl`, `prewarm` or
`comparison_response_id`, any other keyword — is an `ArgumentError` before
anything is sent. The answers come back
typed as on any backend, and a question the API declined comes back as a
[`RefusalAnswer`](@ref): [`nl_dispatch`](@ref), [`nl_classify`](@ref) and
[`@branch`](@ref) take it as a decline — the fallback, or
[`DecisionDeclinedError`](@ref). Probabilities and confidences arrive rounded to
two decimals, so a distribution over `n` outcomes is accepted when it sums to 1
within `0.005n`, and the chosen option when it lies at most 0.01 below the most
probable. `usage` counts input, cached, cache-write (the 12-hour ones apart) and
output tokens, and holds the service tier when the reply names one; a count that is not a
non-negative whole number, or a `…_tokens_details` value that is not an object,
reads as 0 with a warning. A failure's `error_type` is the error's `code`, and
the request id is the `x-request-id` header. A recording
([`with_recorded_answers`](@ref)) is keyed without the three options, and holds
each image as the SHA-256 of its base64 text, `data:<mime>;sha256,<hex>`, never
its bytes. [`estimated_cost`](@ref) charges what the guide
prices: input tokens only — \$0.10 per million, the `gpt-6-luna` row's input rate —
cached and cache-written ones at the full input rate, twice that rate for a
request of more than 272K input tokens.

The calls are bounded as TypeSafe's are (60 s per attempt, 150 s in all, while
you set no `RequestConfig` of your own), and [`ask_each`](@ref) keeps 16 of them in
flight. [`list_models`](@ref) on this endpoint lists every model OpenAI serves the key
([Model Listings](@ref models_api)).
On the 129 support tickets of the [Backends](@ref system_one_backend_api) section,
OpenAI answered in 0.48 s at the median and 2.44 s at the 95th percentile (Jev:
0.25 s and 0.33 s), at \$0.0000296 a ticket (Jev: \$0.0000225).

```@docs
UniLM.OpenAIDecisionsProtocol
```

## [Ollama](@id system_one_ollama)

An [`OllamaEndpoint`](@ref) answers System One requests from a local decision
model — Tev1, Nimble, Kev, Clef and the others Ollama lists as [decision
models](https://ollama.com/search?c=decision) — through Ollama's
TypeSafe-compatible route `POST /v1/systemone`: Ollama 0.35 or later, 0.35.1 or
later for pictures, and 0.40.0 or later for the MLX builds of Tev1 and Clef Flash
used below; what this section reports was measured on 0.40.6. The
verbs, the questions and the result types are the ones below; no key is sent and
nothing is billed. Start the server and pull the decision models first: the
example below uses `tev1:4b`, `clef-flash:9b` (a vision model, for pictures) and
`kev:0.8b-q8_0` (`ollama pull tev1:4b`, and so on).

```julia
ollama = OllamaEndpoint()                     # OLLAMA_HOST, else http://127.0.0.1:11434

r = ask("Help! My payouts have been failing for 3 days.",
        "department" => choice("Which team should handle this ticket?",
                               (billing = "Payments, invoicing, payouts, refunds",
                                technical = "Bugs, outages, integrations")),
        "is_frustrated" => noul("Is the customer frustrated?");
        service = ollama, model = "tev1:4b")

photo = ImageAttachment("parcel.jpg")         # a vision decision model sees the state's images
ask([photo, "A photo sent in with a return."], "damaged" => noul("Is the item damaged?");
    service = ollama, model = "clef-flash:9b")

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
- **Refused before any request.** [`ask`](@ref) throws an `ArgumentError`
  naming Ollama, the rule and the fix for what Ollama's route refuses in its shared
  checks or documents as its maximum (Ollama 0.40.6): more than 64 questions; a
  Choice of more than 255 options; a Score of fewer than 2 or more than 10 levels;
  a blank text `state` without images; a Box question, which Ollama does not
  answer. By Ollama's source, a Kev model on Ollama's own runner (such as
  `kev:0.8b-mxfp8`) skips the shared checks and takes 1 to 255 levels and a blank
  state; that is untested. It also refuses what Ollama's API reference does not
  list, though some models read it: `instructions` given as anything but a string,
  an object or an array; an option described by anything but a string or
  `nothing`; a Score level or a Noul criterion that is not a string (on Ollama
  0.40.6, `kev:0.8b-q8_0` and `d1:3b`, on the llama.cpp path, answered an option
  described by an object, and `tev1:0.8b` refused it) — and, a rule of UniLM's
  own, blank (whitespace-only) `instructions`, which Ollama would read as none.

| Rule | Ollama |
| :--- | :--- |
| Questions per request | 1–64 |
| `instructions` | optional on the route; when given, a string, an object or an array — and not blank, a refusal of UniLM's own, as Ollama would read blank `instructions` as none |
| Choice options | at most 255 (one is sent), each described by a string or `nothing` |
| Score levels | 2–10, each a string |
| Noul criteria | strings |
| `state` | a text that is not blank, unless images come with it |
| Images | in the state: an image, or a vector of images then texts; refused after a text, inside a JSON object or array, beside values that are not texts, and in a question |
| Box questions | refused: the route answers Choice, Score and Noul |
| Options | `keep_alive` only |

- **A model's own limits.** A model may take less than the route, and its HTTP 400
  says so, in its own words, as a [`SystemOneFailure`](@ref) carrying Ollama's
  message. Measured on Ollama 0.40.6 (an Apple M4 Max, 21 October 2026): Tev1
  takes at most 24 options, Nimble and Clef Flash 26. A Choice of one option is
  sent: the models on the llama.cpp path (`kev:0.8b-q8_0`, `kev:9b-q4_K_M`,
  `d1-omni:600m`, `d1:3b`) answered it, and `tev1:0.8b`, `tev1:4b`, `nimble:9b`
  and `clef-flash:9b` answered 400
  `question "pick": criteria must contain 2–26 candidates`, `strands-decider:2b`
  `question "pick": criteria must contain 2–255 candidates` (for a question named
  `pick`). The route accepts a question without `instructions`, but `tev1:4b`,
  `nimble:9b` and `strands-decider:2b` answered it with 400
  `instructions must be a nonempty string, object, or array` and `kev:0.8b-q8_0`
  with 400 `questions.intent: "instructions" must be provided` (for a question
  named `intent`), while `clef-flash:9b` answered it: pass `instructions` with
  local models. A model without vision answers an image with its own 400:
  `image inputs are not supported by this decision model` from `tev1:0.8b`,
  `registry.ollama.ai/library/kev:0.8b-q8_0 does not support vision` from
  `kev:0.8b-q8_0` ([What a model takes](@ref backends_ollama_limits)). Ollama
  caps a request's text and schema, images excluded, at 64 KiB for the models
  measured, which [`ask`](@ref) does not check: a 76,284-byte state came back as a
  413 [`SystemOneFailure`](@ref), `text and schema must not exceed 64 KiB`, from
  `kev:0.8b-q8_0` (the installed GGUF Kev), `d1-omni:600m` and `tev1:0.8b`; by
  Ollama's source, a Kev model on Ollama's own runner gets 1 MiB instead
  (untested). A [`RawQuestion`](@ref) is sent verbatim, for the server to judge.
- **Images.** The state's images go in Ollama's `images` (raw base64), which the
  model sees before the text and every question shares; the state's texts become
  its `state` — none: `""`, one: that string, several: an array — and a blank text
  is allowed beside images. A recording keys and stores each image as
  `data:<mime>;sha256,<hex>`, never its bytes.
- **`keep_alive`.** `ask(...; keep_alive = 600)` keeps the model loaded for 600
  seconds after the request, `0` unloads it at once and `Inf` keeps it loaded;
  without it the endpoint's own `keep_alive` applies, else the server's default.
  It is left out of a recording's key and file: it changes how long the model
  stays loaded, never an answer. The endpoint's `runner` is sent, and is part of
  the key: a runner the model has no variant for is the server's 400
  ([Runners](@ref ollama_runners)). Its other settings — `truncate`, `shift` and
  the [`OllamaOptions`](@ref) such as `num_ctx` — have no field on this route and
  are not sent, and [`load_model`](@ref) loads a decision model without them.
- **Confidence.** `confidence` is TypeSafe's published measure, computed from the
  returned probabilities — `(n·p_max − 1)/(n − 1)` for a Choice and
  `1 − Σᵢ pᵢ·|i − m| / MAD_unif`, floored at 0, for a Score
  ([Confidence](https://docs.typesafe.ai/confidence)) — so a threshold means the
  same on every backend. Measured on Ollama 0.40.6, the five MLX models
  (`tev1:0.8b`, `tev1:4b`, `nimble:9b`, `strands-decider:2b`, `clef-flash:9b`)
  report `1 − H(p)/ln n` instead, while Kev and the models on the llama.cpp path
  (`kev`, `d1`, `d1-omni`) report TypeSafe's; the server's value stays in each
  answer's `raw`. Probabilities are accepted when they sum to 1 within 0.00005 per
  outcome — Kev models round each to four decimals on Ollama's own runner — and
  the chosen option when it lies at most 0.0001 below the most probable.
- **Speed.** On 129 labelled tickets (21 October 2026, Ollama 0.40.6, an Apple M4
  Max, one request at a time), a warm call took from 0.025 s (`d1-omni:600m`) to
  0.64 s (`clef-flash:9b`) at the median, and a first call, which loads the model,
  0.69 s to 3.44 s; [Which model](@ref backends_ollama_models) has each model's
  accuracy, calibration and latency. The server's decision runners answer one
  request at a time, so [`ask_each`](@ref) keeps one in flight unless its
  `ntasks` says otherwise.
- **Results.** `request_id` is `nothing`, as Ollama sends none; `usage` counts
  `prompt_eval_cached_count` as cached tokens where the server reports it (the
  MLX path; on llama.cpp it reads 0); [`estimated_cost`](@ref) is `0.0`, or the
  price of its name for a cloud model (a tag of `cloud`, or one ending in
  `-cloud`), whose answer keeps the requested name when the reply drops the tag.
  [`with_recorded_answers`](@ref) files the exchanges under
  `/ollama/v1/systemone`, apart from TypeSafe's.
- **Known upstream issues**, still open on 21 October 2026: a model may score a
  request's questions one after another, so its latency grows with their number
  ([#18750](https://github.com/ollama/ollama/issues/18750)); a runner returned
  wrong answers with HTTP 200 until the model was unloaded
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
[`SystemOneWire`](@ref UniLM.SystemOneWire) — what is sent, and the
`record_body` a recording is keyed on and writes; refusing, before any I/O, what
its backend cannot carry — and whose [`decode_system_one`](@ref UniLM.decode_system_one)
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
TypeSafe's `list_models` (and so `nl_dispatch`, `nl_classify` and `@branch`),
[`chatrequest!`](@ref) and [`respond`](@ref), streamed or not, and
[`embeddingrequest!`](@ref) (and so the tool loops), the Responses lifecycle calls,
[`fim_complete`](@ref) and [`prefix_complete`](@ref) exchange through a directory of
recordings keyed by the exact
request bytes — with any credential the body carries redacted, every base64
`data:` URL it carries (an image, a PDF, audio, any media) replaced by the
SHA-256 of its base64 text, and a System One backend's options that change how
a request is served left out (media sent as bare base64 are written as sent, such
as Ollama's chat `images` or an `input_audio` clip, except the images of an Ollama
System One request and the media of an Ollama embeddings request, keyed and written
by digest too); images, audio, files, MCP and
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
