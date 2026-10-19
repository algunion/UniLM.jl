# [Backends: Jev, OpenAI and Local Models](@id system_one_backends)

UniLM has one System One surface. The questions — [`choice`](@ref),
[`score`](@ref), [`noul`](@ref), [`box`](@ref) — the verbs — [`ask`](@ref),
[`ask_each`](@ref), [`nl_classify`](@ref), [`nl_dispatch`](@ref),
[`@branch`](@ref) — and the typed results are the same whichever model answers.
Three backends answer them:

- **Jev**, TypeSafe's System One model, the default
  ([`TYPESAFEServiceEndpoint`](@ref)).
- **OpenAI's Decisions API**, a public beta ([`OPENAIServiceEndpoint`](@ref)).
- **Local decision models**, served by Ollama on your own machine
  ([`OllamaEndpoint`](@ref)).

Each backend has a wrapper that knows its rules. What a backend cannot carry is
refused with an `ArgumentError` before anything is sent, so a request never
reaches a backend in a form that backend would read differently.

| I want to… | Use | You get | Section |
| :--- | :--- | :--- | :--- |
| ask Jev | nothing to choose: `export TYPESAFE_API_KEY=…` | TypeSafe's `jev-latest`, the default | [Choose per call or per scope](@ref backends_choose) |
| ask another backend in one call | `service = …`, and `model = …` where it has no default | the same typed results, from that backend | [The same call on three backends](@ref backends_same_call) |
| move a block of code to another backend | [`with_system_one`](@ref)`(service; model)` | every System One call inside asks it | [Choose per call or per scope](@ref backends_choose) |
| know what a backend takes before I send | the comparison table | questions, limits, images, Box, price, status | [What each backend takes](@ref backends_table) |
| reuse a confidence threshold on another backend | a new measurement: `min_confidence` is tuned per backend | a bar that means what you tuned it for | [One formula, calibrated per model](@ref backends_confidence) |
| decide on my own machine | [`OllamaEndpoint`](@ref) and a decision model | no key, no cost, no network | [Local decision models with Ollama](@ref backends_ollama) |
| use OpenAI's Decisions API | [`OPENAIServiceEndpoint`](@ref) | `gpt-6-luna`'s answers; a refused question is a decline | [OpenAI's Decisions API](@ref backends_openai) |
| test without the network | [`with_recorded_answers`](@ref) | each backend's answers, recorded apart | [Recordings per backend](@ref backends_recordings) |

## [The same call on three backends](@id backends_same_call)

**The job:** route the support desk's inbox ([Start Here](@ref jev_start)) on
the backend that suits it — a hosted model, or one on your own machine — without
rewriting the router. **Without one surface:** a client per provider, each with
its own request shape, answer format, errors and tests. **With it:** the same
[`nl_classify`](@ref) call, and `service =` names the backend.

```@example backends
using UniLM

const MESSAGES = [
    "I was charged twice for order #4471. Please refund the duplicate payment.",
    "The app crashes every time I open my order history. iPhone 15, latest version.",
    "My parcel was due last Monday and the tracking hasn't moved in six days.",
    "Do you ship to Norway, and how much does delivery cost?",
    "The phone I received has a cracked screen and the box was crushed. I want my money back.",
    "Your update deleted my saved addresses and now I can't check out. Third time this month!",
]

const TEAM = (billing   = "payments, charges, invoices or refunds",
              technical = "the app or the website does not work as expected",
              shipping  = "a parcel that is late, lost or arrived damaged",
              other     = "anything else")

# Each backend as the keywords that select it. Ollama has no default model: name one.
const BACKENDS = ["Jev"           => (service = TYPESAFEServiceEndpoint, model = nothing),
                  "OpenAI"        => (service = OPENAIServiceEndpoint,   model = nothing),
                  "tev1 (Ollama)" => (service = OllamaEndpoint(),        model = "tev1:0.8b")]

# The team nl_classify returns, and the answer it read it from (on_response hands it over).
function classify(message, backend)
    seen = SystemOneSuccess[]
    team = nl_classify(message, TEAM; backend..., on_response = r -> push!(seen, r))
    (team, only(seen)["classify"])
end

results = [classify(m, b) for m in MESSAGES, (_, b) in BACKENDS]     # a row per message

println(rpad("", 30), join(rpad(name, 16) for (name, _) in BACKENDS))
for (i, message) in enumerate(MESSAGES)
    println(rpad(first(message, 28), 30),
            join(rpad(team, 10) * rpad(round(a.confidence; digits = 2), 6) for (team, a) in results[i, :]))
end
```

Each column is the team a backend chose, and the confidence of its answer. The
three agree on four of the six messages. They part on the other two: the
question about shipping to Norway goes to `billing` on Jev, at a confidence of
only 0.49, and to `other` on the other two; the cracked phone — a damaged parcel
and a refund request in one message — goes to `shipping` on Jev and to
`billing` on the other two, both times at a confidence below 0.5. The code
around the call did not change: whichever backend answered, the result is a key
of `TEAM`. The confidences in different columns are not the same currency ([One
formula, calibrated per model](@ref backends_confidence)).

## [Choose per call or per scope](@id backends_choose)

A call takes its backend from the first of these that is set:

1. its own `service = …` — on [`ask`](@ref), [`ask_each`](@ref),
   [`list_models`](@ref), [`nl_classify`](@ref), [`nl_dispatch`](@ref) and
   [`@branch`](@ref);
2. the innermost [`with_system_one`](@ref) scope around it, which covers the
   tasks started inside it too;
3. TypeSafe.

Its model is the call's own `model = …`; else, when the call names no `service`,
the scope's model; else the backend's default: on TypeSafe `jev-latest`, or
`TYPESAFE_DEFAULT_MODEL` when it is set, read when the request is sent; on
OpenAI `gpt-6-luna`; on Ollama none, so a call that names no model there is an
`ArgumentError` before anything is sent.

```@example backends
with_system_one(OllamaEndpoint(); model = "tev1:0.8b") do
    (scope = nl_classify(MESSAGES[1], TEAM),                                     # tev1, on this machine
     own   = nl_classify(MESSAGES[1], TEAM; service = TYPESAFEServiceEndpoint))   # Jev, at jev-latest
end
```

The second call names its own `service`, and the scope's model does not follow
it: a model belongs to its backend, and `tev1:0.8b` means nothing to TypeSafe.

**Tune it**

- Set the backend once, where a program starts its work, with
  `with_system_one`; keep `service =` for the call that must go elsewhere.
- Pin the model a threshold was tuned on in the same place:
  `with_system_one(TYPESAFEServiceEndpoint; model = "jev-1.13.0")`
  ([Models and versions](@ref jev_models)).
- `with_system_one` checks its endpoint before running anything: one that does
  not declare `:system_one`, such as a chat-only backend, is an `ArgumentError`,
  and so is a blank model.

## [What each backend takes](@id backends_table)

| | Jev (TypeSafe) | OpenAI's Decisions API | Local models (Ollama) |
| :--- | :--- | :--- | :--- |
| `service` | [`TYPESAFEServiceEndpoint`](@ref), the default | [`OPENAIServiceEndpoint`](@ref) | [`OllamaEndpoint`](@ref)`()` |
| Route | `POST https://api.typesafe.ai/v1/systemone` | `POST https://api.openai.com/v1/decisions` | `POST {host}/v1/systemone`, Ollama 0.35 or later |
| Credentials | `TYPESAFE_API_KEY` | `OPENAI_API_KEY` | none |
| Model | `jev-latest` unless the call names one (`TYPESAFE_DEFAULT_MODEL`); `jev-preview`, or a version such as `jev-1.13.0` | `gpt-6-luna`, the only one | no default: an installed decision model, such as `tev1:0.8b` |
| Questions | Choice, Score, Noul, Box | Choice, Score, Noul | Choice, Score, Noul |
| Limits | 1–255 options, 1–10 levels; 64k tokens per request | 2–255 options, 2–10 levels, 200 questions | 2–26 options, 2–10 levels, 64 questions |
| Guidance | strings, objects, arrays and images, anywhere | strings; `instructions` on every question; no Noul `yes`/`no` | strings, objects or arrays as `instructions`, on every question; strings elsewhere |
| Images | anywhere in the request; experimental | in the state only | not sent |
| Box | Jev 1.14 or later | refused | refused |
| Refusals | none | per question, a [`RefusalAnswer`](@ref) | none |
| `confidence` | TypeSafe's, as sent | OpenAI's, as sent, to two decimals | computed by UniLM; Ollama's own value in `raw` |
| Request id | `x-typesafe-request-id` | `x-request-id` | none |
| Options | `long_context` | `prompt_cache_key`, `safety_identifier` | `keep_alive` |
| Price | input tokens only ([Models](https://docs.typesafe.ai/models)) | input tokens only, \$0.10 per million, cached ones included ([Decisions guide](https://developers.openai.com/api/docs/guides/decisions)) | nothing: your machine |
| Default timeouts | 60 s per attempt, 150 s in all | 60 s per attempt, 150 s in all | the package's: 600 s per attempt, 900 s in all |
| [`ask_each`](@ref) in flight | 16 | 16 | 1 |
| Recordings | `/v1/systemone` | `/openai-decisions/v1/decisions` | `/ollama/v1/systemone` |
| Status, 20 October 2026 | `jev-latest` is Jev 1.13, `jev-preview` Jev 1.14 (14 October 2026); images and Box experimental | public beta since 6 October 2026 | verified with Ollama 0.40.6, `tev1:0.8b` and `kev:0.8b-q8_0` |

A timeout you set yourself — `config =`, [`with_request_config`](@ref) or
[`set_default_config!`](@ref) — replaces a backend's default. The [System One
API](@ref system_one_api) has each backend's details: TypeSafe's
[limits](@ref jev_limits) and [errors](@ref jev_errors),
[Ollama](@ref system_one_ollama) and [OpenAI Decisions](@ref system_one_openai).

### [Refused before sending](@id backends_refusals)

**The job:** move a program to another backend and learn, before a request is
paid for, what it uses that the backend cannot carry. **Without a wrapper:** the
backend answers 400 — or accepts the request and reads it differently. **With
one:** each wrapper checks a request against its backend's rules first, and its
`ArgumentError` names the backend, the rule and the fix:

```@example backends
scene = ImageAttachment("../assets/scene.png")    # a 512 × 512 test picture

for (what, call) in [
        "a Box question, to OpenAI"  => () -> ask(scene, "rectangle" => box("the red rectangle");
                                                  service = OPENAIServiceEndpoint),
        "a Noul's yes/no, to OpenAI" => () -> ask(MESSAGES[1], "refund" => noul("Does the customer want a refund?";
                                                      yes = "They ask for their money back");
                                                  service = OPENAIServiceEndpoint),
        "an image, to Ollama"        => () -> ask(scene, "red" => noul("Is there a red shape?");
                                                  service = OllamaEndpoint(), model = "tev1:0.8b"),
        "no model, to Ollama"        => () -> nl_classify(MESSAGES[1], TEAM; service = OllamaEndpoint()),
        "256 options, to TypeSafe"   => () -> ask(MESSAGES[1], "pick" => choice("Pick one.", ["option $i" for i in 1:256]))]
    try
        call()
    catch err
        println(what, ":\n  ", sprint(showerror, err))
    end
end
```

None of these reached a server. A wrapper refuses rather than adapts: dropping a
Noul's `yes`/`no`, or sending an image as text, would answer a question you did
not write, and the answer would carry no sign of the change. The question types
themselves set no limit, so a program built for one backend is checked against
another's rules only when it is sent there.

## [One formula, calibrated per model](@id backends_confidence)

**The job:** act on an answer only when the backend is sure enough, on whichever
backend answers. **Without a shared definition:** each provider's "confidence"
is its own number. **With one:** `confidence` is the same formula on every
backend — and how much a value is worth still depends on the model, so a
threshold is tuned for each.

`confidence` is TypeSafe's published measure
([Confidence](https://docs.typesafe.ai/confidence)): `(n·p_max − 1)/(n − 1)` for
a Choice of `n` options, and `1 − Σᵢ pᵢ·|i − m| / MAD_unif` for a Score whose
most likely level is `m`. TypeSafe and OpenAI report it with each answer.
Ollama's generic decision models report `1 − H(p)/ln n` instead, a measure of
the whole distribution's entropy; the Ollama wrapper computes TypeSafe's from the
probabilities and keeps the server's value in the answer's `raw`:

```@example backends
team, a = results[4, 3]          # the Norway question, answered by tev1
println("team ", team, ", p_max ", round(maximum(values(a.probabilities)); digits = 3),
        ", confidence ", round(a.confidence; digits = 3),
        ", Ollama's own value ", round(a.raw["confidence"]; digits = 3))
```

tev1 put 0.811 of the probability on `other`, out of four options: TypeSafe's
formula makes that a confidence of `(4 × 0.811 − 1)/3 = 0.748`, while the
entropy measure Ollama reports reads the same answer as 0.638. On every
backend, `confidence` is the first kind of number.

One formula is not one calibration. On our labelled tickets, Jev's top
probabilities matched how often it was right, while OpenAI's Decisions API was
right more often than its top probability said: it is under-confident. The same
`min_confidence` therefore hands a different share of the inbox to a person on
each backend — at 0.8, 14 of 129 tickets on Jev and 28 on OpenAI, though both
were right on nearly every ticket they kept. A threshold tuned on one backend
does not carry over to another.

!!! details "Evidence"
    Measured on 20 October 2026 on our 129 labelled support tickets (60 clear,
    32 borderline, 15 adversarial, 22 out of scope): one Choice of seven intents
    per ticket, each option a name with a description, the same request on both
    backends, one call at a time.

    | | Jev (`jev-1.13.0`) | OpenAI (`gpt-6-luna`) |
    | :--- | :--- | :--- |
    | accuracy | 0.969 | 0.946 |
    | expected calibration error (10 bins) | 0.016 | 0.080 |
    | Brier score | 0.052 | 0.076 |
    | mean top probability | 0.96 | 0.89 |
    | tickets at confidence ≥ 0.8, and their accuracy | 115, 0.991 | 101, 1.000 |
    | latency p50 / p95 | 0.25 s / 0.33 s | 0.48 s / 2.44 s |
    | cost per ticket | \$0.0000225 | \$0.0000296 |

    On OpenAI, accuracy was at or above the mean top probability in every bin
    above 0.5. The expectation we wrote down before the run, OpenAI's accuracy
    within 3 points of Jev's, was not refuted (−2.3 points). Limits: one
    labelled set, one run per backend, one way of writing the options.

**Tune it**

- Tune `min_confidence` on each backend you run, on labelled messages of your
  own, with the model pinned ([Act on the answer](@ref jev_act_on_answer)).
- A `decide` policy that prices mistakes ([When mistakes have different
  prices](@ref jev_loss_matrix)) trusts the probabilities to be calibrated:
  check them on a backend before you move the policy there.
- On Ollama, gate on `confidence`, not on `raw["confidence"]`, when a threshold
  must mean what it means on the other backends.

## [Local decision models with Ollama](@id backends_ollama)

**The job:** decide without the text leaving the machine — offline, in a test,
or where data must stay put. **Without a local decision route:** a hosted
service, or an LLM prompted for a label whose reply you parse. **With it:** a
small decision model served by Ollama answers the same questions, over Ollama's
own TypeSafe-compatible `POST /v1/systemone`, with no key and no cost.

1. Run Ollama 0.35 or later (`ollama serve`, or the desktop app).
   [`OllamaEndpoint`](@ref)`()` reads `OLLAMA_HOST`, else
   `http://127.0.0.1:11434`.
2. Pull a decision model: `ollama pull tev1:0.8b` in a shell, or
   [`pull_model`](@ref)`("tev1:0.8b")` in Julia. Ollama lists the others as
   [decision models](https://ollama.com/search?c=decision).
3. Name it: `model = "tev1:0.8b"` on the call, or
   `with_system_one(OllamaEndpoint(); model = "tev1:0.8b")` around the code.

The third column of the first example came from `tev1:0.8b` this way.

- **No default model.** A call that names none is an `ArgumentError` before any
  request; a model that is not installed comes back as a 404
  [`SystemOneFailure`](@ref) whose message says to pull it.
- **Text only.** Images are refused: no vision decision model has been verified
  with this wrapper yet. Box questions are refused: the route answers Choice,
  Score and Noul.
- **Stricter rules.** 1–64 questions per request; `instructions` on every
  question; 2–26 options per Choice, each described by a string or `nothing`;
  2–10 levels per Score, each a string. A model may cap lower, and its own 400
  says so: `tev1:0.8b` takes at most 24 options.
- **`keep_alive`.** `keep_alive = 600` keeps the model loaded for 600 s after
  the request, `0` unloads it at once and `Inf` keeps it loaded.
- **One request at a time.** A decision runner answers one request at a time,
  so [`ask_each`](@ref) keeps one in flight unless `ntasks` says otherwise, and
  the timeouts are the package's own, since loading a model on a CPU can take
  minutes.
- **Results.** `request_id` is `nothing`, as Ollama sends none, and
  [`estimated_cost`](@ref) is `0.0`. With `service = OllamaEndpoint()`,
  [`list_models`](@ref) lists the installed models.
- **Known upstream issues**, open at Ollama 0.40.6: a model may score a
  request's questions one after another, so its latency grows with their number
  ([#18750](https://github.com/ollama/ollama/issues/18750)); one over-long
  request left a runner answering later ones with wrong probabilities until the
  model was unloaded ([#19090](https://github.com/ollama/ollama/issues/19090));
  a vision decision runner returned repeated HTTP 500s after many sequential
  requests ([#19062](https://github.com/ollama/ollama/issues/19062)).
  `keep_alive = 0` unloads a model whose answers stop making sense.

!!! details "Evidence"
    Measured on 20 October 2026 on an Apple M4 Max with Ollama 0.40.6, one
    request at a time: a request to a loaded model took about 17 ms on
    `kev:0.8b-q8_0` and 25 to 140 ms on `tev1:0.8b`; the first request, which
    loads the model, 0.9 s on `kev` and 2.8 s on `tev1`. Every probability
    distribution the two models returned summed to 1 within 4.4e-16.

**Tune it**

- Measure a local model on labelled messages of your own before you rely on
  it, as you would a hosted one: six messages, as above, are a demonstration,
  not a measurement.
- Keep the model loaded while a batch runs (`keep_alive`), and unload it when
  the batch is done.

## [OpenAI's Decisions API](@id backends_openai)

**The job:** run the same decisions on OpenAI's account and infrastructure.
**Without a wrapper:** OpenAI's own request format — questions as an array,
answers matched back by name, refusals to handle. **With one:**
`service = OPENAIServiceEndpoint` and `OPENAI_API_KEY`; the second column of the
first example came from it.

- **Status.** A public beta since 6 October 2026, so its contract may still
  change ([Decisions guide](https://developers.openai.com/api/docs/guides/decisions)).
- **Model.** `gpt-6-luna`, the only one it serves, named when the call names
  none.
- **What it takes.** Choice (2–255 options), Score (2–10 levels, each a string)
  and Noul, sent as a predicate; `instructions`, a string, on every question;
  at most 200 questions. A string state is sent as it is, any other state as its
  JSON text. Images go in the state only — an [`ImageAttachment`](@ref), or a
  `Vector` of strings and images — at most 128 of them.
- **Refused before sending.** A Box question; a Noul's `yes`/`no` criteria, which
  a predicate has no place for (write them into its instructions); a description
  or level that is not a string; an image anywhere but the state; any option but
  `prompt_cache_key` and `safety_identifier`.
- **Refusals.** The API may decline any question. That answer is a
  [`RefusalAnswer`](@ref), and [`nl_classify`](@ref), [`nl_dispatch`](@ref) and
  [`@branch`](@ref) take it as a decline whatever the policy: `fallback` runs,
  or [`DecisionDeclinedError`](@ref) is thrown. After [`ask`](@ref), check
  `a isa RefusalAnswer` before reading a value.
- **Numbers.** Probabilities and confidences arrive rounded to two decimals, so
  a distribution over `n` outcomes is accepted when it sums to 1 within
  `0.005n`.
- **Options.** Keywords of [`ask`](@ref): `safety_identifier`, an opaque id of
  your end user of at most 128 characters, and `prompt_cache_key`, at most 64,
  which partitions OpenAI's prompt cache ([API
  reference](https://developers.openai.com/api/reference/resources/decisions/methods/create)).
- **Results.** `request_id` is the `x-request-id` header, and `usage` counts
  cached input tokens.
- **Price.** Input tokens only, \$0.10 per million on `gpt-6-luna`, cached ones
  at the same rate, and twice that rate for a request of more than 272K input
  tokens: [`estimated_cost`](@ref) charges it that way. A regional-processing
  premium does not show in a response and is not added.

**Tune it**

- Give every Choice an `other` option and every call a `fallback`: a declined
  question then has a path, as a low confidence does.
- Tune `min_confidence` for this backend, not for Jev ([One formula, calibrated
  per model](@ref backends_confidence)).

## [Recordings per backend](@id backends_recordings)

[`with_recorded_answers`](@ref) records and replays the System One calls of
every backend, and files each backend's apart: TypeSafe's under
`/v1/systemone`, as always; Ollama's under `/ollama/v1/systemone`; OpenAI's under
`/openai-decisions/v1/decisions`. The same question asked of two backends is two
recordings, and a replay never answers one backend with another's answer.
Replaying needs no key and no server. Recording needs each backend's access —
`TYPESAFE_API_KEY`, `OPENAI_API_KEY` or a running Ollama server — and a request
with no recording throws [`ReplayMissError`](@ref), whose message says which.
The examples on this page replay that way ([Test and Develop](@ref
jev_testing_recorded)).

## See also

- [Start Here: Decisions in Five Minutes](@ref jev_start) — the support desk on
  Jev, the default backend
- [Route and Decide](@ref system_one_guide) — [`ask`](@ref), the question types,
  images and Box, and decision rules on the answers
- [Many Items at Once](@ref jev_algorithms_guide) — [`ask_each`](@ref) and its
  batch width per backend
- [Test and Develop](@ref jev_testing_guide) — recorded answers and tests that
  need no key
- [System One API](@ref system_one_api) — setup, limits and errors of each
  backend, then every type and verb
- [Local Models with Ollama](@ref ollama_guide) — Ollama's chat, embedding and
  model-management verbs
