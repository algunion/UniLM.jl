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
| move a block of code to another backend | [`with_system_one`](@ref)`(service; …) do … end` | every System One call inside asks it | [Choose per call or per scope](@ref backends_choose) |
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

for (i, message) in enumerate(MESSAGES)
    println(first(message, 32), "…")
    for ((name, _), (team, a)) in zip(BACKENDS, results[i, :])
        println("  ", rpad(name, 15), rpad(team, 11), round(a.confidence; digits = 2))
    end
end
```

Under each message, a line per backend: the team it chose, and the confidence
of its answer. The three agree on four of the six messages. They part on the
other two: the question about shipping to Norway goes to `billing` on Jev, at a
confidence of only 0.49, and to `other` on the other two; the cracked phone — a
damaged parcel and a refund request in one message — goes to `shipping` on Jev
and to `billing` on the other two, both times at a confidence below 0.5. The
code around the call did not change: whichever backend answered, the result is
a key of `TEAM`. The confidences of different backends are not the same
currency ([One formula, calibrated per model](@ref backends_confidence)).

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
`ArgumentError` before anything is sent. Its provider options follow the same
rule: the call's own — keywords of [`ask`](@ref) and [`ask_each`](@ref),
`options = (…)` on the natural-language verbs — else, when the call names no
`service`, the scope's, set as keywords of `with_system_one`:
`with_system_one(OllamaEndpoint(); model = "tev1:0.8b", keep_alive = 600) do … end`.
The backend's wrapper checks them when each request is sent.

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
  `with_system_one(TYPESAFEServiceEndpoint; model = "jev-1.13.0") do … end`
  ([Models and versions](@ref jev_models)).
- `with_system_one` checks its endpoint before running anything: one that does
  not declare `:system_one`, such as a chat-only backend, is an `ArgumentError`,
  and so is a blank model.

## [What each backend takes](@id backends_table)

| | Jev (TypeSafe) | OpenAI's Decisions API | Local models (Ollama) |
| :--- | :--- | :--- | :--- |
| `service` | [`TYPESAFEServiceEndpoint`](@ref), the default | [`OPENAIServiceEndpoint`](@ref) | [`OllamaEndpoint`](@ref)`()` |
| Route | `POST /v1/systemone` | `POST /v1/decisions` | `POST /v1/systemone` |
| Credentials | `TYPESAFE_API_KEY` | `OPENAI_API_KEY` | none |
| Model when a call names none | `TYPESAFE_DEFAULT_MODEL`, else `jev-latest` | `gpt-6-luna`, the only one | none: name an installed decision model, such as `tev1:4b` (`clef-flash:9b` for pictures) |
| Questions | Choice, Score, Noul, Box | Choice, Score, Noul | Choice, Score, Noul |
| Limits | 1–255 options, 1–10 levels; 64k tokens per request | 2–255 options, 2–10 levels, 200 questions | 1–255 options, 2–10 levels, 64 questions; a model may take fewer ([What a model takes](@ref backends_ollama_limits)) |
| Guidance | strings, objects, arrays and images, anywhere | strings; `instructions` on every question; no Noul `yes`/`no` | strings, objects or arrays as `instructions`, when given; strings elsewhere |
| Images | anywhere in the request; experimental | in the state only | in the state, before its texts; vision models only |
| Box | Jev 1.14 or later | refused | refused |
| Refusals | none | per question, a [`RefusalAnswer`](@ref) | none |
| `confidence` | TypeSafe's, as sent | OpenAI's, as sent, to two decimals | computed by UniLM; Ollama's own value in `raw` |
| Request id | `x-typesafe-request-id` | `x-request-id` | none |
| Options | `long_context` | `prompt_cache_key`, `safety_identifier`, `prompt_cache_options` | `keep_alive` |
| Price | input tokens only ([Models](https://docs.typesafe.ai/models)) | input tokens only, \$0.10 per million, cached ones included ([Decisions guide](https://developers.openai.com/api/docs/guides/decisions)) | nothing: your machine (a cloud model — a tag of `cloud`, or one ending in `-cloud` — is priced by the requested name) |
| Default timeouts | 60 s per attempt, 150 s in all | 60 s per attempt, 150 s in all | the package's: 600 s per attempt, 900 s in all |
| [`ask_each`](@ref) in flight | 16 | 16 | 1 |
| Status, 20 October 2026 | `jev-latest` is Jev 1.13, `jev-preview` Jev 1.14 (14 October 2026); images and Box experimental | public beta since 6 October 2026 | verified with Ollama 0.40.6 on nine decision models, three of them with vision (21 October 2026) |

- **Hosts.** Jev answers at `https://api.typesafe.ai` and OpenAI at
  `https://api.openai.com`; Ollama at the endpoint's host: `OLLAMA_HOST`, else
  `http://127.0.0.1:11434`. Ollama answers System One from 0.35 on, takes pictures
  from 0.35.1, and serves the MLX decision models recommended here (`tev1:4b`,
  `clef-flash:9b`) from 0.40.0; every Ollama result on this page was measured on
  0.40.6.
- **Other models.** On Jev, a call may name `jev-preview`, or a version such as
  `jev-1.13.0`, instead ([Models and versions](@ref jev_models)).
- **Recordings.** Each backend's go under a path of its own: `/v1/systemone`,
  `/openai-decisions/v1/decisions` and `/ollama/v1/systemone` ([Recordings per
  backend](@ref backends_recordings)).

A timeout you set yourself — `config =`, [`with_request_config`](@ref) or
[`set_default_config!`](@ref) — replaces a backend's default. The [System One
API (Jev, OpenAI Decisions, Ollama)](@ref system_one_api) has each backend's
details: TypeSafe's [limits](@ref jev_limits) and [errors](@ref jev_errors),
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
        "an image after a text, to Ollama" => () -> ask(["A customer photo:", scene], "red" => noul("Is there a red shape?");
                                                        service = OllamaEndpoint(), model = "clef-flash:9b"),
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
most likely level is `m`. TypeSafe and OpenAI report it with each answer. On
Ollama (0.40.6, measured on an Apple M4 Max on 21 October 2026), the five MLX
decision models (`tev1:0.8b`, `tev1:4b`, `nimble:9b`, `strands-decider:2b`,
`clef-flash:9b`) report `1 − H(p)/ln n` instead, a measure of the whole
distribution's entropy, while Kev and the models on the llama.cpp path (`kev`,
`d1`, `d1-omni`) report TypeSafe's; the Ollama wrapper computes TypeSafe's from the
probabilities for every model and keeps the server's value in the answer's `raw`:

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

**The job:** decide without the text — or the picture — leaving the machine:
offline, in a test, or where data must stay put. **Without a local decision
route:** a hosted service, or an LLM prompted for a label whose reply you parse.
**With it:** a decision model served by Ollama answers the same questions, over
Ollama's own TypeSafe-compatible `POST /v1/systemone`, with no key and no cost.

1. Run Ollama (`ollama serve`, or the desktop app): 0.35 or later for the route,
   0.35.1 or later for pictures, and 0.40.0 or later for the MLX decision models
   recommended below (`tev1:4b`, `clef-flash:9b`); this section was measured on
   0.40.6. [`OllamaEndpoint`](@ref)`()` reads `OLLAMA_HOST`, else
   `http://127.0.0.1:11434`.
2. Pull a decision model: `ollama pull tev1:4b` in a shell, or
   [`pull_model`](@ref)`("tev1:4b")` in Julia ([Which model](@ref
   backends_ollama_models) compares nine; this page's examples ran on
   `tev1:0.8b`, and on `clef-flash:9b` for pictures). Ollama lists them as
   [decision models](https://ollama.com/search?c=decision).
3. Name it: `model = "tev1:4b"` on the call, or
   `with_system_one(OllamaEndpoint(); model = "tev1:4b") do … end` around the
   code.

The `tev1 (Ollama)` lines of the first example came from `tev1:0.8b` this way.

### [Pictures](@id backends_ollama_images)

A vision decision model answers questions about pictures: measured here,
`clef-flash:9b` (`ollama pull clef-flash:9b`), `d1:3b` and `d1-omni:600m`. Other
decision models whose [`model_info`](@ref) capabilities include `:vision` take the
same request, untested. The pictures go first in the state, then its texts, and
every question of the request sees them:

```@example backends
photo = ImageAttachment("../assets/red_circle.png")
r = ask([photo, "A photo sent in with a return."],
        "colour" => choice("What colour is the shape?", ["red", "green", "blue"]),
        "round"  => noul("Is the shape round?");
        service = OllamaEndpoint(), model = "clef-flash:9b")
(r["colour"].choice, round(r["colour"].confidence; digits = 3), round(r["round"].noul; digits = 3))
```

One request asked both questions about the picture: `clef-flash:9b` saw red, at a
confidence of 0.991, and gave "Is the shape round?" a probability of 0.945.

- **Forms.** The state is an image, or a vector of images followed by texts. The
  texts become the state — none: an empty string, one: that string, several: an
  array — and a blank text beside images is allowed.
- **Refused before sending:** an image after a text, which Ollama would show
  first; an image inside a JSON object or array (to [`nl_dispatch`](@ref), which
  sends its arguments inside an object, pass `state = [image, …texts]`); images
  beside values that are not texts; an image inside a question.
- **A model without vision** answers with its own HTTP 400, a
  [`SystemOneFailure`](@ref) carrying the model's own words: `image inputs are not
  supported by this decision model` from `tev1:0.8b`,
  `registry.ollama.ai/library/kev:0.8b-q8_0 does not support vision` from
  `kev:0.8b-q8_0`. Check `status == 400`, not the text.
- **Recordings** key and store each image by its SHA-256, never its bytes.
- **`d1-omni:600m`** takes pictures but read them unreliably: it called a red
  square "blue" (probability 0.954), and gave "Is there a red shape?" about the red
  circle a probability of 0.0000006. For pictures, prefer `clef-flash:9b`, or
  `d1:3b` where MLX does not run.

### [What a model takes](@id backends_ollama_limits)

[`ask`](@ref) refuses, before sending, what Ollama's route refuses in its shared
checks or documents as its maximum (Ollama 0.40.6): more than 64 questions; a Choice
of more than 255 options; a Score of fewer than 2 or more than 10 levels; a blank
text state without images; a Box question, which Ollama does not answer. It also
refuses what Ollama's API reference does not list, though some models read it:
`instructions` given as anything but a string, an object or an array; an option
described by anything but a string or `nothing`; a Score level or a Noul criterion
that is not a string (`kev:0.8b-q8_0` and `d1:3b`, on the llama.cpp path, answered
an option described by an object, and `tev1:0.8b` refused it) — and, a rule of
UniLM's own, blank (whitespace-only) `instructions`, which Ollama would read as
none. By Ollama's source, a Kev model on Ollama's own runner (such as
`kev:0.8b-mxfp8`) skips the shared checks and takes 1 to 255 levels and a blank
state; that is untested. A model may take less, and says so with its own HTTP 400,
which comes back as a [`SystemOneFailure`](@ref) carrying Ollama's message (the
messages in the table below begin with the question's name, `question "pick": …`).
A Choice of one option is sent too:

| Model | Runner | Options per Choice | One option came back as | 27 options came back as |
| :--- | :--- | :--- | :--- | :--- |
| `tev1:0.8b`, `tev1:4b` | MLX | 2 to 24 | 400 `criteria must contain 2–26 candidates` | 400 `criteria must contain 2–26 candidates` (26: 400 `Tev1 supports at most 24 candidates`) |
| `nimble:9b`, `clef-flash:9b` | MLX | 2 to 26 | 400 `criteria must contain 2–26 candidates` | 400 `criteria must contain 2–26 candidates` |
| `strands-decider:2b` | MLX | 2 to 27 measured | 400 `criteria must contain 2–255 candidates` | an answer |
| `kev:0.8b-q8_0`, `kev:9b-q4_K_M`, `d1:3b`, `d1-omni:600m` | llama.cpp | 1 to 27 measured | an answer, `{"a": 1.0}` | an answer |

The route accepts a question without `instructions`, but most models do not:
`tev1:4b`, `nimble:9b` and `strands-decider:2b` answered 400 `instructions must be a
nonempty string, object, or array`, `kev:0.8b-q8_0` 400 `questions.intent:
"instructions" must be provided` (for a question named `intent`), and of the five
tried only `clef-flash:9b` answered — so pass `instructions` with local models.
Ollama also caps a request's text and schema, images excluded, at 64 KiB for the
models measured: a 76,284-byte state came back as a 413 [`SystemOneFailure`](@ref),
`text and schema must not exceed 64 KiB`, from `kev:0.8b-q8_0` (the installed GGUF
Kev), `d1-omni:600m` and `tev1:0.8b`. By Ollama's source, a Kev model on Ollama's
own runner gets 1 MiB instead (untested). UniLM does not check the cap. Measured
with Ollama 0.40.6 on an Apple M4 Max on 21 October 2026.

### [Which model](@id backends_ollama_models)

Nine decision models answered the 129 labelled support tickets of [One formula,
calibrated per model](@ref backends_confidence) — the same Choice of seven intents
— through UniLM on one machine:

- **Most accurate:** `tev1:4b`, right on 0.938 of the tickets and the best
  calibrated of the nine (expected calibration error 0.038), in about 0.24 s a
  ticket.
- **Small and fast:** `d1-omni:600m`, 0.868 in 0.025 s, and `strands-decider:2b`,
  0.899 in 0.084 s.
- **Pictures:** `clef-flash:9b`, 0.915 on these text tickets in 0.64 s, or
  `d1:3b`, 0.845 in 0.14 s on llama.cpp.
- `kev:9b-q4_K_M` is nearly as accurate (0.922), but its probabilities are
  poorly calibrated (0.145); `kev:0.8b-q8_0` was right on 0.690.

Jev and OpenAI's Decisions API scored 0.969 and 0.946 on the same tickets. Each
model's answer was the one Ollama sent: UniLM's probabilities equalled the
server's reply to the same bytes, on all nine.

!!! details "Evidence"
    Measured on 21 October 2026 with Ollama 0.40.6 on an Apple M4 Max with 64 GB,
    through UniLM: one Choice of seven intents per ticket (60 clear, 32 borderline,
    22 out of scope, 15 adversarial), one call at a time, `keep_alive = Inf`.
    Latency is the client's wall clock per warm call, measured again on a quiet
    machine; the first call loads the model. Jev and OpenAI: the 20 October run
    above.

    | Model | Runner | Accuracy | ECE (10 bins) | p50 / p95 | First call |
    | :--- | :--- | :--- | :--- | :--- | :--- |
    | `tev1:4b` | MLX | 0.938 | 0.038 | 0.241 s / 0.265 s | 1.37 s |
    | `kev:9b-q4_K_M` | llama.cpp | 0.922 | 0.145 | 0.413 s / 0.417 s | 3.44 s |
    | `clef-flash:9b` | MLX | 0.915 | 0.075 | 0.636 s / 0.718 s | 3.19 s |
    | `strands-decider:2b` | MLX | 0.899 | 0.090 | 0.084 s / 0.090 s | 1.01 s |
    | `nimble:9b` | MLX | 0.884 | 0.064 | 0.547 s / 0.608 s | 2.67 s |
    | `d1-omni:600m` | llama.cpp | 0.868 | 0.055 | 0.025 s / 0.036 s | 0.80 s |
    | `d1:3b` | llama.cpp | 0.845 | 0.047 | 0.139 s / 0.147 s | 2.42 s |
    | `tev1:0.8b` | MLX | 0.806 | 0.059 | 0.041 s / 0.045 s | 0.69 s |
    | `kev:0.8b-q8_0` | llama.cpp | 0.690 | 0.167 | 0.046 s / 0.048 s | 1.12 s |
    | Jev (hosted) | — | 0.969 | 0.016 | 0.254 s / 0.328 s | — |
    | OpenAI Decisions (hosted) | — | 0.946 | 0.080 | 0.475 s / 2.437 s | — |

    The refutation we wrote down before the run — a model of 2B parameters or more
    right on fewer than half of the 60 clear tickets — did not occur (0.850 to
    0.983). Repeated runs gave the same answers to every ticket; under other load on
    the machine, latencies rose up to about twofold. Limits: one labelled set, one
    machine, one way of writing the options; n = 129, so an accuracy near 0.95 is
    known to about ±0.04. Separately, in UniLM's live test suite on the same nine
    models (Ollama 0.40.6), every probability distribution summed to 1 within
    4.4e-16.

- **No default model.** A call that names none is an `ArgumentError` before any
  request; a model that is not installed comes back as a 404
  [`SystemOneFailure`](@ref) whose message says to pull it.
- **`keep_alive`.** `keep_alive = 600` keeps the model loaded for 600 s after
  the request, `0` unloads it at once and `Inf` keeps it loaded. It is not part
  of a recording's key, since it changes no answer.
- **`runner`.** Leave it unset, or pin per model the runner its listing names
  (`OllamaModel.runner`): `OllamaEndpoint(runner = :llamacpp)` with
  `model = "kev:0.8b-q8_0"`. A runner the model has no variant for is the server's
  HTTP 400 `no compatible manifest found for runners: <runner>`, never a fallback —
  on Ollama 0.40.6, every runner but the listed one on each of the nine models
  measured here — and the pin goes with every request of the endpoint
  ([Runners](@ref ollama_runners)). It is part of a recording's key.
- **One request at a time.** A decision runner answers one request at a time,
  so [`ask_each`](@ref) keeps one in flight unless `ntasks` says otherwise, and
  the timeouts are the package's own, since loading a model on a CPU can take
  minutes.
- **Results.** `request_id` is `nothing`, as Ollama sends none, and
  [`estimated_cost`](@ref) is `0.0` — for a cloud model reached through the
  server, a tag of `cloud` or one ending in `-cloud`, the price of the requested
  name. With `service = OllamaEndpoint()`, [`list_models`](@ref) lists the
  installed models.
- **Known upstream issues**, still open on 21 October 2026: a model may score a
  request's questions one after another, so its latency grows with their number
  ([#18750](https://github.com/ollama/ollama/issues/18750)); a runner returned
  wrong answers with HTTP 200 until the model was unloaded
  ([#19090](https://github.com/ollama/ollama/issues/19090));
  a vision decision runner returned repeated HTTP 500s after many sequential
  requests ([#19062](https://github.com/ollama/ollama/issues/19062)).
  `keep_alive = 0` unloads a model whose answers stop making sense.

**Tune it**

- Pick the model for the job — a small one where speed matters, `tev1:4b` where
  accuracy does, a vision model for pictures — and measure it on labelled
  messages of your own before you rely on it, as you would a hosted one.
- Give every question `instructions` and keep a Choice to 2–24 options: then no
  model measured here refuses a request for its questions (pictures need a vision
  model).
- Tune `min_confidence` per model: the same threshold hands a different share of
  the work to a person on each.
- Keep the model loaded while a batch runs (`keep_alive`), and unload it when
  the batch is done.

## [OpenAI's Decisions API](@id backends_openai)

**The job:** run the same decisions on OpenAI's account and infrastructure.
**Without a wrapper:** OpenAI's own request format — questions as an array,
answers matched back by name, refusals to handle. **With one:**
`service = OPENAIServiceEndpoint` and `OPENAI_API_KEY`; the `OpenAI` lines of
the first example came from it.

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
  `prompt_cache_key`, `safety_identifier` and `prompt_cache_options`.
- **Refusals.** The API may decline any question. That answer is a
  [`RefusalAnswer`](@ref), and [`nl_classify`](@ref), [`nl_dispatch`](@ref) and
  [`@branch`](@ref) take it as a decline whatever the policy: `fallback` runs,
  or [`DecisionDeclinedError`](@ref) is thrown. After [`ask`](@ref), check
  `a isa RefusalAnswer` before reading a value.
- **Numbers.** Probabilities and confidences arrive rounded to two decimals, so
  a distribution over `n` outcomes is accepted when it sums to 1 within
  `0.005n`.
- **Options.** Keywords of [`ask`](@ref): `safety_identifier`, an opaque id of
  your end user of at most 128 characters; `prompt_cache_key`, at most 64,
  which partitions OpenAI's prompt cache ([API
  reference](https://developers.openai.com/api/reference/resources/decisions/methods/create));
  and `prompt_cache_options`, a [`PromptCacheOptions`](@ref) of `mode` and a `ttl`
  of `"30m"` or `"12h"`. None changes an answer, so none is part of a recording's key.
- **Results.** `request_id` is the `x-request-id` header, and `usage` counts
  cached input tokens and cache writes.
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
recordings, and a replay never answers one backend with another's answer. The
options that change how a request is served but never what it is answered —
OpenAI's `prompt_cache_key`, `safety_identifier` and `prompt_cache_options`, Ollama's
`keep_alive` — are left out of the key and the file; TypeSafe's `long_context` and the runner an
`OllamaEndpoint` names are part of the key. An image is recorded by its SHA-256,
never its bytes: TypeSafe's as a digest of each image part, OpenAI's and Ollama's
as `data:<mime>;sha256,<hex>` in place of its base64.
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
- [System One API (Jev, OpenAI Decisions, Ollama)](@ref system_one_api) — setup,
  limits and errors of each backend, then every type and verb
- [Local Models with Ollama](@ref ollama_guide) — Ollama's chat, embedding and
  model-management verbs
