# [Choosing a decision provider](@id system_one_backends)

```@setup choosing
using UniLM
```

UniLM has one decision surface. The questions — [`choice`](@ref), [`score`](@ref), [`noul`](@ref), [`box`](@ref) —
the verbs — [`ask`](@ref), [`ask_each`](@ref), [`nl_classify`](@ref), [`nl_dispatch`](@ref), [`@branch`](@ref) —
and the typed results are the same whichever model answers. Three providers answer them, each with an entry of its
own: [Jev (TypeSafe)](@ref jev_start), the default; [OpenAI Decisions API](@ref openai_decisions_start); and
[Local decision models](@ref ollama_decisions_start) served by Ollama on your own machine. Each provider has a
wrapper that knows its rules. What a provider cannot carry is refused with an `ArgumentError` before anything is
sent, so a request never reaches a provider in a form that provider would read differently.

| I want to… | Use | You get |
| :--- | :--- | :--- |
| [ask another provider in one call](@ref backends_same_call) | `service = …`, and `model = …` where it has no default | the same typed results, from that provider |
| [move a block of code to another provider](@ref backends_choose) | [`with_system_one`](@ref)`(service; …) do … end` | every decision call inside asks it |
| [know what a provider takes before I send](@ref backends_table) | the comparison table | questions, limits, pictures, Box, price, status |
| [see what each provider refuses](@ref backends_refusals) | the same requests on each | the `ArgumentError` that names the provider, the rule and the fix |
| [reuse a confidence threshold on another provider](@ref backends_confidence) | a new measurement: `min_confidence` is tuned per provider | a bar that means what you tuned it for |

## [The same call on three providers](@id backends_same_call)

**The job:** route the support desk's inbox on the provider that suits it — a hosted model, or one on your own
machine — without rewriting the router. **Without one surface:** a client per provider, each with its own request
shape, answer format, errors and tests. **With it:** the same [`nl_classify`](@ref) call, and `service =` names the
provider.

```@example choosing
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

# Each provider as the keywords that select it. Ollama has no default model: name one.
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

Under each message, a line per provider: the team it chose, and the confidence of its answer. The three agree on
four of the six messages. They part on the other two: the question about shipping to Norway goes to `billing` on
Jev, at a confidence of only 0.49, and to `other` on the other two; the cracked phone — a damaged parcel and a
refund request in one message — goes to `shipping` on Jev, at 0.98, and to `billing` on the other two, both times
at a confidence below 0.5. The code around the call did not change: whichever provider answered, the result is a key of
`TEAM`. The confidences of different providers are not the same currency
([One formula, calibrated per model](@ref backends_confidence)).

## [Choose per call or per scope](@id backends_choose)

A call takes its provider from the first of these that is set:

1. its own `service = …` — on [`ask`](@ref), [`ask_each`](@ref), [`list_models`](@ref), [`nl_classify`](@ref),
   [`nl_dispatch`](@ref) and [`@branch`](@ref);
2. the innermost [`with_system_one`](@ref) scope around it, which covers the tasks started inside it too;
3. TypeSafe.

Its model is the call's own `model = …`; else, when the call names no `service`, the scope's model; else the
provider's default, the model at the top of its entry — on TypeSafe `TYPESAFE_DEFAULT_MODEL` instead when it is set,
read when the request is sent; on Ollama none, so a call that names no model there is an `ArgumentError` before
anything is sent. Its provider options follow the same rule: the call's own — keywords of [`ask`](@ref) and
[`ask_each`](@ref), `options = (…)` on the natural-language verbs — else, when the call names no `service`, the
scope's, set as keywords of `with_system_one`:
`with_system_one(OllamaEndpoint(); model = "tev1:0.8b", keep_alive = 600) do … end`. The provider's wrapper checks
them when each request is sent.

```@example choosing
with_system_one(OllamaEndpoint(); model = "tev1:0.8b") do
    (scope = nl_classify(MESSAGES[1], TEAM),                                     # tev1, on this machine
     own   = nl_classify(MESSAGES[1], TEAM; service = TYPESAFEServiceEndpoint))   # Jev, at jev-latest
end
```

The second call names its own `service`, and the scope's model does not follow it: a model belongs to its provider,
and `tev1:0.8b` means nothing to TypeSafe.

**Tune it**

- Set the provider once, where a program starts its work, with `with_system_one`; keep `service =` for the call
  that must go elsewhere.
- Pin the model a threshold was tuned on in the same place:
  `with_system_one(TYPESAFEServiceEndpoint; model = "jev-1.13.0") do … end`
  ([Models](@ref jev_models__next)).
- `with_system_one` checks its endpoint before running anything: one that does not declare `:system_one`, such as
  a chat-only endpoint, is an `ArgumentError`, and so is a blank model.

## [What each provider takes](@id backends_table)

| | Jev (TypeSafe) | OpenAI's Decisions API | Local models (Ollama) |
| :--- | :--- | :--- | :--- |
| `service` | [`TYPESAFEServiceEndpoint`](@ref), the default | [`OPENAIServiceEndpoint`](@ref) | [`OllamaEndpoint`](@ref)`()` |
| Route | `POST /v1/systemone` | `POST /v1/decisions` | `POST /v1/systemone` |
| Credentials | `TYPESAFE_API_KEY` | `OPENAI_API_KEY` | none |
| Model when a call names none | `TYPESAFE_DEFAULT_MODEL`, else the default model of [its entry](@ref jev_start) | the default model of [its entry](@ref openai_decisions_start), the only one | none: name an installed decision model, such as `tev1:4b` (`clef-flash:9b` for pictures) |
| Questions | Choice, Score, Noul, Box | Choice, Score, Noul | Choice, Score, Noul |
| Limits UniLM checks | 1–255 options, 1–10 levels | 2–255 options, 2–10 levels, 200 questions (OpenAI takes 400) | 1–255 options, 2–10 levels, 64 questions; a model may take fewer ([Limits](@ref ollama_decisions_limits)) |
| Guidance | strings, objects, arrays and pictures, anywhere | strings; `instructions` on every question; no Noul `yes`/`no` | strings, objects or arrays as `instructions`; strings elsewhere |
| Where a picture may stand | anywhere a string can: the state, instructions, options, levels | the state only: an image, or a `Vector` of strings and images | the state only, the images before the texts; vision models only |
| Box | Jev 1.14 | refused | refused |
| Refusals | none | per question, a [`RefusalAnswer`](@ref) | none |
| `confidence` | TypeSafe's, as sent | OpenAI's, as sent, to two decimals | computed by UniLM; Ollama's own value in `raw` |
| Request id | `x-typesafe-request-id` | `x-request-id` | none |
| Options | `long_context` | `prompt_cache_key`, `safety_identifier`, `prompt_cache_options` | `keep_alive` |
| Price | input tokens only ([Models](https://docs.typesafe.ai/models)) | input tokens only: \$0.10 per million uncached, cache reads and writes at the model's caching rates ([Decisions guide](https://developers.openai.com/api/docs/guides/decisions#pricing-and-availability)) | nothing: your machine (a cloud model — a tag of `cloud`, or one ending in `-cloud` — is priced by the requested name) |
| Default timeouts | 60 s per attempt, 150 s in all | 60 s per attempt, 150 s in all | the package's: 600 s per attempt, 900 s in all |
| [`ask_each`](@ref) in flight | 16 | 16 | 1 |
| Recordings filed under | `/v1/systemone` | `/openai-decisions/v1/decisions` | `/ollama/v1/systemone` |
| Status, 25 October 2026 | `jev-latest` and `jev-preview` are Jev 1.14; images "dreamer-ready" | generally available | measured with Ollama 0.40.6 on nine decision models, three of them with vision (21 October 2026) |

The Where a picture may stand row is what UniLM sends. TypeSafe takes a picture "anywhere text can go in a System
One request" ([Images](https://docs.typesafe.ai/images)). OpenAI's API also takes pictures in a question's
instructions, one of the additions its guide lists since the public beta
([Decisions guide](https://developers.openai.com/api/docs/guides/decisions#include-images-in-question-instructions)),
which UniLM refuses. Ollama takes a request's pictures as one list, `images`, shared by every question
([System One](https://docs.ollama.com/api/systemone)), apart from the state's text, and by Ollama's source Clef reads
them before that text ([decision/clef.go](https://github.com/ollama/ollama/blob/v0.40.11/decision/clef.go)); UniLM
refuses a picture after a text on every model. Each entry has the rest: [Jev's Images](@ref jev_images__next),
[OpenAI's Questions](@ref openai_decisions_questions) and [Ollama's Images](@ref ollama_decisions_images).

- **Hosts.** Jev answers at `https://api.typesafe.ai` and OpenAI at `https://api.openai.com`; Ollama at the
  endpoint's host: `OLLAMA_HOST`, else `http://127.0.0.1:11434` ([Setup](@ref ollama_decisions_setup)).
- **Other models.** On Jev, a call may name `jev-preview`, or a version such as `jev-1.13.0`, instead
  ([Models](@ref jev_models__next)); on Ollama, any installed decision model
  ([Models](@ref ollama_decisions_models)).
- **Recordings.** Each provider's go under a path of its own, so the same question asked of two providers is two
  recordings, and a replay never answers one provider with another's answer. The options that change how a request
  is served but never what it is answered — OpenAI's three, Ollama's `keep_alive` — are left out of the key and the
  file; TypeSafe's `long_context` and the runner an `OllamaEndpoint` names are part of the key. Recording needs
  each provider's access — `TYPESAFE_API_KEY`, `OPENAI_API_KEY` or a running Ollama server — and replaying needs
  none ([Test and Develop](@ref jev_testing_recorded)).

A timeout you set yourself — `config =`, [`with_request_config`](@ref) or [`set_default_config!`](@ref) — replaces
a provider's default.

### [Refused before sending](@id backends_refusals)

**The job:** move a program to another provider and learn, before a request is paid for, what it uses that the
provider cannot carry. **Without a wrapper:** the provider answers 400 — or accepts the request and reads it
differently. **With one:** each wrapper checks a request against its provider's rules first, and its
`ArgumentError` names the provider, the rule and the fix:

```@example choosing
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

None of these reached a server. A wrapper refuses rather than adapts: dropping a Noul's `yes`/`no`, or sending an
image as text, would answer a question you did not write, and the answer would carry no sign of the change. The
question types themselves set no limit, so a program built for one provider is checked against another's rules
only when it is sent there. Each entry's Limits and refusals section lists its refusals:
[Jev](@ref jev_limits__next), [OpenAI](@ref openai_decisions_limits), [Ollama](@ref ollama_decisions_limits).

## [One formula, calibrated per model](@id backends_confidence)

**The job:** act on an answer only when the provider is sure enough, on whichever provider answers. **Without a
shared definition:** each provider's "confidence" is its own number. **With one:** TypeSafe sends `confidence` by one
formula, UniLM computes it by the same formula on Ollama, and OpenAI's own value matched it in the Choice answers
recorded here; how much a value is worth still depends on the model, so a threshold is tuned for each.

`confidence` is TypeSafe's published measure ([Confidence](https://docs.typesafe.ai/confidence)):
`(n·p_max − 1)/(n − 1)` for a Choice of `n` options, and `1 − Σᵢ pᵢ·|i − m| / MAD_unif` for a Score whose most
likely level is `m`, floored at 0. TypeSafe sends it with each answer, and OpenAI's own `confidence` equals the
Choice formula, to two decimals, in every Choice answer recorded for this manual
([Confidence and tuning](@ref openai_decisions_confidence)). Ollama documents its own measure,
`1 − H(p)/ln N`, a measure of the whole distribution's entropy ([System One](https://docs.ollama.com/api/systemone)),
so the Ollama wrapper computes TypeSafe's from the probabilities for every model and keeps the server's value in
the answer's `raw` ([Confidence and tuning](@ref ollama_decisions_confidence)):

```@example choosing
team, a = results[4, 3]          # the Norway question, answered by tev1
println("team ", team, ", p_max ", round(maximum(values(a.probabilities)); digits = 3),
        ", confidence ", round(a.confidence; digits = 3),
        ", Ollama's own value ", round(a.raw["confidence"]; digits = 3))
```

tev1 put 0.811 of the probability on `other`, out of four options: TypeSafe's formula makes that a confidence of
`(4 × 0.811 − 1)/3 = 0.748`, while the entropy measure Ollama reports reads the same answer as 0.638. On TypeSafe
and Ollama, `confidence` is the first kind of number, and so is OpenAI's in the Choice answers recorded here.

One formula is not one calibration. On our labelled tickets, Jev's top probabilities matched how often it was
right, while OpenAI's Decisions API was right more often than its top probability said: it is under-confident. The
same `min_confidence` therefore hands a different share of the inbox to a person on each provider — at 0.8, 14 of
129 tickets on Jev and 28 on OpenAI, though both were right on nearly every ticket they kept. A threshold tuned on
one provider does not carry over to another.

!!! details "Evidence"
    Measured on 20 October 2026 on our 129 labelled support tickets (60 clear, 32 borderline, 15 adversarial, 22 out
    of scope): one Choice of seven intents per ticket, each option a name with a description, the same request on
    both providers, one call at a time.

    | | Jev (`jev-1.13.0`) | OpenAI (`gpt-6-luna`) |
    | :--- | :--- | :--- |
    | accuracy | 0.969 | 0.946 |
    | expected calibration error (10 bins) | 0.016 | 0.080 |
    | Brier score | 0.052 | 0.076 |
    | mean top probability | 0.96 | 0.89 |
    | tickets at confidence ≥ 0.8, and their accuracy | 115, 0.991 | 101, 1.000 |
    | latency p50 / p95 | 0.25 s / 0.33 s | 0.48 s / 2.44 s |
    | cost per ticket | \$0.0000225 | \$0.0000296 |

    On OpenAI, accuracy was at or above the mean top probability in every bin above 0.5. The expectation we wrote
    down before the run, OpenAI's accuracy within 3 points of Jev's, was not refuted (−2.3 points). Limits: one
    labelled set, one run per provider, one way of writing the options. On the same tickets, the nine local decision
    models measured on 21 October 2026 were right on 0.690 to 0.938 of them
    ([Models](@ref ollama_decisions_models)).

**Tune it**

- Tune `min_confidence` on each provider you run, on labelled messages of your own, with the model pinned
  ([Act on the answer](@ref jev_act_on_answer)).
- A `decide` policy that prices mistakes ([When mistakes have different prices](@ref jev_loss_matrix)) trusts the
  probabilities to be calibrated: check them on a provider before you move the policy there.
- On Ollama, gate on `confidence`, not on `raw["confidence"]`, when a threshold must mean what it means on the
  other providers.

Recorded on 28 September 2026 on `jev-1.13.0` with UniLM's development branch after 0.21.0, and on 19 October 2026
on `gpt-6-luna` and on `tev1:0.8b` (Ollama 0.40.6) with UniLM's development branch after 0.23.0, through
TypeSafe's API at `/v1`, OpenAI's at `/v1/decisions` and Ollama's `/v1/systemone`.
