# [Local decision models](@id ollama_decisions_start)

```@setup ollama_decisions
using UniLM, JSON
```

Ollama serves decision models on your own machine — Tev1, Nimble, Kev, Clef Flash, Strands Decider and the others
it lists as [decision models](https://ollama.com/search?c=decision) — through its System One route,
`POST /v1/systemone`, which answers choice, yes/no and scoring questions about a state with probabilities and
writes no text ([System One](https://docs.ollama.com/api/systemone)); no key is sent to a server of your own, and a
local model costs nothing. `OllamaEndpoint()`, the endpoint of the [Ollama](@ref ollama_guide) language-model
entry, sends [`ask`](@ref) and the verbs built on it there, and no decision verb refuses it. Of the route, UniLM does
not wrap a `keep_alive` given as a duration string; a cloud model reached through your local server takes the same
requests as a local one, and a request to Ollama's cloud itself, `https://ollama.com`, carries the `OLLAMA_API_KEY`
its inference needs and is not sent without it ([Setup](@ref ollama_setup)).

```@eval
Main.ProviderFacts.at_a_glance(:ollama_decisions)
```

| I want to… | Use | You get |
| :--- | :--- | :--- |
| [run a decision model on this machine](@ref ollama_decisions_setup) | `ollama pull tev1:0.8b`, then `service = OllamaEndpoint(), model = "tev1:0.8b"` | decisions with no key and no cost |
| [make a first decision](@ref ollama_decisions_first_decision) | `ask(state, "name" => choice(…); service = OllamaEndpoint(), model = …)` | the chosen option, its probability, and the model that answered |
| [keep a model loaded, or pin its runner](@ref ollama_decisions_options) | `keep_alive`; `OllamaEndpoint(runner = …)` | the model in memory between requests; the build you measured |
| [pick a model](@ref ollama_decisions_models) | the nine-model evaluation | accuracy, calibration and latency on one labelled set |
| [ask several kinds of question at once](@ref ollama_decisions_questions) | `choice`, `score` and `noul` in one `ask` | every answer from one request |
| [decide about a picture](@ref ollama_decisions_images) | `[image, "text"]` as the state, a vision decision model | answers about the picture |
| [know what is refused before a request is sent](@ref ollama_decisions_limits) | the table of refusals, and what each model takes | the words of the `ArgumentError`, and the fix |
| [act only when the model is sure enough](@ref ollama_decisions_confidence) | `min_confidence` and `fallback`, or a `decide` policy | the answer, or a hand-over to a person |
| [test without a server](@ref ollama_decisions_testing) | `with_recorded_answers` | answers recorded once, replayed with no server |
| [decide about many texts](@ref ollama_decisions_throughput) | `ask_each(texts, questions...)` | one result per text, in input order |
| [handle a failure](@ref ollama_decisions_failures) | the result's type | the status and Ollama's message, or the reason no answer came |

## [Setup](@id ollama_decisions_setup)

Run Ollama, the desktop app or `ollama serve`, and pull a decision model:

```bash
ollama pull tev1:0.8b        # the model this page's text examples run on
ollama pull tev1:4b          # the most accurate of the nine measured below
ollama pull clef-flash:9b    # pictures
```

Ollama 0.35.0 or later serves the route, and the Clef models, which read pictures, need 0.35.1 or later
([Decision](https://docs.ollama.com/capabilities/decision)); from 0.40.0, Nimble, Tev1, Clef and Clef Flash run on
MLX on Apple silicon ([v0.40.0](https://github.com/ollama/ollama/releases/tag/v0.40.0)), and a cloud model through
the local server needs 0.40.5 ([System One](https://docs.ollama.com/api/systemone)). This page's measurements ran on
Ollama 0.40.6, and its new examples on 0.40.11.

A call goes to Ollama when it names `service = OllamaEndpoint()`, or when it names no `service` and a
[`with_system_one`](@ref)`(OllamaEndpoint(); model = …)` scope encloses it. Ollama has no default model: a call names
an installed decision model with `model = …`, or, when it names no `service`, takes the scope's; a call left with no
model is refused before anything is sent ([Limits and refusals](@ref ollama_decisions_limits)). `OllamaEndpoint()`
reads `OLLAMA_HOST` when it is built, else `http://127.0.0.1:11434` ([Setup](@ref ollama_setup)); no key is sent to
a server of your own.

## [Options](@id ollama_decisions_options)

Of an [`OllamaEndpoint`](@ref)'s arguments, a decision request carries three: its `base_url`, its `keep_alive`
unless the call names one, and its `runner`. Its other arguments — `truncate`, `shift` and the
[`OllamaOptions`](@ref) such as `num_ctx` — have no field on this route and are not sent, so the window is the one
the model is loaded with, and "input is never truncated" ([System One](https://docs.ollama.com/api/systemone));
[`load_model`](@ref) loads a decision model without them too ([Models](@ref ollama_models)). The constructor's
checks are on [Options](@ref ollama_options).

- **`keep_alive`**, the one request option, is a keyword of [`ask`](@ref) and [`ask_each`](@ref) —
  `options = (keep_alive = …,)` on the natural-language verbs, a keyword of `with_system_one` for a scope —, else the
  endpoint's own: the seconds the model stays loaded after the request, `0` to unload it at once and `Inf` to keep it
  loaded, which UniLM sends as `-1`, the negative value Ollama reads as "keeps it loaded". Unset, Ollama's default
  applies, "5m unless configured otherwise" ([System One](https://docs.ollama.com/api/systemone)). It changes how long
  a model stays in memory, never an answer, so it is not part of a recording's key.
- **`runner`**, an argument of the endpoint, goes with every decision request and is part of a recording's key. Leave
  it unset, or pin per model the runner its listing names (`OllamaModel.runner`):
  `OllamaEndpoint(runner = :llamacpp)` with `model = "kev:0.8b-q8_0"`. A runner the model has no build for is the
  server's HTTP 400 `no compatible manifest found for runners: <runner>`, never a fallback
  ([Runners](@ref ollama_runners)).

## [First decision](@id ollama_decisions_first_decision)

One request carries the state once and every question about it; each answer comes back under the name you gave its
question:

```@example ollama_decisions
const TEAM = (billing   = "payments, charges, invoices or refunds",
              technical = "the app or the website does not work as expected",
              shipping  = "a parcel that is late, lost or arrived damaged",
              other     = "anything else")
message = "I was charged twice for order #4471. Please refund the duplicate payment."

r = ask(message, "team" => choice("Which team should handle this message?", TEAM);
        service = OllamaEndpoint(), model = "tev1:0.8b")
a = r["team"]
(a.choice, a.probabilities[a.choice], a.confidence)
```

`a.choice` is the most probable option, `a.probabilities` holds one probability per option, and `a.confidence`
says how peaked they are, which UniLM computes ([Confidence and tuning](@ref ollama_decisions_confidence)). A failed
call has no answers: `r["team"]` throws a `SystemOneError` on it ([When something fails](@ref ollama_decisions_failures)).

```@example ollama_decisions
r.response.model, token_usage(r).prompt_tokens, r.response.request_id, estimated_cost(r)
```

`r.response.model` is the model the request named, Ollama sends no request id, and a local model's answer costs
`0.0`, with no warning. A cloud model reached through the server — a tag of `cloud`, or one ending in `-cloud` — is
priced by the requested name instead, as is every model a request to Ollama's cloud itself names (an endpoint whose
server is `https://ollama.com`); UniLM's price table holds no row for one, so [`estimated_cost`](@ref) gives `0.0` with
a warning that the row is missing ([Cost Tracking](@ref cost_guide)). [`nl_classify`](@ref) makes the
same kind of decision in one call and returns the key whose description fits
([Dispatch on Meaning](@ref nl_dispatch_guide)):

```@example ollama_decisions
nl_classify(message, TEAM; service = OllamaEndpoint(), model = "tev1:0.8b")
```

## [Models](@id ollama_decisions_models)

On 25 October 2026 Ollama lists ten [decision models](https://ollama.com/search?c=decision): d3, tev1, nimble, clef,
clef-flash, nimble-v3, pplx-decider, kev, laya and strands-decider; `clef` also runs in Ollama's cloud. The d1 and d1-omni
models measured below are decision models by their own pages, with text and image input
([d1](https://ollama.com/library/d1), [d1-omni](https://ollama.com/library/d1-omni)). A model's name is `model:tag`;
name the tag you measured, since a name without one means its `latest` tag ([Setup](@ref ollama_setup)). The model
that answered is the one the request named: Ollama documents the reply's `model` as the "Model name from the
request", and a cloud model whose reply drops its tag keeps the requested name in `r.response.model`, the reply's
in `raw`.

Nine decision models answered the same 129 labelled support tickets, one Choice of seven intents each, through
UniLM on one machine:

- **Most accurate:** `tev1:4b`, right on 0.938 of the tickets and the best calibrated of the nine (expected
  calibration error 0.038), in about 0.24 s a ticket.
- **Small and fast:** `d1-omni:600m`, 0.868 in 0.025 s, and `strands-decider:2b`, 0.899 in 0.084 s.
- **Pictures:** `clef-flash:9b`, 0.915 on these text tickets in 0.64 s, or `d1:3b`, 0.845 in 0.14 s, on the
  `llamacpp` runner where MLX does not run.
- `kev:9b-q4_K_M` is nearly as accurate (0.922), but its probabilities are poorly calibrated (0.145);
  `kev:0.8b-q8_0` was right on 0.690.

Each model's answer was the one Ollama sent: UniLM's probabilities equalled the server's reply to the same bytes,
on all nine. How the hosted providers did on the same tickets is on
[Choosing a decision provider](@ref system_one_backends).

!!! details "Evidence"
    Measured on 21 October 2026 with Ollama 0.40.6 on an Apple M4 Max with 64 GB, through UniLM: one Choice of seven
    intents per ticket (60 clear, 32 borderline, 22 out of scope, 15 adversarial), one call at a time,
    `keep_alive = Inf`. Latency is the client's wall clock per warm call, measured again on a quiet machine; the
    first call loads the model. Jev and OpenAI: the run of 20 October 2026, on the same tickets.

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

    The refutation we wrote down before the run — a model of 2B parameters or more right on fewer than half of the
    60 clear tickets — did not occur (0.850 to 0.983). Repeated runs gave the same answers to every ticket; under
    other load on the machine, latencies rose up to about twofold. Limits: one labelled set, one machine, one way of
    writing the options; n = 129, so an accuracy near 0.95 is known to about ±0.04.

[`list_models`](@ref)`(service = OllamaEndpoint())` lists the installed models, decision models among them, as
[`OllamaModel`](@ref) cards; it is not recorded, so it reaches the server even in a replay scope. On this machine:

```julia
ms = list_models(service = OllamaEndpoint()).response
[(m.name, m.runner) for m in ms if :decision in m.capabilities]
# Ollama 0.40.11, 25 October 2026:
# 9-element Vector{Tuple{String, Symbol}}:
#  ("kev:9b-q4_K_M", :llamacpp)
#  ("clef-flash:9b", :mlx)
#  ("nimble:9b", :mlx)
#  ("tev1:4b", :mlx)
#  ("d1:3b", :llamacpp)
#  ("strands-decider:2b", :mlx)
#  ("d1-omni:600m", :llamacpp)
#  ("kev:0.8b-q8_0", :llamacpp)
#  ("tev1:0.8b", :mlx)
```

## [Questions](@id ollama_decisions_questions)

The route answers three kinds of question, in any mix in one request
([System One](https://docs.ollama.com/api/systemone)): a [`choice`](@ref) picks one option, a [`score`](@ref) places
the state on levels lowest first, and a [`noul`](@ref) gives the probability that a statement holds, its `yes` and
`no` sent as the descriptions of `"true"` and `"false"`. A [`RawQuestion`](@ref) is sent as written, for the server
to judge; a [`box`](@ref) is refused ([Limits and refusals](@ref ollama_decisions_limits)). No answer is passed to
a later question: "Answers are not passed to later questions" ([System One](https://docs.ollama.com/api/systemone)).
By Ollama's source at v0.40.11, though, every prompt for Nimble and Clef holds all the questions of the request,
Clef's under "Decide every field jointly"
([decision/prompts.go](https://github.com/ollama/ollama/blob/v0.40.11/decision/prompts.go),
[decision/clef.go](https://github.com/ollama/ollama/blob/v0.40.11/decision/clef.go)), while a Tev1 prompt holds its
own question alone.

```@example ollama_decisions
q = ask("The app crashes every time I open my order history. Third time this week, and my parcel is due today!",
        "team"    => choice("Which team should handle this message?", TEAM),
        "urgency" => score("How urgent is this message?",
                           ["Can wait", "Needs attention this week", "Needs attention today"]),
        "repeat"  => noul("Has the customer had this problem before?");
        service = OllamaEndpoint(), model = "tev1:0.8b")
(q["team"].choice, q["urgency"].score, q["repeat"].noul)
```

What reaches the model depends on the model, since each family has its own prompt:

- **Question names travel, and may be read.** The request keys its questions by name. By Ollama's source, the
  prompt it builds for Nimble ends with the question's name, while Tev1's holds only the state, the instructions and
  the options ([decision/prompts.go](https://github.com/ollama/ollama/blob/v0.40.11/decision/prompts.go)): write the
  whole question in its instructions.
- **Option names and descriptions both travel.** A Choice's criteria map each option to its description, or to no
  description for a bare name.
- **Give every question instructions.** Ollama's API reference lists `instructions` as required for each question,
  and UniLM refuses a question without them before sending ([Limits and refusals](@ref ollama_decisions_limits)).
  Instructions may be a string, an object or an array; an option's description, a Score level and a Noul's `yes` and
  `no` are strings.

The route takes 1 to 64 questions, a Choice of 2 to 255 options and a Score of 2 to 10 levels, and a model may take
fewer: the limits, and what each model measured takes, are on [Limits and refusals](@ref ollama_decisions_limits).
Reading the answers and the decision rules on them are on [Route and Decide](@ref system_one_guide).

The natural-language verbs ask one Choice of the meanings you write. [`nl_dispatch`](@ref) sends the arguments it
dispatches on as a JSON object, which the route takes as the state, and [`@branch`](@ref) its state as it is; with
no model of their own they are refused like any call ([Limits and refusals](@ref ollama_decisions_limits)):

```@example ollama_decisions
route(::nl"the customer wants a refund", msg)           = :refunds
route(::nl"the customer reports a bug in the app", msg) = :engineering
route(::nl"anything else", msg)                         = :frontline

nl_dispatch(route, message; service = OllamaEndpoint(), model = "tev1:0.8b")
```

```@example ollama_decisions
@branch message service = OllamaEndpoint() model = "tev1:0.8b" begin
    "the customer wants a refund"           => :refunds
    "the customer reports a bug in the app" => :engineering
end
```

## [Images](@id ollama_decisions_images)

A vision decision model answers questions about pictures: Clef and Clef Flash read PNG, JPEG and WebP images, from
Ollama 0.35.1 ([Decision](https://docs.ollama.com/capabilities/decision)), and the pages of d1 and d1-omni list
image input. Measured here: `clef-flash:9b`, `d1:3b` and `d1-omni:600m`; other decision models whose
[`model_info`](@ref) capabilities include `:vision` take the same request, untested. The pictures go first in the
state, then its texts:

```@example ollama_decisions
photo = ImageAttachment("../assets/red_circle.png")
r = ask([photo, "A photo sent in with a return."],
        "colour" => choice("What colour is the shape?", ["red", "green", "blue"]),
        "round"  => noul("Is the shape round?");
        service = OllamaEndpoint(), model = "clef-flash:9b")
(r["colour"].choice, round(r["colour"].confidence; digits = 3), round(r["round"].noul; digits = 3))
```

One request asked both questions about the picture: `clef-flash:9b` saw red, at a confidence of 0.991, and gave
"Is the shape round?" a probability of 0.945.

- **How they are sent.** Ollama takes a request's pictures as one list, `images`, base64-encoded and "shared by all
  questions, in request order" ([System One](https://docs.ollama.com/api/systemone)), and by its source Clef reads
  them before the state's text ([decision/clef.go](https://github.com/ollama/ollama/blob/v0.40.11/decision/clef.go)).
  So the state is a picture, or a vector or a tuple of pictures followed by texts; the texts become the
  request's `state` — none: an empty string, one: that string, several: an array — and a blank text beside
  pictures is sent, as Ollama's source accepts an empty state that comes with pictures
  ([decision/systemone.go](https://github.com/ollama/ollama/blob/v0.40.11/decision/systemone.go)).
- **Refused before sending:** a picture after a text, which Clef would read first; a picture inside a JSON object
  or array (to [`nl_dispatch`](@ref), which sends its arguments inside an object, pass `state = [image, …texts]`);
  pictures beside values that are not texts; a picture in a question
  ([Limits and refusals](@ref ollama_decisions_limits)).
- **A model without vision** answers with its own HTTP 400, a [`SystemOneFailure`](@ref) carrying the model's own
  words ([When something fails](@ref ollama_decisions_failures)): check `status == 400`, not the text.
- **Cost and usage.** A local model's answer costs nothing; the reply's `input_tokens` counts the pictures' token
  positions ([System One](https://docs.ollama.com/api/systemone)).
- **Recordings** key and store each picture by its SHA-256, never its bytes
  ([Testing and recordings](@ref ollama_decisions_testing)).
- **`d1-omni:600m`** takes pictures but read them unreliably: it called a red square "blue" (probability 0.954),
  and gave "Is there a red shape?" about the red circle a probability of 0.0000006. For pictures, prefer
  `clef-flash:9b`, or `d1:3b` where MLX does not run.

## [Limits and refusals](@id ollama_decisions_limits)

What a request meets on Ollama's route ([System One](https://docs.ollama.com/api/systemone)), and which of it UniLM
checks before sending:

| Limit | Value | Checked before sending |
| :--- | :--- | :--- |
| Questions | 1 to 64 per request | yes |
| Choice options | 2 to 255 per question, by the API reference; fewer for most models (below) | yes |
| Score levels | 2 to 10 per question | yes |
| Instructions | required, a string, an object or an array that is not blank | yes |
| State | a string, a JSON object or array, or pictures then texts; a blank string only with pictures | yes |
| Text and schema, pictures excluded | 64 KiB for a local request | no: the server answers 413 |
| The whole request | 32 MiB, base64 and JSON included, for a local request | no: the server answers 413 |
| Context | "The complete input must fit the loaded context window"; "Input is never truncated" | no: the server answers 400 |
| Pictures | one list, `images`, apart from the text and shared by all questions; a vision model | yes, where they stand |
| A cloud model | 64 questions; names of up to 100 ASCII letters, digits, `_`, `.` or `-`; up to 4 pictures of at most 4 MiB and 16 megapixels each, 8 MiB in all; 13 MiB a request; its 65,536-token window | no |

[`ask`](@ref), [`ask_each`](@ref) and the verbs built on them throw these `ArgumentError`s before anything is sent.
UniLM refuses what Ollama's route refuses in its shared checks or documents as its maximum, and what its API
reference does not list, though some models read it; blank instructions are a refusal of UniLM's own, as Ollama
would read them as none. A wrapper refuses rather than adapts: a request changed to fit would ask a question you did
not write.

| Refused | Why: the message's words | What to do |
| :--- | :--- | :--- |
| a call that names no model, unless it names no `service` inside a scope that names one | `has no default System One model`: Ollama has no default | `model = "tev1:0.8b"`, or a `with_system_one(OllamaEndpoint(); model = …)` scope |
| more than 64 questions | `Ollama accepts at most 64 questions per request` | split them across requests |
| a Choice of one option | `Ollama's API reference gives a Choice 2 to 255 options` | add an option, such as `"anything else"` |
| a Choice of more than 255 options | `Ollama accepts at most 255 options per question` | split it, or narrow the options |
| an option described by anything but a string or `nothing` | `Ollama reads a Choice option's description as a string or` | describe it in words, or pass `nothing` |
| a Score of fewer than 2 or more than 10 levels | `Ollama accepts 2 to 10 levels per question` | offer at least two, or merge levels into a coarser rubric |
| a Score level that is not a string | `Ollama reads a Score level as a string` | write the level as text |
| a Noul `yes` or `no` that is not a string | `Ollama reads a Noul criterion as a string` | write it as text |
| a Choice, Noul or Score without instructions | `Ollama's API reference requires`: a question's `instructions` are among its required fields | say what it asks, as the first argument of `choice`, `score` or `noul` |
| blank instructions | `Ollama reads blank`: Ollama would read them as none | say what the question asks |
| instructions that are not a string, an object or an array | `as a string, object or array` | write them as text |
| a `box` question | `Box questions are not supported by Ollama`: the route answers Choice, Score and Noul | ask it of a provider that answers Box questions |
| a question type of your own, a subtype of `SystemOneQuestion` | `its System One route answers Choice, Score and Noul questions, and this wrapper sends a RawQuestion verbatim` | build it as a `RawQuestion` |
| a picture in a question | `Ollama takes images in the state only, shared by all questions` | move it into the state, `[image, "…"]` |
| a picture with a `detail` level | `sends a System One request's images without a detail level`: Ollama's `images` list holds each picture's base64 bytes alone | build the `ImageAttachment` without `detail` |
| a picture after a text in the state | `Ollama takes a request's images in a list apart from its text`: it could only be moved | put the pictures first, `[image, "What is in it?"]` |
| pictures beside values that are not texts | `Ollama takes a state of images then texts, and this one mixes images with` | write the other values as text, after the pictures |
| a picture inside a JSON object or array | `not inside a JSON object or array` | the pictures and texts as the state itself, or `state = [image, …texts]` to `nl_dispatch` |
| a blank text state without pictures | `Ollama refuses a blank`: the state holds what the questions are about | send the content |
| a provider option other than `keep_alive` | `accepts one System One option` | `keep_alive`, or no option |
| a `keep_alive` that is not a number of seconds from 0 to 1e9, or `Inf` | `is a number of seconds from 0 to 1e9, or Inf` | seconds, `0`, or `Inf` |

Like every decision provider, the endpoint also refuses, before anything is sent, a blank or repeated question name,
criteria that are not valid, a state or guidance holding `NaN` or `Inf`, and an `AudioAttachment` or a
`DocumentAttachment` anywhere in the request; [Reading answers](@ref Reading-answers) covers a malformed request on
every provider.

```@example ollama_decisions
for refused in (() -> ask(["A customer photo:", photo], "red" => noul("Is there a red shape?");
                          service = OllamaEndpoint(), model = "clef-flash:9b"),
                () -> nl_classify(message, TEAM; service = OllamaEndpoint()))
    try
        refused()
    catch err
        println(sprint(showerror, err))
    end
end
```

A model may take less than the route, and says so with its own HTTP 400, which comes back as a
[`SystemOneFailure`](@ref) carrying Ollama's message. Measured on Ollama 0.40.6 (21 October 2026), the nine models
answered as below; the messages begin with the question's name, `question "pick": …`, and the one-option Choice,
which UniLM refuses before sending, was sent past that check:

| Model | Runner | Options per Choice | One option came back as | 27 options came back as |
| :--- | :--- | :--- | :--- | :--- |
| `tev1:0.8b`, `tev1:4b` | MLX | 2 to 24 | 400 `criteria must contain 2–26 candidates` | 400 `criteria must contain 2–26 candidates` (26: 400 `Tev1 supports at most 24 candidates`) |
| `nimble:9b`, `clef-flash:9b` | MLX | 2 to 26 | 400 `criteria must contain 2–26 candidates` | 400 `criteria must contain 2–26 candidates` |
| `strands-decider:2b` | MLX | 2 to 27 measured | 400 `criteria must contain 2–255 candidates` | an answer |
| `kev:0.8b-q8_0`, `kev:9b-q4_K_M`, `d1:3b`, `d1-omni:600m` | llama.cpp | 1 to 27 measured | an answer, `{"a": 1.0}` | an answer |

A question without instructions, which UniLM refuses before sending: `tev1:4b`, `nimble:9b` and
`strands-decider:2b` answered 400 `instructions must be a nonempty string, object, or array`, `kev:0.8b-q8_0` 400
`questions.intent: "instructions" must be provided` (for a question named `intent`), and of the five tried only
`clef-flash:9b` answered. An option described by an object:
`kev:0.8b-q8_0` and `d1:3b` answered it, and `tev1:0.8b` refused it; UniLM refuses one for every model, as Ollama's
API reference gives a description as a string.
A 76,284-byte state came back as a 413 `text and schema must not exceed 64 KiB` from `kev:0.8b-q8_0` (the installed
GGUF Kev), `d1-omni:600m` and `tev1:0.8b`. By Ollama's source, a Kev model on Ollama's own runner (such as
`kev:0.8b-mxfp8`) skips the shared checks, and takes 1 to 255 levels and a blank state
([decision/kev.go](https://github.com/ollama/ollama/blob/v0.40.11/decision/kev.go)); that is untested.

!!! details "Evidence"
    Measured with Ollama 0.40.6 on an Apple M4 Max on 21 October 2026, on the nine models of
    [Models](@ref ollama_decisions_models); the API reference and the 64 KiB and 32 MiB limits are Ollama's page of
    25 October 2026.

## [Confidence and tuning](@id ollama_decisions_confidence)

Ollama documents its `confidence` as "Distribution concentration, calculated as 1 - H(p) / ln(N)", an entropy
measure that is "Not calibrated correctness" ([System One](https://docs.ollama.com/api/systemone)). UniLM does not
pass it on: so that a threshold means the same on every provider, it computes `confidence` from the returned
probabilities — `(n·p_max − 1)/(n − 1)` for a Choice of `n` options, clamped to `0 … 1`, as on
[Reading answers](@ref Reading-answers), and `1 − Σᵢ pᵢ·|i − m| / MAD_unif` for a Score whose most likely level is
`m`, floored at 0 ([Confidence](https://docs.typesafe.ai/confidence)); one outcome gives 1 — and keeps the server's
value in the answer's `raw`:

```@example ollama_decisions
const MESSAGES = [
    "I was charged twice for order #4471. Please refund the duplicate payment.",
    "The app crashes every time I open my order history. iPhone 15, latest version.",
    "My parcel was due last Monday and the tracking hasn't moved in six days.",
    "Do you ship to Norway, and how much does delivery cost?",
    "The phone I received has a cracked screen and the box was crushed. I want my money back.",
    "Your update deleted my saved addresses and now I can't check out. Third time this month!",
]

seen = SystemOneSuccess[]
team = nl_classify(MESSAGES[5], TEAM; service = OllamaEndpoint(), model = "tev1:0.8b", min_confidence = 0.6,
                   fallback = message -> :person, on_response = r -> push!(seen, r))
answered = only(seen)["classify"]
(team, round(maximum(values(answered.probabilities)); digits = 3), round(answered.confidence; digits = 3),
 round(answered.raw["confidence"]; digits = 3))
```

tev1 put 0.534 of the probability on `billing`, out of four options: the formula above makes that a confidence of
`(4 × 0.5337 − 1)/3 ≈ 0.378`, below the 0.6 asked for, so the message went to a person; the entropy measure Ollama
reports reads the same answer as 0.375. The two measures part most on a peaked answer: tev1's answer about shipping
to Norway, 0.811 on `other`, is 0.748 by the formula and 0.638 by the entropy
([One formula, calibrated per model](@ref backends_confidence)).

Probabilities are accepted when they sum to 1 within 0.00005 per outcome, since Kev models on Ollama's own runner
round each to four decimals, and the chosen option when it lies at most 0.0001 below the most probable; a reply
further off is a [`SystemOneCallError`](@ref).

- Gate on `confidence`, not on `raw["confidence"]`, when a threshold must mean what it means on another provider.
- Tune `min_confidence` per model, on labelled messages of your own: the same bar hands a different share of the
  work to a person on each model. Where some mistakes cost more than others, price each one and let a `decide`
  policy choose the cheapest action ([When mistakes have different prices](@ref jev_loss_matrix)), once you have
  checked that the model's probabilities are calibrated ([Models](@ref ollama_decisions_models)).
- Pin the model's tag, and its runner where it has builds for several: the answers are the build's.

!!! details "Evidence"
    Measured on Ollama 0.40.6, an Apple M4 Max, 21 October 2026: the five MLX models (`tev1:0.8b`, `tev1:4b`,
    `nimble:9b`, `strands-decider:2b`, `clef-flash:9b`) reported the entropy measure, while Kev and the models on the
    llama.cpp runner (`kev`, `d1`, `d1-omni`) reported the Choice formula above. Repeated runs of the 129 tickets
    gave the same answer to every ticket, and in UniLM's live test suite on the nine models every probability
    distribution summed to 1 within 4.4e-16.

## [Testing and recordings](@id ollama_decisions_testing)

[`with_recorded_answers`](@ref) records an exchange with Ollama's route once and replays it with no server: `ask`,
`ask_each` and the verbs built on them. A recording is filed under `/ollama/v1/systemone`, the request's path behind
the namespace `ollama`, and keyed on the exact request: its method, that path, and its body, which names the model
and, when the endpoint sets one, the `runner`. `keep_alive` is sent and left out of the key and the file, and a
picture is keyed and written as `data:<mime>;sha256,<hex>`, the SHA-256 of its base64 text, never its bytes.

```@example ollama_decisions
recordings = mktempdir()      # in a package: a directory committed next to the tests
recorded = with_recorded_answers(recordings; mode = :record) do
    ask(message, "team" => choice("Which team should handle this message?", TEAM);
        service = OllamaEndpoint(), model = "tev1:0.8b")
end
JSON.parsefile(only(readdir(recordings; join = true)))["request"]["path"]
```

```@example ollama_decisions
replayed = with_recorded_answers(recordings) do
    ask(message, "team" => choice("Which team should handle this message?", TEAM);
        service = OllamaEndpoint(base_url = "http://127.0.0.1:9"), model = "tev1:0.8b", keep_alive = 0)
end
replayed["team"].probabilities == recorded["team"].probabilities
```

The replay names another host, where no server runs, and a `keep_alive` the recording did not, and is answered from
the recording: neither is part of the key. Recording takes a running server with the model pulled, and replaying
takes nothing; in this manual's build the recording scope above replays the manual's own recordings, so the example
reaches no server either ([How this manual runs its examples](@ref jev_testing_manual)).
[`list_models`](@ref) on Ollama is not recorded: inside a scope it goes to the server, as outside one.

- **What a recording cannot hold.** To test what your code does with an answer the model rarely gives — a near-tie,
  a failed call, an option it was never offered — point an `OllamaEndpoint` at a local server that answers what you
  wrote, in the route's shape. Run such tests outside `with_recorded_answers`. This block is not run when the
  manual is built:

```julia
using UniLM, HTTP, JSON, Test

route(::nl"the customer wants a refund", msg)           = :refunds
route(::nl"the customer reports a bug in the app", msg) = :engineering
route(::nl"anything else", msg)                         = :frontline

canned = Ref{Any}()                  # (status, body) the server answers next
server = HTTP.serve!("127.0.0.1", 8125; verbose = false) do req
    status, body = canned[]
    HTTP.Response(status, ["Content-Type" => "application/json"], JSON.json(body))
end
mock = OllamaEndpoint(base_url = "http://127.0.0.1:8125")

# A 200 in the route's shape: the one question is "meaning_1"; `option` gets probability p.
const OPTIONS = meanings(route, Tuple{String})[1]
choosing(option, p) = (200, (model = "tev1:0.8b", usage = (input_tokens = 96, output_tokens = 0),
    answers = (meaning_1 = (type = "choice", choice = option, confidence = 0.5,
                            probabilities = Dict(o => o == option ? p : (1 - p) / (length(OPTIONS) - 1)
                                                 for o in OPTIONS)),)))

@testset "route on a local decision model" begin
    canned[] = choosing("the customer wants a refund", 0.97)
    @test nl_dispatch(route, "I want my money back."; service = mock, model = "tev1:0.8b") === :refunds

    canned[] = choosing("the customer wants a refund", 0.40)      # a confidence of 0.1 by UniLM's formula
    @test nl_dispatch(route, "Hm."; service = mock, model = "tev1:0.8b",
                      min_confidence = 0.7, fallback = m -> :human) === :human

    canned[] = (404, (error = "model 'tev1:0.8b' not found",))
    @test_throws SystemOneError nl_dispatch(route, "Hm."; service = mock, model = "tev1:0.8b")
end
close(server)
```

The reply's `confidence` there is ignored: UniLM computes its own from the probabilities. An answer outside the
offered options makes the call a `SystemOneCallError`, so `nl_dispatch` throws a `SystemOneError`, and no method
runs.

## [Throughput](@id ollama_decisions_throughput)

[`ask_each`](@ref) sends one request per item and returns the results in input order; on Ollama it keeps one request
in flight unless its `ntasks` says otherwise, as a local decision runner may serve one request at a time: Ollama
starts Nimble's for one ([#18750](https://github.com/ollama/ollama/issues/18750)). How a batch backs off when a
server is busy is on [Many requests at once](@ref jev_items_each).

```@example ollama_decisions
results = ask_each(MESSAGES, "team" => choice("Which team should handle this message?", collect(TEAM));
                   service = OllamaEndpoint(), model = "tev1:0.8b")
[(findfirst(==(r["team"].choice), TEAM), round(r["team"].confidence; digits = 2)) for r in results]
```

A request's questions may be scored one after another, so its latency grows with their number
([#18750](https://github.com/ollama/ollama/issues/18750), open on 25 October 2026), and in the reply's
`input_tokens`, "Shared context is counted again when the model scores questions separately"
([System One](https://docs.ollama.com/api/systemone)). Keep the model loaded while a batch runs (`keep_alive`), and
unload it when the batch is done; the first call loads it. Each model's warm latency, and its first call, are in the
Evidence of [Models](@ref ollama_decisions_models).

## [When something fails](@id ollama_decisions_failures)

A call that reaches Ollama returns a [`SystemOneSuccess`](@ref); a [`SystemOneFailure`](@ref) for a status other
than 200, with the `status` and Ollama's `message`, and no request id; or a [`SystemOneCallError`](@ref) when no
usable response came. A 200 whose answers do not fit the request — a question unanswered, an option that was not
offered, probabilities that are no distribution within the rounding above — is a `SystemOneCallError` too. Reading
an answer of a failed call (`r["name"]`, [`answer`](@ref), [`answers`](@ref)) throws a [`SystemOneError`](@ref).

UniLM sends a request again after a 408, 429, 500, 502, 503, 504, 520 to 524 or 529, up to `max_attempts` times
within `total_deadline`, waiting at least what a `retry-after-ms` header asks for, else a `Retry-After`; any other
status comes back as it is.
Ollama brings no bounds of its own, since loading a model on a CPU can take minutes: with no
[`RequestConfig`](@ref) of yours, a call has the field defaults, 600 s per attempt and 900 s in all
([Timeouts & Retries](@ref timeouts_guide)).

```julia
r = ask(message, "team" => choice("Which team?", TEAM); service = OllamaEndpoint(), model = "tev1:2b")
(r.status, r.message)
# Ollama 0.40.11, 25 October 2026, a model that is not installed:
# (404, "model 'tev1:2b' not found — pull it first: `ollama pull tev1:2b`, or `pull_model(\"tev1:2b\")`")

r = ask(message, "team" => choice("Which team?", TEAM); service = OllamaEndpoint(base_url = "http://127.0.0.1:9"),
        model = "tev1:0.8b", config = RequestConfig(max_attempts = 1))
r.error
# no server at the address:
# "http connect error to 127.0.0.1:9: connect tcp -> 127.0.0.1:9: SystemError: connect: Connection refused — no
#  Ollama server answered at http://127.0.0.1:9: start it with `ollama serve` (or the Ollama app), or point
#  OLLAMA_HOST or base_url at the server"
```

| What you see | Why | What to do |
| :--- | :--- | :--- |
| a `SystemOneCallError` whose `error` holds `no Ollama server answered at` | no server at the endpoint's address | start Ollama, or point `OLLAMA_HOST` or `base_url` at it |
| a `SystemOneFailure` with status 404, `model '…' not found`, then `pull it first` | the model is not installed | `ollama pull` it, or [`pull_model`](@ref) |
| a `SystemOneFailure` with status 400 and the model's own words | the model takes less than the route: more options than it takes, a picture it cannot see | see what each model takes ([Limits and refusals](@ref ollama_decisions_limits)); a vision model for pictures |
| status 400, `no compatible manifest found for runners: <runner>` | the endpoint pins a runner the model has no build for | leave `runner` unset, or pin the runner the listing names ([Options](@ref ollama_decisions_options)) |
| status 400 about the loaded context | the input does not fit the window the model is loaded with, and Ollama never truncates it | shorten the state, or ask fewer questions at once |
| status 413 (`text and schema must not exceed 64 KiB` for the first) | the text and the questions of a local request over 64 KiB, or the whole request over 32 MiB | shorten the state, or split the questions |
| status 500 | the model failed to load, render or score; a vision runner returned repeated 500s after many sequential requests ([#19062](https://github.com/ollama/ollama/issues/19062)) | retry; restart Ollama if it repeats |
| answers that stop making sense, with status 200 | a runner returned wrong answers until the model was unloaded ([#19090](https://github.com/ollama/ollama/issues/19090)) | `keep_alive = 0` on a request unloads the model |
| status 401 or 402 | a cloud model without signing in, or not on your plan | `ollama signin`, or a plan with the model |
| a `SystemOneCallError` whose `cause` is a `UniLMTimeout` or a `UniLMCancelled` | a timeout, or a cancelled token | a `RequestConfig` of your own |
| a `SystemOneCallError` after a 200 | an answer that does not fit the request | read its `error`, which names the answer that did not fit, and check the model's build |
| a `ReplayMissError`, thrown | in a replay scope, a request with no recording | record it with the server running ([Testing and recordings](@ref ollama_decisions_testing)) |
| a `SystemOneError`, thrown | an answer read from a failed call | test `issuccess(r)` first |

!!! details "Evidence"
    The model-specific 400s are those of [Limits and refusals](@ref ollama_decisions_limits), captured on Ollama
    0.40.6. A model without vision answered a picture with `image inputs are not supported by this decision model`
    (`tev1:0.8b`) and `registry.ollama.ai/library/kev:0.8b-q8_0 does not support vision` (`kev:0.8b-q8_0`); a pinned
    runner without a build, on every model measured, with the `no compatible manifest found` 400. Ollama's page lists
    the 400, 401, 402, 404, 413 and 500 of the route ([System One](https://docs.ollama.com/api/systemone)); the issues
    #18750, #19090 and #19062 were open on 25 October 2026.

Recorded on 25 October 2026 on `tev1:0.8b` through Ollama 0.40.11 with UniLM's development branch after 0.26.0; on
21 October 2026 the picture, on `clef-flash:9b` through Ollama 0.40.6, with UniLM's development branch after 0.24.0;
and on 19 October 2026 the `nl_classify` calls, on `tev1:0.8b` through Ollama 0.40.6, with UniLM's development
branch after 0.23.0; all through Ollama's `/v1/systemone`.

See also: [`OllamaEndpoint`](@ref), [`ask`](@ref), [`UniLM.OllamaSystemOneProtocol`](@ref) and [`OllamaModel`](@ref)
in the API reference, and [Ollama](@ref ollama_guide) in the Language models chapter.
