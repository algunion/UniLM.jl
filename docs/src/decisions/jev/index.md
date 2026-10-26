# [Jev (TypeSafe)](@id jev_start)

```@setup jev
using UniLM
```

Jev is TypeSafe's System One model: it reads a state — a text or a JSON record — and answers the questions you
list about it with a probability for each answer you allowed (a box's edges, for a Box question), writing no
text, and TypeSafe bills only the input tokens it reads. `TYPESAFEServiceEndpoint` speaks TypeSafe's API, `POST /v1/systemone` for decisions and
`GET /v1/models` for the model listing, and UniLM wraps both. The endpoint serves these two alone: the verb of
every capability the table below does not list, `chatrequest!` and `respond` among them, refuses it before
sending anything.

```@eval
Main.ProviderFacts.at_a_glance(:jev)
```

| I want to… | Use | You get |
| :--- | :--- | :--- |
| [make a first decision](@ref jev_first_decision) | `ask(state, "name" => choice(…))` | the chosen option, its probability, and the model that answered |
| [set the key, the API root or the default model](@ref jev_setup__next) | `TYPESAFE_API_KEY`, `TYPESAFE_BASE_URL`, `TYPESAFE_DEFAULT_MODEL` | every call that names no other provider, on Jev |
| [pin the version a threshold was tuned on](@ref jev_models__next) | `model = "jev-1.13.0"`, or `TYPESAFE_DEFAULT_MODEL` | answers from that version, wherever the alias moves |
| [ask several kinds of question at once](@ref jev_questions) | `choice`, `score`, `noul` and `box` in one `ask` | every answer from one request, the state read once |
| [know what is refused before a request is sent](@ref jev_limits__next) | the table of refusals | the words of the `ArgumentError`, and the fix |
| [judge a picture, or find something in it](@ref jev_images__next) | an `ImageAttachment` in the state; a `box` question | answers about the picture; a box's edges |
| [send a state longer than 32k tokens](@ref jev_long_context__next) | `long_context = true` | 262k tokens of context, at the same price per token |
| [act only when Jev is sure enough](@ref jev_confidence) | `min_confidence` and `fallback`, or a `decide` policy | the answer, or a hand-over to a person |
| [know what a decision cost](@ref jev_cost__next) | `token_usage`, `estimated_cost` | the input tokens billed and their USD estimate |
| [test without a key](@ref jev_testing) | `with_recorded_answers` | answers recorded once, replayed with no key and no network |
| [decide about many texts at once](@ref jev_throughput) | `ask_each(texts, questions...)` | one result per text, in input order |
| [handle a failure](@ref jev_errors__next) | the result's type | the status and TypeSafe's message, never an empty success |

## [Setup](@id jev_setup__next)

Get an API key from the [TypeSafe console](https://console.typesafe.ai/keys) and set it in the environment
Julia starts from:

```bash
export TYPESAFE_API_KEY="..."
```

| Variable | When unset | Meaning |
| :--- | :--- | :--- |
| `TYPESAFE_API_KEY` | a call sends nothing and returns a `SystemOneCallError` whose `error` reads `TYPESAFE_API_KEY is not set` | the key, sent as `Authorization: Bearer …` |
| `TYPESAFE_BASE_URL` | `https://api.typesafe.ai` | the API root: a proxy or a gateway in front of TypeSafe, or a server of your own in tests ([Testing and recordings](@ref jev_testing)) |
| `TYPESAFE_DEFAULT_MODEL` | the default model at the top of this page | the model of a call that names none ([Models](@ref jev_models__next)) |

UniLM reads the three when each request is sent, so a value exported after `using UniLM` takes effect, and a
blank value counts as unset. A call goes to Jev when it names no `service` and no [`with_system_one`](@ref)
scope around it names another provider; `service = TYPESAFEServiceEndpoint` names Jev explicitly, and
`TYPESAFEServiceEndpoint()` returns that same type. A call's model is its own `model = …`; else, when it names
no `service`, the model of the scope around it; else `TYPESAFE_DEFAULT_MODEL`, and the default model at the top
of this page when that is unset. The endpoint has no options of its own, and its one request option, `long_context`, has a section of
its own ([Long context](@ref jev_long_context__next)).

## [First decision](@id jev_first_decision)

One request carries the state once and every question about it; each answer comes back under the name you gave
its question:

```@example jev
const TEAM = (billing   = "payments, charges, invoices or refunds",
              technical = "the app or the website does not work as expected",
              shipping  = "a parcel that is late, lost or arrived damaged",
              other     = "anything else")
message = "I was charged twice for order #4471. Please refund the duplicate payment."

r = ask(message, "team" => choice("Which team should handle this message?", TEAM))
a = r["team"]
(a.choice, a.probabilities[a.choice], a.confidence)
```

`a.choice` is the most probable option, `a.probabilities` holds one probability per option, and `a.confidence`
says how peaked they are ([Confidence and tuning](@ref jev_confidence)). A failed call has no answers:
`r["team"]` throws a `SystemOneError` on it, so a decision is never read from a call that did not happen
([When something fails](@ref jev_errors__next)).

```@example jev
r.response.model, token_usage(r).prompt_tokens
```

The call named no model, so it was sent with the default model at the top of this page, an alias, and
`r.response.model` is the versioned id that answered ([Models](@ref jev_models__next)). [`nl_classify`](@ref) makes the same kind of decision in one call
and returns the key whose description fits ([Dispatch on Meaning](@ref nl_dispatch_guide)):

```@example jev
nl_classify(message, TEAM)
```

## [Models](@id jev_models__next)

TypeSafe serves Jev under aliases and versioned ids, and a call sends either as its `model`
([Models](https://docs.typesafe.ai/models)):

- `jev-latest`, the most recent stable release, and `jev-preview`, the most recent release of any kind, are
  aliases. An alias moves when a new release ships, so its answers can change with no change to
  your code. On 25 October 2026 TypeSafe's Models page points both at `jev-1.14.0` (Jev 1.14, available from
  14 October 2026) and says no preview build is available.
- A versioned id, `jev-1.14.0` or `jev-1.13.0`, names one release. The service accepts it whether or not the
  listing shows it. Both releases take pictures and Box questions need Jev 1.14, by TypeSafe's
  [Images](https://docs.typesafe.ai/images) page, though its Models and System One pages still list text input
  only; both take long context.

[`list_models`](@ref) returns the names the account may send, each with a description and a release date;
TypeSafe's listing holds the aliases:

```@example jev
m = list_models()
[(card.name, card.release_date) for card in m.models]
```

`release_date` is the string the service sends, an RFC 3339 timestamp, where TypeSafe's API reference documents
`YYYY-MM-DD` ([List models](https://docs.typesafe.ai/api/list-models)); UniLM does not parse it. A live listing
on 25 October 2026 returned these two cards unchanged.

`r.response.model` names the versioned id that answered, which TypeSafe's response reports in place of the
alias ([System One](https://docs.typesafe.ai/api/system-one)): log it, since an alias that moves is otherwise
invisible. A threshold tuned against one version is a claim about that version, so once you have tuned one, pin
the version — `model = …` on the call, `TYPESAFE_DEFAULT_MODEL` in the environment, or the `model` of a
`with_system_one` scope — and move the pin on your own schedule:

```@example jev
pinned = ask(message, "team" => choice("Which team should handle this message?", TEAM); model = "jev-1.13.0")
pinned.response.model
```

TypeSafe publishes the failure modes it knows of Jev 1.13
([Jev 1.13 jaggedness](https://docs.typesafe.ai/model-jaggedness/jev-1.13)); on 25 October 2026 its
documentation index lists no such page for Jev 1.14. Jev 1.13 reads a question literally, does not count,
compute or compare dates reliably, loses accuracy as unrelated material grows around the decision, and is not
trained to generate text.
Compute in Julia, send only what the decision needs, and leave text to a language model;
[Designing good questions](@ref jev_designing_questions) turns these into rules for writing questions.

## [Questions](@id jev_questions)

Jev answers four kinds of question, in any mix in one request: a [`choice`](@ref) picks one option of a set, a
[`score`](@ref) places the state on levels you describe, lowest first, a [`noul`](@ref) gives the probability that
a statement holds, and a [`box`](@ref) locates something in a picture ([Locate with Box](@ref jev_box__next)). A
[`RawQuestion`](@ref) goes out as you wrote it, for a kind UniLM does not model. Each question is evaluated against
the same state on its own: one answer is never context for another
([Primitives](https://docs.typesafe.ai/primitives)).

```@example jev
r = ask("The app crashes every time I open my order history. Third time this week, and my parcel is due today!",
        "team"    => choice("Which team should handle this message?", TEAM),
        "urgency" => score("How urgent is this message?",
                           ["Can wait", "Needs attention this week", "Needs attention today"]),
        "repeat"  => noul("Has the customer had this problem before?"))
(r["team"].choice, r["urgency"].score, r["repeat"].noul)
```

What reaches the model:

- **Question names do not.** On Jev, the names (`"team"`, `"urgency"`, `"repeat"`) only key the answers: TypeSafe
  does not show them to the model, so write the whole question in its instructions
  ([Primitives](https://docs.typesafe.ai/primitives)).
- **Option names do.** A Choice's option names and their descriptions both reach the model
  ([Choice](https://docs.typesafe.ai/primitives/choice)), so a short name can pull an answer toward it;
  [Choice, Score, Noul](@ref jev_primitives) shows when to send the descriptions themselves as the names.
- **Guidance may be structured.** Instructions, an option's description, a Score level and a Noul's yes and no may
  each be a string, an object or an array, and may hold a picture ([Structured guidance](@ref jev_structured)).

A Choice takes up to 255 options and a Score up to 10 levels, and TypeSafe advises at least two levels
([Score](https://docs.typesafe.ai/primitives/score)); a question over either limit is refused before the request
is sent ([Limits and refusals](@ref jev_limits__next)). A request may hold any number of questions as long as it
fits the context. Reading the answers and the decision rules on them are on
[Route and Decide](@ref system_one_guide).

## [Limits and refusals](@id jev_limits__next)

What a request meets on TypeSafe ([Models](https://docs.typesafe.ai/models),
[Request rules and errors](https://docs.typesafe.ai/api/errors), [Images](https://docs.typesafe.ai/images)), and
which of it UniLM checks before sending:

| Limit | Value | Checked before sending |
| :--- | :--- | :--- |
| Context | 64k tokens for the state and every question, 32k for the state with any one question; 262k for both with `long_context = true` ([Long context](@ref jev_long_context__next)) | no: the service answers 400 with `max_tokens_exceeded` |
| Questions | no limit of their own: the request must fit the context | — |
| Choice options | 1 to 255 per question | yes |
| Score levels | 1 to 10 per question | yes |
| Nesting | 64 levels of objects and arrays in each of the state, the instructions and the criteria | yes |
| Numbers | finite only | yes |
| Request body | 32 MiB, multipart framing included | yes |
| Each image | PNG or JPEG, a still image; 16 MiB per file; 280 to 1568 pixels on each side | yes |
| All images together, as TypeSafe re-encodes them; the request with every picture in place of each reference to it | 32 MiB each | no: the service answers 413 |
| Rate limits | per organization, in tokens and in requests per second | no ([When something fails](@ref jev_errors__next)) |

[`ask`](@ref), [`ask_each`](@ref) and the verbs built on them throw these `ArgumentError`s before anything is sent.
A wrapper refuses rather than adapts: a request changed to fit would ask a question you did not write, and its
answer would carry no sign of the change.

| Refused | Why: the message's words | What to do |
| :--- | :--- | :--- |
| a Choice of more than 255 options | `TypeSafe accepts at most 255 options per Choice question`: the service answers 422 for more | split it, or narrow the options |
| a Score of more than 10 levels | `TypeSafe accepts at most 10 levels per Score question`: the service answers 422 for more | merge levels into a coarser rubric |
| a question type of your own, a subtype of `SystemOneQuestion` | `its wire carries Choice, Score, Noul and Box questions`: TypeSafe's request has no other kind | build it as a `RawQuestion`, which is sent as written |
| a `box` question in a request that holds no image | `locates something in an image, and this request carries none`: a box drawn on no picture means nothing | put an `ImageAttachment` in the state ([Locate with Box](@ref jev_box__next)) |
| a provider option other than `long_context`, or a `long_context` that is not `true` or `false` | `TypeSafe does not accept the request option`, `option is`: the one option TypeSafe takes is `long_context`, a Boolean | `long_context = true`, or no option |
| a state, instructions or criteria nested more than 64 levels deep | `TypeSafe accepts at most 64 levels of nested objects and arrays`: the service answers 400 | flatten it |
| a request body over 32 MiB | `TypeSafe accepts a request body of at most 32 MiB`: the service answers 413 | shorten the state, send fewer or smaller images, or split the request |
| an image that is not PNG or JPEG | `TypeSafe accepts PNG and JPEG images only` | convert it to PNG or JPEG |
| an image file over 16 MiB | `TypeSafe accepts image files of at most 16 MiB` | resize it so neither side is over 1568 pixels, or save it compressed |
| an image under 280 or over 1568 pixels on a side | `TypeSafe accepts images of 280 to 1568 pixels on each side` | resize it, scaling up or padding a side that is too short |
| an animated PNG, an image whose header cannot be read or is missing, a PNG with no image data, a file cut short, or a JPEG that does not end with its end-of-image marker | `TypeSafe cannot take image`, then the reason: `it is an animated PNG`, `cannot be read`, `is missing`, `holds no image data`, `is cut short`, `does not end with the end-of-image marker` | export it again as a still PNG or JPEG |
| an image that only a struct of your own holds | `not inside a struct of your own`, `inside a value that is not a dictionary, vector, tuple or NamedTuple`: UniLM finds images in those containers alone | put the image in the state, the instructions or the criteria, or in a dictionary, vector, tuple or `NamedTuple` there |

Like every decision provider, the endpoint also refuses, before anything is sent, a blank or repeated question
name, criteria that are not valid, a state or guidance holding `NaN` or `Inf` (TypeSafe's request rules read
"Finite numbers only. `NaN`, `Infinity`, and numbers too large to represent, such as `1e400`, are rejected",
[Request rules and errors](https://docs.typesafe.ai/api/errors)), an `AudioAttachment` or a `DocumentAttachment`
anywhere in the request, and an `ImageAttachment` with a `detail` level; [Reading answers](@ref Reading-answers)
covers a malformed request on every provider.

```@example jev
try
    ask(message, "pick" => choice("Pick one.", ["option $i" for i in 1:256]))
catch err
    println(sprint(showerror, err))
end
```

Recorded on 25 October 2026 on `jev-1.14.0` and `jev-1.13.0` with UniLM's development branch after 0.26.0, and,
on 28 September 2026, the `nl_classify` call (on `jev-1.13.0`) with UniLM's development branch after 0.21.0 and
the model listing with UniLM's development branch after 0.20.1, through TypeSafe's API at `/v1`.

See also: [`TYPESAFEServiceEndpoint`](@ref), [`ask`](@ref), [`SystemOneSuccess`](@ref) and
[`TypeSafeModelsSuccess`](@ref) in the API reference.
