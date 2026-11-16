# [Gemini Decisions API](@id gemini_decisions_start)

```@setup gemini_decisions
using UniLM, JSON
```

Google's Decisions API reads an input — a text, a JSON value, pictures — and answers the yes/no, choice and score
questions you list about it with calibrated probabilities, writing no text; Google bills the input and no output
([Decisions API](https://ai.google.dev/gemini-api/docs/decisions-api)). The API is in Beta, on the Gemini API's
`v1beta` only, and its one model is Gemini Decision 1. `GEMINIServiceEndpoint`, the endpoint of the
[Google Gemini](@ref gemini_start) language-model entry, speaks it at `POST /v1beta/decisions` for [`ask`](@ref) and
the verbs built on it, with the same key; `GEMINIOpenAIServiceEndpoint` serves no decisions. Of the API, UniLM does
not wrap audio, video and PDF blocks in the input, Files API and HTTPS URIs, the `medium` and `ultra_high`
resolutions, or a Noul's confidence apart from its probability.

```@eval
Main.ProviderFacts.at_a_glance(:gemini_decisions)
```

| I want to… | Use | You get |
| :--- | :--- | :--- |
| [make a first decision](@ref gemini_decisions_first_decision) | `ask(state, "name" => choice(…); service = GEMINIServiceEndpoint)` | the chosen option, its probability, and the model that answered |
| [send every call of a block to Gemini](@ref gemini_decisions_setup) | `with_system_one(GEMINIServiceEndpoint) do … end` | each decision inside it, on the Decisions API |
| [ask several kinds of question at once](@ref gemini_decisions_questions) | `choice`, `score` and `noul` in one `ask` | every answer from one request |
| [decide about a picture](@ref gemini_decisions_images) | `[text, image]` as the state | answers about the picture, at the resolution its `detail` asks for |
| [handle an input Google's filters block](@ref gemini_decisions_blocked) | `a isa RefusalAnswer`; `fallback` on the natural-language verbs | a decline, never a guessed answer |
| [know what is refused before a request is sent](@ref gemini_decisions_limits) | the table of refusals | the words of the `ArgumentError`, and the fix |
| [act only when the answer is sure enough](@ref gemini_decisions_confidence) | `min_confidence` and `fallback`, or a `decide` policy | the answer, or a hand-over to a person |
| [know what a decision cost](@ref gemini_decisions_cost) | `token_usage`, `estimated_cost` | the input tokens and their USD estimate |
| [test without a key](@ref gemini_decisions_testing) | `with_recorded_answers` | answers recorded once, replayed with no key and no network |
| [handle a failure](@ref gemini_decisions_failures) | the result's type | the status and Google's message, never an empty success |

## [Setup](@id gemini_decisions_setup)

Set the API key of your Google AI Studio project in the environment Julia starts from:

```bash
export GEMINI_API_KEY="..."
```

A call goes to the Decisions API when it names `service = GEMINIServiceEndpoint`, or when it names no `service`
and a [`with_system_one`](@ref)`(GEMINIServiceEndpoint)` scope encloses it; a call that names neither goes to the
chapter's default provider ([Decisions](@ref decisions_start)). A call's model is its own `model = …`; else, when
it names no `service`, the scope's; else the default model at the top of this page. UniLM reads `GEMINI_API_KEY`
when each request is sent, so a key exported after `using UniLM` takes effect; with none, a call sends nothing and
returns a [`SystemOneCallError`](@ref) ([When something fails](@ref gemini_decisions_failures)). Requests go to
`https://generativelanguage.googleapis.com/v1beta/decisions`, and no variable moves them; the API has no `v1`
route (HTTP 404 on 16 November 2026).

The API takes no request options — no caching, batch, Flex or Priority, "Repeated input across requests counts in
full" ([Limitations](https://ai.google.dev/gemini-api/docs/decisions-api#limitations)) — so a keyword of
[`ask`](@ref) other than its own, and an option of a `with_system_one` scope, is refused before anything is sent
([Limits and refusals](@ref gemini_decisions_limits)).

## [First decision](@id gemini_decisions_first_decision)

One request carries the input once and every question about it; each answer comes back under the name you gave its
question ([How it works](https://ai.google.dev/gemini-api/docs/decisions-api#how-it-works)):

```@example gemini_decisions
const TEAM = (billing   = "payments, charges, invoices or refunds",
              technical = "the app or the website does not work as expected",
              shipping  = "a parcel that is late, lost or arrived damaged",
              other     = "anything else")
message = "I was charged twice for order #4471. Please refund the duplicate payment."

r = ask(message, "team" => choice("Which team should handle this message?", TEAM); service = GEMINIServiceEndpoint)
a = r["team"]
(a.choice, a.probabilities[a.choice], a.confidence)
```

`a.choice` is the most probable option, `a.probabilities` holds one probability per option, and `a.confidence`
says how peaked they are ([Confidence and tuning](@ref gemini_decisions_confidence)). A failed call has no answers:
`r["team"]` throws a `SystemOneError` on it ([When something fails](@ref gemini_decisions_failures)), and an input
Google's filters block comes back as a [`RefusalAnswer`](@ref) for every question, which has no `choice`
([Blocked inputs](@ref gemini_decisions_blocked)).

```@example gemini_decisions
r.response.model, token_usage(r).prompt_tokens, r.response.request_id
```

The call named no model, so it was sent with the default model at the top of this page; Google's reply names no
model, so `r.response.model` is the one the request named, and no header carries a request id, so
`r.response.request_id` is `nothing`. [`nl_classify`](@ref) makes the same kind of decision in one call and returns
the key whose description fits ([Dispatch on Meaning](@ref nl_dispatch_guide)):

```@example gemini_decisions
nl_classify(message, TEAM; service = GEMINIServiceEndpoint)
```

## [Models](@id gemini_decisions_models)

Google serves the Decisions API with one model, the default model at the top of this page: "Available on the Gemini
Developer API (v1beta) with Gemini Decision 1 only"
([Limitations](https://ai.google.dev/gemini-api/docs/decisions-api#limitations)); its
[model page](https://ai.google.dev/gemini-api/docs/models/gemini-decision-1) lists text, image, video, audio and PDF
inputs, an input token limit of 262,144, and calibrated probabilities with 0 output tokens. A call may name another
`model`, which UniLM sends as written: Google answered `models/gemini-decision-1` with HTTP 200, and an unknown
model with HTTP 404 and the code `not_found` (16 November 2026). The reply names no model, so `r.response.model` is
the name the request sent; [`estimated_cost`](@ref) prices the bare id, and the prefixed name has no price row of
its own ([Cost](@ref gemini_decisions_cost)).

[`list_models`](@ref)`(service = GEMINIServiceEndpoint)` lists every model Google serves the key, the language
models with this one ([Models](@ref gemini_models)); no recording holds it, so inside a replay scope it reaches
Google with your key ([Testing and recordings](@ref gemini_decisions_testing)).

## [Questions](@id gemini_decisions_questions)

The API answers three kinds of question, in any mix in one request
([Question types](https://ai.google.dev/gemini-api/docs/decisions-api#question-types)), and UniLM sends each
question type as the one that fits:

| You write | Google receives | You get |
| :--- | :--- | :--- |
| [`choice`](@ref) | a `choice` question: its `criteria` map each option's name to its description, `""` for `nothing` | a [`ChoiceAnswer`](@ref) |
| [`score`](@ref) | a `score` question, its `criteria` the levels, lowest first | a [`ScoreAnswer`](@ref), whose `legend` maps each level to its text |
| [`noul`](@ref) | a `noul` question, `yes` and `no` as the `true_description` and `false_description` of its `criteria` | a [`NoulAnswer`](@ref): the probability that the statement holds |
| [`RawQuestion`](@ref) | its fields as written, under its name | the answer Google sends for it |
| [`bounding_box`](@ref) | nothing: refused ([Limits and refusals](@ref gemini_decisions_limits)) | — |

The state becomes the request's `input`:

| State | Google receives |
| :--- | :--- |
| a string that is not blank | the string |
| a JSON value: a dictionary, an array, a tuple, a `NamedTuple` | the value's JSON text, a string, as the guide advises ([Structured data](https://ai.google.dev/gemini-api/docs/decisions-api#structured-data)) |
| an [`ImageAttachment`](@ref), or a vector or tuple of strings and images | `text` and `image` content blocks in their order ([Images](@ref gemini_decisions_images)) |

```@example gemini_decisions
q = ask("The app crashes every time I open my order history. Third time this week, and my parcel is due today!",
        "team"    => choice("Which team should handle this message?", TEAM),
        "urgency" => score("How urgent is this message?",
                           ["Can wait", "Needs attention this week", "Needs attention today"]),
        "repeat"  => noul("Has the customer had this problem before?");
        service = GEMINIServiceEndpoint)
(q["team"].choice, q["urgency"].score, q["repeat"].noul)
```

What reaches the model:

- **Question names do not.** "The model never sees question keys, so write the full question in the instructions
  field" ([Request and response reference](https://ai.google.dev/gemini-api/docs/decisions-api#request)). A name is
  1 to 64 ASCII letters, digits, `_` and `-`; [`nl_dispatch`](@ref) names each question after its argument, so an
  argument named outside that set is refused.
- **Instructions are text.** Every question needs instructions, and they are text only: a string, or a vector or
  tuple of strings, which UniLM sends as `text` blocks and Google joins "with no separator", so put your own spaces
  or newlines between them ([Input](https://ai.google.dev/gemini-api/docs/decisions-api#input)). A picture belongs
  in the state, and a JSON object is not turned into text for you: write `JSON.json(data)` to send its text.
- **Option names and descriptions travel, in Google's order.** "Before evaluation, the API sorts option keys
  alphabetically in UTF-8 byte order. Changing key order in your JSON request doesn't change option positions, and
  exact probability ties break to the alphabetically first key"
  ([Pick one option](https://ai.google.dev/gemini-api/docs/decisions-api#choice)). The order you pass is sent, and
  read by no model: a catch-all named `none_of_the_above` sorts ahead of `shipping`, which is why the guide names
  one `other`.
- **Other guidance is text.** An option's description, a Score level and a Noul's `yes` and `no` are strings; a
  level is not empty. Anything else is a refusal ([Limits and refusals](@ref gemini_decisions_limits)).

A Choice takes 2 to 255 options, a Score 2 to 255 levels (Google recommends 2 to 10), and a request 1 to 1,000
questions ([Limits](https://ai.google.dev/gemini-api/docs/decisions-api#limits)); a request outside these is refused
before it is sent ([Limits and refusals](@ref gemini_decisions_limits)). Where each provider takes a picture is on
[Choosing a decision provider](@ref system_one_backends). Reading the answers and the decision rules on them are on
[Route and Decide](@ref system_one_guide).

## [Images](@id gemini_decisions_images)

A picture goes in the state: an [`ImageAttachment`](@ref) alone, or a vector or tuple of strings and images, sent as
`image` and `text` blocks in their order, each image its base64 bytes and its MIME type. An image's `detail` is the
block's `resolution`: `"low"` and `"high"` are sent as written, and `"auto"`, Google's default, sends none
([Media resolution](https://ai.google.dev/gemini-api/docs/decisions-api#media-resolution)). The images count as
input at the same rate as text ([Cost](@ref gemini_decisions_cost)): the 266-byte red circle of this manual's
assets counted 1,089 tokens at the default resolution and 256 at `"low"` (16 November 2026).

```@example gemini_decisions
circle = ImageAttachment(joinpath(pkgdir(UniLM), "docs", "src", "assets", "red_circle.png"); detail = "low")
p = ask(["The customer sent this picture of the sticker they received:", circle],
        "colour" => choice("What colour is the sticker?", ["red", "green", "blue"]);
        service = GEMINIServiceEndpoint)
(p["colour"].choice, token_usage(p).prompt_tokens)
```

A picture anywhere else — inside a JSON state, in a question's instructions, an option or a level — is refused,
since instructions and descriptions are text; so are `detail = "original"`, which has no Gemini resolution, and
images of more than 100 MB in all ([Limits and refusals](@ref gemini_decisions_limits)). A text block is not empty,
though one of whitespace alone, a separator, is sent.

## [Blocked inputs](@id gemini_decisions_blocked)

Google answers no question of an input its filters block, and documents two shapes for it. The API reference's: a
200 whose `prompt_feedback.block_reason` is set, when "no decisions were evaluated, and `answers` is empty"
([CreateDecisionResponse](https://ai.google.dev/api/decisions)); UniLM returns it as a success whose every answer is a
[`RefusalAnswer`](@ref), its `raw` the `block_reason` (`safety`, `other`, `blocklist`, `prohibited_content`,
`image_safety`) and the `block_reason_message`. The guide's: an HTTP 400 whose error `code` is `safety`,
`prohibited_content`, `blocklist`, `image_safety` or `content_blocked`
([Request and response reference](https://ai.google.dev/gemini-api/docs/decisions-api#request)); UniLM returns a
[`SystemOneFailure`](@ref) whose `error_type` is that code. Test `a isa RefusalAnswer` before you read `a.choice`,
`a.score` or `a.noul`. [`nl_classify`](@ref), [`nl_dispatch`](@ref) and [`@branch`](@ref) take a refusal as a
decline whatever their policy, which never sees it: `fallback` runs (the `_` line, in `@branch`), or
[`DecisionDeclinedError`](@ref) is thrown with the refusal as its `answer`; a failure throws a `SystemOneError`.

No recording can be made to hold a blocked input without sending content a filter blocks, so to test what your code
does with one, declare an endpoint of your own that speaks this API's wire to a local server, through the protocol
seam ([Write your own decision provider](@ref decisions_custom)); `_resolve_base_url` is an internal hook. Run such
tests outside `with_recorded_answers`. This block is not run when the manual is built:

```julia
using UniLM, HTTP, JSON, Test

struct DecisionsMock <: UniLM.ServiceEndpoint end
UniLM.provider_capabilities(::Type{DecisionsMock}) = Set([:system_one])
UniLM.system_one_protocol(::Type{DecisionsMock}) = UniLM.GeminiDecisionsProtocol()
UniLM._resolve_base_url(::Type{DecisionsMock}) = "http://127.0.0.1:8125"     # internal hook
UniLM.auth_header(::Type{DecisionsMock}) = ["Content-Type" => "application/json"]

TEAM = (billing = "payments, charges, invoices or refunds", other = "anything else")
server = HTTP.serve!("127.0.0.1", 8125; verbose = false) do req
    # A blocked input, as Google's API reference shapes it: no answers, the reason in prompt_feedback.
    body = (prompt_feedback = (block_reason = "safety", block_reason_message = "Blocked."),
            usage = (total_input_tokens = 42, total_output_tokens = 0))
    HTTP.Response(200, ["Content-Type" => "application/json"], JSON.json(body))
end

@testset "a blocked input" begin
    @test nl_classify("Where is my parcel?", TEAM; service = DecisionsMock, fallback = m -> :person) === :person
    @test_throws DecisionDeclinedError nl_classify("Where is my parcel?", TEAM; service = DecisionsMock)
    r = ask("Where is my parcel?", "classify" => choice("Which team?", TEAM); service = DecisionsMock)
    @test r["classify"] isa RefusalAnswer && r["classify"].raw["block_reason"] == "safety"
end
close(server)
```

## [Limits and refusals](@id gemini_decisions_limits)

What a request meets on the Decisions API ([Limits](https://ai.google.dev/gemini-api/docs/decisions-api#limits),
[Input](https://ai.google.dev/gemini-api/docs/decisions-api#input)), and which of it UniLM checks before sending.
Google answers a request over a limit with HTTP 400 "naming the rule you exceeded"; each row UniLM checks was
answered so on 16 November 2026, or, for the 100 MB, is read from the guide:

| Limit | Value | Checked before sending |
| :--- | :--- | :--- |
| Questions | 1 to 1,000 per request | yes |
| Question names | 1 to 64 ASCII letters, digits, `_` and `-` | yes |
| Choice options | 2 to 255 per question; a name of 1 to 128 characters and at most 512 bytes, with no leading or trailing whitespace and no control, format or non-ASCII space characters | yes |
| Score levels | 2 to 255 per question, each a non-empty string | yes |
| Instructions | text only, at least one non-empty block | yes |
| Input | not empty: a string that is not blank, no empty text block | yes |
| Inline data | "up to 100 MB per request", read as 100 MiB of image bytes (untested at the limit) | yes |
| Context | 262,144 tokens across the input and all questions | no: Google counts them |
| Rate limits | per project and tier ([Rate limits](https://ai.google.dev/gemini-api/docs/rate-limits)) | no ([When something fails](@ref gemini_decisions_failures)) |

[`ask`](@ref), [`ask_each`](@ref) and the verbs built on them throw these `ArgumentError`s before anything is sent.
A wrapper refuses rather than adapts: a request changed to fit would ask a question you did not write, and its
answer would carry no sign of the change.

| Refused | Why: the message's words | What to do |
| :--- | :--- | :--- |
| more than 1,000 questions | `Gemini's Decisions API takes at most 1000 questions per request` | split them across requests |
| a question name outside 1 to 64 ASCII letters, digits, `_` and `-` | `takes a question's name as 1 to 64 ASCII letters, digits`: Google keys the questions by it | rename the question; on `nl_dispatch`, the argument |
| a question without instructions, or with empty ones | `every question needs`, `its instructions are empty`, `its instructions hold no text`: Google requires non-empty instructions, a Noul's too | write the question, as the first argument of `choice`, `score` or `noul` |
| instructions that are not text: an object, a picture, a vector holding anything but strings | `the API takes instructions as text only`: "Question instructions accept text only" | write them as text; for data, `JSON.json(data)`; a picture goes in the state |
| an empty text block, in the state or in the instructions | `takes no empty text block`: Google answers one with HTTP 400 | drop it, or write the text |
| a Choice of fewer than 2 or more than 255 options | `a choice offers 2 to 255 options, and it has` | add an option such as `"other"`, or narrow the options |
| an option name over 128 characters or 512 bytes — such as an `nl_classify` sentence | `an option's name takes at most 128 characters and 512 bytes` | a short option name, with the sentence as its description: `ask` with `choice(…, ["refund" => "the customer wants a refund"])` |
| an option name with leading or trailing whitespace | `begins or ends with whitespace` | trim it |
| an option name with a control or format character, or a space other than the ASCII one | `holds a control or format character, or a space other than the ASCII one` | remove it |
| an option described by anything but a string or `nothing` | `the description of its option`: a description is a string | describe it in words |
| a Score of fewer than 2 or more than 255 levels | `a score has 2 to 255 levels, and it has` | add a level, or merge levels into a coarser rubric |
| a Score level that is not a non-empty string | `must be a non-empty string, its description` | write the level as text |
| a Noul `yes` or `no` that is not a string | `criterion must be a string` | write it as text |
| a `bounding_box` question | `it is a Box question, and the API answers noul, choice and score questions only` | ask it of a provider that answers Box questions |
| a question type of your own, a subtype of `SystemOneQuestion` | `the API answers noul, choice and score questions; a RawQuestion is sent verbatim` | build it as a `RawQuestion` |
| a `RawQuestion` that holds a picture | `it holds an image, and a question's instructions are text only` | put the picture in the state |
| a blank state, or an empty vector or tuple | `takes no empty input`: "Input can't be empty" | pass the text the questions are about |
| a vector or tuple state that holds a [`Message`](@ref) | `takes no conversation as its input`: "Serialize multi-turn context as text" | pass the conversation's text |
| a picture inside a JSON state | `takes images in the state only as`: a JSON state is sent as its text, which holds no picture | pass the texts and the pictures as a vector or tuple of strings and images |
| an image of `detail = "original"` | `Gemini has no media resolution for an image at its own dimensions` | `"high"`, `"low"` or `"auto"` |
| images of more than 100 MiB in all | `takes at most 104857600 bytes of inline data per request` | downscale them, or split them across requests |
| an `AudioAttachment` anywhere in the request | `UniLM does not send audio in a decision's input`: Google takes audio blocks, which UniLM's System One state does not carry | put the clip's transcript in the request |
| any provider option | `Gemini's Decisions API takes no provider options` | drop it |

Like every decision provider, the endpoint also refuses, before anything is sent, a blank or repeated question name,
criteria that are not valid, a state or guidance holding `NaN` or `Inf`, and a `DocumentAttachment` anywhere in the
request; [Reading answers](@ref Reading-answers) covers a malformed request on every provider. An image's `detail`
level goes out as its block's `resolution`, as [Images](@ref gemini_decisions_images) says.

```@example gemini_decisions
for refused in (() -> ask(message, "team" => choice("Which team should handle this message?", ["billing"]);
                          service = GEMINIServiceEndpoint),
                () -> ask(message, "team" => choice(TEAM); service = GEMINIServiceEndpoint),
                () -> nl_classify(message, (refund = "the customer wants the duplicate payment of their order " *
                                                     "refunded to the card they paid with, as soon as possible, " *
                                                     "and a written confirmation of the refund",
                                            other = "anything else"); service = GEMINIServiceEndpoint))
    try
        refused()
    catch err
        println(sprint(showerror, err))
    end
end
```

## [Confidence and tuning](@id gemini_decisions_confidence)

A Noul's answer is the probability that its statement holds, from 0.01 to 0.99 ("a check above 0.99 never
matches"); a Choice or a Score answer carries a probability distribution, and Google's own `confidence` is "the top
outcome's probability, or `max(noul, 1 - noul)` for yes/no questions"
([Answers, probabilities, and confidence](https://ai.google.dev/gemini-api/docs/decisions-api#confidence)). UniLM
recomputes a Choice's and a Score's `confidence` with the formulas of [Reading answers](@ref Reading-answers), so a
threshold means the same on every provider, and keeps Google's value in the answer's `raw`:

```@example gemini_decisions
(a.confidence, a.raw["confidence"], a.probabilities[a.choice])
```

The replies give every probability unrounded, so `ask` accepts a distribution that sums to 1 within `1e-6`, and a
chosen option no more than that below the most probable; a reply further off is a [`SystemOneCallError`](@ref). The
guide's own examples are rounded to two or three decimals, and two of them sum to 0.999 and 1.01: a reply like them
would be a `SystemOneCallError` here.

!!! details "Evidence"
    Measured on 16 November 2026 on `gemini-decision-1`, 13 replies with HTTP 200: every probability carried 15 or
    16 decimals; the 9 distributions of 3 or 4 outcomes summed to 1 within 2.3e-16; each Choice and Score
    `confidence` equalled the top probability, and each Noul's `max(noul, 1 - noul)`, exactly; a score equalled
    the expectation of its probabilities exactly. The same 4-option request sent four times gave the same
    probabilities each time, in another key order. Latency 0.26 s to 0.72 s, median 0.36 s. Limits: one day, one
    key, short English inputs.

The bar is yours: below `min_confidence`, a natural-language verb returns what `fallback` returns, and with no
`fallback` it throws `LowConfidenceError`. Google advises "thresholds using labeled examples from your own traffic"
and weighing "the cost of a false positive against a false negative"; where some mistakes cost more than others,
price each one and let a `decide` policy choose the cheapest action
([When mistakes have different prices](@ref jev_loss_matrix)). The decision rules on every provider are on
[Act on the answer](@ref jev_act_on_answer).

## [Cost](@id gemini_decisions_cost)

Google bills a decision's input tokens, at one rate for text, images, audio and video, and no output, which is
always 0 tokens: "Each request bills shared input tokens once, regardless of question count. Each question adds
tokens for its own instructions, criteria, and formatting"
([Pricing](https://ai.google.dev/gemini-api/docs/pricing#gemini-decision)). The rate is the default model's
row of [Cost and usage](@ref gemini_cost).

```@example gemini_decisions
u = token_usage(r)
(u.prompt_tokens, u.completion_tokens, estimated_cost(r))
```

`prompt_tokens` is Google's `total_input_tokens` — images, video frames and PDF pages count as image tokens —,
`completion_tokens` its `total_output_tokens`, `cached_tokens` its `total_cached_tokens` (the API caches nothing),
and `prompt_audio_tokens` the audio among the input. [`estimated_cost`](@ref) bills `prompt_tokens` at the row's
input rate and no output. The free tier bills nothing, which no response shows. Totals across calls and prices of
your own are on [Cost Tracking](@ref cost_guide).

## [Testing and recordings](@id gemini_decisions_testing)

[`with_recorded_answers`](@ref) records an exchange with the Decisions API once and replays it with no key and no
network: `ask`, `ask_each` and the verbs built on them. A recording is filed under
`/gemini-decisions/v1beta/decisions`, the request's path behind the namespace `gemini-decisions`, so a question
asked of two providers is two recordings, and it is keyed on the exact request: its method, that path, and its
body, which names the model. A picture is keyed and written as `data:<mime>;sha256,<hex>`, the SHA-256 of its base64
text, never its bytes.

```@example gemini_decisions
recordings = mktempdir()      # in a package: a directory committed next to the tests
recorded = with_recorded_answers(recordings; mode = :record) do
    ask(message, "team" => choice("Which team should handle this message?", TEAM); service = GEMINIServiceEndpoint)
end
JSON.parsefile(only(readdir(recordings; join = true)))["request"]["path"]
```

```@example gemini_decisions
replayed = withenv("GEMINI_API_KEY" => nothing) do
    with_recorded_answers(recordings) do
        ask(message, "team" => choice("Which team should handle this message?", TEAM); service = GEMINIServiceEndpoint)
    end
end
replayed["team"].probabilities == recorded["team"].probabilities
```

Recording takes `GEMINI_API_KEY`, and replaying takes nothing; in this manual's build the recording scope above
replays the manual's own recordings, so the example reaches no service either
([How this manual runs its examples](@ref jev_testing_manual)). [`list_models`](@ref) on this endpoint is not
recorded: inside a scope it goes to Google, with the key, as outside one. A blocked input cannot be recorded; a
local server can stand in for it ([Blocked inputs](@ref gemini_decisions_blocked)). Previews of what a call sends,
tests of the methods and of the judgment, and an audit trail are on [Test and Develop](@ref jev_testing_guide).

## [When something fails](@id gemini_decisions_failures)

A call that reaches Google returns a [`SystemOneSuccess`](@ref); a [`SystemOneFailure`](@ref) for a status other
than 200, with the `status`, Google's `message` and its `error_type` — the error's `code` when it is a string, such
as `"invalid_request"`, else its `status`, such as `"INVALID_ARGUMENT"`; or a [`SystemOneCallError`](@ref) when no
usable response came. A 200 whose answers do not fit the request — a question unanswered, an option that was not
offered, probabilities that are no distribution — is a `SystemOneCallError` too. Reading an answer of a failed call
(`r["name"]`, [`answer`](@ref), [`answers`](@ref)) throws a [`SystemOneError`](@ref).

UniLM sends a request again after a 408, 429, 500, 502, 503, 504, 520 to 524 or 529, up to `max_attempts` times
within `total_deadline`, waiting at least what the reply's `Retry-After` header asks for, as the shared backoff of
[`ask_each`](@ref) does; any other status comes back as it is. With no [`RequestConfig`](@ref) of your own, a call
to the Decisions API is bounded by 60 s per attempt and 150 s in all, and [`ask_each`](@ref) keeps 8 requests in
flight unless its `ntasks` says otherwise ([Many requests at once](@ref jev_items_each)); a config of your own
replaces the bounds, per call or for a scope:

```@example gemini_decisions
scoped = with_request_config(request_timeout = 20.0, max_attempts = 5) do
    ask(message, "team" => choice("Which team should handle this message?", TEAM); service = GEMINIServiceEndpoint)
end
issuccess(scoped)
```

Google's rate limits for Gemini Decision 1 are per project and tier: 200 requests a minute on the free tier and 1,500
on Tier 1 ([Rate limits](https://ai.google.dev/gemini-api/docs/rate-limits)); 8 requests in flight at about 0.4 s each
stay under Tier 1's.

| What you see | Why | What to do |
| :--- | :--- | :--- |
| a `SystemOneCallError` whose `error` reads `KeyError: key "GEMINI_API_KEY" not found` | no key in the environment; nothing was sent | set it ([Setup](@ref gemini_decisions_setup)) |
| a `SystemOneFailure` with status 400 and `error_type` `"INVALID_ARGUMENT"`: `API key not valid` | the key is not valid | a key from your Google AI Studio project |
| a `SystemOneFailure` with status 404, `error_type` `"not_found"`: `Requested model not found` | a model the API does not serve ([Models](@ref gemini_decisions_models)) | name no model |
| a `SystemOneFailure` with status 400, `error_type` `"invalid_request"`, whose message names a field of the request | Google refuses what UniLM's checks let through, such as a `RawQuestion` of a type it does not answer | the field the message names |
| a `SystemOneFailure` with status 400 and `error_type` `"safety"`, `"prohibited_content"`, `"blocklist"`, `"image_safety"` or `"content_blocked"` | Google's input filters blocked the request ([Blocked inputs](@ref gemini_decisions_blocked)) | another input |
| status 429 after the retries, with `retry_after` when Google sent one | over a rate limit | fewer requests in flight (`ntasks`), or a higher tier |
| status 500, 503 or another retried status, after the retries | a failure on Google's side | a larger `max_attempts` or `total_deadline` |
| a `SystemOneCallError` whose `cause` is a `UniLMTimeout` or a `UniLMCancelled` | a timeout, or a cancelled token | a `RequestConfig` of your own ([Timeouts & Retries](@ref timeouts_guide)) |
| a `SystemOneCallError` after a 200 | an answer that does not fit the request | keep the request: the reply carries no id |
| a `ReplayMissError`, thrown | in a replay scope, a request with no recording | record it with `GEMINI_API_KEY` ([Testing and recordings](@ref gemini_decisions_testing)) |
| a `SystemOneError`, thrown | an answer read from a failed call | test `issuccess(r)` first |
| a `FieldError`, thrown: `type RefusalAnswer has no field` | a blocked input's answer read as an answer | test `a isa RefusalAnswer` first ([Blocked inputs](@ref gemini_decisions_blocked)) |

Recorded on 16 November 2026 on `gemini-decision-1` with UniLM's development branch after 0.32.0, through Google's
Gemini API at `/v1beta/decisions`.

See also: [`GEMINIServiceEndpoint`](@ref), [`ask`](@ref), [`RefusalAnswer`](@ref) and
[`UniLM.GeminiDecisionsProtocol`](@ref) in the API reference, and [Google Gemini](@ref gemini_start) in the Language
models chapter.
