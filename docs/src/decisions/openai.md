# [OpenAI Decisions API](@id openai_decisions_start)

```@setup openai_decisions
using UniLM, JSON
```

OpenAI's Decisions API reads an input — a text, a JSON value, pictures — and answers the predicate, choice and
score questions you list about it with a probability for each answer you allowed, writing no text; OpenAI bills
the input and no output ([Decisions guide](https://developers.openai.com/api/docs/guides/decisions)).
`OPENAIServiceEndpoint`, the endpoint of the [OpenAI](@ref openai_start) language-model entry, speaks it at
`POST /v1/decisions` for [`ask`](@ref) and the verbs built on it, with the same key, and no decision verb refuses
the endpoint. Of the API, UniLM does not wrap a conversation as input (messages, tool calls and their outputs),
`input_json` parts, image URLs, pictures in a question's instructions, a score level's
description, explicit cache breakpoints, the organization's Decisions usage endpoint or GPT-Live's Decisions
integration.

```@eval
Main.ProviderFacts.at_a_glance(:openai_decisions)
```

| I want to… | Use | You get |
| :--- | :--- | :--- |
| [make a first decision](@ref openai_decisions_first_decision) | `ask(state, "name" => choice(…); service = OPENAIServiceEndpoint)` | the chosen option, its probability, and the model that answered |
| [send every call of a block to OpenAI](@ref openai_decisions_setup) | `with_system_one(OPENAIServiceEndpoint) do … end` | each decision inside it, on the Decisions API |
| [partition OpenAI's prompt cache, or name my end user](@ref openai_decisions_options) | `prompt_cache_key`, `safety_identifier`, `prompt_cache_options` | request fields that change no answer |
| [ask several kinds of question at once](@ref openai_decisions_questions) | `choice`, `score` and `noul` in one `ask` | every answer from one request |
| [handle a question OpenAI declines](@ref openai_decisions_refusals) | `a isa RefusalAnswer`; `fallback` on the natural-language verbs | a decline, never a guessed answer |
| [know what is refused before a request is sent](@ref openai_decisions_limits) | the table of refusals | the words of the `ArgumentError`, and the fix |
| [act only when the answer is sure enough](@ref openai_decisions_confidence) | `min_confidence` and `fallback`, or a `decide` policy | the answer, or a hand-over to a person |
| [know what a decision cost](@ref openai_decisions_cost) | `token_usage`, `estimated_cost` | the input tokens and their USD estimate |
| [test without a key](@ref openai_decisions_testing) | `with_recorded_answers` | answers recorded once, replayed with no key and no network |
| [handle a failure](@ref openai_decisions_failures) | the result's type | the status and OpenAI's message, never an empty success |

## [Setup](@id openai_decisions_setup)

Set the API key of your OpenAI account in the environment Julia starts from:

```bash
export OPENAI_API_KEY="..."
```

A call goes to the Decisions API when it names `service = OPENAIServiceEndpoint`, or when it names no `service`
and a [`with_system_one`](@ref)`(OPENAIServiceEndpoint)` scope encloses it; a call that names neither goes to the
chapter's default provider ([Decisions](@ref decisions_start)). A call's model is its own `model = …`; else, when
it names no `service`, the scope's; else the default model at the top of this page. UniLM reads `OPENAI_API_KEY`
when each request is sent, so a key exported after `using UniLM` takes effect; with none, a call sends nothing and
returns a [`SystemOneCallError`](@ref) ([When something fails](@ref openai_decisions_failures)). Requests go to
`https://api.openai.com`, and no variable moves them.

OpenAI's guide, fetched on 25 October 2026, opens with "The Decisions API is now generally available" and lists
what the API gained "since the public beta": prompt caching, pictures in question instructions, JSON input, up to
400 questions per request, image URLs, conversation history, usage reporting, GPT-Live's integration and Private
Link ([What's new](https://developers.openai.com/api/docs/guides/decisions#whats-new)). UniLM sends up to 400
questions, and still refuses what the beta refused: a picture anywhere but the state
([Limits and refusals](@ref openai_decisions_limits)).

## [Options](@id openai_decisions_options)

The endpoint takes no constructor arguments. Three request options of the API are keywords of [`ask`](@ref) and
[`ask_each`](@ref), `options = (…)` on the natural-language verbs, or keywords of `with_system_one` for a scope;
each is sent as a field of the request
([Create a decision](https://developers.openai.com/api/reference/resources/decisions/methods/create)):

| Keyword | What OpenAI does with it | Checked before sending |
| :--- | :--- | :--- |
| `prompt_cache_key` | "Partition prompt cache reuse by caller-provided key." | a `String` of at most 64 characters |
| `safety_identifier` | an "opaque caller-provided end-user identifier, scoped by the verified org" | a `String` of at most 128 characters |
| `prompt_cache_options` | its `mode` and `ttl` choose the cache breakpoints and how long they last | a [`PromptCacheOptions`](@ref) of `mode` and a `ttl` of `"30m"` or `"12h"` |

OpenAI reuses a request's shared input once it holds at least 1,024 tokens, with a breakpoint at its end unless
`mode = "explicit"`, and the questions are outside the reusable prefix
([Reuse shared input with prompt caching](https://developers.openai.com/api/docs/guides/decisions#reuse-shared-input-with-prompt-caching)).
UniLM marks no breakpoint, so with `mode = "explicit"` a request uses no cache. OpenAI's reference calls `"30m"`
"currently the only supported value" of `ttl`; UniLM also sends `"12h"`, which OpenAI answered with HTTP 200 on
25 October 2026. None of the three changes an answer, so none is part of a recording's key
([Testing and recordings](@ref openai_decisions_testing)), and a value outside the last column is refused before
anything is sent ([Limits and refusals](@ref openai_decisions_limits)).

## [First decision](@id openai_decisions_first_decision)

One request carries the input once and every question about it; each answer comes back under the name you gave its
question, which the API echoes
([How decisions work](https://developers.openai.com/api/docs/guides/decisions#how-decisions-work)):

```@example openai_decisions
const TEAM = (billing   = "payments, charges, invoices or refunds",
              technical = "the app or the website does not work as expected",
              shipping  = "a parcel that is late, lost or arrived damaged",
              other     = "anything else")
message = "I was charged twice for order #4471. Please refund the duplicate payment."

r = ask(message, "team" => choice("Which team should handle this message?", TEAM); service = OPENAIServiceEndpoint)
a = r["team"]
(a.choice, a.probabilities[a.choice], a.confidence)
```

`a.choice` is the most probable option, `a.probabilities` holds one probability per option, and `a.confidence`
says how peaked they are ([Confidence and tuning](@ref openai_decisions_confidence)). A failed call has no answers:
`r["team"]` throws a `SystemOneError` on it ([When something fails](@ref openai_decisions_failures)), and a
question OpenAI declines comes back as a [`RefusalAnswer`](@ref), which has no `choice`
([Declined questions](@ref openai_decisions_refusals)).

```@example openai_decisions
r.response.model, token_usage(r).prompt_tokens, r.response.request_id
```

The call named no model, so it was sent with the default model at the top of this page, and `r.response.model` is
the model OpenAI's reply names; `r.response.request_id` is the reply's `x-request-id` header.
[`nl_classify`](@ref) makes the same kind of decision in one call and returns the key whose description fits
([Dispatch on Meaning](@ref nl_dispatch_guide)):

```@example openai_decisions
nl_classify(message, TEAM; service = OPENAIServiceEndpoint)
```

## [Models](@id openai_decisions_models)

OpenAI serves the Decisions API with one model, the default model at the top of this page: the guide calls it "the
only currently supported model", and its [model page](https://developers.openai.com/api/docs/models/gpt-6-luna)
gives one snapshot, under the same name. A call may name another `model`; UniLM sends it, and on 25 October 2026
OpenAI answered `gpt-6-sol` with HTTP 404 and the code `model_not_found`
([When something fails](@ref openai_decisions_failures)). Each reply names the model that answered, and
`r.response.model` holds it: log it with each decision, since a threshold tuned on one model is a claim about that
model ([Confidence and tuning](@ref openai_decisions_confidence)).

[`list_models`](@ref)`(service = OPENAIServiceEndpoint)` lists every model OpenAI serves the key, the language
models with this one ([Models](@ref openai_models)); no recording holds it, so inside a replay scope it reaches
OpenAI with your key ([Testing and recordings](@ref openai_decisions_testing)).

## [Questions](@id openai_decisions_questions)

The API answers three kinds of question, in any mix in one request
([Choose a question type](https://developers.openai.com/api/docs/guides/decisions#choose-a-question-type)), and
UniLM sends each question type as the one that fits:

| You write | OpenAI receives | You get |
| :--- | :--- | :--- |
| [`choice`](@ref) | a `choice` question: each option's name as its `value`, a string description as its `description` | a [`ChoiceAnswer`](@ref) |
| [`score`](@ref) | a `score` question, each level a `label`, lowest first | a [`ScoreAnswer`](@ref), whose `legend` maps each level to its label |
| [`noul`](@ref) | a `predicate` question | a [`NoulAnswer`](@ref): the probability that the statement holds |
| [`RawQuestion`](@ref) | its fields as written, with the question's name added as `name` | the answer OpenAI sends for it |
| [`box`](@ref) | nothing: refused ([Limits and refusals](@ref openai_decisions_limits)) | — |

The state becomes the request's `input`:

| State | OpenAI receives |
| :--- | :--- |
| a string | the string |
| a JSON value: a dictionary, an array, a tuple, a `NamedTuple` | the value's JSON text, a string |
| an [`ImageAttachment`](@ref), or a `Vector` of strings and images | one user message of `input_text` and `input_image` parts in their order, each image a base64 data URL, with its `detail` level when it sets one |

```@example openai_decisions
q = ask("The app crashes every time I open my order history. Third time this week, and my parcel is due today!",
        "team"    => choice("Which team should handle this message?", TEAM),
        "urgency" => score("How urgent is this message?",
                           ["Can wait", "Needs attention this week", "Needs attention today"]),
        "repeat"  => noul("Has the customer had this problem before?");
        service = OPENAIServiceEndpoint)
(q["team"].choice, q["urgency"].score, q["repeat"].noul)
```

What reaches the model:

- **Question names travel.** Each question carries its name, which the API echoes so that UniLM matches each answer
  to its question. OpenAI's pages do not say whether the model reads it, so write the whole question in its
  instructions.
- **Option names and descriptions travel.** A Choice's option names and their descriptions are both in the request;
  OpenAI's guide asks for "distinct values and descriptions that explain when each option applies"
  ([Select from fixed options](https://developers.openai.com/api/docs/guides/decisions#select-from-fixed-options)).
- **Guidance is text.** Every question needs instructions, a string; an option's description and a Score level are
  strings; and a predicate has no place for a Noul's `yes` and `no`, so fold them into its instructions. Each is a
  refusal ([Limits and refusals](@ref openai_decisions_limits)).

A Choice takes 2 to 255 options
([Create a decision](https://developers.openai.com/api/reference/resources/decisions/methods/create)) and a Score 2 to
10 levels ([Score against a rubric](https://developers.openai.com/api/docs/guides/decisions#score-against-a-rubric)), and a
request 1 to 400 questions
([Ask multiple questions](https://developers.openai.com/api/docs/guides/decisions#ask-multiple-questions)); a
request outside these is refused before it is sent ([Limits and refusals](@ref openai_decisions_limits)). A picture
goes in the state only, as the table shows; where each provider takes one is on
[Choosing a decision provider](@ref system_one_backends). Reading the answers and the decision rules on them
are on [Route and Decide](@ref system_one_guide).

## [Declined questions](@id openai_decisions_refusals)

OpenAI may decline any question: "Any question can return a refusal instead of an answer; other questions can still
receive answers" ([Create a decision](https://developers.openai.com/api/reference/resources/decisions/methods/create)).
UniLM returns such an answer as a [`RefusalAnswer`](@ref), whose `raw` holds what OpenAI sent and which has no value
to read: after [`ask`](@ref), test `a isa RefusalAnswer` before you read `a.choice`, `a.score` or `a.noul`; the
other answers of the request read as usual. [`nl_classify`](@ref), [`nl_dispatch`](@ref) and [`@branch`](@ref)
take a refusal as a decline whatever their policy, which never sees it: `fallback` runs (the `_` line, in
`@branch`), or [`DecisionDeclinedError`](@ref) is thrown with the refusal as its `answer`.

```@example openai_decisions
route(::nl"the customer wants a refund", msg)           = :refunds
route(::nl"the customer reports a bug in the app", msg) = :engineering
route(::nl"anything else", msg)                         = :frontline

nl_dispatch(route, message; service = OPENAIServiceEndpoint, fallback = msg -> :person)
```

```@example openai_decisions
@branch message service = OPENAIServiceEndpoint min_confidence = 0.5 begin
    "the customer wants a refund"           => :refunds
    "the customer reports a bug in the app" => :engineering
    _                                       => :person
end
```

Had OpenAI declined either question, the `fallback` and the `_` line would have answered `:person`.
[`nl_dispatch`](@ref) sends the arguments it dispatches on as a JSON object, which OpenAI receives as its JSON text;
a picture among them is refused, since a picture goes in the state only: pass `state = [image, "…"]` to send one
([Limits and refusals](@ref openai_decisions_limits)). A verb whose call offers a single option is refused too, since a
Choice takes two or more ([Limits and refusals](@ref openai_decisions_limits)): add a catch-all such as
`"anything else"`, which OpenAI's guide advises "when your categories don't cover every possible input"
([Select from fixed options](https://developers.openai.com/api/docs/guides/decisions#select-from-fixed-options)).

## [Limits and refusals](@id openai_decisions_limits)

What a request meets on the Decisions API
([Create a decision](https://developers.openai.com/api/reference/resources/decisions/methods/create),
[Decisions guide](https://developers.openai.com/api/docs/guides/decisions)), and which of it UniLM checks before
sending:

| Limit | Value | Checked before sending |
| :--- | :--- | :--- |
| Questions | 1 to 400 per request | yes |
| Choice options | 2 to 255 per question | yes |
| Score levels | 2 to 10 per question | yes |
| Text | 10,485,760 characters for the input, and for each text part of a message | yes |
| Guidance | 1,048,576 characters for a question's instructions and its name, an option's value and description, a level's label, and the model's name | yes |
| Images | in the input, or in a question's instructions; no number on OpenAI's pages | in the input only, at most 128: UniLM refuses more |
| Rate limits | per organization and project, varying by model | no ([When something fails](@ref openai_decisions_failures)) |

[`ask`](@ref), [`ask_each`](@ref) and the verbs built on them throw these `ArgumentError`s before anything is sent.
A wrapper refuses rather than adapts: a request changed to fit would ask a question you did not write, and its
answer would carry no sign of the change.

| Refused | Why: the message's words | What to do |
| :--- | :--- | :--- |
| more than 400 questions | `OpenAI's Decisions API takes at most 400 questions per request` | split them across requests |
| a question without instructions, or with instructions that are not a string | `every question needs`, `must be a string, not`: OpenAI requires a question's instructions, and UniLM sends them as a string | write the question as text, as the first argument of `choice`, `score` or `noul` |
| a Noul with `yes` or `no` | `a predicate has no true/false criteria`: a predicate question has no field for them | fold what they say into its instructions |
| a Choice of fewer than 2 or more than 255 options | `a choice offers 2 to 255 options, and it has` | add an option such as `"anything else"`, or narrow the options |
| an option described by anything but a string or `nothing` | `the description of its option`: an option's description is a string | describe it in words |
| a Score of fewer than 2 or more than 10 levels | `a score has 2 to 10 levels, and it has` | add a level, or merge levels into a coarser rubric |
| a Score level that is not a string | `must be a string, the level's label`: a level is sent as its label | write the level as text |
| a `box` question | `it is a Box question, and the API answers predicate, choice and score questions only` | ask it of a provider that answers Box questions |
| a question type of your own, a subtype of `SystemOneQuestion` | `the API answers predicate, choice and score questions; a RawQuestion is sent verbatim` | build it as a `RawQuestion` |
| a `RawQuestion` that holds a picture | `it holds an image, and images go in the state only` | put the picture in the state |
| a `RawQuestion` whose `"name"` field differs from its name | `and the answer is matched to the question by the name it echoes` | drop the field, or name the question as it does |
| a picture inside a JSON state | `takes images in the state only as`: a JSON state is sent as its text, which holds no picture | pass the texts and the pictures as a `Vector` of strings and images |
| a text over 10,485,760 characters | `OpenAI's Decisions API takes text of at most 10485760 characters` | shorten it |
| a name, instructions, an option or a level over 1,048,576 characters | `OpenAI's Decisions API takes at most 1048576 characters in`, then the field | shorten it |
| more than 128 pictures | `OpenAI's Decisions API takes at most 128 images per request`: the limit of OpenAI's published schema during the beta, which its pages no longer state | split them across requests |
| an option other than the three of [Options](@ref openai_decisions_options) | `OpenAI's Decisions API accepts the options` | one of the three, or none |
| a `prompt_cache_key` or `safety_identifier` that is not a string, or longer than 64 or 128 characters | `as a String of at most` | a shorter string |
| a `prompt_cache_options` that is not a `PromptCacheOptions`, sets `prewarm` or `comparison_response_id`, or has a `ttl` other than `"30m"` or `"12h"` | `as a PromptCacheOptions`, `takes prompt_cache_options mode and ttl only`, `accepts prompt_cache_options ttl` | a `PromptCacheOptions` of `mode` and `ttl` alone |

Like every decision provider, the endpoint also refuses, before anything is sent, a blank or repeated question name,
criteria that are not valid, a state or guidance holding `NaN` or `Inf`, an `AudioAttachment` or a
`DocumentAttachment` anywhere in the request; [Reading answers](@ref Reading-answers) covers a malformed request on
every provider. An image's `detail` level — `"low"`, `"high"`, `"auto"` or `"original"` — goes out on its
`input_image` part, which takes one
([Create a decision](https://developers.openai.com/api/reference/resources/decisions/methods/create)).

```@example openai_decisions
for refused in (() -> ask(message, "refund" => noul("Does the customer want a refund?";
                                                    yes = "They ask for their money back");
                          service = OPENAIServiceEndpoint),
                () -> ask(message, "team" => choice("Which team should handle this message?", ["billing"]);
                          service = OPENAIServiceEndpoint))
    try
        refused()
    catch err
        println(sprint(showerror, err))
    end
end
```

## [Confidence and tuning](@id openai_decisions_confidence)

A predicate's answer is the probability that its statement holds; a Choice or a Score answer carries a probability
distribution and "a separate `confidence` field", and OpenAI tells you to set thresholds on "labeled examples from
your application", by "the cost of false positives and false negatives"
([Interpret the answers](https://developers.openai.com/api/docs/guides/decisions#interpret-the-answers)). OpenAI
computes `confidence` and UniLM passes it on; in the replies recorded for this manual it equals the Choice formula
of [Reading answers](@ref Reading-answers), to two decimals. Those replies give every probability and confidence to
two decimals, so `ask` accepts a distribution within 0.005 per outcome of a sum of 1, and a chosen option at most
0.01 below the most probable one; a reply further off is a [`SystemOneCallError`](@ref).

The bar is yours: below `min_confidence`, a natural-language verb returns what `fallback` returns, and with no
`fallback` it throws `LowConfidenceError`. Of the six messages of a support desk's inbox, the cracked phone is the
unclear one:

```@example openai_decisions
const MESSAGES = [
    "I was charged twice for order #4471. Please refund the duplicate payment.",
    "The app crashes every time I open my order history. iPhone 15, latest version.",
    "My parcel was due last Monday and the tracking hasn't moved in six days.",
    "Do you ship to Norway, and how much does delivery cost?",
    "The phone I received has a cracked screen and the box was crushed. I want my money back.",
    "Your update deleted my saved addresses and now I can't check out. Third time this month!",
]

seen = SystemOneSuccess[]
team = nl_classify(MESSAGES[5], TEAM; service = OPENAIServiceEndpoint, min_confidence = 0.6,
                   fallback = message -> :person, on_response = r -> push!(seen, r))
answered = only(seen)["classify"]
(team, findfirst(==(answered.choice), TEAM), answered.confidence)
```

OpenAI's best guess, `billing`, came with a confidence below the 0.6 asked for, and the message went to a person.
On a labelled set of ours, OpenAI's top probabilities ran below how often it was right: the model is
under-confident there, so a bar tuned on another provider, or on another set, hands a different share of the work
to a person here. Tune `min_confidence` on messages of your own, on this provider; where some mistakes cost more
than others, price each one and let a `decide` policy choose the cheapest action
([When mistakes have different prices](@ref jev_loss_matrix)), once you have checked that the probabilities are
calibrated. The decision rules on every provider are on [Act on the answer](@ref jev_act_on_answer), and the same
messages on each provider are on [Choosing a decision provider](@ref system_one_backends).

!!! details "Evidence"
    Measured on 20 October 2026 on our 129 labelled support tickets (60 clear, 32 borderline, 15 adversarial, 22 out
    of scope), one Choice of seven intents per ticket, each option a name with a description, one call at a time, on
    `gpt-6-luna`: accuracy 0.946, expected calibration error 0.080 (10 bins), Brier score 0.076, mean top probability
    0.89; 101 tickets at a confidence of 0.8 or more, all of them right, so 28 went to a person at that bar; in every
    bin above 0.5, accuracy was at least the mean top probability. Latency 0.48 s at the median and 2.44 s at the
    95th percentile; \$0.0000296 a ticket. Limits: one labelled set, one run, one way of writing the options.

## [Cost](@id openai_decisions_cost)

OpenAI bills a decision's input tokens and no output: uncached input at the model's input rate, while "prompt cache
hits receive discounted input pricing; cache writes are charged", both at the model's caching rates; "regional
processing premiums and long-context input pricing multipliers apply"
([Pricing and availability](https://developers.openai.com/api/docs/guides/decisions#pricing-and-availability)).
The rates are the default model's row of [Cost and usage](@ref openai_cost).

```@example openai_decisions
u = token_usage(r)
(u.prompt_tokens, u.cached_tokens, u.cache_write_tokens, u.completion_tokens, estimated_cost(r))
```

`prompt_tokens` is OpenAI's `input_tokens`, which includes the cached and the cache-written ones, `cached_tokens`
its `cached_tokens` and `cache_write_tokens` its `cache_write_tokens`. [`estimated_cost`](@ref) prices them as the
guide does: the uncached input at the model's input rate, the cache reads at its cached-input rate, the cache writes
at its write rate — a 12-hour write, counted in `cache_write_extended_tokens`, at its own — every rate twice for a
request of more than 272K input tokens, and no output; it adds no regional premium, which a response does not show.
A request whose shared input is under the 1,024 tokens caching needs, such as the one above, reads and writes no
cache. Totals across calls and prices of your own are on [Cost Tracking](@ref cost_guide).

## [Testing and recordings](@id openai_decisions_testing)

[`with_recorded_answers`](@ref) records an exchange with the Decisions API once and replays it with no key and no
network: `ask`, `ask_each` and the verbs built on them. A recording is filed under
`/openai-decisions/v1/decisions`, the request's path behind the namespace `openai-decisions`, so a question asked of
two providers is two recordings, and it is keyed on the exact request: its method, that path, and its body, which
names the model. `prompt_cache_key`, `safety_identifier` and `prompt_cache_options` are sent and left out of the
key and the file, and a picture is keyed and written as `data:<mime>;sha256,<hex>`, the SHA-256 of its base64 text,
never its bytes.

```@example openai_decisions
recordings = mktempdir()      # in a package: a directory committed next to the tests
recorded = with_recorded_answers(recordings; mode = :record) do
    ask(message, "team" => choice("Which team should handle this message?", TEAM); service = OPENAIServiceEndpoint)
end
JSON.parsefile(only(readdir(recordings; join = true)))["request"]["path"]
```

```@example openai_decisions
replayed = withenv("OPENAI_API_KEY" => nothing) do
    with_recorded_answers(recordings) do
        ask(message, "team" => choice("Which team should handle this message?", TEAM);
            service = OPENAIServiceEndpoint, prompt_cache_key = "support-inbox")
    end
end
replayed["team"].probabilities == recorded["team"].probabilities
```

The replay names a `prompt_cache_key` the recording did not, and is answered from it. Recording takes
`OPENAI_API_KEY`, and replaying takes nothing; in this manual's build the recording scope above replays the
manual's own recordings, so the example reaches no service either
([How this manual runs its examples](@ref jev_testing_manual)). [`list_models`](@ref) on this endpoint is not
recorded: inside a scope it goes to OpenAI, with the key, as outside one.

- **What a recording cannot hold.** OpenAI declines a question when it chooses to, so no recording can be made to
  hold a refusal. Requests go to `https://api.openai.com` alone, so to test what your code does with one, declare an
  endpoint of your own that speaks this API's wire to a local server, through the protocol seam
  ([Write your own decision provider](@ref decisions_custom)); `_api_base_url` is an internal hook. Run such tests
  outside `with_recorded_answers`. This block is not run when the manual is built:

```julia
using UniLM, HTTP, JSON, Test

struct DecisionsMock <: UniLM.ServiceEndpoint end
UniLM.provider_capabilities(::Type{DecisionsMock}) = Set([:system_one])
UniLM.system_one_protocol(::Type{DecisionsMock}) = UniLM.OpenAIDecisionsProtocol()
UniLM._api_base_url(::Type{DecisionsMock}) = "http://127.0.0.1:8124"     # internal hook
UniLM.auth_header(::Type{DecisionsMock}) = ["Content-Type" => "application/json"]

TEAM = (billing = "payments, charges, invoices or refunds", other = "anything else")
server = HTTP.serve!("127.0.0.1", 8124; verbose = false) do req
    # A refusal of the one question, as OpenAI's reference shapes it; the name is the question's.
    body = (model = "gpt-6-luna", answers = [(type = "refusal", name = "classify")],
            usage = (input_tokens = 42, output_tokens = 0))
    HTTP.Response(200, ["Content-Type" => "application/json"], JSON.json(body))
end

@testset "a declined question" begin
    @test nl_classify("Where is my parcel?", TEAM; service = DecisionsMock, fallback = m -> :person) === :person
    @test_throws DecisionDeclinedError nl_classify("Where is my parcel?", TEAM; service = DecisionsMock)
    r = ask("Where is my parcel?", "classify" => choice("Which team?", TEAM); service = DecisionsMock)
    @test r["classify"] isa RefusalAnswer
end
close(server)
```

- **A pipeline with a language model.** A test that also calls a language model records each request with its own
  provider's key and replays with none
  ([Test a decision + LLM pipeline offline](@ref jev_testing_pipeline)).

Previews of what a call sends, tests of the methods and of the judgment, and an audit trail are on
[Test and Develop](@ref jev_testing_guide).

## [When something fails](@id openai_decisions_failures)

A call that reaches OpenAI returns a [`SystemOneSuccess`](@ref); a [`SystemOneFailure`](@ref) for a status other
than 200, with the `status`, OpenAI's `message`, its `error_type` — the error's `code`, else its `type` — and the
`request_id`; or a [`SystemOneCallError`](@ref) when no usable response came. A 200 whose answers do not fit the
request — a question unanswered, answers out of the order asked, an option that was not offered, probabilities
that are no distribution within the rounding above — is a `SystemOneCallError` too. Reading an answer of a failed
call (`r["name"]`, [`answer`](@ref), [`answers`](@ref)) throws a [`SystemOneError`](@ref).

UniLM sends a request again after a 408, 429, 500, 502, 503, 504, 520 to 524 or 529, up to `max_attempts` times
within `total_deadline`, waiting at least what the reply's `retry-after-ms` header asks for, else its
`Retry-After`, as the shared backoff of [`ask_each`](@ref) does. A 429 whose code is a spent credit, a
spend or usage limit, or `insufficient_quota` is not sent again, since "retrying billing, spend, or quota errors
won't restore API access" ([Error codes](https://developers.openai.com/api/docs/guides/error-codes)); any other
status comes back as it is. With no [`RequestConfig`](@ref) of your own, a call to the Decisions API is bounded by
60 s per attempt and 150 s in all, and [`ask_each`](@ref) keeps 16 requests in flight unless its `ntasks` says
otherwise ([Many requests at once](@ref jev_items_each)); a config of your own replaces the bounds, per call or for
a scope:

```@example openai_decisions
scoped = with_request_config(request_timeout = 20.0, max_attempts = 5) do
    ask(message, "team" => choice("Which team should handle this message?", TEAM); service = OPENAIServiceEndpoint)
end
issuccess(scoped)
```

OpenAI sets its rate limits per organization and project, and they "vary by the model being used"
([Rate limits](https://developers.openai.com/api/docs/guides/rate-limits)).

| What you see | Why | What to do |
| :--- | :--- | :--- |
| a `SystemOneCallError` whose `error` reads `KeyError: key "OPENAI_API_KEY" not found` | no key in the environment; nothing was sent | set it ([Setup](@ref openai_decisions_setup)) |
| a `SystemOneFailure` with status 401 | the key is not valid ([Error codes](https://developers.openai.com/api/docs/guides/error-codes)) | a key from your OpenAI account |
| a `SystemOneFailure` with status 404, `error_type` `"model_not_found"`, and a message that the model `does not exist or you do not have access to it` | a model the API does not serve ([Models](@ref openai_decisions_models)) | name no model |
| a `SystemOneFailure` whose message names a field of the request | OpenAI's schema refuses what UniLM's checks let through, such as a `RawQuestion` | the field the message names |
| status 429 after the retries, with `retry_after`, the wait OpenAI asked for | over a rate limit | fewer requests in flight (`ntasks`), or higher limits |
| status 429 at once, with an `error_type` such as `insufficient_quota` | no credit left, or a spend or usage limit reached | credits, or a higher limit |
| status 500, 503 or another retried status, after the retries | a failure on OpenAI's side, or the model overloaded | a larger `max_attempts` or `total_deadline` |
| a `SystemOneCallError` whose `cause` is a `UniLMTimeout` or a `UniLMCancelled` | a timeout, or a cancelled token | a `RequestConfig` of your own ([Timeouts & Retries](@ref timeouts_guide)) |
| a `SystemOneCallError` after a 200 | an answer that does not fit the request | keep `request_id` (the reply's `x-request-id`) |
| a `ReplayMissError`, thrown | in a replay scope, a request with no recording | record it with `OPENAI_API_KEY` ([Testing and recordings](@ref openai_decisions_testing)) |
| a `SystemOneError`, thrown | an answer read from a failed call | test `issuccess(r)` first |
| a `FieldError`, thrown: `type RefusalAnswer has no field` | a declined question read as an answer | test `a isa RefusalAnswer` first ([Declined questions](@ref openai_decisions_refusals)) |

Recorded on 25 October 2026 on `gpt-6-luna` with UniLM's development branch after 0.26.0, and, on 19 October 2026,
the `nl_classify` calls (on `gpt-6-luna`) with UniLM's development branch after 0.23.0, through OpenAI's API at
`/v1/decisions`.

See also: [`OPENAIServiceEndpoint`](@ref), [`ask`](@ref), [`RefusalAnswer`](@ref),
[`UniLM.OpenAIDecisionsProtocol`](@ref) and [`PromptCacheOptions`](@ref) in the API reference, and
[OpenAI](@ref openai_start) in the Language models chapter.
