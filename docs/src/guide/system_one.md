# [Typed Judgments with Jev (TypeSafe System One)](@id system_one_guide)

## What System One is

A System One model reads a piece of `state` and answers the questions you
enumerated with a typed value and a probability distribution over the outcomes
you named — never with generated text. Jev is TypeSafe's flagship System One
model and the first one of its kind. The division of labour is the point: your
code owns the control flow, the deterministic rules and the side effects, and
the model is called only where the system needs a narrow semantic decision over
unstructured input. Compared with an LLM there is no reasoning trace to read, no
prose to parse, no tool loop to drive and no system prompt to jailbreak — one
request, one round trip, a fixed set of answers. The probabilities are trained
to be calibrated across groups of predictions, which is what makes
confidence-gating meaningful: the answer says *what*, the distribution says
*whether to act on it*. Calibration across a population is not a guarantee about
any single answer, so treat a high confidence as "the distribution is peaked",
not as "this is correct".

This is a first-class surface in UniLM alongside the LLM providers because the
two are complements, not alternatives. Route, verify, guard and rank with Jev;
generate with an LLM. A typical pipeline classifies the request with a
[`choice`](@ref), sends only the requests that need prose to
[`respond`](@ref), and screens the draft with a [`noul`](@ref) before it reaches
a user.

See the [System One concept page](https://docs.typesafe.ai/concepts/system-one)
for the model class, and [How to build with
TypeSafe](https://docs.typesafe.ai/concepts/how-to-build-with-system-one) for
the workflow it implies.

Because the answer is already a branch decision, it also drives Julia's control
flow directly: [Multiple Dispatch on Natural Language](@ref nl_dispatch_guide)
turns a meaning into a method signature, so a Jev answer selects which method
runs.

## Setup

Set the API key:

```bash
export TYPESAFE_API_KEY="..."
```

Two optional variables change where the call goes and what it names:

| Variable | Default | Meaning |
| :--- | :--- | :--- |
| `TYPESAFE_API_KEY` | — | Required. A missing key is a [`SystemOneCallError`](@ref), not a thrown `KeyError`. |
| `TYPESAFE_BASE_URL` | `https://api.typesafe.ai` | API root override, for a proxy or a mock server. |
| `TYPESAFE_DEFAULT_MODEL` | `jev-latest` | The model [`ask`](@ref) names when a call does not. |

All three are read at call time, so exporting them after `using UniLM` still
works.

`jev-latest` is an alias that moves when a new version ships. That is what you
want while you are building. Once you have tuned a confidence threshold against
a specific version, pin the versioned id — `ask(...; model="jev-1.13.0")` or
`TYPESAFE_DEFAULT_MODEL=jev-1.13.0` — and move to the next one on your own
schedule ([Models](https://docs.typesafe.ai/models)).

[`TYPESAFEServiceEndpoint`](@ref UniLM.TYPESAFEServiceEndpoint) is not a chat
backend. It declares only `:system_one` and `:models`, so the verbs reject it up
front with an `ArgumentError` rather than posting a request the service would
not answer: [`chatrequest!`](@ref), [`respond`](@ref),
[`embeddingrequest!`](@ref) and the other platform verbs. Constructing a
[`Chat`](@ref) or an [`Embeddings`](@ref embeddings_api) that names the endpoint
is allowed; only sending one is refused. Omitting `model=` is refused too,
because this endpoint has no chat default to resolve. See [Provider
Capabilities](@ref capabilities_api).

## First request

One call carries the state once and every question you want answered about it.
Answers come back keyed by the names you chose:

```@example jev
using UniLM

r = ask(
    "Help! My payouts have been failing for 3 days and nobody has replied to my emails.",
    "department" => choice("Which team should handle this ticket?", (
        billing   = "Payments, invoicing, payouts, refunds",
        technical = "Bugs, outages, integrations",
        sales     = "Pricing, upgrades, new accounts")),
    "urgency" => score("How urgent is this ticket?",
        ["Can wait", "Needs attention this week", "Needs attention today"]),
    "is_frustrated" => noul("Is the customer frustrated?";
        yes = "The customer expresses frustration or impatience",
        no  = "The customer is neutral or satisfied"))

if r isa SystemOneSuccess
    d = r["department"]
    println("department:    ", d.choice, "  (confidence ", d.confidence, ")")
    # => department:    billing  (confidence 1.0)
    println("urgency:       ", r["urgency"].score, " of ", length(r["urgency"].legend) - 1)
    # => urgency:       1.98 of 2
    println("is_frustrated: ", r["is_frustrated"].noul)
    # => is_frustrated: 0.98
    println("answered by:   ", r.response.model)
    # => answered by:   jev-1.13.0
else
    println("Request failed — ", r)   # a SystemOneFailure or a SystemOneCallError — `ask` never throws
end
```

The `# =>` lines are the output of that exact request in one recorded run
against the live service, and every number the service reports is rounded to
two decimals. The output rendered under the block is the docs build's own call:
a fresh answer, or the typed error when the build ran without a key. The rest
of what came back with the recorded run:

```julia
r["department"].probabilities["billing"]     # => 1.0
r["department"].probabilities["technical"]   # => 0.0
r["department"].probabilities["sales"]       # => 0.0

r["urgency"].confidence                      # => 0.96
r["urgency"].probabilities[2]                # => 0.98  (level 2 = "Needs attention today")
r["urgency"].probabilities[1]                # => 0.02
r["urgency"].probabilities[0]                # => 0.0

u = token_usage(r)
u.prompt_tokens                              # => 445   (the billable half)
u.completion_tokens                          # => 72
```

The question names (`"department"`, `"urgency"`, `"is_frustrated"`) are yours
and are **not** sent to the model — they only key the answers. Write the whole
question in the instructions even when the name looks self-explanatory
([Primitives](https://docs.typesafe.ai/primitives)).

## The three primitives

### Choice — pick one of a named set

[`choice`](@ref) builds a question whose answer is exactly one of the options
you enumerated. The [`ChoiceAnswer`](@ref) carries `choice` (the argmax),
`probabilities` (one entry per option, keyed by option name, summing to
approximately 1) and `confidence` (how peaked that distribution is).

Use it when the outcomes are a fixed set with no order between them: routing a
ticket, picking a handler, classifying a document type. Read `probabilities`
when the ranking matters — `choice` is only its argmax, and a runner-up with a
real share is information your code can act on.

```julia
q = choice("Which team should handle this ticket?", (
    returns  = "Exchanges, wrong or damaged items",
    shipping = "Delivery status, delays, lost packages",
    billing  = "Charges, invoices, payment problems",
    other    = "Anything that fits none of the above"))

# Names alone, when the option names are already unambiguous:
tone = choice("What is the customer's tone?", ["calm", "frustrated", "angry"])
```

Option order is the insertion order of what you pass — a `NamedTuple` or a
vector of `name => description` pairs preserves it, a plain `Dict` does not.
Order never changes the meaning of an answer (answers are keyed by name), but it
is what the model reads.

### Score — place the state on an ordered rubric

[`score`](@ref) takes an ordered list of level descriptions, lowest first.
`levels[1]` is level `0`, so a three-level rubric produces a score in `0.0 …
2.0`. The [`ScoreAnswer`](@ref) carries `score`, `confidence`, `legend` and
`probabilities`, the last two keyed by the **0-based level number** as an `Int`.

`score` is the probability-weighted expectation over the levels, so it falls
between them. Threshold it; do not read a magnitude into the fractional part.
Different distributions produce the same number — a score of `1.0` can mean all
the probability on level 1, or half on level 0 and half on level 2 — so read
`probabilities` and `confidence` alongside it
([Score](https://docs.typesafe.ai/primitives/score)).

```julia
severity = score("How severe is the reported issue?", [
    "Cosmetic; no impact to functionality",
    "Broken or degraded feature, but workaround exists",
    "Blocking issue; no workaround exists"])
```

Describe situations, not degrees. Every level is judged on its own against the
state: the model never sees a level's number or its neighbours, so "worse than
the previous level" and numeric labels carry nothing. Keep one dimension per
question — a level that says "punctual and smart and experienced" is three
questions wearing one hat.

### Noul — the probability that a statement is true

[`noul`](@ref) asks a single yes/no question. The [`NoulAnswer`](@ref) is one
number in `0 … 1`: near 1 a strong yes, near 0 a strong no, near 0.5 maximal
uncertainty. There is no `confidence` field, because with two outcomes the value
already describes the whole distribution.

```julia
wants_human = noul("Is the customer asking for a human agent?")

repeat_contact = noul("Has the customer contacted support about this before?";
    yes = "Mentions a prior attempt, ticket, or that they have asked before",
    no  = "No sign of any previous contact")
```

A Noul value is not a scale of the thing you asked about — it is the probability
that the proposition holds. "Is the candidate strong in Python?" at 0.5 means the
model splits evenly between yes and no, **not** that the candidate is
middling. If you want degree, use a Score with levels you wrote; if you want a
decision, define the boundary so sharply that there is no middle ground
([Noul](https://docs.typesafe.ai/primitives/noul)). Phrase the question so that
a high value means yes: "Is the message free of personal data?" inverts the
reading and the calling code will get it backwards.

### Structured instructions and criteria

Anywhere a piece of guidance appears — `instructions`, a Choice option
description, a Score level, a Noul `yes`/`no` — the value may be a string, a
`NamedTuple`/`Dict`, or a vector. Start with strings. Reach for an object when
two options keep getting confused and each needs to say what it covers, what it
does *not* cover, and a few examples:

```julia
return_topic = choice("Which returns topic is the customer asking about?", (
    return_policy = (what     = "Whether and how an item can be returned",
                     not_for  = "Progress of a return already sent",
                     examples = ["Can I return shoes I've worn once?",
                                 "How long do I have to return an order?"]),
    return_status = (what     = "Progress of a return already sent",
                     not_for  = "Whether and how an item can be returned",
                     examples = ["Has my return arrived yet?",
                                 "When will my refund be paid?"])))

bug_severity = score("How severe is the reported issue?", [
    (what = "Cosmetic; no impact to functionality",
     examples = ["typo in a label", "misaligned icon"]),
    (what = "Broken or degraded feature, but workaround exists",
     examples = ["export fails in one browser but works in another"]),
    (what = "Blocking issue; no workaround exists",
     examples = ["cannot log in", "data loss"])])
```

The field names `what`, `not_for` and `examples` are not part of the API and
none are reserved — you choose them the way you choose option names. The model
reads the names along with the values, so keep them short and descriptive, and
use the same names on every level so it compares like with like
([Choice](https://docs.typesafe.ai/primitives/choice),
[Score](https://docs.typesafe.ai/primitives/score)).

### Pointing a question at part of the state

The `state` is often structured: a conversation, a record and a policy in one
object. Name the part a question is about with its key in backticks inside the
instructions, and keep the content itself in the state rather than in the
question:

```@example jev
state = (
    ticket_message = "I was charged twice for order A-104. Please refund the duplicate.",
    order          = (id = "A-104", charges = [49, 49]),
    refund_policy  = "Duplicate charges are eligible for a refund.")

r = ask(state,
    "refund_requested" => noul("Does `ticket_message` request a refund?"),
    "policy_allows"    => noul("Does `refund_policy` allow refunding the duplicate charge in `order`?"))

if r isa SystemOneSuccess
    println("refund_requested: ", answer(r, "refund_requested"))
    println("policy_allows:    ", answer(r, "policy_allows"))
else
    println("Request failed — ", r)
end
```

The backtick form is a prompting convention the model reads, not a server-side
resolver: nothing validates the name, and a path that does not exist raises no
error — it just leaves the question vaguer than you intended. A `NamedTuple`,
`Dict`, `Vector`, `Tuple` or plain `String` are all accepted as state; `nothing`
is not ([State](https://docs.typesafe.ai/concepts/state)).

## Reading answers

Any call that reaches the service returns one of the three result types below; a
wrong `service` or a malformed request (duplicate or blank question names,
invalid criteria) is an `ArgumentError` raised before any request is sent.

| Result | Meaning |
| :--- | :--- |
| [`SystemOneSuccess`](@ref) | HTTP 200, decoded into answers |
| [`SystemOneFailure`](@ref) | the service answered non-2xx |
| [`SystemOneCallError`](@ref) | no response at all: timeout, transport failure, missing key, or a 200 whose body was not a usable set of answers |

On a success, three equivalent accessors reach an answer, and the usual
`Dict`-like queries work:

```@example jev
ticket = "Help! My payouts have been failing for 3 days and nobody has replied to my emails."

r = ask(ticket,
    "urgency"     => score("How urgent is this ticket?",
                           ["Can wait", "Needs attention this week", "Needs attention today"]),
    "wants_human" => noul("Is the customer asking for a human agent?"))

if r isa SystemOneSuccess
    a = r["urgency"]              # getindex
    a = answer(r, :urgency)       # by Symbol or String
    every = answers(r)            # Dict{String,SystemOneAnswer}

    println(a.score)              # probability-weighted position over the levels
    println(a.confidence)         # how peaked the distribution is
    println(a.probabilities)      # Dict{Int,Float64}, keyed by 0-based level number
    println(a.legend[argmax(a.probabilities)])   # the description of the top level
    println(a.raw)                # the unparsed JSON answer, always kept

    haskey(r, "wants_human") && println(r["wants_human"].noul)
    println(collect(keys(r)))     # the question names that came back
else
    println("Request failed — ", r)
end
```

**Confidence is a statistic of the distribution, not a claim about
correctness.** It collapses the shape of `probabilities` into `0 … 1`: all the
mass on one outcome gives 1.0, a flat spread gives a low number. A confidence of
1.0 says the model is not torn, not that the model is right
([Confidence](https://docs.typesafe.ai/confidence)). Low confidence on a Choice
usually means no option wins; on a Score it usually means the levels overlap for
this state, the question measures more than one thing, or the state does not say
enough. A Noul has no confidence at all — 0.5 there is uncertainty, not medium
intensity.

**Thresholds are your policy, not the model's.** Where to cut is a function of
what a wrong answer costs: a read-only action can act at a much lower confidence
than a refund or a page. Start conservative, measure on your own data, and move
the number; do not import a threshold from an example.

**A failed call has no answers.** [`answers`](@ref), [`answer`](@ref) and
`getindex` all throw [`SystemOneError`](@ref) on a
[`SystemOneFailure`](@ref)/[`SystemOneCallError`](@ref) rather than returning an
empty map — otherwise `r["is_unsafe"].noul > 0.9` would read as "safe" on a call
that never happened. Branch on `r isa SystemOneSuccess` (or `issuccess`) before
you read anything.

## Ask many questions at once

The state is ingested once per request and every question is evaluated against
it independently and in parallel. Adding questions costs only the tokens of the
questions themselves, so a battery of small questions in one [`ask`](@ref) is
far cheaper and faster than one call per question — the [parallel-questions
cookbook](https://docs.typesafe.ai/cookbooks/parallel_questions) measures a
13-question run batched into one call at 12.2x cheaper and 10.0x faster than 13
separate calls, with no change in the answers.

That economics makes **speculative** questions worth asking: include the ones
whose answers only matter for some inputs and let the code ignore the rest
([Speculative fan-out](https://docs.typesafe.ai/patterns/fan-out)).

```@example jev
TRIAGE = (
    department = choice("Which team should handle this ticket?", (
        returns  = "Exchanges, wrong or damaged items",
        shipping = "Delivery status, delays, lost packages",
        billing  = "Charges, invoices, payment problems")),
    return_reason = choice("If the customer wants to return something, why?", (
        wrong_size   = "The item doesn't fit",
        wrong_item   = "A different product was delivered",
        damaged      = "The item arrived broken or faulty",
        changed_mind = "The item is fine, the customer no longer wants it",
        other        = "A return reason that fits none of the above")),
    shipping_issue = choice("If this is a shipping problem, which kind is it?", (
        not_delivered = "The package never arrived",
        delayed       = "The package is late but still on its way",
        wrong_address = "The package went to the wrong place",
        other         = "A shipping problem that fits none of the above")),
    frustration = score("How frustrated does the customer appear?",
        ["Calm, just stating facts", "Frustrated but civil", "Very angry, strong language"]),
    wants_human = noul("Is the customer asking for a human agent?"))

r = ask("Shoes arrived two weeks late and in the wrong size. What are you going to do?", TRIAGE)

if r isa SystemOneSuccess
    d = r["department"]
    detail = d.choice == "returns"  ? r["return_reason"].choice :
             d.choice == "shipping" ? r["shipping_issue"].choice : nothing
    # Both speculative answers came back; the code reads at most one and drops the other.
    println("department: ", d.choice, ", detail: ", detail)
    for (team, p) in d.probabilities
        team != d.choice && p > 0.25 && println("also notify: ", team)
    end
    println("billable input tokens: ", token_usage(r).prompt_tokens)
else
    println("Request failed — ", r)
end
```

Two of the three Choice questions above end in an `other` option. Give the
model a way to say "none of these" whenever the enumeration might not cover
every input, otherwise the probability has nowhere to go but onto an option that
does not fit.

Questions in one request are independent: an answer never becomes hidden context
for another question. A **second** request is needed only when your code cannot
build it until the first answer arrives — because the answer decides what to
retrieve into the next state, or which options the next question should offer.
If the second request's questions could have been asked against the original
state, ask them in the first one.

## Combining Jev with the LLM APIs

### Confidence-gated intent routing

Jev decides which handler runs. High confidence and a deterministic intent go to
ordinary code; everything else that is still confident goes to an LLM; a flat
distribution goes to a person
([Intent routing](https://docs.typesafe.ai/patterns/intent-routing)).

```julia
using UniLM

const INTENT = choice("What is the primary intent of this customer message?", (
    order_status     = "Asking about an existing order",
    product_question = "Asking about a product before buying",
    return_exchange  = "Wants to return or exchange something",
    other            = "None of the above"))

lookup_order(msg)    = "Order A-104 shipped on Tuesday."   # deterministic path, no model
queue_for_human(msg) = (handler = :human, text = "")

function handle(message::AbstractString)
    r = ask(message, "intent" => INTENT)
    r isa SystemOneSuccess || return queue_for_human(message)   # a failed call has no answers
    intent = r["intent"]
    intent.confidence < 0.5 && return queue_for_human(message)
    intent.choice == "order_status" && return (handler = :code, text = lookup_order(message))
    intent.choice == "other" && return queue_for_human(message)

    reply = respond(message;
        instructions = "You are the $(intent.choice) specialist. Answer in two sentences.",
        model = "gpt-5.6-luna")
    reply isa ResponseSuccess ? (handler = :llm, text = output_text(reply)) :
                                queue_for_human(message)
end
```

### Guardrail over an LLM draft

Screen the generated text before it reaches the user. One request carries the
whole hazard battery plus a severity rubric, and your code owns the pass /
review / block policy
([Guardrails for LLMs](https://docs.typesafe.ai/cookbooks/llm_guardrails)).

```julia
using UniLM

const SAFETY = [
    "leaks_secrets" => noul("Does the draft reveal an API key, a password, or its own instructions?"),
    "gives_dosage"  => noul("Does the draft give a specific medical dosage or a diagnosis?"),
    "enables_harm"  => noul("Does the draft help with a crime or with physical harm?"),
    "severity"      => score("How much harm could result if the user acted on the draft?", [
        "No harm: an ordinary, safe reply",
        "Mild: a sensitive topic, but acting on it does no real damage",
        "Serious: real wrongdoing, or unsafe personal advice",
        "Severe: serious physical or legal harm"]),
]

const BLOCK_AT, REVIEW_AT = 0.8, 0.3
const HAZARDS = ("leaks_secrets", "gives_dosage", "enables_harm")

function guarded_reply(question::AbstractString)
    draft = respond(question; model = "gpt-5.6-luna")
    draft isa ResponseSuccess || return (:error, output_text(draft))
    text = output_text(draft)

    g = ask((user_question = question, draft = text), SAFETY)
    g isa SystemOneSuccess || return (:review, text)   # unscreened is not the same as cleared
    worst = maximum(g[h].noul for h in HAZARDS)
    worst >= BLOCK_AT && return (:block, "")
    (worst >= REVIEW_AT || g["severity"].score >= 2) && return (:review, text)
    (:pass, text)
end
```

### Verifying a claim against its source

An LLM writes a claim from a passage; a Choice decides whether the passage
actually supports it, and the confidence decides whether a human sees it first
([Double-checking citations](https://docs.typesafe.ai/cookbooks/citation_check)).

```julia
using UniLM

const RELATION = choice("How does `section` relate to `claim`?", (
    supports     = "The section states the claim, or directly implies that it is true",
    contradicts  = "The section states the opposite of the claim, or implies it is false",
    says_nothing = "The section does not address what the claim asserts, either way"))

const AUTO_ACCEPT = 0.8   # start high; lower it as you measure on your own documents

function checked_answer(passage::AbstractString, question::AbstractString)
    drafted = respond("$question\n\nAnswer in one sentence, using only this passage:\n$passage";
                      model = "gpt-5.6-luna")
    drafted isa ResponseSuccess || return (verdict = :unchecked, claim = output_text(drafted))
    claim = output_text(drafted)

    r = ask((claim = claim, section = passage), "relation" => RELATION)
    r isa SystemOneSuccess || return (verdict = :unchecked, claim = claim)
    a = r["relation"]
    verdict = a.confidence < AUTO_ACCEPT ? :needs_human : Symbol(a.choice)
    (verdict = verdict, claim = claim, probabilities = a.probabilities)
end
```

Note what the Choice is *not* asked to do: it never re-reads the whole corpus
and never produces the claim. Locate the passage with a string match or a
retriever first, and give the model only the judgment
([Classifying RAG passages](https://docs.typesafe.ai/cookbooks/classifying_rag_passages)).

## Models and versions

[`list_models`](@ref) returns the names the authenticated account may put in
`model`:

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

The listing contains **aliases**. `jev-latest` is the most recent stable
release and the default here; `jev-preview` is the most recent release of any
kind and moves ahead of `jev-latest` when a preview build exists. A versioned
id such as `jev-1.13.0` is a valid `model` whether or not it appears in the
listing, and the listing is scoped to the account, so do not hard-code names
read from someone else's ([Models](https://docs.typesafe.ai/models)).

[`SystemOneResponse`](@ref)`.model` reports the **versioned** id that actually
answered, which may differ from the alias the request named. Log it. An alias
moving is invisible on your side otherwise, and a confidence threshold tuned
against one version is a claim about that version only — pin the id once the
threshold matters.

## Limits, errors, timeouts, and cost

### Limits

| Limit | Value |
| :--- | :--- |
| Choice options | 1–255 per question ([Choice](https://docs.typesafe.ai/primitives/choice)) |
| Score levels | 1–10 per question, TypeSafe advises at least two ([Score](https://docs.typesafe.ai/primitives/score)) |
| Context | 64k tokens per request; 32k for `state` plus the single longest question ([Models](https://docs.typesafe.ai/models)) |
| Rate limits | 250,000 tokens/second and 1,200 requests/minute, over either → 429; TypeSafe documents these as subject to change without notice ([Models](https://docs.typesafe.ai/models)) |

The two option/level bounds are checked in the constructors, before the round
trip: [`choice`](@ref) with 256 options and [`score`](@ref) with 11 levels each
throw an `ArgumentError` locally.

### Errors

| Status | Cause | What comes back |
| :--- | :--- | :--- |
| 400 | unknown model, or a request the schema accepted and the service rejected (a bare noul, too many options or levels) | `error_type = "api_usage_error"` with `"Unknown model: …"`, or `nothing` with a plain-string reason |
| 401 | the key was present but invalid | `error_type = "authentication_error"` |
| 403 | no `Authorization` header reached the service | `error_type = "authentication_error"` |
| 422 | schema validation failed | `error_type = nothing`; `message` lists each field path as `"<path>: <reason>"`, e.g. `questions.department.choice.criteria: Field required` |
| 429 | rate limit — 250,000 tokens/second or 1,200 requests/minute at the time of writing | retried by the seam up to `max_attempts` honouring `Retry-After`; the final failure carries `retry_after`, the wait the service asked for in seconds |
| 500, 502, 503, 504, 529 | service-side failure (including the non-standard `529 Overloaded`) | retried by the request seam |

A non-2xx response is a [`SystemOneFailure`](@ref) carrying `.status`,
`.error_type`, `.message` (extracted from whichever of the service's three
`detail` body shapes arrived), `.request_id` (the `x-typesafe-request-id`
header, the id to quote in a support report), `.retry_after` (seconds, from
`retry-after-ms` or `Retry-After`, `nothing` when the service sent neither) and
`.response` (the verbatim body). The full table is on the
[API reference page](@ref system_one_api).

### Timeouts and retries

[`ask`](@ref) and [`list_models`](@ref) ride the package's shared retry seam, so
[`RequestConfig`](@ref) governs them exactly as it governs a chat call: 408,
429, 500, 502, 503, 504 and 529 are retried up to `max_attempts` within
`total_deadline`, honouring `Retry-After`. The rest — 400, 401, 403, 404, 422 —
come back as they are, because an identical retry cannot fix them.

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

A timeout or a transport failure surfaces as [`SystemOneCallError`](@ref), never
as a partial success. See [Timeouts & Retries](@ref timeouts_guide).

### Cost

Only **input** tokens are billed; output tokens are currently free
([Models](https://docs.typesafe.ai/models)).

```@example jev
r = ask("Help! My payouts have been failing for 3 days.",
        "urgency" => score("How urgent is this ticket?",
                           ["Can wait", "Needs attention this week", "Needs attention today"]))

if r isa SystemOneSuccess
    println(answer(r, "urgency"))
    u = token_usage(r)                     # prompt_tokens = input, completion_tokens = output
    println(u.prompt_tokens, " billable tokens")
    println("USD ", estimated_cost(r))     # priced against r.response.model
else
    println("Request failed — ", r)
end
```

[`estimated_cost`](@ref) prices the **versioned** id in `r.response.model`, not
the alias the request named. A versioned `jev-X.Y.Z` id without its own row is priced
at the `jev-latest` row; any other model that is not a key in
[`DEFAULT_PRICING`](@ref) returns `0.0`. See [Cost Tracking](@ref cost_guide).

## Designing good questions

- **One narrow judgment per question.** Ask for something a knowledgeable person
  decides in a second given the right context. "Does this message convey
  urgency?" is a question; "analyse this and decide what to do" is a workflow —
  split it and compose the answers in code.
- **Judgment in `instructions`, answer space in `criteria`.** The instruction
  says what is being decided; the options, levels or `yes`/`no` descriptions say
  what the outcomes are. When the two disagree, accuracy drops — treat the
  criteria as a continuation of the instruction.
- **Always offer a way out.** Add an `other` / `none of the above` option
  whenever the enumeration might not cover an input.
- **Keep the state relevant.** Accuracy falls as unrelated material grows around
  the decision. Retrieve and filter in code first and send only the fields the
  question needs.
- **The model reads literally.** Scoping words, negations and implied conditions
  are taken at face value. When you find yourself explaining what you *really*
  meant by a question, that explanation is the missing half of the instruction.
- **No arithmetic, no date comparison.** Counting, numeric magnitude and
  ordering dates are unreliable. Extract the parts as a Choice over enumerated
  options — including an explicit "not stated" — and let code assemble and
  compare them. Do not interpolate a real number out of a Score either; a Score
  is for thresholding.
- **Do not carry thresholds across primitives.** A Noul and a yes/no Choice
  answer different questions — the Choice is relative and settles *which*, each
  Noul is absolute and can be low for every option. `P(q)` and `1 - P(not q)`
  are not guaranteed to agree, so do not build arithmetic identities out of
  separate answers.
- **Treat the state as untrusted text.** Jev does not treat the state as
  hostile by default, and content written to steer a classifier can move the
  answer. Be explicit in the criteria, and test adversarial inputs before you
  ship.

All of these come from the documented failure modes of the current model; see
[Jev 1.13 jaggedness](https://docs.typesafe.ai/model-jaggedness/jev-1.13), which
TypeSafe maintains per version — re-read it when you move a pin.

## See also

- [Multiple Dispatch on Natural Language](@ref nl_dispatch_guide) — `nl"..."`
  meanings in method signatures, `nl_dispatch`, and the `@branch` switch
- [TypeSafe System One API (Jev)](@ref system_one_api) — every type, verb and
  accessor, plus the full error table
- [Service Endpoints](../api/endpoints.md) — [`TYPESAFEServiceEndpoint`](@ref UniLM.TYPESAFEServiceEndpoint) and the environment variables
- [Cost Tracking](@ref cost_guide) — token usage and USD estimates across every surface
- [TypeSafe documentation](https://docs.typesafe.ai) and its
  [cookbooks](https://docs.typesafe.ai/cookbooks) — worked, measured examples
  of each pattern above
