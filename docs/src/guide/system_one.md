# [Route and Decide](@id system_one_guide)

```@setup system_one
using UniLM
```

A decision model reads a piece of state — a text, a record, a picture — and answers the questions you list
about it — which one, how much, yes or no — with a probability for each outcome, never with generated text.
Your code keeps the control flow: [`ask`](@ref) sends the questions, and your code reads the typed answers and
decides what to do. This page shows the question kinds, how to read an answer, and decision rules built on its
probabilities, the same on every decision provider.

```@example system_one
service, model = TYPESAFEServiceEndpoint, "jev-latest"
nothing # hide
```

The examples use only the names bound above. `service` and `model` are the decision provider and model they
run on: bind your own there, once its entry's setup is done ([Decisions](@ref decisions_start) lists the
providers, and [Choosing a decision provider](@ref system_one_backends) sends the same calls to each). The
tables [at the end](@ref system_one_providers) link each provider's rules.

| I want to… | Use | You get | Section |
| :--- | :--- | :--- | :--- |
| pick one of a set: a team, an intent | [`choice`](@ref) | the chosen option and a probability for each option | [Choice, Score, Noul](@ref jev_primitives) |
| place a text on a scale: how urgent, how severe | [`score`](@ref) | a position on your levels and a probability for each level | [Choice, Score, Noul](@ref jev_primitives) |
| know whether a statement holds: does the customer want a refund? | [`noul`](@ref) | the probability of yes | [Choice, Score, Noul](@ref jev_primitives) |
| ask several questions about one text | one [`ask`](@ref) with every question | every answer from one request, the text sent once | [Ask many questions at once](@ref jev_many_questions) |
| fill a struct from a text | one `ask`, one question per field | a typed value and a certainty for each field | [Fill a struct in one request](@ref jev_fill_struct) |
| act only when the answer is sure enough, and hand the rest to a person | a confidence threshold | the answer, or a hand-over | [Act on the answer](@ref jev_act_on_answer) |
| act when some mistakes cost more than others | a price for each mistake | the cheapest action, a person included | [When mistakes have different prices](@ref jev_loss_matrix) |
| catch the messages that fit none of my options | a catch-all option | the probability that none fits | [Let the model say none of these](@ref jev_none_of_these) |
| act on a small chance of the worst case | the probability at the top level | an alarm that a middling score would not raise | [Act on the risk of the worst level](@ref jev_worst_level) |
| judge what an image shows | an [`ImageAttachment`](@ref) in the state | answers about the picture, as about a text | [What each provider takes](@ref backends_table) |
| find where something is in an image | a [`box`](@ref) question, where a provider answers one | the box's edges as fractions of the image, and a confidence | [Locate with Box](@ref jev_box) |
| send a state longer than a request's usual context | a provider's long-context option, where it has one | a longer context | [Long context](@ref jev_long_context) |

## [First request](@id system_one_first)

One call carries the text once and every question you want answered about it.
Answers come back keyed by the names you chose:

```@example system_one
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
        no  = "The customer is neutral or satisfied");
    service, model)

if r isa SystemOneSuccess
    d = r["department"]
    println("department:    ", d.choice, "  (confidence ", d.confidence, ")")
    println("urgency:       ", r["urgency"].score, " of ", length(r["urgency"].legend) - 1)
    println("is_frustrated: ", r["is_frustrated"].noul)
    println("answered by:   ", r.response.model)
else
    println("Request failed — ", r)   # a SystemOneFailure or SystemOneCallError: returned, not thrown
end
```

The Noul's `yes` and `no` describe its two outcomes. A provider whose API has no
place for them refuses them before sending: fold what they say into the question
there ([Limits and refusals](@ref openai_decisions_limits)).

The output under each block is the docs build's own: it replays an answer
recorded from a live call ([Test and Develop](@ref jev_testing_guide)). The rest
of what came back:

```@example system_one
for team in ("billing", "technical", "sales")
    println("P(", team, ") = ", r["department"].probabilities[team])
end
u = r["urgency"]
println("urgency confidence ", u.confidence)
for level in 2:-1:0
    println("P(level ", level, ") = ", u.probabilities[level], "   ", u.legend[level])
end
usage = token_usage(r)
println(usage.prompt_tokens, " input tokens, ", usage.completion_tokens, " output tokens")
```

The question names (`"department"`, `"urgency"`, `"is_frustrated"`) are yours,
and they key the answers. Whether a provider shows a question's name to its model
differs, as each entry's Questions section says, so on every provider write the
whole question in the instructions, even when the name looks self-explanatory.

## [Choice, Score, Noul](@id jev_primitives)

Three question types cover the judgments: pick one of a set, place on a scale,
say whether a statement holds, and every decision provider answers all three.
How many options a Choice and how many levels a Score may have is each
provider's: its entry's Questions section gives them, and which of them UniLM
checks before the request is sent.

### [Choice — pick one of a named set](@id system_one_choice)

[`choice`](@ref) builds a question whose answer is exactly one of the options
you enumerated. The [`ChoiceAnswer`](@ref) carries `choice` (the most likely
option), `probabilities` (one per option, keyed by option name, summing to
about 1) and `confidence` (how peaked they are). Use it for a fixed set with no
order between the options: a team, an intent, a document type. A runner-up with
a real share is information your code can act on.

```@example system_one
department = choice("Which team should handle this ticket?", (
    returns  = "Exchanges, wrong or damaged items",
    shipping = "Delivery status, delays, lost packages",
    billing  = "Charges, invoices, payment problems",
    other    = "Anything that fits none of the above"))

# Names alone, when the option names are already unambiguous:
tone = choice("What is the customer's tone?", ["calm", "frustrated", "angry"])
nothing # hide
```

Option order is the order of what you pass, and it is the order the model reads:
on ambiguous inputs it moves the probabilities (answers stay keyed by name).
Build options and state from ordered containers — a `NamedTuple`, a vector of
`name => description` pairs or a `JSON.Object` — to choose that order. A plain
`Dict` has none of its own: its options, and its keys in a state, go out sorted
by name, so `"t10"` comes before `"t2"` — the same on every Julia version, but
not the order you wrote.

!!! details "Evidence"
    Measured on jev-1.13.0 (September 2026) on our own labeled sets, changing
    the option order shifted the probabilities of ambiguous items by 0.12 on
    average, and reversing it moved one ambiguous ticket's probability by about
    0.5; changing the key order of the state moved probabilities by up to 0.16.

An option's name goes out with its description, so a short name such as
`returns` is part of what the model reads, and a word in the text can pull the
answer toward it. When only the description should decide, send the sentences
themselves as the names — `choice(question, collect(TEAM))` — and map the chosen
sentence back to your key, which is what [`nl_classify`](@ref) does
([Design note: what the model reads](@ref nl_dispatch_design)).

### [Score — place the text on an ordered rubric](@id system_one_score)

[`score`](@ref) takes the level descriptions lowest first; `levels[1]` is level
`0`, so three levels give a score in `0.0 … 2.0`. The [`ScoreAnswer`](@ref)
carries `score`, `confidence`, `legend` and `probabilities`, the last two keyed
by the **0-based level number** as an `Int`. `score` is the probability-weighted
average level, so it falls between levels: threshold it, and read
`probabilities` when the shape matters — a `1.0` can be all the probability on
level 1, or half on level 0 and half on level 2.

```@example system_one
severity = score("How severe is the reported issue?", [
    "Cosmetic; no impact to functionality",
    "Broken or degraded feature, but workaround exists",
    "Blocking issue; no workaround exists"])
nothing # hide
```

Describe situations, not degrees: a level is a description the state is
matched against, so "Broken or degraded feature, but workaround exists" gives
the model something to match, where "worse than the previous level" names no
situation at all. Keep one dimension per question.

### [Noul — the probability that a statement is true](@id system_one_noul)

[`noul`](@ref) asks one yes/no question. The [`NoulAnswer`](@ref) is one number
in `0 … 1` — near 1 a strong yes, near 0 a strong no, near 0.5 maximal
uncertainty — and has no `confidence` field: with two outcomes the value is the
whole distribution.

```@example system_one
wants_human = noul("Is the customer asking for a human agent?")

repeat_contact = noul("Has the customer contacted support about this before?";
    yes = "Mentions a prior attempt, ticket, or that they have asked before",
    no  = "No sign of any previous contact")
nothing # hide
```

A Noul is the probability that the statement holds, not a degree: "Is the
candidate strong in Python?" at 0.5 means the model is split, **not** that the
candidate is middling — ask a Score for degree. Phrase the question so that a
high value means yes.

### [Structured criteria](@id system_one_structured)

Guidance — `instructions`, a Choice option's description, a Score level, a
Noul's `yes` and `no` — is a string to start with. The question constructors also
take an object (a `NamedTuple` or a `Dict`), a vector or a tuple, or an image
wherever guidance appears, so that two options that keep getting confused can each say
what they cover, what they do not, and a few examples. Not every provider's API
has room for structure, or for every guidance field: a wrapper refuses what its
provider cannot carry, before anything is sent, rather than send a question you
did not write ([What each provider takes](@ref backends_table) compares them, and
[Structured guidance](@ref jev_structured) shows such a Choice answered).

### [Pointing a question at part of the state](@id system_one_pointing)

When the state holds several parts — a message, a record and a policy — name the
part a question is about with its key in backticks, and keep the content in the
state rather than in the question:

```@example system_one
state = (
    ticket_message = "I was charged twice for order A-104. Please refund the duplicate.",
    order          = (id = "A-104", charges = [49, 49]),
    refund_policy  = "Duplicate charges are eligible for a refund.")

r = ask(state,
    "refund_requested" => noul("Does `ticket_message` request a refund?"),
    "policy_allows"    => noul("Does `refund_policy` allow refunding the duplicate charge in `order`?");
    service, model)

println("refund_requested: ", answer(r, "refund_requested"))
println("policy_allows:    ", answer(r, "policy_allows"))
```

The backticks are a prompting convention the model reads, not a resolver: a
name that does not exist raises no error, it only leaves the question vaguer.
The state may be a `String`, a `NamedTuple`, a `Dict`, a `Vector`, a `Tuple` or
an [`ImageAttachment`](@ref) — not `nothing` — and the entry of a provider that
receives it in another form, such as one JSON text or its pictures apart from its
texts, says how. A state's keys go out in their order, which can move an answer
as the option order does, so prefer the ordered forms.

## [Reading answers](@id Reading-answers)

Any call that reaches the provider returns one of three result types. A malformed
request is an `ArgumentError` before anything is sent, on every provider: a
`service` that serves no decisions, a blank or repeated question name, criteria
that are not valid, a state or guidance holding `NaN` or `Inf`, an
[`AudioAttachment`](@ref) or a [`DocumentAttachment`](@ref) anywhere in the
request, and what a provider's wrapper refuses on its own account, such as an
image's `detail` level where its wire has no field for one, which its entry's
Limits and refusals section lists.
Inside a [`with_recorded_answers`](@ref) replay scope, a request with no recording
throws [`ReplayMissError`](@ref): a gap in the recordings is not a service
failure.

| Result | Meaning |
| :--- | :--- |
| [`SystemOneSuccess`](@ref) | HTTP 200, decoded into answers |
| [`SystemOneFailure`](@ref) | the provider answered with a status other than 200, which its entry's failure table explains |
| [`SystemOneCallError`](@ref) | no usable response: a timeout, a transport failure, a missing key, or a 200 whose body was not a usable set of answers |

A failed call has no answers: [`answers`](@ref), [`answer`](@ref) and
`getindex` throw [`SystemOneError`](@ref) on it rather than returning an empty
map, so `r["is_unsafe"].noul > 0.9` can never read as "safe" on a call that
never happened. Where a failed call is a case your code handles, branch on
`r isa SystemOneSuccess` first, as the first example does. On a success, three
equivalent accessors reach an answer:

```@example system_one
ticket = "Help! My payouts have been failing for 3 days and nobody has replied to my emails."

r = ask(ticket,
    "urgency"     => score("How urgent is this ticket?",
                           ["Can wait", "Needs attention this week", "Needs attention today"]),
    "wants_human" => noul("Is the customer asking for a human agent?");
    service, model)

a = r["urgency"]              # getindex
a = answer(r, :urgency)       # by Symbol or String
every = answers(r)            # Dict{String,SystemOneAnswer}

println(a.score)              # probability-weighted position over the levels
println(a.confidence)         # how peaked the distribution is
println(a.probabilities)      # Dict{Int,Float64}, keyed by 0-based level number
println(a.legend[argmax(a.probabilities)])   # the description of the top level
println(a.raw)                # the answer as the provider sent it, parsed, always kept

haskey(r, "wants_human") && println(r["wants_human"].noul)
println(collect(keys(r)))     # the question names that came back
```

A provider may round the numbers it sends, so a distribution need not sum to
exactly 1: [`ask`](@ref) accepts a sum within a tolerance set per provider, which
its entry's Confidence section gives, and returns a reply further off as a
[`SystemOneCallError`](@ref). A Score answer is held to its own numbers the same
way: its `score` must be the expected level its probabilities give, within n − 1
times that tolerance plus one rounding step for a Score of n levels, and its
`legend` must repeat each level you asked as a string, unchanged. A reply that breaks either, chooses an option that was not offered,
leaves a question unanswered, or names no model is a `SystemOneCallError` too: no
success is built from a malformed answer.

**Confidence describes the distribution, not correctness.** It collapses
`probabilities` into `0 … 1`: all the mass on one outcome gives 1.0, a flat
spread a low number. For a Choice over `n` options it is
`clamp((n·p_max − 1)/(n − 1), 0, 1)`, the top probability `p_max` rescaled so
that a uniform answer scores 0. The same confidence is therefore a different top
probability for a different number of options: 0.5 means `p_max ≥ 0.75` with 2
options, 0.6 with 5 and 0.55 with 10; a Choice of one option reports 1.0. A
Score's confidence likewise measures how concentrated the probability is around
the most likely level. Low confidence on a Choice usually means no option wins;
on a Score, that the levels overlap for this state, the question measures more
than one thing, or the state does not say enough. Who computes `confidence`
differs per provider, and so does how well a model's probabilities are
calibrated ([One formula, calibrated per model](@ref backends_confidence)).

!!! details "Evidence"
    TypeSafe publishes the formula on its
    [Confidence](https://docs.typesafe.ai/confidence) page. It fits every answer
    we measured within 0.02, on jev-1.13.0 in September 2026.

**Answers can vary between identical calls.** A provider need not answer a
repeated request bit for bit: noise on a clear-cut input, but a decision near a
threshold can flip between two identical calls. Log what a decision was based
on — the answer's `raw`, `r.response.request_id` and `r.response.model` — and,
for reproducible tests and docs, record an answer once and replay it with
[`with_recorded_answers`](@ref) ([Test and Develop](@ref jev_testing_guide)).

!!! details "Evidence"
    Measured on jev-1.13.0 (September 2026), identical requests differed in 60%
    of repeated pairs, by 0.011 in a probability on average and by up to 0.11.

## [Ask many questions at once](@id jev_many_questions)

**The job:** learn several things about one message — which team, why, how
upset, whether the customer wants a person — without paying for the message
once per question. **Without a decision model:** one classifier or one prompt per
question, or one long prompt whose answer you parse. **With one:** one
[`ask`](@ref) carries every question; the text is sent once, and each answer
comes back under its own question's name.

A further question adds its own tokens to the request. Where a provider reads
the state once for all the questions, it barely moves the latency either, which
makes **speculative** questions worth asking: include the ones whose answers
matter only for some inputs, and let the code ignore the rest. A local model may
score a request's questions one after another instead
([Throughput](@ref ollama_decisions_throughput)).

!!! details "Evidence"
    TypeSafe's [parallel-questions
    cookbook](https://docs.typesafe.ai/cookbooks/parallel_questions) runs a
    13-question briefing on `jev-1.12` and measures it batched into one call at
    12.2x cheaper and 10.0x faster than 13 separate calls, with no change in the
    answers. Measured on jev-1.13.0 (September 2026), one question took 0.30 s
    and 22 questions 0.31 s.

```@example system_one
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

r = ask("Shoes arrived two weeks late and in the wrong size. What are you going to do?", TRIAGE; service, model)

d = r["department"]
detail = d.choice == "returns"  ? r["return_reason"].choice :
         d.choice == "shipping" ? r["shipping_issue"].choice : nothing
# Both speculative answers came back; the code reads at most one and drops the other.
println("department: ", d.choice, ", detail: ", detail)
for (team, p) in d.probabilities
    team != d.choice && p > 0.25 && println("also notify: ", team)
end
println("input tokens: ", token_usage(r).prompt_tokens)
```

**Tune it**

- End every enumeration that might not cover an input with an `other` option,
  as two of the three Choices above do: otherwise the probability has nowhere to
  go but onto an option that does not fit.
- When one question depends on another's answer — because that answer decides
  what to retrieve, or which options to offer next — send a second request once
  the first answer is in; otherwise ask the questions together.
- To judge many items in one request, put each under its own key in the state
  and name the key in the question — "Which team should handle ticket `t17`?" —
  as [Many Items at Once](@ref jev_algorithms_guide) does; addressing them by
  array index fails as the list grows.

!!! details "Evidence"
    Addressing items by array index (`items[17]`) collapsed to 33% accuracy at
    150 items on our own labeled set (jev-1.13.0, September 2026).

## [Fill a struct in one request](@id jev_fill_struct)

**The job:** turn a support message into a typed record — which team, how
urgent, whether the customer wants a refund — that your code can store and act
on. **Without a decision model:** an LLM filling a JSON schema, whose output you
then validate and parse, or one classifier per field. **With one:** a struct
already says what each field may hold, so each field type maps to a question — an
enum to a [`choice`](@ref), an ordered enum to a [`score`](@ref), a `Bool` to a
[`noul`](@ref) — and one request fills them all, with a certainty per field.

Each enum value is offered to the model as its sentence in `DESCRIPTIONS`: the
sentence is the option's name, and the sentence the model picks maps back to the
value, so a value's name never reaches the model — the same idea as
`nl_classify` ([Dispatch on Meaning](@ref nl_dispatch_guide)). The constructor
only ever receives values its field types allow.

```@example system_one
@enum Team billing technical shipping
@enum Urgency low normal high

struct Ticket
    team::Team
    urgency::Urgency
    wants_refund::Bool
end

const QUESTIONS = (team         = "Which team should handle this ticket?",
                   urgency      = "How urgent is this ticket?",
                   wants_refund = "Does the customer ask for their money back?")

# One table per enum: each value with the sentence the model reads for it, in the order it reads them.
const DESCRIPTIONS = Dict(
    Team    => [billing   => "Charges, invoices, refunds, payment methods, subscription prices",
                technical => "Bugs, crashes, errors, outages, integration or configuration problems",
                shipping  => "Delivery of physical goods: tracking, delays, damaged or missing parcels"],
    Urgency => [low    => "Low: a question or a minor annoyance; no loss of money, data or access",
                normal => "Normal: something is wrong, but there is a workaround or no deadline",
                high   => "High: the customer is blocked, is losing money, or has a deadline within days"])
sentences(E) = last.(DESCRIPTIONS[E])

ordinal(::Type) = false
ordinal(::Type{Urgency}) = true         # an ordered rubric: asked as a Score, not a Choice

question(::Type{Bool}, q) = noul(q)
question(E::Type{<:Enum}, q) = ordinal(E) ? score(q, sentences(E)) : choice(q, sentences(E))

# The chosen sentence, or the most likely level, back to its enum value.
value(::Type{Bool}, a::NoulAnswer) = a.noul >= 0.5
value(E::Type{<:Enum}, a::ChoiceAnswer) = only(v for (v, s) in DESCRIPTIONS[E] if s == a.choice)
value(E::Type{<:Enum}, a::ScoreAnswer) = first(DESCRIPTIONS[E][argmax(a.probabilities) + 1])   # levels count from 0
certainty(a::NoulAnswer) = abs(2a.noul - 1)             # 0 at a coin flip, 1 at a sure yes or no
certainty(a::Union{ChoiceAnswer,ScoreAnswer}) = a.confidence

# One request for the whole struct. A failed call throws: it never becomes a default T.
function extract(::Type{T}, text) where {T}
    fields = fieldnames(T)
    r = ask(text, [String(f) => question(fieldtype(T, f), QUESTIONS[f]) for f in fields]; service, model)
    (T((value(fieldtype(T, f), answer(r, f)) for f in fields)...),
     NamedTuple{fields}(map(f -> certainty(answer(r, f)), fields)))
end

ticket, certainties = extract(Ticket, "I was charged twice this month. Refund the duplicate now!")
for f in fieldnames(Ticket)
    println(rpad(f, 13), rpad(getfield(ticket, f), 9), "certainty ", certainties[f])
end
```

The ticket comes back as a `billing` ticket of `high` urgency that asks for a
refund, and urgency is its least certain field (0.72).

On 32 hand-labeled tickets, one request per ticket filled a five-field version
of this struct at 0.979 accuracy per field. A confidence threshold tuned for one
option wording may not carry over to another: with the sentences as option
names, the Choice confidence on that set came out slightly lower at the same
accuracy.

!!! details "Evidence"
    Measured on jev-1.13.0 (September 2026) on 32 hand-labeled tickets and a
    five-field version of this struct, both encodings in one session, runs
    interleaved. Sentences as the option names (this recipe): 0.979 per field
    over three runs (0.981, 0.981, 0.975); exact-match rate 0.896. Labels with
    the sentences as their descriptions (the other common encoding), same session:
    0.981 in all three runs; exact-match rate 0.906. The whole gap
    is one near-tie flip (0.40 vs 0.39) in the urgency Score, whose question is
    identical under both encodings; neither encoding made an error on a Choice
    field. Across the three runs, 1 of 480 predictions changed: that urgency
    near-tie.

    The three-field struct above: both encodings 0.974 per field (team 1.0 on 21
    gradable items, urgency 0.923 on 26, wants_refund 1.0 on 32).

    Choice confidence was somewhat lower with the sentences as option names
    (department: mean 0.894 vs 0.910; 24 vs 18 of 96 answers below 0.8), with no
    change in accuracy. Input tokens per request: 627 with the sentences as
    option names, 706 with labels and descriptions.

    From an earlier run on the same items and questions: gpt-5.6-luna filling a
    strict JSON schema reached 0.967 per field, at a median latency of 1.3–1.5 s
    against 0.28 s for Jev, and changed 1–3% of its fields between runs.

    Limits: 32 items, one annotator. The Choice fields are at ceiling under both
    encodings, so this set can show a loss but not a gain; the encoding study on
    keyword traps, in Dispatch on Meaning's [design note](@ref
    nl_dispatch_design), is where sentences measurably beat labels.

**Tune it**

- Keep the questions and sentences in code, as `QUESTIONS` and `DESCRIPTIONS`
  do. Field docstrings look like their natural home, but reading them goes
  through Julia's non-public docstring internals, and they are silently absent
  when the struct itself has no docstring.
- An ordered enum is a rubric: mark it `ordinal`, so it is asked as a Score,
  and list its sentences lowest first.
- Read `certainties` before you store a record: a low value marks the field to
  check.

## [Act on the answer](@id jev_act_on_answer)

**The job:** send each message to the team that handles it, and hand the ones
the model is unsure about to a person. **Without a decision model:** keyword rules
per team, and a queue for whatever they miss. **With one:** one Choice per
message, and its `confidence` says when to hand over.

```@example system_one
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

# The sentences are the options, so the model reads only those; `findfirst` maps the chosen one to its key.
const WHICH_TEAM = choice("Which team should handle this message?", collect(TEAM))

for message in MESSAGES
    answered = ask(message, "team" => WHICH_TEAM; service, model)["team"]
    team = findfirst(==(answered.choice), TEAM)            # :billing, :technical, :shipping or :other
    println(rpad(answered.confidence >= 0.8 ? team : :person, 10), rpad(answered.confidence, 6), message)
end
```

Four messages go straight to their team. Two go to a person: the question about
delivery to Norway (confidence 0.67), and the cracked phone (0.75), a damaged
parcel and a refund request in one message.

An answer is a distribution: the answer says *what*, the distribution says
*whether to act on it*. A model's calibration holds across groups of answers, not
for any single one — a high confidence says the distribution is peaked, not that
this answer is right — so each policy below is one rule applied to every answer,
and how well a model is calibrated is measured per provider ([One formula,
calibrated per model](@ref backends_confidence)).

!!! details "Evidence"
    Measured on jev-1.13.0 (September 2026) on our own labeled sets: over 129
    routed tickets the expected calibration error was 0.015, and the 113 answers
    whose top probability was at least 0.9 were 99.1% correct. Other sets
    measured higher errors: 0.047 on a stress set of 100 items built on the
    model's documented failure modes, and 0.071 on 74 yes/no (Noul) answers,
    which were underconfident.

**Tune it**

- The bar is your policy, not the model's: a read-only action can act at a much
  lower confidence than a refund. Start conservative, measure on your own
  messages, and move it — and pin the model first (`model = …` on the call, or a
  [`with_system_one`](@ref) scope's), since a tuned bar is a claim about one model
  version and one provider: on our labelled tickets the same 0.8 sent twice as
  many to a person on one provider as on another
  ([One formula, calibrated per model](@ref backends_confidence)).
- Confidence is rescaled for the number of options, so adding a team moves the
  bar ([Reading answers](@ref Reading-answers)); rewording the options moves it too
  ([Fill a struct in one request](@ref jev_fill_struct)).
- `nl_classify` packages this pattern — the sentences as options, the key back,
  a `min_confidence` gate or a `decide` policy — in one call ([Dispatch on
  Meaning](@ref nl_dispatch_guide)).

### [When mistakes have different prices](@id jev_loss_matrix)

**The job:** route tickets when a wrong refund costs far more than a wrong
tracking link, with "ask a person" as one of the actions at its own price.
**Without priced mistakes:** one confidence threshold for every action, tuned on
labeled tickets. **With them:** write down the price of each mistake; calibrated
probabilities give every action's expected cost in one line of arithmetic, and
the cheapest action — the Bayes action — wins.

```@example system_one
# Minutes of staff time. A person resolves any ticket in HUMAN minutes. A wrong automated
# action does its own harm and still ends with that person, so it is priced as both.
const HUMAN = 2.0
const HARM = (refund = 8.0, cancel_subscription = 10.0, shipping_delivery = 1.0, other = 0.5)
cost(action, truth) = action === :escalate ? HUMAN : action === truth ? 0.0 : HARM[action] + HUMAN

const WANTS = "What does the customer primarily want us to do?"
const OUTCOMES = (refund              = "The customer asks to get money back.",
                  cancel_subscription = "The customer wants to cancel or not renew a subscription.",
                  shipping_delivery   = "The status, tracking, delay or address of a delivery.",
                  other               = "None of the above.")
const INTENT = choice(WANTS, OUTCOMES)

function triage(message)
    a = ask(message, "intent" => INTENT; service, model)["intent"]
    expected(action) = sum(a.probabilities[String(t)] * cost(action, t) for t in keys(OUTCOMES))
    action = argmin(expected, (:escalate, keys(OUTCOMES)...))           # the Bayes action; a tie escalates
    (action = action, expected_cost = round(expected(action); digits = 2),
     argmax = a.choice, confidence = a.confidence)
end

const TICKETS = ("Where is my parcel? Tracking hasn't moved in a week.",
                 "I was charged but my order never shipped.",
                 "Your update broke everything. I'm done with you, and I want this month back.")
for ticket in TICKETS
    println(ticket, "\n  => ", triage(ticket))
end
```

The second ticket's most likely intent is a refund (confidence 0.69), yet the
rule escalates it: a wrong refund costs 10 minutes against a person's 2, and at
these probabilities the person is cheaper. The third ticket is refunded
(expected cost 1.0), and the first goes to shipping at no expected cost.

On our labeled tickets this rule had a lower held-out cost than the best tuned
threshold (56.5 against 83.5), and it needs no labels to set — only prices.

!!! details "Evidence"
    Measured on jev-1.13.0 (September 2026) with this question and this cost
    table, on our 129 labeled tickets (their labels mapped onto these four
    outcomes, two-fold cross-validation): the Bayes rule's held-out cost was
    56.5 against 83.5 for the best `min_confidence` tuned on the other fold, and
    the 95% bootstrap interval of the paired difference, −53.0 to −8.0, excluded
    zero. The Bayes rule escalated 2 of the tickets, the tuned threshold 18. The
    threshold needed labeled tickets to tune; the Bayes rule needs none — only
    costs.

    An earlier table of ours priced a wrong `other` below escalation, and its
    Bayes rule escalated no ticket at all.

**Tune it**

- Price a wrong automated action as its harm **plus** the person who then fixes
  it. Priced at its harm alone, a cheap action such as `other` (0.5) would
  undercut escalation (2.0) under every distribution and become a free hedge:
  nothing would ever reach a person.
- The rule trusts the probabilities to be calibrated: check them on a provider
  before you move the rule there ([One formula, calibrated per
  model](@ref backends_confidence)).
- The same arithmetic can decide inside `nl_classify`, [`nl_dispatch`](@ref)
  and [`@branch`](@ref): their `decide` keyword takes a function of the answer
  that returns the option to act on, or `nothing` to hand the case to
  `fallback`. A threshold is itself such a policy, so `decide` replaces
  `min_confidence` ([Confidence and control](@ref nl_dispatch_control)).

### [Let the model say none of these](@id jev_none_of_these)

**The job:** notice the messages that fit none of your options, instead of
forcing each onto one. **Without a catch-all:** read low confidence as "out of
scope", which misses the out-of-scope messages a model routes confidently.
**With one:** add a catch-all option; its probability is the out-of-scope
signal. Here the same message is asked against a closed and an open option set,
as two questions in one request:

```@example system_one
const IN_SCOPE = ["refund"           => "The customer asks to get money back.",
                  "technical_issue"  => "The product, app or website is malfunctioning.",
                  "billing_question" => "A question about a charge, invoice, price or payment method."]
const CLOSED   = choice(WANTS, IN_SCOPE)
const OPEN_SET = choice(WANTS, [IN_SCOPE; "other" => "None of the above."])

for message in ("The export button crashes the app.",           # in scope
                "Please add a dark mode, it would be great.",   # out of scope
                "What's the capital of Australia?")             # out of scope
    both = ask(message, "closed" => CLOSED, "open_set" => OPEN_SET; service, model)
    closed, open_set = both["closed"], both["open_set"]
    println(rpad(message, 44), " closed: ", rpad(closed.choice, 16), " confidence ", closed.confidence,
            " | open: ", rpad(open_set.choice, 16), " P(other) ", open_set.probabilities["other"])
end
```

In the closed set both out-of-scope messages went to `technical_issue`,
"Please add a dark mode" at confidence 1.0. With a catch-all, `P(other)` took
both at 1.0, and the in-scope message kept its team.

On our labeled set, the catch-all's probability separated the out-of-scope
messages perfectly; one minus the top probability of the closed question did
not.

!!! details "Evidence"
    Measured on jev-1.13.0 (September 2026) on our labeled set (25 out-of-scope
    and 104 in-scope tickets): `P(other)` from a catch-all option separated the
    out-of-scope inputs perfectly (AUROC 1.00), while one minus the top
    probability of the same question without the catch-all did not (AUROC 0.86);
    the extra option left in-scope accuracy unchanged (0.962 either way).

**Tune it**

- A confidence threshold is not an out-of-scope detector; a catch-all option
  is. Add one wherever the options might not cover every input.
- On our set the catch-all cost no in-scope accuracy, so there is little reason
  to leave it out.

### [Act on the risk of the worst level](@id jev_worst_level)

**The job:** page someone when a report might be critical, even when it is
probably not. **Without the tail:** a severity label or an average score, which
hides a small but real chance of the worst case. **With it:** read the
probability mass at and above the worst level, and page when it beats the ratio
of your prices. A [`ScoreAnswer`](@ref)'s `score` is the probability-weighted
average — fine for ranking, misleading for a decision whose cost sits on one
level.

```@example system_one
const SEVERITY = score("How severe is the problem this customer reports?", [
    "No problem: a question, praise, or feedback",
    "Minor: a cosmetic issue or small inconvenience; everything still works",
    "Moderate: a feature is broken or degraded, but there is a workaround",
    "Major: a core feature is unusable for this customer, with no workaround",
    "Critical: data loss, a security breach, a physical safety risk, or an outage"])

# P(level ≥ k): the probability mass at or above rung k (levels count from 0).
at_least(a::ScoreAnswer, k) = sum(p for (level, p) in a.probabilities if level >= k; init = 0.0)

# A false page costs 1, a missed critical report costs 10: page when P(critical) > 1 / (1 + 10).
for message in ("The tooltip on the export icon is misspelled.",
                "The heater smelled like burning for a moment. Might have been dust, might not.",
                "Our whole customer database was deleted after your update.")
    answered = ask(message, "severity" => SEVERITY; service, model)["severity"]
    println(rpad(message, 80), " score ", answered.score, "  P(critical) ", round(at_least(answered, 4); digits = 2),
            "  page: ", at_least(answered, 4) > 1 / 11, "  (score ≥ 3.5: ", answered.score >= 3.5, ")")
end
```

The heater report's score, 2.24, sits in the middle of the scale, where a
threshold at 3.5 does not fire, yet it puts 0.4 on "critical" — far above the
1/11 that pays for a page.

Paging pays when the expected cost of silence, 10 · P(critical), exceeds that
of a false alarm, 1 · (1 − P(critical)) — that is, when P(critical) > 1/11. The
two rules part only on split distributions like the heater's; on our labeled
severity reports they never did.

!!! details "Evidence"
    Measured on jev-1.13.0 (September 2026): the two rules part only on such
    split distributions, and on our 75 labeled severity reports none occurred:
    both made identical decisions.

**Tune it**

- The cut is the ratio of your prices: with a false page at `c_false` and a
  missed critical report at `c_miss`, page when P(critical) >
  `c_false / (c_false + c_miss)`.
- Keep `score` for ranking and for thresholds on the middle of the scale; read
  the tail when one level carries the cost.

## [Designing good questions](@id jev_designing_questions)

- **One narrow judgment per question.** Ask what a knowledgeable person decides
  in a second given the right context. "Does this message convey urgency?" is a
  question; "analyse this and decide what to do" is a workflow — split it and
  compose the answers in code.
- **Judgment in `instructions`, answer space in `criteria`.** The instruction
  says what is being decided; the options, levels or `yes`/`no` descriptions say
  what the outcomes are. When the two disagree, accuracy drops.
- **Always offer a way out.** Add an `other` option whenever the enumeration
  might not cover an input ([Let the model say none of these](@ref
  jev_none_of_these)).
- **Keep the state relevant.** Accuracy falls as unrelated material grows around
  the decision: retrieve and filter in code, and send only the fields the
  question needs.
- **The model reads literally.** Scoping words, negations and implied conditions
  are taken at face value. When you find yourself explaining what you *really*
  meant, that explanation is the missing half of the instruction.
- **No arithmetic, no date comparison.** Counting, numeric magnitude and
  ordering dates are unreliable: extract the parts as a Choice over enumerated
  options — including an explicit "not stated" — and let code compare them. Do
  not read a real number out of a Score either; a Score is for thresholding.
- **Do not carry thresholds across primitives.** A Noul and a yes/no Choice
  answer different questions — the Choice settles *which*, each Noul is
  absolute — and `P(q)` and `1 - P(not q)` are not guaranteed to agree.
- **Treat the state as untrusted text.** Content written to steer a classifier
  can move the answer. Be explicit in the criteria, and test adversarial inputs
  before you ship.

Check them against what your provider documents its model to do poorly, and
again when you move a pin.

!!! details "Evidence"
    These rules restate the failure modes TypeSafe documents for Jev 1.13
    ([Jev 1.13 jaggedness](https://docs.typesafe.ai/model-jaggedness/jev-1.13)),
    with this manual's measurements on jev-1.13.0 in September 2026.

## [With an LLM](@id system_one_with_llm)

A decision model decides and a language model writes, so they meet on both sides
of a generation: before it, [route the message](@ref jev_llm_route) and decide
whether and how to generate; after it, [check the draft](@ref jev_llm_check_draft)
before a customer sees it and [check its claims against the source](@ref
jev_llm_check_claims). [Decisions with LLMs](@ref jev_llm_guide) builds each step.

## [See also](@id system_one_see_also)

- [Decisions](@ref decisions_start) — the decision providers, each with its entry
- [Choosing a decision provider](@ref system_one_backends) — the same calls on
  each provider, and what each one takes
- [Decisions with LLMs](@ref jev_llm_guide) — a decision before and after a
  language model's call
- [Dispatch on Meaning](@ref nl_dispatch_guide) — `nl_classify`,
  [`nl_dispatch`](@ref) and [`@branch`](@ref): the answer selects the code that
  runs
- [Many Items at Once](@ref jev_algorithms_guide) — many judgments per request
  over collections: routing, ranking, search and joins
- [Test and Develop](@ref jev_testing_guide) — recorded answers and tests that
  need no key
- [Cost Tracking](@ref cost_guide) — token usage and USD estimates across every
  surface
- [`ask`](@ref), [`SystemOneSuccess`](@ref), [`ChoiceAnswer`](@ref),
  [`ScoreAnswer`](@ref) and [`NoulAnswer`](@ref) in the API reference

## [On your provider](@id system_one_providers)

The question kinds each provider answers, what of a question reaches its model,
and its limits:

```@eval
Main.ProviderFacts.on_your_provider(Main.ProviderFacts.decisions, :questions)
```

Who computes `confidence`, the tolerance `ask` allows, and how a bar is tuned:

```@eval
Main.ProviderFacts.on_your_provider(Main.ProviderFacts.decisions, :confidence)
```

How a failed call comes back, and what is sent again:

```@eval
Main.ProviderFacts.on_your_provider(Main.ProviderFacts.decisions, :failures)
```
