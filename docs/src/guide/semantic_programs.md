# [Semantic Programs with Jev](@id jev_programs_guide)

A Jev answer is a calibrated probability distribution over outcomes you
enumerated, delivered as a typed value. Measured on jev-1.13.0 (September 2026)
on our own labeled sets: over 129 routed tickets the expected calibration error
was 0.015, and the 113 answers whose top probability was at least 0.9 were
99.1% correct. Other sets measured higher errors: 0.047 on a stress set of 100
items built on the model's documented failure modes, and 0.071 on 74 yes/no
(Noul) answers, which were underconfident. Julia turns such values into
programs cheaply — a decision rule is one line of arithmetic, a taxonomy is a
type hierarchy, a record is a struct, a protocol is a method table. Each section
below is a system that would otherwise need a trained model, an LLM plus
parsing, or hand-maintained tables.

Every block that calls Jev runs when this page is built, so what it prints is a
real answer — a live call, or a recorded live answer when the build has no API
key. The numbers in the prose are measurements, not what a block prints.

## Decide with a loss matrix, not a threshold

`min_confidence` asks one question — is the model sure enough? — and answers it
the same way for a refund and for a tracking link. A routing decision needs the
price of each mistake. Write the prices down as a loss matrix, and the
calibrated distribution gives the expected cost of every action in one line of
arithmetic; the Bayes action is the cheapest one. Escalating to a person is one
of the actions, at a fixed price.

```@example jevprog
using UniLM

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
    a = ask(message, "intent" => INTENT)["intent"]
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

Price a wrong automated action as its harm **plus** the person who then fixes
it. Priced at its harm alone, a cheap action such as `other` (0.5) would
undercut escalation (2.0) under every distribution and become a free hedge —
nothing would ever reach a person (an earlier table of ours priced a wrong
`other` below escalation, and its Bayes rule escalated no ticket at all).
Measured on jev-1.13.0 (September 2026) with this question and this cost table,
on our 129 labeled tickets (their labels mapped onto these four outcomes,
two-fold cross-validation): the Bayes rule's held-out cost was 56.5 against 83.5
for the best `min_confidence` tuned on the other fold, and the 95% bootstrap
interval of the paired difference, −53.0 to −8.0, excluded zero. The Bayes rule
escalated 2 of the tickets, the tuned threshold 18. The threshold needed labeled
tickets to tune; the Bayes rule needs none — only costs.

The same rule runs inside [`nl_dispatch`](@ref) as its `decide` policy. The
policy receives each slot's [`ChoiceAnswer`](@ref) and returns the offered
meaning to dispatch on — whether or not it is the argmax — or `nothing` to
decline, which calls `fallback` with the caller's arguments (without a
`fallback`, a decline throws [`DecisionDeclinedError`](@ref)). The option names
are the meanings themselves, here the values of `OUTCOMES`:

```@example jevprog
route(::nl"The customer asks to get money back.", ticket)                      = :refund
route(::nl"The customer wants to cancel or not renew a subscription.", ticket) = :cancel_subscription
route(::nl"The status, tracking, delay or address of a delivery.", ticket)     = :shipping_delivery
route(::nl"None of the above.", ticket)                                        = :other
escalate(ticket) = :escalate

function cheapest(a::ChoiceAnswer)
    expected(action) = sum(a.probabilities[OUTCOMES[t]] * cost(action, t) for t in keys(OUTCOMES))
    best = argmin(expected, (:escalate, keys(OUTCOMES)...))
    best === :escalate ? nothing : OUTCOMES[best]            # declining hands the ticket to `fallback`
end

for ticket in TICKETS
    action = nl_dispatch(route, ticket; state = ticket, instructions = WANTS, decide = cheapest, fallback = escalate)
    println(rpad(ticket, 78), repr(action))
end
```

With `state` and `instructions`, the model reads what the first version sent in
all but the option text — each sentence is now an option's name instead of a
key's description — and that alone can give a borderline ticket a different
distribution, and with it a different action.

A threshold is a policy too — `a -> a.confidence >= τ ? a.choice : nothing` —
so `decide` replaces `min_confidence` rather than stacking on it: combining it
with a non-zero `min_confidence` is an `ArgumentError`.

## Let the model say "none of these"

A Choice always names a winner, and `confidence` only says how peaked the
distribution over *your* options is. An input that fits none of them still has
to put its probability somewhere, and it can put all of it on one wrong option.
Give it an explicit catch-all instead. Below, the same message is asked against
a closed and an open option set, as two questions in one request:

```@example jevprog
const TEAMS = ["refund"           => "The customer asks to get money back.",
               "technical_issue"  => "The product, app or website is malfunctioning.",
               "billing_question" => "A question about a charge, invoice, price or payment method."]
const CLOSED   = choice(WANTS, TEAMS)
const OPEN_SET = choice(WANTS, [TEAMS; "other" => "None of the above."])

for message in ("The export button crashes the app.",           # in scope
                "Please add a dark mode, it would be great.",   # out of scope
                "What's the capital of Australia?")             # out of scope
    r = ask(message, "closed" => CLOSED, "open_set" => OPEN_SET)
    closed, open_set = r["closed"], r["open_set"]
    println(rpad(message, 44), " closed: ", rpad(closed.choice, 16), " confidence ", closed.confidence,
            " | open: ", rpad(open_set.choice, 16), " P(other) ", open_set.probabilities["other"])
end
```

Measured on our labeled set (25 out-of-scope and 104 in-scope tickets):
`P(other)` from a catch-all option separated the out-of-scope inputs perfectly
(AUROC 1.00), while one minus the top probability of the same question without
the catch-all did not (AUROC 0.86); the extra option left in-scope accuracy
unchanged (0.962 either way). When we ran this example, "Please add a dark
mode" went to `technical_issue` at top probability 1.00 in the closed set.
`min_confidence` is not an out-of-scope detector; a catch-all option is.

## Ordinal decisions: read the tail, not the average

A [`ScoreAnswer`](@ref)'s `score` is the probability-weighted average level —
fine for ranking, misleading for a decision whose cost sits on one level. When
missing the top level is expensive, read the probability mass at and above it,
and act when that exceeds the cost ratio:

```@example jevprog
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
    a = ask(message, "severity" => SEVERITY)["severity"]
    println(rpad(message, 80), " score ", a.score, "  P(critical) ", round(at_least(a, 4); digits = 2),
            "  page: ", at_least(a, 4) > 1 / 11, "  (score ≥ 3.5: ", a.score >= 3.5, ")")
end
```

Paging pays when the expected cost of silence, 10 · P(critical), exceeds that
of a false alarm, 1 · (1 − P(critical)) — that is, when P(critical) > 1/11. A
hedged report can carry real probability on "critical" while its expected
score stays in the middle of the scale, where a threshold on `score` does not
fire. The two rules part only on such split distributions: on our 75 labeled
severity reports none occurred, and both made identical decisions.

## Taxonomies are type hierarchies

A category tree is a type hierarchy: abstract types for the inner nodes,
singleton structs for the leaves. Ask one Choice over the leaves, sum the
probabilities up the tree with `<:`, and take the most specific node whose
probability clears a threshold. The handler is ordinary dispatch on that node:
`desk(::Type{<:Billing}, t)` catches the billing tickets that
`desk(::Type{Refund}, t)` does not, by Julia's own specificity rules, with no
table mapping nodes to handlers.

```@example jevprog
abstract type Intent end
abstract type Billing <: Intent end
abstract type Technical <: Intent end
struct Refund <: Billing end
struct DoubleCharge <: Billing end
struct Crash <: Technical end
struct Login <: Technical end
struct Other <: Intent end

# The leaves the model chooses between, in the order it reads them.
const LEAVES = [Refund       => "the customer wants their money back",
                DoubleCharge => "the customer was charged twice or charged an amount they did not expect",
                Crash        => "the app crashes, freezes, or shows an error",
                Login        => "the customer cannot sign in to their account",
                Other        => "anything else"]
const LEAF = choice("Select the option that best describes the provided state.",
                    [text => nothing for (_, text) in LEAVES])

# Handlers at any level of the tree; Julia picks the most specific one for the chosen node.
desk(::Type{<:Intent}, ticket)    = :human_triage
desk(::Type{<:Billing}, ticket)   = :billing_desk
desk(::Type{<:Technical}, ticket) = :tech_desk
desk(::Type{Refund}, ticket)      = :refund_flow

function route_by_meaning(ticket; threshold = 0.7)
    p = ask((ticket = ticket,), "intent" => LEAF)["intent"].probabilities
    marginal(node) = sum(p[text] for (leaf, text) in LEAVES if leaf <: node)
    candidates = [first.(LEAVES); Billing; Technical]        # most specific first
    i = findfirst(node -> marginal(node) >= threshold, candidates)
    node = isnothing(i) ? Intent : candidates[i]
    (node = node, marginal = round(marginal(node); digits = 2), action = desk(node, ticket))
end

for ticket in ("Please refund my annual plan, I cancelled on day two.",
               "Two subscriptions renewed on my card. Please cancel one and return that payment.",
               "Since the update I can't get past the login screen, it just freezes.",
               "There is a charge from your company on my card that I don't recognize at all.")
    r = route_by_meaning(ticket)
    println(rpad(nameof(r.node), 13), rpad(r.marginal, 6), rpad(r.action, 14), ticket)
end
```

Measured at threshold 0.7 on 65 labeled tickets, each asked with the leaves in
four orders (260 decisions; correct / escalated / wrong): this rule
250 / 7 / 3; flat leaf routing with the same threshold 238 / 19 / 3; top-down
classification with one request per level 238 / 7 / 15. Summing up the tree
turned 12 escalations into coarse routes, all 12 correct, and added no wrong
one; asking level by level escalated as rarely but was wrong five times as
often.

**Types, not keys.** Each leaf's identity is a type scoped to its module, so
two packages can both define `Refund` without colliding. The text the model
reads is the sentence next to the type, never the type's name — short keys
measurably mislead the model; see [Design note: meanings are sentences, not
keys](@ref nl_dispatch_design).

## The struct is the prompt: typed extraction

A struct already states what a record may contain: an `@enum` field is a
closed set, an ordered enum is a rubric, a `Bool` is a yes-or-no. Map each
field type to its primitive — [`choice`](@ref), [`score`](@ref),
[`noul`](@ref) — and one request fills the whole struct, plus a certainty per
field. The constructor only ever receives values its field types allow.

```@example jevprog
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

# One line per enum value: the text the model reads for it, instead of a bare label.
const DESCRIPTIONS = (
    billing   = "Charges, invoices, refunds, payment methods, subscription prices",
    technical = "Bugs, crashes, errors, outages, integration or configuration problems",
    shipping  = "Delivery of physical goods: tracking, delays, damaged or missing parcels",
    low       = "Low: a question or a minor annoyance; no loss of money, data or access",
    normal    = "Normal: something is wrong, but there is a workaround or no deadline",
    high      = "High: the customer is blocked, is losing money, or has a deadline within days")
describe(x::Enum) = DESCRIPTIONS[Symbol(x)]

ordinal(::Type) = false
ordinal(::Type{Urgency}) = true         # an ordered rubric: asked as a Score, not a Choice

labels(E) = [string(x) for x in instances(E)]
descriptions(E) = [describe(x) for x in instances(E)]
question(::Type{Bool}, q) = noul(q)
question(E::Type{<:Enum}, q) = ordinal(E) ? score(q, descriptions(E)) : choice(q, labels(E) .=> descriptions(E))

value(::Type{Bool}, a::NoulAnswer) = a.noul >= 0.5
value(E::Type{<:Enum}, a::ChoiceAnswer) = instances(E)[findfirst(==(a.choice), labels(E))]
value(E::Type{<:Enum}, a::ScoreAnswer) = instances(E)[argmax(a.probabilities) + 1]   # levels count from 0
certainty(a::NoulAnswer) = abs(2a.noul - 1)             # 0 at a coin flip, 1 at a sure yes or no
certainty(a::Union{ChoiceAnswer,ScoreAnswer}) = a.confidence

# One request for the whole struct. A failed call throws: it never becomes a default T.
function extract(::Type{T}, text) where {T}
    fields = fieldnames(T)
    r = ask(text, [String(f) => question(fieldtype(T, f), QUESTIONS[f]) for f in fields])
    (T((value(fieldtype(T, f), answer(r, f)) for f in fields)...),
     NamedTuple{fields}(map(f -> certainty(answer(r, f)), fields)))
end

ticket, certainties = extract(Ticket, "I was charged twice this month. Refund the duplicate now!")
for f in fieldnames(Ticket)
    println(rpad(f, 13), rpad(getfield(ticket, f), 9), "certainty ", certainties[f])
end
```

Measured on a five-field version of this struct, with descriptions like these,
over 32 labeled tickets: per-field accuracy 0.981 against 0.967 for
gpt-5.6-luna filling a strict JSON schema, median latency 0.28 s against
1.3–1.5 s, and not one prediction changed across three repeated runs, while the
LLM changed 1–3% of its fields between runs. Keep the questions and
descriptions in code, as `QUESTIONS` and `DESCRIPTIONS` do. Field docstrings
look like their natural home, but reading them goes through Julia's non-public
docstring internals, and they are silently absent when the struct itself has no
docstring.

## Conversation state machines

A conversation is a state machine whose transitions are meanings. Give each
state a type and write one method per (state, meaning) transition;
[`nl_dispatch`](@ref) then offers the model only the meanings whose method
accepts the current state's type, so a transition that does not exist in this
state is never an option. [`meanings`](@ref) with the argument types previews
that list without a request:

```@example jevprog
abstract type Conversation end
struct AwaitingOrderId <: Conversation end
struct ConfirmingRefund <: Conversation
    order::String
end
struct Closed <: Conversation
    outcome::Symbol
end
# A state prints as its constructor call, e.g. Closed(:refunded).
Base.show(io::IO, s::Conversation) =
    print(io, nameof(typeof(s)), "(", join((repr(getfield(s, f)) for f in fieldnames(typeof(s))), ", "), ")")

asked(::AwaitingOrderId) = "Which order would you like refunded?"
asked(s::ConfirmingRefund) = "Shall I refund order $(s.order)? Please confirm."
asked(::Closed) = "Anything else I can help with?"
order_number(msg) = match(r"\d{4,}", msg).match      # the value is read by code, not by the model

transition(::AwaitingOrderId,  ::nl"The customer gives an order number", msg)          = ConfirmingRefund(order_number(msg))
transition(::ConfirmingRefund, ::nl"The customer confirms the refund", msg)            = Closed(:refunded)
transition(::ConfirmingRefund, ::nl"The customer declines or changes their mind", msg) = Closed(:kept)
transition(::Closed,           ::nl"The customer says goodbye or thanks", msg)          = Closed(:done)
transition(::Conversation,     ::nl"The customer asks for a human agent", msg)          = Closed(:handoff)
transition(s::Conversation,    ::nl"Something else", msg)                              = s

println(length(meanings(transition)[2]), " meanings in all; offered while confirming a refund:")
foreach(println, meanings(transition, Tuple{ConfirmingRefund, String})[2])
```

Each turn is one request whose state is what the assistant asked and what the
customer replied:

```@example jevprog
context(s, msg) = (assistant_asked = asked(s), customer_replied = msg)

let s = AwaitingOrderId()
    for msg in ("It's 48213.", "thanks, bye", "yes please", "no, that's all")
        s = nl_dispatch(transition, s, msg; state = context(s, msg))
        println(rpad(repr(msg), 18), "→ ", s)
    end
end
```

Measured on 30 live turns of a larger version of this machine (nine meanings):
offering every meaning in every state ended 7 turns in a `MethodError` — 3 of
them at confidence 1.00 — and got 22 transitions right; offering only the
meanings valid in the current state removed every `MethodError` and got 29
right. Values the model must not guess are read by code: the model decides that
the customer gave an order number, and `order_number` parses it.

## Guarding tool calls

An agent's LLM proposes tool calls; a guard decides whether the user asked for
this one. The guard below wraps any [`tool_loop`](@ref) dispatcher with the
same `(name, args)` signature and checks only the sensitive tools. Identifiers
are compared in code. The judgment sees the user's own words and the proposed
call — never retrieved text such as an email the agent just read, where an
injected instruction would live. It acts in three bands: allow at P ≥ 0.8, ask
the user to confirm in between, block at P ≤ 0.2. A failed Jev call throws
[`SystemOneError`](@ref) before the tool runs, so the guard fails closed.

```@example jevprog
const ASKED_FOR = noul("Does `user_request` ask for the action in `proposed_call` to be carried out on this target? " *
                       "The request may also ask for other things.";
    yes = "The user asks for this action on this target, possibly among other requests",
    no  = "The user asks only for other things, only asks a question, states a condition, " *
          "reports what someone else said, or names a different target")

struct Refused <: Exception
    msg::String
end
Base.showerror(io::IO, e::Refused) = print(io, e.msg)

function guarded(dispatcher, user_request; sensitive, account)
    function (name, args)
        name in sensitive || return dispatcher(name, args)
        get(args, "account_id", account) == account || throw(Refused("$name: not the signed-in account"))
        call = (tool = name, arguments = sort!(collect(args); by = first))   # sorted keys: stable request bytes
        p = ask((user_request = user_request, proposed_call = call), "asked" => ASKED_FOR)["asked"].noul
        p >= 0.8 ? dispatcher(name, args) :
        p > 0.2  ? throw(Refused("$name needs the user's explicit confirmation (p = $p)")) :
                   throw(Refused("$name was not requested by the user (p = $p)"))
    end
end

run_tool(name, args) = "$name: done"
for (request, name, arg) in [("Summarize my latest support emails.", "delete_account", "account_id" => "acc-7731"),
                             ("Please close my account permanently.", "delete_account", "account_id" => "acc-7731"),
                             ("Please close my account permanently.", "delete_account", "account_id" => "acc-1002"),
                             ("Refund 58213 and 58214, then summarize my inbox.", "issue_refund", "order_id" => "58214")]
    g = guarded(run_tool, request; sensitive = ("delete_account", "issue_refund"), account = "acc-7731")
    outcome = try g(name, Dict{String,Any}(arg)) catch e; e isa Refused || rethrow(); sprint(showerror, e) end
    println(rpad(request, 50), outcome)
end
```

In an agent, pass `guarded(run_tool, request; sensitive, account)` wherever
[`tool_loop`](@ref) takes a dispatcher: a `Refused` reaches the model as that
call's error output, the loop continues, and the model can ask the user to
confirm. In a replay scope, a guard request with no recording throws
[`ReplayMissError`](@ref), which escapes `tool_loop` instead of becoming a tool
error.

Measured limits of the wording shown, on 13 legitimate and 25 unrequested calls
of our own — in-sample, since the wording was tuned on these cases: all 13
legitimate calls were allowed, including a refund requested together with a
second action (P = 0.99). Of the 25 unrequested calls the judgment blocked 17,
sent 5 to confirmation and allowed 3; the identifier check refuses two of those
three, so one call got through — "Unsubscribe me from the newsletter." allowed
a subscription cancellation (P = 0.85). This is a confirmation filter that keeps
an agent from acting on what the user did not ask for — not a security boundary.

## Shortlisting tools for an LLM

An LLM given a large tool registry pays for every schema on every turn and has
more tools to confuse. One Choice ranks the whole registry — up to 255 tools,
the Choice limit — and the LLM receives only the top few, in rank order:

```@example jevprog
no_args = Dict("type" => "object", "properties" => Dict{String,Any}(), "required" => String[],
               "additionalProperties" => false)
const REGISTRY = [function_tool(name, description; parameters = no_args, strict = true) for (name, description) in [
    "get_order_status"      => "Look up the current status of an order",
    "track_shipment"        => "Get the carrier tracking events for a shipped parcel",
    "report_missing_parcel" => "Open a claim for a parcel marked delivered that never arrived",
    "start_return"          => "Start a return and generate a return shipping label",
    "issue_refund"          => "Refund an order to the original payment method",
    "reset_password"        => "Send a password reset link",
    "update_payment_method" => "Replace the card or payment method on file",
    "transfer_to_human"     => "Hand the conversation to a human support agent"]]

# One Choice over the whole registry; tools with zero probability drop out.
function ranking(tools, request)
    q = choice("Which tool should handle the user's request?", [t.name => t.description for t in tools])
    p = ask(request, "tool" => q)["tool"].probabilities
    sort([(prob, name) for (name, prob) in p if prob > 0]; rev = true)
end

# Keep the rank order: in two runs the LLM picked the first-listed tool.
shortlist(tools, ranked; k = 3) = [only(filter(t -> t.name == name, tools)) for (_, name) in first(ranked, k)]

request = "Tracking says delivered but there's nothing at my door."
ranked = ranking(REGISTRY, request)
tools = shortlist(REGISTRY, ranked)
println("Jev ranking: ", ranked)
println("shortlist:   ", [t.name for t in tools])
```

The LLM step is not executed here. Its `# =>` lines are the output of one live
run, pasted as comments; in that run the shortlist was `report_missing_parcel`,
`track_shipment`:

```julia
res = respond(request; tools, model = "gpt-6-luna", reasoning = Reasoning(effort = "none"),
              instructions = "Call exactly one tool that handles the request.")
only(function_calls(res))["name"]   # => "report_missing_parcel"
token_usage(res).prompt_tokens      # => 86
```

Measured with a 50-tool registry over 40 requests (gpt-5.6-luna, low reasoning
effort): the right tool was in Jev's top 5 for 40 of 40 requests, and handing
the LLM that top 5 instead of all 50 tools cut its input from 979 to 152 tokens
per request (−84%) while accuracy went from 0.950 to 0.975. Keep the shortlist
in rank order: in two runs with the shortlist in registry order, the LLM picked
the first-listed tool.

## See also

- [Typed Judgments with Jev (TypeSafe System One)](@ref system_one_guide) —
  [`ask`](@ref), the three primitives, and reading answers
- [Multiple Dispatch on Natural Language](@ref nl_dispatch_guide) — meanings in
  method signatures, [`nl_dispatch`](@ref) and [`@branch`](@ref)
- [Semantic Algorithms with Jev](@ref jev_algorithms_guide) — many judgments
  over a collection in one request
- TypeSafe: [Intent routing](https://docs.typesafe.ai/patterns/intent-routing),
  [Confidence](https://docs.typesafe.ai/confidence) and
  [Hierarchical classification](https://docs.typesafe.ai/cookbooks/hierarchical_classification)
