# [Test and Develop](@id jev_testing_guide)

A program built on Jev is ordinary Julia plus answers from a service. Most of it
can be tested with no service at all. The answers vary from one call to the
next, so tests record them once and replay them — Jev's answers and an LLM's
replies alike.

| I want to… | Use | You get | Section |
| :--- | :--- | :--- | :--- |
| see what a call will send, before it sends anything | [`meanings`](@ref), [`meaning_gaps`](@ref) | the options each call offers, in the order it sends them, and the combinations no method covers; no network | [Preview a call](@ref jev_testing_preview) |
| record real answers once and replay them on every run | [`with_recorded_answers`](@ref) | the same answers on every run, with no key and no network | [Recorded answers](@ref jev_testing_recorded) |
| test the methods, the resolution and the judgment | `@testset`s over `nl"…"()` instances, recorded answers, and Nouls over generated text | tests that run in CI without a key | [Tests](@ref jev_testing_tests) |
| test a whole Jev + LLM pipeline offline | `with_recorded_answers` around route, draft and check | the routing and the check asserted, with nothing sent | [Test a Jev + LLM pipeline offline](@ref jev_testing_pipeline) |
| keep an audit trail | `on_response` on [`nl_dispatch`](@ref), [`nl_classify`](@ref) and [`@branch`](@ref) | the request id, the model and the raw answer behind each decision | [Keep an audit trail](@ref jev_testing_audit) |

!!! details "Evidence"
    Measured on jev-1.13.0 (September 2026): 60% of pairs of identical requests
    came back with different numbers, by up to 0.11 in a probability, and a
    decision flipped only on inputs that were ambiguous to begin with.

Every example that calls Jev or an LLM runs when this manual is built, from
recorded answers ([the last section](@ref jev_testing_manual) says how). Most
share one program, a router whose ordinary argument is a chat message (a
`String`) or an `Email`:

```@example jevtest
using UniLM, Test

struct Email
    subject::String
    body::String
end

route(::nl"the customer wants a refund", msg)           = :refunds
route(::nl"the customer reports a bug in the app", msg) = :engineering
route(::nl"an automatic out-of-office reply", ::Email)  = :drop
route(::nl"anything else", msg)                         = :frontline
nothing # hide
```

## [Preview a call](@id jev_testing_preview)

**The job:** see what a call will ask the model — which options, in which order,
and whether every combination has a method — before anything is sent.
**Without Jev**, you would render an LLM prompt and read it. **With Jev**, the
options are the method table: [`meanings`](@ref) lists them in the order a call
sends them, and [`meaning_gaps`](@ref) lists the combinations no method covers,
with no network.

`meanings(route)` lists the meanings of every method, by slot:

```@example jevtest
meanings(route)[1]                     # slot 1: every method's meaning
```

`meanings(route, Tuple{…})` lists what one call offers, in the order it sends
them: only the meanings whose method accepts the argument types. A chat message
is never an automatic out-of-office reply, so a call with an `Email` offers all
four meanings and a call with a `String` offers three:

```@example jevtest
meanings(route, Tuple{String})[1]      # slot 1, for a chat message
```

Read each list as the model will: an option's sentence is the only text it gets
for that branch.

**Check coverage.** A gap is a combination of offered meanings that no method
covers, or covers ambiguously. With several slots the answers combine, and gaps
are easy to leave:

```@example jevtest
reply(::nl"a question", ::nl"a calm tone", msg)   = :answer
reply(::nl"a question", ::nl"an angry tone", msg) = :answer_and_apologise
reply(::nl"a complaint", ::nl"a calm tone", msg)  = :apologise

meaning_gaps(reply, Tuple{String})
```

[`nl_dispatch`](@ref) runs the same check before every request and, while a gap
exists, sends nothing and throws an `ArgumentError` that lists it. Close it with
the missing method, or with a backstop that is wild in every slot
(`reply(::Meaning, ::Meaning, msg)`): it is not offered, and it covers every
combination that has no method. A backstop does not settle an *ambiguous*
combination, which two methods match with neither more specific; the error names
both, and a method for their intersection settles it. Here the missing method
closes the gap:

```@example jevtest
reply(::nl"a complaint", ::nl"an angry tone", msg) = :escalate
meaning_gaps(reply, Tuple{String})
```

**Then look at one whole answer.** A `decide` policy is called with the whole
[`ChoiceAnswer`](@ref). One that prints the distribution and returns the argmax
shows the whole answer without changing the decision:

```@example jevtest
function argmax_shown(a)
    for (option, p) in sort(collect(a.probabilities); by = last, rev = true)
        println(rpad(p, 6), option)
    end
    a.choice
end

destination = nl_dispatch(route, "My package arrived crushed and the screen is cracked. I want my money back.";
                          decide = argmax_shown)
println("route returned ", repr(destination))
```

**Tune it**

- **Per argument type.** The options, and so the gaps, depend on the types of
  the ordinary arguments: preview every type the program calls with.
- **A runner-up with a real share** means two meanings overlap for inputs like
  the one you sent. Reword them before you tune a threshold.
- **Keyed tables** preview the same way: pass `texts` to `meanings` and
  `meaning_gaps` ([Keyed dispatch](@ref jev_testing_keyed)).

## [Recorded answers](@id jev_testing_recorded)

**The job:** make tests that call Jev or an LLM give the same answer on every
run, with no key. **Without recordings**, every run asks again and tests a new
sample. **With [`with_recorded_answers`](@ref)**, a real answer is recorded
once and replayed from a directory.

`with_recorded_answers(f, dir; mode)` runs `f()` with these exchanges passing
through a directory of recordings:

- **Recorded:** System One — [`ask`](@ref), and so [`nl_dispatch`](@ref),
  [`nl_classify`](@ref) and [`@branch`](@ref), and [`list_models`](@ref) — and
  the non-streaming LLM verbs [`chatrequest!`](@ref), [`respond`](@ref) and
  [`embeddingrequest!`](@ref), and so the tool loops [`tool_loop!`](@ref) and
  [`tool_loop`](@ref).
- **Not recorded:** a streamed call, images, files, audio, MCP, the Responses
  lifecycle calls ([`get_response`](@ref), [`cancel_response`](@ref), …) and
  every other verb. Inside a scope they behave exactly as outside it: they reach
  the network, with their key.

| `mode` | Behaviour |
| :--- | :--- |
| `:replay` (default) | Answers come from `dir`, which must exist. No recorded verb reaches the network or needs an API key. A request with no recording throws [`ReplayMissError`](@ref). |
| `:record` | Every request goes to the service; each HTTP 200 is written to `dir`, replacing an earlier recording of the same request. A failure is returned as usual and never recorded. |
| `:record_missing` | What `dir` holds is replayed; the rest goes to the service, and its 200s are recorded. |

- **One file per request.** A recording is `<dir>/<key>.json`. The file holds
  the request, the response body with its request id and the name of the header
  that carried it, and the time of recording, pretty-printed so a diff is
  readable. The API key and every other header are never written.
- **The key is the exact request:** the SHA-256 of the method, the path of the
  URL and the exact request body — except a credential the body carries, such as
  an MCP tool's `authorization`, which reads `"<redacted>"` in the key and the
  file, so a replay needs none. The host is not part of it — a base URL that
  differs only in its host files its requests under the same keys — and neither
  is the query, which can carry a credential. The key is never a canonical form,
  because the order of a Choice's options and of the state's keys moves the
  answer: a reordered request is a different request, with a recording of its
  own. So is a request with another model — for an unpinned System One request,
  another `TYPESAFE_DEFAULT_MODEL` ([Setup](@ref jev_setup)) — other
  instructions, or one changed byte in the state or the prompt.
- **A miss is loud.** In `:replay`, a request with no recording throws
  `ReplayMissError` out of the verb itself, and so out of `nl_dispatch` and the
  tool loops — never a silent call to the service, and never a call error
  ([`SystemOneCallError`](@ref), [`LLMCallError`](@ref), …) that a
  `r isa SystemOneSuccess || fallback()` path would absorb. The message names
  the directory, the start of the key, and what the request asked for: a System
  One request's questions, any other request's model.
- **Scopes nest.** An inner scope's service is the enclosing scope, not the
  network. A scope also covers the tasks started inside it (`Threads.@spawn`,
  `asyncmap`).
- **A replay pins one sample.** A recording is one draw from a service whose
  answers vary. Replaying it makes a test reproducible; it does not make that draw
  typical. Before you build a test on an answer near a tie, look at its recorded
  probabilities.

The demo pins the model first, since the model name is part of every request and
so of its key:

```@example jevtest
const JEV = "jev-1.13.0"
triage(msg) = nl_dispatch(route, msg; model = JEV)
nothing # hide
```

It records two calls into a temporary directory, replays them, and shows a
changed request missing. In this manual's build the recording scope's service is
whatever encloses it ([the last section](@ref jev_testing_manual)): the
committed recordings in a default build; in a recording build, the committed
recordings where they exist and the service for the rest; the service itself in
a live build.

```@example jevtest
recordings = mktempdir()     # in a package: a directory committed next to the tests
msgs = ["My package arrived crushed and the screen is cracked. I want my money back.",
        Email("Automatic reply: Out of office", "I am out of the office until Monday, with no access to email.")]

recorded = with_recorded_answers(recordings; mode = :record) do
    map(triage, msgs)
end
println(recorded)
foreach(println, readdir(recordings))     # one file per request, named by its key
```

Replaying needs no key. With `TYPESAFE_API_KEY` removed, no request could be
sent:

```@example jevtest
replayed = withenv("TYPESAFE_API_KEY" => nothing) do
    with_recorded_answers(() -> map(triage, msgs), recordings)
end
replayed == recorded
```

A request that differs in anything — here one trailing space, or the alias
instead of the pin — has no recording:

```@example jevtest
with_recorded_answers(recordings) do
    @testset showtiming=false "changed requests miss" begin
        @test_throws ReplayMissError triage(msgs[1] * " ")
        @test_throws ReplayMissError nl_dispatch(route, msgs[1])
    end
end
nothing # hide
```

**Tune it**

- **Pin the model.** `jev-latest` is an alias and moves when a new version ships
  ([Models](https://docs.typesafe.ai/models)). With the version pinned, moving
  the pin makes every recording miss until you record again, instead of testing
  the new version against the old one's answers. A request without `model =`
  names the client default — `jev-latest`, or whatever `TYPESAFE_DEFAULT_MODEL`
  holds at call time — so in code you record, pin with `model =`, or record with
  the same environment you replay with.
- **Record from a fresh session**, such as a run of the test suite. Re-evaluating
  a method whose sentence you edited defines a second method beside the first,
  which stays on offer until you restart Julia or delete it
  (`Base.delete_method(which(route, Tuple{nl"the old sentence", Any}))`), and
  re-evaluating any method can move its meaning to the end of the options: an
  edited session sends requests a fresh one never sends. Methods loaded from a
  package image are ordered by source file path, then line, not by `include`
  order, so keep one function's meanings in one file. A keyed table is sent in
  table order, whatever the definition order.
- **Prune by copying.** Recordings are never pruned for you: a request no test
  makes any more leaves its file behind. Run the tests once in a `:record` scope
  over a new directory, nested inside a `:replay` scope over the old one: the new
  directory receives exactly the recordings the tests use, copied without
  touching the network.

## [Tests: the methods, the resolution, the judgment](@id jev_testing_tests)

**The job:** check in CI, without a key, that the program still decides the way
you need. **Without Jev**, a classifier is tested against labelled examples, one
call each. **With Jev**, a program has three layers, and each is tested on its
own:

- **The methods** are ordinary Julia. A method with `nl"…"` in its signature runs
  like any other when it is called with a [`Meaning`](@ref) instance.
- **The resolution** is what goes on the wire and which method a given answer
  selects: the options the method table offers, the state the arguments become,
  the policy that turns an answer into a meaning.
- **The judgment** is whether the model decides your inputs the way you need.

Only the judgment needs the service.

### The methods

Call them with `nl"…"()` instances, or with `Meaning(text)` for a sentence held
in a variable. No service, no key:

```@example jevtest
@testset showtiming=false "route: the methods" begin
    @test route(nl"the customer wants a refund"(), "I want my money back.") === :refunds
    @test route(Meaning("anything else"), Email("Invoice 1042", "Please find it attached.")) === :frontline
    @test route(nl"an automatic out-of-office reply"(), msgs[2]) === :drop
    @test_throws MethodError route(nl"an automatic out-of-office reply"(), "Back on Monday.")
end
nothing # hide
```

The last line is why a `String` is never offered that meaning: no method could
take it.

### Coverage

```@example jevtest
@testset showtiming=false "every offered combination has a method" begin
    @test isempty(meaning_gaps(reply, Tuple{String}))
    for T in (String, Email)
        @test isempty(meaning_gaps(route, Tuple{T}))
    end
end
nothing # hide
```

List every argument type the program calls with: the options, and so the gaps,
are per type. The test moves the refusal `nl_dispatch` would raise in production
into CI.

### [Keyed dispatch: the table and the methods agree](@id jev_testing_keyed)

With `texts`, the methods dispatch on short keys and a table holds the sentence
the model reads for each key ([Dispatch on Meaning](@ref nl_dispatch_keyed)).
The table and the methods live apart, so test that they agree:
`meanings(f; texts)` makes the check `nl_dispatch` makes before its request and
throws when a key has no method, `meanings(f, argtypes; texts)` pins what a call
offers, in table order, and `meaning_gaps(f, argtypes; texts)` lists the
combinations no method covers. Here the support desk's teams from [Jev with
LLMs](@ref jev_llm_route) each go to a queue:

```@example jevdesk
using UniLM, Test

const TEAM = (billing   = "payments, charges, invoices or refunds",
              technical = "the app or the website does not work as expected",
              shipping  = "a parcel that is late, lost or arrived damaged",
              other     = "anything else")

queue(::Val{:billing}, message)   = "Payments"
queue(::Val{:technical}, message) = "App support"
queue(::Val{:shipping}, message)  = "Warehouse"
queue(::Val{:other}, message)     = "Front desk"

@testset showtiming=false "queue: the table and the methods agree" begin
    @test meanings(queue; texts = TEAM) == Dict(1 => collect(TEAM))       # throws if a key has no method
    @test meanings(queue, Tuple{String}; texts = TEAM) == Dict(1 => collect(TEAM))
    @test isempty(meaning_gaps(queue, Tuple{String}; texts = TEAM))
    with_returns = merge(TEAM, (returns = "the customer wants to send an item back",))
    @test_throws "no method of queue takes :returns" meanings(queue; texts = with_returns)
end
nothing # hide
```

The last test is the drift a separate table invites: a key added to the table
and not to the methods. The error names the method to define,
`queue(::Val{:returns}, _)`, and `nl_dispatch` raises it before any request. A
method whose key is not in the table is no error: it is never offered.

### The resolution, on recorded answers

Replay the calls the program makes and assert on the method they select. The
test needs no key and sends nothing, and it covers everything that shapes the
request — the meanings and their order, the argument names that key the state,
the pin: change any of them and the replay misses, loudly, until the answer is
recorded again.

In a package the recordings live next to the tests, and a switch of your own
decides when to record:

```julia
# test/runtests.jl
const RECORDINGS = joinpath(@__DIR__, "recorded_answers")
const MODE = get(ENV, "RECORD_ANSWERS", "") == "1" ? :record_missing : :replay

with_recorded_answers(RECORDINGS; mode = MODE) do
    @testset "route, on recorded answers" begin
        @test triage("My package arrived crushed and the screen is cracked. I want my money back.") === :refunds
    end
end
```

Here the directory recorded above plays that part:

```@example jevtest
with_recorded_answers(recordings; mode = :replay) do
    @testset showtiming=false "route, on recorded answers" begin
        @test triage(msgs[1]) === :refunds
        @test triage(msgs[2]) === :drop
    end
end
nothing # hide
```

### Edge cases, from a local server

A recording holds an answer the model gave. To test what the program does with an
answer the model rarely gives — a near-tie, a failed call, an option it was never
offered — write the answer yourself, and point `TYPESAFE_BASE_URL`
([Setup](@ref jev_setup)) at a local server that returns it. This one answers
with whatever `canned[]` holds and keeps every request body it receives, so a
test can check the wire as well:

```julia
using UniLM, HTTP, JSON, Test

canned = Ref{Any}()                  # (status, body) the server answers next
sent = String[]                      # every request body it received
server = HTTP.serve!("127.0.0.1", 8123; verbose = false) do req
    push!(sent, String(copy(req.body)))
    status, body = canned[]
    HTTP.Response(status, ["Content-Type" => "application/json"], JSON.json(body))
end

const OPTIONS = meanings(route, Tuple{String})[1]     # what a String message is offered

# A 200 in the service's shape: route's one question is "meaning_1" (its slot is unnamed).
# `option` gets probability p, the others share the rest, and confidence is (n·p − 1)/(n − 1).
function choosing(option, p)
    n = length(OPTIONS)
    probabilities = Dict(o => o == option ? p : (1 - p) / (n - 1) for o in OPTIONS)
    (200, (model = "jev-1.13.0", usage = (input_tokens = 96, output_tokens = 3),
           answers = (meaning_1 = (type = "choice", choice = option, probabilities,
                                   confidence = clamp((n * p - 1) / (n - 1), 0, 1)),)))
end

withenv("TYPESAFE_BASE_URL" => "http://127.0.0.1:8123", "TYPESAFE_API_KEY" => "test") do
    @testset "route: resolution edge cases" begin
        canned[] = choosing("the customer wants a refund", 0.97)
        @test nl_dispatch(route, "I want my money back.") === :refunds
        wire = JSON.parse(last(sent))                     # the request, in wire order
        @test wire["state"] == Dict("msg" => "I want my money back.")
        @test collect(keys(wire["questions"]["meaning_1"]["criteria"])) == OPTIONS

        canned[] = choosing("the customer wants a refund", 0.40)        # confidence 0.1
        @test nl_dispatch(route, "Hm."; min_confidence = 0.7, fallback = m -> :human) === :human
        @test_throws LowConfidenceError nl_dispatch(route, "Hm."; min_confidence = 0.7)

        canned[] = (503, (detail = "Service overloaded",))
        @test_throws SystemOneError nl_dispatch(route, "Hm."; config = RequestConfig(max_attempts = 1))

        canned[] = choosing("an automatic out-of-office reply", 0.97)   # never offered for a String
        @test_throws ArgumentError nl_dispatch(route, "Hm.")
    end
end
close(server)
```

The response shape is the whole contract: a `model`, an `answers` object keyed by
the question names the client sent — `meaning_1` here, the slot's argument name
when it has one — each a Choice with `choice`, `confidence` and `probabilities`,
and a `usage` object. `RequestConfig(max_attempts = 1)` makes the retryable 503
fail at once instead of backing off. The last case is one a recording cannot
produce: an answer outside the offered options is refused with an
`ArgumentError`, and no method runs. A gap in the method table needs no server
at all, since `nl_dispatch` refuses it before sending. Run these tests outside
`with_recorded_answers`: in a replay scope a request with no recording throws
before it reaches the server, and a recording scope would file the canned
answers as real ones.

### [The judgment: semantic assertions over LLM output](@id jev_testing_assertions)

A property of generated text — the reply apologises, the reply promises no
refund — is a judgment, and Jev can make it: one Noul per property, all in one
request. The reply comes from an LLM call, which is recorded like Jev's answers,
so every run asserts on the same reply:

```@example jevtest
const RULES = "You are a support agent. At most 3 sentences. Apologise when something went wrong. " *
              "Never promise a refund; say a colleague will review eligibility."
customer = "My order #58213 arrived two weeks late and the box was soaked. I want my money back."
draft = output_text(respond(customer; instructions = RULES, model = "gpt-5.4-mini"))
println(draft)

verdict = ask((customer_message = customer, reply = draft),
    "apologises"        => noul("Does `reply` apologise to the customer?"),
    "offers_refund"     => noul("Does `reply` promise or offer a refund?";
                                yes = "It commits to or offers a refund, even conditionally",
                                no  = "It only says eligibility will be reviewed, or makes no refund commitment"),
    "asks_order_number" => noul("Does `reply` ask the customer for their order number?");
    model = JEV)
for property in ("apologises", "offers_refund", "asks_order_number")
    println(rpad(property, 18), verdict[property].noul)
end

band(p) = p >= 0.8 ? :yes : p <= 0.2 ? :no : :inconclusive

@testset showtiming=false "the reply follows the refund policy" begin
    @test band(verdict["apologises"].noul) === :yes
    @test band(verdict["offers_refund"].noul) === :no
    @test band(verdict["asks_order_number"].noul) === :no      # the customer already gave it
end
nothing # hide
```

A verdict falls in one of three bands — at or above 0.8 a yes, at or below 0.2 a
no, in between inconclusive — instead of one cut at 0.5, where an answer near the
cut can land on either side from one call to the next. An inconclusive verdict
fails its assertion as a wrong one does, but says so in the failure
(`Evaluated: inconclusive === yes`): the remedy is a sharper question or a look
at the reply, not a re-run.

Here the refund verdict, 0.17, is a no near the edge of its band: the reply
names a refund only to say that it cannot promise one. A replay keeps that
verdict; a fresh run could land it in the middle band.

In a test suite, run the block inside the suite's `with_recorded_answers` scope,
as `test/runtests.jl` above does: the reply and the verdict are both pinned.
Assert on the reply you recorded, not on a fresh one: a new reply is a new
input.

!!! details "Evidence"
    Measured on jev-1.13.0 (September 2026), four such assertions over 24
    replies of our own: all 95 verdicts we could label agreed with our labels,
    and none changed across three identical runs. Regenerating the replies
    changed 12 of 96 verdicts.

### Paraphrase tests

A router can be checked without labels: rewrites of one input that keep its
meaning must route the same way. Write paraphrases in several styles — formal,
terse, misspelt, another language, rambling — and assert that each lands where
the original does:

```@example jevtest
seed = "My package arrived crushed and the screen is cracked. I want my money back."
paraphrases = [
    "The parcel I received was crushed in transit and its screen is cracked. I would like a full refund.",
    "crushed box, cracked screen. refund pls",
    "my pakage came crushd and the scren is craked, i want my mony back",
    "El paquete llegó aplastado y la pantalla está rota. Quiero que me devuelvan el dinero.",
    "So the delivery finally showed up today, the box was completely flattened and of course the " *
    "screen inside is cracked. Honestly I just want my money back at this point."]

expected = triage(seed)
@testset showtiming=false "paraphrases route like the original" begin
    @testset "$p" for p in paraphrases
        @test triage(p) === expected
    end
end
nothing # hide
```

It is a cheap smoke test that needs no labels, not proof: a router that sends an
input and all its paraphrases to the same wrong place passes.

!!! details "Evidence"
    Measured on jev-1.13.0 (September 2026) with three versions of one five-way
    router — options with descriptions, bare option names, deliberately vague
    descriptions — on 15 inputs with six paraphrases each: the share of
    paraphrases routed like their original was 1.000, 0.989 and 0.967, in the
    same order as labelled accuracy (1.000, 0.990, 0.971). The paraphrases that
    moved all came from a single input (1 of 15).

## [Test a Jev + LLM pipeline offline](@id jev_testing_pipeline)

**The job:** test the whole path a message takes — route, draft, check — in CI,
with no key and nothing sent. **Without recordings**, the test calls Jev and the
LLM on every run, pays for both, and asserts on a new draft each time. **With
[`with_recorded_answers`](@ref)**, the LLM's draft is recorded like Jev's
answers, and the whole pipeline replays.

The desk from [Jev with LLMs](@ref jev_llm_desk), cut to three steps: route with
[`nl_classify`](@ref) over `TEAM` (above), draft with [`respond`](@ref), and
check the draft against the shop's policy with one Jev question:

```@example jevdesk
const MESSAGES = [
    "I was charged twice for order #4471. Please refund the duplicate payment.",
    "The app crashes every time I open my order history. iPhone 15, latest version.",
    "My parcel was due last Monday and the tracking hasn't moved in six days.",
    "Do you ship to Norway, and how much does delivery cost?",
    "The phone I received has a cracked screen and the box was crushed. I want my money back.",
    "Your update deleted my saved addresses and now I can't check out. Third time this month!",
]

const POLICY = """
Refunds: we refund the full price of an item returned within 30 days of delivery, in its original packaging.
Damaged items: report them within 7 days with a photo; we send a replacement or refund the item, as you prefer.
Shipping: orders leave our warehouse within 2 business days; standard delivery takes 3 to 7 business days.
We refund shipping costs only when the item arrived damaged."""

# Route: the team's key, or :other when Jev is unsure.
team_for(message; kwargs...) = nl_classify(message, TEAM; min_confidence = 0.6, fallback = _ -> :other, kwargs...)

# Draft: the LLM writes with the team's instructions and the shop's policy.
instructions(team) = "You write replies for the $team team of a small online shop that sells phones and " *
                     "accessories. Follow the shop's policy:\n$POLICY\nAnswer in at most three sentences."
draft(message, team) = output_text(respond(message; instructions = instructions(team), model = "gpt-5.4-mini"))

# Check: does the policy back what the draft says?
const RELATION = choice("How does `policy` relate to `reply`?", (
    supports     = "The policy states what the reply says, or directly implies that it is true",
    contradicts  = "The policy states the opposite of what the reply says, or implies it is false",
    says_nothing = "The policy does not address what the reply asserts, either way"))

function grounded(reply)
    a = ask((policy = POLICY, reply = reply), "relation" => RELATION)["relation"]
    a.confidence >= 0.8 ? Symbol(a.choice) : :unsure
end

# The path a message takes: its team, and the check's verdict on the draft.
function desk(message)
    team = team_for(message)
    team === :other && return (team, :person)
    (team, grounded(draft(message, team)))
end
nothing # hide
```

The test records once and replays on every run after that. Here the recording
goes into a temporary directory, and the replay runs with both keys removed:

```@example jevdesk
desk_answers = mktempdir()     # in a package: a directory committed next to the tests

# Once, with TYPESAFE_API_KEY and OPENAI_API_KEY set: record every answer the test needs.
with_recorded_answers(() -> map(desk, MESSAGES[[1, 5]]), desk_answers; mode = :record)

# Every run after that: no key, and nothing is sent.
withenv("TYPESAFE_API_KEY" => nothing, "OPENAI_API_KEY" => nothing) do
    with_recorded_answers(desk_answers) do
        @testset showtiming=false "the desk, offline" begin
            @test desk(MESSAGES[5]) == (:shipping, :supports)       # drafted, and the policy backs the draft
            @test desk(MESSAGES[1]) == (:billing, :says_nothing)    # drafted; the policy is silent, so a person reads it
        end
    end
end
nothing # hide
```

Both paths replay: the routes and the checks from Jev, the drafts from the LLM.
The draft is part of the check's request, so the two stay in step: change
`instructions` and the draft's request misses, loudly, until it is recorded
again — and a new draft is a new input for the check, so read it before you
trust the assertion again.

**Tune it**

- **Record with both keys, replay with none.** A recording scope sends each
  request to its own service: Jev with `TYPESAFE_API_KEY`, the LLM with
  `OPENAI_API_KEY`.
- **Keep the tested path non-streaming.** A streamed call is not recorded: it
  reaches the network even inside a replay scope.
- **Assert on the path, not on the wording.** The team and the verdict are what
  the program acts on; the draft's words are one sample.

## [Keep an audit trail](@id jev_testing_audit)

**The job:** keep, with every decision the program acts on, what a support
request or an auditor will ask for. **Without `on_response`**, keeping the
answer meant making the decision with [`ask`](@ref) yourself. **With it**,
[`nl_dispatch`](@ref), [`nl_classify`](@ref) and [`@branch`](@ref) hand you the
whole [`SystemOneSuccess`](@ref) before they decide. Keep three values from its
`response`:

- `request_id` — the `x-typesafe-request-id` header, the id to quote in a
  support report.
- `model` — the versioned id that answered, which the alias you named does not
  tell you.
- `raw` — the JSON body the service sent; the typed fields are read from it.

```@example jevdesk
audit = SystemOneSuccess[]
team = team_for(MESSAGES[1]; on_response = r -> push!(audit, r))

r = only(audit)
println(team, "  ", r.response.model, "  ", r.response.request_id)
println(sort(collect(keys(r.response.raw))))
```

`on_response` runs once, when the request has succeeded and before the decision
policy, so it also sees the answers the policy then declines; `decide`, called
with the `ChoiceAnswer` alone, cannot see the request id or the model. It is not
called for a failed call: the thrown [`SystemOneError`](@ref) carries that
result. `nl_dispatch(f, args...; on_response)` takes it the same way, and
`@branch` as a keyword: `@branch message on_response = keep begin … end`.

**Tune it**

- **Log before you act.** An exception from `on_response` propagates, and no
  method, branch body or fallback runs: a decision the log could not record is
  not made.

## [How this manual runs its examples](@id jev_testing_manual)

`docs/make.jl` runs the build inside one [`with_recorded_answers`](@ref) scope
over `docs/recorded_answers`, and picks its mode by flag, never by which API
keys are set. The examples that call a service replay from it: System One
([`ask`](@ref), [`list_models`](@ref)) and the non-streaming LLM verbs
([`chatrequest!`](@ref), [`respond`](@ref), [`embeddingrequest!`](@ref), the
tool loops).

| Flag | Scope | What the examples do |
| :--- | :--- | :--- |
| neither | `:replay` | Every provider key is hidden, even when the shell exports it: the build spends nothing and renders what the keyless CI build renders. An example with no recording fails the build. |
| `UNILM_DOCS_RECORD=1` | `:record_missing` | Recorded requests replay; the rest go live, with the keys of their providers, and are recorded. |
| `UNILM_DOCS_LIVE=1` | none | No replay: every example calls its service live. |

Each flag is `1`, `0` or unset, and at most one is `1`; anything else stops the
build before anything runs. What the scope does not record — a streamed call,
images, files, MCP — goes to its service in a recording or a live build. Replay
and record ignore `TYPESAFE_DEFAULT_MODEL`: an unpinned request names the
default model, so an exported default would change every recording's key. After
a recording run, commit the new files in `docs/recorded_answers`.

The output under an example is the build's own: a recorded answer in a default
build, a fresh one in a live build, and in a recording build the committed
recording where one exists and a fresh answer otherwise. The numbers in the
Evidence boxes were measured separately.

## See also

- [Start Here: Jev in Five Minutes](@ref jev_start) — which page answers which question
- [Route and Decide](@ref system_one_guide) — `ask`, the three primitives, and
  reading answers
- [Dispatch on Meaning](@ref nl_dispatch_guide) — meanings in method signatures
  and keyed tables, `nl_dispatch`, `decide` and `@branch`
- [Jev with LLMs](@ref jev_llm_guide) — the support desk tested above, step by
  step
- [TypeSafe System One API (Jev)](@ref system_one_api) — every type and verb,
  including [`with_recorded_answers`](@ref) and [`ReplayMissError`](@ref)
