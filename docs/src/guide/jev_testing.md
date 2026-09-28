# [Developing and Testing with Jev](@id jev_testing_guide)

A program built on Jev has three layers, and they are tested differently:

- **The methods** are ordinary Julia. A method with `nl"…"` in its signature runs
  like any other when it is called with a [`Meaning`](@ref) instance.
- **The resolution** is what goes on the wire and which method a given answer
  selects: the options the method table offers, the state the arguments become,
  the policy that turns an answer into a meaning.
- **The judgment** is whether the model decides your inputs the way you need.

Only the judgment needs the service, and the service does not answer a repeated
request identically. Measured on jev-1.13.0 (September 2026): 60% of pairs of
identical requests came back with different numbers, by up to 0.11 in a
probability, and a decision flipped only on inputs that were ambiguous to begin
with. A test that asks again tests a new sample on every run, so tests pin real
answers instead: record once, replay on every run.

Every example that calls Jev runs when this manual is built (the last section
says how). They share one program, a router whose ordinary argument is a chat
message (a `String`) or an `Email`:

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

## The development loop

**Preview the options.** [`meanings`](@ref)`(route)` lists the meanings of every
method, by slot:

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
(`reply(::Meaning, ::Meaning, msg)`), which is not offered and covers every
combination:

```@example jevtest
reply(::nl"a complaint", ::nl"an angry tone", msg) = :escalate
meaning_gaps(reply, Tuple{String})
```

**Call the service while you design.** A `decide` policy is called with the
whole [`ChoiceAnswer`](@ref). One that prints the distribution and returns the
argmax shows the whole answer without changing the decision:

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

A runner-up with a real share means two meanings overlap for inputs like this
one. Reword them before you tune a threshold.

**Pin the model.** `jev-latest` is an alias and moves when a new version ships
([Models](https://docs.typesafe.ai/models)). Once the answers matter, name the
version:

```@example jevtest
const JEV = "jev-1.13.0"
triage(msg) = nl_dispatch(route, msg; model = JEV)
nothing # hide
```

The model name is part of the request, and so of the key a recorded answer is
filed under: when you move the pin, every recorded answer misses until you
record again, instead of testing the new version against the old one's answers.

**Record from a fresh session.** Re-evaluating a method whose sentence you edited
defines a second method beside the first, which stays on offer until you restart
Julia or delete it
(`Base.delete_method(which(route, Tuple{nl"the old sentence", Any}))`). Options
also follow definition order, so re-evaluating a method — even one whose sentence
did not change — can move its meaning to the end of the list. A session with such
edits sends requests a fresh one never sends; record in a fresh session, such as
a run of the test suite.

## Recorded answers

[`with_recorded_answers`](@ref)`(f, dir; mode)` runs `f()` with every System One
exchange inside it — [`ask`](@ref), and so `nl_dispatch` and [`@branch`](@ref),
and [`list_models`](@ref) — passing through a directory of recordings:

| `mode` | Behaviour |
| :--- | :--- |
| `:replay` (default) | Answers come from `dir`, which must exist. Nothing reaches the network and no API key is needed. A request with no recording throws [`ReplayMissError`](@ref). |
| `:record` | Every request goes to the service; each HTTP 200 is written to `dir`, replacing an earlier recording of the same request. A failure is returned as usual and never recorded. |
| `:record_missing` | What `dir` holds is replayed; the rest goes to the service, and its 200s are recorded. |

- **One file per request.** A recording is `<dir>/<key>.json`, where `key` is the
  SHA-256 of the request line and the exact request body. The file holds the
  request, the response body with its request id, and the time of recording,
  pretty-printed so a diff is readable. The API key and the headers are never
  written.
- **The key is the exact request.** It is never a canonical form, because the
  order of a Choice's options and of the state's keys moves the answer: a
  reordered request is a different request, with a recording of its own. So is a
  request with another model, other instructions, or one changed byte in the
  state.
- **A miss is loud.** In `:replay`, a request with no recording throws
  `ReplayMissError` out of `ask` itself — never a silent call to the service,
  and never a [`SystemOneCallError`](@ref) that a
  `r isa SystemOneSuccess || fallback()` path would absorb. The message names the
  directory, the start of the key and the questions the request asked.
- **Scopes nest.** An inner scope's service is the enclosing scope, not the
  network. A scope also covers the tasks started inside it (`Threads.@spawn`,
  `asyncmap`).
- **A replay pins one sample.** A recording is one draw from a service whose
  answers vary. Replaying it makes a test reproducible; it does not make that draw
  typical. Before you build a test on an answer near a tie, look at its recorded
  probabilities.

Recordings are never pruned for you: a request no test makes any more leaves its
file behind. To prune and re-record at once, delete the directory and run once
with `TYPESAFE_API_KEY` set and `mode = :record_missing`.

The demo records two calls into a temporary directory, replays them, and shows a
changed request missing. In this manual's build the recording scope's service is
whatever encloses it (last section): the committed recordings in a keyless
build, the service itself in a build with a key.

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

## Tests

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

### Resolution, on recorded answers

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
offered — write the answer yourself, and point `TYPESAFE_BASE_URL` at a local
server that returns it. This one answers with whatever `canned[]` holds and keeps
every request body it receives, so a test can check the wire as well:

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

### Semantic assertions over LLM output

A property of generated text — the reply apologises, the reply promises no
refund — is a judgment, and Jev can make it: one Noul per property, all in one
request. Generate the text once and keep it as a fixture; this LLM call ran once,
and is not part of the test:

```julia
const POLICY = "You are a support agent. At most 3 sentences. Apologise when something went wrong. " *
               "Never promise a refund; say a colleague will review eligibility."
customer = "My order #58213 arrived two weeks late and the box was soaked. I want my money back."
draft = output_text(respond(customer; instructions = POLICY, model = "gpt-6-luna",
                            reasoning = Reasoning(effort = "none")))
```

The test asserts on the reply that call returned, kept as a string:

```@example jevtest
customer = "My order #58213 arrived two weeks late and the box was soaked. I want my money back."
draft = "I’m sorry your order #58213 arrived two weeks late and in a soaked box. I’ll pass this along " *
        "so a colleague can review your refund eligibility; please share any photos of the damaged " *
        "packaging or items, if available."

verdict = ask((customer_message = customer, reply = draft),
    "apologises"        => noul("Does `reply` apologise to the customer?"),
    "offers_refund"     => noul("Does `reply` promise or offer a refund?";
                                yes = "It commits to or offers a refund, even conditionally",
                                no  = "It only says eligibility will be reviewed, or makes no refund commitment"),
    "asks_order_number" => noul("Does `reply` ask the customer for their order number?");
    model = JEV)

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

Measured on jev-1.13.0 (September 2026), four such assertions over 24 replies of
our own: all 95 verdicts we could label agreed with our labels, and none changed
across three identical runs. Regenerating the replies changed 12 of 96 verdicts —
a new reply is a new input. Assert on the reply you recorded, not on a fresh one,
and run the check inside `with_recorded_answers` so that its verdict is pinned
too.

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

Measured on jev-1.13.0 (September 2026) with three versions of one five-way
router — options with descriptions, bare option names, deliberately vague
descriptions — on 15 inputs with six paraphrases each: the share of paraphrases
routed like their original was 1.000, 0.989 and 0.967, in the same order as
labelled accuracy (1.000, 0.990, 0.971). Every paraphrase that moved came from
one of the 15 inputs, and a router that sends an input and all its paraphrases
to the same wrong place passes. It is a cheap smoke test that needs no labels,
not proof.

## Logging for audits

Keep three values with every decision the program acts on:

- `raw` — the JSON body the service sent; the typed fields are read from it.
- `request_id` — the `x-typesafe-request-id` header, the id to quote in a support
  report.
- `response.model` — the versioned id that answered, which the alias you named
  does not tell you.

[`ask`](@ref) returns all three on its result:

```@example jevtest
r = ask(msgs[1], "route" => choice("Which of these describes the message?", meanings(route, Tuple{String})[1]);
        model = JEV)
entry = (decision = r["route"].choice, raw = r.response.raw,
         request_id = r.response.request_id, model = r.response.model)
println(entry.decision, "  (", entry.model, ")")
```

`nl_dispatch` and `@branch` return the selected method's value, not the
response. Their `decide` policy is called with every `ChoiceAnswer`, whose `raw`
is the answer as sent, so it can log the decision —
`decide = a -> (push!(decisions, a.raw); a.choice)` logs and keeps the argmax —
but the request id and the model do not reach it. When those must be kept too,
make the decision with `ask`.

## How this manual runs its examples

`docs/make.jl` reads the environment to decide whether the whole build runs
inside one `with_recorded_answers` scope over `docs/recorded_answers`, and in
which mode:

| Environment | Scope |
| :--- | :--- |
| no `TYPESAFE_API_KEY` | `:replay`: the committed recordings answer, and an example with no recording fails the build |
| `TYPESAFE_API_KEY` and `UNILM_DOCS_RECORD=1` | `:record_missing`: recorded requests are replayed; the rest go to the service and are recorded |
| `TYPESAFE_API_KEY` alone | none: every example calls the service |

`UNILM_DOCS_RECORD` set to anything but `1` or `0`, or set to `1` without a key,
stops the build before anything runs. After a recording run, commit the new files
in `docs/recorded_answers`; to prune the ones no example uses any more, delete
the directory and record again. The output under an example is the build's own —
a recorded answer in a keyless build, a fresh one in a build with a key — and the
numbers in the prose were measured separately.

## See also

- [Typed Judgments with Jev (TypeSafe System One)](@ref system_one_guide) — `ask`,
  the three primitives, and reading answers
- [Multiple Dispatch on Natural Language](@ref nl_dispatch_guide) — meanings in
  method signatures, `nl_dispatch`, `decide` and `@branch`
- [TypeSafe System One API (Jev)](@ref system_one_api) — every type and verb,
  including [`with_recorded_answers`](@ref) and [`ReplayMissError`](@ref)
