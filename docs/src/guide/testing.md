# [Testing and recordings](@id testing_guide)

```@setup testing
using UniLM
```

A test that calls a model live needs a key and the network, and a model's reply varies from call to
call, so the test cannot be reproduced. [`with_recorded_answers`](@ref) records a real exchange once and
replays it from a directory: the test then runs with no key, sends nothing, and gets the same answer on
every run.

```@example testing
service, model = OPENAIServiceEndpoint, "gpt-5.4-mini"
key = "OPENAI_API_KEY"
nothing # hide
```

The examples use only the names bound above. `service` and `model` are the provider and model they run
on, and `key` the variable the provider's key is read from: bind your own there. The table
[at the end](@ref testing_providers) links each provider's start page.

## [Record once, replay without a key](@id testing_record)

`with_recorded_answers(f, dir; mode)` runs `f()` with the requests inside it passing through the
recordings in `dir`, and returns `f()`'s value. Here a call is recorded into a new directory:

```@example testing
calculate() = chatrequest!(systemprompt="You are a calculator. Respond only with the number.",
                           userprompt="What is 42 * 17?", service=service, model=model, temperature=0.0)

recordings = mktempdir()      # in a package: a directory committed next to the tests
recorded = with_recorded_answers(calculate, recordings; mode = :record)
text(recorded), readdir(recordings)
```

Scopes nest: an inner scope's service is the enclosing scope. This manual's own build runs inside a scope that
replays its committed recordings, so the answer above is the one recorded for this request
([How this manual runs its examples](@ref jev_testing_manual)); in your code, a `:record` scope sends it.

The default mode replays. With the key removed, nothing could be sent, and the answer is the recorded one:

```@example testing
replayed = withenv(key => nothing) do
    with_recorded_answers(calculate, recordings)
end
text(replayed) == text(recorded)
```

A request that differs in anything — here one number of the prompt — has no recording, and the verb throws
[`ReplayMissError`](@ref):

```@example testing
miss = try
    with_recorded_answers(recordings) do
        chatrequest!(systemprompt="You are a calculator. Respond only with the number.",
                     userprompt="What is 42 * 18?", service=service, model=model, temperature=0.0)
    end
    nothing
catch err
    err
end
miss isa ReplayMissError || error("expected the changed request to miss")
miss.path, miss.reason
```

## [What is recorded](@id testing_scope)

- **Recorded:** [`chatrequest!`](@ref) (with its keyword form and [`tool_loop!`](@ref)) and
  [`respond`](@ref) (with its convenience forms and [`tool_loop`](@ref)), streamed or not;
  [`embeddingrequest!`](@ref); the Responses lifecycle calls ([`get_response`](@ref),
  [`delete_response`](@ref), [`list_input_items`](@ref), [`cancel_response`](@ref),
  [`compact_response`](@ref), [`count_input_tokens`](@ref)); [`fim_complete`](@ref) and
  [`prefix_complete`](@ref); and the decision calls, [`ask`](@ref) and the verbs built on it, with the
  one model listing [Test and Develop](@ref jev_testing_guide) names.
- **Not recorded:** image generation, audio, files, batch, moderation, the other model listings, model
  downloads, realtime, MCP and every other verb. Inside a scope they behave exactly as outside it: they
  reach the network, with their key.

A replayed answer goes through the same decoding as a live one and has the same type and fields: a `Chat`
gets the reply appended and its cost accumulated, and `Embeddings` get their vectors. A replayed stream is
read through the same parsers and callbacks as a live one.

| `mode` | Behaviour |
| :--- | :--- |
| `:replay` (default) | Answers come from `dir`, which must exist. No recorded verb reaches the network or needs an API key. A request with no recording throws [`ReplayMissError`](@ref). |
| `:record` | Every request goes to the service; each HTTP 200 is written to `dir`, replacing an earlier recording of the same request. A failure is returned as usual and never recorded. |
| `:record_missing` | What `dir` holds is replayed; the rest goes to the service, and its 200s are recorded. |

A stream is recorded only when its reply is whole — its final event arrived, and it was read with no
stop. Both recording modes create `dir` if needed and check that it takes a file before `f` runs; an
answer whose recording still cannot be written throws [`RecordingWriteError`](@ref), since the service has
billed it.

## [The key is the exact request](@id testing_key)

A recording is `<dir>/<key>.json`, and the key is the SHA-256 of the method, the path of the URL and the
exact request body: a request that differs in one byte — another prompt, another model, another option,
`stream=true` — is another request, with a recording of its own; a decision request is keyed as
[Recorded answers](@ref jev_testing_recorded) says. The host is not part of the key, and
neither is the query, which can carry a credential; [`list_input_items`](@ref) adds the page its query
names instead. A credential the body carries, such as an MCP tool's `authorization`, and the bytes of media sent
as a `data:` URL are replaced before anything is keyed or written: the key and the file hold
`"<redacted>"` and the media's digest. The file holds the request, the response body (a stream's lines),
the request id, the headers a replay gives back and the time of recording, pretty-printed so a diff is
readable; never the API key or any other header.

So pin what the request names: a `Chat` without `model` names the provider's default model, and a new
default makes every recording miss.

## [A miss is loud](@id testing_miss)

In `:replay`, a request with no recording throws `ReplayMissError` out of the verb itself, and so out of
the tool loops — never a call error that a `result isa LLMSuccess || fallback()` path would absorb. A
miss on a later request of a streamed turn, or a request a stream's callback makes, is thrown inside the
call's task, so `fetch` throws a `TaskFailedException` around it. The message names the request, the start
of the key, the model and the directory, and how to record the answer: run the same code once with access
to the provider — its API key, or its local server running — and `mode = :record_missing`.

## [In a test suite](@id testing_suite)

Commit the recordings directory next to the tests, record it once with the provider's key and
`:record_missing`, and replay it on every run after that, in CI with no key:

```julia
using Test, UniLM

const ANSWERS = joinpath(@__DIR__, "recorded_answers")   # committed next to the tests
# calculate() as above

@testset "calculator" begin
    with_recorded_answers(ANSWERS) do
        @test text(calculate()) == "714"
    end
end
```

A recording is one sample of a reply that varies: replaying it makes a test reproducible, not the reply
typical. A scope covers the tasks started inside it. Recordings are never pruned: run the tests once in a
`:record` scope over a new directory, nested inside a `:replay` scope over the old one, and the new directory
receives exactly the recordings the tests use, copied without touching the network.

## [See also](@id testing_see_also)

- [Test and Develop](@ref jev_testing_guide) — tests of decision calls, recorded answers in a pipeline,
  and the semantic assertions they make possible
- [The request body](@ref chat_completions_body) — the body a `Chat` sends, without sending it
- [Results](@ref results_guide) — what a test can branch on

## [On your provider](@id testing_providers)

```@eval
Main.ProviderFacts.provider_table(Main.ProviderFacts.llm)
```
