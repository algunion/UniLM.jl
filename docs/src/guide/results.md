# [Results](@id results_guide)

```@setup results
using UniLM
```

A request to a model returns what the service or the network did as a typed result instead of throwing
it: a success holding the reply, a failure holding what the provider answered with, or a call error
holding what ended the call. Branch on the type, or ask [`issuccess`](@ref), before you read the reply.

```@example results
service, model = OPENAIServiceEndpoint, "gpt-5.4-mini"
nothing # hide
```

The examples use only the names bound above. `service` and `model` are the provider and model they run
on: bind your own there. The table [at the end](@ref results_providers) links each provider's rules.

## [Three outcomes](@id results_outcomes)

| Verb | Success | Failure | Call error |
| :--- | :--- | :--- | :--- |
| `chatrequest!`, `prefix_complete` | [`LLMSuccess`](@ref) | [`LLMFailure`](@ref) | [`LLMCallError`](@ref) |
| `respond` | [`ResponseSuccess`](@ref) | [`ResponseFailure`](@ref) | [`ResponseCallError`](@ref) |
| `embeddingrequest!` | [`EmbeddingSuccess`](@ref) | [`EmbeddingFailure`](@ref) | [`EmbeddingCallError`](@ref) |
| `fim_complete` | [`FIMSuccess`](@ref) | [`FIMFailure`](@ref) | [`FIMCallError`](@ref) |

Each is an [`LLMRequestResponse`](@ref):

- **A success** holds the reply: an `LLMSuccess` its `message`, the conversation as `self` and the
  `usage`; a `ResponseSuccess` the response as a [`ResponseObject`](@ref), with its `id`, `status`,
  `model`, `output` items, `usage` and the whole body as `raw`.
- **A failure** holds the HTTP `status` and, as `response`, the body the provider answered with. From
  `respond`, a `ResponseFailure` also stands for a response that ended `failed`, and a response with
  any other status — `incomplete`, `in_progress`, `cancelled`, … — is a `ResponseSuccess`, read with
  [`response_status`](@ref).
- **A call error** holds, as `error`, the text of what ended the call without a reply it could use, its
  HTTP `status` when there was one, and the exception, when there was one, as `cause`: a
  [`UniLMTimeout`](@ref) for a bound that ran out, a [`UniLMCancelled`](@ref) for a stop or a cancel,
  the connection's exception, or the error of a reply it could not read.

A failure or call error of `chatrequest!`, `respond` or `fim_complete` also keeps the provider's request
id header, when one came, as `request_id`. A tool loop returns a [`ToolLoopResult`](@ref) whose
`response` is the last result it received ([Inspecting the result](@ref tool_calling_result)). Of the
lifecycle verbs of stored responses, [`delete_response`](@ref), [`list_input_items`](@ref),
[`compact_response`](@ref) and [`count_input_tokens`](@ref) return the provider's answer as a `Dict` on
success, and a `ResponseFailure` or `ResponseCallError` otherwise.

```@example results
result = chatrequest!(
    systemprompt="You are a calculator. Respond only with the number.",
    userprompt="What is 42 * 17?",
    service=service,
    model=model,
    temperature=0.0
)
result isa LLMSuccess, issuccess(result)
```

## [Reading a success](@id results_success)

[`text`](@ref) is the reply of an `LLMSuccess`, and [`output_text`](@ref) the answer of a
`ResponseSuccess`; `result.message` holds the whole reply, its `finish_reason`, `tool_calls` and
`refusal_message` among it. An `LLMSuccess`'s `model` is the model the provider reports having served
the turn where UniLM keeps the name its reply gives, and `nothing` otherwise (each entry's first call says
which); a `ResponseSuccess`'s is `result.response.model`.
[`token_usage`](@ref) reads the usage in one shape for every provider,
a [`TokenUsage`](@ref) — `prompt_tokens` counts all the input, cache reads and writes included, and
`completion_tokens` all the output, reasoning included — and [`estimated_cost`](@ref) prices it in USD
([Cost Tracking](@ref cost_guide)):

```@example results
u = token_usage(result)
(; reply = text(result), result.message.finish_reason, u.prompt_tokens, u.completion_tokens,
   cost = round(estimated_cost(result); digits=6))
```

A streamed result also counts, as `sse_dropped`, the stream payloads that could not be read and were
dropped while it was assembled, `0` for a clean stream and for an unstreamed call
([Dropped SSE payloads](@ref streaming_dropped)).

## [Handling a failure](@id results_failures)

A failure holds no reply: `text`, `output_text`, `reasoning_text`, `embedding_vectors` and `fim_text`
throw [`LLMResultError`](@ref) for one, which names its status and the start of its body, and its usage
is zero. This failure is built by hand, as a 429 would arrive:

```@example results
chat = Chat(service=service, model=model)
failure = LLMFailure(status=429, response="{\"error\": {\"message\": \"Rate limit reached\"}}", self=chat)
issuccess(failure), token_usage(failure).prompt_tokens, estimated_cost(failure)
```

```@example results
err = try text(failure); nothing catch e; e end
err isa LLMResultError || error("expected text to refuse a failure")
print(sprint(showerror, err))
```

What a call throws instead of returning is what UniLM refuses before sending — an `ArgumentError` for a
request that breaks a rule it checks, an [`InvalidConversationError`](@ref) for a conversation the reply
could not be appended to —, a user `InterruptException`, and, in a recorded call, a
[`ReplayMissError`](@ref) or [`RecordingWriteError`](@ref) ([Recorded answers](@ref jev_testing_recorded)).
In a stream, the last three can come from its task, at `fetch`, wrapped in a `TaskFailedException`.
Which statuses and errors a provider returns, and what to do about each, is in its entry's failure
section.

## [On your provider](@id results_providers)

```@eval
Main.ProviderFacts.on_your_provider(Main.ProviderFacts.llm, :failures)
```
