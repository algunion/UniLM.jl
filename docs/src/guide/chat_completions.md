# [Conversations with Chat](@id chat_guide)

```@setup chat_completions
using UniLM
```

A [`Chat`](@ref) holds a conversation — its messages and the options of its requests — and
[`chatrequest!`](@ref) sends it to a provider and, with `history=true` (the default), appends the
reply, so the next turn carries the conversation so far. Use it to talk with a language model turn by
turn; [`respond`](@ref) sends one request of a provider's agentic API instead
([Agentic respond](@ref agentic_guide)).

```@example chat_completions
service, model = OPENAIServiceEndpoint, "gpt-5.4-mini"
nothing # hide
```

The examples use only the names bound above. `service` and `model` are the provider and model they run
on: bind your own there. The table [at the end](@ref chat_completions_providers) links each provider's
rules.

## [Creating a Chat](@id chat_completions_create)

```@example chat_completions
chat = Chat(service=service, model=model, temperature=0.7)
println("Model: ", chat.model)
println("Messages: ", length(chat))
```

`service` names the provider and `model` its model; a `Chat` without `model` takes the provider's
default model, and throws `ArgumentError` for a provider that has none. Every field has a default, and
the constructor checks the ranges of the options it is given ([Parameter validation](@ref chat_completions_validation)).
See [`Chat`](@ref) for the full list.

## [Building conversations](@id chat_completions_messages)

Messages are added with `push!`, which refuses one that would make the conversation invalid: it throws
[`InvalidConversationError`](@ref) for a first message that is not a system message, and for a message
whose role repeats the last one's — consecutive `tool` results excepted. `pop!` removes the last message
and throws for an empty conversation. `chat[i] = msg` and direct edits to `chat.messages` bypass these
checks.

```@example chat_completions
# System message must come first
push!(chat, Message(Val(:system), "You are a helpful Julia programming tutor. Answer in plain text, without Markdown, in at most three sentences."))

# Then user messages
push!(chat, Message(Val(:user), "What are parametric types?"))

println("Conversation length: ", length(chat))
println("First message role: ", chat[1].role)
println("Last message role: ", chat[end].role)
```

The convenience `Val(:system)` and `Val(:user)` constructors keep things concise. You can also use the keyword constructor:

```@example chat_completions
chat2 = Chat(service=service, model=model)
push!(chat2, Message(role="system", content="Be helpful"))
push!(chat2, Message(role="user", content="Tell me more"))
println("chat2 length: ", length(chat2))
```

A user message can also carry images, sound clips and PDF documents ([Attachments](@ref attachments_guide)).

### [Conversation rules](@id chat_completions_rules)

- A [`Message`](@ref) has the role `"system"`, `"user"`, `"assistant"` or `"tool"`, and at least a text,
  tool calls, a refusal, attachments or an audio reply; its constructor throws `ArgumentError` for one
  that has none, for another role, and for a `tool` message without its `tool_call_id`
- The **first** message must have role `system`
- Messages must **alternate roles** (no two consecutive messages from the same role; consecutive `tool` results are the exception)
- A `push!` that breaks either rule throws [`InvalidConversationError`](@ref) — the invalid message is never added
- With `history=true`, sending a conversation that ends with an assistant message throws
  `InvalidConversationError` before any request: the reply could not be appended to it

```@example chat_completions
# Demonstrate validation — an invalid mutation throws and leaves the chat unchanged
chat3 = Chat(service=service, model=model)
push!(chat3, Message(Val(:system), "sys"))
push!(chat3, Message(Val(:user), "hello"))
try
    push!(chat3, Message(Val(:user), "hello again"))  # same role — rejected
catch e
    println("Rejected: ", e isa InvalidConversationError)
end
println("Length after rejected push: ", length(chat3), " (still 2 — the invalid message was not added)")
```

## [Sending requests](@id chat_completions_send)

```julia
result = chatrequest!(chat)
```

The `!` suffix is a Julia convention — `chatrequest!` mutates `chat` by appending the assistant's response to the message history (when `history=true`).

```@example chat_completions
result = chatrequest!(chat)
if result isa LLMSuccess
    println(result.message.content)
    println("\nFinish reason: ", result.message.finish_reason)
    println("Conversation length: ", length(chat))
else
    println("Request failed — see result for details")
end
```

`chatrequest!` returns what the service or the network did as a typed result instead of throwing it:
an [`LLMSuccess`](@ref) holding the reply, an `LLMFailure` holding the HTTP `status` and body the
provider answered with, or an `LLMCallError` for a call that brought no reply it could use, the
exception — a timeout, a cancellation — in its `cause`. [Results](@ref results_guide) has them all.

### [One-shot requests via keywords](@id chat_completions_keywords)

Skip the `Chat` object entirely for simple one-off requests; every keyword besides the prompts is a
`Chat` field:

```@example chat_completions
result = chatrequest!(
    systemprompt="You are a calculator. Respond only with the number.",
    userprompt="What is 42 * 17?",
    service=service,
    model=model,
    temperature=0.0
)
if result isa LLMSuccess
    println(result.message.content)
else
    println("Request failed — see result for details")
end
```

## [Multi-turn conversations](@id chat_completions_turns)

Because `chatrequest!` appends the response, you can keep chatting:

```@example chat_completions
chat = Chat(service=service, model=model)
push!(chat, Message(Val(:system), "You are a concise Julia programming tutor. Answer in plain text, without Markdown."))
push!(chat, Message(Val(:user), "What is multiple dispatch? Answer in 2-3 sentences."))
result = chatrequest!(chat)
if result isa LLMSuccess
    println(result.message.content)
else
    println("Request failed — see result for details")
end
```

```@example chat_completions
# `chatrequest!` appends the assistant reply on success, which is what makes the next
# user turn valid. If the previous call appended none (e.g. it failed), pushing another
# user message would throw InvalidConversationError — so guard the follow-up turn.
if !isempty(chat) && chat[end].role == RoleAssistant
    push!(chat, Message(Val(:user), "Give a short Julia code example of it."))
    result = chatrequest!(chat)
    if result isa LLMSuccess
        println(result.message.content)
        println("\nConversation length: ", length(chat))
    else
        println("Request failed — see result for details")
    end
else
    println("No assistant reply to build on — skipping the follow-up turn.")
end
```

## [Checking conversation validity](@id chat_completions_validity)

```@example chat_completions
fresh = Chat(service=service, model=model)
push!(fresh, Message(Val(:system), "You are a concise Julia programming tutor. Answer in plain text, without Markdown."))
push!(fresh, Message(Val(:user), "What is multiple dispatch?"))
println("Is chat valid? ", issendvalid(fresh))  # true — system + user

empty_chat = Chat(service=service, model=model)
println("Is empty chat valid? ", issendvalid(empty_chat))  # false
```

This checks:
- At least 2 messages
- First message is `system`
- Last message is `user`
- No consecutive same-role messages

## [The request body](@id chat_completions_body)

[`UniLM.encode_request`](@ref)`(service, chat)` returns the body `chatrequest!` sends the provider, with
no key and no request — or throws the refusal of a request the provider's encoder checks:

```@example chat_completions
println(UniLM.encode_request(service, chat))
```

`JSON.json(chat)` writes the `Chat` in the Chat Completions shape of the OpenAI-compatible wire, which
is not every provider's body.

## [Retries and failures](@id chat_completions_retries)

`chatrequest!` automatically retries transient HTTP statuses (408, 429, 500, 502–504, 520–524, 529) with
exponential backoff and jitter; a `Retry-After` header is a floor under the jittered wait, so concurrent
callers that received the same header do not retry in lockstep. A 429 whose error names spent credit or
a reached spend, usage or quota limit is returned at once, since no retry clears it; each provider's
failure section names the codes it sends. Attempts and total time are bounded by the resolved
[`RequestConfig`](@ref) (`max_attempts`, default 3; `total_deadline`, default 900 s). Pass
`config=RequestConfig(max_attempts=1)` to disable retries for a call, or set scoped/process-wide defaults
with `with_request_config` / `set_default_config!` ([Timeouts & Retries](@ref timeouts_guide)). Timeouts
surface as `LLMCallError` with `status = nothing` and the `UniLMTimeout` (phase, elapsed, limit) in
`.cause`; a cancelled call (see [Concurrency, Tasks and Cancellation](@ref concurrency_guide)) carries a
`UniLMCancelled` there instead.

A check made before sending is not a result: `chatrequest!` throws `ArgumentError` for a `callback` or
`on_tool_call` without `stream=true`, for an endpoint whose declared capabilities lack `:chat`, and for a
request the provider's encoder refuses, and `InvalidConversationError` for a conversation the reply could
not be appended to — each before any request is sent.

## [Parameter validation](@id chat_completions_validation)

The `Chat` constructor validates parameter ranges at construction time:

| Parameter           | Valid Range      |
| :------------------ | :--------------- |
| `temperature`       | 0.0–2.0          |
| `top_p`             | 0.0–1.0          |
| `n`                 | 1 (a result carries a single choice) |
| `max_tokens`, `max_completion_tokens` | ≥ 1 |
| `presence_penalty`  | -2.0–2.0         |
| `frequency_penalty` | -2.0–2.0         |
| `top_logprobs`      | 0–20             |
| `logit_bias` values | -100–100 (any `Real`) |
| `reasoning_effort`  | `"none"`, `"minimal"`, `"low"`, `"medium"`, `"high"`, `"xhigh"`, `"max"` |

Out-of-range values throw `ArgumentError`. Additionally, `temperature` and `top_p` are mutually
exclusive, and an empty `tools` vector is stored as `nothing`. Which fields and values a provider and
its models take is checked when the request is encoded, before any network I/O, by the rules its
entry's limits section lists. UniLM refuses only what those rules name: a request they let through goes
to the provider, whose answer is the check.

## [Model notes](@id chat_completions_models)

`model` names any model the provider serves. The rules and prices UniLM holds for a provider's models
are on its entry, those of the provider bound above on [Models](@ref openai_models); its descriptions of
some of them:

| `model` | The provider's description |
| :--- | :--- |
| `gpt-6-sol` | "built for complex coding and agentic workflows" ([GPT-6 Sol](https://developers.openai.com/api/docs/models/gpt-6-sol)) |
| `gpt-6.1-sol` | "Near-Astra performance for complex work at a lower cost" ([GPT-6 guide](https://developers.openai.com/api/docs/guides/latest-model/gpt-6-astra)) |
| `gpt-6-astra` | "Highest intelligence", "For the most demanding reasoning, coding, and professional work" ([GPT-6 guide](https://developers.openai.com/api/docs/guides/latest-model/gpt-6-astra)) |
| `gpt-6-luna` | "Fastest and most cost-effective", "Strong performance for focused, high-volume tasks" ([GPT-6 guide](https://developers.openai.com/api/docs/guides/latest-model/gpt-6-astra)) |
| `gpt-5.6-sol` | "the model for flagship capability" ([GPT-5.6 guide](https://developers.openai.com/api/docs/guides/latest-model/gpt-5.6)), whose "promotional pricing is available at least through November 21, 2026" ([pricing](https://developers.openai.com/api/docs/pricing)) |
| `gpt-5.6-terra` | "for a balance of intelligence and cost" ([GPT-5.6 guide](https://developers.openai.com/api/docs/guides/latest-model/gpt-5.6)) |
| `gpt-5.6-luna` | "for efficient, high-volume workloads" ([GPT-5.6 guide](https://developers.openai.com/api/docs/guides/latest-model/gpt-5.6)) |
| `gpt-5.4-mini` | reasoning effort "none (default), low, medium, high and xhigh" ([GPT-5.4 Mini](https://developers.openai.com/api/docs/models/gpt-5.4-mini)) |

## [See also](@id chat_completions_see_also)

- [`Chat`](@ref) — full type reference
- [`Message`](@ref) — message type reference
- [Tool Calling](@ref tools_guide) — function calling with a `Chat`
- [Streaming](@ref streaming_guide) — real-time streaming
- [Structured Output](@ref structured_guide) — JSON-constrained generation
- [Results](@ref results_guide) — success, failure and call error
- [One API across providers](@ref backend_guide) — switching providers
- [Decisions with LLMs](@ref jev_llm_guide) — route a message before the request, or check a reply before a user sees it

## [On your provider](@id chat_completions_providers)

```@eval
Main.ProviderFacts.on_your_provider(Main.ProviderFacts.llm, :conversations)
```
