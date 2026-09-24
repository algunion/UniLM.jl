# [Chat Completions](@id chat_guide)

The Chat Completions API is the standard way to talk to LLM providers.
UniLM.jl wraps it in a type-safe, stateful `Chat` object that tracks conversation history
automatically — and works with every supported backend (OpenAI, DeepSeek, Ollama, Gemini,
Mistral, and more).

```@setup chat
using UniLM
using JSON
```

## Creating a Chat

```@example chat
chat = Chat(
    model="gpt-5.4-mini",   # model name
    temperature=0.7,        # sampling temperature
)
println("Model: ", chat.model)
println("Messages: ", length(chat))
```

All parameters are optional with sensible defaults. See [`Chat`](@ref) for the full list.

## Building Conversations

Messages are added with `push!`, which enforces the conversation structure: a `push!` (or
`pop!`) that would produce an invalid sequence throws instead. The checks live in `push!` and
`pop!`; `chat[i] = msg` and direct edits to `chat.messages` bypass them.

```@example chat
# System message must come first
push!(chat, Message(Val(:system), "You are a helpful Julia programming tutor."))

# Then user messages
push!(chat, Message(Val(:user), "What are parametric types?"))

println("Conversation length: ", length(chat))
println("First message role: ", chat[1].role)
println("Last message role: ", chat[end].role)
```

The convenience `Val(:system)` and `Val(:user)` constructors keep things concise. You can also use the keyword constructor:

```@example chat
chat2 = Chat()
push!(chat2, Message(role="system", content="Be helpful"))
push!(chat2, Message(role="user", content="Tell me more"))
println("chat2 length: ", length(chat2))
```

### Conversation Rules

- The **first** message must have role `system`
- Messages must **alternate roles** (no two consecutive messages from the same role; consecutive `tool` results are the exception)
- At least `content`, `tool_calls`, or `refusal_message` must be non-`nothing`
- Attempting to violate these rules throws [`InvalidConversationError`](@ref) — the invalid message is never added
- A `Message` role must be `"system"`, `"user"`, `"assistant"` or `"tool"` (anything else throws `ArgumentError`)
- With `history=true`, sending a conversation that ends with an assistant message throws
  `InvalidConversationError` before any request: the reply could not be appended to it

```@example chat
# Demonstrate validation — an invalid mutation throws and leaves the chat unchanged
chat3 = Chat()
push!(chat3, Message(Val(:system), "sys"))
push!(chat3, Message(Val(:user), "hello"))
try
    push!(chat3, Message(Val(:user), "hello again"))  # same role — rejected
catch e
    println("Rejected: ", e isa InvalidConversationError)
end
println("Length after rejected push: ", length(chat3), " (still 2 — the invalid message was not added)")
```

## Sending Requests

```julia
result = chatrequest!(chat)
```

The `!` suffix is a Julia convention — `chatrequest!` mutates `chat` by appending the assistant's response to the message history (when `history=true`).

### Result Handling

```@example chat
result = chatrequest!(chat)
if result isa LLMSuccess
    println(result.message.content)
    println("\nFinish reason: ", result.message.finish_reason)
    println("Conversation length: ", length(chat))
else
    println("Request failed — see result for details")
end
```

### One-Shot Requests via Keywords

Skip the `Chat` object entirely for simple one-off requests:

```@example chat
result = chatrequest!(
    systemprompt="You are a calculator. Respond only with the number.",
    userprompt="What is 42 * 17?",
    model="gpt-5.4-mini",
    temperature=0.0
)
if result isa LLMSuccess
    println(result.message.content)
else
    println("Request failed — see result for details")
end
```

## Multi-Turn Conversations

Because `chatrequest!` appends the response, you can keep chatting:

```@example chat
chat = Chat(model="gpt-5.4-mini")
push!(chat, Message(Val(:system), "You are a concise Julia programming tutor."))
push!(chat, Message(Val(:user), "What is multiple dispatch? Answer in 2-3 sentences."))
result = chatrequest!(chat)
if result isa LLMSuccess
    println(result.message.content)
else
    println("Request failed — see result for details")
end
```

```@example chat
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

## Checking Conversation Validity

```@example chat
fresh = Chat(model="gpt-5.4-mini")
push!(fresh, Message(Val(:system), "You are a concise Julia programming tutor."))
push!(fresh, Message(Val(:user), "What is multiple dispatch?"))
println("Is chat valid? ", issendvalid(fresh))  # true — system + user

empty_chat = Chat()
println("Is empty chat valid? ", issendvalid(empty_chat))  # false
```

This checks:
- At least 2 messages
- First message is `system`
- Last message is `user`
- No consecutive same-role messages

## Models

UniLM.jl works with any model name string. Common choices:

| Model            | Usage                  |
| :--------------- | :--------------------- |
| `"gpt-5.6-sol"`  | Default; Chat tools require no reasoning; promotional price guaranteed through November 21, 2026 |
| `"gpt-6-astra"`  | Most capable; Responses-only tools, no `none`/`minimal` effort, \$10/\$50 per M tokens |
| `"gpt-6-sol"`    | Built for complex coding and agentic workflows, \$2/\$10 per M tokens |
| `"gpt-6-luna"`   | Fast and cheap, \$0.10/\$0.50 |
| `"gpt-5.6-luna"` | Fast and cheap         |
| `"gpt-5.4-mini"` | Mini model; reasoning effort `"none"` by default |
| `"gpt-4.1-mini"` | Balanced performance   |
| `"o4-mini"`      | Fast reasoning; shuts down October 23, 2026 (replacement `gpt-5.6-terra`) |

All eight are keys in [`DEFAULT_PRICING`](@ref), so cost accounting works out of
the box. Any other model name is accepted — it is just a string on the wire — but
an unpriced one estimates at `0.0` (with a one-time warning) — see
[Unpriced models cost \$0](@ref unpriced-zero) before relying on
`estimated_cost`.

## Using Other Providers

Pass a `service` to target any supported backend:

```julia
# DeepSeek (default model: deepseek-flash)
chat = Chat(service=DeepSeekEndpoint())

# Ollama (local)
chat = Chat(service=OllamaEndpoint(), model="llama3.1")

# Mistral
chat = Chat(service=MistralEndpoint(), model="mistral-large-latest")
```

See the [Multi-Backend Guide](@ref backend_guide) for the full list of providers and configuration.

## JSON Serialization

The `Chat` object serializes cleanly to JSON for the API:

```@example chat
println(JSON.json(chat))
```

## Retry Behaviour

`chatrequest!` automatically retries transient HTTP statuses (408, 429, 500, 502, 503, 504, 529) with exponential backoff and jitter; a `Retry-After` header is a floor under the jittered wait, so concurrent callers that received the same header do not retry in lockstep. Attempts and total time are bounded by the resolved [`RequestConfig`](@ref) (`max_attempts`, default 3; `total_deadline`, default 900 s). Pass `config=RequestConfig(max_attempts=1)` to disable retries for a call, or set scoped/process-wide defaults with `with_request_config` / `set_default_config!`. Timeouts surface as `LLMCallError` with `status = nothing` and the `UniLMTimeout` (phase, elapsed, limit) in `.cause`; a cancelled call (see [Concurrency, Tasks and Cancellation](@ref concurrency_guide)) carries a `UniLMCancelled` there instead.

Local validation is not a result: an option the provider or model cannot express, a service
that does not declare `:chat`, or a conversation the reply could not be appended to throws
(`ArgumentError` / `InvalidConversationError`) before any request is sent.

## Parameter Validation

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
exclusive, and an empty `tools` vector is stored as `nothing`. Provider- and model-specific
limits (for example GPT-5.6 Chat tools requiring `reasoning_effort="none"`, or Claude's
temperature range) are checked when the request is encoded, before any network I/O.

## See Also

- [`Chat`](@ref) — full type reference
- [`Message`](@ref) — message type reference
- [Tool Calling](@ref tools_guide) — function calling with Chat Completions
- [Streaming](@ref streaming_guide) — real-time streaming
- [Structured Output](@ref structured_guide) — JSON-constrained generation
