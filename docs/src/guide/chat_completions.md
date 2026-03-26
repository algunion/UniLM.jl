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
    model="gpt-5.2",        # model name
    temperature=0.7,        # sampling temperature
)
println("Model: ", chat.model)
println("Messages: ", length(chat))
```

All parameters are optional with sensible defaults. See [`Chat`](@ref) for the full list.

## Building Conversations

Messages are added with `push!`. UniLM.jl enforces conversation structure at the type level — you cannot create invalid message sequences:

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
- Messages must **alternate roles** (no two consecutive messages from the same role)
- At least `content`, `tool_calls`, or `refusal_message` must be non-`nothing`
- Attempting to violate these rules logs a warning and the message is **not added**

```@example chat
# Demonstrate validation
chat3 = Chat()
push!(chat3, Message(Val(:system), "sys"))
push!(chat3, Message(Val(:user), "hello"))
push!(chat3, Message(Val(:user), "hello again"))  # rejected — same role
println("Length after duplicate push: ", length(chat3), " (second user msg rejected)")
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
    model="gpt-4o-mini",
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
chat = Chat(model="gpt-4o-mini")
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
push!(chat, Message(Val(:user), "Give a short Julia code example of it."))
result = chatrequest!(chat)
if result isa LLMSuccess
    println(result.message.content)
    println("\nConversation length: ", length(chat))
else
    println("Request failed — see result for details")
end
```

## Checking Conversation Validity

```@example chat
println("Is chat valid? ", issendvalid(chat))  # true — system + user

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
| `"gpt-5.2"`      | Best quality (default) |
| `"gpt-4o-mini"`  | Fast and cheap         |
| `"gpt-4.1-mini"` | Balanced performance   |
| `"o3"`           | Extended reasoning     |
| `"o4-mini"`      | Fast reasoning         |

## Using Other Providers

Pass a `service` to target any supported backend:

```julia
# DeepSeek
chat = Chat(service=DeepSeekEndpoint(), model="deepseek-chat")

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

`chatrequest!` automatically retries on HTTP 429, 500, and 503 errors with exponential backoff and jitter (up to 30 attempts, max 60s delay). On 429 responses, the `Retry-After` header is respected. This is transparent and requires no configuration.

## Parameter Validation

The `Chat` constructor validates parameter ranges at construction time:

| Parameter           | Valid Range      |
| :------------------ | :--------------- |
| `temperature`       | 0.0–2.0          |
| `top_p`             | 0.0–1.0          |
| `n`                 | 1–10             |
| `presence_penalty`  | -2.0–2.0         |
| `frequency_penalty` | -2.0–2.0         |

Out-of-range values throw `ArgumentError`. Additionally, `temperature` and `top_p` are mutually exclusive.

## See Also

- [`Chat`](@ref) — full type reference
- [`Message`](@ref) — message type reference
- [Tool Calling](@ref tools_guide) — function calling with Chat Completions
- [Streaming](@ref streaming_guide) — real-time streaming
- [Structured Output](@ref structured_guide) — JSON-constrained generation
