# [Chat Completions](@id chat_guide)

The Chat Completions API is the classic way to interact with OpenAI's language models.
UniLM.jl wraps it in a type-safe, stateful `Chat` object that tracks conversation history automatically.

## Creating a Chat

```julia
using UniLM

chat = Chat(
    model="gpt-4o",        # model name
    temperature=0.7,        # sampling temperature
    max_tokens=1000,        # response length limit
)
```

All parameters are optional with sensible defaults. See [`Chat`](@ref) for the full list.

## Building Conversations

Messages are added with `push!`. UniLM.jl enforces conversation structure at the type level — you cannot create invalid message sequences:

```julia
# System message must come first
push!(chat, Message(Val(:system), "You are a helpful Julia programming tutor."))

# Then user messages
push!(chat, Message(Val(:user), "What are parametric types?"))
```

The convenience `Val(:system)` and `Val(:user)` constructors keep things concise. You can also use the keyword constructor:

```julia
push!(chat, Message(role="user", content="Tell me more"))
```

### Conversation Rules

- The **first** message must have role `system`
- Messages must **alternate roles** (no two consecutive messages from the same role)
- At least `content`, `tool_calls`, or `refusal_message` must be non-`nothing`
- Attempting to violate these rules logs a warning and the message is **not added**

## Sending Requests

```julia
result = chatrequest!(chat)
```

The `!` suffix is a Julia convention — `chatrequest!` mutates `chat` by appending the assistant's response to the message history (when `history=true`).

### Result Handling

```julia
if result isa LLMSuccess
    println(result.message.content)

    # The chat now has 3 messages: system, user, assistant
    @assert length(chat) == 3
elseif result isa LLMFailure
    @warn "HTTP $(result.status): $(result.response)"
elseif result isa LLMCallError
    @error "Exception: $(result.error)"
end
```

### One-Shot Requests via Keywords

Skip the `Chat` object entirely for simple one-off requests:

```julia
result = chatrequest!(
    systemprompt="You are a translator.",
    userprompt="Translate 'Hello world' to French.",
    model="gpt-4o-mini",
    temperature=0.0
)
```

## Multi-Turn Conversations

Because `chatrequest!` appends the response, you can keep chatting:

```julia
chat = Chat(model="gpt-4o")
push!(chat, Message(Val(:system), "You are a math tutor. Show your work."))

# Turn 1
push!(chat, Message(Val(:user), "What is the integral of x²?"))
r1 = chatrequest!(chat)
println(r1.message.content)

# Turn 2 — history is automatic
push!(chat, Message(Val(:user), "Now compute the definite integral from 0 to 1"))
r2 = chatrequest!(chat)
println(r2.message.content)
```

## Checking Conversation Validity

```julia
issendvalid(chat)  # => true if the conversation is well-formed
```

This checks:
- At least 2 messages
- First message is `system`
- Last message is `user`
- No consecutive same-role messages

## Models

UniLM.jl works with any model name string. Common choices:

| Model                  | Usage                  |
| :--------------------- | :--------------------- |
| `"gpt-4o"`             | Best quality (default) |
| `"gpt-4o-mini"`        | Fast and cheap         |
| `"gpt-4-1106-preview"` | GPT-4 Turbo            |

## Retry Behaviour

`chatrequest!` automatically retries on HTTP 500/503 errors with a 1-second delay, up to 30 retries. This is transparent and requires no configuration.

## See Also

- [`Chat`](@ref) — full type reference
- [`Message`](@ref) — message type reference
- [Tool Calling](@ref tools_guide) — function calling with Chat Completions
- [Streaming](@ref streaming_guide) — real-time streaming
- [Structured Output](@ref structured_guide) — JSON-constrained generation
