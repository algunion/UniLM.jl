# [Chat Completions API](@id chat_api)

Types and functions for the **Chat Completions API**.

## Chat Object

```@docs
Chat
```

## Messages

```@docs
Message
```

### Convenience Constructors

```@example chat_api
using UniLM

sys = Message(Val(:system), "You are a helpful assistant")
usr = Message(Val(:user), "Hello!")
println("System role: ", sys.role)
println("User role: ", usr.role)
println("User content: ", usr.content)
```

## Conversation Management

```@docs
issendvalid
update!
```

### Building a Conversation

```@example chat_api
chat = Chat(model="gpt-5.2")
push!(chat, Message(Val(:system), "You are a helpful assistant"))
push!(chat, Message(Val(:user), "What is Julia?"))
println("Messages: ", length(chat))
println("Valid for sending: ", issendvalid(chat))
```

## Tools

```@docs
GPTTool
GPTFunctionSignature
GPTToolCall
GPTFunctionCallResult
```

## Output Format

```@docs
ResponseFormat
```

```@example chat_api
using JSON

# JSON object format
fmt = ResponseFormat()
println("Type: ", fmt.type)
println("JSON: ", JSON.json(fmt))
```

## Request Function

```@docs
chatrequest!
```

## Role Constants

```@docs
RoleSystem
RoleUser
RoleAssistant
```

## Model Constants

```@example chat_api
println("GPT5_2: ", UniLM.GPT5_2)
```
