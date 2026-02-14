# [Chat Types](@id chat_api)

```@docs
UniLM.UniLM
```

## Core Types

```@docs
Chat
Message
```

## Role Constants

```@docs
RoleSystem
RoleUser
RoleAssistant
UniLM.RoleTool
```

## Tool Types

```@docs
GPTTool
GPTFunctionSignature
GPTToolCall
GPTFunctionCallResult
```

## Output Format

```@docs
ResponseFormat
UniLM.GPTImageContent
```

## Functions

```@docs
chatrequest!
issendvalid
UniLM.getcontent
UniLM.getrole
UniLM.iscall
Base.push!(::Chat, ::Message)
Base.pop!(::Chat)
Base.getindex(::Chat, ::Int64)
Base.setindex!(::Chat, ::Message, ::Int64)
Base.last(::Chat)
Base.firstindex(::Chat)
Base.lastindex(::Chat)
UniLM.update!(::Chat, ::Message)
```
