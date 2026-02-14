# Result Types

All API calls return subtypes of [`LLMRequestResponse`](@ref). Use Julia's type dispatch
or `isa` checks to handle different outcomes.

## Abstract Base

```@docs
LLMRequestResponse
```

## Chat Completions Results

```@docs
LLMSuccess
LLMFailure
LLMCallError
```

## Responses API Results

```@docs
ResponseSuccess
ResponseFailure
ResponseCallError
```

## Exceptions

```@docs
InvalidConversationError
```
