# [Request Configuration & Timeouts](@id timeouts_api)

See the [Timeouts & Retries guide](@ref timeouts_guide) for usage and rationale.

Timeout, deadline, and retry-budget configuration for UniLM network
operations, plus the typed timeout errors.

## Configuration

```@docs
RequestConfig
current_config
with_request_config
set_default_config!
```

## Timeout Errors

```@docs
UniLMTimeout
MCPTimeoutError
```

## Cancellation

A cooperative, level-triggered token: cancel it from any task, and every operation
that observes it — explicitly through a `cancel` keyword, or as the ambient token of a
[`with_cancel`](@ref) scope — stops and reports [`UniLMCancelled`](@ref) in its
call-error result. See [Concurrency, Tasks and Cancellation](@ref concurrency_guide).

```@docs
CancelToken
cancel!
iscancelled
with_cancel
UniLMCancelled
```

## Resolution Precedence

A request resolves its `RequestConfig` struct-wise (whichever channel wins
supplies every field):

1. A per-call `config` keyword (a complete struct wins outright).
2. The innermost active [`with_request_config`](@ref) scope.
3. The process default set by [`set_default_config!`](@ref).
4. The built-in field defaults.
