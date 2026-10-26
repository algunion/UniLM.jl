# [FIM & Prefix Completion](@id completions_guide)

```@setup completions
using UniLM
using JSON
```

**Fill-in-the-Middle (FIM)** completion has a model write the text between a prefix and a suffix — the
gap in the code you are editing, say — and **chat prefix completion** has it continue an assistant reply
from the start you give it, to force an output format such as a code block. A provider can serve them
only when its endpoint's capabilities hold `:fim` or `:prefix_completion`, and a generic endpoint
(`GenericOpenAIEndpoint`) declares `:fim` for any server: whether yours fills in the middle is on its entry,
which the table at the end links.

```@example completions
service, model = DeepSeekEndpoint(), "deepseek-flash"
nothing # hide
```

The examples use only the names bound above. `service` and `model` are the provider and model they run
on: bind your own there. The table [at the end](@ref completions_providers) links each provider's rules.

## [FIM completion](@id completions_fim)

FIM generates text between a `prompt` (prefix) and an optional `suffix`. This is the
standard pattern for code completion — the model fills in the gap.

### [Building a request](@id completions_fim_request)

```@example completions
fim = FIMCompletion(
    service=service,
    model=model,
    prompt="def fib(a):",
    suffix="    return fib(a-1) + fib(a-2)",
    max_tokens=128,
    stop=["\n\n"]
)
println("Model: ", fim.model)
println("Prompt: ", fim.prompt)
println("Suffix: ", fim.suffix)
println("JSON body:")
println(JSON.json(fim, 2))
```

A `FIMCompletion` without `model` takes the endpoint's default FIM model, and the request throws
`ArgumentError` for an endpoint that has none; `max_tokens` is 128 unless it is set, and the other fields
are sent only when they are set.

### [Making requests](@id completions_fim_send)

```julia
# Struct form
result = fim_complete(fim)
if result isa FIMSuccess
    println(fim_text(result))
end

# Convenience form
result = fim_complete("def hello():";
    service, model,
    suffix="    print('done')",
    max_tokens=64,
    stop=["\n\n"])
```

`fim_complete` sends the request to the endpoint's completions route, which its entry names, and reads
the completion's text from each choice of the reply.

### [Result types](@id completions_fim_results)

FIM returns [`FIMSuccess`](@ref), [`FIMFailure`](@ref), or [`FIMCallError`](@ref). Local
validation — the endpoint's `:fim` capability, model resolution, routing, and
`stream=true` (FIM has no streaming path) — throws `ArgumentError` before any request;
every later failure, including a `200` whose body is not a completions response, is a
`FIMCallError` carrying the exception in `cause`. A `FIMSuccess` reports token usage and
is priced by [`estimated_cost`](@ref).

[`fim_text`](@ref) returns the generated text of a success. A failed call has no text,
so on a `FIMFailure` or `FIMCallError` it throws [`LLMResultError`](@ref) instead of
returning `""` — guard with [`issuccess`](@ref):

```@example completions
failed = FIMFailure(response="err", status=400)
println("issuccess: ", issuccess(failed))
try
    fim_text(failed)
catch e
    println(sprint(showerror, e))
end
```

## [Chat prefix completion](@id completions_prefix)

Prefix completion lets you provide a partial assistant message that the model continues
from. Useful for forcing specific output formats (e.g., starting with a code block).

### [Usage](@id completions_prefix_usage)

The last message in the chat must be `role=assistant` with the partial text:

```julia
chat = Chat(service=service, model=model)
push!(chat, Message(Val(:system), "You are a coding assistant."))
push!(chat, Message(Val(:user), "Write a Python quicksort"))
push!(chat, Message(role=RoleAssistant, content="```python\n"))

result = prefix_complete(chat)
# The model continues from "```python\n"
result isa LLMSuccess && println(result.message.content)
```

The request is the endpoint's own `Chat` request, its last message marked as the prefix to continue. The
result's `message` is the continuation the API returns. With `chat.history` (the
default) the conversation keeps the whole assistant turn — the prefix followed by the
continuation. A failure or a cancellation (`cancel=`, or an ambient `with_cancel`
token) leaves `chat` untouched.

### [Validation](@id completions_prefix_validation)

`prefix_complete` validates that:
- The chat is not empty
- The last message has `role=assistant`

```@example completions
chat = Chat(service=service, model=model)
push!(chat, Message(Val(:system), "sys"))
push!(chat, Message(Val(:user), "hello"))
println("Last role: ", chat[end].role)
# This would throw — last message is not assistant:
# prefix_complete(chat)  # ArgumentError
```

## [Capability validation](@id completions_capabilities)

Both `fim_complete` and `prefix_complete` check the endpoint's capabilities before making a request: an
endpoint without `:fim`, or without `:prefix_completion`, throws `ArgumentError` naming the feature and the
capabilities it has. [Capabilities by provider](@ref capabilities_guide) shows which endpoints declare
them, a generic endpoint's as sent unchecked: it declares `:fim` whatever its server serves.

```@example completions
println(has_capability(service, :fim))
println(has_capability(service, :prefix_completion))
```

## [See also](@id completions_see_also)

- [FIM Types](@ref fim_api) — full type and function reference
- [Provider Capabilities](@ref capabilities_api) — capability system
- [One API across providers](@ref backend_guide) — switching providers

## [On your provider](@id completions_providers)

```@eval
Main.ProviderFacts.on_your_provider(Main.ProviderFacts.llm, :completions)
```
