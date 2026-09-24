# [FIM & Prefix Completion](@id completions_guide)

UniLM.jl supports **Fill-in-the-Middle (FIM)** completion and **Chat Prefix Completion** —
features for code completion and controlled text generation.

FIM is supported by [`DeepSeekEndpoint`](@ref) (beta), [`MistralEndpoint`](@ref) and
Ollama; vLLM's `/v1/completions` does not accept `suffix`, so it cannot fill in the
middle. Prefix completion is supported by [`DeepSeekEndpoint`](@ref) (beta). The default
DeepSeek model for both is `deepseek-flash`.

```@setup completions
using UniLM
using JSON
```

---

## FIM Completion

FIM generates text between a `prompt` (prefix) and an optional `suffix`. This is the
standard pattern for code completion — the model fills in the gap.

### Basic Usage

```@example completions
fim = FIMCompletion(
    service=DeepSeekEndpoint("demo-key"),
    model="deepseek-flash",
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

### Making Requests

```julia
# Struct form
result = fim_complete(fim)
if result isa FIMSuccess
    println(fim_text(result))
end

# Convenience form
result = fim_complete("def hello():",
    service=DeepSeekEndpoint(),
    suffix="    print('done')",
    max_tokens=64,
    stop=["\n\n"])
```

### Result Types

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

### Provider Support

| Provider | Endpoint | Notes |
|---|---|---|
| DeepSeek | `DeepSeekEndpoint()` | Beta — uses `api.deepseek.com/beta` internally |
| Mistral | `MistralEndpoint()` | Via `/v1/fim/completions`; its `message.content` choices are read |
| Ollama | `OllamaEndpoint()` | Via `/v1/completions` |

```@example completions
# URL routing adapts per provider
println("DeepSeek: ", UniLM.get_url(DeepSeekEndpoint("k"), fim))
println("Mistral:  ", UniLM.get_url(MistralEndpoint(api_key="k"), fim))
println("Ollama:   ", UniLM.get_url(OllamaEndpoint(), fim))
```

---

## Chat Prefix Completion

Prefix completion lets you provide a partial assistant message that the model continues
from. Useful for forcing specific output formats (e.g., starting with a code block).

### Usage

The last message in the chat must be `role=assistant` with the partial text:

```julia
chat = Chat(service=DeepSeekEndpoint())   # deepseek-flash
push!(chat, Message(Val(:system), "You are a coding assistant."))
push!(chat, Message(Val(:user), "Write a Python quicksort"))
push!(chat, Message(role=RoleAssistant, content="```python\n"))

result = prefix_complete(chat)
# The model continues from "```python\n"
result isa LLMSuccess && println(result.message.content)
```

The result's `message` is the continuation the API returns. With `chat.history` (the
default) the conversation keeps the whole assistant turn — the prefix followed by the
continuation. A failure or a cancellation (`cancel=`, or an ambient `with_cancel`
token) leaves `chat` untouched.

### Validation

`prefix_complete` validates that:
- The chat is not empty
- The last message has `role=assistant`

```@example completions
chat = Chat(service=DeepSeekEndpoint("k"))
push!(chat, Message(Val(:system), "sys"))
push!(chat, Message(Val(:user), "hello"))
println("Last role: ", chat[end].role)
# This would throw — last message is not assistant:
# prefix_complete(chat)  # ArgumentError
```

---

## Capability Validation

Both `fim_complete` and `prefix_complete` check provider capabilities before making
requests. If a provider doesn't support the feature, you get a clear error:

```@example completions
# OpenAI does not support FIM
println(has_capability(OPENAIServiceEndpoint, :fim))       # false
println(has_capability(DeepSeekEndpoint("k"), :fim))        # true
println(has_capability(DeepSeekEndpoint("k"), :prefix_completion))  # true
```

## See Also

- [FIM API Reference](@ref fim_api) — full type and function reference
- [Provider Capabilities](@ref capabilities_api) — capability system
- [Multi-Backend Guide](@ref backend_guide) — all supported providers
