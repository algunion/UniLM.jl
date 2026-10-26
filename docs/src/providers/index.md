# [Language models](@id llm_start)

```@setup llm
using UniLM
```

A language model reads a conversation — text, and on the providers that take them images, sound clips and
PDF documents — and writes the next turn: an answer, a call of one of your functions, or JSON that fits a
schema. Use one when the output is text a person reads, or work the model does through your tools; when the
outcome is one of a set you can name and your code acts on it, a decision model answers with a probability
for each option ([Decisions](@ref decisions_start)).

## [Choose your provider](@id llm_providers)

```@eval
Main.ProviderFacts.provider_table(Main.ProviderFacts.llm)
```

## [Concepts](@id llm_concepts)

- [Conversations with Chat](@ref chat_guide) — a `Chat` holds a conversation and `chatrequest!` sends it,
  turn by turn.
- [Agentic respond](@ref agentic_guide) — `respond` sends one request of a provider's agentic API and returns
  its output items.
- [Tool Calling](@ref tools_guide) — functions the model asks to call, and the tool loops that run the round
  trips.
- [Streaming](@ref streaming_guide) — the reply handed to your callback as it is generated.
- [Structured Output](@ref structured_guide) — replies in JSON, or JSON that follows your schema.
- [Attachments](@ref attachments_guide) — images, sound clips and PDF documents on a user message.
- [Embeddings](@ref embeddings_guide) — vectors that stand for inputs, to search, group and compare them.
- [Image Generation](@ref images_guide) — images made from a prompt, and images edited.
- [FIM & Prefix Completion](@ref completions_guide) — text written between a prefix and a suffix, and a reply
  continued from the start you give it.
- [Results](@ref results_guide) — the typed success, failure or call error every call returns.
- [Cost Tracking](@ref cost_guide) — token usage in one shape on every provider, and its price.
- [Timeouts & Retries](@ref timeouts_guide) — the bounds every call waits under, the retries they bound, and
  the typed errors a breach gives.
- [Concurrency, Tasks and Cancellation](@ref concurrency_guide) — many calls at once, what they may share, and
  stopping one.
- [Testing and recordings](@ref testing_guide) — a call recorded once and replayed, with no key and no
  network.
- [Model Context Protocol (MCP)](@ref mcp_guide) — MCP servers' tools in a tool loop, and a server of your own.

## [Across providers](@id llm_across)

- [One API across providers](@ref backend_guide) — what every provider shares, how to switch, and where the
  providers differ.
- [Capabilities by provider](@ref capabilities_guide) — what each provider's endpoints serve.
- [Write your own backend](@ref custom_backends_guide) — a provider UniLM does not ship, on the seams the
  built-in ones use.
