---
title: "Fixture: Docstrings"
nextjs:
  metadata:
    title: "Fixture: Docstrings"
    description: "Docstrings as the writer emits them: one tag per binding, with Documenter's anchor as its id."
---

Docstrings as the writer emits them\: one tag per binding\, with Documenter\'s anchor as its id\.

## Classify {% id="Classify" %}

{% docstring id="UniLM.nl_classify" name="UniLM.nl_classify" kind="Function" %}

```julia
nl_classify(state, texts; min_confidence=0.0, decide=nothing, fallback=nothing,
            instructions=nothing, model=nothing, service=TYPESAFEServiceEndpoint,
            config=nothing, cancel=nothing, on_response=nothing)
```

Classify `state` against ONE table of `key => sentence` entries and return the chosen key as the table writes it\: `:refund` for a `NamedTuple` table\.

#### Examples

```julia
const INTENT = (refund = "the customer wants a refund",
                other  = "anything else")

nl_classify(ticket, INTENT)                                            # :refund
```

---

A second docstring of the same binding follows a horizontal rule\, as Documenter shows several docstrings of one binding\.

{% /docstring %}

## Conversations {% id="Conversations" %}

{% docstring id="Base.push!-Tuple{Chat, Message}" name="Base.push!" kind="Method" %}

```julia
push!(chat::Chat, msg::Message)
```

Add a message to the conversation\, refusing any mutation that would make it invalid\.

#### Throws

- `InvalidConversationError` when the mutation would produce an invalid conversation\.

{% /docstring %}
