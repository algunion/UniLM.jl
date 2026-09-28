---
title: "UniLM.jl"
nextjs:
  metadata:
    title: "UniLM.jl"
    description: "A unified Julia interface for large language models."
---

*A unified Julia interface for large language models\.*

[![CI](https://github.com/algunion/UniLM.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/algunion/UniLM.jl/actions/workflows/CI.yml)

## What is UniLM\.jl\? {% id="What-is-UniLM.jl?" %}

| I want to… | Use | Start with |
| :--- | :--- | :--- |
| label a text with one of my own values | [`nl_classify`](/api/fixture/#UniLM.nl_classify)\(text\, TEAM\) | [Every construct](/guide/fixture/) |
| see how a docstring renders | [`push!`](</api/fixture/#Base.push!-Tuple{Chat, Message}>) | [Docstrings](/api/fixture/) |

This home page is a fixture\: the writer turns `docs/src/index.md` into a page like it\, and `npm run fixtures` copies it into place so the site builds before the manual exists\. See [anchors that start with \@](/guide/fixture/#@ref-target)\.

### Key features {% id="Key-Features" %}

- **Chat Completions** — stateful conversations with automatic history management
- **Jev \(TypeSafe System One\)** — decisions about a text instead of generated text\: `nl_classify` returns one of your own keys
