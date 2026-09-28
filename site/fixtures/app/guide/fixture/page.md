---
title: "Fixture: Every Construct"
nextjs:
  metadata:
    title: "Fixture: Every Construct"
    description: "Every construct the Documenter writer emits, in its exact output format, so the site can be built and reviewed before the manual's pages exist."
---

Every construct the Documenter writer emits\, in its exact output format\, so the site can be built and reviewed before the manual\'s pages exist\.

## Text and links {% id="Text-and-links" %}

Emphasis is *emphasis*\, strong is **strong**\, inline code is `nl_classify(text, TEAM)` and code with a backtick is `` `x` ``\.

Links\: [the home page](/)\, [a docstring on another page](/api/fixture/#UniLM.nl_classify)\, [a method docstring](</api/fixture/#Base.push!-Tuple{Chat, Message}>)\, [a section on this page](#Callouts-and-details)\, [an h3 on this page](#jev_output) and [TypeSafe\'s documentation](https://docs.typesafe.ai)\.

### Escaped text {% id="escaped-text" %}

Every ASCII punctuation character stays literal\: \! \" \# \$ \% \& \' \( \) \* \+ \, \- \. \/ \: \; \< \= \> \? \@ \[ \\ \] \^ \_ \` \{ \| \} \~

Markup that must not be interpreted\: \{\% callout \%\}\, \<b\>not HTML\<\/b\>\, \*not emphasis\*\, \_not emphasis\_\, \[not a link\]\(\/\)\, \# not a heading\.

> A block quote\, with `code`\.

1. An ordered list item
2. A second item\, with a nested list\:

   - nested bullet one
   - nested bullet two

---

## Anchors Documenter writes {% id="@ref-target" %}

Some of Documenter\'s anchors start with a digit or \"\@\"\; the page keeps them verbatim\: [step 1](#1.-Upload-the-file)\.

### 1\. Upload the file {% id="1.-Upload-the-file" %}

A numbered step\, as a guide\'s third\-level heading\.

## Code {% id="Code" %}

Julia\, with the output an example printed and returned\:

```julia
using UniLM

const TEAM = (billing   = "payments, charges, invoices or refunds",
              technical = "the app or the website does not work as expected",
              shipping  = "a parcel that is late, lost or arrived damaged",
              other     = "anything else")

nl_classify("I was charged twice for order #4471. Please refund the duplicate payment.", TEAM)
```

```output
:billing
```

### Output that wraps {% id="jev_output" %}

```julia
println("A line longer than the page wraps in the output block: ", join(1:40, " "))
"returned"
```

```output
A line longer than the page wraps in the output block: 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40
"returned"
```

A fence whose code contains Markdoc tag syntax\, and one whose code contains a triple backtick\:

```julia
prompt = "{% for item in items %}{{ item }}{% endfor %}"  # a template, not a Markdoc tag
{% callout type="warning" %}
```

````julia
md"""
```julia
x = 1
```
"""
````

Shell\, JSON\, TOML and plain text\:

```bash
export TYPESAFE_API_KEY=…
julia --project -e 'using UniLM'
```

```json
{"state": "I was charged twice.", "questions": {"classify": {"type": "choice"}}}
```

```toml
[deps]
UniLM = "2a65a866-de6b-4bc8-8daf-c130204fdf93"
```

```text
plain text keeps < > & { } as written
```

A language the highlighter does not know renders as plain text\:

```julia-repl
julia> nl_classify("Where is my parcel?", TEAM)
:shipping
```

## Callouts and details {% id="Callouts-and-details" %}

{% callout type="note" %}

A note with the default title\.

{% /callout %}

{% callout type="tip" title="Recorded answers" %}

A tip with its own title\. Its body holds `code` and a [link](/guide/fixture/#Code)\.

{% /callout %}

{% callout type="warning" title="Spend" %}

A warning\. Live examples call a paid service\.

```bash
UNILM_DOCS_LIVE=1 julia --project=docs docs/make.jl
```

{% /callout %}

{% details summary="The request Jev received" %}

A collapsed block\; its body is ordinary content\:

```json
{"model": "jev-latest", "questions": {"classify": {"type": "choice"}}}
```

{% /details %}

#### A fourth\-level heading {% id="h4-anchor" %}

Fourth\-level headings keep their ids and stay out of the table of contents\.

## Tables and images {% id="Tables-and-images" %}

| Construct | Writer output | Aligned |
| :--- | :---: | ---: |
| a pipe in a cell | a \| b | 1 |
| inline code | `x -> x^2` | 22 |
| a link | [home](/) | 333 |

![A figure copied from docs\/src\/assets](/assets/fixture.svg)
