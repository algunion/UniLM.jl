---
title: "Everything"
nextjs:
  metadata:
    title: "Everything"
    description: "The guide page: emphasis, strong, code, a link and a same-page link."
---

The guide page\: *emphasis*\, **strong**\, `code`\, [a link](https://example.com) and a [same\-page link](#Repeated)\.

## Sections {% id="sections" %}

### Repeated {% id="Repeated" %}

#### Repeated {% id="Repeated-2" %}

- one
- two
  - nested
- three

1. first
2. second

- loose one

- loose two

> A quoted line\.

| Left | Centre | Default |
| :--- | :---: | ---: |
| `a\|b` | x \| y | [home](/) |

```julia
x = 1 + 1
```

```text
plain text
```

````bash
echo "```"
````

---

```julia
println("printed")
```

```output
printed
```

```julia
secret + 1
```

```output
42
```

```julia
nothing
```

{% callout type="note" %}

A note with the default title\.

{% /callout %}

{% callout type="tip" title="Custom tip" %}

A tip with its own title\.

{% /callout %}

{% callout type="warning" %}

A warning\.

{% /callout %}

{% details summary="More" %}

Hidden until opened\, with `code`\.

{% /details %}

See [`WriterFixture.greet`](/api/#Main.WriterFixture.greet) and [`WriterFixture.Greeting`](/api/#Main.WriterFixture.Greeting)\.
