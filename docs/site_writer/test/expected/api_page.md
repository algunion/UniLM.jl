---
title: "API"
nextjs:
  metadata:
    title: "API"
---

{% docstring id="Main.WriterFixture.greet" name="Main.WriterFixture.greet" kind="Function" %}

```julia
greet(name::String)
```

Say hello to `name`\; see [`WriterFixture.Greeting`](#Main.WriterFixture.Greeting)\.

---

```julia
greet(n::Int)
```

Say hello `n` times\.

**Returns**

A vector of greetings\.

{% /docstring %}

{% docstring id="Main.WriterFixture.Greeting" name="Main.WriterFixture.Greeting" kind="Type" %}

A greeting\.

{% /docstring %}
