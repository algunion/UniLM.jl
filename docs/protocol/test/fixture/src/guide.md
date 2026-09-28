# Everything

The guide page: *emphasis*, **strong**, `code`, [a link](https://example.com) and a
[same-page link](#Repeated).

## [Sections](@id sections)

### Repeated

#### Repeated

- one
- two
  - nested
- three

1. first
2. second

- loose one

- loose two

> A quoted line.

| Left | Centre | Default |
| :--- | :----: | ------- |
| `a\|b` | x \| y | [home](@ref fixture_home) |

```julia
x = 1 + 1
```

```
plain text
```

````bash
echo "```"
````

---

```@setup ex
secret = 41
```

```@example ex
println("printed")
```

```@example ex
secret + 1
```

```@example ex
nothing
```

!!! note
    A note with the default title.

!!! tip "Custom tip"
    A tip with its own title.

!!! warning
    A warning.

!!! details "More"
    Hidden until opened, with `code`.

See [`WriterFixture.greet`](@ref) and [`WriterFixture.Greeting`](@ref).
