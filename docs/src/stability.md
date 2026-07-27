# Versioning & Stability

UniLM follows [Semantic Versioning](https://semver.org), with the pre-1.0 conventions
below. The aim is that the version number alone tells you whether an upgrade can break
your code.

## What `0.x` means here

UniLM is pre-1.0 and under active development. While on `0.x`:

- **MINOR** releases (e.g. `0.13.0 → 0.14.0`) may include breaking changes. Every
  breaking change is listed under a **Breaking** heading in the
  [CHANGELOG](https://github.com/algunion/UniLM.jl/blob/main/CHANGELOG.md), with
  migration notes.
- **PATCH** releases (e.g. `0.14.0 → 0.14.1`) never contain breaking changes — only
  fixes, additions, and documentation.

Pin accordingly: allowing patch upgrades is always safe; allowing minor upgrades means
reading the CHANGELOG's Breaking section first.

## Breaking changes are batched

Breaking changes are grouped into infrequent minor releases rather than dribbled across
many small ones. When one lands, related changes tend to land with it, so you migrate
once per minor instead of repeatedly.

## Renames keep their old names

When an exported name changes, the old name keeps working as an alias for at least until
`1.0`. A rename is a deprecation path, not a same-day hard break.

## Toward 1.0

There is no date. `1.0` is defined by contract, not calendar — it ships when these hold:

- a **stable provider-extension contract**, so adding or maintaining a backend does not
  depend on internals that shift underneath it;
- a **unified tool API** across the chat and agentic surfaces, so a tool is defined once
  and used the same way everywhere;
- **semver-strict** guarantees from then on: after `1.0`, a breaking change requires a
  major version bump.

## The CHANGELOG is authoritative

The [CHANGELOG](https://github.com/algunion/UniLM.jl/blob/main/CHANGELOG.md) is the
authoritative migration record. If a release breaks something, the exact change and how
to migrate are documented there under that version's **Breaking** heading.
