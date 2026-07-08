# v0.10.3 → main Documentation Sync + Drift Gate — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bring the Documenter site + `README.md` + `CHANGELOG.md` in sync with the surface added since v0.10.3 (breaking Gemini rename, native Anthropic/Gemini chat, cross-provider agentic verb, hosted tools, `tool_result`/`tool_choice_*`, `thought_signature`) using live build-verified examples, and add a scoped drift gate so undocumented public symbols fail the docs build.

**Architecture:** A custom coverage gate in `docs/make.jl` compares `names(UniLM)` against symbols found in `@docs` fences under `docs/src`, tolerating only an explicit `KNOWN_UNDOCUMENTED` allowlist (the debt ledger). Content tasks document new symbols (removing them from the ledger) and rewrite/add guide pages with live-guarded `@example` blocks that call the real APIs (all four provider keys are already GitHub secrets; the docs workflow is fixed to forward them).

**Tech Stack:** Julia ≥ 1.12, Documenter.jl, GitHub Actions.

## Global Constraints

- **Julia ≥ 1.12.** Docs build command: `julia --project=docs docs/make.jl` (run from repo root).
- **Real default models** (use these in NEW examples; omit `model=` to show default resolution): OpenAI chat/responses `gpt-5.5` (`capabilities.jl:55`), native Gemini `gemini-3.5-flash` (`gemini.jl:31`), native Anthropic `claude-opus-4-8` (`anthropic.jl:25`, `max_tokens` auto-defaults to 4096), image `gpt-image-2` (`capabilities.jl:69`).
- **Model-name policy:** fix only false *default-claims* (statements asserting what the default IS). Do NOT mass-rewrite the ~35 valid illustrative `model="gpt-5.2"`/`gpt-4o-mini` calls — they run fine and are out of scope (drive-by refactor).
- **Live-example policy:** single-shot calls → live-guarded `@example` blocks (real round-trip). Multi-step / stateful / infra flows (`tool_loop`, `get_response`/`cancel_response` round-trips, `tool_result` feedback, MCP servers) → non-executed ` ```julia ` fences.
- **Guard idioms (copy verbatim):**
  - Responses: `if result isa ResponseSuccess; println(output_text(result)); else; println("Request failed — ", output_text(result)); end`
  - Chat: `if result isa LLMSuccess; println(result.message.content); else; println("Request failed — see result for details"); end`
- **Message idiom:** `push!(chat, Message(Val(:user), "..."))` / `Message(Val(:system), "...")`.
- **`respond` provider selection:** `service` is a `Respond` field (`responses.jl:670`); the convenience method `respond(input; kwargs...)` forwards it, so `respond("..."; service=GEMINIServiceEndpoint)` works. There is NO `service` param on `respond(r::Respond; ...)`.
- **Gemini `tool_choice` caveat:** on `GEMINIServiceEndpoint` only `"auto"`/`"none"`/`"required"` and `tool_choice_function(...)` are valid; `tool_choice_hosted`/`_mcp`/`_custom`/`_allowed` throw `ArgumentError` (`interactions.jl:85-91`) — they are OpenAI-Responses selectors.
- **Git:** branch `docs/sync-v0.10.3-and-drift-gate` (already checked out). Conventional-commit messages. NO `Co-Authored-By: Claude` trailer, NO "Generated with Claude Code" line.
- **Drift-gate discipline:** the allowlist self-seeds (T2); every later task REMOVES the symbols it documents from `docs/undocumented_allowlist.jl`. Never hand-add entries.

---

### Task 1: Forward provider keys into the docs workflow

**Files:**
- Modify: `.github/workflows/Documentation.yml:33-36`

**Interfaces:**
- Produces: the CI environment in which every later task's live `@example` blocks actually reach Gemini/Anthropic/DeepSeek (secrets already exist: `gh secret list` shows `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, `DEEPSEEK_API_KEY`, `OPENAI_API_KEY`).

- [ ] **Step 1: Edit the `env:` block.** Replace the current block:

```yaml
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
          DOCUMENTER_KEY: ${{ secrets.DOCUMENTER_KEY }}
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
```

with (mirrors `CI.yml:17-21`):

```yaml
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
          DOCUMENTER_KEY: ${{ secrets.DOCUMENTER_KEY }}
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
          ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
          GEMINI_API_KEY: ${{ secrets.GEMINI_API_KEY }}
          DEEPSEEK_API_KEY: ${{ secrets.DEEPSEEK_API_KEY }}
```

- [ ] **Step 2: Validate YAML parses.**

Run: `julia -e 'using Pkg; Pkg.add("YAML")' 2>/dev/null; julia -e 'using YAML; YAML.load_file(".github/workflows/Documentation.yml"); println("OK")'`
Expected: `OK` (or simply eyeball: 6 keys under `env:`, correct indentation).

- [ ] **Step 3: Commit.**

```bash
git add .github/workflows/Documentation.yml
git commit -m "ci(docs): forward Anthropic/Gemini/DeepSeek keys into docs build"
```

---

### Task 2: Drift-gate machinery + seeded allowlist (TDD)

**Files:**
- Create: `docs/doc_coverage.jl`
- Create: `docs/undocumented_allowlist.jl`
- Create: `docs/test_doc_coverage.jl`
- Modify: `docs/make.jl:1-2` (includes) and after `docs/make.jl:46` (gate call)

**Interfaces:**
- Produces: `exported_names(mod)::Set{String}`, `parse_documented_symbols(dir)::Set{String}`, `missing_docs(exported, documented, allow)::Vector{String}`, `stale_allow(...)`, `resolved_allow(...)`, `assert_doc_coverage(mod, docsrc, allow)`, and `const KNOWN_UNDOCUMENTED::Set{String}`. Later tasks edit `KNOWN_UNDOCUMENTED`.

- [ ] **Step 1: Ensure the docs environment is instantiated** (one-time; safe to repeat).

Run: `julia --project=docs -e 'using Pkg; Pkg.develop(PackageSpec(path=pwd())); Pkg.instantiate()'`
Expected: resolves without error; `UniLM` available under `--project=docs`.

- [ ] **Step 2: Write the failing test** at `docs/test_doc_coverage.jl`:

```julia
include(joinpath(@__DIR__, "doc_coverage.jl"))
using Test

@testset "doc-coverage gate" begin
    exported   = Set(["Foo", "Bar", "Baz", "@mac"])
    documented = Set(["Foo", "@mac"])
    allow      = Set(["Bar"])

    @test missing_docs(exported, documented, allow) == ["Baz"]
    @test isempty(missing_docs(Set(["Foo"]), Set(["Foo"]), Set{String}()))
    @test stale_allow(exported, Set(["Bar", "Gone"])) == ["Gone"]
    @test resolved_allow(documented, Set(["Foo", "Bar"])) == ["Foo"]

    dir = mktempdir()
    write(joinpath(dir, "a.md"), """
    # Title
    ```@docs
    Foo
    UniLM.Bar
    @mac
    ```
    prose
    ```julia
    NotADocEntry
    ```
    """)
    got = parse_documented_symbols(dir)
    @test "Foo" in got
    @test "Bar" in got            # `UniLM.` prefix stripped
    @test "@mac" in got
    @test !("NotADocEntry" in got)  # plain ```julia fence ignored
end
```

- [ ] **Step 3: Run the test — verify it fails.**

Run: `julia --project=docs docs/test_doc_coverage.jl`
Expected: FAIL — `could not open file .../doc_coverage.jl` (or `UndefVarError`), because `doc_coverage.jl` does not exist yet.

- [ ] **Step 4: Implement `docs/doc_coverage.jl`:**

```julia
# Doc-coverage gate. Every EXPORTED UniLM symbol must appear in an `@docs`
# block under docs/src, OR be listed in KNOWN_UNDOCUMENTED (docs/undocumented_allowlist.jl).
# Assumes explicit `@docs` listing; there are currently no `@autodocs` blocks
# (if one is added that splices a whole module, extend this parser).

"Exported names of `mod` as strings, excluding the module name itself."
exported_names(mod::Module)::Set{String} =
    Set(string(n) for n in names(mod) if n != nameof(mod))

"Symbol names referenced inside ```@docs``` fences under `docsrc` (recursive)."
function parse_documented_symbols(docsrc::AbstractString)::Set{String}
    documented = Set{String}()
    for (root, _, files) in walkdir(docsrc)
        for f in files
            endswith(f, ".md") || continue
            indocs = false
            for line in eachline(joinpath(root, f))
                s = strip(line)
                if startswith(s, "```@docs") || startswith(s, "```@autodocs")
                    indocs = true; continue
                elseif indocs && startswith(s, "```")
                    indocs = false; continue
                end
                if indocs && !isempty(s)
                    (occursin('=', s) || occursin('[', s)) && continue  # skip @autodocs config lines
                    push!(documented, replace(s, "UniLM." => ""))
                end
            end
        end
    end
    return documented
end

missing_docs(exported::Set{String}, documented::Set{String}, allow::Set{String})::Vector{String} =
    sort(collect(setdiff(exported, documented, allow)))

stale_allow(exported::Set{String}, allow::Set{String})::Vector{String} =
    sort(collect(setdiff(allow, exported)))

resolved_allow(documented::Set{String}, allow::Set{String})::Vector{String} =
    sort(collect(intersect(documented, allow)))

"""
    assert_doc_coverage(mod, docsrc, allow)

Error (failing the build) unless every exported symbol of `mod` is documented
in an `@docs` block under `docsrc` or listed in `allow`; also errors on stale or
already-resolved allow-list entries so the ledger stays honest.
"""
function assert_doc_coverage(mod::Module, docsrc::AbstractString, allow::Set{String})
    exported   = exported_names(mod)
    documented = parse_documented_symbols(docsrc)
    problems = String[]
    miss     = missing_docs(exported, documented, allow)
    stale    = stale_allow(exported, allow)
    resolved = resolved_allow(documented, allow)
    isempty(miss)     || push!(problems, "Undocumented exported symbols (add to an @docs block or KNOWN_UNDOCUMENTED):\n  " * join(miss, "\n  "))
    isempty(stale)    || push!(problems, "KNOWN_UNDOCUMENTED lists names no longer exported (remove them):\n  " * join(stale, "\n  "))
    isempty(resolved) || push!(problems, "KNOWN_UNDOCUMENTED lists names that are now documented (remove them):\n  " * join(resolved, "\n  "))
    isempty(problems) && return nothing
    error("Doc-coverage gate failed.\n\n" * join(problems, "\n\n"))
end
```

- [ ] **Step 5: Run the test — verify it passes.**

Run: `julia --project=docs docs/test_doc_coverage.jl`
Expected: PASS — `Test Summary: | Pass N`.

- [ ] **Step 6: Seed the allowlist WITHOUT a docs build (no API spend).** Run:

```bash
julia --project=docs -e 'using UniLM; include(joinpath("docs","doc_coverage.jl")); for s in sort(collect(setdiff(exported_names(UniLM), parse_documented_symbols(joinpath("docs","src"))))); println("    \"", s, "\","); end'
```

Create `docs/undocumented_allowlist.jl` with a header comment and the printed lines pasted inside the `Set([...])`:

```julia
# Known-undocumented EXPORTED symbols — the explicit, falsifiable debt ledger.
# The doc-coverage gate (docs/doc_coverage.jl) fails when an exported symbol is
# neither in an @docs block nor listed here. Entries may only be REMOVED (by
# documenting the symbol) — never silently added. Seeded 2026-07-08.
const KNOWN_UNDOCUMENTED = Set([
    # <paste the printed "    \"Name\"," lines here>
])
```

- [ ] **Step 7: Wire the gate into `docs/make.jl`.** After line 2 (`using UniLM`) add:

```julia
include(joinpath(@__DIR__, "doc_coverage.jl"))
include(joinpath(@__DIR__, "undocumented_allowlist.jl"))
```

Immediately after the `makedocs(...)` call closes (current line 46) and before `deploydocs(` (current line 48) add:

```julia
assert_doc_coverage(UniLM, joinpath(@__DIR__, "src"), KNOWN_UNDOCUMENTED)
```

- [ ] **Step 8: Verify the gate is green against the real tree WITHOUT a build/spend.**

Run: `julia --project=docs -e 'using UniLM; include("docs/doc_coverage.jl"); include("docs/undocumented_allowlist.jl"); assert_doc_coverage(UniLM, "docs/src", KNOWN_UNDOCUMENTED); println("GATE GREEN")'`
Expected: `GATE GREEN` (allowlist exactly covers the current undocumented set).

- [ ] **Step 9: Commit.**

```bash
git add docs/doc_coverage.jl docs/undocumented_allowlist.jl docs/test_doc_coverage.jl docs/make.jl
git commit -m "docs(gate): scoped doc-coverage gate + seeded undocumented allowlist"
```

---

### Task 3: Gemini breaking rename + migration (multi_backend §Google Gemini, endpoints `@docs`)

**Files:**
- Modify: `docs/src/guide/multi_backend.md` (add a `@setup` block near top; rewrite §Google Gemini, lines 56-69; add a migration callout)
- Modify: `docs/src/api/endpoints.md:13-17` (add `GEMINIOpenAIServiceEndpoint`)
- Modify: `docs/undocumented_allowlist.jl` (remove `"GEMINIOpenAIServiceEndpoint"`)

**Interfaces:**
- Consumes: gate from Task 2. `GEMINIServiceEndpoint` docstring is already native (`api.jl:352`); `GEMINIOpenAIServiceEndpoint` has a docstring (`api.jl:349`).

- [ ] **Step 1: Add a live-example setup block** near the top of `docs/src/guide/multi_backend.md` (after the intro, before `## Available Backends` at line 7):

```markdown
```@setup multibackend
using UniLM
```
```

- [ ] **Step 2: Add a breaking-change migration callout** at the top of §Google Gemini (replacing lines 56-69). New content:

````markdown
## Google Gemini

!!! warning "Breaking change since v0.10.3"
    `GEMINIServiceEndpoint` now targets Google's **native `generateContent` API**
    (auth header `x-goog-api-key`, model in the URL, default model
    `gemini-3.5-flash`). The old **OpenAI-compatible** Gemini path is renamed
    [`GEMINIOpenAIServiceEndpoint`](@ref). Migrate code that relied on the
    OpenAI-compatible behavior — including `Embeddings(...; service=GEMINIServiceEndpoint)`,
    which the native endpoint does not support — to `GEMINIOpenAIServiceEndpoint`.

Native Gemini chat (real call, guarded so a failure never breaks the build):

```@example multibackend
gemini_chat = Chat(service=GEMINIServiceEndpoint)   # default model: gemini-3.5-flash
push!(gemini_chat, Message(Val(:user), "Say hello in one short sentence."))
result = chatrequest!(gemini_chat)
if result isa LLMSuccess
    println(result.message.content)
else
    println("Request failed — see result for details")
end
```

To keep using the OpenAI-compatible endpoint, switch the service type:

```julia
chat = Chat(service=GEMINIOpenAIServiceEndpoint, model="gemini-2.5-flash")
```
````

- [ ] **Step 3: Add `GEMINIOpenAIServiceEndpoint` to the endpoints `@docs` block** — edit `docs/src/api/endpoints.md:13-17`:

```markdown
```@docs
UniLM.OPENAIServiceEndpoint
UniLM.AZUREServiceEndpoint
UniLM.GEMINIServiceEndpoint
UniLM.GEMINIOpenAIServiceEndpoint
```
```

- [ ] **Step 4: Remove `"GEMINIOpenAIServiceEndpoint"`** from `docs/undocumented_allowlist.jl`.

- [ ] **Step 5: Build the docs and verify green (this spends on OpenAI + Gemini).** First load the Gemini key locally:

Run: `source ~/.zshrc && julia --project=docs docs/make.jl`
Expected: build completes; `GATE GREEN` (no error from `assert_doc_coverage`); the native Gemini `@example` renders a real one-sentence reply (not "Request failed").

- [ ] **Step 6: Commit.**

```bash
git add docs/src/guide/multi_backend.md docs/src/api/endpoints.md docs/undocumented_allowlist.jl
git commit -m "docs(gemini): document native generateContent rename + migration note"
```

---

### Task 4: Native Anthropic chat (multi_backend §Anthropic, endpoints `@docs`)

**Files:**
- Modify: `docs/src/guide/multi_backend.md:123-134` (replace the "compatibility layer" shim note with a native section)
- Modify: `docs/src/api/endpoints.md:13-17` (add `ANTHROPICServiceEndpoint`)
- Modify: `docs/undocumented_allowlist.jl` (remove `"ANTHROPICServiceEndpoint"`)

- [ ] **Step 1: Replace §Anthropic** (`docs/src/guide/multi_backend.md:123-134`) with a native-first section:

````markdown
### Anthropic (native Messages API)

`ANTHROPICServiceEndpoint` calls Anthropic's native `/v1/messages` API
(`x-api-key` + `anthropic-version` headers). Default model `claude-opus-4-8`;
`max_tokens` is required on the wire and defaults to 4096 if you omit it.

```@example multibackend
claude_chat = Chat(service=ANTHROPICServiceEndpoint)  # default: claude-opus-4-8
push!(claude_chat, Message(Val(:user), "Say hello in one short sentence."))
result = chatrequest!(claude_chat)
if result isa LLMSuccess
    println(result.message.content)
else
    println("Request failed — see result for details")
end
```

An OpenAI-compatible Anthropic shim is still reachable via a
`GenericOpenAIEndpoint("https://api.anthropic.com/v1", ENV["ANTHROPIC_API_KEY"])`
for evaluation, but the native endpoint above is preferred (tools, streaming,
and usage/cost accounting are supported).
````

- [ ] **Step 2: Add `ANTHROPICServiceEndpoint` to the endpoints `@docs` block** (now lines ~13-18 after Task 3):

```markdown
```@docs
UniLM.OPENAIServiceEndpoint
UniLM.AZUREServiceEndpoint
UniLM.GEMINIServiceEndpoint
UniLM.GEMINIOpenAIServiceEndpoint
UniLM.ANTHROPICServiceEndpoint
```
```

- [ ] **Step 3: Remove `"ANTHROPICServiceEndpoint"`** from `docs/undocumented_allowlist.jl`.

- [ ] **Step 4: Build and verify (spends on OpenAI + Gemini + Anthropic).**

Run: `source ~/.zshrc && julia --project=docs docs/make.jl`
Expected: green build; `GATE GREEN`; the Anthropic `@example` renders a real reply.

- [ ] **Step 5: Commit.**

```bash
git add docs/src/guide/multi_backend.md docs/src/api/endpoints.md docs/undocumented_allowlist.jl
git commit -m "docs(anthropic): document native ANTHROPICServiceEndpoint Messages API"
```

---

### Task 5: Agentic API coverage (`gemini_*` docstrings, responses `@docs` for `tool_result`/`tool_choice_*`/`gemini_*`)

**Files:**
- Modify: `src/interactions.jl:56-58` (add docstrings to the three hosted-tool constructors — REQUIRED; they have none)
- Modify: `docs/src/api/responses.md` (add two `@docs` blocks after line 114)
- Modify: `docs/undocumented_allowlist.jl` (remove the nine symbols documented here)

**Interfaces:**
- Consumes: gate from Task 2. Produces: `@docs`-covered `tool_result`, `tool_choice_function/_hosted/_allowed/_mcp/_custom`, `gemini_google_search/_code_execution/_url_context`.

- [ ] **Step 1: Add docstrings to the three hosted-tool constructors** in `src/interactions.jl`. Replace lines 56-58:

```julia
gemini_google_search()  = Dict{String,Any}("type" => "google_search")
gemini_code_execution() = Dict{String,Any}("type" => "code_execution")
gemini_url_context()    = Dict{String,Any}("type" => "url_context")
```

with:

```julia
"""
    gemini_google_search() -> Dict

Hosted Google Search tool for Gemini Interactions. Pass in `respond(...; tools=[gemini_google_search()], service=GEMINIServiceEndpoint)`.
"""
gemini_google_search()  = Dict{String,Any}("type" => "google_search")

"""
    gemini_code_execution() -> Dict

Hosted code-execution tool for Gemini Interactions. Pass in `respond(...; tools=[gemini_code_execution()], service=GEMINIServiceEndpoint)`.
"""
gemini_code_execution() = Dict{String,Any}("type" => "code_execution")

"""
    gemini_url_context() -> Dict

Hosted URL-context tool for Gemini Interactions. Pass in `respond(...; tools=[gemini_url_context()], service=GEMINIServiceEndpoint)`.
"""
gemini_url_context()    = Dict{String,Any}("type" => "url_context")
```

- [ ] **Step 2: Verify the package still loads and unit tests pass (zero-spend).**

Run: `env -u OPENAI_API_KEY -u DEEPSEEK_API_KEY -u ANTHROPIC_API_KEY -u GEMINI_API_KEY julia --project=. -e 'using Pkg; Pkg.test()'`
Expected: PASS (docstrings are inert; nothing else changed).

- [ ] **Step 3: Add two `@docs` blocks to `docs/src/api/responses.md`** immediately after the Tool Constructors block (after line 114). Insert:

```markdown

### Tool Choice

```@docs
tool_choice_function
tool_choice_hosted
tool_choice_allowed
tool_choice_mcp
tool_choice_custom
```

### Tool Results & Hosted (Gemini) Tools

```@docs
tool_result
gemini_google_search
gemini_code_execution
gemini_url_context
```
```

- [ ] **Step 4: Remove these nine names** from `docs/undocumented_allowlist.jl`: `"tool_result"`, `"tool_choice_function"`, `"tool_choice_hosted"`, `"tool_choice_allowed"`, `"tool_choice_mcp"`, `"tool_choice_custom"`, `"gemini_google_search"`, `"gemini_code_execution"`, `"gemini_url_context"`.

- [ ] **Step 5: Build and verify (spends on OpenAI + Gemini + Anthropic).**

Run: `source ~/.zshrc && julia --project=docs docs/make.jl`
Expected: green build; `GATE GREEN`; no "missing docstring" warnings for the nine symbols.

- [ ] **Step 6: Commit.**

```bash
git add src/interactions.jl docs/src/api/responses.md docs/undocumented_allowlist.jl
git commit -m "docs(agentic): docstring hosted-tool ctors + @docs tool_result/tool_choice_*"
```

---

### Task 6: New `guide/agentic.md` page (cross-provider verb) + pages entry + responses_api cross-link

**Files:**
- Create: `docs/src/guide/agentic.md`
- Modify: `docs/make.jl` (add pages entry after "Tool Calling", current line 26)
- Modify: `docs/src/guide/responses_api.md:1-5` (add a one-line pointer to the new page)

- [ ] **Step 1: Create `docs/src/guide/agentic.md`** with this content (live single-shot examples; non-executed fences for multi-step):

````markdown
# [Agentic Workflows](@id agentic_guide)

`respond` is the unified *agentic verb*. The same call targets OpenAI's Responses
API by default, or Google's Gemini Interactions API by setting
`service=GEMINIServiceEndpoint` — identical inputs, tools, lifecycle, and
usage/cost accounting.

```@setup agentic
using UniLM
```

## One call, two providers

```@example agentic
result = respond("Explain multiple dispatch in one sentence.")
if result isa ResponseSuccess
    println(output_text(result))
else
    println("Request failed — ", output_text(result))
end
```

Swap the provider with a single keyword:

```@example agentic
result = respond("Explain multiple dispatch in one sentence."; service=GEMINIServiceEndpoint)
if result isa ResponseSuccess
    println(output_text(result))
else
    println("Request failed — ", output_text(result))
end
```

## Hosted tools (Gemini)

Gemini Interactions exposes server-side hosted tools via
[`gemini_google_search`](@ref), [`gemini_code_execution`](@ref), and
[`gemini_url_context`](@ref) — pass them in `tools=`:

```@example agentic
result = respond("What are the latest stable Julia releases?";
                 service=GEMINIServiceEndpoint, tools=[gemini_google_search()])
if result isa ResponseSuccess
    println(output_text(result))
else
    println("Request failed — ", output_text(result))
end
```

## Constraining tool choice

Force a specific function with [`tool_choice_function`](@ref) (works on both
providers). The other builders — [`tool_choice_hosted`](@ref),
[`tool_choice_mcp`](@ref), [`tool_choice_custom`](@ref),
[`tool_choice_allowed`](@ref) — are OpenAI-Responses selectors and raise an
error on Gemini.

```julia
respond("What's the weather in Paris?";
        tools=[my_function_tool],
        tool_choice=tool_choice_function("get_weather"))
```

## Automated tool loop

The [`tool_loop`](@ref) driver (see the [Tool Calling guide](@ref tools_guide))
runs the call/execute/respond cycle automatically, and works across providers —
add `service=` to target Gemini:

```julia
ct = CallableTool(function_tool("get_weather", "Get weather",
        parameters=Dict("type" => "object",
                        "properties" => Dict("location" => Dict("type" => "string")))),
    (name, args) -> "22C, sunny")
result = tool_loop("What's the weather in Paris?"; service=GEMINIServiceEndpoint, tools=[ct])
# result.completed == true when the model returns a final text answer
```

## Feeding tool output back manually

Use [`tool_result`](@ref) to return a function's output on the next turn:

```julia
r2 = respond(; service=GEMINIServiceEndpoint,
             previous_response_id=r1.response.id,
             input=[tool_result("call_abc", "get_weather", "72F and sunny")])
```

## Lifecycle (background requests)

[`get_response`](@ref) and [`cancel_response`](@ref) take a `service=` keyword,
so background Gemini Interactions are managed the same way as OpenAI:

```julia
status = get_response("<interaction_id>"; service=GEMINIServiceEndpoint)
cancel_response("<interaction_id>"; service=GEMINIServiceEndpoint)
```

## Usage & cost (cross-provider)

`token_usage` and `estimated_cost` work for Gemini too — the Interactions
decoder normalizes usage into the shared shape and `DEFAULT_PRICING` includes
`gemini-3.5-flash` (hosted-tool per-call fees are not modeled). (These accessors
are intentionally referenced in plain code font, not `@ref`: their API-reference
`@docs` blocks are deferred to a later cost-tracking pass, so they remain on the
`KNOWN_UNDOCUMENTED` ledger.)

```@example agentic
r = respond("What is 2+2?"; service=GEMINIServiceEndpoint)
if r isa ResponseSuccess
    println("usage: ", token_usage(r))
    println("est. cost: \$", round(estimated_cost(r); digits=6))
else
    println("Request failed — ", output_text(r))
end
```

## See Also

- [Responses API](@ref responses_guide) — OpenAI-specific `respond`/`Respond` details
- [Tool Calling](@ref tools_guide) — defining tools and the tool loop
- [Multi-Backend](@ref backend_guide) — provider setup and capabilities
````

- [ ] **Step 2: Add the pages entry** in `docs/make.jl`. In the `"Guide" => [ ... ]` vector, after the `"Tool Calling" => "guide/tool_calling.md",` line, add:

```julia
            "Agentic Workflows" => "guide/agentic.md",
```

- [ ] **Step 3: Add a cross-link in `docs/src/guide/responses_api.md`.** After line 5 (end of the intro paragraph, before the `@setup` block at line 7), insert:

```markdown

!!! tip
    For the cross-provider view — the same `respond` verb against **Gemini
    Interactions** — see the [Agentic Workflows guide](@ref agentic_guide).
```

- [ ] **Step 4: Build and verify (spends on OpenAI + Gemini).**

Run: `source ~/.zshrc && julia --project=docs docs/make.jl`
Expected: green build; new "Agentic Workflows" page in the sidebar; the three live `@example` blocks render real output; `@ref` links resolve (no `:cross_references` errors beyond the tolerated warn).

- [ ] **Step 5: Commit.**

```bash
git add docs/src/guide/agentic.md docs/make.jl docs/src/guide/responses_api.md
git commit -m "docs(agentic): add cross-provider Agentic Workflows guide page"
```

---

### Task 7: `tool_calling.md` enrichment + `GPTToolCall.thought_signature` docstring

**Files:**
- Modify: `docs/src/guide/tool_calling.md` (append a subsection after line 210, before `## Automated Tool Loop`)
- Modify: `src/api.jl:85` (extend the `GPTToolCall` docstring to mention `thought_signature`, so `api/chat.md`'s existing `@docs GPTToolCall` renders it)

- [ ] **Step 1: Append a Responses-tools subsection** to `docs/src/guide/tool_calling.md` at line 211 (end of the "Responses API Tool Calling" section, before `## Automated Tool Loop`):

````markdown
### Tool Choice, Tool Results & Hosted Tools

Constrain which tool the model may call with the `tool_choice=` builders
([`tool_choice_function`](@ref), [`tool_choice_hosted`](@ref),
[`tool_choice_allowed`](@ref), [`tool_choice_mcp`](@ref),
[`tool_choice_custom`](@ref)):

```@example tools
r = Respond(input="What's the weather?",
            tools=[function_tool("get_weather", "Get weather",
                       parameters=Dict("type" => "object",
                                       "properties" => Dict("location" => Dict("type" => "string"))))],
            tool_choice=tool_choice_function("get_weather"))
println(r.tool_choice)
```

Return a tool's output on the next turn with [`tool_result`](@ref):

```julia
respond(; previous_response_id=r1.response.id,
        input=[tool_result("call_abc", "get_weather", "72F and sunny")])
```

Gemini Interactions adds server-side hosted tools — see the
[Agentic Workflows guide](@ref agentic_guide) for
[`gemini_google_search`](@ref) and friends. When Gemini returns tool calls, the
provider's opaque reasoning token is preserved on
[`GPTToolCall`](@ref)`.thought_signature` and echoed automatically on the next
turn.
````

(The `tools` tag already has a `@setup` block earlier in the file — verify it defines `using UniLM`; the block above only constructs objects, no network.)

- [ ] **Step 2: Extend the `GPTToolCall` docstring** in `src/api.jl` (the docstring above `@kwdef struct GPTToolCall` at line 85). Add a sentence documenting the field, e.g.:

```
Set `thought_signature` only for Gemini-3 multi-turn tool calls: the decoder
captures the provider's opaque reasoning token and the encoder echoes it back;
it is `nothing` and ignored for all other providers.
```

- [ ] **Step 3: Verify package loads + unit tests pass (zero-spend).**

Run: `env -u OPENAI_API_KEY -u DEEPSEEK_API_KEY -u ANTHROPIC_API_KEY -u GEMINI_API_KEY julia --project=. -e 'using Pkg; Pkg.test()'`
Expected: PASS.

- [ ] **Step 4: Build and verify (spends on OpenAI).**

Run: `source ~/.zshrc && julia --project=docs docs/make.jl`
Expected: green build; `GATE GREEN`; the `tools` `@example` prints the tool_choice dict; `GPTToolCall` docstring now shows the `thought_signature` note.

- [ ] **Step 5: Commit.**

```bash
git add docs/src/guide/tool_calling.md src/api.jl
git commit -m "docs(tools): document tool_choice_*/tool_result/hosted tools + thought_signature"
```

---

### Task 8: `llm.md` truthfulness (version label, false default-claims, native/agentic mentions)

**Files:**
- Modify: `docs/src/llm.md` (lines 4, 34, 102, 123, 314, 601 + add native-provider/agentic mentions)

- [ ] **Step 1: Drop the drift-prone version self-label.** `docs/src/llm.md:4` — remove the `UniLM.jl v0.9.1 · ` prefix, keeping the rest:

Change `> UniLM.jl v0.9.1 · Julia ≥ 1.12 · Deps: ...` to `> Julia ≥ 1.12 · Deps: ...`.

- [ ] **Step 2: Fix the false defaults claim** at `docs/src/llm.md:123`. Replace:

`- **Model defaults**: `"gpt-5.2"` for OpenAI, `"gemini-2.5-flash"` for Gemini, `"deepseek-chat"` for DeepSeek.`

with:

`- **Model defaults**: `"gpt-5.5"` for OpenAI, `"gemini-3.5-flash"` for native Gemini, `"claude-opus-4-8"` for native Anthropic, `"deepseek-chat"` for DeepSeek.`

- [ ] **Step 3: Fix struct-default annotations.** At `docs/src/llm.md:102` and `:314` change `model::String = "gpt-5.2"` → `model::String = "gpt-5.5"`. At `docs/src/llm.md:601` change `model::String = "gpt-image-1.5"` → `model::String = "gpt-image-2"`. At `docs/src/llm.md:34` change the image-model prose `gpt-image-1.5` → `gpt-image-2`.

- [ ] **Step 4: Add native-provider + agentic mentions.** In the service-endpoints listing, add `ANTHROPICServiceEndpoint` (native Messages), `GEMINIServiceEndpoint` (native generateContent), and `GEMINIOpenAIServiceEndpoint` (OpenAI-compat shim); and add a one-paragraph "Agentic verb across providers" note pointing to `respond(...; service=GEMINIServiceEndpoint)`, hosted tools, and cross-provider `estimated_cost`. (Keep it concise — `llm.md` is a reference index; link to `guide/agentic.md` for depth.)

- [ ] **Step 5: Build and verify (spends on OpenAI).**

Run: `source ~/.zshrc && julia --project=docs docs/make.jl`
Expected: green build; `GATE GREEN`.

- [ ] **Step 6: Commit.**

```bash
git add docs/src/llm.md
git commit -m "docs(llm): fix stale defaults/version + add native providers & agentic verb"
```

---

### Task 9: README + getting_started + index (native/agentic mentions, false-default fixes)

**Files:**
- Modify: `README.md` (Features list ~14-26; Multi-Backend ~240-266; Documentation links ~287-296)
- Modify: `docs/src/index.md:20` (image model), `docs/src/getting_started.md` (per-provider setup section)

- [ ] **Step 1: README Features** — add a native-Anthropic bullet and note native Gemini in the Multi-Backend bullet (line ~25). Keep `gpt-image-2` (line 16) — already correct.

- [ ] **Step 2: README Multi-Backend** — add a native Anthropic example and update the Gemini example to note the native default. Append after the existing examples (around line 259):

```julia
# Native Anthropic (Claude) chat
chat = Chat(service=ANTHROPICServiceEndpoint)   # default: claude-opus-4-8
```

Add an "Agentic Workflows" link under `## Documentation` (line ~287) pointing to `guide/agentic.md`.

- [ ] **Step 3: index.md image model** — `docs/src/index.md:20` change `gpt-image-1.5` → `gpt-image-2`.

- [ ] **Step 4: getting_started.md** — in the provider-setup section add a short native-Anthropic entry (`ANTHROPICServiceEndpoint`, `ANTHROPIC_API_KEY`) alongside the existing Gemini/DeepSeek entries; note native Gemini default `gemini-3.5-flash`.

- [ ] **Step 5: Build and verify (spends on OpenAI).**

Run: `source ~/.zshrc && julia --project=docs docs/make.jl`
Expected: green build; `GATE GREEN`.

- [ ] **Step 6: Commit.**

```bash
git add README.md docs/src/index.md docs/src/getting_started.md
git commit -m "docs: surface native Anthropic/Gemini + agentic verb in README & entry pages"
```

---

### Task 10: CHANGELOG `[Unreleased]` — add the agentic verb

**Files:**
- Modify: `CHANGELOG.md:9-13` (append to the `### Added` list under `[Unreleased]`)

- [ ] **Step 1: Append agentic-verb bullets** after the existing `thought_signature` line (current line 13), inside `### Added`:

```markdown
- Unified agentic verb across providers: `respond`/`Respond` now targets Google's **Gemini Interactions** API via `service=GEMINIServiceEndpoint` (in addition to OpenAI Responses), sharing inputs, `tool_loop`/`tool_loop!`, lifecycle (`get_response`/`cancel_response`), and usage/cost accounting.
- Gemini hosted-tool constructors `gemini_google_search`, `gemini_code_execution`, `gemini_url_context` for use in `respond(...; tools=[...])`.
- Cross-provider `estimated_cost`/`token_usage` for Gemini Interactions results (usage normalized to the shared shape; `gemini-3.5-flash` priced in `DEFAULT_PRICING`).
```

(Do NOT add `tool_choice_*` here — those shipped in 0.10.0 and are already released; they were only missing from the docs, now fixed in Task 5.)

- [ ] **Step 2: Verify the section reads correctly** (Breaking = Gemini rename; Added = native Gemini chat, native Anthropic chat, `thought_signature`, agentic verb, hosted tools, cross-provider cost).

- [ ] **Step 3: Commit.**

```bash
git add CHANGELOG.md
git commit -m "docs(changelog): record the cross-provider agentic verb under [Unreleased]"
```

---

### Task 11: Final verification (falsification pass)

**Files:** none (verification only).

- [ ] **Step 1: Full live build, all providers.**

Run: `source ~/.zshrc && julia --project=docs docs/make.jl`
Expected: green; `GATE GREEN`; sidebar shows "Agentic Workflows"; spot-check the rendered HTML under `docs/build/` — the Gemini and Anthropic `@example` blocks show real replies, not "Request failed".

- [ ] **Step 2: Prove the gate BITES (negative test).** Temporarily delete one still-allow-listed symbol from `docs/undocumented_allowlist.jl` (e.g. a Files/Videos symbol) and run the fast gate:

Run: `julia --project=docs -e 'using UniLM; include("docs/doc_coverage.jl"); include("docs/undocumented_allowlist.jl"); assert_doc_coverage(UniLM, "docs/src", KNOWN_UNDOCUMENTED)'`
Expected: **FAILS** with "Undocumented exported symbols (…): `<that symbol>`". Then `git checkout docs/undocumented_allowlist.jl` to revert.

- [ ] **Step 3: Confirm the DoD falsification conditions** from the spec are all satisfied:
  - `git grep -nE 'GEMINIOpenAIServiceEndpoint|ANTHROPICServiceEndpoint|tool_result|tool_choice_|gemini_google_search|gemini_code_execution|gemini_url_context' docs/undocumented_allowlist.jl` → **no output** (all new-surface symbols removed from the ledger).
  - `CHANGELOG.md [Unreleased]` contains the agentic-verb bullets.
  - No new-surface example is a plain `julia` fence where a single-shot live `@example` was possible.

- [ ] **Step 4: Push the branch and open a PR** (only after the user chooses to — see Execution Handoff). Suggested:

```bash
git push -u origin docs/sync-v0.10.3-and-drift-gate
gh pr create --title "docs: sync v0.10.3→main + scoped drift gate" --body "<summary>"
```

Note: opening the PR triggers the Documentation workflow, which now spends on all four providers per build.
