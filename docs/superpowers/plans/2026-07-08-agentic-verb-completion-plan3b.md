# Complete the Gemini agentic verb (Plan 3b) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use `- [ ]`.

**Goal:** Finish the Gemini agentic surface — cost accounting, the background/get/cancel lifecycle, and native hosted tools — so the unified `respond()` verb is release-ready.

**Architecture:** Extend the neutral-IR seam the Gemini decoder already uses (`steps[]`→OpenAI-shaped `output[]`). Usage is normalized to OpenAI keys at decode; lifecycle ops dispatch a single `_agentic_url(service)` and reuse `decode_agentic`; hosted tools get flat `{type:…}` constructors + step surfacing. OpenAI paths stay behavior-preserving.

**Tech Stack:** Julia (latest), HTTP.jl (`1.9, 2`), the repo JSON layer, `Test` + `Aqua`.

## Global Constraints

- **Neutral-IR discipline:** normalize Gemini→OpenAI shape at the decode seam; downstream (`token_usage`, `estimated_cost`, `output_text`, `function_calls`) unchanged. OpenAI encoder/decoder untouched.
- **Confirmed wire (Phase-0 live 2026-07-08):** usage `total_input_tokens`/`total_output_tokens`/`total_thought_tokens`/`total_cached_tokens`/`total_tool_use_tokens` with `total = input+output+thought` (output excludes thought). `GET /v1beta/interactions/{id}` + `POST /{id}/cancel` (same subpaths as OpenAI); background create → `in_progress`+id. Hosted-tool declaration = flat `{"type":"google_search"|"code_execution"|"url_context"}`; each emits `<name>_call`+`<name>_result` steps.
- **Provider-native hosted-tool API** (user decision): `gemini_google_search()`, `gemini_code_execution()`, `gemini_url_context()`.
- **`estimated_cost` is token-based only** — hosted-tool per-call fees (e.g. google_search per-1k-queries) are not modeled; `total_tool_use_tokens` folds into output tokens as an approximation. Document on the constructors.
- **Zero-spend testing** (⚠️ OpenAI+Anthropic keys LIVE in sandbox — bare `Pkg.test()` bills): `env -u OPENAI_API_KEY -u DEEPSEEK_API_KEY -u ANTHROPIC_API_KEY -u GEMINI_API_KEY julia --project=. -e 'using Pkg; Pkg.test()'`. Live witness runs deliberately via `source ~/.zshrc` (Gemini only, cheapest model).
- **Both HTTP majors** green; hold the project's ~99.5% coverage bar. Git: conventional commits, no `Co-Authored-By`/"Generated with" trailers.

## File Structure

- **Modify `src/interactions.jl`** — `_interaction_usage` normalizer + wire into `_interaction_response_dict`/`_interaction_response_object` (`:123-139`); replace the `get_url(::GEMINI,::Respond)` override with an `_agentic_url` override (`:18`); `gemini_*` constructors; surface hosted-tool steps in `_interaction_output` (`:113`).
- **Modify `src/responses.jl`** — add `_agentic_url(service)`; route `get_url(service,::Respond)` + `get_response`/`cancel_response`/`delete_response`/`list_input_items` through it; swap `parse_response`→`decode_agentic` in `get_response`/`cancel_response`.
- **Modify `src/UniLM.jl`** — export the three `gemini_*` constructors (lifecycle ops already exported at `:153-156`).
- **Modify `test/interactions.jl`, `test/integration_interactions.jl`.**

---

### Task 1: Phase A — usage→cost normalization

**Files:** Modify `src/interactions.jl` (`:123-139` + new helper); Test `test/interactions.jl`.

**Interfaces produced:** `_interaction_usage(u)::Union{Dict{String,Any},Nothing}` — Gemini usage → OpenAI-shaped usage dict.

- [ ] **Step 1: failing test** — append to `test/interactions.jl`:
```julia
@testset "Interactions decode — usage normalized for cost accounting" begin
    raw = Dict("total_input_tokens" => 12, "total_output_tokens" => 69, "total_thought_tokens" => 5,
               "total_tool_use_tokens" => 3, "total_cached_tokens" => 4, "total_tokens" => 93)
    u = UniLM._interaction_usage(raw)
    @test u["input_tokens"] == 12
    @test u["output_tokens"] == 77                              # 69 + 5 thought + 3 tool_use (billable output)
    @test u["input_tokens_details"]["cached_tokens"] == 4
    @test u["output_tokens_details"]["reasoning_tokens"] == 5
    @test isnothing(UniLM._interaction_usage(nothing))
    # end-to-end: a decoded interaction yields correct token_usage + non-zero cost (cached=0 → clean rate identity)
    raw0 = Dict("total_input_tokens" => 12, "total_output_tokens" => 69, "total_thought_tokens" => 5,
                "total_tool_use_tokens" => 3, "total_cached_tokens" => 0, "total_tokens" => 89)
    ro = UniLM._interaction_response_object(Dict("id" => "v1_u", "status" => "completed",
        "model" => "gemini-3.1-flash-lite",
        "steps" => [Dict("type" => "model_output", "content" => [Dict("type" => "text", "text" => "hi")])],
        "usage" => raw0))
    res = ResponseSuccess(response=ro)
    @test token_usage(res).prompt_tokens == 12
    @test token_usage(res).completion_tokens == 77
    @test token_usage(res).reasoning_tokens == 5
    @test estimated_cost(res) ≈ 12 * 0.25/1_000_000 + 77 * 1.5/1_000_000   # gemini-3.1-flash-lite rates
    @test estimated_cost(res) > 0
end
```
- [ ] **Step 2: run → FAIL** (`_interaction_usage` undefined; today usage passes through raw so cost is ~0):
`env -u OPENAI_API_KEY -u DEEPSEEK_API_KEY -u ANTHROPIC_API_KEY -u GEMINI_API_KEY julia --project=. -e 'using Test, HTTP, JSON, UniLM; @testset "i" begin include("test/interactions.jl") end' 2>&1 | tail -8`
- [ ] **Step 3: implement** — in `src/interactions.jl`, add the normalizer just above `_interaction_response_dict` (~`:122`):
```julia
# Gemini Interactions usage → OpenAI-Responses-shaped usage so token_usage/estimated_cost
# work unchanged. Gemini bills thought + tool-use at the output rate, so they fold into
# billable output_tokens; reasoning_tokens breaks out the thought subset (OpenAI semantics).
# The raw usage is preserved on ResponseObject.raw. Per-call hosted-tool fees are NOT modeled.
_interaction_usage(::Nothing) = nothing
function _interaction_usage(u::AbstractDict)
    _n(k) = (v = get(u, k, 0); v isa Integer ? Int(v) : 0)
    Dict{String,Any}(
        "input_tokens"  => _n("total_input_tokens"),
        "output_tokens" => _n("total_output_tokens") + _n("total_thought_tokens") + _n("total_tool_use_tokens"),
        "total_tokens"  => _n("total_tokens"),
        "input_tokens_details"  => Dict{String,Any}("cached_tokens" => _n("total_cached_tokens")),
        "output_tokens_details" => Dict{String,Any}("reasoning_tokens" => _n("total_thought_tokens")))
end
```
Then wire it into both decoders — `_interaction_response_dict` (`:128`) and `_interaction_response_object` (`:136`): change `get(data, "usage", nothing)` to `_interaction_usage(get(data, "usage", nothing))` in **both** places.
- [ ] **Step 4: run → PASS** (same command).
- [ ] **Step 5: commit**
```bash
git add src/interactions.jl test/interactions.jl
git commit -m "feat(interactions): normalize Gemini usage → OpenAI shape so estimated_cost works"
```

---

### Task 2: Phase B — lifecycle (get/cancel) for Gemini

**Files:** Modify `src/responses.jl` (`:1139`, `:1275`, `:1301`, `:1334`, `:1369`) + `src/interactions.jl` (`:18`); Test `test/interactions.jl`.

**Interfaces produced:** `_agentic_url(service)::String` — the agentic collection URL (OpenAI `/v1/responses`, Gemini `/v1beta/interactions`), single source of truth for create + lifecycle ops.

- [ ] **Step 1: failing test** — append to `test/interactions.jl`:
```julia
@testset "Agentic lifecycle URL is service-dispatched (_agentic_url)" begin
    @test UniLM._agentic_url(GEMINIServiceEndpoint) == "https://generativelanguage.googleapis.com/v1beta/interactions"
    @test UniLM._agentic_url(OPENAIServiceEndpoint) == UniLM._api_base_url(OPENAIServiceEndpoint) * UniLM.RESPONSES_PATH
    # the create URL still resolves through the same source of truth (no divergence)
    r = Respond(service=GEMINIServiceEndpoint, input="x")
    @test UniLM.get_url(GEMINIServiceEndpoint, r) == UniLM._agentic_url(GEMINIServiceEndpoint)
end
```
- [ ] **Step 2: run → FAIL** (`_agentic_url` undefined):
`env -u OPENAI_API_KEY -u DEEPSEEK_API_KEY -u ANTHROPIC_API_KEY -u GEMINI_API_KEY julia --project=. -e 'using Test, HTTP, JSON, UniLM; @testset "i" begin include("test/interactions.jl") end' 2>&1 | tail -6`
- [ ] **Step 3: implement.**
  1. In `src/responses.jl`, replace `get_url(service, r::Respond) = _api_base_url(service) * RESPONSES_PATH` (`:1139`) with:
     ```julia
     _agentic_url(service) = _api_base_url(service) * RESPONSES_PATH
     get_url(service, r::Respond) = _agentic_url(service)
     ```
  2. In `src/interactions.jl`, replace the override (`:18`) `get_url(::Type{GEMINIServiceEndpoint}, ::Respond) = GEMINI_NATIVE_BASE * INTERACTIONS_PATH` with:
     ```julia
     _agentic_url(::Type{GEMINIServiceEndpoint}) = GEMINI_NATIVE_BASE * INTERACTIONS_PATH
     ```
  3. In `src/responses.jl`, route the four lifecycle ops through `_agentic_url` and decode via the seam. `get_response` (`:1275`, `:1278`):
     ```julia
         url = _agentic_url(service) * "/" * response_id
         resp = HTTP.get(url, headers=auth_header(service); status_exception=false)
         if resp.status == 200
             return ResponseSuccess(response=decode_agentic(service, resp))
     ```
     `cancel_response` (`:1369`, `:1372`):
     ```julia
         url = _agentic_url(service) * "/" * response_id * "/cancel"
         resp = HTTP.post(url, headers=auth_header(service); status_exception=false)
         if resp.status == 200
             return ResponseSuccess(response=decode_agentic(service, resp))
     ```
     `delete_response` (`:1301`) and `list_input_items` (`:1334`) — URL base only (they return raw dicts, leave the parse):
     ```julia
         url = _agentic_url(service) * "/" * response_id                       # delete_response
         url = _agentic_url(service) * "/" * response_id * "/input_items"       # list_input_items
     ```
- [ ] **Step 4: run → PASS** — the new test + the existing `get_url` tests (`test/interactions.jl:6-7`) + the OpenAI mock lifecycle regression:
`env -u OPENAI_API_KEY -u DEEPSEEK_API_KEY -u ANTHROPIC_API_KEY -u GEMINI_API_KEY julia --project=. -e 'using Pkg; Pkg.test()' 2>&1 | tail -4`
Expected: all pass — OpenAI `get_response`/`cancel_response` are behavior-preserving (`_agentic_url(Mock)` = `_api_base_url*RESPONSES_PATH`, `decode_agentic(Mock,·)` = `parse_response`).
- [ ] **Step 5: commit**
```bash
git add src/responses.jl src/interactions.jl test/interactions.jl
git commit -m "feat(agentic): dispatch lifecycle URL via _agentic_url + decode_agentic (Gemini get/cancel)"
```

---

### Task 3: Phase C — Gemini hosted tools (constructors + step surfacing)

**Files:** Modify `src/interactions.jl` (constructors + `_interaction_output` `:113`), `src/UniLM.jl` (`:192`); Test `test/interactions.jl`.

**Interfaces produced:** `gemini_google_search()`/`gemini_code_execution()`/`gemini_url_context()` → `Dict{String,Any}("type"=>…)`.

- [ ] **Step 1: failing tests** — append to `test/interactions.jl`:
```julia
@testset "Gemini hosted-tool constructors + encode passthrough" begin
    @test gemini_google_search()  == Dict("type" => "google_search")
    @test gemini_code_execution() == Dict("type" => "code_execution")
    @test gemini_url_context()    == Dict("type" => "url_context")
    b = JSON.parse(UniLM.encode_agentic(GEMINIServiceEndpoint,
        Respond(service=GEMINIServiceEndpoint, input="x", tools=[gemini_google_search()])); dicttype=Dict{String,Any})
    @test b["tools"][1] == Dict("type" => "google_search")
end

@testset "Interactions decode — hosted-tool steps surfaced, not dropped" begin
    data = Dict("id" => "v1_h", "status" => "completed", "model" => "gemini-3.1-flash-lite", "steps" => [
        Dict("type" => "google_search_call", "id" => "s1", "arguments" => Dict("queries" => ["x"])),
        Dict("type" => "google_search_result", "id" => "s1"),
        Dict("type" => "model_output", "content" => [Dict("type" => "text", "text" => "Answer.")])])
    ro = UniLM._interaction_response_object(data)
    types = [get(o, "type", "") for o in ro.output]
    @test "google_search_call" in types                 # surfaced
    @test "google_search_result" in types
    res = ResponseSuccess(response=ro)
    @test output_text(res) == "Answer."                 # message still decoded
    @test isempty(function_calls(res))                  # hosted step ≠ function call
end
```
- [ ] **Step 2: run → FAIL** (`gemini_google_search` undefined; hosted steps dropped so not in `types`):
`env -u OPENAI_API_KEY -u DEEPSEEK_API_KEY -u ANTHROPIC_API_KEY -u GEMINI_API_KEY julia --project=. -e 'using Test, HTTP, JSON, UniLM; @testset "i" begin include("test/interactions.jl") end' 2>&1 | tail -8`
- [ ] **Step 3: implement.**
  1. In `src/interactions.jl`, add the constructors after `_interactions_tool` (~`:52`):
```julia
# ─── Gemini native hosted tools ──────────────────────────────────────────────
# Flat {type:<name>} declarations (captured live). NOTE: estimated_cost is token-based
# and does NOT model hosted-tool per-call fees (e.g. google_search per-1k-queries).
gemini_google_search()  = Dict{String,Any}("type" => "google_search")
gemini_code_execution() = Dict{String,Any}("type" => "code_execution")
gemini_url_context()    = Dict{String,Any}("type" => "url_context")
```
  2. In `_interaction_output`, add a new `elseif` branch to the `if`-chain (before its `end`) and drop the pass-over comment. Replace this exact block (`:110-113`):
```julia
        elseif t == "thought"
            push!(out, Dict{String,Any}("type" => "reasoning", "summary" => Any[]))
        end
        # hosted-tool steps (google_search_call/…) are passed over in Plan 2.
```
with:
```julia
        elseif t == "thought"
            push!(out, Dict{String,Any}("type" => "reasoning", "summary" => Any[]))
        elseif !isempty(t)
            # hosted-tool + other steps (google_search_call/_result, code_execution_*, url_context_*,
            # …): surface them (native type + fields preserved) rather than dropping. output_text still
            # comes from model_output; function_calls() ignores them (no "function_call" type).
            push!(out, Dict{String,Any}(s))
        end
```
  3. In `src/UniLM.jl`, after the tool-constructor block (after `    custom_tool,`, `:192`) add:
```julia
    # Gemini native hosted tools
    gemini_google_search,
    gemini_code_execution,
    gemini_url_context,
```
- [ ] **Step 4: run → PASS** (same targeted command as Step 2).
- [ ] **Step 5: commit**
```bash
git add src/interactions.jl src/UniLM.jl test/interactions.jl
git commit -m "feat(interactions): Gemini hosted-tool constructors + surface hosted-tool steps"
```

---

### Task 4: live witnesses — cost, background lifecycle, hosted tool (both HTTP majors)

**Files:** Modify `test/integration_interactions.jl`.

- [ ] **Step 1: add witnesses** — inside the `if haskey(ENV, "GEMINI_API_KEY")` block of `test/integration_interactions.jl`, before the closing `end  # if GEMINI_API_KEY`:
```julia
@testset "Interactions — estimated_cost > 0 (live usage)" begin
    res = respond(Respond(service=UniLM.GEMINIServiceEndpoint, model="gemini-3.1-flash-lite",
                          input="Name three primary colors."))
    @test res isa ResponseSuccess
    @test token_usage(res).prompt_tokens > 0
    @test estimated_cost(res) > 0
end

@testset "Interactions — background create → poll → cancel (live)" begin
    started = respond(Respond(service=UniLM.GEMINIServiceEndpoint, model="gemini-3.1-flash-lite",
                              input="Write one sentence about the ocean.", background=true))
    @test started isa ResponseSuccess
    id = started.response.id
    got = get_response(id; service=UniLM.GEMINIServiceEndpoint)   # bounded single poll
    @test got isa ResponseSuccess
    @test got.response.id == id
    @test got.response.status in ("in_progress", "completed")
    cancelled = cancel_response(id; service=UniLM.GEMINIServiceEndpoint)
    @test cancelled isa ResponseSuccess || cancelled isa ResponseFailure   # 200 decode, or benign if already completed
end

@testset "Interactions — google_search grounded round-trip (live)" begin
    res = respond(Respond(service=UniLM.GEMINIServiceEndpoint, model="gemini-3.1-flash-lite",
                          input="Who won the 2022 FIFA World Cup? Search if needed.",
                          tools=[gemini_google_search()]))
    @test res isa ResponseSuccess
    @test !isempty(output_text(res))
    @test any(o -> get(o, "type", "") == "google_search_call", res.response.output)
end
```
- [ ] **Step 2: run live** (Gemini only; billed; cheapest model):
```bash
source ~/.zshrc >/dev/null 2>&1
env -u OPENAI_API_KEY -u DEEPSEEK_API_KEY -u ANTHROPIC_API_KEY julia --project=. -e 'using Test, HTTP, JSON, UniLM; @testset "witness" begin include("test/integration_interactions.jl") end' 2>&1 | tail -6`
```
Expected: PASS — real cost > 0; background id retrievable + cancellable; google_search grounds a non-empty answer with a `google_search_call` output item.
- [ ] **Step 3: commit**
```bash
git add test/integration_interactions.jl
git commit -m "test(interactions): live witnesses — cost, background lifecycle, google_search"
```

---

## Exit gate

- Full zero-spend suite + Aqua on the local HTTP major, then **both majors**: `julia --project=. -e 'using Pkg; Pkg.add(PackageSpec(name="HTTP", version="1"))'` → zero-spend suite; repeat `version="2"`; restore `version="2"`.
- **Live witnesses on both HTTP majors** (Task 4 Step 2 under HTTP 1 and HTTP 2).
- **Coverage:** confirm the new branches (`_interaction_usage` incl. `nothing`/non-dict, `_agentic_url` both services, hosted-step `else`, three constructors) are hit — hold the ~99.5% bar (CI Codecov is the gate; add unit cases if `codecov/patch` dips).
- **Falsifier:** any existing OpenAI responses/lifecycle/mock/Aqua test failing means the OpenAI path regressed — the changes must be Gemini-only or behavior-preserving (usage normalize is Gemini-decode-only; `_agentic_url`/`decode_agentic` defaults reproduce the old OpenAI URL + parser).

## Self-Review

**1. Spec coverage:** Phase A → Task 1 (usage normalize + cost); Phase B → Task 2 (`_agentic_url` + get/cancel + delete/list base); Phase C → Task 3 (constructors + step surfacing); live witnesses (A/B/C) → Task 4. `delete`/`list_input_items` Gemini support stays best-effort (correct base URL; 404→ResponseFailure) — spec-consistent. All spec IN-scope items map to a task.

**2. Placeholder scan:** none — every step has exact code + commands; all wire captured (no deferred bytes).

**3. Type consistency:** `_interaction_usage` returns `Dict{String,Any}`/`Nothing`, consumed by `_interaction_response_dict`/`_interaction_response_object` (`usage` field) and read by `_token_usage_from` via keys `input_tokens`/`output_tokens`/`input_tokens_details.cached_tokens`/`output_tokens_details.reasoning_tokens` (matches `requests.jl:100`). `_agentic_url(service)::String` used identically in `get_url` + all four lifecycle ops. `gemini_*` return `Dict{String,Any}` consumed by `_interactions_tool`'s existing `AbstractDict` passthrough. Consistent across tasks.
