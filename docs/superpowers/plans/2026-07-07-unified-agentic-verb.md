# Unified Agentic Verb — Implementation Plan (Plan 1: OpenAI seam migration)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Migrate the agentic verb `respond()` off its OpenAI-hardwired path onto a service-dispatched translation seam — *behavior-preserving for OpenAI* — so Plan 2 can add Gemini Interactions as one more dispatch target.

**Architecture:** Mirror the proven chat seam (`src/requests.jl:252-273`). Introduce three agentic generics — `encode_agentic` / `decode_agentic` / `decode_agentic_stream` — plus `get_url(service, ::Respond)` URL dispatch, all with untyped-`service` defaults equal to today's OpenAI logic *moved verbatim*. Route `respond()` and `_respond_stream()` through them. Nothing else changes; the existing Responses suite is the falsifier.

**Tech Stack:** Julia (latest), HTTP.jl (compat `HTTP = "1.9, 2"`), the repo JSON layer (`JSON.json` / `JSON.parse` / `JSON.lower`), `Test` + `Aqua`.

**Why Plan 1 stops at OpenAI:** Gemini's Interactions wire (request schema, `steps[]` response shape, SSE event names, `usage` keys) is *unverified against a live call* — the spec's assumption ledger and Phase 0 gate exist for exactly this. Fabricating Plan 2's golden bodies from the research summary would trip the spec's own falsifier (*"theory-laden mocks encode the same mistake twice"*). The capture gate at the end of this document unblocks Plan 2, which is written against real bytes.

## Global Constraints

*(Every task implicitly includes these — values copied verbatim from the spec.)*

- **Neutral-IR discipline:** provider-specific concepts become **neutral, typed, defaulted fields**, never raw escape-hatch dicts (precedent: `GPTToolCall.thought_signature`, `src/api.jl:85-94`).
- **Public API preserved:** `respond(input; kwargs...)` (`src/responses.jl:1184`) keeps its signature and behavior.
- **Distinct seam names:** the agentic decode generic must NOT be named `decode_response` — that collides with the chat seam's `decode_response(service, ::HTTP.Response)` (identical argument types).
- **Zero-spend testing** (⚠️ OpenAI+Anthropic keys are LIVE in the sandbox shell — a bare `Pkg.test()` bills real money): always run
  `env -u OPENAI_API_KEY -u DEEPSEEK_API_KEY -u ANTHROPIC_API_KEY -u GEMINI_API_KEY julia --project=. -e 'using Pkg; Pkg.test()'`
- **Both HTTP majors** must stay green in CI (`HTTP = "1.9, 2"`).
- **Git:** conventional-commit messages; **no** `Co-Authored-By: Claude` / "Generated with Claude Code" trailers.

## File Structure

- **Modify `src/responses.jl`** — add the agentic seam generics (OpenAI-wire defaults) in the `# ─── Request Functions ───` region (just above `respond`, `src/responses.jl:1099`); rewire `respond` (`:1122`) and `_respond_stream` (`:1029`) to call them. One file owns the agentic verb + its OpenAI seam defaults, exactly as `src/requests.jl` owns the chat verb + its defaults.
- **Modify `test/responses.jl`** — add a `@testset` proving the generics exist, return the OpenAI defaults, and are actually dispatched through by `respond`/`_respond_stream`.

No new source files in Plan 1. (`src/interactions.jl` is Plan 2, included after `gemini.jl` at `src/UniLM.jl:56`.)

---

### Task 1: Introduce the agentic seam generics + URL dispatch (OpenAI-wire defaults)

**Files:**
- Modify: `src/responses.jl` (insert at the `# ─── Request Functions ───` header, `src/responses.jl:1099`)
- Test: `test/responses.jl`

**Interfaces:**
- Consumes: `_api_base_url(service)` (`src/requests.jl:34`), `RESPONSES_PATH` (`src/constants.jl:21`), `parse_response(::HTTP.Response)::ResponseObject` (`src/responses.jl:960`), `_parse_response_stream_chunk(chunk, textbuff, failbuff, last_event)` (`src/responses.jl:977`), `Respond` (`src/responses.jl:656`), `ResponseObject` (`src/responses.jl:740`).
- Produces (Plan 2 overrides these per-service):
  - `get_url(r::Respond)::String` and `get_url(service, r::Respond)::String`
  - `encode_agentic(service, r::Respond)::String`
  - `decode_agentic(service, resp::HTTP.Response)::ResponseObject`
  - `decode_agentic_stream(service, chunk::String, textbuff::IOBuffer, failbuff::IOBuffer, last_event::Ref{String})` → `(; done::Bool, event::String, data, terminal::Symbol)`

- [ ] **Step 1: Write the failing test**

Add to `test/responses.jl`:

```julia
@testset "agentic seam — OpenAI-wire defaults" begin
    r = Respond(input="hi")                      # service defaults to OPENAIServiceEndpoint, model→"gpt-5.5"

    # URL dispatch reproduces the current _api_base_url * RESPONSES_PATH
    @test get_url(r) == "https://api.openai.com/v1/responses"
    @test get_url(OPENAIServiceEndpoint, r) == "https://api.openai.com/v1/responses"

    # encode default == today's JSON.json(r)
    @test encode_agentic(OPENAIServiceEndpoint, r) == JSON.json(r)

    # decode default delegates to parse_response
    canned = Dict("id"=>"resp_1", "status"=>"completed", "model"=>"gpt-5.5", "output"=>Any[])
    resp = HTTP.Response(200, JSON.json(canned))
    obj = decode_agentic(OPENAIServiceEndpoint, resp)
    @test obj isa ResponseObject
    @test obj.id == "resp_1"

    # stream-chunk default accumulates output_text deltas like _parse_response_stream_chunk
    textbuff = IOBuffer(); failbuff = IOBuffer(); ev = Ref("")
    chunk = "event: response.output_text.delta\ndata: {\"delta\":\"Hel\"}\n\n" *
            "event: response.output_text.delta\ndata: {\"delta\":\"lo\"}\n\n"
    st = decode_agentic_stream(OPENAIServiceEndpoint, chunk, textbuff, failbuff, ev)
    @test String(take!(textbuff)) == "Hello"
    @test st.terminal == :none
end
```

- [ ] **Step 2: Run test to verify it fails**

Run: `env -u OPENAI_API_KEY -u DEEPSEEK_API_KEY -u ANTHROPIC_API_KEY -u GEMINI_API_KEY julia --project=. -e 'using Pkg; Pkg.test()'`
Expected: FAIL — `UndefVarError: encode_agentic not defined` (the testset errors on first use of an undefined generic).

- [ ] **Step 3: Write minimal implementation**

Insert into `src/responses.jl` immediately after the `# ─── Request Functions ───` header (`src/responses.jl:1099`), before `function respond`:

```julia
# ─── Agentic wire-translation seam ───────────────────────────────────────────
# Parallel to the chat seam (src/requests.jl:252-273): three generics dispatched
# on `service` translate between the neutral Respond/ResponseObject IR and a
# provider's agentic wire. Untyped-`service` methods below are the OpenAI
# Responses defaults; a provider with a different surface (Gemini Interactions,
# Plan 2) overrides them. `respond`/`_respond_stream` call ONLY these generics,
# so retry/HTTP/cost/streaming orchestration stays provider-agnostic.
# NB: names are `*_agentic`, NOT `decode_response` — that would collide with the
# chat seam's `decode_response(service, ::HTTP.Response)` (same argument types).

get_url(r::Respond) = get_url(r.service, r)
get_url(service, r::Respond) = _api_base_url(service) * RESPONSES_PATH

encode_agentic(service, r::Respond)::String = JSON.json(r)

decode_agentic(service, resp::HTTP.Response)::ResponseObject = parse_response(resp)

decode_agentic_stream(service, chunk::String, textbuff::IOBuffer, failbuff::IOBuffer,
                      last_event::Ref{String}) =
    _parse_response_stream_chunk(chunk, textbuff, failbuff, last_event)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `env -u OPENAI_API_KEY -u DEEPSEEK_API_KEY -u ANTHROPIC_API_KEY -u GEMINI_API_KEY julia --project=. -e 'using Pkg; Pkg.test()'`
Expected: PASS — the new `@testset "agentic seam — OpenAI-wire defaults"` is green, and the full existing suite remains green.

- [ ] **Step 5: Commit**

```bash
git add src/responses.jl test/responses.jl
git commit -m "feat(agentic): add service-dispatched agentic seam with OpenAI-wire defaults"
```

---

### Task 2: Route non-streaming `respond()` through the seam

**Files:**
- Modify: `src/responses.jl:1122-1136` (`respond(r::Respond; ...)`)
- Test: `test/responses.jl`

**Interfaces:**
- Consumes: `encode_agentic`, `get_url`, `decode_agentic` (Task 1).
- Produces: no new symbols — `respond` now dispatches its wire through the seam.

- [ ] **Step 1: Write the failing test**

Add to `test/responses.jl`. The observable proof that `respond` builds its URL via `get_url` (not the old inline `_api_base_url`) is that a service whose `get_url(service, ::Respond)` throws (via `_api_base_url`) surfaces that throw through `respond` — and that the default OpenAI service does not:

```julia
@testset "respond routes URL through get_url dispatch" begin
    # Azure has no Responses surface: get_url → _api_base_url(::AZURE) throws.
    @test_throws ArgumentError respond(Respond(service=AZUREServiceEndpoint, input="x"))
    # Native Gemini likewise throws pre-Plan-2 (get_url → _api_base_url(::GEMINI) throws).
    @test_throws ArgumentError respond(Respond(service=GEMINIServiceEndpoint, input="x"))
end
```

- [ ] **Step 2: Run test to verify it fails**

Run: `env -u OPENAI_API_KEY -u DEEPSEEK_API_KEY -u ANTHROPIC_API_KEY -u GEMINI_API_KEY julia --project=. -e 'using Pkg; Pkg.test()'`
Expected: PASS-BY-ACCIDENT is possible here because the *old* code also throws (both paths call `_api_base_url`). To make this a true RED, first confirm the assertion is meaningful by temporarily checking it targets `get_url`: this test **pins** behavior for the refactor rather than driving new code. Treat Step 4's *full-suite green after the edit* as the primary gate; this testset guards against a future regression where `respond` stops consulting `get_url`.

- [ ] **Step 3: Write minimal implementation**

In `src/responses.jl`, edit the top of the `try` block in `respond` (`src/responses.jl:1125`, `:1132`, `:1136`):

```julia
    try
        body = encode_agentic(r.service, r)

        # Streaming path
        if !isnothing(r.stream) && r.stream
            return _respond_stream(r, body, callback)
        end

        url = get_url(r.service, r)
        resp = HTTP.post(url, body=body, headers=auth_header(r.service); status_exception=false)

        if resp.status == 200
            return ResponseSuccess(response=decode_agentic(r.service, resp))
```

(Only three lines change: `JSON.json(r)` → `encode_agentic(r.service, r)`; `_api_base_url(r.service) * RESPONSES_PATH` → `get_url(r.service, r)`; `parse_response(resp)` → `decode_agentic(r.service, resp)`. Leave the retry/else branches untouched.)

- [ ] **Step 4: Run test to verify it passes**

Run: `env -u OPENAI_API_KEY -u DEEPSEEK_API_KEY -u ANTHROPIC_API_KEY -u GEMINI_API_KEY julia --project=. -e 'using Pkg; Pkg.test()'`
Expected: PASS — the new testset is green **and** the full `test/responses.jl` + `test/mock_server.jl` Responses coverage is unchanged (byte-identical OpenAI wire).

- [ ] **Step 5: Commit**

```bash
git add src/responses.jl test/responses.jl
git commit -m "refactor(agentic): route respond() wire through the agentic seam"
```

---

### Task 3: Route `_respond_stream()` through the seam

**Files:**
- Modify: `src/responses.jl:1035` (stream URL) and `src/responses.jl:1048` (chunk parse)
- Test: `test/responses.jl`

**Interfaces:**
- Consumes: `get_url`, `decode_agentic_stream` (Task 1).
- Produces: no new symbols — streaming now dispatches URL + per-chunk parse through the seam. (The OpenAI-specific `ResponseObject` assembly from the `response.completed` event stays inline in `_respond_stream`; Plan 2 generalizes terminal-event assembly when Gemini's real SSE is known.)

- [ ] **Step 1: Write the failing test**

Add to `test/responses.jl` — a canned terminal-event stream through the default generic, proving it still detects `response.completed`:

```julia
@testset "agentic stream default — detects response.completed" begin
    textbuff = IOBuffer(); failbuff = IOBuffer(); ev = Ref("")
    completed = Dict("response" => Dict("id"=>"resp_9", "status"=>"completed",
                                        "model"=>"gpt-5.5", "output"=>Any[]))
    chunk = "event: response.completed\ndata: $(JSON.json(completed))\n\n"
    st = decode_agentic_stream(OPENAIServiceEndpoint, chunk, textbuff, failbuff, ev)
    @test st.terminal == :completed
    @test st.data isa AbstractDict && haskey(st.data, "response")
end
```

- [ ] **Step 2: Run test to verify it fails**

Run: `env -u OPENAI_API_KEY -u DEEPSEEK_API_KEY -u ANTHROPIC_API_KEY -u GEMINI_API_KEY julia --project=. -e 'using Pkg; Pkg.test()'`
Expected: PASS for the assertion (the Task-1 default already delegates to `_parse_response_stream_chunk`). This testset locks the streaming contract that Task 3's edit must not break; the RED for the *edit itself* is that before editing `_respond_stream`, it still calls `_parse_response_stream_chunk` directly — the wiring change is verified by Step 4's full suite plus this contract test.

- [ ] **Step 3: Write minimal implementation**

In `src/responses.jl`, inside `_respond_stream` (`src/responses.jl:1029`), change the two lines:

`src/responses.jl:1035`:
```julia
            url = get_url(r.service, r)
```

`src/responses.jl:1048`:
```julia
                    status = decode_agentic_stream(r.service, chunk, text_buffer, fail_buffer, last_event)
```

(Everything else in `_respond_stream` — the `HTTP.open` loop, the `status.terminal == :completed` assembly, callbacks — is unchanged.)

- [ ] **Step 4: Run test to verify it passes**

Run: `env -u OPENAI_API_KEY -u DEEPSEEK_API_KEY -u ANTHROPIC_API_KEY -u GEMINI_API_KEY julia --project=. -e 'using Pkg; Pkg.test()'`
Expected: PASS — new testset green, full suite green, Aqua green.

- [ ] **Step 5: Commit**

```bash
git add src/responses.jl test/responses.jl
git commit -m "refactor(agentic): route respond streaming through the agentic seam"
```

---

## Plan 1 exit gate

- Full zero-spend suite + Aqua green on the local HTTP major.
- **Verify both HTTP majors** (the streaming path is major-sensitive — `src/requests.jl:285-287`): swap and re-run —
  `julia --project=. -e 'using Pkg; Pkg.add(PackageSpec(name="HTTP", version="1"))'` then the zero-spend suite; repeat with `version="2"`. (Manifest is untracked → no git dirtiness.)
- **Falsifier:** any existing Responses/mock-server test or Aqua check that fails means the migration was **not** behavior-preserving — stop and fix before Plan 2.

---

## Prerequisite gate for Plan 2 — capture the live Interactions wire

**This is discovery, not codegen. It produces the fixtures Plan 2's goldens are built from. Do NOT write Plan 2 from the research summary.**

Requires a live, billing-enabled `GEMINI_API_KEY` (in `~/.zshrc`, absent from the sandbox). One small spend; capture once, commit the fixtures.

```bash
source ~/.zshrc                 # load GEMINI_API_KEY (billing-enabled)
mkdir -p test/fixtures/interactions
BASE="https://generativelanguage.googleapis.com/v1beta/interactions"
H_KEY="x-goog-api-key: $GEMINI_API_KEY"
H_JSON="Content-Type: application/json"
MODEL="gemini-3.1-flash-lite"   # cheapest durable witness; reconfirm still-GA first

# (1) TEXT — starting body from research; iterate on any 4xx until 200, the API
#     error names the missing/renamed field. Commit whatever body actually 200s.
curl -sS -X POST "$BASE" -H "$H_KEY" -H "$H_JSON" \
  -d "{\"model\":\"$MODEL\",\"input\":\"Say hello in exactly one word.\"}" \
  | tee test/fixtures/interactions/text_completed.json

# (2) TOOL round-trip — capture BOTH the requires_action response AND the
#     follow-up (submit function_result via previous_interaction_id). Save the
#     exact request bodies too (they are Plan 2's encoder goldens).
#     Build the tool + function_result bodies from the docs, iterate to 200.

# (3) STREAM — named SSE events; save the raw event stream verbatim.
curl -sS -N -X POST "$BASE" -H "$H_KEY" -H "$H_JSON" \
  -d "{\"model\":\"$MODEL\",\"input\":\"Count to three.\",\"stream\":true}" \
  | tee test/fixtures/interactions/text_stream.sse
```

**What to extract from the capture (feeds the spec's assumption ledger → Plan 2):**
- exact request field names/casing (`input` shape, `system_instruction`, `tools`, `generation_config`, `tool_choice.allowed_tools.mode`, `previous_interaction_id`, `store`, `background`, `stream`).
- response shape: `id`, `steps[]` member kinds (`model_output`/`function_call`/`function_result`/`thoughts`), `status` values, `usage` keys + inclusion semantics.
- SSE: event names, `delta.text` path, terminal event (`interaction.completed` vs a sentinel).
- whether server-stateful Interactions needs any `thoughtSignature`-equivalent echo (assumed NO).

```bash
git add test/fixtures/interactions
git commit -m "test(interactions): capture live Interactions wire fixtures (text/tool/stream)"
```

## Plan 2 preview (written after the gate, against real bytes)

New `src/interactions.jl` (included at `src/UniLM.jl:57`, after `gemini.jl`): `get_url(::Type{GEMINIServiceEndpoint}, ::Respond)` (Interactions URL, replacing the `_api_base_url` throw at `src/gemini.jl:19`); `encode_agentic(::Type{GEMINIServiceEndpoint}, r)` (→ Interactions body); `decode_agentic(::Type{GEMINIServiceEndpoint}, resp)` (`steps[]` → OpenAI-style `output[]` so `output_text`/`function_calls`/`reasoning_summaries` are reused); `decode_agentic_stream(::Type{GEMINIServiceEndpoint}, …)` (`interaction.*` SSE); the `:agentic` capability on OpenAI + Gemini and a `validate_capability(:agentic)` gate in `respond`; `previous_response_id`→`previous_interaction_id` mapping; `get_response`/`cancel_response` generalization for `background`; Gemini pricing rows; golden/canned unit tests from the captured fixtures + a key-gated live witness on both HTTP majors.

---

## Self-Review

**1. Spec coverage (Plan 1's slice):** Spec §"Phase 1 — dispatch the agentic verb" → Tasks 1-3. Spec §"distinct generic names (a collision forces it)" → Task 1 impl + Global Constraints. Spec §"Phase 0 — reconfirm the wire live" → the capture gate. Spec §"streaming (the known landmine) … verify on both majors" → Plan 1 exit gate. Everything else in the spec (Gemini encode/decode/stream, state-handle mapping, `:agentic` capability, background, normalization, pricing, live witness) is **explicitly deferred to Plan 2** and previewed — no silent gaps.

**2. Placeholder scan:** No "TBD/TODO/handle edge cases". The capture gate's "iterate until 200" is a genuine discovery instruction with exact commands, not a placeholder — the unknowns are wire bytes that *must not* be invented (per the spec's falsifier).

**3. Type consistency:** `encode_agentic(service, r::Respond)::String`, `decode_agentic(service, ::HTTP.Response)::ResponseObject`, `decode_agentic_stream(service, ::String, ::IOBuffer, ::IOBuffer, ::Ref{String})`, `get_url(service, ::Respond)::String` — used identically in Tasks 1, 2, 3 and the Plan 2 preview. The stream generic's arg list matches `_parse_response_stream_chunk` (`src/responses.jl:977`) exactly.

**Note on Tasks 2 & 3 TDD shape:** these are behavior-preserving refactors, so their new testsets *pin* the dispatch contract while the full-suite-green is the primary falsifier — an honest deviation from strict red-green (there is no new externally-visible behavior to drive), flagged in each task's Step 2.
