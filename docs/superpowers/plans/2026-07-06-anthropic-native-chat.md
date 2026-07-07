# Native Anthropic Chat Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add native Anthropic (Claude) Messages-API chat — messages, tools, streaming, usage/cost — to UniLM.jl via a dispatched wire-translation seam, with zero duplication of the shared chat orchestration.

**Architecture:** Introduce three generic functions dispatched on `service` — `encode_request` / `decode_response` / `decode_stream_chunk` — sitting exactly where `chatrequest!`/`_chatrequeststream` currently hardcode `JSON.json(chat)` / `extract_message` / `_parse_chunk`. The untyped-`service` defaults ARE the current OpenAI-wire behavior (so DeepSeek, Azure, the Gemini OpenAI-compat shim, and `GenericOpenAIEndpoint` are unchanged); a new `ANTHROPICServiceEndpoint` overrides the three with native-Messages translation. The neutral `Chat`/`Message`/`StreamState` types are **not** changed — Anthropic's content-block structure is assembled inside the encoder and disassembled inside the decoder.

**Tech Stack:** Julia ≥ 1.12; HTTP.jl (compat `1.9, 2`); JSON.jl; Base64; SHA. Design spec: `docs/superpowers/specs/2026-07-06-anthropic-native-chat-design.md`.

## Global Constraints

- **Julia:** ≥ 1.12; use modern syntax. Every task's requirements implicitly include this section.
- **HTTP majors:** code must work under `HTTP = "1.9, 2"` (CI matrix tests both). Only `HTTP.Response(status, body)` construction and `resp.body` byte access are used here — both stable across majors.
- **Neutral IR is frozen:** do **not** modify `Message`, `Chat`, `StreamState`, `GPTToolCall`, `GPTFunction`, or `_build_stream_message`. If a keystone case cannot round-trip through the flat `Message` without loss, STOP — that falsifies the design (see spec §"Falsification") and must be raised, not worked around.
- **Verified Anthropic wire facts (against the `claude-api` reference, 2026-07-06):** `anthropic-version: 2023-06-01`; base `https://api.anthropic.com`; path `/v1/messages`; auth header `x-api-key`. Models + pricing per 1M tokens: `claude-opus-4-8` = $5 in / $25 out, `claude-sonnet-5` = $3 / $15, `claude-haiku-4-5` = $1 / $5 (cache-read input ≈ 0.1× input). Newest models (`claude-opus-4-8`, `claude-sonnet-5`) **reject** `temperature`/`top_p` with HTTP 400 — the encoder forwards them transparently when set (the provider's 400 is the loud signal), never silently drops or mangles.
- **Zero-spend test command (unset ALL provider keys):**
  ```bash
  env -u OPENAI_API_KEY -u DEEPSEEK_API_KEY -u ANTHROPIC_API_KEY julia --project=. -e 'using Pkg; Pkg.test()'
  ```
  ⚠️ Unsetting only some keys still spends. Keyed runs make live billed calls.
- **Targeted test command (single self-contained test file, no billing):**
  ```bash
  env -u OPENAI_API_KEY -u DEEPSEEK_API_KEY -u ANTHROPIC_API_KEY \
    julia --project=. -e 'using UniLM, Test, HTTP, JSON; include("test/<file>.jl")'
  ```
- **Git hygiene:** commit messages must NOT contain `Co-Authored-By: Claude` or `🤖 Generated with Claude Code` (repo owner considers them noise). All work is on branch `anthropic-native-chat`; commit locally, do **not** push (avoids triggering billed CI) unless the user asks.

---

### Task 1: Wire-translation seam + Phase-1 falsifier

Introduce the three dispatched generics with OpenAI-wire defaults, and route `chatrequest!`/`_chatrequeststream` through them. Pure refactor — zero behavior change for every existing provider.

**Files:**
- Modify: `src/requests.jl` (add generics after the retry/URL section; rewire `chatrequest!` body-build + `decode_response`; rewire `_chatrequeststream` chunk parse)
- Test: `test/requests.jl` (add byte-equality falsifier)

**Interfaces:**
- Produces (used by every later task):
  - `encode_request(service, chat::Chat) -> String`
  - `decode_response(service, resp::HTTP.Response) -> @NamedTuple{message::Message, usage::Union{TokenUsage,Nothing}}`
  - `decode_stream_chunk(service, chunk::String, state::StreamState, failbuff) -> @NamedTuple{eos::Bool}`
  - Default methods (untyped `service`) delegate to the existing `JSON.json`, `extract_message`, `_parse_chunk`.

- [ ] **Step 1: Write the failing test** — append to `test/requests.jl`:

```julia
@testset "wire seam — OpenAI defaults are byte-identical to legacy path" begin
    sig = GPTFunctionSignature(name="f", parameters=Dict("type" => "object", "properties" => Dict()))
    chat = Chat(model="gpt-5.5", tools=[GPTTool(func=sig)], temperature=0.7,
                stream=true, logit_bias=Dict("50256" => -100.0), seed=7)
    push!(chat, Message(Val(:system), "sys"))
    push!(chat, Message(Val(:user), "hi"))
    # The default (untyped-service) encoder IS the legacy OpenAI body.
    @test UniLM.encode_request(chat.service, chat) == JSON.json(chat)
    @test UniLM.encode_request(OPENAIServiceEndpoint, chat) == JSON.json(chat)
    @test UniLM.encode_request(DeepSeekEndpoint(api_key="x"), chat) == JSON.json(chat)
end
```

- [ ] **Step 2: Run test to verify it fails**

```bash
env -u OPENAI_API_KEY -u DEEPSEEK_API_KEY -u ANTHROPIC_API_KEY \
  julia --project=. -e 'using UniLM, Test, HTTP, JSON; include("test/requests.jl")'
```
Expected: FAIL — `UndefVarError: encode_request not defined` (or `MethodError`).

- [ ] **Step 3: Add the generics** — in `src/requests.jl`, insert immediately **after** the `# ─── URL Dispatch ───` block's `_api_base_url` definitions and **before** the `_accumulate_cost!` stub (around line 88):

```julia
# ─── Wire-translation seam ───────────────────────────────────────────────────
# Three generics translate between the neutral Chat/Message IR and a provider's
# wire format. The untyped-`service` methods below are the OpenAI-wire defaults —
# DeepSeek, Azure, the Gemini OpenAI-compat shim, and GenericOpenAIEndpoint all
# speak this format. Providers with a different wire format (Anthropic) override
# the three. `chatrequest!`/`_chatrequeststream` call ONLY these generics, so the
# retry/HTTP/cost/tool-loop/streaming orchestration stays provider-agnostic.

"""
    encode_request(service, chat::Chat) -> String

Serialize `chat` into the provider's request body. Default: OpenAI Chat Completions JSON.
"""
encode_request(service, chat::Chat) = JSON.json(chat)

"""
    decode_response(service, resp::HTTP.Response)

Parse a provider's 200 response into `(; message::Message, usage::Union{TokenUsage,Nothing})`.
Default: OpenAI Chat Completions (`extract_message`).
"""
decode_response(service, resp::HTTP.Response) = extract_message(resp)

"""
    decode_stream_chunk(service, chunk::String, state::StreamState, failbuff) -> (; eos::Bool)

Accumulate a streamed chunk into `state`. Default: OpenAI SSE (`_parse_chunk`).
"""
decode_stream_chunk(service, chunk::String, state::StreamState, failbuff) =
    _parse_chunk(chunk, state, failbuff)
```

Note: untyped `service` is strictly least-specific, so a later `::Type{ANTHROPICServiceEndpoint}` method wins with zero ambiguity risk.

- [ ] **Step 4: Rewire `chatrequest!`** — in `src/requests.jl`, in `function chatrequest!(chat::Chat; ...)`, change the body-build line

```julia
        body = JSON.json(chat)
```
to
```julia
        body = encode_request(chat.service, chat)
```

and change the non-streaming decode line

```julia
                extracted = extract_message(resp)
```
to
```julia
                extracted = decode_response(chat.service, resp)
```

- [ ] **Step 5: Rewire `_chatrequeststream`** — in `function _chatrequeststream(chat, body, callback=nothing; on_tool_call=nothing)`, change

```julia
                    streamstatus = _parse_chunk(chunk, state, fail_buffer)
```
to
```julia
                    streamstatus = decode_stream_chunk(chat.service, chunk, state, fail_buffer)
```

- [ ] **Step 6: Run the falsifier + targeted regression**

```bash
env -u OPENAI_API_KEY -u DEEPSEEK_API_KEY -u ANTHROPIC_API_KEY \
  julia --project=. -e 'using UniLM, Test, HTTP, JSON; include("test/api.jl"); include("test/requests.jl")'
```
Expected: PASS (byte-equality holds; serialization/streaming-parse unit tests still green).

- [ ] **Step 7: Run the mock-server chat path (exercises `chatrequest!` end-to-end through the seam)**

```bash
env -u OPENAI_API_KEY -u DEEPSEEK_API_KEY -u ANTHROPIC_API_KEY \
  julia --project=. -e 'using Pkg; Pkg.test()' 2>&1 | tail -40
```
Expected: full zero-spend suite PASSES (2815+ tests). This confirms the refactor is behavior-preserving before any Anthropic code exists. If anything fails, the seam changed behavior — fix before proceeding.

- [ ] **Step 8: Commit**

```bash
git add src/requests.jl test/requests.jl
git commit -m "feat(seam): dispatch chat encode/decode/stream on service (OpenAI-wire default)

Introduce encode_request/decode_response/decode_stream_chunk generics; route
chatrequest!/_chatrequeststream through them. Untyped-service defaults preserve
the exact OpenAI-wire path (byte-identical body), so DeepSeek/Azure/Gemini-shim/
Generic are unchanged. Phase 1 of native-provider support: behavior-preserving."
```

---

### Task 2: Anthropic endpoint scaffolding (routing, auth, capabilities, defaults, pricing)

Add the `ANTHROPICServiceEndpoint` marker and everything that does **not** yet require translation: URL, auth header, capability set, default model, `default_max_tokens`, and pricing rows.

**Files:**
- Modify: `src/constants.jl` (Anthropic constants)
- Modify: `src/api.jl` (`ANTHROPICServiceEndpoint` struct)
- Modify: `src/UniLM.jl` (export + `include("anthropic.jl")`)
- Create: `src/anthropic.jl` (routing/auth/caps/defaults)
- Modify: `src/accounting.jl` (`DEFAULT_PRICING` rows)
- Test: `test/capabilities.jl` (Anthropic capability + default-model assertions)

**Interfaces:**
- Consumes: `encode_request`/`decode_response`/`decode_stream_chunk` generics from Task 1; `ServiceEndpoint`, `Chat`, `provider_capabilities`, `default_model`, `_resolve_model`, `PriceRow`, `_price`.
- Produces (used by Tasks 3–6):
  - `struct ANTHROPICServiceEndpoint <: ServiceEndpoint end` (exported)
  - `get_url(::Type{ANTHROPICServiceEndpoint}, ::Chat)`, `auth_header(::Type{ANTHROPICServiceEndpoint})`
  - `default_model(::Type{ANTHROPICServiceEndpoint}) = "claude-opus-4-8"`
  - `default_max_tokens(::Type{ANTHROPICServiceEndpoint}, ::AbstractString) -> Int`
  - `provider_capabilities(::Type{ANTHROPICServiceEndpoint}) = Set([:chat, :tools, :json_output, :streaming])`
  - constants `ANTHROPIC_API_KEY`, `ANTHROPIC_BASE_URL`, `ANTHROPIC_MESSAGES_PATH`, `ANTHROPIC_VERSION`, `_ANTHROPIC_DEFAULT_MAX_TOKENS`

- [ ] **Step 1: Write the failing test** — append to `test/capabilities.jl`:

```julia
@testset "Anthropic — capabilities & defaults" begin
    @test has_capability(ANTHROPICServiceEndpoint, :chat)
    @test has_capability(ANTHROPICServiceEndpoint, :tools)
    @test has_capability(ANTHROPICServiceEndpoint, :streaming)
    @test !has_capability(ANTHROPICServiceEndpoint, :embeddings)
    @test UniLM.default_model(ANTHROPICServiceEndpoint) == "claude-opus-4-8"
    @test UniLM.default_max_tokens(ANTHROPICServiceEndpoint, "claude-opus-4-8") == 4096
    # A Chat with no model resolves to the Anthropic default.
    chat = Chat(service=ANTHROPICServiceEndpoint)
    @test chat.model == "claude-opus-4-8"
    @test UniLM.get_url(chat) == "https://api.anthropic.com/v1/messages"
    @test haskey(UniLM.DEFAULT_PRICING, "claude-opus-4-8")
    @test UniLM.DEFAULT_PRICING["claude-haiku-4-5"].output ≈ 5.0 / 1_000_000
end
```

- [ ] **Step 2: Run test to verify it fails**

```bash
env -u OPENAI_API_KEY -u DEEPSEEK_API_KEY -u ANTHROPIC_API_KEY \
  julia --project=. -e 'using UniLM, Test, HTTP, JSON; include("test/capabilities.jl")'
```
Expected: FAIL — `UndefVarError: ANTHROPICServiceEndpoint`.

- [ ] **Step 3: Add constants** — append to `src/constants.jl`:

```julia
# ─── Anthropic (Claude) native Messages API ──────────────────────────────────

"""Anthropic API key env var name."""
const ANTHROPIC_API_KEY::String = "ANTHROPIC_API_KEY"

"""Anthropic API base URL."""
const ANTHROPIC_BASE_URL::String = "https://api.anthropic.com"

"""Anthropic Messages API path."""
const ANTHROPIC_MESSAGES_PATH::String = "/v1/messages"

"""Required `anthropic-version` request header value (stable since 2024)."""
const ANTHROPIC_VERSION::String = "2023-06-01"

"""Moderate, overridable default for Anthropic's REQUIRED `max_tokens` when the
caller leaves it unset. Not the model ceiling — a ceiling-sized cap invites
runaway output; unused headroom is not billed. Raise `max_tokens` explicitly for
long generations."""
const _ANTHROPIC_DEFAULT_MAX_TOKENS::Int = 4096
```

- [ ] **Step 4: Add the endpoint type** — in `src/api.jl`, immediately after the `GEMINIServiceEndpoint` definition (the line `struct GEMINIServiceEndpoint <: ServiceEndpoint end`), add:

```julia
"""Anthropic (Claude) native Messages API endpoint. Requires the `ANTHROPIC_API_KEY` env variable.
Native wire format (content blocks, top-level `system`, `user`/`assistant` roles) — NOT OpenAI-compatible."""
struct ANTHROPICServiceEndpoint <: ServiceEndpoint end
```

- [ ] **Step 5: Export + include** — in `src/UniLM.jl`, add `ANTHROPICServiceEndpoint,` to the Service Endpoints export block (after `GEMINIServiceEndpoint,`), and add the include after `include("capabilities.jl")`:

```julia
include("capabilities.jl")
include("anthropic.jl")
```

- [ ] **Step 6: Create `src/anthropic.jl` with scaffolding**

```julia
# ============================================================================
# Anthropic (Claude) native Messages API
# Plugs into the wire-translation seam (encode_request / decode_response /
# decode_stream_chunk from requests.jl) so all chat orchestration is shared.
# Wire shape verified against the claude-api reference on 2026-07-06.
# ============================================================================

# ─── Routing & auth ──────────────────────────────────────────────────────────

get_url(::Type{ANTHROPICServiceEndpoint}, ::Chat) = ANTHROPIC_BASE_URL * ANTHROPIC_MESSAGES_PATH

function auth_header(::Type{ANTHROPICServiceEndpoint})
    [
        "x-api-key" => ENV[ANTHROPIC_API_KEY],
        "anthropic-version" => ANTHROPIC_VERSION,
        "Content-Type" => "application/json",
    ]
end

# ─── Capabilities & defaults ─────────────────────────────────────────────────

provider_capabilities(::Type{ANTHROPICServiceEndpoint}) =
    Set([:chat, :tools, :json_output, :streaming])

default_model(::Type{ANTHROPICServiceEndpoint}) = "claude-opus-4-8"

"""
    default_max_tokens(service, model::AbstractString) -> Int

`max_tokens` supplied when the caller leaves it unset. Anthropic *requires* the
field; OpenAI does not. Returns a moderate, overridable default (see
`_ANTHROPIC_DEFAULT_MAX_TOKENS`), not the model ceiling.
"""
default_max_tokens(::Type{ANTHROPICServiceEndpoint}, ::AbstractString) = _ANTHROPIC_DEFAULT_MAX_TOKENS
```

- [ ] **Step 7: Add pricing rows** — in `src/accounting.jl`, inside the `DEFAULT_PRICING` Dict literal, add before the closing `)` (after the O-series block):

```julia
    # Anthropic Claude (claude-api reference, 2026-07-06; cache-read input ≈ 0.1× input)
    "claude-opus-4-8"  => _price(5.0, 0.50, 25.0),
    "claude-sonnet-5"  => _price(3.0, 0.30, 15.0),
    "claude-haiku-4-5" => _price(1.0, 0.10, 5.0),
```

- [ ] **Step 8: Run test to verify it passes**

```bash
env -u OPENAI_API_KEY -u DEEPSEEK_API_KEY -u ANTHROPIC_API_KEY \
  julia --project=. -e 'using UniLM, Test, HTTP, JSON; include("test/capabilities.jl")'
```
Expected: PASS.

- [ ] **Step 9: Commit**

```bash
git add src/constants.jl src/api.jl src/UniLM.jl src/anthropic.jl src/accounting.jl test/capabilities.jl
git commit -m "feat(anthropic): endpoint scaffolding — routing, auth, capabilities, pricing

ANTHROPICServiceEndpoint marker + x-api-key/anthropic-version auth, /v1/messages
routing, {chat,tools,json_output,streaming} capabilities, claude-opus-4-8 default,
default_max_tokens, and Claude pricing rows. No translation yet."
```

---

### Task 3: `encode_request` for Anthropic (neutral Chat → Messages body)

**Files:**
- Modify: `src/anthropic.jl` (append encoder + helpers)
- Create: `test/anthropic.jl` (encode tests; self-contained `using` header)

**Interfaces:**
- Consumes: `Chat`, `Message`, `GPTTool`, `GPTToolChoice`, `GPTToolCall`, `GPTFunction`, `RoleSystem/User/Assistant/Tool`, `default_max_tokens`, constants from Task 2.
- Produces: `encode_request(::Type{ANTHROPICServiceEndpoint}, chat::Chat) -> String`.

- [ ] **Step 1: Write the failing tests** — create `test/anthropic.jl`:

```julia
# Native Anthropic translation — deterministic, zero-spend unit tests.
using UniLM
using UniLM: encode_request, decode_response, decode_stream_chunk, StreamState,
             _build_stream_message, ANTHROPICServiceEndpoint,
             RoleSystem, RoleUser, RoleAssistant, RoleTool, TOOL_CALLS, STOP
using Test, HTTP, JSON

@testset "encode — system split + user turn" begin
    chat = Chat(service=ANTHROPICServiceEndpoint, model="claude-opus-4-8")
    push!(chat, Message(Val(:system), "You are helpful."))
    push!(chat, Message(Val(:user), "Hi"))
    body = JSON.parse(encode_request(ANTHROPICServiceEndpoint, chat))
    @test body["model"] == "claude-opus-4-8"
    @test body["system"] == "You are helpful."
    @test body["max_tokens"] == 4096                     # default supplied
    @test length(body["messages"]) == 1
    @test body["messages"][1]["role"] == "user"
    @test body["messages"][1]["content"] == "Hi"
    @test !haskey(body, "temperature")                   # unset → omitted
end

@testset "encode — explicit max_tokens & stop_sequences preserved" begin
    chat = Chat(service=ANTHROPICServiceEndpoint, model="claude-opus-4-8",
                max_tokens=1000, stop=["END"])
    push!(chat, Message(Val(:system), "s"))
    push!(chat, Message(Val(:user), "u"))
    body = JSON.parse(encode_request(ANTHROPICServiceEndpoint, chat))
    @test body["max_tokens"] == 1000
    @test body["stop_sequences"] == ["END"]
end

@testset "encode — tools become {name,description,input_schema}" begin
    sig = GPTFunctionSignature(name="get_weather", description="Get weather",
        parameters=Dict("type" => "object",
                        "properties" => Dict("location" => Dict("type" => "string")),
                        "required" => ["location"]))
    chat = Chat(service=ANTHROPICServiceEndpoint, model="claude-opus-4-8",
                tools=[GPTTool(func=sig)], tool_choice="auto")
    push!(chat, Message(Val(:system), "s"))
    push!(chat, Message(Val(:user), "weather in Paris?"))
    body = JSON.parse(encode_request(ANTHROPICServiceEndpoint, chat))
    @test length(body["tools"]) == 1
    @test body["tools"][1]["name"] == "get_weather"
    @test body["tools"][1]["description"] == "Get weather"
    @test body["tools"][1]["input_schema"]["type"] == "object"
    @test !haskey(body["tools"][1], "parameters")        # renamed, not OpenAI's key
    @test body["tool_choice"] == Dict("type" => "auto")
end

@testset "encode — multi-turn tool_use → tool_result collapse" begin
    chat = Chat(service=ANTHROPICServiceEndpoint, model="claude-opus-4-8")
    push!(chat, Message(Val(:system), "s"))
    push!(chat, Message(Val(:user), "weather?"))
    tc = GPTToolCall(id="toolu_1", func=GPTFunction("get_weather", Dict("location" => "Paris")))
    push!(chat, Message(role=RoleAssistant, tool_calls=[tc], finish_reason=TOOL_CALLS))
    push!(chat, Message(role=RoleTool, tool_call_id="toolu_1", content="72F"))
    body = JSON.parse(encode_request(ANTHROPICServiceEndpoint, chat))
    msgs = body["messages"]
    @test [m["role"] for m in msgs] == ["user", "assistant", "user"]
    au = msgs[2]["content"]
    @test au[1]["type"] == "tool_use"
    @test au[1]["id"] == "toolu_1"
    @test au[1]["name"] == "get_weather"
    @test au[1]["input"] == Dict("location" => "Paris")
    tr = msgs[3]["content"]
    @test tr[1]["type"] == "tool_result"
    @test tr[1]["tool_use_id"] == "toolu_1"
    @test tr[1]["content"] == "72F"
end

@testset "encode — orphan tool_result fails loud" begin
    msgs = [Message(Val(:user), "hi"),
            Message(role=RoleTool, tool_call_id="ghost", content="x")]
    chat = Chat(service=ANTHROPICServiceEndpoint, model="claude-opus-4-8", messages=msgs)
    @test_throws ArgumentError encode_request(ANTHROPICServiceEndpoint, chat)
end
```

- [ ] **Step 2: Run to verify it fails**

```bash
env -u OPENAI_API_KEY -u DEEPSEEK_API_KEY -u ANTHROPIC_API_KEY \
  julia --project=. -e 'using UniLM, Test, HTTP, JSON; include("test/anthropic.jl")'
```
Expected: FAIL — `MethodError` (no Anthropic `encode_request` method; the untyped default returns OpenAI JSON, so `body["system"]` etc. are missing).

- [ ] **Step 3: Append the encoder + helpers to `src/anthropic.jl`**

```julia
# ─── Request encoding (neutral Chat → Anthropic Messages body) ───────────────

function encode_request(::Type{ANTHROPICServiceEndpoint}, chat::Chat)
    body = Dict{Symbol,Any}(:model => chat.model)
    # max_tokens is REQUIRED by Anthropic; fall back to the moderate default.
    body[:max_tokens] = something(chat.max_completion_tokens, chat.max_tokens,
                                  default_max_tokens(ANTHROPICServiceEndpoint, chat.model))
    system, msgs = _anthropic_messages(chat.messages)
    isnothing(system) || (body[:system] = system)
    body[:messages] = msgs
    isnothing(chat.tools)       || (body[:tools] = [_anthropic_tool(t) for t in chat.tools])
    isnothing(chat.tool_choice) || (body[:tool_choice] = _anthropic_tool_choice(chat.tool_choice))
    isnothing(chat.stop)        || (body[:stop_sequences] = chat.stop isa String ? [chat.stop] : chat.stop)
    # NB: newest Claude models reject temperature/top_p (HTTP 400). Forward
    # transparently when set — the provider's 400 is the loud signal, not a
    # silent drop or mangle.
    isnothing(chat.temperature) || (body[:temperature] = chat.temperature)
    isnothing(chat.top_p)       || (body[:top_p] = chat.top_p)
    isnothing(chat.metadata)    || (body[:metadata] = chat.metadata)
    chat.stream === true        && (body[:stream] = true)
    JSON.json(body)
end

# Split neutral messages into (system::Union{String,Nothing}, Anthropic messages).
# - system messages → concatenated top-level `system`
# - consecutive `tool` messages → collapsed into ONE user message of tool_result blocks
# - assistant tool_calls → tool_use blocks; a tool_result referencing an id no
#   preceding assistant emitted → loud ArgumentError.
function _anthropic_messages(messages)
    system = nothing
    out = Vector{Dict{Symbol,Any}}()
    seen_tool_use_ids = Set{String}()
    pending = Vector{Dict{Symbol,Any}}()
    flush!() = (isempty(pending) ||
        (push!(out, Dict{Symbol,Any}(:role => "user", :content => copy(pending))); empty!(pending)))
    for m in messages
        if m.role == RoleSystem
            system = isnothing(system) ? m.content : string(system, "\n\n", something(m.content, ""))
        elseif m.role == RoleTool
            tcid = something(m.tool_call_id, "")
            tcid in seen_tool_use_ids || throw(ArgumentError(
                "tool_result references unknown tool_use id $(repr(tcid)); no preceding assistant tool_use emitted it"))
            push!(pending, Dict{Symbol,Any}(:type => "tool_result",
                :tool_use_id => tcid, :content => something(m.content, "")))
        elseif m.role == RoleAssistant
            flush!()
            isnothing(m.tool_calls) || foreach(tc -> push!(seen_tool_use_ids, tc.id), m.tool_calls)
            push!(out, Dict{Symbol,Any}(:role => "assistant", :content => _anthropic_assistant_content(m)))
        else  # RoleUser
            flush!()
            push!(out, Dict{Symbol,Any}(:role => "user", :content => something(m.content, "")))
        end
    end
    flush!()
    (system, out)
end

# Assistant turn → Anthropic content: optional text block + tool_use blocks.
function _anthropic_assistant_content(m::Message)
    isnothing(m.tool_calls) && return something(m.content, "")
    blocks = Vector{Dict{Symbol,Any}}()
    (isnothing(m.content) || isempty(m.content)) ||
        push!(blocks, Dict{Symbol,Any}(:type => "text", :text => m.content))
    for tc in m.tool_calls
        push!(blocks, Dict{Symbol,Any}(:type => "tool_use", :id => tc.id,
            :name => tc.func.name, :input => tc.func.arguments))  # input: parsed dict → JSON object
    end
    blocks
end

function _anthropic_tool(t::GPTTool)
    f = t.func
    d = Dict{Symbol,Any}(:name => f.name,
        :input_schema => something(f.parameters, Dict("type" => "object", "properties" => Dict())))
    isnothing(f.description) || (d[:description] = f.description)
    d
end

_anthropic_tool_choice(tc::String) =
    tc == "auto"     ? Dict(:type => "auto") :
    tc == "none"     ? Dict(:type => "none") :
    tc == "required" ? Dict(:type => "any")  :
    Dict(:type => "auto")
_anthropic_tool_choice(tc::GPTToolChoice) = Dict(:type => "tool", :name => string(tc.func))
```

- [ ] **Step 4: Run to verify it passes**

```bash
env -u OPENAI_API_KEY -u DEEPSEEK_API_KEY -u ANTHROPIC_API_KEY \
  julia --project=. -e 'using UniLM, Test, HTTP, JSON; include("test/anthropic.jl")'
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/anthropic.jl test/anthropic.jl
git commit -m "feat(anthropic): encode neutral Chat → Messages body

System-split to top-level system, assistant tool_calls → tool_use blocks,
consecutive tool messages collapsed into one user tool_result turn, required
max_tokens defaulted, stop→stop_sequences, tools→input_schema. Orphan
tool_result fails loud. No IR change."
```

---

### Task 4: `decode_response` for Anthropic (Messages response → neutral Message)

**Files:**
- Modify: `src/anthropic.jl` (append decoder + usage/finish-reason helpers)
- Modify: `test/anthropic.jl` (decode tests)

**Interfaces:**
- Consumes: `HTTP.Response`, `Message`, `GPTToolCall`, `GPTFunction`, `TokenUsage`, `STOP`/`TOOL_CALLS`/`CONTENT_FILTER`, `RoleAssistant`.
- Produces: `decode_response(::Type{ANTHROPICServiceEndpoint}, resp) -> (; message, usage)`; helpers `_anthropic_finish_reason`, `_anthropic_usage`.

- [ ] **Step 1: Write the failing tests** — append to `test/anthropic.jl`:

```julia
@testset "decode — plain text" begin
    body = JSON.json(Dict("type" => "message", "role" => "assistant",
        "stop_reason" => "end_turn",
        "content" => [Dict("type" => "text", "text" => "Hello there")],
        "usage" => Dict("input_tokens" => 10, "output_tokens" => 3)))
    r = decode_response(ANTHROPICServiceEndpoint, HTTP.Response(200, body))
    @test r.message.role == RoleAssistant
    @test r.message.content == "Hello there"
    @test r.message.finish_reason == STOP
    @test r.usage.prompt_tokens == 10
    @test r.usage.completion_tokens == 3
    @test r.usage.total_tokens == 13
end

@testset "decode — text + tool_use" begin
    body = JSON.json(Dict("stop_reason" => "tool_use",
        "content" => [Dict("type" => "text", "text" => "Let me check."),
                      Dict("type" => "tool_use", "id" => "toolu_9",
                           "name" => "get_weather", "input" => Dict("location" => "Paris"))],
        "usage" => Dict("input_tokens" => 20, "output_tokens" => 15)))
    r = decode_response(ANTHROPICServiceEndpoint, HTTP.Response(200, body))
    @test r.message.finish_reason == TOOL_CALLS
    @test r.message.content == "Let me check."
    @test length(r.message.tool_calls) == 1
    @test r.message.tool_calls[1].id == "toolu_9"
    @test r.message.tool_calls[1].func.name == "get_weather"
    @test r.message.tool_calls[1].func.arguments == Dict("location" => "Paris")
end

@testset "decode — tool_use only (no text) round-trips through flat Message" begin
    body = JSON.json(Dict("stop_reason" => "tool_use",
        "content" => [Dict("type" => "tool_use", "id" => "toolu_2",
                           "name" => "f", "input" => Dict())],
        "usage" => Dict("input_tokens" => 5, "output_tokens" => 8)))
    r = decode_response(ANTHROPICServiceEndpoint, HTTP.Response(200, body))
    @test isnothing(r.message.content)
    @test r.message.tool_calls[1].id == "toolu_2"
end

@testset "decode — max_tokens → length" begin
    body = JSON.json(Dict("stop_reason" => "max_tokens",
        "content" => [Dict("type" => "text", "text" => "partial")],
        "usage" => Dict("input_tokens" => 5, "output_tokens" => 100)))
    r = decode_response(ANTHROPICServiceEndpoint, HTTP.Response(200, body))
    @test r.message.finish_reason == "length"
    @test r.message.content == "partial"
end

@testset "decode — refusal" begin
    body = JSON.json(Dict("stop_reason" => "refusal", "content" => [],
        "usage" => Dict("input_tokens" => 4, "output_tokens" => 0)))
    r = decode_response(ANTHROPICServiceEndpoint, HTTP.Response(200, body))
    @test r.message.finish_reason == UniLM.CONTENT_FILTER
    @test !isnothing(r.message.refusal_message)
end

@testset "decode — cache_read counts toward prompt_tokens (for correct billing)" begin
    body = JSON.json(Dict("stop_reason" => "end_turn",
        "content" => [Dict("type" => "text", "text" => "hi")],
        "usage" => Dict("input_tokens" => 4, "output_tokens" => 2,
                        "cache_read_input_tokens" => 100)))
    r = decode_response(ANTHROPICServiceEndpoint, HTTP.Response(200, body))
    @test r.usage.cached_tokens == 100
    @test r.usage.prompt_tokens == 104          # input + cache_read; estimated_cost bills fresh=input
end
```

- [ ] **Step 2: Run to verify it fails**

```bash
env -u OPENAI_API_KEY -u DEEPSEEK_API_KEY -u ANTHROPIC_API_KEY \
  julia --project=. -e 'using UniLM, Test, HTTP, JSON; include("test/anthropic.jl")'
```
Expected: FAIL — the untyped-default `decode_response` calls `extract_message`, which expects OpenAI's `choices[]` and errors on the Anthropic body.

- [ ] **Step 3: Append the decoder + helpers to `src/anthropic.jl`**

```julia
# ─── Response decoding (Anthropic Messages → neutral Message) ────────────────

# Anthropic stop_reason → neutral finish_reason.
function _anthropic_finish_reason(stop_reason)
    stop_reason == "end_turn"      ? STOP :
    stop_reason == "stop_sequence" ? STOP :
    stop_reason == "tool_use"      ? TOOL_CALLS :
    stop_reason == "max_tokens"    ? "length" :
    stop_reason == "refusal"       ? CONTENT_FILTER :
    something(stop_reason, STOP)
end

# Anthropic usage → neutral TokenUsage. NOTE: Anthropic `input_tokens` is the
# UNCACHED remainder; `cache_read_input_tokens` is separate. The neutral model
# treats `prompt_tokens` as TOTAL input with `cached_tokens` a subset, so add
# them — then estimated_cost bills fresh = prompt - cached = input_tokens.
# (cache_creation_input_tokens is billed at a write premium not modeled here.)
function _anthropic_usage(u)::Union{TokenUsage,Nothing}
    u isa AbstractDict || return nothing
    _i(x) = x isa Integer ? Int(x) : 0
    inp = _i(get(u, "input_tokens", 0))
    out = _i(get(u, "output_tokens", 0))
    cache_read = _i(get(u, "cache_read_input_tokens", 0))
    TokenUsage(prompt_tokens = inp + cache_read, completion_tokens = out,
        total_tokens = inp + cache_read + out, cached_tokens = cache_read, reasoning_tokens = 0)
end

function decode_response(::Type{ANTHROPICServiceEndpoint}, resp::HTTP.Response)
    data = JSON.parse(resp.body; dicttype=Dict{String,Any})
    finish = _anthropic_finish_reason(get(data, "stop_reason", nothing))
    text = IOBuffer()
    tool_calls = GPTToolCall[]
    for b in get(data, "content", [])
        bt = get(b, "type", "")
        if bt == "text"
            print(text, get(b, "text", ""))
        elseif bt == "tool_use"
            args = get(b, "input", Dict{String,Any}())
            args isa AbstractDict || (args = Dict{String,Any}())
            push!(tool_calls, GPTToolCall(id=b["id"], func=GPTFunction(b["name"], args)))
        end
        # thinking / redacted_thinking / other block types ignored (keystone)
    end
    usage = _anthropic_usage(get(data, "usage", nothing))
    txt = String(take!(text))
    msg = if !isempty(tool_calls)
        Message(role=RoleAssistant, content=(isempty(txt) ? nothing : txt),
                tool_calls=tool_calls, finish_reason=finish)
    elseif finish == CONTENT_FILTER && isempty(txt)
        Message(role=RoleAssistant, refusal_message="Model refused to respond.", finish_reason=finish)
    else
        Message(role=RoleAssistant, content=(isempty(txt) ? "No response from the model." : txt),
                finish_reason=finish)
    end
    (; message=msg, usage)
end
```

- [ ] **Step 4: Run to verify it passes**

```bash
env -u OPENAI_API_KEY -u DEEPSEEK_API_KEY -u ANTHROPIC_API_KEY \
  julia --project=. -e 'using UniLM, Test, HTTP, JSON; include("test/anthropic.jl")'
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/anthropic.jl test/anthropic.jl
git commit -m "feat(anthropic): decode Messages response → neutral Message

Text blocks → content, tool_use blocks → tool_calls, stop_reason mapped
(end_turn/stop_sequence→stop, tool_use→tool_calls, max_tokens→length,
refusal→content_filter). usage maps input+cache_read→prompt_tokens so
estimated_cost bills fresh=input_tokens."
```

---

### Task 5: `decode_stream_chunk` for Anthropic (SSE → StreamState)

Populate the existing `StreamState` from Anthropic's SSE events so `_build_stream_message` rebuilds the neutral `Message` unchanged. **No `StreamState` changes.**

**Files:**
- Modify: `src/anthropic.jl` (append stream decoder)
- Modify: `test/anthropic.jl` (canned-SSE tests)

**Interfaces:**
- Consumes: `StreamState`, `_build_stream_message`, `_anthropic_usage`, `_anthropic_finish_reason`, `TokenUsage`, `STOP`, `TOOL_CALLS`.
- Produces: `decode_stream_chunk(::Type{ANTHROPICServiceEndpoint}, chunk, state, failbuff) -> (; eos)`.

- [ ] **Step 1: Write the failing tests** — append to `test/anthropic.jl`. Build SSE `data:` lines via `JSON.json` to avoid manual escaping:

```julia
@testset "stream — text deltas + usage" begin
    lines = [
        "event: message_start",
        "data: " * JSON.json(Dict("type" => "message_start",
            "message" => Dict("usage" => Dict("input_tokens" => 8, "output_tokens" => 1)))),
        "",
        "data: " * JSON.json(Dict("type" => "content_block_start", "index" => 0,
            "content_block" => Dict("type" => "text", "text" => ""))),
        "",
        "data: " * JSON.json(Dict("type" => "content_block_delta", "index" => 0,
            "delta" => Dict("type" => "text_delta", "text" => "Hello"))),
        "",
        "data: " * JSON.json(Dict("type" => "content_block_delta", "index" => 0,
            "delta" => Dict("type" => "text_delta", "text" => " world"))),
        "",
        "data: " * JSON.json(Dict("type" => "message_delta",
            "delta" => Dict("stop_reason" => "end_turn"), "usage" => Dict("output_tokens" => 5))),
        "",
        "data: " * JSON.json(Dict("type" => "message_stop")),
    ]
    state = StreamState()
    st = decode_stream_chunk(ANTHROPICServiceEndpoint, join(lines, "\n"), state, IOBuffer())
    @test st.eos == true
    @test state.finish_reason == STOP
    @test state.usage.completion_tokens == 5
    @test state.usage.prompt_tokens == 8
    msg = _build_stream_message(state)
    @test msg.content == "Hello world"
    @test msg.finish_reason == STOP
end

@testset "stream — tool_use with input_json_delta" begin
    lines = [
        "data: " * JSON.json(Dict("type" => "message_start",
            "message" => Dict("usage" => Dict("input_tokens" => 12, "output_tokens" => 1)))),
        "",
        "data: " * JSON.json(Dict("type" => "content_block_start", "index" => 0,
            "content_block" => Dict("type" => "tool_use", "id" => "toolu_7",
                                     "name" => "get_weather", "input" => Dict()))),
        "",
        "data: " * JSON.json(Dict("type" => "content_block_delta", "index" => 0,
            "delta" => Dict("type" => "input_json_delta", "partial_json" => "{\"loc"))),
        "",
        "data: " * JSON.json(Dict("type" => "content_block_delta", "index" => 0,
            "delta" => Dict("type" => "input_json_delta", "partial_json" => "ation\":\"Paris\"}"))),
        "",
        "data: " * JSON.json(Dict("type" => "message_delta",
            "delta" => Dict("stop_reason" => "tool_use"), "usage" => Dict("output_tokens" => 20))),
        "",
        "data: " * JSON.json(Dict("type" => "message_stop")),
    ]
    state = StreamState()
    st = decode_stream_chunk(ANTHROPICServiceEndpoint, join(lines, "\n"), state, IOBuffer())
    @test st.eos == true
    @test state.finish_reason == TOOL_CALLS
    msg = _build_stream_message(state)
    @test msg.finish_reason == TOOL_CALLS
    @test length(msg.tool_calls) == 1
    @test msg.tool_calls[1].id == "toolu_7"
    @test msg.tool_calls[1].func.name == "get_weather"
    @test msg.tool_calls[1].func.arguments == Dict("location" => "Paris")
end
```

- [ ] **Step 2: Run to verify it fails**

```bash
env -u OPENAI_API_KEY -u DEEPSEEK_API_KEY -u ANTHROPIC_API_KEY \
  julia --project=. -e 'using UniLM, Test, HTTP, JSON; include("test/anthropic.jl")'
```
Expected: FAIL — the untyped-default `decode_stream_chunk` runs `_parse_chunk`, which expects OpenAI `choices[].delta` and never sets `eos` for Anthropic events.

- [ ] **Step 3: Append the stream decoder to `src/anthropic.jl`**

```julia
# ─── Streaming decode (Anthropic SSE → StreamState) ──────────────────────────
# Populates the SAME StreamState fields the OpenAI path uses, so the shared
# `_build_stream_message` rebuilds the neutral Message unchanged:
#   text_delta        → state.content
#   tool_use block    → state.tool_calls[index] = {"id","type","function"{"name","arguments"}}
#   input_json_delta  → append partial_json to that entry's "arguments"
#   stop_reason       → state.finish_reason (mapped)
#   usage             → state.usage
# EOS on `message_stop` (or an `error` event).

function decode_stream_chunk(::Type{ANTHROPICServiceEndpoint}, chunk::String, state::StreamState, failbuff)
    eos = false
    for line in filter(!isempty, strip.(split(chunk, "\n")))
        startswith(line, "data:") || continue          # skip `event:` lines, blanks, pings
        payload = strip(line[6:end])
        isempty(payload) && continue
        try
            ev = JSON.parse(payload; dicttype=Dict{String,Any})
            t = get(ev, "type", "")
            if t == "message_start"
                u = get(get(ev, "message", Dict{String,Any}()), "usage", nothing)
                u isa AbstractDict && (state.usage = _anthropic_usage(u))
            elseif t == "content_block_start"
                cb = get(ev, "content_block", Dict{String,Any}())
                if get(cb, "type", "") == "tool_use"
                    state.tool_calls[ev["index"]] = Dict{String,Any}(
                        "id" => get(cb, "id", ""), "type" => "function",
                        "function" => Dict{String,Any}("name" => get(cb, "name", ""), "arguments" => ""))
                end
            elseif t == "content_block_delta"
                idx = ev["index"]
                d = get(ev, "delta", Dict{String,Any}())
                dt = get(d, "type", "")
                if dt == "text_delta"
                    print(state.content, get(d, "text", ""))
                elseif dt == "input_json_delta" && haskey(state.tool_calls, idx)
                    state.tool_calls[idx]["function"]["arguments"] *= get(d, "partial_json", "")
                end
            elseif t == "message_delta"
                sr = get(get(ev, "delta", Dict{String,Any}()), "stop_reason", nothing)
                isnothing(sr) || (state.finish_reason = _anthropic_finish_reason(sr))
                u = get(ev, "usage", nothing)
                out = u isa AbstractDict ? get(u, "output_tokens", nothing) : nothing
                if out isa Integer && !isnothing(state.usage)
                    prev = state.usage
                    state.usage = TokenUsage(prompt_tokens=prev.prompt_tokens,
                        completion_tokens=Int(out), total_tokens=prev.prompt_tokens + Int(out),
                        cached_tokens=prev.cached_tokens, reasoning_tokens=0)
                end
            elseif t == "message_stop" || t == "error"
                eos = true
            end
        catch
            print(failbuff, line)                        # partial line — re-joined on next read
        end
    end
    (; eos)
end
```

- [ ] **Step 4: Run to verify it passes**

```bash
env -u OPENAI_API_KEY -u DEEPSEEK_API_KEY -u ANTHROPIC_API_KEY \
  julia --project=. -e 'using UniLM, Test, HTTP, JSON; include("test/anthropic.jl")'
```
Expected: PASS (all encode + decode + stream tests).

- [ ] **Step 5: Commit**

```bash
git add src/anthropic.jl test/anthropic.jl
git commit -m "feat(anthropic): decode Messages SSE stream into shared StreamState

message_start/content_block_start/content_block_delta(text_delta,input_json_delta)/
message_delta(stop_reason,usage)/message_stop mapped onto the existing StreamState
fields — _build_stream_message rebuilds the neutral Message unchanged. No IR change."
```

---

### Task 6: Wire tests into the suite, live witness, full zero-spend gate

Register the deterministic tests, add the key-gated live witness (mirrors `integration_deepseek.jl`, uses the cheapest model `claude-haiku-4-5`), and run the full zero-spend suite.

**Files:**
- Create: `test/integration_anthropic.jl` (key-gated live witness)
- Modify: `test/runtests.jl` (register `anthropic.jl` + `integration_anthropic.jl`)

**Interfaces:**
- Consumes: everything from Tasks 1–5.

- [ ] **Step 1: Create the live witness** — `test/integration_anthropic.jl`:

```julia
# ─── Anthropic Integration Tests (live) ──────────────────────────────────────
# Requires ANTHROPIC_API_KEY. Uses claude-haiku-4-5 (cheapest) to minimize spend.

if !haskey(ENV, "ANTHROPIC_API_KEY")
    @info "Skipping Anthropic integration tests (ANTHROPIC_API_KEY not set)"
else

@testset "Anthropic Chat — basic" begin
    chat = Chat(service=ANTHROPICServiceEndpoint, model="claude-haiku-4-5", max_tokens=64)
    push!(chat, Message(Val(:system), "You are a helpful assistant."))
    push!(chat, Message(Val(:user), "Reply with exactly: hello"))
    result = chatrequest!(chat)
    @test result isa LLMSuccess
    @test !isempty(result.message.content)
    @test result.usage.completion_tokens > 0
    @test cumulative_cost(chat) > 0.0
end

@testset "Anthropic Chat — tool round-trip" begin
    sig = GPTFunctionSignature(name="get_current_weather",
        description="Get the current weather for a location",
        parameters=Dict("type" => "object",
            "properties" => Dict("location" => Dict("type" => "string", "description" => "City name")),
            "required" => ["location"]))
    chat = Chat(service=ANTHROPICServiceEndpoint, model="claude-haiku-4-5", max_tokens=256,
                tools=[GPTTool(func=sig)], tool_choice="auto")
    push!(chat, Message(Val(:system), "Use the weather tool when asked about weather."))
    push!(chat, Message(Val(:user), "What is the weather in Paris?"))
    result = chatrequest!(chat)
    @test result isa LLMSuccess
    m = result.message
    if m.finish_reason == UniLM.TOOL_CALLS
        @test m.tool_calls[1].func.name == "get_current_weather"
        @test haskey(m.tool_calls[1].func.arguments, "location")
        # feed the tool result back and continue the turn
        push!(chat, Message(role=RoleTool, tool_call_id=m.tool_calls[1].id, content="72F and sunny"))
        follow = chatrequest!(chat)
        @test follow isa LLMSuccess
        @test !isempty(something(follow.message.content, ""))
    else
        @test m.finish_reason == UniLM.STOP
    end
end

@testset "Anthropic Chat — streaming" begin
    chunks = String[]
    chat = Chat(service=ANTHROPICServiceEndpoint, model="claude-haiku-4-5", max_tokens=64, stream=true)
    push!(chat, Message(Val(:system), "You are helpful."))
    push!(chat, Message(Val(:user), "Reply with exactly: hello"))
    task = chatrequest!(chat; callback=(c, _) -> c isa String && push!(chunks, c))
    result = fetch(task)
    @test result isa LLMSuccess
    @test !isempty(result.message.content)
    @test !isempty(chunks)
end

end  # if ANTHROPIC_API_KEY
```

- [ ] **Step 2: Register both test files** — in `test/runtests.jl`, add after the `@testset "capabilities" ... end` block:

```julia
    @testset "anthropic" begin
        include("anthropic.jl")
    end
```

and after the `@testset "integration — deepseek" ... end` block:

```julia
    @testset "integration — anthropic" begin
        include("integration_anthropic.jl")
    end
```

- [ ] **Step 3: Run the full zero-spend suite (definitive Phase-1+Phase-2 gate)**

```bash
env -u OPENAI_API_KEY -u DEEPSEEK_API_KEY -u ANTHROPIC_API_KEY julia --project=. -e 'using Pkg; Pkg.test()' 2>&1 | tail -60
```
Expected: PASS — prior count (2815) + the new Anthropic deterministic tests; the Anthropic (and OpenAI/DeepSeek) live testsets log "Skipping … not set". Aqua passes (`ambiguities=false`). If Aqua flags a stale export or undefined symbol, fix before committing.

- [ ] **Step 4: Commit**

```bash
git add test/integration_anthropic.jl test/runtests.jl
git commit -m "test(anthropic): register deterministic suite + key-gated live witness

test/anthropic.jl into the zero-spend suite; test/integration_anthropic.jl
(claude-haiku-4-5) covers a text call, a tool round-trip, and streaming under
ANTHROPIC_API_KEY. Full zero-spend suite green."
```

- [ ] **Step 5: Live witness (ONE keyed run — the end-to-end observation)**

Per the design's verification gate, green mocks alone do not suffice — run the witness once against the real API to confirm the translator round-trips end to end:

```bash
julia --project=. -e 'using UniLM, Test; include("test/integration_anthropic.jl")'
```
Expected: the three Anthropic testsets PASS (basic text, tool round-trip, streaming). This makes ~4 billed `claude-haiku-4-5` calls (a few tenths of a cent). Per the zero-spend memory, do NOT rerun this once green.

**If the witness fails** where the mocks passed, the canned wire shapes encode the same mistake as the translator (theory-laden). Diagnose against the live response/stream bytes (log `result` / the raw body), fix the translator, re-run the deterministic suite, then the witness once more.

---

## Self-Review (completed against the spec)

- **Spec coverage:** seam + OpenAI-default (Task 1) ✓; `ANTHROPICServiceEndpoint` + routing/auth/caps/defaults/pricing (Task 2) ✓; encode incl. system-split, tool_use→tool_result collapse, max_tokens default, stop_sequences, fail-loud orphan (Task 3) ✓; decode incl. stop_reason map, usage/cache mapping, refusal (Task 4) ✓; streaming SSE (Task 5) ✓; zero-spend deterministic tests + key-gated witness across both HTTP majors via CI matrix (Task 6) ✓. Anthropic pricing rows for `_accumulate_cost!`/`estimated_cost` ✓ (Task 2).
- **Deviations from spec (deliberate, tighter):** (a) `StreamState` gains **no** fields — Anthropic deltas populate the existing `tool_calls::Dict{Int}` and `_build_stream_message` is reused unchanged. (b) Streaming is tested by calling `decode_stream_chunk` directly on canned SSE rather than extending `test/mock_server.jl` — deterministic, no server, exercises the exact function; the live witness is the end-to-end stream check. (c) `default_max_tokens` = 4096 (moderate, documented constant), per spec §"why the max_tokens default is moderate".
- **Out of scope (own later specs, per spec §Scope):** native Gemini, embeddings, images, Anthropic Batches/Files/caching/citations, multimodal & thinking-block IR enrichment, the `GPT*`→neutral rename.
- **Assumption ledger resolved (claude-api reference, 2026-07-06):** `anthropic-version 2023-06-01`; models `claude-opus-4-8`/`claude-sonnet-5`/`claude-haiku-4-5` + pricing; SSE event/field names; tool_choice shapes; error-body shape. The one live-only assumption is that `message_stop` reliably terminates the stream — corroborated by the Task 6 witness (Step 5).
- **Type consistency:** `encode_request`/`decode_response`/`decode_stream_chunk` signatures identical across Task 1 defaults and Task 3–5 Anthropic methods; `_anthropic_usage`/`_anthropic_finish_reason` defined once (Task 4) and reused in Task 5; `default_max_tokens` signature identical in Task 2 def and Task 3 call.
