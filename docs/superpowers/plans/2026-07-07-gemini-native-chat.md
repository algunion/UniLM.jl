# Gemini Native Chat Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a native Google Gemini `generateContent` chat provider (messages + tools + streaming + usage/cost) by riding the existing dispatched wire-translation seam.

**Architecture:** A new self-contained `src/gemini.jl` mirrors `src/anthropic.jl`: it overrides `get_url`/`auth_header`/`provider_capabilities`/`default_model` and the three seam generics (`encode_request`/`decode_response`/`decode_stream_chunk`) for a new native `GEMINIServiceEndpoint` singleton. The neutral orchestration (`chatrequest!`, `tool_loop!`, `StreamState`, cost) is reused untouched. The pre-existing OpenAI-compat shim is renamed `GEMINIOpenAIServiceEndpoint`, freeing the canonical name for native. Exactly one optional field (`thought_signature`) is added to the neutral `GPTToolCall`.

**Tech Stack:** Julia (latest), HTTP.jl (`1.9, 2` compat — CI tests both), JSON.jl. Gemini REST `v1beta` `generateContent`/`streamGenerateContent`.

## Global Constraints

- **Modern Julia** — latest syntax/constructs.
- **Zero-spend tests** — always `env -u OPENAI_API_KEY -u DEEPSEEK_API_KEY -u ANTHROPIC_API_KEY -u GEMINI_API_KEY julia --project=. -e 'using Pkg; Pkg.test()'`. OpenAI + Anthropic keys are LIVE in the sandbox shell; a bare `Pkg.test()` bills them. Gemini's key lives only in `~/.zshrc` and is **billing-enabled** (not free tier).
- **Verify both HTTP majors** in CI (`HTTP = "1.9, 2"`); local is 2.x.
- **Git hygiene** — NO self-attribution trailers (`Co-Authored-By`, "Generated with…") on commits. Conventional-commit subjects (`feat(gemini):`, `refactor(gemini):`, `test(gemini):`).
- **Neutral IR** — the ONLY permitted change is the one additive optional `GPTToolCall.thought_signature::Union{Nothing,String}=nothing`.
- **Naming** — native singleton = `GEMINIServiceEndpoint`; OpenAI-compat shim = `GEMINIOpenAIServiceEndpoint` (breaking; note under `## Breaking changes` at release).
- **Load order** — `src/gemini.jl` is `include`d after `requests.jl`/`capabilities.jl` so the `state::StreamState` annotation resolves at definition time.
- **Wire facts to re-confirm against live docs while coding** (assumption ledger in the spec): `usageMetadata` inclusion semantics (does `promptTokenCount` include cached? does `candidatesTokenCount` include thoughts?), exact `thoughtSignature` placement on the part, whether streamed `functionCall` parts fragment, pricing numbers.

---

### Task 1: Rename OpenAI-compat shim → `GEMINIOpenAIServiceEndpoint` (+ bump deprecated default)

Pure mechanical rename across src + test, behavior-preserving except the deprecated `default_model` bump (`gemini-2.5-flash` retires 2026-10-16). Carries the breaking surface alone. Gated by the full existing suite staying green.

**Files:**
- Modify: `src/api.jl:327,339` · `src/UniLM.jl:81` · `src/requests.jl:29,32,36,76` · `src/capabilities.jl:24,57,63`
- Modify (tests): `test/requests.jl:25,26,33,34,35,48,49,101` · `test/capabilities.jl:13,14,110,117,121,122,136,141` · `test/files.jl:40` · `test/api.jl:417,853,854` · `test/mock_server.jl:465`

**Interfaces:**
- Produces: `GEMINIOpenAIServiceEndpoint <: ServiceEndpoint` (was `GEMINIServiceEndpoint`), exported; `default_model(::Type{GEMINIOpenAIServiceEndpoint}) == "gemini-3.5-flash"`.

- [ ] **Step 1: Rename the type everywhere `GEMINIServiceEndpoint` currently appears (src + test).**

Global identifier rename `GEMINIServiceEndpoint` → `GEMINIOpenAIServiceEndpoint` in exactly these files (the constants `GEMINI_CHAT_URL`, `GEMINI_OPENAI_BASE`, `GEMINI_API_KEY` are NOT renamed):

```bash
grep -rl 'GEMINIServiceEndpoint' src test --include='*.jl' \
  | xargs sed -i '' 's/GEMINIServiceEndpoint/GEMINIOpenAIServiceEndpoint/g'
```

Then fix the `src/api.jl` docstring line (now reads `GEMINIOpenAIServiceEndpoint`) to clarify:

```julia
- `GEMINIOpenAIServiceEndpoint` — Google Gemini via OpenAI-compatible endpoint (see `GEMINIServiceEndpoint` for native)
```

- [ ] **Step 2: Bump the deprecated default model (src + its test assertion).**

`src/capabilities.jl:57`:

```julia
default_model(::Type{GEMINIOpenAIServiceEndpoint})  = "gemini-3.5-flash"
```

`test/capabilities.jl:136`:

```julia
    @test UniLM.default_model(GEMINIOpenAIServiceEndpoint) == "gemini-3.5-flash"
```

- [ ] **Step 3: Run the full zero-spend suite — expect GREEN (behavior preserved).**

Run:
```bash
env -u OPENAI_API_KEY -u DEEPSEEK_API_KEY -u ANTHROPIC_API_KEY -u GEMINI_API_KEY \
  julia --project=. -e 'using Pkg; Pkg.test()'
```
Expected: PASS (2886 tests + Aqua). If any test still names `GEMINIServiceEndpoint`, the rename missed a site — grep again: `grep -rn 'GEMINIServiceEndpoint' src test`. It must return nothing after this task.

- [ ] **Step 4: Commit.**

```bash
git add src test
git commit -m "refactor(gemini): rename OpenAI-compat shim → GEMINIOpenAIServiceEndpoint; bump default 2.5→3.5"
```

---

### Task 2: Add `thought_signature` to the neutral `GPTToolCall`

The one neutral-IR change. Gemini-3 `functionCall` parts carry an opaque `thoughtSignature` that must be echoed verbatim next turn or stateless multi-turn tool calls 400.

**Files:**
- Modify: `src/api.jl:80-86` (the `GPTToolCall` struct + `JSON.lower`)
- Test: `test/api.jl` (append a testset)

**Interfaces:**
- Produces: `GPTToolCall(; id, type="function", func, thought_signature=nothing)` with `thought_signature::Union{Nothing,String}`. `JSON.lower(::GPTToolCall)` still emits only `{id,type,function}`.

- [ ] **Step 1: Write the failing test.**

Append to `test/api.jl`:

```julia
@testset "GPTToolCall.thought_signature (Gemini-3 opaque echo)" begin
    tc = GPTToolCall(id="fc_1", func=GPTFunction("f", Dict("x" => 1)))
    @test isnothing(tc.thought_signature)                      # optional, defaults nothing
    tc2 = GPTToolCall(id="fc_2", func=GPTFunction("f", Dict()), thought_signature="SIG")
    @test tc2.thought_signature == "SIG"
    # MUST NOT leak into OpenAI wire serialization:
    lowered = JSON.lower(tc2)
    @test !haskey(lowered, :thoughtSignature) && !haskey(lowered, :thought_signature)
    @test Set(keys(lowered)) == Set([:id, :type, :function])
end
```

- [ ] **Step 2: Run it — expect FAIL.**

Run: `env -u OPENAI_API_KEY -u DEEPSEEK_API_KEY -u ANTHROPIC_API_KEY -u GEMINI_API_KEY julia --project=. -e 'using UniLM, Test, JSON; include("test/api.jl")'`
Expected: FAIL (`GPTToolCall` has no `thought_signature` keyword).

- [ ] **Step 3: Add the field.**

`src/api.jl`, the `GPTToolCall` struct (currently lines 80-84):

```julia
@kwdef struct GPTToolCall
    id::String
    type::String = "function"
    func::GPTFunction
    # Gemini-3 opaque function-calling signature; MUST be echoed verbatim on the
    # next turn or stateless multi-turn tool calls 400. Set only by the Gemini
    # decoder; ignored (nothing) by every other provider. Deliberately absent
    # from JSON.lower below, so OpenAI-wire serialization is byte-identical.
    thought_signature::Union{Nothing,String} = nothing
end
```

`JSON.lower` (line 86) stays exactly as-is — it already lists fields explicitly:

```julia
JSON.lower(x::GPTToolCall) = Dict(:id => x.id, :type => x.type, :function => x.func)
```

- [ ] **Step 4: Audit positional construction (0.10.3 arity lesson).**

`@kwdef` defaults do NOT extend positional constructors. Confirm nothing constructs `GPTToolCall` positionally:

Run: `grep -rn 'GPTToolCall(' src test --include='*.jl' | grep -v 'id='`
Expected: no matches (every call site uses keywords). If a positional call exists, add an explicit old-arity constructor below the struct (mirroring `GPTFunctionSignature`, `api.jl:41`): `GPTToolCall(id, type, func) = GPTToolCall(id, type, func, nothing)`.

- [ ] **Step 5: Run the test — expect PASS.**

Run: `env -u OPENAI_API_KEY -u DEEPSEEK_API_KEY -u ANTHROPIC_API_KEY -u GEMINI_API_KEY julia --project=. -e 'using UniLM, Test, JSON; include("test/api.jl")'`
Expected: PASS.

- [ ] **Step 6: Commit.**

```bash
git add src/api.jl test/api.jl
git commit -m "feat(gemini): add optional GPTToolCall.thought_signature for Gemini-3 echo"
```

---

### Task 3: Native endpoint scaffolding — type, constants, routing, auth, capabilities

Re-add `GEMINIServiceEndpoint` as the native singleton; wire URL (model-in-URL, stream-branch), auth (`x-goog-api-key`), capabilities/default. Create `test/gemini.jl` and register it.

**Files:**
- Create: `src/gemini.jl`
- Modify: `src/constants.jl` (add native base URL, after line 118) · `src/api.jl` (re-add struct + docstring, near line 339) · `src/UniLM.jl` (include after `anthropic.jl`; export the native name)
- Create/Test: `test/gemini.jl`; Modify `test/runtests.jl` (register)

**Interfaces:**
- Produces: `GEMINIServiceEndpoint <: ServiceEndpoint` (exported); `GEMINI_NATIVE_BASE`; `get_url(::Type{GEMINIServiceEndpoint}, ::Chat)`; `auth_header(::Type{GEMINIServiceEndpoint})`; `provider_capabilities(::Type{GEMINIServiceEndpoint}) == Set([:chat,:tools,:streaming])`; `default_model(::Type{GEMINIServiceEndpoint}) == "gemini-3.5-flash"`.
- Consumes: the `ServiceEndpoint` abstract type; `chat.stream`, `chat.model` fields.

- [ ] **Step 1: Add the native base-URL constant.**

`src/constants.jl`, after line 118 (`GEMINI_CHAT_URL`):

```julia
# ─── Gemini (native generateContent) ─────────────────────────────────────────

"""Google Gemini native API base (model + method appended: `/models/{model}:generateContent`)."""
const GEMINI_NATIVE_BASE::String = "https://generativelanguage.googleapis.com/v1beta"
```

- [ ] **Step 2: Re-add the native singleton type + docstring.**

`src/api.jl`, immediately after the (renamed) shim struct near line 339:

```julia
"""Native Google Gemini `generateContent` API (`x-goog-api-key`; model in URL). Requires `GEMINI_API_KEY`."""
struct GEMINIServiceEndpoint <: ServiceEndpoint end
```

- [ ] **Step 3: Create `src/gemini.jl` with routing, auth, capabilities, default.**

```julia
# ============================================================================
# Google Gemini native generateContent API
# Plugs into the wire-translation seam (encode_request / decode_response /
# decode_stream_chunk from requests.jl) so all chat orchestration is shared.
# Wire shape verified against ai.google.dev live docs on 2026-07-07.
# ============================================================================

# ─── Routing & auth ──────────────────────────────────────────────────────────
# Model is in the URL (like Azure); streaming is the URL METHOD, not a body flag.

function get_url(::Type{GEMINIServiceEndpoint}, chat::Chat)
    if chat.stream === true
        "$(GEMINI_NATIVE_BASE)/models/$(chat.model):streamGenerateContent?alt=sse"
    else
        "$(GEMINI_NATIVE_BASE)/models/$(chat.model):generateContent"
    end
end

_api_base_url(::Type{GEMINIServiceEndpoint}) =
    throw(ArgumentError("Responses API is only supported with OPENAIServiceEndpoint"))

auth_header(::Type{GEMINIServiceEndpoint}) = [
    "x-goog-api-key" => ENV[GEMINI_API_KEY],
    "Content-Type"   => "application/json",
]

# ─── Capabilities & defaults ─────────────────────────────────────────────────

provider_capabilities(::Type{GEMINIServiceEndpoint}) = Set([:chat, :tools, :streaming])

default_model(::Type{GEMINIServiceEndpoint}) = "gemini-3.5-flash"
```

- [ ] **Step 4: Include + export.**

`src/UniLM.jl`, after `include("anthropic.jl")` (line 55):

```julia
    include("gemini.jl")
```

`src/UniLM.jl` export block — the line that (after Task 1) reads `GEMINIOpenAIServiceEndpoint,` gains a sibling:

```julia
    GEMINIServiceEndpoint,
    GEMINIOpenAIServiceEndpoint,
```

- [ ] **Step 5: Write the failing test — create `test/gemini.jl`.**

```julia
# Native Gemini translation — deterministic, zero-spend unit tests.
using UniLM
using UniLM: encode_request, decode_response, decode_stream_chunk, StreamState,
             _build_stream_message, GEMINIServiceEndpoint, GPTFunction, GPTToolChoice,
             GEMINI_NATIVE_BASE, RoleSystem, RoleUser, RoleAssistant, RoleTool,
             TOOL_CALLS, STOP, CONTENT_FILTER
using Test, HTTP, JSON

@testset "routing — model in URL, stream branches on method" begin
    chat = Chat(service=GEMINIServiceEndpoint, model="gemini-3.5-flash")
    @test UniLM.get_url(chat) == "$(GEMINI_NATIVE_BASE)/models/gemini-3.5-flash:generateContent"
    schat = Chat(service=GEMINIServiceEndpoint, model="gemini-3.5-flash", stream=true)
    @test UniLM.get_url(schat) == "$(GEMINI_NATIVE_BASE)/models/gemini-3.5-flash:streamGenerateContent?alt=sse"
end

@testset "auth — x-goog-api-key" begin
    withenv("GEMINI_API_KEY" => "test-key") do
        h = Dict(UniLM.auth_header(GEMINIServiceEndpoint))
        @test h["x-goog-api-key"] == "test-key"
        @test !haskey(h, "Authorization")            # NOT Bearer
    end
end

@testset "capabilities & default" begin
    @test UniLM.provider_capabilities(GEMINIServiceEndpoint) == Set([:chat, :tools, :streaming])
    @test UniLM.default_model(GEMINIServiceEndpoint) == "gemini-3.5-flash"
    @test_throws ArgumentError UniLM._api_base_url(GEMINIServiceEndpoint)
end
```

- [ ] **Step 6: Register `test/gemini.jl` in `test/runtests.jl`.**

After the `anthropic` testset (line 72-74):

```julia
    @testset "gemini" begin
        include("gemini.jl")
    end
```

- [ ] **Step 7: Run the new test — expect PASS.**

Run: `env -u OPENAI_API_KEY -u DEEPSEEK_API_KEY -u ANTHROPIC_API_KEY -u GEMINI_API_KEY julia --project=. -e 'using UniLM, Test, HTTP, JSON; include("test/gemini.jl")'`
Expected: PASS (3 testsets).

- [ ] **Step 8: Commit.**

```bash
git add src/gemini.jl src/constants.jl src/api.jl src/UniLM.jl test/gemini.jl test/runtests.jl
git commit -m "feat(gemini): native GEMINIServiceEndpoint scaffolding — routing, auth, caps"
```

---

### Task 4: `encode_request` — neutral Chat → Gemini body

**Files:**
- Modify: `src/gemini.jl` (append encoder + helpers)
- Test: `test/gemini.jl` (append)

**Interfaces:**
- Produces: `encode_request(::Type{GEMINIServiceEndpoint}, chat::Chat)::String`; helpers `_gemini_contents`, `_gemini_model_parts`, `_gemini_tool`, `_gemini_tool_config`, `_gemini_tool_response`.
- Consumes: `GPTToolCall.thought_signature` (Task 2); `chat.messages/tools/tool_choice/max_tokens/max_completion_tokens/temperature/top_p/stop`.

- [ ] **Step 1: Write the failing tests.** Append to `test/gemini.jl`:

```julia
@testset "encode — system → systemInstruction, user turn" begin
    chat = Chat(service=GEMINIServiceEndpoint, model="gemini-3.5-flash")
    push!(chat, Message(Val(:system), "You are helpful."))
    push!(chat, Message(Val(:user), "Hi"))
    body = JSON.parse(encode_request(GEMINIServiceEndpoint, chat))
    @test body["systemInstruction"]["parts"][1]["text"] == "You are helpful."
    @test length(body["contents"]) == 1
    @test body["contents"][1]["role"] == "user"
    @test body["contents"][1]["parts"][1]["text"] == "Hi"
    @test !haskey(body, "generationConfig")                 # nothing set → omitted
    @test !haskey(body, "stream")                            # stream is URL-only
end

@testset "encode — generationConfig present only when set" begin
    chat = Chat(service=GEMINIServiceEndpoint, model="gemini-3.5-flash",
                max_tokens=256, temperature=0.5, stop=["END"])
    push!(chat, Message(Val(:user), "u"))
    body = JSON.parse(encode_request(GEMINIServiceEndpoint, chat))
    @test body["generationConfig"]["maxOutputTokens"] == 256
    @test body["generationConfig"]["temperature"] == 0.5
    @test body["generationConfig"]["stopSequences"] == ["END"]
end

@testset "encode — tools → functionDeclarations + toolConfig" begin
    sig = GPTFunctionSignature(name="get_weather", description="Get weather",
        parameters=Dict("type" => "object",
                        "properties" => Dict("location" => Dict("type" => "string")),
                        "required" => ["location"]))
    chat = Chat(service=GEMINIServiceEndpoint, model="gemini-3.5-flash",
                tools=[GPTTool(func=sig)], tool_choice="auto")
    push!(chat, Message(Val(:user), "weather?"))
    body = JSON.parse(encode_request(GEMINIServiceEndpoint, chat))
    fd = body["tools"][1]["functionDeclarations"][1]
    @test fd["name"] == "get_weather"
    @test fd["description"] == "Get weather"
    @test fd["parameters"]["type"] == "object"
    @test body["toolConfig"]["functionCallingConfig"]["mode"] == "AUTO"
end

@testset "encode — multi-turn: functionCall echo (+thoughtSignature) & functionResponse correlation" begin
    chat = Chat(service=GEMINIServiceEndpoint, model="gemini-3.5-flash")
    push!(chat, Message(Val(:user), "weather?"))
    tc = GPTToolCall(id="fc_1", func=GPTFunction("get_weather", Dict("location" => "Paris")),
                     thought_signature="SIG123")
    push!(chat, Message(role=RoleAssistant, tool_calls=[tc], finish_reason=TOOL_CALLS))
    push!(chat, Message(role=RoleTool, tool_call_id="fc_1", content="72F"))
    body = JSON.parse(encode_request(GEMINIServiceEndpoint, chat))
    c = body["contents"]
    @test [x["role"] for x in c] == ["user", "model", "user"]
    fc = c[2]["parts"][1]
    @test fc["functionCall"]["name"] == "get_weather"
    @test fc["functionCall"]["args"] == Dict("location" => "Paris")
    @test fc["thoughtSignature"] == "SIG123"                 # echoed
    fr = c[3]["parts"][1]["functionResponse"]
    @test fr["name"] == "get_weather"                        # correlated by id
    @test fr["id"] == "fc_1"
    @test fr["response"] == Dict("result" => "72F")          # string wrapped as object
end

@testset "encode — orphan functionResponse fails loud" begin
    msgs = [Message(Val(:user), "hi"),
            Message(role=RoleTool, tool_call_id="ghost", content="x")]
    chat = Chat(service=GEMINIServiceEndpoint, model="gemini-3.5-flash", messages=msgs)
    @test_throws ArgumentError encode_request(GEMINIServiceEndpoint, chat)
end
```

- [ ] **Step 2: Run — expect FAIL** (`encode_request` for Gemini not defined → falls to default, wrong shape).

Run: `env -u OPENAI_API_KEY -u DEEPSEEK_API_KEY -u ANTHROPIC_API_KEY -u GEMINI_API_KEY julia --project=. -e 'using UniLM, Test, HTTP, JSON; include("test/gemini.jl")'`
Expected: FAIL on the encode testsets.

- [ ] **Step 3: Implement the encoder + helpers.** Append to `src/gemini.jl`:

```julia
# ─── Request encoding (neutral Chat → Gemini generateContent body) ───────────

function encode_request(::Type{GEMINIServiceEndpoint}, chat::Chat)
    body = Dict{Symbol,Any}()
    sysinstr, contents = _gemini_contents(chat.messages)
    isnothing(sysinstr) || (body[:systemInstruction] = Dict(:parts => [Dict(:text => sysinstr)]))
    body[:contents] = contents
    isnothing(chat.tools) ||
        (body[:tools] = [Dict(:functionDeclarations => [_gemini_tool(t) for t in chat.tools])])
    isnothing(chat.tool_choice) || (body[:toolConfig] = _gemini_tool_config(chat.tool_choice))
    gen = Dict{Symbol,Any}()
    # Gemini does NOT require maxOutputTokens; omit when unset (a low cap truncates
    # Gemini-3 thinking before any answer). No default_max_tokens override.
    # NB: plain `something(a, b, nothing)` THROWS when all are nothing — use a ternary
    # so "both unset" yields nothing (→ omitted), not an ArgumentError.
    mot = !isnothing(chat.max_completion_tokens) ? chat.max_completion_tokens : chat.max_tokens
    isnothing(mot)              || (gen[:maxOutputTokens] = mot)
    isnothing(chat.temperature) || (gen[:temperature] = chat.temperature)
    isnothing(chat.top_p)       || (gen[:topP] = chat.top_p)
    isnothing(chat.stop)        || (gen[:stopSequences] = chat.stop isa String ? [chat.stop] : chat.stop)
    isempty(gen) || (body[:generationConfig] = gen)
    # NB: `stream` is expressed in the URL method (get_url), never in the body.
    JSON.json(body)
end

# Split neutral messages into (systemInstruction::Union{String,Nothing}, contents).
# - system → concatenated top-level systemInstruction text
# - user   → {role:"user", parts:[{text}]}
# - assistant → {role:"model", parts:[{text}?, {functionCall,thoughtSignature?}...]}
# - consecutive tool results → ONE {role:"user"} of functionResponse parts
# - a tool result with no preceding model functionCall of that id → loud ArgumentError.
function _gemini_contents(messages)
    sysinstr = nothing
    out = Vector{Dict{Symbol,Any}}()
    tool_names = Dict{String,String}()        # id → function name (functionResponse correlation)
    pending = Vector{Dict{Symbol,Any}}()      # buffered functionResponse parts
    function flush!()
        isempty(pending) && return
        push!(out, Dict{Symbol,Any}(:role => "user", :parts => copy(pending)))
        empty!(pending)
    end
    for m in messages
        if m.role == RoleSystem
            sysinstr = isnothing(sysinstr) ? m.content :
                       string(sysinstr, "\n\n", something(m.content, ""))
        elseif m.role == RoleTool
            tcid = something(m.tool_call_id, "")
            haskey(tool_names, tcid) || throw(ArgumentError(
                "functionResponse references unknown tool_call id $(repr(tcid)); no preceding model functionCall emitted it"))
            push!(pending, Dict{Symbol,Any}(:functionResponse => Dict{Symbol,Any}(
                :id => tcid, :name => tool_names[tcid], :response => _gemini_tool_response(m.content))))
        elseif m.role == RoleAssistant
            flush!()
            push!(out, Dict{Symbol,Any}(:role => "model", :parts => _gemini_model_parts(m, tool_names)))
        else  # RoleUser
            flush!()
            push!(out, Dict{Symbol,Any}(:role => "user",
                :parts => [Dict{Symbol,Any}(:text => something(m.content, ""))]))
        end
    end
    flush!()
    (sysinstr, out)
end

# Assistant turn → Gemini model parts: optional text + functionCall parts (id, args,
# thoughtSignature echoed). Records id→name into `tool_names` for later functionResponse.
function _gemini_model_parts(m::Message, tool_names)
    parts = Vector{Dict{Symbol,Any}}()
    (isnothing(m.content) || isempty(m.content)) ||
        push!(parts, Dict{Symbol,Any}(:text => m.content))
    isnothing(m.tool_calls) && return parts
    for tc in m.tool_calls
        tool_names[tc.id] = tc.func.name
        part = Dict{Symbol,Any}(:functionCall => Dict{Symbol,Any}(
            :id => tc.id, :name => tc.func.name, :args => tc.func.arguments))
        isnothing(tc.thought_signature) || (part[:thoughtSignature] = tc.thought_signature)
        push!(parts, part)
    end
    parts
end

function _gemini_tool(t::GPTTool)
    f = t.func
    d = Dict{Symbol,Any}(:name => f.name)
    isnothing(f.description) || (d[:description] = f.description)
    isnothing(f.parameters)  || (d[:parameters] = f.parameters)
    d
end

_gemini_tool_config(tc::String) = Dict(:functionCallingConfig => Dict(:mode =>
    tc == "auto"     ? "AUTO" :
    tc == "none"     ? "NONE" :
    tc == "required" ? "ANY"  : "AUTO"))
_gemini_tool_config(tc::GPTToolChoice) = Dict(:functionCallingConfig =>
    Dict(:mode => "ANY", :allowedFunctionNames => [string(tc.func)]))

# Gemini requires functionResponse.response to be a JSON OBJECT. Pass through a
# JSON-object string; otherwise wrap the raw string as {"result": ...}.
function _gemini_tool_response(content)
    s = something(content, "")
    try
        v = JSON.parse(s; dicttype=Dict{String,Any})
        v isa AbstractDict ? v : Dict{String,Any}("result" => s)
    catch
        Dict{String,Any}("result" => s)
    end
end
```

- [ ] **Step 4: Run — expect PASS.**

Run: `env -u OPENAI_API_KEY -u DEEPSEEK_API_KEY -u ANTHROPIC_API_KEY -u GEMINI_API_KEY julia --project=. -e 'using UniLM, Test, HTTP, JSON; include("test/gemini.jl")'`
Expected: PASS.

- [ ] **Step 5: Commit.**

```bash
git add src/gemini.jl test/gemini.jl
git commit -m "feat(gemini): encode neutral Chat → generateContent body"
```

---

### Task 5: `decode_response` — Gemini → neutral Message

**Files:**
- Modify: `src/gemini.jl` (append decoder + `_gemini_finish_reason`, `_gemini_usage`)
- Test: `test/gemini.jl` (append)

**Interfaces:**
- Produces: `decode_response(::Type{GEMINIServiceEndpoint}, resp::HTTP.Response)::@NamedTuple{message::Message, usage::Union{TokenUsage,Nothing}}`; `_gemini_finish_reason(fr)`; `_gemini_usage(u)`.
- Consumes: `GPTToolCall.thought_signature`; `TOOL_CALLS`/`STOP`/`CONTENT_FILTER`.

- [ ] **Step 1: Write the failing tests.** Append to `test/gemini.jl`:

```julia
@testset "decode — plain text" begin
    body = JSON.json(Dict("candidates" => [Dict(
        "content" => Dict("role" => "model", "parts" => [Dict("text" => "Hello there")]),
        "finishReason" => "STOP")],
        "usageMetadata" => Dict("promptTokenCount" => 10, "candidatesTokenCount" => 3,
                                "totalTokenCount" => 13)))
    r = decode_response(GEMINIServiceEndpoint, HTTP.Response(200, [], Vector{UInt8}(body)))
    @test r.message.role == RoleAssistant
    @test r.message.content == "Hello there"
    @test r.message.finish_reason == STOP
    @test r.usage.prompt_tokens == 10
    @test r.usage.completion_tokens == 3
    @test r.usage.total_tokens == 13
end

@testset "decode — text + functionCall (thoughtSignature captured; presence → TOOL_CALLS)" begin
    body = JSON.json(Dict("candidates" => [Dict(
        "content" => Dict("role" => "model", "parts" => [
            Dict("text" => "Let me check."),
            Dict("functionCall" => Dict("id" => "fc_9", "name" => "get_weather",
                                        "args" => Dict("location" => "Paris")),
                 "thoughtSignature" => "SIGX")]),
        "finishReason" => "STOP")],                        # Gemini says STOP even for tool calls
        "usageMetadata" => Dict("promptTokenCount" => 20, "candidatesTokenCount" => 15,
                                "totalTokenCount" => 35)))
    r = decode_response(GEMINIServiceEndpoint, HTTP.Response(200, [], Vector{UInt8}(body)))
    @test r.message.finish_reason == TOOL_CALLS
    @test r.message.content == "Let me check."
    @test r.message.tool_calls[1].id == "fc_9"
    @test r.message.tool_calls[1].func.name == "get_weather"
    @test r.message.tool_calls[1].func.arguments == Dict("location" => "Paris")
    @test r.message.tool_calls[1].thought_signature == "SIGX"
end

@testset "decode — MAX_TOKENS → length" begin
    body = JSON.json(Dict("candidates" => [Dict(
        "content" => Dict("role" => "model", "parts" => [Dict("text" => "partial")]),
        "finishReason" => "MAX_TOKENS")],
        "usageMetadata" => Dict("promptTokenCount" => 5, "candidatesTokenCount" => 100,
                                "totalTokenCount" => 105)))
    r = decode_response(GEMINIServiceEndpoint, HTTP.Response(200, [], Vector{UInt8}(body)))
    @test r.message.finish_reason == "length"
    @test r.message.content == "partial"
end

@testset "decode — SAFETY → content_filter refusal" begin
    body = JSON.json(Dict("candidates" => [Dict(
        "content" => Dict("role" => "model", "parts" => []),
        "finishReason" => "SAFETY")],
        "usageMetadata" => Dict("promptTokenCount" => 4, "candidatesTokenCount" => 0,
                                "totalTokenCount" => 4)))
    r = decode_response(GEMINIServiceEndpoint, HTTP.Response(200, [], Vector{UInt8}(body)))
    @test r.message.finish_reason == CONTENT_FILTER
    @test !isnothing(r.message.refusal_message)
end

@testset "decode — UNKNOWN finishReason does not crash (open enum)" begin
    body = JSON.json(Dict("candidates" => [Dict(
        "content" => Dict("role" => "model", "parts" => [Dict("text" => "partial")]),
        "finishReason" => "TOO_MANY_TOOL_CALLS")],       # never-seen value
        "usageMetadata" => Dict("promptTokenCount" => 5, "candidatesTokenCount" => 2,
                                "totalTokenCount" => 7)))
    r = decode_response(GEMINIServiceEndpoint, HTTP.Response(200, [], Vector{UInt8}(body)))
    @test r.message.finish_reason == STOP                  # unknown → safe default
    @test r.message.content == "partial"
end

@testset "decode — usage: cached is a subset of prompt; thoughts bill as output" begin
    body = JSON.json(Dict("candidates" => [Dict(
        "content" => Dict("role" => "model", "parts" => [Dict("text" => "hi")]),
        "finishReason" => "STOP")],
        "usageMetadata" => Dict("promptTokenCount" => 104, "candidatesTokenCount" => 5,
                                "thoughtsTokenCount" => 20, "cachedContentTokenCount" => 100,
                                "totalTokenCount" => 129)))
    r = decode_response(GEMINIServiceEndpoint, HTTP.Response(200, [], Vector{UInt8}(body)))
    @test r.usage.prompt_tokens == 104                     # promptTokenCount already includes cached
    @test r.usage.cached_tokens == 100
    @test r.usage.completion_tokens == 25                  # candidates(5) + thoughts(20), billed as output
    @test r.usage.reasoning_tokens == 20
    @test r.usage.total_tokens == 129
end
```

- [ ] **Step 2: Run — expect FAIL.**

Run: `env -u OPENAI_API_KEY -u DEEPSEEK_API_KEY -u ANTHROPIC_API_KEY -u GEMINI_API_KEY julia --project=. -e 'using UniLM, Test, HTTP, JSON; include("test/gemini.jl")'`
Expected: FAIL on the decode testsets.

- [ ] **Step 3: Implement the decoder + helpers.** Append to `src/gemini.jl`:

```julia
# ─── Response decoding (Gemini generateContent → neutral Message) ────────────

# Gemini finishReason → neutral finish_reason. OPEN ENUM: Google adds values
# unannounced, so unknown → STOP (never throw). Tool-call detection is by
# functionCall presence in the decoder, NOT by this reason (Gemini says STOP).
function _gemini_finish_reason(fr)
    fr == "STOP"       ? STOP :
    fr == "MAX_TOKENS" ? "length" :
    fr in ("SAFETY", "RECITATION", "BLOCKLIST", "PROHIBITED_CONTENT", "SPII", "IMAGE_SAFETY") ? CONTENT_FILTER :
    isnothing(fr)      ? STOP :
    STOP
end

# Gemini usageMetadata → neutral TokenUsage. Assumptions (verify vs live docs):
# promptTokenCount INCLUDES cachedContentTokenCount (cached is a subset), so we
# do NOT add; candidatesTokenCount EXCLUDES thoughtsTokenCount, and thoughts bill
# at the output rate → completion = candidates + thoughts.
function _gemini_usage(u)::Union{TokenUsage,Nothing}
    u isa AbstractDict || return nothing
    _i(x) = x isa Integer ? Int(x) : 0
    prompt   = _i(get(u, "promptTokenCount", 0))
    cand     = _i(get(u, "candidatesTokenCount", 0))
    thoughts = _i(get(u, "thoughtsTokenCount", 0))
    cached   = _i(get(u, "cachedContentTokenCount", 0))
    total    = _i(get(u, "totalTokenCount", prompt + cand + thoughts))
    TokenUsage(prompt_tokens = prompt, completion_tokens = cand + thoughts,
        total_tokens = total, cached_tokens = cached, reasoning_tokens = thoughts)
end

function decode_response(::Type{GEMINIServiceEndpoint}, resp::HTTP.Response)
    data = JSON.parse(resp.body; dicttype=Dict{String,Any})
    cands = get(data, "candidates", [])
    cand = isempty(cands) ? Dict{String,Any}() : cands[1]
    fr_raw = get(cand, "finishReason", nothing)
    text = IOBuffer()
    tool_calls = GPTToolCall[]
    for p in get(get(cand, "content", Dict{String,Any}()), "parts", [])
        if haskey(p, "text")
            print(text, p["text"])
        elseif haskey(p, "functionCall")
            fc = p["functionCall"]
            args = get(fc, "args", Dict{String,Any}())
            args isa AbstractDict || (args = Dict{String,Any}())
            push!(tool_calls, GPTToolCall(id=get(fc, "id", ""),
                func=GPTFunction(get(fc, "name", ""), args),
                thought_signature=get(p, "thoughtSignature", nothing)))
        end
    end
    finish = isempty(tool_calls) ? _gemini_finish_reason(fr_raw) : TOOL_CALLS
    usage = _gemini_usage(get(data, "usageMetadata", nothing))
    txt = String(take!(text))
    msg = if !isempty(tool_calls)
        Message(role=RoleAssistant, content=(isempty(txt) ? nothing : txt),
                tool_calls=tool_calls, finish_reason=finish)
    elseif finish == CONTENT_FILTER && isempty(txt)
        Message(role=RoleAssistant, refusal_message="Model response blocked by safety filter.",
                finish_reason=finish)
    else
        Message(role=RoleAssistant, content=(isempty(txt) ? "No response from the model." : txt),
                finish_reason=finish)
    end
    (; message=msg, usage)
end
```

- [ ] **Step 4: Run — expect PASS.**

Run: `env -u OPENAI_API_KEY -u DEEPSEEK_API_KEY -u ANTHROPIC_API_KEY -u GEMINI_API_KEY julia --project=. -e 'using UniLM, Test, HTTP, JSON; include("test/gemini.jl")'`
Expected: PASS.

- [ ] **Step 5: Commit.**

```bash
git add src/gemini.jl test/gemini.jl
git commit -m "feat(gemini): decode generateContent response → neutral Message"
```

---

### Task 6: `decode_stream_chunk` — SSE → StreamState (+ shared thought_signature rebuild)

**Files:**
- Modify: `src/gemini.jl` (append streaming decoder)
- Modify: `src/requests.jl:227` (one line in `_build_stream_message`)
- Test: `test/gemini.jl` (append)

**Interfaces:**
- Produces: `decode_stream_chunk(::Type{GEMINIServiceEndpoint}, chunk::String, state::StreamState, failbuff)::@NamedTuple{eos::Bool}`.
- Consumes: `_gemini_finish_reason`, `_gemini_usage` (Task 5); `state.content/tool_calls/finish_reason/usage`.

- [ ] **Step 1: Write the failing tests.** Append to `test/gemini.jl`:

```julia
@testset "stream — text deltas + final usage/finishReason + EOS" begin
    lines = [
        "data: " * JSON.json(Dict("candidates" => [Dict("content" =>
            Dict("role" => "model", "parts" => [Dict("text" => "Hello")]))],
            "usageMetadata" => Dict("promptTokenCount" => 8))),
        "",
        "data: " * JSON.json(Dict("candidates" => [Dict("content" =>
            Dict("role" => "model", "parts" => [Dict("text" => " world")]))])),
        "",
        "data: " * JSON.json(Dict("candidates" => [Dict(
            "content" => Dict("role" => "model", "parts" => [Dict("text" => "")]),
            "finishReason" => "STOP")],
            "usageMetadata" => Dict("promptTokenCount" => 8, "candidatesTokenCount" => 5,
                                    "totalTokenCount" => 13))),
    ]
    state = StreamState()
    st = decode_stream_chunk(GEMINIServiceEndpoint, join(lines, "\n"), state, IOBuffer())
    @test st.eos == true
    @test state.finish_reason == STOP
    @test state.usage.completion_tokens == 5
    @test state.usage.prompt_tokens == 8
    msg = _build_stream_message(state)
    @test msg.content == "Hello world"
    @test msg.finish_reason == STOP
end

@testset "stream — functionCall + thoughtSignature via _build_stream_message" begin
    lines = [
        "data: " * JSON.json(Dict("candidates" => [Dict("content" =>
            Dict("role" => "model", "parts" => [Dict(
                "functionCall" => Dict("id" => "fc_7", "name" => "get_weather",
                                       "args" => Dict("location" => "Paris")),
                "thoughtSignature" => "SIG7")]))])),
        "",
        "data: " * JSON.json(Dict("candidates" => [Dict(
            "content" => Dict("role" => "model", "parts" => [Dict("text" => "")]),
            "finishReason" => "STOP")],
            "usageMetadata" => Dict("promptTokenCount" => 12, "candidatesTokenCount" => 20,
                                    "totalTokenCount" => 32))),
    ]
    state = StreamState()
    st = decode_stream_chunk(GEMINIServiceEndpoint, join(lines, "\n"), state, IOBuffer())
    @test st.eos == true
    @test state.finish_reason == TOOL_CALLS                 # functionCall present overrides STOP
    msg = _build_stream_message(state)
    @test msg.finish_reason == TOOL_CALLS
    @test length(msg.tool_calls) == 1
    @test msg.tool_calls[1].id == "fc_7"
    @test msg.tool_calls[1].func.name == "get_weather"
    @test msg.tool_calls[1].func.arguments == Dict("location" => "Paris")
    @test msg.tool_calls[1].thought_signature == "SIG7"
end
```

- [ ] **Step 2: Run — expect FAIL** (streamed `functionCall` test fails: `_build_stream_message` drops `thought_signature`; Gemini stream decoder undefined).

Run: `env -u OPENAI_API_KEY -u DEEPSEEK_API_KEY -u ANTHROPIC_API_KEY -u GEMINI_API_KEY julia --project=. -e 'using UniLM, Test, HTTP, JSON; include("test/gemini.jl")'`
Expected: FAIL.

- [ ] **Step 3: Add `thought_signature` to the shared rebuild.** `src/requests.jl`, in `_build_stream_message`, the tool-call push (currently line 227):

```julia
            push!(tcalls, GPTToolCall(id=tc_data["id"], func=gptfunc,
                thought_signature=get(tc_data, "thought_signature", nothing)))
```

(Harmless for OpenAI/Anthropic — they never set the `"thought_signature"` key, so `get(...) → nothing`.)

- [ ] **Step 4: Implement the Gemini stream decoder.** Append to `src/gemini.jl`:

```julia
# ─── Streaming decode (Gemini SSE → StreamState) ─────────────────────────────
# Each `data:` line is a full PARTIAL GenerateContentResponse. Accumulate text and
# functionCall parts into the SAME StreamState the OpenAI path uses (tool args stored
# as a JSON string so the shared _build_stream_message JSON.parses them). No `[DONE]`
# sentinel — EOS is signalled by finishReason on the final chunk.

function decode_stream_chunk(::Type{GEMINIServiceEndpoint}, chunk::String, state::StreamState, failbuff)
    eos = false
    for line in filter(!isempty, strip.(split(chunk, "\n")))
        startswith(line, "data:") || continue
        payload = strip(line[6:end])
        isempty(payload) && continue
        try
            ev = JSON.parse(payload; dicttype=Dict{String,Any})
            cands = get(ev, "candidates", [])
            if !isempty(cands)
                cand = cands[1]
                for p in get(get(cand, "content", Dict{String,Any}()), "parts", [])
                    if haskey(p, "text")
                        print(state.content, p["text"])
                    elseif haskey(p, "functionCall")
                        fc = p["functionCall"]
                        idx = length(state.tool_calls)
                        state.tool_calls[idx] = Dict{String,Any}(
                            "id" => get(fc, "id", ""), "type" => "function",
                            "function" => Dict{String,Any}(
                                "name" => get(fc, "name", ""),
                                "arguments" => JSON.json(get(fc, "args", Dict{String,Any}()))),
                            "thought_signature" => get(p, "thoughtSignature", nothing))
                    end
                end
                fr = get(cand, "finishReason", nothing)
                if !isnothing(fr)
                    state.finish_reason = isempty(state.tool_calls) ? _gemini_finish_reason(fr) : TOOL_CALLS
                    eos = true
                end
            end
            u = get(ev, "usageMetadata", nothing)
            u isa AbstractDict && (state.usage = _gemini_usage(u))
        catch
            print(failbuff, line)
        end
    end
    (; eos)
end
```

- [ ] **Step 5: Run the Gemini unit suite — expect PASS.**

Run: `env -u OPENAI_API_KEY -u DEEPSEEK_API_KEY -u ANTHROPIC_API_KEY -u GEMINI_API_KEY julia --project=. -e 'using UniLM, Test, HTTP, JSON; include("test/gemini.jl")'`
Expected: PASS.

- [ ] **Step 6: Run the FULL suite — the `_build_stream_message` edit must not regress OpenAI/Anthropic streaming.**

Run:
```bash
env -u OPENAI_API_KEY -u DEEPSEEK_API_KEY -u ANTHROPIC_API_KEY -u GEMINI_API_KEY \
  julia --project=. -e 'using Pkg; Pkg.test()'
```
Expected: PASS (all prior tests + new Gemini tests + Aqua).

- [ ] **Step 7: Commit.**

```bash
git add src/gemini.jl src/requests.jl test/gemini.jl
git commit -m "feat(gemini): decode streamGenerateContent SSE into shared StreamState"
```

---

### Task 7: Pricing rows for `estimated_cost`

**Files:**
- Modify: `src/accounting.jl:10-30` (`DEFAULT_PRICING`)
- Test: `test/accounting.jl` (append)

**Interfaces:**
- Consumes: `_price(input, cached_input, output)` per-1M-token helper.

- [ ] **Step 1: Re-confirm live pricing before committing numbers** (prices drift; the dict comment mandates re-verification).

Run: `WebFetch https://ai.google.dev/gemini-api/docs/pricing` — confirm/correct: `gemini-3.5-flash` $1.50 in / $9.00 out / $0.15 cached; `gemini-3.1-flash-lite` $0.25 / $1.50 / $0.025; `gemini-2.5-flash` $0.30 / $2.50 / $0.03; `gemini-2.5-flash-lite` $0.10 / $0.40 / $0.01. (Verified 2026-07-07; correct the `_price(...)` args below if the page differs.)

- [ ] **Step 2: Write the failing test.** Append to `test/accounting.jl`:

```julia
@testset "Gemini pricing rows" begin
    @test haskey(UniLM.DEFAULT_PRICING, "gemini-3.5-flash")
    @test haskey(UniLM.DEFAULT_PRICING, "gemini-3.1-flash-lite")
    # 1000 prompt + 500 output on gemini-3.1-flash-lite = 1000*0.25/1e6 + 500*1.5/1e6
    chat = Chat(service=GEMINIServiceEndpoint, model="gemini-3.1-flash-lite")
    usage = UniLM.TokenUsage(prompt_tokens=1000, completion_tokens=500, total_tokens=1500)
    result = UniLM.LLMSuccess(message=Message(role=RoleAssistant, content="x"), self=chat, usage=usage)
    @test UniLM.estimated_cost(result) ≈ (1000 * 0.25 + 500 * 1.5) / 1_000_000
end
```

(Add `GEMINIServiceEndpoint`, `RoleAssistant` to the `using UniLM:` line in `test/accounting.jl` if not already imported.)

- [ ] **Step 3: Run — expect FAIL** (rows absent).

Run: `env -u OPENAI_API_KEY -u DEEPSEEK_API_KEY -u ANTHROPIC_API_KEY -u GEMINI_API_KEY julia --project=. -e 'using UniLM, Test; include("test/accounting.jl")'`
Expected: FAIL.

- [ ] **Step 4: Add the rows.** `src/accounting.jl`, inside `DEFAULT_PRICING` (after the Anthropic block, before the Embeddings block):

```julia
    # Google Gemini (native + OpenAI-compat shim; live-verified 2026-07-07)
    "gemini-3.5-flash"      => _price(1.5,  0.15,  9.0),
    "gemini-3.1-flash-lite" => _price(0.25, 0.025, 1.5),
    "gemini-2.5-flash"      => _price(0.3,  0.03,  2.5),
    "gemini-2.5-flash-lite" => _price(0.1,  0.01,  0.4),
```

- [ ] **Step 5: Run — expect PASS.**

Run: `env -u OPENAI_API_KEY -u DEEPSEEK_API_KEY -u ANTHROPIC_API_KEY -u GEMINI_API_KEY julia --project=. -e 'using UniLM, Test; include("test/accounting.jl")'`
Expected: PASS.

- [ ] **Step 6: Commit.**

```bash
git add src/accounting.jl test/accounting.jl
git commit -m "feat(gemini): DEFAULT_PRICING rows for gemini-3.x/2.5 flash"
```

---

### Task 8: Key-gated live witness + final full-suite verification

The end-to-end observation. Key-gated (skips without `GEMINI_API_KEY`); billed when run, so cheapest model + minimal calls.

**Files:**
- Create: `test/integration_gemini.jl`
- Modify: `test/runtests.jl` (register after `integration — anthropic`)

**Interfaces:**
- Consumes: `chatrequest!`, `LLMSuccess`, `cumulative_cost`, exported `GEMINIServiceEndpoint`.

- [ ] **Step 1: Create `test/integration_gemini.jl`** (mirrors `test/integration_anthropic.jl`):

```julia
# ─── Gemini Integration Tests (live) ─────────────────────────────────────────
# Requires GEMINI_API_KEY (billing-enabled). Uses gemini-3.1-flash-lite (cheapest)
# to minimize spend. Run once when green; do not rerun.

if !haskey(ENV, "GEMINI_API_KEY")
    @info "Skipping Gemini integration tests (GEMINI_API_KEY not set)"
else

@testset "Gemini Chat — basic" begin
    chat = Chat(service=UniLM.GEMINIServiceEndpoint, model="gemini-3.1-flash-lite", max_tokens=64)
    push!(chat, Message(Val(:system), "You are a helpful assistant."))
    push!(chat, Message(Val(:user), "Reply with exactly: hello"))
    result = chatrequest!(chat)
    @test result isa LLMSuccess
    @test !isempty(result.message.content)
    @test result.usage.completion_tokens > 0
    @test cumulative_cost(chat) > 0.0
end

@testset "Gemini Chat — tool round-trip" begin
    sig = GPTFunctionSignature(name="get_current_weather",
        description="Get the current weather for a location",
        parameters=Dict("type" => "object",
            "properties" => Dict("location" => Dict("type" => "string", "description" => "City name")),
            "required" => ["location"]))
    chat = Chat(service=UniLM.GEMINIServiceEndpoint, model="gemini-3.1-flash-lite", max_tokens=256,
                tools=[GPTTool(func=sig)], tool_choice="auto")
    push!(chat, Message(Val(:system), "Use the weather tool when asked about weather."))
    push!(chat, Message(Val(:user), "What is the weather in Paris?"))
    result = chatrequest!(chat)
    @test result isa LLMSuccess
    m = result.message
    if m.finish_reason == UniLM.TOOL_CALLS
        @test m.tool_calls[1].func.name == "get_current_weather"
        @test haskey(m.tool_calls[1].func.arguments, "location")
        # feed the tool result back — exercises thoughtSignature echo on the next turn
        push!(chat, Message(role=UniLM.RoleTool, tool_call_id=m.tool_calls[1].id, content="72F and sunny"))
        follow = chatrequest!(chat)
        @test follow isa LLMSuccess
        @test !isempty(something(follow.message.content, ""))
    else
        @test m.finish_reason == UniLM.STOP
    end
end

@testset "Gemini Chat — streaming" begin
    payloads = Any[]
    chat = Chat(service=UniLM.GEMINIServiceEndpoint, model="gemini-3.1-flash-lite",
                max_tokens=128, stream=true)
    push!(chat, Message(Val(:system), "You are helpful."))
    push!(chat, Message(Val(:user), "Count from 1 to 10, one number per line."))
    task = chatrequest!(chat; callback=(c, _) -> push!(payloads, c))
    result = fetch(task)
    @test result isa LLMSuccess
    @test !isempty(result.message.content)   # decode_stream_chunk accumulated real SSE
    @test !isempty(payloads)                 # callback fired (deltas and/or final message)
end

end  # if GEMINI_API_KEY
```

- [ ] **Step 2: Register it in `test/runtests.jl`** after the `integration — anthropic` testset (line 96-98):

```julia
    @testset "integration — gemini" begin
        include("integration_gemini.jl")
    end
```

- [ ] **Step 3: Zero-spend full suite — witness SKIPS, everything else GREEN.**

Run:
```bash
env -u OPENAI_API_KEY -u DEEPSEEK_API_KEY -u ANTHROPIC_API_KEY -u GEMINI_API_KEY \
  julia --project=. -e 'using Pkg; Pkg.test()'
```
Expected: PASS, with `@info "Skipping Gemini integration tests…"`.

- [ ] **Step 4: Verify the second HTTP major** (CI runs both; `HTTP = "1.9, 2"`).

Run:
```bash
julia --project=. -e 'using Pkg; Pkg.add(PackageSpec(name="HTTP", version="1"))'
env -u OPENAI_API_KEY -u DEEPSEEK_API_KEY -u ANTHROPIC_API_KEY -u GEMINI_API_KEY \
  julia --project=. -e 'using Pkg; Pkg.test()'
julia --project=. -e 'using Pkg; Pkg.add(PackageSpec(name="HTTP", version="2"))'
```
Expected: PASS on HTTP 1.x too. (Restores HTTP 2.x after.)

- [ ] **Step 5: Run the live witness ONCE (billed).** Only when everything above is green:

```bash
zsh -c 'source ~/.zshrc; env -u OPENAI_API_KEY -u ANTHROPIC_API_KEY -u DEEPSEEK_API_KEY \
  julia --project=. -e "using UniLM, Test, HTTP, JSON; include(\"test/integration_gemini.jl\")"'
```
Expected: 3 testsets PASS (basic text, tool round-trip, streaming). This proves the translator end-to-end; do NOT rerun. If a wire assumption from the ledger was wrong (usage reconciliation, `thoughtSignature` echo, streamed `functionCall` shape), it surfaces HERE — fix the decoder/encoder, re-run once.

- [ ] **Step 6: Commit.**

```bash
git add test/integration_gemini.jl test/runtests.jl
git commit -m "test(gemini): key-gated live witness — text, tool round-trip, streaming"
```

---

## Definition of Done

- All 8 tasks committed on `gemini-native-chat`.
- Full zero-spend suite + Aqua green on both HTTP majors; Gemini witness green once (billed) or skipping.
- `grep -rn 'GEMINIServiceEndpoint' src test` shows only NATIVE usages; the shim is `GEMINIOpenAIServiceEndpoint` everywhere.
- Neutral IR changed by exactly one optional field (`GPTToolCall.thought_signature`).
- Release note drafted under `## Breaking changes`: `GEMINIServiceEndpoint` now targets the native `generateContent` API; the OpenAI-compat shim is `GEMINIOpenAIServiceEndpoint`.
