# ============================================================================
# Anthropic (Claude) native Messages API
# Plugs into the wire-translation seam (encode_request / decode_response /
# handle_sse_event! from sse.jl) so all chat orchestration is shared.
# Wire shape verified against the Anthropic Messages API docs on 2026-07-06.
# ============================================================================

# ─── Routing & auth ──────────────────────────────────────────────────────────

get_url(::Type{ANTHROPICServiceEndpoint}, ::Chat)::String = ANTHROPIC_BASE_URL * ANTHROPIC_MESSAGES_PATH

function auth_header(::Type{ANTHROPICServiceEndpoint})::Vector{Pair{String,String}}
    [
        "x-api-key" => ENV[ANTHROPIC_API_KEY],
        "anthropic-version" => ANTHROPIC_VERSION,
        "Content-Type" => "application/json",
    ]
end

# ─── Capabilities & defaults ─────────────────────────────────────────────────

provider_capabilities(::Type{ANTHROPICServiceEndpoint}) =
    Set([:chat, :tools, :json_output, :streaming])

default_model(::Type{ANTHROPICServiceEndpoint}) = "claude-opus-5-5"

"""
    default_max_tokens(service, model::AbstractString) -> Int

`max_tokens` supplied when the caller leaves it unset. Anthropic *requires* the
field; OpenAI does not. Returns a moderate, overridable default (see
`_ANTHROPIC_DEFAULT_MAX_TOKENS`), not the model ceiling.
"""
default_max_tokens(::Type{ANTHROPICServiceEndpoint}, ::AbstractString) = _ANTHROPIC_DEFAULT_MAX_TOKENS

# ─── Claude model families ───────────────────────────────────────────────────
# Request contract per family, transcribed on 2026-09-24 from
#   thinking modes and rejected configs: https://platform.claude.com/docs/en/build-with-claude/thinking.md
#   effort levels per model:             https://platform.claude.com/docs/en/build-with-claude/effort.md
#   sampling, prefill, forced tools:     thinking.md "Limits and feature compatibility",
#                                        https://platform.claude.com/docs/en/models/sonnet-5/migration-guide.md
#   mid-conversation system messages:    https://platform.claude.com/docs/en/build-with-claude/mid-conversation-system-messages.md
# `thinking`: :always = adaptive, cannot be disabled; :on = adaptive by default, can be
# disabled; :off = adaptive on request; :manual = extended thinking (budget_tokens) only.
# `efforts`: accepted output_config.effort levels (empty = no effort parameter).
# `sampling`: non-default temperature/top_p accepted. `forced`: tool_choice any/tool
# accepted. `prefill`: a trailing assistant turn accepted. `system`: role "system"
# messages after the conversation starts accepted.
const _CLAUDE_EFFORTS = ["low", "medium", "high", "xhigh", "max"]
const _CLAUDE_EFFORTS_NO_XHIGH = ["low", "medium", "high", "max"]

@kwdef struct _ClaudeFamily
    thinking::Symbol
    efforts::Vector{String} = _CLAUDE_EFFORTS
    sampling::Bool = false
    forced::Bool = true
    prefill::Bool = false
    system::Bool = false
end

const _CLAUDE_LEGACY = _ClaudeFamily(thinking=:manual, efforts=String[], sampling=true, prefill=true)

const _CLAUDE_FAMILIES = (
    "claude-fable-5-1"         => _ClaudeFamily(thinking=:always, forced=false, system=true),
    "claude-mythos-5-1"        => _ClaudeFamily(thinking=:always, forced=false, system=true),
    "claude-fable-5"           => _ClaudeFamily(thinking=:always, system=true),
    "claude-mythos-5"          => _ClaudeFamily(thinking=:always, system=true),
    "claude-mythos-preview"    => _ClaudeFamily(thinking=:always, efforts=_CLAUDE_EFFORTS_NO_XHIGH),
    "claude-opus-5-5"          => _ClaudeFamily(thinking=:always, forced=false, system=true),
    "claude-opus-5"            => _ClaudeFamily(thinking=:on, system=true),
    "claude-sonnet-5"          => _ClaudeFamily(thinking=:on),
    "claude-opus-4-8"          => _ClaudeFamily(thinking=:off, system=true),
    "claude-opus-4-7"          => _ClaudeFamily(thinking=:off),
    "claude-opus-4-6"          => _ClaudeFamily(thinking=:off, efforts=_CLAUDE_EFFORTS_NO_XHIGH, sampling=true),
    "claude-sonnet-4-6"        => _ClaudeFamily(thinking=:off, efforts=_CLAUDE_EFFORTS_NO_XHIGH, sampling=true),
    "claude-opus-4-5"          => _ClaudeFamily(thinking=:manual, efforts=["low", "medium", "high"],
                                                sampling=true, prefill=true),
    "claude-sonnet-4-5"        => _CLAUDE_LEGACY,
    "claude-haiku-4-5"         => _CLAUDE_LEGACY,
    "claude-opus-4-1"          => _CLAUDE_LEGACY,
    "claude-opus-4-0"          => _CLAUDE_LEGACY,
    "claude-opus-4-20250514"   => _CLAUDE_LEGACY,
    "claude-sonnet-4-0"        => _CLAUDE_LEGACY,
    "claude-sonnet-4-20250514" => _CLAUDE_LEGACY,
    "claude-3"                 => _CLAUDE_LEGACY,
)

# Longest matching family (claude-opus-5-5 is also in the claude-opus-5 family);
# `nothing` for an id no row covers, e.g. a newer model, which the API validates alone.
function _claude_family(model::AbstractString)::Union{_ClaudeFamily,Nothing}
    rows = filter(row -> _model_family(model, first(row)), _CLAUDE_FAMILIES)
    isempty(rows) ? nothing : last(argmax(row -> length(first(row)), rows))
end

# reasoning_effort → (thinking, output_config.effort). A level also turns adaptive
# thinking on where the family has it, so an explicit request reasons on the models
# that run without thinking by default; "none" disables thinking only where it is on
# by default and can be turned off, and sends nothing where it is off by default.
function _anthropic_reasoning(model::String, effort::Union{String,Nothing},
                              fam::Union{_ClaudeFamily,Nothing})
    isnothing(effort) && return (nothing, nothing)
    effort == "minimal" && throw(ArgumentError(
        "Claude models have no minimal reasoning_effort; use \"low\""))
    effort == "none" || effort in _CLAUDE_EFFORTS || throw(ArgumentError(
        "Anthropic reasoning_effort must be none, low, medium, high, xhigh or max (got $(repr(effort)))"))
    if !isnothing(fam)
        isempty(fam.efforts) && throw(ArgumentError("$model does not support reasoning_effort"))
        effort == "none" && fam.thinking === :always && throw(ArgumentError(
            "$model cannot disable thinking, so reasoning_effort \"none\" is unavailable; use \"low\""))
        effort == "none" || effort in fam.efforts || throw(ArgumentError(
            "$model supports reasoning_effort $(join(fam.efforts, ", ")) (got $(repr(effort)))"))
    end
    effort == "none" &&
        return ((isnothing(fam) || fam.thinking === :on) ? Dict(:type => "disabled") : nothing, nothing)
    ((isnothing(fam) || fam.thinking !== :manual) ? Dict(:type => "adaptive") : nothing, effort)
end

# ─── Request encoding (neutral Chat → Anthropic Messages body) ───────────────

function encode_request(::Type{ANTHROPICServiceEndpoint}, chat::Chat)
    # OpenAI request controls with no Messages counterpart: dropping a moderation
    # policy or cache setting would send a request the caller did not ask for.
    for f in (:moderation, :prompt_cache_options)
        isnothing(getfield(chat, f)) || throw(ArgumentError(
            "Anthropic Messages does not support $f; it is an OpenAI-only option"))
    end
    fam = _claude_family(chat.model)
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
    thinking, effort = _anthropic_reasoning(chat.model, chat.reasoning_effort, fam)
    isnothing(thinking) || (body[:thinking] = thinking)
    output_config = Dict{Symbol,Any}()
    isnothing(effort) || (output_config[:effort] = effort)
    (fmt = _anthropic_output_format(chat.response_format)) === nothing || (output_config[:format] = fmt)
    isempty(output_config) || (body[:output_config] = output_config)
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

# Assistant turn → Anthropic content: echo captured provider-native blocks
# verbatim when this provider produced them (signatures intact, thinking
# first); otherwise reconstruct optional text block + tool_use blocks.
function _anthropic_assistant_content(m::Message)
    pc = m.provider_content
    pc isa ProviderContent && pc.provider === :anthropic && !isempty(pc.blocks) &&
        return pc.blocks
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

function _anthropic_tool(t::Tool)
    f = t.func
    d = Dict{Symbol,Any}(:name => f.name,
        :input_schema => something(f.parameters, Dict("type" => "object", "properties" => Dict())))
    isnothing(f.description) || (d[:description] = f.description)
    isnothing(f.strict)      || (d[:strict] = f.strict)
    d
end

# Neutral response_format → output_config.format (structured outputs,
# https://platform.claude.com/docs/en/build-with-claude/structured-outputs.md). The
# format object is {type: "json_schema", schema}: the OpenAI schema name, description
# and strict flag have no counterpart (the output is always constrained to the
# schema), and there is no schema-less JSON mode.
_anthropic_output_format(::Nothing) = nothing
function _anthropic_output_format(rf::ResponseFormat)
    js = rf.json_schema
    rf.type == "text" && isnothing(js) && return nothing
    rf.type == "json_object" && throw(ArgumentError(
        "Anthropic has no schema-less JSON mode for response_format json_object; use a json_schema response_format"))
    rf.type == "json_schema" || throw(ArgumentError(
        "Anthropic supports response_format json_schema (got $(repr(rf.type)))"))
    schema = js isa JsonSchemaAPI ? js.schema :
             js isa AbstractDict ? get(js, "schema", get(js, :schema, nothing)) : nothing
    schema isa AbstractDict || throw(ArgumentError("Anthropic json_schema response_format needs a schema object"))
    Dict(:type => "json_schema", :schema => schema)
end

_anthropic_tool_choice(tc::String) =
    tc == "auto"     ? Dict(:type => "auto") :
    tc == "none"     ? Dict(:type => "none") :
    tc == "required" ? Dict(:type => "any")  :
    Dict(:type => "auto")
_anthropic_tool_choice(tc::GPTToolChoice) = Dict(:type => "tool", :name => string(tc.func))

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
    blocks = get(data, "content", Any[])
    # Verbatim capture for round-trip: thinking/redacted_thinking signatures
    # must be echoed unmodified on the next turn (thinking models reject
    # modified blocks). Empty arrays are not captured — echoing [] back is a 400.
    pc = blocks isa AbstractVector && !isempty(blocks) ?
         ProviderContent(:anthropic, blocks) : nothing
    text = IOBuffer()
    tool_calls = ToolCall[]
    for b in (blocks isa AbstractVector ? blocks : Any[])
        bt = get(b, "type", "")
        if bt == "text"
            print(text, get(b, "text", ""))
        elseif bt == "tool_use"
            args = get(b, "input", Dict{String,Any}())
            args isa AbstractDict || (args = Dict{String,Any}())
            push!(tool_calls, ToolCall(id=b["id"], func=GPTFunction(b["name"], args)))
        end
        # thinking / redacted_thinking blocks are not flattened into the neutral
        # fields; they ride along verbatim in provider_content.
    end
    usage = _anthropic_usage(get(data, "usage", nothing))
    txt = String(take!(text))
    msg = if !isempty(tool_calls)
        Message(role=RoleAssistant, content=(isempty(txt) ? nothing : txt),
                tool_calls=tool_calls, finish_reason=finish, provider_content=pc)
    elseif finish == CONTENT_FILTER && isempty(txt)
        Message(role=RoleAssistant, refusal_message="Model refused to respond.",
                finish_reason=finish, provider_content=pc)
    else
        # A well-formed turn that produced no text is a real turn: thinking models
        # routinely spend the whole budget on thought blocks and stop at max_tokens
        # with no text block at all. Report the empty turn the provider sent —
        # substituting prose would inject content nobody generated into the reply and
        # into the next request's history (the thinking blocks themselves ride along
        # verbatim in provider_content).
        Message(role=RoleAssistant, content=txt, finish_reason=finish, provider_content=pc)
    end
    (; message=msg, usage)
end

# ─── Streaming event handler for the shared SSE machine (src/sse.jl) —
# replaces decode_stream_chunk (removed in 0.11.3).
# Populates the SAME StreamState fields the OpenAI path uses so the shared
# _build_stream_message rebuilds the neutral Message unchanged. EOS on
# `message_stop`; an in-band `error` event stores its payload in state.error
# and returns :error (the documented 529-equivalent arrives on an HTTP-200
# stream — it must never build an LLMSuccess). `content_block_stop` on a tool
# index marks that call complete for the driver's on_tool_call detection.
# Content blocks are additionally snapshotted verbatim and re-assembled into
# state.raw_blocks so streamed turns round-trip with provider-native fidelity
# (thinking signatures intact).
function handle_sse_event!(::Type{ANTHROPICServiceEndpoint}, event::AbstractString,
                           payload::AbstractString, state::StreamState)::Symbol
    ev = JSON.parse(payload; dicttype=Dict{String,Any})
    ev isa AbstractDict || return :continue
    t = get(ev, "type", "")
    if event == "error" || t == "error"
        state.error = ev
        return :error
    elseif t == "message_start"
        u = get(get(ev, "message", Dict{String,Any}()), "usage", nothing)
        u isa AbstractDict && (state.usage = _anthropic_usage(u))
    elseif t == "content_block_start"
        cb = get(ev, "content_block", nothing)   # no {} default: a missing block must not fabricate a raw entry
        idx = get(ev, "index", nothing)
        # Concrete Dict{String,Any} (raw_pending's value type): if the parse
        # dicttype ever changes, capture disables loudly here — keep in sync
        # with the AbstractDict tool-branch guard below.
        if idx isa Integer && cb isa Dict{String,Any}
            # Verbatim snapshot for round-trip assembly. The parsed event owns
            # this dict exclusively, so in-place delta accumulation is safe.
            state.raw_pending[idx] = cb
            state.raw_provider = :anthropic
        end
        if cb isa AbstractDict && get(cb, "type", "") == "tool_use"
            state.tool_calls[ev["index"]] = Dict{String,Any}(
                "id" => get(cb, "id", ""), "type" => "function",
                "function" => Dict{String,Any}("name" => get(cb, "name", ""), "arguments" => ""))
        end
    elseif t == "content_block_delta"
        idx = ev["index"]
        d = get(ev, "delta", Dict{String,Any}())
        dt = get(d, "type", "")
        blk = get(state.raw_pending, idx, nothing)
        if dt == "text_delta"
            txt = get(d, "text", "")
            print(state.content, txt)
            print(state.pending_delta, txt)
            isnothing(blk) || (blk["text"] = get(blk, "text", "") * txt)
        elseif dt == "input_json_delta"
            pj = get(d, "partial_json", "")
            haskey(state.tool_calls, idx) &&
                (state.tool_calls[idx]["function"]["arguments"] *= pj)
            isnothing(blk) || (state.raw_json[idx] = get(state.raw_json, idx, "") * pj)
        elseif dt == "thinking_delta"
            isnothing(blk) || (blk["thinking"] = get(blk, "thinking", "") * get(d, "thinking", ""))
        elseif dt == "signature_delta"
            isnothing(blk) || (blk["signature"] = get(blk, "signature", "") * get(d, "signature", ""))
        end
    elseif t == "content_block_stop"
        idx = get(ev, "index", nothing)
        idx isa Integer && haskey(state.tool_calls, idx) && (state.tool_calls[idx]["complete"] = true)
        if idx isa Integer && haskey(state.raw_pending, idx)
            blk = state.raw_pending[idx]
            # Streamed tool input arrives as partial JSON: finalize to a parsed
            # object so the block matches the non-streaming wire shape.
            haskey(state.raw_json, idx) &&
                (blk["input"] = _parse_tool_arguments(pop!(state.raw_json, idx)))
            push!(state.raw_blocks, blk)
            delete!(state.raw_pending, idx)
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
    elseif t == "message_stop"
        return :done
    end
    :continue
end
