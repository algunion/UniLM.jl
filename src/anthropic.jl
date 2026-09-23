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

# Fail closed when a neutral option has no Messages API counterpart: dropping it
# would send a request the caller did not ask for.
const _ANTHROPIC_CHAT_MAPPED_FIELDS = (:service, :model, :messages, :history, :tools,
    :tool_choice, :parallel_tool_calls, :temperature, :top_p, :n, :stream, :stop,
    :max_tokens, :max_completion_tokens, :response_format, :user, :reasoning_effort,
    :metadata, :service_tier, :safety_identifier, :_cumulative_cost)
const _ANTHROPIC_CHAT_UNMAPPED_FIELDS = Tuple(setdiff(fieldnames(Chat), _ANTHROPIC_CHAT_MAPPED_FIELDS))

function _anthropic_validate_fields(chat::Chat)
    for f in (:moderation, :prompt_cache_options)
        isnothing(getfield(chat, f)) || throw(ArgumentError(
            "Anthropic Messages does not support $f; it is an OpenAI-only option"))
    end
    isnothing(chat.n) || chat.n == 1 || throw(ArgumentError(
        "Anthropic Messages returns one completion; n must be 1"))
    fields = Symbol[f for f in _ANTHROPIC_CHAT_UNMAPPED_FIELDS if !isnothing(getfield(chat, f))]
    isempty(fields) || throw(ArgumentError(
        "Anthropic Messages does not support field(s) $(join(fields, ", "))"))
    # Messages API service_tier values: https://platform.claude.com/docs/en/api/messages.md
    isnothing(chat.service_tier) || chat.service_tier in ("auto", "standard_only") || throw(ArgumentError(
        "Anthropic service_tier must be \"auto\" or \"standard_only\" (got $(repr(chat.service_tier)))"))
    nothing
end

# safety_identifier, user and metadata user_id all name the end user; the Messages
# API has one slot for it, metadata.user_id, and no other metadata key.
function _anthropic_user_id(chat::Chat)::Union{String,Nothing}
    ids = Pair{Symbol,String}[]
    if !isnothing(chat.metadata)
        for (k, v) in chat.metadata
            string(k) == "user_id" || throw(ArgumentError(
                "Anthropic metadata accepts only user_id (got key $(repr(string(k))))"))
            v isa Union{AbstractString,Nothing} || throw(ArgumentError(
                "Anthropic metadata user_id must be a string (got $(typeof(v)))"))
            isnothing(v) || push!(ids, :metadata => v)
        end
    end
    isnothing(chat.safety_identifier) || push!(ids, :safety_identifier => chat.safety_identifier)
    isnothing(chat.user) || push!(ids, :user => chat.user)
    allequal(last.(ids)) || throw(ArgumentError(
        "Anthropic sends $(join(first.(ids), ", ")) as metadata.user_id; set one, or set them equal"))
    isempty(ids) ? nothing : last(first(ids))
end

# Requests a Claude model answers with HTTP 400, rejected before the round trip (the
# per-family limits are in the table above). The Messages API temperature range is 0..1.
function _anthropic_validate_model(chat::Chat, fam::Union{_ClaudeFamily,Nothing})
    m, t = chat.model, chat.temperature
    isnothing(t) || 0.0 <= t <= 1.0 || throw(ArgumentError(
        "Anthropic temperature must be in [0, 1] (got $t)"))
    isnothing(fam) && return nothing
    if !fam.sampling
        isnothing(t) || t == 1.0 || throw(ArgumentError(
            "$m accepts only the default temperature (1.0); remove temperature"))
        isnothing(chat.top_p) || throw(ArgumentError("$m does not accept top_p; remove it"))
    end
    fam.forced || !(chat.tool_choice isa GPTToolChoice || chat.tool_choice == "required") ||
        throw(ArgumentError("$m rejects forced tool_choice; use tool_choice=\"auto\" with strict " *
                            "tools (FunctionSignature(strict=true)) or a json_schema response_format"))
    nothing
end

function encode_request(::Type{ANTHROPICServiceEndpoint}, chat::Chat)
    _anthropic_validate_fields(chat)
    fam = _claude_family(chat.model)
    _anthropic_validate_model(chat, fam)
    body = Dict{Symbol,Any}(:model => chat.model)
    # max_tokens is REQUIRED by Anthropic; fall back to the moderate default.
    body[:max_tokens] = something(chat.max_completion_tokens, chat.max_tokens,
                                  default_max_tokens(ANTHROPICServiceEndpoint, chat.model))
    system, msgs = _anthropic_messages(chat.messages; model=chat.model,
                                       mid_system=isnothing(fam) || fam.system)
    isnothing(fam) || fam.prefill || isempty(msgs) || msgs[end][:role] != "assistant" ||
        throw(ArgumentError("$(chat.model) rejects messages that end with an assistant turn " *
                            "(response prefill); end with a user turn or use a json_schema response_format"))
    isnothing(system) || (body[:system] = system)
    body[:messages] = msgs
    isnothing(chat.tools)       || (body[:tools] = [_anthropic_tool(t) for t in chat.tools])
    disable_parallel = !isnothing(chat.tools) && !isempty(chat.tools) && chat.parallel_tool_calls === false
    (!isnothing(chat.tool_choice) || disable_parallel) &&
        (body[:tool_choice] = _anthropic_tool_choice(something(chat.tool_choice, "auto"), disable_parallel))
    isnothing(chat.stop)        || (body[:stop_sequences] = chat.stop isa String ? [chat.stop] : chat.stop)
    isnothing(chat.temperature) || (body[:temperature] = chat.temperature)
    isnothing(chat.top_p)       || (body[:top_p] = chat.top_p)
    (uid = _anthropic_user_id(chat)) === nothing || (body[:metadata] = Dict(:user_id => uid))
    isnothing(chat.service_tier) || (body[:service_tier] = chat.service_tier)
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
# - leading system messages → concatenated top-level `system`; a later one stays in
#   place as role "system" where the model accepts mid-conversation system messages
#   (`mid_system`) and raises elsewhere — hoisting it would edit the top-level prompt,
#   which invalidates the thinking blocks of every later turn on the newest models
# - consecutive `tool` messages → collapsed into ONE user message of tool_result blocks
# - assistant tool_calls → tool_use blocks; a tool_result referencing an id no
#   preceding assistant emitted → loud ArgumentError
# - an assistant turn with neither text nor tool calls (a refusal, an empty reply) is
#   skipped: the API rejects empty assistant content.
function _anthropic_messages(messages; model::AbstractString="", mid_system::Bool=true)
    system = nothing
    out = Vector{Dict{Symbol,Any}}()
    seen_tool_use_ids = Set{String}()
    pending = Vector{Dict{Symbol,Any}}()
    flush!() = (isempty(pending) ||
        (push!(out, Dict{Symbol,Any}(:role => "user", :content => copy(pending))); empty!(pending)))
    for m in messages
        if m.role == RoleSystem && isempty(out) && isempty(pending)
            system = isnothing(system) ? m.content : string(system, "\n\n", something(m.content, ""))
        elseif m.role == RoleSystem
            mid_system || throw(ArgumentError("$model does not accept system messages after the " *
                "conversation starts; put the instruction in the leading system message"))
            flush!()
            push!(out, Dict{Symbol,Any}(:role => "system", :content => something(m.content, "")))
        elseif m.role == RoleTool
            tcid = something(m.tool_call_id, "")
            tcid in seen_tool_use_ids || throw(ArgumentError(
                "tool_result references unknown tool_use id $(repr(tcid)); no preceding assistant tool_use emitted it"))
            push!(pending, Dict{Symbol,Any}(:type => "tool_result",
                :tool_use_id => tcid, :content => something(m.content, "")))
        elseif m.role == RoleAssistant
            flush!()
            isempty(something(m.content, "")) && (isnothing(m.tool_calls) || isempty(m.tool_calls)) && continue
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

# parallel_tool_calls=false rides on the tool_choice object as disable_parallel_tool_use
# (every variant but "none" carries it).
function _anthropic_tool_choice(tc::Union{String,GPTToolChoice}, disable_parallel::Bool)
    d = tc isa GPTToolChoice ? Dict{Symbol,Any}(:type => "tool", :name => string(tc.func)) :
        tc == "auto"         ? Dict{Symbol,Any}(:type => "auto") :
        tc == "none"         ? Dict{Symbol,Any}(:type => "none") :
        tc == "required"     ? Dict{Symbol,Any}(:type => "any") :
        throw(ArgumentError("Unknown Anthropic tool_choice $(repr(tc)); use \"auto\", \"none\", \"required\" or a GPTToolChoice"))
    disable_parallel && d[:type] != "none" && (d[:disable_parallel_tool_use] = true)
    d
end

# ─── Response decoding (Anthropic Messages → neutral Message) ────────────────

# Anthropic stop_reason → neutral finish_reason (stop reasons:
# https://platform.claude.com/docs/en/build-with-claude/handling-stop-reasons.md).
# model_context_window_exceeded is a truncation, like max_tokens.
function _anthropic_finish_reason(stop_reason)
    stop_reason in ("end_turn", "stop_sequence")                  ? STOP :
    stop_reason == "tool_use"                                     ? TOOL_CALLS :
    stop_reason in ("max_tokens", "model_context_window_exceeded") ? "length" :
    stop_reason == "refusal"                                      ? CONTENT_FILTER :
    something(stop_reason, STOP)
end

# A refusal's stop_details.explanation is human-readable text and may be null
# (https://platform.claude.com/docs/en/build-with-claude/refusals-and-fallback.md).
function _anthropic_refusal_text(stop_details)::String
    x = stop_details isa AbstractDict ? get(stop_details, "explanation", nothing) : nothing
    x isa AbstractString && !isempty(x) ? x : "Model refused to respond."
end

# Anthropic usage → neutral TokenUsage. `input_tokens` is the uncached remainder;
# cache reads and cache writes are reported beside it. The neutral model counts all
# input in `prompt_tokens` with `cached_tokens` the cache-read subset, so estimated_cost
# bills reads at the cached rate and writes at the base input rate (the cache-write
# premium is not modeled). `output_tokens_details.thinking_tokens` is the reasoning
# share of `output_tokens`. `prev` is the running stream total: message_delta counts
# are cumulative and carry the input side only on some streams, so a usage object
# without `input_tokens` keeps the message_start input counts.
function _anthropic_usage(u, prev::Union{TokenUsage,Nothing}=nothing)::Union{TokenUsage,Nothing}
    u isa AbstractDict || return prev
    has(k) = get(u, k, nothing) isa Integer
    n(k) = has(k) ? Int(u[k]) : 0
    base = something(prev, TokenUsage())
    input, cached = has("input_tokens") || isnothing(prev) ?
        (n("input_tokens") + n("cache_read_input_tokens") + n("cache_creation_input_tokens"),
         n("cache_read_input_tokens")) : (base.prompt_tokens, base.cached_tokens)
    out = has("output_tokens") ? n("output_tokens") : base.completion_tokens
    details = get(u, "output_tokens_details", nothing)
    thinking = details isa AbstractDict && get(details, "thinking_tokens", nothing) isa Integer ?
               Int(details["thinking_tokens"]) : base.reasoning_tokens
    TokenUsage(prompt_tokens=input, completion_tokens=out, total_tokens=input + out,
               cached_tokens=cached, reasoning_tokens=thinking)
end

function decode_response(::Type{ANTHROPICServiceEndpoint}, resp::HTTP.Response)
    data = JSON.parse(resp.body; dicttype=Dict{String,Any})
    # A 200 that is not a Message has no turn to report: fail loud, and the verb turns
    # the throw into its typed call error instead of an empty success.
    data isa AbstractDict && get(data, "content", nothing) isa AbstractVector &&
        get(data, "stop_reason", nothing) isa AbstractString ||
        error("Anthropic response is not a message with a content array and a stop_reason (got ",
              data isa AbstractDict ? "keys " * join(sort!(collect(keys(data))), ", ") : typeof(data), ")")
    blocks = data["content"]
    finish = _anthropic_finish_reason(data["stop_reason"])
    usage = _anthropic_usage(get(data, "usage", nothing))
    # Output before a refusal is incomplete and is discarded (Anthropic's guidance), so
    # the turn carries only the explanation — the same turn the stream handler builds.
    finish == CONTENT_FILTER && return (; message=Message(role=RoleAssistant, finish_reason=finish,
        refusal_message=_anthropic_refusal_text(get(data, "stop_details", nothing))), usage)
    # Verbatim capture for round-trip: thinking/redacted_thinking signatures
    # must be echoed unmodified on the next turn (thinking models reject
    # modified blocks). Empty arrays are not captured — echoing [] back is a 400.
    pc = isempty(blocks) ? nothing : ProviderContent(:anthropic, blocks)
    text = IOBuffer()
    tool_calls = ToolCall[]
    for b in blocks
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
    txt = takestring!(text)
    msg = if !isempty(tool_calls)
        Message(role=RoleAssistant, content=(isempty(txt) ? nothing : txt),
                tool_calls=tool_calls, finish_reason=finish, provider_content=pc)
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

# In-flight block fields accumulate in IOBuffers held by the pending block and
# become Strings once, when the block stops: `*` on a growing String copies the
# whole prefix per delta, which is quadratic in the number of deltas.
function _anthropic_buf!(blk::Dict{String,Any}, key::String)::IOBuffer
    v = get(blk, key, "")
    v isa IOBuffer && return v
    buf = IOBuffer()
    v isa AbstractString && print(buf, v)   # a start snapshot's text; a tool's input object is replaced
    blk[key] = buf
end

# The driver reads a streamed tool call's arguments once a later tool block starts or
# the message ends, even if its own stop line never arrived (a dropped line): publish
# the partial JSON accumulated so far without finalizing the block.
function _anthropic_publish_tool_args!(state::StreamState)
    for (idx, blk) in state.raw_pending
        buf = get(blk, "input", nothing)
        buf isa IOBuffer && haskey(state.tool_calls, idx) &&
            (state.tool_calls[idx]["function"]["arguments"] = takestring!(copy(buf)))
    end
end

# A refusal can follow partial output, which Anthropic says to discard as incomplete:
# drop the streamed text (including deltas not yet forwarded), tool calls and captured
# blocks, so the assembled turn matches the non-streaming decode.
function _anthropic_stream_refusal!(state::StreamState, stop_details)
    take!(state.content); take!(state.pending_delta); take!(state.refusal)
    empty!(state.tool_calls); empty!(state.raw_blocks); empty!(state.raw_pending)
    print(state.refusal, _anthropic_refusal_text(stop_details))
    nothing
end

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
            _anthropic_publish_tool_args!(state)
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
            isnothing(blk) || print(_anthropic_buf!(blk, "text"), txt::AbstractString)
        elseif dt == "input_json_delta"
            # One partial-JSON buffer per block feeds both the raw block's input and
            # the neutral tool-call arguments (published at stop).
            isnothing(blk) || print(_anthropic_buf!(blk, "input"), get(d, "partial_json", "")::AbstractString)
        elseif dt == "thinking_delta"
            isnothing(blk) || print(_anthropic_buf!(blk, "thinking"), get(d, "thinking", "")::AbstractString)
        elseif dt == "signature_delta"
            isnothing(blk) || print(_anthropic_buf!(blk, "signature"), get(d, "signature", "")::AbstractString)
        end
    elseif t == "content_block_stop"
        idx = get(ev, "index", nothing)
        idx isa Integer && haskey(state.tool_calls, idx) && (state.tool_calls[idx]["complete"] = true)
        if idx isa Integer && haskey(state.raw_pending, idx)
            blk = state.raw_pending[idx]
            for key in findall(v -> v isa IOBuffer, blk)
                blk[key] = s = takestring!(blk[key])
                key == "input" || continue
                # Streamed tool input arrives as partial JSON: finalize to a parsed
                # object so the block matches the non-streaming wire shape. The
                # arguments are published first, so an undecodable input still
                # reaches the tool call (whose own parse reports it).
                haskey(state.tool_calls, idx) && (state.tool_calls[idx]["function"]["arguments"] = s)
                blk[key] = _parse_tool_arguments(s)
            end
            push!(state.raw_blocks, blk)
            delete!(state.raw_pending, idx)
        end
    elseif t == "message_delta"
        _anthropic_publish_tool_args!(state)
        d = get(ev, "delta", Dict{String,Any}())
        sr = get(d, "stop_reason", nothing)
        isnothing(sr) || (state.finish_reason = _anthropic_finish_reason(sr))
        # stop_details arrives on message_delta alongside stop_reason.
        sr == "refusal" && _anthropic_stream_refusal!(state, get(d, "stop_details", nothing))
        state.usage = _anthropic_usage(get(ev, "usage", nothing), state.usage)
    elseif t == "message_stop"
        _anthropic_publish_tool_args!(state)
        return :done
    end
    :continue
end
