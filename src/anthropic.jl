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
