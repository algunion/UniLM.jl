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
