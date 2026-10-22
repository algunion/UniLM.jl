# ============================================================================
# Anthropic (Claude) native Messages API
# Plugs into the wire-translation seam (encode_request / decode_response /
# handle_sse_event! from sse.jl) so all chat orchestration is shared.
# Wire shape and model contracts verified against the Anthropic docs on 2026-10-22.
# ============================================================================

# ─── Routing & auth ──────────────────────────────────────────────────────────

# The bare type and an instance are one endpoint: every Anthropic method takes either, and
# reads the options through `_anthropic_endpoint` (the type is the endpoint with none set).
const _AnthropicEndpoint = Union{Type{ANTHROPICServiceEndpoint},ANTHROPICServiceEndpoint}
_anthropic_endpoint(::Type{ANTHROPICServiceEndpoint})::ANTHROPICServiceEndpoint = ANTHROPICServiceEndpoint()
_anthropic_endpoint(e::ANTHROPICServiceEndpoint)::ANTHROPICServiceEndpoint = e

get_url(::_AnthropicEndpoint, ::Chat)::String = ANTHROPIC_BASE_URL * ANTHROPIC_MESSAGES_PATH

# What every request through the endpoint sends: the key and the API version, then its user
# profile and its workspace (https://platform.claude.com/docs/en/api/messages, Headers).
# `profile=false` leaves the user profile out, for an API whose reference names no such header.
function _anthropic_base_headers(e::ANTHROPICServiceEndpoint; profile::Bool=true)::Vector{Pair{String,String}}
    headers = ["x-api-key" => ENV[ANTHROPIC_API_KEY], "anthropic-version" => ANTHROPIC_VERSION]
    profile && !isnothing(e.user_profile_id) && push!(headers, "anthropic-user-profile-id" => e.user_profile_id)
    isnothing(e.workspace_id) || push!(headers, "anthropic-workspace-id" => e.workspace_id)
    headers
end

# The base headers, the body's type, and the endpoint's beta names in one header, comma-joined
# (https://platform.claude.com/docs/en/api/beta-headers). The beta names are checked again here,
# by the constructor's rule: `betas` is a vector, which can be changed after the endpoint is
# built (the profile and workspace ids are strings, which cannot).
function auth_header(service::_AnthropicEndpoint)::Vector{Pair{String,String}}
    e = _anthropic_endpoint(service)
    headers = push!(_anthropic_base_headers(e), "Content-Type" => "application/json")
    isempty(e.betas) || push!(headers, "anthropic-beta" => join(_anthropic_beta_names(e.betas), ","))
    headers
end

# ─── Capabilities & defaults ─────────────────────────────────────────────────

provider_capabilities(::_AnthropicEndpoint) = Set([:chat, :tools, :json_output, :streaming, :models])

default_model(::_AnthropicEndpoint) = "claude-opus-5-5"

"""
    default_max_tokens(service, model::AbstractString) -> Int

`max_tokens` supplied when the caller leaves it unset. Anthropic *requires* the
field; OpenAI does not. Returns a moderate, overridable default (see
`_ANTHROPIC_DEFAULT_MAX_TOKENS`), not the model ceiling.
"""
default_max_tokens(::_AnthropicEndpoint, ::AbstractString) = _ANTHROPIC_DEFAULT_MAX_TOKENS

# ─── Claude models ───────────────────────────────────────────────────────────
# Request contract per model, transcribed on 2026-10-22 from
#   thinking modes and rejected configs: https://platform.claude.com/docs/en/build-with-claude/thinking.md
#   effort levels per model:             https://platform.claude.com/docs/en/build-with-claude/effort.md
#   prefill, forced tools:               thinking.md "Limits and feature compatibility",
#                                        https://platform.claude.com/docs/en/models/sonnet-5-5/migration-guide.md,
#                                        https://platform.claude.com/docs/en/models/haiku-5-5/migration-guide.md
#   mid-conversation system messages:    https://platform.claude.com/docs/en/build-with-claude/mid-conversation-system-messages.md
#   the thinking-block prefix check:     https://platform.claude.com/docs/en/build-with-claude/preserved-thinking.md
#   fast mode:                           https://platform.claude.com/docs/en/build-with-claude/fast-mode.md
#   inference_geo:                       https://platform.claude.com/docs/en/manage-claude/data-residency.md
# `thinking`: :always = adaptive, cannot be disabled; :on = adaptive by default, "disabled"
# turns it off; :between = adaptive by default and cannot be disabled, "between_tools" turns
# up-front thinking off; :off = adaptive on request; :manual = extended thinking
# (budget_tokens) only.
# `efforts`: accepted output_config.effort levels (empty = no effort parameter).
# `forced`: tool_choice any/tool accepted. `prefill`: a trailing assistant turn accepted.
# `system`: role "system" messages after the conversation starts accepted.
# `prefix_checked`: a thinking block is valid only while the top-level system prompt, the
# tools and the messages before it are unchanged (see `_anthropic_messages`); such a model
# accepts mid-conversation system messages, which is how its instructions change.
# `fast`: speed "fast" (fast mode) — :ga served; :beta served in research preview, only with
# the `_ANTHROPIC_FAST_BETA` header (HTTP 400 without it); :none not served (an error, except
# on Opus 4.6, which runs the request at standard speed).
# `inference_geo`: the request's inference_geo accepted (HTTP 400 on the 4.5 models and earlier).
const _CLAUDE_EFFORTS = ["low", "medium", "high", "xhigh", "max"]
const _CLAUDE_EFFORTS_NO_XHIGH = ["low", "medium", "high", "max"]
const _CLAUDE_THINKING = (:always, :on, :between, :off, :manual)
const _CLAUDE_FAST = (:ga, :beta, :none)

@kwdef struct _ClaudeFamily
    thinking::Symbol
    efforts::Vector{String} = _CLAUDE_EFFORTS
    forced::Bool = true
    prefill::Bool = false
    system::Bool = false
    prefix_checked::Bool = false
    fast::Symbol = :none
    inference_geo::Bool = true
    function _ClaudeFamily(thinking, efforts, forced, prefill, system, prefix_checked, fast, inference_geo)
        thinking in _CLAUDE_THINKING || throw(ArgumentError(
            "Claude thinking mode must be one of $(join(_CLAUDE_THINKING, ", ")) (got $(repr(thinking)))"))
        !prefix_checked || system || throw(ArgumentError(
            "a prefix-checked Claude model must accept mid-conversation system messages (system=true)"))
        fast in _CLAUDE_FAST || throw(ArgumentError(
            "Claude fast mode must be one of $(join(_CLAUDE_FAST, ", ")) (got $(repr(fast)))"))
        new(thinking, efforts, forced, prefill, system, prefix_checked, fast, inference_geo)
    end
end

const _CLAUDE_LEGACY = _ClaudeFamily(thinking=:manual, efforts=String[], prefill=true, inference_geo=false)

# Dates: https://platform.claude.com/docs/en/about-claude/model-deprecations.md
const _CLAUDE_FAMILIES = Dict{String,_ClaudeFamily}(
    "claude-fable-5-1"      => _ClaudeFamily(thinking=:always, forced=false, system=true, prefix_checked=true),
    "claude-mythos-5-1"     => _ClaudeFamily(thinking=:always, forced=false, system=true),
    "claude-fable-5"        => _ClaudeFamily(thinking=:always, system=true),
    "claude-mythos-5"       => _ClaudeFamily(thinking=:always, system=true),
    "claude-opus-5-5"       => _ClaudeFamily(thinking=:always, forced=false, system=true, prefix_checked=true, fast=:ga),
    "claude-sonnet-5-5"     => _ClaudeFamily(thinking=:between, forced=false, system=true, prefix_checked=true),
    "claude-haiku-5-5"      => _ClaudeFamily(thinking=:on, system=true, prefix_checked=true),
    "claude-opus-5"         => _ClaudeFamily(thinking=:on, system=true, fast=:beta),
    "claude-sonnet-5"       => _ClaudeFamily(thinking=:on),
    "claude-opus-4-8"       => _ClaudeFamily(thinking=:off, system=true, fast=:beta),
    "claude-opus-4-7"       => _ClaudeFamily(thinking=:off),
    "claude-opus-4-6"       => _ClaudeFamily(thinking=:off, efforts=_CLAUDE_EFFORTS_NO_XHIGH),
    "claude-sonnet-4-6"     => _ClaudeFamily(thinking=:off, efforts=_CLAUDE_EFFORTS_NO_XHIGH),
    # Retire not sooner than 2026-11-24 (Opus 4.5) and 2026-10-15 (Haiku 4.5).
    "claude-opus-4-5"       => _ClaudeFamily(thinking=:manual, efforts=["low", "medium", "high"], prefill=true,
                                             inference_geo=false),
    "claude-haiku-4-5"      => _CLAUDE_LEGACY,
    # Deprecated 2026-06-09, retirement to be announced.
    "claude-mythos-preview" => _ClaudeFamily(thinking=:always, efforts=_CLAUDE_EFFORTS_NO_XHIGH),
    # Deprecated 2026-09-30, retires 2026-11-30.
    "claude-sonnet-4-5"     => _CLAUDE_LEGACY,
)

# The row for `model` in a per-model table: the exact id, else the id without a dated
# snapshot suffix (-YYYYMMDD: claude-haiku-4-5-20251001 → claude-haiku-4-5), else `nothing`.
# No id inherits another's row through a shared prefix (claude-sonnet-5-5 is not
# claude-sonnet-5), and an id no row names — a newer model, a near miss — gets no
# per-model rule: the API answers for it.
_claude_row(table::Dict{String,T}, model::AbstractString) where {T} =
    get(() -> get(table, replace(model, r"-\d{8}$" => ""), nothing), table, model)

_claude_family(model::AbstractString)::Union{_ClaudeFamily,Nothing} = _claude_row(_CLAUDE_FAMILIES, model)

# reasoning_effort → (thinking, output_config.effort). A level also turns adaptive
# thinking on where the model has it, so an explicit request reasons on the models that
# run without thinking by default, and asks for its summary: most current models return
# thinking blocks with the text omitted unless display is "summarized"
# (https://platform.claude.com/docs/en/build-with-claude/thinking.md#controlling-thinking-display).
# "none" turns thinking off and sends no effort.
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
    effort == "none" && return (_anthropic_thinking_off(fam), nothing)
    ((isnothing(fam) || fam.thinking !== :manual) ? Dict(:type => "adaptive", :display => "summarized") : nothing, effort)
end

# The thinking value for reasoning_effort "none": "disabled" where thinking is on by
# default (and for an id no row names); "between_tools", the lowest setting, where
# "disabled" is a 400 — accepted at the default effort, hence no effort with it; nothing
# where thinking is off by default.
_anthropic_thinking_off(::Nothing) = Dict(:type => "disabled")
_anthropic_thinking_off(fam::_ClaudeFamily) =
    fam.thinking === :on      ? Dict(:type => "disabled") :
    fam.thinking === :between ? Dict(:type => "between_tools") : nothing

# ─── Request encoding (neutral Chat → Anthropic Messages body) ───────────────

# Fail closed when a neutral option has no Messages API counterpart: dropping it
# would send a request the caller did not ask for.
const _ANTHROPIC_CHAT_MAPPED_FIELDS = (:service, :model, :messages, :history, :tools,
    :tool_choice, :parallel_tool_calls, :n, :stream, :stop,
    :max_tokens, :max_completion_tokens, :response_format, :user, :reasoning_effort,
    :metadata, :service_tier, :safety_identifier, :prompt_cache_options, :_cumulative_cost)
const _ANTHROPIC_CHAT_UNMAPPED_FIELDS = Tuple(setdiff(fieldnames(Chat), _ANTHROPIC_CHAT_MAPPED_FIELDS))

# service_tier: the Messages API's own "auto" and "standard_only"
# (https://platform.claude.com/docs/en/api/messages.md), and "fast", which is not a tier there
# but fast mode, the request's speed (see `_ClaudeFamily`).
const _ANTHROPIC_SERVICE_TIERS = ("auto", "standard_only", "fast")

function _anthropic_validate_fields(chat::Chat)
    isnothing(chat.moderation) || throw(ArgumentError(
        "Anthropic Messages does not support moderation; it is an OpenAI-only option"))
    # API version 2026-10-21 removed the sampling parameters: the API answers HTTP 400 for
    # one on every model, even at its default value ("Sampling parameters are removed",
    # https://platform.claude.com/docs/en/api/migration-2026-10-21#sampling-parameters).
    for f in (:temperature, :top_p)
        isnothing(getfield(chat, f)) || throw(ArgumentError(
            "Anthropic API version $ANTHROPIC_VERSION has no $f: the Messages API answers HTTP 400 " *
            "for it on every model, even at its default value; remove $f"))
    end
    isnothing(chat.n) || chat.n == 1 || throw(ArgumentError(
        "Anthropic Messages returns one completion; n must be 1"))
    fields = Symbol[f for f in _ANTHROPIC_CHAT_UNMAPPED_FIELDS if !isnothing(getfield(chat, f))]
    isempty(fields) || throw(ArgumentError(
        "Anthropic Messages does not support field(s) $(join(fields, ", "))"))
    isnothing(chat.service_tier) || chat.service_tier in _ANTHROPIC_SERVICE_TIERS || throw(ArgumentError(
        "Anthropic service_tier must be $(join(repr.(_ANTHROPIC_SERVICE_TIERS), ", ", " or ")) " *
        "(got $(repr(chat.service_tier)))"))
    nothing
end

# prompt_cache_options → a top-level cache_control, Anthropic's automatic caching: the API
# sets the breakpoint on the last cacheable block, for 5 minutes unless ttl is "1h"
# (https://platform.claude.com/docs/en/build-with-claude/prompt-caching.md#automatic-caching).
# A prompt shorter than the model's minimum (512 tokens on the 5.5 models) is not cached and
# the API says nothing: the usage reports no cache write. prewarm=false asks for what an unset
# prewarm sends.
const _ANTHROPIC_CACHE_TTLS = ("5m", "1h")
_anthropic_cache_control(::Nothing) = nothing
function _anthropic_cache_control(o::PromptCacheOptions)::Dict{Symbol,String}
    refuse(why) = throw(ArgumentError("Anthropic prompt caching takes prompt_cache_options mode=\"implicit\" " *
        "(or unset) and ttl=$(join(repr.(_ANTHROPIC_CACHE_TTLS), " or ")) (or unset, 5 minutes): $why"))
    o.mode == "explicit" && refuse("mode=\"explicit\" caches only the breakpoints a request marks, and a Chat marks none")
    o.prewarm === true && refuse("prewarm=true is not sent; the Messages API pre-warms a cache with max_tokens 0, " *
                                 "which a Chat does not take")
    isnothing(o.comparison_response_id) || refuse("comparison_response_id is not sent; Anthropic's cache " *
                                                  "diagnostics are not mapped")
    isnothing(o.ttl) || o.ttl in _ANTHROPIC_CACHE_TTLS || refuse("got ttl=$(repr(o.ttl))")
    cc = Dict{Symbol,String}(:type => "ephemeral")
    isnothing(o.ttl) || (cc[:ttl] = o.ttl)
    cc
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

# Requests a Claude model answers with HTTP 400 (or runs other than asked), rejected before
# the round trip (the per-model limits are in the table above). An id no row names gets none
# of these checks: the API answers for it.
_anthropic_validate_model(::Chat, ::Nothing, ::ANTHROPICServiceEndpoint) = nothing
function _anthropic_validate_model(chat::Chat, fam::_ClaudeFamily, e::ANTHROPICServiceEndpoint)
    fam.forced || !(chat.tool_choice isa GPTToolChoice || chat.tool_choice == "required") ||
        throw(ArgumentError("$(chat.model) rejects forced tool_choice; use tool_choice=\"auto\" with strict " *
                            "tools (FunctionSignature(strict=true)) or a json_schema response_format"))
    chat.service_tier == "fast" && _check_claude_fast(chat.model, fam, e)
    isnothing(e.inference_geo) || fam.inference_geo || throw(ArgumentError(
        "$(chat.model) takes no inference_geo: the API answers HTTP 400 for it on the Claude 4.5 models and " *
        "earlier; send the request through an ANTHROPICServiceEndpoint without inference_geo"))
    nothing
end

const _ANTHROPIC_FAST_BETA = "fast-mode-2026-02-01"
const _CLAUDE_FAST_MODELS = sort!([m for (m, row) in _CLAUDE_FAMILIES if row.fast === :ga])
const _CLAUDE_FAST_PREVIEW_MODELS = sort!([m for (m, row) in _CLAUDE_FAMILIES if row.fast === :beta])

function _check_claude_fast(model::String, fam::_ClaudeFamily, e::ANTHROPICServiceEndpoint)::Nothing
    fam.fast === :none && throw(ArgumentError("$model has no fast mode (service_tier=\"fast\"): Anthropic " *
        "serves it on $(join(_CLAUDE_FAST_MODELS, ", ")), and in research preview on " *
        "$(join(_CLAUDE_FAST_PREVIEW_MODELS, " and ")) with the $_ANTHROPIC_FAST_BETA beta header"))
    fam.fast === :beta && !(_ANTHROPIC_FAST_BETA in e.betas) && throw(ArgumentError(
        "$model serves fast mode (service_tier=\"fast\") in research preview, only with the " *
        "$_ANTHROPIC_FAST_BETA beta header (the API answers HTTP 400 without it): send the request through " *
        "ANTHROPICServiceEndpoint(betas=[\"$_ANTHROPIC_FAST_BETA\"])"))
    nothing
end

# Where a system message after the conversation starts goes (see `_anthropic_messages`):
# _HOIST joins every one into the top-level `system`; _IN_PLACE keeps one where the API
# accepts it and hoists the rest; _APPEND_ONLY never edits the top-level `system` once an
# assistant turn is sent. An id no row names is placed append-only: Anthropic asks every
# integration to be append-only whatever the account
# (https://platform.claude.com/docs/en/build-with-claude/preserved-thinking.md), and a
# newer model may check thinking blocks.
@enum _SystemPlacement _HOIST _IN_PLACE _APPEND_ONLY
_system_placement(::Nothing) = _APPEND_ONLY
_system_placement(fam::_ClaudeFamily) = fam.prefix_checked ? _APPEND_ONLY : fam.system ? _IN_PLACE : _HOIST

# A system message after an assistant turn with no user turn after it has no place: the
# Messages API takes one only right after a user turn, and joined into the top-level
# `system` it would end the request with the assistant turn — a response prefill. The user
# turn that comes next is a user message, or the results of the reply's tool calls.
function _system_after_reply(model::AbstractString, reply::Message)
    fix = isnothing(reply.tool_calls) || isempty(reply.tool_calls) ? "push a user message after the system message" :
        "push the results of its tool calls (role \"tool\" messages, which the API takes as a user turn) after " *
        "the system message"
    ArgumentError("$model takes a system message after an assistant turn only once a user turn follows it: " *
        "the Messages API accepts one only right after a user turn, and in the top-level system prompt " *
        "it would leave the request ending with the assistant turn, a response prefill; $fix")
end

# A request that ends with an assistant turn: no prefill when the last message it sends is a
# system message after that reply — turns left out after the system message (a refusal, an
# empty reply) do not make it one — and refused on a model that takes no prefill.
function _anthropic_check_prefill(chat::Chat, fam::Union{_ClaudeFamily,Nothing})::Nothing
    sent(m) = !_skipped_turn(m)
    chat.messages[something(findlast(sent, chat.messages))].role == RoleSystem && throw(_system_after_reply(
        chat.model, chat.messages[something(findlast(m -> m.role == RoleAssistant && sent(m), chat.messages))]))
    isnothing(fam) || fam.prefill || throw(ArgumentError("$(chat.model) rejects messages that end with an " *
        "assistant turn (response prefill); end with a user turn or use a json_schema response_format"))
    nothing
end

function encode_request(service::_AnthropicEndpoint, chat::Chat)
    _anthropic_validate_fields(chat)
    endpoint = _anthropic_endpoint(service)
    fam = _claude_family(chat.model)
    _anthropic_validate_model(chat, fam, endpoint)
    body = Dict{Symbol,Any}(:model => chat.model)
    # max_tokens is REQUIRED by Anthropic; fall back to the moderate default.
    body[:max_tokens] = something(chat.max_completion_tokens, chat.max_tokens, default_max_tokens(service, chat.model))
    system, msgs = _anthropic_messages(chat.messages; placement=_system_placement(fam), model=chat.model)
    !isempty(msgs) && msgs[end][:role] == "assistant" && _anthropic_check_prefill(chat, fam)
    isnothing(system) || (body[:system] = system)
    body[:messages] = msgs
    isnothing(chat.tools)       || (body[:tools] = [_anthropic_tool(t) for t in chat.tools])
    disable_parallel = !isnothing(chat.tools) && !isempty(chat.tools) && chat.parallel_tool_calls === false
    (!isnothing(chat.tool_choice) || disable_parallel) &&
        (body[:tool_choice] = _anthropic_tool_choice(something(chat.tool_choice, "auto"), disable_parallel))
    isnothing(chat.stop)        || (body[:stop_sequences] = chat.stop isa String ? [chat.stop] : chat.stop)
    (uid = _anthropic_user_id(chat)) === nothing || (body[:metadata] = Dict(:user_id => uid))
    if chat.service_tier == "fast"
        body[:speed] = "fast"          # fast mode is the request's speed, not a service tier
    elseif !isnothing(chat.service_tier)
        body[:service_tier] = chat.service_tier
    end
    isnothing(endpoint.inference_geo) || (body[:inference_geo] = endpoint.inference_geo)
    (cache = _anthropic_cache_control(chat.prompt_cache_options)) === nothing || (body[:cache_control] = cache)
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
# - leading system messages join the top-level `system`. A later one goes where
#   `placement` says (`_system_placement`): in place as role "system" where the API
#   accepts one (`_mid_system_fits`), unless the model has none (_HOIST); else into the
#   top-level `system` — except under _APPEND_ONLY once an assistant turn is sent, where
#   editing that prompt invalidates the thinking blocks of every later turn (an HTTP 400):
#   there it moves forward to right after the next user turn, so each request extends the
#   one before, and with no user turn after it the request is refused
# - consecutive `tool` messages → collapsed into ONE user message of tool_result blocks
# - assistant tool_calls → tool_use blocks; a tool_result referencing an id no
#   preceding assistant emitted → loud ArgumentError
# - an assistant turn with nothing to send — no text, tool call or Anthropic block (a
#   refusal, an empty reply) — is skipped: the API rejects empty assistant content.
# - a user turn's attachments → image and document blocks (`_anthropic_user_content`)
function _anthropic_messages(messages; placement::_SystemPlacement=_IN_PLACE, model::AbstractString="")
    system = nothing
    out = Vector{Dict{Symbol,Any}}()
    seen_tool_use_ids = Set{String}()
    pending = Vector{Dict{Symbol,Any}}()
    held = Message[]          # later system messages waiting for the next user turn
    reply = nothing           # the last assistant turn sent
    flush!() = (isempty(pending) ||
        (push!(out, Dict{Symbol,Any}(:role => "user", :content => copy(pending))); empty!(pending)))
    place!(m) = push!(out, Dict{Symbol,Any}(:role => "system", :content => something(m.content, "")))
    release!() = (foreach(place!, held); empty!(held))
    for (i, m) in pairs(messages)
        if m.role == RoleSystem && placement != _HOIST && _mid_system_fits(messages, i, out, pending)
            flush!()
            release!()
            place!(m)
        elseif m.role == RoleSystem && placement == _APPEND_ONLY && !isnothing(reply)
            push!(held, m)
        elseif m.role == RoleSystem
            system = isnothing(system) ? m.content : string(system, "\n\n", something(m.content, ""))
        elseif m.role == RoleTool
            tcid = something(m.tool_call_id, "")
            tcid in seen_tool_use_ids || throw(ArgumentError(
                "tool_result references unknown tool_use id $(repr(tcid)); no preceding assistant tool_use emitted it"))
            push!(pending, Dict{Symbol,Any}(:type => "tool_result",
                :tool_use_id => tcid, :content => something(m.content, "")))
        elseif m.role == RoleAssistant
            flush!()
            _skipped_turn(m) && continue
            isempty(held) || out[end][:role] != "user" || release!()
            isnothing(m.tool_calls) || foreach(tc -> push!(seen_tool_use_ids, tc.id), m.tool_calls)
            push!(out, Dict{Symbol,Any}(:role => "assistant", :content => _anthropic_assistant_content(m)))
            reply = m
        else  # RoleUser
            flush!()
            push!(out, Dict{Symbol,Any}(:role => "user", :content => _anthropic_user_content(m)))
        end
    end
    flush!()
    isempty(held) || out[end][:role] == "user" || throw(_system_after_reply(model, something(reply)))
    release!()
    (system, out)
end

# A user turn's content: its text, or with attachments their blocks and then the text — the
# order Anthropic's vision and PDF guides recommend (media before the question,
# https://platform.claude.com/docs/en/build-with-claude/vision.md,
# https://platform.claude.com/docs/en/build-with-claude/pdf-support.md). Empty text is left
# out: the API refuses an empty text block.
function _anthropic_user_content(m::Message)
    isnothing(m.attachments) && return something(m.content, "")
    blocks = Dict{Symbol,Any}[_anthropic_block(a) for a in m.attachments]
    isempty(something(m.content, "")) || push!(blocks, Dict{Symbol,Any}(:type => "text", :text => m.content))
    blocks
end

# The base64 image (JPEG, PNG, GIF, WebP — the formats an ImageAttachment holds) and PDF
# document blocks (https://platform.claude.com/docs/en/api/messages.md). The image block has
# no detail level and the Messages API no audio input: either is refused rather than dropped.
_anthropic_source(a::Union{ImageAttachment,DocumentAttachment}) =
    Dict{Symbol,Any}(:type => "base64", :media_type => a.mime, :data => base64encode(a.data))
function _anthropic_block(a::ImageAttachment)::Dict{Symbol,Any}
    isnothing(a.detail) || throw(ArgumentError("an Anthropic image block has no detail level: build the " *
        "ImageAttachment without `detail` (got $(repr(a.detail)))"))
    Dict{Symbol,Any}(:type => "image", :source => _anthropic_source(a))
end
_anthropic_block(a::DocumentAttachment) = Dict{Symbol,Any}(:type => "document", :source => _anthropic_source(a))
_anthropic_block(a::AudioAttachment) = throw(ArgumentError("Anthropic Messages takes images and PDF documents, " *
    "not audio ($(a)): send the clip through an OpenAI-compatible endpoint, or its transcript in the message"))

_skipped_turn(m::Message)::Bool = m.role == RoleAssistant &&
    isempty(something(m.content, "")) && (isnothing(m.tool_calls) || isempty(m.tool_calls)) &&
    isnothing(_anthropic_blocks(m))

# The provider-native blocks of an assistant turn this provider produced, echoed as
# returned (see `_anthropic_assistant_content`), else `nothing`. A turn cut at max_tokens
# that holds only thinking blocks goes back too: to continue it, "pass the assistant turn
# back unchanged and append a new user message"
# (https://platform.claude.com/docs/en/build-with-claude/thinking.md). An empty text block
# (a streamed one that got no delta) is never sent — the API rejects empty text content —
# so a turn of nothing else has nothing to send.
function _anthropic_blocks(m::Message)::Union{Vector{Any},Nothing}
    pc = m.provider_content
    pc isa ProviderContent || return nothing
    pc.provider === :anthropic || return nothing
    blocks = any(_empty_text_block, pc.blocks) ? filter(!_empty_text_block, pc.blocks) : pc.blocks
    isempty(blocks) ? nothing : blocks
end

_empty_text_block(b)::Bool = b isa AbstractDict && get(b, "type", nothing) == "text" && get(b, "text", "") == ""

# Whether `messages[i]`, a system message, may stay in place: the API accepts a
# mid-conversation system message right after a user turn (tool results are one) that is
# either the last entry or right before an assistant turn, and a run of them as one
# (https://platform.claude.com/docs/en/build-with-claude/mid-conversation-system-messages.md).
# `out` and `pending` are the turns built so far.
function _mid_system_fits(messages, i::Int, out, pending)::Bool
    isempty(pending) && (isempty(out) || out[end][:role] ∉ ("user", "system")) && return false
    next = findnext(m -> m.role != RoleSystem && !_skipped_turn(m), messages, i + 1)
    isnothing(next) || messages[next].role == RoleAssistant
end

# Assistant turn → Anthropic content: echo captured provider-native blocks
# verbatim when this provider produced them (signatures intact, thinking
# first); otherwise reconstruct optional text block + tool_use blocks.
function _anthropic_assistant_content(m::Message)
    blocks = _anthropic_blocks(m)
    isnothing(blocks) || return blocks
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

# Anthropic usage → neutral TokenUsage (https://platform.claude.com/docs/en/api/messages.md,
# Usage). `input_tokens` is the uncached remainder; cache reads and cache writes are
# reported beside it, and `cache_creation` splits the writes by retention. The neutral
# counts put all input in `prompt_tokens`, the reads in `cached_tokens`, the writes in
# `cache_write_tokens` and the 1-hour ones in `cache_write_extended_tokens`.
# `output_tokens_details.thinking_tokens` is the reasoning share of `output_tokens`. The
# tier is "fast" for a call that ran in fast mode (`speed`), which bills its own rates,
# else the reported `service_tier`; `inference_geo` is the region inference ran in.
# `prev` is the running stream total: message_delta counts are cumulative and carry the
# input side only on some streams, and never the write breakdown, tier or region, so a
# usage object keeps the earlier value of whatever it does not report.
function _anthropic_usage(u, prev::Union{TokenUsage,Nothing}=nothing)::Union{TokenUsage,Nothing}
    u isa AbstractDict || return prev
    has(k) = get(u, k, nothing) isa Integer
    n(k) = has(k) ? Int(u[k]) : 0
    base = something(prev, TokenUsage())
    input, cached, writes = has("input_tokens") || isnothing(prev) ?
        (n("input_tokens") + n("cache_read_input_tokens") + n("cache_creation_input_tokens"),
         n("cache_read_input_tokens"), n("cache_creation_input_tokens")) :
        (base.prompt_tokens, base.cached_tokens, base.cache_write_tokens)
    out = has("output_tokens") ? n("output_tokens") : base.completion_tokens
    details = get(u, "output_tokens_details", nothing)
    thinking = details isa AbstractDict && get(details, "thinking_tokens", nothing) isa Integer ?
               Int(details["thinking_tokens"]) : base.reasoning_tokens
    creation = get(u, "cache_creation", nothing)
    extended = creation isa AbstractDict && get(creation, "ephemeral_1h_input_tokens", nothing) isa Integer ?
               Int(creation["ephemeral_1h_input_tokens"]) : base.cache_write_extended_tokens
    tier = _wire_string(u, "speed") == "fast" ? "fast" : _wire_string(u, "service_tier")
    geo = _wire_string(u, "inference_geo")
    TokenUsage(prompt_tokens=input, completion_tokens=out, total_tokens=input + out,
               cached_tokens=cached, reasoning_tokens=thinking, cache_write_tokens=writes,
               cache_write_extended_tokens=extended,
               service_tier=isnothing(tier) ? base.service_tier : tier,
               inference_geo=isnothing(geo) ? base.inference_geo : geo)
end

# ─── Pricing ─────────────────────────────────────────────────────────────────
# What Anthropic bills on top of a model's price row
# (https://platform.claude.com/docs/en/about-claude/pricing.md): cache writes at the input
# rate × 1.25 (5-minute) or × 2 (1-hour); fast mode at every rate × 2 (Opus 5.5 $8/$40 for
# $4/$20, Opus 5 and Opus 4.8 $10/$50 for $5/$25); US-only inference at every rate × 1.1 on
# Claude 4.6 and later, the models whose row takes `inference_geo` ("Earlier models ... always
# use standard pricing"; an id no row names is taken for a newer model); a prompt over a
# model's long-prompt threshold — every input token, cache reads and writes included — at
# every rate × its multiplier. The multipliers stack. A reply at Priority Tier is billed at
# these rates too: Priority Tier draws down a contracted capacity commitment
# (https://platform.claude.com/docs/en/api/service-tiers.md), which no rate here models.
const _ANTHROPIC_CACHE_WRITE = 1.25
const _ANTHROPIC_CACHE_WRITE_1H = 2.0
const _ANTHROPIC_FAST = 2.0
const _ANTHROPIC_US_INFERENCE = 1.1
const _CLAUDE_LONG_PROMPT_TIERS = Dict{String,@NamedTuple{threshold::Int, multiplier::Float64}}(
    "claude-haiku-5-5" => (threshold=100_000, multiplier=5.0),
)

function _billed_cost(::_AnthropicEndpoint, model::String, u::TokenUsage, rates::PriceRow)::Float64
    cached = min(u.cached_tokens, u.prompt_tokens)        # cache reads, at the discounted rate
    fresh = u.prompt_tokens - cached - u.cache_write_tokens
    writes = (u.cache_write_tokens - u.cache_write_extended_tokens) * _ANTHROPIC_CACHE_WRITE +
             u.cache_write_extended_tokens * _ANTHROPIC_CACHE_WRITE_1H
    standard = fresh * rates.input + cached * rates.cached_input + writes * rates.input +
               u.completion_tokens * rates.output
    standard * _anthropic_rate_multiplier(model, u)
end

# The multiplier on every rate of the call `u` reports.
function _anthropic_rate_multiplier(model::String, u::TokenUsage)::Float64
    tier = _claude_row(_CLAUDE_LONG_PROMPT_TIERS, model)
    long = isnothing(tier) ? 1.0 : u.prompt_tokens > tier.threshold ? tier.multiplier : 1.0
    fam = _claude_family(model)
    us = u.inference_geo == "us" && (isnothing(fam) || fam.inference_geo)
    (u.service_tier == "fast" ? _ANTHROPIC_FAST : 1.0) * (us ? _ANTHROPIC_US_INFERENCE : 1.0) * long
end

function decode_response(::_AnthropicEndpoint, resp::HTTP.Response)
    data = JSON.parse(resp.body; dicttype=Dict{String,Any})
    # A 200 that is not a Message has no turn to report: fail loud, and the verb turns
    # the throw into its typed call error instead of an empty success.
    data isa AbstractDict && get(data, "content", nothing) isa AbstractVector &&
        get(data, "stop_reason", nothing) isa AbstractString ||
        error("Anthropic response is not a message with a content array and a stop_reason (got ",
              data isa AbstractDict ? "keys $(join(sort!(collect(keys(data))), ", "))" : "a $(typeof(data))", ")")
    blocks = data["content"]
    finish = _anthropic_finish_reason(data["stop_reason"])
    usage = _anthropic_usage(get(data, "usage", nothing))
    model = _wire_string(data, "model")
    # Output before a refusal is incomplete and is discarded (Anthropic's guidance), so
    # the turn carries only the explanation — the same turn the stream handler builds.
    finish == CONTENT_FILTER && return (; message=Message(role=RoleAssistant, finish_reason=finish,
        refusal_message=_anthropic_refusal_text(get(data, "stop_details", nothing))), usage, model)
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
    (; message=msg, usage, model)
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
# blocks, so the assembled turn matches the non-streaming decode. Deltas and tool calls
# already handed to the callbacks cannot be recalled.
function _anthropic_stream_refusal!(state::StreamState, stop_details)
    take!(state.content); take!(state.pending_delta); take!(state.refusal)
    empty!(state.tool_calls); empty!(state.raw_blocks); empty!(state.raw_pending)
    print(state.refusal, _anthropic_refusal_text(stop_details))
    nothing
end

function handle_sse_event!(::_AnthropicEndpoint, event::AbstractString,
                           payload::AbstractString, state::StreamState)::Symbol
    ev = JSON.parse(payload; dicttype=Dict{String,Any})
    ev isa AbstractDict || return :continue
    t = get(ev, "type", "")
    if event == "error" || t == "error"
        state.error = ev
        return :error
    elseif t == "message_start"
        msg = get(ev, "message", nothing)
        if msg isa AbstractDict
            u = get(msg, "usage", nothing)
            u isa AbstractDict && (state.usage = _anthropic_usage(u))
            model = _wire_string(msg, "model")      # the model serving this stream
            isnothing(model) || (state.model = model)
        end
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
                # arguments are published first, so an input that is no JSON object
                # still reaches the tool call, whose own decode decides: an error
                # under a tool_use stop, a dropped call on a turn cut at max_tokens
                # (which ends the input mid-JSON). Such a block stays pending: the
                # capture is incomplete, so the turn is rebuilt from its neutral fields.
                haskey(state.tool_calls, idx) && (state.tool_calls[idx]["function"]["arguments"] = s)
                _is_json_object(s) || return :continue
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
