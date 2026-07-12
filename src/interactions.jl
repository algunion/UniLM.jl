# ============================================================================
# Google Gemini Interactions API — native agentic surface for `respond`.
# https://ai.google.dev/gemini-api/docs/interactions-overview
#
# Server-stateful agentic API. Rides the same agentic seam as the OpenAI
# Responses path (src/responses.jl): overrides get_url / encode_agentic /
# decode_agentic / decode_agentic_stream for GEMINIServiceEndpoint, dispatched by
# request type (Respond) — exactly as OPENAIServiceEndpoint hosts both Chat and
# Respond. The decoder normalizes Interactions `steps[]` into the
# OpenAI-Responses-shaped `output[]`, so ResponseObject accessors
# (output_text / function_calls / …) are reused.
# ============================================================================

# ─── Routing (streaming is a BODY flag, so the URL is stream-independent) ─────

_agentic_url(::Type{GEMINIServiceEndpoint}) = GEMINI_NATIVE_BASE * INTERACTIONS_PATH

# ─── Request encoding (neutral Respond → Interactions body, snake_case) ───────

function encode_agentic(::Type{GEMINIServiceEndpoint}, r::Respond)::String
    body = Dict{Symbol,Any}(:model => r.model, :input => _interactions_input(r.input))
    isnothing(r.instructions) || (body[:system_instruction] = r.instructions)
    isnothing(r.tools) || (body[:tools] = [_interactions_tool(t) for t in r.tools])
    gen = Dict{Symbol,Any}()
    isnothing(r.temperature)       || (gen[:temperature] = r.temperature)
    isnothing(r.top_p)             || (gen[:top_p] = r.top_p)
    isnothing(r.max_output_tokens)  || (gen[:max_output_tokens] = r.max_output_tokens)
    isnothing(r.tool_choice)        || (gen[:tool_choice] = _interactions_tool_choice(r.tool_choice))
    isempty(gen) || (body[:generation_config] = gen)
    # Neutral continuation handle (previous_response_id) → Gemini's server-state id.
    isnothing(r.previous_response_id) || (body[:previous_interaction_id] = r.previous_response_id)
    isnothing(r.store)      || (body[:store] = r.store)
    isnothing(r.background) || (body[:background] = r.background)
    isnothing(r.stream)     || (body[:stream] = r.stream)
    JSON.json(body)
end

# Interactions function tools use the flat OpenAI-Responses shape observed on the wire:
# {type:"function", name, description?, parameters?}. No functionDeclarations wrapper.
function _interactions_tool(t)
    if t isa FunctionTool
        d = Dict{Symbol,Any}(:type => "function", :name => t.name)
        isnothing(t.description) || (d[:description] = t.description)
        isnothing(t.parameters)  || (d[:parameters] = t.parameters)
        return d
    end
    t isa AbstractDict && return t                    # pre-shaped passthrough
    throw(ArgumentError("Gemini Interactions supports only FunctionTool/Dict tools (got $(typeof(t)))"))
end

# ─── Gemini native hosted tools ──────────────────────────────────────────────
# Flat {type:<name>} declarations. NOTE: estimated_cost is token-based and does
# NOT model hosted-tool per-call fees (e.g. google_search per-1k-queries).

"""
    gemini_google_search() -> Dict

Hosted Google Search tool for the Gemini Interactions API. Pass in
`respond(...; tools=[gemini_google_search()], service=GEMINIServiceEndpoint)`.
"""
gemini_google_search()  = Dict{String,Any}("type" => "google_search")

"""
    gemini_code_execution() -> Dict

Hosted code-execution tool for the Gemini Interactions API. Pass in
`respond(...; tools=[gemini_code_execution()], service=GEMINIServiceEndpoint)`.
"""
gemini_code_execution() = Dict{String,Any}("type" => "code_execution")

"""
    gemini_url_context() -> Dict

Hosted URL-context tool for the Gemini Interactions API. Pass in
`respond(...; tools=[gemini_url_context()], service=GEMINIServiceEndpoint)`.
"""
gemini_url_context()    = Dict{String,Any}("type" => "url_context")

# Neutral input items → Interactions input. A `function_call_output` tool-result item
# (OpenAI-shaped neutral, from tool_result/tool_loop) → Gemini `function_result{call_id,
# name, result}`; a String input and any other item pass through unchanged.
_interactions_input(input::AbstractString) = input
_interactions_input(input::AbstractVector) = Any[_interactions_input_item(x) for x in input]

function _interactions_input_item(x)
    (x isa AbstractDict && get(x, "type", "") == "function_call_output") || return x
    haskey(x, "name") || throw(ArgumentError(
        "Gemini function_result requires a name; build the item with tool_result(call_id, name, output)"))
    Dict{String,Any}(
        "type" => "function_result",
        "call_id" => get(x, "call_id", ""),
        "name" => x["name"],
        "result" => _gemini_tool_response(get(x, "output", "")))
end

# Neutral tool_choice → generation_config.tool_choice.allowed_tools.{mode, tools}
# (confirmed live: mode auto/any/none; tools = function-name strings).
_interactions_tool_choice(tc::AbstractString) =
    tc == "auto"     ? Dict{Symbol,Any}(:allowed_tools => Dict{Symbol,Any}(:mode => "auto")) :
    tc == "none"     ? Dict{Symbol,Any}(:allowed_tools => Dict{Symbol,Any}(:mode => "none")) :
    tc == "required" ? Dict{Symbol,Any}(:allowed_tools => Dict{Symbol,Any}(:mode => "any")) :
    throw(ArgumentError("Unknown tool_choice string $(repr(tc)) for Gemini Interactions"))

function _interactions_tool_choice(tc::AbstractDict)
    _g(k) = get(tc, k, get(tc, String(k), nothing))   # tolerate Symbol- or String-keyed dicts
    _g(:type) == "function" || throw(ArgumentError(
        "Gemini Interactions tool_choice supports \"auto\"/\"none\"/\"required\" or a specific " *
        "function (tool_choice_function); hosted-tool selectors are not applicable. Got $(repr(tc))"))
    Dict{Symbol,Any}(:allowed_tools => Dict{Symbol,Any}(:mode => "any", :tools => [_g(:name)]))
end

# ─── Response decoding (Interactions steps[] → neutral ResponseObject) ────────
# Normalize into OpenAI-Responses-shaped output[] so existing accessors work.

function _interaction_output(steps)::Vector{Any}
    out = Any[]
    for s in (steps isa AbstractVector ? steps : ())
        s isa AbstractDict || continue
        t = get(s, "type", "")
        if t == "model_output"
            parts = Any[]
            for c in get(s, "content", ())
                c isa AbstractDict && get(c, "type", "") == "text" &&
                    push!(parts, Dict{String,Any}("type" => "output_text", "text" => get(c, "text", "")))
            end
            push!(out, Dict{String,Any}("type" => "message", "role" => "assistant", "content" => parts))
        elseif t == "function_call"
            args = get(s, "arguments", Dict{String,Any}())
            push!(out, Dict{String,Any}(
                "type" => "function_call",
                "call_id" => get(s, "id", ""),
                "name" => get(s, "name", ""),
                # Interactions returns arguments as a JSON OBJECT; the reused function_calls
                # accessor JSON.parses a STRING → stringify here so the accessor round-trips.
                "arguments" => (args isa AbstractString ? args : JSON.json(args))))
        elseif !isempty(t)
            # thought + hosted-tool + other steps (google_search_call/_result,
            # code_execution_*, url_context_*, …): surface them verbatim (native
            # type + fields, e.g. a thought step's signature) rather than
            # dropping. output_text still comes from model_output;
            # function_calls() ignores them (no "function_call" type).
            push!(out, Dict{String,Any}(s))
        end
    end
    out
end

# Build a single OpenAI-Responses-shaped assistant message from raw text.
_text_message(txt::AbstractString) = Dict{String,Any}("type" => "message", "role" => "assistant",
    "content" => Any[Dict{String,Any}("type" => "output_text", "text" => txt)])

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

# OpenAI-Responses-shaped dict (used by non-stream decode + streaming assembly).
_interaction_response_dict(data::AbstractDict) = Dict{String,Any}(
    "id" => get(data, "id", ""),
    "status" => get(data, "status", ""),
    "model" => get(data, "model", ""),
    "output" => _interaction_output(get(data, "steps", Any[])),
    "usage" => _interaction_usage(get(data, "usage", nothing)))

function _interaction_response_object(data::AbstractDict)::ResponseObject
    ResponseObject(
        id = get(data, "id", ""),
        status = get(data, "status", ""),
        model = get(data, "model", ""),
        output = _interaction_output(get(data, "steps", Any[])),
        usage = _interaction_usage(get(data, "usage", nothing)),
        error = get(data, "error", nothing),
        metadata = get(data, "metadata", nothing),
        raw = Dict{String,Any}(data))
end

decode_agentic(::Type{GEMINIServiceEndpoint}, resp::HTTP.Response)::ResponseObject =
    _interaction_response_object(JSON.parse(resp.body; dicttype=Dict{String,Any}))

# ─── Streaming decode (Interactions SSE → per-step assembly + final rebuild) ──
# Named events: interaction.created/status_update, step.start, step.delta
# (delta.type: text | arguments_delta | thought_summary | thought_signature),
# step.stop, interaction.completed (final object + usage but NO steps; its
# `status` may be "completed" or "requires_action" — there is no dedicated
# requires_action event), then event:done / [DONE]. Function-call arguments
# arrive as partial-JSON STRING deltas that must be accumulated per index.
# The terminal output[] is rebuilt from the assembled steps: function_call
# steps (arguments kept as the accumulated JSON string — the reused
# function_calls accessor JSON.parses strings), thought steps surfaced raw
# (signature assembled from deltas), and one text message from the
# accumulated text deltas.
function decode_agentic_stream(::Type{GEMINIServiceEndpoint}, chunk::String,
                               state::AgenticStreamState)
    for (ev, payload) in _sse_events!(state.carry, state.last_event, chunk)
        payload == "[DONE]" && return (; done=true, event=state.last_event[], data=nothing, terminal=:done)
        try
            data = JSON.parse(payload; dicttype=Dict{String,Any})
            if ev == "step.start"
                idx = get(data, "index", nothing)
                step = get(data, "step", nothing)
                if idx isa Integer && step isa Dict{String,Any}
                    haskey(state.steps, idx) || push!(state.order, idx)
                    state.steps[idx] = step
                    delete!(state.args_json, idx)   # a re-sent start must not inherit stale argument bytes
                end
            elseif ev == "step.delta"
                idx = get(data, "index", nothing)
                d = get(data, "delta", nothing)
                if idx isa Integer && d isa AbstractDict
                    dt = get(d, "type", "")
                    if dt == "arguments_delta"
                        a = get(d, "arguments", "")
                        a isa AbstractString && (state.args_json[idx] = get(state.args_json, idx, "") * a)
                    elseif dt == "thought_signature"
                        s = get(d, "signature", "")
                        if s isa AbstractString && haskey(state.steps, idx)
                            blk = state.steps[idx]
                            blk["signature"] = get(blk, "signature", "") * s
                        end
                    else
                        # Answer-text deltas carry a top-level `text` key and
                        # accumulate for output_text. thought_summary deltas
                        # nest their prose under `content` and are deliberately
                        # NOT accumulated: summaries are display material, not
                        # the answer and not replay material (the signature is).
                        t = get(d, "text", "")
                        t isa AbstractString && print(state.textbuff, t)
                    end
                end
            elseif ev == "interaction.completed"
                rdict = _interaction_response_dict(get(data, "interaction", data))
                if isempty(rdict["output"])
                    rdict["output"] = _assembled_interaction_output(state)
                end
                return (; done=true, event=ev,
                        data=Dict{String,Any}("response" => rdict), terminal=:completed)
            end
            # step.stop needs no handling beyond what assembly already holds:
            # arguments are complete once their deltas stop arriving, and the
            # terminal rebuild reads the accumulated state.
        catch e
            Threads.atomic_add!(_SSE_DROPPED_LINES, 1)
            @debug "Interactions SSE: dropped undecodable data payload" event = ev payload = String(payload) exception = e
        end
    end
    return (; done=false, event=state.last_event[], data=nothing, terminal=:none)
end

# Rebuild OpenAI-shaped output[] from the streamed step assembly (the terminal
# interaction.completed event carries no steps). Order: first-seen step order,
# then the accumulated text (if any) as a single assistant message.
function _assembled_interaction_output(state::AgenticStreamState)::Vector{Any}
    out = Any[]
    for idx in state.order
        step = state.steps[idx]
        t = get(step, "type", "")
        if t == "function_call"
            args = get(state.args_json, idx, "")
            isempty(args) && (a0 = get(step, "arguments", nothing); args = a0 isa AbstractString ? a0 : JSON.json(something(a0, Dict{String,Any}())))
            push!(out, Dict{String,Any}(
                "type" => "function_call",
                "call_id" => get(step, "id", ""),
                "name" => get(step, "name", ""),
                "arguments" => args))
        elseif !isempty(t)
            push!(out, Dict{String,Any}(step))   # thought + hosted-tool steps: raw, signature intact
        end
    end
    txt = String(take!(state.textbuff))
    isempty(txt) || push!(out, _text_message(txt))
    out
end
