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

_agentic_url(::Type{GEMINIServiceEndpoint})::String = GEMINI_NATIVE_BASE * INTERACTIONS_PATH

# ─── Request encoding (neutral Respond → Interactions body, snake_case) ───────

# The neutral Respond fields this wire maps. Everything else is derived from
# `fieldnames(Respond)`, so a field added to the neutral request is unsupported
# here until it is deliberately mapped — new surfaces fail loud by default.
const _INTERACTIONS_MAPPED_FIELDS = (:service, :model, :input, :instructions, :tools, :tool_choice,
    :temperature, :top_p, :max_output_tokens, :stream, :text, :store, :previous_response_id,
    :background, :reasoning)
const _INTERACTIONS_UNMAPPED_FIELDS = Tuple(setdiff(fieldnames(Respond), _INTERACTIONS_MAPPED_FIELDS))

# A set field the body never carries would vanish between the caller's request and
# the wire (`metadata`, `truncation`, …). Refuse instead.
function _interactions_reject_unmapped(r::Respond)
    set = Symbol[f for f in _INTERACTIONS_UNMAPPED_FIELDS if !isnothing(getfield(r, f))]
    isempty(set) || throw(ArgumentError(
        "Gemini Interactions does not support the Respond field(s) $(join(set, ", ")) — " *
        "unset them or send the request to an OpenAI Responses service"))
end

# The largest max_output_tokens an interaction takes without background execution, by exact
# model id: Gemini 4 Argon answers more with HTTP 400 "max_output_tokens greater than 65536 is
# not yet supported with background=false" (https://ai.google.dev/gemini-api/docs/long-decode;
# observed 2026-10-22), and in the background decodes up to its 1M-token context.
const _INTERACTIONS_SYNC_OUTPUT_LIMITS = Dict{String,Int}("gemini-4-argon" => 65_536)

function _interactions_validate_output_limit(r::Respond)::Nothing
    limit = get(_INTERACTIONS_SYNC_OUTPUT_LIMITS, r.model, nothing)
    isnothing(limit) || isnothing(r.max_output_tokens) || r.max_output_tokens <= limit || r.background === true ||
        throw(ArgumentError("$(r.model) takes max_output_tokens above $limit only in the background " *
            "(long decode, https://ai.google.dev/gemini-api/docs/long-decode): set background=true, or " *
            "lower max_output_tokens (got $(r.max_output_tokens))"))
    nothing
end

function encode_agentic(::Type{GEMINIServiceEndpoint}, r::Respond)::String
    _gemini_refuse_cache_options(r.prompt_cache_options)
    _interactions_reject_unmapped(r)
    _gemini_validate_sampling(r.model, r.temperature, r.top_p)
    _interactions_validate_output_limit(r)
    body = Dict{Symbol,Any}(:model => r.model, :input => _interactions_input(r.input))
    isnothing(r.instructions) || (body[:system_instruction] = r.instructions)
    isnothing(r.tools) || (body[:tools] = [_interactions_tool(t) for t in r.tools])
    (fmt = _interactions_response_format(r.text)) === nothing || (body[:response_format] = fmt)
    gen = Dict{Symbol,Any}()
    # The API reference's generation_config listing omits temperature/top_p, yet the live
    # API takes both (gemini-4-argon answered HTTP 200 on 2026-10-22, ignoring them) while
    # it answers 400 "Unknown parameter" to an unknown generation_config key; they reach
    # the wire only for the models that honour them (`_gemini_validate_sampling`).
    isnothing(r.temperature)       || (gen[:temperature] = r.temperature)
    isnothing(r.top_p)             || (gen[:top_p] = r.top_p)
    isnothing(r.max_output_tokens)  || (gen[:max_output_tokens] = r.max_output_tokens)
    isnothing(r.tool_choice)        || (gen[:tool_choice] = _interactions_tool_choice(r.tool_choice))
    if (reasoning = r.reasoning) !== nothing
        isnothing(reasoning.context) && isnothing(reasoning.mode) || throw(ArgumentError(
            "Gemini Interactions does not support reasoning context or mode"))
        effort = reasoning.effort
        isnothing(effort) || (gen[:thinking_level] = _gemini_thinking_level(r.model, effort))
        summary = _reasoning_summary(reasoning)
        isnothing(summary) || summary == "auto" || throw(ArgumentError(
            "Gemini Interactions supports only Reasoning(summary=\"auto\")"))
        isnothing(summary) || (gen[:thinking_summaries] = summary)
    end
    isempty(gen) || (body[:generation_config] = gen)
    # Neutral continuation handle (previous_response_id) → Gemini's server-state id.
    isnothing(r.previous_response_id) || (body[:previous_interaction_id] = r.previous_response_id)
    isnothing(r.store)      || (body[:store] = r.store)
    isnothing(r.background) || (body[:background] = r.background)
    isnothing(r.stream)     || (body[:stream] = r.stream)
    JSON.json(body)
end

# Neutral text.format → top-level response_format (TextResponseFormat: type "text",
# mime_type, optional schema). Plain text needs no format. Verbosity has no
# counterpart; the OpenAI schema name, description, and strict flag are not sent.
_interactions_response_format(::Nothing) = nothing
function _interactions_response_format(t::TextConfig)
    isnothing(t.verbosity) || throw(ArgumentError("Gemini Interactions does not support text verbosity"))
    f = t.format
    f.type == "text" && isnothing(f.schema) && return nothing
    f.type == "json_object" && isnothing(f.schema) &&
        return Dict{Symbol,Any}(:type => "text", :mime_type => "application/json")
    f.type == "json_schema" && !isnothing(f.schema) &&
        return Dict{Symbol,Any}(:type => "text", :mime_type => "application/json", :schema => f.schema)
    throw(ArgumentError("Gemini Interactions supports text formats text, json_object, or json_schema " *
        "with a schema (got $(repr(f.type))$(isnothing(f.schema) ? " without" : " with") a schema)"))
end

# Interactions function tools use the flat OpenAI-Responses shape observed on the wire:
# {type:"function", name, description?, parameters?}. No functionDeclarations wrapper.
# The API reference's Function tool lists only those fields, so a set OpenAI-only
# FunctionTool option is refused rather than dropped; `strict` is not sent.
const _INTERACTIONS_UNSUPPORTED_TOOL_FIELDS = (:async, :allowed_callers, :defer_loading, :output_schema)

function _interactions_tool(t)
    if t isa FunctionTool
        set = Symbol[f for f in _INTERACTIONS_UNSUPPORTED_TOOL_FIELDS if !isnothing(getfield(t, f))]
        isempty(set) || throw(ArgumentError(
            "Gemini Interactions function tools do not support $(join(set, ", "))"))
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

# A text content part of a model_output step.
_interaction_text_part(c) = c isa AbstractDict && get(c, "type", "") == "text"

# A model answering across several steps (the hops of a long decode, text resumed after a
# thought) adds one model_output step each, and Google's guides read the answer as the text of
# those steps joined with no separator (https://ai.google.dev/gemini-api/docs/long-decode). So
# a run of model_output steps, with only thoughts between them, is one message holding one
# output_text part: the text parts of every step of the run, joined as written. Any other step
# ends the run.
function _interaction_output(steps)::Vector{Any}
    out = Any[]
    answer = 0   # index in `out` of the message the next model_output step extends; 0 for none
    for s in (steps isa AbstractVector ? steps : ())
        s isa AbstractDict || continue
        t = get(s, "type", "")
        if t == "model_output"
            if answer == 0
                push!(out, Dict{String,Any}("type" => "message", "role" => "assistant", "content" => Any[]))
                answer = length(out)
            end
            _extend_answer!(out[answer], s)
        elseif t == "function_call"
            answer = 0
            # Absent or null arguments: a call without arguments.
            args = something(get(s, "arguments", nothing), Dict{String,Any}())
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
            # function_calls() ignores them (no "function_call" type). A thought
            # leaves the answer open; every other step ends it.
            t == "thought" || (answer = 0)
            push!(out, Dict{String,Any}(s))
        end
    end
    out
end

# Append a model_output step's text to the answer message. `content` is null on a step that
# produced no parts, which adds nothing.
function _extend_answer!(message::Dict{String,Any}, step::AbstractDict)::Nothing
    io = IOBuffer()
    for c in _as_iter(get(step, "content", ()))
        _interaction_text_part(c) || continue
        t = get(c, "text", "")
        t isa AbstractString && print(io, t)
    end
    text = takestring!(io)
    isempty(text) && return nothing
    content = message["content"]::Vector{Any}
    if isempty(content)
        push!(content, Dict{String,Any}("type" => "output_text", "text" => text))
    else
        part = only(content)
        part["text"] = string(part["text"], text)
    end
    nothing
end

# Gemini Interactions usage → OpenAI-Responses-shaped usage so token_usage/estimated_cost
# work unchanged. The tool-use prompt (total_tool_use_tokens, e.g. a page url_context read) is
# input the model read, billed at the input rate ("URL context: charged as input tokens",
# https://ai.google.dev/gemini-api/docs/pricing), and it is not part of total_input_tokens: a
# url_context interaction on gemini-3.8-flash reported input 30, tool use 61, output 64 and
# total 155 on 2026-10-22. Thoughts bill at the output rate, so they fold into billable
# output_tokens, and reasoning_tokens breaks out the thought subset (OpenAI semantics).
# The raw usage is preserved on ResponseObject.raw. Per-call hosted-tool fees are NOT modeled.
_interaction_usage(::Nothing) = nothing
function _interaction_usage(u::AbstractDict)
    _n(k) = (v = get(u, k, 0); v isa Integer ? Int(v) : 0)
    Dict{String,Any}(
        "input_tokens"  => _n("total_input_tokens") + _n("total_tool_use_tokens"),
        "output_tokens" => _n("total_output_tokens") + _n("total_thought_tokens"),
        "total_tokens"  => _n("total_tokens"),
        "input_tokens_details"  => Dict{String,Any}("cached_tokens" => _n("total_cached_tokens")),
        "output_tokens_details" => Dict{String,Any}("reasoning_tokens" => _n("total_thought_tokens")))
end

# OpenAI-Responses-shaped dict for the streamed terminal. The normalized views are
# overlaid on a COPY of the raw interaction, because the driver reads the response
# object AND its raw capture from this one dict: keeping the wire's other fields
# (error, metadata, …) gives a streamed result the same surface the non-stream
# `_interaction_response_object` preserves.
function _interaction_response_dict(data::AbstractDict)::Dict{String,Any}
    d = Dict{String,Any}(data)
    d["id"] = get(data, "id", "")
    d["status"] = get(data, "status", "")
    d["model"] = get(data, "model", "")
    d["output"] = _interaction_output(get(data, "steps", Any[]))
    d["usage"] = _interaction_usage(get(data, "usage", nothing))
    err = _interaction_error(data)
    isnothing(err) || (d["error"] = err)
    d
end

# An interaction records its faults in `errors`, an array of {code, message}
# (https://ai.google.dev/api/interactions-api). The error surface takes one object, as an
# OpenAI response's does, so it holds the first; `raw` keeps the array. An `error`, where a
# reply carries one, is kept.
function _interaction_error(data::AbstractDict)
    err = get(data, "error", nothing)
    if isnothing(err)
        faults = get(data, "errors", nothing)
        err = faults isa AbstractVector && !isempty(faults) ? first(faults) : nothing
    end
    err isa AbstractDict ? Dict{String,Any}(err) : err
end

# `id` and `status` are required on every Interaction (API reference): a 200 body
# without them, such as `{}`, is not one, and decoding it as one would report a
# success carrying an empty id. The complaint about the first one missing, else nothing;
# the streamed and the non-streamed decode share it.
function _interaction_defect(data::AbstractDict)::Union{Nothing,String}
    for key in ("id", "status")
        v = get(data, key, nothing)
        v isa String && !isempty(v) && continue
        return "Gemini Interactions response carries no \"$key\" (got $(repr(v))); keys: [" *
               join(sort!(collect(keys(data))), ", ") * "]"
    end
    nothing
end

function _interaction_response_object(data::AbstractDict)::ResponseObject
    defect = _interaction_defect(data)
    isnothing(defect) || error(defect)
    ResponseObject(
        id = data["id"],
        status = data["status"],
        model = get(data, "model", ""),
        output = _interaction_output(get(data, "steps", Any[])),
        usage = _interaction_usage(get(data, "usage", nothing)),
        error = _interaction_error(data),
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
# requires_action event), then event:done / [DONE].
# The assembly keeps every step in the shape a non-streamed interaction carries, and
# the terminal output[] is `_interaction_output` applied to the assembled steps in
# first-seen order, so the streamed and the non-streamed decode of one interaction
# yield the same output items. A step's streamed string grows in one buffer
# (`state.text_by_step`) and is written into the step once, when the interaction
# completes; by step type it is the function-call argument JSON (partial-JSON string
# deltas), the answer text (extending the step's last text part), or the thought
# signature. Each thought_summary delta adds one item to the thought's `summary`.
function decode_agentic_stream(::Type{GEMINIServiceEndpoint}, chunk::String,
                               state::AgenticStreamState)
    for (ev, payload) in _sse_events!(state.carry, state.last_event, chunk)
        payload == "[DONE]" && return (; done=true, event=state.last_event[], data=nothing, terminal=:done)
        try
            data = JSON.parse(payload; dicttype=Dict{String,Any})
            if ev == "step.start"
                _interaction_step_start!(state, data)
            elseif ev == "step.delta"
                _interaction_step_delta!(state, data)
            elseif ev == "interaction.completed"
                interaction = get(data, "interaction", data)
                # No id or status: not an interaction, and the non-streamed decode's
                # error — the driver makes an error terminal a ResponseCallError.
                defect = _interaction_defect(interaction)
                isnothing(defect) || return (; done=true, event=ev,
                    data=Dict{String,Any}("message" => defect), terminal=:error)
                rdict = _interaction_response_dict(interaction)
                if isempty(rdict["output"])
                    rdict["output"] = _assembled_interaction_output(state)
                end
                # A failed interaction takes the driver's typed-failure limb (as
                # OpenAI's response.failed does); every other terminal status —
                # completed, requires_action — is a normal result.
                term = rdict["status"] == "failed" ? :failed : :completed
                return (; done=true, event=ev,
                        data=Dict{String,Any}("response" => rdict), terminal=term)
            elseif ev == "error"
                return (; done=true, event=ev, data, terminal=:error)
            end
            # step.stop needs no handling: a step is complete once its deltas stop
            # arriving, and the terminal rebuild reads the assembly.
        catch e
            e isa InterruptException && rethrow()
            Threads.atomic_add!(_SSE_DROPPED_LINES, 1)
            state.sse_dropped += 1
            @debug "Interactions SSE: dropped undecodable data payload" event = ev payload = String(payload) exception = e
        end
    end
    return (; done=false, event=state.last_event[], data=nothing, terminal=:none)
end

function _interaction_step_start!(state::AgenticStreamState, data::Dict{String,Any})
    idx = get(data, "index", nothing)
    step = get(data, "step", nothing)
    (idx isa Integer && step isa Dict{String,Any}) || return nothing
    haskey(state.steps, idx) || push!(state.order, idx)
    state.steps[idx] = step
    delete!(state.text_by_step, idx)   # a re-sent start must not inherit stale streamed bytes
    get(step, "type", "") == "model_output" || return nothing
    # A step that has produced no parts yet carries no `content` array.
    content = get(step, "content", nothing)
    content isa Vector{Any} || (step["content"] = content = Any[])
    for c in content
        t = _interaction_text_part(c) ? get(c, "text", "") : nothing
        t isa AbstractString || continue
        print(state.textbuff, t)
        print(state.pending_delta, t)
    end
    nothing
end

function _interaction_step_delta!(state::AgenticStreamState, data::Dict{String,Any})
    idx = get(data, "index", nothing)
    d = get(data, "delta", nothing)
    (idx isa Integer && d isa AbstractDict) || return nothing
    step = get(state.steps, idx, nothing)
    kind = isnothing(step) ? "" : string(get(step, "type", ""))::String
    dt = get(d, "type", "")
    if dt == "arguments_delta"
        a = get(d, "arguments", "")
        a isa AbstractString && kind == "function_call" && print(get!(IOBuffer, state.text_by_step, idx), a)
    elseif dt == "thought_signature"
        # Kept on the steps surfaced verbatim (thoughts, hosted tools); the
        # function_call and message rebuilds carry no signature.
        s = get(d, "signature", "")
        s isa AbstractString && !isnothing(step) && kind ∉ ("function_call", "model_output") &&
            print(get!(IOBuffer, state.text_by_step, idx), s)
    elseif dt == "thought_summary"
        # Each delta carries one new summary item of the thought.
        c = get(d, "content", nothing)
        (c isa AbstractDict && !isnothing(step)) || return nothing
        summary = get(step, "summary", nothing)
        summary isa Vector{Any} || (step["summary"] = summary = Any[])
        push!(summary, c)
    else
        # Answer text (a top-level `text`). `textbuff` is the full accumulation
        # diagnostics can read; `pending_delta` is what the driver forwards.
        t = get(d, "text", "")
        (t isa AbstractString && !isempty(t)) || return nothing
        if isnothing(step)
            push!(state.order, idx)
            state.steps[idx] = Dict{String,Any}("type" => "model_output", "content" => Any[])
            kind = "model_output"
        end
        kind == "model_output" && print(get!(IOBuffer, state.text_by_step, idx), t)
        print(state.textbuff, t)
        print(state.pending_delta, t)
    end
    nothing
end

# Write step `idx`'s streamed string into the step, once: the argument JSON replaces
# the start event's placeholder, the answer text extends the step's last text part
# (or opens one), and a signature extends the one the start event carried.
function _interaction_finalize_step!(state::AgenticStreamState, idx::Int)::Dict{String,Any}
    step = state.steps[idx]
    buffer = pop!(state.text_by_step, idx, nothing)
    streamed = isnothing(buffer) ? "" : takestring!(buffer)
    isempty(streamed) && return step
    kind = get(step, "type", "")
    if kind == "function_call"
        step["arguments"] = streamed
    elseif kind == "model_output"
        content = step["content"]::Vector{Any}
        part = isempty(content) ? nothing : last(content)
        if _interaction_text_part(part) && get(part, "text", nothing) isa AbstractString
            part["text"] = string(part["text"], streamed)
        else
            push!(content, Dict{String,Any}("type" => "text", "text" => streamed))
        end
    else
        signature = get(step, "signature", "")
        step["signature"] = string(signature isa AbstractString ? signature : "", streamed)
    end
    step
end

# The terminal interaction.completed event carries no steps: output[] is the
# non-stream decode of the assembled steps, in first-seen step order.
_assembled_interaction_output(state::AgenticStreamState)::Vector{Any} =
    _interaction_output(Any[_interaction_finalize_step!(state, idx) for idx in state.order])
