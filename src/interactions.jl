# ============================================================================
# Google Gemini Interactions API — Layer-B agentic verb
# https://ai.google.dev/gemini-api/docs/interactions-overview
#
# Native, server-STATEFUL agentic surface. Rides the SAME agentic seam as OpenAI
# Responses (src/responses.jl): overrides get_url / encode_agentic / decode_agentic
# / decode_agentic_stream for GEMINIServiceEndpoint, dispatched by request type
# (Respond) — exactly as OPENAIServiceEndpoint hosts both Chat and Respond. The
# decoder normalizes Interactions `steps[]` into the OpenAI-Responses-shaped
# `output[]`, so ResponseObject accessors (output_text/function_calls/…) are reused.
#
# Wire shape captured LIVE against the interactions endpoint on 2026-07-07
# (see docs/superpowers/plans/2026-07-07-gemini-interactions-plan2.md ledger).
# ============================================================================

# ─── Routing (streaming is a BODY flag, so the URL is stream-independent) ─────

get_url(::Type{GEMINIServiceEndpoint}, ::Respond) = GEMINI_NATIVE_BASE * INTERACTIONS_PATH

# ─── Request encoding (neutral Respond → Interactions body, snake_case) ───────

function encode_agentic(::Type{GEMINIServiceEndpoint}, r::Respond)::String
    isnothing(r.tool_choice) || throw(ArgumentError(
        "tool_choice is not yet supported for Gemini Interactions (Plan 3); omit it or steer via the prompt"))
    body = Dict{Symbol,Any}(:model => r.model, :input => _interactions_input(r.input))
    isnothing(r.instructions) || (body[:system_instruction] = r.instructions)
    isnothing(r.tools) || (body[:tools] = [_interactions_tool(t) for t in r.tools])
    gen = Dict{Symbol,Any}()
    isnothing(r.temperature)       || (gen[:temperature] = r.temperature)
    isnothing(r.top_p)             || (gen[:top_p] = r.top_p)
    isnothing(r.max_output_tokens)  || (gen[:max_output_tokens] = r.max_output_tokens)
    isempty(gen) || (body[:generation_config] = gen)
    # Neutral continuation handle (previous_response_id) → Gemini's server-state id.
    isnothing(r.previous_response_id) || (body[:previous_interaction_id] = r.previous_response_id)
    isnothing(r.store)      || (body[:store] = r.store)
    isnothing(r.background) || (body[:background] = r.background)
    isnothing(r.stream)     || (body[:stream] = r.stream)
    JSON.json(body)
end

# Interactions function tools use the flat OpenAI-Responses shape (captured):
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

# Neutral input items → Interactions input. A `function_call_output` tool-result item
# (OpenAI-shaped neutral, from tool_result/tool_loop) → Gemini `function_result{call_id,
# name, result}`; a String input and any other item pass through unchanged.
_interactions_input(input::AbstractString) = input
_interactions_input(input::AbstractVector) = Any[_interactions_input_item(x) for x in input]
_interactions_input(input) = input

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
        elseif t == "thought"
            push!(out, Dict{String,Any}("type" => "reasoning", "summary" => Any[]))
        end
        # hosted-tool steps (google_search_call/…) are passed over in Plan 2.
    end
    out
end

# Build a single OpenAI-Responses-shaped assistant message from raw text.
_text_message(txt::AbstractString) = Dict{String,Any}("type" => "message", "role" => "assistant",
    "content" => Any[Dict{String,Any}("type" => "output_text", "text" => txt)])

# OpenAI-Responses-shaped dict (used by non-stream decode + streaming assembly).
_interaction_response_dict(data::AbstractDict) = Dict{String,Any}(
    "id" => get(data, "id", ""),
    "status" => get(data, "status", ""),
    "model" => get(data, "model", ""),
    "output" => _interaction_output(get(data, "steps", Any[])),
    "usage" => get(data, "usage", nothing))

function _interaction_response_object(data::AbstractDict)::ResponseObject
    ResponseObject(
        id = get(data, "id", ""),
        status = get(data, "status", ""),
        model = get(data, "model", ""),
        output = _interaction_output(get(data, "steps", Any[])),
        usage = get(data, "usage", nothing),
        error = get(data, "error", nothing),
        metadata = get(data, "metadata", nothing),
        raw = Dict{String,Any}(data))
end

decode_agentic(::Type{GEMINIServiceEndpoint}, resp::HTTP.Response)::ResponseObject =
    _interaction_response_object(JSON.parse(resp.body; dicttype=Dict{String,Any}))

# ─── Streaming decode (Interactions SSE → text accumulation + final assembly) ──
# Named events: interaction.created/status_update, step.start/delta/stop,
# interaction.completed (final object+usage but NO steps), then event:done/[DONE].
# Mirrors _parse_response_stream_chunk's carry-over + (; done,event,data,terminal)
# so _respond_stream assembles unchanged. On interaction.completed, hand back
# {"response": <OpenAI-shaped dict>}; since the completed event omits steps, the
# message output is rebuilt from the deltas accumulated in `textbuff`.
function decode_agentic_stream(::Type{GEMINIServiceEndpoint}, chunk::String, textbuff::IOBuffer,
                               failbuff::IOBuffer, last_event::Ref{String})
    chunk = String(take!(failbuff)) * chunk
    last_nl = findlast('\n', chunk)
    if isnothing(last_nl)
        print(failbuff, chunk)
        return (; done=false, event=last_event[], data=nothing, terminal=:none)
    end
    if last_nl < lastindex(chunk)
        print(failbuff, chunk[nextind(chunk, last_nl):end])
        chunk = chunk[1:last_nl]
    end
    for line in filter(!isempty, strip.(split(chunk, "\n")))
        if startswith(line, "event: ")
            last_event[] = strip(line[8:end])
        elseif startswith(line, "data: ")
            payload_str = strip(line[7:end])
            payload_str == "[DONE]" && return (; done=true, event=last_event[], data=nothing, terminal=:done)
            try
                payload = JSON.parse(payload_str; dicttype=Dict{String,Any})
                ev = last_event[]
                if ev == "step.delta"
                    d = get(payload, "delta", nothing)
                    d isa AbstractDict && print(textbuff, get(d, "text", ""))
                elseif ev == "interaction.completed"
                    rdict = _interaction_response_dict(get(payload, "interaction", payload))
                    if isempty(rdict["output"])
                        txt = String(take!(textbuff))
                        isempty(txt) || (rdict["output"] = Any[_text_message(txt)])
                    end
                    return (; done=true, event=ev,
                            data=Dict{String,Any}("response" => rdict), terminal=:completed)
                end
            catch
                print(failbuff, line)
                continue
            end
        end
    end
    return (; done=false, event=last_event[], data=nothing, terminal=:none)
end
